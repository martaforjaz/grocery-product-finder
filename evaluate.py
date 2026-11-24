"""Evaluate vision, OCR and late fusion using distinct Open Food Facts uploads."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).parent
SITE = ROOT / "site"
MIX = .55


def words(text):
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return {t for t in re.findall("[a-z0-9]+", text) if len(t) > 1 and not t.isdigit() or t.isdigit() and 2 <= len(t) <= 4}


def pixels(path):
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((224,224), Image.Resampling.BILINEAR)
    tile = Image.new("RGB",(224,224),"#eaf0f7")
    tile.paste(image, ((224-image.width)//2,(224-image.height)//2))
    values = np.array(tile,dtype=np.float32)/255
    return ((values-np.array([.485,.456,.406],dtype=np.float32)) /
            np.array([.229,.224,.225],dtype=np.float32)).transpose(2,0,1)[None]


def main():
    rows = json.loads((SITE/"manifest.json").read_text(encoding="utf8"))
    # Deterministic product partitions. Held-out products never enter the index.
    order = sorted(range(len(rows)),key=lambda i:hashlib.sha256(rows[i]["code"].encode()).digest())
    gallery, unknown = order[:-max(4,len(rows)//5)], order[-max(4,len(rows)//5):]
    if len(gallery) < 6:
        raise RuntimeError("Insufficient catalog size")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    model = ort.InferenceSession(str(SITE/"mobilenet.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    ocr = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
    encode = lambda path: model.run(None,{"pixels":pixels(path)})[0][0]
    def image_path(i,split):
        return SITE/"images"/f'{rows[i]["code"]}-{split}.jpg'
    vectors = np.stack([encode(image_path(i,"gallery")) for i in gallery])
    vectors /= np.linalg.norm(vectors,axis=1,keepdims=True).clip(1e-12)
    vocab = [words(" ".join(str(rows[i].get(k) or "") for k in ("name","brand","quantity"))) for i in gallery]
    weights = {term: 1+np.log((1+len(vocab))/(1+sum(term in doc for doc in vocab)))
               for term in set().union(*vocab)}
    cache = ROOT/"cache"
    cache.mkdir(exist_ok=True)
    predictions = []
    for i in order:
        path = image_path(i,"query")
        key = hashlib.sha256(path.read_bytes()).hexdigest()
        saved = cache/f"ocr-{key}.txt"
        if not saved.exists():
            lines, _ = ocr(str(path))
            saved.write_text(" ".join(line[1] for line in lines or ()), encoding="utf8")
        text = saved.read_text(encoding="utf8")
        query = encode(path)
        vision = (vectors @ (query/max(np.linalg.norm(query),1e-12))).clip(0,1)
        terms = words(text)
        lexical = np.array([sum(weights[t] for t in terms & doc)/max(1,sum(weights[t] for t in doc)) for doc in vocab])
        scores = {"image":vision,"ocr":lexical,"fusion": MIX*vision+(1-MIX)*lexical}
        rank = {mode: sorted(range(len(gallery)),key=lambda j:-values[j])[:3] for mode, values in scores.items()}
        predictions.append(dict(i=i,text=text,rank={mode:[gallery[j] for j in ids] for mode,ids in rank.items()},
                                confidence=float(scores["fusion"][rank["fusion"][0]])))
    calibration = [p for p in predictions if p["i"] in set(gallery[::4]+unknown[::2])]
    test = [p for p in predictions if p not in calibration]
    truth = set(gallery)
    def safe(p):
        return p["i"] in truth and p["rank"]["fusion"][0]==p["i"]
    threshold = max((p["confidence"] for p in calibration if not safe(p)),default=0)
    known = [p for p in test if p["i"] in truth]
    absent = [p for p in test if p["i"] not in truth]
    metrics = {mode:{"top1":sum(p["rank"][mode][0]==p["i"] for p in known),
                     "top3":sum(p["i"] in p["rank"][mode] for p in known)}
               for mode in ("image","ocr","fusion")}
    report = dict(catalog=len(gallery), known=len(known), unknown=len(absent), calibration=len(calibration),
                  metrics=metrics, threshold=threshold,
                  dataset_products=len(rows), alternate_front_pairs=sum(r["alternate_front"] for r in rows),
                  accepted=sum(p["confidence"]>threshold for p in test),
                  incorrect_accepts=sum(p["confidence"]>threshold and not safe(p) for p in test),
                  unknown_accepts=sum(p["confidence"]>threshold for p in absent))
    fix = next((p for p in known if p["rank"]["image"][0]!=p["i"] and safe(p)),None)
    sample = fix or next((p for p in known if not safe(p)),known[0])
    examples = [sample, next((p for p in absent if p["confidence"] <= threshold), absent[0])]
    output = dict(report=report, examples=examples, products=rows,
                  index=[dict(i=i,vector=vector.tolist(),tokens=sorted(doc)) for i,vector,doc in zip(gallery,vectors,vocab)],
                  weights=weights, mix=MIX)
    (SITE/"results.json").write_text(json.dumps(output,ensure_ascii=False),encoding="utf8")
    (ROOT/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    (ROOT/"predictions.json").write_text(json.dumps(test,indent=2,ensure_ascii=False),encoding="utf8")
    fig, ax = plt.subplots(figsize=(6.2,3.4),layout="constrained")
    positions = np.arange(3)
    for shift,metric,color in [(-.17,"top1","#244e79"),(.17,"top3","#489487")]:
        bars=ax.bar(positions+shift,[metrics[mode][metric] for mode in metrics],width=.32,label=metric.upper(),color=color)
        ax.bar_label(bars)
    ax.set(xticks=positions,xticklabels=["Image","OCR","Image + OCR"],ylabel=f"Correct queries (n={len(known)})",
           ylim=(0,len(known)*1.2))
    ax.spines[["right","top"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(SITE/"benchmark.png",dpi=180)
    print(json.dumps(report))
    print("Examples",[(rows[p["i"]]["name"],[rows[p["rank"][m][0]]["name"] for m in metrics]) for p in examples])


if __name__ == "__main__":
    main()
