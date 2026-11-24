"""Render the first recorded result comparison for the README."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).parent
SITE = ROOT / "site"


def main():
    data = json.loads((SITE / "results.json").read_text(encoding="utf8"))
    case = data["examples"][0]
    truth = case["i"]
    modes = ("image", "ocr", "fusion")
    indices = [truth] + [case["rank"][mode][0] for mode in modes]
    titles = ["Query", "Image", "OCR", "Image + OCR"]
    fig, axes = plt.subplots(1,4,figsize=(8.8,2.85),layout="constrained")
    for n,(ax,i,title) in enumerate(zip(axes,indices,titles)):
        item = data["products"][i]
        side = "query" if n==0 else "gallery"
        with Image.open(SITE / "images" / f'{item["code"]}-{side}.jpg') as image:
            ax.imshow(image)
        ax.set(title=title,xticks=[],yticks=[])
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color("#378373" if n and i==truth else "#b37852" if n else "#33577c")
    (ROOT/"figures").mkdir(exist_ok=True)
    fig.savefig(ROOT/"figures"/"decision.png",dpi=180)


if __name__=="__main__":
    main()
