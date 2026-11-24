import concurrent.futures as cf
import io
import json
from pathlib import Path
import re
import time

from PIL import Image, ImageOps
import requests

ROOT = Path(__file__).parent
SITE = ROOT / "site"
CACHE = ROOT / "cache"
HEADERS = {"User-Agent": "MartaForjaz-GroceryFinder/1.0 (public portfolio research)"}
BASE = "https://world.openfoodfacts.org/api/v2"


def get(url, **kwargs):
    with requests.get(url, headers=HEADERS, timeout=15, **kwargs) as response:
        response.raise_for_status()
        return response.content


def candidates(pages):
    seen = set()
    for page in range(1, pages + 1):
        for category, tag in (("cereals", "breakfast-cereals"), ("spreads", "chocolate-spreads")):
            saved = ROOT / f"{category}.json" if page == 1 else CACHE / f"{category}-{page}.json"
            try:
                if not saved.exists():
                    parameters = {"categories_tags": "en:" + tag, "page_size": 50,
                                  "fields": "code,product_name,brands,quantity,image_front_url"}
                    if page > 1:
                        parameters["page"] = page
                    saved.write_bytes(get(f"{BASE}/search", params=parameters))
                    time.sleep(7)
                rows = json.loads(saved.read_text(encoding="utf8"))["products"]
            except (ValueError, KeyError, requests.RequestException) as error:
                print(category, page, error, flush=True)
                time.sleep(7)
                continue
            for row in rows:
                if row.get("image_front_url") and row.get("product_name") and row.get("brands") and row["code"] not in seen:
                    seen.add(row["code"])
                    yield category, row


def details(item):
    category, row = item
    code = row["code"]
    local = CACHE / (code + ".json")
    try:
        if not local.exists():
            raw = get(f"{BASE}/product/{code}", params={"fields":"images,code"})
            local.write_bytes(raw)
        images = json.loads(local.read_bytes())["product"].get("images", {})
        fronts = list(dict.fromkeys(str(v["imgid"]) for k,v in images.items()
                                   if k.startswith("front") and v.get("imgid")))
        if not fronts:
            return None
        primary = fronts[0]
        # Other uploads are legitimate test queries, including non-front views.
        alternates = fronts[1:] + sorted((k for k in images if k.isdigit() and k != primary), key=int)
        for other in dict.fromkeys(alternates):
            if other not in images:
                continue
            if images.get(primary, {}).get("uploaded_t") == images[other].get("uploaded_t"):
                continue
            images_dir = row["image_front_url"].rsplit("/",1)[0]
            if row.get("archive"):
                prefix = images_dir.split("/images/products/", 1)[1]
                urls = [f"https://openfoodfacts-images.s3.eu-west-3.amazonaws.com/data/{prefix}/{idx}.jpg"
                        for idx in (primary, other)]
            else:
                urls = [f"{images_dir}/{idx}.400.jpg" for idx in (primary,other)]
            try:
                thumbs = []
                for url in urls:
                    data = get(url)
                    with Image.open(io.BytesIO(data)) as im:
                        thumb = ImageOps.exif_transpose(im).convert("RGB")
                        thumb.thumbnail((400, 400), Image.Resampling.LANCZOS)
                        thumbs.append(thumb)
                if thumbs[0].size == thumbs[1].size and thumbs[0].tobytes() == thumbs[1].tobytes():
                    continue
                for split, img in zip(("gallery","query"),thumbs):
                    img.save(SITE / "images" / f"{code}-{split}.jpg",quality=88)
                return dict(code=code, category=category, name=row["product_name"], brand=row["brands"],
                            quantity=row.get("quantity",""), urls=urls,
                            contributors=[images.get(idx,{}).get("uploader","openfoodfacts-contributors") for idx in (primary,other)],
                            alternate_front=other in fronts)
            except (OSError, requests.RequestException):
                continue
    except (ValueError, OSError, KeyError, requests.RequestException):
        return None


def main():
    (SITE / "images").mkdir(parents=True,exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument("--parquet", action="store_true")
    args = parser.parse_args()
    manifest = SITE / "manifest.json"
    rows = json.loads(manifest.read_text(encoding="utf8")) if manifest.exists() else []
    known = {row["code"] for row in rows}
    source = (json.loads((CACHE / "parquet-candidates.json").read_text(encoding="utf8"))
              if args.parquet else candidates(args.pages))
    items = []
    for category, row in source:
        if row["code"] not in known:
            known.add(row["code"])
            items.append((category, dict(row, archive=args.parquet)))
    with cf.ThreadPoolExecutor(max_workers=14 if args.parquet else 5) as pool:
        start = 0
        while start < len(items) and len(rows) < args.target:
            started = time.monotonic()
            batch_size = min(50, args.target - len(rows))
            pending = items[start:start + batch_size]
            rows.extend(row for row in pool.map(details, pending) if row)
            start += len(pending)
            print(f"Accepted {len(rows)} pairs after {start} candidates", flush=True)
            if not args.parquet and start < len(items) and len(rows) < args.target:
                time.sleep(max(0, 32 - (time.monotonic() - started)))
    rows = rows[:args.target]
    if len(rows) < 10:
        raise RuntimeError(f"Only {len(rows)} usable product pairs")
    (SITE / "manifest.json").write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding="utf8")
    print(f"{len(rows)} products, {sum(r['alternate_front'] for r in rows)} alternate front views")


if __name__ == "__main__":
    main()
