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


def candidates():
    for category, tag in (("cereals", "breakfast-cereals"), ("spreads", "chocolate-spreads")):
        saved = ROOT / f"{category}.json"
        if not saved.exists():
            saved.write_bytes(get(f"{BASE}/search", params={
                "categories_tags": "en:" + tag, "page_size": 50,
                "fields": "code,product_name,brands,quantity,image_front_url",
            }))
        rows = json.loads(saved.read_text(encoding="utf8"))["products"]
        for row in rows:
            if row.get("image_front_url") and row.get("product_name") and row.get("brands"):
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
            urls = [f"{images_dir}/{idx}.400.jpg" for idx in (primary,other)]
            try:
                thumbs = []
                for url in urls:
                    data = get(url)
                    with Image.open(io.BytesIO(data)) as im:
                        thumbs.append(ImageOps.exif_transpose(im).convert("RGB"))
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
    items = list(candidates())
    rows = []
    # The product endpoint has a 100 requests/minute documented limit.
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        for chunk in (items[:48], items[48:96]):
            if rows:
                time.sleep(4)
            rows.extend(row for row in pool.map(details,chunk) if row)
    if len(rows) < 10:
        raise RuntimeError(f"Only {len(rows)} usable product pairs")
    (SITE / "manifest.json").write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding="utf8")
    print(f"{len(rows)} products, {sum(r['alternate_front'] for r in rows)} alternate front views")


if __name__ == "__main__":
    main()
