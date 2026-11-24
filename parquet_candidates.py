"""Find eligible package pairs in the official Open Food Facts Parquet mirror."""
import json

import duckdb

from ingest import CACHE

URL = "https://huggingface.co/datasets/openfoodfacts/product-database/resolve/main/food.parquet"


def main():
    CACHE.mkdir(exist_ok=True)
    db = duckdb.connect()
    db.execute("SET threads=4")
    output = []
    for category, tag in [("cereals","en:breakfast-cereals"),("spreads","en:chocolate-spreads")]:
        result = db.execute("""
            SELECT code, product_name, brands, quantity, images
            FROM read_parquet(?)
            WHERE list_contains(categories_tags, ?) AND len(images) >= 4 AND brands IS NOT NULL
              AND len(product_name)>0 AND regexp_full_match(code, '[0-9]{13}')
            LIMIT 210
        """,[URL,tag]).fetchall()
        for code, names, brands, quantity, images in result:
            title = next((x["text"] for x in names if x.get("text") and x.get("lang")=="main"), None)
            title = title or next((x.get("text") for x in names if x.get("text")),None)
            if not title:
                continue
            subpath = "/".join((code[:3],code[3:6],code[6:9],code[9:]))
            row = dict(code=code,product_name=title,brands=brands,quantity=quantity,
                       image_front_url=f"https://images.openfoodfacts.org/images/products/{subpath}/front.jpg")
            metadata = CACHE / (code + ".json")
            if not metadata.exists():
                metadata.write_text(json.dumps({"product":{"images":{x["key"]:x for x in images}}}),encoding="utf8")
            output.append((category,row))
        print(category,len(result),flush=True)
    (CACHE/"parquet-candidates.json").write_text(json.dumps(output),encoding="utf8")


if __name__ == "__main__":
    main()
