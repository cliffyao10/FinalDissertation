"""Print traceable positive/negative examples from a prepared Polyvore split."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from recommendation_training.prepare_polyvore import image_identifier, load_split


def metadata_by_item_id(path):
    records = {}
    for outfit in json.loads(path.read_text(encoding="utf-8")):
        for item in outfit.get("items", []):
            item_id = image_identifier(item)
            if item_id:
                records[item_id] = {
                    "name": item.get("name", ""),
                    "category_id": int(item["categoryid"]),
                }
    return records


def sampled_pairs(metadata_path, images_root, seed, count):
    outfits, items = load_split(metadata_path, images_root, minimum_slots=3)
    metadata = metadata_by_item_id(metadata_path)
    pools = defaultdict(list)
    for item_id, item in items.items():
        pools[item["slot"]].append(item_id)

    rng = random.Random(seed)
    examples = []
    for outfit in outfits:
        replaceable = [
            slot
            for slot, item_id in outfit.items()
            if any(
                candidate != item_id
                and items[candidate]["set_id"] != items[item_id]["set_id"]
                for candidate in pools[slot]
            )
        ]
        if not replaceable:
            continue
        replaced_slot = rng.choice(replaceable)
        original_id = outfit[replaced_slot]
        replacements = [
            candidate
            for candidate in pools[replaced_slot]
            if candidate != original_id
            and items[candidate]["set_id"] != items[original_id]["set_id"]
        ]
        replacement_id = rng.choice(replacements)

        def describe(item_id):
            return {
                "item_id": item_id,
                "slot": items[item_id]["slot"],
                "name": metadata.get(item_id, {}).get("name", ""),
                "category_id": metadata.get(item_id, {}).get("category_id"),
                "image_path": items[item_id]["path"],
            }

        examples.append(
            {
                "source_set_id": items[original_id]["set_id"],
                "positive": [describe(outfit[slot]) for slot in outfit],
                "negative": {
                    "replaced_slot": replaced_slot,
                    "removed": describe(original_id),
                    "inserted": describe(replacement_id),
                },
            }
        )
        if len(examples) >= count:
            break
    return {
        "metadata_path": str(metadata_path),
        "seed": seed,
        "retained_outfits": len(outfits),
        "unique_retained_items": len(items),
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/polyvore/train_no_dup.json")
    parser.add_argument("--images", default="data/polyvore_images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    report = sampled_pairs(
        Path(args.metadata), Path(args.images), args.seed, max(1, args.count)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
