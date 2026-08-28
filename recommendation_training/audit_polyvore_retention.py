"""Explain exactly why official Maryland Polyvore outfits are retained or dropped."""

import argparse
import json
from collections import Counter
from pathlib import Path

from recommendation_training.prepare_polyvore import (
    CATEGORY_TO_SLOT,
    image_identifier,
    image_path,
)


def category_names(path):
    names = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        identifier, _, name = line.partition(" ")
        if identifier.isdigit():
            names[int(identifier)] = name
    return names


def audit_split(metadata_path, images_root, names):
    raw_outfits = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    unsupported = Counter()
    supported_slot_counts = Counter()
    matched_slot_counts = Counter()
    supported_items = matched_items = stable_id_missing = 0
    retained = excluded_scope = excluded_images = 0

    for outfit in raw_outfits:
        supported_slots = set()
        matched_slots = set()
        set_id = str(outfit["set_id"])
        for item in outfit.get("items", []):
            category_id = int(item["categoryid"])
            slot = CATEGORY_TO_SLOT.get(category_id)
            if slot is None:
                unsupported[category_id] += 1
                continue
            supported_items += 1
            supported_slots.add(slot)
            stable_id = image_identifier(item)
            if stable_id is None:
                stable_id_missing += 1
            path = image_path(
                Path(images_root),
                set_id,
                str(item["index"]),
                item_id=stable_id,
            )
            if path is not None:
                matched_items += 1
                matched_slots.add(slot)

        supported_slot_counts[len(supported_slots)] += 1
        matched_slot_counts[len(matched_slots)] += 1
        if len(matched_slots) >= 3:
            retained += 1
        elif len(supported_slots) < 3:
            excluded_scope += 1
        else:
            excluded_images += 1

    return {
        "raw_outfits": len(raw_outfits),
        "raw_items": sum(len(outfit.get("items", [])) for outfit in raw_outfits),
        "supported_items": supported_items,
        "supported_item_image_matches": matched_items,
        "supported_item_image_missing": supported_items - matched_items,
        "supported_item_image_match_rate": (
            matched_items / supported_items if supported_items else 0.0
        ),
        "supported_items_without_stable_tid": stable_id_missing,
        "retained_outfits_minimum_3_slots": retained,
        "excluded_fewer_than_3_supported_slots": excluded_scope,
        "excluded_due_to_images_after_3_supported_slots": excluded_images,
        "outfits_by_unique_supported_slot_count": dict(sorted(supported_slot_counts.items())),
        "outfits_by_unique_matched_slot_count": dict(sorted(matched_slot_counts.items())),
        "top_unsupported_categories": [
            {
                "category_id": category_id,
                "name": names.get(category_id, "Unknown"),
                "items": count,
            }
            for category_id, count in unsupported.most_common(20)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", default="data/polyvore")
    parser.add_argument("--images-dir", default="data/polyvore_images")
    parser.add_argument("--output", default="results/polyvore_retention_audit.json")
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir)
    names = category_names(metadata_dir / "category_id.txt")
    splits = {
        "train": "train_no_dup.json",
        "validation": "valid_no_dup.json",
        "test": "test_no_dup.json",
    }
    report = {
        split: audit_split(metadata_dir / filename, args.images_dir, names)
        for split, filename in splits.items()
    }
    totals = {
        key: sum(split[key] for split in report.values())
        for key in (
            "raw_outfits",
            "raw_items",
            "supported_items",
            "supported_item_image_matches",
            "supported_item_image_missing",
            "retained_outfits_minimum_3_slots",
            "excluded_fewer_than_3_supported_slots",
            "excluded_due_to_images_after_3_supported_slots",
        )
    }
    totals["supported_item_image_match_rate"] = (
        totals["supported_item_image_matches"] / totals["supported_items"]
    )
    report["overall"] = totals

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved retention audit to {output}")


if __name__ == "__main__":
    main()
