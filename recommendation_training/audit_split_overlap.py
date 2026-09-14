"""Audit item and outfit identifier overlap in Polyvore split metadata."""

import argparse
import itertools
import json
from pathlib import Path


DEFAULT_FILES = {
    "train": "train.json",
    "validation": "valid.json",
    "test": "test.json",
}


def identifiers(path):
    outfits = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "outfits": {str(outfit["set_id"]) for outfit in outfits},
        "items": {
            str(item["item_id"])
            for outfit in outfits
            for item in outfit.get("items", [])
            if item.get("item_id") is not None
        },
    }


def audit(directory):
    directory = Path(directory)
    split_ids = {
        split: identifiers(directory / filename)
        for split, filename in DEFAULT_FILES.items()
    }
    report = {
        "source_directory": str(directory),
        "split_counts": {
            split: {
                "outfits": len(values["outfits"]),
                "unique_item_ids": len(values["items"]),
            }
            for split, values in split_ids.items()
        },
        "pairwise_overlap": {},
    }
    for left, right in itertools.combinations(DEFAULT_FILES, 2):
        report["pairwise_overlap"][f"{left}__{right}"] = {
            "outfit_set_ids": len(
                split_ids[left]["outfits"] & split_ids[right]["outfits"]
            ),
            "item_ids": len(split_ids[left]["items"] & split_ids[right]["items"]),
        }
    report["strict_item_disjoint_verified"] = all(
        comparison["item_ids"] == 0
        for comparison in report["pairwise_overlap"].values()
    )
    report["outfit_set_disjoint_verified"] = all(
        comparison["outfit_set_ids"] == 0
        for comparison in report["pairwise_overlap"].values()
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="data/polyvore_disjoint")
    parser.add_argument("--output", default="results/polyvore_disjoint_overlap_audit.json")
    args = parser.parse_args()
    result = audit(args.directory)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
