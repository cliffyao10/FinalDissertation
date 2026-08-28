"""Extend the frozen item cache with images referenced by official FITB."""

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from recommendation_training.embedding import DEFAULT_MODEL
from recommendation_training.evaluate_official_fitb import load_reference_lookup
from recommendation_training.prepare_polyvore import create_embedding_cache


def required_fitb_items(
    metadata_path, questions_path, images_root, item_metadata_path=None
):
    lookup = load_reference_lookup(metadata_path, item_metadata_path)
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    references = {
        reference
        for question in questions
        for reference in (*question["question"], *question["answers"])
    }
    items = {}
    skipped = {"unsupported": 0, "missing_image": 0, "unknown_reference": 0}
    for reference in references:
        record = lookup.get(reference)
        if record is None:
            skipped["unknown_reference"] += 1
            continue
        if record["slot"] is None or record["item_id"] is None:
            skipped["unsupported"] += 1
            continue
        path = Path(images_root) / "images" / f"{record['item_id']}.jpg"
        if not path.is_file():
            skipped["missing_image"] += 1
            continue
        items[record["item_id"]] = {
            "item_id": record["item_id"],
            "path": str(path),
        }
    return items, skipped, len(references)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/polyvore/test_no_dup.json")
    parser.add_argument("--item-metadata")
    parser.add_argument("--questions", default="data/polyvore/fill_in_blank_test.json")
    parser.add_argument("--images-root", default="data/polyvore_images")
    parser.add_argument("--base-cache", default="data/processed/polyvore_item_embeddings.pt")
    parser.add_argument("--output", default="data/processed/fitb_item_embeddings.pt")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-save-every", type=int, default=10)
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.exists():
        base = torch.load(args.base_cache, map_location="cpu", weights_only=False)
        if base.get("model_name") != args.model:
            raise ValueError("Base cache model does not match the requested encoder.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.base_cache, output_path)

    items, skipped, reference_count = required_fitb_items(
        args.metadata, args.questions, args.images_root, args.item_metadata
    )
    cache = create_embedding_cache(
        items,
        output_path,
        args.model,
        args.batch_size,
        save_every_batches=args.cache_save_every,
    )
    print(json.dumps({
        "official_references": reference_count,
        "supported_references_with_images": len(items),
        "cache_items": len(cache["item_ids"]),
        "skipped": skipped,
        "output": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()
