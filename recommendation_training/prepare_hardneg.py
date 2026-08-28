"""Convert Maryland Polyvore hard-negative compatibility files to tensors.

The published compatibility rows use ``set_id_item_index`` identifiers.  This
module maps those identifiers back to stable image IDs in the official outfit
metadata, retains the four garment slots supported by Cove, and reuses the
frozen SigLIP cache produced by :mod:`prepare_polyvore`.
"""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from recommendation_training.dataset import SLOTS
from recommendation_training.embedding import DEFAULT_MODEL
from recommendation_training.prepare_polyvore import (
    CATEGORY_TO_SLOT,
    create_embedding_cache,
    image_identifier,
    image_path,
)


SPLIT_METADATA = {
    "train": "train_no_dup.json",
    "validation": "valid_no_dup.json",
    "test": "test_no_dup.json",
}

SPLIT_COMPATIBILITY = {
    "train": "compatibility_train.txt",
    "validation": "compatibility_valid.txt",
    "test": "compatibility_test.txt",
}


def split_metadata_path(directory, split):
    """Resolve either Maryland ``*_no_dup`` or Polyvore-Outfits JSON names."""

    directory = Path(directory)
    primary = directory / SPLIT_METADATA[split]
    if primary.is_file():
        return primary
    alternative_name = "valid.json" if split == "validation" else f"{split}.json"
    alternative = directory / alternative_name
    if alternative.is_file():
        return alternative
    raise FileNotFoundError(
        f"No metadata file found for {split}: {primary} or {alternative}"
    )


def build_item_lookup(metadata_path, images_root=None, item_metadata=None):
    """Map each published ``set_id_index`` key to a supported item record."""

    outfits = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    lookup = {}
    for outfit in outfits:
        set_id = str(outfit["set_id"])
        for item in outfit.get("items", []):
            item_id = str(item.get("item_id") or image_identifier(item) or "")
            category_id = item.get("categoryid")
            if category_id is None and item_metadata is not None:
                category_id = item_metadata.get(item_id, {}).get("category_id")
            slot = (
                CATEGORY_TO_SLOT.get(int(category_id))
                if category_id not in (None, "")
                else None
            )
            if slot is None or item_id is None:
                continue
            key = f'{set_id}_{item["index"]}'
            record = {"item_id": item_id, "slot": slot, "set_id": set_id}
            if images_root is not None:
                path = image_path(
                    Path(images_root),
                    set_id,
                    str(item["index"]),
                    item_id=item_id,
                )
                if path is not None:
                    record["path"] = str(path)
            lookup[key] = record
    return lookup


def parse_compatibility_rows(path):
    """Yield integer labels and published item keys from a compatibility file."""

    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.split()
            if not fields:
                continue
            if fields[0] not in {"0", "1"}:
                raise ValueError(f"Invalid label on line {line_number}: {fields[0]}")
            yield float(fields[0]), fields[1:]


def build_hard_negative_examples(
    rows,
    item_lookup,
    embedding_by_id,
    minimum_slots=3,
):
    """Create four-slot tensors and report why incompatible rows were skipped."""

    if not embedding_by_id:
        raise ValueError("The Polyvore embedding cache is empty.")
    dimension = next(iter(embedding_by_id.values())).shape[-1]
    embeddings, masks, labels = [], [], []
    skipped = {
        "fewer_than_minimum_slots": 0,
        "missing_embedding": 0,
        "duplicate_slot": 0,
    }
    for label, item_keys in rows:
        selected = {}
        duplicate_slot = False
        missing_embedding = False
        for item_key in item_keys:
            item = item_lookup.get(item_key)
            if item is None:
                continue
            if item["item_id"] not in embedding_by_id:
                missing_embedding = True
                continue
            if item["slot"] in selected:
                duplicate_slot = True
                continue
            selected[item["slot"]] = item["item_id"]
        if len(selected) < minimum_slots:
            skipped["fewer_than_minimum_slots"] += 1
            if missing_embedding:
                skipped["missing_embedding"] += 1
            if duplicate_slot:
                skipped["duplicate_slot"] += 1
            continue

        example_embeddings, example_mask = [], []
        for slot in SLOTS:
            item_id = selected.get(slot)
            example_mask.append(item_id is not None)
            example_embeddings.append(
                embedding_by_id[item_id]
                if item_id is not None
                else torch.zeros(dimension)
            )
        embeddings.append(torch.stack(example_embeddings))
        masks.append(example_mask)
        labels.append(label)

    if not labels:
        raise RuntimeError("No hard-negative compatibility rows could be retained.")
    return {
        "embeddings": torch.stack(embeddings),
        "masks": torch.tensor(masks, dtype=torch.bool),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "positive_examples": int(sum(labels)),
        "negative_examples": int(len(labels) - sum(labels)),
        "minimum_slots": int(minimum_slots),
        "skipped_rows": skipped,
        "negative_strategy": "published_same-type_hard_negatives",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", default="data/polyvore")
    parser.add_argument("--hardneg-dir", default="data/polyvore_hardneg")
    parser.add_argument("--images-dir", default="data/polyvore_images")
    parser.add_argument(
        "--item-metadata",
        help="Optional item metadata for splits that contain no category IDs.",
    )
    parser.add_argument(
        "--embedding-cache",
        default="data/processed/polyvore_item_embeddings.pt",
    )
    parser.add_argument("--output-dir", default="data/processed/hardneg")
    parser.add_argument("--minimum-slots", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-save-every", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--source-label",
        default="Maryland Polyvore hard-negative compatibility split",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_METADATA),
        default=list(SPLIT_METADATA),
    )
    args = parser.parse_args()

    item_metadata = (
        json.loads(Path(args.item_metadata).read_text(encoding="utf-8"))
        if args.item_metadata
        else None
    )
    lookups = {
        split: build_item_lookup(
            split_metadata_path(args.metadata_dir, split),
            args.images_dir,
            item_metadata,
        )
        for split in args.splits
    }
    required_items = {}
    for split, lookup in lookups.items():
        used_keys = {
            key
            for _, keys in parse_compatibility_rows(
                Path(args.hardneg_dir) / SPLIT_COMPATIBILITY[split]
            )
            for key in keys
        }
        for key in used_keys:
            item = lookup.get(key)
            if item is not None and item.get("path"):
                required_items[item["item_id"]] = item
    cache = create_embedding_cache(
        required_items,
        Path(args.embedding_cache),
        args.model,
        args.batch_size,
        save_every_batches=args.cache_save_every,
    )
    embedding_by_id = dict(zip(cache["item_ids"], cache["embeddings"].float()))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        lookup = lookups[split]
        rows = parse_compatibility_rows(
            Path(args.hardneg_dir) / SPLIT_COMPATIBILITY[split]
        )
        payload = build_hard_negative_examples(
            rows,
            lookup,
            embedding_by_id,
            minimum_slots=args.minimum_slots,
        )
        payload.update(
            {
                "source_rows": payload["positive_examples"]
                + payload["negative_examples"]
                + payload["skipped_rows"]["fewer_than_minimum_slots"],
                "siglip_model": cache.get("model_name"),
                "slot_names": list(SLOTS),
                "source": args.source_label,
            }
        )
        output_path = output_dir / f"{split}.pt"
        torch.save(payload, output_path)
        print(
            f"saved={output_path} positives={payload['positive_examples']} "
            f"negatives={payload['negative_examples']} skipped={payload['skipped_rows']}"
        )


if __name__ == "__main__":
    main()
