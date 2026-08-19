"""Convert official Polyvore metadata and images into trainable tensors.

For every retained real outfit this script creates a positive sample and a
negative sample by replacing one present item with an item from the same slot.
SigLIP stays frozen; its normalised image vectors are cached for repeatability.
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from recommendation_training.dataset import SLOTS
from recommendation_training.embedding import DEFAULT_MODEL, embed_paths, load_encoder


# Polyvore's category taxonomy contains both parent and leaf IDs.  The IDs
# below cover garment categories used by the four-slot product interface.
CATEGORY_TO_SLOT = {
    **{
        value: "inner_top"
        for value in (
            3, 4, 5, 6, 11, 15, 17, 18, 19, 21, 31, 104, 243, 252,
            272, 273, 275, 282, 286, 309, 341, 343, 1606, 4454, 4495,
            4496, 4497, 4498, 4516, 4517,
        )
    },
    **{
        value: "outer_top"
        for value in (23, 24, 25, 26, 256, 277, 289, 4455, 4456, 4457)
    },
    **{
        value: "bottom"
        for value in (
            7, 8, 9, 10, 27, 28, 29, 30, 237, 238, 239, 240, 253,
            254, 255, 278, 279, 280, 281, 287, 288, 310, 332, 4452,
            4458, 4459,
        )
    },
    **{
        value: "shoes"
        for value in (
            41, 42, 43, 46, 47, 48, 49, 50, 261, 262, 263, 264, 268,
            291, 292, 293, 294, 296, 297, 298, 4465, 4522,
        )
    },
}


def image_identifier(item):
    """Read the stable Polyvore item ID embedded as ``tid`` in its image URL."""

    match = re.search(r"[?&]tid=(\d+)", str(item.get("image", "")))
    return match.group(1) if match else None


def image_path(images_root, set_id, index, item_id=None):
    candidates = (
        images_root / "images" / f"{item_id}.jpg" if item_id else None,
        images_root / f"{item_id}.jpg" if item_id else None,
        images_root / str(set_id) / f"{index}.jpg",
        images_root / str(set_id) / f"{index}.png",
        images_root / f"{set_id}_{index}.jpg",
        images_root / "images" / str(set_id) / f"{index}.jpg",
    )
    return next((path for path in candidates if path is not None and path.is_file()), None)


def load_split(path, images_root, minimum_slots=3, maximum_outfits=None):
    raw_outfits = json.loads(path.read_text(encoding="utf-8"))
    outfits, items = [], {}
    for raw in raw_outfits:
        selected = {}
        set_id = str(raw["set_id"])
        for item in raw.get("items", []):
            slot = CATEGORY_TO_SLOT.get(int(item["categoryid"]))
            if slot is None or slot in selected:
                continue
            index = str(item["index"])
            stable_item_id = image_identifier(item)
            path_value = image_path(
                images_root,
                set_id,
                index,
                item_id=stable_item_id,
            )
            if path_value is None:
                continue
            item_id = stable_item_id or f"{set_id}_{index}"
            record = {
                "item_id": item_id,
                "set_id": set_id,
                "slot": slot,
                "path": str(path_value),
            }
            items[item_id] = record
            selected[slot] = item_id
        if len(selected) >= minimum_slots:
            outfits.append(selected)
            if maximum_outfits and len(outfits) >= maximum_outfits:
                break
    return outfits, items


def _save_embedding_cache(path, item_ids, embeddings, model_name):
    """Atomically replace the resumable image-vector cache."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "item_ids": item_ids,
            "embeddings": embeddings,
            "model_name": model_name,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def create_embedding_cache(
    items,
    output_path,
    model_name,
    batch_size,
    save_every_batches=10,
):
    """Embed missing items and resume from a compatible partial cache."""

    cached_ids = []
    cached_embeddings = None
    if output_path.exists():
        candidate = torch.load(output_path, map_location="cpu", weights_only=False)
        candidate_ids = candidate.get("item_ids", [])
        candidate_embeddings = candidate.get("embeddings")
        compatible = (
            candidate.get("model_name") == model_name
            and isinstance(candidate_ids, list)
            and isinstance(candidate_embeddings, torch.Tensor)
            and len(candidate_ids) == len(candidate_embeddings)
        )
        if compatible:
            cached_ids = list(candidate_ids)
            cached_embeddings = candidate_embeddings.float()
            print(f"resumed_cached_items={len(cached_ids)}")

    cached_id_set = set(cached_ids)
    missing_ids = [item_id for item_id in sorted(items) if item_id not in cached_id_set]
    if not missing_ids:
        return {
            "item_ids": cached_ids,
            "embeddings": cached_embeddings,
            "model_name": model_name,
        }

    processor, model, device = load_encoder(model_name)
    pending_ids, pending_batches = [], []
    completed = 0
    total_required = len(missing_ids)
    for batch_number, start in enumerate(range(0, total_required, batch_size), start=1):
        batch_ids = missing_ids[start : start + batch_size]
        pending_ids.extend(batch_ids)
        pending_batches.append(
            embed_paths(
                [items[item_id]["path"] for item_id in batch_ids],
                processor,
                model,
                device,
            )
        )
        completed += len(batch_ids)
        print(f"embedded_missing={completed}/{total_required}")

        should_save = (
            batch_number % max(1, save_every_batches) == 0
            or completed == total_required
        )
        if should_save:
            new_embeddings = torch.cat(pending_batches)
            cached_embeddings = (
                new_embeddings
                if cached_embeddings is None
                else torch.cat((cached_embeddings, new_embeddings))
            )
            cached_ids.extend(pending_ids)
            _save_embedding_cache(
                output_path,
                cached_ids,
                cached_embeddings,
                model_name,
            )
            pending_ids.clear()
            pending_batches.clear()
            print(f"cache_saved_items={len(cached_ids)}")

    return {
        "item_ids": cached_ids,
        "embeddings": cached_embeddings,
        "model_name": model_name,
    }


def build_examples(outfits, items, embedding_by_id, seed):
    rng = random.Random(seed)
    pools = defaultdict(list)
    for item_id, item in items.items():
        pools[item["slot"]].append(item_id)

    embeddings, masks, labels = [], [], []
    for outfit in outfits:
        positive = dict(outfit)
        replaceable = [
            slot
            for slot, item_id in positive.items()
            if any(
                candidate != item_id
                and items[candidate]["set_id"] != items[item_id]["set_id"]
                for candidate in pools[slot]
            )
        ]
        if not replaceable:
            continue
        replacement_slot = rng.choice(replaceable)
        original_id = positive[replacement_slot]
        replacements = [
            candidate
            for candidate in pools[replacement_slot]
            if candidate != original_id
            and items[candidate]["set_id"] != items[original_id]["set_id"]
        ]
        negative = dict(positive)
        negative[replacement_slot] = rng.choice(replacements)

        for example, label in ((positive, 1.0), (negative, 0.0)):
            example_embeddings = []
            example_mask = []
            dimension = next(iter(embedding_by_id.values())).shape[-1]
            for slot in SLOTS:
                item_id = example.get(slot)
                example_mask.append(item_id is not None)
                example_embeddings.append(
                    embedding_by_id[item_id]
                    if item_id is not None
                    else torch.zeros(dimension)
                )
            embeddings.append(torch.stack(example_embeddings))
            masks.append(example_mask)
            labels.append(label)
    return {
        "embeddings": torch.stack(embeddings),
        "masks": torch.tensor(masks, dtype=torch.bool),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "positive_examples": int(sum(labels)),
        "negative_examples": int(len(labels) - sum(labels)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", default="data/polyvore")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--cache-save-every",
        type=int,
        default=10,
        help="Persist the resumable embedding cache after this many batches.",
    )
    parser.add_argument("--minimum-slots", type=int, default=3)
    parser.add_argument("--maximum-outfits", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir)
    images_root = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    split_names = {"train": "train_no_dup.json", "validation": "valid_no_dup.json", "test": "test_no_dup.json"}
    split_outfits, split_items, all_items = {}, {}, {}
    for split, filename in split_names.items():
        outfits, items = load_split(
            metadata_dir / filename,
            images_root,
            minimum_slots=args.minimum_slots,
            maximum_outfits=args.maximum_outfits,
        )
        split_outfits[split] = outfits
        split_items[split] = items
        all_items.update(items)
        print(f"split={split} usable_outfits={len(outfits)} items={len(items)}")
    if not all_items:
        raise RuntimeError("No images matched the Polyvore metadata paths.")

    cache_path = output_dir / "polyvore_item_embeddings.pt"
    cache = create_embedding_cache(
        all_items,
        cache_path,
        args.model,
        args.batch_size,
        save_every_batches=args.cache_save_every,
    )
    embedding_by_id = dict(zip(cache["item_ids"], cache["embeddings"]))
    for offset, (split, outfits) in enumerate(split_outfits.items()):
        payload = build_examples(
            outfits,
            split_items[split],
            embedding_by_id,
            args.seed + offset,
        )
        payload["source_outfits"] = len(outfits)
        payload["siglip_model"] = cache["model_name"]
        payload["slot_names"] = list(SLOTS)
        payload["seed"] = args.seed + offset
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_dir / f"{split}.pt")
        print(
            f"saved={output_dir / f'{split}.pt'} positives={payload['positive_examples']} "
            f"negatives={payload['negative_examples']}"
        )


if __name__ == "__main__":
    main()
