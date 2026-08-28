"""Derive a compact, brand-free recommendation space from Polyvore training data.

The saved artifact contains one real-embedding medoid for every supported
``slot × broad type × colour × style`` state.  Product photographs, titles,
brands and URLs are deliberately omitted: the application renders only a
filled garment icon for these abstract recommendations.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from recommendation_training.embedding import (
    DEFAULT_MODEL,
    load_encoder,
    pooled_image_features,
)
from recommendation_training.prepare_polyvore import CATEGORY_TO_SLOT
from src.audience import audience_for_category_id


COLOUR_PROMPTS = {
    f"a piece of {colour.lower()} clothing": colour
    for colour in (
        "Black", "White", "Grey", "Red", "Orange", "Yellow", "Green",
        "Blue", "Purple", "Pink", "Brown", "Beige",
    )
}

STYLE_PROMPTS = {
    "casual everyday clothing with a relaxed appearance": "Casual",
    "formal clothing for ceremonies or elegant occasions": "Formal",
    "business or professional office clothing": "Business",
    "sporty athletic or activewear clothing": "Sporty",
    "outdoor technical hiking or weather-protective clothing": "Outdoor",
    "urban streetwear clothing": "Streetwear",
    "minimalist clothing with a simple clean design": "Minimalist",
    "party or evening social-event clothing": "Party",
    "beachwear or clothing for swimming": "Beachwear",
}


TYPE_PROMPTS = {
    "inner_top": {
        "Tank Top": "a photo of a sleeveless camisole or tank top",
        "T-Shirt": "a photo of a short-sleeved T-shirt",
        "Shirt": "a photo of a collared button-up shirt",
        "Blouse": "a photo of a blouse",
        "Sweater": "a photo of a knitted sweater or pullover",
        "Hoodie": "a photo of a hooded sweatshirt",
        "Top": "a photo of a general upper-body top",
    },
    "outer_top": {
        "Jacket": "a photo of a short outer jacket or windbreaker",
        "Blazer": "a photo of a structured blazer or suit jacket",
        "Coat": "a photo of a long outer coat",
        "Cardigan": "a photo of an open-front knitted cardigan",
        "Overshirt": "a photo of a relaxed outer overshirt",
    },
    "bottom": {
        "Jeans": "a photo of denim jeans",
        "Trousers": "a photo of trousers or casual pants",
        "Shorts": "a photo of shorts",
        "Skirt": "a photo of a skirt",
        "Joggers": "a photo of joggers or sweatpants",
    },
    "shoes": {
        "Boots": "a photo of boots",
        "Sandals": "a photo of sandals",
        "Trainers": "a photo of trainers or sneakers",
        "Loafers": "a photo of loafers",
        "Heels": "a photo of high-heeled shoes",
        "Shoes": "a photo of general flat shoes",
    },
}

# Respect the catalogue's explicit men's taxonomy even when zero-shot visual
# type prompts are uncertain.  This is a candidate-safety rule, not a claim
# about what an individual is allowed to wear; the UI exposes clothing ranges.
MENSWEAR_TYPE_EXCLUSIONS = {
    "inner_top": {"Blouse"},
    "bottom": {"Skirt"},
    "shoes": {"Heels"},
}


def _slug(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value).casefold()).strip("-")


@torch.inference_mode()
def text_embeddings(prompts, processor, model, device):
    inputs = processor(
        text=list(prompts),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}
    features = pooled_image_features(model.get_text_features(**inputs))
    return torch.nn.functional.normalize(features, dim=-1).cpu()


def _weather_fields(garment_type, style):
    """Attach conservative weather meaning to the abstract garment concept."""

    garment_type = str(garment_type)
    style = str(style)
    minimum, maximum, tags = -5, 32, []
    if garment_type in {"Coat", "Sweater", "Hoodie", "Boots"}:
        minimum, maximum = -10, 20
        tags.append("cold")
    elif garment_type in {"Tank Top", "Shorts", "Sandals"}:
        minimum, maximum = 18, 40
        tags.append("warm")
    elif garment_type == "Jacket":
        minimum, maximum = -5, 26
    if style == "Outdoor" and garment_type in {"Jacket", "Coat", "Boots", "Shoes"}:
        tags.extend(("rain", "water-resistant"))
    return minimum, maximum, ";".join(tags)


def select_medoid(embeddings):
    """Return the member nearest the normalised group centroid."""

    if embeddings.ndim != 2 or not len(embeddings):
        raise ValueError("Medoid selection expects a non-empty embedding matrix.")
    normalised = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    centroid = torch.nn.functional.normalize(normalised.mean(dim=0), dim=0)
    return int((normalised @ centroid).argmax())


def _load_training_records(split_path, metadata_path, cache_path, maximum_items=None):
    outfits = json.loads(Path(split_path).read_text(encoding="utf-8"))
    training_ids = []
    seen = set()
    for outfit in outfits:
        for item in outfit.get("items", []):
            item_id = str(item.get("item_id", ""))
            if item_id and item_id not in seen:
                seen.add(item_id)
                training_ids.append(item_id)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache_index = {item_id: index for index, item_id in enumerate(cache["item_ids"])}
    records = []
    skipped = Counter()
    for item_id in training_ids:
        details = metadata.get(item_id, {})
        category_id = details.get("category_id")
        slot = (
            CATEGORY_TO_SLOT.get(int(category_id))
            if category_id not in (None, "")
            else None
        )
        if slot is None:
            skipped["unsupported_category"] += 1
            continue
        embedding_index = cache_index.get(item_id)
        if embedding_index is None:
            skipped["missing_embedding"] += 1
            continue
        records.append(
            (
                item_id,
                slot,
                audience_for_category_id(category_id),
                cache["embeddings"][embedding_index].float(),
            )
        )
        if maximum_items and len(records) >= maximum_items:
            break
    if not records:
        raise RuntimeError("No supported Polyvore training items matched the embedding cache.")
    return records, skipped, cache.get("model_name", DEFAULT_MODEL), len(training_ids)


@torch.inference_mode()
def build_prototypes(records, model_name=DEFAULT_MODEL, batch_size=4096, device=None):
    """Classify abstract states and retain one medoid per supported state."""

    processor, model, encoder_device = load_encoder(model_name, device=device)
    colour_text = text_embeddings(COLOUR_PROMPTS, processor, model, encoder_device)
    style_text = text_embeddings(STYLE_PROMPTS, processor, model, encoder_device)
    colour_labels = list(COLOUR_PROMPTS.values())
    style_labels = list(STYLE_PROMPTS.values())
    type_text = {
        slot: text_embeddings(prompts.values(), processor, model, encoder_device)
        for slot, prompts in TYPE_PROMPTS.items()
    }
    type_labels = {slot: list(prompts) for slot, prompts in TYPE_PROMPTS.items()}

    groups = defaultdict(list)
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        embeddings = torch.stack([record[3] for record in batch]).to(encoder_device)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        colours = (embeddings @ colour_text.to(encoder_device).T).argmax(dim=1)
        styles = (embeddings @ style_text.to(encoder_device).T).argmax(dim=1)
        for row, (item_id, slot, audience, embedding) in enumerate(batch):
            slot_embedding = embeddings[row]
            type_scores = slot_embedding @ type_text[slot].to(encoder_device).T
            if audience == "menswear":
                for index, label in enumerate(type_labels[slot]):
                    if label in MENSWEAR_TYPE_EXCLUSIONS.get(slot, set()):
                        type_scores[index] = -torch.inf
            type_index = int(type_scores.argmax())
            key = (
                slot,
                type_labels[slot][type_index],
                colour_labels[int(colours[row])],
                style_labels[int(styles[row])],
                audience,
            )
            groups[key].append((item_id, embedding))

    prototypes = []
    for key in sorted(groups):
        slot, garment_type, colour, style, audience = key
        members = groups[key]
        member_embeddings = torch.stack([member[1] for member in members])
        medoid_index = select_medoid(member_embeddings)
        minimum, maximum, weather_tags = _weather_fields(garment_type, style)
        prototype_id = "abstract-" + "-".join(_slug(value) for value in key)
        prototypes.append(
            {
                "item_id": prototype_id,
                "prototype_id": prototype_id,
                "slot": slot,
                "occupies_slots": slot,
                "type": garment_type,
                "colour": colour,
                "style": style,
                "audience": audience,
                "embedding": member_embeddings[medoid_index].clone(),
                "support_count": len(members),
                "minimum_temperature": minimum,
                "maximum_temperature": maximum,
                "weather_tags": weather_tags,
                "image_path": "",
                "abstract": True,
                "source": "polyvore_training_derived_prototype",
            }
        )
    return prototypes, str(encoder_device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="data/polyvore_disjoint/train.json")
    parser.add_argument(
        "--item-metadata",
        default="data/polyvore_nondisjoint/polyvore_item_metadata.json",
    )
    parser.add_argument(
        "--embedding-cache",
        default="data/processed_disjoint/polyvore_item_embeddings.pt",
    )
    parser.add_argument(
        "--output",
        default="models/polyvore_abstract_prototypes.pt",
    )
    parser.add_argument(
        "--report",
        default="results/abstract_prototype_build.json",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--maximum-items", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()

    started = time.perf_counter()
    records, skipped, cache_model, source_item_count = _load_training_records(
        args.split,
        args.item_metadata,
        args.embedding_cache,
        maximum_items=args.maximum_items,
    )
    if cache_model != args.model:
        raise ValueError(
            f"Embedding cache uses {cache_model!r}, but {args.model!r} was requested."
        )
    prototypes, device = build_prototypes(
        records,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )
    elapsed = time.perf_counter() - started
    payload = {
        "items": prototypes,
        "model_name": args.model,
        "schema": "abstract_garment_prototypes_v2",
        "source_split": str(args.split),
        "source_policy": "training_split_only_no_product_media_or_brand_output",
        "source_unique_items": source_item_count,
        "classified_items": len(records),
        "prototype_count": len(prototypes),
        "skipped_items": dict(skipped),
        "device": device,
        "build_seconds": elapsed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = {key: value for key, value in payload.items() if key != "items"}
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
