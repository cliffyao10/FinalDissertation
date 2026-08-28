"""Build a compact candidate catalogue from a published Zara dataset ZIP.

The script reads the dataset's bundled CSV, selects a deterministic and
balanced set of products, extracts only one representative image per product,
and uses the same frozen SigLIP encoder as the application to label colour and
style.  It never contacts Zara or any other product website.
"""

import argparse
import ast
import csv
import re
import shutil
import sys
from collections import defaultdict, deque
from pathlib import Path
from zipfile import ZipFile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from recommendation_training.embedding import (
    DEFAULT_MODEL,
    embed_paths,
    load_encoder,
    pooled_image_features,
)


SLOT_TERMS = {
    "inner_top": {
        "sweaters", "cardigans", "hoodies", "sweatshirts", "t-shirts",
        "tops", "bodysuits", "knitwear", "linen",
    },
    "outer_top": {"jackets", "puffers", "overshirts", "blazers", "coats"},
    "bottom": {"pants", "jeans", "shorts", "skirts", "tracksuits", "suits"},
    "shoes": {"shoes"},
}

SLOT_KEYWORDS = {
    "shoes": (
        "shoe", "shoes", "sneaker", "sneakers", "sandal", "sandals",
        "loafer", "loafers", "boot", "boots", "heel", "heels", "pump",
        "pumps", "moccasin", "moccasins", "espadrille", "espadrilles",
        "high top", "high tops",
    ),
    "outer_top": (
        "jacket", "jackets", "coat", "coats", "blazer", "blazers",
        "overshirt", "overshirts", "puffer", "puffers", "parka", "parkas",
        "trench", "windbreaker",
    ),
    "bottom": (
        "jean", "jeans", "pant", "pants", "trouser", "trousers", "shorts",
        "skirt", "skirts", "legging", "leggings", "jogger", "joggers",
        "cargo pants", "culottes",
    ),
    "inner_top": (
        "top", "tops", "shirt", "shirts", "sweater", "sweaters",
        "cardigan", "cardigans", "hoodie", "hoodies", "sweatshirt",
        "sweatshirts", "t-shirt", "t-shirts", "tee", "tees", "bodysuit",
        "bodysuits", "vest", "vests", "polo", "blouse", "blouses",
        "tank top", "camisole",
    ),
}

ACCESSORY_KEYWORDS = (
    "bag", "backpack", "ball", "wallet", "purse", "fragrance", "perfume",
    "sunglasses", "necklace", "earring", "bracelet", "watch", "scarf",
    "glove", "gloves", "hat", "cap",
)

ONE_PIECE_KEYWORDS = ("dress", "dresses", "jumpsuit", "jumpsuits", "romper")

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


def parse_download_ids(value):
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if item]


def infer_slot(name, description):
    """Infer a four-piece slot from the product name.

    Descriptions often mention styling partners (for example shoes for a pair
    of ski pants), so using them for category inference creates false matches.
    One-piece garments use the ``inner_top`` anchor slot.  Their additional
    bottom-slot occupancy is recorded separately by :func:`occupied_slots`.
    Accessories remain outside this model.
    """

    del description
    text = str(name).casefold()
    if any(re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in ACCESSORY_KEYWORDS):
        return None
    # "Dress shoes" are footwear, but a T-shirt dress is a one-piece garment.
    for keyword in SLOT_KEYWORDS["shoes"]:
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            return "shoes"
    if any(re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in ONE_PIECE_KEYWORDS):
        return "inner_top"
    for slot in ("outer_top", "bottom", "inner_top"):
        for keyword in SLOT_KEYWORDS[slot]:
            if re.search(rf"\b{re.escape(keyword)}\b", text):
                return slot
    return None


def occupied_slots(name, inferred_slot):
    """Return the logical slots occupied by a catalogue garment."""

    text = str(name).casefold()
    if inferred_slot == "inner_top" and any(
        re.search(rf"\b{re.escape(keyword)}\b", text)
        for keyword in ONE_PIECE_KEYWORDS
    ):
        return "inner_top;bottom"
    return str(inferred_slot or "")


def _round_robin_candidates(frame, slot, limit):
    groups = defaultdict(deque)
    for _, row in frame.sort_values(["terms", "sku"]).iterrows():
        if row["inferred_slot"] == slot:
            groups[row["terms"]].append(row)
    selected = []
    group_keys = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for key in group_keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].popleft())
    return selected


def has_precipitation_protection(row):
    """Return whether published product text claims wet-weather protection."""

    text = f'{row.get("name", "")} {row.get("description", "")} {row.get("terms", "")}'.casefold()
    return any(
        phrase in text
        for phrase in (
            "waterproof",
            "water-resistant",
            "water resistant",
            "water-repellent",
            "water repellent",
            "raincoat",
            "rain jacket",
            "rain boot",
        )
    )


def select_products(frame, archive, per_slot):
    available_entries = set(archive.namelist())
    selected = []
    for slot in SLOT_TERMS:
        by_section = {
            section: deque(
                _round_robin_candidates(
                    frame[frame["section"] == section],
                    slot,
                    per_slot * 3,
                )
            )
            for section in ("WOMAN", "MAN")
        }
        candidates = []
        while len(candidates) < per_slot * 3 and any(by_section.values()):
            for section in ("WOMAN", "MAN"):
                if by_section[section]:
                    candidates.append(by_section[section].popleft())
        usable_candidates = []
        seen_skus, seen_images = set(), set()
        for row in candidates:
            image_id = next(
                (
                    image_id
                    for image_id in parse_download_ids(row["image_downloads"])
                    if f"images/zara/{image_id}.jpg" in available_entries
                ),
                None,
            )
            if image_id is None:
                continue
            sku = str(row["sku"])
            if sku in seen_skus or image_id in seen_images:
                continue
            seen_skus.add(sku)
            seen_images.add(image_id)
            usable_candidates.append((row, image_id))

        protected_required = (
            max(2, round(per_slot * 0.15))
            if slot in {"outer_top", "shoes"}
            else 0
        )
        protected = [
            candidate
            for candidate in usable_candidates
            if has_precipitation_protection(candidate[0])
        ][:protected_required]
        protected_skus = {str(row["sku"]) for row, _ in protected}
        slot_products = protected + [
            candidate
            for candidate in usable_candidates
            if str(candidate[0]["sku"]) not in protected_skus
        ][: per_slot - len(protected)]
        if len(slot_products) < per_slot:
            raise RuntimeError(
                f"Only {len(slot_products)} usable products were found for {slot}."
            )
        selected.extend((slot, row, image_id) for row, image_id in slot_products)
    return selected


def extract_selected_images(archive, selected, output_directory):
    image_directory = output_directory / "images"
    image_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for _, _, image_id in selected:
        archive_name = f"images/zara/{image_id}.jpg"
        destination = image_directory / f"{image_id}.jpg"
        if not destination.exists():
            with archive.open(archive_name) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
        paths.append(destination)
    return paths


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


def classify_images(paths, model_name, batch_size):
    processor, model, device = load_encoder(model_name)
    colour_text = text_embeddings(COLOUR_PROMPTS, processor, model, device)
    style_text = text_embeddings(STYLE_PROMPTS, processor, model, device)
    colour_labels = list(COLOUR_PROMPTS.values())
    style_labels = list(STYLE_PROMPTS.values())
    results = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = embed_paths(batch_paths, processor, model, device)
        colour_scores = images @ colour_text.T
        style_scores = images @ style_text.T
        for colour_row, style_row in zip(colour_scores, style_scores):
            colour = colour_labels[int(colour_row.argmax())]
            top_styles = torch.topk(style_row, k=2).indices.tolist()
            styles = ";".join(style_labels[index] for index in top_styles)
            results.append((colour, styles))
        print(f"classified={min(start + batch_size, len(paths))}/{len(paths)}")
    return results


def weather_metadata(name, description, terms):
    text = f"{name} {description} {terms}".casefold()
    tags = []
    minimum, maximum = -5, 32
    if any(word in text for word in ("puffer", "coat", "wool", "shearling")):
        minimum, maximum = -10, 15
        tags.append("cold")
    elif any(word in text for word in ("linen", "shorts", "sandal", "tank")):
        minimum, maximum = 18, 40
        tags.append("warm")
    if any(
        word in text
        for word in (
            "waterproof",
            "water-resistant",
            "water resistant",
            "water-repellent",
            "water repellent",
            "raincoat",
            "rain jacket",
            "rain boot",
        )
    ):
        tags.extend(("rain", "waterproof"))
    return minimum, maximum, ";".join(dict.fromkeys(tags))


def build_records(selected, paths, classifications, project_root):
    records = []
    for (slot, row, image_id), image_path, (colour, style) in zip(
        selected, paths, classifications
    ):
        minimum, maximum, weather_tags = weather_metadata(
            row["name"], row.get("description", ""), row["terms"]
        )
        relative_image = image_path.resolve().relative_to(project_root.resolve())
        records.append(
            {
                "item_id": str(row["sku"]),
                "slot": slot,
                "occupies_slots": occupied_slots(row["name"], slot),
                "type": str(row["name"]).title(),
                "colour": colour,
                "style": style,
                "image_path": relative_image.as_posix(),
                "minimum_temperature": minimum,
                "maximum_temperature": maximum,
                "weather_tags": weather_tags,
                "section": row["section"],
                "source_term": row["terms"],
                "source_url": row["url"],
                "price": row["price"],
                "currency": row["currency"],
                "image_id": image_id,
                "source_scraped_at": row["scraped_at"],
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="data/incoming/archive.zip")
    parser.add_argument("--output-dir", default="data/zara_products")
    parser.add_argument("--catalogue", default="data/catalogue.csv")
    parser.add_argument("--per-slot", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    archive_path = Path(args.archive)
    output_directory = Path(args.output_dir)
    project_root = Path(__file__).resolve().parents[1]
    with ZipFile(archive_path) as archive:
        with archive.open("store_zara.csv") as csv_file:
            frame = pd.read_csv(csv_file)
        usable = frame[
            frame["error"].isna()
            & frame["image_downloads"].notna()
        ].copy()
        usable["inferred_slot"] = usable.apply(
            lambda row: infer_slot(row["name"], row.get("description", "")),
            axis=1,
        )
        usable = usable[usable["inferred_slot"].notna()]
        selected = select_products(usable, archive, args.per_slot)
        paths = extract_selected_images(archive, selected, output_directory)

    classifications = classify_images(paths, args.model, args.batch_size)
    records = build_records(selected, paths, classifications, project_root)
    catalogue_path = Path(args.catalogue)
    catalogue_path.parent.mkdir(parents=True, exist_ok=True)
    with catalogue_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    counts = pd.DataFrame(records).groupby("slot").size().to_dict()
    print(f"Saved {len(records)} products to {catalogue_path}: {counts}")


if __name__ == "__main__":
    main()
