"""Optional production adapter for the trained compatibility checkpoint."""

from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "compatibility_ranker.pt"
DEFAULT_CATALOGUE = PROJECT_ROOT / "models" / "catalogue_embeddings.pt"


def artifacts_available(checkpoint=DEFAULT_CHECKPOINT, catalogue=DEFAULT_CATALOGUE):
    return Path(checkpoint).is_file() and Path(catalogue).is_file()


@lru_cache(maxsize=2)
def _load_runtime(checkpoint_string, catalogue_string):
    import torch

    from recommendation_training.ranker import OutfitCandidateRanker

    checkpoint = Path(checkpoint_string)
    catalogue = Path(catalogue_string)
    ranker = OutfitCandidateRanker(checkpoint)
    payload = torch.load(catalogue, map_location="cpu", weights_only=False)
    items = payload["items"]
    return ranker, items, payload.get("model_name")


def _weather_filter(items, weather):
    if not weather:
        return items, []
    temperature = float(weather.get("feels_like", 20.0))
    rain = float(weather.get("rain_probability", 0.0)) >= 50
    condition = str(weather.get("condition", "")).casefold()
    retained, constraints = [], []
    for item in items:
        minimum = float(item.get("minimum_temperature") or -100)
        maximum = float(item.get("maximum_temperature") or 100)
        tags = {
            tag.strip().casefold()
            for tag in str(item.get("weather_tags", "")).split(";")
            if tag.strip()
        }
        if not minimum <= temperature <= maximum:
            continue
        if rain or condition in {"rain", "thunderstorm", "snow"}:
            if item["slot"] in {"outer_top", "shoes"} and tags and not (
                tags & {"rain", "waterproof", "water-resistant", "snow"}
            ):
                continue
            constraints.append("precipitation")
        retained.append(item)
    return retained, list(dict.fromkeys(constraints))


def _style_filter(items, style):
    matched = [
        item
        for item in items
        if not item.get("style")
        or style.casefold()
        in {value.strip().casefold() for value in str(item["style"]).split(";")}
    ]
    return matched if matched else items


def recommend_with_trained_model(
    input_slot,
    input_category,
    input_colour,
    input_embedding,
    style,
    weather,
    slot_labels,
    checkpoint=DEFAULT_CHECKPOINT,
    catalogue=DEFAULT_CATALOGUE,
):
    """Return two model-ranked outfits, or ``None`` when artifacts are absent."""

    if input_embedding is None or not artifacts_available(checkpoint, catalogue):
        return None

    import torch

    from recommendation_training.ranker import choose_diverse_pair

    ranker, catalogue_items, embedding_model = _load_runtime(
        str(Path(checkpoint).resolve()), str(Path(catalogue).resolve())
    )
    candidates, constraints = _weather_filter(catalogue_items, weather)
    candidates = _style_filter(candidates, style)
    candidates_by_slot = {slot: [] for slot in slot_labels}
    for item in candidates:
        candidates_by_slot.setdefault(item["slot"], []).append(item)
    for slot in candidates_by_slot:
        candidates_by_slot[slot] = candidates_by_slot[slot][:10]

    fixed_item = {
        "item_id": "uploaded_item",
        "slot": input_slot,
        "type": input_category,
        "colour": input_colour,
        "embedding": torch.as_tensor(input_embedding).float().cpu(),
    }
    ranked = ranker.rank({input_slot: fixed_item}, candidates_by_slot, limit=1000)
    primary, alternative = choose_diverse_pair(ranked)
    if primary is None or alternative is None:
        raise ValueError("The filtered catalogue cannot form two complete outfits.")

    def format_outfit(candidate):
        companion_items = []
        for item in candidate["items"]:
            if item["slot"] == input_slot:
                continue
            companion_items.append(
                {
                    "item_id": str(item.get("item_id", "")),
                    "slot": item["slot"],
                    "slot_label": slot_labels[item["slot"]],
                    "type": item["type"],
                    "colour": item["colour"],
                    "label": f'{item["colour"]} {item["type"]}',
                    "image_path": str(item.get("image_path", "")),
                }
            )
        return {
            "style": style,
            "input_slot": input_slot,
            "items": companion_items,
            "constraints_applied": constraints,
            "model_score": round(candidate["compatibility_score"] * 100, 1),
            "score_components": {},
        }

    primary_outfit = format_outfit(primary)
    alternative_outfit = format_outfit(alternative)
    return {
        "items": [item["label"] for item in primary_outfit["items"]],
        "primary": primary_outfit,
        "alternative": alternative_outfit,
        "method": "trained_siglip_compatibility_ranker_v1",
        "model": {
            "type": "trained_neural_compatibility_ranker",
            "checkpoint": str(checkpoint),
            "catalogue": str(catalogue),
            "embedding_model": embedding_model,
            "training_metadata": ranker.metadata,
            "candidates_scored": len(ranked),
        },
        "explanation": (
            "A compatibility network trained on frozen SigLIP item embeddings "
            "ranked complete catalogue outfits after hard weather filtering."
        ),
    }
