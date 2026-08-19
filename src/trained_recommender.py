"""Optional production adapter for the trained compatibility checkpoint."""

from functools import lru_cache
from pathlib import Path

from src.garment_taxonomy import broad_garment_category


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
    retained, constraints = [], ["temperature"]
    precipitation = rain or condition in {"rain", "thunderstorm", "snow"}
    if precipitation:
        constraints.append("precipitation")
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
        if precipitation:
            if item["slot"] in {"outer_top", "shoes"} and not (
                tags & {"rain", "waterproof", "water-resistant", "snow"}
            ):
                continue
        retained.append(item)
    return retained, list(dict.fromkeys(constraints))


def _style_filter(items, style):
    """Prefer the selected style without accidentally emptying a slot."""

    by_slot = {}
    for item in items:
        by_slot.setdefault(item["slot"], []).append(item)

    retained = []
    target = style.casefold()
    for slot_items in by_slot.values():
        matched = [
            item
            for item in slot_items
            if not item.get("style")
            or target
            in {
                value.strip().casefold()
                for value in str(item["style"]).split(";")
            }
        ]
        retained.extend(matched or slot_items)
    return retained


def _shortlist_candidates(items, input_embedding, limit=10, diverse_colours=4):
    """Choose an input-dependent, colour-diverse shortlist from a full slot."""

    import torch

    target = torch.nn.functional.normalize(
        torch.as_tensor(input_embedding).float().cpu(),
        dim=0,
    )
    ranked = []
    for item in items:
        embedding = torch.as_tensor(item["embedding"]).float().cpu()
        if embedding.shape != target.shape:
            raise ValueError("Catalogue and uploaded-item embeddings have different shapes.")
        similarity = torch.dot(
            target,
            torch.nn.functional.normalize(embedding, dim=0),
        ).item()
        ranked.append((similarity, str(item.get("item_id", "")), item))
    ranked.sort(key=lambda candidate: (-candidate[0], candidate[1]))

    selected, selected_ids, colours = [], set(), set()
    for _, item_id, item in ranked:
        colour = str(item.get("colour", "")).casefold()
        if colour in colours:
            continue
        selected.append(item)
        selected_ids.add(item_id)
        colours.add(colour)
        if len(selected) == min(diverse_colours, limit):
            break
    for _, item_id, item in ranked:
        if item_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item_id)
        if len(selected) == limit:
            break
    return selected


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
    all_candidates_by_slot = {slot: [] for slot in slot_labels}
    for item in candidates:
        all_candidates_by_slot.setdefault(item["slot"], []).append(item)
    candidates_by_slot = {}
    for slot, slot_items in all_candidates_by_slot.items():
        candidates_by_slot[slot] = _shortlist_candidates(
            slot_items,
            input_embedding,
            limit=10,
        )

    required_candidate_slots = set(slot_labels) - {input_slot}
    missing_slots = sorted(
        slot for slot in required_candidate_slots if not candidates_by_slot.get(slot)
    )
    if missing_slots:
        raise ValueError(
            "No catalogue candidates remain after weather/style filtering for: "
            + ", ".join(missing_slots)
        )

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

    def format_item(item):
        broad_category = broad_garment_category(
            item["slot"], item.get("type", "")
        )
        return {
            "item_id": str(item.get("item_id", "")),
            "slot": item["slot"],
            "slot_label": slot_labels[item["slot"]],
            "type": broad_category,
            "colour": item["colour"],
            "label": f'{item["colour"]} {broad_category}',
            "image_path": str(item.get("image_path", "")),
        }

    def format_outfit(candidate):
        fixed_by_slot = {item["slot"]: item for item in candidate["items"]}
        alternatives_by_slot = {}
        for slot, current_item in fixed_by_slot.items():
            if slot == input_slot:
                continue
            current_category = broad_garment_category(
                slot, current_item.get("type", "")
            )
            fixed_items = {
                fixed_slot: fixed_item
                for fixed_slot, fixed_item in fixed_by_slot.items()
                if fixed_slot != slot
            }
            rescored = ranker.rank(
                fixed_items,
                {
                    slot: [
                        item
                        for item in all_candidates_by_slot.get(slot, [])
                        if broad_garment_category(
                            slot, item.get("type", "")
                        ) == current_category
                    ]
                },
                limit=max(1, len(all_candidates_by_slot.get(slot, []))),
            )
            replacements = []
            seen_colours = set()
            for scored_outfit in rescored:
                replacement = next(
                    item for item in scored_outfit["items"] if item["slot"] == slot
                )
                if broad_garment_category(
                    slot, replacement.get("type", "")
                ) != current_category:
                    continue
                replacement_colour = str(replacement.get("colour", ""))
                if (
                    replacement.get("item_id") == current_item.get("item_id")
                    or replacement_colour == str(current_item.get("colour", ""))
                    or replacement_colour.casefold() in seen_colours
                ):
                    continue
                formatted = format_item(replacement)
                formatted.update(
                    {
                        "compatibility_score": round(
                            scored_outfit["compatibility_score"] * 100, 1
                        ),
                        "selection_source": "trained_model_ranked_slot_replacement",
                    }
                )
                replacements.append(formatted)
                seen_colours.add(replacement_colour.casefold())
                if len(replacements) == 3:
                    break
            alternatives_by_slot[slot] = replacements

        companion_items = []
        for item in candidate["items"]:
            if item["slot"] == input_slot:
                continue
            companion_items.append(format_item(item))
        return {
            "style": style,
            "input_slot": input_slot,
            "items": companion_items,
            "constraints_applied": constraints,
            "model_score": round(candidate["compatibility_score"] * 100, 1),
            "score_components": {},
            "alternatives_by_slot": alternatives_by_slot,
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
            "catalogue_items_considered": len(candidates),
            "candidates_scored": len(ranked),
        },
        "explanation": (
            "A compatibility network trained on frozen SigLIP item embeddings "
            "ranked complete catalogue outfits after hard weather filtering."
        ),
    }
