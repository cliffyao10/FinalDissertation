"""Lightweight, interpretable outfit recommendation model.

The recogniser supplies a garment category, colour and one or more style cues.
This module generates a small set of complete outfit candidates and ranks them
with a transparent linear model.  It deliberately has no fitted parameters or
network dependency, so it is suitable for a dissertation prototype and can be
evaluated independently from the much larger vision model.
"""

from itertools import product

from src.trained_recommender import recommend_with_trained_model

SLOTS = ("inner_top", "outer_top", "bottom", "shoes")

SLOT_LABELS = {
    "inner_top": "Inner Top",
    "outer_top": "Outer Layer",
    "bottom": "Bottom",
    "shoes": "Shoes",
}

CATEGORY_TO_SLOT = {
    "Tank Top": "inner_top",
    "T-Shirt": "inner_top",
    "Shirt": "inner_top",
    "Blouse": "inner_top",
    "Sweater": "inner_top",
    "Hoodie": "inner_top",
    "Top": "inner_top",
    "Jacket": "outer_top",
    "Blazer": "outer_top",
    "Coat": "outer_top",
    "Outerwear": "outer_top",
    "Jeans": "bottom",
    "Trousers": "bottom",
    "Shorts": "bottom",
    "Skirt": "bottom",
    "Bottom": "bottom",
    "Shoes": "shoes",
    "Footwear": "shoes",
    # One-piece and swimwear are treated as the outfit's central garment.
    # This preserves the four-slot UI while still returning three companions.
    "Dress": "inner_top",
    "One-piece": "inner_top",
    "Swimwear": "inner_top",
}

STYLE_ITEMS = {
    "Casual": {
        "inner_top": "Cotton T-Shirt",
        "outer_top": "Casual Jacket",
        "bottom": "Straight-leg Trousers",
        "shoes": "Clean Trainers",
    },
    "Outdoor": {
        "inner_top": "Breathable Base Layer",
        "outer_top": "Technical Shell Jacket",
        "bottom": "Utility Trousers",
        "shoes": "Trail Trainers",
    },
    "Sporty": {
        "inner_top": "Performance Top",
        "outer_top": "Track Jacket",
        "bottom": "Tapered Joggers",
        "shoes": "Running Trainers",
    },
    "Formal": {
        "inner_top": "Crisp Shirt",
        "outer_top": "Tailored Blazer",
        "bottom": "Tailored Trousers",
        "shoes": "Leather Shoes",
    },
    "Business": {
        "inner_top": "Smart Shirt",
        "outer_top": "Structured Blazer",
        "bottom": "Smart Trousers",
        "shoes": "Loafers",
    },
    "Streetwear": {
        "inner_top": "Relaxed T-Shirt",
        "outer_top": "Oversized Jacket",
        "bottom": "Cargo Trousers",
        "shoes": "Chunky Trainers",
    },
    "Minimalist": {
        "inner_top": "Simple Top",
        "outer_top": "Clean-cut Jacket",
        "bottom": "Straight Trousers",
        "shoes": "Minimal Trainers",
    },
    "Party": {
        "inner_top": "Statement Top",
        "outer_top": "Cropped Jacket",
        "bottom": "Dress Trousers",
        "shoes": "Dress Shoes",
    },
    "Beachwear": {
        "inner_top": "Lightweight Top",
        "outer_top": "Linen Overshirt",
        "bottom": "Relaxed Shorts",
        "shoes": "Sandals",
    },
}

COLOUR_PALETTES = {
    "Black": ["White", "Grey", "Black"],
    "White": ["Blue", "Beige", "White"],
    "Grey": ["White", "Black", "White"],
    "Blue": ["White", "Grey", "White"],
    "Red": ["Black", "Grey", "White"],
    "Green": ["White", "Beige", "Brown"],
    "Brown": ["Cream", "Blue", "Brown"],
    "Beige": ["White", "Brown", "White"],
    "Purple": ["White", "Grey", "Black"],
    "Pink": ["White", "Grey", "White"],
    "Orange": ["Cream", "Brown", "White"],
    "Yellow": ["White", "Blue", "White"],
}

# The second outfit deliberately keeps the selected style but explores a
# different compatible palette. This makes the comparison useful without
# forcing the user to evaluate two unrelated aesthetics.
ALTERNATIVE_COLOUR_PALETTES = {
    "Black": ["Cream", "Blue", "White"],
    "White": ["Grey", "Black", "Brown"],
    "Grey": ["Blue", "Cream", "Black"],
    "Blue": ["Cream", "Brown", "Grey"],
    "Red": ["Cream", "Black", "Grey"],
    "Green": ["Cream", "Navy", "White"],
    "Brown": ["White", "Olive", "Cream"],
    "Beige": ["Blue", "White", "Brown"],
    "Purple": ["Cream", "Black", "Grey"],
    "Pink": ["Cream", "Blue", "Grey"],
    "Orange": ["White", "Navy", "Brown"],
    "Yellow": ["Grey", "Navy", "White"],
}


NEUTRALS = {"Black", "White", "Grey", "Navy", "Brown", "Beige", "Cream"}
LIGHT_COLOURS = {"White", "Beige", "Cream", "Pink", "Yellow"}
DARK_COLOURS = {"Black", "Navy", "Brown", "Olive", "Purple"}

COLOUR_FAMILIES = {
    "Black": "neutral", "White": "neutral", "Grey": "neutral",
    "Beige": "neutral", "Cream": "neutral", "Brown": "earth",
    "Green": "earth", "Olive": "earth", "Blue": "cool", "Navy": "cool",
    "Purple": "cool", "Red": "warm", "Pink": "warm", "Orange": "warm",
    "Yellow": "warm",
}

STYLE_COLOURS = {
    "Casual": {"White", "Grey", "Blue", "Navy", "Beige"},
    "Outdoor": {"Black", "Navy", "Green", "Olive", "Brown"},
    "Sporty": {"Black", "White", "Grey", "Blue", "Red"},
    "Formal": {"Black", "White", "Grey", "Navy", "Cream"},
    "Business": {"Black", "White", "Grey", "Navy", "Beige", "Brown"},
    "Streetwear": {"Black", "White", "Grey", "Red", "Blue"},
    "Minimalist": {"Black", "White", "Grey", "Beige", "Cream"},
    "Party": {"Black", "White", "Red", "Purple", "Pink"},
    "Beachwear": {"White", "Cream", "Blue", "Green", "Orange", "Yellow"},
}

SLOT_COLOURS = {
    "inner_top": {"White", "Black", "Grey", "Cream", "Blue"},
    "outer_top": {"Black", "Grey", "Navy", "Brown", "Olive"},
    "bottom": {"Black", "Grey", "Navy", "Blue", "Brown", "Beige"},
    "shoes": {"Black", "White", "Grey", "Brown", "Cream"},
}


class LightweightOutfitRanker:
    """Content-based linear ranker with human-readable feature weights."""

    FEATURE_WEIGHTS = {
        "input_colour_harmony": 0.31,
        "palette_coherence": 0.18,
        "style_match": 0.20,
        "weather_comfort": 0.16,
        "slot_versatility": 0.10,
        "colour_variety": 0.05,
    }

    def _colour_harmony(self, first, second):
        if first == second:
            return 0.78
        if first in NEUTRALS or second in NEUTRALS:
            return 0.96
        first_family = COLOUR_FAMILIES.get(first, "neutral")
        second_family = COLOUR_FAMILIES.get(second, "neutral")
        if first_family == second_family:
            return 0.82
        if {first_family, second_family} == {"earth", "warm"}:
            return 0.76
        if {first_family, second_family} == {"cool", "warm"}:
            return 0.70
        return 0.74

    @staticmethod
    def _weather_comfort(colours, weather):
        if not weather:
            return 0.82
        feels_like = float(weather.get("feels_like", 20.0))
        preferred = LIGHT_COLOURS if feels_like >= 24 else DARK_COLOURS
        if 12 < feels_like < 24:
            return 0.90
        return sum(1.0 if colour in preferred else 0.68 for colour in colours) / len(colours)

    def features(self, input_colour, style, slots, colours, weather):
        input_harmony = sum(
            self._colour_harmony(input_colour, colour) for colour in colours
        ) / len(colours)
        pairs = tuple(
            self._colour_harmony(colours[left], colours[right])
            for left in range(len(colours))
            for right in range(left + 1, len(colours))
        )
        style_colours = STYLE_COLOURS.get(style, STYLE_COLOURS["Casual"])
        style_match = sum(colour in style_colours for colour in colours) / len(colours)
        versatility = sum(
            colour in SLOT_COLOURS[slot]
            for slot, colour in zip(slots, colours)
        ) / len(colours)
        # Two or three colours are normally more useful than a monochrome set.
        unique_colours = len(set(colours))
        variety = 1.0 if unique_colours in (2, 3) else 0.60
        return {
            "input_colour_harmony": input_harmony,
            "palette_coherence": sum(pairs) / len(pairs),
            "style_match": style_match,
            "weather_comfort": self._weather_comfort(colours, weather),
            "slot_versatility": versatility,
            "colour_variety": variety,
        }

    def score(self, input_colour, style, slots, colours, weather):
        components = self.features(input_colour, style, slots, colours, weather)
        score = sum(
            components[name] * weight
            for name, weight in self.FEATURE_WEIGHTS.items()
        )
        return round(score * 100, 1), {
            name: round(value * 100, 1) for name, value in components.items()
        }

    def rank(self, input_colour, style, slots, weather):
        suggested = COLOUR_PALETTES.get(input_colour, ["Cream", "Blue", "White"])
        alternative = ALTERNATIVE_COLOUR_PALETTES.get(
            input_colour, ["Grey", "Navy", "Brown"]
        )
        # A compact per-request catalogue keeps the exhaustive search below
        # 350 combinations while retaining neutral and authored alternatives.
        colour_pool = tuple(dict.fromkeys(
            suggested + alternative + ["Black", "White", "Grey", "Navy", "Beige", "Cream"]
        ))
        ranked = []
        for colours in product(colour_pool, repeat=len(slots)):
            score, components = self.score(
                input_colour, style, slots, colours, weather
            )
            ranked.append((score, colours, components))
        ranked.sort(key=lambda candidate: (-candidate[0], candidate[1]))
        return ranked


RANKER = LightweightOutfitRanker()


def _weather_adjustment(slot, item_name, weather):
    if not weather:
        return item_name, []

    feels_like = float(weather.get("feels_like", 20.0))
    rain_probability = float(weather.get("rain_probability", 0.0))
    condition = weather.get("condition", "Unknown")
    constraints = []

    if feels_like <= 5:
        constraints.append("very_cold")
        if slot == "inner_top":
            item_name = "Thermal Base Layer"
        elif slot == "outer_top":
            item_name = "Insulated Coat"
        elif slot == "shoes":
            item_name = "Warm Boots"
    elif feels_like <= 12:
        constraints.append("cold")
        if slot == "outer_top":
            item_name = "Warm Jacket"
    elif feels_like >= 28:
        constraints.append("hot")
        if slot == "inner_top":
            item_name = "Breathable Tank Top"
        elif slot == "outer_top":
            item_name = "TOO HOT - No Outer Layer Needed"
        elif slot == "bottom":
            item_name = "Breathable Shorts"
    elif feels_like >= 24:
        constraints.append("warm")
        if slot == "inner_top":
            item_name = "Breathable Lightweight Top"
        elif slot == "outer_top":
            item_name = "Lightweight Overshirt"
        elif slot == "bottom":
            item_name = "Lightweight Trousers"

    if rain_probability >= 50 or condition in {"Rain", "Thunderstorm"}:
        constraints.append("rain")
        if slot == "outer_top":
            item_name = "Waterproof Jacket"
        elif slot == "shoes":
            item_name = "Water-resistant Shoes"

    if condition == "Snow":
        constraints.append("snow")
        if slot == "outer_top":
            item_name = "Insulated Coat"
        elif slot == "shoes":
            item_name = "Winter Boots"

    return item_name, list(dict.fromkeys(constraints))


def _build_outfit(category, colour, style, weather, ranked_candidate):
    input_slot = CATEGORY_TO_SLOT.get(category, "inner_top")
    style_items = STYLE_ITEMS.get(style, STYLE_ITEMS["Casual"])
    recommendation_slots = [slot for slot in SLOTS if slot != input_slot]
    outfit_score, palette, score_components = ranked_candidate
    items = []
    constraints = []

    for slot, item_colour in zip(recommendation_slots, palette):
        item_name, item_constraints = _weather_adjustment(
            slot,
            style_items[slot],
            weather,
        )
        constraints.extend(item_constraints)
        recommends_no_item = "No Outer Layer Needed" in item_name
        if recommends_no_item:
            item_colour = None
        items.append(
            {
                "slot": slot,
                "slot_label": SLOT_LABELS[slot],
                "type": item_name,
                "colour": item_colour,
                "label": (
                    item_name
                    if recommends_no_item
                    else f"{item_colour} {item_name}"
                ),
            }
        )

    return {
        "style": style,
        "input_slot": input_slot,
        "items": items,
        "constraints_applied": list(dict.fromkeys(constraints)),
        "model_score": outfit_score,
        "score_components": score_components,
    }


def recommend_outfit(
    category,
    colour,
    styles=None,
    weather=None,
    selected_style=None,
    input_embedding=None,
):
    """Use trained compatibility ranking when possible, else the baseline."""

    recognised_styles = list(dict.fromkeys(styles or ["Casual"]))
    primary_style = selected_style or recognised_styles[0]

    input_slot = CATEGORY_TO_SLOT.get(category, "inner_top")
    fallback_reason = "trained artifacts or input embedding unavailable"
    try:
        trained_result = recommend_with_trained_model(
            input_slot=input_slot,
            input_category=category,
            input_colour=colour,
            input_embedding=input_embedding,
            style=primary_style,
            weather=weather,
            slot_labels=SLOT_LABELS,
        )
        if trained_result is not None:
            trained_result.update(
                {
                    "available_styles": recognised_styles,
                    "input_category": category,
                    "input_colour": colour,
                    "weather": weather,
                }
            )
            return trained_result
    except Exception as error:
        # Recommendation must remain available when a local checkpoint or
        # product catalogue is incomplete. Developer metadata records why.
        fallback_reason = f"{type(error).__name__}: {error}"

    recommendation_slots = [slot for slot in SLOTS if slot != input_slot]
    ranked_candidates = RANKER.rank(
        colour, primary_style, recommendation_slots, weather
    )
    primary_candidate = ranked_candidates[0]
    # Make the second result meaningfully different, not merely a one-colour
    # permutation of the highest-ranked outfit.
    alternative_candidate = next(
        candidate
        for candidate in ranked_candidates[1:]
        if sum(
            left != right
            for left, right in zip(primary_candidate[1], candidate[1])
        ) >= 2
    )

    primary = _build_outfit(
        category, colour, primary_style, weather, primary_candidate
    )
    alternative = _build_outfit(
        category, colour, primary_style, weather, alternative_candidate
    )

    return {
        "items": [item["label"] for item in primary["items"]],
        "primary": primary,
        "alternative": alternative,
        "available_styles": recognised_styles,
        "method": "lightweight_content_ranker_v1",
        "model": {
            "type": "interpretable_linear_ranker",
            "feature_weights": dict(RANKER.FEATURE_WEIGHTS),
            "candidates_scored": len(ranked_candidates),
            "fallback_reason": fallback_reason,
        },
        "input_category": category,
        "input_colour": colour,
        "weather": weather,
        "explanation": (
            "A lightweight linear model ranks complete outfits using colour "
            "harmony, palette coherence, style, weather and slot versatility."
        ),
    }
