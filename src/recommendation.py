"""
General rule-based outfit recommendation baseline.

The algorithm separates category compatibility from colour harmony.
This avoids creating a separate rule for every category-colour pair.
"""


CATEGORY_ALIASES = {
    "t-shirt": "T-Shirt",
    "tshirt": "T-Shirt",
    "tee": "T-Shirt",
    "shirt": "Shirt",
    "sweater": "Sweater",
    "hoodie": "Hoodie",
    "jacket": "Jacket",
    "coat": "Coat",
    "jeans": "Jeans",
    "trousers": "Trousers",
    "pants": "Trousers",
    "shorts": "Shorts",
    "skirt": "Skirt",
    "dress": "Dress",
    "shoes": "Shoes",
}


COLOUR_ALIASES = {
    "black": "Black",
    "white": "White",
    "grey": "Grey",
    "gray": "Grey",
    "blue": "Blue",
    "red": "Red",
    "green": "Green",
    "yellow": "Yellow",
    "orange": "Orange",
    "purple": "Purple",
    "pink": "Pink",
    "brown": "Brown",
    "beige": "Beige",
}


# Category rules determine which item types complete the outfit.
CATEGORY_RULES = {
    "T-Shirt": ["Trousers", "Jacket", "Shoes"],
    "Shirt": ["Trousers", "Jacket", "Shoes"],
    "Sweater": ["Trousers", "Coat", "Shoes"],
    "Hoodie": ["Trousers", "Jacket", "Shoes"],
    "Jeans": ["T-Shirt", "Jacket", "Shoes"],
    "Trousers": ["Shirt", "Jacket", "Shoes"],
    "Shorts": ["T-Shirt", "Jacket", "Shoes"],
    "Skirt": ["Shirt", "Jacket", "Shoes"],
    "Dress": ["Jacket", "Shoes", "Accessories"],
    "Jacket": ["T-Shirt", "Trousers", "Shoes"],
    "Coat": ["Sweater", "Trousers", "Shoes"],
    "Shoes": ["T-Shirt", "Trousers", "Jacket"],
}


# Colour rules are independent from clothing categories.
# Each list supplies colours for the three recommended item types.
COLOUR_PALETTES = {
    "Black": ["White", "Grey", "Black"],
    "White": ["Blue", "Beige", "White"],
    "Grey": ["White", "Black", "White"],
    "Blue": ["White", "Grey", "White"],
    "Red": ["Black", "Grey", "White"],
    "Green": ["White", "Beige", "Brown"],
    "Yellow": ["White", "Blue", "White"],
    "Orange": ["White", "Brown", "Beige"],
    "Purple": ["White", "Grey", "Black"],
    "Pink": ["White", "Blue", "Beige"],
    "Brown": ["White", "Blue", "Brown"],
    "Beige": ["White", "Brown", "Beige"],
}


DEFAULT_ITEM_TYPES = ["Top", "Jacket", "Shoes"]
DEFAULT_COLOURS = ["White", "Grey", "Black"]


def normalise_category(category):
    """Convert a category label into the standard system format."""

    if not isinstance(category, str):
        return "Unknown"

    cleaned = category.strip().casefold()
    return CATEGORY_ALIASES.get(cleaned, category.strip().title())


def normalise_colour(colour):
    """Convert a colour label into the standard system format."""

    if not isinstance(colour, str):
        return "Unknown"

    cleaned = colour.strip().casefold()
    return COLOUR_ALIASES.get(cleaned, colour.strip().title())


def recommend_outfit(category, colour):
    """
    Recommend an outfit using separate category and colour rules.

    Parameters:
        category (str): Recognised clothing category.
        colour (str): Automatically detected clothing colour.

    Returns:
        dict: Recommended items, scores and explanation.
    """

    category = normalise_category(category)
    colour = normalise_colour(colour)

    item_types = CATEGORY_RULES.get(category, DEFAULT_ITEM_TYPES)
    recommended_colours = COLOUR_PALETTES.get(colour, DEFAULT_COLOURS)

    items = [
        f"{item_colour} {item_type}"
        for item_colour, item_type in zip(
            recommended_colours,
            item_types,
        )
    ]

    return {
        "items": items,
        "scores": [1.0, 0.90, 0.85],
        "method": "rule_based_baseline",
        "input_category": category,
        "input_colour": colour,
        "explanation": (
            f"The baseline first uses the {category} category to select "
            f"complementary item types. It then uses the detected {colour} "
            "colour to select a compatible colour palette."
        ),
        "constraints_applied": [],
    }