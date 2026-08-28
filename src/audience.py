"""Garment-presentation labels used for safe candidate filtering.

The application asks which clothing range the user wants to browse.  It does
not infer a person's gender from an image.  Polyvore exposes explicit men's
category branches; supported garment categories outside that branch belong to
the dataset's default womenswear branch.
"""

MENSWEAR_CATEGORY_IDS = frozenset(
    {
        272, 273, 275, 277, 278, 279, 280, 281, 282, 286, 287, 288, 289,
        291, 292, 293, 294, 296, 297, 298, 341, 342, 343, 4454, 4455,
        4456, 4457, 4458, 4459, 4465, 4497, 4498, 4522,
    }
)

AUDIENCES = ("menswear", "womenswear")


def normalise_audience(value, default="womenswear"):
    label = str(value or "").strip().casefold().replace(" ", "")
    aliases = {
        "man": "menswear",
        "men": "menswear",
        "male": "menswear",
        "menswear": "menswear",
        "woman": "womenswear",
        "women": "womenswear",
        "female": "womenswear",
        "womenswear": "womenswear",
    }
    return aliases.get(label, default)


def audience_for_category_id(category_id):
    return "menswear" if int(category_id) in MENSWEAR_CATEGORY_IDS else "womenswear"


def item_matches_audience(item, audience):
    item_audience = str(item.get("audience", "neutral")).casefold()
    return item_audience in {"neutral", "all", normalise_audience(audience)}
