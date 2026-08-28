"""Brand-free garment categories for user-facing recommendation output."""


def broad_garment_category(slot, item_type=""):
    """Return a broad category constrained by its outfit slot."""

    name = str(item_type or "").casefold()

    if slot == "outer_top":
        if "too hot" in name or "no outer" in name:
            return "No Outer Layer"
        if "coat" in name:
            return "Coat"
        if "blazer" in name:
            return "Blazer"
        if "cardigan" in name:
            return "Cardigan"
        if "overshirt" in name:
            return "Overshirt"
        return "Jacket"

    if slot == "inner_top":
        if "dress" in name:
            return "Dress"
        if any(word in name for word in ("tank", "vest", "cami")):
            return "Tank Top"
        if "hoodie" in name:
            return "Hoodie"
        if any(word in name for word in ("sweater", "jumper", "knit")):
            return "Sweater"
        if "blouse" in name:
            return "Blouse"
        if any(word in name for word in ("t-shirt", "tee")):
            return "T-Shirt"
        if "shirt" in name:
            return "Shirt"
        return "Top"

    if slot == "bottom":
        if "short" in name:
            return "Shorts"
        if "skirt" in name:
            return "Skirt"
        if any(word in name for word in ("jean", "denim")):
            return "Jeans"
        if any(word in name for word in ("jogger", "jogging", "sweatpant")):
            return "Joggers"
        return "Trousers"

    if slot == "shoes":
        if "boot" in name:
            return "Boots"
        if "sandal" in name:
            return "Sandals"
        if any(word in name for word in ("trainer", "sneaker", "running", "trail")):
            return "Trainers"
        if "loafer" in name:
            return "Loafers"
        if any(word in name for word in ("heel", "pump")):
            return "Heels"
        return "Shoes"

    return "Garment"
