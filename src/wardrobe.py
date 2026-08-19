"""Local, private wardrobe storage and lightweight garment image cleanup."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageFilter


SLOTS = ("outer_top", "inner_top", "bottom", "shoes")
SLOT_LABELS = {
    "inner_top": "Inner top",
    "outer_top": "Outer layer",
    "bottom": "Bottom",
    "shoes": "Shoes",
}

DEFAULT_STORE_PATH = Path("data/wardrobe.json")
DEFAULT_IMAGE_DIR = Path("data/wardrobe_images")
DEFAULT_OUTFIT_GROUPS = ("Everyday", "Work", "Occasion", "Travel")


def filter_pieces_by_slot(pieces, selected_slot):
    """Return all pieces or only one supported wardrobe category."""
    if selected_slot in {None, "All"}:
        return list(pieces)
    if selected_slot not in SLOTS:
        raise ValueError(f"Unsupported wardrobe slot: {selected_slot}")
    return [piece for piece in pieces if piece.get("slot") == selected_slot]


def empty_store():
    return {
        "version": 2,
        "pieces": [],
        "outfits": [],
        "outfit_groups": list(DEFAULT_OUTFIT_GROUPS),
    }


def load_store(path=DEFAULT_STORE_PATH):
    path = Path(path)
    if not path.is_file():
        return empty_store()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_store()
    if not isinstance(payload, dict):
        return empty_store()
    payload["version"] = 2
    payload.setdefault("pieces", [])
    payload.setdefault("outfits", [])
    stored_groups = payload.get("outfit_groups", [])
    payload["outfit_groups"] = list(
        dict.fromkeys([*DEFAULT_OUTFIT_GROUPS, *stored_groups])
    )
    for outfit in payload["outfits"]:
        outfit.setdefault("group", "Everyday")
    return payload


def save_store(store, path=DEFAULT_STORE_PATH):
    """Atomically save the user's local wardrobe."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(store, indent=2), encoding="utf-8")
    temporary.replace(path)


def new_piece(slot, item_type, colour, image_path=None, source="user"):
    if slot not in SLOTS:
        raise ValueError(f"Unsupported wardrobe slot: {slot}")
    return {
        "id": uuid4().hex,
        "slot": slot,
        "type": str(item_type or SLOT_LABELS[slot]),
        "colour": str(colour or ""),
        "image_path": str(image_path or ""),
        "display_mode": "image" if image_path else "icon",
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def new_outfit(
    slots=None,
    mood="Green",
    style="Casual",
    title="Saved look",
    group="Everyday",
):
    clean_slots = {slot: None for slot in SLOTS}
    for slot, piece in (slots or {}).items():
        if slot in clean_slots:
            clean_slots[slot] = piece
    return {
        "id": uuid4().hex,
        "title": str(title),
        "mood": str(mood),
        "style": str(style),
        "group": str(group or "Everyday"),
        "slots": clean_slots,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def add_outfit_group(store, name):
    """Add a user-defined outfit group and return its normalised name."""

    clean_name = " ".join(str(name or "").split())
    if not clean_name:
        raise ValueError("Group name cannot be empty.")
    groups = store.setdefault("outfit_groups", list(DEFAULT_OUTFIT_GROUPS))
    existing = next(
        (group for group in groups if group.casefold() == clean_name.casefold()),
        None,
    )
    if existing:
        return existing
    groups.append(clean_name)
    return clean_name


def remove_outfit_group(store, name):
    """Remove a custom group and move its outfits back to Everyday."""

    if name in DEFAULT_OUTFIT_GROUPS:
        raise ValueError("Default outfit groups cannot be removed.")
    store["outfit_groups"] = [
        group for group in store.get("outfit_groups", []) if group != name
    ]
    for outfit in store.get("outfits", []):
        if outfit.get("group") == name:
            outfit["group"] = "Everyday"


def update_outfit_details(store, outfit_id, *, title=None, group=None):
    """Update the user-facing name and group of a saved outfit."""

    outfit = find_outfit(store, outfit_id)
    if outfit is None:
        raise KeyError(f"Unknown outfit: {outfit_id}")
    if title is not None:
        clean_title = " ".join(str(title).split())
        outfit["title"] = clean_title or "Untitled look"
    if group is not None:
        outfit["group"] = add_outfit_group(store, group)
    return outfit


def find_outfit(store, outfit_id):
    return next(
        (outfit for outfit in store.get("outfits", []) if outfit.get("id") == outfit_id),
        None,
    )


def pieces_for_slot(store, slot):
    """Return reusable pieces from the library and every saved outfit."""

    if slot not in SLOTS:
        raise ValueError(f"Unsupported wardrobe slot: {slot}")
    reusable, seen = [], set()
    candidates = list(store.get("pieces", []))
    for outfit in store.get("outfits", []):
        piece = outfit.get("slots", {}).get(slot)
        if piece:
            candidates.append(piece)
    for piece in candidates:
        if not piece or piece.get("slot") != slot:
            continue
        identity = (
            str(piece.get("id", "")),
            str(piece.get("image_path", "")),
            str(piece.get("type", "")),
            str(piece.get("colour", "")),
        )
        if identity in seen:
            continue
        seen.add(identity)
        reusable.append(piece)
    return reusable


def set_outfit_slot(store, outfit_id, slot, piece):
    if slot not in SLOTS:
        raise ValueError(f"Unsupported wardrobe slot: {slot}")
    outfit = find_outfit(store, outfit_id)
    if outfit is None:
        raise KeyError(f"Unknown outfit: {outfit_id}")
    outfit.setdefault("slots", {name: None for name in SLOTS})[slot] = piece


def remove_outfit_slot(store, outfit_id, slot):
    set_outfit_slot(store, outfit_id, slot, None)


def remove_outfit(store, outfit_id):
    store["outfits"] = [
        outfit for outfit in store.get("outfits", []) if outfit.get("id") != outfit_id
    ]


def remove_piece(store, piece_id):
    store["pieces"] = [
        piece for piece in store.get("pieces", []) if piece.get("id") != piece_id
    ]


def update_piece_name(store, piece_id, name):
    """Rename a library piece and every saved outfit reference to it."""

    clean_name = " ".join(str(name or "").split())
    if not clean_name:
        raise ValueError("Piece name cannot be empty.")

    updated = False
    for piece in store.get("pieces", []):
        if piece.get("id") == piece_id:
            piece["type"] = clean_name
            updated = True
    for outfit in store.get("outfits", []):
        for piece in outfit.get("slots", {}).values():
            if piece and piece.get("id") == piece_id:
                piece["type"] = clean_name
                updated = True
    if not updated:
        raise KeyError(f"Unknown piece: {piece_id}")
    return clean_name


def prune_unused_images(store, directory=DEFAULT_IMAGE_DIR):
    """Delete local wardrobe PNGs no longer referenced by a piece or outfit."""

    directory = Path(directory)
    if not directory.is_dir():
        return []
    referenced = set()
    for piece in store.get("pieces", []):
        if piece and piece.get("image_path"):
            referenced.add(Path(piece["image_path"]).resolve())
    for outfit in store.get("outfits", []):
        for piece in outfit.get("slots", {}).values():
            if piece and piece.get("image_path"):
                referenced.add(Path(piece["image_path"]).resolve())
    removed = []
    for path in directory.glob("*.png"):
        if path.resolve() not in referenced:
            path.unlink()
            removed.append(path)
    return removed


def lightweight_background_cleanup(image):
    """Remove a sufficiently uniform edge background without a heavy ML model.

    Busy lifestyle backgrounds are retained because an automatic colour-key
    mask would be unreliable. A lasso crop still gives those images a clean,
    focused wardrobe preview.
    """

    image = image.convert("RGB")
    array = np.asarray(image).astype(np.float32)
    if min(array.shape[:2]) < 8:
        return image.convert("RGBA")
    edge = np.concatenate(
        (array[0], array[-1], array[:, 0], array[:, -1]),
        axis=0,
    )
    if float(edge.std(axis=0).mean()) > 34.0:
        return image.convert("RGBA")
    background = np.median(edge, axis=0)
    distance = np.linalg.norm(array - background, axis=2)
    alpha = np.clip((distance - 16.0) * 7.0, 0, 255).astype(np.uint8)
    alpha_image = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(1.2))
    result = image.convert("RGBA")
    result.putalpha(alpha_image)
    return result


def save_piece_image(image, directory=DEFAULT_IMAGE_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid4().hex}.png"
    lightweight_background_cleanup(image).save(path, format="PNG", optimize=True)
    return path.as_posix()
