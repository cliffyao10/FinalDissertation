import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.wardrobe import (
    DEFAULT_OUTFIT_GROUPS,
    add_outfit_group,
    load_store,
    lightweight_background_cleanup,
    new_outfit,
    new_piece,
    pieces_for_slot,
    prune_unused_images,
    remove_outfit_group,
    remove_outfit_slot,
    save_store,
    set_outfit_slot,
    update_outfit_details,
    update_piece_name,
)


class WardrobeTests(unittest.TestCase):
    def test_store_round_trip_and_partial_outfit_editing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wardrobe.json"
            piece = new_piece("inner_top", "T-Shirt", "Blue", "shirt.png")
            outfit = new_outfit({"inner_top": piece}, mood="Blue")
            store = {"version": 1, "pieces": [piece], "outfits": [outfit]}
            save_store(store, path)

            loaded = load_store(path)
            shoes = new_piece("shoes", "Trainers", "White")
            set_outfit_slot(loaded, outfit["id"], "shoes", shoes)
            self.assertEqual(loaded["outfits"][0]["slots"]["shoes"]["type"], "Trainers")
            remove_outfit_slot(loaded, outfit["id"], "shoes")
            self.assertIsNone(loaded["outfits"][0]["slots"]["shoes"])

    def test_uniform_background_becomes_transparent(self):
        array = np.full((40, 40, 3), 245, dtype=np.uint8)
        array[10:30, 12:28] = [30, 80, 140]
        cleaned = lightweight_background_cleanup(Image.fromarray(array))
        alpha = np.asarray(cleaned.getchannel("A"))
        self.assertLess(int(alpha[0, 0]), 10)
        self.assertGreater(int(alpha[20, 20]), 240)

    def test_outfit_pieces_are_reusable_in_other_outfits(self):
        shared = new_piece("inner_top", "T-Shirt", "Red", "shirt.png")
        store = {
            "pieces": [],
            "outfits": [new_outfit({"inner_top": shared})],
        }

        reusable = pieces_for_slot(store, "inner_top")

        self.assertEqual([piece["id"] for piece in reusable], [shared["id"]])

    def test_piece_rename_updates_saved_outfit_references(self):
        library_piece = new_piece("outer_top", "Jacket", "Black", "coat.png")
        outfit_copy = dict(library_piece)
        store = {
            "pieces": [library_piece],
            "outfits": [new_outfit({"outer_top": outfit_copy})],
        }

        update_piece_name(store, library_piece["id"], "  Snow day jacket ")

        self.assertEqual(store["pieces"][0]["type"], "Snow day jacket")
        self.assertEqual(
            store["outfits"][0]["slots"]["outer_top"]["type"],
            "Snow day jacket",
        )

    def test_outfit_names_and_custom_groups_persist(self):
        store = {
            "version": 2,
            "pieces": [],
            "outfits": [new_outfit(title="Untitled look")],
            "outfit_groups": list(DEFAULT_OUTFIT_GROUPS),
        }
        group = add_outfit_group(store, "  Summer   trip ")
        update_outfit_details(
            store,
            store["outfits"][0]["id"],
            title="  Lisbon evenings ",
            group=group,
        )

        self.assertEqual(group, "Summer trip")
        self.assertEqual(store["outfits"][0]["title"], "Lisbon evenings")
        self.assertEqual(store["outfits"][0]["group"], "Summer trip")

        remove_outfit_group(store, "Summer trip")
        self.assertEqual(store["outfits"][0]["group"], "Everyday")
        self.assertNotIn("Summer trip", store["outfit_groups"])

    def test_only_unreferenced_images_are_pruned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kept = root / "kept.png"
            removed = root / "removed.png"
            kept.touch()
            removed.touch()
            store = {
                "version": 1,
                "pieces": [new_piece("shoes", "Boots", "Black", kept)],
                "outfits": [],
            }

            deleted = prune_unused_images(store, root)

            self.assertTrue(kept.exists())
            self.assertFalse(removed.exists())
            self.assertEqual(deleted, [removed])


if __name__ == "__main__":
    unittest.main()
