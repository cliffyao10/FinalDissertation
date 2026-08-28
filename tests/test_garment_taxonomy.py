import unittest

from src.garment_taxonomy import broad_garment_category


class GarmentTaxonomyTests(unittest.TestCase):
    def test_brand_and_collection_copy_is_removed(self):
        self.assertEqual(
            broad_garment_category(
                "outer_top", "Recco Locator Ski Jacket Ski Collection"
            ),
            "Jacket",
        )
        self.assertEqual(
            broad_garment_category("bottom", "Plush Jogging Pants Collection"),
            "Joggers",
        )

    def test_keywords_cannot_escape_their_slot(self):
        self.assertEqual(
            broad_garment_category("outer_top", "Topstitched Leather Jacket"),
            "Jacket",
        )
        self.assertEqual(
            broad_garment_category("shoes", "Leather Loafers"),
            "Loafers",
        )

    def test_no_outer_layer_state_is_preserved(self):
        self.assertEqual(
            broad_garment_category(
                "outer_top", "TOO HOT - No Outer Layer Needed"
            ),
            "No Outer Layer",
        )

    def test_t_shirt_is_not_collapsed_into_shirt(self):
        self.assertEqual(
            broad_garment_category("inner_top", "T-Shirt"),
            "T-Shirt",
        )


if __name__ == "__main__":
    unittest.main()
