import unittest

from recommendation_training.prepare_zara_catalogue import infer_slot


class ZaraCataloguePreparationTests(unittest.TestCase):
    def test_infers_each_supported_slot_from_product_name(self):
        self.assertEqual(infer_slot("Linen Shirt", ""), "inner_top")
        self.assertEqual(infer_slot("Water-Resistant Puffer Jacket", ""), "outer_top")
        self.assertEqual(infer_slot("Flared Pants", ""), "bottom")
        self.assertEqual(infer_slot("Denim Casual Sneakers", ""), "shoes")
        self.assertEqual(infer_slot("Quilted Nylon High Tops", ""), "shoes")

    def test_description_cannot_override_product_name(self):
        self.assertEqual(
            infer_slot("Waterproof Flared Pants", "Designed to wear with ski boots"),
            "bottom",
        )

    def test_rejects_accessories_and_one_piece_garments(self):
        self.assertIsNone(infer_slot("Rubberized Belt Bag With Pockets", ""))
        self.assertIsNone(infer_slot("Rib T-Shirt Dress", ""))
        self.assertIsNone(infer_slot("Medicine Ball", ""))


if __name__ == "__main__":
    unittest.main()
