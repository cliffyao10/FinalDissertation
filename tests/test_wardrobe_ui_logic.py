import unittest

from src.wardrobe import filter_pieces_by_slot


class WardrobeUiLogicTests(unittest.TestCase):
    def setUp(self):
        self.pieces = [
            {"id": "outer", "slot": "outer_top"},
            {"id": "inner", "slot": "inner_top"},
            {"id": "bottom", "slot": "bottom"},
            {"id": "shoes", "slot": "shoes"},
        ]

    def test_all_keeps_every_piece(self):
        self.assertEqual(filter_pieces_by_slot(self.pieces, "All"), self.pieces)

    def test_each_supported_slot_filters_independently(self):
        for piece in self.pieces:
            with self.subTest(slot=piece["slot"]):
                self.assertEqual(
                    filter_pieces_by_slot(self.pieces, piece["slot"]),
                    [piece],
                )

    def test_unknown_slot_is_rejected(self):
        with self.assertRaises(ValueError):
            filter_pieces_by_slot(self.pieces, "accessories")


if __name__ == "__main__":
    unittest.main()
