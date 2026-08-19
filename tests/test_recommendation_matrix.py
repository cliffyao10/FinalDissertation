import unittest

from src.garment_taxonomy import broad_garment_category
from src.recommendation import (
    CATEGORY_TO_SLOT,
    COLOUR_PALETTES,
    SLOT_LABELS,
    STYLE_ITEMS,
    recommend_outfit,
)


WEATHER_SCENARIOS = (
    None,
    {"feels_like": -4, "rain_probability": 0, "condition": "Clear"},
    {"feels_like": 1, "rain_probability": 40, "condition": "Snow"},
    {"feels_like": 14, "rain_probability": 85, "condition": "Rain"},
    {"feels_like": 25, "rain_probability": 0, "condition": "Clear"},
    {"feels_like": 31, "rain_probability": 0, "condition": "Clear"},
)


class RecommendationMatrixTests(unittest.TestCase):
    def assert_valid_result(self, result, category):
        self.assertIn("primary", result)
        self.assertIn("alternative", result)
        input_slot = CATEGORY_TO_SLOT[category]
        expected_slots = set(SLOT_LABELS) - {input_slot}

        for outfit_name in ("primary", "alternative"):
            outfit = result[outfit_name]
            self.assertEqual(len(outfit["items"]), 3)
            self.assertEqual(
                {item["slot"] for item in outfit["items"]}, expected_slots
            )
            for item in outfit["items"]:
                self.assertEqual(item["slot_label"], SLOT_LABELS[item["slot"]])
                generic = broad_garment_category(
                    item["slot"], item.get("type", "")
                )
                self.assertNotEqual(generic, "Garment")
                if item.get("colour") is None:
                    self.assertEqual(item["slot"], "outer_top")
                    self.assertEqual(generic, "No Outer Layer")

            for slot, candidates in outfit["alternatives_by_slot"].items():
                current = next(item for item in outfit["items"] if item["slot"] == slot)
                seen_colours = set()
                for candidate in candidates:
                    self.assertNotEqual(candidate.get("colour"), current.get("colour"))
                    self.assertNotIn(candidate.get("colour"), seen_colours)
                    seen_colours.add(candidate.get("colour"))

    def test_every_supported_category_and_colour(self):
        for category in CATEGORY_TO_SLOT:
            for colour in COLOUR_PALETTES:
                with self.subTest(category=category, colour=colour):
                    result = recommend_outfit(
                        category,
                        colour,
                        styles=["Casual"],
                        weather=None,
                    )
                    self.assert_valid_result(result, category)

    def test_all_slots_styles_and_weather_extremes(self):
        category_for_slot = {
            "inner_top": "T-Shirt",
            "outer_top": "Jacket",
            "bottom": "Trousers",
            "shoes": "Shoes",
        }
        for category in category_for_slot.values():
            for style in STYLE_ITEMS:
                for weather in WEATHER_SCENARIOS:
                    with self.subTest(
                        category=category, style=style, weather=weather
                    ):
                        result = recommend_outfit(
                            category,
                            "Red",
                            styles=[style],
                            weather=weather,
                        )
                        self.assert_valid_result(result, category)


if __name__ == "__main__":
    unittest.main()
