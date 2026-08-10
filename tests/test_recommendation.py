import unittest

from src.recommendation import recommend_outfit


class RecommendationModelTests(unittest.TestCase):
    def test_returns_two_ranked_complete_outfits(self):
        result = recommend_outfit("T-Shirt", "Blue", styles=["Casual"])

        self.assertEqual(result["method"], "lightweight_content_ranker_v1")
        self.assertEqual(len(result["primary"]["items"]), 3)
        self.assertEqual(len(result["alternative"]["items"]), 3)
        self.assertGreaterEqual(
            result["primary"]["model_score"],
            result["alternative"]["model_score"],
        )
        self.assertEqual(
            set(result["primary"]["score_components"]),
            set(result["model"]["feature_weights"]),
        )

    def test_alternative_changes_at_least_two_colours(self):
        result = recommend_outfit("Jacket", "Red", selected_style="Streetwear")
        primary = [item["colour"] for item in result["primary"]["items"]]
        alternative = [item["colour"] for item in result["alternative"]["items"]]

        self.assertGreaterEqual(
            sum(left != right for left, right in zip(primary, alternative)), 2
        )

    def test_weather_constraints_change_item_types(self):
        result = recommend_outfit(
            "Trousers",
            "Black",
            selected_style="Outdoor",
            weather={
                "feels_like": 2,
                "rain_probability": 80,
                "condition": "Rain",
            },
        )
        types = {item["slot"]: item["type"] for item in result["primary"]["items"]}

        self.assertEqual(types["outer_top"], "Waterproof Jacket")
        self.assertEqual(types["shoes"], "Water-resistant Shoes")
        self.assertIn("very_cold", result["primary"]["constraints_applied"])
        self.assertIn("rain", result["primary"]["constraints_applied"])

    def test_model_is_deterministic(self):
        arguments = ("Dress", "Purple")
        first = recommend_outfit(*arguments, selected_style="Party")
        second = recommend_outfit(*arguments, selected_style="Party")

        self.assertEqual(first["primary"], second["primary"])
        self.assertEqual(first["alternative"], second["alternative"])


if __name__ == "__main__":
    unittest.main()
