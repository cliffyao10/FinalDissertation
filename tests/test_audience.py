import unittest

from src.audience import (
    audience_for_category_id,
    item_matches_audience,
    normalise_audience,
)


class AudienceTests(unittest.TestCase):
    def test_polyvore_menswear_branch_is_distinct(self):
        self.assertEqual(audience_for_category_id(272), "menswear")
        self.assertEqual(audience_for_category_id(7), "womenswear")

    def test_user_choice_is_clothing_range_not_inferred_identity(self):
        self.assertEqual(normalise_audience("Men"), "menswear")
        self.assertEqual(normalise_audience("Women"), "womenswear")
        self.assertTrue(
            item_matches_audience({"audience": "neutral"}, "menswear")
        )


if __name__ == "__main__":
    unittest.main()
