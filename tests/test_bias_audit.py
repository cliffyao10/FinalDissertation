import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recommendation_training.bias_audit import catalogue_audit


class BiasAuditTests(unittest.TestCase):
    def test_catalogue_audit_reports_slot_and_colour_concentration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalogue.csv"
            pd.DataFrame([
                {"slot": "shoes", "section": "MAN", "colour": "Black"},
                {"slot": "shoes", "section": "WOMAN", "colour": "Black"},
                {"slot": "inner_top", "section": "WOMAN", "colour": "Red"},
            ]).to_csv(path, index=False)
            report = catalogue_audit(path)
        self.assertEqual(report["by_slot"]["shoes"], 2)
        self.assertAlmostEqual(report["largest_colour_share"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
