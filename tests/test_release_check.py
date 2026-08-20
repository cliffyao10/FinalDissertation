import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from release_check import REQUIRED_FILES, run_checks


class ReleaseCheckTests(unittest.TestCase):
    def make_minimum_project(self, root):
        for relative in REQUIRED_FILES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("placeholder", encoding="utf-8")
        pd.DataFrame([
            {
                "item_id": slot,
                "slot": slot,
                "type": "Item",
                "colour": "Black",
                "image_path": "image.jpg",
            }
            for slot in ("inner_top", "outer_top", "bottom", "shoes")
        ]).to_csv(root / "data/catalogue.csv", index=False)
        (root / "results/compatibility_test_metrics.json").write_text(
            json.dumps({
                "metrics": {"auc": 0.7},
                "checkpoint_metadata": {"calibration": {"temperature": 1.1}},
            }),
            encoding="utf-8",
        )

    @patch("release_check.git_ignored", return_value=True)
    def test_missing_artifacts_warn_in_normal_mode(self, _):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_minimum_project(root)
            report = run_checks(root)
        self.assertTrue(report["passed"])
        self.assertEqual(report["warnings"], 1)

    @patch("release_check.git_ignored", return_value=True)
    def test_missing_artifacts_fail_in_strict_mode(self, _):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_minimum_project(root)
            report = run_checks(root, strict_artifacts=True)
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
