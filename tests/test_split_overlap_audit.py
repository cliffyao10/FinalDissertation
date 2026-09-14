import json
import tempfile
import unittest
from pathlib import Path

from recommendation_training.audit_split_overlap import audit


class SplitOverlapAuditTests(unittest.TestCase):
    def test_reports_item_overlap_separately_from_outfit_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = {
                "train.json": [{"set_id": "a", "items": [{"item_id": "shared"}]}],
                "valid.json": [{"set_id": "b", "items": [{"item_id": "v"}]}],
                "test.json": [{"set_id": "c", "items": [{"item_id": "shared"}]}],
            }
            for name, payload in payloads.items():
                (root / name).write_text(json.dumps(payload), encoding="utf-8")
            report = audit(root)
            self.assertTrue(report["outfit_set_disjoint_verified"])
            self.assertFalse(report["strict_item_disjoint_verified"])
            self.assertEqual(report["pairwise_overlap"]["train__test"]["item_ids"], 1)


if __name__ == "__main__":
    unittest.main()
