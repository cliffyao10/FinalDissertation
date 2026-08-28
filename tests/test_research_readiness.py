import unittest

from recommendation_training.research_readiness_check import finalise


class ResearchReadinessTest(unittest.TestCase):
    def test_nonblocking_limitation_does_not_block_freeze(self):
        ready, failures = finalise(
            [
                {"name": "model", "passed": True, "blocking": True},
                {"name": "domain", "passed": False, "blocking": False},
            ]
        )
        self.assertTrue(ready)
        self.assertEqual(failures, [])

    def test_blocking_failure_is_reported(self):
        ready, failures = finalise(
            [{"name": "tests", "passed": False, "blocking": True}]
        )
        self.assertFalse(ready)
        self.assertEqual(failures, ["tests"])


if __name__ == "__main__":
    unittest.main()
