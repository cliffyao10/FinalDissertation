import hashlib
import tempfile
import unittest
from pathlib import Path

from recommendation_training.reproducibility_manifest import sha256


class ReproducibilityManifestTests(unittest.TestCase):
    def test_sha256_matches_standard_library(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.bin"
            path.write_bytes(b"cove")
            self.assertEqual(sha256(path), hashlib.sha256(b"cove").hexdigest())


if __name__ == "__main__":
    unittest.main()
