"""Record exact artifact hashes and runtime versions for repeatable evaluation."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ARTIFACTS = (
    "data/processed/train.pt",
    "data/processed/validation.pt",
    "data/processed/test.pt",
    "models/compatibility_ranker.pt",
    "models/catalogue_embeddings.pt",
    "data/catalogue.csv",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", nargs="+", default=list(DEFAULT_ARTIFACTS))
    parser.add_argument("--output", default="results/reproducibility_manifest.json")
    args = parser.parse_args()
    artifacts = {}
    for value in args.artifacts:
        path = Path(value)
        artifacts[value] = {
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path) if path.is_file() else None,
        }
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(
            ("numpy", "pandas", "Pillow", "scikit-learn", "torch", "transformers", "streamlit")
        ),
        "artifacts": artifacts,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
