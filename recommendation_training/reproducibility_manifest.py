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
    "data/processed_disjoint/train.pt",
    "data/processed_disjoint/validation.pt",
    "data/processed_disjoint/test.pt",
    "models/compatibility_ranker.pt",
    "models/catalogue_embeddings.pt",
    "models/polyvore_abstract_prototypes.pt",
    "data/catalogue.csv",
    "results/abstract_prototype_build.json",
    "results/abstract_search_benchmark.json",
    "results/v3_system_comparison.json",
    "results/v3_release_check.json",
    "results/d2_compatibility_test_metrics.json",
    "results/d2_evaluation.json",
    "results/d2_official_fitb_ablation.json",
    "results/d2_overfitting_audit.json",
    "results/d2_promotion.json",
    "results/disjoint_official_fitb_comparison.json",
    "results/disjoint_fashion_model_comparison.json",
    "results/disjoint_imbalance_study.json",
    "results/disjoint_bias_audit.json",
    "results/disjoint_domain_shift.json",
    "results/disjoint_runtime_benchmark.json",
    "results/research_readiness_check.json",
    "requirements-reproducible.txt",
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


def git_worktree_dirty():
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
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
        "git_revision_at_generation": git_revision(),
        "git_worktree_dirty_at_generation": git_worktree_dirty(),
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
