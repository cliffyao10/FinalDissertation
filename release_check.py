"""Fast, deterministic release-readiness check for Cove."""

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
import torch


REQUIRED_FILES = (
    "app.py",
    "requirements.txt",
    "requirements-reproducible.txt",
    "data/catalogue.csv",
    "recommendation_training/MODEL_CARD.md",
    "results/compatibility_test_metrics.json",
    "results/bias_audit.json",
    "results/reproducibility_manifest.json",
)
TRAINED_ARTIFACTS = (
    "models/compatibility_ranker.pt",
    "models/catalogue_embeddings.pt",
)
LOCAL_PRIVATE_PATHS = (
    "data/user_preferences.json",
    "data/wardrobe.json",
    "data/wardrobe_images",
)


def record(checks, name, status, detail):
    checks.append({"name": name, "status": status, "detail": detail})


def git_ignored(root, relative_path):
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", relative_path],
            cwd=root,
            check=False,
            capture_output=True,
        )
    except OSError:
        return None
    return result.returncode == 0


def run_checks(root, strict_artifacts=False):
    root = Path(root)
    checks = []
    for relative in REQUIRED_FILES:
        exists = (root / relative).is_file()
        record(
            checks,
            f"required:{relative}",
            "pass" if exists else "fail",
            "present" if exists else "missing",
        )

    artifact_presence = [(root / relative).is_file() for relative in TRAINED_ARTIFACTS]
    if all(artifact_presence):
        checkpoint = torch.load(
            root / TRAINED_ARTIFACTS[0], map_location="cpu", weights_only=False
        )
        catalogue = torch.load(
            root / TRAINED_ARTIFACTS[1], map_location="cpu", weights_only=False
        )
        expected_dimension = int(checkpoint["config"]["embedding_dim"])
        items = catalogue.get("items", [])
        dimensions = {
            int(item["embedding"].shape[-1])
            for item in items
            if isinstance(item.get("embedding"), torch.Tensor)
        }
        compatible = bool(items) and dimensions == {expected_dimension}
        record(
            checks,
            "trained_artifacts",
            "pass" if compatible else "fail",
            f"checkpoint_dim={expected_dimension}, catalogue_items={len(items)}, catalogue_dims={sorted(dimensions)}",
        )
    elif any(artifact_presence):
        record(
            checks,
            "trained_artifacts",
            "fail",
            "checkpoint/catalogue pair is incomplete",
        )
    else:
        status = "fail" if strict_artifacts else "warning"
        record(
            checks,
            "trained_artifacts",
            status,
            "absent; application will use the documented rules fallback",
        )

    catalogue_path = root / "data/catalogue.csv"
    if catalogue_path.is_file():
        frame = pd.read_csv(catalogue_path)
        required_columns = {"item_id", "slot", "type", "colour", "image_path"}
        counts = (
            frame.groupby("slot").size().to_dict()
            if required_columns <= set(frame)
            else {}
        )
        valid = (
            required_columns <= set(frame)
            and set(counts) == {"inner_top", "outer_top", "bottom", "shoes"}
            and min(counts.values()) > 0
        )
        record(
            checks,
            "catalogue_schema",
            "pass" if valid else "fail",
            f"slot_counts={counts}",
        )

    metrics_path = root / "results/compatibility_test_metrics.json"
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        auc = payload.get("metrics", {}).get("auc")
        calibrated = "calibration" in payload.get("checkpoint_metadata", {})
        valid = isinstance(auc, (int, float)) and auc > 0.5 and calibrated
        record(
            checks,
            "model_evidence",
            "pass" if valid else "fail",
            f"auc={auc}, calibrated={calibrated}",
        )

    for relative in LOCAL_PRIVATE_PATHS:
        ignored = git_ignored(root, relative)
        status = "pass" if ignored is True else "warning" if ignored is None else "fail"
        detail = "Git-ignored" if ignored else "unable to confirm Git ignore"
        record(checks, f"local_privacy:{relative}", status, detail)

    return {
        "passed": not any(check["status"] == "fail" for check in checks),
        "failures": sum(check["status"] == "fail" for check in checks),
        "warnings": sum(check["status"] == "warning" for check in checks),
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-artifacts", action="store_true")
    parser.add_argument("--output", default="results/release_check.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    report = run_checks(root, strict_artifacts=args.strict_artifacts)
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
