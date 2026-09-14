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
    "results/abstract_prototype_build.json",
    "results/abstract_search_benchmark.json",
    "results/v3_system_comparison.json",
    "results/v3_release_check.json",
    "results/d2_evaluation.json",
    "results/d2_official_fitb_ablation.json",
    "results/d2_overfitting_audit.json",
    "results/d2_promotion.json",
    "results/research_readiness_check.json",
)
TRAINED_ARTIFACTS = (
    "models/compatibility_ranker.pt",
    "models/polyvore_abstract_prototypes.pt",
)
LOCAL_PRIVATE_PATHS = (
    "data/user_preferences.json",
    "data/wardrobe.json",
    "data/wardrobe_images",
)


def record(checks, name, status, detail):
    checks.append({"name": name, "status": status, "detail": detail})


def git_ignored(root, relative_path):
    # A directory-only ignore rule (for example ``data/wardrobe_images/``)
    # does not match the bare path when that directory has not been created
    # yet. Probe a hypothetical child so the check behaves consistently in a
    # clean CI checkout and in a local workspace where the directory exists.
    candidate = Path(relative_path)
    probe = candidate / ".privacy-check" if not candidate.suffix else candidate
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", str(probe)],
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
        model_version = checkpoint.get("extra", {}).get("model_version")
        items = catalogue.get("items", [])
        dimensions = {
            int(item["embedding"].shape[-1])
            for item in items
            if isinstance(item.get("embedding"), torch.Tensor)
        }
        compatible = (
            bool(items)
            and dimensions == {expected_dimension}
            and model_version == "D2"
        )
        forbidden_fields = {"brand", "price", "source_url", "title"}
        abstract_safe = (
            catalogue.get("schema") in {
                "abstract_garment_prototypes_v1",
                "abstract_garment_prototypes_v2",
            }
            and catalogue.get("source_policy")
            == "training_split_only_no_product_media_or_brand_output"
            and all(item.get("abstract") for item in items)
            and all(not item.get("image_path") for item in items)
            and all(not (forbidden_fields & set(item)) for item in items)
            and all(item.get("audience") in {"menswear", "womenswear"} for item in items)
        )
        record(
            checks,
            "trained_artifacts",
            "pass" if compatible and abstract_safe else "fail",
            f"model_version={model_version}, checkpoint_dim={expected_dimension}, abstract_prototypes={len(items)}, "
            f"catalogue_dims={sorted(dimensions)}, abstract_safe={abstract_safe}",
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

    readiness_path = root / "results/research_readiness_check.json"
    if readiness_path.is_file():
        try:
            readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            readiness = {}
            readiness_detail = f"invalid readiness artifact: {error}"
        else:
            readiness_detail = readiness.get("decision")
        ready = readiness.get("ready_for_report_rewrite") is True
        if ready or strict_artifacts:
            record(
                checks,
                "research_readiness",
                "pass" if ready else "fail",
                readiness_detail,
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
