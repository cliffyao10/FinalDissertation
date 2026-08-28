"""Evidence-based release gate for the official-disjoint candidate model."""

import argparse
import json
from pathlib import Path

import torch


REQUIRED_RESULTS = {
    "test": "results/disjoint_compatibility_test_metrics.json",
    "baselines": "results/disjoint_baseline_comparison.json",
    "architectures": "results/disjoint_fashion_model_comparison.json",
    "fitb": "results/disjoint_official_fitb_comparison.json",
    "imbalance": "results/disjoint_imbalance_study.json",
    "bias": "results/disjoint_bias_audit.json",
    "domain_shift": "results/disjoint_domain_shift.json",
    "runtime": "results/disjoint_runtime_benchmark.json",
}


def run_checks(root):
    root = Path(root)
    checks = []

    def record(name, passed, detail, *, blocking=True):
        checks.append({
            "name": name,
            "passed": bool(passed),
            "blocking": bool(blocking),
            "detail": detail,
        })

    paths = {name: root / value for name, value in REQUIRED_RESULTS.items()}
    for name, path in paths.items():
        record(f"artifact:{name}", path.is_file(), str(path.relative_to(root)))
    checkpoint_path = root / "models/disjoint/compatibility_ranker.pt"
    record("artifact:checkpoint", checkpoint_path.is_file(), str(checkpoint_path.relative_to(root)))
    if any(not check["passed"] for check in checks):
        return {
            "passed": False,
            "production_promotion_allowed": False,
            "checks": checks,
        }

    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    extra = checkpoint.get("extra", {})
    calibration = extra.get("calibration", {})
    record(
        "validation_only_calibration",
        extra.get("checkpoint_version", 0) >= 2
        and calibration.get("fit_scope") == "validation_only"
        and calibration.get("validation_nll_after", 1e9)
        < calibration.get("validation_nll_before", -1e9),
        calibration,
    )

    learned = payloads["baselines"]["baselines"]["learned_compatibility_ranker"]
    cosine = payloads["baselines"]["baselines"]["mean_pairwise_siglip_cosine"]
    record(
        "compatibility_auc_exceeds_frozen_cosine",
        learned["auc_95_percent_ci"][0] > cosine["auc_95_percent_ci"][1],
        {"learned": learned, "cosine": cosine},
    )

    architecture_summary = payloads["architectures"]["summary"]
    proposed = architecture_summary["lightweight_pairwise_proposed"]
    proposed_auc = proposed["metrics"]["auc"]
    baseline_aucs = {
        name: value["metrics"]["auc"]["mean"]
        for name, value in architecture_summary.items()
        if name != "lightweight_pairwise_proposed"
    }
    record(
        "three_seed_architecture_result",
        proposed_auc["available_runs"] == 3
        and proposed_auc["sample_standard_deviation"] < 0.01
        and proposed_auc["mean"] > max(baseline_aucs.values()),
        {"proposed": proposed_auc, "baselines": baseline_aucs},
    )

    imbalance = payloads["imbalance"]["selection"]
    record(
        "imbalance_decision_is_guarded",
        imbalance.get("deployed_configuration") == "unweighted"
        and not imbalance["deployment_gate"]["passed"],
        imbalance,
    )

    fitb = payloads["fitb"]
    fitb_accuracy = fitb["summary"]["lightweight_pairwise_proposed"]["accuracy"]
    record(
        "official_fitb_above_chance",
        fitb_accuracy["mean"] > 0.25,
        {
            "mean_accuracy": fitb_accuracy["mean"],
            "chance_accuracy": 0.25,
            "retained_questions": fitb["coverage"]["retained_questions"],
            "retained_fraction": fitb["coverage"]["retained_fraction"],
        },
    )

    domain_auc = payloads["domain_shift"]["overall"][
        "domain_classifier_cross_validated_auc"
    ]
    record(
        "polyvore_catalogue_domain_overlap",
        domain_auc < 0.8,
        {"domain_classifier_auc": domain_auc},
        blocking=False,
    )
    blocking_failures = [
        check for check in checks if check["blocking"] and not check["passed"]
    ]
    return {
        "passed": not blocking_failures,
        "production_promotion_allowed": not blocking_failures,
        "decision": (
            "promote_disjoint_candidate"
            if not blocking_failures
            else "retain_current_production_checkpoint"
        ),
        "checks": checks,
        "blocking_failures": [check["name"] for check in blocking_failures],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/disjoint_release_check.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = run_checks(root)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
