"""Machine-readable release gate for the Cove v3 search system."""

import argparse
import json
from pathlib import Path


REQUIRED = (
    "results/compatibility_test_metrics.json",
    "results/abstract_prototype_build.json",
    "results/abstract_search_benchmark.json",
    "results/v3_system_comparison.json",
    "models/compatibility_ranker.pt",
    "models/polyvore_abstract_prototypes.pt",
    "results/d2_evaluation.json",
    "results/d2_official_fitb_ablation.json",
)


def evaluate_gate(root):
    root = Path(root)
    checks = []

    def add(name, passed, detail, blocking=True):
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "blocking": blocking,
                "detail": detail,
            }
        )

    for relative in REQUIRED:
        add(f"artifact:{relative}", (root / relative).is_file(), relative)
    if not all((root / relative).is_file() for relative in REQUIRED):
        return {
            "passed": False,
            "system_version": "3.0",
            "decision": "v3_incomplete",
            "checks": checks,
            "blocking_failures": [
                check["name"] for check in checks if check["blocking"] and not check["passed"]
            ],
        }

    comparison = json.loads(
        (root / "results/v3_system_comparison.json").read_text(encoding="utf-8")
    )
    metrics = comparison["shared_head_metrics_v2_and_v3"]
    search = comparison["search_ablation_same_abstract_candidate_space"]
    runtime = comparison["runtime_seconds_same_machine"]["v3_exact"]
    abstract = comparison["candidate_spaces"]["v3_abstract"]
    privacy = abstract["privacy_contract"]
    weather = comparison["weather_style_coverage"]
    d2 = json.loads((root / "results/d2_evaluation.json").read_text(encoding="utf-8"))

    add("shared_head_auc_above_0_70", metrics["auc"] > 0.70, metrics["auc"])
    add(
        "shared_head_auc_ci_above_chance",
        metrics["auc_95_percent_ci"][0] > 0.5,
        metrics["auc_95_percent_ci"],
    )
    add(
        "exact_search_deterministic",
        search["deterministic_repeat_top10"],
        search["deterministic_repeat_top10"],
    )
    add(
        "exact_search_never_worse_than_quota_objective",
        search["model_score_regret_quota"]["minimum"] >= -1e-7,
        search["model_score_regret_quota"],
    )
    add(
        "all_weather_style_cases_have_logical_slots",
        weather["all_cases_have_four_logical_slots"],
        {"cases": weather["cases"]},
    )
    add(
        "abstract_output_has_no_product_media",
        privacy["all_abstract"]
        and privacy["nonempty_image_paths"] == 0
        and privacy["records_with_forbidden_commercial_fields"] == 0,
        privacy,
    )
    add(
        "cpu_median_exact_search_below_1_second",
        runtime["median"] < 1.0,
        runtime,
    )
    add(
        "cpu_p95_exact_search_below_3_seconds",
        runtime["p95"] < 3.0,
        runtime,
    )
    add("d2_research_gate_passed", d2.get("passed"), d2.get("gates", {}))
    add(
        "corrected_disjoint_candidate_promoted",
        comparison["research_candidate_gate"]["promotion_allowed"],
        comparison["research_candidate_gate"],
    )

    failures = [
        check["name"]
        for check in checks
        if check["blocking"] and not check["passed"]
    ]
    return {
        "passed": not failures,
        "system_version": "3.0",
        "decision": "v3_release_ready" if not failures else "v3_incomplete",
        "checks": checks,
        "blocking_failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/v3_release_check.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = evaluate_gate(root)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
