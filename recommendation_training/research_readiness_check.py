"""Final machine-readable audit before dissertation results are rewritten."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ARTIFACTS = {
    "v1_rule_baseline": [
        "src/recommendation.py",
        "results/baseline_comparison.json",
    ],
    "v2_historical_learned_retrieval": [
        "models/legacy/compatibility_ranker_p1.pt",
        "results/compatibility_error_analysis.json",
        "results/ablation_summary.json",
    ],
    "d2_disjoint_model": [
        "models/d2/compatibility_ranker.pt",
        "results/d2_compatibility_test_metrics.json",
        "results/d2_official_fitb_ablation.json",
        "results/d2_overfitting_audit.json",
        "results/d2_evaluation.json",
        "results/disjoint_baseline_comparison.json",
        "results/disjoint_fashion_model_comparison.json",
        "results/disjoint_imbalance_study.json",
        "results/disjoint_bias_audit.json",
        "results/disjoint_domain_shift.json",
        "results/disjoint_runtime_benchmark.json",
        "results/disjoint_official_fitb_comparison.json",
    ],
    "v3_abstract_retrieval_system": [
        "models/polyvore_abstract_prototypes.pt",
        "results/abstract_prototype_build.json",
        "results/abstract_search_benchmark.json",
        "results/abstract_search_benchmark_menswear.json",
        "results/v3_system_comparison.json",
        "results/v3_release_check.json",
    ],
    "reproducibility_and_governance": [
        "DATA_PROVENANCE.md",
        "recommendation_training/MODEL_CARD.md",
        "results/reproducibility_manifest.json",
        "release_check.py",
    ],
}


def finalise(checks):
    blocking_failures = [
        check["name"]
        for check in checks
        if check.get("blocking", True) and not check["passed"]
    ]
    return not blocking_failures, blocking_failures


def read_json(root, relative):
    return json.loads((root / relative).read_text(encoding="utf-8"))


def command_check(command, cwd):
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    return result.returncode == 0, output


def evaluate(root, *, run_tests=False):
    root = Path(root)
    checks = []

    def add(name, passed, detail, *, blocking=True):
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "blocking": bool(blocking),
                "detail": detail,
            }
        )

    version_matrix = {}
    for version, relatives in ARTIFACTS.items():
        missing = [relative for relative in relatives if not (root / relative).is_file()]
        version_matrix[version] = {
            "complete": not missing,
            "artifact_count": len(relatives),
            "missing": missing,
        }
        add(f"artifact_family:{version}", not missing, version_matrix[version])

    required_json = [
        "results/d2_evaluation.json",
        "results/disjoint_release_check.json",
        "results/v3_release_check.json",
    ]
    if all((root / relative).is_file() for relative in required_json):
        d2 = read_json(root, required_json[0])
        disjoint = read_json(root, required_json[1])
        v3 = read_json(root, required_json[2])
        add("d2_gate", d2.get("passed"), d2.get("gates", {}))
        add(
            "d2_corrected_three_seed_ablation",
            d2.get("gates", {}).get("corrected_structural_ablation_completed"),
            d2.get("corrected_structural_ablation", {}),
        )
        add(
            "d2_pairwise_module_supported",
            d2.get("gates", {}).get(
                "pairwise_module_supported_by_paired_fitb_interval"
            ),
            d2.get("corrected_structural_ablation", {})
            .get("paired_comparisons", {})
            .get("no_pairwise", {}),
        )
        add(
            "disjoint_model_release_gate",
            disjoint.get("production_promotion_allowed"),
            disjoint.get("decision"),
        )
        add("v3_system_release_gate", v3.get("passed"), v3.get("decision"))

    overfitting_path = root / "results/d2_overfitting_audit.json"
    if overfitting_path.is_file():
        overfitting = read_json(root, "results/d2_overfitting_audit.json")
        add(
            "d2_overfitting_audited",
            overfitting.get("audit", {}).get("decision")
            == "retain_frozen_checkpoint",
            overfitting.get("audit", {}),
        )

    if run_tests:
        passed, output = command_check(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
            root,
        )
        match = re.search(r"Ran (\d+) tests?", output)
        add(
            "unit_test_suite",
            passed,
            {
                "tests_run": int(match.group(1)) if match else None,
                "tail": output[-1200:],
            },
        )
        compiled, compile_output = command_check(
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "app.py",
                "src",
                "recommendation_training",
                "tests",
            ],
            root,
        )
        add("python_compile", compiled, compile_output[-1200:])

    ready, failures = finalise(checks)
    return {
        "decision": "ready_for_report_rewrite" if ready else "code_or_evidence_incomplete",
        "ready_for_report_rewrite": ready,
        "version_scope_note": (
            "V1/V2/V3 identify retrieval-system generations; D2 identifies the "
            "second disjoint-trained compatibility head. D2 is used inside V3 "
            "and is not a fourth system generation."
        ),
        "version_matrix": version_matrix,
        "checks": checks,
        "blocking_failures": failures,
        "known_limitations_to_preserve_in_the_dissertation": [
            "Corrected official FITB covers 4,468 of 15,145 questions (29.50%) because the product taxonomy has four slots.",
            "Pure menswear evidence is sparse; the interface applies a user-selected hard candidate-range filter rather than claiming a separately validated menswear model.",
            "The frozen-feature catalogue/domain diagnostic is strongly separable (domain classifier AUC 1.0), so external catalogue generalisation is not established.",
            "The slot embedding's small FITB point gain is not clear under the paired 95% interval; only the pairwise module has clear ablation support.",
            "No participant study is claimed; usability evidence is software verification only under the project's ethics constraint.",
            "Dress-specific modelling is intentionally outside the frozen four-slot scope.",
            "Historical V1/V2 metrics use different candidate/data protocols and must not be presented as a causal head-to-head performance gain.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--output", default="results/research_readiness_check.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = evaluate(root, run_tests=args.run_tests)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ready_for_report_rewrite"] else 2)


if __name__ == "__main__":
    main()
