"""Evaluate D2 structural ablations on corrected official FITB labels.

The ablation checkpoints were trained on identical disjoint splits and frozen
SigLIP embeddings. This module adds the missing ranking-task evaluation and a
paired question bootstrap, without retraining or touching the production model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from recommendation_training.ablation_study import AblationRanker
from recommendation_training.compatibility_model import CompatibilityRanker, ModelConfig
from recommendation_training.evaluate_official_fitb import (
    fitb_metrics,
    prepare_official_fitb,
)
from recommendation_training.fashion_model_comparison import score_tensors


VARIANTS = {
    "no_slot": lambda config: AblationRanker(
        config, use_slot_embedding=False, use_pairwise=True
    ),
    "no_pairwise": lambda config: AblationRanker(
        config, use_slot_embedding=True, use_pairwise=False
    ),
    "full": lambda config: CompatibilityRanker(config),
}


def correct_vector(scores, targets):
    rows = np.asarray(scores, dtype=np.float64).reshape(-1, 4)
    targets = np.asarray(targets, dtype=np.int64)
    correct_scores = rows[np.arange(len(rows)), targets]
    competitors = np.arange(4)[None, :] != targets[:, None]
    ranks = 1 + np.sum((rows >= correct_scores[:, None]) & competitors, axis=1)
    return ranks == 1


def paired_question_bootstrap(full_by_seed, ablated_by_seed, *, seed=20260821, iterations=5000):
    """Bootstrap questions while averaging the repeated-seed outcomes."""

    full = np.stack(full_by_seed).astype(np.float64)
    ablated = np.stack(ablated_by_seed).astype(np.float64)
    per_question_difference = (full - ablated).mean(axis=0)
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    question_count = per_question_difference.shape[0]
    for index in range(iterations):
        draw = rng.integers(0, question_count, size=question_count)
        samples[index] = per_question_difference[draw].mean()
    return {
        "mean_accuracy_difference_full_minus_ablation": float(
            per_question_difference.mean()
        ),
        "paired_question_bootstrap_95_percent_ci": [
            float(value) for value in np.quantile(samples, [0.025, 0.975])
        ],
        "iterations": iterations,
        "unit_resampled": "official_fitb_question",
        "seed_repeats_averaged_within_question": int(full.shape[0]),
    }


def summarise(runs):
    result = {}
    for name, values in runs.items():
        result[name] = {}
        for metric in ("accuracy", "mean_reciprocal_rank", "ndcg"):
            numbers = [row[metric] for row in values]
            result[name][metric] = {
                "values": numbers,
                "mean": float(np.mean(numbers)),
                "sample_standard_deviation": float(np.std(numbers, ddof=1)),
            }
    return result


def evaluate(args):
    payload = prepare_official_fitb(
        args.metadata,
        args.questions,
        args.cache,
        minimum_slots=args.minimum_slots,
        item_metadata_path=args.item_metadata,
    )
    if not payload["retained_questions"]:
        raise RuntimeError("No official FITB questions are representable.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ModelConfig(embedding_dim=payload["embeddings"].shape[-1])
    runs = {name: [] for name in VARIANTS}
    correctness = {name: [] for name in VARIANTS}

    for seed in args.seeds:
        for variant, factory in VARIANTS.items():
            checkpoint_name = "full_retrained" if variant == "full" else variant
            checkpoint_path = (
                Path(args.checkpoint_directory)
                / f"seed{seed}"
                / f"{checkpoint_name}.pt"
            )
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            model = factory(config).to(device)
            model.load_state_dict(checkpoint["model_state"])
            scores = score_tensors(
                model,
                payload["embeddings"],
                payload["masks"],
                device,
                batch_size=args.batch_size,
            )
            metrics = fitb_metrics(scores, payload["correct_answer_indices"])
            metrics.update(
                {
                    "seed": seed,
                    "checkpoint": str(checkpoint_path),
                    "trainable_parameters": int(
                        sum(parameter.numel() for parameter in model.parameters())
                    ),
                }
            )
            runs[variant].append(metrics)
            correctness[variant].append(
                correct_vector(scores, payload["correct_answer_indices"])
            )

    comparisons = {
        name: paired_question_bootstrap(
            correctness["full"], correctness[name], iterations=args.bootstrap_iterations
        )
        for name in ("no_slot", "no_pairwise")
    }
    report = {
        "protocol": (
            "Controlled three-seed D2 structural ablation on the corrected "
            "official-disjoint FITB subset. Correct answers are matched by set_id."
        ),
        "coverage": {
            "total_questions": payload["total_questions"],
            "retained_questions": payload["retained_questions"],
            "retained_fraction": payload["retained_questions"]
            / payload["total_questions"],
            "skipped": payload["skipped"],
            "selection_bias_warning": (
                "The four-slot filtered result is not interchangeable with a "
                "published full-dataset FITB result."
            ),
        },
        "device": device,
        "seeds": args.seeds,
        "runs": runs,
        "summary": summarise(runs),
        "paired_comparisons": comparisons,
        "interpretation_rule": (
            "A component has clear positive FITB evidence only when the paired "
            "95% interval for full-minus-ablation lies entirely above zero."
        ),
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/polyvore_disjoint/test.json")
    parser.add_argument(
        "--item-metadata",
        default="data/polyvore_nondisjoint/polyvore_item_metadata.json",
    )
    parser.add_argument(
        "--questions", default="data/polyvore_disjoint/fill_in_blank_test.json"
    )
    parser.add_argument("--cache", default="data/processed_disjoint/fitb_item_embeddings.pt")
    parser.add_argument(
        "--checkpoint-directory", default="models/disjoint_ablations"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--minimum-slots", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--output", default="results/d2_official_fitb_ablation.json")
    args = parser.parse_args()

    report = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "paired": report["paired_comparisons"]}, indent=2))
    print(f"Saved corrected D2 FITB ablation to {output}")


if __name__ == "__main__":
    main()
