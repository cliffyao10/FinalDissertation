"""Compare group-balanced training without using the test set for selection."""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from recommendation_training.balance import example_weights, group_diagnostics
from recommendation_training.compatibility_model import CompatibilityRanker, ModelConfig
from recommendation_training.dataset import OutfitDataset
from recommendation_training.evaluate import classification_metrics, predict
from recommendation_training.fashion_model_comparison import ranking_metrics


CONFIGURATIONS = {
    "unweighted": {"strategy": "none", "exponent": 0.0, "maximum": 1.0},
    "slot_sqrt": {"strategy": "slot", "exponent": 0.5, "maximum": 3.0},
    "slot_length_sqrt": {
        "strategy": "slot_length", "exponent": 0.5, "maximum": 3.0,
    },
    "slot_length_inverse": {
        "strategy": "slot_length", "exponent": 1.0, "maximum": 3.0,
    },
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def selection_score(overall_auc, groups):
    """Equally value discrimination, macro groups, and the weakest slot."""

    return float(np.mean([
        overall_auc,
        groups["macro_slot_auc"],
        groups["worst_slot_auc"],
        groups["macro_length_auc"],
    ]))


def train_run(datasets, configuration_name, seed, args, device):
    seed_everything(seed)
    configuration = CONFIGURATIONS[configuration_name]
    config = ModelConfig(embedding_dim=datasets["train"].embeddings.shape[-1])
    model = CompatibilityRanker(config).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    weights = example_weights(datasets["train"], **configuration)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(datasets["validation"], batch_size=args.batch_size)
    best_score, best_epoch, best_state, best_validation = -1.0, 0, None, None
    without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimiser.zero_grad(set_to_none=True)
            logits = model(batch["embeddings"].to(device), batch["mask"].to(device))
            losses = loss_function(logits, batch["label"].to(device))
            batch_weights = weights[batch["index"]].to(device)
            loss = (losses * batch_weights).sum() / batch_weights.sum()
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(batch["label"])
        labels, probabilities = predict(model, validation_loader, device)
        overall = classification_metrics(labels, probabilities)
        groups = group_diagnostics(datasets["validation"], labels, probabilities)
        score = selection_score(overall["auc"], groups)
        history.append({
            "epoch": epoch,
            "train_loss": float(total_loss / len(datasets["train"])),
            "validation_auc": overall["auc"],
            "validation_balanced_selection_score": score,
            "validation_macro_slot_auc": groups["macro_slot_auc"],
            "validation_worst_slot_auc": groups["worst_slot_auc"],
            "validation_slot_auc_gap": groups["slot_auc_gap"],
        })
        print(
            f"config={configuration_name} seed={seed} epoch={epoch:02d} "
            f"auc={overall['auc']:.4f} balanced={score:.4f} "
            f"worst_slot={groups['worst_slot_auc']:.4f}"
        )
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            best_validation = {"overall": overall, "groups": groups}
            without_improvement = 0
        else:
            without_improvement += 1
            if args.patience and without_improvement >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size)
    labels, probabilities = predict(model, test_loader, device)
    test_metrics = classification_metrics(labels, probabilities)
    test_metrics["groups"] = group_diagnostics(
        datasets["test"], labels, probabilities
    )
    test_metrics.update(
        ranking_metrics(
            model, datasets["test"], device, seed=args.fitb_seed,
            batch_size=args.batch_size,
        )
    )
    checkpoint_path = Path(args.checkpoint_directory) / (
        f"{configuration_name}_seed{seed}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "config": vars(config),
        "extra": {
            "checkpoint_version": 2,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_balanced_selection_score": best_score,
            "best_validation_metrics": best_validation,
            "training_config": {**configuration, "name": configuration_name},
            "selection_protocol": (
                "Equal mean of overall AUC, macro slot AUC, worst-slot AUC, "
                "and macro outfit-length AUC on validation data only."
            ),
        },
    }, checkpoint_path)
    return {
        "configuration": configuration_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_balanced_selection_score": best_score,
        "best_validation_metrics": best_validation,
        "test_metrics": test_metrics,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "training_weight_range": [float(weights.min()), float(weights.max())],
    }


def mean_metric(runs, path):
    values = []
    for run in runs:
        value = run
        for key in path:
            value = value[key]
        values.append(value)
    return {
        "values": values,
        "mean": float(np.mean(values)),
        "sample_standard_deviation": float(np.std(values, ddof=1)),
    }


def summarise(all_runs):
    paths = {
        "auc": ("test_metrics", "auc"),
        "pair_ranking_accuracy": ("test_metrics", "pair_ranking_accuracy"),
        "fitb4_accuracy": ("test_metrics", "fitb4_accuracy"),
        "macro_slot_auc": ("test_metrics", "groups", "macro_slot_auc"),
        "worst_slot_auc": ("test_metrics", "groups", "worst_slot_auc"),
        "slot_auc_gap": ("test_metrics", "groups", "slot_auc_gap"),
        "macro_length_auc": ("test_metrics", "groups", "macro_length_auc"),
        "length_auc_gap": ("test_metrics", "groups", "length_auc_gap"),
    }
    return {
        name: {metric: mean_metric(runs, path) for metric, path in paths.items()}
        for name, runs in all_runs.items()
    }


def choose_configuration(all_runs):
    """Choose only from aggregate validation scores; never inspect test metrics."""

    scores = {
        name: float(np.mean([
            run["best_validation_balanced_selection_score"] for run in runs
        ]))
        for name, runs in all_runs.items()
    }
    chosen = max(scores, key=scores.get)
    representative = max(
        all_runs[chosen], key=lambda run: run["best_validation_balanced_selection_score"]
    )
    return chosen, scores, representative["checkpoint"]


def deployment_gate(summary, chosen):
    """Require subgroup gains without material overall discrimination loss."""

    baseline = summary["unweighted"]
    candidate = summary[chosen]
    checks = {
        "overall_auc_loss_at_most_0_01": (
            candidate["auc"]["mean"] >= baseline["auc"]["mean"] - 0.01
        ),
        "macro_slot_auc_loss_at_most_0_002": (
            candidate["macro_slot_auc"]["mean"]
            >= baseline["macro_slot_auc"]["mean"] - 0.002
        ),
        "worst_slot_auc_improves": (
            candidate["worst_slot_auc"]["mean"] > baseline["worst_slot_auc"]["mean"]
        ),
        "slot_auc_gap_reduces": (
            candidate["slot_auc_gap"]["mean"] < baseline["slot_auc_gap"]["mean"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.pt")
    parser.add_argument("--validation", default="data/processed/validation.pt")
    parser.add_argument("--test", default="data/processed/test.pt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fitb-seed", type=int, default=2026)
    parser.add_argument("--checkpoint-directory", default="models/imbalance_mitigation")
    parser.add_argument("--output", default="results/imbalance_mitigation_study.json")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = {
        "train": OutfitDataset(args.train),
        "validation": OutfitDataset(args.validation),
        "test": OutfitDataset(args.test),
    }
    all_runs = {
        name: [train_run(datasets, name, seed, args, device) for seed in args.seeds]
        for name in CONFIGURATIONS
    }
    summary = summarise(all_runs)
    chosen, validation_scores, representative = choose_configuration(all_runs)
    gate = deployment_gate(summary, chosen)
    report = {
        "protocol": {
            "purpose": "Mitigate changed-slot and outfit-length imbalance.",
            "test_set_selection_warning": (
                "Configuration and representative checkpoint are selected only "
                "from validation balanced-selection scores. Test metrics are final reporting."
            ),
            "selection_score": (
                "Equal mean of overall AUC, macro slot AUC, worst-slot AUC, and "
                "macro outfit-length AUC."
            ),
            "configurations": CONFIGURATIONS,
            "seeds": args.seeds,
        },
        "runs": all_runs,
        "summary": summary,
        "selection": {
            "validation_scores": validation_scores,
            "chosen_configuration": chosen,
            "representative_checkpoint": representative,
            "deployment_gate": gate,
            "equivalence_margins": {
                "overall_auc": 0.01,
                "macro_slot_auc": 0.002,
                "rationale": (
                    "Small tolerance avoids treating sub-thousandth sampling and "
                    "floating-point differences as a material fairness regression."
                ),
            },
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "selection": report["selection"]}, indent=2))
    print(f"Saved imbalance mitigation study to {output_path}")


if __name__ == "__main__":
    main()
