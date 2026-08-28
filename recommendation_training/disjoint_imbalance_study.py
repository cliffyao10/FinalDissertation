"""Length-aware imbalance study for published, unpaired compatibility rows.

Unlike the synthetic replacement dataset, the official disjoint compatibility
files do not identify which slot was replaced.  This study therefore balances
only the observable three/four-item outfit-length groups and never fabricates a
changed-slot label.
"""

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from recommendation_training.compatibility_model import CompatibilityRanker, ModelConfig
from recommendation_training.dataset import OutfitDataset
from recommendation_training.evaluate import classification_metrics, predict


CONFIGURATIONS = {
    "unweighted": {"exponent": 0.0, "maximum": 1.0},
    "length_sqrt": {"exponent": 0.5, "maximum": 3.0},
    "length_inverse": {"exponent": 1.0, "maximum": 3.0},
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def length_weights(dataset, exponent=0.5, maximum=3.0):
    lengths = [int(value) for value in dataset.masks.sum(dim=1).tolist()]
    counts = Counter(lengths)
    target = len(lengths) / len(counts)
    values = [min(maximum, (target / counts[length]) ** exponent) for length in lengths]
    weights = torch.tensor(values, dtype=torch.float32)
    return weights / weights.mean()


def length_diagnostics(dataset, labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    lengths = dataset.masks.sum(dim=1).numpy()
    groups = {}
    aucs = []
    for length in sorted(set(lengths.tolist())):
        selected = lengths == length
        auc = float(roc_auc_score(labels[selected], probabilities[selected]))
        aucs.append(auc)
        groups[str(length)] = {
            "examples": int(selected.sum()),
            "positive_rate": float(labels[selected].mean()),
            "auc": auc,
        }
    return {
        "by_present_slot_count": groups,
        "macro_length_auc": float(np.mean(aucs)),
        "worst_length_auc": float(min(aucs)),
        "length_auc_gap": float(max(aucs) - min(aucs)),
    }


def selection_score(overall_auc, diagnostics):
    return float(np.mean([
        overall_auc,
        diagnostics["macro_length_auc"],
        diagnostics["worst_length_auc"],
    ]))


def train_run(datasets, name, seed, args, device):
    seed_everything(seed)
    configuration = CONFIGURATIONS[name]
    config = ModelConfig(embedding_dim=datasets["train"].embeddings.shape[-1])
    model = CompatibilityRanker(config).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    weights = length_weights(datasets["train"], **configuration)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(datasets["validation"], batch_size=args.batch_size)
    best_score, best_epoch, best_state, best_validation = -1.0, 0, None, None
    without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            optimiser.zero_grad(set_to_none=True)
            logits = model(batch["embeddings"].to(device), batch["mask"].to(device))
            losses = loss_function(logits, batch["label"].to(device))
            batch_weights = weights[batch["index"]].to(device)
            loss = (losses * batch_weights).sum() / batch_weights.sum()
            loss.backward()
            optimiser.step()
        labels, probabilities = predict(model, validation_loader, device)
        overall = classification_metrics(labels, probabilities)
        groups = length_diagnostics(datasets["validation"], labels, probabilities)
        score = selection_score(overall["auc"], groups)
        print(
            f"config={name} seed={seed} epoch={epoch:02d} "
            f"auc={overall['auc']:.4f} balanced={score:.4f}"
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
    labels, probabilities = predict(
        model, DataLoader(datasets["test"], batch_size=args.batch_size), device
    )
    test_metrics = classification_metrics(labels, probabilities)
    test_metrics["groups"] = length_diagnostics(
        datasets["test"], labels, probabilities
    )
    checkpoint_path = Path(args.checkpoint_directory) / f"{name}_seed{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "config": vars(config),
        "extra": {
            "checkpoint_version": 1,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_balanced_selection_score": best_score,
            "best_validation_metrics": best_validation,
            "training_config": {**configuration, "name": name},
        },
    }, checkpoint_path)
    return {
        "configuration": name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_balanced_selection_score": best_score,
        "best_validation_metrics": best_validation,
        "test_metrics": test_metrics,
        "checkpoint": str(checkpoint_path),
        "training_weight_range": [float(weights.min()), float(weights.max())],
    }


def metric_summary(runs, path):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed_disjoint/train.pt")
    parser.add_argument("--validation", default="data/processed_disjoint/validation.pt")
    parser.add_argument("--test", default="data/processed_disjoint/test.pt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-directory", default="models/disjoint_imbalance")
    parser.add_argument("--output", default="results/disjoint_imbalance_study.json")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = {
        name: OutfitDataset(getattr(args, name))
        for name in ("train", "validation", "test")
    }
    runs = {
        name: [train_run(datasets, name, seed, args, device) for seed in args.seeds]
        for name in CONFIGURATIONS
    }
    paths = {
        "auc": ("test_metrics", "auc"),
        "macro_length_auc": ("test_metrics", "groups", "macro_length_auc"),
        "worst_length_auc": ("test_metrics", "groups", "worst_length_auc"),
        "length_auc_gap": ("test_metrics", "groups", "length_auc_gap"),
    }
    summary = {
        name: {metric: metric_summary(values, path) for metric, path in paths.items()}
        for name, values in runs.items()
    }
    validation_scores = {
        name: float(np.mean([
            run["best_validation_balanced_selection_score"] for run in values
        ]))
        for name, values in runs.items()
    }
    chosen = max(validation_scores, key=validation_scores.get)
    baseline = summary["unweighted"]
    candidate = summary[chosen]
    checks = {
        "overall_auc_loss_at_most_0_01": (
            candidate["auc"]["mean"] >= baseline["auc"]["mean"] - 0.01
        ),
        "macro_length_auc_loss_at_most_0_002": (
            candidate["macro_length_auc"]["mean"]
            >= baseline["macro_length_auc"]["mean"] - 0.002
        ),
        "worst_length_auc_not_lower": (
            candidate["worst_length_auc"]["mean"]
            >= baseline["worst_length_auc"]["mean"]
        ),
    }
    gate = {"passed": bool(all(checks.values())), "checks": checks}
    deployed = chosen if gate["passed"] else "unweighted"
    representative = max(
        runs[deployed], key=lambda run: run["best_validation_balanced_selection_score"]
    )
    report = {
        "protocol": {
            "purpose": "Assess observable three/four-item outfit-length imbalance.",
            "changed_slot_note": (
                "Not evaluated because published compatibility rows do not expose "
                "a defensible changed-slot pair identifier."
            ),
            "selection_scope": "validation_only",
            "configurations": CONFIGURATIONS,
            "seeds": args.seeds,
        },
        "runs": runs,
        "summary": summary,
        "selection": {
            "validation_scores": validation_scores,
            "chosen_configuration": chosen,
            "deployment_gate": gate,
            "deployed_configuration": deployed,
            "representative_checkpoint": representative["checkpoint"],
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "selection": report["selection"]}, indent=2))


if __name__ == "__main__":
    main()
