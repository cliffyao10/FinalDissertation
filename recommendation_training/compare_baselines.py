"""Compare learned compatibility against reproducible non-training baselines."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from recommendation_training.compatibility_model import load_checkpoint
from recommendation_training.dataset import OutfitDataset
from recommendation_training.evaluate import bootstrap_auc_interval, predict


def mean_pairwise_cosine(embeddings, masks):
    """Score an outfit by the mean cosine similarity of its present item pairs."""

    embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    scores = []
    for outfit, mask in zip(embeddings, masks.bool()):
        present = outfit[mask]
        pair_scores = [
            torch.dot(present[left], present[right])
            for left in range(len(present))
            for right in range(left + 1, len(present))
        ]
        scores.append(
            torch.stack(pair_scores).mean().item() if pair_scores else 0.0
        )
    return np.asarray(scores, dtype=np.float64)


def auc_report(labels, scores, iterations, seed):
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "auc_95_percent_ci": bootstrap_auc_interval(
            labels,
            scores,
            iterations=iterations,
            seed=seed,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/compatibility_ranker.pt")
    parser.add_argument("--test", default="data/processed/test.pt")
    parser.add_argument("--output", default="results/baseline_comparison.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = OutfitDataset(args.test)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    model, metadata = load_checkpoint(args.checkpoint, device)
    labels, model_probabilities = predict(model, loader, device)
    cosine_scores = mean_pairwise_cosine(dataset.embeddings, dataset.masks)
    random_scores = np.random.default_rng(args.seed).random(len(labels))

    report = {
        "test_dataset": str(Path(args.test)),
        "examples": int(len(labels)),
        "baselines": {
            "random_ranking": auc_report(
                labels,
                random_scores,
                args.bootstrap_iterations,
                args.seed,
            ),
            "mean_pairwise_siglip_cosine": auc_report(
                labels,
                cosine_scores,
                args.bootstrap_iterations,
                args.seed,
            ),
            "learned_compatibility_ranker": auc_report(
                labels,
                model_probabilities,
                args.bootstrap_iterations,
                args.seed,
            ),
        },
        "checkpoint_metadata": metadata,
        "dataset_metadata": dataset.metadata,
        "notes": {
            "random": "Deterministic random scores provide a chance-level control.",
            "cosine": (
                "Mean pairwise similarity uses the same frozen image features "
                "but no learned compatibility parameters."
            ),
            "rule_based": (
                "The production colour/weather generator cannot score arbitrary "
                "Polyvore outfits and is therefore excluded from this AUC table."
            ),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved baseline comparison to {output_path}")


if __name__ == "__main__":
    main()
