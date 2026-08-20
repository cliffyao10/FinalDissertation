"""Fit a scalar temperature on validation data and update a checkpoint."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader

from recommendation_training.compatibility_model import load_checkpoint
from recommendation_training.dataset import OutfitDataset
from recommendation_training.error_analysis import expected_calibration_error


@torch.no_grad()
def collect_logits(model, loader, device):
    logits, labels = [], []
    model.eval()
    for batch in loader:
        logits.append(model(batch["embeddings"].to(device), batch["mask"].to(device)).cpu())
        labels.append(batch["label"].cpu())
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits, labels, maximum_steps=100):
    """Minimise validation negative log likelihood with one positive scalar."""

    log_temperature = nn.Parameter(torch.zeros(()))
    optimiser = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=maximum_steps, line_search_fn="strong_wolfe"
    )

    def closure():
        optimiser.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 10.0)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits / temperature, labels
        )
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_temperature.exp().clamp(0.05, 10.0).detach())


def calibration_report(logits, labels, temperature):
    before = torch.sigmoid(logits).numpy()
    after = torch.sigmoid(logits / temperature).numpy()
    label_values = labels.numpy().astype(int)
    return {
        "temperature": temperature,
        "validation_nll_before": float(
            nn.functional.binary_cross_entropy_with_logits(logits, labels)
        ),
        "validation_nll_after": float(
            nn.functional.binary_cross_entropy_with_logits(logits / temperature, labels)
        ),
        "validation_ece_before": expected_calibration_error(label_values, before)["ece"],
        "validation_ece_after": expected_calibration_error(label_values, after)["ece"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/compatibility_ranker.pt")
    parser.add_argument("--validation", default="data/processed/validation.pt")
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output) if args.output else checkpoint_path
    dataset = OutfitDataset(args.validation)
    model, _ = load_checkpoint(checkpoint_path)
    # Calibration is always fitted to raw logits, not previously scaled output.
    model.calibration_temperature = 1.0
    logits, labels = collect_logits(
        model, DataLoader(dataset, batch_size=args.batch_size), "cpu"
    )
    temperature = fit_temperature(logits, labels)
    report = calibration_report(logits, labels, temperature)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload.setdefault("extra", {})["calibration"] = {
        **report,
        "dataset": str(Path(args.validation)),
        "method": "scalar_temperature_scaling",
        "fit_scope": "validation_only",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(json.dumps(report, indent=2))
    print(f"Saved calibrated checkpoint to {output_path}")


if __name__ == "__main__":
    main()
