"""Promote a gated D2 checkpoint while preserving the previous production head."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="models/d2/compatibility_ranker.pt")
    parser.add_argument("--evaluation", default="results/d2_evaluation.json")
    parser.add_argument("--production", default="models/compatibility_ranker.pt")
    parser.add_argument("--backup", default="models/legacy/compatibility_ranker_p1.pt")
    parser.add_argument("--output", default="results/d2_promotion.json")
    args = parser.parse_args()

    evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    if not evaluation.get("passed"):
        raise RuntimeError("D2 evaluation gates did not pass; promotion refused.")
    candidate = Path(args.candidate)
    production = Path(args.production)
    backup = Path(args.backup)
    backup.parent.mkdir(parents=True, exist_ok=True)
    if production.is_file() and not backup.is_file():
        shutil.copy2(production, backup)

    checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
    candidate_state = {
        name: value.clone() for name, value in checkpoint["model_state"].items()
    }
    checkpoint.setdefault("extra", {}).update(
        {
            "model_version": "D2",
            "training_dataset": "Polyvore Outfits official disjoint compatibility split",
            "fitb_answer_policy": "correct answer matched by outfit set_id",
            "audience_policy": "user-selected hard candidate filtering; no gender inference",
            "menswear_training_limitation": (
                "Only 34 positive and 34 negative pure-menswear source rows in training; "
                "a separate menswear compatibility head is not claimed."
            ),
        }
    )
    production.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, production)
    report = {
        "promoted": True,
        "model_version": "D2",
        "candidate": str(candidate),
        "production": str(production),
        "legacy_backup": str(backup),
        "candidate_sha256": sha256(candidate),
        "production_sha256": sha256(production),
        "model_state_identical": all(
            torch.equal(value, checkpoint["model_state"][name])
            for name, value in candidate_state.items()
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
