"""Produce a transparent operational-bias audit for the recommendation model."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from recommendation_training.balance import infer_pair_groups
from recommendation_training.dataset import OutfitDataset, SLOTS


def dataset_group_counts(dataset):
    try:
        groups = infer_pair_groups(dataset)
    except ValueError as error:
        present_counts = Counter(
            int(value) for value in dataset.masks.sum(dim=1).tolist()
        )
        label_counts = Counter(int(value) for value in dataset.labels.tolist())
        return {
            "examples": len(dataset),
            "positive_rate": dataset.positive_rate,
            "examples_by_outfit_length": {
                str(key): int(value) for key, value in sorted(present_counts.items())
            },
            "examples_by_label": {
                str(key): int(value) for key, value in sorted(label_counts.items())
            },
            "changed_slot_diagnostics": "unavailable_for_unpaired_published_rows",
            "unavailable_reason": str(error),
        }
    joint = Counter(
        (SLOTS[group["slot"]], group["present_slots"]) for group in groups
    )
    return {
        "examples": len(dataset),
        "positive_rate": dataset.positive_rate,
        "pairs_by_changed_slot_and_outfit_length": {
            f"{slot}|{length}": count
            for (slot, length), count in sorted(joint.items())
        },
    }


def catalogue_audit(path):
    frame = pd.read_csv(path)
    by_slot = frame.groupby("slot").size().to_dict()
    by_section = frame.groupby(["slot", "section"]).size().to_dict()
    by_colour = frame["colour"].value_counts().to_dict()
    total = len(frame)
    return {
        "products": total,
        "by_slot": {str(key): int(value) for key, value in by_slot.items()},
        "by_slot_and_published_section": {
            f"{slot}|{section}": int(value)
            for (slot, section), value in by_section.items()
        },
        "by_inferred_colour": {
            str(key): int(value) for key, value in by_colour.items()
        },
        "largest_colour_share": (
            float(max(by_colour.values()) / total) if total else None
        ),
        "slot_count_gap": int(max(by_slot.values()) - min(by_slot.values())),
        "section_balance_note": (
            "Published MAN/WOMAN sections are catalogue-source metadata, not "
            "user identity labels and are not used by the compatibility model."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.pt")
    parser.add_argument("--validation", default="data/processed/validation.pt")
    parser.add_argument("--test", default="data/processed/test.pt")
    parser.add_argument("--catalogue", default="data/catalogue.csv")
    parser.add_argument("--domain-shift", default="results/domain_shift.json")
    parser.add_argument("--imbalance-study", default="results/imbalance_mitigation_study.json")
    parser.add_argument("--output", default="results/bias_audit.json")
    args = parser.parse_args()
    domain_shift = json.loads(Path(args.domain_shift).read_text(encoding="utf-8"))
    imbalance = json.loads(Path(args.imbalance_study).read_text(encoding="utf-8"))
    selection = imbalance["selection"]
    chosen = selection["chosen_configuration"]
    deployed = selection.get("deployed_configuration", chosen)
    published_unpaired = "changed_slot_note" in imbalance.get("protocol", {})
    report = {
        "scope": (
            "Operational data and performance slices. No demographic attributes "
            "are collected, so demographic fairness is neither measured nor claimed."
        ),
        "datasets": {
            split: dataset_group_counts(OutfitDataset(path))
            for split, path in {
                "train": args.train,
                "validation": args.validation,
                "test": args.test,
            }.items()
        },
        "catalogue": catalogue_audit(args.catalogue),
        "mitigation_result": {
            "chosen_configuration": chosen,
            "deployed_configuration": deployed,
            "deployment_gate": selection["deployment_gate"],
            "summary": imbalance["summary"][deployed],
        },
        "unresolved_domain_shift": domain_shift["overall"],
        "risk_register": [
            {
                "risk": "outfit-length imbalance",
                "status": "measured_no_safe_weighting_gain" if published_unpaired else "mitigated_not_eliminated",
                "control": (
                    "Three/four-item slices and a validation-only weighting study; "
                    "unweighted BCE retained because the candidate reduced the worst-group AUC."
                    if published_unpaired
                    else "capped square-root inverse-frequency weighted BCE"
                ),
            },
            {
                "risk": "changed-slot imbalance",
                "status": "unavailable_in_published_rows" if published_unpaired else "measured",
                "control": (
                    "No changed-slot fairness claim because official compatibility "
                    "rows do not expose defensible adjacent replacement pairs."
                    if published_unpaired
                    else "paired changed-slot diagnostics"
                ),
            },
            {
                "risk": "candidate slot and published-section imbalance",
                "status": "controlled",
                "control": "75 products per slot and near-even MAN/WOMAN source sections",
            },
            {
                "risk": "candidate colour concentration",
                "status": "mitigated_not_eliminated",
                "control": "colour-diverse shortlist and distinct-colour alternatives",
            },
            {
                "risk": "Polyvore-to-catalogue domain shift",
                "status": "unresolved_measured_limitation",
                "control": "no cross-domain accuracy claim; rules fallback and user alternatives",
            },
            {
                "risk": "SigLIP and Polyvore cultural/demographic representation",
                "status": "unresolved_unmeasurable_without_attributes",
                "control": "no objective-taste claim, no protected-attribute inference, user control",
            },
            {
                "risk": "plausible false negatives from item replacement",
                "status": "unresolved_label_noise",
                "control": (
                    "Published same-type hard negatives, error analysis and an "
                    "explicit limitation; no claim that every negative is aesthetically invalid."
                    if published_unpaired
                    else "ranking metrics, error analysis, explicit limitation"
                ),
            },
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved operational bias audit to {output_path}")


if __name__ == "__main__":
    main()
