# Cove — Your wardrobe

Cove is a local-first Streamlit dissertation prototype that recognises one
garment from a circled photo, combines it with live weather, and ranks complete
four-slot outfits. It also provides a private local wardrobe for saving,
editing and reusing individual pieces and partial or complete outfits.

## Current scope

The implemented system contains:

- SigLIP-based garment category, colour and style recognition;
- a lightweight neural outfit-compatibility ranker trained on secondary
  Polyvore data;
- hard weather constraints and an interpretable rules fallback;
- model-ranked colour-diverse alternatives for every recommended slot;
- a local wardrobe with reusable pieces, groups, filters and partial outfits;
- controlled architecture baselines, ablations, multi-seed evaluation,
  calibration, imbalance mitigation, error slices and a bias audit.

Automatic long-term learning from an individual user's clicks is deliberately
out of scope for this version. The wardrobe is persistent, but it does not
update the recommendation model. This avoids claiming personalisation without
an exposure-aware feedback design and adequate evaluation.

## Recommendation pipeline

```text
Circled garment image
        |
Frozen SigLIP recognition and image embedding
        |
Weather and supported-slot constraints
        |
Compatibility model scores complete catalogue outfits
        |
Primary outfit + model-ranked colour-diverse alternative
        |
Optional save to the local wardrobe
```

When both `models/compatibility_ranker.pt` and
`models/catalogue_embeddings.pt` exist, the trained model is used. If either
artifact or the uploaded embedding is unavailable, Cove automatically uses the
deterministic `lightweight_content_ranker_v1` fallback and records the reason.

## Reproduced environment

Python 3.12 is recommended. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-reproducible.txt
```

`requirements.txt` contains unpinned direct dependencies for development.
`requirements-reproducible.txt` records the direct versions used for the final
evaluation. Exact runtime versions and artifact SHA-256 hashes are also stored
in `results/reproducibility_manifest.json`.

## Run the application

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

The current workspace contains the local trained artifacts. They are ignored
by Git because they are derived from separately obtained secondary datasets.
A clean clone still runs with the fallback model. To reproduce the trained
path, follow [`recommendation_training/README.md`](recommendation_training/README.md).

## Verify the project

Fast release and artifact check:

```powershell
.\.venv\Scripts\python.exe release_check.py
```

Require the trained checkpoint and embedded catalogue as well:

```powershell
.\.venv\Scripts\python.exe release_check.py --strict-artifacts
```

Full local verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q app.py release_check.py src recommendation_training tests
.\.venv\Scripts\python.exe -m recommendation_training.smoke_test
```

## Final model result

The production checkpoint was selected using validation data only. It uses
capped square-root inverse-frequency weights for changed-slot/outfit-length
groups and scalar temperature calibration.

| Metric | Final result |
|---|---:|
| Test ROC AUC | 0.7447 |
| Average precision | 0.7426 |
| Pair ranking accuracy | 0.7983 |
| Project FITB-4 | 0.5854 |
| Three-piece AUC | 0.7568 |
| Four-piece AUC | 0.7102 |
| Outer-layer AUC | 0.6584 |
| Brier score | 0.2057 |

Project FITB-4 is a deterministic local diagnostic, not the official Polyvore
FITB benchmark. No official score is claimed because the available secondary
image archive does not provide a complete supported answer set.

## Responsible-use boundary

Cove learns dataset compatibility patterns, not an objective definition of
fashion. Polyvore-to-catalogue domain shift remains substantial, and frozen
SigLIP features may inherit cultural, representation and photography biases.
No demographic attributes are collected, so demographic fairness is neither
measured nor claimed. See the
[`MODEL_CARD.md`](recommendation_training/MODEL_CARD.md) and
[`bias_audit.json`](results/bias_audit.json) for the complete boundary.

User wardrobe images and metadata stay in ignored local files under
`data/wardrobe_images/` and `data/wardrobe.json`. Users retain the final choice
and may replace, omit or delete any suggested piece.

## Repository guide

- `app.py`: Streamlit product interface.
- `src/`: recognition, recommendation, weather and wardrobe runtime logic.
- `recommendation_training/`: preparation, training and evaluation pipeline.
- `tests/`: deterministic unit and integration-level logic tests.
- `results/`: versioned compact evaluation evidence.
- `assets/`: local UI assets and explicitly labelled designer concepts.
