# Cove compatibility ranker — model card

## Model and intended use

Cove ranks complete four-slot outfits consisting of an inner top, optional
outer layer, bottom and shoes. A frozen `google/siglip-base-patch16-224` image
encoder produces item vectors. A 379,265-parameter neural compatibility head
learns pooled item and pairwise slot relationships. Weather constraints are
applied separately as explicit rules. The system is a low-stakes outfit aid,
not an authority on taste, identity, suitability, safety or purchasing.

## Training and evaluation data

The head uses secondary Polyvore data only. Each retained outfit creates a
positive example and a negative example by replacing one item with another
item from the same broad slot. Train, validation and test contain 7,756, 354
and 714 examples respectively. Candidate products come from a separately
published Zara products dataset; no live retailer scraping is performed by
this project.

## Imbalance mitigation

Labels are exactly balanced, so class-level `pos_weight` is not used. Changed
slots and three/four-piece outfits are not balanced. Training therefore uses
smoothed inverse-frequency weights for the joint `(changed slot, outfit
length)` group. The square root limits rare-group amplification and weights
are capped at 3 before mean-one normalisation. Configuration and checkpoint
selection use validation data only. The deployment gate requires:

- overall AUC loss no greater than 0.01;
- macro-slot AUC loss no greater than 0.002;
- improved worst-slot AUC; and
- a reduced gap between best and worst slot AUC.

## Evaluation boundaries

Project FITB-4 uses one true completion and three same-slot replacements. It
is not the official Polyvore FITB benchmark. The available secondary image
archive retained none of the 3,076 official questions with all required,
supported answer images, so no official FITB score is claimed.

## Known biases and limitations

- Polyvore reflects its platform's historic users and does not define
  objective or universal fashion taste.
- Frozen SigLIP features can inherit cultural, gender-presentation, body,
  brand, photography and web-data biases.
- No demographic attributes are collected, so demographic parity cannot be
  measured or claimed. Slot performance is an operational data slice, not a
  substitute for human-group fairness.
- Polyvore and catalogue feature distributions are strongly separable. The
  recorded domain-classifier cross-validated AUC is 1.0; cross-domain
  generalisation claims are therefore inappropriate without target-domain
  labels.
- Random same-slot replacements can be false negatives because a replacement
  may remain compatible. The model learns dataset compatibility patterns, not
  a universal aesthetic truth.
- Three-piece outfits outnumber four-piece outfits, and outer-layer decisions
  remain less reliable than other slots even after mitigation.
- Model scores are ranking signals and must not be presented as calibrated
  probabilities of whether an outfit is objectively good. Scalar temperature
  scaling fitted only on validation data reduces score overconfidence without
  changing recommendation order.

## Product safeguards

The application keeps the user's selected item fixed, hard-filters weather
constraints, offers multiple model-ranked alternatives, preserves colour
diversity, and falls back to a deterministic rules model if trained artifacts
are unavailable. Brand and collection copy is removed from generated garment
labels. Users retain the final decision and can replace or omit any suggested
slot.

## Reproducibility

Seeds, dependency versions, file hashes, per-run checkpoints, validation
histories, group metrics, error slices and domain-shift diagnostics are stored
with the project. Unit tests cover tensor shape, paired-group recovery,
weighting, validation-only selection and deployment gates.

## Open-source engineering references

The evaluation structure adapts practices from actively maintained projects
verified on 20 August 2026: `recommenders-team/recommenders` (controlled
recommendation experiments and operational tests), `fairlearn/fairlearn`
(performance slices, disparity reporting and mitigation as a sociotechnical
process), and `huggingface/lighteval` (reproducible configurations and
sample-level diagnostics). These are engineering references, not benchmark
systems whose published scores are compared with Cove.
