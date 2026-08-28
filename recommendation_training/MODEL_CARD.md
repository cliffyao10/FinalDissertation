# Cove compatibility ranker — model card

## Model and intended use

Cove ranks complete four-slot outfits consisting of an inner top, optional
outer layer, bottom and shoes. A frozen `google/siglip-base-patch16-224` image
encoder produces item vectors. A 379,265-parameter neural compatibility head
learns pooled item and pairwise slot relationships. Weather constraints are
applied separately as explicit rules. The system is a low-stakes outfit aid,
not an authority on taste, identity, suitability, safety or purchasing.

## Training and evaluation data

The deployed D2 head uses published Polyvore Outfits disjoint compatibility
rows: 18,243 train, 3,274 validation and 16,062
test examples. Its labels are balanced and its negatives are the published
same-type hard negatives. Runtime recommendations are abstract garment states
derived only from the Polyvore training split. A separately published Zara
dataset is retained for recognition/domain-shift diagnostics; no live retailer
scraping is performed by this project.

## Abstract search space

The runtime artifact groups training items by slot, broad visual type, colour,
primary style and catalogue clothing range, then stores one observed SigLIP medoid per group. It stores
no source image, product title, brand, URL or price. Polyvore frequency is kept
as support metadata rather than a hard eligibility rule. The UI converts each
winning state into a filled garment icon; only user-owned wardrobe pieces are
shown as photographs.

All feasible states for the selected style and hard weather constraints are
scored in bounded batches. Unique item projections are cached once per query,
and a heap retains the global top-k. A five-million-combination safety gate
fails closed instead of presenting a partial prefix as exact. The measured
local CPU median is approximately 4.05 seconds across 12 complete womenswear Casual-search
runs (204,624--350,784 combinations per run); this is hardware-specific,
not a general performance guarantee.

Across a separate 18-query held-out disjoint-input experiment rotating all
nine styles, exact search had a 0.47-second CPU median and 2.11-second p95. The
old quota search recovered the exact top result in 27.78% of queries and had
24.44% mean recall of the exact top ten while examining 3.41% of combinations.
The mean deployed-model score regret was 0.0179 (bootstrap 95% interval
0.0063--0.0357). This is evidence about search completeness only; it is not a
human-preference effect size.

## Imbalance mitigation

Labels are exactly balanced, so class-level `pos_weight` is not used. Smoothed
inverse-frequency weighting was tested, but its worst-group deployment gate
failed; D2 therefore retains unweighted BCE. Configuration and checkpoint
selection use validation data only. The mitigation gate requires:

- overall AUC loss no greater than 0.01;
- macro-slot AUC loss no greater than 0.002;
- improved worst-slot AUC; and
- a reduced gap between best and worst slot AUC.

## Evaluation boundaries

For D2, 4,468 of 15,145 official disjoint FITB questions can be represented by
the product's four-slot taxonomy. Disjoint answer order is shuffled; the
correct answer is determined by outfit `set_id`, not array position. D2
achieves 60.25% accuracy (bootstrap 95% CI 58.73--61.75), while the proposed
architecture's three-seed mean is 59.94%. This filtered score is not
interchangeable with published full-dataset results.

The corrected official-subset structural ablation finds clear support for the
pairwise module: removing it lowers three-seed mean FITB to 49.66%, with a
paired full-minus-ablation 95% interval of +8.96 to +11.57 percentage points.
Removing the slot embedding gives 59.47%; its full-minus-ablation interval of
-0.13 to +1.06 points crosses zero. The slot embedding is retained as a small,
low-cost slot-identity prior, but a clear empirical benefit is not claimed.

The frozen D2 checkpoint has train, validation and test AUC values of 0.9699,
0.8598 and 0.8583. The train--test gap shows a real overfitting tendency. The
validation--test gap is only 0.00145, and validation AUC peaks at the saved
epoch 6 before declining by about 0.010 by epoch 11. The defensible conclusion
is therefore that validation-only early stopping controlled, but did not erase,
training-set overfitting.

## Known biases and limitations

- Polyvore reflects its platform's historic users and does not define
  objective or universal fashion taste.
- Frozen SigLIP features can inherit cultural, gender-presentation, body,
  brand, photography and web-data biases.
- No demographic attributes are collected, so demographic parity cannot be
  measured or claimed. Slot performance is an operational data slice, not a
  substitute for human-group fairness.
- The user selects `Menswear` or `Womenswear`; the application never infers a
  person's gender from an image. Pure-menswear training contains only 34
  positive and 34 negative rows. D2 therefore uses one shared head and makes
  no claim of equal reliability across clothing ranges.
- Polyvore and the legacy Zara catalogue feature distributions are strongly separable. The
  recorded domain-classifier cross-validated AUC is 1.0; cross-domain
  generalisation claims are therefore inappropriate without target-domain
  labels.
- Random same-slot replacements can be false negatives because a replacement
  may remain compatible. The model learns dataset compatibility patterns, not
  a universal aesthetic truth.
- Three-piece outfits outnumber four-piece outfits, and outer-layer decisions
  remain less reliable than other slots even after mitigation.
- Published disjoint compatibility rows do not expose adjacent replacement
  pairs, so changed-slot fairness and the project FITB-4 diagnostic are not
  claimed for that candidate.
- Model scores are ranking signals and must not be presented as calibrated
  probabilities of whether an outfit is objectively good. Scalar temperature
  scaling fitted only on validation data reduces score overconfidence without
  changing recommendation order.
- Prototype support is highly uneven: the womenswear range contains 307
  singleton states and its largest group has support 940. Exact search prevents
  rare states from being removed by a frequency quota, but medoid quality and
  zero-shot type/colour/style labels remain potential error sources.

## Product safeguards

The application keeps the user's selected item fixed, hard-filters weather
constraints, offers multiple model-ranked alternatives, preserves colour
diversity, and falls back to a deterministic rules model if trained artifacts
are unavailable. Brand and collection copy is removed from generated garment
labels. Users retain the final decision and can replace or omit any suggested
slot.

For weather at or above 28°C, v3 inserts a visible `No Outer Layer` concept
whose embedding is masked from the compatibility head. This uses the same
three-item mask behaviour seen during training rather than assigning a random
vector to the absence of clothing.

## Reproducibility

Seeds, dependency versions, file hashes, per-run checkpoints, validation
histories, group metrics, error slices and domain-shift diagnostics are stored
with the project. Unit tests cover tensor shape, paired-group recovery,
weighting, shuffled FITB labels, clothing-range isolation, paired ablation
uncertainty, validation-only selection and deployment gates.

The machine-readable D2 and v3 release gates prevent promotion when a blocking
criterion fails. D2 passed compatibility, corrected three-seed FITB and
structural-ablation, calibration, candidate-isolation and privacy gates.
Compatibility classification and
item-completion ranking remain distinct evaluation tasks.

## Open-source engineering references

The evaluation structure adapts practices from actively maintained projects
verified on 20 August 2026: `recommenders-team/recommenders` (controlled
recommendation experiments and operational tests), `fairlearn/fairlearn`
(performance slices, disparity reporting and mitigation as a sociotechnical
process), and `huggingface/lighteval` (reproducible configurations and
sample-level diagnostics). These are engineering references, not benchmark
systems whose published scores are compared with Cove.

The abstract-search design was also checked against `facebookresearch/faiss`
(exact versus approximate vector retrieval and clustering trade-offs) and
`google/or-tools` (constraint optimisation). Cove uses neither as a runtime
dependency: its small four-slot abstract lattice remains cheap enough for an
exact batched PyTorch search, while the non-additive neural score does not
provide the admissible branch bounds required for an exact branch-and-bound
claim.
