# Trainable compatibility recommendation

This folder implements the learned recommendation path. Garment recognition
and hard weather constraints remain separate from compatibility learning.

## Data flow

1. Official Polyvore outfits are positive examples.
2. One present garment is replaced by an item from the same broad slot to make
   a category-controlled random negative without introducing an obvious slot error.
3. Frozen `google/siglip-base-patch16-224` image vectors are cached once.
4. `CompatibilityRanker` learns a probability for a masked four-slot outfit.
5. The epoch with the highest validation AUC is saved.
6. Brand-free abstract garment prototypes derived from the Polyvore training
   split are searched exactly after hard weather constraints; the earlier Zara
   catalogue remains a documented legacy/domain-shift artifact, not the user-
   visible recommendation source.

## Abstract recommendation search

The product does not present Polyvore or Zara records as products a user owns
or should buy. Learned recommendations are deliberately abstract: the UI
shows only a filled icon with a broad garment type and colour, while genuine
user wardrobe items retain their own photographs.

Build the compact search space from the **Polyvore training split only**:

```powershell
$env:HF_HUB_OFFLINE='1'
.\.venv\Scripts\python.exe -m recommendation_training.build_abstract_prototypes
```

Every retained item is classified into a slot, broad type, colour, primary
style and catalogue clothing range (`menswear` or `womenswear`) with frozen
SigLIP vectors. Each distinct state is represented by the
observed item embedding nearest its group centroid (a medoid). The resulting
artifact intentionally contains no source photograph, product title, brand,
URL or price. Support frequency is recorded as evidence but is not used as a
hard filter, so low-frequency abstract states remain searchable.

At runtime the compatibility head projects each unique prototype once, streams
the complete Cartesian product in bounded batches, and retains the global
top-k in a heap. The safety limit raises an error instead of returning a prefix
when exact enumeration is too large. This replaces the previous ten-per-slot
quota and 1,000-combination prefix cutoff.

Benchmark the full abstract search with:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.benchmark_abstract_search
```

The machine-readable build and runtime evidence is written to
`results/abstract_prototype_build.json` and
`results/abstract_search_benchmark.json`. The current local CPU benchmark
scores approximately 204,624--350,784 complete Casual outfits per query; the
median across four possible fixed slots and three repetitions is about 4.05
seconds. These timings are machine-specific and must not be presented as a
general latency guarantee.

### Cove v3 system comparison

Version 3.0 now deploys the gated D2 disjoint checkpoint and keeps the exact
candidate/search layer. A controlled legacy replay uses the same D2 head, so
its classification metrics are identical by design; claiming an AUC
improvement from exact search would be methodologically incorrect. The valid comparison isolates the
old ten-per-slot quota and the exact search on the same abstract candidate
space, using held-out positive embeddings from the disjoint test split:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.evaluate_v3_system
.\.venv\Scripts\python.exe -m recommendation_training.v3_release_check
```

The report includes global-top-one recovery, top-ten recall, model-objective
regret with a bootstrap interval, candidate-state coverage, privacy fields,
weather/style coverage, deterministic repeats and same-machine latency for the
v1 rule ranker, v2 Zara quota path, the quota ablation and v3 exact search.
Absolute model scores from Zara and abstract candidates are not treated as a
quality comparison because the two candidate distributions differ.

The current 18-query CPU experiment found that the quota path examined about
3.41% of the abstract combination space, recovered the exact top result in
27.78% of queries and recalled 24.44% of the exact top ten. Exact search had a
median latency of about 0.47 seconds across all nine styles (p95 about 2.11
seconds). These results measure optimisation of the deployed model objective,
not human aesthetic preference.

## D2 disjoint production head

The stronger research candidate uses the Polyvore Outfits **disjoint** split
and the published compatibility labels rather than locally generated random
negatives.  It is prepared under `data/processed_disjoint/`, while its
research checkpoints remain under `models/disjoint/`; the gated D2 checkpoint
is stored under `models/d2/` and promoted explicitly.

The full candidate release chain includes validation-only scalar temperature
calibration, three-seed architecture comparisons, structural ablations,
three/four-item length weighting, official FITB subset evaluation, error
analysis, domain-shift measurement, runtime benchmarking and an explicit
release gate. Run the final gate with:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.disjoint_release_check
```

The original FITB evaluator incorrectly assumed that the correct disjoint
answer was always first. Disjoint answers are shuffled; the correct answer is
the candidate whose `set_id` matches the question outfit. After correcting
that defect, the proposed lightweight architecture reaches 59.94% mean FITB
accuracy across three seeds. The freshly trained and validation-calibrated D2
checkpoint reaches AUC 0.8583 and FITB 60.25%, passes the recorded gate and is
now the production `models/compatibility_ranker.pt`.

The corrected structural FITB ablation uses the same 4,468 retained questions
and the same three seeds. Removing the pairwise module reduces mean FITB from
59.94% to 49.66%; the paired question-bootstrap interval for the full-minus-
ablation difference is +8.96 to +11.57 percentage points. Removing the slot
embedding produces 59.47%; the corresponding interval is -0.13 to +1.06
points, so the slot embedding's small positive estimate is **not** claimed as
clear evidence of benefit. Reproduce this result with:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.evaluate_d2_ablations
```

The UI asks users to select a clothing range rather than inferring gender from
an image. Candidate filtering is hard: menswear and womenswear prototypes are
never mixed. Only 34 positive and 34 negative pure-menswear source rows exist
in the disjoint training file, so a separate reliable menswear head is not
claimed; the shared head is used with an explicit limitation.

The correction follows the public Polyvore annotation contract documented by
[OpenMMLab MMFashion](https://github.com/open-mmlab/mmfashion/blob/master/docs/dataset/FASHION_COMPATIBILITY_DATASET.md):
the correct FITB answer is identified by matching the question outfit's
`set_id`. The experiment also keeps compatibility prediction and FITB as
separate reported tasks, consistent with the official
[type-aware compatibility implementation](https://github.com/mvasil/fashion-compatibility)
and the 100+ star
[context-aware compatibility implementation](https://github.com/gcucurull/visual-compatibility).

The official metadata is from the
[Maryland Polyvore repository](https://github.com/xthan/polyvore-dataset). The
original image URLs no longer work; the repository points to the
[Maryland Polyvore Images mirror](https://www.kaggle.com/datasets/dnepozitek/maryland-polyvore-images).

## Prepare Polyvore tensors

Download and extract the Kaggle images so the layout contains
`<images-dir>/<set_id>/<item_index>.jpg`, then run:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.prepare_polyvore `
  --metadata-dir data/polyvore `
  --images-dir C:\path\to\maryland-polyvore-images `
  --output-dir data/processed
```

Use `--maximum-outfits 100` for a quick pipeline check before processing the
full dataset. `polyvore_item_embeddings.pt` is saved incrementally and reused,
so an interrupted CPU embedding run can continue without starting over. Use
`--cache-save-every 1` for the safest smoke run or keep the default of 10
batches for the full dataset.

## Train

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.train `
  --train data/processed/train.pt `
  --validation data/processed/validation.pt `
  --output models/compatibility_ranker.pt
```

Training uses validation AUC for checkpoint selection, records the best epoch
and reproducibility settings inside the checkpoint, writes the complete epoch
history to `models/compatibility_ranker.training.json`, and stops early after
four epochs without improvement by default.

## Held-out evaluation

Do not select the model on the test split. After training is complete, run the
test evaluation once:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.evaluate `
  --checkpoint models/compatibility_ranker.pt `
  --test data/processed/test.pt `
  --output results/compatibility_test_metrics.json
```

The report contains ROC AUC with a deterministic bootstrap 95% confidence
interval, average precision, accuracy, balanced accuracy, precision, recall,
specificity, F1, Matthews correlation coefficient, Brier score, the confusion
matrix, checkpoint metadata, and dataset metadata. This JSON is intended to be the
source for dissertation result tables; the structural smoke test is not a
model-quality result.

Compare the learned network with chance ranking and a zero-training SigLIP
pairwise-cosine baseline using:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.compare_baselines `
  --test data/processed/test.pt `
  --output results/baseline_comparison.json
```

## Controlled baselines and ablations

Run a mean-pooled frozen-SigLIP logistic baseline together with no-slot,
no-pairwise and fully retrained neural variants:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.ablation_study
```

The experiment uses identical frozen embeddings, data splits, random seed and
neural training settings. It writes `results/ablation_study.json` and keeps its
experimental checkpoints under `models/ablations/`; it never overwrites the
production recommendation checkpoint.

Measure the trainable parameter count, artifact sizes, CPU head latency and
warm end-to-end catalogue-ranking latency with:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.benchmark_runtime
```

## Controlled comparison with fashion-model families

The dissertation comparison uses representative architecture families from
fashion compatibility research, adapted to the same cached tensors:

- a slot-ordered Bi-LSTM, motivated by Han et al. (2017);
- slot-pair-specific projections, motivated by the type-aware compatibility
  spaces of Vasileva et al. (2018);
- a small outfit-token Transformer, motivated by Sarkar et al. (2023); and
- the proposed lightweight pooled/pair-difference model.

Run all four under the same split, frozen SigLIP features, optimiser, training
budget, early-stopping rule and completion candidates:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.fashion_model_comparison
```

The output `results/fashion_model_comparison.json` contains three-seed results,
per-run predictions, compatibility AUC, matched-pair ranking accuracy, a fixed
four-choice completion diagnostic, trainable parameter counts and paired
bootstrap AUC-difference intervals. The comparison models preserve the central
architectural idea of each research family but are not exact reproductions of
the published systems. Published benchmark scores must therefore be presented
separately and must not be compared numerically with this processed split.

Primary methodological sources:

- Han et al., *Learning Fashion Compatibility with Bidirectional LSTMs*, ACM
  Multimedia 2017: https://arxiv.org/abs/1707.05691
- Vasileva et al., *Learning Type-Aware Embeddings for Fashion Compatibility*,
  ECCV 2018: https://arxiv.org/abs/1803.09196
- Sarkar et al., *OutfitTransformer: Learning Outfit Representations for
  Fashion Recommendation*, WACV 2023: https://arxiv.org/abs/2204.04812

The four-choice metric is deliberately called a project-specific FITB-4
diagnostic. It uses one true completion, the matched same-slot corruption and
two deterministic replacements from other test pairs. It is reproducible and
fair across the four local models, but it is not the official Polyvore FITB
protocol and may contain plausible false negatives.

## Official FITB subset and diagnostics

The official disjoint `fill_in_blank_test.json` is evaluated without scraping
additional sources. With the locally supplied secondary-data archive, 4,468
of 15,145 questions are representable by the product's four slots. The result
therefore carries a 29.50% coverage warning and is not numerically
interchangeable with a published full-dataset FITB score. Evaluate all
comparison checkpoints with explicit disjoint paths:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.evaluate_official_fitb `
  --metadata data/polyvore_disjoint/test.json `
  --item-metadata data/polyvore_nondisjoint/polyvore_item_metadata.json `
  --questions data/polyvore_disjoint/fill_in_blank_test.json `
  --cache data/processed_disjoint/fitb_item_embeddings.pt `
  --checkpoint-directory models/disjoint_fashion_model_comparison `
  --output results/disjoint_official_fitb_comparison.json
```

Correct answers are determined by matching answer `set_id` to the question
outfit because disjoint answer order is shuffled. Coverage and every exclusion
reason are written alongside the scores.

Generate structural error slices and a frozen-feature domain-shift diagnostic:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.error_analysis
.\.venv\Scripts\python.exe -m recommendation_training.domain_shift
```

`error_analysis` reports calibration, three-versus-four-slot performance,
matched ranking by replaced slot and the highest-confidence errors.
`domain_shift` compares real positive Polyvore test items with the 300-product
candidate catalogue using centroid similarity, nearest-neighbour similarity,
RBF MMD and a cross-validated domain classifier.

An explicit matched-pair margin-ranking objective was also tested without
changing the architecture:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.ranking_objective_study
```

It did not improve held-out AUC, matched-pair accuracy or FITB-4 over the
standard BCE-trained model, so the production checkpoint remains unchanged.

Capture the exact Python/package versions, Git revision and SHA-256 hashes of
the datasets and trained artifacts used for a result:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.reproducibility_manifest
```

Before freezing dissertation results, run the combined evidence and software
audit:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.research_readiness_check --run-tests
```

## Published Maryland hard negatives

`prepare_hardneg.py` converts the published same-type replacement benchmark
into the same four-slot tensor format and reuses the SigLIP cache:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.prepare_hardneg `
  --metadata-dir data/polyvore `
  --hardneg-dir data/polyvore_hardneg `
  --images-dir C:\path\to\maryland-polyvore-images `
  --splits test
```

The Hugging Face root `images.zip` contains the larger Polyvore Outfits image
collection, not every image referenced by the Maryland hard-negative split.
A report built from only the overlapping images must be labelled as a partial,
selection-biased diagnostic rather than the full hard-negative benchmark.

## Prepare the Zara candidate catalogue

Place the published secondary-data archive at `data/incoming/archive.zip`.
The preparation command reads its bundled metadata and images locally; it
does not request Zara pages or scrape current product data. The production
configuration selects a balanced 300-product catalogue (75 per supported
slot), prioritises genuine wet-weather claims for outerwear and shoes,
extracts one representative image per product, and assigns colour/style labels
with the frozen SigLIP encoder.

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.prepare_zara_catalogue `
  --archive data/incoming/archive.zip `
  --catalogue data/catalogue.csv `
  --per-slot 75
```

The generated catalogue contains the required `item_id`, `slot`, `type`,
`colour`, and `image_path` fields plus weather and source metadata. Cache its
image vectors with:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.embed_catalogue `
  --catalogue data/catalogue.csv `
  --output models/catalogue_embeddings.pt
```

When both `.pt` files exist, `recommend_outfit()` uses the trained model.
Missing/corrupt artifacts or inference errors automatically use the
interpretable baseline and record the fallback reason in `result["model"]`.
After weather/style filtering, the runtime compares the uploaded garment with
all remaining products, creates an input-dependent colour-diverse shortlist,
and exhaustively scores complete outfits from that shortlist.

## Imbalance mitigation and model card

Run the controlled group-weighting study with:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.imbalance_mitigation_study
```

The study balances joint changed-slot/outfit-length groups with capped,
smoothed inverse-frequency weights. It selects configurations on validation
data only and reports overall, macro-slot, worst-slot and disparity metrics.
Deployment is guarded by explicit performance/fairness equivalence margins.
Intended use, evaluation boundaries and unresolved biases are documented in
`recommendation_training/MODEL_CARD.md`.

Generate the machine-readable operational bias register and catalogue/data
distribution audit with:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.bias_audit
```

For official unpaired compatibility rows, changed-slot labels cannot be
reconstructed defensibly. Use the separate length-only study instead:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.disjoint_imbalance_study
```
