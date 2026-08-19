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
6. Zara catalogue combinations are ranked, hard weather constraints are
   applied, and a colour-diverse alternative is selected.

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

## Official FITB availability and diagnostics

The official `fill_in_blank_test.json` can be parsed without downloading or
scraping any additional source. Extend a separate cache with locally available
secondary-data images, then evaluate all comparison checkpoints:

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.prepare_official_fitb_cache
.\.venv\Scripts\python.exe -m recommendation_training.evaluate_official_fitb
```

The current image archive does not contain all four answer images for any
question that can be mapped into the product's four slots. The evaluator writes
an explicit `unavailable_with_current_image_archive` result with exclusion
counts rather than silently reporting a selected or incomplete score.

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
