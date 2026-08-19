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
