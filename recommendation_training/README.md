# Trainable compatibility recommendation

This folder implements the learned recommendation path. Garment recognition
and hard weather constraints remain separate from compatibility learning.

## Data flow

1. Official Polyvore outfits are positive examples.
2. One present garment is replaced by an item from the same broad slot to make
   a hard negative without introducing an obvious category error.
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
full dataset. Existing `polyvore_item_embeddings.pt` is reused.

## Train

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.train `
  --train data/processed/train.pt `
  --validation data/processed/validation.pt `
  --output models/compatibility_ranker.pt
```

## Prepare the Zara candidate catalogue

Place the published secondary-data archive at `data/incoming/archive.zip`.
The preparation command reads its bundled metadata and images locally; it
does not request Zara pages or scrape current product data. It selects a
balanced 40-product catalogue, extracts one representative image per product,
and assigns colour/style labels with the frozen SigLIP encoder.

```powershell
.\.venv\Scripts\python.exe -m recommendation_training.prepare_zara_catalogue `
  --archive data/incoming/archive.zip `
  --catalogue data/catalogue.csv `
  --per-slot 10
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
