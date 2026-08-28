# Data provenance and redistribution status

## Maryland Polyvore

- Metadata source: Xintong Han et al.'s `xthan/polyvore-dataset` repository.
- Local source snapshot: `data/polyvore_source/` (not versioned because the
  downloaded dataset is intentionally excluded from Git).
- Repository licence: Apache License 2.0; a complete `LICENSE` file is present
  in the local source snapshot.
- Required scholarly attribution: Han, X., Wu, Z., Jiang, Y.-G. and Davis,
  L.S. (2017), *Learning Fashion Compatibility with Bidirectional LSTMs*, ACM
  Multimedia. https://doi.org/10.1145/3123266.3123391
- Image provenance caveat: the official repository states that the original
  Polyvore image URLs are dead and points to an **unofficial** Kaggle image
  mirror. The repository-level Apache licence is documented, but the project
  has not established an item-by-item copyright licence for the underlying
  third-party product photographs.

Consequently, raw Polyvore images, archives, caches and trained artefacts are
kept local and excluded from Git. They must not be redistributed with the
public code repository. Any dissertation reproduction of example product
images requires a separate university copyright decision; the safer default is
to use native diagrams and aggregate results instead.

This record documents provenance and risk controls; it is not legal advice and
does not treat an ethics waiver as copyright permission.
