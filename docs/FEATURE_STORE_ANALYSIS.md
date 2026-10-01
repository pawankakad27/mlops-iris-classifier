# Experiment 5: Feature Store Analysis

The Feast repository under `practical5_feast/iris_feature_repo/feature_repo` registers one `sample_id` entity, two feature views (`iris_measurements` and `iris_engineered_features`), and the `iris_feature_service`. `prepare_feature_source.py` adapts Experiment 4's validated CSV into timestamped Parquet; the target column is deliberately excluded from both feature views.

## Observed benefits

- **Consistent feature access:** Training-time historical retrieval and online inference request the same registered views and fields. This reduces opportunities for consumers to select or name features differently.
- **Reuse:** The clustering example retrieves numeric engineered features through `iris_feature_service`; it does not independently recalculate sepal area, petal area, or the length ratio.
- **Centralized schema:** Entity keys, field names, types, TTLs, and the reusable service are defined in one version-controlled `features.py` file.
- **Point-in-time retrieval:** The historical example joins each entity row at its own event timestamp rather than using the latest online values for all training rows.

Feast does not itself compute the Experiment 4 transformations in this setup: those values are produced upstream and loaded from the validated CSV. The feature store centralizes their schema and retrieval. Keeping that upstream transformation pipeline as the single computation path is still necessary to prevent training-serving skew; a feature store alone cannot guarantee identical computations if producers independently reimplement them.

## Verification

From `practical5_feast/iris_feature_repo/feature_repo`, run `feast apply`, materialize the data with `feast materialize-incremental`, then run the three example scripts. The online script checks three entities for missing values. The historical script checks that all 169 source rows and all eight requested features are present. The clustering script retrieves the registered feature service and reports three KMeans cluster counts. Registry and online SQLite databases are local runtime artifacts and are not committed.