"""Use the shared online feature service for a second, clustering model."""
import pandas as pd
from feast import FeatureStore

NUMERIC_FEATURES = ["sepal_area", "petal_area", "sepal_to_petal_length_ratio"]


def main() -> pd.Series:
    from scipy.cluster.vq import kmeans2

    source_df = pd.read_parquet("data/iris_features.parquet")
    sample_ids = source_df["sample_id"].astype(int).tolist()
    store = FeatureStore(repo_path=".")
    feature_service = store.get_feature_service("iris_feature_service")
    online_features = store.get_online_features(
        features=feature_service,
        entity_rows=[{"sample_id": sample_id} for sample_id in sample_ids],
    ).to_df()
    if online_features[NUMERIC_FEATURES].isnull().any().any():
        raise RuntimeError("Feature service returned missing values for clustering")

    _, clusters = kmeans2(
        online_features[NUMERIC_FEATURES].to_numpy(dtype=float),
        3,
        minit="++",
        rng=42,
    )
    counts = pd.Series(clusters, name="cluster").value_counts().sort_index()
    print("Cluster counts using Feast-registered features:")
    print(counts.to_string())
    return counts


if __name__ == "__main__":
    main()