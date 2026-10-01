"""Retrieve the latest registered measurements and engineered features."""
import pandas as pd
from feast import FeatureStore

FEATURES = [
    "iris_measurements:sepal length (cm)",
    "iris_measurements:sepal width (cm)",
    "iris_measurements:petal length (cm)",
    "iris_measurements:petal width (cm)",
    "iris_engineered_features:sepal_area",
    "iris_engineered_features:petal_area",
    "iris_engineered_features:sepal_to_petal_length_ratio",
    "iris_engineered_features:petal_length_bin",
]


def main() -> pd.DataFrame:
    store = FeatureStore(repo_path=".")
    result = store.get_online_features(
        features=FEATURES,
        entity_rows=[{"sample_id": sample_id} for sample_id in range(3)],
    ).to_df()
    feature_columns = [column for column in result.columns if column != "sample_id"]
    if result[feature_columns].isnull().any().any():
        raise RuntimeError("Online feature retrieval returned missing feature values")
    print("Features retrieved for sample_ids 0, 1, and 2:")
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    main()