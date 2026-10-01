"""Retrieve an offline, point-in-time-correct training dataset."""
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
    source_df = pd.read_parquet("data/iris_features.parquet")
    entity_df = source_df[["sample_id", "event_timestamp"]].copy()
    store = FeatureStore(repo_path=".")
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURES,
    ).to_df()
    feature_columns = [column.split(":", maxsplit=1)[-1] for column in FEATURES]
    missing_columns = set(feature_columns) - set(training_df.columns)
    if missing_columns:
        raise RuntimeError(f"Historical retrieval omitted features: {sorted(missing_columns)}")
    if len(training_df) != len(source_df):
        raise RuntimeError(
            f"Expected {len(source_df)} historical rows, received {len(training_df)}"
        )
    if training_df[feature_columns].isnull().any().any():
        raise RuntimeError("Historical feature retrieval returned missing feature values")
    print(f"Retrieved {len(training_df)} point-in-time feature rows:")
    print(training_df.head().to_string(index=False))
    return training_df


if __name__ == "__main__":
    main()