"""Build the timestamped Parquet source consumed by Feast."""
from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
    "petal_length_bin",
]
REPOSITORY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPOSITORY_DIR.parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "iris_features.csv"
OUTPUT_FILE = REPOSITORY_DIR / "data" / "iris_features.parquet"


def prepare_feature_source() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)
    missing_columns = set(FEATURE_COLUMNS + ["species"]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Experiment 4 output is missing columns: {sorted(missing_columns)}")
    if df.empty or df[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("Experiment 4 feature data must be non-empty and contain no null features")

    df.insert(0, "sample_id", range(len(df)))
    timestamp_end = pd.Timestamp.now(tz="UTC").floor("min")
    df["event_timestamp"] = pd.date_range(
        end=timestamp_end,
        periods=len(df),
        freq="min",
    )
    df["created_timestamp"] = df["event_timestamp"]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    prepare_feature_source()