"""Stage 2: Clean and normalize the collected Iris data."""
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess")

NUMERIC_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    initial_rows = len(df)
    df = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows", initial_rows - len(df))

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        missing_count = df[column].isna().sum()
        if missing_count:
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
            logger.info(
                "Imputed %d missing values in '%s' with median=%.3f",
                missing_count,
                column,
                median_value,
            )

    df = df.dropna(subset=["species"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Preprocessed %d rows -> %s", len(df), output_path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/iris_raw.csv")
    parser.add_argument("--output", default="data/processed/iris_preprocessed.csv")
    args = parser.parse_args()
    preprocess(args.input, args.output)