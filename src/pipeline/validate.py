"""Stage 4: Enforce schema and statistical checks before training."""
import argparse
import logging
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate")

EXPECTED_COLUMNS = {
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "species",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
    "petal_length_bin",
}
VALID_SPECIES = {"setosa", "versicolor", "virginica"}
RANGE_CHECKS = {
    "sepal length (cm)": (3.0, 9.0),
    "sepal width (cm)": (1.5, 5.5),
    "petal length (cm)": (0.5, 8.0),
    "petal width (cm)": (0.05, 3.0),
}


class DataValidationError(Exception):
    """Raised when data fails one or more pipeline validation checks."""


def validate(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    errors = []

    missing_columns = EXPECTED_COLUMNS - set(df.columns)
    if missing_columns:
        errors.append(f"Missing expected columns: {sorted(missing_columns)}")
    if df.empty:
        errors.append("Dataset contains no rows")

    null_columns = df.columns[df.isnull().any()].tolist()
    if null_columns:
        errors.append(f"Unexpected null values in columns: {null_columns}")

    if "species" in df.columns:
        invalid_species = set(df["species"].dropna().unique()) - VALID_SPECIES
        if invalid_species:
            errors.append(f"Unexpected species values: {sorted(invalid_species)}")

    for column, (low, high) in RANGE_CHECKS.items():
        if column not in df.columns:
            continue
        out_of_range = df[column].dropna().loc[lambda values: (values < low) | (values > high)]
        if not out_of_range.empty:
            errors.append(
                f"{len(out_of_range)} values out of expected range for '{column}' ({low}-{high})"
            )

    if errors:
        for error in errors:
            logger.error(error)
        raise DataValidationError(f"Validation failed with {len(errors)} error(s)")

    logger.info(
        "Validation PASSED: %d rows, %d columns, all checks satisfied",
        len(df),
        df.shape[1],
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/iris_features.csv")
    args = parser.parse_args()
    try:
        validate(args.input)
    except DataValidationError as error:
        logger.error("Pipeline halted: %s", error)
        sys.exit(1)