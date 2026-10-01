"""Stage 1: Collect the versioned raw Iris dataset."""
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("collect")

SPECIES_LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}


def collect_data(source_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(source_path)
    if "target" not in df.columns:
        raise ValueError(f"Source dataset must contain a 'target' column: {source_path}")

    labels = df["target"].map(SPECIES_LABELS)
    unknown_targets = df.loc[df["target"].notna() & labels.isna(), "target"].unique()
    if len(unknown_targets):
        raise ValueError(f"Unexpected target values in source dataset: {unknown_targets.tolist()}")

    df = df.drop(columns="target")
    df["species"] = labels
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Collected %d rows -> %s", len(df), output_path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw/iris_v1.csv")
    parser.add_argument("--output", default="data/raw/iris_raw.csv")
    args = parser.parse_args()
    collect_data(args.source, args.output)