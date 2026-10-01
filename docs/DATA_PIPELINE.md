# Iris Data Pipeline

The pipeline turns the DVC-versioned source dataset into validated, feature-rich data. Run the complete workflow with `dvc repro`; DVC follows the dependencies below and reuses stages whose inputs have not changed.

## Pipeline stages

| Stage | Purpose | Inputs | Outputs |
|---|---|---|---|
| Collect | Read the versioned raw dataset and map numeric targets to species names | `data/raw/iris_v1.csv`, `src/pipeline/collect.py` | `data/raw/iris_raw.csv` |
| Preprocess | Drop duplicate rows, coerce measurements to numeric values, impute missing measurements with the median, and drop rows without a species | `data/raw/iris_raw.csv`, `src/pipeline/preprocess.py` | `data/processed/iris_preprocessed.csv` |
| Features | Add sepal area, petal area, sepal-to-petal length ratio, and petal-length category | `data/processed/iris_preprocessed.csv`, `src/pipeline/features.py` | `data/processed/iris_features.csv` |
| Validate | Reject missing columns, empty datasets, null values, unexpected species, and measurements outside configured ranges | `data/processed/iris_features.csv`, `src/pipeline/validate.py` | Validation result; exits non-zero on failure |

## Dependency graph

```text
data/raw/iris_v1.csv + collect.py
              |
              v
           collect
              |
              v
         iris_raw.csv + preprocess.py
              |
              v
          preprocess
              |
              v
 iris_preprocessed.csv + features.py
              |
              v
           features
              |
              v
    iris_features.csv + validate.py
              |
              v
           validate
```

The collected CSV intentionally excludes a run-time timestamp so identical source data produces identical output and DVC can cache the collection stage. Generated CSV files are ignored by Git and managed as DVC pipeline outputs; `dvc.yaml` and the generated `dvc.lock` belong in Git.

## Run and verify

```bash
dvc repro
dvc repro
dvc dag
```

The second `dvc repro` should report that the stages are unchanged. To check the hard-failure behavior, change a measurement in `data/processed/iris_features.csv` to an out-of-range value and run `python src/pipeline/validate.py`; validation should exit with status 1. Restore the file with `dvc checkout` or rerun `dvc repro` afterward.