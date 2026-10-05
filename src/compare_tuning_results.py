"""Compare the baseline, grid-search, and random-search MLflow runs."""

import os
from typing import Any, Dict, List

import pandas as pd
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "iris-hyperparameter-tuning"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")


def get_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT}/mlflow.db")


def _metric_value(run: Any, *names: str) -> float:
    for name in names:
        if name in run.data.metrics:
            return float(run.data.metrics[name])
    return float("-inf")


def compare_tuning_results(experiment_name: str = EXPERIMENT_NAME) -> pd.DataFrame:
    client = MlflowClient(tracking_uri=get_tracking_uri())
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {experiment_name!r} was not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )

    rows: List[Dict[str, Any]] = []
    for run in runs:
        params = dict(run.data.params)
        rows.append(
            {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", run.info.run_id),
                "search_method": params.get("search_method", "baseline"),
                "model_type": params.get("model_type", "DecisionTreeClassifier"),
                "score": _metric_value(run, "cv_f1_macro_mean", "mean_test_f1_macro", "test_f1_macro"),
                "params": params,
            }
        )

    if not rows:
        raise RuntimeError(f"No MLflow runs found in experiment {experiment_name!r}")

    results = pd.DataFrame(rows).sort_values("score", ascending=False)
    print(results[["run_name", "search_method", "model_type", "score"]].to_string(index=False))
    return results


def main() -> None:
    compare_tuning_results()


if __name__ == "__main__":
    main()
