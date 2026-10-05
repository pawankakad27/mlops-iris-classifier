"""Run a random search over a Random Forest and log each sampled trial."""

import os
from typing import Dict, List

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from baseline_model import EXPERIMENT_NAME, get_tracking_uri, load_dataset

PARAM_DISTRIBUTIONS = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}


def run_random_search(data_path: str = "data/processed/iris_features.csv", n_iter: int = 12) -> pd.DataFrame:
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_dataset(data_path)
    estimator = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=5,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    results = []
    for index, (params, score) in enumerate(zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])):
        with mlflow.start_run(run_name=f"random_trial_{index + 1}") as run:
            mlflow.log_params({k: str(v) if v is None else v for k, v in params.items()})
            mlflow.log_param("search_method", "random")
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("cv_folds", 5)
            mlflow.log_metric("mean_test_f1_macro", float(score))
            results.append(
                {
                    "run_id": run.info.run_id,
                    "run_name": run.data.tags.get("mlflow.runName", f"random_trial_{index + 1}"),
                    **params,
                    "mean_test_f1_macro": float(score),
                }
            )

    results_df = pd.DataFrame(results).sort_values("mean_test_f1_macro", ascending=False)
    print("Best random-search configuration:")
    print(results_df.head(1).to_string(index=False))
    return results_df


def main() -> None:
    run_random_search()


if __name__ == "__main__":
    main()
