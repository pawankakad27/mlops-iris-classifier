"""Establish a simple Decision Tree baseline and log it with MLflow."""

import os
from typing import Tuple

import mlflow
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]
EXPERIMENT_NAME = "iris-hyperparameter-tuning"


def get_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT}/mlflow.db")


def load_dataset(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(data_path)
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(df["species"]), name="species")
    X = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median())
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def run_baseline(data_path: str = "data/processed/iris_features.csv") -> dict:
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_dataset(data_path)

    with mlflow.start_run(run_name="baseline_decision_tree") as run:
        model = DecisionTreeClassifier(random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        f1_macro = f1_score(y_test, predictions, average="macro")

        mlflow.log_param("model_type", "DecisionTreeClassifier")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_metric("cv_f1_macro_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_f1_macro_std", float(cv_scores.std()))
        mlflow.log_metric("test_accuracy", float(accuracy))
        mlflow.log_metric("test_f1_macro", float(f1_macro))

        summary = {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", "baseline_decision_tree"),
            "model_type": "DecisionTreeClassifier",
            "cv_f1_macro_mean": float(cv_scores.mean()),
            "cv_f1_macro_std": float(cv_scores.std()),
            "test_accuracy": float(accuracy),
            "test_f1_macro": float(f1_macro),
        }
        print(summary)
        return summary


def main() -> None:
    run_baseline()


if __name__ == "__main__":
    main()
