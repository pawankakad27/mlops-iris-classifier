"""Train and track the Iris baseline models with MLflow."""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "iris_features.csv"
EXPERIMENT_NAME = "iris-classification-baseline"
FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]
TARGET_COLUMN = "species"


def main() -> pd.DataFrame:
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Feature CSV is missing columns: {sorted(missing_columns)}")
    if df.empty or df[TARGET_COLUMN].isna().any():
        raise ValueError("Feature CSV must contain rows and non-null species labels")

    features = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median())
    if features.isna().any().any():
        raise ValueError("Feature CSV contains columns with no usable numeric values")

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df[TARGET_COLUMN])
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=200, C=1.0),
        "random_forest_shallow": RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42,
        ),
        "random_forest_deep": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
        ),
    }

    results = []
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision_macro": precision_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "f1_macro": f1_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
            }

            mlflow.log_param("model_type", model_name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param(
                "target_classes", json.dumps(label_encoder.classes_.tolist())
            )
            if model_name == "logistic_regression":
                mlflow.log_params({"max_iter": model.max_iter, "C": model.C})
            else:
                mlflow.log_params(
                    {
                        "n_estimators": model.n_estimators,
                        "max_depth": model.max_depth or "None",
                    }
                )
            mlflow.log_metrics(metrics)

            figure, axis = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(
                y_test,
                predictions,
                display_labels=label_encoder.classes_,
                ax=axis,
            )
            axis.set_title(f"Confusion Matrix - {model_name}")
            figure.tight_layout()
            mlflow.log_figure(
                figure,
                f"confusion_matrix_{model_name}.png",
            )
            plt.close(figure)
            mlflow.sklearn.log_model(model, artifact_path="model")

            results.append(
                {
                    "run_id": mlflow.active_run().info.run_id,
                    "model": model_name,
                    **metrics,
                }
            )
            print(f"{model_name}: {metrics}")

    results_df = pd.DataFrame(results).sort_values(
        "f1_macro", ascending=False
    )
    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model by F1 score: {results_df.iloc[0]['model']}")
    return results_df


if __name__ == "__main__":
    main()