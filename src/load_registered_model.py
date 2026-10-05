"""Load the staged classifier from MLflow and run a small inference batch."""
import json
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from train_with_mlflow import DATA_PATH, FEATURE_COLUMNS

REGISTERED_MODEL_NAME = "iris-classifier-prod"
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}/Staging"


def main() -> pd.DataFrame:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    model_versions = client.get_latest_versions(
        REGISTERED_MODEL_NAME,
        stages=["Staging"],
    )
    if not model_versions:
        raise RuntimeError(f"No Staging version found for {REGISTERED_MODEL_NAME!r}")

    model_version = model_versions[0]
    run = client.get_run(model_version.run_id)
    classes = json.loads(run.data.params["target_classes"])
    model = mlflow.sklearn.load_model(MODEL_URI)

    df = pd.read_csv(DATA_PATH)
    features = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median()).head(5)
    predictions = model.predict(features).astype(int)
    result = pd.DataFrame(
        {
            "sample_id": features.index,
            "predicted_species": [classes[index] for index in predictions],
        }
    )
    print(f"Loaded {REGISTERED_MODEL_NAME} version {model_version.version} from Staging")
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    main()