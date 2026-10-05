"""Register the highest-F1 run as the staged Iris classifier."""
import os

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "iris-classification-baseline"
REGISTERED_MODEL_NAME = "iris-classifier-prod"


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {EXPERIMENT_NAME!r} was not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_macro DESC"],
        max_results=1000,
    )
    if not runs:
        raise RuntimeError(f"No completed runs found in {EXPERIMENT_NAME!r}")

    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    print(f"Best run: {best_run.info.run_id}")
    print(f"Model type: {best_run.data.params['model_type']}")
    print(f"F1 score: {best_run.data.metrics['f1_macro']:.4f}")

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=registered_model.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    print(
        f"Registered {REGISTERED_MODEL_NAME} version {registered_model.version} "
        "and moved it to Staging."
    )


if __name__ == "__main__":
    main()