# Experiment 6: MLflow Tracking and Model Registration

This workflow trains three classifiers from the validated Experiment 4 feature
CSV, tracks their parameters, metrics, models, and confusion matrices, registers
the highest-F1 run, and loads the staged model for inference.

## Setup

From the project root in PowerShell, create a separate environment so the
MLflow stack does not change the main project or Feast environment:

```powershell
python -m venv .venv-mlflow
.\.venv-mlflow\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-mlflow.txt
```

Start the tracking server in one terminal and leave it running:

```powershell
.\.venv-mlflow\Scripts\mlflow.exe server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

Open <http://127.0.0.1:5000>. In a second terminal, from the project root,
activate the same environment and run:

```powershell
.\.venv-mlflow\Scripts\Activate.ps1
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python src/train_with_mlflow.py
python src/register_best_model.py
python src/load_registered_model.py
```

The experiment `iris-classification-baseline` contains one run per model:
logistic regression, a 50-tree random forest with depth 3, and a 200-tree
unbounded random forest. Every run records accuracy, macro precision, macro
recall, macro F1, hyperparameters, a confusion-matrix PNG, and a serialized
`model/` artifact. The registration script selects the highest macro-F1 run,
registers it as `iris-classifier-prod`, and moves the new version to Staging.
The final script loads `models:/iris-classifier-prod/Staging` and predicts Iris
species for five input rows.

Use the MLflow UI's experiment comparison view to inspect all runs and their
metrics. The SQLite tracking database and local artifact directory are generated
runtime data and are ignored by Git.