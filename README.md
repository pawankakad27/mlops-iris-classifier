# MLOps Iris Classifier
Version B

A sample ML project used to demonstrate Git-based version control
workflows in an MLOps context.

## Setup
```bash
pip install -r requirements.txt
python src/train.py
```

## Experiment 7: Baseline and hyperparameter tuning

Use the MLflow environment from the project setup to run the tuning workflow:

```bash
.venv-mlflow\Scripts\Activate.ps1
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python src/baseline_model.py
python src/grid_search_tuning.py
python src/random_search_tuning.py
python src/compare_tuning_results.py
```

The baseline script logs a default Decision Tree run, while the grid/random search scripts log every candidate configuration to the `iris-hyperparameter-tuning` experiment and compare the best-performing settings.
