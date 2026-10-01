# Experiment 5: Feast Feature Store

This isolated Feast repository consumes the validated feature output from Experiment 4. Its virtual environment is separate from the main project because Feast has its own dependency set.

## Setup

From the project root in PowerShell:

```powershell
python -m venv practical5_feast\.venv
.\practical5_feast\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r practical5_feast/requirements.txt
Set-Location practical5_feast/iris_feature_repo/feature_repo
python prepare_feature_source.py
feast apply
$end = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss")
feast materialize-incremental $end
python get_online_features.py
python get_historical_features.py
python reuse_features_for_clustering.py
```

The source builder reads `data/processed/iris_features.csv` from Experiment 4. Run the source builder again after reproducing Experiment 4. Feast's registry and SQLite online store are local generated files under `iris_feature_repo/feature_repo/data/` and are intentionally ignored by Git.

The generated source assigns sequential `sample_id` values for this dataset snapshot and UTC event timestamps immediately before the current time. `species` stays out of the feature views to avoid target leakage.