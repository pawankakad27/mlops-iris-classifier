# Testing strategy

This project implements the Experiment 8 test suite for data processing, training pipeline smoke checks, and model quality validation.

## Test files and coverage

| File | Layer | Purpose |
| --- | --- | --- |
| `tests/test_preprocess.py` | Data | Verifies duplicate removal, median imputation, target dropping, and empty-data handling. |
| `tests/test_features.py` | Data | Verifies engineered features and zero-division safety. |
| `tests/test_validate.py` | Data / pipeline | Verifies pass/fail validation behavior for schema, species, and range checks. |
| `tests/test_training_pipeline.py` | Pipeline | Exercises the training flow on a small synthetic dataset and checks model output structure. |
| `tests/test_model_quality.py` | Model quality | Confirms the model meets a minimum quality threshold and emits valid labels. |

The project uses `pytest` with `pytest-cov` to run the suite and emit JUnit XML output. The current measured coverage for the active source files is approximately 17% overall, with the pipeline modules covering the majority of their branches and edge cases.

## Execution

```bash
pytest -v
pytest --cov=src --cov-report=term-missing --junitxml=test-results.xml
```

The JUnit XML file is intended for CI systems that consume pass/fail summaries and coverage metrics.
