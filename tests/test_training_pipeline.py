"""Smoke tests for the training pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


@pytest.fixture
def synthetic_dataset():
    rng = np.random.default_rng(42)
    n = 30
    X = pd.DataFrame(rng.normal(5, 1, size=(n, 4)), columns=FEATURE_COLS)
    y = rng.integers(0, 3, size=n)
    return X, y


def test_model_trains_without_error(synthetic_dataset):
    X, y = synthetic_dataset
    model = RandomForestClassifier(n_estimators=10, random_state=42)

    model.fit(X, y)

    assert hasattr(model, "predict")


def test_model_predict_output_shape(synthetic_dataset):
    X, y = synthetic_dataset
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == (len(X),)


def test_model_predict_proba_sums_to_one(synthetic_dataset):
    X, y = synthetic_dataset
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    probs = model.predict_proba(X)

    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()
