import numpy as np
import pandas as pd

from models import PLSDAClassifier
from predictive_health import combine_health, predictive_subscores


def test_plsda_classifier_is_sklearn_compatible_for_basic_fit_predict():
    rng = np.random.default_rng(4)
    X0 = rng.normal(0, 0.2, (16, 6))
    X1 = rng.normal(0, 0.2, (16, 6)) + np.array([1, 0.6, 0.2, 0, 0, 0])
    X = np.vstack([X0, X1])
    y = np.array(["A"] * len(X0) + ["B"] * len(X1))
    model = PLSDAClassifier(n_components=2).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert set(pred).issubset({"A", "B"})
    assert model.n_components_ == 2


def test_predictive_health_uses_outer_performance_stability_and_gap():
    result = {
        "classes": np.array(["A", "B"]),
        "balanced_accuracy": 0.82,
        "train_validation_gap": 0.06,
        "folds": pd.DataFrame({"outer_balanced_accuracy": [0.80, 0.84, 0.82]}),
    }
    table, reasons = predictive_subscores(result)
    assert set(table.subscore) == {
        "Predictive held-out performance",
        "Predictive fold stability",
        "Train-to-held-out gap",
    }
    assert table.score_0_100.between(0, 100).all()
    assert len(reasons) == 3


def test_combined_health_keeps_subscores_visible():
    exploratory = {
        "table": pd.DataFrame({"subscore": ["Data QC"], "score_0_100": [90]}),
        "reasons": ["QC okay"],
        "disclaimer": "transparent summary",
    }
    predictive = {
        "classes": np.array(["A", "B"]),
        "balanced_accuracy": 0.80,
        "train_validation_gap": 0.05,
        "folds": pd.DataFrame({"outer_balanced_accuracy": [0.75, 0.80, 0.85]}),
    }
    combined = combine_health(exploratory, predictive)
    assert len(combined["table"]) == 4
    assert np.isfinite(combined["overall"])
    assert "transparent" in combined["disclaimer"]
