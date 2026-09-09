import json

import numpy as np
import pandas as pd

from review_tools import trial_projection
from spectral_constraints import apply_hard_exclusion, normalize_exclusion


def test_hard_exclusion_removes_forbidden_wavenumbers():
    wn = np.linspace(400, 1200, 17)
    X = np.arange(3 * len(wn), dtype=float).reshape(3, -1)
    filtered, kept_wn, info = apply_hard_exclusion(X, wn, "500-1000")
    assert filtered.shape[1] == len(kept_wn)
    assert not np.any((kept_wn >= 500) & (kept_wn <= 1000))
    assert info["removed_variables"] > 0
    assert info["normalized"] == "500-1000"


def test_hard_exclusion_merges_overlapping_ranges():
    assert normalize_exclusion("500-800,700-1000, 2350-2450") == "500-1000,2350-2450"


def test_hard_exclusion_rejects_removing_everything():
    wn = np.linspace(400, 1200, 17)
    X = np.ones((4, len(wn)))
    try:
        apply_hard_exclusion(X, wn, "0-5000")
    except ValueError as exc:
        assert "at least two" in str(exc).lower()
    else:
        raise AssertionError("Expected a ValueError when all spectral variables are removed.")


def _row(recipe, use_pca, model="SVM", pcs=3):
    return pd.Series(
        {
            "recipe": json.dumps(recipe),
            "use_pca": use_pca,
            "model": model,
            "pcs": pcs,
            "score": 0.75,
            "trial": 1,
        }
    )


def test_trial_projection_rebuilds_stored_recipe_instead_of_stale_pca():
    rng = np.random.default_rng(7)
    wn = np.linspace(400, 1800, 81)
    base = np.sin(wn / 110)
    X = np.array(
        [base + 0.05 * i + rng.normal(0, 0.01, len(wn)) for i in range(18)]
    )

    mean_recipe = [
        {"name": "Mean center", "params": {}, "enabled": True},
    ]
    derivative_recipe = [
        {
            "name": "Derivative",
            "params": {"order": 1, "window": 11, "poly": 2},
            "enabled": True,
        },
        {"name": "Mean center", "params": {}, "enabled": True},
    ]

    first = trial_projection(X, wn, _row(mean_recipe, True))
    second = trial_projection(X, wn, _row(derivative_recipe, True))

    assert first["uses_model_pca"] is True
    assert second["uses_model_pca"] is True
    assert "Derivative" in second["recipe_text"]
    assert first["scores"].shape == second["scores"].shape
    assert not np.allclose(first["scores"], second["scores"])


def test_trial_projection_marks_non_pca_model_view_as_display_only():
    rng = np.random.default_rng(3)
    wn = np.linspace(400, 1800, 61)
    X = rng.normal(size=(16, len(wn)))
    recipe = [{"name": "Mean center", "params": {}, "enabled": True}]
    result = trial_projection(X, wn, _row(recipe, False, model="Random Forest"))
    assert result["display_only"] is True
    assert result["uses_model_pca"] is False
    assert result["scores"].shape[1] >= 2
