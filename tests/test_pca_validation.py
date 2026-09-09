import numpy as np

import core
from pca_validation import pca_cv_enhanced


def test_enhanced_pca_cv_reports_variance_fold_range_and_one_se_rule():
    rng = np.random.default_rng(21)
    latent = rng.normal(size=(30, 3))
    loadings = rng.normal(size=(3, 50))
    X = latent @ loadings + rng.normal(0, 0.06, (30, 50))
    wn = np.linspace(700, 1800, 50)
    result, best = pca_cv_enhanced(
        X,
        [core.Step("Mean center", {})],
        max_components=8,
        folds=5,
        wn=wn,
    )
    required = {
        "RMSEC_X",
        "RMSECV_X",
        "RMSECV_X_SD",
        "RMSECV_X_SE",
        "RMSECV_X_BestFold",
        "RMSECV_X_WorstFold",
        "PRESS_X",
        "Q2_X",
        "ExplainedVariance_X",
        "CumulativeExplainedVariance_X",
        "within_one_standard_error_of_minimum",
        "recommended",
    }
    assert required.issubset(result.columns)
    minimum = result.loc[result.RMSECV_X.idxmin()]
    threshold = float(minimum.RMSECV_X + minimum.RMSECV_X_SE)
    eligible = result.loc[result.RMSECV_X <= threshold, "components"]
    assert best == int(eligible.min())
    assert bool(result.loc[result.components == best, "recommended"].iloc[0])
    assert np.all(np.diff(result.CumulativeExplainedVariance_X.dropna()) >= -1e-12)


def test_enhanced_pca_cv_range_uses_physical_axis_inside_folds():
    rng = np.random.default_rng(8)
    wn = np.linspace(500, 2000, 75)
    X = rng.normal(size=(24, len(wn)))
    steps = [
        core.Step("Range", {"minimum": 900.0, "maximum": 1500.0}),
        core.Step("Mean center", {}),
    ]
    result, best = pca_cv_enhanced(X, steps, max_components=5, folds=4, wn=wn)
    assert len(result) == 5
    assert 1 <= best <= 5
    processed, retained, _ = core.apply_recipe(X, wn, steps)
    assert retained.min() >= 900.0
    assert retained.max() <= 1500.0
    assert processed.shape[1] == len(retained)
