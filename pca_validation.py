from __future__ import annotations

"""Enhanced, leakage-safe PCA reconstruction cross-validation.

The component recommendation is based on held-out reconstruction error. Explained
variance is reported for interpretation only and never drives the selection rule.
"""

import math

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, KFold

import core


def pca_cv_enhanced(X, steps, max_components=30, folds=5, groups=None, wn=None):
    X = np.asarray(X, dtype=float)
    if len(X) < 4:
        raise ValueError("At least four samples are required for PCA cross-validation.")
    physical_wn = np.arange(X.shape[1], dtype=float) if wn is None else np.asarray(wn, dtype=float)
    if len(physical_wn) != X.shape[1]:
        raise ValueError("Wavenumber axis length does not match the spectral matrix.")

    nmax = max(1, min(int(max_components), len(X) - 2, X.shape[1]))
    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        n_splits = min(int(folds), len(unique_groups))
        if n_splits < 2:
            raise ValueError("At least two independent groups are required for grouped PCA CV.")
        splitter = GroupKFold(n_splits)
        splits = list(splitter.split(X, groups=groups))
        validation_mode = "grouped"
    else:
        n_splits = min(int(folds), len(X))
        if n_splits < 2:
            raise ValueError("At least two folds are required.")
        splitter = KFold(n_splits, shuffle=True, random_state=42)
        splits = list(splitter.split(X))
        validation_mode = "K-fold"

    nmax = min(nmax, min(len(train) - 1 for train, _ in splits))
    if nmax < 1:
        raise ValueError("Cross-validation folds are too small to fit PCA.")

    metrics = {
        n: {"cal": [], "val": [], "press": [], "tss": []}
        for n in range(1, nmax + 1)
    }
    for train, test in splits:
        # RecipeTransformer fits median imputation and any learned centering/scaling
        # only on this training fold, then reuses those parameters on the held-out fold.
        prep = core.RecipeTransformer(steps, wn=physical_wn).fit(X[train])
        calibration = prep.transform(X[train])
        validation = prep.transform(X[test])
        fold_nmax = min(nmax, len(train) - 1, calibration.shape[1])
        pca = PCA(fold_nmax, svd_solver="full").fit(calibration)
        tc = pca.transform(calibration)
        tv = pca.transform(validation)

        for n in range(1, nmax + 1):
            k = min(n, fold_nmax)
            rc = calibration - (pca.mean_ + tc[:, :k] @ pca.components_[:k])
            rv = validation - (pca.mean_ + tv[:, :k] @ pca.components_[:k])
            metrics[n]["cal"].append(math.sqrt(float(np.mean(rc**2))))
            metrics[n]["val"].append(math.sqrt(float(np.mean(rv**2))))
            metrics[n]["press"].append(float(np.sum(rv**2)))
            metrics[n]["tss"].append(
                float(np.sum((validation - calibration.mean(axis=0)) ** 2))
            )

    # Full-data explained variance is an interpretation metric only. The component
    # count below is still selected solely from the held-out reconstruction curve.
    full_prep = core.RecipeTransformer(steps, wn=physical_wn).fit(X)
    full_processed = full_prep.transform(X)
    full_nmax = min(nmax, len(X) - 1, full_processed.shape[1])
    full_pca = PCA(full_nmax, svd_solver="full").fit(full_processed)
    explained = np.asarray(full_pca.explained_variance_ratio_, dtype=float)
    cumulative = np.cumsum(explained)

    rows = []
    for n in range(1, nmax + 1):
        metric = metrics[n]
        val = np.asarray(metric["val"], dtype=float)
        press = float(np.sum(metric["press"]))
        tss = float(np.sum(metric["tss"]))
        sd = float(np.std(val, ddof=1)) if len(val) > 1 else 0.0
        se = sd / math.sqrt(len(val)) if len(val) else np.nan
        rows.append({
            "components": n,
            "RMSEC_X": float(np.mean(metric["cal"])),
            "RMSECV_X": float(np.mean(val)),
            "RMSECV_X_SD": sd,
            "RMSECV_X_SE": se,
            "RMSECV_X_BestFold": float(np.min(val)),
            "RMSECV_X_WorstFold": float(np.max(val)),
            "PRESS_X": press,
            "Q2_X": 1 - press / tss if tss else np.nan,
            "ExplainedVariance_X": float(explained[n - 1]) if n <= len(explained) else np.nan,
            "CumulativeExplainedVariance_X": float(cumulative[n - 1]) if n <= len(cumulative) else np.nan,
            "validation_mode": validation_mode,
            "folds": int(n_splits),
        })

    frame = pd.DataFrame(rows)
    minimum_index = int(frame.RMSECV_X.idxmin())
    minimum = frame.loc[minimum_index]
    threshold = float(minimum.RMSECV_X + minimum.RMSECV_X_SE)
    candidates = frame.loc[frame.RMSECV_X <= threshold, "components"]
    best = int(candidates.min()) if len(candidates) else int(minimum.components)
    frame["within_one_standard_error_of_minimum"] = frame.RMSECV_X <= threshold
    frame["recommended"] = frame.components == best
    frame.attrs["selection_rule"] = (
        "Smallest PC count whose mean held-out RMSECV-X is within one standard error "
        "of the minimum RMSECV-X. Explained variance is reported but not used for selection."
    )
    frame.attrs["minimum_rmsecv_components"] = int(minimum.components)
    frame.attrs["minimum_rmsecv"] = float(minimum.RMSECV_X)
    frame.attrs["one_se_threshold"] = threshold
    return frame, best
