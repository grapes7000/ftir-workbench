from __future__ import annotations

"""Pure helpers for rebuilding Model Review trial projections.

This module also wires the new transparent scientific audit and audited nested-CV
implementation into the existing v5.2 workflow without changing the public GUI API.
"""

import json

import numpy as np
import pandas as pd

import core
import scientific_analysis
import transparent_guided
import validation_audit


def _record(row):
    if isinstance(row, pd.Series):
        return row.to_dict()
    if isinstance(row, dict):
        return dict(row)
    return dict(row)


def trial_steps(row):
    record = _record(row)
    raw = record.get("recipe", "[]")
    recipe = json.loads(raw) if isinstance(raw, str) else raw
    return [core.Step(**item) for item in recipe]


def recipe_text(steps) -> str:
    parts = []
    for step in steps:
        if not step.enabled:
            continue
        if step.params:
            params = ", ".join(f"{k}={v}" for k, v in step.params.items())
            parts.append(f"{step.name} ({params})")
        else:
            parts.append(step.name)
    return " → ".join(parts)


def trial_projection(X, wn, row, max_display_components=3):
    """Rebuild the pre-classifier transform represented by one optimization row."""
    record = _record(row)
    X = np.asarray(X, dtype=float)
    wn = np.asarray(wn, dtype=float)
    steps = trial_steps(record)

    prep = core.RecipeTransformer(steps, wn=wn).fit(X)
    transformed = prep.transform(X)

    model_name = str(record.get("model", ""))
    if model_name == "SVM":
        transformed = core.StandardScaler().fit_transform(transformed)

    use_pca = bool(record.get("use_pca", False))
    if use_pca:
        pcs = int(record.get("pcs", 2))
        n_components = max(2, min(pcs, len(X) - 1, transformed.shape[1]))
        pca = core.PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=42,
        ).fit(transformed)
        model_scores = pca.transform(transformed)
        shown = min(int(max_display_components), model_scores.shape[1])
        scores = model_scores[:, :shown]
        description = (
            f"This trial really uses PCA in the model pipeline ({n_components} PCs). "
            f"The plot shows its first {shown} model PCA score axes."
        )
        display_only = False
    else:
        shown = max(2, min(int(max_display_components), len(X) - 1, transformed.shape[1]))
        pca = core.PCA(n_components=shown, svd_solver="full").fit(transformed)
        scores = pca.transform(transformed)
        description = (
            "This trial does not use PCA in the classifier. The plot is a display-only "
            f"{shown}-PC projection of this trial's exact processed variables."
        )
        display_only = True

    return {
        "scores": np.asarray(scores),
        "steps": steps,
        "recipe_text": recipe_text(steps),
        "description": description,
        "display_only": display_only,
        "uses_model_pca": use_pca,
    }


# --- Workflow integration -------------------------------------------------
# app.py imports review_tools before it runs any analysis.  We use that stable import
# point to enrich Guided Analysis and replace the older nested-CV routine while
# keeping app_ui.py backward-compatible.
_original_guided_analysis = transparent_guided.guided_analysis


def guided_analysis_with_scientific_audit(X, wn, meta, max_components=20):
    result = _original_guided_analysis(X, wn, meta, max_components=max_components)
    science = scientific_analysis.comprehensive_analysis(
        X, wn, meta, result, n_resamples=8
    )
    result["scientific"] = science
    result["report"] = (
        science["report"]
        + "\n\nDETAILED GUIDED-SELECTION AUDIT\n"
        + result["report"]
    )
    return result


def nested_optimize_compatible(*args, **kwargs):
    result = validation_audit.nested_optimize_audited(*args, **kwargs)
    # Existing plots expect these legacy names. They are aliases of OUTER held-out
    # metrics, not training metrics.
    result["folds"] = result["folds"].copy()
    result["folds"]["balanced_accuracy"] = result["folds"]["outer_balanced_accuracy"]
    result["folds"]["macro_f1"] = result["folds"]["outer_macro_f1"]
    # Surface validation evidence in the optimization history so the trial browser
    # shows the outer-fold context rather than only an inner-CV score.
    by_fold = result["folds"].set_index("fold")
    history = result["history"].copy()
    history["outer_balanced_accuracy"] = history.outer_fold.map(by_fold.outer_balanced_accuracy)
    history["outer_macro_f1"] = history.outer_fold.map(by_fold.outer_macro_f1)
    history["outer_train_validation_gap"] = history.outer_fold.map(by_fold.train_outer_gap)
    result["history"] = history
    return result


transparent_guided.guided_analysis = guided_analysis_with_scientific_audit
core.guided_analysis = guided_analysis_with_scientific_audit
core.nested_optimize = nested_optimize_compatible
