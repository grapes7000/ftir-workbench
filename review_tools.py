from __future__ import annotations

"""Pure helpers for rebuilding Model Review trial projections.

The GUI slider calls these functions so each slider position is derived from that
history row's stored preprocessing recipe and PCA flag. No stale global PCA is used.
"""

import json

import numpy as np
import pandas as pd

import core


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
