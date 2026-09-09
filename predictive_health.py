from __future__ import annotations

import numpy as np
import pandas as pd


def predictive_subscores(result):
    """Return transparent supervised-validation health subscores."""
    classes = result.get("classes", [])
    n_classes = max(2, len(classes))
    chance = 1.0 / n_classes
    bal = float(result.get("balanced_accuracy", np.nan))
    folds = result.get("folds", pd.DataFrame())
    gap = float(result.get("train_validation_gap", np.nan))

    rows = []
    reasons = []
    if np.isfinite(bal):
        normalized = (bal - chance) / max(1e-12, 1 - chance)
        score = int(round(100 * max(0.0, min(1.0, normalized))))
        rows.append(("Predictive held-out performance", score))
        reasons.append(
            f"Nested grouped-CV balanced accuracy={bal:.3f}; balanced-accuracy chance reference for {n_classes} classes is {chance:.3f}."
        )

    if isinstance(folds, pd.DataFrame) and not folds.empty:
        column = "outer_balanced_accuracy" if "outer_balanced_accuracy" in folds else "balanced_accuracy"
        sd = float(folds[column].std(ddof=1)) if len(folds) > 1 else 0.0
        score = 100 if sd <= 0.05 else 85 if sd <= 0.10 else 65 if sd <= 0.15 else 40 if sd <= 0.25 else 20
        rows.append(("Predictive fold stability", score))
        reasons.append(f"Outer-fold balanced-accuracy SD={sd:.3f}.")

    if np.isfinite(gap):
        score = 100 if gap <= 0.03 else 90 if gap <= 0.07 else 75 if gap <= 0.10 else 55 if gap <= 0.20 else 25
        rows.append(("Train-to-held-out gap", score))
        reasons.append(f"Mean training minus outer held-out balanced-accuracy gap={gap:.3f}.")

    return pd.DataFrame(rows, columns=["subscore", "score_0_100"]), reasons


def combine_health(exploratory_health, predictive_result):
    base = exploratory_health["table"].copy()
    predictive, reasons = predictive_subscores(predictive_result)
    table = pd.concat([base, predictive], ignore_index=True)
    overall = float(table.score_0_100.mean()) if len(table) else np.nan
    if np.isfinite(overall) and overall >= 75:
        band = "GREEN — strong / trustworthy"
    elif np.isfinite(overall) and overall >= 50:
        band = "YELLOW — useful but caution required"
    else:
        band = "RED — unreliable / insufficient evidence"
    return {
        "table": table,
        "overall": overall,
        "band": band,
        "reasons": list(exploratory_health.get("reasons", [])) + reasons,
        "disclaimer": exploratory_health.get(
            "disclaimer",
            "This score is a transparent engineering summary, not an established statistical quantity.",
        ),
    }
