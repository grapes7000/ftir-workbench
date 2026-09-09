from __future__ import annotations

"""Auditable supervised validation helpers for FTIR Workbench.

Outer folds estimate generalization. Inner folds perform optimization. The returned
objects explicitly record which samples/groups were held out, training-vs-validation
gaps, and out-of-fold predictions so users can inspect the evidence directly.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

import core


def _specificity_table(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    total = cm.sum()
    rows = []
    for i, label in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        sensitivity = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        rows.append({
            "class": label,
            "support": int(tp + fn),
            "sensitivity_recall": float(sensitivity) if np.isfinite(sensitivity) else np.nan,
            "specificity": float(specificity) if np.isfinite(specificity) else np.nan,
        })
    return pd.DataFrame(rows)


def nested_optimize_audited(
    X,
    y,
    groups,
    base,
    trials=20,
    maxpcs=20,
    outer_folds=3,
    inner_folds=2,
    wn=None,
    seed=42,
):
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y, and groups must contain the same number of samples.")

    outer = StratifiedGroupKFold(int(outer_folds), shuffle=True, random_state=int(seed))
    predictions = np.empty(len(y), dtype=object)
    outer_assignment = np.full(len(y), -1, dtype=int)
    folds = []
    history = []
    best_rows = []
    split_rows = []

    for fold, (train, test) in enumerate(outer.split(X, y, groups), 1):
        train_groups = set(groups[train].astype(str))
        test_groups = set(groups[test].astype(str))
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise RuntimeError(
                f"Grouped CV leakage detected in outer fold {fold}: {sorted(overlap)[:5]}"
            )

        inner_support = (
            pd.DataFrame({"class": y[train].astype(str), "group": groups[train].astype(str)})
            .drop_duplicates()
            .groupby("class")
            .size()
        )
        if inner_support.empty:
            raise ValueError(f"Outer fold {fold} has no class/group support for inner CV.")
        inner_n = min(int(inner_folds), int(inner_support.min()))
        if inner_n < 2:
            limiting = ", ".join(f"{label}={count}" for label, count in inner_support.items())
            raise ValueError(
                f"Outer fold {fold} leaves fewer than two independent groups in at least one class for inner CV ({limiting}). "
                "Use fewer outer folds, collect more independent groups, or treat the supervised result as unsupported."
            )
        inner = StratifiedGroupKFold(inner_n, shuffle=True, random_state=1000 + int(seed) + fold)

        steps, params, fold_history = core.search(
            X[train],
            y[train],
            groups[train],
            base,
            int(trials),
            min(int(maxpcs), len(train) - 1),
            inner,
            2000 + int(seed) + fold,
            wn=wn,
        )
        complete = fold_history.loc[fold_history.status == "complete"].copy()
        if complete.empty:
            raise RuntimeError(f"All inner optimization trials failed in outer fold {fold}.")
        selected_inner = complete.sort_values("objective", ascending=False).iloc[0]

        model = core.pipe(steps, params, wn=wn).fit(X[train], y[train])
        pred_train = model.predict(X[train])
        pred_test = model.predict(X[test])
        predictions[test] = pred_test
        outer_assignment[test] = fold

        train_bal = float(balanced_accuracy_score(y[train], pred_train))
        train_f1 = float(f1_score(y[train], pred_train, average="macro", zero_division=0))
        test_bal = float(balanced_accuracy_score(y[test], pred_test))
        test_f1 = float(f1_score(y[test], pred_test, average="macro", zero_division=0))
        inner_bal = float(selected_inner.score)
        inner_std = float(selected_inner.get("std", np.nan))

        fold_history["outer_fold"] = fold
        history.append(fold_history)
        best_rows.append({
            "fold": fold,
            "recipe": selected_inner.recipe,
            "inner_cv_balanced_accuracy": inner_bal,
            "inner_cv_std": inner_std,
            **params,
        })
        folds.append({
            "fold": fold,
            "n_train": len(train),
            "n_test": len(test),
            "train_balanced_accuracy": train_bal,
            "inner_cv_balanced_accuracy": inner_bal,
            "inner_cv_std": inner_std,
            "outer_balanced_accuracy": test_bal,
            "train_macro_f1": train_f1,
            "outer_macro_f1": test_f1,
            "train_outer_gap": train_bal - test_bal,
            "inner_outer_gap": inner_bal - test_bal,
            "train_group_count": len(train_groups),
            "test_group_count": len(test_groups),
            "minimum_inner_groups_per_class": int(inner_support.min()),
            "inner_folds_used": int(inner_n),
            "group_overlap_count": 0,
        })
        for idx in train:
            split_rows.append({"sample_index": int(idx), "outer_fold": fold, "role": "train", "group": str(groups[idx])})
        for idx in test:
            split_rows.append({"sample_index": int(idx), "outer_fold": fold, "role": "test", "group": str(groups[idx])})

    if np.any(outer_assignment < 0):
        raise RuntimeError("Some samples never appeared in an outer held-out fold.")

    classes = np.unique(y)
    cm = confusion_matrix(y, predictions, labels=classes)
    report = pd.DataFrame(classification_report(y, predictions, output_dict=True, zero_division=0)).T
    fold_table = pd.DataFrame(folds)
    oof = pd.DataFrame({
        "sample_index": np.arange(len(y)),
        "outer_fold": outer_assignment,
        "group": groups.astype(str),
        "truth": y,
        "prediction": predictions,
        "correct": predictions == y,
    })
    specificity = _specificity_table(y, predictions, classes)

    bal = float(balanced_accuracy_score(y, predictions))
    macro = float(f1_score(y, predictions, average="macro", zero_division=0))
    mean_train = float(fold_table.train_balanced_accuracy.mean())
    mean_outer = float(fold_table.outer_balanced_accuracy.mean())
    gap = mean_train - mean_outer
    if gap >= 0.20:
        overfit = "high"
    elif gap >= 0.10:
        overfit = "moderate"
    else:
        overfit = "low"

    summary = (
        f"Nested grouped CV balanced accuracy is {bal:.3f}; macro F1 is {macro:.3f}. "
        f"Outer-fold balanced accuracy averages {mean_outer:.3f} ± "
        f"{float(fold_table.outer_balanced_accuracy.std(ddof=1)) if len(fold_table) > 1 else 0.0:.3f}. "
        f"Training balanced accuracy averages {mean_train:.3f}, giving a training-to-outer gap of {gap:.3f} ({overfit} overfitting warning). "
        f"Worst outer fold={float(fold_table.outer_balanced_accuracy.min()):.3f}; best={float(fold_table.outer_balanced_accuracy.max()):.3f}. "
        "Every outer test group is disjoint from its training groups, all learned preprocessing/model steps are fitted inside the training side of each split, and all reported predictions are out-of-fold."
    )

    return {
        "pred": predictions,
        "oof_predictions": oof,
        "outer_assignments": outer_assignment,
        "splits": pd.DataFrame(split_rows),
        "folds": fold_table,
        "history": pd.concat(history, ignore_index=True),
        "best": pd.DataFrame(best_rows),
        "classes": classes,
        "cm": cm,
        "report": report,
        "specificity": specificity,
        "balanced_accuracy": bal,
        "macro_f1": macro,
        "mean_train_balanced_accuracy": mean_train,
        "mean_outer_balanced_accuracy": mean_outer,
        "train_validation_gap": gap,
        "overfitting_level": overfit,
        "validation_summary": summary,
    }


def permutation_test_nested(
    X,
    y,
    groups,
    base,
    observed_score,
    n_permutations=20,
    trials=10,
    maxpcs=15,
    outer_folds=3,
    inner_folds=2,
    wn=None,
    seed=123,
):
    """Optional nested-CV null test. Permutations are not optimized more than the real model."""
    rng = np.random.default_rng(seed)
    scores = []
    y = np.asarray(y)
    for i in range(int(n_permutations)):
        shuffled = rng.permutation(y)
        try:
            result = nested_optimize_audited(
                X,
                shuffled,
                groups,
                base,
                trials=trials,
                maxpcs=maxpcs,
                outer_folds=outer_folds,
                inner_folds=inner_folds,
                wn=wn,
                seed=seed + i + 1,
            )
            scores.append(float(result["balanced_accuracy"]))
        except Exception:
            continue
    if not scores:
        return {"scores": [], "p_value": np.nan, "summary": "Permutation testing could not complete."}
    arr = np.asarray(scores)
    p = float((1 + np.sum(arr >= observed_score)) / (1 + len(arr)))
    summary = (
        f"Observed nested-CV balanced accuracy={observed_score:.3f}. Under {len(arr)} shuffled-label runs, "
        f"mean balanced accuracy={arr.mean():.3f}; empirical p≈{p:.3f}."
    )
    return {"scores": scores, "p_value": p, "summary": summary}
