from __future__ import annotations

"""Auditable supervised validation helpers for FTIR Workbench.

Outer folds estimate generalization. Inner folds perform *all* model/preprocessing
selection. The outer test fold never participates in recipe selection, PCA fitting,
hyperparameter tuning, imputation, centering/scaling, or model fitting.
"""

import copy
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import core
from models import PLSDAClassifier
import transparent_guided


MODEL_FAMILIES = ("SVM", "Random Forest", "PLS-DA")
INNER_STABILITY_PENALTY = 0.25
COMPONENT_COMPLEXITY_PENALTY = 0.002
PREPROCESSING_COMPLEXITY_PENALTY = 0.003


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


def _recipe_key(steps):
    return json.dumps([asdict(step) for step in steps], sort_keys=True)


def _candidate_recipes(wn, n_features, base):
    axis = np.arange(n_features, dtype=float) if wn is None else np.asarray(wn, dtype=float)
    named = [(name, copy.deepcopy(steps)) for name, steps in transparent_guided.candidate_recipes(axis, n_features)]
    if base:
        key = _recipe_key(base)
        if key not in {_recipe_key(steps) for _, steps in named}:
            named.insert(0, ("Current expert recipe", copy.deepcopy(base)))
    return named


def _random_params(rng, model, maxpcs, n_samples, n_features):
    max_pc = max(2, min(int(maxpcs), max(2, n_samples - 2), n_features))
    if model == "SVM":
        gamma_options = ["scale", 0.001, 0.01, 0.1]
        return {
            "model": model,
            "use_pca": bool(rng.integers(2)),
            "pcs": int(rng.integers(2, max_pc + 1)),
            "C": float(10 ** rng.uniform(-2, 2)),
            "kernel": ["linear", "rbf"][int(rng.integers(2))],
            "gamma": gamma_options[int(rng.integers(len(gamma_options)))],
        }
    if model == "Random Forest":
        return {
            "model": model,
            "use_pca": bool(rng.integers(2)),
            "pcs": int(rng.integers(2, max_pc + 1)),
            "trees": [200, 400][int(rng.integers(2))],
            "depth": [None, 5, 10, 20][int(rng.integers(4))],
            "leaf": int(rng.integers(1, 5)),
        }
    pls_max = max(1, min(10, max_pc, n_features, n_samples - 1))
    return {
        "model": "PLS-DA",
        "use_pca": False,
        "pcs": 0,
        "pls_components": int(rng.integers(1, pls_max + 1)),
    }


def build_pipeline(steps, params, wn=None):
    chain = [("prep", core.RecipeTransformer(copy.deepcopy(steps), wn=wn))]
    model = params["model"]
    if model == "SVM":
        chain.append(("scale", StandardScaler()))
    if bool(params.get("use_pca", False)):
        chain.append((
            "pca",
            PCA(
                n_components=int(params["pcs"]),
                svd_solver="randomized",
                random_state=42,
            ),
        ))
    if model == "SVM":
        classifier = SVC(
            C=float(params["C"]),
            kernel=params["kernel"],
            gamma=params["gamma"],
            class_weight="balanced",
        )
    elif model == "Random Forest":
        classifier = RandomForestClassifier(
            n_estimators=int(params["trees"]),
            max_depth=params["depth"],
            min_samples_leaf=int(params["leaf"]),
            class_weight="balanced",
            n_jobs=1,
            random_state=42,
        )
    elif model == "PLS-DA":
        classifier = PLSDAClassifier(
            n_components=int(params["pls_components"]),
            scale=False,
        )
    else:
        raise ValueError(f"Unknown model family: {model}")
    chain.append(("clf", classifier))
    return Pipeline(chain)


def _candidate_from_history_row(row):
    """Reconstruct one inner-selected candidate without consulting outer-test data."""
    steps = [core.Step(**item) for item in json.loads(str(row.recipe))]
    model = str(row.model)
    use_pca = bool(row.get("use_pca", False))
    params = {
        "model": model,
        "use_pca": use_pca,
        "pcs": int(row.pcs) if use_pca and pd.notna(row.get("pcs", np.nan)) else 0,
    }
    if model == "SVM":
        gamma = row.gamma
        if isinstance(gamma, str):
            cleaned = gamma
        else:
            cleaned = float(gamma)
        params.update(C=float(row.C), kernel=str(row.kernel), gamma=cleaned)
    elif model == "Random Forest":
        depth = row.depth
        depth = None if pd.isna(depth) else int(depth)
        params.update(trees=int(row.trees), depth=depth, leaf=int(row.leaf))
    else:
        params.update(pls_components=int(row.pls_components))
    return steps, params


def search_audited(X, y, groups, base, trials, maxpcs, cv, seed, wn=None):
    """Inner-CV search across explicit FTIR recipes and three model families.

    The search space is intentionally restrained and returned row-by-row. There is
    no opaque optimizer. Recipe/model choices are deterministic given the seed.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    recipes = _candidate_recipes(wn, X.shape[1], base)
    history = []
    best = None

    for trial in range(int(trials)):
        # Cycle deterministically so small trial counts still inspect multiple model
        # families and preprocessing recipes before random repeats occur.
        model = MODEL_FAMILIES[trial % len(MODEL_FAMILIES)]
        recipe_index = (trial // len(MODEL_FAMILIES)) % len(recipes)
        if trial >= len(MODEL_FAMILIES) * len(recipes):
            recipe_index = int(rng.integers(len(recipes)))
            model = MODEL_FAMILIES[int(rng.integers(len(MODEL_FAMILIES)))]
        recipe_name, steps = recipes[recipe_index]
        params = _random_params(rng, model, maxpcs, len(X), X.shape[1])
        try:
            values = cross_val_score(
                build_pipeline(steps, params, wn=wn),
                X,
                y,
                groups=groups,
                cv=cv,
                scoring="balanced_accuracy",
                error_score="raise",
                n_jobs=1,
            )
            score = float(values.mean())
            std = float(values.std())
            component_complexity = (
                int(params.get("pls_components", 0))
                if model == "PLS-DA"
                else int(params.get("pcs", 0)) if params.get("use_pca") else 0
            )
            prep_complexity = transparent_guided.recipe_complexity(steps)
            objective = (
                score
                - INNER_STABILITY_PENALTY * std
                - COMPONENT_COMPLEXITY_PENALTY * component_complexity
                - PREPROCESSING_COMPLEXITY_PENALTY * prep_complexity
            )
            row = {
                "trial": trial,
                "status": "complete",
                "score": score,
                "std": std,
                "objective": objective,
                "recipe_name": recipe_name,
                "recipe_complexity": prep_complexity,
                "recipe": _recipe_key(steps),
                **params,
            }
            history.append(row)
            if best is None or objective > best[0]:
                best = (objective, copy.deepcopy(steps), dict(params))
        except Exception as exc:
            history.append({
                "trial": trial,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "recipe_name": recipe_name,
                "recipe_complexity": transparent_guided.recipe_complexity(steps),
                "recipe": _recipe_key(steps),
                **params,
            })

    if best is None:
        raise RuntimeError("All inner optimization trials failed.")
    return best[1], best[2], pd.DataFrame(history)


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
    validation_mode="grouped",
):
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y, and groups must contain the same number of samples.")

    if validation_mode not in {"grouped", "exploratory"}:
        raise ValueError("Unknown validation mode.")
    exploratory = validation_mode == "exploratory"
    if exploratory:
        groups = np.arange(len(y))
    splitter = StratifiedKFold if exploratory else StratifiedGroupKFold
    outer = splitter(int(outer_folds), shuffle=True, random_state=int(seed))
    predictions = np.empty(len(y), dtype=object)
    outer_assignment = np.full(len(y), -1, dtype=int)
    folds = []
    history = []
    best_rows = []
    split_rows = []
    family_fold_rows = []
    family_oof_rows = []

    for fold, (train, test) in enumerate(outer.split(X, y) if exploratory else outer.split(X, y, groups), 1):
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
            .reindex(np.unique(y.astype(str)), fill_value=0)
        )
        if inner_support.empty:
            raise ValueError(f"Outer fold {fold} has no class/group support for inner CV.")
        inner_n = min(int(inner_folds), int(inner_support.min()))
        if inner_n < 2:
            limiting = ", ".join(f"{label}={count}" for label, count in inner_support.items())
            raise ValueError(
                f"Outer fold {fold} leaves fewer than two independent groups in at least one class for inner CV ({limiting}). "
                "Check source columns and class eligibility; collect more independent sources or use explicitly exploratory individual-spectra validation. Fewer outer folds can leave less training data."
            )
        inner = splitter(inner_n, shuffle=True, random_state=1000 + int(seed) + fold)

        if exploratory:
            inner = list(inner.split(X[train], y[train]))

        steps, params, fold_history = search_audited(
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

        # Main nested-selection result: inner CV chooses recipe/model jointly, then
        # the untouched outer fold evaluates that complete selection procedure.
        model = build_pipeline(steps, params, wn=wn).fit(X[train], y[train])
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

        # Fair family comparison: for each family, select its best recipe/parameters
        # using INNER data only, then evaluate all families on this SAME outer fold.
        for family in MODEL_FAMILIES:
            family_candidates = complete.loc[complete.model == family]
            if family_candidates.empty:
                continue
            family_inner = family_candidates.sort_values("objective", ascending=False).iloc[0]
            family_steps, family_params = _candidate_from_history_row(family_inner)
            family_model = build_pipeline(family_steps, family_params, wn=wn).fit(
                X[train], y[train]
            )
            family_train_pred = family_model.predict(X[train])
            family_test_pred = family_model.predict(X[test])
            family_train_bal = float(balanced_accuracy_score(y[train], family_train_pred))
            family_outer_bal = float(balanced_accuracy_score(y[test], family_test_pred))
            family_outer_f1 = float(
                f1_score(y[test], family_test_pred, average="macro", zero_division=0)
            )
            family_fold_rows.append({
                "fold": fold,
                "model": family,
                "recipe_name": family_inner.get("recipe_name", ""),
                "recipe": family_inner.recipe,
                "inner_cv_balanced_accuracy": float(family_inner.score),
                "inner_cv_std": float(family_inner.get("std", np.nan)),
                "train_balanced_accuracy": family_train_bal,
                "outer_balanced_accuracy": family_outer_bal,
                "outer_macro_f1": family_outer_f1,
                "train_outer_gap": family_train_bal - family_outer_bal,
                "n_train": len(train),
                "n_test": len(test),
            })
            for sample_index, truth, prediction, group_value in zip(
                test, y[test], family_test_pred, groups[test]
            ):
                family_oof_rows.append({
                    "sample_index": int(sample_index),
                    "outer_fold": fold,
                    "model": family,
                    "group": str(group_value),
                    "truth": truth,
                    "prediction": prediction,
                    "correct": bool(prediction == truth),
                })

        fold_history["outer_fold"] = fold
        history.append(fold_history)
        best_rows.append({
            "fold": fold,
            "recipe": selected_inner.recipe,
            "recipe_name": selected_inner.get("recipe_name", ""),
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
            "selected_model": params["model"],
            "selected_recipe": selected_inner.get("recipe_name", ""),
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

    family_fold_table = pd.DataFrame(family_fold_rows)
    family_oof = pd.DataFrame(family_oof_rows)
    if not family_fold_table.empty:
        family_summary = (
            family_fold_table.groupby("model", as_index=False)
            .agg(
                folds=("fold", "nunique"),
                mean_inner_cv_balanced_accuracy=("inner_cv_balanced_accuracy", "mean"),
                mean_outer_balanced_accuracy=("outer_balanced_accuracy", "mean"),
                outer_balanced_accuracy_sd=("outer_balanced_accuracy", "std"),
                worst_outer_balanced_accuracy=("outer_balanced_accuracy", "min"),
                best_outer_balanced_accuracy=("outer_balanced_accuracy", "max"),
                mean_outer_macro_f1=("outer_macro_f1", "mean"),
                mean_train_outer_gap=("train_outer_gap", "mean"),
            )
            .sort_values(
                ["mean_outer_balanced_accuracy", "outer_balanced_accuracy_sd"],
                ascending=[False, True],
            )
            .reset_index(drop=True)
        )
        family_summary["outer_balanced_accuracy_sd"] = family_summary["outer_balanced_accuracy_sd"].fillna(0.0)
    else:
        family_summary = pd.DataFrame()

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

    model_counts = fold_table.selected_model.value_counts().to_dict()
    family_text = ""
    if not family_summary.empty:
        leader = family_summary.iloc[0]
        family_text = (
            f" Fair outer-fold family comparison ranks {leader.model} highest on mean held-out balanced accuracy "
            f"({float(leader.mean_outer_balanced_accuracy):.3f} ± {float(leader.outer_balanced_accuracy_sd):.3f}); "
            "all families use the same outer partitions and family-specific tuning occurs only in inner CV."
        )
    summary = (
        f"Nested grouped CV balanced accuracy is {bal:.3f}; macro F1 is {macro:.3f}. "
        f"Outer-fold balanced accuracy averages {mean_outer:.3f} ± "
        f"{float(fold_table.outer_balanced_accuracy.std(ddof=1)) if len(fold_table) > 1 else 0.0:.3f}. "
        f"Training balanced accuracy averages {mean_train:.3f}, giving a training-to-outer gap of {gap:.3f} ({overfit} overfitting warning). "
        f"Worst outer fold={float(fold_table.outer_balanced_accuracy.min()):.3f}; best={float(fold_table.outer_balanced_accuracy.max()):.3f}. "
        f"Outer-fold selected model families: {model_counts}."
        + family_text
        + " Every outer test group is disjoint from its training groups; preprocessing recipe, PCA/PLS dimensionality, and model hyperparameters are chosen only inside inner grouped CV; all reported predictions are out-of-fold."
    )

    if exploratory:
        summary = ("EXPLORATORY individual-spectra validation. Repeated measurements may inflate scores; "
                   "this does not estimate performance on unseen stations or batches. "
                   + summary.replace("Nested grouped CV", "Nested individual-spectra CV")
                   .replace("Every outer test group is disjoint from its training groups",
                            "Every outer test spectrum is disjoint from its training spectra")
                   .replace("inner grouped CV", "inner individual-spectra CV"))

    return {
        "validation_mode": validation_mode,
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
        "model_family_folds": family_fold_table,
        "model_family_summary": family_summary,
        "model_family_oof_predictions": family_oof,
        "selection_rules": {
            "model_families": list(MODEL_FAMILIES),
            "inner_stability_penalty": INNER_STABILITY_PENALTY,
            "component_complexity_penalty": COMPONENT_COMPLEXITY_PENALTY,
            "preprocessing_complexity_penalty": PREPROCESSING_COMPLEXITY_PENALTY,
            "preprocessing_candidates": "transparent_guided.candidate_recipes plus the current expert recipe when distinct",
            "family_comparison": "Each model family is tuned only on inner training data and evaluated on identical outer held-out folds.",
        },
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
    validation_mode="grouped",
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
                validation_mode=validation_mode,
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
