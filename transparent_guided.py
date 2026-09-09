from __future__ import annotations

"""Transparent, rule-based guided PCA selection for FTIR Workbench v5.2.

This module intentionally contains no opaque optimizer. Every candidate recipe is
listed explicitly, every metric is returned to the UI, and the recommendation is
made by a short set of documented thresholds in ``choose_recommendation``.

Known metadata labels are NEVER used to fit PCA. When a reliable label exists,
it is used only after PCA to describe how clearly PC1/PC2 display the known groups.
Predictive-model validation remains a separate workflow in core.py.
"""

import copy
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import core


# These constants are deliberately named and documented so the automatic behavior
# is inspectable and easy to change. They are not learned or hidden parameters.
Q2_TOLERANCE_FOR_VISUAL_CHOICE = 0.08
MIN_USEFUL_GROUP_SILHOUETTE = 0.10
SNV_REQUIRED_GROUP_GAIN = 0.05
SNV_REQUIRED_Q2_GAIN = 0.03
NEAR_TIE_GROUP = 0.02
NEAR_TIE_Q2 = 0.02


def _range_step(wn):
    return core.Step(
        "Range",
        {"minimum": float(np.nanmin(wn)), "maximum": float(np.nanmax(wn))},
    )


def _recipe(name, wn, *steps):
    return name, [_range_step(wn), *steps, core.Step("Mean center", {})]


def candidate_recipes(wn, n_features):
    """Return the complete, explicit set of preprocessing recipes we compare.

    SNV is deliberately only a subset of the candidates. Non-SNV versions of the
    same common operations are always present so SNV has to demonstrate a benefit.
    """
    window = core.suggest_savgol_window(int(n_features))
    recipes = [
        _recipe("Mean-centered raw", wn),
        _recipe(
            "Smoothed + mean center",
            wn,
            core.Step("Savitzky-Golay", {"window": window, "poly": 2}),
        ),
        _recipe(
            "1st derivative + mean center",
            wn,
            core.Step("Derivative", {"order": 1, "window": window, "poly": 2}),
        ),
        _recipe(
            "2nd derivative + mean center",
            wn,
            core.Step("Derivative", {"order": 2, "window": window, "poly": 2}),
        ),
        _recipe(
            "Baseline corrected + mean center",
            wn,
            core.Step("Baseline polynomial", {"order": 2}),
        ),
        _recipe("SNV + mean center", wn, core.Step("SNV", {})),
        _recipe(
            "Smoothed + SNV + mean center",
            wn,
            core.Step("Savitzky-Golay", {"window": window, "poly": 2}),
            core.Step("SNV", {}),
        ),
        _recipe(
            "1st derivative + SNV + mean center",
            wn,
            core.Step("Derivative", {"order": 1, "window": window, "poly": 2}),
            core.Step("SNV", {}),
        ),
        _recipe(
            "Baseline corrected + SNV + mean center",
            wn,
            core.Step("Baseline polynomial", {"order": 2}),
            core.Step("SNV", {}),
        ),
    ]

    # A fixed, familiar SG setting is included as a visible benchmark when the
    # spectrum is long enough. This is not privileged; it is scored like the rest.
    if int(n_features) >= 21:
        recipes.insert(
            4,
            _recipe(
                "1st derivative (window 21, poly 3) + mean center",
                wn,
                core.Step("Derivative", {"order": 1, "window": 21, "poly": 3}),
            ),
        )
    return recipes


def recipe_uses_snv(steps) -> bool:
    return any(step.enabled and step.name == "SNV" for step in steps)


def recipe_complexity(steps) -> int:
    """Count transformations other than range selection and mean centering."""
    ignored = {"Range", "Mean center"}
    return sum(1 for step in steps if step.enabled and step.name not in ignored)


def recipe_text(steps) -> str:
    bits = []
    for step in steps:
        if not step.enabled:
            continue
        if step.params:
            params = ", ".join(f"{key}={value}" for key, value in step.params.items())
            bits.append(f"{step.name} ({params})")
        else:
            bits.append(step.name)
    return " → ".join(bits)


def _safe_label(meta: pd.DataFrame | None):
    if meta is None or meta.empty:
        return None
    schema = core.infer_metadata_schema(meta)
    label = core.suggest_label_column(meta, schema)
    if not label or label not in meta:
        return None

    row = schema.loc[schema.column == label]
    confidence = float(row.iloc[0].confidence) if len(row) else 0.0
    values = meta[label].dropna().astype(str)
    counts = values.value_counts()
    n_classes = int(values.nunique())

    # Do not let a weakly inferred or nearly unique field drive visual selection.
    if confidence < 0.55:
        return None
    if n_classes < 2 or n_classes > min(20, max(2, len(values) // 2)):
        return None
    if len(counts) < 2 or (counts >= 2).sum() < 2:
        return None
    return str(label)


def group_separation_score(scores, labels) -> float:
    """Descriptive PC1/PC2 group separation; PCA itself remains unsupervised.

    PC1 and PC2 are standardized before silhouette calculation so the score describes
    the geometry of a two-axis plot rather than letting PC1 dominate only because it
    carries more variance. This score is descriptive, not a prediction accuracy.
    """
    z = np.asarray(scores, dtype=float)
    labels = pd.Series(labels).astype(str).to_numpy()
    if z.ndim != 2 or z.shape[1] < 2 or len(z) != len(labels):
        return np.nan
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return np.nan
    counts = pd.Series(labels).value_counts()
    if (counts < 2).any():
        return np.nan
    xy = StandardScaler().fit_transform(z[:, :2])
    try:
        return float(silhouette_score(xy, labels))
    except Exception:
        return np.nan


def _best_row(frame: pd.DataFrame, column: str):
    finite = frame[np.isfinite(frame[column].to_numpy(dtype=float))]
    if finite.empty:
        return None
    return finite.sort_values(
        [column, "complexity"], ascending=[False, True]
    ).iloc[0]


def choose_recommendation(summary: pd.DataFrame, label_column: str | None):
    """Choose a default using simple, inspectable rules.

    1. Find the best reconstruction Q²-X.
    2. Keep recipes within 0.08 Q²-X of that best result.
    3. If a reliable known label exists and PC1/PC2 separation is useful, choose the
       best descriptive group separation within that reconstruction-safe set.
    4. SNV must beat the best comparable non-SNV recipe by a visible margin; if it
       does not, prefer the non-SNV recipe.
    5. Near ties prefer fewer preprocessing operations.
    """
    if summary.empty:
        raise ValueError("No PCA candidates were evaluated.")

    finite_q2 = summary[np.isfinite(summary.Q2_X.to_numpy(dtype=float))]
    if finite_q2.empty:
        eligible = summary.copy()
        best_general = summary.sort_values(
            ["RMSECV_X", "complexity"], ascending=[True, True]
        ).iloc[0]
        best_q2 = np.nan
    else:
        best_general = finite_q2.sort_values(
            ["Q2_X", "complexity"], ascending=[False, True]
        ).iloc[0]
        best_q2 = float(best_general.Q2_X)
        eligible = summary.loc[
            summary.Q2_X >= best_q2 - Q2_TOLERANCE_FOR_VISUAL_CHOICE
        ].copy()
        if eligible.empty:
            eligible = finite_q2.copy()

    best_visual = _best_row(eligible, "group_separation") if label_column else None
    use_visual = (
        best_visual is not None
        and np.isfinite(float(best_visual.group_separation))
        and float(best_visual.group_separation) >= MIN_USEFUL_GROUP_SILHOUETTE
    )

    selected = best_visual if use_visual else best_general
    mode = "known-group PC1/PC2 separation" if use_visual else "cross-validated PCA reconstruction"

    # SNV guard: normalization has to earn its place against a non-SNV candidate.
    snv_guard_applied = False
    snv_guard_message = "SNV was not the selected recipe."
    if bool(selected.uses_snv):
        non_snv = eligible.loc[~eligible.uses_snv.astype(bool)].copy()
        if not non_snv.empty:
            if use_visual:
                challenger = _best_row(non_snv, "group_separation")
                selected_gain = float(selected.group_separation)
                challenger_score = float(challenger.group_separation) if challenger is not None else np.nan
                keep_snv = (
                    challenger is None
                    or not np.isfinite(challenger_score)
                    or selected_gain >= challenger_score + SNV_REQUIRED_GROUP_GAIN
                )
                if not keep_snv and challenger is not None:
                    selected = challenger
                    snv_guard_applied = True
                    snv_guard_message = (
                        f"SNV was rejected because its PC1/PC2 group-separation gain was less than "
                        f"{SNV_REQUIRED_GROUP_GAIN:.2f} over the best non-SNV candidate."
                    )
                else:
                    snv_guard_message = (
                        f"SNV was retained because it improved descriptive group separation by at least "
                        f"{SNV_REQUIRED_GROUP_GAIN:.2f} over the best non-SNV candidate."
                    )
            else:
                challenger = non_snv.sort_values(
                    ["Q2_X", "complexity"], ascending=[False, True]
                ).iloc[0]
                selected_q2 = float(selected.Q2_X)
                challenger_q2 = float(challenger.Q2_X)
                keep_snv = selected_q2 >= challenger_q2 + SNV_REQUIRED_Q2_GAIN
                if not keep_snv:
                    selected = challenger
                    snv_guard_applied = True
                    snv_guard_message = (
                        f"SNV was rejected because its Q²-X improvement was less than "
                        f"{SNV_REQUIRED_Q2_GAIN:.2f} over the best non-SNV candidate."
                    )
                else:
                    snv_guard_message = (
                        f"SNV was retained because it improved Q²-X by at least "
                        f"{SNV_REQUIRED_Q2_GAIN:.2f} over the best non-SNV candidate."
                    )

    # Near-tie simplicity rule. Only compare candidates that are practically tied
    # on the metric that drove selection and are also similar in Q²-X.
    metric = "group_separation" if use_visual else "Q2_X"
    metric_tol = NEAR_TIE_GROUP if use_visual else NEAR_TIE_Q2
    value = float(selected[metric]) if np.isfinite(float(selected[metric])) else np.nan
    if np.isfinite(value):
        near = eligible.loc[
            np.isfinite(eligible[metric].to_numpy(dtype=float))
            & (eligible[metric] >= value - metric_tol)
        ].copy()
        if np.isfinite(float(selected.Q2_X)):
            near = near.loc[
                ~np.isfinite(near.Q2_X.to_numpy(dtype=float))
                | (near.Q2_X >= float(selected.Q2_X) - NEAR_TIE_Q2)
            ]
        if not near.empty:
            simplest = near.sort_values(
                ["complexity", metric], ascending=[True, False]
            ).iloc[0]
            if int(simplest.complexity) < int(selected.complexity):
                selected = simplest

    decision = {
        "selected_name": str(selected["name"]),
        "selection_mode": mode,
        "label_column": label_column,
        "best_general_name": str(best_general["name"]),
        "best_general_q2": float(best_general.Q2_X) if np.isfinite(float(best_general.Q2_X)) else np.nan,
        "q2_tolerance": Q2_TOLERANCE_FOR_VISUAL_CHOICE,
        "snv_guard_applied": bool(snv_guard_applied),
        "snv_guard_message": snv_guard_message,
        "rules": [
            f"Keep recipes within {Q2_TOLERANCE_FOR_VISUAL_CHOICE:.2f} Q²-X of the best reconstruction result before using a visual group-separation preference.",
            f"Use known-group PC1/PC2 separation only when its silhouette is at least {MIN_USEFUL_GROUP_SILHOUETTE:.2f}.",
            f"SNV must improve group separation by at least {SNV_REQUIRED_GROUP_GAIN:.2f}, or Q²-X by at least {SNV_REQUIRED_Q2_GAIN:.2f}, versus the best non-SNV alternative.",
            "Near ties prefer the simpler preprocessing recipe.",
        ],
    }
    return selected, decision


def compare_pca_recipes(X, wn, meta=None, max_components=20, folds=4):
    X = np.asarray(X, dtype=float)
    wn = np.asarray(wn, dtype=float)
    nmax = max(2, min(int(max_components), len(X) - 2, X.shape[1], 20))
    label_column = _safe_label(meta)

    rows = []
    results = []
    for name, steps in candidate_recipes(wn, X.shape[1]):
        cv, best = core.pca_cv(
            X,
            steps,
            nmax,
            min(int(folds), max(2, len(X) // 4)),
            wn=wn,
        )
        best = max(2, int(best))
        fitted = core.exploratory_pca(X, wn, steps, best)
        selected_row = cv.loc[cv.components == best].iloc[0]
        q2 = float(selected_row.Q2_X)
        rmsecv = float(selected_row.RMSECV_X)
        group_score = (
            group_separation_score(fitted["scores"], meta[label_column].astype(str).to_numpy())
            if label_column
            else np.nan
        )
        row = {
            "name": name,
            "components": int(best),
            "Q2_X": q2,
            "RMSECV_X": rmsecv,
            "PC1_PC2_percent": float(100 * fitted["variance"][:2].sum()),
            "group_separation": float(group_score) if np.isfinite(group_score) else np.nan,
            "label_used_for_display_score": label_column or "",
            "uses_snv": recipe_uses_snv(steps),
            "complexity": recipe_complexity(steps),
            "recipe": recipe_text(steps),
        }
        rows.append(row)
        results.append(
            {
                "name": name,
                "steps": copy.deepcopy(steps),
                "cv": cv,
                "best_components": int(best),
                "pca_result": fitted,
                **row,
            }
        )

    summary = pd.DataFrame(rows)
    selected_row, decision = choose_recommendation(summary, label_column)
    selected_name = str(selected_row["name"])

    # The UI expects an objective column but deliberately does not display it.
    # Use a simple rank marker rather than a mysterious weighted score.
    summary["objective"] = 0.0
    summary.loc[summary.name == selected_name, "objective"] = 1.0
    summary["recommendation"] = np.where(
        summary.name == selected_name,
        "DEFAULT",
        np.where(summary.name == decision["best_general_name"], "best reconstruction", "compare"),
    )

    # Put the actual default first, followed by the reconstruction leader and then
    # the remaining candidates ordered by the relevant visible metric.
    if label_column and np.isfinite(summary.group_separation).any():
        summary = summary.sort_values(
            ["objective", "group_separation", "Q2_X", "complexity"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    else:
        summary = summary.sort_values(
            ["objective", "Q2_X", "complexity"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    by_name = {result["name"]: result for result in results}
    ordered = [by_name[name] for name in summary.name]
    for result in ordered:
        result["decision"] = decision
    return ordered, summary


def _snv_summary(table: pd.DataFrame, label_column: str | None) -> str:
    snv = table.loc[table.uses_snv.astype(bool)]
    non = table.loc[~table.uses_snv.astype(bool)]
    if snv.empty or non.empty:
        return "SNV comparison was not available."

    if label_column and np.isfinite(table.group_separation).any():
        snv_best = _best_row(snv, "group_separation")
        non_best = _best_row(non, "group_separation")
        if snv_best is not None and non_best is not None:
            delta = float(snv_best.group_separation - non_best.group_separation)
            direction = "improved" if delta > 0 else "reduced"
            return (
                f"SNV check: the best SNV recipe had PC1/PC2 group-separation silhouette "
                f"{float(snv_best.group_separation):.3f}; the best non-SNV recipe had "
                f"{float(non_best.group_separation):.3f}. On this dataset SNV {direction} "
                f"that descriptive separation by {abs(delta):.3f}."
            )

    snv_best = _best_row(snv, "Q2_X")
    non_best = _best_row(non, "Q2_X")
    if snv_best is not None and non_best is not None:
        delta = float(snv_best.Q2_X - non_best.Q2_X)
        direction = "improved" if delta > 0 else "reduced"
        return (
            f"SNV check: best SNV Q²-X {float(snv_best.Q2_X):.3f} versus best non-SNV "
            f"Q²-X {float(non_best.Q2_X):.3f}; SNV {direction} reconstruction Q²-X by "
            f"{abs(delta):.3f}."
        )
    return "SNV comparison could not be summarized reliably."


def guided_analysis(X, wn, meta, max_components=20):
    schema = core.infer_metadata_schema(meta)
    suggested_label = core.suggest_label_column(meta, schema)
    suggested_groups = core.suggest_group_columns(meta, schema)

    comparisons, comparison_table = compare_pca_recipes(
        X,
        wn,
        meta=meta,
        max_components=max_components,
    )
    selected = comparisons[0]
    decision = selected["decision"]
    pca_result = selected["pca_result"]
    diagnostics = core.pca_diagnostics(pca_result)
    cluster_pcs = max(
        2,
        min(selected["best_components"], pca_result["scores"].shape[1], 10),
    )
    clustering = core.analyze_clustering(pca_result["scores"][:, :cluster_pcs])

    flagged = int(diagnostics["table"].review_flag.sum())
    q2 = float(selected["Q2_X"])
    label_for_scoring = decision.get("label_column")

    rules_text = "\n".join(f"  {i+1}. {rule}" for i, rule in enumerate(decision["rules"]))
    label_note = (
        f"Known-group display check used '{label_for_scoring}' only AFTER PCA was fitted; labels did not influence the PCA fit."
        if label_for_scoring
        else "No metadata label was trusted strongly enough to influence the default PCA view; selection used reconstruction quality only."
    )
    snv_note = _snv_summary(comparison_table, label_for_scoring)

    report = (
        f"RECOMMENDED DEFAULT\n"
        f"{selected['name']} with {selected['best_components']} PCA component(s).\n\n"
        f"WHY THIS WAS CHOSEN\n"
        f"Selection mode: {decision['selection_mode']}.\n"
        f"Cross-validated Q²-X: {q2:.3f}.\n"
        f"Best reconstruction recipe: {decision['best_general_name']} "
        f"(Q²-X {decision['best_general_q2']:.3f}).\n"
        f"{label_note}\n"
        f"{decision['snv_guard_message']}\n"
        f"{snv_note}\n\n"
        f"AUTOMATIC DECISION RULES — NO HIDDEN MODEL\n{rules_text}\n\n"
        f"WHAT ELSE WAS FOUND\n"
        f"{flagged} of {len(meta)} sample(s) are flagged for review by the PCA diagnostics. "
        f"{clustering['report']}\n\n"
        f"Predictive-model suggestion: label={suggested_label or 'none'}, "
        f"groups={', '.join(suggested_groups) if suggested_groups else 'none detected'}. "
        f"Predictive modeling uses its own grouped validation and is not validated by the PCA separation score.\n\n"
        f"All candidate recipes and their Q²-X, RMSECV-X, PC1/PC2 variance, SNV status, "
        f"and descriptive group-separation score are visible in PCA Compare. You can override the recipe in Expert mode."
    )

    return {
        "schema": schema,
        "suggested_label": suggested_label,
        "suggested_groups": suggested_groups,
        "comparisons": comparisons,
        "comparison_table": comparison_table,
        "selected": selected,
        "pca_result": pca_result,
        "diagnostics": diagnostics,
        "clustering": clustering,
        "report": report,
        "cluster_pcs": cluster_pcs,
        "decision": decision,
    }
