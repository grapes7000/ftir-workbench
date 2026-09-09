from __future__ import annotations

"""Transparent scientific audit layer for FTIR Workbench v5.2.

The functions in this module are intentionally deterministic, rule-based, and
individually testable.  They do not fit a hidden meta-model.  Every health score
is assembled from named subscores and every threshold is exposed below.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import chi2_contingency, spearmanr
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import core

RANDOM_SEED = 42
MAX_CATEGORICAL_LEVELS = 30
QC_ROBUST_Z = 3.5
FLAT_RELATIVE_STD = 0.03
SATURATION_FRACTION = 0.05
REPLICATE_CORRELATION_WARNING = 0.95
PREPROCESSING_DISTANCE_STABLE = 0.70
LOADING_CORRELATION_STABLE = 0.80
CLUSTER_STABILITY_HIGH = 0.80
CLUSTER_STABILITY_MODERATE = 0.55
STRONG_CONFOUND_V = 0.80
MODERATE_CONFOUND_V = 0.60


def _filled_matrix(X):
    x = np.asarray(X, dtype=float).copy()
    if x.ndim != 2:
        raise ValueError("Spectral matrix must be two-dimensional.")
    med = np.nanmedian(x, axis=0)
    med[~np.isfinite(med)] = 0.0
    rows, cols = np.where(~np.isfinite(x))
    if len(rows):
        x[rows, cols] = med[cols]
    return x


def _robust_z(values):
    values = np.asarray(values, dtype=float)
    center = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - center))
    if not np.isfinite(mad) or mad <= 1e-15:
        scale = np.nanstd(values)
        if not np.isfinite(scale) or scale <= 1e-15:
            return np.zeros_like(values)
        return (values - center) / scale
    return 0.67448975 * (values - center) / mad


def _schema_role_map(meta):
    if meta is None or meta.empty:
        return {}, pd.DataFrame()
    schema = core.infer_metadata_schema(meta)
    return schema.set_index("column")["role"].to_dict(), schema


def _replicate_consistency(x, meta, role_map):
    replicate_cols = [c for c in meta.columns if role_map.get(str(c)) == "replicate"]
    if not replicate_cols:
        return pd.DataFrame(), "No replicate metadata was confidently detected."

    preferred_roles = {
        "brand_or_manufacturer",
        "grade_or_product_type",
        "site_or_station",
        "batch_or_lot",
        "campaign_or_run",
        "group",
    }
    group_cols = [
        c for c in meta.columns
        if role_map.get(str(c)) in preferred_roles and c not in replicate_cols
    ][:3]
    if not group_cols:
        return pd.DataFrame(), (
            f"Replicate column '{replicate_cols[0]}' was detected, but no reliable metadata "
            "fields were available to determine which rows are replicates of the same sample."
        )

    keys = meta[group_cols].fillna("<missing>").astype(str).agg("|".join, axis=1)
    rows = []
    for key, idx in keys.groupby(keys).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        if len(idx) < 2:
            continue
        block = x[idx]
        corr = np.corrcoef(block)
        tri = corr[np.triu_indices(len(idx), 1)]
        tri = tri[np.isfinite(tri)]
        if not len(tri):
            continue
        rows.append({
            "replicate_group": key,
            "n": len(idx),
            "mean_pairwise_correlation": float(np.mean(tri)),
            "minimum_pairwise_correlation": float(np.min(tri)),
            "warning": bool(np.min(tri) < REPLICATE_CORRELATION_WARNING),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table, "Replicate metadata was detected, but no groups contained multiple spectra."
    bad = int(table.warning.sum())
    if bad:
        text = (
            f"{bad} replicate group(s) contain pairwise spectral correlation below "
            f"{REPLICATE_CORRELATION_WARNING:.2f}; inspect those groups before relying on subtle differences."
        )
    else:
        text = "Detected replicate groups are spectrally consistent at the current threshold."
    return table, text


def spectral_qc(X, wn, meta=None):
    """Compute interpretable sample-level QC flags without deleting samples."""
    raw = np.asarray(X, dtype=float)
    x = _filled_matrix(raw)
    wn = np.asarray(wn, dtype=float)
    if x.shape[1] != len(wn):
        raise ValueError("Wavenumber axis does not match the spectral matrix.")

    missing_fraction = np.mean(~np.isfinite(raw), axis=1)
    row_std = np.std(x, axis=1)
    row_range = np.ptp(x, axis=1)
    row_median = np.median(x, axis=1)
    max_abs = np.max(np.abs(x), axis=1)
    second_diff = np.diff(x, n=2, axis=1) if x.shape[1] >= 3 else np.zeros((len(x), 1))
    noise = np.median(np.abs(second_diff), axis=1) / np.maximum(row_range, 1e-12)

    grid = np.linspace(-1.0, 1.0, x.shape[1])
    slopes = np.array([np.polyfit(grid, row, 1)[0] for row in x])
    baseline_ratio = np.abs(slopes) / np.maximum(row_range, 1e-12)

    near_extreme = []
    for row, span in zip(x, row_range):
        if span <= 1e-12:
            near_extreme.append(1.0)
            continue
        tol = 0.002 * span
        fmax = np.mean(row >= np.max(row) - tol)
        fmin = np.mean(row <= np.min(row) + tol)
        near_extreme.append(max(fmax, fmin))
    near_extreme = np.asarray(near_extreme)

    std_floor = max(float(np.median(row_std) * FLAT_RELATIVE_STD), 1e-12)
    flat = row_std <= std_floor
    extreme_intensity = np.abs(_robust_z(max_abs)) > QC_ROBUST_Z
    high_noise = _robust_z(noise) > QC_ROBUST_Z
    high_baseline = _robust_z(baseline_ratio) > QC_ROBUST_Z
    saturated = (near_extreme >= SATURATION_FRACTION) & ~flat

    table = pd.DataFrame({
        "sample_index": np.arange(len(x)),
        "missing_fraction": missing_fraction,
        "signal_std": row_std,
        "signal_range": row_range,
        "median_signal": row_median,
        "max_abs_signal": max_abs,
        "noise_ratio": noise,
        "baseline_slope_ratio": baseline_ratio,
        "near_extreme_fraction": near_extreme,
        "flat_flag": flat,
        "extreme_intensity_flag": extreme_intensity,
        "high_noise_flag": high_noise,
        "possible_saturation_flag": saturated,
        "high_baseline_flag": high_baseline,
    })
    flag_cols = [c for c in table.columns if c.endswith("_flag")]
    table["qc_flag_count"] = table[flag_cols].sum(axis=1)
    table["qc_review"] = table.qc_flag_count > 0

    role_map, _ = _schema_role_map(meta if meta is not None else pd.DataFrame())
    replicate_table, replicate_text = (
        _replicate_consistency(x, meta, role_map)
        if meta is not None and not meta.empty
        else (pd.DataFrame(), "No metadata were available for replicate consistency checks.")
    )

    parts = []
    if np.any(missing_fraction > 0):
        parts.append(f"{int(np.sum(missing_fraction > 0))} sample(s) contain missing spectral values.")
    if flat.sum():
        parts.append(f"{int(flat.sum())} sample(s) are flat or nearly constant.")
    if extreme_intensity.sum():
        parts.append(f"{int(extreme_intensity.sum())} sample(s) have unusually extreme signal intensity.")
    if high_noise.sum():
        parts.append(f"{int(high_noise.sum())} sample(s) have unusually high high-frequency noise.")
    if saturated.sum():
        parts.append(f"{int(saturated.sum())} sample(s) show a possible saturation/plateau pattern.")
    if high_baseline.sum():
        parts.append(f"{int(high_baseline.sum())} sample(s) have unusually large baseline slope variation.")
    if not parts:
        parts.append("No major sample-level QC warnings were detected by the current transparent heuristics.")
    parts.append(replicate_text)

    return {
        "table": table,
        "replicate_table": replicate_table,
        "summary": " ".join(parts),
        "thresholds": {
            "robust_z": QC_ROBUST_Z,
            "flat_relative_std": FLAT_RELATIVE_STD,
            "saturation_fraction": SATURATION_FRACTION,
            "replicate_correlation_warning": REPLICATE_CORRELATION_WARNING,
        },
    }


def _cramers_v(a, b):
    table = pd.crosstab(pd.Series(a, dtype=str), pd.Series(b, dtype=str))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, np.nan
    chi2, p, _, _ = chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    denom = max(1, min(table.shape[0] - 1, table.shape[1] - 1))
    return float(math.sqrt(max(0.0, chi2 / max(n, 1) / denom))), float(p)


def _eta_squared(values, labels):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=str)
    overall = np.mean(values)
    total = np.sum((values - overall) ** 2)
    if total <= 1e-15:
        return 0.0
    between = 0.0
    for level in np.unique(labels):
        group = values[labels == level]
        if len(group):
            between += len(group) * (np.mean(group) - overall) ** 2
    return float(max(0.0, min(1.0, between / total)))


def metadata_associations(scores, meta, cluster_labels=None, permutations=49, seed=RANDOM_SEED):
    """Screen metadata associations with PCA scores and cluster membership."""
    if meta is None or meta.empty:
        return pd.DataFrame()
    z = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    n_pc = min(3, z.shape[1])

    for column in meta.columns:
        s = meta[column]
        valid = s.notna().to_numpy()
        if valid.sum() < 5:
            continue
        values = s.loc[valid]
        zv = z[valid, :n_pc]
        numeric = pd.to_numeric(values, errors="coerce")
        numeric_fraction = float(numeric.notna().mean())
        nunique = int(values.nunique())

        if numeric_fraction >= 0.95 and nunique > 5:
            arr = numeric.to_numpy(float)
            effects = []
            pvals = []
            for j in range(n_pc):
                rho, p = spearmanr(arr, zv[:, j])
                effects.append(abs(float(rho)) if np.isfinite(rho) else 0.0)
                pvals.append(float(p) if np.isfinite(p) else np.nan)
            effect = max(effects) if effects else np.nan
            pca_p = min([p for p in pvals if np.isfinite(p)], default=np.nan)
            kind = "numeric"
        elif 2 <= nunique <= min(MAX_CATEGORICAL_LEVELS, max(2, len(values) // 2)):
            labels = values.astype(str).to_numpy()
            effects = [_eta_squared(zv[:, j], labels) for j in range(n_pc)]
            effect = max(effects) if effects else np.nan
            if permutations and np.isfinite(effect):
                null = []
                for _ in range(int(permutations)):
                    perm = rng.permutation(labels)
                    null.append(max(_eta_squared(zv[:, j], perm) for j in range(n_pc)))
                pca_p = float((1 + np.sum(np.asarray(null) >= effect)) / (1 + len(null)))
            else:
                pca_p = np.nan
            kind = "categorical"
        else:
            continue

        cluster_v = np.nan
        cluster_p = np.nan
        if cluster_labels is not None and kind == "categorical":
            cl = np.asarray(cluster_labels)[valid]
            cluster_v, cluster_p = _cramers_v(values.astype(str).to_numpy(), cl.astype(str))

        if effect >= 0.50:
            strength = "strong"
        elif effect >= 0.25:
            strength = "moderate"
        elif effect >= 0.10:
            strength = "weak"
        else:
            strength = "minimal"
        rows.append({
            "metadata": str(column),
            "type": kind,
            "pca_association": float(effect),
            "pca_p_value": pca_p,
            "cluster_cramers_v": cluster_v,
            "cluster_p_value": cluster_p,
            "strength": strength,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["pca_association", "cluster_cramers_v"], ascending=False, na_position="last"
    ).reset_index(drop=True)


def confound_analysis(meta, label_column=None):
    """Identify strong categorical metadata associations that can confound interpretation."""
    if meta is None or meta.empty:
        return pd.DataFrame(), []
    candidates = []
    for column in meta.columns:
        n = int(meta[column].nunique(dropna=True))
        if 2 <= n <= min(MAX_CATEGORICAL_LEVELS, max(2, len(meta) // 2)):
            candidates.append(str(column))
    rows = []
    warnings = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            valid = meta[a].notna() & meta[b].notna()
            if valid.sum() < 5:
                continue
            v, p = _cramers_v(meta.loc[valid, a].astype(str), meta.loc[valid, b].astype(str))
            if not np.isfinite(v):
                continue
            level = "strong" if v >= STRONG_CONFOUND_V else "moderate" if v >= MODERATE_CONFOUND_V else "low"
            rows.append({"metadata_a": a, "metadata_b": b, "cramers_v": v, "p_value": p, "level": level})
            if level in {"strong", "moderate"} and (label_column is None or label_column in {a, b}):
                warnings.append(
                    f"{a} and {b} are {level}ly associated (Cramér's V={v:.2f}); effects attributed to one may partly reflect the other."
                )
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("cramers_v", ascending=False).reset_index(drop=True)
    return table, warnings


def loading_regions(pca_result, pcs=3, quantile=0.97, max_regions_per_pc=6):
    """Collapse adjacent high-loading wavenumbers into interpretable spectral regions."""
    wn = np.asarray(pca_result["wn"], dtype=float)
    loadings = np.asarray(pca_result["loadings"], dtype=float)
    if not len(wn):
        return pd.DataFrame()
    spacing = np.median(np.abs(np.diff(np.sort(wn)))) if len(wn) > 1 else 1.0
    gap = max(1e-12, 3.0 * spacing)
    rows = []
    for j in range(min(int(pcs), loadings.shape[1])):
        vector = loadings[:, j]
        cutoff = np.quantile(np.abs(vector), quantile)
        idx = np.where(np.abs(vector) >= cutoff)[0]
        if not len(idx):
            continue
        idx = idx[np.argsort(wn[idx])]
        groups = [[int(idx[0])]]
        for current in idx[1:]:
            if abs(float(wn[current] - wn[groups[-1][-1]])) <= gap:
                groups[-1].append(int(current))
            else:
                groups.append([int(current)])
        pc_rows = []
        for group in groups:
            g = np.asarray(group)
            local = vector[g]
            peak_i = int(g[np.argmax(np.abs(local))])
            pc_rows.append({
                "PC": j + 1,
                "region_min_cm-1": float(np.min(wn[g])),
                "region_max_cm-1": float(np.max(wn[g])),
                "peak_cm-1": float(wn[peak_i]),
                "peak_loading": float(vector[peak_i]),
                "direction": "positive" if vector[peak_i] >= 0 else "negative",
                "region_strength": float(np.max(np.abs(local))),
            })
        pc_rows = sorted(pc_rows, key=lambda r: r["region_strength"], reverse=True)[:max_regions_per_pc]
        rows.extend(pc_rows)
    return pd.DataFrame(rows)


def preprocessing_stability(comparisons):
    """Compare PCA geometry across reasonable preprocessing candidates."""
    if not comparisons:
        return {"table": pd.DataFrame(), "score": np.nan, "classification": "unavailable", "summary": "No preprocessing comparison was available."}
    ref = np.asarray(comparisons[0]["pca_result"]["scores"], dtype=float)
    ref_xy = StandardScaler().fit_transform(ref[:, :2])
    ref_dist = pdist(ref_xy)
    rows = []
    for candidate in comparisons:
        scores = np.asarray(candidate["pca_result"]["scores"], dtype=float)
        xy = StandardScaler().fit_transform(scores[:, :2])
        dist = pdist(xy)
        rho = spearmanr(ref_dist, dist).statistic if len(ref_dist) else np.nan
        rows.append({
            "name": candidate["name"],
            "distance_structure_correlation": float(rho) if np.isfinite(rho) else np.nan,
            "Q2_X": float(candidate.get("Q2_X", np.nan)),
            "group_separation": float(candidate.get("group_separation", np.nan)),
            "uses_snv": bool(candidate.get("uses_snv", False)),
        })
    table = pd.DataFrame(rows)
    values = table.distance_structure_correlation.to_numpy(float)
    finite = values[np.isfinite(values)]
    score = float(np.mean(finite >= PREPROCESSING_DISTANCE_STABLE)) if len(finite) else np.nan
    classification = "high" if np.isfinite(score) and score >= 0.75 else "moderate" if np.isfinite(score) and score >= 0.45 else "low"
    stable_n = int(np.sum(finite >= PREPROCESSING_DISTANCE_STABLE)) if len(finite) else 0
    summary = (
        f"PCA geometry is {classification} across preprocessing choices: {stable_n} of {len(finite)} "
        f"evaluated recipes preserve the selected model's pairwise sample-distance pattern at correlation ≥ {PREPROCESSING_DISTANCE_STABLE:.2f}."
        if len(finite)
        else "Preprocessing stability could not be estimated."
    )
    return {"table": table, "score": score, "classification": classification, "summary": summary}


def pca_resampling_stability(X, wn, steps, pca_result, n_resamples=8, component_resamples=4, seed=RANDOM_SEED):
    """Estimate loading/variance/component-count stability by deterministic subsampling."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    ref_load = np.asarray(pca_result["loadings"], dtype=float)
    ref_var = np.asarray(pca_result["variance"], dtype=float)
    n_components = ref_load.shape[1]
    corrs = [[] for _ in range(n_components)]
    variances = [[] for _ in range(n_components)]
    component_choices = []

    if len(X) < 6:
        return {"table": pd.DataFrame(), "component_choices": [], "score": np.nan, "classification": "unavailable", "summary": "Too few samples for PCA resampling stability."}

    take = max(4, int(round(len(X) * 0.80)))
    take = min(take, len(X))
    for b in range(int(n_resamples)):
        idx = np.sort(rng.choice(len(X), size=take, replace=False))
        fit = core.exploratory_pca(X[idx], wn, steps, n_components)
        load = np.asarray(fit["loadings"])
        var = np.asarray(fit["variance"])
        kmax = min(n_components, load.shape[1])
        for k in range(kmax):
            a = ref_load[:, k]
            bvec = load[:, k]
            rho = np.corrcoef(a, bvec)[0, 1]
            if np.isfinite(rho):
                corrs[k].append(abs(float(rho)))
            if k < len(var):
                variances[k].append(float(var[k]))

    for _ in range(int(component_resamples)):
        idx = np.sort(rng.choice(len(X), size=take, replace=False))
        maxpc = min(max(3, n_components + 3), 10, len(idx) - 2, X.shape[1])
        if maxpc < 2:
            continue
        try:
            _, best = core.pca_cv(X[idx], steps, maxpc, min(3, len(idx)), wn=wn)
            component_choices.append(int(best))
        except Exception:
            pass

    rows = []
    for k in range(n_components):
        rows.append({
            "PC": k + 1,
            "median_loading_abs_correlation": float(np.median(corrs[k])) if corrs[k] else np.nan,
            "explained_variance_mean": float(np.mean(variances[k])) if variances[k] else np.nan,
            "explained_variance_sd": float(np.std(variances[k], ddof=1)) if len(variances[k]) > 1 else 0.0 if variances[k] else np.nan,
            "reference_explained_variance": float(ref_var[k]) if k < len(ref_var) else np.nan,
        })
    table = pd.DataFrame(rows)
    first = table.head(min(3, len(table))).median_loading_abs_correlation.to_numpy(float)
    first = first[np.isfinite(first)]
    score = float(np.mean(first)) if len(first) else np.nan
    classification = "high" if np.isfinite(score) and score >= LOADING_CORRELATION_STABLE else "moderate" if np.isfinite(score) and score >= 0.60 else "low"
    component_text = (
        f"Resampled PCA selected {min(component_choices)}–{max(component_choices)} components (median {np.median(component_choices):.1f})."
        if component_choices else "Component-count resampling was unavailable."
    )
    summary = (
        f"Loading stability is {classification}; the median absolute loading correlation for the leading PCs is {score:.2f}. {component_text}"
        if np.isfinite(score) else component_text
    )
    return {"table": table, "component_choices": component_choices, "score": score, "classification": classification, "summary": summary}


def cluster_stability(pca_result, clustering, n_resamples=12, seed=RANDOM_SEED):
    """Perturb score space/component count and measure assignment agreement by ARI."""
    scores = np.asarray(pca_result["scores"], dtype=float)
    base = np.asarray(clustering["best_labels"])
    method = clustering["best_method"]
    params = clustering["best_params"]
    k = int(params.get("k", len(np.unique(base[base >= 0])) or 2))
    rng = np.random.default_rng(seed)
    aris = []
    dimensions = sorted(set([max(2, min(scores.shape[1], d)) for d in (2, 3, min(5, scores.shape[1]))]))

    for i in range(int(n_resamples)):
        d = dimensions[i % len(dimensions)]
        z = scores[:, :d].copy()
        scale = np.std(z, axis=0, keepdims=True)
        z += rng.normal(0, 0.01, z.shape) * np.maximum(scale, 1e-12)
        if method == "KMeans":
            labels = KMeans(k, n_init=30, random_state=seed + i).fit_predict(z)
        elif method == "Agglomerative":
            labels = AgglomerativeClustering(k, linkage="ward").fit_predict(z)
        else:
            eps = float(params.get("eps", 0.8))
            min_samples = int(params.get("min_samples", 3))
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(z)
        aris.append(float(adjusted_rand_score(base, labels)))

    score = float(np.median(aris)) if aris else np.nan
    classification = "high" if np.isfinite(score) and score >= CLUSTER_STABILITY_HIGH else "moderate" if np.isfinite(score) and score >= CLUSTER_STABILITY_MODERATE else "low"
    summary = (
        f"Cluster stability is {classification}: median adjusted Rand agreement under small perturbations/component changes is {score:.2f}."
        if np.isfinite(score) else "Cluster stability could not be estimated."
    )
    return {"scores": aris, "score": score, "classification": classification, "summary": summary}


def _independent_group_support(meta, label, groups):
    if not label or label not in meta or not groups or not all(c in meta for c in groups):
        return pd.DataFrame(), np.nan
    group_id = core.make_groups(meta, groups)
    frame = pd.DataFrame({"label": meta[label].astype(str), "group": group_id}).drop_duplicates()
    counts = frame.groupby("label").size().rename("independent_groups").reset_index()
    return counts, float(counts.independent_groups.min()) if len(counts) else np.nan


def model_health(guided, qc, prep_stability, pca_stability, cluster_stability_result, confounds, support_min):
    """Transparent descriptive health score; not an established statistical index."""
    scores = []
    reasons = []

    q2 = float(guided["selected"].get("Q2_X", np.nan))
    if np.isfinite(q2):
        q2_score = 95 if q2 >= 0.90 else 85 if q2 >= 0.75 else 70 if q2 >= 0.50 else 50 if q2 >= 0.25 else 25
        scores.append(("PCA cross-validation", q2_score))
        reasons.append(f"PCA cross-validated Q²-X is {q2:.3f}.")

    qc_burden = float(qc["table"].qc_review.mean()) if len(qc["table"]) else 0.0
    qc_score = 100 if qc_burden <= 0.03 else 85 if qc_burden <= 0.08 else 65 if qc_burden <= 0.15 else 40 if qc_burden <= 0.30 else 20
    scores.append(("Data QC", qc_score))
    reasons.append(f"{qc_burden:.1%} of samples have at least one QC review flag.")

    if np.isfinite(prep_stability.get("score", np.nan)):
        ps = float(prep_stability["score"])
        scores.append(("Preprocessing stability", round(100 * ps)))
        reasons.append(prep_stability["summary"])

    if np.isfinite(pca_stability.get("score", np.nan)):
        ls = float(pca_stability["score"])
        scores.append(("Loading stability", round(100 * ls)))
        reasons.append(pca_stability["summary"])

    if np.isfinite(cluster_stability_result.get("score", np.nan)):
        cs = float(cluster_stability_result["score"])
        scores.append(("Cluster stability", round(max(0, min(100, 100 * cs)))))
        reasons.append(cluster_stability_result["summary"])

    if confounds is not None and not confounds.empty:
        vmax = float(confounds.cramers_v.max())
        conf_score = 35 if vmax >= STRONG_CONFOUND_V else 65 if vmax >= MODERATE_CONFOUND_V else 90
        scores.append(("Metadata confounding", conf_score))
        reasons.append(f"Strongest categorical metadata association has Cramér's V={vmax:.2f}.")
    else:
        scores.append(("Metadata confounding", 95))
        reasons.append("No strong categorical metadata confound was detected among analyzable fields.")

    if np.isfinite(support_min):
        support_score = 100 if support_min >= 6 else 90 if support_min >= 5 else 80 if support_min >= 4 else 65 if support_min >= 3 else 40 if support_min >= 2 else 20
        scores.append(("Independent-group support", support_score))
        reasons.append(f"Smallest class has {int(support_min)} independent validation group(s).")

    table = pd.DataFrame(scores, columns=["subscore", "score_0_100"])
    overall = float(table.score_0_100.mean()) if len(table) else np.nan
    band = "GREEN — strong / trustworthy" if np.isfinite(overall) and overall >= 75 else "YELLOW — useful but caution required" if np.isfinite(overall) and overall >= 50 else "RED — unreliable / insufficient evidence"
    return {
        "table": table,
        "overall": overall,
        "band": band,
        "reasons": reasons,
        "disclaimer": "This 0–100 health score is a transparent engineering summary, not an established statistical quantity. Inspect the subscores and evidence before making scientific claims.",
    }


def _pca_component_narrative(cv, best):
    if cv is None or len(cv) == 0:
        return "PCA component cross-validation details were unavailable."
    cv = cv.sort_values("components")
    row = cv.loc[cv.components == best]
    if row.empty:
        return f"{best} PCs were selected by the configured parsimonious rule."
    best_rmse = float(row.iloc[0].RMSECV_X)
    later = cv.loc[cv.components > best]
    if later.empty:
        return f"{best} PCs are recommended by the parsimonious cross-validation rule."
    improvement = best_rmse - float(later.RMSECV_X.min())
    relative = improvement / max(abs(best_rmse), 1e-12)
    if relative < 0.01:
        extra = "Additional PCs improve validation error by less than about 1%, so added complexity is not justified."
    elif relative < 0.03:
        extra = "Additional PCs provide only a small marginal reduction in validation error."
    else:
        extra = "Some later PCs reduce validation error, but the one-standard-error-style rule prefers the smaller model because the improvement is not robust relative to fold variation."
    return f"{best} PCs are recommended. {extra}"


def comprehensive_analysis(X, wn, meta, guided, n_resamples=8):
    """Run the full post-exclusion scientific audit on the selected guided model."""
    pca_result = guided["pca_result"]
    qc = spectral_qc(X, wn, meta)
    prep = preprocessing_stability(guided.get("comparisons", []))
    pca_stab = pca_resampling_stability(
        X, wn, guided["selected"]["steps"], pca_result,
        n_resamples=n_resamples,
        component_resamples=max(2, min(4, n_resamples // 2)),
    )
    cluster_stab = cluster_stability(pca_result, guided["clustering"], n_resamples=max(6, n_resamples))
    load_regions = loading_regions(pca_result)
    label = guided.get("suggested_label")
    groups = guided.get("suggested_groups") or []
    assoc = metadata_associations(
        pca_result["scores"], meta,
        cluster_labels=guided["clustering"]["best_labels"],
    )
    confounds, confound_warnings = confound_analysis(meta, label)
    support_table, support_min = _independent_group_support(meta, label, groups)
    health = model_health(guided, qc, prep, pca_stab, cluster_stab, confounds, support_min)

    diag = guided["diagnostics"]["table"]
    flagged = int(diag.review_flag.sum())
    dominance = float(np.max(diag.Hotelling_T2) / max(guided["diagnostics"]["t2_limit"], 1e-12)) if len(diag) else np.nan
    component_text = _pca_component_narrative(guided["selected"].get("cv"), guided["selected"]["best_components"])

    loading_text = "No stable influential loading regions were identified."
    if not load_regions.empty:
        peaks = load_regions.sort_values("region_strength", ascending=False).head(6).peak_cm_1.tolist() if "peak_cm_1" in load_regions else load_regions.sort_values("region_strength", ascending=False).head(6)["peak_cm-1"].tolist()
        loading_text = "Leading PCA loadings are strongly influenced near approximately " + ", ".join(f"{v:.0f}" for v in peaks) + " cm⁻¹. These are influential regions, not compound identifications."

    association_text = "No metadata variable showed a strong reproducible association with the leading PCA scores under the current screening rules."
    if not assoc.empty:
        top = assoc.iloc[0]
        if float(top.pca_association) >= 0.25:
            association_text = f"'{top.metadata}' shows the strongest metadata association with the leading PCA structure (effect {float(top.pca_association):.2f}, descriptive strength {top.strength})."

    confound_text = "No major confounding warning was identified for the suggested label."
    if confound_warnings:
        confound_text = " ".join(confound_warnings[:3])

    if dominance > 2.0:
        dominance_text = "At least one sample has very high score distance relative to the review limit, so a small number of samples may be disproportionately influencing PCA structure."
    else:
        dominance_text = "No single sample has an extreme score-distance dominance signal under the current PCA diagnostic threshold."

    report = (
        f"MODEL HEALTH\n{health['band']}"
        + (f" ({health['overall']:.0f}/100 transparent summary score)" if np.isfinite(health["overall"]) else "")
        + "\n\nDATA QUALITY\n"
        + qc["summary"]
        + "\n\nPCA VALIDATION\n"
        + component_text
        + f" Cross-validated Q²-X is {float(guided['selected']['Q2_X']):.3f}. "
        + f"{flagged} sample(s) have high T² and/or Q residual and should be inspected rather than automatically deleted. "
        + dominance_text
        + "\n\nSTABILITY\n"
        + prep["summary"] + " " + pca_stab["summary"] + " " + cluster_stab["summary"]
        + "\n\nSPECTRAL DRIVERS\n"
        + loading_text
        + "\n\nMETADATA / CONFOUNDING\n"
        + association_text + " " + confound_text
        + "\n\nINTERPRETATION BOUNDARY\n"
        + "PCA and clustering are exploratory: they show reproducible structure, not predictive accuracy. A predictive claim requires held-out grouped/nested validation on genuinely unseen samples or groups."
        + "\n\nHEALTH SCORE NOTE\n" + health["disclaimer"]
    )

    return {
        "qc": qc,
        "preprocessing_stability": prep,
        "pca_stability": pca_stab,
        "cluster_stability": cluster_stab,
        "loading_regions": load_regions,
        "metadata_associations": assoc,
        "confounds": confounds,
        "confound_warnings": confound_warnings,
        "independent_group_support": support_table,
        "health": health,
        "report": report,
    }


def export_analysis(result, out_dir):
    """Write audit-friendly scientific-analysis tables and plain-English report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result["qc"]["table"].to_csv(out / "spectral_qc.csv", index=False)
    result["qc"]["replicate_table"].to_csv(out / "replicate_consistency.csv", index=False)
    result["preprocessing_stability"]["table"].to_csv(out / "preprocessing_stability.csv", index=False)
    result["pca_stability"]["table"].to_csv(out / "pca_resampling_stability.csv", index=False)
    result["loading_regions"].to_csv(out / "pca_loading_regions.csv", index=False)
    result["metadata_associations"].to_csv(out / "metadata_associations.csv", index=False)
    result["confounds"].to_csv(out / "metadata_confounds.csv", index=False)
    result["independent_group_support"].to_csv(out / "independent_group_support.csv", index=False)
    result["health"]["table"].to_csv(out / "model_health_subscores.csv", index=False)
    (out / "smart_analysis_report.txt").write_text(result["report"], encoding="utf-8")
    payload = {
        "overall_health": result["health"]["overall"],
        "health_band": result["health"]["band"],
        "health_disclaimer": result["health"]["disclaimer"],
        "pca_component_choices_resampling": result["pca_stability"]["component_choices"],
        "cluster_stability_ari": result["cluster_stability"]["scores"],
        "thresholds": result["qc"]["thresholds"],
    }
    (out / "model_health.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
