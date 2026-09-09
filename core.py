from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.signal import savgol_filter
from scipy.stats import f as f_distribution
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, normalize
from sklearn.svm import SVC

META_DEFAULT = ["SampleID", "Brand", "Grade", "City", "State", "Year"]
METADATA_ROLES = [
    "sample_id",
    "brand_or_manufacturer",
    "grade_or_product_type",
    "supplier",
    "site_or_station",
    "city",
    "state_or_province",
    "region",
    "country",
    "batch_or_lot",
    "campaign_or_run",
    "year",
    "date",
    "replicate",
    "class_label",
    "group",
    "other",
]

ROLE_ALIASES = {
    "sample_id": (
        "sample", "sampleid", "sample_id", "specimen", "specimenid", "specimen_id",
        "samplecode", "sample_code", "bottleid", "bottle_id", "recordid", "record_id",
        "id",
    ),
    "brand_or_manufacturer": (
        "brand", "manufacturer", "maker", "make", "mfr", "producer", "company",
        "brandname", "brand_name", "manufacturername", "manufacturer_name",
    ),
    "grade_or_product_type": (
        "grade", "product", "producttype", "product_type", "type", "fueltype", "fuel_type",
        "material", "materialtype", "material_type", "formulation",
    ),
    "supplier": ("supplier", "vendor", "distributor", "source"),
    "site_or_station": (
        "site", "station", "locationid", "location_id", "siteid", "site_id",
        "stationid", "station_id", "store", "facility", "collectionsite", "collection_site",
    ),
    "city": ("city", "town", "municipality"),
    "state_or_province": ("state", "province", "stateprovince", "state_province"),
    "region": ("region", "area", "district", "zone", "territory"),
    "country": ("country", "nation"),
    "batch_or_lot": ("batch", "lot", "batchid", "batch_id", "lotid", "lot_id"),
    "campaign_or_run": (
        "campaign", "run", "collectionrun", "collection_run", "survey", "visit", "trip",
        "session",
    ),
    "year": ("year", "yr", "collectionyear", "collection_year", "productionyear", "production_year"),
    "date": ("date", "collectiondate", "collection_date", "timestamp", "datetime", "time"),
    "replicate": ("replicate", "rep", "repeat", "technicalreplicate", "technical_replicate"),
    "class_label": ("class", "label", "target", "category"),
    "group": ("group", "groupid", "group_id", "subject", "subjectid", "subject_id"),
}


@dataclass
class Step:
    name: str
    params: dict
    enabled: bool = True


DEFAULTS = {
    "Range": {"minimum": 400.0, "maximum": 4000.0},
    "Exclude": {"ranges": ""},
    "Baseline polynomial": {"order": 2},
    "Savitzky-Golay": {"window": 11, "poly": 2},
    "Derivative": {"order": 1, "window": 11, "poly": 2},
    "SNV": {},
    "Vector normalize": {},
    "Area normalize": {},
    "Mean center": {},
    "Autoscale": {},
    "Robust scale": {},
}


def _usable_headers(headers: Iterable[object], count: int) -> list[str]:
    headers = [str(x).strip() for x in list(headers)[:count]]
    bad = 0
    for h in headers:
        if not h or h.lower().startswith("unnamed:") or re.fullmatch(r"\d+(\.\d+)?", h):
            bad += 1
    if headers and bad < len(headers):
        return [
            (h if h and not h.lower().startswith("unnamed:") else f"Metadata_{i+1}")
            for i, h in enumerate(headers)
        ]
    fallback = META_DEFAULT[:count]
    if count > len(fallback):
        fallback += [f"Metadata_{i+1}" for i in range(len(fallback), count)]
    return fallback


def load_ftir(path):
    raw = pd.read_csv(path, low_memory=False)
    marker = raw.iloc[0, 0] if len(raw) else None

    if isinstance(marker, str) and "wavenumber" in marker.lower():
        start = 7
        wn = pd.to_numeric(raw.iloc[0, start:], errors="coerce").to_numpy(float)
        meta = raw.iloc[1:, 1:start].copy().reset_index(drop=True)
        meta.columns = _usable_headers(raw.columns[1:start], meta.shape[1])
        spectra = raw.iloc[1:, start:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    else:
        numeric = []
        for column in raw.columns:
            try:
                numeric.append((column, float(column)))
            except (TypeError, ValueError):
                pass
        if not numeric:
            raise ValueError("No numeric wavenumber columns detected.")
        columns = [column for column, _ in numeric]
        wn = np.array([value for _, value in numeric], dtype=float)
        spectra = raw[columns].apply(pd.to_numeric, errors="coerce")
        meta = raw.drop(columns=columns)

    good = np.isfinite(wn)
    wn = wn[good]
    spectra = spectra.loc[:, good]
    order = np.argsort(wn)
    return (
        meta.reset_index(drop=True),
        wn[order],
        spectra.iloc[:, order].reset_index(drop=True),
    )


def parse_ranges(text):
    ranges = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.replace("–", "-").split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid range '{token}'. Use values like 500-1000.")
        a, b = map(float, parts)
        ranges.append((min(a, b), max(a, b)))
    return ranges


def snv(x):
    x = np.asarray(x, dtype=float)
    sd = x.std(axis=1, keepdims=True)
    sd[sd == 0] = 1
    return (x - x.mean(axis=1, keepdims=True)) / sd


def apply_recipe(X, wn, steps, state=None):
    x = np.asarray(X, dtype=float).copy()
    w = np.asarray(wn, dtype=float).copy()
    fitting = state is None
    state = {} if state is None else state

    for i, step in enumerate(steps):
        if not step.enabled:
            continue
        name, params = step.name, step.params

        if name == "Range":
            mask = (w >= float(params["minimum"])) & (w <= float(params["maximum"]))
            x, w = x[:, mask], w[mask]
        elif name == "Exclude":
            mask = np.ones(len(w), dtype=bool)
            for a, b in parse_ranges(params.get("ranges", "")):
                mask &= ~((w >= a) & (w <= b))
            x, w = x[:, mask], w[mask]
        elif name == "Baseline polynomial":
            grid = np.linspace(-1, 1, x.shape[1])
            order = int(params["order"])
            x = np.array(
                [row - np.polyval(np.polyfit(grid, row, order), grid) for row in x]
            )
        elif name in ("Savitzky-Golay", "Derivative"):
            poly = int(params["poly"])
            window = max(int(params["window"]), poly + 2)
            window += window % 2 == 0
            max_window = x.shape[1] if x.shape[1] % 2 else x.shape[1] - 1
            window = min(window, max_window)
            if window <= poly:
                raise ValueError("Too few variables remain for Savitzky-Golay preprocessing.")
            deriv = int(params.get("order", 0)) if name == "Derivative" else 0
            x = savgol_filter(x, window, poly, deriv=deriv, axis=1)
        elif name == "SNV":
            x = snv(x)
        elif name == "Vector normalize":
            x = normalize(x)
        elif name == "Area normalize":
            area = np.trapz(np.abs(x), w, axis=1)
            area[area == 0] = 1
            x = x / area[:, None]
        elif name in ("Mean center", "Autoscale", "Robust scale"):
            if fitting:
                scaler = (
                    RobustScaler()
                    if name == "Robust scale"
                    else StandardScaler(with_std=name == "Autoscale")
                )
                scaler.fit(x)
                state[i] = scaler
            x = state[i].transform(x)

    if x.shape[1] < 2:
        raise ValueError("Recipe leaves fewer than 2 spectral variables.")
    return x, w, state


class RecipeTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible preprocessing that preserves the real spectral axis."""

    def __init__(self, steps, wn=None):
        self.steps = steps
        self.wn = wn

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.imp_ = SimpleImputer(strategy="median").fit(X)
        self.wn_ = (
            np.arange(X.shape[1], dtype=float)
            if self.wn is None
            else np.asarray(self.wn, dtype=float)
        )
        if len(self.wn_) != X.shape[1]:
            raise ValueError("Wavenumber axis length does not match the spectral matrix.")
        _, self.output_wn_, self.state_ = apply_recipe(
            self.imp_.transform(X), self.wn_, self.steps
        )
        return self

    def transform(self, X):
        transformed, _, _ = apply_recipe(
            self.imp_.transform(X), self.wn_, self.steps, self.state_
        )
        return transformed


def exploratory_pca(X, wn, steps, components=20):
    imputed = SimpleImputer(strategy="median").fit_transform(X)
    processed, retained_wn, _ = apply_recipe(imputed, wn, steps)
    n_components = max(1, min(int(components), len(processed) - 1, processed.shape[1]))
    pca = PCA(n_components=n_components, svd_solver="full").fit(processed)
    return {
        "processed": processed,
        "wn": retained_wn,
        "scores": pca.transform(processed),
        "loadings": pca.components_.T,
        "variance": pca.explained_variance_ratio_,
        "pca": pca,
        "steps": copy.deepcopy(steps),
    }


def pca_cv(X, steps, max_components=30, folds=5, groups=None, wn=None):
    """Efficient reconstruction CV: one PCA fit per fold, then nested reconstructions."""
    X = np.asarray(X, dtype=float)
    if len(X) < 4:
        raise ValueError("At least four samples are required for PCA cross-validation.")

    nmax = max(1, min(int(max_components), len(X) - 2, X.shape[1]))
    if groups is not None:
        unique_groups = np.unique(groups)
        n_splits = min(int(folds), len(unique_groups))
        if n_splits < 2:
            raise ValueError("At least two independent groups are required for grouped PCA CV.")
        splitter = GroupKFold(n_splits)
        splits = list(splitter.split(X, groups=groups))
    else:
        n_splits = min(int(folds), len(X))
        if n_splits < 2:
            raise ValueError("At least two folds are required.")
        splitter = KFold(n_splits, shuffle=True, random_state=42)
        splits = list(splitter.split(X))

    fold_metrics = {n: {"cal": [], "val": [], "press": [], "tss": []} for n in range(1, nmax + 1)}
    for train, test in splits:
        prep = RecipeTransformer(steps, wn=wn).fit(X[train])
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
            cal_mse = float(np.mean(rc**2))
            val_mse = float(np.mean(rv**2))
            fold_metrics[n]["cal"].append(math.sqrt(cal_mse))
            fold_metrics[n]["val"].append(math.sqrt(val_mse))
            fold_metrics[n]["press"].append(float(np.sum(rv**2)))
            fold_metrics[n]["tss"].append(float(np.sum((validation - calibration.mean(0)) ** 2)))

    rows = []
    for n in range(1, nmax + 1):
        metric = fold_metrics[n]
        press = float(np.sum(metric["press"]))
        tss = float(np.sum(metric["tss"]))
        rows.append(
            {
                "components": n,
                "RMSEC_X": float(np.mean(metric["cal"])),
                "RMSECV_X": float(np.mean(metric["val"])),
                "RMSECV_X_SD": float(np.std(metric["val"], ddof=1)) if len(metric["val"]) > 1 else 0.0,
                "PRESS_X": press,
                "Q2_X": 1 - press / tss if tss else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    minimum_index = int(df.RMSECV_X.idxmin())
    threshold = float(df.loc[minimum_index, "RMSECV_X"] + df.loc[minimum_index, "RMSECV_X_SD"])
    candidates = df.loc[df.RMSECV_X <= threshold, "components"]
    best = int(candidates.min()) if len(candidates) else int(df.loc[minimum_index, "components"])
    return df, best


def embedding(X, method, components=2, **kwargs):
    if method == "PCA":
        return PCA(components).fit_transform(X)
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError("UMAP unavailable. Install umap-learn.") from exc
    return umap.UMAP(
        n_components=components,
        n_neighbors=kwargs.get("n_neighbors", 15),
        min_dist=kwargs.get("min_dist", 0.1),
        metric=kwargs.get("metric", "euclidean"),
        random_state=kwargs.get("random_state", 42),
    ).fit_transform(X)


def cluster(z, method, k=3, eps=0.8, min_samples=3):
    z = np.asarray(z, dtype=float)
    if method == "KMeans":
        labels = KMeans(k, n_init=30, random_state=42).fit_predict(z)
    elif method == "Agglomerative":
        labels = AgglomerativeClustering(k, linkage="ward").fit_predict(z)
    else:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(z)
    valid = 1 < len(set(labels)) < len(labels)
    return labels, silhouette_score(z, labels) if valid else np.nan


def kmeans_history(z, k, max_iter=30):
    z = np.asarray(z, dtype=float)
    rng = np.random.default_rng(42)
    centers = z[rng.choice(len(z), k, replace=False)].copy()
    frames = []
    for iteration in range(max_iter):
        labels = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(2).argmin(1)
        frames.append(
            {"iteration": iteration, "labels": labels.copy(), "centers": centers.copy()}
        )
        new_centers = np.array(
            [
                z[labels == j].mean(0) if np.any(labels == j) else centers[j]
                for j in range(k)
            ]
        )
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return frames


def _normalized_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _header_role_score(column: object, role: str) -> float:
    normalized = _normalized_name(column)
    aliases = ROLE_ALIASES.get(role, ())
    best = 0.0
    for alias in aliases:
        alias_n = _normalized_name(alias)
        if normalized == alias_n:
            best = max(best, 0.92)
        elif alias_n and alias_n in normalized:
            best = max(best, 0.78)
    return best


def _value_role_scores(series: pd.Series) -> dict[str, float]:
    values = series.dropna()
    if values.empty:
        return {}
    text = values.astype(str).str.strip()
    n = len(text)
    unique = text.nunique(dropna=True)
    unique_ratio = unique / max(1, n)
    scores: dict[str, float] = {}

    numeric = pd.to_numeric(values, errors="coerce")
    numeric_fraction = float(numeric.notna().mean())
    if numeric_fraction > 0.9:
        finite = numeric.dropna()
        if len(finite):
            year_fraction = float(((finite >= 1900) & (finite <= 2100) & (finite % 1 == 0)).mean())
            if year_fraction > 0.8:
                scores["year"] = 0.82

    upper = text.str.upper()
    if unique <= 70 and float(upper.str.fullmatch(r"[A-Z]{2,3}").mean()) > 0.8:
        scores["state_or_province"] = 0.58

    if unique_ratio > 0.85 and unique >= max(5, int(n * 0.5)):
        scores["sample_id"] = max(scores.get("sample_id", 0), 0.48)

    # Date parsing is intentionally conservative so ordinary labels are not mistaken for dates.
    if n >= 3 and text.str.match(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}").mean() > 0.5:
        parsed = pd.to_datetime(text, errors="coerce")
        if float(parsed.notna().mean()) > 0.8:
            scores["date"] = 0.68

    if 2 <= unique <= max(30, int(n * 0.5)):
        scores["class_label"] = 0.34
    return scores


def infer_metadata_schema(meta: pd.DataFrame) -> pd.DataFrame:
    """Infer semantic metadata roles from both headers and observed values."""
    rows = []
    for column in meta.columns:
        series = meta[column]
        value_scores = _value_role_scores(series)
        role_scores = {}
        for role in METADATA_ROLES:
            if role == "other":
                continue
            header_score = _header_role_score(column, role)
            value_score = value_scores.get(role, 0.0)
            # Header semantics are strongest; compatible values increase confidence.
            role_scores[role] = min(0.99, max(header_score, value_score, header_score + 0.12 * value_score))

        role, confidence = max(role_scores.items(), key=lambda item: item[1]) if role_scores else ("other", 0.0)
        if confidence < 0.42:
            role = "other"
            confidence = max(confidence, 0.2)

        values = series.dropna().astype(str)
        example = ", ".join(values.head(3).tolist())
        reason_bits = []
        if _header_role_score(column, role) >= 0.5:
            reason_bits.append("column name")
        if value_scores.get(role, 0) >= 0.45:
            reason_bits.append("value pattern")
        reason = " + ".join(reason_bits) if reason_bits else "weak/ambiguous evidence"

        rows.append(
            {
                "column": str(column),
                "role": role,
                "confidence": float(confidence),
                "unique_values": int(series.nunique(dropna=True)),
                "missing_percent": float(series.isna().mean() * 100),
                "example": example,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def suggest_label_column(meta: pd.DataFrame, schema: pd.DataFrame | None = None):
    if meta.empty:
        return None
    schema = infer_metadata_schema(meta) if schema is None else schema
    priorities = [
        "brand_or_manufacturer",
        "grade_or_product_type",
        "supplier",
        "class_label",
        "region",
    ]
    for role in priorities:
        candidates = schema.loc[schema.role == role].sort_values("confidence", ascending=False)
        for column in candidates.column:
            nunique = meta[column].nunique(dropna=True)
            counts = meta[column].value_counts(dropna=True)
            if 2 <= nunique <= max(50, len(meta) // 2) and (counts >= 2).sum() >= 2:
                return column

    # Fallback to a sensible categorical metadata column.
    for column in meta.columns:
        nunique = meta[column].nunique(dropna=True)
        if 2 <= nunique <= min(30, max(2, len(meta) // 2)):
            return str(column)
    return None


def suggest_group_columns(meta: pd.DataFrame, schema: pd.DataFrame | None = None):
    if meta.empty:
        return []
    schema = infer_metadata_schema(meta) if schema is None else schema
    priorities = [
        "site_or_station",
        "batch_or_lot",
        "campaign_or_run",
        "group",
        "city",
        "state_or_province",
    ]
    selected = []
    for role in priorities:
        matches = schema.loc[schema.role == role].sort_values("confidence", ascending=False)
        if not matches.empty:
            column = str(matches.iloc[0].column)
            if meta[column].nunique(dropna=True) >= 2:
                selected.append(column)
        if len(selected) >= 2:
            break
    return selected


def make_groups(meta, cols):
    if not cols:
        return np.arange(len(meta)).astype(str)
    missing = [column for column in cols if column not in meta]
    if missing:
        raise ValueError(f"Grouping columns not found: {', '.join(missing)}")
    return meta[cols].fillna("<missing>").astype(str).agg("|".join, axis=1).to_numpy()


def eligibility(meta, label, cols, min_groups):
    y = meta[label].astype(str).to_numpy()
    groups = make_groups(meta, cols)
    counts = (
        pd.DataFrame({"label": y, "group": groups})
        .drop_duplicates()
        .groupby("label")
        .size()
    )
    result = pd.DataFrame(
        {
            "class": counts.index,
            "independent_groups": counts.values,
            "eligible": counts.values >= min_groups,
        }
    )
    return result, y, groups


def suggest_savgol_window(n_features: int) -> int:
    """Choose a conservative odd SG window from spectral resolution, not a magic constant."""
    if n_features < 7:
        return max(3, n_features if n_features % 2 else n_features - 1)
    target = max(7, min(31, int(round(n_features * 0.015))))
    if target % 2 == 0:
        target += 1
    return min(target, n_features if n_features % 2 else n_features - 1)


def _candidate_recipes(wn, n_features):
    window = suggest_savgol_window(n_features)
    minimum, maximum = float(np.nanmin(wn)), float(np.nanmax(wn))
    range_step = Step("Range", {"minimum": minimum, "maximum": maximum})
    return [
        (
            "Mean-centered",
            [copy.deepcopy(range_step), Step("Mean center", {})],
        ),
        (
            "SNV",
            [copy.deepcopy(range_step), Step("SNV", {}), Step("Mean center", {})],
        ),
        (
            "Smoothed + SNV",
            [
                copy.deepcopy(range_step),
                Step("Savitzky-Golay", {"window": window, "poly": 2}),
                Step("SNV", {}),
                Step("Mean center", {}),
            ],
        ),
        (
            "1st derivative + SNV",
            [
                copy.deepcopy(range_step),
                Step("Derivative", {"order": 1, "window": window, "poly": 2}),
                Step("SNV", {}),
                Step("Mean center", {}),
            ],
        ),
        (
            "Baseline + SNV",
            [
                copy.deepcopy(range_step),
                Step("Baseline polynomial", {"order": 2}),
                Step("SNV", {}),
                Step("Mean center", {}),
            ],
        ),
    ]


def compare_pca_recipes(X, wn, meta=None, max_components=20, folds=4):
    X = np.asarray(X, dtype=float)
    nmax = max(2, min(int(max_components), len(X) - 2, X.shape[1], 20))
    rows = []
    results = []
    for complexity, (name, steps) in enumerate(_candidate_recipes(wn, X.shape[1]), start=1):
        cv, best = pca_cv(X, steps, nmax, min(folds, max(2, len(X) // 4)), wn=wn)
        best = max(2, int(best))
        fitted = exploratory_pca(X, wn, steps, best)
        selected_row = cv.loc[cv.components == best].iloc[0]
        q2 = float(selected_row.Q2_X)
        rmsecv = float(selected_row.RMSECV_X)
        # Favor predictive reconstruction while mildly preferring simpler preprocessing.
        objective = (q2 if np.isfinite(q2) else -1e9) - 0.008 * (complexity - 1)
        row = {
            "name": name,
            "components": int(best),
            "Q2_X": q2,
            "RMSECV_X": rmsecv,
            "PC1_PC2_percent": float(100 * fitted["variance"][:2].sum()),
            "objective": float(objective),
        }
        rows.append(row)
        results.append(
            {
                "name": name,
                "steps": steps,
                "cv": cv,
                "best_components": int(best),
                "pca_result": fitted,
                **row,
            }
        )
    summary = pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)
    order = list(summary.name)
    results = sorted(results, key=lambda r: order.index(r["name"]))
    return results, summary


def pca_diagnostics(pca_result, alpha=0.95):
    processed = np.asarray(pca_result["processed"], dtype=float)
    scores = np.asarray(pca_result["scores"], dtype=float)
    pca = pca_result["pca"]
    eigenvalues = np.maximum(np.asarray(pca.explained_variance_, dtype=float), 1e-12)

    t2 = np.sum((scores**2) / eigenvalues[None, :], axis=1)
    reconstructed = pca.inverse_transform(scores)
    residuals = processed - reconstructed
    q = np.sum(residuals**2, axis=1)

    n, a = scores.shape
    if n > a:
        t2_limit = float(
            (a * (n - 1) / (n - a))
            * f_distribution.ppf(alpha, a, n - a)
        )
    else:
        t2_limit = float(np.quantile(t2, alpha))
    # Empirical residual limit is intentionally transparent and robust for guided use.
    q_limit = float(np.quantile(q, alpha))

    table = pd.DataFrame(
        {
            "sample_index": np.arange(n),
            "Hotelling_T2": t2,
            "Q_residual": q,
            "T2_flag": t2 > t2_limit,
            "Q_flag": q > q_limit,
        }
    )
    table["review_flag"] = table.T2_flag | table.Q_flag
    return {
        "table": table,
        "t2_limit": t2_limit,
        "q_limit": q_limit,
        "residuals": residuals,
        "reconstructed": reconstructed,
        "alpha": alpha,
    }


def sample_contributions(pca_result, sample_index):
    sample_index = int(sample_index)
    processed = np.asarray(pca_result["processed"])
    scores = np.asarray(pca_result["scores"])
    pca = pca_result["pca"]
    if sample_index < 0 or sample_index >= len(processed):
        raise IndexError("Sample index is out of range.")

    reconstructed = pca.inverse_transform(scores[[sample_index]])[0]
    residual = processed[sample_index] - reconstructed
    q_contribution = residual**2

    eigenvalues = np.maximum(np.asarray(pca.explained_variance_), 1e-12)
    standardized_scores = scores[sample_index] / np.sqrt(eigenvalues)
    signed_t2 = pca.components_.T @ standardized_scores
    t2_related = signed_t2**2

    return pd.DataFrame(
        {
            "wavenumber": np.asarray(pca_result["wn"]),
            "Q_contribution": q_contribution,
            "T2_related_contribution": t2_related,
            "T2_signed_direction": signed_t2,
        }
    )


def _silhouette_quality(value: float) -> str:
    if not np.isfinite(value):
        return "not interpretable"
    if value >= 0.60:
        return "very strong"
    if value >= 0.45:
        return "strong"
    if value >= 0.25:
        return "moderate"
    return "weak"


def analyze_clustering(z, max_k=8):
    """Benchmark clustering choices and return a plain-English interpretation."""
    z = np.asarray(z, dtype=float)
    n = len(z)
    if n < 4:
        raise ValueError("At least four samples are required for clustering analysis.")

    kmax = max(2, min(int(max_k), n - 1, max(3, int(np.sqrt(n)) + 2)))
    rows = []
    labels_by_key = {}

    for k in range(2, kmax + 1):
        km = KMeans(k, n_init=40, random_state=42).fit_predict(z)
        ag = AgglomerativeClustering(k, linkage="ward").fit_predict(z)
        km_sil = float(silhouette_score(z, km))
        ag_sil = float(silhouette_score(z, ag))
        agreement = float(adjusted_rand_score(km, ag))
        rows.extend(
            [
                {"method": "KMeans", "k": k, "silhouette": km_sil, "coverage": 1.0, "agreement": agreement},
                {"method": "Agglomerative", "k": k, "silhouette": ag_sil, "coverage": 1.0, "agreement": agreement},
            ]
        )
        labels_by_key[("KMeans", k)] = km
        labels_by_key[("Agglomerative", k)] = ag

    nn = min(max(3, int(round(math.sqrt(n) / 2))), n - 1)
    distances, _ = NearestNeighbors(n_neighbors=nn).fit(z).kneighbors(z)
    kth = distances[:, -1]
    for quantile in (0.70, 0.80, 0.85, 0.90, 0.95):
        eps = float(np.quantile(kth, quantile))
        labels = DBSCAN(eps=max(eps, 1e-12), min_samples=nn).fit_predict(z)
        non_noise = labels != -1
        cluster_ids = set(labels[non_noise])
        coverage = float(non_noise.mean())
        if len(cluster_ids) >= 2 and non_noise.sum() > len(cluster_ids):
            sil = float(silhouette_score(z[non_noise], labels[non_noise]))
            score = sil * coverage
            rows.append(
                {
                    "method": "DBSCAN",
                    "k": len(cluster_ids),
                    "silhouette": sil,
                    "coverage": coverage,
                    "agreement": np.nan,
                    "eps": eps,
                    "min_samples": nn,
                    "selection_score": score,
                }
            )
            labels_by_key[("DBSCAN", quantile)] = labels

    table = pd.DataFrame(rows)
    if "selection_score" not in table:
        table["selection_score"] = np.nan
    standard = table.method != "DBSCAN"
    table.loc[standard, "selection_score"] = (
        table.loc[standard, "silhouette"] - 0.01 * (table.loc[standard, "k"] - 2)
    )

    best_row = table.loc[table.selection_score.idxmax()]
    method = str(best_row.method)
    if method == "DBSCAN":
        # Find the nearest matching DBSCAN labels.
        target_eps = float(best_row.eps)
        best_key = min(
            [key for key in labels_by_key if key[0] == "DBSCAN"],
            key=lambda key: abs(float(np.quantile(kth, key[1])) - target_eps),
        )
    else:
        best_key = (method, int(best_row.k))
    best_labels = labels_by_key[best_key]

    counts = pd.Series(best_labels[best_labels >= 0]).value_counts().sort_index()
    noise = int((best_labels == -1).sum())
    best_silhouette = float(best_row.silhouette)
    quality = _silhouette_quality(best_silhouette)

    # Agreement is most meaningful between KMeans and Ward at the selected k.
    k_for_agreement = int(best_row.k)
    if ("KMeans", k_for_agreement) in labels_by_key and ("Agglomerative", k_for_agreement) in labels_by_key:
        agreement = float(
            adjusted_rand_score(
                labels_by_key[("KMeans", k_for_agreement)],
                labels_by_key[("Agglomerative", k_for_agreement)],
            )
        )
    else:
        agreement = np.nan

    if np.isfinite(agreement):
        agreement_phrase = (
            "The two conventional clustering methods agree strongly."
            if agreement >= 0.75
            else "The two conventional clustering methods show partial agreement."
            if agreement >= 0.4
            else "Different clustering methods disagree, so the cluster structure should be treated cautiously."
        )
    else:
        agreement_phrase = "Agreement between conventional methods is not available for this choice."

    cluster_sizes = ", ".join(f"{int(idx)}: {int(value)}" for idx, value in counts.items())
    report = (
        f"Recommended clustering: {method}"
        + (f" with {int(best_row.k)} clusters" if method != "DBSCAN" else "")
        + f". The separation is {quality} (silhouette {best_silhouette:.3f}). "
        f"{agreement_phrase} "
        f"Cluster sizes are {cluster_sizes or 'not available'}."
    )
    if noise:
        report += f" DBSCAN-style noise detection marks {noise} sample(s) as outside dense groups."
    if quality == "weak":
        report += " The data may form a continuum rather than cleanly separated groups; do not over-interpret cluster names."

    return {
        "table": table.sort_values("selection_score", ascending=False).reset_index(drop=True),
        "best_method": method,
        "best_labels": np.asarray(best_labels),
        "best_params": best_row.to_dict(),
        "report": report,
        "linkage": linkage(z, method="ward"),
        "agreement_ari": agreement,
    }


def guided_analysis(X, wn, meta, max_components=20):
    """Run the non-expert workflow and return choices plus human-readable reasoning."""
    schema = infer_metadata_schema(meta)
    label = suggest_label_column(meta, schema)
    groups = suggest_group_columns(meta, schema)

    comparisons, comparison_table = compare_pca_recipes(
        X, wn, meta=meta, max_components=max_components
    )
    selected = comparisons[0]
    pca_result = selected["pca_result"]
    diagnostics = pca_diagnostics(pca_result)

    cluster_pcs = max(2, min(selected["best_components"], pca_result["scores"].shape[1], 10))
    clustering = analyze_clustering(pca_result["scores"][:, :cluster_pcs])

    flagged = int(diagnostics["table"].review_flag.sum())
    total = len(meta)
    q2 = selected["Q2_X"]
    label_text = f"'{label}'" if label else "no reliable prediction label"
    group_text = ", ".join(groups) if groups else "no independent grouping field was confidently detected"

    report = (
        f"Guided analysis selected {selected['name']} preprocessing and "
        f"{selected['best_components']} PCA component(s). "
        f"The cross-validated reconstruction Q²-X is {q2:.3f}. "
        f"{flagged} of {total} sample(s) are flagged for review by the PCA diagnostics. "
        f"{clustering['report']} "
        f"For predictive modeling, the best automatic label suggestion is {label_text}; "
        f"grouping suggestion: {group_text}. "
        "These are recommendations, not proof of sample identity or class membership."
    )

    return {
        "schema": schema,
        "suggested_label": label,
        "suggested_groups": groups,
        "comparisons": comparisons,
        "comparison_table": comparison_table,
        "selected": selected,
        "pca_result": pca_result,
        "diagnostics": diagnostics,
        "clustering": clustering,
        "report": report,
        "cluster_pcs": cluster_pcs,
    }


def _ordered_steps(steps):
    stages = {
        "Range": 0,
        "Exclude": 1,
        "Baseline polynomial": 2,
        "Savitzky-Golay": 3,
        "Derivative": 4,
        "SNV": 5,
        "Vector normalize": 5,
        "Area normalize": 5,
        "Mean center": 6,
        "Autoscale": 6,
        "Robust scale": 6,
    }
    return sorted(copy.deepcopy(steps), key=lambda step: stages.get(step.name, 99))


def candidate(rng, maxpcs, base):
    """Generate only chemically sensible candidates; do not randomly scramble preprocessing order."""
    steps = _ordered_steps(base)
    for step in steps:
        if step.name in ("Savitzky-Golay", "Derivative"):
            step.params["window"] = [7, 11, 15, 21][int(rng.integers(4))]
        if step.name == "Derivative":
            step.params["order"] = int(rng.integers(1, 3))

    model = ["SVM", "Random Forest"][int(rng.integers(2))]
    params = {
        "model": model,
        "use_pca": bool(rng.integers(2)),
        "pcs": int(rng.integers(2, max(3, maxpcs + 1))),
    }
    if model == "SVM":
        gamma_options = ["scale", 0.001, 0.01, 0.1]
        params.update(
            C=float(10 ** rng.uniform(-2, 2)),
            kernel=["linear", "rbf"][int(rng.integers(2))],
            gamma=gamma_options[int(rng.integers(len(gamma_options)))],
        )
    else:
        params.update(
            trees=[200, 400][int(rng.integers(2))],
            depth=[None, 5, 10, 20][int(rng.integers(4))],
            leaf=int(rng.integers(1, 5)),
        )
    return steps, params


def pipe(steps, params, max_pc_fit=None, wn=None):
    chain = [("prep", RecipeTransformer(steps, wn=wn))]
    if params["model"] == "SVM":
        chain.append(("scale", StandardScaler()))
    if params["use_pca"]:
        chain.append(
            ("pca", PCA(params["pcs"], svd_solver="randomized", random_state=42))
        )
    if params["model"] == "SVM":
        classifier = SVC(
            C=params["C"],
            kernel=params["kernel"],
            gamma=params["gamma"],
            class_weight="balanced",
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=params["trees"],
            max_depth=params["depth"],
            min_samples_leaf=params["leaf"],
            class_weight="balanced",
            n_jobs=1,
            random_state=42,
        )
    chain.append(("clf", classifier))
    return Pipeline(chain)


def search(X, y, groups, base, trials, maxpcs, cv, seed, wn=None):
    rng = np.random.default_rng(seed)
    history = []
    best = None

    for trial in range(trials):
        steps, params = candidate(rng, maxpcs, base)
        try:
            values = cross_val_score(
                pipe(steps, params, wn=wn),
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
            objective = score - 0.25 * std - 0.002 * (params["pcs"] if params["use_pca"] else 0)
            history.append(
                {
                    "trial": trial,
                    "status": "complete",
                    "score": score,
                    "std": std,
                    "objective": objective,
                    "recipe": json.dumps([asdict(step) for step in steps]),
                    **params,
                }
            )
            if best is None or objective > best[0]:
                best = (objective, steps, params)
        except Exception as exc:
            history.append(
                {
                    "trial": trial,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "recipe": json.dumps([asdict(step) for step in steps]),
                    **params,
                }
            )

    if best is None:
        raise RuntimeError("All optimization trials failed.")
    return best[1], best[2], pd.DataFrame(history)


def nested_optimize(
    X,
    y,
    groups,
    base,
    trials=20,
    maxpcs=20,
    outer_folds=3,
    inner_folds=2,
    wn=None,
):
    outer = StratifiedGroupKFold(outer_folds, shuffle=True, random_state=42)
    predictions = np.empty(len(y), dtype=object)
    folds = []
    history = []
    best_rows = []

    for fold, (train, test) in enumerate(outer.split(X, y, groups), 1):
        inner = StratifiedGroupKFold(inner_folds, shuffle=True, random_state=100 + fold)
        steps, params, fold_history = search(
            X[train],
            y[train],
            groups[train],
            base,
            trials,
            min(maxpcs, len(train) - 1),
            inner,
            200 + fold,
            wn=wn,
        )
        model = pipe(steps, params, wn=wn).fit(X[train], y[train])
        pred = model.predict(X[test])
        predictions[test] = pred
        fold_history["outer_fold"] = fold
        history.append(fold_history)
        best_rows.append(
            {"fold": fold, "recipe": json.dumps([asdict(step) for step in steps]), **params}
        )
        folds.append(
            {
                "fold": fold,
                "balanced_accuracy": balanced_accuracy_score(y[test], pred),
                "macro_f1": f1_score(y[test], pred, average="macro", zero_division=0),
            }
        )

    classes = np.unique(y)
    return {
        "pred": predictions,
        "folds": pd.DataFrame(folds),
        "history": pd.concat(history, ignore_index=True),
        "best": pd.DataFrame(best_rows),
        "classes": classes,
        "cm": confusion_matrix(y, predictions, labels=classes),
        "report": pd.DataFrame(
            classification_report(y, predictions, output_dict=True, zero_division=0)
        ).T,
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro", zero_division=0),
    }
