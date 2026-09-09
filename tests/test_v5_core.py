import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from core import (
    RecipeTransformer,
    Step,
    analyze_clustering,
    apply_recipe,
    eligibility,
    guided_analysis,
    infer_metadata_schema,
    make_groups,
    pca_cv,
    pca_diagnostics,
    search,
)


def sample_data():
    rng = np.random.default_rng(1)
    wn = np.linspace(400, 1800, 80)
    meta = []
    X = []
    for brand, shift in [("A", 0.0), ("B", 0.7)]:
        for site in range(4):
            for rep in range(3):
                meta.append(
                    {
                        "specimen_id": f"{brand}-{site}-{rep}",
                        "manufacturer_name": brand,
                        "station_code": f"S{site}",
                        "City": f"C{site}",
                        "State": "CA",
                        "collection_year": 2025,
                    }
                )
                X.append(np.sin(wn / 90) + shift + rng.normal(0, 0.03, len(wn)))
    return pd.DataFrame(meta), wn, np.asarray(X)


def test_preprocessing_order_changes_result():
    _, wn, X = sample_data()
    a = [Step("Derivative", {"order": 1, "window": 11, "poly": 2}), Step("SNV", {})]
    b = list(reversed(a))
    assert not np.allclose(apply_recipe(X, wn, a)[0], apply_recipe(X, wn, b)[0])


def test_pca_cv_returns_finite_curve():
    meta, wn, X = sample_data()
    groups = make_groups(meta, ["City", "State"])
    result, best = pca_cv(X, [Step("SNV", {})], 6, 2, groups, wn=wn)
    assert len(result) == 6
    assert 1 <= best <= 6
    assert np.isfinite(result.RMSECV_X).all()
    assert np.isfinite(result.Q2_X).all()


def test_eligibility_counts_independent_groups():
    meta, _, _ = sample_data()
    result, _, _ = eligibility(meta, "manufacturer_name", ["City", "State"], 3)
    assert result.eligible.all()


def test_search_completes_and_gamma_type_is_valid():
    meta, wn, X = sample_data()
    y = meta.manufacturer_name.to_numpy()
    groups = make_groups(meta, ["City", "State"])
    cv = StratifiedGroupKFold(2, shuffle=True, random_state=1)
    _, params, history = search(
        X, y, groups, [Step("SNV", {})], 4, 5, cv, 1, wn=wn
    )
    assert (history.status == "complete").any()
    assert params.get("gamma", "scale") == "scale" or isinstance(params.get("gamma"), float)


def test_recipe_transformer_uses_real_wavenumber_axis():
    _, wn, X = sample_data()
    steps = [Step("Range", {"minimum": 700, "maximum": 1200}), Step("SNV", {})]
    expected = ((wn >= 700) & (wn <= 1200)).sum()
    transformed = RecipeTransformer(steps, wn=wn).fit_transform(X)
    assert transformed.shape == (len(X), expected)


def test_metadata_schema_uses_headers_and_values():
    meta, _, _ = sample_data()
    schema = infer_metadata_schema(meta).set_index("column")
    assert schema.loc["specimen_id", "role"] == "sample_id"
    assert schema.loc["manufacturer_name", "role"] == "brand_or_manufacturer"
    assert schema.loc["station_code", "role"] == "site_or_station"
    assert schema.loc["collection_year", "role"] == "year"


def test_pca_diagnostics_are_sample_aligned():
    _, wn, X = sample_data()
    from core import exploratory_pca

    result = exploratory_pca(X, wn, [Step("SNV", {}), Step("Mean center", {})], 5)
    diagnostics = pca_diagnostics(result)
    assert len(diagnostics["table"]) == len(X)
    assert np.isfinite(diagnostics["table"].Hotelling_T2).all()
    assert np.isfinite(diagnostics["table"].Q_residual).all()
    assert diagnostics["t2_limit"] > 0
    assert diagnostics["q_limit"] >= 0


def test_clustering_analysis_returns_recommendation():
    _, wn, X = sample_data()
    from core import exploratory_pca

    result = exploratory_pca(X, wn, [Step("Mean center", {})], 5)
    clustering = analyze_clustering(result["scores"][:, :3], max_k=5)
    assert clustering["best_method"] in {"KMeans", "Agglomerative", "DBSCAN"}
    assert len(clustering["best_labels"]) == len(X)
    assert "Recommended clustering" in clustering["report"]
    assert not clustering["table"].empty


def test_guided_analysis_selects_metadata_and_pca():
    meta, wn, X = sample_data()
    result = guided_analysis(X, wn, meta, max_components=6)
    assert result["suggested_label"] == "manufacturer_name"
    assert "station_code" in result["suggested_groups"]
    assert result["selected"]["best_components"] >= 1
    assert len(result["comparisons"]) >= 4
    assert len(result["diagnostics"]["table"]) == len(X)
