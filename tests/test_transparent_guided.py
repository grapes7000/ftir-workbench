import numpy as np
import pandas as pd

from transparent_guided import (
    candidate_recipes,
    compare_pca_recipes,
    guided_analysis,
    recipe_uses_snv,
)


def amplitude_separated_data():
    """Two known groups separated mainly by multiplicative spectral intensity.

    SNV should largely remove this real between-group signal, so the transparent
    selector must be able to prefer a non-SNV recipe instead of assuming SNV helps.
    """
    rng = np.random.default_rng(7)
    wn = np.linspace(450, 1800, 160)
    base = (
        0.9 * np.exp(-((wn - 720) / 70) ** 2)
        + 0.6 * np.exp(-((wn - 1120) / 95) ** 2)
        + 0.35 * np.exp(-((wn - 1510) / 55) ** 2)
    )
    rows = []
    meta = []
    for brand, scale in (("A", 1.0), ("B", 1.45)):
        for site in range(4):
            for rep in range(4):
                rows.append(scale * base + rng.normal(0, 0.012, len(wn)))
                meta.append(
                    {
                        "Brand": brand,
                        "Station": f"S{site}",
                        "SampleID": f"{brand}-{site}-{rep}",
                    }
                )
    return pd.DataFrame(meta), wn, np.asarray(rows)


def test_candidate_set_always_contains_snv_and_non_snv_alternatives():
    _, wn, X = amplitude_separated_data()
    recipes = candidate_recipes(wn, X.shape[1])
    flags = [recipe_uses_snv(steps) for _, steps in recipes]
    assert any(flags)
    assert not all(flags)
    names = [name for name, _ in recipes]
    assert "Mean-centered raw" in names
    assert "1st derivative + mean center" in names
    assert "SNV + mean center" in names


def test_snv_is_not_assumed_best_when_it_erases_known_group_signal():
    meta, wn, X = amplitude_separated_data()
    comparisons, table = compare_pca_recipes(X, wn, meta=meta, max_components=6, folds=3)
    selected = comparisons[0]
    assert selected["name"] == table.iloc[0]["name"]
    assert not selected["uses_snv"]

    snv = table.loc[table.uses_snv.astype(bool), "group_separation"].max()
    non_snv = table.loc[~table.uses_snv.astype(bool), "group_separation"].max()
    assert np.isfinite(non_snv)
    assert non_snv > snv


def test_transparent_table_exposes_every_selection_metric():
    meta, wn, X = amplitude_separated_data()
    _, table = compare_pca_recipes(X, wn, meta=meta, max_components=5, folds=3)
    required = {
        "name",
        "components",
        "Q2_X",
        "RMSECV_X",
        "PC1_PC2_percent",
        "group_separation",
        "label_used_for_display_score",
        "uses_snv",
        "complexity",
        "recipe",
        "recommendation",
    }
    assert required.issubset(table.columns)
    assert (table.recommendation == "DEFAULT").sum() == 1


def test_guided_report_states_rules_and_label_is_post_pca_only():
    meta, wn, X = amplitude_separated_data()
    result = guided_analysis(X, wn, meta, max_components=5)
    text = result["report"]
    assert "NO HIDDEN MODEL" in text
    assert "SNV" in text
    assert "labels did not influence the PCA fit" in text
    assert result["decision"]["rules"]
    assert result["selected"]["name"] == result["decision"]["selected_name"]
