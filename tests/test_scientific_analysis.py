import numpy as np
import pandas as pd

import core
from scientific_analysis import (
    confound_analysis,
    loading_regions,
    metadata_associations,
    spectral_qc,
)
from validation_audit import nested_optimize_audited


def synthetic_spectra():
    rng = np.random.default_rng(7)
    wn = np.linspace(650, 1800, 90)
    X = []
    meta = []
    for label, shift, city in [("A", 0.0, "Phoenix"), ("B", 0.25, "Tucson")]:
        for site in range(4):
            for rep in range(3):
                signal = np.sin(wn / 120) + 0.25 * np.cos(wn / 55)
                signal += shift * np.exp(-((wn - 1450) / 80) ** 2)
                X.append(signal + rng.normal(0, 0.025, len(wn)))
                meta.append({
                    "Brand": label,
                    "City": city,
                    "Site": f"{label}-S{site}",
                    "Replicate": rep + 1,
                })
    return pd.DataFrame(meta), wn, np.asarray(X)


def test_qc_flags_flat_spectrum_without_deleting_it():
    meta, wn, X = synthetic_spectra()
    X[0] = 3.0
    result = spectral_qc(X, wn, meta)
    assert len(result["table"]) == len(X)
    assert bool(result["table"].loc[0, "flat_flag"])
    assert bool(result["table"].loc[0, "qc_review"])


def test_confound_detection_finds_known_brand_city_confound():
    meta, _, _ = synthetic_spectra()
    table, warnings = confound_analysis(meta, label_column="Brand")
    hit = table[
        ((table.metadata_a == "Brand") & (table.metadata_b == "City"))
        | ((table.metadata_a == "City") & (table.metadata_b == "Brand"))
    ]
    assert not hit.empty
    assert float(hit.iloc[0].cramers_v) > 0.95
    assert any("Brand" in warning and "City" in warning for warning in warnings)


def test_metadata_association_detects_brand_signal():
    meta, wn, X = synthetic_spectra()
    result = core.exploratory_pca(
        X, wn, [core.Step("Mean center", {})], components=5
    )
    assoc = metadata_associations(result["scores"], meta, permutations=9)
    brand = assoc.loc[assoc.metadata == "Brand"]
    assert not brand.empty
    assert float(brand.iloc[0].pca_association) > 0.05


def test_loading_regions_are_grouped_and_on_physical_axis():
    _, wn, X = synthetic_spectra()
    result = core.exploratory_pca(
        X, wn, [core.Step("Mean center", {})], components=4
    )
    regions = loading_regions(result, pcs=2, quantile=0.90)
    assert not regions.empty
    assert regions["peak_cm-1"].between(wn.min(), wn.max()).all()
    assert (regions["region_max_cm-1"] >= regions["region_min_cm-1"]).all()


def test_audited_nested_cv_has_no_group_leakage_and_oof_alignment():
    meta, wn, X = synthetic_spectra()
    y = meta.Brand.to_numpy()
    groups = meta.Site.to_numpy()
    result = nested_optimize_audited(
        X,
        y,
        groups,
        [core.Step("Mean center", {})],
        trials=3,
        maxpcs=4,
        outer_folds=2,
        inner_folds=2,
        wn=wn,
        seed=11,
    )
    assert len(result["oof_predictions"]) == len(X)
    assert sorted(result["oof_predictions"].sample_index.tolist()) == list(range(len(X)))
    assert (result["folds"].group_overlap_count == 0).all()
    assert (result["outer_assignments"] > 0).all()
    for fold in result["splits"].outer_fold.unique():
        block = result["splits"].loc[result["splits"].outer_fold == fold]
        train_groups = set(block.loc[block.role == "train", "group"])
        test_groups = set(block.loc[block.role == "test", "group"])
        assert not train_groups.intersection(test_groups)
    assert np.isfinite(result["train_validation_gap"])
    assert result["overfitting_level"] in {"low", "moderate", "high"}
