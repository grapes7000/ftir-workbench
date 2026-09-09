from __future__ import annotations

"""Model-review helpers and v5.2 science-workflow integration.

The trial slider is rebuilt from each history row's stored recipe/PCA settings.
Scientific helpers remain importable in headless test environments; the desktop
launcher explicitly installs the optional PySide6 Model Health workspace only
after the GUI module is available.
"""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

import core
import predictive_health
import scientific_analysis
import transparent_guided
import validation_audit
from busy_ui import busy


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


# --- Analysis-engine integration -----------------------------------------
_original_guided_analysis = transparent_guided.guided_analysis


def guided_analysis_with_scientific_audit(X, wn, meta, max_components=20):
    result = _original_guided_analysis(X, wn, meta, max_components=max_components)
    science = scientific_analysis.comprehensive_analysis(
        X, wn, meta, result, n_resamples=8
    )
    result["scientific"] = science
    result["report"] = (
        science["report"]
        + "\n\nDETAILED GUIDED-SELECTION AUDIT\n"
        + result["report"]
    )
    return result


def nested_optimize_compatible(*args, **kwargs):
    result = validation_audit.nested_optimize_audited(*args, **kwargs)
    # Existing plots expect these legacy names. They are aliases of OUTER held-out
    # metrics, not training metrics.
    result["folds"] = result["folds"].copy()
    result["folds"]["balanced_accuracy"] = result["folds"]["outer_balanced_accuracy"]
    result["folds"]["macro_f1"] = result["folds"]["outer_macro_f1"]

    # Surface outer-fold context in each optimization-history row so the slider
    # never looks like an isolated training/inner-CV score.
    by_fold = result["folds"].set_index("fold")
    history = result["history"].copy()
    history["outer_balanced_accuracy"] = history.outer_fold.map(by_fold.outer_balanced_accuracy)
    history["outer_macro_f1"] = history.outer_fold.map(by_fold.outer_macro_f1)
    history["outer_train_validation_gap"] = history.outer_fold.map(by_fold.train_outer_gap)
    result["history"] = history
    return result


transparent_guided.guided_analysis = guided_analysis_with_scientific_audit
core.guided_analysis = guided_analysis_with_scientific_audit
core.nested_optimize = nested_optimize_compatible


# --- GUI extension --------------------------------------------------------
def _results_dir(window):
    text = window.path.text().strip() if hasattr(window, "path") else ""
    if not text:
        return None
    path = Path(text)
    return path.with_name(path.stem + "_v5_2_results")


def _write_guided_exports(window):
    if not getattr(window, "guided", None):
        return
    result = window.guided
    out = _results_dir(window)
    if out is None:
        return
    out.mkdir(parents=True, exist_ok=True)

    science = result.get("scientific")
    if science:
        scientific_analysis.export_analysis(science, out)

    if "comparison_table" in result:
        result["comparison_table"].to_csv(out / "preprocessing_comparison.csv", index=False)
    selected = result.get("selected") or {}
    if isinstance(selected.get("cv"), pd.DataFrame):
        selected["cv"].to_csv(out / "pca_cross_validation.csv", index=False)

    pca = result.get("pca_result")
    if pca:
        scores = pd.DataFrame(
            pca["scores"],
            columns=[f"PC{i+1}" for i in range(pca["scores"].shape[1])],
        )
        meta = window.data[0].reset_index(drop=True) if getattr(window, "data", None) is not None else pd.DataFrame()
        pd.concat([meta, scores], axis=1).to_csv(out / "pca_scores.csv", index=False)
        loadings = pd.DataFrame(
            pca["loadings"],
            columns=[f"PC{i+1}_loading" for i in range(pca["loadings"].shape[1])],
        )
        loadings.insert(0, "wavenumber_cm-1", pca["wn"])
        loadings.to_csv(out / "pca_loadings.csv", index=False)

    diagnostics = result.get("diagnostics", {}).get("table")
    if isinstance(diagnostics, pd.DataFrame):
        meta = window.data[0].reset_index(drop=True) if getattr(window, "data", None) is not None else pd.DataFrame()
        pd.concat([meta, diagnostics.reset_index(drop=True)], axis=1).to_csv(
            out / "pca_outlier_diagnostics.csv", index=False
        )

    clustering = result.get("clustering", {})
    if isinstance(clustering.get("table"), pd.DataFrame):
        clustering["table"].to_csv(out / "clustering_metrics.csv", index=False)
    if "best_labels" in clustering:
        labels = pd.DataFrame({
            "sample_index": np.arange(len(clustering["best_labels"])),
            "cluster": clustering["best_labels"],
        })
        labels.to_csv(out / "cluster_assignments.csv", index=False)

    manifest = {
        "selected_preprocessing": selected.get("name"),
        "selected_pca_components": selected.get("best_components"),
        "selected_recipe": [asdict(step) for step in selected.get("steps", [])],
        "hard_exclusion": result.get("hard_exclusion", {}).get("normalized", ""),
        "suggested_label": result.get("suggested_label"),
        "suggested_groups": result.get("suggested_groups", []),
        "analysis_claim": "Exploratory PCA/clustering results are not predictive validation.",
    }
    (out / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    (out / "analysis_summary.txt").write_text(
        result.get("report", ""), encoding="utf-8"
    )


def _write_predictive_exports(window):
    if not getattr(window, "opt", None):
        return
    result = window.opt[0]
    out = _results_dir(window)
    if out is None:
        return
    out.mkdir(parents=True, exist_ok=True)
    for key, name in (
        ("oof_predictions", "out_of_fold_predictions.csv"),
        ("splits", "validation_split_audit.csv"),
        ("specificity", "per_class_sensitivity_specificity.csv"),
        ("best", "outer_fold_selected_models.csv"),
        ("model_family_folds", "model_family_outer_fold_metrics.csv"),
        ("model_family_summary", "model_family_outer_comparison.csv"),
        ("model_family_oof_predictions", "model_family_out_of_fold_predictions.csv"),
    ):
        value = result.get(key)
        if isinstance(value, pd.DataFrame):
            value.to_csv(out / name, index=False)
    if "validation_summary" in result:
        (out / "predictive_validation_summary.txt").write_text(
            result["validation_summary"], encoding="utf-8"
        )
    permutation = result.get("permutation")
    if permutation:
        pd.DataFrame({"permuted_balanced_accuracy": permutation.get("scores", [])}).to_csv(
            out / "permutation_test_scores.csv", index=False
        )
        (out / "permutation_test_summary.txt").write_text(
            permutation.get("summary", ""), encoding="utf-8"
        )


def _populate_health_workspace(window):
    if not hasattr(window, "health_report"):
        return
    guided = getattr(window, "guided", None) or {}
    science = guided.get("scientific")
    if not science:
        window.health_report.setPlainText(
            "Run Guided Analysis to calculate model health, stability, metadata associations, and confounding."
        )
        return

    health = science["health"]
    predictive_text = ""
    if getattr(window, "opt", None):
        predictive_result = window.opt[0]
        combined = predictive_health.combine_health(health, predictive_result)
        display_health = combined
        predictive_text = (
            "\n\nPREDICTIVE VALIDATION\n"
            + predictive_result.get("validation_summary", "")
        )
        permutation = predictive_result.get("permutation")
        if permutation:
            predictive_text += "\n" + permutation.get("summary", "")
    else:
        display_health = health

    header = (
        f"{display_health['band']}"
        + (f" ({display_health['overall']:.0f}/100 transparent summary score)" if np.isfinite(display_health["overall"]) else "")
        + "\n"
        + display_health["disclaimer"]
    )
    window.health_report.setPlainText(
        header + "\n\n" + science["report"] + predictive_text
    )

    window._table(window.health_subscores, display_health["table"])
    qc = science["qc"]["table"].copy()
    flagged = qc.loc[qc.qc_review].copy()
    window._table(window.health_qc, flagged if not flagged.empty else qc.head(0))
    window._table(window.health_loadings, science["loading_regions"])
    window._table(window.health_metadata, science["metadata_associations"])
    window._table(window.health_confounds, science["confounds"])
    stability = pd.DataFrame([
        {
            "analysis": "Preprocessing",
            "classification": science["preprocessing_stability"]["classification"],
            "score": science["preprocessing_stability"]["score"],
            "summary": science["preprocessing_stability"]["summary"],
        },
        {
            "analysis": "PCA loadings/resampling",
            "classification": science["pca_stability"]["classification"],
            "score": science["pca_stability"]["score"],
            "summary": science["pca_stability"]["summary"],
        },
        {
            "analysis": "Clustering",
            "classification": science["cluster_stability"]["classification"],
            "score": science["cluster_stability"]["score"],
            "summary": science["cluster_stability"]["summary"],
        },
    ])
    window._table(window.health_stability, stability)


@busy("Running permutation test")
def _run_permutation_test(window):
    if not getattr(window, "opt", None):
        try:
            import app_ui
            app_ui.QMessageBox.information(
                window,
                "Permutation test",
                "Run predictive nested optimization first. The null test reuses the same grouped validation strategy.",
            )
        except Exception:
            pass
        return
    try:
        import app_ui
        result, X, y, groups = window.opt
        window.permutation_button.setEnabled(False)
        window.statusBar().showMessage(
            "Running 10 shuffled-label nested-CV null models. This is intentionally optional and may be slow."
        )
        app_ui.QApplication.processEvents()
        permutation = validation_audit.permutation_test_nested(
            X,
            y,
            groups,
            window.steps,
            observed_score=float(result["balanced_accuracy"]),
            n_permutations=10,
            trials=min(8, int(window.trials.value())),
            maxpcs=min(15, int(window.maxpc.value())),
            outer_folds=max(2, len(result["folds"])),
            inner_folds=int(window.inner.value()),
            wn=getattr(window, "opt_wn", None),
            seed=123,
            validation_mode=result.get("validation_mode", "grouped"),
        )
        result["permutation"] = permutation
        _populate_health_workspace(window)
        _write_predictive_exports(window)
        window.statusBar().showMessage(permutation.get("summary", "Permutation test complete."))
    except Exception:
        window.err()
    finally:
        if hasattr(window, "permutation_button"):
            window.permutation_button.setEnabled(True)


def install_gui_extensions(app_ui):
    """Install the Model Health workspace after the Qt UI module is already loaded."""
    if getattr(app_ui.Main, "_science_extensions_installed", False):
        return

    original_init = app_ui.Main.__init__
    original_load = app_ui.Main.load
    original_render_guided = app_ui.Main.render_guided
    original_render_review = app_ui.Main.render_review

    def enhanced_init(self):
        original_init(self)
        page = app_ui.QWidget()
        layout = app_ui.QVBoxLayout(page)
        intro = app_ui.QLabel(
            "<b>Model Health</b> combines transparent evidence from QC, PCA cross-validation, "
            "preprocessing/resampling stability, outliers, clustering, metadata confounding, "
            "independent-group support, and—after predictive modeling—held-out generalization. "
            "The 0–100 number is not an established statistical index; inspect the subscores."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.health_report = app_ui.QPlainTextEdit()
        self.health_report.setReadOnly(True)
        self.health_report.setMaximumHeight(280)
        self.health_report.setPlainText(
            "Load data and run Guided Analysis. Automatic QC will appear here after loading; full health assessment follows Guided Analysis."
        )
        layout.addWidget(self.health_report)

        self.health_tabs = app_ui.QTabWidget()
        self.health_subscores = app_ui.QTableWidget()
        self.health_qc = app_ui.QTableWidget()
        self.health_loadings = app_ui.QTableWidget()
        self.health_metadata = app_ui.QTableWidget()
        self.health_confounds = app_ui.QTableWidget()
        self.health_stability = app_ui.QTableWidget()
        for widget, title in (
            (self.health_subscores, "Health subscores"),
            (self.health_qc, "QC samples"),
            (self.health_loadings, "Loading regions"),
            (self.health_metadata, "Metadata associations"),
            (self.health_confounds, "Confounds"),
            (self.health_stability, "Stability"),
        ):
            self.health_tabs.addTab(widget, title)
        layout.addWidget(self.health_tabs, 1)

        row = app_ui.QHBoxLayout()
        self.permutation_button = app_ui.QPushButton("Run 10-label permutation null test (optional / slow)")
        self.permutation_button.clicked.connect(lambda: _run_permutation_test(self))
        row.addWidget(self.permutation_button)
        export_note = app_ui.QLabel(
            "Guided and predictive audit tables are automatically written beside the dataset in the *_v5_2_results folder."
        )
        export_note.setWordWrap(True)
        row.addWidget(export_note, 1)
        layout.addLayout(row)
        self.tabs.addTab(page, "Model Health")

    @busy("Loading data and checking quality")
    def enhanced_load(self):
        original_load(self)
        if getattr(self, "data", None) is None or not hasattr(self, "health_report"):
            return
        try:
            meta, wn, spectra = self.data
            if hasattr(self, "_constrained"):
                X, used_wn, info = self._constrained(spectra.to_numpy(), wn)
                exclusion = info.get("normalized", "") or "none"
            else:
                X, used_wn = spectra.to_numpy(), wn
                exclusion = "not yet applied"
            qc = scientific_analysis.spectral_qc(X, used_wn, meta)
            self.health_report.setPlainText(
                "AUTOMATIC POST-LOAD QC\n"
                f"Hard exclusion: {exclusion}\n"
                f"Retained spectral variables: {len(used_wn)}\n"
                + qc["summary"]
                + "\n\nRun Guided Analysis for PCA validation, stability, metadata/confound analysis, and the full model-health assessment."
            )
            flagged = qc["table"].loc[qc["table"].qc_review]
            self._table(self.health_qc, flagged if not flagged.empty else qc["table"].head(0))
        except Exception:
            # Loading the dataset should not fail merely because the advisory QC screen cannot render.
            pass

    def enhanced_render_guided(self):
        original_render_guided(self)
        _populate_health_workspace(self)
        try:
            _write_guided_exports(self)
        except Exception:
            # Analysis stays usable if an export path is read-only.
            self.statusBar().showMessage(
                "Analysis completed, but one or more automatic result exports could not be written."
            )

    def enhanced_render_review(self):
        original_render_review(self)

        # Replace the old inner-trial-only boxplot with a fair model-family comparison
        # whenever audited outer-fold family metrics are available. Each family is
        # tuned inside the same outer-training data and evaluated on identical held-out folds.
        if getattr(self, "opt", None):
            result = self.opt[0]
            family = result.get("model_family_folds")
            if isinstance(family, pd.DataFrame) and not family.empty:
                self.model_compare.fig.clear()
                ax = self.model_compare.fig.add_subplot()
                family.boxplot(
                    column="outer_balanced_accuracy",
                    by="model",
                    ax=ax,
                )
                self.model_compare.fig.suptitle("")
                ax.set_title("Model families · same outer held-out folds")
                ax.set_xlabel("Model family")
                ax.set_ylabel("Outer held-out balanced accuracy")
                self.model_compare.draw()

        _populate_health_workspace(self)
        try:
            _write_predictive_exports(self)
        except Exception:
            self.statusBar().showMessage(
                "Predictive review completed, but one or more audit exports could not be written."
            )

    app_ui.Main.__init__ = enhanced_init
    app_ui.Main.load = enhanced_load
    app_ui.Main.render_guided = enhanced_render_guided
    app_ui.Main.render_review = enhanced_render_review
    app_ui.Main._science_extensions_installed = True
