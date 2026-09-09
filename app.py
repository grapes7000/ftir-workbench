from __future__ import annotations

"""FTIR Workbench v5.2 launcher with transparent guided analysis and hard constraints."""

import copy
import sys

import numpy as np
import pandas as pd

import core
import transparent_guided
from review_tools import trial_projection
from spectral_constraints import apply_hard_exclusion, describe_exclusion

core.compare_pca_recipes = transparent_guided.compare_pca_recipes
core.guided_analysis = transparent_guided.guided_analysis

import app_ui


class Main(app_ui.Main):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FTIR Workbench v5.2 — Transparent Guided Analysis")
        self.steps = [
            core.Step("Range", dict(core.DEFAULTS["Range"])),
            core.Step("Mean center", {}),
        ]
        self.refresh()
        self.opt_wn = None
        self.opt_constraint = None
        self._build_hard_exclusion_dock()
        self.statusBar().showMessage(
            "Load data, confirm the hard excluded region, then run Guided Analysis."
        )

    def _build_hard_exclusion_dock(self):
        dock = app_ui.QDockWidget("Hard spectral exclusion", self)
        container = app_ui.QWidget()
        layout = app_ui.QVBoxLayout(container)
        note = app_ui.QLabel(
            "<b>Hard constraint:</b> these wavenumbers are removed before every "
            "automatic PCA candidate, diagnostic, clustering analysis, and predictive model. "
            "Automation cannot add them back. Use comma-separated ranges such as "
            "<code>500-1000, 2350-2450</code>. Leave blank to disable."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        row = app_ui.QHBoxLayout()
        row.addWidget(app_ui.QLabel("Excluded range(s)"))
        self.hard_exclude = app_ui.QLineEdit("500-1000")
        self.hard_exclude.setPlaceholderText("e.g. 500-1000, 2350-2450")
        self.hard_exclude.editingFinished.connect(self._hard_exclusion_changed)
        row.addWidget(self.hard_exclude, 1)

        clear_button = app_ui.QPushButton("Clear")
        clear_button.clicked.connect(self._clear_hard_exclusion)
        rerun_button = app_ui.QPushButton("Re-run Guided Analysis")
        rerun_button.clicked.connect(self.auto_analyze)
        row.addWidget(clear_button)
        row.addWidget(rerun_button)
        layout.addLayout(row)

        self.hard_exclude_status = app_ui.QLabel(
            "Default hard exclusion: 500–1000 cm⁻¹. Edit this if your solvent region is different."
        )
        self.hard_exclude_status.setWordWrap(True)
        layout.addWidget(self.hard_exclude_status)

        dock.setWidget(container)
        self.addDockWidget(app_ui.Qt.TopDockWidgetArea, dock)
        self.hard_exclude_dock = dock

    def _clear_hard_exclusion(self):
        self.hard_exclude.clear()
        self._hard_exclusion_changed()

    def _hard_exclusion_changed(self):
        try:
            if self.data is None:
                text = self.hard_exclude.text().strip()
                self.hard_exclude_status.setText(
                    f"Hard exclusion set to {text or 'none'}. It will be applied when data are loaded."
                )
                return
            _, wn, spectra = self.data
            _, _, info = apply_hard_exclusion(
                spectra.to_numpy(), wn, self.hard_exclude.text()
            )
            self.hard_exclude_status.setText(describe_exclusion(info))
            self.auto_status.setText("Hard exclusion changed — re-run analysis to refresh results")
            self.statusBar().showMessage(
                "Hard spectral exclusion changed. Existing graphs are stale until the analysis is re-run."
            )
        except Exception:
            self.err()

    def _constrained(self, X, wn):
        return apply_hard_exclusion(X, wn, self.hard_exclude.text())

    def _constraint_report_note(self, info):
        if not info["normalized"]:
            return (
                "\n\nHARD SPECTRAL EXCLUSION\n"
                "None. All loaded spectral variables were available to this analysis."
            )
        return (
            "\n\nHARD SPECTRAL EXCLUSION\n"
            f"{info['normalized']} cm⁻¹ was removed before any preprocessing/model selection. "
            f"{info['removed_variables']} spectral variables were removed and "
            f"{info['retained_variables']} were retained. Automatic methods cannot reintroduce "
            "the excluded wavelengths."
        )

    def auto_analyze(self):
        if self.data is None:
            app_ui.QMessageBox.information(self, "Guided Analysis", "Load a dataset first.")
            return
        try:
            self.auto_button.setEnabled(False)
            self.auto_status.setText(
                "Applying hard exclusion, then analyzing preprocessing, PCA, diagnostics, and clustering…"
            )
            app_ui.QApplication.processEvents()

            meta, wn, spectra = self.data
            X, constrained_wn, constraint = self._constrained(
                spectra.to_numpy(), wn
            )
            result = transparent_guided.guided_analysis(
                X,
                constrained_wn,
                meta,
                max_components=min(self.maxpc.value(), 20),
            )
            result["hard_exclusion"] = constraint
            result["comparison_table"] = result["comparison_table"].copy()
            result["comparison_table"]["hard_exclusion_cm-1"] = (
                constraint["normalized"] or "none"
            )
            result["report"] += self._constraint_report_note(constraint)

            self.guided = result
            self.last = result["pca_result"]
            self.embed = self.last["scores"]
            self.steps = copy.deepcopy(result["selected"]["steps"])
            self.refresh()
            self.npcs.setValue(max(2, result["selected"]["best_components"]))

            if result["suggested_label"]:
                self.label.setText(result["suggested_label"])
            if result["suggested_groups"]:
                self.groups.setText(",".join(result["suggested_groups"]))

            self.render_guided()
            self.render_compare()
            self.render_cluster_analysis(result["clustering"])
            self.auto_status.setText(
                f"Recommended: {result['selected']['name']} · "
                f"{result['selected']['best_components']} PCA component(s) · "
                f"hard exclusion {constraint['normalized'] or 'none'}"
            )
            self.hard_exclude_status.setText(describe_exclusion(constraint))
            self.statusBar().showMessage(
                "Guided analysis complete. Click any point to inspect that sample."
            )
        except Exception:
            self.err()
        finally:
            self.auto_button.setEnabled(True)

    def run_compare(self):
        if self.data is None:
            app_ui.QMessageBox.information(self, "PCA Compare", "Load a dataset first.")
            return
        try:
            meta, wn, spectra = self.data
            X, constrained_wn, constraint = self._constrained(
                spectra.to_numpy(), wn
            )
            comparisons, table = transparent_guided.compare_pca_recipes(
                X,
                constrained_wn,
                meta=meta,
                max_components=min(self.maxpc.value(), 20),
            )
            table = table.copy()
            table["hard_exclusion_cm-1"] = constraint["normalized"] or "none"

            decision = comparisons[0].get("decision") if comparisons else None
            existing = self.guided if isinstance(self.guided, dict) else {}
            self.guided = {
                **existing,
                "comparisons": comparisons,
                "comparison_table": table,
                "selected": comparisons[0] if comparisons else None,
                "suggested_label": core.suggest_label_column(
                    meta, self.metadata_schema
                ),
                "decision": decision,
                "hard_exclusion": constraint,
            }
            self.render_compare()
            self.hard_exclude_status.setText(describe_exclusion(constraint))
        except Exception:
            self.err()

    def project(self):
        if self.data is None:
            return
        try:
            self.sync()
            meta, wn, spectra = self.data
            X, constrained_wn, constraint = self._constrained(
                spectra.to_numpy(), wn
            )
            result = core.exploratory_pca(
                X, constrained_wn, self.steps, self.npcs.value()
            )
            self.last = result
            if self.method.currentText() == "PCA":
                z = result["scores"]
            else:
                z = core.embedding(
                    result["processed"],
                    "UMAP",
                    3 if self.dim.currentText() == "3D" else 2,
                    n_neighbors=min(self.neigh.value(), max(2, len(meta) - 1)),
                    min_dist=self.mind.value(),
                )
            self.embed = z
            self._draw_raw_processed()
            self.draw_projection(z)
            self._draw_loadings_variance()
            self.hard_exclude_status.setText(describe_exclusion(constraint))
        except Exception:
            self.err()

    def run_cv(self):
        if self.data is None:
            return
        try:
            self.sync()
            meta, wn, spectra = self.data
            X, constrained_wn, constraint = self._constrained(
                spectra.to_numpy(), wn
            )
            cols = [c.strip() for c in self.groups.text().split(",") if c.strip()]
            groups = (
                core.make_groups(meta, cols)
                if cols and all(c in meta for c in cols)
                else None
            )
            data, best = core.pca_cv(
                X,
                self.steps,
                self.maxpc.value(),
                self.outer.value(),
                groups,
                wn=constrained_wn,
            )
            self.cv_plot.fig.clear()
            ax = self.cv_plot.fig.add_subplot()
            ax.plot(data.components, data.RMSEC_X, label="RMSEC-X")
            ax.plot(data.components, data.RMSECV_X, label="RMSECV-X")
            ax.fill_between(
                data.components,
                np.maximum(0, data.RMSECV_X - data.RMSECV_X_SD),
                data.RMSECV_X + data.RMSECV_X_SD,
                alpha=0.15,
            )
            ax.axvline(best, linestyle="--", label=f"recommended: {best}")
            ax.legend()
            ax.set_title(
                "PCA reconstruction cross-validation · hard exclusion "
                f"{constraint['normalized'] or 'none'}"
            )
            self.cv_plot.draw()
            self.et.setCurrentWidget(self.cv_plot)
        except Exception:
            self.err()

    def run_cluster_analysis(self):
        if self.data is None:
            return
        try:
            if self.last is None:
                self.project()
            if self.last is None:
                return
            super().run_cluster_analysis()
        except Exception:
            self.err()

    def optimize(self):
        if self.data is None:
            return
        try:
            self.sync()
            meta, wn, spectra = self.data
            label = self.label.text().strip()
            if label not in meta:
                raise ValueError("Choose a valid label column.")

            valid = meta[label].notna()
            filtered_meta = meta.loc[valid].reset_index(drop=True)
            X_full = spectra.loc[valid].to_numpy()
            X, constrained_wn, constraint = self._constrained(X_full, wn)

            cols = [c.strip() for c in self.groups.text().split(",") if c.strip()]
            eligibility_table, y, groups = core.eligibility(
                filtered_meta, label, cols, self.minG.value()
            )
            allowed = set(
                eligibility_table.loc[eligibility_table.eligible, "class"]
            )
            keep = np.array([value in allowed for value in y])
            X, y, groups = X[keep], y[keep], groups[keep]
            if len(np.unique(y)) < 2:
                raise ValueError("Fewer than two eligible classes remain.")

            per_class = (
                pd.DataFrame({"y": y, "g": groups})
                .drop_duplicates()
                .groupby("y")
                .size()
            )
            outer = min(self.outer.value(), int(per_class.min()))
            if outer < 2:
                raise ValueError("Not enough independent groups for nested validation.")

            result = core.nested_optimize(
                X,
                y,
                groups,
                self.steps,
                self.trials.value(),
                self.maxpc.value(),
                outer,
                self.inner.value(),
                wn=constrained_wn,
            )
            self.opt = (result, X, y, groups)
            self.opt_wn = constrained_wn
            self.opt_constraint = constraint
            self.render_review()
            self.tabs.setCurrentWidget(self.tabs.widget(self.tabs.count() - 1))
            self.statusBar().showMessage(
                "Predictive optimization complete · hard exclusion "
                f"{constraint['normalized'] or 'none'} enforced in every trial."
            )
        except Exception:
            self.err()

    def replay_trial(self, index):
        if not hasattr(self, "trials_df") or self.trials_df.empty:
            return
        try:
            index = min(max(0, int(index)), len(self.trials_df) - 1)
            row = self.trials_df.iloc[index]
            wn = (
                self.opt_wn
                if self.opt_wn is not None
                else np.arange(self.opt[1].shape[1], dtype=float)
            )
            view = trial_projection(self.opt[1], wn, row)
            scores = view["scores"]

            note = (
                f"\n\nTRIAL-SPECIFIC VIEW\n{view['description']}\n"
                f"Recipe used by this trial: {view['recipe_text']}\n"
                f"Hard exclusion: "
                f"{(self.opt_constraint or {}).get('normalized', 'unknown') or 'none'}\n\n"
                "The Confusion Matrix and Fold Stability tabs summarize the final nested-CV "
                "evaluation and therefore do not change with this trial slider. "
                "Optimization History and Trial Projection do update."
            )
            self.trialText.setPlainText(row.to_string() + note)

            history = self.trials_df.reset_index(drop=True)
            x = np.arange(len(history))
            self.hist.fig.clear()
            ax_hist = self.hist.fig.add_subplot()
            ax_hist.plot(x, history.score, ".", label="score")
            if "running_best" in history:
                ax_hist.plot(x, history.running_best, label="running best")
            ax_hist.axvline(index, linestyle="--", linewidth=1)
            ax_hist.scatter([index], [float(row.score)], s=80, zorder=4)
            fold_text = (
                f" · outer fold {int(row.outer_fold)}"
                if "outer_fold" in row and pd.notna(row.outer_fold)
                else ""
            )
            ax_hist.set_title(
                f"Optimization history · selected record {index + 1}/{len(history)}"
                f"{fold_text} · trial {int(row.trial)}"
            )
            ax_hist.set_xlabel("Completed trial record")
            ax_hist.set_ylabel("Balanced accuracy")
            ax_hist.legend()
            self.hist.draw()

            self.trialplot.fig.clear()
            colors = pd.Categorical(self.opt[2]).codes
            if scores.shape[1] >= 3:
                ax = self.trialplot.fig.add_subplot(projection="3d")
                ax.scatter(
                    scores[:, 0], scores[:, 1], scores[:, 2],
                    c=colors, alpha=0.82
                )
                ax.set_zlabel("View PC3")
            else:
                ax = self.trialplot.fig.add_subplot()
                ax.scatter(scores[:, 0], scores[:, 1], c=colors, alpha=0.82)
            ax.set_xlabel("View PC1")
            ax.set_ylabel("View PC2")
            ax.set_title(
                f"Trial {int(row.trial)} · {row.model} · "
                f"PCA in model={'yes' if bool(row.use_pca) else 'no'} · "
                f"score={float(row.score):.3f}"
            )
            self.trialplot.draw()
        except Exception:
            self.err()

    def render_compare(self):
        super().render_compare()
        if not self.guided:
            return
        decision = self.guided.get("decision")
        if not decision and self.guided.get("comparisons"):
            decision = self.guided["comparisons"][0].get("decision")
        constraint = self.guided.get("hard_exclusion", {})
        constraint_text = constraint.get("normalized", "") or "none"
        if decision:
            self.compare_summary.setText(
                f"Default: {decision['selected_name']} · chosen by "
                f"{decision['selection_mode']} · hard exclusion {constraint_text}. "
                "See the table for every metric; no hidden weighted score is used."
            )


if __name__ == "__main__":
    app = app_ui.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = Main()
    window.show()
    sys.exit(app.exec())
