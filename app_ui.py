import copy
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)

from core import *


class Plot(Canvas):
    """Matplotlib canvas with a reusable sample-picking hook."""

    def __init__(self, figsize=(8, 5)):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        super().__init__(self.fig)
        self.pick_callback = None
        self.mpl_connect("pick_event", self._picked)

    def set_pick_callback(self, callback):
        self.pick_callback = callback

    def scatter_samples(self, ax, x, y, z=None, labels=None, indices=None, **kwargs):
        indices = np.arange(len(x)) if indices is None else np.asarray(indices)
        colors = labels
        if labels is not None and not np.issubdtype(np.asarray(labels).dtype, np.number):
            colors = pd.Categorical(labels).codes
        if z is None:
            artist = ax.scatter(x, y, c=colors, picker=5, **kwargs)
        else:
            artist = ax.scatter(x, y, z, c=colors, picker=5, **kwargs)
        artist._sample_indices = indices
        return artist

    def _picked(self, event):
        artist = event.artist
        indices = getattr(artist, "_sample_indices", None)
        if indices is None or not len(event.ind):
            return
        point = int(event.ind[0])
        if point < len(indices) and self.pick_callback:
            self.pick_callback(int(indices[point]))


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FTIR Workbench v5.2 — Guided Analysis")
        self.resize(1650, 1000)

        self.data = None
        self.steps = [
            Step("Range", dict(DEFAULTS["Range"])),
            Step("Savitzky-Golay", dict(DEFAULTS["Savitzky-Golay"])),
            Step("SNV", {}),
            Step("Mean center", {}),
        ]
        self.last = None
        self.opt = None
        self.guided = None
        self.embed = None
        self.metadata_schema = None
        self.selected_index = None
        self.frames = []
        self.manual_cluster_labels = None

        self.build()

    def build(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        header = QHBoxLayout()
        header.addWidget(QLabel("Dataset"))
        self.path = QLineEdit()
        self.path.setPlaceholderText("Choose a CSV dataset…")
        load_button = QPushButton("Load CSV")
        load_button.clicked.connect(self.load)
        header.addWidget(self.path, 1)
        header.addWidget(load_button)
        layout.addLayout(header)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self._build_guided_tab()
        self._build_qc_tab()
        self._build_compare_tab()
        self._build_exploratory_tab()
        self._build_cluster_tab()
        self._build_preprocessing_tab()
        self._build_predictive_tab()
        self._build_review_tab()
        self._build_sample_inspector()

        self.statusBar().showMessage("Load a dataset to begin. Guided Analysis is the recommended workflow.")

    def _build_guided_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)

        intro = QLabel(
            "<b>Guided Analysis</b> chooses defensible settings automatically. "
            "You do not need to know PCA, clustering, or chemometrics theory to start. "
            "Expert controls remain available in the other tabs."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        row = QHBoxLayout()
        self.auto_button = QPushButton("Analyze dataset automatically")
        self.auto_button.setMinimumHeight(42)
        self.auto_button.clicked.connect(self.auto_analyze)
        self.auto_status = QLabel("Waiting for data")
        row.addWidget(self.auto_button)
        row.addWidget(self.auto_status, 1)
        outer.addLayout(row)

        split = QSplitter(Qt.Horizontal)
        self.guide_tabs = QTabWidget()
        self.guide_pca = Plot()
        self.guide_diag = Plot()
        self.guide_dendro = Plot()
        self.guide_contrib = Plot()
        for plot in (self.guide_pca, self.guide_diag, self.guide_dendro, self.guide_contrib):
            plot.set_pick_callback(self.show_sample)
        self.guide_tabs.addTab(self.guide_pca, "PCA overview")
        self.guide_tabs.addTab(self.guide_diag, "Outlier diagnostics")
        self.guide_tabs.addTab(self.guide_dendro, "Dendrogram")
        self.guide_tabs.addTab(self.guide_contrib, "Selected-sample contributions")

        self.guide_report = QPlainTextEdit()
        self.guide_report.setReadOnly(True)
        self.guide_report.setPlainText(
            "Load a dataset, then click ‘Analyze dataset automatically’. "
            "The workbench will compare sensible preprocessing recipes, choose PCA dimensionality "
            "with cross-validation, assess clustering, and flag samples that deserve review."
        )
        split.addWidget(self.guide_tabs)
        split.addWidget(self.guide_report)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        outer.addWidget(split, 1)

        self.tabs.addTab(page, "Guided Analysis")

    def _build_qc_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        split = QSplitter(Qt.Vertical)

        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        split.addWidget(self.info)

        box = QWidget()
        box_layout = QVBoxLayout(box)
        label = QLabel(
            "<b>Detected metadata</b> — roles are inferred from both column names and the actual values. "
            "You can override any suggestion."
        )
        label.setWordWrap(True)
        box_layout.addWidget(label)

        self.meta_table = QTableWidget()
        self.meta_table.setColumnCount(6)
        self.meta_table.setHorizontalHeaderLabels(
            ["Column", "Detected role", "Confidence", "Unique", "Missing %", "Examples"]
        )
        box_layout.addWidget(self.meta_table)
        split.addWidget(box)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        layout.addWidget(split)
        self.tabs.addTab(page, "Data & Metadata QC")

    def _build_compare_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        text = QLabel(
            "The workbench compares several scientifically sensible PCA preprocessing choices. "
            "Side-by-side views are used instead of overlaying PCA axes, because different PCA models "
            "can rotate or flip their axes and an overlay can be misleading."
        )
        text.setWordWrap(True)
        layout.addWidget(text)

        row = QHBoxLayout()
        run = QPushButton("Build / refresh PCA comparison")
        run.clicked.connect(self.run_compare)
        self.compare_summary = QLabel("No comparison has been run yet.")
        row.addWidget(run)
        row.addWidget(self.compare_summary, 1)
        layout.addLayout(row)

        self.compare_tabs = QTabWidget()
        self.compare_gallery = Plot(figsize=(10, 7))
        self.compare_gallery.set_pick_callback(self.show_sample)
        self.compare_focus = Plot()
        self.compare_focus.set_pick_callback(self.show_sample)
        self.compare_tabs.addTab(self.compare_gallery, "Gallery")
        self.compare_tabs.addTab(self.compare_focus, "Slider view")
        layout.addWidget(self.compare_tabs, 1)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("PCA candidate"))
        self.compare_slider = QSlider(Qt.Horizontal)
        self.compare_slider.setRange(0, 0)
        self.compare_slider.valueChanged.connect(self.render_compare_focus)
        self.compare_detail = QLabel("—")
        slider_row.addWidget(self.compare_slider, 1)
        slider_row.addWidget(self.compare_detail)
        layout.addLayout(slider_row)

        self.compare_table = QTableWidget()
        self.compare_table.setMaximumHeight(180)
        layout.addWidget(self.compare_table)
        self.tabs.addTab(page, "PCA Compare")

    def _build_exploratory_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        controls = QHBoxLayout()
        self.method = QComboBox()
        self.method.addItems(["PCA", "UMAP"])
        self.dim = QComboBox()
        self.dim.addItems(["2D", "3D"])
        self.pc1 = QSpinBox()
        self.pc2 = QSpinBox()
        self.pc3 = QSpinBox()
        for widget in (self.pc1, self.pc2, self.pc3):
            widget.setRange(1, 50)
        self.pc1.setValue(1)
        self.pc2.setValue(2)
        self.pc3.setValue(3)

        self.npcs = QSpinBox()
        self.npcs.setRange(2, 50)
        self.npcs.setValue(12)
        self.neigh = QSpinBox()
        self.neigh.setRange(2, 200)
        self.neigh.setValue(15)
        self.mind = QDoubleSpinBox()
        self.mind.setRange(0, 1)
        self.mind.setSingleStep(0.05)
        self.mind.setValue(0.1)

        run = QPushButton("Run projection")
        run.clicked.connect(self.project)
        cv = QPushButton("PCA CV")
        cv.clicked.connect(self.run_cv)

        for widget in (
            QLabel("Method"), self.method, QLabel("View"), self.dim,
            QLabel("X/PC"), self.pc1, QLabel("Y/PC"), self.pc2,
            QLabel("Z/PC"), self.pc3, QLabel("PCA components"), self.npcs,
            QLabel("UMAP neighbors"), self.neigh, QLabel("min_dist"), self.mind,
            run, cv,
        ):
            controls.addWidget(widget)
        layout.addLayout(controls)

        self.et = QTabWidget()
        self.raw = Plot()
        self.proc = Plot()
        self.proj = Plot()
        self.loadings_plot = Plot()
        self.var = Plot()
        self.cv_plot = Plot()
        for plot in (self.raw, self.proc, self.proj, self.loadings_plot, self.var, self.cv_plot):
            plot.set_pick_callback(self.show_sample)
        for plot, title in (
            (self.raw, "Raw"),
            (self.proc, "Processed"),
            (self.proj, "2D/3D projection"),
            (self.loadings_plot, "Loadings"),
            (self.var, "Variance"),
            (self.cv_plot, "PCA CV"),
        ):
            self.et.addTab(plot, title)
        layout.addWidget(self.et)

        manual = QHBoxLayout()
        self.cl = QComboBox()
        self.cl.addItems(["KMeans", "Agglomerative", "DBSCAN"])
        self.k = QSpinBox()
        self.k.setRange(2, 20)
        self.k.setValue(3)
        cluster_button = QPushButton("Apply manual cluster")
        cluster_button.clicked.connect(self.do_cluster)
        self.iter = QSlider(Qt.Horizontal)
        self.iter.setRange(0, 0)
        self.iter.valueChanged.connect(self.replay_cluster)
        for widget in (
            QLabel("Manual clustering"), self.cl, QLabel("K"), self.k,
            cluster_button, QLabel("K-means iteration"), self.iter,
        ):
            manual.addWidget(widget)
        layout.addLayout(manual)

        self.tabs.addTab(page, "Exploratory / Expert")

    def _build_cluster_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        row = QHBoxLayout()
        button = QPushButton("Analyze clustering automatically")
        button.clicked.connect(self.run_cluster_analysis)
        self.cluster_summary = QLabel("No clustering analysis yet.")
        row.addWidget(button)
        row.addWidget(self.cluster_summary, 1)
        layout.addLayout(row)

        split = QSplitter(Qt.Horizontal)
        self.cluster_plot = Plot()
        self.cluster_plot.set_pick_callback(self.show_sample)
        self.cluster_report = QPlainTextEdit()
        self.cluster_report.setReadOnly(True)
        split.addWidget(self.cluster_plot)
        split.addWidget(self.cluster_report)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        layout.addWidget(split, 1)

        self.cluster_table = QTableWidget()
        self.cluster_table.setMaximumHeight(220)
        layout.addWidget(self.cluster_table)
        self.tabs.addTab(page, "Cluster Analysis")

    def _build_preprocessing_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(
            "<b>Expert preprocessing controls.</b> Guided Analysis will populate these with its recommended recipe. "
            "Drag to reorder only when you intentionally want to override the guided workflow."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.lst = QListWidget()
        self.lst.setDragDropMode(QAbstractItemView.InternalMove)
        layout.addWidget(self.lst)

        row = QHBoxLayout()
        self.add = QComboBox()
        self.add.addItems(DEFAULTS)
        row.addWidget(self.add)
        for title, callback in (
            ("Add", self.add_step),
            ("Remove", self.remove_step),
            ("Edit", self.edit_step),
            ("Save recipe", self.save_recipe),
            ("Load recipe", self.load_recipe),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        self.tabs.addTab(page, "Preprocessing / Expert")
        self.refresh()

    def _build_predictive_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        note = QLabel(
            "Predictive modeling uses grouped nested cross-validation when independent grouping metadata is available. "
            "Guided Analysis fills in recommended label/group fields automatically."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        grid = QGridLayout()
        self.label = QLineEdit()
        self.label.setPlaceholderText("Detected automatically after loading data")
        self.groups = QLineEdit()
        self.groups.setPlaceholderText("Detected independent grouping columns")
        self.minG = QSpinBox()
        self.minG.setRange(2, 20)
        self.minG.setValue(3)
        self.outer = QSpinBox()
        self.outer.setRange(2, 10)
        self.outer.setValue(3)
        self.inner = QSpinBox()
        self.inner.setRange(2, 10)
        self.inner.setValue(2)
        self.trials = QSpinBox()
        self.trials.setRange(1, 1000)
        self.trials.setValue(20)
        self.maxpc = QSpinBox()
        self.maxpc.setRange(2, 100)
        self.maxpc.setValue(20)

        controls = [
            ("Label", self.label),
            ("Group columns", self.groups),
            ("Min groups/class", self.minG),
            ("Outer folds", self.outer),
            ("Inner folds", self.inner),
            ("Trials/fold", self.trials),
            ("Max PCs", self.maxpc),
        ]
        for i, (name, widget) in enumerate(controls):
            grid.addWidget(QLabel(name), i // 4, (i % 4) * 2)
            grid.addWidget(widget, i // 4, (i % 4) * 2 + 1)
        layout.addLayout(grid)

        row = QHBoxLayout()
        audit = QPushButton("Preview eligibility")
        audit.clicked.connect(self.preview)
        go = QPushButton("Run nested optimization")
        go.clicked.connect(self.optimize)
        row.addWidget(audit)
        row.addWidget(go)
        layout.addLayout(row)

        self.elig = QTableWidget()
        layout.addWidget(self.elig)
        self.tabs.addTab(page, "Predictive Modeling")

    def _build_review_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.reviewTabs = QTabWidget()
        self.hist = Plot()
        self.model_compare = Plot()
        self.conf = Plot()
        self.fold = Plot()
        self.trialplot = Plot()
        for plot, title in (
            (self.hist, "Optimization history"),
            (self.model_compare, "Model comparison"),
            (self.conf, "Confusion matrix"),
            (self.fold, "Fold stability"),
            (self.trialplot, "Trial projection"),
        ):
            self.reviewTabs.addTab(plot, title)
        layout.addWidget(self.reviewTabs)

        row = QHBoxLayout()
        self.trialSlider = QSlider(Qt.Horizontal)
        self.trialSlider.setRange(0, 0)
        self.trialSlider.valueChanged.connect(self.replay_trial)
        row.addWidget(QLabel("Trial"))
        row.addWidget(self.trialSlider)
        layout.addLayout(row)

        self.trialText = QPlainTextEdit()
        self.trialText.setMaximumHeight(150)
        layout.addWidget(self.trialText)
        self.tabs.addTab(page, "Model Review")

    def _build_sample_inspector(self):
        dock = QDockWidget("Sample Inspector", self)
        container = QWidget()
        layout = QVBoxLayout(container)
        hint = QLabel("Click any sample point in PCA, UMAP, diagnostics, PCA Compare, or clustering.")
        hint.setWordWrap(True)
        self.sample_text = QPlainTextEdit()
        self.sample_text.setReadOnly(True)
        self.sample_text.setPlainText("No sample selected.")
        layout.addWidget(hint)
        layout.addWidget(self.sample_text)
        dock.setWidget(container)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.sample_dock = dock

    def load(self):
        path = self.path.text().strip()
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open analytical CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            self.data = load_ftir(path)
            self.path.setText(path)
            meta, wn, spectra = self.data
            self.metadata_schema = infer_metadata_schema(meta)
            self._populate_qc()
            self._populate_metadata_table()
            self._apply_metadata_defaults()
            self._apply_dataset_defaults()

            self.guided = None
            self.last = None
            self.embed = None
            self.selected_index = None
            self.auto_status.setText("Data loaded — ready for automatic analysis")
            self.guide_report.setPlainText(
                f"Loaded {len(meta)} samples with {len(wn)} spectral variables. "
                "Metadata roles were inferred from the real column names and values. "
                "Click ‘Analyze dataset automatically’ for the recommended workflow."
            )
            self.tabs.setCurrentIndex(0)
            self.statusBar().showMessage("Dataset loaded. Guided Analysis is ready.")
        except Exception:
            self.err()

    def _populate_qc(self):
        meta, wn, spectra = self.data
        numeric_missing = spectra.isna().mean().mean() * 100
        lines = [
            f"Samples: {len(meta)}",
            f"Spectral variables: {len(wn)}",
            f"Spectral range: {wn.min():.2f} – {wn.max():.2f}",
            f"Spectral missingness: {numeric_missing:.2f}%",
            f"Metadata columns: {', '.join(map(str, meta.columns)) or 'none'}",
            "",
            "Metadata missing values:",
            meta.isna().sum().to_string() if len(meta.columns) else "No metadata columns detected.",
        ]
        self.info.setPlainText("\n".join(lines))

    def _populate_metadata_table(self):
        schema = self.metadata_schema
        self.meta_table.setRowCount(len(schema))
        for row, record in schema.reset_index(drop=True).iterrows():
            self.meta_table.setItem(row, 0, QTableWidgetItem(str(record["column"])))
            combo = QComboBox()
            combo.addItems(METADATA_ROLES)
            combo.setCurrentText(str(record["role"]))
            combo.currentTextChanged.connect(
                lambda role, r=row: self._metadata_role_override(r, role)
            )
            self.meta_table.setCellWidget(row, 1, combo)
            self.meta_table.setItem(row, 2, QTableWidgetItem(f"{record['confidence']:.0%}"))
            self.meta_table.setItem(row, 3, QTableWidgetItem(str(record["unique_values"])))
            self.meta_table.setItem(row, 4, QTableWidgetItem(f"{record['missing_percent']:.1f}"))
            self.meta_table.setItem(row, 5, QTableWidgetItem(str(record["example"])))
        self.meta_table.resizeColumnsToContents()

    def _metadata_role_override(self, row, role):
        if self.metadata_schema is None or row >= len(self.metadata_schema):
            return
        self.metadata_schema.loc[row, "role"] = role
        self.metadata_schema.loc[row, "confidence"] = 1.0
        self.metadata_schema.loc[row, "reason"] = "user override"
        self.meta_table.setItem(row, 2, QTableWidgetItem("100%"))
        self._apply_metadata_defaults()

    def _apply_metadata_defaults(self):
        if self.data is None:
            return
        meta, _, _ = self.data
        label = suggest_label_column(meta, self.metadata_schema)
        groups = suggest_group_columns(meta, self.metadata_schema)
        if label:
            self.label.setText(label)
        if groups:
            self.groups.setText(",".join(groups))

        if label and groups and label in meta:
            try:
                eligible, _, _ = eligibility(
                    meta.dropna(subset=[label]).reset_index(drop=True),
                    label,
                    groups,
                    2,
                )
                if not eligible.empty:
                    minimum = int(eligible.independent_groups.min())
                    self.minG.setValue(max(2, min(3, minimum)))
                    self.outer.setValue(max(2, min(5, minimum)))
                    self.inner.setValue(max(2, min(3, max(2, minimum - 1))))
            except Exception:
                pass

    def _apply_dataset_defaults(self):
        if self.data is None:
            return
        meta, wn, spectra = self.data
        n = len(meta)
        self.neigh.setValue(max(5, min(40, int(round(np.sqrt(max(4, n)) * 2)))))
        self.mind.setValue(0.05 if n < 60 else 0.10 if n < 250 else 0.15)
        self.npcs.setValue(max(3, min(20, len(meta) - 2, len(wn))))
        self.maxpc.setValue(max(2, min(25, len(meta) - 2, len(wn))))

    def auto_analyze(self):
        if self.data is None:
            QMessageBox.information(self, "Guided Analysis", "Load a dataset first.")
            return
        try:
            self.auto_button.setEnabled(False)
            self.auto_status.setText("Analyzing preprocessing, PCA, diagnostics, and clustering…")
            QApplication.processEvents()

            meta, wn, spectra = self.data
            result = guided_analysis(
                spectra.to_numpy(),
                wn,
                meta,
                max_components=min(self.maxpc.value(), 20),
            )
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
                f"{result['selected']['best_components']} PCA component(s)"
            )
            self.statusBar().showMessage("Guided analysis complete. Click any point to inspect that sample.")
        except Exception:
            self.err()
        finally:
            self.auto_button.setEnabled(True)

    def render_guided(self):
        if not self.guided:
            return
        result = self.guided
        self.guide_report.setPlainText(
            result["report"]
            + "\n\nHow to use this screen:\n"
              "• PCA overview: patterns and groups in the main variation.\n"
              "• Outlier diagnostics: samples beyond either guide line deserve review.\n"
              "• Dendrogram: a second view of sample similarity.\n"
              "• Click any point to open its metadata and contribution profile."
        )
        self._render_guided_pca()
        self._render_diagnostics()
        self._render_dendrogram()
        self._render_contributions()

    def _default_colors(self):
        if self.data is None:
            return None, None
        meta = self.data[0]
        label = None
        if self.guided and self.guided.get("suggested_label") in meta:
            label = self.guided["suggested_label"]
        elif self.label.text().strip() in meta:
            label = self.label.text().strip()
        if label:
            return meta[label].astype(str).to_numpy(), label
        if self.guided and "clustering" in self.guided:
            return self.guided["clustering"]["best_labels"], "automatic cluster"
        return None, None

    def _render_guided_pca(self):
        if self.last is None:
            return
        scores = self.last["scores"]
        colors, label = self._default_colors()
        self.guide_pca.fig.clear()
        ax = self.guide_pca.fig.add_subplot()
        self.guide_pca.scatter_samples(
            ax, scores[:, 0], scores[:, 1], labels=colors, s=48, alpha=0.82
        )
        if self.selected_index is not None and self.selected_index < len(scores):
            ax.scatter(
                scores[self.selected_index, 0],
                scores[self.selected_index, 1],
                s=180,
                facecolors="none",
                edgecolors="black",
                linewidths=2,
            )
        variance = self.last["variance"]
        ax.set_xlabel(f"PC1 ({variance[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({variance[1]*100:.1f}%)")
        ax.set_title(f"Recommended PCA" + (f" · colored by {label}" if label else ""))
        self.guide_pca.draw()

    def _render_diagnostics(self):
        if not self.guided or "diagnostics" not in self.guided:
            return
        diagnostics = self.guided["diagnostics"]
        table = diagnostics["table"]
        colors, _ = self._default_colors()
        self.guide_diag.fig.clear()
        ax = self.guide_diag.fig.add_subplot()
        self.guide_diag.scatter_samples(
            ax,
            table.Hotelling_T2.to_numpy(),
            table.Q_residual.to_numpy(),
            labels=colors,
            s=50,
            alpha=0.82,
        )
        ax.axvline(diagnostics["t2_limit"], linestyle="--", linewidth=1)
        ax.axhline(diagnostics["q_limit"], linestyle="--", linewidth=1)
        if self.selected_index is not None and self.selected_index < len(table):
            row = table.iloc[self.selected_index]
            ax.scatter(
                [row.Hotelling_T2],
                [row.Q_residual],
                s=180,
                facecolors="none",
                edgecolors="black",
                linewidths=2,
            )
        ax.set_xlabel("Hotelling T² — unusual within modeled PCA space")
        ax.set_ylabel("Q residual — variation not explained by PCA")
        ax.set_title("Samples beyond either guide line deserve review")
        self.guide_diag.draw()

    def _render_dendrogram(self):
        if not self.guided or "clustering" not in self.guided:
            return
        self.guide_dendro.fig.clear()
        ax = self.guide_dendro.fig.add_subplot()
        meta = self.data[0]
        id_column = None
        if self.metadata_schema is not None:
            ids = self.metadata_schema.loc[self.metadata_schema.role == "sample_id", "column"]
            if len(ids):
                id_column = str(ids.iloc[0])
        labels = (
            meta[id_column].astype(str).tolist()
            if id_column and id_column in meta
            else [str(i + 1) for i in range(len(meta))]
        )
        dendrogram(
            self.guided["clustering"]["linkage"],
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=6,
            ax=ax,
        )
        ax.set_title("Ward dendrogram on the recommended PCA scores")
        ax.set_ylabel("Distance")
        self.guide_dendro.draw()

    def _render_contributions(self):
        self.guide_contrib.fig.clear()
        ax = self.guide_contrib.fig.add_subplot()
        if self.last is None or self.selected_index is None:
            ax.text(
                0.5, 0.5, "Click a sample point to see which spectral regions contribute most.",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_axis_off()
            self.guide_contrib.draw()
            return

        contribution = sample_contributions(self.last, self.selected_index)
        q = contribution.Q_contribution.to_numpy()
        t2 = contribution.T2_related_contribution.to_numpy()
        q_scaled = q / max(float(np.max(q)), 1e-12)
        t2_scaled = t2 / max(float(np.max(t2)), 1e-12)
        ax.plot(contribution.wavenumber, q_scaled, label="Unmodeled (Q) contribution", linewidth=1)
        ax.plot(contribution.wavenumber, t2_scaled, label="PCA-space (T²-related) contribution", linewidth=1)
        ax.invert_xaxis()
        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Relative contribution")
        ax.set_title(f"Sample {self.selected_index + 1}: spectral regions driving its unusualness")
        ax.legend(fontsize=8)
        self.guide_contrib.draw()

    def run_compare(self):
        if self.data is None:
            QMessageBox.information(self, "PCA Compare", "Load a dataset first.")
            return
        try:
            if not self.guided or "comparisons" not in self.guided:
                meta, wn, spectra = self.data
                comparisons, table = compare_pca_recipes(
                    spectra.to_numpy(), wn, meta=meta, max_components=min(self.maxpc.value(), 20)
                )
                self.guided = {
                    "comparisons": comparisons,
                    "comparison_table": table,
                    "suggested_label": suggest_label_column(meta, self.metadata_schema),
                }
            self.render_compare()
        except Exception:
            self.err()

    def render_compare(self):
        if not self.guided or "comparisons" not in self.guided:
            return
        comparisons = self.guided["comparisons"]
        summary = self.guided["comparison_table"]
        self.compare_slider.setRange(0, max(0, len(comparisons) - 1))
        self._table(self.compare_table, summary.drop(columns=["objective"], errors="ignore"))

        self.compare_gallery.fig.clear()
        count = min(4, len(comparisons))
        colors, color_label = self._default_colors()
        for i in range(count):
            candidate = comparisons[i]
            scores = candidate["pca_result"]["scores"]
            ax = self.compare_gallery.fig.add_subplot(2, 2, i + 1)
            self.compare_gallery.scatter_samples(
                ax, scores[:, 0], scores[:, 1], labels=colors, s=25, alpha=0.78
            )
            variance = candidate["pca_result"]["variance"]
            ax.set_title(
                f"{candidate['name']}\n"
                f"{candidate['best_components']} PCs · Q²-X {candidate['Q2_X']:.3f}"
            )
            ax.set_xlabel(f"PC1 {variance[0]*100:.1f}%")
            ax.set_ylabel(f"PC2 {variance[1]*100:.1f}%")
        self.compare_gallery.fig.suptitle(
            "Top PCA preprocessing views" + (f" · color = {color_label}" if color_label else "")
        )
        self.compare_gallery.draw()

        self.compare_summary.setText(
            f"Best recommendation: {comparisons[0]['name']} "
            f"with {comparisons[0]['best_components']} component(s)."
        )
        self.render_compare_focus(self.compare_slider.value())

    def render_compare_focus(self, index):
        if not self.guided or "comparisons" not in self.guided:
            return
        comparisons = self.guided["comparisons"]
        if not comparisons:
            return
        index = min(max(0, int(index)), len(comparisons) - 1)
        candidate = comparisons[index]
        scores = candidate["pca_result"]["scores"]
        colors, color_label = self._default_colors()

        self.compare_focus.fig.clear()
        ax = self.compare_focus.fig.add_subplot()
        self.compare_focus.scatter_samples(
            ax, scores[:, 0], scores[:, 1], labels=colors, s=48, alpha=0.82
        )
        variance = candidate["pca_result"]["variance"]
        ax.set_xlabel(f"PC1 ({variance[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({variance[1]*100:.1f}%)")
        ax.set_title(candidate["name"] + (f" · color = {color_label}" if color_label else ""))
        self.compare_focus.draw()
        self.compare_detail.setText(
            f"{index+1}/{len(comparisons)} · {candidate['best_components']} PCs · "
            f"Q²-X {candidate['Q2_X']:.3f} · RMSECV-X {candidate['RMSECV_X']:.4g}"
        )

    def show_sample(self, index):
        if self.data is None:
            return
        meta = self.data[0]
        index = int(index)
        if index < 0 or index >= len(meta):
            return
        self.selected_index = index

        lines = [f"Sample #{index + 1}", "=" * 32, "Metadata"]
        row = meta.iloc[index]
        schema_by_column = {}
        if self.metadata_schema is not None:
            schema_by_column = self.metadata_schema.set_index("column")["role"].to_dict()
        for column, value in row.items():
            role = schema_by_column.get(str(column), "other")
            lines.append(f"{column} [{role}]: {value}")

        if self.last is not None and index < len(self.last["scores"]):
            lines += ["", "PCA"]
            for pc, value in enumerate(self.last["scores"][index, : min(5, self.last["scores"].shape[1])], 1):
                lines.append(f"PC{pc}: {value:.5g}")

        if self.guided and "diagnostics" in self.guided:
            diag = self.guided["diagnostics"]["table"].iloc[index]
            lines += [
                "",
                "Diagnostics",
                f"Hotelling T²: {diag.Hotelling_T2:.5g}",
                f"Q residual: {diag.Q_residual:.5g}",
                f"Review flag: {'YES' if diag.review_flag else 'no'}",
            ]
        if self.guided and "clustering" in self.guided:
            label = self.guided["clustering"]["best_labels"][index]
            lines += [
                "",
                "Automatic clustering",
                f"Recommended method: {self.guided['clustering']['best_method']}",
                f"Cluster: {label if label >= 0 else 'noise / unassigned'}",
            ]

        lines += [
            "",
            "Interpretation note",
            "A diagnostic flag means ‘review this sample,’ not ‘this sample is wrong.’ "
            "Use the contribution view and original metadata before drawing conclusions.",
        ]
        self.sample_text.setPlainText("\n".join(lines))
        self.sample_dock.show()
        self._render_contributions()
        if self.guided:
            self._render_guided_pca()
            self._render_diagnostics()
            if "clustering" in self.guided:
                self.render_cluster_analysis(self.guided["clustering"])

    def sync(self):
        order = [self.lst.item(i).data(Qt.UserRole) for i in range(self.lst.count())]
        old = self.steps[:]
        if len(order) == len(old) and all(isinstance(i, int) and i < len(old) for i in order):
            self.steps = [old[i] for i in order]

    def refresh(self):
        self.lst.clear()
        for i, step in enumerate(self.steps):
            item = QListWidgetItem(f"{step.name} | {json.dumps(step.params)}")
            item.setData(Qt.UserRole, i)
            self.lst.addItem(item)

    def add_step(self):
        self.sync()
        name = self.add.currentText()
        self.steps.append(Step(name, dict(DEFAULTS[name])))
        self.refresh()

    def remove_step(self):
        self.sync()
        index = self.lst.currentRow()
        if index >= 0:
            self.steps.pop(index)
            self.refresh()

    def edit_step(self):
        self.sync()
        index = self.lst.currentRow()
        if index < 0:
            return
        text, ok = QInputDialog.getMultiLineText(
            self, "Parameters", "JSON", json.dumps(self.steps[index].params, indent=2)
        )
        if ok:
            try:
                self.steps[index].params = json.loads(text)
                self.refresh()
            except Exception as exc:
                QMessageBox.warning(self, "Invalid JSON", str(exc))

    def save_recipe(self):
        self.sync()
        path, _ = QFileDialog.getSaveFileName(self, "Save recipe", "", "JSON (*.json)")
        if path:
            Path(path).write_text(
                json.dumps([asdict(step) for step in self.steps], indent=2),
                encoding="utf-8",
            )

    def load_recipe(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load recipe", "", "JSON (*.json)")
        if path:
            self.steps = [
                Step(**record)
                for record in json.loads(Path(path).read_text(encoding="utf-8"))
            ]
            self.refresh()

    def axes(self):
        return self.pc1.value() - 1, self.pc2.value() - 1, self.pc3.value() - 1

    def project(self):
        if self.data is None:
            return
        try:
            self.sync()
            meta, wn, spectra = self.data
            result = exploratory_pca(
                spectra.to_numpy(), wn, self.steps, self.npcs.value()
            )
            self.last = result
            if self.method.currentText() == "PCA":
                z = result["scores"]
            else:
                z = embedding(
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
        except Exception:
            self.err()

    def _draw_raw_processed(self):
        meta, wn, spectra = self.data
        result = self.last
        self.raw.fig.clear()
        ax = self.raw.fig.add_subplot()
        ax.plot(wn, spectra.iloc[:80].T, alpha=0.18, linewidth=0.5)
        ax.invert_xaxis()
        ax.set_title("Raw spectra")
        self.raw.draw()

        self.proc.fig.clear()
        ax = self.proc.fig.add_subplot()
        ax.plot(result["wn"], result["processed"][:80].T, alpha=0.18, linewidth=0.5)
        ax.invert_xaxis()
        ax.set_title("Processed spectra")
        self.proc.draw()

    def _draw_loadings_variance(self):
        result = self.last
        self.loadings_plot.fig.clear()
        ax = self.loadings_plot.fig.add_subplot()
        ax.plot(
            result["wn"],
            result["loadings"][:, : min(5, result["loadings"].shape[1])],
        )
        ax.invert_xaxis()
        ax.set_title("PCA loadings")
        self.loadings_plot.draw()

        self.var.fig.clear()
        ax = self.var.fig.add_subplot()
        variance = result["variance"]
        ax.bar(range(1, len(variance) + 1), 100 * variance)
        ax.plot(range(1, len(variance) + 1), 100 * np.cumsum(variance), marker="o")
        ax.set_title("Explained variance")
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Percent")
        self.var.draw()

    def draw_projection(self, z, labels=None):
        self.proj.fig.clear()
        x, y, zz = self.axes()
        is_3d = self.dim.currentText() == "3D"
        ax = self.proj.fig.add_subplot(projection="3d" if is_3d else None)
        if self.method.currentText() == "UMAP":
            x, y, zz = 0, 1, 2
        x = min(x, z.shape[1] - 1)
        y = min(y, z.shape[1] - 1)
        zz = min(zz, z.shape[1] - 1)

        if labels is None:
            labels, _ = self._default_colors()
        if is_3d and z.shape[1] >= 3:
            self.proj.scatter_samples(
                ax, z[:, x], z[:, y], z[:, zz], labels=labels, s=42, alpha=0.82
            )
            ax.set_zlabel(f"{self.method.currentText()} {zz + 1}")
        else:
            self.proj.scatter_samples(
                ax, z[:, x], z[:, y], labels=labels, s=42, alpha=0.82
            )
        ax.set_xlabel(f"{self.method.currentText()} {x + 1}")
        ax.set_ylabel(f"{self.method.currentText()} {y + 1}")
        self.proj.draw()

    def run_cv(self):
        if self.data is None:
            return
        try:
            self.sync()
            meta, wn, spectra = self.data
            cols = [c.strip() for c in self.groups.text().split(",") if c.strip()]
            groups = make_groups(meta, cols) if cols and all(c in meta for c in cols) else None
            data, best = pca_cv(
                spectra.to_numpy(),
                self.steps,
                self.maxpc.value(),
                self.outer.value(),
                groups,
                wn=wn,
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
            ax.set_title("PCA reconstruction cross-validation")
            self.cv_plot.draw()
            self.et.setCurrentWidget(self.cv_plot)
        except Exception:
            self.err()

    def do_cluster(self):
        if self.embed is None:
            self.project()
        if self.embed is None:
            return
        try:
            z = self.embed[:, : min(10, self.embed.shape[1])]
            labels, score = cluster(z, self.cl.currentText(), self.k.value())
            self.manual_cluster_labels = labels
            self.draw_projection(self.embed, labels)
            self.frames = (
                kmeans_history(self.embed[:, : min(3, self.embed.shape[1])], self.k.value())
                if self.cl.currentText() == "KMeans"
                else []
            )
            self.iter.setRange(0, max(0, len(self.frames) - 1))
            self.statusBar().showMessage(
                f"Manual {self.cl.currentText()} clustering · silhouette {score:.3f}"
                if np.isfinite(score)
                else f"Manual {self.cl.currentText()} clustering complete."
            )
        except Exception:
            self.err()

    def replay_cluster(self, index):
        if self.frames:
            self.draw_projection(self.embed, self.frames[index]["labels"])

    def run_cluster_analysis(self):
        if self.data is None:
            return
        try:
            if self.last is None:
                self.sync()
                meta, wn, spectra = self.data
                self.last = exploratory_pca(
                    spectra.to_numpy(), wn, self.steps, self.npcs.value()
                )
            n_pcs = min(10, self.last["scores"].shape[1])
            result = analyze_clustering(self.last["scores"][:, :n_pcs])
            if self.guided is None:
                self.guided = {
                    "clustering": result,
                    "pca_result": self.last,
                    "suggested_label": suggest_label_column(self.data[0], self.metadata_schema),
                }
            else:
                self.guided["clustering"] = result
            self.render_cluster_analysis(result)
        except Exception:
            self.err()

    def render_cluster_analysis(self, result):
        self.cluster_report.setPlainText(
            result["report"]
            + "\n\nReading this safely:\n"
              "A strong silhouette means samples are separated cleanly in the selected PCA score space. "
              "Agreement between K-Means and Ward clustering makes the pattern more reproducible. "
              "Weak agreement or weak silhouette means the data may be a continuum rather than distinct groups."
        )
        self.cluster_summary.setText(
            f"Recommended: {result['best_method']} · "
            f"silhouette {float(result['best_params']['silhouette']):.3f}"
        )
        table = result["table"].copy()
        self._table(
            self.cluster_table,
            table[[c for c in ("method", "k", "silhouette", "coverage", "agreement", "eps", "min_samples") if c in table]],
        )

        scores = self.last["scores"] if self.last is not None else self.embed
        labels = result["best_labels"]
        self.cluster_plot.fig.clear()
        ax = self.cluster_plot.fig.add_subplot()
        self.cluster_plot.scatter_samples(
            ax, scores[:, 0], scores[:, 1], labels=labels, s=50, alpha=0.82
        )
        if self.selected_index is not None and self.selected_index < len(scores):
            ax.scatter(
                scores[self.selected_index, 0],
                scores[self.selected_index, 1],
                s=180,
                facecolors="none",
                edgecolors="black",
                linewidths=2,
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Automatic clustering: {result['best_method']}")
        self.cluster_plot.draw()

    def preview(self):
        if self.data is None:
            return
        try:
            meta, _, _ = self.data
            label = self.label.text().strip()
            cols = [c.strip() for c in self.groups.text().split(",") if c.strip()]
            if label not in meta:
                raise ValueError("Choose a valid metadata label.")
            if not cols:
                raise ValueError("Choose at least one independent grouping column.")
            data, _, _ = eligibility(
                meta.dropna(subset=[label]).reset_index(drop=True),
                label,
                cols,
                self.minG.value(),
            )
            self._table(self.elig, data)
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
            X = spectra.loc[valid].to_numpy()
            cols = [c.strip() for c in self.groups.text().split(",") if c.strip()]
            eligibility_table, y, groups = eligibility(
                filtered_meta, label, cols, self.minG.value()
            )
            allowed = set(eligibility_table.loc[eligibility_table.eligible, "class"])
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

            result = nested_optimize(
                X,
                y,
                groups,
                self.steps,
                self.trials.value(),
                self.maxpc.value(),
                outer,
                self.inner.value(),
                wn=wn,
            )
            self.opt = (result, X, y, groups)
            self.render_review()
            self.tabs.setCurrentWidget(self.tabs.widget(self.tabs.count() - 1))
        except Exception:
            self.err()

    def render_review(self):
        result, X, y, groups = self.opt
        history = result["history"]
        complete = history[history.status == "complete"].copy()
        complete["running_best"] = complete.objective.cummax()

        self.hist.fig.clear()
        ax = self.hist.fig.add_subplot()
        ax.plot(complete.trial, complete.score, ".", label="score")
        ax.plot(complete.trial, complete.running_best, label="running best")
        ax.legend()
        self.hist.draw()

        self.model_compare.fig.clear()
        ax = self.model_compare.fig.add_subplot()
        complete.boxplot(column="score", by=["model", "use_pca"], ax=ax)
        self.model_compare.fig.suptitle("")
        self.model_compare.draw()

        self.conf.fig.clear()
        ax = self.conf.fig.add_subplot()
        ax.imshow(result["cm"])
        ax.set_xticks(range(len(result["classes"])), result["classes"], rotation=45, ha="right")
        ax.set_yticks(range(len(result["classes"])), result["classes"])
        self.conf.draw()

        self.fold.fig.clear()
        ax = self.fold.fig.add_subplot()
        ax.plot(result["folds"].fold, result["folds"].balanced_accuracy, "o-", label="balanced accuracy")
        ax.plot(result["folds"].fold, result["folds"].macro_f1, "s-", label="macro F1")
        ax.legend()
        self.fold.draw()

        self.trials_df = complete.reset_index(drop=True)
        self.trialSlider.setRange(0, max(0, len(self.trials_df) - 1))
        self.replay_trial(0)

        out = Path(self.path.text()).with_name(Path(self.path.text()).stem + "_v5_2_results")
        out.mkdir(exist_ok=True)
        history.to_csv(out / "optimization_history.csv", index=False)
        result["folds"].to_csv(out / "outer_fold_metrics.csv", index=False)
        result["report"].to_csv(out / "classification_report.csv")
        pd.DataFrame(
            result["cm"], index=result["classes"], columns=result["classes"]
        ).to_csv(out / "confusion_matrix.csv")

    def replay_trial(self, index):
        if not hasattr(self, "trials_df") or self.trials_df.empty:
            return
        row = self.trials_df.iloc[index]
        self.trialText.setPlainText(row.to_string())
        self.trialplot.fig.clear()
        ax = self.trialplot.fig.add_subplot(projection="3d")
        result = self.last
        if result is None:
            result = exploratory_pca(
                self.opt[1],
                np.arange(self.opt[1].shape[1]),
                self.steps,
                3,
            )
        scores = result["scores"]
        if scores.shape[1] >= 3:
            ax.scatter(scores[:, 0], scores[:, 1], scores[:, 2])
        ax.set_title(
            f"Trial {int(row.trial)} {row.model} PCA={row.use_pca} score={row.score:.3f}"
        )
        self.trialplot.draw()

    def _table(self, widget, data):
        data = data.reset_index(drop=True)
        widget.setRowCount(len(data))
        widget.setColumnCount(len(data.columns))
        widget.setHorizontalHeaderLabels([str(column) for column in data.columns])
        for i, row in data.iterrows():
            for j, value in enumerate(row):
                if isinstance(value, (float, np.floating)):
                    text = "" if not np.isfinite(value) else f"{value:.5g}"
                else:
                    text = str(value)
                widget.setItem(i, j, QTableWidgetItem(text))
        widget.resizeColumnsToContents()

    def err(self):
        text = traceback.format_exc()
        message = text.splitlines()[-1] if text.strip() else "Unknown error"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("FTIR Workbench")
        box.setText(message)
        box.setDetailedText(text)
        box.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = Main()
    window.show()
    sys.exit(app.exec())
