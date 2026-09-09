from __future__ import annotations

"""FTIR Workbench v5.2 launcher.

The full PySide6 interface lives in ``app_ui.py``. This launcher wires in the
transparent guided-analysis rules before the UI imports names from ``core``.
Keeping the decision engine separate makes the automatic behavior easy to audit,
test, and change without hiding statistical choices inside GUI code.
"""

import sys

import core
import transparent_guided

# Explicitly replace only the two guided-selection entry points. All preprocessing,
# PCA, diagnostics, clustering, and predictive-model implementations remain in
# core.py and are unchanged by this launcher.
core.compare_pca_recipes = transparent_guided.compare_pca_recipes
core.guided_analysis = transparent_guided.guided_analysis

import app_ui


class Main(app_ui.Main):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FTIR Workbench v5.2 — Transparent Guided Analysis")

        # Start Expert mode from the least assumptive useful PCA preprocessing.
        # Guided Analysis will replace this with its selected recipe after it has
        # compared all explicit candidates. SNV is never applied by default here.
        self.steps = [
            core.Step("Range", dict(core.DEFAULTS["Range"])),
            core.Step("Mean center", {}),
        ]
        self.refresh()
        self.statusBar().showMessage(
            "Load a dataset, then use Guided Analysis. Every automatic choice is shown in PCA Compare."
        )

    def render_compare(self):
        super().render_compare()
        if not self.guided:
            return
        decision = self.guided.get("decision")
        if not decision and self.guided.get("comparisons"):
            decision = self.guided["comparisons"][0].get("decision")
        if decision:
            self.compare_summary.setText(
                f"Default: {decision['selected_name']} · chosen by {decision['selection_mode']}. "
                "See the table for every metric; no hidden weighted score is used."
            )


if __name__ == "__main__":
    app = app_ui.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = Main()
    window.show()
    sys.exit(app.exec())
