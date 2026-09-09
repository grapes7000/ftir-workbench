# v5.2 Transparent Guided Analysis

The v5.2 automatic PCA workflow is intentionally rule-based. There is no hidden machine-learning model choosing preprocessing settings.

## What is compared

Guided Analysis evaluates an explicit list of preprocessing recipes. The current candidates are:

1. Mean-centered raw spectra
2. Savitzky-Golay smoothing + mean centering
3. First derivative + mean centering
4. Second derivative + mean centering
5. First derivative with window 21 / polynomial 3 + mean centering, when the spectrum is long enough
6. Polynomial baseline correction + mean centering
7. SNV + mean centering
8. Smoothing + SNV + mean centering
9. First derivative + SNV + mean centering
10. Baseline correction + SNV + mean centering

The exact recipe and parameters for every candidate are shown in the PCA Compare table.

## How PCA dimensionality is chosen

Each recipe is evaluated by reconstruction cross-validation using the existing PCA CV implementation. The table exposes:

- selected PCA component count
- Q²-X
- RMSECV-X
- variance represented by PC1 + PC2

The PCA model itself never uses metadata class labels.

## When known metadata is available

If a metadata field is confidently identified as a plausible known class, such as Brand, the app also calculates a **descriptive PC1/PC2 group-separation score** after PCA has already been fitted.

That score is a silhouette score on standardized PC1 and PC2 coordinates. It answers only:

> “How clearly do the known groups appear in this two-dimensional PCA view?”

It is not prediction accuracy and it is not used to fit PCA.

## Default recommendation rules

The recommendation is made by these visible rules:

1. Find the recipe with the best cross-validated reconstruction Q²-X.
2. Keep recipes within **0.08 Q²-X** of that best reconstruction result.
3. If a reliable known label exists and its best PC1/PC2 group-separation silhouette is at least **0.10**, choose the strongest group-separation view from that reconstruction-safe set.
4. **SNV must earn its place.** If SNV is selected for a known-group view, it must improve group separation by at least **0.05** over the best non-SNV alternative. If selection is based only on reconstruction, SNV must improve Q²-X by at least **0.03**.
5. Near ties prefer the simpler preprocessing recipe.

These thresholds are named constants in `transparent_guided.py` and can be changed directly.

## Why SNV is not a default

SNV removes each spectrum's own mean and rescales by its own standard deviation. That can be useful when multiplicative scatter or intensity differences are nuisance variation, but it can also remove genuine sample-to-sample information.

Therefore v5.2 does not assume SNV is beneficial. Every SNV recipe is compared against non-SNV alternatives, and the report explicitly summarizes whether SNV improved or reduced the relevant score for the loaded dataset.

## Predictive modeling remains separate

The PCA group-separation score is descriptive only. SVM / Random Forest predictive modeling still uses grouped nested cross-validation in `core.py`.

Any preprocessing decision that uses class labels for predictive modeling must happen inside validation. A visually separated PCA plot is not treated as proof that a classifier will generalize.

## Files

- `app.py` — small launcher that wires transparent guided selection into the UI
- `app_ui.py` — the full v5.2 PySide6 interface
- `transparent_guided.py` — explicit candidates, metrics, thresholds, and selection rules
- `core.py` — preprocessing, PCA, diagnostics, clustering, and predictive-model implementations
- `tests/test_transparent_guided.py` — tests including a dataset where SNV intentionally destroys useful group separation
