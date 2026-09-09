# FTIR Workbench v5.2

A desktop chemometrics and machine-learning workbench for FTIR, Raman, NIR, GC, and similar analytical datasets.

v5.2 is designed around one principle:

> **A user should be able to get a defensible first analysis without already knowing PCA, clustering, preprocessing theory, or cross-validation.**

Expert controls are still available, but the recommended workflow is **Guided Analysis**.

## Guided workflow

1. Load a CSV dataset.
2. The workbench identifies spectral columns and inspects metadata.
3. Metadata roles are inferred from both column names and observed values.
4. Click **Analyze dataset automatically**.
5. The workbench:
   - compares several sensible preprocessing recipes,
   - cross-validates PCA reconstruction,
   - chooses a parsimonious PCA component count,
   - calculates Hotelling T² and Q-residual diagnostics,
   - benchmarks K-Means, Ward agglomerative clustering, and DBSCAN,
   - creates a dendrogram,
   - reports the clustering result in plain English,
   - suggests prediction labels and independent grouping fields when metadata supports them.
6. Click any plotted sample to inspect its metadata, PCA scores, diagnostic values, cluster assignment, and spectral contribution profile.

## v5.2 highlights

### Automatic metadata understanding

The importer no longer depends on exact metadata headers such as `Brand`, `City`, or `SampleID`.

The schema detector uses:

- header aliases,
- value types,
- uniqueness,
- year/date patterns,
- categorical structure,
- identifier-like value patterns.

Examples that can be recognized:

- `manufacturer_name`, `mfr`, `maker` → brand/manufacturer
- `specimen_id`, `sample_code`, `bottle_id` → sample identifier
- `station_code`, `collection_site` → independent site/station
- `collection_year` → year

Every detected role is shown in **Data & Metadata QC** and can be overridden by the user.

### Automatic PCA settings

Guided Analysis compares multiple conservative recipes:

- mean-centered,
- SNV,
- Savitzky-Golay smoothing + SNV,
- first derivative + SNV,
- polynomial baseline correction + SNV.

For each candidate, the workbench performs PCA reconstruction cross-validation and evaluates:

- RMSEC-X,
- RMSECV-X,
- RMSECV-X variability,
- PRESS-X,
- Q²-X.

The recommended component count uses a one-standard-error-style rule so it prefers the smallest model whose cross-validation error remains near the minimum.

### PCA Compare

The **PCA Compare** workspace provides two ways to inspect preprocessing sensitivity:

- a gallery showing the top PCA views side by side,
- a slider that moves through each PCA candidate in detail.

The app intentionally does not overlay multiple PCA coordinate systems on one set of axes because PCA models produced by different preprocessing can rotate or flip, making direct overlays misleading.

### Click-to-inspect samples

Sample identity is retained through PCA, UMAP, diagnostics, and clustering views.

Click a data point to open the **Sample Inspector**, which displays:

- original metadata,
- inferred metadata roles,
- PCA scores,
- Hotelling T²,
- Q residual,
- diagnostic review flag,
- automatic cluster assignment.

The selected-sample contribution view shows spectral regions associated with:

- unmodeled Q-residual variation,
- T²-related PCA-space variation.

### Clustering analysis

The clustering analysis window benchmarks:

- K-Means over multiple K values,
- Ward agglomerative clustering over multiple K values,
- DBSCAN over data-derived neighborhood-distance thresholds.

It evaluates separation, coverage, and K-Means/Ward agreement and then reports a plain-English interpretation.

A weak cluster solution is explicitly described as weak rather than being presented as a discovery.

### Dendrogram

Ward hierarchical clustering is shown as a dendrogram using the recommended PCA score space.

When an inferred sample-identifier column is available, it is used for dendrogram labels automatically.

### Predictive modeling

The v5.1 predictive stack remains available:

- SVM,
- Random Forest,
- grouped eligibility screening,
- nested grouped cross-validation,
- automated model/preprocessing parameter search,
- confusion matrix,
- fold stability,
- optimization history.

v5.2 fixes an important preprocessing consistency issue: predictive sklearn pipelines can now receive the real spectral axis, so range selection and exclusions use actual wavenumbers rather than column positions.

The automatic model search also keeps preprocessing operations in a chemically sensible stage order instead of arbitrarily scrambling them.

## Installation

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

## Tests

```bash
python -m pytest
python -m compileall -q app.py core.py tests
```

## Interpretation boundaries

The workbench is intended to assist analytical exploration and model development.

- A PCA diagnostic flag means **review this sample**, not **this sample is wrong**.
- A cluster is a mathematical grouping, not proof of chemical identity.
- High classification accuracy is not meaningful if independent samples or groups leak across validation folds.
- Automated recommendations remain reviewable and overrideable in the expert controls.
