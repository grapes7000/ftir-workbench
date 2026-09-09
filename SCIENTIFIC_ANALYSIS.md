# FTIR Workbench v5.2 — Scientific Analysis and Validation

This document describes what the automatic analysis does and, equally importantly, what it does **not** claim.

## Analysis order

The intended guided workflow is:

1. Load spectra and metadata.
2. Confirm the hard excluded solvent/interference region.
3. Remove hard-excluded wavenumbers from the spectral matrix.
4. Run sample-level spectral QC.
5. Compare an explicit list of scientifically reasonable preprocessing recipes.
6. Select PCA dimensionality from held-out reconstruction CV using a parsimonious rule.
7. Run T²/Q diagnostics and contribution analysis.
8. Assess preprocessing, PCA-loading, component-count, and cluster stability.
9. Screen metadata associations and categorical confounding.
10. Produce a transparent model-health summary and plain-English interpretation.
11. Optionally run supervised modeling using nested grouped CV.

Hard-excluded wavenumbers are removed **before** steps 4–11. Automatic methods cannot reintroduce them.

## No black-box meta-model

The application does not train a hidden model to decide whether an analysis is trustworthy. Automatic recommendations use named thresholds and returned metrics. The overall 0–100 health value is explicitly an engineering summary rather than an established statistical quantity. Its individual subscores are the primary evidence.

## Spectral QC

`scientific_analysis.spectral_qc` reports, but does not automatically delete, spectra with:

- missing values
- flat/nearly constant signal
- robustly extreme signal intensity
- unusually high high-frequency noise
- possible saturation/plateau behavior
- unusually large baseline slope variation
- poor replicate consistency when useful replicate/group metadata can be inferred

The saturation and artifact checks are screening heuristics. They are deliberately described as **possible** issues rather than instrument-specific diagnoses.

## PCA component selection

PCA dimensionality is based on held-out reconstruction, not explained variance alone. The PCA CV computes RMSEC-X, RMSECV-X, RMSECV standard deviation, PRESS-X, and Q²-X. The selected model uses a one-standard-error-style parsimonious rule so a nearly flat validation curve does not automatically choose the absolute minimum RMSECV at a larger component count.

## Preprocessing stability

The guided workflow compares explicit preprocessing candidates rather than arbitrary thousands of recipes. `preprocessing_stability` compares pairwise sample-distance structure across the candidate PCA score spaces. If the apparent structure only survives one narrow preprocessing choice, the report calls the result preprocessing-dependent.

SNV is optional and has an explicit guard against being selected without a visible benefit over non-SNV alternatives.

## PCA resampling stability

`pca_resampling_stability` repeatedly fits the selected preprocessing/PCA model to deterministic 80% subsamples and reports:

- loading correlation for each retained PC
- mean and SD of explained variance under resampling
- a small resampled distribution of recommended PCA component counts

The leading loading vectors are sign-invariant, so stability uses absolute loading correlation.

## Loading regions

Large adjacent loading coefficients are grouped into spectral regions. The report may state that variation near a wavenumber is influential. It does **not** assign a compound or claim chemical identity from a loading alone.

## Outlier diagnostics

Hotelling T² / score-distance and Q/SPE reconstruction error are used as review diagnostics. Samples are flagged for inspection rather than automatically removed. Clickable sample inspection remains the preferred way to combine diagnostics with original metadata and spectra.

## Clustering

K-Means, Ward agglomerative clustering, and DBSCAN are compared. The analysis reports silhouette, conventional-method agreement, and a perturbation/component-count stability assessment using adjusted Rand agreement.

A cluster is never described as a chemical class merely because an algorithm created it.

## Metadata association and confounding

Categorical metadata are screened against PCA scores using effect-size style eta-squared summaries with deterministic label permutations for a descriptive p-value. Numeric variables use Spearman association. Cluster membership can be compared with categorical metadata using Cramér's V.

Categorical metadata fields are also compared with one another. Strong Cramér's V values trigger confounding warnings, especially when they involve the suggested prediction label. For example, a Brand/City confound means PCA separation cannot be uniquely attributed to Brand.

These screens are exploratory and do not substitute for a designed causal study.

## Grouped and nested validation

`validation_audit.nested_optimize_audited` uses:

- stratified grouped **outer** folds for final generalization estimates
- stratified grouped **inner** folds for preprocessing/model/hyperparameter selection
- training-only fitting of imputation, centering/scaling, PCA/PLS components, preprocessing state, and classifiers
- real physical wavenumber axes in each preprocessing transformer
- inner fold counts bounded by the smallest independent-group support in any class

Inside each outer-training set, the inner search compares the explicit FTIR preprocessing candidates plus the current expert recipe when distinct. It simultaneously evaluates SVM, Random Forest, and PLS-DA settings. The outer test fold does not participate in this selection.

The returned audit includes:

- train/test sample counts by fold
- train/test group counts
- explicit zero-overlap group audit
- training balanced accuracy
- selected inner-CV balanced accuracy and SD
- outer held-out balanced accuracy and macro F1
- train→outer and inner→outer gaps
- sample-aligned out-of-fold predictions
- outer-fold assignment per sample
- sensitivity/recall and specificity by class
- worst/best fold and fold-to-fold variability
- selected model family and preprocessing recipe for each outer fold

Training accuracy is not used as the primary predictive claim.

## Overfitting

The supervised audit labels the training-to-outer balanced-accuracy gap as low, moderate, or high. Large training performance paired with weaker grouped outer-CV performance is explicitly reported as probable overfitting.

## Permutation testing

`validation_audit.permutation_test_nested` provides an optional shuffled-label null test that reruns the nested grouped validation strategy. Permutation models are not optimized more aggressively than the observed model.

Because this is computationally expensive, it is an optional validation tool rather than an automatic step on every dataset. The Model Health tab exposes a 10-permutation button.

## Exploratory versus predictive claims

The workbench must keep these statements separate:

- **Exploratory:** spectral structure is associated with Brand in this dataset.
- **Predictive:** spectra predict Brand for genuinely unseen independent groups with held-out performance above chance.

PCA separation, silhouette, or metadata association cannot establish the second statement.

## Supervised model families

SVM, Random Forest, and PLS-DA are available to the nested inner search. They are evaluated under the same grouped outer-validation procedure, and family selection is made inside each outer training fold rather than using the outer test data.

The current Model Review comparison plot summarizes inner optimization trials. The final predictive claim remains the combined outer held-out prediction performance of the complete selection procedure, not the prettiest PCA plot or the highest training-score model.
