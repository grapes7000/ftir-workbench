from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression


class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    """Minimal sklearn-compatible PLS-DA estimator.

    This estimator is intentionally not inserted into automatic model search yet.
    It exists so PLS-DA can be compared on the same outer partitions as SVM and
    Random Forest in a later model-comparison layer without changing validation
    boundaries or preprocessing semantics.
    """

    def __init__(self, n_components=2, scale=False):
        self.n_components = n_components
        self.scale = scale

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("PLS-DA requires at least two classes.")
        n_components = max(1, min(int(self.n_components), X.shape[1], len(X) - 1))
        target = np.eye(len(self.classes_), dtype=float)[encoded]
        self.model_ = PLSRegression(n_components=n_components, scale=bool(self.scale))
        self.model_.fit(X, target)
        self.n_components_ = n_components
        return self

    def decision_function(self, X):
        pred = np.asarray(self.model_.predict(np.asarray(X, dtype=float)))
        if pred.ndim == 1:
            pred = pred[:, None]
        return pred

    def predict(self, X):
        scores = self.decision_function(X)
        index = np.argmax(scores, axis=1)
        return self.classes_[index]
