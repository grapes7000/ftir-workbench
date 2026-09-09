import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication
import app
import validation_audit


def test_selection_keeps_metadata_and_spectra_aligned_and_restores():
    application = QApplication.instance() or QApplication([])
    window = app.Main()
    meta = pd.DataFrame({'Brand': ['A', 'B', 'A', 'B'], 'Station': ['a', 'b', 'a', 'b']})
    spectra = pd.DataFrame(np.arange(24).reshape(4, 6))
    window.full_data = (meta, np.arange(6), spectra)
    window.apply_data_selection([1, 3])
    assert window.data[0].Brand.tolist() == ['B', 'B']
    np.testing.assert_array_equal(window.data[2], spectra.iloc[[1, 3]])
    assert window.opt is None
    window.apply_data_selection([0, 1, 2, 3])
    assert len(window.data[0]) == 4
    window.validation_mode.setCurrentIndex(1)
    table, y, groups = window.predictive_eligibility(meta, 'Brand')
    assert len(np.unique(groups)) == 4
    assert not table.eligible.any()
    window.close()


def test_exploratory_mode_is_explicit_and_grouped_mode_still_rejects():
    import pytest
    rng = np.random.default_rng(4)
    X = rng.normal(size=(24, 8))
    y = np.repeat(['A', 'B'], 12)
    groups = np.repeat(['station_a', 'station_b'], 12)
    with pytest.raises(ValueError):
        validation_audit.nested_optimize_audited(X, y, groups, [], trials=1)
    result = validation_audit.nested_optimize_audited(
        X, y, groups, [], trials=1, maxpcs=2, validation_mode='exploratory')
    assert result['validation_mode'] == 'exploratory'
    assert result['validation_summary'].startswith('EXPLORATORY')
    assert len(result['oof_predictions']) == len(X)
    assert (result['outer_assignments'] > 0).all()
