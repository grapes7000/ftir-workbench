import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import pytest
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from busy_ui import busy_state


def test_busy_state_is_visible_and_restores_after_nested_failure():
    application = QApplication.instance() or QApplication([])
    window = QMainWindow()
    window.setCentralWidget(QWidget())
    window.show()
    with pytest.raises(RuntimeError):
        with busy_state(window, "Testing"):
            indicator = window._busy_indicator
            assert indicator.isVisible()
            assert "Testing" in indicator.text()
            assert not window.centralWidget().isEnabled()
            assert QApplication.overrideCursor() is not None
            with busy_state(window, "Nested"):
                assert window._busy_indicator is indicator
            assert window._busy_depth == 1
            raise RuntimeError("failure")
    assert window.centralWidget().isEnabled()
    assert window._busy_indicator is None
    assert window._busy_depth == 0
    assert QApplication.overrideCursor() is None
    window.centralWidget().setEnabled(False)
    with busy_state(window, "Already disabled"):
        pass
    assert not window.centralWidget().isEnabled()
    window.close()
