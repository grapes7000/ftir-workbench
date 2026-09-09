"""Visible busy state for synchronous desktop operations.

This paints a status indicator before computation; it does not move Qt plotting
or model work onto a background thread.
"""
from contextlib import contextmanager
from functools import wraps


@contextmanager
def busy_state(window, message):
    from PySide6.QtCore import QEventLoop, Qt
    from PySide6.QtWidgets import QApplication, QLabel

    depth = getattr(window, "_busy_depth", 0)
    window._busy_depth = depth + 1
    if depth:
        try:
            yield
        finally:
            window._busy_depth -= 1
        return

    indicator = QLabel(f"Working… {message}", window)
    indicator.setStyleSheet("font-weight: bold; padding: 4px 12px; color: #885500;")
    central = window.centralWidget()
    was_enabled = central.isEnabled() if central is not None else False
    window.statusBar().addPermanentWidget(indicator)
    window._busy_indicator = indicator
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        if central is not None:
            central.setEnabled(False)
        indicator.show()
        window.repaint()
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
        yield
    finally:
        window._busy_depth = 0
        if central is not None:
            central.setEnabled(was_enabled)
        QApplication.restoreOverrideCursor()
        window.statusBar().removeWidget(indicator)
        indicator.deleteLater()
        window._busy_indicator = None


def busy(message):
    def decorate(function):
        @wraps(function)
        def wrapped(window, *args, **kwargs):
            with busy_state(window, message):
                return function(window, *args, **kwargs)
        return wrapped
    return decorate
