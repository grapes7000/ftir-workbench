import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from app_ui import GroupSelector, Plot


def test_class_legend_and_picking_preserve_sample_identity():
    application = QApplication.instance() or QApplication([])
    plot = Plot()
    ax = plot.fig.add_subplot(projection="3d")
    picked = []
    plot.set_pick_callback(picked.append)
    artists = plot.scatter_samples(
        ax, [1, 2, 3], [4, 5, 6], [7, 8, 9],
        labels=["Brand B", "Brand A", "Brand B"], indices=[10, 20, 30],
    )
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["Brand A", "Brand B"]
    assert artists[1]._sample_indices.tolist() == [10, 30]
    plot._picked(SimpleNamespace(artist=artists[1], ind=[1]))
    assert picked == [30]
    assert not np.array_equal(artists[0].get_paths()[0].vertices, artists[1].get_paths()[0].vertices)
    plot.draw()
    plot.close()


def test_group_dropdown_supports_multiple_columns_and_clears_on_reload():
    application = QApplication.instance() or QApplication([])
    selector = GroupSelector()
    selector.set_columns(["Brand", "Site", "Batch"], ["Site"])
    selector.menu().actions()[2].setChecked(True)
    assert selector.selected_columns() == ["Site", "Batch"]
    selector.set_columns(["Supplier"], [])
    assert selector.selected_columns() == []
    assert [action.text() for action in selector.menu().actions()] == ["Supplier"]
    selector.close()
