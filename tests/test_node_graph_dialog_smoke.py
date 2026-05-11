import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.node_graph_dialog import NodeGraphDialog


class NodeGraphDialogSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_dialog_initializes_after_corridorkey_ui_changes(self):
        dialog = NodeGraphDialog(lambda key: key)
        try:
            self.assertIsNotNone(dialog.corridorkey_props_panel)
            self.assertFalse(hasattr(dialog.corridorkey_props_panel, "output_mode_combo"))
        finally:
            dialog.close()
            dialog.deleteLater()
