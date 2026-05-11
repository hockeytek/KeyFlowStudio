import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.node_graph.corridorkey_properties_panel import CorridorKeyPropertiesPanel
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

    def test_legacy_cloud_corridorkey_flag_only_marks_green_ready(self):
        legacy_models = {"corridorkey": True}

        self.assertTrue(CorridorKeyPropertiesPanel._cloud_corridorkey_weights_ready("green", legacy_models))
        self.assertFalse(CorridorKeyPropertiesPanel._cloud_corridorkey_weights_ready("blue", legacy_models))
        self.assertFalse(CorridorKeyPropertiesPanel._cloud_corridorkey_weights_ready("auto", legacy_models))

    def test_detailed_cloud_corridorkey_flags_require_both_for_auto(self):
        green_only = {"corridorkey": False, "corridorkey_green": True, "corridorkey_blue": False}
        both = {"corridorkey": True, "corridorkey_green": True, "corridorkey_blue": True}

        self.assertTrue(CorridorKeyPropertiesPanel._cloud_corridorkey_weights_ready("green", green_only))
        self.assertFalse(CorridorKeyPropertiesPanel._cloud_corridorkey_weights_ready("blue", green_only))
        self.assertFalse(CorridorKeyPropertiesPanel._cloud_corridorkey_weights_ready("auto", green_only))
        self.assertTrue(CorridorKeyPropertiesPanel._cloud_corridorkey_weights_ready("auto", both))
