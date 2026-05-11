"""Load Media node properties panel."""

from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy, QWidget

from app.node_graph.properties_style import (
    apply_properties_reference_style,
    configure_form_layout,
    configure_inline_layout,
    configure_properties_panel,
)


class LoadMediaPropertiesPanel(QWidget):
    """Compact Load Media controls used inside node properties."""

    def __init__(self, translate: Callable[[str], str], parent=None) -> None:
        super().__init__(parent)
        self._tr = translate

        configure_properties_panel(self)

        self.form = QFormLayout(self)
        configure_form_layout(self.form)

        self.media_type_combo = QComboBox(self)
        self.path_edit = QLineEdit(self)
        self.browse_button = QPushButton(self)

        self.path_row = QWidget(self)
        self.path_layout = QHBoxLayout(self.path_row)
        configure_inline_layout(self.path_layout)
        self.path_layout.addWidget(self.path_edit, 1)
        self.path_layout.addWidget(self.browse_button)

        self.info_label = QLabel(self)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #8ca0ba; font-size: 11px;")

        self.form.addRow(self.media_type_combo)
        self.form.addRow("", self.path_row)
        self.form.addRow("", self.info_label)

        apply_properties_reference_style(self)

        self.retranslate_ui()

    def set_translator(self, translate: Callable[[str], str]) -> None:
        self._tr = translate
        self.retranslate_ui()

    def _set_form_label_text(self, row: int, text: str) -> None:
        item = self.form.itemAt(row, QFormLayout.ItemRole.LabelRole)
        if item is None or item.widget() is None:
            return
        item.widget().setText(text)

    def retranslate_ui(self) -> None:
        current_index = self.media_type_combo.currentIndex()
        self._set_form_label_text(1, self._tr("node_props_path"))
        self._set_form_label_text(2, self._tr("node_props_media_info"))

        self.media_type_combo.blockSignals(True)
        self.media_type_combo.clear()
        self.media_type_combo.addItems([
            self._tr("node_props_media_video"),
            self._tr("node_props_media_image"),
        ])
        if current_index >= 0:
            self.media_type_combo.setCurrentIndex(min(current_index, self.media_type_combo.count() - 1))
        self.media_type_combo.blockSignals(False)
        self.browse_button.setText(self._tr("node_props_browse"))

    def load_from_properties(self, props: dict) -> None:
        self.media_type_combo.setCurrentIndex(0 if str(props.get("media_type", "video")) == "video" else 1)
        self.path_edit.setText(str(props.get("path", "")))

    def write_to_properties(self, props: dict) -> None:
        props["media_type"] = "video" if self.media_type_combo.currentIndex() == 0 else "image"
        props["path"] = self.path_edit.text().strip()

    def set_info_text(self, text: str) -> None:
        self.info_label.setText(text)