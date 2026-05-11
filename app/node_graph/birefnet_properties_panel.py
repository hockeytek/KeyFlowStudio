"""BiRefNet node properties panel."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QLabel, QMessageBox, QProgressBar,
    QPushButton, QSizePolicy, QSpinBox, QStyledItemDelegate, QStyle,
    QVBoxLayout, QWidget,
)

from app.node_graph.node_panel_mixin import NodePanelMixin
from app.services.birefnet_service import BiRefNetService


READY_ROLE = Qt.ItemDataRole.UserRole + 1


class BiRefNetPresetDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index) -> None:
        ready = bool(index.data(READY_ROLE))
        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")

        style = option.widget.style() if option.widget is not None else QApplication.style()
        opt = option
        self.initStyleOption(opt, index)
        opt.text = text
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, option.widget)

        if not ready:
            return

        painter.save()
        color = QColor("#39d98a")
        if option.state & QStyle.StateFlag.State_Selected:
            color = QColor("#dfffea")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)

        dot_diameter = 8
        right_padding = 14
        x = option.rect.right() - dot_diameter - right_padding
        y = option.rect.y() + (option.rect.height() - dot_diameter) // 2
        painter.drawEllipse(x, y, dot_diameter, dot_diameter)
        painter.restore()


class BiRefNetPropertiesPanel(QWidget, NodePanelMixin):
    """Compact BiRefNet controls used inside node properties."""

    def __init__(self, translate: Callable[[str], str], parent=None) -> None:
        super().__init__(parent)
        self._tr = translate
        self._init_panel_layout()

        self.setObjectName("birefnetPropertiesPanel")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.setFixedWidth(self._PANEL_WIDTH)
        self.setStyleSheet(
            "QWidget#birefnetPropertiesPanel { background: #10151d; }"
            "QLabel { color: #d8dee7; }"
        )

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(6, 6, 6, 6)
        self.root.setSpacing(8)

        # ── Usage preset selector ─────────────────────────────────────────────
        self.usage_label = QLabel(self)
        self.usage_combo = QComboBox(self)
        self.usage_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.usage_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.usage_combo.setMinimumWidth(220)
        self.usage_combo.setItemDelegate(BiRefNetPresetDelegate(self.usage_combo))
        self._apply_reference_combo_style(self.usage_combo)

        # ── Download ──────────────────────────────────────────────────────────
        self.download_button = QPushButton(self)
        self.download_button.clicked.connect(self._download_weights)

        self.download_progress = QProgressBar(self)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_progress.setTextVisible(False)
        self.download_progress.setFixedHeight(6)
        self.download_progress.setVisible(False)

        self.PRESETS = [
            "General",
            "General-dynamic",
            "General-HR",
            "General-Lite",
            "General-Lite-2K",
            "General-reso_512",
            "Matting",
            "Matting-dynamic",
            "Matting-HR",
            "Matting-Lite",
            "Portrait",
            "DIS5K",
            "HRSOD",
            "COD",
        ]

        # ── Section: Post-process ─────────────────────────────────────────────
        self.post_section, self.post_section_title, self.post_form = self._create_section(expanded=True)

        self.half_precision_label = QLabel(self)
        self.half_precision_check = QCheckBox(self)
        self._apply_reference_checkbox_style(self.half_precision_check)

        self.dilate_radius_label = QLabel(self)
        self.dilate_radius_spin = QSpinBox(self)
        self.dilate_radius_spin.setRange(0, 50)
        self.dilate_field, self.dilate_slider = self._make_int_slider_field(
            self.dilate_radius_spin, min_value=0, max_value=50
        )

        self.erode_radius_label = QLabel(self)
        self.erode_radius_spin = QSpinBox(self)
        self.erode_radius_spin.setRange(0, 50)
        self.erode_field, self.erode_slider = self._make_int_slider_field(
            self.erode_radius_spin, min_value=0, max_value=50
        )

        self._add_checkbox_row(self.post_form, self.half_precision_check, self.half_precision_label)
        self._add_form_row(self.post_form, self.dilate_radius_label, self.dilate_field)
        self._add_form_row(self.post_form, self.erode_radius_label, self.erode_field)

        # ── Assemble root ─────────────────────────────────────────────────────
        self.root.addWidget(self.usage_label)
        self.root.addWidget(self.usage_combo)
        self.root.addWidget(self.download_button)
        self.root.addWidget(self.download_progress)
        self.root.addWidget(self.post_section)

        self.usage_combo.currentIndexChanged.connect(self._refresh_download_button_state)

        self.retranslate_ui()

    def set_translator(self, translate: Callable[[str], str]) -> None:
        self._tr = translate
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.post_section_title.setText(self._tr("birefnet_section_postprocess"))

        self.usage_label.setText(self._tr("birefnet_usage"))
        self.half_precision_label.setText(self._tr("birefnet_half_precision"))
        self.dilate_radius_label.setText(self._tr("birefnet_dilate_radius"))
        self.erode_radius_label.setText(self._tr("birefnet_erode_radius"))

        self.usage_label.setToolTip(self._tr("birefnet_usage_tooltip"))
        self.half_precision_label.setToolTip(self._tr("birefnet_half_precision_tooltip"))
        self.dilate_radius_label.setToolTip(self._tr("birefnet_dilate_radius_tooltip"))
        self.erode_radius_label.setToolTip(self._tr("birefnet_erode_radius_tooltip"))

        current_preset = self.usage_combo.currentData()
        self._rebuild_preset_items(current_preset if current_preset else "General")
        self._refresh_download_button_state()

    def _refresh_download_button_state(self) -> None:
        if self._is_cloud_mode():
            if getattr(self, "_cloud_weights_ready", False):
                self.download_button.setText(self._tr("birefnet_download_button_ready_cloud"))
                self.download_button.setEnabled(True)
                self.download_button.setToolTip(self._tr("birefnet_download_button_ready_tooltip"))
            else:
                self.download_button.setText(self._tr("birefnet_download_button_missing"))
                self.download_button.setEnabled(True)
                self.download_button.setToolTip(self._tr("birefnet_download_button_missing_tooltip"))
            return

        usage = str(self.usage_combo.currentData() or "General")
        status = BiRefNetService.get_weight_status(usage)
        state = status.get("state", "missing")

        if state == "ready":
            self.download_button.setText(self._tr("birefnet_download_button_ready"))
            self.download_button.setEnabled(True)
            self.download_button.setToolTip(self._tr("birefnet_download_button_ready_tooltip"))
        elif state == "external":
            self.download_button.setText(self._tr("birefnet_download_button_external"))
            self.download_button.setEnabled(False)
            self.download_button.setToolTip(self._tr("birefnet_download_button_external_tooltip"))
        else:
            self.download_button.setText(self._tr("birefnet_download_button_missing"))
            self.download_button.setEnabled(True)
            self.download_button.setToolTip(self._tr("birefnet_download_button_missing_tooltip"))

    def _on_download_progress(self, percent: int, _message: str) -> None:
        self.download_progress.setVisible(True)
        self.download_progress.setValue(max(0, min(100, int(percent))))
        QApplication.processEvents()

    def _download_weights(self) -> None:
        if self._is_cloud_mode():
            usage = str(self.usage_combo.currentData() or "General")
            self._start_cloud_download("birefnet", preset=usage)
            return

        usage = str(self.usage_combo.currentData() or "General")
        service = BiRefNetService()
        self.download_button.setEnabled(False)
        self.download_progress.setVisible(True)
        self.download_progress.setValue(0)
        service.set_callbacks(self._on_download_progress, self._tr)

        try:
            snapshot_path = service.ensure_weights_available(usage)
            self._rebuild_preset_items(usage)
            self._refresh_download_button_state()
            self.download_progress.setValue(100)

            if snapshot_path is None:
                QMessageBox.information(
                    self,
                    self._tr("info_title"),
                    self._tr("birefnet_weights_external_message"),
                )
            else:
                QMessageBox.information(
                    self,
                    self._tr("info_title"),
                    self._tr("birefnet_weights_downloaded_ui").format(name=usage),
                )
        except Exception as exc:
            QMessageBox.critical(
                self,
                self._tr("inference_error_title"),
                self._tr("birefnet_weights_download_failed").format(error=str(exc)),
            )
            self.download_progress.setValue(0)
            self._rebuild_preset_items(usage)
            self._refresh_download_button_state()
        finally:
            service.set_callbacks(None, None)
            if self.download_button.isEnabled():
                self.download_button.setEnabled(True)
            if self.download_progress.value() >= 100:
                self.download_progress.setVisible(False)
            elif self.download_progress.value() == 0:
                self.download_progress.setVisible(False)

    def _rebuild_preset_items(self, selected_preset: str | None = None) -> None:
        self.usage_combo.blockSignals(True)
        self.usage_combo.clear()
        for preset in self.PRESETS:
            status = BiRefNetService.get_weight_status(preset)
            self.usage_combo.addItem(preset, preset)
            index = self.usage_combo.count() - 1
            self.usage_combo.setItemData(index, status["state"] == "ready", READY_ROLE)

        target_preset = selected_preset or "General"
        index = self.usage_combo.findData(target_preset)
        if index >= 0:
            self.usage_combo.setCurrentIndex(index)
        self.usage_combo.blockSignals(False)

    def load_from_properties(self, props: dict) -> None:
        """Load properties from node data dict."""
        usage = str(props.get("usage", "General"))
        self._rebuild_preset_items(usage)
        index = self.usage_combo.findData(usage)
        if index >= 0:
            self.usage_combo.setCurrentIndex(index)
        else:
            # Fallback to "General" if not found
            self.usage_combo.setCurrentIndex(self.usage_combo.findData("General"))

        half_precision = bool(props.get("half_precision", False))
        self.half_precision_check.setChecked(half_precision)
        self.dilate_radius_spin.setValue(int(props.get("dilate_radius", 0)))
        self.erode_radius_spin.setValue(int(props.get("erode_radius", 0)))
        self._refresh_download_button_state()

    def write_to_properties(self, props: dict) -> None:
        """Write properties to node data dict."""
        props["usage"] = self.usage_combo.currentData()
        props["half_precision"] = self.half_precision_check.isChecked()
        props["dilate_radius"] = int(self.dilate_radius_spin.value())
        props["erode_radius"] = int(self.erode_radius_spin.value())
