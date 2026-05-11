"""SAM2 node properties mini-panel (internal key: sam2)."""

from __future__ import annotations

import os
from typing import Callable

from PySide6.QtCore import QSize, Qt, QSignalBlocker
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressBar, QComboBox, QSizePolicy
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.node_graph.properties_style import apply_properties_reference_style, configure_inline_layout, configure_properties_panel
from app.services.sam2_service import Sam2Service
from app.utils import get_device


READY_ROLE = Qt.ItemDataRole.UserRole + 1


class SamPropertiesPanel(QWidget):
    """Compact SAM controls panel used inside node properties."""

    MODELS = ["vit_h", "vit_b", "vit_l"]
    _MODEL_SHORT_LABELS = {"vit_h": "SAM2 Large", "vit_l": "SAM2 Base+", "vit_b": "SAM2 Small"}

    def __init__(self, translate: Callable[[str], str], parent=None) -> None:
        super().__init__(parent)
        self._tr = translate
        self._is_cpu_cached: bool | None = None

        configure_properties_panel(self)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # ── Status ───────────────────────────────────────────────────
        self.status_label = QLabel(self)
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 12px; color: #8ca0ba; font-weight: 600;")
        self.status_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        root.addWidget(self.status_label)

        self.prompts_label = QLabel(self)
        self.prompts_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prompts_label.setStyleSheet("color: #666666; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        root.addWidget(self.prompts_label)

        self.points_row = QWidget(self)
        self.points_layout = QHBoxLayout(self.points_row)
        configure_inline_layout(self.points_layout)

        self.btn_positive = QPushButton(self.points_row)
        self.btn_positive.setFixedWidth(28)
        self.btn_positive.setCheckable(True)
        self.btn_positive.setAutoExclusive(True)
        self.points_layout.addWidget(self.btn_positive)

        self.btn_negative = QPushButton(self.points_row)
        self.btn_negative.setFixedWidth(28)
        self.btn_negative.setCheckable(True)
        self.btn_negative.setAutoExclusive(True)
        self.points_layout.addWidget(self.btn_negative)

        self.btn_clear = QPushButton(self.points_row)
        self.btn_clear.setMinimumSize(64, 26)
        self.points_layout.addWidget(self.btn_clear)

        self.btn_live_sam2 = QPushButton(self.points_row)
        self.btn_live_sam2.setObjectName("prop_sam2_live_button")
        self.btn_live_sam2.setMinimumSize(82, 26)
        self.btn_live_sam2.setCheckable(True)
        self.points_layout.addWidget(self.btn_live_sam2)

        root.addWidget(self.points_row)

        self.masks_label = QLabel(self)
        self.masks_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.masks_label.setStyleSheet("color: #666666; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        root.addWidget(self.masks_label)

        self.btn_generate_mask = QPushButton(self)
        self.btn_generate_mask.setObjectName("prop_sam_generate_mask")
        self.btn_generate_mask.setMinimumSize(170, 28)
        root.addWidget(self.btn_generate_mask)

        self.mask_actions_row = QWidget(self)
        self.mask_actions_layout = QHBoxLayout(self.mask_actions_row)
        configure_inline_layout(self.mask_actions_layout)

        self.btn_add_mask = QPushButton(self.mask_actions_row)
        self.btn_add_mask.setObjectName("prop_sam_add_mask")
        self.btn_add_mask.setMinimumSize(50, 28)
        self.mask_actions_layout.addWidget(self.btn_add_mask)

        self.btn_remove_mask = QPushButton(self.mask_actions_row)
        self.btn_remove_mask.setObjectName("prop_sam_remove_mask")
        self.btn_remove_mask.setMinimumSize(50, 28)
        self.mask_actions_layout.addWidget(self.btn_remove_mask)

        root.addWidget(self.mask_actions_row)

        self.load_mask_row = QWidget(self)
        self.load_mask_layout = QHBoxLayout(self.load_mask_row)
        configure_inline_layout(self.load_mask_layout)

        self.btn_load_mask = QPushButton(self.load_mask_row)
        self.btn_load_mask.setObjectName("prop_sam_load_mask")
        self.btn_load_mask.setMinimumSize(50, 28)
        self.load_mask_layout.addWidget(self.btn_load_mask)

        root.addWidget(self.load_mask_row)

        self.masks_list = QListWidget(self)
        self.masks_list.setMinimumSize(170, 40)
        self.masks_list.setMaximumSize(16777215, 80)
        root.addWidget(self.masks_list)

        # ── SAM2 propagation actions ───────────────────────────────
        self.sam2_actions_label = QLabel(self)
        self.sam2_actions_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sam2_actions_label.setStyleSheet(
            "color: #666666; font-size: 10px; font-weight: bold; letter-spacing: 1px;"
        )
        root.addWidget(self.sam2_actions_label)

        self.sam2_actions_row = QWidget(self)
        self.sam2_actions_layout = QHBoxLayout(self.sam2_actions_row)
        configure_inline_layout(self.sam2_actions_layout)

        self.btn_sam2_propagate_backward = QPushButton(self.sam2_actions_row)
        self.btn_sam2_propagate_backward.setObjectName("prop_sam2_backward")
        self.btn_sam2_propagate_backward.setFixedWidth(28)
        self.sam2_actions_layout.addWidget(self.btn_sam2_propagate_backward)

        self.btn_sam2_propagate_forward = QPushButton(self.sam2_actions_row)
        self.btn_sam2_propagate_forward.setObjectName("prop_sam2_forward")
        self.btn_sam2_propagate_forward.setFixedWidth(28)
        self.sam2_actions_layout.addWidget(self.btn_sam2_propagate_forward)

        self.btn_sam2_reprompt = QPushButton(self.sam2_actions_row)
        self.btn_sam2_reprompt.setObjectName("prop_sam2_reprompt")
        self.btn_sam2_reprompt.setMinimumSize(64, 26)
        self.sam2_actions_layout.addWidget(self.btn_sam2_reprompt)

        self.btn_sam2_reset_session = QPushButton(self.sam2_actions_row)
        self.btn_sam2_reset_session.setObjectName("prop_sam2_reset")
        self.btn_sam2_reset_session.setMinimumSize(64, 26)
        self.sam2_actions_layout.addWidget(self.btn_sam2_reset_session)

        root.addWidget(self.sam2_actions_row)

        # ── Model selection (bottom) — label | combo | download button ──
        self.model_row = QWidget(self)
        self.model_row_layout = QHBoxLayout(self.model_row)
        configure_inline_layout(self.model_row_layout)

        self.model_label = QLabel(self.model_row)
        self.model_label.setStyleSheet("color: #8ca0ba; font-size: 11px; font-weight: 600;")
        self.model_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.model_row_layout.addWidget(self.model_label)

        self.model_combo = QComboBox(self.model_row)
        self.model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.model_combo.setIconSize(QSize(10, 10))
        self.model_row_layout.addWidget(self.model_combo)

        self.download_button = QPushButton(self.model_row)
        self.download_button.setFixedWidth(120)
        self.download_button.clicked.connect(self._on_weights_button_clicked)
        self.model_row_layout.addWidget(self.download_button)

        root.addWidget(self.model_row)

        self.download_progress = QProgressBar(self)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_progress.setTextVisible(False)
        self.download_progress.setFixedHeight(6)
        self.download_progress.setVisible(False)
        root.addWidget(self.download_progress)

        self.model_combo.currentIndexChanged.connect(self._refresh_download_button_state)
        self.btn_live_sam2.toggled.connect(lambda _checked: self._on_backend_or_live_changed())

        apply_properties_reference_style(self)
        self._apply_styles()
        self._update_status_height()
        self.retranslate_ui()
        self._loading_properties = False

    # ── Model combo helpers ──────────────────────────────────────────

    _MODEL_TOOLTIP_KEYS = {
        "vit_h": "sam2_model_vit_h_tooltip",
        "vit_l": "sam2_model_vit_l_tooltip",
        "vit_b": "sam2_model_vit_b_tooltip",
    }

    def _rebuild_model_items(self, selected: str | None = None) -> None:
        service = Sam2Service
        label_map = self._MODEL_SHORT_LABELS
        tooltip_map = self._MODEL_TOOLTIP_KEYS
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for mt in self.MODELS:
            short_label = label_map.get(mt, mt)
            status = service.get_weight_status(mt)
            tooltip_key = tooltip_map.get(mt, "")
            tooltip = self._tr(tooltip_key) if tooltip_key else ""
            ready = status["state"] == "ready"
            self.model_combo.addItem(self._model_status_icon(ready), short_label, mt)
            idx = self.model_combo.count() - 1
            self.model_combo.setItemData(idx, ready, READY_ROLE)
            self.model_combo.setItemData(idx, tooltip, Qt.ItemDataRole.ToolTipRole)
        target = selected or "vit_h"
        idx = self.model_combo.findData(target)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.model_combo.blockSignals(False)


    def _refresh_download_button_state(self) -> None:
        mt = str(self.model_combo.currentData() or "vit_h")
        status = Sam2Service.get_weight_status(mt)
        if status["state"] == "ready":
            self.download_button.setText(self._tr("sam2_download_button_ready"))
            self.download_button.setEnabled(True)
            self.download_button.setToolTip(self._tr("sam2_download_button_ready_tooltip"))
        else:
            self.download_button.setText(self._tr("sam2_download_button_missing"))
            self.download_button.setEnabled(True)
            self.download_button.setToolTip(self._tr("sam2_download_button_missing_tooltip"))

    @staticmethod
    def _model_status_icon(ready: bool) -> QIcon:
        color = "#39d98a" if ready else "#7b8492"
        border = "#74f0b0" if ready else "#8b95a4"
        pixmap = QPixmap(10, 10)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor(border))
        painter.setBrush(QColor(color))
        painter.drawEllipse(1, 1, 8, 8)
        painter.end()
        return QIcon(pixmap)

    def _on_download_progress(self, percent: int, _message: str) -> None:
        self.download_progress.setVisible(True)
        self.download_progress.setValue(max(0, min(100, int(percent))))
        QApplication.processEvents()

    def _on_weights_button_clicked(self) -> None:
        mt = str(self.model_combo.currentData() or "vit_h")
        status = Sam2Service.get_weight_status(mt)
        if status["state"] == "ready":
            self._rebuild_model_items(mt)
            self._refresh_download_button_state()
            self.status_label.setText(self._tr("sam_weights_verified_status"))
            return

        self._download_weights()

    def _download_weights(self) -> None:
        mt = str(self.model_combo.currentData() or "vit_h")
        name = Sam2Service.SAM2_LABELS.get(mt, mt)
        self.download_button.setEnabled(False)
        self.download_progress.setVisible(True)
        self.download_progress.setValue(0)
        try:
            Sam2Service.download_checkpoint_for(mt, self._on_download_progress)
            self._rebuild_model_items(mt)
            self._refresh_download_button_state()
            self.download_progress.setValue(100)
            # Settle layout before showing modal so closing it doesn't cause a jump
            self.download_progress.setVisible(False)
            self.download_button.setEnabled(True)
            QMessageBox.information(
                self,
                self._tr("info_title"),
                self._tr("sam_weights_downloaded_ui").format(name=name),
            )
        except Exception as exc:
            self.download_progress.setVisible(False)
            self.download_button.setEnabled(True)
            QMessageBox.critical(
                self,
                self._tr("inference_error_title"),
                self._tr("sam_weights_download_failed").format(error=str(exc)),
            )
            self._rebuild_model_items(mt)
            self._refresh_download_button_state()

    # ── Internal helpers ─────────────────────────────────────────────

    def _update_status_height(self) -> None:
        metrics = self.status_label.fontMetrics()
        vertical_margins = self.status_label.contentsMargins().top() + self.status_label.contentsMargins().bottom()
        status_height = metrics.lineSpacing() * 2 + vertical_margins + 4
        self.status_label.setFixedHeight(status_height)

    def _apply_styles(self) -> None:
        compact_point_style = (
            "background-color: #1b212b;"
            "border: 1px solid #2a3444;"
            "color: #eef3fb;"
            "padding: 0px;"
            "border-radius: 8px;"
            "font-weight: 700;"
            "font-size: 16px;"
            "min-height: 26px;"
        )
        compact_point_hover = "background-color: #243145; border: 1px solid #43c7ff;"
        compact_point_checked = "background-color: #0f3c57; border: 1px solid #34c2ff; color: #f7fbff;"
        compact_point_disabled = "background-color: #14181f; border: 1px solid #1a2030; color: #4a5568;"
        for button in [self.btn_positive, self.btn_negative]:
            button.setStyleSheet(
                f"QPushButton {{{compact_point_style}}}"
                f"QPushButton:hover {{{compact_point_hover}}}"
                f"QPushButton:checked {{{compact_point_checked}}}"
                f"QPushButton:disabled {{{compact_point_disabled}}}"
            )

        self.btn_live_sam2.setStyleSheet(
            "QPushButton#prop_sam2_live_button {"
            " background-color: #3d1e00;"
            " border: 1px solid #b85200;"
            " color: #ffc87a;"
            " padding: 0 10px;"
            " min-height: 26px;"
            " border-radius: 9px;"
            " font-weight: 700;"
            " font-size: 12px;"
            " letter-spacing: 0.5px;"
            "}"
            "QPushButton#prop_sam2_live_button:hover {"
            " background-color: #5a2e00;"
            " border: 1px solid #e87020;"
            " color: #ffe4aa;"
            "}"
            "QPushButton#prop_sam2_live_button:checked {"
            " background-color: #c45200;"
            " border: 2px solid #ff8c30;"
            " color: #ffffff;"
            "}"
            "QPushButton#prop_sam2_live_button:checked:hover {"
            " background-color: #de6510;"
            " border: 2px solid #ffaa55;"
            " color: #ffffff;"
            "}"
        )
        self.btn_generate_mask.setStyleSheet(
            "QPushButton#prop_sam_generate_mask {"
            " background-color: #0f5a33;"
            " border: 1px solid #1f8f58;"
            " color: #ffffff;"
            " padding: 0 12px;"
            " min-height: 26px;"
            " border-radius: 9px;"
            " font-weight: 600;"
            "}"
            "QPushButton#prop_sam_generate_mask:hover {"
            " background-color: #1f8f58;"
            " border: 1px solid #2fb878;"
            "}"
            "QPushButton#prop_sam_generate_mask:pressed {"
            " background-color: #0a3a1f;"
            " border: 1px solid #147d42;"
            "}"
            "QPushButton#prop_sam_generate_mask:disabled {"
            " background-color: #16241c;"
            " border: 1px solid #2a3a31;"
            " color: #6f8477;"
            "}"
        )

        action_style = (
            "background-color: #0052a3;"
            "border: 1px solid #0066cc;"
            "color: #ffffff;"
            "padding: 0 10px;"
            "min-height: 26px;"
            "border-radius: 8px;"
            "font-weight: 600;"
            "font-size: 12px;"
        )
        action_hover = "background-color: #0066cc; border: 1px solid #0080ff;"
        action_pressed = "background-color: #003d7a; border: 1px solid #0052a3;"
        for button in [self.btn_add_mask, self.btn_remove_mask, self.btn_load_mask]:
            button.setStyleSheet(
                f"QPushButton#{button.objectName()} {{{action_style}}}"
                f"QPushButton#{button.objectName()}:hover {{{action_hover}}}"
                f"QPushButton#{button.objectName()}:pressed {{{action_pressed}}}"
            )

        sam2_action_style = (
            "background-color: #2d4a1a;"
            "border: 1px solid #5e9c2f;"
            "color: #f0ffe2;"
            "padding: 0 10px;"
            "min-height: 26px;"
            "border-radius: 8px;"
            "font-weight: 600;"
            "font-size: 12px;"
        )
        sam2_action_hover = "background-color: #3d6524; border: 1px solid #78bf3c;"
        sam2_action_pressed = "background-color: #244013; border: 1px solid #4d8227;"
        sam2_action_disabled = "background-color: #1a2316; border: 1px solid #2f3a2a; color: #62725c;"
        for button in [
            self.btn_sam2_propagate_forward,
            self.btn_sam2_propagate_backward,
            self.btn_sam2_reprompt,
            self.btn_sam2_reset_session,
        ]:
            button.setStyleSheet(
                f"QPushButton#{button.objectName()} {{{sam2_action_style}}}"
                f"QPushButton#{button.objectName()}:hover {{{sam2_action_hover}}}"
                f"QPushButton#{button.objectName()}:pressed {{{sam2_action_pressed}}}"
                f"QPushButton#{button.objectName()}:disabled {{{sam2_action_disabled}}}"
            )

        for button in [self.btn_sam2_propagate_forward, self.btn_sam2_propagate_backward]:
            button.setStyleSheet(
                button.styleSheet()
                + "QPushButton { font-size: 16px; font-weight: 800; padding: 0px; }"
            )

        self.masks_list.setStyleSheet(
            "QListWidget {"
            " background-color: #0f141c;"
            " border: 1px solid #2a3444;"
            " border-radius: 7px;"
            " color: #e8edf5;"
            "}"
            "QListWidget::item {"
            " padding: 4px;"
            " border-radius: 4px;"
            "}"
            "QListWidget::item:selected {"
            " background-color: #0f3c57;"
            " color: #f7fbff;"
            "}"
            "QListWidget::item:hover {"
            " background-color: #1d2734;"
            "}"
        )

    def set_translator(self, translate: Callable[[str], str]) -> None:
        self._tr = translate
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.model_label.setText(self._tr("sam_model"))
        self.model_label.setToolTip(self._tr("sam_model_tooltip"))
        current_mt = str(self.model_combo.currentData() or "vit_h")
        self._rebuild_model_items(current_mt)
        self._refresh_download_button_state()
        self.status_label.setText(self._tr("lbl_sam_status_default"))
        self.prompts_label.setText(self._tr("lbl_sam_prompts"))
        self.btn_positive.setToolTip(self._tr("btn_positive_tooltip"))
        self.btn_positive.setText("+")
        self.btn_negative.setToolTip(self._tr("btn_negative_tooltip"))
        self.btn_negative.setText("-")
        self.btn_clear.setText(self._tr("btn_clear_points"))
        self.btn_live_sam2.setToolTip(self._tr("btn_live_sam2_tooltip"))
        self.btn_live_sam2.setText(self._tr("btn_live_sam2"))
        self.masks_label.setText(self._tr("lbl_masks"))
        self.btn_generate_mask.setToolTip(self._tr("btn_generate_mask_tooltip"))
        self.btn_generate_mask.setText(self._tr("btn_generate_mask"))
        self.sam2_actions_label.setText(self._tr("sam2_actions_label"))
        self.btn_sam2_propagate_backward.setText("◀")
        self.btn_sam2_propagate_backward.setToolTip(self._tr("sam2_btn_propagate_backward_tooltip"))
        self.btn_sam2_propagate_forward.setText("▶")
        self.btn_sam2_propagate_forward.setToolTip(self._tr("sam2_btn_propagate_forward_tooltip"))
        self.btn_sam2_reprompt.setText(self._tr("sam2_btn_reprompt"))
        self.btn_sam2_reprompt.setToolTip(self._tr("sam2_btn_reprompt_tooltip"))
        self.btn_sam2_reset_session.setText(self._tr("sam2_btn_reset_session"))
        self.btn_sam2_reset_session.setToolTip(self._tr("sam2_btn_reset_session_tooltip"))
        self.btn_add_mask.setToolTip(self._tr("btn_add_mask_tooltip"))
        self.btn_add_mask.setText(self._tr("btn_add_mask"))
        self.btn_remove_mask.setToolTip(self._tr("btn_remove_mask_tooltip"))
        self.btn_remove_mask.setText(self._tr("btn_remove_mask"))
        self.btn_load_mask.setToolTip(self._tr("btn_load_mask_tooltip"))
        self.btn_load_mask.setText(self._tr("btn_load_mask"))
        self.masks_list.setToolTip(self._tr("masks_list_tooltip"))
        self._sync_sam2_action_buttons(live_checked=self.btn_live_sam2.isChecked())

    def load_from_properties(self, props: dict) -> None:
        self._loading_properties = True
        try:
            model_type = str(props.get("model_type", "vit_h"))
            self._rebuild_model_items(model_type)
            self._refresh_download_button_state()
            self.status_label.setText(self._resolve_status_text(props))
            point_mode = str(props.get("point_mode", "positive")).strip().lower()
            with QSignalBlocker(self.btn_positive), QSignalBlocker(self.btn_negative), QSignalBlocker(self.btn_live_sam2):
                self.btn_positive.setChecked(point_mode != "negative")
                self.btn_negative.setChecked(point_mode == "negative")
                requested_live = bool(props.get("live_sam2", False))
                self.btn_live_sam2.setChecked(requested_live)
            self._sync_sam2_action_buttons(live_checked=bool(props.get("live_sam2", False)))
            live_checked = self.btn_live_sam2.isChecked()
            self.btn_positive.setEnabled(not live_checked)
            self.btn_negative.setEnabled(not live_checked)
            self.btn_generate_mask.setEnabled(not live_checked)
            props["live_sam2"] = live_checked

            raw_items = props.get("mask_items", [])
            mask_items = [str(item) for item in raw_items] if isinstance(raw_items, list) else []
            self.masks_list.clear()
            for item in mask_items:
                self.masks_list.addItem(item)
        finally:
            self._loading_properties = False

    def write_to_properties(self, props: dict) -> None:
        props["model_type"] = str(self.model_combo.currentData() or "vit_h")
        props["live_sam2"] = self.btn_live_sam2.isChecked()
        props["point_mode"] = "positive" if self.btn_positive.isChecked() else "negative"
        props["mask_items"] = [self.masks_list.item(i).text() for i in range(self.masks_list.count())]
        props["sam_status"] = self.status_label.text().strip()

    def _resolve_status_text(self, props: dict) -> str:
        raw_status = str(props.get("sam_status", "")).strip()
        return raw_status or self._tr("sam_live_off")

    def set_status(self, text: str, props: dict | None = None) -> None:
        self.status_label.setText(text)
        if props is not None:
            props["sam_status"] = text

    def refresh_masks_list(self, mask_items: list[str]) -> None:
        self.masks_list.clear()
        for item in mask_items:
            self.masks_list.addItem(item)

    def sync_controls(self, props: dict) -> None:
        point_mode = str(props.get("point_mode", "positive")).strip().lower()
        self.btn_positive.setChecked(point_mode != "negative")
        self.btn_negative.setChecked(point_mode == "negative")
        requested_live = bool(props.get("live_sam2", False))
        self.btn_live_sam2.setChecked(requested_live)
        self._sync_sam2_action_buttons(live_checked=requested_live)
        live_checked = self.btn_live_sam2.isChecked()
        self.btn_positive.setEnabled(not live_checked)
        self.btn_negative.setEnabled(not live_checked)
        self.btn_generate_mask.setEnabled(not live_checked)
        props["live_sam2"] = live_checked

    def _sync_sam2_action_buttons(self, *, live_checked: bool) -> None:
        self.btn_live_sam2.setVisible(True)
        enabled = not live_checked
        self.sam2_actions_label.setVisible(True)
        self.sam2_actions_row.setVisible(True)
        self.mask_actions_row.setVisible(True)
        self.load_mask_row.setVisible(True)
        self.btn_sam2_propagate_forward.setEnabled(enabled)
        self.btn_sam2_propagate_backward.setEnabled(enabled)
        self.btn_sam2_reprompt.setEnabled(enabled)
        self.btn_sam2_reset_session.setEnabled(True)

    def _is_cpu_device(self) -> bool:
        if self._is_cpu_cached is not None:
            return self._is_cpu_cached
        forced = os.environ.get("KEYFLOW_DEVICE", "").strip().lower()
        if forced:
            self._is_cpu_cached = forced == "cpu"
            return self._is_cpu_cached
        self._is_cpu_cached = str(get_device().type).strip().lower() == "cpu"
        return self._is_cpu_cached

    def _on_backend_or_live_changed(self) -> None:
        if self._loading_properties:
            return
        current_mt = str(self.model_combo.currentData() or "vit_h")
        if self._is_cpu_device() and current_mt == "vit_h":
            # On CPU, defaulting SAM2 to Small avoids accidental very slow runs.
            current_mt = "vit_b"
        self._rebuild_model_items(current_mt)
        self._refresh_download_button_state()
        self._sync_sam2_action_buttons(live_checked=self.btn_live_sam2.isChecked())
        live_checked = self.btn_live_sam2.isChecked()
        self.btn_positive.setEnabled(not live_checked)
        self.btn_negative.setEnabled(not live_checked)
        self.btn_generate_mask.setEnabled(not live_checked)
        if self._is_cpu_device():
            selected_mt = str(self.model_combo.currentData() or "vit_h")
            if selected_mt != "vit_b":
                self.status_label.setText(self._tr("sam2_cpu_recommend_small_status"))