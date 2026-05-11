"""SAM3 node properties mini-panel."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from app.node_graph.properties_style import apply_properties_reference_style, configure_inline_layout, configure_properties_panel
from app.services.sam3_service import Sam3Service


READY_ROLE = Qt.ItemDataRole.UserRole + 1


class Sam3ModelDelegate(QStyledItemDelegate):
    """Draws a green dot next to models whose weights are already downloaded."""

    def paint(self, painter, option, index) -> None:
        ready = bool(index.data(READY_ROLE))
        style = option.widget.style() if option.widget is not None else QApplication.style()
        opt = option
        self.initStyleOption(opt, index)
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, option.widget)
        if not ready:
            return
        painter.save()
        color = QColor("#39d98a")
        if option.state & QStyle.StateFlag.State_Selected:
            color = QColor("#dfffea")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        dot_d = 8
        x = option.rect.right() - dot_d - 14
        y = option.rect.y() + (option.rect.height() - dot_d) // 2
        painter.drawEllipse(x, y, dot_d, dot_d)
        painter.restore()


class Sam3PropertiesPanel(QWidget):
    """Compact SAM3 controls panel used inside node properties."""

    MODELS = ["sam3", "sam3.1"]
    _MODEL_SHORT_LABELS = {"sam3": "SAM3", "sam3.1": "SAM3.1"}
    _MODEL_TOOLTIP_KEYS = {
        "sam3":   "sam3_model_sam3_tooltip",
        "sam3.1": "sam3_model_sam31_tooltip",
    }
    _loading_properties: bool = False

    def __init__(self, translate: Callable[[str], str], parent=None) -> None:
        super().__init__(parent)
        self._tr = translate

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

        # ── Concept prompt ───────────────────────────────────────────
        self.prompts_label = QLabel(self)
        self.prompts_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prompts_label.setStyleSheet(
            "color: #666666; font-size: 10px; font-weight: bold; letter-spacing: 1px;"
        )
        root.addWidget(self.prompts_label)

        self.concept_edit = QPlainTextEdit(self)
        self.concept_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.concept_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        root.addWidget(self.concept_edit)

        self.btn_generate_mask = QPushButton(self)
        self.btn_generate_mask.setObjectName("prop_sam3_generate_mask")
        self.btn_generate_mask.setMinimumSize(170, 28)
        root.addWidget(self.btn_generate_mask)

        # ── Model selection ──────────────────────────────────────────
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
        self.model_combo.setItemDelegate(Sam3ModelDelegate(self.model_combo))
        self.model_row_layout.addWidget(self.model_combo)

        self.download_button = QPushButton(self.model_row)
        self.download_button.setFixedWidth(120)
        self.download_button.clicked.connect(self._download_weights)
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

        apply_properties_reference_style(self)
        self._apply_styles()
        self._update_status_height()
        self._update_prompt_height()
        self.retranslate_ui()

    # ── Model combo helpers ──────────────────────────────────────────

    def _rebuild_model_items(self, selected: str | None = None) -> None:
        label_map = self._MODEL_SHORT_LABELS
        tooltip_map = self._MODEL_TOOLTIP_KEYS
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for mt in self.MODELS:
            short_label = label_map.get(mt, mt)
            status = Sam3Service.get_weight_status(mt)
            tooltip_key = tooltip_map.get(mt, "")
            tooltip = self._tr(tooltip_key) if tooltip_key else ""
            self.model_combo.addItem(short_label, mt)
            idx = self.model_combo.count() - 1
            self.model_combo.setItemData(idx, status["state"] == "ready", READY_ROLE)
            self.model_combo.setItemData(idx, tooltip, Qt.ItemDataRole.ToolTipRole)
        target = selected or "sam3"
        idx = self.model_combo.findData(target)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.model_combo.blockSignals(False)

    def _refresh_download_button_state(self) -> None:
        mt = str(self.model_combo.currentData() or "sam3")
        status = Sam3Service.get_weight_status(mt)
        if status["state"] == "ready":
            self.download_button.setText(self._tr("sam3_download_button_ready"))
            self.download_button.setEnabled(True)
            self.download_button.setToolTip(self._tr("sam3_download_button_ready_tooltip"))
        else:
            self.download_button.setText(self._tr("sam3_download_button_missing"))
            self.download_button.setEnabled(True)
            self.download_button.setToolTip(self._tr("sam3_download_button_missing_tooltip"))

    def _on_download_progress(self, percent: int, _message: str) -> None:
        self.download_progress.setVisible(True)
        self.download_progress.setValue(max(0, min(100, int(percent))))
        QApplication.processEvents()

    def _download_weights(self) -> None:
        mt = str(self.model_combo.currentData() or "sam3")
        name = Sam3Service.SAM3_LABELS.get(mt, mt)
        self.download_button.setEnabled(False)
        self.download_progress.setVisible(True)
        self.download_progress.setValue(0)
        try:
            Sam3Service.download_checkpoint_for(mt, self._on_download_progress)
            self._rebuild_model_items(mt)
            self._refresh_download_button_state()
            self.download_progress.setValue(100)
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
        vertical_margins = (
            self.status_label.contentsMargins().top()
            + self.status_label.contentsMargins().bottom()
        )
        self.status_label.setFixedHeight(metrics.lineSpacing() * 2 + vertical_margins + 4)

    def _update_prompt_height(self) -> None:
        metrics = self.concept_edit.fontMetrics()
        self.concept_edit.setFixedHeight(metrics.lineSpacing() * 3 + 18)

    def _apply_styles(self) -> None:
        self.concept_edit.setStyleSheet(
            "QPlainTextEdit {"
            " background-color: #111823;"
            " border: 1px solid #2a3444;"
            " border-radius: 7px;"
            " color: #eef3fb;"
            " padding: 4px 8px;"
            " font-size: 12px;"
            "}"
            "QPlainTextEdit:focus {"
            " border: 1px solid #43c7ff;"
            "}"
        )

        self.btn_generate_mask.setStyleSheet(
            "QPushButton#prop_sam3_generate_mask {"
            " background-color: #0f5a33;"
            " border: 1px solid #1f8f58;"
            " color: #ffffff;"
            " padding: 0 12px;"
            " min-height: 26px;"
            " border-radius: 9px;"
            " font-weight: 600;"
            "}"
            "QPushButton#prop_sam3_generate_mask:hover {"
            " background-color: #1f8f58;"
            " border: 1px solid #2fb878;"
            "}"
            "QPushButton#prop_sam3_generate_mask:pressed {"
            " background-color: #0a3a1f;"
            " border: 1px solid #147d42;"
            "}"
            "QPushButton#prop_sam3_generate_mask:disabled {"
            " background-color: #16241c;"
            " border: 1px solid #2a3a31;"
            " color: #6f8477;"
            "}"
        )

    # ── Translator / retranslate ─────────────────────────────────────

    def set_translator(self, translate: Callable[[str], str]) -> None:
        self._tr = translate
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.model_label.setText(self._tr("sam3_model"))
        self.model_label.setToolTip(self._tr("sam3_model_tooltip"))
        current_mt = str(self.model_combo.currentData() or "sam3")
        self._rebuild_model_items(current_mt)
        self._refresh_download_button_state()
        self.status_label.setText(self._tr("lbl_sam3_status_default"))
        self.prompts_label.setText(self._tr("lbl_sam3_concept_prompt"))
        self.concept_edit.setPlaceholderText(self._tr("sam3_concept_placeholder"))
        self.concept_edit.setToolTip(self._tr("sam3_concept_tooltip"))
        self.btn_generate_mask.setToolTip(self._tr("btn_generate_sam3_mask_tooltip"))
        self.btn_generate_mask.setText(self._tr("btn_generate_mask"))

    # ── Properties load / save ───────────────────────────────────────

    def load_from_properties(self, props: dict) -> None:
        self._loading_properties = True
        try:
            model_type = str(props.get("model_type", "sam3"))
            self._rebuild_model_items(model_type)
            self._refresh_download_button_state()
            self.status_label.setText(self._resolve_status_text(props))
            self.concept_edit.setPlainText(str(props.get("concept", "") or ""))
        finally:
            self._loading_properties = False

    def write_to_properties(self, props: dict) -> None:
        props["model_type"] = str(self.model_combo.currentData() or "sam3")
        props["concept"] = self.concept_edit.toPlainText().strip()
        props.pop("point_mode", None)
        props.pop("live_sam2", None)
        props.pop("prompt_points", None)
        props.pop("prompt_labels", None)
        props.pop("mask_items", None)
        props.pop("selected_mask_rows", None)
        props["sam_status"] = self.status_label.text().strip()

    def _resolve_status_text(self, props: dict) -> str:
        raw_status = str(props.get("sam_status", "")).strip()
        return raw_status or self._tr("lbl_sam3_status_default")

    def set_status(self, text: str, props: dict | None = None) -> None:
        self.status_label.setText(text)
        if props is not None:
            props["sam_status"] = text
