"""MatAnyone2 node properties panel."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt, QThread, QObject, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.node_graph.node_panel_mixin import NodePanelMixin
from app.services.model_service import ModelService


class _MatAnyone2DownloadWorker(QObject):
    """Download MatAnyone2 weights in a background thread."""

    progress = Signal(int, str)
    finished = Signal(str)
    error = Signal(str)

    def run(self) -> None:
        try:
            weights_path = ModelService.ensure_weights_available(self.progress.emit)
            self.finished.emit(str(weights_path))
        except Exception as exc:
            self.error.emit(str(exc).strip() or repr(exc))

FG_BACKGROUND_OPTIONS: list[tuple[str, str]] = [
    ("green", "fg_bg_green"),
    ("checker", "fg_bg_checker"),
]


PRESET_VALUES: dict[str, tuple[int, int, int]] = {
    "Balanced": (10, 10, 10),
    "Eval LR (512p)": (4, 4, 1),
    "Eval HR (1080p)": (15, 15, 10),
}

PRESET_LABEL_KEYS: dict[str, str] = {
    "Balanced": "preset_balanced",
    "Eval LR (512p)": "preset_eval_lr",
    "Eval HR (1080p)": "preset_eval_hr",
    "Custom": "preset_custom",
}

PRESET_HELP_KEYS: dict[str, str] = {
    "Balanced": "preset_help_balanced",
    "Eval LR (512p)": "preset_help_eval_lr",
    "Eval HR (1080p)": "preset_help_eval_hr",
    "Custom": "preset_help_custom",
}

PRESET_ORDER = ["Eval LR (512p)", "Eval HR (1080p)", "Balanced", "Custom"]


class MattingPropertiesPanel(QWidget, NodePanelMixin):
    """Compact MatAnyone2 controls used inside node properties."""

    _start_download = Signal()

    def __init__(self, translate: Callable[[str], str], parent=None) -> None:
        super().__init__(parent)
        self._tr = translate
        self._init_panel_layout()
        self._preset_sync_in_progress = False

        self.setObjectName("mattingPropertiesPanel")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.setFixedWidth(self._PANEL_WIDTH)
        self.setStyleSheet(
            "QWidget#mattingPropertiesPanel { background: #10151d; }"
            "QLabel { color: #d8dee7; }"
        )

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(6, 6, 6, 6)
        self.root.setSpacing(8)

        # ── Section: Settings ─────────────────────────────────────────────────
        self.settings_section, self.settings_section_title, self.settings_form = self._create_section(expanded=True)

        self.preset_label = QLabel(self)
        self.preset_combo = QComboBox(self)
        self.preset_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.preset_combo.setFixedWidth(self._right_column_width)
        self.preset_combo.setFixedHeight(self._field_height)
        self._apply_reference_combo_style(self.preset_combo)

        self.fg_background_label = QLabel(self)
        self.fg_background_combo = QComboBox(self)
        self.fg_background_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.fg_background_combo.setFixedWidth(self._right_column_width)
        self.fg_background_combo.setFixedHeight(self._field_height)
        self._apply_reference_combo_style(self.fg_background_combo)

        self._add_form_row(self.settings_form, self.preset_label, self.preset_combo)
        self._add_form_row(self.settings_form, self.fg_background_label, self.fg_background_combo)

        # ── Section: Post-process ─────────────────────────────────────────────
        self.post_section, self.post_section_title, self.post_form = self._create_section(expanded=True)

        self.erode_label = QLabel(self)
        self.erode_spin = QSpinBox(self)
        self.erode_spin.setRange(0, 99)
        self.erode_field, self.erode_slider = self._make_int_slider_field(
            self.erode_spin, min_value=0, max_value=99
        )

        self.dilate_label = QLabel(self)
        self.dilate_spin = QSpinBox(self)
        self.dilate_spin.setRange(0, 99)
        self.dilate_field, self.dilate_slider = self._make_int_slider_field(
            self.dilate_spin, min_value=0, max_value=99
        )

        self.warmup_label = QLabel(self)
        self.warmup_spin = QSpinBox(self)
        self.warmup_spin.setRange(0, 99)
        self.warmup_field, self.warmup_slider = self._make_int_slider_field(
            self.warmup_spin, min_value=0, max_value=99
        )

        self._add_form_row(self.post_form, self.erode_label, self.erode_field)
        self._add_form_row(self.post_form, self.dilate_label, self.dilate_field)
        self._add_form_row(self.post_form, self.warmup_label, self.warmup_field)

        # ── Assemble root ─────────────────────────────────────────────────────
        # ── Download button ───────────────────────────────────────────────────
        self.download_button = QPushButton(self)
        self.download_button.clicked.connect(self._download_weights)
        self.download_progress = QProgressBar(self)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_progress.setTextVisible(True)
        self.download_progress.setFixedHeight(18)
        self.download_progress.setVisible(False)

        self.root.addWidget(self.settings_section)
        self.root.addWidget(self.post_section)
        self.root.addWidget(self.download_button)
        self.root.addWidget(self.download_progress)
        self.root.addStretch(1)

        self._download_thread: QThread | None = None
        self._download_worker: _MatAnyone2DownloadWorker | None = None
        self._download_active = False

        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.erode_spin.valueChanged.connect(self._sync_preset_selection)
        self.dilate_spin.valueChanged.connect(self._sync_preset_selection)
        self.warmup_spin.valueChanged.connect(self._sync_preset_selection)

        self.retranslate_ui()

    def set_translator(self, translate: Callable[[str], str]) -> None:
        self._tr = translate
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.settings_section_title.setText(self._tr("matting_section_settings"))
        self.post_section_title.setText(self._tr("matting_section_postprocess"))

        self.preset_label.setText(self._tr("lbl_preset"))
        self.erode_label.setText(self._tr("lbl_erode"))
        self.dilate_label.setText(self._tr("lbl_dilate"))
        self.warmup_label.setText(self._tr("lbl_warmup"))
        self.fg_background_label.setText(self._tr("lbl_fg_background"))

        current_bg = self.fg_background_combo.currentData() or "green"
        self.fg_background_combo.blockSignals(True)
        self.fg_background_combo.clear()
        for value, key in FG_BACKGROUND_OPTIONS:
            self.fg_background_combo.addItem(self._tr(key), value)
        idx = self.fg_background_combo.findData(current_bg)
        self.fg_background_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.fg_background_combo.blockSignals(False)

        self.preset_label.setToolTip(self._tr("node_props_preset_tooltip"))
        self.erode_spin.setToolTip(self._tr("node_props_erode_tooltip"))
        self.dilate_spin.setToolTip(self._tr("node_props_dilate_tooltip"))
        self.warmup_spin.setToolTip(self._tr("node_props_warmup_tooltip"))

        current_preset = self._current_preset_key()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for preset_name in PRESET_ORDER:
            self.preset_combo.addItem(self._tr(PRESET_LABEL_KEYS[preset_name]), preset_name)
        if current_preset:
            index = self.preset_combo.findData(current_preset)
            if index >= 0:
                self.preset_combo.setCurrentIndex(index)
        self.preset_combo.blockSignals(False)
        self._update_preset_tooltip()
        self._refresh_download_button_state()

    def _current_preset_key(self) -> str:
        preset_key = self.preset_combo.currentData()
        if isinstance(preset_key, str) and preset_key:
            return preset_key
        current_text = self.preset_combo.currentText().strip()
        for preset_name, label_key in PRESET_LABEL_KEYS.items():
            if current_text in {preset_name, self._tr(label_key)}:
                return preset_name
        return "Custom"

    def _set_current_preset_key(self, preset_name: str) -> None:
        index = self.preset_combo.findData(preset_name)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)

    def _apply_preset_values(self, preset_name: str) -> None:
        values = PRESET_VALUES.get(preset_name)
        if values is None:
            return
        self._preset_sync_in_progress = True
        try:
            erode, dilate, warmup = values
            self.erode_spin.setValue(erode)
            self.dilate_spin.setValue(dilate)
            self.warmup_spin.setValue(warmup)
        finally:
            self._preset_sync_in_progress = False

    def _on_preset_changed(self, _index: int) -> None:
        if self._preset_sync_in_progress:
            return
        preset_key = self._current_preset_key()
        self._apply_preset_values(preset_key)
        self._update_preset_tooltip(preset_key)

    def _update_preset_tooltip(self, preset_name: str | None = None) -> None:
        if preset_name is None:
            preset_name = self._current_preset_key()
        help_key = PRESET_HELP_KEYS.get(preset_name, PRESET_HELP_KEYS["Custom"])
        self.preset_combo.setToolTip(self._tr(help_key))

    def _sync_preset_selection(self) -> None:
        if self._preset_sync_in_progress:
            return

        current_values = (
            self.erode_spin.value(),
            self.dilate_spin.value(),
            self.warmup_spin.value(),
        )
        matched_preset = next(
            (name for name, values in PRESET_VALUES.items() if values == current_values),
            "Custom",
        )
        if self._current_preset_key() == matched_preset:
            return

        self._preset_sync_in_progress = True
        try:
            self._set_current_preset_key(matched_preset)
        finally:
            self._preset_sync_in_progress = False
        self._update_preset_tooltip(matched_preset)

    # ── Download ──────────────────────────────────────────────────────────────

    def _refresh_download_button_state(self) -> None:
        if self._is_cloud_mode():
            if getattr(self, "_cloud_weights_ready", False):
                self.download_button.setText(self._tr("matanyone2_download_button_ready_cloud"))
                self.download_button.setToolTip(self._tr("matanyone2_download_button_ready_tooltip"))
            else:
                self.download_button.setText(self._tr("matanyone2_download_button_missing"))
                self.download_button.setToolTip(self._tr("matanyone2_download_button_missing_tooltip"))
            return

        status = ModelService.get_weights_status()
        if status.get("state") == "ready":
            self.download_button.setText(self._tr("matanyone2_download_button_ready"))
            self.download_button.setToolTip(self._tr("matanyone2_download_button_ready_tooltip"))
        else:
            self.download_button.setText(self._tr("matanyone2_download_button_missing"))
            self.download_button.setToolTip(self._tr("matanyone2_download_button_missing_tooltip"))

    def _download_weights(self) -> None:
        if self._is_cloud_mode():
            self._start_cloud_download("matanyone2")
            return

        if self._download_active:
            return

        status = ModelService.get_weights_status()
        if status.get("state") == "ready":
            self._refresh_download_button_state()
            QMessageBox.information(
                self,
                self._tr("info_title"),
                self._tr("matanyone2_weights_already_present"),
            )
            return

        self._ensure_download_worker()
        if self._download_worker is None:
            return

        self._download_active = True
        self.download_button.setEnabled(False)
        self.download_progress.setVisible(True)
        self.download_progress.setValue(0)
        self._start_download.emit()

    def _ensure_download_worker(self) -> None:
        if self._download_worker is not None and self._download_thread is not None:
            return

        self._download_thread = QThread(self)
        self._download_worker = _MatAnyone2DownloadWorker()
        self._download_worker.moveToThread(self._download_thread)

        self._start_download.connect(self._download_worker.run, Qt.ConnectionType.QueuedConnection)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.error.connect(self._on_download_error)
        self._download_worker.finished.connect(self._download_thread.quit)
        self._download_worker.error.connect(self._download_thread.quit)
        self._download_thread.finished.connect(self._on_download_thread_finished)
        self._download_thread.finished.connect(self._download_thread.deleteLater)
        self._download_thread.start()

    def _on_download_progress(self, percent: int, message: str) -> None:
        self.download_progress.setVisible(True)
        self.download_progress.setValue(max(0, min(100, int(percent))))
        self.download_progress.setToolTip(message)

    def _on_download_finished(self, _path: str) -> None:
        self._download_active = False
        self.download_button.setEnabled(True)
        self.download_progress.setValue(100)
        self.download_progress.setVisible(False)
        self._refresh_download_button_state()
        QMessageBox.information(
            self,
            self._tr("info_title"),
            self._tr("matanyone2_weights_downloaded"),
        )

    def _on_download_error(self, error_message: str) -> None:
        self._download_active = False
        self.download_button.setEnabled(True)
        self.download_progress.setValue(0)
        self.download_progress.setVisible(False)
        self._refresh_download_button_state()
        QMessageBox.critical(
            self,
            self._tr("inference_error_title"),
            self._tr("matanyone2_weights_download_failed").format(error=error_message),
        )

    def _on_download_thread_finished(self) -> None:
        if self._download_worker is not None:
            self._download_worker.deleteLater()
        self._download_worker = None
        self._download_thread = None

    def closeEvent(self, event) -> None:
        if self._download_thread is not None and self._download_thread.isRunning():
            self._download_thread.quit()
            self._download_thread.wait()
        super().closeEvent(event)

    def load_from_properties(self, props: dict) -> None:
        preset = str(props.get("preset", "Eval HR (1080p)"))
        index = self.preset_combo.findData(preset)
        target_index = index if index >= 0 else self.preset_combo.findData("Custom")
        self.preset_combo.setCurrentIndex(max(target_index, 0))

        if preset in PRESET_VALUES:
            self._apply_preset_values(preset)
        else:
            self.erode_spin.setValue(int(props.get("erode", 15)))
            self.dilate_spin.setValue(int(props.get("dilate", 15)))
            self.warmup_spin.setValue(int(props.get("warmup", 10)))
            self._sync_preset_selection()
        self._update_preset_tooltip()

        bg = str(props.get("fg_background", "green"))
        idx = self.fg_background_combo.findData(bg)
        self.fg_background_combo.setCurrentIndex(idx if idx >= 0 else 0)

    def write_to_properties(self, props: dict) -> None:
        props["preset"] = self._current_preset_key()
        props["erode"] = self.erode_spin.value()
        props["dilate"] = self.dilate_spin.value()
        props["warmup"] = self.warmup_spin.value()
        props["fg_background"] = self.fg_background_combo.currentData() or "green"