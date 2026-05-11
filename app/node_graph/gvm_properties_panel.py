"""GVM node properties panel."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable

from PySide6.QtCore import Qt, QThread, QObject, Signal
from PySide6.QtWidgets import (
    QCheckBox,
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
from app.services.gvm_service import GVMService


class _GVMDownloadWorker(QObject):
    """Download GVM weights in a background thread."""

    progress = Signal(int, str)
    finished = Signal(str)
    error = Signal(str)

    def run(self) -> None:
        try:
            weights_path = GVMService.ensure_weights_available(self.progress.emit)
            self.finished.emit(str(weights_path))
        except Exception as exc:
            self.error.emit(str(exc).strip() or repr(exc))


class GVMPropertiesPanel(QWidget, NodePanelMixin):
    """Compact GVM controls used inside node properties.

    Layout mirrors CorridorKey: sectioned GridLayout with label + slider + spinbox rows.
    """

    _start_download = Signal()

    def __init__(self, translate: Callable[[str], str], parent=None) -> None:
        super().__init__(parent)
        self._tr = translate
        self._init_panel_layout()

        self.setObjectName("gvmPropertiesPanel")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.setFixedWidth(self._PANEL_WIDTH)
        self.setStyleSheet(
            "QWidget#gvmPropertiesPanel { background: #10151d; }"
            "QLabel { color: #d8dee7; }"
        )

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(6, 6, 6, 6)
        self.root.setSpacing(8)

        # ── Section: Inference ────────────────────────────────────────────────
        self.infer_section, self.infer_section_title, self.infer_form = self._create_section(expanded=True)

        self.batch_label = QLabel(self)
        self.batch_spin = QSpinBox(self)
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(8)
        self.batch_field, self.batch_slider = self._make_int_slider_field(
            self.batch_spin, min_value=1, max_value=32
        )

        self.chunk_label = QLabel(self)
        self.chunk_spin = QSpinBox(self)
        self.chunk_spin.setRange(1, 16)
        self.chunk_spin.setValue(4)
        self.chunk_field, self.chunk_slider = self._make_int_slider_field(
            self.chunk_spin, min_value=1, max_value=16
        )

        self.overlap_label = QLabel(self)
        self.overlap_spin = QSpinBox(self)
        self.overlap_spin.setRange(0, 8)
        self.overlap_spin.setValue(1)
        self.overlap_field, self.overlap_slider = self._make_int_slider_field(
            self.overlap_spin, min_value=0, max_value=8
        )

        self.interp_label = QLabel(self)
        self.interp_spin = QSpinBox(self)
        self.interp_spin.setRange(0, 8)
        self.interp_spin.setValue(1)
        self.interp_field, self.interp_slider = self._make_int_slider_field(
            self.interp_spin, min_value=0, max_value=8
        )

        self._add_form_row(self.infer_form, self.batch_label, self.batch_field)
        self._add_form_row(self.infer_form, self.chunk_label, self.chunk_field)
        self._add_form_row(self.infer_form, self.overlap_label, self.overlap_field)
        self._add_form_row(self.infer_form, self.interp_label, self.interp_field)

        self.noise_label = QLabel(self)
        self.noise_combo = QComboBox(self)
        self.noise_combo.addItem("Zeros", "zeros")
        self.noise_combo.addItem("Gaussian", "gaussian")

        self.clip_emb_check = QCheckBox(self)

        self._add_form_row(self.infer_form, self.noise_label, self.noise_combo)
        self._add_form_row(self.infer_form, QLabel(self), self.clip_emb_check)

        # ── Section: Post-process ─────────────────────────────────────────────
        self.post_section, self.post_section_title, self.post_form = self._create_section(expanded=True)

        self.dilate_label = QLabel(self)
        self.dilate_spin = QSpinBox(self)
        self.dilate_spin.setRange(0, 50)
        self.dilate_spin.setValue(0)
        self.dilate_field, self.dilate_slider = self._make_int_slider_field(
            self.dilate_spin, min_value=0, max_value=50
        )

        self._add_form_row(self.post_form, self.dilate_label, self.dilate_field)

        # ── Download button ───────────────────────────────────────────────────
        self.download_button = QPushButton(self)
        self.download_button.clicked.connect(self._download_weights)
        self.download_progress = QProgressBar(self)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_progress.setTextVisible(True)
        self.download_progress.setFixedHeight(18)
        self.download_progress.setVisible(False)

        # ── Device info label (shown only on MPS) ─────────────────────────────
        self.device_info_label = QLabel(self)
        self.device_info_label.setWordWrap(True)
        self.device_info_label.setStyleSheet("color: #f0b35c; font-size: 11px;")
        self.device_info_label.setVisible(False)

        self.root.addWidget(self.infer_section)
        self.root.addWidget(self.post_section)
        self.root.addWidget(self.device_info_label)
        self.root.addWidget(self.download_button)
        self.root.addWidget(self.download_progress)
        self.root.addStretch(1)

        self._download_thread: QThread | None = None
        self._download_worker: _GVMDownloadWorker | None = None
        self._download_active = False

        self.retranslate_ui()

    # ── Translator / retranslation ────────────────────────────────────────────

    def set_translator(self, translate: Callable[[str], str]) -> None:
        self._tr = translate
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.infer_section_title.setText(self._tr("gvm_section_inference"))
        self.post_section_title.setText(self._tr("gvm_section_postprocess"))

        self.batch_label.setText(self._tr("gvm_num_frames_per_batch"))
        self.batch_label.setToolTip(self._tr("gvm_num_frames_per_batch_tooltip"))
        self.batch_spin.setToolTip(self._tr("gvm_num_frames_per_batch_tooltip"))

        self.chunk_label.setText(self._tr("gvm_decode_chunk_size"))
        self.chunk_label.setToolTip(self._tr("gvm_decode_chunk_size_tooltip"))
        self.chunk_spin.setToolTip(self._tr("gvm_decode_chunk_size_tooltip"))

        self.overlap_label.setText(self._tr("gvm_num_overlap_frames"))
        self.overlap_label.setToolTip(self._tr("gvm_num_overlap_frames_tooltip"))
        self.overlap_spin.setToolTip(self._tr("gvm_num_overlap_frames_tooltip"))

        self.interp_label.setText(self._tr("gvm_num_interp_frames"))
        self.interp_label.setToolTip(self._tr("gvm_num_interp_frames_tooltip"))
        self.interp_spin.setToolTip(self._tr("gvm_num_interp_frames_tooltip"))

        self.noise_label.setText(self._tr("gvm_noise_type"))
        self.noise_label.setToolTip(self._tr("gvm_noise_type_tooltip"))
        self.noise_combo.setToolTip(self._tr("gvm_noise_type_tooltip"))

        self.clip_emb_check.setText(self._tr("gvm_use_clip_img_emb"))
        self.clip_emb_check.setToolTip(self._tr("gvm_use_clip_img_emb_tooltip"))

        self.dilate_label.setText(self._tr("gvm_dilate_radius"))
        self.dilate_label.setToolTip(self._tr("gvm_dilate_radius_tooltip"))
        self.dilate_spin.setToolTip(self._tr("gvm_dilate_radius_tooltip"))

        self._refresh_device_info()
        self._refresh_download_button_state()

    def _refresh_device_info(self) -> None:
        if self._is_cloud_mode():
            self.device_info_label.setVisible(False)
            return
        try:
            from app.utils import get_device
            device = str(get_device())
        except Exception:
            device = "cpu"
        if device in ("mps", "cpu"):
            self.device_info_label.setText(self._tr("gvm_mps_cpu_notice"))
            self.device_info_label.setVisible(True)
        else:
            self.device_info_label.setVisible(False)

    def _refresh_download_button_state(self) -> None:
        if self._is_cloud_mode():
            if getattr(self, "_cloud_weights_ready", False):
                self.download_button.setText(self._tr("gvm_download_button_ready_cloud"))
                self.download_button.setToolTip(self._tr("gvm_download_button_ready_tooltip"))
            else:
                self.download_button.setText(self._tr("gvm_download_button_missing"))
                self.download_button.setToolTip(self._tr("gvm_download_button_missing_tooltip"))
            return

        status = GVMService.get_weights_status()
        if status.get("state") == "ready":
            self.download_button.setText(self._tr("gvm_download_button_ready"))
            self.download_button.setToolTip(self._tr("gvm_download_button_ready_tooltip"))
        else:
            self.download_button.setText(self._tr("gvm_download_button_missing"))
            self.download_button.setToolTip(self._tr("gvm_download_button_missing_tooltip"))

    # ── Download ──────────────────────────────────────────────────────────────

    def _download_weights(self) -> None:
        if self._is_cloud_mode():
            self._start_cloud_download("gvm")
            return

        if self._download_active:
            return

        status = GVMService.get_weights_status()
        if status.get("state") == "ready":
            self._refresh_download_button_state()
            QMessageBox.information(
                self,
                self._tr("info_title"),
                self._tr("gvm_weights_already_present"),
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
        self._download_worker = _GVMDownloadWorker()
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
            self._tr("gvm_weights_downloaded"),
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
            self._tr("gvm_weights_download_failed").format(error=error_message),
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

    # ── Properties I/O ────────────────────────────────────────────────────────

    def load_from_properties(self, props: dict) -> None:
        self._refresh_device_info()
        self.batch_spin.setValue(int(props.get("num_frames_per_batch", 8)))
        self.chunk_spin.setValue(int(props.get("decode_chunk_size", 4)))
        self.overlap_spin.setValue(int(props.get("num_overlap_frames", 1)))
        self.interp_spin.setValue(int(props.get("num_interp_frames", 1)))
        noise = str(props.get("noise_type", "zeros")).strip().lower()
        idx = self.noise_combo.findData(noise)
        self.noise_combo.setCurrentIndex(max(idx, 0))
        self.clip_emb_check.setChecked(bool(props.get("use_clip_img_emb", False)))
        self.dilate_spin.setValue(int(props.get("dilate_radius", 0)))

    def write_to_properties(self, props: dict) -> None:
        props["num_frames_per_batch"] = int(self.batch_spin.value())
        props["decode_chunk_size"] = int(self.chunk_spin.value())
        props["num_overlap_frames"] = int(self.overlap_spin.value())
        props["num_interp_frames"] = int(self.interp_spin.value())
        props["noise_type"] = self.noise_combo.currentData() or "zeros"
        props["use_clip_img_emb"] = bool(self.clip_emb_check.isChecked())
        props["dilate_radius"] = int(self.dilate_spin.value())
