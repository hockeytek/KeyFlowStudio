"""Self-contained MatAnyone2 matting node controller.

Owns InferenceWorker + QThread and all inference lifecycle logic
(start, cancel, progress, finish, error).  MainWindow delegates to
this controller instead of managing the worker directly.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal

from app.workers import InferenceWorker
from app.runtime_contract import (
    RuntimeConfig,
    RuntimeResult,
    is_runtime_cancelled,
    normalize_frame_progress,
    normalize_stage_progress,
    tr_with_fallback,
)


class MattingNodeController(QObject):
    """Encapsulates all MatAnyone2 inference logic as a node-level controller."""

    # ── Outbound signals (MainWindow connects to these) ──
    status_changed = Signal(str)
    stage_progress = Signal(int, str)          # percent, status_text
    node_frame_progress = Signal(str, int, int)  # node_type, current_frame, total_frames
    frame_progress = Signal(int, int)           # current_frame, total_frames
    frame_preview = Signal(object, object, int)  # foreground_rgb, alpha_rgb, frame_index
    graph_stream_preview = Signal(str, object, int)  # write_node_id, preview_rgb_or_gray, frame_index
    log_message = Signal(str)
    corridorkey_mode_resolved = Signal(str, str, str)  # requested_mode, effective_mode, reason_key
    processing_started = Signal()
    processing_finished = Signal(dict)          # RuntimeResult
    error_occurred = Signal(str)                # error message
    controls_busy_changed = Signal(bool)        # True = busy
    _start_job_requested = Signal(object, object, object, object, object)

    def __init__(self, translate: Callable[[str], str], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._tr = translate
        self._language_code = "ru"

        # ── Worker (created per-job, destroyed on cleanup) ──
        self._worker_thread: QThread | None = None
        self._worker: InferenceWorker | None = None

    # ── Public properties ──

    @property
    def is_active(self) -> bool:
        """True while an inference job is running."""
        return self._worker is not None

    # ── Public API called by MainWindow ──

    def _tr_status(self, matting_key: str, fallback_key: str) -> str:
        return tr_with_fallback(self._tr, matting_key, fallback_key)

    def set_language(self, language_code: str) -> None:
        self._language_code = language_code
        if self._worker is not None:
            self._worker.set_language(language_code)

    def set_translator(self, translate: Callable[[str], str]) -> None:
        self._tr = translate

    def start(
        self,
        input_path: str,
        mask_path: str,
        output_dir: str,
        config: RuntimeConfig,
        sam_service=None,
    ) -> None:
        """Launch an inference job in a background thread.

        ``config`` keys: erode_kernel, dilate_kernel, is_video,
        n_warmup, start_frame, end_frame.
        ``sam_service``: optional SamService instance; if provided it will be
        unloaded from memory before MatAnyone2 is loaded.
        """
        if self._worker is not None:
            self.status_changed.emit(self._tr_status("matting_wait_processing", "status_start"))
            return

        self._worker_thread = QThread(self)
        self._worker = InferenceWorker()
        self._worker.set_language(self._language_code)
        self._worker.moveToThread(self._worker_thread)
        self._start_job_requested.connect(self._worker.start_job)

        # Forward worker signals
        self._worker.stage_progress.connect(self._on_stage_progress)
        self._worker.node_frame_progress.connect(self.node_frame_progress)
        self._worker.progress.connect(self._on_frame_progress)
        self._worker.preview_frame.connect(self._on_frame_preview)
        self._worker.graph_stream_preview.connect(self._on_graph_stream_preview)
        self._worker.log_message.connect(self.log_message)
        self._worker.corridorkey_mode_resolved.connect(self.corridorkey_mode_resolved)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)

        self.controls_busy_changed.emit(True)
        self.processing_started.emit()
        self._worker_thread.start()
        self._start_job_requested.emit(input_path, mask_path, output_dir, config, sam_service)

    def cancel(self) -> None:
        """Request cancellation of the running job."""
        if self._worker is not None:
            self._worker.set_cancel()
        self.status_changed.emit(self._tr_status("matting_status_cancel", "status_cancel"))

    def cleanup(self) -> None:
        """Stop thread and release worker resources."""
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait()
        if self._worker is not None:
            try:
                self._start_job_requested.disconnect(self._worker.start_job)
            except (TypeError, RuntimeError):
                pass
        self._worker_thread = None
        self._worker = None

    def shutdown(self) -> None:
        """Cancel + cleanup (called on app close)."""
        if self._worker is not None:
            self._worker.set_cancel()
        self.cleanup()

    # ── Worker callbacks (private) ──

    def _on_stage_progress(self, percent: int, status_text: str) -> None:
        p, text = normalize_stage_progress(percent, status_text)
        self.stage_progress.emit(p, text)
        self.status_changed.emit(text)

    def _on_frame_progress(self, current: int, total: int) -> None:
        cur, tot = normalize_frame_progress(current, total)
        self.frame_progress.emit(cur, tot)

    def _on_frame_preview(self, foreground_rgb, alpha_rgb, frame_index: int) -> None:
        self.frame_preview.emit(foreground_rgb, alpha_rgb, frame_index)

    def _on_graph_stream_preview(self, write_node_id: str, preview_frame, frame_index: int) -> None:
        self.graph_stream_preview.emit(write_node_id, preview_frame, frame_index)

    def _on_finished(self, result: RuntimeResult) -> None:
        self.cleanup()
        self.controls_busy_changed.emit(False)
        self.processing_finished.emit(result)
        if is_runtime_cancelled(result):
            self.status_changed.emit(self._tr_status("matting_status_stopped", "status_stopped"))
        else:
            self.status_changed.emit(self._tr_status("matting_status_done", "status_done"))

    def _on_error(self, error_message: str) -> None:
        self.cleanup()
        self.controls_busy_changed.emit(False)
        self.error_occurred.emit(error_message)
        self.status_changed.emit(self._tr_status("matting_status_error", "status_error"))
