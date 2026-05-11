"""Self-contained SAM node controller.

Owns SamRuntimeState, SamMaskWorker+QThread, and all interactive
SAM logic (points, masks, generation, Live SAM).  MainWindow delegates
to this controller instead of managing SAM internals directly.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QThread, Qt, Signal

from app.sam_runtime_state import SamRuntimeState
from app.services.sam2_service import Sam2Service
from app.services.sam3_service import Sam3Service
from app.workers import SamMaskWorker


class Sam2NodeController(QObject):
    """Encapsulates all SAM interactive logic as a node-level controller."""

    # ── Outbound signals (MainWindow connects to these) ──
    status_changed = Signal(str)
    input_preview_needed = Signal()           # request main to repaint input viewer
    mask_preview_available = Signal(object)    # np.ndarray | None → show on output viewer
    progress_updated = Signal(int, str)        # percent, status
    generation_started = Signal()
    generation_finished = Signal()
    controls_busy_changed = Signal(bool)       # True = busy, False = idle
    mask_list_changed = Signal()               # masks_list needs refresh
    node_frame_progress = Signal(str, int, int)  # node_type, current, total
    error_occurred = Signal(str, bool)         # message, show_dialog

    # ── Internal: cross-thread dispatch to worker ──
    _do_generate = Signal(object, object, object, object)  # image, points, labels, context
    _do_set_model_type = Signal(str)               # model_type string → worker.set_model_type
    _do_set_backend = Signal(str)                  # backend string → worker.set_backend
    _do_propagate = Signal(object)                 # context dict → worker.propagate_masks
    _do_reprompt = Signal(object)                  # context dict → worker.reprompt_frame
    _do_reset_video_session = Signal()             # worker.reset_video_session
    _do_cancel = Signal()                          # worker.request_cancel

    def __init__(self, translate: Callable[[str], str], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._tr = translate

        # ── State ──
        self.state = SamRuntimeState()
        self.generation_active = False
        self._request_show_errors = True
        self._request_live_mode = False
        self._request_click: tuple[int, int] | None = None
        self._request_frame_index = 0
        self._request_sequence_mode = False
        self._backend = "sam2"
        self._model_type = "vit_h"

        # ── Worker thread ──
        self._worker_thread = QThread(self)
        self._worker = SamMaskWorker()
        self._worker.moveToThread(self._worker_thread)
        self._worker.stage_progress.connect(self._on_stage_progress)
        self._worker.node_frame_progress.connect(self.node_frame_progress)
        self._worker.finished.connect(self._on_mask_ready)
        self._worker.error.connect(self._on_mask_error)
        self._do_generate.connect(self._worker.generate_mask)
        self._do_set_model_type.connect(self._worker.set_model_type)
        self._do_set_backend.connect(self._worker.set_backend)
        self._do_propagate.connect(self._worker.propagate_masks)
        self._do_reprompt.connect(self._worker.reprompt_frame)
        self._do_reset_video_session.connect(self._worker.reset_video_session)
        self._do_cancel.connect(self._worker.request_cancel, Qt.ConnectionType.DirectConnection)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()

    def set_model_type(self, model_type: str) -> None:
        """Switch SAM model (unloads existing weights, reloads on next generation)."""
        normalized = str(model_type or "").strip().lower()
        if normalized not in {"vit_h", "vit_l", "vit_b", "sam3", "sam3.1"}:
            normalized = "sam3" if self._backend == "sam3" else "vit_h"
        self._model_type = normalized
        self._do_set_model_type.emit(normalized)

    def set_backend(self, backend: str) -> None:
        normalized = str(backend or "sam2").strip().lower()
        if normalized not in {"sam2", "sam3"}:
            normalized = "sam2"
        if self._backend == normalized:
            return
        self._backend = normalized
        self._do_set_backend.emit(normalized)

    # ── Public API called by MainWindow / NodeGraphDialog ────────────

    def set_language(self, language_code: str) -> None:
        self._worker.set_language(language_code)

    def set_translator(self, translate: Callable[[str], str]) -> None:
        self._tr = translate

    def reset_for_media(self) -> None:
        self.state.reset_for_media()
        self.mask_list_changed.emit()
        self.status_changed.emit(self._tr("sam_live_off"))

    def set_point_mode(self, positive: bool) -> None:
        self.state.sync_controls(point_mode="positive" if positive else "negative")

    def toggle_live_sam2(self, checked: bool) -> None:
        self.state.sync_controls(live_sam2=checked)
        if checked:
            self.status_changed.emit(self._tr("sam_live_hint"))
        else:
            self.status_changed.emit(self._tr("sam_live_off"))

    def sync_controls(
        self,
        *,
        point_mode: str | None = None,
        live_sam2: bool | None = None,
        backend: str | None = None,
    ) -> None:
        if point_mode is not None:
            self.state.sync_controls(point_mode=point_mode)
        if live_sam2 is not None:
            self.state.sync_controls(live_sam2=live_sam2)
        if backend is not None:
            self.set_backend(backend)

    def add_point(self, x: int, y: int, positive: bool) -> None:
        self.state.add_point(x, y, positive)

    def pop_last_point(self) -> bool:
        popped = self.state.pop_last_point()
        return popped

    def clear_points(self) -> None:
        self.state.clear_prompt_state()
        self.input_preview_needed.emit()
        self.status_changed.emit(self._tr("sam_points_cleared"))

    def _ensure_weights_available(self, *, show_dialog: bool) -> bool:
        if self._backend == "sam3":
            model_type = self._model_type if self._model_type in {"sam3", "sam3.1"} else "sam3"
            status = Sam3Service.get_weight_status(model_type)
            if status.get("state") == "ready":
                return True
            model_name = "SAM3.1" if model_type == "sam3.1" else "SAM3"
            message = f"SAM3 weights missing for {model_name}"
            self.status_changed.emit(message)
            if show_dialog:
                self.error_occurred.emit(message, True)
            return False

        status = Sam2Service.get_weight_status(self._model_type)
        if status.get("state") == "ready":
            return True

        model_name = Sam2Service.SAM2_LABELS.get(self._model_type, self._model_type)
        message = self._tr("sam2_weights_missing").format(name=model_name)
        self.status_changed.emit(self._tr("sam2_weights_missing_status").format(name=model_name))
        if show_dialog:
            self.error_occurred.emit(message, True)
        return False

    def propagate_video(
        self,
        *,
        direction: str,
        all_frames: list[np.ndarray] | None,
        current_frame_index: int,
        frame_index_offset: int = 0,
        current_frame_index_global: int | None = None,
        processing_active: bool = False,
    ) -> None:
        logger.debug("[SAM] propagate_video called: direction=%s backend=%s frames=%s points=%s",
                     direction, self._backend,
                     len(all_frames) if isinstance(all_frames, list) else "None",
                     len(self.state.points))
        if self._backend != "sam2":
            self.status_changed.emit("SAM3 interactive propagation is not supported")
            return
        if processing_active:
            self.status_changed.emit(self._tr("sam_wait_processing"))
            return
        if self.generation_active:
            self.status_changed.emit(self._tr("sam_generating"))
            return
        if not self._ensure_weights_available(show_dialog=True):
            return
        if not isinstance(all_frames, list) or len(all_frames) <= 1:
            self.error_occurred.emit(self._tr("sam2_need_video_sequence"), True)
            return
        if not self.state.points:
            self.error_occurred.emit(self._tr("err_no_points"), True)
            return

        dir_norm = str(direction or "").strip().lower()
        if dir_norm not in {"forward", "backward"}:
            dir_norm = "forward"

        global_frame_index = (
            int(current_frame_index_global)
            if current_frame_index_global is not None
            else int(current_frame_index + frame_index_offset)
        )

        self.generation_active = True
        self._request_show_errors = True
        self._request_live_mode = False
        self._request_click = None
        self._request_frame_index = global_frame_index
        self.generation_started.emit()
        self.controls_busy_changed.emit(True)

        self._do_propagate.emit(
            {
                "frames": all_frames,
                "current_frame_index": int(current_frame_index),
                "current_frame_index_global": global_frame_index,
                "frame_index_offset": int(frame_index_offset),
                "direction": dir_norm,
                "points": list(self.state.points),
                "labels": list(self.state.point_labels),
            }
        )

    def reprompt_video_frame(
        self,
        *,
        current_frame: np.ndarray | None,
        all_frames: list[np.ndarray] | None,
        current_frame_index: int,
        processing_active: bool = False,
    ) -> None:
        if self._backend != "sam2":
            self.status_changed.emit("SAM3 interactive reprompt is not supported")
            return
        if processing_active:
            self.status_changed.emit(self._tr("sam_wait_processing"))
            return
        if self.generation_active:
            self.status_changed.emit(self._tr("sam_generating"))
            return
        if not self._ensure_weights_available(show_dialog=True):
            return
        if current_frame is None:
            self.status_changed.emit(self._tr("err_no_media"))
            return
        if not isinstance(all_frames, list) or len(all_frames) <= 1:
            self.status_changed.emit(self._tr("sam2_need_video_sequence"))
            return
        if not self.state.points:
            self.status_changed.emit(self._tr("err_no_points"))
            return

        self.generation_active = True
        self._request_show_errors = True
        self._request_live_mode = False
        self._request_click = None
        self._request_frame_index = int(current_frame_index)
        self.generation_started.emit()
        self.controls_busy_changed.emit(True)

        self._do_reprompt.emit(
            {
                "frames": all_frames,
                "current_frame_index": int(current_frame_index),
                "current_frame": current_frame.copy(),
                "points": list(self.state.points),
                "labels": list(self.state.point_labels),
            }
        )

    def reset_video_session(self) -> None:
        self._do_reset_video_session.emit()
        had_masks = bool(self.state.added_masks) or self.state.current_mask is not None or bool(self.state.mask_path)
        self.state.mask_path = None
        self.state.added_masks = []
        self.state.set_current_mask(None)
        if had_masks:
            self.mask_list_changed.emit()
            self.input_preview_needed.emit()
        self.status_changed.emit(self._tr("sam2_session_reset"))

    def cancel_current_operation(self) -> None:
        if not self.generation_active:
            return
        self._do_cancel.emit()
        self.status_changed.emit(self._tr("status_cancel"))

    def generate_mask(
        self,
        current_frame: np.ndarray | None,
        *,
        show_errors: bool = True,
        live_mode: bool = False,
        click_coords: tuple[int, int] | None = None,
        processing_active: bool = False,
        current_frame_index: int = 0,
        concept: str = "",
    ) -> None:
        if processing_active:
            self.status_changed.emit(self._tr("sam_wait_processing"))
            return
        if self.generation_active:
            self.status_changed.emit(self._tr("sam_generating"))
            return
        if current_frame is None:
            self.status_changed.emit(self._tr("err_no_media"))
            if show_errors:
                self.error_occurred.emit(self._tr("err_no_media"), True)
            return
        concept = str(concept or "").strip()
        if self._backend == "sam3":
            missing_prompts = not concept
            key = "err_no_sam3_prompt"
        else:
            missing_prompts = not self.state.points
            key = "err_no_points"
        if missing_prompts:
            self.status_changed.emit(self._tr(key))
            if show_errors:
                self.error_occurred.emit(self._tr(key), True)
            return
        if not self._ensure_weights_available(show_dialog=show_errors):
            return

        self.generation_active = True
        self._request_show_errors = show_errors
        self._request_live_mode = live_mode
        self._request_click = click_coords
        self._request_frame_index = int(current_frame_index)
        self.generation_started.emit()
        self.controls_busy_changed.emit(True)

        context = {
            "current_frame_index": self._request_frame_index,
            "concept": concept,
        }

        self._do_generate.emit(
            current_frame.copy(),
            [] if self._backend == "sam3" else list(self.state.points),
            [] if self._backend == "sam3" else list(self.state.point_labels),
            context,
        )

    def add_current_mask(self, frame_index: int = 0) -> bool:
        if self.state.current_mask is None:
            self.status_changed.emit(self._tr("err_no_mask"))
            return False
        self.state.add_current_mask(frame_index)
        self.mask_list_changed.emit()
        self.input_preview_needed.emit()
        self.status_changed.emit(f"{self._tr('sam_masks_count')} {len(self.state.added_masks)}")
        return True

    def remove_masks(self, rows: list[int]) -> None:
        if not rows:
            self.status_changed.emit(self._tr("sam_select_mask"))
            return
        self.state.remove_masks(rows)
        self.mask_list_changed.emit()
        self.input_preview_needed.emit()
        self.status_changed.emit(f"{self._tr('sam_masks_count')} {len(self.state.added_masks)}")

    def load_mask_file(self, file_path: str) -> None:
        if not file_path:
            return
        self.state.load_mask_path(file_path)
        self.status_changed.emit(f"{self._tr('mask_loaded')} {Path(file_path).name}")

    @staticmethod
    def is_temporary_processing_mask_path(path: str | Path | None) -> bool:
        candidate = Path(path) if path else None
        if candidate is None:
            return False
        try:
            resolved = candidate.resolve(strict=False)
            temp_root = Path(tempfile.gettempdir()).resolve(strict=False)
        except Exception:
            return False
        if resolved.parent != temp_root:
            return False
        return resolved.name in {"matanyone2_qt_mask.png", "matanyone2_qt_mask_current.png"}

    @classmethod
    def cleanup_temporary_processing_mask_path(cls, path: str | Path | None) -> None:
        if not cls.is_temporary_processing_mask_path(path):
            return
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass

    def resolve_mask_path_for_processing(self, selected_rows: list[int] | None = None) -> str | None:
        if self.state.mask_path and Path(self.state.mask_path).exists():
            return self.state.mask_path

        if self.state.added_masks:
            combined = self.state.combined_mask(selected_rows)
            if combined is not None:
                temp = Path(tempfile.gettempdir()) / "matanyone2_qt_mask.png"
                Image.fromarray(combined).save(str(temp))
                return str(temp)

        if self.state.current_mask is not None:
            temp = Path(tempfile.gettempdir()) / "matanyone2_qt_mask_current.png"
            Image.fromarray(self.state.current_mask.astype(np.uint8)).save(str(temp))
            return str(temp)

        return None

    # ── Graph sync helpers ──

    def graph_sync_dict(self, selected_mask_rows: list[int] | None = None) -> dict:
        return {
            "status_text": self.state.status_text or None,
            "backend": "sam2",
            "model_type": self._model_type,
            "point_mode": self.state.point_mode,
            "live_sam2": self.state.live_sam2,
            "mask_items": self.state.mask_items(),
            "selected_mask_rows": selected_mask_rows or [],
            "current_mask_ready": self.state.current_mask is not None,
        }

    # ── Worker callbacks (private) ──

    def _on_stage_progress(self, percent: int, status_text: str) -> None:
        if not self.generation_active:
            return
        self.progress_updated.emit(max(0, min(100, int(percent))), status_text)
        # Per-frame progress text is routed to statusbar via progress_updated only.
        # status_changed is reserved for final/meaningful state transitions
        # so that the props panel and node annotation don't rebuild on every frame.

    def _on_mask_ready(self, payload: object) -> None:
        self.generation_active = False
        sequence_applied = False
        was_cancelled = False

        if isinstance(payload, dict):
            op = str(payload.get("op") or "").strip().lower()
            if op == "generate":
                self._apply_generate_result(payload)
            elif op in {"propagate", "reprompt"}:
                sequence_applied = self._apply_sequence_result(payload)
            elif op == "cancelled":
                was_cancelled = True

        self.controls_busy_changed.emit(False)

        if self._request_live_mode:
            click = self._request_click
            self.input_preview_needed.emit()
            if click is not None:
                self.status_changed.emit(
                    f"{self._tr('sam_live_click')} ({click[0]}, {click[1]}), "
                    f"{self._tr('sam_points_count').lower()} {len(self.state.points)}"
                )
            else:
                self.status_changed.emit(self._tr("sam_live_updated"))
        else:
            self.input_preview_needed.emit()
            if was_cancelled:
                self.status_changed.emit(self._tr("status_stopped"))
            elif sequence_applied:
                self.status_changed.emit(
                    self._tr("sam2_sequence_ready").format(count=len(self.state.added_masks))
                )
            else:
                self.status_changed.emit(self._tr("sam_mask_ready"))

        self._request_click = None
        self.generation_finished.emit()
        self.mask_preview_available.emit(self.state.current_preview_mask())

    def _apply_generate_result(self, payload: dict) -> None:
        mask = payload.get("mask")
        if mask is not None:
            self.state.set_current_mask(mask)

    def _apply_sequence_result(self, payload: dict) -> bool:
        seq_map_raw = payload.get("sequence_masks_map") or {}
        if not isinstance(seq_map_raw, dict):
            return False

        by_frame: dict[int, np.ndarray] = {}
        for fi, existing in self.state.added_masks:
            by_frame[int(fi)] = np.asarray(existing, dtype=np.uint8)
        for k, v in seq_map_raw.items():
            try:
                frame_idx = int(k)
            except Exception:
                continue
            arr = np.asarray(v, dtype=np.uint8)
            if arr.ndim != 2:
                continue
            by_frame[frame_idx] = np.where(arr > 127, 255, 0).astype(np.uint8)

        if not by_frame:
            return False

        self.state.added_masks = sorted(by_frame.items(), key=lambda pair: pair[0])
        idx = int(payload.get("current_frame_index", self._request_frame_index))
        if idx in by_frame:
            self.state.set_current_mask(by_frame[idx])
        self.mask_list_changed.emit()

        tracked = int(payload.get("tracked_count", 0) or 0)
        total = int(payload.get("total_frames", 0) or 0)
        if tracked > 0 and total > 0:
            self.node_frame_progress.emit("sam2", tracked, total)
        return True

    def _on_mask_error(self, error_message: str) -> None:
        self.generation_active = False
        self._request_click = None
        self.controls_busy_changed.emit(False)
        self.generation_finished.emit()

        if self._request_show_errors:
            self.error_occurred.emit(error_message, True)

        details = (error_message or "").strip()
        if details:
            self.status_changed.emit(f"{self._tr('sam_unavailable')}: {details}")
        else:
            self.status_changed.emit(self._tr("sam_unavailable"))

    def shutdown(self) -> None:
        if self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait(3000)
