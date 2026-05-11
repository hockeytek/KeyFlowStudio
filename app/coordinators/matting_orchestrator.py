"""MatAnyone2 orchestration adapter extracted from MainWindow.

This module keeps orchestration responsibilities out of main.py:
- graph config assembly
- write stream planning and passthrough saves
- preview routing for streamed outputs
- finish/error/cancel semantics for matting runs
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtWidgets import QMessageBox

from app.utils.media import is_supported_image_file
from app.utils.write_output import resolve_write_output_format
from app.runtime_contract import (
    RUNTIME_CANCEL_IMMEDIATE,
    RUNTIME_SEMANTICS_PRODUCTION_SAFE,
    build_runtime_config,
    is_runtime_cancelled,
    runtime_partial_saved_paths,
    runtime_primary_outputs,
    runtime_saved_paths,
    tr_with_fallback,
)
from app.utils.write_paths import build_graph_write_output_dir, build_keyflow_base_dir
from .write_output_adapter import HostWriteOutputAdapter, WriteOutputAdapter

logger = logging.getLogger(__name__)


class MattingOrchestrator:
    """Adapter that orchestrates MatAnyone2 run lifecycle for MainWindow."""

    def __init__(self, host: Any, write_output_adapter: WriteOutputAdapter | None = None) -> None:
        self._host = host
        self._write_output_adapter = write_output_adapter or HostWriteOutputAdapter(host)
        self._active_run_uses_matting_node = True
        self._write_node_saved_paths: dict[str, str] = {}
        self._selected_export_preview_node_id = ""
        self._selected_birefnet_preview_node_id = ""
        self._selected_merge_preview_node_id = ""
        self._active_fg_write_node_id = ""
        self._active_alpha_write_node_id = ""
        self._pending_temp_sam_mask_path = ""
        self._sleep_guard_proc: subprocess.Popen | None = None

    def clear_write_outputs(self) -> None:
        self._write_node_saved_paths.clear()

    def _tr_status(self, matting_key: str, fallback_key: str) -> str:
        if not self._active_run_uses_matting_node:
            return self._host._tr(fallback_key)
        return tr_with_fallback(self._host._tr, matting_key, fallback_key)

    def _track_temp_sam_mask_path(self, mask_path: str) -> None:
        sam2_controller = getattr(self._host, "sam2", None)
        if sam2_controller is None or not hasattr(sam2_controller, "is_temporary_processing_mask_path"):
            self._pending_temp_sam_mask_path = ""
            return
        self._pending_temp_sam_mask_path = (
            str(mask_path or "").strip() if sam2_controller.is_temporary_processing_mask_path(mask_path) else ""
        )

    def _cleanup_pending_temp_sam_mask(self) -> None:
        pending = str(self._pending_temp_sam_mask_path or "").strip()
        if not pending:
            return
        sam2_controller = getattr(self._host, "sam2", None)
        if sam2_controller is not None and hasattr(sam2_controller, "cleanup_temporary_processing_mask_path"):
            sam2_controller.cleanup_temporary_processing_mask_path(pending)
        self._pending_temp_sam_mask_path = ""

    def _resolve_cancel_policy(self) -> str:
        # UX rule: Stop is always immediate for MatAnyone2 runtime.
        return RUNTIME_CANCEL_IMMEDIATE

    def _acquire_sleep_guard(self) -> None:
        """Prevent system sleep during long-running processing on macOS.

        Uses `caffeinate` as a best-effort guard. If unavailable or failed,
        processing continues normally.
        """
        if self._sleep_guard_proc is not None:
            self._notify_sleep_guard_state(True)
            return

        if os.uname().sysname.lower() != "darwin":
            self._notify_sleep_guard_state(False)
            return

        caffeinate_bin = shutil.which("caffeinate")
        if not caffeinate_bin:
            logger.warning("Sleep guard: `caffeinate` not found; continuing without sleep inhibition")
            self._notify_sleep_guard_state(False)
            return

        try:
            # -d: prevent display sleep, -i: prevent idle sleep,
            # -m: prevent disk sleep, -s: prevent system sleep (while AC power).
            self._sleep_guard_proc = subprocess.Popen(
                [caffeinate_bin, "-dims"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Sleep guard enabled (caffeinate pid=%s)", self._sleep_guard_proc.pid)
            self._notify_sleep_guard_state(True)
        except Exception as exc:
            self._sleep_guard_proc = None
            logger.warning("Sleep guard: failed to start caffeinate: %s", exc)
            self._notify_sleep_guard_state(False)

    def _release_sleep_guard(self) -> None:
        proc = self._sleep_guard_proc
        self._sleep_guard_proc = None
        if proc is None:
            self._notify_sleep_guard_state(False)
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        logger.info("Sleep guard disabled")
        self._notify_sleep_guard_state(False)

    def _notify_sleep_guard_state(self, active: bool) -> None:
        host_setter = getattr(self._host, "_set_sleep_guard_indicator", None)
        if callable(host_setter):
            try:
                host_setter(bool(active))
            except Exception:
                pass

    def clear_preview_selection(self) -> None:
        self._selected_export_preview_node_id = ""
        self._selected_birefnet_preview_node_id = ""
        self._selected_merge_preview_node_id = ""
        self._merge_frame_cache: dict[str, object] = {}

    def set_export_preview_node(self, node_id: str) -> None:
        self._selected_export_preview_node_id = str(node_id or "").strip()

    def set_birefnet_preview_node(self, node_id: str) -> None:
        self._selected_birefnet_preview_node_id = str(node_id or "").strip()

    def set_merge_preview_node(self, node_id: str) -> None:
        self._selected_merge_preview_node_id = str(node_id or "").strip()
        cached = self._merge_frame_cache.get(self._selected_merge_preview_node_id)
        if cached is not None:
            w = self._host
            preview_rgb = w._preview_array_to_rgb(cached)
            if preview_rgb is not None:
                w._set_selected_node_preview(frame=preview_rgb)

    def selected_export_preview_node_id(self) -> str:
        return self._selected_export_preview_node_id

    def saved_output_path_for_node(self, node_id: str) -> str:
        return self._write_node_saved_paths.get(str(node_id or "").strip(), "")

    def apply_export_preview_path(self, write_node_id: str, path: str) -> None:
        w = self._host
        node_id = str(write_node_id or "").strip()
        out_path = str(path or "").strip()
        if not node_id or not out_path or not os.path.exists(out_path):
            return

        self._write_node_saved_paths[node_id] = out_path
        if w._node_graph_dialog is not None and hasattr(w._node_graph_dialog, "set_write_last_output_path"):
            try:
                w._node_graph_dialog.set_write_last_output_path(node_id, out_path)
            except Exception as exc:
                logger.warning("Failed to persist Write last_output_path for node %s: %s", node_id, exc)
        if node_id == self._selected_export_preview_node_id:
            w._set_selected_node_preview(source=out_path)

    def _dispatch_export_preview_path(self, write_node_id: str, path: str) -> None:
        """Dispatch path update to host override when present, else apply locally.

        Avoid calling the default MainWindow delegating method to prevent recursion.
        """
        w = self._host
        hook = getattr(w, "_apply_export_preview_path", None)
        if callable(hook):
            bound_func = getattr(hook, "__func__", None)
            if bound_func is None or bound_func.__name__ != "_apply_export_preview_path":
                try:
                    hook(write_node_id, path)
                    return
                except Exception:
                    pass
        self.apply_export_preview_path(write_node_id, path)

    def start_processing(self) -> None:
        w = self._host
        self._active_run_uses_matting_node = True
        if w.sam2.generation_active:
            QMessageBox.warning(w, w._tr("status_error"), w._tr("err_wait_mask"))
            return
        if w.matting.is_active:
            return
        if not w.input_path:
            QMessageBox.warning(w, w._tr("status_error"), w._tr("err_no_file"))
            return

        if w._node_graph_dialog is not None and hasattr(w._node_graph_dialog, "has_connected_write_sink"):
            if not w._node_graph_dialog.has_connected_write_sink():
                message = w._tr("err_no_write_connection")
                w._set_status(message)
                QMessageBox.warning(w, w._tr("status_error"), message)
                return

        src = Path(w.input_path)
        output_dir = build_keyflow_base_dir(src)

        self.clear_write_outputs()
        self._selected_export_preview_node_id = ""

        effective_start, effective_end = w._resolve_effective_video_frame_bounds()
        end_frame = effective_end if w.is_video_input and effective_end > 0 else -1

        if self.try_graph_inference_run(output_dir, effective_start, end_frame):
            return

        fg_write, alpha_write, passthrough_targets, needs_mask_input = self.collect_write_targets(output_dir)

        mask_path = ""
        if needs_mask_input:
            mask_path = w._resolve_mask_path_for_processing()
            if not mask_path:
                QMessageBox.warning(w, w._tr("status_error"), w._tr("err_no_mask_for_run"))
                return
            self._track_temp_sam_mask_path(mask_path)
        else:
            self._pending_temp_sam_mask_path = ""

        passthrough_fg_path, passthrough_alpha_path = self.execute_passthrough_targets(
            passthrough_targets, mask_path, output_dir
        )

        if fg_write is None and alpha_write is None and passthrough_targets:
            self._cleanup_pending_temp_sam_mask()
            w.last_output_dir = str(Path((passthrough_fg_path or passthrough_alpha_path)).parent)
            w._show_output_preview(passthrough_fg_path, passthrough_alpha_path)
            w.ui.progress_bar.setValue(100)
            w._set_status(self._tr_status("matting_status_done", "status_done"))
            w._play_completion_sound()
            return

        config = build_runtime_config(
            erode_kernel=w._spin_value("spin_erode_kernel", 10),
            dilate_kernel=w._spin_value("spin_dilate_kernel", 10),
            is_video=w.is_video_input,
            n_warmup=w._spin_value("spin_warmup_frames", 10),
            start_frame=effective_start,
            end_frame=end_frame,
            compatibility_profile=w._compatibility_profile,
            fg_write=fg_write,
            alpha_write=alpha_write,
            cancel_policy=self._resolve_cancel_policy(),
        )
        self.start_matting_run(mask_path, output_dir, config)

    def try_graph_inference_run(self, output_dir: Path, start_frame: int, end_frame: int) -> bool:
        w = self._host
        if w._node_graph_dialog is None or not hasattr(w._node_graph_dialog, "export_graph_preset"):
            return False

        preset = w._node_graph_dialog.export_graph_preset()
        nodes_data = preset.get("nodes", []) if isinstance(preset, dict) else []
        node_types = {str(node.get("type", "")).strip().lower() for node in nodes_data if isinstance(node, dict)}
        self._active_run_uses_matting_node = "matting" in node_types
        if not any(nt in {"birefnet", "chromakey", "corridorkey", "matting", "merge", "gvm", "sam3"} for nt in node_types):
            return False

        graph_nodes: list[dict[str, Any]] = []
        for node in nodes_data:
            if not isinstance(node, dict):
                continue
            graph_nodes.append(
                {
                    "id": node.get("id"),
                    "type": node.get("type"),
                    "title": node.get("title", ""),
                    "properties": node.get("properties", {}),
                    "enabled": bool(node.get("properties", {}).get("enabled", True)),
                }
            )

        graph_edges: list[dict[str, Any]] = []
        for edge in preset.get("connections", []):
            if not isinstance(edge, dict):
                continue
            graph_edges.append(
                {
                    "src_id": edge.get("src"),
                    "dst_id": edge.get("dst"),
                    "src_port": edge.get("src_port", "out"),
                    "dst_port": edge.get("dst_port", ""),
                }
            )

        mask_path = ""
        node_type_by_id = {
            n.get("id"): str(n.get("type", "")).strip().lower()
            for n in graph_nodes
            if n.get("id") is not None
        }
        matting_ids = [
            n.get("id")
            for n in graph_nodes
            if str(n.get("type", "")).strip().lower() == "matting" and n.get("id") is not None
        ]

        needs_external_mask = False
        for matting_id in matting_ids:
            mask_inputs = [
                e
                for e in graph_edges
                if e.get("dst_id") == matting_id and str(e.get("dst_port", "")).strip().lower() == "mask"
            ]
            if not mask_inputs:
                needs_external_mask = True
                break
            # SAM source on mask port still relies on selected/uploaded mask path.
            if any(node_type_by_id.get(e.get("src_id")) in {"sam2"} for e in mask_inputs):
                needs_external_mask = True
                break

        if needs_external_mask:
            mask_path = w._resolve_mask_path_for_processing()
            if not mask_path:
                QMessageBox.warning(w, w._tr("status_error"), w._tr("err_no_mask_for_run"))
                return True
            self._track_temp_sam_mask_path(mask_path)
        else:
            self._pending_temp_sam_mask_path = ""

        connected_ids = {e["src_id"] for e in graph_edges} | {e["dst_id"] for e in graph_edges}
        missing_ck = [
            n["id"]
            for n in graph_nodes
            if str(n.get("type", "")).strip().lower() == "corridorkey"
            and n["id"] in connected_ids
            and not any(e["dst_id"] == n["id"] and e["dst_port"] == "alphahint" for e in graph_edges)
        ]
        if missing_ck:
            QMessageBox.information(
                w,
                w._tr("err_corridorkey_no_alphahint_title"),
                w._tr("err_corridorkey_no_alphahint"),
            )
            return True

        graph_selected = w.sam2_graph.selected_graph_mask_rows()
        correction_masks = w.sam2.state.get_correction_masks_by_frame(graph_selected or None)

        config = build_runtime_config(
            is_video=w.is_video_input,
            start_frame=start_frame,
            end_frame=end_frame,
            compatibility_profile=w._compatibility_profile,
            correction_masks=correction_masks or None,
            cancel_policy=self._resolve_cancel_policy(),
            node_graph={
                "nodes": graph_nodes,
                "edges": graph_edges,
            },
        )
        self.start_matting_run(mask_path, output_dir, config)
        return True

    def collect_write_targets(self, output_dir: Path) -> tuple[dict | None, dict | None, list[dict], bool]:
        w = self._host
        fg_write: dict | None = None
        alpha_write: dict | None = None
        passthrough_targets: list[dict] = []
        self._active_fg_write_node_id = ""
        self._active_alpha_write_node_id = ""
        needs_mask_input = False

        if w._node_graph_dialog is None or not hasattr(w._node_graph_dialog, "connected_write_targets"):
            return fg_write, alpha_write, passthrough_targets, needs_mask_input

        for target in w._node_graph_dialog.connected_write_targets():
            source_node = str(target.get("source_node_type", "")).strip().lower()
            stream = str(target.get("stream", "")).strip().lower()
            cfg = dict(target)

            if cfg.get("auto_output_dir") or not cfg.get("output_dir", "").strip():
                # For SAM source, prefer "sam_mask" if stream is empty
                fallback_stream = "sam_mask" if source_node in {"sam2"} else "out"
                cfg["output_dir"] = str(output_dir / (stream or fallback_stream))

            if source_node in {"sam2"}:
                needs_mask_input = True

            passthrough_targets.append(cfg)

        return fg_write, alpha_write, passthrough_targets, needs_mask_input

    def restore_write_outputs_from_disk(self) -> None:
        w = self._host
        if w._node_graph_dialog is None or not hasattr(w._node_graph_dialog, "connected_write_targets"):
            return

        targets = w._node_graph_dialog.connected_write_targets()
        if not targets:
            return

        restored_any = False
        for target in targets:
            node_id = str(target.get("graph_node_id", "")).strip()
            if not node_id:
                continue
            restored_path = self._find_existing_write_output_path(target)
            if not restored_path:
                continue
            self._dispatch_export_preview_path(node_id, restored_path)
            restored_any = True

        if restored_any:
            self._ensure_write_node_thumbnails()

    def save_sam_outputs_to_connected_write_nodes(self) -> tuple[int, int]:
        w = self._host
        if w._node_graph_dialog is None or not hasattr(w._node_graph_dialog, "connected_write_targets"):
            return 0, 0
        if not w.input_path:
            return 0, 0

        targets = [
            dict(target)
            for target in w._node_graph_dialog.connected_write_targets()
            if str(target.get("source_node_type", "")).strip().lower() in {"sam2"}
        ]
        if not targets:
            return 0, 0

        fallback_base = self._default_run_output_dir(w.input_path)
        fallback_output_dir = fallback_base if fallback_base is not None else (Path(tempfile.gettempdir()) / "keyflow_sam_out")

        frame_masks = w.sam2_graph.build_frame_masks()
        if not frame_masks:
            return 0, 0

        frame_indices = sorted(frame_masks.keys())
        frame_count = len(frame_indices)
        frames_rgb = [np.stack([frame_masks[idx]] * 3, axis=-1) for idx in frame_indices]
        default_stem = f"sam_mask_f{frame_indices[0] + 1:04d}" if frame_indices else "sam_mask"
        saved_targets = 0

        for target_cfg in targets:
            write_cfg = dict(target_cfg)
            stream = str(write_cfg.get("stream", "")).strip().lower() or "sam_mask"
            if write_cfg.get("auto_output_dir") or not str(write_cfg.get("output_dir", "")).strip():
                write_cfg["output_dir"] = str(fallback_output_dir / stream)

            try:
                out_path = self._write_output_adapter.save_frames_to_write_output(
                    frames_rgb,
                    write_cfg,
                    fallback_output_dir / "sam_mask",
                    default_stem=default_stem,
                    source_is_video=len(frames_rgb) > 1,
                    source_ext=Path(w.input_path).suffix or ".png",
                )
            except Exception as exc:
                logger.warning("SAM->Write immediate save failed for node %s: %s", write_cfg.get("graph_node_id", ""), exc)
                continue

            node_id = str(write_cfg.get("graph_node_id", "")).strip()
            if node_id and out_path:
                saved_targets += 1
                self._dispatch_export_preview_path(node_id, out_path)
                try:
                    w._node_graph_dialog.set_write_runtime_preview_for_node(
                        node_id,
                        w._to_qimage(frames_rgb[-1]),
                    )
                except Exception:
                    pass

        return saved_targets, frame_count

    def execute_passthrough_targets(
        self,
        passthrough_targets: list[dict],
        mask_path: str,
        output_dir: Path,
    ) -> tuple[str, str]:
        passthrough_fg_path = ""
        passthrough_alpha_path = ""
        
        # If we have SAM sources, skip Load passthrough to avoid duplicates
        has_sam_target = any(str(t.get("source_node_type", "")).strip().lower() in {"sam2"} for t in passthrough_targets)
        
        for target_cfg in passthrough_targets:
            source_node = str(target_cfg.get("source_node_type", "")).strip().lower()
            out_path = ""
            if source_node in {"sam2"}:
                out_path = self._write_output_adapter.save_sam_mask_output(mask_path, target_cfg, output_dir)
                if out_path and not passthrough_alpha_path:
                    passthrough_alpha_path = out_path
            elif source_node == "load" and not has_sam_target:
                # Only process Load if there's no SAM target (avoid duplicates)
                out_path = self._write_output_adapter.save_load_output(target_cfg, output_dir)
                if out_path and not passthrough_fg_path:
                    passthrough_fg_path = out_path
        return passthrough_fg_path, passthrough_alpha_path

    def start_matting_run(self, mask_path: str, output_dir: Path, config: dict) -> None:
        w = self._host
        w.ui.btn_run.setEnabled(False)
        w.ui.btn_stop.setEnabled(True)
        w.ui.progress_bar.setValue(0)
        w._set_status(self._tr_status("matting_status_start", "status_start"))
        if w._node_graph_dialog is not None and hasattr(w._node_graph_dialog, "clear_corridorkey_runtime_mode"):
            w._node_graph_dialog.clear_corridorkey_runtime_mode()
        if w._node_graph_dialog is not None and hasattr(w._node_graph_dialog, "clear_node_frame_progress"):
            w._node_graph_dialog.clear_node_frame_progress()
        self._acquire_sleep_guard()
        try:
            w.matting.start(
                w.input_path,
                mask_path,
                str(output_dir),
                config,
                sam_service=w.sam2._worker.sam_service,
            )
        except Exception:
            self._release_sleep_guard()
            raise

    def cancel_processing(self) -> None:
        w = self._host
        if w._media_loading_active:
            if w._media_loader_worker is not None:
                w._media_loader_worker.request_cancel()
            w._set_status(self._tr_status("matting_status_cancel", "status_cancel"))
            return
        if w.sam2.generation_active:
            w.sam2.cancel_current_operation()
            w._set_status(self._tr_status("matting_status_cancel", "status_cancel"))
            return
        w.matting.cancel()

    def on_matting_stage_progress(self, percent: int, status_text: str) -> None:
        w = self._host
        clamped = max(0, min(100, int(percent)))
        w.ui.progress_bar.setValue(clamped)
        w._set_status(status_text)
        if w._node_graph_dialog is not None and hasattr(w._node_graph_dialog, "set_birefnet_runtime_progress"):
            text = str(status_text or "")
            if "birefnet" in text.lower():
                w._node_graph_dialog.set_birefnet_runtime_progress(clamped, text)

    def on_node_frame_progress(self, node_type: str, current: int, total: int) -> None:
        w = self._host
        if w._node_graph_dialog is not None and hasattr(w._node_graph_dialog, "set_node_frame_progress"):
            w._node_graph_dialog.set_node_frame_progress(node_type, current, total)

        total_i = int(total) if total is not None else 0
        current_i = int(current) if current is not None else 0
        if total_i <= 0:
            return

        # Keep progress moving during graph execution (instead of staying on early stage values).
        ratio = max(0.0, min(1.0, float(current_i) / float(total_i)))
        percent = 20 + int(ratio * 72)  # 20..92 reserved runtime range
        try:
            current_bar = int(w.ui.progress_bar.value())
        except Exception:
            current_bar = 0
        w.ui.progress_bar.setValue(max(current_bar, max(20, min(92, percent))))

        node_key = str(node_type or "").strip().lower()
        labels = {
            "sam2": "SAM2",
            "matting": "MatAnyone2",
            "birefnet": "BiRefNet",
            "gvm": "GVM",
            "corridorkey": "CorridorKey",
            "chromakey": "ChromaKey",
            "alpha": "Alpha",
            "source": "Source",
            "load": "Load",
            "export": "Write",
        }
        node_label = labels.get(node_key, node_type.upper() if isinstance(node_type, str) else "Node")
        w._set_status(f"{node_label}: {current_i}/{total_i}")

    def on_matting_frame_progress(self, current: int, total: int) -> None:
        w = self._host
        if total > 0:
            percent = 18 + int((current / total) * 72)
        else:
            percent = 18
        w.ui.progress_bar.setValue(max(18, min(90, percent)))
        w._set_status(f"{self._tr_status('matting_status_frame', 'status_frame')} {current}/{total}")

    def on_matting_frame_preview(self, foreground_rgb, alpha_rgb, _frame_index: int) -> None:
        w = self._host
        if w._node_graph_dialog is None or not hasattr(w._node_graph_dialog, "set_write_runtime_preview_for_node"):
            return
        try:
            fg_q = w._to_qimage(np.asarray(foreground_rgb, dtype=np.uint8)) if foreground_rgb is not None else None
            a_q = w._to_qimage(np.asarray(alpha_rgb, dtype=np.uint8)) if alpha_rgb is not None else None
            if self._active_fg_write_node_id:
                w._node_graph_dialog.set_write_runtime_preview_for_node(self._active_fg_write_node_id, fg_q)
            if self._active_alpha_write_node_id:
                w._node_graph_dialog.set_write_runtime_preview_for_node(self._active_alpha_write_node_id, a_q)
        except Exception:
            pass

    def on_graph_stream_preview(self, write_node_id: str, preview_frame, _frame_index: int) -> None:
        w = self._host
        if w._node_graph_dialog is None or not hasattr(w._node_graph_dialog, "set_write_runtime_preview_for_node"):
            return

        stream = ""
        saved_path = ""
        frame_for_preview = preview_frame
        if isinstance(preview_frame, dict):
            frame_for_preview = preview_frame.get("frame")
            stream = str(preview_frame.get("stream", "")).strip().lower()
            saved_path = str(preview_frame.get("path", "")).strip()
            semantics = str(preview_frame.get("semantics", "")).strip().lower()
        else:
            semantics = ""

        qimage = None
        if saved_path and os.path.exists(saved_path) and is_supported_image_file(saved_path):
            try:
                qimage = w._to_qimage(w._load_image_for_preview(saved_path))
            except Exception:
                qimage = None
            if semantics in {"", RUNTIME_SEMANTICS_PRODUCTION_SAFE}:
                self._dispatch_export_preview_path(write_node_id, saved_path)

        if qimage is None:
            qimage = self._preview_array_to_qimage(frame_for_preview)

        if saved_path and os.path.exists(saved_path) and semantics in {"", RUNTIME_SEMANTICS_PRODUCTION_SAFE}:
            self._dispatch_export_preview_path(write_node_id, saved_path)

        if write_node_id and str(write_node_id).strip() == self._selected_birefnet_preview_node_id:
            if stream == "alpha":
                logger.debug(
                    "BiRefNet preview streaming: write_node_id=%s, stream=%s",
                    write_node_id,
                    stream,
                )
                preview_rgb = w._preview_array_to_rgb(frame_for_preview)
                if preview_rgb is not None:
                    w._set_selected_node_preview(frame=preview_rgb)
                return

        if write_node_id and str(write_node_id).strip() == self._selected_merge_preview_node_id:
            if stream == "out":
                self._merge_frame_cache[str(write_node_id).strip()] = frame_for_preview
                preview_rgb = w._preview_array_to_rgb(frame_for_preview)
                if preview_rgb is not None:
                    w._set_selected_node_preview(frame=preview_rgb)
                return

        # Always cache the last merge "out" frame so clicking after processing works.
        if write_node_id and stream == "out" and frame_for_preview is not None:
            self._merge_frame_cache[str(write_node_id).strip()] = frame_for_preview


        if write_node_id and str(write_node_id).strip() == self._selected_export_preview_node_id:
            preview_rgb = w._preview_array_to_rgb(frame_for_preview)
            if preview_rgb is not None:
                w._set_selected_node_preview(frame=preview_rgb)

        if qimage is None:
            return
        w._node_graph_dialog.set_write_runtime_preview_for_node(write_node_id, qimage)

    def on_matting_finished(self, result: dict) -> None:
        w = self._host
        self._release_sleep_guard()
        w.ui.btn_run.setEnabled(True)
        w._refresh_stop_button_state()
        self._cleanup_pending_temp_sam_mask()
        if w._node_graph_dialog is not None and hasattr(w._node_graph_dialog, "clear_birefnet_runtime_progress"):
            w._node_graph_dialog.clear_birefnet_runtime_progress()

        if is_runtime_cancelled(result):
            partial_saved_paths = runtime_partial_saved_paths(result)
            if partial_saved_paths:
                for node_id, path in partial_saved_paths.items():
                    if path and os.path.exists(path):
                        self._write_node_saved_paths[node_id] = path

                fgr_path = ""
                alpha_path = ""
                if w._node_graph_dialog is not None and hasattr(w._node_graph_dialog, "write_node_ids_for_stream"):
                    fg_ids = set(w._node_graph_dialog.write_node_ids_for_stream("fg"))
                    alpha_ids = set(w._node_graph_dialog.write_node_ids_for_stream("alpha"))
                    for nid, p in partial_saved_paths.items():
                        if not p or not os.path.exists(p):
                            continue
                        if nid in fg_ids and not fgr_path:
                            fgr_path = p
                        elif nid in alpha_ids and not alpha_path:
                            alpha_path = p
                if not fgr_path and not alpha_path:
                    all_paths = [p for p in partial_saved_paths.values() if p and os.path.exists(p)]
                    fgr_path = all_paths[0] if len(all_paths) >= 1 else ""
                    alpha_path = all_paths[1] if len(all_paths) >= 2 else ""

                output_ref = fgr_path or alpha_path
                w.last_output_dir = str(Path(output_ref).parent) if output_ref else None
                w._show_output_preview(fgr_path, alpha_path)
                self._ensure_write_node_thumbnails()
                w.ui.progress_bar.setValue(100)
                w._set_status(self._tr_status("matting_status_stopped_partial", "status_stopped"))
            else:
                w.ui.progress_bar.setValue(0)
                w._set_status(self._tr_status("matting_status_stopped", "status_stopped"))
            if w._node_graph_dialog is not None and hasattr(w._node_graph_dialog, "clear_node_frame_progress"):
                w._node_graph_dialog.clear_node_frame_progress()
        else:
            saved_paths = runtime_saved_paths(result)
            if saved_paths:
                for node_id, path in saved_paths.items():
                    if path and os.path.exists(path):
                        self._write_node_saved_paths[node_id] = path

                fgr_path = ""
                alpha_path = ""
                if w._node_graph_dialog is not None and hasattr(w._node_graph_dialog, "write_node_ids_for_stream"):
                    fg_ids = set(w._node_graph_dialog.write_node_ids_for_stream("fg"))
                    alpha_ids = set(w._node_graph_dialog.write_node_ids_for_stream("alpha"))
                    for nid, p in saved_paths.items():
                        if not p or not os.path.exists(p):
                            continue
                        if nid in fg_ids and not fgr_path:
                            fgr_path = p
                        elif nid in alpha_ids and not alpha_path:
                            alpha_path = p
                if not fgr_path and not alpha_path:
                    all_paths = [p for p in saved_paths.values() if p and os.path.exists(p)]
                    fgr_path = all_paths[0] if len(all_paths) >= 1 else ""
                    alpha_path = all_paths[1] if len(all_paths) >= 2 else ""
            else:
                fgr_path, alpha_path = runtime_primary_outputs(result)
                if fgr_path and os.path.exists(fgr_path) and self._active_fg_write_node_id:
                    self._write_node_saved_paths[self._active_fg_write_node_id] = fgr_path
                if alpha_path and os.path.exists(alpha_path) and self._active_alpha_write_node_id:
                    self._write_node_saved_paths[self._active_alpha_write_node_id] = alpha_path

            output_ref = fgr_path or alpha_path
            w.last_output_dir = str(Path(output_ref).parent) if output_ref else None
            w._show_output_preview(fgr_path, alpha_path)
            self._ensure_write_node_thumbnails()
            w.ui.progress_bar.setValue(100)
            w._set_status(self._tr_status("matting_status_done", "status_done"))
            w._play_completion_sound()

        if not w.sam2.generation_active:
            w._set_sam_controls_busy(False)

    def on_matting_error(self, error_message: str) -> None:
        w = self._host
        self._release_sleep_guard()
        w.ui.btn_run.setEnabled(True)
        w._refresh_stop_button_state()
        self._cleanup_pending_temp_sam_mask()
        w.ui.progress_bar.setValue(0)
        w._set_status(self._tr_status("matting_status_error", "status_error"))
        logger.error("Matting error: %s", error_message)
        if w._node_graph_dialog is not None and hasattr(w._node_graph_dialog, "clear_birefnet_runtime_progress"):
            w._node_graph_dialog.clear_birefnet_runtime_progress()
        QMessageBox.warning(w, w._tr("inference_error_title"), error_message)

        if not w.sam2.generation_active:
            w._set_sam_controls_busy(False)

    def on_matting_busy_changed(self, busy: bool) -> None:
        w = self._host
        if not busy:
            # Safety net: ensure sleep guard is released if runtime exits unexpectedly.
            self._release_sleep_guard()
        if not busy and not w.sam2.generation_active:
            w._set_sam_controls_busy(False)

    def on_matting_log_message(self, message: str) -> None:
        text = str(message or "").strip()
        if text:
            logger.info("%s", text)

    def on_corridorkey_mode_resolved(self, requested_mode: str, effective_mode: str, _reason_key: str) -> None:
        w = self._host
        if w._node_graph_dialog is None or not hasattr(w._node_graph_dialog, "set_corridorkey_runtime_mode"):
            return
        w._node_graph_dialog.set_corridorkey_runtime_mode(requested_mode, effective_mode)

    def _default_run_output_dir(self, source_hint: str = "") -> Path | None:
        w = self._host
        host_override = getattr(w, "_default_run_output_dir", None)
        if callable(host_override):
            try:
                resolved = host_override(source_hint)
            except Exception:
                resolved = None
            if resolved is not None:
                return Path(resolved)
        src = str(source_hint or w.input_path or "").strip()
        if not src:
            return None
        src_path = Path(src)
        return build_keyflow_base_dir(src_path)

    @staticmethod
    def _resolve_write_output_format_for_restore(target_cfg: dict, source_path: Path | None) -> str:
        return resolve_write_output_format(target_cfg, source_path or Path("input.png"))

    def _find_existing_write_output_path(self, target_cfg: dict) -> str:
        w = self._host
        stream = str(target_cfg.get("stream", "")).strip().lower() or "out"
        auto_output_dir = bool(target_cfg.get("auto_output_dir", True))
        custom_output_dir = str(target_cfg.get("output_dir", "")).strip()
        node_id = str(target_cfg.get("graph_node_id", "")).strip()
        source_hint = str(w.input_path or target_cfg.get("source_path", "")).strip()
        remembered_path = str(target_cfg.get("last_output_path", "")).strip()

        if remembered_path and os.path.exists(remembered_path):
            return remembered_path

        base_default_dir = self._default_run_output_dir(source_hint)
        out_dir = None
        resolved_output_dir = str(target_cfg.get("resolved_output_dir", "")).strip()
        if resolved_output_dir:
            out_dir = Path(resolved_output_dir)
        else:
            if auto_output_dir or not custom_output_dir:
                if base_default_dir is not None:
                    source_node_title = str(target_cfg.get("source_node_title", "")).strip()
                    port_label = str(target_cfg.get("port_label", "")).strip()
                    out_dir = build_graph_write_output_dir(
                        base_default_dir,
                        source_node_title=source_node_title,
                        port_label=port_label,
                        stream_label=stream,
                    )
                else:
                    out_dir = None
            else:
                out_dir = Path(custom_output_dir)

        if out_dir is None or not out_dir.exists() or not out_dir.is_dir():
            return ""

        source_path = Path(source_hint) if source_hint else None
        output_fmt = self._resolve_write_output_format_for_restore(target_cfg, source_path)
        video_exts = {"mp4", "mov", "avi", "mkv", "webm", "m4v"}
        img_ext = ".exr" if output_fmt == "exr" else (".jpg" if output_fmt in {"jpg", "jpeg"} else ".png")

        stem = str(target_cfg.get("file_name", "")).strip()
        if not stem:
            if source_path is not None:
                stem = source_path.stem or "result"
            else:
                stem = "result"

        direct_candidates: list[Path] = []
        if output_fmt in video_exts:
            direct_candidates.append(out_dir / f"{stem}.{output_fmt}")
        else:
            direct_candidates.append(out_dir / f"{stem}{img_ext}")
            direct_candidates.append(out_dir / f"0001{img_ext}")
            direct_candidates.append(out_dir / stem / f"0001{img_ext}")

        for candidate in direct_candidates:
            if candidate.exists():
                return str(candidate)

        if output_fmt in video_exts:
            candidates = sorted(out_dir.glob(f"*.{output_fmt}"))
            if candidates:
                return str(candidates[0])
        else:
            frame_candidates = sorted(out_dir.glob(f"*{img_ext}"))
            if frame_candidates:
                return str(frame_candidates[0])

            nested = sorted(out_dir.glob(f"*/0001{img_ext}"))
            if nested:
                return str(nested[0])

        if node_id:
            logger.info("Write restore: no existing output found for node %s in %s", node_id, out_dir)
        return ""

    def resolve_write_output_path(self, target_cfg: dict) -> str:
        return self._find_existing_write_output_path(target_cfg)

    def _preview_array_to_qimage(self, frame):
        w = self._host
        arr = w._preview_array_to_rgb(frame)
        if arr is None:
            return None
        return w._to_qimage(arr)

    def _ensure_write_node_thumbnails(self) -> None:
        w = self._host
        if w._node_graph_dialog is None or not hasattr(w._node_graph_dialog, "set_write_runtime_preview_for_node"):
            return
        for node_id, path in self._write_node_saved_paths.items():
            try:
                if not os.path.exists(path):
                    continue
                frame_rgb = w._load_preview_image_or_video_frame(path)
                if frame_rgb is None:
                    continue
                qimage = w._to_qimage(frame_rgb)
                w._node_graph_dialog.set_write_runtime_preview_for_node(node_id, qimage)
            except Exception as exc:
                logger.warning("_ensure_write_node_thumbnails: node %s path %s: %s", node_id, path, exc)
