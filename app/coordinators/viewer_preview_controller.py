"""Viewer/split preview orchestration extracted from MainWindow."""

from __future__ import annotations

import logging
import os
import zipfile
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSignalBlocker, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap

from app.utils.media import (
    load_image_float,
    is_supported_image_file,
    load_rgb_image,
    resolve_numbered_image_sequence,
)

logger = logging.getLogger(__name__)


class ViewerPreviewController:
    """Owns split/viewer preview logic while MainWindow stays thin."""

    def __init__(self, host) -> None:
        self._host = host

    def _simplify_output_preview_tools(self) -> None:
        self._set_output_display_transform_controls_enabled(False)

    def _setup_output_display_transform_buttons(self) -> None:
        w = self._host
        gamma_icon = w._app_assets_dir / "preview-gamma-display.svg"
        linear_icon = w._app_assets_dir / "preview-gamma-linear.svg"

        w.ui.btn_preview_foreground.setIcon(QIcon(gamma_icon.as_posix()))
        w.ui.btn_preview_alpha.setIcon(QIcon(linear_icon.as_posix()))
        w.ui.btn_preview_foreground.setIconSize(QSize(14, 14))
        w.ui.btn_preview_alpha.setIconSize(QSize(14, 14))
        w.ui.btn_preview_foreground.clicked.connect(
            lambda _checked=False: w._set_output_display_transform("display_gamma")
        )
        w.ui.btn_preview_alpha.clicked.connect(
            lambda _checked=False: w._set_output_display_transform("linear")
        )
        self._sync_output_display_transform_buttons()

    def _set_output_display_transform_controls_enabled(self, enabled: bool) -> None:
        w = self._host
        w.ui.btn_preview_foreground.setEnabled(enabled)
        w.ui.btn_preview_alpha.setEnabled(enabled)

    def _sync_output_display_transform_buttons(self) -> None:
        w = self._host
        gamma_checked = w._output_display_transform == "display_gamma"
        with QSignalBlocker(w.ui.btn_preview_foreground):
            w.ui.btn_preview_foreground.setChecked(gamma_checked)
        with QSignalBlocker(w.ui.btn_preview_alpha):
            w.ui.btn_preview_alpha.setChecked(not gamma_checked)

    def _set_output_display_transform(self, mode: str) -> None:
        w = self._host
        normalized = str(mode or "").strip().lower()
        if normalized not in {"display_gamma", "linear"}:
            return

        w._output_display_transform = normalized
        self._sync_output_display_transform_buttons()
        w._selected_node_frame_cache.clear()
        if self._has_selected_node_preview():
            w._render_output_preview_for_index(w.current_frame_index)

    def _setup_split_view_button(self) -> None:
        w = self._host
        split_icon = w._app_assets_dir / "preview-split.svg"
        w.ui.btn_split_view.setIcon(QIcon(split_icon.as_posix()))
        w.ui.btn_split_view.setIconSize(QSize(14, 14))
        w.ui.btn_split_view.setToolTip(
            w._format_tooltip(
                w._tr("split_view_tooltip")
            )
        )
        w.ui.btn_split_view.toggled.connect(w._on_split_view_toggled)

        # Wire output-label mouse events for split drag directly in the controller
        # so main.py does not need to know about split-view internals.
        w.ui.output_video_label.mousePressEvent = self._on_output_label_mouse_press
        w.ui.output_video_label.mouseMoveEvent = self._on_output_label_mouse_move
        w.ui.output_video_label.mouseReleaseEvent = self._on_output_label_mouse_release

    def _on_split_view_toggled(self, checked: bool) -> None:
        w = self._host
        w._split_view_enabled = checked
        if not checked:
            w._split_view_dragging = False
        w._render_output_preview_for_index(w.current_frame_index)

    def _on_output_label_mouse_press(self, event) -> None:
        w = self._host
        if w._split_view_enabled:
            w._split_view_dragging = True
            w._update_split_from_mouse(event.position().x())

    def _on_output_label_mouse_move(self, event) -> None:
        w = self._host
        if w._split_view_dragging and w._split_view_enabled:
            w._update_split_from_mouse(event.position().x())

    def _on_output_label_mouse_release(self, _event) -> None:
        self._host._split_view_dragging = False

    def _update_split_from_mouse(self, click_x: float) -> None:
        w = self._host
        if w.current_frame is None:
            return
        img_h, img_w = w.current_frame.shape[:2]
        lbl = w.ui.output_video_label
        lw = max(1, lbl.width())
        lh = max(1, lbl.height())
        scale = min(lw / img_w, lh / img_h)
        disp_w = img_w * scale
        off_x = (lw - disp_w) / 2
        ratio = (click_x - off_x) / max(1.0, disp_w)
        new_ratio = max(0.0, min(1.0, ratio))
        # Skip microscopic updates to keep drag responsive and avoid jitter.
        if abs(new_ratio - float(w._split_x_ratio)) < 0.001:
            return
        w._split_x_ratio = new_ratio
        w._render_output_preview_for_index(w.current_frame_index)

    def _preview_array_to_rgb(self, frame, *, apply_display_gamma: bool | None = None) -> np.ndarray | None:
        w = self._host
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        else:
            return None

        if arr.dtype != np.uint8:
            arr_f = arr.astype(np.float32)
            min_val = float(np.nanmin(arr_f)) if arr_f.size else 0.0
            max_val = float(np.nanmax(arr_f)) if arr_f.size else 0.0
            if min_val >= -0.5 and max_val <= 1.5:
                should_apply_gamma = w._output_display_transform == "display_gamma"
                if apply_display_gamma is not None:
                    should_apply_gamma = bool(apply_display_gamma)

                arr_f = np.clip(arr_f, 0.0, 1.0)
                if should_apply_gamma:
                    arr_f = np.power(arr_f, 1.0 / 2.2)
                arr = (arr_f * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr_f, 0.0, 255.0).astype(np.uint8)

        return np.asarray(arr, dtype=np.uint8)

    def _load_image_preview_source(self, path: str) -> np.ndarray:
        return load_image_float(path)

    def _load_image_for_preview(self, path: str) -> np.ndarray:
        source_frame = self._load_image_preview_source(path)
        preview_frame = self._preview_array_to_rgb(source_frame)
        if preview_frame is None:
            raise RuntimeError(f"Failed to prepare preview image: {path}")
        return preview_frame

    def _load_media_frame_for_index(self, path: str, media_type: str, frame_index: int) -> np.ndarray | None:
        media_path = str(path or "").strip()
        if not media_path or not os.path.exists(media_path):
            return None

        idx = max(0, int(frame_index))
        media_type = str(media_type or "video").strip().lower()

        try:
            if media_type == "image" and not resolve_numbered_image_sequence(media_path):
                return self._load_image_preview_source(media_path)
        except Exception:
            pass

        try:
            seq = resolve_numbered_image_sequence(media_path)
            if seq:
                safe_idx = max(0, min(len(seq) - 1, idx))
                return self._load_image_preview_source(seq[safe_idx])
        except Exception:
            pass

        try:
            cap = cv2.VideoCapture(media_path)
            if not cap.isOpened():
                cap.release()
                return None
            if idx > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    def _try_show_merge_quick_preview(self, payload: object) -> bool:
        w = self._host
        if not isinstance(payload, dict):
            return False

        quick = payload.get("_quick_preview")
        if not isinstance(quick, dict):
            return False

        fg_meta = quick.get("fg")
        bg_meta = quick.get("bg")
        mask_meta = quick.get("mask")
        if not isinstance(fg_meta, dict) or not isinstance(bg_meta, dict):
            return False

        fg_path = str(fg_meta.get("path", "")).strip()
        bg_path = str(bg_meta.get("path", "")).strip()
        fg_media_type = str(fg_meta.get("media_type", "video")).strip().lower()
        bg_media_type = str(bg_meta.get("media_type", "video")).strip().lower()

        if str(fg_meta.get("node_type", "")).strip().lower() == "source" and not fg_path:
            fg_path = str(w.input_path or "").strip()
        if str(bg_meta.get("node_type", "")).strip().lower() == "source" and not bg_path:
            bg_path = str(w.input_path or "").strip()

        frame_idx = int(getattr(w, "current_frame_index", 0) or 0)
        fg_frame = self._load_media_frame_for_index(fg_path, fg_media_type, frame_idx)
        bg_frame = self._load_media_frame_for_index(bg_path, bg_media_type, frame_idx)
        if fg_frame is None or bg_frame is None:
            return False

        mask_frame = None
        if isinstance(mask_meta, dict):
            mask_path = str(mask_meta.get("path", "")).strip()
            mask_media_type = str(mask_meta.get("media_type", "video")).strip().lower()
            mask_frame = self._load_media_frame_for_index(mask_path, mask_media_type, frame_idx)

        mode = str(payload.get("mode", "over")).strip().lower()
        set_bbox_to = str(payload.get("set_bbox_to", "union")).strip().lower()
        opacity = float(payload.get("opacity", 1.0))
        mix = float(payload.get("mix", 1.0))
        mask_enabled = bool(payload.get("mask_enabled", True))
        mask_channel = str(payload.get("mask_channel", "auto")).strip().lower()
        mask_inject = bool(payload.get("mask_inject", False))
        invert_mask = bool(payload.get("invert_mask", False))
        fringe = bool(payload.get("fringe", False))
        alpha_masking = bool(payload.get("alpha_masking", True))

        try:
            from app.workers.inference_worker import InferenceWorker

            merged = InferenceWorker._apply_merge_blend(
                fg_frame,
                bg_frame,
                mode=mode,
                opacity=opacity,
                mask=mask_frame,
                mix=mix,
                mask_enabled=mask_enabled,
                mask_channel=mask_channel,
                mask_inject=mask_inject,
                invert_mask=invert_mask,
                fringe=fringe,
                alpha_masking=alpha_masking,
            )
            bbox_a = InferenceWorker._frame_bbox(fg_frame)
            bbox_b = InferenceWorker._frame_bbox(bg_frame)
            if set_bbox_to == "intersection":
                out_bbox = InferenceWorker._bbox_intersection(bbox_a, bbox_b)
            elif set_bbox_to == "a":
                out_bbox = bbox_a
            elif set_bbox_to == "b":
                out_bbox = bbox_b
            else:
                out_bbox = InferenceWorker._bbox_union(bbox_a, bbox_b)
            merged = InferenceWorker._clip_frame_to_bbox(merged, out_bbox)
        except Exception as exc:
            logger.debug("Merge quick preview failed: %s", exc)
            return False

        preview_rgb = self._preview_array_to_rgb(merged)
        if preview_rgb is None:
            return False

        self._set_selected_node_preview(frame=preview_rgb)
        return True

    def _clear_selected_node_preview(self) -> None:
        w = self._host
        self._reset_selected_node_preview_source()
        self._set_output_display_transform_controls_enabled(False)
        w.ui.btn_split_view.setEnabled(False)
        w.ui.btn_split_view.setChecked(False)
        w._split_view_enabled = False
        w._split_view_dragging = False
        w.ui.output_video_label.clear()
        w.ui.output_video_label.setText(w._output_preview_placeholder_text)

    def _set_selected_node_preview(self, *, source: str = "", frame=None) -> None:
        w = self._host
        w._selected_node_live_stream_mode = False
        preview_frame = np.asarray(frame).copy() if frame is not None else None
        preview_source = str(source or "").strip()

        if preview_frame is None and (not preview_source or not os.path.exists(preview_source)):
            self._clear_selected_node_preview()
            return

        self._prepare_selected_node_preview_source(
            preview_frame if preview_frame is not None else preview_source
        )
        self._set_output_display_transform_controls_enabled(True)
        w.ui.btn_split_view.setEnabled(bool(w.all_frames))
        w._render_output_preview_for_index(w.current_frame_index)

    def _reset_selected_node_preview_source(self) -> None:
        w = self._host
        selected_node_video_cap = getattr(w, "_selected_node_video_cap", None)
        if selected_node_video_cap is not None:
            selected_node_video_cap.release()
        w._selected_node_video_cap = None
        w._selected_node_preview_path = None
        w._selected_node_preview_is_image = False
        w._selected_node_preview_image = None
        w._selected_node_preview_sequence_paths = []
        w._selected_node_video_frame_count = 0
        w._selected_node_uses_main_frames = False
        if not hasattr(w, "_selected_node_frame_cache") or w._selected_node_frame_cache is None:
            w._selected_node_frame_cache = OrderedDict()
        w._selected_node_frame_cache.clear()
        w._selected_node_live_stream_mode = False

    def _prepare_selected_node_preview_source(self, source) -> None:
        w = self._host
        self._reset_selected_node_preview_source()

        if isinstance(source, np.ndarray):
            w._selected_node_preview_is_image = True
            w._selected_node_preview_image = np.asarray(source).copy()
            w._selected_node_video_frame_count = 1
            return

        path = str(source or "").strip()
        if not path or not os.path.exists(path):
            return

        w._selected_node_preview_path = path
        if is_supported_image_file(path):
            seq_paths = self._resolve_output_image_sequence(path)
            if len(seq_paths) > 1:
                w._selected_node_preview_sequence_paths = seq_paths
                w._selected_node_video_frame_count = len(seq_paths)
            else:
                w._selected_node_preview_is_image = True
                w._selected_node_preview_image = self._load_image_preview_source(path)
                w._selected_node_video_frame_count = 1
            return

        # Optimisation: if the path matches the already-loaded input and
        # all_frames is populated, skip the second VideoCapture entirely and
        # read directly from RAM on every frame change.
        input_path = str(getattr(w, "input_path", "") or "").strip()
        main_frames = getattr(w, "all_frames", None) or []
        if input_path and path == input_path and main_frames:
            w._selected_node_uses_main_frames = True
            w._selected_node_video_frame_count = len(main_frames)
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return

        w._selected_node_video_cap = cap
        w._selected_node_video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def _has_selected_node_preview(self) -> bool:
        w = self._host
        return bool(
            getattr(w, "_selected_node_uses_main_frames", False)
            or w._selected_node_preview_is_image
            or w._selected_node_preview_path
            or w._selected_node_preview_sequence_paths
            or w._selected_node_video_cap is not None
        )

    def _load_selected_node_frame_by_index(self, idx: int):
        w = self._host

        # Fast path: Source node preview == input already in RAM → zero disk I/O.
        if getattr(w, "_selected_node_uses_main_frames", False):
            main_frames = getattr(w, "all_frames", None) or []
            if main_frames:
                safe_idx = max(0, min(len(main_frames) - 1, idx))
                return self._preview_array_to_rgb(main_frames[safe_idx])

        if idx in w._selected_node_frame_cache:
            source_frame = w._selected_node_frame_cache.pop(idx)
            w._selected_node_frame_cache[idx] = source_frame
            return self._preview_array_to_rgb(source_frame)

        if w._selected_node_preview_path and is_supported_image_file(w._selected_node_preview_path):
            seq_paths_live = self._resolve_output_image_sequence(w._selected_node_preview_path)
            if len(seq_paths_live) > 1:
                if w._selected_node_preview_sequence_paths != seq_paths_live:
                    w._selected_node_preview_sequence_paths = seq_paths_live
                    w._selected_node_preview_is_image = False
                    w._selected_node_preview_image = None
                    w._selected_node_video_frame_count = len(seq_paths_live)

        if w._selected_node_live_stream_mode and not w._selected_node_preview_sequence_paths:
            if w._selected_node_frame_cache:
                keys = list(w._selected_node_frame_cache.keys())
                nearest_idx = min(keys, key=lambda key: abs(int(key) - int(idx)))
                source_frame = w._selected_node_frame_cache.pop(nearest_idx)
                w._selected_node_frame_cache[nearest_idx] = source_frame
                return self._preview_array_to_rgb(source_frame)
            if w._selected_node_preview_is_image or w._selected_node_video_cap is not None:
                return None

        if w._selected_node_preview_sequence_paths:
            n = len(w._selected_node_preview_sequence_paths)
            safe_idx = max(0, min(n - 1, idx)) if n > 0 else 0
            if safe_idx in w._selected_node_frame_cache:
                source_frame = w._selected_node_frame_cache.pop(safe_idx)
                w._selected_node_frame_cache[safe_idx] = source_frame
                return self._preview_array_to_rgb(source_frame)
            try:
                source_frame = self._load_image_preview_source(w._selected_node_preview_sequence_paths[safe_idx])
            except Exception:
                return None
            w._selected_node_frame_cache[safe_idx] = source_frame
            if len(w._selected_node_frame_cache) > w._selected_node_frame_cache_size:
                w._selected_node_frame_cache.popitem(last=False)
            return self._preview_array_to_rgb(source_frame)

        if w._selected_node_preview_is_image:
            return self._preview_array_to_rgb(w._selected_node_preview_image)

        if w._selected_node_video_cap is None:
            return None

        if w._selected_node_video_frame_count > 0:
            safe_idx = max(0, min(w._selected_node_video_frame_count - 1, idx))
        else:
            safe_idx = max(0, idx)

        if safe_idx in w._selected_node_frame_cache:
            source_frame = w._selected_node_frame_cache.pop(safe_idx)
            w._selected_node_frame_cache[safe_idx] = source_frame
            return self._preview_array_to_rgb(source_frame)

        w._selected_node_video_cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
        ret, frame = w._selected_node_video_cap.read()
        if not ret:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        w._selected_node_frame_cache[safe_idx] = frame_rgb
        if len(w._selected_node_frame_cache) > w._selected_node_frame_cache_size:
            w._selected_node_frame_cache.popitem(last=False)
        return self._preview_array_to_rgb(frame_rgb)

    def _on_graph_preview_request_changed(self, node_type: str, payload: object) -> None:
        w = self._host
        if not node_type:
            w._ensure_matting_orchestrator().clear_preview_selection()
            self._clear_selected_node_preview()
            return

        if node_type not in {"export", "birefnet", "merge"}:
            w._ensure_matting_orchestrator().clear_preview_selection()

        if node_type in {"sam2"}:
            w._show_mask_preview_on_output(w.sam2.state.mask_for_frame(w.current_frame_index))
            return

        if node_type == "source":
            node_path = str(payload.get("path", "")).strip() if isinstance(payload, dict) else ""
            source_path = node_path or str(w.input_path or "").strip()
            if source_path and os.path.exists(source_path):
                self._set_selected_node_preview(source=source_path)
            else:
                self._clear_selected_node_preview()
            return

        if node_type in {"load", "alpha"}:
            node_path = str(payload.get("path", "")).strip() if isinstance(payload, dict) else ""
            if node_path and os.path.exists(node_path):
                self._set_selected_node_preview(source=node_path)
            else:
                self._clear_selected_node_preview()
            return

        if node_type == "export":
            node_id = str(payload.get("graph_node_id", "")).strip() if isinstance(payload, dict) else ""
            orchestrator = w._ensure_matting_orchestrator()
            orchestrator.set_export_preview_node(node_id)
            selected_path = orchestrator.saved_output_path_for_node(node_id)
            if (not selected_path or not os.path.exists(selected_path)) and isinstance(payload, dict):
                remembered_path = str(payload.get("last_output_path", "")).strip()
                if remembered_path and os.path.exists(remembered_path):
                    selected_path = remembered_path
                elif (
                    w._node_graph_dialog is not None
                    and hasattr(w._node_graph_dialog, "connected_write_targets")
                ):
                    for target in w._node_graph_dialog.connected_write_targets():
                        if str(target.get("graph_node_id", "")).strip() != node_id:
                            continue
                        selected_path = orchestrator.resolve_write_output_path(target)
                        if selected_path and os.path.exists(selected_path):
                            try:
                                self._apply_export_preview_path(node_id, selected_path)
                            except Exception:
                                pass
                        break
            self._set_selected_node_preview(source=selected_path)
            return

        if node_type == "birefnet":
            node_id = str(payload.get("graph_node_id", "")).strip() if isinstance(payload, dict) else ""
            w._ensure_matting_orchestrator().set_birefnet_preview_node(node_id)
            logger.debug("BiRefNet preview request: node_id=%s, payload=%s", node_id, payload)
            w._set_status(w._tr("status_ready"))
            return

        if node_type == "merge":
            node_id = str(payload.get("graph_node_id", "")).strip() if isinstance(payload, dict) else ""
            w._ensure_matting_orchestrator().set_merge_preview_node(node_id)
            self._try_show_merge_quick_preview(payload)
            w._set_status(w._tr("status_ready"))
            return

        if node_type in {"matting", "chromakey", "corridorkey"}:
            self._clear_selected_node_preview()
            return

        self._clear_selected_node_preview()

    def _apply_export_preview_path(self, write_node_id: str, path: str) -> None:
        self._host._ensure_matting_orchestrator().apply_export_preview_path(write_node_id, path)

    def _show_output_preview(self, fgr_path: str, alpha_path: str):
        w = self._host
        has_foreground = bool(fgr_path and os.path.exists(fgr_path))
        has_alpha = bool(alpha_path and os.path.exists(alpha_path))

        # If alpha_path is a ZIP (from cloud node_graph), extract it and show first frame.
        if has_alpha and str(alpha_path).endswith(".zip"):
            try:
                with zipfile.ZipFile(alpha_path, "r") as zf:
                    # Extract to temp directory next to ZIP.
                    extract_dir = Path(alpha_path).parent / f"_{Path(alpha_path).stem}_extracted"
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    zf.extractall(extract_dir)
                    # Find first PNG/image frame.
                    images = sorted(extract_dir.glob("*.png")) + sorted(extract_dir.glob("*.jpg"))
                    if images:
                        alpha_path = str(images[0])
                        has_alpha = True
            except Exception:
                # If extraction fails, still try to proceed.
                pass

        preferred_source = fgr_path if has_foreground else alpha_path
        self._set_selected_node_preview(source=preferred_source)

        if has_alpha:
            w._set_status(w._tr("status_done_fg_alpha"))
        elif has_foreground:
            w._set_status(w._tr("status_done_fg"))
        else:
            w._set_status(w._tr("status_done"))

    def _reset_viewer_preview(self, clear_outputs: bool = True):
        w = self._host
        self._reset_selected_node_preview_source()
        self._set_output_display_transform_controls_enabled(False)
        w.ui.output_video_label.clear()
        w.ui.output_video_label.setText(w._output_preview_placeholder_text)

        if w._node_graph_dialog is not None and hasattr(w._node_graph_dialog, "clear_write_runtime_previews"):
            if clear_outputs:
                w._node_graph_dialog.clear_write_runtime_previews()

        if clear_outputs:
            w.ui.btn_split_view.setEnabled(False)
            w.ui.btn_split_view.setChecked(False)
            w._split_view_enabled = False
            w._split_view_dragging = False

    def _resolve_output_image_sequence(self, path: str) -> list[str]:
        return [str(item) for item in resolve_numbered_image_sequence(path)]

    def _load_preview_image_or_video_frame(self, path: str):
        if not path or not os.path.exists(path):
            return None

        if is_supported_image_file(path):
            return load_rgb_image(path)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _render_output_preview_for_index(self, idx: int):
        w = self._host

        # ── Fast path: Source node == input already loaded in RAM ────────────
        # Skip the full numpy→QImage→SmoothTransformation pipeline.
        # Pre-scale with cv2 to display resolution so the QImage is tiny.
        if getattr(w, "_selected_node_uses_main_frames", False) and not w._split_view_enabled:
            main_frames = getattr(w, "all_frames", None) or []
            if main_frames:
                safe_idx = max(0, min(len(main_frames) - 1, idx))
                raw = np.asarray(main_frames[safe_idx])
                if raw.ndim == 3 and raw.shape[2] >= 3:
                    raw = raw[:, :, :3]
                lbl = w.ui.output_video_label
                lw, lh = max(1, lbl.width()), max(1, lbl.height())
                fh, fw = raw.shape[:2]
                scale = min(lw / max(1, fw), lh / max(1, fh))
                dw, dh = max(1, int(fw * scale)), max(1, int(fh * scale))
                resized = cv2.resize(raw, (dw, dh), interpolation=cv2.INTER_AREA)
                if resized.dtype != np.uint8:
                    resized = np.clip(resized, 0, 255).astype(np.uint8)
                qimg = w._to_qimage(resized)
                lbl.setPixmap(QPixmap.fromImage(qimg))
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                return

        frame = self._load_selected_node_frame_by_index(idx)
        if frame is None:
            return

        if (
            w._node_graph_dialog is not None
            and hasattr(w._node_graph_dialog, "set_write_runtime_preview_for_node")
        ):
            try:
                selected_id = w._ensure_matting_orchestrator().selected_export_preview_node_id()
                if selected_id:
                    w._node_graph_dialog.set_write_runtime_preview_for_node(
                        selected_id, w._to_qimage(frame)
                    )
            except Exception as exc:
                logger.warning("_render_output_preview_for_index: Write-node thumbnail update failed: %s", exc)

        if w._split_view_enabled:
            if 0 <= idx < len(w.all_frames):
                original = w.all_frames[idx]
            elif w._original_foreground_for_splitter is not None:
                original = np.asarray(w._original_foreground_for_splitter, dtype=np.uint8)
            else:
                original = frame
            h, wid = frame.shape[:2]
            if original.shape[:2] != (h, wid):
                original = cv2.resize(original, (wid, h))
            split_x = int(wid * w._split_x_ratio)
            split_x = max(0, min(wid - 1, split_x))
            composite = np.asarray(original).copy()
            composite[:, split_x:] = frame[:, split_x:]

            # Normalise composite to uint8 before drawing.
            if composite.dtype != np.uint8:
                max_val = float(np.nanmax(composite)) if composite.size else 0.0
                if max_val <= 1.5:
                    composite = np.clip(composite * 255.0, 0.0, 255.0).astype(np.uint8)
                else:
                    composite = np.clip(composite, 0.0, 255.0).astype(np.uint8)

            # ── Classic splitter overlay ──────────────────────────────────
            # All sizes are expressed in screen pixels and converted to source
            # pixels so the handle looks consistent at any video resolution.
            lbl = w.ui.output_video_label
            lw = max(1, int(lbl.width()))
            lh = max(1, int(lbl.height()))
            display_scale = min(lw / max(1, wid), lh / max(1, h))
            display_scale = max(display_scale, 1e-6)

            def _sp(screen_px: float, lo: int = 1, hi: int = 512) -> int:
                """Convert a screen-space pixel size to source-image pixel size."""
                return max(lo, min(hi, int(round(screen_px / display_scale))))

            mid_y = h // 2

            # 2 px white line — drawn once at native resolution, no shadow
            cv2.line(composite, (split_x, 0), (split_x, h - 1),
                     (200, 200, 200), 2, cv2.LINE_AA)

            # Small circle drag handle: no outline, white fill + two arrows inside
            r_main  = _sp(5, lo=2, hi=40)
            cv2.circle(composite, (split_x, mid_y), r_main,
                       (220, 220, 220), -1, cv2.LINE_AA)
            # Two small inward-pointing triangles (left ◀  right ▶)
            arr     = _sp(3, lo=1, hi=20)   # arrow arm length
            gap     = _sp(2, lo=1, hi=12)   # distance from centre
            a_th    = max(1, _sp(1, lo=1, hi=6))
            dark    = (80, 80, 80)
            # left arrow ◀
            lx = split_x - gap
            cv2.line(composite, (lx, mid_y), (lx - arr, mid_y - arr), dark, a_th, cv2.LINE_AA)
            cv2.line(composite, (lx, mid_y), (lx - arr, mid_y + arr), dark, a_th, cv2.LINE_AA)
            # right arrow ▶
            rx2 = split_x + gap
            cv2.line(composite, (rx2, mid_y), (rx2 + arr, mid_y - arr), dark, a_th, cv2.LINE_AA)
            cv2.line(composite, (rx2, mid_y), (rx2 + arr, mid_y + arr), dark, a_th, cv2.LINE_AA)

            w._set_label_pixmap(w.ui.output_video_label, QPixmap.fromImage(w._to_qimage(composite)))
        else:
            w._set_label_pixmap(
                w.ui.output_video_label,
                QPixmap.fromImage(w._to_qimage(frame)),
            )
