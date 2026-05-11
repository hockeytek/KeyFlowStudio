"""Coordinator for interactive SAM2/SAM3 UI-graph synchronization."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFileDialog, QMessageBox


class SamInteractionCoordinator:
    """Moves SAM interaction glue logic out of MainWindow."""

    def __init__(self, host) -> None:
        self._host = host

    def on_active_node_changed(self, node_type: str) -> None:
        host = self._host
        prev = host._active_node_type
        host._active_node_type = node_type
        host._set_viewer_interactive(node_type == "sam2")

        if node_type in {"sam2", "sam3"}:
            host.sam2.set_backend(node_type)
            status_key = "lbl_sam3_status_default" if node_type == "sam3" else "lbl_sam_status_default"
            host._on_sam2_status_changed(host._tr(status_key))

        # Restore / clear input viewer mask overlay on SAM node selection change.
        if node_type in {"sam2"} and prev not in {"sam2"}:
            host.sam2_graph.restore_masks_from_graph_node()
            host._render_input_preview()
        elif node_type not in {"sam2"} and prev in {"sam2"}:
            host._show_clean_input_frame()

    def on_graph_controls_changed(self, point_mode: str, live_sam2: bool, backend: str) -> None:
        host = self._host
        backend_norm = str(backend or "").strip().lower()
        if backend_norm not in {"sam2", "sam3"}:
            backend_norm = "sam2"
        host.sam2.sync_controls(point_mode=point_mode, backend=backend_norm)
        host.sam2.toggle_live_sam2(live_sam2)
        if host._optional_controls_present:
            self.sync_sam_interaction_buttons(
                live_sam2=live_sam2,
                controls_enabled=True,
                point_mode=point_mode,
                sync_live_toggle=True,
            )

    def sync_sam3_prompt_state_to_graph(self, status_text: str | None = None) -> None:
        host = self._host
        if host._active_node_type != "sam3":
            return

        host._call_node_graph_dialog(
            "sync_sam3_prompt_state",
            prompt_points=[],
            prompt_labels=[],
            status_text=status_text,
            point_mode=None,
            live_sam2=False,
        )

    def set_point_mode(self, positive: bool) -> None:
        host = self._host
        if not host._optional_controls_present:
            return
        host.sam2.set_point_mode(positive)
        if positive:
            host.ui.btn_positive_point.setChecked(True)
        else:
            host.ui.btn_negative_point.setChecked(True)

    def on_live_sam2_toggled(self, checked: bool) -> None:
        host = self._host
        if not host._optional_controls_present:
            return
        host.sam2.toggle_live_sam2(checked)
        self.sync_sam_interaction_buttons(
            live_sam2=checked,
            controls_enabled=host.ui.btn_live_sam2.isEnabled(),
        )

    def on_input_label_mouse_press(self, event) -> None:
        host = self._host
        if host.current_frame is None:
            return

        if host._active_node_type != "sam2":
            return

        if host.sam2.generation_active:
            host._on_sam2_status_changed(host._tr("sam_wait_mask"))
            return

        x, y = self.map_click_to_image(event.position().x(), event.position().y())
        if x is None:
            return

        is_live_mode = host.sam2.state.live_sam2
        if is_live_mode:
            if event.button() == Qt.MouseButton.LeftButton:
                host.sam2.add_point(x, y, positive=True)
                host._render_input_preview()
                status_text = f"{host._tr('sam_points_count')} {len(host.sam2.state.points)}"
                host._on_sam2_status_changed(status_text)
                host.sam2.generate_mask(
                    host.current_frame,
                    show_errors=False,
                    live_mode=True,
                    click_coords=(x, y),
                )
            return

        if event.button() == Qt.MouseButton.RightButton:
            host.sam2.pop_last_point()
        else:
            is_pos = host.sam2.state.point_mode != "negative"
            host.sam2.add_point(x, y, positive=is_pos)

        host._render_input_preview()
        status_text = f"{host._tr('sam_points_count')} {len(host.sam2.state.points)}"
        host._on_sam2_status_changed(status_text)
    def on_generate_mask(self) -> None:
        host = self._host
        host.sam2.generate_mask(
            host.current_frame,
            show_errors=True,
            processing_active=host.matting.is_active,
            current_frame_index=host.current_frame_index,
        )

    def on_clear_points(self) -> None:
        host = self._host
        host.sam2.clear_points()
        self.sync_sam3_prompt_state_to_graph(host._tr("sam_points_cleared"))

    def on_add_mask(self) -> None:
        host = self._host
        if not host.sam2.add_current_mask(host.current_frame_index):
            QMessageBox.warning(host, host._tr("status_error"), host._tr("err_no_mask"))
            return
        if host._optional_controls_present:
            host.ui.masks_list.setCurrentRow(len(host.sam2.state.added_masks) - 1)

    def map_click_to_image(self, click_x: float, click_y: float):
        host = self._host
        if host.current_frame is None:
            return None, None

        img_h, img_w = host.current_frame.shape[:2]
        lbl = host.ui.input_video_label
        lw = max(1, lbl.width())
        lh = max(1, lbl.height())

        scale = min(lw / img_w, lh / img_h)
        disp_w = img_w * scale
        disp_h = img_h * scale
        off_x = (lw - disp_w) / 2
        off_y = (lh - disp_h) / 2

        if click_x < off_x or click_x > off_x + disp_w:
            return None, None
        if click_y < off_y or click_y > off_y + disp_h:
            return None, None

        x = int((click_x - off_x) / scale)
        y = int((click_y - off_y) / scale)
        x = max(0, min(img_w - 1, x))
        y = max(0, min(img_h - 1, y))
        return x, y

    def set_sam_base_controls_enabled(self, enabled: bool) -> None:
        host = self._host
        if host._optional_controls_present:
            host.ui.btn_load.setEnabled(enabled)
            host.ui.combo_input_type.setEnabled(enabled)
            host.ui.spin_erode_kernel.setEnabled(enabled)
            host.ui.spin_dilate_kernel.setEnabled(enabled)
            host.ui.spin_warmup_frames.setEnabled(enabled)
            host.ui.btn_clear_points.setEnabled(enabled)
            host.ui.btn_add_mask.setEnabled(enabled)
            host.ui.btn_remove_mask.setEnabled(enabled)
            host.ui.btn_load_mask.setEnabled(enabled)
        host.ui.btn_settings.setEnabled(enabled)
        host.ui.spin_start_frame.setEnabled(enabled)
        host.ui.spin_num_frames.setEnabled(enabled)
        host.ui.spin_end_frame.setEnabled(enabled)
        host._set_transport_controls_enabled(enabled)

    def sync_sam_interaction_buttons(
        self,
        *,
        live_sam2: bool,
        controls_enabled: bool,
        point_mode: str | None = None,
        sync_live_toggle: bool = False,
    ) -> None:
        host = self._host
        if not host._optional_controls_present:
            return
        if point_mode is not None:
            host.ui.btn_positive_point.setChecked(point_mode != "negative")
            host.ui.btn_negative_point.setChecked(point_mode == "negative")
        if sync_live_toggle:
            host.ui.btn_live_sam2.blockSignals(True)
            try:
                host.ui.btn_live_sam2.setChecked(live_sam2)
            finally:
                host.ui.btn_live_sam2.blockSignals(False)
        host.ui.btn_live_sam2.setEnabled(controls_enabled)
        point_controls_enabled = controls_enabled and not live_sam2
        host.ui.btn_positive_point.setEnabled(point_controls_enabled)
        host.ui.btn_negative_point.setEnabled(point_controls_enabled)
        host.ui.btn_generate_mask.setEnabled(point_controls_enabled)

    def set_sam_controls_busy(self, active: bool) -> None:
        host = self._host
        self.set_sam_base_controls_enabled(not active)

        if active:
            self.sync_sam_interaction_buttons(
                live_sam2=bool(host._optional_controls_present and host.ui.btn_live_sam2.isChecked()),
                controls_enabled=False,
            )
            host.ui.btn_run.setEnabled(False)
            return

        if not host._optional_controls_present:
            host.ui.btn_run.setEnabled(not host.matting.is_active)
            return

        host.ui.btn_run.setEnabled(not host.matting.is_active)
        self.sync_sam_interaction_buttons(
            live_sam2=host.ui.btn_live_sam2.isChecked(),
            controls_enabled=True,
        )

    def on_remove_masks(self) -> None:
        host = self._host
        graph_selected = host.sam2_graph.selected_graph_mask_rows()
        if graph_selected:
            selected = sorted(set(graph_selected), reverse=True)
        else:
            selected = sorted(self.selected_mask_rows(), reverse=True)
        host.sam2.remove_masks(selected)

    def refresh_mask_list(self) -> None:
        host = self._host
        if not host._optional_controls_present:
            return
        host.ui.masks_list.clear()
        for item in host.sam2.state.mask_items():
            host.ui.masks_list.addItem(item)

    def on_load_mask_file(self) -> None:
        host = self._host
        file_path, _ = QFileDialog.getOpenFileName(
            host,
            host._tr("dlg_select_mask_title"),
            "",
            f"{host._tr('dlg_filter_image')} (*.png *.jpg *.jpeg);;{host._tr('dlg_filter_all')} (*)",
        )
        if not file_path:
            return
        host.sam2.load_mask_file(file_path)

    def resolve_mask_path_for_processing(self) -> str | None:
        host = self._host
        graph_selected = host.sam2_graph.selected_graph_mask_rows()
        if graph_selected:
            selected = graph_selected
        else:
            selected = self.selected_mask_rows()
        path = host.sam2.resolve_mask_path_for_processing(selected or None)
        if path:
            return path
        stored = host._call_node_graph_dialog("sam_node_mask_source_path")
        if stored and Path(stored).exists():
            return stored
        return None

    def selected_mask_rows(self) -> list[int]:
        host = self._host
        if not host._optional_controls_present:
            return []
        rows = sorted({idx.row() for idx in host.ui.masks_list.selectedIndexes()})
        return [row for row in rows if 0 <= row < len(host.sam2.state.added_masks)]

    def effective_selected_sam_mask_rows(self) -> list[int]:
        host = self._host
        rows = set(host.sam2_graph.selected_graph_mask_rows())
        rows.update(self.selected_mask_rows())
        return sorted(row for row in rows if 0 <= row < len(host.sam2.state.added_masks))

    @staticmethod
    def _apply_mask_overlay(frame: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
        mask_bool = mask > 127
        if np.any(mask_bool):
            frame[mask_bool] = frame[mask_bool] * (1 - alpha) + color * alpha

    @staticmethod
    def _draw_mask_contour(frame: np.ndarray, mask: np.ndarray, color: np.ndarray, thickness: int = 2) -> None:
        mask_u8 = np.where(mask > 127, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        contour_color = tuple(int(v) for v in np.clip(color, 0, 255))
        cv2.drawContours(frame, contours, -1, contour_color, max(1, int(thickness)), lineType=cv2.LINE_AA)

    @staticmethod
    def _sam_overlay_color_for_row(row: int) -> np.ndarray:
        palette = (
            (58, 255, 114),
            (0, 220, 255),
            (255, 174, 66),
            (255, 90, 132),
            (180, 120, 255),
            (255, 235, 95),
            (64, 224, 208),
            (255, 140, 0),
        )
        r, g, b = palette[int(row) % len(palette)]
        return np.array([r, g, b], dtype=np.float32)

    def show_clean_input_frame(self) -> None:
        host = self._host
        if host.current_frame is None:
            return
        qimg = host._to_qimage(host.current_frame)
        pix = QPixmap.fromImage(qimg)
        host._set_label_pixmap(host.ui.input_video_label, pix)

    def render_input_preview(self) -> None:
        host = self._host
        if host.current_frame is None:
            return

        frame = host.current_frame.copy().astype(np.float32)
        expected_shape = host.current_frame.shape[:2]
        selected_rows = set(self.effective_selected_sam_mask_rows())
        selected_contours: list[tuple[int, np.ndarray, np.ndarray]] = []

        for row, (_fi, mask) in enumerate(host.sam2.state.added_masks):
            if _fi != host.current_frame_index:
                continue
            if mask is None or mask.shape[:2] != expected_shape:
                continue

            base_color = self._sam_overlay_color_for_row(row)
            if row in selected_rows:
                self._apply_mask_overlay(frame, mask, base_color, 0.45)
                selected_contours.append((row, mask, base_color))
            else:
                self._apply_mask_overlay(frame, mask, base_color, 0.28)
                selected_contours.append((row, mask, base_color))

        if host.sam2.state.current_mask is not None and host.sam2.state.current_mask.shape[:2] == expected_shape:
            self._apply_mask_overlay(
                frame,
                host.sam2.state.current_mask,
                np.array([255, 140, 0], dtype=np.float32),
                0.52,
            )

        frame = np.clip(frame, 0, 255).astype(np.uint8)
        for row, mask, color in selected_contours:
            thickness = 2 if row in selected_rows else 1
            self._draw_mask_contour(frame, mask, color, thickness=thickness)

        qimg = host._to_qimage(frame)
        pix = QPixmap.fromImage(qimg)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for i, (x, y) in enumerate(host.sam2.state.points):
            label = host.sam2.state.point_labels[i] if i < len(host.sam2.state.point_labels) else 1
            color = QColor(0, 255, 0) if label == 1 else QColor(255, 80, 80)
            r = 6
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.setBrush(color)
            painter.drawEllipse(x - r, y - r, r * 2, r * 2)
        painter.end()

        host._set_label_pixmap(host.ui.input_video_label, pix)

    def on_sam2_status_changed(self, text: str) -> None:
        host = self._host
        if hasattr(host, "sam2") and getattr(host.sam2, "state", None) is not None:
            try:
                host.sam2.state.set_status(text)
            except Exception:
                pass
        if (
            hasattr(host, "_node_graph_dialog")
            and host._node_graph_dialog is not None
            and not bool(getattr(host, "_suspend_sam2_graph_sync", False))
        ):
            host.sam2_graph.sync_to_graph(text)
        if host._optional_controls_present and hasattr(host.ui, "lbl_status") and host.ui.lbl_status.isVisible():
            host.ui.lbl_status.setText(text)
        host._set_status(text)

    def on_sam2_progress(self, percent: int, status_text: str) -> None:
        host = self._host
        host.ui.progress_bar.setRange(0, 100)
        host.ui.progress_bar.setValue(max(0, min(100, int(percent))))
        if status_text:
            host._set_status(status_text)

    def on_sam2_generation_finished(self) -> None:
        host = self._host
        self.ensure_sam2_auto_propagation_state()
        host.ui.progress_bar.setRange(0, 100)
        host.ui.progress_bar.setValue(100)
        final_status = str(getattr(getattr(host.sam2, "state", None), "status_text", "") or "").strip()
        host.sam2_graph.sync_to_graph(final_status or host._tr("sam_mask_ready"))
        exported_count, exported_frames = host._save_sam2_outputs_to_connected_write_nodes()
        if exported_count > 0:
            host._set_status(
                host._tr("sam_write_immediate_export_done").format(count=exported_count, frames=exported_frames)
            )

        if host._active_node_type in {"sam2"}:
            self.show_mask_preview_on_output(host.sam2.state.mask_for_frame(host.current_frame_index))

        if host._pending_processing_after_sam2_auto_propagate and not host.matting.is_active:
            host._pending_processing_after_sam2_auto_propagate = False
            host._skip_next_auto_sam2_propagate = True
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, host.start_processing)

    def on_sam2_error(self, message: str, show_dialog: bool) -> None:
        host = self._host
        host.ui.progress_bar.setRange(0, 100)
        host.ui.progress_bar.setValue(0)
        if show_dialog:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(host, host._tr("sam_error_title"), message)

    def on_graph_sam2_remove_mask_requested(self) -> None:
        self.on_remove_masks()

    def on_graph_sam2_load_mask_requested(self) -> None:
        self.on_load_mask_file()

    def on_graph_sam2_propagate_requested(self, direction: str) -> None:
        host = self._host
        frame_start, frame_end = host._resolve_effective_video_frame_bounds()
        if host.current_frame_index < frame_start or host.current_frame_index >= frame_end:
            host.sam2.status_changed.emit(
                host._tr("sam2_current_frame_out_of_range").format(
                    start=frame_start + 1,
                    end=max(frame_start + 1, frame_end),
                )
            )
            return

        tracking_frames = host.all_frames[frame_start:frame_end]
        host.sam2.propagate_video(
            direction=direction,
            all_frames=tracking_frames,
            current_frame_index=host.current_frame_index - frame_start,
            frame_index_offset=frame_start,
            current_frame_index_global=host.current_frame_index,
            processing_active=host.matting.is_active,
        )

    def on_graph_sam2_reprompt_requested(self) -> None:
        host = self._host
        host.sam2.reprompt_video_frame(
            current_frame=host.current_frame,
            all_frames=host.all_frames,
            current_frame_index=host.current_frame_index,
            processing_active=host.matting.is_active,
        )

    def on_graph_sam2_session_reset_requested(self) -> None:
        host = self._host
        host.sam2.reset_video_session()
        if host._node_graph_dialog is not None and hasattr(host._node_graph_dialog, "clear_node_frame_progress"):
            host._node_graph_dialog.clear_node_frame_progress("sam2")

    def has_sam2_node_in_graph(self) -> bool:
        host = self._host
        dialog = getattr(host, "_node_graph_dialog", None)
        if dialog is None or not hasattr(dialog, "export_graph_preset"):
            return False

        preset = dialog.export_graph_preset()
        nodes = preset.get("nodes", []) if isinstance(preset, dict) else []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if str(node.get("type", "")).strip().lower() not in {"sam2"}:
                continue
            props = node.get("properties", {}) or {}
            if not bool(props.get("enabled", True)):
                continue
            return True
        return False

    def has_sam2_to_matting_mask_link_in_graph(self) -> bool:
        host = self._host
        dialog = getattr(host, "_node_graph_dialog", None)
        if dialog is None or not hasattr(dialog, "export_graph_preset"):
            return False

        preset = dialog.export_graph_preset()
        if not isinstance(preset, dict):
            return False

        nodes = preset.get("nodes") or []
        node_map: dict[str, tuple[str, bool]] = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            node_type = str(node.get("type", "")).strip().lower()
            props = node.get("properties", {}) or {}
            enabled = bool(props.get("enabled", True))
            node_map[node_id] = (node_type, enabled)

        for edge in preset.get("connections", []) or []:
            if not isinstance(edge, dict):
                continue
            dst_port = str(edge.get("dst_port", "")).strip().lower()
            if dst_port != "mask":
                continue
            src_id = str(edge.get("src", "")).strip()
            dst_id = str(edge.get("dst", "")).strip()
            src_meta = node_map.get(src_id)
            dst_meta = node_map.get(dst_id)
            if src_meta is None or dst_meta is None:
                continue
            src_type, src_enabled = src_meta
            dst_type, dst_enabled = dst_meta
            if src_enabled and dst_enabled and src_type in {"sam2"} and dst_type == "matting":
                return True
        return False

    def has_ready_sam2_mask_for_auto_propagation(self) -> bool:
        host = self._host
        if host.sam2.state.current_mask is not None:
            return True
        if bool(host.sam2.state.added_masks):
            return True
        mask_path = str(getattr(host.sam2.state, "mask_path", "") or "").strip()
        return bool(mask_path and Path(mask_path).exists())

    def ensure_sam2_auto_propagation_state(self) -> None:
        host = self._host
        if not hasattr(host, "_pending_processing_after_sam2_auto_propagate"):
            host._pending_processing_after_sam2_auto_propagate = False
        if not hasattr(host, "_skip_next_auto_sam2_propagate"):
            host._skip_next_auto_sam2_propagate = False

    def try_auto_propagate_sam2_before_processing(self) -> bool:
        host = self._host
        self.ensure_sam2_auto_propagation_state()
        if host._skip_next_auto_sam2_propagate:
            host._skip_next_auto_sam2_propagate = False
            return False
        if host._pending_processing_after_sam2_auto_propagate:
            return True
        if host.sam2.generation_active or host.matting.is_active:
            return False
        if not host.is_video_input or len(host.all_frames or []) <= 1:
            return False
        if not self.has_sam2_node_in_graph():
            return False
        if self.has_sam2_to_matting_mask_link_in_graph():
            self._notify_auto_propagate_skipped_for_matting_link()
            return False
        if not self.has_ready_sam2_mask_for_auto_propagation():
            return False
        if not bool(host.sam2.state.points):
            return False

        host._pending_processing_after_sam2_auto_propagate = True
        self.on_graph_sam2_propagate_requested("forward")
        if host.sam2.generation_active:
            return True

        host._pending_processing_after_sam2_auto_propagate = False
        return False

    def _notify_auto_propagate_skipped_for_matting_link(self) -> None:
        host = self._host
        text = "Auto SAM2 skipped: SAM is connected to Matting mask input"
        tr_method = getattr(host, "_tr", None)
        if callable(tr_method):
            try:
                text = tr_method("sam2_auto_propagate_skipped_matting_mask")
            except Exception:
                pass

        sam_status = getattr(getattr(host, "sam2", None), "status_changed", None)
        if hasattr(sam_status, "emit") and callable(sam_status.emit):
            try:
                sam_status.emit(text)
                return
            except Exception:
                pass

        set_status = getattr(host, "_set_status", None)
        if callable(set_status):
            try:
                set_status(text)
            except Exception:
                pass

    def show_mask_preview_on_output(self, mask: np.ndarray | None) -> None:
        if mask is None:
            return
        mask_array = np.asarray(mask, dtype=np.uint8)
        if mask_array.ndim != 2:
            return

        alpha_rgb = np.stack([mask_array, mask_array, mask_array], axis=-1)
        host = self._host
        host._set_selected_node_preview(frame=alpha_rgb)

    def on_frame_changed_show_sam_mask(self, frame_index: int) -> None:
        """On frame slider change, show SAM mask if SAM2 node is active."""
        host = self._host
        if host._active_node_type in {"sam2"}:
            self.show_mask_preview_on_output(host.sam2.state.mask_for_frame(frame_index))

    def update_graph_sam_tracking_tooltips(self) -> None:
        """Update SAM2 propagation button tooltips with current frame range."""
        host = self._host
        dialog = getattr(host, "_node_graph_dialog", None)
        if dialog is None or not hasattr(dialog, "sam_props_panel"):
            return

        try:
            panel = dialog.sam_props_panel
        except Exception:
            return

        frame_start, frame_end = host._resolve_effective_video_frame_bounds()
        if frame_end <= frame_start:
            return

        range_text = host._tr("sam2_tracking_range_tooltip").format(
            start=frame_start + 1,
            end=frame_end,
        )
        backward_tip = host._format_tooltip(
            f"{host._tr('sam2_btn_propagate_backward_tooltip')}\n\n{range_text}",
            width=320,
        )
        forward_tip = host._format_tooltip(
            f"{host._tr('sam2_btn_propagate_forward_tooltip')}\n\n{range_text}",
            width=320,
        )
        if hasattr(panel, "btn_sam2_propagate_backward"):
            panel.btn_sam2_propagate_backward.setToolTip(backward_tip)
        if hasattr(panel, "btn_sam2_propagate_forward"):
            panel.btn_sam2_propagate_forward.setToolTip(forward_tip)

    def save_sam2_mask_output(self, mask_path: str, write_cfg: dict, fallback_output_dir: Path) -> str:
        """Export SAM2 mask file to output location."""
        from PIL import Image
        
        host = self._host
        mask_file = Path(mask_path)
        if not mask_file.exists():
            return ""
        
        with Image.open(mask_file) as _mask_raw:
            mask_img = _mask_raw.convert("L")
            mask_arr = np.asarray(mask_img, dtype=np.uint8).copy()
        
        mask_rgb = np.stack([mask_arr, mask_arr, mask_arr], axis=-1)
        node_id = str(write_cfg.get("graph_node_id", "")).strip()
        if node_id and host._node_graph_dialog is not None and hasattr(host._node_graph_dialog, "set_write_runtime_preview_for_node"):
            host._node_graph_dialog.set_write_runtime_preview_for_node(node_id, host._to_qimage(mask_rgb))
        
        return host._save_frames_to_write_output(
            [mask_rgb],
            write_cfg,
            fallback_output_dir / "sam_mask",
            default_stem=mask_file.stem or "sam_mask",
            source_is_video=False,
            source_ext=mask_file.suffix,
        )
