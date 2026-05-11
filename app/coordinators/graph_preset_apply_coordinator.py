"""Graph preset apply orchestration extracted from MainWindow."""

from __future__ import annotations

from typing import Callable

from app.utils.frame_range_helper import FrameRangeController


class GraphPresetApplyCoordinator:
    """Applies post-load preset effects: SAM masks, frame range, runtime state."""

    def __init__(
        self,
        *,
        sam2_graph,
        clear_write_outputs: Callable[[], None],
        restore_write_outputs: Callable[[], None],
        set_selected_preset_key: Callable[[str], None],
        set_baseline_from_current: Callable[[], None],
        refresh_preset_combo: Callable[[str], None],
        get_start_frame: Callable[[], int],
        get_end_frame: Callable[[], int],
        set_start_frame: Callable[[int], None],
        set_num_frames: Callable[[int], None],
        set_end_frame: Callable[[int], None],
        block_frame_controls_signals: Callable[[bool], None],
        get_total_frames: Callable[[], int],
    ) -> None:
        self._sam2_graph = sam2_graph
        self._clear_write_outputs = clear_write_outputs
        self._restore_write_outputs = restore_write_outputs
        self._set_selected_preset_key = set_selected_preset_key
        self._set_baseline_from_current = set_baseline_from_current
        self._refresh_preset_combo = refresh_preset_combo
        self._get_start_frame = get_start_frame
        self._get_end_frame = get_end_frame
        self._set_start_frame = set_start_frame
        self._set_num_frames = set_num_frames
        self._set_end_frame = set_end_frame
        self._block_frame_controls_signals = block_frame_controls_signals
        self._get_total_frames = get_total_frames

    def apply_frame_range_from_preset(self, preset: dict | None, *, total_frames: int | None = None) -> None:
        if not isinstance(preset, dict):
            return
        if not any(k in preset for k in ("start_frame", "end_frame", "num_frames")):
            return

        total = max(1, int(total_frames or self._get_total_frames() or 1))

        def _to_int(value, default):
            try:
                return int(value)
            except Exception:
                return int(default)

        current_start = self._get_start_frame()
        current_end = self._get_end_frame()
        start = _to_int(preset.get("start_frame", current_start), current_start)
        start = max(0, min(total - 1, start))

        has_count = "num_frames" in preset
        if has_count:
            count = max(0, _to_int(preset.get("num_frames", 0), 0))
            end = -1 if count == 0 else min(total - 1, max(start, start + count - 1))
        else:
            end = _to_int(preset.get("end_frame", current_end), current_end)
            if end >= 0:
                end = min(total - 1, max(start, end))
            else:
                end = -1
            count = FrameRangeController.on_end_frame_changed(start, end, 0).updated_frame_count or 0

        self._block_frame_controls_signals(True)
        try:
            self._set_start_frame(start)
            self._set_num_frames(count)
            self._set_end_frame(end)
        finally:
            self._block_frame_controls_signals(False)

    def finalize_preset_apply(self, *, preset: dict | None, key: str) -> None:
        self._sam2_graph.restore_masks_from_graph_node()
        self.apply_frame_range_from_preset(preset)
        self._clear_write_outputs()
        self._set_selected_preset_key(key)
        self._set_baseline_from_current()
        self._refresh_preset_combo(key)
        self._restore_write_outputs()
