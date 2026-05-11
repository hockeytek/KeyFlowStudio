"""Graph preset save orchestration extracted from MainWindow."""

from __future__ import annotations

from typing import Callable


class GraphPresetSaveCoordinator:
    """Builds a graph preset snapshot with SAM payloads and frame range."""

    def __init__(
        self,
        *,
        sam2_graph,
        get_dialog: Callable,
        get_start_frame: Callable[[], int],
        get_end_frame: Callable[[], int],
        get_num_frames: Callable[[], int],
    ) -> None:
        self._sam2_graph = sam2_graph
        self._get_dialog = get_dialog
        self._get_start_frame = get_start_frame
        self._get_end_frame = get_end_frame
        self._get_num_frames = get_num_frames

    def build_current_preset(self) -> dict | None:
        """Return current graph preset payload augmented with frame range."""
        dialog = self._get_dialog()
        if dialog is None:
            return None

        self._sam2_graph.sync_to_graph()
        mask_source_path, mask_payloads = self._sam2_graph.persist_masks(force_disk=True)
        dialog.sync_sam_runtime_state(mask_source_path=mask_source_path, mask_payloads=mask_payloads)

        preset = dialog.export_graph_preset()
        preset["start_frame"] = int(self._get_start_frame())
        preset["end_frame"] = int(self._get_end_frame())
        preset["num_frames"] = int(self._get_num_frames())
        return preset
