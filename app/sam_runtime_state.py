"""Centralized runtime state for interactive SAM workflow."""

from __future__ import annotations

from typing import Iterable

import numpy as np


class SamRuntimeState:
    """Owns mutable SAM state independently from the main window UI."""

    def __init__(self) -> None:
        self.point_mode = "positive"
        self.live_sam2 = False
        self.status_text = ""
        self.reset_for_media()

    def reset_for_media(self) -> None:
        self.mask_path: str | None = None
        self.added_masks: list[tuple[int, np.ndarray]] = []  # (frame_index, mask)
        self.clear_prompt_state()

    def clear_prompt_state(self) -> None:
        self.points: list[tuple[int, int]] = []
        self.point_labels: list[int] = []
        self.current_mask: np.ndarray | None = None

    def set_status(self, text: str) -> None:
        self.status_text = str(text or "")

    def sync_controls(
        self,
        *,
        point_mode: str | None = None,
        live_sam2: bool | None = None,
    ) -> None:
        if point_mode in {"positive", "negative"}:
            self.point_mode = point_mode
        if live_sam2 is not None:
            self.live_sam2 = bool(live_sam2)

    def add_point(self, x: int, y: int, positive: bool) -> None:
        self.points.append((int(x), int(y)))
        self.point_labels.append(1 if positive else 0)

    def pop_last_point(self) -> bool:
        if not self.points:
            return False
        self.points.pop()
        if self.point_labels:
            self.point_labels.pop()
        return True

    def set_current_mask(self, mask: np.ndarray | None) -> None:
        self.current_mask = None if mask is None else np.asarray(mask, dtype=np.uint8)

    def add_current_mask(self, frame_index: int = 0) -> bool:
        if self.current_mask is None:
            return False
        self.added_masks.append((int(frame_index), self.current_mask.copy()))
        self.clear_prompt_state()
        return True

    def remove_masks(self, rows: Iterable[int]) -> int:
        removed = 0
        for row in sorted({int(value) for value in rows}, reverse=True):
            if 0 <= row < len(self.added_masks):
                self.added_masks.pop(row)
                removed += 1
        return removed

    def load_mask_path(self, path: str) -> None:
        self.mask_path = path
        self.current_mask = None

    def mask_items(self) -> list[str]:
        return [f"F{fi}: mask_{i + 1:03d}" for i, (fi, _) in enumerate(self.added_masks)]

    def current_preview_mask(self) -> np.ndarray | None:
        return self.mask_for_frame(None)

    def mask_for_frame(self, frame_index: int | None) -> np.ndarray | None:
        """Return the most relevant mask for a timeline frame.

        Priority:
        1. Explicit mask stored for the requested frame.
        2. Last mask from a previous frame (closest <= requested).
        3. First mask from a later frame.
        4. Current interactive mask.
        5. Last known stored mask.
        """
        if frame_index is None:
            if self.current_mask is not None:
                return self.current_mask
            if self.added_masks:
                return self.added_masks[-1][1]
            return None

        idx = int(frame_index)
        by_frame: dict[int, np.ndarray] = {}
        for fi, mask in self.added_masks:
            by_frame[int(fi)] = mask

        if idx in by_frame:
            return by_frame[idx]

        lower = [fi for fi in by_frame if fi <= idx]
        if lower:
            return by_frame[max(lower)]

        higher = [fi for fi in by_frame if fi > idx]
        if higher:
            return by_frame[min(higher)]

        if self.current_mask is not None:
            return self.current_mask
        if self.added_masks:
            return self.added_masks[-1][1]
        return None

    def combined_mask(self, selected_rows: Iterable[int] | None = None) -> np.ndarray | None:
        """Return OR-combined frame-0 masks for use as the inference seed.

        Only entries with frame_index == 0 are considered; correction masks
        (frame_index > 0) are handled separately by get_correction_masks_by_frame().

        If frame-0 masks are absent but there are stored masks, the earliest
        available frame is used as a fallback seed so processing can still start.
        """
        if not self.added_masks:
            return None

        # Only frame-0 entries are valid as the initial inference seed.
        frame0 = [(i, m) for i, (fi, m) in enumerate(self.added_masks) if fi == 0]
        if frame0:
            seed_items = frame0
        else:
            min_frame = min(int(fi) for fi, _ in self.added_masks)
            seed_items = [(i, m) for i, (fi, m) in enumerate(self.added_masks) if int(fi) == min_frame]

        if selected_rows is not None:
            sel_set = {int(r) for r in selected_rows}
            chosen = [(i, m) for i, m in seed_items if i in sel_set]
            if not chosen:
                chosen = seed_items
        else:
            chosen = seed_items

        combined = np.zeros_like(chosen[0][1], dtype=np.uint8)
        for _, m in chosen:
            combined = np.where(m > 127, 255, combined).astype(np.uint8)
        return combined

    def get_correction_masks_by_frame(
        self, selected_rows: Iterable[int] | None = None
    ) -> dict[int, np.ndarray]:
        """Return per-frame OR-combined correction masks for frames > 0."""
        items = list(self.added_masks)
        if selected_rows is not None:
            rows = [int(r) for r in selected_rows if 0 <= int(r) < len(items)]
            items = [items[r] for r in rows] if rows else items
        by_frame: dict[int, list[np.ndarray]] = {}
        for frame_idx, mask in items:
            if frame_idx > 0:
                by_frame.setdefault(frame_idx, []).append(mask)
        result: dict[int, np.ndarray] = {}
        for frame_idx, masks in by_frame.items():
            combined = masks[0].copy()
            for m in masks[1:]:
                combined = np.where(m > 127, 255, combined).astype(np.uint8)
            result[frame_idx] = combined
        return result