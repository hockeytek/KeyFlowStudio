"""Write-output adapter interface and host-backed implementation.

This module decouples MattingOrchestrator from MainWindow internals for
save-related runtime branches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class WriteOutputAdapter(Protocol):
    def save_sam_mask_output(self, mask_path: str, write_cfg: dict, fallback_output_dir: Path) -> str:
        """Save SAM mask output according to Write node configuration."""

    def save_load_output(self, write_cfg: dict, fallback_output_dir: Path) -> str:
        """Save Load passthrough output according to Write node configuration."""

    def save_frames_to_write_output(
        self,
        frames_rgb,
        write_cfg: dict,
        fallback_output_dir: Path,
        default_stem: str,
        *,
        source_is_video: bool,
        source_ext: str,
    ) -> str:
        """Save frame list according to Write node configuration."""


class HostWriteOutputAdapter:
    """Adapter that forwards write-save calls to MainWindow host methods."""

    def __init__(self, host: Any) -> None:
        self._host = host

    def save_sam_mask_output(self, mask_path: str, write_cfg: dict, fallback_output_dir: Path) -> str:
        return self._host._save_sam_mask_output(mask_path, write_cfg, fallback_output_dir)

    def save_load_output(self, write_cfg: dict, fallback_output_dir: Path) -> str:
        return self._host._save_load_output(write_cfg, fallback_output_dir)

    def save_frames_to_write_output(
        self,
        frames_rgb,
        write_cfg: dict,
        fallback_output_dir: Path,
        default_stem: str,
        *,
        source_is_video: bool,
        source_ext: str,
    ) -> str:
        return self._host._save_frames_to_write_output(
            frames_rgb,
            write_cfg,
            fallback_output_dir,
            default_stem,
            source_is_video=source_is_video,
            source_ext=source_ext,
        )
