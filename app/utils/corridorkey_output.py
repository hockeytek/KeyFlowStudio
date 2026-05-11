"""Pure array helpers for CorridorKey output post-processing."""

from __future__ import annotations

import numpy as np


def coerce_alpha_2d(alpha: np.ndarray | None) -> np.ndarray | None:
    if alpha is None:
        return None
    arr = np.asarray(alpha, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] >= 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        return None
    return np.clip(arr, 0.0, 1.0)


def coerce_rgb_float01(rgb: np.ndarray | None) -> np.ndarray | None:
    if rgb is None:
        return None
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    arr = arr[:, :, :3]
    max_val = float(np.nanmax(arr)) if arr.size else 0.0
    if max_val > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def build_corridorkey_processed_output(
    output_mode: str,
    source_rgb: np.ndarray,
    fg_rgb: np.ndarray,
    alpha_2d: np.ndarray,
) -> np.ndarray:
    mode = str(output_mode or "processed").strip().lower()
    if mode == "matte_only":
        rgb = np.repeat(alpha_2d[:, :, np.newaxis], 3, axis=2)
    elif mode == "foreground_only":
        rgb = fg_rgb
    elif mode == "source_matte":
        rgb = source_rgb
    else:
        x = np.clip(fg_rgb, 0.0, 1.0).astype(np.float32)
        fg_lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
        alpha_3d = np.clip(alpha_2d[:, :, np.newaxis], 0.0, 1.0)
        rgb = fg_lin * alpha_3d
    rgba = np.concatenate([np.clip(rgb, 0.0, 1.0), alpha_2d[:, :, np.newaxis]], axis=2)
    return rgba.astype(np.float32)