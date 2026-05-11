"""Pure frame preview and geometry helpers used by worker/viewer paths."""

from __future__ import annotations

import numpy as np

from app.utils.write_output import is_normalized_float_range


def coerce_preview_frame(frame) -> np.ndarray | None:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
        min_val = float(np.nanmin(gray)) if gray.size else 0.0
        max_val = float(np.nanmax(gray)) if gray.size else 0.0
        if gray.dtype != np.uint8:
            if is_normalized_float_range(min_val, max_val):
                gray = np.clip(gray, 0.0, 1.0)
                gray = np.power(gray, 1.0 / 2.2) * 255.0
            gray = np.clip(gray, 0.0, 255.0).astype(np.uint8)
        else:
            gray = arr
        return np.stack([gray] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return coerce_preview_frame(arr[:, :, 0])
    if arr.ndim == 3 and arr.shape[2] >= 3:
        if arr.dtype != np.uint8 and arr.shape[2] >= 4:
            rgb_premul = arr[:, :, :3].astype(np.float32)
            alpha_ch = np.clip(arr[:, :, 3:4].astype(np.float32), 0.0, 1.0)
            min_val = float(np.nanmin(rgb_premul)) if rgb_premul.size else 0.0
            max_val = float(np.nanmax(rgb_premul)) if rgb_premul.size else 0.0
            if is_normalized_float_range(min_val, max_val):
                bg_lin = 0.214
                comp_lin = np.clip(rgb_premul + bg_lin * (1.0 - alpha_ch), 0.0, 1.0)
                comp_srgb = np.where(
                    comp_lin <= 0.0031308,
                    comp_lin * 12.92,
                    1.055 * np.power(np.clip(comp_lin, 1e-9, 1.0), 1.0 / 2.4) - 0.055,
                )
                return np.clip(comp_srgb * 255.0, 0.0, 255.0).astype(np.uint8)
            return np.clip(rgb_premul, 0.0, 255.0).astype(np.uint8)

        rgb = arr[:, :, :3].astype(np.float32)
        if arr.dtype != np.uint8:
            min_val = float(np.nanmin(rgb)) if rgb.size else 0.0
            max_val = float(np.nanmax(rgb)) if rgb.size else 0.0
            if is_normalized_float_range(min_val, max_val):
                rgb = np.clip(rgb, 0.0, 1.0)
                rgb = np.power(rgb, 1.0 / 2.2) * 255.0
            rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        else:
            rgb = arr[:, :, :3]
        return rgb
    return None


def frame_bbox(frame) -> tuple[int, int, int, int]:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        coverage = np.asarray(arr, dtype=np.float32) > 1e-6
    elif arr.ndim == 3 and arr.shape[2] >= 4:
        coverage = np.asarray(arr[:, :, 3], dtype=np.float32) > 1e-6
    elif arr.ndim == 3:
        coverage = np.any(np.asarray(arr[:, :, :3], dtype=np.float32) > 1e-6, axis=2)
    else:
        return (0, 0, 0, 0)

    ys, xs = np.where(coverage)
    if ys.size == 0 or xs.size == 0:
        return (0, 0, 0, 0)
    x0 = int(np.min(xs))
    y0 = int(np.min(ys))
    x1 = int(np.max(xs)) + 1
    y1 = int(np.max(ys)) + 1
    return (x0, y0, x1, y1)


def bbox_union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if a[2] <= a[0] or a[3] <= a[1]:
        return b
    if b[2] <= b[0] or b[3] <= b[1]:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def bbox_intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0)
    return (x0, y0, x1, y1)


def clip_frame_to_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = np.asarray(frame).copy()
    h, w = out.shape[:2]
    x0 = max(0, min(w, int(bbox[0])))
    y0 = max(0, min(h, int(bbox[1])))
    x1 = max(0, min(w, int(bbox[2])))
    y1 = max(0, min(h, int(bbox[3])))
    if x1 <= x0 or y1 <= y0:
        return np.zeros_like(out)

    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    if out.ndim == 2:
        out[~mask] = 0
    else:
        out[~mask, ...] = 0
    return out
