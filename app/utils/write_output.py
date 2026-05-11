"""Shared helpers for Write-node output format resolution and file encoding."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.constants import (
    DEFAULT_VIDEO_CODEC,
    DEFAULT_VIDEO_CRF,
    DEFAULT_VIDEO_PRESET,
    FFMPEG_CODEC_MAP,
    PRORES_PROFILES,
    VALID_VIDEO_PRESETS,
)
from app.utils.media import save_exr_image


VIDEO_OUTPUT_FORMATS: frozenset[str] = frozenset({"mp4", "mov"})
IMAGE_OUTPUT_FORMATS: frozenset[str] = frozenset({"png", "jpg", "jpeg", "exr"})
COMPAT_VIDEO_OUTPUT_FORMATS: frozenset[str] = VIDEO_OUTPUT_FORMATS | frozenset({"avi", "mkv", "webm", "m4v"})
COMPAT_IMAGE_OUTPUT_FORMATS: frozenset[str] = IMAGE_OUTPUT_FORMATS | frozenset({"bmp", "webp", "tif", "tiff"})
WRITE_OUTPUT_FORMATS: frozenset[str] = frozenset({"source"}) | COMPAT_VIDEO_OUTPUT_FORMATS | COMPAT_IMAGE_OUTPUT_FORMATS
ALPHA_CAPABLE_IMAGE_FORMATS: frozenset[str] = frozenset({"png", "webp", "tif", "tiff", "exr"})


def is_normalized_float_range(min_val: float, max_val: float) -> bool:
    return min_val >= -0.5 and max_val <= 1.5


def to_u8_frame(src: np.ndarray) -> np.ndarray:
    if src.dtype == np.uint8:
        return src
    if src.dtype == np.uint16:
        return (src.astype(np.float32) / 257.0).astype(np.uint8)
    if np.issubdtype(src.dtype, np.floating):
        min_val = float(np.nanmin(src)) if src.size else 0.0
        max_val = float(np.nanmax(src)) if src.size else 0.0
        if is_normalized_float_range(min_val, max_val):
            return (np.clip(src, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.clip(src, 0.0, 255.0).astype(np.uint8)
    return np.clip(src.astype(np.float32), 0.0, 255.0).astype(np.uint8)


def to_u16_frame(src: np.ndarray) -> np.ndarray:
    if src.dtype == np.uint16:
        return src
    if src.dtype == np.uint8:
        return src.astype(np.uint16) * 257
    if np.issubdtype(src.dtype, np.floating):
        min_val = float(np.nanmin(src)) if src.size else 0.0
        max_val = float(np.nanmax(src)) if src.size else 0.0
        if is_normalized_float_range(min_val, max_val):
            return (np.clip(src, 0.0, 1.0) * 65535.0).astype(np.uint16)
        if max_val <= 255.0 + 1e-6:
            return (np.clip(src, 0.0, 255.0) * 257.0).astype(np.uint16)
        return np.clip(src, 0.0, 65535.0).astype(np.uint16)
    return np.clip(src.astype(np.float32), 0.0, 65535.0).astype(np.uint16)


def promote_alpha_to_rgba_exr(src: np.ndarray) -> np.ndarray:
    arr = np.asarray(src)
    alpha = arr[:, :, 0] if arr.ndim == 3 else arr

    if alpha.dtype == np.uint8:
        alpha_f = alpha.astype(np.float32) / 255.0
    elif alpha.dtype == np.uint16:
        alpha_f = alpha.astype(np.float32) / 65535.0
    else:
        alpha_f = alpha.astype(np.float32)
        min_val = float(np.nanmin(alpha_f)) if alpha_f.size else 0.0
        max_val = float(np.nanmax(alpha_f)) if alpha_f.size else 0.0
        if not is_normalized_float_range(min_val, max_val):
            if max_val <= 255.0 + 1e-6:
                alpha_f = alpha_f / 255.0
            elif max_val > 0.0:
                alpha_f = alpha_f / max_val

    alpha_f = np.clip(alpha_f, 0.0, 1.0).astype(np.float32)
    rgba = np.zeros((alpha_f.shape[0], alpha_f.shape[1], 4), dtype=np.float32)
    rgba[:, :, 3] = alpha_f
    return rgba


def resolve_write_output_format(write_config: dict | None, source: Path) -> str:
    if not write_config:
        return ""

    output_fmt = str(write_config.get("output_format", "source")).strip().lower()
    if output_fmt and output_fmt != "source":
        return output_fmt if output_fmt in WRITE_OUTPUT_FORMATS else "png"

    src_ext = source.suffix.lower().lstrip(".")
    if src_ext in VIDEO_OUTPUT_FORMATS or src_ext in IMAGE_OUTPUT_FORMATS:
        return src_ext
    if src_ext in COMPAT_VIDEO_OUTPUT_FORMATS:
        return "mp4"
    if src_ext in COMPAT_IMAGE_OUTPUT_FORMATS:
        return "png"
    return "png"


def image_extension_for_format(output_fmt: str) -> str:
    normalized = str(output_fmt or "png").strip().lower()
    if normalized not in COMPAT_IMAGE_OUTPUT_FORMATS:
        normalized = "png"
    return ".jpg" if normalized in {"jpg", "jpeg"} else f".{normalized}"


def _has_embedded_alpha(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[2] >= 4


def _is_normalized_rgba_float(arr: np.ndarray) -> bool:
    return (
        _has_embedded_alpha(arr)
        and np.issubdtype(arr.dtype, np.floating)
        and is_normalized_float_range(
            float(np.nanmin(arr)) if arr.size else 0.0,
            float(np.nanmax(arr)) if arr.size else 0.0,
        )
    )


def _straight_srgb_from_premultiplied_rgba(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr_f = arr.astype(np.float32)
    rgb_premul = arr_f[:, :, :3]
    alpha = np.clip(arr_f[:, :, 3:4], 0.0, 1.0)
    straight_lin = np.clip(rgb_premul / (alpha + 1e-6), 0.0, 1.0)
    straight_srgb = np.where(
        straight_lin <= 0.0031308,
        straight_lin * 12.92,
        1.055 * np.power(np.clip(straight_lin, 1e-9, 1.0), 1.0 / 2.4) - 0.055,
    )
    return np.clip(straight_srgb, 0.0, 1.0).astype(np.float32), alpha.astype(np.float32)


def build_video_output_params(
    codec: str = DEFAULT_VIDEO_CODEC,
    *,
    crf: int = DEFAULT_VIDEO_CRF,
    preset: str = DEFAULT_VIDEO_PRESET,
) -> tuple[str, list[str]]:
    normalized = str(codec or DEFAULT_VIDEO_CODEC).strip().lower() or DEFAULT_VIDEO_CODEC
    if normalized in PRORES_PROFILES:
        params = ["-profile:v", PRORES_PROFILES[normalized], "-vendor", "apl0"]
        if normalized == "prores4444":
            params += ["-pix_fmt", "yuva444p10le"]
        else:
            params += ["-pix_fmt", "yuv422p10le"]
        return "prores_ks", params

    safe_preset = preset if preset in VALID_VIDEO_PRESETS else DEFAULT_VIDEO_PRESET
    ffmpeg_codec = FFMPEG_CODEC_MAP.get(normalized, "libx264")
    return ffmpeg_codec, ["-crf", str(crf), "-preset", safe_preset]


def prepare_video_frame(frame, codec: str = DEFAULT_VIDEO_CODEC) -> np.ndarray:
    frame_u8 = to_u8_frame(np.asarray(frame))
    if frame_u8.ndim == 2:
        frame_u8 = np.stack([frame_u8, frame_u8, frame_u8], axis=-1)
    elif frame_u8.ndim == 3 and frame_u8.shape[2] == 1:
        frame_u8 = np.repeat(frame_u8, 3, axis=2)
    elif frame_u8.ndim == 3 and frame_u8.shape[2] >= 4:
        normalized = str(codec or DEFAULT_VIDEO_CODEC).strip().lower()
        if normalized != "prores4444":
            frame_u8 = frame_u8[:, :, :3]
    return frame_u8


def save_image_frame(
    frame_arr,
    out_path: Path,
    *,
    output_fmt: str,
    png_compression: int,
    png_bit_depth: int,
    jpg_quality: int,
    embed_alpha: bool = False,
) -> None:
    arr = np.asarray(frame_arr)
    output_fmt = str(output_fmt or "png").strip().lower() or "png"
    is_jpg = output_fmt in {"jpg", "jpeg"}
    is_exr = output_fmt == "exr"
    is_png = output_fmt == "png"
    is_tiff = output_fmt in {"tif", "tiff"}
    is_webp = output_fmt == "webp"
    is_bmp = output_fmt == "bmp"
    embed_alpha = bool(embed_alpha) and output_fmt in ALPHA_CAPABLE_IMAGE_FORMATS

    if is_exr and embed_alpha and (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1)):
        arr = promote_alpha_to_rgba_exr(arr)

    if _is_normalized_rgba_float(arr) and output_fmt in {"png", "webp", "tif", "tiff"}:
        straight_srgb, alpha = _straight_srgb_from_premultiplied_rgba(arr)
        arr = np.concatenate([straight_srgb, alpha], axis=2) if embed_alpha else straight_srgb
    elif not embed_alpha and _has_embedded_alpha(arr) and output_fmt in ALPHA_CAPABLE_IMAGE_FORMATS:
        arr = arr[:, :, :3]

    is_single_ch = arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1)

    if is_jpg:
        arr_8 = to_u8_frame(arr)
        if arr_8.ndim == 2:
            arr_8 = np.stack([arr_8, arr_8, arr_8], axis=-1)
        elif arr_8.ndim == 3 and arr_8.shape[2] == 1:
            arr_8 = np.repeat(arr_8, 3, axis=2)
        elif arr_8.ndim == 3 and arr_8.shape[2] >= 4:
            arr_8 = arr_8[:, :, :3]
        Image.fromarray(arr_8, "RGB").save(str(out_path), quality=jpg_quality, subsampling=0)
        return

    if is_exr:
        save_exr_image(arr, out_path, single_channel=is_single_ch)
        return

    n_ch = arr.shape[2] if arr.ndim == 3 else 1

    if is_png and n_ch == 4 and np.issubdtype(arr.dtype, np.floating) and is_normalized_float_range(
        float(np.nanmin(arr)) if arr.size else 0.0,
        float(np.nanmax(arr)) if arr.size else 0.0,
    ):
        alpha = np.clip(arr[:, :, 3:4].astype(np.float32), 0.0, 1.0)
        straight_srgb = np.clip(arr[:, :, :3].astype(np.float32), 0.0, 1.0)
        if png_bit_depth == 16:
            rgb_16 = (straight_srgb * 65535.0).astype(np.uint16)
            alpha_16 = (alpha[:, :, 0] * 65535.0).astype(np.uint16)
            rgba_16 = np.dstack([rgb_16, alpha_16])
            cv2.imwrite(str(out_path), cv2.cvtColor(rgba_16, cv2.COLOR_RGBA2BGRA))
        else:
            rgb_8 = (straight_srgb * 255.0).astype(np.uint8)
            alpha_8 = (alpha[:, :, 0] * 255.0).astype(np.uint8)
            rgba_8 = np.dstack([rgb_8, alpha_8])
            Image.fromarray(rgba_8, "RGBA").save(str(out_path), compress_level=png_compression)
        return

    if is_png or is_tiff:
        if png_bit_depth == 16:
            arr_16 = to_u16_frame(arr)
            if n_ch == 4:
                cv2.imwrite(str(out_path), cv2.cvtColor(arr_16, cv2.COLOR_RGBA2BGRA))
            elif is_single_ch:
                gray = arr_16[:, :, 0] if arr_16.ndim == 3 else arr_16
                cv2.imwrite(str(out_path), gray)
            else:
                cv2.imwrite(str(out_path), cv2.cvtColor(arr_16, cv2.COLOR_RGB2BGR))
            return

        arr_8 = to_u8_frame(arr)
        if is_png:
            if n_ch == 4:
                Image.fromarray(arr_8, "RGBA").save(str(out_path), compress_level=png_compression)
            elif is_single_ch:
                gray = arr_8[:, :, 0] if arr_8.ndim == 3 else arr_8
                Image.fromarray(gray, "L").save(str(out_path), compress_level=png_compression)
            else:
                Image.fromarray(arr_8).save(str(out_path), compress_level=png_compression)
            return

        if n_ch == 4:
            cv2.imwrite(str(out_path), cv2.cvtColor(arr_8, cv2.COLOR_RGBA2BGRA))
        elif is_single_ch:
            gray = arr_8[:, :, 0] if arr_8.ndim == 3 else arr_8
            cv2.imwrite(str(out_path), gray)
        else:
            cv2.imwrite(str(out_path), cv2.cvtColor(arr_8, cv2.COLOR_RGB2BGR))
        return

    arr_8 = to_u8_frame(arr)
    if is_webp:
        if arr_8.ndim == 2:
            Image.fromarray(arr_8, "L").save(str(out_path), format="WEBP", quality=jpg_quality)
        elif arr_8.shape[2] >= 4:
            Image.fromarray(arr_8[:, :, :4], "RGBA").save(str(out_path), format="WEBP", quality=jpg_quality)
        else:
            Image.fromarray(arr_8[:, :, :3], "RGB").save(str(out_path), format="WEBP", quality=jpg_quality)
        return

    if is_bmp:
        if arr_8.ndim == 2:
            cv2.imwrite(str(out_path), arr_8)
        elif arr_8.shape[2] >= 4:
            cv2.imwrite(str(out_path), cv2.cvtColor(arr_8[:, :, :4], cv2.COLOR_RGBA2BGRA))
        else:
            cv2.imwrite(str(out_path), cv2.cvtColor(arr_8[:, :, :3], cv2.COLOR_RGB2BGR))
        return

    Image.fromarray(arr_8).save(str(out_path))