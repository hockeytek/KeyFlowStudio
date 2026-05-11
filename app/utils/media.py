"""Shared media loading helpers for images and numbered image sequences."""

from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".exr"}


def is_supported_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def resolve_numbered_image_sequence(path: str | Path) -> list[Path]:
    source = Path(path)
    if not source.exists() or not source.is_file() or not is_supported_image_file(source):
        return []

    match = re.match(r"^(.*?)(\d+)$", source.stem)
    if not match:
        return [source]

    prefix, digits = match.groups()
    width = len(digits)
    pattern = f"{prefix}*{source.suffix}"
    candidates: list[tuple[int, Path]] = []

    for candidate in source.parent.glob(pattern):
        if not candidate.is_file() or candidate.suffix.lower() != source.suffix.lower():
            continue
        candidate_match = re.match(rf"^{re.escape(prefix)}(\d+)$", candidate.stem)
        if not candidate_match:
            continue
        candidate_digits = candidate_match.group(1)
        if len(candidate_digits) != width:
            continue
        candidates.append((int(candidate_digits), candidate))

    if not candidates:
        return [source]

    candidates.sort(key=lambda item: item[0])
    return [candidate for _, candidate in candidates]


def is_numbered_image_sequence(path: str | Path) -> bool:
    return len(resolve_numbered_image_sequence(path)) > 1


def read_media_dimensions(path: str | Path, media_type: str) -> tuple[int, int] | None:
    if not path or not Path(path).exists():
        return None
    if media_type == "video":
        if is_numbered_image_sequence(path):
            sequence = resolve_numbered_image_sequence(path)
            if not sequence:
                return None
            frame = load_rgb_image(sequence[0])
            height, width = frame.shape[:2]
            return (width, height)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            cap.release()
        if width > 0 and height > 0:
            return (width, height)
        return None

    frame = load_rgb_image(path)
    height, width = frame.shape[:2]
    return (width, height)


def load_image_sequence(path: str | Path) -> list[np.ndarray]:
    sequence_paths = resolve_numbered_image_sequence(path)
    if not sequence_paths:
        raise FileNotFoundError(f"Image sequence not found: {path}")
    return [load_rgb_image(sequence_path) for sequence_path in sequence_paths]


def load_rgb_image(path: str | Path) -> np.ndarray:
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".exr":
        frame = _load_exr_image(source)
    else:
        try:
            with Image.open(source) as image:
                frame = np.array(image.convert("RGB"))
        except Exception:
            frame = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise
            if frame.ndim == 3:
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
                elif frame.shape[2] >= 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return _ensure_uint8_rgb(frame)


def load_rgb_image_float(path: str | Path) -> np.ndarray:
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".exr":
        frame = _load_exr_image(source)
    else:
        try:
            with Image.open(source) as image:
                frame = np.array(image.convert("RGB"))
        except Exception:
            frame = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise
            if frame.ndim == 3:
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
                elif frame.shape[2] >= 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return _ensure_float32_rgb(frame)


def load_image_float(path: str | Path) -> np.ndarray:
    """Load image as float32 in [0..1] while preserving alpha when present."""
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".exr":
        frame = _load_exr_image(source)
    else:
        try:
            with Image.open(source) as image:
                if image.mode in {"RGBA", "LA"}:
                    frame = np.array(image.convert("RGBA"))
                else:
                    frame = np.array(image.convert("RGB"))
        except Exception:
            frame = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise
            if frame.ndim == 3:
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
                elif frame.shape[2] >= 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return _ensure_float32_image(frame)


def save_exr_image(frame: np.ndarray, path: str | Path, *, single_channel: bool = False) -> None:
    array = np.asarray(frame)
    if single_channel:
        if array.ndim == 3:
            array = array[:, :, 0]
        array = _ensure_float32_exr(array)
        if not cv2.imwrite(str(path), array):
            raise RuntimeError(f"Failed to write EXR file: {path}")
        return

    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    elif array.ndim == 3 and array.shape[2] in {3, 4}:
        array = array[:, :, : array.shape[2]]
    elif array.ndim == 3 and array.shape[2] > 4:
        array = array[:, :, :4]
    else:
        raise ValueError("Unsupported image shape for EXR output")

    exr_data = _ensure_float32_exr(array)
    # cv2 expects BGR/BGRA for EXR files.
    if exr_data.ndim == 3 and exr_data.shape[2] == 4:
        exr_data = cv2.cvtColor(exr_data, cv2.COLOR_RGBA2BGRA)
    else:
        exr_data = cv2.cvtColor(exr_data, cv2.COLOR_RGB2BGR)

    if not cv2.imwrite(str(path), exr_data):
        raise RuntimeError(f"Failed to write EXR file: {path}")


def _load_exr_image(path: Path) -> np.ndarray:
    # For EXR files, cv2.imread is the best option as it preserves float32 data
    # (imageio.v3 converts to uint8, losing linear float precision)
    try:
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame is not None:
            if frame.ndim == 3:
                # cv2.imread reads EXR in BGR; swap to RGB
                if frame.shape[2] == 4:
                    frame = frame[:, :, [2, 1, 0, 3]]  # BGRA -> RGBA
                elif frame.shape[2] >= 3:
                    frame = frame[:, :, ::-1]  # BGR -> RGB
            return frame
    except cv2.error:
        pass
    
    # Fallback to imageio.v2 if cv2 fails
    try:
        import imageio.v2 as iio
        return iio.imread(path)
    except Exception:
        pass
    
    raise RuntimeError(f"Failed to load EXR file: {path}")


def _ensure_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    elif array.ndim == 3:
        channels = array.shape[2]
        if channels == 1:
            array = np.repeat(array, 3, axis=2)
        elif channels >= 3:
            array = array[:, :, :3]
        else:
            raise ValueError("Unsupported image channel count")
    else:
        raise ValueError("Unsupported image shape")

    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.max(array)) if array.size else 0.0
        if max_value <= 2.0:
            # HDR / EXR linear data in ~[0,1]; apply display gamma for UI conversion.
            array = np.clip(array, 0.0, 1.0)
            array = np.power(array, 1.0 / 2.2) * 255.0
        elif max_value > 255.0:
            array = array / max_value * 255.0
        array = np.clip(array, 0.0, 255.0)
        return np.ascontiguousarray(array.astype(np.uint8))

    if np.issubdtype(array.dtype, np.integer):
        max_value = float(np.iinfo(array.dtype).max)
        if max_value > 255.0:
            array = array.astype(np.float32) / max_value * 255.0
        array = np.clip(array, 0.0, 255.0)
        return np.ascontiguousarray(array.astype(np.uint8))

    return np.ascontiguousarray(array.astype(np.uint8))


def _ensure_float32_exr(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    finite_max = np.finfo(np.float32).max
    finite_min = np.finfo(np.float32).min
    array = np.nan_to_num(array, nan=0.0, posinf=finite_max, neginf=finite_min)
    if np.issubdtype(array.dtype, np.floating):
        return np.ascontiguousarray(array.astype(np.float32))

    if np.issubdtype(array.dtype, np.integer):
        max_value = float(np.iinfo(array.dtype).max)
        if max_value > 0.0:
            array = array.astype(np.float32) / max_value
        else:
            array = array.astype(np.float32)
        return np.ascontiguousarray(array.astype(np.float32))

    return np.ascontiguousarray(array.astype(np.float32))


def _ensure_float32_rgb(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    elif array.ndim == 3:
        channels = array.shape[2]
        if channels == 1:
            array = np.repeat(array, 3, axis=2)
        elif channels >= 3:
            array = array[:, :, :3]
        else:
            raise ValueError("Unsupported image channel count")
    else:
        raise ValueError("Unsupported image shape")

    finite_max = np.finfo(np.float32).max
    finite_min = np.finfo(np.float32).min
    array = np.nan_to_num(array, nan=0.0, posinf=finite_max, neginf=finite_min)

    if np.issubdtype(array.dtype, np.floating):
        return np.ascontiguousarray(array.astype(np.float32))

    if np.issubdtype(array.dtype, np.integer):
        max_value = float(np.iinfo(array.dtype).max)
        if max_value > 0.0:
            array = array.astype(np.float32) / max_value
        else:
            array = array.astype(np.float32)
        return np.ascontiguousarray(array.astype(np.float32))

    return np.ascontiguousarray(array.astype(np.float32))


def _ensure_float32_image(frame: np.ndarray) -> np.ndarray:
    """Convert image to float32 [0..1] preserving 1/3/4 channels."""
    array = np.asarray(frame)
    if array.ndim == 2:
        pass
    elif array.ndim == 3:
        channels = array.shape[2]
        if channels not in {1, 3, 4}:
            if channels > 4:
                array = array[:, :, :4]
            else:
                raise ValueError("Unsupported image channel count")
    else:
        raise ValueError("Unsupported image shape")

    finite_max = np.finfo(np.float32).max
    finite_min = np.finfo(np.float32).min
    array = np.nan_to_num(array, nan=0.0, posinf=finite_max, neginf=finite_min)

    if np.issubdtype(array.dtype, np.floating):
        return np.ascontiguousarray(array.astype(np.float32))

    if np.issubdtype(array.dtype, np.integer):
        max_value = float(np.iinfo(array.dtype).max)
        if max_value > 0.0:
            array = array.astype(np.float32) / max_value
        else:
            array = array.astype(np.float32)
        return np.ascontiguousarray(array.astype(np.float32))

    return np.ascontiguousarray(array.astype(np.float32))