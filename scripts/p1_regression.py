#!/usr/bin/env python3
"""Minimal P1 regression checks for MatAnyone2 app runtime stability and output sanity."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.model_service import InferenceService
from app.services.model_service import MODEL_URL, MODEL_VARIANT
from app.utils import get_model_variant_dir


def build_frames(height: int, width: int, count: int) -> list[np.ndarray]:
    """Create deterministic synthetic RGB frames with mild motion and texture."""
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]

    frames: list[np.ndarray] = []
    for i in range(count):
        shift = (i * 7) % max(1, width)
        r = np.roll(np.broadcast_to(x, (height, width)), shift, axis=1)
        g = np.broadcast_to(y, (height, width))
        b = ((r.astype(np.uint16) + g.astype(np.uint16)) // 2).astype(np.uint8)
        frame = np.stack([r, g, b], axis=-1)
        frames.append(frame)
    return frames


def build_mask(height: int, width: int) -> np.ndarray:
    """Create a centered rectangular mask compatible with inference input."""
    mask = np.zeros((height, width), dtype=np.uint8)
    y1 = int(height * 0.2)
    y2 = int(height * 0.8)
    x1 = int(width * 0.25)
    x2 = int(width * 0.75)
    mask[y1:y2, x1:x2] = 255
    return mask


def assert_alpha_batch(alphas: list[np.ndarray], expected_len: int, h: int, w: int, case_name: str) -> None:
    if len(alphas) != expected_len:
        raise AssertionError(f"{case_name}: expected {expected_len} alpha frames, got {len(alphas)}")

    for idx, alpha in enumerate(alphas):
        if alpha.shape != (h, w):
            raise AssertionError(
                f"{case_name}: alpha[{idx}] has shape {alpha.shape}, expected {(h, w)}"
            )
        if not np.isfinite(alpha).all():
            raise AssertionError(f"{case_name}: alpha[{idx}] contains non-finite values")
        amin = float(np.min(alpha))
        amax = float(np.max(alpha))
        if amin < -1e-6 or amax > 1.000001:
            raise AssertionError(
                f"{case_name}: alpha[{idx}] is out of [0,1] range: min={amin:.6f}, max={amax:.6f}"
            )


def run_case(service: InferenceService, name: str, h: int, w: int, n_frames: int, warmup: int, erode: int, dilate: int) -> None:
    frames = build_frames(h, w, n_frames)
    mask = build_mask(h, w)
    t0 = time.perf_counter()
    alphas = service.process_video(
        frames=frames,
        mask=mask,
        n_warmup=warmup,
        r_erode=erode,
        r_dilate=dilate,
    )
    elapsed = time.perf_counter() - t0
    assert_alpha_batch(alphas, n_frames, h, w, name)
    print(f"[PASS] {name}: {n_frames} frames {w}x{h} in {elapsed:.2f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run minimal P1 regression checks")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device selection through KEYFLOW_DEVICE",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller resolutions for faster smoke checks",
    )
    args = parser.parse_args()

    os.environ["KEYFLOW_DEVICE"] = "" if args.device == "auto" else args.device

    print(f"[INFO] P1 regression started (device={args.device})")
    print("[INFO] Loading model and running deterministic synthetic checks...")

    service = InferenceService()
    model_dir = get_model_variant_dir("matanyone2", MODEL_VARIANT)
    model_file = Path(MODEL_URL).name
    model_path = model_dir / model_file
    print(f"[INFO] Expected model path: {model_path}")
    print(f"[INFO] Model file exists: {model_path.exists()}")
    print(f"[INFO] Resolved runtime device: {service.model_service.get_device()}")
    # Use default model resolver (downloads/caches in platform app-data models dir).
    service.model_service.load_model()

    try:
        if args.quick:
            run_case(service, "smoke-image", h=240, w=320, n_frames=1, warmup=1, erode=4, dilate=4)
            run_case(service, "smoke-video", h=270, w=480, n_frames=3, warmup=1, erode=4, dilate=4)
            run_case(service, "resolution-switch-A", h=360, w=640, n_frames=2, warmup=1, erode=4, dilate=4)
            run_case(service, "resolution-switch-B", h=200, w=320, n_frames=2, warmup=1, erode=4, dilate=4)
            run_case(service, "resolution-switch-A2", h=360, w=640, n_frames=2, warmup=1, erode=4, dilate=4)
        else:
            run_case(service, "smoke-image", h=360, w=640, n_frames=1, warmup=1, erode=4, dilate=4)
            run_case(service, "smoke-video", h=540, w=960, n_frames=5, warmup=2, erode=4, dilate=4)
            run_case(service, "resolution-switch-A", h=720, w=1280, n_frames=3, warmup=2, erode=4, dilate=4)
            run_case(service, "resolution-switch-B", h=360, w=640, n_frames=3, warmup=2, erode=4, dilate=4)
            run_case(service, "resolution-switch-A2", h=720, w=1280, n_frames=3, warmup=2, erode=4, dilate=4)
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1

    print("[OK] P1 regression suite passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
