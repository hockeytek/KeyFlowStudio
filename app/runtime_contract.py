"""Unified runtime contract for Matting pipeline.

Defines a single schema for:
- runtime inputs/config
- runtime outputs/results
- cancel semantics
- write preview semantics (preview-only vs production-safe)
"""

from __future__ import annotations

from typing import Callable, Literal, TypedDict

RuntimeWriteSemantics = Literal["preview_only", "production_safe"]
RUNTIME_SEMANTICS_PREVIEW_ONLY: RuntimeWriteSemantics = "preview_only"
RUNTIME_SEMANTICS_PRODUCTION_SAFE: RuntimeWriteSemantics = "production_safe"

RuntimeCancelPolicy = Literal["immediate", "save_partial", "cleanup_partial"]
RUNTIME_CANCEL_IMMEDIATE: RuntimeCancelPolicy = "immediate"
RUNTIME_CANCEL_SAVE_PARTIAL: RuntimeCancelPolicy = "save_partial"
RUNTIME_CANCEL_CLEANUP_PARTIAL: RuntimeCancelPolicy = "cleanup_partial"


class RuntimeGraphPayload(TypedDict):
    nodes: list[dict]
    edges: list[dict]


class RuntimeConfig(TypedDict, total=False):
    is_video: bool
    start_frame: int
    end_frame: int
    compatibility_profile: str
    correction_masks: dict
    node_graph: RuntimeGraphPayload
    erode_kernel: int
    dilate_kernel: int
    n_warmup: int
    fg_write: dict | None
    alpha_write: dict | None
    cancel_policy: RuntimeCancelPolicy


class RuntimeResult(TypedDict, total=False):
    status: Literal["ok", "cancelled", "error"]
    cancelled: bool
    saved_paths: dict[str, str]
    n_frames: int
    fgr_path: str
    alpha_path: str
    error: str
    partial_result: bool
    partial_saved_paths: dict[str, str]


class GraphStreamPreviewPayload(TypedDict, total=False):
    frame: object
    path: str
    stream: str
    semantics: RuntimeWriteSemantics


def normalize_stage_progress(percent: int, status_text: str) -> tuple[int, str]:
    clamped = max(0, min(100, int(percent)))
    return clamped, str(status_text or "")


def normalize_frame_progress(current: int, total: int) -> tuple[int, int]:
    cur = max(0, int(current))
    tot = max(0, int(total))
    return cur, tot


def build_runtime_config(**kwargs) -> RuntimeConfig:
    """Build runtime config with explicit schema keys only."""
    cfg: RuntimeConfig = {}
    for key in RuntimeConfig.__annotations__.keys():
        if key in kwargs:
            cfg[key] = kwargs[key]
    return cfg


def make_runtime_result_ok(saved_paths: dict[str, str] | None, n_frames: int) -> RuntimeResult:
    return {
        "status": "ok",
        "cancelled": False,
        "saved_paths": dict(saved_paths or {}),
        "n_frames": int(n_frames),
    }


def make_runtime_result_cancelled() -> RuntimeResult:
    return {
        "status": "cancelled",
        "cancelled": True,
        "saved_paths": {},
        "n_frames": 0,
        "partial_result": False,
        "partial_saved_paths": {},
    }


def make_runtime_result_cancelled_partial(saved_paths: dict[str, str] | None, n_frames: int = 0) -> RuntimeResult:
    partial = dict(saved_paths or {})
    return {
        "status": "cancelled",
        "cancelled": True,
        "saved_paths": {},
        "n_frames": int(n_frames),
        "partial_result": bool(partial),
        "partial_saved_paths": partial,
    }


def is_runtime_cancelled(result: dict) -> bool:
    status = str(result.get("status", "")).strip().lower()
    return bool(result.get("cancelled")) or status == "cancelled"


def runtime_saved_paths(result: dict) -> dict[str, str]:
    raw = result.get("saved_paths") or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        k = str(key or "").strip()
        v = str(value or "").strip()
        if k and v:
            out[k] = v
    return out


def runtime_partial_saved_paths(result: dict) -> dict[str, str]:
    raw = result.get("partial_saved_paths") or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        k = str(key or "").strip()
        v = str(value or "").strip()
        if k and v:
            out[k] = v
    return out


def normalize_cancel_policy(value: str | None) -> RuntimeCancelPolicy:
    raw = str(value or "").strip().lower()
    if raw == RUNTIME_CANCEL_IMMEDIATE:
        return RUNTIME_CANCEL_IMMEDIATE
    if raw == RUNTIME_CANCEL_CLEANUP_PARTIAL:
        return RUNTIME_CANCEL_CLEANUP_PARTIAL
    return RUNTIME_CANCEL_SAVE_PARTIAL


def runtime_primary_outputs(result: dict) -> tuple[str, str]:
    """Compatibility fallback for legacy result schema."""
    return (
        str(result.get("fgr_path", "") or "").strip(),
        str(result.get("alpha_path", "") or "").strip(),
    )


def make_stream_preview_payload(
    frame,
    path: str,
    stream: str,
    semantics: RuntimeWriteSemantics,
) -> GraphStreamPreviewPayload:
    return {
        "frame": frame,
        "path": str(path or ""),
        "stream": str(stream or ""),
        "semantics": semantics,
    }


def tr_with_fallback(
    tr: Callable[[str], str],
    primary_key: str,
    fallback_key: str,
) -> str:
    """Translate via primary i18n key; fallback when translation is missing.

    Missing-key behavior in this project returns the key itself.
    """
    text = tr(primary_key)
    if text == primary_key:
        return tr(fallback_key)
    return text
