"""Shared helpers for Write-node stream names and output directories."""

from __future__ import annotations

from pathlib import Path

_SEQUENCE_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".exr", ".webp"}


def _safe_stream_token(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def normalize_write_stream_name(
    *,
    source_node_type: str = "",
    source_port: str = "",
    port_type: str = "",
    port_label: str = "",
) -> str:
    """Return a stable folder name for a Write input stream."""
    node_type = _safe_stream_token(source_node_type)
    port_name = _safe_stream_token(source_port)
    semantic_type = _safe_stream_token(port_type)
    label = _safe_stream_token(port_label)

    if semantic_type == "alpha":
        return "alpha"

    for candidate in (port_name, label):
        if candidate in {"fg", "alpha", "comp", "processed", "input", "img"}:
            return candidate

    if port_name in {"mask", "sam_mask"} or label in {"mask", "sam_mask"}:
        return "alpha" if node_type in {"sam2"} or semantic_type == "alpha" else "mask"

    if port_name == "out":
        if node_type in {"sam2"}:
            return "alpha"
        if label and label not in {"out", "output", "mask"}:
            return label
        if node_type in {"source", "load", "load_media"}:
            return "img"
        if semantic_type == "image":
            return "img"
        if semantic_type:
            return semantic_type
        return "out"

    if port_name:
        return port_name
    if label:
        return label
    if semantic_type == "image":
        return "img"
    if semantic_type:
        return semantic_type
    return "out"


def build_keyflow_base_dir(source: Path) -> Path:
    """Return the base auto output directory for a source file or sequence frame."""
    if source.suffix.lower() in _SEQUENCE_IMAGE_EXTS and source.stem.isdigit():
        return source.parent.parent / f"{source.parent.name}_keyflow"
    return source.parent / f"{source.stem}_keyflow"


def _sanitize_path_component(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace("/", "_").replace("\\", "_").strip()


def build_graph_write_output_dir(
    base_output_dir: Path,
    *,
    source_node_title: str = "",
    port_label: str = "",
    stream_label: str = "",
) -> Path:
    """Return canonical Write path: <base>/<node title>/<port label>."""
    base_dir = Path(base_output_dir)
    safe_node_title = _sanitize_path_component(source_node_title)
    safe_port_label = _sanitize_path_component(port_label)
    fallback_stream = normalize_write_stream_name(source_port=stream_label) if stream_label else "out"

    if safe_node_title and safe_port_label:
        return base_dir / safe_node_title / safe_port_label
    if safe_node_title:
        return base_dir / safe_node_title / _sanitize_path_component(fallback_stream)
    if safe_port_label:
        return base_dir / safe_port_label
    return base_dir / fallback_stream


def get_port_output_label(node_type: str, port_name: str) -> str:
    """Return the display label for an output port from its spec, or a prettified fallback."""
    from app.node_graph.specs import get_node_spec

    fallback = str(port_name).replace("_", " ").title()
    spec = get_node_spec(str(node_type).strip().lower())
    if spec is None:
        return fallback

    for port in (spec.outputs or ()):  # pragma: no branch - specs usually have outputs
        if str(port.name) == str(port_name):
            return str(port.label) if port.label else fallback
    return fallback


def resolve_graph_write_output_dir(
    write_cfg: dict,
    output_dir: Path,
    stream_label: str,
    source_node_title: str = "",
    port_label: str = "",
) -> Path:
    """Return the actual Write-node output directory for auto or custom output settings."""
    custom_dir = str(write_cfg.get("output_dir", "")).strip()
    auto_output_dir = bool(write_cfg.get("auto_output_dir", True))
    if not auto_output_dir and custom_dir:
        return Path(custom_dir)
    return build_graph_write_output_dir(
        output_dir,
        source_node_title=source_node_title,
        port_label=port_label,
        stream_label=stream_label,
    )


def build_keyflow_output_dir(source: Path, stream_label: str) -> Path:
    """Return the final auto output directory for a specific stream."""
    return build_keyflow_base_dir(source) / normalize_write_stream_name(source_port=stream_label)
