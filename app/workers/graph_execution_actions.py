"""Pure helpers for graph execution action handling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from app.node_graph.models import GraphEdge, GraphNode
from app.node_graph.specs import get_node_spec
from app.utils.media import is_numbered_image_sequence, load_image_float

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PassthroughSourceFrames:
    frames: list
    log_message: str | None = None


@dataclass(frozen=True)
class GatheredNodeInputs:
    inputs: dict[Any, Any]
    missing_source_ids: tuple[Any, ...] = ()


def build_graph_downstream_targets(
    nodes_by_id: dict[str, GraphNode],
    edges: list[GraphEdge],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    targets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for edge in edges:
        src_key = (str(edge.src_id), str(edge.src_port or "").strip().lower())
        dst_node = nodes_by_id.get(edge.dst_id)
        targets.setdefault(src_key, []).append(
            {
                "dst_id": str(edge.dst_id),
                "dst_port": str(edge.dst_port or "").strip().lower(),
                "dst_type": str(getattr(dst_node, "type", "") or "").strip().lower(),
                "dst_enabled": bool(getattr(dst_node, "enabled", True)) if dst_node is not None else True,
            }
        )
    return targets


def gather_graph_node_inputs(
    nodes_by_id: dict[Any, GraphNode],
    edges: list[GraphEdge],
    node_id: Any,
    outputs: dict[Any, Any],
) -> GatheredNodeInputs:
    inputs: dict[Any, Any] = {}
    missing_source_ids: list[Any] = []

    for edge in edges:
        if edge.dst_id != node_id:
            continue

        src_node_id = edge.src_id
        src_port = str(edge.src_port or "").strip().lower()
        dst_port = edge.dst_port
        dst_node = nodes_by_id.get(node_id)
        src_node = nodes_by_id.get(src_node_id)

        if not src_port:
            src_node_type = str(getattr(src_node, "type", "") or "").strip().lower()
            src_spec = get_node_spec(src_node_type)
            if src_spec is not None and len(src_spec.outputs) == 1:
                src_port = str(src_spec.outputs[0].name or "out").strip().lower() or "out"
            else:
                src_port = "out"
        if str(getattr(dst_node, "type", "") or "") == "export" and not str(dst_port or "").strip():
            dst_port = "in"

        if src_node_id not in outputs:
            missing_source_ids.append(src_node_id)
            continue

        src_output = outputs[src_node_id]
        if not isinstance(src_output, dict) or src_port not in src_output:
            continue

        inputs[dst_port] = src_output[src_port]
        inputs[f"__src_port__{dst_port}"] = src_port
        inputs[f"__src_node_type__{dst_port}"] = str(getattr(src_node, "type", "") or "")
        inputs[f"__src_node_title__{dst_port}"] = str(getattr(src_node, "title", "") or "")
        src_meta = src_output.get("__meta__")
        if isinstance(src_meta, dict) and src_port in src_meta:
            inputs[f"__meta__{dst_port}"] = src_meta[src_port]

    return GatheredNodeInputs(inputs, tuple(missing_source_ids))


def resolve_requested_output_ports(
    downstream_targets: dict[tuple[Any, Any], Any],
    node_id: Any,
    default_ports: set[str],
) -> set[str]:
    requested: set[str] = set()
    node_id_str = str(node_id)
    for (src_node_id, src_port), downstream in (downstream_targets or {}).items():
        if str(src_node_id) != node_id_str:
            continue
        if not isinstance(downstream, list):
            continue
        if any(bool(item.get("dst_enabled", True)) for item in downstream if isinstance(item, dict)):
            port = str(src_port or "").strip().lower()
            if port:
                requested.add(port)

    if not requested:
        return set(default_ports)
    return requested & set(default_ports)


def build_passthrough_source_output(
    node_frames: list,
    frame_bbox: Callable[[Any], tuple[int, int, int, int]],
) -> dict[str, Any]:
    bbox_sequence = [frame_bbox(frame) for frame in node_frames] if node_frames else []
    port_meta = {"bbox_sequence": bbox_sequence}
    return {
        "out": node_frames,
        "image": node_frames,
        "frame_sequence": node_frames,
        "__meta__": {
            "out": dict(port_meta),
            "image": dict(port_meta),
            "frame_sequence": dict(port_meta),
        },
    }


def load_passthrough_source_frames(
    node: GraphNode,
    initial_frames: list,
    *,
    graph_source_path: str,
    graph_output_dir: Path | None,
    graph_start_frame: int,
    graph_end_frame: int,
    load_video: Callable[[str, Path], tuple[list, float, str]],
    load_image: Callable[[str], Any] = load_image_float,
    is_image_sequence: Callable[[str], bool] = is_numbered_image_sequence,
) -> PassthroughSourceFrames:
    node_type = str(node.type)
    if node_type == "source":
        return PassthroughSourceFrames(initial_frames)

    node_props = node.properties or {}
    node_path = str(node_props.get("path", "")).strip()
    if not node_path or node_path == graph_source_path or not Path(node_path).exists():
        return PassthroughSourceFrames(initial_frames)

    try:
        node_media_type = str(node_props.get("media_type", "video")).strip().lower()
        if node_media_type == "image" and not is_image_sequence(node_path):
            node_frames = [load_image(node_path)]
        else:
            node_frames, _, _ = load_video(node_path, graph_output_dir or Path("."))
            if graph_end_frame > 0:
                node_frames = node_frames[graph_start_frame:graph_end_frame]
            elif graph_start_frame > 0:
                node_frames = node_frames[graph_start_frame:]
        return PassthroughSourceFrames(
            node_frames,
            f"{node_type.capitalize()} node {node.id}: loaded {len(node_frames)} frame(s) from {Path(node_path).name}",
        )
    except Exception as load_exc:
        logger.warning(
            "%s node %s: failed to load %s: %s, using global frames",
            node_type.capitalize(),
            node.id,
            node_path,
            load_exc,
        )
        return PassthroughSourceFrames(initial_frames)


def build_deferred_action_output(node_type: str) -> dict[str, Any]:
    if node_type in {"sam2"}:
        return {"__deferred_sam_disk__": True, "out": None, "mask": None}
    return {"__deferred_staged__": True, "alpha": None}


def format_deferred_action_log(node_id: str, node_type: str) -> str:
    if node_type in {"sam2"}:
        return f"SAM2 node {node_id}: deferred (disk-streaming masks into CorridorKey)"
    return f"BiRefNet node {node_id}: deferred (staged into CorridorKey)"