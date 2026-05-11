"""Pure helpers for graph execution action handling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from app.node_graph.models import GraphEdge, GraphNode
from app.utils.media import is_numbered_image_sequence, load_image_float

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PassthroughSourceFrames:
    frames: list
    log_message: str | None = None


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