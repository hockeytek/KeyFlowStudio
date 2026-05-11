"""Pure helpers for graph execution action handling."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.node_graph.models import GraphEdge, GraphNode


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


def build_deferred_action_output(node_type: str) -> dict[str, Any]:
    if node_type in {"sam2"}:
        return {"__deferred_sam_disk__": True, "out": None, "mask": None}
    return {"__deferred_staged__": True, "alpha": None}


def format_deferred_action_log(node_id: str, node_type: str) -> str:
    if node_type in {"sam2"}:
        return f"SAM2 node {node_id}: deferred (disk-streaming masks into CorridorKey)"
    return f"BiRefNet node {node_id}: deferred (staged into CorridorKey)"