"""Planning helpers for graph Write/export targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.node_graph.models import GraphEdge, GraphNode
from app.node_graph.specs import get_node_spec
from app.utils.write_paths import (
    get_port_output_label,
    normalize_write_stream_name,
    resolve_graph_write_output_dir,
)


@dataclass(frozen=True)
class GraphWritePlanTarget:
    node_id: str
    source_node_id: str
    stream_label: str
    target_dir: Path
    write_cfg: dict[str, Any]


def build_graph_write_plan_targets(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    output_dir: Path,
) -> list[GraphWritePlanTarget]:
    node_types_by_id = {str(node.id): str(node.type) for node in nodes}
    node_titles_by_id = {str(node.id): str(node.title or node.type) for node in nodes}
    source_ports_by_dst: dict[str, str] = {}
    source_nodes_by_dst: dict[str, str] = {}

    for edge in edges:
        dst_id = str(edge.dst_id)
        src_id = str(edge.src_id)
        edge_dst_port = str(edge.dst_port or "").strip().lower()
        dst_type = node_types_by_id.get(dst_id, "")
        if edge_dst_port != "in" and not (dst_type == "export" and not edge_dst_port):
            continue

        edge_src_port = str(edge.src_port or "").strip().lower()
        if not edge_src_port:
            src_type = node_types_by_id.get(src_id, "")
            src_spec = get_node_spec(src_type)
            if src_spec is not None and len(src_spec.outputs) == 1:
                edge_src_port = str(src_spec.outputs[0].name or "out").strip().lower()
            else:
                edge_src_port = "out"
        source_ports_by_dst[dst_id] = edge_src_port
        source_nodes_by_dst[dst_id] = src_id

    targets: list[GraphWritePlanTarget] = []
    for node in nodes:
        if node.type != "export" or not node.enabled:
            continue

        node_id = str(node.id)
        source_node_id = str(source_nodes_by_dst.get(node_id, "")).strip()
        if not source_node_id:
            continue

        src_type = node_types_by_id.get(source_node_id, "")
        src_port = source_ports_by_dst.get(node_id, "")
        stream_label = normalize_write_stream_name(source_node_type=src_type, source_port=src_port)
        if not stream_label:
            continue

        source_node_title = node_titles_by_id.get(source_node_id, "")
        port_label = get_port_output_label(src_type, src_port)
        write_cfg = dict(node.properties or {})
        target_dir = resolve_graph_write_output_dir(
            write_cfg,
            output_dir,
            stream_label,
            source_node_title,
            port_label,
        )
        write_cfg["output_dir"] = str(target_dir)
        targets.append(
            GraphWritePlanTarget(
                node_id=node_id,
                source_node_id=source_node_id,
                stream_label=stream_label,
                target_dir=target_dir,
                write_cfg=write_cfg,
            )
        )
    return targets