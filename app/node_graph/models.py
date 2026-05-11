"""Shared graph models for node-graph execution."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GraphNode:
    id: str
    type: str
    title: str
    properties: dict = field(default_factory=dict)
    enabled: bool = True


@dataclass(frozen=True)
class GraphEdge:
    src_id: str
    dst_id: str
    src_port: str = "out"
    dst_port: str = ""
