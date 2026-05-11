"""Base interfaces for node handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.node_graph.models import GraphNode


class NodeExecutionError(RuntimeError):
    """Execution failure for a node."""


@dataclass
class NodeExecutionContext:
    language: str = "en"
    state: dict = field(default_factory=dict)


class NodeHandler(Protocol):
    key: str

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, dict],
        context: NodeExecutionContext,
    ) -> dict[str, dict]:
        """Execute node and return output payload keyed by output port name."""
