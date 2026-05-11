"""Alpha node runtime handler."""

from __future__ import annotations

import os

from app.node_graph.models import GraphNode
from app.node_graph.nodes.base import NodeExecutionContext, NodeExecutionError
from app.node_graph.nodes.load_media_node import LoadMediaNodeHandler


class AlphaNodeHandler(LoadMediaNodeHandler):
    key = "alpha"

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, dict],
        context: NodeExecutionContext,
    ) -> dict[str, dict]:
        props = node.properties or {}
        media_path = str(props.get("path", "")).strip()
        media_type = str(props.get("media_type", "image")).strip().lower()

        if not media_path:
            raise NodeExecutionError("Alpha: path is empty")
        if not os.path.exists(media_path):
            raise NodeExecutionError(f"Alpha: file not found: {media_path}")

        metadata = self._collect_metadata(media_path, media_type)
        payload = {
            "media_path": media_path,
            "media_type": media_type,
            "metadata": metadata,
        }
        context.state["alpha_media"] = payload
        return {"out": payload}
