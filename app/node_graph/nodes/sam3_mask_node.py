"""SAM3 Mask node runtime handler."""

from __future__ import annotations

from app.node_graph.models import GraphNode
from app.node_graph.nodes.base import NodeExecutionContext, NodeExecutionError


class Sam3MaskNodeHandler:
    key = "sam3"

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, dict],
        context: NodeExecutionContext,
    ) -> dict[str, dict]:
        img_data = inputs.get("img", {})
        if not img_data:
            raise NodeExecutionError("SAM3 Mask: missing upstream media on 'img' port")

        props = node.properties or {}
        return {"out": {
            "media_path": img_data.get("media_path", ""),
            "sam3": {
                "model_type": str(props.get("model_type", "sam3")),
                "concept": str(props.get("concept", "")),
            },
            "upstream": img_data,
        }}
