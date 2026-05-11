"""SAM Mask node runtime handler."""

from __future__ import annotations

from app.node_graph.models import GraphNode
from app.node_graph.nodes.base import NodeExecutionContext, NodeExecutionError


class SamMaskNodeHandler:
    key = "sam2"

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, dict],
        context: NodeExecutionContext,
    ) -> dict[str, dict]:
        img_data = inputs.get("img", {})
        if not img_data:
            raise NodeExecutionError("SAM2 Mask: missing upstream media on 'img' port")

        props = node.properties or {}
        return {"out": {
            "media_path": img_data.get("media_path", ""),
            "sam2": {
                "auto_refine": bool(props.get("auto_refine", True)),
                "multimask": bool(props.get("multimask", True)),
                "live_sam2": bool(props.get("live_sam2", False)),
                "point_mode": str(props.get("point_mode", "positive")),
                "clear_points_before_run": bool(props.get("clear_points_before_run", False)),
            },
            "upstream": img_data,
        }}
