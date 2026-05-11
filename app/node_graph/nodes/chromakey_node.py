"""HSV Chroma Key node runtime handler.

Passes image frames downstream; actual OpenCV processing happens in InferenceWorker.
"""
from __future__ import annotations

from app.node_graph.models import GraphNode
from app.node_graph.nodes.base import NodeExecutionContext, NodeExecutionError


class ChromaKeyNodeHandler:
    """Handler for HSV Chroma Key node execution in the graph."""

    key = "chromakey"

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, dict],
        context: NodeExecutionContext,
    ) -> dict[str, dict]:
        """Validate inputs and prepare chroma key context for the inference worker.

        Input ports:
            - image: RGB image from upstream node (required)

        Output ports:
            - mask: Foreground mask (1 = foreground, 0 = chroma screen)

        Raises:
            NodeExecutionError: If required input port is missing
        """
        img_data = inputs.get("image", {})
        if not img_data:
            raise NodeExecutionError(
                "HSV Chroma Key: missing 'image' input port. "
                "Connect Load Media or another image source."
            )

        props = node.properties or {}

        return {
            "mask": {
                "media_path": img_data.get("media_path", ""),
                "node_type": "chromakey",
                "chromakey": {
                    "hue_center": int(props.get("hue_center", 120)),
                    "hue_range": int(props.get("hue_range", 30)),
                    "saturation_min": float(props.get("saturation_min", 0.15)),
                    "value_min": float(props.get("value_min", 0.10)),
                    "blur_radius": int(props.get("blur_radius", 3)),
                },
                "upstream_image": img_data,
            }
        }
