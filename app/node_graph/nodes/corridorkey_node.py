"""CorridorKey chromakeying node runtime handler.

This handler prepares CorridorKey inference context for execution in the node graph.
It validates input ports and extracts parameters, but the actual inference happens
in the InferenceWorker (QThread).
"""
from __future__ import annotations

from app.node_graph.models import GraphNode
from app.node_graph.nodes.base import NodeExecutionContext, NodeExecutionError


class CorridorKeyNodeHandler:
    """Handler for CorridorKey node execution in the graph."""
    
    key = "corridorkey"

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, dict],
        context: NodeExecutionContext,
    ) -> dict[str, dict]:
        """Validate inputs and prepare CorridorKey inference context.
        
        Input ports:
            - image: RGB image from upstream node (required)
            - alphahint: Coarse alpha mask (optional)
        
        Output ports:
            - alpha: Alpha matte
            - fg: Clean straight foreground for comp
            - comp: Preview composite for UI review only
            - processed: Premultiplied RGBA convenience output
        
        Args:
            node: The CorridorKey node definition
            inputs: Input port data from upstream nodes
            context: Node execution context
        
        Returns:
            dict with output port data prepared for inference worker
        
        Raises:
            NodeExecutionError: If required input ports are missing or invalid
        """
        # Get upstream image data (required)
        img_data = inputs.get("image", {})
        if not img_data:
            raise NodeExecutionError(
                "CorridorKey: missing 'image' input port. "
                "Connect Load Media or another image source."
            )
        
        # Alpha hint is mandatory for CorridorKey workflow in this project.
        alphahint_data = inputs.get("alphahint", {})
        if not alphahint_data:
            raise NodeExecutionError(
                "CorridorKey: missing 'alphahint' input port. "
                "Connect an alpha-hint source to CorridorKey Alpha Hint."
            )
        
        # Extract node properties with defaults
        props = node.properties or {}
        
        # Shared payload for all CorridorKey output ports.
        payload = {
            "media_path": img_data.get("media_path", ""),
            "node_type": "corridorkey",
            "corridorkey": {
                "despill_strength": float(props.get("despill_strength", 5.0)),
                "despeckle": bool(props.get("despeckle", True)),
                "despeckle_size": int(props.get("despeckle_size", 400)),
                "refiner_strength": float(props.get("refiner_strength", 1.0)),
                "use_refiner": bool(props.get("use_refiner", True)),
            },
            "upstream_image": img_data,
            "upstream_alphahint": alphahint_data,
        }

        return {
            "alpha": dict(payload),
            "fg": dict(payload),
            "comp": dict(payload),
            "processed": dict(payload),
        }
