"""BiRefNet alpha mask generation node runtime handler.

This handler prepares BiRefNet inference context for execution in the node graph.
BiRefNet generates coarse alpha masks that serve as hints for CorridorKey.
"""
from __future__ import annotations

from app.node_graph.models import GraphNode
from app.node_graph.nodes.base import NodeExecutionContext, NodeExecutionError


class BiRefNetNodeHandler:
    """Handler for BiRefNet node execution in the graph."""
    
    key = "birefnet"

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, dict],
        context: NodeExecutionContext,
    ) -> dict[str, dict]:
        """Validate inputs and prepare BiRefNet alpha generation context.
        
        Input ports:
            - image: RGB image from upstream node (required)
        
        Output ports:
            - alpha: Grayscale alpha mask (0-1)
        
        Args:
            node: The BiRefNet node definition
            inputs: Input port data from upstream nodes
            context: Node execution context
        
        Returns:
            dict with output port data prepared for inference worker
        
        Raises:
            NodeExecutionError: If required input ports are missing
        """
        # Get upstream image data (required)
        img_data = inputs.get("image", {})
        if not img_data:
            raise NodeExecutionError(
                "BiRefNet: missing 'image' input port. "
                "Connect Load Media or another image source."
            )
        
        # Extract node properties with defaults
        props = node.properties or {}
        
        # Return output port data with all necessary context
        return {
            "alpha": {
                "media_path": img_data.get("media_path", ""),
                "node_type": "birefnet",
                
                # BiRefNet-specific parameters
                "birefnet": {
                    "usage": str(props.get("usage", "General")),
                    "half_precision": bool(props.get("half_precision", True)),
                },
                
                # Reference to upstream output for worker to fetch
                "upstream_image": img_data,
            }
        }
