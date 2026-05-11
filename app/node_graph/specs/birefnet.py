"""BiRefNet node specification.

Defines the structure, ports, and properties of the BiRefNet node in the graph.
BiRefNet is a lightweight background removal model used to generate alpha hints
for CorridorKey.
"""
from .base import NodeSpec, PortSpec

SPEC = NodeSpec(
    key="birefnet",
    title="BiRefNet",
    subtitle="",
    title_i18n_key="node_graph_node_birefnet",
    subtitle_i18n_key="node_graph_node_birefnet_subtitle",
    
    # Input ports
    inputs=(
        PortSpec("image", "image", label="Image", required=True),
    ),
    
    # Output ports
    outputs=(
        PortSpec("alpha", "alpha", label="Alpha"),
    ),
    
    # Which nodes this can connect to
    allowed_targets=set(),
    
    # Default properties
    default_properties={
        "enabled": True,
        "note": "",
        
        # Model selection
        "usage": "General",  # General, Matting, Portrait, etc.
        "half_precision": True,  # Uses float16 on CUDA (faster)
        "dilate_radius": 0,
        "erode_radius": 0,
    },
)
