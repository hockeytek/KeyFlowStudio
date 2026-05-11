"""CorridorKey node specification.

Defines the structure, ports, and properties of the CorridorKey node in the graph.
CorridorKey is a professional neural network for green screen removal with
accurate color restoration.
"""
from .base import NodeSpec, PortSpec

SPEC = NodeSpec(
    key="corridorkey",
    title="CorridorKey",
    subtitle="",
    title_i18n_key="node_graph_node_corridorkey",
    subtitle_i18n_key="node_graph_node_corridorkey_subtitle",
    
    # Input ports
    inputs=(
        PortSpec("image", "image", label="Image", required=True),
        PortSpec("alphahint", "alpha", label="Alpha Hint", required=True),
    ),
    
    # Output ports
    outputs=(
        PortSpec("alpha", "alpha", label="Alpha"),
        PortSpec("fg", "image", label="FG Clean"),
        PortSpec("comp", "image", label="Preview"),
        PortSpec("processed", "image", label="Premult RGBA"),
    ),
    
    # Which nodes this can connect to
    allowed_targets=set(),
    
    # Default properties with sliders, checkboxes, etc.
    default_properties={
        "enabled": True,
        "note": "",
        "preset": "balanced",
        "alpha_hint_mode": "auto",  # auto|batch|staged
        "input_colorspace": "auto",  # auto|srgb|linear
        "screen_color": "green",  # green|blue|auto; green preserves older project behavior
        
        # Main parameters
        "despill_strength": 0.5,  # 0-1, removes green spill
        "despeckle": True,         # morphological cleanup
        "despeckle_size": 400,     # pixels
        "matte_clip_black": 0.0,
        "matte_clip_white": 1.0,
        "matte_shrink_grow": 0.0,
        "matte_edge_blur": 0.0,
        "matte_gamma": 1.0,
        "temporal_smoothing": 0.0,
        "refiner_strength": 1.0,   # 0-2, CNN edge enhancement
        "use_refiner": True,       # enable CNN refiner
        "output_mode": "processed",  # processed|comp|fg|alpha
        "hint_dilate_radius": 0,     # 0=off; pixels to dilate alpha hint before CorridorKey
    },
)
