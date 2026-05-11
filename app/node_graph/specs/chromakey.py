"""HSV Chroma Key node specification.

Defines the structure, ports, and properties of the Chroma Key node in the graph.
Performs pure OpenCV HSV-based green screen removal — no neural network weights required.
Outputs a foreground mask (1 = foreground, 0 = chroma screen) suitable as alphahint
for CorridorKey or as a standalone alpha matte.
"""
from .base import NodeSpec, PortSpec

SPEC = NodeSpec(
    key="chromakey",
    title="HSV Chroma Key",
    subtitle="",
    title_i18n_key="node_graph_node_chromakey",
    subtitle_i18n_key="node_graph_node_chromakey_subtitle",

    # Input ports
    inputs=(
        PortSpec("image", "image", label="Image", required=True),
    ),

    # Output ports
    outputs=(
        PortSpec("mask", "mask", label="Mask"),
    ),

    # Which nodes this can connect to
    allowed_targets=set(),

    # Default properties
    default_properties={
        "enabled": True,
        "note": "",

        # Hue target in 0–360° (120 = green)
        "hue_center": 120,
        # ± tolerance around hue_center (degrees)
        "hue_range": 30,
        # Minimum saturation to be classified as chroma (0–1)
        "saturation_min": 0.15,
        # Minimum value/brightness to be classified as chroma (0–1)
        "value_min": 0.10,
        # Gaussian blur radius for soft edges (0 = no blur)
        "blur_radius": 3,
    },
)
