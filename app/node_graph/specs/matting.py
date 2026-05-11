"""MatAnyone2 node spec."""

from .base import NodeSpec, PortSpec

SPEC = NodeSpec(
    key="matting",
    title="MatAnyone2",
    subtitle="Matting inference",
    title_i18n_key="node_graph_node_matting",
    subtitle_i18n_key="node_graph_node_matting_subtitle",
    inputs=(PortSpec("img", "image", label="img"), PortSpec("mask", "mask", label="mask")),
    outputs=(PortSpec("fg", "image", label="fg"), PortSpec("alpha", "alpha", label="alpha")),
    allowed_targets=set(),
    default_properties={
        "enabled": True,
        "note": "",
        "preset": "Eval HR (1080p)",
        "erode": 15,
        "dilate": 15,
        "warmup": 10,
        "fg_background": "green",
    },
)
