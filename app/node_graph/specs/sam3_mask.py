"""SAM3 Mask node spec."""

from .base import NodeSpec, PortSpec

SPEC = NodeSpec(
    key="sam3",
    title="SAM3 Mask",
    subtitle="Mask creation with concept prompts",
    title_i18n_key="node_graph_node_sam3",
    subtitle_i18n_key="node_graph_node_sam3_subtitle",
    inputs=(PortSpec("img", "image", label="img"),),
    outputs=(PortSpec("out", "alpha", label="mask"),),
    allowed_targets=set(),
    default_properties={
        "enabled": True,
        "note": "",
        "model_type": "sam3",
        "concept": "",
        "sam_status": "",
    },
)
