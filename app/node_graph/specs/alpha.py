"""Alpha node spec."""

from .base import NodeSpec, PortSpec

SPEC = NodeSpec(
    key="alpha",
    title="Alpha",
    subtitle="External Alpha / Mask",
    title_i18n_key="node_graph_node_alpha",
    subtitle_i18n_key="node_graph_node_alpha_subtitle",
    inputs=(),
    outputs=(PortSpec("out", "alpha", label="alpha"),),
    allowed_targets=set(),
    default_properties={"enabled": True, "note": "", "media_type": "image", "path": ""},
)
