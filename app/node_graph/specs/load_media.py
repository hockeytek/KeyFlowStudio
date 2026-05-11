"""Load Media node spec."""

from .base import NodeSpec, PortSpec

SPEC = NodeSpec(
    key="load",
    title="Read",
    subtitle="Source Media",
    title_i18n_key="node_graph_node_load",
    subtitle_i18n_key="node_graph_node_load_subtitle",
    inputs=(),
    outputs=(PortSpec("out", "image", label="img"),),
    allowed_targets=set(),
    default_properties={"enabled": True, "note": "", "media_type": "video", "path": ""},
)
