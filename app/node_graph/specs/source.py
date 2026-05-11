"""Source node spec - the primary media source for processing."""

from .base import NodeSpec, PortSpec

SPEC = NodeSpec(
    key="source",
    title="Source",
    subtitle="Primary Media Source",
    title_i18n_key="node_graph_node_source",
    subtitle_i18n_key="node_graph_node_source_subtitle",
    inputs=(),
    outputs=(PortSpec("out", "image", label="img"),),
    allowed_targets=set(),
    default_properties={"enabled": True, "note": "", "media_type": "video", "path": ""},
)
