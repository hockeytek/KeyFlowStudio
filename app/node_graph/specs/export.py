"""Export node spec."""

from app.constants import (
    DEFAULT_JPG_QUALITY,
    DEFAULT_PNG_BIT_DEPTH,
    DEFAULT_PNG_COMPRESSION,
    DEFAULT_VIDEO_CODEC,
    DEFAULT_VIDEO_CRF,
    DEFAULT_VIDEO_PRESET,
)

from .base import NodeSpec, PortSpec

SPEC = NodeSpec(
    key="export",
    title="Write",
    subtitle="Render to Disk",
    title_i18n_key="node_graph_node_export",
    subtitle_i18n_key="node_graph_node_export_subtitle",
    inputs=(PortSpec("in", "image", label="Input", required=False),),
    outputs=(),
    allowed_targets=set(),
    default_properties={
        "enabled": True,
        "note": "",
        "auto_output_dir": True,
        "output_dir": "",
        "file_name": "",
        "output_format": "source",
        "video_codec": DEFAULT_VIDEO_CODEC,
        "video_quality": DEFAULT_VIDEO_CRF,
        "video_preset": DEFAULT_VIDEO_PRESET,
        "png_compression": DEFAULT_PNG_COMPRESSION,
        "png_bit_depth": DEFAULT_PNG_BIT_DEPTH,
        "png_embed_alpha": False,
        "jpg_quality": DEFAULT_JPG_QUALITY,
    },
)
