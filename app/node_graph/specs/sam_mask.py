"""SAM Mask node spec."""

from .base import NodeSpec, PortSpec

SPEC = NodeSpec(
    key="sam2",
    title="SAM2 Mask",
    subtitle="Mask creation from clicks",
    title_i18n_key="node_graph_node_sam",
    subtitle_i18n_key="node_graph_node_sam_subtitle",
    inputs=(PortSpec("img", "image", label="img"),),
    outputs=(PortSpec("out", "alpha", label="mask"),),
    allowed_targets=set(),
    default_properties={
        "enabled": True,
        "note": "",
        "backend": "sam2",
        "model_type": "vit_h",
        "live_sam2": False,
        "point_mode": "positive",
        "clear_points_before_run": False,
        "sam_status": "",
        "mask_items": [],
        "selected_mask_rows": [],
        "current_mask_ready": False,
    },
)
