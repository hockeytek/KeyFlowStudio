"""Node contracts: Central definition of all node interaction rules.

This module defines the semantic contract for every node type in the graph,
including ports, data types, required fields, and interaction rules with
downstream/upstream nodes.

Purpose: Instead of rules scattered across engine.py, inference_worker.py,
and matting_orchestrator.py, all node semantics are defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.constants import (
    DEFAULT_JPG_QUALITY,
    DEFAULT_PNG_BIT_DEPTH,
    DEFAULT_PNG_COMPRESSION,
    DEFAULT_VIDEO_CODEC,
    DEFAULT_VIDEO_CRF,
    DEFAULT_VIDEO_PRESET,
)
from app.node_graph.specs import NODE_SPECS


@dataclass
class PortContract:
    """Port specification: name, type, required flag."""

    name: str
    data_type: str
    required: bool = True
    label: str = ""


@dataclass
class NodeContract:
    """Full semantic contract for a node type."""

    node_type: str
    title: str
    subtitle: str = ""
    inputs: list = None  # type: ignore[assignment]
    outputs: list = None  # type: ignore[assignment]
    default_properties: dict = None  # type: ignore[assignment]
    execution_rules: dict = None  # type: ignore[assignment]
    upstream_allowed: list = None  # type: ignore[assignment]
    downstream_allowed: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.inputs is None:
            self.inputs = []
        if self.outputs is None:
            self.outputs = []
        if self.default_properties is None:
            self.default_properties = {}
        if self.execution_rules is None:
            self.execution_rules = {}
        if self.upstream_allowed is None:
            self.upstream_allowed = []
        if self.downstream_allowed is None:
            self.downstream_allowed = []

    def get_input(self, port_name: str) -> Optional[PortContract]:
        """Get input port by name."""
        for port in self.inputs:
            if port.name == port_name:
                return port
        return None

    def get_output(self, port_name: str) -> Optional[PortContract]:
        """Get output port by name."""
        for port in self.outputs:
            if port.name == port_name:
                return port
        return None


def _inputs_from_spec(node_type: str) -> list[PortContract]:
    """Build input contracts from NodeSpec to avoid duplicated input declarations."""
    spec = NODE_SPECS.get(node_type)
    if spec is None:
        return []
    return [
        PortContract(
            name=port.name,
            data_type=port.data_type,
            required=port.required,
            label=port.label,
        )
        for port in spec.inputs
    ]


def _outputs_from_spec(node_type: str) -> list[PortContract]:
    """Build output contracts from NodeSpec to avoid duplicated output declarations."""
    spec = NODE_SPECS.get(node_type)
    if spec is None:
        return []
    return [
        PortContract(
            name=port.name,
            data_type=port.data_type,
            required=port.required,
            label=port.label,
        )
        for port in spec.outputs
    ]


def _title_from_spec(node_type: str, fallback: str) -> str:
    spec = NODE_SPECS.get(node_type)
    if spec is None:
        return fallback
    return str(spec.title or fallback)


def _subtitle_from_spec(node_type: str) -> str:
    spec = NODE_SPECS.get(node_type)
    if spec is None:
        return ""
    return str(spec.subtitle or "")


# ── Node contract definitions ────────────────────────────────────────────────

SOURCE = NodeContract(
    node_type="source",
    title=_title_from_spec("source", "Source"),
    subtitle=_subtitle_from_spec("source"),
    inputs=_inputs_from_spec("source"),
    outputs=_outputs_from_spec("source"),
    default_properties={"enabled": True, "note": "", "media_type": "video", "path": ""},
    downstream_allowed=["load", "sam2", "sam3", "birefnet", "gvm", "chromakey", "corridorkey", "matting", "alpha", "merge", "export"],
)

LOAD = NodeContract(
    node_type="load",
    title=_title_from_spec("load", "Load Media"),
    subtitle=_subtitle_from_spec("load"),
    inputs=_inputs_from_spec("load"),
    outputs=_outputs_from_spec("load"),
    default_properties={"enabled": True, "note": "", "media_type": "video", "path": ""},
    downstream_allowed=["sam2", "sam3", "birefnet", "gvm", "chromakey", "corridorkey", "matting", "alpha", "merge", "export"],
)

SAM = NodeContract(
    node_type="sam2",
    title=_title_from_spec("sam2", "SAM2 Mask"),
    subtitle=_subtitle_from_spec("sam2"),
    inputs=_inputs_from_spec("sam2"),
    outputs=_outputs_from_spec("sam2"),
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
    execution_rules={
        "auto_propagate_before_run": True,
        "requires_frame_selection": True,
        # When all SAM outputs lead only to corridorkey.alphahint, masks are
        # read per-frame from disk instead of loading all into RAM at once.
        "can_defer_disk_masks": True,
        "defers_when": "outputs_only_to_corridorkey.alphahint",
    },
    downstream_allowed=["alpha", "chromakey", "export"],
)

SAM3 = NodeContract(
    node_type="sam3",
    title=_title_from_spec("sam3", "SAM3 Mask"),
    subtitle=_subtitle_from_spec("sam3"),
    inputs=_inputs_from_spec("sam3"),
    outputs=_outputs_from_spec("sam3"),
    default_properties={
        "enabled": True,
        "note": "",
        "model_type": "sam3",
        "concept": "",
        "sam_status": "",
    },
    execution_rules={
        "requires_frame_selection": True,
    },
    downstream_allowed=["alpha", "chromakey", "export"],
)

BIREFNET = NodeContract(
    node_type="birefnet",
    title=_title_from_spec("birefnet", "BiRefNet"),
    subtitle=_subtitle_from_spec("birefnet"),
    inputs=_inputs_from_spec("birefnet"),
    outputs=_outputs_from_spec("birefnet"),
    default_properties={
        "enabled": True,
        "note": "",
        "usage": "General",
        "half_precision": True,
        "dilate_radius": 0,
        "erode_radius": 0,
    },
    execution_rules={
        "can_defer_to_downstream": True,
        "defers_when": "outputs_only_to_corridorkey.alphahint",
        "binarization_threshold": 10,
        "unload_after_batch": True,
    },
    downstream_allowed=["corridorkey", "export"],
)

GVM = NodeContract(
    node_type="gvm",
    title=_title_from_spec("gvm", "GVM"),
    subtitle=_subtitle_from_spec("gvm"),
    inputs=_inputs_from_spec("gvm"),
    outputs=_outputs_from_spec("gvm"),
    default_properties={
        "enabled": True,
        "note": "",
        "num_frames_per_batch": 8,
        "denoise_steps": 1,
        "decode_chunk_size": 4,
        "num_overlap_frames": 1,
        "num_interp_frames": 1,
        "noise_type": "zeros",
        "use_clip_img_emb": False,
        "dilate_radius": 0,
    },
    execution_rules={
        "can_defer_to_downstream": True,
        "defers_when": "outputs_only_to_corridorkey.alphahint",
        "unload_after_batch": True,
    },
    downstream_allowed=["corridorkey", "export"],
)

CHROMAKEY = NodeContract(
    node_type="chromakey",
    title=_title_from_spec("chromakey", "ChromaKey"),
    subtitle=_subtitle_from_spec("chromakey"),
    inputs=_inputs_from_spec("chromakey"),
    outputs=_outputs_from_spec("chromakey"),
    default_properties={
        "enabled": True,
        "note": "",
        "hue_center": 120,
        "hue_range": 30,
        "saturation_min": 0.15,
        "value_min": 0.10,
        "blur_radius": 3,
    },
    downstream_allowed=["corridorkey", "matting", "export"],
)

CORRIDORKEY = NodeContract(
    node_type="corridorkey",
    title=_title_from_spec("corridorkey", "CorridorKey"),
    subtitle=_subtitle_from_spec("corridorkey"),
    inputs=_inputs_from_spec("corridorkey"),
    outputs=_outputs_from_spec("corridorkey"),
    default_properties={
        "enabled": True,
        "note": "",
        "preset": "balanced",
        "alpha_hint_mode": "auto",
        "input_colorspace": "auto",
        "screen_color": "green",
        "despill_strength": 0.5,
        "despeckle": True,
        "despeckle_size": 400,
        "matte_clip_black": 0.0,
        "matte_clip_white": 1.0,
        "matte_shrink_grow": 0.0,
        "matte_edge_blur": 0.0,
        "matte_gamma": 1.0,
        "temporal_smoothing": 0.0,
        "refiner_strength": 1.0,
        "use_refiner": True,
        "output_mode": "processed",
        "hint_dilate_radius": 0,
    },
    execution_rules={
        "requires_matching_frame_count": True,
        "frame_count_mismatch_error": "err_corridorkey_frame_mismatch",
        "staged_mode_reason": "BiRefNet -> CorridorKey pipeline",
    },
    upstream_allowed=["load", "source", "birefnet", "gvm", "chromakey"],
    downstream_allowed=["alpha", "matting", "merge", "export"],
)

MATTING = NodeContract(
    node_type="matting",
    title=_title_from_spec("matting", "Matting"),
    subtitle=_subtitle_from_spec("matting"),
    inputs=_inputs_from_spec("matting"),
    outputs=_outputs_from_spec("matting"),
    default_properties={
        "enabled": True,
        "note": "",
        "preset": "Eval HR (1080p)",
        "erode": 15,
        "dilate": 15,
        "warmup": 10,
        "fg_background": "green",
    },
    upstream_allowed=["load", "source", "corridorkey"],
    downstream_allowed=["merge", "export"],
)

ALPHA = NodeContract(
    node_type="alpha",
    title=_title_from_spec("alpha", "Alpha"),
    subtitle=_subtitle_from_spec("alpha"),
    inputs=_inputs_from_spec("alpha"),
    outputs=_outputs_from_spec("alpha"),
    default_properties={
        "enabled": True,
        "note": "",
        "media_type": "image",
        "path": "",
    },
    downstream_allowed=["export"],
)

EXPORT = NodeContract(
    node_type="export",
    title=_title_from_spec("export", "Export"),
    subtitle=_subtitle_from_spec("export"),
    inputs=_inputs_from_spec("export"),
    outputs=_outputs_from_spec("export"),
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
    upstream_allowed=["load", "source", "sam2", "sam3", "birefnet", "gvm", "chromakey", "corridorkey", "matting", "alpha", "merge"],
    downstream_allowed=[],
)

MERGE = NodeContract(
    node_type="merge",
    title=_title_from_spec("merge", "Merge"),
    subtitle=_subtitle_from_spec("merge"),
    inputs=_inputs_from_spec("merge"),
    outputs=_outputs_from_spec("merge"),
    default_properties={
        "enabled": True,
        "note": "",
        "mode": "over",
        "opacity": 1.0,
        "mix": 1.0,
        "mask_enabled": True,
        "mask_channel": "auto",
        "mask_inject": False,
        "invert_mask": False,
        "fringe": False,
        "alpha_masking": True,
        "set_bbox_to": "union",
    },
    upstream_allowed=[],
    downstream_allowed=["merge", "export"],
)

# Master registry — keyed by node_type
ALL_NODE_CONTRACTS: dict[str, NodeContract] = {
    "source": SOURCE,
    "load": LOAD,
    "sam2": SAM,
    "sam3": SAM3,
    "birefnet": BIREFNET,
    "gvm": GVM,
    "chromakey": CHROMAKEY,
    "corridorkey": CORRIDORKEY,
    "matting": MATTING,
    "alpha": ALPHA,
    "merge": MERGE,
    "export": EXPORT,
}


def get_contract(node_type: str) -> Optional[NodeContract]:
    """Get contract for a node type."""
    return ALL_NODE_CONTRACTS.get(node_type)


def all_node_types() -> list[str]:
    """Get all available node types."""
    return list(ALL_NODE_CONTRACTS.keys())
