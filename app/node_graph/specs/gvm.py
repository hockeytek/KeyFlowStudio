"""GVM node specification.

GVM (Generative Video Matting, SIGGRAPH 2025) is a diffusion-based video
matting model that generates temporally consistent alpha hints for CorridorKey
without requiring manual interaction.
"""
from .base import NodeSpec, PortSpec

SPEC = NodeSpec(
    key="gvm",
    title="GVM",
    subtitle="",
    title_i18n_key="node_graph_node_gvm",
    subtitle_i18n_key="",

    # Input ports
    inputs=(
        PortSpec("image", "image", label="Image", required=True),
    ),

    # Output ports
    outputs=(
        PortSpec("alpha", "alpha", label="Alpha Hint"),
    ),

    allowed_targets=set(),

    default_properties={
        "enabled": True,
        "note": "",

        # Inference parameters
        "num_frames_per_batch": 8,   # Reduce to 4 if OOM
        "denoise_steps": 1,          # 1-step is standard
        "decode_chunk_size": 4,      # Reduce if OOM
        "num_overlap_frames": 1,     # Temporal consistency overlap
        "num_interp_frames": 1,      # Interpolation between batches
        "noise_type": "zeros",       # 'zeros' or 'gaussian'
        "use_clip_img_emb": False,   # Use CLIP image embedding
        "dilate_radius": 0,          # Post-process dilate (pixels)
    },
)
