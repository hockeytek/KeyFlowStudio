"""Merge node specification.

Composites two image streams using one of 32 operations matching NUKE's Merge node.
Analog to the Merge node in NUKE (Foundry) or the Merge/Blend node in DaVinci Resolve Fusion.

Porter-Duff compositing operators:
    over, under, atop, in, out, mask, stencil, matte, xor, copy,
    conjoint-over, disjoint-over

Additive:
    plus, hypot

Arithmetic blend modes:
    average, multiply, divide, minus, from

Contrast / light blend modes:
    screen, overlay, hard-light, soft-light

Difference blend modes:
    difference, exclusion

Darken / lighten blend modes:
    min (darken), max (lighten)

Dodge / burn blend modes:
    color-burn, color-dodge

Mathematical blend modes:
    reflect, geometric, pinlight

Legacy aliases (transparently remapped at runtime):
    add → plus  |  subtract → minus  |  darken → min  |  lighten → max

Inputs:
    fg (image, required) — foreground / source layer  (A in NUKE notation)
    bg (image, required) — background / destination layer  (B in NUKE notation)
    mask (mask, optional) — limits the merge effect to non-black areas,
        equivalent to Merge.mask in NUKE

Controls:
    mode — compositing / blend operation
    opacity — foreground alpha multiplier before compositing
    mix — dissolve between original B at 0 and full Merge result at 1
    mask_enabled — enable/disable using the optional mask input
    mask_channel — channel extraction mode for the optional mask input
    mask_inject — copy mask into output alpha for downstream reuse
    invert_mask — invert the optional mask input
    fringe — apply the effect only near the mask edge
    alpha_masking — keep blend modes alpha-aware instead of blending alpha numerically
    set_bbox_to — output extents policy: union, intersection, A or B

Output:
    out (image) — composited RGBA result (float32 [0..1])
"""
from .base import NodeSpec, PortSpec

SPEC = NodeSpec(
    key="merge",
    title="Merge",
    subtitle="",
    title_i18n_key="node_graph_node_merge",
    subtitle_i18n_key="node_graph_node_merge_subtitle",

    inputs=(
        PortSpec("fg", "image", label="FG", required=True),
        PortSpec("bg", "image", label="BG", required=True),
        PortSpec("mask", "mask", label="Mask", required=False),
    ),

    outputs=(
        PortSpec("out", "image", label="Out"),
    ),

    allowed_targets=set(),

    default_properties={
        "enabled": True,
        "note": "",
        # Blend / compositing operation — see module docstring for full list
        # Default: over (A over B — standard Porter-Duff source-over)
        "mode": "over",
        # Foreground opacity multiplier applied to A's alpha: 0.0–1.0
        "opacity": 1.0,
        # Dissolve between original B (0.0) and full merge result (1.0)
        "mix": 1.0,
        # Enable/disable mask input influence
        "mask_enabled": True,
        # Channel selection for mask input: auto|luma|red|green|blue|alpha
        "mask_channel": "auto",
        # Copy mask into output alpha channel
        "mask_inject": False,
        # Invert optional external mask before applying it
        "invert_mask": False,
        # Limit masked effect to the mask edge region only
        "fringe": False,
        # Use source-over/screen alpha for blend modes instead of blending alpha values
        "alpha_masking": True,
        # Output bbox policy: union|intersection|a|b
        "set_bbox_to": "union",
    },
)
