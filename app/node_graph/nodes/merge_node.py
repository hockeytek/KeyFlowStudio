"""Merge node runtime handler.

Validates fg/bg/(optional mask) inputs and passes blend parameters downstream to
InferenceWorker. Actual pixel compositing is performed in
InferenceWorker._execute_merge_node.
"""
from __future__ import annotations

from app.node_graph.models import GraphNode
from app.node_graph.nodes.base import NodeExecutionContext, NodeExecutionError

_VALID_MODES = frozenset({
    "over", "under", "atop", "in", "out", "mask", "stencil", "matte", "xor",
    "copy", "conjoint-over", "disjoint-over", "plus", "hypot", "average",
    "multiply", "divide", "minus", "from", "screen", "overlay", "hard-light",
    "soft-light", "difference", "exclusion", "min", "max", "color-burn",
    "color-dodge", "reflect", "geometric", "pinlight",
    "add", "subtract", "darken", "lighten",
})


class MergeNodeHandler:
    """Handler for Merge (compositing) node execution in the graph."""

    key = "merge"

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, dict],
        context: NodeExecutionContext,
    ) -> dict[str, dict]:
        """Validate inputs and prepare merge context for the inference worker.

        Input ports:
            - fg: foreground image (required)
            - bg: background image (required)
            - mask: optional mask/alpha limiting where the merge applies

        Output ports:
            - out: composited image

        Raises:
            NodeExecutionError: If required input port is missing
        """
        fg_data = inputs.get("fg", {})
        bg_data = inputs.get("bg", {})
        mask_data = inputs.get("mask", {})

        if not fg_data:
            raise NodeExecutionError(
                "Merge: missing 'fg' input port. "
                "Connect an image source (Load, Source, CorridorKey, MatAnyone2)."
            )
        if not bg_data:
            raise NodeExecutionError(
                "Merge: missing 'bg' input port. "
                "Connect a background image source."
            )

        props = node.properties or {}
        mode = str(props.get("mode", "over")).strip().lower()
        if mode not in _VALID_MODES:
            mode = "over"
        opacity = float(props.get("opacity", 1.0))
        opacity = max(0.0, min(1.0, opacity))
        mix = max(0.0, min(1.0, float(props.get("mix", 1.0))))
        mask_enabled = bool(props.get("mask_enabled", True))
        mask_channel = str(props.get("mask_channel", "auto")).strip().lower()
        mask_inject = bool(props.get("mask_inject", False))
        invert_mask = bool(props.get("invert_mask", False))
        fringe = bool(props.get("fringe", False))
        alpha_masking = bool(props.get("alpha_masking", True))
        set_bbox_to = str(props.get("set_bbox_to", "union")).strip().lower()
        if set_bbox_to not in {"union", "intersection", "a", "b"}:
            set_bbox_to = "union"

        return {
            "out": {
                "media_path": bg_data.get("media_path", ""),
                "node_type": "merge",
                "merge": {
                    "mode": mode,
                    "opacity": opacity,
                    "mix": mix,
                    "mask_enabled": mask_enabled,
                    "mask_channel": mask_channel,
                    "mask_inject": mask_inject,
                    "invert_mask": invert_mask,
                    "fringe": fringe,
                    "alpha_masking": alpha_masking,
                    "set_bbox_to": set_bbox_to,
                },
                "upstream_fg": fg_data,
                "upstream_bg": bg_data,
                "upstream_mask": mask_data,
            }
        }
