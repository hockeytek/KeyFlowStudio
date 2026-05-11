"""Node specification model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PortSpec:
    """Typed input or output port on a node (Nuke-style)."""
    name: str
    data_type: str   # "image", "mask", "alpha"
    label: str = ""
    required: bool = True


# Colour constants shared across rendering layers.
PORT_COLORS: dict[str, dict[str, str]] = {
    "image": {"border": "#78d8ff", "fill": "#153246", "border_hl": "#d6f3ff", "fill_hl": "#2f7ea3"},
    "mask":  {"border": "#a3a3a3", "fill": "#2a2a2a", "border_hl": "#ececec", "fill_hl": "#666666"},
    "alpha": {"border": "#a3a3a3", "fill": "#2a2a2a", "border_hl": "#ececec", "fill_hl": "#666666"},
}
DEFAULT_PORT_COLORS = PORT_COLORS["image"]

EDGE_COLORS: dict[str, str] = {
    "image": "#57b8e9",
    "mask":  "#e8943b",
    "alpha": "#a0a0a0",
}
DEFAULT_EDGE_COLOR = "#57b8e9"


@dataclass(frozen=True)
class NodeSpec:
    key: str
    title: str
    subtitle: str
    title_i18n_key: str
    subtitle_i18n_key: str = ""
    inputs: tuple[PortSpec, ...] = ()
    outputs: tuple[PortSpec, ...] = ()
    # Empty set means there is no explicit target whitelist and graph wiring is
    # governed by port-type compatibility only.
    allowed_targets: set[str] = field(default_factory=set)
    default_properties: dict = field(default_factory=dict)
