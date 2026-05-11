"""Node specs registry."""

from __future__ import annotations

from .base import NodeSpec, PORT_COLORS, DEFAULT_PORT_COLORS, EDGE_COLORS, DEFAULT_EDGE_COLOR  # noqa: F401

__all__ = [
    "NodeSpec",
    "PORT_COLORS",
    "DEFAULT_PORT_COLORS",
    "EDGE_COLORS",
    "DEFAULT_EDGE_COLOR",
    "NODE_SPECS",
    "get_node_spec",
    "list_node_specs",
]
from .alpha import SPEC as ALPHA_SPEC
from .birefnet import SPEC as BIREFNET_SPEC
from .chromakey import SPEC as CHROMAKEY_SPEC
from .corridorkey import SPEC as CORRIDORKEY_SPEC
from .export import SPEC as EXPORT_SPEC
from .load_media import SPEC as LOAD_SPEC
from .matting import SPEC as MATTING_SPEC
from .merge import SPEC as MERGE_SPEC
from .sam_mask import SPEC as SAM_SPEC
from .sam3_mask import SPEC as SAM3_SPEC
from .source import SPEC as SOURCE_SPEC
from .gvm import SPEC as GVM_SPEC

NODE_SPECS: dict[str, NodeSpec] = {
    SOURCE_SPEC.key: SOURCE_SPEC,
    LOAD_SPEC.key: LOAD_SPEC,
    ALPHA_SPEC.key: ALPHA_SPEC,
    SAM_SPEC.key: SAM_SPEC,
    SAM3_SPEC.key: SAM3_SPEC,
    MATTING_SPEC.key: MATTING_SPEC,
    BIREFNET_SPEC.key: BIREFNET_SPEC,
    GVM_SPEC.key: GVM_SPEC,
    CHROMAKEY_SPEC.key: CHROMAKEY_SPEC,
    CORRIDORKEY_SPEC.key: CORRIDORKEY_SPEC,
    MERGE_SPEC.key: MERGE_SPEC,
    EXPORT_SPEC.key: EXPORT_SPEC,
}


def get_node_spec(key: str) -> NodeSpec | None:
    return NODE_SPECS.get(key)


def list_node_specs() -> list[NodeSpec]:
    return [
        NODE_SPECS["source"],
        NODE_SPECS["load"],
        NODE_SPECS["alpha"],
        NODE_SPECS["sam2"],
        NODE_SPECS["sam3"],
        NODE_SPECS["matting"],
        NODE_SPECS["birefnet"],
        NODE_SPECS["gvm"],
        NODE_SPECS["chromakey"],
        NODE_SPECS["corridorkey"],
        NODE_SPECS["merge"],
        NODE_SPECS["export"],
    ]
