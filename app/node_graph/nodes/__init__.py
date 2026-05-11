"""Node handlers registry."""

from __future__ import annotations

from app.node_graph.nodes.base import NodeHandler
from app.node_graph.nodes.alpha_node import AlphaNodeHandler
from app.node_graph.nodes.birefnet_node import BiRefNetNodeHandler
from app.node_graph.nodes.chromakey_node import ChromaKeyNodeHandler
from app.node_graph.nodes.corridorkey_node import CorridorKeyNodeHandler
from app.node_graph.nodes.export_node import ExportNodeHandler
from app.node_graph.nodes.load_media_node import LoadMediaNodeHandler
from app.node_graph.nodes.merge_node import MergeNodeHandler
from app.node_graph.nodes.sam_mask_node import SamMaskNodeHandler
from app.node_graph.nodes.sam3_mask_node import Sam3MaskNodeHandler
from app.node_graph.nodes.source_node import SourceNodeHandler

NODE_HANDLERS: dict[str, NodeHandler] = {
    "source": SourceNodeHandler(),
    "load": LoadMediaNodeHandler(),
    "alpha": AlphaNodeHandler(),
    "sam2": SamMaskNodeHandler(),
    "sam3": Sam3MaskNodeHandler(),
    "birefnet": BiRefNetNodeHandler(),
    "chromakey": ChromaKeyNodeHandler(),
    "corridorkey": CorridorKeyNodeHandler(),
    "merge": MergeNodeHandler(),
    "export": ExportNodeHandler(),
}
