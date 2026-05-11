"""Node Rules Registry: Central API for node interaction rules and validation.

This is the main entry point for all node rule queries. Instead of scattered
checks across the codebase, all rule lookups go through this registry.
"""

from __future__ import annotations

from typing import Optional

from .node_contracts import (
    NodeContract,
    PortContract,
    get_contract,
    all_node_types,
)

# Module-level singleton
_registry_instance: Optional[NodeRulesRegistry] = None


class NodeRulesRegistry:
    """Central API for validating node graph connections and execution rules."""

    def __init__(self) -> None:
        """Initialize registry from contracts library."""
        pass  # Contracts are module-level constants; no build step needed.

    # ── Contract access ──────────────────────────────────────────────────────

    def get_contract(self, node_type: str) -> Optional[NodeContract]:
        """Get full contract for a node type."""
        return get_contract(node_type)

    def get_all_node_types(self) -> list[str]:
        """Get list of all available node types."""
        return all_node_types()

    def get_input_port(self, node_type: str, port_name: str) -> Optional[PortContract]:
        """Get input port contract."""
        contract = get_contract(node_type)
        if contract is None:
            return None
        return contract.get_input(port_name)

    def get_output_port(self, node_type: str, port_name: str) -> Optional[PortContract]:
        """Get output port contract."""
        contract = get_contract(node_type)
        if contract is None:
            return None
        return contract.get_output(port_name)

    # ── Connection validation ────────────────────────────────────────────────

    def can_connect_ports(
        self,
        src_node_type: str,
        src_port_name: str,
        dst_node_type: str,
        dst_port_name: str,
    ) -> bool:
        """Check if two ports can be connected.

        Rules:
        1. Port data types must match or be compatible
        2. Destination port type must accept source type
        3. Both node types must exist

        Returns True if compatible, False otherwise.
        Unknown node/port combinations are treated as incompatible in strict mode.
        """
        src_contract = get_contract(src_node_type)
        dst_contract = get_contract(dst_node_type)
        if not src_contract or not dst_contract:
            return False

        src_port = src_contract.get_output(src_port_name)
        dst_port = dst_contract.get_input(dst_port_name)
        if not src_port or not dst_port:
            return False

        # Write input is intentionally generic and accepts any graph payload.
        if dst_node_type == "export" and dst_port_name == "in":
            return True

        # CorridorKey alphahint is source-agnostic but accepts only mask/alpha payloads.
        if dst_node_type == "corridorkey" and dst_port_name == "alphahint":
            return src_port.data_type in {"mask", "alpha"}

        # MatAnyone2 mask input is source-agnostic but accepts only mask/alpha payloads.
        if dst_node_type == "matting" and dst_port_name == "mask":
            return src_port.data_type in {"mask", "alpha"}

        # Merge mask input is source-agnostic but accepts only mask/alpha payloads.
        if dst_node_type == "merge" and dst_port_name == "mask":
            return src_port.data_type in {"mask", "alpha"}

        # BiRefNet source material must be RGB image.
        if dst_node_type == "birefnet" and dst_port_name == "image":
            return src_port.data_type == "image"

        # GVM source material must be RGB image.
        if dst_node_type == "gvm" and dst_port_name == "image":
            return src_port.data_type == "image"

        return self._port_types_compatible(src_port.data_type, dst_port.data_type)

    def _port_types_compatible(self, src_type: str, dst_type: str) -> bool:
        """Check if source data type is compatible with destination.

        Compatibility rules:
        - 'alpha' is compatible with 'alpha'
        - 'image' is compatible with 'image'
        - 'mask' is compatible with 'alpha' (both single-channel)
        - Any other combinations are incompatible (strict typing)
        """
        if src_type == dst_type:
            return True
        # mask and alpha are both single-channel grayscale — treat as compatible
        if {src_type, dst_type} <= {"mask", "alpha"}:
            return True
        return False

    # ── Topology rules ───────────────────────────────────────────────────────

    def can_upstream(self, src_node_type: str, dst_node_type: str) -> bool:
        """Check if src_node_type can feed into dst_node_type as upstream."""
        dst_contract = get_contract(dst_node_type)
        if dst_contract is None:
            return False
        allowed = dst_contract.upstream_allowed
        if not allowed:
            return True  # empty list means no restriction
        return src_node_type in allowed

    def can_downstream(self, src_node_type: str, dst_node_type: str) -> bool:
        """Check if src_node_type can feed into dst_node_type as downstream."""
        src_contract = get_contract(src_node_type)
        if src_contract is None:
            return False
        allowed = src_contract.downstream_allowed
        if not allowed:
            return True
        return dst_node_type in allowed

    def can_connect_topology(
        self,
        src_node_type: str,
        src_port_name: str,
        dst_node_type: str,
        dst_port_name: str,
    ) -> bool:
        """Check topology allowance for a concrete edge (node+port aware).

        Most rules are node-level (`can_downstream` + `can_upstream`).
        CorridorKey `alphahint` is an exception: it is intentionally source-agnostic
        and may receive mask/alpha from any node; payload typing is validated by
        `can_connect_ports`.

        MatAnyone2 `mask` is also source-agnostic: any node with `alpha` or `mask`
        output may feed this input, while all other ports remain governed by the
        regular node-level topology allowlists.

        Merge `mask` follows the same pattern: topology is source-agnostic for the
        mask input only, with payload type restrictions enforced by
        `can_connect_ports`.
        """
        if dst_node_type == "corridorkey" and dst_port_name == "alphahint":
            return True
        if dst_node_type == "matting" and dst_port_name == "mask":
            return True
        if dst_node_type == "merge" and dst_port_name == "mask":
            return True
        return (
            self.can_downstream(src_node_type, dst_node_type)
            and self.can_upstream(src_node_type, dst_node_type)
        )

    # ── Execution rules ──────────────────────────────────────────────────────

    def execution_rules(self, node_type: str) -> dict:
        """Get execution rules for a node type."""
        contract = get_contract(node_type)
        if contract is None:
            return {}
        return contract.execution_rules or {}

    def should_auto_propagate_sam_before_run(self) -> bool:
        """Check if SAM2 node should auto-propagate before Run.
        (Rule from SAM_NODE_RULES.md)
        """
        sam_contract = get_contract("sam2")
        if sam_contract is None:
            return False
        rules = sam_contract.execution_rules or {}
        return bool(rules.get("auto_propagate_before_run", False))

    def can_defer_birefnet_to_staged(self) -> bool:
        """Check if BiRefNet can be deferred to staged mode."""
        birefnet_contract = get_contract("birefnet")
        if birefnet_contract is None:
            return False
        rules = birefnet_contract.execution_rules or {}
        return bool(rules.get("can_defer_to_downstream", False))

    def can_defer_sam_disk_to_corridorkey(self) -> bool:
        """Check if SAM2 masks can be streamed per-frame from disk into CorridorKey.

        When True, a SAM2 node whose only outgoing connection is corridorkey.alphahint
        becomes disk-deferred: masks are never fully loaded into RAM; CorridorKey reads
        one mask per frame directly from the payload paths.
        """
        sam_contract = get_contract("sam2")
        if sam_contract is None:
            return False
        rules = sam_contract.execution_rules or {}
        return bool(rules.get("can_defer_disk_masks", False))

    def birefnet_binarization_threshold(self) -> int:
        """Get binarization threshold for BiRefNet alpha masks.
        (Rule: BiRefNet soft probabilities must be binarized before morphology)
        """
        birefnet_contract = get_contract("birefnet")
        if birefnet_contract is None:
            return 10
        rules = birefnet_contract.execution_rules or {}
        return int(rules.get("binarization_threshold", 10))

    def requires_matching_frame_count(self, node_type: str) -> bool:
        """Check if node requires matching input frame counts.
        (Rule: CorridorKey.image and CorridorKey.alphahint must match)
        """
        contract = get_contract(node_type)
        if contract is None:
            return False
        rules = contract.execution_rules or {}
        return bool(rules.get("requires_matching_frame_count", False))

    def frame_count_mismatch_error_key(self, node_type: str) -> str:
        """Get i18n error key for frame count mismatch."""
        contract = get_contract(node_type)
        if contract is None:
            return "err_frame_mismatch"
        rules = contract.execution_rules or {}
        return str(rules.get("frame_count_mismatch_error", "err_frame_mismatch"))

    def birefnet_output_consumed_only_by_corridorkey_hint(
        self,
        birefnet_outputs: dict,
        edges: list,
        nodes_by_id: dict,
    ) -> bool:
        """Check if BiRefNet output goes only to CorridorKey.alphahint.

        If True, BiRefNet can be deferred to staged mode.
        (Implementation of BIREFNET_NODE_RULES.md: "Staged Workflow")
        """
        for node_id, node_data in nodes_by_id.items():
            if node_data.get("type") != "birefnet":
                continue
            outgoing_edges = [e for e in edges if e.get("src_id") == node_id]
            for edge in outgoing_edges:
                dst_node = nodes_by_id.get(edge.get("dst_id"))
                if dst_node is None:
                    return False
                dst_type = dst_node.get("type")
                dst_port = edge.get("dst_port")
                src_port = edge.get("src_port")
                if src_port != "alpha":
                    return False
                if dst_type != "corridorkey" or dst_port != "alphahint":
                    return False
        return True

    def get_summary(self) -> dict:
        """Get summary of all nodes and their port counts."""
        summary = {}
        for node_type in self.get_all_node_types():
            contract = get_contract(node_type)
            if contract:
                rules = contract.execution_rules or {}
                summary[node_type] = {
                    "inputs": len(contract.inputs),
                    "outputs": len(contract.outputs),
                    "can_defer": rules.get("can_defer_to_downstream", False),
                }
        return summary


def get_registry() -> NodeRulesRegistry:
    """Get or create the singleton registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = NodeRulesRegistry()
    return _registry_instance
