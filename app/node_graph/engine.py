"""Execution engine for node graphs."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

from app.node_graph.models import GraphEdge, GraphNode
from app.node_graph.specs import NODE_SPECS
from app.node_graph.rules import get_registry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphDiagnostic:
    """Structured validation diagnostic with machine-readable context."""

    code: str
    message: str
    node_id: str = ""
    src_node_id: str = ""
    dst_node_id: str = ""
    src_port: str = ""
    dst_port: str = ""
    rule: str = ""

    def to_text(self) -> str:
        parts = [f"[{self.code}] {self.message}"]
        if self.node_id:
            parts.append(f"node={self.node_id}")
        if self.src_node_id:
            parts.append(f"src={self.src_node_id}")
        if self.src_port:
            parts.append(f"src_port={self.src_port}")
        if self.dst_node_id:
            parts.append(f"dst={self.dst_node_id}")
        if self.dst_port:
            parts.append(f"dst_port={self.dst_port}")
        if self.rule:
            parts.append(f"rule={self.rule}")
        return " | ".join(parts)


@dataclass(frozen=True)
class ExecutionPlan:
    """Precomputed execution plan between validate and execute phases."""

    execution_order: list[str]
    connected_node_ids: set[str] = field(default_factory=set)
    deferred_node_ids: set[str] = field(default_factory=set)
    deferred_corridorkey_sources: dict[str, str] = field(default_factory=dict)
    node_actions: dict[str, str] = field(default_factory=dict)


class NodeGraphEngine:
    """Validate graph contracts and provide shared ordering utilities.
    
    This engine handles:
    - Topological sorting of nodes
    - Port validation based on specifications
    - Cycle detection in active execution path
    """

    def validate(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> tuple[bool, list[str]]:
        """Validate graph structure and port connections.
        
        Args:
            nodes: List of nodes in the graph
            edges: List of edges connecting nodes
        
        Returns:
            (is_valid, error_messages)
        """
        is_valid, diagnostics = self.validate_with_diagnostics(nodes, edges)
        return is_valid, [diag.to_text() for diag in diagnostics]

    def validate_with_diagnostics(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        *,
        strict_isolated_required_inputs: bool = False,
    ) -> tuple[bool, list[GraphDiagnostic]]:
        """Validate graph and return structured diagnostics."""
        errors: list[GraphDiagnostic] = []
        node_by_id = {node.id: node for node in nodes}
        registry = get_registry()
        
        # Check all nodes have specs. Execution handlers are validated by the
        # active runtime path (InferenceWorker dispatcher).
        for node in nodes:
            if node.type not in NODE_SPECS:
                errors.append(
                    GraphDiagnostic(
                        code="NG001",
                        message=f"Node has no spec for type '{node.type}'",
                        node_id=node.id,
                        rule="node_spec_exists",
                    )
                )
        
        # Validate edges and ports
        input_edge_count: dict[tuple[str, str], int] = defaultdict(int)
        for edge in edges:
            # Check nodes exist
            if edge.src_id not in node_by_id:
                errors.append(
                    GraphDiagnostic(
                        code="NG002",
                        message="Edge source node not found",
                        src_node_id=edge.src_id,
                        dst_node_id=edge.dst_id,
                        src_port=edge.src_port,
                        dst_port=edge.dst_port,
                        rule="edge_source_exists",
                    )
                )
                continue
            if edge.dst_id not in node_by_id:
                errors.append(
                    GraphDiagnostic(
                        code="NG003",
                        message="Edge destination node not found",
                        src_node_id=edge.src_id,
                        dst_node_id=edge.dst_id,
                        src_port=edge.src_port,
                        dst_port=edge.dst_port,
                        rule="edge_destination_exists",
                    )
                )
                continue
            
            src_node = node_by_id[edge.src_id]
            dst_node = node_by_id[edge.dst_id]

            # Enforce declared topology constraints from the central rules registry.
            # Some edges are port-specific exceptions (e.g. corridorkey.alphahint).
            if not registry.can_connect_topology(
                src_node.type,
                edge.src_port,
                dst_node.type,
                edge.dst_port,
            ):
                if not registry.can_downstream(src_node.type, dst_node.type):
                    errors.append(
                        GraphDiagnostic(
                            code="NG004",
                            message=f"Topology not allowed '{src_node.type}' -> '{dst_node.type}'",
                            src_node_id=edge.src_id,
                            dst_node_id=edge.dst_id,
                            src_port=edge.src_port,
                            dst_port=edge.dst_port,
                            rule="downstream_allowed",
                        )
                    )
                if not registry.can_upstream(src_node.type, dst_node.type):
                    errors.append(
                        GraphDiagnostic(
                            code="NG005",
                            message=f"Topology not allowed '{src_node.type}' -> '{dst_node.type}'",
                            src_node_id=edge.src_id,
                            dst_node_id=edge.dst_id,
                            src_port=edge.src_port,
                            dst_port=edge.dst_port,
                            rule="upstream_allowed",
                        )
                    )
            
            # Get specs
            src_spec = NODE_SPECS.get(src_node.type)
            dst_spec = NODE_SPECS.get(dst_node.type)
            
            if not src_spec or not dst_spec:
                continue  # Already reported above
            
            # Check output port exists in source
            output_ports = {p.name: p for p in src_spec.outputs}
            if edge.src_port not in output_ports:
                errors.append(
                    GraphDiagnostic(
                        code="NG006",
                        message=(
                            f"Source node '{edge.src_id}' ({src_node.type}) has no output port '{edge.src_port}'. "
                            f"Available: {list(output_ports.keys())}"
                        ),
                        src_node_id=edge.src_id,
                        dst_node_id=edge.dst_id,
                        src_port=edge.src_port,
                        dst_port=edge.dst_port,
                        rule="source_port_exists",
                    )
                )
            
            # Check input port exists in destination
            input_ports = {p.name: p for p in dst_spec.inputs}
            if edge.dst_port not in input_ports:
                errors.append(
                    GraphDiagnostic(
                        code="NG007",
                        message=(
                            f"Destination node '{edge.dst_id}' ({dst_node.type}) has no input port '{edge.dst_port}'. "
                            f"Available: {list(input_ports.keys())}"
                        ),
                        src_node_id=edge.src_id,
                        dst_node_id=edge.dst_id,
                        src_port=edge.src_port,
                        dst_port=edge.dst_port,
                        rule="destination_port_exists",
                    )
                )
            else:
                input_edge_count[(edge.dst_id, edge.dst_port)] += 1
            
            # Check port data types match using centralized rules registry.
            # can_connect_ports returns True/False when both ports are known in
            # the contract, or None when the port is not declared in the contract
            # (the spec already verified the port exists, so None == "no opinion").
            if edge.src_port in output_ports and edge.dst_port in input_ports:
                src_type = output_ports[edge.src_port].data_type
                dst_type = input_ports[edge.dst_port].data_type
                compat = registry.can_connect_ports(
                    src_node.type,
                    edge.src_port,
                    dst_node.type,
                    edge.dst_port,
                )
                if not compat:
                    errors.append(
                        GraphDiagnostic(
                            code="NG008",
                            message=(
                                f"Port type mismatch between '{edge.src_id}.{edge.src_port}' ({src_type}) "
                                f"and '{edge.dst_id}.{edge.dst_port}' ({dst_type})"
                            ),
                            src_node_id=edge.src_id,
                            dst_node_id=edge.dst_id,
                            src_port=edge.src_port,
                            dst_port=edge.dst_port,
                            rule="port_types_compatible",
                        )
                    )

        # One connection per destination input port.
        for (dst_id, dst_port), count in input_edge_count.items():
            if count > 1:
                errors.append(
                    GraphDiagnostic(
                        code="NG009",
                        message=f"Node '{dst_id}' has {count} incoming edges to input port '{dst_port}' (max 1)",
                        node_id=dst_id,
                        dst_port=dst_port,
                        rule="single_connection_per_input_port",
                    )
                )

        # Validate required input ports are connected.
        # In soft mode, detached/floating nodes are skipped except SAM2/SAM3.
        # In strict mode, detached nodes are validated too.
        connected_node_ids = {edge.src_id for edge in edges} | {edge.dst_id for edge in edges}
        always_validate_required_inputs_for = {"sam2", "sam3"}

        for node in nodes:
            if not node.enabled:
                continue
            if (
                node.id not in connected_node_ids
                and not strict_isolated_required_inputs
                and node.type not in always_validate_required_inputs_for
            ):
                continue  # Isolated node — not part of active flow, skip

            spec = NODE_SPECS.get(node.type)
            if spec is None:
                continue

            connected_input_ports = {
                edge.dst_port
                for edge in edges
                if edge.dst_id == node.id and edge.dst_port
            }

            for input_port in spec.inputs:
                if input_port.required and input_port.name not in connected_input_ports:
                    errors.append(
                        GraphDiagnostic(
                            code="NG010",
                            message=f"Node '{node.id}' ({node.type}) missing required input port '{input_port.name}'",
                            node_id=node.id,
                            dst_port=input_port.name,
                            rule="required_input_connected",
                        )
                    )

        # Detect cycles in active (enabled) execution path.
        try:
            self.topological_order(nodes, edges, enabled_only=True)
        except ValueError as exc:
            errors.append(
                GraphDiagnostic(
                    code="NG011",
                    message=f"Graph structure error: {exc}",
                    rule="acyclic_graph",
                )
            )
        
        return len(errors) == 0, errors

    def build_execution_plan(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> ExecutionPlan:
        """Build validated execution plan used by runtime worker."""
        plan, diagnostics = self.build_execution_plan_with_diagnostics(nodes, edges)
        if plan is None:
            joined = "\n".join(diag.to_text() for diag in diagnostics)
            raise ValueError(f"Graph validation failed:\n{joined}")
        return plan

    def build_execution_plan_with_diagnostics(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        *,
        strict_isolated_required_inputs: bool = False,
    ) -> tuple[ExecutionPlan | None, list[GraphDiagnostic]]:
        """Build execution plan and always return diagnostics for callers/UI logs."""
        is_valid, diagnostics = self.validate_with_diagnostics(
            nodes,
            edges,
            strict_isolated_required_inputs=strict_isolated_required_inputs,
        )
        if not is_valid:
            return None, diagnostics

        node_by_id = {node.id: node for node in nodes}
        execution_order = self.topological_order(nodes, edges)
        connected_node_ids = {edge.src_id for edge in edges} | {edge.dst_id for edge in edges}

        deferred_node_ids: set[str] = set()
        deferred_corridorkey_sources: dict[str, str] = {}
        registry = get_registry()
        if registry.can_defer_birefnet_to_staged():
            for node in nodes:
                if not node.enabled or node.type != "birefnet":
                    continue
                outgoing = [edge for edge in edges if edge.src_id == node.id]
                if not outgoing:
                    continue

                deferred = True
                for edge in outgoing:
                    dst = node_by_id.get(edge.dst_id)
                    if dst is None or not dst.enabled:
                        deferred = False
                        break
                    if dst.type != "corridorkey":
                        deferred = False
                        break
                    if edge.src_port != "alpha" or edge.dst_port != "alphahint":
                        deferred = False
                        break

                if not deferred:
                    continue

                deferred_node_ids.add(node.id)
                for edge in outgoing:
                    deferred_corridorkey_sources[edge.dst_id] = node.id

        if registry.can_defer_sam_disk_to_corridorkey():
            for node in nodes:
                if not node.enabled or node.type != "sam2":
                    continue
                outgoing = [edge for edge in edges if edge.src_id == node.id]
                if not outgoing:
                    continue
                # SAM defers only when ALL its outputs go to corridorkey.alphahint
                deferred = True
                for edge in outgoing:
                    dst = node_by_id.get(edge.dst_id)
                    if dst is None or not dst.enabled:
                        deferred = False
                        break
                    if dst.type != "corridorkey":
                        deferred = False
                        break
                    if edge.dst_port != "alphahint":
                        deferred = False
                        break
                if not deferred:
                    continue
                deferred_node_ids.add(node.id)
                for edge in outgoing:
                    deferred_corridorkey_sources[edge.dst_id] = node.id

        node_actions: dict[str, str] = {}
        for node_id in execution_order:
            node = node_by_id.get(node_id)
            if node is None:
                continue
            if not node.enabled:
                node_actions[node_id] = "skip_disabled"
                continue
            if node_id not in connected_node_ids:
                node_actions[node_id] = "skip_isolated"
                continue
            if node_id in deferred_node_ids:
                node_actions[node_id] = "deferred"
                continue
            if node.type in {"source", "load", "alpha"}:
                node_actions[node_id] = "passthrough_source"
                continue
            if node.type == "export":
                node_actions[node_id] = "write_sink"
                continue
            node_actions[node_id] = "execute"

        plan = ExecutionPlan(
            execution_order=execution_order,
            connected_node_ids=connected_node_ids,
            deferred_node_ids=deferred_node_ids,
            deferred_corridorkey_sources=deferred_corridorkey_sources,
            node_actions=node_actions,
        )
        return plan, diagnostics

    def topological_order(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        *,
        enabled_only: bool = False,
    ) -> list[str]:
        """Return node ids in topological order for graph execution.
        
        Args:
            nodes: List of graph nodes
            edges: Connections between nodes
            enabled_only: Ignore disabled nodes and their edges if True
        
        Returns:
            Node ids in execution order
        
        Raises:
            ValueError: When graph contains cycles
        """
        active_nodes = [node for node in nodes if node.enabled] if enabled_only else list(nodes)
        node_by_id = {node.id: node for node in active_nodes}
        if not node_by_id:
            return []

        valid_edges = [
            edge
            for edge in edges
            if edge.src_id in node_by_id and edge.dst_id in node_by_id
        ]

        # Build dependency graph
        incoming: dict[str, list[str]] = defaultdict(list)
        outgoing: dict[str, list[str]] = defaultdict(list)
        indegree: dict[str, int] = {node_id: 0 for node_id in node_by_id}

        for edge in valid_edges:
            outgoing[edge.src_id].append(edge.dst_id)
            incoming[edge.dst_id].append(edge.src_id)
            indegree[edge.dst_id] += 1

        # Topological sort (Kahn's algorithm)
        queue = deque([node_id for node_id, deg in indegree.items() if deg == 0])
        topo_order: list[str] = []

        while queue:
            node_id = queue.popleft()
            topo_order.append(node_id)
            for dst_id in outgoing.get(node_id, []):
                indegree[dst_id] -= 1
                if indegree[dst_id] == 0:
                    queue.append(dst_id)

        if len(topo_order) != len(node_by_id):
            blocked = sorted(node_id for node_id, deg in indegree.items() if deg > 0)
            raise ValueError(
                "Graph has cycles or invalid dependency structure"
                f". Blocked nodes: {blocked}"
            )

        logger.debug(f"Topological order resolved: {topo_order}")
        return topo_order

    def can_connect(
        self,
        src_node_type: str,
        src_port: str,
        dst_node_type: str,
        dst_port: str,
    ) -> tuple[bool, str]:
        """Compatibility API for UI/tests: validate a single candidate edge."""
        src_spec = NODE_SPECS.get(src_node_type)
        dst_spec = NODE_SPECS.get(dst_node_type)

        if not src_spec:
            return False, f"Unknown node type: {src_node_type}"
        if not dst_spec:
            return False, f"Unknown node type: {dst_node_type}"

        output_ports = {p.name: p for p in src_spec.outputs}
        input_ports = {p.name: p for p in dst_spec.inputs}
        if src_port not in output_ports:
            return False, f"Source has no output port '{src_port}'"
        if dst_port not in input_ports:
            return False, f"Destination has no input port '{dst_port}'"

        registry = get_registry()
        if not registry.can_connect_topology(src_node_type, src_port, dst_node_type, dst_port):
            return False, f"Topology not allowed: {src_node_type} -> {dst_node_type}"
        if not registry.can_connect_ports(src_node_type, src_port, dst_node_type, dst_port):
            return (
                False,
                (
                    "Port type mismatch: "
                    f"{output_ports[src_port].data_type} -> {input_ports[dst_port].data_type}"
                ),
            )
        return True, ""

