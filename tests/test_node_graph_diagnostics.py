import unittest

from app.node_graph.diagnostics import (
    format_graph_diagnostic_context,
    format_graph_diagnostics_summary,
    format_graph_diagnostics_text,
)
from app.node_graph.engine import GraphDiagnostic


_TRANSLATIONS = {
    "graph_diagnostics_status_ok": "Graph: no issues",
    "graph_diagnostics_status_errors": "Graph: {count} issue(s) [{codes}]",
    "graph_diagnostics_group_line": "{code} • {label} ({count})",
    "graph_diagnostics_entry_line": "- {detail}",
    "graph_diag_context_node": "Node: {node_id}",
    "graph_diag_context_edge": "Edge: {src} -> {dst}",
    "graph_diag_context_edge_blocked": "Blocked edge: {src} -> {dst}",
    "graph_diag_context_port": "Port: {port}",
    "graph_diag_context_rule": "Rule: {rule}",
    "graph_diag_context_reason": "Reason: {reason}",
    "graph_diag_required_input_missing": "Missing required input: {port} ({port_name})",
    "graph_diag_code_NG004": "Disallowed downstream topology",
    "graph_diag_code_NG010": "Required input is not connected",
    "graph_diag_rule_downstream_allowed": "is not allowed by downstream rules",
    "graph_diag_rule_required_input_connected": "required input must be connected",
    "graph_diag_port_mask": "Mask",
    "graph_diag_missing_output_port_detail": "Missing output port: {port}. Available outputs: {available}",
    "graph_diag_missing_input_port_detail": "Missing input port: {port}. Available inputs: {available}",
    "graph_diag_type_mismatch_detail": "Type mismatch: {src} ({src_type}) -> {dst} (expected {expected}, got {actual})",
    "graph_diagnostics_ok": "Graph is valid",
}


def _tr(key: str) -> str:
    return _TRANSLATIONS.get(key, key)


class NodeGraphDiagnosticsFormattingTests(unittest.TestCase):
    def test_summary_groups_by_code(self):
        diagnostics = [
            GraphDiagnostic(code="NG004", message="", src_node_id="load_1", dst_node_id="sam_1"),
            GraphDiagnostic(code="NG004", message="", src_node_id="sam_1", dst_node_id="matting_1"),
            GraphDiagnostic(code="NG010", message="", node_id="matting_1", dst_port="mask"),
        ]

        summary = format_graph_diagnostics_summary(_tr, diagnostics)

        self.assertEqual(summary, "Graph: 3 issue(s) [NG004x2, NG010x1]")

    def test_context_uses_edge_and_rule_labels(self):
        diagnostic = GraphDiagnostic(
            code="NG004",
            message="",
            src_node_id="load_1",
            dst_node_id="sam_1",
            src_port="out",
            dst_port="img",
            rule="downstream_allowed",
        )

        context = format_graph_diagnostic_context(_tr, diagnostic)

        self.assertIn("Blocked edge: load_1.out -> sam_1.img", context)
        self.assertIn("Reason: is not allowed by downstream rules", context)

    def test_text_contains_group_heading_and_detail_lines(self):
        diagnostics = [
            GraphDiagnostic(
                code="NG010",
                message="required input missing",
                node_id="matting_1",
                dst_port="mask",
                rule="required_input_connected",
            )
        ]

        text = format_graph_diagnostics_text(_tr, diagnostics)

        self.assertIn("NG010 • Required input is not connected (1)", text)
        self.assertIn("Node: matting_1", text)
        self.assertIn("Missing required input: Mask (mask)", text)
        self.assertIn("Rule: required input must be connected", text)

    def test_context_supports_readable_node_labels(self):
        diagnostic = GraphDiagnostic(
            code="NG004",
            message="",
            src_node_id="n1",
            dst_node_id="n2",
            src_port="alpha",
            dst_port="alphahint",
            rule="downstream_allowed",
        )

        context = format_graph_diagnostic_context(
            _tr,
            diagnostic,
            node_label_for_id=lambda node_id: {"n1": "BiRefNet", "n2": "CorridorKey"}.get(node_id, node_id),
        )

        self.assertIn("Blocked edge: BiRefNet [n1].alpha -> CorridorKey [n2].alphahint", context)

    def test_context_shows_expected_actual_for_ng008(self):
        diagnostic = GraphDiagnostic(
            code="NG008",
            message="Port type mismatch between 'birefnet.alpha' (alpha) and 'matting.img' (image)",
            src_node_id="n1",
            dst_node_id="n2",
            src_port="alpha",
            dst_port="img",
            rule="port_types_compatible",
        )

        context = format_graph_diagnostic_context(_tr, diagnostic)
        self.assertIn("Type mismatch: birefnet.alpha (alpha) -> matting.img (expected image, got alpha)", context)

    def test_context_shows_available_ports_for_ng006(self):
        diagnostic = GraphDiagnostic(
            code="NG006",
            message="Source node 'n1' (source) has no output port 'foo'. Available: ['out']",
            src_node_id="n1",
            dst_node_id="n2",
            src_port="foo",
            dst_port="img",
            rule="source_port_exists",
        )

        context = format_graph_diagnostic_context(_tr, diagnostic)
        self.assertIn("Missing output port: foo. Available outputs: out", context)

    def test_context_shows_available_ports_for_ng007(self):
        diagnostic = GraphDiagnostic(
            code="NG007",
            message="Destination node 'n2' (sam) has no input port 'foo'. Available: ['img']",
            src_node_id="n1",
            dst_node_id="n2",
            src_port="out",
            dst_port="foo",
            rule="destination_port_exists",
        )

        context = format_graph_diagnostic_context(_tr, diagnostic)
        self.assertIn("Missing input port: foo. Available inputs: img", context)


if __name__ == "__main__":
    unittest.main()