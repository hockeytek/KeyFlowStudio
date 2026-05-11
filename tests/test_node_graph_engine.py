import unittest

from app.node_graph.engine import NodeGraphEngine
from app.node_graph.models import GraphEdge, GraphNode


class NodeGraphEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = NodeGraphEngine()

    @staticmethod
    def _node(node_id: str, node_type: str = "source", enabled: bool = True) -> GraphNode:
        return GraphNode(
            id=node_id,
            type=node_type,
            title=node_id,
            properties={},
            enabled=enabled,
        )

    def test_topological_order_respects_dependencies(self):
        nodes = [
            self._node("load_1", "load"),
            self._node("matting_1", "matting"),
            self._node("export_1", "export"),
        ]
        edges = [
            GraphEdge(src_id="load_1", dst_id="matting_1", src_port="out", dst_port="image"),
            GraphEdge(src_id="matting_1", dst_id="export_1", src_port="fg", dst_port="in"),
        ]

        order = self.engine.topological_order(nodes, edges)

        self.assertLess(order.index("load_1"), order.index("matting_1"))
        self.assertLess(order.index("matting_1"), order.index("export_1"))

    def test_topological_order_raises_on_cycle(self):
        nodes = [
            self._node("a", "load"),
            self._node("b", "export"),
        ]
        edges = [
            GraphEdge(src_id="a", dst_id="b", src_port="out", dst_port="in"),
            GraphEdge(src_id="b", dst_id="a", src_port="in", dst_port="image"),
        ]

        with self.assertRaises(ValueError):
            self.engine.topological_order(nodes, edges)

    def test_validate_reports_cycle_error(self):
        nodes = [
            self._node("a", "load"),
            self._node("b", "export"),
        ]
        edges = [
            GraphEdge(src_id="a", dst_id="b", src_port="out", dst_port="in"),
            GraphEdge(src_id="b", dst_id="a", src_port="in", dst_port="image"),
        ]

        is_valid, errors = self.engine.validate(nodes, edges)

        self.assertFalse(is_valid)
        self.assertTrue(any("Graph structure error" in error for error in errors))

    def test_validate_reports_topology_violation(self):
        nodes = [
            self._node("load_1", "load"),
            self._node("sam2_1", "sam2"),
            self._node("birefnet_1", "birefnet"),
        ]
        edges = [
            GraphEdge(src_id="load_1", dst_id="sam2_1", src_port="out", dst_port="img"),
            GraphEdge(src_id="sam2_1", dst_id="birefnet_1", src_port="out", dst_port="image"),
        ]

        is_valid, errors = self.engine.validate(nodes, edges)

        self.assertFalse(is_valid)
        self.assertTrue(any("Topology not allowed" in error for error in errors))

    def test_validate_reports_multiple_edges_for_same_input(self):
        nodes = [
            self._node("load_1", "load"),
            self._node("source_1", "source"),
            self._node("matting_1", "matting"),
        ]
        edges = [
            GraphEdge(src_id="load_1", dst_id="matting_1", src_port="out", dst_port="img"),
            GraphEdge(src_id="source_1", dst_id="matting_1", src_port="out", dst_port="img"),
        ]

        is_valid, errors = self.engine.validate(nodes, edges)

        self.assertFalse(is_valid)
        self.assertTrue(any("incoming edges to input port 'img'" in error for error in errors))

    def test_can_connect_allows_generic_write_input(self):
        can_connect, reason = self.engine.can_connect("sam2", "out", "export", "in")
        self.assertTrue(can_connect)
        self.assertEqual(reason, "")

    def test_can_connect_rejects_corridorkey_alphahint_from_image(self):
        can_connect, reason = self.engine.can_connect("load", "out", "corridorkey", "alphahint")
        self.assertFalse(can_connect)
        self.assertIn("Port type mismatch", reason)

    def test_can_connect_allows_corridorkey_alphahint_from_alpha_mask_any_node(self):
        can_connect_alpha, reason_alpha = self.engine.can_connect("sam2", "out", "corridorkey", "alphahint")
        can_connect_mask, reason_mask = self.engine.can_connect("chromakey", "mask", "corridorkey", "alphahint")
        self.assertTrue(can_connect_alpha)
        self.assertEqual(reason_alpha, "")
        self.assertTrue(can_connect_mask)
        self.assertEqual(reason_mask, "")

    def test_can_connect_allows_matting_mask_from_generic_alpha_source(self):
        can_connect, reason = self.engine.can_connect("alpha", "out", "matting", "mask")
        self.assertTrue(can_connect)
        self.assertEqual(reason, "")

    def test_can_connect_rejects_unknown_node_type(self):
        can_connect, reason = self.engine.can_connect("unknown", "out", "export", "in")
        self.assertFalse(can_connect)
        self.assertTrue(reason.startswith("Unknown node type:"))

    def test_validate_with_diagnostics_includes_code_and_context(self):
        nodes = [
            self._node("load_1", "load"),
            self._node("sam2_1", "sam2"),
            self._node("birefnet_1", "birefnet"),
        ]
        edges = [
            GraphEdge(src_id="load_1", dst_id="sam2_1", src_port="out", dst_port="img"),
            GraphEdge(src_id="sam2_1", dst_id="birefnet_1", src_port="out", dst_port="image"),
        ]

        is_valid, diagnostics = self.engine.validate_with_diagnostics(nodes, edges)

        self.assertFalse(is_valid)
        self.assertTrue(diagnostics)
        self.assertTrue(any(d.code in {"NG004", "NG005"} for d in diagnostics))
        self.assertTrue(any(d.src_node_id == "sam2_1" and d.dst_node_id == "birefnet_1" for d in diagnostics))

    def test_build_execution_plan_marks_deferred_birefnet(self):
        nodes = [
            self._node("load_1", "load"),
            self._node("birefnet_1", "birefnet"),
            self._node("corridorkey_1", "corridorkey"),
            self._node("export_1", "export"),
        ]
        edges = [
            GraphEdge(src_id="load_1", dst_id="birefnet_1", src_port="out", dst_port="image"),
            GraphEdge(src_id="load_1", dst_id="corridorkey_1", src_port="out", dst_port="image"),
            GraphEdge(src_id="birefnet_1", dst_id="corridorkey_1", src_port="alpha", dst_port="alphahint"),
            GraphEdge(src_id="corridorkey_1", dst_id="export_1", src_port="alpha", dst_port="in"),
        ]

        plan = self.engine.build_execution_plan(nodes, edges)

        self.assertIn("birefnet_1", plan.execution_order)
        self.assertIn("corridorkey_1", plan.execution_order)
        self.assertIn("birefnet_1", plan.deferred_node_ids)
        self.assertEqual(plan.deferred_corridorkey_sources.get("corridorkey_1"), "birefnet_1")
        self.assertEqual(plan.node_actions.get("load_1"), "passthrough_source")
        self.assertEqual(plan.node_actions.get("birefnet_1"), "deferred")
        self.assertEqual(plan.node_actions.get("corridorkey_1"), "execute")
        self.assertEqual(plan.node_actions.get("export_1"), "write_sink")

    def test_build_execution_plan_marks_isolated_and_disabled_nodes(self):
        nodes = [
            self._node("load_1", "load", enabled=True),
            self._node("export_1", "export", enabled=True),
            self._node("floating_1", "matting", enabled=True),
            self._node("disabled_1", "sam2", enabled=False),
        ]
        edges = [
            GraphEdge(src_id="load_1", dst_id="export_1", src_port="out", dst_port="in"),
        ]

        plan = self.engine.build_execution_plan(nodes, edges)

        self.assertEqual(plan.node_actions.get("load_1"), "passthrough_source")
        self.assertEqual(plan.node_actions.get("export_1"), "write_sink")
        self.assertEqual(plan.node_actions.get("floating_1"), "skip_isolated")
        self.assertEqual(plan.node_actions.get("disabled_1"), "skip_disabled")

    def test_build_execution_plan_with_diagnostics_returns_none_on_invalid_graph(self):
        nodes = [
            self._node("load_1", "load"),
            self._node("sam2_1", "sam2"),
            self._node("birefnet_1", "birefnet"),
        ]
        edges = [
            GraphEdge(src_id="load_1", dst_id="sam2_1", src_port="out", dst_port="img"),
            GraphEdge(src_id="sam2_1", dst_id="birefnet_1", src_port="out", dst_port="image"),
        ]

        plan, diagnostics = self.engine.build_execution_plan_with_diagnostics(nodes, edges)

        self.assertIsNone(plan)
        self.assertTrue(diagnostics)
        self.assertTrue(any(d.code in {"NG004", "NG005"} for d in diagnostics))


if __name__ == "__main__":
    unittest.main()
