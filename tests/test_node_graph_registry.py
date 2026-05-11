import unittest

from app.node_graph.rules.registry import NodeRulesRegistry


class NodeRulesRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = NodeRulesRegistry()

    def test_export_input_accepts_any_payload_type(self):
        self.assertTrue(self.registry.can_connect_ports("sam2", "out", "export", "in"))
        self.assertTrue(self.registry.can_connect_ports("alpha", "out", "export", "in"))
        self.assertTrue(self.registry.can_connect_ports("chromakey", "mask", "export", "in"))

    def test_corridorkey_alphahint_accepts_broad_payload_types(self):
        self.assertFalse(self.registry.can_connect_ports("load", "out", "corridorkey", "alphahint"))
        self.assertTrue(self.registry.can_connect_ports("alpha", "out", "corridorkey", "alphahint"))
        self.assertTrue(self.registry.can_connect_ports("chromakey", "mask", "corridorkey", "alphahint"))
        self.assertTrue(self.registry.can_connect_ports("sam2", "out", "corridorkey", "alphahint"))


if __name__ == "__main__":
    unittest.main()
