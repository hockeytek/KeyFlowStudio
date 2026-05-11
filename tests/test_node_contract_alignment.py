import unittest

from app.node_graph.rules.node_contracts import ALL_NODE_CONTRACTS, PortContract
from app.node_graph.specs import NODE_SPECS


class NodeContractAlignmentTests(unittest.TestCase):
    def test_port_contract_required_default_is_true(self):
        port = PortContract(name="x", data_type="image")
        self.assertTrue(port.required)

    def test_input_required_flags_match_specs(self):
        for node_type, spec in NODE_SPECS.items():
            contract = ALL_NODE_CONTRACTS.get(node_type)
            self.assertIsNotNone(contract, f"Missing contract for node type: {node_type}")

            spec_inputs = {p.name: p for p in spec.inputs}
            contract_inputs = {p.name: p for p in contract.inputs}

            self.assertEqual(
                set(spec_inputs.keys()),
                set(contract_inputs.keys()),
                f"Input port names mismatch for node type: {node_type}",
            )

            for port_name, spec_port in spec_inputs.items():
                contract_port = contract_inputs[port_name]
                self.assertEqual(
                    spec_port.data_type,
                    contract_port.data_type,
                    f"Data type mismatch for {node_type}.{port_name}",
                )
                self.assertEqual(
                    spec_port.required,
                    contract_port.required,
                    f"Required flag mismatch for {node_type}.{port_name}",
                )
                self.assertEqual(
                    spec_port.label,
                    contract_port.label,
                    f"Label mismatch for {node_type}.{port_name}",
                )

    def test_output_flags_and_types_match_specs(self):
        for node_type, spec in NODE_SPECS.items():
            contract = ALL_NODE_CONTRACTS.get(node_type)
            self.assertIsNotNone(contract, f"Missing contract for node type: {node_type}")

            spec_outputs = {p.name: p for p in spec.outputs}
            contract_outputs = {p.name: p for p in contract.outputs}

            self.assertEqual(
                set(spec_outputs.keys()),
                set(contract_outputs.keys()),
                f"Output port names mismatch for node type: {node_type}",
            )

            for port_name, spec_port in spec_outputs.items():
                contract_port = contract_outputs[port_name]
                self.assertEqual(
                    spec_port.data_type,
                    contract_port.data_type,
                    f"Data type mismatch for output {node_type}.{port_name}",
                )
                self.assertEqual(
                    spec_port.required,
                    contract_port.required,
                    f"Required flag mismatch for output {node_type}.{port_name}",
                )
                self.assertEqual(
                    spec_port.label,
                    contract_port.label,
                    f"Label mismatch for output {node_type}.{port_name}",
                )

    def test_titles_and_subtitles_match_specs(self):
        for node_type, spec in NODE_SPECS.items():
            contract = ALL_NODE_CONTRACTS.get(node_type)
            self.assertIsNotNone(contract, f"Missing contract for node type: {node_type}")
            self.assertEqual(spec.title, contract.title, f"Title mismatch for node type: {node_type}")
            self.assertEqual(spec.subtitle, contract.subtitle, f"Subtitle mismatch for node type: {node_type}")

    def test_default_properties_match_specs(self):
        for node_type, spec in NODE_SPECS.items():
            contract = ALL_NODE_CONTRACTS.get(node_type)
            self.assertIsNotNone(contract, f"Missing contract for node type: {node_type}")

            spec_defaults = spec.default_properties or {}
            contract_defaults = contract.default_properties or {}

            self.assertEqual(
                set(spec_defaults.keys()),
                set(contract_defaults.keys()),
                f"Default property keys mismatch for {node_type}",
            )

            for key, spec_value in spec_defaults.items():
                contract_value = contract_defaults[key]
                self.assertEqual(
                    spec_value,
                    contract_value,
                    (
                        f"Default property mismatch for {node_type}.{key}: "
                        f"spec={spec_value!r} ({type(spec_value).__name__}), "
                        f"contract={contract_value!r} ({type(contract_value).__name__})"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
