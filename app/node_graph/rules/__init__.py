"""Node graph rules module: Central management of all node interaction contracts.

This module contains:
- node_contracts.py: Definition of all node contracts (inputs, outputs, rules)
- registry.py: API for querying and validating rules
"""

from .node_contracts import (
    NodeContract,
    PortContract,
    get_contract,
    all_node_types,
    ALL_NODE_CONTRACTS,
)
from .registry import NodeRulesRegistry, get_registry

__all__ = (
    "NodeContract",
    "PortContract",
    "get_contract",
    "all_node_types",
    "ALL_NODE_CONTRACTS",
    "NodeRulesRegistry",
    "get_registry",
)
