## Node Graph Standardization Roadmap

Primary standard document:
- See [../NODE_GRAPH_STANDARD.md](../NODE_GRAPH_STANDARD.md)

This file is now a compact roadmap and status tracker.
The former long-form plan has been replaced by the standard document above.

### Goal
Create a single, maintainable node standard so new nodes can be added without scattered logic or semantic drift.

### Completed Phases
1. Phase 1: Inventory and baseline completed.
2. Phase 2: Canonical standard completed at document level.
3. Phase 3: Safe implementation alignment completed.
4. Phase 4: Scenario-to-test mapping completed.
5. Phase 5: Adoption playbook completed.

### What Was Standardized
1. Node inventory, topology, compatibility, and runtime scenarios.
2. Node cards for all current nodes.
3. Type compatibility matrix and topology matrix.
4. SAM propagation policy.
5. New-node checklist and legacy naming policy.

### What Was Aligned in Code
1. `required` default unified.
2. `inputs` deduplicated from `NodeSpec` into `NodeContract`.
3. `outputs` deduplicated from `NodeSpec` into `NodeContract`.
4. `title/subtitle` deduplicated from `NodeSpec` into `NodeContract`.
5. `export.default_properties` aligned with `NodeSpec` and shared constants.

### Test Coverage Achieved
1. Contract/spec alignment tests.
2. Topology validation tests.
3. Compatibility exception tests.
4. Deferred BiRefNet planning tests.
5. Worker failure-path tests for CorridorKey and Matting.
6. Boundary-level SAM propagation test.

### Remaining Intentional Non-Deduplicated Areas
1. `execution_rules` stay contract-native.
2. `upstream_allowed` and `downstream_allowed` stay contract-native.

### Next Practical Uses
1. Add a new node by following [../NODE_GRAPH_STANDARD.md](../NODE_GRAPH_STANDARD.md).
2. Use this roadmap only for tracking future follow-up work.
3. If new drift appears, update tests first and standard second.

### Done Criteria Status
1. Standard document exists: done.
2. Roadmap exists: done.
3. Tests cover standardized behavior: done.
4. Extension playbook exists: done.
