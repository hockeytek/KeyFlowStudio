# Node Graph Standard

Related documents:
- Roadmap and rollout status: [plans/plan-nodeGraphStandardization.prompt.md](plans/plan-nodeGraphStandardization.prompt.md)
- Write node preview and disk-restore contract: [plans/write-node-preview-contract.md](plans/write-node-preview-contract.md)

## 1. Purpose
This document is the primary standard for node definitions, connectivity, runtime expectations, and extension workflow in the project.

Use this file as the main source when:
- reviewing existing nodes
- adding new nodes
- validating compatibility rules
- checking runtime behavior expectations

## 2. Source of Truth

Primary source by concern:

1. `NodeSpec`
- ports
- labels
- required flags
- title/subtitle
- default properties

2. `NodeContract`
- topology rules
- execution rules
- semantic restrictions not expressible by plain types

3. `NodeRulesRegistry`
- shared compatibility logic
- explicit compatibility exceptions

4. `NodeGraphEngine`
- graph validation
- diagnostics
- execution planning

5. `InferenceWorker`
- runtime execution only
- must not redefine node contracts

## 3. Standard Node Schema

Each node must be described with the following sections:

1. Identity
- key
- title
- subtitle
- i18n keys

2. Ports
- inputs: name, type, required, label
- outputs: name, type, label

3. Topology
- upstream_allowed
- downstream_allowed

4. Compatibility
- default type compatibility
- explicit exceptions

5. Runtime Contract
- execution mode
- prerequisites
- output semantics

6. Diagnostics
- validation failures
- runtime failures

7. Recovery
- stop behavior
- partial output behavior

8. Reference Scenarios
- happy path
- negative path
- optional staged/deferred path

## 4. Type Compatibility Matrix

Default compatibility:

| Source Type | Destination Type | Compatible |
|---|---|---|
| image | image | yes |
| alpha | alpha | yes |
| mask | mask | yes |
| mask | alpha | yes |
| alpha | mask | yes |
| image | alpha | no |
| image | mask | no |
| alpha | image | no |
| mask | image | no |

Explicit registry exceptions:

1. `export.in` is a terminal sink input and accepts any upstream payload type.
2. `corridorkey.alphahint` is a flexible hint input that accepts only `mask`/`alpha` payloads from any source node (port-specific topology exception).
3. `birefnet.image` accepts only `image`.

## 5. Topology Matrix

| From \ To | source | load | alpha | sam | matting | birefnet | chromakey | corridorkey | export |
|---|---|---|---|---|---|---|---|---|---|
| source | no | yes | yes | yes | yes | yes | yes | yes | yes |
| load | no | no | yes | yes | yes | yes | yes | yes | yes |
| alpha | no | no | no | no | no | no | no | no | yes |
| sam | no | no | yes | no | no | no | yes | no | yes |
| matting | no | no | no | no | no | no | no | no | yes |
| birefnet | no | no | no | no | no | no | no | yes | yes |
| chromakey | no | no | no | no | yes | no | no | yes | yes |
| corridorkey | no | no | yes | no | yes | no | no | no | yes |
| export | no | no | no | no | no | no | no | no | no |

## 6. Node Cards

### 6.1 Source
Identity:
- key: `source`
- title: `Source`
- subtitle: `Primary Media Source`

Ports:
- inputs: none
- outputs: `out:image`

Topology:
- downstream_allowed: `load,sam,birefnet,chromakey,corridorkey,matting,alpha,export`

Runtime:
- passthrough provider only

Scenarios:
- happy: `source -> sam -> export`
- negative: invalid destination port

### 6.2 Load
Identity:
- key: `load`
- title: `Read`
- subtitle: `Source Media`

Ports:
- inputs: none
- outputs: `out:image`

Topology:
- downstream_allowed: `sam,birefnet,chromakey,corridorkey,matting,alpha,export`

Runtime:
- passthrough provider for loaded media

Scenarios:
- happy: `load -> birefnet -> export`
- negative: isolated disabled load node

### 6.3 Alpha
Identity:
- key: `alpha`
- title: `Alpha`
- subtitle: `External Alpha / Mask`

Ports:
- inputs: none
- outputs: `out:alpha`

Topology:
- downstream_allowed: `export`

Runtime:
- passthrough provider of external alpha or mask

Scenarios:
- happy: `alpha -> export`
- negative: forbidden downstream node

### 6.4 SAM
Identity:
- key: `sam`
- title: `SAM2 Mask`
- subtitle: `Mask creation from clicks`

Ports:
- inputs: `img:image` required
- outputs: `out:alpha`

Topology:
- downstream_allowed: `alpha,chromakey,export`

Runtime:
- builds per-frame mask sequence from payload masks and optional fallback mask file
- policy v1: nearest-frame propagation for missing frame mask
- does not fabricate masks when no payload or fallback exists

Diagnostics:
- missing `img` input -> runtime failure

Scenarios:
- happy: `source -> sam -> export`
- negative: SAM without upstream image
- recovery: fallback mask reused when payload masks are absent

### 6.5 BiRefNet
Identity:
- key: `birefnet`
- title: `BiRefNet`
- subtitle: `lightweight alpha hint generation`

Ports:
- inputs: `image:image` required
- outputs: `alpha:alpha`

Topology:
- downstream_allowed: `corridorkey,export`

Runtime:
- batch inference over frames
- may be deferred when consumed only by `corridorkey.alphahint`
- optional morphology post-processing

Diagnostics:
- missing `image` input -> runtime failure

Scenarios:
- happy: `source -> birefnet -> export`
- staged: `source -> birefnet(deferred) -> corridorkey`
- negative: non-image payload into `birefnet.image`

### 6.6 ChromaKey
Identity:
- key: `chromakey`
- title: `HSV Chroma Key`
- subtitle: `classical HSV keying`

Ports:
- inputs: `image:image` required
- outputs: `mask:mask`

Topology:
- downstream_allowed: `corridorkey,matting,export`

Runtime:
- OpenCV HSV-based mask generation

Diagnostics:
- missing `image` input -> runtime failure

Scenarios:
- happy: `source -> chromakey -> corridorkey -> export`
- happy: `source -> chromakey -> matting -> export`

### 6.7 CorridorKey
Identity:
- key: `corridorkey`
- title: `CorridorKey`
- subtitle: `neural green-screen keying`

Ports:
- inputs: `image:image` required, `alphahint:alpha` required by contract
- outputs: `alpha:alpha`, `fg:image`, `comp:image`, `processed:image`

Topology:
- upstream_allowed: `load,source,birefnet,chromakey`
- downstream_allowed: `alpha,matting,export`

Compatibility exception:
- `alphahint` accepts only `mask/alpha` payloads
- `alphahint` may receive those payloads from any node (port-specific topology exception)
- `image` payload into `alphahint` is invalid

Runtime:
- batch mode when hints are directly available
- staged mode when BiRefNet is deferred and hints are generated to temp dir

Diagnostics:
- missing `image` -> runtime failure
- missing `alphahint` and no deferred source -> runtime failure
- frame mismatch -> execution-rule failure

Scenarios:
- happy: `source -> chromakey -> corridorkey -> export`
- happy: `source -> sam -> corridorkey(alphahint) -> export`
- staged: `source -> birefnet(deferred) -> corridorkey -> export`
- negative: CorridorKey without alpha-hint source
- negative: `source.out(image) -> corridorkey.alphahint`

### 6.8 Matting
Identity:
- key: `matting`
- title: `MatAnyone2`
- subtitle: `matting inference`

Ports:
- inputs: `img:image` required, `mask:mask` required
- outputs: `fg:image`, `alpha:alpha`

Topology:
- upstream_allowed: `load,source,corridorkey`
- downstream_allowed: `export`

Runtime:
- runs inference service with warmup/erode/dilate options
- streams `alpha` and `fg`

Diagnostics:
- missing `img` -> runtime failure
- missing `mask` -> runtime failure

Scenarios:
- happy: `source -> corridorkey -> matting -> export`
- happy: `source -> chromakey -> matting -> export`
- negative: Matting without mask

### 6.9 Export
Identity:
- key: `export`
- title: `Write`
- subtitle: `Render to Disk`

Ports:
- inputs: `in:image` optional in contract, but terminal exception accepts any payload
- outputs: none

Topology:
- upstream_allowed: `load,source,sam,birefnet,chromakey,corridorkey,matting,alpha`
- downstream_allowed: none

Runtime:
- terminal write sink
- format/codec/compression controlled by properties

Scenarios:
- happy: any single upstream node -> export
- happy: split graph with multiple export nodes for different streams

## 7. SAM Propagation Policy

Frozen rule for standard v1:

1. exact frame mask is preferred
2. if missing, latest previous frame mask is used
3. if no previous exists, earliest next frame mask is used
4. if payload masks are absent and a fallback mask file exists, fallback is reused
5. if neither payload nor fallback exists, downstream required-input validation must fail explicitly

## 8. Practical Checklist: Add a New Node

1. Create spec
- file: `app/node_graph/specs/<node>.py`
- define ports, labels, required flags, title/subtitle, default properties

2. Register spec
- update `app/node_graph/specs/__init__.py`

3. Add contract
- update `app/node_graph/rules/node_contracts.py`
- define topology and execution rules

4. Update registry only if special compatibility is needed
- update `app/node_graph/rules/registry.py`
- do not add exception logic if default type compatibility is enough

5. Add runtime implementation only if node computes data
- primary runtime path: `app/workers/inference_worker.py`
- optional dedicated handler/controller under `app/node_graph/nodes/`

6. Add UI properties panel only if node has editable settings
- `app/node_graph/<node>_properties_panel.py`
- register in `app/node_graph_dialog.py`

7. Add i18n strings
- `app/i18n.py`

8. Add tests
- alignment coverage
- compatibility/topology tests if needed
- happy-path runtime test if computational
- failure-path test for required inputs

## 9. Reference Flow: Example Utility Node

Example target:
- `normalize_mask`
- input: `mask:mask`
- output: `mask:mask`

Recommended implementation flow:

1. add `NodeSpec`
2. register in `NODE_SPECS`
3. add `NodeContract`
4. avoid registry exception if standard compatibility already fits
5. add runtime branch or handler
6. add tests
7. add one node card entry here

Definition of complete:
- node appears in graph
- ports connect by documented rules
- runtime succeeds or fails with controlled diagnostics
- tests pass
- documentation updated

## 10. Legacy Naming and Deprecation Policy

Current rule:
- existing names remain valid
- no breaking rename for style-only cleanup

For new nodes:
- prefer descriptive names over ambiguous legacy abbreviations
- use `image`, `mask`, `alpha` when semantics are clear
- use `in` or `out` only for truly generic ports

Deprecation policy:
1. do not rename existing ports just for style consistency
2. if a legacy name becomes a problem, add compatibility first
3. document the legacy name before any removal
4. remove only with migration note, tests, and preset compatibility review

## 11. Contributor Rules

1. `NodeSpec` is primary for structural node data.
2. `NodeContract` is primary for topology and execution semantics.
3. `registry.py` holds shared compatibility rules and explicit exceptions only.
4. `engine.py` validates; it must not redefine semantics.
5. `InferenceWorker` executes; it must not invent contracts.

## 12. Verification Expectations

Required coverage categories:

1. compatibility matrix tests
2. required-input and cycle diagnostics tests
3. critical pipeline execution tests
4. negative runtime failure-path tests
5. regression coverage for existing preset behavior

At the time of writing, the project has direct automated coverage for:
- topology validation
- compatibility exceptions
- deferred BiRefNet planning
- SAM propagation policy
- CorridorKey missing alphahint failure path
- Matting missing mask failure path
- contract/spec alignment

## 13. Status

This document reflects the completed standardization work through:

1. Phase 1: inventory and baseline
2. Phase 2: canonical standard
3. Phase 3: safe implementation alignment
4. Phase 4: scenario-to-test mapping and closure
5. Phase 5: adoption playbook