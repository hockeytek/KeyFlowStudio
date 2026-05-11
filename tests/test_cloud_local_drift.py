import ast
from pathlib import Path

from app.cloud_manager import build_cloud_worker_bundle_manifest
from app.node_graph.specs import NODE_SPECS
from app.node_graph.specs.corridorkey import SPEC as CORRIDORKEY_SPEC
from app.node_graph.specs.gvm import SPEC as GVM_SPEC


REPO_ROOT = Path(__file__).resolve().parents[1]
CLOUD_WORKER_PATH = REPO_ROOT / "ec2_worker" / "worker.py"


def _worker_literal(name: str):
    tree = ast.parse(CLOUD_WORKER_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {CLOUD_WORKER_PATH}")


def _spec_defaults(spec, keys: set[str]) -> dict:
    return {key: spec.default_properties[key] for key in keys}


def test_cloud_worker_bundle_manifest_includes_gvm_patch_layer():
    manifest = build_cloud_worker_bundle_manifest()
    file_paths = {entry["path"] for entry in manifest["files"]}

    assert "ec2_worker/_patch/gvm/pipelines/pipeline_gvm.py" in file_paths
    assert "ec2_worker/_patch/gvm_core/wrapper.py" in file_paths


def test_cloud_processing_types_are_registered_node_specs():
    processing_types = _worker_literal("_CLOUD_PROCESSING_TYPES")

    assert processing_types == {"gvm", "corridorkey"}
    assert processing_types <= set(NODE_SPECS)


def test_cloud_gvm_defaults_match_node_spec():
    cloud_defaults = _worker_literal("_CLOUD_GVM_DEFAULTS")

    assert cloud_defaults == _spec_defaults(GVM_SPEC, set(cloud_defaults))


def test_cloud_corridorkey_defaults_match_node_spec():
    cloud_defaults = _worker_literal("_CLOUD_CORRIDORKEY_DEFAULTS")

    assert cloud_defaults == _spec_defaults(CORRIDORKEY_SPEC, set(cloud_defaults))