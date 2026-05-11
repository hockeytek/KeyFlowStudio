from app.cloud_manager import build_cloud_worker_bundle_manifest


def test_cloud_worker_bundle_manifest_contains_expected_core_files():
    manifest = build_cloud_worker_bundle_manifest()

    assert len(manifest["revision"]) == 16
    assert manifest["file_count"] > 0

    file_paths = {entry["path"] for entry in manifest["files"]}
    assert "ec2_worker/worker.py" in file_paths
    assert "ec2_worker/download_models.py" in file_paths
    assert "app/services/model_service.py" in file_paths
    assert "app/services/birefnet_service.py" in file_paths
    assert "app/utils/__init__.py" in file_paths