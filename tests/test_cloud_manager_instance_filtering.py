from app import cloud_manager


def test_filter_keeps_only_keyflow_suitable_instance_types_when_available():
    instance_types = [
        "g4dn.xlarge",
        "g5.2xlarge",
        "g5.xlarge",
        "g5.12xlarge",
        "g6.xlarge",
        "p3.2xlarge",
    ]

    filtered = cloud_manager._filter_keyflow_supported_instance_types(instance_types)

    assert filtered == ["g5.xlarge", "g5.2xlarge", "g6.xlarge"]


def test_filter_falls_back_to_original_ordered_list_when_no_suitable_types_exist():
    instance_types = [
        "p3.2xlarge",
        "g4dn.2xlarge",
        "g4dn.xlarge",
    ]

    filtered = cloud_manager._filter_keyflow_supported_instance_types(instance_types)

    assert filtered == ["g4dn.xlarge", "g4dn.2xlarge", "p3.2xlarge"]