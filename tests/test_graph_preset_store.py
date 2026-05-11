import json

from app.coordinators.graph_preset_store_coordinator import GraphPresetStoreCoordinator


class _SettingsStub:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def value(self, key, default=""):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value


class _DialogStub:
    def builtin_matanyone2_preset(self):
        return {"nodes": [{"id": "fallback", "type": "load"}], "connections": []}

    def builtin_corridorkey_gvm_preset(self):
        return {"nodes": [{"id": "fallback", "type": "source"}], "connections": []}


def _store(settings=None):
    return GraphPresetStoreCoordinator(
        settings=settings or _SettingsStub(),
        get_dialog=lambda: _DialogStub(),
        graph_matanyone2_template_settings_key="graph_presets/matanyone2_template_json",
        graph_builtin_preset_key="builtin:matanyone2",
        graph_builtin_corridorkey_gvm_preset_key="builtin:corridorkey_gvm",
    )


def test_builtin_presets_are_loaded_from_packaged_templates():
    presets = _store().graph_builtin_presets()

    assert set(presets) == {"builtin:matanyone2", "builtin:corridorkey_gvm"}
    assert [node["type"] for node in presets["builtin:matanyone2"]["nodes"]] == [
        "source",
        "sam2",
        "matting",
        "export",
        "export",
    ]
    assert [node["type"] for node in presets["builtin:corridorkey_gvm"]["nodes"]] == [
        "source",
        "gvm",
        "corridorkey",
        "export",
        "export",
        "export",
        "export",
    ]


def test_local_matanyone2_template_does_not_override_packaged_builtin():
    local_template = {"nodes": [{"id": "local", "type": "debug"}], "connections": []}
    settings = _SettingsStub({"graph_presets/matanyone2_template_json": json.dumps(local_template)})

    presets = _store(settings).graph_builtin_presets()

    assert presets["builtin:matanyone2"]["nodes"][0]["id"] == "n0"