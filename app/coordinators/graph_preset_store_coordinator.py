"""Graph preset storage/signature logic extracted from MainWindow."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable


class GraphPresetStoreCoordinator:
    """Owns custom/builtin preset storage, migration, and signature helpers."""

    _PACKAGED_PRESET_FILES = {
        "matanyone2": "matanyone2.json",
        "corridorkey_gvm": "corridorkey_gvm.json",
    }

    def __init__(
        self,
        *,
        settings,
        get_dialog: Callable,
        graph_matanyone2_template_settings_key: str,
        graph_builtin_preset_key: str,
        graph_builtin_corridorkey_gvm_preset_key: str,
    ) -> None:
        self._settings = settings
        self._get_dialog = get_dialog
        self._graph_matanyone2_template_settings_key = graph_matanyone2_template_settings_key
        self._graph_builtin_preset_key = graph_builtin_preset_key
        self._graph_builtin_corridorkey_gvm_preset_key = graph_builtin_corridorkey_gvm_preset_key

    def load_custom_presets(self) -> dict[str, dict]:
        raw = self._settings.value("graph_presets/custom_json", "")
        if not raw:
            return {}
        try:
            data = json.loads(str(raw))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        result: dict[str, dict] = {}
        reserved_casefold = {"matanyone2", "corridorkey+gvm", "111"}
        for name, preset in data.items():
            if isinstance(name, str) and isinstance(preset, dict):
                if name.strip().casefold() in reserved_casefold:
                    continue
                result[name] = preset
        return result

    def load_template_preset(self, settings_key: str) -> dict | None:
        raw = self._settings.value(settings_key, "")
        if not raw:
            return None
        try:
            data = json.loads(str(raw))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def save_template_preset(self, settings_key: str, preset: dict) -> None:
        self._settings.setValue(settings_key, json.dumps(preset, ensure_ascii=False))

    def migrate_legacy_matanyone2_graph_template(self) -> None:
        raw = self._settings.value("graph_presets/custom_json", "")
        if not raw:
            return
        try:
            data = json.loads(str(raw))
        except Exception:
            return
        if not isinstance(data, dict):
            return

        template = self.load_template_preset(self._graph_matanyone2_template_settings_key)
        changed = False
        migrated_template = template

        for name in list(data.keys()):
            preset = data.get(name)
            if not isinstance(name, str) or not isinstance(preset, dict):
                continue
            normalized_name = name.strip().casefold()
            if normalized_name not in {"111", "matanyone2"}:
                continue
            if migrated_template is None:
                migrated_template = preset
            data.pop(name, None)
            changed = True

        if migrated_template is not None and migrated_template is not template:
            self.save_template_preset(self._graph_matanyone2_template_settings_key, migrated_template)
            changed = True

        if changed:
            self._settings.setValue("graph_presets/custom_json", json.dumps(data, ensure_ascii=False))

    def save_custom_presets(self, custom_presets: dict[str, dict]) -> None:
        self._settings.setValue("graph_presets/custom_json", json.dumps(custom_presets, ensure_ascii=False))

    def graph_builtin_presets(self) -> dict[str, dict]:
        dialog = self._get_dialog()
        if dialog is None:
            return {}
        result: dict[str, dict] = {}
        matanyone2_template = self._load_packaged_preset("matanyone2")
        if isinstance(matanyone2_template, dict):
            result[self._graph_builtin_preset_key] = matanyone2_template
        else:
            matanyone2 = getattr(dialog, "builtin_matanyone2_preset", None)
            if callable(matanyone2):
                result[self._graph_builtin_preset_key] = matanyone2()
        packaged_corridorkey = self._load_packaged_preset("corridorkey_gvm")
        if isinstance(packaged_corridorkey, dict):
            result[self._graph_builtin_corridorkey_gvm_preset_key] = packaged_corridorkey
            return result
        corridorkey = getattr(dialog, "builtin_corridorkey_gvm_preset", None)
        if callable(corridorkey):
            result[self._graph_builtin_corridorkey_gvm_preset_key] = corridorkey()
        return result

    def _load_packaged_preset(self, name: str) -> dict | None:
        filename = self._PACKAGED_PRESET_FILES.get(name)
        if not filename:
            return None
        path = Path(__file__).resolve().parents[1] / "assets" / "graph_presets" / filename
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def graph_signature_from_preset(self, preset: dict | None) -> str:
        if not isinstance(preset, dict):
            return ""

        nodes_data = preset.get("nodes", [])
        edges_data = preset.get("connections", [])
        if not isinstance(nodes_data, list) or not isinstance(edges_data, list):
            return ""

        volatile_keys = {
            "sam_status",
            "mask_items",
            "selected_mask_rows",
            "current_mask_ready",
            "_mask_source_path",
            "mask_payloads",
        }
        normalized_nodes: list[dict] = []
        for node in nodes_data:
            if not isinstance(node, dict):
                continue
            raw_props = node.get("properties", {})
            props = {}
            if isinstance(raw_props, dict):
                props = {str(k): copy.deepcopy(v) for k, v in raw_props.items() if str(k) not in volatile_keys}
            normalized_nodes.append(
                {
                    "id": str(node.get("id", "")),
                    "type": str(node.get("type", "")),
                    "x": float(node.get("x", 0.0)),
                    "y": float(node.get("y", 0.0)),
                    "title": str(node.get("title", "")),
                    "custom_title": bool(node.get("custom_title", False)),
                    "properties": props,
                }
            )

        normalized_edges: list[dict] = []
        for edge in edges_data:
            if not isinstance(edge, dict):
                continue
            normalized_edges.append(
                {
                    "src": str(edge.get("src", "")),
                    "dst": str(edge.get("dst", "")),
                    "src_port": str(edge.get("src_port", "")),
                    "dst_port": str(edge.get("dst_port", "")),
                }
            )

        normalized = {
            "nodes": sorted(
                normalized_nodes,
                key=lambda n: (
                    n["id"],
                    n["type"],
                    n["x"],
                    n["y"],
                    n["title"],
                    json.dumps(n["properties"], sort_keys=True, ensure_ascii=False),
                ),
            ),
            "connections": sorted(
                normalized_edges,
                key=lambda e: (e["src"], e["dst"], e["src_port"], e["dst_port"]),
            ),
        }
        return json.dumps(normalized, sort_keys=True, ensure_ascii=False)

    def current_graph_signature(self) -> str:
        dialog = self._get_dialog()
        if dialog is None:
            return ""
        return self.graph_signature_from_preset(dialog.export_graph_preset())

    def graph_preset_payload(self, key: str, custom_presets: dict[str, dict]) -> dict | None:
        builtins = self.graph_builtin_presets()
        if key in builtins:
            return builtins.get(key)
        if key.startswith("custom:"):
            return custom_presets.get(key.split(":", 1)[1])
        return None
