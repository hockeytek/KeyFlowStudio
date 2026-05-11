"""Export node runtime handler."""

from __future__ import annotations

from pathlib import Path

from app.node_graph.models import GraphNode
from app.node_graph.nodes.base import NodeExecutionContext, NodeExecutionError
from app.utils.write_paths import build_keyflow_base_dir


class ExportNodeHandler:
    key = "export"

    @staticmethod
    def _extract_media_path(payload: dict) -> str:
        current = payload
        seen: set[int] = set()
        while isinstance(current, dict):
            current_id = id(current)
            if current_id in seen:
                return ""
            seen.add(current_id)

            media_path = str(current.get("media_path", "")).strip()
            if media_path:
                return media_path
            current = current.get("upstream")
        return ""

    @staticmethod
    def _resolve_output_dir(media_path: str, props: dict) -> str:
        if bool(props.get("auto_output_dir", True)) and media_path:
            return str(build_keyflow_base_dir(Path(media_path)))
        return str(props.get("output_dir", "")).strip()

    @staticmethod
    def _resolve_output_name(media_path: str, props: dict) -> str:
        custom_name = str(props.get("file_name", "")).strip()
        if custom_name:
            return custom_name
        if media_path:
            return Path(media_path).stem
        return ""

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, dict],
        context: NodeExecutionContext,
    ) -> dict[str, dict]:
        img_data = inputs.get("in", {})
        if not img_data:
            raise NodeExecutionError("Export: missing upstream payload on 'in' port")

        props = node.properties or {}
        media_path = self._extract_media_path(img_data)
        return {"out": {
            "export": {
                "auto_output_dir": bool(props.get("auto_output_dir", True)),
                "output_dir": self._resolve_output_dir(media_path, props),
                "file_name": self._resolve_output_name(media_path, props),
                "output_format": str(props.get("output_format", "source")).strip().lower() or "source",
                "save_foreground": bool(props.get("save_foreground", True)),
                "save_alpha": bool(props.get("save_alpha", True)),
            },
            "upstream": img_data,
        }}
