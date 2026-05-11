"""Load Media node runtime handler."""

from __future__ import annotations

import os
from pathlib import Path

from app.constants import DEFAULT_FPS
from app.node_graph.models import GraphNode
from app.node_graph.nodes.base import NodeExecutionContext, NodeExecutionError
from app.utils.media import is_numbered_image_sequence, read_media_dimensions, resolve_numbered_image_sequence


class LoadMediaNodeHandler:
    key = "load"

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, dict],
        context: NodeExecutionContext,
    ) -> dict[str, dict]:
        props = node.properties or {}
        media_path = str(props.get("path", "")).strip()
        media_type = str(props.get("media_type", "video")).strip().lower()

        if not media_path:
            raise NodeExecutionError("Load Media: path is empty")
        if not os.path.exists(media_path):
            raise NodeExecutionError(f"Load Media: file not found: {media_path}")

        metadata = self._collect_metadata(media_path, media_type)
        payload = {
            "media_path": media_path,
            "media_type": media_type,
            "metadata": metadata,
        }
        context.state["last_media"] = payload
        return {"out": payload}

    def _collect_metadata(self, media_path: str, media_type: str) -> dict:
        result = {
            "name": Path(media_path).name,
            "size_bytes": int(os.path.getsize(media_path)),
        }

        if is_numbered_image_sequence(media_path):
            try:
                dimensions = read_media_dimensions(media_path, "video")
                if dimensions is None:
                    raise ValueError("sequence dimensions unavailable")
                sequence_paths = resolve_numbered_image_sequence(media_path)
                w, h = dimensions
                result.update(
                    {
                        "width": w,
                        "height": h,
                        "frames": len(sequence_paths),
                        "fps": DEFAULT_FPS,
                        "duration_sec": len(sequence_paths) / DEFAULT_FPS,
                        "kind": "sequence",
                    }
                )
            except Exception:
                result.update({"kind": "video", "read_error": True})
            return result

        if media_type == "image":
            try:
                dimensions = read_media_dimensions(media_path, media_type)
                if dimensions is None:
                    raise ValueError("image dimensions unavailable")
                w, h = dimensions
                result.update({"width": w, "height": h, "kind": "image"})
            except Exception:
                result.update({"kind": "image", "read_error": True})
            return result

        try:
            import cv2

            cap = cv2.VideoCapture(media_path)
            try:
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    result.update(
                        {
                            "width": w,
                            "height": h,
                            "frames": frames,
                            "fps": fps,
                            "duration_sec": (frames / fps) if fps > 0 else 0.0,
                            "kind": "video",
                        }
                    )
                else:
                    result.update({"kind": "video", "read_error": True})
            finally:
                cap.release()
        except Exception:
            result.update({"kind": "video", "read_error": True})

        return result
