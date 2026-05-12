"""Bridges Sam2NodeController <-> NodeGraphDialog."""
from __future__ import annotations
import logging, shutil
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from app.utils.write_paths import build_keyflow_internal_dir
logger = logging.getLogger(__name__)


class Sam2GraphCoordinator:
    def __init__(self, sam2, get_dialog, get_input_path, get_frame_index, get_fallback_rows=None):
        self._sam2 = sam2
        self._get_dialog = get_dialog
        self._get_input_path = get_input_path
        self._get_frame_index = get_frame_index
        self._get_fallback_rows = get_fallback_rows or (lambda: [])
        self._syncing = False
        self._persist_masks_cache = None

    @staticmethod
    def normalize_mask_to_binary(mask):
        arr = np.asarray(mask, dtype=np.uint8)
        if arr.ndim != 2:
            return None
        return np.where(arr > 127, 255, 0).astype(np.uint8)

    def load_mask_from_file(self, path):
        mask_path = Path(path) if path else None
        if mask_path is None or not mask_path.exists():
            return None
        try:
            arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        except Exception:
            return None
        return self.normalize_mask_to_binary(arr)

    def selected_graph_mask_rows(self):
        dialog = self._get_dialog()
        if dialog is None:
            return []
        method = getattr(dialog, "selected_sam_mask_rows", None)
        if not callable(method):
            return []
        rows = method()
        return rows if isinstance(rows, list) else []

    def has_connected_write_target(self):
        dialog = self._get_dialog()
        if dialog is None or not hasattr(dialog, "connected_write_targets"):
            return False
        try:
            targets = dialog.connected_write_targets()
        except Exception:
            return False
        for target in targets or []:
            if str((target or {}).get("source_node_type", "")).strip().lower() in {"sam2"}:
                return True
        return False

    def build_frame_masks(self, current_frame_index=None):
        frame_masks = {}
        for frame_idx, mask in list(self._sam2.state.added_masks or []):
            normalized = self.normalize_mask_to_binary(mask)
            if normalized is None:
                continue
            frame_masks[int(frame_idx)] = normalized
        return frame_masks

    @staticmethod
    def _mask_cache_dir(src: Path) -> Path:
        return build_keyflow_internal_dir(src, "sam_graph_masks")

    @staticmethod
    def _cleanup_legacy_sidecar_paths(src: Path) -> None:
        legacy_file = src.parent / f"{src.stem}__sam_graph_mask.png"
        legacy_dir = src.parent / f"{src.stem}__sam_graph_masks"
        try:
            legacy_file.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if legacy_dir.exists() and legacy_dir.is_dir():
                shutil.rmtree(legacy_dir)
        except Exception:
            pass

    def persist_masks(self, *, force_disk=False):
        mask_source_path = ""
        mask_payloads = []
        if not force_disk:
            return mask_source_path, mask_payloads
        input_path = self._get_input_path() or ""
        masks_now = list(self._sam2.state.added_masks or [])
        if not self.has_connected_write_target():
            if input_path:
                self._cleanup_legacy_sidecar_paths(Path(input_path))
            self._persist_masks_cache = None
            return mask_source_path, mask_payloads

        cache_entries = []
        for frame_index, mask in masks_now:
            mask_bin = self.normalize_mask_to_binary(mask)
            if mask_bin is None:
                continue
            digest = hashlib.blake2b(mask_bin.tobytes(), digest_size=8).hexdigest()
            cache_entries.append((int(frame_index), mask_bin.shape, str(mask_bin.dtype), digest))

        cache_key = (input_path, cache_entries)
        if self._persist_masks_cache == cache_key and self._persist_masks_cache is not None:
            dialog = self._get_dialog()
            if dialog is not None:
                src_method = getattr(dialog, "sam_node_mask_source_path", None)
                pay_method = getattr(dialog, "sam_node_mask_payloads", None)
                cached_src = str(src_method() if callable(src_method) else "").strip()
                cached_pay = pay_method() if callable(pay_method) else []
                if isinstance(cached_pay, list) and cached_pay:
                    return cached_src, cached_pay
        self._persist_masks_cache = cache_key
        mask_path = self._sam2.resolve_mask_path_for_processing(None)
        if mask_path and Path(mask_path).exists():
            if not input_path:
                mask_source_path = mask_path
            else:
                src = Path(input_path)
                mask_dir = self._mask_cache_dir(src)
                try:
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    persistent = mask_dir / "sam_graph_mask.png"
                    shutil.copy2(mask_path, str(persistent))
                    self._cleanup_legacy_sidecar_paths(src)
                    mask_source_path = str(persistent)
                except Exception:
                    mask_source_path = mask_path
        if not masks_now or not input_path:
            return mask_source_path, mask_payloads
        src = Path(input_path)
        mask_dir = self._mask_cache_dir(src)
        try:
            mask_dir.mkdir(parents=True, exist_ok=True)
            self._cleanup_legacy_sidecar_paths(src)
        except Exception:
            return mask_source_path, mask_payloads
        valid_names = set()
        for frame_index, mask in masks_now:
            mask_bin = self.normalize_mask_to_binary(mask)
            if mask_bin is None:
                continue
            file_name = f"sam_mask_{int(frame_index):05d}.png"
            out_path = mask_dir / file_name
            try:
                Image.fromarray(mask_bin).save(str(out_path))
            except Exception:
                continue
            valid_names.add(file_name)
            mask_payloads.append({"frame_index": int(frame_index), "path": str(out_path)})
        try:
            for stale in mask_dir.glob("sam_mask_*.png"):
                if stale.name not in valid_names:
                    stale.unlink(missing_ok=True)
        except Exception:
            pass
        return mask_source_path, mask_payloads

    def restore_masks_from_graph_node(self):
        dialog = self._get_dialog()
        if dialog is None:
            return
        payloads = []
        method = getattr(dialog, "sam_node_mask_payloads", None)
        if callable(method):
            result = method()
            if isinstance(result, list):
                payloads = result
        restored = []
        for entry in payloads:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path", "")).strip()
            if not path:
                continue
            try:
                frame_index = int(entry.get("frame_index", 0) or 0)
            except Exception:
                frame_index = 0
            normalized = self.load_mask_from_file(path)
            if normalized is None:
                continue
            restored.append((frame_index, normalized))
        if not restored:
            src_method = getattr(dialog, "sam_node_mask_source_path", None)
            single = str(src_method() if callable(src_method) else "").strip()
            normalized = self.load_mask_from_file(single)
            if normalized is not None:
                restored.append((0, normalized))
        if not restored:
            return
        self._sam2.state.added_masks = restored
        self._sam2.state.current_mask = None
        self._sam2.mask_list_changed.emit()

    def sync_to_graph(self, status_text=None):
        if self._syncing:
            return
        dialog = self._get_dialog()
        if dialog is None or not callable(getattr(dialog, "sync_sam_runtime_state", None)):
            return
        masks_now = list(self._sam2.state.added_masks or [])
        has_sequence_masks = len({int(frame_idx) for frame_idx, _mask in masks_now}) > 1
        selected_rows = [] if has_sequence_masks else self.selected_graph_mask_rows()
        if not selected_rows and not has_sequence_masks:
            selected_rows = self._get_fallback_rows()
        self._syncing = True
        try:
            sync = self._sam2.graph_sync_dict(selected_rows)
            if status_text is not None:
                sync["status_text"] = status_text
            sync["mask_source_path"], sync["mask_payloads"] = self.persist_masks(force_disk=True)
            dialog.sync_sam_runtime_state(**sync)
        finally:
            self._syncing = False
