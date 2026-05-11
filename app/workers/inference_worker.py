"""Worker thread for inference"""
import shutil
import tempfile
import threading
import traceback
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import logging

from PySide6.QtCore import Signal, QObject

from app.constants import (
    DEFAULT_FPS,
    DEFAULT_VIDEO_CODEC,
    DEFAULT_VIDEO_CRF,
    DEFAULT_VIDEO_PRESET,
    PRORES_PROFILES,
)
from app.i18n import t
from app.settings import get_app_settings
from app.services import InferenceService
from app.services.corridorkey_service import CorridorKeyService
from app.services.birefnet_service import BiRefNetService
from app.services.gvm_service import GVMService
from app.services.sam3_service import Sam3Service
from app.node_graph.diagnostics import format_graph_diagnostics_summary, format_graph_diagnostics_text
from app.node_graph.models import GraphNode, GraphEdge
from app.workers.graph_write_planner import build_graph_write_plan_targets
from app.utils.media import (
    is_supported_image_file,
    is_numbered_image_sequence,
    load_image_sequence,
    load_rgb_image,
    load_image_float,
)
from app.utils.corridorkey_output import (
    build_corridorkey_processed_output,
    coerce_alpha_2d,
    coerce_rgb_float01,
)
from app.utils.write_output import (
    COMPAT_IMAGE_OUTPUT_FORMATS,
    COMPAT_VIDEO_OUTPUT_FORMATS,
    build_video_output_params,
    prepare_video_frame,
    promote_alpha_to_rgba_exr,
    resolve_write_output_format,
    save_image_frame,
    to_u16_frame,
    to_u8_frame,
    is_normalized_float_range,
)
from app.utils.write_paths import (
    build_keyflow_output_dir,
    get_port_output_label,
    normalize_write_stream_name,
    resolve_graph_write_output_dir,
)
import app.node_graph.engine as engine_module
from app.runtime_contract import (
    RUNTIME_CANCEL_CLEANUP_PARTIAL,
    RUNTIME_CANCEL_SAVE_PARTIAL,
    RUNTIME_SEMANTICS_PREVIEW_ONLY,
    RUNTIME_SEMANTICS_PRODUCTION_SAFE,
    make_runtime_result_cancelled_partial,
    make_runtime_result_cancelled,
    make_runtime_result_ok,
    make_stream_preview_payload,
    normalize_cancel_policy,
)

logger = logging.getLogger(__name__)

def _build_keyflow_output_dir(source: Path, stream_label: str) -> Path:
    """Return auto output directory based on input file/folder name.

    - Single file  /data/clip.mp4   → /data/clip_keyflow/<stream_label>/
    - Image seq    /imgs/0001.png   → /data/imgs_keyflow/<stream_label>/ (next to folder)
    """
    return build_keyflow_output_dir(source, stream_label)


class InferenceWorker(QObject):
    """
    Worker для обработки видео в отдельном потоке.
    Использует сигналы для обновления UI.
    """

    # Сигналы
    progress = Signal(int, int)   # (current_frame, total_frames)
    preview_frame = Signal(object, object, int)  # (foreground_rgb, alpha_rgb, frame_index_0_based)
    graph_stream_preview = Signal(str, object, int)  # (write_node_id, frame_rgb_or_gray, frame_index_0_based)
    stage_progress = Signal(int, str)  # (percent, status_text)
    node_frame_progress = Signal(str, int, int)  # (node_type, current_frame_1based, total_frames)
    log_message = Signal(str)
    corridorkey_mode_resolved = Signal(str, str, str)  # (requested_mode, effective_mode, reason_key)
    finished = Signal(dict)       # результаты обработки
    error = Signal(str)           # сообщение об ошибке

    def __init__(self):
        super().__init__()
        self._cleanup_stale_temp_dirs()
        self.inference_service = InferenceService()
        self.corridorkey_service = CorridorKeyService()
        self.birefnet_service = BiRefNetService()
        self.birefnet_service.set_callbacks(
            progress_callback=lambda percent, msg: self.stage_progress.emit(percent, msg),
            translate=self._tr,
        )
        self.gvm_service = GVMService()
        self.cancel_flag = threading.Event()
        self._pending_job = None
        self.language_code = "ru"
        self._graph_output_dir: Path | None = None
        self._graph_source_path: str = ""
        self._graph_mask_path: str = ""
        self._graph_fps: float = DEFAULT_FPS
        self._graph_audio_path: str = ""
        self._graph_write_plans: dict[tuple[str, str], list[dict]] = {}
        self._graph_stream_saved_paths: dict[str, Path] = {}
        self._graph_downstream_targets: dict[tuple[str, str], list[dict]] = {}
        self._graph_correction_masks: dict[int, np.ndarray] | None = None
        self._sam_service = None

    def set_language(self, language_code: str):
        self.language_code = language_code if language_code in {"ru", "en"} else "ru"
        self.birefnet_service.set_callbacks(
            progress_callback=lambda percent, msg: self.stage_progress.emit(percent, msg),
            translate=self._tr,
        )
        self.gvm_service.set_callbacks(
            progress_callback=lambda percent, msg: self.stage_progress.emit(percent, msg),
            translate=self._tr,
        )

    @staticmethod
    def _cleanup_stale_temp_dirs() -> None:
        """Remove leftover keyflow_alphahint_* and keyflow_gvm_in_* dirs from previous crashed runs."""
        try:
            tmp = Path(tempfile.gettempdir())
            for pattern in ("keyflow_alphahint_*", "keyflow_gvm_in_*"):
                for d in tmp.glob(pattern):
                    if d.is_dir():
                        shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass

    def _tr(self, key: str) -> str:
        return t(key, self.language_code)

    def set_cancel(self):
        self.cancel_flag.set()

    def reset_cancel(self):
        self.cancel_flag.clear()

    def _unload_sam_service_if_loaded(self) -> bool:
        """Unload SAM/SAM2 service if the current job still holds it.

        By the time graph execution reaches CorridorKey or MatAnyone2, SAM outputs
        are already materialized in memory, so keeping model weights resident only
        wastes VRAM/RAM.
        """
        sam_service = getattr(self, "_sam_service", None)
        if sam_service is None or not hasattr(sam_service, "unload"):
            return False
        try:
            sam_service.unload()
            self.log_message.emit(self._tr("worker_sam_unloaded"))
            return True
        finally:
            self._sam_service = None

    def configure_job(self, video_path, mask_path, output_dir, config):
        self._pending_job = (video_path, mask_path, output_dir, config)

    def start_job(self, video_path, mask_path, output_dir, config, sam_service=None):
        """Configure and run in worker thread via queued signal."""
        self._sam_service = sam_service
        self.configure_job(video_path, mask_path, output_dir, config)
        self.run()

    def run(self):
        if self._pending_job is None:
            self.error.emit(self._tr("worker_job_not_configured"))
            return

        video_path, mask_path, output_dir, config = self._pending_job
        self.process_video(video_path, mask_path, output_dir, config)

    # ------------------------------------------------------------------
    # Точка входа
    # ------------------------------------------------------------------
    def process_video(self, video_path, mask_path, output_dir, config):
        """
        Обработать входные данные через node graph execution path.

        config keys:
            erode_kernel  int  (default 0)
            dilate_kernel int  (default 0)
            is_video      bool
            n_warmup      int  warmup кадры  (default 10)
            start_frame   int  0-based, включительно     (default 0)
            end_frame     int  0-based, исключительно    (default -1 = все кадры)
            node_graph    dict (required)
                Has keys: 'nodes' (list of GraphNode), 'edges' (list of GraphEdge)
        """
        self.reset_cancel()

        node_graph_data = config.get("node_graph")
        if node_graph_data is None:
            self.error.emit(
                "InferenceWorker requires 'node_graph' config. "
                "Legacy single-node MatAnyone2 path has been removed."
            )
            return

        self._process_with_node_graph(node_graph_data, video_path, mask_path, output_dir, config)

    def _process_with_node_graph(self, node_graph_data, video_path, mask_path, output_dir, config):
        """Обработать видео используя граф нод.
        
        node_graph_data: dict with 'nodes' and 'edges' keys
        """
        try:
            self.stage_progress.emit(2, self._tr("worker_inference_init"))
            self.log_message.emit(self._tr("worker_graph_processing"))

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._graph_output_dir = output_dir
            self._graph_source_path = str(video_path or "")
            self._graph_mask_path = str(mask_path or "")
            self._graph_fps = DEFAULT_FPS
            self._graph_audio_path = ""
            self._graph_write_plans = {}
            self._graph_stream_saved_paths = {}
            self._graph_correction_masks = config.get("correction_masks") or None
            cancel_policy = normalize_cancel_policy(config.get("cancel_policy"))

            is_video = config.get("is_video", True)
            start_frame = int(config.get("start_frame", 0))
            end_frame = int(config.get("end_frame", -1))
            self._graph_start_frame = start_frame
            self._graph_end_frame = end_frame

            # Parse node graph data
            nodes_data = node_graph_data.get("nodes", [])
            edges_data = node_graph_data.get("edges", [])
            
            # Convert to GraphNode and GraphEdge objects if needed
            if nodes_data and not isinstance(nodes_data[0], GraphNode):
                # Assume they're dicts
                nodes = [GraphNode(
                    id=n.get("id"),
                    type=n.get("type"),
                    title=n.get("title", n.get("id")),
                    properties=n.get("properties", {}),
                    enabled=n.get("enabled", True)
                ) for n in nodes_data]
            else:
                nodes = nodes_data
            
            if edges_data and not isinstance(edges_data[0], GraphEdge):
                # Assume they're dicts
                from app.node_graph.specs import get_node_spec
                node_types_by_id = {str(n.get("id")): str(n.get("type")) for n in nodes_data if isinstance(n, dict)}
                edges = []
                for e in edges_data:
                    src_id = e.get("src_id") or e.get("src")
                    dst_id = e.get("dst_id") or e.get("dst")
                    src_port = str(e.get("src_port", "") or "").strip().lower()
                    if not src_port:
                        src_type = node_types_by_id.get(str(src_id), "")
                        src_spec = get_node_spec(src_type)
                        if src_spec is not None and len(src_spec.outputs) == 1:
                            src_port = str(src_spec.outputs[0].name or "out").strip().lower() or "out"
                        else:
                            src_port = "out"
                    dst_port = str(e.get("dst_port", "") or "").strip().lower()
                    if node_types_by_id.get(str(dst_id), "") == "export" and not dst_port:
                        dst_port = "in"
                    edges.append(
                        GraphEdge(
                            src_id=src_id,
                            dst_id=dst_id,
                            src_port=src_port,
                            dst_port=dst_port,
                        )
                    )
            else:
                edges = edges_data

            # Load frames
            self.stage_progress.emit(12, self._tr("worker_graph_loading_input"))
            # If no global video_path, try to resolve it from a source node's properties
            effective_video_path = str(video_path or "").strip()
            if not effective_video_path:
                for _n in nodes:
                    if str(getattr(_n, "type", "")) in {"source", "load", "load_media"}:
                        _np = str((_n.properties or {}).get("path", "")).strip()
                        if _np and Path(_np).exists():
                            effective_video_path = _np
                            break
            if effective_video_path:
                self._graph_source_path = effective_video_path
            if is_video:
                if effective_video_path:
                    frames, fps, audio_path = self._load_video(effective_video_path, output_dir)
                else:
                    frames, fps, audio_path = [], DEFAULT_FPS, ""
                if end_frame > 0:
                    frames = frames[start_frame:end_frame]
                elif start_frame > 0:
                    frames = frames[start_frame:]
            else:
                if effective_video_path:
                    frame = self._load_image_frame(effective_video_path)
                    frames = [frame]
                else:
                    frames = []
                fps = DEFAULT_FPS
                audio_path = ""

            self._graph_fps = float(fps)
            self._graph_audio_path = str(audio_path or "")

            self.log_message.emit(self._tr("worker_graph_frames_loaded").format(total=len(frames)))

            self._prepare_graph_write_targets(nodes, edges, output_dir)

            # Execute node graph
            self.stage_progress.emit(15, self._tr("worker_graph_execute"))
            outputs = self._execute_node_graph(nodes, edges, frames)
            self._finalize_graph_stream_writes()

            if self.cancel_flag.is_set():
                if cancel_policy == RUNTIME_CANCEL_SAVE_PARTIAL:
                    self._finalize_graph_stream_writes(keep_outputs=True, emit_preview=True)
                    partial_saved = {k: str(v) for k, v in self._graph_stream_saved_paths.items() if v is not None}
                    self.finished.emit(make_runtime_result_cancelled_partial(partial_saved, len(frames)))
                elif cancel_policy == RUNTIME_CANCEL_CLEANUP_PARTIAL:
                    self._finalize_graph_stream_writes(keep_outputs=False, emit_preview=False)
                    self.finished.emit(make_runtime_result_cancelled())
                else:  # immediate
                    self._finalize_graph_stream_writes(keep_outputs=False, emit_preview=False)
                    self.finished.emit(make_runtime_result_cancelled())
                return

            # Save results
            self.stage_progress.emit(92, self._tr("worker_inference_saving"))
            saved_paths = self._save_node_graph_results(outputs, nodes, video_path, output_dir, fps, audio_path)

            self.stage_progress.emit(100, self._tr("status_done"))
            self.log_message.emit(self._tr("status_done"))
            self.finished.emit(
                make_runtime_result_ok(
                    {k: str(v) for k, v in saved_paths.items()},
                    len(frames),
                )
            )

        except ValueError as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(f"Graph execution failed: {str(e)}\n{traceback.format_exc()}")
        finally:
            # Keep default finalize on non-cancel paths and as safety-net.
            self._finalize_graph_stream_writes()
            self._graph_write_plans = {}
            self._graph_stream_saved_paths = {}
            self._graph_output_dir = None
            self._graph_source_path = ""
            self._graph_mask_path = ""
            self._graph_fps = DEFAULT_FPS
            self._graph_audio_path = ""
            self._graph_correction_masks = None
            self._graph_start_frame = 0
            self._graph_end_frame = -1

    def _save_alpha_sequence(self, alpha_frames: list[np.ndarray], target_dir: Path) -> None:
        """Save alpha frames as numbered 8-bit PNG sequence."""
        target_dir.mkdir(parents=True, exist_ok=True)
        for i, alpha in enumerate(alpha_frames):
            self._save_alpha_frame(alpha, i, target_dir)

    @staticmethod
    def _apply_birefnet_mask_morphology(alpha: np.ndarray, dilate_radius: int, erode_radius: int) -> np.ndarray:
        """Apply optional post-process morphology for BiRefNet alpha mask.

        Includes binary thresholding as in the original CorridorKey project:
        CorridorKey is trained on coarse binary masks and recovers fine detail
        from the hint itself, so soft BiRefNet probabilities are binarised first.
        """
        mask = np.asarray(alpha, dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Binary threshold matching original CorridorKey pipeline
        # (threshold 10/255 ≈ 0.039 on float scale)
        mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        _, mask_u8 = cv2.threshold(mask_u8, 10, 255, cv2.THRESH_BINARY)
        mask = mask_u8.astype(np.float32) / 255.0

        if dilate_radius > 0:
            k = int(dilate_radius) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=1)

        if erode_radius > 0:
            k = int(erode_radius) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.erode(mask, kernel, iterations=1)

        return np.clip(mask.astype(np.float32), 0.0, 1.0)

    @staticmethod
    def _apply_corridorkey_alpha_controls(
        alpha: np.ndarray,
        *,
        clip_black: float,
        clip_white: float,
        matte_gamma: float,
        shrink_grow: float,
        edge_blur: float,
    ) -> np.ndarray:
        """Apply artist controls to CorridorKey alpha matte.

        This is a post-process stage that mirrors OFX-style matte controls.
        """
        mask = np.asarray(alpha, dtype=np.float32)
        if mask.ndim == 3 and mask.shape[2] >= 1:
            mask = mask[:, :, 0]
        mask = np.clip(mask, 0.0, 1.0)

        # Clip black/white remap into [0, 1].
        b = float(np.clip(clip_black, 0.0, 1.0))
        w = float(np.clip(clip_white, 0.0, 1.0))
        if w <= b:
            w = min(1.0, b + 1e-3)
        mask = np.clip((mask - b) / (w - b), 0.0, 1.0)

        # Shrink/Grow via morphology on 8-bit matte.
        sg = float(shrink_grow)
        if abs(sg) > 1e-6:
            radius = int(round(abs(sg)))
            if radius > 0:
                k = radius * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
                if sg > 0:
                    mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
                else:
                    mask_u8 = cv2.erode(mask_u8, kernel, iterations=1)
                mask = mask_u8.astype(np.float32) / 255.0

        # Edge blur.
        blur = float(max(0.0, edge_blur))
        if blur > 1e-6:
            radius = int(round(blur))
            if radius > 0:
                k = radius * 2 + 1
                mask = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)

        # Matte gamma.
        g = float(max(0.2, matte_gamma))
        if abs(g - 1.0) > 1e-6:
            mask = np.power(np.clip(mask, 0.0, 1.0), 1.0 / g)

        return np.clip(mask.astype(np.float32), 0.0, 1.0)

    @staticmethod
    def _is_normalized_float_range(min_val: float, max_val: float) -> bool:
        return is_normalized_float_range(min_val, max_val)

    @staticmethod
    def _save_alpha_frame(alpha_frame: np.ndarray, frame_index: int, target_dir: Path) -> None:
        """Save one alpha frame as numbered 8-bit PNG."""
        arr = np.asarray(alpha_frame)
        if arr.ndim == 3 and arr.shape[2] >= 1:
            arr = arr[:, :, 0]
        if arr.dtype != np.uint8:
            arr_f = arr.astype(np.float32)
            min_val = float(np.nanmin(arr_f)) if arr_f.size else 0.0
            max_val = float(np.nanmax(arr_f)) if arr_f.size else 0.0
            if InferenceWorker._is_normalized_float_range(min_val, max_val):
                arr_u8 = (np.clip(arr_f, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr_u8 = np.clip(arr_f, 0.0, 255.0).astype(np.uint8)
        else:
            arr_u8 = arr
        Image.fromarray(arr_u8, "L").save(str(target_dir / f"{frame_index:04d}.png"), compress_level=1)

    @staticmethod
    def _resolve_alpha_hint_mode(node: GraphNode | dict) -> str:
        """Resolve CorridorKey alpha hint mode from node properties."""
        props = {}
        if isinstance(node, GraphNode):
            props = node.properties or {}
        elif isinstance(node, dict):
            props = (node or {}).get("properties", {}) or {}

        mode = str(props.get("alpha_hint_mode", "auto")).strip().lower()
        if mode not in {"auto", "batch", "staged"}:
            return "auto"
        return mode

    def _mode_label(self, mode: str) -> str:
        mode_key = {
            "auto": "corridorkey_alpha_hint_mode_auto",
            "batch": "corridorkey_alpha_hint_mode_batch",
            "staged": "corridorkey_alpha_hint_mode_staged",
            "sam_disk": "corridorkey_alpha_hint_mode_sam_disk",
        }.get(mode, "corridorkey_alpha_hint_mode_auto")
        return self._tr(mode_key)

    def _emit_corridorkey_mode_indicator(
        self,
        requested_mode: str,
        effective_mode: str,
        reason_key: str,
    ) -> None:
        self.log_message.emit(
            self._tr("worker_corridorkey_mode_indicator").format(
                requested=self._mode_label(requested_mode),
                effective=self._mode_label(effective_mode),
                reason=self._tr(reason_key),
            )
        )

    def _execute_node_graph(self, nodes: list[GraphNode], edges: list[GraphEdge], frames: list) -> dict:
        """Выполнить граф нод для обработки кадров.
        
        Returns:
            dict: node_id → node_output (port_name → data)
        """
        engine = engine_module.NodeGraphEngine()
        strict_mode = bool(get_app_settings().value("node_graph/diag_strict_required_inputs", True, type=bool))
        plan, diagnostics = engine.build_execution_plan_with_diagnostics(
            nodes,
            edges,
            strict_isolated_required_inputs=strict_mode,
        )
        if plan is None:
            summary = format_graph_diagnostics_summary(self._tr, diagnostics)
            details = format_graph_diagnostics_text(self._tr, diagnostics)
            self.log_message.emit(summary)
            for line in details.splitlines():
                if line:
                    self.log_message.emit(line)
            errors_text = details
            raise ValueError(self._tr("worker_graph_validation_failed").format(errors=errors_text))

        # Build node lookup
        nodes_by_id = {node.id: node for node in nodes}
        self._graph_downstream_targets = {}
        for edge in edges:
            src_key = (str(edge.src_id), str(edge.src_port or "").strip().lower())
            dst_node = nodes_by_id.get(edge.dst_id)
            self._graph_downstream_targets.setdefault(src_key, []).append(
                {
                    "dst_id": str(edge.dst_id),
                    "dst_port": str(edge.dst_port or "").strip().lower(),
                    "dst_type": str(getattr(dst_node, "type", "") or "").strip().lower(),
                    "dst_enabled": bool(getattr(dst_node, "enabled", True)) if dst_node is not None else True,
                }
            )

        topo_order = list(plan.execution_order)
        logger.info(f"Executing nodes in order: {topo_order}")

        # Initialize outputs storage
        outputs = {}

        # Execute nodes in order
        for node_id in topo_order:
            if self.cancel_flag.is_set():
                break

            node = nodes_by_id.get(node_id)
            if node is None:
                continue

            action = str(plan.node_actions.get(node_id, "execute"))
            if action == "skip_disabled":
                continue
            if action == "skip_isolated":
                logger.debug(f"Skipping isolated node {node_id} ({node.type}) — not connected")
                continue

            node_type = node.type
            self.log_message.emit(
                self._tr("worker_graph_executing_node").format(node_id=node_id, node_type=node_type)
            )

            # Prepare inputs for this node
            inputs = self._gather_node_inputs(nodes_by_id, edges, node_id, outputs, frames)

            if action == "passthrough_source":
                node_props = node.properties or {}
                node_path = str(node_props.get("path", "")).strip()
                if node_type == "source":
                    # Source is the primary timeline media loaded at graph start.
                    node_frames = frames
                elif node_path and node_path != self._graph_source_path and Path(node_path).exists():
                    # Second (or override) load node — load from its own path
                    node_media_type = str(node_props.get("media_type", "video")).strip().lower()
                    is_image_sequence = is_numbered_image_sequence(node_path)
                    try:
                        if node_media_type == "image" and not is_image_sequence:
                            node_frames = [load_image_float(node_path)]
                        else:
                            node_frames, _, _ = self._load_video(
                                node_path, self._graph_output_dir or Path("."))
                            # Apply the same frame range that was applied to global frames
                            _sf = getattr(self, "_graph_start_frame", 0)
                            _ef = getattr(self, "_graph_end_frame", -1)
                            if _ef > 0:
                                node_frames = node_frames[_sf:_ef]
                            elif _sf > 0:
                                node_frames = node_frames[_sf:]
                        self.log_message.emit(
                            f"{node_type.capitalize()} node {node_id}: loaded {len(node_frames)} frame(s) from {Path(node_path).name}"
                        )
                    except Exception as _le:
                        logger.warning(
                            "%s node %s: failed to load %s: %s, using global frames",
                            node_type.capitalize(),
                            node_id,
                            node_path,
                            _le,
                        )
                        node_frames = frames
                else:
                    node_frames = frames
                bbox_sequence = [self._frame_bbox(frame) for frame in node_frames] if node_frames else []
                port_meta = {"bbox_sequence": bbox_sequence}
                outputs[node_id] = {
                    "out": node_frames,
                    "image": node_frames,
                    "frame_sequence": node_frames,
                    "__meta__": {
                        "out": dict(port_meta),
                        "image": dict(port_meta),
                        "frame_sequence": dict(port_meta),
                    },
                }
                continue

            if action == "deferred":
                deferred_type = nodes_by_id.get(node_id)
                deferred_type_str = deferred_type.type if deferred_type is not None else "unknown"
                if deferred_type_str in {"sam2"}:
                    outputs[node_id] = {"__deferred_sam_disk__": True, "out": None, "mask": None}
                    self.log_message.emit(
                        f"SAM2 node {node_id}: deferred (disk-streaming masks into CorridorKey)"
                    )
                else:
                    outputs[node_id] = {"__deferred_staged__": True, "alpha": None}
                    self.log_message.emit(
                        f"BiRefNet node {node_id}: deferred (staged into CorridorKey)"
                    )
                continue

            if action == "write_sink":
                outputs[node_id] = inputs
                continue

            if node_type == "corridorkey":
                self._unload_sam_service_if_loaded()
                deferred_id = plan.deferred_corridorkey_sources.get(node_id)
                if deferred_id:
                    deferred_node_obj = nodes_by_id.get(deferred_id)
                    if deferred_node_obj is not None:
                        if deferred_node_obj.type in {"sam2"}:
                            inputs["__deferred_sam_node"] = deferred_node_obj.__dict__
                        else:
                            inputs["__deferred_birefnet_node"] = deferred_node_obj.__dict__

            # Execute the node; unload heavy models in a finally block so
            # memory is freed even when the node raises an exception.
            _exc: BaseException | None = None
            try:
                node_output = self._execute_node(node_type, node.__dict__, inputs)
                outputs[node_id] = node_output
            except Exception as e:
                logger.error(f"Node {node_id} execution failed: {e}")
                _exc = e
            finally:
                # Always unload after GVM — diffusion model is very heavy.
                if node_type == "gvm":
                    self.gvm_service.unload()
                    self.log_message.emit("GVM: model unloaded after batch execution")

            if _exc is not None:
                raise _exc

            # Unload BiRefNet after it finishes (batch path) to free memory
            # before CorridorKey loads. Mirrors original CorridorKey project.
            if node_type == "birefnet":
                self.birefnet_service.unload_model()
                self.log_message.emit("BiRefNet: model unloaded after batch execution")

            # Unload CorridorKey after it finishes to free VRAM/RAM.
            # Symmetric to BiRefNet unload: heavy model (~300-600 MB) no longer
            # needed once this node's outputs are stored in `outputs`.
            if node_type == "corridorkey":
                corridorkey_service = getattr(self, "corridorkey_service", None)
                if corridorkey_service is not None and hasattr(corridorkey_service, "unload_engine"):
                    corridorkey_service.unload_engine()
                    self.log_message.emit("CorridorKey: model unloaded after execution")

        return outputs

    def _gather_node_inputs(self, nodes_by_id: dict, edges: list[GraphEdge], node_id: str, outputs: dict, initial_frames: list) -> dict:
        """Собрать входные данные для узла из выходов предыдущих узлов."""
        inputs = {}

        # Get all edges that point to this node
        for edge in edges:
            if edge.dst_id == node_id:
                src_node_id = edge.src_id
                src_port = str(edge.src_port or "").strip().lower()
                dst_port = edge.dst_port
                dst_node = nodes_by_id.get(node_id)
                src_node = nodes_by_id.get(src_node_id)
                if not src_port:
                    src_node_type = str(getattr(src_node, "type", "") or "").strip().lower()
                    from app.node_graph.specs import get_node_spec
                    src_spec = get_node_spec(src_node_type)
                    if src_spec is not None and len(src_spec.outputs) == 1:
                        src_port = str(src_spec.outputs[0].name or "out").strip().lower() or "out"
                    else:
                        src_port = "out"
                if str(getattr(dst_node, "type", "") or "") == "export" and not str(dst_port or "").strip():
                    dst_port = "in"

                # Get data from source node output
                if src_node_id in outputs:
                    src_output = outputs[src_node_id]
                    if isinstance(src_output, dict) and src_port in src_output:
                        inputs[dst_port] = src_output[src_port]
                        inputs[f"__src_port__{dst_port}"] = src_port
                        inputs[f"__src_node_type__{dst_port}"] = str(getattr(src_node, "type", "") or "")
                        inputs[f"__src_node_title__{dst_port}"] = str(getattr(src_node, "title", "") or "")
                        src_meta = src_output.get("__meta__") if isinstance(src_output, dict) else None
                        if isinstance(src_meta, dict) and src_port in src_meta:
                            inputs[f"__meta__{dst_port}"] = src_meta[src_port]
                else:
                    self.log_message.emit(
                        self._tr("worker_graph_source_not_ready").format(node_id=src_node_id)
                    )

        return inputs

    def _resolve_requested_output_ports(self, node_id: str, default_ports: set[str]) -> set[str]:
        """Return node output ports that are actually consumed by enabled downstream nodes."""
        targets = getattr(self, "_graph_downstream_targets", {}) or {}
        requested: set[str] = set()
        node_id_str = str(node_id)
        for (src_node_id, src_port), downstream in targets.items():
            if str(src_node_id) != node_id_str:
                continue
            if not isinstance(downstream, list):
                continue
            if any(bool(item.get("dst_enabled", True)) for item in downstream if isinstance(item, dict)):
                port = str(src_port or "").strip().lower()
                if port:
                    requested.add(port)

        if not requested:
            return set(default_ports)
        return requested & set(default_ports)

    def _save_node_graph_results(
        self,
        outputs: dict,
        nodes: list[GraphNode],
        source_path: str,
        output_dir: Path,
        fps: float = DEFAULT_FPS,
        audio_path: str = "",
    ) -> dict[str, Path]:
        """Сохранить результаты из граф-выполнения по нодам Write(export).

        Returns a dict mapping ``node_id`` → saved path for every Write node
        that produced output (either streamed or saved in this call).
        """
        source = Path(source_path)
        saved_paths: dict[str, Path] = {}

        for node in nodes:
            if node.type != "export" or not node.enabled:
                continue

            node_id = str(node.id)
            streamed_path = self._graph_stream_saved_paths.get(node_id)
            if streamed_path is not None:
                saved_paths[node_id] = streamed_path
                self.log_message.emit(f"Write node {node_id}: already streamed -> {streamed_path}")
                continue

            node_output = outputs.get(node.id, {})
            if not isinstance(node_output, dict) or "in" not in node_output:
                logger.info(f"Write node {node_id}: no connected input, skip saving")
                continue

            rendered_frames = self._normalize_node_output_frames(node_output.get("in"))
            if not rendered_frames:
                logger.info(f"Write node {node_id}: empty input, skip saving")
                continue

            source_port = str(node_output.get("__src_port__in", "")).strip().lower()
            source_node_type = str(node_output.get("__src_node_type__in", "")).strip().lower()
            source_node_title = str(node_output.get("__src_node_title__in", "")).strip()
            stream_label = normalize_write_stream_name(source_node_type=source_node_type, source_port=source_port)
            port_label = get_port_output_label(source_node_type, source_port)

            write_cfg = dict(node.properties or {})
            write_cfg["output_dir"] = str(
                resolve_graph_write_output_dir(write_cfg, output_dir, stream_label, source_node_title, port_label)
            )
            is_video = len(rendered_frames) > 1

            saved_path = self._save_stream(
                rendered_frames,
                source,
                is_video,
                fps,
                audio_path,
                write_cfg,
                stream_label=stream_label,
            )

            saved_paths[node_id] = saved_path
            self.log_message.emit(f"Write node {node_id}: saved {stream_label} -> {saved_path}")

        return saved_paths

    def _prepare_graph_write_targets(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        output_dir: Path,
    ) -> None:
        """Create write target folders before processing so they are visible immediately."""
        self._graph_write_plans = {}
        targets = build_graph_write_plan_targets(nodes, edges, output_dir)
        connected_export_ids = {target.node_id for target in targets}
        for node in nodes:
            if node.type == "export" and node.enabled and str(node.id) not in connected_export_ids:
                logger.debug("Skipping unconnected Write node %s during target preparation", node.id)

        for target in targets:
            target_dir = target.target_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            self.log_message.emit(
                self._tr("worker_graph_write_target_prepared").format(
                    node_id=target.node_id,
                    stream=target.stream_label,
                    path=str(target_dir),
                )
            )

            key = (target.source_node_id, target.stream_label)
            self._graph_write_plans.setdefault(key, []).append(
                {
                    "node_id": target.node_id,
                    "stream_label": target.stream_label,
                    "write_cfg": target.write_cfg,
                    "initialized": False,
                    "closed": False,
                }
            )

    @staticmethod
    def _coerce_preview_frame(frame) -> np.ndarray | None:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            gray = arr.astype(np.float32)
            min_val = float(np.nanmin(gray)) if gray.size else 0.0
            max_val = float(np.nanmax(gray)) if gray.size else 0.0
            if gray.dtype != np.uint8:
                if InferenceWorker._is_normalized_float_range(min_val, max_val):
                    gray = np.clip(gray, 0.0, 1.0)
                    gray = np.power(gray, 1.0 / 2.2) * 255.0
                gray = np.clip(gray, 0.0, 255.0).astype(np.uint8)
            else:
                gray = arr
            return np.stack([gray] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[2] == 1:
            return InferenceWorker._coerce_preview_frame(arr[:, :, 0])
        if arr.ndim == 3 and arr.shape[2] >= 3:
            if arr.dtype != np.uint8 and arr.shape[2] >= 4:
                # 4-channel float data = linear premultiplied RGBA (e.g. CorridorKey
                # 'processed' output).  Stripping alpha and gamma-correcting the
                # premultiplied RGB produces false blue/dark on transparent areas
                # (e.g. glass: premul_b > premul_r after gamma).
                # Correct approach: composite over a neutral mid-grey so that
                # transparent areas show as grey, not as dark premultiplied colour.
                rgb_premul = arr[:, :, :3].astype(np.float32)
                alpha_ch = np.clip(arr[:, :, 3:4].astype(np.float32), 0.0, 1.0)
                min_val = float(np.nanmin(rgb_premul)) if rgb_premul.size else 0.0
                max_val = float(np.nanmax(rgb_premul)) if rgb_premul.size else 0.0
                if InferenceWorker._is_normalized_float_range(min_val, max_val):
                    # linear premul + neutral grey bg (linear 0.214 ≈ sRGB 0.5)
                    bg_lin = 0.214
                    comp_lin = np.clip(rgb_premul + bg_lin * (1.0 - alpha_ch), 0.0, 1.0)
                    # linear → sRGB
                    comp_srgb = np.where(comp_lin <= 0.0031308,
                                         comp_lin * 12.92,
                                         1.055 * np.power(np.clip(comp_lin, 1e-9, 1.0), 1.0 / 2.4) - 0.055)
                    return np.clip(comp_srgb * 255.0, 0.0, 255.0).astype(np.uint8)
                else:
                    return np.clip(rgb_premul, 0.0, 255.0).astype(np.uint8)

            rgb = arr[:, :, :3].astype(np.float32)
            if arr.dtype != np.uint8:
                min_val = float(np.nanmin(rgb)) if rgb.size else 0.0
                max_val = float(np.nanmax(rgb)) if rgb.size else 0.0
                if InferenceWorker._is_normalized_float_range(min_val, max_val):
                    rgb = np.clip(rgb, 0.0, 1.0)
                    rgb = np.power(rgb, 1.0 / 2.2) * 255.0
                rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
            else:
                rgb = arr[:, :, :3]
            return rgb
        return None

    @staticmethod
    def _frame_bbox(frame) -> tuple[int, int, int, int]:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            coverage = np.asarray(arr, dtype=np.float32) > 1e-6
        elif arr.ndim == 3 and arr.shape[2] >= 4:
            coverage = np.asarray(arr[:, :, 3], dtype=np.float32) > 1e-6
        elif arr.ndim == 3:
            coverage = np.any(np.asarray(arr[:, :, :3], dtype=np.float32) > 1e-6, axis=2)
        else:
            return (0, 0, 0, 0)

        ys, xs = np.where(coverage)
        if ys.size == 0 or xs.size == 0:
            return (0, 0, 0, 0)
        x0 = int(np.min(xs))
        y0 = int(np.min(ys))
        x1 = int(np.max(xs)) + 1
        y1 = int(np.max(ys)) + 1
        return (x0, y0, x1, y1)

    @staticmethod
    def _bbox_union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if a[2] <= a[0] or a[3] <= a[1]:
            return b
        if b[2] <= b[0] or b[3] <= b[1]:
            return a
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

    @staticmethod
    def _bbox_intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x0 = max(a[0], b[0])
        y0 = max(a[1], b[1])
        x1 = min(a[2], b[2])
        y1 = min(a[3], b[3])
        if x1 <= x0 or y1 <= y0:
            return (0, 0, 0, 0)
        return (x0, y0, x1, y1)

    @staticmethod
    def _clip_frame_to_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        out = np.asarray(frame).copy()
        h, w = out.shape[:2]
        x0 = max(0, min(w, int(bbox[0])))
        y0 = max(0, min(h, int(bbox[1])))
        x1 = max(0, min(w, int(bbox[2])))
        y1 = max(0, min(h, int(bbox[3])))
        if x1 <= x0 or y1 <= y0:
            return np.zeros_like(out)

        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = True
        if out.ndim == 2:
            out[~mask] = 0
        else:
            out[~mask, ...] = 0
        return out

    @staticmethod
    def _to_u8_frame(src: np.ndarray) -> np.ndarray:
        return to_u8_frame(src)

    @staticmethod
    def _to_u16_frame(src: np.ndarray) -> np.ndarray:
        return to_u16_frame(src)

    @staticmethod
    def _promote_alpha_to_rgba_exr(src: np.ndarray) -> np.ndarray:
        return promote_alpha_to_rgba_exr(src)

    def _save_image_frame(
        self,
        frame_arr,
        out_path: Path,
        *,
        output_fmt: str,
        png_compression: int,
        png_bit_depth: int,
        jpg_quality: int,
        embed_alpha: bool = False,
    ) -> None:
        save_image_frame(
            frame_arr,
            out_path,
            output_fmt=output_fmt,
            png_compression=png_compression,
            png_bit_depth=png_bit_depth,
            jpg_quality=jpg_quality,
            embed_alpha=embed_alpha,
        )

    def _initialize_graph_write_plan(self, plan: dict, *, is_video: bool) -> None:
        if plan.get("initialized"):
            return

        source = Path(self._graph_source_path)
        write_cfg = dict(plan.get("write_cfg") or {})
        output_fmt = self._resolve_write_output_format(write_cfg, source)
        stream_label = str(plan.get("stream_label") or "fg")
        out_dir = Path(write_cfg.get("output_dir", "") or str(_build_keyflow_output_dir(source, stream_label)))
        stem = str(write_cfg.get("file_name", "")).strip() or source.stem or "result"
        plan["output_fmt"] = output_fmt
        plan["out_dir"] = out_dir
        plan["stem"] = stem
        plan["is_video"] = bool(is_video)
        plan["png_compression"] = int(write_cfg.get("png_compression", 6))
        plan["png_bit_depth"] = int(write_cfg.get("png_bit_depth", 8))
        plan["jpg_quality"] = int(write_cfg.get("jpg_quality", 90))
        plan["embed_alpha"] = bool(write_cfg.get("png_embed_alpha", False))
        plan["created_paths"] = set()
        plan["video_codec"] = str(write_cfg.get("video_codec", "h264")).strip().lower() or "h264"
        plan["video_quality"] = int(write_cfg.get("video_quality", 23))
        plan["video_preset"] = str(write_cfg.get("video_preset", "medium")).strip().lower() or "medium"
        out_dir.mkdir(parents=True, exist_ok=True)

        video_exts = set(COMPAT_VIDEO_OUTPUT_FORMATS)
        if output_fmt in video_exts:
            import imageio

            video_ext = f".{output_fmt}"
            plan["tmp_path"] = out_dir / f"{stem}_tmp{video_ext}"
            plan["final_path"] = out_dir / f"{stem}{video_ext}"
            plan["created_paths"].add(plan["tmp_path"])
            codec = plan["video_codec"]
            ffmpeg_codec, output_params = build_video_output_params(
                codec,
                crf=int(plan["video_quality"]),
                preset=str(plan["video_preset"]),
            )
            if codec in PRORES_PROFILES:
                writer = imageio.get_writer(
                    str(plan["tmp_path"]),
                    fps=self._graph_fps,
                    codec=ffmpeg_codec,
                    macro_block_size=1,
                    output_params=output_params,
                )
            else:
                writer = imageio.get_writer(
                    str(plan["tmp_path"]),
                    fps=self._graph_fps,
                    codec=ffmpeg_codec,
                    macro_block_size=1,
                    output_params=output_params,
                )
            plan["writer"] = writer
        else:
            ext = ".jpg" if output_fmt in {"jpg", "jpeg"} else f".{output_fmt}"
            plan["img_ext"] = ext
            if is_video:
                plan["first_path"] = out_dir / f"0001{ext}"
            else:
                plan["first_path"] = out_dir / f"{stem}{ext}"

        plan["initialized"] = True

    def _stream_graph_write_frame(self, source_node_id: str, source_port: str, frame, frame_index_0_based: int, *, is_video: bool) -> None:
        lookup_key = (str(source_node_id), str(source_port).strip().lower())
        plans = self._graph_write_plans.get(lookup_key, [])
        if not plans:
            return
        source_port_key = str(source_port).strip().lower()

        # UI slider and preview routing use global frame indices in source timeline.
        # Internal write pipelines may iterate local sliced frames (0..N-1),
        # so we offset emitted preview index by configured graph start frame.
        try:
            frame_index_global = int(frame_index_0_based) + int(getattr(self, "_graph_start_frame", 0) or 0)
        except Exception:
            frame_index_global = int(frame_index_0_based)

        for plan in plans:
            preview = self._coerce_preview_frame(frame)
            self._initialize_graph_write_plan(plan, is_video=is_video)
            output_fmt = str(plan.get("output_fmt") or "png")
            video_exts = set(COMPAT_VIDEO_OUTPUT_FORMATS)
            if output_fmt in video_exts:
                writer = plan.get("writer")
                if writer is not None:
                    frame_u8 = prepare_video_frame(frame, str(plan.get("video_codec") or "h264"))
                    writer.append_data(frame_u8)
                if preview is not None:
                    self.graph_stream_preview.emit(
                        str(plan.get("node_id", "")),
                        make_stream_preview_payload(
                            preview,
                            str(plan.get("final_path") or ""),
                            str(plan.get("stream_label") or ""),
                            RUNTIME_SEMANTICS_PREVIEW_ONLY,
                        ),
                        frame_index_global,
                    )
            else:
                out_dir = Path(plan["out_dir"])
                if is_video:
                    out_path = out_dir / f"{frame_index_0_based:04d}{plan['img_ext']}"
                else:
                    out_path = Path(plan["first_path"])
                self._save_image_frame(
                    frame,
                    out_path,
                    output_fmt=output_fmt,
                    png_compression=int(plan.get("png_compression", 6)),
                    png_bit_depth=int(plan.get("png_bit_depth", 8)),
                    jpg_quality=int(plan.get("jpg_quality", 90)),
                    embed_alpha=bool(plan.get("embed_alpha", False)),
                )
                plan["created_paths"].add(out_path)
                self._graph_stream_saved_paths[str(plan["node_id"])] = Path(plan["first_path"])
                self.graph_stream_preview.emit(
                    str(plan.get("node_id", "")),
                    make_stream_preview_payload(
                        preview,
                        str(out_path),
                        str(plan.get("stream_label") or ""),
                        RUNTIME_SEMANTICS_PREVIEW_ONLY if source_port_key == "comp" else RUNTIME_SEMANTICS_PRODUCTION_SAFE,
                    ),
                    frame_index_global,
                )

    def _finalize_graph_stream_writes(self, *, keep_outputs: bool = True, emit_preview: bool = True) -> None:
        for plans in self._graph_write_plans.values():
            for plan in plans:
                if not plan.get("initialized") or plan.get("closed"):
                    continue
                writer = plan.get("writer")
                if writer is not None:
                    try:
                        writer.close()
                    except Exception as _wclose_exc:
                        logger.warning("Failed to close stream writer: %s", _wclose_exc)
                    tmp_path = plan.get("tmp_path")
                    final_path = plan.get("final_path")
                    if isinstance(tmp_path, Path) and isinstance(final_path, Path) and tmp_path.exists():
                        if keep_outputs:
                            if self._graph_audio_path:
                                muxed = Path(self._mux_audio(str(tmp_path), self._graph_audio_path, final_path))
                                if muxed == tmp_path and tmp_path.exists():
                                    tmp_path.replace(final_path)
                            else:
                                tmp_path.replace(final_path)
                            self._graph_stream_saved_paths[str(plan["node_id"])] = final_path
                            plan.get("created_paths", set()).add(final_path)
                            if emit_preview:
                                self.graph_stream_preview.emit(
                                    str(plan.get("node_id", "")),
                                    make_stream_preview_payload(
                                        None,
                                        str(final_path),
                                        str(plan.get("stream_label") or ""),
                                        RUNTIME_SEMANTICS_PRODUCTION_SAFE,
                                    ),
                                    0,
                                )
                        else:
                            try:
                                tmp_path.unlink(missing_ok=True)
                            except Exception:
                                pass
                if not keep_outputs:
                    for p in list(plan.get("created_paths", set())):
                        try:
                            Path(p).unlink(missing_ok=True)
                        except Exception:
                            pass
                plan["closed"] = True

    @staticmethod
    def _normalize_node_output_frames(data) -> list[np.ndarray]:
        """Normalize graph node output to a frame list accepted by _save_stream."""
        if data is None:
            return []
        if isinstance(data, list):
            return [np.asarray(frame) for frame in data]

        arr = np.asarray(data)
        if arr.ndim == 4:
            return [arr[i] for i in range(arr.shape[0])]
        if arr.ndim == 3:
            # Could be either (H, W, C) single image or (N, H, W) mask sequence.
            if arr.shape[2] in (1, 3, 4):
                return [arr]
            return [arr[i] for i in range(arr.shape[0])]
        if arr.ndim == 2:
            return [arr]
        return []

    @staticmethod
    def _combine_binary_masks(masks: list[np.ndarray]) -> np.ndarray | None:
        if not masks:
            return None
        combined = np.zeros_like(np.asarray(masks[0], dtype=np.uint8))
        for mask in masks:
            arr = np.asarray(mask, dtype=np.uint8)
            if arr.shape != combined.shape:
                continue
            combined = np.where(arr > 127, 255, combined).astype(np.uint8)
        return combined

    def _load_sam_masks_from_payloads(self, properties: dict, ref_shape: tuple[int, int] | None) -> dict[int, np.ndarray]:
        payloads = properties.get("mask_payloads") or []
        if not isinstance(payloads, list):
            return {}

        by_frame: dict[int, np.ndarray] = {}
        for entry in payloads:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path", "")).strip()
            if not path or not Path(path).exists() or ref_shape is None:
                continue
            try:
                frame_idx = int(entry.get("frame_index", 0) or 0)
            except Exception:
                continue
            try:
                mask = self._load_mask(path, ref_shape)
            except Exception:
                continue
            if mask is None:
                continue
            by_frame[frame_idx] = np.where(np.asarray(mask, dtype=np.uint8) > 127, 255, 0).astype(np.uint8)
        return by_frame

    @staticmethod
    def _resolve_frame_mask(mask_map: dict[int, np.ndarray], frame_index: int) -> np.ndarray | None:
        if not mask_map:
            return None
        idx = int(frame_index)
        if idx in mask_map:
            return mask_map[idx]

        lower = [fi for fi in mask_map if fi <= idx]
        if lower:
            return mask_map[max(lower)]

        higher = [fi for fi in mask_map if fi > idx]
        if higher:
            return mask_map[min(higher)]

        return None

    @staticmethod
    def _resolve_sam_frame_path(
        path_map: dict[int, str], fallback_path: str, frame_index: int
    ) -> str | None:
        """Resolve a SAM mask file path for a given global frame index.

        Uses the same proximity strategy as _resolve_frame_mask: exact match
        first, then nearest lower keyframe, then nearest higher keyframe,
        finally the single-frame fallback path.
        """
        if not path_map:
            return fallback_path if fallback_path else None
        idx = int(frame_index)
        if idx in path_map:
            return path_map[idx]
        lower = [fi for fi in path_map if fi <= idx]
        if lower:
            return path_map[max(lower)]
        higher = [fi for fi in path_map if fi > idx]
        if higher:
            return path_map[min(higher)]
        return fallback_path if fallback_path else None

    # ------------------------------------------------------------------
    # Загрузка
    # ------------------------------------------------------------------
    def _load_video(self, video_path, output_dir):
        """Загружает кадры, fps и извлекает аудио."""
        if is_numbered_image_sequence(video_path):
            frames = load_image_sequence(video_path)
            if not frames:
                raise RuntimeError(self._tr("worker_video_frames_failed"))
            return frames, DEFAULT_FPS, ""

        # Preset data may have stale media_type="video" for still images (e.g. EXR).
        # Treat supported image files as single-frame sources instead of failing in VideoCapture.
        # Use load_image_float so alpha channel is preserved (needed for EXR fg sources in Merge).
        if is_supported_image_file(video_path):
            frame = load_image_float(video_path)
            return [frame], DEFAULT_FPS, ""

        frames = []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"{self._tr('worker_open_video_failed')} {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if self.cancel_flag.is_set():
                break
        cap.release()

        if not frames:
            raise RuntimeError(self._tr("worker_video_frames_failed"))

        # Попытка извлечь аудио через ffmpeg
        audio_path = ""
        try:
            import subprocess
            from app.utils.ffmpeg import get_ffmpeg_exe
            audio_tmp = str(output_dir / "_audio.wav")
            subprocess.run(
                [get_ffmpeg_exe(), "-y", "-i", video_path, "-vn",
                 "-acodec", "pcm_s16le", "-ac", "2", "-ar", "44100", audio_tmp],
                check=True, capture_output=True
            )
            audio_path = audio_tmp
        except Exception:
            pass  # видео без аудио или ffmpeg недоступен

        return frames, fps, audio_path

    def _load_image_frame(self, image_path):
        return load_rgb_image(image_path)

    def _load_mask(self, mask_path, target_shape):
        with Image.open(mask_path) as _mask_raw:
            mask_img = _mask_raw.convert("L")
            mask = np.array(mask_img)
        if mask.shape != target_shape:
            mask = cv2.resize(
                mask, (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        return mask

    @staticmethod
    def _coerce_matting_mask(mask_data, target_shape: tuple[int, int]) -> np.ndarray | None:
        """Coerce graph-provided matting mask to 2D uint8 [0..255]."""
        if mask_data is None:
            return None

        candidate = mask_data
        if isinstance(candidate, list):
            if not candidate:
                return None
            candidate = candidate[0]

        arr = np.asarray(candidate)

        if arr.ndim == 4:
            if arr.shape[0] == 0:
                return None
            arr = arr[0]

        if arr.ndim == 3:
            if arr.shape[0] == 0:
                return None
            h, w = target_shape
            is_color_image = arr.shape[0] == h and arr.shape[1] == w and arr.shape[2] in (1, 3, 4)
            if is_color_image and arr.shape[2] == 1:
                arr = arr[:, :, 0]
            elif is_color_image and arr.shape[2] in (3, 4):
                arr = np.asarray(arr[:, :, :3], dtype=np.float32).mean(axis=2)
            else:
                # Sequence-like (N, H, W): use first frame as base trimap/mask.
                arr = arr[0]

        if arr.ndim != 2:
            raise ValueError("MatAnyone2 node received unsupported mask shape")

        if arr.shape != target_shape:
            arr = cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

        arr_f = np.asarray(arr, dtype=np.float32)
        if arr_f.size > 0 and float(arr_f.max()) <= 1.0:
            arr_f *= 255.0
        return np.clip(arr_f, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _coerce_merge_mask(mask_data, target_shape: tuple[int, int], channel: str = "auto") -> np.ndarray | None:
        """Coerce merge mask to 2D uint8 [0..255] with explicit channel selection."""
        if mask_data is None:
            return None

        candidate = mask_data
        if isinstance(candidate, list):
            if not candidate:
                return None
            candidate = candidate[0]

        arr = np.asarray(candidate)
        if arr.ndim == 4:
            if arr.shape[0] == 0:
                return None
            arr = arr[0]

        channel = str(channel or "auto").strip().lower()
        h, w = target_shape
        if arr.ndim == 3:
            is_color_image = arr.shape[0] == h and arr.shape[1] == w and arr.shape[2] in (1, 3, 4)
            if not is_color_image:
                arr = arr[0]
            elif arr.shape[2] == 1:
                arr = arr[:, :, 0]
            else:
                arr_f = np.asarray(arr, dtype=np.float32)
                if channel == "red":
                    arr = arr_f[:, :, 0]
                elif channel == "green":
                    arr = arr_f[:, :, 1]
                elif channel == "blue":
                    arr = arr_f[:, :, 2]
                elif channel == "alpha":
                    arr = arr_f[:, :, 3] if arr.shape[2] >= 4 else arr_f[:, :, :3].mean(axis=2)
                elif channel == "luma":
                    rgb = arr_f[:, :, :3]
                    arr = rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722
                else:  # auto
                    if arr.shape[2] >= 4:
                        arr = arr_f[:, :, 3]
                    else:
                        arr = arr_f[:, :, :3].mean(axis=2)

        if arr.ndim != 2:
            raise ValueError("Merge node received unsupported mask shape")

        if arr.shape != target_shape:
            arr = cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

        arr_f = np.asarray(arr, dtype=np.float32)
        if arr_f.size > 0 and float(arr_f.max()) <= 1.0:
            arr_f *= 255.0
        return np.clip(arr_f, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _resolve_write_output_format(write_config: dict | None, source: Path) -> str:
        return resolve_write_output_format(write_config, source)

    def _save_stream(
        self,
        rendered_frames: list,
        source: Path,
        is_video: bool,
        fps: float,
        audio_path: str,
        write_config: dict,
        stream_label: str,
    ) -> Path:
        """Save one graph stream (fg/alpha/other) according to Write-node config."""
        _VIDEO_EXTS = set(COMPAT_VIDEO_OUTPUT_FORMATS)
        _IMAGE_EXTS = set(COMPAT_IMAGE_OUTPUT_FORMATS)

        output_fmt: str = str(write_config.get("output_format", "source")).strip().lower()
        out_dir = Path(write_config.get("output_dir", "") or str(_build_keyflow_output_dir(source, stream_label)))
        file_name: str = str(write_config.get("file_name", "")).strip()
        stem = file_name if file_name else source.stem or "result"

        video_codec: str = str(write_config.get("video_codec", "h264")).strip().lower() or "h264"
        video_quality: int = int(write_config.get("video_quality", 23))
        video_preset: str = str(write_config.get("video_preset", "medium")).strip().lower() or "medium"
        png_compression: int = int(write_config.get("png_compression", 6))
        png_bit_depth: int = int(write_config.get("png_bit_depth", 8))
        jpg_quality: int = int(write_config.get("jpg_quality", 90))
        embed_alpha: bool = bool(write_config.get("png_embed_alpha", False))

        out_dir.mkdir(parents=True, exist_ok=True)

        if output_fmt == "source":
            src_ext = source.suffix.lower().lstrip(".")
            if src_ext in _VIDEO_EXTS:
                output_fmt = src_ext
            elif src_ext in _IMAGE_EXTS:
                output_fmt = src_ext
            else:
                output_fmt = "png"
        elif output_fmt not in _VIDEO_EXTS and output_fmt not in _IMAGE_EXTS:
            output_fmt = "png"

        if output_fmt in _VIDEO_EXTS:
            video_ext = f".{output_fmt}"
            out_path = out_dir / f"{stem}{video_ext}"
            tmp_path = out_dir / f"{stem}_tmp{video_ext}"
            self._write_video(
                rendered_frames,
                str(tmp_path),
                fps,
                codec=video_codec,
                crf=video_quality,
                preset=video_preset,
            )
            if audio_path:
                muxed = Path(self._mux_audio(str(tmp_path), audio_path, out_path))
                if muxed == tmp_path:
                    tmp_path.replace(out_path)
            else:
                tmp_path.replace(out_path)
            return out_path

        img_ext = ".jpg" if output_fmt in {"jpg", "jpeg"} else f".{output_fmt}"

        def _save_img(frame_arr, out_path: Path) -> None:
            save_image_frame(
                frame_arr,
                out_path,
                output_fmt=output_fmt,
                png_compression=png_compression,
                png_bit_depth=png_bit_depth,
                jpg_quality=jpg_quality,
                embed_alpha=embed_alpha,
            )

        if is_video and len(rendered_frames) > 1:
            first_path = out_dir / f"0000{img_ext}"
            for i, frame in enumerate(rendered_frames):
                _save_img(frame, out_dir / f"{i:04d}{img_ext}")
            return first_path

        out_path = out_dir / f"{stem}{img_ext}"
        _save_img(rendered_frames[0], out_path)
        return out_path

    # ------------------------------------------------------------------
    # Выполнение нод графа
    # ------------------------------------------------------------------
    def _execute_node(self, node_type: str, node_data: dict, inputs: dict) -> dict:
        """Выполнить одну ноду графа по типу.
        
        Args:
            node_type: Тип ноды (e.g., 'birefnet', 'corridorkey')
            node_data: Данные ноды (properties, ports)
            inputs: Входные данные {port_name: data}
        
        Returns:
            dict: Выходные данные {port_name: data}
        """
        if self.cancel_flag.is_set():
            return {}
        
        try:
            if node_type == "birefnet":
                return self._execute_birefnet_node(node_data, inputs)
            elif node_type == "gvm":
                return self._execute_gvm_node(node_data, inputs)
            elif node_type == "chromakey":
                return self._execute_chromakey_node(node_data, inputs)
            elif node_type == "corridorkey":
                return self._execute_corridorkey_node(node_data, inputs)
            elif node_type == "matting":
                # Обработка через существующий InferenceService
                return self._execute_matanyone2_node(node_data, inputs)
            elif node_type in {"sam2"}:
                properties = node_data.get("properties", {})
                img_frames = inputs.get("img") or inputs.get("image") or inputs.get("out")
                ref_shape = img_frames[0].shape[:2] if img_frames is not None and len(img_frames) > 0 else None

                by_frame_masks = self._load_sam_masks_from_payloads(properties, ref_shape)

                mask_path = (
                    str(properties.get("_mask_source_path", "")).strip()
                    or getattr(self, "_graph_mask_path", "")
                )
                fallback_mask = None
                if mask_path and Path(mask_path).exists():
                    loaded_mask = self._load_mask(mask_path, ref_shape) if ref_shape else None
                    if loaded_mask is not None:
                        fallback_mask = np.where(np.asarray(loaded_mask, dtype=np.uint8) > 127, 255, 0).astype(np.uint8)

                sam_masks_sequence: list[np.ndarray] = []
                if img_frames is not None and len(img_frames) > 0 and (by_frame_masks or fallback_mask is not None):
                    node_id_str = str(node_data.get("id", ""))
                    is_video = len(img_frames) > 1
                    start_frame = int(getattr(self, "_graph_start_frame", 0) or 0)
                    total_sam_frames = len(img_frames)

                    for local_idx, _frame in enumerate(img_frames):
                        global_idx = start_frame + int(local_idx)
                        mask_for_frame = self._resolve_frame_mask(by_frame_masks, global_idx)
                        if mask_for_frame is None:
                            mask_for_frame = fallback_mask
                        if mask_for_frame is None:
                            continue
                        sam_masks_sequence.append(np.asarray(mask_for_frame, dtype=np.uint8).copy())
                        self._stream_graph_write_frame(
                            node_id_str,
                            "out",
                            mask_for_frame,
                            local_idx,
                            is_video=is_video,
                        )
                        self.node_frame_progress.emit("sam2", local_idx + 1, total_sam_frames)

                if sam_masks_sequence:
                    return {**inputs, "out": sam_masks_sequence, "mask": sam_masks_sequence}
                if fallback_mask is not None:
                    return {**inputs, "out": fallback_mask, "mask": fallback_mask}
                return inputs
            elif node_type == "sam3":
                properties = node_data.get("properties", {})
                img_frames = inputs.get("img") or inputs.get("image") or inputs.get("out")
                if img_frames is None or len(img_frames) == 0:
                    raise ValueError("SAM3 node requires 'img' input")

                ref_shape = img_frames[0].shape[:2]
                by_frame_masks = self._load_sam_masks_from_payloads(properties, ref_shape)

                mask_path = (
                    str(properties.get("_mask_source_path", "")).strip()
                    or getattr(self, "_graph_mask_path", "")
                )
                fallback_mask = None
                if mask_path and Path(mask_path).exists():
                    loaded_mask = self._load_mask(mask_path, ref_shape)
                    if loaded_mask is not None:
                        fallback_mask = np.where(np.asarray(loaded_mask, dtype=np.uint8) > 127, 255, 0).astype(np.uint8)

                node_id_str = str(node_data.get("id", ""))
                is_video = len(img_frames) > 1
                start_frame = int(getattr(self, "_graph_start_frame", 0) or 0)
                total_sam_frames = len(img_frames)

                sam_masks_sequence: list[np.ndarray] = []
                if by_frame_masks or fallback_mask is not None:
                    for local_idx, _frame in enumerate(img_frames):
                        global_idx = start_frame + int(local_idx)
                        mask_for_frame = self._resolve_frame_mask(by_frame_masks, global_idx)
                        if mask_for_frame is None:
                            mask_for_frame = fallback_mask
                        if mask_for_frame is None:
                            continue
                        mask_u8 = np.asarray(mask_for_frame, dtype=np.uint8).copy()
                        sam_masks_sequence.append(mask_u8)
                        self._stream_graph_write_frame(
                            node_id_str,
                            "out",
                            mask_u8,
                            local_idx,
                            is_video=is_video,
                        )
                        self.node_frame_progress.emit("sam3", local_idx + 1, total_sam_frames)

                    if sam_masks_sequence:
                        return {**inputs, "out": sam_masks_sequence, "mask": sam_masks_sequence}
                    if fallback_mask is not None:
                        return {**inputs, "out": fallback_mask, "mask": fallback_mask}

                # No persisted masks: run SAM3 image inference directly from concept prompt.
                model_type = str(properties.get("model_type", "sam3") or "sam3")
                concept = str(properties.get("concept", "") or "").strip()
                if not concept:
                    raise ValueError(
                        "SAM3 node has no prompt. Enter a concept prompt before running the graph."
                    )

                sam3_runtime_notice_emitted = False

                for local_idx, frame in enumerate(img_frames):
                    masks = Sam3Service.predict_image(
                        model_type=model_type,
                        image=frame,
                        points=[],
                        concept=concept,
                    )
                    if not sam3_runtime_notice_emitted:
                        runtime_notice = Sam3Service.consume_runtime_notice()
                        if runtime_notice:
                            frame_ratio = float(local_idx + 1) / float(max(1, total_sam_frames))
                            stage_percent = 20 + int(max(0.0, min(1.0, frame_ratio)) * 72)
                            status_text = self._tr("worker_sam3_runtime_cpu_fallback").format(
                                reason=runtime_notice
                            )
                            self.stage_progress.emit(max(20, min(92, stage_percent)), status_text)
                            self.log_message.emit(status_text)
                            sam3_runtime_notice_emitted = True
                    if not masks:
                        continue

                    mask_u8 = self._combine_binary_masks([np.asarray(mask, dtype=np.uint8) for mask in masks])
                    if mask_u8 is None:
                        continue
                    sam_masks_sequence.append(mask_u8)

                    self._stream_graph_write_frame(
                        node_id_str,
                        "out",
                        mask_u8,
                        local_idx,
                        is_video=is_video,
                    )
                    self.node_frame_progress.emit("sam3", local_idx + 1, total_sam_frames)

                if sam_masks_sequence:
                    return {**inputs, "out": sam_masks_sequence, "mask": sam_masks_sequence}
                raise RuntimeError("SAM3 node produced no masks")
            elif node_type == "merge":
                return self._execute_merge_node(node_data, inputs)
            elif node_type in {"source", "load", "alpha"}:
                # Read узел не нуждается в обработке здесь (уже загружены)
                return inputs
            elif node_type == "export":
                # Write узел обрабатывается отдельно
                return inputs
            else:
                logger.warning(f"Unknown node type: {node_type}")
                return {}
        except Exception as e:
            logger.error(f"Error executing node {node_type}: {e}")
            raise

    def _execute_merge_node(self, node_data: dict, inputs: dict) -> dict:
        """Composite two image streams using Merge operations with mask/mix/bbox controls.

        Input ports (resolved frame lists from upstream):
            fg  — foreground / source layer
            bg  — background / destination layer

        Output ports:
            out — composited RGBA frames (float32 [0..1])
        """
        fg_raw = inputs.get("fg")
        bg_raw = inputs.get("bg")
        mask_raw = inputs.get("mask")

        if fg_raw is None:
            raise ValueError("Merge: 'fg' input required")
        if bg_raw is None:
            raise ValueError("Merge: 'bg' input required")

        # Normalise to list of frames
        fg_frames = fg_raw if isinstance(fg_raw, list) else [fg_raw]
        bg_frames = bg_raw if isinstance(bg_raw, list) else [bg_raw]

        props = node_data.get("properties", {})
        mode = str(props.get("mode", "over")).strip().lower()
        opacity = float(props.get("opacity", 1.0))
        mix = float(props.get("mix", 1.0))
        mask_enabled = bool(props.get("mask_enabled", True))
        mask_channel = str(props.get("mask_channel", "auto")).strip().lower()
        mask_inject = bool(props.get("mask_inject", False))
        invert_mask = bool(props.get("invert_mask", False))
        fringe = bool(props.get("fringe", False))
        alpha_masking = bool(props.get("alpha_masking", True))
        set_bbox_to = str(props.get("set_bbox_to", "union")).strip().lower()
        if set_bbox_to not in {"union", "intersection", "a", "b"}:
            set_bbox_to = "union"

        fg_meta = inputs.get("__meta__fg")
        bg_meta = inputs.get("__meta__bg")

        def _bbox_from_meta(meta, frame, index: int) -> tuple[int, int, int, int]:
            if isinstance(meta, dict):
                seq = meta.get("bbox_sequence")
                if isinstance(seq, list) and seq:
                    raw = seq[min(index, len(seq) - 1)]
                    if isinstance(raw, (list, tuple)) and len(raw) == 4:
                        return (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
            return self._frame_bbox(frame)

        n_frames = max(len(fg_frames), len(bg_frames))
        node_id_str = str(node_data.get("id", ""))
        is_video = n_frames > 1

        self.log_message.emit(
            f"Merge: {n_frames} frame(s), mode={mode}, opacity={opacity:.2f}"
        )

        mask_frames = None
        if mask_raw is not None:
            mask_frames = mask_raw if isinstance(mask_raw, list) else [mask_raw]

        result_frames = []
        out_bbox_sequence: list[tuple[int, int, int, int]] = []
        for i in range(n_frames):
            if self.cancel_flag.is_set():
                break

            fg = fg_frames[min(i, len(fg_frames) - 1)]
            bg = bg_frames[min(i, len(bg_frames) - 1)]
            mask = None if not mask_frames else mask_frames[min(i, len(mask_frames) - 1)]

            bbox_a = _bbox_from_meta(fg_meta, fg, i)
            bbox_b = _bbox_from_meta(bg_meta, bg, i)

            out = self._apply_merge_blend(
                fg,
                bg,
                mode=mode,
                opacity=opacity,
                mask=mask,
                mix=mix,
                mask_enabled=mask_enabled,
                mask_channel=mask_channel,
                mask_inject=mask_inject,
                invert_mask=invert_mask,
                fringe=fringe,
                alpha_masking=alpha_masking,
            )

            if set_bbox_to == "intersection":
                out_bbox = self._bbox_intersection(bbox_a, bbox_b)
            elif set_bbox_to == "a":
                out_bbox = bbox_a
            elif set_bbox_to == "b":
                out_bbox = bbox_b
            else:
                out_bbox = self._bbox_union(bbox_a, bbox_b)

            out = self._clip_frame_to_bbox(out, out_bbox)
            result_frames.append(out)
            out_bbox_sequence.append(out_bbox)

            self._stream_graph_write_frame(node_id_str, "out", out, i, is_video=is_video)

            # Stream preview directly to viewer when Merge node is selected
            preview = self._coerce_preview_frame(out)
            if preview is not None:
                preview_frame_index = i + int(getattr(self, "_graph_start_frame", 0) or 0)
                self.graph_stream_preview.emit(
                    node_id_str,
                    {
                        "frame": preview,
                        "path": "",
                        "stream": "out",
                    },
                    preview_frame_index,
                )

            self.node_frame_progress.emit("merge", i + 1, n_frames)

            stage_text = self._tr("worker_merge_processing").format(current=i + 1, total=n_frames)
            stage_pct = 20 + int((i + 1) / n_frames * 70)
            self.stage_progress.emit(min(90, stage_pct), stage_text)

        return {
            "out": result_frames,
            "__meta__": {
                "out": {
                    "bbox_sequence": out_bbox_sequence,
                }
            },
        }

    @staticmethod
    def _apply_merge_blend(
        fg: np.ndarray,
        bg: np.ndarray,
        *,
        mode: str,
        opacity: float,
        mask=None,
        mix: float = 1.0,
        mask_enabled: bool = True,
        mask_channel: str = "auto",
        mask_inject: bool = False,
        invert_mask: bool = False,
        fringe: bool = False,
        alpha_masking: bool = True,
    ) -> np.ndarray:
        """Return RGBA float32 [0..1] composite of fg (A) over bg (B) using the
        requested Merge operation.

        Supports 32 operations matching NUKE's Merge node:
          Porter-Duff:  over, under, atop, in, out, mask, stencil, matte, xor,
                        copy, conjoint-over, disjoint-over
          Arithmetic:   plus, minus, from, multiply, divide, average, hypot
          Blend (W3C):  screen, overlay, hard-light, soft-light, difference,
                        exclusion, min (darken), max (lighten), color-burn,
                        color-dodge, reflect, geometric, pinlight

        Legacy aliases handled transparently:
          add → plus  |  subtract → minus  |  darken → min  |  lighten → max

        References:
          NUKE Reference Guide — Merge node
          W3C Compositing and Blending Level 1  https://www.w3.org/TR/compositing-1/
          Porter & Duff (1984) — Compositing Digital Images
        """
        # --- legacy alias normalisation ------------------------------------
        _ALIASES = {"add": "plus", "subtract": "minus", "darken": "min", "lighten": "max"}
        mode = _ALIASES.get(mode, mode)

        # --- helpers -------------------------------------------------------
        def _to_float(arr: np.ndarray) -> np.ndarray:
            a = np.asarray(arr, dtype=np.float32)
            if a.size and float(np.nanmax(a)) > 1.5:
                a = a / 255.0
            return np.clip(a, 0.0, 1.0)

        def _to_rgba(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Return (rgb H×W×3, alpha H×W) both float32 [0..1]."""
            if arr.ndim == 2:
                rgb = np.stack([arr, arr, arr], axis=-1)
                alpha = np.ones(arr.shape[:2], dtype=np.float32)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                rgb = np.repeat(arr, 3, axis=2)
                alpha = np.ones(arr.shape[:2], dtype=np.float32)
            elif arr.ndim == 3 and arr.shape[2] == 3:
                rgb = arr
                alpha = np.ones(arr.shape[:2], dtype=np.float32)
            else:  # ≥4 channels
                rgb = arr[:, :, :3]
                alpha = arr[:, :, 3]
            return np.clip(rgb, 0.0, 1.0), np.clip(alpha, 0.0, 1.0)

        # --- prepare inputs ------------------------------------------------
        fg_arr = _to_float(fg)
        bg_arr = _to_float(bg)

        if fg_arr.shape[:2] != bg_arr.shape[:2]:
            import cv2 as _cv2
            h, w = fg_arr.shape[:2]
            bg_arr = _cv2.resize(bg_arr, (w, h), interpolation=_cv2.INTER_LINEAR).astype(np.float32)

        A, fa = _to_rgba(fg_arr)
        B, fb = _to_rgba(bg_arr)

        fa = np.clip(fa * max(0.0, min(1.0, float(opacity))), 0.0, 1.0)

        fa3 = fa[:, :, np.newaxis]
        fb3 = fb[:, :, np.newaxis]
        A_pre = A * fa3
        B_pre = B * fb3
        EPS = 1e-6

        def _blend_values(src, dst, blend_mode: str):
            if blend_mode == "minus":
                return np.clip(src - dst, 0.0, 1.0)
            if blend_mode == "from":
                return np.clip(dst - src, 0.0, 1.0)
            if blend_mode == "multiply":
                return src * dst
            if blend_mode == "divide":
                return np.clip(src / np.maximum(dst, EPS), 0.0, 1.0)
            if blend_mode == "screen":
                return 1.0 - (1.0 - src) * (1.0 - dst)
            if blend_mode == "overlay":
                return np.where(dst < 0.5, 2.0 * src * dst, 1.0 - 2.0 * (1.0 - src) * (1.0 - dst))
            if blend_mode == "hard-light":
                return np.where(src < 0.5, 2.0 * src * dst, 1.0 - 2.0 * (1.0 - src) * (1.0 - dst))
            if blend_mode == "soft-light":
                D = np.where(dst >= 0.25, np.sqrt(np.maximum(dst, 0.0)), ((16.0 * dst - 12.0) * dst + 4.0) * dst)
                return np.where(src <= 0.5, dst - (1.0 - 2.0 * src) * dst * (1.0 - dst), dst + (2.0 * src - 1.0) * (D - dst))
            if blend_mode == "average":
                return (src + dst) * 0.5
            if blend_mode == "difference":
                return np.abs(src - dst)
            if blend_mode == "exclusion":
                return src + dst - 2.0 * src * dst
            if blend_mode == "min":
                return np.minimum(src, dst)
            if blend_mode == "max":
                return np.maximum(src, dst)
            if blend_mode == "color-burn":
                return np.clip(1.0 - (1.0 - dst) / np.maximum(src, EPS), 0.0, 1.0)
            if blend_mode == "color-dodge":
                return np.clip(dst / np.maximum(1.0 - src, EPS), 0.0, 1.0)
            if blend_mode == "reflect":
                return np.clip(src * src / np.maximum(1.0 - dst, EPS), 0.0, 1.0)
            if blend_mode == "geometric":
                return np.clip(2.0 * src * dst / np.maximum(src + dst, EPS), 0.0, 1.0)
            if blend_mode == "pinlight":
                return np.clip(np.minimum(2.0 * src, np.maximum(2.0 * src - 1.0, dst)), 0.0, 1.0)
            return src

        # Porter-Duff compositing operators
        if mode == "over":
            out_pre = A_pre + B_pre * (1.0 - fa3)
            out_a = fa + fb * (1.0 - fa)

        elif mode == "under":
            out_pre = A_pre * (1.0 - fb3) + B_pre
            out_a = fa * (1.0 - fb) + fb

        elif mode == "atop":
            out_pre = A_pre * fb3 + B_pre * (1.0 - fa3)
            out_a = fb

        elif mode == "in":
            out_pre = A_pre * fb3
            out_a = fa * fb

        elif mode == "out":
            out_pre = A_pre * (1.0 - fb3)
            out_a = fa * (1.0 - fb)

        elif mode == "mask":
            out_pre = B_pre * fa3
            out_a = fb * fa

        elif mode == "stencil":
            out_pre = B_pre * (1.0 - fa3)
            out_a = fb * (1.0 - fa)

        elif mode == "matte":
            out_pre = A * fa3 + B_pre * (1.0 - fa3)
            out_a = fa + fb * (1.0 - fa)

        elif mode == "xor":
            out_pre = A_pre * (1.0 - fb3) + B_pre * (1.0 - fa3)
            out_a = fa * (1.0 - fb) + fb * (1.0 - fa)

        elif mode == "copy":
            out_pre = A_pre
            out_a = fa

        elif mode == "conjoint-over":
            safe_fb3 = np.maximum(fb3, EPS)
            normal = A_pre + B_pre * (1.0 - fa3 / safe_fb3)
            out_pre = np.where(fa3 >= fb3, A_pre, normal)
            out_a = np.maximum(fa, fb)

        elif mode == "disjoint-over":
            safe_fb3 = np.maximum(fb3, EPS)
            normal = A_pre + B_pre * (1.0 - fa3 / safe_fb3)
            full = A_pre + B_pre
            out_pre = np.where((fa3 + fb3) <= 1.0, normal, full)
            out_a = np.minimum(fa + fb, 1.0)

        elif mode == "plus":
            out_pre = np.clip(A_pre + B_pre, 0.0, 1.0)
            out_a = np.minimum(fa + fb, 1.0)

        elif mode == "hypot":
            out_pre = np.clip(np.sqrt(A_pre ** 2 + B_pre ** 2), 0.0, 1.0)
            out_a = np.minimum(fa + fb, 1.0)

        else:
            blended = _blend_values(A, B, mode)
            effective_A = (1.0 - fb3) * A + fb3 * blended
            out_pre = effective_A * fa3 + B_pre * (1.0 - fa3)
            out_a = fa + fb * (1.0 - fa) if alpha_masking else np.clip(_blend_values(fa, fb, mode), 0.0, 1.0)

        # --- convert premultiplied result back to unpremultiplied ----------
        mix = max(0.0, min(1.0, float(mix)))
        mask_u8 = InferenceWorker._coerce_merge_mask(mask, A.shape[:2], channel=mask_channel) if (mask is not None and mask_enabled) else None
        inject_mask_f = None
        if mask_u8 is not None:
            inject_mask_f = np.asarray(mask_u8, dtype=np.float32) / 255.0
            if invert_mask:
                mask_u8 = 255 - mask_u8
            if fringe:
                kernel = np.ones((3, 3), dtype=np.uint8)
                edge_mask = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)
                if int(np.max(edge_mask)) > 0:
                    mask_u8 = edge_mask
            mask_f = np.asarray(mask_u8, dtype=np.float32) / 255.0
        else:
            mask_f = np.ones(A.shape[:2], dtype=np.float32)

        effect_f = np.clip(mask_f * mix, 0.0, 1.0)
        effect3 = effect_f[:, :, np.newaxis]
        out_pre = B_pre * (1.0 - effect3) + out_pre * effect3
        out_a = fb * (1.0 - effect_f) + out_a * effect_f

        if mask_inject and inject_mask_f is not None:
            out_a = np.clip(inject_mask_f, 0.0, 1.0)

        out_a   = np.clip(out_a, 0.0, 1.0)
        out_pre = np.clip(out_pre, 0.0, 1.0)
        # conjoint/disjoint can produce (H,W,1) out_a — squeeze if needed
        if out_a.ndim == 3:
            out_a = out_a[:, :, 0]
        out_a_safe = np.where(out_a > EPS, out_a, 1.0)
        out_rgb    = np.clip(out_pre / out_a_safe[:, :, np.newaxis], 0.0, 1.0)

        return np.concatenate(
            [out_rgb, out_a[:, :, np.newaxis]], axis=2
        ).astype(np.float32)

    def _execute_chromakey_node(self, node_data: dict, inputs: dict) -> dict:
        """Выполнить HSV Chroma Key узел — чисто OpenCV, без нейросети.

        Input:  {image: RGB_frames}
        Output: {mask: foreground_mask_frames}  (float32 0-1, 1=foreground)
        """
        if "image" not in inputs:
            raise ValueError("HSV Chroma Key node requires 'image' input")

        frames = inputs["image"]
        props = node_data.get("properties", {})
        hue_center = int(props.get("hue_center", 120))
        hue_range = int(props.get("hue_range", 30))
        sat_min = float(props.get("saturation_min", 0.15))
        val_min = float(props.get("value_min", 0.10))
        blur_radius = int(props.get("blur_radius", 3))

        self.log_message.emit(
            f"HSV Chroma Key: {len(frames)} frame(s), "
            f"hue={hue_center}\u00b1{hue_range}\u00b0, "
            f"sat_min={sat_min:.2f}, val_min={val_min:.2f}, blur={blur_radius}"
        )

        masks = []
        for i, frame in enumerate(frames):
            if self.cancel_flag.is_set():
                break
            mask = self._apply_hsv_chromakey(frame, hue_center, hue_range, sat_min, val_min, blur_radius)
            masks.append(mask)

            progress = int((i + 1) / len(frames) * 100)
            stage_text = self._tr("worker_chromakey_processing").format(
                current=i + 1, total=len(frames)
            )
            self.stage_progress.emit(progress, stage_text)
            self.node_frame_progress.emit("chromakey", i + 1, len(frames))

        return {"mask": np.array(masks) if masks else np.array([])}

    @staticmethod
    def _apply_hsv_chromakey(
        frame: np.ndarray,
        hue_center: int,
        hue_range: int,
        saturation_min: float,
        value_min: float,
        blur_radius: int,
    ) -> np.ndarray:
        """Return foreground mask (float32 0-1) via HSV threshold.

        1 = foreground (not chroma), 0 = chroma screen.
        """
        # Normalize to uint8 if needed
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            img_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        else:
            img_u8 = frame.astype(np.uint8)

        # RGB → BGR → HSV  (OpenCV uses BGR by default)
        bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # OpenCV HSV: H ∈ [0, 179], S ∈ [0, 255], V ∈ [0, 255]
        h_c = int(hue_center / 2)            # 360° → 179°
        h_r = max(1, int(hue_range / 2))
        s_lo = int(saturation_min * 255)
        v_lo = int(value_min * 255)

        h_lo = h_c - h_r
        h_hi = h_c + h_r

        if h_lo < 0:
            # Wrap around lower boundary
            m1 = cv2.inRange(hsv, np.array([0, s_lo, v_lo]), np.array([h_hi, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([h_lo + 180, s_lo, v_lo]), np.array([179, 255, 255]))
            chroma_mask = cv2.bitwise_or(m1, m2)
        elif h_hi > 179:
            # Wrap around upper boundary
            m1 = cv2.inRange(hsv, np.array([h_lo, s_lo, v_lo]), np.array([179, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([0, s_lo, v_lo]), np.array([h_hi - 180, 255, 255]))
            chroma_mask = cv2.bitwise_or(m1, m2)
        else:
            chroma_mask = cv2.inRange(
                hsv, np.array([h_lo, s_lo, v_lo]), np.array([h_hi, 255, 255])
            )

        # Invert: chroma=0, foreground=255
        fg_mask = cv2.bitwise_not(chroma_mask)

        # Optional Gaussian blur for soft edges
        if blur_radius > 0:
            k = blur_radius * 2 + 1
            fg_f = cv2.GaussianBlur(fg_mask.astype(np.float32), (k, k), 0)
        else:
            fg_f = fg_mask.astype(np.float32)

        return fg_f / 255.0

    def _execute_birefnet_node(self, node_data: dict, inputs: dict) -> dict:
        """Выполнить BiRefNet узел для генерации альфа-маски.
        
        Input: {image: RGB_frames}
        Output: {alpha: alpha_frames}
        """
        if "image" not in inputs:
            raise ValueError("BiRefNet node requires 'image' input")
        
        frames = inputs["image"]
        properties = node_data.get("properties", {})
        usage = properties.get("usage", "General")
        half_precision = properties.get("half_precision", False)
        dilate_radius = max(0, int(properties.get("dilate_radius", 0) or 0))
        erode_radius = max(0, int(properties.get("erode_radius", 0) or 0))

        self.birefnet_service.set_callbacks(
            progress_callback=lambda percent, msg: self.stage_progress.emit(percent, msg),
            translate=self._tr,
        )
        self.birefnet_service.load_model(usage=usage)
        
        self.log_message.emit(
            self._tr("worker_birefnet_processing_start").format(total=len(frames), preset=usage)
        )
        self.log_message.emit(
            f"BiRefNet: usage={usage}, half_precision={bool(half_precision)}, "
            f"dilate={dilate_radius}, erode={erode_radius}"
        )
        
        alphas = []
        node_id = str(node_data.get("id", ""))
        downstream_targets = [
            t for t in self._graph_downstream_targets.get((node_id, "alpha"), []) if bool(t.get("dst_enabled", True))
        ]
        has_non_write_consumers = any(str(t.get("dst_type", "")) != "export" for t in downstream_targets)

        disk_sequence_paths: list[str] = []
        disk_sequence_dir: Path | None = None
        if has_non_write_consumers:
            root_dir = self._graph_output_dir if self._graph_output_dir is not None else Path(tempfile.mkdtemp(prefix="keyflow_birefnet_"))
            disk_sequence_dir = Path(root_dir) / "_keyflow_birefnet_alpha" / f"{node_id or 'birefnet'}"
            disk_sequence_dir.mkdir(parents=True, exist_ok=True)
            for stale in disk_sequence_dir.glob("*.png"):
                stale.unlink(missing_ok=True)
            self.log_message.emit(
                f"BiRefNet: disk-first mode enabled ({len(downstream_targets)} downstream consumer(s), including non-write)"
            )

        self.log_message.emit(f"BiRefNet: node_id={node_id}, starting processing")
        birefnet_runtime_notice_emitted = False
        for i, frame in enumerate(frames):
            if self.cancel_flag.is_set():
                break
            
            # frame is RGB numpy array (H, W, 3)
            alpha = self.birefnet_service.process_image(
                frame, usage=usage, half_precision=half_precision
            )
            if not birefnet_runtime_notice_emitted:
                runtime_notice = self.birefnet_service.consume_runtime_notice()
                if runtime_notice:
                    status_text = self._tr("worker_birefnet_runtime_cpu_fallback").format(
                        reason=runtime_notice
                    )
                    self.stage_progress.emit(20, status_text)
                    self.log_message.emit(status_text)
                    birefnet_runtime_notice_emitted = True
            alpha = self._apply_birefnet_mask_morphology(alpha, dilate_radius, erode_radius)
            if disk_sequence_dir is not None:
                alpha_u8 = np.clip(np.asarray(alpha, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
                alpha_path = disk_sequence_dir / f"{i + 1:04d}.png"
                cv2.imwrite(str(alpha_path), alpha_u8)
                disk_sequence_paths.append(str(alpha_path))
            else:
                alphas.append(alpha)
            
            # Stream to downstream export nodes (if connected)
            self._stream_graph_write_frame(
                node_id,
                "alpha",
                alpha,
                i,
                is_video=len(frames) > 1,
            )
            
            # Stream preview directly to viewer when BiRefNet node is selected
            preview = self._coerce_preview_frame(alpha)
            if preview is not None:
                preview_frame_index = i + int(getattr(self, "_graph_start_frame", 0) or 0)
                self.graph_stream_preview.emit(
                    node_id,
                    {
                        "frame": preview,
                        "path": "",
                        "stream": "alpha",
                    },
                    preview_frame_index,
                )
                if i == 0:
                    self.log_message.emit(f"BiRefNet: streaming preview for node_id={node_id}")
            
            # Emit progress
            progress = 75 + int((i + 1) / len(frames) * 25)
            stage_text = self._tr("worker_birefnet_processing").format(current=i + 1, total=len(frames))
            self.stage_progress.emit(progress, stage_text)
            self.node_frame_progress.emit("birefnet", i + 1, len(frames))

        if disk_sequence_dir is not None:
            return {
                "alpha": {
                    "__disk_sequence__": True,
                    "paths": disk_sequence_paths,
                    "count": len(disk_sequence_paths),
                    "format": "png",
                    "source_node": "birefnet",
                }
            }

        return {"alpha": np.array(alphas) if alphas else np.array([])}

    def _execute_gvm_node(self, node_data: dict, inputs: dict) -> dict:
        """Выполнить GVM узел для генерации альфа-маски всего клипа целиком.

        Input:  {image: RGB_frames}
        Output: {alpha: disk_sequence}

        GVM — диффузионная модель с темпоральным вниманием. В отличие от BiRefNet
        обрабатывает весь клип за один вызов (не покадрово), поэтому кадры сначала
        записываются на диск, затем вызывается process_sequence(), а затем результат
        возвращается как __disk_sequence__.
        """
        if "image" not in inputs:
            raise ValueError("GVM node requires 'image' input")

        frames = inputs["image"]
        if not frames:
            return {"alpha": np.array([])}

        properties = node_data.get("properties", {})
        num_frames_per_batch = max(1, int(properties.get("num_frames_per_batch", 8) or 8))
        denoise_steps = max(1, int(properties.get("denoise_steps", 1) or 1))
        decode_chunk_size = max(1, int(properties.get("decode_chunk_size", 4) or 4))
        num_overlap_frames = max(0, int(properties.get("num_overlap_frames", 1) if properties.get("num_overlap_frames") is not None else 1))
        num_interp_frames = max(0, int(properties.get("num_interp_frames", 1) if properties.get("num_interp_frames") is not None else 1))
        dilate_radius = max(0, int(properties.get("dilate_radius", 0) or 0))
        noise_type = str(properties.get("noise_type", "zeros") or "zeros")
        use_clip_img_emb = bool(properties.get("use_clip_img_emb", False))
        node_id = str(node_data.get("id", ""))

        self.gvm_service.set_callbacks(
            progress_callback=lambda percent, msg: self.stage_progress.emit(percent, msg),
            translate=self._tr,
        )

        self.log_message.emit(
            self._tr("worker_gvm_processing_start").format(total=len(frames))
        )
        self.log_message.emit(
            f"GVM: batch={num_frames_per_batch}, chunk={decode_chunk_size}, "
            f"overlap={num_overlap_frames}, interp={num_interp_frames}, dilate={dilate_radius}"
        )

        # 1. Save input frames as PNGs (GVM needs a file-system path).
        self.stage_progress.emit(5, self._tr("worker_gvm_saving_frames"))
        tmp_input_dir = Path(tempfile.mkdtemp(prefix="keyflow_gvm_in_"))
        for i, frame in enumerate(frames):
            frame_bgr = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(tmp_input_dir / f"{i:05d}.png"), frame_bgr)

        # 2. Prepare output directory on a temporary path (avoids polluting user output dir).
        _gvm_out_root = self._graph_output_dir if self._graph_output_dir is not None else Path(tempfile.mkdtemp(prefix="keyflow_gvm_alpha_"))
        disk_sequence_dir = _gvm_out_root / f"gvm_alpha_{node_id}" if self._graph_output_dir is not None else _gvm_out_root
        disk_sequence_dir.mkdir(parents=True, exist_ok=True)
        for stale in disk_sequence_dir.glob("*.png"):
            stale.unlink(missing_ok=True)

        # 3. Load model and run inference.
        if self.cancel_flag.is_set():
            shutil.rmtree(tmp_input_dir, ignore_errors=True)
            return {}

        self.gvm_service.load_model()

        if self.cancel_flag.is_set():
            shutil.rmtree(tmp_input_dir, ignore_errors=True)
            return {}

        # Track how many frames already streamed via per_batch callback.
        _streamed_count: list[int] = [0]
        _last_progress_count: list[int] = [0]
        total_frames_hint = len(frames)  # approximate; actual may differ after overlap

        def _on_progress(done_frames: int, total_frames: int) -> None:
            total_i = max(1, int(total_frames or total_frames_hint or 1))
            done_i = max(0, min(int(done_frames or 0), total_i))
            if done_i < _last_progress_count[0]:
                done_i = _last_progress_count[0]
            _last_progress_count[0] = done_i
            self.node_frame_progress.emit("gvm", done_i, total_i)

        def _per_batch(new_paths: list, start_idx: int) -> None:
            """Called by GVMService after each diffusion batch finishes writing PNGs."""
            is_video = total_frames_hint > 1
            first_preview_done = start_idx > 0
            for j, p in enumerate(new_paths):
                if self.cancel_flag.is_set():
                    return
                alpha_u8 = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if alpha_u8 is None:
                    continue
                frame_idx = start_idx + j
                alpha_f32 = alpha_u8.astype(np.float32) / 255.0
                # Apply dilation here so streamed frames and disk files stay in sync.
                if dilate_radius > 0:
                    alpha_f32 = self._apply_birefnet_mask_morphology(alpha_f32, dilate_radius, 0)
                    alpha_u8_out = np.clip(alpha_f32 * 255.0, 0, 255).astype(np.uint8)
                    cv2.imwrite(str(p), alpha_u8_out)
                # Stream alpha frame to connected Write nodes.
                self._stream_graph_write_frame(
                    node_id, "alpha", alpha_f32, frame_idx, is_video=is_video
                )
                # Emit preview from first batch's first frame.
                if not first_preview_done and j == 0:
                    preview = self._coerce_preview_frame(alpha_f32)
                    if preview is not None:
                        preview_frame_index = int(getattr(self, "_graph_start_frame", 0) or 0)
                        self.graph_stream_preview.emit(
                            node_id,
                            {"frame": preview, "path": "", "stream": "alpha"},
                            preview_frame_index,
                        )
                    first_preview_done = True
            _streamed_count[0] = max(_streamed_count[0], start_idx + len(new_paths))

        alpha_paths = self.gvm_service.process_sequence(
            tmp_input_dir,
            disk_sequence_dir,
            num_frames_per_batch=num_frames_per_batch,
            denoise_steps=denoise_steps,
            decode_chunk_size=decode_chunk_size,
            num_overlap_frames=num_overlap_frames,
            num_interp_frames=num_interp_frames,
            noise_type=noise_type,
            use_clip_img_emb=use_clip_img_emb,
            progress_callback=_on_progress,
            per_batch_callback=_per_batch,
        )

        if alpha_paths:
            final_total = max(total_frames_hint, len(alpha_paths))
            final_done = min(len(alpha_paths), final_total)
            if final_done > _last_progress_count[0]:
                self.node_frame_progress.emit("gvm", final_done, final_total)

        # 4. Stream preview from first frame (if batches didn't emit one yet).
        # Note: dilation is applied inside _per_batch so disk files and streamed frames stay in sync.
        if alpha_paths and _streamed_count[0] == 0:
            first_alpha = cv2.imread(str(alpha_paths[0]), cv2.IMREAD_GRAYSCALE)
            if first_alpha is not None:
                preview_f32 = first_alpha.astype(np.float32) / 255.0
                preview = self._coerce_preview_frame(preview_f32)
                if preview is not None:
                    preview_frame_index = int(getattr(self, "_graph_start_frame", 0) or 0)
                    self.graph_stream_preview.emit(
                        node_id,
                        {"frame": preview, "path": "", "stream": "alpha"},
                        preview_frame_index,
                    )

        # 5. Stream any remaining frames not yet covered by per_batch_callback
        #    (safety fallback — normally _streamed_count[0] == len(alpha_paths)).
        for i in range(_streamed_count[0], len(alpha_paths)):
            p = alpha_paths[i]
            if self.cancel_flag.is_set():
                break
            alpha_u8 = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if alpha_u8 is not None:
                alpha_f32 = alpha_u8.astype(np.float32) / 255.0
                self._stream_graph_write_frame(
                    node_id, "alpha", alpha_f32, i, is_video=len(alpha_paths) > 1
                )

        disk_sequence_paths = [str(p) for p in alpha_paths]
        self.log_message.emit(
            self._tr("gvm_done").format(count=len(disk_sequence_paths))
        )
        shutil.rmtree(tmp_input_dir, ignore_errors=True)

        return {
            "alpha": {
                "__disk_sequence__": True,
                "paths": disk_sequence_paths,
                "count": len(disk_sequence_paths),
                "format": "png",
                "source_node": "gvm",
            }
        }

    def _execute_corridorkey_node(self, node_data: dict, inputs: dict) -> dict:
        """Выполнить CorridorKey узел для green screen removal.

        Input: {image: RGB_frames, alphahint (optional): alpha_frames}
        Output: any subset of {alpha, fg, comp, processed}
        """
        if "image" not in inputs:
            raise ValueError("CorridorKey node requires 'image' input")
        
        frames = inputs["image"]
        alpha_hints = inputs.get("alphahint")
        disk_alpha_paths: list[str] = []
        disk_sequence_source_node: str = "gvm"
        if isinstance(alpha_hints, dict) and bool(alpha_hints.get("__disk_sequence__")):
            disk_alpha_paths = [str(p) for p in (alpha_hints.get("paths") or []) if str(p).strip()]
            disk_sequence_source_node = str(alpha_hints.get("source_node") or "gvm").strip().lower()
            alpha_hints = None
        deferred_birefnet_node = inputs.get("__deferred_birefnet_node")
        deferred_sam_node = inputs.get("__deferred_sam_node")
        properties = node_data.get("properties", {})
        node_id_str = str(node_data.get("id", ""))
        requested_outputs = self._resolve_requested_output_ports(
            node_id_str,
            {"alpha", "fg", "comp", "processed"},
        )
        alpha_hint_mode = self._resolve_alpha_hint_mode(node_data)
        input_colorspace = str(properties.get("input_colorspace", "auto")).strip().lower()
        if input_colorspace not in {"auto", "srgb", "linear"}:
            input_colorspace = "auto"

        if input_colorspace == "auto":
            # Auto-detect from source path extension AND first-frame dtype.
            # Rules:
            #   - Single EXR still:    load_image_float → float32 linear  → linear ✓
            #   - EXR sequence:        load_rgb_image → _ensure_uint8_rgb → uint8  → sRGB ✓
            #   - MP4/MOV/PNG/JPG:     uint8 or float32 sRGB              → sRGB  ✓
            # Only treat as linear when BOTH dtype is float32 AND ext is .exr.
            _source_ext = Path(self._graph_source_path).suffix.lower() if getattr(self, "_graph_source_path", None) else ""
            _first_frame = frames[0] if isinstance(frames, list) and frames else None
            _frame_dtype = getattr(np.asarray(_first_frame), "dtype", None) if _first_frame is not None else None
            _is_float_frame = _frame_dtype is not None and np.issubdtype(_frame_dtype, np.floating)
            if _is_float_frame and _source_ext == ".exr":
                input_colorspace = "linear"
                self.log_message.emit(
                    f"CorridorKey: auto-detected input_colorspace=linear "
                    f"(source={_source_ext!r}, frame_dtype={_frame_dtype})"
                )
            else:
                input_colorspace = "srgb"

        input_is_linear = input_colorspace == "linear"
        
        despill_strength = float(properties.get("despill_strength", 0.5))
        # Pre-compute despill_01 once (same formula used in service).
        despill_01 = float(np.clip(despill_strength, 0.0, 1.0))
        despeckle = bool(properties.get("despeckle", True))
        despeckle_size = int(properties.get("despeckle_size", 400))
        hint_dilate_radius = max(0, int(properties.get("hint_dilate_radius", 0)))
        matte_clip_black = float(properties.get("matte_clip_black", 0.0))
        matte_clip_white = float(properties.get("matte_clip_white", 1.0))
        matte_shrink_grow = float(properties.get("matte_shrink_grow", 0.0))
        matte_edge_blur = float(properties.get("matte_edge_blur", 0.0))
        matte_gamma = float(properties.get("matte_gamma", 1.0))
        temporal_smoothing = float(properties.get("temporal_smoothing", 0.0))
        # Processed mode is the single production-safe output contract.
        output_mode = "processed"
        refiner_strength = float(properties.get("refiner_strength", 1.0))
        use_refiner = bool(properties.get("use_refiner", True))

        if alpha_hints is None and not disk_alpha_paths and deferred_birefnet_node is None and deferred_sam_node is None:
            raise ValueError(
                "CorridorKey requires an alpha hint sequence. "
                "Connect an alpha-hint source to CorridorKey Alpha Hint input."
            )

        if alpha_hints is not None:
            effective_mode = "batch"
        elif disk_alpha_paths:
            effective_mode = "staged"
        elif deferred_birefnet_node is not None:
            effective_mode = "staged"   # BiRefNet: generate masks → save to disk → unload → read per-frame
        else:
            effective_mode = "sam_disk"  # SAM: masks already on disk → read per-frame, no RAM spike

        if effective_mode == "sam_disk":
            reason_key = "worker_corridorkey_mode_reason_auto_sam_disk"
        elif alpha_hint_mode == "batch" and effective_mode == "staged":
            reason_key = "worker_corridorkey_mode_reason_staged_enforced"
        elif alpha_hint_mode == "batch":
            reason_key = "worker_corridorkey_mode_reason_batch_forced"
        elif alpha_hint_mode == "staged" and effective_mode == "staged":
            reason_key = "worker_corridorkey_mode_reason_staged_forced"
        elif alpha_hint_mode == "staged" and effective_mode == "batch":
            reason_key = "worker_corridorkey_mode_reason_staged_unavailable"
        elif effective_mode == "staged":
            reason_key = "worker_corridorkey_mode_reason_auto_staged"
        else:
            reason_key = "worker_corridorkey_mode_reason_auto_batch"
        self.corridorkey_mode_resolved.emit(alpha_hint_mode, effective_mode, reason_key)
        self._emit_corridorkey_mode_indicator(alpha_hint_mode, effective_mode, reason_key)

        # Staged mode (matching original CorridorKey project):
        # Stage 1 — generate ALL alpha hints with BiRefNet, save to disk
        # Stage 2 — unload BiRefNet to free memory
        # Stage 3 — run CorridorKey reading masks from disk one-by-one
        staged_hints_dir: Path | None = None
        staged_hint_count = 0
        # SAM disk-streaming mode: path map built here; no model, no temp dir
        sam_frame_paths: dict[int, str] = {}
        sam_fallback_path: str = ""
        if alpha_hints is None and not disk_alpha_paths and deferred_birefnet_node is not None:
            bprops = (deferred_birefnet_node or {}).get("properties", {})
            usage = bprops.get("usage", "General")
            half_precision = bprops.get("half_precision", False)
            dilate_radius = max(0, int(bprops.get("dilate_radius", 0) or 0))
            erode_radius = max(0, int(bprops.get("erode_radius", 0) or 0))

            self.birefnet_service.set_callbacks(
                progress_callback=lambda percent, msg: self.stage_progress.emit(percent, msg),
                translate=self._tr,
            )
            self.birefnet_service.load_model(usage=usage)

            self.log_message.emit(
                self._tr("worker_birefnet_processing_start").format(total=len(frames), preset=usage)
            )
            self.log_message.emit(
                f"BiRefNet(staged): usage={usage}, half_precision={bool(half_precision)}, "
                f"dilate={dilate_radius}, erode={erode_radius}"
            )
            birefnet_runtime_notice_emitted = False

            # --- Stage 1: generate all alpha hints and save to disk ---
            # Matches the original CorridorKey project: masks are written as
            # PNG files so they don't occupy RAM while CorridorKey runs.
            staged_hints_dir = Path(tempfile.mkdtemp(prefix="keyflow_alphahint_"))
            staged_hint_count = 0
            deferred_birefnet_node_id = str(deferred_birefnet_node.get("id", ""))
            try:
                for i, frame in enumerate(frames):
                    if self.cancel_flag.is_set():
                        break

                    hint = self.birefnet_service.process_image(
                        frame,
                        usage=usage,
                        half_precision=half_precision,
                    )
                    if not birefnet_runtime_notice_emitted:
                        runtime_notice = ""
                        if hasattr(self.birefnet_service, "consume_runtime_notice"):
                            runtime_notice = self.birefnet_service.consume_runtime_notice()
                        if runtime_notice:
                            status_text = self._tr("worker_birefnet_runtime_cpu_fallback").format(
                                reason=runtime_notice
                            )
                            self.stage_progress.emit(20, status_text)
                            self.log_message.emit(status_text)
                            birefnet_runtime_notice_emitted = True
                    hint = self._apply_birefnet_mask_morphology(hint, dilate_radius, erode_radius)

                    # Save as 8-bit PNG (binary mask 0/255) — same as original project
                    hint_u8 = np.clip(hint * 255.0, 0, 255).astype(np.uint8)
                    cv2.imwrite(str(staged_hints_dir / f"{i + 1:04d}.png"), hint_u8)
                    staged_hint_count += 1
                    
                    # Stream preview directly to viewer when BiRefNet node is selected (staged mode)
                    preview = self._coerce_preview_frame(hint)
                    if preview is not None:
                        preview_frame_index = i + int(getattr(self, "_graph_start_frame", 0) or 0)
                        self.graph_stream_preview.emit(
                            deferred_birefnet_node_id,
                            {
                                "frame": preview,
                                "path": "",
                                "stream": "alpha",
                            },
                            preview_frame_index,
                        )

                    progress = int((i + 1) / len(frames) * 50)  # 0-50% for masks
                    stage_text = self._tr("worker_birefnet_processing").format(current=i + 1, total=len(frames))
                    self.stage_progress.emit(progress, stage_text)
                    self.node_frame_progress.emit("birefnet", i + 1, len(frames))
            except Exception:
                shutil.rmtree(staged_hints_dir, ignore_errors=True)
                raise

            # --- Stage 2: unload BiRefNet to free memory ---
            self.birefnet_service.unload_model()
            self.log_message.emit("BiRefNet: model unloaded, freeing memory for CorridorKey")

            if self.cancel_flag.is_set() or staged_hint_count == 0:
                shutil.rmtree(staged_hints_dir, ignore_errors=True)
                return {}

            self.log_message.emit(
                f"BiRefNet: {staged_hint_count} masks saved to temp dir"
            )
            progress_base = 50  # Staged: BiRefNet used 0-50%, CorridorKey uses 50-100%
        elif deferred_sam_node is not None:
            # SAM disk-streaming: build frame-index → file path map (no disk writes).
            # Masks were pre-computed by SAM2 propagation and stored as payload paths.
            sam_props = (deferred_sam_node or {}).get("properties", {})
            for entry in (sam_props.get("mask_payloads") or []):
                if not isinstance(entry, dict):
                    continue
                path = str(entry.get("path", "")).strip()
                if not path or not Path(path).exists():
                    continue
                try:
                    frame_idx = int(entry.get("frame_index", 0) or 0)
                except Exception:
                    continue
                sam_frame_paths[frame_idx] = path

            sam_fallback_path = str(sam_props.get("_mask_source_path", "")).strip()
            if sam_fallback_path and not Path(sam_fallback_path).exists():
                sam_fallback_path = ""

            if not sam_frame_paths and not sam_fallback_path:
                raise ValueError(
                    "SAM2 node has no propagated masks. "
                    "Run SAM2 propagation on the node before executing the graph."
                )

            sam_node_id = str((deferred_sam_node or {}).get("id", ""))
            self.log_message.emit(
                f"SAM disk-streaming: {len(sam_frame_paths)} keyframe mask(s), "
                f"fallback={'yes' if sam_fallback_path else 'no'} "
                f"for {len(frames)} frame(s)"
            )
            progress_base = 0  # SAM disk: no split, CorridorKey uses full 0-100%
        else:
            progress_base = 0  # Pure batch: CorridorKey uses full 0-100%

        # Flag combinations for per-frame loading:
        #   is_staged_disk  → BiRefNet staged temp dir  (i+1 → 4-digit PNG)
        #   is_sam_disk     → SAM payload paths         (global frame idx → resolved path)
        is_staged_disk = staged_hints_dir is not None and staged_hints_dir.is_dir()
        is_sam_disk = deferred_sam_node is not None and (bool(sam_frame_paths) or bool(sam_fallback_path))
        is_disk_sequence = bool(disk_alpha_paths)

        frame_count = len(frames)
        if is_staged_disk:
            alpha_count = staged_hint_count
        elif is_disk_sequence:
            alpha_count = len(disk_alpha_paths)
        elif is_sam_disk:
            # SAM masks are indexed by keyframes; gaps are filled by proximity-resolve.
            # We accept any non-empty map as sufficient for all frames.
            alpha_count = frame_count
        else:
            try:
                alpha_count = len(alpha_hints)
            except TypeError:
                alpha_count = 0

        self.log_message.emit(
            f"CorridorKey: validating alpha hints ({alpha_count}/{frame_count} frames)"
        )
        corridorkey_runtime_notice_emitted = False
        if alpha_count != frame_count:
            if is_staged_disk:
                shutil.rmtree(staged_hints_dir, ignore_errors=True)
            raise ValueError(
                self._tr("err_corridorkey_frame_mismatch").format(
                    video=frame_count, alpha=alpha_count
                )
            )

        # Only validate shape for in-memory batch hints upfront.
        # Staged/SAM-disk/disk-sequence hints are validated per-frame during the loop.
        if not is_staged_disk and not is_sam_disk and not is_disk_sequence:
            for idx in range(frame_count):
                frame_shape = np.asarray(frames[idx]).shape
                alpha_shape = np.asarray(alpha_hints[idx]).shape
                if len(frame_shape) < 2:
                    raise ValueError(f"CorridorKey: invalid image frame shape at index {idx}: {frame_shape}")
                if len(alpha_shape) < 2:
                    raise ValueError(f"CorridorKey: invalid alpha frame shape at index {idx}: {alpha_shape}")
                if frame_shape[0] != alpha_shape[0] or frame_shape[1] != alpha_shape[1]:
                    raise ValueError(
                        "CorridorKey alpha/image size mismatch at frame "
                        f"{idx}: image={frame_shape[:2]}, alpha={alpha_shape[:2]}"
                    )
        
        self.log_message.emit(self._tr("worker_corridorkey_processing_start").format(total=len(frames)))
        
        outputs_by_key: dict[str, list[np.ndarray]] = {key: [] for key in requested_outputs}
        need_fg_path = "fg" in requested_outputs or "processed" in requested_outputs or "comp" in requested_outputs
        need_processed = "processed" in requested_outputs
        need_comp_rebuild = "comp" in requested_outputs  # rebuild comp from despilled fg
        need_alpha_post = "alpha" in requested_outputs or need_fg_path
        
        try:
            previous_alpha_post: np.ndarray | None = None
            for i, frame in enumerate(frames):
                if self.cancel_flag.is_set():
                    break
                
                # Load alpha hint from the appropriate source:
                #   is_staged_disk → BiRefNet temp dir (sequential 4-digit PNG)
                #   is_sam_disk    → SAM payload paths (per global frame index)
                #   batch          → in-memory list
                if is_staged_disk:
                    hint_path = staged_hints_dir / f"{i + 1:04d}.png"
                    hint_u8 = cv2.imread(str(hint_path), cv2.IMREAD_GRAYSCALE)
                    if hint_u8 is None:
                        raise FileNotFoundError(
                            f"CorridorKey staged: mask file missing: {hint_path}"
                        )
                    alpha_hint = hint_u8.astype(np.float32) / 255.0
                elif is_disk_sequence:
                    hint_path = disk_alpha_paths[i] if i < len(disk_alpha_paths) else ""
                    if not hint_path:
                        raise FileNotFoundError(
                            f"CorridorKey disk-sequence: missing mask path for frame {i}"
                        )
                    hint_u8 = cv2.imread(str(hint_path), cv2.IMREAD_GRAYSCALE)
                    if hint_u8 is None:
                        raise FileNotFoundError(
                            f"CorridorKey disk-sequence: mask file unreadable: {hint_path}"
                        )
                    # GVM (and other disk-sequence sources) may produce alpha frames at a
                    # different resolution than the input (e.g. GVM infers at 1024×1820
                    # while the source is 4K).  Resize to match the current frame.
                    frame_h, frame_w = np.asarray(frame).shape[:2]
                    if hint_u8.shape[0] != frame_h or hint_u8.shape[1] != frame_w:
                        hint_u8 = cv2.resize(hint_u8, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
                    alpha_hint = hint_u8.astype(np.float32) / 255.0
                    self.node_frame_progress.emit(disk_sequence_source_node, i + 1, frame_count)
                elif is_sam_disk:
                    global_idx = int(getattr(self, "_graph_start_frame", 0) or 0) + i
                    resolved_path = self._resolve_sam_frame_path(
                        sam_frame_paths, sam_fallback_path, global_idx
                    )
                    if not resolved_path:
                        raise FileNotFoundError(
                            f"CorridorKey SAM-disk: no mask resolved for frame {global_idx}"
                        )
                    hint_u8 = cv2.imread(resolved_path, cv2.IMREAD_GRAYSCALE)
                    if hint_u8 is None:
                        raise FileNotFoundError(
                            f"CorridorKey SAM-disk: mask file unreadable: {resolved_path}"
                        )
                    alpha_hint = hint_u8.astype(np.float32) / 255.0
                    # Stream SAM mask preview to viewer (same as BiRefNet staged streaming)
                    preview = self._coerce_preview_frame(hint_u8)
                    if preview is not None:
                        preview_frame_index = global_idx
                        self.graph_stream_preview.emit(
                            sam_node_id,
                            {
                                "frame": preview,
                                "path": "",
                                "stream": "mask",
                            },
                            preview_frame_index,
                        )
                    self.node_frame_progress.emit("sam2", i + 1, frame_count)
                else:
                    alpha_hint = alpha_hints[i] if alpha_hints is not None else None

                # Optionally widen the alpha hint before feeding CorridorKey.
                # A slightly expanded mask gives the model more edge context.
                if hint_dilate_radius > 0 and alpha_hint is not None:
                    alpha_hint = self._apply_birefnet_mask_morphology(
                        alpha_hint, hint_dilate_radius, 0
                    )

                result = self.corridorkey_service.process_frame(
                    frame,
                    alpha_hint=alpha_hint,
                    despill_strength=despill_strength,
                    despeckle=despeckle,
                    despeckle_size=despeckle_size,
                    refiner_strength=refiner_strength,
                    use_refiner=use_refiner,
                    input_is_linear=input_is_linear,
                )
                if not corridorkey_runtime_notice_emitted:
                    runtime_notice = ""
                    if hasattr(self.corridorkey_service, "consume_runtime_notice"):
                        runtime_notice = self.corridorkey_service.consume_runtime_notice()
                    if runtime_notice:
                        status_text = self._tr("worker_corridorkey_runtime_cpu_fallback").format(
                            reason=runtime_notice
                        )
                        self.stage_progress.emit(progress_base, status_text)
                        self.log_message.emit(status_text)
                        corridorkey_runtime_notice_emitted = True
                
                # Apply post controls only for outputs that are consumed downstream.
                alpha_ctrl: np.ndarray | None = None
                engine_alpha_raw: np.ndarray | None = None
                if need_alpha_post:
                    engine_alpha_raw = coerce_alpha_2d(result.get("alpha"))
                    alpha_ctrl = engine_alpha_raw
                    if alpha_ctrl is not None:
                        alpha_ctrl = self._apply_corridorkey_alpha_controls(
                            alpha_ctrl,
                            clip_black=matte_clip_black,
                            clip_white=matte_clip_white,
                            matte_gamma=matte_gamma,
                            shrink_grow=matte_shrink_grow,
                            edge_blur=matte_edge_blur,
                        )
                        if previous_alpha_post is not None and temporal_smoothing > 1e-6:
                            t = float(np.clip(temporal_smoothing, 0.0, 1.0))
                            alpha_ctrl = (
                                previous_alpha_post * t + alpha_ctrl * (1.0 - t)
                            ).astype(np.float32)
                        previous_alpha_post = alpha_ctrl
                        result["alpha"] = alpha_ctrl[:, :, np.newaxis]

                source_rgb: np.ndarray | None = None
                fg_rgb: np.ndarray | None = None
                if need_fg_path:
                    source_rgb = coerce_rgb_float01(frame)
                    fg_rgb = coerce_rgb_float01(result.get("fg"))
                    if fg_rgb is None and source_rgb is not None:
                        fg_rgb = source_rgb.copy()
                    if source_rgb is None and fg_rgb is not None:
                        source_rgb = fg_rgb.copy()

                alpha_was_modified = (alpha_ctrl is not None and engine_alpha_raw is not None
                                      and not np.array_equal(alpha_ctrl, engine_alpha_raw))

                if need_processed and fg_rgb is not None:
                    _alpha_for_proc = alpha_ctrl if alpha_ctrl is not None else coerce_alpha_2d(result.get("alpha"))
                    if _alpha_for_proc is not None and (alpha_was_modified or despill_01 > 1e-6):
                        # Rebuild from raw fg using max(R,B) despill mode.
                        if despill_01 > 0:
                            _r = fg_rgb[:, :, 0]; _g = fg_rgb[:, :, 1]; _b = fg_rgb[:, :, 2]
                            _limit = np.maximum(_r, _b)  # max(R,B): teal/cyan preserved
                            _spill = np.maximum(_g - _limit, 0.0)
                            # Luminance-compensating despill: add green's Rec.601 luminance
                            # weight (0.587) back as neutral gray to avoid purple shift.
                            _lum = _spill * 0.587
                            _g_new = _g - _spill + _lum
                            _r_new = _r + _lum
                            _b_new = _b + _lum
                            _despilled = np.stack([_r_new, _g_new, _b_new], axis=-1)
                            if despill_01 < 1.0:
                                _fg_for_proc = fg_rgb * (1.0 - despill_01) + _despilled * despill_01
                            else:
                                _fg_for_proc = _despilled
                            _fg_for_proc = np.clip(_fg_for_proc, 0.0, 1.0)
                        else:
                            _fg_for_proc = fg_rgb
                        result["processed"] = build_corridorkey_processed_output(
                            output_mode, source_rgb, _fg_for_proc, _alpha_for_proc,
                        )

                if need_comp_rebuild and fg_rgb is not None:
                    _alpha_for_comp = alpha_ctrl if alpha_ctrl is not None else coerce_alpha_2d(result.get("alpha"))
                    if _alpha_for_comp is not None and (alpha_was_modified or despill_01 > 1e-6):
                        if despill_01 > 0:
                            _r = fg_rgb[:, :, 0]; _g = fg_rgb[:, :, 1]; _b = fg_rgb[:, :, 2]
                            _limit = np.maximum(_r, _b)  # max(R,B)
                            _spill = np.maximum(_g - _limit, 0.0)
                            _lum_c = _spill * 0.587
                            _despilled_c = np.stack([_r + _lum_c, _g - _spill + _lum_c, _b + _lum_c], axis=-1)
                            if despill_01 < 1.0:
                                _fg_comp = fg_rgb * (1.0 - despill_01) + _despilled_c * despill_01
                            else:
                                _fg_comp = _despilled_c
                            _fg_comp = np.clip(_fg_comp, 0.0, 1.0)
                        else:
                            _fg_comp = fg_rgb
                        _a3 = np.clip(_alpha_for_comp, 0.0, 1.0)[:, :, np.newaxis]
                        _x = np.clip(_fg_comp, 0.0, 1.0).astype(np.float32)
                        _fg_lin = np.where(_x <= 0.04045, _x / 12.92, ((_x + 0.055) / 1.055) ** 2.4)
                        _h, _w = _fg_comp.shape[:2]
                        _gy = np.arange(_h) // 64
                        _gx = np.arange(_w) // 64
                        _checker = ((_gy[:, None] + _gx[None, :]) % 2).astype(np.float32)
                        _bg_srgb = _checker * 0.4 + 0.15
                        _bg_lin = np.where(_bg_srgb <= 0.04045, _bg_srgb / 12.92,
                                           ((_bg_srgb + 0.055) / 1.055) ** 2.4)[..., np.newaxis]
                        _comp_lin = _fg_lin * _a3 + _bg_lin * (1.0 - _a3)
                        _c = np.clip(_comp_lin, 0.0, 1.0)
                        _comp_srgb = np.where(_c <= 0.0031308, _c * 12.92, 1.055 * _c ** (1.0 / 2.4) - 0.055)
                        result["comp"] = np.clip(_comp_srgb, 0.0, 1.0)

                for key in requested_outputs:
                    if key in result and result[key] is not None:
                        frame_out = np.asarray(result[key])
                        outputs_by_key[key].append(frame_out)
                        self._stream_graph_write_frame(
                            node_id_str,
                            key,
                            frame_out,
                            i,
                            is_video=len(frames) > 1,
                        )
                
                # Emit progress
                progress_range = 100 - progress_base
                progress = progress_base + int((i + 1) / len(frames) * progress_range)
                stage_text = self._tr("worker_corridorkey_processing").format(current=i + 1, total=len(frames))
                self.stage_progress.emit(progress, stage_text)
                self.node_frame_progress.emit("corridorkey", i + 1, len(frames))
                
                # Log every 10 frames or on first/last frame
                if (i + 1) % 10 == 0 or i + 1 == len(frames):
                    self.log_message.emit(f"CorridorKey: обработано {i + 1}/{len(frames)} кадров")
        finally:
            # Cleanup temp mask directory (staged path)
            if is_staged_disk:
                shutil.rmtree(staged_hints_dir, ignore_errors=True)
                self.log_message.emit("BiRefNet: temp mask directory cleaned up")
        
        output: dict[str, np.ndarray] = {}
        for key, frames_list in outputs_by_key.items():
            if frames_list:
                output[key] = np.array(frames_list)

        if not output:
            raise RuntimeError("CorridorKey node produced no outputs")

        return output

    def _execute_matanyone2_node(self, node_data: dict, inputs: dict) -> dict:
        """Выполнить MatAnyone2 узел для matting.
        
        Input: {img: RGB_frames, mask: mask_array (optional)}
        Output: {alpha: alpha_frames, fg: fg_frames}
        """
        frames = inputs.get("img") or inputs.get("image")
        if frames is None:
            raise ValueError("MatAnyone2 node requires 'img' input")
        
        properties = node_data.get("properties", {})
        
        # Load mask: prefer direct graph input; fallback to selected/uploaded SAM mask path.
        mask = inputs.get("mask")
        if isinstance(mask, dict) and bool(mask.get("__disk_sequence__")):
            disk_paths = [str(p) for p in (mask.get("paths") or []) if str(p).strip()]
            first_path = disk_paths[0] if disk_paths else ""
            if first_path and Path(first_path).exists():
                mask = self._load_mask(first_path, frames[0].shape[:2])
            else:
                mask = None
        if mask is not None and not isinstance(mask, dict):
            mask = self._coerce_matting_mask(mask, frames[0].shape[:2])

        if mask is None or isinstance(mask, dict):
            mask_path = getattr(self, "_graph_mask_path", "")
            if mask_path and Path(mask_path).exists():
                mask = self._load_mask(mask_path, frames[0].shape[:2])
            else:
                raise ValueError("MatAnyone2 node requires a mask (SAM mask not found)")
        
        node_id_str = str(node_data.get("id", ""))
        is_video = len(frames) > 1
        fg_bg = properties.get("fg_background", "green")
        # Accumulators — populated per-frame inside the callback so PNG/video
        # frames are written to disk immediately instead of after full run.
        alphas_streamed: list[np.ndarray] = []
        fg_streamed: list[np.ndarray] = []

        def _matting_progress(current: int, total: int, frame_rgb, alpha_np):
            self.node_frame_progress.emit("matting", current, total)
            if self.cancel_flag.is_set():
                return
            frame_idx = current - 1  # current is 1-based
            # Compute fg composite on the fly
            a_f = alpha_np[:, :, np.newaxis] if alpha_np.ndim == 2 else alpha_np
            a_f = np.asarray(a_f, dtype=np.float32)
            src = np.asarray(frame_rgb, dtype=np.float32) / 255.0
            if fg_bg == "checker":
                h, w = src.shape[:2]
                tile = 16
                ys = (np.arange(h) // tile) % 2
                xs = (np.arange(w) // tile) % 2
                checker = ((ys[:, None] ^ xs[None, :]) * 0.5 + 0.25).astype(np.float32)
                bg = np.stack([checker, checker, checker], axis=-1)
                fg = src * a_f + bg * (1.0 - a_f)
            else:  # green (default)
                green = np.array([120, 255, 155], dtype=np.float32) / 255.0
                fg = src * a_f + green * (1.0 - a_f)
            fg_u8 = np.clip(fg * 255.0, 0, 255).astype(np.uint8)
            # Write to disk immediately (PNG seq) or to streaming video writer
            self._stream_graph_write_frame(node_id_str, "alpha", alpha_np, frame_idx, is_video=is_video)
            self._stream_graph_write_frame(node_id_str, "fg", fg_u8, frame_idx, is_video=is_video)
            alphas_streamed.append(alpha_np)
            fg_streamed.append(fg_u8)

        # Ensure model is loaded
        if not self.inference_service.model_service.is_loaded():
            self.stage_progress.emit(8, self._tr("worker_inference_prepare_model"))
            self.log_message.emit(self._tr("worker_inference_load_model"))
            # Free SAM weights before loading MatAnyone2 to reclaim VRAM/RAM.
            self._unload_sam_service_if_loaded()
            self.inference_service.model_service.load_model()

        # Использовать синглтон InferenceService для процесса
        correction_masks = getattr(self, "_graph_correction_masks", None)
        self.inference_service.process_video(
            frames,
            mask=mask,
            n_warmup=properties.get("warmup", 10),
            r_erode=properties.get("erode", 0),
            r_dilate=properties.get("dilate", 0),
            progress_callback=_matting_progress,
            cancel_flag=self.cancel_flag,
            correction_masks=correction_masks,
        )

        return {
            "alpha": np.array(alphas_streamed) if alphas_streamed else np.array([]),
            "fg": np.array(fg_streamed) if fg_streamed else np.array([]),
        }

    @staticmethod
    def _write_video(frames, path, fps, *, codec: str = DEFAULT_VIDEO_CODEC, crf: int = DEFAULT_VIDEO_CRF, preset: str = DEFAULT_VIDEO_PRESET):
        """Записывает видео через imageio/ffmpeg с выбранным кодеком и CRF."""
        import imageio

        prepared_frames = [prepare_video_frame(frame, codec) for frame in frames]
        ffmpeg_codec, output_params = build_video_output_params(codec, crf=crf, preset=preset)

        if codec in PRORES_PROFILES:
            try:
                imageio.mimwrite(
                    path, prepared_frames, fps=fps,
                    codec=ffmpeg_codec,
                    macro_block_size=1,
                    output_params=output_params,
                )
                return
            except Exception:
                imageio.mimwrite(path, prepared_frames, fps=fps, quality=8)
                return

        try:
            imageio.mimwrite(
                path, prepared_frames, fps=fps,
                codec=ffmpeg_codec,
                macro_block_size=1,
                output_params=output_params,
            )
        except Exception:
            # Fallback: без explicit codec params
            imageio.mimwrite(path, prepared_frames, fps=fps, quality=8)

    @staticmethod
    def _mux_audio(video_path, audio_path, out_path):
        """Добавляет аудио в видео через ffmpeg. Возвращает финальный путь."""
        try:
            import subprocess
            from app.utils.ffmpeg import get_ffmpeg_exe
            subprocess.run(
                [get_ffmpeg_exe(), "-y", "-i", video_path, "-i", audio_path,
                 "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)],
                check=True, capture_output=True
            )
            return out_path
        except Exception:
            return video_path  # fallback: вернуть видео без аудио

