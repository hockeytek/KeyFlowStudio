"""Background worker for interactive SAM mask generation."""
import logging
import threading
import traceback

import numpy as np

from PySide6.QtCore import QObject, Signal

from app.i18n import t
from app.services.sam2_service import Sam2Service
from app.services.sam3_service import Sam3Service

logger = logging.getLogger(__name__)
_SAM_CANCELLED_TOKEN = "__SAM_CANCELLED__"


class _Sam3InteractiveAdapter:
    """Tiny adapter exposing Sam2-like interface for interactive SAM3 image prompts."""

    def __init__(self, model_type: str):
        normalized = str(model_type or "sam3").strip().lower()
        self._model_type = normalized if normalized in {"sam3", "sam3.1"} else "sam3"

    def unload(self) -> None:
        # Sam3Service manages cached runtimes at class level.
        return None

    def generate_mask(self, image_rgb, points, labels, context=None):
        ctx = context if isinstance(context, dict) else {}
        concept = str(ctx.get("concept", "") or "").strip()
        prompt_points: list[list[int]] = []
        for (x, y), label in zip(points or [], labels or []):
            try:
                prompt_points.append([int(x), int(y), int(label)])
            except Exception:
                continue
        masks = Sam3Service.predict_image(
            model_type=self._model_type,
            image=image_rgb,
            points=prompt_points,
            concept=concept,
        )
        if not masks:
            return None
        combined = np.zeros_like(np.asarray(masks[0], dtype=np.uint8))
        for mask in masks:
            arr = np.asarray(mask, dtype=np.uint8)
            combined = np.where(arr > 127, 255, combined).astype(np.uint8)
        return combined


class SamMaskWorker(QObject):
    """Generate SAM masks in a dedicated worker thread."""

    stage_progress = Signal(int, str)
    node_frame_progress = Signal(str, int, int)  # node_type, current, total
    finished = Signal(object)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.sam_service = None
        self.language_code = "ru"
        self._model_type = "vit_h"
        self._backend = "sam2"
        self._service_backend = ""
        self._cancel_event = threading.Event()

    def set_language(self, language_code: str) -> None:
        self.language_code = language_code if language_code in {"ru", "en"} else "ru"

    def set_model_type(self, model_type: str) -> None:
        """Switch to a different SAM model. Unloads the current one if loaded."""
        if self._model_type != model_type:
            self._model_type = model_type
            if self.sam_service is not None:
                self.sam_service.unload()
                self.sam_service = None
                self._service_backend = ""

    def set_backend(self, backend: str) -> None:
        normalized = str(backend or "sam2").strip().lower()
        if normalized not in {"sam2", "sam3"}:
            normalized = "sam2"
        self._backend = normalized
        if self.sam_service is not None:
            self.sam_service.unload()
            self.sam_service = None
            self._service_backend = ""

    def _tr(self, key: str) -> str:
        return t(key, self.language_code)

    def _status_text(self, sam2_key: str, sam3_key: str) -> str:
        return self._tr(sam3_key if self._backend == "sam3" else sam2_key)

    def _create_service(self):
        if self._backend == "sam3":
            return _Sam3InteractiveAdapter(self._model_type)
        return Sam2Service(
            model_type=self._model_type,
            progress_callback=lambda percent, msg: self.stage_progress.emit(percent, msg),
            translate=self._tr,
            frame_progress_callback=lambda cur, tot: self.node_frame_progress.emit("sam2", cur, tot),
            should_abort=self._cancel_event.is_set,
        )

    def _dispose_service(self, *, reason: str = "") -> None:
        if self.sam_service is None:
            self._service_backend = ""
            return
        try:
            self.sam_service.unload()
        except Exception:
            if reason:
                logger.debug("[SAM] unload failed after %s", reason, exc_info=True)
            else:
                logger.debug("[SAM] unload failed", exc_info=True)
        finally:
            self.sam_service = None
            self._service_backend = ""

    @staticmethod
    def _to_binary_mask_u8(mask) -> np.ndarray | None:
        """Normalize SAM mask-like data to 2D uint8 {0,255}."""
        if mask is None:
            return None
        arr = np.asarray(mask)
        if arr.ndim != 2:
            return None

        if arr.dtype == np.bool_:
            return arr.astype(np.uint8) * 255

        arr_f = arr.astype(np.float32)
        if arr_f.size == 0:
            return None

        min_val = float(np.nanmin(arr_f))
        max_val = float(np.nanmax(arr_f))

        if min_val >= -0.01 and max_val <= 1.01:
            return (arr_f > 0.5).astype(np.uint8) * 255

        if min_val >= -0.5 and max_val <= 255.5:
            return (arr_f > 127.0).astype(np.uint8) * 255

        # Fallback for logits-like outputs: positive region is foreground.
        return (arr_f > 0.0).astype(np.uint8) * 255

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def unload_if_loaded(self) -> None:
        if self.sam_service is None:
            return
        try:
            self.sam_service.unload()
        except Exception:
            logger.debug("[SAM2] unload request failed", exc_info=True)
        finally:
            self.sam_service = None
            self._service_backend = ""

    def generate_mask(self, image_rgb, points, labels, context=None):
        try:
            self._cancel_event.clear()
            self.stage_progress.emit(
                5,
                self._status_text("worker_sam2_prepare", "worker_sam3_prepare"),
            )

            if self.sam_service is None or self._service_backend != self._backend:
                self.stage_progress.emit(
                    20,
                    self._status_text("worker_sam2_load_model", "worker_sam3_load_model"),
                )
                self.sam_service = self._create_service()
                self._service_backend = self._backend

            if self._backend == "sam3":
                mask = self.sam_service.generate_mask(image_rgb, points, labels, context=context)
            else:
                mask = self.sam_service.generate_mask(image_rgb, points, labels)
            if mask is None:
                raise RuntimeError(self._tr("sam_unavailable"))
            mask_arr = self._to_binary_mask_u8(mask)
            if mask_arr is None:
                raise RuntimeError(self._tr("sam_unavailable"))

            self.stage_progress.emit(100, self._tr("sam_mask_ready"))
            self.finished.emit(
                {
                    "op": "generate",
                    "mask": mask_arr,
                }
            )
        except Exception as exc:
            details = str(exc).strip() or repr(exc)
            if _SAM_CANCELLED_TOKEN in details:
                self.finished.emit({"op": "cancelled"})
                return
            logger.error("[SAM] %s\n%s", details, traceback.format_exc())
            self.error.emit(details)

    def propagate_masks(self, context=None):
        try:
            self._cancel_event.clear()
            if self.sam_service is None or self._service_backend != self._backend:
                self.stage_progress.emit(20, self._tr("worker_sam2_load_model"))
                self.sam_service = self._create_service()
                self._service_backend = self._backend

            if not hasattr(self.sam_service, "propagate_with_prompt"):
                raise RuntimeError(self._tr("sam2_native_video_unavailable"))

            ctx = context if isinstance(context, dict) else {}
            seed_mask = ctx.get("seed_mask")
            if seed_mask is not None and not (ctx.get("points") or []):
                if not hasattr(self.sam_service, "propagate_with_seed_mask"):
                    raise RuntimeError(self._tr("sam2_native_video_unavailable"))
                result = self.sam_service.propagate_with_seed_mask(
                    ctx.get("frames") or [],
                    seed_mask,
                    start_index=int(ctx.get("current_frame_index", 0) or 0),
                    direction=str(ctx.get("direction", "forward") or "forward"),
                    reset_session=False,
                )
            else:
                result = self.sam_service.propagate_with_prompt(
                    ctx.get("frames") or [],
                    ctx.get("points") or [],
                    ctx.get("labels") or [],
                    start_index=int(ctx.get("current_frame_index", 0) or 0),
                    direction=str(ctx.get("direction", "forward") or "forward"),
                    reset_session=False,
                )

            frame_index_offset = int(ctx.get("frame_index_offset", 0) or 0)
            sequence_masks_map = {}
            for key, value in (result.get("sequence_masks_map", {}) or {}).items():
                try:
                    mapped_key = int(key) + frame_index_offset
                except Exception:
                    continue
                sequence_masks_map[mapped_key] = (np.asarray(value, dtype=np.uint8) > 0).astype(np.uint8) * 255

            payload = {
                "op": "propagate",
                "sequence_masks_map": sequence_masks_map,
                "current_frame_index": int(
                    ctx.get("current_frame_index_global", ctx.get("current_frame_index", 0)) or 0
                ),
                "tracked_count": len(sequence_masks_map),
                "total_frames": int(result.get("total_frames", 0) or 0),
            }
            self.finished.emit(payload)
            self._dispose_service(reason="propagate")
        except Exception as exc:
            details = str(exc).strip() or repr(exc)
            if _SAM_CANCELLED_TOKEN in details:
                self._dispose_service(reason="propagate-cancel")
                self.finished.emit({"op": "cancelled"})
                return
            self._dispose_service(reason="propagate-error")
            logger.error("[SAM2] propagate error: %s\n%s", details, traceback.format_exc())
            self.error.emit(details)

    def reprompt_frame(self, context=None):
        try:
            self._cancel_event.clear()
            if self.sam_service is None or self._service_backend != self._backend:
                self.stage_progress.emit(20, self._tr("worker_sam2_load_model"))
                self.sam_service = self._create_service()
                self._service_backend = self._backend

            ctx = context if isinstance(context, dict) else {}
            frame_idx = int(ctx.get("current_frame_index", 0) or 0)
            points = ctx.get("points") or []
            labels = ctx.get("labels") or []
            frames = ctx.get("frames") or []
            frame = ctx.get("current_frame")

            if hasattr(self.sam_service, "add_reprompt"):
                result = self.sam_service.add_reprompt(
                    frames,
                    frame_index=frame_idx,
                    points=points,
                    labels=labels,
                )
            else:
                if frame is None:
                    raise RuntimeError(self._tr("err_no_media"))
                mask = self.sam_service.generate_mask(frame, points, labels)
                result = {
                    "frame_index": frame_idx,
                    "mask": np.asarray(mask, dtype=np.uint8),
                }

            mask = np.asarray(result.get("mask"), dtype=np.uint8)
            if mask.ndim != 2:
                raise RuntimeError(self._tr("sam_unavailable"))

            self.finished.emit(
                {
                    "op": "reprompt",
                    "sequence_masks_map": {int(result.get("frame_index", frame_idx)): (mask > 0).astype(np.uint8) * 255},
                    "current_frame_index": frame_idx,
                }
            )
        except Exception as exc:
            details = str(exc).strip() or repr(exc)
            if _SAM_CANCELLED_TOKEN in details:
                self.finished.emit({"op": "cancelled"})
                return
            logger.error("[SAM2] reprompt error: %s\n%s", details, traceback.format_exc())
            self.error.emit(details)

    def reset_video_session(self):
        try:
            if self.sam_service is not None and hasattr(self.sam_service, "reset_video_session"):
                self.sam_service.reset_video_session()
        except Exception:
            logger.debug("[SAM2] reset session failed", exc_info=True)