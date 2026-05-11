"""SAM2 backend adapter.

The service uses official SAM2 image/video predictor APIs.
"""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile
import shutil
import urllib.request

import numpy as np
from PIL import Image

from app.utils import get_device, get_model_variant_dir


logger = logging.getLogger(__name__)


class Sam2Service:
    """SAM2 service used by the SAM2 graph node (internal key: sam2)."""

    # Common SAM2 checkpoint names used in the official repository docs.
    SAM2_CKPT_CANDIDATES = {
        "vit_h": [
            "sam2.1_hiera_large.pt",
            "sam2_hiera_large.pt",
        ],
        "vit_l": [
            "sam2.1_hiera_base_plus.pt",
            "sam2_hiera_base_plus.pt",
        ],
        "vit_b": [
            "sam2.1_hiera_small.pt",
            "sam2_hiera_small.pt",
            "sam2.1_hiera_tiny.pt",
            "sam2_hiera_tiny.pt",
        ],
    }

    SAM2_URLS = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    }

    SAM2_LABELS = {
        "vit_h": "SAM2.1 Hiera Large",
        "vit_l": "SAM2.1 Hiera Base+",
        "vit_b": "SAM2.1 Hiera Small",
    }

    @classmethod
    def _model_dir_for(cls, model_type: str) -> Path:
        return get_model_variant_dir("sam2", model_type)

    @classmethod
    def _hint_file_name(cls, model_type: str) -> str:
        return f"sam2_{model_type}_checkpoint.txt"

    @classmethod
    def get_weight_status(cls, model_type: str) -> dict:
        """Return {"state": "ready"|"missing"} for *model_type*."""
        candidates = cls.SAM2_CKPT_CANDIDATES.get(model_type, [])
        if not candidates:
            return {"state": "missing"}

        model_dir = cls._model_dir_for(model_type)
        for filename in candidates:
            path = model_dir / filename
            if path.is_file():
                return {"state": "ready", "path": str(path)}
        return {"state": "missing", "path": str(model_dir / candidates[0])}

    @classmethod
    def download_checkpoint_for(cls, model_type: str, progress_callback=None) -> str:
        """Download SAM2 checkpoint for *model_type*; return local path."""
        if model_type not in cls.SAM2_URLS:
            raise ValueError(f"Unsupported SAM2 model type: {model_type}")

        url = cls.SAM2_URLS[model_type]
        model_dir = cls._model_dir_for(model_type)
        filename = url.rsplit("/", 1)[-1]
        target_path = model_dir / filename
        model_dir.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback(5, "")
        with urllib.request.urlopen(url) as response, open(target_path, "wb") as f:
            total = int(response.headers.get("Content-Length", "0") or 0)
            downloaded = 0
            chunk_size = 1024 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and progress_callback:
                    progress_callback(int(downloaded * 95 / total) + 5, "")
        if progress_callback:
            progress_callback(100, "")
        return str(target_path)

    # The exact config names can differ between sam2 releases, so we try both.
    SAM2_CFG_CANDIDATES = {
        "vit_h": ["configs/sam2.1/sam2.1_hiera_l.yaml", "configs/sam2/sam2_hiera_l.yaml"],
        "vit_l": ["configs/sam2.1/sam2.1_hiera_b+.yaml", "configs/sam2/sam2_hiera_b+.yaml"],
        "vit_b": [
            "configs/sam2.1/sam2.1_hiera_s.yaml",
            "configs/sam2/sam2_hiera_s.yaml",
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            "configs/sam2/sam2_hiera_t.yaml",
        ],
    }

    def __init__(
        self,
        model_type: str = "vit_h",
        progress_callback=None,
        translate=None,
        frame_progress_callback=None,
        should_abort=None,
    ):
        self.model_type = model_type if model_type in {"vit_h", "vit_l", "vit_b"} else "vit_h"
        self.progress_callback = progress_callback
        self.translate = translate
        self.frame_progress_callback = frame_progress_callback
        self.should_abort = should_abort

        self._predictor = None
        self._native_enabled = False
        self._native_failed_reason = ""
        self._native_load_attempted = False

        self._video_predictor = None
        self._native_video_enabled = False
        self._native_video_failed_reason = ""
        self._native_video_load_attempted = False
        self._video_state = None
        self._video_session_dir: str | None = None
        self._session_frames: list[np.ndarray] = []
        self._session_masks_map: dict[int, np.ndarray] = {}
        self._session_prompt_frame_index = 0
        self._session_points: list[tuple[int, int]] = []
        self._session_labels: list[int] = []
        self._session_obj_id = 1

    def _tr(self, key: str) -> str:
        if self.translate is None:
            return key
        try:
            return self.translate(key)
        except Exception:
            return key

    def _emit_progress(self, percent: int, message: str) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(max(0, min(100, int(percent))), message)
        except Exception:
            pass

    def _emit_frame_progress(self, current: int, total: int) -> None:
        if self.frame_progress_callback is None:
            return
        try:
            self.frame_progress_callback(int(current), int(total))
        except Exception:
            pass

    def _raise_if_cancelled(self) -> None:
        try:
            if callable(self.should_abort) and bool(self.should_abort()):
                raise RuntimeError("__SAM_CANCELLED__")
        except RuntimeError:
            raise
        except Exception:
            pass

    def _resolve_checkpoint_path(self) -> str | None:
        model_dir = self._model_dir_for(self.model_type)
        candidates = self.SAM2_CKPT_CANDIDATES.get(self.model_type, [])
        for name in candidates:
            p = model_dir / name
            if p.is_file():
                return str(p)

        # Also support explicit absolute path via env-like convention file in models dir.
        hint_files = [model_dir / self._hint_file_name(self.model_type)]
        for hint_file in hint_files:
            if not hint_file.is_file():
                continue
            try:
                hinted = hint_file.read_text(encoding="utf-8").strip()
                if hinted and Path(hinted).is_file():
                    return hinted
            except Exception:
                pass
        return None

    def _load_native_predictor(self) -> None:
        if self._predictor is not None or self._native_load_attempted:
            return
        self._native_load_attempted = True

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            self._native_failed_reason = f"sam2 package unavailable: {exc}"
            logger.info("[SAM2] Native image predictor unavailable: %s", self._native_failed_reason)
            return

        ckpt = self._resolve_checkpoint_path()
        if ckpt is None:
            self._native_failed_reason = "sam2 checkpoint not found"
            return

        device = str(get_device())
        errors: list[str] = []
        for cfg in self.SAM2_CFG_CANDIDATES.get(self.model_type, []):
            try:
                self._emit_progress(70, self._tr("worker_sam_load_to_device"))
                model = build_sam2(cfg, ckpt, device=device)
                self._predictor = SAM2ImagePredictor(model)
                self._native_enabled = True
                self._native_failed_reason = ""
                logger.info("[SAM2] Native image predictor loaded (%s, %s)", self.model_type, cfg)
                return
            except Exception as exc:
                errors.append(f"{cfg}: {exc}")

        self._native_failed_reason = "; ".join(errors) if errors else "unknown SAM2 init error"

    def _load_native_video_predictor(self) -> None:
        if self._video_predictor is not None or self._native_video_load_attempted:
            return
        self._native_video_load_attempted = True

        ckpt = self._resolve_checkpoint_path()
        if ckpt is None:
            self._native_video_failed_reason = "sam2 checkpoint not found"
            return

        errors: list[str] = []
        device = str(get_device())

        for cfg in self.SAM2_CFG_CANDIDATES.get(self.model_type, []):
            try:
                # Preferred API from official sam2 package.
                from sam2.build_sam import build_sam2_video_predictor

                predictor = build_sam2_video_predictor(cfg, ckpt, device=device)
                self._video_predictor = predictor
                self._native_video_enabled = True
                self._native_video_failed_reason = ""
                logger.info("[SAM2] Native video predictor loaded (%s, %s)", self.model_type, cfg)
                return
            except Exception as exc:
                errors.append(f"{cfg}: {exc}")

        self._native_video_failed_reason = "; ".join(errors) if errors else "unknown SAM2 video init error"

    def _cleanup_video_session_dir(self) -> None:
        if self._video_session_dir and Path(self._video_session_dir).exists():
            try:
                shutil.rmtree(self._video_session_dir, ignore_errors=True)
            except Exception:
                pass
        self._video_session_dir = None

    def _prepare_video_session_dir(self, frames: list[np.ndarray]) -> str:
        self._cleanup_video_session_dir()
        session_dir = Path(tempfile.mkdtemp(prefix="sam2_video_"))
        for idx, frame in enumerate(frames):
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3 and arr.shape[2] >= 3:
                arr = arr[:, :, :3]
            else:
                raise ValueError("Invalid frame shape for SAM2 video session")
            arr = np.asarray(arr, dtype=np.uint8)
            Image.fromarray(arr).save(str(session_dir / f"{idx:06d}.jpg"), quality=95)
        self._video_session_dir = str(session_dir)
        return self._video_session_dir

    @staticmethod
    def _mask_from_logits(mask_logits, obj_ids, target_obj_id: int) -> np.ndarray | None:
        try:
            logits = mask_logits
            if hasattr(logits, "detach"):
                logits = logits.detach().cpu().numpy()
            else:
                logits = np.asarray(logits)
            if logits.ndim == 4 and logits.shape[1] == 1:
                logits = logits[:, 0, :, :]
            if logits.ndim != 3:
                return None

            idx = 0
            if obj_ids is not None:
                try:
                    ids = list(obj_ids)
                    if target_obj_id in ids:
                        idx = ids.index(target_obj_id)
                except Exception:
                    idx = 0
            idx = max(0, min(logits.shape[0] - 1, idx))
            return (logits[idx] > 0).astype(np.uint8) * 255
        except Exception:
            return None

    def reset_video_session(self) -> None:
        self._video_state = None
        self._session_frames = []
        self._session_masks_map = {}
        self._session_points = []
        self._session_labels = []
        self._session_prompt_frame_index = 0
        self._cleanup_video_session_dir()

    def _ensure_video_session(self, frames_rgb, *, start_index: int = 0, force_reset: bool = False) -> None:
        frames = [np.asarray(frame, dtype=np.uint8) for frame in list(frames_rgb or [])]
        if not frames:
            raise ValueError("frames_rgb is empty")

        need_reset = force_reset or not self._session_frames or len(self._session_frames) != len(frames)
        if need_reset:
            self.reset_video_session()
            self._session_frames = frames
            self._session_prompt_frame_index = int(start_index)

        self._load_native_video_predictor()

        if self._native_video_enabled and self._video_predictor is not None:
            if self._video_state is None or need_reset:
                video_path = self._prepare_video_session_dir(frames)
                try:
                    if hasattr(self._video_predictor, "init_state"):
                        self._video_state = self._video_predictor.init_state(video_path=video_path)
                    else:
                        raise RuntimeError("SAM2 video predictor missing init_state")
                except Exception as exc:
                    self._native_video_enabled = False
                    self._native_video_failed_reason = str(exc)
                    logger.warning("[SAM2] Failed to init native video state, fallback mode: %s", exc)

    def add_reprompt(self, frames_rgb, *, frame_index: int, points, labels, obj_id: int = 1) -> dict:
        self._ensure_video_session(frames_rgb, start_index=frame_index, force_reset=False)
        frame_index = int(frame_index)
        pts = [tuple(map(int, p)) for p in points]
        lbs = [int(v) for v in labels]
        if not pts or len(pts) != len(lbs):
            raise ValueError("points/labels are empty or length mismatch")

        self._session_prompt_frame_index = frame_index
        self._session_points = pts
        self._session_labels = lbs
        self._session_obj_id = int(obj_id)

        if self._native_video_enabled and self._video_predictor is not None and self._video_state is not None:
            if not hasattr(self._video_predictor, "add_new_points_or_box"):
                raise RuntimeError("SAM2 video predictor missing add_new_points_or_box")

            point_coords = np.array(pts, dtype=np.float32)
            point_labels = np.array(lbs, dtype=np.int32)

            _frame_idx, out_obj_ids, out_mask_logits = self._video_predictor.add_new_points_or_box(
                inference_state=self._video_state,
                frame_idx=frame_index,
                obj_id=self._session_obj_id,
                points=point_coords,
                labels=point_labels,
            )
            mask = self._mask_from_logits(out_mask_logits, out_obj_ids, self._session_obj_id)
            if mask is None:
                raise RuntimeError("SAM2 did not return valid logits for re-prompt")
            self._session_masks_map[frame_index] = mask
            return {"frame_index": frame_index, "mask": mask}

        # Framewise mode without SAM2 video API still uses SAM2 image predictor.
        frame = self._session_frames[frame_index]
        mask = self.generate_mask(frame, pts, lbs)
        mask_u8 = np.asarray(mask, dtype=np.uint8)
        self._session_masks_map[frame_index] = mask_u8
        return {"frame_index": frame_index, "mask": mask_u8}

    def propagate_with_prompt(
        self,
        frames_rgb,
        points,
        labels,
        *,
        start_index: int,
        direction: str,
        reset_session: bool = False,
    ) -> dict:
        direction_norm = str(direction or "forward").strip().lower()
        reverse = direction_norm == "backward"
        self._raise_if_cancelled()
        self._ensure_video_session(frames_rgb, start_index=start_index, force_reset=reset_session)
        self._emit_progress(35, self._tr("sam2_status_session_ready"))

        reprompt = self.add_reprompt(
            self._session_frames,
            frame_index=int(start_index),
            points=points,
            labels=labels,
            obj_id=self._session_obj_id,
        )
        _ = reprompt
        self._raise_if_cancelled()

        direction_key = "sam2_direction_backward" if reverse else "sam2_direction_forward"
        direction_label = self._tr(direction_key)
        total_session_frames = len(self._session_frames)
        total_steps = (int(start_index) + 1) if reverse else max(0, total_session_frames - int(start_index))

        runtime_mode = "native-video" if (
            self._native_video_enabled and self._video_predictor is not None and self._video_state is not None
        ) else "fallback-framewise"
        logger.info(
            "[SAM2] Tracking start: direction=%s start=%d steps=%d frames=%d mode=%s reset=%s",
            "backward" if reverse else "forward",
            int(start_index),
            int(total_steps),
            int(total_session_frames),
            runtime_mode,
            bool(reset_session),
        )

        if self._native_video_enabled and self._video_predictor is not None and self._video_state is not None:
            if not hasattr(self._video_predictor, "propagate_in_video"):
                raise RuntimeError("SAM2 video predictor missing propagate_in_video")

            iterator = self._video_predictor.propagate_in_video(self._video_state, reverse=reverse)
            seen = 0
            for step in iterator:
                self._raise_if_cancelled()
                frame_idx = None
                obj_ids = None
                mask_logits = None
                if isinstance(step, tuple) and len(step) >= 3:
                    frame_idx, obj_ids, mask_logits = step[0], step[1], step[2]
                else:
                    continue
                mask = self._mask_from_logits(mask_logits, obj_ids, self._session_obj_id)
                if mask is None:
                    continue
                self._session_masks_map[int(frame_idx)] = mask
                seen += 1
                progress_percent = int((seen / max(1, total_steps)) * 100)
                percent = 40 + int((seen / max(1, total_steps)) * 55)
                self._emit_progress(
                    percent,
                    self._tr("worker_sam2_sequence_frame").format(
                        direction=direction_label,
                        percent=min(100, progress_percent),
                        current=min(total_steps, seen),
                        total=max(1, total_steps),
                    ),
                )
                self._emit_frame_progress(seen, total_session_frames)
        else:
            prompt_points = list(self._session_points)
            prompt_labels = list(self._session_labels)
            if reverse:
                indices = range(int(start_index), -1, -1)
            else:
                indices = range(int(start_index), total_session_frames)
            processed = 0
            for fi in indices:
                self._raise_if_cancelled()
                frame = self._session_frames[fi]
                mask = self.generate_mask(frame, prompt_points, prompt_labels, report_progress=False)
                self._session_masks_map[int(fi)] = np.asarray(mask, dtype=np.uint8)
                processed += 1
                progress_percent = int((processed / max(1, total_steps)) * 100)
                percent = 40 + int((processed / max(1, total_steps)) * 55)
                self._emit_progress(
                    percent,
                    self._tr("worker_sam2_sequence_frame").format(
                        direction=direction_label,
                        percent=min(100, progress_percent),
                        current=min(total_steps, processed),
                        total=max(1, total_steps),
                    ),
                )
                self._emit_frame_progress(processed, total_session_frames)

        self._emit_progress(100, self._tr("sam2_sequence_ready").format(count=len(self._session_masks_map)))
        logger.info(
            "[SAM2] Tracking complete: direction=%s masks=%d",
            "backward" if reverse else "forward",
            len(self._session_masks_map),
        )
        return {
            "sequence_masks_map": {
                int(k): (np.asarray(v, dtype=np.uint8) > 0).astype(np.uint8) * 255
                for k, v in self._session_masks_map.items()
            },
            "current_frame_index": int(start_index),
            "total_frames": total_session_frames,
        }

    @staticmethod
    def _best_from_candidates(masks, scores):
        best_idx = int(np.argmax(scores))
        return masks[best_idx], float(scores[best_idx]), best_idx

    @staticmethod
    def _binarize_mask(mask, threshold: float) -> np.ndarray:
        arr = np.asarray(mask, dtype=np.float32)
        return (arr > float(threshold)).astype(np.uint8) * 255

    def generate_mask(self, image_rgb, points, labels, multimask_output=True, auto_refine=True, report_progress=True):
        self._raise_if_cancelled()
        if report_progress:
            self._emit_progress(20, self._tr("worker_sam2_load_model"))
        self._load_native_predictor()

        if self._native_enabled and self._predictor is not None:
            try:
                point_coords = np.array(points, dtype=np.float32)
                point_labels = np.array(labels, dtype=np.int32)

                self._predictor.set_image(image_rgb)
                if report_progress:
                    self._emit_progress(90, self._tr("worker_sam_prepare_segmentation"))
                masks, scores, logits = self._predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=multimask_output,
                    return_logits=True,
                )
                mask_threshold = float(getattr(self._predictor, "mask_threshold", 0.0))
                best_mask, best_score, best_idx = self._best_from_candidates(masks, scores)

                if auto_refine:
                    try:
                        mask_input = logits[best_idx:best_idx + 1, :, :]
                        masks_r, scores_r, _ = self._predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            mask_input=mask_input,
                            multimask_output=multimask_output,
                            return_logits=True,
                        )
                        refined_mask, refined_score, _ = self._best_from_candidates(masks_r, scores_r)
                        if refined_score > best_score:
                            return self._binarize_mask(refined_mask, mask_threshold)
                    except Exception:
                        pass

                return self._binarize_mask(best_mask, mask_threshold)
            except Exception as exc:
                raise RuntimeError(f"SAM2 image predictor failed: {exc}") from exc

        reason = self._native_failed_reason or "native image predictor unavailable"
        raise RuntimeError(f"SAM2 unavailable: {reason}")

    def generate_mask_sequence(
        self,
        frames_rgb,
        points,
        labels,
        *,
        start_index: int = 0,
        multimask_output: bool = True,
        auto_refine: bool = True,
    ) -> dict:
        frames = list(frames_rgb or [])
        if not frames:
            raise ValueError("frames_rgb is empty")

        masks: list[np.ndarray] = []
        total = len(frames)
        for i, frame in enumerate(frames):
            percent = 25 + int((i / max(1, total)) * 70)
            self._emit_progress(
                percent,
                self._tr("worker_sam2_sequence_frame").format(current=i + 1, total=total),
            )
            mask = self.generate_mask(
                frame,
                points,
                labels,
                multimask_output=multimask_output,
                auto_refine=auto_refine,
            )
            masks.append(np.asarray(mask, dtype=np.uint8))

        self._emit_progress(100, self._tr("sam2_sequence_ready").format(count=total))
        return {
            "sequence_masks": masks,
            "start_index": int(start_index),
        }

    def unload(self) -> None:
        self._predictor = None
        self._native_enabled = False
        self._video_predictor = None
        self._native_video_enabled = False
        self.reset_video_session()
