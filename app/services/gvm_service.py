"""GVM (Generative Video Matting) service for alpha hint generation.

GVM is a diffusion-based video matting model (SIGGRAPH 2025) that generates
temporally consistent alpha hints for CorridorKey without requiring manual
interaction. Based on Stable Video Diffusion backbone.

License: BSD-2-Clause (academic use only).
Model weights: https://huggingface.co/geyongtao/gvm
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

_GVM_WEIGHTS_DIR = Path(__file__).parent.parent.parent / "gvm_core" / "weights"


class _Float32PipeWrapper:
    """Wraps a diffusers pipeline so float16 input tensors are up-cast to float32.

    gvm_core's process_sequence hardcodes ``batch.to(device, dtype=torch.float16)``
    which hangs on CPU (no float16 attention kernels).  Wrapping the pipeline
    intercepts the first positional tensor argument and converts it to float32
    while leaving everything else unchanged.

    Additionally, GVMPipeline.__call__ contains ``self.vae.to(dtype=torch.float16)``
    which re-casts the VAE to float16 on every forward pass.  We patch the VAE's
    ``to`` method so dtype=torch.float16 calls are silently ignored, keeping the
    VAE in float32 throughout inference.
    """

    def __init__(self, pipe):
        import torch
        self._pipe = pipe

        # Patch VAE so pipeline's internal `self.vae.to(dtype=torch.float16)`
        # is a no-op — without this the VAE silently switches back to float16
        # on every forward pass, causing CPU hangs.
        vae = getattr(pipe, "vae", None)
        if vae is not None:
            _original_vae_to = vae.to.__func__ if hasattr(vae.to, "__func__") else None
            _vae_ref = vae

            def _safe_vae_to(*args, **kwargs):
                # Drop any request to cast to float16; allow float32 and device moves
                new_dtype = kwargs.get("dtype", None)
                if len(args) >= 1 and isinstance(args[0], torch.dtype):
                    new_dtype = args[0]
                if new_dtype == torch.float16:
                    return _vae_ref
                return type(_vae_ref).to(_vae_ref, *args, **kwargs)

            import types as _types
            vae.to = _types.MethodType(lambda self_, *a, **kw: _safe_vae_to(*a, **kw), vae)

    def __call__(self, frames, **kwargs):
        import torch
        if isinstance(frames, torch.Tensor) and frames.dtype == torch.float16:
            frames = frames.float()
        return self._pipe(frames, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._pipe, name)


def _patch_upblock_upsample_size() -> None:
    """Monkey-patch diffusers *UpBlockSpatioTemporal.forward to accept upsample_size.

    gvm_core's UNetSpatioTemporalConditionModel calls:
        upsample_block(..., upsample_size=down_block_res_samples[-1].shape[2:])

    In old diffusers the block's forward passed this size to its upsampler:
        upsampler(hidden_states, upsample_size)
    so Upsample2D.forward used F.interpolate(..., size=upsample_size) instead
    of scale_factor=2.0.  Without this, odd-sized inputs produce an off-by-one
    spatial mismatch (e.g. 58 vs 57) at the skip-connection torch.cat.

    Diffusers ≥ 0.29 removed the parameter entirely.  We restore the behaviour:
    the patched forward temporarily nullifies self.upsamplers so the original
    forward body skips upsampling, then calls them manually with the target size.

    Uses the already-cached module object when available to avoid re-triggering
    torch dispatch library registration (which conflicts when sys.modules is
    temporarily reset by unittest.mock.patch.dict in tests).
    """
    import sys

    # Use the cached module if already loaded to avoid re-import side effects.
    _mod = sys.modules.get("diffusers.models.unets.unet_3d_blocks")
    if _mod is None:
        try:
            import diffusers.models.unets.unet_3d_blocks as _mod
        except (ImportError, RuntimeError):
            return  # diffusers not available or torch dispatch table conflict

    _CLASSES = ["UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal"]
    for _cls_name in _CLASSES:
        _cls = getattr(_mod, _cls_name, None)
        if _cls is None:
            continue
        if getattr(_cls, "_keyflow_upsample_patched", False):
            continue  # already patched

        _original_forward = _cls.forward

        def _make_patched(orig):
            def _patched_forward(self, hidden_states, res_hidden_states_tuple,
                                 temb=None, upsample_size=None, **kwargs):
                # Temporarily detach upsamplers so orig() skips them.
                # We re-apply them manually below with the correct target size.
                saved_upsamplers = self.upsamplers
                self.upsamplers = None
                try:
                    hidden_states = orig(
                        self, hidden_states, res_hidden_states_tuple,
                        temb=temb, **kwargs
                    )
                finally:
                    self.upsamplers = saved_upsamplers
                # Apply upsamplers with upsample_size so spatial dims align with
                # the next block's skip connection (fixes 58 vs 57 mismatch).
                if saved_upsamplers is not None:
                    for upsampler in saved_upsamplers:
                        hidden_states = upsampler(hidden_states, upsample_size)
                return hidden_states
            return _patched_forward

        _cls.forward = _make_patched(_original_forward)
        _cls._keyflow_upsample_patched = True
        logger.info("GVM: patched %s.forward to properly handle upsample_size", _cls_name)


class GVMService:
    """Singleton wrapper around GVMProcessor for use in the KeyFlow Studio node graph.

    GVM processes a full frame sequence at once (diffusion model with temporal
    attention), so unlike BiRefNet we batch the entire clip into process_sequence()
    and collect per-frame alpha PNGs from disk.

    Thread-safe: designed for QThread workers.
    """

    _instance: Optional[GVMService] = None
    _processor = None
    _RUNTIME_NOTICE: str = ""

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.logger = logger
        self.progress_callback: Optional[Callable[[int, str], None]] = None
        self.translate: Optional[Callable[[str], str]] = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_callbacks(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        translate: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.progress_callback = progress_callback
        self.translate = translate

    def _tr(self, key: str) -> str:
        if self.translate is not None:
            return self.translate(key)
        return key

    def _emit_progress(self, percent: int, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(percent, message)

    @classmethod
    def consume_runtime_notice(cls) -> str:
        note = str(cls._RUNTIME_NOTICE or "").strip()
        cls._RUNTIME_NOTICE = ""
        return note

    @classmethod
    def get_weights_status(cls) -> dict:
        """Return dict with 'state': 'ready' | 'missing'."""
        weights_dir = _GVM_WEIGHTS_DIR
        vae_ok = (weights_dir / "vae" / "diffusion_pytorch_model.safetensors").is_file()
        unet_ok = (weights_dir / "unet" / "diffusion_pytorch_model.safetensors").is_file()
        if vae_ok and unet_ok:
            return {"state": "ready", "path": str(weights_dir)}
        return {"state": "missing", "path": str(weights_dir)}

    @classmethod
    def ensure_weights_available(
        cls,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Path:
        """Download GVM weights from HuggingFace if not present."""
        from huggingface_hub import snapshot_download

        weights_dir = _GVM_WEIGHTS_DIR
        status = cls.get_weights_status()
        if status["state"] == "ready":
            return weights_dir

        if progress_callback:
            progress_callback(5, "Downloading GVM weights (~5-7 GB)…")

        logger.info("Downloading GVM weights from geyongtao/gvm …")

        def _hf_progress(transferred, total):
            if total and progress_callback:
                pct = int(transferred / total * 90)
                progress_callback(pct, f"Downloading GVM: {transferred // 1024 // 1024} MB / {total // 1024 // 1024} MB")

        snapshot_download(
            repo_id="geyongtao/gvm",
            local_dir=str(weights_dir),
            tqdm_class=None,
        )

        if progress_callback:
            progress_callback(100, "GVM weights ready")

        return weights_dir

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _ensure_gvm_core_importable(self) -> None:
        """Add gvm_core parent directory to sys.path if needed."""
        gvm_core_parent = str(Path(__file__).parent.parent.parent)
        if gvm_core_parent not in sys.path:
            sys.path.insert(0, gvm_core_parent)

    def load_model(self, device: str | None = None) -> None:
        """Load GVMProcessor if not already loaded."""
        if self._processor is not None:
            return

        if device is None:
            from app.utils import get_device
            device = get_device()

        # GVM's diffusers video-UNet is hardcoded to float16.
        # CPU does not support float16 attention kernels → hang.
        # MPS (Apple Silicon) does not support float16 diffusers ops → hang.
        # Fix: keep the selected device but convert the pipeline to float32
        # after loading, and patch the VAE so its internal
        # self.vae.to(dtype=float16) call inside __call__ is a no-op.
        _need_float32_patch = str(device) in ("cpu", "mps")
        if _need_float32_patch:
            logger.info(f"GVM: device={device} — will run in float32 mode")

        self._ensure_gvm_core_importable()

        status = self.get_weights_status()
        if status["state"] != "ready":
            raise RuntimeError(
                self._tr("gvm_weights_missing").format(path=status["path"])
            )

        try:
            from gvm_core import GVMProcessor  # type: ignore
        except ImportError as exc:
            raise ImportError(
                f"gvm_core not found. Make sure gvm_core/ is in the project root. Original error: {exc}"
            ) from exc

        weights_dir = status["path"]
        logger.info(f"GVM: loading model on {device} from {weights_dir}")
        self._emit_progress(10, self._tr("gvm_loading_model"))

        self._processor = GVMProcessor(
            model_base=weights_dir,
            unet_base=str(Path(weights_dir) / "unet"),
            lora_base=str(Path(weights_dir) / "unet"),
            device=device,
        )

        # On CPU/MPS: convert pipeline to float32 and patch VAE to prevent
        # GVMPipeline.__call__'s internal self.vae.to(dtype=float16) from
        # reverting the model back to float16 on every forward pass.
        if _need_float32_patch:
            import torch
            self._processor.pipe = self._processor.pipe.to(torch.float32)
            self._processor.pipe = _Float32PipeWrapper(self._processor.pipe)
            logger.info(f"GVM: converted pipeline to float32 for {device} inference")

        # gvm_core's UNetSpatioTemporalConditionModel calls upsample_block(
        # ..., upsample_size=...) but diffusers UpBlockSpatioTemporal.forward()
        # dropped that parameter in 0.30+.  Patch it to accept and ignore it.
        _patch_upblock_upsample_size()

        logger.info("GVM: model loaded")
        self._emit_progress(30, self._tr("gvm_model_ready"))

    def unload(self) -> None:
        """Free model from memory (GPU/MPS/CPU)."""
        if self._processor is not None:
            del self._processor
            self._processor = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
            logger.info("GVM: model unloaded")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process_sequence(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        *,
        num_frames_per_batch: int = 8,
        denoise_steps: int = 1,
        decode_chunk_size: int = 4,
        num_overlap_frames: int = 1,
        num_interp_frames: int = 1,
        noise_type: str = "zeros",
        use_clip_img_emb: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        per_batch_callback: Optional[Callable[[list, int], None]] = None,
    ) -> list[Path]:
        """Run GVM on input_path and write alpha PNGs to output_dir.

        Returns sorted list of generated alpha PNG paths.

        Parameters
        ----------
        input_path:
            Video file or directory of image frames.
        output_dir:
            Directory where alpha PNGs will be written.
        num_frames_per_batch:
            Frames processed per diffusion step. Reduce if OOM.
        decode_chunk_size:
            VAE decode chunk. Reduce if OOM.
        num_overlap_frames:
            Overlap between batches for temporal consistency.
        num_interp_frames:
            Interpolation frames between batches.
        noise_type:
            Noise initialization type passed to gvm_core ("zeros" or "random").
        use_clip_img_emb:
            Use CLIP image embedding for conditioning (passed to gvm_core).
        progress_callback:
            Called with (done_frames, total_frames) after each batch.
        per_batch_callback:
            Called after each batch is written to output_dir with
            (new_paths: list[Path], start_index: int). Used for progressive
            streaming to Write nodes without waiting for full completion.
        """
        if self._processor is None:
            raise RuntimeError("GVM model not loaded. Call load_model() first.")

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Count total input frames upfront so progress callbacks report real numbers.
        video_exts = {".mp4", ".mkv", ".gif", ".mov", ".avi"}
        if input_path.suffix.lower() in video_exts:
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(str(input_path))
            _total_frames = max(1, int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 1))
            _cap.release()
        elif input_path.is_dir():
            _img_exts = {".png", ".jpg", ".jpeg", ".exr"}
            _total_frames = max(1, len([f for f in input_path.iterdir() if f.suffix.lower() in _img_exts]))
        else:
            _total_frames = 1

        logger.info(
            f"GVM: processing {input_path.name} → {output_dir} "
            f"(batch={num_frames_per_batch}, overlap={num_overlap_frames}, "
            f"total_frames={_total_frames})"
        )

        # Track how many paths have been reported to per_batch_callback.
        _reported_count: list[int] = [0]

        def _batch_progress(completed_batches: int, total_batches: int) -> None:
            # Estimate done frames from batch count. This keeps UI progress moving
            # even while disk writes lag behind diffusion completion.
            done_frames = min(completed_batches * num_frames_per_batch, _total_frames)

            if progress_callback:
                try:
                    progress_callback(done_frames, _total_frames)
                except Exception as _cb_exc:
                    logger.warning("GVM progress_callback error: %s", _cb_exc)
            if self.progress_callback:
                try:
                    pct = 30 + int(done_frames / max(_total_frames, 1) * 65)
                    self.progress_callback(pct, self._tr("gvm_processing_batch").format(
                        current=done_frames, total=_total_frames
                    ))
                except Exception as _cb_exc:
                    logger.warning("GVM internal progress callback error: %s", _cb_exc)
            # Report newly written frames to per_batch_callback (glob only here).
            if per_batch_callback is not None:
                try:
                    current_paths = sorted(output_dir.glob("*.png"))
                    start = _reported_count[0]
                    new_paths = current_paths[start:]
                    if new_paths:
                        per_batch_callback(new_paths, start)
                        _reported_count[0] += len(new_paths)
                except Exception as _cb_exc:
                    logger.warning("GVM per_batch_callback error: %s", _cb_exc)

        self._processor.process_sequence(
            input_path=str(input_path),
            output_dir=None,
            num_frames_per_batch=num_frames_per_batch,
            denoise_steps=denoise_steps,
            decode_chunk_size=decode_chunk_size,
            num_overlap_frames=num_overlap_frames,
            num_interp_frames=num_interp_frames,
            noise_type=str(noise_type or "zeros"),
            use_clip_img_emb=bool(use_clip_img_emb),
            mode="matte",
            write_video=False,
            direct_output_dir=str(output_dir),
            progress_callback=_batch_progress,
        )

        alpha_paths = sorted(output_dir.glob("*.png"))
        logger.info(f"GVM: generated {len(alpha_paths)} alpha frames")
        self._emit_progress(95, self._tr("gvm_done").format(count=len(alpha_paths)))
        return alpha_paths

    def load_alpha_frames(self, alpha_paths: list[Path]) -> list[np.ndarray]:
        """Load PNG alpha frames from disk as float32 [0..1] arrays."""
        import cv2
        result = []
        for p in alpha_paths:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                result.append(img.astype(np.float32) / 255.0)
            else:
                logger.warning(f"GVM: could not read alpha frame {p}")
        return result
