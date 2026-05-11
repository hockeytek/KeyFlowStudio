"""BiRefNet model service for alpha mask generation.

BiRefNet is a lightweight background removal model used to generate alpha hints
for CorridorKey. This service manages an explicit local cache for supported
Hugging Face presets and downloads only the files required for inference.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from app.utils import get_device, get_model_variant_dir

logger = logging.getLogger(__name__)

try:
    from BiRefNetModule.wrapper import BiRefNetHandler
except ImportError as e:
    BiRefNetHandler = None
    IMPORT_ERROR = str(e)


class _NativeBiRefNetHandler:
    """Built-in BiRefNet fallback when external BiRefNetModule is unavailable."""

    _RESOLUTIONS = {
        "General-Lite-2K": (2560, 1440),
        "General-reso_512": (512, 512),
        "General-HR": (2048, 2048),
        "Matting-HR": (2048, 2048),
    }

    def __init__(self, *, device: str, usage: str, model_dir: str) -> None:
        self.device = device
        self.usage = usage
        self.model_dir = model_dir
        self.use_half = bool(str(device).startswith("cuda"))

        self._fixed_resolution = self._RESOLUTIONS.get(usage)
        if self._fixed_resolution is None and "-dynamic" not in usage:
            self._fixed_resolution = (1024, 1024)

        self._normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self._to_tensor = transforms.ToTensor()
        self._to_pil = transforms.ToPILImage()

        self.model = AutoModelForImageSegmentation.from_pretrained(model_dir, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        if self.use_half:
            self.model.half()

    def _target_resolution(self, image: Image.Image) -> tuple[int, int]:
        if self._fixed_resolution is not None:
            return self._fixed_resolution

        # Dynamic models: keep dimensions divisible by 32.
        w, h = image.size
        rw = max(32, int(w // 32) * 32)
        rh = max(32, int(h // 32) * 32)
        return (rw, rh)

    def predict(self, image: Image.Image) -> np.ndarray:
        orig_w, orig_h = image.size
        target = self._target_resolution(image)

        image_proc = image.resize(target, Image.Resampling.BILINEAR)
        image_proc = self._to_tensor(image_proc)
        image_proc = self._normalize(image_proc).unsqueeze(0).to(self.device)
        if self.use_half:
            image_proc = image_proc.half()

        with torch.no_grad():
            outputs = self.model(image_proc)

        if isinstance(outputs, (tuple, list)):
            logits = outputs[-1]
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs

        if isinstance(logits, (tuple, list)):
            logits = logits[-1]
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        if logits.ndim == 4 and logits.shape[1] > 1:
            logits = logits[:, :1, :, :]

        pred = torch.sigmoid(logits)
        pred = F.interpolate(pred, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        mask = pred[0, 0].float().cpu().numpy().astype(np.float32)
        return np.clip(mask, 0.0, 1.0)


class BiRefNetService:
    """Singleton for BiRefNet model management.

    Provides lazy loading and caching for alpha mask generation.
    Thread-safe: designed to work with QThread workers.
    """

    _instance: Optional[BiRefNetService] = None
    _model: Optional[object] = None
    _lock = None
    _RUNTIME_NOTICE: str = ""

    AVAILABLE_PRESETS = [
        "General",
        "General-dynamic",
        "General-HR",
        "General-Lite",
        "General-Lite-2K",
        "General-reso_512",
        "Matting",
        "Matting-dynamic",
        "Matting-HR",
        "Matting-Lite",
        "Portrait",
        "DIS5K",
        "HRSOD",
        "COD",
    ]

    MANAGED_MODEL_REPOS = {
        "General": "ZhengPeng7/BiRefNet",
        "General-dynamic": "ZhengPeng7/BiRefNet_dynamic",
        "General-HR": "ZhengPeng7/BiRefNet_HR",
        "General-Lite": "ZhengPeng7/BiRefNet_lite",
        "General-Lite-2K": "ZhengPeng7/BiRefNet_lite-2K",
        "General-reso_512": "ZhengPeng7/BiRefNet_512x512",
        "Matting": "ZhengPeng7/BiRefNet-matting",
        "Matting-dynamic": "ZhengPeng7/BiRefNet_dynamic-matting",
        "Matting-HR": "ZhengPeng7/BiRefNet_HR-matting",
        "Matting-Lite": "ZhengPeng7/BiRefNet_lite-matting",
        "Portrait": "ZhengPeng7/BiRefNet-portrait",
        "DIS5K": "ZhengPeng7/BiRefNet-DIS5K",
        "HRSOD": "ZhengPeng7/BiRefNet-HRSOD",
        "COD": "ZhengPeng7/BiRefNet-COD",
    }

    MODEL_ALLOW_PATTERNS = [
        "config.json",
        "*.py",
        "*.safetensors",
        "*.bin",
        "*.pth",
    ]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.device = self._select_device()
        self.logger = logger
        self.current_preset = None
        self._runtime_force_device: str | None = None
        self.progress_callback: Optional[Callable[[int, str], None]] = None
        self.translate: Optional[Callable[[str], str]] = None

    @classmethod
    def consume_runtime_notice(cls) -> str:
        note = str(cls._RUNTIME_NOTICE or "").strip()
        cls._RUNTIME_NOTICE = ""
        return note

    @staticmethod
    def _is_mps_runtime_error(exc: Exception) -> bool:
        message = str(exc)
        markers = [
            "MPS backend out of memory",
            "not currently implemented for the MPS device",
            "BFloat16 is not supported on MPS",
            "unsupported autocast device_type 'mps'",
            "Input type (MPS",
        ]
        return any(marker in message for marker in markers)

    @staticmethod
    def _use_external_module() -> bool:
        raw = os.environ.get("MATANYONE_USE_EXTERNAL_BIREFNETMODULE", "").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _select_device() -> str:
        device = get_device()
        device_str = device.type
        logger.info(f"BiRefNet: Using device: {device_str}")
        return device_str

    def set_callbacks(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        translate: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.progress_callback = progress_callback
        self.translate = translate

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

    @classmethod
    def get_cache_root(cls) -> Path:
        return get_model_variant_dir("birefnet", "hf-cache")

    @classmethod
    def _usage_env_key(cls, usage: str) -> str:
        return "MATANYONE_BIREFNET_REPO_" + "".join(ch if ch.isalnum() else "_" for ch in usage.upper())

    @classmethod
    def get_managed_repo_id(cls, usage: str) -> Optional[str]:
        override = os.environ.get(cls._usage_env_key(usage), "").strip()
        if override:
            return override
        return cls.MANAGED_MODEL_REPOS.get(usage)

    @classmethod
    def _repo_cache_dir(cls, repo_id: str) -> Path:
        return cls.get_cache_root() / f"models--{repo_id.replace('/', '--')}"

    @classmethod
    def _dir_has_model_files(cls, path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        has_config = (path / "config.json").exists()
        # Native runtime uses transformers with trust_remote_code=True, so model python files are required.
        has_remote_code = any(path.glob("*.py"))
        has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin")) or any(path.glob("*.pth"))
        return has_config and has_remote_code and has_weights

    @classmethod
    def find_cached_snapshot(cls, usage: str) -> Optional[Path]:
        repo_id = cls.get_managed_repo_id(usage)
        if repo_id is None:
            return None

        snapshots_dir = cls._repo_cache_dir(repo_id) / "snapshots"
        if not snapshots_dir.exists():
            return None

        candidates = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            if cls._dir_has_model_files(candidate):
                return candidate
        return None

    @classmethod
    def get_weight_status(cls, usage: str) -> dict:
        cache_root = cls.get_cache_root()
        repo_id = cls.get_managed_repo_id(usage)
        snapshot_path = cls.find_cached_snapshot(usage)
        if repo_id is None:
            return {
                "state": "external",
                "repo_id": None,
                "snapshot_path": None,
                "cache_dir": str(cache_root),
            }
        return {
            "state": "ready" if snapshot_path is not None else "missing",
            "repo_id": repo_id,
            "snapshot_path": str(snapshot_path) if snapshot_path is not None else None,
            "cache_dir": str(cache_root),
        }

    def _configure_hf_cache_env(self) -> None:
        cache_root = self.get_cache_root()
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_root.parent))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root))

    def _make_hf_tqdm_class(self, base_bytes: int, total_bytes: int):
        service = self
        bounded_total = max(int(total_bytes), 1)

        class DownloadTqdm(tqdm):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("leave", False)
                kwargs.pop("name", None)
                super().__init__(*args, **kwargs)
                self._last_percent = -1

            def update(self, n=1):
                result = super().update(n)
                current = min(bounded_total, base_bytes + int(self.n))
                percent = int(current * 100 / bounded_total)
                if percent != self._last_percent:
                    stage = 24 + int(percent * 36 / 100)
                    service._emit_progress(
                        stage,
                        service._tr("worker_birefnet_download_weights").format(percent=percent),
                    )
                    self._last_percent = percent
                return result

        return DownloadTqdm

    def _download_managed_weights(self, usage: str) -> Path:
        repo_id = self.get_managed_repo_id(usage)
        if repo_id is None:
            raise RuntimeError(f"No managed BiRefNet repository configured for preset '{usage}'")

        self._configure_hf_cache_env()
        cache_root = self.get_cache_root()
        cache_root.mkdir(parents=True, exist_ok=True)

        try:
            planned_files = snapshot_download(
                repo_id,
                cache_dir=cache_root,
                allow_patterns=self.MODEL_ALLOW_PATTERNS,
                dry_run=True,
            )

            files_to_download = [info for info in planned_files if getattr(info, "will_download", True)]
            total_bytes = sum(int(getattr(info, "file_size", 0) or 0) for info in files_to_download)

            downloaded_bytes = 0
            for info in files_to_download:
                hf_hub_download(
                    repo_id,
                    info.filename,
                    cache_dir=cache_root,
                    tqdm_class=self._make_hf_tqdm_class(downloaded_bytes, total_bytes),
                )
                downloaded_bytes += int(getattr(info, "file_size", 0) or 0)

        except TypeError as exc:
            if "dry_run" not in str(exc):
                raise RuntimeError(f"Failed to inspect BiRefNet repository '{repo_id}': {exc}") from exc

            # huggingface_hub<0.23 does not support dry_run.
            self._emit_progress(28, self._tr("worker_birefnet_download_weights").format(percent=0))
            try:
                snapshot_download(
                    repo_id,
                    cache_dir=cache_root,
                    allow_patterns=self.MODEL_ALLOW_PATTERNS,
                )
            except HfHubHTTPError as inner_exc:
                raise RuntimeError(
                    f"BiRefNet weights are unavailable for preset '{usage}' from {repo_id}: {inner_exc}"
                ) from inner_exc
            except Exception as inner_exc:
                raise RuntimeError(f"Failed to download BiRefNet repository '{repo_id}': {inner_exc}") from inner_exc
            self._emit_progress(60, self._tr("worker_birefnet_download_weights").format(percent=100))

        except HfHubHTTPError as exc:
            raise RuntimeError(f"BiRefNet weights are unavailable for preset '{usage}' from {repo_id}: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to inspect BiRefNet repository '{repo_id}': {exc}") from exc

        snapshot_path = self.find_cached_snapshot(usage)
        if snapshot_path is None:
            raise RuntimeError(
                f"BiRefNet weights were downloaded from {repo_id}, but the local snapshot was not found in {cache_root}"
            )

        self._emit_progress(62, self._tr("worker_birefnet_weights_downloaded"))
        return snapshot_path

    def ensure_weights_available(self, usage: str = "General") -> Optional[Path]:
        if usage not in self.AVAILABLE_PRESETS:
            raise ValueError(
                f"Unknown BiRefNet preset '{usage}'. "
                f"Available: {', '.join(self.AVAILABLE_PRESETS)}"
            )

        managed_repo_id = self.get_managed_repo_id(usage)
        self._emit_progress(18, self._tr("worker_birefnet_prepare"))
        if managed_repo_id is None:
            self._emit_progress(40, self._tr("worker_birefnet_external_weights"))
            return None

        self._emit_progress(22, self._tr("worker_birefnet_check_weights"))
        snapshot_path = self.find_cached_snapshot(usage)
        if snapshot_path is None:
            snapshot_path = self._download_managed_weights(usage)
        else:
            self._emit_progress(62, self._tr("worker_birefnet_weights_ready"))
        return snapshot_path

    def load_model(
        self,
        usage: str = "General",
        force_reload: bool = False,
    ) -> BiRefNetHandler:
        # Fast path: when the model for the same preset is already loaded,
        # avoid calling get_device() repeatedly (it logs KEYFLOW_DEVICE warning).
        if self._model is not None and self.current_preset == usage and not force_reload:
            return self._model

        current_device = self._runtime_force_device or get_device().type
        if current_device != self.device:
            logger.info(
                f"BiRefNet: Device changed {self.device} -> {current_device}, reloading model"
            )
            self.device = current_device
            force_reload = True

        if usage not in self.AVAILABLE_PRESETS:
            raise ValueError(
                f"Unknown BiRefNet preset '{usage}'. "
                f"Available: {', '.join(self.AVAILABLE_PRESETS)}"
            )

        if self._model is not None and self.current_preset == usage and not force_reload:
            return self._model

        if BiRefNetService._lock is None:
            import threading

            BiRefNetService._lock = threading.Lock()

        with BiRefNetService._lock:
            if self._model is not None and self.current_preset == usage and not force_reload:
                return self._model

            managed_repo_id = self.get_managed_repo_id(usage)
            snapshot_path = None
            if managed_repo_id is not None:
                snapshot_path = self.ensure_weights_available(usage)
                if snapshot_path is not None:
                    logger.info(f"BiRefNet: Using cached weights for '{usage}' at {snapshot_path}")
                self._configure_hf_cache_env()

            self._emit_progress(68, self._tr("worker_birefnet_load_model"))
            logger.info(f"BiRefNet: Loading model preset '{usage}'")

            try:
                if self._use_external_module():
                    if BiRefNetHandler is None:
                        raise RuntimeError(
                            "MATANYONE_USE_EXTERNAL_BIREFNETMODULE=1, but BiRefNetModule is not available in environment. "
                            f"Import error: {IMPORT_ERROR}"
                        )
                    self._model = BiRefNetHandler(
                        device=self.device,
                        usage=usage,
                    )
                else:
                    if snapshot_path is None:
                        raise RuntimeError(
                            "BiRefNetModule is unavailable and no managed repository is configured for this preset."
                        )
                    self._model = _NativeBiRefNetHandler(
                        device=self.device,
                        usage=usage,
                        model_dir=str(snapshot_path),
                    )
                self.current_preset = usage
                logger.info(f"BiRefNet: Model '{usage}' loaded successfully")
                self._emit_progress(72, self._tr("worker_birefnet_model_ready"))
                return self._model

            except Exception as e:
                logger.error(f"BiRefNet: Failed to load model '{usage}': {e}")
                raise

    def process_image(
        self,
        image: np.ndarray,
        usage: str = "General",
        half_precision: bool = False,
        _is_retry: bool = False,
    ) -> np.ndarray:
        if not _is_retry:
            BiRefNetService._RUNTIME_NOTICE = ""
        model = self.load_model(usage=usage)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Image must have shape (H, W, 3), got {image.shape}"
            )

        if image.dtype != np.uint8:
            raise ValueError(
                f"Image must be uint8, got {image.dtype}"
            )

        logger.debug(
            f"BiRefNet: Processing {image.shape} with preset '{usage}', "
            f"half_precision={half_precision}"
        )

        try:
            pil_image = Image.fromarray(image, mode="RGB")
            with torch.inference_mode():
                mask = model.predict(pil_image)

            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            mask = np.asarray(mask, dtype=np.float32)

            if mask.max() > 1.0:
                mask = mask / 255.0

            logger.debug(
                f"BiRefNet: Mask generated, shape={mask.shape}, range=[{mask.min():.3f}, {mask.max():.3f}]"
            )
            return mask

        except Exception as e:
            if self.device == "mps" and self._is_mps_runtime_error(e) and not _is_retry:
                logger.warning(
                    "BiRefNet: MPS inference failed (%s). Falling back to CPU and retrying once.",
                    e,
                )
                BiRefNetService._RUNTIME_NOTICE = str(e)
                self._runtime_force_device = "cpu"
                self.unload_model()
                return self.process_image(
                    image=image,
                    usage=usage,
                    half_precision=half_precision,
                    _is_retry=True,
                )
            logger.error(f"BiRefNet: Inference failed: {e}")
            raise RuntimeError(f"BiRefNet inference failed: {e}") from e

    def unload_model(self):
        if self._model is not None:
            self._model = None
            self.current_preset = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            logger.info("BiRefNet: Model unloaded from memory")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device_name(self) -> str:
        if self.device == "cuda":
            try:
                return f"CUDA: {torch.cuda.get_device_name(0)}"
            except Exception:
                return "CUDA"
        if self.device == "mps":
            return "Apple Metal (MPS)"
        return "CPU"
