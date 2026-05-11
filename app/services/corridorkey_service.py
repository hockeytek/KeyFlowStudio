"""CorridorKey model service — lazy loading, device management, and caching.

This service manages the CorridorKey neural network for professional green screen removal.
It follows the Singleton pattern for efficient model caching in memory.
"""
from __future__ import annotations

import logging
import os
import shutil
import inspect
from pathlib import Path
from typing import Callable, Optional

import torch
import numpy as np
import requests
from huggingface_hub import hf_hub_url

from app.utils import get_device, get_model_variant_dir

logger = logging.getLogger(__name__)

# Try to import CorridorKey (will be available if installed from /Volumes/MAC MEDIA/Temp/CorridorKey)
try:
    from CorridorKeyModule.inference_engine import CorridorKeyEngine
except ImportError as e:
    CorridorKeyEngine = None
    IMPORT_ERROR = str(e)


class CorridorKeyService:
    """Singleton for CorridorKey model management.
    
    Provides lazy loading, device selection, and safe caching for multi-threaded use.
    Thread-safe: designed to work with QThread workers.
    """
    
    _instance: Optional[CorridorKeyService] = None
    _engine: Optional[CorridorKeyEngine] = None
    _lock = None  # threading.Lock, created on first access
    CHECKPOINT_REPO_ID = "nikopueringer/CorridorKey_v1.0"
    CHECKPOINT_REPO_ID_BLUE = "nikopueringer/CorridorKeyBlue_1.0"
    CHECKPOINT_FILENAMES_GREEN = (
        "CorridorKey_v1.0.safetensors",
        "CorridorKey_v1.0.pth",
        "CorridorKey.pth",
        "corridorkey.pth",
    )
    CHECKPOINT_FILENAMES_BLUE = (
        "CorridorKeyBlue_1.0.safetensors",
        "CorridorKeyBlue_1.0.pth",
    )
    CHECKPOINT_FILENAMES = CHECKPOINT_FILENAMES_GREEN
    CHECKPOINT_VARIANT = "v1.0"
    _RUNTIME_NOTICE: str = ""
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize service (called once due to Singleton pattern)."""
        if self._initialized:
            return
        
        self._initialized = True
        self.device = self._select_device()
        self.checkpoint_path = None  # Will be lazy-loaded
        self._checkpoint_paths: dict[str, str] = {}
        self.engine_use_refiner: bool | None = None
        self.engine_screen_color: str | None = None
        self.logger = logger

    @classmethod
    def consume_runtime_notice(cls) -> str:
        note = str(cls._RUNTIME_NOTICE or "").strip()
        cls._RUNTIME_NOTICE = ""
        return note
    
    
    @staticmethod
    def _select_device() -> torch.device:
        """Select device using global setting (respects KEYFLOW_DEVICE env var).
        
        Returns:
            torch.device: Selected device for inference
        """
        device = get_device()
        logger.info(f"CorridorKey: Using device: {device}")
        return device
    @staticmethod
    def _normalize_screen_color(screen_color: str | None, *, allow_auto: bool = False) -> str:
        color = str(screen_color or "green").strip().lower()
        allowed = {"green", "blue"} | ({"auto"} if allow_auto else set())
        if color not in allowed:
            valid = ", ".join(sorted(allowed))
            raise ValueError(f"Unknown CorridorKey screen_color {color!r}. Valid: {valid}")
        return color

    @classmethod
    def _checkpoint_filenames(cls, screen_color: str) -> tuple[str, ...]:
        color = cls._normalize_screen_color(screen_color)
        return cls.CHECKPOINT_FILENAMES_BLUE if color == "blue" else cls.CHECKPOINT_FILENAMES_GREEN

    @classmethod
    def _checkpoint_repo_id(cls, screen_color: str) -> str:
        color = cls._normalize_screen_color(screen_color)
        return cls.CHECKPOINT_REPO_ID_BLUE if color == "blue" else cls.CHECKPOINT_REPO_ID

    @staticmethod
    def _screen_channel_for_color(screen_color: str) -> int:
        return 2 if CorridorKeyService._normalize_screen_color(screen_color) == "blue" else 1

    @classmethod
    def _checkpoint_candidates(cls, screen_color: str = "green") -> list[Path]:
        """Return ordered checkpoint candidates for local lookup."""
        models_dir = get_model_variant_dir("corridorkey", cls.CHECKPOINT_VARIANT)
        return [models_dir / filename for filename in cls._checkpoint_filenames(screen_color)]

    @classmethod
    def find_local_checkpoint(cls, screen_color: str = "green") -> Path | None:
        """Return local checkpoint path if present, else None."""
        for path in cls._checkpoint_candidates(screen_color):
            if path.exists():
                return path.resolve()
        return None

    @classmethod
    def get_checkpoint_status(cls, screen_color: str = "green") -> dict:
        """Return checkpoint availability status for UI."""
        color = cls._normalize_screen_color(screen_color, allow_auto=True)
        green = cls.find_local_checkpoint("green")
        blue = cls.find_local_checkpoint("blue")
        if color == "auto":
            ready = green is not None and blue is not None
            local = green if ready else None
        elif color == "blue":
            ready = blue is not None
            local = blue
        else:
            ready = green is not None
            local = green
        return {
            "state": "ready" if ready else "missing",
            "path": str(local) if local is not None else None,
            "repo_id": cls._checkpoint_repo_id("green" if color == "auto" else color),
            "screen_color": color,
            "green_ready": green is not None,
            "blue_ready": blue is not None,
        }

    @classmethod
    def _emit_download_progress(
        cls,
        progress_callback: Optional[Callable[[int, str], None]],
        percent: int,
        message: str,
    ) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(max(0, min(100, int(percent))), message)
        except Exception:
            pass

    @classmethod
    def _download_checkpoint_file(
        cls,
        repo_id: str,
        filename: str,
        destination_dir: Path,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Path:
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / filename
        temp_path = destination_dir / f"{filename}.part"
        url = hf_hub_url(repo_id=repo_id, filename=filename)

        cls._emit_download_progress(progress_callback, 0, f"CorridorKey {filename}: 0%")

        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total_bytes = int(response.headers.get("Content-Length", "0") or 0)
            downloaded_bytes = 0
            last_percent = -1

            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded_bytes += len(chunk)

                    if total_bytes > 0:
                        percent = int(downloaded_bytes * 100 / total_bytes)
                        if percent != last_percent:
                            last_percent = percent
                            cls._emit_download_progress(
                                progress_callback,
                                percent,
                                f"CorridorKey {filename}: {percent}%",
                            )

        shutil.move(str(temp_path), str(destination_path))
        cls._emit_download_progress(progress_callback, 100, f"CorridorKey {filename}: 100%")
        return destination_path.resolve()

    @classmethod
    def ensure_checkpoint_available(
        cls,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        screen_color: str = "green",
    ) -> Path:
        """Ensure CorridorKey checkpoint exists locally, downloading when missing."""
        color = cls._normalize_screen_color(screen_color, allow_auto=True)
        if color == "auto":
            green = cls.ensure_checkpoint_available(progress_callback, screen_color="green")
            cls.ensure_checkpoint_available(progress_callback, screen_color="blue")
            return green

        local = cls.find_local_checkpoint(color)
        if local is not None:
            cls._emit_download_progress(progress_callback, 100, f"CorridorKey {color} checkpoint ready")
            return local

        models_dir = get_model_variant_dir("corridorkey", cls.CHECKPOINT_VARIANT)
        last_error: Exception | None = None
        repo_id = cls._checkpoint_repo_id(color)
        for filename in cls._checkpoint_filenames(color):
            try:
                downloaded = cls._download_checkpoint_file(
                    repo_id=repo_id,
                    filename=filename,
                    destination_dir=models_dir,
                    progress_callback=progress_callback,
                )
                return downloaded
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise RuntimeError(
                f"Failed to download CorridorKey {color} checkpoint from {repo_id}: {last_error}"
            ) from last_error
        raise RuntimeError(f"Failed to download CorridorKey {color} checkpoint from {repo_id}")

    @classmethod
    def _find_checkpoint(cls, screen_color: str = "green") -> str:
        """Find CorridorKey checkpoint from standard locations.
        
        Tries multiple locations in order:
        1. get_model_variant_dir("corridorkey", "v1.0")
        
        Returns:
            str: Path to checkpoint file
        
        Raises:
            FileNotFoundError: If checkpoint not found in any location
        """
        color = cls._normalize_screen_color(screen_color)
        local = cls.find_local_checkpoint(color)
        if local is not None:
            logger.info(f"CorridorKey: Found {color} checkpoint at {local}")
            return str(local)

        # Attempt managed auto-download for autonomous app flow.
        try:
            downloaded = cls.ensure_checkpoint_available(screen_color=color)
            logger.info(f"CorridorKey: Downloaded {color} checkpoint to {downloaded}")
            return str(downloaded)
        except Exception as exc:
            logger.warning(f"CorridorKey: Auto-download failed: {exc}")
        
        # Provide helpful error message
        candidates = cls._checkpoint_candidates(color)
        logger.error(f"CorridorKey: Checkpoint not found. Checked: {[str(p) for p in candidates]}")
        raise FileNotFoundError(
            f"CorridorKey {color} checkpoint not found.\n"
            f"Tried: {', '.join(str(p) for p in candidates)}\n"
            f"Download from: https://huggingface.co/{cls._checkpoint_repo_id(color)}"
        )

    @staticmethod
    def _process_frame_accepts_kw(engine: object, keyword: str) -> bool:
        try:
            sig = inspect.signature(engine.process_frame)  # type: ignore[attr-defined]
        except Exception:
            return False
        return keyword in sig.parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )

    def _resolve_screen_color_for_frame(
        self,
        screen_color: str,
        image: np.ndarray,
        alpha_hint: Optional[np.ndarray],
    ) -> str:
        color = self._normalize_screen_color(screen_color, allow_auto=True)
        if color != "auto":
            return color
        if self._engine is not None and self.engine_screen_color in {"green", "blue"}:
            return str(self.engine_screen_color)
        if alpha_hint is None:
            logger.warning("CorridorKey: auto screen_color has no alpha hint sample; defaulting to green")
            return "green"
        try:
            from CorridorKeyModule.core.color_utils import estimate_screen_color
            detected = str(estimate_screen_color(np.asarray(image), np.asarray(alpha_hint))).strip().lower()
            return detected if detected in {"green", "blue"} else "green"
        except Exception as exc:
            logger.warning("CorridorKey: auto screen_color detection failed (%s); defaulting to green", exc)
            return "green"
    
    def load_engine(
        self,
        force_reload: bool = False,
        use_refiner: bool = True,
        screen_color: str = "green",
    ) -> CorridorKeyEngine:
        """Load or return cached CorridorKey engine.
        
        Uses double-checked locking for thread-safe lazy loading.
        
        Args:
            force_reload: If True, unload and reload the engine
        
        Returns:
            CorridorKeyEngine: Loaded and initialized engine
        
        Raises:
            RuntimeError: If CorridorKey module is not installed
            FileNotFoundError: If checkpoint cannot be found
        """
        # Fast path: already loaded with matching config — skip device check entirely.
        requested_use_refiner = bool(use_refiner)
        requested_screen_color = self._normalize_screen_color(screen_color)
        if (
            self._engine is not None
            and not force_reload
            and self.engine_use_refiner == requested_use_refiner
            and self.engine_screen_color == requested_screen_color
        ):
            return self._engine

        # Refresh device in case user switched runtime/device in settings.
        current_device = get_device()
        forced_device = os.environ.get("KEYFLOW_DEVICE", "").strip().lower()
        if forced_device == "cpu" and current_device.type != "cpu":
            logger.warning(
                "CorridorKey: KEYFLOW_DEVICE=cpu but resolved device is %s; forcing CPU",
                current_device,
            )
            current_device = torch.device("cpu")
        if str(current_device) != str(self.device):
            logger.info(
                f"CorridorKey: Device changed {self.device} -> {current_device}, reloading engine"
            )
            self.device = current_device
            force_reload = True

        if self.engine_use_refiner is not None and self.engine_use_refiner != requested_use_refiner:
            logger.info(
                "CorridorKey: use_refiner changed %s -> %s, reloading engine",
                self.engine_use_refiner,
                requested_use_refiner,
            )
            force_reload = True

        if self.engine_screen_color is not None and self.engine_screen_color != requested_screen_color:
            logger.info(
                "CorridorKey: screen_color changed %s -> %s, reloading engine",
                self.engine_screen_color,
                requested_screen_color,
            )
            force_reload = True
        
        # Thread-safe loading with lock
        if CorridorKeyService._lock is None:
            import threading
            CorridorKeyService._lock = threading.Lock()
        
        with CorridorKeyService._lock:
            # Double-check after acquiring lock
            if (
                self._engine is not None
                and not force_reload
                and self.engine_use_refiner == requested_use_refiner
                and self.engine_screen_color == requested_screen_color
            ):
                return self._engine
            
            if CorridorKeyEngine is None:
                raise RuntimeError(
                    f"CorridorKeyModule not installed or not in PYTHONPATH.\n"
                    f"Error: {IMPORT_ERROR}\n"
                    "Install CorridorKey runtime in current environment, for example:\n"
                    "pip install --no-deps git+https://github.com/nikopueringer/CorridorKey.git\n"
                    "or bundle CorridorKeyModule with the app build."
                )
            
            # Lazy-load checkpoint path if not found yet
            if requested_screen_color not in self._checkpoint_paths:
                self._checkpoint_paths[requested_screen_color] = self._find_checkpoint(requested_screen_color)
            self.checkpoint_path = self._checkpoint_paths[requested_screen_color]
            
            # Verify checkpoint file exists and is readable
            checkpoint_path_obj = Path(self.checkpoint_path)
            if not checkpoint_path_obj.exists():
                raise FileNotFoundError(
                    f"CorridorKey checkpoint not found at {checkpoint_path_obj.absolute()}"
                )
            
            # Check checkpoint file validity
            checkpoint_size = checkpoint_path_obj.stat().st_size
            logger.info(f"CorridorKey: Checkpoint file size: {checkpoint_size / 1024 / 1024:.1f} MB")
            
            if checkpoint_size < 100 * 1024 * 1024:  # Less than 100MB is suspicious
                logger.warning(
                    f"CorridorKey: Checkpoint file is small ({checkpoint_size / 1024 / 1024:.1f} MB), "
                    "may be corrupted or incomplete. Expected ~200-300 MB."
                )
            
            # Try to validate checkpoint file by loading metadata with torch
            try:
                checkpoint_metadata = torch.load(
                    str(checkpoint_path_obj), map_location="cpu", weights_only=False
                )
                if isinstance(checkpoint_metadata, dict):
                    keys = list(checkpoint_metadata.keys())[:5]
                    logger.info(f"CorridorKey: Checkpoint keys preview: {keys}")
                else:
                    logger.info(f"CorridorKey: Checkpoint type: {type(checkpoint_metadata)}")
            except Exception as e:
                logger.warning(f"CorridorKey: Could not validate checkpoint with torch.load: {e}")
            
            logger.info(f"CorridorKey: Loading {requested_screen_color} engine from {self.checkpoint_path}")
            
            try:
                self._engine = CorridorKeyEngine(
                    checkpoint_path=self.checkpoint_path,
                    device=str(self.device),
                    img_size=2048,
                    use_refiner=requested_use_refiner,
                    mixed_precision=(self.device.type != "cpu"),
                    model_precision=torch.float32,
                )
                
                self.engine_use_refiner = requested_use_refiner
                self.engine_screen_color = requested_screen_color
                logger.info("CorridorKey: Engine loaded successfully")
                return self._engine
            
            except Exception as e:
                logger.error(f"CorridorKey: Failed to load engine: {e}")
                self._engine = None
                raise
    
    def process_frame(
        self,
        image: np.ndarray,
        alpha_hint: Optional[np.ndarray] = None,
        despill_strength: float = 0.5,
        despeckle: bool = True,
        despeckle_size: int = 400,
        refiner_strength: float = 1.0,
        use_refiner: bool = True,
        input_is_linear: bool = False,
        screen_color: str = "green",
        _is_retry: bool = False,
    ) -> dict[str, np.ndarray]:
        """Process a single frame with CorridorKey.
        
        Args:
            image: RGB frame, shape (H, W, 3), dtype uint8 or float32 (0-1)
            alpha_hint: Optional alpha hint mask, shape (H, W), float32 (0-1)
            despill_strength: Green spill removal (0-1), default 0.5
            despeckle: Enable morphological cleanup
            despeckle_size: Size threshold for despeckle (pixels)
            refiner_strength: CNN refiner enhancement (0-2), default 1.0
            use_refiner: Enable or disable CorridorKey refiner at engine level
            input_is_linear: Input gamma interpretation. False=sRGB, True=Linear
        
        Returns:
            dict with keys:
                "alpha": Alpha matte (H, W, 1), float32
                "fg": Clean straight foreground for comp (H, W, 3), float32
                "comp": Preview composite for review only (H, W, 3), float32
                "processed": Premultiplied RGBA convenience output (H, W, 4), float32
        
        Raises:
            ValueError: If input shapes are invalid
            RuntimeError: If inference fails
        """
        if not _is_retry:
            CorridorKeyService._RUNTIME_NOTICE = ""

        # Validate inputs
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Image must have shape (H, W, 3), got {image.shape}"
            )
        
        if not isinstance(image, np.ndarray) or image.dtype not in [np.uint8, np.float32, np.float64]:
            raise ValueError(
                f"Image must be uint8 or float32, got {image.dtype}"
            )
        
        if alpha_hint is not None:
            # Auto-convert RGB/RGBA alpha_hint to single-channel grayscale.
            # Read node always loads files as 3-channel arrays, even grayscale ones.
            if alpha_hint.ndim == 3:
                if alpha_hint.shape[2] >= 3:
                    alpha_hint = (
                        0.299 * alpha_hint[:, :, 0].astype(np.float32)
                        + 0.587 * alpha_hint[:, :, 1].astype(np.float32)
                        + 0.114 * alpha_hint[:, :, 2].astype(np.float32)
                    )
                else:
                    alpha_hint = alpha_hint[:, :, 0].astype(np.float32)
            else:
                alpha_hint = np.asarray(alpha_hint, dtype=np.float32)
            # Normalize to [0, 1]: uint8 inputs arrive as [0, 255] after
            # conversion above; BiRefNet masks are already float32 [0, 1].
            if alpha_hint.max() > 1.0 + 1e-6:
                alpha_hint = alpha_hint / 255.0
            alpha_hint = np.clip(alpha_hint, 0.0, 1.0)
            if alpha_hint.ndim != 2 or alpha_hint.shape != image.shape[:2]:
                raise ValueError(
                    f"alpha_hint must be 2D with shape {image.shape[:2]}, "
                    f"got {alpha_hint.shape}"
                )
        
        logger.debug(
            f"CorridorKey: Processing frame {image.shape} "
            f"despill={despill_strength}, despeckle={despeckle}"
        )
        
        if alpha_hint is None:
            raise ValueError("CorridorKey requires alpha_hint (mask_linear)")

        resolved_screen_color = self._resolve_screen_color_for_frame(screen_color, image, alpha_hint)
        engine = self.load_engine(use_refiner=use_refiner, screen_color=resolved_screen_color)
        screen_channel = self._screen_channel_for_color(resolved_screen_color)

        # Safety check: ensure engine has model (catch silent initialization failures)
        if not hasattr(engine, "model"):
            raise RuntimeError(
                "CorridorKey engine loaded but has no 'model' attribute. "
                "The checkpoint may have failed to load during engine initialization. "
                f"Device: {self.device}, Checkpoint: {self.checkpoint_path}"
            )

        # process_frame expects despill in 0..1 range; our UI uses 0..10.
        despill_01 = max(0.0, min(1.0, float(despill_strength)))

        try:
            with torch.inference_mode():
                frame_kwargs = {
                    "image": image,
                    "mask_linear": alpha_hint,
                    "refiner_scale": float(refiner_strength),
                    "input_is_linear": bool(input_is_linear),
                    "fg_is_straight": True,
                    "despill_strength": despill_01,
                    "auto_despeckle": bool(despeckle),
                    "despeckle_size": int(despeckle_size),
                }
                if self._process_frame_accepts_kw(engine, "screen_channel"):
                    frame_kwargs["screen_channel"] = screen_channel
                elif screen_channel != 1:
                    raise RuntimeError(
                        "This CorridorKey runtime does not support blue-screen processing. "
                        "Update upstream CorridorKey to a build with screen_channel/screen_color support."
                    )
                raw_output = engine.process_frame(**frame_kwargs)

            if not isinstance(raw_output, dict):
                raise RuntimeError(f"Unexpected CorridorKey output type: {type(raw_output)}")

            alpha = raw_output.get("alpha")
            fg = raw_output.get("fg")
            comp = raw_output.get("comp")
            processed = raw_output.get("processed")

            output: dict[str, np.ndarray] = {}
            if alpha is not None:
                output["alpha"] = np.asarray(alpha, dtype=np.float32)
            if fg is not None:
                output["fg"] = np.asarray(fg, dtype=np.float32)
            if comp is not None:
                output["comp"] = np.asarray(comp, dtype=np.float32)
            if processed is not None:
                output["processed"] = np.asarray(processed, dtype=np.float32)

            logger.debug("CorridorKey: Frame processed successfully")
            return output
        
        except Exception as e:
            message = str(e).lower()
            is_mps_autocast_error = (
                self.device.type == "mps"
                and "autocast" in message
                and "mps" in message
            )
            if is_mps_autocast_error:
                logger.warning(
                    "CorridorKey: MPS autocast failed (%s). Falling back to CPU and retrying once.",
                    e,
                )
                CorridorKeyService._RUNTIME_NOTICE = str(e)
                self.device = torch.device("cpu")
                self._engine = None
                self.engine_screen_color = None
                self.engine_use_refiner = None
                try:
                    return self.process_frame(
                        image=image,
                        alpha_hint=alpha_hint,
                        despill_strength=despill_strength,
                        despeckle=despeckle,
                        despeckle_size=despeckle_size,
                        refiner_strength=refiner_strength,
                        use_refiner=use_refiner,
                        input_is_linear=input_is_linear,
                        screen_color=resolved_screen_color,
                        _is_retry=True,
                    )
                except Exception as retry_error:
                    logger.error(
                        "CorridorKey: CPU retry after MPS fallback failed: %s",
                        retry_error,
                    )
                    raise RuntimeError(f"CorridorKey inference failed: {retry_error}") from retry_error

            logger.error(f"CorridorKey: Inference failed: {e}")
            raise RuntimeError(f"CorridorKey inference failed: {e}") from e
    
    def unload_engine(self):
        """Unload model from memory to free VRAM.
        
        Useful when switching between models or low memory situations.
        """
        if self._engine is not None:
            self._engine = None
            self.engine_screen_color = None
            self.engine_use_refiner = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("CorridorKey: Engine unloaded from memory")
    
    @property
    def is_loaded(self) -> bool:
        """Check if engine is currently loaded."""
        return self._engine is not None
    
    @property
    def device_name(self) -> str:
        """Get human-readable device name."""
        if self.device.type == "cuda":
            try:
                return f"CUDA: {torch.cuda.get_device_name(0)}"
            except Exception:
                return "CUDA"
        elif self.device.type == "mps":
            return "Apple Metal (MPS)"
        else:
            return "CPU"
