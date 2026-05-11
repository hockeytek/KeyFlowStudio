"""Model service for MatAnyone2 inference"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Callable, Optional

from app.utils import get_device, get_model_variant_dir


MODEL_URL = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
MODEL_VARIANT = "v1"

_ORIGINAL_MPS_IS_AVAILABLE = torch.backends.mps.is_available


def _apply_forced_device_patch():
    """Force matanyone2 internals to resolve device as CPU when requested."""
    forced = os.environ.get("KEYFLOW_DEVICE", "").strip().lower()
    if forced != "cpu":
        try:
            torch.backends.mps.is_available = _ORIGINAL_MPS_IS_AVAILABLE
        except Exception:
            pass
        return

    # matanyone2 internally calls torch.backends.mps.is_available() repeatedly.
    # If it returns True while model is on CPU, mixed-device errors happen.
    try:
        torch.backends.mps.is_available = lambda: False
    except Exception:
        pass


def _move_tensor_attributes_to_device(module, device):
    """Move tensor attributes that are not registered as params/buffers.

    Some third-party models keep tensors like pixel_mean/pixel_std as plain
    attributes. `.to(device)` does not always move those reliably.
    """
    for attr_name, attr_value in vars(module).items():
        if isinstance(attr_value, torch.Tensor) and attr_value.device != device:
            setattr(module, attr_name, attr_value.to(device))


def _gen_dilate(alpha, kernel_size):
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    return (cv2.dilate(fg_and_unknown, kernel, iterations=1) * 255).astype(np.float32)


def _gen_erosion(alpha, kernel_size):
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    return (cv2.erode(fg, kernel, iterations=1) * 255).astype(np.float32)


def _resolve_matanyone2_config_dir() -> Path:
    """Resolve matanyone2 Hydra config directory in dev and bundled modes."""
    import matanyone2

    candidates = [
        Path(matanyone2.__file__).resolve().parent / "config",
        Path(getattr(sys, "_MEIPASS", "")) / "matanyone2" / "config",
        Path(sys.executable).resolve().parents[1] / "Resources" / "matanyone2" / "config",
    ]

    for cfg_dir in candidates:
        if cfg_dir and cfg_dir.exists():
            return cfg_dir

    checked = "\n".join(str(p) for p in candidates)
    raise RuntimeError(
        "Не найден каталог конфигурации matanyone2/config. Проверены пути:\n"
        f"{checked}"
    )


def _build_matanyone2_model(ckpt_path: str, device):
    """Build MatAnyone2 with explicit Hydra config dir to support frozen app."""
    from omegaconf import open_dict
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from matanyone2.model.matanyone2 import MatAnyone2

    cfg_dir = _resolve_matanyone2_config_dir()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3.2", config_dir=str(cfg_dir), job_name="eval_our_config"):
        cfg = compose(config_name="eval_matanyone_config")

    with open_dict(cfg):
        cfg["weights"] = ckpt_path

    model = MatAnyone2(cfg, single_object=True).to(device).eval()
    model_weights = torch.load(cfg.weights, map_location=device)
    model.load_weights(model_weights)
    return model


class ModelService:
    """Service для загрузки и кэширования модели MatAnyone2"""

    _instance = None
    _model = None               # (model, processor)
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._device is None:
            self._device = get_device()

    def load_model(self, model_path=None):
        """
        Загружает модель MatAnyone2.
        Если уже загружена — возвращает кэшированную.
        """
        if self._model is not None:
            return self._model

        try:
            _apply_forced_device_patch()
            from matanyone2.inference.inference_core import InferenceCore

            if model_path is None:
                from matanyone2.utils.download_util import load_file_from_url
                model_path = load_file_from_url(
                    MODEL_URL,
                    str(get_model_variant_dir("matanyone2", MODEL_VARIANT)),
                )

            model = _build_matanyone2_model(model_path, self._device)
            model = model.eval()
            model = model.to(self._device)
            
            processor = InferenceCore(model, cfg=model.cfg)
            processor.network = processor.network.to(self._device)
            _move_tensor_attributes_to_device(processor.network, self._device)

            self._model = (model, processor)
            return self._model

        except ImportError as e:
            raise RuntimeError(
                f"MatAnyone2 не установлена. Установите: pip install -e /path/to/MatAnyone2\n"
                f"Ошибка: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели: {e}")

    def get_model(self):
        """Возвращает загруженную пару (model, processor) или None"""
        return self._model

    def is_loaded(self):
        """Проверяет, загружена ли модель"""
        return self._model is not None

    def get_device(self):
        return self._device

    def reinit_device(self):
        """Пересчитать устройство из окружения (при переключении режима)."""
        _apply_forced_device_patch()
        self._device = get_device()
        self._model = None  # Очистить кэш модели при изменении устройства

    @classmethod
    def get_weights_status(cls) -> dict:
        """Return {'state': 'ready'|'missing', 'path': str}."""
        weights_dir = get_model_variant_dir("matanyone2", MODEL_VARIANT)
        pth_file = weights_dir / "matanyone2.pth"
        if pth_file.is_file():
            return {"state": "ready", "path": str(pth_file)}
        return {"state": "missing", "path": str(pth_file)}

    @classmethod
    def ensure_weights_available(
        cls,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Path:
        """Download matanyone2.pth if not present, with optional progress."""
        weights_dir = get_model_variant_dir("matanyone2", MODEL_VARIANT)
        status = cls.get_weights_status()
        if status["state"] == "ready":
            return Path(status["path"])
        if progress_callback:
            progress_callback(5, "Downloading MatAnyone2 weights…")
        from matanyone2.utils.download_util import load_file_from_url
        model_path = load_file_from_url(MODEL_URL, str(weights_dir))
        if progress_callback:
            progress_callback(100, "MatAnyone2 weights ready")
        return Path(model_path)


class InferenceService:
    """Service для инференса (обработки видео)"""

    def __init__(self):
        self.model_service = ModelService()

    @staticmethod
    def _ensure_processor_on_device(processor, device):
        network = getattr(processor, "network", None)
        if network is None:
            return
        processor.network = network.to(device)
        _move_tensor_attributes_to_device(processor.network, device)

    @staticmethod
    def _resolve_processor_device(processor, fallback_device):
        """Return the actual runtime device of processor.network.

        Global device state may change from UI while a worker is starting.
        We bind all runtime tensors to the processor's real network device.
        """
        network = getattr(processor, "network", None)
        if network is None:
            return fallback_device

        pixel_mean = getattr(network, "pixel_mean", None)
        if isinstance(pixel_mean, torch.Tensor):
            return pixel_mean.device

        first_param = next(network.parameters(), None)
        if first_param is not None:
            return first_param.device

        first_buffer = next(network.buffers(), None)
        if first_buffer is not None:
            return first_buffer.device

        return fallback_device

    def process_video(
        self,
        frames,
        mask,
        n_warmup=10,
        r_erode=0,
        r_dilate=0,
        progress_callback=None,
        cancel_flag=None,
        correction_masks: "dict[int, np.ndarray] | None" = None,
    ):
        """
        Обрабатывает видео с нуль-warmup кадрами (полный паритет с matanyone2_wrapper.py).

        Args:
            frames: list of np.ndarray (H, W, 3) uint8, RGB
            mask: np.ndarray (H, W) uint8
            n_warmup: количество warmup-итераций (default 10)
            r_erode: радиус эрозии маски (0 = отключено)
            r_dilate: радиус дилации маски (0 = отключено)
            progress_callback: fn(current, total)
            cancel_flag: threading.Event

        Returns:
            list of np.ndarray (H, W) float in [0, 1] — alpha per frame
        """
        if not self.model_service.is_loaded():
            raise RuntimeError("Модель не загружена")

        _, processor = self.model_service.get_model()
        _apply_forced_device_patch()
        fallback_device = self.model_service.get_device()
        device = self._resolve_processor_device(processor, fallback_device)
        self._ensure_processor_on_device(processor, device)

        # Reset processor state between runs.
        # The processor is cached (singleton) and may retain sensory memory and
        # cached image features from a previous run.  If the image size changed
        # between runs, stale feature-store entries at the same time-index would
        # cause a spatial mismatch inside sensory_update (e.g. g_chunk H=172 vs
        # sensory H=68).  clear_memory() rebuilds MemoryManager (empty sensory)
        # and resetting _store removes old per-frame feature caches entirely.
        processor.clear_memory()
        processor.image_feature_store._store.clear()

        from torchvision.transforms.functional import to_tensor

        # Применить erode / dilate к исходной маске
        mask_f = mask.astype(np.float32)
        if r_dilate > 0:
            mask_f = _gen_dilate(mask_f, r_dilate)
        if r_erode > 0:
            mask_f = _gen_erosion(mask_f, r_erode)
        mask_tensor = torch.from_numpy(mask_f).to(device=device, dtype=torch.float32)

        objects = [1]
        # Warmup: n_warmup копий первого кадра + реальные кадры
        all_frames = [frames[0]] * n_warmup + list(frames)
        n_actual = len(frames)
        phas = []

        with torch.inference_mode():
            for ti, frame in enumerate(all_frames):
                if cancel_flag is not None and cancel_flag.is_set():
                    break

                image = to_tensor(frame).to(device=device, dtype=torch.float32)

                if ti == 0:
                    # Закодировать маску
                    processor.step(image, mask_tensor, objects=objects)
                    # Сбросить память для warmup
                    output_prob = processor.step(image, first_frame_pred=True)
                elif ti <= n_warmup:
                    output_prob = processor.step(image, first_frame_pred=True)
                else:
                    # actual_frame_idx: 0-based index into the real frames list
                    actual_frame_idx = ti - n_warmup
                    if correction_masks and actual_frame_idx in correction_masks:
                        cm = correction_masks[actual_frame_idx]
                        # Apply same erode/dilate as the initial mask to soften hard SAM edges.
                        cm_f = cm.astype(np.float32)
                        if r_dilate > 0:
                            cm_f = _gen_dilate(cm_f, r_dilate)
                        if r_erode > 0:
                            cm_f = _gen_erosion(cm_f, r_erode)
                        # Pass mask in [0..255] float range — step() divides by 255
                        # internally (matting branch: mask / 255.), so we must NOT
                        # pre-normalise here or the result collapses to near-zero alpha.
                        cm_tensor = torch.from_numpy(cm_f).to(device=device, dtype=torch.float32)
                        # Inject correction mask as a new memory anchor without
                        # first_frame_pred — that would call clear_temp_mem() and
                        # wipe all temporal context accumulated so far.
                        output_prob = processor.step(image, cm_tensor, objects=objects)
                        # The matting branch uses the raw mask directly as alpha (no neural
                        # refinement). Apply Gaussian blur to smooth the SAM edges on this frame.
                        import cv2 as _cv2
                        alpha_np = processor.output_prob_to_mask(output_prob).detach().cpu().numpy()
                        alpha_np = _cv2.GaussianBlur(alpha_np, (7, 7), 1.5).clip(0.0, 1.0)
                        phas.append(alpha_np)
                        actual_idx = ti - n_warmup + 1
                        if progress_callback:
                            progress_callback(actual_idx, n_actual, frame, phas[-1])
                        continue
                    else:
                        output_prob = processor.step(image)

                # Пропустить warmup-кадры
                if ti <= n_warmup - 1:
                    continue

                alpha = processor.output_prob_to_mask(output_prob)
                phas.append(alpha.detach().cpu().numpy())  # (H, W) float [0,1]

                actual_idx = ti - n_warmup + 1
                if progress_callback:
                    # Provide current frame + alpha so UI can refresh Write node runtime previews.
                    progress_callback(actual_idx, n_actual, frame, phas[-1])

        return phas
