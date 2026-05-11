"""Device selection utility for torch"""
import logging
import os
import torch

logger = logging.getLogger(__name__)


def get_device():
    """
    Автоматический выбор устройства:
    - CUDA (если есть NVIDIA GPU)
    - MPS (если Mac с Apple Silicon)
    - CPU (fallback)
    """
    forced_device = os.environ.get("KEYFLOW_DEVICE", "").strip().lower()

    if forced_device == "cpu":
        device = torch.device("cpu")
        logger.warning("Принудительно выбран CPU (KEYFLOW_DEVICE=cpu)")
        return device

    if forced_device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Принудительно выбран CUDA: %s", torch.cuda.get_device_name(0))
            return device
        logger.warning("Запрошен CUDA, но он недоступен. Переход к авто-выбору.")

    if forced_device == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Принудительно выбран MPS (Apple Silicon)")
            return device
        logger.warning("Запрошен MPS, но он недоступен. Переход к авто-выбору.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("CUDA доступна: %s", torch.cuda.get_device_name(0))
        return device
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Metal Performance Shaders (Apple Silicon) доступны")
        return device
    else:
        device = torch.device("cpu")
        logger.warning("Используется CPU (обработка может быть медленной)")
        return device


def get_device_name():
    """Возвращает название текущего устройства"""
    device = get_device()
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    elif device.type == "mps":
        return "Apple Silicon (MPS)"
    else:
        return "CPU"
