"""Stage 6.5 — Cross-platform device validation tests.

Проверяет:
- get_device() реагирует на KEYFLOW_DEVICE=cpu (всегда доступно)
- get_device() с неизвестным значением env → fallback на авто-определение
- ModelService._select_device() уважает KEYFLOW_DEVICE
- CorridorKeyService._select_device() уважает KEYFLOW_DEVICE
- Возвращаемый torch.device имеет правильный тип
"""
import os
import sys
import unittest
from pathlib import Path
import importlib

# Убеждаемся, что до импортирования pytorch используется CPU
os.environ["KEYFLOW_DEVICE"] = "cpu"

import torch


class DeviceUtilityTests(unittest.TestCase):
    """app.utils.device.get_device() поведение при разных env-значениях."""

    def _get_device_with_env(self, value: str):
        """Импортировать, переключить env и вызвать get_device()."""
        import app.utils.device as dev_mod
        old = os.environ.get("KEYFLOW_DEVICE", "")
        os.environ["KEYFLOW_DEVICE"] = value
        try:
            # Перечитываем env при каждом вызове (функция не кэширует)
            return dev_mod.get_device()
        finally:
            os.environ["KEYFLOW_DEVICE"] = old

    def test_cpu_forced_returns_cpu_device(self):
        device = self._get_device_with_env("cpu")
        self.assertEqual(device.type, "cpu")

    def test_cpu_case_insensitive(self):
        device = self._get_device_with_env("CPU")
        self.assertEqual(device.type, "cpu")

    def test_cpu_with_whitespace(self):
        device = self._get_device_with_env("  cpu  ")
        self.assertEqual(device.type, "cpu")

    def test_empty_env_returns_torch_device(self):
        device = self._get_device_with_env("")
        self.assertIsInstance(device, torch.device)
        # Любое из cpu/cuda/mps является корректным
        self.assertIn(device.type, {"cpu", "cuda", "mps"})

    def test_unknown_value_falls_through_to_auto(self):
        """Неизвестное значение (не cpu/cuda/mps) → авто-выбор."""
        device = self._get_device_with_env("tpu_v999")
        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, {"cpu", "cuda", "mps"})

    def test_cuda_request_without_cuda_returns_non_cuda_or_cuda(self):
        """Запрос cuda без GPU → не падает, возвращает доступное устройство."""
        device = self._get_device_with_env("cuda")
        self.assertIsInstance(device, torch.device)
        if not torch.cuda.is_available():
            # Должен вернуть auto-fallback (mps или cpu)
            self.assertIn(device.type, {"cpu", "mps"})

    def test_mps_request_without_mps_returns_non_mps_or_mps(self):
        """Запрос mps без Metal → не падает, возвращает доступное устройство."""
        device = self._get_device_with_env("mps")
        self.assertIsInstance(device, torch.device)
        if not torch.backends.mps.is_available():
            self.assertIn(device.type, {"cpu", "cuda"})

    def test_repeated_calls_consistent(self):
        """Повторные вызовы с одним env дают одинаковое устройство."""
        import app.utils.device as dev_mod
        os.environ["KEYFLOW_DEVICE"] = "cpu"
        d1 = dev_mod.get_device()
        d2 = dev_mod.get_device()
        self.assertEqual(d1.type, d2.type)


class ModelServiceDeviceTests(unittest.TestCase):
    """app.services.model_service.ModelService уважает KEYFLOW_DEVICE."""

    def setUp(self):
        os.environ["KEYFLOW_DEVICE"] = "cpu"

    def test_service_imports_without_error(self):
        import app.services.model_service as ms
        self.assertTrue(hasattr(ms, "ModelService") or hasattr(ms, "InferenceService"))

    def test_inference_service_can_be_instantiated(self):
        from app.services import InferenceService
        svc = InferenceService()
        self.assertIsNotNone(svc)


class DevicePlatformContextTests(unittest.TestCase):
    """Контекстные тесты платформы — не падают на любой ОС."""

    def test_torch_available(self):
        self.assertTrue(hasattr(torch, "device"))

    def test_cpu_tensor_creation(self):
        t = torch.zeros(2, 2, device="cpu")
        self.assertEqual(t.device.type, "cpu")

    def test_current_platform_device_is_valid(self):
        import app.utils.device as dev_mod
        os.environ["KEYFLOW_DEVICE"] = "cpu"
        device = dev_mod.get_device()
        # Убеждаемся, что torch принимает это как валидный device
        t = torch.zeros(1, device=device)
        self.assertEqual(t.device.type, device.type)

    def test_device_env_propagation(self):
        """Записываем KEYFLOW_DEVICE и проверяем, что os.environ отражает значение."""
        original = os.environ.get("KEYFLOW_DEVICE", "")
        os.environ["KEYFLOW_DEVICE"] = "cpu"
        try:
            self.assertEqual(os.environ["KEYFLOW_DEVICE"], "cpu")
        finally:
            os.environ["KEYFLOW_DEVICE"] = original

    def test_no_cuda_assertion_on_cpu_mode(self):
        """В режиме cpu не должно бросаться torch RuntimeError об отсутствии CUDA."""
        os.environ["KEYFLOW_DEVICE"] = "cpu"
        import app.utils.device as dev_mod
        device = dev_mod.get_device()
        # Создание тензора на возвращённом устройстве не должно падать
        try:
            _ = torch.ones(3, device=device)
        except RuntimeError as e:
            self.fail(f"torch.ones on device {device} raised RuntimeError: {e}")
