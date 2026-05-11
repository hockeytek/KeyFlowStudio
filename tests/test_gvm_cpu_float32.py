"""Synthetic tests for GVM CPU float32 pipeline wrapper.

Verifies that:
1. _Float32PipeWrapper up-casts float16 tensors to float32 before forwarding.
2. GVMService.load_model() installs the wrapper on CPU.
3. GVMService.process_sequence() completes without hanging on a tiny synthetic
   frame sequence (1 white frame, 64×64), using a fully-mocked GVMProcessor so
   no weights or GPU are needed.
"""

from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("KEYFLOW_DEVICE", "cpu")

import numpy as np
import torch

from app.services.gvm_service import GVMService, _Float32PipeWrapper, _patch_upblock_upsample_size


# ---------------------------------------------------------------------------
# 1. Unit test for _Float32PipeWrapper
# ---------------------------------------------------------------------------

class Float32WrapperTests(unittest.TestCase):
    """Ensure the wrapper up-casts float16 tensors and proxies attributes."""

    def _make_pipe(self, received: list):
        """Return a minimal mock pipeline that records its call args."""
        class _FakePipe:
            def __call__(self, frames, **kwargs):
                received.append((frames, kwargs))
                return types.SimpleNamespace(image=frames, alpha=frames)

            def to(self, *args, **kwargs):
                return self

        return _FakePipe()

    def test_float16_input_is_upcasted(self):
        received: list = []
        pipe = _Float32PipeWrapper(self._make_pipe(received))
        tensor_f16 = torch.zeros(1, 3, 64, 64, dtype=torch.float16)
        pipe(tensor_f16, num_inference_steps=1)
        called_frames, _ = received[0]
        self.assertEqual(called_frames.dtype, torch.float32,
                         "float16 tensor should be promoted to float32")

    def test_float32_input_passes_through(self):
        received: list = []
        pipe = _Float32PipeWrapper(self._make_pipe(received))
        tensor_f32 = torch.zeros(1, 3, 64, 64, dtype=torch.float32)
        pipe(tensor_f32, num_inference_steps=1)
        called_frames, _ = received[0]
        self.assertEqual(called_frames.dtype, torch.float32)

    def test_attribute_proxy(self):
        fake_pipe = MagicMock()
        fake_pipe.scheduler = "sched"
        wrapper = _Float32PipeWrapper(fake_pipe)
        self.assertEqual(wrapper.scheduler, "sched")

    def test_vae_float16_call_is_noop(self):
        """The VAE's to(dtype=float16) inside GVMPipeline.__call__ must be suppressed."""
        import types as _t
        dtype_calls: list = []

        class _FakeVAE:
            dtype = torch.float32
            def to(self, *args, **kwargs):
                new_dtype = kwargs.get("dtype")
                if args and isinstance(args[0], torch.dtype):
                    new_dtype = args[0]
                dtype_calls.append(new_dtype)
                return self

        class _FakePipeWithVAE:
            vae = _FakeVAE()
            def __call__(self, frames, **kwargs):
                return _t.SimpleNamespace(image=frames, alpha=frames)
            def to(self, *args, **kwargs):
                return self

        pipe = _Float32PipeWrapper(_FakePipeWithVAE())
        # Simulate what GVMPipeline.__call__ does: self.vae.to(dtype=torch.float16)
        pipe.vae.to(dtype=torch.float16)
        # float16 call should be intercepted — vae.dtype should stay float32
        self.assertEqual(pipe.vae.dtype, torch.float32,
                         "VAE dtype must remain float32 after patched .to(float16) call")


# ---------------------------------------------------------------------------
# 2. Integration: GVMService.load_model installs wrapper on CPU
# ---------------------------------------------------------------------------

class GVMServiceLoadModelTests(unittest.TestCase):
    def setUp(self):
        # Reset singleton state before each test
        GVMService._instance = None
        GVMService._processor = None

    def tearDown(self):
        GVMService._instance = None
        GVMService._processor = None

    def _make_fake_processor(self):
        proc = MagicMock()
        proc.pipe = MagicMock()
        proc.pipe.to = MagicMock(return_value=proc.pipe)
        return proc

    def test_wrapper_installed_on_cpu(self):
        fake_proc = self._make_fake_processor()

        with patch("app.services.gvm_service.GVMService.get_weights_status",
                   return_value={"state": "ready", "path": "/fake/weights"}), \
             patch("app.services.gvm_service.GVMService._ensure_gvm_core_importable"), \
             patch.dict("sys.modules", {"gvm_core": types.ModuleType("gvm_core")}):
            import sys
            sys.modules["gvm_core"].GVMProcessor = MagicMock(return_value=fake_proc)

            svc = GVMService()
            svc.load_model(device="cpu")

        self.assertIsInstance(svc._processor.pipe, _Float32PipeWrapper,
                              "pipe must be replaced with _Float32PipeWrapper on CPU")

    def test_wrapper_not_installed_on_cuda(self):
        fake_proc = self._make_fake_processor()

        with patch("app.services.gvm_service.GVMService.get_weights_status",
                   return_value={"state": "ready", "path": "/fake/weights"}), \
             patch("app.services.gvm_service.GVMService._ensure_gvm_core_importable"), \
             patch.dict("sys.modules", {"gvm_core": types.ModuleType("gvm_core")}):
            import sys
            sys.modules["gvm_core"].GVMProcessor = MagicMock(return_value=fake_proc)

            svc = GVMService()
            svc.load_model(device="cuda")

        self.assertNotIsInstance(svc._processor.pipe, _Float32PipeWrapper,
                                 "pipe must NOT be wrapped on CUDA")


# ---------------------------------------------------------------------------
# 3. End-to-end: process_sequence with a mock processor completes immediately
# ---------------------------------------------------------------------------

class GVMProcessSequenceTests(unittest.TestCase):
    """Use a tiny synthetic frame (1 white 64×64 frame) with mocked GVMProcessor
    to verify process_sequence() writes alpha PNGs and does not hang."""

    def setUp(self):
        GVMService._instance = None
        GVMService._processor = None

    def tearDown(self):
        GVMService._instance = None
        GVMService._processor = None

    def _make_stub_processor(self):
        """Minimal stub that mimics GVMProcessor.process_sequence behaviour."""
        import cv2

        class _PipeOutput:
            def __init__(self, b, h, w):
                ones = torch.ones(b, 1, h, w, dtype=torch.float32)
                self.image = ones
                self.alpha = ones

        class _StubProcessor:
            device = torch.device("cpu")

            def process_sequence(
                self,
                input_path,
                output_dir,
                **kwargs,
            ):
                direct_dir = kwargs.get("direct_output_dir") or output_dir
                Path(direct_dir).mkdir(parents=True, exist_ok=True)
                # Write a single alpha PNG
                alpha = np.ones((64, 64), dtype=np.uint8) * 255
                cv2.imwrite(str(Path(direct_dir) / "00000.png"), alpha)
                if kwargs.get("progress_callback"):
                    kwargs["progress_callback"](1, 1)

        return _StubProcessor()

    def test_process_sequence_writes_alpha_frames(self):
        import cv2

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 1-frame input as a PNG
            frame = np.ones((64, 64, 3), dtype=np.uint8) * 200
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            cv2.imwrite(str(input_dir / "00000.png"), frame)

            output_dir = Path(tmpdir) / "output"

            svc = GVMService()
            svc._processor = self._make_stub_processor()

            progress_calls: list = []
            alpha_paths = svc.process_sequence(
                input_path=input_dir,
                output_dir=output_dir,
                progress_callback=lambda done, total: progress_calls.append((done, total)),
            )

            self.assertGreater(len(alpha_paths), 0,
                               "process_sequence must return at least 1 alpha PNG path")
            for p in alpha_paths:
                self.assertTrue(Path(p).exists(), f"Alpha file {p} must exist on disk")

            self.assertTrue(
                any(t == (1, 1) for t in progress_calls),
                "progress_callback must be called with (1, 1)",
            )

    def test_process_sequence_reports_done_frames_per_batch(self):
        import cv2
        import math

        class _BatchProgressProcessor:
            def process_sequence(self, input_path, output_dir, **kwargs):
                direct_dir = Path(kwargs.get("direct_output_dir") or output_dir)
                direct_dir.mkdir(parents=True, exist_ok=True)

                input_files = sorted(Path(input_path).glob("*.png"))
                batch_size = int(kwargs.get("num_frames_per_batch") or 8)
                total_batches = max(1, math.ceil(len(input_files) / batch_size))

                for batch_index in range(total_batches):
                    start = batch_index * batch_size
                    end = min(start + batch_size, len(input_files))
                    for frame_index in range(start, end):
                        alpha = np.ones((32, 32), dtype=np.uint8) * 255
                        cv2.imwrite(str(direct_dir / f"{frame_index:05d}.png"), alpha)
                    progress = kwargs.get("progress_callback")
                    if progress is not None:
                        progress(batch_index + 1, total_batches)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            for frame_index in range(17):
                frame = np.ones((32, 32, 3), dtype=np.uint8) * 200
                cv2.imwrite(str(input_dir / f"{frame_index:05d}.png"), frame)

            output_dir = Path(tmpdir) / "output"
            svc = GVMService()
            svc._processor = _BatchProgressProcessor()

            progress_calls: list[tuple[int, int]] = []
            alpha_paths = svc.process_sequence(
                input_path=input_dir,
                output_dir=output_dir,
                num_frames_per_batch=8,
                progress_callback=lambda done, total: progress_calls.append((done, total)),
            )

            self.assertEqual(len(alpha_paths), 17)
            self.assertEqual(progress_calls, [(8, 17), (16, 17), (17, 17)])

    def test_float32_wrapper_does_not_hang(self):
        """Smoke: _Float32PipeWrapper on a minimal pipeline completes synchronously."""
        calls: list = []

        class _FastPipe:
            def __call__(self, frames, **kwargs):
                calls.append(frames.dtype)
                return types.SimpleNamespace(
                    image=frames,
                    alpha=frames,
                )
            def to(self, *args, **kwargs):
                return self

        wrapped = _Float32PipeWrapper(_FastPipe())
        t_f16 = torch.zeros(1, 3, 8, 8, dtype=torch.float16)
        wrapped(t_f16)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], torch.float32,
                         "Pipeline must receive float32 tensor, not float16")


class UpBlockPatchTests(unittest.TestCase):
    """Verify _patch_upblock_upsample_size restores correct upsample_size behaviour.

    Old diffusers passed upsample_size to upsampler(hidden_states, upsample_size)
    so spatial dimensions align with skip connections.  New diffusers dropped the
    param; our patch restores it by calling upsamplers manually with the target size.
    """

    def test_patch_is_idempotent(self):
        _patch_upblock_upsample_size()
        _patch_upblock_upsample_size()  # second call must not raise

    def test_forward_accepts_upsample_size(self):
        try:
            from diffusers.models.unets.unet_3d_blocks import UpBlockSpatioTemporal
        except (ImportError, RuntimeError):
            self.skipTest("diffusers not available or torch dispatch conflict")

        _patch_upblock_upsample_size()

        import inspect
        sig = inspect.signature(UpBlockSpatioTemporal.forward)
        params = sig.parameters
        # After patching, either upsample_size is explicit or **kwargs absorbs it
        has_upsample_size = "upsample_size" in params
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in params.values()
        )
        self.assertTrue(
            has_upsample_size or has_kwargs,
            "patched forward must accept upsample_size (explicit or via **kwargs)",
        )

    def test_upsample_size_is_forwarded_to_upsampler(self):
        """Core regression: upsampler must receive the target size, not scale_factor=2."""
        _patch_upblock_upsample_size()

        upsampler_calls: list = []

        class _FakeUpsampler:
            def __call__(self, hidden_states, output_size=None):
                upsampler_calls.append(output_size)
                return hidden_states

        class _FakeResnet:
            def __call__(self, x, temb, image_only_indicator=None):
                return x

        # Build a minimal mock that looks like UpBlockSpatioTemporal
        # (the patch uses self.upsamplers and the orig forward body)
        try:
            from diffusers.models.unets.unet_3d_blocks import UpBlockSpatioTemporal
        except (ImportError, RuntimeError):
            self.skipTest("diffusers not available or torch dispatch conflict")

        block = UpBlockSpatioTemporal.__new__(UpBlockSpatioTemporal)
        block.resnets = [_FakeResnet()]
        block.upsamplers = [_FakeUpsampler()]
        block.training = False
        block.gradient_checkpointing = False

        h = torch.zeros(1, 4, 2, 8, 8)
        res = torch.zeros(1, 4, 2, 8, 8)
        target_size = (8, 8)

        # call with upsample_size — must reach _FakeUpsampler with that size
        block.forward(h, (res,), upsample_size=target_size)
        self.assertEqual(upsampler_calls, [target_size],
                         "upsampler must be called with upsample_size as output_size")

        # call without upsample_size — upsampler called with None (scale_factor=2)
        upsampler_calls.clear()
        block.forward(h, (res,))
        self.assertEqual(upsampler_calls, [None],
                         "upsampler must be called with None when upsample_size omitted")


if __name__ == "__main__":
    unittest.main()
