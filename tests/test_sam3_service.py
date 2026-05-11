import unittest
from pathlib import Path
from unittest.mock import patch
import types

import numpy as np

from app.services.sam3_service import Sam3Service


class _FakeProcessor:
    def set_image(self, image):
        return {"image_size": image.size}

    def set_text_prompt(self, prompt: str, state: dict):
        st = dict(state)
        st["concept"] = prompt
        st["masks"] = np.array([[[1, 0], [0, 1]]], dtype=np.uint8)
        return st

    def add_geometric_prompt(self, box, label: bool, state: dict):
        _ = box
        st = dict(state)
        if label:
            st["masks"] = np.array([[[1, 1], [0, 0]]], dtype=np.uint8)
        else:
            st["masks"] = np.array([[[0, 0], [1, 1]]], dtype=np.uint8)
        return st


class Sam3ServiceTests(unittest.TestCase):
    def test_select_device_uses_global_device_setting(self):
        with patch("app.services.sam3_service.get_device", return_value=types.SimpleNamespace(type="mps")):
            self.assertEqual(Sam3Service._select_device(), "mps")

    def test_weight_status_reports_ready_when_checkpoint_exists(self):
        with patch.object(Sam3Service, "_model_dir_for", return_value=Path(__file__).parent):
            with patch.object(
                Sam3Service,
                "SAM3_CKPT_CANDIDATES",
                {"sam3": [Path(__file__).name]},
            ):
                status = Sam3Service.get_weight_status("sam3")
                self.assertEqual(status["state"], "ready")

    def test_predict_image_requires_prompt_or_points(self):
        with self.assertRaises(ValueError):
            Sam3Service.predict_image("sam3", np.zeros((2, 2, 3), dtype=np.uint8), [], "")

    def test_predict_image_returns_uint8_masks_with_text_prompt(self):
        fake_runtime = {"processor": _FakeProcessor()}
        with patch.object(Sam3Service, "_get_or_build_image_runtime", return_value=fake_runtime):
            masks = Sam3Service.predict_image(
                "sam3",
                np.zeros((2, 2, 3), dtype=np.uint8),
                points=[],
                concept="person",
            )

        self.assertEqual(len(masks), 1)
        self.assertEqual(masks[0].dtype, np.uint8)
        self.assertTrue(np.array_equal(masks[0], np.array([[255, 0], [0, 255]], dtype=np.uint8)))

    def test_predict_image_supports_points_without_text(self):
        fake_runtime = {"processor": _FakeProcessor()}
        with patch.object(Sam3Service, "_get_or_build_image_runtime", return_value=fake_runtime):
            masks = Sam3Service.predict_image(
                "sam3",
                np.zeros((10, 10, 3), dtype=np.uint8),
                points=[(4, 5, 1)],
                concept="",
            )

        self.assertEqual(len(masks), 1)
        self.assertTrue(np.array_equal(masks[0], np.array([[255, 255], [0, 0]], dtype=np.uint8)))

    def test_predict_image_retries_on_cpu_after_mps_failure(self):
        runtimes = [
            {"device": "mps", "processor": _FakeProcessor()},
            {"device": "cpu", "processor": _FakeProcessor()},
        ]

        def fake_run(runtime, pil_image, points, concept):
            _ = (pil_image, points, concept)
            if runtime["device"] == "mps":
                raise RuntimeError("MPS backend out of memory")
            return [np.array([[255, 0], [0, 255]], dtype=np.uint8)]

        with patch.object(Sam3Service, "_get_or_build_image_runtime", side_effect=runtimes) as get_runtime:
            with patch.object(Sam3Service, "_run_image_inference", side_effect=fake_run):
                masks = Sam3Service.predict_image(
                    "sam3",
                    np.zeros((2, 2, 3), dtype=np.uint8),
                    points=[],
                    concept="person",
                )

        self.assertEqual(len(masks), 1)
        self.assertEqual(get_runtime.call_count, 2)
        self.assertEqual(get_runtime.call_args_list[1].kwargs["preferred_device"], "cpu")

    def test_runtime_notice_is_exposed_for_mps_fallback(self):
        runtimes = [
            {"device": "mps", "processor": _FakeProcessor()},
            {"device": "cpu", "processor": _FakeProcessor()},
        ]

        def fake_run(runtime, pil_image, points, concept):
            _ = (pil_image, points, concept)
            if runtime["device"] == "mps":
                raise RuntimeError("MPS backend out of memory")
            return [np.array([[255, 0], [0, 255]], dtype=np.uint8)]

        with patch.object(Sam3Service, "_get_or_build_image_runtime", side_effect=runtimes):
            with patch.object(Sam3Service, "_run_image_inference", side_effect=fake_run):
                Sam3Service.predict_image(
                    "sam3",
                    np.zeros((2, 2, 3), dtype=np.uint8),
                    points=[],
                    concept="person",
                )

        notice = Sam3Service.consume_runtime_notice()
        self.assertIn("MPS backend out of memory", notice)
        self.assertEqual(Sam3Service.consume_runtime_notice(), "")


if __name__ == "__main__":
    unittest.main()
