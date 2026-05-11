import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from app.workers.inference_worker import InferenceWorker
from app.services.birefnet_service import BiRefNetService
from app.services.corridorkey_service import CorridorKeyService
from app.utils.corridorkey_output import (
    build_corridorkey_processed_output,
    coerce_alpha_2d,
    coerce_rgb_float01,
)


class InferenceWorkerFailurePathTests(unittest.TestCase):
    def test_birefnet_runtime_notice_is_exposed_for_mps_fallback(self):
        service = BiRefNetService()
        service.device = "mps"
        service._runtime_force_device = None

        class _Model:
            def predict(self, _image):
                if service._runtime_force_device == "cpu":
                    service.device = "cpu"
                if service.device == "mps":
                    raise RuntimeError("MPS backend out of memory")
                return np.ones((2, 2), dtype=np.float32)

        original_load_model = service.load_model
        try:
            service.load_model = lambda **_kwargs: _Model()
            _ = service.process_image(
                np.zeros((2, 2, 3), dtype=np.uint8),
                usage="General",
            )
        finally:
            service.load_model = original_load_model

        notice = service.consume_runtime_notice()
        self.assertIn("MPS backend out of memory", notice)
        self.assertEqual(service.consume_runtime_notice(), "")

    def test_corridorkey_runtime_notice_is_exposed_for_mps_fallback(self):
        service = CorridorKeyService()
        service.device = torch.device("mps")

        class _Engine:
            model = object()

            def process_frame(self, **_kwargs):
                if service.device.type == "mps":
                    raise RuntimeError("unsupported autocast device_type 'mps'")
                return {
                    "alpha": np.ones((2, 2, 1), dtype=np.float32),
                }

        original_load_engine = service.load_engine
        try:
            service.load_engine = lambda **_kwargs: _Engine()
            _ = service.process_frame(
                np.zeros((2, 2, 3), dtype=np.uint8),
                alpha_hint=np.ones((2, 2), dtype=np.float32),
            )
        finally:
            service.load_engine = original_load_engine

        notice = service.consume_runtime_notice()
        self.assertIn("autocast", notice.lower())
        self.assertEqual(service.consume_runtime_notice(), "")

    def test_corridorkey_service_accepts_fg_without_processed(self):
        service = CorridorKeyService()

        class _Engine:
            model = object()

            @staticmethod
            def process_frame(**_kwargs):
                return {
                    "alpha": np.ones((2, 2, 1), dtype=np.float32),
                    "fg": np.ones((2, 2, 3), dtype=np.float32) * 0.5,
                    "comp": np.ones((2, 2, 3), dtype=np.float32),
                }

        original_load_engine = service.load_engine
        try:
            service.load_engine = lambda **_kwargs: _Engine()
            result = service.process_frame(
                np.zeros((2, 2, 3), dtype=np.uint8),
                alpha_hint=np.ones((2, 2), dtype=np.float32),
            )
        finally:
            service.load_engine = original_load_engine

        self.assertIn("alpha", result)
        self.assertIn("fg", result)
        self.assertIn("comp", result)
        self.assertNotIn("processed", result)

    def test_corridorkey_requires_alphahint_or_deferred_source(self):
        # Use __new__ to keep this unit test lightweight and focused on validation path.
        worker = InferenceWorker.__new__(InferenceWorker)

        node_data = {
            "id": "corridor_1",
            "properties": {
                "use_refiner": True,
                "alpha_hint_mode": "auto",
            },
        }
        inputs = {
            "image": [np.zeros((2, 2, 3), dtype=np.uint8)],
        }

        with self.assertRaisesRegex(ValueError, "CorridorKey requires an alpha hint sequence"):
            worker._execute_corridorkey_node(node_data, inputs)

    def test_matting_requires_mask(self):
        # Use __new__ to keep this unit test lightweight and focused on validation path.
        worker = InferenceWorker.__new__(InferenceWorker)

        node_data = {
            "id": "matting_1",
            "properties": {
                "warmup": 10,
                "erode": 0,
                "dilate": 0,
            },
        }
        inputs = {
            "img": [np.zeros((2, 2, 3), dtype=np.uint8)],
        }

        with self.assertRaisesRegex(ValueError, "MatAnyone2 node requires a mask"):
            worker._execute_matanyone2_node(node_data, inputs)

    def test_sam_worker_boundary_uses_nearest_frame_mask_propagation(self):
        # Boundary-level check: execute SAM branch via worker dispatcher.
        worker = InferenceWorker.__new__(InferenceWorker)
        worker.cancel_flag = SimpleNamespace(is_set=lambda: False)
        worker._graph_start_frame = 10

        stream_calls = []
        progress_calls = []
        worker._stream_graph_write_frame = lambda *args, **kwargs: stream_calls.append((args, kwargs))
        worker.node_frame_progress = SimpleNamespace(emit=lambda *args: progress_calls.append(args))

        mask_a = np.ones((2, 2), dtype=np.uint8) * 255
        mask_b = np.zeros((2, 2), dtype=np.uint8)
        worker._load_sam_masks_from_payloads = lambda _props, _shape: {
            10: mask_a,
            12: mask_b,
        }

        frames = [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.uint8),
        ]
        node_data = {"id": "sam_1", "properties": {}}

        result = worker._execute_node("sam2", node_data, {"img": frames})

        self.assertIn("out", result)
        self.assertEqual(len(result["out"]), 3)
        # global frame 10 -> exact A
        self.assertTrue(np.array_equal(result["out"][0], mask_a))
        # global frame 11 -> nearest previous (A), not next (B)
        self.assertTrue(np.array_equal(result["out"][1], mask_a))
        # global frame 12 -> exact B
        self.assertTrue(np.array_equal(result["out"][2], mask_b))
        self.assertEqual(len(stream_calls), 3)
        self.assertEqual(len(progress_calls), 3)

    def test_comp_stream_preview_is_marked_preview_only(self):
        worker = InferenceWorker.__new__(InferenceWorker)
        worker._graph_start_frame = 0
        worker._graph_stream_saved_paths = {}
        emitted = []
        worker.graph_stream_preview = SimpleNamespace(emit=lambda *args: emitted.append(args))

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            worker._graph_write_plans = {
                ("corridor_1", "comp"): [
                    {
                        "node_id": "write_1",
                        "stream_label": "comp",
                        "initialized": True,
                        "closed": False,
                        "output_fmt": "png",
                        "out_dir": out_dir,
                        "img_ext": ".png",
                        "first_path": out_dir / "0000.png",
                        "png_compression": 6,
                        "png_bit_depth": 8,
                        "jpg_quality": 90,
                        "created_paths": set(),
                    }
                ]
            }

            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            worker._stream_graph_write_frame("corridor_1", "comp", frame, 0, is_video=False)

        self.assertTrue(emitted)
        payload = emitted[0][1]
        self.assertEqual(payload.get("semantics"), "preview_only")

    def test_stream_graph_write_frame_initializes_all_matching_image_plans(self):
        worker = InferenceWorker.__new__(InferenceWorker)
        worker._graph_start_frame = 10
        worker._graph_stream_saved_paths = {}
        worker._graph_fps = 25.0
        emitted = []
        worker.graph_stream_preview = SimpleNamespace(emit=lambda *args: emitted.append(args))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_path = tmp_path / "clip.mp4"
            source_path.write_bytes(b"0")
            worker._graph_source_path = str(source_path)
            alpha_dir = tmp_path / "alpha_out"
            review_dir = tmp_path / "review_out"
            worker._graph_write_plans = {
                ("corridor_1", "alpha"): [
                    {
                        "node_id": "write_alpha",
                        "stream_label": "alpha",
                        "write_cfg": {"output_dir": str(alpha_dir), "output_format": "png"},
                        "initialized": False,
                        "closed": False,
                    },
                    {
                        "node_id": "write_review",
                        "stream_label": "alpha",
                        "write_cfg": {"output_dir": str(review_dir), "output_format": "png"},
                        "initialized": False,
                        "closed": False,
                    },
                ]
            }

            frame = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
            worker._stream_graph_write_frame("corridor_1", "alpha", frame, 2, is_video=True)

            plans = worker._graph_write_plans[("corridor_1", "alpha")]
            for plan, out_dir in zip(plans, (alpha_dir, review_dir)):
                self.assertTrue(plan["initialized"])
                self.assertEqual(plan["out_dir"], out_dir)
                self.assertIn(out_dir / "0002.png", plan["created_paths"])
                self.assertTrue((out_dir / "0002.png").exists())
                self.assertEqual(worker._graph_stream_saved_paths[plan["node_id"]], out_dir / "0001.png")

        self.assertEqual([event[0] for event in emitted], ["write_alpha", "write_review"])
        self.assertEqual([event[2] for event in emitted], [12, 12])
        self.assertTrue(all(event[1].get("semantics") == "production_safe" for event in emitted))

    def test_finalize_graph_stream_writes_closes_video_plan_and_records_output(self):
        class _Writer:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        worker = InferenceWorker.__new__(InferenceWorker)
        worker._graph_audio_path = ""
        worker._graph_stream_saved_paths = {}
        emitted = []
        worker.graph_stream_preview = SimpleNamespace(emit=lambda *args: emitted.append(args))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "clip_tmp.mp4"
            final_path = Path(tmp_dir) / "clip.mp4"
            tmp_path.write_bytes(b"video")
            writer = _Writer()
            worker._graph_write_plans = {
                ("corridor_1", "alpha"): [
                    {
                        "node_id": "write_video",
                        "stream_label": "alpha",
                        "initialized": True,
                        "closed": False,
                        "writer": writer,
                        "tmp_path": tmp_path,
                        "final_path": final_path,
                        "created_paths": {tmp_path},
                    }
                ]
            }

            worker._finalize_graph_stream_writes(keep_outputs=True, emit_preview=True)

            self.assertTrue(writer.closed)
            self.assertFalse(tmp_path.exists())
            self.assertTrue(final_path.exists())
            self.assertEqual(worker._graph_stream_saved_paths["write_video"], final_path)
            self.assertIn(final_path, worker._graph_write_plans[("corridor_1", "alpha")][0]["created_paths"])

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0][0], "write_video")
        self.assertEqual(emitted[0][1]["semantics"], "production_safe")

    def test_sam3_node_uses_persisted_masks_like_sam(self):
        worker = InferenceWorker.__new__(InferenceWorker)
        worker.cancel_flag = SimpleNamespace(is_set=lambda: False)
        worker._graph_start_frame = 20

        stream_calls = []
        progress_calls = []
        worker._stream_graph_write_frame = lambda *args, **kwargs: stream_calls.append((args, kwargs))
        worker.node_frame_progress = SimpleNamespace(emit=lambda *args: progress_calls.append(args))

        mask_a = np.ones((2, 2), dtype=np.uint8) * 255
        mask_b = np.zeros((2, 2), dtype=np.uint8)
        worker._load_sam_masks_from_payloads = lambda _props, _shape: {
            20: mask_a,
            22: mask_b,
        }

        frames = [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.uint8),
        ]
        node_data = {"id": "sam3_1", "properties": {"concept": "person"}}

        result = worker._execute_node("sam3", node_data, {"img": frames})

        self.assertIn("out", result)
        self.assertEqual(len(result["out"]), 3)
        self.assertTrue(np.array_equal(result["out"][0], mask_a))
        self.assertTrue(np.array_equal(result["out"][1], mask_a))
        self.assertTrue(np.array_equal(result["out"][2], mask_b))
        self.assertEqual(len(stream_calls), 3)
        self.assertEqual(len(progress_calls), 3)

    def test_merge_mask_limits_composite_region(self):
        fg = np.zeros((2, 2, 3), dtype=np.float32)
        fg[:, :, 0] = 1.0
        bg = np.zeros((2, 2, 3), dtype=np.float32)
        bg[:, :, 2] = 1.0
        mask = np.array(
            [
                [255, 0],
                [0, 255],
            ],
            dtype=np.uint8,
        )

        out = InferenceWorker._apply_merge_blend(
            fg,
            bg,
            mode="over",
            opacity=1.0,
            mask=mask,
        )

        # masked pixels use FG over BG, unmasked pixels stay as BG
        self.assertGreater(float(out[0, 0, 0]), 0.9)
        self.assertLess(float(out[0, 0, 2]), 0.1)
        self.assertLess(float(out[0, 1, 0]), 0.1)
        self.assertGreater(float(out[0, 1, 2]), 0.9)
        self.assertLess(float(out[1, 0, 0]), 0.1)
        self.assertGreater(float(out[1, 0, 2]), 0.9)
        self.assertGreater(float(out[1, 1, 0]), 0.9)
        self.assertLess(float(out[1, 1, 2]), 0.1)

    def test_merge_mix_dissolves_between_bg_and_result(self):
        fg = np.zeros((1, 1, 3), dtype=np.float32)
        fg[:, :, 0] = 1.0
        bg = np.zeros((1, 1, 3), dtype=np.float32)
        bg[:, :, 2] = 1.0

        out = InferenceWorker._apply_merge_blend(
            fg,
            bg,
            mode="over",
            opacity=1.0,
            mix=0.5,
        )

        self.assertAlmostEqual(float(out[0, 0, 0]), 0.5, places=2)
        self.assertAlmostEqual(float(out[0, 0, 2]), 0.5, places=2)

    def test_merge_invert_mask_flips_affected_region(self):
        fg = np.zeros((1, 2, 3), dtype=np.float32)
        fg[:, :, 0] = 1.0
        bg = np.zeros((1, 2, 3), dtype=np.float32)
        bg[:, :, 2] = 1.0
        mask = np.array([[255, 0]], dtype=np.uint8)

        out = InferenceWorker._apply_merge_blend(
            fg,
            bg,
            mode="over",
            opacity=1.0,
            mask=mask,
            invert_mask=True,
        )

        self.assertLess(float(out[0, 0, 0]), 0.1)
        self.assertGreater(float(out[0, 0, 2]), 0.9)
        self.assertGreater(float(out[0, 1, 0]), 0.9)
        self.assertLess(float(out[0, 1, 2]), 0.1)

    def test_merge_fringe_limits_effect_to_mask_edge(self):
        fg = np.zeros((5, 5, 3), dtype=np.float32)
        fg[:, :, 0] = 1.0
        bg = np.zeros((5, 5, 3), dtype=np.float32)
        bg[:, :, 2] = 1.0
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1:4, 1:4] = 255

        out = InferenceWorker._apply_merge_blend(
            fg,
            bg,
            mode="over",
            opacity=1.0,
            mask=mask,
            fringe=True,
        )

        # center stays as background, edge pixels get the merge effect
        self.assertLess(float(out[2, 2, 0]), 0.1)
        self.assertGreater(float(out[2, 2, 2]), 0.9)
        self.assertGreater(float(out[1, 2, 0]), 0.9)
        self.assertLess(float(out[1, 2, 2]), 0.1)

    def test_merge_mask_channel_red_uses_only_red_channel(self):
        fg = np.zeros((1, 2, 3), dtype=np.float32)
        fg[:, :, 0] = 1.0
        bg = np.zeros((1, 2, 3), dtype=np.float32)
        bg[:, :, 2] = 1.0
        mask_rgb = np.zeros((1, 2, 3), dtype=np.uint8)
        mask_rgb[0, 0, 0] = 255
        mask_rgb[0, 1, 1] = 255

        out = InferenceWorker._apply_merge_blend(
            fg,
            bg,
            mode="over",
            opacity=1.0,
            mask=mask_rgb,
            mask_channel="red",
        )

        self.assertGreater(float(out[0, 0, 0]), 0.9)
        self.assertLess(float(out[0, 0, 2]), 0.1)
        self.assertLess(float(out[0, 1, 0]), 0.1)
        self.assertGreater(float(out[0, 1, 2]), 0.9)

    def test_merge_mask_enabled_false_ignores_mask_input(self):
        fg = np.zeros((1, 1, 3), dtype=np.float32)
        fg[0, 0, 0] = 1.0
        bg = np.zeros((1, 1, 3), dtype=np.float32)
        bg[0, 0, 2] = 1.0
        mask = np.zeros((1, 1), dtype=np.uint8)

        out = InferenceWorker._apply_merge_blend(
            fg,
            bg,
            mode="over",
            opacity=1.0,
            mask=mask,
            mask_enabled=False,
        )

        self.assertGreater(float(out[0, 0, 0]), 0.9)
        self.assertLess(float(out[0, 0, 2]), 0.1)

    def test_merge_mask_inject_overrides_output_alpha_with_mask(self):
        fg = np.zeros((1, 2, 4), dtype=np.float32)
        fg[0, :, 0] = 1.0
        fg[0, :, 3] = 1.0
        bg = np.zeros((1, 2, 4), dtype=np.float32)
        bg[0, :, 2] = 1.0
        bg[0, :, 3] = 1.0
        mask = np.array([[255, 64]], dtype=np.uint8)

        out = InferenceWorker._apply_merge_blend(
            fg,
            bg,
            mode="over",
            opacity=1.0,
            mask=mask,
            mask_inject=True,
        )

        self.assertAlmostEqual(float(out[0, 0, 3]), 1.0, places=3)
        self.assertAlmostEqual(float(out[0, 1, 3]), 64.0 / 255.0, places=3)

    def test_merge_alpha_masking_false_blends_alpha_numerically(self):
        fg = np.zeros((1, 1, 4), dtype=np.float32)
        fg[0, 0, 0] = 1.0
        fg[0, 0, 3] = 0.5
        bg = np.zeros((1, 1, 4), dtype=np.float32)
        bg[0, 0, 2] = 1.0
        bg[0, 0, 3] = 0.25

        out_default = InferenceWorker._apply_merge_blend(
            fg,
            bg,
            mode="multiply",
            opacity=1.0,
            alpha_masking=True,
        )
        out_numeric = InferenceWorker._apply_merge_blend(
            fg,
            bg,
            mode="multiply",
            opacity=1.0,
            alpha_masking=False,
        )

        self.assertNotAlmostEqual(float(out_default[0, 0, 3]), float(out_numeric[0, 0, 3]), places=3)

    def test_merge_set_bbox_intersection_clips_to_overlap(self):
        worker = InferenceWorker.__new__(InferenceWorker)
        worker.cancel_flag = SimpleNamespace(is_set=lambda: False)
        worker._graph_start_frame = 0
        worker._stream_graph_write_frame = lambda *args, **kwargs: None
        worker.node_frame_progress = SimpleNamespace(emit=lambda *args: None)
        worker.stage_progress = SimpleNamespace(emit=lambda *args: None)
        worker.graph_stream_preview = SimpleNamespace(emit=lambda *args: None)
        worker.log_message = SimpleNamespace(emit=lambda *args: None)
        worker._tr = lambda key: key

        fg = np.zeros((3, 3, 4), dtype=np.float32)
        fg[0:2, 0:2, 0] = 1.0
        fg[0:2, 0:2, 3] = 1.0
        bg = np.zeros((3, 3, 4), dtype=np.float32)
        bg[1:3, 1:3, 2] = 1.0
        bg[1:3, 1:3, 3] = 1.0

        out = worker._execute_merge_node(
            {
                "id": "merge_1",
                "properties": {
                    "mode": "over",
                    "opacity": 1.0,
                    "set_bbox_to": "intersection",
                },
            },
            {
                "fg": [fg],
                "bg": [bg],
                "__meta__fg": {"bbox_sequence": [(0, 0, 2, 2)]},
                "__meta__bg": {"bbox_sequence": [(1, 1, 3, 3)]},
            },
        )

        frame = out["out"][0]
        self.assertEqual(tuple(out["__meta__"]["out"]["bbox_sequence"][0]), (1, 1, 2, 2))
        self.assertGreater(float(frame[1, 1, 3]), 0.9)
        self.assertLess(float(frame[0, 0, 3]), 0.1)
        self.assertLess(float(frame[2, 2, 3]), 0.1)

    def test_merge_set_bbox_a_uses_foreground_extents(self):
        worker = InferenceWorker.__new__(InferenceWorker)
        worker.cancel_flag = SimpleNamespace(is_set=lambda: False)
        worker._graph_start_frame = 0
        worker._stream_graph_write_frame = lambda *args, **kwargs: None
        worker.node_frame_progress = SimpleNamespace(emit=lambda *args: None)
        worker.stage_progress = SimpleNamespace(emit=lambda *args: None)
        worker.graph_stream_preview = SimpleNamespace(emit=lambda *args: None)
        worker.log_message = SimpleNamespace(emit=lambda *args: None)
        worker._tr = lambda key: key

        fg = np.zeros((3, 3, 4), dtype=np.float32)
        fg[0:2, 0:2, 0] = 1.0
        fg[0:2, 0:2, 3] = 1.0
        bg = np.zeros((3, 3, 4), dtype=np.float32)
        bg[:, :, 2] = 1.0
        bg[:, :, 3] = 1.0

        out = worker._execute_merge_node(
            {
                "id": "merge_2",
                "properties": {
                    "mode": "over",
                    "opacity": 1.0,
                    "set_bbox_to": "a",
                },
            },
            {
                "fg": [fg],
                "bg": [bg],
                "__meta__fg": {"bbox_sequence": [(0, 0, 2, 2)]},
                "__meta__bg": {"bbox_sequence": [(0, 0, 3, 3)]},
            },
        )

        frame = out["out"][0]
        self.assertEqual(tuple(out["__meta__"]["out"]["bbox_sequence"][0]), (0, 0, 2, 2))
        self.assertGreater(float(frame[0, 0, 3]), 0.9)
        self.assertLess(float(frame[2, 2, 3]), 0.1)

    def test_sam3_node_runs_concept_inference_when_no_persisted_masks(self):
        worker = InferenceWorker.__new__(InferenceWorker)
        worker.cancel_flag = SimpleNamespace(is_set=lambda: False)
        worker._graph_start_frame = 0

        stream_calls = []
        progress_calls = []
        worker._stream_graph_write_frame = lambda *args, **kwargs: stream_calls.append((args, kwargs))
        worker.node_frame_progress = SimpleNamespace(emit=lambda *args: progress_calls.append(args))
        worker._load_sam_masks_from_payloads = lambda _props, _shape: {}

        frames = [
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4, 3), dtype=np.uint8),
        ]
        node_data = {
            "id": "sam3_2",
            "properties": {
                "model_type": "sam3",
                "concept": "person",
            },
        }

        m0 = np.zeros((4, 4), dtype=np.uint8)
        m0[:2, :2] = 255
        m1 = np.zeros((4, 4), dtype=np.uint8)
        m1[2:, 2:] = 255
        expected = np.where((m0 > 127) | (m1 > 127), 255, 0).astype(np.uint8)
        with patch("app.workers.inference_worker.Sam3Service.predict_image", return_value=[m0, m1]) as mocked:
            result = worker._execute_node("sam3", node_data, {"img": frames})

        self.assertEqual(mocked.call_count, 2)
        self.assertIn("out", result)
        self.assertEqual(len(result["out"]), 2)
        self.assertTrue(np.array_equal(result["out"][0], expected))
        self.assertTrue(np.array_equal(result["out"][1], expected))
        self.assertEqual(len(stream_calls), 2)
        self.assertEqual(len(progress_calls), 2)

    def test_sam3_node_ignores_legacy_prompt_points(self):
        worker = InferenceWorker.__new__(InferenceWorker)
        worker.cancel_flag = SimpleNamespace(is_set=lambda: False)
        worker._graph_start_frame = 0

        stream_calls = []
        progress_calls = []
        worker._stream_graph_write_frame = lambda *args, **kwargs: stream_calls.append((args, kwargs))
        worker.node_frame_progress = SimpleNamespace(emit=lambda *args: progress_calls.append(args))
        worker._load_sam_masks_from_payloads = lambda _props, _shape: {}

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        node_data = {
            "id": "sam3_3",
            "properties": {
                "model_type": "sam3",
                "concept": "person",
                "prompt_points": [[1, 2], [3, 1]],
                "prompt_labels": [1, 0],
            },
        }

        with patch(
            "app.workers.inference_worker.Sam3Service.predict_image",
            return_value=[np.ones((4, 4), dtype=np.uint8) * 255],
        ) as mocked:
            result = worker._execute_node("sam3", node_data, {"img": [frame]})

        self.assertIn("out", result)
        self.assertEqual(len(result["out"]), 1)
        self.assertEqual(len(stream_calls), 1)
        self.assertEqual(len(progress_calls), 1)
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs.get("points"), [])
        self.assertEqual(kwargs.get("concept"), "person")

    def test_corridorkey_output_mode_builds_expected_rgba(self):
        src = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)
        fg = np.array([[[0.7, 0.6, 0.5]]], dtype=np.float32)
        alpha = np.array([[0.4]], dtype=np.float32)

        matte = build_corridorkey_processed_output("matte_only", src, fg, alpha)
        fore = build_corridorkey_processed_output("foreground_only", src, fg, alpha)
        sm = build_corridorkey_processed_output("source_matte", src, fg, alpha)
        proc = build_corridorkey_processed_output("processed", src, fg, alpha)

        self.assertEqual(matte.shape, (1, 1, 4))
        self.assertAlmostEqual(float(matte[0, 0, 0]), 0.4, places=5)
        self.assertAlmostEqual(float(matte[0, 0, 3]), 0.4, places=5)

        self.assertAlmostEqual(float(fore[0, 0, 0]), 0.7, places=5)
        self.assertAlmostEqual(float(fore[0, 0, 1]), 0.6, places=5)
        self.assertAlmostEqual(float(fore[0, 0, 2]), 0.5, places=5)
        self.assertAlmostEqual(float(fore[0, 0, 3]), 0.4, places=5)

        self.assertAlmostEqual(float(sm[0, 0, 0]), 0.1, places=5)
        self.assertAlmostEqual(float(sm[0, 0, 1]), 0.2, places=5)
        self.assertAlmostEqual(float(sm[0, 0, 2]), 0.3, places=5)
        self.assertAlmostEqual(float(sm[0, 0, 3]), 0.4, places=5)

        # "processed" mode: linear premultiplied — RGB = srgb_to_linear(fg) * alpha, A = alpha
        self.assertAlmostEqual(float(proc[0, 0, 0]), 0.1791953649767533, places=4)
        self.assertAlmostEqual(float(proc[0, 0, 1]), 0.12741871125003676, places=4)
        self.assertAlmostEqual(float(proc[0, 0, 2]), 0.08561645619289303, places=4)
        self.assertAlmostEqual(float(proc[0, 0, 3]), 0.4, places=5)

    def test_corridorkey_output_helpers_coerce_shapes_and_ranges(self):
        alpha = coerce_alpha_2d(np.array([[[0.2, 0.9], [1.2, -1.0]]], dtype=np.float32))
        self.assertIsNotNone(alpha)
        np.testing.assert_allclose(alpha, np.array([[0.2, 1.0]], dtype=np.float32))

        rgb = coerce_rgb_float01(np.array([[[0, 128, 255, 99]]], dtype=np.uint8))
        self.assertIsNotNone(rgb)
        np.testing.assert_allclose(rgb, np.array([[[0.0, 128.0 / 255.0, 1.0]]], dtype=np.float32))

        self.assertIsNone(coerce_alpha_2d(np.zeros((1,), dtype=np.float32)))
        self.assertIsNone(coerce_rgb_float01(np.zeros((1, 1), dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
