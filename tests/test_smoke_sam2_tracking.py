import importlib
import os
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication, QPlainTextEdit

from app.sam_runtime_state import SamRuntimeState
from app.coordinators.sam_graph_coordinator import Sam2GraphCoordinator
from app.services.sam2_service import Sam2Service
from app.node_graph.sam3_properties_panel import Sam3PropertiesPanel
from app.workers.sam_mask_worker import SamMaskWorker
from app.workers.inference_worker import InferenceWorker

from app.utils.frame_range_helper import (
    FrameRangeController,
    calculate_end_frame,
    calculate_frame_count,
)


class SmokeImportTests(unittest.TestCase):
    def test_import_main_module(self):
        module = importlib.import_module("main")
        self.assertIsNotNone(module)

    def test_import_sam_controller_module(self):
        module = importlib.import_module("app.node_graph.nodes.sam_controller")
        self.assertTrue(hasattr(module, "Sam2NodeController"))


class FrameRangeHelperTests(unittest.TestCase):
    def test_count_updates_end_frame(self):
        state = FrameRangeController.on_count_changed(10, 0, 5)
        self.assertEqual(state.updated_end_frame, 14)

    def test_end_frame_updates_count(self):
        state = FrameRangeController.on_end_frame_changed(10, 14, 0)
        self.assertEqual(state.updated_frame_count, 5)

    def test_calculate_frame_count(self):
        self.assertEqual(calculate_frame_count(5, 15), 11)

    def test_calculate_end_frame(self):
        self.assertEqual(calculate_end_frame(5, 11), 15)


class MainWindowFrameBoundsTests(unittest.TestCase):
    def _make_spin(self, value: int):
        return SimpleNamespace(value=lambda: value)

    class _SpinBoxStub:
        def __init__(self, value: int):
            self.current = value
            self.block_calls = []

        def setValue(self, value: int):
            self.current = value

        def value(self):
            return self.current

        def blockSignals(self, blocked: bool):
            self.block_calls.append(blocked)

    def test_resolve_effective_video_frame_bounds_with_count_priority(self):
        main_module = importlib.import_module("main")
        window = main_module.MainWindow.__new__(main_module.MainWindow)
        window.is_video_input = True
        window.all_frames = [object()] * 100
        window.ui = SimpleNamespace(
            spin_start_frame=self._make_spin(10),
            spin_num_frames=self._make_spin(5),
            spin_end_frame=self._make_spin(99),
        )

        start, end = window._resolve_effective_video_frame_bounds()

        self.assertEqual((start, end), (10, 15))

    def test_resolve_effective_video_frame_bounds_with_end_frame(self):
        main_module = importlib.import_module("main")
        window = main_module.MainWindow.__new__(main_module.MainWindow)
        window.is_video_input = True
        window.all_frames = [object()] * 100
        window.ui = SimpleNamespace(
            spin_start_frame=self._make_spin(10),
            spin_num_frames=self._make_spin(0),
            spin_end_frame=self._make_spin(20),
        )

        start, end = window._resolve_effective_video_frame_bounds()

        self.assertEqual((start, end), (10, 21))

    def test_apply_loaded_media_frame_range_updates_video_values(self):
        main_module = importlib.import_module("main")
        window = main_module.MainWindow.__new__(main_module.MainWindow)
        start_spin = self._SpinBoxStub(9)
        count_spin = self._SpinBoxStub(0)
        end_spin = self._SpinBoxStub(0)
        window.ui = SimpleNamespace(
            spin_start_frame=start_spin,
            spin_num_frames=count_spin,
            spin_end_frame=end_spin,
        )

        window._apply_loaded_media_frame_range(42, True)

        self.assertEqual(start_spin.value(), 0)
        self.assertEqual(count_spin.value(), 42)
        self.assertEqual(end_spin.value(), 41)

    def test_apply_loaded_media_frame_range_updates_image_values(self):
        main_module = importlib.import_module("main")
        window = main_module.MainWindow.__new__(main_module.MainWindow)
        start_spin = self._SpinBoxStub(9)
        count_spin = self._SpinBoxStub(7)
        end_spin = self._SpinBoxStub(8)
        window.ui = SimpleNamespace(
            spin_start_frame=start_spin,
            spin_num_frames=count_spin,
            spin_end_frame=end_spin,
        )

        window._apply_loaded_media_frame_range(1, False)

        self.assertEqual(start_spin.value(), 0)
        self.assertEqual(count_spin.value(), 1)
        self.assertEqual(end_spin.value(), 0)


class SamMaskWorkerRemapTests(unittest.TestCase):
    def test_sam3_properties_preserve_concept_prompt(self):
        app = QApplication.instance() or QApplication([])
        self.assertIsNotNone(app)

        panel = Sam3PropertiesPanel(lambda key: key)
        props = {
            "model_type": "sam3",
            "concept": "person",
            "point_mode": "negative",
            "live_sam2": True,
            "prompt_points": [[1, 2]],
            "prompt_labels": [1],
            "mask_items": ["old mask"],
            "selected_mask_rows": [0],
        }

        panel.load_from_properties(props)
        self.assertIsInstance(panel.concept_edit, QPlainTextEdit)
        self.assertGreaterEqual(panel.concept_edit.height(), panel.concept_edit.fontMetrics().lineSpacing() * 3)
        self.assertEqual(panel.concept_edit.toPlainText(), "person")
        self.assertFalse(hasattr(panel, "btn_positive"))
        self.assertFalse(hasattr(panel, "masks_list"))

        panel.concept_edit.setPlainText("player in red")
        panel.write_to_properties(props)

        self.assertEqual(props["concept"], "player in red")
        self.assertNotIn("point_mode", props)
        self.assertNotIn("live_sam2", props)
        self.assertNotIn("prompt_points", props)
        self.assertNotIn("prompt_labels", props)
        self.assertNotIn("mask_items", props)
        self.assertNotIn("selected_mask_rows", props)

    def test_sam3_worker_passes_concept_context_to_service(self):
        worker = SamMaskWorker()
        worker._backend = "sam3"
        worker._service_backend = "sam3"

        class _Service:
            def __init__(self):
                self.context = None

            def generate_mask(self, _image, _points, _labels, context=None):
                self.context = context
                return np.ones((2, 2), dtype=np.uint8) * 255

        service = _Service()
        worker.sam_service = service

        received = []
        errors = []
        worker.finished.connect(received.append)
        worker.error.connect(errors.append)

        worker.generate_mask(
            np.zeros((2, 2, 3), dtype=np.uint8),
            [],
            [],
            {"concept": "person"},
        )

        self.assertEqual(errors, [])
        self.assertEqual(service.context, {"concept": "person"})
        self.assertEqual(len(received), 1)

    def test_generate_mask_none_emits_error(self):
        worker = SamMaskWorker()
        worker._backend = "sam2"
        worker._service_backend = "sam2"

        class _Service:
            def generate_mask(self, *_args, **_kwargs):
                return None

        worker.sam_service = _Service()

        received = []
        errors = []
        worker.finished.connect(received.append)
        worker.error.connect(errors.append)

        worker.generate_mask(
            np.zeros((4, 4, 3), dtype=np.uint8),
            [(1, 1)],
            [1],
        )

        self.assertEqual(received, [])
        self.assertTrue(errors)

    def test_generate_mask_soft_float_thresholded(self):
        worker = SamMaskWorker()
        worker._backend = "sam2"
        worker._service_backend = "sam2"

        class _Service:
            def generate_mask(self, *_args, **_kwargs):
                return np.array(
                    [
                        [0.0, 0.2],
                        [0.8, 1.0],
                    ],
                    dtype=np.float32,
                )

        worker.sam_service = _Service()

        received = []
        errors = []
        worker.finished.connect(received.append)
        worker.error.connect(errors.append)

        worker.generate_mask(
            np.zeros((2, 2, 3), dtype=np.uint8),
            [(1, 1)],
            [1],
        )

        self.assertEqual(errors, [])
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].get("op"), "generate")
        mask = np.asarray(received[0].get("mask"), dtype=np.uint8)
        self.assertTrue(np.array_equal(mask, np.array([[0, 0], [255, 255]], dtype=np.uint8)))

    def test_propagate_masks_remaps_sequence_indices_to_global_timeline(self):
        worker = SamMaskWorker()
        worker._backend = "sam2"
        worker._service_backend = "sam2"

        class FakeSamService:
            def propagate_with_prompt(self, _frames, _points, _labels, *, start_index, direction, reset_session):
                self.last_call = {
                    "start_index": start_index,
                    "direction": direction,
                    "reset_session": reset_session,
                }
                return {
                    "sequence_masks_map": {
                        0: np.zeros((2, 2), dtype=np.uint8),
                        1: np.ones((2, 2), dtype=np.uint8) * 255,
                    }
                }

        worker.sam_service = FakeSamService()

        received = []
        worker.finished.connect(received.append)

        worker.propagate_masks(
            {
                "frames": [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)],
                "points": [(1, 1)],
                "labels": [1],
                "current_frame_index": 0,
                "current_frame_index_global": 12,
                "frame_index_offset": 12,
                "direction": "forward",
            }
        )

        self.assertEqual(len(received), 1)
        payload = received[0]
        self.assertEqual(payload["current_frame_index"], 12)
        self.assertEqual(sorted(payload["sequence_masks_map"].keys()), [12, 13])

    def test_propagate_masks_cancelled_emits_cancelled_payload(self):
        worker = SamMaskWorker()
        worker._backend = "sam2"
        worker._service_backend = "sam2"

        class _CancelledService:
            def __init__(self):
                self.unloaded = False

            def propagate_with_prompt(self, *_args, **_kwargs):
                raise RuntimeError("__SAM_CANCELLED__")

            def unload(self):
                self.unloaded = True

        service = _CancelledService()
        worker.sam_service = service

        received = []
        errors = []
        worker.finished.connect(received.append)
        worker.error.connect(errors.append)

        worker.propagate_masks(
            {
                "frames": [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)],
                "points": [(1, 1)],
                "labels": [1],
                "current_frame_index": 0,
                "direction": "forward",
            }
        )

        self.assertEqual(received, [{"op": "cancelled"}])
        self.assertEqual(errors, [])
        self.assertTrue(service.unloaded)
        self.assertIsNone(worker.sam_service)
        self.assertEqual(worker._service_backend, "")

    def test_propagate_masks_error_unloads_service(self):
        worker = SamMaskWorker()
        worker._backend = "sam2"
        worker._service_backend = "sam2"

        class _FailingService:
            def __init__(self):
                self.unloaded = False

            def propagate_with_prompt(self, *_args, **_kwargs):
                raise RuntimeError("boom")

            def unload(self):
                self.unloaded = True

        service = _FailingService()
        worker.sam_service = service

        received = []
        errors = []
        worker.finished.connect(received.append)
        worker.error.connect(errors.append)

        worker.propagate_masks(
            {
                "frames": [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)],
                "points": [(1, 1)],
                "labels": [1],
                "current_frame_index": 0,
                "direction": "forward",
            }
        )

        self.assertEqual(received, [])
        self.assertEqual(errors, ["boom"])
        self.assertTrue(service.unloaded)
        self.assertIsNone(worker.sam_service)
        self.assertEqual(worker._service_backend, "")


class Sam2ProgressFormatTests(unittest.TestCase):
    def test_generate_mask_raises_when_native_predictor_unavailable(self):
        service = Sam2Service.__new__(Sam2Service)
        service._native_enabled = False
        service._predictor = None
        service._native_failed_reason = "native image predictor unavailable"
        service._load_native_predictor = lambda: None
        service._raise_if_cancelled = lambda: None
        service._emit_progress = lambda *_args, **_kwargs: None
        service._tr = lambda key: key

        with self.assertRaises(RuntimeError):
            service.generate_mask(
                np.zeros((4, 4, 3), dtype=np.uint8),
                [(1, 1)],
                [1],
            )

    def test_generate_mask_thresholds_logits_like_upstream(self):
        class _Predictor:
            mask_threshold = 0.0

            def set_image(self, _image):
                return None

            def predict(self, **_kwargs):
                masks = np.array(
                    [[[-0.2, 0.2], [1.0, -1.0]]],
                    dtype=np.float32,
                )
                scores = np.array([0.9], dtype=np.float32)
                logits = np.zeros((1, 2, 2), dtype=np.float32)
                return masks, scores, logits

        service = Sam2Service.__new__(Sam2Service)
        service._native_enabled = True
        service._predictor = _Predictor()
        service._native_failed_reason = ""
        service._load_native_predictor = lambda: None
        service._raise_if_cancelled = lambda: None
        service._emit_progress = lambda *_args, **_kwargs: None
        service._tr = lambda key: key

        mask = service.generate_mask(
            np.zeros((2, 2, 3), dtype=np.uint8),
            [(1, 1)],
            [1],
            multimask_output=False,
            auto_refine=False,
            report_progress=False,
        )

        self.assertTrue(
            np.array_equal(
                np.asarray(mask, dtype=np.uint8),
                np.array([[0, 255], [255, 0]], dtype=np.uint8),
            )
        )

    def test_propagate_progress_status_includes_direction_percent_and_counts(self):
        service = Sam2Service.__new__(Sam2Service)
        captured_progress = []

        translations = {
            "sam2_status_session_ready": "ready",
            "sam2_direction_forward": "forward",
            "sam2_direction_backward": "backward",
            "worker_sam2_sequence_frame": "SAM2: tracking {direction} {percent}% ({current}/{total} frames)",
            "sam2_sequence_ready": "done",
        }

        service._tr = lambda key: translations[key]
        service._emit_progress = lambda percent, message: captured_progress.append((percent, message))
        service._emit_frame_progress = lambda current, total: None
        service._ensure_video_session = lambda frames_rgb, start_index=0, force_reset=False: setattr(
            service, "_session_frames", list(frames_rgb)
        )
        service.add_reprompt = lambda frames_rgb, frame_index, points, labels, obj_id=1: (
            setattr(service, "_session_points", list(points)),
            setattr(service, "_session_labels", list(labels)),
            setattr(service, "_session_obj_id", obj_id),
            {"frame_index": frame_index, "mask": np.zeros((2, 2), dtype=np.uint8)},
        )[-1]
        service.generate_mask = lambda frame, prompt_points, prompt_labels, **kw: np.zeros((2, 2), dtype=np.uint8)
        service._native_video_enabled = False
        service._video_predictor = None
        service._video_state = None
        service._session_obj_id = 1
        service._session_masks_map = {}

        frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(5)]
        service.propagate_with_prompt(
            frames,
            [(1, 1)],
            [1],
            start_index=2,
            direction="forward",
            reset_session=False,
        )

        tracking_messages = [message for _percent, message in captured_progress if message.startswith("SAM2: tracking")]
        self.assertEqual(
            tracking_messages,
            [
                "SAM2: tracking forward 33% (1/3 frames)",
                "SAM2: tracking forward 66% (2/3 frames)",
                "SAM2: tracking forward 100% (3/3 frames)",
            ],
        )


class SamRuntimeStateFrameMaskTests(unittest.TestCase):
    def test_mask_for_frame_prefers_exact_then_nearest(self):
        state = SamRuntimeState()
        m0 = np.zeros((2, 2), dtype=np.uint8)
        m5 = np.ones((2, 2), dtype=np.uint8) * 255
        state.added_masks = [(0, m0), (5, m5)]

        self.assertTrue(np.array_equal(state.mask_for_frame(5), m5))
        self.assertTrue(np.array_equal(state.mask_for_frame(3), m0))
        self.assertTrue(np.array_equal(state.mask_for_frame(8), m5))

    def test_combined_mask_uses_earliest_frame_when_frame0_absent(self):
        state = SamRuntimeState()
        m3 = np.ones((2, 2), dtype=np.uint8) * 255
        m7 = np.zeros((2, 2), dtype=np.uint8)
        state.added_masks = [(3, m3), (7, m7)]

        combined = state.combined_mask()

        self.assertIsNotNone(combined)
        self.assertTrue(np.array_equal(combined, m3))


class Sam2GraphCoordinatorRestoreTests(unittest.TestCase):
    def test_restore_does_not_clear_runtime_masks_when_graph_has_no_persisted_masks(self):
        class _Signal:
            def __init__(self):
                self.calls = 0

            def emit(self):
                self.calls += 1

        state = SimpleNamespace(
            added_masks=[(0, np.ones((2, 2), dtype=np.uint8) * 255)],
            current_mask=np.ones((2, 2), dtype=np.uint8) * 255,
        )
        mask_list_changed = _Signal()
        sam2 = SimpleNamespace(state=state, mask_list_changed=mask_list_changed)

        dialog = SimpleNamespace(
            sam_node_mask_payloads=lambda: [],
            sam_node_mask_source_path=lambda: "",
        )

        coordinator = Sam2GraphCoordinator(
            sam2,
            get_dialog=lambda: dialog,
            get_input_path=lambda: "",
            get_frame_index=lambda: 0,
        )

        coordinator.restore_masks_from_graph_node()

        self.assertEqual(len(sam2.state.added_masks), 1)
        self.assertIsNotNone(sam2.state.current_mask)
        self.assertEqual(mask_list_changed.calls, 0)


class MainWindowSamGraphBehaviorTests(unittest.TestCase):
    def test_quick_save_custom_preset_persists_frame_range(self):
        main_module = importlib.import_module("main")
        window = main_module.MainWindow.__new__(main_module.MainWindow)

        captured_runtime = []

        class _DialogStub:
            def sync_sam_runtime_state(self, **kwargs):
                captured_runtime.append(kwargs)

            def export_graph_preset(self):
                return {
                    "nodes": [{"id": "sam_1", "type": "sam2", "properties": {}}],
                    "connections": [],
                }

        class _Spin:
            def __init__(self, value):
                self._value = value

            def value(self):
                return self._value

        window._node_graph_dialog = _DialogStub()
        window._selected_graph_preset_key = "custom:my-preset"
        window._graph_custom_presets = {}
        window._save_graph_custom_presets = lambda: None
        window.ui = SimpleNamespace(
            spin_start_frame=_Spin(12),
            spin_num_frames=_Spin(23),
            spin_end_frame=_Spin(34),
        )
        window.sam2_graph = SimpleNamespace(
            sync_to_graph=lambda: None,
            persist_masks=lambda force_disk=False: ("/tmp/mask.png", [{"frame_index": 1, "path": "/tmp/m1.png"}]),
        )

        key = window._quick_save_current_graph_preset()

        self.assertEqual(key, "custom:my-preset")
        self.assertIn("my-preset", window._graph_custom_presets)
        saved = window._graph_custom_presets["my-preset"]
        self.assertEqual(saved.get("start_frame"), 12)
        self.assertEqual(saved.get("num_frames"), 23)
        self.assertEqual(saved.get("end_frame"), 34)
        self.assertTrue(captured_runtime)

    def test_on_sam2_status_changed_skips_graph_sync_when_suspended(self):
        main_module = importlib.import_module("main")
        window = main_module.MainWindow.__new__(main_module.MainWindow)

        sync_calls = []
        window.sam2 = SimpleNamespace(state=SimpleNamespace(set_status=lambda _t: None))
        window.sam2_graph = SimpleNamespace(sync_to_graph=lambda _t: sync_calls.append(True))
        window._node_graph_dialog = object()
        window._suspend_sam2_graph_sync = True
        window._optional_controls_present = False
        window._set_status = lambda _t: None

        window._on_sam2_status_changed("loading")

        self.assertEqual(sync_calls, [])

    def test_on_graph_preset_selected_restores_sam_masks(self):
        main_module = importlib.import_module("main")
        window = main_module.MainWindow.__new__(main_module.MainWindow)

        class _Combo:
            def __init__(self, key):
                self._key = key

            def itemData(self, _index):
                return self._key

        class _DialogStub:
            def graph_is_empty(self):
                return True

            def apply_graph_preset(self, _preset):
                return True

        class _Spin:
            def blockSignals(self, _):
                return None

            def setValue(self, _):
                return None

        restored = []
        window._combo_playback_presets = _Combo("custom:test")
        window._node_graph_dialog = _DialogStub()
        window._selected_graph_preset_key = "empty"
        window._graph_save_preset_key = "save"
        window._graph_delete_preset_key = "delete"
        window._graph_empty_preset_key = "empty"
        window._graph_preset_payload = lambda _key: {"nodes": [], "connections": []}
        window.sam2_graph = SimpleNamespace(restore_masks_from_graph_node=lambda: restored.append(True))
        window.ui = SimpleNamespace(spin_start_frame=_Spin(), spin_end_frame=_Spin())
        window._ensure_matting_orchestrator = lambda: SimpleNamespace(clear_write_outputs=lambda: None)
        window._set_graph_preset_baseline_from_current = lambda: None
        window._refresh_graph_preset_combo = lambda *_args, **_kwargs: None
        window._restore_write_outputs_from_disk = lambda: None

        window._on_graph_preset_selected(0)

        self.assertEqual(restored, [True])

    def test_frame_slider_updates_active_read_node_preview_frame(self):
        main_module = importlib.import_module("main")
        window = main_module.MainWindow.__new__(main_module.MainWindow)

        updated = []

        class _DialogStub:
            def update_active_read_node_preview_frame(self, idx: int):
                updated.append(idx)

        window._node_graph_dialog = _DialogStub()
        window.all_frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(4)]
        window.current_frame_index = 0
        window._render_input_preview = lambda: None
        window._render_output_preview_for_index = lambda _idx: None
        window._update_frame_info = lambda: None
        window._active_node_type = "source"
        window.sam2 = SimpleNamespace(state=SimpleNamespace(mask_for_frame=lambda _idx: None))

        window.on_frame_slider_changed(2)

        self.assertEqual(updated, [2])

    def test_detects_sam_to_matting_mask_link_in_graph(self):
        main_module = importlib.import_module("main")
        window = main_module.MainWindow.__new__(main_module.MainWindow)

        class _DialogStub:
            def export_graph_preset(self):
                return {
                    "nodes": [
                        {"id": "sam_1", "type": "sam2", "properties": {"enabled": True}},
                        {"id": "matting_1", "type": "matting", "properties": {"enabled": True}},
                    ],
                    "connections": [
                        {"src": "sam_1", "dst": "matting_1", "src_port": "out", "dst_port": "mask"}
                    ],
                }

        window._node_graph_dialog = _DialogStub()
        self.assertTrue(window._has_sam2_to_matting_mask_link_in_graph())

    def test_auto_propagate_skipped_when_sam_feeds_matting_mask(self):
        main_module = importlib.import_module("main")
        window = main_module.MainWindow.__new__(main_module.MainWindow)

        window._pending_processing_after_sam2_auto_propagate = False
        window._skip_next_auto_sam2_propagate = False
        window.is_video_input = True
        window.all_frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(12)]
        window.matting = SimpleNamespace(is_active=False)

        status_messages = []

        class _SignalStub:
            def emit(self, text):
                status_messages.append(text)

        window._tr = lambda key: {
            "sam2_auto_propagate_skipped_matting_mask": "Auto SAM2 skipped",
        }.get(key, key)
        window.sam2 = SimpleNamespace(
            generation_active=False,
            status_changed=_SignalStub(),
            state=SimpleNamespace(
                current_mask=np.ones((2, 2), dtype=np.uint8) * 255,
                added_masks=[(0, np.ones((2, 2), dtype=np.uint8) * 255)],
                mask_path="",
                points=[(1, 1)],
            ),
        )

        class _DialogStub:
            def export_graph_preset(self):
                return {
                    "nodes": [
                        {"id": "sam_1", "type": "sam2", "properties": {"enabled": True}},
                        {"id": "matting_1", "type": "matting", "properties": {"enabled": True}},
                    ],
                    "connections": [
                        {"src": "sam_1", "dst": "matting_1", "src_port": "out", "dst_port": "mask"}
                    ],
                }

        window._node_graph_dialog = _DialogStub()

        triggered = []
        window._on_graph_sam2_propagate_requested = lambda direction: triggered.append(direction)

        started = window._try_auto_propagate_sam2_before_processing()

        self.assertFalse(started)
        self.assertEqual(triggered, [])
        self.assertEqual(status_messages, ["Auto SAM2 skipped"])

    def test_graph_preview_request_for_sam_uses_current_frame_mask(self):
        main_module = importlib.import_module("main")
        window = main_module.MainWindow.__new__(main_module.MainWindow)
        requested = []

        window.current_frame_index = 7
        window._selected_export_preview_node_id = "x"
        window._selected_birefnet_preview_node_id = "y"
        window._show_mask_preview_on_output = requested.append
        window._clear_selected_node_preview = lambda: None
        window.sam2 = SimpleNamespace(
            state=SimpleNamespace(mask_for_frame=lambda idx: f"mask@{idx}")
        )

        window._on_graph_preview_request_changed("sam2", {})

        self.assertEqual(requested, ["mask@7"])

    def test_save_sam_outputs_to_connected_write_nodes_uses_write_config(self):
        main_module = importlib.import_module("main")
        window = main_module.MainWindow.__new__(main_module.MainWindow)

        class _DialogStub:
            def connected_write_targets(self):
                return [
                    {
                        "source_node_type": "sam2",
                        "stream": "alpha",
                        "graph_node_id": "write_1",
                        "auto_output_dir": True,
                        "output_dir": "",
                        "output_format": "png",
                        "file_name": "",
                    }
                ]

            def set_write_runtime_preview_for_node(self, _node_id, _image):
                return None

        saves = []
        applied = []

        window._node_graph_dialog = _DialogStub()
        window.input_path = "/tmp/input.mp4"
        window.current_frame_index = 3
        window._default_run_output_dir = lambda _src: Path("/tmp/out")
        window.sam2_graph = SimpleNamespace(
            build_frame_masks=lambda: {
                2: np.ones((2, 2), dtype=np.uint8) * 255,
                3: np.zeros((2, 2), dtype=np.uint8),
            }
        )
        window.sam2 = SimpleNamespace(
            state=SimpleNamespace(
                added_masks=[
                    (2, np.ones((2, 2), dtype=np.uint8) * 255),
                    (3, np.zeros((2, 2), dtype=np.uint8)),
                ],
                current_mask=None,
            )
        )
        window._save_frames_to_write_output = lambda frames, cfg, fallback, default_stem, *, source_is_video, source_ext: (
            saves.append((len(frames), cfg.get("output_dir"), str(fallback), default_stem, source_is_video, source_ext))
            or "/tmp/out/alpha/0001.png"
        )
        window._apply_export_preview_path = lambda node_id, path: applied.append((node_id, path))
        window._to_qimage = lambda _arr: None

        result = window._save_sam2_outputs_to_connected_write_nodes()

        node_count, frame_count = result
        self.assertEqual(node_count, 1)
        self.assertEqual(frame_count, 2)
        self.assertEqual(len(saves), 1)
        self.assertEqual(saves[0][0], 2)
        self.assertEqual(saves[0][1], "/tmp/out/alpha")
        self.assertTrue(saves[0][4])
        self.assertEqual(saves[0][5], ".mp4")
        self.assertTrue(applied)


class InferenceWorkerMaskCoercionTests(unittest.TestCase):
    def test_coerce_matting_mask_from_list_rgb_frame(self):
        rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        rgb[:, :, 1] = 200

        mask = InferenceWorker._coerce_matting_mask([rgb], (3, 4))

        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (3, 4))
        self.assertEqual(mask.dtype, np.uint8)

    def test_coerce_matting_mask_from_sequence_tensor(self):
        seq = np.ones((2, 3, 4), dtype=np.float32)

        mask = InferenceWorker._coerce_matting_mask(seq, (3, 4))

        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (3, 4))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(int(mask.max()), 255)

    def test_coerce_matting_mask_from_sequence_uses_first_frame_only(self):
        seq = np.zeros((2, 3, 4), dtype=np.uint8)
        seq[0, :, :] = 255
        seq[1, :, :] = 0

        mask = InferenceWorker._coerce_matting_mask(seq, (3, 4))

        self.assertTrue(np.array_equal(mask, np.full((3, 4), 255, dtype=np.uint8)))

    def test_on_sam2_generation_finished_shows_immediate_export_status(self):
        main_module = importlib.import_module("main")
        window = main_module.MainWindow.__new__(main_module.MainWindow)

        synced = []
        statusbar = []
        window.ui = SimpleNamespace(progress_bar=SimpleNamespace(setRange=lambda *_: None, setValue=lambda *_: None))
        window.sam2_graph = SimpleNamespace(sync_to_graph=lambda text=None: synced.append(text))
        window._save_sam2_outputs_to_connected_write_nodes = lambda: (2, 51)
        window._set_status = statusbar.append
        window._tr = lambda key: {
            "sam_mask_ready": "mask ready",
            "sam_write_immediate_export_done": "SAM->Write {frames}fr/{count}nd",
        }[key]
        window._active_node_type = ""
        window._show_mask_preview_on_output = lambda *_: None
        window.sam2 = SimpleNamespace(state=SimpleNamespace(mask_for_frame=lambda _idx: None, status_text="seq ready (51)"))
        window.current_frame_index = 0

        window._on_sam2_generation_finished()

        self.assertIn("seq ready (51)", synced)
        self.assertIn("SAM->Write 51fr/2nd", statusbar)

    def test_cancel_processing_routes_to_sam_when_generation_active(self):
        main_module = importlib.import_module("main")
        window = main_module.MainWindow.__new__(main_module.MainWindow)

        sam_cancelled = []
        matting_cancelled = []
        statuses = []

        window._media_loading_active = False
        window._media_loader_worker = None
        window.sam2 = SimpleNamespace(
            generation_active=True,
            cancel_current_operation=lambda: sam_cancelled.append(True),
        )
        window.matting = SimpleNamespace(cancel=lambda: matting_cancelled.append(True))
        window._set_status = statuses.append
        window._tr = lambda key: {"status_cancel": "cancel"}.get(key, key)

        window.cancel_processing()

        self.assertEqual(sam_cancelled, [True])
        self.assertEqual(matting_cancelled, [])
        self.assertIn("cancel", statuses)

    def test_on_cloud_processing_finished_updates_write_node_preview_and_viewer(self):
        main_module = importlib.import_module("main")
        window = main_module.MainWindow.__new__(main_module.MainWindow)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            output_previews = []
            applied_paths = []
            runtime_previews = []
            statuses = []
            sounds = []

            class _DialogStub:
                def set_gvm_cloud_status(self, _value):
                    return None

                def set_write_runtime_preview_for_node(self, node_id, image):
                    runtime_previews.append((node_id, image))

            window.ui = SimpleNamespace(
                btn_run=SimpleNamespace(setEnabled=lambda *_: None),
                progress_bar=SimpleNamespace(setRange=lambda *_: None, setValue=lambda *_: None, maximum=lambda: 100),
            )
            window._refresh_stop_button_state = lambda: None
            window._node_graph_dialog = _DialogStub()
            window._set_status = statuses.append
            window._tr = lambda key: {
                "status_done": "done",
                "status_error": "error",
                "inference_error_title": "error title",
                "cloud_worker_err_no_result": "no result",
            }.get(key, key)
            window._show_output_preview = lambda fg, alpha: output_previews.append((fg, alpha))
            window._apply_export_preview_path = lambda node_id, path: applied_paths.append((node_id, path))
            window._load_preview_image_or_video_frame = lambda path: np.zeros((2, 2, 3), dtype=np.uint8) if path == tmp_path else None
            window._to_qimage = lambda arr: ("qimage", tuple(arr.shape))
            window._play_completion_sound = lambda: sounds.append(True)

            window._on_cloud_processing_finished(
                {
                    "cancelled": False,
                    "result_path": tmp_path,
                    "write_node_id": "write_1",
                }
            )

            self.assertEqual(window.last_output_dir, str(Path(tmp_path).parent))
            self.assertEqual(applied_paths, [("write_1", tmp_path)])
            self.assertEqual(output_previews, [("", tmp_path)])
            self.assertEqual(runtime_previews, [("write_1", ("qimage", (2, 2, 3)))])
            self.assertIn("done", statuses)
            self.assertEqual(sounds, [True])
        finally:
            os.unlink(tmp_path)

    def test_export_preview_request_restores_saved_output_from_connected_target(self):
        from PIL import Image as PILImage

        main_module = importlib.import_module("main")
        window = main_module.MainWindow.__new__(main_module.MainWindow)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            # Write a valid 2×2 PNG so PIL can open it (empty file → UnidentifiedImageError)
            img = PILImage.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))
            img.save(tmp, format="PNG")
            tmp_path = tmp.name

        try:
            applied = []
            export_nodes = []

            orchestrator = SimpleNamespace(
                clear_preview_selection=lambda: None,
                set_export_preview_node=lambda node_id: export_nodes.append(node_id),
                saved_output_path_for_node=lambda _node_id: "",
                resolve_write_output_path=lambda target: tmp_path if target.get("graph_node_id") == "write_1" else "",
                # Viewer controller calls orchestrator.apply_export_preview_path, not window method
                apply_export_preview_path=lambda node_id, path: applied.append((node_id, path)),
            )

            class _DialogStub:
                def connected_write_targets(self):
                    return [{"graph_node_id": "write_1"}]

            # Stubs required by ViewerPreviewController._set_selected_node_preview
            _noop = lambda *_a, **_kw: None  # noqa: E731
            _btn = SimpleNamespace(setEnabled=_noop)
            window.ui = SimpleNamespace(
                btn_split_view=_btn,
                btn_preview_foreground=_btn,
                btn_preview_alpha=_btn,
            )
            window.all_frames = []
            window.current_frame_index = 0
            window._render_output_preview_for_index = _noop

            window._node_graph_dialog = _DialogStub()
            window._ensure_matting_orchestrator = lambda: orchestrator

            window._on_graph_preview_request_changed(
                "export",
                {"graph_node_id": "write_1", "last_output_path": ""},
            )

            self.assertEqual(export_nodes, ["write_1"])
            self.assertEqual(applied, [("write_1", tmp_path)])
            # Viewer loaded the image from disk and set it on the window host
            self.assertTrue(getattr(window, "_selected_node_preview_is_image", False))
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()