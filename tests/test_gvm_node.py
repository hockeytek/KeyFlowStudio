"""Тесты ноды GVM.

Покрывает три блока:
  1. Spec & contract: структура портов, default properties, регистрация
  2. Registry topology: разрешённые/запрещённые соединения
  3. Worker execution: _execute_gvm_node с замоканным GVMService
"""
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("KEYFLOW_DEVICE", "cpu")

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_worker():
    """Создаёт InferenceWorker без Qt-объектов (как в test_stage64)."""
    from app.workers.inference_worker import InferenceWorker

    w = InferenceWorker.__new__(InferenceWorker)
    w.cancel_flag = threading.Event()
    w.language_code = "ru"
    w._graph_output_dir = None
    w._graph_source_path = ""
    w._graph_mask_path = ""
    w._graph_fps = 25.0
    w._graph_audio_path = ""
    w._graph_write_plans = {}
    w._graph_stream_saved_paths = {}
    w._graph_correction_masks = None
    w._graph_start_frame = 0
    w._graph_end_frame = -1
    w._graph_downstream_targets = {}
    w.log_message = type("Sig", (), {"emit": staticmethod(lambda m: None)})()
    w.stage_progress = type("Sig", (), {"emit": staticmethod(lambda p, m: None)})()
    w.graph_stream_preview = type("Sig", (), {"emit": staticmethod(lambda *a: None)})()
    w.node_frame_progress = type("Sig", (), {"emit": staticmethod(lambda *a: None)})()
    return w


def _rgb_frames(n: int = 3, h: int = 8, w: int = 8) -> list:
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]


def _alpha_pngs(output_dir: Path, count: int) -> list[Path]:
    """Записывает синтетические alpha PNG в output_dir, возвращает пути."""
    import cv2
    paths = []
    for i in range(count):
        p = output_dir / f"{i:05d}.png"
        cv2.imwrite(str(p), np.full((8, 8), 128, dtype=np.uint8))
        paths.append(p)
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# 1. Spec & contract
# ──────────────────────────────────────────────────────────────────────────────

class GVMSpecTests(unittest.TestCase):
    def setUp(self):
        from app.node_graph.specs.gvm import SPEC
        self.spec = SPEC

    def test_key_is_gvm(self):
        self.assertEqual(self.spec.key, "gvm")

    def test_has_image_input(self):
        names = [p.name for p in self.spec.inputs]
        self.assertIn("image", names)

    def test_image_input_is_required(self):
        port = next(p for p in self.spec.inputs if p.name == "image")
        self.assertTrue(port.required)

    def test_image_input_data_type_is_image(self):
        port = next(p for p in self.spec.inputs if p.name == "image")
        self.assertEqual(port.data_type, "image")

    def test_has_alpha_output(self):
        names = [p.name for p in self.spec.outputs]
        self.assertIn("alpha", names)

    def test_alpha_output_data_type_is_alpha(self):
        port = next(p for p in self.spec.outputs if p.name == "alpha")
        self.assertEqual(port.data_type, "alpha")

    def test_default_properties_present(self):
        dp = self.spec.default_properties
        self.assertIn("num_frames_per_batch", dp)
        self.assertIn("decode_chunk_size", dp)
        self.assertIn("num_overlap_frames", dp)
        self.assertIn("num_interp_frames", dp)
        self.assertIn("dilate_radius", dp)

    def test_default_num_frames_per_batch(self):
        self.assertEqual(self.spec.default_properties["num_frames_per_batch"], 8)

    def test_default_dilate_radius_is_zero(self):
        self.assertEqual(self.spec.default_properties["dilate_radius"], 0)

    def test_i18n_keys_set(self):
        self.assertEqual(self.spec.title_i18n_key, "node_graph_node_gvm")
        self.assertEqual(self.spec.subtitle_i18n_key, "")


class GVMContractTests(unittest.TestCase):
    def setUp(self):
        from app.node_graph.rules.node_contracts import ALL_NODE_CONTRACTS
        self.contract = ALL_NODE_CONTRACTS.get("gvm")

    def test_contract_exists(self):
        self.assertIsNotNone(self.contract, "GVM must be registered in ALL_NODE_CONTRACTS")

    def test_contract_node_type(self):
        self.assertEqual(self.contract.node_type, "gvm")

    def test_contract_has_image_input(self):
        names = [p.name for p in self.contract.inputs]
        self.assertIn("image", names)

    def test_contract_has_alpha_output(self):
        names = [p.name for p in self.contract.outputs]
        self.assertIn("alpha", names)

    def test_downstream_includes_corridorkey(self):
        self.assertIn("corridorkey", self.contract.downstream_allowed)

    def test_downstream_includes_export(self):
        self.assertIn("export", self.contract.downstream_allowed)

    def test_downstream_does_not_include_gvm(self):
        """GVM не должна цепляться к другой GVM."""
        self.assertNotIn("gvm", self.contract.downstream_allowed)

    def test_default_props_in_contract(self):
        dp = self.contract.default_properties
        self.assertIn("num_frames_per_batch", dp)
        self.assertIn("dilate_radius", dp)


class GVMSpecContractAlignmentTests(unittest.TestCase):
    """Spec и Contract должны быть выровнены (как test_node_contract_alignment.py)."""

    def test_port_names_align(self):
        from app.node_graph.specs.gvm import SPEC
        from app.node_graph.rules.node_contracts import ALL_NODE_CONTRACTS
        contract = ALL_NODE_CONTRACTS["gvm"]

        spec_in = {p.name for p in SPEC.inputs}
        contract_in = {p.name for p in contract.inputs}
        self.assertEqual(spec_in, contract_in, "GVM input port name mismatch spec vs contract")

        spec_out = {p.name for p in SPEC.outputs}
        contract_out = {p.name for p in contract.outputs}
        self.assertEqual(spec_out, contract_out, "GVM output port name mismatch spec vs contract")

    def test_data_types_align(self):
        from app.node_graph.specs.gvm import SPEC
        from app.node_graph.rules.node_contracts import ALL_NODE_CONTRACTS
        contract = ALL_NODE_CONTRACTS["gvm"]

        spec_in_map = {p.name: p.data_type for p in SPEC.inputs}
        for cp in contract.inputs:
            self.assertEqual(
                spec_in_map[cp.name], cp.data_type,
                f"GVM input {cp.name}: data_type mismatch spec vs contract",
            )

        spec_out_map = {p.name: p.data_type for p in SPEC.outputs}
        for cp in contract.outputs:
            self.assertEqual(
                spec_out_map[cp.name], cp.data_type,
                f"GVM output {cp.name}: data_type mismatch spec vs contract",
            )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Registry topology & port compatibility
# ──────────────────────────────────────────────────────────────────────────────

class GVMRegistryTopologyTests(unittest.TestCase):
    def setUp(self):
        from app.node_graph.rules.registry import get_registry
        self.reg = get_registry()

    # Разрешённые upstream-соединения
    def test_source_to_gvm_allowed(self):
        self.assertTrue(self.reg.can_downstream("source", "gvm"))

    def test_load_to_gvm_allowed(self):
        self.assertTrue(self.reg.can_downstream("load", "gvm"))

    # Разрешённые downstream-соединения
    def test_gvm_to_corridorkey_allowed(self):
        self.assertTrue(self.reg.can_downstream("gvm", "corridorkey"))

    def test_gvm_to_export_allowed(self):
        self.assertTrue(self.reg.can_downstream("gvm", "export"))

    # Запрещённые соединения
    def test_gvm_cannot_feed_matting(self):
        self.assertFalse(self.reg.can_downstream("gvm", "matting"))

    def test_gvm_cannot_feed_sam(self):
        self.assertFalse(self.reg.can_downstream("gvm", "sam2"))

    def test_gvm_cannot_feed_gvm(self):
        self.assertFalse(self.reg.can_downstream("gvm", "gvm"))

    def test_gvm_cannot_feed_birefnet(self):
        self.assertFalse(self.reg.can_downstream("gvm", "birefnet"))

    # Corridorkey upstream должен принимать gvm
    def test_corridorkey_upstream_accepts_gvm(self):
        self.assertTrue(self.reg.can_upstream("gvm", "corridorkey"))

    # Export upstream должен принимать gvm
    def test_export_upstream_accepts_gvm(self):
        self.assertTrue(self.reg.can_upstream("gvm", "export"))


class GVMPortCompatibilityTests(unittest.TestCase):
    def setUp(self):
        from app.node_graph.rules.registry import get_registry
        self.reg = get_registry()

    def test_source_out_to_gvm_image(self):
        self.assertTrue(self.reg.can_connect_ports("source", "out", "gvm", "image"))

    def test_load_out_to_gvm_image(self):
        self.assertTrue(self.reg.can_connect_ports("load", "out", "gvm", "image"))

    def test_gvm_alpha_to_corridorkey_alphahint(self):
        """GVM alpha → CorridorKey alphahint — ключевое соединение."""
        self.assertTrue(self.reg.can_connect_ports("gvm", "alpha", "corridorkey", "alphahint"))

    def test_gvm_alpha_to_export_in(self):
        self.assertTrue(self.reg.can_connect_ports("gvm", "alpha", "export", "in"))

    def test_alpha_stream_rejected_as_gvm_image_input(self):
        """alpha → gvm.image запрещено: нода принимает только RGB image."""
        self.assertFalse(self.reg.can_connect_ports("birefnet", "alpha", "gvm", "image"))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Worker execution
# ──────────────────────────────────────────────────────────────────────────────

class GVMExecuteNodeTests(unittest.TestCase):
    """Тестирует _execute_gvm_node с замоканным GVMService."""

    def _make_mock_gvm_service(self, output_dir_ref: list) -> MagicMock:
        """Создаёт mock GVMService, который пишет синтетические PNG в out_dir."""
        mock_svc = MagicMock()

        def _fake_process_sequence(input_path, output_dir, **kwargs):
            paths = _alpha_pngs(Path(output_dir), 3)
            output_dir_ref.append(Path(output_dir))
            return paths

        mock_svc.process_sequence.side_effect = _fake_process_sequence
        mock_svc.unload.return_value = None
        mock_svc.load_model.return_value = None
        mock_svc.set_callbacks.return_value = None
        return mock_svc

    def test_returns_disk_sequence(self):
        worker = _make_worker()
        out_dir_ref = []
        worker.gvm_service = self._make_mock_gvm_service(out_dir_ref)

        result = worker._execute_gvm_node(
            {"id": "gvm_1", "properties": {}},
            {"image": _rgb_frames(3)},
        )

        self.assertIn("alpha", result)
        alpha = result["alpha"]
        self.assertIsInstance(alpha, dict)
        self.assertTrue(alpha.get("__disk_sequence__"), "alpha must be a disk sequence")

    def test_disk_sequence_has_paths(self):
        worker = _make_worker()
        out_dir_ref = []
        worker.gvm_service = self._make_mock_gvm_service(out_dir_ref)

        result = worker._execute_gvm_node(
            {"id": "gvm_1", "properties": {}},
            {"image": _rgb_frames(3)},
        )

        paths = result["alpha"]["paths"]
        self.assertEqual(len(paths), 3)
        for p in paths:
            self.assertTrue(Path(p).exists(), f"alpha PNG должен существовать: {p}")

    def test_source_node_is_gvm(self):
        worker = _make_worker()
        out_dir_ref = []
        worker.gvm_service = self._make_mock_gvm_service(out_dir_ref)

        result = worker._execute_gvm_node(
            {"id": "gvm_1", "properties": {}},
            {"image": _rgb_frames(3)},
        )

        self.assertEqual(result["alpha"]["source_node"], "gvm")

    def test_properties_forwarded_to_process_sequence(self):
        worker = _make_worker()
        out_dir_ref = []
        mock_svc = self._make_mock_gvm_service(out_dir_ref)
        worker.gvm_service = mock_svc

        worker._execute_gvm_node(
            {
                "id": "gvm_1",
                "properties": {
                    "num_frames_per_batch": 4,
                    "denoise_steps": 2,
                    "decode_chunk_size": 2,
                    "num_overlap_frames": 2,
                    "num_interp_frames": 0,
                },
            },
            {"image": _rgb_frames(3)},
        )

        call_kwargs = mock_svc.process_sequence.call_args
        self.assertEqual(call_kwargs.kwargs.get("num_frames_per_batch"), 4)
        self.assertEqual(call_kwargs.kwargs.get("denoise_steps"), 2)
        self.assertEqual(call_kwargs.kwargs.get("decode_chunk_size"), 2)
        self.assertEqual(call_kwargs.kwargs.get("num_overlap_frames"), 2)
        self.assertEqual(call_kwargs.kwargs.get("num_interp_frames"), 0)

    def test_empty_frames_returns_empty_array(self):
        worker = _make_worker()
        worker.gvm_service = MagicMock()

        result = worker._execute_gvm_node(
            {"id": "gvm_1", "properties": {}},
            {"image": []},
        )

        alpha = result["alpha"]
        self.assertIsInstance(alpha, np.ndarray)
        self.assertEqual(len(alpha), 0)

    def test_missing_image_input_raises(self):
        worker = _make_worker()
        worker.gvm_service = MagicMock()

        with self.assertRaises(ValueError):
            worker._execute_gvm_node({"id": "gvm_1", "properties": {}}, {})

    def test_cancel_flag_stops_execution(self):
        worker = _make_worker()
        worker.cancel_flag.set()
        worker.gvm_service = MagicMock()

        result = worker._execute_gvm_node(
            {"id": "gvm_1", "properties": {}},
            {"image": _rgb_frames(3)},
        )

        self.assertEqual(result, {})
        worker.gvm_service.load_model.assert_not_called()

    def test_dilate_radius_applied(self):
        """С dilate_radius > 0 все PNG должны быть перезаписаны (проверяем вызов write)."""
        worker = _make_worker()
        out_dir_ref = []
        worker.gvm_service = self._make_mock_gvm_service(out_dir_ref)

        result = worker._execute_gvm_node(
            {"id": "gvm_1", "properties": {"dilate_radius": 3}},
            {"image": _rgb_frames(3)},
        )

        # После dilate все файлы должны существовать (не были удалены)
        for p in result["alpha"]["paths"]:
            self.assertTrue(Path(p).exists(), f"PNG должен существовать после dilate: {p}")

    def test_output_dir_uses_temp_dir_not_graph_output_dir(self):
        """GVM scratch не должен писать в пользовательский _graph_output_dir."""
        worker = _make_worker()
        with tempfile.TemporaryDirectory() as tmpdir:
            worker._graph_output_dir = Path(tmpdir)
            out_dir_ref = []
            worker.gvm_service = self._make_mock_gvm_service(out_dir_ref)

            result = worker._execute_gvm_node(
                {"id": "gvm_42", "properties": {}},
                {"image": _rgb_frames(2)},
            )

            for p in result["alpha"]["paths"]:
                self.assertFalse(str(p).startswith(tmpdir), f"Scratch не должен быть внутри _graph_output_dir: {p}")
                self.assertIn("keyflow_gvm_alpha_", str(p))

            worker._cleanup_graph_temp_dirs()
            for p in result["alpha"]["paths"]:
                self.assertFalse(Path(p).exists(), f"Scratch должен удаляться cleanup-ом: {p}")

    def test_emits_gvm_progress_from_service_callback(self):
        worker = _make_worker()
        out_dir_ref = []
        mock_svc = MagicMock()

        progress_signal = MagicMock()
        worker.node_frame_progress = SimpleNamespace(emit=progress_signal)

        def _fake_process_sequence(input_path, output_dir, **kwargs):
            paths = _alpha_pngs(Path(output_dir), 3)
            out_dir_ref.append(Path(output_dir))
            kwargs["progress_callback"](1, 3)
            kwargs["progress_callback"](2, 3)
            kwargs["progress_callback"](3, 3)
            return paths

        mock_svc.process_sequence.side_effect = _fake_process_sequence
        mock_svc.unload.return_value = None
        mock_svc.load_model.return_value = None
        mock_svc.set_callbacks.return_value = None
        worker.gvm_service = mock_svc

        worker._execute_gvm_node(
            {"id": "gvm_1", "properties": {}},
            {"image": _rgb_frames(3)},
        )

        self.assertEqual(
            progress_signal.call_args_list[:3],
            [
                unittest.mock.call("gvm", 1, 3),
                unittest.mock.call("gvm", 2, 3),
                unittest.mock.call("gvm", 3, 3),
            ],
        )


class GVMGraphChainTests(unittest.TestCase):
    """Интеграционный smoke-тест связки source -> gvm -> export(write)."""

    def _make_mock_gvm_service(self) -> MagicMock:
        mock_svc = MagicMock()

        def _fake_process_sequence(_input_path, output_dir, **_kwargs):
            return _alpha_pngs(Path(output_dir), 3)

        mock_svc.process_sequence.side_effect = _fake_process_sequence
        mock_svc.unload.return_value = None
        mock_svc.load_model.return_value = None
        mock_svc.set_callbacks.return_value = None
        return mock_svc

    def test_source_gvm_export_chain_passes_disk_sequence_to_write_sink(self):
        from app.node_graph.models import GraphEdge, GraphNode

        worker = _make_worker()
        worker.gvm_service = self._make_mock_gvm_service()

        # Избегаем побочных эффектов реального stream writer в юнит-тесте.
        worker._stream_graph_write_frame = lambda *_args, **_kwargs: None
        worker._coerce_preview_frame = lambda *_args, **_kwargs: None

        with tempfile.TemporaryDirectory() as tmpdir:
            worker._graph_output_dir = Path(tmpdir)
            nodes = [
                GraphNode(id="src_1", type="source", title="Source"),
                GraphNode(id="gvm_1", type="gvm", title="GVM"),
                GraphNode(id="exp_1", type="export", title="Write"),
            ]
            edges = [
                GraphEdge(src_id="src_1", dst_id="gvm_1", src_port="out", dst_port="image"),
                GraphEdge(src_id="gvm_1", dst_id="exp_1", src_port="alpha", dst_port="in"),
            ]

            outputs = worker._execute_node_graph(nodes, edges, _rgb_frames(3))

        self.assertIn("gvm_1", outputs)
        self.assertIn("exp_1", outputs)

        gvm_alpha = outputs["gvm_1"].get("alpha")
        self.assertIsInstance(gvm_alpha, dict)
        self.assertTrue(gvm_alpha.get("__disk_sequence__"))
        self.assertEqual(gvm_alpha.get("source_node"), "gvm")
        self.assertEqual(gvm_alpha.get("count"), 3)

        write_inputs = outputs["exp_1"]
        self.assertIn("in", write_inputs)
        self.assertEqual(write_inputs["in"], gvm_alpha)

        worker.gvm_service.unload.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────────
# 4. i18n keys
# ──────────────────────────────────────────────────────────────────────────────

class GVMi18nTests(unittest.TestCase):
    """Проверяет, что все i18n-ключи ноды GVM зарегистрированы."""

    def setUp(self):
        from app.i18n import STRINGS
        self.strings = STRINGS

    def test_node_title_key(self):
        self.assertIn("node_graph_node_gvm", self.strings)

    def test_num_frames_per_batch_key(self):
        self.assertIn("gvm_num_frames_per_batch", self.strings)

    def test_decode_chunk_size_key(self):
        self.assertIn("gvm_decode_chunk_size", self.strings)

    def test_num_overlap_frames_key(self):
        self.assertIn("gvm_num_overlap_frames", self.strings)

    def test_num_interp_frames_key(self):
        self.assertIn("gvm_num_interp_frames", self.strings)

    def test_dilate_radius_key(self):
        self.assertIn("gvm_dilate_radius", self.strings)

    def test_download_button_ready_key(self):
        self.assertIn("gvm_download_button_ready", self.strings)

    def test_download_button_missing_key(self):
        self.assertIn("gvm_download_button_missing", self.strings)

    def test_worker_processing_start_key(self):
        self.assertIn("worker_gvm_processing_start", self.strings)

    def test_worker_saving_frames_key(self):
        self.assertIn("worker_gvm_saving_frames", self.strings)

    def test_gvm_done_key(self):
        self.assertIn("gvm_done", self.strings)

    def test_keys_have_both_languages(self):
        keys = [
            "node_graph_node_gvm",
            "gvm_num_frames_per_batch",
            "gvm_dilate_radius",
            "worker_gvm_processing_start",
        ]
        for key in keys:
            with self.subTest(key=key):
                val = self.strings[key]
                self.assertIn("ru", val, f"{key} missing 'ru'")
                self.assertIn("en", val, f"{key} missing 'en'")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Orchestrator routing: GVM graph triggers graph inference path
# ──────────────────────────────────────────────────────────────────────────────

class GVMOrchestratorRouteTests(unittest.TestCase):
    """Проверяет, что граф с GVM попадает на правильный путь в оркестраторе."""

    def _trigger_set(self):
        """Набор типов нод, который должен активировать graph inference path."""
        import re
        import ast
        src = Path(__file__).parent.parent / "app" / "coordinators" / "matting_orchestrator.py"
        text = src.read_text()
        m = re.search(r'not any\(nt in (\{[^}]+\}) for nt in node_types\)', text)
        if m is None:
            self.fail("Не удалось найти фильтр node_types в try_graph_inference_run")
        return ast.literal_eval(m.group(1))

    def test_gvm_in_trigger_set(self):
        """gvm должен быть в наборе типов, активирующих graph inference."""
        trigger = self._trigger_set()
        self.assertIn("gvm", trigger,
                      "'gvm' missing from try_graph_inference_run trigger set — "
                      "Read→GVM→Write chain will silently skip graph inference!")

    def test_birefnet_still_in_trigger_set(self):
        self.assertIn("birefnet", self._trigger_set())

    def test_corridorkey_still_in_trigger_set(self):
        self.assertIn("corridorkey", self._trigger_set())


if __name__ == "__main__":
    unittest.main()
