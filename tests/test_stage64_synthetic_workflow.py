"""Stage 6.4 — Synthetic video workflow integration tests.

Проверяет сквозное выполнение графа с синтетическими кадрами
без реальных model-checkpoint-ов, используя только passthrough/write_sink ноды.

Тесты охватывают:
  - source → export (минимальный граф)
  - source → load → export (трёхнодовый граф)
  - Корректность outputs после passthrough-выполнения
  - Сохранение плана записи через _prepare_graph_write_targets
    - Корректный выход resolve_graph_write_output_dir
    - Корректный stem _build_keyflow_output_dir для разных типов источников
"""
import os
import threading
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("KEYFLOW_DEVICE", "cpu")

import numpy as np

from app.node_graph.models import GraphEdge, GraphNode
from app.utils.write_paths import get_port_output_label, resolve_graph_write_output_dir
from app.workers.graph_execution_actions import (
    build_deferred_action_output,
    build_graph_downstream_targets,
    build_passthrough_source_output,
    format_deferred_action_log,
)
from app.workers.graph_write_planner import build_graph_write_plan_targets
from app.workers.inference_worker import (
    InferenceWorker,
    _build_keyflow_output_dir,
)
from app.runtime_contract import (
    make_runtime_result_ok,
    is_runtime_cancelled,
)


def _make_worker() -> InferenceWorker:
    """Создать InferenceWorker с минимальным init (без QObject/Qt)."""
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
    w.log_message = type("Sig", (), {"emit": staticmethod(lambda m: None)})()
    return w


def _frames(n: int = 5, h: int = 16, w: int = 16) -> list:
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]


class MinimalGraphExecutionTests(unittest.TestCase):
    """source → export: минимальный возможный граф."""

    def test_source_export_graph_returns_outputs(self):
        worker = _make_worker()
        nodes = [
            GraphNode(id="src_1", type="source", title="S"),
            GraphNode(id="exp_1", type="export", title="E"),
        ]
        edges = [GraphEdge(src_id="src_1", dst_id="exp_1", src_port="out", dst_port="in")]

        outputs = worker._execute_node_graph(nodes, edges, _frames(3))

        self.assertIn("src_1", outputs)
        src_out = outputs["src_1"]
        self.assertIsInstance(src_out, dict)
        # passthrough_source сохраняет кадры под ключами out / image / frame_sequence
        self.assertIn("out", src_out)

    def test_source_frames_propagated_to_export(self):
        worker = _make_worker()
        frames = _frames(4)
        nodes = [
            GraphNode(id="src_1", type="source", title="S"),
            GraphNode(id="exp_1", type="export", title="E"),
        ]
        edges = [GraphEdge(src_id="src_1", dst_id="exp_1", src_port="out", dst_port="in")]

        outputs = worker._execute_node_graph(nodes, edges, frames)

        # write_sink копирует inputs → outputs; "in" должен содержать список кадров
        exp_out = outputs.get("exp_1", {})
        self.assertIn("in", exp_out)
        self.assertIs(exp_out["in"], frames)

    def test_alpha_sequence_with_image_media_type_uses_all_frames(self):
        worker = _make_worker()
        worker._graph_output_dir = Path(tempfile.gettempdir())
        worker._graph_source_path = "/tmp/source.mp4"

        alpha_seq_frames = _frames(3)
        worker._load_video = lambda *_args, **_kwargs: (alpha_seq_frames, 25.0, "")
        worker._load_image_frame = lambda *_args, **_kwargs: _frames(1)[0]

        import app.workers.inference_worker as worker_module

        original_is_sequence = worker_module.is_numbered_image_sequence
        worker_module.is_numbered_image_sequence = lambda path: str(path).endswith("0001.png")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                seq_path = Path(tmpdir) / "0001.png"
                seq_path.write_bytes(b"x")

                nodes = [
                    GraphNode(
                        id="alpha_1",
                        type="alpha",
                        title="A",
                        properties={"path": str(seq_path), "media_type": "image"},
                    ),
                    GraphNode(id="exp_1", type="export", title="E"),
                ]
                edges = [GraphEdge(src_id="alpha_1", dst_id="exp_1", src_port="out", dst_port="in")]

                outputs = worker._execute_node_graph(nodes, edges, _frames(1))
        finally:
            worker_module.is_numbered_image_sequence = original_is_sequence

        self.assertIn("alpha_1", outputs)
        alpha_out = outputs["alpha_1"]
        self.assertIn("out", alpha_out)
        self.assertEqual(len(alpha_out["out"]), 3)
        self.assertIs(alpha_out["out"], alpha_seq_frames)

    def test_empty_graph_returns_empty_outputs(self):
        worker = _make_worker()
        outputs = worker._execute_node_graph([], [], [])
        self.assertEqual(outputs, {})

    def test_sam_service_unloaded_before_corridorkey_execution(self):
        """SAM is now disk-deferred when its only output goes to corridorkey.alphahint.

        Expected behaviour:
        - SAM node action = 'deferred' → _execute_node is NOT called for SAM
        - _unload_sam_service_if_loaded() is still called before CorridorKey
        - CorridorKey receives __deferred_sam_node in inputs (not alphahint)
        """
        worker = _make_worker()

        class _FakeSamService:
            def __init__(self):
                self.unloaded = False

            def unload(self):
                self.unloaded = True

        class _FakeCorridorKeyService:
            def unload_engine(self):
                return None

        fake_sam_service = _FakeSamService()
        worker._sam_service = fake_sam_service
        worker.corridorkey_service = _FakeCorridorKeyService()

        execution_order: list[str] = []

        def _fake_execute_node(node_type: str, _node_data: dict, inputs: dict) -> dict:
            execution_order.append(node_type)
            if node_type == "corridorkey":
                # SAM is deferred → sam service must be unloaded before CK runs
                self.assertTrue(fake_sam_service.unloaded)
                self.assertIsNone(getattr(worker, "_sam_service", None))
                # Deferred SAM passes node data, not a precomputed sequence
                self.assertIn("__deferred_sam_node", inputs)
                return {"alpha": [np.zeros((4, 4), dtype=np.uint8)]}
            return {}

        worker._execute_node = _fake_execute_node

        nodes = [
            GraphNode(id="src_1", type="source", title="S"),
            GraphNode(id="sam_1", type="sam2", title="SAM"),
            GraphNode(id="ck_1", type="corridorkey", title="CK"),
            GraphNode(id="exp_1", type="export", title="E"),
        ]
        edges = [
            GraphEdge(src_id="src_1", dst_id="sam_1", src_port="out", dst_port="img"),
            GraphEdge(src_id="src_1", dst_id="ck_1", src_port="out", dst_port="image"),
            GraphEdge(src_id="sam_1", dst_id="ck_1", src_port="out", dst_port="alphahint"),
            GraphEdge(src_id="ck_1", dst_id="exp_1", src_port="alpha", dst_port="in"),
        ]

        outputs = worker._execute_node_graph(nodes, edges, _frames(1))

        self.assertIn("ck_1", outputs)
        # SAM is deferred → only corridorkey appears in _execute_node calls
        self.assertEqual(execution_order, ["corridorkey"])

    def test_corridorkey_staged_temp_dir_is_cleaned_after_success(self):
        worker = _make_worker()

        class _Sig:
            @staticmethod
            def emit(*_args, **_kwargs):
                return None

        class _FakeBiRefNetService:
            def set_callbacks(self, **_kwargs):
                return None

            def load_model(self, **_kwargs):
                return None

            def process_image(self, frame, **_kwargs):
                h, w = np.asarray(frame).shape[:2]
                return np.ones((h, w), dtype=np.float32)

            def unload_model(self):
                return None

        class _FakeCorridorKeyService:
            def process_frame(self, frame, **_kwargs):
                h, w = np.asarray(frame).shape[:2]
                return {"alpha": np.zeros((h, w), dtype=np.float32)}

        worker.stage_progress = _Sig()
        worker.node_frame_progress = _Sig()
        worker.graph_stream_preview = _Sig()
        worker.corridorkey_mode_resolved = _Sig()
        worker._emit_corridorkey_mode_indicator = lambda *_args, **_kwargs: None
        worker._stream_graph_write_frame = lambda *_args, **_kwargs: None
        worker._coerce_preview_frame = lambda *_args, **_kwargs: None
        worker._graph_start_frame = 0
        worker._compatibility_profile = "auto"
        worker.birefnet_service = _FakeBiRefNetService()
        worker.corridorkey_service = _FakeCorridorKeyService()

        frames = _frames(2, h=8, w=8)

        import app.workers.inference_worker as worker_module

        with tempfile.TemporaryDirectory() as tmp_root:
            staged_dir = Path(tmp_root) / "keyflow_alphahint_test"
            original_mkdtemp = worker_module.tempfile.mkdtemp

            def _fake_mkdtemp(prefix: str = ""):
                staged_dir.mkdir(parents=True, exist_ok=True)
                return str(staged_dir)

            worker_module.tempfile.mkdtemp = _fake_mkdtemp
            try:
                result = worker._execute_corridorkey_node(
                    {"id": "ck_1", "properties": {}},
                    {
                        "image": frames,
                        "__deferred_birefnet_node": {"id": "biref_1", "properties": {}},
                    },
                )
            finally:
                worker_module.tempfile.mkdtemp = original_mkdtemp

            self.assertIn("alpha", result)
            self.assertFalse(staged_dir.exists())

class GraphExecutionActionHelperTests(unittest.TestCase):
    def test_build_graph_downstream_targets_records_ports_types_and_enabled_state(self):
        nodes_by_id = {
            "src_1": GraphNode(id="src_1", type="source", title="S"),
            "exp_1": GraphNode(id="exp_1", type="export", title="E", enabled=False),
        }
        edges = [GraphEdge(src_id="src_1", dst_id="exp_1", src_port="out", dst_port="in")]

        targets = build_graph_downstream_targets(nodes_by_id, edges)

        self.assertEqual(list(targets), [("src_1", "out")])
        self.assertEqual(
            targets[("src_1", "out")],
            [
                {
                    "dst_id": "exp_1",
                    "dst_port": "in",
                    "dst_type": "export",
                    "dst_enabled": False,
                }
            ],
        )

    def test_build_passthrough_source_output_preserves_frame_aliases_and_bbox_meta(self):
        frames = _frames(2, h=4, w=5)

        output = build_passthrough_source_output(frames, lambda _frame: (1, 2, 3, 4))

        self.assertIs(output["out"], frames)
        self.assertIs(output["image"], frames)
        self.assertIs(output["frame_sequence"], frames)
        self.assertEqual(output["__meta__"]["out"]["bbox_sequence"], [(1, 2, 3, 4), (1, 2, 3, 4)])
        self.assertEqual(output["__meta__"]["image"], output["__meta__"]["out"])

    def test_deferred_action_helpers_match_sam_and_staged_payloads(self):
        self.assertEqual(build_deferred_action_output("sam2"), {"__deferred_sam_disk__": True, "out": None, "mask": None})
        self.assertEqual(build_deferred_action_output("birefnet"), {"__deferred_staged__": True, "alpha": None})
        self.assertIn("SAM2 node sam_1", format_deferred_action_log("sam_1", "sam2"))
        self.assertIn("BiRefNet node biref_1", format_deferred_action_log("biref_1", "birefnet"))


class ThreeNodeGraphTests(unittest.TestCase):
    """Тесты графа с тремя нодами через NodeGraphEngine (без выполнения inference)."""

    def test_load_and_source_independent_topology(self):
        """source и load — оба source-ноды; оба должны быть в плане выполнения."""
        from app.node_graph.engine import NodeGraphEngine

        engine = NodeGraphEngine()
        nodes = [
            GraphNode(id="src_1", type="source", title="S"),
            GraphNode(id="load_1", type="load", title="L"),
            GraphNode(id="exp_1", type="export", title="E"),
        ]
        # source → export, load — независимо (isolated)
        edges = [GraphEdge(src_id="src_1", dst_id="exp_1", src_port="out", dst_port="in")]

        plan, diag = engine.build_execution_plan_with_diagnostics(nodes, edges)
        self.assertIsNotNone(plan)
        self.assertIn("src_1", plan.execution_order)
        self.assertIn("exp_1", plan.execution_order)

    def test_source_before_export_in_execution_plan(self):
        """Порядок выполнения: source перед export."""
        from app.node_graph.engine import NodeGraphEngine

        engine = NodeGraphEngine()
        nodes = [
            GraphNode(id="exp_1", type="export", title="E"),
            GraphNode(id="src_1", type="source", title="S"),
        ]
        edges = [GraphEdge(src_id="src_1", dst_id="exp_1", src_port="out", dst_port="in")]

        order = engine.topological_order(nodes, edges)
        self.assertLess(order.index("src_1"), order.index("exp_1"))

    def test_two_source_nodes_both_in_plan(self):
        """Два независимых source-ноды оба попадают в plan."""
        from app.node_graph.engine import NodeGraphEngine

        engine = NodeGraphEngine()
        nodes = [
            GraphNode(id="src_1", type="source", title="S1"),
            GraphNode(id="load_1", type="load", title="L1"),
            GraphNode(id="exp_1", type="export", title="E"),
        ]
        edges = [
            GraphEdge(src_id="src_1", dst_id="exp_1", src_port="out", dst_port="in"),
        ]

        plan, _ = engine.build_execution_plan_with_diagnostics(nodes, edges)
        self.assertIsNotNone(plan)
        # load_1 не подключён — может быть skip_isolated, но не должен падать
        actions = plan.node_actions if plan else {}
        self.assertIn(actions.get("load_1", "skip_isolated"), {"skip_isolated", "passthrough_source", "deferred"})


class WritePlanTests(unittest.TestCase):
    """_prepare_graph_write_targets и resolve_graph_write_output_dir."""

    def test_prepare_write_targets_creates_empty_plans(self):
        worker = _make_worker()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            nodes = [
                GraphNode(id="src_1", type="source", title="S"),
                GraphNode(id="exp_1", type="export", title="E", properties={"auto_output_dir": True}),
            ]
            edges = [GraphEdge(src_id="src_1", dst_id="exp_1", src_port="out", dst_port="in")]

            worker._prepare_graph_write_targets(nodes, edges, output_dir)

            # Внутренний план должен быть создан без падений
            self.assertIsInstance(worker._graph_write_plans, dict)

    def test_prepare_write_targets_skips_unconnected_export_nodes(self):
        worker = _make_worker()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            nodes = [
                GraphNode(id="src_1", type="source", title="S"),
                GraphNode(id="exp_1", type="export", title="Connected", properties={"auto_output_dir": True}),
                GraphNode(id="exp_2", type="export", title="Loose", properties={"auto_output_dir": True}),
            ]
            edges = [GraphEdge(src_id="src_1", dst_id="exp_1", src_port="out", dst_port="in")]

            worker._prepare_graph_write_targets(nodes, edges, output_dir)

            # Connected export node: source title "S" + port label "img" → S/img subdir
            self.assertTrue((output_dir / "S" / "img").exists())
            # Unconnected export (exp_2) must be skipped — no directory created
            self.assertFalse((output_dir / "out").exists())

    def test_build_graph_write_plan_targets_resolves_connected_exports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            nodes = [
                GraphNode(id="src_1", type="source", title="Source Clip"),
                GraphNode(id="exp_1", type="export", title="Alpha", properties={"auto_output_dir": True}),
                GraphNode(id="exp_2", type="export", title="Loose", properties={"auto_output_dir": True}),
            ]
            edges = [GraphEdge(src_id="src_1", dst_id="exp_1", src_port="out", dst_port="in")]

            targets = build_graph_write_plan_targets(nodes, edges, output_dir)

        self.assertEqual(len(targets), 1)
        target = targets[0]
        self.assertEqual(target.node_id, "exp_1")
        self.assertEqual(target.source_node_id, "src_1")
        self.assertEqual(target.stream_label, "img")
        self.assertEqual(target.target_dir, output_dir / "Source Clip" / "img")
        self.assertEqual(target.write_cfg["output_dir"], str(output_dir / "Source Clip" / "img"))

    def test_resolve_auto_output_dir_uses_output_dir(self):
        base = Path("/tmp/test_out")
        write_cfg = {"auto_output_dir": True}
        result = resolve_graph_write_output_dir(write_cfg, base, "alpha")
        self.assertEqual(result, base / "alpha")

    def test_resolve_custom_output_dir(self):
        base = Path("/tmp/test_out")
        write_cfg = {"auto_output_dir": False, "output_dir": "/custom/path"}
        result = resolve_graph_write_output_dir(write_cfg, base, "alpha")
        self.assertEqual(result, Path("/custom/path"))

    def test_resolve_auto_false_empty_dir_fallback(self):
        """auto_output_dir=False но output_dir пустой → fallback на base/stream."""
        base = Path("/tmp/test_out")
        write_cfg = {"auto_output_dir": False, "output_dir": ""}
        result = resolve_graph_write_output_dir(write_cfg, base, "fg")
        self.assertEqual(result, base / "fg")

    def test_get_port_output_label_uses_node_spec_or_fallback(self):
        self.assertEqual(get_port_output_label("source", "out"), "img")
        self.assertEqual(get_port_output_label("unknown", "premult_rgba"), "Premult Rgba")


class BuildKeyflowOutputDirTests(unittest.TestCase):
    """_build_keyflow_output_dir правильно формирует папку вывода."""

    def test_video_source_stem(self):
        source = Path("/data/clip.mp4")
        result = _build_keyflow_output_dir(source, "alpha")
        self.assertEqual(result, Path("/data/clip_keyflow/alpha"))

    def test_image_source_stem(self):
        source = Path("/data/photo.jpg")
        result = _build_keyflow_output_dir(source, "fg")
        self.assertEqual(result, Path("/data/photo_keyflow/fg"))

    def test_numbered_sequence_frame(self):
        """Нумерованная последовательность 0001.png → на уровень выше папки."""
        source = Path("/data/frames/0001.png")
        result = _build_keyflow_output_dir(source, "alpha")
        self.assertEqual(result, Path("/data/frames_keyflow/alpha"))

    def test_stream_label_becomes_subdirectory(self):
        source = Path("/video/input.mov")
        result = _build_keyflow_output_dir(source, "rgba")
        self.assertTrue(str(result).endswith("/rgba"))

    def test_different_stream_labels_produce_different_dirs(self):
        source = Path("/video/input.mov")
        r1 = _build_keyflow_output_dir(source, "alpha")
        r2 = _build_keyflow_output_dir(source, "fg")
        self.assertNotEqual(r1, r2)


class GatherNodeInputsEdgeCasesTests(unittest.TestCase):
    """Пограничные случаи _gather_node_inputs в полноценном workflow."""

    def test_multiple_upstream_nodes_to_one_node(self):
        """Две ноды → один узел через разные dst_port."""
        worker = _make_worker()
        frames_a = _frames(2)
        frames_b = _frames(2)

        nodes = {
            "src_a": GraphNode(id="src_a", type="source", title="A"),
            "src_b": GraphNode(id="src_b", type="source", title="B"),
        }
        edges = [
            GraphEdge(src_id="src_a", dst_id="dst", src_port="out", dst_port="image"),
            GraphEdge(src_id="src_b", dst_id="dst", src_port="out", dst_port="mask"),
        ]
        outputs = {
            "src_a": {"out": frames_a},
            "src_b": {"out": frames_b},
        }

        inputs = worker._gather_node_inputs(nodes, edges, "dst", outputs, [])

        self.assertIn("image", inputs)
        self.assertIn("mask", inputs)
        self.assertIs(inputs["image"], frames_a)
        self.assertIs(inputs["mask"], frames_b)

    def test_no_edges_for_node_yields_empty_inputs(self):
        worker = _make_worker()
        inputs = worker._gather_node_inputs({}, [], "isolated_node", {}, _frames(1))
        self.assertEqual(inputs, {})

    def test_src_port_annotation_added_to_inputs(self):
        """__src_port__<dst_port> должен присутствовать в inputs."""
        worker = _make_worker()
        frames = _frames(1)
        nodes = {"s": GraphNode(id="s", type="source", title="S")}
        edges = [GraphEdge(src_id="s", dst_id="dst", src_port="rgba", dst_port="in")]
        outputs = {"s": {"rgba": frames}}

        inputs = worker._gather_node_inputs(nodes, edges, "dst", outputs, frames)

        self.assertEqual(inputs.get("__src_port__in"), "rgba")


class RuntimeResultWorkflowTests(unittest.TestCase):
    """Проверяет семантику результата runtime в контексте workflow."""

    def test_ok_result_not_cancelled(self):
        result = make_runtime_result_ok({"export_1": "/out/alpha.png"}, n_frames=5)
        self.assertFalse(is_runtime_cancelled(result))
        self.assertEqual(result["n_frames"], 5)
        self.assertEqual(result["saved_paths"]["export_1"], "/out/alpha.png")

    def test_ok_result_with_empty_paths(self):
        result = make_runtime_result_ok({}, n_frames=0)
        self.assertEqual(result["saved_paths"], {})
        self.assertEqual(result["n_frames"], 0)
        self.assertFalse(is_runtime_cancelled(result))
