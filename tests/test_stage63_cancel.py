"""Stage 6.3 — Multi-node cancel integration tests.

Проверяет:
- Сброс и установку cancel_flag в InferenceWorker
- Остановку _execute_node_graph при установленном флаге
- Поведение normalize_cancel_policy для всех трёх политик
- Семантику результатов при отмене (make_runtime_result_cancelled / _cancelled_partial)
- Корректность is_runtime_cancelled / runtime_saved_paths
"""
import os
import threading
import time
import unittest

os.environ.setdefault("KEYFLOW_DEVICE", "cpu")

import numpy as np

from app.node_graph.models import GraphEdge, GraphNode
from app.runtime_contract import (
    RUNTIME_CANCEL_CLEANUP_PARTIAL,
    RUNTIME_CANCEL_IMMEDIATE,
    RUNTIME_CANCEL_SAVE_PARTIAL,
    is_runtime_cancelled,
    make_runtime_result_cancelled,
    make_runtime_result_cancelled_partial,
    make_runtime_result_ok,
    normalize_cancel_policy,
    runtime_saved_paths,
)
from app.workers.inference_worker import InferenceWorker


def _make_worker() -> InferenceWorker:
    """Создать InferenceWorker без QApplication (без GUI)."""
    return InferenceWorker.__new__(InferenceWorker)


def _init_worker(worker: InferenceWorker) -> None:
    """Минимальная инициализация Worker без QObject.__init__."""
    worker.cancel_flag = threading.Event()
    worker.language_code = "ru"


class CancelFlagTests(unittest.TestCase):
    """Базовые операции с cancel_flag."""

    def setUp(self):
        self.worker = _make_worker()
        _init_worker(self.worker)

    def test_initial_state_is_clear(self):
        self.assertFalse(self.worker.cancel_flag.is_set())

    def test_set_cancel_sets_flag(self):
        self.worker.set_cancel()
        self.assertTrue(self.worker.cancel_flag.is_set())

    def test_reset_cancel_clears_flag(self):
        self.worker.set_cancel()
        self.worker.reset_cancel()
        self.assertFalse(self.worker.cancel_flag.is_set())

    def test_set_cancel_is_idempotent(self):
        self.worker.set_cancel()
        self.worker.set_cancel()
        self.assertTrue(self.worker.cancel_flag.is_set())

    def test_reset_after_double_set(self):
        self.worker.set_cancel()
        self.worker.set_cancel()
        self.worker.reset_cancel()
        self.assertFalse(self.worker.cancel_flag.is_set())

    def test_thread_safety_set_then_read(self):
        """Флаг виден из другого потока."""
        results = []

        def setter():
            time.sleep(0.01)
            self.worker.set_cancel()

        t = threading.Thread(target=setter, daemon=True)
        t.start()
        t.join(timeout=1.0)
        results.append(self.worker.cancel_flag.is_set())
        self.assertTrue(results[0])


class NormalizeCancelPolicyTests(unittest.TestCase):
    """Преобразование строковых политик отмены."""

    def test_immediate_string(self):
        self.assertEqual(normalize_cancel_policy("immediate"), RUNTIME_CANCEL_IMMEDIATE)

    def test_cleanup_partial_string(self):
        self.assertEqual(normalize_cancel_policy("cleanup_partial"), RUNTIME_CANCEL_CLEANUP_PARTIAL)

    def test_save_partial_string(self):
        self.assertEqual(normalize_cancel_policy("save_partial"), RUNTIME_CANCEL_SAVE_PARTIAL)

    def test_none_defaults_to_save_partial(self):
        self.assertEqual(normalize_cancel_policy(None), RUNTIME_CANCEL_SAVE_PARTIAL)

    def test_empty_string_defaults_to_save_partial(self):
        self.assertEqual(normalize_cancel_policy(""), RUNTIME_CANCEL_SAVE_PARTIAL)

    def test_unknown_string_defaults_to_save_partial(self):
        self.assertEqual(normalize_cancel_policy("unknown_policy"), RUNTIME_CANCEL_SAVE_PARTIAL)

    def test_whitespace_normalised(self):
        self.assertEqual(normalize_cancel_policy("  IMMEDIATE  "), RUNTIME_CANCEL_IMMEDIATE)


class CancelledResultTests(unittest.TestCase):
    """Семантика результата отмены."""

    def test_cancelled_result_is_cancelled(self):
        result = make_runtime_result_cancelled()
        self.assertTrue(is_runtime_cancelled(result))

    def test_ok_result_is_not_cancelled(self):
        result = make_runtime_result_ok({"node1": "/tmp/out.png"}, n_frames=5)
        self.assertFalse(is_runtime_cancelled(result))

    def test_cancelled_partial_is_cancelled(self):
        result = make_runtime_result_cancelled_partial({"node1": "/tmp/out.png"}, n_frames=3)
        self.assertTrue(is_runtime_cancelled(result))

    def test_cancelled_partial_has_partial_flag(self):
        result = make_runtime_result_cancelled_partial({"n": "/x"}, n_frames=2)
        self.assertTrue(result.get("partial_result"))

    def test_cancelled_no_partial_flag(self):
        result = make_runtime_result_cancelled()
        self.assertFalse(result.get("partial_result"))

    def test_cancelled_partial_saved_paths_present(self):
        saved = {"write_1": "/tmp/a.png", "write_2": "/tmp/b.mp4"}
        result = make_runtime_result_cancelled_partial(saved, n_frames=10)
        self.assertEqual(result["partial_saved_paths"], saved)

    def test_runtime_saved_paths_ok(self):
        paths = {"n1": "/a", "n2": "/b"}
        result = make_runtime_result_ok(paths, n_frames=5)
        self.assertEqual(runtime_saved_paths(result), paths)

    def test_runtime_saved_paths_cancelled_empty(self):
        result = make_runtime_result_cancelled()
        self.assertEqual(runtime_saved_paths(result), {})

    def test_ok_result_status(self):
        result = make_runtime_result_ok({}, n_frames=1)
        self.assertEqual(result["status"], "ok")

    def test_cancelled_result_status(self):
        result = make_runtime_result_cancelled()
        self.assertEqual(result["status"], "cancelled")


class ExecuteNodeGraphCancelTests(unittest.TestCase):
    """Проверка _execute_node_graph при установленном cancel_flag.

    Используем только пассажирские (passthrough_source / write_sink) ноды,
    чтобы не требовать реальных model-checkpoint-ов.
    """

    def _make_full_worker(self) -> InferenceWorker:
        """Создать Worker с полной инициализацией (без Qt event loop)."""
        worker = InferenceWorker.__new__(InferenceWorker)
        # Минимальный init без QObject
        worker.cancel_flag = threading.Event()
        worker.language_code = "ru"
        # Пустые внутренние состояния, используемые в методах
        worker._graph_output_dir = None
        worker._graph_source_path = ""
        worker._graph_mask_path = ""
        worker._graph_fps = 25.0
        worker._graph_audio_path = ""
        worker._graph_write_plans = {}
        worker._graph_stream_saved_paths = {}
        worker._graph_correction_masks = None
        worker._graph_start_frame = 0
        worker._graph_end_frame = -1
        return worker

    @staticmethod
    def _source_export_graph():
        nodes = [
            GraphNode(id="src_1", type="source", title="Source"),
            GraphNode(id="exp_1", type="export", title="Export"),
        ]
        edges = [
            GraphEdge(src_id="src_1", dst_id="exp_1", src_port="out", dst_port="in"),
        ]
        return nodes, edges

    @staticmethod
    def _load_export_graph():
        """source → load → export — три ноды, чтобы проверить остановку на второй."""
        nodes = [
            GraphNode(id="src_1", type="source", title="Source"),
            GraphNode(id="load_1", type="load", title="Load", properties={"path": ""}),
            GraphNode(id="exp_1", type="export", title="Export"),
        ]
        edges = [
            GraphEdge(src_id="src_1", dst_id="load_1", src_port="out", dst_port="image"),
            GraphEdge(src_id="load_1", dst_id="exp_1", src_port="out", dst_port="in"),
        ]
        return nodes, edges

    def _make_frames(self, n: int = 3) -> list[np.ndarray]:
        return [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(n)]

    def test_cancel_before_execution_stops_all_nodes(self):
        """Если флаг установлен ДО вызова, ни одна нода не должна выполниться."""
        worker = self._make_full_worker()
        worker.set_cancel()

        # Захватываем log_message-эмиты через monkey-patch
        executed_nodes = []

        def fake_log(msg):
            executed_nodes.append(msg)

        worker.log_message = type("S", (), {"emit": staticmethod(lambda m: executed_nodes.append(m))})()

        nodes, edges = self._source_export_graph()
        frames = self._make_frames()

        outputs = worker._execute_node_graph(nodes, edges, frames)

        # cancel до старта → outputs пуст (ни одна нода не выполнилась)
        self.assertEqual(outputs, {})

    def test_cancel_flag_cleared_after_reset(self):
        """После reset_cancel граф должен выполниться хотя бы до passthrough ноды."""
        worker = self._make_full_worker()
        worker.set_cancel()
        worker.reset_cancel()

        logged = []
        worker.log_message = type("S", (), {"emit": staticmethod(lambda m: logged.append(m))})()

        nodes, edges = self._source_export_graph()
        frames = self._make_frames()

        # Должен выполниться без ошибок
        outputs = worker._execute_node_graph(nodes, edges, frames)

        # source нода — passthrough; export — write_sink
        self.assertIn("src_1", outputs)

    def test_output_dict_empty_when_cancelled_before_any_node(self):
        """Если cancel установлен до итерации, outputs == {}."""
        worker = self._make_full_worker()
        worker.cancel_flag.set()
        worker.log_message = type("S", (), {"emit": staticmethod(lambda m: None)})()

        nodes, edges = self._source_export_graph()
        frames = self._make_frames()

        outputs = worker._execute_node_graph(nodes, edges, frames)
        self.assertIsInstance(outputs, dict)
        self.assertEqual(len(outputs), 0)

    def test_gather_node_inputs_routes_data(self):
        """_gather_node_inputs правильно маршрутизирует данные между нодами."""
        worker = self._make_full_worker()
        worker.log_message = type("S", (), {"emit": staticmethod(lambda m: None)})()

        frames = self._make_frames(2)
        nodes = {
            "src_1": GraphNode(id="src_1", type="source", title="S"),
            "exp_1": GraphNode(id="exp_1", type="export", title="E"),
        }
        edges = [GraphEdge(src_id="src_1", dst_id="exp_1", src_port="out", dst_port="in")]
        outputs = {"src_1": {"out": frames, "image": frames}}

        inputs = worker._gather_node_inputs(nodes, edges, "exp_1", outputs, frames)

        self.assertIn("in", inputs)
        self.assertIs(inputs["in"], frames)

    def test_gather_node_inputs_missing_source_not_in_outputs(self):
        """Если источниковая нода не в outputs, dst не получает данных."""
        worker = self._make_full_worker()
        logged = []
        worker.log_message = type("S", (), {"emit": staticmethod(lambda m: logged.append(m))})()

        frames = self._make_frames(1)
        nodes = {"src_1": GraphNode(id="src_1", type="source", title="S")}
        edges = [GraphEdge(src_id="src_1", dst_id="exp_1", src_port="out", dst_port="in")]
        outputs = {}  # src_1 не был выполнен

        inputs = worker._gather_node_inputs(nodes, edges, "exp_1", outputs, frames)
        self.assertNotIn("in", inputs)
        # должен быть log-warning о неготовом источнике
        self.assertTrue(any("src_1" in msg for msg in logged))


class MultiNodeGraphExecutionTests(unittest.TestCase):
    """Тесты планирования графа с несколькими нодами."""

    def test_topological_order_source_before_export(self):
        from app.node_graph.engine import NodeGraphEngine

        engine = NodeGraphEngine()
        nodes = [
            GraphNode(id="src_1", type="source", title="S"),
            GraphNode(id="exp_1", type="export", title="E"),
        ]
        edges = [GraphEdge(src_id="src_1", dst_id="exp_1", src_port="out", dst_port="in")]

        order = engine.topological_order(nodes, edges)
        self.assertLess(order.index("src_1"), order.index("exp_1"))

    def test_three_node_chain_order(self):
        from app.node_graph.engine import NodeGraphEngine

        engine = NodeGraphEngine()
        nodes = [
            GraphNode(id="load_1", type="load", title="Load"),
            GraphNode(id="exp_1", type="export", title="E"),
            GraphNode(id="src_1", type="source", title="S"),
        ]
        edges = [
            GraphEdge(src_id="src_1", dst_id="load_1", src_port="out", dst_port="image"),
            GraphEdge(src_id="load_1", dst_id="exp_1", src_port="out", dst_port="in"),
        ]
        order = engine.topological_order(nodes, edges)
        self.assertLess(order.index("src_1"), order.index("load_1"))
        self.assertLess(order.index("load_1"), order.index("exp_1"))

    def test_disabled_node_skipped_in_plan(self):
        from app.node_graph.engine import NodeGraphEngine

        engine = NodeGraphEngine()
        nodes = [
            GraphNode(id="src_1", type="source", title="S"),
            GraphNode(id="exp_1", type="export", title="E", enabled=False),
        ]
        edges = [GraphEdge(src_id="src_1", dst_id="exp_1", src_port="out", dst_port="in")]

        plan, _ = engine.build_execution_plan_with_diagnostics(nodes, edges)
        if plan is not None:
            # Выключенная нода должна быть помечена skip_disabled или отсутствовать
            action = plan.node_actions.get("exp_1", "skip_disabled")
            self.assertIn(action, {"skip_disabled", "deferred"})
