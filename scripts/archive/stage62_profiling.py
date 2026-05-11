"""Stage 6.2 — Performance profiling script.

Запуск:
    KEYFLOW_DEVICE=cpu python scripts/archive/stage62_profiling.py [--frames N] [--verbose]

Измеряет:
  • Время создания/валидации/топосортировки графа
  • Время инициализации InferenceWorker
  • Время инициализации NodeGraphEngine
  • Throughput при синтетической обработке passthrough-нод (кадры/с)
  • Использование памяти процессами (если psutil доступен)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KEYFLOW_DEVICE", "cpu")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# ── опциональный psutil ──────────────────────────────────────────────────────
try:
    import psutil as _psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ────────────────────────────────────────────────────────────────────────────
def _rss_mb() -> float:
    """RSS процесса в МБ (0 если psutil недоступен)."""
    if not _HAS_PSUTIL:
        return 0.0
    return _psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _bench(label: str, fn, iterations: int = 1, verbose: bool = False):
    """Запустить fn() iterations раз, вернуть (avg_ms, total_ms)."""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    avg_ms = (sum(times) / len(times)) * 1000
    total_ms = sum(times) * 1000
    if verbose:
        print(f"  {label}: avg={avg_ms:.2f}ms  total={total_ms:.2f}ms  (n={iterations})")
    else:
        print(f"  {label}: {avg_ms:.2f}ms")
    return avg_ms, total_ms, result


# ────────────────────────────────────────────────────────────────────────────
def bench_engine(n_nodes: int = 6, iterations: int = 50, verbose: bool = False):
    print(f"\n── Graph Engine ({n_nodes} nodes, {iterations} iterations) ──────────────────")
    from app.node_graph.engine import NodeGraphEngine
    from app.node_graph.models import GraphEdge, GraphNode

    def _make_graph():
        nodes = [GraphNode(id=f"n{i}", type="source" if i == 0 else ("export" if i == n_nodes - 1 else "load"),
                           title=f"Node{i}")
                 for i in range(n_nodes)]
        edges = [GraphEdge(src_id=f"n{i}", dst_id=f"n{i+1}", src_port="out", dst_port="in" if i == n_nodes - 2 else "image")
                 for i in range(n_nodes - 1)]
        return nodes, edges

    def do_topo():
        eng = NodeGraphEngine()
        nodes, edges = _make_graph()
        return eng.topological_order(nodes, edges)

    def do_validate():
        eng = NodeGraphEngine()
        nodes, edges = _make_graph()
        return eng.validate(nodes, edges)

    def do_build_plan():
        eng = NodeGraphEngine()
        nodes, edges = _make_graph()
        return eng.build_execution_plan_with_diagnostics(nodes, edges)

    mem_before = _rss_mb()
    _bench("topological_order", do_topo, iterations, verbose)
    _bench("validate", do_validate, iterations, verbose)
    _bench("build_execution_plan_with_diagnostics", do_build_plan, iterations, verbose)
    mem_after = _rss_mb()
    if _HAS_PSUTIL:
        print(f"  memory delta: +{mem_after - mem_before:.1f} MB")


def bench_worker_init(iterations: int = 10, verbose: bool = False):
    print(f"\n── InferenceWorker.__init__ alternatives ({iterations} iterations) ──────")
    import threading as _threading

    def make_minimal():
        from app.workers.inference_worker import InferenceWorker
        w = InferenceWorker.__new__(InferenceWorker)
        w.cancel_flag = _threading.Event()
        w.language_code = "ru"
        return w

    _bench("__new__ + cancel_flag", make_minimal, iterations, verbose)


def bench_passthrough_graph(n_frames: int = 30, iterations: int = 5, verbose: bool = False):
    print(f"\n── Passthrough graph execution ({n_frames} frames, {iterations} runs) ──")
    import threading as _threading
    from app.workers.inference_worker import InferenceWorker
    from app.node_graph.models import GraphEdge, GraphNode

    def _make_worker():
        w = InferenceWorker.__new__(InferenceWorker)
        w.cancel_flag = _threading.Event()
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
        w.log_message = type("S", (), {"emit": staticmethod(lambda m: None)})()
        return w

    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(n_frames)]
    nodes = [
        GraphNode(id="src_1", type="source", title="S"),
        GraphNode(id="exp_1", type="export", title="E"),
    ]
    edges = [GraphEdge(src_id="src_1", dst_id="exp_1", src_port="out", dst_port="in")]

    def do_exec():
        w = _make_worker()
        return w._execute_node_graph(nodes, edges, frames)

    avg_ms, _, _ = _bench(f"_execute_node_graph ({n_frames}f)", do_exec, iterations, verbose)
    fps = n_frames / (avg_ms / 1000) if avg_ms > 0 else 0
    print(f"  → throughput: {fps:.0f} passthrough frames/s")


def bench_gather_inputs(n_edges: int = 10, n_frames: int = 20, iterations: int = 200, verbose: bool = False):
    print(f"\n── _gather_node_inputs ({n_edges} edges, {iterations} iterations) ────────")
    import threading as _threading
    from app.workers.inference_worker import InferenceWorker
    from app.node_graph.models import GraphEdge, GraphNode

    w = InferenceWorker.__new__(InferenceWorker)
    w.cancel_flag = _threading.Event()
    w.language_code = "ru"
    w.log_message = type("S", (), {"emit": staticmethod(lambda m: None)})()

    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(n_frames)]
    # Создаём n_edges источников, каждый → финальная нода
    src_nodes = {f"s{i}": GraphNode(id=f"s{i}", type="source", title=f"S{i}") for i in range(n_edges)}
    edges = [GraphEdge(src_id=f"s{i}", dst_id="dst", src_port="out", dst_port=f"in{i}") for i in range(n_edges)]
    outputs = {f"s{i}": {"out": frames} for i in range(n_edges)}

    def do_gather():
        return w._gather_node_inputs(src_nodes, edges, "dst", outputs, frames)

    _bench(f"gather ({n_edges} edges)", do_gather, iterations, verbose)


def bench_runtime_result_builders(iterations: int = 10000, verbose: bool = False):
    print(f"\n── RuntimeResult builders ({iterations} iterations) ─────────────────────")
    from app.runtime_contract import (
        make_runtime_result_ok,
        make_runtime_result_cancelled,
        make_runtime_result_cancelled_partial,
    )

    paths = {f"n{i}": f"/tmp/out{i}.png" for i in range(5)}

    _bench("make_runtime_result_ok", lambda: make_runtime_result_ok(paths, 30), iterations, verbose)
    _bench("make_runtime_result_cancelled", make_runtime_result_cancelled, iterations, verbose)
    _bench("make_runtime_result_cancelled_partial", lambda: make_runtime_result_cancelled_partial(paths, 15), iterations, verbose)


# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 6.2 — KeyFlow Studio profiling")
    parser.add_argument("--frames", type=int, default=30, help="Кол-во синтетических кадров (default 30)")
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод")
    args = parser.parse_args()

    print("=" * 60)
    print("KeyFlow Studio — Stage 6.2 Performance Profiling")
    print("=" * 60)
    print(f"KEYFLOW_DEVICE = {os.environ.get('KEYFLOW_DEVICE', 'auto')}")
    if _HAS_PSUTIL:
        import psutil
        print(f"CPU count: {psutil.cpu_count(logical=True)}")
    else:
        print("psutil not available — memory measurements disabled")

    bench_engine(n_nodes=6, iterations=50, verbose=args.verbose)
    bench_worker_init(iterations=10, verbose=args.verbose)
    bench_passthrough_graph(n_frames=args.frames, iterations=5, verbose=args.verbose)
    bench_gather_inputs(n_edges=10, n_frames=20, iterations=200, verbose=args.verbose)
    bench_runtime_result_builders(iterations=10000, verbose=args.verbose)

    print("\n" + "=" * 60)
    print("Profiling complete.")


if __name__ == "__main__":
    main()
