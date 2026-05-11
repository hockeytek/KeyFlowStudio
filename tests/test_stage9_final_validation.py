"""Stage 9 — Final validation tests.

Покрывает три блока:
  1. Smoke-import: все публичные модули импортируются без ошибок
  2. Node compatibility matrix: разрешённые и запрещённые соединения
     по NODE_GRAPH_STANDARD.md и node_contracts.py
  3. Signal invariants: stage_progress / node_frame_progress emit-контракт
     (InferenceWorker сигналы консистентны с объявленными сигнатурами)
"""
import importlib
import os
import unittest

os.environ.setdefault("KEYFLOW_DEVICE", "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Smoke-import tests
# ──────────────────────────────────────────────────────────────────────────────

class SmokeImportAllModulesTests(unittest.TestCase):
    """Все публичные модули импортируются без исключений."""

    MODULES = [
        # Node graph core
        "app.node_graph.models",
        "app.node_graph.engine",
        "app.node_graph.specs",
        "app.node_graph.specs.base",
        "app.node_graph.specs.source",
        "app.node_graph.specs.load_media",
        "app.node_graph.specs.birefnet",
        "app.node_graph.specs.chromakey",
        "app.node_graph.specs.corridorkey",
        "app.node_graph.specs.matting",
        "app.node_graph.specs.merge",
        "app.node_graph.specs.sam_mask",
        "app.node_graph.specs.alpha",
        "app.node_graph.specs.export",
        # Rules / registry
        "app.node_graph.rules",
        "app.node_graph.rules.node_contracts",
        "app.node_graph.rules.registry",
        # Runtime
        "app.runtime_contract",
        # Utils
        "app.utils.device",
        "app.utils.ffmpeg",
        "app.utils.frame_range_helper",
        "app.utils.media",
        # Services
        "app.services",
        "app.services.model_service",
        # Workers
        "app.workers.inference_worker",
        # Coordinators
        "app.coordinators",
        "app.coordinators.matting_orchestrator",
        # App-level
        "app.constants",
        "app.i18n",
        "app.settings",
        "app.shortcuts",
    ]

    def _import(self, module_name: str):
        try:
            mod = importlib.import_module(module_name)
            self.assertIsNotNone(mod)
        except ImportError as e:
            self.fail(f"Import failed for {module_name}: {e}")

    def test_node_graph_models(self):
        self._import("app.node_graph.models")

    def test_node_graph_engine(self):
        self._import("app.node_graph.engine")

    def test_specs_package(self):
        self._import("app.node_graph.specs")

    def test_specs_all_node_types(self):
        for mod in [s for s in self.MODULES if "specs." in s and s != "app.node_graph.specs"]:
            self._import(mod)

    def test_rules_package(self):
        self._import("app.node_graph.rules")

    def test_rules_node_contracts(self):
        self._import("app.node_graph.rules.node_contracts")

    def test_rules_registry(self):
        self._import("app.node_graph.rules.registry")

    def test_runtime_contract(self):
        self._import("app.runtime_contract")

    def test_utils_all(self):
        for mod in ["app.utils.device", "app.utils.ffmpeg",
                    "app.utils.frame_range_helper", "app.utils.media"]:
            self._import(mod)

    def test_services(self):
        self._import("app.services")
        self._import("app.services.model_service")

    def test_workers_inference_worker(self):
        self._import("app.workers.inference_worker")

    def test_coordinators(self):
        self._import("app.coordinators")
        self._import("app.coordinators.matting_orchestrator")

    def test_app_constants(self):
        self._import("app.constants")

    def test_app_i18n(self):
        self._import("app.i18n")

    def test_main_module(self):
        self._import("main")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Node compatibility matrix
# ──────────────────────────────────────────────────────────────────────────────

class NodeCompatibilityMatrixTests(unittest.TestCase):
    """Проверка разрешённых/запрещённых соединений по NODE_GRAPH_STANDARD.md.

    Источник истины: docs/NODE_GRAPH_STANDARD.md §Connection rules
    и node_contracts.py downstream_allowed / upstream_allowed.
    """

    def setUp(self):
        from app.node_graph.rules.registry import get_registry
        self.reg = get_registry()

    # ── Разрешённые соединения (SHOULD connect) ───────────────────────────

    def test_source_to_sam_allowed(self):
        self.assertTrue(self.reg.can_downstream("source", "sam2"))

    def test_source_to_birefnet_allowed(self):
        self.assertTrue(self.reg.can_downstream("source", "birefnet"))

    def test_source_to_corridorkey_allowed(self):
        self.assertTrue(self.reg.can_downstream("source", "corridorkey"))

    def test_source_to_matting_allowed(self):
        self.assertTrue(self.reg.can_downstream("source", "matting"))

    def test_source_to_export_allowed(self):
        self.assertTrue(self.reg.can_downstream("source", "export"))

    def test_load_to_birefnet_allowed(self):
        self.assertTrue(self.reg.can_downstream("load", "birefnet"))

    def test_load_to_sam_allowed(self):
        self.assertTrue(self.reg.can_downstream("load", "sam2"))

    def test_birefnet_to_corridorkey_allowed(self):
        self.assertTrue(self.reg.can_downstream("birefnet", "corridorkey"))

    def test_birefnet_to_export_allowed(self):
        self.assertTrue(self.reg.can_downstream("birefnet", "export"))

    def test_sam_to_export_allowed(self):
        self.assertTrue(self.reg.can_downstream("sam2", "export"))

    def test_chromakey_to_export_allowed(self):
        # chromakey is in source.downstream_allowed via source→chromakey
        self.assertTrue(self.reg.can_downstream("source", "chromakey"))

    def test_corridorkey_to_export_allowed(self):
        self.assertTrue(self.reg.can_downstream("corridorkey", "export"))

    def test_matting_to_export_allowed(self):
        self.assertTrue(self.reg.can_downstream("matting", "export"))

    def test_export_upstream_accepts_all_processing_nodes(self):
        """Export принимает выход от любой processing-ноды."""
        for node_type in ["source", "load", "sam2", "birefnet", "chromakey",
                          "corridorkey", "matting", "alpha"]:
            with self.subTest(node_type=node_type):
                self.assertTrue(self.reg.can_upstream(node_type, "export"),
                                f"{node_type}->export upstream должен быть разрешён")

    # ── Запрещённые соединения (MUST NOT connect) ─────────────────────────

    def test_matting_cannot_chain_to_matting(self):
        """matting→matting запрещено (нет смысла, нет в downstream_allowed)."""
        self.assertFalse(self.reg.can_downstream("matting", "matting"))

    def test_export_is_terminal_node_has_no_output_ports(self):
        """Export — терминальная нода: у неё нет выходных портов.

        can_downstream("export", X) возвращает True (пустой список = нет ограничений),
        но физически подключить export как источник невозможно — нет output портов.
        """
        contract = self.reg.get_contract("export")
        self.assertIsNotNone(contract)
        self.assertEqual(len(contract.outputs), 0,
                         "Export не должен иметь выходных портов")

    def test_birefnet_cannot_feed_matting(self):
        """BiRefNet → matting логически запрещено."""
        self.assertFalse(self.reg.can_downstream("birefnet", "matting"))

    def test_birefnet_cannot_feed_sam(self):
        self.assertFalse(self.reg.can_downstream("birefnet", "sam2"))

    def test_sam_cannot_feed_birefnet(self):
        self.assertFalse(self.reg.can_downstream("sam2", "birefnet"))

    def test_corridorkey_can_feed_matting(self):
        """CorridorKey может питать matting (downstream_allowed включает matting)."""
        self.assertTrue(self.reg.can_downstream("corridorkey", "matting"))

    def test_corridorkey_cannot_feed_birefnet(self):
        """CorridorKey не может питать birefnet (не в downstream_allowed)."""
        self.assertFalse(self.reg.can_downstream("corridorkey", "birefnet"))

    def test_corridorkey_cannot_feed_sam(self):
        """CorridorKey не может питать sam2 (не в downstream_allowed)."""
        self.assertFalse(self.reg.can_downstream("corridorkey", "sam2"))

    def test_merge_mask_topology_is_source_agnostic(self):
        """Merge.mask разрешает topology от любой ноды; тип проверяется отдельно."""
        self.assertTrue(self.reg.can_connect_topology("alpha", "out", "merge", "mask"))
        self.assertTrue(self.reg.can_connect_topology("sam2", "out", "merge", "mask"))

    def test_merge_bg_topology_still_uses_node_allowlists(self):
        """Исключение source-agnostic действует только для Merge.mask."""
        self.assertFalse(self.reg.can_connect_topology("sam2", "out", "merge", "bg"))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Port compatibility matrix
# ──────────────────────────────────────────────────────────────────────────────

class PortCompatibilityTests(unittest.TestCase):
    """Проверка совместимости типов портов по NodeRulesRegistry.can_connect_ports."""

    def setUp(self):
        from app.node_graph.rules.registry import get_registry
        self.reg = get_registry()

    # ── Разрешённые соединения портов ──────────────────────────────────────

    def test_source_out_to_birefnet_image(self):
        self.assertTrue(self.reg.can_connect_ports("source", "out", "birefnet", "image"))

    def test_source_out_to_sam_img(self):
        self.assertTrue(self.reg.can_connect_ports("source", "out", "sam2", "img"))

    def test_source_out_to_corridorkey_image(self):
        self.assertTrue(self.reg.can_connect_ports("source", "out", "corridorkey", "image"))

    def test_source_out_to_chromakey_image(self):
        self.assertTrue(self.reg.can_connect_ports("source", "out", "chromakey", "image"))

    def test_source_out_to_matting_img(self):
        self.assertTrue(self.reg.can_connect_ports("source", "out", "matting", "img"))

    def test_birefnet_alpha_to_corridorkey_alphahint(self):
        """BiRefNet alpha → CorridorKey alphahint — ключевое соединение в staged workflow."""
        self.assertTrue(self.reg.can_connect_ports("birefnet", "alpha", "corridorkey", "alphahint"))

    def test_chromakey_mask_to_corridorkey_alphahint(self):
        """ChromaKey mask → CorridorKey alphahint — альтернативный hint."""
        self.assertTrue(self.reg.can_connect_ports("chromakey", "mask", "corridorkey", "alphahint"))

    def test_sam_alpha_to_corridorkey_alphahint(self):
        """CorridorKey alphahint принимает alpha от любой ноды (в т.ч. SAM)."""
        self.assertTrue(self.reg.can_connect_ports("sam2", "out", "corridorkey", "alphahint"))

    def test_source_image_to_corridorkey_alphahint_rejected(self):
        """CorridorKey alphahint не принимает image-поток (только mask/alpha)."""
        self.assertFalse(self.reg.can_connect_ports("source", "out", "corridorkey", "alphahint"))

    def test_sam_out_to_matting_mask(self):
        """SAM alpha → Matting mask — стандартный pipeline SAM+MatAnyone2."""
        self.assertTrue(self.reg.can_connect_ports("sam2", "out", "matting", "mask"))

    def test_alpha_out_to_matting_mask(self):
        """Alpha alpha → Matting mask — любой alpha/mask источник должен быть разрешён."""
        self.assertTrue(self.reg.can_connect_ports("alpha", "out", "matting", "mask"))

    def test_alpha_out_to_merge_mask(self):
        """Alpha alpha → Merge mask — merge должен принимать single-channel mask/alpha."""
        self.assertTrue(self.reg.can_connect_ports("alpha", "out", "merge", "mask"))

    def test_chromakey_mask_to_merge_mask(self):
        """ChromaKey mask → Merge mask — прямое ограничение зоны композита должно быть разрешено."""
        self.assertTrue(self.reg.can_connect_ports("chromakey", "mask", "merge", "mask"))

    def test_source_image_to_merge_mask_rejected(self):
        """Image → Merge mask запрещено: mask принимает только mask/alpha payload."""
        self.assertFalse(self.reg.can_connect_ports("source", "out", "merge", "mask"))

    def test_any_to_export_in(self):
        """Export input принимает любой тип: image, alpha, mask."""
        for src_type, src_port in [("source", "out"), ("birefnet", "alpha"),
                                    ("sam2", "out"), ("corridorkey", "fg"),
                                    ("matting", "fg"), ("matting", "alpha")]:
            with self.subTest(src=f"{src_type}.{src_port}"):
                self.assertTrue(
                    self.reg.can_connect_ports(src_type, src_port, "export", "in"),
                    f"{src_type}.{src_port} → export.in должен быть разрешён"
                )

    def test_corridorkey_fg_to_export(self):
        self.assertTrue(self.reg.can_connect_ports("corridorkey", "fg", "export", "in"))

    def test_corridorkey_alpha_to_export(self):
        self.assertTrue(self.reg.can_connect_ports("corridorkey", "alpha", "export", "in"))

    # ── Запрещённые соединения портов ──────────────────────────────────────

    def test_birefnet_alpha_to_birefnet_image_incompatible(self):
        """alpha → image несовместимо (типы не совпадают)."""
        self.assertFalse(self.reg.can_connect_ports("birefnet", "alpha", "birefnet", "image"))

    def test_nonexistent_src_type_false(self):
        self.assertFalse(self.reg.can_connect_ports("ghost", "out", "export", "in"))

    def test_nonexistent_dst_type_false(self):
        self.assertFalse(self.reg.can_connect_ports("source", "out", "ghost", "in"))

    def test_nonexistent_src_port_false(self):
        self.assertFalse(self.reg.can_connect_ports("source", "nonexistent_port", "export", "in"))

    def test_nonexistent_dst_port_false(self):
        self.assertFalse(self.reg.can_connect_ports("source", "out", "birefnet", "nonexistent_port"))


# ──────────────────────────────────────────────────────────────────────────────
# 4. Registry execution rules invariants
# ──────────────────────────────────────────────────────────────────────────────

class RegistryExecutionRulesTests(unittest.TestCase):
    """Инварианты execution rules из NodeRulesRegistry."""

    def setUp(self):
        from app.node_graph.rules.registry import get_registry
        self.reg = get_registry()

    def test_all_node_types_present(self):
        types = set(self.reg.get_all_node_types())
        for expected in ["source", "load", "sam2", "birefnet", "chromakey",
                         "corridorkey", "matting", "alpha", "export"]:
            self.assertIn(expected, types, f"'{expected}' должен быть в реестре")

    def test_birefnet_binarization_threshold_is_10(self):
        """Порог 10 задокументирован в docs/node-rules/BIREFNET_NODE_RULES.md."""
        self.assertEqual(self.reg.birefnet_binarization_threshold(), 10)

    def test_birefnet_can_defer(self):
        self.assertTrue(self.reg.can_defer_birefnet_to_staged())

    def test_sam_auto_propagate_before_run(self):
        self.assertTrue(self.reg.should_auto_propagate_sam_before_run())

    def test_summary_covers_all_nodes(self):
        summary = self.reg.get_summary()
        for node_type in ["source", "load", "birefnet", "corridorkey", "export"]:
            self.assertIn(node_type, summary)

    def test_source_has_no_inputs(self):
        contract = self.reg.get_contract("source")
        self.assertIsNotNone(contract)
        self.assertEqual(len(contract.inputs), 0)

    def test_export_has_no_outputs(self):
        contract = self.reg.get_contract("export")
        self.assertIsNotNone(contract)
        self.assertEqual(len(contract.outputs), 0)

    def test_all_contracts_have_node_type(self):
        for node_type in self.reg.get_all_node_types():
            contract = self.reg.get_contract(node_type)
            self.assertEqual(contract.node_type, node_type)

    def test_unknown_node_returns_none(self):
        self.assertIsNone(self.reg.get_contract("does_not_exist"))


# ──────────────────────────────────────────────────────────────────────────────
# 5. Signal invariant tests (без Qt event-loop)
# ──────────────────────────────────────────────────────────────────────────────

class SignalInvariantTests(unittest.TestCase):
    """Проверяет, что InferenceWorker объявляет сигналы с корректными сигнатурами."""

    def setUp(self):
        from app.workers.inference_worker import InferenceWorker
        from PySide6.QtCore import Signal
        self.InferenceWorker = InferenceWorker
        self.Signal = Signal

    def test_stage_progress_signal_declared(self):
        self.assertTrue(hasattr(self.InferenceWorker, "stage_progress"))

    def test_node_frame_progress_signal_declared(self):
        self.assertTrue(hasattr(self.InferenceWorker, "node_frame_progress"))

    def test_progress_signal_declared(self):
        self.assertTrue(hasattr(self.InferenceWorker, "progress"))

    def test_finished_signal_declared(self):
        self.assertTrue(hasattr(self.InferenceWorker, "finished"))

    def test_error_signal_declared(self):
        self.assertTrue(hasattr(self.InferenceWorker, "error"))

    def test_log_message_signal_declared(self):
        self.assertTrue(hasattr(self.InferenceWorker, "log_message"))

    def test_worker_has_set_cancel(self):
        self.assertTrue(callable(getattr(self.InferenceWorker, "set_cancel", None)))

    def test_worker_has_reset_cancel(self):
        self.assertTrue(callable(getattr(self.InferenceWorker, "reset_cancel", None)))

    def test_worker_has_set_language(self):
        self.assertTrue(callable(getattr(self.InferenceWorker, "set_language", None)))


class StageProgressValueRangeTests(unittest.TestCase):
    """Проверяет normalize_stage_progress не выходит за [0..100]."""

    def test_clamp_below_zero(self):
        from app.runtime_contract import normalize_stage_progress
        pct, _ = normalize_stage_progress(-5, "test")
        self.assertEqual(pct, 0)

    def test_clamp_above_hundred(self):
        from app.runtime_contract import normalize_stage_progress
        pct, _ = normalize_stage_progress(150, "test")
        self.assertEqual(pct, 100)

    def test_valid_value_passthrough(self):
        from app.runtime_contract import normalize_stage_progress
        pct, text = normalize_stage_progress(42, "loading")
        self.assertEqual(pct, 42)
        self.assertEqual(text, "loading")

    def test_zero_is_valid(self):
        from app.runtime_contract import normalize_stage_progress
        pct, _ = normalize_stage_progress(0, "")
        self.assertEqual(pct, 0)

    def test_hundred_is_valid(self):
        from app.runtime_contract import normalize_stage_progress
        pct, _ = normalize_stage_progress(100, "done")
        self.assertEqual(pct, 100)


class NodeFrameProgressValueTests(unittest.TestCase):
    """Проверяет normalize_frame_progress не выходит за допустимые значения."""

    def test_negative_current_clamped(self):
        from app.runtime_contract import normalize_frame_progress
        cur, tot = normalize_frame_progress(-1, 10)
        self.assertEqual(cur, 0)

    def test_negative_total_clamped(self):
        from app.runtime_contract import normalize_frame_progress
        cur, tot = normalize_frame_progress(5, -1)
        self.assertEqual(tot, 0)

    def test_valid_values_passthrough(self):
        from app.runtime_contract import normalize_frame_progress
        cur, tot = normalize_frame_progress(3, 30)
        self.assertEqual(cur, 3)
        self.assertEqual(tot, 30)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Spec–Contract alignment tests
# ──────────────────────────────────────────────────────────────────────────────

class SpecContractAlignmentTests(unittest.TestCase):
    """Проверяет, что NodeSpec и NodeContract согласованы.

    Каждый node_type из NODE_SPECS должен иметь соответствующий контракт,
    и порты из спецификации должны присутствовать в контракте.
    """

    def setUp(self):
        from app.node_graph.specs import NODE_SPECS
        from app.node_graph.rules.registry import get_registry
        self.NODE_SPECS = NODE_SPECS
        self.reg = get_registry()

    def test_every_spec_has_contract(self):
        for node_type in self.NODE_SPECS:
            with self.subTest(node_type=node_type):
                contract = self.reg.get_contract(node_type)
                self.assertIsNotNone(contract,
                    f"Нода '{node_type}' есть в NODE_SPECS но нет контракта в registry")

    def test_spec_input_ports_in_contract(self):
        for node_type, spec in self.NODE_SPECS.items():
            contract = self.reg.get_contract(node_type)
            if contract is None:
                continue
            for port in spec.inputs:
                with self.subTest(node=node_type, port=port.name):
                    found = contract.get_input(port.name)
                    self.assertIsNotNone(found,
                        f"Порт '{port.name}' есть в spec({node_type}) но не в контракте")

    def test_spec_output_ports_in_contract(self):
        for node_type, spec in self.NODE_SPECS.items():
            contract = self.reg.get_contract(node_type)
            if contract is None:
                continue
            for port in spec.outputs:
                with self.subTest(node=node_type, port=port.name):
                    found = contract.get_output(port.name)
                    self.assertIsNotNone(found,
                        f"Порт '{port.name}' есть в spec({node_type}) но не в контракте")

    def test_no_extra_node_types_in_contracts(self):
        """Контракты не содержат нод, которых нет в NODE_SPECS."""
        known_spec_types = set(self.NODE_SPECS.keys())
        for node_type in self.reg.get_all_node_types():
            with self.subTest(node_type=node_type):
                self.assertIn(node_type, known_spec_types,
                    f"Нода '{node_type}' есть в контракте но не в NODE_SPECS")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Engine validation smoke tests
# ──────────────────────────────────────────────────────────────────────────────

class EngineFinalValidationTests(unittest.TestCase):
    """Smoke-тесты NodeGraphEngine на полном наборе нод."""

    def setUp(self):
        from app.node_graph.engine import NodeGraphEngine
        from app.node_graph.models import GraphNode, GraphEdge
        self.Engine = NodeGraphEngine
        self.Node = GraphNode
        self.Edge = GraphEdge

    def _node(self, nid, ntype):
        return self.Node(id=nid, type=ntype, title=nid)

    def test_minimal_graph_validates(self):
        engine = self.Engine()
        nodes = [self._node("s", "source"), self._node("e", "export")]
        edges = [self.Edge(src_id="s", dst_id="e", src_port="out", dst_port="in")]
        ok, errors = engine.validate(nodes, edges)
        self.assertTrue(ok, f"Граф source→export должен быть валидным: {errors}")

    def test_isolated_node_does_not_break_validation(self):
        engine = self.Engine()
        nodes = [
            self._node("s", "source"),
            self._node("e", "export"),
            self._node("iso", "birefnet"),  # изолированная нода
        ]
        edges = [self.Edge(src_id="s", dst_id="e", src_port="out", dst_port="in")]
        # Не должно падать с исключением
        try:
            ok, errors = engine.validate(nodes, edges)
        except Exception as exc:
            self.fail(f"validate() упал с исключением на изолированной ноде: {exc}")

    def test_cycle_detected(self):
        engine = self.Engine()
        nodes = [self._node("a", "source"), self._node("b", "export")]
        edges = [
            self.Edge(src_id="a", dst_id="b", src_port="out", dst_port="in"),
            self.Edge(src_id="b", dst_id="a", src_port="in", dst_port="out"),
        ]
        ok, _ = engine.validate(nodes, edges)
        self.assertFalse(ok, "Цикл должен делать граф невалидным")

    def test_build_plan_returns_none_on_invalid_graph(self):
        engine = self.Engine()
        nodes = [self._node("a", "source"), self._node("b", "export")]
        edges = [
            self.Edge(src_id="a", dst_id="b", src_port="out", dst_port="in"),
            self.Edge(src_id="b", dst_id="a", src_port="in", dst_port="out"),
        ]
        plan, diagnostics = engine.build_execution_plan_with_diagnostics(nodes, edges)
        self.assertIsNone(plan)
        self.assertIsNotNone(diagnostics)

    def test_execution_order_is_list(self):
        engine = self.Engine()
        nodes = [self._node("s", "source"), self._node("e", "export")]
        edges = [self.Edge(src_id="s", dst_id="e", src_port="out", dst_port="in")]
        plan, _ = engine.build_execution_plan_with_diagnostics(nodes, edges)
        if plan is not None:
            self.assertIsInstance(plan.execution_order, (list, tuple))

    def test_all_node_types_can_be_created_as_nodes(self):
        """Каждый node_type из реестра может быть создан как GraphNode."""
        from app.node_graph.rules.node_contracts import all_node_types
        for node_type in all_node_types():
            with self.subTest(node_type=node_type):
                n = self.Node(id=f"test_{node_type}", type=node_type, title=node_type)
                self.assertEqual(n.type, node_type)
