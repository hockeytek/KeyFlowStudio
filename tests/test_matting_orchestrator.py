import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.coordinators import MattingOrchestrator
from app.utils.write_paths import build_graph_write_output_dir


class _DialogStub:
    def __init__(self, targets=None):
        self._targets = list(targets or [])
        self.persisted_paths = []
        self.runtime_previews = []

    def connected_write_targets(self):
        return list(self._targets)

    def set_write_last_output_path(self, node_id: str, path: str):
        self.persisted_paths.append((node_id, path))

    def set_write_runtime_preview_for_node(self, node_id: str, image):
        self.runtime_previews.append((node_id, image))


class _WriteOutputAdapterStub:
    def __init__(self):
        self.calls = []

    def save_sam_mask_output(self, mask_path: str, write_cfg: dict, fallback_output_dir: Path) -> str:
        self.calls.append(("sam2", mask_path, write_cfg.get("stream", ""), str(fallback_output_dir)))
        return "/tmp/sam_out/0001.png"

    def save_load_output(self, write_cfg: dict, fallback_output_dir: Path) -> str:
        self.calls.append(("load", "", write_cfg.get("stream", ""), str(fallback_output_dir)))
        return "/tmp/load_out/0001.png"

    def save_frames_to_write_output(
        self,
        frames_rgb,
        write_cfg: dict,
        fallback_output_dir: Path,
        default_stem: str,
        *,
        source_is_video: bool,
        source_ext: str,
    ) -> str:
        self.calls.append(("frames", len(frames_rgb), write_cfg.get("stream", ""), str(fallback_output_dir), default_stem))
        return "/tmp/frames_out/0001.png"


class MattingOrchestratorTests(unittest.TestCase):
    def test_resolve_cancel_policy_always_immediate_for_stop(self):
        host = SimpleNamespace(
            _settings=SimpleNamespace(value=lambda *_args, **_kwargs: "save_partial"),
            _tr=lambda key: key,
        )
        orchestrator = MattingOrchestrator(host)
        self.assertEqual(orchestrator._resolve_cancel_policy(), "immediate")

    def test_on_node_frame_progress_updates_status_and_progress_bar(self):
        status = []

        class _ProgressBar:
            def __init__(self):
                self._value = 0

            def value(self):
                return self._value

            def setValue(self, value: int):
                self._value = int(value)

        dialog_calls = []
        host = SimpleNamespace(
            ui=SimpleNamespace(progress_bar=_ProgressBar()),
            _set_status=lambda text: status.append(text),
            _node_graph_dialog=SimpleNamespace(
                set_node_frame_progress=lambda node, cur, tot: dialog_calls.append((node, cur, tot))
            ),
            _tr=lambda key: key,
        )

        orchestrator = MattingOrchestrator(host)
        orchestrator.on_node_frame_progress("matting", 3, 10)

        self.assertEqual(dialog_calls, [("matting", 3, 10)])
        self.assertTrue(status)
        self.assertEqual(status[-1], "MatAnyone2: 3/10")
        self.assertGreaterEqual(host.ui.progress_bar.value(), 20)

    def test_apply_export_preview_path_updates_state_and_selected_preview(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            selected_calls = []
            dialog = _DialogStub()
            host = SimpleNamespace(
                _node_graph_dialog=dialog,
                _set_selected_node_preview=lambda **kwargs: selected_calls.append(kwargs),
            )
            orchestrator = MattingOrchestrator(host)
            orchestrator.set_export_preview_node("write_1")

            orchestrator.apply_export_preview_path("write_1", tmp_path)

            self.assertEqual(orchestrator.saved_output_path_for_node("write_1"), tmp_path)
            self.assertEqual(dialog.persisted_paths, [("write_1", tmp_path)])
            self.assertEqual(selected_calls, [{"source": tmp_path}])
        finally:
            os.unlink(tmp_path)

    def test_apply_export_preview_path_ignores_incomplete_tmp_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "result_tmp.mov"
            tmp_path.write_bytes(b"not finalized")

            selected_calls = []
            dialog = _DialogStub()
            host = SimpleNamespace(
                _node_graph_dialog=dialog,
                _set_selected_node_preview=lambda **kwargs: selected_calls.append(kwargs),
            )
            orchestrator = MattingOrchestrator(host)
            orchestrator.set_export_preview_node("write_1")

            orchestrator.apply_export_preview_path("write_1", str(tmp_path))

            self.assertEqual(orchestrator.saved_output_path_for_node("write_1"), "")
            self.assertEqual(dialog.persisted_paths, [])
            self.assertEqual(selected_calls, [])

    def test_on_graph_stream_preview_routes_selected_nodes(self):
        selected_calls = []
        runtime_preview_calls = []
        dialog = _DialogStub()
        dialog.set_write_runtime_preview_for_node = lambda node_id, image: runtime_preview_calls.append((node_id, image))

        host = SimpleNamespace(
            _node_graph_dialog=dialog,
            _to_qimage=lambda arr: ("qimg", tuple(arr.shape)),
            _preview_array_to_rgb=lambda frame: np.asarray(frame, dtype=np.uint8),
            _set_selected_node_preview=lambda **kwargs: selected_calls.append(kwargs),
            _load_image_for_preview=lambda _path: np.zeros((2, 2, 3), dtype=np.uint8),
        )

        orchestrator = MattingOrchestrator(host)
        frame = np.ones((2, 2, 3), dtype=np.uint8) * 128

        orchestrator.set_birefnet_preview_node("biref_1")
        orchestrator.on_graph_stream_preview("biref_1", {"frame": frame, "stream": "alpha"}, 0)
        self.assertTrue(any("frame" in item for item in selected_calls))

        selected_calls.clear()
        orchestrator.set_export_preview_node("write_1")
        orchestrator.on_graph_stream_preview("write_1", {"frame": frame, "stream": "fg"}, 0)
        self.assertTrue(any("frame" in item for item in selected_calls))
        self.assertTrue(runtime_preview_calls)

    def test_on_graph_stream_preview_preview_only_does_not_persist_output_path(self):
        selected_calls = []
        persisted = []
        dialog = _DialogStub()
        dialog.set_write_last_output_path = lambda node_id, path: persisted.append((node_id, path))

        host = SimpleNamespace(
            _node_graph_dialog=dialog,
            _to_qimage=lambda arr: ("qimg", tuple(arr.shape)),
            _preview_array_to_rgb=lambda frame: np.asarray(frame, dtype=np.uint8),
            _set_selected_node_preview=lambda **kwargs: selected_calls.append(kwargs),
            _load_image_for_preview=lambda _path: np.zeros((2, 2, 3), dtype=np.uint8),
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            orchestrator = MattingOrchestrator(host)
            orchestrator.set_export_preview_node("write_1")
            orchestrator.on_graph_stream_preview(
                "write_1",
                {
                    "frame": np.ones((2, 2, 3), dtype=np.uint8),
                    "path": tmp_path,
                    "stream": "fg",
                    "semantics": "preview_only",
                },
                0,
            )

            self.assertEqual(orchestrator.saved_output_path_for_node("write_1"), "")
            self.assertEqual(persisted, [])
            self.assertTrue(any("frame" in item for item in selected_calls))
        finally:
            os.unlink(tmp_path)

    def test_save_sam_outputs_uses_host_default_output_override(self):
        saves = []
        dialog = _DialogStub(
            targets=[
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
        )
        host = SimpleNamespace(
            _node_graph_dialog=dialog,
            input_path="/tmp/input.mp4",
            _default_run_output_dir=lambda _src: Path("/tmp/out"),
            sam2_graph=SimpleNamespace(
                build_frame_masks=lambda: {
                    0: np.ones((2, 2), dtype=np.uint8) * 255,
                    1: np.zeros((2, 2), dtype=np.uint8),
                },
            ),
            _save_frames_to_write_output=lambda frames, cfg, fallback, default_stem, *, source_is_video, source_ext: (
                saves.append((len(frames), cfg.get("output_dir"), str(fallback), source_is_video, source_ext))
                or "/tmp/out/alpha/0001.png"
            ),
            _to_qimage=lambda _arr: None,
            _set_selected_node_preview=lambda **_kwargs: None,
        )

        orchestrator = MattingOrchestrator(host)
        node_count, frame_count = orchestrator.save_sam_outputs_to_connected_write_nodes()

        self.assertEqual(node_count, 1)
        self.assertEqual(frame_count, 2)
        self.assertEqual(len(saves), 1)
        self.assertEqual(saves[0][1], "/tmp/out/alpha")
        self.assertTrue(saves[0][3])
        self.assertEqual(saves[0][4], ".mp4")

    def test_save_sam_outputs_prefers_resolved_write_output_dir(self):
        saves = []
        dialog = _DialogStub(
            targets=[
                {
                    "source_node_type": "sam2",
                    "stream": "alpha",
                    "graph_node_id": "write_1",
                    "auto_output_dir": True,
                    "output_dir": "",
                    "resolved_output_dir": "/tmp/input_keyflow/SAM2 Mask/Output",
                    "output_format": "png",
                    "file_name": "",
                }
            ]
        )
        host = SimpleNamespace(
            _node_graph_dialog=dialog,
            input_path="/tmp/input.mp4",
            _default_run_output_dir=lambda _src: Path("/tmp/input_keyflow"),
            sam2_graph=SimpleNamespace(build_frame_masks=lambda: {0: np.ones((2, 2), dtype=np.uint8) * 255}),
            _save_frames_to_write_output=lambda frames, cfg, fallback, default_stem, *, source_is_video, source_ext: (
                saves.append((cfg.get("output_dir"), cfg.get("resolved_output_dir"), str(fallback)))
                or "/tmp/input_keyflow/SAM2 Mask/Output/sam_mask_f0001.png"
            ),
            _to_qimage=lambda _arr: None,
            _set_selected_node_preview=lambda **_kwargs: None,
        )

        MattingOrchestrator(host).save_sam_outputs_to_connected_write_nodes()

        self.assertEqual(saves, [("", "/tmp/input_keyflow/SAM2 Mask/Output", "/tmp/input_keyflow/sam_mask")])

    def test_execute_passthrough_targets_uses_write_output_adapter(self):
        adapter = _WriteOutputAdapterStub()
        host = SimpleNamespace(_node_graph_dialog=None)
        orchestrator = MattingOrchestrator(host, write_output_adapter=adapter)

        fg_path, alpha_path = orchestrator.execute_passthrough_targets(
            [
                {"source_node_type": "sam2", "stream": "alpha"},
                {"source_node_type": "load", "stream": "fg"},
            ],
            mask_path="/tmp/mask.png",
            output_dir=Path("/tmp/out"),
        )

        self.assertEqual(alpha_path, "/tmp/sam_out/0001.png")
        self.assertEqual(fg_path, "")
        self.assertEqual(adapter.calls[0][0], "sam2")

    def test_execute_passthrough_targets_load_only_uses_write_output_adapter(self):
        adapter = _WriteOutputAdapterStub()
        host = SimpleNamespace(_node_graph_dialog=None)
        orchestrator = MattingOrchestrator(host, write_output_adapter=adapter)

        fg_path, alpha_path = orchestrator.execute_passthrough_targets(
            [
                {"source_node_type": "load", "stream": "img"},
            ],
            mask_path="",
            output_dir=Path("/tmp/out"),
        )

        self.assertEqual(alpha_path, "")
        self.assertEqual(fg_path, "/tmp/load_out/0001.png")
        self.assertEqual(adapter.calls[0][0], "load")

    def test_find_existing_write_output_path_uses_resolved_graph_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "input.mp4"
            source_path.write_bytes(b"0")
            base_dir = Path(tmpdir) / "input_keyflow"
            resolved_dir = build_graph_write_output_dir(
                base_dir,
                source_node_title="GVM",
                port_label="Alpha",
                stream_label="alpha",
            )
            resolved_dir.mkdir(parents=True, exist_ok=True)
            expected = resolved_dir / "0000.png"
            expected.write_bytes(b"1")

            host = SimpleNamespace(input_path=str(source_path))
            orchestrator = MattingOrchestrator(host)

            restored = orchestrator.resolve_write_output_path(
                {
                    "graph_node_id": "write_1",
                    "stream": "alpha",
                    "source_path": str(source_path),
                    "source_node_title": "GVM",
                    "port_label": "Alpha",
                    "auto_output_dir": True,
                    "output_dir": "",
                    "resolved_output_dir": str(resolved_dir),
                    "output_format": "png",
                    "file_name": "",
                    "last_output_path": "",
                }
            )

            self.assertEqual(restored, str(expected))

    def test_find_existing_write_output_path_skips_tmp_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "input.mp4"
            source_path.write_bytes(b"0")
            out_dir = Path(tmpdir) / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp_video = out_dir / "input_tmp.mov"
            tmp_video.write_bytes(b"not finalized")

            host = SimpleNamespace(input_path=str(source_path))
            orchestrator = MattingOrchestrator(host)

            restored = orchestrator.resolve_write_output_path(
                {
                    "graph_node_id": "write_1",
                    "stream": "alpha",
                    "source_path": str(source_path),
                    "auto_output_dir": False,
                    "output_dir": str(out_dir),
                    "output_format": "mov",
                    "file_name": "input",
                    "last_output_path": "",
                }
            )

            self.assertEqual(restored, "")

    def test_cleanup_pending_temp_sam_mask_deletes_only_tracked_temp_file(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        host = SimpleNamespace(
            sam2=SimpleNamespace(
                cleanup_temporary_processing_mask_path=lambda path: os.unlink(path),
                is_temporary_processing_mask_path=lambda path: path == tmp_path,
            ),
            _tr=lambda key: key,
        )
        orchestrator = MattingOrchestrator(host)
        orchestrator._track_temp_sam_mask_path(tmp_path)
        orchestrator._cleanup_pending_temp_sam_mask()

        self.assertFalse(os.path.exists(tmp_path))
        self.assertEqual(orchestrator._pending_temp_sam_mask_path, "")

    def test_try_graph_run_alpha_to_matting_mask_does_not_require_external_mask(self):
        preset = {
            "nodes": [
                {"id": "alpha_1", "type": "alpha", "title": "Alpha", "properties": {"enabled": True}},
                {"id": "matting_1", "type": "matting", "title": "MatAnyone2", "properties": {"enabled": True}},
            ],
            "connections": [
                {"src": "alpha_1", "dst": "matting_1", "src_port": "out", "dst_port": "mask"},
            ],
        }
        dialog = SimpleNamespace(export_graph_preset=lambda: preset)
        host = SimpleNamespace(
            _node_graph_dialog=dialog,
            _resolve_mask_path_for_processing=lambda: (_ for _ in ()).throw(AssertionError("must not be called")),
            _tr=lambda key: key,
            sam2_graph=SimpleNamespace(selected_graph_mask_rows=lambda: []),
            sam2=SimpleNamespace(state=SimpleNamespace(get_correction_masks_by_frame=lambda _rows: {})),
            is_video_input=False,
            _compatibility_profile="quality",
        )

        orchestrator = MattingOrchestrator(host)
        start_calls = []
        orchestrator.start_matting_run = lambda mask_path, output_dir, config: start_calls.append(
            (mask_path, output_dir, config)
        )

        started = orchestrator.try_graph_inference_run(Path("/tmp/out"), 0, -1)

        self.assertTrue(started)
        self.assertEqual(len(start_calls), 1)
        self.assertEqual(start_calls[0][0], "")

    def test_try_graph_run_sam2_to_matting_mask_does_not_pass_correction_sequence(self):
        preset = {
            "nodes": [
                {"id": "sam_1", "type": "sam2", "title": "SAM2", "properties": {"enabled": True}},
                {"id": "matting_1", "type": "matting", "title": "MatAnyone2", "properties": {"enabled": True}},
            ],
            "connections": [
                {"src": "sam_1", "dst": "matting_1", "src_port": "out", "dst_port": "mask"},
            ],
        }
        correction_calls = []
        dialog = SimpleNamespace(export_graph_preset=lambda: preset)
        host = SimpleNamespace(
            _node_graph_dialog=dialog,
            _resolve_mask_path_for_processing=lambda: "/tmp/sam_seed.png",
            _tr=lambda key: key,
            sam2_graph=SimpleNamespace(selected_graph_mask_rows=lambda: []),
            sam2=SimpleNamespace(
                state=SimpleNamespace(
                    get_correction_masks_by_frame=lambda rows: correction_calls.append(rows)
                    or {1: np.ones((2, 2), dtype=np.uint8) * 255}
                )
            ),
            is_video_input=True,
            _compatibility_profile="quality",
        )

        orchestrator = MattingOrchestrator(host)
        start_calls = []
        orchestrator.start_matting_run = lambda mask_path, output_dir, config: start_calls.append(
            (mask_path, output_dir, config)
        )

        started = orchestrator.try_graph_inference_run(Path("/tmp/out"), 0, 100)

        self.assertTrue(started)
        self.assertEqual(start_calls[0][0], "/tmp/sam_seed.png")
        self.assertIsNone(start_calls[0][2].get("correction_masks"))
        self.assertEqual(correction_calls, [])

    def test_try_graph_run_sam3_to_write_starts_graph_inference(self):
        preset = {
            "nodes": [
                {"id": "source_1", "type": "source", "title": "Source", "properties": {"enabled": True}},
                {
                    "id": "sam3_1",
                    "type": "sam3",
                    "title": "SAM3 Mask",
                    "properties": {"enabled": True, "concept": "person"},
                },
                {"id": "write_1", "type": "export", "title": "Write", "properties": {"enabled": True}},
            ],
            "connections": [
                {"src": "source_1", "dst": "sam3_1", "src_port": "out", "dst_port": "img"},
                {"src": "sam3_1", "dst": "write_1", "src_port": "out", "dst_port": "in"},
            ],
        }
        dialog = SimpleNamespace(export_graph_preset=lambda: preset)
        host = SimpleNamespace(
            _node_graph_dialog=dialog,
            _resolve_mask_path_for_processing=lambda: (_ for _ in ()).throw(AssertionError("must not be called")),
            _tr=lambda key: key,
            sam2_graph=SimpleNamespace(selected_graph_mask_rows=lambda: []),
            sam2=SimpleNamespace(state=SimpleNamespace(get_correction_masks_by_frame=lambda _rows: {})),
            is_video_input=True,
            _compatibility_profile="quality",
        )

        orchestrator = MattingOrchestrator(host)
        start_calls = []
        orchestrator.start_matting_run = lambda mask_path, output_dir, config: start_calls.append(
            (mask_path, output_dir, config)
        )

        started = orchestrator.try_graph_inference_run(Path("/tmp/out"), 2, 7)

        self.assertTrue(started)
        self.assertEqual(len(start_calls), 1)
        self.assertEqual(start_calls[0][0], "")
        self.assertEqual(start_calls[0][2]["start_frame"], 2)
        self.assertEqual(start_calls[0][2]["end_frame"], 7)
        self.assertEqual(start_calls[0][2]["node_graph"]["nodes"][1]["type"], "sam3")

    def test_on_matting_finished_cancelled_with_partial_saved_paths_sets_partial_status(self):
        statuses = []
        previews = []

        class _Dialog:
            def clear_birefnet_runtime_progress(self):
                return None

            def clear_node_frame_progress(self):
                return None

            def write_node_ids_for_stream(self, stream: str):
                if stream == "fg":
                    return ["write_fg"]
                if stream == "alpha":
                    return ["write_alpha"]
                return []

            def set_write_runtime_preview_for_node(self, _node_id, _image):
                return None

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fg_tmp, tempfile.NamedTemporaryFile(suffix=".png", delete=False) as alpha_tmp:
            fg_path = fg_tmp.name
            alpha_path = alpha_tmp.name

        try:
            host = SimpleNamespace(
                ui=SimpleNamespace(
                    btn_run=SimpleNamespace(setEnabled=lambda _v: None),
                    progress_bar=SimpleNamespace(setValue=lambda _v: None),
                ),
                _refresh_stop_button_state=lambda: None,
                _node_graph_dialog=_Dialog(),
                _set_status=lambda text: statuses.append(text),
                _show_output_preview=lambda fg, alpha: previews.append((fg, alpha)),
                _play_completion_sound=lambda: None,
                _set_sam_controls_busy=lambda _busy: None,
                sam2=SimpleNamespace(generation_active=False),
                _tr=lambda key: {
                    "matting_status_stopped_partial": "partial stopped",
                    "matting_status_stopped": "stopped",
                }.get(key, key),
                _load_preview_image_or_video_frame=lambda _path: np.zeros((2, 2, 3), dtype=np.uint8),
                _to_qimage=lambda _arr: None,
            )

            orchestrator = MattingOrchestrator(host)
            orchestrator.on_matting_finished(
                {
                    "status": "cancelled",
                    "cancelled": True,
                    "partial_result": True,
                    "partial_saved_paths": {
                        "write_fg": fg_path,
                        "write_alpha": alpha_path,
                    },
                }
            )

            self.assertIn("partial stopped", statuses)
            self.assertEqual(previews, [(fg_path, alpha_path)])
        finally:
            os.unlink(fg_path)
            os.unlink(alpha_path)


if __name__ == "__main__":
    unittest.main()
