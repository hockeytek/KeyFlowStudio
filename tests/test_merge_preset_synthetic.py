"""Synthetic test for merge preset - verify RGBA foreground compositing over background."""
import os
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import cv2

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("KEYFLOW_DEVICE", "cpu")

from app.node_graph.models import GraphEdge, GraphNode
from app.utils.media import load_image_float, save_exr_image
from app.workers.inference_worker import InferenceWorker


def _make_worker() -> InferenceWorker:
    """Create InferenceWorker with minimal init (no QObject/Qt)."""
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
    # Mock all signals used by worker
    def noop_emit(*args, **kwargs):
        pass
    w.log_message = type("Sig", (), {"emit": noop_emit})()
    w.graph_stream_preview = type("Sig", (), {"emit": noop_emit})()
    w.node_execution_started = type("Sig", (), {"emit": noop_emit})()
    w.node_execution_complete = type("Sig", (), {"emit": noop_emit})()
    w.node_frame_progress = type("Sig", (), {"emit": noop_emit})()
    w.error_occurred = type("Sig", (), {"emit": noop_emit})()
    w.stage_progress = type("Sig", (), {"emit": noop_emit})()
    w._tr = lambda key: key  # Minimal translation function
    return w


class MergePresetSyntheticTest(unittest.TestCase):
    """Test merge preset with synthetic RGBA + RGB images."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create temp directory and synthetic images."""
        cls.tmpdir = tempfile.mkdtemp(prefix="test_merge_")
        
        # Synthetic RGBA foreground (semi-transparent red square)
        # OpenCV uses BGR, so R channel is at index 2
        fg_bgra = np.zeros((128, 128, 4), dtype=np.uint8)
        # Red square: B=50, G=50, R=200 (BGR format)
        fg_bgra[32:96, 32:96, 0] = 50   # B
        fg_bgra[32:96, 32:96, 1] = 50   # G
        fg_bgra[32:96, 32:96, 2] = 200  # R ← красный канал
        fg_bgra[32:96, 32:96, 3] = 200  # Alpha (semi-transparent)
        
        cls.fg_path = Path(cls.tmpdir) / "fg_bgra.png"
        cv2.imwrite(str(cls.fg_path), fg_bgra)

        cls.fg_exr_path = Path(cls.tmpdir) / "fg_bgra.exr"
        save_exr_image(fg_bgra.astype(np.float32) / 255.0, cls.fg_exr_path)
        
        # Synthetic BGR background (green = B=0, G=180, R=0)
        bg_bgr = np.zeros((128, 128, 3), dtype=np.uint8)
        bg_bgr[:, :, 0] = 0      # B
        bg_bgr[:, :, 1] = 180    # G ← зелёный канал
        bg_bgr[:, :, 2] = 0      # R
        
        cls.bg_path = Path(cls.tmpdir) / "bg_bgr.png"
        cv2.imwrite(str(cls.bg_path), bg_bgr)
        
        cls.output_dir = Path(cls.tmpdir) / "output"
        cls.output_dir.mkdir()

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temp files."""
        import shutil
        if Path(cls.tmpdir).exists():
            shutil.rmtree(cls.tmpdir)

    def test_preview_loader_preserves_alpha_for_rgba_images(self) -> None:
        """Merge quick preview must keep alpha from RGBA foreground inputs."""
        frame = load_image_float(self.fg_path)

        self.assertEqual(frame.shape, (128, 128, 4))
        self.assertEqual(frame.dtype, np.float32)
        self.assertGreater(frame[64, 64, 3], 0.7)
        self.assertEqual(float(frame[10, 10, 3]), 0.0)

    def test_invalid_write_output_format_falls_back_to_png(self) -> None:
        fmt = InferenceWorker._resolve_write_output_format(
            {"output_format": "uint8"},
            Path(self.fg_path),
        )
        self.assertEqual(fmt, "png")

    def test_load_video_accepts_single_image_sources(self) -> None:
        worker = _make_worker()
        frames, fps, audio = worker._load_video(str(self.fg_exr_path), Path(self.output_dir))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].shape[0], 128)
        self.assertEqual(frames[0].shape[1], 128)
        self.assertGreater(float(fps), 0.0)
        self.assertEqual(audio, "")

    def test_merge_synthetic_composition(self) -> None:
        """Test that merge correctly composites RGBA fg over RGB bg."""
        worker = _make_worker()
        worker._graph_output_dir = self.output_dir
        
        # Store paths for routing in mock function
        fg_path = str(self.fg_path)
        bg_path = str(self.bg_path)
        
        def mock_load_image_frame(image_path):
            """Route to correct image based on path."""
            if "bg_rgb" in image_path or "bg" in image_path.lower():
                return cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
            return cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
        
        worker._load_image_frame = mock_load_image_frame
        
        # Create graph: source (fg) → merge ← load (bg) → export
        nodes = [
            GraphNode(id="n0", type="source", title="FG", properties={
                "path": fg_path,
                "media_type": "image",
                "enabled": True,
            }),
            GraphNode(id="n1", type="load", title="BG", properties={
                "path": bg_path,
                "media_type": "image",
                "enabled": True,
            }),
            GraphNode(id="n2", type="merge", title="Merge", properties={
                "mode": "over",  # FG поверх BG (правильный режим для композинга)
                "opacity": 1.0,
                "mix": 1.0,
                "mask_enabled": False,
                "alpha_masking": False,
                "fringe": False,
                "enabled": True,
            }),
            GraphNode(id="n3", type="export", title="Export", properties={
                "output_port": "comp",
                "output_format": "uint8",
                "output_dir": str(self.output_dir),
                "enabled": True,
            }),
        ]
        
        edges = [
            GraphEdge(src_id="n0", dst_id="n2", src_port="out", dst_port="fg"),
            GraphEdge(src_id="n1", dst_id="n2", src_port="out", dst_port="bg"),
            GraphEdge(src_id="n2", dst_id="n3", src_port="out", dst_port="in"),
        ]
        
        # Load foreground (RGBA) for source node
        fg_frames = [cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)]
        
        # Execute graph
        outputs = worker._execute_node_graph(nodes, edges, fg_frames)
        
        # Check outputs exist
        self.assertIn("n3", outputs, "Export node missing from outputs")
        export_out = outputs["n3"]
        self.assertIn("in", export_out, "Export input missing")
        
        # export_out["in"] should contain composed frames
        composed_frames = export_out["in"]
        self.assertIsInstance(composed_frames, list)
        self.assertGreater(len(composed_frames), 0, "No composed frames")
        
        result = composed_frames[0]
        if isinstance(result, np.ndarray):
            print(f"Result type: {result.dtype}, shape: {result.shape}")
            print(f"Sample values at center: {result[64, 64]}")
            print(f"Sample values outside: {result[10, 10]}")
            
            # Check dimensions
            self.assertEqual(result.shape[0], 128, f"Height mismatch: {result.shape}")
            self.assertEqual(result.shape[1], 128, f"Width mismatch: {result.shape}")
            
            # Normalize if float [0..1]
            if result.dtype == np.float32 or result.dtype == np.float64:
                result = (result * 255).astype(np.uint8)
            
            # Check composition: red square should be visible (fg over bg)
            # In OPENCV BGR format: B=0, G=50, R=200 at center (red square)
            # Outside: B=0, G=180, R=0 (green background)
            center_pixel = result[64, 64]
            if len(center_pixel) >= 3:
                b, g, r = center_pixel[:3]
                
                # Red channel (index 2 in BGR) should dominate in center
                self.assertGreater(r, g, f"Red should dominate over green in center: BGR={center_pixel}")
                self.assertGreater(r, 100, f"Red too low at center: {center_pixel}")
                
                # Sample outside square (should be green background B=0, G~180, R=0)
                outside_pixel = result[10, 10]
                b_out, g_out, r_out = outside_pixel[:3]
                
                # Green channel should be dominant (background)
                self.assertGreater(g_out, 100, f"Green too low outside: BGR={outside_pixel}")
                # Red should be low
                self.assertLess(r_out, 50, f"Red too high outside: {outside_pixel}")
                
                print(f"✅ Merge composition test passed (mode=over, BGR format)")
                print(f"   Center (fg red over bg green): BGR={center_pixel}")
                print(f"   Outside (bg only green):       BGR={outside_pixel}")
        else:
            self.fail(f"Expected numpy array, got {type(result)}")

    def test_load_image_float_preserves_alpha_for_exr_in_load_video(self):
        """Regression: _load_video early-return for EXR must preserve alpha channel.

        Before fix: _load_image_frame → load_rgb_image → _ensure_uint8_rgb stripped alpha.
        After fix:  load_image_float is used → RGBA preserved for merge compositing.
        """
        # Build a synthetic RGBA EXR: red subject in center, transparent outside
        H, W = 64, 64
        fg_rgba = np.zeros((H, W, 4), dtype=np.float32)
        fg_rgba[16:48, 16:48, 0] = 0.8  # R
        fg_rgba[16:48, 16:48, 3] = 1.0  # alpha = 1 for subject
        # outside subject: alpha = 0

        with tempfile.TemporaryDirectory() as tmp:
            exr_path = Path(tmp) / "fg_test.exr"
            save_exr_image(fg_rgba, exr_path)

            # Call _load_video — should use load_image_float internally
            worker = _make_worker()
            worker._load_image_frame = InferenceWorker._load_image_frame.__get__(
                worker, InferenceWorker
            )

            from app.utils.media import is_supported_image_file
            self.assertTrue(is_supported_image_file(str(exr_path)))

            frames, fps, audio = InferenceWorker._load_video(
                worker, str(exr_path), Path(tmp)
            )
            self.assertEqual(len(frames), 1)
            frame = np.asarray(frames[0])
            # Must have 4 channels (alpha preserved)
            self.assertEqual(frame.ndim, 3, "Expected 3D array")
            self.assertEqual(frame.shape[2], 4, f"Alpha stripped — got shape {frame.shape}")
            # Alpha should vary: 1.0 inside subject, 0.0 outside
            alpha = frame[:, :, 3]
            # Normalize if needed (uint8 would have max 255)
            if alpha.max() > 1.5:
                alpha = alpha / 255.0
            self.assertAlmostEqual(float(alpha[8, 8]), 0.0, delta=0.05,
                                   msg="Alpha outside subject should be 0")
            self.assertAlmostEqual(float(alpha[32, 32]), 1.0, delta=0.05,
                                   msg="Alpha inside subject should be 1")

    def test_merge_over_composite_shows_background_in_transparent_areas(self):
        """Regression: merge 'over' must show bg where fg alpha = 0.

        Before fix: EXR fg was loaded without alpha → fa=1 everywhere → bg hidden.
        After fix:  fg alpha preserved → bg visible outside subject.
        """
        H, W = 64, 64
        # fg: red square 16..48 with alpha mask (transparent outside)
        fg = np.zeros((H, W, 4), dtype=np.float32)
        fg[16:48, 16:48, 0] = 0.9   # R = red
        fg[16:48, 16:48, 3] = 1.0   # alpha = 1 for subject; 0 elsewhere

        # bg: solid blue image (RGB, no alpha)
        bg = np.zeros((H, W, 3), dtype=np.float32)
        bg[:, :, 2] = 0.8   # B = blue everywhere

        result = InferenceWorker._apply_merge_blend(
            fg, bg,
            mode="over",
            opacity=1.0,
            mix=1.0,
            mask_enabled=False,
            mask_channel="auto",
            mask_inject=False,
            invert_mask=False,
            fringe=False,
            alpha_masking=True,
        )
        # result is RGBA float32
        self.assertEqual(result.shape[2], 4, "Result must be RGBA")
        r, g, b, a = result[:,:,0], result[:,:,1], result[:,:,2], result[:,:,3]
        # Inside subject: red channel dominant, alpha=1
        self.assertGreater(float(r[32, 32]), 0.7, "Subject should be red")
        self.assertLess(float(b[32, 32]), 0.2, "Subject should not be blue")
        self.assertAlmostEqual(float(a[32, 32]), 1.0, delta=0.05)
        # Outside subject: bg color (blue), alpha=1 (bg is opaque)
        self.assertLess(float(r[4, 4]), 0.1, "Outside subject should not be red")
        self.assertGreater(float(b[4, 4]), 0.6, "Outside subject should be blue")
        self.assertAlmostEqual(float(a[4, 4]), 1.0, delta=0.05,
                               msg="Output alpha outside subject should be 1 (bg is opaque)")


if __name__ == "__main__":
    unittest.main()
