import os
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("KEYFLOW_DEVICE", "cpu")

from app.workers.cloud_inference_worker import _unpack_cloud_frames_dir, _unpack_cloud_result


def _make_rgba_frame() -> np.ndarray:
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[:, :, 0] = 220
    rgba[:, :, 1] = 40
    rgba[:, :, 2] = 10
    rgba[:, :, 3] = 128
    return rgba


def _write_cloud_zip(zip_path: Path, rgba_frame: np.ndarray) -> None:
    with tempfile.TemporaryDirectory(prefix="kf_cloud_src_") as tmp_dir:
        png_path = Path(tmp_dir) / "0001.png"
        Image.fromarray(rgba_frame, "RGBA").save(png_path)
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.write(png_path, arcname="0001.png")


def _write_frames_dir(frames_dir: Path, rgba_frame: np.ndarray) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba_frame, "RGBA").save(frames_dir / "0001.png")


class CloudUnpackWriteFormatsTest(unittest.TestCase):
    def test_unpack_cloud_result_exr_preserves_alpha_when_enabled(self) -> None:
        rgba = _make_rgba_frame()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            frames_dir = tmp_path / "frames"
            _write_frames_dir(frames_dir, rgba)

            out_path = _unpack_cloud_frames_dir(
                frames_dir,
                export_props={
                    "output_format": "exr",
                    "file_name": "shot",
                    "png_embed_alpha": True,
                },
                fallback_output_dir=tmp_path / "out",
                video_stem="input",
                source_path=Path("input.png"),
                source_node_title="GVM",
                port_label="Alpha",
            )

            self.assertTrue(out_path.exists())
            frame = cv2.imread(str(out_path), cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape[2], 4)

    def test_unpack_cloud_result_jpg_strips_alpha(self) -> None:
        rgba = _make_rgba_frame()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            frames_dir = tmp_path / "frames"
            _write_frames_dir(frames_dir, rgba)

            out_path = _unpack_cloud_frames_dir(
                frames_dir,
                export_props={
                    "output_format": "jpg",
                    "file_name": "shot",
                },
                fallback_output_dir=tmp_path / "out",
                video_stem="input",
                source_path=Path("input.png"),
                source_node_title="GVM",
                port_label="Alpha",
            )

            self.assertTrue(out_path.exists())
            with Image.open(out_path) as image:
                self.assertEqual(image.mode, "RGB")

    def test_unpack_cloud_result_source_webp_falls_back_to_png(self) -> None:
        rgba = _make_rgba_frame()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            frames_dir = tmp_path / "frames"
            _write_frames_dir(frames_dir, rgba)

            out_path = _unpack_cloud_frames_dir(
                frames_dir,
                export_props={
                    "output_format": "source",
                    "file_name": "shot",
                },
                fallback_output_dir=tmp_path / "out",
                video_stem="input",
                source_path=Path("input.webp"),
                source_node_title="GVM",
                port_label="Alpha",
            )

            self.assertTrue(out_path.exists())
            self.assertEqual(out_path.suffix.lower(), ".png")

    def test_unpack_cloud_result_routes_stream_to_matching_export(self) -> None:
        rgba = _make_rgba_frame()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "job_alpha.zip"
            _write_cloud_zip(zip_path, rgba)

            graph_payload = {
                "nodes": [
                    {"id": "n1", "type": "gvm", "title": "GVM"},
                    {
                        "id": "e1",
                        "type": "export",
                        "enabled": True,
                        "properties": {
                            "output_format": "jpg",
                            "file_name": "shot",
                        },
                    },
                ],
                "edges": [
                    {
                        "src": "n1",
                        "src_port": "alpha",
                        "dst": "e1",
                        "dst_port": "in",
                    }
                ],
            }

            results = _unpack_cloud_result(
                zip_path=zip_path,
                graph_payload=graph_payload,
                fallback_output_dir=tmp_path / "out",
                video_stem="input",
                source_path=Path("input.png"),
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["source_port"], "alpha")
            self.assertEqual(results[0]["write_node_id"], "e1")
            result_path = Path(results[0]["result_path"])
            self.assertTrue(result_path.exists())
            self.assertEqual(result_path.suffix.lower(), ".jpg")
            self.assertFalse(zip_path.exists())


if __name__ == "__main__":
    unittest.main()