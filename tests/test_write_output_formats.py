import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("KEYFLOW_DEVICE", "cpu")

from app.utils.write_output import (
    build_video_output_params,
    prepare_video_frame,
    resolve_write_output_format,
    save_image_frame,
)


class WriteOutputFormatsTest(unittest.TestCase):
    def test_resolve_source_output_format_falls_back_from_legacy_image_to_png(self) -> None:
        fmt = resolve_write_output_format({"output_format": "source"}, Path("clip.webp"))
        self.assertEqual(fmt, "png")

    def test_save_image_frame_jpeg_strips_alpha(self) -> None:
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[:, :, 0] = 255
        rgba[:, :, 3] = 128

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "frame.jpg"
            save_image_frame(
                rgba,
                out_path,
                output_fmt="jpg",
                png_compression=6,
                png_bit_depth=8,
                jpg_quality=90,
            )

            self.assertTrue(out_path.exists())
            with Image.open(out_path) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (4, 4))

    def test_save_image_frame_png_strips_alpha_when_flag_disabled(self) -> None:
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[:, :, 1] = 255
        rgba[:, :, 3] = 128

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "frame.png"
            save_image_frame(
                rgba,
                out_path,
                output_fmt="png",
                png_compression=6,
                png_bit_depth=8,
                jpg_quality=90,
                embed_alpha=False,
            )

            with Image.open(out_path) as image:
                self.assertEqual(image.mode, "RGB")

    def test_save_image_frame_png_preserves_alpha_when_flag_enabled(self) -> None:
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[:, :, 1] = 255
        rgba[:, :, 3] = 128

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "frame.png"
            save_image_frame(
                rgba,
                out_path,
                output_fmt="png",
                png_compression=6,
                png_bit_depth=8,
                jpg_quality=90,
                embed_alpha=True,
            )

            with Image.open(out_path) as image:
                self.assertEqual(image.mode, "RGBA")

    def test_save_image_frame_tiff_16bit_still_works_as_legacy_compat(self) -> None:
        gray = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "frame.tiff"
            save_image_frame(
                gray,
                out_path,
                output_fmt="tiff",
                png_compression=6,
                png_bit_depth=16,
                jpg_quality=90,
            )

            self.assertTrue(out_path.exists())
            loaded = cv2.imread(str(out_path), cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.dtype, np.uint16)
            self.assertEqual(loaded.shape[:2], (4, 4))

    def test_build_video_output_params_prores4444_requests_alpha_pix_fmt(self) -> None:
        codec, params = build_video_output_params("prores4444")
        self.assertEqual(codec, "prores_ks")
        self.assertIn("yuva444p10le", params)

    def test_prepare_video_frame_handles_alpha_by_codec(self) -> None:
        rgba = np.zeros((3, 3, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255

        h264_frame = prepare_video_frame(rgba, "h264")
        prores_frame = prepare_video_frame(rgba, "prores4444")

        self.assertEqual(h264_frame.shape, (3, 3, 3))
        self.assertEqual(prores_frame.shape, (3, 3, 4))


if __name__ == "__main__":
    unittest.main()