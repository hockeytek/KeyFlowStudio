"""File-side helpers for graph Write stream plans."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.constants import DEFAULT_VIDEO_CODEC, PRORES_PROFILES
from app.utils.write_output import (
    COMPAT_VIDEO_OUTPUT_FORMATS,
    build_video_output_params,
    prepare_video_frame,
    resolve_write_output_format,
    save_image_frame,
)
from app.utils.write_paths import build_keyflow_output_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphWriteFrameResult:
    preview_path: str
    saved_path: Path | None


def initialize_graph_write_plan(
    plan: dict,
    *,
    source_path: str,
    graph_fps: float,
    is_video: bool,
) -> None:
    if plan.get("initialized"):
        return

    source = Path(source_path)
    write_cfg = dict(plan.get("write_cfg") or {})
    output_fmt = resolve_write_output_format(write_cfg, source)
    stream_label = str(plan.get("stream_label") or "fg")
    out_dir = Path(write_cfg.get("output_dir", "") or str(build_keyflow_output_dir(source, stream_label)))
    stem = str(write_cfg.get("file_name", "")).strip() or source.stem or "result"
    plan["output_fmt"] = output_fmt
    plan["out_dir"] = out_dir
    plan["stem"] = stem
    plan["is_video"] = bool(is_video)
    plan["png_compression"] = int(write_cfg.get("png_compression", 6))
    plan["png_bit_depth"] = int(write_cfg.get("png_bit_depth", 8))
    plan["jpg_quality"] = int(write_cfg.get("jpg_quality", 90))
    plan["embed_alpha"] = bool(write_cfg.get("png_embed_alpha", False))
    plan["created_paths"] = set()
    plan["video_codec"] = str(write_cfg.get("video_codec", DEFAULT_VIDEO_CODEC)).strip().lower() or DEFAULT_VIDEO_CODEC
    plan["video_quality"] = int(write_cfg.get("video_quality", 23))
    plan["video_preset"] = str(write_cfg.get("video_preset", "medium")).strip().lower() or "medium"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_exts = set(COMPAT_VIDEO_OUTPUT_FORMATS)
    if output_fmt in video_exts:
        import imageio

        video_ext = f".{output_fmt}"
        plan["tmp_path"] = out_dir / f"{stem}_tmp{video_ext}"
        plan["final_path"] = out_dir / f"{stem}{video_ext}"
        plan["created_paths"].add(plan["tmp_path"])
        codec = plan["video_codec"]
        ffmpeg_codec, output_params = build_video_output_params(
            codec,
            crf=int(plan["video_quality"]),
            preset=str(plan["video_preset"]),
        )
        if codec in PRORES_PROFILES:
            writer = imageio.get_writer(
                str(plan["tmp_path"]),
                fps=graph_fps,
                codec=ffmpeg_codec,
                macro_block_size=1,
                output_params=output_params,
            )
        else:
            writer = imageio.get_writer(
                str(plan["tmp_path"]),
                fps=graph_fps,
                codec=ffmpeg_codec,
                macro_block_size=1,
                output_params=output_params,
            )
        plan["writer"] = writer
    else:
        ext = ".jpg" if output_fmt in {"jpg", "jpeg"} else f".{output_fmt}"
        plan["img_ext"] = ext
        if is_video:
            plan["first_path"] = out_dir / f"0001{ext}"
        else:
            plan["first_path"] = out_dir / f"{stem}{ext}"

    plan["initialized"] = True


def write_graph_plan_frame(
    plan: dict,
    frame,
    frame_index_0_based: int,
    *,
    is_video: bool,
) -> GraphWriteFrameResult:
    output_fmt = str(plan.get("output_fmt") or "png")
    video_exts = set(COMPAT_VIDEO_OUTPUT_FORMATS)
    if output_fmt in video_exts:
        writer = plan.get("writer")
        if writer is not None:
            frame_u8 = prepare_video_frame(frame, str(plan.get("video_codec") or DEFAULT_VIDEO_CODEC))
            writer.append_data(frame_u8)
        return GraphWriteFrameResult(str(plan.get("final_path") or ""), None)

    out_dir = Path(plan["out_dir"])
    if is_video:
        out_path = out_dir / f"{frame_index_0_based:04d}{plan['img_ext']}"
    else:
        out_path = Path(plan["first_path"])
    save_image_frame(
        frame,
        out_path,
        output_fmt=output_fmt,
        png_compression=int(plan.get("png_compression", 6)),
        png_bit_depth=int(plan.get("png_bit_depth", 8)),
        jpg_quality=int(plan.get("jpg_quality", 90)),
        embed_alpha=bool(plan.get("embed_alpha", False)),
    )
    plan["created_paths"].add(out_path)
    return GraphWriteFrameResult(str(out_path), Path(plan["first_path"]))


def finalize_graph_write_plan(
    plan: dict,
    *,
    keep_outputs: bool,
    audio_path: str,
    mux_audio: Callable[[str, str, Path], str],
) -> Path | None:
    output_path: Path | None = None
    writer = plan.get("writer")
    if writer is not None:
        try:
            writer.close()
        except Exception as writer_close_exc:
            logger.warning("Failed to close stream writer: %s", writer_close_exc)
        tmp_path = plan.get("tmp_path")
        final_path = plan.get("final_path")
        if isinstance(tmp_path, Path) and isinstance(final_path, Path) and tmp_path.exists():
            if keep_outputs:
                if audio_path:
                    muxed = Path(mux_audio(str(tmp_path), audio_path, final_path))
                    if muxed == tmp_path and tmp_path.exists():
                        tmp_path.replace(final_path)
                else:
                    tmp_path.replace(final_path)
                output_path = final_path
                plan.get("created_paths", set()).add(final_path)
            else:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    if not keep_outputs:
        for path in list(plan.get("created_paths", set())):
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass
    plan["closed"] = True
    return output_path