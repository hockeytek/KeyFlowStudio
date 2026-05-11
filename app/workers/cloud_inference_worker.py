"""Cloud inference worker/controller running in a QThread."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from app.services.cloud_inference_client import (
    CloudConnectionSettings,
    CloudInferenceClient,
    CloudInferenceError,
)
from app.utils.media import load_image_float
from app.utils.write_output import (
    COMPAT_IMAGE_OUTPUT_FORMATS,
    COMPAT_VIDEO_OUTPUT_FORMATS,
    IMAGE_OUTPUT_FORMATS,
    VIDEO_OUTPUT_FORMATS,
    build_video_output_params,
    image_extension_for_format,
    prepare_video_frame,
    resolve_write_output_format,
    save_image_frame,
)
from app.utils.write_paths import build_graph_write_output_dir


def _empty_export_info() -> dict:
    return {
        "export_props": {},
        "export_node_id": "",
        "source_node_title": "",
        "port_label": "",
        "source_port": "",
    }


def _infer_result_stream_from_name(file_name: str) -> str:
    """Infer exported stream name from downloaded server result filename."""
    lowered = str(file_name or "").strip().lower()
    for stream in ("processed", "comp", "fg", "alpha"):
        if f"_{stream}.zip" in lowered or lowered.endswith(f"_{stream}"):
            return stream
    return ""


def _find_export_info(graph_payload: dict, preferred_source_port: str = "") -> dict:
    """Return metadata for the matching enabled Export node in a cloud graph."""
    from app.node_graph.specs import get_node_spec
    nodes_by_id: dict = {n["id"]: n for n in (graph_payload.get("nodes") or []) if isinstance(n, dict)}
    # Graph payload from _build_cloud_graph_payload uses key "edges"; support both for safety.
    connections = graph_payload.get("edges") or graph_payload.get("connections") or []
    preferred = str(preferred_source_port or "").strip().lower()

    candidates: list[dict] = []
    for node in (graph_payload.get("nodes") or []):
        if not isinstance(node, dict):
            continue
        if str(node.get("type", "")).strip().lower() != "export":
            continue
        if not bool(node.get("enabled", True)):
            continue
        export_props = dict(node.get("properties") or {})
        export_node_id = str(node.get("id") or "").strip()
        for conn in connections:
            if not isinstance(conn, dict):
                continue
            dst_id = conn.get("dst") or conn.get("dst_id")
            dst_port = str(conn.get("dst_port", "")).strip().lower()
            if dst_id != node.get("id") or dst_port not in {"", "in"}:
                continue
            src_id = conn.get("src") or conn.get("src_id")
            src_node = nodes_by_id.get(src_id) or {}
            src_port = str(conn.get("src_port") or "").strip().lower()
            node_title = str(src_node.get("title") or src_node.get("type") or "").strip() or "Node"
            node_type = str(src_node.get("type") or "").strip().lower()
            spec = get_node_spec(node_type)
            if not src_port:
                if spec is not None and len(spec.outputs or ()) == 1:
                    src_port = str(spec.outputs[0].name or "out").strip().lower() or "out"
                else:
                    src_port = "out"
            port_label = src_port.replace("_", " ").title()
            if spec is not None:
                for p in (spec.outputs or ()):
                    if str(p.name) == src_port:
                        port_label = str(p.label) if p.label else port_label
                        break
            candidates.append({
                "export_props": export_props,
                "export_node_id": export_node_id,
                "source_node_title": node_title,
                "port_label": port_label,
                "source_port": src_port,
            })

    if preferred:
        for candidate in candidates:
            if str(candidate.get("source_port") or "") == preferred:
                return candidate

    if candidates:
        return candidates[0]

    return _empty_export_info()


def _unpack_cloud_frames_dir(
    frames_dir: Path,
    *,
    export_props: dict,
    fallback_output_dir: Path,
    video_stem: str,
    source_path: Path,
    video_fps: float = 25.0,
    source_node_title: str = "",
    port_label: str = "",
) -> Path:
    """Save already-extracted PNG frames according to Export node settings."""
    output_fmt = resolve_write_output_format(export_props, source_path)
    output_dir_raw = str(export_props.get("output_dir", "")).strip()
    auto_output = bool(export_props.get("auto_output_dir", True))
    file_name_stem = str(export_props.get("file_name", "")).strip() or video_stem
    png_compression = int(export_props.get("png_compression", 6))
    png_bit_depth = int(export_props.get("png_bit_depth", 8))
    jpg_quality = int(export_props.get("jpg_quality", 90))
    embed_alpha = bool(export_props.get("png_embed_alpha", False))

    effective_output_dir = build_graph_write_output_dir(
        fallback_output_dir if auto_output or not output_dir_raw else Path(output_dir_raw),
        source_node_title=source_node_title,
        port_label=port_label,
        stream_label=port_label,
    )
    effective_output_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob("*.png"))
    if output_fmt in COMPAT_VIDEO_OUTPUT_FORMATS:
        with tempfile.TemporaryDirectory(prefix="kf_cloud_encode_") as encode_tmp:
            encode_dir = Path(encode_tmp)
            for i, src in enumerate(frames):
                dst = encode_dir / f"kf_{i:06d}.png"
                shutil.copy2(src, dst)
            pattern = str(encode_dir / "kf_%06d.png")

            codec_key = str(export_props.get("video_codec", "h264")).strip().lower() or "h264"
            crf = int(export_props.get("video_quality", 23))
            preset = str(export_props.get("video_preset", "medium")).strip().lower() or "medium"
            vcodec, output_params = build_video_output_params(codec_key, crf=crf, preset=preset)

            out_video = effective_output_dir / f"{file_name_stem}.{output_fmt}"
            cmd = ["ffmpeg", "-y", "-framerate", str(video_fps), "-i", pattern, "-vcodec", vcodec]
            cmd += output_params
            if not codec_key.startswith("prores"):
                cmd += ["-pix_fmt", "yuv420p"]
            cmd += [str(out_video)]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"ffmpeg encoding failed: {exc.stderr.decode(errors='replace')}"
                ) from exc
            return out_video

    if output_fmt in COMPAT_IMAGE_OUTPUT_FORMATS:
        first_path: Path | None = None
        ext = image_extension_for_format(output_fmt)
        for i, src in enumerate(frames, start=1):
            dst = effective_output_dir / f"{i:04d}{ext}"
            frame = load_image_float(src)
            if output_fmt != "exr" and not np.issubdtype(frame.dtype, np.floating):
                frame = np.asarray(frame)
            save_image_frame(
                frame,
                dst,
                output_fmt=output_fmt,
                png_compression=png_compression,
                png_bit_depth=png_bit_depth,
                jpg_quality=jpg_quality,
                embed_alpha=embed_alpha,
            )
            if first_path is None:
                first_path = dst
        return first_path or effective_output_dir

    first_path: Path | None = None
    for i, src in enumerate(frames, start=1):
        dst = effective_output_dir / f"{i:04d}.png"
        shutil.copy2(src, dst)
        if first_path is None:
            first_path = dst
    return first_path or effective_output_dir


def _unpack_cloud_result(
    zip_path: Path,
    graph_payload: dict,
    fallback_output_dir: Path,
    video_stem: str,
    source_path: Path,
    video_fps: float = 25.0,
) -> list[dict]:
    """Extract a ZIP of server PNG frames and save each stream to matching Export nodes."""
    with tempfile.TemporaryDirectory(prefix="kf_cloud_") as tmp_str:
        tmp_dir = Path(tmp_str)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        stream_dirs: dict[str, Path] = {}
        flat_frames = sorted(tmp_dir.glob("*.png"))
        if flat_frames:
            inferred_stream = _infer_result_stream_from_name(zip_path.name)
            stream_dirs[inferred_stream or "out"] = tmp_dir
        else:
            for child in tmp_dir.iterdir():
                if child.is_dir() and any(child.glob("*.png")):
                    stream_dirs[child.name.strip().lower() or "out"] = child

        if not stream_dirs:
            zip_path.unlink(missing_ok=True)
            return []

        results: list[dict] = []
        stream_order = {"processed": 0, "fg": 1, "alpha": 2, "comp": 3, "out": 99}
        for stream_name, frames_dir in sorted(stream_dirs.items(), key=lambda item: stream_order.get(item[0], 50)):
            export_info = _find_export_info(graph_payload, preferred_source_port=stream_name)
            result_path = _unpack_cloud_frames_dir(
                frames_dir,
                export_props=dict(export_info.get("export_props") or {}),
                fallback_output_dir=fallback_output_dir,
                video_stem=video_stem,
                source_path=source_path,
                video_fps=video_fps,
                source_node_title=str(export_info.get("source_node_title") or ""),
                port_label=str(export_info.get("port_label") or ""),
            )
            results.append(
                {
                    "result_path": str(result_path),
                    "write_node_id": str(export_info.get("export_node_id") or ""),
                    "source_port": stream_name,
                }
            )

        zip_path.unlink(missing_ok=True)
        return results


class _CloudInferenceTask(QObject):
    """Background task that communicates with the EC2 worker API."""

    stage_progress = Signal(int, str)
    log_message = Signal(str)
    finished = Signal(dict)
    error = Signal(str)
    remote_health = Signal(dict)
    worker_connected = Signal(str)  # emitted with base_url once /health succeeds

    def __init__(self, translate: Callable[[str], str]) -> None:
        super().__init__()
        self._tr = translate
        self._cancel_flag = threading.Event()
        self._active_job_id = ""

    def set_cancel(self) -> None:
        self._cancel_flag.set()

    def _cleanup_server_job_files(
        self,
        client: CloudInferenceClient,
        base_url: str,
        job_id: str,
        *,
        reason: str,
    ) -> None:
        if not base_url or not job_id:
            return
        client.cleanup_job_files(base_url, job_id)
        self.log_message.emit(f"Server upload/output files cleaned up ({reason})")

    def run_job(
        self,
        video_path: str,
        mask_path: str,
        output_dir: str,
        cloud_settings: dict,
        params: dict,
    ) -> None:
        self._cancel_flag.clear()
        client = CloudInferenceClient()
        base_url = ""
        job_id = ""
        final_state = ""
        cleaned_server_files = False

        try:
            self.stage_progress.emit(2, self._tr("status_start"))

            conn = CloudConnectionSettings(
                instance_id=str(cloud_settings.get("instance_id") or ""),
                region=str(cloud_settings.get("region") or "eu-west-1"),
                aws_profile=str(cloud_settings.get("aws_profile") or ""),
                api_port=int(cloud_settings.get("api_port") or 8080),
                api_host=str(cloud_settings.get("api_host") or ""),
            )

            self.stage_progress.emit(6, self._tr("cloud_worker_status_connecting"))
            base_url = client.resolve_worker_base_url(conn)
            health = client.check_health(base_url)
            self.remote_health.emit(health)
            self.worker_connected.emit(base_url)
            self.log_message.emit(
                f"Cloud worker: {base_url} | device={health.get('device', 'unknown')}"
            )

            if self._cancel_flag.is_set():
                self.finished.emit({"cancelled": True, "result_path": "", "job_id": ""})
                return

            # Background thread polls EC2 health every 4 s and emits stats to UI.
            _health_stop = threading.Event()

            def _poll_health() -> None:
                while not _health_stop.wait(4.0):
                    try:
                        self.remote_health.emit(client.check_health(base_url))
                    except Exception:
                        pass

            _health_thread = threading.Thread(target=_poll_health, daemon=True)
            _health_thread.start()

            graph_payload = params.get("node_graph")
            is_node_graph_job = isinstance(graph_payload, dict) and bool(graph_payload)
            upload_status_key = "cloud_worker_status_uploading_graph" if is_node_graph_job else "cloud_worker_status_uploading"
            self.stage_progress.emit(12, self._tr(upload_status_key))
            if is_node_graph_job:
                job_id = client.submit_node_graph_job(
                    base_url,
                    video_path=video_path,
                    graph_payload=graph_payload,
                    frame_start=int(params.get("frame_start") or 0),
                    frame_end=int(params.get("frame_end") or 0),
                    source_is_sequence=bool(params.get("source_is_sequence", False)),
                )
            else:
                job_id = client.submit_matanyone2_job(
                    base_url,
                    video_path=video_path,
                    mask_path=mask_path,
                    n_warmup=int(params.get("n_warmup") or 10),
                    r_erode=int(params.get("r_erode") or 0),
                    r_dilate=int(params.get("r_dilate") or 0),
                )
            self._active_job_id = job_id
            self.log_message.emit(f"Cloud job submitted: {job_id}")

            _min_progress_floor: list[int] = [18]

            def _on_progress(status: dict) -> None:
                raw_progress = int(status.get("progress") or 0)
                # Never go backward: clamp to at least what we already showed.
                progress = max(_min_progress_floor[0], min(99, raw_progress))
                _min_progress_floor[0] = progress
                stage = str(status.get("stage") or "").strip()
                state = str(status.get("status") or "").strip()
                text = self._format_status_text(state, stage)
                self.stage_progress.emit(progress, text)

            self.stage_progress.emit(18, self._tr("cloud_worker_status_processing"))
            final_status = client.wait_for_job(
                base_url,
                job_id,
                poll_interval_sec=0.25,
                on_progress=_on_progress,
                is_cancel_requested=self._cancel_flag.is_set,
            )

            if self._cancel_flag.is_set():
                client.cancel_job(base_url, job_id)
                self._cleanup_server_job_files(client, base_url, job_id, reason="cancelled")
                cleaned_server_files = True
                self.finished.emit({"cancelled": True, "result_path": "", "job_id": job_id})
                return

            state = str(final_status.get("status") or "").strip().lower()
            final_state = state
            if state == "failed":
                self._cleanup_server_job_files(client, base_url, job_id, reason="failed")
                cleaned_server_files = True
                details = str(final_status.get("error") or "Cloud job failed")
                raise CloudInferenceError(details)
            if state == "cancelled":
                self._cleanup_server_job_files(client, base_url, job_id, reason="cancelled")
                cleaned_server_files = True
                self.finished.emit({"cancelled": True, "result_path": "", "job_id": job_id})
                return
            if state != "done":
                raise CloudInferenceError(f"Unexpected cloud job status: {state or 'unknown'}")

            self.stage_progress.emit(97, self._tr("cloud_worker_status_downloading"))
            fallback_name = (
                f"{Path(video_path).stem}_node_graph_output.zip"
                if is_node_graph_job
                else f"{Path(video_path).stem}_alpha.mp4"
            )
            result_path = client.download_result(
                base_url,
                job_id,
                output_dir=output_dir,
                fallback_name=fallback_name,
            )

            # Download succeeded — clean up server files to free disk space.
            self._cleanup_server_job_files(client, base_url, job_id, reason="downloaded")
            cleaned_server_files = True

            # Unpack node-graph ZIP result to the path/format specified by Export node.
            write_results: list[dict] = []
            if is_node_graph_job and str(result_path).endswith(".zip"):
                self.stage_progress.emit(98, self._tr("cloud_worker_status_downloading"))
                graph_payload = params.get("node_graph") or {}
                video_fps = float(params.get("source_fps") or 25.0)
                write_results = _unpack_cloud_result(
                    zip_path=result_path,
                    graph_payload=graph_payload,
                    fallback_output_dir=Path(output_dir),
                    video_stem=Path(video_path).stem,
                    source_path=Path(video_path),
                    video_fps=video_fps,
                )
                if write_results:
                    result_path = str(write_results[0].get("result_path") or result_path)
                    for item in write_results:
                        self.log_message.emit(
                            f"Result unpacked: {item.get('source_port') or 'out'} -> {item.get('result_path') or ''}"
                        )
                export_info = _empty_export_info()
            else:
                export_info = _empty_export_info()

            self.stage_progress.emit(100, self._tr("status_done"))
            self.finished.emit(
                {
                    "cancelled": False,
                    "result_path": str(result_path),
                    "write_node_id": str(export_info.get("export_node_id") or ""),
                    "write_results": write_results,
                    "job_id": job_id,
                    "base_url": base_url,
                }
            )
        except Exception as exc:
            if (
                not cleaned_server_files
                and job_id
                and base_url
                and final_state in {"failed", "cancelled"}
            ):
                self._cleanup_server_job_files(client, base_url, job_id, reason=final_state)
            self.error.emit(str(exc))
        finally:
            try:
                _health_stop.set()
            except NameError:
                pass
            client.close()
            self._active_job_id = ""

    # Mapping from EC2 server-side stage strings to i18n keys (translated client-side).
    _EC2_STAGE_KEYS: dict[str, str] = {
        "loading graph":        "cloud_stage_loading_graph",
        "extracting sequence":  "cloud_stage_extracting_sequence",
        "extracting frames":    "cloud_stage_extracting_frames",
        "loading model":        "cloud_stage_loading_model",
        "gvm inference":        "cloud_stage_gvm_inference",
        "loading corridorkey":  "cloud_stage_loading_corridorkey",
        "corridorkey inference": "cloud_stage_corridorkey_inference",
        "packing result":       "cloud_stage_packing_result",
        "loading video":        "cloud_stage_loading_video",
        "loading mask":         "cloud_stage_loading_mask",
        "inference":            "cloud_stage_inference",
        "saving result":        "cloud_stage_saving_result",
    }

    def _format_status_text(self, state: str, stage: str) -> str:
        normalized = state.strip().lower()
        stage_text = stage.strip()
        if normalized == "running":
            if stage_text:
                i18n_key = self._EC2_STAGE_KEYS.get(stage_text.lower())
                return self._tr(i18n_key) if i18n_key else f"☁ {stage_text}"
            return self._tr("cloud_worker_status_processing")
        if normalized == "queued":
            return self._tr("cloud_worker_status_queued")
        if normalized == "done":
            return self._tr("status_done")
        if normalized == "failed":
            return self._tr("status_error")
        if normalized == "cancelled":
            return self._tr("status_cancel")
        return stage_text or self._tr("cloud_worker_status_processing")


class CloudInferenceController(QObject):
    """High-level cloud inference controller with lifecycle similar to matting."""

    stage_progress = Signal(int, str)
    log_message = Signal(str)
    processing_started = Signal()
    processing_finished = Signal(dict)
    error_occurred = Signal(str)
    controls_busy_changed = Signal(bool)
    remote_health = Signal(dict)
    worker_connected = Signal(str)  # base_url once connected
    _start_job_requested = Signal(object, object, object, object, object)

    def __init__(self, translate: Callable[[str], str], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._tr = translate
        self._worker_thread: QThread | None = None
        self._worker: _CloudInferenceTask | None = None

    @property
    def is_active(self) -> bool:
        return self._worker is not None

    def set_translator(self, translate: Callable[[str], str]) -> None:
        self._tr = translate

    def start(
        self,
        *,
        video_path: str,
        mask_path: str,
        output_dir: str,
        cloud_settings: dict,
        params: dict,
    ) -> None:
        if self._worker is not None:
            return

        self._worker_thread = QThread(self)
        self._worker = _CloudInferenceTask(self._tr)
        self._worker.moveToThread(self._worker_thread)

        self._start_job_requested.connect(self._worker.run_job)
        self._worker.stage_progress.connect(self.stage_progress)
        self._worker.log_message.connect(self.log_message)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.remote_health.connect(self.remote_health)
        self._worker.worker_connected.connect(self.worker_connected)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)

        self.controls_busy_changed.emit(True)
        self.processing_started.emit()
        self._worker_thread.start()
        self._start_job_requested.emit(video_path, mask_path, output_dir, cloud_settings, params)

    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.set_cancel()

    def cleanup(self) -> None:
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait()
        if self._worker is not None:
            try:
                self._start_job_requested.disconnect(self._worker.run_job)
            except (TypeError, RuntimeError):
                pass
        self._worker = None
        self._worker_thread = None

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.set_cancel()
            time.sleep(0.05)
        self.cleanup()

    def _on_finished(self, result: dict) -> None:
        self.cleanup()
        self.controls_busy_changed.emit(False)
        self.processing_finished.emit(result)

    def _on_error(self, error_message: str) -> None:
        self.cleanup()
        self.controls_busy_changed.emit(False)
        self.error_occurred.emit(error_message)
