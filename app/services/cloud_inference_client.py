"""HTTP client for KeyFlow cloud inference worker."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import time
from urllib.parse import urlsplit
import zipfile

import requests
from requests import RequestException

from app.cloud_manager import InstanceState, get_instance_state


class CloudInferenceError(RuntimeError):
    """Cloud inference client error."""


@dataclass(slots=True)
class CloudConnectionSettings:
    """Connection settings needed to locate cloud worker on EC2."""

    instance_id: str
    region: str
    aws_profile: str
    api_port: int = 8080
    scheme: str = "http"
    api_host: str = ""   # direct IP/hostname override; skips AWS lookup when non-empty


class CloudInferenceClient:
    """Small synchronous client for the EC2 worker API."""

    _ACTIVE_JOB_STATES = {"queued", "running"}

    def __init__(self, *, request_timeout_sec: float = 30.0) -> None:
        self._session = requests.Session()
        self._timeout = float(request_timeout_sec)

    def close(self) -> None:
        self._session.close()

    def resolve_worker_base_url(self, settings: CloudConnectionSettings) -> str:
        # Fast path: user provided a direct IP/host — use it without AWS API call.
        api_host = str(settings.api_host or "").strip()
        if api_host:
            return f"{settings.scheme}://{api_host}:{int(settings.api_port)}"

        # Slow path: resolve IP dynamically from AWS instance ID.
        instance_id = str(settings.instance_id or "").strip()
        if not instance_id:
            raise CloudInferenceError(
                "Cloud host is not configured. "
                "Enter the server IP in cloud settings or set an EC2 instance ID."
            )

        state, public_ip = get_instance_state(instance_id, settings.region, settings.aws_profile)
        if state == InstanceState.ERROR:
            details = str(public_ip or "unknown error")
            raise CloudInferenceError(f"Failed to query EC2 state: {details}")
        if state != InstanceState.RUNNING:
            raise CloudInferenceError(f"EC2 instance is not running (state: {state.value})")

        ip = str(public_ip or "").strip()
        if not ip:
            raise CloudInferenceError("EC2 instance has no public IP")

        return f"{settings.scheme}://{ip}:{int(settings.api_port)}"

    def check_health(self, base_url: str) -> dict:
        response = self._safe_request("get", f"{base_url}/health", timeout=self._timeout)
        self._raise_for_http(response)
        return self._read_json(response)

    def submit_matanyone2_job(
        self,
        base_url: str,
        *,
        video_path: str,
        mask_path: str,
        n_warmup: int,
        r_erode: int,
        r_dilate: int,
    ) -> str:
        video_file = Path(video_path)
        mask_file = Path(mask_path)
        if not video_file.exists():
            raise CloudInferenceError(f"Input video does not exist: {video_file}")
        if not mask_file.exists():
            raise CloudInferenceError(f"Input mask does not exist: {mask_file}")

        with video_file.open("rb") as vf, mask_file.open("rb") as mf:
            response = self._safe_request(
                "post",
                f"{base_url}/jobs/matanyone2",
                files={
                    "video": (video_file.name, vf, "application/octet-stream"),
                    "mask": (mask_file.name, mf, "application/octet-stream"),
                },
                data={
                    "n_warmup": int(n_warmup),
                    "r_erode": int(r_erode),
                    "r_dilate": int(r_dilate),
                },
                timeout=max(self._timeout, 180.0),
            )

        self._raise_for_http(response)
        payload = self._read_json(response)
        job_id = str(payload.get("job_id") or "").strip()
        if not job_id:
            raise CloudInferenceError(f"Invalid submit response: {payload}")
        return job_id

    def submit_node_graph_job(
        self,
        base_url: str,
        *,
        video_path: str,
        graph_payload: dict,
        frame_start: int = 0,
        frame_end: int = 0,
        source_is_sequence: bool = False,
    ) -> str:
        video_file = Path(video_path)
        if not video_file.exists():
            raise CloudInferenceError(f"Input video does not exist: {video_file}")
        if not isinstance(graph_payload, dict) or not graph_payload:
            raise CloudInferenceError("Graph payload is empty")

        graph_json = json.dumps(graph_payload, ensure_ascii=False)

        extra_data: dict = {"graph_json": graph_json}
        if frame_start > 0 or frame_end > 0:
            extra_data["frame_start"] = str(frame_start)
            extra_data["frame_end"] = str(frame_end)

        if source_is_sequence:
            # For image sequences: upload only the selected frame range as a ZIP.
            from app.utils.media import resolve_numbered_image_sequence

            seq_paths = resolve_numbered_image_sequence(video_file)
            end = frame_end if frame_end > 0 else len(seq_paths)
            selected = seq_paths[frame_start:end]
            if not selected:
                raise CloudInferenceError(
                    f"No frames found in range [{frame_start}:{end}] "
                    f"for sequence: {video_file}"
                )

            with tempfile.NamedTemporaryFile(
                suffix=".zip", prefix="kf_seq_", delete=False
            ) as tmp:
                tmp_zip_path = Path(tmp.name)

            try:
                with zipfile.ZipFile(tmp_zip_path, "w", zipfile.ZIP_STORED) as zf:
                    for p in selected:
                        zf.write(p, arcname=Path(p).name)

                with tmp_zip_path.open("rb") as zf_file:
                    response = self._safe_request(
                        "post",
                        f"{base_url}/jobs/node-graph",
                        files={
                            "video": ("sequence.zip", zf_file, "application/zip"),
                        },
                        data=extra_data,
                        timeout=max(self._timeout, 180.0),
                    )
            finally:
                try:
                    tmp_zip_path.unlink()
                except OSError:
                    pass
        else:
            with video_file.open("rb") as vf:
                response = self._safe_request(
                    "post",
                    f"{base_url}/jobs/node-graph",
                    files={
                        "video": (video_file.name, vf, "application/octet-stream"),
                    },
                    data=extra_data,
                    timeout=max(self._timeout, 180.0),
                )

        self._raise_for_http(response)
        payload = self._read_json(response)
        job_id = str(payload.get("job_id") or "").strip()
        if not job_id:
            raise CloudInferenceError(f"Invalid graph submit response: {payload}")
        return job_id

    def get_job_status(self, base_url: str, job_id: str) -> dict:
        response = self._safe_request("get", f"{base_url}/jobs/{job_id}", timeout=self._timeout)
        self._raise_for_http(response)
        return self._read_json(response)

    def wait_for_job(
        self,
        base_url: str,
        job_id: str,
        *,
        poll_interval_sec: float = 1.0,
        on_progress=None,
        is_cancel_requested=None,
    ) -> dict:
        stream_url = f"{base_url}/jobs/{job_id}/stream"
        try:
            with self._session.get(
                stream_url,
                stream=True,
                timeout=(min(self._timeout, 10.0), max(300.0, self._timeout)),
            ) as response:
                self._raise_for_http(response)
                for raw_line in response.iter_lines(decode_unicode=True):
                    if callable(is_cancel_requested) and is_cancel_requested():
                        return self.get_job_status(base_url, job_id)
                    if not raw_line:
                        continue
                    line = str(raw_line).strip()
                    if not line.startswith("data:"):
                        continue
                    payload_text = line[5:].strip()
                    if not payload_text:
                        continue
                    try:
                        status = json.loads(payload_text)
                    except Exception:
                        continue

                    if callable(on_progress):
                        on_progress(status)

                    state = str(status.get("status") or "").strip().lower()
                    if state in {"done", "failed", "cancelled"}:
                        return status
        except Exception:
            pass

        while True:
            status = self.get_job_status(base_url, job_id)
            if callable(on_progress):
                on_progress(status)

            state = str(status.get("status") or "").strip().lower()
            if state in {"done", "failed", "cancelled"}:
                return status

            if callable(is_cancel_requested) and is_cancel_requested():
                return status

            time.sleep(max(0.2, float(poll_interval_sec)))

    def cancel_job(self, base_url: str, job_id: str) -> None:
        response = self._safe_request("delete", f"{base_url}/jobs/{job_id}", timeout=self._timeout)
        if response.status_code in {200, 202, 404}:
            return
        self._raise_for_http(response)

    def download_result(
        self,
        base_url: str,
        job_id: str,
        *,
        output_dir: str,
        fallback_name: str = "cloud_alpha.mp4",
    ) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        response = self._safe_request(
            "get",
            f"{base_url}/jobs/{job_id}/result",
            timeout=max(self._timeout, 180.0),
            stream=True,
        )
        self._raise_for_http(response)

        file_name = self._resolve_download_name(response, fallback_name=fallback_name)
        destination = out_dir / file_name
        with destination.open("wb") as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fd.write(chunk)
        return destination

    def cleanup_job_files(self, base_url: str, job_id: str) -> None:
        """Delete uploaded input and output files on the server after a successful download.

        Non-fatal: errors are silently ignored so that a cleanup failure never
        causes the local job to be reported as failed.
        """
        try:
            response = self._safe_request(
                "delete",
                f"{base_url}/jobs/{job_id}/files",
                timeout=self._timeout,
            )
            # 404 is acceptable — job may have been cleaned up already or never existed
            if response.status_code not in {200, 202, 404}:
                self._raise_for_http(response)
        except Exception:
            pass  # cleanup failure must not fail the calling workflow

    def _wait_for_worker_idle(self, url: str, *, timeout_sec: float = 20 * 60.0) -> bool:
        parts = urlsplit(url)
        base_url = f"{parts.scheme}://{parts.netloc}"
        jobs_url = f"{base_url}/jobs"
        health_url = f"{base_url}/health"
        deadline = time.time() + max(5.0, float(timeout_sec))

        while time.time() < deadline:
            try:
                jobs_resp = self._session.get(jobs_url, timeout=min(self._timeout, 8.0))
                if 200 <= jobs_resp.status_code < 300:
                    payload = jobs_resp.json() if jobs_resp.content else {}
                    jobs = payload.get("jobs") if isinstance(payload, dict) else None
                    if isinstance(jobs, list):
                        active = [
                            item for item in jobs
                            if isinstance(item, dict)
                            and str(item.get("status") or "").strip().lower() in self._ACTIVE_JOB_STATES
                        ]
                        if not active:
                            return True
                        time.sleep(5.0)
                        continue
            except Exception:
                pass

            try:
                health_resp = self._session.get(health_url, timeout=min(self._timeout, 8.0))
                if 200 <= health_resp.status_code < 300:
                    health = health_resp.json() if health_resp.content else {}
                    if isinstance(health, dict):
                        free_gb = float(health.get("gpu_free_gb") or 0.0)
                        total_gb = float(health.get("gpu_total_gb") or 0.0)
                        if total_gb <= 0.0 or free_gb >= 1.0:
                            return True
            except Exception:
                pass

            time.sleep(5.0)

        return False

    @staticmethod
    def _rewind_request_files(kwargs: dict) -> None:
        files = kwargs.get("files")
        if not isinstance(files, dict):
            return
        for item in files.values():
            if not isinstance(item, tuple) or len(item) < 2:
                continue
            file_obj = item[1]
            if hasattr(file_obj, "seek"):
                try:
                    file_obj.seek(0)
                except Exception:
                    pass

    @staticmethod
    def _read_json(response: requests.Response) -> dict:
        try:
            payload = response.json()
        except Exception as exc:
            text = response.text[:600].strip()
            raise CloudInferenceError(f"Invalid JSON response: {text}") from exc
        if not isinstance(payload, dict):
            raise CloudInferenceError(f"Unexpected response payload: {payload}")
        return payload

    @staticmethod
    def _resolve_download_name(response: requests.Response, *, fallback_name: str) -> str:
        cd = str(response.headers.get("content-disposition") or "")
        marker = "filename="
        if marker in cd.lower():
            idx = cd.lower().find(marker)
            raw_name = cd[idx + len(marker):].strip().strip('"')
            safe = Path(raw_name).name.strip()
            if safe:
                return safe
        return Path(fallback_name).name

    @staticmethod
    def _raise_for_http(response: requests.Response) -> None:
        if 200 <= response.status_code < 300:
            return
        body = response.text[:800].strip()
        raise CloudInferenceError(f"HTTP {response.status_code}: {body}")

    def _safe_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Wrap requests errors with user-facing cloud diagnostics."""
        method_u = str(method or "GET").upper()
        retries = 2 if method_u == "GET" else 3
        last_exc: RequestException | None = None

        for attempt in range(retries):
            try:
                return self._session.request(method=method_u, url=url, **kwargs)
            except RequestException as exc:
                last_exc = exc
                message = str(exc).strip() or exc.__class__.__name__
                is_remote_disconnect = (
                    "RemoteDisconnected" in message
                    or "Connection aborted" in message
                )
                if is_remote_disconnect and attempt + 1 < retries:
                    if self._wait_for_worker_idle(url):
                        self._rewind_request_files(kwargs)
                        continue
                if attempt + 1 < retries:
                    time.sleep(0.35)
                    continue

        exc = last_exc or RequestException("Unknown request error")
        message = str(exc).strip() or exc.__class__.__name__

        # If the socket was closed while submitting a job, probe /health and
        # surface "busy" instead of low-level RemoteDisconnected details.
        if "RemoteDisconnected" in message or "Connection aborted" in message:
            try:
                parts = urlsplit(url)
                health_url = f"{parts.scheme}://{parts.netloc}/health"
                health_resp = self._session.get(health_url, timeout=min(self._timeout, 8.0))
                if 200 <= health_resp.status_code < 300:
                    health = health_resp.json() if health_resp.content else {}
                    if isinstance(health, dict):
                        free_gb = float(health.get("gpu_free_gb") or 0.0)
                        if free_gb < 1.0:
                            raise CloudInferenceError(
                                "Cloud worker is busy (GPU memory is saturated by another job). "
                                "Wait until the current job finishes and retry."
                            ) from exc
            except CloudInferenceError:
                raise
            except Exception:
                pass

        raise CloudInferenceError(
            "Cloud connection failed: "
            f"{message}. "
            "If a long GPU job is already running, wait until it finishes and retry."
        ) from exc
