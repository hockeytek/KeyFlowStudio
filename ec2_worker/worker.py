"""
KeyFlow Studio — EC2 GPU Worker
================================
FastAPI headless inference service for MatAnyone2 / BiRefNet / CorridorKey.

Deploy to EC2 (g5.2xlarge / any CUDA instance):
    source ~/keyflow-venv/bin/activate
    uvicorn worker:app --host 0.0.0.0 --port 8080 --workers 1

Environment variables (optional):
    KEYFLOW_MODELS_DIR  — path to model weights (default: ~/.local/share/com.keyflow.studio/models)
    KEYFLOW_DEVICE      — cuda / cpu (default: auto-detect)
    KEYFLOW_UPLOAD_DIR  — temp upload dir (default: /tmp/keyflow_uploads)
    KEYFLOW_OUTPUT_DIR  — output dir      (default: /tmp/keyflow_outputs)
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from enum import Enum
from pathlib import Path

# NOTE: expandable_segments:True was removed — it causes private pool memory
# to NOT be releasable by empty_cache() between GVM sliding-window batches,
# leading to 17+ GiB stuck in private pools on A10G (22 GiB).
# Use garbage_collection_threshold instead to handle fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")
from typing import Any, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────────

class _MemoryLogHandler(logging.Handler):
    """Thread-safe ring-buffer that stores the last *maxlen* formatted log lines.

    Exposes a cursor-based API so clients only pull new lines:
        lines, next_seq = handler.since(after_seq)
    Each line is prepended with its sequence number for cheap filtering.
    """
    def __init__(self, maxlen: int = 500):
        super().__init__()
        self._buf: deque[tuple[int, str]] = deque(maxlen=maxlen)
        self._seq = 0
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:
            line = record.getMessage()
        with self._lock:
            self._seq += 1
            self._buf.append((self._seq, line))

    def since(self, after_seq: int) -> tuple[list[str], int]:
        """Return (new_lines, next_seq) where next_seq can be stored by the client."""
        with self._lock:
            result = [line for seq, line in self._buf if seq > after_seq]
            next_seq = self._seq
        return result, next_seq


_mem_handler = _MemoryLogHandler(maxlen=500)
_mem_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logging.getLogger().addHandler(_mem_handler)
logger = logging.getLogger("keyflow.worker")

# ── Paths ─────────────────────────────────────────────────────────────────────
_MODELS_DIR = Path(os.environ.get("KEYFLOW_MODELS_DIR", "")).expanduser() \
    if os.environ.get("KEYFLOW_MODELS_DIR") \
    else Path.home() / ".local" / "share" / "com.keyflow.studio" / "models"

_UPLOAD_DIR = Path(os.environ.get("KEYFLOW_UPLOAD_DIR", "/tmp/keyflow_uploads"))
_OUTPUT_DIR = Path(os.environ.get("KEYFLOW_OUTPUT_DIR", "/tmp/keyflow_outputs"))

for _d in (_UPLOAD_DIR, _OUTPUT_DIR, _MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
def _detect_device() -> str:
    forced = os.environ.get("KEYFLOW_DEVICE", "").strip().lower()
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"

DEVICE = _detect_device()
logger.info("KeyFlow Worker starting — device=%s  models=%s", DEVICE, _MODELS_DIR)
WORKER_VERSION = "1.0.0"


def _load_bundle_manifest() -> dict[str, Any]:
    worker_dir = Path(__file__).resolve().parent
    candidates = [
        worker_dir.parent / ".keyflow_bundle_manifest.json",
        worker_dir / ".keyflow_bundle_manifest.json",
    ]
    for manifest_path in candidates:
        if not manifest_path.is_file():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            logger.exception("Failed to read bundle manifest: %s", manifest_path)
    return {}

# ── Job store ─────────────────────────────────────────────────────────────────
class JobStatus(str, Enum):
    QUEUED     = "queued"
    RUNNING    = "running"
    DONE       = "done"
    FAILED     = "failed"
    CANCELLED  = "cancelled"

class Job:
    def __init__(self, job_id: str, model: str, params: dict):
        self.job_id   = job_id
        self.model    = model
        self.params   = params
        self.status   = JobStatus.QUEUED
        self.progress = 0          # 0-100
        self.stage    = "queued"
        self.error    = ""
        self.result_path: Optional[Path] = None
        self.created_at = time.time()
        self._cancel_event = threading.Event()
        self._log: list[str] = []

    def log(self, msg: str):
        self._log.append(msg)
        logger.info("[%s] %s", self.job_id[:8], msg)

    def to_dict(self) -> dict:
        return {
            "job_id":      self.job_id,
            "model":       self.model,
            "status":      self.status.value,
            "progress":    self.progress,
            "stage":       self.stage,
            "error":       self.error,
            "result_path": str(self.result_path) if self.result_path else None,
            "created_at":  self.created_at,
        }

_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()
_executor_semaphore = threading.Semaphore(1)  # only one job at a time (GPU limit)


def _get_job(job_id: str) -> Job:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return job


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="KeyFlow GPU Worker",
    description="Headless ML inference service for KeyFlow Studio",
    version=WORKER_VERSION,
)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Quick liveness check. Returns GPU info if available."""
    manifest = _load_bundle_manifest()
    info: dict[str, Any] = {
        "status": "ok",
        "device": DEVICE,
        "models_dir": str(_MODELS_DIR),
        "worker_version": WORKER_VERSION,
        "bundle_revision": str(manifest.get("revision") or ""),
        "bundle_file_count": int(manifest.get("file_count") or 0),
        "bundle_generated_at": str(manifest.get("generated_at") or ""),
    }
    try:
        import torch
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            mem = torch.cuda.mem_get_info(0)
            info["gpu_free_gb"]  = round(mem[0] / 1e9, 2)
            info["gpu_total_gb"] = round(mem[1] / 1e9, 2)
    except Exception as e:
        info["cuda_error"] = str(e)
    return info


# ── System logs ───────────────────────────────────────────────────────────────
@app.get("/system/logs")
def system_logs(after_seq: int = 0, lines: int = 200):
    """Return worker log lines collected since *after_seq*.

    Client workflow (cursor-based, no duplicates):
        seq = 0
        while True:
            data = GET /system/logs?after_seq={seq}&lines=200
            print(data["lines"])
            seq = data["next_seq"]
            sleep(3)
    """
    lines = min(lines, 500)
    new_lines, next_seq = _mem_handler.since(after_seq)
    return {"lines": new_lines[-lines:], "next_seq": next_seq}


# ── Model info ────────────────────────────────────────────────────────────────
_CORRIDORKEY_CHECKPOINT_FILENAMES = (
    "CorridorKey_v1.0.pth",
    "CorridorKey.pth",
    "corridorkey.pth",
)


def _find_corridorkey_checkpoint() -> Optional[Path]:
    """Return the first CorridorKey .pth file found under the models dir, or None."""
    base = _MODELS_DIR / "corridorkey" / "v1.0"
    for fname in _CORRIDORKEY_CHECKPOINT_FILENAMES:
        p = base / fname
        if p.exists():
            return p
    return None


@app.get("/models")
def list_models():
    """List available (downloaded) models."""
    available = {
        "matanyone2": (_MODELS_DIR / "matanyone2" / "v1" / "matanyone2.pth").exists(),
        "birefnet":   (_MODELS_DIR / "birefnet").exists(),
        "corridorkey": _find_corridorkey_checkpoint() is not None,
        "gvm":        (_MODELS_DIR / "gvm").exists(),
    }
    return {"models": available, "models_dir": str(_MODELS_DIR)}


# ═════════════════════════════════════════════════════════════════════════════
# MODEL DOWNLOAD ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

_download_tasks: dict[str, dict] = {}


def _run_download(task_id: str, model: str, preset: str = "") -> None:
    """Background task: download model weights on the server."""
    task = _download_tasks[task_id]
    try:
        import importlib.util as _ilu
        _dm_path = Path(__file__).parent / "download_models.py"
        spec = _ilu.spec_from_file_location("_dm", _dm_path)
        dm = _ilu.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(dm)  # type: ignore[union-attr]

        task["message"] = f"Downloading {model}..."
        if model == "matanyone2":
            dm.download_matanyone2()
        elif model == "birefnet":
            dm.download_birefnet(preset or "General")
        elif model == "gvm":
            dm.download_gvm()
        elif model == "corridorkey":
            dm.download_corridorkey()
        else:
            raise ValueError(f"Unknown model: {model!r}")

        task["status"] = "done"
        task["progress"] = 100
        task["message"] = "Downloaded"
    except Exception as exc:
        task["status"] = "error"
        task["message"] = str(exc).strip() or repr(exc)


@app.post("/models/download")
async def start_model_download(
    model: str,
    preset: str = "",
    background_tasks: BackgroundTasks = None,  # type: ignore[assignment]
):
    """Start downloading a model on the server. Poll GET /models/download/{task_id} for status."""
    task_id = str(uuid.uuid4())
    _download_tasks[task_id] = {"status": "running", "progress": 5, "message": f"Starting {model} download..."}
    background_tasks.add_task(_run_download, task_id, model, preset)
    return {"task_id": task_id}


@app.get("/models/download/{task_id}")
def get_download_task_status(task_id: str):
    """Poll download task status: {status, progress, message}."""
    task = _download_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Download task not found")
    return task


# ═════════════════════════════════════════════════════════════════════════════
# JOB SUBMISSION
# ═════════════════════════════════════════════════════════════════════════════

class MatAnyoneParams(BaseModel):
    n_warmup:    int   = Field(10, ge=0, le=50,  description="Warmup frames")
    r_erode:     int   = Field(0,  ge=0, le=50,  description="Mask erosion radius")
    r_dilate:    int   = Field(0,  ge=0, le=50,  description="Mask dilation radius")

class BiRefNetParams(BaseModel):
    preset:      str   = Field("General", description="BiRefNet preset name")

class JobSubmitResponse(BaseModel):
    job_id:      str
    status:      str


def _parse_graph_json(graph_json: str) -> dict:
    try:
        payload = json.loads(str(graph_json or "").strip())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid graph_json: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="graph_json must be an object")
    return payload


@app.post("/jobs/matanyone2", response_model=JobSubmitResponse)
async def submit_matanyone2(
    video: UploadFile = File(..., description="Input video file (mp4/mov/avi)"),
    mask:  UploadFile = File(..., description="Initial alpha mask (PNG, single channel or RGB)"),
    n_warmup: int  = Form(10),
    r_erode:  int  = Form(0),
    r_dilate: int  = Form(0),
):
    """
    Submit a MatAnyone2 matting job.

    Uploads video + mask image, starts async processing.
    Poll /jobs/{job_id} for status, then GET /jobs/{job_id}/result to download.
    """
    job_id   = str(uuid.uuid4())
    job_dir  = _UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_path = job_dir / _safe_filename(video.filename or "input.mp4")
    mask_path  = job_dir / _safe_filename(mask.filename  or "mask.png")

    async with _async_save(video_path, video):
        pass
    async with _async_save(mask_path, mask):
        pass

    params = {
        "video_path": str(video_path),
        "mask_path":  str(mask_path),
        "n_warmup":   n_warmup,
        "r_erode":    r_erode,
        "r_dilate":   r_dilate,
    }

    job = Job(job_id, "matanyone2", params)
    with _jobs_lock:
        _jobs[job_id] = job

    thread = threading.Thread(target=_run_job_safe, args=(job,), daemon=True, name=f"job-{job_id[:8]}")
    thread.start()

    return JobSubmitResponse(job_id=job_id, status=job.status)


@app.post("/jobs/birefnet", response_model=JobSubmitResponse)
async def submit_birefnet(
    image:  UploadFile = File(..., description="Input image (PNG/JPG)"),
    preset: str        = Form("General"),
):
    """
    Submit a BiRefNet background removal job (single image).
    Returns alpha mask PNG.
    """
    job_id  = str(uuid.uuid4())
    job_dir = _UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    image_path = job_dir / _safe_filename(image.filename or "input.png")
    async with _async_save(image_path, image):
        pass

    params = {"image_path": str(image_path), "preset": preset}
    job = Job(job_id, "birefnet", params)
    with _jobs_lock:
        _jobs[job_id] = job

    thread = threading.Thread(target=_run_job_safe, args=(job,), daemon=True, name=f"job-{job_id[:8]}")
    thread.start()

    return JobSubmitResponse(job_id=job_id, status=job.status)


@app.post("/jobs/node-graph", response_model=JobSubmitResponse)
async def submit_node_graph(
    video: UploadFile = File(..., description="Input video file (mp4/mov/avi) or sequence.zip"),
    graph_json: str = Form(..., description="Serialized node graph payload"),
    frame_start: int = Form(0, description="0-based start frame (inclusive)"),
    frame_end: int = Form(0, description="0-based end frame (exclusive, 0 = all)"),
):
    """Submit supported cloud node-graph job.

    Currently supported topology: source(out) -> gvm(image) -> export(in from alpha).
    """
    job_id = str(uuid.uuid4())
    job_dir = _UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_path = job_dir / _safe_filename(video.filename or "input.mp4")
    async with _async_save(video_path, video):
        pass

    params = {
        "video_path": str(video_path),
        "graph": _parse_graph_json(graph_json),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
    }

    job = Job(job_id, "node_graph", params)
    with _jobs_lock:
        _jobs[job_id] = job

    thread = threading.Thread(target=_run_job_safe, args=(job,), daemon=True, name=f"job-{job_id[:8]}")
    thread.start()

    return JobSubmitResponse(job_id=job_id, status=job.status)


# ═════════════════════════════════════════════════════════════════════════════
# JOB STATUS & RESULT
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """Get job status, progress (0-100), and stage description."""
    return _get_job(job_id).to_dict()


@app.get("/jobs/{job_id}/log")
def get_job_log(job_id: str):
    """Get full activity log for a job."""
    job = _get_job(job_id)
    return {"job_id": job_id, "log": job._log}


@app.get("/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str):
    """
    Server-Sent Events stream of job progress.
    Client receives JSON events:
      {"progress": 42, "stage": "processing frame 420/1000", "status": "running"}
    Closes when job reaches done/failed/cancelled.
    """
    job = _get_job(job_id)

    async def _generator():
        last_progress = -1
        last_stage = ""
        last_status = ""
        while True:
            status_value = job.status.value
            if (
                job.progress != last_progress
                or job.stage != last_stage
                or status_value != last_status
            ):
                last_progress = job.progress
                last_stage = job.stage
                last_status = status_value
                payload = json.dumps({
                    "progress": job.progress,
                    "stage":    job.stage,
                    "status":   status_value,
                    "error":    job.error,
                })
                yield f"data: {payload}\n\n"
                if job.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
                    break
            await asyncio.sleep(0.2)

    return StreamingResponse(_generator(), media_type="text/event-stream")


@app.get("/jobs/{job_id}/result")
def download_result(job_id: str):
    """Download the result file when job status is 'done'."""
    job = _get_job(job_id)
    if job.status != JobStatus.DONE:
        raise HTTPException(status_code=400, detail=f"Job is {job.status.value}, not done yet")
    if not job.result_path or not job.result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    return FileResponse(
        path=str(job.result_path),
        filename=job.result_path.name,
        media_type="application/octet-stream",
    )


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    """Request cancellation of a running job."""
    job = _get_job(job_id)
    if job.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
        return {"job_id": job_id, "status": job.status.value, "message": "Job already finished"}
    job._cancel_event.set()
    job.status = JobStatus.CANCELLED
    job.log("Cancelled by client request")
    return {"job_id": job_id, "status": job.status.value}


@app.delete("/jobs/{job_id}/files")
def cleanup_job_files(job_id: str):
    """Delete uploaded input files and output result for a completed job.

    Call after the client has successfully downloaded the result to free server
    disk space.  The job record itself is kept in memory for status queries.
    """
    job = _get_job(job_id)
    if job.status not in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
        raise HTTPException(
            status_code=400,
            detail=f"Job is still {job.status.value}; wait until it finishes before cleaning up",
        )

    cleaned: list[str] = []
    for directory in (_UPLOAD_DIR / job_id, _OUTPUT_DIR / job_id):
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
            cleaned.append(str(directory))

    job.log(f"Files cleaned up: {cleaned}")
    return {"job_id": job_id, "cleaned": cleaned}


@app.get("/jobs")
def list_jobs():
    """List all known jobs (last 50)."""
    with _jobs_lock:
        recent = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)[:50]
    return {"jobs": [j.to_dict() for j in recent]}


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE RUNNERS
# ═════════════════════════════════════════════════════════════════════════════

def _run_job_safe(job: Job):
    """Thread entry: acquire semaphore, run job, handle exceptions."""
    _executor_semaphore.acquire()
    try:
        job.status = JobStatus.RUNNING
        job.log(f"Started: model={job.model}")
        if job.model == "matanyone2":
            _run_matanyone2(job)
        elif job.model == "birefnet":
            _run_birefnet(job)
        elif job.model == "node_graph":
            _run_node_graph(job)
        else:
            raise RuntimeError(f"Unknown model: {job.model!r}")

        job.status   = JobStatus.DONE
        job.progress = 100
        job.stage    = "done"
        job.log(f"Finished: result={job.result_path}")
    except Exception:
        job.status = JobStatus.FAILED
        job.error  = traceback.format_exc()
        job.stage  = "error"
        job.log(f"FAILED:\n{job.error}")
        _unload_models_after_failure(job.model)
    finally:
        _gpu_cleanup()
        _executor_semaphore.release()


def _unload_models_after_failure(model_name: str) -> None:
    """Release model singletons and GPU cache after a failed job.

    If one model fails, we primarily unload that model singleton. If unload fails,
    we still clear Python and torch caches to recover for the next request.
    """
    global _matanyone2_service, _birefnet_service, _gvm_service, _corridorkey_service

    def _try_unload(service_obj, method_names: tuple[str, ...]) -> None:
        if service_obj is None:
            return
        for method_name in method_names:
            method = getattr(service_obj, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    logger.exception("Unload method failed: %s", method_name)
                return

    try:
        normalized = str(model_name or "").strip().lower()

        if normalized == "matanyone2":
            _try_unload(_matanyone2_service, ("unload", "unload_model", "unload_engine"))
            _matanyone2_service = None
            # ModelService is a class-level singleton; clear cached model explicitly.
            try:
                _add_project_to_path()
                from app.services.model_service import ModelService
                ModelService._model = None
                ModelService._instance = None
            except Exception:
                logger.exception("Failed to reset ModelService singleton")

        elif normalized == "birefnet":
            _try_unload(_birefnet_service, ("unload_model", "unload", "unload_engine"))
            _birefnet_service = None

        elif normalized == "node_graph":
            _try_unload(_gvm_service, ("unload", "unload_model", "unload_engine"))
            _gvm_service = None
            _try_unload(_corridorkey_service, ("unload", "unload_model", "unload_engine"))
            _corridorkey_service = None

        else:
            # Unknown failure source: drop everything to maximize recovery chance.
            _try_unload(_matanyone2_service, ("unload", "unload_model", "unload_engine"))
            _try_unload(_birefnet_service, ("unload_model", "unload", "unload_engine"))
            _try_unload(_gvm_service, ("unload", "unload_model", "unload_engine"))
            _try_unload(_corridorkey_service, ("unload", "unload_model", "unload_engine"))
            _matanyone2_service = None
            _birefnet_service = None
            _gvm_service = None
            _corridorkey_service = None

    finally:
        try:
            gc.collect()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


def _gpu_cleanup() -> None:
    """Free Python and GPU memory caches after model unload."""
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Release CUDA Graph private pool memory that empty_cache() misses.
            # This is critical after OOM failures to avoid 17+GiB stuck in pools.
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            try:
                # PyTorch 2.x internal: clears cuBLAS workspace allocations.
                torch._C._cuda_clearCublasWorkspaces()
            except Exception:
                pass
            # Second empty_cache after synchronize to catch deferred frees.
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _run_matanyone2(job: Job):
    """Run MatAnyone2 matting on a video.

    Lifecycle: upload already saved → load media → load model → infer →
               save result → unload model (always, even on error).
    """
    global _matanyone2_service
    p = job.params
    video_path = Path(p["video_path"])
    mask_path  = Path(p["mask_path"])

    # ── 1. Load media (already on disk after upload) ──────────────────────────
    job.stage = "loading video"
    job.log(f"Loading video: {video_path}")
    frames = _load_video_frames(video_path)
    if not frames:
        raise RuntimeError(f"Could not read frames from {video_path}")

    job.log(f"Loaded {len(frames)} frames, size={frames[0].shape[:2]}")
    fps = _get_video_fps(video_path)

    job.stage = "loading mask"
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")
    h, w = frames[0].shape[:2]
    if mask_img.shape[:2] != (h, w):
        mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)

    # ── 2. Load model ─────────────────────────────────────────────────────────
    job.stage = "loading model"
    job.log("Loading MatAnyone2 model...")
    model_service = _get_matanyone2_service()
    if not model_service.is_loaded():
        ckpt = _MODELS_DIR / "matanyone2" / "v1" / "matanyone2.pth"
        if not ckpt.exists():
            raise RuntimeError(
                f"MatAnyone2 checkpoint not found: {ckpt}\n"
                "Run: python3 download_models.py --model matanyone2"
            )
        model_service.load_model(str(ckpt))
    job.log("Model loaded.")

    try:
        # ── 3. Inference ──────────────────────────────────────────────────────
        job.stage = "inference"

        def _progress(cur, tot):
            job.progress = max(5, int(cur / tot * 85))
            job.stage    = f"processing frame {cur}/{tot}"
            if cur % 50 == 0:
                job.log(f"Frame {cur}/{tot}")

        alphas = model_service.process_video(
            frames=frames,
            mask=mask_img,
            n_warmup=p.get("n_warmup", 10),
            r_erode=p.get("r_erode",  0),
            r_dilate=p.get("r_dilate", 0),
            progress_callback=_progress,
            cancel_flag=job._cancel_event,
        )

        if job._cancel_event.is_set():
            job.status = JobStatus.CANCELLED
            job.log("Cancelled during inference")
            return

        # ── 4. Save result ────────────────────────────────────────────────────
        job.stage    = "saving result"
        job.progress = 90
        job.log("Saving result video...")

        out_dir  = _OUTPUT_DIR / job.job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{video_path.stem}_alpha.mp4"

        _save_alpha_video(alphas, frames, out_path, fps=fps)

        job.result_path = out_path
        job.progress    = 100
        job.log(f"Result saved: {out_path} ({out_path.stat().st_size // 1024} KB)")

    finally:
        # ── 5. Unload model (always) ──────────────────────────────────────────
        job.log("Unloading MatAnyone2 model...")
        try:
            for method in ("unload", "unload_model", "unload_engine"):
                fn = getattr(model_service, method, None)
                if callable(fn):
                    fn()
                    break
        except Exception:
            logger.exception("MatAnyone2 unload failed")
        _matanyone2_service = None
        try:
            _add_project_to_path()
            from app.services.model_service import ModelService
            ModelService._model    = None
            ModelService._instance = None
        except Exception:
            pass
        _gpu_cleanup()
        job.log("MatAnyone2 model unloaded.")


def _run_birefnet(job: Job):
    """Run BiRefNet single-image background removal.

    Lifecycle: upload already saved → load model → infer →
               save result → unload model (always, even on error).
    """
    global _birefnet_service
    from PIL import Image as PILImage

    p = job.params
    image_path = Path(p["image_path"])
    preset     = p.get("preset", "General")

    # ── 1. Media already on disk (uploaded) ──────────────────────────────────

    # ── 2. Load model ─────────────────────────────────────────────────────────
    job.stage = "loading model"
    job.log(f"Loading BiRefNet preset={preset}...")

    service = _get_birefnet_service()
    model_dir = str(_MODELS_DIR / "birefnet" / preset)
    if not Path(model_dir).exists():
        raise RuntimeError(
            f"BiRefNet model not found: {model_dir}\n"
            f"Run: python3 download_models.py --model birefnet --preset {preset}"
        )
    service.load_model_from_dir(model_dir=model_dir, device=DEVICE)
    job.log("BiRefNet loaded.")

    try:
        # ── 3. Inference ──────────────────────────────────────────────────────
        job.stage    = "inference"
        job.progress = 20
        img   = PILImage.open(image_path).convert("RGB")
        alpha = service.predict(img)  # float32 [0..1] HxW

        job.progress = 80

        # ── 4. Save result ────────────────────────────────────────────────────
        job.stage = "saving result"
        out_dir  = _OUTPUT_DIR / job.job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{image_path.stem}_alpha.png"

        alpha_u8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(out_path), alpha_u8)

        job.result_path = out_path
        job.progress    = 100
        job.log(f"BiRefNet result: {out_path}")

    finally:
        # ── 5. Unload model (always) ──────────────────────────────────────────
        job.log("Unloading BiRefNet model...")
        try:
            for method in ("unload_model", "unload", "unload_engine"):
                fn = getattr(service, method, None)
                if callable(fn):
                    fn()
                    break
        except Exception:
            logger.exception("BiRefNet unload failed")
        _birefnet_service = None
        _gpu_cleanup()
        job.log("BiRefNet model unloaded.")


# Processing node types currently supported for cloud execution.
# Source and Export are always the input/output boundaries of any cloud graph.
# These are the allowed middle nodes between them.
_CLOUD_PROCESSING_TYPES: set[str] = {"gvm", "corridorkey"}


def _validate_supported_graph(graph: dict) -> None:
    """Validate the graph follows the Source → [processing nodes] → Export architecture.

    Architecture contract:
    - **Source** node: input boundary — always the start of the pipeline.
    - **Export** node: output boundary — always the end of the pipeline.
    - **Between them**: any enabled processing nodes from _CLOUD_PROCESSING_TYPES.

    Connectivity: Source must connect (directly or through other nodes) to at least
    one processing node, and at least one processing node must connect to Export.
    """
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise RuntimeError("Graph payload must contain 'nodes' and 'edges' lists")

    boundary_types = {"source", "export"}
    allowed_types  = boundary_types | _CLOUD_PROCESSING_TYPES

    node_type_by_id: dict[str, str] = {}
    enabled_types:   set[str]       = set()

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id   = str(node.get("id")   or "").strip()
        node_type = str(node.get("type") or "").strip().lower()
        if not node_id or not node_type:
            continue
        enabled = bool(node.get("enabled", True))
        if enabled and node_type not in allowed_types:
            raise RuntimeError(
                f"Unsupported cloud node type '{node_type}'. "
                f"Allowed: {sorted(allowed_types)}"
            )
        node_type_by_id[node_id] = node_type
        if enabled:
            enabled_types.add(node_type)

    # Required boundary + GVM stage.
    if "source" not in enabled_types:
        raise RuntimeError("Cloud graph requires an enabled Source node (input boundary)")
    if "export" not in enabled_types:
        raise RuntimeError("Cloud graph requires an enabled Export node (output boundary)")
    if "gvm" not in enabled_types:
        raise RuntimeError("Cloud graph requires an enabled GVM node")

    # Required links for supported chains:
    #   source -> gvm
    #   (optional) source -> corridorkey
    #   (optional) gvm(alpha) -> corridorkey(alphahint)
    #   export can be fed either from gvm(alpha) or corridorkey(any output)
    has_source_to_gvm = False
    has_source_to_corridorkey = False
    has_gvm_to_corridorkey = False
    has_gvm_to_export = False
    has_corridorkey_to_export = False

    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src_id = str(edge.get("src_id") or "").strip()
        dst_id = str(edge.get("dst_id") or "").strip()
        src_port = str(edge.get("src_port") or "").strip().lower()
        dst_port = str(edge.get("dst_port") or "").strip().lower()
        if not src_id or not dst_id:
            continue
        src_type = node_type_by_id.get(src_id, "")
        dst_type = node_type_by_id.get(dst_id, "")

        if src_type == "source" and dst_type == "gvm" and dst_port == "image":
            has_source_to_gvm = True
        if src_type == "source" and dst_type == "corridorkey" and dst_port == "image":
            has_source_to_corridorkey = True
        if src_type == "gvm" and dst_type == "corridorkey" and src_port == "alpha" and dst_port == "alphahint":
            has_gvm_to_corridorkey = True
        if src_type == "gvm" and dst_type == "export" and src_port == "alpha" and dst_port in {"", "in"}:
            has_gvm_to_export = True
        if (
            src_type == "corridorkey"
            and dst_type == "export"
            and src_port in {"alpha", "fg", "comp", "processed"}
            and dst_port in {"", "in"}
        ):
            has_corridorkey_to_export = True

    if not has_source_to_gvm:
        raise RuntimeError("Source node must connect to GVM.image")

    has_corridorkey = "corridorkey" in enabled_types
    if has_corridorkey:
        if not has_source_to_corridorkey:
            raise RuntimeError("Source node must connect to CorridorKey.image")
        if not has_gvm_to_corridorkey:
            raise RuntimeError("GVM.alpha must connect to CorridorKey.alphahint")
        if not has_corridorkey_to_export:
            raise RuntimeError("CorridorKey output must connect to Export.in")
    else:
        if not has_gvm_to_export:
            raise RuntimeError("GVM.alpha must connect to Export.in")


def _resolve_export_source(graph: dict, node_type_by_id: dict[str, str]) -> tuple[str, str]:
    """Return (source_node_type, source_port) for enabled Export input edges.

    If multiple Export nodes are present, prefer CorridorKey outputs in the
    following order to avoid accidental Preview-only export when Premult is
    available: processed > fg > alpha > comp.
    """
    candidates = _resolve_export_targets(graph, node_type_by_id)
    if not candidates:
        return "gvm", "alpha"

    corridorkey_priority = {"processed": 0, "fg": 1, "alpha": 2, "comp": 3}
    corridorkey_candidates = [
        (src_type, src_port)
        for (src_type, src_port) in candidates
        if src_type == "corridorkey"
    ]
    if corridorkey_candidates:
        corridorkey_candidates.sort(key=lambda item: corridorkey_priority.get(item[1], 99))
        return corridorkey_candidates[0]

    return candidates[0]


def _resolve_export_targets(graph: dict, node_type_by_id: dict[str, str]) -> list[tuple[str, str]]:
    """Return ordered unique export targets as (source_node_type, source_port)."""
    export_ids = {
        str(n.get("id") or "").strip()
        for n in (graph.get("nodes") or [])
        if isinstance(n, dict)
        and bool(n.get("enabled", True))
        and str(n.get("type") or "").strip().lower() == "export"
    }
    targets: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for edge in (graph.get("edges") or []):
        if not isinstance(edge, dict):
            continue
        dst_id = str(edge.get("dst_id") or "").strip()
        dst_port = str(edge.get("dst_port") or "").strip().lower()
        if dst_id not in export_ids or dst_port not in {"", "in"}:
            continue
        src_id = str(edge.get("src_id") or "").strip()
        src_type = node_type_by_id.get(src_id, "")
        src_port = str(edge.get("src_port") or "").strip().lower() or "out"
        target = (src_type, src_port)
        if target not in seen:
            seen.add(target)
            targets.append(target)
    return targets


def _run_corridorkey_phase(
    job: Job,
    *,
    source_path: Path,
    alpha_dir: Path,
    graph: dict,
    output_root: Path,
    output_ports: set[str],
) -> dict[str, Path]:
    """Run CorridorKey using source frames + GVM alpha hints and return output dirs by port."""
    image_exts = {".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".bmp", ".webp"}
    alpha_paths = sorted(alpha_dir.glob("*.png"))
    if not alpha_paths:
        raise RuntimeError("CorridorKey phase requires GVM alpha PNG frames")

    corridorkey_props: dict[str, Any] = {}
    for node in (graph.get("nodes") or []):
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("type") or "").strip().lower()
        if node_type == "corridorkey" and bool(node.get("enabled", True)):
            corridorkey_props = dict(node.get("properties") or {})

    normalized_ports = {
        port if port in {"alpha", "fg", "comp", "processed"} else "processed"
        for port in (output_ports or {"processed"})
    }
    output_dirs = {
        port: output_root / f"corridorkey_{port}"
        for port in sorted(normalized_ports)
    }
    for out_dir in output_dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    hint_dilate_radius = int(corridorkey_props.get("hint_dilate_radius", 0))
    input_colorspace = str(corridorkey_props.get("input_colorspace", "auto")).strip().lower()
    input_is_linear = input_colorspace == "linear"

    service = _get_corridorkey_service()
    job.stage = "loading corridorkey"
    job.log("Loading CorridorKey model...")
    job.log(f"CorridorKey device: {DEVICE}")
    service.load_model(device=DEVICE)

    def _to_rgb_uint8(bgr_or_other: np.ndarray) -> np.ndarray:
        if bgr_or_other.ndim == 2:
            rgb = cv2.cvtColor(bgr_or_other, cv2.COLOR_GRAY2RGB)
        elif bgr_or_other.ndim == 3 and bgr_or_other.shape[2] == 4:
            rgb = cv2.cvtColor(bgr_or_other, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(bgr_or_other, cv2.COLOR_BGR2RGB)

        if rgb.dtype == np.uint8:
            return rgb
        if np.issubdtype(rgb.dtype, np.floating):
            arr = np.asarray(rgb, dtype=np.float32)
            if (float(arr.max()) if arr.size else 0.0) <= 1.0 + 1e-6:
                arr = arr * 255.0
            return np.clip(arr, 0.0, 255.0).astype(np.uint8)
        if np.issubdtype(rgb.dtype, np.integer):
            max_val = float(np.iinfo(rgb.dtype).max)
            arr = np.asarray(rgb, dtype=np.float32) / max(max_val, 1.0)
            return np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        return np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 255.0).astype(np.uint8)

    def _to_mask_linear(mask_arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(mask_arr)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if np.issubdtype(arr.dtype, np.integer):
            max_val = float(np.iinfo(arr.dtype).max)
            arr = arr.astype(np.float32) / max(max_val, 1.0)
        else:
            arr = arr.astype(np.float32)
            if (float(arr.max()) if arr.size else 0.0) > 1.0 + 1e-6:
                arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    def _save_output_frame(port: str, frame: np.ndarray, dst_path: Path) -> None:
        arr = np.asarray(frame)
        if port == "alpha":
            if arr.ndim == 3 and arr.shape[2] >= 1:
                arr = arr[:, :, 0]
            if np.issubdtype(arr.dtype, np.floating):
                u8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                u8 = np.clip(arr, 0, 255).astype(np.uint8)
            cv2.imwrite(str(dst_path), u8)
            return

        if port in {"fg", "comp"}:
            if arr.ndim == 3 and arr.shape[2] >= 3:
                rgb = arr[:, :, :3]
            else:
                raise RuntimeError(f"CorridorKey '{port}' output has invalid shape: {arr.shape}")
            if np.issubdtype(rgb.dtype, np.floating):
                rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
            cv2.imwrite(str(dst_path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
            return

        # processed (RGBA)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 4, axis=2)
        if arr.ndim != 3 or arr.shape[2] not in {3, 4}:
            raise RuntimeError(f"CorridorKey 'processed' output has invalid shape: {arr.shape}")
        if arr.shape[2] == 3:
            alpha = np.ones((arr.shape[0], arr.shape[1], 1), dtype=arr.dtype)
            arr = np.concatenate([arr, alpha], axis=2)
        if np.issubdtype(arr.dtype, np.floating):
            rgba_u8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            rgba_u8 = np.clip(arr, 0, 255).astype(np.uint8)
        cv2.imwrite(str(dst_path), cv2.cvtColor(rgba_u8, cv2.COLOR_RGBA2BGRA))

    total_written = 0
    job.stage = "corridorkey inference"

    if source_path.is_dir():
        frame_paths = sorted(
            p for p in source_path.iterdir()
            if p.is_file() and p.suffix.lower() in image_exts
        )
        total = min(len(frame_paths), len(alpha_paths))
        for idx in range(total):
            if job._cancel_event.is_set():
                job.status = JobStatus.CANCELLED
                job.log("Cancelled during CorridorKey phase")
                return output_dirs

            frame_bgr = cv2.imread(str(frame_paths[idx]), cv2.IMREAD_UNCHANGED)
            mask_raw = cv2.imread(str(alpha_paths[idx]), cv2.IMREAD_UNCHANGED)
            if frame_bgr is None or mask_raw is None:
                continue

            frame_rgb = _to_rgb_uint8(frame_bgr)
            mask_linear = _to_mask_linear(mask_raw)
            if mask_linear.shape[:2] != frame_rgb.shape[:2]:
                mask_linear = cv2.resize(mask_linear, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

            if hint_dilate_radius > 0:
                mask_u8 = (np.clip(mask_linear, 0.0, 1.0) * 255.0).astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * hint_dilate_radius + 1, 2 * hint_dilate_radius + 1))
                mask_linear = cv2.dilate(mask_u8, kernel).astype(np.float32) / 255.0

            result = service.process_frame(
                image=frame_rgb,
                alpha_hint=mask_linear,
                despill_strength=float(corridorkey_props.get("despill_strength", 0.5)),
                despeckle=bool(corridorkey_props.get("despeckle", True)),
                despeckle_size=int(corridorkey_props.get("despeckle_size", 400)),
                refiner_strength=float(corridorkey_props.get("refiner_strength", 1.0)),
                use_refiner=bool(corridorkey_props.get("use_refiner", True)),
                input_is_linear=input_is_linear,
            )

            stem = frame_paths[idx].stem or f"{idx:05d}"
            for output_port in sorted(output_dirs):
                output_frame = result.get(output_port)
                if output_frame is None:
                    raise RuntimeError(f"CorridorKey did not return '{output_port}' output")
                _save_output_frame(output_port, output_frame, output_dirs[output_port] / f"{stem}.png")
            total_written += 1

            pct = max(93, min(98, 93 + int(((idx + 1) / max(total, 1)) * 5)))
            job.progress = pct
            if idx % 25 == 0:
                job.log(f"CorridorKey frame {idx + 1}/{total} ({pct}%)")
    else:
        cap = cv2.VideoCapture(str(source_path))
        try:
            total = len(alpha_paths)
            idx = 0
            while idx < total:
                if job._cancel_event.is_set():
                    job.status = JobStatus.CANCELLED
                    job.log("Cancelled during CorridorKey phase")
                    return output_dirs

                ok, frame_bgr = cap.read()
                if not ok:
                    break
                mask_raw = cv2.imread(str(alpha_paths[idx]), cv2.IMREAD_UNCHANGED)
                if frame_bgr is None or mask_raw is None:
                    idx += 1
                    continue

                frame_rgb = _to_rgb_uint8(frame_bgr)
                mask_linear = _to_mask_linear(mask_raw)
                if mask_linear.shape[:2] != frame_rgb.shape[:2]:
                    mask_linear = cv2.resize(mask_linear, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

                if hint_dilate_radius > 0:
                    mask_u8 = (np.clip(mask_linear, 0.0, 1.0) * 255.0).astype(np.uint8)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * hint_dilate_radius + 1, 2 * hint_dilate_radius + 1))
                    mask_linear = cv2.dilate(mask_u8, kernel).astype(np.float32) / 255.0

                result = service.process_frame(
                    image=frame_rgb,
                    alpha_hint=mask_linear,
                    despill_strength=float(corridorkey_props.get("despill_strength", 0.5)),
                    despeckle=bool(corridorkey_props.get("despeckle", True)),
                    despeckle_size=int(corridorkey_props.get("despeckle_size", 400)),
                    refiner_strength=float(corridorkey_props.get("refiner_strength", 1.0)),
                    use_refiner=bool(corridorkey_props.get("use_refiner", True)),
                    input_is_linear=input_is_linear,
                )
                for output_port in sorted(output_dirs):
                    output_frame = result.get(output_port)
                    if output_frame is None:
                        raise RuntimeError(f"CorridorKey did not return '{output_port}' output")
                    _save_output_frame(output_port, output_frame, output_dirs[output_port] / f"{idx:05d}.png")
                total_written += 1
                pct = max(93, min(98, 93 + int(((idx + 1) / max(total, 1)) * 5)))
                job.progress = pct
                if idx % 25 == 0:
                    job.log(f"CorridorKey frame {idx + 1}/{total} ({pct}%)")
                idx += 1
        finally:
            cap.release()

    if total_written <= 0:
        raise RuntimeError("CorridorKey phase produced no output frames")

    job.log(
        f"CorridorKey phase completed: {total_written} frame(s), outputs={','.join(sorted(output_dirs))}"
    )
    return output_dirs


def _run_node_graph(job: Job):
    """Run supported cloud node-graph job (GVM and optional CorridorKey phase)."""
    import subprocess
    import zipfile as _zipfile

    p = job.params
    upload_path = Path(p["video_path"])
    graph = p.get("graph")
    frame_start = int(p.get("frame_start") or 0)
    frame_end   = int(p.get("frame_end")   or 0)
    if not isinstance(graph, dict):
        raise RuntimeError("Missing graph payload")

    _validate_supported_graph(graph)

    job.stage = "loading graph"
    job.progress = 5
    job.log("Cloud graph accepted")

    node_type_by_id: dict[str, str] = {}
    for _node in (graph.get("nodes") or []):
        if isinstance(_node, dict) and bool(_node.get("enabled", True)):
            _node_id = str(_node.get("id") or "").strip()
            if _node_id:
                node_type_by_id[_node_id] = str(_node.get("type") or "").strip().lower()

    output_root = _OUTPUT_DIR / job.job_id
    alpha_dir = output_root / "gvm_alpha"
    output_root.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve actual input for GVM ──────────────────────────────────────────
    # Case 1: client uploaded a sequence ZIP (already pre-filtered to range)
    if upload_path.suffix.lower() == ".zip":
        job.stage = "extracting sequence"
        job.log(f"Extracting uploaded sequence: {upload_path}")
        seq_dir = upload_path.parent / "seq_frames"
        seq_dir.mkdir(parents=True, exist_ok=True)
        with _zipfile.ZipFile(upload_path, "r") as zf:
            zf.extractall(seq_dir)
        video_path = seq_dir  # GVM will iterate this directory
    # Case 2: video file — extract frame-accurate PNG sequence for requested range
    elif frame_start > 0 or frame_end > 0:
        end_inclusive = (frame_end - 1) if frame_end > 0 else 999999
        frames_dir = upload_path.parent / "extracted_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        job.stage = "extracting frames"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(upload_path),
            "-vf", f"select=between(n\\,{frame_start}\\,{end_inclusive}),setpts=PTS-STARTPTS",
            "-vsync", "0",
            str(frames_dir / "%05d.png"),
        ]
        n_frames = end_inclusive - frame_start + 1
        job.log(f"Extracting frames {frame_start}..{end_inclusive} ({n_frames} frames) as PNG sequence")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffmpeg frame extraction failed: {e.stderr.decode(errors='replace')}"
            ) from e
        video_path = frames_dir  # GVM accepts image directory
    else:
        video_path = upload_path

    # ── Load model ────────────────────────────────────────────────────────────
    job.stage = "loading model"
    job.progress = 12
    job.log("Loading GVM model...")
    gvm = _get_gvm_service()
    gvm.load_model(device=DEVICE)

    try:
        if job._cancel_event.is_set():
            job.status = JobStatus.CANCELLED
            job.log("Cancelled before graph inference")
            return

        job.stage = "gvm inference"

        # ── Extract GVM node properties from graph ──────────────────────────
        gvm_props: dict = {}
        for _n in (graph.get("nodes") or []):
            if isinstance(_n, dict) and str(_n.get("type", "")).lower() == "gvm" and _n.get("enabled", True):
                gvm_props = dict(_n.get("properties") or {})
                break

        _batch_size = int(gvm_props.get("num_frames_per_batch", 8))

        def _on_progress(done_frames: int, total_frames: int) -> None:
            pct = max(15, min(92, 15 + int((done_frames / max(total_frames, 1)) * 77)))
            job.progress = pct
            done_batches = (done_frames + _batch_size - 1) // _batch_size if done_frames > 0 else 0
            total_batches = max(1, (total_frames + _batch_size - 1) // _batch_size)
            job.stage = f"GVM {done_batches}/{total_batches} batches"
            # Log only at 0%, 25%, 50%, 75%, 100% milestones to avoid log spam.
            prev_pct = getattr(_on_progress, "_last_logged_pct", -1)
            milestone = (pct // 25) * 25
            if milestone > prev_pct:
                job.log(f"GVM batch {done_batches}/{total_batches} ({pct}%)")
                _on_progress._last_logged_pct = milestone  # type: ignore[attr-defined]

        alpha_paths = gvm.process_sequence(
            input_path=video_path,
            output_dir=alpha_dir,
            progress_callback=_on_progress,
            num_frames_per_batch=_batch_size,
            decode_chunk_size=int(gvm_props.get("decode_chunk_size", 4)),
            num_overlap_frames=int(gvm_props.get("num_overlap_frames", 1)),
            num_interp_frames=int(gvm_props.get("num_interp_frames", 1)),
            noise_type=str(gvm_props.get("noise_type", "zeros")),
            use_clip_img_emb=bool(gvm_props.get("use_clip_img_emb", False)),
            dilate_radius=int(gvm_props.get("dilate_radius", 0)),
        )

        if job._cancel_event.is_set():
            job.status = JobStatus.CANCELLED
            job.log("Cancelled during graph inference")
            return

        if not alpha_paths:
            raise RuntimeError("GVM did not produce output frames")

        # ── Unload GVM immediately after use to free memory for CorridorKey ──
        job.log("Unloading GVM model to free GPU memory...")
        try:
            gvm.unload()
        except Exception:
            logger.exception("GVM unload failed (will retry in finally)")

        export_targets = _resolve_export_targets(graph, node_type_by_id)
        if not export_targets:
            export_targets = [("gvm", "alpha")]

        corridorkey_ports = {
            src_port
            for src_type, src_port in export_targets
            if src_type == "corridorkey"
        }
        result_dirs: dict[str, Path] = {}
        if corridorkey_ports:
            result_dirs.update(
                _run_corridorkey_phase(
                    job,
                    source_path=video_path,
                    alpha_dir=alpha_dir,
                    graph=graph,
                    output_root=output_root,
                    output_ports=corridorkey_ports,
                )
            )
            if job._cancel_event.is_set() or job.status == JobStatus.CANCELLED:
                return

        for src_type, src_port in export_targets:
            if src_type == "corridorkey":
                if src_port not in result_dirs:
                    raise RuntimeError(f"CorridorKey export port was not produced: {src_port}")
            elif src_type == "gvm":
                result_dirs.setdefault("alpha", alpha_dir)
            else:
                raise RuntimeError(f"Unsupported export source for cloud graph: {src_type!r}")

        ordered_ports = [
            port
            for port in ["processed", "fg", "alpha", "comp"]
            if port in result_dirs
        ]
        if not ordered_ports:
            raise RuntimeError("Cloud graph did not produce any export streams")

        result_suffix = f"corridorkey_{ordered_ports[0]}" if any(port != "alpha" for port in ordered_ports) else "gvm_alpha"

        job.stage = "packing result"
        job.progress = 95
        result_base = output_root / f"{upload_path.stem}_{result_suffix}"
        if len(result_dirs) == 1:
            only_dir = result_dirs[ordered_ports[0]]
            zip_path = Path(
                shutil.make_archive(
                    base_name=str(result_base),
                    format="zip",
                    root_dir=str(only_dir),
                )
            )
        else:
            zip_path = result_base.with_suffix(".zip")
            with _zipfile.ZipFile(zip_path, "w", compression=_zipfile.ZIP_DEFLATED) as zf:
                for port in ordered_ports:
                    frames_dir = result_dirs[port]
                    for frame_path in sorted(frames_dir.iterdir()):
                        if frame_path.is_file():
                            zf.write(frame_path, arcname=f"{port}/{frame_path.name}")
        job.result_path = zip_path
        job.progress = 100
        job.log(f"Graph result packed: {zip_path}")
    finally:
        # ── 5. Unload model (always) ──────────────────────────────────────────
        job.log("Unloading GVM model...")
        try:
            gvm.unload()
        except Exception:
            logger.exception("GVM unload failed")
        global _gvm_service, _corridorkey_service
        _gvm_service = None
        if _corridorkey_service is not None:
            try:
                _corridorkey_service.unload()
            except Exception:
                logger.exception("CorridorKey unload failed")
            _corridorkey_service = None
        _gpu_cleanup()
        job.log("Cloud node-graph models unloaded.")


# ═════════════════════════════════════════════════════════════════════════════
# VIDEO UTILS
# ═════════════════════════════════════════════════════════════════════════════

def _load_video_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def _get_video_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps


def _save_alpha_video(
    alphas: list[np.ndarray],
    frames: list[np.ndarray],
    out_path: Path,
    fps: float = 25.0,
):
    """Save RGBA video with premultiplied alpha using ffmpeg (better quality)."""
    import subprocess

    h, w = frames[0].shape[:2]
    tmp_dir = out_path.parent / "_tmp_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i, (frame_rgb, alpha) in enumerate(zip(frames, alphas)):
            # Build RGBA PNG
            alpha_u8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
            if alpha_u8.shape[:2] != (h, w):
                alpha_u8 = cv2.resize(alpha_u8, (w, h), interpolation=cv2.INTER_LINEAR)
            rgba = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGRA)
            rgba[:, :, 3] = alpha_u8
            cv2.imwrite(str(tmp_dir / f"{i:06d}.png"), rgba)

        # Encode with ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-r", str(fps),
            "-i", str(tmp_dir / "%06d.png"),
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL SINGLETONS
# ═════════════════════════════════════════════════════════════════════════════

_matanyone2_service   = None
_birefnet_service     = None
_gvm_service          = None
_corridorkey_service  = None
_service_lock         = threading.Lock()


class _WorkerGVMService:
    """Minimal GVM service for EC2 worker (no dependency on app package)."""

    def __init__(self) -> None:
        self._processor = None

    def load_model(self, *, device: str) -> None:
        if self._processor is not None:
            return

        model_dir = _MODELS_DIR / "gvm"
        if not model_dir.exists():
            raise RuntimeError(
                f"GVM model dir not found: {model_dir}\n"
                "Run: python3 download_models.py --model gvm"
            )

        def _is_valid_model_base(base: Path) -> bool:
            return (
                (base / "vae" / "config.json").is_file()
                and (base / "unet" / "config.json").is_file()
                and (base / "scheduler" / "scheduler_config.json").is_file()
            )

        candidates = [
            model_dir,
            model_dir / "weights",
            model_dir / "gvm_core" / "weights",
        ]
        model_base = next((c for c in candidates if _is_valid_model_base(c)), None)
        if model_base is None:
            raise RuntimeError(
                "GVM weights layout is invalid. Expected one of:\n"
                f"  - {model_dir}/(vae,unet,scheduler)\n"
                f"  - {model_dir}/weights/(vae,unet,scheduler)\n"
                f"  - {model_dir}/gvm_core/weights/(vae,unet,scheduler)"
            )

        try:
            _add_project_to_path()
            from gvm_core import GVMProcessor
        except Exception as exc:
            raise RuntimeError(
                "gvm_core package is not available on worker. "
                "Deploy gvm_core/ and required dependencies."
            ) from exc

        self._processor = GVMProcessor(
            model_base=str(model_base),
            unet_base=str(model_base),
            lora_base=str(model_base / "unet"),
            device=device,
        )
        # Enable VAE tiling to avoid CUDA OOM on large resolutions.
        # AutoencoderKLTemporalDecoder supports enable_tiling() (spatial tiles),
        # but NOT enable_slicing() (batch slicing is not implemented for it).
        try:
            self._processor.pipe.vae.enable_tiling()
        except (AttributeError, NotImplementedError):
            pass  # older diffusers without tiling support

    def process_sequence(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        progress_callback,
        num_frames_per_batch: int = 8,
        decode_chunk_size: int = 4,
        num_overlap_frames: int = 1,
        num_interp_frames: int = 1,
        noise_type: str = "zeros",
        use_clip_img_emb: bool = False,
        dilate_radius: int = 0,
    ) -> list[Path]:
        if self._processor is None:
            raise RuntimeError("GVM processor is not loaded")

        output_dir.mkdir(parents=True, exist_ok=True)

        video_exts = {".mp4", ".mkv", ".gif", ".mov", ".avi"}
        if input_path.suffix.lower() in video_exts:
            _cap = cv2.VideoCapture(str(input_path))
            total_frames = max(1, int(_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
            _cap.release()
        elif input_path.is_dir():
            image_exts = {".png", ".jpg", ".jpeg", ".exr"}
            total_frames = max(
                1,
                len([fp for fp in input_path.iterdir() if fp.suffix.lower() in image_exts]),
            )
        else:
            total_frames = 1

        def _batch_progress(completed_batches: int, total_batches: int) -> None:
            done_frames = min(completed_batches * num_frames_per_batch, total_frames)
            if progress_callback is not None:
                progress_callback(done_frames, total_frames)

        self._processor.process_sequence(
            input_path=str(input_path),
            output_dir=str(output_dir),
            num_frames_per_batch=num_frames_per_batch,
            denoise_steps=1,
            decode_chunk_size=decode_chunk_size,
            num_overlap_frames=num_overlap_frames,
            num_interp_frames=num_interp_frames,
            noise_type=str(noise_type or "zeros"),
            use_clip_img_emb=bool(use_clip_img_emb),
            mode="matte",
            write_video=False,
            direct_output_dir=str(output_dir),
            progress_callback=_batch_progress,
        )
        frames = sorted(output_dir.glob("*.png"))
        if dilate_radius > 0 and frames:
            import cv2 as _cv2
            kernel = _cv2.getStructuringElement(
                _cv2.MORPH_ELLIPSE, (2 * dilate_radius + 1, 2 * dilate_radius + 1)
            )
            for fp in frames:
                img = _cv2.imread(str(fp), _cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    _cv2.imwrite(str(fp), _cv2.dilate(img, kernel))
        return frames

    def unload(self) -> None:
        if self._processor is None:
            return
        # Explicitly delete VAE and UNet tensors from GPU before zeroing reference
        try:
            pipe = getattr(self._processor, "pipe", None)
            if pipe is not None:
                for attr in ("vae", "unet", "scheduler"):
                    module = getattr(pipe, attr, None)
                    if module is not None:
                        try:
                            module.to("cpu")
                        except Exception:
                            pass
                        setattr(pipe, attr, None)
        except Exception:
            pass
        self._processor = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


class _WorkerCorridorKeyService:
    """Worker-local CorridorKey wrapper using original upstream CorridorKeyModule."""

    def __init__(self) -> None:
        self._engine = None
        self._engine_use_refiner: bool | None = None

    def load_model(self, *, device: str, use_refiner: bool = True) -> None:
        requested_use_refiner = bool(use_refiner)
        if self._engine is not None and self._engine_use_refiner == requested_use_refiner:
            return

        if self._engine is not None and self._engine_use_refiner != requested_use_refiner:
            self.unload()

        checkpoint = _find_corridorkey_checkpoint()
        if checkpoint is None:
            raise RuntimeError(
                "CorridorKey weights are missing on the cloud server. "
                "Open CorridorKey node in Cloud mode and click 'Check/Download model', then run again."
            )

        try:
            from CorridorKeyModule.inference_engine import CorridorKeyEngine
        except Exception as exc:
            raise RuntimeError(
                "Original CorridorKeyModule is not available on server. "
                "Install upstream CorridorKey repository first."
            ) from exc

        import torch

        self._engine = CorridorKeyEngine(
            checkpoint_path=str(checkpoint),
            device=str(device or "cpu"),
            img_size=2048,
            use_refiner=requested_use_refiner,
            mixed_precision=True,
            model_precision=torch.float32,
        )
        self._engine_use_refiner = requested_use_refiner

    def process_frame(
        self,
        *,
        image: np.ndarray,
        alpha_hint: np.ndarray,
        despill_strength: float,
        despeckle: bool,
        despeckle_size: int,
        refiner_strength: float,
        use_refiner: bool,
        input_is_linear: bool,
    ) -> dict[str, np.ndarray]:
        if self._engine is None or self._engine_use_refiner != bool(use_refiner):
            self.load_model(device=DEVICE, use_refiner=bool(use_refiner))

        despill_01 = max(0.0, min(1.0, float(despill_strength)))

        import torch
        with torch.inference_mode():
            result = self._engine.process_frame(
                image=image,
                mask_linear=alpha_hint,
                refiner_scale=float(refiner_strength),
                input_is_linear=bool(input_is_linear),
                fg_is_straight=True,
                despill_strength=despill_01,
                auto_despeckle=bool(despeckle),
                despeckle_size=int(despeckle_size),
                generate_comp=True,
                post_process_on_gpu=True,
            )

        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected CorridorKey output type: {type(result)}")

        # Apply max(R,B) luminance-compensating despill on CPU after engine output.
        # The engine uses average (R+B)/2 internally which shifts green clothing purple;
        # max(R,B) correctly preserves teal/cyan and keeps green -> neutral gray.
        fg = result.get("fg")
        if despill_01 > 1e-6 and fg is not None:
            fg_f = np.asarray(fg, dtype=np.float32)
            if fg_f.ndim == 3 and fg_f.shape[2] >= 3:
                _r = fg_f[:, :, 0]; _g = fg_f[:, :, 1]; _b = fg_f[:, :, 2]
                _limit = np.maximum(_r, _b)           # max(R,B): teal/cyan = no spill
                _spill = np.maximum(_g - _limit, 0.0)
                _lum = _spill * 0.587                 # Rec.601 luminance compensation
                _r_d = _r + _lum
                _g_d = _g - _spill + _lum
                _b_d = _b + _lum
                _despilled = np.stack([_r_d, _g_d, _b_d], axis=-1)
                if despill_01 < 1.0:
                    _fg_out = np.clip(fg_f * (1.0 - despill_01) + _despilled * despill_01, 0.0, 1.0)
                else:
                    _fg_out = np.clip(_despilled, 0.0, 1.0)

                result["fg"] = _fg_out

                alpha_arr = result.get("alpha")
                if alpha_arr is not None:
                    _a = np.clip(np.asarray(alpha_arr, dtype=np.float32), 0.0, 1.0)
                    if _a.ndim == 3:
                        _a = _a[:, :, 0]
                    _a3 = _a[:, :, np.newaxis]

                    if "processed" in result and result["processed"] is not None:
                        result["processed"] = np.concatenate([_fg_out * _a3, _a3], axis=-1).astype(np.float32)

                    if "comp" in result and result["comp"] is not None:
                        _fg_lin = np.where(_fg_out <= 0.04045, _fg_out / 12.92,
                                           ((_fg_out + 0.055) / 1.055) ** 2.4)
                        _h, _w = _fg_out.shape[:2]
                        _gy = np.arange(_h) // 64
                        _gx = np.arange(_w) // 64
                        _checker = ((_gy[:, None] + _gx[None, :]) % 2).astype(np.float32)
                        _bg_srgb = _checker * 0.4 + 0.15
                        _bg_lin = np.where(_bg_srgb <= 0.04045, _bg_srgb / 12.92,
                                           ((_bg_srgb + 0.055) / 1.055) ** 2.4)[..., np.newaxis]
                        _comp_lin = _fg_lin * _a3 + _bg_lin * (1.0 - _a3)
                        _c = np.clip(_comp_lin, 0.0, 1.0)
                        _comp_srgb = np.where(_c <= 0.0031308, _c * 12.92,
                                              1.055 * _c ** (1.0 / 2.4) - 0.055)
                        result["comp"] = np.clip(_comp_srgb, 0.0, 1.0).astype(np.float32)

        return result

    def unload(self) -> None:
        if self._engine is None:
            return
        self._engine = None
        self._engine_use_refiner = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


def _get_matanyone2_service():
    """Return singleton ModelService (lazy init, thread-safe)."""
    global _matanyone2_service
    if _matanyone2_service is not None:
        return _matanyone2_service
    with _service_lock:
        if _matanyone2_service is None:
            # Add project root to path so app.services can be imported
            _add_project_to_path()
            os.environ.setdefault("KEYFLOW_DEVICE", DEVICE)
            from app.services.model_service import ModelService
            _matanyone2_service = ModelService()
    return _matanyone2_service


def _get_birefnet_service():
    """Return singleton BiRefNetService (lazy init, thread-safe)."""
    global _birefnet_service
    if _birefnet_service is not None:
        return _birefnet_service
    with _service_lock:
        if _birefnet_service is None:
            _add_project_to_path()
            os.environ.setdefault("KEYFLOW_DEVICE", DEVICE)
            from app.services.birefnet_service import BiRefNetService
            _birefnet_service = BiRefNetService()
    return _birefnet_service


def _get_gvm_service():
    """Return singleton worker-local GVM service (lazy init, thread-safe)."""
    global _gvm_service
    if _gvm_service is not None:
        return _gvm_service
    with _service_lock:
        if _gvm_service is None:
            _gvm_service = _WorkerGVMService()
    return _gvm_service


def _get_corridorkey_service():
    """Return singleton worker-local CorridorKey service (lazy init, thread-safe)."""
    global _corridorkey_service
    if _corridorkey_service is not None:
        return _corridorkey_service
    with _service_lock:
        if _corridorkey_service is None:
            _corridorkey_service = _WorkerCorridorKeyService()
    return _corridorkey_service


def _add_project_to_path():
    """Add likely project roots to sys.path so app.* imports work in dev and EC2 bundle layouts."""
    worker_dir = Path(__file__).resolve().parent
    for project_root in (worker_dir.parent, worker_dir.parent.parent):
        app_dir = project_root / "app"
        if app_dir.is_dir() and str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _safe_filename(name: str) -> str:
    """Strip path separators and limit length."""
    safe = Path(name).name
    safe = "".join(c for c in safe if c.isalnum() or c in "._-")
    return safe[:128] or "upload"


class _async_save:
    """Async context manager: save UploadFile to disk without blocking event loop."""
    def __init__(self, path: Path, upload: UploadFile):
        self._path   = path
        self._upload = upload

    async def __aenter__(self):
        contents = await self._upload.read()
        self._path.write_bytes(contents)
        return self._path

    async def __aexit__(self, *_):
        pass


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "worker:app",
        host="0.0.0.0",
        port=8080,
        workers=1,     # 1 worker — GPU не может параллелить
        log_level="info",
        reload=False,
        h11_max_incomplete_event_size=256 * 1024 * 1024,  # 256MB upload limit
    )
