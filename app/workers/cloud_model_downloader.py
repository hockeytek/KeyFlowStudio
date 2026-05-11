"""Cloud-side model download worker (polls EC2 worker API)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlencode

from PySide6.QtCore import QObject, Signal


class CloudModelDownloadWorker(QObject):
    """QThread worker: check and optionally download a model on the EC2 worker.

    Signals:
        progress(int, str)   – progress 0-100 + message string
        already_present()    – model is already on the server
        finished(str)        – download complete (model name)
        error(str)           – error message
    """

    progress = Signal(int, str)
    already_present = Signal()
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, api_base: str, model: str, preset: str = "") -> None:
        super().__init__()
        self._api_base = api_base.rstrip("/")
        self._model = model
        self._preset = preset

    # ── helpers ──────────────────────────────────────────────────────────────

    def _get_json(self, path: str, timeout: int = 10) -> dict:
        url = f"{self._api_base}{path}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _post_json(self, path: str, params: dict, timeout: int = 15) -> dict:
        # FastAPI Query params via POST (simple model, no request body needed)
        qs = urlencode({k: v for k, v in params.items() if v})
        url = f"{self._api_base}{path}?{qs}" if qs else f"{self._api_base}{path}"
        req = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    # ── main ─────────────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            # 1. Check if already present (skip for birefnet — need per-preset check)
            self.progress.emit(0, "Checking server...")
            models_info = self._get_json("/models").get("models", {})

            if self._model != "birefnet" and models_info.get(self._model):
                self.progress.emit(100, "Already on server")
                self.already_present.emit()
                return

            # 2. Start download on EC2
            self.progress.emit(5, "Starting download on server...")
            params: dict = {"model": self._model}
            if self._preset:
                params["preset"] = self._preset
            resp = self._post_json("/models/download", params)
            task_id = resp.get("task_id")
            if not task_id:
                self.error.emit("Server did not return task_id")
                return

            # 3. Poll until done
            pulse = 5
            while True:
                time.sleep(3)
                status = self._get_json(f"/models/download/{task_id}")
                state = str(status.get("status", "running"))
                message = str(status.get("message", "Downloading..."))
                reported = int(status.get("progress", 0))

                if state == "running":
                    # Pulse 5→90 to indicate activity
                    pulse = min(90, pulse + 3)
                    self.progress.emit(max(reported, pulse), message)
                elif state == "done":
                    self.progress.emit(100, "Done")
                    self.finished.emit(self._model)
                    return
                elif state == "error":
                    self.error.emit(message or "Unknown error on server")
                    return

        except urllib.error.URLError as exc:
            self.error.emit(f"Connection error: {exc.reason}")
        except Exception as exc:
            self.error.emit(str(exc).strip() or repr(exc))
