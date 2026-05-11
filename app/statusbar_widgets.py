"""Status bar widgets for KeyFlow Studio main window."""

from __future__ import annotations

import json
import threading
import urllib.request

import psutil
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget


class CloudGpuWidget(QWidget):
    """Statusbar widget that shows live EC2 GPU stats when cloud is enabled.

    Polls /health every 6 s independently of any running job.
    Call ``start(base_url)`` when the cloud is connected / job starts.
    Call ``stop()`` to hide when cloud is disabled.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lbl = QLabel(self)
        self._lbl.setStyleSheet(
            "color: #43c7ff; font-size: 11px; padding: 0 8px;"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._lbl)

        self._base_url: str = ""
        self._max_failures_before_hide = 2
        self._consecutive_failures = 0
        self._timer = QTimer(self)
        self._timer.setInterval(6000)
        self._timer.timeout.connect(self._poll)
        self.hide()

    def start(self, base_url: str) -> None:
        """Begin polling the given EC2 base URL."""
        self._base_url = base_url
        self._consecutive_failures = 0
        self._poll()          # immediate first update
        self._timer.start()

    def _mark_unavailable(self) -> None:
        """Hide stale GPU text when remote worker is not reachable."""
        self._lbl.clear()
        self.hide()

    def update_health(self, health: dict) -> None:
        """Inject health data received from an active job (no extra HTTP call)."""
        self._consecutive_failures = 0
        self._render(health)
        if not self._timer.isActive():
            self._timer.start()
        self.show()

    def stop(self) -> None:
        self._timer.stop()
        self._base_url = ""
        self._consecutive_failures = 0
        self._mark_unavailable()

    def _poll(self) -> None:
        if not self._base_url:
            self._mark_unavailable()
            return
        url = self._base_url
        result: list = [None]

        def _fetch():
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=4) as r:
                    result[0] = json.loads(r.read())
            except Exception:
                result[0] = {}

        def _check():
            if result[0] is None:
                QTimer.singleShot(200, _check)
                return
            if result[0]:
                self._consecutive_failures = 0
                self._render(result[0])
                self.show()
                return

            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_failures_before_hide:
                self._mark_unavailable()

        threading.Thread(target=_fetch, daemon=True).start()
        QTimer.singleShot(200, _check)

    def _render(self, health: dict) -> None:
        gpu_free  = health.get("gpu_free_gb")
        gpu_total = health.get("gpu_total_gb")
        gpu_name  = str(health.get("gpu") or "GPU")
        if gpu_total is not None and gpu_free is not None:
            used = float(gpu_total) - float(gpu_free)
            self._lbl.setText(f"☁ {gpu_name}: VRAM {used:.1f}/{float(gpu_total):.0f} GB")
        else:
            self._lbl.setText(f"☁ {gpu_name}")


class ResourceMonitorWidget(QWidget):
    """Compact CPU / RAM / VRAM monitor for the status bar.

    When a remote cloud job is active, shows EC2 GPU stats instead of local VRAM.
    Call ``set_remote_health(health_dict)`` to inject EC2 health data.
    Call ``clear_remote_health()`` to revert to local display.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lbl = QLabel(self)
        self._lbl.setStyleSheet(
            "color: #7a9ab5; font-size: 11px; padding: 0 6px;"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._lbl)

        self._proc = psutil.Process()
        self._has_cuda = False
        try:
            import torch
            self._has_cuda = torch.cuda.is_available()
        except ImportError:
            pass

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(2000)
        self._refresh()

    def _refresh(self):
        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            ram_used = ram.used / 1024 ** 3
            ram_total = ram.total / 1024 ** 3

            parts = [
                f"CPU {cpu:4.1f}%",
                f"RAM {ram_used:.1f}/{ram_total:.0f} GB",
            ]

            if self._has_cuda:
                try:
                    import torch
                    vram_used = torch.cuda.memory_allocated() / 1024 ** 3
                    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                    parts.append(f"VRAM {vram_used:.1f}/{vram_total:.0f} GB")
                except Exception:
                    pass
            else:
                # macOS: show process RSS as model memory proxy
                try:
                    rss = self._proc.memory_info().rss / 1024 ** 3
                    parts.append(f"Proc {rss:.1f} GB")
                except Exception:
                    pass

            self._lbl.setText("  │  ".join(parts))
        except Exception:
            pass
