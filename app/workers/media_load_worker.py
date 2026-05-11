"""Background worker for media loading with progress reporting."""

from __future__ import annotations

import threading

import cv2

from PySide6.QtCore import QObject, Signal

from app.constants import DEFAULT_FPS
from app.i18n import t
from app.utils.media import (
    is_numbered_image_sequence,
    load_rgb_image,
    resolve_numbered_image_sequence,
)


class MediaLoadWorker(QObject):
    """Load image/video/sequence data in worker thread."""

    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._media_path = ""
        self._is_video = True
        self._language_code = "ru"
        self._cancel_flag = threading.Event()

    def configure_job(self, media_path: str, is_video: bool, language_code: str) -> None:
        self._media_path = str(media_path or "")
        self._is_video = bool(is_video)
        self._language_code = language_code if language_code in {"ru", "en"} else "ru"
        self._cancel_flag.clear()

    def request_cancel(self) -> None:
        self._cancel_flag.set()

    def _is_cancelled(self) -> bool:
        return self._cancel_flag.is_set()

    def _tr(self, key: str) -> str:
        return t(key, self._language_code)

    def run(self) -> None:
        try:
            if self._is_video:
                result = self._load_video_media(self._media_path)
            else:
                result = self._load_image_media(self._media_path)
            self.finished.emit(result)
        except Exception as exc:
            details = str(exc).strip() or repr(exc)
            self.error.emit(details)

    def _load_image_media(self, media_path: str) -> dict:
        if self._is_cancelled():
            return {"cancelled": True}
        self.progress.emit(10, self._tr("worker_media_loading_image"))
        frame = load_rgb_image(media_path)
        if self._is_cancelled():
            return {"cancelled": True}
        self.progress.emit(100, self._tr("worker_media_loading_done"))
        return {
            "path": media_path,
            "is_video": False,
            "fps": DEFAULT_FPS,
            "frames": [frame],
        }

    def _load_video_media(self, media_path: str) -> dict:
        if is_numbered_image_sequence(media_path):
            paths = resolve_numbered_image_sequence(media_path)
            if not paths:
                raise RuntimeError(self._tr("worker_video_frames_failed"))

            total = len(paths)
            frames = []
            last_percent = -1
            for index, path in enumerate(paths, start=1):
                if self._is_cancelled():
                    return {"cancelled": True}
                frame = load_rgb_image(path)
                frames.append(frame)
                percent = int(index * 100 / max(total, 1))
                if percent != last_percent:
                    last_percent = percent
                    self.progress.emit(
                        percent,
                        self._tr("worker_media_loading_sequence").format(current=index, total=total),
                    )

            if not frames:
                raise RuntimeError(self._tr("worker_video_frames_failed"))

            return {
                "path": media_path,
                "is_video": True,
                "fps": DEFAULT_FPS,
                "frames": frames,
            }

        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            raise RuntimeError(self._tr("worker_open_video_failed") + f" {media_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frames = []
            last_percent = -1
            frame_idx = 0

            while True:
                if self._is_cancelled():
                    return {"cancelled": True}
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if total > 0:
                    percent = int(frame_idx * 100 / total)
                    if percent != last_percent:
                        last_percent = percent
                        self.progress.emit(
                            percent,
                            self._tr("worker_media_loading_video").format(current=frame_idx, total=total),
                        )
        finally:
            cap.release()

        if not frames:
            raise RuntimeError(self._tr("worker_video_frames_failed"))

        if total <= 0:
            self.progress.emit(100, self._tr("worker_media_loading_done"))

        return {
            "path": media_path,
            "is_video": True,
            "fps": float(fps),
            "frames": frames,
        }