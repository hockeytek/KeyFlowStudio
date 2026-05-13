"""Hybrid input viewer: stacks a numpy/QPixmap label and a hardware-decoded
QVideoWidget so plain video playback uses VideoToolbox / Metal while pause,
SAM interaction, image sequences and processed previews keep going through
the existing QLabel + numpy path.

The widget is API-compatible with the previous bare ``QLabel`` for the subset
of methods MainWindow used (``setText``, ``setPixmap``, ``setAlignment``,
``setCursor``, ``setMinimumSize``, ``setMaximumSize``, ``setFixedHeight``,
``setSizePolicy``, ``setStyleSheet``, ``mousePressEvent`` assignment,
``size()`` / ``width()`` / ``height()``), so no other call sites need to be
rewritten.

Activate hardware playback with ``play_video(path, fps)``; switch back to the
overlay-friendly QLabel with ``show_label_frame()``.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QObject, QSize, QUrl, Qt, Signal
from PySide6.QtGui import QMouseEvent, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QLabel, QSizePolicy, QStackedLayout, QWidget


class _ClickableVideoWidget(QVideoWidget):
    """QVideoWidget that forwards mouse presses to a user callback."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mouse_press_cb: Optional[Callable[[QMouseEvent], None]] = None

    def set_mouse_press_callback(self, cb: Optional[Callable[[QMouseEvent], None]]) -> None:
        self._mouse_press_cb = cb

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if self._mouse_press_cb is not None:
            try:
                self._mouse_press_cb(event)
                return
            except Exception:
                pass
        super().mousePressEvent(event)


class HybridInputViewer(QWidget):
    """Drop-in replacement for the old ``input_video_label`` QLabel.

    Two pages on a QStackedLayout:
      * page 0 — ``QLabel`` (existing numpy → QPixmap path, supports SAM
        overlays, image sequences and any processed frame).
      * page 1 — ``QVideoWidget`` driven by ``QMediaPlayer`` for hardware
        decoded MP4/MOV/MKV playback.

    Slider sync uses media position (ms) → frame index via fps, exposed via
    ``frame_changed`` and ``playback_finished`` signals.
    """

    frame_changed = Signal(int)        # current frame index (best-effort)
    playback_finished = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.setStackingMode(QStackedLayout.StackingMode.StackOne)

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._video = _ClickableVideoWidget(self)
        self._video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        try:
            self._video.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        except Exception:
            pass

        self._stack.addWidget(self._label)   # index 0
        self._stack.addWidget(self._video)   # index 1
        self._stack.setCurrentIndex(0)

        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._audio.setMuted(True)  # preview window: no audio by default
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._video)

        self._fps: float = 30.0
        self._user_mouse_press_cb: Optional[Callable[[QMouseEvent], None]] = None
        self._video_active: bool = False
        self._suppress_position_signal: bool = False

        self._player.positionChanged.connect(self._on_position_changed)
        self._player.mediaStatusChanged.connect(self._on_media_status_changed)

    # ------------------------------------------------------------------
    # Public API used by MainWindow / coordinators
    # ------------------------------------------------------------------
    @property
    def label(self) -> QLabel:
        """Underlying ``QLabel`` for direct overlay/painter usage."""
        return self._label

    @property
    def video_widget(self) -> QVideoWidget:
        return self._video

    @property
    def is_video_active(self) -> bool:
        return self._video_active

    def set_audio_muted(self, muted: bool) -> None:
        self._audio.setMuted(bool(muted))

    def set_audio_volume(self, volume_0_to_1: float) -> None:
        try:
            self._audio.setVolume(max(0.0, min(1.0, float(volume_0_to_1))))
        except Exception:
            pass

    def show_label_frame(self) -> None:
        """Switch back to the QLabel page; pause + reset the media player."""
        if self._video_active:
            try:
                self._player.pause()
            except Exception:
                pass
        self._video_active = False
        self._stack.setCurrentIndex(0)

    def play_video(self, path: str, fps: float, *, start_position_ms: int = 0,
                   rate: float = 1.0) -> bool:
        """Start hardware-accelerated playback of a real video file.

        Returns False if the source could not be set (caller should fall back
        to the QLabel render path).
        """
        if not path:
            return False
        url = QUrl.fromLocalFile(path)
        try:
            self._player.setSource(url)
        except Exception:
            return False
        self._fps = float(fps) if fps and fps > 0 else 30.0
        try:
            self._player.setPlaybackRate(float(rate) or 1.0)
        except Exception:
            pass
        if start_position_ms > 0:
            try:
                self._player.setPosition(int(start_position_ms))
            except Exception:
                pass
        self._video_active = True
        self._stack.setCurrentIndex(1)
        try:
            self._player.play()
        except Exception:
            self.show_label_frame()
            return False
        return True

    def pause_video(self) -> None:
        if self._video_active:
            try:
                self._player.pause()
            except Exception:
                pass

    def resume_video(self) -> None:
        if self._video_active:
            try:
                self._player.play()
            except Exception:
                pass

    def stop_video(self) -> None:
        try:
            self._player.stop()
        except Exception:
            pass
        self.show_label_frame()

    def seek_to_frame(self, frame_index: int) -> None:
        if not self._video_active:
            return
        ms = int(max(0, frame_index) * 1000.0 / max(1.0, self._fps))
        self._suppress_position_signal = True
        try:
            self._player.setPosition(ms)
        finally:
            self._suppress_position_signal = False

    def current_frame_index(self) -> int:
        if not self._video_active:
            return 0
        return int(self._player.position() * self._fps / 1000.0)

    # ------------------------------------------------------------------
    # QLabel-shaped pass-through API (for compatibility with old call sites)
    # ------------------------------------------------------------------
    def setPixmap(self, pixmap: QPixmap) -> None:  # noqa: N802 - Qt API
        if self._video_active:
            self.show_label_frame()
        self._label.setPixmap(pixmap)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        if self._video_active:
            self.show_label_frame()
        self._label.setText(text)

    def setAlignment(self, flag) -> None:  # noqa: N802 - Qt API
        self._label.setAlignment(flag)

    def setCursor(self, cursor) -> None:  # noqa: N802 - Qt API
        super().setCursor(cursor)
        self._label.setCursor(cursor)
        self._video.setCursor(cursor)

    def setStyleSheet(self, style: str) -> None:  # noqa: N802 - Qt API
        # Apply visual styling (border, background) to the label page; the
        # video widget paints its frames natively and ignores QSS anyway.
        self._label.setStyleSheet(style)

    def setMinimumSize(self, *args) -> None:  # noqa: N802 - Qt API
        super().setMinimumSize(*args)

    def setMaximumSize(self, *args) -> None:  # noqa: N802 - Qt API
        super().setMaximumSize(*args)

    def setFixedHeight(self, h: int) -> None:  # noqa: N802 - Qt API
        super().setFixedHeight(h)

    def setSizePolicy(self, *args) -> None:  # noqa: N802 - Qt API
        super().setSizePolicy(*args)
        if args:
            try:
                self._label.setSizePolicy(*args)
                self._video.setSizePolicy(*args)
            except Exception:
                pass

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return self._label.sizeHint()

    # Forward mouse press assignment (``hybrid.mousePressEvent = cb``) used by
    # the SAM click handler. We capture the bound callback and dispatch to
    # whichever page is currently showing.
    def __setattr__(self, name: str, value) -> None:
        if name == "mousePressEvent" and callable(value):
            cb = value
            object.__setattr__(self, "_user_mouse_press_cb", cb)
            try:
                self._label.mousePressEvent = cb  # type: ignore[assignment]
            except Exception:
                pass
            try:
                self._video.set_mouse_press_callback(cb)
            except Exception:
                pass
            return
        super().__setattr__(name, value)

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------
    def _on_position_changed(self, position_ms: int) -> None:
        if self._suppress_position_signal or not self._video_active:
            return
        idx = int(position_ms * self._fps / 1000.0)
        self.frame_changed.emit(idx)

    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.playback_finished.emit()
