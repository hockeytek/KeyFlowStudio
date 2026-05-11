"""Shared layout and style helpers for node property panels.

Mix this class into any ``QWidget``-subclass panel, always after ``QWidget``
in the MRO, e.g.::

    class MyPropertiesPanel(QWidget, NodePanelMixin):
        def __init__(self, ...):
            super().__init__(parent)
            self._init_panel_layout()
            ...

Subclasses must call ``_init_panel_layout()`` early in their ``__init__``
before calling any other mixin method.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class NodePanelMixin:
    """Shared layout/style helpers for all node property panels.

    Provides slider fields, sectioned GridLayout, and consistent styling
    matching the CorridorKey panel design language.
    """

    # ── Dimension constants ───────────────────────────────────────────────────

    _PANEL_WIDTH: int = 324
    _LABEL_WIDTH: int = 96
    _FIELD_HEIGHT: int = 28
    _PANEL_PADDING: int = 6
    _FORM_COLUMN_SPACING: int = 12
    _VALUE_FIELD_WIDTH: int = 72
    _SLIDER_GAP: int = 6
    _WIDTH_SAFETY_MARGIN: int = 15

    # ── Slider stylesheet ─────────────────────────────────────────────────────

    _SLIDER_STYLE: str = (
        "QSlider::groove:horizontal {"
        " height: 4px; background: #323946; border: 0; border-radius: 2px;"
        "}"
        "QSlider::sub-page:horizontal { background: #323946; border-radius: 2px; }"
        "QSlider::add-page:horizontal { background: #323946; border-radius: 2px; }"
        "QSlider::handle:horizontal {"
        " background: #e8edf4; width: 8px; margin: -2px 0;"
        " border-radius: 4px; border: 1px solid #8e99a8;"
        "}"
        "QSlider::handle:horizontal:hover { background: #f3f6fa; }"
    )

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_panel_layout(self) -> None:
        """Compute derived dimension fields. Call once in ``__init__``."""
        self._label_width: int = self._LABEL_WIDTH
        self._field_height: int = self._FIELD_HEIGHT
        self._panel_padding: int = self._PANEL_PADDING
        self._form_column_spacing: int = self._FORM_COLUMN_SPACING
        self._value_field_width: int = self._VALUE_FIELD_WIDTH
        self._slider_gap: int = self._SLIDER_GAP
        self._right_column_width: int = (
            self._PANEL_WIDTH
            - (self._PANEL_PADDING * 2)
            - self._LABEL_WIDTH
            - self._FORM_COLUMN_SPACING
            - self._WIDTH_SAFETY_MARGIN
        )
        self._slider_width: int = (
            self._right_column_width - self._SLIDER_GAP - self._VALUE_FIELD_WIDTH
        )
        self._slider_sync_in_progress: bool = False

    # ── Static style helpers ──────────────────────────────────────────────────

    @staticmethod
    def _apply_reference_combo_style(combo: QComboBox) -> None:
        combo.setStyleSheet(
            "QComboBox {"
            " border: 1px solid #2b3140;"
            " border-radius: 5px;"
            " background: #171d2a;"
            " color: #e2e7ee;"
            " padding: 0 7px;"
            " min-height: 20px;"
            "}"
            "QComboBox::drop-down { border: 0; width: 20px; }"
            "QComboBox::down-arrow {"
            " image: none; width: 0; height: 0;"
            " border-left: 5px solid transparent;"
            " border-right: 5px solid transparent;"
            " border-top: 6px solid #95a2b2;"
            " margin-right: 6px;"
            "}"
            "QComboBox QAbstractItemView {"
            " background: #171d2a; color: #e2e7ee;"
            " border: 1px solid #2b3140;"
            " selection-background-color: #263047;"
            "}"
        )

    @staticmethod
    def _apply_reference_spin_style(spin: QAbstractSpinBox) -> None:
        spin.setStyleSheet(
            "QSpinBox, QDoubleSpinBox {"
            " border: 1px solid #090d14;"
            " border-radius: 5px;"
            " background: #131923;"
            " color: #eef2f7;"
            " padding: 0 7px;"
            " font-size: 10px;"
            " font-weight: 700;"
            "}"
            "QSpinBox:focus, QDoubleSpinBox:focus {"
            " border: 1px solid #7b889b;"
            "}"
            "QSpinBox::up-button, QSpinBox::down-button,"
            "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {"
            " width: 0px; border: 0;"
            "}"
        )

    @staticmethod
    def _apply_reference_checkbox_style(check: QCheckBox) -> None:
        check.setStyleSheet(
            "QCheckBox::indicator {"
            " width: 14px; height: 14px;"
            " border-radius: 3px;"
            " border: 1px solid #384257;"
            " background: #121723;"
            "}"
            "QCheckBox::indicator:checked {"
            " border: 1px solid #57b8e9;"
            " background: #57b8e9;"
            "}"
        )

    # ── Section / form layout helpers ─────────────────────────────────────────

    def _create_section(
        self, *, expanded: bool = True
    ) -> tuple[QWidget, QToolButton, QGridLayout]:
        """Create a collapsible section with a title button and a grid form."""
        wrap = QWidget(self)  # type: ignore[arg-type]
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QToolButton(wrap)
        title.setCheckable(True)
        title.setChecked(bool(expanded))
        title.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        title.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        title.setStyleSheet(
            "QToolButton {"
            " color: #e6ebf2; font-size: 12px; font-weight: 700;"
            " border: 0; padding: 2px 0; text-align: left;"
            "}"
            "QToolButton::menu-indicator { image: none; }"
        )

        content = QWidget(wrap)
        content.setVisible(bool(expanded))
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

        separator = QFrame(content)
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("QFrame { color: #242a33; }")

        form = QGridLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(self._form_column_spacing)
        form.setVerticalSpacing(8)
        form.setColumnStretch(0, 0)
        form.setColumnStretch(1, 1)

        content_layout.addWidget(separator)
        content_layout.addLayout(form)
        layout.addWidget(title)
        layout.addWidget(content)

        def _on_toggled(checked: bool) -> None:
            content.setVisible(bool(checked))
            title.setArrowType(
                Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
            )

        title.toggled.connect(_on_toggled)
        return wrap, title, form

    def _add_form_row(
        self, form: QGridLayout, label: QLabel, field: QWidget
    ) -> None:
        """Append a label→field row to a section grid form."""
        row = form.rowCount()
        label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        label.setFixedWidth(self._label_width)
        label.setStyleSheet("color: #c7ccd5; font-size: 11px; font-weight: 600;")
        form.addWidget(label, row, 0)
        form.addWidget(
            field,
            row,
            1,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

    def _add_checkbox_row(
        self, form: QGridLayout, check: QCheckBox, text_label: QLabel
    ) -> None:
        """Append a checkbox row (check on left, label text on right) to a section grid form."""
        row = form.rowCount()
        check_wrap = QWidget(self)  # type: ignore[arg-type]
        check_wrap.setFixedWidth(self._label_width)
        check_layout = QHBoxLayout(check_wrap)
        check_layout.setContentsMargins(0, 0, 0, 0)
        check_layout.setSpacing(0)
        check_layout.addStretch(1)
        check_layout.addWidget(
            check, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        form.addWidget(check_wrap, row, 0)

        label_wrap = QWidget(self)  # type: ignore[arg-type]
        label_wrap.setFixedWidth(self._right_column_width)
        label_layout = QHBoxLayout(label_wrap)
        label_layout.setContentsMargins(0, 0, 0, 0)
        label_layout.setSpacing(0)
        label_layout.addWidget(
            text_label, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        label_layout.addStretch(1)
        form.addWidget(
            label_wrap,
            row,
            1,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

    # ── Slider field factories ────────────────────────────────────────────────

    def _make_slider_field(
        self,
        spin: QDoubleSpinBox,
        *,
        min_value: float,
        max_value: float,
        slider_scale: int,
    ) -> tuple[QWidget, QSlider]:
        """Fixed-width float slider field for use with ``_add_form_row`` / QGridLayout."""
        field = QWidget(self)  # type: ignore[arg-type]
        field.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        field.setFixedWidth(self._right_column_width)
        field.setFixedHeight(self._field_height)
        layout = QHBoxLayout(field)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self._slider_gap)

        slider = QSlider(Qt.Orientation.Horizontal, field)
        slider.setRange(
            int(round(min_value * slider_scale)),
            int(round(max_value * slider_scale)),
        )
        slider.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        slider.setFixedHeight(self._field_height)
        slider.setFixedWidth(self._slider_width)
        slider.setStyleSheet(self._SLIDER_STYLE)

        spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        spin.setFixedHeight(self._field_height)
        spin.setFixedWidth(self._value_field_width)
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._apply_reference_spin_style(spin)

        def _sync_from_slider(value: int) -> None:
            if self._slider_sync_in_progress:
                return
            self._slider_sync_in_progress = True
            try:
                spin.setValue(float(value) / float(slider_scale))
            finally:
                self._slider_sync_in_progress = False

        def _sync_from_spin(value: float) -> None:
            if self._slider_sync_in_progress:
                return
            self._slider_sync_in_progress = True
            try:
                slider.setValue(int(round(float(value) * float(slider_scale))))
            finally:
                self._slider_sync_in_progress = False

        slider.valueChanged.connect(_sync_from_slider)
        spin.valueChanged.connect(_sync_from_spin)
        slider.setValue(int(round(float(spin.value()) * float(slider_scale))))

        layout.addWidget(slider, 0)
        layout.addWidget(spin, 0)
        return field, slider

    def _make_int_slider_field(
        self,
        spin: QSpinBox,
        *,
        min_value: int,
        max_value: int,
    ) -> tuple[QWidget, QSlider]:
        """Fixed-width integer slider field for use with ``_add_form_row`` / QGridLayout."""
        field = QWidget(self)  # type: ignore[arg-type]
        field.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        field.setFixedWidth(self._right_column_width)
        field.setFixedHeight(self._field_height)
        layout = QHBoxLayout(field)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self._slider_gap)

        slider = QSlider(Qt.Orientation.Horizontal, field)
        slider.setRange(int(min_value), int(max_value))
        slider.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        slider.setFixedHeight(self._field_height)
        slider.setFixedWidth(self._slider_width)
        slider.setStyleSheet(self._SLIDER_STYLE)

        spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        spin.setFixedHeight(self._field_height)
        spin.setFixedWidth(self._value_field_width)
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._apply_reference_spin_style(spin)

        def _sync_from_slider(value: int) -> None:
            if self._slider_sync_in_progress:
                return
            self._slider_sync_in_progress = True
            try:
                spin.setValue(int(value))
            finally:
                self._slider_sync_in_progress = False

        def _sync_from_spin(value: int) -> None:
            if self._slider_sync_in_progress:
                return
            self._slider_sync_in_progress = True
            try:
                slider.setValue(int(value))
            finally:
                self._slider_sync_in_progress = False

        slider.valueChanged.connect(_sync_from_slider)
        spin.valueChanged.connect(_sync_from_spin)
        slider.setValue(int(spin.value()))

        layout.addWidget(slider, 0)
        layout.addWidget(spin, 0)
        return field, slider

    def _make_expanding_slider_field(
        self,
        spin: QAbstractSpinBox,
        *,
        min_value: float,
        max_value: float,
        slider_scale: int = 1,
    ) -> tuple[QWidget, QSlider]:
        """Expanding slider field for use with QFormLayout (width adapts to available space).

        Use this variant when the panel uses ``QFormLayout`` rather than the sectioned
        ``QGridLayout`` created by ``_create_section``.
        """
        field = QWidget(self)  # type: ignore[arg-type]
        field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        field.setFixedHeight(self._field_height)
        layout = QHBoxLayout(field)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self._slider_gap)

        slider = QSlider(Qt.Orientation.Horizontal, field)
        slider.setRange(
            int(round(min_value * slider_scale)),
            int(round(max_value * slider_scale)),
        )
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        slider.setFixedHeight(self._field_height)
        slider.setMinimumWidth(60)
        slider.setStyleSheet(self._SLIDER_STYLE)

        spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        spin.setFixedHeight(self._field_height)
        spin.setFixedWidth(self._value_field_width)
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._apply_reference_spin_style(spin)

        def _sync_from_slider(value: int) -> None:
            if self._slider_sync_in_progress:
                return
            self._slider_sync_in_progress = True
            try:
                if slider_scale == 1:
                    spin.setValue(int(value))
                else:
                    spin.setValue(float(value) / float(slider_scale))
            finally:
                self._slider_sync_in_progress = False

        def _sync_from_spin(value: object) -> None:
            if self._slider_sync_in_progress:
                return
            self._slider_sync_in_progress = True
            try:
                if slider_scale == 1:
                    slider.setValue(int(value))  # type: ignore[arg-type]
                else:
                    slider.setValue(int(round(float(value) * float(slider_scale))))  # type: ignore[arg-type]
            finally:
                self._slider_sync_in_progress = False

        slider.valueChanged.connect(_sync_from_slider)
        spin.valueChanged.connect(_sync_from_spin)
        if slider_scale == 1:
            slider.setValue(int(spin.value()))
        else:
            slider.setValue(int(round(float(spin.value()) * float(slider_scale))))

        layout.addWidget(slider, 1)
        layout.addWidget(spin, 0)
        return field, slider

    # ── Cloud model download helpers ──────────────────────────────────────────

    def _is_cloud_mode(self) -> bool:
        """Return True when cloud GPU is enabled and an API host is configured."""
        from app.cloud_settings import get_cloud_setting
        return bool(get_cloud_setting("cloud/enabled")) and bool(
            str(get_cloud_setting("cloud/api_host") or "").strip()
        )

    def _cloud_api_host(self) -> str:
        from app.cloud_settings import get_cloud_setting
        host = str(get_cloud_setting("cloud/api_host") or "").strip()
        return f"http://{host}:8080"

    def _start_cloud_download(self, model: str, preset: str = "") -> None:
        """Kick off a cloud-side model download. Reuses self.download_button/download_progress."""
        from app.workers.cloud_model_downloader import CloudModelDownloadWorker
        from PySide6.QtCore import QThread

        if getattr(self, "_cloud_dl_thread", None) is not None:
            return  # already running

        thread = QThread(self)  # type: ignore[arg-type]
        worker = CloudModelDownloadWorker(self._cloud_api_host(), model, preset)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_cloud_dl_progress)
        worker.already_present.connect(self._on_cloud_dl_already_present)
        worker.finished.connect(self._on_cloud_dl_finished)
        worker.error.connect(self._on_cloud_dl_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.already_present.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_cloud_dl_thread_finished)

        self._cloud_dl_thread = thread
        self._cloud_dl_worker = worker

        if hasattr(self, "download_button"):
            self.download_button.setEnabled(False)
        if hasattr(self, "download_progress"):
            self.download_progress.setVisible(True)
            self.download_progress.setValue(0)

        thread.start()

    def _on_cloud_dl_progress(self, percent: int, message: str) -> None:
        if hasattr(self, "download_progress"):
            self.download_progress.setVisible(True)
            self.download_progress.setValue(max(0, min(100, percent)))
            self.download_progress.setToolTip(message)

    def _on_cloud_dl_already_present(self) -> None:
        self._cloud_weights_ready = True  # weights confirmed on server
        if hasattr(self, "download_button"):
            self.download_button.setEnabled(True)
        if hasattr(self, "download_progress"):
            self.download_progress.setVisible(False)
        if hasattr(self, "_refresh_download_button_state"):
            self._refresh_download_button_state()
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,  # type: ignore[arg-type]
            self._tr("info_title"),
            self._tr("cloud_model_already_on_server"),
        )

    def _on_cloud_dl_finished(self, _model: str) -> None:
        self._cloud_weights_ready = True  # weights just downloaded to server
        if hasattr(self, "download_button"):
            self.download_button.setEnabled(True)
        if hasattr(self, "download_progress"):
            self.download_progress.setValue(100)
            self.download_progress.setVisible(False)
        if hasattr(self, "_refresh_download_button_state"):
            self._refresh_download_button_state()
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,  # type: ignore[arg-type]
            self._tr("info_title"),
            self._tr("cloud_model_downloaded_on_server"),
        )

    def _on_cloud_dl_error(self, message: str) -> None:
        if hasattr(self, "download_button"):
            self.download_button.setEnabled(True)
        if hasattr(self, "download_progress"):
            self.download_progress.setValue(0)
            self.download_progress.setVisible(False)
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(
            self,  # type: ignore[arg-type]
            self._tr("inference_error_title"),
            message,
        )

    def _on_cloud_dl_thread_finished(self) -> None:
        self._cloud_dl_thread = None
        self._cloud_dl_worker = None
