"""CorridorKey node properties panel."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.node_graph.node_panel_mixin import NodePanelMixin


CORRIDORKEY_PRESET_VALUES: dict[str, dict[str, object]] = {
    "preview": {
        "despill_strength": 0.4,
        "despeckle": False,
        "despeckle_size": 200,
        "refiner_strength": 0.85,
        "use_refiner": False,
    },
    "balanced": {
        "despill_strength": 0.5,
        "despeckle": False,
        "despeckle_size": 400,
        "refiner_strength": 1.0,
        "use_refiner": True,
    },
    "max": {
        "despill_strength": 0.6,
        "despeckle": True,
        "despeckle_size": 400,
        "refiner_strength": 1.1,
        "use_refiner": True,
    },
    "ultra": {
        "despill_strength": 0.7,
        "despeckle": True,
        "despeckle_size": 600,
        "refiner_strength": 1.2,
        "use_refiner": True,
    },
    "green_outfit": {
        "despill_strength": 0.15,
        "despeckle": False,
        "despeckle_size": 200,
        "refiner_strength": 1.0,
        "use_refiner": True,
    },
}

CORRIDORKEY_PRESET_LABEL_KEYS: dict[str, str] = {
    "preview": "corridorkey_preset_preview",
    "balanced": "corridorkey_preset_balanced",
    "max": "corridorkey_preset_max",
    "ultra": "corridorkey_preset_ultra",
    "green_outfit": "corridorkey_preset_green_outfit",
    "custom": "corridorkey_preset_custom",
}

CORRIDORKEY_PRESET_HELP_KEYS: dict[str, str] = {
    "preview": "corridorkey_preset_help_preview",
    "balanced": "corridorkey_preset_help_balanced",
    "max": "corridorkey_preset_help_max",
    "ultra": "corridorkey_preset_help_ultra",
    "green_outfit": "corridorkey_preset_help_green_outfit",
    "custom": "corridorkey_preset_help_custom",
}

CORRIDORKEY_PRESET_ORDER: list[str] = ["preview", "balanced", "max", "ultra", "green_outfit", "custom"]


class _CorridorKeyDownloadWorker(QObject):
    """Prepare CorridorKey checkpoint in background thread."""

    progress = Signal(int, str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, screen_color: str = "green") -> None:
        super().__init__()
        self.screen_color = screen_color

    def run(self) -> None:
        from app.services.corridorkey_service import CorridorKeyService

        try:
            checkpoint_path = CorridorKeyService.ensure_checkpoint_available(
                self.progress.emit,
                screen_color=self.screen_color,
            )
            self.finished.emit(str(checkpoint_path))
        except Exception as exc:
            details = str(exc).strip() or repr(exc)
            self.error.emit(details)


class CorridorKeyPropertiesPanel(QWidget, NodePanelMixin):
    """Compact CorridorKey controls used inside node properties."""

    _start_download = Signal()
    checkpoint_status_changed = Signal()

    def __init__(self, translate: Callable[[str], str], parent=None) -> None:
        super().__init__(parent)
        self._tr = translate
        self.setObjectName("corridorkeyPropertiesPanel")
        # Keep panel width stable so switching node types does not shift the properties pane.
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self._panel_width = 324
        self.setFixedWidth(self._panel_width)
        self.setStyleSheet(
            "QWidget#corridorkeyPropertiesPanel { background: #10151d; }"
            "QLabel { color: #d8dee7; }"
        )

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(6, 6, 6, 6)
        self.root.setSpacing(8)

        self._label_width = 96
        # Keep all right-column controls visually consistent without forcing panel width.
        self._field_height = 28
        self._panel_padding = 6
        self._form_column_spacing = 12
        self._value_field_width = 72
        self._slider_gap = 6
        self._width_safety_margin = 15
        self._right_column_width = (
            self._panel_width
            - (self._panel_padding * 2)
            - self._label_width
            - self._form_column_spacing
            - self._width_safety_margin
        )
        self._slider_width = self._right_column_width - self._slider_gap - self._value_field_width
        self._combo_width = self._right_column_width
        self._slider_sync_in_progress = False


        self.main_section, self.main_section_title, self.main_form = self._create_section(expanded=True)
        self.matte_section, self.matte_section_title, self.matte_form = self._create_section(expanded=True)
        self.edge_spill_section, self.edge_spill_section_title, self.edge_spill_form = self._create_section(expanded=True)
        self.advanced_section, self.advanced_section_title, self.advanced_form = self._create_section(expanded=False)

        # Despill strength (float 0-10)
        self.preset_label = QLabel(self)
        self.preset_combo = QComboBox(self)
        self.preset_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.preset_combo.setFixedWidth(self._combo_width)
        self.preset_combo.setFixedHeight(self._field_height)
        self._apply_reference_combo_style(self.preset_combo)

        self.alpha_hint_mode_label = QLabel(self)
        self.alpha_hint_mode_combo = QComboBox(self)
        self.alpha_hint_mode_combo.addItem("Auto", "auto")
        self.alpha_hint_mode_combo.addItem("Batch", "batch")
        self.alpha_hint_mode_combo.addItem("Staged", "staged")
        self.alpha_hint_mode_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.alpha_hint_mode_combo.setFixedWidth(self._combo_width)
        self.alpha_hint_mode_combo.setFixedHeight(self._field_height)
        self._apply_reference_combo_style(self.alpha_hint_mode_combo)

        self.input_colorspace_label = QLabel(self)
        self.input_colorspace_combo = QComboBox(self)
        self.input_colorspace_combo.addItem("Auto", "auto")
        self.input_colorspace_combo.addItem("sRGB", "srgb")
        self.input_colorspace_combo.addItem("Linear", "linear")
        self.input_colorspace_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.input_colorspace_combo.setFixedWidth(self._combo_width)
        self.input_colorspace_combo.setFixedHeight(self._field_height)
        self._apply_reference_combo_style(self.input_colorspace_combo)

        self.screen_color_label = QLabel(self)
        self.screen_color_combo = QComboBox(self)
        self.screen_color_combo.addItem("Auto", "auto")
        self.screen_color_combo.addItem("Green", "green")
        self.screen_color_combo.addItem("Blue", "blue")
        self.screen_color_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.screen_color_combo.setFixedWidth(self._combo_width)
        self.screen_color_combo.setFixedHeight(self._field_height)
        self._apply_reference_combo_style(self.screen_color_combo)

        self.hint_dilate_radius_label = QLabel(self)
        self.hint_dilate_radius_spin = QSpinBox(self)
        self.hint_dilate_radius_spin.setRange(0, 100)
        self.hint_dilate_radius_spin.setSingleStep(1)
        self.hint_dilate_radius_field, self.hint_dilate_radius_slider = self._make_int_slider_field(
            self.hint_dilate_radius_spin,
            min_value=0,
            max_value=100,
        )

        self.despill_strength_label = QLabel(self)
        self.despill_strength_spin = QDoubleSpinBox(self)
        self.despill_strength_spin.setRange(0.0, 1.0)
        self.despill_strength_spin.setSingleStep(0.01)
        self.despill_strength_spin.setDecimals(2)
        self.despill_strength_field, self.despill_strength_slider = self._make_slider_field(
            self.despill_strength_spin,
            min_value=0.0,
            max_value=10.0,
            slider_scale=100,
        )

        # Despeckle (checkbox)
        self.despeckle_label = QLabel(self)
        self.despeckle_check = QCheckBox(self)
        self.despeckle_field = QWidget(self)
        self.despeckle_field.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.despeckle_field.setFixedWidth(self._right_column_width)
        self.despeckle_field.setFixedHeight(self._field_height)
        self.despeckle_field_layout = QHBoxLayout(self.despeckle_field)
        self.despeckle_field_layout.setContentsMargins(0, 0, 0, 0)
        self.despeckle_field_layout.setSpacing(6)
        self.despeckle_field_layout.addWidget(self.despeckle_check, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.despeckle_inline_label = QLabel(self.despeckle_field)
        self.despeckle_inline_label.setStyleSheet("color: #c7ccd5; font-size: 11px; font-weight: 600;")
        self.despeckle_field_layout.addWidget(self.despeckle_inline_label, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.despeckle_field_layout.addStretch(1)
        self._apply_reference_checkbox_style(self.despeckle_check)

        # Despeckle size (int 1-10)
        self.despeckle_size_label = QLabel(self)
        self.despeckle_size_spin = QSpinBox(self)
        self.despeckle_size_spin.setRange(1, 5000)
        self.despeckle_size_spin.setSingleStep(10)
        self.despeckle_size_field, self.despeckle_size_slider = self._make_int_slider_field(
            self.despeckle_size_spin,
            min_value=1,
            max_value=5000,
        )

        # Refiner strength (float 0-2)
        self.refiner_strength_label = QLabel(self)
        self.refiner_strength_spin = QDoubleSpinBox(self)
        self.refiner_strength_spin.setRange(0.0, 2.0)
        self.refiner_strength_spin.setSingleStep(0.1)
        self.refiner_strength_spin.setDecimals(2)  # Changed from 1 to 2 for better precision
        self.refiner_strength_field, self.refiner_strength_slider = self._make_slider_field(
            self.refiner_strength_spin,
            min_value=0.0,
            max_value=2.0,
            slider_scale=100,
        )

        self.matte_clip_black_label = QLabel(self)
        self.matte_clip_black_spin = QDoubleSpinBox(self)
        self.matte_clip_black_spin.setRange(0.0, 1.0)
        self.matte_clip_black_spin.setSingleStep(0.01)
        self.matte_clip_black_spin.setDecimals(2)
        self.matte_clip_black_field, self.matte_clip_black_slider = self._make_slider_field(
            self.matte_clip_black_spin,
            min_value=0.0,
            max_value=1.0,
            slider_scale=100,
        )

        self.matte_clip_white_label = QLabel(self)
        self.matte_clip_white_spin = QDoubleSpinBox(self)
        self.matte_clip_white_spin.setRange(0.0, 1.0)
        self.matte_clip_white_spin.setSingleStep(0.01)
        self.matte_clip_white_spin.setDecimals(2)
        self.matte_clip_white_spin.setValue(1.0)
        self.matte_clip_white_field, self.matte_clip_white_slider = self._make_slider_field(
            self.matte_clip_white_spin,
            min_value=0.0,
            max_value=1.0,
            slider_scale=100,
        )

        self.matte_shrink_grow_label = QLabel(self)
        self.matte_shrink_grow_spin = QDoubleSpinBox(self)
        self.matte_shrink_grow_spin.setRange(-20.0, 20.0)
        self.matte_shrink_grow_spin.setSingleStep(0.5)
        self.matte_shrink_grow_spin.setDecimals(1)
        self.matte_shrink_grow_field, self.matte_shrink_grow_slider = self._make_slider_field(
            self.matte_shrink_grow_spin,
            min_value=-20.0,
            max_value=20.0,
            slider_scale=10,
        )

        self.matte_edge_blur_label = QLabel(self)
        self.matte_edge_blur_spin = QDoubleSpinBox(self)
        self.matte_edge_blur_spin.setRange(0.0, 20.0)
        self.matte_edge_blur_spin.setSingleStep(0.5)
        self.matte_edge_blur_spin.setDecimals(1)
        self.matte_edge_blur_field, self.matte_edge_blur_slider = self._make_slider_field(
            self.matte_edge_blur_spin,
            min_value=0.0,
            max_value=20.0,
            slider_scale=10,
        )

        self.matte_gamma_label = QLabel(self)
        self.matte_gamma_spin = QDoubleSpinBox(self)
        self.matte_gamma_spin.setRange(0.2, 3.0)
        self.matte_gamma_spin.setSingleStep(0.05)
        self.matte_gamma_spin.setDecimals(2)
        self.matte_gamma_spin.setValue(1.0)
        self.matte_gamma_field, self.matte_gamma_slider = self._make_slider_field(
            self.matte_gamma_spin,
            min_value=0.2,
            max_value=3.0,
            slider_scale=100,
        )
        for matte_label in (
            self.matte_clip_black_label,
            self.matte_clip_white_label,
            self.matte_shrink_grow_label,
            self.matte_edge_blur_label,
            self.matte_gamma_label,
        ):
            matte_label.setFixedWidth(self._label_width)
            matte_label.setStyleSheet("color: #d7dde6; font-size: 13px; font-weight: 600;")

        self.temporal_smoothing_label = QLabel(self)
        self.temporal_smoothing_spin = QDoubleSpinBox(self)
        self.temporal_smoothing_spin.setRange(0.0, 1.0)
        self.temporal_smoothing_spin.setSingleStep(0.05)
        self.temporal_smoothing_spin.setDecimals(2)
        self.temporal_smoothing_field, self.temporal_smoothing_slider = self._make_slider_field(
            self.temporal_smoothing_spin,
            min_value=0.0,
            max_value=1.0,
            slider_scale=100,
        )


        # Keep spin controls visually consistent regardless of global QSS rules.
        for spin in (
            self.hint_dilate_radius_spin,
            self.despill_strength_spin,
            self.despeckle_size_spin,
            self.refiner_strength_spin,
            self.matte_clip_black_spin,
            self.matte_clip_white_spin,
            self.matte_shrink_grow_spin,
            self.matte_edge_blur_spin,
            self.matte_gamma_spin,
            self.temporal_smoothing_spin,
        ):
            spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            spin.setFixedHeight(self._field_height)
            self._apply_reference_spin_style(spin)
        # Use refiner (checkbox)
        self.use_refiner_label = QLabel(self)
        self.use_refiner_check = QCheckBox(self)
        self.use_refiner_field = QWidget(self)
        self.use_refiner_field.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.use_refiner_field.setFixedWidth(self._right_column_width)
        self.use_refiner_field.setFixedHeight(self._field_height)
        self.use_refiner_field_layout = QHBoxLayout(self.use_refiner_field)
        self.use_refiner_field_layout.setContentsMargins(0, 0, 0, 0)
        self.use_refiner_field_layout.setSpacing(6)
        self.use_refiner_field_layout.addWidget(self.use_refiner_check, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.use_refiner_inline_label = QLabel(self.use_refiner_field)
        self.use_refiner_inline_label.setStyleSheet("color: #c7ccd5; font-size: 11px; font-weight: 600;")
        self.use_refiner_field_layout.addWidget(self.use_refiner_inline_label, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.use_refiner_field_layout.addStretch(1)
        self._apply_reference_checkbox_style(self.use_refiner_check)

        self._preset_sync_in_progress = False

        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.screen_color_combo.currentIndexChanged.connect(lambda _index: self._refresh_download_button_state())
        self.despill_strength_spin.valueChanged.connect(self._sync_preset_selection)
        self.despeckle_check.toggled.connect(self._sync_preset_selection)
        self.despeckle_size_spin.valueChanged.connect(self._sync_preset_selection)
        self.refiner_strength_spin.valueChanged.connect(self._sync_preset_selection)
        self.use_refiner_check.toggled.connect(self._sync_preset_selection)

        self._add_form_row(self.main_form, self.preset_label, self.preset_combo)
        self._add_form_row(self.main_form, self.alpha_hint_mode_label, self.alpha_hint_mode_combo)
        self._add_form_row(self.main_form, self.input_colorspace_label, self.input_colorspace_combo)
        self._add_form_row(self.main_form, self.screen_color_label, self.screen_color_combo)
        self._add_form_row(self.main_form, self.hint_dilate_radius_label, self.hint_dilate_radius_field)

        self._add_form_row(self.matte_form, self.despill_strength_label, self.despill_strength_field)
        self._add_form_row(self.matte_form, self.matte_clip_black_label, self.matte_clip_black_field)
        self._add_form_row(self.matte_form, self.matte_clip_white_label, self.matte_clip_white_field)
        self._add_form_row(self.matte_form, self.matte_shrink_grow_label, self.matte_shrink_grow_field)
        self._add_form_row(self.matte_form, self.matte_edge_blur_label, self.matte_edge_blur_field)
        self._add_form_row(self.matte_form, self.matte_gamma_label, self.matte_gamma_field)

        self._add_form_row(self.edge_spill_form, self.temporal_smoothing_label, self.temporal_smoothing_field)
        self._add_form_row(self.edge_spill_form, self.refiner_strength_label, self.refiner_strength_field)
        self._add_checkbox_row(self.edge_spill_form, self.use_refiner_check, self.use_refiner_inline_label)

        self._add_checkbox_row(self.advanced_form, self.despeckle_check, self.despeckle_inline_label)
        self._add_form_row(self.advanced_form, self.despeckle_size_label, self.despeckle_size_field)

        self.download_button = QPushButton(self)
        self.download_button.clicked.connect(self._download_model)

        self.download_progress = QProgressBar(self)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_progress.setTextVisible(True)
        self.download_progress.setFixedHeight(18)
        self.download_progress.setVisible(False)

        self.root.addWidget(self.main_section)
        self.root.addWidget(self.matte_section)
        self.root.addWidget(self.edge_spill_section)
        self.root.addWidget(self.advanced_section)
        self.root.addWidget(self.download_button)
        self.root.addWidget(self.download_progress)
        self.root.addStretch(1)

        self._download_thread: QThread | None = None
        self._download_worker: _CorridorKeyDownloadWorker | None = None
        self._download_active = False

        for label in (
            self.preset_label,
            self.alpha_hint_mode_label,
            self.input_colorspace_label,
            self.screen_color_label,
            self.hint_dilate_radius_label,
            self.despill_strength_label,
            self.despeckle_label,
            self.despeckle_size_label,
            self.matte_clip_black_label,
            self.matte_clip_white_label,
            self.matte_shrink_grow_label,
            self.matte_edge_blur_label,
            self.matte_gamma_label,
            self.temporal_smoothing_label,
            self.refiner_strength_label,
            self.use_refiner_label,
        ):
            label.setFixedWidth(self._label_width)
            label.setStyleSheet("color: #c7ccd5; font-size: 11px; font-weight: 600;")

        self.retranslate_ui()

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
            "QComboBox::drop-down {"
            " border: 0;"
            " width: 20px;"
            "}"
            "QComboBox::down-arrow {"
            " image: none;"
            " width: 0;"
            " height: 0;"
            " border-left: 5px solid transparent;"
            " border-right: 5px solid transparent;"
            " border-top: 6px solid #95a2b2;"
            " margin-right: 6px;"
            "}"
            "QComboBox QAbstractItemView {"
            " background: #171d2a;"
            " color: #e2e7ee;"
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
            "QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {"
            " width: 0px;"
            " border: 0;"
            "}"
        )

    @staticmethod
    def _apply_reference_checkbox_style(check: QCheckBox) -> None:
        check.setStyleSheet(
            "QCheckBox::indicator {"
            " width: 14px;"
            " height: 14px;"
            " border-radius: 3px;"
            " border: 1px solid #384257;"
            " background: #121723;"
            "}"
            "QCheckBox::indicator:checked {"
            " border: 1px solid #57b8e9;"
            " background: #57b8e9;"
            "}"
        )

    def _create_section(self, *, expanded: bool = True) -> tuple[QWidget, QToolButton, QGridLayout]:
        wrap = QWidget(self)
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QToolButton(wrap)
        title.setCheckable(True)
        title.setChecked(bool(expanded))
        title.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        title.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        title.setStyleSheet(
            "QToolButton {"
            " color: #e6ebf2;"
            " font-size: 12px;"
            " font-weight: 700;"
            " border: 0;"
            " padding: 2px 0;"
            " text-align: left;"
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
            title.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

        title.toggled.connect(_on_toggled)
        return wrap, title, form

    def _add_form_row(self, form: QGridLayout, label: QLabel, field: QWidget) -> None:
        row = form.rowCount()
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        label.setFixedWidth(self._label_width)
        form.addWidget(label, row, 0)
        form.addWidget(field, row, 1, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    def _add_checkbox_row(self, form: QGridLayout, check: QCheckBox, text_label: QLabel) -> None:
        row = form.rowCount()
        check_wrap = QWidget(self)
        check_wrap.setFixedWidth(self._label_width)
        check_layout = QHBoxLayout(check_wrap)
        check_layout.setContentsMargins(0, 0, 0, 0)
        check_layout.setSpacing(0)
        check_layout.addStretch(1)
        check_layout.addWidget(check, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.addWidget(check_wrap, row, 0)

        label_wrap = QWidget(self)
        label_wrap.setFixedWidth(self._right_column_width)
        label_layout = QHBoxLayout(label_wrap)
        label_layout.setContentsMargins(0, 0, 0, 0)
        label_layout.setSpacing(0)
        label_layout.addWidget(text_label, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label_layout.addStretch(1)
        form.addWidget(label_wrap, row, 1, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    def _make_slider_field(
        self,
        spin: QDoubleSpinBox,
        *,
        min_value: float,
        max_value: float,
        slider_scale: int,
    ) -> tuple[QWidget, QSlider]:
        field = QWidget(self)
        field.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        field.setFixedWidth(self._right_column_width)
        field.setFixedHeight(self._field_height)
        layout = QHBoxLayout(field)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self._slider_gap)

        slider = QSlider(Qt.Orientation.Horizontal, field)
        slider.setRange(int(round(min_value * slider_scale)), int(round(max_value * slider_scale)))
        slider.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        slider.setFixedHeight(self._field_height)
        slider.setFixedWidth(self._slider_width)
        slider.setStyleSheet(
            "QSlider::groove:horizontal {"
            " height: 4px;"
            " background: #323946;"
            " border: 0;"
            " border-radius: 2px;"
            "}"
            "QSlider::sub-page:horizontal {"
            " background: #323946;"
            " border-radius: 2px;"
            "}"
            "QSlider::add-page:horizontal {"
            " background: #323946;"
            " border-radius: 2px;"
            "}"
            "QSlider::handle:horizontal {"
            " background: #e8edf4;"
            " width: 8px;"
            " margin: -2px 0;"
            " border-radius: 4px;"
            " border: 1px solid #8e99a8;"
            "}"
            "QSlider::handle:horizontal:hover {"
            " background: #f3f6fa;"
            "}"
        )

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
        field = QWidget(self)
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
        slider.setStyleSheet(
            "QSlider::groove:horizontal {"
            " height: 4px;"
            " background: #323946;"
            " border: 0;"
            " border-radius: 2px;"
            "}"
            "QSlider::sub-page:horizontal {"
            " background: #323946;"
            " border-radius: 2px;"
            "}"
            "QSlider::add-page:horizontal {"
            " background: #323946;"
            " border-radius: 2px;"
            "}"
            "QSlider::handle:horizontal {"
            " background: #e8edf4;"
            " width: 8px;"
            " margin: -2px 0;"
            " border-radius: 4px;"
            " border: 1px solid #8e99a8;"
            "}"
            "QSlider::handle:horizontal:hover {"
            " background: #f3f6fa;"
            "}"
        )

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

    def set_translator(self, translate: Callable[[str], str]) -> None:
        self._tr = translate
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.main_section_title.setText(self._tr("corridorkey_section_key_setup"))
        self.matte_section_title.setText(self._tr("corridorkey_section_matte"))
        self.edge_spill_section_title.setText(self._tr("corridorkey_section_edge_spill"))
        self.advanced_section_title.setText(self._tr("corridorkey_section_advanced"))

        self.preset_label.setText(self._tr("corridorkey_preset_label"))
        current_preset = self._current_preset_key()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for preset_name in CORRIDORKEY_PRESET_ORDER:
            label_key = CORRIDORKEY_PRESET_LABEL_KEYS[preset_name]
            help_key = CORRIDORKEY_PRESET_HELP_KEYS[preset_name]
            self.preset_combo.addItem(self._tr(label_key), preset_name)
            index = self.preset_combo.count() - 1
            self.preset_combo.setItemData(index, self._tr(help_key), Qt.ItemDataRole.ToolTipRole)
        idx = self.preset_combo.findData(current_preset)
        self.preset_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.preset_combo.blockSignals(False)

        self.alpha_hint_mode_label.setText(self._tr("corridorkey_alpha_hint_mode"))
        self.alpha_hint_mode_combo.setItemText(0, self._tr("corridorkey_alpha_hint_mode_auto"))
        self.alpha_hint_mode_combo.setItemText(1, self._tr("corridorkey_alpha_hint_mode_batch"))
        self.alpha_hint_mode_combo.setItemText(2, self._tr("corridorkey_alpha_hint_mode_staged"))
        self.input_colorspace_label.setText(self._tr("corridorkey_input_colorspace"))
        self.input_colorspace_combo.setItemText(0, self._tr("corridorkey_input_colorspace_auto"))
        self.input_colorspace_combo.setItemText(1, self._tr("corridorkey_input_colorspace_srgb"))
        self.input_colorspace_combo.setItemText(2, self._tr("corridorkey_input_colorspace_linear"))
        self.screen_color_label.setText(self._tr("corridorkey_screen_color"))
        self.screen_color_combo.setItemText(0, self._tr("corridorkey_screen_color_auto"))
        self.screen_color_combo.setItemText(1, self._tr("corridorkey_screen_color_green"))
        self.screen_color_combo.setItemText(2, self._tr("corridorkey_screen_color_blue"))
        self.hint_dilate_radius_label.setText(self._tr("corridorkey_hint_dilate_radius"))
        self.despill_strength_label.setText(self._tr("corridorkey_despill_strength"))
        self.despeckle_label.setText("")
        self.despeckle_inline_label.setText(self._tr("corridorkey_despeckle"))
        self.despeckle_size_label.setText(self._tr("corridorkey_despeckle_size"))
        self.matte_clip_black_label.setText(self._tr("corridorkey_matte_clip_black"))
        self.matte_clip_white_label.setText(self._tr("corridorkey_matte_clip_white"))
        self.matte_shrink_grow_label.setText(self._tr("corridorkey_matte_shrink_grow"))
        self.matte_edge_blur_label.setText(self._tr("corridorkey_matte_edge_blur"))
        self.matte_gamma_label.setText(self._tr("corridorkey_matte_gamma"))
        self.temporal_smoothing_label.setText(self._tr("corridorkey_temporal_smoothing"))
        self.refiner_strength_label.setText(self._tr("corridorkey_refiner_strength"))
        self.use_refiner_label.setText("")
        self.use_refiner_inline_label.setText(self._tr("corridorkey_use_refiner"))

        # Tooltips
        self.preset_label.setToolTip(self._tr("corridorkey_preset_label_tooltip"))
        self.alpha_hint_mode_label.setToolTip(self._tr("corridorkey_alpha_hint_mode_tooltip"))
        self.input_colorspace_label.setToolTip(self._tr("corridorkey_input_colorspace_tooltip"))
        self.screen_color_label.setToolTip(self._tr("corridorkey_screen_color_tooltip"))
        self.screen_color_combo.setToolTip(self._tr("corridorkey_screen_color_tooltip"))
        self.hint_dilate_radius_label.setToolTip(self._tr("corridorkey_hint_dilate_radius_tooltip"))
        self.despill_strength_label.setToolTip(self._tr("corridorkey_despill_strength_tooltip"))
        self.despeckle_label.setToolTip("")
        self.despeckle_inline_label.setToolTip(self._tr("corridorkey_despeckle_tooltip"))
        self.despeckle_check.setToolTip(self._tr("corridorkey_despeckle_tooltip"))
        self.despeckle_size_label.setToolTip(self._tr("corridorkey_despeckle_size_tooltip"))
        self.matte_clip_black_label.setToolTip(self._tr("corridorkey_matte_clip_black_tooltip"))
        self.matte_clip_white_label.setToolTip(self._tr("corridorkey_matte_clip_white_tooltip"))
        self.matte_shrink_grow_label.setToolTip(self._tr("corridorkey_matte_shrink_grow_tooltip"))
        self.matte_edge_blur_label.setToolTip(self._tr("corridorkey_matte_edge_blur_tooltip"))
        self.matte_gamma_label.setToolTip(self._tr("corridorkey_matte_gamma_tooltip"))
        self.temporal_smoothing_label.setToolTip(self._tr("corridorkey_temporal_smoothing_tooltip"))
        self.refiner_strength_label.setToolTip(self._tr("corridorkey_refiner_strength_tooltip"))
        self.use_refiner_label.setToolTip("")
        self.use_refiner_inline_label.setToolTip(self._tr("corridorkey_use_refiner_tooltip"))
        self.use_refiner_check.setToolTip(self._tr("corridorkey_use_refiner_tooltip"))
        self._update_preset_tooltip()
        self._refresh_download_button_state()









    def _current_preset_key(self) -> str:
        preset_key = self.preset_combo.currentData()
        if isinstance(preset_key, str) and preset_key:
            return preset_key
        return "balanced"

    def _set_current_preset_key(self, preset_name: str) -> None:
        index = self.preset_combo.findData(preset_name)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)

    def _apply_preset_values(self, preset_name: str) -> None:
        values = CORRIDORKEY_PRESET_VALUES.get(preset_name)
        if values is None:
            return
        self._preset_sync_in_progress = True
        try:
            self.despill_strength_spin.setValue(float(values["despill_strength"]))
            self.despeckle_check.setChecked(bool(values["despeckle"]))
            self.despeckle_size_spin.setValue(int(values["despeckle_size"]))
            self.refiner_strength_spin.setValue(float(values["refiner_strength"]))
            self.use_refiner_check.setChecked(bool(values["use_refiner"]))
            # Optional keys: present only in some presets (e.g. glass)
            if "matte_edge_blur" in values:
                self.matte_edge_blur_spin.setValue(float(values["matte_edge_blur"]))
            if "matte_gamma" in values:
                self.matte_gamma_spin.setValue(float(values["matte_gamma"]))
        finally:
            self._preset_sync_in_progress = False


    def _on_preset_changed(self, _index: int) -> None:
        if self._preset_sync_in_progress:
            return
        preset_key = self._current_preset_key()
        self._apply_preset_values(preset_key)
        self._update_preset_tooltip(preset_key)

    def _update_preset_tooltip(self, preset_name: str | None = None) -> None:
        if preset_name is None:
            preset_name = self._current_preset_key()
        help_key = CORRIDORKEY_PRESET_HELP_KEYS.get(preset_name, CORRIDORKEY_PRESET_HELP_KEYS["custom"])
        self.preset_combo.setToolTip(self._tr(help_key))

    def _sync_preset_selection(self) -> None:
        if self._preset_sync_in_progress:
            return

        current_values = {
            "despill_strength": round(float(self.despill_strength_spin.value()), 2),
            "despeckle": bool(self.despeckle_check.isChecked()),
            "despeckle_size": int(self.despeckle_size_spin.value()),
            "refiner_strength": round(float(self.refiner_strength_spin.value()), 2),
            "use_refiner": bool(self.use_refiner_check.isChecked()),
            "matte_edge_blur": round(float(self.matte_edge_blur_spin.value()), 1),
            "matte_gamma": round(float(self.matte_gamma_spin.value()), 2),
        }

        matched_preset = "custom"
        for preset_name, preset_values in CORRIDORKEY_PRESET_VALUES.items():
            # Build comparison dict: base 5 keys + any extended keys declared in preset
            preset_cmp = {
                "despill_strength": round(float(preset_values["despill_strength"]), 2),
                "despeckle": bool(preset_values["despeckle"]),
                "despeckle_size": int(preset_values["despeckle_size"]),
                "refiner_strength": round(float(preset_values["refiner_strength"]), 2),
                "use_refiner": bool(preset_values["use_refiner"]),
            }
            # Extended keys: compare only if declared in preset; otherwise skip (don't require match)
            for ext_key in ("matte_edge_blur", "matte_gamma"):
                if ext_key in preset_values:
                    preset_cmp[ext_key] = preset_values[ext_key]
                else:
                    # Remove from current_values comparison to avoid mismatch for simple presets
                    current_values.pop(ext_key, None)

            if {k: current_values.get(k) for k in preset_cmp} == preset_cmp:
                matched_preset = preset_name
                break
            # Restore extended keys for next iteration
            current_values.update({
                "matte_edge_blur": round(float(self.matte_edge_blur_spin.value()), 1),
                "matte_gamma": round(float(self.matte_gamma_spin.value()), 2),
            })

        if self._current_preset_key() == matched_preset:
            return

        self._preset_sync_in_progress = True
        try:
            self._set_current_preset_key(matched_preset)
        finally:
            self._preset_sync_in_progress = False
        self._update_preset_tooltip(matched_preset)

    def _refresh_download_button_state(self) -> None:
        screen_color = str(self.screen_color_combo.currentData() or "green")
        if self._is_cloud_mode():
            models_info = getattr(self, "_cloud_models_info", {})
            if self._cloud_corridorkey_weights_ready(screen_color, models_info):
                self.download_button.setText(self._tr("corridorkey_download_button_ready_cloud"))
                self.download_button.setToolTip(self._tr("corridorkey_download_button_ready_tooltip"))
            else:
                self.download_button.setText(self._tr("corridorkey_download_button_missing"))
                self.download_button.setToolTip(self._tr("corridorkey_download_button_missing_tooltip"))
            return

        from app.services.corridorkey_service import CorridorKeyService

        status = CorridorKeyService.get_checkpoint_status(screen_color=screen_color)
        if status.get("state") == "ready":
            self.download_button.setText(self._tr("corridorkey_download_button_ready"))
            self.download_button.setToolTip(self._tr("corridorkey_download_button_ready_tooltip"))
        else:
            self.download_button.setText(self._tr("corridorkey_download_button_missing"))
            self.download_button.setToolTip(self._tr("corridorkey_download_button_missing_tooltip"))

    @staticmethod
    def _cloud_corridorkey_weights_ready(screen_color: str, models_info: dict) -> bool:
        if not isinstance(models_info, dict):
            return bool(models_info)
        has_green = bool(models_info.get("corridorkey_green"))
        has_blue = bool(models_info.get("corridorkey_blue"))
        has_detailed_flags = "corridorkey_green" in models_info or "corridorkey_blue" in models_info
        legacy_ready = bool(models_info.get("corridorkey")) and not (
            has_detailed_flags
        )
        if screen_color == "blue":
            return has_blue
        if screen_color == "auto":
            return has_green and has_blue if has_detailed_flags else False
        return has_green or legacy_ready

    def _download_model(self) -> None:
        if self._is_cloud_mode():
            self._start_cloud_download("corridorkey")
            return

        if self._download_active:
            return

        from app.services.corridorkey_service import CorridorKeyService

        screen_color = str(self.screen_color_combo.currentData() or "green")
        status = CorridorKeyService.get_checkpoint_status(screen_color=screen_color)
        if status.get("state") == "ready":
            self._refresh_download_button_state()
            QMessageBox.information(
                self,
                self._tr("info_title"),
                self._tr("corridorkey_weights_already_present_ui"),
            )
            self.checkpoint_status_changed.emit()
            return

        self._ensure_download_worker()
        if self._download_worker is None:
            return
        self._download_worker.screen_color = screen_color

        self._download_active = True
        self.download_button.setEnabled(False)
        self.download_progress.setVisible(True)
        self.download_progress.setValue(0)
        self.download_progress.setFormat("%p%")
        self._start_download.emit()

    def _ensure_download_worker(self) -> None:
        if self._download_worker is not None and self._download_thread is not None:
            return

        self._download_thread = QThread(self)
        self._download_worker = _CorridorKeyDownloadWorker(str(self.screen_color_combo.currentData() or "green"))
        self._download_worker.moveToThread(self._download_thread)

        self._start_download.connect(self._download_worker.run, Qt.ConnectionType.QueuedConnection)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.error.connect(self._on_download_error)
        self._download_worker.finished.connect(self._download_thread.quit)
        self._download_worker.error.connect(self._download_thread.quit)
        self._download_thread.finished.connect(self._on_download_thread_finished)
        self._download_thread.finished.connect(self._download_thread.deleteLater)
        self._download_thread.start()

    def _on_download_progress(self, percent: int, message: str) -> None:
        self.download_progress.setVisible(True)
        self.download_progress.setValue(max(0, min(100, int(percent))))
        self.download_progress.setToolTip(message)

    def _on_download_finished(self, _checkpoint_path: str) -> None:
        self._download_active = False
        self._refresh_download_button_state()
        self.download_button.setEnabled(True)
        self.download_progress.setValue(100)
        self.download_progress.setVisible(False)
        self.checkpoint_status_changed.emit()
        QMessageBox.information(
            self,
            self._tr("info_title"),
            self._tr("corridorkey_weights_downloaded_ui"),
        )

    def _on_download_error(self, error_message: str) -> None:
        self._download_active = False
        self._refresh_download_button_state()
        self.download_button.setEnabled(True)
        self.download_progress.setValue(0)
        self.download_progress.setVisible(False)
        QMessageBox.critical(
            self,
            self._tr("inference_error_title"),
            self._tr("corridorkey_weights_download_failed").format(error=error_message),
        )

    def _on_download_thread_finished(self) -> None:
        if self._download_worker is not None:
            self._download_worker.deleteLater()
        self._download_worker = None
        self._download_thread = None

    def closeEvent(self, event) -> None:
        if self._download_thread is not None and self._download_thread.isRunning():
            self._download_thread.quit()
            self._download_thread.wait()
        super().closeEvent(event)

    def load_from_properties(self, props: dict) -> None:
        """Load properties from node data dict."""
        self._refresh_download_button_state()

        has_preset = "preset" in props
        preset = str(props.get("preset", "custom" if not has_preset else "balanced")).strip().lower()
        if preset not in CORRIDORKEY_PRESET_VALUES and preset != "custom":
            preset = "custom"

        idx = self.preset_combo.findData(preset)
        self.preset_combo.setCurrentIndex(idx if idx >= 0 else self.preset_combo.findData("balanced"))

        alpha_hint_mode = str(props.get("alpha_hint_mode", "auto")).strip().lower()
        if alpha_hint_mode not in {"auto", "batch", "staged"}:
            alpha_hint_mode = "auto"
        idx = self.alpha_hint_mode_combo.findData(alpha_hint_mode)
        self.alpha_hint_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)

        input_colorspace = str(props.get("input_colorspace", "auto")).strip().lower()
        if input_colorspace not in {"auto", "srgb", "linear"}:
            input_colorspace = "auto"
        idx = self.input_colorspace_combo.findData(input_colorspace)
        self.input_colorspace_combo.setCurrentIndex(idx if idx >= 0 else 0)

        screen_color = str(props.get("screen_color", "green")).strip().lower()
        if screen_color not in {"auto", "green", "blue"}:
            screen_color = "green"
        idx = self.screen_color_combo.findData(screen_color)
        self.screen_color_combo.setCurrentIndex(idx if idx >= 0 else self.screen_color_combo.findData("green"))

        despill_strength = float(props.get("despill_strength", 0.5))
        self.despill_strength_spin.setValue(despill_strength)

        despeckle = bool(props.get("despeckle", True))
        self.despeckle_check.setChecked(despeckle)

        despeckle_size = int(props.get("despeckle_size", 400))
        self.despeckle_size_spin.setValue(despeckle_size)

        self.matte_clip_black_spin.setValue(float(props.get("matte_clip_black", 0.0)))
        self.matte_clip_white_spin.setValue(float(props.get("matte_clip_white", 1.0)))
        self.matte_shrink_grow_spin.setValue(float(props.get("matte_shrink_grow", 0.0)))
        self.matte_edge_blur_spin.setValue(float(props.get("matte_edge_blur", 0.0)))
        self.matte_gamma_spin.setValue(float(props.get("matte_gamma", 1.0)))
        self.temporal_smoothing_spin.setValue(float(props.get("temporal_smoothing", 0.0)))

        refiner_strength = float(props.get("refiner_strength", 1.0))
        self.refiner_strength_spin.setValue(refiner_strength)

        use_refiner = bool(props.get("use_refiner", True))
        self.use_refiner_check.setChecked(use_refiner)

        self.hint_dilate_radius_spin.setValue(int(props.get("hint_dilate_radius", 0)))

        if has_preset and preset in CORRIDORKEY_PRESET_VALUES:
            self._apply_preset_values(preset)
        else:
            self._sync_preset_selection()
        self._update_preset_tooltip()

    def write_to_properties(self, props: dict) -> None:
        """Write properties to node data dict."""
        props["preset"] = self._current_preset_key()
        props["alpha_hint_mode"] = self.alpha_hint_mode_combo.currentData() or "auto"
        props["input_colorspace"] = self.input_colorspace_combo.currentData() or "auto"
        props["screen_color"] = self.screen_color_combo.currentData() or "green"
        props["despill_strength"] = self.despill_strength_spin.value()
        props["despeckle"] = self.despeckle_check.isChecked()
        props["despeckle_size"] = self.despeckle_size_spin.value()
        props["matte_clip_black"] = self.matte_clip_black_spin.value()
        props["matte_clip_white"] = self.matte_clip_white_spin.value()
        props["matte_shrink_grow"] = self.matte_shrink_grow_spin.value()
        props["matte_edge_blur"] = self.matte_edge_blur_spin.value()
        props["matte_gamma"] = self.matte_gamma_spin.value()
        props["temporal_smoothing"] = self.temporal_smoothing_spin.value()
        props["refiner_strength"] = self.refiner_strength_spin.value()
        props["use_refiner"] = self.use_refiner_check.isChecked()
        props["output_mode"] = "processed"
        props["hint_dilate_radius"] = int(self.hint_dilate_radius_spin.value())
