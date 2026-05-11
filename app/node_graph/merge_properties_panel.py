"""Merge node properties panel.

Provides blend mode selection (32 modes matching NUKE's Merge node) and
mask/mix controls for the Merge node.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QSignalBlocker

from app.node_graph.node_panel_mixin import NodePanelMixin

# Internal mode key → i18n key  (alphabetical order, matching NUKE)
# Keys are the canonical mode strings stored in node properties.
_MODE_I18N_KEYS = [
    # Porter-Duff compositing operators
    ("over",          "merge_mode_over"),
    ("under",         "merge_mode_under"),
    ("atop",          "merge_mode_atop"),
    ("in",            "merge_mode_in"),
    ("out",           "merge_mode_out"),
    ("mask",          "merge_mode_mask"),
    ("stencil",       "merge_mode_stencil"),
    ("matte",         "merge_mode_matte"),
    ("xor",           "merge_mode_xor"),
    ("copy",          "merge_mode_copy"),
    ("conjoint-over", "merge_mode_conjoint_over"),
    ("disjoint-over", "merge_mode_disjoint_over"),
    # Additive
    ("plus",          "merge_mode_plus"),
    ("hypot",         "merge_mode_hypot"),
    # Arithmetic blend modes
    ("average",       "merge_mode_average"),
    ("multiply",      "merge_mode_multiply"),
    ("divide",        "merge_mode_divide"),
    ("minus",         "merge_mode_minus"),
    ("from",          "merge_mode_from"),
    # Contrast / light blend modes
    ("screen",        "merge_mode_screen"),
    ("overlay",       "merge_mode_overlay"),
    ("hard-light",    "merge_mode_hard_light"),
    ("soft-light",    "merge_mode_soft_light"),
    # Difference
    ("difference",    "merge_mode_difference"),
    ("exclusion",     "merge_mode_exclusion"),
    # Darken / lighten
    ("min",           "merge_mode_min"),
    ("max",           "merge_mode_max"),
    # Dodge / burn
    ("color-burn",    "merge_mode_color_burn"),
    ("color-dodge",   "merge_mode_color_dodge"),
    # Mathematical
    ("reflect",       "merge_mode_reflect"),
    ("geometric",     "merge_mode_geometric"),
    ("pinlight",      "merge_mode_pinlight"),
]

# Legacy aliases: old mode key → canonical key used in _MODE_I18N_KEYS
_MODE_ALIASES = {
    "add":      "plus",
    "subtract": "minus",
    "darken":   "min",
    "lighten":  "max",
}


class MergePropertiesPanel(QWidget, NodePanelMixin):
    """Compact Merge blend controls used inside node properties."""

    def __init__(self, translate: Callable[[str], str], parent=None) -> None:
        super().__init__(parent)
        self._tr = translate
        self._init_panel_layout()

        self.setObjectName("mergePropertiesPanel")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.setFixedWidth(self._PANEL_WIDTH)
        self.setStyleSheet(
            "QWidget#mergePropertiesPanel { background: #10151d; }"
            "QLabel { color: #d8dee7; }"
        )

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(6, 6, 6, 6)
        self.root.setSpacing(8)

        # ── Section: Blend ────────────────────────────────────────────────────
        self.blend_section, self.blend_section_title, self.blend_form = self._create_section(expanded=True)

        self.mode_label = QLabel(self)
        self.mode_combo = QComboBox(self)
        for mode_key, _ in _MODE_I18N_KEYS:
            self.mode_combo.addItem("", mode_key)
        self.mode_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.mode_combo.setFixedWidth(self._right_column_width)
        self.mode_combo.setFixedHeight(self._field_height)
        self._apply_reference_combo_style(self.mode_combo)

        self.bbox_label = QLabel(self)
        self.bbox_combo = QComboBox(self)
        for bbox_key, bbox_i18n in [
            ("union", "merge_bbox_union"),
            ("intersection", "merge_bbox_intersection"),
            ("a", "merge_bbox_a"),
            ("b", "merge_bbox_b"),
        ]:
            self.bbox_combo.addItem("", (bbox_key, bbox_i18n))
        self.bbox_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.bbox_combo.setFixedWidth(self._right_column_width)
        self.bbox_combo.setFixedHeight(self._field_height)
        self._apply_reference_combo_style(self.bbox_combo)

        self.opacity_label = QLabel(self)
        self.opacity_spin = QDoubleSpinBox(self)
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setDecimals(2)
        self.opacity_spin.setValue(1.0)
        self.opacity_field, self.opacity_slider = self._make_slider_field(
            self.opacity_spin, min_value=0.0, max_value=1.0, slider_scale=100
        )

        self.mix_label = QLabel(self)
        self.mix_spin = QDoubleSpinBox(self)
        self.mix_spin.setRange(0.0, 1.0)
        self.mix_spin.setSingleStep(0.05)
        self.mix_spin.setDecimals(2)
        self.mix_spin.setValue(1.0)
        self.mix_field, self.mix_slider = self._make_slider_field(
            self.mix_spin, min_value=0.0, max_value=1.0, slider_scale=100
        )

        self._add_form_row(self.blend_form, self.mode_label, self.mode_combo)
        self._add_form_row(self.blend_form, self.bbox_label, self.bbox_combo)
        self._add_form_row(self.blend_form, self.opacity_label, self.opacity_field)
        self._add_form_row(self.blend_form, self.mix_label, self.mix_field)

        # ── Section: Mask ─────────────────────────────────────────────────────
        self.mask_section, self.mask_section_title, self.mask_form = self._create_section(expanded=True)

        self.mask_enabled_label = QLabel(self)
        self.mask_enabled_check = QCheckBox(self)
        self._apply_reference_checkbox_style(self.mask_enabled_check)

        self.mask_channel_label = QLabel(self)
        self.mask_channel_combo = QComboBox(self)
        for channel_key, channel_i18n in [
            ("auto", "merge_mask_channel_auto"),
            ("luma", "merge_mask_channel_luma"),
            ("red", "merge_mask_channel_red"),
            ("green", "merge_mask_channel_green"),
            ("blue", "merge_mask_channel_blue"),
            ("alpha", "merge_mask_channel_alpha"),
        ]:
            self.mask_channel_combo.addItem("", (channel_key, channel_i18n))
        self.mask_channel_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.mask_channel_combo.setFixedWidth(self._right_column_width)
        self.mask_channel_combo.setFixedHeight(self._field_height)
        self._apply_reference_combo_style(self.mask_channel_combo)

        self.invert_mask_label = QLabel(self)
        self.invert_mask_check = QCheckBox(self)
        self._apply_reference_checkbox_style(self.invert_mask_check)

        self.mask_inject_label = QLabel(self)
        self.mask_inject_check = QCheckBox(self)
        self._apply_reference_checkbox_style(self.mask_inject_check)

        self.fringe_label = QLabel(self)
        self.fringe_check = QCheckBox(self)
        self._apply_reference_checkbox_style(self.fringe_check)

        self.alpha_masking_label = QLabel(self)
        self.alpha_masking_check = QCheckBox(self)
        self._apply_reference_checkbox_style(self.alpha_masking_check)

        self._add_checkbox_row(self.mask_form, self.mask_enabled_check, self.mask_enabled_label)
        self._add_form_row(self.mask_form, self.mask_channel_label, self.mask_channel_combo)
        self._add_checkbox_row(self.mask_form, self.mask_inject_check, self.mask_inject_label)
        self._add_checkbox_row(self.mask_form, self.invert_mask_check, self.invert_mask_label)
        self._add_checkbox_row(self.mask_form, self.fringe_check, self.fringe_label)
        self._add_checkbox_row(self.mask_form, self.alpha_masking_check, self.alpha_masking_label)

        # ── Assemble root ─────────────────────────────────────────────────────
        self.root.addWidget(self.blend_section)
        self.root.addWidget(self.mask_section)

        self.mask_enabled_check.toggled.connect(self._sync_mask_controls_state)

        self.retranslate_ui()
        self._sync_mask_controls_state(self.mask_enabled_check.isChecked())

    def _sync_mask_controls_state(self, enabled: bool) -> None:
        if not enabled:
            # Keep a deterministic state when mask is disabled.
            with QSignalBlocker(self.mask_channel_combo):
                self.mask_channel_combo.setCurrentIndex(0)  # auto
            with QSignalBlocker(self.mask_inject_check):
                self.mask_inject_check.setChecked(False)
            with QSignalBlocker(self.invert_mask_check):
                self.invert_mask_check.setChecked(False)
            with QSignalBlocker(self.fringe_check):
                self.fringe_check.setChecked(False)

        mask_controls = (
            self.mask_channel_label,
            self.mask_channel_combo,
            self.mask_inject_label,
            self.mask_inject_check,
            self.invert_mask_label,
            self.invert_mask_check,
            self.fringe_label,
            self.fringe_check,
        )
        for widget in mask_controls:
            widget.setEnabled(bool(enabled))

    def set_translator(self, translate: Callable[[str], str]) -> None:
        self._tr = translate
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.blend_section_title.setText(self._tr("merge_section_blend"))
        self.mask_section_title.setText(self._tr("merge_section_mask"))

        self.mode_label.setText(self._tr("merge_mode_label"))
        self.bbox_label.setText(self._tr("merge_bbox_label"))
        self.opacity_label.setText(self._tr("merge_opacity_label"))
        self.mix_label.setText(self._tr("merge_mix_label"))
        self.mask_enabled_label.setText(self._tr("merge_mask_enabled_label"))
        self.mask_channel_label.setText(self._tr("merge_mask_channel_label"))
        self.mask_inject_label.setText(self._tr("merge_mask_inject_label"))
        self.invert_mask_label.setText(self._tr("merge_invert_mask_label"))
        self.fringe_label.setText(self._tr("merge_fringe_label"))
        self.alpha_masking_label.setText(self._tr("merge_alpha_masking_label"))
        self.mode_label.setToolTip(self._tr("merge_mode_tooltip"))
        self.bbox_label.setToolTip(self._tr("merge_bbox_tooltip"))
        self.bbox_combo.setToolTip(self._tr("merge_bbox_tooltip"))
        self.opacity_label.setToolTip(self._tr("merge_opacity_tooltip"))
        self.mix_label.setToolTip(self._tr("merge_mix_tooltip"))
        self.mix_spin.setToolTip(self._tr("merge_mix_tooltip"))
        self.mask_enabled_label.setToolTip(self._tr("merge_mask_enabled_tooltip"))
        self.mask_enabled_check.setToolTip(self._tr("merge_mask_enabled_tooltip"))
        self.mask_channel_label.setToolTip(self._tr("merge_mask_channel_tooltip"))
        self.mask_channel_combo.setToolTip(self._tr("merge_mask_channel_tooltip"))
        self.mask_inject_label.setToolTip(self._tr("merge_mask_inject_tooltip"))
        self.mask_inject_check.setToolTip(self._tr("merge_mask_inject_tooltip"))
        self.invert_mask_label.setToolTip(self._tr("merge_invert_mask_tooltip"))
        self.invert_mask_check.setToolTip(self._tr("merge_invert_mask_tooltip"))
        self.fringe_label.setToolTip(self._tr("merge_fringe_tooltip"))
        self.fringe_check.setToolTip(self._tr("merge_fringe_tooltip"))
        self.alpha_masking_label.setToolTip(self._tr("merge_alpha_masking_tooltip"))
        self.alpha_masking_check.setToolTip(self._tr("merge_alpha_masking_tooltip"))

        for i, (_, i18n_key) in enumerate(_MODE_I18N_KEYS):
            self.mode_combo.setItemText(i, self._tr(i18n_key))
        for i in range(self.bbox_combo.count()):
            bbox_key, bbox_i18n = self.bbox_combo.itemData(i)
            self.bbox_combo.setItemText(i, self._tr(bbox_i18n))
        for i in range(self.mask_channel_combo.count()):
            channel_key, channel_i18n = self.mask_channel_combo.itemData(i)
            self.mask_channel_combo.setItemText(i, self._tr(channel_i18n))

    def load_from_properties(self, props: dict) -> None:
        """Load properties from node data dict."""
        mode = str(props.get("mode", "over")).strip().lower()
        # Resolve legacy aliases
        mode = _MODE_ALIASES.get(mode, mode)
        idx = next((i for i, (k, _) in enumerate(_MODE_I18N_KEYS) if k == mode), 0)
        self.mode_combo.setCurrentIndex(idx)
        bbox_mode = str(props.get("set_bbox_to", "union")).strip().lower()
        bbox_idx = next((i for i in range(self.bbox_combo.count()) if self.bbox_combo.itemData(i)[0] == bbox_mode), 0)
        self.bbox_combo.setCurrentIndex(bbox_idx)
        self.opacity_spin.setValue(float(props.get("opacity", 1.0)))
        self.mix_spin.setValue(float(props.get("mix", 1.0)))
        self.mask_enabled_check.setChecked(bool(props.get("mask_enabled", True)))
        mask_channel = str(props.get("mask_channel", "auto")).strip().lower()
        mask_idx = next((i for i in range(self.mask_channel_combo.count()) if self.mask_channel_combo.itemData(i)[0] == mask_channel), 0)
        self.mask_channel_combo.setCurrentIndex(mask_idx)
        self.mask_inject_check.setChecked(bool(props.get("mask_inject", False)))
        self.invert_mask_check.setChecked(bool(props.get("invert_mask", False)))
        self.fringe_check.setChecked(bool(props.get("fringe", False)))
        self.alpha_masking_check.setChecked(bool(props.get("alpha_masking", True)))
        self._sync_mask_controls_state(self.mask_enabled_check.isChecked())

    def write_to_properties(self, props: dict) -> None:
        """Write properties to node data dict."""
        idx = self.mode_combo.currentIndex()
        if 0 <= idx < len(_MODE_I18N_KEYS):
            props["mode"] = _MODE_I18N_KEYS[idx][0]
        bbox_idx = self.bbox_combo.currentIndex()
        if 0 <= bbox_idx < self.bbox_combo.count():
            props["set_bbox_to"] = self.bbox_combo.itemData(bbox_idx)[0]
        props["opacity"] = self.opacity_spin.value()
        props["mix"] = self.mix_spin.value()
        props["mask_enabled"] = self.mask_enabled_check.isChecked()
        if props["mask_enabled"]:
            channel_idx = self.mask_channel_combo.currentIndex()
            if 0 <= channel_idx < self.mask_channel_combo.count():
                props["mask_channel"] = self.mask_channel_combo.itemData(channel_idx)[0]
            props["mask_inject"] = self.mask_inject_check.isChecked()
            props["invert_mask"] = self.invert_mask_check.isChecked()
            props["fringe"] = self.fringe_check.isChecked()
        else:
            props["mask_channel"] = "auto"
            props["mask_inject"] = False
            props["invert_mask"] = False
            props["fringe"] = False
        props["alpha_masking"] = self.alpha_masking_check.isChecked()
