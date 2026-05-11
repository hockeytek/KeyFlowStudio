"""Write node properties panel."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QStackedWidget, QWidget, QSizePolicy,
)

from app.node_graph.node_panel_mixin import NodePanelMixin
from app.node_graph.properties_style import (
    apply_properties_reference_style,
    configure_form_layout,
    configure_inline_layout,
    configure_properties_panel,
)


class WritePropertiesPanel(QWidget, NodePanelMixin):
    """Compact Write controls used inside node properties."""

    def __init__(self, translate: Callable[[str], str], parent=None) -> None:
        super().__init__(parent)
        self._tr = translate
        self._init_panel_layout()
        configure_properties_panel(self)

        self.form = QFormLayout(self)
        configure_form_layout(self.form)

        # ── Path/name/format controls ──
        self.auto_output_check = QCheckBox(self)
        self.path_edit = QLineEdit(self)
        self.file_name_edit = QLineEdit(self)
        self.format_combo = QComboBox(self)
        self.browse_button = QPushButton(self)
        self.save_foreground_check = QCheckBox(self)
        self.save_alpha_check = QCheckBox(self)

        self.path_row = QWidget(self)
        self.path_row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.path_row.setFixedHeight(30)
        self.path_layout = QHBoxLayout(self.path_row)
        configure_inline_layout(self.path_layout)
        self.path_layout.addWidget(self.path_edit, 1)
        self.path_layout.addWidget(self.browse_button)

        # ── Codec/quality stacked pages ──
        # Page 0: empty (source / unknown)
        # Page 1: video (mp4 / mov)
        # Page 2: png
        # Page 3: jpg
        self.codec_stack = QStackedWidget(self)
        self.codec_stack.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        # Page 0 – empty
        self.codec_stack.addWidget(QWidget())

        # Page 1 – video
        vid_page = QWidget()
        vid_form = QFormLayout(vid_page)
        configure_form_layout(vid_form)
        self.codec_combo = QComboBox(vid_page)
        self.quality_spin = QSpinBox(vid_page)
        self.quality_spin.setRange(0, 51)
        self.quality_spin.setValue(23)
        self.quality_spin.setToolTip("")
        self.quality_field, self.quality_slider = self._make_expanding_slider_field(
            self.quality_spin, min_value=0, max_value=51
        )
        self.preset_combo = QComboBox(vid_page)
        self._lbl_codec = QLabel(vid_page)
        self._lbl_quality = QLabel(vid_page)
        self._lbl_preset = QLabel(vid_page)
        self._lbl_prores_note = QLabel(vid_page)
        self._lbl_prores_note.setWordWrap(True)
        self._lbl_prores_note.setStyleSheet("color: #8ca0ba; font-size: 11px;")
        vid_form.addRow(self._lbl_codec, self.codec_combo)
        vid_form.addRow(self._lbl_quality, self.quality_field)
        vid_form.addRow(self._lbl_preset, self.preset_combo)
        vid_form.addRow(self._lbl_prores_note)
        self._vid_form = vid_form
        self.codec_stack.addWidget(vid_page)
        self.codec_combo.currentIndexChanged.connect(self._on_codec_changed)

        # Page 2 – png
        png_page = QWidget()
        png_form = QFormLayout(png_page)
        configure_form_layout(png_form)
        self._lbl_png_bit_depth = QLabel(png_page)
        self.png_bit_depth_combo = QComboBox(png_page)
        self._lbl_png_compression = QLabel(png_page)
        self.png_compression_spin = QSpinBox(png_page)
        self.png_compression_spin.setRange(0, 9)
        self.png_compression_spin.setValue(6)
        self.png_compression_field, self.png_compression_slider = self._make_expanding_slider_field(
            self.png_compression_spin, min_value=0, max_value=9
        )
        self.png_embed_alpha_check = QCheckBox(png_page)
        png_form.addRow(self._lbl_png_bit_depth, self.png_bit_depth_combo)
        png_form.addRow(self._lbl_png_compression, self.png_compression_field)
        png_form.addRow(self.png_embed_alpha_check)
        self.codec_stack.addWidget(png_page)

        # Page 3 – jpg
        jpg_page = QWidget()
        jpg_form = QFormLayout(jpg_page)
        configure_form_layout(jpg_form)
        self.jpg_quality_spin = QSpinBox(jpg_page)
        self.jpg_quality_spin.setRange(1, 100)
        self.jpg_quality_spin.setValue(90)
        self.jpg_quality_field, self.jpg_quality_slider = self._make_expanding_slider_field(
            self.jpg_quality_spin, min_value=1, max_value=100
        )
        self._lbl_jpg_quality = QLabel(jpg_page)
        jpg_form.addRow(self._lbl_jpg_quality, self.jpg_quality_field)
        self.codec_stack.addWidget(jpg_page)

        # ── Info label ──
        self.info_label = QLabel(self)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #8ca0ba; font-size: 11px;")
        self.info_label.setTextFormat(Qt.TextFormat.RichText)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.info_label.setMinimumHeight(130)

        # ── Assemble form ──
        self.form.addRow("", self.auto_output_check)   # row 0
        self.form.addRow("", self.path_row)             # row 1
        self.form.addRow("", self.file_name_edit)       # row 2
        self.form.addRow("", self.format_combo)         # row 3
        self.form.addRow(self.codec_stack)              # row 4 (spans both cols)
        self.form.addRow("", self.save_foreground_check)
        self.form.addRow("", self.save_alpha_check)
        self.form.addRow("", self.info_label)

        # Saving targets are now derived from connected fg/alpha ports in node graph.
        self.form.setRowVisible(self.save_foreground_check, False)
        self.form.setRowVisible(self.save_alpha_check, False)

        self.auto_output_check.toggled.connect(self._update_path_controls)
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)

        apply_properties_reference_style(self)
        self._apply_reference_checkbox_style(self.auto_output_check)
        self._apply_reference_checkbox_style(self.png_embed_alpha_check)

        self.retranslate_ui()

    def set_translator(self, translate: Callable[[str], str]) -> None:
        self._tr = translate
        self.retranslate_ui()

    def _set_form_label_text(self, row: int, text: str) -> None:
        item = self.form.itemAt(row, QFormLayout.ItemRole.LabelRole)
        if item is None or item.widget() is None:
            return
        item.widget().setText(text)

    def retranslate_ui(self) -> None:
        current_format = self.current_format_key()
        self._set_form_label_text(0, self._tr("node_props_auto_output_dir"))
        self._set_form_label_text(1, self._tr("node_props_output_dir"))
        self._set_form_label_text(2, self._tr("node_props_output_name"))
        self._set_form_label_text(3, self._tr("node_props_output_format"))

        self.auto_output_check.setText(self._tr("node_props_auto_output_dir_hint"))
        self.auto_output_check.setToolTip(self._tr("node_props_auto_output_dir_tooltip"))
        self.browse_button.setText(self._tr("node_props_browse"))
        self.browse_button.setToolTip(self._tr("node_props_browse_output"))
        self.path_edit.setPlaceholderText(self._tr("node_props_output_dir_placeholder"))
        self.path_edit.setToolTip(self._tr("node_props_output_dir_tooltip"))
        self.file_name_edit.setPlaceholderText(self._tr("node_props_output_name_placeholder"))
        self.file_name_edit.setToolTip(self._tr("node_props_output_name_tooltip"))
        self.save_foreground_check.setText(self._tr("node_props_save_foreground_hint"))
        self.save_alpha_check.setText(self._tr("node_props_save_alpha_hint"))

        # Format combo
        self.format_combo.blockSignals(True)
        self.format_combo.clear()
        for key, label_key in [
            ("source", "node_props_format_source"),
            ("mp4", "node_props_format_mp4"),
            ("mov", "node_props_format_mov"),
            ("png", "node_props_format_png"),
            ("jpg", "node_props_format_jpg"),
            ("exr", "node_props_format_exr"),
        ]:
            self.format_combo.addItem(self._tr(label_key), key)
        index = self.format_combo.findData(current_format)
        if index >= 0:
            self.format_combo.setCurrentIndex(index)
        self.format_combo.blockSignals(False)
        self.format_combo.setToolTip(self._tr("node_props_output_format_tooltip"))

        # Video page labels
        self._lbl_codec.setText(self._tr("node_props_video_codec"))
        self._lbl_quality.setText(self._tr("node_props_video_quality"))
        self.quality_spin.setToolTip(self._tr("node_props_video_quality_hint"))
        self._lbl_preset.setText(self._tr("node_props_video_preset"))
        cur_codec = self.codec_combo.currentData() or "h264"
        cur_preset = self.preset_combo.currentData() or "medium"
        self.codec_combo.blockSignals(True)
        self.codec_combo.clear()
        for k, lk in [
            ("h264", "node_props_codec_h264"),
            ("h265", "node_props_codec_h265"),
            ("prores422", "node_props_codec_prores422"),
            ("prores422hq", "node_props_codec_prores422hq"),
            ("prores4444", "node_props_codec_prores4444"),
        ]:
            self.codec_combo.addItem(self._tr(lk), k)
        ci = self.codec_combo.findData(cur_codec)
        self.codec_combo.setCurrentIndex(max(ci, 0))
        self.codec_combo.blockSignals(False)
        self._lbl_prores_note.setText(self._tr("node_props_prores_note"))
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for k, lk in [("fast", "node_props_preset_fast"), ("medium", "node_props_preset_medium"), ("slow", "node_props_preset_slow")]:
            self.preset_combo.addItem(self._tr(lk), k)
        pi = self.preset_combo.findData(cur_preset)
        self.preset_combo.setCurrentIndex(max(pi, 0))
        self.preset_combo.blockSignals(False)

        # PNG/JPG page labels
        self._lbl_png_bit_depth.setText(self._tr("node_props_png_bit_depth"))
        cur_bd = self.png_bit_depth_combo.currentData() or 8
        self.png_bit_depth_combo.blockSignals(True)
        self.png_bit_depth_combo.clear()
        for v, lk in [(8, "node_props_bit_depth_8"), (16, "node_props_bit_depth_16")]:
            self.png_bit_depth_combo.addItem(self._tr(lk), v)
        bdi = self.png_bit_depth_combo.findData(cur_bd)
        self.png_bit_depth_combo.setCurrentIndex(max(bdi, 0))
        self.png_bit_depth_combo.blockSignals(False)
        self._lbl_png_compression.setText(self._tr("node_props_png_compression"))
        self.png_compression_spin.setToolTip(self._tr("node_props_png_compression_hint"))
        self.png_embed_alpha_check.setText(self._tr("node_props_png_embed_alpha"))
        self.png_embed_alpha_check.setToolTip(self._tr("node_props_png_embed_alpha_hint"))
        self._lbl_jpg_quality.setText(self._tr("node_props_jpg_quality"))
        self.jpg_quality_spin.setToolTip(self._tr("node_props_jpg_quality_hint"))

        self._update_path_controls(self.auto_output_check.isChecked())
        self._on_format_changed()
        self._on_codec_changed()

    def _on_codec_changed(self) -> None:
        codec = self.codec_combo.currentData() or "h264"
        is_prores = codec.startswith("prores")
        # CRF and preset are not used for ProRes
        self._lbl_quality.setVisible(not is_prores)
        self.quality_field.setVisible(not is_prores)
        self._lbl_preset.setVisible(not is_prores)
        self.preset_combo.setVisible(not is_prores)
        self._lbl_prores_note.setVisible(is_prores)

    def _on_format_changed(self) -> None:
        fmt = self.current_format_key()
        if fmt in {"mp4", "mov"}:
            self.codec_stack.setCurrentIndex(1)
        elif fmt in {"png", "exr"}:
            self.codec_stack.setCurrentIndex(2)
        elif fmt == "jpg":
            self.codec_stack.setCurrentIndex(3)
        else:
            self.codec_stack.setCurrentIndex(0)

        is_png = fmt == "png"
        is_exr = fmt == "exr"
        self._lbl_png_bit_depth.setVisible(is_png)
        self.png_bit_depth_combo.setVisible(is_png)
        self._lbl_png_compression.setVisible(is_png)
        self.png_compression_field.setVisible(is_png)
        self.png_embed_alpha_check.setVisible(is_png or is_exr)

    def current_format_key(self) -> str:
        value = self.format_combo.currentData()
        if isinstance(value, str) and value:
            return value
        return "source"

    def current_format_label(self) -> str:
        return self.format_combo.currentText().strip()

    def _update_path_controls(self, checked: bool) -> None:
        self.path_edit.setEnabled(not checked)
        self.browse_button.setEnabled(not checked)
        if checked:
            self.path_edit.setPlaceholderText(self._tr("node_props_output_dir_placeholder"))

    def load_from_properties(self, props: dict) -> None:
        self.auto_output_check.setChecked(bool(props.get("auto_output_dir", True)))
        self.path_edit.setText(str(props.get("output_dir", "")))
        self.file_name_edit.setText(str(props.get("file_name", "")))
        format_key = str(props.get("output_format", "source")).strip().lower() or "source"
        format_index = self.format_combo.findData(format_key)
        self.format_combo.blockSignals(True)
        self.format_combo.setCurrentIndex(max(format_index, 0))
        self.format_combo.blockSignals(False)

        # Video params
        codec_key = str(props.get("video_codec", "h264")).strip().lower() or "h264"
        ci = self.codec_combo.findData(codec_key)
        self.codec_combo.setCurrentIndex(max(ci, 0))
        self.quality_spin.setValue(int(props.get("video_quality", 23)))
        preset_key = str(props.get("video_preset", "medium")).strip().lower() or "medium"
        pi = self.preset_combo.findData(preset_key)
        self.preset_combo.setCurrentIndex(max(pi, 0))

        # PNG/JPG params
        self.png_compression_spin.setValue(int(props.get("png_compression", 6)))
        bd = int(props.get("png_bit_depth", 8))
        bdi = self.png_bit_depth_combo.findData(bd)
        self.png_bit_depth_combo.setCurrentIndex(max(bdi, 0))
        self.png_embed_alpha_check.setChecked(bool(props.get("png_embed_alpha", False)))
        self.jpg_quality_spin.setValue(int(props.get("jpg_quality", 90)))

        self._update_path_controls(self.auto_output_check.isChecked())
        self._on_format_changed()

    def write_to_properties(self, props: dict) -> None:
        props["auto_output_dir"] = self.auto_output_check.isChecked()
        props["output_dir"] = self.path_edit.text().strip()
        props["file_name"] = self.file_name_edit.text().strip()
        props["output_format"] = self.current_format_key()
        props["video_codec"] = self.codec_combo.currentData() or "h264"
        props["video_quality"] = self.quality_spin.value()
        props["video_preset"] = self.preset_combo.currentData() or "medium"
        props["png_compression"] = self.png_compression_spin.value()
        props["png_bit_depth"] = self.png_bit_depth_combo.currentData() or 8
        props["png_embed_alpha"] = self.png_embed_alpha_check.isChecked()
        props["jpg_quality"] = self.jpg_quality_spin.value()

    def set_info_text(self, text: str) -> None:
        self.info_label.setText(text)
