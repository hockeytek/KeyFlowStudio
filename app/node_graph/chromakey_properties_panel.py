"""HSV Chroma Key node properties panel."""

from __future__ import annotations

from typing import Callable

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.node_graph.properties_style import (
    apply_properties_reference_style,
    configure_form_layout,
    configure_properties_panel,
)


class ChromaKeyPropertiesPanel(QWidget):
    """Compact HSV Chroma Key controls used inside node properties."""

    def __init__(self, translate: Callable[[str], str], parent=None) -> None:
        super().__init__(parent)
        self._tr = translate
        configure_properties_panel(self)

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(0, 0, 0, 0)
        self.root.setSpacing(8)

        self.form = QFormLayout()
        configure_form_layout(self.form)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)

        _h = 34

        # Hue center (0–360°)
        self.hue_center_label = QLabel(self)
        self.hue_center_spin = QSpinBox(self)
        self.hue_center_spin.setRange(0, 360)
        self.hue_center_spin.setSingleStep(1)
        self.hue_center_spin.setSuffix("°")
        self.hue_center_spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.hue_center_spin.setFixedHeight(_h)

        # Hue range (±° tolerance)
        self.hue_range_label = QLabel(self)
        self.hue_range_spin = QSpinBox(self)
        self.hue_range_spin.setRange(1, 90)
        self.hue_range_spin.setSingleStep(1)
        self.hue_range_spin.setSuffix("°")
        self.hue_range_spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.hue_range_spin.setFixedHeight(_h)

        # Saturation min (0.0–1.0)
        self.saturation_min_label = QLabel(self)
        self.saturation_min_spin = QDoubleSpinBox(self)
        self.saturation_min_spin.setRange(0.0, 1.0)
        self.saturation_min_spin.setSingleStep(0.05)
        self.saturation_min_spin.setDecimals(2)
        self.saturation_min_spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.saturation_min_spin.setFixedHeight(_h)

        # Value min (0.0–1.0)
        self.value_min_label = QLabel(self)
        self.value_min_spin = QDoubleSpinBox(self)
        self.value_min_spin.setRange(0.0, 1.0)
        self.value_min_spin.setSingleStep(0.05)
        self.value_min_spin.setDecimals(2)
        self.value_min_spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.value_min_spin.setFixedHeight(_h)

        # Blur radius (0 = no blur)
        self.blur_radius_label = QLabel(self)
        self.blur_radius_spin = QSpinBox(self)
        self.blur_radius_spin.setRange(0, 20)
        self.blur_radius_spin.setSingleStep(1)
        self.blur_radius_spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.blur_radius_spin.setFixedHeight(_h)

        self.form.addRow(self.hue_center_label, self.hue_center_spin)
        self.form.addRow(self.hue_range_label, self.hue_range_spin)
        self.form.addRow(self.saturation_min_label, self.saturation_min_spin)
        self.form.addRow(self.value_min_label, self.value_min_spin)
        self.form.addRow(self.blur_radius_label, self.blur_radius_spin)

        self.root.addLayout(self.form)

        # Color-picker button
        self.pick_btn = QPushButton(self)
        self.pick_btn.setFixedHeight(_h)
        self.pick_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.pick_btn.clicked.connect(self._on_pick_color)
        self.root.addWidget(self.pick_btn)

        apply_properties_reference_style(self)

        self.retranslate_ui()

    def set_translator(self, translate: Callable[[str], str]) -> None:
        self._tr = translate
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.hue_center_label.setText(self._tr("chromakey_hue_center"))
        self.hue_range_label.setText(self._tr("chromakey_hue_range"))
        self.saturation_min_label.setText(self._tr("chromakey_saturation_min"))
        self.value_min_label.setText(self._tr("chromakey_value_min"))
        self.blur_radius_label.setText(self._tr("chromakey_blur_radius"))

        self.hue_center_label.setToolTip(self._tr("chromakey_hue_center_tooltip"))
        self.hue_range_label.setToolTip(self._tr("chromakey_hue_range_tooltip"))
        self.saturation_min_label.setToolTip(self._tr("chromakey_saturation_min_tooltip"))
        self.value_min_label.setToolTip(self._tr("chromakey_value_min_tooltip"))
        self.blur_radius_label.setToolTip(self._tr("chromakey_blur_radius_tooltip"))

        self.pick_btn.setText(self._tr("btn_chromakey_pick"))
        self.pick_btn.setToolTip(self._tr("btn_chromakey_pick_tooltip"))

    def _on_pick_color(self) -> None:
        """Open QColorDialog; apply picked hue to hue_center_spin."""
        hue = self.hue_center_spin.value() % 360
        # Pre-populate with a fully-saturated, mid-brightness version of current hue
        initial = QColor.fromHsv(hue, 220, 180)
        picked = QColorDialog.getColor(
            initial,
            self,
            self._tr("btn_chromakey_pick"),
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if not picked.isValid():
            return
        h = picked.hsvHue()  # -1 for achromatic (pure grey/white/black)
        if h >= 0:
            self.hue_center_spin.setValue(h)
        s = picked.hsvSaturationF()
        if s > 0.05:
            # Suggest a sensible saturation floor: half the picked saturation, clamped
            floor = round(min(max(s * 0.5, 0.05), 0.80), 2)
            self.saturation_min_spin.setValue(floor)

    def load_from_properties(self, props: dict) -> None:
        """Load properties from node data dict."""
        self.hue_center_spin.setValue(int(props.get("hue_center", 120)))
        self.hue_range_spin.setValue(int(props.get("hue_range", 30)))
        self.saturation_min_spin.setValue(float(props.get("saturation_min", 0.15)))
        self.value_min_spin.setValue(float(props.get("value_min", 0.10)))
        self.blur_radius_spin.setValue(int(props.get("blur_radius", 3)))

    def write_to_properties(self, props: dict) -> None:
        """Write properties to node data dict."""
        props["hue_center"] = self.hue_center_spin.value()
        props["hue_range"] = self.hue_range_spin.value()
        props["saturation_min"] = self.saturation_min_spin.value()
        props["value_min"] = self.value_min_spin.value()
        props["blur_radius"] = self.blur_radius_spin.value()
