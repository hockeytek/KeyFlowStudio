"""Shared styling helpers for node properties panels."""

from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QLayout, QSizePolicy, QWidget


PROPERTIES_PANEL_WIDTH = 324
PROPERTIES_FORM_SPACING = 8
PROPERTIES_INLINE_SPACING = 6


def configure_properties_panel(widget: QWidget) -> None:
    """Apply fixed-width policy used by the right properties sidebar content area."""
    widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
    widget.setFixedWidth(PROPERTIES_PANEL_WIDTH)


def configure_form_layout(form: QFormLayout) -> None:
    """Apply unified form rhythm used by node property panels."""
    form.setContentsMargins(0, 0, 0, 0)
    form.setSpacing(PROPERTIES_FORM_SPACING)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)


def configure_inline_layout(layout: QLayout) -> None:
    """Apply unified spacing for inline row layouts."""
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(PROPERTIES_INLINE_SPACING)


def apply_properties_reference_style(widget: QWidget) -> None:
    """Apply CorridorKey-like baseline styling to generic controls."""
    widget.setStyleSheet(
        "QWidget {"
        " color: #d8dee7;"
        "}"
        "QLabel {"
        " color: #c7ccd5;"
        " font-size: 11px;"
        " font-weight: 600;"
        "}"
        "QLineEdit, QListWidget {"
        " border: 1px solid #090d14;"
        " border-radius: 5px;"
        " background: #131923;"
        " color: #eef2f7;"
        " padding: 0 7px;"
        " min-height: 26px;"
        "}"
        "QComboBox {"
        " border: 1px solid #2b3140;"
        " border-radius: 5px;"
        " background: #171d2a;"
        " color: #e2e7ee;"
        " padding: 0 7px;"
        " min-height: 20px;"
        "}"
        "QSpinBox, QDoubleSpinBox {"
        " border: 1px solid #090d14;"
        " border-radius: 5px;"
        " background: #131923;"
        " color: #eef2f7;"
        " padding: 0 7px;"
        " font-size: 10px;"
        " font-weight: 700;"
        " min-height: 26px;"
        "}"
        "QLineEdit:focus, QComboBox:focus, QListWidget:focus, QSpinBox:focus, QDoubleSpinBox:focus {"
        " border: 1px solid #7b889b;"
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
        "QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {"
        " width: 0px;"
        " border: 0;"
        "}"
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
        "QPushButton {"
        " border: 1px solid #2b3140;"
        " border-radius: 5px;"
        " background: #171d2a;"
        " color: #dfe8f4;"
        " min-height: 26px;"
        " padding: 0 8px;"
        "}"
        "QPushButton:hover {"
        " border-color: #3b475b;"
        " background: #1c2434;"
        "}"
        "QPushButton:pressed {"
        " background: #101722;"
        "}"
        "QProgressBar {"
        " border: 1px solid #2b3140;"
        " border-radius: 4px;"
        " background: #111722;"
        " color: #dfe8f4;"
        " text-align: center;"
        "}"
        "QProgressBar::chunk {"
        " background: #4b7597;"
        " border-radius: 3px;"
        "}"
    )
