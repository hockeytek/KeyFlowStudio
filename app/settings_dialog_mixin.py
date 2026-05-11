"""Settings dialog mixin for KeyFlow Studio MainWindow.

Provides:
  - open_settings_dialog()
  - _create_settings_device_combo()
  - _create_settings_compat_combo()
  - _create_settings_language_combo()
  - _create_settings_buttons()

Cloud AWS settings tab has been extracted to app/cloud_aws_settings_tab.py.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.cloud_settings import get_cloud_setting, save_cloud_settings
from app.cloud_aws_settings_tab import _region_code, create_cloud_aws_settings_tab

logger = logging.getLogger(__name__)


class SettingsDialogMixin:
    """Mixin that adds the Settings dialog to MainWindow."""

    # ── Public entry point ────────────────────────────────────────────────────

    def open_settings_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(self._tr("settings_title"))
        dialog.setModal(True)
        dialog.resize(760, 680)
        dialog.setMinimumSize(QSize(720, 600))

        root = QVBoxLayout(dialog)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(8)

        tabs = QTabWidget(dialog)

        # ── Вкладка: Основные ──
        general_widget = QWidget()
        general_layout = QVBoxLayout(general_widget)
        general_layout.setContentsMargins(12, 12, 12, 12)
        general_layout.setSpacing(10)

        combo_device = self._create_settings_device_combo(dialog)
        combo_compat_profile = self._create_settings_compat_combo(dialog)
        combo_language = self._create_settings_language_combo(dialog)
        check_completion_sound = QCheckBox(self._tr("settings_completion_sound_label"), dialog)
        check_completion_sound.setChecked(bool(getattr(self, "_completion_sound_enabled", True)))

        grp_rendering = QGroupBox(self._tr("settings_group_rendering"), general_widget)
        render_form = QFormLayout(grp_rendering)
        render_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        render_form.setContentsMargins(14, 10, 14, 14)
        render_form.setSpacing(9)
        render_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        render_form.addRow(self._tr("settings_device_label"), combo_device)
        render_form.addRow(self._tr("settings_compat_profile_label"), combo_compat_profile)
        general_layout.addWidget(grp_rendering)

        grp_iface = QGroupBox(self._tr("settings_group_interface"), general_widget)
        iface_form = QFormLayout(grp_iface)
        iface_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        iface_form.setContentsMargins(14, 10, 14, 14)
        iface_form.setSpacing(9)
        iface_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        iface_form.addRow(self._tr("settings_language_label"), combo_language)
        iface_form.addRow("", check_completion_sound)
        general_layout.addWidget(grp_iface)
        general_layout.addStretch(1)

        tabs.addTab(general_widget, self._tr("settings_tab_general"))

        # ── Вкладка: Cloud AWS ──
        cloud_widget, cloud_refs = create_cloud_aws_settings_tab(self, dialog)
        tabs.addTab(cloud_widget, self._tr("settings_tab_cloud"))

        # Автообновление статуса если Instance ID уже сохранён
        saved_iid = get_cloud_setting("cloud/instance_id")
        if saved_iid:
            QTimer.singleShot(200, cloud_refs["refresh_fn"])

        root.addWidget(tabs, stretch=1)

        buttons = self._create_settings_buttons(dialog)
        root.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        # Cloud настройки сохраняем ПЕРВЫМИ — чтобы on_device_changed
        # уже читал актуальное значение cloud/enabled из QSettings
        save_cloud_settings(
            instance_id=cloud_refs["instance_id"].text(),
            api_host=cloud_refs["api_host"].text(),
            region=_region_code(cloud_refs["region"]),
            ssh_key_path=cloud_refs["ssh_key"].text(),
            ssh_user=cloud_refs["ssh_user"].text(),
            aws_profile=cloud_refs["aws_profile"].text(),
            enabled=cloud_refs["enabled"].isChecked(),
            watchdog_idle_min=cloud_refs["spin_idle"].value(),
            watchdog_gpu_pct=cloud_refs["spin_gpu"].value(),
        )

        self.on_device_changed(combo_device.currentText())
        self.on_compatibility_profile_changed(
            str(combo_compat_profile.currentData() or "auto")
        )
        self._set_completion_sound_enabled(check_completion_sound.isChecked())
        self._set_language(str(combo_language.currentData()), announce=True)
        self._update_run_button_label()

    # ── Helper combos & buttons ───────────────────────────────────────────────

    def _create_settings_device_combo(self, parent: QWidget) -> QComboBox:
        combo = QComboBox(parent)
        combo.addItems(["Auto", "CPU", "MPS", "CUDA"])
        combo.setMinimumWidth(210)
        combo.setMaximumWidth(260)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        combo.setCurrentText(
            self._device_selection.upper() if self._device_selection != "auto" else "Auto"
        )
        combo.setToolTip(self._update_device_tooltip(self._device_selection))
        combo.currentTextChanged.connect(
            lambda text: combo.setToolTip(self._update_device_tooltip(text))
        )
        return combo

    def _create_settings_compat_combo(self, parent: QWidget) -> QComboBox:
        combo = QComboBox(parent)
        combo.addItem(self._tr("compat_profile_auto"), "auto")
        combo.addItem(self._tr("compat_profile_legacy_intel"), "legacy_intel")
        combo.addItem(self._tr("compat_profile_apple_silicon"), "apple_silicon")
        combo.setMinimumWidth(210)
        combo.setMaximumWidth(330)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        compat_index = combo.findData(self._compatibility_profile)
        combo.setCurrentIndex(compat_index if compat_index >= 0 else 0)
        combo.setToolTip(self._update_compat_profile_tooltip(self._compatibility_profile))
        combo.currentTextChanged.connect(
            lambda _text: combo.setToolTip(
                self._update_compat_profile_tooltip(str(combo.currentData() or "auto"))
            )
        )
        return combo

    def _create_settings_language_combo(self, parent: QWidget) -> QComboBox:
        combo = QComboBox(parent)
        combo.addItem(self._tr("lang_russian"), "ru")
        combo.addItem(self._tr("lang_english"), "en")
        combo.setMinimumWidth(210)
        combo.setMaximumWidth(260)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        index = combo.findData(self._language_code)
        combo.setCurrentIndex(index if index >= 0 else 0)
        return combo

    def _create_settings_buttons(self, dialog: QDialog) -> QDialogButtonBox:
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Help,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        buttons.helpRequested.connect(self.open_about_dialog)
        ok_button     = buttons.button(QDialogButtonBox.StandardButton.Ok)
        cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        help_button   = buttons.button(QDialogButtonBox.StandardButton.Help)
        if ok_button is not None:
            ok_button.setText("OK" if self._language_code == "en" else "ОК")
        if cancel_button is not None:
            cancel_button.setText("Cancel" if self._language_code == "en" else "Отмена")
        if help_button is not None:
            help_button.setText("About" if self._language_code == "en" else "О программе")
        return buttons
