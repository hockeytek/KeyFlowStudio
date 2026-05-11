"""Instance Manager dialog for KeyFlow Studio.

Opens the cloud instance manager modal from the cloud settings tab.
All parent-tab state is passed via *InstanceManagerContext* so this module
stays independent of ``settings_dialog_mixin``.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from app.cloud_manager import (
    InstanceState,
    change_instance_type,
    create_ami_from_instance,
    find_keyflow_ami,
    find_public_gpu_ami,
    format_instance_type_label,
    get_available_gpu_types,
    get_key_pairs,
    get_ondemand_price,
    get_security_groups,
    get_spot_price,
    list_instances,
    list_user_amis,
    start_instance,
    stop_instance,
    terminate_instance,
)
from app.cloud_settings import (
    get_ami_for_profile_region,
    get_key_name_for_profile_region,
    get_sg_for_profile_region,
    set_ami_for_profile_region,
    set_key_name_for_profile_region,
    set_sg_for_profile_region,
)


@dataclass
class InstanceManagerContext:
    """All parent-tab dependencies needed by the Instance Manager dialog."""

    # Parent widget (dialog owner)
    parent: QWidget

    # Localisation callback: tr(key) -> str
    tr: Callable[[str], str]

    # Accessor callables (return current values from QSettings-backed widgets)
    reg: Callable[[], str]    # current region code
    prof: Callable[[], str]   # current AWS profile (or None)
    iid: Callable[[], str]    # current instance ID

    # Region combo in the settings tab (for signal connect/disconnect)
    region_combo: QComboBox

    # Logging (appends to the log panel in the settings tab)
    log: Callable[[str], None]

    # Status helpers
    market_label: Callable[[str], str]
    set_status: Callable[..., None]
    state_labels: dict

    # Parent-tab callbacks that update the main settings tab
    apply_selected: Callable[[dict], None]
    launch_new: Callable[[str, bool], bool]
    schedule_price_refresh: Callable[[], None]

    # Parent-tab live widgets (written back when an instance is terminated)
    poll_timer: QTimer
    edit_instance_id: QLineEdit
    lbl_ip_value: QLabel


def open_instance_manager(ctx: InstanceManagerContext) -> None:
    """Open the Instance Manager modal dialog.

    All state from *_create_cloud_settings_tab* is accessed through *ctx*.
    """
    dialog = QDialog(ctx.parent)
    dialog.setWindowTitle(ctx.tr("cloud_manager_title"))
    dialog.resize(780, 640)
    dialog.setMinimumWidth(720)

    dialog_root = QVBoxLayout(dialog)
    dialog_root.setContentsMargins(14, 14, 14, 14)
    dialog_root.setSpacing(10)

    _COMBO_QSS = (
        "QComboBox {"
        "  background-color: #171d27;"
        "  border: 1px solid #2a3444;"
        "  color: #e8edf5;"
        "  padding: 0px 10px;"
        "  border-radius: 8px;"
        "  min-height: 34px;"
        "  max-height: 36px;"
        "  height: 36px;"
        "  font-size: 13px;"
        "}"
        "QComboBox:hover { border: 1px solid #43c7ff; }"
        "QComboBox:disabled { color: #5b6473; }"
        "QComboBox::drop-down { width: 0; border: none; }"
        "QComboBox::down-arrow { image: none; width: 0; height: 0; }"
        "QComboBox QAbstractItemView {"
        "  background-color: #131a23;"
        "  color: #e8edf5;"
        "  selection-background-color: #0f3c57;"
        "  border: 1px solid #2a3444;"
        "}"
    )

    # ── AMI Configuration ─────────────────────────────────────────────
    grp_ami = QGroupBox(ctx.tr("cloud_group_ami_config"), dialog)
    ami_form = QVBoxLayout(grp_ami)
    ami_form.setContentsMargins(14, 10, 14, 14)
    ami_form.setSpacing(8)

    _lbl_w = 120  # label column width

    # Row 1: AMI label + value + Detect + Browse
    ami_row1 = QHBoxLayout()
    ami_row1.setSpacing(8)
    lbl_ami = QLabel(ctx.tr("cloud_ami_label"), dialog)
    lbl_ami.setStyleSheet("color: #8fa8c0;")
    lbl_ami.setFixedWidth(_lbl_w)
    lbl_ami_value = QLabel(ctx.tr("cloud_ami_not_set"), dialog)
    lbl_ami_value.setStyleSheet("font-family: monospace; color: #c8d8e8;")
    lbl_ami_value.setMinimumWidth(200)
    lbl_ami_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    btn_detect_ami = QPushButton(ctx.tr("cloud_btn_detect_ami"), dialog)
    btn_detect_ami.setMinimumWidth(130)
    btn_detect_ami.setFixedHeight(36)
    btn_browse_amis = QPushButton(ctx.tr("cloud_btn_browse_amis"), dialog)
    btn_browse_amis.setMinimumWidth(150)
    btn_browse_amis.setFixedHeight(36)
    btn_clear_ami = QPushButton("✕", dialog)
    btn_clear_ami.setFixedSize(28, 28)
    btn_clear_ami.setToolTip(ctx.tr("cloud_tt_clear_ami") if ctx.tr("cloud_tt_clear_ami") != "cloud_tt_clear_ami" else "Clear saved AMI for this region")
    btn_clear_ami.setStyleSheet(
        "QPushButton{background:transparent;color:#888;border:none;font-size:14px;}"
        "QPushButton:hover{color:#e05050;}"
    )
    ami_row1.addWidget(lbl_ami)
    ami_row1.addWidget(lbl_ami_value, 1)
    ami_row1.addWidget(btn_clear_ami)
    ami_row1.addWidget(btn_detect_ami)
    ami_row1.addWidget(btn_browse_amis)
    ami_form.addLayout(ami_row1)

    # Row 2: EC2 key pair
    ami_row2 = QHBoxLayout()
    ami_row2.setSpacing(8)
    lbl_kp = QLabel(ctx.tr("cloud_key_pair_label"), dialog)
    lbl_kp.setStyleSheet("color: #8fa8c0;")
    lbl_kp.setFixedWidth(_lbl_w)
    combo_key_pair = QComboBox(dialog)
    combo_key_pair.addItem(ctx.tr("cloud_resources_loading"), "")
    combo_key_pair.setEnabled(False)
    combo_key_pair.setMinimumWidth(220)
    combo_key_pair.setMaximumWidth(260)
    combo_key_pair.setFixedHeight(36)
    combo_key_pair.setStyleSheet(_COMBO_QSS)
    btn_quick_create = QPushButton(ctx.tr("cloud_btn_quick_create"), dialog)
    btn_quick_create.setMinimumWidth(200)
    btn_quick_create.setFixedHeight(36)
    btn_quick_create.setStyleSheet(
        "QPushButton{background:#1f5fbf;color:#ffffff;border:1px solid #2a74de;"
        "border-radius:6px;padding:6px 16px;font-weight:bold;}"
        "QPushButton:hover{background:#2a74de;}"
        "QPushButton:pressed{background:#184b97;}"
    )
    ami_row2.addWidget(lbl_kp)
    ami_row2.addWidget(combo_key_pair)
    ami_row2.addSpacing(8)
    ami_row2.addWidget(btn_quick_create, 1)
    ami_form.addLayout(ami_row2)

    # Row 3: Security Group
    ami_row3 = QHBoxLayout()
    ami_row3.setSpacing(8)
    lbl_sg = QLabel(ctx.tr("cloud_sg_label"), dialog)
    lbl_sg.setStyleSheet("color: #8fa8c0;")
    lbl_sg.setFixedWidth(_lbl_w)
    combo_sg = QComboBox(dialog)
    combo_sg.addItem(ctx.tr("cloud_resources_loading"), "")
    combo_sg.setEnabled(False)
    combo_sg.setMinimumWidth(320)
    combo_sg.setFixedHeight(36)
    combo_sg.setStyleSheet(_COMBO_QSS)
    ami_row3.addWidget(lbl_sg)
    ami_row3.addWidget(combo_sg)
    ami_row3.addStretch()
    ami_form.addLayout(ami_row3)

    dialog_root.addWidget(grp_ami)

    # ── helpers used by AMI section ───────────────────────────────────
    def _ami_label_text(ami_id: str, name: str) -> str:
        if not ami_id:
            return ctx.tr("cloud_ami_not_set")
        return f"{ami_id}  ({name})" if name else ami_id

    def _clear_ami():
        prof = ctx.prof() or "default"
        reg = ctx.reg()
        set_ami_for_profile_region(prof, reg, "")
        lbl_ami_value.setText(ctx.tr("cloud_ami_not_set"))
        lbl_ami_value.setStyleSheet("font-family: monospace; color: #e0a040;")
        ctx.log(f"AMI cleared for {reg}")

    def _load_current_ami():
        prof = ctx.prof() or "default"
        reg = ctx.reg()
        saved = get_ami_for_profile_region(prof, reg)
        if saved:
            lbl_ami_value.setText(_ami_label_text(saved, ""))
            lbl_ami_value.setStyleSheet("font-family: monospace; color: #4cde8a;")
        else:
            lbl_ami_value.setText(ctx.tr("cloud_ami_not_set"))
            lbl_ami_value.setStyleSheet("font-family: monospace; color: #e0a040;")

    def _apply_ami(ami_id: str, name: str):
        prof = ctx.prof() or "default"
        reg = ctx.reg()
        set_ami_for_profile_region(prof, reg, ami_id)
        lbl_ami_value.setText(_ami_label_text(ami_id, name))
        lbl_ami_value.setStyleSheet("font-family: monospace; color: #4cde8a;")
        ctx.log(ctx.tr("cloud_log_ami_saved").format(ami_id=ami_id))

    # Async AMI detect
    _ami_detect_running = [False]
    _ami_detect_cancelled = [False]

    def _detect_ami():
        if _ami_detect_running[0]:
            # Cancel in-flight detection
            _ami_detect_cancelled[0] = True
            _ami_detect_running[0] = False
            btn_detect_ami.setText(ctx.tr("cloud_btn_detect_ami"))
            lbl_ami_value.setText(ctx.tr("cloud_ami_not_set"))
            lbl_ami_value.setStyleSheet("font-family: monospace; color: #e0a040;")
            ctx.log("⛔ AMI search cancelled")
            return
        _ami_detect_cancelled[0] = False
        _ami_detect_running[0] = True
        btn_detect_ami.setText("⛔ " + ctx.tr("cloud_btn_detect_ami").replace("Определить", "Отмена").replace("Detect", "Cancel"))
        lbl_ami_value.setText(ctx.tr("cloud_ami_loading"))
        lbl_ami_value.setStyleSheet("font-family: monospace; color: #888888;")
        prof = ctx.prof()
        reg = ctx.reg()
        ctx.log(ctx.tr("cloud_log_ami_search").format(reg=reg, prof=prof or "default"))

        _detect_result_holder: list[tuple | None] = [None]

        def _worker():
            _detect_result_holder[0] = find_keyflow_ami(reg, prof)

        def _poll_detect():
            if _detect_result_holder[0] is None:
                QTimer.singleShot(200, _poll_detect)
                return
            _on_ami_detected(_detect_result_holder[0])

        def _on_ami_detected(result: tuple[str, str, str]):
            _ami_detect_running[0] = False
            btn_detect_ami.setText(ctx.tr("cloud_btn_detect_ami"))
            if _ami_detect_cancelled[0]:
                return
            ami_id, name, err = result
            if err:
                ctx.log(ctx.tr("cloud_log_ami_search_err").format(msg=err))
                lbl_ami_value.setText(ctx.tr("cloud_ami_not_set"))
                lbl_ami_value.setStyleSheet("font-family: monospace; color: #e06060;")
                QMessageBox.warning(dialog, ctx.tr("cloud_manager_title"), err)
                return
            if ami_id:
                ctx.log(ctx.tr("cloud_log_ami_found").format(ami_id=ami_id, name=name or "—"))
                _apply_ami(ami_id, name)
            else:
                ctx.log(ctx.tr("cloud_log_ami_not_found"))
                lbl_ami_value.setText(ctx.tr("cloud_ami_not_set"))
                lbl_ami_value.setStyleSheet("font-family: monospace; color: #e0a040;")

        threading.Thread(target=_worker, daemon=True).start()
        QTimer.singleShot(200, _poll_detect)

    def _browse_amis():
        browse_dlg = QDialog(dialog)
        browse_dlg.setWindowTitle(ctx.tr("cloud_ami_select_title"))
        browse_dlg.resize(680, 400)
        vb = QVBoxLayout(browse_dlg)
        vb.setContentsMargins(14, 12, 14, 12)
        vb.setSpacing(8)
        lst = QListWidget(browse_dlg)
        lst.setMinimumHeight(280)
        lst.setStyleSheet("""
            QListWidget {
                background: #0e1218;
                color: #c8d8e8;
                border: 1px solid #1e2730;
                border-radius: 4px;
                font-size: 12px;
                outline: none;
            }
            QListWidget::item {
                padding: 5px 8px;
                border-bottom: 1px solid #161c24;
            }
            QListWidget::item:selected {
                background: #1e3248;
                color: #e8f0f8;
                border-bottom: 1px solid #1e3248;
            }
            QListWidget::item:hover:!selected { background: #141c26; }
            QScrollBar:vertical {
                background: #0e1218; width: 8px; border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #2a3a4d; min-height: 30px; border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover { background: #3a5068; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar:horizontal {
                background: #0e1218; height: 8px; border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                background: #2a3a4d; min-width: 30px; border-radius: 4px;
            }
            QScrollBar::handle:horizontal:hover { background: #3a5068; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
        """)
        vb.addWidget(lst, 1)
        info_lbl = QLabel(ctx.tr("cloud_resources_loading"), browse_dlg)
        info_lbl.setStyleSheet("color: #8fa8c0; font-size: 11px;")
        vb.addWidget(info_lbl)
        bb = QDialogButtonBox(browse_dlg)
        btn_use = bb.addButton(ctx.tr("cloud_ami_browse_select_btn"), QDialogButtonBox.ButtonRole.AcceptRole)
        bb.addButton(QDialogButtonBox.StandardButton.Cancel)
        bb.rejected.connect(browse_dlg.reject)
        vb.addWidget(bb)

        # _amis_data[0] == None  → still loading
        # _amis_data[0] == list  → done (may be empty)
        _amis_data: list[list[dict] | None] = [None]
        _fetch_error: list[str] = [""]

        def _fetch():
            amis, err = list_user_amis(ctx.reg(), ctx.prof())
            _fetch_error[0] = err
            _amis_data[0] = amis  # signal ready (even if empty)

        def _poll_ami_list():
            if _amis_data[0] is None:
                # still loading — check again in 300 ms
                QTimer.singleShot(300, _poll_ami_list)
                return
            amis = _amis_data[0]
            lst.clear()
            if _fetch_error[0]:
                info_lbl.setText(f"❌ {_fetch_error[0]}")
                item = QListWidgetItem(f"❌ {_fetch_error[0]}")
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                lst.addItem(item)
                return
            if not amis:
                item = QListWidgetItem(ctx.tr("cloud_ami_browse_empty"))
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                lst.addItem(item)
                info_lbl.setText(ctx.tr("cloud_ami_browse_empty"))
                return
            info_lbl.setText(f"{len(amis)} AMI(s)")
            for a in amis:
                label = f"{a['id']}  {a['name']}  [{a['creation_date']}]"
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, a)
                lst.addItem(item)

        def _use_selected():
            item = lst.currentItem()
            if not item:
                QMessageBox.information(browse_dlg, ctx.tr("cloud_ami_select_title"),
                                        ctx.tr("cloud_ami_browse_no_selection"))
                return
            a = item.data(Qt.ItemDataRole.UserRole)
            if not a:
                return
            _apply_ami(a["id"], a.get("name", ""))
            browse_dlg.accept()

        btn_use.clicked.connect(_use_selected)
        lst.itemDoubleClicked.connect(lambda _: _use_selected())
        threading.Thread(target=_fetch, daemon=True).start()
        QTimer.singleShot(300, _poll_ami_list)
        browse_dlg.exec()

    # Async key pairs + SGs
    _kp_result: list[list[str] | None] = [None]
    _sg_result: list[list[dict] | None] = [None]
    _kp_gen = [0]  # generation counter — incremented on each reload
    _sg_gen = [0]

    def _fetch_keypairs(reg: str, prof: str | None):
        names, _ = get_key_pairs(reg, prof)
        _kp_result[0] = names

    def _fetch_sgs(reg: str, prof: str | None):
        sgs, _ = get_security_groups(reg, prof)
        _sg_result[0] = sgs

    def _apply_keypairs(gen: int = 0):
        if gen != _kp_gen[0]:
            return  # stale, a newer reload already started
        names = _kp_result[0]
        if names is None:
            QTimer.singleShot(250, lambda: _apply_keypairs(gen))
            return
        saved_kn = get_key_name_for_profile_region(ctx.prof() or "default", ctx.reg())
        with QSignalBlocker(combo_key_pair):
            combo_key_pair.clear()
            combo_key_pair.addItem(ctx.tr("cloud_keypair_none"), "")
            for n in names:
                combo_key_pair.addItem(n, n)
            combo_key_pair.setEnabled(True)
            # Only search by name when a name was actually saved; otherwise
            # findData("") would match the "None" placeholder (index 0) and
            # the fallback to the first real key pair would never trigger.
            idx = combo_key_pair.findData(saved_kn) if saved_kn else -1
            if idx >= 0:
                combo_key_pair.setCurrentIndex(idx)
            elif combo_key_pair.count() > 1:
                # Default to the first real key pair so a new user can SSH
                # into the instance without extra configuration steps.
                combo_key_pair.setCurrentIndex(1)
        current_item = lst_instances.currentItem()
        if current_item:
            _prefill_launch_params_from_instance(current_item.data(Qt.ItemDataRole.UserRole))

    def _apply_sgs(gen: int = 0):
        if gen != _sg_gen[0]:
            return  # stale
        sgs = _sg_result[0]
        if sgs is None:
            QTimer.singleShot(250, lambda: _apply_sgs(gen))
            return
        saved_sg = get_sg_for_profile_region(ctx.prof() or "default", ctx.reg())
        with QSignalBlocker(combo_sg):
            combo_sg.clear()
            combo_sg.addItem(ctx.tr("cloud_sg_none"), "")
            for sg in sgs:
                label = f"{sg['id']}  ({sg['name']})" if sg["name"] else sg["id"]
                combo_sg.addItem(label, sg["id"])
            combo_sg.setEnabled(True)
            # Only search by ID when one was actually saved; findData("") would
            # match the "None" placeholder and suppress the auto-select fallback.
            idx = combo_sg.findData(saved_sg) if saved_sg else -1
            if idx >= 0:
                combo_sg.setCurrentIndex(idx)
            elif combo_sg.count() > 1:
                # Auto-select the first available security group for new regions.
                combo_sg.setCurrentIndex(1)
        current_item = lst_instances.currentItem()
        if current_item:
            _prefill_launch_params_from_instance(current_item.data(Qt.ItemDataRole.UserRole))

    def _save_key_sg():
        prof = ctx.prof() or "default"
        reg = ctx.reg()
        set_key_name_for_profile_region(prof, reg, str(combo_key_pair.currentData() or ""))
        set_sg_for_profile_region(prof, reg, str(combo_sg.currentData() or ""))

    combo_key_pair.currentIndexChanged.connect(lambda _: _save_key_sg())
    combo_sg.currentIndexChanged.connect(lambda _: _save_key_sg())
    btn_detect_ami.clicked.connect(_detect_ami)
    btn_browse_amis.clicked.connect(_browse_amis)
    btn_clear_ami.clicked.connect(_clear_ami)

    # When the region changes — reload key pairs, SGs and AMI for the new region
    _region_reload_timer = QTimer(dialog)
    _region_reload_timer.setSingleShot(True)
    _region_reload_timer.setInterval(600)

    def _on_region_changed_reload():
        """Re-fetch key pairs and SGs for the newly selected region."""
        _kp_result[0] = None
        _sg_result[0] = None
        _kp_gen[0] += 1
        _sg_gen[0] += 1
        kp_g = _kp_gen[0]
        sg_g = _sg_gen[0]
        # Block signals so _save_key_sg does NOT fire during clear/reset
        # (otherwise it would overwrite the saved key/sg for the new region
        # with an empty string before _apply_keypairs can restore it).
        with QSignalBlocker(combo_key_pair), QSignalBlocker(combo_sg):
            combo_key_pair.setEnabled(False)
            combo_key_pair.clear()
            combo_key_pair.addItem(ctx.tr("cloud_resources_loading"), "")
            combo_sg.setEnabled(False)
            combo_sg.clear()
            combo_sg.addItem(ctx.tr("cloud_resources_loading"), "")
        threading.Thread(target=_fetch_keypairs, args=(ctx.reg(), ctx.prof()), daemon=True).start()
        threading.Thread(target=_fetch_sgs, args=(ctx.reg(), ctx.prof()), daemon=True).start()
        QTimer.singleShot(250, lambda: _apply_keypairs(kp_g))
        QTimer.singleShot(250, lambda: _apply_sgs(sg_g))
        _load_current_ami()

    _region_reload_timer.timeout.connect(_on_region_changed_reload)
    _region_slot = lambda _: _region_reload_timer.start()
    ctx.region_combo.currentIndexChanged.connect(_region_slot)
    dialog.finished.connect(lambda _: ctx.region_combo.currentIndexChanged.disconnect(_region_slot))

    threading.Thread(target=_fetch_keypairs, args=(ctx.reg(), ctx.prof()), daemon=True).start()
    threading.Thread(target=_fetch_sgs, args=(ctx.reg(), ctx.prof()), daemon=True).start()
    QTimer.singleShot(250, lambda: _apply_keypairs(_kp_gen[0]))
    QTimer.singleShot(250, lambda: _apply_sgs(_sg_gen[0]))
    _load_current_ami()

    # ── Launch New Instance ───────────────────────────────────────────
    grp_launch = QGroupBox(ctx.tr("cloud_group_launch"), dialog)
    launch_vbox = QVBoxLayout(grp_launch)
    launch_vbox.setContentsMargins(14, 10, 14, 14)
    launch_vbox.setSpacing(8)

    launch_row_market = QHBoxLayout()
    launch_row_market.setSpacing(8)
    lbl_launch_market = QLabel(ctx.tr("cloud_launch_type_label"), dialog)
    lbl_launch_market.setStyleSheet("color: #8fa8c0;")
    lbl_launch_market.setFixedWidth(120)
    market_widget = QWidget(dialog)
    market_inner = QHBoxLayout(market_widget)
    market_inner.setContentsMargins(0, 0, 0, 0)
    market_inner.setSpacing(14)
    radio_launch_spot = QRadioButton(ctx.tr("cloud_market_spot_label"), market_widget)
    radio_launch_ondemand = QRadioButton(ctx.tr("cloud_market_ondemand_label"), market_widget)
    radio_launch_spot.setChecked(True)
    radio_launch_spot.setToolTip(ctx.tr("cloud_tt_market_type"))
    radio_launch_ondemand.setToolTip(ctx.tr("cloud_tt_market_type"))
    market_inner.addWidget(radio_launch_spot)
    market_inner.addWidget(radio_launch_ondemand)
    market_inner.addStretch()
    launch_row_market.addWidget(lbl_launch_market)
    launch_row_market.addWidget(market_widget, 1)
    launch_vbox.addLayout(launch_row_market)

    launch_row_type = QHBoxLayout()
    launch_row_type.setSpacing(8)
    lbl_launch_type = QLabel(ctx.tr("cloud_launch_instance_type_label"), dialog)
    lbl_launch_type.setStyleSheet("color: #8fa8c0;")
    lbl_launch_type.setFixedWidth(120)

    class _LaunchTypePriceDelegate(QStyledItemDelegate):
        _separator = "  -  "
        _price_color = QColor("#59d66f")
        _selected_price_color = QColor("#8bf29d")

        def paint(self, painter, option, index):
            text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
            if self._separator not in text:
                super().paint(painter, option, index)
                return

            prefix, price = text.split(self._separator, 1)
            prefix = f"{prefix}{self._separator}"

            opt = QStyleOptionViewItem(option)
            self.initStyleOption(opt, index)
            opt.text = ""
            style = opt.widget.style() if opt.widget else QApplication.style()
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)
            text_rect = style.subElementRect(
                QStyle.SubElement.SE_ItemViewItemText,
                opt,
                opt.widget,
            )
            metrics = opt.fontMetrics
            prefix_width = metrics.horizontalAdvance(prefix)
            price_width = metrics.horizontalAdvance(price)

            if prefix_width + price_width > text_rect.width():
                reserved_price_width = min(price_width, max(0, text_rect.width() // 2))
                prefix = metrics.elidedText(
                    prefix,
                    Qt.TextElideMode.ElideRight,
                    max(0, text_rect.width() - reserved_price_width),
                )
                prefix_width = metrics.horizontalAdvance(prefix)
                price = metrics.elidedText(
                    price,
                    Qt.TextElideMode.ElideRight,
                    max(0, text_rect.width() - prefix_width),
                )

            normal_color = opt.palette.color(
                QPalette.ColorRole.HighlightedText
                if opt.state & QStyle.StateFlag.State_Selected
                else QPalette.ColorRole.Text
            )
            price_color = (
                self._selected_price_color
                if opt.state & QStyle.StateFlag.State_Selected
                else self._price_color
            )

            painter.save()
            painter.setClipRect(text_rect)
            painter.setPen(normal_color)
            painter.drawText(
                text_rect,
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                prefix,
            )
            painter.setPen(price_color)
            price_rect = text_rect.adjusted(prefix_width, 0, 0, 0)
            painter.drawText(
                price_rect,
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                price,
            )
            painter.restore()

    combo_launch_type = QComboBox(dialog)
    combo_launch_type.addItem(ctx.tr("cloud_cost_loading"), "")
    combo_launch_type.setEnabled(False)
    combo_launch_type.setMinimumWidth(390)
    combo_launch_type.setMaximumWidth(470)
    combo_launch_type.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    combo_launch_type.setFixedHeight(36)
    combo_launch_type.setStyleSheet(_COMBO_QSS)
    combo_launch_type.setItemDelegate(_LaunchTypePriceDelegate(combo_launch_type))
    if hasattr(QComboBox, "LabelDrawingMode"):
        combo_launch_type.setLabelDrawingMode(QComboBox.LabelDrawingMode.UseDelegate)
    combo_launch_type.view().setMinimumWidth(425)
    combo_launch_type.setToolTip(ctx.tr("cloud_tt_instance_type"))
    btn_launch_from_manager = QPushButton(ctx.tr("cloud_btn_launch"), dialog)
    btn_launch_from_manager.setMinimumWidth(180)
    btn_launch_from_manager.setFixedHeight(36)
    btn_launch_from_manager.setToolTip(ctx.tr("cloud_tt_btn_launch"))
    launch_row_type.addWidget(lbl_launch_type)
    launch_row_type.addWidget(combo_launch_type)
    launch_row_type.addSpacing(16)
    launch_row_type.addStretch()
    launch_row_type.addWidget(btn_launch_from_manager)
    launch_vbox.addLayout(launch_row_type)
    dialog_root.addWidget(grp_launch)

    grp_instances = QGroupBox(ctx.tr("cloud_manager_list_title"), dialog)
    instances_vbox = QVBoxLayout(grp_instances)
    instances_vbox.setContentsMargins(14, 10, 14, 14)
    instances_vbox.setSpacing(8)
    lst_instances = QListWidget(dialog)
    lst_instances.setMinimumHeight(220)
    lst_instances.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    lst_instances.setStyleSheet("""
        QListWidget {
            background: #0e1218;
            color: #c8d8e8;
            border: 1px solid #1e2730;
            border-radius: 4px;
            font-size: 12px;
            outline: none;
        }
        QListWidget::item {
            padding: 5px 8px;
            border-bottom: 1px solid #161c24;
        }
        QListWidget::item:selected {
            background: #1e3248;
            color: #e8f0f8;
            border-bottom: 1px solid #1e3248;
        }
        QListWidget::item:hover:!selected {
            background: #141c26;
        }
        QScrollBar:vertical {
            background: #0e1218; width: 8px; border-radius: 4px;
        }
        QScrollBar::handle:vertical {
            background: #2a3a4d; min-height: 30px; border-radius: 4px;
        }
        QScrollBar::handle:vertical:hover { background: #3a5068; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QScrollBar:horizontal {
            background: #0e1218; height: 8px; border-radius: 4px;
        }
        QScrollBar::handle:horizontal {
            background: #2a3a4d; min-width: 30px; border-radius: 4px;
        }
        QScrollBar::handle:horizontal:hover { background: #3a5068; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
    """)
    instances_vbox.addWidget(lst_instances, 1)

    manager_actions = QHBoxLayout()
    manager_actions.setSpacing(8)
    btn_refresh_manager = QPushButton(ctx.tr("cloud_btn_refresh"), dialog)
    manager_actions.addWidget(btn_refresh_manager)
    lbl_ctx_hint = QLabel(ctx.tr("cloud_manager_ctx_hint"), dialog)
    lbl_ctx_hint.setStyleSheet("color: #5a7a99; font-size: 11px;")
    manager_actions.addWidget(lbl_ctx_hint)
    manager_actions.addStretch()
    instances_vbox.addLayout(manager_actions)
    dialog_root.addWidget(grp_instances, 1)

    bb_close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
    bb_close.rejected.connect(dialog.reject)
    dialog_root.addWidget(bb_close)

    _launch_types_result: list[list[str] | None] = [None]
    _launch_types_request_id = [0]

    def _launch_type_label(
        instance_type: str,
        price_info: dict | None = None,
        loading: bool = False,
    ) -> str:
        label = format_instance_type_label(instance_type)
        if loading:
            return f"{label}  -  ..."
        if not price_info or price_info.get("price") is None:
            return label
        suffix = f"${price_info['price']:.4f}/hr"
        if radio_launch_spot.isChecked() and price_info.get("az"):
            suffix += f"  [{price_info['az']}]"
        return f"{label}  -  {suffix}"

    def _refresh_launch_prices(request_id: int, instance_types: list[str]):
        price_state = {
            "prices": {},
            "done": False,
        }

        def _fetch_prices():


            def _fetch_one(instance_type: str) -> tuple[str, dict]:
                if radio_launch_spot.isChecked():
                    return instance_type, get_spot_price(instance_type, ctx.reg(), ctx.prof())
                return instance_type, get_ondemand_price(instance_type, ctx.reg(), ctx.prof())

            max_workers = min(6, max(1, len(instance_types)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_fetch_one, instance_type) for instance_type in instance_types]
                for future in as_completed(futures):
                    instance_type, price_info = future.result()
                    price_state["prices"][instance_type] = price_info
            price_state["done"] = True

        def _apply_prices():
            if request_id != _launch_types_request_id[0]:
                return
            prices = price_state["prices"]
            for index in range(combo_launch_type.count()):
                instance_type = str(combo_launch_type.itemData(index) or "")
                if not instance_type:
                    continue
                price_info = prices.get(instance_type)
                combo_launch_type.setItemText(
                    index,
                    _launch_type_label(
                        instance_type,
                        price_info,
                        loading=(price_info is None and not price_state["done"]),
                    ),
                )
            combo_launch_type.view().update()
            if not price_state["done"]:
                QTimer.singleShot(250, _apply_prices)

        threading.Thread(target=_fetch_prices, daemon=True).start()
        QTimer.singleShot(250, _apply_prices)

    def _refresh_launch_types():
        _launch_types_request_id[0] += 1
        request_id = _launch_types_request_id[0]
        _launch_types_result[0] = None
        combo_launch_type.clear()
        combo_launch_type.addItem(ctx.tr("cloud_cost_loading"), "")
        combo_launch_type.setEnabled(False)

        def _fetch():
            _launch_types_result[0] = get_available_gpu_types(
                ctx.reg(),
                ctx.prof(),
                use_spot=radio_launch_spot.isChecked(),
            )

        def _apply():
            if request_id != _launch_types_request_id[0]:
                return
            types = _launch_types_result[0]
            if types is None:
                QTimer.singleShot(250, _apply)
                return
            current_type = str(combo_launch_type.currentData() or "")
            combo_launch_type.clear()
            for instance_type in (types or ["g5.xlarge"]):
                combo_launch_type.addItem(
                    _launch_type_label(instance_type, loading=True),
                    instance_type,
                )
            combo_launch_type.setEnabled(True)
            idx = combo_launch_type.findData(current_type) if current_type else -1
            if idx < 0:
                idx = combo_launch_type.findData("g5.xlarge")
            combo_launch_type.setCurrentIndex(idx if idx >= 0 else 0)
            _refresh_launch_prices(request_id, types or ["g5.xlarge"])

        threading.Thread(target=_fetch, daemon=True).start()
        QTimer.singleShot(250, _apply)

    def _render_instance_label(inst: dict) -> str:
        name = inst.get("name") or "(no name)"
        ip = inst.get("public_ip") or ""
        market = ctx.market_label(inst.get("market", "on-demand"))
        return (
            f"{inst['id']}  [{format_instance_type_label(inst.get('instance_type', ''))} / {market}]  "
            f"{inst.get('state', '')}  {name}  {ip}"
        )

    def _prefill_launch_params_from_instance(inst: dict):
        """Sync launch selectors with parameters of an existing instance."""
        if not inst:
            return
        key_name = str(inst.get("key_name") or "")
        sg_ids = inst.get("security_group_ids") or []
        sg_id = str(sg_ids[0]) if sg_ids else ""

        if key_name:
            key_idx = combo_key_pair.findData(key_name)
            if key_idx >= 0 and combo_key_pair.currentIndex() != key_idx:
                combo_key_pair.setCurrentIndex(key_idx)

        if sg_id:
            sg_idx = combo_sg.findData(sg_id)
            if sg_idx >= 0 and combo_sg.currentIndex() != sg_idx:
                combo_sg.setCurrentIndex(sg_idx)

    def _refresh_manager_instances():
        ctx.log(ctx.tr("cloud_log_searching").format(reg=ctx.reg(), prof=ctx.prof() or "default"))
        instances, err = list_instances(ctx.reg(), ctx.prof())
        lst_instances.clear()
        if err:
            ctx.log(f"❌ {err}")
            QMessageBox.warning(dialog, ctx.tr("cloud_manager_title"), err)
            return
        if not instances:
            ctx.log(ctx.tr("cloud_no_instances"))
            item = QListWidgetItem(ctx.tr("cloud_no_instances"))
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            lst_instances.addItem(item)
            return
        ctx.log(ctx.tr("cloud_log_found").format(n=len(instances)))
        instances = sorted(instances, key=lambda inst: inst.get("launch_time", ""), reverse=True)
        selected_item = None
        for inst in instances:
            item = QListWidgetItem(_render_instance_label(inst))
            item.setData(Qt.ItemDataRole.UserRole, inst)
            lst_instances.addItem(item)
            if inst["id"] == ctx.iid():
                selected_item = item

        if selected_item is None and lst_instances.count() > 0:
            selected_item = lst_instances.item(0)

        if selected_item is not None:
            lst_instances.setCurrentItem(selected_item)
            selected_inst = selected_item.data(Qt.ItemDataRole.UserRole)
            _prefill_launch_params_from_instance(selected_inst)

    def _get_selected_manager_instance() -> dict | None:
        item = lst_instances.currentItem()
        inst = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not inst:
            QMessageBox.information(
                dialog,
                ctx.tr("cloud_manager_title"),
                ctx.tr("cloud_manager_no_selection"),
            )
            return None
        return inst

    def _use_selected_instance():
        inst = _get_selected_manager_instance()
        if not inst:
            return
        ctx.apply_selected(inst)
        dialog.accept()

    def _delete_selected_instance():
        inst = _get_selected_manager_instance()
        if not inst:
            return
        ans = QMessageBox.question(
            dialog,
            ctx.tr("cloud_manager_confirm_delete_title"),
            ctx.tr("cloud_manager_confirm_delete_msg").format(iid=inst["id"]),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        ok, msg = terminate_instance(inst["id"], ctx.reg(), ctx.prof())
        if ok:
            ctx.log(ctx.tr("cloud_log_terminate_ok").format(iid=inst["id"]))
            if inst["id"] == ctx.iid():
                ctx.poll_timer.stop()
                ctx.edit_instance_id.clear()
                ctx.lbl_ip_value.clear()
                ctx.set_status(InstanceState.TERMINATED, ctx.state_labels[InstanceState.TERMINATED])
                ctx.schedule_price_refresh()
            _refresh_manager_instances()
        else:
            ctx.log(ctx.tr("cloud_log_terminate_err").format(msg=msg))
            QMessageBox.warning(dialog, ctx.tr("cloud_manager_title"), msg)

    def _create_ami_from_selected():
        inst = _get_selected_manager_instance()
        if not inst:
            return
        default_name = f"KeyFlowStudio-{inst['id']}-{inst.get('instance_type', 'ec2')}"
        name, ok_name = QInputDialog.getText(
            dialog,
            ctx.tr("cloud_create_ami_title"),
            ctx.tr("cloud_create_ami_prompt"),
            text=default_name,
        )
        if not ok_name or not name.strip():
            return
        ok, result = create_ami_from_instance(
            inst["id"],
            ctx.reg(),
            ctx.prof(),
            name.strip(),
            description=f"KeyFlow Studio image from {inst['id']}",
            no_reboot=True,
        )
        if ok:
            set_ami_for_profile_region(ctx.prof() or "default", ctx.reg(), result)
            _load_current_ami()
            ctx.log(ctx.tr("cloud_log_create_ami_ok").format(ami_id=result, iid=inst["id"]))
            QMessageBox.information(
                dialog,
                ctx.tr("cloud_create_ami_title"),
                ctx.tr("cloud_create_ami_done").format(ami_id=result),
            )
        else:
            ctx.log(ctx.tr("cloud_log_create_ami_err").format(msg=result))
            QMessageBox.warning(dialog, ctx.tr("cloud_create_ami_title"), result)

    def _change_type_for_selected():
        inst = _get_selected_manager_instance()
        if not inst:
            return
        if str(inst.get("state", "")).lower() != "stopped":
            QMessageBox.information(
                dialog,
                ctx.tr("cloud_change_type_title"),
                ctx.tr("cloud_change_type_need_stopped"),
            )
            return
        target_type = str(combo_launch_type.currentData() or "").strip()
        if not target_type:
            QMessageBox.information(
                dialog,
                ctx.tr("cloud_change_type_title"),
                ctx.tr("cloud_change_type_no_target"),
            )
            return
        if target_type == str(inst.get("instance_type") or ""):
            QMessageBox.information(
                dialog,
                ctx.tr("cloud_change_type_title"),
                ctx.tr("cloud_change_type_same_type"),
            )
            return
        ans = QMessageBox.question(
            dialog,
            ctx.tr("cloud_change_type_title"),
            ctx.tr("cloud_change_type_confirm").format(
                iid=inst["id"],
                old=inst.get("instance_type", ""),
                new=target_type,
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        ok, result = change_instance_type(inst["id"], ctx.reg(), ctx.prof(), target_type)
        if ok:
            ctx.log(ctx.tr("cloud_log_change_type_ok").format(
                iid=inst["id"],
                old=inst.get("instance_type", ""),
                new=target_type,
            ))
            _refresh_manager_instances()
        else:
            ctx.log(ctx.tr("cloud_log_change_type_err").format(msg=result))
            QMessageBox.warning(dialog, ctx.tr("cloud_change_type_title"), result)

    def _create_instance_from_manager():
        itype = str(combo_launch_type.currentData() or "").strip()
        if not itype:
            return
        if ctx.launch_new(itype, radio_launch_spot.isChecked()):
            dialog.accept()

    def _quick_create_instance():
        # One-click flow for new users:
        # 1) Prefer a saved AMI.
        # 2) Otherwise auto-detect a public GPU AMI.
        # 3) Force Spot + recommended g5.xlarge (fallback to first available type).
        prof = ctx.prof() or "default"
        reg = ctx.reg()
        saved_ami = get_ami_for_profile_region(prof, reg)

        if combo_key_pair.currentIndex() <= 0 and combo_key_pair.count() > 1:
            combo_key_pair.setCurrentIndex(1)

        preferred_idx = combo_launch_type.findData("g5.xlarge")
        if preferred_idx >= 0:
            combo_launch_type.setCurrentIndex(preferred_idx)
        elif combo_launch_type.count() > 0:
            combo_launch_type.setCurrentIndex(0)

        radio_launch_spot.setChecked(True)
        _save_key_sg()

        def _launch_now():
            itype = str(combo_launch_type.currentData() or "g5.xlarge").strip() or "g5.xlarge"
            ctx.log(ctx.tr("cloud_log_quick_create_start").format(
                reg=reg,
                prof=prof,
                itype=itype,
            ))
            if ctx.launch_new(itype, True):
                dialog.accept()

        if saved_ami:
            _launch_now()
            return

        btn_quick_create.setEnabled(False)
        btn_quick_create.setText(ctx.tr("cloud_quick_create_loading"))
        lbl_ami_value.setText(ctx.tr("cloud_ami_loading"))
        lbl_ami_value.setStyleSheet("font-family: monospace; color: #888888;")
        ctx.log(ctx.tr("cloud_log_public_ami_search").format(reg=reg, prof=prof))

        _qc_result: list[tuple | None] = [None]

        def _worker():
            _qc_result[0] = find_public_gpu_ami(reg, ctx.prof())

        def _poll_qc():
            if _qc_result[0] is None:
                QTimer.singleShot(200, _poll_qc)
                return
            _on_public_ami_ready(_qc_result[0])

        def _on_public_ami_ready(result: tuple[str, str, str]):
            btn_quick_create.setEnabled(True)
            btn_quick_create.setText(ctx.tr("cloud_btn_quick_create"))
            ami_id, name, err = result
            if err:
                ctx.log(ctx.tr("cloud_log_public_ami_search_err").format(msg=err))
                QMessageBox.warning(
                    dialog,
                    ctx.tr("cloud_manager_title"),
                    err,
                )
                lbl_ami_value.setText(ctx.tr("cloud_ami_not_set"))
                lbl_ami_value.setStyleSheet("font-family: monospace; color: #e06060;")
                return
            if not ami_id:
                ctx.log(ctx.tr("cloud_log_public_ami_not_found"))
                QMessageBox.warning(
                    dialog,
                    ctx.tr("cloud_manager_title"),
                    ctx.tr("cloud_log_public_ami_not_found"),
                )
                lbl_ami_value.setText(ctx.tr("cloud_ami_not_set"))
                lbl_ami_value.setStyleSheet("font-family: monospace; color: #e0a040;")
                return
            _apply_ami(ami_id, name)
            ctx.log(ctx.tr("cloud_log_public_ami_found").format(ami_id=ami_id, name=name or "—"))
            _launch_now()

        threading.Thread(target=_worker, daemon=True).start()
        QTimer.singleShot(200, _poll_qc)

    def _show_instance_context_menu(pos):
        item = lst_instances.itemAt(pos)
        inst = item.data(Qt.ItemDataRole.UserRole) if item else None
        menu = QMenu(dialog)
        state = str(inst.get("state", "")).lower() if inst else ""

        act_use = menu.addAction(ctx.tr("cloud_ctx_use_selected"))
        act_start = menu.addAction(ctx.tr("cloud_ctx_start"))
        act_stop = menu.addAction(ctx.tr("cloud_ctx_stop"))
        menu.addSeparator()
        act_create_ami = menu.addAction(ctx.tr("cloud_btn_create_ami"))
        act_change_type = menu.addAction(ctx.tr("cloud_btn_change_type"))
        menu.addSeparator()
        act_delete = menu.addAction(ctx.tr("cloud_btn_delete_instance"))

        if not inst:
            for a in (act_use, act_start, act_stop, act_create_ami, act_change_type, act_delete):
                a.setEnabled(False)
        else:
            act_start.setEnabled(state == "stopped")
            act_stop.setEnabled(state == "running")
            act_change_type.setEnabled(state == "stopped")

        chosen = menu.exec(lst_instances.mapToGlobal(pos))
        if chosen == act_use:
            _use_selected_instance()
        elif chosen == act_start:
            ok, msg = start_instance(inst["id"], ctx.reg(), ctx.prof())
            if ok:
                ctx.log(ctx.tr("cloud_log_start_ok").format(iid=inst["id"]))
                _refresh_manager_instances()
            elif msg == "INSUFFICIENT_CAPACITY":
                capacity_msg = ctx.tr("cloud_log_start_capacity").format(region=ctx.reg())
                ctx.log(capacity_msg)
                QMessageBox.warning(dialog, ctx.tr("cloud_manager_title"), capacity_msg)
            else:
                ctx.log(msg)
                QMessageBox.warning(dialog, ctx.tr("cloud_manager_title"), msg)
        elif chosen == act_stop:
            ok, msg = stop_instance(inst["id"], ctx.reg(), ctx.prof())
            if ok:
                ctx.log(ctx.tr("cloud_log_stop_ok").format(iid=inst["id"]))
                _refresh_manager_instances()
            else:
                ctx.log(msg)
                QMessageBox.warning(dialog, ctx.tr("cloud_manager_title"), msg)
        elif chosen == act_create_ami:
            _create_ami_from_selected()
        elif chosen == act_change_type:
            _change_type_for_selected()
        elif chosen == act_delete:
            _delete_selected_instance()

    lst_instances.customContextMenuRequested.connect(_show_instance_context_menu)
    lst_instances.currentItemChanged.connect(
        lambda cur, _prev: _prefill_launch_params_from_instance(
            cur.data(Qt.ItemDataRole.UserRole) if cur else None
        )
    )
    btn_refresh_manager.clicked.connect(_refresh_manager_instances)
    btn_launch_from_manager.clicked.connect(_create_instance_from_manager)
    btn_quick_create.clicked.connect(_quick_create_instance)
    radio_launch_spot.toggled.connect(lambda checked: checked and _refresh_launch_types())
    radio_launch_ondemand.toggled.connect(lambda checked: checked and _refresh_launch_types())
    lst_instances.itemDoubleClicked.connect(lambda _item: _use_selected_instance())

    _refresh_launch_types()
    _refresh_manager_instances()
    dialog.exec()
