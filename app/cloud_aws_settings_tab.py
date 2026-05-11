"""Cloud AWS Settings tab for the KeyFlow Studio Settings dialog.

All Amazon Web Services UI code lives here. If other cloud providers are added
(GCP, Azure, etc.) each gets its own settings-tab module so names stay
unambiguous.

Public API
----------
    create_cloud_aws_settings_tab(mixin, parent) -> (QWidget, refs_dict)
    _region_code(combo) -> str           # also importable for convenience
    _region_list_cache: dict[str | None, list[str]]  # module-level session cache
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from app.cloud_manager import (
    AWS_REGION_NAMES,
    InstanceState,
    build_cloud_worker_bundle_manifest,
    check_credentials,
    check_environment_ssh,
    format_instance_type_label,
    get_instance_launch_config,
    get_instance_state,
    get_monthly_costs,
    get_ondemand_price,
    get_regions_with_gpu_quota,
    get_spot_price,
    get_system_logs_http,
    get_watchdog_status_ssh,
    install_environment_ssh,
    install_watchdog_ssh,
    launch_instance,
    save_aws_credentials_to_file,
    start_instance,
    stop_instance,
)
from app.cloud_settings import (
    get_ami_for_profile_region,
    get_cloud_setting,
    get_key_name_for_profile_region,
    get_sg_for_profile_region,
    save_cloud_settings,
    set_cloud_setting,
)
from app.cloud_instance_manager_dialog import InstanceManagerContext, open_instance_manager

logger = logging.getLogger(__name__)

# Cached region list — dict keyed by AWS profile; fetched once per profile per session.
_region_list_cache: dict[str | None, list[str]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Region combo helpers
# ─────────────────────────────────────────────────────────────────────────────

def _region_code(combo: QComboBox) -> str:
    """Read region code from the region combo regardless of display format.

    Items are stored as 'eu-west-1 — Ireland'; currentData() holds the bare
    code when selected from the list.  For manually typed text we split on
    ' — ' so the user can still type just the code or the full label.
    """
    data = combo.currentData()
    if data:
        return str(data).strip()
    return combo.currentText().split(" — ")[0].strip()

def _green_dot_icon(size: int = 10) -> QIcon:
    """Return a small QIcon with a filled green circle."""
    px = QPixmap(size, size)
    px.fill(QColor(0, 0, 0, 0))  # transparent background
    painter = QPainter(px)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#4CAF50"))
    painter.setPen(QColor("#388E3C"))
    painter.drawEllipse(1, 1, size - 2, size - 2)
    painter.end()
    return QIcon(px)

def _fill_region_combo(combo: QComboBox, regions: list[str], current_region: str) -> None:
    """Populate *combo* with region items and restore *current_region* selection.
    Called both for initial single-item fill and after async AWS fetch."""
    blocker = QSignalBlocker(combo)
    combo.clear()
    dot = _green_dot_icon()
    for code in regions:
        city = AWS_REGION_NAMES.get(code)
        display = f"{code} — {city}" if city else code
        combo.addItem(display, code)
    idx = next(
        (i for i in range(combo.count()) if combo.itemData(i) == current_region),
        -1,
    )
    combo.setCurrentIndex(idx if idx >= 0 else 0)
    if idx >= 0:
        combo.setItemIcon(idx, dot)
    del blocker

def _populate_cloud_region_combo(
    combo: QComboBox,
    profile: str | None,
    current_region: str,
) -> None:
    """Fill combo with region list; uses per-profile session cache."""
    fallback_region = current_region or "eu-west-1"

    if profile in _region_list_cache:
        # Already fetched for this profile — apply immediately
        _fill_region_combo(combo, _region_list_cache[profile], fallback_region)
        return

    result_holder: list[list[str] | None] = [None]

    def _do_fetch():
        regions, _err = get_regions_with_gpu_quota(profile=profile, fallback_region=fallback_region)
        choices = sorted(set(list(regions) + [fallback_region]))
        _region_list_cache[profile] = choices
        result_holder[0] = choices

    def _apply():
        if result_holder[0] is None:
            QTimer.singleShot(150, _apply)
            return
        try:
            _fill_region_combo(combo, result_holder[0], fallback_region)
        except RuntimeError:
            # combo's C++ object was deleted (dialog closed before AWS responded)
            pass

    threading.Thread(target=_do_fetch, daemon=True).start()
    QTimer.singleShot(150, _apply)

def _create_cloud_region_combo(
    mixin,
    parent: QWidget,
    profile: str | None,
    current_region: str,
) -> QComboBox:
    """Create the region combo.  Shows current region immediately; full list
    loads from AWS in background so the dialog opens without blocking."""
    combo = QComboBox(parent)
    combo.setFixedWidth(160)
    # Make the drop-down popup wider than the combo itself so the full
    # "eu-west-1 — Ireland" label is readable without truncation.
    combo.view().setMinimumWidth(240)
    combo.setToolTip(mixin._tr("cloud_tt_region"))

    def _update_dot(idx: int) -> None:
        """Move the green dot to whichever item is currently selected."""
        dot = _green_dot_icon()
        empty = QIcon()
        for i in range(combo.count()):
            combo.setItemIcon(i, dot if i == idx else empty)

    combo.currentIndexChanged.connect(_update_dot)

    # Instant display: just the saved region so dialog opens immediately
    fallback = current_region or "eu-west-1"
    _fill_region_combo(combo, [fallback], fallback)
    # Async: replace with full region list once AWS responds
    _populate_cloud_region_combo(combo, profile=profile, current_region=fallback)
    return combo

def _load_cloud_help_html(mixin) -> str:
    language_code = str(getattr(mixin, "_language_code", "ru") or "ru")
    roots: list[Path] = []
    bundle_root = getattr(sys, "_MEIPASS", "")
    if bundle_root:
        roots.append(Path(bundle_root))
    roots.append(Path(__file__).resolve().parents[1])

    candidates: list[Path] = []
    for root in roots:
        candidates.append(root / "docs" / f"cloud_aws_setup.{language_code}.html")
        if language_code != "ru":
            candidates.append(root / "docs" / "cloud_aws_setup.ru.html")
        if language_code != "en":
            candidates.append(root / "docs" / "cloud_aws_setup.en.html")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8")
        except Exception:
            logger.exception("Failed to load cloud help document: %s", candidate)

    return mixin._tr("cloud_help_text")



# ─────────────────────────────────────────────────────────────────────────────
# Main tab builder
# ─────────────────────────────────────────────────────────────────────────────

def create_cloud_aws_settings_tab(mixin, parent: QWidget):
    """Создаёт панель Cloud/AWS настроек. Возвращает (widget, refs_dict)."""
    widget = QWidget(parent)
    root = QVBoxLayout(widget)
    root.setContentsMargins(12, 12, 12, 8)
    root.setSpacing(10)

    # Guard flag: set to False when the tab widget is destroyed so in-flight
    # QTimer callbacks can bail out without touching deleted C++ objects.
    _alive = [True]
    widget.destroyed.connect(lambda: _alive.__setitem__(0, False))

    # ── Шапка: 2 колонки — слева [чекбокс + профиль], справа [расходы] ──
    header_outer = QHBoxLayout()
    header_outer.setSpacing(16)
    header_outer.setContentsMargins(0, 0, 0, 0)

    # Левая колонка
    header_left = QVBoxLayout()
    header_left.setSpacing(6)
    header_left.setContentsMargins(0, 0, 0, 0)

    # Строка 1: чекбокс + кнопка ℹ
    chk_row = QHBoxLayout()
    chk_row.setSpacing(8)
    chk_row.setContentsMargins(0, 0, 0, 0)
    check_enabled = QCheckBox(mixin._tr("cloud_enabled_label"), widget)
    check_enabled.setChecked(bool(get_cloud_setting("cloud/enabled")))
    check_enabled.setToolTip(mixin._tr("cloud_tt_enabled"))
    chk_row.addWidget(check_enabled)

    _info_icon_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "info.svg"
    )
    btn_cloud_help = QPushButton(widget)
    btn_cloud_help.setIcon(QIcon(_info_icon_path))
    btn_cloud_help.setIconSize(QSize(16, 16))
    btn_cloud_help.setFixedSize(24, 24)
    btn_cloud_help.setFlat(False)
    btn_cloud_help.setToolTip(mixin._tr("cloud_help_title"))
    def _show_cloud_help():
        dlg = QDialog(widget)
        dlg.setWindowTitle(mixin._tr("cloud_help_title"))
        dlg.resize(860, 760)
        dlg.setMinimumSize(720, 620)
        vb = QVBoxLayout(dlg)
        vb.setContentsMargins(20, 16, 20, 16)
        browser = QTextBrowser(dlg)
        browser.setOpenExternalLinks(True)
        browser.setOpenLinks(True)
        browser.setHtml(_load_cloud_help_html(mixin))
        browser.setStyleSheet(
            "QTextBrowser {"
            " background: #0e1218; color: #d9e3ef; border: 1px solid #223041;"
            " border-radius: 8px; padding: 12px; font-size: 13px; line-height: 1.45;"
            "}"
            "QScrollBar:vertical { background: #0e1218; width: 10px; border-radius: 5px; }"
            "QScrollBar::handle:vertical { background: #2a3a4d; min-height: 30px; border-radius: 5px; }"
            "QScrollBar::handle:vertical:hover { background: #3a5068; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )
        vb.addWidget(browser, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dlg)
        bb.rejected.connect(dlg.reject)
        vb.addWidget(bb)
        dlg.exec()
    btn_cloud_help.clicked.connect(_show_cloud_help)
    chk_row.addWidget(btn_cloud_help)
    chk_row.addStretch()
    header_left.addLayout(chk_row)

    # Строка 2: AWS профиль
    profile_row = QHBoxLayout()
    profile_row.setSpacing(8)
    profile_row.setContentsMargins(0, 0, 0, 0)
    lbl_aws_profile = QLabel(mixin._tr("cloud_aws_profile_label"), widget)
    lbl_aws_profile.setStyleSheet("color: #8fa8c0;")
    lbl_aws_profile.setFixedWidth(150)
    edit_aws_profile = QLineEdit(widget)
    edit_aws_profile.setText(str(get_cloud_setting("cloud/aws_profile") or ""))
    edit_aws_profile.setPlaceholderText("default")
    edit_aws_profile.setFixedWidth(160)
    edit_aws_profile.setFixedHeight(38)
    edit_aws_profile.setToolTip(mixin._tr("cloud_tt_aws_profile"))
    # Re-check credentials when profile changes (debounced)
    _profile_creds_timer = QTimer(widget)
    _profile_creds_timer.setSingleShot(True)
    _profile_creds_timer.setInterval(1500)
    edit_aws_profile.textChanged.connect(lambda _: _profile_creds_timer.start())
    _profile_creds_timer.timeout.connect(lambda: _check_and_show_creds_banner())
    profile_row.addWidget(lbl_aws_profile)
    profile_row.addWidget(edit_aws_profile)
    profile_row.addStretch()
    header_left.addLayout(profile_row)

    header_outer.addLayout(header_left)
    header_outer.addStretch()

    # Правая колонка: блок расходов AWS
    cost_widget = QWidget(widget)
    cost_layout = QVBoxLayout(cost_widget)
    cost_layout.setContentsMargins(0, 0, 0, 0)
    cost_layout.setSpacing(3)

    _cost_label_style = "color: #8fa8c0; font-size: 11px;"
    _cost_value_style = "color: #e0e8f0; font-size: 11px;"
    _loading = mixin._tr("cloud_cost_loading")

    def _make_cost_row(key):
        row = QHBoxLayout()
        row.setSpacing(8)
        lbl = QLabel(mixin._tr(key), cost_widget)
        lbl.setStyleSheet(_cost_label_style)
        val = QLabel(_loading, cost_widget)
        val.setStyleSheet(_cost_value_style)
        val.setMinimumWidth(60)
        val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(lbl)
        row.addWidget(val)
        return row, lbl, val

    row_m, _lbl_cost_month_name, lbl_cost_month = _make_cost_row("cloud_cost_month")
    row_y, _lbl_cost_yday_name, lbl_cost_yday  = _make_cost_row("cloud_cost_yesterday")
    row_f, _lbl_cost_fc_name, lbl_cost_fc      = _make_cost_row("cloud_cost_forecast")
    row_s, lbl_cost_price_name, lbl_cost_price = _make_cost_row("cloud_cost_current_price")
    cost_layout.addLayout(row_s)
    cost_layout.addLayout(row_m)
    cost_layout.addLayout(row_y)
    cost_layout.addLayout(row_f)

    header_outer.addWidget(cost_widget)
    root.addLayout(header_outer)

    # Фоновая загрузка расходов и цены текущего инстанса
    _cost_result: list[dict | None] = [None]
    _price_result: list[dict | None] = [None]
    _price_request_id = [0]
    _cost_loaded = [False]

    def _do_fetch_costs():
        profile = edit_aws_profile.text().strip() or None
        _cost_result[0] = get_monthly_costs(profile)

    threading.Thread(target=_do_fetch_costs, daemon=True).start()

    _cost_poll_timer = QTimer(widget)
    _cost_poll_timer.setInterval(300)

    def _poll_cost_ready():
        if _price_result[0] is not None:
            price_result = _price_result[0]
            _price_result[0] = None
            if price_result.get("request_id") == _price_request_id[0]:
                instance_type = price_result.get("instance_type")
                use_spot = price_result.get("use_spot")
                if instance_type and use_spot is not None:
                    market_text = mixin._tr("cloud_market_spot_label") if use_spot else mixin._tr("cloud_market_ondemand_label")
                    lbl_cost_price_name.setText(
                        mixin._tr("cloud_cost_current_price_fmt").format(
                            market=market_text,
                            instance_type=instance_type,
                        )
                    )
                else:
                    lbl_cost_price_name.setText(mixin._tr("cloud_cost_current_price"))

                if price_result.get("error"):
                    err = str(price_result["error"])
                    lbl_cost_price.setText(err[:50])
                    lbl_cost_price.setStyleSheet("color: #e06060; font-size: 11px;")
                    lbl_cost_price.setToolTip(err)
                else:
                    price_info = price_result.get("price_info") or {}
                    if price_info.get("price") is not None:
                        text = f"${price_info['price']:.4f}/hr"
                        if use_spot and price_info.get("az"):
                            text += f"  ({price_info['az']})"
                        lbl_cost_price.setText(text)
                        lbl_cost_price.setStyleSheet("color: #4cde8a; font-size: 11px; font-weight: bold;")
                        lbl_cost_price.setToolTip(
                            f"{instance_type} / "
                            f"{mixin._tr('cloud_market_spot_label') if use_spot else mixin._tr('cloud_market_ondemand_label')}"
                        )
                    else:
                        lbl_cost_price.setText(mixin._tr("cloud_cost_na"))
                        lbl_cost_price.setStyleSheet(_cost_value_style)
                        lbl_cost_price.setToolTip(price_info.get("error") or "")

        if _cost_loaded[0] or _cost_result[0] is None:
            return  # ещё не готово
        _cost_loaded[0] = True
        result = _cost_result[0]
        na = mixin._tr("cloud_cost_na")
        err = result.get("error")
        if err:
            short_err = err[:60] + ("…" if len(err) > 60 else "")
            lbl_cost_month.setText(short_err)
            lbl_cost_month.setStyleSheet("color: #e06060; font-size: 11px;")
            lbl_cost_month.setToolTip(err)
            lbl_cost_yday.setText("")
            lbl_cost_fc.setText("")
        else:
            lbl_cost_month.setText(result["month"] or na)
            lbl_cost_yday.setText(result["yesterday"] or na)
            lbl_cost_fc.setText(result["forecast"] or na)

    _cost_poll_timer.timeout.connect(_poll_cost_ready)
    _cost_poll_timer.start()

    # ═════════════════════════════════════════════════════════════════════
    # CREDENTIALS BANNER
    # Shown when AWS credentials are missing or invalid for the current profile.
    # Hidden when credentials are OK.
    # ═════════════════════════════════════════════════════════════════════
    creds_banner = QWidget(widget)
    creds_banner.setStyleSheet(
        "QWidget { background: #2c1a00; border: 1px solid #cc7700;"
        " border-radius: 6px; padding: 4px; }"
    )
    creds_banner_layout = QHBoxLayout(creds_banner)
    creds_banner_layout.setContentsMargins(10, 6, 10, 6)
    creds_banner_layout.setSpacing(10)
    lbl_creds_warn = QLabel("", creds_banner)
    lbl_creds_warn.setWordWrap(True)
    lbl_creds_warn.setTextFormat(Qt.TextFormat.RichText)
    lbl_creds_warn.setStyleSheet("color: #ffcc55; font-size: 12px; background: transparent; border: none;")
    btn_setup_creds = QPushButton(mixin._tr("cloud_btn_setup_creds"), creds_banner)
    btn_setup_creds.setMinimumWidth(180)
    btn_setup_creds.setStyleSheet(
        "QPushButton { background: #cc7700; color: #fff; border: none;"
        " border-radius: 5px; padding: 5px 14px; font-weight: bold; }"
        "QPushButton:hover { background: #e08800; }"
    )
    creds_banner_layout.addWidget(lbl_creds_warn, 1)
    creds_banner_layout.addWidget(btn_setup_creds)
    creds_banner.setVisible(False)
    root.addWidget(creds_banner)

    # ── Credentials check helpers (defined early; _log defined later) ────
    def _open_credentials_dialog():
        """Modal dialog for entering and saving AWS Access Key / Secret."""
        prof_default = edit_aws_profile.text().strip() or "keyflow"
        cdlg = QDialog(widget)
        cdlg.setWindowTitle(mixin._tr("cloud_creds_dialog_title"))
        cdlg.setModal(True)
        cdlg.setMinimumWidth(540)
        cvb = QVBoxLayout(cdlg)
        cvb.setContentsMargins(20, 16, 20, 16)
        cvb.setSpacing(10)

        # Hint text
        hint = QLabel(mixin._tr("cloud_creds_hint"), cdlg)
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.TextFormat.RichText)
        hint.setStyleSheet("color: #8fa8c0; font-size: 12px;")
        cvb.addWidget(hint)

        # Form
        cform = QFormLayout()
        cform.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        cform.setSpacing(8)

        edit_key_id = QLineEdit(cdlg)
        edit_key_id.setPlaceholderText("AKIAIOSFODNN7EXAMPLE")
        edit_key_id.setFixedHeight(36)
        edit_key_id.setToolTip("AWS Access Key ID (starts with AKIA...)")
        cform.addRow(mixin._tr("cloud_creds_access_key_label"), edit_key_id)

        edit_secret = QLineEdit(cdlg)
        edit_secret.setEchoMode(QLineEdit.EchoMode.Password)
        edit_secret.setPlaceholderText("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        edit_secret.setFixedHeight(36)
        edit_secret.setToolTip("AWS Secret Access Key")
        cform.addRow(mixin._tr("cloud_creds_secret_key_label"), edit_secret)

        edit_prof_name = QLineEdit(cdlg)
        edit_prof_name.setText(prof_default)
        edit_prof_name.setFixedHeight(36)
        edit_prof_name.setToolTip("Profile name in ~/.aws/credentials")
        cform.addRow(mixin._tr("cloud_creds_profile_label"), edit_prof_name)

        cvb.addLayout(cform)

        # Status label inside dialog
        dlg_status = QLabel("", cdlg)
        dlg_status.setWordWrap(True)
        dlg_status.setStyleSheet("font-size: 12px; color: #c8d8e8;")
        cvb.addWidget(dlg_status)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_verify_dlg = QPushButton(mixin._tr("cloud_creds_btn_verify"), cdlg)
        btn_verify_dlg.setMinimumWidth(100)
        btn_save_dlg = QPushButton(mixin._tr("cloud_creds_btn_save"), cdlg)
        btn_save_dlg.setMinimumWidth(100)
        btn_save_dlg.setStyleSheet(
            "QPushButton { background: #1a6b35; color: #fff; border: none;"
            " border-radius: 5px; padding: 6px 18px; font-weight: bold; }"
            "QPushButton:hover { background: #228040; }"
        )
        btn_cancel_dlg = QPushButton(mixin._tr("settings_cancel") if hasattr(mixin, "_tr") else "Cancel", cdlg)
        btn_cancel_dlg.setMinimumWidth(80)
        btn_row.addWidget(btn_verify_dlg)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel_dlg)
        btn_row.addWidget(btn_save_dlg)
        cvb.addLayout(btn_row)

        def _do_verify():
            kid = edit_key_id.text().strip()
            sec = edit_secret.text().strip()
            if not kid or not sec:
                dlg_status.setText(mixin._tr("cloud_creds_fill_both"))
                dlg_status.setStyleSheet("color: #e0a040; font-size: 12px;")
                return
            dlg_status.setText(mixin._tr("cloud_creds_verifying"))
            dlg_status.setStyleSheet("color: #8fa8c0; font-size: 12px;")
            cdlg.repaint()
            # Save temporarily to test
            _prof_name = edit_prof_name.text().strip() or "keyflow"
            ok_save, path = save_aws_credentials_to_file(kid, sec, _prof_name)
            if not ok_save:
                dlg_status.setText(mixin._tr("cloud_creds_save_err").format(msg=path))
                dlg_status.setStyleSheet("color: #e06060; font-size: 12px;")
                return
            ok, msg = check_credentials(_prof_name, _region_code(edit_region) or "us-east-1")
            if ok:
                dlg_status.setText(mixin._tr("cloud_creds_verify_ok").format(identity=msg))
                dlg_status.setStyleSheet("color: #4cde8a; font-size: 12px;")
            else:
                dlg_status.setText(mixin._tr("cloud_creds_verify_fail").format(msg=msg))
                dlg_status.setStyleSheet("color: #e06060; font-size: 12px;")

        def _do_save():
            kid = edit_key_id.text().strip()
            sec = edit_secret.text().strip()
            if not kid or not sec:
                dlg_status.setText(mixin._tr("cloud_creds_fill_both"))
                dlg_status.setStyleSheet("color: #e0a040; font-size: 12px;")
                return
            _prof_name = edit_prof_name.text().strip() or "keyflow"
            ok, result = save_aws_credentials_to_file(kid, sec, _prof_name)
            if ok:
                # Sync profile field in main tab
                edit_aws_profile.setText(_prof_name)
                _check_and_show_creds_banner()
                cdlg.accept()
            else:
                dlg_status.setText(mixin._tr("cloud_creds_save_err").format(msg=result))
                dlg_status.setStyleSheet("color: #e06060; font-size: 12px;")

        btn_verify_dlg.clicked.connect(_do_verify)
        btn_save_dlg.clicked.connect(_do_save)
        btn_cancel_dlg.clicked.connect(cdlg.reject)
        cdlg.exec()

    def _check_and_show_creds_banner():
        """Async credentials check; shows/hides banner based on result."""
        prof = edit_aws_profile.text().strip() or "default"
        reg = _region_code(edit_region) or "us-east-1"

        _creds_result_holder: list[tuple | None] = [None]

        def _worker():
            result = check_credentials(prof, reg)
            _creds_result_holder[0] = (result, prof)

        def _poll_creds():
            if _creds_result_holder[0] is None:
                QTimer.singleShot(200, _poll_creds)
                return
            result, prof_name = _creds_result_holder[0]
            _apply_creds_result(result, prof_name)

        def _apply_creds_result(result: tuple[bool, str], prof_name: str):
            ok, msg = result
            if ok:
                creds_banner.setVisible(False)
                # Log credentials OK once on open (only when log is available)
                try:
                    _log(mixin._tr("cloud_log_creds_ok").format(identity=msg))
                except Exception:
                    pass
            else:
                # Choose message based on error type
                if msg == "no_credentials" or msg == "profile_not_found":
                    text = mixin._tr("cloud_creds_warn_msg").format(prof=prof_name)
                elif msg == "invalid_credentials":
                    text = mixin._tr("cloud_creds_invalid_msg").format(prof=prof_name)
                else:
                    text = mixin._tr("cloud_creds_warn_msg").format(prof=prof_name)
                lbl_creds_warn.setText(text)
                creds_banner.setVisible(True)
                try:
                    _log(mixin._tr("cloud_log_creds_fail").format(msg=msg))
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True).start()
        QTimer.singleShot(200, _poll_creds)

    btn_setup_creds.clicked.connect(_open_credentials_dialog)

    # ═════════════════════════════════════════════════════════════════════
    # GROUP: Connection Settings
    # ═════════════════════════════════════════════════════════════════════
    grp_conn = QGroupBox(mixin._tr("cloud_group_connection"), widget)
    conn_vbox = QVBoxLayout(grp_conn)
    conn_vbox.setContentsMargins(14, 10, 14, 14)
    conn_vbox.setSpacing(8)

    # Строка 1: Регион: [eu-west-1]   Instance ID: [i-xxx...]   [Менеджер]
    row1 = QHBoxLayout()
    row1.setSpacing(8)
    lbl_region = QLabel(mixin._tr("cloud_region_label"), widget)
    lbl_region.setStyleSheet("color: #8fa8c0;")
    lbl_region.setFixedWidth(80)
    current_region = str(get_cloud_setting("cloud/region") or "eu-west-1")
    edit_region = _create_cloud_region_combo(
        mixin,
        widget,
        profile=edit_aws_profile.text().strip() or None,
        current_region=current_region,
    )
    # Debounced region-change notification in the activity log
    _region_log_timer = QTimer(widget)
    _region_log_timer.setSingleShot(True)
    _region_log_timer.setInterval(1200)
    def _on_region_text_changed():
        _region_log_timer.start()
    def _on_region_committed():
        reg = _region_code(edit_region)
        if reg:
            _log(mixin._tr("cloud_log_region_changed").format(reg=reg))
    edit_region.currentTextChanged.connect(lambda _: _on_region_text_changed())
    edit_aws_profile.editingFinished.connect(
        lambda: _populate_cloud_region_combo(
            edit_region,
            profile=edit_aws_profile.text().strip() or None,
            current_region=_region_code(edit_region) or current_region,
        )
    )
    _region_log_timer.timeout.connect(_on_region_committed)
    lbl_instance = QLabel(mixin._tr("cloud_instance_id_label"), widget)
    lbl_instance.setStyleSheet("color: #8fa8c0;")
    lbl_instance.setFixedWidth(110)
    edit_instance_id = QLineEdit(widget)
    edit_instance_id.setPlaceholderText(mixin._tr("cloud_instance_id_placeholder"))
    edit_instance_id.setText(str(get_cloud_setting("cloud/instance_id") or ""))
    edit_instance_id.setToolTip(mixin._tr("cloud_tt_instance_id"))
    btn_find = QPushButton(mixin._tr("cloud_btn_find"), widget)
    btn_find.setFixedWidth(130)
    btn_find.setToolTip(mixin._tr("cloud_tt_find"))
    row1.addWidget(lbl_region)
    row1.addWidget(edit_region)
    row1.addSpacing(12)
    row1.addWidget(lbl_instance)
    row1.addWidget(edit_instance_id, 1)
    row1.addWidget(btn_find)
    conn_vbox.addLayout(row1)

    # Строка 2: SSH пользователь: [ubuntu]   SSH ключ (.pem): [path]   [Выбрать...]
    row2 = QHBoxLayout()
    row2.setSpacing(8)
    lbl_ssh_user = QLabel(mixin._tr("cloud_ssh_user_label"), widget)
    lbl_ssh_user.setStyleSheet("color: #8fa8c0;")
    lbl_ssh_user.setFixedWidth(80)
    edit_ssh_user = QLineEdit(widget)
    edit_ssh_user.setText(str(get_cloud_setting("cloud/ssh_user") or "ubuntu"))
    edit_ssh_user.setFixedWidth(120)
    edit_ssh_user.setToolTip(mixin._tr("cloud_tt_ssh_user"))
    lbl_ssh_key = QLabel(mixin._tr("cloud_ssh_key_label"), widget)
    lbl_ssh_key.setStyleSheet("color: #8fa8c0;")
    lbl_ssh_key.setFixedWidth(110)
    edit_ssh_key = QLineEdit(widget)
    edit_ssh_key.setText(str(get_cloud_setting("cloud/ssh_key_path") or ""))
    edit_ssh_key.setPlaceholderText("~/.ssh/keyflow-gpu.pem")
    edit_ssh_key.setToolTip(mixin._tr("cloud_tt_ssh_key"))
    btn_browse_key = QPushButton(mixin._tr("cloud_browse_key"), widget)
    btn_browse_key.setFixedWidth(130)
    btn_browse_key.setToolTip(mixin._tr("cloud_tt_browse_key"))
    def _browse_key():
        path, _ = QFileDialog.getOpenFileName(
            widget, mixin._tr("cloud_ssh_key_label"), "", "PEM files (*.pem);;All files (*)"
        )
        if path:
            edit_ssh_key.setText(path)
    btn_browse_key.clicked.connect(_browse_key)
    row2.addWidget(lbl_ssh_user)
    row2.addWidget(edit_ssh_user)
    row2.addSpacing(12)
    row2.addWidget(lbl_ssh_key)
    row2.addWidget(edit_ssh_key, 1)
    row2.addWidget(btn_browse_key)
    conn_vbox.addLayout(row2)

    root.addWidget(grp_conn)

    # ═════════════════════════════════════════════════════════════════════
    # GROUP: Instance Control
    # ═════════════════════════════════════════════════════════════════════
    grp_ctrl = QGroupBox(mixin._tr("cloud_group_control"), widget)
    ctrl_vbox = QVBoxLayout(grp_ctrl)
    ctrl_vbox.setContentsMargins(14, 10, 14, 14)
    ctrl_vbox.setSpacing(10)

    # Строка статуса: Статус: [■] Остановлен   IP адрес: —
    status_row = QHBoxLayout()
    status_row.setSpacing(6)
    _lbl_status_prefix = QLabel(mixin._tr("cloud_status_label"), widget)
    _lbl_status_prefix.setStyleSheet("color: #8fa8c0;")
    lbl_status_icon = QLabel(widget)
    lbl_status_icon.setFixedSize(14, 14)
    lbl_status_icon.setStyleSheet("background:#888888; border-radius:2px;")
    lbl_status_value = QLabel(mixin._tr("cloud_state_unknown"), widget)
    lbl_status_value.setStyleSheet("color: #888888; font-weight: bold; font-size: 13px;")
    _lbl_ip_prefix = QLabel(mixin._tr("cloud_ip_label"), widget)
    _lbl_ip_prefix.setStyleSheet("color: #8fa8c0; margin-left: 16px;")
    lbl_ip_value = QLineEdit(widget)
    lbl_ip_value.setText(str(get_cloud_setting("cloud/api_host") or ""))
    lbl_ip_value.setPlaceholderText("IP / hostname")
    lbl_ip_value.setMinimumWidth(160)
    lbl_ip_value.setToolTip(
        "Публичный IP или hostname EC2 инстанса.\n"
        "Заполняется автоматически при обнаружении нового IP.\n"
        "Можно вписать вручную — тогда AWS API не используется."
    )
    lbl_ip_value.setStyleSheet(
        "QLineEdit { font-family: monospace; color: #c8d8e8;"
        " background: transparent; border: 1px solid #2a3a4d;"
        " border-radius: 3px; padding: 1px 6px; }"
        "QLineEdit:focus { border: 1px solid #3a7bbf; }"
    )
    status_row.addWidget(_lbl_status_prefix)
    status_row.addWidget(lbl_status_icon)
    status_row.addWidget(lbl_status_value)
    status_row.addWidget(_lbl_ip_prefix)
    status_row.addWidget(lbl_ip_value)
    status_row.addStretch()
    ctrl_vbox.addLayout(status_row)

    # Строка кнопок: Запустить | Остановить | Обновить | Проверить окружение
    buttons_row = QHBoxLayout()
    buttons_row.setSpacing(8)
    btn_start     = QPushButton(mixin._tr("cloud_btn_start"), widget)
    btn_stop      = QPushButton(mixin._tr("cloud_btn_stop"), widget)
    btn_refresh   = QPushButton(mixin._tr("cloud_btn_refresh"), widget)
    btn_check_env = QPushButton(mixin._tr("cloud_btn_check_env"), widget)
    btn_start.setMinimumWidth(110)
    btn_stop.setMinimumWidth(110)
    btn_refresh.setMinimumWidth(90)
    btn_check_env.setMinimumWidth(150)
    btn_start.setToolTip(mixin._tr("cloud_tt_btn_start"))
    btn_stop.setToolTip(mixin._tr("cloud_tt_btn_stop"))
    btn_refresh.setToolTip(mixin._tr("cloud_tt_btn_refresh"))
    btn_check_env.setToolTip(mixin._tr("cloud_tt_btn_check_env"))
    btn_start.setStyleSheet(
        "QPushButton{background:#1a6b35;color:#ffffff;border:1px solid #1e8040;"
        "border-radius:6px;padding:6px 16px;font-weight:bold;}"
        "QPushButton:hover{background:#228040;}"
        "QPushButton:pressed{background:#145228;}"
    )
    btn_stop.setStyleSheet(
        "QPushButton{background:#b03030;color:#ffffff;border:1px solid #cc3333;"
        "border-radius:6px;padding:6px 16px;font-weight:bold;}"
        "QPushButton:hover{background:#c83838;}"
        "QPushButton:pressed{background:#8a2020;}"
    )
    buttons_row.addWidget(btn_start)
    buttons_row.addWidget(btn_stop)
    buttons_row.addWidget(btn_refresh)
    buttons_row.addWidget(btn_check_env)
    buttons_row.addStretch()
    ctrl_vbox.addLayout(buttons_row)

    root.addWidget(grp_ctrl)

    # ═════════════════════════════════════════════════════════════════════
    # GROUP: Watchdog (Auto-shutdown)
    # ═════════════════════════════════════════════════════════════════════
    grp_wd = QGroupBox(mixin._tr("cloud_group_watchdog"), widget)
    wd_row = QHBoxLayout(grp_wd)
    wd_row.setContentsMargins(14, 10, 14, 10)
    wd_row.setSpacing(10)

    wd_row.addWidget(QLabel(mixin._tr("cloud_watchdog_idle_label"), widget))
    spin_idle = QSpinBox(widget)
    spin_idle.setRange(5, 120)
    spin_idle.setValue(int(get_cloud_setting("cloud/watchdog_idle_min") or 15))
    spin_idle.setSuffix(" мин")
    spin_idle.setFixedWidth(90)
    spin_idle.setToolTip(mixin._tr("cloud_tt_spin_idle"))
    wd_row.addWidget(spin_idle)
    wd_row.addSpacing(12)
    wd_row.addWidget(QLabel(mixin._tr("cloud_watchdog_gpu_label"), widget))
    spin_gpu = QSpinBox(widget)
    spin_gpu.setRange(1, 50)
    spin_gpu.setValue(int(get_cloud_setting("cloud/watchdog_gpu_pct") or 5))
    spin_gpu.setSuffix(" %")
    spin_gpu.setFixedWidth(75)
    spin_gpu.setToolTip(mixin._tr("cloud_tt_spin_gpu"))
    wd_row.addWidget(spin_gpu)
    wd_row.addSpacing(16)
    btn_watchdog_install = QPushButton(mixin._tr("cloud_watchdog_install"), widget)
    btn_watchdog_status  = QPushButton(mixin._tr("cloud_watchdog_status"), widget)
    btn_watchdog_install.setFixedWidth(88)
    btn_watchdog_status.setFixedWidth(78)
    btn_watchdog_install.setToolTip(mixin._tr("cloud_tt_watchdog_install"))
    btn_watchdog_status.setToolTip(mixin._tr("cloud_tt_watchdog_status"))
    wd_row.addWidget(btn_watchdog_install)
    wd_row.addWidget(btn_watchdog_status)
    wd_row.addStretch()

    root.addWidget(grp_wd)

    # ═════════════════════════════════════════════════════════════════════
    # GROUP: Activity Log
    # ═════════════════════════════════════════════════════════════════════
    grp_log = QGroupBox(mixin._tr("cloud_group_log"), widget)
    log_vbox = QVBoxLayout(grp_log)
    log_vbox.setContentsMargins(8, 6, 8, 8)
    log_vbox.setSpacing(4)

    # Toolbar row: streaming indicator + clear button
    log_toolbar = QHBoxLayout()
    log_toolbar.setContentsMargins(0, 0, 0, 0)
    log_toolbar.setSpacing(6)
    lbl_log_stream = QLabel("", widget)
    lbl_log_stream.setStyleSheet("color: #556677; font-size: 10px;")
    log_toolbar.addWidget(lbl_log_stream, stretch=1)
    btn_log_clear = QPushButton(mixin._tr("cloud_log_clear"), widget)
    btn_log_clear.setFixedHeight(20)
    btn_log_clear.setStyleSheet(
        "QPushButton { background: #1a2330; color: #8fa8c0; border: 1px solid #2a3a4d;"
        " border-radius: 3px; font-size: 10px; padding: 0 8px; }"
        " QPushButton:hover { background: #243040; }"
    )
    log_toolbar.addWidget(btn_log_clear)
    log_vbox.addLayout(log_toolbar)

    log_edit = QPlainTextEdit(widget)
    log_edit.setReadOnly(True)
    log_edit.setMinimumHeight(80)
    log_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    log_edit.setStyleSheet("""
        QPlainTextEdit {
            background-color: rgba(11, 15, 21, 200);
            color: #c8d8e8;
            font-family: "SF Mono", "Menlo", "Consolas", monospace;
            font-size: 11px;
            border: none;
            border-radius: 4px;
            padding: 4px 6px;
        }
        QScrollBar:vertical {
            background: #0e1218; width: 8px; border-radius: 4px;
        }
        QScrollBar::handle:vertical {
            background: #2a3a4d; min-height: 30px; border-radius: 4px;
        }
        QScrollBar::handle:vertical:hover { background: #3a5068; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    """)
    log_vbox.addWidget(log_edit)
    root.addWidget(grp_log, stretch=1)

    # ── Вспомогательные функции ───────────────────────────────────────────
    _STATE_LABELS = {
        InstanceState.RUNNING:    mixin._tr("cloud_state_running"),
        InstanceState.STOPPED:    mixin._tr("cloud_state_stopped"),
        InstanceState.STOPPING:   mixin._tr("cloud_state_stopping"),
        InstanceState.PENDING:    mixin._tr("cloud_state_pending"),
        InstanceState.ERROR:      mixin._tr("cloud_state_error"),
        InstanceState.UNKNOWN:    mixin._tr("cloud_state_unknown"),
        InstanceState.TERMINATED: mixin._tr("cloud_state_terminated"),
    }
    _STATE_COLORS = {
        InstanceState.RUNNING:    "#4caf50",
        InstanceState.STOPPED:    "#e53935",
        InstanceState.STOPPING:   "#ff9800",
        InstanceState.PENDING:    "#ff9800",
        InstanceState.ERROR:      "#e53935",
        InstanceState.UNKNOWN:    "#888888",
        InstanceState.TERMINATED: "#888888",
    }

    def _set_status(state: InstanceState, label: str):
        color = _STATE_COLORS.get(state, "#c8d8e8")
        lbl_status_value.setText(label)
        lbl_status_value.setStyleSheet(f"color: {color}; font-weight: bold;")
        lbl_status_icon.setStyleSheet(f"background:{color}; border-radius:2px;")

    def _iid():  return edit_instance_id.text().strip()
    def _reg():  return _region_code(edit_region) or "eu-west-1"
    def _prof(): return edit_aws_profile.text().strip() or None
    def _key():  return edit_ssh_key.text().strip() or "~/.ssh/keyflow-gpu.pem"
    def _user(): return edit_ssh_user.text().strip() or "ubuntu"

    def _market_label(market: str) -> str:
        return "Spot" if str(market or "").lower() == "spot" else "On-Demand"

    def _schedule_current_price_refresh(instance_type: str | None = None, use_spot: bool | None = None):
        _price_request_id[0] += 1
        request_id = _price_request_id[0]
        lbl_cost_price.setText(_loading)
        lbl_cost_price.setStyleSheet(_cost_value_style)
        lbl_cost_price.setToolTip("")

        def _worker():
            resolved_type = instance_type
            resolved_use_spot = use_spot

            if (not resolved_type or resolved_use_spot is None) and _iid():
                info, err = get_instance_launch_config(_iid(), _reg(), _prof())
                if err:
                    _price_result[0] = {
                        "request_id": request_id,
                        "instance_type": None,
                        "use_spot": None,
                        "price_info": None,
                        "error": err,
                    }
                    return
                if info:
                    resolved_type = str(info.get("instance_type") or "")
                    resolved_use_spot = str(info.get("market") or "on-demand").lower() == "spot"

            if not resolved_type or resolved_use_spot is None:
                _price_result[0] = {
                    "request_id": request_id,
                    "instance_type": None,
                    "use_spot": None,
                    "price_info": None,
                    "error": None,
                }
                return

            price_info = (
                get_spot_price(resolved_type, _reg(), _prof())
                if resolved_use_spot
                else get_ondemand_price(resolved_type, _reg(), _prof())
            )
            _price_result[0] = {
                "request_id": request_id,
                "instance_type": resolved_type,
                "use_spot": resolved_use_spot,
                "price_info": price_info,
                "error": price_info.get("error"),
            }

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_selected_instance(inst: dict):
        edit_instance_id.setText(inst["id"])
        _pub_ip = inst.get("public_ip") or ""
        lbl_ip_value.setText(_pub_ip)
        if _pub_ip:
            set_cloud_setting("cloud/api_host", _pub_ip)
        sel_state = inst.get("state") or InstanceState.UNKNOWN.value
        if sel_state in InstanceState._value2member_map_:
            state_enum = InstanceState(sel_state)
            _set_status(state_enum, _STATE_LABELS.get(state_enum, sel_state))
        else:
            lbl_status_value.setText(sel_state)
        _log(mixin._tr("cloud_log_selected").format(
            iid=inst["id"],
            itype=(
                f"{format_instance_type_label(inst.get('instance_type', ''))} / "
                f"{_market_label(inst.get('market', 'on-demand'))}"
            ),
            state=sel_state,
            ip=inst.get("public_ip") or "",
        ))
        _schedule_current_price_refresh(
            str(inst.get("instance_type") or ""),
            str(inst.get("market") or "on-demand").lower() == "spot",
        )

    _poll_timer = QTimer(widget)
    _poll_timer.setInterval(8000)        # 8 s — more responsive than 20 s
    _DEFAULT_POLL_MAX = 40              # ~5 min at 8 s intervals
    _MAX_CONSECUTIVE_ERRORS = 4        # stop after 4 errors in a row
    _poll_attempts = [0]
    _poll_max = [_DEFAULT_POLL_MAX]
    _consecutive_errors = [0]          # reset to 0 on any non-ERROR state
    _last_poll_error = [""]            # last error message for diagnostics

    _watchdog_timer = QTimer(widget)
    _watchdog_timer.setSingleShot(True)
    _watchdog_retry = [0]
    _MAX_WATCHDOG_RETRIES = 5
    _WATCHDOG_RETRY_MS = 30_000

    # ── Remote log streaming ──────────────────────────────────────────────────
    _log_seq = [0]           # cursor: last seen sequence number on the worker
    _log_poll_timer = QTimer(widget)
    _log_poll_timer.setInterval(3000)  # poll every 3 s

    def _log(msg: str):
        logger.info("[Cloud] %s", msg)
        try:
            log_edit.appendPlainText(msg)
            sb = log_edit.verticalScrollBar()
            sb.setValue(sb.maximum())
        except RuntimeError:
            pass  # widget already destroyed (tab closed while background task was running)

    def _log_colored(prefix: str, colored_part: str, suffix: str, color: str):
        """Добавляет строку лога с цветным фрагментом через QTextCursor."""
        from PySide6.QtGui import QTextCharFormat, QColor
        logger.info("[Cloud] %s%s%s", prefix, colored_part, suffix)
        cursor = log_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        # Начинаем с новой строки если предыдущий текст не пустой
        if cursor.positionInBlock() > 0:
            cursor.insertBlock()
        # prefix
        if prefix:
            fmt_plain = QTextCharFormat()
            fmt_plain.setForeground(QColor("#c8d8e8"))
            cursor.insertText(prefix, fmt_plain)
        # colored part
        fmt_colored = QTextCharFormat()
        fmt_colored.setForeground(QColor(color))
        fmt_colored.setFontWeight(700)
        cursor.insertText(colored_part, fmt_colored)
        # suffix
        if suffix:
            fmt_plain2 = QTextCharFormat()
            fmt_plain2.setForeground(QColor("#c8d8e8"))
            cursor.insertText(suffix, fmt_plain2)
        # newline
        fmt_reset = QTextCharFormat()
        fmt_reset.setForeground(QColor("#c8d8e8"))
        fmt_reset.setFontWeight(400)
        cursor.insertText("\n", fmt_reset)
        log_edit.setTextCursor(cursor)
        sb = log_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    _refresh_lock = [False]  # prevents concurrent refreshes

    def _poll_remote_logs():
        """Fetch new worker log lines from /system/logs and append to log_edit."""
        if not _alive[0]:
            return
        ip = lbl_ip_value.text().strip()
        if not ip or ip == mixin._tr("cloud_no_ip"):
            return
        base_url = f"http://{ip}:8080"
        after = _log_seq[0]
        _result: list[tuple | None] = [None]

        def _fetch():
            _result[0] = get_system_logs_http(base_url, after_seq=after)

        def _check():
            if not _alive[0]:
                return
            if _result[0] is None:
                QTimer.singleShot(200, _check)
                return
            new_lines, next_seq = _result[0]
            _log_seq[0] = next_seq
            _ec2_log = logging.getLogger("ec2.worker")
            for line in new_lines:
                _ec2_log.info(line)

        threading.Thread(target=_fetch, daemon=True).start()
        QTimer.singleShot(200, _check)

    _log_poll_timer.timeout.connect(_poll_remote_logs)

    btn_log_clear.clicked.connect(log_edit.clear)

    def _set_log_streaming(active: bool) -> None:
        if active:
            lbl_log_stream.setText(mixin._tr("cloud_log_stream_on"))
            lbl_log_stream.setStyleSheet("color: #4caf50; font-size: 10px;")
        else:
            lbl_log_stream.setText("")
            lbl_log_stream.setStyleSheet("color: #556677; font-size: 10px;")

    def _apply_state(state: InstanceState, ip: str, on_done=None):
        """Update UI with state result — must be called in main thread."""
        label = _STATE_LABELS.get(state, state.value)
        _set_status(state, label)
        if state == InstanceState.ERROR:
            # When state is ERROR, 'ip' contains the exception message — log it,
            # but don't show an exception string in the IP address field.
            _last_poll_error[0] = ip
            _log_colored("", mixin._tr("cloud_log_poll_error").format(msg=ip[:120]), "", "#e06060")
            # Keep current IP label unchanged so it doesn't flicker to an error string
        else:
            _last_poll_error[0] = ""
            if ip:
                lbl_ip_value.setText(ip)
                set_cloud_setting("cloud/api_host", ip)  # persist for next launch
                # Notify main window so the status-bar GPU widget polls the new IP.
                _on_ip_live = getattr(mixin, "_on_cloud_ip_changed", None)
                if callable(_on_ip_live):
                    _on_ip_live(f"http://{ip}:8080")
            else:
                lbl_ip_value.clear()
            color = _STATE_COLORS.get(state, "#c8d8e8")
            _log_colored(
                prefix=f"{mixin._tr('cloud_status_label')} ",
                colored_part=label,
                suffix=f"  IP: {ip or '—'}",
                color=color,
            )
        _refresh_lock[0] = False
        # Start/stop remote log streaming based on instance state
        if state == InstanceState.RUNNING:
            if not _log_poll_timer.isActive():
                _log_seq[0] = 0  # reset cursor so we get recent history on reconnect
                _log_poll_timer.start()
                _set_log_streaming(True)
        else:
            if _log_poll_timer.isActive():
                _log_poll_timer.stop()
                _set_log_streaming(False)
        if on_done:
            on_done(state)

    def _refresh_async(on_done=None):
        """Fetch instance state in a background thread; update UI when done."""
        if not _iid():
            return
        if _refresh_lock[0]:
            # Previous request still in-flight — skip this poll tick.
            # Decrement the attempt counter so this skipped tick isn't charged.
            if _poll_attempts[0] > 0:
                _poll_attempts[0] -= 1
            return
        _refresh_lock[0] = True
        iid, reg, prof = _iid(), _reg(), _prof()

        # Use a shared list so the main-thread polling timer can check the result.
        # QTimer.singleShot(0, cb) from a background thread is not reliable on
        # macOS/PySide6 — the callback may never fire, leaving _refresh_lock stuck.
        _state_result: list[tuple | None] = [None]

        def _worker():
            state, ip = get_instance_state(iid, reg, prof)
            _state_result[0] = (state, ip)

        def _check():
            if not _alive[0]:
                return
            if _state_result[0] is None:
                QTimer.singleShot(150, _check)
                return
            state, ip = _state_result[0]
            _apply_state(state, ip, on_done)

        threading.Thread(target=_worker, daemon=True).start()
        QTimer.singleShot(150, _check)

    def _refresh():
        """Manual refresh button handler — async, no blocking."""
        if not _iid():
            lbl_status_value.setText("—")
            lbl_status_value.setStyleSheet("color: #888888;")
            lbl_ip_value.clear()
            _log(mixin._tr("cloud_log_no_iid"))
            return
        _refresh_async()

    def _stop_polling():
        if _poll_timer.isActive():
            _poll_timer.stop()
            _poll_attempts[0] = 0
            _poll_max[0] = _DEFAULT_POLL_MAX
            _log(mixin._tr("cloud_log_poll_stopped"))

    _watchdog_running = [False]  # prevent concurrent install attempts

    def _try_install_watchdog():
        ip = lbl_ip_value.text().strip()
        if not ip:
            return
        if _watchdog_running[0]:
            return  # already in-flight
        attempt = _watchdog_retry[0] + 1
        _log(f"🔧 Watchdog install attempt {attempt}/{_MAX_WATCHDOG_RETRIES}...")
        _watchdog_running[0] = True

        # Snapshot params at call time to avoid race if settings change mid-flight
        _snap_ip   = ip
        _snap_key  = _key()
        _snap_user = _user()
        _snap_idle = spin_idle.value()
        _snap_gpu  = spin_gpu.value()

        _result: list[tuple | None] = [None]

        def _worker():
            _result[0] = install_watchdog_ssh(
                _snap_ip, _snap_key, _snap_user, _snap_idle, _snap_gpu,
            )

        def _check():
            if not _alive[0]:
                _watchdog_running[0] = False
                return
            if _result[0] is None:
                QTimer.singleShot(200, _check)
                return
            _watchdog_running[0] = False
            ok, out = _result[0]
            _log(out[-800:])
            if ok:
                _log(mixin._tr("cloud_log_watchdog_ok"))
                _watchdog_retry[0] = 0
            else:
                _watchdog_retry[0] += 1
                if _watchdog_retry[0] < _MAX_WATCHDOG_RETRIES:
                    _log(f"⏳ SSH not ready, retry in 30s... ({_watchdog_retry[0]}/{_MAX_WATCHDOG_RETRIES})")
                    _watchdog_timer.start(_WATCHDOG_RETRY_MS)
                else:
                    _log(mixin._tr("cloud_log_watchdog_err") + " (max retries exceeded)")
                    _watchdog_retry[0] = 0

        threading.Thread(target=_worker, daemon=True).start()
        QTimer.singleShot(200, _check)

    _watchdog_timer.timeout.connect(_try_install_watchdog)

    def _poll_tick():
        _poll_attempts[0] += 1
        _log(mixin._tr("cloud_log_poll_tick").format(n=_poll_attempts[0])
             + f"  [{_iid()} / {_reg()} / {_prof() or 'default'}]")

        def _on_poll_result(state: InstanceState):
            if state == InstanceState.ERROR:
                _consecutive_errors[0] += 1
                if _consecutive_errors[0] >= _MAX_CONSECUTIVE_ERRORS:
                    _stop_polling()
                    _log_colored(
                        "", 
                        mixin._tr("cloud_log_poll_error_stop").format(
                            msg=_last_poll_error[0][:200]
                        ), 
                        "",
                        "#e06060",
                    )
                return

            # Non-error state — reset error counter
            _consecutive_errors[0] = 0

            if state in _poll_terminal_states[0]:
                _stop_polling()
                _log(mixin._tr("cloud_log_poll_done").format(
                    label=_STATE_LABELS.get(state, state.value)))
                if state == InstanceState.RUNNING and check_enabled.isChecked():
                    ip = lbl_ip_value.text()
                    if ip and ip != mixin._tr("cloud_no_ip"):
                        _log(mixin._tr("cloud_log_watchdog_auto"))
                        _watchdog_retry[0] = 0
                        _log("⏳ Waiting 30s for SSH daemon to become available...")
                        _watchdog_timer.start(_WATCHDOG_RETRY_MS)
            elif _poll_attempts[0] >= _poll_max[0]:
                _stop_polling()
                _log(mixin._tr("cloud_log_poll_timeout"))
                _refresh_async()

        _refresh_async(on_done=_on_poll_result)

    _poll_timer.timeout.connect(_poll_tick)

    _poll_terminal_states: list[set] = [{InstanceState.RUNNING, InstanceState.STOPPED, InstanceState.TERMINATED}]

    def _start_polling(initial_state_label: str, max_attempts: int = _DEFAULT_POLL_MAX,
                       terminal_states: set | None = None):
        lbl_status_value.setText(initial_state_label)
        _poll_timer.stop()
        _poll_attempts[0] = 0
        _poll_max[0] = max_attempts
        _poll_terminal_states[0] = terminal_states if terminal_states is not None else {
            InstanceState.RUNNING, InstanceState.STOPPED, InstanceState.TERMINATED
        }
        _refresh_lock[0] = False  # reset any stale lock
        _consecutive_errors[0] = 0
        _last_poll_error[0] = ""
        _log(mixin._tr("cloud_log_poll_start").format(label=initial_state_label))
        # Immediate first poll so user sees real status without waiting 8s
        _poll_tick()
        _poll_timer.start()

    def _start():
        if not _iid():
            _log(mixin._tr("cloud_log_no_iid"))
            return
        _log(mixin._tr("cloud_log_starting").format(iid=_iid()))
        ok, msg = start_instance(_iid(), _reg(), _prof())
        if ok:
            _log(mixin._tr("cloud_log_start_ok").format(msg=msg))
            _set_status(InstanceState.PENDING, mixin._tr("cloud_state_pending"))
            _start_polling(mixin._tr("cloud_state_pending"))
        elif msg == "INSUFFICIENT_CAPACITY":
            capacity_msg = mixin._tr("cloud_log_start_capacity").format(region=_reg())
            _log(capacity_msg)
            _set_status(InstanceState.ERROR, "❌ InsufficientInstanceCapacity")
        else:
            _log(mixin._tr("cloud_log_start_err").format(msg=msg))
            _set_status(InstanceState.ERROR, f"❌ {msg}")

    def _stop():
        if not _iid():
            _log(mixin._tr("cloud_log_no_iid"))
            return
        _log(mixin._tr("cloud_log_stopping").format(iid=_iid()))
        ok, msg = stop_instance(_iid(), _reg(), _prof())
        if ok:
            _log(mixin._tr("cloud_log_stop_ok").format(msg=msg))
            _set_status(InstanceState.STOPPING, mixin._tr("cloud_state_stopping"))
            _start_polling(
                mixin._tr("cloud_state_stopping"),
                terminal_states={InstanceState.STOPPED, InstanceState.TERMINATED},
            )
        else:
            _log(mixin._tr("cloud_log_stop_err").format(msg=msg))
            _set_status(InstanceState.ERROR, f"❌ {msg}")

    def _launch_new(itype: str, use_spot: bool) -> bool:
        prof = _prof() or "default"
        reg = _reg()
        ami = get_ami_for_profile_region(prof, reg)
        if not ami:
            _log(mixin._tr("cloud_log_ami_missing_launch"))
            QMessageBox.warning(
                widget,
                mixin._tr("cloud_manager_title"),
                mixin._tr("cloud_log_ami_missing_launch"),
            )
            return False
        key_n = get_key_name_for_profile_region(prof, reg)
        sg = get_sg_for_profile_region(prof, reg)
        market_label = "Spot" if use_spot else "On-Demand"
        _log(mixin._tr("cloud_log_launching").format(reg=reg, prof=prof))
        _log(f"Instance type: {format_instance_type_label(itype)} ({market_label})")
        ok, result = launch_instance(
            reg, _prof(),
            instance_type=itype,
            ami_id=ami,
            key_name=key_n,
            security_group_id=sg,
            use_spot=use_spot,
        )
        if ok:
            edit_instance_id.setText(result)
            lbl_ip_value.clear()
            _log(mixin._tr("cloud_log_launched_ok").format(iid=result))
            _schedule_current_price_refresh(itype, use_spot)
            _start_polling(mixin._tr("cloud_state_pending"))
            return True
        else:
            _log(mixin._tr("cloud_log_launched_err").format(msg=result))
            QMessageBox.warning(widget, mixin._tr("cloud_manager_title"), str(result))
            return False

    def _check_env():
        ip = lbl_ip_value.text().strip()
        if not ip:
            _log(mixin._tr("cloud_no_ip_for_check"))
            return
        if _env_task_state["running"]:
            _log("Environment task is already running...")
            return

        local_manifest = build_cloud_worker_bundle_manifest()
        _log(mixin._tr("cloud_log_ssh_check").format(user=_user(), ip=ip, key=_key()))
        _log(f"Local worker bundle revision: {local_manifest['revision']} ({local_manifest['file_count']} files)")

        _env_task_state["running"] = True
        _env_task_state["result"] = None
        _env_task_state["messages"].clear()
        btn_check_env.setEnabled(False)

        def _worker():
            info = check_environment_ssh(ip, _key(), _user(), expected_revision=local_manifest["revision"])
            _env_task_state["result"] = {
                "phase": "check",
                "ip": ip,
                "manifest": local_manifest,
                "info": info,
            }

        def _poll():
            # Guard: stop polling if the widget has been destroyed
            try:
                _ = log_edit.objectName()  # cheap C++ liveness check
            except RuntimeError:
                _env_task_state["running"] = False
                return
            while _env_task_state["messages"]:
                _log(_env_task_state["messages"].pop(0))
            result = _env_task_state["result"]
            if result is None:
                QTimer.singleShot(200, _poll)
                return

            _env_task_state["result"] = None
            if result["phase"] == "check":
                info = result["info"]
                _log(info["output"][:2000])
                _log(mixin._tr("cloud_log_env_stats").format(
                    cuda=info["cuda"], venv=info["venv"], fastapi=info["fastapi"]))
                health = info.get("health") or {}
                if health:
                    _log(
                        f"Worker version: {health.get('worker_version') or 'unknown'} | "
                        f"Remote bundle revision: {health.get('bundle_revision') or info.get('bundle_revision') or 'missing'}"
                    )
                ready = all([
                    info.get("ok"),
                    info.get("cuda"),
                    info.get("venv"),
                    info.get("fastapi"),
                    info.get("deps_ok"),
                    info.get("worker_files"),
                    info.get("worker_running"),
                    info.get("health_ok"),
                    info.get("revision_match"),
                ])
                if ready:
                    _log(mixin._tr("cloud_log_env_ok"))
                    _env_task_state["running"] = False
                    btn_check_env.setEnabled(True)
                    return

                ans = QMessageBox.question(
                    widget,
                    mixin._tr("cloud_env_missing_title"),
                    mixin._tr("cloud_env_missing_msg"),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if ans != QMessageBox.StandardButton.Yes:
                    _env_task_state["running"] = False
                    btn_check_env.setEnabled(True)
                    return

                _log(mixin._tr("cloud_log_installing"))
                def _progress(message: str):
                    _env_task_state["messages"].append(message)

                def _prepare_worker():
                    ok2, out2 = install_environment_ssh(
                        result["ip"],
                        _key(),
                        _user(),
                        progress_callback=_progress,
                    )
                    verify = None
                    if ok2:
                        verify = check_environment_ssh(
                            result["ip"],
                            _key(),
                            _user(),
                            expected_revision=result["manifest"]["revision"],
                        )
                    _env_task_state["result"] = {
                        "phase": "prepare",
                        "ok": ok2,
                        "output": out2,
                        "verify": verify,
                    }

                threading.Thread(target=_prepare_worker, daemon=True).start()
                QTimer.singleShot(200, _poll)
                return

            if result["phase"] == "prepare":
                _log(result["output"][-2000:])
                if result.get("verify"):
                    _log(result["verify"]["output"][:2000])
                _log(mixin._tr("cloud_log_install_ok") if result["ok"] else mixin._tr("cloud_log_install_err"))
                _env_task_state["running"] = False
                btn_check_env.setEnabled(True)
                return

        threading.Thread(target=_worker, daemon=True).start()
        QTimer.singleShot(200, _poll)

    btn_refresh.clicked.connect(_refresh)
    btn_start.clicked.connect(_start)
    btn_stop.clicked.connect(_stop)
    btn_find.clicked.connect(lambda: open_instance_manager(InstanceManagerContext(
        parent=widget,
        tr=mixin._tr,
        reg=_reg,
        prof=_prof,
        iid=_iid,
        region_combo=edit_region,
        log=_log,
        market_label=_market_label,
        set_status=_set_status,
        state_labels=_STATE_LABELS,
        apply_selected=_apply_selected_instance,
        launch_new=_launch_new,
        schedule_price_refresh=_schedule_current_price_refresh,
        poll_timer=_poll_timer,
        edit_instance_id=edit_instance_id,
        lbl_ip_value=lbl_ip_value,
    )))
    btn_check_env.clicked.connect(_check_env)

    _env_task_state = {
        "running": False,
        "result": None,
        "messages": [],
    }

    _price_refresh_timer = QTimer(widget)
    _price_refresh_timer.setSingleShot(True)
    _price_refresh_timer.setInterval(500)
    _price_refresh_timer.timeout.connect(lambda: _schedule_current_price_refresh())
    edit_instance_id.textChanged.connect(lambda _: _price_refresh_timer.start())
    edit_region.currentTextChanged.connect(lambda _: _price_refresh_timer.start())
    edit_aws_profile.textChanged.connect(lambda _: _price_refresh_timer.start())
    _schedule_current_price_refresh()

    def _watchdog_install():
        ip = lbl_ip_value.text().strip()
        if not ip:
            _log(mixin._tr("cloud_no_ip_for_check"))
            return
        idle_min = spin_idle.value()
        gpu_pct  = spin_gpu.value()
        _log(mixin._tr("cloud_log_watchdog_inst").format(idle=idle_min, gpu=gpu_pct))
        save_cloud_settings(
            instance_id=_iid(), region=_reg(),
            ssh_key_path=_key(), ssh_user=_user(),
            aws_profile=edit_aws_profile.text().strip(),
            enabled=check_enabled.isChecked(),
            watchdog_idle_min=idle_min,
            watchdog_gpu_pct=gpu_pct,
            api_host=lbl_ip_value.text().strip(),
        )
        btn_watchdog_install.setEnabled(False)

        _snap_ip, _snap_key, _snap_user = ip, _key(), _user()
        _result: list[tuple | None] = [None]

        def _worker():
            _result[0] = install_watchdog_ssh(_snap_ip, _snap_key, _snap_user, idle_min, gpu_pct)

        def _check():
            if not _alive[0]:
                return
            if _result[0] is None:
                QTimer.singleShot(200, _check)
                return
            btn_watchdog_install.setEnabled(True)
            ok, out = _result[0]
            _log(out[-1500:])
            _log(mixin._tr("cloud_log_watchdog_ok") if ok else mixin._tr("cloud_log_watchdog_err"))

        threading.Thread(target=_worker, daemon=True).start()
        QTimer.singleShot(200, _check)

    def _watchdog_status():
        ip = lbl_ip_value.text().strip()
        if not ip or ip == mixin._tr("cloud_no_ip"):
            _log(mixin._tr("cloud_no_ip_for_check"))
            return
        _log(mixin._tr("cloud_log_watchdog_chk"))
        btn_watchdog_status.setEnabled(False)

        _snap_ip, _snap_key, _snap_user = ip, _key(), _user()
        _result: list[tuple | None] = [None]

        def _worker():
            _result[0] = get_watchdog_status_ssh(_snap_ip, _snap_key, _snap_user)

        def _check():
            if not _alive[0]:
                return
            if _result[0] is None:
                QTimer.singleShot(200, _check)
                return
            btn_watchdog_status.setEnabled(True)
            installed, out = _result[0]
            _log(out[-1500:])
            _log(mixin._tr("cloud_log_watchdog_active") if installed else mixin._tr("cloud_log_watchdog_missing"))

        threading.Thread(target=_worker, daemon=True).start()
        QTimer.singleShot(200, _check)

    btn_watchdog_install.clicked.connect(_watchdog_install)
    btn_watchdog_status.clicked.connect(_watchdog_status)

    # Kick off credentials check on tab open (after a short delay so UI renders)
    QTimer.singleShot(400, _check_and_show_creds_banner)
    # Auto-refresh instance state on open so log streaming starts if already RUNNING
    QTimer.singleShot(600, _refresh_async)

    refs = {
        "enabled": check_enabled,
        "instance_id": edit_instance_id,
        "api_host": lbl_ip_value,
        "region": edit_region,
        "ssh_key": edit_ssh_key,
        "ssh_user": edit_ssh_user,
        "aws_profile": edit_aws_profile,
        "status": lbl_status_value,
        "ip": lbl_ip_value,  # alias kept for backward compat
        "refresh_fn": _refresh,
        "spin_idle": spin_idle,
        "spin_gpu": spin_gpu,
    }
    return widget, refs

