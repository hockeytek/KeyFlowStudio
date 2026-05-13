"""Main entry point for KeyFlow Studio Qt Application"""
import logging
import os
import re
import sys
import ssl
import tempfile
import subprocess
from logging.handlers import RotatingFileHandler


def _ensure_valid_startup_cwd() -> None:
    """Ensure process cwd exists before heavy imports that call os.path.abspath."""
    try:
        os.getcwd()
    except FileNotFoundError:
        fallback_cwd = os.path.dirname(os.path.abspath(__file__))
        os.chdir(fallback_cwd)
        try:
            sys.stderr.write(
                f"[startup] Invalid working directory detected; switched to: {fallback_cwd}\n"
            )
        except Exception:
            pass


_ensure_valid_startup_cwd()

# ── Set environment variables BEFORE any imports ──
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # Enable EXR support for CorridorKey
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizer parallelism warnings

# ── Fix SSL for PyInstaller bundles on macOS ──
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    ssl._create_default_https_context = ssl._create_unverified_context

from collections import OrderedDict
from html import escape
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Add parent directory to path so we can import workspace modules
sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtCore import Qt, QThread, QTimer, Signal, QSignalBlocker, QSize, QLockFile, QStandardPaths, QUrl
from PySide6.QtGui import QDesktopServices, QImage, QPainter, QPixmap, QIcon
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QStatusBar,
    QPushButton,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QInputDialog,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTextEdit,
    QFrame,
)

from app.utils.frame_range_helper import FrameRangeController
from app.utils.write_output import (
    COMPAT_VIDEO_OUTPUT_FORMATS,
    build_video_output_params,
    image_extension_for_format,
    prepare_video_frame,
    resolve_write_output_format,
    save_image_frame,
)


class _QtLogHandler(logging.Handler, QWidget):
    """Logging handler that emits a Qt signal for each log record."""

    class _Emitter(QWidget):
        log_record = Signal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        self._emitter = self._Emitter()
        self.log_record = self._emitter.log_record

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_record.emit(msg)
        except Exception:
            self.handleError(record)


class SecretLogOverlay(QFrame):
    """Transparent overlay with a scrollable log viewer and close button."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("secret_log_overlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setStyleSheet("""
            QFrame#secret_log_overlay {
                background-color: rgba(10, 14, 20, 77);
                border: 1px solid rgba(41, 65, 88, 120);
                border-radius: 10px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        # ── header row ──
        header = QHBoxLayout()
        title = QLabel("Application Log")
        title.setStyleSheet("color: #43c7ff; font-size: 13px; font-weight: 700; background: transparent;")
        header.addWidget(title)
        header.addStretch()
        btn_close = QPushButton("Close")
        btn_close.setObjectName("secret_log_close_btn")
        btn_close.setFixedHeight(30)
        btn_close.setMinimumWidth(60)
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.setStyleSheet(
            "#secret_log_close_btn {"
            "  background-color: rgba(40, 50, 70, 220);"
            "  color: #ddeeff;"
            "  font-size: 13px;"
            "  font-weight: 600;"
            "  border: 1px solid #4a6080;"
            "  border-radius: 6px;"
            "  padding: 4px 14px;"
            "}"
            "#secret_log_close_btn:hover {"
            "  background-color: rgba(180, 50, 50, 230);"
            "  color: #ffffff;"
            "  border: 1px solid #ff6b6b;"
            "}"
        )
        btn_close.clicked.connect(self.hide)
        header.addWidget(btn_close)
        layout.addLayout(header)

        # ── log text area ──
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setStyleSheet("""
            QTextEdit {
                background-color: rgba(11, 15, 21, 90);
                color: #c8d8e8;
                font-family: "SF Mono", "Menlo", "Consolas", monospace;
                font-size: 11px;
                border: 1px solid #1e2a38;
                border-radius: 6px;
                padding: 4px;
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
        layout.addWidget(self._text)

        # ── connect to root logger ──
        self._handler = _QtLogHandler()
        self._handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        self._handler.log_record.connect(self._append_log)
        logging.getLogger().addHandler(self._handler)

        self.hide()

    # ── public ──
    def _append_log(self, text: str) -> None:
        self._text.append(text)
        sb = self._text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def toggle(self) -> None:
        self.setVisible(not self.isVisible())


from app.statusbar_widgets import CloudGpuWidget, ResourceMonitorWidget

try:
    from PySide6.QtMultimedia import QSoundEffect
except Exception:  # pragma: no cover - multimedia backend may be unavailable
    QSoundEffect = None

from UI.main_ui import Ui_MainWindow
from app.constants import (
    PRORES_PROFILES,
)
from app.i18n import t
from app.node_graph_dialog import NodeGraphDialog
from app.embedded_node_graph import EmbeddedNodeGraphEditor
from app.node_graph.nodes.sam_controller import Sam2NodeController
from app.node_graph.nodes.matting_controller import MattingNodeController
from app.coordinators import (
    GraphPresetApplyCoordinator,
    GraphPresetFlowCoordinator,
    GraphPresetSaveCoordinator,
    GraphPresetStoreCoordinator,
    MattingOrchestrator,
    Sam2GraphCoordinator,
    SamInteractionCoordinator,
    ViewerPreviewController,
)
from app.workers import CloudInferenceController, MediaLoadWorker
from app.services.model_service import ModelService
from app.shortcuts import create_save_shortcut
from app.utils import check_ffmpeg, get_ffmpeg_info, install_ffmpeg_info
from app.utils.write_paths import build_keyflow_base_dir
from app.utils.media import resolve_numbered_image_sequence
from app.settings import get_app_settings
from app.cloud_settings import get_cloud_setting
from app.settings_dialog_mixin import SettingsDialogMixin

APP_VERSION = "0.1.0"
logger = logging.getLogger(__name__)


class MainWindow(SettingsDialogMixin, QMainWindow):
    """Runtime window based on the designed Qt .ui file."""

    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self._optional_controls_present = (
            hasattr(self.ui, "combo_input_type")
            and hasattr(self.ui, "combo_param_preset")
            and hasattr(self.ui, "btn_positive_point")
        )
        self.ui.lbl_brand_badge.setText(APP_VERSION)
        self._brand_click_count = 0
        self._brand_click_timer = QTimer(self)
        self._brand_click_timer.setSingleShot(True)
        self._brand_click_timer.setInterval(1500)
        self._brand_click_timer.timeout.connect(self._reset_brand_clicks)
        self.ui.lbl_brand_badge.mousePressEvent = self._on_brand_badge_clicked
        self._app_assets_dir = (Path(__file__).parent / "app" / "assets").resolve()
        self._app_root = Path(__file__).parent
        self._language_code = "ru"
        
        # Создаём statusBar программно, если его нет в UI
        self.statusbar = QStatusBar(self)
        self.statusbar.setStyleSheet("QStatusBar { color: #c0d0e0; font-size: 12px; }")
        self.setStatusBar(self.statusbar)
        self._sleep_guard_indicator = QLabel(self)
        self._sleep_guard_indicator.setObjectName("sleep_guard_indicator")
        self._sleep_guard_indicator.setStyleSheet(
            "QLabel#sleep_guard_indicator { color: #8fa6bf; font-size: 11px; padding-right: 12px; }"
        )
        self.statusbar.addPermanentWidget(self._sleep_guard_indicator)
        self._cloud_gpu_widget = CloudGpuWidget()
        self.statusbar.addPermanentWidget(self._cloud_gpu_widget)
        self._resource_monitor = ResourceMonitorWidget()
        self.statusbar.addPermanentWidget(self._resource_monitor)
        self._set_sleep_guard_indicator(False)
        
        self.ui.hl_viewers.setStretch(0, 1)
        self.ui.hl_viewers.setStretch(1, 1)
        self._base_window_title = self.windowTitle()
        self._output_display_transform = "display_gamma"
        self._viewer_preview = ViewerPreviewController(self)
        # ── All UI styling should come from main.ui only ──
        self._simplify_output_preview_tools()
        self._setup_output_display_transform_buttons()
        self._setup_split_view_button()
        self._relocate_primary_actions_to_topbar()
        self._setup_scalable_viewers()
        self._cleanup_bottom_ui_for_graph_editor()
        self._setup_playback_presets_dropdown()
        self._setup_graph_save_shortcut()
        self._settings = get_app_settings()
        self._node_graph_dialog = None
        self._completion_sound_enabled = bool(self._settings.value("ui/play_completion_sound", True, type=bool))
        self._setup_completion_sound()
        self._graph_builtin_preset_key = "builtin:matanyone2"
        self._graph_builtin_corridorkey_gvm_preset_key = "builtin:corridorkey_gvm"
        self._graph_matanyone2_template_settings_key = "graph_presets/matanyone2_template_json"
        self._graph_empty_preset_key = "__empty__"
        self._graph_save_preset_key = "__save_current__"
        self._graph_delete_preset_key = "__delete_selected__"
        self.graph_preset_store = GraphPresetStoreCoordinator(
            settings=self._settings,
            get_dialog=lambda: getattr(self, "_node_graph_dialog", None),
            graph_matanyone2_template_settings_key=self._graph_matanyone2_template_settings_key,
            graph_builtin_preset_key=self._graph_builtin_preset_key,
            graph_builtin_corridorkey_gvm_preset_key=self._graph_builtin_corridorkey_gvm_preset_key,
        )
        self._migrate_legacy_matanyone2_graph_template()
        self._graph_custom_presets = self._load_graph_custom_presets()
        self._selected_graph_preset_key = self._graph_empty_preset_key
        self._graph_preset_dirty = False
        self._graph_preset_baseline_signature = ""
        self._graph_dirty_timer = QTimer(self)
        self._graph_dirty_timer.setSingleShot(True)
        self._graph_dirty_timer.timeout.connect(self._update_graph_preset_dirty_state)
        self._pending_processing_after_sam2_auto_propagate = False
        self._skip_next_auto_sam2_propagate = False
        self._graph_diagnostics_status_active = False
        self._suspend_sam2_graph_sync = False

        self.sam2 = Sam2NodeController(self._tr, self)
        self.sam2_graph = Sam2GraphCoordinator(
            sam2=self.sam2,
            get_dialog=lambda: getattr(self, "_node_graph_dialog", None),
            get_input_path=lambda: getattr(self, "input_path", None),
            get_frame_index=lambda: int(getattr(self, "current_frame_index", 0)),
            get_fallback_rows=lambda: self._selected_mask_rows() if self._optional_controls_present else [],
        )
        self.sam_interaction = SamInteractionCoordinator(self)
        self.graph_preset_save = GraphPresetSaveCoordinator(
            sam2_graph=self.sam2_graph,
            get_dialog=lambda: getattr(self, "_node_graph_dialog", None),
            get_start_frame=lambda: int(self.ui.spin_start_frame.value()),
            get_end_frame=lambda: int(self.ui.spin_end_frame.value()),
            get_num_frames=lambda: int(self.ui.spin_num_frames.value()),
        )
        self.graph_preset_apply = GraphPresetApplyCoordinator(
            sam2_graph=self.sam2_graph,
            clear_write_outputs=lambda: self._ensure_matting_orchestrator().clear_write_outputs(),
            restore_write_outputs=self._restore_write_outputs_from_disk,
            set_selected_preset_key=lambda key: setattr(self, "_selected_graph_preset_key", key),
            set_baseline_from_current=self._set_graph_preset_baseline_from_current,
            refresh_preset_combo=self._refresh_graph_preset_combo,
            get_start_frame=lambda: int(self.ui.spin_start_frame.value()),
            get_end_frame=lambda: int(self.ui.spin_end_frame.value()),
            set_start_frame=lambda value: self.ui.spin_start_frame.setValue(int(value)),
            set_num_frames=lambda value: self.ui.spin_num_frames.setValue(int(value)),
            set_end_frame=lambda value: self.ui.spin_end_frame.setValue(int(value)),
            block_frame_controls_signals=lambda blocked: (
                self.ui.spin_start_frame.blockSignals(bool(blocked)),
                self.ui.spin_num_frames.blockSignals(bool(blocked)),
                self.ui.spin_end_frame.blockSignals(bool(blocked)),
            ),
            get_total_frames=lambda: len(self.all_frames) if getattr(self, "all_frames", None) else 1,
        )
        self.graph_preset_flow = GraphPresetFlowCoordinator(
            save_preset_key=self._graph_save_preset_key,
            delete_preset_key=self._graph_delete_preset_key,
            empty_preset_key=self._graph_empty_preset_key,
            save_current=self._save_current_graph_preset,
            delete_selected=self._delete_selected_graph_preset,
            refresh_combo=self._refresh_graph_preset_combo,
            set_selected_key=lambda key: setattr(self, "_selected_graph_preset_key", key),
            set_baseline_from_current=self._set_graph_preset_baseline_from_current,
            set_saved_status=self._set_graph_preset_saved_status,
            graph_is_empty=lambda: bool(getattr(self, "_node_graph_dialog", None).graph_is_empty()),
            confirm_replace=self._confirm_graph_preset_replace,
            clear_graph=lambda: getattr(self, "_node_graph_dialog", None).clear_graph(),
            reset_view=lambda: getattr(self, "_node_graph_dialog", None).reset_view(),
            payload_for_key=self._graph_preset_payload,
            apply_preset=lambda preset: bool(getattr(self, "_node_graph_dialog", None).apply_graph_preset(preset)),
            finalize_apply=lambda preset, key: self._ensure_graph_preset_apply_coordinator().finalize_preset_apply(
                preset=preset,
                key=key,
            ),
        )
        self.matting = MattingNodeController(self._tr, self)
        self._cloud_inference = CloudInferenceController(self._tr, self)
        self._matting_orchestrator = MattingOrchestrator(self)
        self._active_node_type: str = ""

        self._input_source_pixmap: QPixmap | None = None
        self._output_source_pixmap: QPixmap | None = None
        self._original_foreground_for_splitter = None
        self._selected_node_preview_path = None
        self._selected_node_preview_is_image = False
        self._selected_node_preview_image = None
        self._selected_node_preview_sequence_paths: list[str] = []
        self._selected_node_video_cap = None
        self._selected_node_video_frame_count = 0
        self._selected_node_frame_cache = OrderedDict()
        self._selected_node_frame_cache_size = 16
        self._selected_node_live_stream_mode = False
        self._output_preview_placeholder_text = self.ui.output_video_label.text()
        self._split_view_enabled = False
        self._split_view_dragging = False
        self._split_x_ratio = 0.5

        self._setup_embedded_node_graph()

        # ── secret log overlay (over the graph) ──
        self._secret_log = SecretLogOverlay(self._embedded_graph_editor)
        self._secret_log.hide()

        # Default: viewer is not interactive until a SAM node is selected
        self.ui.input_video_label.setCursor(Qt.CursorShape.ArrowCursor)
        self._set_viewer_interactive(False)

        self._file_label_style_inactive = "color: #7e91a8; font-size: 12px; font-weight: 500;"
        self._file_label_style_active = "color: #8ce1ff; font-size: 12px; font-weight: 600;"
        self._parameter_presets = {
            "Balanced": (10, 10, 10),
            "Fine Edges": (6, 12, 12),
            "Clean Cut": (12, 8, 8),
            "Stable Start": (10, 10, 18),
            "Eval LR (512p)": (4, 4, 1),
            "Eval HR (1080p)": (15, 15, 10),
        }
        self._parameter_preset_help = {
            "Balanced": "preset_help_balanced",
            "Fine Edges": "preset_help_fine_edges",
            "Clean Cut": "preset_help_clean_cut",
            "Stable Start": "preset_help_stable_start",
            "Eval LR (512p)": "preset_help_eval_lr",
            "Eval HR (1080p)": "preset_help_eval_hr",
            "Custom": "preset_help_custom",
        }
        self._parameter_preset_labels = {
            "Balanced": "preset_balanced",
            "Fine Edges": "preset_fine_edges",
            "Clean Cut": "preset_clean_cut",
            "Stable Start": "preset_stable_start",
            "Eval LR (512p)": "preset_eval_lr",
            "Eval HR (1080p)": "preset_eval_hr",
            "Custom": "preset_custom",
        }
        self._device_help = {
            "auto": "device_help_auto",
            "cpu": "device_help_cpu",
            "mps": "device_help_mps",
            "cuda": "device_help_cuda",
        }
        self._compat_profile_help = {
            "auto": "compat_profile_help_auto",
            "legacy_intel": "compat_profile_help_legacy_intel",
            "apple_silicon": "compat_profile_help_apple_silicon",
        }
        self._preset_sync_in_progress = False
        self._auto_eval_preset_enabled = True
        self._device_selection = str(self._settings.value("runtime/device", "auto")).strip().lower()
        if self._device_selection not in self._device_help:
            self._device_selection = "auto"
        self._compatibility_profile = str(
            self._settings.value("runtime/compatibility_profile", "auto")
        ).strip().lower()
        if self._compatibility_profile not in self._compat_profile_help:
            self._compatibility_profile = "auto"
        self._language_code = str(self._settings.value("ui/language", "en")).strip().lower()
        if self._language_code not in {"ru", "en"}:
            self._language_code = "en"
        self.sam2.set_language(self._language_code)
        self.matting.set_language(self._language_code)
        self._cloud_inference.set_translator(self._tr)
        self._ensure_parameter_preset_options()

        self.input_path = None
        self.is_video_input = True
        self.all_frames = []
        self.video_fps = 30.0
        self.current_frame_index = 0
        self.current_frame = None
        self._media_loader_thread: QThread | None = None
        self._media_loader_worker: MediaLoadWorker | None = None
        self._media_loading_active = False
        self._media_loading_target_is_video = True
        self._media_loading_request_node_type = ""

        self.last_output_dir = None

        self.play_timer = QTimer(self)
        self.play_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.play_timer.timeout.connect(self._play_next_frame)
        self._play_direction = 1
        self._play_loop_enabled = False
        self._play_started_monotonic = 0.0
        self._play_start_index = 0
        self._apply_playback_transport_icons()

        self._apply_language(announce=False)
        self._setup_sam2_signals()
        self._setup_matting_signals()
        self._setup_cloud_inference_signals()
        self._wire_events()
        self._check_dependencies()
        self._apply_device_selection()
        self._set_file_label_state(False)
        self._configure_tooltips()
        self._sync_preset_selection()
        self._update_parameter_preset_tooltip()
        self._set_status(self._tr("status_ready"))

        # ── Auto-start cloud GPU widget if api_host already saved ──
        _saved_host = str(get_cloud_setting("cloud/api_host") or "").strip()
        if _saved_host:
            self._cloud_gpu_widget.start(f"http://{_saved_host}:8080")

        # ── Set window size to display graph editor and properties properly ──
        self.setMinimumSize(QSize(1400, 1000))
        self.resize(1600, 1200)

    def _ui_widget(self, name: str):
        """Get optional UI attribute by name."""
        return getattr(self.ui, name, None)

    def _ensure_viewer_preview(self) -> ViewerPreviewController:
        controller = getattr(self, "_viewer_preview", None)
        if controller is None:
            controller = ViewerPreviewController(self)
            self._viewer_preview = controller
        return controller

    def _spin_value(self, name: str, default: int) -> int:
        """Safe spinbox value accessor for optional controls that may be absent in UI."""
        widget = self._ui_widget(name)
        if isinstance(widget, QSpinBox):
            return int(widget.value())
        return int(default)

    def _tr(self, key: str) -> str:
        return t(key, getattr(self, "_language_code", "ru"))

    def _preset_display_name(self, preset_key: str) -> str:
        return self._tr(self._parameter_preset_labels.get(preset_key, "preset_custom"))

    def _current_preset_key(self) -> str:
        preset_key = self.ui.combo_param_preset.currentData()
        if isinstance(preset_key, str) and preset_key:
            return preset_key

        current_text = self.ui.combo_param_preset.currentText().strip()
        for key in [*self._parameter_presets.keys(), "Custom"]:
            if current_text in {key, self._preset_display_name(key)}:
                return key
        return "Balanced"

    def _set_current_preset_key(self, preset_key: str) -> None:
        combo = self.ui.combo_param_preset
        index = combo.findData(preset_key)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _set_input_type_combo(self) -> None:
        if not self._optional_controls_present:
            return
        current_value = "video" if self.is_video_input else "image"
        combo = self.ui.combo_input_type
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem(self._tr("combo_video"), "video")
            combo.addItem(self._tr("combo_image"), "image")
            index = combo.findData(current_value)
            combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            combo.blockSignals(False)

    def _apply_language(self, announce: bool = False) -> None:
        self._set_input_type_combo()
        self._ensure_parameter_preset_options()

        self.ui.lbl_input_title.setText(self._tr("lbl_input_title"))
        self.ui.lbl_output_title.setText(self._tr("lbl_output_title"))
        self.ui.lbl_start_frame.setText(self._tr("lbl_start_frame"))
        self.ui.lbl_num_frames.setText(self._tr("lbl_frames"))
        self.ui.lbl_end_frame.setText(self._tr("lbl_end_frame"))
        self._update_run_button_label()
        # self.ui.btn_run.setText(...)  — handled by _update_run_button_label()
        self.ui.btn_stop.setText(self._tr("btn_stop"))
        self.ui.btn_save_result.setText(self._tr("btn_save_result"))
        self.ui.spin_num_frames.setSpecialValueText(self._tr("btn_all_frames"))
        self.ui.spin_end_frame.setSpecialValueText(self._tr("spin_end_frame_auto"))

        if self._optional_controls_present:
            self.ui.lbl_media_format.setText(self._tr("media_format_label"))
            self.ui.btn_load.setText(self._tr("btn_load_text"))
            self.ui.lbl_sec_model_params.setText(self._tr("lbl_model_options"))
            self.ui.lbl_param_preset.setText(self._tr("lbl_preset"))
            self.ui.lbl_erode_kernel.setText(self._tr("lbl_erode"))
            self.ui.lbl_dilate_kernel.setText(self._tr("lbl_dilate"))
            self.ui.lbl_warmup_frames.setText(self._tr("lbl_warmup"))
            self.ui.lbl_sec_points.setText(self._tr("lbl_sam_prompts"))
            self.ui.btn_clear_points.setText(self._tr("btn_clear_points"))
            self.ui.btn_live_sam2.setText(self._tr("btn_live_sam2"))
            self.ui.lbl_sec_masks.setText(self._tr("lbl_masks"))
            self.ui.btn_generate_mask.setText(self._tr("btn_generate_mask"))
            self.ui.btn_add_mask.setText(self._tr("btn_add_mask"))
            self.ui.btn_remove_mask.setText(self._tr("btn_remove_mask"))
            self.ui.btn_load_mask.setText(self._tr("btn_load_mask"))
            self.ui.lbl_sec_actions.setText(self._tr("lbl_actions"))
        if hasattr(self, "_combo_playback_presets"):
            self._refresh_graph_preset_combo()
        if hasattr(self, "_node_graph_dialog") and self._node_graph_dialog is not None:
            self._node_graph_dialog.set_translator(self._tr)
        if hasattr(self, "_embedded_graph_editor"):
            self._embedded_graph_editor.set_translator(self._tr)

        if not self.input_path:
            self.ui.input_video_label.setText(self._tr("input_placeholder"))
            if self._optional_controls_present:
                self.ui.lbl_status.setText(self._tr("lbl_sam_status_default"))

        if not self._has_selected_node_preview():
            self.ui.output_video_label.setText(self._tr("output_placeholder"))

        self._output_preview_placeholder_text = self._tr("output_placeholder")
        self._configure_tooltips()
        self._set_file_label_state(bool(self.input_path), Path(self.input_path).name if self.input_path else "")

        if announce:
            language_label = self._tr("lang_russian") if self._language_code == "ru" else self._tr("lang_english")
            self._set_status(f"{self._tr('status_language_set')} {language_label}")

        self._set_sleep_guard_indicator(False)

    def _set_sleep_guard_indicator(self, active: bool) -> None:
        if not hasattr(self, "_sleep_guard_indicator"):
            return
        if active:
            self._sleep_guard_indicator.setText(self._tr("status_sleep_guard_on"))
            self._sleep_guard_indicator.setStyleSheet(
                "QLabel#sleep_guard_indicator { color: #86e6a8; font-size: 11px; font-weight: 600; padding-right: 12px; }"
            )
        else:
            self._sleep_guard_indicator.setText(self._tr("status_sleep_guard_off"))
            self._sleep_guard_indicator.setStyleSheet(
                "QLabel#sleep_guard_indicator { color: #8fa6bf; font-size: 11px; padding-right: 12px; }"
            )

    def _setup_sam2_signals(self):
        self.sam2.status_changed.connect(self.sam_interaction.on_sam2_status_changed)
        self.sam2.input_preview_needed.connect(self._render_input_preview)
        self.sam2.mask_preview_available.connect(self.sam_interaction.show_mask_preview_on_output)
        self.sam2.progress_updated.connect(self.sam_interaction.on_sam2_progress)
        self.sam2.generation_started.connect(lambda: self.sam_interaction.set_sam_controls_busy(True))
        self.sam2.generation_started.connect(self._refresh_stop_button_state)
        self.sam2.controls_busy_changed.connect(self.sam_interaction.set_sam_controls_busy)
        self.sam2.mask_list_changed.connect(self.sam_interaction.refresh_mask_list)
        self.sam2.generation_finished.connect(self.sam_interaction.on_sam2_generation_finished)
        self.sam2.generation_finished.connect(self._refresh_stop_button_state)
        self.sam2.node_frame_progress.connect(self._on_node_frame_progress)
        self.sam2.error_occurred.connect(self.sam_interaction.on_sam2_error)

    def _setup_matting_signals(self):
        self.matting.stage_progress.connect(self._on_matting_stage_progress)
        self.matting.node_frame_progress.connect(self._on_node_frame_progress)
        self.matting.frame_progress.connect(self._on_matting_frame_progress)
        self.matting.frame_preview.connect(self._on_matting_frame_preview)
        self.matting.graph_stream_preview.connect(self._on_graph_stream_preview)
        self.matting.log_message.connect(self._on_matting_log_message)
        self.matting.corridorkey_mode_resolved.connect(self._on_corridorkey_mode_resolved)
        self.matting.processing_finished.connect(self._on_matting_finished)
        self.matting.error_occurred.connect(self._on_matting_error)
        self.matting.controls_busy_changed.connect(self._on_matting_busy_changed)

    def _setup_cloud_inference_signals(self):
        self._cloud_inference.stage_progress.connect(self._on_cloud_stage_progress)
        self._cloud_inference.log_message.connect(self._on_cloud_log_message)
        self._cloud_inference.processing_finished.connect(self._on_cloud_processing_finished)
        self._cloud_inference.error_occurred.connect(self._on_cloud_processing_error)
        self._cloud_inference.controls_busy_changed.connect(self._on_cloud_busy_changed)
        self._cloud_inference.remote_health.connect(self._cloud_gpu_widget.update_health)
        self._cloud_inference.worker_connected.connect(self._cloud_gpu_widget.start)

    def _on_cloud_ip_changed(self, base_url: str) -> None:
        """Called by cloud settings tab when instance IP is updated (start/poll result).
        Restarts GPU widget polling on the new URL immediately so the status bar
        reflects the live instance even between processing jobs."""
        self._cloud_gpu_widget.start(base_url)
        if base_url and self._node_graph_dialog is not None:
            self._node_graph_dialog.refresh_cloud_weights_status(base_url)

    def _wire_events(self):
        if self._optional_controls_present:
            self.ui.btn_load.clicked.connect(self.on_load_media)
            self.ui.combo_input_type.currentIndexChanged.connect(self.on_input_type_changed)
        self.ui.btn_settings.clicked.connect(self.open_settings_dialog)
        if self._optional_controls_present:
            self.ui.combo_param_preset.currentIndexChanged.connect(self.on_parameter_preset_changed)

            self.ui.btn_positive_point.clicked.connect(lambda: self._set_point_mode(True))
            self.ui.btn_negative_point.clicked.connect(lambda: self._set_point_mode(False))
            self.ui.btn_clear_points.clicked.connect(self.on_clear_points)
            self.ui.btn_live_sam2.toggled.connect(self._on_live_sam2_toggled)

            self.ui.btn_generate_mask.clicked.connect(self.on_generate_mask)
            self.ui.btn_add_mask.clicked.connect(self.on_add_mask)
            self.ui.btn_remove_mask.clicked.connect(self.on_remove_masks)
            self.ui.btn_load_mask.clicked.connect(self.on_load_mask_file)

        self.ui.btn_run.clicked.connect(self.start_processing)
        self.ui.btn_stop.clicked.connect(self.cancel_processing)
        self.ui.btn_save_result.clicked.connect(self.open_output_folder)

        frame_slider = self._transport_slider()
        if frame_slider is not None:
            frame_slider.valueChanged.connect(self.on_frame_slider_changed)

        transport_connections = (
            ("btn_first_frame", "clicked", self.on_first_frame),
            ("btn_prev_frame", "clicked", self.on_prev_frame),
            ("btn_play_reverse", "toggled", self.on_play_reverse_toggled),
            ("btn_play", "toggled", self.on_play_toggled),
            ("btn_next_frame", "clicked", self.on_next_frame),
            ("btn_last_frame", "clicked", self.on_last_frame),
            ("btn_play_loop", "toggled", self.on_play_loop_toggled),
        )
        for name, signal_name, handler in transport_connections:
            button = self._transport_button(name)
            if button is not None:
                getattr(button, signal_name).connect(handler)
        if self._optional_controls_present:
            self.ui.masks_list.itemSelectionChanged.connect(self._render_input_preview)
            self.ui.spin_erode_kernel.valueChanged.connect(self._sync_preset_selection)
            self.ui.spin_dilate_kernel.valueChanged.connect(self._sync_preset_selection)
            self.ui.spin_warmup_frames.valueChanged.connect(self._sync_preset_selection)

        # Connect frame range controls for auto-calculation
        self.ui.spin_start_frame.valueChanged.connect(self._on_start_frame_changed)
        self.ui.spin_end_frame.valueChanged.connect(self._on_end_frame_changed)
        self.ui.spin_num_frames.valueChanged.connect(self._on_frame_count_changed)

        self.ui.input_video_label.mousePressEvent = self._on_input_label_mouse_press
        # output_video_label mouse events for split-view drag are wired
        # inside ViewerPreviewController._setup_split_view_button().

        # Sync dependent controls on startup (in case Live SAM is pre-enabled in .ui)
        if self._optional_controls_present:
            self._on_live_sam2_toggled(self.ui.btn_live_sam2.isChecked())

    def _check_dependencies(self):
        if not check_ffmpeg():
            QMessageBox.warning(self, self._tr("ffmpeg_not_found_title"), install_ffmpeg_info())
        else:
            info = get_ffmpeg_info()
            if info:
                self._set_status(info)

    def _set_status(self, text: str):
        self.statusbar.showMessage(text, 0)

    def _on_sam2_status_changed(self, text: str):
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            self.sam_interaction.on_sam2_status_changed(text)
            return
        if hasattr(self, "sam2") and getattr(self.sam2, "state", None) is not None:
            try:
                self.sam2.state.set_status(text)
            except Exception:
                pass
        if (
            hasattr(self, "_node_graph_dialog")
            and self._node_graph_dialog is not None
            and not bool(getattr(self, "_suspend_sam2_graph_sync", False))
        ):
            self.sam2_graph.sync_to_graph(text)
        if self._optional_controls_present and hasattr(self.ui, "lbl_status") and self.ui.lbl_status.isVisible():
            self.ui.lbl_status.setText(text)
        self._set_status(text)

    def _apply_frame_range_from_preset(self, preset: dict | None, *, total_frames: int | None = None) -> None:
        self._ensure_graph_preset_apply_coordinator().apply_frame_range_from_preset(
            preset,
            total_frames=total_frames,
        )

    def _on_sam2_progress(self, percent: int, status_text: str):
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            self.sam_interaction.on_sam2_progress(percent, status_text)
            return
        self.ui.progress_bar.setRange(0, 100)
        self.ui.progress_bar.setValue(max(0, min(100, int(percent))))
        if status_text:
            self._set_status(status_text)

    def _on_graph_diagnostics_changed(self, summary: str, _details: str, has_errors: bool) -> None:
        was_active = bool(getattr(self, "_graph_diagnostics_status_active", False))
        self._graph_diagnostics_status_active = bool(has_errors)
        if getattr(self.matting, "is_active", False):
            return
        if has_errors:
            self._set_status(summary)
            return
        if was_active:
            self._set_status(self._tr("status_ready"))

    def _on_sam2_generation_finished(self):
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            self.sam_interaction.on_sam2_generation_finished()
            return
        self._ensure_sam2_auto_propagation_state()
        self.ui.progress_bar.setRange(0, 100)
        self.ui.progress_bar.setValue(100)
        final_status = str(getattr(getattr(self.sam2, "state", None), "status_text", "") or "").strip()
        self.sam2_graph.sync_to_graph(final_status or self._tr("sam_mask_ready"))
        exported_count, exported_frames = self._save_sam2_outputs_to_connected_write_nodes()
        if exported_count > 0:
            self._set_status(
                self._tr("sam_write_immediate_export_done").format(count=exported_count, frames=exported_frames)
            )

        if self._active_node_type in {"sam2"}:
            self._show_mask_preview_on_output(self.sam2.state.mask_for_frame(self.current_frame_index))

        if self._pending_processing_after_sam2_auto_propagate and not self.matting.is_active:
            self._pending_processing_after_sam2_auto_propagate = False
            self._skip_next_auto_sam2_propagate = True
            QTimer.singleShot(0, self.start_processing)

    def _on_sam2_error(self, message: str, show_dialog: bool):
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            self.sam_interaction.on_sam2_error(message, show_dialog)
            return
        self.ui.progress_bar.setRange(0, 100)
        self.ui.progress_bar.setValue(0)
        if show_dialog:
            QMessageBox.warning(self, self._tr("sam_error_title"), message)

    @staticmethod
    def _normalize_device_label(device: object | None) -> str:
        if device is None:
            return "AUTO"
        return str(device).split(":", 1)[0].upper()

    def _get_requested_device_label(self) -> str:
        return self._device_selection.upper() or "AUTO"

    def _update_run_button_label(self) -> None:
        """Switch btn_run label between local and cloud modes."""
        cloud_enabled = bool(get_cloud_setting("cloud/enabled"))
        if cloud_enabled:
            self.ui.btn_run.setText(self._tr("btn_run_cloud"))
            region = str(get_cloud_setting("cloud/region") or "eu-west-1")
            self.ui.btn_run.setToolTip(f"☁ {region}")
        else:
            self.ui.btn_run.setText(self._tr("btn_run"))
            self.ui.btn_run.setToolTip("")

    def _get_device_display_label(self) -> str:
        cloud_enabled = bool(get_cloud_setting("cloud/enabled"))
        if cloud_enabled:
            region = str(get_cloud_setting("cloud/region") or "eu-west-1")
            return f"{self._tr('status_cloud_gpu')} · {region}"
        requested_label = self._get_requested_device_label()
        actual_label = self._normalize_device_label(ModelService().get_device())
        if requested_label == "AUTO":
            return f"AUTO -> {actual_label}"
        if requested_label == actual_label:
            return actual_label
        return f"{requested_label} -> {actual_label}"

    def _update_window_title_with_device(self) -> None:
        self.setWindowTitle(f"{self._base_window_title} [{self._get_device_display_label()}]")

    def _transport_button(self, name: str) -> QPushButton | None:
        button = getattr(self.ui, name, None)
        return button if isinstance(button, QPushButton) else None

    def _transport_slider(self):
        return getattr(self.ui, "frame_slider", None)

    def _set_transport_controls_enabled(self, enabled: bool) -> None:
        for name in (
            "btn_first_frame",
            "btn_prev_frame",
            "btn_play_reverse",
            "btn_play",
            "btn_next_frame",
            "btn_last_frame",
            "btn_play_loop",
        ):
            button = self._transport_button(name)
            if button is not None:
                button.setEnabled(enabled)

        slider = self._transport_slider()
        if slider is not None:
            slider.setEnabled(enabled)

    def _apply_playback_transport_icons(self) -> None:
        a = self._app_assets_dir
        first_icon = a / "transport-first.svg"
        prev_icon = a / "transport-prev.svg"
        play_icon = a / "transport-play.svg"
        pause_icon = a / "transport-pause.svg"
        next_icon = a / "transport-next.svg"
        last_icon = a / "transport-last.svg"
        reverse_icon = a / "play-reverse.svg"
        loop_icon = a / "play-loop.svg"

        self._icon_play = QIcon(play_icon.as_posix())
        self._icon_pause = QIcon(pause_icon.as_posix())

        button_specs = (
            ("btn_first_frame", QIcon(first_icon.as_posix())),
            ("btn_prev_frame", QIcon(prev_icon.as_posix())),
            ("btn_play_reverse", QIcon(reverse_icon.as_posix())),
            ("btn_play", self._icon_play),
            ("btn_next_frame", QIcon(next_icon.as_posix())),
            ("btn_last_frame", QIcon(last_icon.as_posix())),
            ("btn_play_loop", QIcon(loop_icon.as_posix())),
        )
        for name, icon in button_specs:
            button = self._transport_button(name)
            if button is None:
                continue
            button.setText("")
            button.setIcon(icon)

    def _simplify_output_preview_tools(self) -> None:
        self._ensure_viewer_preview()._simplify_output_preview_tools()

    def _setup_output_display_transform_buttons(self) -> None:
        self._ensure_viewer_preview()._setup_output_display_transform_buttons()

    def _set_output_display_transform_controls_enabled(self, enabled: bool) -> None:
        self._ensure_viewer_preview()._set_output_display_transform_controls_enabled(enabled)

    def _sync_output_display_transform_buttons(self) -> None:
        self._ensure_viewer_preview()._sync_output_display_transform_buttons()

    def _set_output_display_transform(self, mode: str) -> None:
        self._ensure_viewer_preview()._set_output_display_transform(mode)

    def _setup_split_view_button(self) -> None:
        self._ensure_viewer_preview()._setup_split_view_button()

    def _on_split_view_toggled(self, checked: bool) -> None:
        self._ensure_viewer_preview()._on_split_view_toggled(checked)

    def _on_output_label_mouse_press(self, event) -> None:
        self._ensure_viewer_preview()._on_output_label_mouse_press(event)

    def _on_output_label_mouse_move(self, event) -> None:
        self._ensure_viewer_preview()._on_output_label_mouse_move(event)

    def _on_output_label_mouse_release(self, event) -> None:
        self._ensure_viewer_preview()._on_output_label_mouse_release(event)

    def _update_split_from_mouse(self, click_x: float) -> None:
        self._ensure_viewer_preview()._update_split_from_mouse(click_x)

    def _relocate_primary_actions_to_topbar(self) -> None:
        if self._optional_controls_present:
            self.ui.lbl_media_format.hide()
            self.ui.combo_input_type.hide()
            self.ui.btn_load.hide()
            self.ui.lbl_sec_actions.hide()
        self.ui.vl_sidebar.setContentsMargins(0, 0, 0, 0)
        self.ui.vl_sidebar.setSpacing(0)
        self.ui.sp_sidebar_bottom.changeSize(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        for button in (self.ui.btn_run, self.ui.btn_stop, self.ui.btn_save_result):
            button.setCursor(Qt.CursorShape.PointingHandCursor)

        self.ui.vl_main.invalidate()

    def _setup_scalable_viewers(self) -> None:
        """Allow preview panels to scale proportionally (16:9) with the window."""
        self.ui.input_video_label.setMaximumSize(QSize(16777215, 16777215))
        self.ui.output_video_label.setMaximumSize(QSize(16777215, 16777215))
        sp = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ui.input_video_label.setSizePolicy(sp)
        self.ui.output_video_label.setSizePolicy(sp)
        # hl_viewers is at index 2 in vl_main — give it stretch so it grows when window expands
        self.ui.vl_main.setStretch(2, 1)
        QTimer.singleShot(0, self._update_viewer_aspect)

    def _setup_playback_presets_dropdown(self) -> None:
        """Configure the graph preset combo defined in the UI file."""
        self._combo_playback_presets = getattr(self.ui, "combo_playback_presets", None)
        if self._combo_playback_presets is None:
            return
        self._combo_playback_presets.setMinimumWidth(272)
        view = self._combo_playback_presets.view()
        if view is not None:
            view.setMinimumWidth(366)
        self._combo_playback_presets.clear()
        self._combo_playback_presets.activated.connect(self._on_graph_preset_selected)

    def _setup_graph_save_shortcut(self) -> None:
        self._save_graph_shortcut = create_save_shortcut(self, self._on_graph_save_shortcut)

    def _set_graph_preset_saved_status(self, key: str | None) -> None:
        key_str = str(key or "")
        if key_str.startswith("custom:"):
            preset_name = key_str.split(":", 1)[1].strip()
            if preset_name:
                self._set_status(self._tr("graph_preset_saved_named").format(name=preset_name))
                return
        self._set_status(self._tr("graph_preset_saved"))

    def _ensure_graph_preset_save_coordinator(self) -> GraphPresetSaveCoordinator:
        coordinator = getattr(self, "graph_preset_save", None)
        if coordinator is None:
            coordinator = GraphPresetSaveCoordinator(
                sam2_graph=self.sam2_graph,
                get_dialog=lambda: getattr(self, "_node_graph_dialog", None),
                get_start_frame=lambda: int(self.ui.spin_start_frame.value()),
                get_end_frame=lambda: int(self.ui.spin_end_frame.value()),
                get_num_frames=lambda: int(self.ui.spin_num_frames.value()),
            )
            self.graph_preset_save = coordinator
        return coordinator

    def _ensure_graph_preset_apply_coordinator(self) -> GraphPresetApplyCoordinator:
        coordinator = getattr(self, "graph_preset_apply", None)
        if coordinator is None:
            coordinator = GraphPresetApplyCoordinator(
                sam2_graph=self.sam2_graph,
                clear_write_outputs=lambda: self._ensure_matting_orchestrator().clear_write_outputs(),
                restore_write_outputs=self._restore_write_outputs_from_disk,
                set_selected_preset_key=lambda key: setattr(self, "_selected_graph_preset_key", key),
                set_baseline_from_current=self._set_graph_preset_baseline_from_current,
                refresh_preset_combo=self._refresh_graph_preset_combo,
                get_start_frame=lambda: int(self.ui.spin_start_frame.value()),
                get_end_frame=lambda: int(self.ui.spin_end_frame.value()),
                set_start_frame=lambda value: self.ui.spin_start_frame.setValue(int(value)),
                set_num_frames=lambda value: self.ui.spin_num_frames.setValue(int(value)),
                set_end_frame=lambda value: self.ui.spin_end_frame.setValue(int(value)),
                block_frame_controls_signals=lambda blocked: (
                    self.ui.spin_start_frame.blockSignals(bool(blocked)),
                    self.ui.spin_num_frames.blockSignals(bool(blocked)),
                    self.ui.spin_end_frame.blockSignals(bool(blocked)),
                ),
                get_total_frames=lambda: len(self.all_frames) if getattr(self, "all_frames", None) else 1,
            )
            self.graph_preset_apply = coordinator
        return coordinator

    def _ensure_graph_preset_flow_coordinator(self) -> GraphPresetFlowCoordinator:
        coordinator = getattr(self, "graph_preset_flow", None)
        if coordinator is None:
            coordinator = GraphPresetFlowCoordinator(
                save_preset_key=self._graph_save_preset_key,
                delete_preset_key=self._graph_delete_preset_key,
                empty_preset_key=self._graph_empty_preset_key,
                save_current=self._save_current_graph_preset,
                delete_selected=self._delete_selected_graph_preset,
                refresh_combo=self._refresh_graph_preset_combo,
                set_selected_key=lambda key: setattr(self, "_selected_graph_preset_key", key),
                set_baseline_from_current=self._set_graph_preset_baseline_from_current,
                set_saved_status=self._set_graph_preset_saved_status,
                graph_is_empty=lambda: bool(getattr(self, "_node_graph_dialog", None).graph_is_empty()),
                confirm_replace=self._confirm_graph_preset_replace,
                clear_graph=lambda: getattr(self, "_node_graph_dialog", None).clear_graph(),
                reset_view=lambda: getattr(self, "_node_graph_dialog", None).reset_view(),
                payload_for_key=self._graph_preset_payload,
                apply_preset=lambda preset: bool(getattr(self, "_node_graph_dialog", None).apply_graph_preset(preset)),
                finalize_apply=lambda preset, key: self._ensure_graph_preset_apply_coordinator().finalize_preset_apply(
                    preset=preset,
                    key=key,
                ),
            )
            self.graph_preset_flow = coordinator
        return coordinator

    def _ensure_graph_preset_store_coordinator(self) -> GraphPresetStoreCoordinator:
        coordinator = getattr(self, "graph_preset_store", None)
        if coordinator is None:
            coordinator = GraphPresetStoreCoordinator(
                settings=self._settings,
                get_dialog=lambda: getattr(self, "_node_graph_dialog", None),
                graph_matanyone2_template_settings_key=self._graph_matanyone2_template_settings_key,
                graph_builtin_preset_key=self._graph_builtin_preset_key,
                graph_builtin_corridorkey_gvm_preset_key=self._graph_builtin_corridorkey_gvm_preset_key,
            )
            self.graph_preset_store = coordinator
        return coordinator

    def _confirm_graph_preset_replace(self) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(self._tr("graph_preset_replace_title"))
        box.setText(self._tr("graph_preset_replace_message"))
        btn_yes = box.addButton(self._tr("dlg_yes"), QMessageBox.ButtonRole.YesRole)
        box.addButton(self._tr("dlg_no"), QMessageBox.ButtonRole.NoRole)
        box.exec()
        return box.clickedButton() == btn_yes

    def _on_graph_save_shortcut(self) -> None:
        saved_key = self._quick_save_current_graph_preset()
        if not saved_key:
            return

        self._selected_graph_preset_key = saved_key
        self._set_graph_preset_baseline_from_current()
        self._refresh_graph_preset_combo(saved_key)
        self._set_graph_preset_saved_status(saved_key)

    def _quick_save_current_graph_preset(self) -> str | None:
        dialog = getattr(self, "_node_graph_dialog", None)
        if dialog is None:
            return None

        current_key = self._selected_graph_preset_key
        if current_key.startswith("custom:"):
            preset_name = current_key.split(":", 1)[1]
            if preset_name:
                preset = self._ensure_graph_preset_save_coordinator().build_current_preset()
                if preset is None:
                    return None
                self._graph_custom_presets[preset_name] = preset
                self._save_graph_custom_presets()
                return current_key

        return self._save_current_graph_preset()

    def _load_graph_custom_presets(self) -> dict[str, dict]:
        return self._ensure_graph_preset_store_coordinator().load_custom_presets()

    def _load_graph_template_preset(self, settings_key: str) -> dict | None:
        return self._ensure_graph_preset_store_coordinator().load_template_preset(settings_key)

    def _save_graph_template_preset(self, settings_key: str, preset: dict) -> None:
        self._ensure_graph_preset_store_coordinator().save_template_preset(settings_key, preset)

    def _migrate_legacy_matanyone2_graph_template(self) -> None:
        self._ensure_graph_preset_store_coordinator().migrate_legacy_matanyone2_graph_template()

    def _save_graph_custom_presets(self) -> None:
        self._ensure_graph_preset_store_coordinator().save_custom_presets(self._graph_custom_presets)

    def _graph_builtin_presets(self) -> dict[str, dict]:
        return self._ensure_graph_preset_store_coordinator().graph_builtin_presets()

    def _refresh_graph_preset_combo(self, selected_key: str | None = None) -> None:
        combo = getattr(self, "_combo_playback_presets", None)
        if combo is None:
            return

        target_key = selected_key if selected_key is not None else self._selected_graph_preset_key
        show_delete_action = isinstance(target_key, str) and target_key.startswith("custom:")
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem(self._tr("graph_preset_empty"), self._graph_empty_preset_key)
            combo.addItem(self._tr("graph_preset_matanyone2"), self._graph_builtin_preset_key)
            combo.addItem(self._tr("graph_preset_corridorkey_gvm"), self._graph_builtin_corridorkey_gvm_preset_key)
            for name in sorted(self._graph_custom_presets.keys(), key=str.casefold):
                combo.addItem(name, f"custom:{name}")
            combo.addItem(self._tr("graph_preset_save_current"), self._graph_save_preset_key)
            if show_delete_action:
                combo.addItem(self._tr("graph_preset_delete_selected"), self._graph_delete_preset_key)

            index = combo.findData(target_key)
            combo.setCurrentIndex(index if index >= 0 else 0)

            if index >= 0 and self._graph_preset_dirty and target_key not in {
                self._graph_save_preset_key,
                self._graph_delete_preset_key,
            }:
                current_text = combo.itemText(index)
                combo.setItemText(index, f"{current_text} {self._tr('graph_preset_modified_suffix')}")
        finally:
            combo.blockSignals(False)

    def _graph_signature_from_preset(self, preset: dict | None) -> str:
        return self._ensure_graph_preset_store_coordinator().graph_signature_from_preset(preset)

    def _current_graph_signature(self) -> str:
        return self._ensure_graph_preset_store_coordinator().current_graph_signature()

    def _set_graph_preset_baseline_from_current(self) -> None:
        self._graph_preset_baseline_signature = self._current_graph_signature()
        self._graph_preset_dirty = False

    def _schedule_graph_dirty_state_update(self) -> None:
        if getattr(self, "_graph_dirty_timer", None) is None:
            return
        self._graph_dirty_timer.start(150)

    def _update_graph_preset_dirty_state(self) -> None:
        current_signature = self._current_graph_signature()
        is_dirty = bool(self._graph_preset_baseline_signature) and current_signature != self._graph_preset_baseline_signature
        if self._graph_preset_dirty == is_dirty:
            return
        self._graph_preset_dirty = is_dirty
        self._refresh_graph_preset_combo()

    def _graph_preset_payload(self, key: str) -> dict | None:
        return self._ensure_graph_preset_store_coordinator().graph_preset_payload(key, self._graph_custom_presets)

    def _on_graph_preset_selected(self, index: int) -> None:
        combo = getattr(self, "_combo_playback_presets", None)
        dialog = getattr(self, "_node_graph_dialog", None)
        if combo is None or dialog is None:
            return

        key = str(combo.itemData(index) or "")
        previous_key = self._selected_graph_preset_key
        self._ensure_graph_preset_flow_coordinator().handle_selection(
            key=key,
            previous_key=previous_key,
        )

    def _save_current_graph_preset(self) -> str | None:
        dialog = getattr(self, "_node_graph_dialog", None)
        if dialog is None:
            return None
        if dialog.graph_is_empty():
            QMessageBox.information(self, self._tr("graph_preset_name_title"), self._tr("graph_preset_save_empty"))
            return None

        name, accepted = QInputDialog.getText(
            self,
            self._tr("graph_preset_name_title"),
            self._tr("graph_preset_name_label"),
        )
        if not accepted:
            return None

        name = str(name).strip()
        if not name:
            return None
        reserved_builtin_names = {
            self._tr("graph_preset_matanyone2").casefold(),
            "matanyone2",
            "111",
            self._tr("graph_preset_corridorkey_gvm").casefold(),
            "corridorkey+gvm",
        }
        if name.casefold() in reserved_builtin_names:
            QMessageBox.warning(self, self._tr("graph_preset_name_title"), self._tr("graph_preset_reserved_name"))
            return None

        if name in self._graph_custom_presets:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle(self._tr("graph_preset_name_title"))
            box.setText(self._tr("graph_preset_overwrite_message"))
            btn_yes = box.addButton(self._tr("dlg_yes"), QMessageBox.ButtonRole.YesRole)
            box.addButton(self._tr("dlg_no"), QMessageBox.ButtonRole.NoRole)
            box.exec()
            if box.clickedButton() != btn_yes:
                return None

        preset = self._ensure_graph_preset_save_coordinator().build_current_preset()
        if preset is None:
            return None
        self._graph_custom_presets[name] = preset
        self._save_graph_custom_presets()
        QMessageBox.information(self, self._tr("graph_preset_name_title"), self._tr("graph_preset_saved"))
        return f"custom:{name}"

    def _delete_selected_graph_preset(self) -> str | None:
        current_key = self._selected_graph_preset_key
        if not current_key.startswith("custom:"):
            QMessageBox.information(self, self._tr("graph_preset_delete_title"), self._tr("graph_preset_delete_blocked"))
            return None

        preset_name = current_key.split(":", 1)[1]
        if preset_name not in self._graph_custom_presets:
            return self._graph_empty_preset_key

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(self._tr("graph_preset_delete_title"))
        box.setText(self._tr("graph_preset_delete_message").format(name=preset_name))
        btn_yes = box.addButton(self._tr("dlg_yes"), QMessageBox.ButtonRole.YesRole)
        box.addButton(self._tr("dlg_no"), QMessageBox.ButtonRole.NoRole)
        box.exec()
        if box.clickedButton() != btn_yes:
            return None

        self._graph_custom_presets.pop(preset_name, None)
        self._save_graph_custom_presets()
        QMessageBox.information(self, self._tr("graph_preset_delete_title"), self._tr("graph_preset_deleted"))
        return self._graph_empty_preset_key

    def _cleanup_bottom_ui_for_graph_editor(self) -> None:
        """Hide properties panel and SAM controls to prepare space for node graph editor."""
        if self._optional_controls_present:
            # Hide all left panel elements (model parameters)
            self.ui.lbl_sec_model_params.hide()
            self.ui.combo_param_preset.hide()
            self.ui.spin_erode_kernel.hide()
            self.ui.spin_dilate_kernel.hide()
            self.ui.spin_warmup_frames.hide()
            self.ui.lbl_param_preset.hide()
            self.ui.lbl_erode_kernel.hide()
            self.ui.lbl_dilate_kernel.hide()
            self.ui.lbl_warmup_frames.hide()
            self.ui.btn_param_preset_info.hide()
        
        # Also hide the spacer anchors in left panel
        if hasattr(self.ui, 'sp_model_left_width_anchor'):
            self.ui.sp_model_left_width_anchor.changeSize(0, 0)
        if hasattr(self.ui, 'sp_model_left_bottom'):
            self.ui.sp_model_left_bottom.changeSize(0, 0)

        # Hide status label
        for attr in ('lbl_status',):
            if hasattr(self.ui, attr):
                getattr(self.ui, attr).hide()

        # Hide SAM points section
        for attr in ('lbl_sec_points', 'btn_positive_point', 'btn_negative_point',
                      'btn_clear_points', 'btn_live_sam2'):
            if hasattr(self.ui, attr):
                getattr(self.ui, attr).hide()

        # Hide masks section
        for attr in ('lbl_sec_masks', 'btn_generate_mask', 'btn_add_mask',
                      'btn_remove_mask', 'btn_load_mask', 'masks_list'):
            if hasattr(self.ui, attr):
                getattr(self.ui, attr).hide()
        
        # Keep visible:
        # - frame_slider (hl_slider_row) - video scrubber
        # - hl_n_frames (frame count spinner and label)
        # - progress_bar
        # - transport control buttons in hl_play_row (play/pause, prev/next frame)
        
        # Set left panel to minimal width and spacing
        if hasattr(self.ui, 'vl_model_left'):
            self.ui.vl_model_left.setSpacing(0)
            self.ui.vl_model_left.setContentsMargins(0, 0, 0, 0)

        # Keep the center controls column expanded and the now-empty sidebar collapsed.
        self.ui.hl_control_row.setStretch(0, 1)
        self.ui.hl_control_row.setStretch(1, 0)
        
        # Adjust layout spacing to be compact
        if hasattr(self.ui, 'hl_control_row'):
            self.ui.hl_control_row.setSpacing(0)
            self.ui.hl_control_row.invalidate()

    def _setup_embedded_node_graph(self) -> None:
        """Create the node graph editor as an embedded workspace (Nuke-style)."""
        # Create the embedded graph editor widget
        self._embedded_graph_editor = EmbeddedNodeGraphEditor(self._tr, self.ui.centralwidget)
        self._embedded_graph_editor.setMinimumHeight(280)
        # Allow it to expand to fill available space
        
        # Create the actual NodeGraphDialog (will be embedded, not shown as dialog)
        if self._node_graph_dialog is None:
            self._node_graph_dialog = NodeGraphDialog(self._tr, self)
            self._node_graph_dialog.read_media_selected.connect(self._on_graph_read_media_selected)
            self._node_graph_dialog.preview_request_changed.connect(self._on_graph_preview_request_changed)
            self._node_graph_dialog.sam_controls_changed.connect(self._on_graph_sam2_controls_changed)
            self._node_graph_dialog.sam_generate_requested.connect(
                lambda: self.sam2.generate_mask(
                    self.current_frame,
                    show_errors=False,
                    processing_active=self.matting.is_active,
                    current_frame_index=self.current_frame_index,
                    concept=(
                        self._node_graph_dialog.active_sam3_concept()
                        if self._node_graph_dialog is not None
                        else ""
                    ),
                )
            )
            self._node_graph_dialog.sam_clear_requested.connect(self.sam2.clear_points)
            self._node_graph_dialog.sam_add_mask_requested.connect(lambda: self.sam2.add_current_mask(self.current_frame_index))
            self._node_graph_dialog.sam_remove_mask_requested.connect(self._on_graph_sam2_remove_mask_requested)
            self._node_graph_dialog.sam_load_mask_requested.connect(self._on_graph_sam2_load_mask_requested)
            self._node_graph_dialog.sam_propagate_requested.connect(self._on_graph_sam2_propagate_requested)
            self._node_graph_dialog.sam_reprompt_requested.connect(self._on_graph_sam2_reprompt_requested)
            self._node_graph_dialog.sam_session_reset_requested.connect(self._on_graph_sam2_session_reset_requested)
            self._node_graph_dialog.sam_model_type_changed.connect(self.sam2.set_model_type)
            self._node_graph_dialog.sam_props_panel.masks_list.itemSelectionChanged.connect(self._render_input_preview)
            self._node_graph_dialog.active_node_changed.connect(self._on_active_node_changed)
            self._node_graph_dialog.graph_diagnostics_changed.connect(self._on_graph_diagnostics_changed)
            self._node_graph_dialog.scene.changed.connect(lambda _region: self._schedule_graph_dirty_state_update())
        
        # Embed the dialog components (view + props) into the editor
        self._embedded_graph_editor.set_dialog(self._node_graph_dialog)
        self._embedded_graph_editor.get_splitter().splitterMoved.connect(
            lambda _pos, _index: self._sync_playback_controls_with_graph_width()
        )
        
        # Insert it into the main layout (vl_main) after hl_control_row
        # Find the index of hl_control_row in vl_main
        control_row_index = self.ui.vl_main.indexOf(self.ui.hl_control_row)
        if control_row_index >= 0:
            # Insert the graph editor after the control row
            self.ui.vl_main.insertWidget(control_row_index + 1, self._embedded_graph_editor)
            # Set it to expand vertically
            self.ui.vl_main.setStretch(control_row_index + 1, 1)
        else:
            # Fallback: just add it at the end
            self.ui.vl_main.addWidget(self._embedded_graph_editor)

        self._sync_playback_controls_with_graph_width()
        self._refresh_graph_preset_combo(self._graph_empty_preset_key)
        self._set_graph_preset_baseline_from_current()

    def _on_active_node_changed(self, node_type: str) -> None:
        self.sam_interaction.on_active_node_changed(node_type)

    # ── secret log overlay helpers ──

    def _on_brand_badge_clicked(self, _event) -> None:
        self._brand_click_count += 1
        self._brand_click_timer.start()
        if self._brand_click_count >= 5:
            self._brand_click_count = 0
            self._brand_click_timer.stop()
            self._toggle_secret_log()

    def _reset_brand_clicks(self) -> None:
        self._brand_click_count = 0

    def _toggle_secret_log(self) -> None:
        self._secret_log.toggle()
        if self._secret_log.isVisible():
            self._position_secret_log()

    def _position_secret_log(self) -> None:
        parent = self._embedded_graph_editor
        margin = 16
        self._secret_log.setGeometry(
            margin, margin,
            parent.width() - 2 * margin,
            parent.height() - 2 * margin,
        )
        self._secret_log.raise_()

    def _set_viewer_interactive(self, interactive: bool) -> None:
        """Toggle the input viewer between interactive (SAM) and passive modes."""
        lbl = self.ui.input_video_label
        if interactive:
            lbl.setCursor(Qt.CursorShape.CrossCursor)
            lbl.setProperty("viewerActive", True)
        else:
            lbl.setCursor(Qt.CursorShape.ArrowCursor)
            lbl.setProperty("viewerActive", False)
        # Force style refresh so dynamic property takes effect
        lbl.style().unpolish(lbl)
        lbl.style().polish(lbl)

    def _on_graph_read_media_selected(self, media_path: str, media_type: str) -> None:
        media_path = (media_path or "").strip()
        if not media_path or not os.path.exists(media_path):
            return

        request_node_type = ""
        if (
            self._node_graph_dialog is not None
            and hasattr(self._node_graph_dialog, "_active_node")
            and self._node_graph_dialog._active_node is not None
        ):
            request_node_type = str(self._node_graph_dialog._active_node.node_type or "").strip().lower()

        # Source is the only graph node allowed to replace the global input media.
        # Load/Alpha previews are rendered separately and must not overwrite viewer baseline.
        if request_node_type in {"load", "alpha"}:
            return

        want_video = str(media_type).strip().lower() == "video"
        if self.input_path == media_path and self.is_video_input == want_video and self.current_frame is not None:
            if request_node_type == "source":
                self._original_foreground_for_splitter = np.asarray(self.current_frame, dtype=np.uint8)
            self._render_input_preview()
            self._update_frame_info()
            self._set_file_label_state(True, Path(media_path).name)
            return

        self._start_media_load(media_path, want_video, request_node_type=request_node_type)

    def _show_mask_preview_on_output(self, mask: np.ndarray | None) -> None:
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            self.sam_interaction.show_mask_preview_on_output(mask)
            return
        if mask is None:
            return
        mask_array = np.asarray(mask, dtype=np.uint8)
        if mask_array.ndim != 2:
            return

        alpha_rgb = np.stack([mask_array, mask_array, mask_array], axis=-1)
        self._set_selected_node_preview(frame=alpha_rgb)

    def _preview_array_to_rgb(self, frame, *, apply_display_gamma: bool | None = None) -> np.ndarray | None:
        return self._ensure_viewer_preview()._preview_array_to_rgb(frame, apply_display_gamma=apply_display_gamma)

    def _load_image_preview_source(self, path: str) -> np.ndarray:
        return self._ensure_viewer_preview()._load_image_preview_source(path)

    def _load_image_for_preview(self, path: str) -> np.ndarray:
        return self._ensure_viewer_preview()._load_image_for_preview(path)

    def _clear_selected_node_preview(self) -> None:
        self._ensure_viewer_preview()._clear_selected_node_preview()

    def _set_selected_node_preview(self, *, source: str = "", frame=None) -> None:
        self._ensure_viewer_preview()._set_selected_node_preview(source=source, frame=frame)

    def _reset_selected_node_preview_source(self) -> None:
        self._ensure_viewer_preview()._reset_selected_node_preview_source()

    def _prepare_selected_node_preview_source(self, source) -> None:
        self._ensure_viewer_preview()._prepare_selected_node_preview_source(source)

    def _has_selected_node_preview(self) -> bool:
        return self._ensure_viewer_preview()._has_selected_node_preview()

    def _load_selected_node_frame_by_index(self, idx: int):
        return self._ensure_viewer_preview()._load_selected_node_frame_by_index(idx)

    def _on_graph_preview_request_changed(self, node_type: str, _payload: object) -> None:
        self._ensure_viewer_preview()._on_graph_preview_request_changed(node_type, _payload)

    def _apply_export_preview_path(self, write_node_id: str, path: str) -> None:
        self._ensure_viewer_preview()._apply_export_preview_path(write_node_id, path)

    def _save_frames_to_write_output(
        self,
        frames_rgb: list[np.ndarray],
        write_cfg: dict,
        fallback_output_dir: Path,
        default_stem: str,
        *,
        source_is_video: bool,
        source_ext: str,
    ) -> str:
        if not frames_rgb:
            return ""

        video_exts = set(COMPAT_VIDEO_OUTPUT_FORMATS)
        output_fmt = resolve_write_output_format(
            write_cfg,
            Path(f"input{source_ext or ''}"),
        )
        output_dir_raw = str(write_cfg.get("resolved_output_dir", "") or write_cfg.get("output_dir", "")).strip()
        output_dir = Path(output_dir_raw) if output_dir_raw else fallback_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = str(write_cfg.get("file_name", "")).strip() or default_stem

        if output_fmt in video_exts:
            out_path = output_dir / f"{stem}.{output_fmt}"
            self._write_video_frames(
                frames_rgb=frames_rgb,
                output_path=str(out_path),
                codec=str(write_cfg.get("video_codec", "h264")).strip().lower() or "h264",
                crf=int(write_cfg.get("video_quality", 23)),
                preset=str(write_cfg.get("video_preset", "medium")).strip().lower() or "medium",
                fps=float(self.video_fps) if source_is_video else 1.0,
            )
            return str(out_path)

        ext = image_extension_for_format(output_fmt)
        png_compression = int(write_cfg.get("png_compression", 6))
        png_bit_depth = int(write_cfg.get("png_bit_depth", 8))
        jpg_quality = int(write_cfg.get("jpg_quality", 90))
        embed_alpha = bool(write_cfg.get("png_embed_alpha", False))
        if source_is_video and len(frames_rgb) > 1:
            first_path = output_dir / f"0001{ext}"
            for idx, frame in enumerate(frames_rgb, start=1):
                out_path = output_dir / f"{idx:04d}{ext}"
                save_image_frame(
                    frame,
                    out_path,
                    output_fmt=output_fmt,
                    png_compression=png_compression,
                    png_bit_depth=png_bit_depth,
                    jpg_quality=jpg_quality,
                    embed_alpha=embed_alpha,
                )
            return str(first_path)

        out_path = output_dir / f"{stem}{ext}"
        save_image_frame(
            frames_rgb[0],
            out_path,
            output_fmt=output_fmt,
            png_compression=png_compression,
            png_bit_depth=png_bit_depth,
            jpg_quality=jpg_quality,
            embed_alpha=embed_alpha,
        )
        return str(out_path)

    def _save_sam2_mask_output(self, mask_path: str, write_cfg: dict, fallback_output_dir: Path) -> str:
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            return self.sam_interaction.save_sam2_mask_output(mask_path, write_cfg, fallback_output_dir)
        mask_file = Path(mask_path)
        if not mask_file.exists():
            return ""
        with Image.open(mask_file) as _mask_raw:
            mask_img = _mask_raw.convert("L")
            mask_arr = np.asarray(mask_img, dtype=np.uint8).copy()
        mask_rgb = np.stack([mask_arr, mask_arr, mask_arr], axis=-1)
        node_id = str(write_cfg.get("graph_node_id", "")).strip()
        if node_id and self._node_graph_dialog is not None and hasattr(self._node_graph_dialog, "set_write_runtime_preview_for_node"):
            self._node_graph_dialog.set_write_runtime_preview_for_node(node_id, self._to_qimage(mask_rgb))
        return self._save_frames_to_write_output(
            [mask_rgb],
            write_cfg,
            fallback_output_dir / "sam_mask",
            default_stem=mask_file.stem or "sam_mask",
            source_is_video=False,
            source_ext=mask_file.suffix,
        )

    def _save_load_output(self, write_cfg: dict, fallback_output_dir: Path) -> str:
        if self.is_video_input and self.all_frames:
            frames = [np.asarray(frame, dtype=np.uint8) for frame in self.all_frames]
            source_is_video = True
        elif self.current_frame is not None:
            frames = [np.asarray(self.current_frame, dtype=np.uint8)]
            source_is_video = False
        else:
            return ""

        node_id = str(write_cfg.get("graph_node_id", "")).strip()
        if node_id and frames and self._node_graph_dialog is not None and hasattr(self._node_graph_dialog, "set_write_runtime_preview_for_node"):
            self._node_graph_dialog.set_write_runtime_preview_for_node(node_id, self._to_qimage(frames[0]))

        source_ext = Path(self.input_path).suffix if self.input_path else ""
        default_stem = Path(self.input_path).stem if self.input_path else "input"
        return self._save_frames_to_write_output(
            frames,
            write_cfg,
            fallback_output_dir / "input",
            default_stem=default_stem,
            source_is_video=source_is_video,
            source_ext=source_ext,
        )

    @staticmethod
    def _write_video_frames(
        frames_rgb: list[np.ndarray],
        output_path: str,
        codec: str,
        crf: int,
        preset: str,
        fps: float,
    ) -> None:
        ffmpeg_codec, output_params = build_video_output_params(codec, crf=crf, preset=preset)
        prepared_frames = [prepare_video_frame(frame, codec) for frame in frames_rgb]

        try:
            import imageio

            if codec in PRORES_PROFILES:
                imageio.mimwrite(
                    output_path,
                    prepared_frames,
                    fps=max(0.1, float(fps)),
                    codec=ffmpeg_codec,
                    output_params=output_params,
                )
                return

            imageio.mimwrite(
                output_path,
                prepared_frames,
                fps=max(0.1, float(fps)),
                codec=ffmpeg_codec,
                output_params=output_params,
            )
            return
        except Exception:
            pass

        # Fallback writer when imageio/ffmpeg is unavailable.
        first = prepared_frames[0]
        h, w = first.shape[:2]
        ext = Path(output_path).suffix.lower()
        fourcc = "mp4v" if ext in {".mp4", ".mov", ".m4v"} else "XVID"
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*fourcc), max(0.1, float(fps)), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for {output_path}")
        for frame in prepared_frames:
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        writer.release()

    def _call_node_graph_dialog(self, method_name: str, *args, **kwargs):
        dialog = getattr(self, "_node_graph_dialog", None)
        if dialog is None:
            return None
        method = getattr(dialog, method_name, None)
        if not callable(method):
            return None
        return method(*args, **kwargs)

    def _has_node_graph_dialog_method(self, method_name: str) -> bool:
        dialog = getattr(self, "_node_graph_dialog", None)
        return callable(getattr(dialog, method_name, None)) if dialog is not None else False

    def _on_graph_sam2_controls_changed(self, point_mode: str, live_sam2: bool, backend: str) -> None:
        self.sam_interaction.on_graph_controls_changed(point_mode, live_sam2, backend)

    def _sync_sam3_prompt_state_to_graph(self, status_text: str | None = None) -> None:
        self.sam_interaction.sync_sam3_prompt_state_to_graph(status_text)

    def _on_graph_sam2_remove_mask_requested(self) -> None:
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            self.sam_interaction.on_graph_sam2_remove_mask_requested()
        else:
            self.on_remove_masks()

    def _on_graph_sam2_load_mask_requested(self) -> None:
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            self.sam_interaction.on_graph_sam2_load_mask_requested()
        else:
            self.on_load_mask_file()

    def _resolve_effective_video_frame_bounds(self) -> tuple[int, int]:
        total_frames = len(self.all_frames) if self.all_frames else 0
        if not self.is_video_input or total_frames <= 0:
            return 0, total_frames

        start_frame_ui = self.ui.spin_start_frame.value()
        n_frames_limit = self.ui.spin_num_frames.value()
        end_frame_ui = self.ui.spin_end_frame.value()

        effective_start = max(0, start_frame_ui)
        if n_frames_limit > 0:
            effective_end = effective_start + n_frames_limit
        elif end_frame_ui >= 0:
            effective_end = end_frame_ui + 1
        else:
            effective_end = total_frames

        effective_end = min(total_frames, max(effective_start, effective_end))
        return effective_start, effective_end

    def _has_sam2_node_in_graph(self) -> bool:
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            return self.sam_interaction.has_sam2_node_in_graph()
        dialog = getattr(self, "_node_graph_dialog", None)
        if dialog is None or not hasattr(dialog, "export_graph_preset"):
            return False

        preset = dialog.export_graph_preset()
        nodes = preset.get("nodes", []) if isinstance(preset, dict) else []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if str(node.get("type", "")).strip().lower() not in {"sam2"}:
                continue
            props = node.get("properties", {}) or {}
            if not bool(props.get("enabled", True)):
                continue
            return True
        return False

    def _has_sam2_to_matting_mask_link_in_graph(self) -> bool:
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            return self.sam_interaction.has_sam2_to_matting_mask_link_in_graph()
        dialog = getattr(self, "_node_graph_dialog", None)
        if dialog is None or not hasattr(dialog, "export_graph_preset"):
            return False

        preset = dialog.export_graph_preset()
        if not isinstance(preset, dict):
            return False

        nodes = preset.get("nodes", []) or []
        node_map: dict[str, tuple[str, bool]] = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            node_type = str(node.get("type", "")).strip().lower()
            props = node.get("properties", {}) or {}
            enabled = bool(props.get("enabled", True))
            node_map[node_id] = (node_type, enabled)

        for edge in preset.get("connections", []) or []:
            if not isinstance(edge, dict):
                continue
            dst_port = str(edge.get("dst_port", "")).strip().lower()
            if dst_port != "mask":
                continue
            src_id = str(edge.get("src", "")).strip()
            dst_id = str(edge.get("dst", "")).strip()
            src_meta = node_map.get(src_id)
            dst_meta = node_map.get(dst_id)
            if src_meta is None or dst_meta is None:
                continue
            src_type, src_enabled = src_meta
            dst_type, dst_enabled = dst_meta
            if src_enabled and dst_enabled and src_type in {"sam2"} and dst_type == "matting":
                return True
        return False

    def _has_ready_sam2_mask_for_auto_propagation(self) -> bool:
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            return self.sam_interaction.has_ready_sam2_mask_for_auto_propagation()
        if self.sam2.state.current_mask is not None:
            return True
        if bool(self.sam2.state.added_masks):
            return True
        mask_path = str(getattr(self.sam2.state, "mask_path", "") or "").strip()
        return bool(mask_path and Path(mask_path).exists())

    def _ensure_sam2_auto_propagation_state(self) -> None:
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            self.sam_interaction.ensure_sam2_auto_propagation_state()
            return
        if not hasattr(self, "_pending_processing_after_sam2_auto_propagate"):
            self._pending_processing_after_sam2_auto_propagate = False
        if not hasattr(self, "_skip_next_auto_sam2_propagate"):
            self._skip_next_auto_sam2_propagate = False

    def _notify_auto_propagate_skipped_for_matting_link(self) -> None:
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            self.sam_interaction._notify_auto_propagate_skipped_for_matting_link()
            return
        text = "Auto SAM2 skipped: SAM is connected to Matting mask input"
        tr_method = getattr(self, "_tr", None)
        if callable(tr_method):
            try:
                text = tr_method("sam2_auto_propagate_skipped_matting_mask")
            except Exception:
                pass

        sam_status = getattr(getattr(self, "sam2", None), "status_changed", None)
        if hasattr(sam_status, "emit") and callable(sam_status.emit):
            try:
                sam_status.emit(text)
                return
            except Exception:
                pass

        set_status = getattr(self, "_set_status", None)
        if callable(set_status):
            try:
                set_status(text)
            except Exception:
                pass

    def _try_auto_propagate_sam2_before_processing(self) -> bool:
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            return self.sam_interaction.try_auto_propagate_sam2_before_processing()
        self._ensure_sam2_auto_propagation_state()
        if self._skip_next_auto_sam2_propagate:
            self._skip_next_auto_sam2_propagate = False
            return False
        if self._pending_processing_after_sam2_auto_propagate:
            return True
        if self.sam2.generation_active or self.matting.is_active:
            return False
        if not self.is_video_input or len(self.all_frames or []) <= 1:
            return False
        if not self._has_sam2_node_in_graph():
            return False
        if self._has_sam2_to_matting_mask_link_in_graph():
            return False
        if not self._has_ready_sam2_mask_for_auto_propagation():
            return False
        if not bool(self.sam2.state.points):
            return False

        self._pending_processing_after_sam2_auto_propagate = True
        self._on_graph_sam2_propagate_requested("forward")
        if self.sam2.generation_active:
            return True

        self._pending_processing_after_sam2_auto_propagate = False
        return False

    def _on_graph_sam2_propagate_requested(self, direction: str) -> None:
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            self.sam_interaction.on_graph_sam2_propagate_requested(direction)
            return
        frame_start, frame_end = self._resolve_effective_video_frame_bounds()
        if self.current_frame_index < frame_start or self.current_frame_index >= frame_end:
            self.sam2.status_changed.emit(
                self._tr("sam2_current_frame_out_of_range").format(
                    start=frame_start,
                    end=max(frame_start, frame_end - 1),
                )
            )
            return

        tracking_frames = self.all_frames[frame_start:frame_end]
        self.sam2.propagate_video(
            direction=direction,
            all_frames=tracking_frames,
            current_frame_index=self.current_frame_index - frame_start,
            frame_index_offset=frame_start,
            current_frame_index_global=self.current_frame_index,
            processing_active=self.matting.is_active,
        )

    def _on_graph_sam2_reprompt_requested(self) -> None:
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            self.sam_interaction.on_graph_sam2_reprompt_requested()
            return
        self.sam2.reprompt_video_frame(
            current_frame=self.current_frame,
            all_frames=self.all_frames,
            current_frame_index=self.current_frame_index,
            processing_active=self.matting.is_active,
        )

    def _on_graph_sam2_session_reset_requested(self) -> None:
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            self.sam_interaction.on_graph_sam2_session_reset_requested()
            return
        self.sam2.reset_video_session()
        if self._node_graph_dialog is not None and hasattr(self._node_graph_dialog, "clear_node_frame_progress"):
            self._node_graph_dialog.clear_node_frame_progress("sam2")

    def _sync_playback_controls_with_graph_width(self) -> None:
        """Keep playback controls visually centred in the available row width."""
        self.ui.hl_control_row.invalidate()
        self.ui.hl_slider_row.invalidate()
        self.ui.hl_play_row.invalidate()

    def _sync_viewer_header_heights(self) -> None:
        """Keep top header row heights equal so both viewer blocks align visually."""
        output_header = getattr(self.ui, "hl_output_header_row", None)
        input_title = getattr(self.ui, "lbl_input_title", None)
        if output_header is None or input_title is None:
            return
        out_h = output_header.sizeHint().height()
        in_h = input_title.sizeHint().height()
        input_title.setFixedHeight(max(in_h, out_h))

    def _setup_completion_sound(self) -> None:
        self._completion_sound_effect = None
        if QSoundEffect is None:
            return

        sound_path = self._app_root / "UI" / "end.wav"
        if not sound_path.exists():
            return

        try:
            effect = QSoundEffect(self)
            effect.setSource(QUrl.fromLocalFile(str(sound_path)))
            effect.setLoopCount(1)
            effect.setVolume(0.35)
            self._completion_sound_effect = effect
        except Exception:
            self._completion_sound_effect = None

    def _set_completion_sound_enabled(self, enabled: bool) -> None:
        self._completion_sound_enabled = bool(enabled)
        self._settings.setValue("ui/play_completion_sound", self._completion_sound_enabled)

    def _play_completion_sound(self) -> None:
        if not getattr(self, "_completion_sound_enabled", True):
            return
        effect = getattr(self, "_completion_sound_effect", None)
        if effect is not None:
            try:
                effect.stop()
                effect.play()
                return
            except Exception:
                pass

        # Fallback for systems without multimedia backend.
        try:
            QApplication.beep()
        except Exception:
            pass

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._secret_log.isVisible():
            self._position_secret_log()
        self._update_viewer_aspect()
        self._sync_playback_controls_with_graph_width()

    def _update_viewer_aspect(self) -> None:
        """Enforce 16:9 on both viewer labels with equal geometry."""
        cw = self.centralWidget()
        if cw is None:
            return
        self._sync_viewer_header_heights()

        def _hint_h(name: str, default: int = 0) -> int:
            obj = getattr(self.ui, name, None)
            if obj is None:
                return default
            try:
                return int(obj.sizeHint().height())
            except Exception:
                return default

        margins = self.ui.vl_main.contentsMargins()
        avail_w = cw.width() - margins.left() - margins.right()
        viewers_layout = getattr(self.ui, "hl_viewers", None)
        spacing = viewers_layout.spacing() if viewers_layout is not None else 0

        viewer_w = max(320, (avail_w - spacing) // 2)
        viewer_h_by_width = int(viewer_w * 9 / 16)

        top_h = _hint_h("hl_topbar")
        media_h = _hint_h("hl_media_select")
        controls_h = _hint_h("hl_control_row")
        spacer_h = _hint_h("sp_between_viewers_and_controls")
        main_spacing = self.ui.vl_main.spacing() * 4
        free_h = (
            cw.height()
            - margins.top()
            - margins.bottom()
            - top_h
            - media_h
            - controls_h
            - spacer_h
            - main_spacing
        )
        viewer_h = min(viewer_h_by_width, max(120, free_h))

        self.ui.input_video_label.setFixedHeight(viewer_h)
        self.ui.output_video_label.setFixedHeight(viewer_h)
        self._rescale_viewer_pixmaps()

    def _rescale_viewer_pixmaps(self) -> None:
        """Re-scale stored original pixmaps to current label sizes."""
        for label, src in (
            (self.ui.input_video_label, self._input_source_pixmap),
            (self.ui.output_video_label, self._output_source_pixmap),
        ):
            if src is None or src.isNull():
                continue
            scaled = src.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            label.setPixmap(scaled)

    @staticmethod
    def _format_tooltip(text: str, width: int = 260) -> str:
        safe_text = escape(text)
        return (
            f"<html><body><div style='width: {width}px; white-space: normal;'>"
            f"{safe_text}"
            "</div></body></html>"
        )

    def _configure_tooltips(self):
        self.ui.btn_settings.setToolTip(
            self._format_tooltip(self._tr("btn_settings_tooltip"))
        )
        self.ui.spin_start_frame.setToolTip(self._format_tooltip(self._tr("spin_start_frame_tooltip")))
        self.ui.spin_num_frames.setToolTip(self._format_tooltip(self._tr("spin_num_frames_tooltip")))
        self.ui.spin_end_frame.setToolTip(self._format_tooltip(self._tr("spin_end_frame_tooltip")))
        self.ui.btn_preview_foreground.setToolTip(self._format_tooltip(self._tr("preview_display_gamma_tooltip")))
        self.ui.btn_preview_alpha.setToolTip(self._format_tooltip(self._tr("preview_linear_tooltip")))
        self.ui.btn_split_view.setToolTip(self._format_tooltip(self._tr("split_view_tooltip")))
        transport_tooltips = (
            ("btn_first_frame", "btn_first_frame_tooltip"),
            ("btn_prev_frame", "btn_prev_frame_tooltip"),
            ("btn_play_reverse", "btn_play_reverse_tooltip"),
            ("btn_play", "btn_play_tooltip"),
            ("btn_next_frame", "btn_next_frame_tooltip"),
            ("btn_last_frame", "btn_last_frame_tooltip"),
            ("btn_play_loop", "btn_play_loop_tooltip"),
        )
        for name, tooltip_key in transport_tooltips:
            button = self._transport_button(name)
            if button is not None:
                button.setToolTip(self._format_tooltip(self._tr(tooltip_key)))
        if hasattr(self, "_combo_playback_presets"):
            preset_tip = "Playback presets for node graph workflow." if self._language_code == "en" else "Пресеты воспроизведения для графа нод."
            self._combo_playback_presets.setToolTip(self._format_tooltip(preset_tip))
        if self._optional_controls_present:
            self.ui.btn_load.setToolTip(self._format_tooltip(self._tr("btn_load_tooltip")))
            self.ui.lbl_param_preset.setToolTip(self._format_tooltip(self._tr("lbl_preset_tooltip")))
            self.ui.lbl_erode_kernel.setToolTip(self._format_tooltip(self._tr("lbl_erode_tooltip")))
            self.ui.spin_erode_kernel.setToolTip(self._format_tooltip(self._tr("spin_erode_tooltip")))
            self.ui.lbl_dilate_kernel.setToolTip(self._format_tooltip(self._tr("lbl_dilate_tooltip")))
            self.ui.spin_dilate_kernel.setToolTip(self._format_tooltip(self._tr("spin_dilate_tooltip")))
            self.ui.lbl_warmup_frames.setToolTip(self._format_tooltip(self._tr("lbl_warmup_tooltip")))
            self.ui.spin_warmup_frames.setToolTip(self._format_tooltip(self._tr("spin_warmup_tooltip")))
            self.ui.btn_positive_point.setToolTip(self._format_tooltip(self._tr("btn_positive_tooltip")))
            self.ui.btn_negative_point.setToolTip(self._format_tooltip(self._tr("btn_negative_tooltip")))
            self.ui.btn_live_sam2.setToolTip(self._format_tooltip(self._tr("btn_live_sam2_tooltip")))
            self.ui.btn_generate_mask.setToolTip(self._format_tooltip(self._tr("btn_generate_mask_tooltip")))
            self.ui.btn_add_mask.setToolTip(self._format_tooltip(self._tr("btn_add_mask_tooltip")))
            self.ui.btn_remove_mask.setToolTip(self._format_tooltip(self._tr("btn_remove_mask_tooltip")))
            self.ui.btn_load_mask.setToolTip(self._format_tooltip(self._tr("btn_load_mask_tooltip")))
        self.ui.btn_save_result.setToolTip(self._format_tooltip(self._tr("btn_save_result_tooltip")))
        self._update_graph_sam_tracking_tooltips()

    def _update_graph_sam_tracking_tooltips(self) -> None:
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            self.sam_interaction.update_graph_sam_tracking_tooltips()
            return
        dialog = getattr(self, "_node_graph_dialog", None)
        if dialog is None or not hasattr(dialog, "sam_props_panel"):
            return

        panel = dialog.sam_props_panel
        frame_start, frame_end = self._resolve_effective_video_frame_bounds()
        if frame_end <= frame_start:
            return

        range_text = self._tr("sam2_tracking_range_tooltip").format(
            start=frame_start + 1,
            end=frame_end,
        )
        backward_tip = self._format_tooltip(
            f"{self._tr('sam2_btn_propagate_backward_tooltip')}\n\n{range_text}",
            width=320,
        )
        forward_tip = self._format_tooltip(
            f"{self._tr('sam2_btn_propagate_forward_tooltip')}\n\n{range_text}",
            width=320,
        )
        panel.btn_sam2_propagate_backward.setToolTip(backward_tip)
        panel.btn_sam2_propagate_forward.setToolTip(forward_tip)
        

    def _set_language(self, language_code: str, announce: bool = True):
        language = (language_code or "ru").strip().lower()
        if language not in {"ru", "en"}:
            language = "ru"
        self._language_code = language
        self._settings.setValue("ui/language", language)
        self.sam2.set_language(language)
        self.matting.set_language(language)
        cloud_controller = getattr(self, "_cloud_inference", None)
        if cloud_controller is not None:
            cloud_controller.set_translator(self._tr)
        self._apply_language(announce=announce)

    def _apply_device_selection(self):
        device = self._device_selection

        if device in {"cpu", "mps", "cuda"}:
            os.environ["KEYFLOW_DEVICE"] = device
        else:
            os.environ["KEYFLOW_DEVICE"] = ""

        ModelService().reinit_device()
        self._update_window_title_with_device()
        cloud_enabled = bool(get_cloud_setting("cloud/enabled"))
        if cloud_enabled:
            self._set_status(
                f"{self._tr('status_device')} {self._tr('status_cloud_gpu')}"
            )
        else:
            self._set_status(
                f"{self._tr('status_device')} {self._get_device_display_label()}"
            )

    def on_device_changed(self, text: str):
        device = text.strip().lower()
        if device not in self._device_help:
            device = "auto"
        self._device_selection = device
        self._settings.setValue("runtime/device", device)
        self._apply_device_selection()

    def on_compatibility_profile_changed(self, text: str):
        profile = text.strip().lower()
        if profile not in self._compat_profile_help:
            profile = "auto"
        self._compatibility_profile = profile
        self._settings.setValue("runtime/compatibility_profile", profile)

    def _update_device_tooltip(self, text: str | None = None):
        if text is None:
            text = self._device_selection

        device = text.strip().lower()
        help_key = device if device in self._device_help else "auto"
        return self._format_tooltip(self._tr(self._device_help[help_key]))

    def _update_compat_profile_tooltip(self, text: str | None = None):
        if text is None:
            text = self._compatibility_profile

        profile = text.strip().lower()
        help_key = profile if profile in self._compat_profile_help else "auto"
        return self._format_tooltip(self._tr(self._compat_profile_help[help_key]))

    def open_about_dialog(self):
        title = "About KeyFlow Studio" if self._language_code == "en" else "О программе"

        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)
        dialog.resize(600, 480)
        dialog.setMinimumSize(QSize(600, 480))
        dialog.setStyleSheet(
            "QDialog { background-color: #10151d; }"
            "QLabel#about_title { font-size: 30px; font-weight: 750; color: #f1f8ff; }"
            "QLabel#about_subtitle { font-size: 15px; color: #8edbff; font-weight: 600; }"
            "QLabel#about_version { font-size: 12px; color: #b7c6d8; }"
            "QLabel#about_text { font-size: 13px; color: #d7e2ef; line-height: 1.35; }"
            "QLabel#about_link { font-size: 13px; color: #9fe3ff; }"
        )

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(26, 22, 26, 18)
        layout.setSpacing(10)

        self._build_about_header(dialog, layout)
        self._build_about_credits(dialog, layout)
        tg_icon_label = self._build_about_tg_row(dialog, layout)
        ht_logo_btn = self._build_about_ht_logo(dialog, layout)
        self._fetch_about_tg_icon(dialog, tg_icon_label)
        self._fetch_about_ht_logo(dialog, ht_logo_btn)

        if self._language_code == "en":
            thanks_text = (
                "Many thanks for the original idea and research contribution:\n"
                "Yang, Peiqing and Zhou, Shangchen and Hao, Kai and Tao, Qingyi, Neyrograph,Ksenia Prokofyeva, Pavel Ushakov"
            )
        else:
            thanks_text = (
                "Большое спасибо за идею и научный вклад:\n"
                "Yang, Peiqing and Zhou, Shangchen and Hao, Kai and Tao, Qingyi, Нейрограф, Ксения Прокофьева, Павел Ушаков"
            )

        thanks_label = QLabel(thanks_text, dialog)
        thanks_label.setObjectName("about_text")
        thanks_label.setWordWrap(True)
        thanks_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(thanks_label)

        layout.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, parent=dialog)
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setText("OK" if self._language_code == "en" else "ОК")
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.exec()

    # ── open_about_dialog helpers ──

    def _build_about_header(self, dialog: QDialog, layout: QVBoxLayout) -> None:
        icon_label = QLabel(dialog)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_candidates = [
            self._app_root / "MatAnyone2.icns",
            Path(sys.executable).resolve().parents[1] / "Resources" / "MatAnyone2.icns",
            Path(sys.executable).resolve().parents[1] / "Resources" / "icon-windowed.icns",
        ]
        icon_pixmap = QPixmap()
        for icon_path in icon_candidates:
            if icon_path.exists() and icon_pixmap.load(icon_path.as_posix()):
                break

        if not icon_pixmap.isNull():
            icon_label.setPixmap(
                icon_pixmap.scaled(
                    132, 132,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        layout.addWidget(icon_label)

        headline = QLabel("KeyFlow Studio", dialog)
        headline.setObjectName("about_title")
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(headline)

        subtitle = QLabel("Scaling Video Matting via a Learned Quality Evaluator", dialog)
        subtitle.setObjectName("about_subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        version_label = QLabel(f"Version {APP_VERSION}", dialog)
        version_label.setObjectName("about_version")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)

    def _build_about_credits(self, dialog: QDialog, layout: QVBoxLayout) -> None:
        if self._language_code == "en":
            author_text = "Development: Alexander Filyukov"
        else:
            author_text = "Разработка: Александр Филюков"

        author_label = QLabel(author_text, dialog)
        author_label.setObjectName("about_text")
        author_label.setWordWrap(True)
        author_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(author_label)

    def _build_about_tg_row(self, dialog: QDialog, layout: QVBoxLayout) -> QLabel:
        tg_row = QWidget(dialog)
        tg_layout = QHBoxLayout(tg_row)
        tg_layout.setContentsMargins(0, 0, 0, 0)
        tg_layout.setSpacing(5)
        tg_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        tg_icon_label = QLabel(tg_row)
        tg_icon_label.setFixedSize(18, 18)
        tg_icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        tg_link_label = QLabel(
            '<a href="https://t.me/filik_by" style="color: #9fe3ff;">@filik_by</a>',
            tg_row,
        )
        tg_link_label.setObjectName("about_link")
        tg_link_label.setOpenExternalLinks(True)
        tg_link_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        tg_link_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        tg_layout.addWidget(tg_icon_label)
        tg_layout.addWidget(tg_link_label)
        layout.addWidget(tg_row)
        return tg_icon_label

    def _build_about_ht_logo(self, dialog: QDialog, layout: QVBoxLayout) -> QPushButton:
        ht_logo_btn = QPushButton(dialog)
        ht_logo_btn.setFlat(True)
        ht_logo_btn.setFixedHeight(32)
        ht_logo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        ht_logo_btn.setToolTip("Hockey Tek")
        ht_logo_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://hockeytek.com"))
        )
        ht_logo_wrap = QWidget(dialog)
        ht_logo_layout = QHBoxLayout(ht_logo_wrap)
        ht_logo_layout.setContentsMargins(0, 0, 0, 0)
        ht_logo_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ht_logo_layout.addWidget(ht_logo_btn)
        layout.addWidget(ht_logo_wrap)
        return ht_logo_btn

    def _fetch_about_tg_icon(self, dialog: QDialog, tg_icon_label: QLabel) -> None:
        nam = QNetworkAccessManager(dialog)
        reply = nam.get(QNetworkRequest(QUrl("https://cdn.simpleicons.org/telegram/0088cc")))

        def _on_ready():
            if reply.error() == reply.NetworkError.NoError:
                svg_data = reply.readAll()
                renderer = QSvgRenderer(svg_data)
                pixmap = QPixmap(18, 18)
                pixmap.fill(Qt.GlobalColor.transparent)
                p = QPainter(pixmap)
                renderer.render(p)
                p.end()
                tg_icon_label.setPixmap(pixmap)
            else:
                tg_icon_label.setText('<span style="color:#2AABEE;font-size:15px;">✈</span>')
            reply.deleteLater()

        reply.finished.connect(_on_ready)

    def _fetch_about_ht_logo(self, dialog: QDialog, ht_logo_btn: QPushButton) -> None:
        nam = QNetworkAccessManager(dialog)
        reply = nam.get(QNetworkRequest(QUrl(
            "https://static.tildacdn.com/tild3366-3334-4762-b866-613364366161/logo_ht.svg"
        )))

        def _on_ready():
            if reply.error() == reply.NetworkError.NoError:
                svg_data = reply.readAll()
                renderer = QSvgRenderer(svg_data)
                vb = renderer.viewBoxF()
                w = int(vb.width() / vb.height() * 32) if vb.height() > 0 else 120
                w = max(60, min(220, w))
                pixmap = QPixmap(w, 32)
                pixmap.fill(Qt.GlobalColor.transparent)
                p = QPainter(pixmap)
                renderer.render(p)
                p.end()
                ht_logo_btn.setIcon(QIcon(pixmap))
                ht_logo_btn.setIconSize(QSize(w, 32))
                ht_logo_btn.setFixedWidth(w)
            reply.deleteLater()

        reply.finished.connect(_on_ready)


    def on_input_type_changed(self, _index: int):
        if not self._optional_controls_present:
            return
        self.is_video_input = self.ui.combo_input_type.currentData() != "image"
        self._set_status(f"{self._tr('status_input_type')} {self.ui.combo_input_type.currentText()}")

    def on_load_media(self):
        if self.is_video_input:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                self._tr("dlg_select_video_title"),
                "",
                f"{self._tr('dlg_filter_video')} (*.mp4 *.avi *.mov *.mkv *.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff *.exr);;{self._tr('dlg_filter_all')} (*)",
            )
            if not file_path:
                return
            self._start_media_load(file_path, True)
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("dlg_select_image_title"),
            "",
            f"{self._tr('dlg_filter_image')} (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff *.exr);;{self._tr('dlg_filter_all')} (*)",
        )
        if not file_path:
            return
        self._start_media_load(file_path, False)

    def _start_media_load(self, media_path: str, is_video: bool, request_node_type: str = "") -> None:
        if self._media_loading_active:
            return
        if self.matting.is_active or self.sam2.generation_active:
            self._set_status(self._tr("sam_wait_processing"))
            return

        self._cleanup_media_loader()
        self._media_loading_active = True
        self._media_loading_target_is_video = bool(is_video)
        self._media_loading_request_node_type = str(request_node_type or "").strip().lower()
        self._set_media_loading_busy(True)
        self.ui.progress_bar.setRange(0, 100)
        self.ui.progress_bar.setValue(0)
        self.ui.progress_bar.setTextVisible(True)
        self._set_status(self._tr("status_media_loading").format(percent=0))

        self._media_loader_thread = QThread(self)
        self._media_loader_worker = MediaLoadWorker()
        self._media_loader_worker.configure_job(media_path, is_video, self._language_code)
        self._media_loader_worker.moveToThread(self._media_loader_thread)

        self._media_loader_thread.started.connect(self._media_loader_worker.run)
        self._media_loader_worker.progress.connect(self._on_media_load_progress)
        self._media_loader_worker.finished.connect(self._on_media_load_finished)
        self._media_loader_worker.error.connect(self._on_media_load_error)
        self._media_loader_thread.finished.connect(self._media_loader_thread.deleteLater)

        self._media_loader_thread.start()

    def _apply_loaded_media_frame_range(self, total_frames: int, is_video: bool) -> None:
        total = max(1, int(total_frames or 0))
        last = total - 1
        start_frame = 0
        frame_count = total if is_video else 1
        end_frame = last if is_video else 0

        self.ui.spin_start_frame.blockSignals(True)
        self.ui.spin_num_frames.blockSignals(True)
        self.ui.spin_end_frame.blockSignals(True)
        try:
            for spin_box, maximum in (
                (self.ui.spin_start_frame, last),
                (self.ui.spin_end_frame, last),
                (self.ui.spin_num_frames, total),
            ):
                set_maximum = getattr(spin_box, "setMaximum", None)
                if callable(set_maximum):
                    set_maximum(maximum)
            self.ui.spin_start_frame.setValue(start_frame)
            self.ui.spin_num_frames.setValue(frame_count)
            self.ui.spin_end_frame.setValue(end_frame)
        finally:
            self.ui.spin_start_frame.blockSignals(False)
            self.ui.spin_num_frames.blockSignals(False)
            self.ui.spin_end_frame.blockSignals(False)

    def _on_media_load_progress(self, percent: int, _status_text: str) -> None:
        clamped = max(0, min(100, int(percent)))
        self.ui.progress_bar.setRange(0, 100)
        self.ui.progress_bar.setValue(clamped)
        self._set_status(self._tr("status_media_loading").format(percent=clamped))

    def _on_media_load_finished(self, payload: dict) -> None:
        request_node_type = self._media_loading_request_node_type
        self._media_loading_request_node_type = ""
        self._cleanup_media_loader()
        self._media_loading_active = False
        self._set_media_loading_busy(False)

        if isinstance(payload, dict) and payload.get("cancelled"):
            self.ui.progress_bar.setValue(0)
            self._set_status(self._tr("status_stopped"))
            return

        if not payload:
            self.input_path = None
            self._set_file_label_state(False)
            self.ui.progress_bar.setValue(0)
            return

        frames = payload.get("frames") or []
        if not frames:
            self.input_path = None
            self._set_file_label_state(False)
            self.ui.progress_bar.setValue(0)
            return

        self.input_path = str(payload.get("path") or "") or None
        self.is_video_input = bool(payload.get("is_video", True))
        self.video_fps = float(payload.get("fps", 30.0) or 30.0)
        self.all_frames = frames
        self.current_frame_index = 0
        self.current_frame = frames[0]
        # Splitter baseline must be anchored to Source node media only.
        if request_node_type in {"", "source"}:
            self._original_foreground_for_splitter = np.asarray(self.current_frame, dtype=np.uint8)
        self._auto_select_eval_preset_for_frame(self.current_frame)
        self._reset_viewer_preview()
        self._suspend_sam2_graph_sync = True
        try:
            self.sam2.reset_for_media()
        finally:
            self._suspend_sam2_graph_sync = False

        frame_slider = self._transport_slider()
        if frame_slider is not None:
            frame_slider.setMaximum(max(0, len(frames) - 1))
        self._apply_loaded_media_frame_range(len(frames), self.is_video_input)
        self._apply_frame_range_from_preset(
            self._graph_preset_payload(getattr(self, "_selected_graph_preset_key", "")),
            total_frames=len(frames),
        )
        if frame_slider is not None:
            frame_slider.setValue(0)

        self._render_input_preview()
        self._update_frame_info()
        self.ui.progress_bar.setValue(100)
        self._set_file_label_state(True, Path(self.input_path).name if self.input_path else "")
        self._set_status(self._get_file_info_string())

        # If graph node selection initiated the load, update splitter preview
        # according to selected node role (Source = baseline, Load/Alpha = compare).
        if (
            self._node_graph_dialog is not None
            and hasattr(self._node_graph_dialog, "_active_node")
            and self._node_graph_dialog._active_node is not None
        ):
            active_type = str(self._node_graph_dialog._active_node.node_type or "").strip().lower()
            if active_type == "source":
                source_path = str(self.input_path or "").strip()
                if source_path and os.path.exists(source_path):
                    self._set_selected_node_preview(source=source_path)
            elif active_type in {"load", "alpha"} and self._original_foreground_for_splitter is not None:
                compare_frame = np.asarray(self.current_frame, dtype=np.uint8)
                if compare_frame.ndim == 3:
                    gray = cv2.cvtColor(compare_frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray = compare_frame
                compare_rgb = np.stack([gray, gray, gray], axis=-1)
                self._set_selected_node_preview(frame=compare_rgb)

        self._restore_write_outputs_from_disk()
        self.sam2_graph.restore_masks_from_graph_node()

    def _on_media_load_error(self, _message: str) -> None:
        was_video = self._media_loading_target_is_video
        self._media_loading_request_node_type = ""
        self._cleanup_media_loader()
        self._media_loading_active = False
        self._set_media_loading_busy(False)
        self.input_path = None
        self._set_file_label_state(False)
        self.ui.progress_bar.setValue(0)

        if was_video:
            QMessageBox.warning(self, self._tr("status_error"), self._tr("err_cant_open_video"))
        else:
            QMessageBox.warning(self, self._tr("status_error"), self._tr("err_cant_open_image"))

    def _cleanup_media_loader(self) -> None:
        if self._media_loader_worker is not None:
            self._media_loader_worker.request_cancel()
        if self._media_loader_thread is not None:
            self._media_loader_thread.quit()
            self._media_loader_thread.wait()
        self._media_loader_thread = None
        self._media_loader_worker = None

    def _set_media_loading_busy(self, busy: bool) -> None:
        if self._optional_controls_present:
            self.ui.btn_load.setEnabled(not busy)
            self.ui.combo_input_type.setEnabled(not busy)
        self.ui.btn_settings.setEnabled(not busy)
        self.ui.spin_start_frame.setEnabled(not busy)
        self.ui.spin_num_frames.setEnabled(not busy)
        self.ui.spin_end_frame.setEnabled(not busy)
        self._set_transport_controls_enabled(not busy)
        self._refresh_stop_button_state()
        if busy:
            self.ui.btn_run.setEnabled(False)
            return

        if self.sam2.generation_active or self.matting.is_active or self._is_cloud_processing_active():
            return
        self.ui.btn_run.setEnabled(True)

    def _refresh_stop_button_state(self) -> None:
        sam_busy = bool(getattr(self, "sam2", None) is not None and self.sam2.generation_active)
        busy = bool(self._media_loading_active or self.matting.is_active or sam_busy or self._is_cloud_processing_active())
        self.ui.btn_stop.setEnabled(busy)

    def _get_file_info_string(self) -> str:
        """Генерирует строку с подробной информацией о загруженном файле."""
        if self.input_path is None:
            return self._tr("file_not_loaded")
        
        info_parts = []
        
        # Имя файла
        filename = Path(self.input_path).name
        info_parts.append(f"{self._tr('file_info_file')} {filename}")
        
        # Размер файла
        try:
            file_size_mb = os.path.getsize(self.input_path) / (1024 * 1024)
            info_parts.append(f"{self._tr('file_info_size')} {file_size_mb:.2f} {self._tr('file_info_size_unit')}")
        except Exception:
            pass
        
        # Разрешение
        if self.current_frame is not None:
            h, w = self.current_frame.shape[:2]
            info_parts.append(f"{self._tr('file_info_res')} {w}×{h}")
        
        # Информация о видео (если видео)
        if self.is_video_input and self.all_frames:
            frame_count = len(self.all_frames)
            fps = self.video_fps
            duration_sec = frame_count / fps if fps > 0 else 0
            
            info_parts.append(f"{self._tr('file_info_frames')} {frame_count}")
            info_parts.append(f"{self._tr('file_info_fps')} {fps:.2f}")
            
            mins = int(duration_sec // 60)
            secs = int(duration_sec % 60)
            info_parts.append(f"{self._tr('file_info_duration')} {mins}:{secs:02d}")
        else:
            info_parts.append(self._tr("file_info_type_image"))
        
        return " | ".join(info_parts)

    def _set_file_label_state(self, loaded: bool, filename: str = ""):
        if loaded and filename:
            self.ui.lbl_file_path.setText(filename)
            self.ui.lbl_file_path.setStyleSheet(self._file_label_style_active)
            
            # Добавляем подробную информацию в tooltip
            full_info = self._get_file_info_string()
            self.ui.lbl_file_path.setToolTip(self._format_tooltip(full_info, width=350))
            return

        self.ui.lbl_file_path.setText(self._tr("file_not_loaded"))
        self.ui.lbl_file_path.setStyleSheet(self._file_label_style_inactive)
        self.ui.lbl_file_path.setToolTip("")

    def on_parameter_preset_changed(self, _index: int):
        if not self._optional_controls_present:
            return
        preset_name = self._current_preset_key()
        self._update_parameter_preset_tooltip(preset_name)

        if self._preset_sync_in_progress:
            return

        values = self._parameter_presets.get(preset_name)
        if values is None:
            return

        self._preset_sync_in_progress = True
        try:
            erode, dilate, warmup = values
            self.ui.spin_erode_kernel.setValue(erode)
            self.ui.spin_dilate_kernel.setValue(dilate)
            self.ui.spin_warmup_frames.setValue(warmup)
        finally:
            self._preset_sync_in_progress = False

    def _ensure_parameter_preset_options(self):
        if not self._optional_controls_present:
            return
        desired_order = [
            "Eval LR (512p)",
            "Eval HR (1080p)",
            "Balanced",
            "Fine Edges",
            "Clean Cut",
            "Stable Start",
            "Custom",
        ]

        combo = self.ui.combo_param_preset
        current_value = self._current_preset_key()

        combo.blockSignals(True)
        try:
            combo.clear()
            for name in desired_order:
                combo.addItem(self._preset_display_name(name), name)

            if current_value in desired_order:
                self._set_current_preset_key(current_value)
            else:
                self._set_current_preset_key("Balanced")
        finally:
            combo.blockSignals(False)

    def _auto_select_eval_preset_for_frame(self, frame: np.ndarray | None):
        if not self._optional_controls_present:
            return
        if not self._auto_eval_preset_enabled or frame is None:
            return

        # Respect explicit manual tuning.
        if self._current_preset_key() == "Custom":
            return

        h, w = frame.shape[:2]
        min_side = min(h, w)

        target_preset = None
        if min_side <= 576:
            target_preset = "Eval LR (512p)"
        elif min_side >= 900:
            target_preset = "Eval HR (1080p)"

        if not target_preset:
            return
        if self._current_preset_key() == target_preset:
            return

        self._set_current_preset_key(target_preset)
        self._set_status(f"{self._tr('status_autopreset')} {self._preset_display_name(target_preset)} ({w}x{h})")

    def _sync_preset_selection(self):
        if not self._optional_controls_present:
            return
        if self._preset_sync_in_progress:
            return

        current_values = (
            self.ui.spin_erode_kernel.value(),
            self.ui.spin_dilate_kernel.value(),
            self.ui.spin_warmup_frames.value(),
        )
        matched_preset = next(
            (name for name, values in self._parameter_presets.items() if values == current_values),
            "Custom",
        )

        if self._current_preset_key() == matched_preset:
            return

        self._preset_sync_in_progress = True
        try:
            self._set_current_preset_key(matched_preset)
        finally:
            self._preset_sync_in_progress = False

        self._update_parameter_preset_tooltip(matched_preset)

    def _update_parameter_preset_tooltip(self, preset_name: str | None = None):
        if not self._optional_controls_present:
            return
        if preset_name is None:
            preset_name = self._current_preset_key()

        hint_key = self._parameter_preset_help.get(preset_name, self._parameter_preset_help["Custom"])
        hint_text = self._tr(hint_key)
        formatted_hint = self._format_tooltip(hint_text)
        self.ui.btn_param_preset_info.setToolTip(formatted_hint)
        self.ui.combo_param_preset.setToolTip(formatted_hint)

    def _set_point_mode(self, positive: bool):
        self.sam_interaction.set_point_mode(positive)

    def _on_live_sam2_toggled(self, checked: bool):
        self.sam_interaction.on_live_sam2_toggled(checked)

    def _on_input_label_mouse_press(self, event):
        self.sam_interaction.on_input_label_mouse_press(event)

    def _map_click_to_image(self, click_x: float, click_y: float):
        return self.sam_interaction.map_click_to_image(click_x, click_y)

    def _selected_mask_rows(self):
        return self.sam_interaction.selected_mask_rows()

    def _effective_selected_sam_mask_rows(self) -> list[int]:
        return self.sam_interaction.effective_selected_sam_mask_rows()

    @staticmethod
    def _apply_mask_overlay(frame: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float):
        SamInteractionCoordinator._apply_mask_overlay(frame, mask, color, alpha)

    @staticmethod
    def _draw_mask_contour(frame: np.ndarray, mask: np.ndarray, color: np.ndarray, thickness: int = 2):
        SamInteractionCoordinator._draw_mask_contour(frame, mask, color, thickness=thickness)

    @staticmethod
    def _sam_overlay_color_for_row(row: int) -> np.ndarray:
        return SamInteractionCoordinator._sam_overlay_color_for_row(row)

    def _show_clean_input_frame(self):
        self.sam_interaction.show_clean_input_frame()

    def _render_input_preview(self):
        self.sam_interaction.render_input_preview()

    @staticmethod
    def _to_qimage(frame_rgb: np.ndarray) -> QImage:
        h, w, c = frame_rgb.shape
        bpl = w * c
        return QImage(frame_rgb.tobytes(), w, h, bpl, QImage.Format.Format_RGB888).copy()

    def _set_label_pixmap(self, label, pixmap: QPixmap):
        if label is self.ui.input_video_label:
            self._input_source_pixmap = pixmap
        elif label is self.ui.output_video_label:
            self._output_source_pixmap = pixmap
        # Use a cheap (nearest) resampler while the playback timer is running:
        # SmoothTransformation on full-resolution HD/4K frames dominates the
        # per-tick cost and causes visible stutter. Quality is restored on stop.
        play_timer = getattr(self, "play_timer", None)
        if play_timer is not None and play_timer.isActive():
            transform_mode = Qt.TransformationMode.FastTransformation
        else:
            transform_mode = Qt.TransformationMode.SmoothTransformation
        scaled = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            transform_mode,
        )
        label.setPixmap(scaled)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def on_generate_mask(self):
        self.sam_interaction.on_generate_mask()

    def _set_sam_base_controls_enabled(self, enabled: bool) -> None:
        self.sam_interaction.set_sam_base_controls_enabled(enabled)

    def _sync_sam_interaction_buttons(
        self,
        *,
        live_sam2: bool,
        controls_enabled: bool,
        point_mode: str | None = None,
        sync_live_toggle: bool = False,
    ) -> None:
        self.sam_interaction.sync_sam_interaction_buttons(
            live_sam2=live_sam2,
            controls_enabled=controls_enabled,
            point_mode=point_mode,
            sync_live_toggle=sync_live_toggle,
        )

    def _set_sam_controls_busy(self, active: bool):
        self.sam_interaction.set_sam_controls_busy(active)

    def on_clear_points(self):
        self.sam_interaction.on_clear_points()

    def on_add_mask(self):
        self.sam_interaction.on_add_mask()

    def on_remove_masks(self):
        self.sam_interaction.on_remove_masks()

    def _refresh_mask_list(self):
        self.sam_interaction.refresh_mask_list()

    def on_load_mask_file(self):
        self.sam_interaction.on_load_mask_file()

    def _resolve_mask_path_for_processing(self):
        return self.sam_interaction.resolve_mask_path_for_processing()

    def _is_cloud_processing_active(self) -> bool:
        controller = getattr(self, "_cloud_inference", None)
        return bool(controller is not None and controller.is_active)

    def _ensure_matting_orchestrator(self) -> MattingOrchestrator:
        orchestrator = getattr(self, "_matting_orchestrator", None)
        if orchestrator is None:
            orchestrator = MattingOrchestrator(self)
            self._matting_orchestrator = orchestrator
        return orchestrator

    def _build_cloud_graph_payload(self) -> tuple[dict | None, str]:
        """Return supported cloud graph payload or (None, reason).

        Supported cloud topologies:
        - source -> gvm -> export
        - source -> gvm -> corridorkey -> export
        """
        graph_dialog = getattr(self, "_node_graph_dialog", None)
        if graph_dialog is None or not hasattr(graph_dialog, "export_graph_preset"):
            return None, ""
        if hasattr(graph_dialog, "graph_is_empty") and graph_dialog.graph_is_empty():
            return None, ""

        preset = graph_dialog.export_graph_preset()
        if not isinstance(preset, dict):
            return None, "invalid graph preset"

        nodes_raw = preset.get("nodes")
        edges_raw = preset.get("connections")
        if not isinstance(nodes_raw, list) or not isinstance(edges_raw, list):
            return None, "invalid graph payload"

        graph_nodes: list[dict] = []
        node_type_by_id: dict[str, str] = {}
        enabled_types: set[str] = set()
        allowed_types = {"source", "gvm", "corridorkey", "export"}

        for node in nodes_raw:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or "").strip()
            node_type = str(node.get("type") or "").strip().lower()
            if not node_id or not node_type:
                continue

            props = dict(node.get("properties") or {})
            enabled = bool(props.get("enabled", True))
            if enabled and node_type not in allowed_types:
                return None, f"unsupported node type: {node_type}"

            node_payload = {
                "id": node_id,
                "type": node_type,
                "title": str(node.get("title") or ""),
                "properties": props,
                "enabled": enabled,
            }
            graph_nodes.append(node_payload)
            node_type_by_id[node_id] = node_type
            if enabled:
                enabled_types.add(node_type)

        if not graph_nodes:
            return None, ""

        has_corridorkey = "corridorkey" in enabled_types
        required_types = {"source", "gvm", "export"} | ({"corridorkey"} if has_corridorkey else set())
        if not required_types.issubset(enabled_types):
            required_list = ", ".join(sorted(required_types))
            return None, f"graph must include enabled nodes: {required_list}"

        graph_edges: list[dict] = []
        has_source_to_gvm = False
        has_source_to_corridorkey = False
        has_gvm_to_corridorkey = False
        has_gvm_to_export = False
        has_corridorkey_to_export = False

        for edge in edges_raw:
            if not isinstance(edge, dict):
                continue
            src_id = str(edge.get("src") or "").strip()
            dst_id = str(edge.get("dst") or "").strip()
            src_port = str(edge.get("src_port") or "out").strip().lower()
            dst_port = str(edge.get("dst_port") or "").strip().lower()
            if not src_id or not dst_id:
                continue

            src_type = node_type_by_id.get(src_id, "")
            dst_type = node_type_by_id.get(dst_id, "")
            if src_type not in allowed_types or dst_type not in allowed_types:
                continue

            if src_type == "source" and dst_type == "gvm" and dst_port == "image":
                has_source_to_gvm = True
            if src_type == "source" and dst_type == "corridorkey" and dst_port == "image":
                has_source_to_corridorkey = True
            if src_type == "gvm" and dst_type == "corridorkey" and src_port == "alpha" and dst_port == "alphahint":
                has_gvm_to_corridorkey = True
            if src_type == "gvm" and dst_type == "export" and src_port == "alpha" and dst_port == "in":
                has_gvm_to_export = True
            if (
                src_type == "corridorkey"
                and dst_type == "export"
                and src_port in {"alpha", "fg", "comp", "processed"}
                and dst_port == "in"
            ):
                has_corridorkey_to_export = True

            graph_edges.append(
                {
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "src_port": src_port,
                    "dst_port": dst_port,
                }
            )

        if has_corridorkey:
            if not has_source_to_gvm:
                return None, "missing edge source(out)->gvm(image)"
            if not has_source_to_corridorkey:
                return None, "missing edge source(out)->corridorkey(image)"
            if not has_gvm_to_corridorkey:
                return None, "missing edge gvm(alpha)->corridorkey(alphahint)"
            if not has_corridorkey_to_export:
                return None, "missing edge corridorkey(alpha|fg|comp|processed)->export(in)"
        else:
            if not has_source_to_gvm or not has_gvm_to_export:
                return None, "supported cloud chain is source(out)->gvm(image)->export(in from alpha)"

        return {"nodes": graph_nodes, "edges": graph_edges}, ""

    def start_processing(self):
        if self._try_auto_propagate_sam2_before_processing():
            return
        if bool(get_cloud_setting("cloud/enabled")):
            self._start_cloud_processing()
            return
        self._ensure_matting_orchestrator().start_processing()

    def _start_cloud_processing(self) -> None:
        if self._is_cloud_processing_active() or self.matting.is_active:
            return
        if not self.input_path:
            QMessageBox.warning(self, self._tr("status_error"), self._tr("err_no_file"))
            return
        if not self.is_video_input:
            QMessageBox.information(self, self._tr("info_title"), self._tr("cloud_worker_err_video_only"))
            return

        graph_payload, graph_error = self._build_cloud_graph_payload()
        if graph_error:
            QMessageBox.information(
                self,
                self._tr("info_title"),
                f"{self._tr('cloud_worker_err_graph_not_supported')}\n\n{graph_error}",
            )
            return

        mask_path = ""
        if graph_payload is None:
            mask_path = self._resolve_mask_path_for_processing()
            if not mask_path:
                QMessageBox.warning(self, self._tr("status_error"), self._tr("err_no_mask_for_run"))
                return

        raw_api_port = get_cloud_setting("cloud/api_port")
        try:
            api_port = int(raw_api_port) if raw_api_port not in {None, ""} else 8080
        except (TypeError, ValueError):
            api_port = 8080

        cloud_settings = {
            "instance_id": str(get_cloud_setting("cloud/instance_id") or "").strip(),
            "api_host": str(get_cloud_setting("cloud/api_host") or "").strip(),
            "region": str(get_cloud_setting("cloud/region") or "eu-west-1").strip(),
            "aws_profile": str(get_cloud_setting("cloud/aws_profile") or "").strip(),
            "api_port": api_port,
        }
        if not cloud_settings["instance_id"] and not cloud_settings["api_host"]:
            QMessageBox.warning(self, self._tr("status_error"), self._tr("cloud_log_no_iid"))
            return

        controller = getattr(self, "_cloud_inference", None)
        if controller is None:
            QMessageBox.warning(self, self._tr("status_error"), self._tr("cloud_worker_err_unavailable"))
            return

        source = Path(self.input_path)
        effective_start, effective_end = self._resolve_effective_video_frame_bounds()
        source_is_sequence = len(resolve_numbered_image_sequence(source)) > 1
        if graph_payload is not None:
            params = {
                "node_graph": graph_payload,
                "frame_start": int(effective_start),
                "frame_end": int(effective_end),
                "source_is_sequence": bool(source_is_sequence),
                "source_fps": float(self.video_fps) if not source_is_sequence else 1.0,
            }
        else:
            params = {
                "n_warmup": self._spin_value("spin_warmup_frames", 10),
                "r_erode": self._spin_value("spin_erode_kernel", 0),
                "r_dilate": self._spin_value("spin_dilate_kernel", 0),
            }
        output_dir = build_keyflow_base_dir(source)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.ui.btn_run.setEnabled(False)
        self.ui.progress_bar.setValue(0)
        self._set_status(self._tr("cloud_worker_status_connecting"))
        self._refresh_stop_button_state()

        controller.start(
            video_path=str(source),
            mask_path=str(mask_path),
            output_dir=str(output_dir),
            cloud_settings=cloud_settings,
            params=params,
        )

    # ── start_processing helpers ──

    def _try_graph_inference_run(self, output_dir: Path, start_frame: int, end_frame: int) -> bool:
        return self._ensure_matting_orchestrator().try_graph_inference_run(output_dir, start_frame, end_frame)

    def _collect_write_targets(self, output_dir: Path) -> tuple[dict | None, dict | None, list[dict], bool]:
        return self._ensure_matting_orchestrator().collect_write_targets(output_dir)

    def _restore_write_outputs_from_disk(self) -> None:
        self._ensure_matting_orchestrator().restore_write_outputs_from_disk()

    def _execute_passthrough_targets(
        self, passthrough_targets: list[dict], mask_path: str, output_dir: Path,
    ) -> tuple[str, str]:
        return self._ensure_matting_orchestrator().execute_passthrough_targets(
            passthrough_targets, mask_path, output_dir
        )

    def _save_sam2_outputs_to_connected_write_nodes(self) -> tuple[int, int]:
        return self._ensure_matting_orchestrator().save_sam_outputs_to_connected_write_nodes()

    def _start_matting_run(self, mask_path: str, output_dir: Path, config: dict) -> None:
        self._ensure_matting_orchestrator().start_matting_run(mask_path, output_dir, config)

    def cancel_processing(self):
        if self._is_cloud_processing_active():
            controller = getattr(self, "_cloud_inference", None)
            if controller is not None:
                controller.cancel()
            self._set_status(self._tr("status_cancel"))
            return
        self._ensure_matting_orchestrator().cancel_processing()

    # ── Matting signal handlers ──

    def _on_matting_stage_progress(self, percent: int, status_text: str):
        self._ensure_matting_orchestrator().on_matting_stage_progress(percent, status_text)

    def _on_node_frame_progress(self, node_type: str, current: int, total: int):
        self._ensure_matting_orchestrator().on_node_frame_progress(node_type, current, total)

    def _on_matting_frame_progress(self, current: int, total: int):
        self._ensure_matting_orchestrator().on_matting_frame_progress(current, total)

    def _on_matting_frame_preview(self, foreground_rgb, alpha_rgb, _frame_index: int) -> None:
        self._ensure_matting_orchestrator().on_matting_frame_preview(foreground_rgb, alpha_rgb, _frame_index)

    def _on_graph_stream_preview(self, write_node_id: str, preview_frame, _frame_index: int) -> None:
        self._ensure_matting_orchestrator().on_graph_stream_preview(write_node_id, preview_frame, _frame_index)

    def _on_matting_finished(self, result: dict):
        self._ensure_matting_orchestrator().on_matting_finished(result)

    def _on_matting_error(self, error_message: str):
        self._ensure_matting_orchestrator().on_matting_error(error_message)

    def _on_matting_busy_changed(self, busy: bool):
        self._ensure_matting_orchestrator().on_matting_busy_changed(busy)

    def _on_matting_log_message(self, message: str):
        self._ensure_matting_orchestrator().on_matting_log_message(message)

    def _on_corridorkey_mode_resolved(self, requested_mode: str, effective_mode: str, _reason_key: str) -> None:
        self._ensure_matting_orchestrator().on_corridorkey_mode_resolved(requested_mode, effective_mode, _reason_key)

    def _on_cloud_stage_progress(self, percent: int, status_text: str) -> None:
        text = str(status_text or "").strip() or self._tr("status_start")
        # Stages where the server is busy with setup and no frame-by-frame progress is
        # available — show a pulsing (indeterminate) bar so the user knows work is ongoing.
        _indeterminate = {
            self._tr(k) for k in (
                "cloud_worker_status_processing",
                "cloud_stage_loading_graph",
                "cloud_stage_extracting_sequence",
                "cloud_stage_extracting_frames",
                "cloud_stage_loading_model",
                "cloud_stage_gvm_inference",
                "cloud_stage_loading_video",
                "cloud_stage_loading_mask",
            )
        }
        if text in _indeterminate:
            if self.ui.progress_bar.maximum() != 0:
                self.ui.progress_bar.setRange(0, 0)
        else:
            if self.ui.progress_bar.maximum() == 0:
                self.ui.progress_bar.setRange(0, 100)
            self.ui.progress_bar.setValue(max(0, min(100, int(percent))))
        self._set_status(text)
        # Mirror frame progress into the GVM node body
        dlg = getattr(self, "_node_graph_dialog", None)
        if dlg is not None:
            m = re.search(r"GVM\s+(~?\d+\s*/\s*\d+\s*(frames?|batches?))", text, re.IGNORECASE)
            dlg.set_gvm_cloud_status(f"☁ GVM {m.group(1)}" if m else None)

    def _on_cloud_log_message(self, message: str) -> None:
        text = str(message or "").strip()
        if text:
            logger.info("%s", text)

    def _on_cloud_processing_finished(self, result: dict) -> None:
        self.ui.btn_run.setEnabled(True)
        self._refresh_stop_button_state()
        dlg = getattr(self, "_node_graph_dialog", None)
        if dlg is not None:
            dlg.set_gvm_cloud_status(None)

        self.ui.progress_bar.setRange(0, 100)
        if bool(result.get("cancelled")):
            self.ui.progress_bar.setValue(0)
            self._set_status(self._tr("status_stopped"))
            return

        result_path = str(result.get("result_path") or "").strip()
        if not result_path or not os.path.exists(result_path):
            self.ui.progress_bar.setValue(0)
            self._set_status(self._tr("status_error"))
            QMessageBox.warning(self, self._tr("inference_error_title"), self._tr("cloud_worker_err_no_result"))
            return

        self.last_output_dir = str(Path(result_path).parent)
        write_results = result.get("write_results") or []
        if isinstance(write_results, list) and write_results:
            for item in write_results:
                write_node_id = str((item or {}).get("write_node_id") or "").strip()
                write_result_path = str((item or {}).get("result_path") or "").strip()
                if not write_node_id or not write_result_path:
                    continue
                try:
                    self._apply_export_preview_path(write_node_id, write_result_path)
                except Exception as exc:
                    logger.warning("Cloud Write path apply failed for node %s: %s", write_node_id, exc)

                if dlg is not None and hasattr(dlg, "set_write_runtime_preview_for_node"):
                    try:
                        preview_frame = self._load_preview_image_or_video_frame(write_result_path)
                        preview_qimage = self._to_qimage(preview_frame) if preview_frame is not None else None
                        dlg.set_write_runtime_preview_for_node(write_node_id, preview_qimage)
                    except Exception as exc:
                        logger.warning("Cloud Write thumbnail update failed for node %s: %s", write_node_id, exc)
        else:
            write_node_id = str(result.get("write_node_id") or "").strip()
            if write_node_id:
                try:
                    self._apply_export_preview_path(write_node_id, result_path)
                except Exception as exc:
                    logger.warning("Cloud Write path apply failed for node %s: %s", write_node_id, exc)

                if dlg is not None and hasattr(dlg, "set_write_runtime_preview_for_node"):
                    try:
                        preview_frame = self._load_preview_image_or_video_frame(result_path)
                        preview_qimage = self._to_qimage(preview_frame) if preview_frame is not None else None
                        dlg.set_write_runtime_preview_for_node(write_node_id, preview_qimage)
                    except Exception as exc:
                        logger.warning("Cloud Write thumbnail update failed for node %s: %s", write_node_id, exc)

        self._show_output_preview("", result_path)
        self.ui.progress_bar.setValue(100)
        self._set_status(self._tr("status_done"))
        self._play_completion_sound()

    def _on_cloud_processing_error(self, error_message: str) -> None:
        self.ui.btn_run.setEnabled(True)
        self._refresh_stop_button_state()
        dlg = getattr(self, "_node_graph_dialog", None)
        if dlg is not None:
            dlg.set_gvm_cloud_status(None)
        self.ui.progress_bar.setRange(0, 100)
        self.ui.progress_bar.setValue(0)
        self._set_status(self._tr("status_error"))
        logger.error("Cloud inference error: %s", error_message)
        QMessageBox.warning(self, self._tr("inference_error_title"), error_message)

    def _on_cloud_busy_changed(self, busy: bool) -> None:
        if busy:
            self.ui.btn_run.setEnabled(False)
        elif not self._media_loading_active and not self.sam2.generation_active and not self.matting.is_active:
            self.ui.btn_run.setEnabled(True)
        self._refresh_stop_button_state()

    def _show_output_preview(self, fgr_path: str, alpha_path: str):
        self._ensure_viewer_preview()._show_output_preview(fgr_path, alpha_path)

    def _reset_viewer_preview(self, clear_outputs: bool = True):
        self._ensure_viewer_preview()._reset_viewer_preview(clear_outputs=clear_outputs)

    def _resolve_output_image_sequence(self, path: str) -> list[str]:
        return self._ensure_viewer_preview()._resolve_output_image_sequence(path)

    def _load_preview_image_or_video_frame(self, path: str):
        return self._ensure_viewer_preview()._load_preview_image_or_video_frame(path)

    def _render_output_preview_for_index(self, idx: int):
        self._ensure_viewer_preview()._render_output_preview_for_index(idx)

    def _on_start_frame_changed(self):
        """When start frame changes, let controller decide what to update."""
        start_frame = self.ui.spin_start_frame.value()
        end_frame = self.ui.spin_end_frame.value()
        frame_count = self.ui.spin_num_frames.value()
        
        state = FrameRangeController.on_start_frame_changed(start_frame, end_frame, frame_count)
        
        if state.updated_end_frame is not None:
            self.ui.spin_end_frame.blockSignals(True)
            self.ui.spin_end_frame.setValue(state.updated_end_frame)
            self.ui.spin_end_frame.blockSignals(False)
        self._update_graph_sam_tracking_tooltips()

    def _on_end_frame_changed(self):
        """When end frame changes, let controller decide what to update."""
        start_frame = self.ui.spin_start_frame.value()
        end_frame = self.ui.spin_end_frame.value()
        frame_count = self.ui.spin_num_frames.value()
        
        state = FrameRangeController.on_end_frame_changed(start_frame, end_frame, frame_count)
        
        if state.updated_frame_count is not None:
            self.ui.spin_num_frames.blockSignals(True)
            self.ui.spin_num_frames.setValue(state.updated_frame_count)
            self.ui.spin_num_frames.blockSignals(False)
        self._update_graph_sam_tracking_tooltips()

    def _on_frame_count_changed(self):
        """When frame count changes, let controller decide what to update."""
        start_frame = self.ui.spin_start_frame.value()
        end_frame = self.ui.spin_end_frame.value()
        frame_count = self.ui.spin_num_frames.value()
        
        state = FrameRangeController.on_count_changed(start_frame, end_frame, frame_count)
        
        if state.updated_end_frame is not None:
            self.ui.spin_end_frame.blockSignals(True)
            self.ui.spin_end_frame.setValue(state.updated_end_frame)
            self.ui.spin_end_frame.blockSignals(False)
        self._update_graph_sam_tracking_tooltips()

    def on_frame_slider_changed(self, value: int):
        if not self.all_frames:
            return
        value = max(0, min(len(self.all_frames) - 1, value))
        self.current_frame_index = value
        self.current_frame = self.all_frames[value]
        if self._node_graph_dialog is not None and hasattr(self._node_graph_dialog, "update_active_read_node_preview_frame"):
            self._node_graph_dialog.update_active_read_node_preview_frame(value)
        self._render_input_preview()
        self._render_output_preview_for_index(value)
        if hasattr(self, "sam_interaction") and self.sam_interaction is not None:
            self.sam_interaction.on_frame_changed_show_sam_mask(value)
        else:
            if self._active_node_type in {"sam2"}:
                self._show_mask_preview_on_output(self.sam2.state.mask_for_frame(value))
        self._update_frame_info()

    def on_first_frame(self):
        frame_slider = self._transport_slider()
        if self.all_frames and frame_slider is not None:
            frame_slider.setValue(0)

    def on_prev_frame(self):
        frame_slider = self._transport_slider()
        if self.all_frames and frame_slider is not None:
            frame_slider.setValue(max(0, frame_slider.value() - 1))

    def on_next_frame(self):
        frame_slider = self._transport_slider()
        if self.all_frames and frame_slider is not None:
            frame_slider.setValue(min(len(self.all_frames) - 1, frame_slider.value() + 1))

    def on_last_frame(self):
        frame_slider = self._transport_slider()
        if self.all_frames and frame_slider is not None:
            frame_slider.setValue(len(self.all_frames) - 1)

    def on_play_toggled(self, checked: bool):
        play_reverse_btn = getattr(self.ui, "btn_play_reverse", None)
        if checked:
            if play_reverse_btn is not None and play_reverse_btn.isChecked():
                with QSignalBlocker(play_reverse_btn):
                    play_reverse_btn.setChecked(False)
            self._start_playback(direction=1)
            return

        if play_reverse_btn is not None and play_reverse_btn.isChecked():
            return
        self._stop_playback()

    def on_play_reverse_toggled(self, checked: bool) -> None:
        play_btn = self._transport_button("btn_play")
        if checked:
            if play_btn is not None and play_btn.isChecked():
                with QSignalBlocker(play_btn):
                    play_btn.setChecked(False)
            self._start_playback(direction=-1)
            return

        if play_btn is not None and play_btn.isChecked():
            return
        self._stop_playback()

    def on_play_loop_toggled(self, checked: bool) -> None:
        self._play_loop_enabled = bool(checked)

    def _start_playback(self, direction: int) -> None:
        if not self.all_frames or len(self.all_frames) <= 1:
            self._stop_playback()
            return
        self._play_direction = -1 if direction < 0 else 1
        interval = max(15, int(1000 / max(1.0, self.video_fps)))
        # Anchor playback to wall-clock so we can drop frames if rendering
        # falls behind, instead of accumulating drift on each tick.
        import time as _time
        self._play_started_monotonic = _time.monotonic()
        frame_slider = self._transport_slider()
        self._play_start_index = (
            frame_slider.value() if frame_slider is not None else self.current_frame_index
        )
        self.play_timer.start(interval)
        play_btn = self._transport_button("btn_play")
        if play_btn is not None:
            play_btn.setIcon(self._icon_pause)

    def _stop_playback(self) -> None:
        was_active = self.play_timer.isActive()
        self.play_timer.stop()
        play_btn = self._transport_button("btn_play")
        if play_btn is not None:
            play_btn.setIcon(self._icon_play)
            if play_btn.isChecked():
                with QSignalBlocker(play_btn):
                    play_btn.setChecked(False)
        play_reverse_btn = getattr(self.ui, "btn_play_reverse", None)
        if play_reverse_btn is not None and play_reverse_btn.isChecked():
            with QSignalBlocker(play_reverse_btn):
                play_reverse_btn.setChecked(False)
        # Repaint the current frame at full SmoothTransformation quality now
        # that the cheap playback resampler is no longer in effect.
        if was_active and self.all_frames:
            try:
                self._render_input_preview()
                self._render_output_preview_for_index(self.current_frame_index)
            except Exception:
                pass

    def _play_next_frame(self):
        if not self.all_frames:
            self._stop_playback()
            return

        frame_slider = self._transport_slider()
        if frame_slider is None:
            self._stop_playback()
            return

        total = len(self.all_frames)
        # Wall-clock target frame: if rendering fell behind, jump straight to
        # where we *should* be instead of stepping ±1 and accumulating lag.
        import time as _time
        elapsed = max(0.0, _time.monotonic() - self._play_started_monotonic)
        steps = int(elapsed * max(1.0, self.video_fps))
        # Always advance at least one frame per tick so the UI keeps moving
        # even if the wall-clock target hasn't crossed the next frame yet.
        cur = frame_slider.value()
        if self._play_direction < 0:
            target_idx = self._play_start_index - steps
            if target_idx >= cur:
                target_idx = cur - 1
        else:
            target_idx = self._play_start_index + steps
            if target_idx <= cur:
                target_idx = cur + 1

        if target_idx < 0 or target_idx >= total:
            if self._play_loop_enabled and total > 1:
                # Re-anchor at the wrap point so timing stays correct.
                target_idx = total - 1 if self._play_direction < 0 else 0
                self._play_started_monotonic = _time.monotonic()
                self._play_start_index = target_idx
            else:
                self._stop_playback()
                return
        frame_slider.setValue(target_idx)

    def _update_frame_info(self):
        if not self.all_frames:
            self.ui.lbl_frame_info.setText("0000 / 0000")
            return
        current = self.current_frame_index
        total = len(self.all_frames) - 1
        self.ui.lbl_frame_info.setText(f"{current:04d} / {total:04d}")

    def open_output_folder(self):
        if not self.last_output_dir or not os.path.exists(self.last_output_dir):
            QMessageBox.information(self, self._tr("info_title"), self._tr("info_no_output_folder"))
            return
        subprocess.run(["open", self.last_output_dir], check=False)

    def closeEvent(self, event):
        self._cleanup_media_loader()
        cloud_controller = getattr(self, "_cloud_inference", None)
        if cloud_controller is not None:
            cloud_controller.shutdown()
        self.matting.shutdown()
        self.sam2.shutdown()

        self._reset_viewer_preview()

        super().closeEvent(event)


def _install_excepthook() -> None:
    """Route unhandled exceptions (incl. Qt slot errors) to the root logger."""
    import traceback as _tb

    def _hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logging.getLogger(__name__).error(
            "Unhandled exception: %s",
            "".join(_tb.format_exception(exc_type, exc_value, exc_tb)),
        )

    sys.excepthook = _hook


def main():
    """Запуск приложения"""
    app = QApplication(sys.argv)
    app.setApplicationName("KeyFlowStudio")
    app.setOrganizationName("KeyFlowStudio")

    _configure_app_logging()
    _install_excepthook()

    lock_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.TempLocation)
    if not lock_dir:
        lock_dir = tempfile.gettempdir()
    instance_lock = QLockFile(str(Path(lock_dir) / "KeyFlowStudio.lock"))
    instance_lock.setStaleLockTime(0)
    if not instance_lock.tryLock(100):
        print("KeyFlow Studio: another instance is already running; exiting duplicate process.")
        return
    app._instance_lock = instance_lock

    # Установка стилей приложения
    app.setStyle("Fusion")
    app.setStyleSheet(
        "QComboBox {"
        "  background-color: #1b212b;"
        "  color: #dce6f5;"
        "  border: 1px solid #2a3444;"
        "  border-radius: 7px;"
        "  padding: 3px 4px 3px 6px;"
        "  font-size: 12px;"
        "}"
        "QComboBox:hover {"
        "  border: 1px solid #3d5a7a;"
        "  background-color: #222b38;"
        "}"
        "QComboBox:focus {"
        "  border: 1px solid #4b7597;"
        "}"
        "QComboBox::drop-down {"
        "  subcontrol-origin: padding;"
        "  subcontrol-position: top right;"
        "  min-width: 0px;"
        "  max-width: 0px;"
        "  width: 0px;"
        "  margin: 0px;"
        "  padding: 0px;"
        "  border: none;"
        "}"
        "QComboBox::down-arrow { image: none; width: 0px; height: 0px; }"
        "QComboBox QAbstractItemView {"
        "  background-color: #151c27;"
        "  color: #dce6f5;"
        "  border: 1px solid #2a3444;"
        "  border-radius: 6px;"
        "  selection-background-color: #0f3c57;"
        "  selection-color: #f7fbff;"
        "  outline: none;"
        "}"
    )

    # Создание главного окна
    window = MainWindow()
    window.show()

    sys.exit(app.exec())

def _configure_app_logging() -> None:
    """Configure runtime logging for terminal + rotating file."""
    root = logging.getLogger()
    if getattr(root, "_keyflow_logging_configured", False):
        return

    log_dir_raw = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    log_dir = Path(log_dir_raw) if log_dir_raw else (Path.home() / ".keyflow_studio")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "keyflow_studio.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    root._keyflow_logging_configured = True

    # Suppress noisy botocore/boto3 credential-discovery INFO messages
    logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
    logging.getLogger("botocore.loaders").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)

    logger.info("Logging initialized: %s", log_path)


if __name__ == "__main__":
    main()
