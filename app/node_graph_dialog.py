"""Node graph dialog scaffold for visual pipeline editing."""

from __future__ import annotations

import copy
import html
import logging
import math
import os
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

from PySide6.QtCore import QPoint, QRectF, Qt, Signal, QTimer, QUrl
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPainterPath, QPainterPathStroker, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QFormLayout,
    QGraphicsItemGroup,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QLineEdit,
    QScrollArea,
    QTextBrowser,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QStyle,
)

from app.node_graph.load_media_properties_panel import LoadMediaPropertiesPanel
from app.node_graph.matting_properties_panel import MattingPropertiesPanel
from app.node_graph.sam_properties_panel import SamPropertiesPanel
from app.node_graph.sam3_properties_panel import Sam3PropertiesPanel
from app.node_graph.write_properties_panel import WritePropertiesPanel
from app.node_graph.birefnet_properties_panel import BiRefNetPropertiesPanel
from app.node_graph.gvm_properties_panel import GVMPropertiesPanel
from app.node_graph.chromakey_properties_panel import ChromaKeyPropertiesPanel
from app.node_graph.corridorkey_properties_panel import CorridorKeyPropertiesPanel
from app.node_graph.merge_properties_panel import MergePropertiesPanel
from app.node_graph.diagnostics import (
    diagnostic_primary_node_id,
    format_graph_diagnostics_html,
    format_graph_diagnostics_summary,
    format_graph_diagnostics_text,
)
from app.node_graph.engine import NodeGraphEngine
from app.node_graph.matting_properties_panel import PRESET_LABEL_KEYS
from app.node_graph.models import GraphEdge, GraphNode
from app.node_graph.specs import get_node_spec, list_node_specs, PORT_COLORS, DEFAULT_PORT_COLORS, EDGE_COLORS, DEFAULT_EDGE_COLOR
from app.node_graph.rules import get_registry
from app.settings import get_app_settings
from app.utils.media import is_numbered_image_sequence, load_rgb_image, read_media_dimensions, resolve_numbered_image_sequence
from app.utils.write_output import VIDEO_OUTPUT_FORMATS, resolve_write_output_format
from app.utils.write_paths import build_graph_write_output_dir, build_keyflow_base_dir, normalize_write_stream_name
from app.shortcuts import handle_node_graph_hotkeys


class NodeQuickAddDialog(QDialog):
    """Quick add popup (Tab) similar to Nuke node search."""

    def __init__(self, translate: Callable[[str], str], parent=None) -> None:
        super().__init__(parent)
        self._tr = translate
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setFixedWidth(192)
        self.selected_key: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self.search_edit = QLineEdit(self)
        self.search_edit.textChanged.connect(self._apply_filter)
        self.search_edit.returnPressed.connect(self._accept_current)
        root.addWidget(self.search_edit)

        self.list_widget = QListWidget(self)
        self.list_widget.setMaximumHeight(210)
        self.list_widget.itemDoubleClicked.connect(lambda _item: self._accept_current())
        root.addWidget(self.list_widget)

        self.setStyleSheet(
            "NodeQuickAddDialog {"
            "  background-color: #131a23;"
            "  border: 1px solid #294158;"
            "  border-radius: 8px;"
            "}"
            "QLineEdit {"
            "  background-color: #171d27;"
            "  color: #e8edf5;"
            "  border: 1px solid #2a3444;"
            "  border-radius: 6px;"
            "  padding: 4px 8px;"
            "}"
            "QLineEdit:focus {"
            "  border: 1px solid #43c7ff;"
            "}"
            "QListWidget {"
            "  background-color: #0f141c;"
            "  color: #e8edf5;"
            "  border: 1px solid #1e2a38;"
            "  border-radius: 6px;"
            "  outline: none;"
            "}"
            "QListWidget::item {"
            "  padding: 3px 6px;"
            "}"
            "QListWidget::item:selected {"
            "  background-color: #0f3c57;"
            "  color: #f7fbff;"
            "}"
            "QListWidget::item:hover {"
            "  background-color: #1a2a3c;"
            "}"
            "QScrollBar:vertical {"
            "  background: rgba(9, 15, 22, 0.88);"
            "  width: 10px;"
            "  margin: 6px 2px 6px 2px;"
            "  border: none;"
            "  border-radius: 5px;"
            "}"
            "QScrollBar::handle:vertical {"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #34506a, stop:1 #4b7597);"
            "  min-height: 28px;"
            "  border-radius: 5px;"
            "}"
            "QScrollBar::handle:vertical:hover {"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3d6485, stop:1 #59a7d8);"
            "}"
            "QScrollBar::handle:vertical:pressed {"
            "  background: #6bc8ff;"
            "}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {"
            "  height: 0px;"
            "  background: transparent;"
            "}"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {"
            "  background: transparent;"
            "}"
        )

        self._node_items: list[tuple[str, str, str, str, str]] = [
            (spec.key, spec.title, spec.subtitle, spec.title_i18n_key, spec.subtitle_i18n_key)
            for spec in list_node_specs()
        ]
        self._recent_keys: list[str] = []
        self._section_recent = "Recent"
        self._section_all = "All Nodes"
        self.retranslate_ui()

    def set_recent_keys(self, keys: list[str]) -> None:
        self._recent_keys = [str(k) for k in keys if isinstance(k, str)]
        self._apply_filter()

    def retranslate_ui(self) -> None:
        self.search_edit.setPlaceholderText(self._tr("node_graph_tab_search_placeholder"))
        self._section_recent = self._tr("node_graph_recent_section")
        self._section_all = self._tr("node_graph_all_nodes_section")
        self._apply_filter()

    def _add_section_item(self, text: str) -> None:
        section = QListWidgetItem(text)
        section.setFlags(Qt.ItemFlag.NoItemFlags)
        section.setForeground(QColor("#8ca0ba"))
        self.list_widget.addItem(section)

    def _apply_filter(self) -> None:
        query = self.search_edit.text().strip().lower()
        self.list_widget.clear()

        def _sort_key(item: tuple[str, str, str, str, str]):
            key = item[0]
            try:
                return (0, self._recent_keys.index(key))
            except ValueError:
                return (1, 999)

        filtered: list[tuple[str, str, str, str, str]] = []
        for key, fallback_title, subtitle, tr_key, subtitle_tr_key in sorted(self._node_items, key=_sort_key):
            title = self._tr(tr_key)
            translated_subtitle = self._tr(subtitle_tr_key) if subtitle_tr_key else subtitle
            haystack = f"{title} {fallback_title} {translated_subtitle} {subtitle}".lower()
            if query and query not in haystack:
                continue
            filtered.append((key, fallback_title, subtitle, tr_key, subtitle_tr_key))

        recent_items = [item for item in filtered if item[0] in self._recent_keys]
        other_items = [item for item in filtered if item[0] not in self._recent_keys]

        if recent_items:
            self._add_section_item(self._section_recent)

        for key, _fallback_title, _subtitle, tr_key, _subtitle_tr_key in recent_items:
            title = self._tr(tr_key)
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.list_widget.addItem(item)

        if other_items and recent_items:
            self._add_section_item(self._section_all)

        for key, _fallback_title, _subtitle, tr_key, _subtitle_tr_key in other_items:
            title = self._tr(tr_key)
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.list_widget.addItem(item)

        if self.list_widget.count() > 0:
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                if item.flags() & Qt.ItemFlag.ItemIsSelectable:
                    self.list_widget.setCurrentRow(i)
                    break

    def _accept_current(self) -> None:
        item = self.list_widget.currentItem()
        if item is None or not (item.flags() & Qt.ItemFlag.ItemIsSelectable):
            return
        self.selected_key = str(item.data(Qt.ItemDataRole.UserRole))
        self.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        if event.key() in {Qt.Key.Key_Down, Qt.Key.Key_Up, Qt.Key.Key_PageDown, Qt.Key.Key_PageUp}:
            self.list_widget.keyPressEvent(event)
            return
        super().keyPressEvent(event)


class GraphDiagnosticsDialog(QDialog):
    """Detached diagnostics window for node-graph validation issues."""

    anchor_clicked = Signal(QUrl)
    strict_mode_toggled = Signal(bool)

    def __init__(self, translate: Callable[[str], str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self._tr = translate
        self._summary_text = ""
        self._html_text = ""

        self.resize(620, 420)
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.title_label = QLabel(self)
        self.title_label.setStyleSheet("font-size: 13px; font-weight: 700; color: #dfe8f4;")
        root.addWidget(self.title_label)

        self.summary_label = QLabel(self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color: #f0b35c; font-weight: 600;")
        root.addWidget(self.summary_label)

        self.strict_mode_checkbox = QCheckBox(self)
        self.strict_mode_checkbox.toggled.connect(self.strict_mode_toggled.emit)
        root.addWidget(self.strict_mode_checkbox)

        self.body = QTextBrowser(self)
        self.body.setReadOnly(True)
        self.body.setOpenLinks(False)
        self.body.setOpenExternalLinks(False)
        self.body.setFrameShape(QFrame.Shape.NoFrame)
        self.body.setStyleSheet(
            "background: rgba(11, 18, 26, 0.92);"
            "border: 1px solid #23354a;"
            "border-radius: 8px;"
            "padding: 8px;"
            "color: #9fb2c8;"
        )
        self.body.anchorClicked.connect(self.anchor_clicked.emit)
        root.addWidget(self.body, 1)

        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        title = self._tr("graph_diagnostics_title")
        self.setWindowTitle(title)
        self.title_label.setText(title)
        self.summary_label.setText(self._summary_text or self._tr("graph_diagnostics_empty"))
        self.strict_mode_checkbox.setText(self._tr("graph_diagnostics_strict_required_inputs"))
        self.body.setHtml(self._html_text or f'<span style="color:#8ca0ba;">{html.escape(self._tr("graph_diagnostics_empty"))}</span>')

    def set_translator(self, translate: Callable[[str], str]) -> None:
        self._tr = translate
        self.retranslate_ui()

    def set_diagnostics_content(self, summary: str, html_text: str) -> None:
        self._summary_text = str(summary or "").strip()
        self._html_text = str(html_text or "").strip()
        self.summary_label.setText(self._summary_text or self._tr("graph_diagnostics_empty"))
        self.body.setHtml(self._html_text or f'<span style="color:#8ca0ba;">{html.escape(self._tr("graph_diagnostics_empty"))}</span>')

    def set_strict_mode(self, enabled: bool) -> None:
        self.strict_mode_checkbox.blockSignals(True)
        self.strict_mode_checkbox.setChecked(bool(enabled))
        self.strict_mode_checkbox.blockSignals(False)


class NodeGraphView(QGraphicsView):
    """Graphics view with grid, zoom, and middle-mouse panning."""

    def __init__(self, parent=None, graph_owner=None) -> None:
        super().__init__(parent)
        self.graph_owner = graph_owner
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._panning = False
        self._space_down = False
        self._pan_start = QPoint()

    def _update_idle_cursor(self) -> None:
        if self._space_down and not self._panning:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif not self._panning:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _start_panning(self, pos: QPoint) -> None:
        self._panning = True
        self._pan_start = pos
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _stop_panning(self) -> None:
        self._panning = False
        self._update_idle_cursor()

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawBackground(painter, rect)
        painter.fillRect(rect, QColor("#111722"))

        minor_pen = QPen(QColor("#1a2433"))
        major_pen = QPen(QColor("#27364a"))

        grid_minor = 20
        grid_major = 100

        left = int(rect.left()) - (int(rect.left()) % grid_minor)
        top = int(rect.top()) - (int(rect.top()) % grid_minor)

        x = left
        while x < int(rect.right()):
            painter.setPen(major_pen if x % grid_major == 0 else minor_pen)
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
            x += grid_minor

        y = top
        while y < int(rect.bottom()):
            painter.setPen(major_pen if y % grid_major == 0 else minor_pen)
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)
            y += grid_minor

    def wheelEvent(self, event) -> None:
        scale_up = 1.15
        scale_down = 1.0 / scale_up
        cur = self.transform().m11()

        if event.angleDelta().y() > 0:
            if cur < 2.8:
                self.scale(scale_up, scale_up)
        else:
            if cur > 0.35:
                self.scale(scale_down, scale_down)

    def mousePressEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if event.button() == Qt.MouseButton.MiddleButton:
            self._start_panning(event.pos())
            event.accept()
            return
        if self._space_down and event.button() == Qt.MouseButton.LeftButton:
            self._start_panning(event.pos())
            event.accept()
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.graph_owner is not None
            and self.graph_owner.try_start_connection_drag(self.mapToScene(event.pos()))
        ):
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        if self.graph_owner is not None and self.graph_owner.is_connection_drag_active():
            self.graph_owner.update_connection_drag(self.mapToScene(event.pos()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.graph_owner is not None
            and self.graph_owner.is_connection_drag_active()
        ):
            self.graph_owner.finish_connection_drag(self.mapToScene(event.pos()))
            event.accept()
            return
        if event.button() in {Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton} and self._panning:
            self._stop_panning()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.graph_owner is not None:
            scene_pos = self.mapToScene(event.pos())
            # Double-click on a Read-like node (Load/Source/Alpha) -> open file browser
            item = self.scene().itemAt(scene_pos, self.transform())
            if isinstance(item, NodeItem) and item.node_type in {"load", "source", "alpha"}:
                self.graph_owner._active_node = item
                self.graph_owner._load_properties_to_ui(item)
                self.graph_owner._browse_load_media()
                event.accept()
                return
            edge = self.graph_owner._find_connection_at(scene_pos)
            if edge is not None:
                self.graph_owner._remove_connection(edge)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        if self.graph_owner is not None and self.graph_owner.handle_graph_key_event(event):
            return
        if event.key() == Qt.Key.Key_Space:
            self._space_down = True
            self._update_idle_cursor()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._space_down = False
            if self._panning:
                self._stop_panning()
            self._update_idle_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:
        if self.graph_owner is not None:
            self.graph_owner.show_context_menu(event.globalPos(), self.mapToScene(event.pos()))
            event.accept()
            return
        super().contextMenuEvent(event)

    def focusNextPrevChild(self, next: bool) -> bool:
        # Tab is reserved for quick node add in the graph, so prevent Qt focus traversal here.
        if self.graph_owner is not None and next:
            self.graph_owner.open_quick_add_popup()
            return True
        return super().focusNextPrevChild(next)


class NodeItem(QGraphicsItem):
    """Simple visual node item."""

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        width: float = 220,
        height: float = 120,
        node_type: str = "generic",
    ):
        super().__init__()
        self.title = title
        self.subtitle = subtitle
        self.w = width
        self.h = height
        self.node_type = node_type
        self.properties = self._default_properties(node_type)
        self.properties.setdefault("custom_title", False)
        self._preview_pixmap: QPixmap | None = None
        self._annotation_lines: list[tuple[str, str]] = []  # [(text, color), ...]
        spec = get_node_spec(node_type)
        self.input_ports: list = list(spec.inputs) if spec else []
        self.output_ports: list = list(spec.outputs) if spec else []
        self._highlighted_input_port: str | None = None
        self._highlighted_output_port: str | None = None
        self._diagnostic_input_ports: set[str] = set()
        self._diagnostic_output_ports: set[str] = set()
        self._diagnostic_count: int = 0
        self._connections: list[ConnectionItem] = []
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

    @staticmethod
    def _default_properties(node_type: str) -> dict:
        spec = get_node_spec(node_type)
        if spec is None:
            return {"enabled": True, "note": ""}
        return copy.deepcopy(spec.default_properties)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.w, self.h)

    def _port_y(self, index: int, total: int) -> float:
        if total <= 1:
            return self.h / 2
        header_h = 30.0
        usable = self.h - header_h
        return header_h + usable * (index + 1) / (total + 1)

    def input_anchor(self, port_name: str = ""):
        if not self.input_ports:
            return self.mapToScene(0, self.h / 2)
        if not port_name:
            port_name = self.input_ports[0].name
        for i, port in enumerate(self.input_ports):
            if port.name == port_name:
                return self.mapToScene(0, self._port_y(i, len(self.input_ports)))
        return self.mapToScene(0, self.h / 2)

    def output_anchor(self, port_name: str = ""):
        if not self.output_ports:
            return self.mapToScene(self.w, self.h / 2)
        if not port_name:
            port_name = self.output_ports[0].name
        for i, port in enumerate(self.output_ports):
            if port.name == port_name:
                return self.mapToScene(self.w, self._port_y(i, len(self.output_ports)))
        return self.mapToScene(self.w, self.h / 2)

    def add_connection(self, edge: "ConnectionItem") -> None:
        if edge not in self._connections:
            self._connections.append(edge)

    def remove_connection(self, edge: "ConnectionItem") -> None:
        if edge in self._connections:
            self._connections.remove(edge)

    def set_input_highlight(self, port_name: str | None) -> None:
        if self._highlighted_input_port != port_name:
            self._highlighted_input_port = port_name
            self.update()

    def set_output_highlight(self, port_name: str | None) -> None:
        if self._highlighted_output_port != port_name:
            self._highlighted_output_port = port_name
            self.update()

    def set_diagnostic_state(
        self,
        *,
        input_ports: set[str] | None = None,
        output_ports: set[str] | None = None,
        count: int = 0,
    ) -> None:
        next_input = set(input_ports or set())
        next_output = set(output_ports or set())
        next_count = max(0, int(count))
        if (
            self._diagnostic_input_ports != next_input
            or self._diagnostic_output_ports != next_output
            or self._diagnostic_count != next_count
        ):
            self._diagnostic_input_ports = next_input
            self._diagnostic_output_ports = next_output
            self._diagnostic_count = next_count
            self.update()

    def set_annotation_lines(self, lines: list[tuple[str, str]]) -> None:
        if self._annotation_lines != lines:
            self._annotation_lines = list(lines)
            self.update()

    def set_preview_pixmap(self, pixmap: QPixmap | None) -> None:
        self._preview_pixmap = pixmap if pixmap is not None and not pixmap.isNull() else None
        self.update()

    def _is_input_port_connected(self, port_name: str) -> bool:
        return any(edge.dst is self and edge.dst_port == port_name for edge in self._connections)

    def _is_output_port_connected(self, port_name: str) -> bool:
        return any(edge.src is self and edge.src_port == port_name for edge in self._connections)

    def _resolve_port_colors(
        self,
        data_type: str,
        *,
        highlighted: bool,
        diagnostic: bool,
        connected: bool,
    ) -> tuple[QColor, QColor, QColor, float]:
        colors = PORT_COLORS.get(data_type, DEFAULT_PORT_COLORS)
        if highlighted:
            border_c = QColor(colors["border_hl"])
            fill_c = QColor(colors["fill_hl"])
            label_c = border_c
        elif diagnostic:
            border_c = QColor("#ff8d8d")
            fill_c = QColor("#6a2026")
            label_c = border_c
        elif connected:
            border_c = QColor(colors["border_hl"]).lighter(108)
            fill_c = QColor(colors["fill_hl"]).lighter(108)
            label_c = border_c
        else:
            border_c = QColor(colors["border"]).darker(130)
            fill_c = QColor(colors["fill"]).darker(130)
            label_c = QColor("#617a92")
        pen_w = 1.8 if highlighted or diagnostic or connected else 1.1
        return border_c, fill_c, label_c, pen_w

    def _tr(self, key: str, default: str = "") -> str:
        scene = self.scene()
        if scene is not None:
            for view in scene.views():
                graph_owner = getattr(view, "graph_owner", None)
                translate = getattr(graph_owner, "_tr", None)
                if callable(translate):
                    try:
                        return str(translate(key))
                    except Exception:
                        break
        return default or key

    def _get_node_colors(self) -> dict[str, str]:
        """Get color scheme for this node based on type and properties.
        
        Source: Red scheme
        Load: Gray if alpha media, default if video
        Others: Default scheme
        """
        # Default colors
        default_colors = {
            "body": "#1a2535",
            "header": "#24364d",
            "border": "#2f435d",
            "border_selected": "#5ec4ff",
        }
        
        if self.node_type == "source":
            # Source node: Red scheme
            return {
                "body": "#3a1b1f",
                "header": "#5a262d",
                "border": "#7a3a45",
                "border_selected": "#ff6b7a",
            }

        if self.node_type == "alpha":
            # Alpha node: Gray scheme
            return {
                "body": "#202020",
                "header": "#323232",
                "border": "#4a4a4a",
                "border_selected": "#9a9a9a",
            }
        
        if self.node_type == "load":
            # Load node: Check media type
            media_type = str(self.properties.get("media_type", "video")).strip().lower()
            if media_type in {"alpha", "mask"}:
                # Alpha/Mask media: Gray scheme
                return {
                    "body": "#1f1f1f",
                    "header": "#333333",
                    "border": "#4a4a4a",
                    "border_selected": "#888888",
                }
            # Video: Default scheme
            return default_colors
        
        if self.node_type in ("birefnet", "sam2", "sam3"):
            # BiRefNet / SAM Mask: mask-generator — gray scheme matching alpha/mask visual language
            return {
                "body": "#1e2020",
                "header": "#2e3232",
                "border": "#484e4e",
                "border_selected": "#a3a3a3",
            }

        if self.node_type == "export":
            # Export node: Green scheme
            return {
                "body": "#1f3a2a",
                "header": "#2a5a3a",
                "border": "#4a7a5a",
                "border_selected": "#4ade80",
            }
        
        # All other nodes: Default scheme
        return default_colors

    def itemChange(self, change, value):
        if change in {
            QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged,
        }:
            for edge in list(self._connections):
                edge.update_path()
        return super().itemChange(change, value)

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        node_rect = self.boundingRect()
        colors = self._get_node_colors()
        has_diagnostics = self._diagnostic_count > 0

        if has_diagnostics and self.isSelected():
            painter.setPen(QPen(QColor("#ff8d8d"), 2.4))
        elif has_diagnostics:
            painter.setPen(QPen(QColor("#d84d57"), 1.9))
        elif self.isSelected():
            painter.setPen(QPen(QColor(colors["border_selected"]), 2.2))
        else:
            painter.setPen(QPen(QColor(colors["border"]), 1.4))

        painter.setBrush(QColor(colors["body"]))
        painter.drawRoundedRect(node_rect, 10, 10)

        header_rect = QRectF(0, 0, self.w, 30)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(colors["header"]))
        painter.drawRoundedRect(header_rect, 10, 10)
        painter.drawRect(0, 15, int(self.w), 15)
        header_right_pad = 10.0

        corridorkey_runtime_prefix = self._tr("corridorkey_annotation_mode_runtime")
        corridorkey_runtime_fallback_prefix = self._tr("corridorkey_annotation_mode_runtime_fallback")
        corridorkey_mode_config_prefix = self._tr("corridorkey_annotation_mode_config")
        header_mode_text = ""
        header_mode_color = "#f4b35a"
        if self.node_type == "corridorkey":
            # Show configured mode in header by default, then override with runtime effective mode if available.
            mode_cfg = str(self.properties.get("alpha_hint_mode", "auto")).strip().lower()
            mode_key = {
                "auto": "corridorkey_alpha_hint_mode_auto",
                "batch": "corridorkey_alpha_hint_mode_batch",
                "staged": "corridorkey_alpha_hint_mode_staged",
            }.get(mode_cfg, "corridorkey_alpha_hint_mode_auto")
            header_mode_text = self._tr(mode_key)
            for txt, clr in self._annotation_lines:
                raw = str(txt or "").strip()
                if not raw:
                    continue
                if raw.startswith(f"{corridorkey_runtime_prefix}:") or raw.startswith(f"{corridorkey_runtime_fallback_prefix}:"):
                    mode_part = raw.split(":", 1)[1].strip()
                    if "->" in mode_part:
                        mode_part = mode_part.split("->")[-1].strip()
                    if mode_part:
                        header_mode_text = mode_part
                    break
        sam_warning_badge_visible = (
            self.node_type == "sam2"
            and not bool(self.properties.get("mask_items", []))
            and not bool(self.properties.get("current_mask_ready", False))
        )

        painter.setPen(QPen(QColor("#dfe8f4")))
        if header_mode_text:
            base_font = painter.font()
            mode_font = QPainter.font(painter)
            mode_pt = mode_font.pointSizeF()
            if mode_pt > 0:
                mode_font.setPointSizeF(max(7.0, mode_pt * 0.84))
            else:
                mode_font.setPixelSize(max(9, int(mode_font.pixelSize() * 0.84)))
            mode_fm = painter.fontMetrics() if mode_font == base_font else None
            if mode_fm is None:
                painter.setFont(mode_font)
                mode_fm = painter.fontMetrics()
                painter.setFont(base_font)
            backend_badge_reserve = 23.0 if sam_warning_badge_visible else 0.0
            header_mode_w = mode_fm.horizontalAdvance(header_mode_text) + 14 + backend_badge_reserve
            left_title_w = max(60.0, self.w - header_mode_w - 24.0)
            painter.drawText(
                QRectF(10, 6, left_title_w, 20),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self.title,
            )
            painter.setPen(QPen(QColor(header_mode_color)))
            painter.setFont(mode_font)
            backend_text_w = max(40.0, self.w - 10 - header_right_pad - backend_badge_reserve)
            painter.drawText(
                QRectF(10, 6, backend_text_w, 20),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                header_mode_text,
            )
            painter.setFont(base_font)
        else:
            painter.drawText(
                QRectF(10, 6, self.w - 20, 20),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self.title,
            )

        painter.setPen(QPen(QColor("#9fb2c8")))
        subtitle_bottom = self.h - 45
        if self.node_type in {"load", "source", "alpha"} and self._preview_pixmap is not None:
            subtitle_bottom = 16
        if self.node_type not in {"load", "alpha", "export"} and not self._annotation_lines:
            painter.drawText(
                QRectF(10, 42, self.w - 20, subtitle_bottom),
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                self.subtitle,
            )

        # ── Annotation (Nuke-style centered info) ──
        if self._annotation_lines and self.node_type not in {"load", "alpha"}:
            font = painter.font()
            pt = font.pointSizeF()
            if pt > 0:
                font.setPointSizeF(pt * 0.85)
            else:
                px = font.pixelSize()
                font.setPixelSize(max(1, int(px * 0.85)))
            painter.setFont(font)
            fm = painter.fontMetrics()
            line_h = fm.lineSpacing()
            body_top = 34.0
            body_h = self.h - 44.0
            annotation_lines = self._annotation_lines
            if self.node_type == "corridorkey":
                annotation_lines = [
                    (txt, clr)
                    for txt, clr in self._annotation_lines
                    if not str(txt or "").strip().startswith(f"{corridorkey_mode_config_prefix}:")
                    if not str(txt or "").strip().startswith(f"{corridorkey_runtime_prefix}:")
                    and not str(txt or "").strip().startswith(f"{corridorkey_runtime_fallback_prefix}:")
                ]

            total_h = line_h * len(annotation_lines)
            y0 = body_top + (body_h - total_h) / 2 + fm.ascent()
            for idx, (txt, clr) in enumerate(annotation_lines):
                painter.setPen(QPen(QColor(clr)))
                tw = fm.horizontalAdvance(txt)
                painter.drawText(int(12 + (self.w - 24 - tw) / 2), int(y0 + idx * line_h), txt)

        if self.node_type in {"load", "source", "alpha", "export"}:
            if self.node_type == "export":
                # Keep the left side free for input port label, preview lives on the right.
                preview_rect = QRectF(self.w * 0.46, 38, self.w * 0.48, self.h - 48)
                preview_border = QColor(colors["border_selected"]).darker(115)
                preview_fill = QColor(colors["body"]).darker(150)
                preview_text = QColor(colors["border_selected"]).lighter(120)
            elif self.node_type in {"source", "alpha"}:
                preview_rect = QRectF(10, 40, self.w - 20, self.h - 50)
                preview_border = QColor(colors["border_selected"]).darker(120)
                preview_fill = QColor(colors["body"]).darker(165)
                preview_text = QColor(colors["border_selected"]).lighter(115)
            else:
                preview_rect = QRectF(10, 40, self.w - 20, self.h - 50)
                preview_border = QColor("#34506b")
                preview_fill = QColor("#111a27")
                preview_text = QColor("#6f88a3")
            painter.setPen(QPen(preview_border, 1.0))
            painter.setBrush(preview_fill)
            painter.drawRoundedRect(preview_rect, 6, 6)

            if self._preview_pixmap is not None:
                scaled = self._preview_pixmap.scaled(
                    int(preview_rect.width() - 4),
                    int(preview_rect.height() - 4),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                x = preview_rect.x() + (preview_rect.width() - scaled.width()) / 2
                y = preview_rect.y() + (preview_rect.height() - scaled.height()) / 2
                painter.drawPixmap(int(x), int(y), scaled)
            else:
                painter.setPen(QPen(preview_text))
                painter.drawText(preview_rect, Qt.AlignmentFlag.AlignCenter, self._tr("node_graph_no_preview", "No Preview"))

        # ── Draw typed ports ──
        port_r = 7
        for i, port in enumerate(self.input_ports):
            y = self._port_y(i, len(self.input_ports))
            input_port_disabled = bool(
                self.node_type == "merge"
                and port.name == "mask"
                and not bool(self.properties.get("mask_enabled", True))
            )
            highlighted = self._highlighted_input_port == port.name
            diagnostic = port.name in self._diagnostic_input_ports
            connected = self._is_input_port_connected(port.name)
            border_c, fill_c, label_c, pen_w = self._resolve_port_colors(
                str(port.data_type or ""),
                highlighted=highlighted,
                diagnostic=diagnostic,
                connected=connected,
            )
            if input_port_disabled:
                border_c = QColor("#4f5d6c")
                fill_c = QColor("#2a333d")
                label_c = QColor("#5f6d7c")
                pen_w = 1.0
                highlighted = False
            painter.setPen(QPen(border_c, pen_w))
            painter.setBrush(fill_c)
            painter.drawEllipse(-port_r, int(y) - port_r, port_r * 2, port_r * 2)
            lbl = port.label or port.name
            if self.node_type == "export" and port.name == "in":
                input_label = self._tr("node_graph_port_input", "Input")
                incoming = next((e for e in self._connections if e.dst is self and e.dst_port == "in"), None)
                incoming_data_type = ""
                if incoming is not None:
                    src_spec = get_node_spec(incoming.src.node_type)
                    if src_spec is not None:
                        src_port = next((p for p in src_spec.outputs if p.name == incoming.src_port), None)
                        if src_port is not None:
                            incoming_data_type = str(src_port.data_type or "").strip().lower()
                if incoming_data_type == "alpha":
                    alpha_colors = PORT_COLORS.get("alpha", DEFAULT_PORT_COLORS)
                    border_c = QColor(alpha_colors["border_hl"]).lighter(108)
                    fill_c = QColor(alpha_colors["fill_hl"]).lighter(108)
                    label_c = border_c
                    painter.setPen(QPen(border_c, 1.6))
                    painter.setBrush(fill_c)
                    painter.drawEllipse(-port_r, int(y) - port_r, port_r * 2, port_r * 2)
                if incoming is not None:
                    port_name = str(incoming.src_port or "").strip()
                    if port_name:
                        lbl_top = f"{input_label}:"
                        src_spec = get_node_spec(incoming.src.node_type)
                        lbl_bot = port_name.capitalize()
                        if src_spec is not None:
                            _sp = next((p for p in src_spec.outputs if p.name == port_name), None)
                            if _sp is not None and _sp.label:
                                lbl_bot = _sp.label
                    else:
                        lbl_top = input_label
                        lbl_bot = ""
                else:
                    lbl_top = input_label
                    lbl_bot = ""
                max_lbl_w = int(self.w * 0.44) - port_r - 4
                painter.setPen(QPen(label_c, 0.9))
                fm = painter.fontMetrics()
                lh = fm.lineSpacing()
                x0 = port_r + 4
                if lbl_bot:
                    y0 = int(y) - lh // 2 + fm.ascent() - lh // 2
                    top_txt = fm.elidedText(lbl_top, Qt.TextElideMode.ElideRight, max_lbl_w)
                    bot_txt = fm.elidedText(lbl_bot, Qt.TextElideMode.ElideRight, max_lbl_w)
                    painter.drawText(x0, y0, top_txt)
                    painter.drawText(x0, y0 + lh, bot_txt)
                else:
                    top_txt = fm.elidedText(lbl_top, Qt.TextElideMode.ElideRight, max_lbl_w)
                    painter.drawText(x0, int(y) + 4, top_txt)
                lbl = ""  # already drawn
            if lbl:
                painter.setPen(QPen(label_c, 0.9))
                if input_port_disabled:
                    painter.drawText(port_r + 4, int(y) + 4, f"{lbl} (off)")
                else:
                    painter.drawText(port_r + 4, int(y) + 4, lbl)

        for i, port in enumerate(self.output_ports):
            y = self._port_y(i, len(self.output_ports))
            highlighted = self._highlighted_output_port == port.name
            diagnostic = port.name in self._diagnostic_output_ports
            connected = self._is_output_port_connected(port.name)
            border_c, fill_c, label_c, pen_w = self._resolve_port_colors(
                str(port.data_type or ""),
                highlighted=highlighted,
                diagnostic=diagnostic,
                connected=connected,
            )
            painter.setPen(QPen(border_c, pen_w))
            painter.setBrush(fill_c)
            painter.drawEllipse(int(self.w) - port_r, int(y) - port_r, port_r * 2, port_r * 2)
            lbl = port.label or port.name
            if lbl:
                painter.setPen(QPen(label_c, 0.9))
                fm_w = painter.fontMetrics().horizontalAdvance(lbl)
                painter.drawText(int(self.w) - port_r - 4 - fm_w, int(y) + 4, lbl)

        # ── Warning badge for SAM node with no masks ──
        if sam_warning_badge_visible:
                badge_r = 9
                badge_cx = int(self.w - header_right_pad - badge_r)
                badge_cy = int(header_rect.center().y())
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#cc2222"))
                painter.drawEllipse(badge_cx - badge_r, badge_cy - badge_r, badge_r * 2, badge_r * 2)
                saved_font = painter.font()
                f = QPainter.font(painter)
                f.setBold(True)
                f.setPixelSize(max(10, badge_r + 3))
                painter.setFont(f)
                painter.setPen(QPen(QColor("#ffffff")))
                painter.drawText(
                    QRectF(badge_cx - badge_r, badge_cy - badge_r, badge_r * 2, badge_r * 2),
                    Qt.AlignmentFlag.AlignCenter,
                    "!",
                )
                painter.setFont(saved_font)

        if has_diagnostics:
            badge_text = str(self._diagnostic_count)
            badge_r = 10
            badge_cx = int(self.w - 12)
            badge_cy = int(self.h - 12)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#d84d57"))
            painter.drawEllipse(badge_cx - badge_r, badge_cy - badge_r, badge_r * 2, badge_r * 2)
            saved_font = painter.font()
            badge_font = QPainter.font(painter)
            badge_font.setBold(True)
            badge_font.setPixelSize(10)
            painter.setFont(badge_font)
            painter.setPen(QPen(QColor("#ffffff")))
            painter.drawText(
                QRectF(badge_cx - badge_r, badge_cy - badge_r, badge_r * 2, badge_r * 2),
                Qt.AlignmentFlag.AlignCenter,
                badge_text,
            )
            painter.setFont(saved_font)


class ConnectionItem(QGraphicsPathItem):
    """Curved edge between two nodes."""

    def __init__(self, src: NodeItem, dst: NodeItem, src_port: str = "out", dst_port: str = ""):
        super().__init__()
        self.src = src
        self.dst = dst
        self.src_port = src_port
        self.dst_port = dst_port
        self.src.add_connection(self)
        self.dst.add_connection(self)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setZValue(-1)
        self._normal_color = self._resolve_edge_color()
        self.setPen(QPen(self._normal_color, 2.0))
        self.update_path()

    def _resolve_edge_color(self) -> QColor:
        spec = get_node_spec(self.src.node_type)
        if spec:
            for port in spec.outputs:
                if port.name == self.src_port:
                    return QColor(EDGE_COLORS.get(port.data_type, DEFAULT_EDGE_COLOR))
        return QColor(DEFAULT_EDGE_COLOR)

    def shape(self) -> QPainterPath:
        stroker = QPainterPathStroker()
        stroker.setWidth(14.0)
        return stroker.createStroke(self.path())

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self.isSelected():
            self.setPen(QPen(self._normal_color.lighter(160), 3.0))
        else:
            self.setPen(QPen(self._normal_color, 2.0))
        super().paint(painter, option, widget)

    def detach(self) -> None:
        self.src.remove_connection(self)
        self.dst.remove_connection(self)

    def update_path(self) -> None:
        p1 = self.src.output_anchor(self.src_port)
        p2 = self.dst.input_anchor(self.dst_port)
        dx = max(80.0, abs(p2.x() - p1.x()) * 0.45)

        path = QPainterPath(p1)
        path.cubicTo(p1.x() + dx, p1.y(), p2.x() - dx, p2.y(), p2.x(), p2.y())
        self.setPath(path)


class NodeGraphDialog(QDialog):
    """Nuke/Houdini-like node graph scaffold."""

    read_media_selected = Signal(str, str)
    preview_request_changed = Signal(str, object)
    active_node_changed = Signal(str)          # node_type or "" when deselected
    sam_controls_changed = Signal(str, bool, str)
    sam_generate_requested = Signal()
    sam_clear_requested = Signal()
    sam_add_mask_requested = Signal()
    sam_remove_mask_requested = Signal()
    sam_load_mask_requested = Signal()
    sam_propagate_requested = Signal(str)  # direction: forward|backward
    sam_reprompt_requested = Signal()
    sam_session_reset_requested = Signal()
    sam_model_type_changed = Signal(str)   # emitted when user picks a different SAM model
    graph_diagnostics_changed = Signal(str, str, bool)

    def __init__(self, translate: Callable[[str], str], parent=None) -> None:
        super().__init__(parent)
        self._tr = translate
        self._settings = get_app_settings()
        self._connections: list[ConnectionItem] = []
        self._groups: list[QGraphicsItemGroup] = []
        self._drag_source_node: NodeItem | None = None
        self._drag_target_node: NodeItem | None = None
        self._drag_source_port: str | None = None
        self._drag_target_port: str | None = None
        self._drag_mode: str | None = None
        self._reconnect_backup: tuple[NodeItem, NodeItem, str, str] | None = None
        self._drag_temp_edge: QGraphicsPathItem | None = None
        self._quick_add_dialog = NodeQuickAddDialog(self._tr, self)
        self._quick_add_recent_keys = self._load_quick_add_recent()
        self._quick_add_dialog.set_recent_keys(self._quick_add_recent_keys)
        self._port_snap_radius = 56.0
        self._active_node: NodeItem | None = None
        self._updating_properties = False
        self._thumbnail_cache: dict[tuple[str, int], QPixmap] = {}  # (path, frame_index) → thumbnail
        self._media_info_cache: dict[str, str] = {}       # path → info text
        self._birefnet_runtime_percent: int | None = None
        self._birefnet_runtime_text: str = ""
        self._birefnet_frame_current: int | None = None
        self._birefnet_frame_total: int | None = None
        self._corridorkey_runtime_requested: str | None = None
        self._corridorkey_runtime_effective: str | None = None
        self._corridorkey_frame_current: int | None = None
        self._corridorkey_frame_total: int | None = None
        self._corridorkey_last_frame_ts: float | None = None
        self._corridorkey_frame_time_avg: float | None = None
        self._corridorkey_frame_time_count: int = 0
        self._matting_frame_current: int | None = None
        self._matting_frame_total: int | None = None
        self._sam_frame_current: int | None = None
        self._sam_frame_total: int | None = None
        self._gvm_frame_current: int | None = None
        self._gvm_frame_total: int | None = None
        self._graph_diagnostics_signature: str = ""
        self._graph_diagnostics_summary: str = ""
        self._graph_diagnostics_html: str = ""
        self._graph_diag_strict_required_inputs: bool = bool(
            self._settings.value("node_graph/diag_strict_required_inputs", True, type=bool)
        )
        self._diagnostics_dialog: GraphDiagnosticsDialog | None = None
        self._graph_diagnostics_refresh_timer = QTimer(self)
        self._graph_diagnostics_refresh_timer.setSingleShot(True)
        # 150 ms: coalesces rapid scene changes (drag, spinner) so validate_with_diagnostics
        # does not run on every mouse-move tick.
        self._graph_diagnostics_refresh_timer.setInterval(150)
        self._graph_diagnostics_refresh_timer.timeout.connect(self._refresh_graph_diagnostics)

        # Debounce timer for Merge quick-preview: the preview involves synchronous
        # disk I/O (cv2.VideoCapture open/seek/read) and numpy compositing; coalescing
        # rapid spinner changes avoids freezing the UI thread.
        self._merge_preview_pending_node: "NodeItem | None" = None
        self._merge_preview_debounce_timer = QTimer(self)
        self._merge_preview_debounce_timer.setSingleShot(True)
        self._merge_preview_debounce_timer.setInterval(80)
        self._merge_preview_debounce_timer.timeout.connect(self._flush_merge_preview_request)

        self.resize(1240, 760)
        self.setMinimumSize(920, 560)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.hint_label = QLabel(self)
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color: #9fb2c8; font-size: 12px;")
        root.addWidget(self.hint_label)

        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(-2400, -1600, 4800, 3200)
        # NodeItem.itemChange already calls edge.update_path() when a node moves,
        # so connecting _update_all_connections to scene.changed would redundantly
        # iterate ALL edges on every geometry event and create a feedback loop
        # (setPath marks the scene dirty → scene.changed again).
        self.scene.changed.connect(self._schedule_graph_diagnostics_refresh)
        self.scene.selectionChanged.connect(self._on_scene_selection_changed)

        self.view = NodeGraphView(self, graph_owner=self)
        self.view.setScene(self.scene)
        self.view.setStyleSheet(
            "QGraphicsView { border: 1px solid #23354a; border-radius: 8px; background: #0f141c; }"
            "QRubberBand { border: 0px; background: transparent; }"
            "QScrollBar:vertical {"
            "  background: rgba(9, 15, 22, 0.88);"
            "  width: 13px;"
            "  margin: 10px 2px 10px 2px;"
            "  border: none;"
            "  border-radius: 6px;"
            "}"
            "QScrollBar::handle:vertical {"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #34506a, stop:1 #4b7597);"
            "  min-height: 34px;"
            "  border-radius: 6px;"
            "}"
            "QScrollBar::handle:vertical:hover {"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3d6485, stop:1 #59a7d8);"
            "}"
            "QScrollBar::handle:vertical:pressed {"
            "  background: #6bc8ff;"
            "}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {"
            "  height: 0px;"
            "  background: transparent;"
            "}"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {"
            "  background: transparent;"
            "}"
            "QScrollBar:horizontal {"
            "  background: rgba(9, 15, 22, 0.88);"
            "  height: 13px;"
            "  margin: 2px 10px 2px 10px;"
            "  border: none;"
            "  border-radius: 6px;"
            "}"
            "QScrollBar::handle:horizontal {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #34506a, stop:1 #4b7597);"
            "  min-width: 34px;"
            "  border-radius: 6px;"
            "}"
            "QScrollBar::handle:horizontal:hover {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3d6485, stop:1 #59a7d8);"
            "}"
            "QScrollBar::handle:horizontal:pressed {"
            "  background: #6bc8ff;"
            "}"
            "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {"
            "  width: 0px;"
            "  background: transparent;"
            "}"
            "QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {"
            "  background: transparent;"
            "}"
            "QScrollBar::corner { background: transparent; }"
        )

        self.props_panel = QWidget(self)
        # Keep one strict properties sidebar width for all node types.
        self._props_panel_width = 340
        self._props_content_width = 324
        # Reserve space for the vertical scrollbar and frame paddings to avoid
        # horizontal overflow in any node properties panel.
        self._props_inner_width = self._props_content_width - 16
        self.props_panel.setFixedWidth(self._props_panel_width)
        props_layout = QVBoxLayout(self.props_panel)
        props_layout.setContentsMargins(8, 8, 8, 8)
        props_layout.setSpacing(8)

        props_header = QHBoxLayout()
        props_header.setContentsMargins(0, 0, 0, 0)
        props_header.setSpacing(6)

        self.props_title = QLabel(self.props_panel)
        self.props_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #dfe8f4;")
        props_header.addWidget(self.props_title)
        props_header.addStretch(1)

        self.btn_open_diagnostics = QPushButton(self.props_panel)
        self.btn_open_diagnostics.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_open_diagnostics.setFixedSize(24, 24)
        self.btn_open_diagnostics.setFlat(True)
        self.btn_open_diagnostics.clicked.connect(self._open_graph_diagnostics_dialog)
        props_header.addWidget(self.btn_open_diagnostics)
        props_layout.addLayout(props_header)

        self.props_empty = QLabel(self.props_panel)
        self.props_empty.setWordWrap(True)
        self.props_empty.setStyleSheet("color: #8ca0ba;")
        props_layout.addWidget(self.props_empty)

        self.props_form_wrap = QWidget(self.props_panel)
        self.props_form_wrap.setFixedWidth(self._props_inner_width)
        self.props_form = QFormLayout(self.props_form_wrap)
        self.props_form.setContentsMargins(0, 0, 0, 0)
        self.props_form.setSpacing(6)
        self.props_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.prop_name = QLineEdit(self.props_form_wrap)
        self.prop_divider = QFrame(self.props_form_wrap)
        self.prop_divider.setObjectName("node_props_divider")
        self.prop_divider.setFrameShape(QFrame.Shape.HLine)
        self.prop_divider.setFrameShadow(QFrame.Shadow.Plain)
        self.prop_divider.setFixedHeight(10)
        self.prop_divider.setStyleSheet(
            "QFrame#node_props_divider {"
            "  border: none;"
            "  border-top: 1px solid #2a3444;"
            "  background: transparent;"
            "  margin-top: 2px;"
            "  margin-bottom: 2px;"
            "}"
        )

        self.load_props_panel = LoadMediaPropertiesPanel(self._tr, self.props_form_wrap)
        self.sam_props_panel = SamPropertiesPanel(self._tr, self.props_form_wrap)
        self.sam3_props_panel = Sam3PropertiesPanel(self._tr, self.props_form_wrap)
        self.matting_props_panel = MattingPropertiesPanel(self._tr, self.props_form_wrap)
        self.birefnet_props_panel = BiRefNetPropertiesPanel(self._tr, self.props_form_wrap)
        self.gvm_props_panel = GVMPropertiesPanel(self._tr, self.props_form_wrap)
        self.chromakey_props_panel = ChromaKeyPropertiesPanel(self._tr, self.props_form_wrap)
        self.corridorkey_props_panel = CorridorKeyPropertiesPanel(self._tr, self.props_form_wrap)
        self.merge_props_panel = MergePropertiesPanel(self._tr, self.props_form_wrap)
        self.write_props_panel = WritePropertiesPanel(self._tr, self.props_form_wrap)

        for panel in (
            self.load_props_panel,
            self.sam_props_panel,
            self.sam3_props_panel,
            self.matting_props_panel,
            self.birefnet_props_panel,
            self.gvm_props_panel,
            self.chromakey_props_panel,
            self.corridorkey_props_panel,
            self.merge_props_panel,
            self.write_props_panel,
        ):
            panel.setMinimumWidth(0)
            panel.setMaximumWidth(self._props_inner_width)

        self.props_form.addRow("", self.prop_name)
        self.props_form.addRow("", self.prop_divider)
        self.props_form.addRow("", self.load_props_panel)
        self.props_form.addRow("", self.matting_props_panel)
        self.props_form.addRow("", self.birefnet_props_panel)
        self.props_form.addRow("", self.gvm_props_panel)
        self.props_form.addRow("", self.chromakey_props_panel)
        self.props_form.addRow("", self.corridorkey_props_panel)
        self.props_form.addRow("", self.merge_props_panel)
        self.props_form.addRow("", self.write_props_panel)
        self.props_form.addRow("", self.sam_props_panel)
        self.props_form.addRow("", self.sam3_props_panel)

        self.props_scroll = QScrollArea(self.props_panel)
        self.props_scroll.setObjectName("node_props_scroll")
        self.props_scroll.setWidgetResizable(True)
        self.props_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.props_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.props_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.props_scroll.setFixedWidth(self._props_content_width)
        self.props_scroll.setWidget(self.props_form_wrap)
        self.props_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
            "QScrollBar:vertical {"
            "  background: rgba(9, 15, 22, 0.88);"
            "  width: 13px;"
            "  margin: 10px 2px 10px 2px;"
            "  border: none;"
            "  border-radius: 6px;"
            "}"
            "QScrollBar::handle:vertical {"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #34506a, stop:1 #4b7597);"
            "  min-height: 34px;"
            "  border-radius: 6px;"
            "}"
            "QScrollBar::handle:vertical:hover {"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3d6485, stop:1 #59a7d8);"
            "}"
            "QScrollBar::handle:vertical:pressed {"
            "  background: #6bc8ff;"
            "}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {"
            "  height: 0px;"
            "  background: transparent;"
            "}"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {"
            "  background: transparent;"
            "}"
        )

        props_layout.addWidget(self.props_scroll)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.addWidget(self.view)
        self.splitter.addWidget(self.props_panel)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([900, 340])
        root.addWidget(self.splitter, 1)

        actions = QHBoxLayout()
        actions.addStretch(1)

        self.btn_reset = QPushButton(self)
        self.btn_reset.clicked.connect(self.reset_view)
        actions.addWidget(self.btn_reset)

        self.btn_close = QPushButton(self)
        self.btn_close.clicked.connect(self.accept)
        actions.addWidget(self.btn_close)

        root.addLayout(actions)

        self._type_widgets = {
            "source": [self.load_props_panel],
            "load": [self.load_props_panel],
            "alpha": [self.load_props_panel],
            "sam2": [self.sam_props_panel],
            "sam3": [self.sam3_props_panel],
            "matting": [self.matting_props_panel],
            "birefnet": [self.birefnet_props_panel],
            "gvm": [self.gvm_props_panel],
            "chromakey": [self.chromakey_props_panel],
            "corridorkey": [self.corridorkey_props_panel],
            "merge": [self.merge_props_panel],
            "export": [self.write_props_panel],
        }

        self._connect_property_signals()

        self.retranslate_ui()
        self._set_properties_enabled(False)
        self.reset_view()
        self._refresh_graph_diagnostics()
        self.view.setFocus()

    def _connect_property_signals(self) -> None:
        self.prop_name.textEdited.connect(self._apply_properties_from_ui)

        self.load_props_panel.media_type_combo.currentIndexChanged.connect(self._apply_properties_from_ui)
        self.load_props_panel.path_edit.textEdited.connect(self._apply_properties_from_ui)
        self.load_props_panel.browse_button.clicked.connect(self._browse_load_media)

        self.sam_props_panel.btn_positive.clicked.connect(self._on_sam_positive_clicked)
        self.sam_props_panel.btn_negative.clicked.connect(self._on_sam_negative_clicked)
        self.sam_props_panel.btn_live_sam2.toggled.connect(self._on_sam_live_button_toggled)
        self.sam_props_panel.btn_clear.clicked.connect(self._on_sam_clear_clicked)
        self.sam_props_panel.btn_generate_mask.clicked.connect(self._on_sam_generate_mask_clicked)
        self.sam_props_panel.btn_add_mask.clicked.connect(self._on_sam_add_mask_clicked)
        self.sam_props_panel.btn_remove_mask.clicked.connect(self._on_sam_remove_mask_clicked)
        self.sam_props_panel.btn_load_mask.clicked.connect(self._on_sam_load_mask_clicked)
        self.sam_props_panel.btn_sam2_propagate_forward.clicked.connect(
            lambda: self._on_sam_propagate_clicked("forward")
        )
        self.sam_props_panel.btn_sam2_propagate_backward.clicked.connect(
            lambda: self._on_sam_propagate_clicked("backward")
        )
        self.sam_props_panel.btn_sam2_reprompt.clicked.connect(self._on_sam_reprompt_clicked)
        self.sam_props_panel.btn_sam2_reset_session.clicked.connect(self._on_sam_reset_session_clicked)
        self.sam_props_panel.masks_list.itemSelectionChanged.connect(self._on_sam_mask_selection_changed)
        self.sam_props_panel.model_combo.currentIndexChanged.connect(self._on_sam_model_changed)

        self.sam3_props_panel.btn_generate_mask.clicked.connect(self._on_sam3_generate_mask_clicked)
        self.sam3_props_panel.model_combo.currentIndexChanged.connect(self._on_sam3_model_changed)
        self.sam3_props_panel.concept_edit.textChanged.connect(self._on_sam3_concept_changed)

        self.matting_props_panel.preset_combo.currentIndexChanged.connect(self._apply_properties_from_ui)
        self.matting_props_panel.erode_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.matting_props_panel.dilate_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.matting_props_panel.warmup_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.matting_props_panel.fg_background_combo.currentIndexChanged.connect(self._apply_properties_from_ui)

        self.birefnet_props_panel.usage_combo.currentIndexChanged.connect(self._apply_properties_from_ui)
        self.birefnet_props_panel.half_precision_check.toggled.connect(self._apply_properties_from_ui)
        self.birefnet_props_panel.dilate_radius_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.birefnet_props_panel.erode_radius_spin.valueChanged.connect(self._apply_properties_from_ui)

        self.gvm_props_panel.batch_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.gvm_props_panel.chunk_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.gvm_props_panel.overlap_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.gvm_props_panel.interp_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.gvm_props_panel.dilate_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.gvm_props_panel.noise_combo.currentIndexChanged.connect(self._apply_properties_from_ui)
        self.gvm_props_panel.clip_emb_check.stateChanged.connect(self._apply_properties_from_ui)

        self.chromakey_props_panel.hue_center_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.chromakey_props_panel.hue_range_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.chromakey_props_panel.saturation_min_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.chromakey_props_panel.value_min_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.chromakey_props_panel.blur_radius_spin.valueChanged.connect(self._apply_properties_from_ui)

        self.corridorkey_props_panel.alpha_hint_mode_combo.currentIndexChanged.connect(self._apply_properties_from_ui)
        self.corridorkey_props_panel.input_colorspace_combo.currentIndexChanged.connect(self._apply_properties_from_ui)
        self.corridorkey_props_panel.screen_color_combo.currentIndexChanged.connect(self._apply_properties_from_ui)
        self.corridorkey_props_panel.preset_combo.currentIndexChanged.connect(self._apply_properties_from_ui)
        self.corridorkey_props_panel.despill_strength_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.corridorkey_props_panel.despeckle_check.toggled.connect(self._apply_properties_from_ui)
        self.corridorkey_props_panel.despeckle_size_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.corridorkey_props_panel.matte_clip_black_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.corridorkey_props_panel.matte_clip_white_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.corridorkey_props_panel.matte_shrink_grow_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.corridorkey_props_panel.matte_edge_blur_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.corridorkey_props_panel.matte_gamma_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.corridorkey_props_panel.temporal_smoothing_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.corridorkey_props_panel.refiner_strength_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.corridorkey_props_panel.use_refiner_check.toggled.connect(self._apply_properties_from_ui)
        self.corridorkey_props_panel.checkpoint_status_changed.connect(self._refresh_corridorkey_annotations)

        self.merge_props_panel.mode_combo.currentIndexChanged.connect(self._apply_properties_from_ui)
        self.merge_props_panel.bbox_combo.currentIndexChanged.connect(self._apply_properties_from_ui)
        self.merge_props_panel.opacity_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.merge_props_panel.mix_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.merge_props_panel.mask_enabled_check.toggled.connect(self._apply_properties_from_ui)
        self.merge_props_panel.mask_channel_combo.currentIndexChanged.connect(self._apply_properties_from_ui)
        self.merge_props_panel.mask_inject_check.toggled.connect(self._apply_properties_from_ui)
        self.merge_props_panel.invert_mask_check.toggled.connect(self._apply_properties_from_ui)
        self.merge_props_panel.fringe_check.toggled.connect(self._apply_properties_from_ui)
        self.merge_props_panel.alpha_masking_check.toggled.connect(self._apply_properties_from_ui)

        self.write_props_panel.auto_output_check.toggled.connect(self._apply_properties_from_ui)
        self.write_props_panel.path_edit.textEdited.connect(self._apply_properties_from_ui)
        self.write_props_panel.browse_button.clicked.connect(self._browse_write_output)
        self.write_props_panel.file_name_edit.textEdited.connect(self._apply_properties_from_ui)
        self.write_props_panel.format_combo.currentIndexChanged.connect(self._apply_properties_from_ui)
        self.write_props_panel.codec_combo.currentIndexChanged.connect(self._apply_properties_from_ui)
        self.write_props_panel.quality_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.write_props_panel.preset_combo.currentIndexChanged.connect(self._apply_properties_from_ui)
        self.write_props_panel.png_bit_depth_combo.currentIndexChanged.connect(self._apply_properties_from_ui)
        self.write_props_panel.png_compression_spin.valueChanged.connect(self._apply_properties_from_ui)
        self.write_props_panel.png_embed_alpha_check.toggled.connect(self._apply_properties_from_ui)
        self.write_props_panel.jpg_quality_spin.valueChanged.connect(self._apply_properties_from_ui)

    def _set_properties_enabled(self, enabled: bool) -> None:
        self.props_form_wrap.setEnabled(enabled)
        self.props_scroll.setVisible(enabled)
        self.props_empty.setVisible(not enabled)

    def clear_active_selection(self) -> None:
        """Clear current node selection and keep the properties pane in placeholder mode."""
        self.scene.clearSelection()
        self._active_node = None
        self._set_properties_enabled(False)
        self.active_node_changed.emit("")
        self.preview_request_changed.emit("", None)

    def _on_sam_positive_clicked(self) -> None:
        if self._updating_properties or self._active_node is None or self._active_node.node_type not in {"sam2"}:
            return
        self._active_node.properties["point_mode"] = "positive"
        self.sam_props_panel.sync_controls(self._active_node.properties)
        self.sam_controls_changed.emit(
            "positive",
            bool(self._active_node.properties.get("live_sam2", False)),
            str(self._active_node.properties.get("backend", "sam2")),
        )

    def _on_sam_negative_clicked(self) -> None:
        if self._updating_properties or self._active_node is None or self._active_node.node_type not in {"sam2"}:
            return
        self._active_node.properties["point_mode"] = "negative"
        self.sam_props_panel.sync_controls(self._active_node.properties)
        self.sam_controls_changed.emit(
            "negative",
            bool(self._active_node.properties.get("live_sam2", False)),
            str(self._active_node.properties.get("backend", "sam2")),
        )

    def _on_sam_live_button_toggled(self, checked: bool) -> None:
        if self._updating_properties or self._active_node is None or self._active_node.node_type not in {"sam2"}:
            return
        self._active_node.properties["live_sam2"] = checked
        self.sam_props_panel.sync_controls(self._active_node.properties)
        self.sam_controls_changed.emit(
            str(self._active_node.properties.get("point_mode", "positive")),
            checked,
            str(self._active_node.properties.get("backend", "sam2")),
        )

    def _on_sam_clear_clicked(self) -> None:
        if self._active_node is None or self._active_node.node_type not in {"sam2"}:
            return
        self.sam_clear_requested.emit()

    def _on_sam_generate_mask_clicked(self) -> None:
        if self._active_node is None or self._active_node.node_type not in {"sam2"}:
            return
        self.sam_generate_requested.emit()

    def _on_sam_model_changed(self) -> None:
        if self._updating_properties or self._active_node is None or self._active_node.node_type not in {"sam2"}:
            return
        model_type = str(self.sam_props_panel.model_combo.currentData() or "vit_h")
        self._active_node.properties["model_type"] = model_type
        self._active_node.set_annotation_lines(self._build_sam_annotation(self._active_node.properties))
        self.sam_model_type_changed.emit(model_type)

    def _on_sam_add_mask_clicked(self) -> None:
        if self._active_node is None or self._active_node.node_type not in {"sam2"}:
            return
        self.sam_add_mask_requested.emit()

    def _on_sam_remove_mask_clicked(self) -> None:
        if self._active_node is None or self._active_node.node_type not in {"sam2"}:
            return
        self.sam_remove_mask_requested.emit()

    def _on_sam_load_mask_clicked(self) -> None:
        if self._active_node is None or self._active_node.node_type not in {"sam2"}:
            return
        self.sam_load_mask_requested.emit()

    def _on_sam_propagate_clicked(self, direction: str) -> None:
        logger.debug("[dialog] propagate clicked direction=%s active_node=%s",
                     direction,
                     getattr(self._active_node, 'node_type', None) if self._active_node else None)
        if self._active_node is None or self._active_node.node_type not in {"sam2"}:
            logger.warning("[dialog] propagate blocked: _active_node=%s", self._active_node)
            return
        direction_norm = str(direction or "").strip().lower()
        if direction_norm not in {"forward", "backward"}:
            return
        self.sam_propagate_requested.emit(direction_norm)

    def _on_sam_reprompt_clicked(self) -> None:
        if self._active_node is None or self._active_node.node_type not in {"sam2"}:
            return
        self.sam_reprompt_requested.emit()

    def _on_sam_reset_session_clicked(self) -> None:
        if self._active_node is None or self._active_node.node_type not in {"sam2"}:
            return
        self.sam_session_reset_requested.emit()

    def _on_sam_mask_selection_changed(self) -> None:
        if self._updating_properties or self._active_node is None or self._active_node.node_type not in {"sam2"}:
            return
        self._active_node.properties["selected_mask_rows"] = [idx.row() for idx in self.sam_props_panel.masks_list.selectedIndexes()]

    # ── SAM3 node handlers ───────────────────────────────────────────

    def _on_sam3_generate_mask_clicked(self) -> None:
        if self._active_node is None or self._active_node.node_type != "sam3":
            return
        self.sam3_props_panel.write_to_properties(self._active_node.properties)
        self._active_node.set_annotation_lines(
            self._build_sam_annotation(self._active_node.properties, node_type="sam3")
        )
        self.sam_generate_requested.emit()

    def _on_sam3_concept_changed(self) -> None:
        if self._updating_properties or self._active_node is None or self._active_node.node_type != "sam3":
            return
        self._active_node.properties["concept"] = self.sam3_props_panel.concept_edit.toPlainText().strip()
        self._active_node.set_annotation_lines(
            self._build_sam_annotation(self._active_node.properties, node_type="sam3")
        )

    def _on_sam3_model_changed(self) -> None:
        if self._updating_properties or self._active_node is None or self._active_node.node_type != "sam3":
            return
        model_type = str(self.sam3_props_panel.model_combo.currentData() or "sam3")
        self._active_node.properties["model_type"] = model_type
        self._active_node.set_annotation_lines(
            self._build_sam_annotation(self._active_node.properties, node_type="sam3")
        )
        self.sam_model_type_changed.emit(model_type)

    def _set_type_property_visibility(self, node_type: str) -> None:
        typed_widgets = set()
        for items in self._type_widgets.values():
            typed_widgets.update(items)

        for w in typed_widgets:
            w.setVisible(False)

        for w in self._type_widgets.get(node_type, []):
            w.setVisible(True)

        for i in range(self.props_form.rowCount()):
            label_item = self.props_form.itemAt(i, QFormLayout.ItemRole.LabelRole)
            field_item = self.props_form.itemAt(i, QFormLayout.ItemRole.FieldRole)
            if field_item is None or field_item.widget() is None:
                continue
            visible = field_item.widget().isVisible()
            if label_item is not None and label_item.widget() is not None:
                label_item.widget().setVisible(visible)

    def _on_scene_selection_changed(self) -> None:
        nodes = [item for item in self.scene.selectedItems() if isinstance(item, NodeItem)]
        if len(nodes) != 1:
            self._active_node = None
            self._set_properties_enabled(False)
            self.active_node_changed.emit("")
            return

        self._active_node = nodes[0]
        self._load_properties_to_ui(self._active_node)
        self._emit_preview_request(self._active_node)
        self.active_node_changed.emit(self._active_node.node_type)

    def _load_properties_to_ui(self, node: NodeItem) -> None:
        self._updating_properties = True
        try:
            self._set_properties_enabled(True)
            self._set_type_property_visibility(node.node_type)

            props = node.properties
            self.prop_name.setText(node.title)

            self.load_props_panel.load_from_properties(props)
            self._refresh_load_node_preview(node)
            self._update_load_media_info()

            self.sam_props_panel.load_from_properties(props)
            if node.node_type in {"sam2"}:
                backend = "sam2"
                props["backend"] = backend
                self.sam_model_type_changed.emit(str(props.get("model_type", "vit_h")))
                self.sam_controls_changed.emit(
                    str(props.get("point_mode", "positive")),
                    bool(props.get("live_sam2", False)),
                    backend,
                )
            self.sam3_props_panel.load_from_properties(props)
            if node.node_type == "sam3":
                backend = "sam3"
                props["backend"] = backend
                self.sam_model_type_changed.emit(str(props.get("model_type", "sam3")))
                self.sam_controls_changed.emit(
                    str(props.get("point_mode", "positive")),
                    bool(props.get("live_sam2", False)),
                    backend,
                )
            self.matting_props_panel.load_from_properties(props)
            self.birefnet_props_panel.load_from_properties(props)
            if node.node_type == "birefnet":
                node.set_annotation_lines(
                    self._build_birefnet_annotation(
                        node.properties,
                        self._birefnet_runtime_percent,
                        self._birefnet_runtime_text,
                    )
                )
            self.gvm_props_panel.load_from_properties(props)
            self.chromakey_props_panel.load_from_properties(props)
            if node.node_type == "corridorkey":
                node.set_annotation_lines(self._build_corridorkey_annotation(node.properties))
            self.corridorkey_props_panel.load_from_properties(props)
            if node.node_type == "corridorkey" and not bool(node.properties.get("custom_title", False)):
                node.title = self._corridorkey_auto_title(node.properties)
                self.prop_name.setText(node.title)

            self.merge_props_panel.load_from_properties(props)

            self.write_props_panel.load_from_properties(props)
            self._refresh_write_panel_info(node)
        finally:
            self._updating_properties = False

    def _apply_properties_from_ui(self) -> None:
        if self._updating_properties or self._active_node is None:
            return

        node = self._active_node
        entered_title = self.prop_name.text().strip()
        default_title, _default_subtitle = self._spec_texts(node.node_type)
        if node.node_type == "corridorkey":
            default_title = self._corridorkey_auto_title(node.properties)
        if entered_title:
            node.title = entered_title
            node.properties["custom_title"] = entered_title != default_title
        else:
            node.title = default_title
            node.properties["custom_title"] = False

        if node.node_type in {"load", "source", "alpha"}:
            self.load_props_panel.write_to_properties(node.properties)
            self._refresh_load_node_preview(node)
            self._update_load_media_info()
            self._sync_connected_matting_presets(node)
            self._sync_connected_write_outputs(node)
            self._emit_read_media_selected(node)
        elif node.node_type in {"sam2"}:
            self.sam_props_panel.write_to_properties(node.properties)
            node.set_annotation_lines(self._build_sam_annotation(node.properties, node_type="sam2"))
        elif node.node_type == "sam3":
            self.sam3_props_panel.write_to_properties(node.properties)
            node.set_annotation_lines(self._build_sam_annotation(node.properties, node_type="sam3"))
        elif node.node_type == "matting":
            self.matting_props_panel.write_to_properties(node.properties)
            node.set_annotation_lines(self._build_matting_annotation(node))
        elif node.node_type == "birefnet":
            self.birefnet_props_panel.write_to_properties(node.properties)
            node.set_annotation_lines(
                self._build_birefnet_annotation(
                    node.properties,
                    self._birefnet_runtime_percent,
                    self._birefnet_runtime_text,
                )
            )
        elif node.node_type == "gvm":
            self.gvm_props_panel.write_to_properties(node.properties)
        elif node.node_type == "chromakey":
            self.chromakey_props_panel.write_to_properties(node.properties)
        elif node.node_type == "merge":
            self.merge_props_panel.write_to_properties(node.properties)
        elif node.node_type == "corridorkey":
            self.corridorkey_props_panel.write_to_properties(node.properties)
            if not bool(node.properties.get("custom_title", False)):
                node.title = self._corridorkey_auto_title(node.properties)
                self.prop_name.setText(node.title)
            node.set_annotation_lines(self._build_corridorkey_annotation(node.properties))
        elif node.node_type == "export":
            prev_output_format = str(node.properties.get("output_format", "source")).strip().lower() or "source"
            self.write_props_panel.write_to_properties(node.properties)
            new_output_format = str(node.properties.get("output_format", "source")).strip().lower() or "source"
            stream_key = self._write_input_stream(node)
            video_exts = {"mp4", "mov", "avi", "mkv", "webm", "m4v"}

            node.properties["_auto_png16_for_matte_applied"] = False
            node.properties["_auto_prores4444_for_processed_applied"] = False
            if (
                prev_output_format != "png"
                and new_output_format == "png"
                and self._write_stream_is_mask(node)
                and int(node.properties.get("png_bit_depth", 8) or 8) < 16
            ):
                node.properties["png_bit_depth"] = 16
                node.properties["_auto_png16_for_matte_applied"] = True
                self._updating_properties = True
                try:
                    self.write_props_panel.load_from_properties(node.properties)
                finally:
                    self._updating_properties = False

            if stream_key == "processed" and new_output_format in video_exts:
                codec_key = str(node.properties.get("video_codec", "h264")).strip().lower() or "h264"
                needs_adjust = new_output_format != "mov" or codec_key != "prores4444"
                if needs_adjust:
                    node.properties["output_format"] = "mov"
                    node.properties["video_codec"] = "prores4444"
                    node.properties["_auto_prores4444_for_processed_applied"] = True
                    self._updating_properties = True
                    try:
                        self.write_props_panel.load_from_properties(node.properties)
                    finally:
                        self._updating_properties = False
            self._refresh_write_panel_info(node)

        node.update()
        if node.node_type == "merge":
            # Defer preview: avoid synchronous disk I/O + numpy compositing on the
            # UI thread for every spinner tick.  Coalesces rapid changes into one call.
            self._merge_preview_pending_node = node
            self._merge_preview_debounce_timer.start()
        else:
            self._emit_preview_request(node)

    def _flush_merge_preview_request(self) -> None:
        """Emit the deferred Merge quick-preview request after debounce idle period."""
        if self._merge_preview_pending_node is not None:
            self._emit_preview_request(self._merge_preview_pending_node)
            self._merge_preview_pending_node = None

    def _browse_load_media(self) -> None:
        if self._active_node is None or self._active_node.node_type not in {"load", "source", "alpha"}:
            return

        is_video = self.load_props_panel.media_type_combo.currentIndex() == 0
        if is_video:
            file_filter = "Video or Sequence Files (*.mp4 *.avi *.mov *.mkv *.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff *.exr);;All Files (*)"
        else:
            file_filter = "Image Files (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff *.exr);;All Files (*)"

        path, _ = QFileDialog.getOpenFileName(self, self._tr("node_props_browse_media"), "", file_filter)
        if not path:
            return

        self.load_props_panel.path_edit.setText(path)
        self._apply_properties_from_ui()

    def _browse_write_output(self) -> None:
        if self._active_node is None or self._active_node.node_type != "export":
            return

        current_path = self.write_props_panel.path_edit.text().strip()
        start_dir = current_path if current_path else str(Path.home())
        path = QFileDialog.getExistingDirectory(self, self._tr("node_props_browse_output"), start_dir)
        if not path:
            return

        self.write_props_panel.path_edit.setText(path)
        self._apply_properties_from_ui()

    def _update_load_media_info(self) -> None:
        path = self.load_props_panel.path_edit.text().strip()
        if not path:
            self.load_props_panel.set_info_text(self._tr("node_props_media_not_selected"))
            return

        info_text = self._build_media_info_text(path)
        self.load_props_panel.set_info_text(info_text)

    def _read_media_dimensions(self, path: str, media_type: str) -> tuple[int, int] | None:
        try:
            return read_media_dimensions(path, media_type)
        except Exception:
            return None

    @staticmethod
    def _resolve_eval_preset_for_dimensions(width: int, height: int) -> str | None:
        min_side = min(width, height)
        if min_side <= 576:
            return "Eval LR (512p)"
        if min_side >= 900:
            return "Eval HR (1080p)"
        return None

    @staticmethod
    def _resolve_default_output_dir(media_path: str) -> str:
        if not media_path:
            return ""
        return str(build_keyflow_base_dir(Path(media_path)))

    @staticmethod
    def _resolve_default_output_name(media_path: str) -> str:
        if not media_path:
            return ""
        return Path(media_path).stem

    def _find_upstream_load_node(self, node: NodeItem) -> NodeItem | None:
        current = node
        seen: set[int] = set()
        while current is not None:
            current_id = id(current)
            if current_id in seen:
                return None
            seen.add(current_id)
            incoming = self._incoming_edges(current)
            if not incoming:
                return current if current.node_type in {"load", "source", "alpha"} else None
            img_edge = next((e for e in incoming if e.dst_port == "img"), None) or incoming[0]
            current = img_edge.src
            if current.node_type in {"load", "source", "alpha"}:
                return current
        return None

    def _resolve_write_context(self, write_node: NodeItem) -> tuple[str, str]:
        load_node = self._find_upstream_load_node(write_node)
        if load_node is None:
            return ("", "")

        media_path = str((load_node.properties or {}).get("path", "")).strip()
        if not media_path:
            return ("", "")

        return (
            self._resolve_default_output_dir(media_path),
            self._resolve_default_output_name(media_path),
        )

    def _write_input_connection(self, write_node: NodeItem) -> ConnectionItem | None:
        if write_node.node_type != "export":
            return None
        incoming = self._incoming_edges(write_node)
        for edge in incoming:
            if self._normalized_edge_dst_port(edge) == "in":
                return edge
        return None

    def _write_stream_is_mask(self, write_node: NodeItem) -> bool:
        edge = self._write_input_connection(write_node)
        if edge is None:
            return False
        if edge.src.node_type == "alpha":
            return True
        stream = self._resolve_connected_write_stream(edge)
        return stream in {"alpha", "sam_mask"}

    def _corridorkey_has_fg_write_pair(self, corridorkey_node: NodeItem, exclude_write_node: NodeItem | None = None) -> bool:
        for edge in self._connections:
            if edge.src is not corridorkey_node:
                continue
            if edge.dst.node_type != "export":
                continue
            edge_dst_port = str(edge.dst_port or "").strip().lower()
            if edge_dst_port not in {"", "in"}:
                continue
            if exclude_write_node is not None and edge.dst is exclude_write_node:
                continue
            if self._resolve_connected_write_stream(edge) == "fg":
                return True
        return False

    @staticmethod
    def _stream_badge_html(stream: str) -> str:
        if stream == "fg":
            return '<span style="color:#57b8e9; font-weight:600;">● FG CLEAN</span>'
        if stream == "alpha":
            return '<span style="color:#a0a0a0; font-weight:600;">● ALPHA</span>'
        if stream == "sam_mask":
            return '<span style="color:#e8943b; font-weight:600;">● SAM2 MASK</span>'
        if stream == "input":
            return '<span style="color:#57b8e9; font-weight:600;">● INPUT</span>'
        if stream == "comp":
            return '<span style="color:#57b8e9; font-weight:600;">● PREVIEW</span>'
        if stream == "processed":
            return '<span style="color:#57b8e9; font-weight:600;">● PREMULT</span>'
        return '<span style="color:#7a9ab8;">● -</span>'

    @staticmethod
    def _resolve_effective_write_format(output_format: str, media_path: str) -> str:
        return resolve_write_output_format({"output_format": output_format}, Path(str(media_path or "input.png")))

    def _build_write_guidance(self, write_node: NodeItem) -> tuple[str, str, str] | None:
        edge = self._write_input_connection(write_node)
        if edge is None:
            return None

        src_node = edge.src
        stream = self._resolve_connected_write_stream(edge)
        props = write_node.properties or {}
        media_path = str((self._find_upstream_load_node(write_node).properties or {}).get("path", "")).strip() if self._find_upstream_load_node(write_node) is not None else ""
        output_format = str(props.get("output_format", "source")).strip().lower() or "source"
        resolved_format = self._resolve_effective_write_format(output_format, media_path)
        video_codec = str(props.get("video_codec", "h264")).strip().lower() or "h264"
        png_bit_depth = int(props.get("png_bit_depth", 8) or 8)
        video_exts = set(VIDEO_OUTPUT_FORMATS)

        semantic = ""
        recommended = ""
        warning = ""

        if src_node.node_type == "corridorkey":
            if stream == "alpha":
                semantic = self._tr("node_props_write_semantic_corridorkey_alpha")
                recommended = self._tr("node_props_write_recommend_corridorkey_alpha")
                if not self._corridorkey_has_fg_write_pair(src_node, write_node):
                    warning = self._tr("node_props_write_warning_corridorkey_alpha_without_fg")
                if output_format == "source":
                    warning = "\n".join(filter(None, [warning, self._tr("node_props_write_warning_mask_source")]))
                elif resolved_format in video_exts:
                    warning = "\n".join(filter(None, [warning, self._tr("node_props_write_warning_mask_video")]))
                elif resolved_format == "jpg":
                    warning = "\n".join(filter(None, [warning, self._tr("node_props_write_warning_mask_jpg")]))
                elif resolved_format == "png" and png_bit_depth < 16:
                    warning = "\n".join(filter(None, [warning, self._tr("node_props_write_warning_mask_png8")]))
            elif stream == "fg":
                semantic = self._tr("node_props_write_semantic_corridorkey_fg")
                recommended = self._tr("node_props_write_recommend_corridorkey_fg")
            elif stream == "comp":
                semantic = self._tr("node_props_write_semantic_corridorkey_comp")
                recommended = self._tr("node_props_write_recommend_corridorkey_comp")
                warning = self._tr("node_props_write_warning_corridorkey_comp")
            elif stream == "processed":
                semantic = self._tr("node_props_write_semantic_corridorkey_processed")
                recommended = self._tr("node_props_write_recommend_corridorkey_processed")
                if output_format == "source":
                    warning = self._tr("node_props_write_warning_processed_source")
                elif resolved_format in video_exts and video_codec != "prores4444":
                    warning = self._tr("node_props_write_warning_processed_video")
        elif src_node.node_type == "alpha" or stream in {"alpha", "sam_mask"}:
            semantic = self._tr("node_props_write_semantic_generic_mask")
            recommended = self._tr("node_props_write_recommend_generic_mask")
            if output_format == "source":
                warning = self._tr("node_props_write_warning_mask_source")
            elif resolved_format in video_exts:
                warning = self._tr("node_props_write_warning_mask_video")
            elif resolved_format == "jpg":
                warning = self._tr("node_props_write_warning_mask_jpg")
            elif resolved_format == "png" and png_bit_depth < 16:
                warning = self._tr("node_props_write_warning_mask_png8")
        elif stream in {"fg", "image", "input", "out"}:
            semantic = self._tr("node_props_write_semantic_generic_image")
            recommended = self._tr("node_props_write_recommend_generic_image")

        source_name = src_node.title or src_node.node_type.title()
        source_text = self._tr("node_props_resolved_output_source").format(
            node=source_name,
            stream=stream or "out",
        )
        return (source_text, semantic, f"{recommended}\n{warning}".strip())

    @staticmethod
    def _recommended_write_settings_for_connection(src_node: NodeItem, stream: str) -> dict[str, object]:
        stream_key = str(stream or "").strip().lower()
        if src_node.node_type == "corridorkey":
            if stream_key == "alpha":
                return {"output_format": "exr"}
            if stream_key == "fg":
                return {"output_format": "exr"}
            if stream_key == "comp":
                return {"output_format": "png"}
            if stream_key == "processed":
                return {"output_format": "exr"}
        return {}

    def _apply_soft_write_recommendation(self, write_node: NodeItem, src_node: NodeItem, stream: str) -> None:
        if write_node.node_type != "export":
            return

        props = write_node.properties or {}
        current_format = str(props.get("output_format", "source")).strip().lower() or "source"
        if current_format != "source":
            return

        recommended = self._recommended_write_settings_for_connection(src_node, stream)
        if not recommended:
            return

        props.update(recommended)

        if self._active_node is write_node:
            self._updating_properties = True
            try:
                self.write_props_panel.load_from_properties(props)
            finally:
                self._updating_properties = False

    def _refresh_write_panel_info(self, node: NodeItem) -> None:
        if node.node_type != "export":
            return

        resolved_dir, default_name = self._resolve_write_context(node)
        props = node.properties or {}
        stream = self._write_input_stream(node)
        auto_output_dir = bool(props.get("auto_output_dir", True))
        custom_base_dir = str(props.get("output_dir", "")).strip()
        base_dir = resolved_dir if auto_output_dir or not custom_base_dir else custom_base_dir
        if base_dir:
            edge = self._write_input_connection(node)
            if edge is not None:
                src_title = str(edge.src.title or edge.src.node_type).strip()
                src_port = self._normalized_edge_src_port(edge)
                port_display = src_port.capitalize()
                _ss = get_node_spec(edge.src.node_type)
                if _ss is not None:
                    _sp = next((p for p in _ss.outputs if p.name == src_port), None)
                    if _sp is not None and _sp.label:
                        port_display = _sp.label
                output_dir = str(
                    build_graph_write_output_dir(
                        Path(base_dir),
                        source_node_title=src_title,
                        port_label=port_display,
                        stream_label=stream,
                    )
                )
            else:
                output_dir = str(build_graph_write_output_dir(Path(base_dir), stream_label=stream))
        else:
            output_dir = ""
        output_name = str(props.get("file_name", "")).strip() or default_name
        output_format = str(props.get("output_format", "source")).strip().lower() or "source"

        format_label = self.write_props_panel.current_format_label()
        if output_format == "source":
            format_label = self._tr("node_props_format_source")

        stream_badge = self._stream_badge_html(stream)

        info_parts: list[str] = []
        if output_dir:
            info_parts.append(f"{html.escape(self._tr('node_props_resolved_output_dir'))} {html.escape(output_dir)}")
        else:
            info_parts.append(html.escape(self._tr("node_props_write_no_input")))

        if output_name:
            info_parts.append(f"{html.escape(self._tr('node_props_resolved_output_name'))} {html.escape(output_name)}")

        info_parts.append(f"{html.escape(self._tr('node_props_resolved_output_format'))} {html.escape(format_label)}")
        info_parts.append(f"{html.escape(self._tr('node_props_resolved_output_stream'))} {stream_badge}")

        guidance = self._build_write_guidance(node)
        if guidance is not None:
            source_text, semantic_text, recommendation_and_warning = guidance
            if source_text:
                info_parts.append(html.escape(source_text))
            if auto_output_dir:
                info_parts.append(html.escape(self._tr("node_props_write_rule_summary")))
            if semantic_text:
                info_parts.append(
                    f"{html.escape(self._tr('node_props_write_semantic_label'))} {html.escape(semantic_text)}"
                )
            if recommendation_and_warning:
                for line in recommendation_and_warning.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("WARNING:"):
                        info_parts.append(
                            f'<span style="color:#f0b35c;">{html.escape(line.replace("WARNING:", self._tr("node_props_write_warning_label") + " ", 1))}</span>'
                        )
                    else:
                        info_parts.append(
                            f"{html.escape(self._tr('node_props_write_recommend_label'))} {html.escape(line)}"
                        )

        if bool(props.get("_auto_png16_for_matte_applied", False)):
            info_parts.append(
                f'<span style="color:#9fd0ff;">{html.escape(self._tr("node_props_write_auto_png16_applied"))}</span>'
            )
        if bool(props.get("_auto_prores4444_for_processed_applied", False)):
            info_parts.append(
                f'<span style="color:#9fd0ff;">{html.escape(self._tr("node_props_write_auto_processed_video_applied"))}</span>'
            )
        self.write_props_panel.set_info_text("<br/>".join(info_parts))

    def _update_matting_output_indicator(self, node: NodeItem) -> None:
        if node.node_type != "matting":
            return
        node.set_annotation_lines(self._build_matting_annotation(node))

    def _refresh_matting_output_indicators(self) -> None:
        self._refresh_matting_annotations()

    def _apply_matting_eval_preset(self, matting_node: NodeItem, preset_name: str) -> None:
        if preset_name == "Eval LR (512p)":
            erode, dilate, warmup = (4, 4, 1)
        elif preset_name == "Eval HR (1080p)":
            erode, dilate, warmup = (15, 15, 10)
        else:
            return

        matting_node.properties["preset"] = preset_name
        matting_node.properties["erode"] = erode
        matting_node.properties["dilate"] = dilate
        matting_node.properties["warmup"] = warmup

        if self._active_node is matting_node:
            self._load_properties_to_ui(matting_node)
        else:
            matting_node.update()

    def _auto_select_eval_preset_for_matting_node(self, matting_node: NodeItem) -> None:
        if matting_node.node_type != "matting":
            return
        if str(matting_node.properties.get("preset", "")).strip() == "Custom":
            return

        load_node = self._find_upstream_load_node(matting_node)
        if load_node is None:
            return

        load_props = load_node.properties or {}
        media_path = str(load_props.get("path", "")).strip()
        if not media_path:
            return
        media_type = str(load_props.get("media_type", "video")).strip().lower()
        dims = self._read_media_dimensions(media_path, media_type)
        if dims is None:
            return

        preset_name = self._resolve_eval_preset_for_dimensions(dims[0], dims[1])
        if not preset_name:
            return
        if str(matting_node.properties.get("preset", "")) == preset_name:
            return

        self._apply_matting_eval_preset(matting_node, preset_name)

    def _sync_connected_matting_presets(self, load_node: NodeItem) -> None:
        if load_node.node_type not in {"load", "source", "alpha"}:
            return

        stack = [load_node]
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            node_id = id(node)
            if node_id in seen:
                continue
            seen.add(node_id)

            for edge in self._connections:
                if edge.src is not node:
                    continue
                if edge.dst.node_type == "matting":
                    self._auto_select_eval_preset_for_matting_node(edge.dst)
                else:
                    stack.append(edge.dst)

    def _sync_connected_write_outputs(self, load_node: NodeItem) -> None:
        if load_node.node_type not in {"load", "source", "alpha"}:
            return

        stack = [load_node]
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            node_id = id(node)
            if node_id in seen:
                continue
            seen.add(node_id)

            for edge in self._connections:
                if edge.src is not node:
                    continue
                if edge.dst.node_type == "export":
                    if self._active_node is edge.dst:
                        self._refresh_write_panel_info(edge.dst)
                    else:
                        edge.dst.update()
                else:
                    stack.append(edge.dst)

    def _build_media_info_text(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            return self._tr("node_props_media_not_found")

        if path in self._media_info_cache:
            return self._media_info_cache[path]

        info_parts: list[str] = []
        info_parts.append(f"{self._tr('file_info_file')} {p.name}")

        try:
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            info_parts.append(f"{self._tr('file_info_size')} {file_size_mb:.2f} {self._tr('file_info_size_unit')}")
        except Exception:
            pass

        media_type = "video" if self.load_props_panel.media_type_combo.currentIndex() == 0 else "image"
        if is_numbered_image_sequence(path):
            try:
                sequence_paths = resolve_numbered_image_sequence(path)
                first_frame = load_rgb_image(sequence_paths[0])
                h, w = first_frame.shape[:2]
                info_parts.append(f"{self._tr('file_info_res')} {w}x{h}")
                info_parts.append(f"{self._tr('file_info_frames')} {len(sequence_paths)}")
                info_parts.append(self._tr("file_info_type_sequence"))
                result = "\n".join(info_parts)
                self._media_info_cache[path] = result
                return result
            except Exception:
                return self._tr("node_props_media_read_error")

        if media_type == "video":
            try:
                import cv2

                cap = cv2.VideoCapture(path)
                try:
                    if not cap.isOpened():
                        return self._tr("node_props_media_read_error")
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                finally:
                    cap.release()

                if w > 0 and h > 0:
                    info_parts.append(f"{self._tr('file_info_res')} {w}x{h}")
                if frame_count > 0:
                    info_parts.append(f"{self._tr('file_info_frames')} {frame_count}")
                if fps > 0:
                    info_parts.append(f"{self._tr('file_info_fps')} {fps:.2f}")
                    duration_sec = frame_count / fps if frame_count > 0 else 0
                    mins = int(duration_sec // 60)
                    secs = int(duration_sec % 60)
                    info_parts.append(f"{self._tr('file_info_duration')} {mins}:{secs:02d}")
                try:
                    from app.utils.ffmpeg import get_color_space_info
                    cs = get_color_space_info(path)
                    if cs:
                        info_parts.append(f"{self._tr('file_info_color_space')} {cs}")
                except Exception:
                    pass
            except Exception:
                return self._tr("node_props_media_read_error")
        else:
            try:
                frame = load_rgb_image(path)
                h, w = frame.shape[:2]
                info_parts.append(f"{self._tr('file_info_res')} {w}x{h}")
                try:
                    from app.utils.ffmpeg import get_image_color_space
                    cs = get_image_color_space(path)
                    if cs:
                        info_parts.append(f"{self._tr('file_info_color_space')} {cs}")
                except Exception:
                    pass
                info_parts.append(self._tr("file_info_type_image"))
            except Exception:
                return self._tr("node_props_media_read_error")

        result = "\n".join(info_parts)
        self._media_info_cache[path] = result
        return result

    def _refresh_load_node_preview(self, node: NodeItem, frame_index: int = 0) -> None:
        if node.node_type not in {"load", "source", "alpha"}:
            return
        props = node.properties or {}
        path = str(props.get("path", "")).strip()
        media_type = str(props.get("media_type", "video")).strip().lower()
        # Invalidate cache entries for paths that no longer exist
        self._thumbnail_cache = {
            (cache_path, idx): pix
            for (cache_path, idx), pix in self._thumbnail_cache.items()
            if Path(cache_path).exists()
        }
        self._media_info_cache = {k: v for k, v in self._media_info_cache.items() if Path(k).exists()}
        node.set_preview_pixmap(self._build_media_thumbnail(path, media_type, frame_index=frame_index))

    def _build_media_thumbnail(self, path: str, media_type: str, *, frame_index: int = 0) -> QPixmap | None:
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None

        safe_frame_index = max(0, int(frame_index))
        cache_key = (path, safe_frame_index)
        if cache_key in self._thumbnail_cache:
            return self._thumbnail_cache[cache_key]

        thumb_w = 160
        thumb_h = 90
        pix: QPixmap | None = None

        if media_type == "image" and not is_numbered_image_sequence(path):
            try:
                frame = load_rgb_image(path)
                image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format.Format_RGB888).copy()
                pix = QPixmap.fromImage(image).scaled(thumb_w, thumb_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            except Exception:
                pass

        elif is_numbered_image_sequence(path):
            try:
                seq = resolve_numbered_image_sequence(path)
                if not seq:
                    return None
                safe_idx = max(0, min(len(seq) - 1, safe_frame_index))
                frame = load_rgb_image(seq[safe_idx])
                image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format.Format_RGB888).copy()
                pix = QPixmap.fromImage(image).scaled(thumb_w, thumb_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            except Exception:
                pass

        else:
            try:
                import cv2

                cap = cv2.VideoCapture(path)
                if safe_frame_index > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, safe_frame_index)
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, _ = rgb.shape
                    image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
                    candidate = QPixmap.fromImage(image)
                    if not candidate.isNull():
                        pix = candidate.scaled(thumb_w, thumb_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            except Exception:
                pass

        if pix is not None and not pix.isNull():
            self._thumbnail_cache[cache_key] = pix
        return pix

    def update_active_read_node_preview_frame(self, frame_index: int) -> None:
        node = self._active_node
        if node is None or node.node_type not in {"load", "source", "alpha"}:
            return
        self._refresh_load_node_preview(node, frame_index=max(0, int(frame_index)))

    def retranslate_ui(self) -> None:
        self._updating_properties = True
        self.setWindowTitle(self._tr("node_graph_title"))
        self.hint_label.setText(self._tr("node_graph_hint"))
        self.btn_reset.setText(self._tr("node_graph_reset_view"))
        self.btn_close.setText(self._tr("node_graph_close"))
        self._update_graph_diagnostics_button(False)

        self.props_title.setText(self._tr("node_props_title"))
        self.props_empty.setText(self._tr("node_props_empty"))

        self._set_form_label_text(0, self._tr("node_props_name"))
        self._set_form_label_text(1, "")
        self._set_form_label_text(2, "")
        self._set_form_label_text(3, "")
        self._set_form_label_text(4, "")
        self._set_form_label_text(5, "")

        self.load_props_panel.set_translator(self._tr)
        self._update_load_media_info()
        self.sam_props_panel.set_translator(self._tr)
        self.sam3_props_panel.set_translator(self._tr)
        self.matting_props_panel.set_translator(self._tr)
        self.birefnet_props_panel.set_translator(self._tr)
        self.gvm_props_panel.set_translator(self._tr)
        self.chromakey_props_panel.set_translator(self._tr)
        self.corridorkey_props_panel.set_translator(self._tr)
        self.write_props_panel.set_translator(self._tr)
        self._retranslate_existing_nodes()
        self._updating_properties = False
        self._refresh_graph_diagnostics()
        if self._diagnostics_dialog is not None:
            self._diagnostics_dialog.set_translator(self._tr)
        if self._active_node is not None and self._active_node.node_type in {"sam2"}:
            self.sam_props_panel.sync_controls(self._active_node.properties)
            self._emit_preview_request(self._active_node)
        if self._active_node is not None and self._active_node.node_type == "sam3":
            self._emit_preview_request(self._active_node)

    def _retranslate_existing_nodes(self) -> None:
        for item in self.scene.items():
            if not isinstance(item, NodeItem):
                continue
            spec = get_node_spec(item.node_type)
            if spec is None:
                continue
            default_title, default_subtitle = self._spec_texts(item.node_type)
            if not bool(item.properties.get("custom_title", False)):
                if item.node_type == "corridorkey":
                    item.title = self._corridorkey_auto_title(item.properties)
                else:
                    item.title = default_title
            item.subtitle = default_subtitle
            if item.node_type == "birefnet":
                item.set_annotation_lines(
                    self._build_birefnet_annotation(
                        item.properties,
                        self._birefnet_runtime_percent,
                        self._birefnet_runtime_text,
                    )
                )
            if item.node_type == "corridorkey":
                item.set_annotation_lines(self._build_corridorkey_annotation(item.properties))
            item.update()

    def _set_form_label_text(self, row: int, text: str) -> None:
        item = self.props_form.itemAt(row, QFormLayout.ItemRole.LabelRole)
        if item is None or item.widget() is None:
            return
        item.widget().setText(text)

    def set_translator(self, translate: Callable[[str], str]) -> None:
        self._tr = translate
        self.retranslate_ui()
        self._quick_add_dialog._tr = translate
        self._quick_add_dialog.retranslate_ui()
        if self._active_node is not None:
            self._load_properties_to_ui(self._active_node)

    def refresh_cloud_weights_status(self, api_host: str) -> None:
        """After instance becomes Running: fetch /models in a background thread
        and update _cloud_weights_ready flag + button text on all 4 panels."""
        import http.client
        import json
        import threading
        import urllib.error
        import urllib.request

        panels = [
            (self.gvm_props_panel,          "gvm"),
            (self.matting_props_panel,       "matanyone2"),
            (self.birefnet_props_panel,      "birefnet"),
            (self.corridorkey_props_panel,   "corridorkey"),
        ]

        def _worker() -> None:
            models: dict = {}
            try:
                req = urllib.request.Request(f"{api_host}/models")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    payload = json.loads(resp.read())
                models = payload.get("models", {}) if isinstance(payload, dict) else {}
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                OSError,
                TimeoutError,
                ValueError,
                json.JSONDecodeError,
            ):
                pass

            from PySide6.QtCore import QTimer
            for panel, key in panels:
                ready = bool(models.get(key))
                # schedule UI update on the main thread
                QTimer.singleShot(0, panel, lambda p=panel, r=ready, m=models: _apply(p, r, m))

        def _apply(panel, ready: bool, models_info: dict) -> None:
            panel._cloud_weights_ready = ready
            panel._cloud_models_info = models_info
            panel._refresh_download_button_state()

        threading.Thread(target=_worker, daemon=True).start()

    def set_gvm_cloud_status(self, text: str | None) -> None:
        """Update annotation lines on all GVM nodes with cloud inference progress text.

        Pass *text* to show the status, or ``None`` / empty string to clear it.
        """
        lines: list[tuple[str, str]] = []
        if text:
            lines = [(str(text), "#f4c97a")]
        for item in self.scene.items():
            if isinstance(item, NodeItem) and item.node_type == "gvm":
                item.set_annotation_lines(lines)

    def clear_graph(self) -> None:
        self.scene.clear()
        self._connections.clear()
        self._groups.clear()
        self.clear_active_selection()
        self._schedule_graph_diagnostics_refresh()

    def graph_is_empty(self) -> bool:
        return not any(isinstance(item, NodeItem) for item in self.scene.items())

    def _schedule_graph_diagnostics_refresh(self, *_args) -> None:
        self._graph_diagnostics_refresh_timer.start()

    def _next_auto_graph_node_id(self) -> str:
        used_ids: set[str] = set()
        for item in self.scene.items():
            if not isinstance(item, NodeItem):
                continue
            node_id = str((item.properties or {}).get("graph_node_id", "")).strip()
            if node_id:
                used_ids.add(node_id)

        index = 0
        while f"n{index}" in used_ids:
            index += 1
        return f"n{index}"

    def _ensure_graph_node_id(self, node: NodeItem) -> str:
        props = node.properties or {}
        node_id = str(props.get("graph_node_id", "")).strip()
        if node_id:
            return node_id

        node_id = self._next_auto_graph_node_id()
        props = dict(props)
        props["graph_node_id"] = node_id
        node.properties = props
        return node_id

    def _ensure_scene_graph_node_ids(self) -> None:
        for item in self.scene.items():
            if not isinstance(item, NodeItem):
                continue
            self._ensure_graph_node_id(item)

    def _graph_runtime_nodes_and_edges(self) -> tuple[list[GraphNode], list[GraphEdge], dict[str, NodeItem]]:
        self._ensure_scene_graph_node_ids()
        nodes = [item for item in self.scene.items() if isinstance(item, NodeItem)]
        nodes.sort(key=lambda node: (str((node.properties or {}).get("graph_node_id", "")), node.node_type, id(node)))

        node_ids: dict[int, str] = {}
        node_items_by_id: dict[str, NodeItem] = {}
        runtime_nodes: list[GraphNode] = []
        for node in nodes:
            node_id = self._ensure_graph_node_id(node)
            node_ids[id(node)] = node_id
            node_items_by_id[node_id] = node
            runtime_nodes.append(
                GraphNode(
                    id=node_id,
                    type=str(node.node_type),
                    title=str(node.title),
                    properties=copy.deepcopy(node.properties or {}),
                    enabled=bool((node.properties or {}).get("enabled", True)),
                )
            )

        runtime_edges: list[GraphEdge] = []
        for edge in self._connections:
            src_id = node_ids.get(id(edge.src))
            dst_id = node_ids.get(id(edge.dst))
            if not src_id or not dst_id:
                continue
            runtime_edges.append(
                GraphEdge(
                    src_id=src_id,
                    dst_id=dst_id,
                    src_port=self._normalized_edge_src_port(edge),
                    dst_port=self._normalized_edge_dst_port(edge),
                )
            )
        return runtime_nodes, runtime_edges, node_items_by_id

    def _find_node_by_graph_id(self, node_id: str) -> NodeItem | None:
        target = str(node_id or "").strip()
        if not target:
            return None
        for item in self.scene.items():
            if not isinstance(item, NodeItem):
                continue
            if str((item.properties or {}).get("graph_node_id", "")).strip() == target:
                return item
        return None

    def _focus_node_by_graph_id(self, node_id: str) -> bool:
        node = self._find_node_by_graph_id(node_id)
        if node is None:
            return False
        self.scene.clearSelection()
        node.setSelected(True)
        self._active_node = node
        self._load_properties_to_ui(node)
        self._emit_preview_request(node)
        self.active_node_changed.emit(node.node_type)
        self.view.centerOn(node.sceneBoundingRect().center())
        return True

    def _on_diagnostics_link_clicked(self, url: QUrl) -> None:
        target = str(url.toString() or "").strip()
        if target.startswith("diag://"):
            self._focus_node_by_graph_id(target.replace("diag://", "", 1))

    def _open_graph_diagnostics_dialog(self) -> None:
        if self._diagnostics_dialog is None:
            self._diagnostics_dialog = GraphDiagnosticsDialog(self._tr, self)
            self._diagnostics_dialog.anchor_clicked.connect(self._on_diagnostics_link_clicked)
            self._diagnostics_dialog.strict_mode_toggled.connect(self._on_diagnostics_strict_mode_toggled)
        self._diagnostics_dialog.set_strict_mode(self._graph_diag_strict_required_inputs)
        self._diagnostics_dialog.set_diagnostics_content(
            self._graph_diagnostics_summary,
            self._graph_diagnostics_html,
        )
        self._diagnostics_dialog.show()
        self._diagnostics_dialog.raise_()
        self._diagnostics_dialog.activateWindow()

    def _on_diagnostics_strict_mode_toggled(self, checked: bool) -> None:
        enabled = bool(checked)
        if self._graph_diag_strict_required_inputs == enabled:
            return
        self._graph_diag_strict_required_inputs = enabled
        self._settings.setValue("node_graph/diag_strict_required_inputs", enabled)
        self._schedule_graph_diagnostics_refresh()

    def _update_graph_diagnostics_button(self, has_errors: bool) -> None:
        tooltip = self._tr("graph_diagnostics_open_tooltip")
        summary = str(getattr(self, "_graph_diagnostics_summary", "") or "").strip()
        if summary:
            tooltip = f"{tooltip}\n{summary}"
        self.btn_open_diagnostics.setToolTip(tooltip)

        icon_kind = (
            QStyle.StandardPixmap.SP_MessageBoxWarning
            if bool(has_errors)
            else QStyle.StandardPixmap.SP_DialogApplyButton
        )
        self.btn_open_diagnostics.setIcon(self.style().standardIcon(icon_kind))

    def _apply_graph_diagnostic_visuals(self, diagnostics, node_items_by_id: dict[str, NodeItem]) -> None:
        node_counts: dict[str, int] = {}
        input_ports: dict[str, set[str]] = {}
        output_ports: dict[str, set[str]] = {}

        for diagnostic in diagnostics:
            for node_id in {
                str(getattr(diagnostic, "node_id", "") or "").strip(),
                str(getattr(diagnostic, "src_node_id", "") or "").strip(),
                str(getattr(diagnostic, "dst_node_id", "") or "").strip(),
            }:
                if not node_id:
                    continue
                node_counts[node_id] = node_counts.get(node_id, 0) + 1

            src_node_id = str(getattr(diagnostic, "src_node_id", "") or "").strip()
            src_port = str(getattr(diagnostic, "src_port", "") or "").strip()
            if src_node_id and src_port:
                output_ports.setdefault(src_node_id, set()).add(src_port)

            dst_node_id = str(getattr(diagnostic, "dst_node_id", "") or "").strip()
            dst_port = str(getattr(diagnostic, "dst_port", "") or "").strip()
            if not dst_node_id:
                dst_node_id = str(getattr(diagnostic, "node_id", "") or "").strip()
            if dst_node_id and dst_port:
                input_ports.setdefault(dst_node_id, set()).add(dst_port)

        for node_id, node in node_items_by_id.items():
            node.set_diagnostic_state(
                input_ports=input_ports.get(node_id, set()),
                output_ports=output_ports.get(node_id, set()),
                count=node_counts.get(node_id, 0),
            )

    def _refresh_graph_diagnostics(self) -> None:
        nodes, edges, node_items_by_id = self._graph_runtime_nodes_and_edges()
        if not nodes and not edges:
            summary = self._tr("graph_diagnostics_status_ok")
            details = self._tr("graph_diagnostics_empty")
            signature = f"{summary}\n{details}"
            self._apply_graph_diagnostic_visuals([], node_items_by_id)
            html_text = f'<span style="color:#8ca0ba;">{html.escape(details)}</span>'
            self._graph_diagnostics_summary = summary
            self._graph_diagnostics_html = html_text
            self._update_graph_diagnostics_button(False)
            if self._diagnostics_dialog is not None:
                self._diagnostics_dialog.set_diagnostics_content(summary, html_text)
            if signature == self._graph_diagnostics_signature:
                return
            self._graph_diagnostics_signature = signature
            self.graph_diagnostics_changed.emit(summary, details, False)
            return

        engine = NodeGraphEngine()
        _is_valid, diagnostics = engine.validate_with_diagnostics(
            nodes,
            edges,
            strict_isolated_required_inputs=self._graph_diag_strict_required_inputs,
        )
        self._apply_graph_diagnostic_visuals(diagnostics, node_items_by_id)

        def _diagnostic_node_label(node_id: str) -> str:
            target = str(node_id or "").strip()
            if not target:
                return ""
            node = node_items_by_id.get(target)
            if node is None:
                return target
            title = str(node.title or "").strip()
            if title:
                return title
            spec = get_node_spec(node.node_type)
            if spec is not None:
                translated = self._tr(spec.title_i18n_key)
                if translated != spec.title_i18n_key:
                    return translated
                return spec.title
            return str(node.node_type or target)

        summary = format_graph_diagnostics_summary(self._tr, diagnostics)
        details = format_graph_diagnostics_text(
            self._tr,
            diagnostics,
            node_label_for_id=_diagnostic_node_label,
        )
        signature = f"{summary}\n{details}"
        html_text = format_graph_diagnostics_html(
            self._tr,
            diagnostics,
            node_label_for_id=_diagnostic_node_label,
            target_for_diagnostic=lambda diagnostic: (
                f"diag://{diagnostic_primary_node_id(diagnostic)}"
                if diagnostic_primary_node_id(diagnostic)
                else ""
            ),
        )
        self._graph_diagnostics_summary = summary
        self._graph_diagnostics_html = html_text
        self._update_graph_diagnostics_button(bool(diagnostics))
        if self._diagnostics_dialog is not None:
            self._diagnostics_dialog.set_diagnostics_content(summary, html_text)
        if signature == self._graph_diagnostics_signature:
            return
        self._graph_diagnostics_signature = signature
        self.graph_diagnostics_changed.emit(summary, details, bool(diagnostics))

    def export_graph_preset(self) -> dict:
        nodes = [item for item in self.scene.items() if isinstance(item, NodeItem)]
        nodes.sort(key=lambda node: (float(node.pos().x()), float(node.pos().y()), node.node_type, id(node)))

        node_ids: dict[int, str] = {}
        node_payload: list[dict] = []
        for index, node in enumerate(nodes):
            node_id = f"n{index}"
            node_ids[id(node)] = node_id
            node.properties["graph_node_id"] = node_id
            node_payload.append(
                {
                    "id": node_id,
                    "type": node.node_type,
                    "x": float(node.pos().x()),
                    "y": float(node.pos().y()),
                    "title": str(node.title),
                    "custom_title": bool(node.properties.get("custom_title", False)),
                    "properties": copy.deepcopy(node.properties or {}),
                }
            )

        edge_payload: list[dict] = []
        for edge in self._connections:
            src_id = node_ids.get(id(edge.src))
            dst_id = node_ids.get(id(edge.dst))
            if not src_id or not dst_id:
                continue
            edge_payload.append(
                {
                    "src": src_id,
                    "dst": dst_id,
                    "src_port": self._normalized_edge_src_port(edge),
                    "dst_port": self._normalized_edge_dst_port(edge),
                }
            )

        return {"nodes": node_payload, "connections": edge_payload}

    def builtin_matanyone2_preset(self) -> dict:
        return {
            "nodes": [
                {"id": "load", "type": "load", "x": -900.0, "y": -200.0, "title": "", "custom_title": False, "properties": {}},
                {"id": "sam2", "type": "sam2", "x": -500.0, "y": -300.0, "title": "", "custom_title": False, "properties": {}},
                {"id": "matting", "type": "matting", "x": -100.0, "y": -200.0, "title": "", "custom_title": False, "properties": {}},
                {"id": "write_fg", "type": "export", "x": 300.0, "y": -350.0, "title": "", "custom_title": False, "properties": {}},
                {"id": "write_alpha", "type": "export", "x": 300.0, "y": -50.0, "title": "", "custom_title": False, "properties": {}},
            ],
            "connections": [
                {"src": "load", "dst": "sam2", "src_port": "out", "dst_port": "img"},
                {"src": "load", "dst": "matting", "src_port": "out", "dst_port": "img"},
                {"src": "sam2", "dst": "matting", "src_port": "out", "dst_port": "mask"},
                {"src": "matting", "dst": "write_fg", "src_port": "fg", "dst_port": "in"},
                {"src": "matting", "dst": "write_alpha", "src_port": "alpha", "dst_port": "in"},
            ],
        }

    def builtin_corridorkey_gvm_preset(self) -> dict:
        return {
            "nodes": [
                {"id": "source", "type": "source", "x": -1000.0, "y": -240.0, "title": "", "custom_title": False, "properties": {}},
                {"id": "gvm", "type": "gvm", "x": -620.0, "y": -80.0, "title": "", "custom_title": False, "properties": {}},
                {"id": "corridorkey", "type": "corridorkey", "x": -220.0, "y": -240.0, "title": "", "custom_title": False, "properties": {}},
                {"id": "write_alpha", "type": "export", "x": 120.0, "y": -330.0, "title": "Alpha Master", "custom_title": True, "properties": {}},
                {"id": "write_fg", "type": "export", "x": 120.0, "y": -190.0, "title": "FG Clean Master", "custom_title": True, "properties": {}},
                {"id": "write_preview", "type": "export", "x": 120.0, "y": -50.0, "title": "Preview Only", "custom_title": True, "properties": {}},
            ],
            "connections": [
                {"src": "source", "dst": "gvm", "src_port": "out", "dst_port": "image"},
                {"src": "source", "dst": "corridorkey", "src_port": "out", "dst_port": "image"},
                {"src": "gvm", "dst": "corridorkey", "src_port": "alpha", "dst_port": "alphahint"},
                {"src": "corridorkey", "dst": "write_alpha", "src_port": "alpha", "dst_port": "in"},
                {"src": "corridorkey", "dst": "write_fg", "src_port": "fg", "dst_port": "in"},
                {"src": "corridorkey", "dst": "write_preview", "src_port": "comp", "dst_port": "in"},
            ],
        }

    def apply_graph_preset(self, preset: dict) -> bool:
        if not isinstance(preset, dict):
            return False

        nodes_data = preset.get("nodes")
        edges_data = preset.get("connections")
        if not isinstance(nodes_data, list) or not isinstance(edges_data, list):
            return False

        self.clear_graph()
        created: dict[str, NodeItem] = {}

        for node_data in nodes_data:
            if not isinstance(node_data, dict):
                continue
            node_type = str(node_data.get("type", "")).strip()
            node_id = str(node_data.get("id", "")).strip()
            if not node_type or not node_id:
                continue

            node = self._make_node(node_type)
            node.setPos(float(node_data.get("x", 0.0)), float(node_data.get("y", 0.0)))

            props = copy.deepcopy(node.properties or {})
            src_props = node_data.get("properties", {})
            if isinstance(src_props, dict):
                props.update(copy.deepcopy(src_props))
            props["graph_node_id"] = node_id
            node.properties = props

            custom_title = bool(node_data.get("custom_title", False))
            node.properties["custom_title"] = custom_title
            if custom_title:
                custom_title_text = str(node_data.get("title", "")).strip()
                if custom_title_text:
                    node.title = custom_title_text

            if node.node_type in {"sam2", "sam3"}:
                node.set_annotation_lines(self._build_sam_annotation(node.properties, node_type=node.node_type))
            if node.node_type == "birefnet":
                node.set_annotation_lines(
                    self._build_birefnet_annotation(
                        node.properties,
                        self._birefnet_runtime_percent,
                        self._birefnet_runtime_text,
                    )
                )
            if node.node_type == "corridorkey":
                node.set_annotation_lines(self._build_corridorkey_annotation(node.properties))
            if node.node_type in {"load", "source", "alpha"}:
                self._refresh_load_node_preview(node)
            self.scene.addItem(node)
            created[node_id] = node

        for edge_data in edges_data:
            if not isinstance(edge_data, dict):
                continue
            src = created.get(str(edge_data.get("src", "")))
            dst = created.get(str(edge_data.get("dst", "")))
            if src is None or dst is None:
                continue
            self._add_connection(
                src,
                dst,
                str(edge_data.get("src_port", "out")),
                str(edge_data.get("dst_port", "")),
            )

        self.clear_active_selection()
        self.reset_view()
        return True

    @staticmethod
    def _distance(a, b) -> float:
        return math.hypot(a.x() - b.x(), a.y() - b.y())

    def _find_node_at(self, scene_pos) -> NodeItem | None:
        for item in self.scene.items(scene_pos):
            if isinstance(item, NodeItem):
                return item
        return None

    def _find_connection_at(self, scene_pos) -> ConnectionItem | None:
        for item in self.scene.items(scene_pos):
            if isinstance(item, ConnectionItem):
                return item
        return None

    def _find_output_port_at(self, node: NodeItem, scene_pos) -> str | None:
        for port in node.output_ports:
            anchor = node.output_anchor(port.name)
            if self._distance(anchor, scene_pos) <= 22.0:
                return port.name
        return None

    def _find_input_port_at(self, node: NodeItem, scene_pos) -> str | None:
        for port in node.input_ports:
            anchor = node.input_anchor(port.name)
            if self._distance(anchor, scene_pos) <= 22.0:
                return port.name
        return None

    def _ports_compatible(self, src_node: NodeItem, src_port: str, dst_node: NodeItem, dst_port: str) -> bool:
        registry = get_registry()
        if not registry.can_connect_topology(
            src_node.node_type,
            src_port,
            dst_node.node_type,
            dst_port,
        ):
            return False
        return registry.can_connect_ports(
            src_node.node_type,
            src_port,
            dst_node.node_type,
            dst_port,
        )

    def _find_compatible_input_port(self, src_node: NodeItem, src_port: str, dst_node: NodeItem, scene_pos) -> str | None:
        compatible: list[str] = []
        for port in dst_node.input_ports:
            if not self._ports_compatible(src_node, src_port, dst_node, port.name):
                continue
            if any(e.dst is dst_node and e.dst_port == port.name for e in self._connections):
                continue
            compatible.append(port.name)
        if not compatible:
            return None
        if len(compatible) == 1:
            return compatible[0]
        best, best_d = None, float("inf")
        for pn in compatible:
            d = self._distance(dst_node.input_anchor(pn), scene_pos)
            if d < best_d:
                best_d, best = d, pn
        return best

    def _find_compatible_output_port(self, dst_node: NodeItem, dst_port: str, src_node: NodeItem, scene_pos) -> str | None:
        compatible: list[str] = []
        for port in src_node.output_ports:
            if not self._ports_compatible(src_node, port.name, dst_node, dst_port):
                continue
            compatible.append(port.name)
        if not compatible:
            return None
        if len(compatible) == 1:
            return compatible[0]
        best, best_d = None, float("inf")
        for pn in compatible:
            d = self._distance(src_node.output_anchor(pn), scene_pos)
            if d < best_d:
                best_d, best = d, pn
        return best

    def _find_best_input_snap(self, src_node: NodeItem, src_port: str, scene_pos, max_distance: float) -> tuple[NodeItem | None, str | None]:
        best_node: NodeItem | None = None
        best_port: str | None = None
        best_d = max_distance
        for item in self.scene.items():
            if not isinstance(item, NodeItem):
                continue
            if item is src_node:
                continue
            candidate_port = self._find_compatible_input_port(src_node, src_port, item, scene_pos)
            if not candidate_port:
                continue
            d = self._distance(item.input_anchor(candidate_port), scene_pos)
            if d <= best_d:
                best_d = d
                best_node = item
                best_port = candidate_port
        return best_node, best_port

    def _find_best_output_snap(self, dst_node: NodeItem, dst_port: str, scene_pos, max_distance: float) -> tuple[NodeItem | None, str | None]:
        best_node: NodeItem | None = None
        best_port: str | None = None
        best_d = max_distance
        for item in self.scene.items():
            if not isinstance(item, NodeItem):
                continue
            if item is dst_node:
                continue
            candidate_port = self._find_compatible_output_port(dst_node, dst_port, item, scene_pos)
            if not candidate_port:
                continue
            d = self._distance(item.output_anchor(candidate_port), scene_pos)
            if d <= best_d:
                best_d = d
                best_node = item
                best_port = candidate_port
        return best_node, best_port

    @staticmethod
    def _normalized_edge_dst_port(edge: "ConnectionItem") -> str:
        dst_port = str(edge.dst_port or "").strip().lower()
        if edge.dst.node_type == "export" and not dst_port:
            return "in"
        return dst_port

    @staticmethod
    def _normalized_edge_src_port(edge: "ConnectionItem") -> str:
        src_port = str(edge.src_port or "").strip().lower()
        if src_port:
            return src_port
        outputs = list(getattr(edge.src, "output_ports", []) or [])
        if len(outputs) == 1:
            return str(getattr(outputs[0], "name", "") or "out").strip().lower() or "out"
        return "out"

    @staticmethod
    def _normalized_src_port_for_node(node: NodeItem, src_port: str) -> str:
        normalized = str(src_port or "").strip().lower()
        if normalized:
            return normalized
        outputs = list(getattr(node, "output_ports", []) or [])
        if len(outputs) == 1:
            return str(getattr(outputs[0], "name", "") or "out").strip().lower() or "out"
        return "out"

    def _incoming_edges(self, node: NodeItem) -> list[ConnectionItem]:
        return [edge for edge in self._connections if edge.dst is node]

    def _write_input_stream(self, write_node: NodeItem) -> str:
        if write_node.node_type != "export":
            return ""
        incoming = self._incoming_edges(write_node)
        for edge in incoming:
            if self._normalized_edge_dst_port(edge) != "in":
                continue
            return self._resolve_connected_write_stream(edge)
        return ""

    @staticmethod
    def _resolve_connected_write_stream(edge: "ConnectionItem") -> str:
        port_name = NodeGraphDialog._normalized_edge_src_port(edge)
        port_type = ""
        port_label = ""
        for port in getattr(edge.src, "output_ports", []) or []:
            if str(getattr(port, "name", "") or "").strip().lower() != port_name:
                continue
            port_type = str(getattr(port, "type", "") or "")
            port_label = str(getattr(port, "label", "") or "")
            break
        return normalize_write_stream_name(
            source_node_type=str(getattr(edge.src, "node_type", "") or ""),
            source_port=port_name,
            port_type=port_type,
            port_label=port_label,
        )

    def clear_write_runtime_previews(self) -> None:
        for item in self.scene.items():
            if isinstance(item, NodeItem) and item.node_type == "export":
                item.set_preview_pixmap(None)

    def set_write_runtime_preview_for_node(self, node_id: str, image: QImage | None) -> None:
        target_id = str(node_id or "").strip()
        if not target_id:
            return

        pixmap = QPixmap.fromImage(image) if image is not None and not image.isNull() else None
        for item in self.scene.items():
            if not isinstance(item, NodeItem) or item.node_type != "export":
                continue
            item_gid = str(item.properties.get("graph_node_id", "")).strip()
            if item_gid != target_id:
                continue
            item.set_preview_pixmap(pixmap)
            return

    def set_write_last_output_path(self, node_id: str, path: str) -> None:
        target_id = str(node_id or "").strip()
        if not target_id:
            return

        value = str(path or "").strip()
        for item in self.scene.items():
            if not isinstance(item, NodeItem) or item.node_type != "export":
                continue
            item_gid = str(item.properties.get("graph_node_id", "")).strip()
            if item_gid != target_id:
                continue
            if value:
                item.properties["last_output_path"] = value
            else:
                item.properties.pop("last_output_path", None)
            return

    def write_node_ids_for_stream(self, stream: str) -> list[str]:
        stream_key = str(stream or "").strip().lower()
        if not stream_key:
            return []

        result: list[str] = []
        for item in self.scene.items():
            if not isinstance(item, NodeItem) or item.node_type != "export":
                continue
            if self._write_input_stream(item) != stream_key:
                continue
            node_id = str(item.properties.get("graph_node_id", "")).strip()
            if node_id:
                result.append(node_id)
        return result

    def has_connected_write_sink(self) -> bool:
        """True when any node output is connected to Write input."""
        for edge in self._connections:
            if edge.dst.node_type == "export" and self._normalized_edge_dst_port(edge) == "in":
                return True
        return False

    def connected_write_targets(self) -> list[dict]:
        """Return per-Write-node stream configs.

        Each dict contains:
            stream       str    — source port name (used as subdirectory name)
            source_node_type str
            source_port  str
            output_format  str
            auto_output_dir bool
            output_dir   str
            file_name    str
        """
        nodes: dict[int, dict] = {}
        for edge in self._connections:
            if edge.dst.node_type != "export" or self._normalized_edge_dst_port(edge) != "in":
                continue
            stream = self._resolve_connected_write_stream(edge)
            key = id(edge.dst)
            if key not in nodes:
                props = edge.dst.properties or {}
                resolved_dir, _default_name = self._resolve_write_context(edge.dst)
                auto_output_dir = bool(props.get("auto_output_dir", True))
                custom_output_dir = str(props.get("output_dir", ""))
                base_dir = resolved_dir if auto_output_dir or not custom_output_dir.strip() else custom_output_dir
                source_title = str(edge.src.title or edge.src.node_type).strip()
                source_port = self._normalized_edge_src_port(edge)
                port_label = source_port.capitalize()
                source_spec = get_node_spec(edge.src.node_type)
                if source_spec is not None:
                    source_port_spec = next((p for p in source_spec.outputs if p.name == source_port), None)
                    if source_port_spec is not None and source_port_spec.label:
                        port_label = str(source_port_spec.label)
                resolved_output_dir = ""
                if base_dir:
                    resolved_output_dir = str(
                        build_graph_write_output_dir(
                            Path(base_dir),
                            source_node_title=source_title,
                            port_label=port_label,
                            stream_label=stream,
                        )
                    )
                nodes[key] = {
                    "graph_node_id":   str(props.get("graph_node_id", "")),
                    "stream": stream,
                    "source_node_type": edge.src.node_type,
                    "source_path": str((edge.src.properties or {}).get("path", "")),
                    "source_port": edge.src_port,
                    "source_node_title": source_title,
                    "port_label": port_label,
                    "output_format":   str(props.get("output_format", "source")),
                    "auto_output_dir": auto_output_dir,
                    "output_dir":      custom_output_dir,
                    "resolved_output_dir": resolved_output_dir,
                    "file_name":       str(props.get("file_name", "")),
                    "last_output_path": str(props.get("last_output_path", "")),
                }
        return list(nodes.values())

    def _has_path(self, start: NodeItem, target: NodeItem) -> bool:
        stack = [start]
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            if node is target:
                return True
            node_id = id(node)
            if node_id in seen:
                continue
            seen.add(node_id)
            for edge in self._connections:
                if edge.src is node:
                    stack.append(edge.dst)
        return False

    def _can_connect(self, src: NodeItem, dst: NodeItem, src_port: str = "out", dst_port: str = "") -> bool:
        src_port = self._normalized_src_port_for_node(src, src_port)
        if dst.node_type == "export" and not str(dst_port or "").strip():
            dst_port = "in"
        if src is dst:
            return False
        if any(e.src is src and e.dst is dst and e.src_port == src_port and e.dst_port == dst_port
               for e in self._connections):
            return False
        if not self._ports_compatible(src, src_port, dst, dst_port):
            return False
        # One connection per input port.
        if any(e.dst is dst and e.dst_port == dst_port for e in self._connections):
            return False
        if self._has_path(dst, src):
            return False
        return True

    def _add_connection(self, src: NodeItem, dst: NodeItem, src_port: str = "out", dst_port: str = "") -> None:
        src_port = self._normalized_src_port_for_node(src, src_port)
        if dst.node_type == "export" and not str(dst_port or "").strip():
            dst_port = "in"
        if not self._can_connect(src, dst, src_port, dst_port):
            return
        edge = ConnectionItem(src, dst, src_port, dst_port)
        self.scene.addItem(edge)
        self._connections.append(edge)
        if src.node_type == "corridorkey" or dst.node_type == "corridorkey":
            self.corridorkey_props_panel._refresh_download_button_state()
        if src.node_type == "matting":
            self._update_matting_output_indicator(src)
        if dst.node_type == "matting":
            self._auto_select_eval_preset_for_matting_node(dst)
        elif dst.node_type == "export":
            self._apply_soft_write_recommendation(dst, src, src_port)
            self._refresh_write_panel_info(dst)
        self._schedule_graph_diagnostics_refresh()

    def _remove_connection(self, edge: ConnectionItem) -> None:
        src = edge.src
        dst = edge.dst
        edge.detach()
        self.scene.removeItem(edge)
        if edge in self._connections:
            self._connections.remove(edge)
        if src.node_type == "matting":
            self._update_matting_output_indicator(src)
        if dst.node_type == "export":
            self._refresh_write_panel_info(dst)
        self._schedule_graph_diagnostics_refresh()

    def _remove_node(self, node: NodeItem) -> None:
        # Remove all linked edges first to avoid dangling references.
        for edge in list(node._connections):
            self._remove_connection(edge)
        if self._active_node is node:
            self._active_node = None
            self._set_properties_enabled(False)
        self.scene.removeItem(node)

    def _remove_temp_drag_edge(self) -> None:
        if self._drag_temp_edge is not None:
            self.scene.removeItem(self._drag_temp_edge)
            self._drag_temp_edge = None

    def _clear_port_highlights(self) -> None:
        for item in self.scene.items():
            if isinstance(item, NodeItem):
                item.set_input_highlight(None)
                item.set_output_highlight(None)

    def _highlight_candidate(self, scene_pos) -> None:
        self._clear_port_highlights()
        if self._drag_mode == "from_output" and self._drag_source_node is not None and self._drag_source_port:
            node, port = self._find_best_input_snap(
                self._drag_source_node,
                self._drag_source_port,
                scene_pos,
                self._port_snap_radius,
            )
            if node is not None and port:
                node.set_input_highlight(port)
        elif self._drag_mode == "to_input" and self._drag_target_node is not None and self._drag_target_port:
            node, port = self._find_best_output_snap(
                self._drag_target_node,
                self._drag_target_port,
                scene_pos,
                self._port_snap_radius,
            )
            if node is not None and port:
                node.set_output_highlight(port)

    def is_connection_drag_active(self) -> bool:
        return self._drag_mode is not None

    def _drag_edge_color(self, port_data_type: str = "") -> str:
        return EDGE_COLORS.get(port_data_type, DEFAULT_EDGE_COLOR)

    def _port_data_type(self, node: NodeItem, port_name: str, is_output: bool) -> str:
        spec = get_node_spec(node.node_type) if node else None
        if not spec:
            return ""
        ports = spec.outputs if is_output else spec.inputs
        ps = next((p for p in ports if p.name == port_name), None)
        return ps.data_type if ps else ""

    def try_start_connection_drag(self, scene_pos) -> bool:
        self._reconnect_backup = None

        node = self._find_node_at(scene_pos)
        if node is not None:
            port_name = self._find_output_port_at(node, scene_pos)
            if port_name is not None:
                self._drag_mode = "from_output"
                self._drag_source_node = node
                self._drag_source_port = port_name
                self._drag_target_node = None
                self._drag_target_port = None
                dt = self._port_data_type(node, port_name, True)
                self._drag_temp_edge = QGraphicsPathItem()
                self._drag_temp_edge.setZValue(-1)
                self._drag_temp_edge.setPen(QPen(QColor(self._drag_edge_color(dt)), 2.2, Qt.PenStyle.DashLine))
                self.scene.addItem(self._drag_temp_edge)
                self.update_connection_drag(scene_pos)
                return True

        edge = self._find_connection_at(scene_pos)
        if edge is not None:
            src_hit = self._find_output_port_at(edge.src, scene_pos)
            dst_hit = self._find_input_port_at(edge.dst, scene_pos)
            if src_hit == edge.src_port:
                self._drag_mode = "from_output"
                self._drag_source_node = edge.src
                self._drag_source_port = edge.src_port
                self._drag_target_node = None
                self._drag_target_port = None
                self._reconnect_backup = (edge.src, edge.dst, edge.src_port, edge.dst_port)
                dt = self._port_data_type(edge.src, edge.src_port, True)
                self._remove_connection(edge)
            elif dst_hit == edge.dst_port:
                self._drag_mode = "to_input"
                self._drag_target_node = edge.dst
                self._drag_target_port = edge.dst_port
                self._drag_source_node = None
                self._drag_source_port = None
                self._reconnect_backup = (edge.src, edge.dst, edge.src_port, edge.dst_port)
                dt = self._port_data_type(edge.dst, edge.dst_port, False)
                self._remove_connection(edge)
            else:
                return False

            self._drag_temp_edge = QGraphicsPathItem()
            self._drag_temp_edge.setZValue(-1)
            self._drag_temp_edge.setPen(QPen(QColor(self._drag_edge_color(dt)), 2.2, Qt.PenStyle.DashLine))
            self.scene.addItem(self._drag_temp_edge)
            self.update_connection_drag(scene_pos)
            return True

        return False

    def update_connection_drag(self, scene_pos) -> None:
        if self._drag_mode is None or self._drag_temp_edge is None:
            return

        if self._drag_mode == "from_output" and self._drag_source_node is not None:
            p1 = self._drag_source_node.output_anchor(self._drag_source_port or "out")
            p2 = scene_pos
        elif self._drag_mode == "to_input" and self._drag_target_node is not None:
            p1 = scene_pos
            p2 = self._drag_target_node.input_anchor(self._drag_target_port or "")
        else:
            return

        dx = max(80.0, abs(p2.x() - p1.x()) * 0.45)
        path = QPainterPath(p1)
        path.cubicTo(p1.x() + dx, p1.y(), p2.x() - dx, p2.y(), p2.x(), p2.y())
        self._drag_temp_edge.setPath(path)
        self._highlight_candidate(scene_pos)

    def finish_connection_drag(self, scene_pos) -> None:
        mode = self._drag_mode
        src = self._drag_source_node
        dst = self._drag_target_node
        src_port = self._drag_source_port
        dst_port = self._drag_target_port
        self._drag_mode = None
        self._drag_source_node = None
        self._drag_target_node = None
        self._drag_source_port = None
        self._drag_target_port = None

        created = False

        if mode is None:
            self._remove_temp_drag_edge()
            self._clear_port_highlights()
            return

        if mode == "from_output" and src is not None and src_port:
            target, compatible = self._find_best_input_snap(src, src_port, scene_pos, self._port_snap_radius)
            if target is None or not compatible:
                target = self._find_node_at(scene_pos)
                if target is not None:
                    compatible = self._find_compatible_input_port(src, src_port, target, scene_pos)
            if target is not None and compatible and self._can_connect(src, target, src_port, compatible):
                self._add_connection(src, target, src_port, compatible)
                created = True
        elif mode == "to_input" and dst is not None and dst_port:
            source, compatible = self._find_best_output_snap(dst, dst_port, scene_pos, self._port_snap_radius)
            if source is None or not compatible:
                source = self._find_node_at(scene_pos)
                if source is not None:
                    compatible = self._find_compatible_output_port(dst, dst_port, source, scene_pos)
            if source is not None and compatible and self._can_connect(source, dst, compatible, dst_port):
                self._add_connection(source, dst, compatible, dst_port)
                created = True

        if not created and self._reconnect_backup is not None:
            bsrc, bdst, bsrc_port, bdst_port = self._reconnect_backup
            if self._can_connect(bsrc, bdst, bsrc_port, bdst_port):
                self._add_connection(bsrc, bdst, bsrc_port, bdst_port)
            else:
                edge = ConnectionItem(bsrc, bdst, bsrc_port, bdst_port)
                self.scene.addItem(edge)
                self._connections.append(edge)

        self._reconnect_backup = None
        self._remove_temp_drag_edge()
        self._clear_port_highlights()

    def _update_all_connections(self) -> None:
        for edge in list(self._connections):
            edge.update_path()

    def _spec_texts(self, key: str) -> tuple[str, str]:
        spec = get_node_spec(key)
        if spec is None:
            return (key, "")
        title = self._tr(spec.title_i18n_key)
        subtitle = self._tr(spec.subtitle_i18n_key) if spec.subtitle_i18n_key else spec.subtitle
        return (title, subtitle)

    def _create_node(self, key: str, scene_pos) -> None:
        title, subtitle = self._spec_texts(key)
        node = NodeItem(title, subtitle, node_type=key)
        if key == "corridorkey":
            node.title = self._corridorkey_auto_title(node.properties)
        node.setPos(scene_pos.x() - node.w / 2, scene_pos.y() - node.h / 2)
        node.properties["graph_node_id"] = self._next_auto_graph_node_id()
        if key in {"sam2", "sam3"}:
            node.set_annotation_lines(self._build_sam_annotation(node.properties, node_type=key))
        if key == "birefnet":
            node.set_annotation_lines(
                self._build_birefnet_annotation(
                    node.properties,
                    self._birefnet_runtime_percent,
                    self._birefnet_runtime_text,
                )
            )
        if key == "corridorkey":
            node.set_annotation_lines(self._build_corridorkey_annotation())
        self.scene.addItem(node)
        self._update_quick_add_recent(key)

    def _make_node(self, key: str) -> NodeItem:
        spec = get_node_spec(key)
        if spec is None:
            return NodeItem(key, "", node_type="generic")
        title, subtitle = self._spec_texts(key)
        node = NodeItem(title, subtitle, node_type=spec.key)
        if key == "corridorkey":
            node.title = self._corridorkey_auto_title(node.properties)
        if key in {"sam2", "sam3"}:
            node.set_annotation_lines(self._build_sam_annotation(node.properties, node_type=key))
        if key == "birefnet":
            node.set_annotation_lines(
                self._build_birefnet_annotation(
                    node.properties,
                    self._birefnet_runtime_percent,
                    self._birefnet_runtime_text,
                )
            )
        if key == "corridorkey":
            node.set_annotation_lines(self._build_corridorkey_annotation())
        return node

    def _load_quick_add_recent(self) -> list[str]:
        raw = self._settings.value("node_graph/quick_add_recent", [])
        if isinstance(raw, str):
            return [raw] if raw else []
        if isinstance(raw, (list, tuple)):
            return [str(v) for v in raw if str(v)]
        return []

    def _save_quick_add_recent(self) -> None:
        self._settings.setValue("node_graph/quick_add_recent", self._quick_add_recent_keys)

    def _update_quick_add_recent(self, key: str) -> None:
        if not key:
            return
        if key in self._quick_add_recent_keys:
            self._quick_add_recent_keys.remove(key)
        self._quick_add_recent_keys.insert(0, key)
        self._quick_add_recent_keys = self._quick_add_recent_keys[:8]
        self._quick_add_dialog.set_recent_keys(self._quick_add_recent_keys)
        self._save_quick_add_recent()

    def _node_spec_by_key(self, key: str) -> tuple[str, str] | None:
        spec = get_node_spec(key)
        if spec is None:
            return None
        return self._spec_texts(key)

    def _open_quick_add(self) -> None:
        cursor_global = QCursor.pos()
        self._quick_add_dialog.move(cursor_global + QPoint(8, 8))
        self._quick_add_dialog.search_edit.clear()
        self._quick_add_dialog.selected_key = None
        self._quick_add_dialog.search_edit.setFocus()
        if self._quick_add_dialog.exec() != QDialog.DialogCode.Accepted:
            return

        key = self._quick_add_dialog.selected_key
        if not key:
            return
        if self._node_spec_by_key(key) is None:
            return

        scene_pos = self.view.mapToScene(self.view.mapFromGlobal(cursor_global))
        self._create_node(key, scene_pos)

    def open_quick_add_popup(self) -> None:
        """Public entry point used by both dialog and view key handlers."""
        self._open_quick_add()

    def group_selected_nodes(self) -> None:
        nodes = [item for item in self.scene.selectedItems() if isinstance(item, NodeItem)]
        if len(nodes) < 2:
            return
        group = self.scene.createItemGroup(nodes)
        group.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        group.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        group.setZValue(0.5)
        self._groups.append(group)

    def ungroup_selected_groups(self) -> None:
        groups = [item for item in self.scene.selectedItems() if isinstance(item, QGraphicsItemGroup)]
        for group in groups:
            self.scene.destroyItemGroup(group)
            if group in self._groups:
                self._groups.remove(group)

    def show_context_menu(self, global_pos, scene_pos) -> None:
        menu = QMenu(self)
        add_menu = menu.addMenu(self._tr("node_graph_add_node"))

        preferred_order = {
            "source": 0,
            "load": 1,
            "alpha": 2,
            "sam2": 3,
            "matting": 4,
            "birefnet": 5,
            "corridorkey": 6,
            "export": 7,
        }
        specs = sorted(
            list_node_specs(),
            key=lambda s: (preferred_order.get(s.key, 100), s.key),
        )
        add_actions: dict[object, str] = {}
        for spec in specs:
            title = self._tr(spec.title_i18n_key) if spec.title_i18n_key else spec.title
            action = add_menu.addAction(title)
            add_actions[action] = spec.key

        menu.addSeparator()
        act_group = menu.addAction(self._tr("node_graph_group"))
        act_ungroup = menu.addAction(self._tr("node_graph_ungroup"))

        chosen = menu.exec(global_pos)
        if chosen in add_actions:
            self._create_node(add_actions[chosen], scene_pos)
        elif chosen == act_group:
            self.group_selected_nodes()
        elif chosen == act_ungroup:
            self.ungroup_selected_groups()

    def _emit_read_media_selected(self, node: NodeItem) -> None:
        if node.node_type not in {"load", "source", "alpha"}:
            return
        media_path = str(node.properties.get("path", "")).strip()
        media_type = str(node.properties.get("media_type", "video")).strip().lower()
        self.read_media_selected.emit(media_path, media_type)

    def _emit_preview_request(self, node: NodeItem | None) -> None:
        if node is None:
            self.preview_request_changed.emit("", None)
            return

        self._ensure_graph_node_id(node)
        payload = dict(node.properties or {})
        if node.node_type == "export":
            payload["stream"] = self._write_input_stream(node)
        elif node.node_type == "birefnet":
            payload["graph_node_id"] = str(node.properties.get("graph_node_id", "")).strip()
            payload["_downstream_write_ids"] = self._downstream_write_ids(node)
        elif node.node_type == "merge":
            payload["graph_node_id"] = str(node.properties.get("graph_node_id", "")).strip()
            payload["_quick_preview"] = {
                "fg": self._upstream_read_node_payload(node, "fg"),
                "bg": self._upstream_read_node_payload(node, "bg"),
                "mask": self._upstream_read_node_payload(node, "mask"),
            }
        elif node.node_type == "corridorkey":
            payload["_downstream_write_ids"] = self._downstream_write_ids(node)
        self.preview_request_changed.emit(node.node_type, payload)
        if node.node_type in {"load", "source", "alpha"}:
            self._emit_read_media_selected(node)

    def _upstream_read_node_payload(self, node: NodeItem, dst_port: str) -> dict | None:
        edge = next((e for e in self._connections if e.dst is node and e.dst_port == dst_port), None)
        if edge is None:
            return None
        src = edge.src
        if src.node_type not in {"source", "load", "alpha"}:
            return None
        props = src.properties or {}
        return {
            "node_type": str(src.node_type),
            "path": str(props.get("path", "")).strip(),
            "media_type": str(props.get("media_type", "video")).strip().lower(),
        }

    def _downstream_write_ids(self, source: NodeItem) -> list[str]:
        """Return graph_node_ids of Write nodes reachable from *source* output."""
        visited: set[int] = set()
        stack = [source]
        result: list[str] = []
        while stack:
            node = stack.pop()
            nid = id(node)
            if nid in visited:
                continue
            visited.add(nid)
            for edge in self._connections:
                if edge.src is not node:
                    continue
                dst = edge.dst
                if dst.node_type == "export":
                    gid = str(dst.properties.get("graph_node_id", "")).strip()
                    if gid and gid not in result:
                        result.append(gid)
                else:
                    stack.append(dst)
        return result

    def _sam_node_target(self) -> NodeItem | None:
        if self._active_node is not None and self._active_node.node_type in {"sam2"}:
            return self._active_node
        for item in self.scene.items():
            if isinstance(item, NodeItem) and item.node_type in {"sam2"}:
                return item
        return None

    def _sam3_node_target(self) -> NodeItem | None:
        if self._active_node is not None and self._active_node.node_type == "sam3":
            return self._active_node
        for item in self.scene.items():
            if isinstance(item, NodeItem) and item.node_type == "sam3":
                return item
        return None

    def active_sam3_concept(self) -> str:
        node = self._sam3_node_target()
        if node is None or node is not self._active_node:
            return ""
        self.sam3_props_panel.write_to_properties(node.properties)
        return str(node.properties.get("concept", "") or "").strip()

    def sync_sam_runtime_state(
        self,
        *,
        status_text: str | None = None,
        backend: str | None = None,
        model_type: str | None = None,
        point_mode: str | None = None,
        live_sam2: bool | None = None,
        mask_items: list[str] | None = None,
        selected_mask_rows: list[int] | None = None,
        current_mask_ready: bool | None = None,
        mask_sequence_count: int | None = None,
        mask_source_path: str | None = None,
        mask_payloads: list[dict] | None = None,
    ) -> None:
        node = self._sam_node_target()
        if node is None:
            return

        props = node.properties
        if status_text is not None:
            props["sam_status"] = status_text
        if backend is not None:
            props["backend"] = backend
        if model_type is not None:
            props["model_type"] = model_type
        if point_mode is not None:
            props["point_mode"] = point_mode
        if live_sam2 is not None:
            props["live_sam2"] = live_sam2
        if mask_items is not None:
            props["mask_items"] = list(mask_items)
        if selected_mask_rows is not None:
            props["selected_mask_rows"] = list(selected_mask_rows)
        if current_mask_ready is not None:
            props["current_mask_ready"] = current_mask_ready
        if mask_sequence_count is not None:
            props["mask_sequence_count"] = int(mask_sequence_count)
        if mask_source_path is not None:
            props["_mask_source_path"] = mask_source_path
        if mask_payloads is not None:
            props["mask_payloads"] = copy.deepcopy(mask_payloads)

        # Update Nuke-style annotation on the SAM2 node (internal key: sam2)
        node.set_annotation_lines(self._build_sam_annotation(props))

        if node is self._active_node and node.node_type in {"sam2"}:
            rows = []
            for value in props.get("selected_mask_rows", []):
                try:
                    rows.append(int(value))
                except (TypeError, ValueError):
                    continue
            self._updating_properties = True
            try:
                self.sam_props_panel.load_from_properties(props)
                self.sam_props_panel.masks_list.clearSelection()
                for row in rows:
                    item = self.sam_props_panel.masks_list.item(row)
                    if item is not None:
                        item.setSelected(True)
            finally:
                self._updating_properties = False

    def sync_sam3_prompt_state(
        self,
        *,
        prompt_points: list[list[int]] | None = None,
        prompt_labels: list[int] | None = None,
        status_text: str | None = None,
        point_mode: str | None = None,
        live_sam2: bool | None = None,
    ) -> None:
        node = self._sam3_node_target()
        if node is None:
            return

        props = node.properties
        if prompt_points is not None:
            props["prompt_points"] = copy.deepcopy(prompt_points)
        if prompt_labels is not None:
            props["prompt_labels"] = [int(v) for v in prompt_labels]
        if status_text is not None:
            props["sam_status"] = status_text
        if point_mode is not None:
            props["point_mode"] = point_mode
        if live_sam2 is not None:
            props["live_sam2"] = bool(live_sam2)

        node.set_annotation_lines(self._build_sam_annotation(props, node_type="sam3"))

        if node is self._active_node and node.node_type == "sam3":
            self._updating_properties = True
            try:
                self.sam3_props_panel.load_from_properties(props)
            finally:
                self._updating_properties = False

    def selected_sam_mask_rows(self) -> list[int]:
        node = self._sam_node_target()
        if node is None:
            return []
        if int(node.properties.get("mask_sequence_count", 0) or 0) > 1:
            return []
        result: list[int] = []
        for value in node.properties.get("selected_mask_rows", []):
            try:
                result.append(int(value))
            except (TypeError, ValueError):
                continue
        return result

    def sam_node_mask_source_path(self) -> str:
        """Return the persistent mask file path stored in the SAM graph node's properties."""
        node = self._sam_node_target()
        if node is None:
            return ""
        return str(node.properties.get("_mask_source_path", ""))

    def sam_node_mask_payloads(self) -> list[dict]:
        """Return persisted SAM mask payloads from graph-node properties."""
        node = self._sam_node_target()
        if node is None:
            return []
        raw = node.properties.get("mask_payloads", [])
        return copy.deepcopy(raw) if isinstance(raw, list) else []

    def _delete_selected_graph_items(self) -> bool:
        selected_items = list(self.scene.selectedItems())
        selected_edges = [item for item in selected_items if isinstance(item, ConnectionItem)]
        selected_nodes = [item for item in selected_items if isinstance(item, NodeItem)]

        # If a group is selected, also collect node children for deletion.
        for item in selected_items:
            if isinstance(item, QGraphicsItemGroup):
                for child in item.childItems():
                    if isinstance(child, NodeItem) and child not in selected_nodes:
                        selected_nodes.append(child)

        if selected_edges:
            for edge in selected_edges:
                self._remove_connection(edge)

        if selected_nodes:
            for node in selected_nodes:
                self._remove_node(node)

        return bool(selected_edges or selected_nodes)

    def handle_graph_key_event(self, event) -> bool:
        return handle_node_graph_hotkeys(
            event,
            open_quick_add_popup=self.open_quick_add_popup,
            reset_view=self.reset_view,
            is_connection_drag_active=self.is_connection_drag_active,
            cancel_connection_drag=self._cancel_connection_drag,
            group_selected_nodes=self.group_selected_nodes,
            ungroup_selected_groups=self.ungroup_selected_groups,
            delete_selected_items=self._delete_selected_graph_items,
        )

    def _cancel_connection_drag(self) -> None:
        self._drag_mode = None
        self._drag_source_node = None
        self._drag_target_node = None
        self._drag_source_port = None
        self._drag_target_port = None
        self._reconnect_backup = None
        self._remove_temp_drag_edge()
        self._clear_port_highlights()

    def _build_birefnet_annotation(
        self,
        props: dict,
        runtime_percent: int | None = None,
        runtime_text: str = "",
    ) -> list[tuple[str, str]]:
        usage = str(props.get("usage", "General")).strip() or "General"
        lines: list[tuple[str, str]] = [(usage, "#7fd4ff")]

        text = str(runtime_text or "").strip()
        if runtime_percent is not None and "birefnet" in text.lower():
            lines.append((f"{max(0, min(100, int(runtime_percent)))}%", "#9fb2c8"))

        if self._birefnet_frame_current is not None and self._birefnet_frame_total is not None:
            lines.append((f"{self._birefnet_frame_current}/{self._birefnet_frame_total}", "#c8d8e4"))

        return lines

    def _build_matting_annotation(self, node: NodeItem) -> list[tuple[str, str]]:
        preset_name = str((node.properties or {}).get("preset", "Custom")).strip() or "Custom"
        preset_key = PRESET_LABEL_KEYS.get(preset_name)
        preset_label = self._tr(preset_key) if preset_key else preset_name
        profile_prefix = self._tr("node_graph_matting_active_profile")
        lines: list[tuple[str, str]] = [
            (f"{profile_prefix}: {preset_label}", "#7fd4ff"),
        ]
        # Frame counter
        if self._matting_frame_current is not None and self._matting_frame_total is not None:
            lines.append((f"{self._matting_frame_current}/{self._matting_frame_total}", "#c8d8e4"))
        return lines

    def _refresh_matting_annotations(self) -> None:
        for item in self.scene.items():
            if not isinstance(item, NodeItem) or item.node_type != "matting":
                continue
            item.set_annotation_lines(self._build_matting_annotation(item))

    def _refresh_sam_annotations(self) -> None:
        for item in self.scene.items():
            if not isinstance(item, NodeItem) or item.node_type not in {"sam2", "sam3"}:
                continue
            item.set_annotation_lines(self._build_sam_annotation(item.properties, node_type=item.node_type))

    def _build_gvm_annotation(self) -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = []
        if self._gvm_frame_current is not None and self._gvm_frame_total is not None:
            lines.append((f"{self._gvm_frame_current}/{self._gvm_frame_total}", "#c8d8e4"))
        return lines

    def _refresh_gvm_annotations(self) -> None:
        ann = self._build_gvm_annotation()
        for item in self.scene.items():
            if not isinstance(item, NodeItem) or item.node_type != "gvm":
                continue
            item.set_annotation_lines(ann)

    def _refresh_birefnet_annotations(self) -> None:
        for item in self.scene.items():
            if not isinstance(item, NodeItem) or item.node_type != "birefnet":
                continue
            item.set_annotation_lines(
                self._build_birefnet_annotation(
                    item.properties,
                    self._birefnet_runtime_percent,
                    self._birefnet_runtime_text,
                )
            )

    def _corridorkey_mode_label(self, mode: str) -> str:
        key = {
            "auto": "corridorkey_alpha_hint_mode_auto",
            "batch": "corridorkey_alpha_hint_mode_batch",
            "staged": "corridorkey_alpha_hint_mode_staged",
        }.get(str(mode or "").strip().lower(), "corridorkey_alpha_hint_mode_auto")
        return self._tr(key)

    def _corridorkey_colorspace_label(self, colorspace: str) -> str:
        key = {
            "auto": "corridorkey_input_colorspace_auto",
            "srgb": "corridorkey_input_colorspace_srgb",
            "linear": "corridorkey_input_colorspace_linear",
        }.get(str(colorspace or "").strip().lower(), "corridorkey_input_colorspace_auto")
        return self._tr(key)

    def _corridorkey_preset_label(self, preset: str) -> str:
        key = {
            "preview": "corridorkey_preset_preview",
            "balanced": "corridorkey_preset_balanced",
            "max": "corridorkey_preset_max",
            "ultra": "corridorkey_preset_ultra",
            "glass": "corridorkey_preset_glass",
            "official": "corridorkey_preset_official",
            "custom": "corridorkey_preset_custom",
        }.get(str(preset or "").strip().lower(), "corridorkey_preset_balanced")
        return self._tr(key)

    def _corridorkey_auto_title(self, props: dict | None = None) -> str:
        base_title, _ = self._spec_texts("corridorkey")
        preset = "balanced"
        if isinstance(props, dict):
            preset = str(props.get("preset", "balanced")).strip().lower() or "balanced"
        return f"{base_title} • {self._corridorkey_preset_label(preset)}"

    def _build_corridorkey_annotation(self, props: dict | None = None) -> list[tuple[str, str]]:
        from app.services.corridorkey_service import CorridorKeyService

        lines: list[tuple[str, str]] = []
        status = CorridorKeyService.get_checkpoint_status()
        if status.get("state") == "ready":
            path = str(status.get("path") or "").strip()
            label = Path(path).stem if path else self._tr("corridorkey_annotation_ready")
            lines.append((label, "#7fd4ff"))
        else:
            lines.append((self._tr("corridorkey_annotation_missing"), "#cc2222"))

        mode_cfg = "auto"
        input_colorspace = "auto"
        if isinstance(props, dict):
            mode_cfg = str(props.get("alpha_hint_mode", "auto")).strip().lower()
            input_colorspace = str(props.get("input_colorspace", "auto")).strip().lower()
        colorspace_color = "#7fd38c" if input_colorspace == "linear" else "#9fb2c8"
        lines.append((
            f"{self._tr('corridorkey_annotation_mode_config')}: {self._corridorkey_mode_label(mode_cfg)}",
            "#9fb2c8",
        ))
        lines.append((
            f"{self._tr('corridorkey_input_colorspace')}: {self._corridorkey_colorspace_label(input_colorspace)}",
            colorspace_color,
        ))

        if self._corridorkey_runtime_requested and self._corridorkey_runtime_effective:
            requested = self._corridorkey_runtime_requested
            effective = self._corridorkey_runtime_effective
            is_fallback = requested == "staged" and effective == "batch"
            if is_fallback:
                runtime_color = "#f4b35a"
            elif effective == "staged":
                runtime_color = "#7fd38c"
            else:
                runtime_color = "#9fb2c8"

            runtime_prefix = self._tr("corridorkey_annotation_mode_runtime")
            if is_fallback:
                runtime_prefix = self._tr("corridorkey_annotation_mode_runtime_fallback")
            lines.append((
                f"{runtime_prefix}: "
                f"{self._corridorkey_mode_label(self._corridorkey_runtime_requested)}"
                f" -> {self._corridorkey_mode_label(self._corridorkey_runtime_effective)}",
                runtime_color,
            ))

        if self._corridorkey_frame_current is not None and self._corridorkey_frame_total is not None:
            lines.append((f"{self._corridorkey_frame_current}/{self._corridorkey_frame_total}", "#c8d8e4"))

        return lines

    def _refresh_corridorkey_annotations(self) -> None:
        for item in self.scene.items():
            if not isinstance(item, NodeItem) or item.node_type != "corridorkey":
                continue
            item.set_annotation_lines(self._build_corridorkey_annotation(item.properties))

    def set_corridorkey_runtime_mode(self, requested_mode: str, effective_mode: str) -> None:
        self._corridorkey_runtime_requested = str(requested_mode or "").strip().lower() or "auto"
        self._corridorkey_runtime_effective = str(effective_mode or "").strip().lower() or "batch"
        self._refresh_corridorkey_annotations()

    def clear_corridorkey_runtime_mode(self) -> None:
        self._corridorkey_runtime_requested = None
        self._corridorkey_runtime_effective = None
        self._refresh_corridorkey_annotations()

    def set_birefnet_runtime_progress(self, percent: int, status_text: str) -> None:
        self._birefnet_runtime_percent = max(0, min(100, int(percent)))
        self._birefnet_runtime_text = str(status_text or "")
        self._refresh_birefnet_annotations()

    def clear_birefnet_runtime_progress(self) -> None:
        self._birefnet_runtime_percent = None
        self._birefnet_runtime_text = ""
        self._birefnet_frame_current = None
        self._birefnet_frame_total = None
        self._refresh_birefnet_annotations()

    def set_node_frame_progress(self, node_type: str, current: int, total: int) -> None:
        if node_type == "birefnet":
            self._birefnet_frame_current = current
            self._birefnet_frame_total = total
            self._refresh_birefnet_annotations()
        elif node_type == "corridorkey":
            self._corridorkey_frame_current = current
            self._corridorkey_frame_total = total
            now = time.monotonic()
            last_frame_s = None
            if self._corridorkey_last_frame_ts is not None:
                last_frame_s = max(0.0, now - self._corridorkey_last_frame_ts)
                self._corridorkey_frame_time_count += 1
                if self._corridorkey_frame_time_avg is None:
                    self._corridorkey_frame_time_avg = last_frame_s
                else:
                    n = float(self._corridorkey_frame_time_count)
                    self._corridorkey_frame_time_avg = (
                        ((n - 1.0) * float(self._corridorkey_frame_time_avg)) + last_frame_s
                    ) / n
            self._corridorkey_last_frame_ts = now
            self._refresh_corridorkey_annotations()
        elif node_type == "matting":
            self._matting_frame_current = current
            self._matting_frame_total = total
            self._refresh_matting_annotations()
        elif node_type in {"sam2", "sam3"}:
            self._sam_frame_current = current
            self._sam_frame_total = total
            self._refresh_sam_annotations()
        elif node_type == "gvm":
            self._gvm_frame_current = current
            self._gvm_frame_total = total
            self._refresh_gvm_annotations()

    def clear_node_frame_progress(self, node_type: str | None = None) -> None:
        normalized = str(node_type or "").strip().lower()
        clear_all = not normalized

        if clear_all or normalized == "birefnet":
            self._birefnet_frame_current = None
            self._birefnet_frame_total = None
        if clear_all or normalized == "corridorkey":
            self._corridorkey_frame_current = None
            self._corridorkey_frame_total = None
            self._corridorkey_last_frame_ts = None
            self._corridorkey_frame_time_avg = None
            self._corridorkey_frame_time_count = 0
        if clear_all or normalized == "matting":
            self._matting_frame_current = None
            self._matting_frame_total = None
        if clear_all or normalized in {"sam2", "sam3"}:
            self._sam_frame_current = None
            self._sam_frame_total = None
        if clear_all or normalized == "gvm":
            self._gvm_frame_current = None
            self._gvm_frame_total = None

        self._refresh_birefnet_annotations()
        self._refresh_corridorkey_annotations()
        self._refresh_matting_annotations()
        self._refresh_sam_annotations()
        self._refresh_gvm_annotations()

    def _build_sam_annotation(self, props: dict, node_type: str = "sam2") -> list[tuple[str, str]]:
        default_color = "#7a9ab8"
        node_kind = str(node_type or "sam2").strip().lower()
        if node_kind == "sam2":
            node_kind = "sam2"
        if node_kind == "sam3":
            model_type = str(props.get("model_type", "sam3"))
            labels = {"sam3": "SAM3", "sam3.1": "SAM3.1"}
        else:
            model_type = str(props.get("model_type", "vit_h"))
            labels = {"vit_h": "SAM2 Large", "vit_l": "SAM2 Base+", "vit_b": "SAM2 Small"}
        model_name = labels.get(model_type, model_type.upper())
        lines: list[tuple[str, str]] = [(model_name, "#a0c4e8")]
        if node_kind == "sam3":
            concept = str(props.get("concept", "") or "").strip()
            if concept:
                lines.append((self._tr("sam_annotation_concept").format(concept=concept), "#d6b4ff"))
            if self._sam_frame_current is not None and self._sam_frame_total is not None:
                lines.append((f"{self._sam_frame_current}/{self._sam_frame_total}", "#c8d8e4"))
            return lines
        live = bool(props.get("live_sam2", False))
        if live:
            lines.append((self._tr("sam_annotation_live"), "#ffa544"))
        else:
            lines.append((self._tr("sam_annotation_manual"), default_color))
        masks = props.get("mask_items", [])
        n = len(masks) if isinstance(masks, list) else 0
        sequence_count = int(props.get("mask_sequence_count", 0) or 0)
        has_preview = bool(props.get("current_mask_ready", False))
        if sequence_count > 1:
            lines.append((self._tr("sam_annotation_sequence_masks").format(count=sequence_count), default_color))
        elif n:
            mask_color = default_color
            lines.append((self._tr("sam_annotation_masks").format(count=n), mask_color))
        elif has_preview:
            lines.append((self._tr("sam_annotation_mask_ready"), "#b8c87a"))
        else:
            lines.append((self._tr("sam_annotation_no_masks"), "#cc2222"))
        if self._sam_frame_current is not None and self._sam_frame_total is not None:
            lines.append((f"{self._sam_frame_current}/{self._sam_frame_total}", "#c8d8e4"))
        return lines

    def keyPressEvent(self, event) -> None:
        if self.handle_graph_key_event(event):
            return
        super().keyPressEvent(event)

    def focusNextPrevChild(self, next: bool) -> bool:
        # Ensure Tab triggers node search even when Qt tries to move focus between widgets.
        fw = self.focusWidget()
        if next and (fw is self.view or self.view.isAncestorOf(fw)):
            self.open_quick_add_popup()
            return True
        return super().focusNextPrevChild(next)

    def reset_view(self) -> None:
        bounds = self.scene.itemsBoundingRect()
        if not bounds.isValid() or bounds.isEmpty():
            return

        margin = 72.0
        framed = bounds.adjusted(-margin, -margin, margin, margin)

        self.view.resetTransform()
        self.view.fitInView(framed, Qt.AspectRatioMode.KeepAspectRatio)

        # Keep startup framing readable: avoid over-zoomed-out or over-zoomed-in view.
        cur = float(self.view.transform().m11())
        min_scale = 0.8
        max_scale = 1.8
        if cur > 0.0:
            if cur < min_scale:
                factor = min_scale / cur
                self.view.scale(factor, factor)
            elif cur > max_scale:
                factor = max_scale / cur
                self.view.scale(factor, factor)

        self.view.centerOn(bounds.center())
