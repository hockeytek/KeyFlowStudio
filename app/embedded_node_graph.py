"""Embedded node graph editor widget for main window integration (Nuke-style)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter, QPushButton
from PySide6.QtCore import Qt, QTimer, QSize, QPoint

from app.node_graph_dialog import NodeGraphDialog


class EmbeddedNodeGraphEditor(QWidget):
    """Embedded workspace that contains the graph editor (Nuke-style)."""

    def __init__(self, translate: Callable[[str], str], parent=None) -> None:
        super().__init__(parent)
        self._tr = translate
        self._dialog: NodeGraphDialog | None = None
        
        # Create main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._btn_center_view = QPushButton(self)
        self._btn_center_view.setObjectName("btn_graph_center_view")
        self._btn_center_view.setFixedSize(34, 30)
        self._btn_center_view.setText("")
        self._btn_center_view.setIcon(QIcon(str(Path(__file__).resolve().parent / "assets" / "graph-center.svg")))
        self._btn_center_view.setIconSize(QSize(18, 18))
        self._btn_center_view.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_center_view.clicked.connect(self._center_graph_view)
        self._btn_center_view.raise_()
        
        # Create a splitter to hold graph view and properties panel side-by-side
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #1a2433;
                width: 4px;
            }
            QSplitter::handle:hover {
                background-color: #2a3a4d;
            }
        """)
        
        self._splitter = splitter
        self._splitter.splitterMoved.connect(lambda _pos, _index: self._position_center_button())
        layout.addWidget(splitter)
        
        self.setStyleSheet("""
            EmbeddedNodeGraphEditor {
                background-color: #0f141c;
            }
            QPushButton#btn_graph_center_view {
                background-color: rgba(15, 26, 39, 220);
                border: 1px solid #294158;
                border-radius: 8px;
                padding: 0;
            }
            QPushButton#btn_graph_center_view:hover {
                background-color: rgba(28, 47, 68, 235);
                border: 1px solid #43c7ff;
            }
            QPushButton#btn_graph_center_view:pressed {
                background-color: rgba(12, 33, 52, 245);
            }
        """)
        self._retranslate_ui()
        QTimer.singleShot(0, self._position_center_button)
    
    def set_dialog(self, dialog: NodeGraphDialog) -> None:
        """Set the dialog and embed its components into this widget."""
        self._dialog = dialog
        
        # Remove the dialog's own dialog window frame - we're embedding it
        dialog.hide()
        
        # Add the graph view to the splitter
        self._splitter.addWidget(dialog.view)
        
        # Add the properties panel to the splitter
        self._splitter.addWidget(dialog.props_panel)
        
        # Set splitter proportions (graph gets 70%, properties gets 30%)
        self._splitter.setSizes([700, 300])
        
        # Make it collapsible
        self._splitter.setCollapsible(0, False)
        self._splitter.setCollapsible(1, True)

        # Start with no selected node so properties stay hidden until explicit selection.
        if hasattr(dialog, "clear_active_selection"):
            dialog.clear_active_selection()

        # Re-fit after embedding/layout so nodes are framed correctly on startup.
        QTimer.singleShot(0, dialog.reset_view)
        QTimer.singleShot(120, dialog.reset_view)
        QTimer.singleShot(0, self._position_center_button)
        QTimer.singleShot(120, self._position_center_button)

    def _center_graph_view(self) -> None:
        """Center and fit the graph workspace when the user loses orientation."""
        if self._dialog is not None:
            self._dialog.reset_view()

    def _retranslate_ui(self) -> None:
        self._btn_center_view.setToolTip(
            f"{self._tr('node_graph_center_view_tooltip')} (F)"
        )
        self._btn_center_view.setAccessibleName(self._tr("node_graph_center_view"))

    def _position_center_button(self) -> None:
        if self._dialog is None:
            margin = 15
            x = max(margin, self.width() - self._btn_center_view.width() - margin)
            y = margin
            self._btn_center_view.move(x, y)
            self._btn_center_view.raise_()
            return

        view = getattr(self._dialog, "view", None)
        if view is None:
            return

        view_top_left = view.mapTo(self, QPoint(0, 0))
        scrollbar = view.verticalScrollBar()
        margin = 15
        scrollbar_w = scrollbar.width() if scrollbar is not None and scrollbar.isVisible() else 0
        x = view_top_left.x() + view.width() - scrollbar_w - self._btn_center_view.width() - margin
        y = view_top_left.y() + margin
        self._btn_center_view.move(x, y)
        self._btn_center_view.raise_()
    
    def get_dialog(self) -> NodeGraphDialog | None:
        """Return the underlying NodeGraphDialog for access to graph data and methods."""
        return self._dialog
    
    def set_translator(self, translate: Callable[[str], str]) -> None:
        """Update the translator function."""
        self._tr = translate
        self._retranslate_ui()
        if self._dialog is not None and hasattr(self._dialog, 'set_translator'):
            self._dialog.set_translator(translate)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_center_button()

    def get_splitter(self) -> QSplitter:
        """Expose splitter so parent UI can sync external controls to graph width."""
        return self._splitter

    def graph_right_padding(self) -> int:
        """Return right-side padding needed to match graph viewport width (exclude props panel)."""
        sizes = self._splitter.sizes()
        if len(sizes) < 2:
            return 0
        return max(0, int(sizes[1]))
