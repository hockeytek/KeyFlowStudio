"""Shortcut helpers for the application."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QWidget


def create_save_shortcut(parent: QWidget, callback: Callable[[], None]) -> QShortcut:
    """Create a window-level Save shortcut (Cmd+S / Ctrl+S)."""
    shortcut = QShortcut(QKeySequence.StandardKey.Save, parent)
    shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
    shortcut.activated.connect(callback)
    return shortcut


def handle_node_graph_hotkeys(
    event: QKeyEvent,
    *,
    open_quick_add_popup: Callable[[], None],
    reset_view: Callable[[], None],
    is_connection_drag_active: Callable[[], bool],
    cancel_connection_drag: Callable[[], None],
    group_selected_nodes: Callable[[], None],
    ungroup_selected_groups: Callable[[], None],
    delete_selected_items: Callable[[], bool],
) -> bool:
    """Handle keyboard shortcuts for the node graph dialog/view.

    Returns True when the event was handled and accepted.
    """
    if event.key() == Qt.Key.Key_Tab:
        open_quick_add_popup()
        event.accept()
        return True

    if event.key() == Qt.Key.Key_F:
        reset_view()
        event.accept()
        return True

    if event.key() == Qt.Key.Key_Escape and is_connection_drag_active():
        cancel_connection_drag()
        event.accept()
        return True

    if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
        if event.key() == Qt.Key.Key_G and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            group_selected_nodes()
            event.accept()
            return True
        if event.key() == Qt.Key.Key_G and (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            ungroup_selected_groups()
            event.accept()
            return True

    if event.key() in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace} and delete_selected_items():
        event.accept()
        return True

    return False
