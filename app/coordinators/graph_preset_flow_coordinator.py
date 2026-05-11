"""Graph preset selection flow extracted from MainWindow."""

from __future__ import annotations

from typing import Callable


class GraphPresetFlowCoordinator:
    """Orchestrates save/delete/select branches for preset combo selection."""

    def __init__(
        self,
        *,
        save_preset_key: str,
        delete_preset_key: str,
        empty_preset_key: str,
        save_current: Callable[[], str | None],
        delete_selected: Callable[[], str | None],
        refresh_combo: Callable[[str], None],
        set_selected_key: Callable[[str], None],
        set_baseline_from_current: Callable[[], None],
        set_saved_status: Callable[[str], None],
        graph_is_empty: Callable[[], bool],
        confirm_replace: Callable[[], bool],
        clear_graph: Callable[[], None],
        reset_view: Callable[[], None],
        payload_for_key: Callable[[str], dict | None],
        apply_preset: Callable[[dict], bool],
        finalize_apply: Callable[[dict, str], None],
    ) -> None:
        self._save_preset_key = save_preset_key
        self._delete_preset_key = delete_preset_key
        self._empty_preset_key = empty_preset_key
        self._save_current = save_current
        self._delete_selected = delete_selected
        self._refresh_combo = refresh_combo
        self._set_selected_key = set_selected_key
        self._set_baseline_from_current = set_baseline_from_current
        self._set_saved_status = set_saved_status
        self._graph_is_empty = graph_is_empty
        self._confirm_replace = confirm_replace
        self._clear_graph = clear_graph
        self._reset_view = reset_view
        self._payload_for_key = payload_for_key
        self._apply_preset = apply_preset
        self._finalize_apply = finalize_apply

    def handle_selection(self, *, key: str, previous_key: str) -> None:
        if key == self._save_preset_key:
            saved_key = self._save_current()
            self._refresh_combo(saved_key or previous_key)
            if saved_key:
                self._set_selected_key(saved_key)
                self._set_baseline_from_current()
                self._refresh_combo(saved_key)
                self._set_saved_status(saved_key)
            return

        if key == self._delete_preset_key:
            selected_key = self._delete_selected()
            self._refresh_combo(selected_key or previous_key)
            if selected_key:
                self._set_selected_key(selected_key)
                self._set_baseline_from_current()
                self._refresh_combo(selected_key)
            return

        if key == previous_key:
            return

        if not self._graph_is_empty() and not self._confirm_replace():
            self._refresh_combo(previous_key)
            return

        if key == self._empty_preset_key:
            self._clear_graph()
            self._reset_view()
            self._set_selected_key(key)
            self._set_baseline_from_current()
            self._refresh_combo(key)
            return

        preset = self._payload_for_key(key)
        if preset is None or not self._apply_preset(preset):
            self._refresh_combo(previous_key)
            return
        self._finalize_apply(preset, key)
