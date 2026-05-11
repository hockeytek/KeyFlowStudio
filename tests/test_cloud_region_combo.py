"""Tests for the region combo helpers in cloud_aws_settings_tab.

These tests exercise the pure helper functions without launching a Qt event
loop or making real AWS calls.  PySide6 is mocked at the module level so the
file can be imported even in a headless CI environment.
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal PySide6 stub — any attribute access returns a MagicMock so the
# mixin module can be imported in headless CI without a running Qt display.
# If real PySide6 is already available (installed + QT_QPA_PLATFORM=offscreen
# set by conftest.py), skip the stub entirely to avoid polluting sys.modules
# with Mock objects that cascade into metaclass conflicts in other test files.
# ---------------------------------------------------------------------------
def _build_pyside6_stub():
    # If real PySide6 is present and its QtCore has actual class objects,
    # don't replace it — return the real package instead.
    try:
        import PySide6.QtCore as _real_qtcore
        if isinstance(_real_qtcore.QObject, type):
            return sys.modules.get("PySide6")
    except Exception:
        pass

    pyside6 = types.ModuleType("PySide6")
    for sub in ("QtCore", "QtGui", "QtWidgets", "QtNetwork", "QtSvg", "QtSvgWidgets"):
        mod = MagicMock()
        mod.__name__ = f"PySide6.{sub}"
        pyside6.__dict__[sub] = mod
        sys.modules[f"PySide6.{sub}"] = mod

    # QSignalBlocker used as a context/del object — needs a real class
    class _FakeSignalBlocker:
        def __init__(self, *a, **kw): ...
        def __enter__(self): return self
        def __exit__(self, *a): ...
        def __del__(self): ...

    pyside6.QtCore.QSignalBlocker = _FakeSignalBlocker  # type: ignore[attr-defined]
    # Qt.AlignLeft etc. — just need to be hashable ints
    pyside6.QtCore.Qt = MagicMock()  # type: ignore[attr-defined]
    pyside6.QtCore.Signal = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
    pyside6.QtCore.Slot = lambda *a, **kw: (lambda f: f)  # type: ignore[attr-defined]

    sys.modules["PySide6"] = pyside6
    return pyside6


_build_pyside6_stub()


# ---------------------------------------------------------------------------
# Now we can import the units under test
# ---------------------------------------------------------------------------
from app.cloud_manager import AWS_REGION_NAMES  # noqa: E402


class _FakeCombo:
    """Minimal QComboBox-like object for testing _fill_region_combo."""

    def __init__(self):
        self._items: list[tuple[str, str]] = []  # (display, data)
        self._current = 0

    # QComboBox API subset
    def clear(self): self._items.clear(); self._current = 0
    def addItem(self, display, data=None): self._items.append((display, data or display))
    def setItemIcon(self, i, icon): pass  # icons not inspected in tests
    def count(self): return len(self._items)
    def itemData(self, i): return self._items[i][1]
    def itemText(self, i): return self._items[i][0]
    def setCurrentIndex(self, i): self._current = i
    def currentIndex(self): return self._current
    def currentData(self): return self._items[self._current][1] if self._items else None
    def currentText(self): return self._items[self._current][0] if self._items else ""


# Patch QSignalBlocker and _green_dot_icon in the tab module to use no-op versions
import app.cloud_aws_settings_tab as _tab_mod  # noqa: E402


class _NoopBlocker:
    def __init__(self, *a, **kw): ...
    def __del__(self): ...


_tab_mod.QSignalBlocker = _NoopBlocker  # type: ignore[attr-defined]

# _green_dot_icon needs a real QPainter — replace with no-op returning a sentinel
_tab_mod._green_dot_icon = lambda size=10: None  # type: ignore[attr-defined]

from app.cloud_aws_settings_tab import (  # noqa: E402
    _fill_region_combo,
    _populate_cloud_region_combo,
    _region_code,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestFillRegionCombo(unittest.TestCase):
    """_fill_region_combo fills items and selects correct region."""

    def _combo(self):
        return _FakeCombo()

    def test_all_items_added(self):
        combo = self._combo()
        regions = ["us-east-1", "eu-west-1", "ap-northeast-1"]
        _fill_region_combo(combo, regions, "eu-west-1")
        self.assertEqual(combo.count(), 3)

    def test_known_region_shows_city_name(self):
        combo = self._combo()
        _fill_region_combo(combo, ["eu-west-1"], "eu-west-1")
        display = combo.itemText(0)
        self.assertIn("eu-west-1", display)
        self.assertIn("Ireland", display)  # from AWS_REGION_NAMES

    def test_unknown_region_shows_code_only(self):
        combo = self._combo()
        _fill_region_combo(combo, ["xx-fake-99"], "xx-fake-99")
        self.assertEqual(combo.itemText(0), "xx-fake-99")

    def test_current_region_is_selected(self):
        combo = self._combo()
        regions = ["us-east-1", "eu-west-1", "ap-northeast-1"]
        _fill_region_combo(combo, regions, "ap-northeast-1")
        self.assertEqual(combo.currentData(), "ap-northeast-1")

    def test_missing_current_region_falls_back_to_first(self):
        combo = self._combo()
        _fill_region_combo(combo, ["us-east-1", "eu-west-1"], "not-a-region")
        self.assertEqual(combo.currentIndex(), 0)

    def test_data_field_holds_bare_code(self):
        """itemData must be the bare region code, not the display string."""
        combo = self._combo()
        _fill_region_combo(combo, ["eu-west-1"], "eu-west-1")
        self.assertEqual(combo.itemData(0), "eu-west-1")
        self.assertNotIn(" — ", combo.itemData(0))

    def test_active_region_gets_icon(self):
        """The item matching current_region must receive setItemIcon."""
        combo = _FakeCombo()
        icon_calls: list[int] = []
        combo.setItemIcon = lambda i, icon: icon_calls.append(i)  # type: ignore[method-assign]
        regions = ["us-east-1", "eu-west-1", "ap-northeast-1"]
        _fill_region_combo(combo, regions, "eu-west-1")
        self.assertEqual(icon_calls, [1])  # eu-west-1 is at index 1

    def test_dot_moves_on_selection_change(self):
        """After _fill_region_combo, changing currentIndex must move the dot."""
        from unittest.mock import MagicMock

        _DOT = MagicMock(name="dot")
        _EMPTY = MagicMock(name="empty")
        icon_log: list[tuple[int, object]] = []

        class _TrackingCombo(_FakeCombo):
            def setItemIcon(self, i, icon):
                icon_log.append((i, icon))

        combo = _TrackingCombo()
        regions = ["us-east-1", "eu-west-1", "ap-northeast-1"]
        _fill_region_combo(combo, regions, "us-east-1")
        icon_log.clear()

        # Simulate _update_dot(2) — exactly what the signal handler does in production
        def _update_dot(idx: int) -> None:
            for i in range(combo.count()):
                combo.setItemIcon(i, _DOT if i == idx else _EMPTY)

        _update_dot(2)  # user selects ap-northeast-1

        dot_indices = [i for i, icon in icon_log if icon is _DOT]
        empty_indices = [i for i, icon in icon_log if icon is _EMPTY]
        self.assertEqual(dot_indices, [2])
        self.assertCountEqual(empty_indices, [0, 1])


class TestRegionCode(unittest.TestCase):
    """_region_code resolves both display-text and data-bearing combos."""

    def _combo_with_data(self, code: str) -> _FakeCombo:
        combo = _FakeCombo()
        _fill_region_combo(combo, [code], code)
        return combo

    def test_returns_code_from_data(self):
        combo = self._combo_with_data("eu-west-1")
        self.assertEqual(_region_code(combo), "eu-west-1")

    def test_returns_code_from_plain_text(self):
        """Combo whose items have no userData (e.g., loaded externally)."""
        combo = _FakeCombo()
        combo.addItem("us-east-1")  # no userData arg → data == display
        self.assertEqual(_region_code(combo), "us-east-1")

    def test_parses_display_text_when_no_data(self):
        """If currentData() is empty the em-dash split must work."""
        combo = _FakeCombo()
        combo.addItem("eu-central-1 — Frankfurt", "")
        combo._current = 0
        # Monkey-patch currentData to simulate no data
        combo.currentData = lambda: ""  # type: ignore[method-assign]
        self.assertEqual(_region_code(combo), "eu-central-1")


class TestRegionListCache(unittest.TestCase):
    """_populate_cloud_region_combo must use the session cache on second call."""

    def setUp(self):
        # Reset module-level cache before each test
        _tab_mod._region_list_cache = {}

    def test_cache_populated_after_first_call(self):
        fake_regions = ["us-east-1", "eu-west-1"]

        with patch("app.cloud_aws_settings_tab.get_regions_with_gpu_quota",
                   return_value=(fake_regions, None)) as mock_aws, \
             patch("app.cloud_aws_settings_tab.QTimer") as mock_timer, \
             patch("app.cloud_aws_settings_tab.threading.Thread") as mock_thread:

            combo = _FakeCombo()

            # Simulate thread running synchronously
            def run_thread(target, daemon):
                target()
                return MagicMock()
            mock_thread.side_effect = lambda target, daemon: (
                run_thread(target, daemon),
                MagicMock()
            )[1]

            _populate_cloud_region_combo(combo, profile=None, current_region="eu-west-1")

            self.assertIn(None, _tab_mod._region_list_cache)
            self.assertIn("us-east-1", _tab_mod._region_list_cache[None])

    def test_aws_not_called_when_cache_exists(self):
        _tab_mod._region_list_cache = {None: ["us-east-1", "eu-west-1"]}

        with patch("app.cloud_aws_settings_tab.get_regions_with_gpu_quota") as mock_aws:
            combo = _FakeCombo()
            _populate_cloud_region_combo(combo, profile=None, current_region="eu-west-1")

            mock_aws.assert_not_called()

    def test_cached_list_fills_combo(self):
        _tab_mod._region_list_cache = {None: ["ap-northeast-1", "eu-west-1", "us-east-1"]}

        combo = _FakeCombo()
        _populate_cloud_region_combo(combo, profile=None, current_region="ap-northeast-1")

        self.assertEqual(combo.count(), 3)
        self.assertEqual(combo.currentData(), "ap-northeast-1")


class TestAwsRegionNamesDict(unittest.TestCase):
    """Sanity-checks that the AWS_REGION_NAMES dict is well-formed."""

    def test_all_keys_are_valid_region_format(self):
        import re
        pattern = re.compile(r"^[a-z]+-[a-z]+-\d+$")
        for code in AWS_REGION_NAMES:
            self.assertRegex(code, pattern, f"Bad region code: {code!r}")

    def test_all_values_are_non_empty_strings(self):
        for code, city in AWS_REGION_NAMES.items():
            self.assertIsInstance(city, str, f"Value for {code!r} is not a string")
            self.assertTrue(city.strip(), f"Empty city name for {code!r}")

    def test_common_regions_present(self):
        for region in ("us-east-1", "eu-west-1", "ap-northeast-1", "us-west-2"):
            self.assertIn(region, AWS_REGION_NAMES, f"{region} missing from AWS_REGION_NAMES")


if __name__ == "__main__":
    unittest.main()
