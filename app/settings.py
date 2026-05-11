"""Application settings helpers for KeyFlow Studio."""

from __future__ import annotations

from PySide6.QtCore import QSettings


APP_SETTINGS_ORG = "KeyFlow"
APP_SETTINGS_APP = "Studio"
LEGACY_SETTINGS_ORG = "MatAnyone2"


def get_app_settings() -> QSettings:
    """Return app settings and migrate legacy MatAnyone2 settings once."""
    settings = QSettings(APP_SETTINGS_ORG, APP_SETTINGS_APP)
    try:
        if settings.allKeys():
            return settings

        legacy = QSettings(LEGACY_SETTINGS_ORG, APP_SETTINGS_APP)
        legacy_keys = legacy.allKeys()
        if not legacy_keys:
            return settings

        for key in legacy_keys:
            settings.setValue(key, legacy.value(key))
        settings.sync()
    except Exception:
        # Never block startup if settings migration fails.
        pass

    return settings