"""Utility modules"""
import os
import sys
from pathlib import Path

from .device import get_device, get_device_name
from .ffmpeg import check_ffmpeg, get_ffmpeg_info, install_ffmpeg_info


def _default_models_dir() -> Path:
    """Return platform-native default directory for model weights."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "com.keyflow.studio" / "models"

    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA", "").strip()
        if appdata:
            return Path(appdata).expanduser() / "KeyFlow Studio" / "models"
        return Path.home() / "AppData" / "Roaming" / "KeyFlow Studio" / "models"

    xdg_data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg_data_home).expanduser() if xdg_data_home else (Path.home() / ".local" / "share")
    return base / "com.keyflow.studio" / "models"


def get_models_dir() -> Path:
    """Return directory for user-downloaded model weights.

    Prefers KEYFLOW_MODELS_DIR, then falls back to platform-native app data path.
    """
    base = os.environ.get("KEYFLOW_MODELS_DIR", "").strip()
    model_dir = Path(base).expanduser() if base else _default_models_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _sanitize_path_component(value: str, fallback: str = "default") -> str:
    """Normalize model-family/version path components to safe filesystem names."""
    raw = str(value or "").strip().lower()
    if not raw:
        return fallback

    normalized = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in raw)
    normalized = normalized.strip("-._")
    return normalized or fallback


def get_model_family_dir(family: str) -> Path:
    """Return a namespaced model family directory, for example models/sam."""
    path = get_models_dir() / _sanitize_path_component(family, fallback="misc")
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_variant_dir(family: str, variant: str) -> Path:
    """Return a namespaced variant directory, for example models/sam/vit_h."""
    path = get_model_family_dir(family) / _sanitize_path_component(variant, fallback="default")
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "get_device",
    "get_device_name",
    "check_ffmpeg",
    "get_ffmpeg_info",
    "install_ffmpeg_info",
    "get_models_dir",
    "get_model_family_dir",
    "get_model_variant_dir",
]
