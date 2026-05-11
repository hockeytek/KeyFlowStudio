"""Shared project constants — single source of truth for magic numbers."""

# ── Video defaults ──
DEFAULT_FPS: float = 30.0
DEFAULT_VIDEO_CODEC: str = "h264"
DEFAULT_VIDEO_CRF: int = 23
DEFAULT_VIDEO_PRESET: str = "medium"
VALID_VIDEO_PRESETS: frozenset[str] = frozenset({"fast", "medium", "slow"})

# ── Image defaults ──
DEFAULT_PNG_COMPRESSION: int = 6
DEFAULT_PNG_BIT_DEPTH: int = 8
DEFAULT_JPG_QUALITY: int = 90

# ── FFmpeg codec mapping ──
FFMPEG_CODEC_MAP: dict[str, str] = {"h264": "libx264", "h265": "libx265"}
PRORES_PROFILES: dict[str, str] = {"prores422": "2", "prores422hq": "3", "prores4444": "4"}
