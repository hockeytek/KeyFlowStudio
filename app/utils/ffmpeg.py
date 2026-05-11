"""FFmpeg utilities"""
import subprocess
from io import BytesIO


def get_ffmpeg_exe() -> str:
    """Возвращает путь к ffmpeg: сначала из imageio_ffmpeg, затем из PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def check_ffmpeg():
    """Проверяет наличие ffmpeg в системе"""
    try:
        result = subprocess.run([get_ffmpeg_exe(), "-version"],
                              capture_output=True,
                              timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_ffmpeg_info():
    """Получает информацию о версии ffmpeg"""
    try:
        result = subprocess.run([get_ffmpeg_exe(), "-version"],
                              capture_output=True,
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            # Первая строка содержит версию
            return result.stdout.split('\n')[0]
        return None
    except Exception:
        return None


def _get_ffprobe_exe() -> str | None:
    """Locate ffprobe: system PATH first, then next to the imageio_ffmpeg binary."""
    import shutil
    # 1. System PATH
    if shutil.which("ffprobe"):
        return "ffprobe"
    # 2. Same directory as ffmpeg binary (imageio_ffmpeg bundles it alongside ffmpeg)
    try:
        ffmpeg_exe = get_ffmpeg_exe()
        from pathlib import Path as _Path
        ffmpeg_dir = _Path(ffmpeg_exe).parent
        for candidate in ffmpeg_dir.iterdir():
            if candidate.name.startswith("ffprobe"):
                return str(candidate)
    except Exception:
        pass
    return None


def get_color_space_info(media_path: str) -> str | None:
    """Return a human-readable color space string for *media_path* via ffprobe.

    Returns None when ffprobe is unavailable or metadata is missing.
    Examples: "sRGB", "Rec.709", "Rec.2020", "HDR (PQ)", "HDR (HLG)"
    """
    try:
        import json
        import subprocess
        ffprobe_exe = _get_ffprobe_exe()
        if ffprobe_exe is None:
            return None
        result = subprocess.run(
            [
                ffprobe_exe,
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-select_streams", "v:0",
                media_path,
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            return None
        s = streams[0]
        primaries = (s.get("color_primaries") or "").lower()
        transfer = (s.get("color_transfer") or "").lower()
        space = (s.get("color_space") or "").lower()
        color_range = (s.get("color_range") or "").lower()

        # HDR transfer functions take priority
        if transfer in ("smpte2084", "smpte st 2084"):
            return "HDR (PQ / SMPTE 2084)"
        if transfer in ("arib-std-b67", "hlg"):
            return "HDR (HLG)"

        # SDR primaries
        if primaries in ("bt709", "bt.709", "bt_709"):
            label = "Rec.709"
        elif primaries in ("bt2020", "bt.2020", "bt_2020"):
            label = "Rec.2020"
        elif primaries in ("smpte170m", "bt601", "bt.601", "bt_601"):
            label = "Rec.601"
        elif primaries in ("smpte432", "p3d65", "display_p3"):
            label = "Display P3"
        elif primaries in ("bt470bg",):
            label = "BT.470 BG"
        elif space in ("rgb",) or primaries == "":
            label = "sRGB"
        else:
            label = primaries or space or None
            if not label:
                return None

        if color_range == "pc" or color_range == "jpeg":
            label += " (full range)"
        return label
    except Exception:
        return None


def get_image_color_space(image_path: str) -> str | None:
    """Return color-space string for an image using Pillow ICC profile info."""
    try:
        from PIL import Image as _Image, ImageCms as _ICC
        with _Image.open(image_path) as im:
            icc_bytes = im.info.get("icc_profile")
            mode = im.mode  # RGB, RGBA, L, CMYK …
            if icc_bytes:
                try:
                    profile = _ICC.ImageCmsProfile(BytesIO(icc_bytes))
                    desc = _ICC.getProfileDescription(profile).strip()
                    if desc:
                        return desc
                except Exception:
                    pass
            # Fall back to mode
            if mode in ("RGB", "RGBA"):
                return "sRGB"
            if mode == "CMYK":
                return "CMYK"
            if mode == "L":
                return "Grayscale"
            return mode or None
    except Exception:
        return None


def install_ffmpeg_info():
    """Возвращает инструкцию по установке ffmpeg на Mac"""
    return """
FFmpeg не найден в системе.

Для установки используйте Homebrew:
    brew install ffmpeg

Если у вас нет Homebrew, установите его с:
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
"""
