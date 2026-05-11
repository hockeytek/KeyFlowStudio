"""
Download model weights to EC2.

Usage:
    source ~/keyflow-venv/bin/activate
    python3 download_models.py                    # all models
    python3 download_models.py --model matanyone2
    python3 download_models.py --model birefnet --preset General
    python3 download_models.py --model birefnet --preset Matting
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Target dir ────────────────────────────────────────────────────────────────
_DEFAULT_MODELS_DIR = Path.home() / ".local" / "share" / "com.keyflow.studio" / "models"
MODELS_DIR = Path(os.environ.get("KEYFLOW_MODELS_DIR", "")).expanduser() \
    if os.environ.get("KEYFLOW_MODELS_DIR") else _DEFAULT_MODELS_DIR
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Models directory: {MODELS_DIR}")


def _hf(repo: str, local_dir: Path, allow_patterns: list[str] | None = None):
    from huggingface_hub import snapshot_download
    print(f"\n📥 Downloading {repo} → {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns or ["*.safetensors", "*.bin", "*.pth", "*.pt", "*.py", "*.json"],
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"✅ {repo} saved to {local_dir}")


def download_matanyone2():
    import urllib.request
    url     = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
    out_dir = MODELS_DIR / "matanyone2" / "v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "matanyone2.pth"
    if out_path.exists():
        print(f"✅ MatAnyone2 already exists: {out_path}")
        return
    print("\n📥 Downloading MatAnyone2 checkpoint...")
    print(f"   {url}")
    print(f"   → {out_path}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, int(downloaded * 100 / total_size))
            mb  = downloaded // 1_000_000
            total_mb = total_size // 1_000_000
            print(f"\r   {pct:3d}%  {mb}/{total_mb} MB", end="", flush=True)

    urllib.request.urlretrieve(url, str(out_path), reporthook=_progress)
    print(f"\n✅ MatAnyone2 saved: {out_path} ({out_path.stat().st_size // 1_000_000} MB)")


_BIREFNET_REPOS = {
    "General":          "ZhengPeng7/BiRefNet",
    "General-dynamic":  "ZhengPeng7/BiRefNet_dynamic",
    "General-HR":       "ZhengPeng7/BiRefNet_HR",
    "General-Lite":     "ZhengPeng7/BiRefNet_lite",
    "Matting":          "ZhengPeng7/BiRefNet-matting",
    "Matting-HR":       "ZhengPeng7/BiRefNet_HR-matting",
    "Portrait":         "ZhengPeng7/BiRefNet-portrait",
}

_CORRIDORKEY_GREEN_CHECKPOINT_FILENAMES = (
    "CorridorKey_v1.0.safetensors",
    "CorridorKey_v1.0.pth",
    "CorridorKey.pth",
    "corridorkey.pth",
)
_CORRIDORKEY_BLUE_CHECKPOINT_FILENAMES = (
    "CorridorKeyBlue_1.0.safetensors",
    "CorridorKeyBlue_1.0.pth",
)
_CORRIDORKEY_REPO_ID = "nikopueringer/CorridorKey_v1.0"
_CORRIDORKEY_BLUE_REPO_ID = "nikopueringer/CorridorKeyBlue_1.0"


def _download_corridorkey_variant(repo_id: str, filenames: tuple[str, ...], label: str) -> Path:
    from huggingface_hub import hf_hub_download

    out_dir = MODELS_DIR / "corridorkey" / "v1.0"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in filenames:
        if (out_dir / fname).exists():
            print(f"✅ CorridorKey {label} already exists: {out_dir / fname}")
            return out_dir / fname

    target_filename = filenames[0]
    print(f"\n📥 Downloading CorridorKey {label} checkpoint from {repo_id}")
    print(f"   → {out_dir / target_filename}")

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=target_filename,
        local_dir=str(out_dir),
    )
    print(
        f"✅ CorridorKey {label} saved: {downloaded} "
        f"({Path(downloaded).stat().st_size // 1_000_000} MB)"
    )
    return Path(downloaded)


def download_corridorkey():
    """Download CorridorKey green and blue checkpoints from HuggingFace."""
    green = _download_corridorkey_variant(
        _CORRIDORKEY_REPO_ID,
        _CORRIDORKEY_GREEN_CHECKPOINT_FILENAMES,
        "green",
    )
    _download_corridorkey_variant(
        _CORRIDORKEY_BLUE_REPO_ID,
        _CORRIDORKEY_BLUE_CHECKPOINT_FILENAMES,
        "blue",
    )
    return green


def download_birefnet(preset: str = "General"):
    if preset not in _BIREFNET_REPOS:
        print(f"❌ Unknown preset {preset!r}. Available: {list(_BIREFNET_REPOS)}")
        sys.exit(1)
    repo      = _BIREFNET_REPOS[preset]
    local_dir = MODELS_DIR / "birefnet" / preset
    _hf(repo, local_dir)


def download_gvm():
    _hf("geyongtao/gvm", MODELS_DIR / "gvm")


def main():
    ap = argparse.ArgumentParser(description="Download KeyFlow model weights")
    ap.add_argument("--model",  default="all",
                    choices=["all", "matanyone2", "birefnet", "gvm", "corridorkey"],
                    help="Which model to download")
    ap.add_argument("--preset", default="General",
                    help="BiRefNet preset (default: General)")
    args = ap.parse_args()

    if args.model in ("all", "matanyone2"):
        download_matanyone2()

    if args.model in ("all", "birefnet"):
        download_birefnet(args.preset)
        if args.model == "all":
            # Also download Matting preset — commonly used
            download_birefnet("Matting")

    if args.model in ("all", "gvm"):
        download_gvm()

    if args.model in ("all", "corridorkey"):
        download_corridorkey()

    print("\n✅ All done.")


if __name__ == "__main__":
    main()
