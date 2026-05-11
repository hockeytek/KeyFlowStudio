#!/usr/bin/env bash
set -euo pipefail

# Unsupported workaround for Intel macOS environments where upstream SAM2
# requires torch>=2.5.1 but only torch 2.2.x wheels are available.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${1:-$ROOT_DIR/.venv/bin/python}"
PATCH_TORCH_REQ="${PATCH_TORCH_REQ:-torch>=2.2.2}"
SAM2_REPO_URL="${SAM2_REPO_URL:-https://github.com/facebookresearch/sam2.git}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  echo "Usage: bash scripts/install_sam2_intel_workaround.sh [path/to/python]" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d /tmp/sam2_patched.XXXXXX)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "[1/5] Cloning SAM2 source..."
git clone --depth 1 "$SAM2_REPO_URL" "$TMP_DIR" >/dev/null

echo "[2/5] Patching torch requirement to '$PATCH_TORCH_REQ'..."
for target in "$TMP_DIR/pyproject.toml" "$TMP_DIR/setup.py"; do
  if [[ -f "$target" ]]; then
    sed -i.bak -E "s/torch>=[0-9]+\.[0-9]+\.[0-9]+/$PATCH_TORCH_REQ/g" "$target"
    rm -f "$target.bak"
  fi
done

echo "[3/5] Installing SAM2 with relaxed dependency checks..."
"$PYTHON_BIN" -m pip install --no-build-isolation --no-deps "$TMP_DIR"

echo "[4/5] Verifying sam2 import..."
"$PYTHON_BIN" - <<'PY'
import importlib.util
spec = importlib.util.find_spec("sam2")
if spec is None:
    raise SystemExit("sam2 import failed: module not found")
import sam2
print("sam2 module:", sam2.__file__)
PY

echo "[5/5] Verifying native SAM2 init path..."
(
  cd "$ROOT_DIR"
  "$PYTHON_BIN" - <<'PY'
from app.services.sam2_service import Sam2Service

svc = Sam2Service(model_type="vit_h")
svc._load_native_predictor()
svc._load_native_video_predictor()

print("native_image:", bool(svc._native_enabled), "reason:", svc._native_failed_reason or "-")
print("native_video:", bool(svc._native_video_enabled), "reason:", svc._native_video_failed_reason or "-")
PY
)

echo
echo "Done. SAM2 workaround installed into: $PYTHON_BIN"
echo "Note: this is an unsupported workaround and may break after upstream updates."
