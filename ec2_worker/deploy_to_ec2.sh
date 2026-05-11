#!/usr/bin/env bash
# deploy_to_ec2.sh — Upload worker to EC2 and start it
# Usage: ./ec2_worker/deploy_to_ec2.sh <EC2_IP>

set -e

EC2_IP="${1:?Usage: $0 <EC2_IP>}"
SSH_KEY="${KEYFLOW_SSH_KEY:-~/.ssh/keyflow-gpu.pem}"
SSH_USER="${KEYFLOW_SSH_USER:-ubuntu}"
REMOTE_DIR="~/keyflow-worker"

echo "=== Deploying KeyFlow Worker to ${SSH_USER}@${EC2_IP} ==="

# 1. Upload worker bundle
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SSH_USER}@${EC2_IP}" \
  "mkdir -p ${REMOTE_DIR}/ec2_worker ${REMOTE_DIR}/app/services ${REMOTE_DIR}/app/utils"

rsync -avz -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no" \
  "$(dirname "$0")/" \
  "${SSH_USER}@${EC2_IP}:${REMOTE_DIR}/ec2_worker/"

rsync -avz -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no" \
  "$(dirname "$0")/../app/services/" \
  "${SSH_USER}@${EC2_IP}:${REMOTE_DIR}/app/services/"

rsync -avz -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no" \
  "$(dirname "$0")/../app/utils/" \
  "${SSH_USER}@${EC2_IP}:${REMOTE_DIR}/app/utils/"

rsync -avz -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no" \
  "$(dirname "$0")/../app/__init__.py" \
  "${SSH_USER}@${EC2_IP}:${REMOTE_DIR}/app/"

# Upload vendored gvm_core package if present in local venv.
LOCAL_GVM_CORE="$(python3 - <<'PY'
import site
from glob import glob
from pathlib import Path
for base in site.getsitepackages():
    p = Path(base) / 'gvm_core'
    if p.exists():
        print(p)
        break
else:
  candidates = sorted(glob(str(Path.cwd() / '.venv' / 'lib' / 'python*' / 'site-packages' / 'gvm_core')))
  if candidates:
    print(candidates[0])
PY
)"

if [[ -n "${LOCAL_GVM_CORE}" && -d "${LOCAL_GVM_CORE}" ]]; then
  echo "=== Uploading gvm_core package from ${LOCAL_GVM_CORE} ==="
  rsync -avz -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no" \
    "${LOCAL_GVM_CORE}/" \
    "${SSH_USER}@${EC2_IP}:${REMOTE_DIR}/gvm_core/"
else
  echo "=== WARN: local gvm_core package not found; cloud GVM graph jobs may fail ==="
fi

# Apply local patches on top of uploaded gvm_core (survives gvm_core upgrades).
PATCH_DIR="$(dirname "$0")/_patch"
if [[ -d "${PATCH_DIR}" ]]; then
  echo "=== Applying patches from ${PATCH_DIR} ==="
  rsync -avz -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no" \
    "${PATCH_DIR}/" \
    "${SSH_USER}@${EC2_IP}:${REMOTE_DIR}/"
  # Clear stale .pyc so Python picks up the new source files.
  ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SSH_USER}@${EC2_IP}" \
    "find ${REMOTE_DIR}/gvm_core -name '*.pyc' -delete 2>/dev/null || true"
fi

echo "=== Files uploaded ==="

# 2. Install deps + start worker
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SSH_USER}@${EC2_IP}" << 'REMOTE_EOF'
set -e
source ~/keyflow-venv/bin/activate

echo "=== Installing worker deps ==="
pip install -q fastapi uvicorn python-multipart easydict diffusers tqdm

echo "=== Stopping old worker (if any) ==="
pkill -f "uvicorn worker:app" 2>/dev/null || true
sleep 1

echo "=== Downloading models (if needed) ==="
cd ~/keyflow-worker/ec2_worker
python3 download_models.py --model matanyone2
python3 download_models.py --model birefnet --preset General
python3 download_models.py --model gvm

echo "=== Starting worker ==="
nohup python3 worker.py > ~/keyflow-worker/worker.log 2>&1 &
sleep 2

# Verify it started
if pgrep -f "python3 worker.py|uvicorn worker:app" > /dev/null; then
    echo "✅ Worker running on port 8080"
else
    echo "❌ Worker failed to start. Check ~/keyflow-worker/worker.log"
    cat ~/keyflow-worker/worker.log | tail -30
fi
REMOTE_EOF

echo ""
echo "✅ Deploy complete!"
echo "   Health check: curl http://${EC2_IP}:8080/health"
echo "   Logs:         ssh -i ${SSH_KEY} ${SSH_USER}@${EC2_IP} tail -f ~/keyflow-worker/worker.log"
