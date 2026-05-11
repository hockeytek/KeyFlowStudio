# KeyFlow Studio - Troubleshooting

This guide covers development installs and common runtime issues. Start with [QUICKSTART.md](QUICKSTART.md) or [docs/installation.md](docs/installation.md) for setup.

## Installation Checks

From the repository root:

```bash
source .venv/bin/activate
python --version
python -c "import PySide6, torch, cv2; print('core imports ok')"
ffmpeg -version | head -n 1
```

Recommended baseline:

- Python 3.11
- FFmpeg on `PATH`
- Dependencies installed with `pip install -r requirements.txt`
- Model weights outside Git, usually in the app model cache or `KEYFLOW_MODELS_DIR`

## App Does Not Start

Run from a terminal so the traceback is visible:

```bash
source .venv/bin/activate
python -u main.py
```

If the error is `No module named 'PySide6'`, reinstall dependencies in the active environment:

```bash
pip install -r requirements.txt
```

If Qt cannot initialize a display on Linux, verify desktop/display access. In headless CI, use:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH="$PWD" python -m pytest tests/test_node_graph_dialog_smoke.py -q
```

## FFmpeg Not Found

Check:

```bash
which ffmpeg
ffmpeg -version
```

Install on macOS:

```bash
brew install ffmpeg
```

Install on Ubuntu:

```bash
sudo apt update
sudo apt install ffmpeg
```

## Device Or Acceleration Issues

KeyFlow Studio can run on CPU, Apple MPS, or CUDA when the selected models support that backend.

Force CPU for debugging:

```bash
KEYFLOW_DEVICE=cpu python main.py
```

Inspect the active device:

```bash
python -c "from app.utils import get_device, get_device_name; d = get_device(); print(d, get_device_name())"
```

Common notes:

- Intel macOS usually runs CPU paths; MPS is expected to be unavailable.
- Apple Silicon can use MPS where PyTorch and the model path support it.
- NVIDIA systems need a PyTorch build compatible with the driver/CUDA runtime.
- CPU is useful for smoke checks but slow for production video inference.

## Model Weights Or Model Package Errors

Weights are intentionally not committed. Supported models download weights into the configured model cache when possible.

Use an explicit cache when needed:

```bash
export KEYFLOW_MODELS_DIR="$HOME/.local/share/com.keyflow.studio/models"
python main.py
```

If a model requires manual setup, tokens, or license acceptance, follow the model-specific instructions in [docs/models.md](docs/models.md) and [docs/node-rules/](docs/node-rules/).

### `No module named 'matanyone2'`

MatAnyone2 is a model package dependency for matting workflows. Install it from source or a local checkout when that workflow is needed:

```bash
pip install git+https://github.com/pq-yang/MatAnyone2.git
```

Or:

```bash
pip install -e /path/to/MatAnyone2
```

### `No module named 'sam2'` On Intel macOS

Some upstream SAM2 package constraints do not match Intel macOS PyTorch wheels. Use the included workaround script from an activated `.venv`:

```bash
bash scripts/install_sam2_intel_workaround.sh
```

Run it again after recreating `.venv`.

### BiRefNet Import Or Dependency Errors

BiRefNet fallback support uses `huggingface_hub`, `torchvision`, `transformers`, and related packages from `requirements.txt`. Reinstall dependencies first:

```bash
pip install -r requirements.txt
```

If the GUI reports `No module named BiRefNetModule`, KeyFlow Studio should use the built-in fallback when dependencies are present. Check the app log for the actual missing dependency.

### CorridorKey Checkpoint Download Fails

CorridorKey can auto-download supported checkpoints. If download fails, check:

- Internet access
- Hugging Face availability
- Whether the target repository requires authentication or license acceptance
- `KEYFLOW_MODELS_DIR` permissions

## Node Graph Problems

Use graph diagnostics first. The validator checks required inputs, port compatibility, frame-count constraints, and node-specific contracts.

Useful references:

- [docs/NODE_GRAPH_STANDARD.md](docs/NODE_GRAPH_STANDARD.md)
- [docs/node-rules/](docs/node-rules/)

Common cases:

- `Source/Load -> keying or matting node -> Write` is the simplest shape to validate.
- `BiRefNet -> CorridorKey.alphahint` can use staged execution to reduce memory pressure.
- `SAM -> CorridorKey.alphahint` can stream masks from disk when the graph contract allows it.
- Matting workflows expect an image/video input plus a valid seed mask or compatible alpha/mask source.

## Out Of Memory

Symptoms include `CUDA out of memory`, MPS memory errors, very slow processing, or the process being killed by the OS.

Try:

```bash
KEYFLOW_DEVICE=cpu python main.py
```

Or reduce input resolution before testing:

```bash
ffmpeg -i input.mp4 -vf scale=-1:720 output_720p.mp4
```

For heavy final runs, use the EC2 worker flow documented in [docs/cloud-gpu.md](docs/cloud-gpu.md).

## Output Looks Empty, Black, Or Incorrect

Check the graph contract and the media inputs:

- Source media is readable by FFmpeg/OpenCV.
- Mask or alpha input has the expected dimensions and range.
- Write node output path is configured.
- Diagnostics do not report missing required inputs or incompatible ports.

Inspect a mask image:

```bash
python - <<'PY'
from PIL import Image
import numpy as np

mask = Image.open("your_mask.png")
arr = np.asarray(mask)
print("mode", mask.mode, "size", mask.size, "min", arr.min(), "max", arr.max())
PY
```

## Cloud GPU Problems

For EC2 worker setup, use [docs/cloud-gpu.md](docs/cloud-gpu.md).

Basic checks:

- AWS credentials stay outside the repository.
- Region and instance type match your quota.
- Security group allows only the access you intend.
- Worker logs show CUDA and PyTorch availability.
- Local and remote model caches are not committed to Git.

## Collect Debug Information

Use this when asking for help:

```bash
{
  echo "=== System ==="
  uname -a

  echo "=== Python ==="
  python --version
  which python

  echo "=== FFmpeg ==="
  ffmpeg -version 2>&1 | head -n 1

  echo "=== PyTorch ==="
  python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
print('mps', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
PY

  echo "=== KeyFlow Device ==="
  python - <<'PY'
from app.utils import get_device, get_device_name
print(get_device(), get_device_name())
PY

  echo "=== Smoke Test ==="
  PYTHONPATH="$PWD" python -m pytest tests/test_node_graph_dialog_smoke.py -q
} | tee debug_info.txt
```

## Final Sanity Check

A healthy development install should be able to:

- Start with `python main.py`
- Show device and FFmpeg information in logs/status UI
- Open the node graph editor
- Validate a simple graph
- Run the focused smoke test
- Keep generated media, weights, caches, and credentials out of Git
