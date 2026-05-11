# KeyFlow Studio - Quick Start

This guide is the shortest path from a fresh checkout to a running development build. For a fuller setup guide, see [docs/installation.md](docs/installation.md).

## Requirements

- Python 3.11 recommended
- FFmpeg available on `PATH`
- macOS, Linux, or Windows, depending on the model backend you use
- Internet access for first-time model weight downloads

On macOS, install FFmpeg with Homebrew if needed:

```bash
brew install ffmpeg
```

On Ubuntu:

```bash
sudo apt update
sudo apt install ffmpeg
```

## Install

```bash
git clone https://github.com/hockeytek/KeyFlowStudio.git
cd KeyFlowStudio

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

You can also run the helper script from the repository root:

```bash
bash setup.sh
```

The helper checks Python and FFmpeg, creates `.venv` when needed, installs `requirements.txt`, and offers optional MatAnyone2 package installation.

## Run

```bash
python main.py
```

Or use the repository helper:

```bash
bash run.sh
```

## First Workflow Check

1. Open KeyFlow Studio.
2. Load source media through the app UI or a Load/Source node.
3. Open the node graph editor.
4. Use a built-in preset or connect a minimal graph such as `Source/Load -> keying or matting node -> Write`.
5. Run the graph and check diagnostics, preview, and write outputs.

For node compatibility and runtime behavior, see [docs/NODE_GRAPH_STANDARD.md](docs/NODE_GRAPH_STANDARD.md) and [docs/node-rules/](docs/node-rules/).

## Model Weights

Model weights are not stored in Git. KeyFlow Studio uses the configured model cache and downloads supported weights automatically when the upstream source allows it.

Use `KEYFLOW_MODELS_DIR` when you want an explicit local cache location:

```bash
export KEYFLOW_MODELS_DIR="$HOME/.local/share/com.keyflow.studio/models"
python main.py
```

Some models may still require manual license acceptance, private tokens, or local source installation. See [docs/models.md](docs/models.md).

## Validate

Run the lightweight UI smoke test:

```bash
PYTHONPATH="$PWD" python -m pytest tests/test_node_graph_dialog_smoke.py -q
```

Run the P1 regression helper:

```bash
./run_p1.sh
```

Optional device override:

```bash
./run_p1.sh --device cpu
./run_p1.sh --device mps
./run_p1.sh --device auto
```

## Common Next Links

- [Installation](docs/installation.md)
- [Model Setup](docs/models.md)
- [Cloud GPU Setup](docs/cloud-gpu.md)
- [Troubleshooting](TROUBLESHOOTING.md)
