# Installation

This guide describes a development installation for KeyFlow Studio. Packaging for end users is still evolving.

## 1. Clone

```bash
git clone https://github.com/hockeytek/KeyFlowStudio.git
cd KeyFlowStudio
```

## 2. Create A Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The pinned defaults in `requirements.txt` are chosen for development compatibility, including Intel macOS constraints. For CUDA systems, install a PyTorch build that matches your NVIDIA driver and CUDA runtime before installing model-specific packages.

Some model workflows may require extra upstream packages, private tokens, or license acceptance. See [models.md](models.md) for model cache and weight behavior.

## 4. Install FFmpeg

macOS:

```bash
brew install ffmpeg
```

Ubuntu:

```bash
sudo apt update
sudo apt install ffmpeg
```

## 5. Run

```bash
python main.py
```

You can also use the helper scripts from the repository root:

```bash
bash setup.sh
bash run.sh
```

## Device Selection

The application supports CPU, Apple MPS, and CUDA paths where the underlying models support them. Use the application settings or environment variables such as `KEYFLOW_DEVICE` for explicit runtime selection.

## Validation

Run a lightweight smoke test:

```bash
pytest tests/test_node_graph_dialog_smoke.py -q
```

Run the P1 regression helper:

```bash
./run_p1.sh
```