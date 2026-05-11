# Model Setup

KeyFlow Studio integrates several third-party model workflows. Model weights are not committed to this repository.

## Storage Policy

Do not commit:

- `*.pth`
- `*.pt`
- `*.ckpt`
- `*.onnx`
- `*.safetensors`
- `weights/`
- `checkpoints/`
- model cache folders

These files are intentionally ignored in `.gitignore`.

## Runtime Model Directory

Use `KEYFLOW_MODELS_DIR` to point the application or worker to a local model cache when needed:

```bash
export KEYFLOW_MODELS_DIR="$HOME/.local/share/com.keyflow.studio/models"
```

## CorridorKey

CorridorKey weights should be downloaded from the upstream model source or by the app-managed download path. Keep the checkpoint outside Git history.

## BiRefNet, SAM, GVM, MatAnyone2

Each model may have separate licensing, hardware, and storage requirements. Check the upstream project before redistributing weights or derived assets.

## Public Repository Rule

Before making the repository public, verify that model weights and generated media are absent from both the working tree and Git history.