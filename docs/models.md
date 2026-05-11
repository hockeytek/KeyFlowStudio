# Model Setup

KeyFlow Studio integrates several third-party model workflows. Model weights are not committed to this repository.

At runtime, the application checks the configured model cache and downloads supported weights automatically when they are available from the configured upstream source. Some upstream models may still require manual license acceptance, private tokens, or external setup, so automatic download should be treated as a supported path rather than a universal guarantee for every model.

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

CorridorKey weights can be resolved through the app-managed download path when available. Keep the checkpoint outside Git history.

## BiRefNet, SAM, GVM, MatAnyone2

Each model may have separate licensing, hardware, and storage requirements. Check the upstream project before redistributing weights or derived assets.

## Public Repository Rule

Before making the repository public, verify that model weights and generated media are absent from both the working tree and Git history.