# Model Setup

KeyFlow Studio integrates several third-party model workflows. Model weights are not distributed in this repository or in the public preview app bundle.

## Attribution And Ownership

KeyFlow Studio provides integration code, graph orchestration, UI controls, and worker/runtime glue. The model architectures, upstream packages, checkpoints, papers, names, and licenses belong to their respective authors and projects.

Before redistributing binaries, screenshots, demos, or generated assets, review each upstream license and model card. Some projects may restrict commercial use, redistribution of weights, or use of generated assets.

Known upstream projects currently referenced by the integration:

| Workflow | Upstream project / source | Notes |
| --- | --- | --- |
| CorridorKey | [nikopueringer/CorridorKey](https://github.com/nikopueringer/CorridorKey), [CorridorKey_v1.0](https://huggingface.co/nikopueringer/CorridorKey_v1.0), [CorridorKeyBlue_1.0](https://huggingface.co/nikopueringer/CorridorKeyBlue_1.0) | Green/blue screen keying checkpoints are resolved outside Git history. Check upstream terms before redistribution. |
| BiRefNet | [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) and related `ZhengPeng7/BiRefNet_*` Hugging Face repos | Used as a mask/alpha source. Presets may map to different upstream repos. |
| GVM | [geyongtao/gvm](https://huggingface.co/geyongtao/gvm) | Used for generative video matting / alpha generation. KeyFlow includes compatibility patch files for the worker/runtime path. |
| MatAnyone2 | [pq-yang/MatAnyone2](https://github.com/pq-yang/MatAnyone2) | Optional matting workflow. The checkpoint is downloaded or placed outside Git history. |
| SAM2 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | Optional segmentation workflow when installed in the active environment. |
| SAM3 | [facebookresearch/sam3](https://github.com/facebookresearch/sam3), [facebook/sam3](https://huggingface.co/facebook/sam3) | Optional SAM3 workflow. Weight download may require manual setup or upstream access rules. |

This table is attribution documentation, not a license grant. Always follow the upstream project instructions for installation, citation, and allowed use.

At runtime, the application checks the configured model cache and downloads supported weights automatically when they are available from the configured upstream source. Some upstream models may still require manual license acceptance, private tokens, or external setup, so automatic download should be treated as a supported path rather than a universal guarantee for every model.

## Model Asset Policy

Keep the following assets outside Git and outside release archives unless the upstream license explicitly allows redistribution and the project maintainers have approved it:

- `*.pth`
- `*.pt`
- `*.ckpt`
- `*.onnx`
- `*.safetensors`
- `weights/`
- `checkpoints/`
- model cache folders

## Runtime Model Directory

Use `KEYFLOW_MODELS_DIR` to point the application or worker to a local model cache when needed:

```bash
export KEYFLOW_MODELS_DIR="$HOME/.local/share/com.keyflow.studio/models"
```

## CorridorKey

CorridorKey weights can be resolved through the app-managed download path when available. Keep the checkpoint outside Git history.

Supported checkpoint sources currently include the green-screen `nikopueringer/CorridorKey_v1.0` and blue-screen `nikopueringer/CorridorKeyBlue_1.0` repositories. The local integration defaults to `screen_color=green` for backward compatibility.

## BiRefNet, SAM, GVM, MatAnyone2

Each model may have separate licensing, hardware, and storage requirements. Check the upstream project before redistributing weights or derived assets.

## Tested Runtime Notes

The current development/test baseline is intentionally conservative:

- Local app and regression tests have been run on an older Intel i7 Mac in CPU mode.
- Cloud/server worker workflows have also been tested on a CUDA GPU worker.
- CPU mode is useful for validation and small samples, but production-sized video model inference is expected to be slow on old Intel hardware.

## Legacy Intel macOS Compatibility

Some modern ML packages no longer fit older Intel macOS constraints cleanly. For that reason the repository uses compatibility pins and local integration workarounds, including:

- `torch==2.2.2` and `torchvision==0.17.2` on `darwin x86_64`, because newer upstream combinations may not provide usable wheels for this architecture.
- `numpy<2` on `darwin x86_64` to avoid binary compatibility problems with older PyTorch wheels.
- `diffusers>=0.30,<0.32` for GVM compatibility, because newer versions changed behavior/dependencies in ways that require newer Torch stacks.
- `scripts/install_sam2_intel_workaround.sh` for Intel macOS SAM2 package constraints.
- `ec2_worker/_patch/` patch files for cloud/runtime compatibility around GVM components.

These patches and pins are pragmatic compatibility work, not forks of the upstream projects. When running on modern Linux/CUDA machines, prefer the CUDA-compatible PyTorch build that matches the driver/runtime and validate the model-specific stack separately.

## Release Hygiene

Public releases should contain the application code, documentation, UI assets, and small graph templates only. Model weights, generated outputs, and private media should remain external to the repository and release archives.