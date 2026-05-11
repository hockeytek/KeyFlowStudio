"""SAM3 backend adapter.

This service implements image inference for the SAM3 node using the official
open-source API (`build_sam3_image_model` + `Sam3Processor`).

Notes:
- Video/session API (`build_sam3_predictor(...).handle_request`) is intentionally
    kept out of this stage and will be integrated separately.
- Weights are expected to be provided manually by the user/developer.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import logging
import math
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
from PIL import Image

from app.utils import get_device, get_model_variant_dir


logger = logging.getLogger(__name__)


class Sam3Service:
    """SAM3 image service used by the SAM3 node."""

    # Checkpoint filenames accepted by this adapter.
    # The primary names come from the current official OSS builder path.
    SAM3_CKPT_CANDIDATES = {
        "sam3": [
            "sam3.pt",
            "model.safetensors",  # Hugging Face file name
            "sam3_sa1b.pt",  # legacy/local naming fallback
        ],
        "sam3.1": [
            "sam3.1_multiplex.pt",
            "model.safetensors",  # Hugging Face file name
            "sam3.1_sa1b.pt",  # legacy/local naming fallback
        ],
    }

    # TODO: replace with verified HuggingFace URLs once auth flow is decided.
    SAM3_URLS: dict[str, str] = {
        "sam3":   "",
        "sam3.1": "",
    }

    SAM3_LABELS = {
        "sam3":   "SAM3",
        "sam3.1": "SAM3.1",
    }

    # Simple in-process cache to avoid reloading image model per click.
    _IMAGE_RUNTIME_CACHE: dict[str, dict[str, Any]] = {}
    _COMPAT_PATCH_DEVICE: str | None = None
    _RUNTIME_NOTICE: str = ""

    # ── Weight status ────────────────────────────────────────────────

    @classmethod
    def _model_dir_for(cls, model_type: str) -> Path:
        return get_model_variant_dir("sam3", model_type)

    @classmethod
    def _resolve_checkpoint_path(cls, model_type: str) -> str | None:
        candidates = cls.SAM3_CKPT_CANDIDATES.get(model_type, [])
        if not candidates:
            return None
        model_dir = cls._model_dir_for(model_type)
        for filename in candidates:
            path = model_dir / filename
            if path.is_file():
                return str(path)
        return None

    @classmethod
    def _prepare_checkpoint_for_builder(cls, checkpoint_path: str) -> str:
        """Return checkpoint path suitable for official SAM3 builder.

        The OSS SAM3 builder expects a torch checkpoint (.pt). If the user placed
        Hugging Face-style `model.safetensors`, convert it once and reuse.
        """
        ckpt = Path(str(checkpoint_path))
        if ckpt.suffix.lower() != ".safetensors":
            return str(ckpt)

        converted = ckpt.with_suffix(".converted.pt")
        if converted.exists() and converted.stat().st_mtime >= ckpt.stat().st_mtime:
            return str(converted)

        try:
            from safetensors.torch import load_file as load_safetensors
        except Exception as exc:
            raise RuntimeError(
                "Found SAM3 safetensors weights but safetensors package is missing. "
                "Install it with: pip install safetensors"
            ) from exc

        try:
            import torch

            state_dict = load_safetensors(str(ckpt))
            torch.save(state_dict, str(converted))
            logger.info("[SAM3] converted %s -> %s", ckpt.name, converted.name)
            return str(converted)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to convert SAM3 safetensors checkpoint: {ckpt}"
            ) from exc

    @staticmethod
    def _select_device() -> str:
        try:
            return str(get_device().type).strip().lower() or "cpu"
        except Exception:
            return "cpu"

    @staticmethod
    def _should_retry_on_cpu(exc: Exception) -> bool:
        message = str(exc)
        markers = [
            "MPS backend out of memory",
            "not currently implemented for the MPS device",
            "BFloat16 is not supported on MPS",
            "Input type (MPS",
        ]
        return any(marker in message for marker in markers)

    @classmethod
    def consume_runtime_notice(cls) -> str:
        note = str(cls._RUNTIME_NOTICE or "").strip()
        cls._RUNTIME_NOTICE = ""
        return note

    @staticmethod
    def _rewrite_requested_device(requested_device: Any, runtime_device: str) -> Any:
        if requested_device == "cuda" and runtime_device != "cuda":
            return runtime_device
        return requested_device

    @classmethod
    def _install_runtime_compatibility_shims(cls, runtime_device: str) -> None:
        try:
            import torch
        except Exception:
            return

        if runtime_device == cls._COMPAT_PATCH_DEVICE:
            return

        if not hasattr(torch, "compiler"):
            torch.compiler = types.SimpleNamespace()
        if not hasattr(torch.compiler, "is_dynamo_compiling"):
            torch.compiler.is_dynamo_compiling = lambda: False

        if not hasattr(torch.nn, "attention"):
            try:
                from torch.backends.cuda import SDPBackend, sdp_kernel as legacy_sdp_kernel

                attention_mod = types.ModuleType("torch.nn.attention")
                attention_mod.SDPBackend = SDPBackend

                @contextmanager
                def compat_sdpa_kernel(backends):
                    if not isinstance(backends, (list, tuple, set)):
                        backends = [backends]
                    requested = set(backends)
                    with legacy_sdp_kernel(
                        enable_math=SDPBackend.MATH in requested,
                        enable_flash=SDPBackend.FLASH_ATTENTION in requested,
                        enable_mem_efficient=SDPBackend.EFFICIENT_ATTENTION in requested,
                    ):
                        yield

                attention_mod.sdpa_kernel = compat_sdpa_kernel
                sys.modules["torch.nn.attention"] = attention_mod
                torch.nn.attention = attention_mod
            except Exception:
                pass

        if "sam3.model.edt" not in sys.modules:
            if importlib.util.find_spec("triton") is None:
                edt_mod = types.ModuleType("sam3.model.edt")

                def edt_triton(data):
                    return torch.zeros_like(data, dtype=torch.float32)

                edt_mod.edt_triton = edt_triton
                sys.modules["sam3.model.edt"] = edt_mod

        try:
            import sam3.model.decoder as sam3_decoder
            import sam3.model.geometry_encoders as sam3_geometry_encoders
            import sam3.model.sam3_image_processor as sam3_image_processor
            import sam3.model.position_encoding as sam3_position_encoding
            import sam3.model.vitdet as sam3_vitdet
            import sam3.perflib.fused as sam3_fused
        except Exception:
            return

        patched_device = runtime_device

        def patched_position_embedding_init(
            self,
            num_pos_feats,
            temperature: int = 10000,
            normalize: bool = True,
            scale: float | None = None,
            precompute_resolution: int | None = None,
        ):
            torch.nn.Module.__init__(self)
            assert num_pos_feats % 2 == 0, "Expecting even model width"
            self.num_pos_feats = num_pos_feats // 2
            self.temperature = temperature
            self.normalize = normalize
            if scale is not None and normalize is False:
                raise ValueError("normalize should be True if scale is passed")
            if scale is None:
                scale = 2 * math.pi
            self.scale = scale

            self.cache = {}
            if precompute_resolution is not None:
                precompute_sizes = [
                    (int(precompute_resolution // 3.5), int(precompute_resolution // 3.5)),
                    (precompute_resolution // 4, precompute_resolution // 4),
                    (int(precompute_resolution // 7), int(precompute_resolution // 7)),
                    (precompute_resolution // 8, precompute_resolution // 8),
                    (int(precompute_resolution // 14), int(precompute_resolution // 14)),
                    (precompute_resolution // 16, precompute_resolution // 16),
                    (int(precompute_resolution // 28), int(precompute_resolution // 28)),
                    (precompute_resolution // 32, precompute_resolution // 32),
                ]
                tensor_device = cls._rewrite_requested_device("cuda", patched_device)
                for size in precompute_sizes:
                    tensors = torch.zeros((1, 1) + size, device=tensor_device)
                    self.forward(tensors)
                    self.cache[size] = self.cache[size].clone().detach()

        orig_get_coords = sam3_decoder.TransformerDecoder._get_coords

        orig_addmm_act = sam3_fused.addmm_act

        def patched_get_coords(height, width, device):
            device = cls._rewrite_requested_device(device, patched_device)
            return orig_get_coords(height, width, device)

        def patched_addmm_act(activation, linear, mat1):
            if patched_device == "cuda":
                return orig_addmm_act(activation, linear, mat1)

            if torch.is_grad_enabled():
                raise ValueError("Expected grad to be disabled.")

            output = torch.nn.functional.linear(
                mat1.to(linear.weight.dtype),
                linear.weight.detach(),
                linear.bias.detach() if linear.bias is not None else None,
            )
            if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
                return torch.nn.functional.relu(output)
            if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
                return torch.nn.functional.gelu(output)
            raise ValueError(f"Unexpected activation {activation}")

        def patched_encode_boxes(self, boxes, boxes_mask, boxes_labels, img_feats):
            boxes_embed = None
            n_boxes, bs = boxes.shape[:2]

            if self.boxes_direct_project is not None:
                proj = self.boxes_direct_project(boxes)
                assert boxes_embed is None
                boxes_embed = proj

            if self.boxes_pool_project is not None:
                height, width = img_feats.shape[-2:]
                boxes_xyxy = sam3_geometry_encoders.box_cxcywh_to_xyxy(boxes)
                scale = torch.tensor([width, height, width, height], dtype=boxes_xyxy.dtype)
                if patched_device == "cuda":
                    scale = scale.pin_memory().to(device=boxes_xyxy.device, non_blocking=True)
                else:
                    scale = scale.to(device=boxes_xyxy.device)
                scale = scale.view(1, 1, 4)
                boxes_xyxy = boxes_xyxy * scale
                sampled = sam3_geometry_encoders.torchvision.ops.roi_align(
                    img_feats,
                    boxes_xyxy.float().transpose(0, 1).unbind(0),
                    self.roi_size,
                )
                assert list(sampled.shape) == [
                    bs * n_boxes,
                    self.d_model,
                    self.roi_size,
                    self.roi_size,
                ]
                proj = self.boxes_pool_project(sampled)
                proj = proj.view(bs, n_boxes, self.d_model).transpose(0, 1)
                if boxes_embed is None:
                    boxes_embed = proj
                else:
                    boxes_embed = boxes_embed + proj

            if self.boxes_pos_enc_project is not None:
                cx, cy, width, height = boxes.unbind(-1)
                enc = self.pos_enc.encode_boxes(
                    cx.flatten(), cy.flatten(), width.flatten(), height.flatten()
                )
                enc = enc.view(boxes.shape[0], boxes.shape[1], enc.shape[-1])

                proj = self.boxes_pos_enc_project(enc)
                if boxes_embed is None:
                    boxes_embed = proj
                else:
                    boxes_embed = boxes_embed + proj

            type_embed = self.label_embed(boxes_labels.long())
            return type_embed + boxes_embed, boxes_mask

        @torch.inference_mode()
        def patched_set_image(self, image, state=None):
            if state is None:
                state = {}

            if isinstance(image, Image.Image):
                width, height = image.size
            elif isinstance(image, (torch.Tensor, np.ndarray)):
                height, width = image.shape[-2:]
            else:
                raise ValueError("Image must be a PIL image or a tensor")

            image_tensor = sam3_image_processor.v2.functional.to_image(image)
            image_tensor = self.transform(image_tensor).unsqueeze(0).to(self.device)

            state["original_height"] = height
            state["original_width"] = width
            state["backbone_out"] = self.model.backbone.forward_image(image_tensor)
            inst_interactivity_en = self.model.inst_interactive_predictor is not None
            if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
                sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
                sam2_backbone_out["backbone_fpn"][0] = (
                    self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                        sam2_backbone_out["backbone_fpn"][0]
                    )
                )
                sam2_backbone_out["backbone_fpn"][1] = (
                    self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                        sam2_backbone_out["backbone_fpn"][1]
                    )
                )
            return state

        @torch.inference_mode()
        def patched_set_image_batch(self, images, state=None):
            if state is None:
                state = {}

            if not isinstance(images, list):
                raise ValueError("Images must be a list of PIL images or tensors")
            assert len(images) > 0, "Images list must not be empty"
            assert isinstance(images[0], Image.Image), "Images must be a list of PIL images"

            state["original_heights"] = [image.height for image in images]
            state["original_widths"] = [image.width for image in images]

            batch = [self.transform(sam3_image_processor.v2.functional.to_image(image)) for image in images]
            batch = torch.stack(batch, dim=0).to(self.device)
            state["backbone_out"] = self.model.backbone.forward_image(batch)
            inst_interactivity_en = self.model.inst_interactive_predictor is not None
            if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
                sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
                sam2_backbone_out["backbone_fpn"][0] = (
                    self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                        sam2_backbone_out["backbone_fpn"][0]
                    )
                )
                sam2_backbone_out["backbone_fpn"][1] = (
                    self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                        sam2_backbone_out["backbone_fpn"][1]
                    )
                )
            return state

        sam3_position_encoding.PositionEmbeddingSine.__init__ = patched_position_embedding_init
        sam3_decoder.TransformerDecoder._get_coords = staticmethod(patched_get_coords)
        sam3_geometry_encoders.SequenceGeometryEncoder._encode_boxes = patched_encode_boxes
        sam3_image_processor.Sam3Processor.set_image = patched_set_image
        sam3_image_processor.Sam3Processor.set_image_batch = patched_set_image_batch
        sam3_fused.addmm_act = patched_addmm_act
        sam3_vitdet.addmm_act = patched_addmm_act
        cls._COMPAT_PATCH_DEVICE = runtime_device

    @staticmethod
    def _to_pil_image(image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        else:
            raise ValueError("SAM3 image must be a PIL image or an HxW/HxWxC array")

        if arr.dtype != np.uint8:
            arr_f = arr.astype(np.float32)
            max_val = float(np.nanmax(arr_f)) if arr_f.size else 0.0
            if max_val <= 1.01:
                arr = np.clip(arr_f * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr_f, 0, 255).astype(np.uint8)

        return Image.fromarray(arr, mode="RGB")

    @classmethod
    def _get_or_build_image_runtime(
        cls,
        model_type: str,
        preferred_device: str | None = None,
    ) -> dict[str, Any]:
        if model_type not in {"sam3", "sam3.1"}:
            raise ValueError(f"Unsupported SAM3 model type: {model_type}")

        checkpoint_path = cls._resolve_checkpoint_path(model_type)
        if checkpoint_path is None:
            expected = cls.SAM3_CKPT_CANDIDATES.get(model_type, ["<checkpoint>.pt"])[0]
            raise RuntimeError(
                f"SAM3 checkpoint not found for '{model_type}'. "
                f"Place '{expected}' into {cls._model_dir_for(model_type)}"
            )

        device = preferred_device or cls._select_device()
        checkpoint_for_builder = cls._prepare_checkpoint_for_builder(checkpoint_path)
        cached = cls._IMAGE_RUNTIME_CACHE.get(model_type)
        if (
            cached is not None
            and cached.get("checkpoint_path") == checkpoint_for_builder
            and cached.get("device") == device
        ):
            return cached

        cls._install_runtime_compatibility_shims(device)

        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ModuleNotFoundError as exc:
            missing_name = str(getattr(exc, "name", "") or "").strip()
            if missing_name == "sam3":
                raise RuntimeError(
                    "SAM3 package is not installed in the active environment. "
                    "Install it with: pip install git+https://github.com/facebookresearch/sam3.git"
                ) from exc
            if missing_name == "torch.nn.attention":
                try:
                    import torch

                    torch_version = str(torch.__version__)
                except Exception:
                    torch_version = "unknown"
                raise RuntimeError(
                    "Installed SAM3 requires a newer PyTorch runtime than the current environment provides. "
                    f"Current torch={torch_version}; missing module: torch.nn.attention. "
                    "SAM3 upstream targets much newer torch builds, so this environment likely needs a dedicated SAM3 venv."
                ) from exc
            raise RuntimeError(
                f"SAM3 import failed due to missing dependency: {missing_name or exc!s}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"SAM3 import failed during initialization: {exc}"
            ) from exc

        model = build_sam3_image_model(
            checkpoint_path=checkpoint_for_builder,
            load_from_HF=False,
            device=device,
            eval_mode=True,
            enable_inst_interactivity=True,
        )
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        processor = Sam3Processor(model, device=device)
        runtime = {
            "model": model,
            "processor": processor,
            "checkpoint_path": checkpoint_for_builder,
            "device": device,
        }
        cls._IMAGE_RUNTIME_CACHE[model_type] = runtime
        logger.info("[SAM3] image runtime initialized (%s, %s)", model_type, device)
        return runtime

    @classmethod
    def _run_interactive_inference(
        cls,
        processor: Any,
        pil_image: Image.Image,
        points: list,
    ) -> list[np.ndarray]:
        """Use SAM2-style interactive predictor for point prompts.

        The tracker's backbone is None (build_tracker uses with_backbone=False).
        Instead of calling interactive.set_image() — which would crash — we call
        processor.set_image() first (which encodes the image via the main VL backbone)
        and then inject the resulting sam2_backbone_out directly into the interactive
        predictor's feature cache, skipping forward_image entirely.
        """
        import torch

        interactive = getattr(getattr(processor, "model", None), "inst_interactive_predictor", None)
        if interactive is None:
            return []

        point_coords: list[list[float]] = []
        point_labels: list[int] = []
        for raw_point in points:
            if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 3:
                continue
            try:
                point_coords.append([float(raw_point[0]), float(raw_point[1])])
                point_labels.append(1 if int(raw_point[2]) > 0 else 0)
            except Exception:
                continue

        if not point_coords:
            return []

        # --- Step 1: encode image via main backbone (sam2_backbone_out already post-processed
        #     by conv_s0/conv_s1 inside patched_set_image) ---
        state = processor.set_image(pil_image)
        sam2_bb = state.get("backbone_out", {}).get("sam2_backbone_out")
        if sam2_bb is None:
            logger.warning("[SAM3] sam2_backbone_out not in processor state; interactive path unavailable")
            return []

        h = state["original_height"]
        w = state["original_width"]

        # --- Step 2: build _features for interactive predictor (mirrors SAM3InteractiveImagePredictor.set_image) ---
        with torch.inference_mode():
            interactive.reset_predictor()
            (_, vision_feats, _, _) = interactive.model._prepare_backbone_features(sam2_bb)
            vision_feats[-1] = vision_feats[-1] + interactive.model.no_mem_embed
            feats = [
                feat.permute(1, 2, 0).view(1, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], interactive._bb_feat_sizes[::-1])
            ][::-1]
            interactive._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
            interactive._orig_hw = [(h, w)]
            interactive._is_image_set = True
            interactive._is_batch = False

        # --- Step 3: predict masks ---
        coords = np.array(point_coords, dtype=np.float32)
        labels = np.array(point_labels, dtype=np.int32)
        masks, iou_scores, _ = interactive.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True,
        )
        if masks is None or len(masks) == 0:
            return []

        best_idx = int(np.argmax(iou_scores))
        best = masks[best_idx]
        return [(np.asarray(best) > 0).astype(np.uint8) * 255]

    @classmethod
    def _run_image_inference(
        cls,
        runtime: dict[str, Any],
        pil_image: Image.Image,
        points: list,
        concept: str,
    ) -> list[np.ndarray]:
        processor = runtime["processor"]

        # --- Path 1: point-only prompts → SAM2-style interactive predictor (no grounding blob) ---
        if points and not concept:
            interactive = getattr(getattr(processor, "model", None), "inst_interactive_predictor", None)
            if interactive is not None:
                logger.debug("[SAM3] using interactive predictor for point-only prompt")
                result = cls._run_interactive_inference(processor, pil_image, points)
                if result:
                    return result
                logger.warning("[SAM3] interactive predictor returned no masks, falling back to grounding")

        # --- Path 2: text (+ optional geometric hints) → grounding path ---
        width = max(1, pil_image.width)
        height = max(1, pil_image.height)
        base_conf = float(getattr(processor, "confidence_threshold", 0.5) or 0.5)

        def run_attempt(radius_ratio: float, confidence_threshold: float) -> list[np.ndarray]:
            if hasattr(processor, "confidence_threshold"):
                processor.confidence_threshold = float(confidence_threshold)

            state = processor.set_image(pil_image)
            if concept:
                state = processor.set_text_prompt(prompt=concept, state=state)
            else:
                # Geometric-only fallback: initialize "visual" text embedding.
                state = processor.set_text_prompt(prompt="visual", state=state)

            if points:
                radius_px = max(2.0, min(width, height) * float(radius_ratio))
                for raw_point in points:
                    if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 3:
                        continue
                    try:
                        x = float(raw_point[0])
                        y = float(raw_point[1])
                        label = bool(int(raw_point[2]) > 0)
                    except Exception:
                        continue

                    cx = float(np.clip(x / width, 0.0, 1.0))
                    cy = float(np.clip(y / height, 0.0, 1.0))
                    bw = float(np.clip((2.0 * radius_px) / width, 1e-4, 1.0))
                    bh = float(np.clip((2.0 * radius_px) / height, 1e-4, 1.0))
                    state = processor.add_geometric_prompt(
                        box=[cx, cy, bw, bh],
                        label=label,
                        state=state,
                    )

            return cls._collect_masks(state)

        masks = run_attempt(radius_ratio=0.05, confidence_threshold=base_conf)
        if masks:
            return masks

        # Softer threshold fallbacks (no threshold=0.0 to avoid quality-less blobs)
        for radius_ratio, threshold in ((0.08, min(base_conf, 0.3)), (0.12, 0.1)):
            masks = run_attempt(radius_ratio=radius_ratio, confidence_threshold=threshold)
            if masks:
                logger.info(
                    "[SAM3] grounding fallback succeeded (radius=%.3f, threshold=%.2f)",
                    radius_ratio,
                    threshold,
                )
                return masks

        raise RuntimeError("SAM3 did not return any masks for the provided prompt")

    @staticmethod
    def _collect_masks(state: dict) -> list[np.ndarray]:
        masks = state.get("masks")
        if masks is None:
            return []

        arr = masks
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        else:
            arr = np.asarray(arr)

        # expected layouts: [N,H,W], [N,1,H,W] or [H,W]
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0, :, :]
        elif arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim != 3:
            return []

        out: list[np.ndarray] = []
        for m in arr:
            mm = (np.asarray(m) > 0).astype(np.uint8) * 255
            out.append(mm)
        return out

    @classmethod
    def get_weight_status(cls, model_type: str) -> dict:
        """Return {"state": "ready"|"missing"} for *model_type*."""
        candidates = cls.SAM3_CKPT_CANDIDATES.get(model_type, [])
        if not candidates:
            return {"state": "missing"}
        model_dir = cls._model_dir_for(model_type)
        for filename in candidates:
            path = model_dir / filename
            if path.is_file():
                return {"state": "ready", "path": str(path)}
        return {"state": "missing", "path": str(model_dir / candidates[0])}

    @classmethod
    def download_checkpoint_for(cls, model_type: str, progress_callback=None) -> str:
        """Download SAM3 checkpoint for *model_type*; return local path.

        TODO: SAM3 weights are distributed via HuggingFace and may require
        an access token. Implement the download flow once the HF repo is
        publicly accessible without auth.
        """
        raise NotImplementedError(
            "SAM3 weight download is not yet implemented. "
            "Please download the checkpoint manually from "
            "https://huggingface.co/facebook/sam3 and place it in "
            f"{cls._model_dir_for(model_type)}"
        )

    # ── Inference (image) ────────────────────────────────────────────

    @classmethod
    def predict_image(
        cls,
        model_type: str,
        image,           # PIL.Image or np.ndarray
        points: list,    # list of (x, y, label) where label ∈ {0, 1}
        concept: str = "",
    ) -> list:
        """Run SAM3 image inference and return list of 2D uint8 masks.

        The function supports:
        - text prompt (`concept`)
        - optional point list converted to tiny geometric boxes for refinement
          via `Sam3Processor.add_geometric_prompt(...)`.
        """
        concept = str(concept or "").strip()
        points = points or []
        if not concept and not points:
            raise ValueError("SAM3 requires either a text concept or at least one point prompt")

        pil_image = cls._to_pil_image(image)
        cls._RUNTIME_NOTICE = ""
        runtime = cls._get_or_build_image_runtime(model_type)
        try:
            return cls._run_image_inference(runtime, pil_image, points, concept)
        except Exception as exc:
            if runtime.get("device") == "mps" and cls._should_retry_on_cpu(exc):
                logger.warning("[SAM3] MPS inference failed, retrying on CPU: %s", exc)
                cls._RUNTIME_NOTICE = str(exc)
                cpu_runtime = cls._get_or_build_image_runtime(model_type, preferred_device="cpu")
                return cls._run_image_inference(cpu_runtime, pil_image, points, concept)
            raise
