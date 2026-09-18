import os
import torch
import logging
import numpy as np
from PIL import Image, ImageFilter
from huggingface_hub import hf_hub_download
from typing import Dict, Any, Optional

logger = logging.getLogger("Pic4Free.SuperResolution")

# ---------------------------------------------------------------------------
# RestoreFormer++  ONNX (dnnagy/RestoreFormerPlusPlus)
# ---------------------------------------------------------------------------

class RestoreFormerONNX:
    """
    Wrapper for RestoreFormer++ exported to ONNX.
    Automatically loads the model from HuggingFace on first use.
    Input: PIL image of any size (resized internally to 512x512).
    Output: PIL image at the same size as the input.
    """
    REPO_ID   = "dnnagy/RestoreFormerPlusPlus"
    FILENAME  = "RestoreFormerPlusPlus.onnx"
    INPUT_RES = 512

    def __init__(self, hf_token: Optional[str] = None):
        import onnxruntime as ort

        logger.info(f"[RestoreFormerONNX] Downloading model from {self.REPO_ID}...")
        onnx_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=self.FILENAME,
            token=hf_token,
        )
        logger.info(f"[RestoreFormerONNX] Checkpoint in: {onnx_path}")

        # Prefer CUDA GPU; fall back to CPU
        providers = ort.get_available_providers()
        exec_providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in providers
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(onnx_path, providers=exec_providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info(
            f"[RestoreFormerONNX] ONNX session started "
            f"(provider: {self.session.get_providers()[0]})"
        )

    @torch.no_grad()
    def enhance(self, image: Image.Image) -> Image.Image:
        """
        Restores a facial image with RestoreFormer++.
        The image is resized to 512x512 for inference and
        the result is returned at the original size.
        """
        orig_w, orig_h = image.size

        # Pre-processing: RGB PIL  float32 [-1, 1] CHW.
        # RestoreFormer / FaceRestoreHelper normalizes with mean=0.5/std=0.5,
        # i.e. x_norm = (x/255 - 0.5) / 0.5. Feeding [0, 1] shifts everything
        # and the [-1, 1] output then clips to black -> dark image.
        img_resized = image.resize((self.INPUT_RES, self.INPUT_RES), Image.LANCZOS)
        img_np = np.array(img_resized.convert("RGB"), dtype=np.float32) / 255.0
        img_np = (img_np - 0.5) / 0.5  # [0,1] -> [-1,1]
        img_np = img_np.transpose(2, 0, 1)[np.newaxis, ...]  # [1, 3, 512, 512]

        # Inference
        outputs = self.session.run([self.output_name], {self.input_name: img_np})
        out_np = outputs[0][0]  # [3, 512, 512]

        # Post-processing: CHW float32 [-1,1]  uint8 HWC PIL
        # denormalize: x = (x * 0.5 + 0.5) * 255. Fall back safely if the
        # exported ONNX already returns [0, 1].
        out_chw = out_np
        if out_chw.min() >= -0.05 and out_chw.max() <= 1.05:
            # Looks like [0, 1] output (converter baked denorm in)
            out_np = (out_chw.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            out_np = ((out_chw.transpose(1, 2, 0) * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        result = Image.fromarray(out_np)

        # Return at the original size
        if (orig_w, orig_h) != (self.INPUT_RES, self.INPUT_RES):
            result = result.resize((orig_w, orig_h), Image.LANCZOS)

        return result

    # Compatibility with Sequential Swapping of pipeline.py
    # (face_restorer.to(device)  ONNX does not use torch, is ignored without error)
    def to(self, *args, **kwargs):
        return self


# ---------------------------------------------------------------------------
# SUPIR Upscaler  Fanghua-Yu/SUPIR (camenduru/SUPIR checkpoint)
# ---------------------------------------------------------------------------

class SUPIRUpscaler(torch.nn.Module):
    """
    Wrapper for SUPIR (SDXL-guided Super-Resolution).
    Requires the package 'SUPIR' this installed
    (via git+https://github.com/Fanghua-Yu/SUPIR.git).
    Automatically loads SUPIR-v0Q.ckpt from camenduru/SUPIR.
    """
    SUPIR_REPO     = "camenduru/SUPIR"
    SUPIR_FILENAME = "SUPIR-v0Q.ckpt"

    def __init__(
        self,
        scale_factor: int = 2,
        device: str = "cpu",
        hf_token: Optional[str] = None,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.device = torch.device(device)
        self.hf_token = hf_token
        self._model = None  # lazy loading; the model is instantiated in _load_model()

        logger.info(
            f"[SUPIRUpscaler] Downloading {self.SUPIR_FILENAME} "
            f"from {self.SUPIR_REPO}..."
        )
        self.ckpt_path = hf_hub_download(
            repo_id=self.SUPIR_REPO,
            filename=self.SUPIR_FILENAME,
            token=hf_token,
        )
        logger.info(f"[SUPIRUpscaler] Checkpoint in: {self.ckpt_path}")

    def _load_model(self):
        """
        Loads the full SUPIR model with the actual architecture.
        Called the first time upscale() is invoked.
        """
        try:
            from SUPIR.util import create_SUPIR_model, load_state_dict
        except ImportError as e:
            raise ImportError(
                "The SUPIR package is not installed. "
                "Ensure the Slurm script runs: "
                "pip install git+https://github.com/Fanghua-Yu/SUPIR.git  # Ensure the main branch is up to date."
            ) from e

        # Locate the YAML configuration file from the installed package
        import SUPIR as _supir_pkg
        supir_pkg_root = os.path.dirname(_supir_pkg.__file__)
        config_path = os.path.join(
            os.path.dirname(supir_pkg_root), "options", "SUPIR_v0.yaml"
        )
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"SUPIR configuration YAML not found at: {config_path}. "
                "Verify that the package was installed from the complete repository."
            )

        # Create an in-memory patched copy of the YAML to disable
        # the automatic loading of SDXL_CKPT and SUPIR_CKPT (set to None
        # and then load the checkpoint manuallyly with load_state_dict).
        from omegaconf import OmegaConf
        import tempfile

        cfg = OmegaConf.load(config_path)
        cfg.SDXL_CKPT = None
        cfg.SUPIR_CKPT = None

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp:
            OmegaConf.save(cfg, tmp.name)
            patched_config = tmp.name

        try:
            logger.info("[SUPIRUpscaler] Instantiating architecture SUPIR...")
            model = create_SUPIR_model(patched_config, SUPIR_sign=None)

            logger.info(
                f"[SUPIRUpscaler] Loading weights SUPIR-v0Q "
                f"from {self.ckpt_path}..."
            )
            state_dict = load_state_dict(self.ckpt_path, location="cpu")
            model.load_state_dict(state_dict, strict=False)
        finally:
            os.unlink(patched_config)

        model.eval()
        self._model = model
        logger.info("[SUPIRUpscaler] SUPIR model loaded successfully.")

    def to(self, *args, **kwargs):
        """Override .to() to move the internal model if already loaded."""
        super().to(*args, **kwargs)
        if self._model is not None:
            self._model = self._model.to(*args, **kwargs)
        # Update self.device if supplied as arg positional or kwarg
        if args and isinstance(args[0], (str, torch.device)):
            self.device = torch.device(args[0])
        if "device" in kwargs:
            self.device = torch.device(kwargs["device"])
        return self

    @torch.no_grad()
    def upscale(self, image: Image.Image, target_size: tuple) -> Image.Image:
        """
        Applies SUPIR to scale the image to the target size.

        Args:
            image: Input PIL image.
            target_size: Tuple (width, height) of the expected result.

        Returns:
            Restored and upscaled PIL image.
        """
        try:
            from SUPIR.util import PIL2Tensor, Tensor2PIL
        except ImportError as e:
            raise ImportError(
                "The SUPIR package is not installed. "
                "Install it with: pip install git+https://github.com/Fanghua-Yu/SUPIR.git  # Check compatibility with your environment."
            ) from e

        if self._model is None:
            self._load_model()

        model = self._model

        # Pre-processing: PIL to SUPIR tensor (normalized, CHW float)
        lq_tensor, h_orig, w_orig = PIL2Tensor(image, upsacle=self.scale_factor)
        lq_tensor = lq_tensor.unsqueeze(0).to(self.device)

        # Restoration prompt without LLaVA
        caption = (
            "Cinematic, highly detailed, hyper-realistic, sharp, "
            "perfect skin pore detail, 4K, RAW photo quality."
        )
        neg_prompt = "blurry, low quality, ugly, deformed, artifact, watermark"

        result_tensors, _ = model.batchify_bsr(
            lq=lq_tensor,
            prompt=caption,
            a_prompt="",
            n_prompt=neg_prompt,
            num_samples=1,
            random_seed=True,
            upscale=self.scale_factor,
            edm_steps=50,
            s_stage1=-1,
            s_stage2=1.0,
            s_cfg=7.5,
            s_churn=5,
            s_noise=1.003,
            control_scale=1.0,
            control_tile_size=512,
            use_linear_CFG=False,
            use_linear_control_scale=False,
            cfg_scale_start=1.0,
            control_scale_start=0.0,
        )

        result_pil = Tensor2PIL(result_tensors[0], h_orig, w_orig)

        # Final adjustment to the exact target size if there is a pixel discrepancy
        target_w, target_h = target_size
        if result_pil.size != (target_w, target_h):
            result_pil = result_pil.resize((target_w, target_h), Image.LANCZOS)

        return result_pil


# ---------------------------------------------------------------------------
# FaceLoRAUpscaler  HD upscaler with LoRA facial refinement (Stage 4, via "lora")
# ---------------------------------------------------------------------------

class FaceLoRAUpscaler:
    """HD output (lado largo 1920) + refinement via face crops with FLUX.1-dev.

    for each face with an assigned identity (vote A/B from Stage 2) is cropped
    the enlarged face is passed through FluxImg2ImgPipeline with the person-specific LoRA
    (trigger P4F_A/P4F_B) and is composited back with mask feathered. Rostros without
    asignar reciben generic pass (without LoRA) if refine_unassigned=True.
    Multi-person correct: each face sees only its own LoRA.

    if a LoRA file does not exist yet (training pending), that label falls back to
    generic pass with warning: the rest of the flow is validated the same way.
    """

    def __init__(
        self,
        flux_model_id: str = "black-forest-labs/FLUX.1-dev",
        lora_paths: Optional[Dict[str, str]] = None,
        triggers: Optional[Dict[str, str]] = None,
        device: str = "cpu",
        hf_token: Optional[str] = None,
        strength: float = 0.45,
        guidance: float = 3.5,
        steps: int = 28,
        seed: int = 42,
        crop_expand: float = 2.0,
        feather_px: int = 24,
        min_crop: int = 512,
        target_long_edge: int = 1920,
        refine_unassigned: bool = True,
    ):
        self.flux_model_id = flux_model_id
        self.lora_paths = lora_paths or {}
        self.triggers = triggers or {}
        self.device = device
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.strength = float(strength)
        self.guidance = float(guidance)
        self.steps = int(steps)
        self.seed = int(seed)
        self.crop_expand = float(crop_expand)
        self.feather_px = int(feather_px)
        self.min_crop = int(min_crop)
        self.target_long_edge = int(target_long_edge)
        self.refine_unassigned = bool(refine_unassigned)
        self.pipe = None
        self.adapters: Dict[str, str] = {}

    def _ensure_pipe(self):
        if self.pipe is not None:
            return
        from diffusers import FluxImg2ImgPipeline

        logger.info(f"[FaceLoRA] Loading {self.flux_model_id} (img2img)...")
        self.pipe = FluxImg2ImgPipeline.from_pretrained(
            self.flux_model_id, torch_dtype=torch.bfloat16, token=self.hf_token
        )
        self.pipe.enable_model_cpu_offload()
        for label, path in self.lora_paths.items():
            if path and os.path.exists(path):
                try:
                    if not self._weights_finite(path):
                        raise ValueError("weights with NaN/Inf: checkpoint contaminado")
                    self.pipe.load_lora_weights(path, adapter_name=f"person_{label}")
                    self.adapters[label] = f"person_{label}"
                    logger.info(f"[FaceLoRA] LoRA {label} loaded: {path}")
                except Exception as e:
                    logger.warning(f"[FaceLoRA] LoRA {label} no cargable ({e}): pase generic.")
            else:
                logger.warning(f"[FaceLoRA] without file LoRA for {label} ({path}): generic pass.")
        if not self.adapters:
            logger.warning("[FaceLoRA] Ningun LoRA loaded: all the refinement sera generic.")

    @staticmethod
    def _weights_finite(path: str) -> bool:
        """Quality gate: rejects safetensors with NaN/Inf before loading them.

        A contaminated checkpoint (e.g. after a CUBLAS crash halfway through the
        training) would paste a black crop onto the face. A generic pass is better.
        """
        try:
            from safetensors.torch import load_file

            sd = load_file(path, device="cpu")
            bad = sum(int(torch.isnan(v).sum() + torch.isinf(v).sum()) for v in sd.values())
            if bad:
                logger.warning(f"[FaceLoRA] {path}: {bad} params NaN/Inf.")
                return False
            return True
        except Exception as e:
            logger.warning(f"[FaceLoRA] Could not verify {path} ({e}).")
            return False

    def _use_adapter(self, label: Optional[str]):
        """Activates the label LoRA or disables it for the generic pass."""
        try:
            if label in self.adapters:
                self.pipe.set_adapters([self.adapters[label]])
            else:
                try:
                    self.pipe.set_adapters([])
                except Exception:
                    self.pipe.disable_lora()
        except Exception as e:
            logger.warning(f"[FaceLoRA] Could not switch adapter ({e}); continuing.")

    @staticmethod
    def _snap16(v: int) -> int:
        return max(16, (int(v) // 16) * 16)

    def _feathered_paste(self, base: Image.Image, patch: Image.Image, box) -> Image.Image:
        x1, y1, x2, y2 = (int(v) for v in box)
        w, h = x2 - x1, y2 - y1
        if patch.size != (w, h):
            patch = patch.resize((w, h), Image.LANCZOS)
        m = Image.new("L", (w, h), 0)
        inset = min(w, h) // 8
        from PIL import ImageDraw

        ImageDraw.Draw(m).rounded_rectangle([inset, inset, w - inset, h - inset], radius=inset, fill=255)
        if self.feather_px > 0:
            m = m.filter(ImageFilter.GaussianBlur(radius=self.feather_px))
        base.paste(patch, (x1, y1), m)
        return base

    @torch.no_grad()
    def upscale_hd(self, image: Image.Image, faces_detail: Optional[Dict[str, Any]] = None) -> Image.Image:
        """Upscales to HD and refines each face with its LoRA. Never fails: when uncertain, returns the base."""
        image = image.convert("RGB")
        W, H = image.size
        scale = self.target_long_edge / max(W, H)
        TW, TH = max(1, round(W * scale)), max(1, round(H * scale))
        base = image.resize((TW, TH), Image.LANCZOS) if (TW, TH) != (W, H) else image.copy()
        logger.info(f"[FaceLoRA] base HD: {(W, H)} -> {(TW, TH)}")

        faces = (faces_detail or {}).get("faces", []) if faces_detail else []
        assignment = (faces_detail or {}).get("assignment", {}) if faces_detail else {}
        if not faces:
            logger.warning("[FaceLoRA] without faces for refinar: returns base HD.")
            return base

        try:
            self._ensure_pipe()
        except Exception as e:
            logger.error(f"[FaceLoRA] without img2img pipeline ({e}): returns base HD.")
            return base

        sx, sy = TW / W, TH / H
        out = base.copy()
        for f in faces:
            idx = str(f.get("idx"))
            try:
                x1, y1, x2, y2 = (float(v) for v in f.get("bbox", []))
            except Exception:
                logger.warning(f"[FaceLoRA] bbox invalida in face {idx}; is skipped.")
                continue
            a = assignment.get(idx, {})
            label = a.get("ref") if a else None
            if label is None and not self.refine_unassigned:
                logger.info(f"[FaceLoRA] face {idx} without asignar: is skipped.")
                continue
            # HD bbox coordinates -> expanded square -> clipped -> multiple of 16.
            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            half = max(x2 - x1, y2 - y1) / 2.0 * self.crop_expand
            nx1, ny1 = max(0, cx - half), max(0, cy - half)
            nx2, ny2 = min(TW, cx + half), min(TH, cy + half)
            nx1, ny1 = self._snap16(nx1), self._snap16(ny1)
            nx2, ny2 = self._snap16(nx2), self._snap16(ny2)
            if nx2 - nx1 < 16 or ny2 - ny1 < 16:
                continue
            crop = base.crop((nx1, ny1, nx2, ny2))
            proc_w, proc_h = crop.size
            if min(proc_w, proc_h) < self.min_crop:
                k = self.min_crop / min(proc_w, proc_h)
                proc_w, proc_h = self._snap16(proc_w * k), self._snap16(proc_h * k)
                crop_proc = crop.resize((proc_w, proc_h), Image.LANCZOS)
            else:
                crop_proc = crop

            trigger = self.triggers.get(label, "") if label else ""
            prompt = (
                f"{trigger} portrait photo, sharp facial detail, natural skin pores, "
                f"clear iris detail, photorealistic, high definition"
                if trigger
                else "clean photorealistic portrait detail, natural skin texture, sharp eyes, high definition"
            ).strip()
            self._use_adapter(label)
            gen = torch.Generator(device="cpu").manual_seed(self.seed + int(f.get("idx", 0)))
            try:
                refined = self.pipe(
                    image=crop_proc,
                    prompt=prompt,
                    strength=self.strength,
                    guidance_scale=self.guidance,
                    num_inference_steps=self.steps,
                    generator=gen,
                ).images[0]
            except Exception as e:
                logger.warning(f"[FaceLoRA] img2img failure in face {idx} ({e}): is conserva original.")
                continue
            out = self._feathered_paste(out, refined, (nx1, ny1, nx2, ny2))
            logger.info(
                f"[FaceLoRA] face {idx} refinado (ref={label or 'generic'}, "
                f"crop={(nx2 - nx1)}x{(ny2 - ny1)})."
            )
        return out

    # Compatibility with Sequential Swapping (the offload of the pipeline controls).
    def to(self, *args, **kwargs):
        return self


# ---------------------------------------------------------------------------
# SUPIRRestoreFormer  Main orchestrator (Stage 4 of the pipeline)
# ---------------------------------------------------------------------------

class SUPIRRestoreFormer:
    """
    Dual SUPIR (SDXL-guided) + RestoreFormer++ Pipeline.
    Restores micro-textures (pores, iris) at 4K resolution.

    Actual models:
    - face_restorer : RestoreFormerONNX  (dnnagy/RestoreFormerPlusPlus)
    - supir_model   : SUPIRUpscaler      (camenduru/SUPIR / SUPIR-v0Q.ckpt)
    """

    def __init__(self, model_id: str, scale_factor: int, device: str = "cpu",
                 enable_upscaling: bool = False, hf_token: Optional[str] = None,
                 config=None):
        self.model_id         = model_id
        self.scale_factor     = scale_factor
        self.device           = device
        self.enable_upscaling = enable_upscaling
        self.config = config

        sr = getattr(config, "superres", None) if config is not None else None
        self.refine_backend = str(getattr(sr, "face_refine_backend", "lora")).lower() if sr is not None else "lora"

        logger.info(
            f"Initializing Stage 4 (refinement backend={self.refine_backend}) "
            f"on {device}"
        )

        # HF token: parameter > env > standard project file.
        if not hf_token:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            token_path = "hf_token.txt"
            if os.path.exists(token_path):
                with open(token_path, "r") as f:
                    hf_token = f.read().strip() or None
        self.hf_token = hf_token
        self.face_restorer = None
        self.supir_model = None

        try:
            # LoRA refinement: lightweight construction; the DiT img2img model
            # loads lazily on the first upscale.
            g = (lambda k, d: getattr(sr, k, d)) if sr is not None else (lambda k, d: d)
            self.lora_refiner = FaceLoRAUpscaler(
                flux_model_id=g("flux_model_id", "black-forest-labs/FLUX.1-dev"),
                lora_paths={"A": g("lora_a_path", ""), "B": g("lora_b_path", "")},
                triggers={"A": g("lora_trigger_a", ""), "B": g("lora_trigger_b", "")},
                device=device,
                hf_token=hf_token,
                strength=g("lora_strength", 0.45),
                guidance=g("lora_guidance", 3.5),
                steps=g("lora_steps", 28),
                seed=getattr(getattr(config, "flux", None), "seed", 42) if config is not None else 42,
                crop_expand=g("lora_crop_expand", 2.0),
                feather_px=g("lora_feather_px", 24),
                min_crop=g("lora_min_crop", 512),
                target_long_edge=g("target_long_edge", 1920),
                refine_unassigned=g("refine_unassigned", True),
            )

            if self.refine_backend != "lora":
                self._ensure_legacy_models()
        except Exception as e:
            logger.error(f"Error initializing Super-Resolution models: {e}")
            raise

    def _ensure_legacy_models(self):
        if self.face_restorer is None:
            logger.info("Loading RestoreFormer++ (ONNX) into RAM...")
            self.face_restorer = RestoreFormerONNX(hf_token=self.hf_token)
        if self.supir_model is None:
            logger.info("Downloading and preparing SUPIR-v0Q...")
            self.supir_model = SUPIRUpscaler(
                scale_factor=self.scale_factor,
                device=self.device,
                hf_token=self.hf_token,
            )

    @torch.no_grad()
    def upscale(self, image: Image.Image, faces_detail: Optional[Dict[str, Any]] = None) -> Image.Image:
        """
        Post-processing and upscaling. Assumes that the active models
        have been moved to CUDA during Sequential Stage Swapping.
        with backend "lora": HD + refinement facial by LoRA (ignora RestoreFormer).
        """
        if self.refine_backend == "lora":
            logger.info("Applying FaceLoRAUpscaler (HD + per-face refinement)...")
            try:
                return self.lora_refiner.upscale_hd(image, faces_detail)
            except torch.cuda.OutOfMemoryError:
                logger.error("OOM detected in FaceLoRA.")
                raise
            except Exception as e:
                logger.error(f"Error in FaceLoRA, falling back to RestoreFormer++: {e}")
                # Continue below with the legacy path.

        logger.info("Applying RestoreFormer++ + SUPIR...")

        try:
            self._ensure_legacy_models()
            # 1. Restaurar detalles faciales with RestoreFormer++ (ONNX)
            logger.info("[Stage 4.1] RestoreFormer++ face restoration...")
            in_arr = np.array(image.convert("RGB"), dtype=np.float32)
            in_mean = float(in_arr.mean())
            image = self.face_restorer.enhance(image)
            out_arr = np.array(image.convert("RGB"), dtype=np.float32)
            out_mean = float(out_arr.mean())
            logger.info(f"[Stage 4.1] brightness in={in_mean:.1f} out={out_mean:.1f}")
            if out_mean < 30 or (out_mean < in_mean * 0.75):
                logger.warning(
                    "[Stage 4.1] RestoreFormer++ collapsed the brightness "
                    f"(in={in_mean:.1f} -> out={out_mean:.1f}). "
                    "Possible ONNX normalization mismatch; the pre-restoration image is preserved."
                )
                image = Image.fromarray(in_arr.astype(np.uint8))

            # 2. Escalar globalmente with SUPIR (opcional)
            if self.enable_upscaling:
                logger.info("[Stage 4.2] SUPIR upscaling...")
                width, height = image.size
                target_size = (width * self.scale_factor, height * self.scale_factor)
                image = self.supir_model.upscale(image, target_size=target_size)
            else:
                logger.info("[Stage 4.2] SUPIR upscaling skipped (enable_upscaling=False).")

            return image

        except torch.cuda.OutOfMemoryError:
            logger.error("OOM detected in Super-Resolution.")
            raise
        except Exception as e:
            logger.error(f"Error during upscaling: {e}")
            raise