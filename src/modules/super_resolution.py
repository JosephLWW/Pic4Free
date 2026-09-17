import os
import torch
import logging
import numpy as np
from PIL import Image, ImageFilter
from huggingface_hub import hf_hub_download
from typing import Dict, Any, Optional

logger = logging.getLogger("Pic4Free.SuperResolution")

# ---------------------------------------------------------------------------
# RestoreFormer++ — ONNX (dnnagy/RestoreFormerPlusPlus)
# ---------------------------------------------------------------------------

class RestoreFormerONNX:
    """
    Wrapper para RestoreFormer++ exportado a ONNX.
    Descarga automáticamente el modelo desde HuggingFace en el primer uso.
    Entrada: imagen PIL cualquier tamaño (se redimensiona a 512×512 internamente).
    Salida: imagen PIL al mismo tamaño que la entrada.
    """
    REPO_ID   = "dnnagy/RestoreFormerPlusPlus"
    FILENAME  = "RestoreFormerPlusPlus.onnx"
    INPUT_RES = 512

    def __init__(self, hf_token: Optional[str] = None):
        import onnxruntime as ort

        logger.info(f"[RestoreFormerONNX] Descargando modelo desde {self.REPO_ID}...")
        onnx_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=self.FILENAME,
            token=hf_token,
        )
        logger.info(f"[RestoreFormerONNX] Checkpoint en: {onnx_path}")

        # Preferir GPU CUDA; fallback a CPU
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
            f"[RestoreFormerONNX] Sesión ONNX iniciada "
            f"(provider: {self.session.get_providers()[0]})"
        )

    @torch.no_grad()
    def enhance(self, image: Image.Image) -> Image.Image:
        """
        Restaura una imagen facial con RestoreFormer++.
        La imagen se redimensiona a 512×512 para la inferencia y
        el resultado se devuelve al tamaño original.
        """
        orig_w, orig_h = image.size

        # Pre-processing: RGB PIL → float32 [-1, 1] CHW.
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

        # Post-processing: CHW float32 [-1,1] → uint8 HWC PIL
        # De-normalize: x = (y * 0.5 + 0.5) * 255. Fall back safely if the
        # exported ONNX already returns [0, 1].
        out_chw = out_np
        if out_chw.min() >= -0.05 and out_chw.max() <= 1.05:
            # Looks like [0, 1] output (converter baked denorm in)
            out_np = (out_chw.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            out_np = ((out_chw.transpose(1, 2, 0) * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        result = Image.fromarray(out_np)

        # Devolver al tamaño original
        if (orig_w, orig_h) != (self.INPUT_RES, self.INPUT_RES):
            result = result.resize((orig_w, orig_h), Image.LANCZOS)

        return result

    # Compatibilidad con Sequential Swapping de pipeline.py
    # (face_restorer.to(device) — ONNX no usa torch, se ignora sin error)
    def to(self, *args, **kwargs):
        return self


# ---------------------------------------------------------------------------
# SUPIR Upscaler — Fanghua-Yu/SUPIR (camenduru/SUPIR checkpoint)
# ---------------------------------------------------------------------------

class SUPIRUpscaler(torch.nn.Module):
    """
    Wrapper para SUPIR (SDXL-guided Super-Resolution).
    Requiere que el paquete 'SUPIR' esté instalado
    (via git+https://github.com/Fanghua-Yu/SUPIR.git).
    Descarga automáticamente SUPIR-v0Q.ckpt desde camenduru/SUPIR.
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
        self._model = None  # carga lazy; el modelo se instancia en _load_model()

        logger.info(
            f"[SUPIRUpscaler] Descargando {self.SUPIR_FILENAME} "
            f"desde {self.SUPIR_REPO}..."
        )
        self.ckpt_path = hf_hub_download(
            repo_id=self.SUPIR_REPO,
            filename=self.SUPIR_FILENAME,
            token=hf_token,
        )
        logger.info(f"[SUPIRUpscaler] Checkpoint en: {self.ckpt_path}")

    def _load_model(self):
        """
        Carga el modelo SUPIR completo con la arquitectura real.
        Se invoca la primera vez que se llama a upscale().
        """
        try:
            from SUPIR.util import create_SUPIR_model, load_state_dict
        except ImportError as e:
            raise ImportError(
                "El paquete SUPIR no está instalado. "
                "Asegúrate de que el slurm script ejecuta: "
                "pip install git+https://github.com/Fanghua-Yu/SUPIR.git  # Asegúrate de tener el canal principal actualizado."
            ) from e

        # Localizar el fichero YAML de configuración del paquete instalado
        import SUPIR as _supir_pkg
        supir_pkg_root = os.path.dirname(_supir_pkg.__file__)
        config_path = os.path.join(
            os.path.dirname(supir_pkg_root), "options", "SUPIR_v0.yaml"
        )
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No se encontró el YAML de configuración de SUPIR en: {config_path}. "
                "Verifica que el paquete fue instalado desde el repo completo."
            )

        # Crear una copia parcheada del YAML en memoria para deshabilitar
        # la carga automática de SDXL_CKPT y SUPIR_CKPT (los ponemos a None
        # y luego cargamos el checkpoint manualmente con load_state_dict).
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
            logger.info("[SUPIRUpscaler] Instanciando arquitectura SUPIR...")
            model = create_SUPIR_model(patched_config, SUPIR_sign=None)

            logger.info(
                f"[SUPIRUpscaler] Cargando pesos SUPIR-v0Q "
                f"desde {self.ckpt_path}..."
            )
            state_dict = load_state_dict(self.ckpt_path, location="cpu")
            model.load_state_dict(state_dict, strict=False)
        finally:
            os.unlink(patched_config)

        model.eval()
        self._model = model
        logger.info("[SUPIRUpscaler] Modelo SUPIR cargado correctamente.")

    def to(self, *args, **kwargs):
        """Override de .to() para mover el modelo interno si ya está cargado."""
        super().to(*args, **kwargs)
        if self._model is not None:
            self._model = self._model.to(*args, **kwargs)
        # Actualizar self.device si se suministra como arg posicional o kwarg
        if args and isinstance(args[0], (str, torch.device)):
            self.device = torch.device(args[0])
        if "device" in kwargs:
            self.device = torch.device(kwargs["device"])
        return self

    @torch.no_grad()
    def upscale(self, image: Image.Image, target_size: tuple) -> Image.Image:
        """
        Aplica SUPIR para escalar la imagen al tamaño objetivo.

        Args:
            image: Imagen PIL de entrada.
            target_size: Tupla (width, height) del resultado esperado.

        Returns:
            Imagen PIL restaurada y escalada.
        """
        try:
            from SUPIR.util import PIL2Tensor, Tensor2PIL
        except ImportError as e:
            raise ImportError(
                "El paquete SUPIR no está instalado. "
                "Instálalo con: pip install git+https://github.com/Fanghua-Yu/SUPIR.git  # Chequear compatibilidad con tu entorno."
            ) from e

        if self._model is None:
            self._load_model()

        model = self._model

        # Pre-processing: PIL → tensor SUPIR (normalizado, CHW float)
        lq_tensor, h_orig, w_orig = PIL2Tensor(image, upsacle=self.scale_factor)
        lq_tensor = lq_tensor.unsqueeze(0).to(self.device)

        # Prompt de restauración sin LLaVA
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

        # Ajuste final al tamaño objetivo exacto si hay discrepancia de píxeles
        target_w, target_h = target_size
        if result_pil.size != (target_w, target_h):
            result_pil = result_pil.resize((target_w, target_h), Image.LANCZOS)

        return result_pil


# ---------------------------------------------------------------------------
# FaceLoRAUpscaler — Upscaler HD con refino facial por LoRA (Etapa 4, vía "lora")
# ---------------------------------------------------------------------------

class FaceLoRAUpscaler:
    """Salida HD (lado largo 1920) + refino por recorte facial con FLUX.1-dev.

    Por cada rostro con identidad asignada (voto A/B de la Etapa 2) se recorta
    la cara ampliada, se pasa por FluxImg2ImgPipeline con el LoRA DE SU PERSONA
    (trigger P4F_A/P4F_B) y se reintegra con máscara feathered. Rostros sin
    asignar reciben pase genérico (sin LoRA) si refine_unassigned=True.
    Multi-persona correcto: cada cara solo ve su LoRA.

    Si un fichero LoRA no existe aún (entreno pendiente), esa etiqueta cae a
    pase genérico con warning: el resto del flujo se valida igual.
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

        logger.info(f"[FaceLoRA] Cargando {self.flux_model_id} (img2img)...")
        self.pipe = FluxImg2ImgPipeline.from_pretrained(
            self.flux_model_id, torch_dtype=torch.bfloat16, token=self.hf_token
        )
        self.pipe.enable_model_cpu_offload()
        for label, path in self.lora_paths.items():
            if path and os.path.exists(path):
                try:
                    if not self._weights_finite(path):
                        raise ValueError("pesos con NaN/Inf: checkpoint contaminado")
                    self.pipe.load_lora_weights(path, adapter_name=f"person_{label}")
                    self.adapters[label] = f"person_{label}"
                    logger.info(f"[FaceLoRA] LoRA {label} cargado: {path}")
                except Exception as e:
                    logger.warning(f"[FaceLoRA] LoRA {label} no cargable ({e}): pase genérico.")
            else:
                logger.warning(f"[FaceLoRA] Sin fichero LoRA para {label} ({path}): pase genérico.")
        if not self.adapters:
            logger.warning("[FaceLoRA] Ningún LoRA cargado: todo el refino será genérico.")

    @staticmethod
    def _weights_finite(path: str) -> bool:
        """Puerta de calidad: rechaza safetensors con NaN/Inf antes de cargarlos.

        Un checkpoint contaminado (p. ej. tras un crash CUBLAS a mitad del
        entreno) pegaría un recorte negro en la cara. Mejor pase genérico.
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
            logger.warning(f"[FaceLoRA] No se pudo verificar {path} ({e}).")
            return False

    def _use_adapter(self, label: Optional[str]):
        """Activa el LoRA de la etiqueta o lo desactiva para pase genérico."""
        try:
            if label in self.adapters:
                self.pipe.set_adapters([self.adapters[label]])
            else:
                try:
                    self.pipe.set_adapters([])
                except Exception:
                    self.pipe.disable_lora()
        except Exception as e:
            logger.warning(f"[FaceLoRA] No se pudo conmutar adaptador ({e}); se continúa.")

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
        """Sube a HD y refina cada rostro con su LoRA. Nunca tumba: a la duda, devuelve base."""
        image = image.convert("RGB")
        W, H = image.size
        scale = self.target_long_edge / max(W, H)
        TW, TH = max(1, round(W * scale)), max(1, round(H * scale))
        base = image.resize((TW, TH), Image.LANCZOS) if (TW, TH) != (W, H) else image.copy()
        logger.info(f"[FaceLoRA] base HD: {(W, H)} -> {(TW, TH)}")

        faces = (faces_detail or {}).get("faces", []) if faces_detail else []
        assignment = (faces_detail or {}).get("assignment", {}) if faces_detail else {}
        if not faces:
            logger.warning("[FaceLoRA] Sin rostros para refinar: se devuelve base HD.")
            return base

        try:
            self._ensure_pipe()
        except Exception as e:
            logger.error(f"[FaceLoRA] Sin pipeline img2img ({e}): se devuelve base HD.")
            return base

        sx, sy = TW / W, TH / H
        out = base.copy()
        for f in faces:
            idx = str(f.get("idx"))
            try:
                x1, y1, x2, y2 = (float(v) for v in f.get("bbox", []))
            except Exception:
                logger.warning(f"[FaceLoRA] bbox inválida en rostro {idx}; se omite.")
                continue
            a = assignment.get(idx, {})
            label = a.get("ref") if a else None
            if label is None and not self.refine_unassigned:
                logger.info(f"[FaceLoRA] rostro {idx} sin asignar: se omite.")
                continue
            # Bbox a coords HD -> cuadrado expandido -> clip -> múltiplo de 16.
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
                logger.warning(f"[FaceLoRA] img2img fallo en rostro {idx} ({e}): se conserva original.")
                continue
            out = self._feathered_paste(out, refined, (nx1, ny1, nx2, ny2))
            logger.info(
                f"[FaceLoRA] rostro {idx} refinado (ref={label or 'genérico'}, "
                f"crop={(nx2 - nx1)}x{(ny2 - ny1)})."
            )
        return out

    # Compatibilidad con Sequential Swapping (el offload del pipeline manda).
    def to(self, *args, **kwargs):
        return self


# ---------------------------------------------------------------------------
# SUPIRRestoreFormer — Orquestador principal (Etapa 4 del pipeline)
# ---------------------------------------------------------------------------

class SUPIRRestoreFormer:
    """
    Dual SUPIR (SDXL-guided) + RestoreFormer++ Pipeline.
    Restores micro-textures (pores, iris) at 4K resolution.

    Modelos reales:
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
            f"Inicializando Etapa 4 (backend refino={self.refine_backend}) "
            f"en {device}"
        )

        # Token HF: parámetro > env > fichero estándar del proyecto.
        if not hf_token:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            token_path = "hf_token.txt"
            if os.path.exists(token_path):
                with open(token_path, "r") as f:
                    hf_token = f.read().strip() or None
        self.hf_token = hf_token

        try:
            logger.info("Cargando RestoreFormer++ (ONNX) en memoria RAM...")
            self.face_restorer = RestoreFormerONNX(hf_token=hf_token)

            logger.info("Descargando y preparando SUPIR-v0Q...")
            self.supir_model = SUPIRUpscaler(
                scale_factor=scale_factor,
                device=device,
                hf_token=hf_token,
            )

            # Refino LoRA: construcción ligera (el DiT img2img se carga lazy en
            # el primer upscale para no penalizar runs que no lo usan).
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

        except Exception as e:
            logger.error(f"Error al inicializar modelos de Super-Resolución: {e}")
            raise

    @torch.no_grad()
    def upscale(self, image: Image.Image, faces_detail: Optional[Dict[str, Any]] = None) -> Image.Image:
        """
        Post-procesamiento y upscaling. Asume que los modelos activos
        han sido movidos a CUDA mediante Sequential Stage Swapping.
        Con backend "lora": HD + refino facial por LoRA (ignora RestoreFormer).
        """
        if self.refine_backend == "lora":
            logger.info("Aplicando FaceLoRAUpscaler (HD + refino por cara)...")
            try:
                return self.lora_refiner.upscale_hd(image, faces_detail)
            except torch.cuda.OutOfMemoryError:
                logger.error("OOM detectado en FaceLoRA.")
                raise
            except Exception as e:
                logger.error(f"Error en FaceLoRA, fallback a RestoreFormer++: {e}")
                # Sigue abajo con la vía legada.

        logger.info("Aplicando RestoreFormer++ + SUPIR...")

        try:
            # 1. Restaurar detalles faciales con RestoreFormer++ (ONNX)
            logger.info("[Etapa 4.1] RestoreFormer++ face restoration...")
            in_arr = np.array(image.convert("RGB"), dtype=np.float32)
            in_mean = float(in_arr.mean())
            image = self.face_restorer.enhance(image)
            out_arr = np.array(image.convert("RGB"), dtype=np.float32)
            out_mean = float(out_arr.mean())
            logger.info(f"[Etapa 4.1] brightness in={in_mean:.1f} out={out_mean:.1f}")
            if out_mean < 30 or (out_mean < in_mean * 0.75):
                logger.warning(
                    "[Etapa 4.1] RestoreFormer++ colapsó el brillo "
                    f"(in={in_mean:.1f} -> out={out_mean:.1f}). "
                    "Posible mismatch de normalización del ONNX; se conserva la imagen pre-restauración."
                )
                image = Image.fromarray(in_arr.astype(np.uint8))

            # 2. Escalar globalmente con SUPIR (opcional)
            if self.enable_upscaling:
                logger.info("[Etapa 4.2] SUPIR upscaling...")
                width, height = image.size
                target_size = (width * self.scale_factor, height * self.scale_factor)
                image = self.supir_model.upscale(image, target_size=target_size)
            else:
                logger.info("[Etapa 4.2] SUPIR upscaling omitido (enable_upscaling=False).")

            return image

        except torch.cuda.OutOfMemoryError:
            logger.error("OOM detectado en Super Resolución.")
            raise
        except Exception as e:
            logger.error(f"Error durante upscaling: {e}")
            raise