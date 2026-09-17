import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from PIL import Image
from torchvision import transforms
from typing import Dict, Any, Optional
import os
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from diffusers import DiffusionPipeline

from src.modules.pulid_official import IDFormer, PerceiverAttentionCA

logger = logging.getLogger("Pic4Free.FluxInpainter")

# NOTA: se eliminó el FluxPuLIDAttnProcessor casero (to_k_ip/to_v_ip): el
# checkpoint oficial NO contiene esas claves y el diseño no coincide con PuLID.
# La inyección sigue el forward oficial (ToTheBeginning/PuLID, Apache-2.0):
#   img = img + id_weight * pulid_ca[id](id_emb, img)
# tras cada bloque double par y cada bloque single múltiplo de 4.


class FluxPuLIDInpainter:
    def __init__(self, model_id: str, pulid_model_id: str, device: str = "cpu", enable_cpu_offload: bool = True, hf_token: Optional[str] = None, config=None):
        self.model_id = model_id
        self.pulid_model_id = pulid_model_id
        # BUG FIX: se usa el parámetro `device` recibido como única fuente de verdad.
        # Asignarlo directamente a self.device evita discordancias en subcomponentes
        # (procesadores PuLID, embeddings de identidad) que dependen de este atributo.
        self.device = torch.device(device)
        self.torch_dtype = torch.bfloat16
        # Config centralizada (FluxConfig): prompt, guidance, id_scale, dropout_p, steps.
        # Se guarda por referencia para que cambios CLI (OmegaConf) se reflejen.
        self.config = config
        
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if self.hf_token is None:
            candidates = []
            if config is not None and getattr(config, "flux", None) is not None:
                candidates.append(getattr(config.flux, "hf_token_file", None))
            candidates.append("hf_token.txt")
            for token_path in candidates:
                if token_path and os.path.exists(token_path):
                    with open(token_path, "r") as f:
                        self.hf_token = f.read().strip() or None
                    if self.hf_token:
                        break
        
        # Backend: "flux" (FLUX.1-dev + mezcla de latentes) o "kontext"
        # (FLUX.1-Kontext-dev, edición instruction-based nativa). Mismo DiT
        # (19+38, hidden 3072): el wrap PuLID oficial se transfiere tal cual.
        self.backend = "flux"
        if config is not None and getattr(config, "flux", None) is not None:
            self.backend = str(getattr(config.flux, "backend", "flux")).lower()
            if self.backend == "kontext":
                kontext_id = getattr(config.flux, "kontext_model_id", None) or "black-forest-labs/FLUX.1-Kontext-dev"
                self.model_id = kontext_id
        # Usar el model_id de la configuración (no hardcodear el repo).
        CLEAN_MODEL_ID = self.model_id or "black-forest-labs/FLUX.1-dev"
        logger.info(f"Inicializando backend={self.backend} {CLEAN_MODEL_ID} | device='{self.device}' | dtype={self.torch_dtype}")

        # Inpainting por mezcla de latentes sobre el pipeline NATIVO del repo.
        # (FluxInpaintPipeline exige el set completo de componentes y falla con
        # repos que no lo traen -> ValueError; la mezcla de latentes no.)
        # Con use_inpaint_pipeline=True, cada paso de denoising mezcla:
        #   latents = mask * noisy(original, t) + (1-mask) * latents_gen
        # de modo que fuera de la máscara se conserva la imagen original y
        # dentro se regenera condicionado en prompt + PuLID.
        if config is not None and getattr(config, "flux", None) is not None:
            self.want_inpaint = bool(getattr(config.flux, "use_inpaint_pipeline", True))
        else:
            self.want_inpaint = True
        self.is_inpaint = self.want_inpaint
        try:
            logger.info(f"[Flux2Engine] Cargando modelo base desde {CLEAN_MODEL_ID}...")
            if self.backend == "kontext":
                try:
                    from diffusers import FluxKontextPipeline
                except (ImportError, AttributeError) as e:
                    raise RuntimeError(
                        f"Este diffusers no trae FluxKontextPipeline ({e}): actualiza diffusers."
                    ) from e
                self.pipeline = FluxKontextPipeline.from_pretrained(
                    CLEAN_MODEL_ID,
                    torch_dtype=self.torch_dtype,
                    token=self.hf_token,
                )
                self.is_inpaint = False  # Kontext condiciona nativamente; sin mezcla ni composite.
                logger.info("[Flux2Engine] FluxKontextPipeline activo (edición instruction-based).")
            else:
                self.pipeline = DiffusionPipeline.from_pretrained(
                    CLEAN_MODEL_ID,
                    torch_dtype=self.torch_dtype,
                    token=self.hf_token,
                )
            if self.is_inpaint:
                logger.info("[Flux2Engine] Inpainting por mezcla de latentes activo (image+mask condicionan).")
            
            if enable_cpu_offload:
                logger.info("[Flux2Engine] Activando enable_model_cpu_offload()...")
                self.pipeline.enable_model_cpu_offload()

            logger.info(f"[Flux2Engine] {CLEAN_MODEL_ID} cargado exitosamente.")
            
        except Exception as exc:
            logger.error(f"[Flux2Engine] Error crítico: {exc}")
            raise RuntimeError("Fallo al inicializar FLUX.")

        self._setup_pulid_adapter()

    def unload(self):
        """Libera el pipeline DiT por completo (RAM+VRAM) tras la Etapa 3.

        Con refino LoRA viene un SEGUNDO DiT de 12B (+segundo T5-XXL); sin esta
        descarga convivirían ~70GB en pesos y el nodo muere por OOM (visto en
        task 53). Nunca tumba: los fallos se registran y se continúa.
        """
        try:
            pipe = getattr(self, "pipeline", None)
            if pipe is not None:
                try:
                    pipe.maybe_free_model_hooks()
                except Exception:
                    pass
                del self.pipeline
            import gc as _gc

            _gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
            logger.info("[Flux2Engine] pipeline DiT descargado (swap real a host).")
        except Exception as e:
            logger.warning(f"[Flux2Engine] unload con problemas ({e}); se continúa.")

    # Intervalos oficiales: double cada 2, single cada 4 (19 double + 38 single
    # -> 10 + 10 = 20 ramas pulid_ca.0..19).
    PULID_DOUBLE_INTERVAL = 2
    PULID_SINGLE_INTERVAL = 4

    def _wrap_block(self, block, ca_idx: int, kind: str):
        """Suma id_weight * pulid_ca[id](id, hidden) tras el bloque (diseño oficial).

        Tanto los bloques double como los single de este diffusers devuelven la
        tupla (encoder_hidden_states, hidden_states): el segundo elemento ya es
        el stream de imagen, sin splits manuales.
        """
        orig_fwd = block.forward
        inpainter = self

        def fwd(*args, **kwargs):
            out = orig_fwd(*args, **kwargs)
            if not isinstance(out, tuple) or len(out) != 2:
                raise RuntimeError(
                    f"PuLID wrap ({kind} ca={ca_idx}): salida inesperada "
                    f"{type(out)}; se esperaba tupla (enc, hidden)."
                )
            enc_out, hidden_out = out
            cur = inpainter._current_id
            w = inpainter._eff_id_weight
            if w is None:
                w = inpainter.id_weight
            if cur is not None and w > 0:
                hidden_out = hidden_out + w * inpainter.pulid_ca[ca_idx](cur, hidden_out)
            return enc_out, hidden_out

        block.forward = fwd

    def _setup_pulid_adapter(self):
        # Estado válido en TODOS los caminos (también con injection=none, que
        # retorna antes de definir nada más): generate_inpaint los asume.
        self._current_id = None
        self._eff_id_weight = None
        self.id_weight = 1.0
        if self.config is not None and getattr(self.config, "flux", None) is not None:
            self.id_weight = float(getattr(self.config.flux, "id_scale", 1.0))
        injection = "pulid"
        if self.config is not None and getattr(self.config, "identity", None) is not None:
            injection = str(getattr(self.config.identity, "injection", "pulid")).lower()
        self.injection = injection
        if injection == "none":
            logger.info("Inyección PuLID desactivada por config (identity.injection=none): Kontext solo.")
            self.has_pulid = False
            return
        try:
            logger.info("Inyectando adaptador oficial PuLID-FLUX (pulid_ca) en el DiT...")
            transformer = self.pipeline.transformer

            # 1. Descargar pesos oficiales (el repo NO tiene "pulid_flux.safetensors";
            #    los ficheros reales son pulid_flux_v0.9.x.safetensors).
            configured = None
            if self.config is not None and getattr(self.config, "flux", None) is not None:
                configured = getattr(self.config.flux, "pulid_ckpt_file", None)
            candidates = [c for c in [configured, "pulid_flux_v0.9.1.safetensors", "pulid_flux_v0.9.0.safetensors"] if c]
            pulid_state_dict, pulid_ckpt_path = None, None
            last_err = None
            for fname in dict.fromkeys(candidates):
                try:
                    logger.info(f"Descargando/Cargando pesos de {self.pulid_model_id}/{fname}...")
                    pulid_ckpt_path = hf_hub_download(repo_id=self.pulid_model_id, filename=fname, token=self.hf_token)
                    pulid_state_dict = load_file(pulid_ckpt_path)
                    break
                except Exception as e:
                    last_err = e
                    logger.warning(f"PuLID {fname} no disponible ({e}); probando siguiente candidato...")
            if pulid_state_dict is None:
                raise RuntimeError(f"Ningún checkpoint PuLID descargable en {self.pulid_model_id}: {last_err}")

            enc_sd = {k[len("pulid_encoder."):]: v for k, v in pulid_state_dict.items() if k.startswith("pulid_encoder.")}
            ca_sd = {k[len("pulid_ca."):]: v for k, v in pulid_state_dict.items() if k.startswith("pulid_ca.")}
            if not enc_sd or not ca_sd:
                raise RuntimeError("Checkpoint sin ramas pulid_encoder/pulid_ca oficiales.")

            exec_device = torch.device("cuda" if torch.cuda.is_available() else str(self.device))

            # 2. Ramas de inyección con pesos oficiales, carga ESTRICTA: cualquier
            # divergencia falla aquí, nunca como capas aleatorias en runtime.
            self.pulid_ca = nn.ModuleList([PerceiverAttentionCA() for _ in range(20)])
            self.pulid_ca.load_state_dict(ca_sd, strict=True)
            self.pulid_ca.to(exec_device, self.torch_dtype)
            self.pulid_ca.eval()
            self.pulid_device = exec_device

            # 3. Envolver bloques del transformer según intervalos oficiales.
            double_blocks = list(getattr(transformer, "transformer_blocks", []))
            single_blocks = list(getattr(transformer, "single_transformer_blocks", []))
            if not double_blocks:
                raise RuntimeError("Transformer sin transformer_blocks: nada que parchear.")
            # Compatibilidad de hidden size: las proyecciones PuLID son fijas 3072.
            hid = getattr(getattr(double_blocks[0].attn, "to_q", None), "in_features", None)
            logger.info(
                f"DiT: {len(double_blocks)} double + {len(single_blocks)} single, "
                f"hidden≈{hid}; PuLID-FLUX-v0.9.x espera hidden=3072."
            )
            if hid is not None and int(hid) != 3072:
                raise RuntimeError(
                    f"Hidden size del DiT ({hid}) != 3072 de PuLID-FLUX-v0.9.x: "
                    "proyecciones incompatibles sin reentrenar."
                )
            # Mapeo de las 20 ramas oficiales a la profundidad real:
            # FLUX.1 (19+38) -> intervalos oficiales 2/4 (10+10 exactos).
            # Otras profundidades -> doubles cada 2 + singles equiespaciados.
            # Cada peso se usa exactamente una vez: sin reutilizar ni omitir.
            dbl_idx = [i for i in range(len(double_blocks)) if i % self.PULID_DOUBLE_INTERVAL == 0]
            need = len(self.pulid_ca) - len(dbl_idx)
            if len(single_blocks) == 38 and need == 10:
                sgl_idx = [i for i in range(38) if i % self.PULID_SINGLE_INTERVAL == 0]
            elif need > 1 and len(single_blocks) >= need:
                n = len(single_blocks)
                sgl_idx = sorted({round(i * (n - 1) / (need - 1)) for i in range(need)})
            else:
                sgl_idx = []
            if len(dbl_idx) + len(sgl_idx) != len(self.pulid_ca):
                raise RuntimeError(
                    f"PuLID trae {len(self.pulid_ca)} ramas pero este transformer "
                    f"({len(double_blocks)} double + {len(single_blocks)} single) selecciona "
                    f"{len(dbl_idx) + len(sgl_idx)} (double={dbl_idx}, single={sgl_idx})."
                )
            logger.info(f"PuLID mapping: double={dbl_idx} single={sgl_idx}")
            ca = 0
            for i in dbl_idx:
                self._wrap_block(double_blocks[i], ca, kind="double")
                ca += 1
            for i in sgl_idx:
                self._wrap_block(single_blocks[i], ca, kind="single")
                ca += 1

            logger.info(
                f"PuLID oficial activo: {len(dbl_idx)} ramas double + {len(sgl_idx)} ramas single "
                f"({pulid_ckpt_path}), id_weight={self.id_weight}, device={exec_device}. "
                "dropout_p no aplica en la rama oficial (sin dropout entrenado)."
            )
            self.has_pulid = True

        except Exception as e:
            logger.error(f"PuLID no disponible: {e}. (generate_inpaint fallará en voz alta)")
            self.has_pulid = False

    @torch.no_grad()
    def generate_inpaint(self, image: Image.Image, mask: torch.Tensor, identity_embeddings: Dict[str, Any], steps: int = 35) -> Image.Image:
        logger.info(f"Iniciando inpainting FLUX a {steps} pasos...")
        
        try:
            width, height = image.size
            width = (width // 16) * 16
            height = (height // 16) * 16
            image = image.resize((width, height), Image.LANCZOS)
            
            if isinstance(mask, torch.Tensor):
                mask_tensor = mask.squeeze()
                if mask_tensor.dim() > 2: mask_tensor = mask_tensor[0]
                mask_pil = transforms.ToPILImage()(mask_tensor.cpu())
            else:
                mask_pil = mask
                
            mask_pil = mask_pil.resize((width, height), Image.NEAREST)
            
            id_embeds = identity_embeddings.get("embeddings")
            if id_embeds is None:
                raise ValueError("Embeddings de identidad no encontrados.")
            # Contrato oficial: (B, 32, 2048) del IDFormer. Cualquier otra forma
            # es un bug aguas arriba: fallar aquí, no generar basura.
            if id_embeds.dim() != 3 or id_embeds.shape[1] != 32 or id_embeds.shape[2] != 2048:
                raise ValueError(f"Embedding de identidad con forma {tuple(id_embeds.shape)}; se espera (B, 32, 2048).")

            if not self.has_pulid and getattr(self, "injection", "pulid") != "none":
                # Generar sin identidad = cara aleatoria silenciosa (lo visto en el
                # troubleshooting). Mejor fallar en voz alta que devolver basura.
                # (Con identity.injection=none el modo Kontext-solo es legítimo.)
                raise RuntimeError(
                    "PuLID no parcheado (has_pulid=False): generar ahora produciría "
                    "una cara aleatoria. Revisa el checkpoint PuLID antes de continuar."
                )

            # Los bloques envueltos leen la identidad de aquí; viaja con ellos a GPU
            # bajo cpu_offload (antes iba a CPU -> Device Mismatch).
            # Peso modulado por confianza (idea AdaFace: norma como calidad):
            # señal mala -> influencia baja, nunca inyección ciega a peso 1.0.
            conf = float(identity_embeddings.get("confidence", 1.0))
            gate = min(max(conf, 0.0), 1.0)
            self._eff_id_weight = float(self.id_weight) * gate
            logger.info(
                f"[PuLID] confianza={conf:.3f} id_scale={float(self.id_weight):.2f} "
                f"-> id_weight_efectivo={self._eff_id_weight:.3f}"
            )
            if getattr(self, "injection", "pulid") != "none" and self.has_pulid:
                self._current_id = id_embeds.to(device=self.pulid_device, dtype=self.torch_dtype)
            else:
                self._current_id = None
            try:
                _cid = self._current_id.float()
                logger.info(
                    f"[ID-Calib] tokens shape={tuple(self._current_id.shape)} "
                    f"mean={float(_cid.mean()):.4f} std={float(_cid.std()):.4f} "
                    f"norm_media={float(_cid.norm(p=2, dim=-1).mean()):.4f} "
                    f"id_weight={self.id_weight}"
                )
            except Exception as _fe:
                logger.warning(f"[ID-Calib] no se pudo auditar identidad: {_fe}")

            prompt = getattr(getattr(self.config, "flux", None), "prompt", None) if self.config is not None else None
            prompt = prompt or "photorealistic high quality restored face, smooth skin microtexture, sharp eyes, 4k resolution"
            guidance = getattr(getattr(self.config, "flux", None), "guidance_scale", 3.5) if self.config is not None else 3.5
            seed = getattr(getattr(self.config, "flux", None), "seed", 42) if self.config is not None else 42
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            kwargs = {
                "prompt": prompt,
                "num_inference_steps": steps,
                "guidance_scale": float(guidance),
                "height": height,
                "width": width,
                "generator": generator,
            }

            if self.backend == "kontext":
                # Edición nativa: ve la imagen completa + instrucción + identidad
                # PuLID (vía bloques envueltos). Sin máscara ni composite.
                instruction = getattr(getattr(self.config, "flux", None), "edit_instruction", None)
                instruction = instruction or prompt
                logger.info(f"[Kontext] instruction: {instruction[:120]}...")
                try:
                    generated_image = self.pipeline(
                        image=image,
                        prompt=instruction,
                        num_inference_steps=steps,
                        guidance_scale=float(guidance),
                        generator=generator,
                        height=height,
                        width=width,
                    ).images[0]
                finally:
                    self._current_id = None
                    self._eff_id_weight = None
                if generated_image.size != (width, height):
                    logger.warning(
                        f"[Kontext] salida {generated_image.size} != entrada {(width, height)}; "
                        "se reescala a la geometría original."
                    )
                    generated_image = generated_image.resize((width, height), Image.LANCZOS)
                return generated_image

            if self.is_inpaint:
                # RETIRADO: la mezcla manual de latentes colapsa el denoising
                # (arcoíris global verificado en jobs 6949812/6951781 aun con
                # id_weight=0). Vía productiva: backend=kontext. Este flag solo
                # admite false (= T2I + composite, solo diagnóstico).
                raise RuntimeError(
                    "Mezcla manual de latentes retirada por inestable. Usa "
                    "backend=kontext para producción, o use_inpaint_pipeline=false "
                    "solo para diagnóstico T2I."
                )
            logger.error(
                "Fallback T2I activo (use_inpaint_pipeline=false): la salida NO está "
                "condicionada en la imagen original; solo se compone el parche."
            )
            try:
                generated_image = self.pipeline(**kwargs).images[0]
            finally:
                self._current_id = None
                self._eff_id_weight = None

            mask_resized = mask_pil.resize((width, height), Image.BILINEAR).convert("L")
            output = Image.composite(generated_image, image, mask_resized)

            return output
            
        except Exception as e:
            logger.error(f"Error fatal en inpainting FLUX: {e}")
            raise