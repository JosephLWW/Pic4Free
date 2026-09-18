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

from pic4free.models.pulid_official import IDFormer, PerceiverAttentionCA

logger = logging.getLogger("Pic4Free.FluxInpainter")

# NOTE: removed the FluxPuLIDAttnProcessor casero (to_k_ip/to_v_ip): the
# the official checkpoint does NOT contain those keys and the design does not match with PuLID.
# the injection continues the forward official (ToTheBeginning/PuLID, Apache-2.0):
#   img = img + id_weight * pulid_ca[id](id_emb, img)
# after each even double block and each single block multiple of 4.


class FluxPuLIDInpainter:
    def __init__(self, model_id: str, pulid_model_id: str, device: str = "cpu", enable_cpu_offload: bool = True, hf_token: Optional[str] = None, config=None):
        self.model_id = model_id
        self.pulid_model_id = pulid_model_id
        # Bug FIX: is uses the parameter `device` recibido as only source of truth.
        # Assigning it directly to self.device avoids mismatches in subcomponents
        # (PuLID processors and identity embeddings) that depend on this attribute.
        self.device = torch.device(device)
        self.torch_dtype = torch.bfloat16
        # Config centralizada (FluxConfig): prompt, guidance, id_scale, dropout_p, steps.
        # is stored by reference so CLI changes (OmegaConf) are reflected.
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
        
        # Backend: "flux" (FLUX.1-dev + blending of latentes) or "kontext"
        # (FLUX.1-Kontext-dev, editing instruction-based native). Mismo DiT
        # (19+38, hidden 3072): the wrap PuLID official is transfiere tal cual.
        self.backend = "flux"
        if config is not None and getattr(config, "flux", None) is not None:
            self.backend = str(getattr(config.flux, "backend", "flux")).lower()
            if self.backend == "kontext":
                kontext_id = getattr(config.flux, "kontext_model_id", None) or "black-forest-labs/FLUX.1-Kontext-dev"
                self.model_id = kontext_id
        # Use the configured model_id of the configuration (do not hardcoof the repository).
        CLEAN_MODEL_ID = self.model_id or "black-forest-labs/FLUX.1-dev"
        logger.info(f"Inicializando backend={self.backend} {CLEAN_MODEL_ID} | device='{self.device}' | dtype={self.torch_dtype}")

        # Inpainting by latent blending over the repository's NATIVE pipeline.
        # (FluxInpaintPipeline exige the set full of componentes and fails with
        # repos that no lo traen -> ValueError; the blending of latentes no.)
        # with use_inpaint_pipeline=True, each pass of denoising blending:
        #   latents = mask * noisy(original, t) + (1-mask) * latents_gen
        # so that outsiof the mask the original image is preserved and
        # insiof it is regenerated, conditioned by the prompt + PuLID.
        if config is not None and getattr(config, "flux", None) is not None:
            self.want_inpaint = bool(getattr(config.flux, "use_inpaint_pipeline", True))
        else:
            self.want_inpaint = True
        self.is_inpaint = self.want_inpaint
        try:
            logger.info(f"[Flux2Engine] Loading base model from {CLEAN_MODEL_ID}...")
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
                self.is_inpaint = False  # Kontext condiciona nativamente; without blending ni composite.
                logger.info("[Flux2Engine] FluxKontextPipeline active (instruction-based editing).")
            else:
                self.pipeline = DiffusionPipeline.from_pretrained(
                    CLEAN_MODEL_ID,
                    torch_dtype=self.torch_dtype,
                    token=self.hf_token,
                )
            if self.is_inpaint:
                logger.info("[Flux2Engine] Inpainting via active latent blending (image and mask proviof conditioning).")
            
            if enable_cpu_offload:
                logger.info("[Flux2Engine] Activando enable_model_cpu_offload()...")
                self.pipeline.enable_model_cpu_offload()

            logger.info(f"[Flux2Engine] {CLEAN_MODEL_ID} loaded exitosamente.")
            
        except Exception as exc:
            logger.error(f"[Flux2Engine] Critical error: {exc}")
            raise RuntimeError("Failed to initialize FLUX.")

        self._setup_pulid_adapter()

    def unload(self):
        """Fully frees the DiT pipeline (RAM+VRAM) after Stage 3.

        with LoRA refinement, a SECOND 12B DiT (+second T5-XXL) is loaded; without it, this
        downloads would coexist as ~70GB of weights and the noof would die from OOM (seen in
        task 53). Never fails the pipeline: failures are logged and execution continues.
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
            logger.info("[Flux2Engine] pipeline DiT downloaded (actual swap to host).")
        except Exception as e:
            logger.warning(f"[Flux2Engine] unload with problemas ({e}); continuing.")

    # Official intervals: double each 2, single each 4 (19 double + 38 single
    # -> 10 + 10 = 20 branches pulid_ca.0..19).
    PULID_DOUBLE_INTERVAL = 2
    PULID_SINGLE_INTERVAL = 4

    def _wrap_block(self, block, ca_idx: int, kind: str):
        """Adds id_weight * pulid_ca[id](id, hidden) after the block (design official).

        Both double and single blocks in this diffusers implementation return the
        tupthe (encoder_hidden_states, hidden_states): the second element is already
        the image stream, without manual splitting.
        """
        orig_fwd = block.forward
        inpainter = self

        def fwd(*args, **kwargs):
            out = orig_fwd(*args, **kwargs)
            if not isinstance(out, tuple) or len(out) != 2:
                raise RuntimeError(
                    f"PuLID wrap ({kind} ca={ca_idx}): output unexpected "
                    f"{type(out)}; expected tuple of (enc, hidden)."
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
        # Estado valido in TODOS the caminos (also with injection=none, that
        # returns before defining anything else): generate_inpaint the asume.
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
            logger.info("PuLID injection disabled by config (identity.injection=none): Kontext only.")
            self.has_pulid = False
            return
        try:
            logger.info("Inyect adapter official PuLID-FLUX (pulid_ca) in the DiT...")
            transformer = self.pipeline.transformer

            # 1. Download official weights (the repo does NOT contain "pulid_flux.safetensors";
            #    the actual files are pulid_flux_v0.9.x.safetensors).
            configured = None
            if self.config is not None and getattr(self.config, "flux", None) is not None:
                configured = getattr(self.config.flux, "pulid_ckpt_file", None)
            candidates = [c for c in [configured, "pulid_flux_v0.9.1.safetensors", "pulid_flux_v0.9.0.safetensors"] if c]
            pulid_state_dict, pulid_ckpt_path = None, None
            last_err = None
            for fname in dict.fromkeys(candidates):
                try:
                    logger.info(f"Downloading/Loading weights of {self.pulid_model_id}/{fname}...")
                    pulid_ckpt_path = hf_hub_download(repo_id=self.pulid_model_id, filename=fname, token=self.hf_token)
                    pulid_state_dict = load_file(pulid_ckpt_path)
                    break
                except Exception as e:
                    last_err = e
                    logger.warning(f"PuLID {fname} unavailable ({e}); probando siguiente candidato...")
            if pulid_state_dict is None:
                raise RuntimeError(f"No downloadable PuLID checkpoint in {self.pulid_model_id}: {last_err}")

            enc_sd = {k[len("pulid_encoder."):]: v for k, v in pulid_state_dict.items() if k.startswith("pulid_encoder.")}
            ca_sd = {k[len("pulid_ca."):]: v for k, v in pulid_state_dict.items() if k.startswith("pulid_ca.")}
            if not enc_sd or not ca_sd:
                raise RuntimeError("Checkpoint lacks the official pulid_encoder/pulid_ca branches.")

            exec_device = torch.device("cuda" if torch.cuda.is_available() else str(self.device))

            # 2. Injection branches with official weights, STRICT loading: any
            # divergence fails here, never as random layers at runtime.
            self.pulid_ca = nn.ModuleList([PerceiverAttentionCA() for _ in range(20)])
            self.pulid_ca.load_state_dict(ca_sd, strict=True)
            self.pulid_ca.to(exec_device, self.torch_dtype)
            self.pulid_ca.eval()
            self.pulid_device = exec_device

            # 3. Wrap blocks of the transformer according to official intervals.
            double_blocks = list(getattr(transformer, "transformer_blocks", []))
            single_blocks = list(getattr(transformer, "single_transformer_blocks", []))
            if not double_blocks:
                raise RuntimeError("Transformer without transformer_blocks: nothing to parch.")
            # Compatibilidad of hidden size: the proyecciones PuLID son fijas 3072.
            hid = getattr(getattr(double_blocks[0].attn, "to_q", None), "in_features", None)
            logger.info(
                f"DiT: {len(double_blocks)} double + {len(single_blocks)} single, "
                f"hidden{hid}; PuLID-FLUX-v0.9.x expects hidden=3072."
            )
            if hid is not None and int(hid) != 3072:
                raise RuntimeError(
                    f"DiT hidden size ({hid}) != 3072 for PuLID-FLUX-v0.9.x: "
                    "incompatible projections without retraining."
                )
            # Mapping the 20 official branches to the actual depth:
            # FLUX.1 (19+38) -> official intervals 2/4 (10+10 exactos).
            # Otras profundidades -> doubles each 2 + singles equiespaciados.
            # Each weight is used exactly once: without reuse or skipping.
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
                    f"PuLID trae {len(self.pulid_ca)} branches pero este transformer "
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
                f"Official PuLID active: {len(dbl_idx)} double branches + {len(sgl_idx)} single branches "
                f"({pulid_ckpt_path}), id_weight={self.id_weight}, device={exec_device}. "
                "dropout_p does not apply in the official branch (without trained dropout)."
            )
            self.has_pulid = True

        except Exception as e:
            logger.error(f"PuLID unavailable: {e}. (generate_inpaint will fail loudly)")
            self.has_pulid = False

    @torch.no_grad()
    def generate_inpaint(self, image: Image.Image, mask: torch.Tensor, identity_embeddings: Dict[str, Any], steps: int = 35) -> Image.Image:
        logger.info(f"Starting FLUX inpainting with {steps} steps...")
        
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
                raise ValueError("Embeddings of identity no encontrados.")
            # Official contract: (B, 32, 2048) of the IDFormer. Any other shape
            # is a bug upstream: fail here rather than generating garbage.
            if id_embeds.dim() != 3 or id_embeds.shape[1] != 32 or id_embeds.shape[2] != 2048:
                raise ValueError(f"Identity embedding has shape {tuple(id_embeds.shape)}; expected (B, 32, 2048).")

            if not self.has_pulid and getattr(self, "injection", "pulid") != "none":
                # Generate without identity = silent random face (as seen in the
                # troubleshooting). Mejor fallar in voz alta that return garbage.
                # (with identity.injection=none Kontext-only moof is legitimate.)
                raise RuntimeError(
                    "PuLID not patched (has_pulid=False): generating now would produces "
                    "a random face. Check the PuLID checkpoint before continuing."
                )

            # wrapped blocks read the identity from here; it travels with them to the GPU
            # under cpu_offload (previously moved to CPU -> Device Mismatch).
            # Weight modulated by confidence (AdaFace idea: norm as quality):
            # poor signal -> low influence, never blindly inject with weight 1.0.
            conf = float(identity_embeddings.get("confidence", 1.0))
            gate = min(max(conf, 0.0), 1.0)
            self._eff_id_weight = float(self.id_weight) * gate
            logger.info(
                f"[PuLID] confidence={conf:.3f} id_scale={float(self.id_weight):.2f} "
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
                    f"norm_mean={float(_cid.norm(p=2, dim=-1).mean()):.4f} "
                    f"id_weight={self.id_weight}"
                )
            except Exception as _fe:
                logger.warning(f"[ID-Calib] could not audit identity: {_fe}")

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
                # Native editing: sees the full image + instruction + identity
                # PuLID (via blocks wrapped). without mask ni composite.
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
                        f"[Kontext] output {generated_image.size} != input {(width, height)}; "
                        "is resized to the original geometry."
                    )
                    generated_image = generated_image.resize((width, height), Image.LANCZOS)
                return generated_image

            if self.is_inpaint:
                # RETIRADO: the blending manual of latentes colapsa the denoising
                # (arcoiris global verificado in jobs 6949812/6951781 aun with
                # id_weight=0). Via productiva: backend=kontext. this flag only
                # allows false (= T2I + composite, diagnostics only).
                raise RuntimeError(
                    "Manual latent blending was removed because it is unstable. Use "
                    "backend=kontext for produccion, o use_inpaint_pipeline=false "
                    "only for diagnostico T2I."
                )
            logger.error(
                "Active T2I fallback (use_inpaint_pipeline=false): the output is NOT "
                "conditioned on the original image; only the patch is composited."
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
            logger.error(f"Error fatal in inpainting FLUX: {e}")
            raise