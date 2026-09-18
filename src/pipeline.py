import torch
import torch.nn.functional as F
import gc
import logging
from PIL import Image
import numpy as np
import os
import time
from pathlib import Path
from typing import List

from src.config import PipelineConfig
from src.modules.mask_generator import BENMaskGenerator
from src.modules.identity_extractor import PulidIdentityExtractor
from src.modules.flux_inpainter import FluxPuLIDInpainter
from src.modules.super_resolution import SUPIRRestoreFormer

logger = logging.getLogger("Pic4Free.Pipeline")

class Pic4FreeRestorationPipeline:
    """
    Main orchestrator for the four SOTA restoration stages.
    Implements "Sequential Stage Swapping" to avoid OOM on 80GB+ nodes.
    """
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        self.host_device = "cpu"
        self.active_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info("Loading all static models into host RAM (CPU)...")

        # Token HF centralizado: env > config.flux.hf_token_file > hf_token.txt.
        # the .sh exports HF_TOKEN and passes flux.hf_token_file; the modules receive
        # as a parameter instead of reading relative paths itself.
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            token_file = getattr(config.flux, "hf_token_file", "hf_token.txt") or "hf_token.txt"
            if os.path.exists(token_file):
                with open(token_file, "r") as f:
                    hf_token = f.read().strip() or None
        if hf_token:
            logger.info("HF token resuelto (longitud %d).", len(hf_token))
        else:
            logger.warning(f"No HF token: the download of {config.flux.model_id} (repo gated) may fail with 401.")

        # Stage 1 (toggle masking.enable_ben_debug: BEN currently only produces the
        # PNG of debug; false = neither downloads nor runs the model).
        if bool(getattr(config.masking, "enable_ben_debug", True)):
            self.mask_generator = BENMaskGenerator(
                model_id=config.masking.model_id,
                device=self.host_device
            )
        else:
            self.mask_generator = None
            logger.info("BEN disabled by config (enable_ben_debug=false).")

        # Stage 2
        self.identity_extractor = PulidIdentityExtractor(
            config=config,
            device=self.host_device
        )

        # Stage 3
        self.inpainter = FluxPuLIDInpainter(
            model_id=config.flux.model_id,
            pulid_model_id=config.flux.pulid_model_id,
            device=self.host_device,
            config=config,
            hf_token=hf_token,
        )

        # Stage 4
        self.super_res = SUPIRRestoreFormer(
            model_id=config.superres.model_id,
            scale_factor=config.superres.scale_factor,
            device=self.host_device,
            enable_upscaling=config.superres.enable_upscaling,
            hf_token=hf_token,
            config=config,
        )

    def _flush_memory(self):
        """Strict, explicit GC and VRAM handling"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    SUPPORTED_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def _resolve_input_image(self) -> tuple:
        """
        Selects the input image pair corresponding to task_id.

        Naming convention in data/input/:
          - watermarked (N).jpg    main image (high resolution, with watermark)
          - thumb-400 (N).jpg      reference thumbnail (clean, low-resolution image)

        Returns:
            Tuple (input_path, thumb_path) with the absolute paths.

        Raises:
            FileNotFoundError: If any of the files do not exist.
        """
        task_idx = int(self.config.task_id)
        input_dir = Path(self.config.paths.input_dir)

        watermarked_path = input_dir / f"watermarked ({task_idx}).jpg"
        thumb_path       = input_dir / f"thumb-400 ({task_idx}).jpg"

        if not watermarked_path.is_file():
            raise FileNotFoundError(
                f"[STAGE 0] Main image not found: '{watermarked_path}'. "
                f"Ensure that 'watermarked ({task_idx}).jpg' in '{input_dir}'."
            )
        if not thumb_path.is_file():
            raise FileNotFoundError(
                f"[STAGE 0] Reference thumbnail not found: '{thumb_path}'. "
                f"Ensure that 'thumb-400 ({task_idx}).jpg' in '{input_dir}'."
            )

        logger.info(
            f"[STAGE 0] Image pair for task_id={task_idx}: "
            f"input='{watermarked_path.name}' | ref='{thumb_path.name}'"
        )
        return str(watermarked_path), str(thumb_path)

    def _resolve_identity_paths(self, identity_dir: str) -> List[str]:
        """
        Collect valid image paths from the identity directory
        configured. Supports .jpg, .jpeg, .png, .webp.

        Args:
            identity_dir: Path to the directory containing reference photos
                          of the person whose identity should be preserved.

        Returns:
            List of sorted absolute file paths.

        Raises:
            FileNotFoundError: If the directory does not exist.
            ValueError: if the directory exists but contains no images.
        """
        dir_path = Path(identity_dir)
        if not dir_path.is_dir():
            raise FileNotFoundError(
                f"[STAGE 2] Identity directory not found: '{identity_dir}'. "
                "Check that 'paths.identity_a_dir' points to the correct directory."
            )

        image_paths = sorted(
            str(p) for p in dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_IMG_EXTENSIONS
        )

        if not image_paths:
            raise ValueError(
                f"[STAGE 2] The identity directory '{identity_dir}' exists but does not "
                f"contains images with supported extensions: {self.SUPPORTED_IMG_EXTENSIONS}. "
                "Add at least one reference photo of the person."
            )

        logger.info(
            f"[STAGE 2] {len(image_paths)} reference image(s) found in "
            f"'{identity_dir}': {[Path(p).name for p in image_paths]}"
        )
        return image_paths

    def _watermark_mask(self, input_image: Image.Image, thumb_image: Image.Image) -> torch.Tensor:
        """Watermark mask from the difference against the clean thumbnail.

        The thumb is the same photo without the overlay: after resizing it to the size of the
        input, use the POSITIVE luminance difference (the overlay
        whitish overlay only adds light). This rejects symmetric noise from
        compression/resizing, because the absolute difference captured it.
        Solo torch/numpy, without cv2.

        Returns:
            Float tensor (1, 1, H, W) on CPU, 1.0 = regenerate.
        """
        import torch.nn.functional as Fnn

        W, H = input_image.size
        thr = float(getattr(self.config.masking, "watermark_diff_threshold", 18.0))
        dil = max(0, int(getattr(self.config.masking, "watermark_dilate_px", 7)))

        ref = thumb_image.convert("RGB").resize((W, H), Image.LANCZOS)
        a = torch.from_numpy(np.array(input_image.convert("RGB"), dtype=np.float32))
        b = torch.from_numpy(np.array(ref, dtype=np.float32))
        # Luminance Rec.601; overlay claro => dpos > 0 only in the mark.
        lum_a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        lum_b = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
        dpos = lum_a - lum_b  # (H, W)

        m = (dpos > thr).to(torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        if dil > 0:
            k = 2 * dil + 1
            m = Fnn.max_pool2d(m, kernel_size=k, stride=1, padding=dil)
            m = (m > 0.5).to(torch.float32)
        return m.clamp(0, 1)

    def run(self) -> str:
        """
        Runs the pipeline, orchestrating sequential swapping on the GPU.
        """
        logger.info("=== STARTING PIPELINE SOTA (SEQUENTIAL SWAP) ===")
        start_time = time.time()
        
        try:
            # ---------------------------------------------------------
            # Stage 0: Selection of input images
            # ---------------------------------------------------------
            input_path, thumb_path = self._resolve_input_image()
            input_image = Image.open(input_path).convert("RGB")
            thumb_image = Image.open(thumb_path).convert("RGB")
            logger.info(
                f"[STAGE 0] Image loaded: {Path(input_path).name} "
                f"({input_image.width}x{input_image.height}px) | "
                f"Reference: {Path(thumb_path).name} "
                f"({thumb_image.width}x{thumb_image.height}px)"
            )

            # ---------------------------------------------------------
            # Stage 1: Masking
            # ---------------------------------------------------------
            logger.info("[STAGE 1] Mask Segmentation and Isolation")
            t0 = time.time()
            
            if self.mask_generator is not None:
                self.mask_generator.model.to(self.active_device)
                mask = self.mask_generator.generate_mask(input_image)
                try:
                    from torchvision import transforms as _T
                    _dbg_dir = os.path.join(self.config.paths.output_dir, "restored")
                    os.makedirs(_dbg_dir, exist_ok=True)
                    _m = mask.detach().squeeze().cpu()
                    if _m.dim() > 2:
                        _m = _m[0]
                    _T.ToPILImage()(_m.clamp(0, 1)).save(os.path.join(_dbg_dir, f"task_{self.config.task_id}_debug_mask.png"))
                    logger.info(f"[DEBUG] mask mean={float(_m.mean()):.3f} min={float(_m.min()):.3f} max={float(_m.max()):.3f}")
                except Exception as _e:
                    logger.warning(f"[DEBUG] could not save the mask: {_e}")
                self.mask_generator.model.to(self.host_device)
            else:
                mask = None
                logger.info("[STAGE 1] BEN omitido (enable_ben_debug=false).")

            # Inpainting mask = watermark area (clean thumb vs
            # watermarked). the BEN mask (full silhouette) is NOT used for
            # inpainting: regenerating the entire silhouette destroys the person.
            wm_mask = self._watermark_mask(input_image, thumb_image)
            wm_frac = float(wm_mask.mean())
            try:
                from torchvision import transforms as _T
                _dbg_dir = os.path.join(self.config.paths.output_dir, "restored")
                os.makedirs(_dbg_dir, exist_ok=True)
                _wm = wm_mask.detach().squeeze().cpu()
                _T.ToPILImage()(_wm.clamp(0, 1)).save(
                    os.path.join(_dbg_dir, f"task_{self.config.task_id}_debug_watermark.png")
                )
            except Exception as _e:
                logger.warning(f"[DEBUG] could not save the mask watermark: {_e}")
            logger.info(f"[STAGE 1] mask watermark: fraction={wm_frac:.4f} (BEN only debug)")

            logger.info(f"Stage 1 completada in {time.time() - t0:.2f}s")
            
            # ---------------------------------------------------------
            # Stage 2: Facial Biometrics  Identities A and B
            # ---------------------------------------------------------
            logger.info("[STAGE 2] Facial Biometrics and Identity Mapping (A + B)")
            t0 = time.time()

            # the thumbnail clean (thumb) actua as fallback of identity if un
            # directory A/B missing, this empty or su extraction fails.
            def _paths_or_thumb(identity_dir: str, label: str) -> List[str]:
                try:
                    return self._resolve_identity_paths(identity_dir)
                except (FileNotFoundError, ValueError, OSError) as e:
                    logger.warning(
                        f"[STAGE 2] {label}: {e} -> fallback to clean thumbnail '{Path(thumb_path).name}' "
                        f"({thumb_image.width}x{thumb_image.height}px)."
                    )
                    return [thumb_path]

            identity_paths_a = _paths_or_thumb(self.config.paths.identity_a_dir, "Identidad A")
            identity_paths_b = _paths_or_thumb(self.config.paths.identity_b_dir, "Identidad B")

            self.identity_extractor.to(self.active_device)

            def _extract_or_thumb(paths: List[str], label: str) -> dict:
                try:
                    return self.identity_extractor.extract_identity(paths)
                except Exception as e:
                    if paths != [thumb_path]:
                        logger.warning(
                            f"[STAGE 2] {label}: extraction failed ({e}) -> retrying with clean thumbnail."
                        )
                        return self.identity_extractor.extract_identity([thumb_path])
                    raise

            id_a = _extract_or_thumb(identity_paths_a, "Identidad A")
            id_b = _extract_or_thumb(identity_paths_b, "Identidad B")

            self.identity_extractor.to(self.host_device)

            # References SEPARADAS (no averaged: A+B mixed no
            # represent two people).
            ref_arcs = {}
            if "arcface" in id_a:
                ref_arcs["A"] = id_a["arcface"]
            if "arcface" in id_b:
                ref_arcs["B"] = id_b["arcface"]
            # Identity selection for injection (config identity.selection):
            # "average" = average A+B (behavior historical),
            # "auto" = the thumb votes and uses only the winner,
            # "A"/"B" = manual.
            sel = str(getattr(self.config.identity, "selection", "average")).lower()
            if sel in ("a", "b") and sel.upper() in ref_arcs:
                chosen = sel.upper()
                combined_embeddings = id_a["embeddings"] if chosen == "A" else id_b["embeddings"]
                logger.info(f"[STAGE 2] manual selection: only identity {chosen}.")
            elif sel == "auto" and ref_arcs:
                vote = self.identity_extractor.select_identity(thumb_image, ref_arcs)
                chosen = vote.get("label")
                if chosen == "A":
                    combined_embeddings = id_a["embeddings"]
                elif chosen == "B":
                    combined_embeddings = id_b["embeddings"]
                else:
                    logger.warning(
                        f"[STAGE 2] vote no concluyente ({vote.get('note')}): average A+B."
                    )
                    combined_embeddings = (id_a["embeddings"] + id_b["embeddings"]) / 2.0
                    chosen = "A+B(fallback)"
                logger.info(f"[STAGE 2] automatic selection: identity {chosen} {vote.get('cosines', {})}.")
            else:
                if tuple(id_a["embeddings"].shape) != tuple(id_b["embeddings"].shape):
                    raise ValueError(
                        f"Embeddings A {tuple(id_a['embeddings'].shape)} and B "
                        f"{tuple(id_b['embeddings'].shape)} incompatible for fusion."
                    )
                combined_embeddings = (id_a["embeddings"] + id_b["embeddings"]) / 2.0
                chosen = "A+B"
            identities = {
                "embeddings": combined_embeddings,
                "confidence": (id_a["confidence"] + id_b["confidence"]) / 2.0,
                "norm":       (id_a["norm"]       + id_b["norm"])       / 2.0,
            }

            logger.info(
                f"Stage 2 completed in {time.time() - t0:.2f}s | "
                f"Confianza A={id_a['confidence']:.3f} B={id_b['confidence']:.3f} | "
                f"Norma A={id_a['norm']:.2f} B={id_b['norm']:.2f}"
            )
            
            # ---------------------------------------------------------
            # Stage 3: Inpainting (FLUX)
            # ---------------------------------------------------------
            logger.info("[STAGE 3] Maximum-Quality Latent Inpainting")
            t0 = time.time()

            # Deep cleanup ONLY before FLUX (the VRAM-intensive stage).
            # torch.cuda.empty_cache() bloquea/sincroniza the GPU: no use after each stage.
            self._flush_memory()

            # diffusers uses .enable_model_cpu_offload() for mover inteligentemente
            # only the necessary blocks (VAE, Transformer) to VRAM.

            min_frac = float(getattr(self.config.masking, "watermark_min_fraction", 0.0005))
            if wm_frac < min_frac:
                logger.warning(
                    f"[STAGE 3] No detectable watermark (fraction={wm_frac:.5f} < {min_frac}): "
                    "FLUX is skipped and the original image is preserved."
                )
                _w, _h = input_image.size
                restored_image = input_image.resize((((_w // 16) * 16), ((_h // 16) * 16)), Image.LANCZOS)
            else:
                restored_image = self.inpainter.generate_inpaint(
                    image=input_image,
                    mask=wm_mask,
                    identity_embeddings=identities,
                    steps=self.config.flux.num_inference_steps
                )
            try:
                import numpy as _np
                _dbg_dir = os.path.join(self.config.paths.output_dir, "restored")
                os.makedirs(_dbg_dir, exist_ok=True)
                restored_image.save(os.path.join(_dbg_dir, f"task_{self.config.task_id}_restored.png"))
                _ra = _np.array(restored_image.convert("RGB"), dtype=_np.float32)
                logger.info(f"[DEBUG] flux out size={restored_image.size} mean={float(_ra.mean()):.1f}")
            except Exception as _e:
                logger.warning(f"[DEBUG] could not save the output of FLUX: {_e}")

            logger.info(f"Stage 3 completada in {time.time() - t0:.2f}s")

            # Swap real: with refinement LoRA viene un segundo DiT 12B (+2o T5);
            # without download this, both coexist in RAM and the noof dies (OOM).
            if str(getattr(self.config.superres, "face_refine_backend", "lora")).lower() == "lora":
                try:
                    self.inpainter.unload()
                except Exception as _e:
                    logger.warning(f"Unload Stage 3 omitido ({_e}).")

            # ---------------------------------------------------------
            # Stage 4: Super-Resolution
            # ---------------------------------------------------------
            if self.config.superres.enable_face_refinement:
                logger.info("[STAGE 4] Post-Processing and Super-Resolution")
                t0 = time.time()

                # Asignacion faceref SOBRE the restored (pre-upscale): the
                # Face refinement needs a LoRA to give each face its own LoRA.
                pre_faces = {"faces": [], "assignment": {}, "min_assigned": None, "note": "not evaluated"}
                try:
                    if ref_arcs:
                        pre_faces = self.identity_extractor.verify_faces(restored_image, ref_arcs, top_k=2)
                        logger.info(f"[STAGE 4] pre-assignment: {pre_faces.get('assignment', {})}")
                except Exception as _e:
                    logger.warning(f"[STAGE 4] pre-assignment skipped: {_e}")

                self.super_res.face_restorer.to(self.active_device)
                self.super_res.supir_model.to(self.active_device)

                final_image = self.super_res.upscale(restored_image, faces_detail=pre_faces)

                self.super_res.face_restorer.to(self.host_device)
                self.super_res.supir_model.to(self.host_device)


                logger.info(f"Stage 4 completada in {time.time() - t0:.2f}s")
            else:
                final_image = restored_image
                pre_faces = {"faces": [], "assignment": {}, "min_assigned": None, "note": "stage skipped"}
            
            # Objective identity metric (discriminative, AdaFace/ArcFace role):
            # matriz each-face-final  each-reference (A, B separately).
            # Only in shape (warning below threshold); never fails the pipeline.
            identity_metric = {"faces": [], "assignment": {}, "min_assigned": None, "note": "not evaluated"}
            try:
                if ref_arcs:
                    identity_metric = self.identity_extractor.verify_faces(final_image, ref_arcs, top_k=2)
                    thr = float(getattr(self.config.identity, "match_threshold", 0.4))
                    for _f in identity_metric.get("faces", []):
                        logger.info(f"[METRICA] face {_f['idx']} cosines={_f.get('cosines', {})}")
                    for _fi, _a in identity_metric.get("assignment", {}).items():
                        _msg = (
                            f"[METRICA] face {_fi} -> ref {_a['ref']} "
                            f"cos={_a['cosine']:.3f} threshold={thr}"
                        )
                        if _a["cosine"] < thr:
                            logger.warning(_msg + " -> BAJO UMBRAL, revisar identity.")
                        else:
                            logger.info(_msg + " -> OK.")
                    if not identity_metric.get("assignment"):
                        logger.warning("[METRICA] no facereference assignment: cannot evaluate.")
            except Exception as _e:
                logger.warning(f"[METRICA] identity verification skipped: {_e}")
            try:
                import json as _json
                _mdir = os.path.join(self.config.paths.output_dir, "metrics")
                os.makedirs(_mdir, exist_ok=True)
                with open(os.path.join(_mdir, f"task_{self.config.task_id}_identity.json"), "w") as _fh:
                    _json.dump(
                        {
                            "task_id": str(self.config.task_id),
                            "backend": str(getattr(self.config.flux, "backend", "flux")),
                            "confidence_a": float(id_a["confidence"]),
                            "confidence_b": float(id_b["confidence"]),
                            "selection": str(getattr(self.config.identity, "selection", "average")),
                            "injected": chosen,
                            "refine_backend": str(getattr(self.config.superres, "face_refine_backend", "lora")),
                            "pre_assignment": pre_faces.get("assignment", {}),
                            "identity_metric": identity_metric,
                        },
                        _fh,
                        indent=2,
                    )
            except Exception as _e:
                logger.warning(f"[METRICA] could not save the JSON: {_e}")

            # Save output
            os.makedirs(os.path.join(self.config.paths.output_dir, "restored"), exist_ok=True)
            output_path = os.path.join(self.config.paths.output_dir, "restored", f"task_{self.config.task_id}_final.png")
            final_image.save(output_path)
            
            total_time = time.time() - start_time
            logger.info(f"=== PIPELINE COMPLETADO in {total_time:.2f}s ===")
            return output_path
            
        except torch.cuda.OutOfMemoryError:
            logger.error("FATAL: OutOfMemoryError even with Sequential Swapping.")
            self._flush_memory()
            raise
        except Exception as e:
            logger.error(f"Error in the pipeline: {e}")
            raise