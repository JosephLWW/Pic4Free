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
    Orquestador principal de las 4 etapas de restauración SOTA.
    Implementa "Sequential Stage Swapping" para evasión de OOM en nodos de 80GB+.
    """
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        self.host_device = "cpu"
        self.active_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info("Cargando todos los modelos estáticos a la RAM del host (CPU)...")

        # Token HF centralizado: env > config.flux.hf_token_file > hf_token.txt.
        # El .sh exporta HF_TOKEN y pasa flux.hf_token_file; los módulos lo reciben
        # como parámetro en vez de leer rutas relativas por su cuenta.
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            token_file = getattr(config.flux, "hf_token_file", "hf_token.txt") or "hf_token.txt"
            if os.path.exists(token_file):
                with open(token_file, "r") as f:
                    hf_token = f.read().strip() or None
        if hf_token:
            logger.info("HF token resuelto (longitud %d).", len(hf_token))
        else:
            logger.warning(f"Sin HF token: la descarga de {config.flux.model_id} (repo gated) puede fallar con 401.")

        # Etapa 1 (toggle masking.enable_ben_debug: BEN hoy solo produce el
        # PNG de debug; false = ni se descarga el modelo ni se ejecuta).
        if bool(getattr(config.masking, "enable_ben_debug", True)):
            self.mask_generator = BENMaskGenerator(
                model_id=config.masking.model_id,
                device=self.host_device
            )
        else:
            self.mask_generator = None
            logger.info("BEN desactivado por config (enable_ben_debug=false).")

        # Etapa 2
        self.identity_extractor = PulidIdentityExtractor(
            config=config,
            device=self.host_device
        )

        # Etapa 3
        self.inpainter = FluxPuLIDInpainter(
            model_id=config.flux.model_id,
            pulid_model_id=config.flux.pulid_model_id,
            device=self.host_device,
            config=config,
            hf_token=hf_token,
        )

        # Etapa 4
        self.super_res = SUPIRRestoreFormer(
            model_id=config.superres.model_id,
            scale_factor=config.superres.scale_factor,
            device=self.host_device,
            enable_upscaling=config.superres.enable_upscaling,
            hf_token=hf_token,
            config=config,
        )

    def _flush_memory(self):
        """Manejo severo y explícito de GC y VRAM"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    SUPPORTED_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def _resolve_input_image(self) -> tuple:
        """
        Selecciona el par de imágenes de input correspondiente al task_id.

        Convención de nombres en data/input/:
          - watermarked (N).jpg  →  imagen principal (alta resolución, con marca de agua)
          - thumb-400 (N).jpg    →  miniatura de referencia (imagen limpia, baja resolución)

        Returns:
            Tupla (input_path, thumb_path) con las rutas absolutas.

        Raises:
            FileNotFoundError: Si alguno de los ficheros no existe.
        """
        task_idx = int(self.config.task_id)
        input_dir = Path(self.config.paths.input_dir)

        watermarked_path = input_dir / f"watermarked ({task_idx}).jpg"
        thumb_path       = input_dir / f"thumb-400 ({task_idx}).jpg"

        if not watermarked_path.is_file():
            raise FileNotFoundError(
                f"[ETAPA 0] Imagen principal no encontrada: '{watermarked_path}'. "
                f"Asegúrate de que existe 'watermarked ({task_idx}).jpg' en '{input_dir}'."
            )
        if not thumb_path.is_file():
            raise FileNotFoundError(
                f"[ETAPA 0] Miniatura de referencia no encontrada: '{thumb_path}'. "
                f"Asegúrate de que existe 'thumb-400 ({task_idx}).jpg' en '{input_dir}'."
            )

        logger.info(
            f"[ETAPA 0] Par de imágenes para task_id={task_idx}: "
            f"input='{watermarked_path.name}' | ref='{thumb_path.name}'"
        )
        return str(watermarked_path), str(thumb_path)

    def _resolve_identity_paths(self, identity_dir: str) -> List[str]:
        """
        Recolecta las rutas de imágenes válidas desde el directorio de identidad
        configurado. Soporta .jpg, .jpeg, .png, .webp.

        Args:
            identity_dir: Ruta al directorio que contiene las fotos de referencia
                          de la persona cuya identidad se quiere preservar.

        Returns:
            Lista de rutas de archivo absolutas ordenadas.

        Raises:
            FileNotFoundError: Si el directorio no existe.
            ValueError: Si el directorio existe pero no contiene ninguna imagen.
        """
        dir_path = Path(identity_dir)
        if not dir_path.is_dir():
            raise FileNotFoundError(
                f"[ETAPA 2] Directorio de identidad no encontrado: '{identity_dir}'. "
                "Comprueba que 'paths.identity_a_dir' apunta a la carpeta correcta."
            )

        image_paths = sorted(
            str(p) for p in dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_IMG_EXTENSIONS
        )

        if not image_paths:
            raise ValueError(
                f"[ETAPA 2] El directorio de identidad '{identity_dir}' existe pero no "
                f"contiene imágenes con extensiones soportadas: {self.SUPPORTED_IMG_EXTENSIONS}. "
                "Añade al menos una foto de referencia de la persona."
            )

        logger.info(
            f"[ETAPA 2] {len(image_paths)} imagen(es) de referencia encontradas en "
            f"'{identity_dir}': {[Path(p).name for p in image_paths]}"
        )
        return image_paths

    def _watermark_mask(self, input_image: Image.Image, thumb_image: Image.Image) -> torch.Tensor:
        """Máscara de la marca de agua por diferencia contra la miniatura limpia.

        La thumb es la misma foto sin overlay: tras reescalarla al tamaño de la
        entrada, se usa la diferencia POSITIVA de luminancia (el overlay
        blanquecino solo suma luz). Esto rechaza el ruido simétrico de
        compresión/reescalado, que la diferencia absoluta sí capturaba.
        Solo torch/numpy, sin cv2.

        Returns:
            Tensor float (1, 1, H, W) en CPU, 1.0 = regenerar.
        """
        import torch.nn.functional as Fnn

        W, H = input_image.size
        thr = float(getattr(self.config.masking, "watermark_diff_threshold", 18.0))
        dil = max(0, int(getattr(self.config.masking, "watermark_dilate_px", 7)))

        ref = thumb_image.convert("RGB").resize((W, H), Image.LANCZOS)
        a = torch.from_numpy(np.array(input_image.convert("RGB"), dtype=np.float32))
        b = torch.from_numpy(np.array(ref, dtype=np.float32))
        # Luminancia Rec.601; overlay claro => dpos > 0 solo en la marca.
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
        Ejecuta el pipeline orquestando el Swapping Secuencial a la GPU.
        """
        logger.info("=== INICIANDO PIPELINE SOTA (SEQUENTIAL SWAP) ===")
        start_time = time.time()
        
        try:
            # ---------------------------------------------------------
            # Etapa 0: Selección de imágenes de entrada
            # ---------------------------------------------------------
            input_path, thumb_path = self._resolve_input_image()
            input_image = Image.open(input_path).convert("RGB")
            thumb_image = Image.open(thumb_path).convert("RGB")
            logger.info(
                f"[ETAPA 0] Imagen cargada: {Path(input_path).name} "
                f"({input_image.width}x{input_image.height}px) | "
                f"Referencia: {Path(thumb_path).name} "
                f"({thumb_image.width}x{thumb_image.height}px)"
            )

            # ---------------------------------------------------------
            # Etapa 1: Masking
            # ---------------------------------------------------------
            logger.info("[ETAPA 1] Segmentación y Aislamiento de Máscaras")
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
                    logger.warning(f"[DEBUG] no se pudo guardar la máscara: {_e}")
                self.mask_generator.model.to(self.host_device)
            else:
                mask = None
                logger.info("[ETAPA 1] BEN omitido (enable_ben_debug=false).")

            # Máscara de inpainting = zona de la marca de agua (thumb limpia vs
            # watermarked). La máscara BEN (silueta completa) NO se usa para
            # inpaintar: regenerar la silueta entera destruye a la persona.
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
                logger.warning(f"[DEBUG] no se pudo guardar la máscara watermark: {_e}")
            logger.info(f"[ETAPA 1] máscara watermark: fracción={wm_frac:.4f} (BEN solo debug)")

            logger.info(f"Etapa 1 completada en {time.time() - t0:.2f}s")
            
            # ---------------------------------------------------------
            # Etapa 2: Biometría — Identidades A y B
            # ---------------------------------------------------------
            logger.info("[ETAPA 2] Biometría Facial y Mapeo de Identidad (A + B)")
            t0 = time.time()

            # La miniatura limpia (thumb) actúa como fallback de identidad si un
            # directorio A/B falta, está vacío o su extracción falla.
            def _paths_or_thumb(identity_dir: str, label: str) -> List[str]:
                try:
                    return self._resolve_identity_paths(identity_dir)
                except (FileNotFoundError, ValueError, OSError) as e:
                    logger.warning(
                        f"[ETAPA 2] {label}: {e} -> fallback a miniatura limpia '{Path(thumb_path).name}' "
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
                            f"[ETAPA 2] {label}: extracción falló ({e}) -> reintento con miniatura limpia."
                        )
                        return self.identity_extractor.extract_identity([thumb_path])
                    raise

            id_a = _extract_or_thumb(identity_paths_a, "Identidad A")
            id_b = _extract_or_thumb(identity_paths_b, "Identidad B")

            self.identity_extractor.to(self.host_device)

            # Referencias SEPARADAS (no promediadas: A+B mezclados no
            # representan a ninguna de las dos personas).
            ref_arcs = {}
            if "arcface" in id_a:
                ref_arcs["A"] = id_a["arcface"]
            if "arcface" in id_b:
                ref_arcs["B"] = id_b["arcface"]
            # Selección de identidad para inyección (config identity.selection):
            # "average" = promedio A+B (comportamiento histórico),
            # "auto" = vota la thumb y usa solo la ganadora,
            # "A"/"B" = manual.
            sel = str(getattr(self.config.identity, "selection", "average")).lower()
            if sel in ("a", "b") and sel.upper() in ref_arcs:
                chosen = sel.upper()
                combined_embeddings = id_a["embeddings"] if chosen == "A" else id_b["embeddings"]
                logger.info(f"[ETAPA 2] selección manual: solo identidad {chosen}.")
            elif sel == "auto" and ref_arcs:
                vote = self.identity_extractor.select_identity(thumb_image, ref_arcs)
                chosen = vote.get("label")
                if chosen == "A":
                    combined_embeddings = id_a["embeddings"]
                elif chosen == "B":
                    combined_embeddings = id_b["embeddings"]
                else:
                    logger.warning(
                        f"[ETAPA 2] voto no concluyente ({vote.get('note')}): promedio A+B."
                    )
                    combined_embeddings = (id_a["embeddings"] + id_b["embeddings"]) / 2.0
                    chosen = "A+B(fallback)"
                logger.info(f"[ETAPA 2] selección auto: identidad {chosen} {vote.get('cosines', {})}.")
            else:
                if tuple(id_a["embeddings"].shape) != tuple(id_b["embeddings"].shape):
                    raise ValueError(
                        f"Embeddings A {tuple(id_a['embeddings'].shape)} y B "
                        f"{tuple(id_b['embeddings'].shape)} incompatibles para fusionar."
                    )
                combined_embeddings = (id_a["embeddings"] + id_b["embeddings"]) / 2.0
                chosen = "A+B"
            identities = {
                "embeddings": combined_embeddings,
                "confidence": (id_a["confidence"] + id_b["confidence"]) / 2.0,
                "norm":       (id_a["norm"]       + id_b["norm"])       / 2.0,
            }

            logger.info(
                f"Etapa 2 completada en {time.time() - t0:.2f}s | "
                f"Confianza A={id_a['confidence']:.3f} B={id_b['confidence']:.3f} | "
                f"Norma A={id_a['norm']:.2f} B={id_b['norm']:.2f}"
            )
            
            # ---------------------------------------------------------
            # Etapa 3: Inpainting (FLUX)
            # ---------------------------------------------------------
            logger.info("[ETAPA 3] Inpainting Latente de Máxima Calidad")
            t0 = time.time()

            # Limpieza profunda SOLO antes de FLUX (etapa intensiva en VRAM).
            # torch.cuda.empty_cache() bloquea/sincroniza la GPU: no usar tras cada etapa.
            self._flush_memory()

            # diffusers usa .enable_model_cpu_offload() para mover inteligentemente
            # solo los bloques necesarios (VAE, Transformer) a la VRAM.

            min_frac = float(getattr(self.config.masking, "watermark_min_fraction", 0.0005))
            if wm_frac < min_frac:
                logger.warning(
                    f"[ETAPA 3] Sin marca de agua detectable (fracción={wm_frac:.5f} < {min_frac}): "
                    "se omite FLUX y se conserva la imagen original."
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
                logger.warning(f"[DEBUG] no se pudo guardar el output de FLUX: {_e}")

            logger.info(f"Etapa 3 completada en {time.time() - t0:.2f}s")

            # Swap real: con refino LoRA viene un segundo DiT 12B (+2º T5);
            # sin descargar este, ambos conviven en RAM y el nodo muere (OOM).
            if str(getattr(self.config.superres, "face_refine_backend", "lora")).lower() == "lora":
                try:
                    self.inpainter.unload()
                except Exception as _e:
                    logger.warning(f"Unload Etapa 3 omitido ({_e}).")

            # ---------------------------------------------------------
            # Etapa 4: Super-Resolución
            # ---------------------------------------------------------
            if self.config.superres.enable_face_refinement:
                logger.info("[ETAPA 4] Post-Procesamiento y Super-Resolución")
                t0 = time.time()

                # Asignación cara↔ref SOBRE la restaurada (pre-upscale): el
                # refino LoRA la necesita para dar a cada cara su LoRA.
                pre_faces = {"faces": [], "assignment": {}, "min_assigned": None, "note": "no evaluada"}
                try:
                    if ref_arcs:
                        pre_faces = self.identity_extractor.verify_faces(restored_image, ref_arcs, top_k=2)
                        logger.info(f"[ETAPA 4] pre-asignación: {pre_faces.get('assignment', {})}")
                except Exception as _e:
                    logger.warning(f"[ETAPA 4] pre-asignación omitida: {_e}")

                self.super_res.face_restorer.to(self.active_device)
                self.super_res.supir_model.to(self.active_device)

                final_image = self.super_res.upscale(restored_image, faces_detail=pre_faces)

                self.super_res.face_restorer.to(self.host_device)
                self.super_res.supir_model.to(self.host_device)


                logger.info(f"Etapa 4 completada en {time.time() - t0:.2f}s")
            else:
                final_image = restored_image
                pre_faces = {"faces": [], "assignment": {}, "min_assigned": None, "note": "etapa omitida"}
            
            # Métrica objetiva de identidad (discriminativa, rol AdaFace/ArcFace):
            # matriz cada-rostro-final × cada-referencia (A, B por separado).
            # Solo informa (warning bajo umbral); nunca tumba el pipeline.
            identity_metric = {"faces": [], "assignment": {}, "min_assigned": None, "note": "no evaluada"}
            try:
                if ref_arcs:
                    identity_metric = self.identity_extractor.verify_faces(final_image, ref_arcs, top_k=2)
                    thr = float(getattr(self.config.identity, "match_threshold", 0.4))
                    for _f in identity_metric.get("faces", []):
                        logger.info(f"[MÉTRICA] rostro {_f['idx']} cosenos={_f.get('cosines', {})}")
                    for _fi, _a in identity_metric.get("assignment", {}).items():
                        _msg = (
                            f"[MÉTRICA] rostro {_fi} -> ref {_a['ref']} "
                            f"cos={_a['cosine']:.3f} umbral={thr}"
                        )
                        if _a["cosine"] < thr:
                            logger.warning(_msg + " -> BAJO UMBRAL, revisar identidad.")
                        else:
                            logger.info(_msg + " -> OK.")
                    if not identity_metric.get("assignment"):
                        logger.warning("[MÉTRICA] sin asignación cara↔ref: no evaluable.")
            except Exception as _e:
                logger.warning(f"[MÉTRICA] verificación de identidad omitida: {_e}")
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
                logger.warning(f"[MÉTRICA] no se pudo guardar el JSON: {_e}")

            # Guardar salida
            os.makedirs(os.path.join(self.config.paths.output_dir, "restored"), exist_ok=True)
            output_path = os.path.join(self.config.paths.output_dir, "restored", f"task_{self.config.task_id}_final.png")
            final_image.save(output_path)
            
            total_time = time.time() - start_time
            logger.info(f"=== PIPELINE COMPLETADO en {total_time:.2f}s ===")
            return output_path
            
        except torch.cuda.OutOfMemoryError:
            logger.error("FATAL: OutOfMemoryError incluso con Sequential Swapping.")
            self._flush_memory()
            raise
        except Exception as e:
            logger.error(f"Error en el pipeline: {e}")
            raise