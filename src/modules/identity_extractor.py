import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.modules.pulid_official import IDFormer

logger = logging.getLogger("Pic4Free.IdentityExtractor")


class PulidIdentityExtractor:
    """Extractor PuLID-FLUX con IDFormer oficial.

    Doble rama como en el diseño original (ToTheBeginning/PuLID):
      - Rama facial: detector SCRFD + embedding ArcFace 512-D (insightface).
      - Rama visual: encoder EVA-CLIP (timm) como proxy del EVA02-CLIP oficial.
    Ambas alimentan el IDFormer oficial (pesos pulid_encoder.*, carga estricta),
    que produce el embedding de identidad (1, 32, 2048) que esperan las ramas
    pulid_ca del DiT. Aproximación documentada: el ViT oficial aporta 5 escalas
    de 1024-D y cond 768-D; aquí se derivan por pooling determinista del
    pooled EVA (sin pesos inventados). La alternativa (inyectar AdaFace puro en
    proyecciones ajenas) rompe la compatibilidad del adaptador.
    """

    ARCFACE_DIM = 512
    ID_TOKENS = 32
    ID_DIM = 2048
    # timm: variante gigante ViT-g/14 con pesos CLIP. Fallbacks si no disponible.
    EVA_CANDIDATES = (
        "eva_giant_patch14_224.clip_ft_in1k",
        "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        "vit_base_patch16_clip_224.openai",
    )

    def __init__(self, config=None, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        # Compatibilidad con pipeline antiguo que esperaba `.model`.
        self.model = None

        self.app = None
        self.eva_model = None
        self.eva_name = None
        self._eva_dim = None
        self.idformer = None

        # --- Rama facial: InsightFace deteccion + reconocimiento ---
        try:
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
            ctx = 0 if "cuda" in str(self.device) else -1
            self.app.prepare(ctx_id=ctx, det_size=(640, 640))
            logger.info("InsightFace buffalo_l (SCRFD + ArcFace) inicializado.")
        except ImportError:
            logger.warning("insightface no instalado: rama ArcFace usara recorte completo con warning por imagen.")
        except Exception as e:
            logger.warning(f"InsightFace no pudo inicializarse ({e}); se usara fallback con warning.")

        # --- Rama visual: EVA-CLIP (carga lazy para no romper import en CPU/HPC) ---
        # No se carga aqui para respetar Sequential Swapping; se carga en .to()/extract.
        logger.info("PulidIdentityExtractor listo (EVA-CLIP carga lazy).")

        self.arcface_t = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.clip_t = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    # --- API compatible con Sequential Swapping ---
    def to(self, device):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        for attr in ("eva_model", "idformer"):
            m = getattr(self, attr, None)
            if m is not None:
                try:
                    setattr(self, attr, m.to(self.device))
                except Exception as e:
                    logger.warning(f"No se pudo mover {attr} a {self.device}: {e}")
        return self

    def _ensure_eva(self):
        if self.eva_model is not None:
            return
        try:
            import timm
        except ImportError:
            logger.warning("timm no instalado: rama EVA-CLIP desactivada, solo ArcFace.")
            return
        last_err = None
        for name in self.EVA_CANDIDATES:
            try:
                m = timm.create_model(name, pretrained=True, num_classes=0)
                m.eval()
                self._eva_dim = getattr(m, "num_features", None) or getattr(m, "embed_dim", None)
                self.eva_model = m.to(self.device)
                self.eva_name = name
                logger.info(f"EVA-CLIP cargado: {name} (dim={self._eva_dim}).")
                return
            except Exception as e:
                last_err = e
                continue
        logger.warning(f"EVA-CLIP no pudo cargarse ({last_err}); solo ArcFace.")

    def _arcface_embed(self, aligned_pil: Image.Image) -> tuple:
        """Devuelve (emb512_crudo, probe_calidad_[0,1], norma_cruda).

        La calidad sigue la idea AdaFace (norma como proxy de calidad):
        - embedding crudo disponible -> norma_cruda/35 (normas ArcFace ~15-30).
        - solo normed_embedding -> det_score del detector.
        - sin rostro -> 0.2 fijo (señal mala, ya advertida en logs).
        NOTA: medir la norma del embedding NORMALIZADO siempre da ~1.0 y rompe
        la métrica (fue el bug del 0.100 clavado en todos los runs).
        """
        divisor = 35.0
        if self.config is not None and getattr(self.config, "identity", None) is not None:
            divisor = float(getattr(self.config.identity, "confidence_divisor", 35.0))
        arr = np.array(aligned_pil.convert("RGB"))
        if self.app is not None:
            try:
                faces = self.app.get(arr)
                if faces:
                    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    raw = getattr(best, "embedding", None)
                    if raw is not None:
                        raw_t = torch.tensor(np.asarray(raw, dtype=np.float32)).unsqueeze(0)
                        raw_norm = float(torch.norm(raw_t, p=2, dim=1).mean())
                        return raw_t, min(1.0, raw_norm / divisor), raw_norm
                    normed = getattr(best, "normed_embedding", None)
                    if normed is not None:
                        det = float(getattr(best, "det_score", 0.8) or 0.8)
                        return (
                            torch.tensor(np.asarray(normed, dtype=np.float32)).unsqueeze(0),
                            min(1.0, max(0.0, det)),
                            det * divisor,
                        )
            except Exception as e:
                logger.warning(f"InsightFace recognition fallo ({e}); fallback visual.")
        # Fallback determinista (no aleatorio): media del tensor alineado proyectada.
        t = self.arcface_t(aligned_pil).flatten().float()
        # Proyectar a 512-D de forma determinista mediante reshape+media.
        n = t.numel()
        t = t[: (n // self.ARCFACE_DIM) * self.ARCFACE_DIM].reshape(self.ARCFACE_DIM, -1).mean(dim=1)
        return F.normalize(t.unsqueeze(0), p=2, dim=1), 0.2, 0.2 * divisor

    def _vit_pooled(self, face_pil: Image.Image) -> torch.Tensor:
        """Pooled visual crudo (sin normalizar) para el IDFormer oficial."""
        self._ensure_eva()
        if self.eva_model is None:
            raise RuntimeError("EVA-CLIP no disponible: imposible alimentar el IDFormer.")
        try:
            x = self.clip_t(face_pil.convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                f = self.eva_model.forward_features(x) if hasattr(self.eva_model, "forward_features") else self.eva_model(x)
                if isinstance(f, (list, tuple)):
                    f = f[0]
                if f.dim() == 3:  # tokens -> pool medio (sin CLS assumptions)
                    f = f.mean(dim=1)
            return f.float().cpu()
        except Exception as e:
            raise RuntimeError(f"EVA-CLIP inferencia fallo ({e}).") from e

    def _pulid_ckpt(self):
        """Localiza el checkpoint oficial (misma caché HF que el inpainter)."""
        from huggingface_hub import hf_hub_download

        repo, token = "guozinan/PuLID", None
        configured = None
        if self.config is not None and getattr(self.config, "flux", None) is not None:
            repo = getattr(self.config.flux, "pulid_model_id", repo)
            configured = getattr(self.config.flux, "pulid_ckpt_file", None)
        if self.config is not None and getattr(self.config, "flux", None) is not None:
            tp = getattr(self.config.flux, "hf_token_file", None)
            if tp and os.path.exists(tp):
                with open(tp) as fh:
                    token = fh.read().strip() or None
        if token is None:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        candidates = [c for c in [configured, "pulid_flux_v0.9.1.safetensors", "pulid_flux_v0.9.0.safetensors"] if c]
        last_err = None
        for fname in dict.fromkeys(candidates):
            try:
                return hf_hub_download(repo_id=repo, filename=fname, token=token)
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Checkpoint PuLID no descargable en {repo}: {last_err}")

    def _ensure_idformer(self):
        if self.idformer is not None:
            return
        from safetensors.torch import load_file

        ckpt_path = self._pulid_ckpt()
        full = load_file(ckpt_path)
        enc_sd = {k[len("pulid_encoder."):]: v for k, v in full.items() if k.startswith("pulid_encoder.")}
        if not enc_sd:
            raise RuntimeError("Checkpoint sin rama pulid_encoder oficial.")
        m = IDFormer()
        m.load_state_dict(enc_sd, strict=True)
        self.idformer = m.to(self.device)
        self.idformer.eval()
        logger.info(f"IDFormer oficial cargado ({ckpt_path}).")

    def _metric_only_active(self) -> bool:
        """Modo métrico: omite EVA+IDFormer salvo que PuLID los necesite."""
        if self.config is None or getattr(self.config, "identity", None) is None:
            return False
        if str(getattr(self.config.identity, "injection", "pulid")).lower() != "none":
            return False  # PuLID activo exige embeddings completos
        return bool(getattr(self.config.identity, "metric_only", True))

    @torch.no_grad()
    def reference_arcface(self, image_paths: List[str]) -> torch.Tensor:
        """Media robusta de ArcFace-512: rostro mayor de la imagen COMPLETA.

        A diferencia del path de extracción (re-detección sobre crops alineados
        de 112px + vectores de fallback que contaminaban la media), aquí solo
        entran embeddings reales; las imágenes sin rostro se omiten con warning.
        Idéntico por construcción a la referencia del verificador offline.
        """
        vecs = []
        for path in image_paths:
            if not os.path.exists(path):
                logger.warning(f"Referencia inexistente, se omite: {path}")
                continue
            if self.app is None:
                raise RuntimeError("InsightFace no disponible para referencia robusta.")
            try:
                faces = self.app.get(np.array(Image.open(path).convert("RGB")))
            except Exception as e:
                logger.warning(f"Detección fallo en {path} ({e}); se omite.")
                continue
            if not faces:
                logger.warning(f"[CALIDAD] Sin rostros en {path}; se excluye de la referencia.")
                continue
            best = max(faces, key=self._face_area)
            emb = getattr(best, "normed_embedding", None)
            if emb is None:
                logger.warning(f"[CALIDAD] Sin embedding en {path}; se excluye.")
                continue
            v = torch.tensor(np.asarray(emb, dtype=np.float32)).reshape(1, -1)
            vecs.append(F.normalize(v, p=2, dim=1))
        if not vecs:
            raise ValueError("Ninguna referencia con rostro válido.")
        return F.normalize(torch.mean(torch.cat(vecs, dim=0), dim=0, keepdim=True), p=2, dim=1)

    @torch.no_grad()
    def extract_identity(self, image_paths: List[str]) -> Dict[str, Any]:
        logger.info(f"Extrayendo identidad PuLID (ArcFace+EVA-CLIP) de {len(image_paths)} imagenes...")
        if not image_paths:
            logger.warning("Sin imagenes de referencia: tensor de ceros (1,32,2048), confianza 0.")
            return {
                "embeddings": torch.zeros(1, self.ID_TOKENS, self.ID_DIM),
                "confidence": 0.0,
                "norm": 0.0,
            }

        if self._metric_only_active():
            # Sin EVA ni IDFormer: solo ArcFace robusto para métrica + voto.
            # embeddings = ceros con forma válida (la generación no los usa con
            # injection=none; el shape-check los acepta).
            arc = self.reference_arcface(image_paths)
            n = max(1, len(image_paths))
            logger.info(f"[ETAPA 2] modo metric-only: {n} imagen(es), sin EVA/IDFormer.")
            return {
                "embeddings": torch.zeros(1, self.ID_TOKENS, self.ID_DIM),
                "arcface": arc,
                "confidence": 0.5,
                "norm": 0.0,
            }

        fused, arc_bank, norms, probes = [], [], [], []
        for path in image_paths:
            if not os.path.exists(path):
                logger.warning(f"Referencia inexistente, se omite: {path}")
                continue
            img_pil = Image.open(path).convert("RGB")
            img_np = np.array(img_pil)

            aligned_pil = None
            face_found = False
            if self.app is not None:
                try:
                    faces = self.app.get(img_np)
                except Exception as e:
                    logger.warning(f"InsightFace deteccion fallo en {path} ({e}).")
                    faces = []
                if faces:
                    face_found = True
                    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    try:
                        from insightface.utils import face_align

                        aligned_np = face_align.norm_crop(img_np, landmark=best.kps)
                        aligned_pil = Image.fromarray(aligned_np)
                    except Exception as e:
                        logger.warning(f"face_align fallo en {path} ({e}); uso recorte bbox.")
                        x1, y1, x2, y2 = (int(v) for v in best.bbox)
                        aligned_pil = img_pil.crop((max(0, x1), max(0, y1), x2, y2))
                else:
                    logger.warning(
                        f"[CALIDAD] Sin rostros InsightFace en {path}: referencia degradada/ocluida, "
                        "se usa imagen completa como fallback (penalizar confianza)."
                    )
            if aligned_pil is None:
                if self.app is None:
                    logger.warning(f"[CALIDAD] InsightFace no disponible; {path} sin alinear (fallback).")
                aligned_pil = img_pil

            # Contrato IDFormer oficial: id_cond (B,1280) = [arcface-512, vit-768-norm],
            # y = 5 escalas ViT (B,N,1024). El pooled EVA local se deriva por pooling
            # determinista (documentado en el docstring de la clase).
            self._ensure_idformer()
            arc, probe, raw_norm = self._arcface_embed(aligned_pil)
            arc = arc.float()  # (1,512), crudo como antelopev2 cuando hay cara
            pooled = self._vit_pooled(aligned_pil).float().reshape(1, 1, -1)  # (1,1,D)
            vit768 = F.normalize(
                F.adaptive_avg_pool1d(pooled, 768).reshape(1, 768), p=2, dim=1
            )
            vit1024 = F.adaptive_avg_pool1d(pooled, 1024).reshape(1, 1, 1024)
            id_cond = torch.cat([arc, vit768], dim=1)  # (1,1280)
            dev = next(self.idformer.parameters()).device
            dt = next(self.idformer.parameters()).dtype
            y = [vit1024.to(dev, dt) for _ in range(5)]
            emb = self.idformer(id_cond.to(dev, dt), y)  # (1,32,2048)
            if tuple(emb.shape) != (1, self.ID_TOKENS, self.ID_DIM):
                raise RuntimeError(f"IDFormer devolvió {tuple(emb.shape)}; se esperaba (1, 32, 2048).")
            # Calibración de magnitud: las entradas ViT aproximadas pueden sacar
            # al resampler de distribución (normas de miles). Normalizar L2 por
            # token preserva la dirección y acota la magnitud a un rango sano
            # para las ramas pulid_ca (forense 6949812: std=338, norma=9179).
            raw_norm = float(emb.float().norm(p=2, dim=-1).mean())
            emb = F.normalize(emb.float(), p=2, dim=-1)
            logger.info(
                f"[ID-Calib] norma media pre={raw_norm:.1f} post={float(emb.norm(p=2, dim=-1).mean()):.3f} "
                f"({path})"
            )

            logger.info(
                f"[CALIBRACIÓN] {Path(path).name}: face_found={face_found} "
                f"norma_cruda={raw_norm:.2f} probe={probe:.3f}"
            )
            fused.append(emb.float().cpu())
            arc_bank.append(F.normalize(arc, p=2, dim=1))
            norms.append(raw_norm)
            probes.append(probe)

        if not fused:
            raise ValueError("Sin embeddings validos: ninguna referencia procesable.")

        avg = torch.mean(torch.cat(fused, dim=0), dim=0, keepdim=True)  # (1,32,2048)
        avg_arc = F.normalize(torch.mean(torch.cat(arc_bank, dim=0), dim=0, keepdim=True), p=2, dim=1)
        avg_norm = sum(norms) / len(norms)
        confidence = min(1.0, max(0.0, sum(probes) / len(probes)))
        return {
            "embeddings": avg,
            "arcface": avg_arc,  # (1,512) normalizado: para métrica coseno final-vs-ref
            "confidence": float(confidence),
            "norm": float(avg_norm),
        }

    @torch.no_grad()
    def verify(self, image_pil: Image.Image, ref_arc: torch.Tensor) -> Dict[str, Any]:
        """Similitud coseno ArcFace entre la imagen final y la referencia.

        Puerta objetiva de identidad: no sustituye a PuLID (generativo), lo mide
        (discriminativo, rol AdaFace/ArcFace). Nunca debe tumbar el pipeline.
        """
        try:
            if self.app is None:
                return {"cosine": 0.0, "face_found": False, "note": "sin InsightFace"}
            arr = np.array(image_pil.convert("RGB"))
            faces = self.app.get(arr)
            if not faces:
                return {"cosine": 0.0, "face_found": False, "note": "sin rostros en final"}
            best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = getattr(best, "normed_embedding", None)
            if emb is None:
                return {"cosine": 0.0, "face_found": True, "note": "sin embedding en final"}
            v = torch.tensor(np.asarray(emb, dtype=np.float32)).reshape(1, -1)
            v = F.normalize(v, p=2, dim=1)
            r = F.normalize(ref_arc.float().reshape(1, -1), p=2, dim=1)
            cos = float((v * r).sum())
            return {"cosine": cos, "face_found": True, "note": "ok"}
        except Exception as e:
            logger.warning(f"verify() fallo ({e}); métrica degradada.")
            return {"cosine": 0.0, "face_found": False, "note": f"error: {e}"}

    @torch.no_grad()
    def select_identity(self, image_pil: Image.Image, refs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Vota qué referencia (A/B/...) es la persona visible en la imagen.

        Compara el rostro mayor de `image_pil` (típicamente la thumb limpia)
        contra cada referencia por coseno ArcFace y devuelve la ganadora.
        Sirve para inyectar UNA identidad en vez del promedio A+B mezclado.
        """
        try:
            if self.app is None:
                return {"label": None, "cosines": {}, "note": "sin InsightFace"}
            arr = np.array(image_pil.convert("RGB"))
            faces = sorted(self.app.get(arr), key=self._face_area, reverse=True)
            if not faces:
                return {"label": None, "cosines": {}, "note": "sin rostros"}
            emb = getattr(faces[0], "normed_embedding", None)
            if emb is None:
                return {"label": None, "cosines": {}, "note": "sin embedding"}
            v = F.normalize(torch.tensor(np.asarray(emb, dtype=np.float32)).reshape(1, -1), p=2, dim=1)
            cosines = {}
            for k, r in refs.items():
                rr = F.normalize(r.float().reshape(1, -1), p=2, dim=1)
                cosines[k] = float((v * rr).sum())
            label = max(cosines, key=cosines.get)
            logger.info(f"[SELECCIÓN] voto identidad: {cosines} -> {label}")
            return {"label": label, "cosines": cosines, "note": "ok"}
        except Exception as e:
            logger.warning(f"select_identity() fallo ({e}).")
            return {"label": None, "cosines": {}, "note": f"error: {e}"}

    @staticmethod
    def _face_area(face) -> float:
        try:
            x1, y1, x2, y2 = (float(v) for v in face.bbox)
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)
        except Exception:
            return 0.0

    @torch.no_grad()
    def verify_faces(
        self,
        image_pil: Image.Image,
        refs: Dict[str, torch.Tensor],
        top_k: int = 2,
    ) -> Dict[str, Any]:
        """Matriz de cosenos cada-rostro-final × cada-referencia (A, B, ...).

        Cada referencia se evalúa POR SEPARADO (no promediadas: mezclar dos
        personas distintas produce una referencia que no es nadie). Asignación
        greedy por máximo. Nunca tumba el pipeline.
        """
        try:
            if self.app is None:
                return {"faces": [], "assignment": {}, "min_assigned": None, "note": "sin InsightFace"}
            arr = np.array(image_pil.convert("RGB"))
            faces = sorted(self.app.get(arr), key=self._face_area, reverse=True)[: max(1, top_k)]
            if not faces:
                return {"faces": [], "assignment": {}, "min_assigned": None, "note": "sin rostros en final"}
            ref_n = {k: F.normalize(v.float().reshape(1, -1), p=2, dim=1) for k, v in refs.items()}
            out_faces = []
            for idx, face in enumerate(faces):
                emb = getattr(face, "normed_embedding", None)
                try:
                    bbox = [float(v) for v in face.bbox]
                except Exception:
                    bbox = []
                if emb is None:
                    out_faces.append({"idx": idx, "bbox": bbox, "cosines": {}, "area": self._face_area(face)})
                    continue
                v = F.normalize(torch.tensor(np.asarray(emb, dtype=np.float32)).reshape(1, -1), p=2, dim=1)
                cosines = {k: float((v * r).sum()) for k, r in ref_n.items()}
                out_faces.append({"idx": idx, "bbox": bbox, "cosines": cosines, "area": self._face_area(face)})
            # Asignación greedy: pares (cara, ref) de mayor a menor coseno, sin repetir.
            pairs = sorted(
                ((f["idx"], k, c) for f in out_faces for k, c in f["cosines"].items()),
                key=lambda t: t[2],
                reverse=True,
            )
            assignment, used_faces, used_refs = {}, set(), set()
            for fi, k, c in pairs:
                if fi in used_faces or k in used_refs:
                    continue
                assignment[str(fi)] = {"ref": k, "cosine": c}
                used_faces.add(fi)
                used_refs.add(k)
            assigned = [a["cosine"] for a in assignment.values()]
            return {
                "faces": out_faces,
                "assignment": assignment,
                "min_assigned": min(assigned) if assigned else None,
                "note": "ok",
            }
        except Exception as e:
            logger.warning(f"verify_faces() fallo ({e}); métrica degradada.")
            return {"faces": [], "assignment": {}, "min_assigned": None, "note": f"error: {e}"}
