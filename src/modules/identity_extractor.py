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
    """Extractor PuLID-FLUX with IDFormer official.

    Dual branch as in the original design (ToTheBeginning/PuLID):
      - Rama facial: detector SCRFD + embedding ArcFace 512-D (insightface).
      - Visual branch: EVA-CLIP encoder (timm) as a proxy for the official EVA02-CLIP.
    Both feed the official IDFormer (weights pulid_encoder.*, strict loading),
    which produces the identity embedding (1, 32, 2048) expected by the branches
    pulid_ca of the DiT. Documented approximation: the official ViT provides 5 scales
    of 1024-D and 768-D conditioning; here they are derived by deterministic pooling of the
    pooled EVA (without invented weights). The alternative (injecting AdaFace purely in
    unrelated projections) breaks adapter compatibility.
    """

    ARCFACE_DIM = 512
    ID_TOKENS = 32
    ID_DIM = 2048
    # timm: variante gigante ViT-g/14 with weights CLIP. Fallbacks if unavailable.
    EVA_CANDIDATES = (
        "eva_giant_patch14_224.clip_ft_in1k",
        "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        "vit_base_patch16_clip_224.openai",
    )

    def __init__(self, config=None, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        # Compatibility with pipeline antiguo that expected `.model`.
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
            logger.info("InsightFace buffalo_l (SCRFD + ArcFace) initialized.")
        except ImportError:
            logger.warning("insightface is not installed: the ArcFace branch will use a full-image crop with a warning.")
        except Exception as e:
            logger.warning(f"InsightFace could not initialize ({e}); will use fallback with warning.")

        # --- Visual branch: EVA-CLIP (lazy loading to avoid breaking imports in CPU/HPC) ---
        # It is not loaded here to preserve Sequential Swapping; it is loaded in .to()/extract.
        logger.info("PulidIdentityExtractor ready (EVA-CLIP lazy loading).")

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

    # --- API compatible with Sequential Swapping ---
    def to(self, device):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        for attr in ("eva_model", "idformer"):
            m = getattr(self, attr, None)
            if m is not None:
                try:
                    setattr(self, attr, m.to(self.device))
                except Exception as e:
                    logger.warning(f"Could not move {attr} to {self.device}: {e}")
        return self

    def _ensure_eva(self):
        if self.eva_model is not None:
            return
        try:
            import timm
        except ImportError:
            logger.warning("timm not installed: EVA-CLIP branch disabled; ArcFace only.")
            return
        last_err = None
        for name in self.EVA_CANDIDATES:
            try:
                m = timm.create_model(name, pretrained=True, num_classes=0)
                m.eval()
                self._eva_dim = getattr(m, "num_features", None) or getattr(m, "embed_dim", None)
                self.eva_model = m.to(self.device)
                self.eva_name = name
                logger.info(f"EVA-CLIP loaded: {name} (dim={self._eva_dim}).")
                return
            except Exception as e:
                last_err = e
                continue
        logger.warning(f"EVA-CLIP could not be loaded ({last_err}); ArcFace only.")

    def _arcface_embed(self, aligned_pil: Image.Image) -> tuple:
        """Returns (emb512_raw, probe_quality_[0,1], raw_norm).

        Quality follows the AdaFace concept (norm as a quality proxy):
        - embedding raw available -> raw_norm/35 (ArcFace norms ~15-30).
        - normed_embedding only -> detector det_score.
        - no face -> fixed 0.2 (poor signal, already already warned in logs).
        NOTE: measuring the norm of the NORMALIZED embedding always gives ~1.0 and breaks
        the metric (was the bug causing 0.100 to be stuck in every run).
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
                logger.warning(f"InsightFace recognition failure ({e}); fallback visual.")
        # Deterministic fallback (not random): mean of the aligned projected tensor.
        t = self.arcface_t(aligned_pil).flatten().float()
        # Project to a deterministic 512-D shape using reshape+mean.
        n = t.numel()
        t = t[: (n // self.ARCFACE_DIM) * self.ARCFACE_DIM].reshape(self.ARCFACE_DIM, -1).mean(dim=1)
        return F.normalize(t.unsqueeze(0), p=2, dim=1), 0.2, 0.2 * divisor

    def _vit_pooled(self, face_pil: Image.Image) -> torch.Tensor:
        """Pooled visual raw (without normalization) for the official IDFormer."""
        self._ensure_eva()
        if self.eva_model is None:
            raise RuntimeError("EVA-CLIP unavailable: cannot feed the IDFormer.")
        try:
            x = self.clip_t(face_pil.convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                f = self.eva_model.forward_features(x) if hasattr(self.eva_model, "forward_features") else self.eva_model(x)
                if isinstance(f, (list, tuple)):
                    f = f[0]
                if f.dim() == 3:  # tokens -> pool medio (without CLS assumptions)
                    f = f.mean(dim=1)
            return f.float().cpu()
        except Exception as e:
            raise RuntimeError(f"EVA-CLIP inference failure ({e}).") from e

    def _pulid_ckpt(self):
        """Locates the official checkpoint (same HF cache as the inpainter)."""
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
        raise RuntimeError(f"Checkpoint PuLID no downloadble in {repo}: {last_err}")

    def _ensure_idformer(self):
        if self.idformer is not None:
            return
        from safetensors.torch import load_file

        ckpt_path = self._pulid_ckpt()
        full = load_file(ckpt_path)
        enc_sd = {k[len("pulid_encoder."):]: v for k, v in full.items() if k.startswith("pulid_encoder.")}
        if not enc_sd:
            raise RuntimeError("Checkpoint without the official pulid_encoder branch.")
        m = IDFormer()
        m.load_state_dict(enc_sd, strict=True)
        self.idformer = m.to(self.device)
        self.idformer.eval()
        logger.info(f"IDFormer official loaded ({ckpt_path}).")

    def _metric_only_active(self) -> bool:
        """Metric mode: skips EVA+IDFormer unless PuLID needs them."""
        if self.config is None or getattr(self.config, "identity", None) is None:
            return False
        if str(getattr(self.config.identity, "injection", "pulid")).lower() != "none":
            return False  # Active PuLID requires complete embeddings
        return bool(getattr(self.config.identity, "metric_only", True))

    @torch.no_grad()
    def reference_arcface(self, image_paths: List[str]) -> torch.Tensor:
        """Robust ArcFace-512 mean: the largest face in the COMPLETE image.

        Unlike the extraction path (re-detection on aligned crops
        112px + fallback vectors that contaminated the mean); here only
        entran embeddings reales; the images no face is skippedn with warning.
        Identical by construction to the offline verifier reference.
        """
        vecs = []
        for path in image_paths:
            if not os.path.exists(path):
                logger.warning(f"Reference inexistente, is skipped: {path}")
                continue
            if self.app is None:
                raise RuntimeError("InsightFace unavailable for reference robusta.")
            try:
                faces = self.app.get(np.array(Image.open(path).convert("RGB")))
            except Exception as e:
                logger.warning(f"Deteccion failure in {path} ({e}); is skipped.")
                continue
            if not faces:
                logger.warning(f"[CALIDAD] No faces in {path}; is excluded from the reference.")
                continue
            best = max(faces, key=self._face_area)
            emb = getattr(best, "normed_embedding", None)
            if emb is None:
                logger.warning(f"[CALIDAD] without embedding in {path}; is excluye.")
                continue
            v = torch.tensor(np.asarray(emb, dtype=np.float32)).reshape(1, -1)
            vecs.append(F.normalize(v, p=2, dim=1))
        if not vecs:
            raise ValueError("Ninguna reference with face valido.")
        return F.normalize(torch.mean(torch.cat(vecs, dim=0), dim=0, keepdim=True), p=2, dim=1)

    @torch.no_grad()
    def extract_identity(self, image_paths: List[str]) -> Dict[str, Any]:
        logger.info(f"Extrayendo identity PuLID (ArcFace+EVA-CLIP) of {len(image_paths)} imagees...")
        if not image_paths:
            logger.warning("without imagees of reference: tensor of ceros (1,32,2048), confidence 0.")
            return {
                "embeddings": torch.zeros(1, self.ID_TOKENS, self.ID_DIM),
                "confidence": 0.0,
                "norm": 0.0,
            }

        if self._metric_only_active():
            # Without EVA or IDFormer: robust ArcFace only for metric + vote.
            # embeddings = ceros with shape valida (the generacion no the uses with
            # injection=none; the shape-check the acepta).
            arc = self.reference_arcface(image_paths)
            n = max(1, len(image_paths))
            logger.info(f"[STAGE 2] metric-only mode: {n} image(s), without EVA/IDFormer.")
            return {
                "embeddings": torch.zeros(1, self.ID_TOKENS, self.ID_DIM),
                "arcface": arc,
                "confidence": 0.5,
                "norm": 0.0,
            }

        fused, arc_bank, norms, probes = [], [], [], []
        for path in image_paths:
            if not os.path.exists(path):
                logger.warning(f"Reference inexistente, is skipped: {path}")
                continue
            img_pil = Image.open(path).convert("RGB")
            img_np = np.array(img_pil)

            aligned_pil = None
            face_found = False
            if self.app is not None:
                try:
                    faces = self.app.get(img_np)
                except Exception as e:
                    logger.warning(f"InsightFace deteccion failure in {path} ({e}).")
                    faces = []
                if faces:
                    face_found = True
                    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    try:
                        from insightface.utils import face_align

                        aligned_np = face_align.norm_crop(img_np, landmark=best.kps)
                        aligned_pil = Image.fromarray(aligned_np)
                    except Exception as e:
                        logger.warning(f"face_align failure in {path} ({e}); uso recorte bbox.")
                        x1, y1, x2, y2 = (int(v) for v in best.bbox)
                        aligned_pil = img_pil.crop((max(0, x1), max(0, y1), x2, y2))
                else:
                    logger.warning(
                        f"[CALIDAD] without InsightFace faces at {path}: degraded/occluded reference, "
                        "use the full image as a fallback (penalizing confidence)."
                    )
            if aligned_pil is None:
                if self.app is None:
                    logger.warning(f"[CALIDAD] InsightFace unavailable; {path} without alinear (fallback).")
                aligned_pil = img_pil

            # Contrato IDFormer official: id_cond (B,1280) = [arcface-512, vit-768-norm],
            # and = 5 scales ViT (B,N,1024). the pooled EVA local is deriva by pooling
            # deterministic (documented in the class docstring).
            self._ensure_idformer()
            arc, probe, raw_norm = self._arcface_embed(aligned_pil)
            arc = arc.float()  # (1,512), raw as antelopev2 when a face is present
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
                raise RuntimeError(f"IDFormer returned {tuple(emb.shape)}; expected (1, 32, 2048).")
            # Magnitude calibration: approximate ViT inputs may produce
            # to the distribution resampler (norms in the thousands). L2-normalize for
            # the token preserves direction and bounds magnitude to a healthy range
            # for the branches pulid_ca (forensic 6949812: std=338, norm=9179).
            raw_norm = float(emb.float().norm(p=2, dim=-1).mean())
            emb = F.normalize(emb.float(), p=2, dim=-1)
            logger.info(
                f"[ID-Calib] mean norm pre={raw_norm:.1f} post={float(emb.norm(p=2, dim=-1).mean()):.3f} "
                f"({path})"
            )

            logger.info(
                f"[CALIBRACION] {Path(path).name}: face_found={face_found} "
                f"raw_norm={raw_norm:.2f} probe={probe:.3f}"
            )
            fused.append(emb.float().cpu())
            arc_bank.append(F.normalize(arc, p=2, dim=1))
            norms.append(raw_norm)
            probes.append(probe)

        if not fused:
            raise ValueError("without embeddings validos: no reference procesable.")

        avg = torch.mean(torch.cat(fused, dim=0), dim=0, keepdim=True)  # (1,32,2048)
        avg_arc = F.normalize(torch.mean(torch.cat(arc_bank, dim=0), dim=0, keepdim=True), p=2, dim=1)
        avg_norm = sum(norms) / len(norms)
        confidence = min(1.0, max(0.0, sum(probes) / len(probes)))
        return {
            "embeddings": avg,
            "arcface": avg_arc,  # (1,512) normalized: for final-vs-reference cosine metric
            "confidence": float(confidence),
            "norm": float(avg_norm),
        }

    @torch.no_grad()
    def verify(self, image_pil: Image.Image, ref_arc: torch.Tensor) -> Dict[str, Any]:
        """ArcFace cosine similarity between the final image and the reference.

        Objective identity gate: it does not replace PuLID (generative); it measures it
        (discriminative, AdaFace/ArcFace role). It must never fail the pipeline.
        """
        try:
            if self.app is None:
                return {"cosine": 0.0, "face_found": False, "note": "without InsightFace"}
            arr = np.array(image_pil.convert("RGB"))
            faces = self.app.get(arr)
            if not faces:
                return {"cosine": 0.0, "face_found": False, "note": "no faces in final"}
            best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = getattr(best, "normed_embedding", None)
            if emb is None:
                return {"cosine": 0.0, "face_found": True, "note": "no embedding in final"}
            v = torch.tensor(np.asarray(emb, dtype=np.float32)).reshape(1, -1)
            v = F.normalize(v, p=2, dim=1)
            r = F.normalize(ref_arc.float().reshape(1, -1), p=2, dim=1)
            cos = float((v * r).sum())
            return {"cosine": cos, "face_found": True, "note": "ok"}
        except Exception as e:
            logger.warning(f"verify() failure ({e}); degraded metric.")
            return {"cosine": 0.0, "face_found": False, "note": f"error: {e}"}

    @torch.no_grad()
    def select_identity(self, image_pil: Image.Image, refs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Votes for which reference (A/B/...) is visible in the image.

        Compares the largest face in `image_pil` (typically the clean thumbnail)
        against each reference by ArcFace cosine and returns the winner.
        Used to inject an identity instead of the mixed A+B average.
        """
        try:
            if self.app is None:
                return {"label": None, "cosines": {}, "note": "without InsightFace"}
            arr = np.array(image_pil.convert("RGB"))
            faces = sorted(self.app.get(arr), key=self._face_area, reverse=True)
            if not faces:
                return {"label": None, "cosines": {}, "note": "no faces"}
            emb = getattr(faces[0], "normed_embedding", None)
            if emb is None:
                return {"label": None, "cosines": {}, "note": "no embedding"}
            v = F.normalize(torch.tensor(np.asarray(emb, dtype=np.float32)).reshape(1, -1), p=2, dim=1)
            cosines = {}
            for k, r in refs.items():
                rr = F.normalize(r.float().reshape(1, -1), p=2, dim=1)
                cosines[k] = float((v * rr).sum())
            label = max(cosines, key=cosines.get)
            logger.info(f"[SELECCION] vote identity: {cosines} -> {label}")
            return {"label": label, "cosines": cosines, "note": "ok"}
        except Exception as e:
            logger.warning(f"select_identity() failure ({e}).")
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
        """Matrix of cosines for each final face and each reference (A, B, ...).

        Each reference is evaluated SEPARATELY (not averaged: mixing two
        different people produce a reference that represents no one). Assignment
        greedy maximum. It never fails the pipeline.
        """
        try:
            if self.app is None:
                return {"faces": [], "assignment": {}, "min_assigned": None, "note": "without InsightFace"}
            arr = np.array(image_pil.convert("RGB"))
            faces = sorted(self.app.get(arr), key=self._face_area, reverse=True)[: max(1, top_k)]
            if not faces:
                return {"faces": [], "assignment": {}, "min_assigned": None, "note": "no faces in final"}
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
            # Greedy assignment: (face, reference) pairs from highest to lowest cosine, without repeats.
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
            logger.warning(f"verify_faces() failure ({e}); degraded metric.")
            return {"faces": [], "assignment": {}, "min_assigned": None, "note": f"error: {e}"}
