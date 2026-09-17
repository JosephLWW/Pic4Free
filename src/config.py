from dataclasses import dataclass, field
from omegaconf import OmegaConf
from typing import List, Optional
import os

@dataclass
class PathsConfig:
    input_dir: str = "data/input"
    identity_a_dir: str = "data/identity_person_A"
    identity_b_dir: str = "data/identity_person_B"
    output_dir: str = "data/output/run_default"

@dataclass
class MaskingConfig:
    model_id: str = "PramaLLC/BEN2"
    ssim_threshold: float = 0.65
    # BEN hoy solo produce debug_mask.png (no alimenta inpainting).
    # false = ni se descarga el modelo ni se ejecuta (ahorra ~1s + VRAM swap).
    # true lo reactiva cuando haga falta diagnosticar segmentación.
    enable_ben_debug: bool = False
    # Máscara de watermark por diferencia contra la miniatura limpia (thumb).
    # La thumb es la versión limpia de la misma foto: lo que difiere es overlay.
    watermark_diff_threshold: float = 18.0  # umbral 0-255 sobre lum(watermarked)-lum(thumb)>0
    watermark_dilate_px: int = 2            # con patrón denso, 7px fusiona todo (35% en task 0)
    watermark_min_fraction: float = 0.0005  # bajo esto se considera "sin marca" y se omite FLUX

@dataclass
class IdentityConfig:
    model_name: str = "adaface_ir101"
    margin_adaptation: bool = True
    confidence_divisor: float = 35.0
    match_threshold: float = 0.4  # coseno ArcFace mínimo final-vs-ref (solo aviso, no tumba)
    # Selección de identidad para inyección: "average" (A+B promediados),
    # "auto" (la cara visible de la thumb vota A/B por coseno, se inyecta la
    # ganadora), "A" o "B" (manual). "auto" evita mezclar dos personas.
    selection: str = "average"
    # Puerta de inyección generativa: "pulid" (activa) o "none" (Kontext solo;
    # la métrica verify_faces sigue funcionando igual).
    # PuLID retirado por evidencia (full-weight destruye identidad: 0.70->0.10
    # en task 53): el código queda dormido, no borrado, por si se rehabilita.
    injection: str = "none"
    # Modo métrico: con injection=none, la Etapa 2 omite EVA-giant + IDFormer
    # (~4 min/task) y solo calcula ArcFace-512 para la métrica + voto.
    # Poner false para restaurar la biometría completa (obligatorio si
    # injection vuelve a "pulid": el modo métrico no produce embeddings).
    metric_only: bool = True

@dataclass
class FluxConfig:
    model_id: str = "black-forest-labs/FLUX.1-dev"
    pulid_model_id: str = "guozinan/PuLID"
    # El repo NO contiene "pulid_flux.safetensors": los ficheros reales son
    # pulid_flux_v0.9.0/v0.9.1.safetensors. Nombre configurable + fallbacks.
    pulid_ckpt_file: str = "pulid_flux_v0.9.1.safetensors"
    use_inpaint_pipeline: bool = True  # mezcla de latentes con mask; si false, T2I+composite con warning
    # Backend de edición: "flux" (FLUX.1-dev + mezcla de latentes con máscara) o
    # "kontext" (FLUX.1-Kontext-dev, edición instruction-based nativa, sin máscara).
    backend: str = "flux"
    kontext_model_id: str = "black-forest-labs/FLUX.1-Kontext-dev"
    edit_instruction: str = (
        "Remove the semi-transparent white repeating text watermark pattern "
        "(words and circular logos) covering the whole photo, restoring the "
        "original scene underneath. Keep the person, surfboard, sea and framing "
        "exactly the same, photorealistic, no new watermarks, no text added."
    )
    num_inference_steps: int = 35
    guidance_scale: float = 3.5
    seed: int = 42
    prompt: str = "photorealistic high quality restored face, smooth skin microtexture, sharp eyes, 4k resolution"
    id_scale: float = 1.0
    dropout_p: float = 0.0
    hf_token_file: str = "hf_token.txt"

@dataclass
class SuperResConfig:
    model_id: str = "SUPIR"
    scale_factor: int = 2
    face_fidelity_weight: float = 0.85
    enable_face_refinement: bool = True
    enable_upscaling: bool = False  # Cambia a 'True' si SUPIR funciona correctamente después de validar dependencias.  # SUPIR upscaling desactivado por defecto (costoso)
    # Salida HD (no 4K): el lado largo se lleva a target_long_edge px.
    target_long_edge: int = 1920
    # Backend de refino facial: "lora" (FLUX.1-dev img2img por recorte facial,
    # con LoRA de la persona si existe su fichero, si no pase genérico) o
    # "restoreformer" (prior genérico ONNX, legado).
    face_refine_backend: str = "lora"
    flux_model_id: str = "black-forest-labs/FLUX.1-dev"
    lora_a_path: str = "weights/lora_person_A.safetensors"
    lora_b_path: str = "weights/lora_person_B.safetensors"
    lora_trigger_a: str = "P4F_A"
    lora_trigger_b: str = "P4F_B"
    # 0.45 por defecto. DECISIÓN REGISTRADA: no se ajusta globalmente con 1-2
    # tasks (el óptimo es por imagen: densidad del watermark, frontal/perfil,
    # tamaño del rostro). Esquema adaptativo futuro aparcado hasta el barrido.
    lora_strength: float = 0.45
    lora_guidance: float = 3.5
    lora_steps: int = 28
    lora_crop_expand: float = 2.0
    lora_feather_px: int = 24
    lora_min_crop: int = 512
    refine_unassigned: bool = True  # caras sin identidad asignada: pase genérico

@dataclass
class PipelineConfig:
    task_id: str = "0"
    job_id: str = "manual"
    paths: PathsConfig = field(default_factory=PathsConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    flux: FluxConfig = field(default_factory=FluxConfig)
    superres: SuperResConfig = field(default_factory=SuperResConfig)

def load_config(cli_args: Optional[List[str]] = None) -> PipelineConfig:
    base_cfg = OmegaConf.structured(PipelineConfig)
    if cli_args:
        cli_cfg = OmegaConf.from_dotlist(cli_args)
        return OmegaConf.merge(base_cfg, cli_cfg)
    return base_cfg
