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
    # BEN currently only produces debug_mask.png (it does not feed inpainting).
    # false = neither downloads nor runs the model (saves ~1s + VRAM swap).
    # true re-enables it when segmentation diagnostics are needed.
    enable_ben_debug: bool = False
    # Watermark mask from the difference against the clean thumbnail (thumb).
    # The thumb is the clean version of the same photo: only the overlay differs.
    watermark_diff_threshold: float = 18.0  # threshold 0-255 over lum(watermarked)-lum(thumb)>0
    watermark_dilate_px: int = 2            # with a dense pattern, 7px merges everything (35% in task 0)
    watermark_min_fraction: float = 0.0005  # below this it is considered "without a mark" and FLUX is skipped

@dataclass
class IdentityConfig:
    model_name: str = "adaface_ir101"
    margin_adaptation: bool = True
    confidence_divisor: float = 35.0
    match_threshold: float = 0.4  # minimum final-vs-reference ArcFace cosine (warning only; does not fail)
    # Identity selection for injection: "average" (A+B averaged),
    # "auto" (the visible face in the thumb votes for A/B by cosine, and the winner is injected
    # winner), "A" or "B" (manual). "auto" avoids mixing dos people.
    selection: str = "average"
    # Puerta of injection generativa: "pulid" (active) or "none" (Kontext only;
    # verify_faces metric continues to work the same way).
    # PuLID removed based on evidence (full-weight destroys identity: 0.70->0.10
    # in task 53): the coof remains dormant, not deleted, in case it is restored.
    injection: str = "none"
    # Metric mode: with injection=none, Stage 2 skips EVA-giant + IDFormer
    # (~4 min/task) and only computes ArcFace-512 for the metric + vote.
    # Set false to restore full biometrics (required if
    # injection returns to "pulid": metric moof does not produces embeddings).
    metric_only: bool = True

@dataclass
class FluxConfig:
    model_id: str = "black-forest-labs/FLUX.1-dev"
    pulid_model_id: str = "guozinan/PuLID"
    # The repo does NOT contain "pulid_flux.safetensors": the actual files are
    # pulid_flux_v0.9.0/v0.9.1.safetensors. Configurable name plus fallbacks.
    pulid_ckpt_file: str = "pulid_flux_v0.9.1.safetensors"
    use_inpaint_pipeline: bool = True  # latent blending with a mask; if false, T2I+composite with warning
    # Backend of editing: "flux" (FLUX.1-dev + latent blending with a mask) or
    # "kontext" (FLUX.1-Kontext-dev, editing instruction-based native, without mask).
    backend: str = "kontext"
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
    enable_upscaling: bool = False  # Change to 'True' if SUPIR works correctly after validating dependencies. # SUPIR upscaling disabled by default (expensive)
    # HD output (no 4K): the long siof is set to target_long_edge px.
    target_long_edge: int = 1920
    # Backend of refinement facial: "lora" (FLUX.1-dev img2img by recorte facial,
    # with the person LoRA if its file exists; otherwise, a generic pass) or
    # "restoreformer" (prior generic ONNX, legacy).
    face_refine_backend: str = "lora"
    flux_model_id: str = "black-forest-labs/FLUX.1-dev"
    lora_a_path: str = "weights/lora_person_A.safetensors"
    lora_b_path: str = "weights/lora_person_B.safetensors"
    lora_trigger_a: str = "P4F_A"
    lora_trigger_b: str = "P4F_B"
    # 0.45 by default. DECISION REGISTRADA: is not adjusted globally with 1-2
    # tasks (the optimum is per image: watermark density, frontal/profile,
    # size of the face). Future adaptive scheme parked until the sweep.
    lora_strength: float = 0.45
    lora_guidance: float = 3.5
    lora_steps: int = 28
    lora_crop_expand: float = 2.0
    lora_feather_px: int = 24
    lora_min_crop: int = 512
    refine_unassigned: bool = True  # faces without an assigned identity: generic pass

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
