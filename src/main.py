#!/usr/bin/env python3
"""
==============================================================================
Pic4Free - main.py: Unified Pipeline Orchestrator (Slurm-Compatible Entrypoint)
==============================================================================
Single entrypoint invoked by Slurm Array Jobs. Receives a $SLURM_ARRAY_TASK_ID,
resolves the corresponding image pair, and executes all four stages sequentially:

  Stage 1: SSIM Differential Mask Generation   (src/utils/mask_pipeline.py)
  Stage 2: Multi-Person Identity Detection      (src/inference/flux2_inpaint.py)
  Stage 3: FLUX 2 Contextual Inpainting         (src/inference/flux2_inpaint.py)
  Stage 4: Super-Resolution & Facial Refinement (src/inference/upscaler.py)

Usage (via Slurm):
  python3 src/main.py --task_id ${SLURM_ARRAY_TASK_ID} --job_id ${JOB_ID} ...

Usage (interactive):
  python3 src/main.py --task_id 0 --job_id manual_test
==============================================================================
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Robust repo root path insertion
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.flux2_inpaint import (
    InpaintConfig,
    run_flux2_identity_restoration,
)
from src.inference.upscaler import (
    UpscalerConfig,
    run_upscale_pipeline,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("Pic4Free.Main")


def main():
    parser = argparse.ArgumentParser(
        description="Pic4Free: Unified Multi-Stage Watermark Restoration Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required identifiers
    parser.add_argument("--task_id", type=str, required=True,
                        help="Numeric image index (0-97) or 'base'. Maps to $SLURM_ARRAY_TASK_ID")
    parser.add_argument("--job_id", type=str, default="manual",
                        help="Slurm Job ID for output directory naming")

    # Directories
    parser.add_argument("--input_dir", type=str, default="data/input")
    parser.add_argument("--identity_a_dir", type=str, default="data/identity_person_A")
    parser.add_argument("--identity_b_dir", type=str, default="data/identity_person_B")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Root output dir. Default: data/output/run_<job_id>")

    # FLUX 2 Inpainting / Generation parameters
    parser.add_argument("--flux2_model_id", type=str, default="diffusers/FLUX.2-dev",
                        help="FLUX.2-dev model repo id (e.g. black-forest-labs/FLUX.2-dev)")
    parser.add_argument("--hf_token_file", type=str, default=None,
                        help="Path to local HF token txt file (recommended, not committed)")
    parser.add_argument("--hf_token_env_var", type=str, default="HF_TOKEN",
                        help="Env var name for Hugging Face token fallback")
    parser.add_argument("--remote_text_encoder_url", type=str,
                        default="https://remote-text-encoder-flux-2.huggingface.co/predict",
                        help="Remote endpoint for FLUX.2 text encoder embeddings")
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--ssim_threshold", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu_offload", action="store_true")

    # Upscaler parameters
    parser.add_argument("--scale", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--face_fidelity_weight", type=float, default=0.85)
    parser.add_argument("--enable_face_refinement", action="store_true", default=True)
    parser.add_argument("--no_face_refinement", action="store_true")

    args = parser.parse_args()

    # Resolve output directory
    output_dir = args.output_dir or f"data/output/run_{args.job_id}"
    output_dir = Path(output_dir).resolve()

    # Create output structure
    for subdir in ["restored", "masks", "intermediates", "metrics", "logs"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("Pic4Free Pipeline - Unified Orchestrator")
    logger.info("=" * 72)
    logger.info(f"  Task ID:    {args.task_id}")
    logger.info(f"  Job ID:     {args.job_id}")
    logger.info(f"  Output Dir: {output_dir}")
    logger.info("=" * 72)

    pipeline_start = time.time()
    timings = {}

    try:
        # ==================================================================
        # Stages 1-3: Mask Generation + Identity Detection + Inpainting
        # ==================================================================
        logger.info("[Stage 1-3] Starting SSIM Masking + Identity Routing + FLUX 2 Inpainting...")
        t0 = time.time()

        inpaint_config = InpaintConfig(
            model_id=args.flux2_model_id,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            enable_cpu_offload=args.cpu_offload,
            hf_token_file=args.hf_token_file,
            hf_token_env_var=args.hf_token_env_var,
            remote_text_encoder_url=args.remote_text_encoder_url,
        )

        restored_intermediate, assigned_faces = run_flux2_identity_restoration(
            task_id=args.task_id,
            input_dir=args.input_dir,
            identity_a_dir=args.identity_a_dir,
            identity_b_dir=args.identity_b_dir,
            output_dir=str(output_dir),
            config=inpaint_config,
        )

        timings["stages_1_to_3_sec"] = round(time.time() - t0, 2)
        logger.info(f"[Stage 1-3] Completed in {timings['stages_1_to_3_sec']}s. "
                     f"Faces detected: {len(assigned_faces)}")

        # ==================================================================
        # Stage 4: Super-Resolution & Facial Refinement
        # ==================================================================
        logger.info("[Stage 4] Starting Super-Resolution & Post-Processing...")
        t1 = time.time()

        upscaler_config = UpscalerConfig(
            scale_factor=args.scale,
            face_refinement=args.enable_face_refinement and not args.no_face_refinement,
            face_fidelity_weight=args.face_fidelity_weight,
        )

        final_path = run_upscale_pipeline(
            task_id=args.task_id,
            output_dir=str(output_dir),
            input_dir=args.input_dir,
            config=upscaler_config,
        )

        timings["stage_4_sec"] = round(time.time() - t1, 2)
        logger.info(f"[Stage 4] Completed in {timings['stage_4_sec']}s.")

        # ==================================================================
        # Pipeline Summary
        # ==================================================================
        total_elapsed = round(time.time() - pipeline_start, 2)
        timings["total_sec"] = total_elapsed

        summary = {
            "task_id": args.task_id,
            "job_id": args.job_id,
            "final_output": str(final_path),
            "faces_detected": len(assigned_faces),
            "face_assignments": [
                {
                    "face_id": f.face_id,
                    "identity": f.assigned_identity,
                    "sim_a": round(f.cosine_similarity_a, 4),
                    "sim_b": round(f.cosine_similarity_b, 4),
                }
                for f in assigned_faces
            ],
            "timings": timings,
            "status": "SUCCESS",
        }

        summary_path = output_dir / "metrics" / f"summary_task_{args.task_id}.json"
        with open(str(summary_path), "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("=" * 72)
        logger.info(f"PIPELINE COMPLETE | Task {args.task_id} | {total_elapsed}s total")
        logger.info(f"  -> Final Image: {final_path}")
        logger.info(f"  -> Summary:     {summary_path}")
        logger.info("=" * 72)
        sys.exit(0)

    except Exception as exc:
        total_elapsed = round(time.time() - pipeline_start, 2)
        logger.error(f"PIPELINE FAILED | Task {args.task_id} | {total_elapsed}s | {exc}",
                      exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
