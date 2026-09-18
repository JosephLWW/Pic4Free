#!/usr/bin/env python3
"""
==============================================================================
Pic4Free - main.py: Unified Pipeline Orchestrator (SOTA Architecture)
==============================================================================
Orchestrates BEN, PuLID-FLUX (EVA-CLIP + ArcFace), FLUX.1-dev + PuLID, SUPIR + RestoreFormer++.
"""

import sys
import logging
from pathlib import Path

# Robust repo root path insertion
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pic4free.core.config import load_config
from pic4free.core.pipeline import Pic4FreeRestorationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("Pic4Free.Main")

def main():
    # Load configuration from the CLI via OmegaConf
    cli_args = sys.argv[1:]
    config = load_config(cli_args)
    
    logger.info("=" * 72)
    logger.info("Pic4Free Pipeline - Unified Orchestrator SOTA")
    logger.info("=" * 72)
    logger.info(f"  Task ID:    {config.task_id}")
    logger.info(f"  Job ID:     {config.job_id}")
    logger.info(f"  Output Dir: {config.paths.output_dir}")
    logger.info("=" * 72)

    try:
        pipeline = Pic4FreeRestorationPipeline(config)
        final_path = pipeline.run()
        
        logger.info("=" * 72)
        logger.info(f"PIPELINE COMPLETE | Task {config.task_id}")
        logger.info(f"  -> Final Image: {final_path}")
        logger.info("=" * 72)
        sys.exit(0)
    except Exception as exc:
        logger.error(f"PIPELINE FAILED | Task {config.task_id} | {exc}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
