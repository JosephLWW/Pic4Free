#!/usr/bin/env python3
"""
==============================================================================
Pic4Free - Stage 1: Dynamic Differential Mask Pipeline (SSIM + Morphology)
==============================================================================
Industrial pipeline to generate binary and soft alpha masks isolating complex,
semi-opaque, and opaque watermarks by comparing degraded high-res images
against clean low-res thumbnail priors using Structural Similarity (SSIM).

Key Technical Highlights:
  - Strict Regex Dataset Parsing supporting indexed files:
      * 'watermarked (X).jpg' / 'thumb-400 (X).jpg'
      * 'watermarked.jpg'     / 'thumb-400.jpg' (base index / index 0 / fallback)
  - Lanczos4 High-Precision Geometric Upscaling of thumbnail priors.
  - Multi-channel CIELAB Color/Luminance Alignment (Reinhard Normalization).
  - OpenCV-Native Vectorized Structural Similarity (SSIM) & DSSIM Computation.
  - Adaptive Thresholding & Elliptical Morphological Dilation (5x5 / 7x7).
  - Boundary Feathering for artifact-free inpainting blend boundaries.
==============================================================================
"""

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

# Configure industrial logging format
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Pic4Free.MaskPipeline")


@dataclass
class MaskPipelineConfig:
    """Configuration dataclass for the SSIM differential mask generation."""
    window_size: int = 11
    sigma: float = 1.5
    k1: float = 0.01
    k2: float = 0.03
    ssim_threshold: float = 0.60
    adaptive_method: str = "otsu"  # Options: 'otsu', 'adaptive_gaussian', 'fixed', 'hybrid'
    morph_kernel_size: int = 7
    morph_iterations: int = 2
    feather_sigma: float = 1.5
    enable_color_transfer: bool = True
    save_intermediates: bool = True


class DatasetResolver:
    """
    Robust resolver for extracting, matching, and verifying image pairs
    from 'data/input/' using compiled Regular Expressions.
    """

    # Regex patterns for indexed files and base unindexed files
    PATTERN_WATERMARKED_INDEXED = re.compile(r"^watermarked\s*\(\s*(?P<id>\d+)\s*\)\.(?:jpe?g|png|webp)$", re.IGNORECASE)
    PATTERN_WATERMARKED_BASE = re.compile(r"^watermarked\.(?:jpe?g|png|webp)$", re.IGNORECASE)

    PATTERN_THUMB_INDEXED = re.compile(r"^thumb-400\s*\(\s*(?P<id>\d+)\s*\)\.(?:jpe?g|png|webp)$", re.IGNORECASE)
    PATTERN_THUMB_BASE = re.compile(r"^thumb-400\.(?:jpe?g|png|webp)$", re.IGNORECASE)

    @classmethod
    def resolve_pair(cls, input_dir: Union[str, Path], task_id: Union[int, str]) -> Tuple[Path, Path]:
        """
        Locates the exact pair (watermarked, thumbnail) corresponding to a numeric task_id.

        Args:
            input_dir: Path to directory containing input images (e.g. data/input).
            task_id: Numeric index or string representation (e.g., 0, 1, 42, 'base').

        Returns:
            Tuple of (watermarked_path, thumbnail_path).

        Raises:
            FileNotFoundError: If matching pair cannot be located.
        """
        input_dir = Path(input_dir).resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        all_files = os.listdir(input_dir)
        str_id = str(task_id).strip()

        # Handle special / base cases: index 0 mapping or explicit base naming
        wm_path: Optional[Path] = None
        th_path: Optional[Path] = None

        # Strategy A: Exact numeric match in parentheses e.g. "watermarked (42).jpg"
        for fname in all_files:
            m_wm = cls.PATTERN_WATERMARKED_INDEXED.match(fname)
            if m_wm and m_wm.group("id") == str_id:
                wm_path = input_dir / fname

            m_th = cls.PATTERN_THUMB_INDEXED.match(fname)
            if m_th and m_th.group("id") == str_id:
                th_path = input_dir / fname

        # Strategy B: If task_id is 0 or 'base' or -1 and exact indexed match not found, check base files
        if wm_path is None and str_id in {"0", "base", "-1", "none"}:
            for fname in all_files:
                if cls.PATTERN_WATERMARKED_BASE.match(fname):
                    wm_path = input_dir / fname
                if cls.PATTERN_THUMB_BASE.match(fname):
                    th_path = input_dir / fname

        # Strategy C: Sequential sorting index fallback
        if wm_path is None or th_path is None:
            indexed_wms = []
            indexed_ths = {}

            for fname in all_files:
                m_wm = cls.PATTERN_WATERMARKED_INDEXED.match(fname)
                if m_wm:
                    indexed_wms.append((int(m_wm.group("id")), fname))
                elif cls.PATTERN_WATERMARKED_BASE.match(fname):
                    indexed_wms.append((-1, fname))

                m_th = cls.PATTERN_THUMB_INDEXED.match(fname)
                if m_th:
                    indexed_ths[int(m_th.group("id"))] = fname
                elif cls.PATTERN_THUMB_BASE.match(fname):
                    indexed_ths[-1] = fname

            indexed_wms.sort(key=lambda x: x[0])
            try:
                numeric_idx = int(str_id)
                if 0 <= numeric_idx < len(indexed_wms):
                    target_id, target_wm_file = indexed_wms[numeric_idx]
                    wm_path = input_dir / target_wm_file
                    if target_id in indexed_ths:
                        th_path = input_dir / indexed_ths[target_id]
            except ValueError:
                pass

        if wm_path is None or not wm_path.exists():
            raise FileNotFoundError(f"Could not locate watermarked image for task_id '{task_id}' in {input_dir}")
        if th_path is None or not th_path.exists():
            raise FileNotFoundError(f"Could not locate corresponding thumbnail for task_id '{task_id}' in {input_dir}")

        logger.info(f"[Resolver] Task ID [{task_id}] resolved to:")
        logger.info(f"  -> Watermarked: {wm_path.name}")
        logger.info(f"  -> Thumbnail:   {th_path.name}")

        return wm_path, th_path


def align_color_lab(source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    """
    Performs Reinhard chromatic and luminance normalization in CIELAB space
    to match the color distribution of the upscaled thumbnail to the target image.

    Args:
        source_bgr: Image to be color-adjusted (upscaled thumbnail).
        target_bgr: Reference image providing color distribution (watermarked image).

    Returns:
        Color-aligned BGR image.
    """
    src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_mean, src_std = np.mean(src_lab, axis=(0, 1), keepdims=True), np.std(src_lab, axis=(0, 1), keepdims=True)
    tgt_mean, tgt_std = np.mean(tgt_lab, axis=(0, 1), keepdims=True), np.std(tgt_lab, axis=(0, 1), keepdims=True)

    # Avoid division by zero
    src_std = np.maximum(src_std, 1e-6)

    # Standardize and rescale
    aligned_lab = ((src_lab - src_mean) / src_std) * tgt_std + tgt_mean
    aligned_lab = np.clip(aligned_lab, 0, 255).astype(np.uint8)

    return cv2.cvtColor(aligned_lab, cv2.COLOR_LAB2BGR)


def compute_ssim_map_opencv(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    dynamic_range: float = 255.0,
) -> np.ndarray:
    """
    Computes exact pixel-wise Structural Similarity Index (SSIM) map using OpenCV.

    Formula:
        SSIM(x, y) = [ (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2) ] /
                     [ (mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2) ]

    Args:
        img1: Grayscale float32 image [0, 255].
        img2: Grayscale float32 image [0, 255].
        window_size: Gaussian kernel window dimension (must be odd).
        sigma: Standard deviation of Gaussian weighting function.
        k1, k2: SSIM stability constants.
        dynamic_range: Dynamic range of pixel values (255.0).

    Returns:
        Pixel-wise SSIM map with values in range [-1.0, 1.0].
    """
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    # Local means
    mu1 = cv2.GaussianBlur(img1, (window_size, window_size), sigma)
    mu2 = cv2.GaussianBlur(img2, (window_size, window_size), sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # Local variances and covariances
    sigma1_sq = cv2.GaussianBlur(img1 * img1, (window_size, window_size), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (window_size, window_size), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (window_size, window_size), sigma) - mu1_mu2

    # SSIM numerator & denominator
    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / np.maximum(denominator, 1e-7)
    return np.clip(ssim_map, -1.0, 1.0)


class MaskPipeline:
    """
    Industrial pipeline for generating watermarking masks from degraded images
    and corresponding clean thumbnail priors.
    """

    def __init__(self, config: Optional[MaskPipelineConfig] = None):
        self.config = config or MaskPipelineConfig()

    def process(
        self,
        watermarked_img_bgr: np.ndarray,
        thumbnail_img_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes the full mask generation workflow.

        Args:
            watermarked_img_bgr: Original high-res watermarked image (H, W, 3).
            thumbnail_img_bgr: Clean low-res thumbnail prior (H_t, W_t, 3).

        Returns:
            Tuple containing:
                1. binary_mask (uint8: 0 or 255)
                2. soft_alpha_mask (float32: 0.0 to 1.0)
                3. upscaled_thumbnail_bgr (H, W, 3)
                4. ssim_disparity_map (uint8: 0 to 255)
        """
        target_h, target_w = watermarked_img_bgr.shape[:2]

        # 1. High-precision Lanczos4 scaling of the thumbnail
        upscaled_thumb = cv2.resize(
            thumbnail_img_bgr,
            (target_w, target_h),
            interpolation=cv2.INTER_LANCZOS4,
        )

        # 2. Chromatic and luminance alignment in CIELAB space
        if self.config.enable_color_transfer:
            thumb_aligned = align_color_lab(upscaled_thumb, watermarked_img_bgr)
        else:
            thumb_aligned = upscaled_thumb

        # 3. Convert to Grayscale float32 for SSIM evaluation
        gray_wm = cv2.cvtColor(watermarked_img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_th = cv2.cvtColor(thumb_aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 4. Compute pixel-wise SSIM
        ssim_map = compute_ssim_map_opencv(
            gray_wm,
            gray_th,
            window_size=self.config.window_size,
            sigma=self.config.sigma,
            k1=self.config.k1,
            k2=self.config.k2,
        )

        # Disparity (DSSIM): 0 where identical (SSIM=1), 255 where maximum divergence
        dssim = ((1.0 - ssim_map) * 0.5 * 255.0).astype(np.float32)
        dssim_uint8 = np.clip(dssim, 0, 255).astype(np.uint8)

        # 5. Compute Multiscale High-Frequency Gradient Difference
        grad_x_wm = cv2.Sobel(gray_wm, cv2.CV_32F, 1, 0, ksize=3)
        grad_y_wm = cv2.Sobel(gray_wm, cv2.CV_32F, 0, 1, ksize=3)
        mag_wm = cv2.magnitude(grad_x_wm, grad_y_wm)

        grad_x_th = cv2.Sobel(gray_th, cv2.CV_32F, 1, 0, ksize=3)
        grad_y_th = cv2.Sobel(gray_th, cv2.CV_32F, 0, 1, ksize=3)
        mag_th = cv2.magnitude(grad_x_th, grad_y_th)

        grad_diff = np.maximum(mag_wm - mag_th, 0.0)
        grad_diff_norm = np.clip((grad_diff / (np.percentile(grad_diff, 99.5) + 1e-5)) * 255.0, 0, 255).astype(np.uint8)

        # Fused disparity map
        fused_disparity = cv2.addWeighted(dssim_uint8, 0.7, grad_diff_norm, 0.3, 0)

        # 6. Adaptive Thresholding
        if self.config.adaptive_method == "otsu":
            _, raw_mask = cv2.threshold(fused_disparity, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.config.adaptive_method == "hybrid":
            _, otsu_m = cv2.threshold(fused_disparity, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            fixed_thresh = int((1.0 - self.config.ssim_threshold) * 255.0)
            _, fixed_m = cv2.threshold(dssim_uint8, fixed_thresh, 255, cv2.THRESH_BINARY)
            raw_mask = cv2.bitwise_or(otsu_m, fixed_m)
        else:
            fixed_thresh = int((1.0 - self.config.ssim_threshold) * 255.0)
            _, raw_mask = cv2.threshold(dssim_uint8, fixed_thresh, 255, cv2.THRESH_BINARY)

        # 7. Morphological Cleanup and Dilation
        k_size = self.config.morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

        # Close small holes inside watermark characters
        closed_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Dilate to guarantee 100% boundary coverage of watermark stroke falloffs
        dilated_mask = cv2.dilate(closed_mask, kernel, iterations=self.config.morph_iterations)

        # 8. Boundary Feathering for Soft Alpha Mask
        feather_ksize = int(self.config.feather_sigma * 6) | 1  # ensure odd
        feathered = cv2.GaussianBlur(
            dilated_mask.astype(np.float32) / 255.0,
            (feather_ksize, feather_ksize),
            self.config.feather_sigma,
        )
        soft_alpha_mask = np.clip(feathered, 0.0, 1.0)

        return dilated_mask, soft_alpha_mask, upscaled_thumb, dssim_uint8


def generate_watermark_mask(
    watermarked_path: Union[str, Path],
    thumbnail_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[MaskPipelineConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, Path]:
    """
    High-level API to process a single pair of images from file paths.

    Args:
        watermarked_path: Path to watermarked image file.
        thumbnail_path: Path to clean thumbnail image file.
        output_dir: Directory where generated masks and intermediates will be saved.
        config: Custom MaskPipelineConfig options.

    Returns:
        Tuple of (binary_mask, soft_alpha_mask, saved_mask_path).
    """
    config = config or MaskPipelineConfig()
    wm_path = Path(watermarked_path)
    th_path = Path(thumbnail_path)

    if not wm_path.exists():
        raise FileNotFoundError(f"Watermarked image not found: {wm_path}")
    if not th_path.exists():
        raise FileNotFoundError(f"Thumbnail image not found: {th_path}")

    img_wm = cv2.imread(str(wm_path), cv2.IMREAD_COLOR)
    img_th = cv2.imread(str(th_path), cv2.IMREAD_COLOR)

    if img_wm is None:
        raise ValueError(f"Failed to read image at: {wm_path}")
    if img_th is None:
        raise ValueError(f"Failed to read image at: {th_path}")

    pipeline = MaskPipeline(config=config)
    binary_mask, soft_alpha, upscaled_th, dssim_map = pipeline.process(img_wm, img_th)

    saved_mask_path = Path()
    if output_dir is not None:
        out_dir = Path(output_dir).resolve()
        masks_dir = out_dir / "masks" if "masks" not in out_dir.name else out_dir
        inter_dir = out_dir / "intermediates"

        masks_dir.mkdir(parents=True, exist_ok=True)
        inter_dir.mkdir(parents=True, exist_ok=True)

        base_stem = wm_path.stem
        saved_mask_path = masks_dir / f"{base_stem}_mask.png"
        cv2.imwrite(str(saved_mask_path), binary_mask)

        if config.save_intermediates:
            cv2.imwrite(str(inter_dir / f"{base_stem}_dssim.png"), dssim_map)
            cv2.imwrite(str(inter_dir / f"{base_stem}_thumb_upscaled.png"), upscaled_th)
            alpha_vis = (soft_alpha * 255.0).astype(np.uint8)
            cv2.imwrite(str(inter_dir / f"{base_stem}_soft_alpha.png"), alpha_vis)

        logger.info(f"[MaskPipeline] Mask successfully saved to: {saved_mask_path}")

    return binary_mask, soft_alpha, saved_mask_path


def main():
    parser = argparse.ArgumentParser(
        description="Pic4Free Stage 1: SSIM Differential Watermark Mask Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image_id", "--task_id", "-i", type=str, required=True,
                        help="Numeric task ID or index (e.g. 0, 1, 96, 'base')")
    parser.add_argument("--input_dir", type=str, default="data/input",
                        help="Directory containing watermarked and thumbnail images")
    parser.add_argument("--output_dir", type=str, default="data/output",
                        help="Root output directory where masks/ and intermediates/ will be written")
    parser.add_argument("--ssim_threshold", type=float, default=0.60,
                        help="Structural similarity threshold for watermark detection")
    parser.add_argument("--morph_kernel", type=int, default=7,
                        help="Morphological kernel size for closing & dilation (5 or 7)")
    parser.add_argument("--dilation_iter", type=int, default=2,
                        help="Number of morphological dilation iterations")
    parser.add_argument("--adaptive_method", type=str, default="hybrid",
                        choices=["otsu", "hybrid", "fixed"],
                        help="Binarization thresholding method")
    parser.add_argument("--no_intermediates", action="store_true",
                        help="Disable saving intermediate disparity maps and upscaled thumbnails")

    args = parser.parse_args()

    try:
        wm_path, th_path = DatasetResolver.resolve_pair(args.input_dir, args.image_id)

        config = MaskPipelineConfig(
            ssim_threshold=args.ssim_threshold,
            morph_kernel_size=args.morph_kernel,
            morph_iterations=args.dilation_iter,
            adaptive_method=args.adaptive_method,
            save_intermediates=not args.no_intermediates,
        )

        binary_mask, soft_alpha, saved_path = generate_watermark_mask(
            watermarked_path=wm_path,
            thumbnail_path=th_path,
            output_dir=args.output_dir,
            config=config,
        )

        white_pixels_pct = (np.mean(binary_mask > 0) * 100.0)
        logger.info(f"Execution complete. Mask Coverage: {white_pixels_pct:.2f}% of total pixels.")
        sys.exit(0)

    except Exception as exc:
        logger.error(f"Fatal error during mask generation for task {args.image_id}: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
