import torch
import logging
from PIL import Image
from typing import Union
import numpy as np
from torchvision import transforms
try:
    from ben2 import BEN_Base
except ImportError as e:
    raise ImportError(
        "The package 'ben2' no this installed. Instalalo with "
        "'pip install -r requirements.txt' (includes BEN2 from GitHub)."
    ) from e

logger = logging.getLogger("Pic4Free.MaskGenerator")

class BENMaskGenerator:
    """
    State-of-the-art Background Erase Network (BEN) with Confidence-Guided Matting.
    Replaces the legacy SSIM + Otsu method for precise sub-pixel alpha matting.
    """
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        logger.info(f"Initializing BENMaskGenerator with model: {self.model_id} in {self.device}")
        
        try:
            self.model = BEN_Base.from_pretrained(
                self.model_id
            ).to(self.device)
            self.model.eval()
            logger.info("BEN model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading BEN model: {e}")
            raise
    
    @torch.no_grad()
    def generate_mask(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Zero-shot inference to extract the alpha channel.
        
        Args:
            image: Imagen original (watermarked).
            
        Returns:
            torch.Tensor: Alpha mask (floating-point 0.0 to 1.0).
        """
        logger.info("Generating mask with BEN + CGM...")
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")
            
        original_size = image.size # (width, height)
        
        try:
            # BEN2 inference returns a PIL.Image object directly
            outputs = self.model.inference(image)
            
            # Native support for handling either a single output or a batch
            if isinstance(outputs, list):
                result_image = outputs[0]
            else:
                result_image = outputs
                
            if isinstance(result_image, Image.Image):
                # Extract the alpha channel if RGBA, otherwise fall back to grayscale
                if result_image.mode == 'RGBA':
                    mask_pil = result_image.split()[-1]
                else:
                    mask_pil = result_image.convert('L')
                
                # Convert to a normalized tensor [0, 1] and move to the active device
                mask = transforms.ToTensor()(mask_pil).to(self.device)
            elif isinstance(result_image, torch.Tensor):
                mask = result_image
            else:
                raise TypeError(f"Unexpected BEN2 output shapet: {type(result_image)}")
            
            # Resize to the original size using bilinear interpolation
            if mask.dim() == 3: mask = mask.unsqueeze(0)
            mask = torch.nn.functional.interpolate(
                mask, 
                size=(original_size[1], original_size[0]), 
                mode='bilinear', 
                align_corners=False
            )
            return mask.clamp(0, 1)
            
        except Exception as e:
            logger.error(f"Error during BEN inference: {e}")
            raise