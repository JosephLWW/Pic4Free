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
        "El paquete 'ben2' no está instalado. Instálalo con "
        "'pip install -r requirements.txt' (incluye BEN2 desde GitHub)."
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
        logger.info(f"Inicializando BENMaskGenerator con modelo: {self.model_id} en {self.device}")
        
        try:
            self.model = BEN_Base.from_pretrained(
                self.model_id
            ).to(self.device)
            self.model.eval()
            logger.info("Modelo BEN cargado exitosamente.")
        except Exception as e:
            logger.error(f"Error al cargar el modelo BEN: {e}")
            raise
    
    @torch.no_grad()
    def generate_mask(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Inferencia Zero-Shot para extraer el canal alfa.
        
        Args:
            image: Imagen original (watermarked).
            
        Returns:
            torch.Tensor: Máscara alfa (0.0 a 1.0 flotante).
        """
        logger.info("Generando máscara con BEN + CGM...")
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")
            
        original_size = image.size # (width, height)
        
        try:
            # BEN2 inference devuelve directamente un objeto PIL.Image
            outputs = self.model.inference(image)
            
            # Soporte nativo para manejo de un solo output o un batch
            if isinstance(outputs, list):
                result_image = outputs[0]
            else:
                result_image = outputs
                
            if isinstance(result_image, Image.Image):
                # Extraer el canal alfa si es RGBA, de lo contrario fallback a escala de grises
                if result_image.mode == 'RGBA':
                    mask_pil = result_image.split()[-1]
                else:
                    mask_pil = result_image.convert('L')
                
                # Convertir a tensor normalizado [0, 1] y mover al dispositivo activo
                mask = transforms.ToTensor()(mask_pil).to(self.device)
            elif isinstance(result_image, torch.Tensor):
                mask = result_image
            else:
                raise TypeError(f"Formato de salida inesperado de BEN2: {type(result_image)}")
            
            # Redimensionar al tamaño original usando interpolación bilineal
            if mask.dim() == 3: mask = mask.unsqueeze(0)
            mask = torch.nn.functional.interpolate(
                mask, 
                size=(original_size[1], original_size[0]), 
                mode='bilinear', 
                align_corners=False
            )
            return mask.clamp(0, 1)
            
        except Exception as e:
            logger.error(f"Error durante la inferencia BEN: {e}")
            raise