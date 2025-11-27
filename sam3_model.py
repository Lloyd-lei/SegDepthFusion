"""
SAM3 Model Wrapper
Provides a clean interface for SAM3 segmentation model.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class SAM3Model:
    """Wrapper for Segment Anything Model 3 (SAM3)."""
    
    def __init__(
        self,
        model_name: str = "sam3_image",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5
    ):
        """
        Initialize SAM3 model.
        
        Args:
            model_name: Model architecture name
            checkpoint_path: Path to checkpoint, None for auto-download
            device: Device to run on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for masks
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        
        logger.info(f"Loading SAM3 model on {self.device}...")
        self._load_model()
        logger.info("SAM3 model loaded successfully")
        
    def _load_model(self):
        """Load SAM3 model and processor."""
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            # Build model
            self.model = build_sam3_image_model()
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Initialize processor
            self.processor = Sam3Processor(self.model)
            
        except Exception as e:
            logger.error(f"Failed to load SAM3 model: {e}")
            raise
    
    def segment(
        self,
        image: np.ndarray,
        prompt: str
    ) -> Dict[str, np.ndarray]:
        """
        Segment objects in image using text prompt.
        
        Args:
            image: Input image (H, W, 3) in RGB format
            prompt: Text prompt describing target object
            
        Returns:
            Dictionary containing:
                - masks: Binary masks (N, H, W)
                - boxes: Bounding boxes (N, 4) in [x1, y1, x2, y2]
                - scores: Confidence scores (N,)
        """
        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image)
            else:
                image_pil = image
            
            # Set image in processor
            with torch.no_grad():
                inference_state = self.processor.set_image(image_pil)
                
                # Run segmentation with text prompt
                output = self.processor.set_text_prompt(
                    state=inference_state,
                    prompt=prompt
                )
            
            # Extract results
            masks = output["masks"]  # (N, H, W)
            boxes = output["boxes"]  # (N, 4)
            scores = output["scores"]  # (N,)
            
            # Filter by confidence threshold
            valid_idx = scores >= self.confidence_threshold
            
            result = {
                "masks": masks[valid_idx].cpu().numpy() if torch.is_tensor(masks) else masks[valid_idx],
                "boxes": boxes[valid_idx].cpu().numpy() if torch.is_tensor(boxes) else boxes[valid_idx],
                "scores": scores[valid_idx].cpu().numpy() if torch.is_tensor(scores) else scores[valid_idx],
            }
            
            logger.debug(f"Segmented {len(result['masks'])} objects for prompt '{prompt}'")
            return result
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            # Return empty results on failure
            return {
                "masks": np.array([]),
                "boxes": np.array([]),
                "scores": np.array([])
            }
    
    def get_largest_mask(self, segmentation_result: Dict) -> Optional[np.ndarray]:
        """
        Get the largest mask from segmentation results.
        
        Args:
            segmentation_result: Output from segment()
            
        Returns:
            Largest mask as (H, W) binary array, or None if no masks
        """
        masks = segmentation_result["masks"]
        
        if len(masks) == 0:
            return None
        
        # Calculate area for each mask
        areas = [mask.sum() for mask in masks]
        largest_idx = np.argmax(areas)
        
        return masks[largest_idx]
    
    def get_mask_info(self, mask: np.ndarray) -> Dict:
        """
        Get information about a mask.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            Dictionary with mask statistics
        """
        if mask is None or mask.size == 0:
            return {
                "area": 0,
                "area_ratio": 0.0,
                "centroid": (0, 0),
                "bbox": [0, 0, 0, 0]
            }
        
        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask.squeeze()
        if mask.ndim == 1:
            # Invalid mask shape, return empty info
            return {
                "area": 0,
                "area_ratio": 0.0,
                "centroid": (0, 0),
                "bbox": [0, 0, 0, 0]
            }
        
        total_pixels = mask.shape[0] * mask.shape[1]
        area = int(mask.sum())
        area_ratio = area / total_pixels
        
        # Calculate centroid
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) > 0:
            centroid_y = int(y_coords.mean())
            centroid_x = int(x_coords.mean())
            
            # Calculate bounding box
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()
            bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
        else:
            centroid_y, centroid_x = 0, 0
            bbox = [0, 0, 0, 0]
        
        return {
            "area": area,
            "area_ratio": area_ratio,
            "centroid": (centroid_x, centroid_y),
            "bbox": bbox
        }
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        torch.cuda.empty_cache()
        logger.info("SAM3 model cleaned up")