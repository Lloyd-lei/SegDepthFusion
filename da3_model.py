"""
Depth Anything 3 Model Wrapper
Provides a clean interface for DA3 depth estimation model.
"""

import torch
import numpy as np
from typing import Optional, Dict
from PIL import Image
import logging
import cv2

# 尝试导入 DepthAnything3，如果失败则推迟到使用时再尝试（或报错）
try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    DepthAnything3 = None

logger = logging.getLogger(__name__)


class DA3Model:
    """Wrapper for Depth Anything 3 model."""
    
    def __init__(
        self,
        model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE",
        device: str = "cuda",
        normalize_depth: bool = True
    ):
        """
        Initialize DA3 model.
        
        Args:
            model_name: Hugging Face model name or local path
            device: Device to run on ('cuda' or 'cpu')
            normalize_depth: Whether to normalize depth to [0, 1]
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.normalize_depth = normalize_depth
        self.model_name = model_name
        
        logger.info(f"Loading DA3 model '{model_name}' on {self.device}...")
        self._load_model()
        logger.info("DA3 model loaded successfully")
    
    def _load_model(self):
        """Load DA3 model from Hugging Face or local path."""
        global DepthAnything3
        
        if DepthAnything3 is None:
            try:
                # 再次尝试导入，以便捕获具体的 ImportError
                from depth_anything_3.api import DepthAnything3 as DA3
                DepthAnything3 = DA3
            except ImportError as e:
                logger.error(f"Failed to import depth_anything_3: {e}")
                raise RuntimeError("depth_anything_3 library not found or failed to import.") from e

        try:
            # Load model
            self.model = DepthAnything3.from_pretrained(self.model_name).to(device=self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load DA3 model: {e}")
            raise
    
    def estimate_depth(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Estimate depth for the input image.
        
        Args:
            image: Input image (H, W, 3) in RGB format
            mask: Optional binary mask (H, W) to focus depth estimation
            
        Returns:
            Dictionary containing:
                - depth: Depth map (H, W) - lower values = closer
                - confidence: Confidence map (H, W)
                - masked_depth: Depth only in masked region (if mask provided)
        """
        try:
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image)
            else:
                image_pil = image
            
            # Run inference
            with torch.no_grad():
                prediction = self.model.inference([image_pil])
            
            # Extract depth map (N, H, W) -> (H, W)
            depth = prediction.depth[0]  # First image in batch
            
            # Convert to numpy if tensor
            if torch.is_tensor(depth):
                depth = depth.cpu().numpy()
            
            # Normalize if requested
            if self.normalize_depth:
                depth_min, depth_max = depth.min(), depth.max()
                if depth_max > depth_min:
                    depth = (depth - depth_min) / (depth_max - depth_min)
            
            result = {
                "depth": depth,
                "confidence": prediction.conf[0].cpu().numpy() if torch.is_tensor(prediction.conf) else prediction.conf[0],
            }
            
            # Apply mask if provided
            if mask is not None and mask.size > 0:
                # Ensure mask is 2D and matches depth dimensions
                if mask.ndim > 2:
                    mask = mask.squeeze()
                
                # Resize mask if dimensions don't match
                if mask.shape != depth.shape:
                    mask = cv2.resize(mask.astype(np.uint8), 
                                    (depth.shape[1], depth.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                
                masked_depth = depth.copy()
                masked_depth[mask == 0] = np.nan  # Set non-masked areas to NaN
                result["masked_depth"] = masked_depth
                
                # Calculate mean depth in masked region
                valid_depth = depth[mask > 0]
                if len(valid_depth) > 0:
                    result["mean_depth"] = float(valid_depth.mean())
                    result["min_depth"] = float(valid_depth.min())
                    result["max_depth"] = float(valid_depth.max())
                else:
                    result["mean_depth"] = 0.0
                    result["min_depth"] = 0.0
                    result["max_depth"] = 0.0
            
            logger.debug(f"Depth estimation completed")
            return result
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            # Return empty results on failure
            h, w = image.shape[:2] if isinstance(image, np.ndarray) else (480, 640)
            return {
                "depth": np.zeros((h, w)),
                "confidence": np.zeros((h, w)),
                "mean_depth": 0.0
            }
    
    def get_depth_stats(self, depth: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate statistics for depth map.
        
        Args:
            depth: Depth map (H, W)
            mask: Optional binary mask to focus on specific region
            
        Returns:
            Dictionary with depth statistics
        """
        if mask is not None and mask.size > 0:
            valid_depth = depth[mask > 0]
        else:
            valid_depth = depth.flatten()
        
        if len(valid_depth) == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0
            }
        
        return {
            "mean": float(np.mean(valid_depth)),
            "median": float(np.median(valid_depth)),
            "min": float(np.min(valid_depth)),
            "max": float(np.max(valid_depth)),
            "std": float(np.std(valid_depth))
        }
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()
        logger.info("DA3 model cleaned up")
