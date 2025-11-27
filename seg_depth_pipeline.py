"""
Seg-Depth Pipeline
Integrates SAM3 segmentation and DA3 depth estimation.
"""

import numpy as np
from typing import Dict, Optional, List
import logging
from collections import deque

logger = logging.getLogger(__name__)


class SegDepthPipeline:
    """
    Main pipeline integrating segmentation and depth estimation.
    Designed for fruit stalk tracking with reward calculation.
    """
    
    def __init__(
        self,
        sam3_model,
        da3_model,
        target_prompt: str = "orange",
        prompts: Optional[List[str]] = None,
        area_weight: float = 0.5,
        depth_weight: float = 0.5,
        history_size: int = 30
    ):
        """
        Initialize pipeline.
        
        Args:
            sam3_model: Initialized SAM3Model instance
            da3_model: Initialized DA3Model instance
            target_prompt: Primary prompt for reward calculation
            prompts: List of all prompts to try
            area_weight: Weight for area component in reward [0, 1]
            depth_weight: Weight for depth component in reward [0, 1]
            history_size: Number of frames to keep for reward smoothing
        """
        self.sam3 = sam3_model
        self.da3 = da3_model
        self.target_prompt = target_prompt
        self.prompts = prompts or [target_prompt]
        
        # Reward weights
        self.area_weight = area_weight
        self.depth_weight = depth_weight
        
        # History tracking
        self.history_size = history_size
        self.area_history = deque(maxlen=history_size)
        self.depth_history = deque(maxlen=history_size)
        self.reward_history = deque(maxlen=history_size)
        
        # Statistics
        self.frame_count = 0
        self.total_objects_detected = 0
        
        logger.info(f"Pipeline initialized with target: '{target_prompt}'")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: Input frame (H, W, 3) in RGB format
            
        Returns:
            Dictionary containing all processing results
        """
        self.frame_count += 1
        
        result = {
            "frame_id": self.frame_count,
            "segmentation": None,
            "depth": None,
            "reward": 0.0,
            "mask": None,
            "mask_info": None,
            "depth_info": None,
            "status": "processing"
        }
        
        try:
            # Step 1: Segmentation with SAM3
            seg_result = self._segment_objects(frame)
            result["segmentation"] = seg_result
            
            # Get the best mask for target object
            target_mask = self._get_target_mask(seg_result)
            result["mask"] = target_mask
            
            if target_mask is None or target_mask.sum() == 0:
                result["status"] = "no_object_detected"
                logger.debug(f"Frame {self.frame_count}: No target object detected")
                return result
            
            # Get mask information
            mask_info = self.sam3.get_mask_info(target_mask)
            result["mask_info"] = mask_info
            
            # Step 2: Depth estimation with DA3
            depth_result = self.da3.estimate_depth(frame, target_mask)
            result["depth"] = depth_result
            
            # Get depth information
            depth_info = {
                "mean_depth": depth_result.get("mean_depth", 0.0),
                "min_depth": depth_result.get("min_depth", 0.0),
                "max_depth": depth_result.get("max_depth", 0.0),
            }
            result["depth_info"] = depth_info
            
            # Step 3: Calculate reward
            reward = self._calculate_reward(mask_info, depth_info)
            result["reward"] = reward
            
            # Update history
            self._update_history(mask_info["area_ratio"], depth_info["mean_depth"], reward)
            
            # Add trends
            result["trends"] = self._get_trends()
            
            result["status"] = "success"
            self.total_objects_detected += 1
            
            logger.debug(
                f"Frame {self.frame_count}: "
                f"Area={mask_info['area_ratio']:.3f}, "
                f"Depth={depth_info['mean_depth']:.3f}, "
                f"Reward={reward:.3f}"
            )
            
        except Exception as e:
            logger.error(f"Frame {self.frame_count} processing failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)
        
        return result
    
    def _segment_objects(self, frame: np.ndarray) -> Dict:
        """
        Segment objects using all prompts.
        
        Returns:
            Dictionary with segmentation results for each prompt
        """
        all_results = {}
        
        for prompt in self.prompts:
            seg_result = self.sam3.segment(frame, prompt)
            if len(seg_result["masks"]) > 0:
                all_results[prompt] = seg_result
        
        return all_results
    
    def _get_target_mask(self, seg_results: Dict) -> Optional[np.ndarray]:
        """
        Get the best mask for the target object.
        
        Args:
            seg_results: Segmentation results from all prompts
            
        Returns:
            Best mask for target prompt, or None
        """
        # Try target prompt first
        if self.target_prompt in seg_results:
            result = seg_results[self.target_prompt]
            return self.sam3.get_largest_mask(result)
        
        # Fallback: use any available prompt
        for prompt, result in seg_results.items():
            mask = self.sam3.get_largest_mask(result)
            if mask is not None:
                logger.debug(f"Using fallback prompt: {prompt}")
                return mask
        
        return None
    
    def _calculate_reward(self, mask_info: Dict, depth_info: Dict) -> float:
        """
        Calculate reward based on mask area and depth.
        
        Reward increases when:
        - Mask area increases (object getting closer/larger in view)
        - Depth decreases (object getting closer)
        
        Args:
            mask_info: Information about the segmentation mask
            depth_info: Information about depth in masked region
            
        Returns:
            Reward value (higher = better)
        """
        # Area component: normalized ratio (0 to 1)
        area_ratio = mask_info["area_ratio"]
        
        # Depth component: inverse depth (closer = higher reward)
        # Since depth is normalized [0, 1], closer objects have lower depth values
        # We want higher reward for closer objects, so use (1 - depth)
        mean_depth = depth_info["mean_depth"]
        closeness = 1.0 - mean_depth  # Convert to closeness score
        
        # Combine components
        reward = (self.area_weight * area_ratio + 
                 self.depth_weight * closeness)
        
        return reward
    
    def _update_history(self, area_ratio: float, mean_depth: float, reward: float):
        """Update tracking history."""
        self.area_history.append(area_ratio)
        self.depth_history.append(mean_depth)
        self.reward_history.append(reward)
    
    def _get_trends(self) -> Dict:
        """
        Calculate trends from history.
        
        Returns:
            Dictionary with trend information
        """
        if len(self.reward_history) < 2:
            return {
                "area_trend": 0.0,
                "depth_trend": 0.0,
                "reward_trend": 0.0,
                "reward_moving_avg": 0.0
            }
        
        # Calculate simple differences (last - second_to_last)
        area_trend = self.area_history[-1] - self.area_history[-2]
        depth_trend = self.depth_history[-1] - self.depth_history[-2]
        reward_trend = self.reward_history[-1] - self.reward_history[-2]
        
        # Calculate moving average
        reward_ma = np.mean(list(self.reward_history))
        
        return {
            "area_trend": area_trend,
            "depth_trend": depth_trend,
            "reward_trend": reward_trend,
            "reward_moving_avg": reward_ma
        }
    
    def get_statistics(self) -> Dict:
        """
        Get overall pipeline statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_frames": self.frame_count,
            "objects_detected": self.total_objects_detected,
            "detection_rate": self.total_objects_detected / max(self.frame_count, 1),
            "history_size": len(self.reward_history),
            "current_reward": self.reward_history[-1] if len(self.reward_history) > 0 else 0.0,
            "avg_reward": np.mean(list(self.reward_history)) if len(self.reward_history) > 0 else 0.0,
            "max_reward": np.max(list(self.reward_history)) if len(self.reward_history) > 0 else 0.0,
        }
    
    def reset_history(self):
        """Reset tracking history."""
        self.area_history.clear()
        self.depth_history.clear()
        self.reward_history.clear()
        self.frame_count = 0
        self.total_objects_detected = 0
        logger.info("Pipeline history reset")