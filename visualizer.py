"""
Visualization Module
Real-time visualization of segmentation and depth results.
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    """Real-time visualizer for seg-depth pipeline results."""
    
    def __init__(
        self,
        window_width: int = 1600,
        window_height: int = 900,
        depth_colormap: str = "turbo",
        show_original: bool = True,
        show_segmentation: bool = True,
        show_depth: bool = True,
        show_metrics: bool = True
    ):
        """
        Initialize visualizer.
        
        Args:
            window_width: Main window width
            window_height: Main window height
            depth_colormap: OpenCV colormap for depth visualization
            show_original: Whether to show original frame
            show_segmentation: Whether to show segmentation overlay
            show_depth: Whether to show depth map
            show_metrics: Whether to show metrics panel
        """
        self.window_width = window_width
        self.window_height = window_height
        self.show_original = show_original
        self.show_segmentation = show_segmentation
        self.show_depth = show_depth
        self.show_metrics = show_metrics
        
        # Colormap for depth visualization
        colormap_dict = {
            "turbo": cv2.COLORMAP_TURBO,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "plasma": cv2.COLORMAP_PLASMA,
            "jet": cv2.COLORMAP_JET,
            "hot": cv2.COLORMAP_HOT,
            "cool": cv2.COLORMAP_COOL,
        }
        self.depth_colormap = colormap_dict.get(depth_colormap.lower(), cv2.COLORMAP_TURBO)
        
        # Window name
        self.window_name = "Seg-Depth Pipeline"
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, window_width, window_height)
        
        logger.info("Visualizer initialized")
    
    def visualize(
        self,
        frame: np.ndarray,
        result: Dict
    ) -> np.ndarray:
        """
        Create visualization from pipeline result.
        
        Args:
            frame: Original frame (H, W, 3) RGB
            result: Pipeline processing result
            
        Returns:
            Visualization image (H, W, 3) RGB
        """
        # Calculate grid layout
        panels = []
        
        # Panel 1: Original + Segmentation overlay
        if self.show_original or self.show_segmentation:
            panel1 = self._create_segmentation_panel(frame, result)
            panels.append(panel1)
        
        # Panel 2: Depth map
        if self.show_depth and result.get("depth") is not None:
            panel2 = self._create_depth_panel(result)
            panels.append(panel2)
        
        # Panel 3: Metrics
        if self.show_metrics:
            panel3 = self._create_metrics_panel(result, frame.shape[:2])
            panels.append(panel3)
        
        # Combine panels
        if len(panels) == 0:
            return frame
        
        # Ensure all panels have same height before combining
        if len(panels) > 1:
            # Find the maximum height
            max_height = max(p.shape[0] for p in panels)
            
            # Resize panels to match height
            resized_panels = []
            for panel in panels:
                h, w = panel.shape[:2]
                if h != max_height:
                    # Calculate new width to maintain aspect ratio
                    new_width = int(w * max_height / h)
                    panel = cv2.resize(panel, (new_width, max_height))
                resized_panels.append(panel)
            panels = resized_panels
        
        # Arrange in grid
        if len(panels) == 1:
            combined = panels[0]
        elif len(panels) == 2:
            # Side by side
            combined = np.hstack(panels)
        else:
            # 2x2 grid
            top_row = np.hstack(panels[:2])
            bottom_row = panels[2]
            # Resize bottom to match width
            h1, w1 = top_row.shape[:2]
            h2, w2 = bottom_row.shape[:2]
            if w2 < w1:
                bottom_row = cv2.resize(bottom_row, (w1, h2))
            combined = np.vstack([top_row, bottom_row])
        
        return combined
    
    def _create_segmentation_panel(
        self,
        frame: np.ndarray,
        result: Dict
    ) -> np.ndarray:
        """Create panel with segmentation overlay."""
        # Start with original frame
        panel = frame.copy()
        
        # Convert RGB to BGR for OpenCV
        panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        
        # Overlay mask if available
        mask = result.get("mask")
        if mask is not None and mask.sum() > 0:
            # Ensure mask is 2D and matches panel dimensions
            if mask.ndim > 2:
                mask = mask.squeeze()
            
            # Check if mask dimensions match panel
            if mask.shape[:2] != panel.shape[:2]:
                logger.warning(f"Mask shape {mask.shape} doesn't match panel shape {panel.shape[:2]}")
                # Skip overlay if shapes don't match
            else:
                # Create colored overlay
                overlay = panel.copy()
                overlay[mask > 0] = [0, 255, 0]  # Green for detected object
                
                # Blend
                panel = cv2.addWeighted(panel, 0.7, overlay, 0.3, 0)
            
            # Draw bounding box
            mask_info = result.get("mask_info")
            if mask_info:
                bbox = mask_info["bbox"]
                x1, y1, x2, y2 = bbox
                cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw centroid
                cx, cy = mask_info["centroid"]
                cv2.circle(panel, (cx, cy), 5, (0, 0, 255), -1)
        
        # Add title
        panel = self._add_title(panel, "Segmentation")
        
        # Add status text
        status = result.get("status", "unknown")
        color = (0, 255, 0) if status == "success" else (0, 0, 255)
        cv2.putText(
            panel,
            f"Status: {status}",
            (10, panel.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        
        return panel
    
    def _create_depth_panel(self, result: Dict) -> np.ndarray:
        """Create depth visualization panel."""
        depth_result = result["depth"]
        depth_map = depth_result["depth"]
        
        # Normalize to 0-255
        depth_normalized = (depth_map * 255).astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, self.depth_colormap)
        
        # Highlight masked region if available
        mask = result.get("mask")
        if mask is not None and mask.sum() > 0:
            # Ensure mask is 2D
            if mask.ndim > 2:
                mask = mask.squeeze()
            
            # Check if mask dimensions match depth map
            if mask.shape[:2] == depth_colored.shape[:2]:
                # Create border around masked region
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(depth_colored, contours, -1, (255, 255, 255), 2)
        
        # Add title
        depth_colored = self._add_title(depth_colored, "Depth Map")
        
        # Add depth info
        depth_info = result.get("depth_info")
        if depth_info:
            text_y = depth_colored.shape[0] - 40
            cv2.putText(
                depth_colored,
                f"Mean Depth: {depth_info['mean_depth']:.3f}",
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return depth_colored
    
    def _create_metrics_panel(
        self,
        result: Dict,
        frame_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Create metrics display panel."""
        h, w = frame_shape
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(
            panel,
            "Reward Metrics",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        y_offset = 70
        line_height = 30
        
        # Frame info
        frame_id = result.get("frame_id", 0)
        cv2.putText(
            panel,
            f"Frame: {frame_id}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2
        )
        y_offset += line_height
        
        # Mask info
        mask_info = result.get("mask_info")
        if mask_info:
            cv2.putText(
                panel,
                f"Area Ratio: {mask_info['area_ratio']:.4f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 255, 100),
                2
            )
            y_offset += line_height
            
            cv2.putText(
                panel,
                f"Area Pixels: {mask_info['area']}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 255, 100),
                2
            )
            y_offset += line_height
        
        # Depth info
        depth_info = result.get("depth_info")
        if depth_info:
            cv2.putText(
                panel,
                f"Mean Depth: {depth_info['mean_depth']:.4f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 100, 255),
                2
            )
            y_offset += line_height
        
        # Reward
        reward = result.get("reward", 0.0)
        reward_color = (0, 255, 0) if reward > 0.5 else (255, 165, 0)
        cv2.putText(
            panel,
            f"Reward: {reward:.4f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            reward_color,
            2
        )
        y_offset += line_height + 10
        
        # Trends
        trends = result.get("trends")
        if trends:
            cv2.putText(
                panel,
                "--- Trends ---",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            y_offset += line_height
            
            area_trend = trends["area_trend"]
            area_arrow = "↑" if area_trend > 0 else "↓" if area_trend < 0 else "→"
            cv2.putText(
                panel,
                f"Area Trend: {area_arrow} {area_trend:+.4f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (150, 255, 150),
                2
            )
            y_offset += line_height
            
            depth_trend = trends["depth_trend"]
            depth_arrow = "↑" if depth_trend > 0 else "↓" if depth_trend < 0 else "→"
            cv2.putText(
                panel,
                f"Depth Trend: {depth_arrow} {depth_trend:+.4f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (150, 150, 255),
                2
            )
            y_offset += line_height
            
            reward_trend = trends["reward_trend"]
            reward_arrow = "↑" if reward_trend > 0 else "↓" if reward_trend < 0 else "→"
            cv2.putText(
                panel,
                f"Reward Trend: {reward_arrow} {reward_trend:+.4f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 150),
                2
            )
            y_offset += line_height
            
            cv2.putText(
                panel,
                f"Reward MA: {trends['reward_moving_avg']:.4f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2
            )
        
        return panel
    
    def _add_title(self, image: np.ndarray, title: str) -> np.ndarray:
        """Add title bar to image."""
        h, w = image.shape[:2]
        
        # Create title bar
        title_bar = np.zeros((40, w, 3), dtype=np.uint8)
        cv2.putText(
            title_bar,
            title,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Combine
        result = np.vstack([title_bar, image])
        return result
    
    def show(self, visualization: np.ndarray):
        """Display visualization in window."""
        # Convert RGB to BGR for display
        if visualization.shape[2] == 3:
            vis_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        else:
            vis_bgr = visualization
        
        cv2.imshow(self.window_name, vis_bgr)
    
    def wait_key(self, delay: int = 1) -> int:
        """
        Wait for key press.
        
        Args:
            delay: Delay in milliseconds
            
        Returns:
            Key code, or -1 if no key pressed
        """
        return cv2.waitKey(delay) & 0xFF
    
    def cleanup(self):
        """Clean up resources."""
        cv2.destroyAllWindows()
        logger.info("Visualizer cleaned up")