"""
Main Entry Point for Seg-Depth Real-Time Pipeline

This script captures screen content (e.g., YouTube video of oranges),
performs real-time segmentation with SAM3 and depth estimation with DA3,
and visualizes the results with reward metrics.

Usage:
    python main.py --config config.yaml
    
Controls:
    'q' - Quit
    'r' - Reset statistics
    'p' - Pause/Resume
    's' - Save current frame
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import yaml
import numpy as np
import cv2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sam3_model import SAM3Model
from da3_model import DA3Model
from screen_capture import ScreenCapture
from seg_depth_pipeline import SegDepthPipeline
from visualizer import Visualizer


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class SegDepthApp:
    """Main application class."""
    
    def __init__(self, config: dict):
        """Initialize application with config."""
        self.config = config
        self.paused = False
        self.save_counter = 0
        
        # Create output directory
        output_dir = Path(config["logging"]["output_dir"])
        output_dir.mkdir(exist_ok=True)
        self.output_dir = output_dir
        
        logging.info("Initializing Seg-Depth Pipeline...")
        
        # Initialize models
        logging.info("Loading SAM3 model...")
        self.sam3 = SAM3Model(
            device=config["sam3"]["device"],
            confidence_threshold=config["sam3"]["confidence_threshold"]
        )
        
        logging.info("Loading DA3 model...")
        self.da3 = DA3Model(
            model_name=config["da3"]["model_name"],
            device=config["da3"]["device"],
            normalize_depth=config["da3"]["normalize_depth"]
        )
        
        # Initialize pipeline
        logging.info("Setting up pipeline...")
        self.pipeline = SegDepthPipeline(
            sam3_model=self.sam3,
            da3_model=self.da3,
            target_prompt=config["reward"]["target_prompt"],
            prompts=config["sam3"]["prompts"],
            area_weight=config["reward"]["area_weight"],
            depth_weight=config["reward"]["depth_weight"],
            history_size=config["reward"]["history_size"]
        )
        
        # Initialize screen capture
        logging.info("Setting up screen capture...")
        region = config["screen"]["region"]
        self.capture = ScreenCapture(
            region=tuple(region) if region else None,
            fps=config["screen"]["fps"],
            monitor=config["screen"]["monitor"]
        )
        
        # Initialize visualizer
        logging.info("Setting up visualizer...")
        self.visualizer = Visualizer(
            window_width=config["visualization"]["window_width"],
            window_height=config["visualization"]["window_height"],
            depth_colormap=config["visualization"]["depth_colormap"],
            show_original=config["visualization"]["show_original"],
            show_segmentation=config["visualization"]["show_segmentation"],
            show_depth=config["visualization"]["show_depth"],
            show_metrics=config["visualization"]["show_metrics"]
        )
        
        # Statistics
        self.frame_count = 0
        self.fps_history = []
        self.last_time = time.time()
        
        logging.info("Initialization complete!")
    
    def run(self):
        """Main application loop."""
        logging.info("Starting screen capture...")
        self.capture.start()
        
        logging.info("=" * 60)
        logging.info("Seg-Depth Pipeline Running")
        logging.info("=" * 60)
        logging.info("Controls:")
        logging.info("  'q' - Quit")
        logging.info("  'r' - Reset statistics")
        logging.info("  'p' - Pause/Resume")
        logging.info("  's' - Save current frame")
        logging.info("=" * 60)
        
        try:
            while True:
                # Handle pause
                if self.paused:
                    key = self.visualizer.wait_key(100)
                    if key == ord('p'):
                        self.paused = False
                        logging.info("Resumed")
                    elif key == ord('q'):
                        break
                    continue
                
                # Capture frame
                frame = self.capture.get_latest_frame()
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Process frame
                loop_start = time.time()
                result = self.pipeline.process_frame(frame)
                
                # Create visualization
                vis = self.visualizer.visualize(frame, result)
                
                # Add FPS overlay
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time)
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                avg_fps = np.mean(self.fps_history)
                
                vis = self._add_fps_overlay(vis, avg_fps)
                self.last_time = current_time
                
                # Show visualization
                self.visualizer.show(vis)
                
                # Handle keyboard input
                key = self.visualizer.wait_key(1)
                
                if key == ord('q'):
                    logging.info("Quit requested")
                    break
                elif key == ord('r'):
                    self.pipeline.reset_history()
                    logging.info("Statistics reset")
                elif key == ord('p'):
                    self.paused = True
                    logging.info("Paused")
                elif key == ord('s'):
                    self._save_frame(frame, result, vis)
                
                self.frame_count += 1
                
                # Log statistics periodically
                if self.frame_count % 100 == 0:
                    stats = self.pipeline.get_statistics()
                    logging.info(
                        f"Stats: Frames={stats['total_frames']}, "
                        f"Detected={stats['objects_detected']}, "
                        f"Rate={stats['detection_rate']:.2%}, "
                        f"AvgReward={stats['avg_reward']:.3f}, "
                        f"FPS={avg_fps:.1f}"
                    )
        
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        
        except Exception as e:
            logging.error(f"Error in main loop: {e}", exc_info=True)
        
        finally:
            self.cleanup()
    
    def _add_fps_overlay(self, image: np.ndarray, fps: float) -> np.ndarray:
        """Add FPS counter to image."""
        img = image.copy()
        
        # Convert to BGR if needed for OpenCV text
        if img.shape[2] == 3 and img.dtype == np.uint8:
            cv2.putText(
                img,
                f"FPS: {fps:.1f}",
                (img.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        
        return img
    
    def _save_frame(self, frame: np.ndarray, result: dict, vis: np.ndarray):
        """Save current frame and visualization."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save original frame
        frame_path = self.output_dir / f"frame_{timestamp}_{self.save_counter:04d}.png"
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frame_path), frame_bgr)
        
        # Save visualization
        vis_path = self.output_dir / f"vis_{timestamp}_{self.save_counter:04d}.png"
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(vis_path), vis_bgr)
        
        self.save_counter += 1
        logging.info(f"Saved frame #{self.save_counter} to {self.output_dir}")
    
    def cleanup(self):
        """Clean up all resources."""
        logging.info("Cleaning up...")
        
        # Print final statistics
        stats = self.pipeline.get_statistics()
        logging.info("=" * 60)
        logging.info("Final Statistics:")
        logging.info(f"  Total Frames: {stats['total_frames']}")
        logging.info(f"  Objects Detected: {stats['objects_detected']}")
        logging.info(f"  Detection Rate: {stats['detection_rate']:.2%}")
        logging.info(f"  Average Reward: {stats['avg_reward']:.3f}")
        logging.info(f"  Max Reward: {stats['max_reward']:.3f}")
        logging.info("=" * 60)
        
        self.capture.cleanup()
        self.visualizer.cleanup()
        self.sam3.cleanup()
        self.da3.cleanup()
        
        logging.info("Cleanup complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time Seg-Depth Pipeline for Fruit Tracking"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return 1
    
    # Override logging level from config if specified
    if "logging" in config and "level" in config["logging"]:
        setup_logging(config["logging"]["level"])
    
    # Create and run app
    app = SegDepthApp(config)
    app.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())