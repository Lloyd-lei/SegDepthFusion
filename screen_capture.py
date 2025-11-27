"""
Screen Capture Module
Captures screen content in real-time for processing.
"""

import numpy as np
import cv2
from typing import Optional, Tuple
import logging
import threading
import time
from collections import deque

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    logging.warning("mss not installed. Install with: pip install mss")

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Real-time screen capture using mss (cross-platform)."""
    
    def __init__(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        fps: int = 10,
        monitor: int = 1
    ):
        """
        Initialize screen capture.
        
        Args:
            region: Capture region as (x, y, width, height), None for full screen
            fps: Target frames per second
            monitor: Monitor number (1 for primary)
        """
        if not MSS_AVAILABLE:
            raise RuntimeError("mss library is required. Install with: pip install mss")
        
        self.region = region
        self.fps = fps
        self.monitor = monitor
        self.frame_delay = 1.0 / fps
        
        # Create temporary mss instance to get monitor info
        with mss.mss() as sct:
            self.monitors = sct.monitors
        
        self._setup_monitor()
        
        # Threading
        self.running = False
        self.thread = None
        self.frame_queue = deque(maxlen=2)  # Keep only latest 2 frames
        self.lock = threading.Lock()
        
        logger.info(f"Screen capture initialized: {self.capture_region}")
    
    def _setup_monitor(self):
        """Setup monitor and capture region."""
        if self.monitor >= len(self.monitors):
            logger.warning(f"Monitor {self.monitor} not found, using primary")
            self.monitor = 1
        
        monitor_info = self.monitors[self.monitor]
        
        if self.region is None:
            # Full screen
            self.capture_region = {
                "top": monitor_info["top"],
                "left": monitor_info["left"],
                "width": monitor_info["width"],
                "height": monitor_info["height"]
            }
        else:
            # Custom region
            x, y, width, height = self.region
            self.capture_region = {
                "top": y,
                "left": x,
                "width": width,
                "height": height
            }
    
    def capture_frame(self, sct) -> np.ndarray:
        """
        Capture a single frame from screen.
        
        Args:
            sct: mss instance (must be created in the same thread)
        
        Returns:
            Frame as numpy array (H, W, 3) in RGB format
        """
        try:
            # Capture screen
            screenshot = sct.grab(self.capture_region)
            
            # Convert to numpy array
            frame = np.array(screenshot)
            
            # Convert BGRA to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            
            return frame
            
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            # Return black frame on error
            return np.zeros((self.capture_region["height"], 
                           self.capture_region["width"], 3), dtype=np.uint8)
    
    def _capture_loop(self):
        """Background thread for continuous capture."""
        logger.info("Capture thread started")
        
        # Create mss instance in this thread (required for Windows thread safety)
        with mss.mss() as sct:
            while self.running:
                start_time = time.time()
                
                # Capture frame
                frame = self.capture_frame(sct)
                
                # Add to queue
                with self.lock:
                    self.frame_queue.append(frame)
                
                # Maintain target FPS
                elapsed = time.time() - start_time
                sleep_time = max(0, self.frame_delay - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        logger.info("Capture thread stopped")
    
    def start(self):
        """Start continuous screen capture in background thread."""
        if self.running:
            logger.warning("Capture already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("Screen capture started")
    
    def stop(self):
        """Stop continuous screen capture."""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Screen capture stopped")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest captured frame.
        
        Returns:
            Latest frame or None if no frames available
        """
        with self.lock:
            if len(self.frame_queue) > 0:
                return self.frame_queue[-1].copy()
            return None
    
    def get_frame_size(self) -> Tuple[int, int]:
        """Get frame dimensions (width, height)."""
        return (self.capture_region["width"], self.capture_region["height"])
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        logger.info("Screen capture cleaned up")


class VideoFileCapture:
    """Alternative: Capture from video file instead of screen."""
    
    def __init__(self, video_path: str, fps: Optional[int] = None):
        """
        Initialize video file capture.
        
        Args:
            video_path: Path to video file
            fps: Override FPS (None to use video's native FPS)
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        self.native_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps is not None else self.native_fps
        self.frame_delay = 1.0 / self.fps
        
        self.running = False
        self.thread = None
        self.frame_queue = deque(maxlen=2)
        self.lock = threading.Lock()
        
        logger.info(f"Video file capture initialized: {video_path} @ {self.fps} FPS")
    
    def _capture_loop(self):
        """Background thread for video playback."""
        while self.running:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                with self.lock:
                    self.frame_queue.append(frame)
            
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_delay - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def start(self):
        """Start video playback."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("Video playback started")
    
    def stop(self):
        """Stop video playback."""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get latest frame from video."""
        with self.lock:
            if len(self.frame_queue) > 0:
                return self.frame_queue[-1].copy()
            return None
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        if hasattr(self, 'cap'):
            self.cap.release()
        logger.info("Video file capture cleaned up")