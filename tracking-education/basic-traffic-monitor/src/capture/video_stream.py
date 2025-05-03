"""
Video streaming module for managing continuous video capture from cameras.
Provides thread-safe frame buffering and stream management.
"""

import time
import queue
import logging
import threading
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np

from .camera_interface import CameraInterface

# Configure logger for this module
logger = logging.getLogger(__name__)

class StreamStatus(Enum):
    """Stream status enumeration."""
    STOPPED = 0
    RUNNING = 1
    ERROR = 2
    PAUSED = 3


class FrameBuffer:
    """Thread-safe buffer for video frames."""
    
    def __init__(self, maxsize: int = 10):
        """
        Initialize frame buffer.
        
        Args:
            maxsize: Maximum buffer size
        """
        self.buffer = queue.Queue(maxsize=maxsize)
        self.frame_count = 0
        self.drop_count = 0
        self.last_frame = None
        self.lock = threading.Lock()
    
    def put(self, frame: np.ndarray, timestamp: float) -> bool:
        """
        Add frame to buffer.
        
        Args:
            frame: Video frame
            timestamp: Frame timestamp
        
        Returns:
            True if frame was added, False if buffer is full and frame was dropped
        """
        try:
            # Create frame tuple with timestamp
            frame_data = (frame.copy(), timestamp)
            
            # Try to add to buffer without blocking
            self.buffer.put_nowait(frame_data)
            
            with self.lock:
                self.frame_count += 1
                self.last_frame = frame_data
            
            return True
        except queue.Full:
            # Buffer is full, drop this frame
            with self.lock:
                self.drop_count += 1
            
            return False
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Get frame from buffer.
        
        Args:
            block: Whether to block waiting for a frame
            timeout: Timeout in seconds if blocking
        
        Returns:
            Tuple of (frame, timestamp) or (None, None) if no frame available
        """
        try:
            return self.buffer.get(block=block, timeout=timeout)
        except queue.Empty:
            return None, None
    
    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Get the most recent frame, emptying the buffer.
        
        Returns:
            Most recent frame and timestamp
        """
        with self.lock:
            # Clear the buffer
            while not self.buffer.empty():
                try:
                    self.last_frame = self.buffer.get_nowait()
                except queue.Empty:
                    break
            
            # Return the last frame if available
            if self.last_frame is not None:
                return self.last_frame
        
        return None, None
    
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            while not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                except queue.Empty:
                    break
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get buffer statistics.
        
        Returns:
            Dictionary of buffer statistics
        """
        with self.lock:
            return {
                'size': self.buffer.qsize(),
                'capacity': self.buffer.maxsize,
                'frames_processed': self.frame_count,
                'frames_dropped': self.drop_count,
                'drop_rate': round(self.drop_count / max(1, self.frame_count + self.drop_count) * 100, 2)
            }


class VideoStream:
    """
    Thread-safe video streaming class that manages continuous frame capture.
    """
    
    def __init__(self, camera: CameraInterface, buffer_size: int = 10):
        """
        Initialize video stream.
        
        Args:
            camera: Camera interface
            buffer_size: Maximum buffer size
        """
        self.camera = camera
        self.buffer = FrameBuffer(maxsize=buffer_size)
        self.thread = None
        self.status = StreamStatus.STOPPED
        self.stop_event = threading.Event()
        self.frame_count = 0
        self.start_time = 0
        self.latest_fps = 0
        self.error_count = 0
        self.max_errors = 5  # Maximum consecutive errors before stopping
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
    
    def start(self) -> bool:
        """
        Start video streaming.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.status == StreamStatus.RUNNING:
            logger.warning("Video stream is already running")
            return True
        
        # Initialize camera if needed
        if not self.camera.initialize():
            logger.error(f"Failed to initialize camera {self.camera.config.camera_id}")
            self.status = StreamStatus.ERROR
            return False
        
        # Clear stop event
        self.stop_event.clear()
        
        # Reset statistics
        self.frame_count = 0
        self.start_time = time.time()
        self.error_count = 0
        self.recovery_attempts = 0
        
        # Start streaming thread
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        
        self.status = StreamStatus.RUNNING
        logger.info(f"Video stream started for camera {self.camera.config.camera_id}")
        
        return True
    
    def stop(self):
        """Stop video streaming."""
        if self.status == StreamStatus.STOPPED:
            return
        
        logger.info(f"Stopping video stream for camera {self.camera.config.camera_id}")
        
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        
        # Release camera
        self.camera.release()
        
        self.status = StreamStatus.STOPPED
        logger.info(f"Video stream stopped for camera {self.camera.config.camera_id}")
    
    def pause(self):
        """Pause video streaming."""
        if self.status == StreamStatus.RUNNING:
            self.status = StreamStatus.PAUSED
            logger.info(f"Video stream paused for camera {self.camera.config.camera_id}")
    
    def resume(self):
        """Resume video streaming."""
        if self.status == StreamStatus.PAUSED:
            self.status = StreamStatus.RUNNING
            logger.info(f"Video stream resumed for camera {self.camera.config.camera_id}")
    
    def _stream_loop(self):
        """Main video streaming loop."""
        consecutive_errors = 0
        
        while not self.stop_event.is_set():
            try:
                # Skip frame if paused
                if self.status == StreamStatus.PAUSED:
                    time.sleep(0.1)
                    continue
                
                # Read frame from camera
                success, frame = self.camera.read_frame()
                
                if not success or frame is None:
                    consecutive_errors += 1
                    logger.warning(f"Failed to read frame ({consecutive_errors}/{self.max_errors})")
                    
                    # Check if we've reached max errors
                    if consecutive_errors >= self.max_errors:
                        if self.recovery_attempts < self.max_recovery_attempts:
                            logger.warning(f"Attempting to recover camera connection ({self.recovery_attempts + 1}/{self.max_recovery_attempts})")
                            # Try to re-initialize the camera
                            self.camera.release()
                            time.sleep(1.0)
                            if self.camera.initialize():
                                logger.info("Camera connection recovered")
                                consecutive_errors = 0
                                self.recovery_attempts += 1
                            else:
                                logger.error("Failed to recover camera connection")
                        else:
                            logger.error(f"Too many consecutive errors, stopping stream")
                            self.status = StreamStatus.ERROR
                            break
                    
                    # Wait before retrying
                    time.sleep(0.5)
                    continue
                
                # If we got here, reset error counter
                consecutive_errors = 0
                self.frame_count += 1
                
                # Update FPS calculation every 30 frames
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    self.latest_fps = 30 / max(0.001, elapsed)
                    self.start_time = time.time()
                
                # Add frame to buffer with current timestamp
                timestamp = time.time()
                self.buffer.put(frame, timestamp)
                
            except Exception as e:
                logger.error(f"Error in video stream loop: {str(e)}")
                consecutive_errors += 1
                
                if consecutive_errors >= self.max_errors:
                    logger.error(f"Too many errors in stream loop, stopping stream")
                    self.status = StreamStatus.ERROR
                    break
                
                time.sleep(0.5)
        
        # Final cleanup if loop exits
        if self.status != StreamStatus.STOPPED:
            self.camera.release()
            self.status = StreamStatus.STOPPED
    
    def read(self, block: bool = True, timeout: Optional[float] = None) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """
        Read a frame from the stream.
        
        Args:
            block: Whether to block waiting for a frame
            timeout: Timeout in seconds if blocking
        
        Returns:
            Tuple of (success, frame, timestamp)
        """
        if self.status != StreamStatus.RUNNING and self.status != StreamStatus.PAUSED:
            return False, None, None
        
        frame, timestamp = self.buffer.get(block=block, timeout=timeout)
        return frame is not None, frame, timestamp
    
    def read_latest(self) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """
        Read the most recent frame from the stream.
        
        Returns:
            Tuple of (success, frame, timestamp)
        """
        if self.status != StreamStatus.RUNNING and self.status != StreamStatus.PAUSED:
            return False, None, None
        
        frame, timestamp = self.buffer.get_latest()
        return frame is not None, frame, timestamp
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get stream status and statistics.
        
        Returns:
            Dictionary with stream status and statistics
        """
        camera_stats = self.camera.get_camera_stats() if self.camera else {}
        buffer_stats = self.buffer.get_stats()
        
        return {
            'status': self.status.name,
            'camera': camera_stats,
            'buffer': buffer_stats,
            'current_fps': round(self.latest_fps, 2),
            'frames_processed': self.frame_count,
            'runtime': round(time.time() - self.start_time, 2) if self.start_time > 0 else 0,
            'error_count': self.error_count,
            'recovery_attempts': self.recovery_attempts
        }
    
    def __del__(self):
        """Clean up resources when object is deleted."""
        self.stop()


def create_video_writer(
    output_path: str,
    frame_width: int,
    frame_height: int,
    fps: float = 30.0,
    codec: str = 'XVID'
) -> cv2.VideoWriter:
    """
    Create a video writer for saving stream to file.
    
    Args:
        output_path: Path to output video file
        frame_width: Width of video frames
        frame_height: Height of video frames
        fps: Frames per second
        codec: Four-character code for video codec
    
    Returns:
        Configured VideoWriter object
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
