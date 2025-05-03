"""
Camera hardware interface module that provides abstraction for different camera types.
Supports USB cameras, Raspberry Pi cameras, IP cameras, and video files.
"""

import os
import time
import logging
import threading
import subprocess
from enum import Enum
from typing import Dict, Any, Tuple, Optional, List

import cv2
import numpy as np
from picamera2 import Picamera2

# Configure logger for this module
logger = logging.getLogger(__name__)

class CameraType(Enum):
    """Enumeration of supported camera types."""
    USB = 0
    PICAMERA = 1
    IP = 2
    FILE = 3
    RTSP = 4

class CameraConfig:
    """Configuration class for camera parameters."""
    
    def __init__(
        self, 
        camera_id: str,
        camera_type: CameraType,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        rotation: int = 0,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        exposure_mode: str = 'auto',
        white_balance: str = 'auto',
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
        buffer_size: int = 5
    ):
        """
        Initialize camera configuration.
        
        Args:
            camera_id: Unique identifier for the camera
            camera_type: Type of camera (USB, PICAMERA, IP, FILE, RTSP)
            width: Image width in pixels
            height: Image height in pixels
            fps: Frames per second
            rotation: Image rotation in degrees (0, 90, 180, 270)
            url: URL for IP or RTSP cameras, or file path for video files
            username: Username for IP cameras requiring authentication
            password: Password for IP cameras requiring authentication
            exposure_mode: Camera exposure mode
            white_balance: White balance setting
            flip_horizontal: Whether to flip image horizontally
            flip_vertical: Whether to flip image vertically
            buffer_size: Size of frame buffer
        """
        self.camera_id = camera_id
        self.camera_type = camera_type
        self.width = width
        self.height = height
        self.fps = fps
        self.rotation = rotation
        self.url = url
        self.username = username
        self.password = password
        self.exposure_mode = exposure_mode
        self.white_balance = white_balance
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.buffer_size = buffer_size
        
        self.validate()
    
    def validate(self):
        """Validate camera configuration parameters."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid dimensions: {self.width}x{self.height}")
        
        if self.fps <= 0:
            raise ValueError(f"Invalid FPS: {self.fps}")
        
        if self.rotation not in [0, 90, 180, 270]:
            raise ValueError(f"Invalid rotation: {self.rotation}")
            
        if self.camera_type in [CameraType.IP, CameraType.RTSP, CameraType.FILE] and not self.url:
            raise ValueError(f"URL is required for camera type: {self.camera_type}")
            
        if self.buffer_size <= 0:
            raise ValueError(f"Invalid buffer size: {self.buffer_size}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'camera_id': self.camera_id,
            'camera_type': self.camera_type.name,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'rotation': self.rotation,
            'url': self.url,
            'username': self.username,
            'password': '********' if self.password else None,
            'exposure_mode': self.exposure_mode,
            'white_balance': self.white_balance,
            'flip_horizontal': self.flip_horizontal,
            'flip_vertical': self.flip_vertical,
            'buffer_size': self.buffer_size
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CameraConfig':
        """Create configuration from dictionary."""
        camera_type = CameraType[config_dict['camera_type']]
        
        # Remove camera_type from dict and pass it separately
        config_dict = config_dict.copy()
        del config_dict['camera_type']
        
        return cls(camera_type=camera_type, **config_dict)


class CameraInterface:
    """Interface for camera hardware interaction."""
    
    def __init__(self, config: CameraConfig):
        """
        Initialize camera interface.
        
        Args:
            config: Camera configuration
        """
        self.config = config
        self.camera = None
        self.running = False
        self.last_frame = None
        self.last_frame_time = 0
        self.frame_count = 0
        self.start_time = 0
        self.fps_avg = 0
        self.lock = threading.Lock()
        self.warmup_frames = 10  # Number of frames to discard during initialization
        
        logger.info(f"Initializing camera: {config.camera_id} ({config.camera_type.name})")
    
    def initialize(self) -> bool:
        """
        Initialize camera hardware.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if self.config.camera_type == CameraType.USB:
                return self._init_usb_camera()
            elif self.config.camera_type == CameraType.PICAMERA:
                return self._init_picamera()
            elif self.config.camera_type == CameraType.IP:
                return self._init_ip_camera()
            elif self.config.camera_type == CameraType.RTSP:
                return self._init_rtsp_camera()
            elif self.config.camera_type == CameraType.FILE:
                return self._init_file_video()
            else:
                logger.error(f"Unsupported camera type: {self.config.camera_type}")
                return False
        except Exception as e:
            logger.error(f"Camera initialization failed: {str(e)}")
            return False
    
    def _init_usb_camera(self) -> bool:
        """Initialize USB camera."""
        try:
            # Try to convert camera_id to integer for numbered devices
            try:
                device_id = int(self.config.camera_id)
            except ValueError:
                device_id = self.config.camera_id
            
            self.camera = cv2.VideoCapture(device_id)
            
            if not self.camera.isOpened():
                logger.error(f"Failed to open USB camera: {self.config.camera_id}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Check if properties were set correctly
            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"USB Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # Warm up camera
            for _ in range(self.warmup_frames):
                self.camera.read()
            
            return True
        except Exception as e:
            logger.error(f"USB camera initialization error: {str(e)}")
            return False
    
    def _init_picamera(self) -> bool:
        """Initialize Raspberry Pi camera."""
        try:
            self.camera = Picamera2()
            
            # Configure camera
            config = self.camera.create_video_configuration(
                main={"size": (self.config.width, self.config.height)},
                controls={
                    "FrameRate": self.config.fps,
                    "ExposureMode": self.config.exposure_mode,
                    "AwbMode": self.config.white_balance
                }
            )
            
            self.camera.configure(config)
            self.camera.start()
            
            # Add a small delay to allow camera to initialize
            time.sleep(0.5)
            
            logger.info(f"Pi Camera initialized: {self.config.width}x{self.config.height} @ {self.config.fps}fps")
            
            # Check if camera is actually working by grabbing a test frame
            test_frame = self.camera.capture_array()
            if test_frame is None or test_frame.size == 0:
                logger.error("Pi Camera initialization failed: Could not capture test frame")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Pi Camera initialization error: {str(e)}")
            return False
    
    def _init_ip_camera(self) -> bool:
        """Initialize IP camera."""
        try:
            # Construct URL with authentication if provided
            url = self.config.url
            if self.config.username and self.config.password:
                # Extract protocol and remainder of the URL
                protocol, remainder = url.split('://', 1)
                auth_url = f"{protocol}://{self.config.username}:{self.config.password}@{remainder}"
            else:
                auth_url = url
            
            self.camera = cv2.VideoCapture(auth_url)
            
            if not self.camera.isOpened():
                logger.error(f"Failed to connect to IP camera: {url}")
                return False
            
            # Attempt to set properties if the camera supports it
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Read actual properties
            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"IP Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # Check connection by reading a test frame
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                logger.error("IP Camera initialization failed: Could not capture test frame")
                return False
                
            return True
        except Exception as e:
            logger.error(f"IP camera initialization error: {str(e)}")
            return False
    
    def _init_rtsp_camera(self) -> bool:
        """Initialize RTSP camera stream."""
        try:
            # Construct RTSP URL with authentication if provided
            url = self.config.url
            if self.config.username and self.config.password:
                # Extract protocol and remainder of the URL
                protocol, remainder = url.split('://', 1)
                auth_url = f"{protocol}://{self.config.username}:{self.config.password}@{remainder}"
            else:
                auth_url = url
            
            # Use FFMPEG backend for better RTSP support
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            
            # Create video capture with optimized parameters
            self.camera = cv2.VideoCapture(auth_url, cv2.CAP_FFMPEG)
            
            if not self.camera.isOpened():
                logger.error(f"Failed to connect to RTSP stream: {url}")
                return False
            
            # Set buffer size to reduce latency
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Read actual properties
            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"RTSP Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # Check connection by reading a test frame
            for _ in range(3):  # Try a few times as RTSP can be slow to start
                ret, test_frame = self.camera.read()
                if ret and test_frame is not None:
                    break
                time.sleep(0.5)
                
            if not ret or test_frame is None:
                logger.error("RTSP Camera initialization failed: Could not capture test frame")
                return False
                
            return True
        except Exception as e:
            logger.error(f"RTSP camera initialization error: {str(e)}")
            return False
    
    def _init_file_video(self) -> bool:
        """Initialize video file as camera input."""
        try:
            file_path = self.config.url
            if not os.path.exists(file_path):
                logger.error(f"Video file does not exist: {file_path}")
                return False
            
            self.camera = cv2.VideoCapture(file_path)
            
            if not self.camera.isOpened():
                logger.error(f"Failed to open video file: {file_path}")
                return False
            
            # Get video file properties
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video file initialized: {actual_width}x{actual_height} @ {actual_fps}fps, {frame_count} frames")
            
            return True
        except Exception as e:
            logger.error(f"Video file initialization error: {str(e)}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple containing success flag and frame (if successful)
        """
        if self.camera is None:
            return False, None
        
        try:
            if self.config.camera_type == CameraType.PICAMERA:
                # PiCamera uses different method for capturing
                frame = self.camera.capture_array()
                success = frame is not None and frame.size > 0
            else:
                # OpenCV-based cameras
                success, frame = self.camera.read()
            
            if not success:
                return False, None
            
            # Apply rotation if needed
            if self.config.rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.config.rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.config.rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Apply flipping if needed
            if self.config.flip_horizontal and self.config.flip_vertical:
                frame = cv2.flip(frame, -1)
            elif self.config.flip_horizontal:
                frame = cv2.flip(frame, 1)
            elif self.config.flip_vertical:
                frame = cv2.flip(frame, 0)
            
            # Update frame metrics
            current_time = time.time()
            if self.start_time == 0:
                self.start_time = current_time
            
            self.frame_count += 1
            self.last_frame_time = current_time
            
            # Calculate FPS every 30 frames
            if self.frame_count % 30 == 0:
                elapsed = current_time - self.start_time
                if elapsed > 0:
                    self.fps_avg = self.frame_count / elapsed
            
            # Store the frame
            with self.lock:
                self.last_frame = frame.copy()
            
            return True, frame
            
        except Exception as e:
            logger.error(f"Error reading frame: {str(e)}")
            return False, None
    
    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the most recent camera frame.
        
        Returns:
            Tuple containing success flag and frame (if available)
        """
        with self.lock:
            if self.last_frame is None:
                return False, None
            return True, self.last_frame.copy()
    
    def get_camera_stats(self) -> Dict[str, Any]:
        """
        Get camera statistics.
        
        Returns:
            Dictionary of camera statistics
        """
        return {
            'camera_id': self.config.camera_id,
            'camera_type': self.config.camera_type.name,
            'resolution': f"{self.config.width}x{self.config.height}",
            'target_fps': self.config.fps,
            'actual_fps': round(self.fps_avg, 2),
            'frames_captured': self.frame_count,
            'running_time': round(time.time() - self.start_time, 2) if self.start_time > 0 else 0,
            'last_frame_time': self.last_frame_time
        }
    
    def release(self):
        """Release camera resources."""
        try:
            if self.camera is not None:
                if self.config.camera_type == CameraType.PICAMERA:
                    self.camera.stop()
                    self.camera.close()
                else:
                    self.camera.release()
                    
                logger.info(f"Camera released: {self.config.camera_id}")
        except Exception as e:
            logger.error(f"Error releasing camera: {str(e)}")
        
        self.camera = None
        self.running = False
    
    def __del__(self):
        """Clean up resources when object is deleted."""
        self.release()


def check_camera_availability() -> List[Dict[str, Any]]:
    """
    Check available cameras on the system.
    
    Returns:
        List of dictionaries with camera information
    """
    available_cameras = []
    
    # Check for USB cameras
    try:
        # Try to detect cameras using OpenCV
        for i in range(10):  # Check first 10 indexes
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # This index is available
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                available_cameras.append({
                    'camera_id': str(i),
                    'camera_type': CameraType.USB.name,
                    'resolution': f"{width}x{height}",
                    'fps': fps
                })
                
                cap.release()
    except Exception as e:
        logger.error(f"Error checking USB cameras: {str(e)}")
    
    # Check for Raspberry Pi camera
    try:
        # Try to create a Picamera2 instance
        picam = Picamera2()
        
        # Get camera information
        info = picam.camera_properties
        
        available_cameras.append({
            'camera_id': 'picamera',
            'camera_type': CameraType.PICAMERA.name,
            'resolution': f"{info.get('MaxWidth', 'unknown')}x{info.get('MaxHeight', 'unknown')}",
            'fps': info.get('MaxFrameRate', 'unknown')
        })
        
        picam.close()
    except Exception as e:
        # Pi Camera not available or error
        logger.debug(f"Pi Camera check: {str(e)}")
    
    return available_cameras
