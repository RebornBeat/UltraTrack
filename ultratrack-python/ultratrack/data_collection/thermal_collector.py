"""
Thermal Collection Module for UltraTrack

This module provides capabilities for collecting and processing thermal imaging data from various
thermal camera sources. It supports multiple thermal camera types, streaming protocols, and 
temperature calibration methods.

Key features:
- Thermal camera discovery and connection management
- Real-time thermal data streaming and processing
- Temperature calibration and normalization
- Multi-format thermal data handling (raw, colorized, temperature maps)
- Integration with the UltraTrack tracking system
- Thermal data validation and quality assessment

Copyright (c) 2025 Your Organization
"""

import logging
import time
import threading
import queue
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import socket
import json
import os
import ipaddress
from urllib.parse import urlparse

from ultratrack.config import ConfigManager
from ultratrack.data_collection.data_validator import DataValidator, ValidationResult
from ultratrack.data_processing.preprocessing import Preprocessor

# Set up module logger
logger = logging.getLogger(__name__)


class ThermalCameraType(Enum):
    """Types of supported thermal cameras."""
    FLIR = "flir"
    SEEK = "seek"
    OPGAL = "opgal"
    XENICS = "xenics"
    AXIS = "axis"
    DAHUA = "dahua"
    HIKVISION = "hikvision"
    BOSCH = "bosch"
    CUSTOM = "custom"
    GENERIC = "generic"


class ThermalStreamProtocol(Enum):
    """Supported thermal data streaming protocols."""
    RTSP = "rtsp"
    HTTP = "http"
    ONVIF = "onvif"
    PROPRIETARY = "proprietary"
    USB = "usb"
    GIGE = "gige"
    SDK = "sdk"
    FILE = "file"


class ThermalColorPalette(Enum):
    """Color palettes for thermal image visualization."""
    IRON = "iron"
    RAINBOW = "rainbow"
    WHITE_HOT = "white_hot"
    BLACK_HOT = "black_hot"
    ARCTIC = "arctic"
    LAVA = "lava"
    MEDICAL = "medical"
    CUSTOM = "custom"


class ThermalDataFormat(Enum):
    """Thermal data output formats."""
    RAW = "raw"                # Raw sensor data
    NORMALIZED = "normalized"  # Normalized 0-1 values
    TEMPERATURE = "temperature"  # Actual temperature values in Celsius
    COLORIZED = "colorized"    # Colorized visualization
    HYBRID = "hybrid"          # Temperature data with colorization


@dataclass
class ThermalSourceInfo:
    """Information about a thermal camera source."""
    id: str
    name: str
    type: ThermalCameraType
    protocol: ThermalStreamProtocol
    address: str
    port: int = 0
    username: str = ""
    password: str = ""
    path: str = ""
    temperature_range: Tuple[float, float] = (-20.0, 150.0)
    resolution: Tuple[int, int] = (640, 480)
    frame_rate: int = 9
    emissivity: float = 0.95
    distance: float = 1.0  # Distance to target in meters
    color_palette: ThermalColorPalette = ThermalColorPalette.IRON
    calibrated: bool = False
    sdk_params: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate and initialize default values."""
        if self.sdk_params is None:
            self.sdk_params = {}
            
        # Ensure temperature range makes sense
        if self.temperature_range[0] >= self.temperature_range[1]:
            raise ValueError(f"Invalid temperature range: {self.temperature_range}")
        
        # Ensure emissivity is in valid range
        if not 0.1 <= self.emissivity <= 1.0:
            raise ValueError(f"Invalid emissivity value: {self.emissivity}")
    
    @property
    def connection_string(self) -> str:
        """Generate a connection string for the thermal source."""
        if self.protocol == ThermalStreamProtocol.RTSP:
            auth = f"{self.username}:{self.password}@" if self.username else ""
            return f"rtsp://{auth}{self.address}:{self.port or 554}{self.path}"
        elif self.protocol == ThermalStreamProtocol.HTTP:
            auth = f"{self.username}:{self.password}@" if self.username else ""
            return f"http://{auth}{self.address}:{self.port or 80}{self.path}"
        elif self.protocol == ThermalStreamProtocol.FILE:
            return self.path
        elif self.protocol == ThermalStreamProtocol.USB:
            return f"usb:{self.path or '0'}"
        elif self.protocol == ThermalStreamProtocol.GIGE:
            return f"gige:{self.address}"
        elif self.protocol == ThermalStreamProtocol.SDK:
            # For SDK connections, return a JSON representation of parameters
            return json.dumps({
                "type": self.type.value,
                "address": self.address,
                "port": self.port,
                **self.sdk_params
            })
        else:
            return f"{self.protocol.value}://{self.address}"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThermalSourceInfo':
        """Create a ThermalSourceInfo instance from a dictionary."""
        # Handle enum conversions
        if 'type' in data and not isinstance(data['type'], ThermalCameraType):
            data['type'] = ThermalCameraType(data['type'])
        if 'protocol' in data and not isinstance(data['protocol'], ThermalStreamProtocol):
            data['protocol'] = ThermalStreamProtocol(data['protocol'])
        if 'color_palette' in data and not isinstance(data['color_palette'], ThermalColorPalette):
            data['color_palette'] = ThermalColorPalette(data['color_palette'])
        
        # Handle tuple conversions
        if 'temperature_range' in data and not isinstance(data['temperature_range'], tuple):
            data['temperature_range'] = tuple(data['temperature_range'])
        if 'resolution' in data and not isinstance(data['resolution'], tuple):
            data['resolution'] = tuple(data['resolution'])
            
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the ThermalSourceInfo to a dictionary."""
        result = {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "protocol": self.protocol.value,
            "address": self.address,
            "port": self.port,
            "username": self.username,
            "password": "********" if self.password else "",  # Don't expose actual password
            "path": self.path,
            "temperature_range": list(self.temperature_range),
            "resolution": list(self.resolution),
            "frame_rate": self.frame_rate,
            "emissivity": self.emissivity,
            "distance": self.distance,
            "color_palette": self.color_palette.value,
            "calibrated": self.calibrated
        }
        
        if self.sdk_params:
            result["sdk_params"] = self.sdk_params
            
        return result


@dataclass
class TemperatureReading:
    """Temperature reading with metadata."""
    value: float  # Temperature in Celsius
    min_value: float  # Minimum temperature in frame/region
    max_value: float  # Maximum temperature in frame/region
    average_value: float  # Average temperature in frame/region
    ambient_temp: float  # Ambient temperature
    region: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    timestamp: float = 0.0  # UNIX timestamp
    confidence: float = 1.0  # Measurement confidence (0-1)
    emissivity: float = 0.95  # Material emissivity used for calculation
    distance: float = 1.0  # Distance to target in meters
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class ThermalFrame:
    """A thermal frame with temperature data and metadata."""
    # Raw thermal data
    raw_data: np.ndarray  # Raw thermal sensor data
    temperature_data: np.ndarray  # Temperature map in Celsius
    
    # Frame information
    source_id: str  # ID of the source camera
    timestamp: float  # UNIX timestamp
    sequence_num: int  # Frame sequence number
    
    # Camera parameters
    emissivity: float  # Material emissivity setting
    distance: float  # Distance to target in meters
    ambient_temp: float  # Ambient temperature in Celsius
    
    # Temperature range
    min_temp: float  # Minimum temperature in frame
    max_temp: float  # Maximum temperature in frame
    avg_temp: float  # Average temperature in frame
    
    # Frame metadata
    resolution: Tuple[int, int]  # Width, height
    format: ThermalDataFormat  # Data format
    
    # Optional colorized visualization
    colorized: Optional[np.ndarray] = None  # BGR colorized image
    color_palette: ThermalColorPalette = ThermalColorPalette.IRON
    
    # Quality metrics
    quality_score: float = 1.0  # Overall quality score (0-1)
    noise_level: float = 0.0  # Estimated noise level (0-1)
    motion_blur: float = 0.0  # Motion blur estimation (0-1)
    
    # Processing metadata
    processed: bool = False  # Whether the frame has been processed
    calibration_applied: bool = False  # Whether calibration has been applied
    
    def __post_init__(self):
        """Validate and initialize frame data."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()
            
        # Ensure temperature data has the correct shape
        if self.temperature_data.shape[:2] != self.resolution[::-1]:  # numpy shape is (height, width)
            raise ValueError(f"Temperature data shape {self.temperature_data.shape} doesn't match resolution {self.resolution}")
    
    def get_temperature_at(self, x: int, y: int) -> float:
        """
        Get temperature at a specific pixel location.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Temperature value in Celsius
        """
        if 0 <= x < self.resolution[0] and 0 <= y < self.resolution[1]:
            return float(self.temperature_data[y, x])
        else:
            raise ValueError(f"Coordinates ({x}, {y}) out of bounds for frame with resolution {self.resolution}")
    
    def get_region_temperature(self, x: int, y: int, width: int, height: int) -> TemperatureReading:
        """
        Get temperature statistics for a rectangular region.
        
        Args:
            x: X coordinate of top-left corner
            y: Y coordinate of top-left corner
            width: Width of region
            height: Height of region
            
        Returns:
            TemperatureReading with region statistics
        """
        # Ensure region is within bounds
        x = max(0, min(x, self.resolution[0] - 1))
        y = max(0, min(y, self.resolution[1] - 1))
        width = max(1, min(width, self.resolution[0] - x))
        height = max(1, min(height, self.resolution[1] - y))
        
        # Extract region
        region = self.temperature_data[y:y+height, x:x+width]
        
        # Calculate statistics
        return TemperatureReading(
            value=float(np.median(region)),
            min_value=float(np.min(region)),
            max_value=float(np.max(region)),
            average_value=float(np.mean(region)),
            ambient_temp=self.ambient_temp,
            region=(x, y, width, height),
            timestamp=self.timestamp,
            emissivity=self.emissivity,
            distance=self.distance
        )
    
    def colorize(self, palette: Optional[ThermalColorPalette] = None) -> np.ndarray:
        """
        Generate a colorized visualization of thermal data.
        
        Args:
            palette: Optional color palette to use (defaults to frame's palette)
            
        Returns:
            BGR colorized image as numpy array
        """
        if palette is None:
            palette = self.color_palette
            
        # If already colorized with requested palette, return existing
        if self.colorized is not None and palette == self.color_palette:
            return self.colorized
            
        # Normalize temperature data to 0-1 range for colorization
        normalized = np.clip((self.temperature_data - self.min_temp) / (self.max_temp - self.min_temp), 0, 1)
        
        # Apply colormap based on palette
        if palette == ThermalColorPalette.IRON:
            colorized = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        elif palette == ThermalColorPalette.RAINBOW:
            colorized = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        elif palette == ThermalColorPalette.WHITE_HOT:
            colorized = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_BONE)
        elif palette == ThermalColorPalette.BLACK_HOT:
            colorized = cv2.applyColorMap(((1-normalized) * 255).astype(np.uint8), cv2.COLORMAP_BONE)
        elif palette == ThermalColorPalette.ARCTIC:
            colorized = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_WINTER)
        elif palette == ThermalColorPalette.LAVA:
            colorized = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        elif palette == ThermalColorPalette.MEDICAL:
            # Custom medical thermal palette (blue-green-yellow-red)
            colorized = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        else:
            # Default to IRON for unknown palettes
            colorized = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            
        # Cache the result if using the frame's own palette
        if palette == self.color_palette:
            self.colorized = colorized
            
        return colorized
    
    def resize(self, width: int, height: int) -> 'ThermalFrame':
        """
        Create a resized copy of the thermal frame.
        
        Args:
            width: Target width
            height: Target height
            
        Returns:
            New resized ThermalFrame instance
        """
        # Resize temperature data
        resized_temp = cv2.resize(self.temperature_data, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Resize colorized data if it exists
        resized_color = None
        if self.colorized is not None:
            resized_color = cv2.resize(self.colorized, (width, height), interpolation=cv2.INTER_CUBIC)
            
        # Create new frame with resized data
        return ThermalFrame(
            raw_data=cv2.resize(self.raw_data, (width, height), interpolation=cv2.INTER_CUBIC),
            temperature_data=resized_temp,
            source_id=self.source_id,
            timestamp=self.timestamp,
            sequence_num=self.sequence_num,
            emissivity=self.emissivity,
            distance=self.distance,
            ambient_temp=self.ambient_temp,
            min_temp=self.min_temp,
            max_temp=self.max_temp,
            avg_temp=self.avg_temp,
            resolution=(width, height),
            format=self.format,
            colorized=resized_color,
            color_palette=self.color_palette,
            quality_score=self.quality_score,
            noise_level=self.noise_level,
            motion_blur=self.motion_blur,
            processed=self.processed,
            calibration_applied=self.calibration_applied
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert frame metadata to a dictionary (without image data).
        
        Returns:
            Dictionary with frame metadata
        """
        return {
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "sequence_num": self.sequence_num,
            "emissivity": self.emissivity,
            "distance": self.distance,
            "ambient_temp": self.ambient_temp,
            "min_temp": self.min_temp,
            "max_temp": self.max_temp,
            "avg_temp": self.avg_temp,
            "resolution": list(self.resolution),
            "format": self.format.value,
            "color_palette": self.color_palette.value,
            "quality_score": self.quality_score,
            "noise_level": self.noise_level,
            "motion_blur": self.motion_blur,
            "processed": self.processed,
            "calibration_applied": self.calibration_applied
        }


class ThermalSourceStatus(Enum):
    """Status of a thermal source connection."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"
    CALIBRATING = "calibrating"
    PAUSED = "paused"


class ThermalSourceConnection:
    """
    Manages connection to a single thermal camera source.
    
    This class handles:
    - Connection establishment and maintenance
    - Frame acquisition and buffering
    - Basic preprocessing and temperature calibration
    - Connection status and statistics
    """
    
    def __init__(self, source_info: ThermalSourceInfo, buffer_size: int = 30):
        """
        Initialize a thermal source connection.
        
        Args:
            source_info: Information about the thermal source
            buffer_size: Size of the frame buffer (number of frames)
        """
        self.source_info = source_info
        self.buffer_size = buffer_size
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.status = ThermalSourceStatus.DISCONNECTED
        self.error_message = ""
        self.last_frame_time = 0.0
        self.frame_count = 0
        self.dropped_frames = 0
        self.connection_time = 0.0
        self.current_fps = 0.0
        self.calibration_matrix = None  # For temperature calibration
        
        # Thread control
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.RLock()
        
        # Camera-specific parameters
        self._camera = None
        self._stream = None
        self._sdk_instance = None
        
        # Performance metrics
        self._acquisition_times = []
        self._processing_times = []
        
        logger.info(f"Created thermal source connection for {source_info.name} ({source_info.id})")
    
    def connect(self) -> bool:
        """
        Establish connection to the thermal camera.
        
        Returns:
            True if connection successful, False otherwise
        """
        with self._lock:
            if self.status in [ThermalSourceStatus.CONNECTED, ThermalSourceStatus.STREAMING]:
                logger.warning(f"Already connected to {self.source_info.name}")
                return True
                
            logger.info(f"Connecting to thermal source {self.source_info.name} ({self.source_info.id})")
            self.status = ThermalSourceStatus.CONNECTING
            self.error_message = ""
            
            try:
                # Different connection methods based on protocol
                if self.source_info.protocol == ThermalStreamProtocol.RTSP:
                    self._connect_rtsp()
                elif self.source_info.protocol == ThermalStreamProtocol.HTTP:
                    self._connect_http()
                elif self.source_info.protocol == ThermalStreamProtocol.FILE:
                    self._connect_file()
                elif self.source_info.protocol == ThermalStreamProtocol.USB:
                    self._connect_usb()
                elif self.source_info.protocol == ThermalStreamProtocol.SDK:
                    self._connect_sdk()
                elif self.source_info.protocol == ThermalStreamProtocol.GIGE:
                    self._connect_gige()
                else:
                    raise NotImplementedError(f"Protocol {self.source_info.protocol} not implemented")
                
                self.status = ThermalSourceStatus.CONNECTED
                self.connection_time = time.time()
                logger.info(f"Successfully connected to {self.source_info.name}")
                return True
                
            except Exception as e:
                self.status = ThermalSourceStatus.ERROR
                self.error_message = str(e)
                logger.error(f"Failed to connect to {self.source_info.name}: {str(e)}", exc_info=True)
                return False
    
    def _connect_rtsp(self):
        """Establish RTSP connection."""
        connection_string = self.source_info.connection_string
        logger.debug(f"Connecting to RTSP stream: {connection_string}")
        
        # OpenCV's VideoCapture for RTSP
        self._stream = cv2.VideoCapture(connection_string)
        
        if not self._stream.isOpened():
            raise ConnectionError(f"Failed to open RTSP stream: {connection_string}")
        
        # Set camera parameters if possible
        self._stream.set(cv2.CAP_PROP_BUFFERSIZE, min(self.buffer_size, 10))  # Buffer size
        
        # Try to set frame rate
        if self.source_info.frame_rate > 0:
            self._stream.set(cv2.CAP_PROP_FPS, self.source_info.frame_rate)
    
    def _connect_http(self):
        """Establish HTTP connection."""
        connection_string = self.source_info.connection_string
        logger.debug(f"Connecting to HTTP stream: {connection_string}")
        
        # OpenCV's VideoCapture for HTTP
        self._stream = cv2.VideoCapture(connection_string)
        
        if not self._stream.isOpened():
            raise ConnectionError(f"Failed to open HTTP stream: {connection_string}")
    
    def _connect_file(self):
        """Connect to a thermal video/image file."""
        file_path = self.source_info.path
        logger.debug(f"Connecting to thermal file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Thermal file not found: {file_path}")
        
        # OpenCV's VideoCapture for files
        self._stream = cv2.VideoCapture(file_path)
        
        if not self._stream.isOpened():
            raise IOError(f"Failed to open thermal file: {file_path}")
    
    def _connect_usb(self):
        """Connect to USB thermal camera."""
        device_id = self.source_info.path
        if not device_id:
            device_id = 0  # Default to first device
        else:
            try:
                device_id = int(device_id)
            except ValueError:
                # Keep as string if not a valid integer (could be a device path)
                pass
                
        logger.debug(f"Connecting to USB thermal camera: {device_id}")
        
        # OpenCV's VideoCapture for USB
        self._stream = cv2.VideoCapture(device_id)
        
        if not self._stream.isOpened():
            raise ConnectionError(f"Failed to open USB thermal camera: {device_id}")
    
    def _connect_sdk(self):
        """Connect using vendor-specific SDK."""
        # This would use a vendor-specific SDK based on camera type
        camera_type = self.source_info.type
        logger.debug(f"Connecting to {camera_type.value} camera using SDK")
        
        # Different SDK connections based on camera type
        if camera_type == ThermalCameraType.FLIR:
            self._connect_flir_sdk()
        elif camera_type == ThermalCameraType.SEEK:
            self._connect_seek_sdk()
        else:
            raise NotImplementedError(f"SDK for {camera_type.value} not implemented")
    
    def _connect_flir_sdk(self):
        """Connect to FLIR camera using their SDK."""
        # This is a placeholder for FLIR SDK implementation
        # In a real implementation, this would import and use the PySpin or similar library
        logger.debug("Connecting to FLIR camera using SDK")
        
        # Placeholder for FLIR SDK initialization
        # In production, this would use the actual FLIR SDK
        try:
            # Mock SDK initialization
            self._sdk_instance = {"type": "FLIR", "initialized": True}
            
            # In production code, this would be:
            # import PySpin
            # self._sdk_instance = PySpin.System.GetInstance()
            # cam_list = self._sdk_instance.GetCameras()
            # self._camera = cam_list.GetBySerial(self.source_info.address)
            # self._camera.Init()
            
            logger.debug("FLIR SDK connection established")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize FLIR SDK: {str(e)}")
    
    def _connect_seek_sdk(self):
        """Connect to Seek camera using their SDK."""
        # This is a placeholder for Seek SDK implementation
        logger.debug("Connecting to Seek camera using SDK")
        
        # Placeholder for Seek SDK initialization
        try:
            # Mock SDK initialization
            self._sdk_instance = {"type": "Seek", "initialized": True}
            
            # In production code, this would use the actual Seek SDK
            logger.debug("Seek SDK connection established")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Seek SDK: {str(e)}")
    
    def _connect_gige(self):
        """Connect to GigE Vision thermal camera."""
        ip_address = self.source_info.address
        logger.debug(f"Connecting to GigE thermal camera at {ip_address}")
        
        try:
            # Validate IP address
            ipaddress.ip_address(ip_address)
            
            # In production, this would use a GigE Vision SDK
            # Placeholder for GigE Vision implementation
            self._sdk_instance = {"type": "GigE", "ip": ip_address, "initialized": True}
            
            logger.debug(f"GigE Vision connection established to {ip_address}")
        except ValueError:
            raise ValueError(f"Invalid IP address for GigE camera: {ip_address}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to GigE camera: {str(e)}")
    
    def disconnect(self):
        """Disconnect from the thermal camera."""
        with self._lock:
            logger.info(f"Disconnecting from thermal source {self.source_info.name}")
            
            # Stop streaming thread if running
            self.stop_streaming()
            
            # Release resources based on connection type
            try:
                if self._stream is not None:
                    self._stream.release()
                    self._stream = None
                    
                if self._camera is not None:
                    # Camera-specific cleanup
                    self._camera = None
                    
                if self._sdk_instance is not None:
                    # SDK-specific cleanup
                    self._sdk_instance = None
                    
                self.status = ThermalSourceStatus.DISCONNECTED
                logger.info(f"Disconnected from {self.source_info.name}")
                
            except Exception as e:
                self.error_message = str(e)
                logger.error(f"Error during disconnect from {self.source_info.name}: {str(e)}", exc_info=True)
    
    def start_streaming(self) -> bool:
        """
        Start frame acquisition thread.
        
        Returns:
            True if streaming started successfully, False otherwise
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                logger.warning(f"Already streaming from {self.source_info.name}")
                return True
                
            if self.status != ThermalSourceStatus.CONNECTED:
                logger.error(f"Cannot start streaming from {self.source_info.name}: not connected")
                return False
                
            # Clear stop event and buffer
            self._stop_event.clear()
            while not self.frame_buffer.empty():
                try:
                    self.frame_buffer.get_nowait()
                except queue.Empty:
                    break
                    
            # Start acquisition thread
            self._thread = threading.Thread(
                target=self._acquisition_loop,
                name=f"thermal_acquisition_{self.source_info.id}",
                daemon=True
            )
            self._thread.start()
            
            self.status = ThermalSourceStatus.STREAMING
            logger.info(f"Started streaming from {self.source_info.name}")
            return True
    
    def stop_streaming(self):
        """Stop frame acquisition thread."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                return
                
            logger.info(f"Stopping streaming from {self.source_info.name}")
            
            # Signal thread to stop
            self._stop_event.set()
            
            # Wait for thread to finish (with timeout)
            self._thread.join(timeout=5.0)
            
            if self._thread.is_alive():
                logger.warning(f"Acquisition thread for {self.source_info.name} did not stop gracefully")
            
            self._thread = None
            
            if self.status == ThermalSourceStatus.STREAMING:
                self.status = ThermalSourceStatus.CONNECTED
                
            logger.info(f"Stopped streaming from {self.source_info.name}")
    
    def _acquisition_loop(self):
        """Main frame acquisition loop."""
        logger.debug(f"Starting acquisition loop for {self.source_info.name}")
        
        frame_interval = 1.0 / self.source_info.frame_rate if self.source_info.frame_rate > 0 else 0
        last_frame_time = 0
        fps_update_time = time.time()
        frames_since_update = 0
        
        try:
            while not self._stop_event.is_set():
                # Rate limiting if needed
                if frame_interval > 0:
                    time_since_last = time.time() - last_frame_time
                    if time_since_last < frame_interval:
                        time.sleep(frame_interval - time_since_last)
                
                # Acquire frame
                start_time = time.time()
                success, frame = self._acquire_frame()
                acquisition_time = time.time() - start_time
                
                # Update acquisition time statistics
                self._acquisition_times.append(acquisition_time)
                if len(self._acquisition_times) > 100:
                    self._acquisition_times.pop(0)
                
                if not success:
                    logger.warning(f"Failed to acquire frame from {self.source_info.name}")
                    self.dropped_frames += 1
                    
                    # Brief pause to avoid CPU spinning on errors
                    time.sleep(0.1)
                    continue
                
                # Process frame
                start_time = time.time()
                processed_frame = self._process_frame(frame)
                processing_time = time.time() - start_time
                
                # Update processing time statistics
                self._processing_times.append(processing_time)
                if len(self._processing_times) > 100:
                    self._processing_times.pop(0)
                
                # Add to buffer (drop oldest if full)
                if self.frame_buffer.full():
                    try:
                        self.frame_buffer.get_nowait()
                        self.dropped_frames += 1
                    except queue.Empty:
                        pass
                        
                try:
                    self.frame_buffer.put(processed_frame, block=False)
                    self.frame_count += 1
                    last_frame_time = time.time()
                    self.last_frame_time = last_frame_time
                    frames_since_update += 1
                except queue.Full:
                    self.dropped_frames += 1
                
                # Update FPS calculation every second
                if time.time() - fps_update_time > 1.0:
                    self.current_fps = frames_since_update / (time.time() - fps_update_time)
                    fps_update_time = time.time()
                    frames_since_update = 0
                    
                    # Log performance stats periodically
                    logger.debug(
                        f"{self.source_info.name}: FPS={self.current_fps:.1f}, "
                        f"Buffer={self.frame_buffer.qsize()}/{self.buffer_size}, "
                        f"Dropped={self.dropped_frames}, "
                        f"Acq={np.mean(self._acquisition_times)*1000:.1f}ms, "
                        f"Proc={np.mean(self._processing_times)*1000:.1f}ms"
                    )
                
        except Exception as e:
            logger.error(f"Error in acquisition loop for {self.source_info.name}: {str(e)}", exc_info=True)
            with self._lock:
                self.status = ThermalSourceStatus.ERROR
                self.error_message = str(e)
        finally:
            logger.info(f"Acquisition loop ended for {self.source_info.name}")
    
    def _acquire_frame(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Acquire a raw frame from the thermal source.
        
        Returns:
            Tuple of (success, frame_data)
        """
        # Different acquisition methods based on connection type
        if self._stream is not None:
            # OpenCV VideoCapture method
            ret, frame = self._stream.read()
            if not ret or frame is None:
                return False, None
                
            # Extract frame dimensions
            height, width = frame.shape[:2]
            
            # For thermal cameras via standard capture, we need to determine if
            # this is raw thermal data or already processed
            # In a real implementation, this would depend on the camera
            is_colorized = len(frame.shape) == 3 and frame.shape[2] == 3
            
            if is_colorized:
                # This appears to be a colorized thermal image
                # We need to convert it to temperature data (estimation)
                # In reality, many cameras provide both streams or additional metadata
                
                # Convert to grayscale for intensity estimation
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Estimate temperature from intensity (simplified example)
                temp_range = self.source_info.temperature_range
                temp_min, temp_max = temp_range
                
                # Linear mapping from intensity to temperature
                # This is a simplified approach - real thermal cameras provide calibration
                normalized = gray.astype(np.float32) / 255.0
                temperature_data = temp_min + normalized * (temp_max - temp_min)
                
                # Create frame data dictionary
                frame_data = {
                    'raw_data': gray,
                    'temperature_data': temperature_data,
                    'colorized': frame,
                    'width': width,
                    'height': height,
                    'is_estimated': True
                }
            else:
                # This appears to be raw/grayscale thermal data
                # Similarly estimate temperature
                raw = frame.astype(np.float32)
                
                # Normalize and map to temperature range
                temp_range = self.source_info.temperature_range
                temp_min, temp_max = temp_range
                
                # Linear mapping with different scaling (assuming 16-bit data)
                if raw.max() > 255:
                    # Likely 16-bit data
                    normalized = raw / 65535.0
                else:
                    # Likely 8-bit data
                    normalized = raw / 255.0
                    
                temperature_data = temp_min + normalized * (temp_max - temp_min)
                
                # Create frame data dictionary
                frame_data = {
                    'raw_data': raw,
                    'temperature_data': temperature_data,
                    'colorized': None,
                    'width': width,
                    'height': height,
                    'is_estimated': True
                }
                
            return True, frame_data
            
        elif self._sdk_instance is not None:
            # SDK-specific acquisition
            if self._sdk_instance["type"] == "FLIR":
                return self._acquire_frame_flir()
            elif self._sdk_instance["type"] == "Seek":
                return self._acquire_frame_seek()
            elif self._sdk_instance["type"] == "GigE":
                return self._acquire_frame_gige()
            else:
                logger.error(f"Unknown SDK type: {self._sdk_instance['type']}")
                return False, None
        else:
            logger.error("No valid stream or SDK instance available")
            return False, None
    
    def _acquire_frame_flir(self) -> Tuple[bool, Dict[str, Any]]:
        """Acquire frame from FLIR camera using SDK."""
        # This is a placeholder for FLIR SDK implementation
        # In a real implementation, this would use the PySpin or similar library
        
        # Simulate frame acquisition
        width, height = self.source_info.resolution
        
        # Create simulated thermal data
        # In a real implementation, this would come from the FLIR SDK
        raw_data = np.random.randint(0, 65536, (height, width), dtype=np.uint16)
        
        # Convert to temperature values (simulated)
        # FLIR cameras typically provide this conversion via SDK
        temperature_data = self._simulate_temperature_data(width, height)
        
        # Create frame data
        frame_data = {
            'raw_data': raw_data,
            'temperature_data': temperature_data,
            'colorized': None,
            'width': width,
            'height': height,
            'is_estimated': False  # Not estimated, would come directly from SDK
        }
        
        return True, frame_data
    
    def _acquire_frame_seek(self) -> Tuple[bool, Dict[str, Any]]:
        """Acquire frame from Seek camera using SDK."""
        # This is a placeholder for Seek SDK implementation
        
        # Simulate frame acquisition similar to FLIR method
        width, height = self.source_info.resolution
        raw_data = np.random.randint(0, 65536, (height, width), dtype=np.uint16)
        temperature_data = self._simulate_temperature_data(width, height)
        
        frame_data = {
            'raw_data': raw_data,
            'temperature_data': temperature_data,
            'colorized': None,
            'width': width,
            'height': height,
            'is_estimated': False
        }
        
        return True, frame_data
    
    def _acquire_frame_gige(self) -> Tuple[bool, Dict[str, Any]]:
        """Acquire frame from GigE Vision camera."""
        # This is a placeholder for GigE Vision SDK implementation
        
        # Simulate frame acquisition similar to other methods
        width, height = self.source_info.resolution
        raw_data = np.random.randint(0, 65536, (height, width), dtype=np.uint16)
        temperature_data = self._simulate_temperature_data(width, height)
        
        frame_data = {
            'raw_data': raw_data,
            'temperature_data': temperature_data,
            'colorized': None,
            'width': width,
            'height': height,
            'is_estimated': False
        }
        
        return True, frame_data
    
    def _simulate_temperature_data(self, width: int, height: int) -> np.ndarray:
        """
        Generate simulated temperature data for testing.
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            Temperature data as numpy array
        """
        # Create base temperature around ambient
        ambient = 22.0  # Ambient temperature in Celsius
        
        # Create temperature gradient
        y, x = np.mgrid[0:height, 0:width]
        temperature = ambient + 5 * np.sin(x/width * np.pi) + 3 * np.cos(y/height * np.pi)
        
        # Add some hot spots
        num_hotspots = np.random.randint(1, 5)
        for _ in range(num_hotspots):
            cx = np.random.randint(0, width)
            cy = np.random.randint(0, height)
            intensity = np.random.uniform(5, 15)
            radius = np.random.uniform(10, 50)
            
            # Create hotspot using Gaussian
            xx, yy = np.meshgrid(np.arange(width), np.arange(height))
            distance = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            hotspot = intensity * np.exp(-distance**2 / (2 * radius**2))
            temperature += hotspot
        
        # Add some noise
        noise = np.random.normal(0, 0.5, (height, width))
        temperature += noise
        
        return temperature
    
    def _process_frame(self, frame_data: Dict[str, Any]) -> ThermalFrame:
        """
        Process raw frame data into a ThermalFrame.
        
        Args:
            frame_data: Raw frame data from _acquire_frame
            
        Returns:
            Processed ThermalFrame
        """
        # Extract data from frame_data
        raw_data = frame_data['raw_data']
        temperature_data = frame_data['temperature_data']
        colorized = frame_data.get('colorized')
        width = frame_data['width']
        height = frame_data['height']
        
        # Apply calibration if available
        if self.calibration_matrix is not None:
            # Apply calibration to temperature data
            # This is a simplified example - real calibration would be more complex
            temperature_data = temperature_data * self.calibration_matrix
            calibration_applied = True
        else:
            calibration_applied = False
        
        # Calculate temperature statistics
        min_temp = float(np.min(temperature_data))
        max_temp = float(np.max(temperature_data))
        avg_temp = float(np.mean(temperature_data))
        
        # Create ThermalFrame
        frame = ThermalFrame(
            raw_data=raw_data,
            temperature_data=temperature_data,
            source_id=self.source_info.id,
            timestamp=time.time(),
            sequence_num=self.frame_count,
            emissivity=self.source_info.emissivity,
            distance=self.source_info.distance,
            ambient_temp=22.0,  # This would come from the camera in a real implementation
            min_temp=min_temp,
            max_temp=max_temp,
            avg_temp=avg_temp,
            resolution=(width, height),
            format=ThermalDataFormat.TEMPERATURE,
            colorized=colorized,
            color_palette=self.source_info.color_palette,
            processed=True,
            calibration_applied=calibration_applied
        )
        
        # Generate colorized version if not already present
        if colorized is None:
            frame.colorize()
        
        return frame
    
    def get_frame(self, timeout: float = 1.0) -> Optional[ThermalFrame]:
        """
        Get the next available frame from the buffer.
        
        Args:
            timeout: Maximum time to wait for a frame (seconds)
            
        Returns:
            ThermalFrame if available, None otherwise
        """
        if self.status != ThermalSourceStatus.STREAMING:
            logger.warning(f"Cannot get frame: {self.source_info.name} is not streaming")
            return None
            
        try:
            return self.frame_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def calibrate(self, reference_temp: float = None) -> bool:
        """
        Calibrate temperature readings using reference temperature.
        
        Args:
            reference_temp: Optional reference temperature in Celsius
            
        Returns:
            True if calibration successful, False otherwise
        """
        with self._lock:
            if self.status != ThermalSourceStatus.CONNECTED and self.status != ThermalSourceStatus.STREAMING:
                logger.warning(f"Cannot calibrate: {self.source_info.name} is not connected")
                return False
                
            prev_status = self.status
            self.status = ThermalSourceStatus.CALIBRATING
            
            try:
                logger.info(f"Calibrating {self.source_info.name}")
                
                # Get a frame for calibration
                if self.status != ThermalSourceStatus.STREAMING:
                    # Temporarily start streaming to get a frame
                    was_streaming = False
                    self.start_streaming()
                else:
                    was_streaming = True
                    
                # Wait for a frame
                frame = self.get_frame(timeout=5.0)
                
                if frame is None:
                    raise ValueError("Failed to get frame for calibration")
                
                # Stop streaming if we started it
                if not was_streaming:
                    self.stop_streaming()
                
                # Perform calibration
                if reference_temp is not None:
                    # Use reference temperature for calibration
                    # This is a simplified approach - real calibration would be more complex
                    current_avg = frame.avg_temp
                    scaling_factor = reference_temp / current_avg
                    
                    # Create calibration matrix (simple scaling in this example)
                    self.calibration_matrix = np.ones_like(frame.temperature_data) * scaling_factor
                else:
                    # Auto-calibration based on black body in image or similar
                    # This is a placeholder - real implementation would be camera-specific
                    self.calibration_matrix = np.ones_like(frame.temperature_data)
                
                self.source_info.calibrated = True
                logger.info(f"Calibration of {self.source_info.name} completed")
                
                # Restore previous status
                self.status = prev_status
                return True
                
            except Exception as e:
                self.status = ThermalSourceStatus.ERROR
                self.error_message = f"Calibration error: {str(e)}"
                logger.error(f"Calibration failed for {self.source_info.name}: {str(e)}", exc_info=True)
                return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get detailed status information about the connection.
        
        Returns:
            Dictionary with status information
        """
        with self._lock:
            status_info = {
                "source_id": self.source_info.id,
                "name": self.source_info.name,
                "status": self.status.value,
                "error": self.error_message if self.error_message else None,
                "connection_time": self.connection_time,
                "uptime": time.time() - self.connection_time if self.connection_time > 0 else 0,
                "frame_count": self.frame_count,
                "dropped_frames": self.dropped_frames,
                "current_fps": self.current_fps,
                "last_frame_time": self.last_frame_time,
                "buffer_size": self.buffer_size,
                "buffer_used": self.frame_buffer.qsize(),
                "calibrated": self.source_info.calibrated,
                "avg_acquisition_time": np.mean(self._acquisition_times) if self._acquisition_times else 0,
                "avg_processing_time": np.mean(self._processing_times) if self._processing_times else 0
            }
            
            return status_info


class ThermalCollectionManager:
    """
    Manages collection of thermal data from multiple sources.
    
    This class provides:
    - Discovery and management of thermal camera sources
    - Connection establishment and monitoring
    - Frame acquisition and distribution
    - Temperature calibration
    - Integration with the UltraTrack tracking system
    """
    
    def __init__(self, config=None):
        """
        Initialize the thermal collection manager.
        
        Args:
            config: Configuration dictionary or None to use system config
        """
        # Load configuration
        if config is None:
            self.config = ConfigManager.get_config().data_collection.thermal
        else:
            self.config = config
            
        # Initialize state
        self.sources = {}  # Dict[source_id, ThermalSourceInfo]
        self.connections = {}  # Dict[source_id, ThermalSourceConnection]
        self.frame_callbacks = []  # Callbacks for new frames
        self.is_running = False
        self._discovery_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        
        # Initialize data validator
        self.validator = DataValidator()
        
        # Track statistics
        self.stats = {
            "total_frames": 0,
            "dropped_frames": 0,
            "sources_count": 0,
            "active_connections": 0,
            "start_time": 0,
            "last_discovery": 0
        }
        
        logger.info("Thermal collection manager initialized")
    
    def start(self):
        """Start the thermal collection manager."""
        with self._lock:
            if self.is_running:
                logger.warning("Thermal collection manager already running")
                return
                
            logger.info("Starting thermal collection manager")
            
            # Reset stop event
            self._stop_event.clear()
            
            # Reset statistics
            self.stats["total_frames"] = 0
            self.stats["dropped_frames"] = 0
            self.stats["start_time"] = time.time()
            
            # Load saved sources if available
            self._load_sources()
            
            # Start automatic discovery if enabled
            if self.config.enabled and self.config.network_discovery:
                self._start_discovery()
                
            # Connect to known sources if auto-connect enabled
            if self.config.enabled and self.config.auto_connect:
                for source_id, source_info in self.sources.items():
                    self.connect_source(source_id)
            
            self.is_running = True
            logger.info("Thermal collection manager started")
    
    def stop(self):
        """Stop the thermal collection manager."""
        with self._lock:
            if not self.is_running:
                return
                
            logger.info("Stopping thermal collection manager")
            
            # Signal discovery thread to stop
            self._stop_event.set()
            
            # Stop discovery thread if running
            if self._discovery_thread and self._discovery_thread.is_alive():
                self._discovery_thread.join(timeout=5.0)
                
            # Disconnect all sources
            for source_id in list(self.connections.keys()):
                self.disconnect_source(source_id)
                
            self.is_running = False
            logger.info("Thermal collection manager stopped")
    
    def shutdown(self):
        """Shut down the thermal collection manager."""
        logger.info("Shutting down thermal collection manager")
        self.stop()
        
        # Clear all resources
        with self._lock:
            self.sources.clear()
            self.connections.clear()
            self.frame_callbacks.clear()
            
        logger.info("Thermal collection manager shut down")
    
    def _load_sources(self):
        """Load saved thermal sources from storage."""
        try:
            # Check if sources file exists
            sources_path = os.path.join(
                self.config.local_storage_path, 
                "thermal_sources.json"
            )
            
            if not os.path.exists(sources_path):
                logger.info("No saved thermal sources found")
                return
                
            with open(sources_path, 'r') as f:
                sources_data = json.load(f)
                
            # Convert dictionaries to ThermalSourceInfo objects
            for source_data in sources_data:
                try:
                    source_info = ThermalSourceInfo.from_dict(source_data)
                    self.sources[source_info.id] = source_info
                except Exception as e:
                    logger.warning(f"Failed to load thermal source: {str(e)}")
                    
            self.stats["sources_count"] = len(self.sources)
            logger.info(f"Loaded {len(self.sources)} thermal sources")
            
        except Exception as e:
            logger.error(f"Error loading thermal sources: {str(e)}", exc_info=True)
    
    def _save_sources(self):
        """Save thermal sources to persistent storage."""
        try:
            # Ensure directory exists
            os.makedirs(self.config.local_storage_path, exist_ok=True)
            
            # Convert ThermalSourceInfo objects to dictionaries
            sources_data = [source_info.to_dict() for source_info in self.sources.values()]
            
            # Write to file
            sources_path = os.path.join(
                self.config.local_storage_path, 
                "thermal_sources.json"
            )
            
            with open(sources_path, 'w') as f:
                json.dump(sources_data, f, indent=2)
                
            logger.debug(f"Saved {len(self.sources)} thermal sources")
            
        except Exception as e:
            logger.error(f"Error saving thermal sources: {str(e)}", exc_info=True)
    
    def _start_discovery(self):
        """Start thermal camera discovery thread."""
        if self._discovery_thread and self._discovery_thread.is_alive():
            logger.warning("Discovery thread already running")
            return
            
        logger.info("Starting thermal camera discovery")
        
        self._discovery_thread = threading.Thread(
            target=self._discovery_loop,
            name="thermal_discovery",
            daemon=True
        )
        self._discovery_thread.start()
    
    def _discovery_loop(self):
        """Main discovery loop for thermal cameras."""
        logger.debug("Thermal camera discovery loop started")
        
        discovery_interval = self.config.discovery_interval_s
        
        while not self._stop_event.is_set():
            try:
                # Perform discovery
                self._discover_cameras()
                
                # Update last discovery time
                self.stats["last_discovery"] = time.time()
                
                # Wait for next discovery interval or stop event
                self._stop_event.wait(discovery_interval)
                
            except Exception as e:
                logger.error(f"Error in thermal camera discovery: {str(e)}", exc_info=True)
                # Wait a bit before retrying after error
                time.sleep(5)
    
    def _discover_cameras(self):
        """Discover available thermal cameras on the network."""
        logger.debug("Performing thermal camera discovery")
        
        # In a real implementation, this would use various discovery methods:
        # - ONVIF discovery
        # - UPnP/SSDP discovery
        # - Scanning IP ranges
        # - SDK-specific discovery
        
        # This is a simplified implementation using manually specified sources
        discovered = []
        
        # Check for RTSP thermal cameras on the network
        discovered.extend(self._discover_rtsp_cameras())
        
        # Check for USB thermal cameras
        discovered.extend(self._discover_usb_cameras())
        
        # Check for vendor-specific cameras using SDKs
        discovered.extend(self._discover_sdk_cameras())
        
        # Process discovered cameras
        for source_info in discovered:
            self.add_source(source_info)
            
        logger.debug(f"Discovery found {len(discovered)} thermal cameras")
    
    def _discover_rtsp_cameras(self) -> List[ThermalSourceInfo]:
        """
        Discover RTSP thermal cameras on the network.
        
        Returns:
            List of discovered ThermalSourceInfo objects
        """
        # This is a placeholder for actual RTSP camera discovery
        # In a real implementation, this would:
        # - Scan the network using ONVIF discovery
        # - Check known thermal camera models
        # - Test connections to potential thermal cameras
        
        discovered = []
        
        # Placeholder for discovery process
        # In this simplified version, we might add simulated devices
        # for demonstration/testing purposes
        if not self.sources:  # Only add demo source if no sources exist
            # Add a simulated RTSP thermal camera
            source_info = ThermalSourceInfo(
                id="demo_thermal_rtsp",
                name="Demo RTSP Thermal Camera",
                type=ThermalCameraType.FLIR,
                protocol=ThermalStreamProtocol.RTSP,
                address="10.0.0.100",
                port=554,
                path="/thermal",
                temperature_range=(-20.0, 150.0),
                resolution=(640, 480),
                frame_rate=9,
                color_palette=ThermalColorPalette.IRON
            )
            discovered.append(source_info)
            
        return discovered
    
    def _discover_usb_cameras(self) -> List[ThermalSourceInfo]:
        """
        Discover USB thermal cameras.
        
        Returns:
            List of discovered ThermalSourceInfo objects
        """
        # This is a placeholder for actual USB camera discovery
        # In a real implementation, this would:
        # - Enumerate USB devices
        # - Check for known thermal camera vendors/models
        # - Test connections to potential thermal cameras
        
        discovered = []
        
        # Add a simulated USB thermal camera for testing
        if not any(s.protocol == ThermalStreamProtocol.USB for s in self.sources.values()):
            source_info = ThermalSourceInfo(
                id="demo_thermal_usb",
                name="Demo USB Thermal Camera",
                type=ThermalCameraType.SEEK,
                protocol=ThermalStreamProtocol.USB,
                address="USB",
                path="0",
                temperature_range=(-20.0, 150.0),
                resolution=(320, 240),
                frame_rate=9,
                color_palette=ThermalColorPalette.RAINBOW
            )
            discovered.append(source_info)
            
        return discovered
    
    def _discover_sdk_cameras(self) -> List[ThermalSourceInfo]:
        """
        Discover thermal cameras using vendor-specific SDKs.
        
        Returns:
            List of discovered ThermalSourceInfo objects
        """
        # This is a placeholder for SDK-based camera discovery
        # In a real implementation, this would use vendor-specific SDKs
        # to discover available cameras (e.g., FLIR, Seek, etc.)
        
        discovered = []
        
        # Add a simulated SDK thermal camera for testing
        if not any(s.protocol == ThermalStreamProtocol.SDK for s in self.sources.values()):
            source_info = ThermalSourceInfo(
                id="demo_thermal_sdk",
                name="Demo FLIR SDK Camera",
                type=ThermalCameraType.FLIR,
                protocol=ThermalStreamProtocol.SDK,
                address="192.168.1.100",
                sdk_params={"serial": "FLIR123456", "model": "FLIR A65"},
                temperature_range=(-20.0, 550.0),
                resolution=(640, 512),
                frame_rate=30,
                color_palette=ThermalColorPalette.IRON
            )
            discovered.append(source_info)
            
        return discovered
    
    def add_source(self, source_info: ThermalSourceInfo) -> bool:
        """
        Add a thermal source to the manager.
        
        Args:
            source_info: Information about the thermal source
            
        Returns:
            True if source was added, False if already exists
        """
        with self._lock:
            if source_info.id in self.sources:
                logger.debug(f"Thermal source {source_info.id} already exists")
                return False
                
            logger.info(f"Adding thermal source: {source_info.name} ({source_info.id})")
            self.sources[source_info.id] = source_info
            self.stats["sources_count"] = len(self.sources)
            
            # Save updated sources
            self._save_sources()
            
            return True
    
    def remove_source(self, source_id: str) -> bool:
        """
        Remove a thermal source from the manager.
        
        Args:
            source_id: ID of the source to remove
            
        Returns:
            True if source was removed, False if not found
        """
        with self._lock:
            if source_id not in self.sources:
                logger.warning(f"Cannot remove: thermal source {source_id} not found")
                return False
                
            # Disconnect if connected
            if source_id in self.connections:
                self.disconnect_source(source_id)
                
            # Remove source
            source_info = self.sources.pop(source_id)
            self.stats["sources_count"] = len(self.sources)
            
            logger.info(f"Removed thermal source: {source_info.name} ({source_id})")
            
            # Save updated sources
            self._save_sources()
            
            return True
    
    def update_source(self, source_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a thermal source's information.
        
        Args:
            source_id: ID of the source to update
            updates: Dictionary of attributes to update
            
        Returns:
            True if source was updated, False if not found
        """
        with self._lock:
            if source_id not in self.sources:
                logger.warning(f"Cannot update: thermal source {source_id} not found")
                return False
                
            source_info = self.sources[source_id]
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(source_info, key):
                    # Special handling for enum types
                    if key == 'type' and not isinstance(value, ThermalCameraType):
                        value = ThermalCameraType(value)
                    elif key == 'protocol' and not isinstance(value, ThermalStreamProtocol):
                        value = ThermalStreamProtocol(value)
                    elif key == 'color_palette' and not isinstance(value, ThermalColorPalette):
                        value = ThermalColorPalette(value)
                        
                    setattr(source_info, key, value)
                else:
                    logger.warning(f"Unknown attribute {key} for thermal source")
            
            logger.info(f"Updated thermal source: {source_info.name} ({source_id})")
            
            # If connected, we may need to reconnect
            if source_id in self.connections:
                logger.info(f"Reconnecting {source_id} after update")
                self.disconnect_source(source_id)
                self.connect_source(source_id)
            
            # Save updated sources
            self._save_sources()
            
            return True
    
    def get_source(self, source_id: str) -> Optional[ThermalSourceInfo]:
        """
        Get information about a thermal source.
        
        Args:
            source_id: ID of the source to retrieve
            
        Returns:
            ThermalSourceInfo if found, None otherwise
        """
        with self._lock:
            return self.sources.get(source_id)
    
    def get_all_sources(self) -> List[ThermalSourceInfo]:
        """
        Get all registered thermal sources.
        
        Returns:
            List of all ThermalSourceInfo objects
        """
        with self._lock:
            return list(self.sources.values())
    
    def connect_source(self, source_id: str) -> bool:
        """
        Connect to a thermal source.
        
        Args:
            source_id: ID of the source to connect to
            
        Returns:
            True if connection successful, False otherwise
        """
        with self._lock:
            if not self.is_running:
                logger.warning("Cannot connect: thermal collection manager not running")
                return False
                
            if source_id not in self.sources:
                logger.warning(f"Cannot connect: thermal source {source_id} not found")
                return False
                
            if source_id in self.connections:
                logger.debug(f"Thermal source {source_id} already connected")
                return True
                
            # Get source info
            source_info = self.sources[source_id]
            
            # Create connection
            logger.info(f"Connecting to thermal source: {source_info.name} ({source_id})")
            connection = ThermalSourceConnection(
                source_info, 
                buffer_size=self.config.frame_buffer_size
            )
            
            # Attempt to connect
            if not connection.connect():
                logger.error(f"Failed to connect to thermal source: {source_info.name}")
                return False
                
            # Store connection
            self.connections[source_id] = connection
            self.stats["active_connections"] = len(self.connections)
            
            # Start streaming if auto-stream enabled
            if self.config.auto_stream:
                connection.start_streaming()
                
            logger.info(f"Connected to thermal source: {source_info.name}")
            return True
    
    def disconnect_source(self, source_id: str) -> bool:
        """
        Disconnect from a thermal source.
        
        Args:
            source_id: ID of the source to disconnect from
            
        Returns:
            True if disconnection successful, False if not connected
        """
        with self._lock:
            if source_id not in self.connections:
                logger.debug(f"Thermal source {source_id} not connected")
                return False
                
            # Get connection
            connection = self.connections[source_id]
            
            # Disconnect
            logger.info(f"Disconnecting from thermal source: {connection.source_info.name}")
            connection.disconnect()
            
            # Remove connection
            del self.connections[source_id]
            self.stats["active_connections"] = len(self.connections)
            
            logger.info(f"Disconnected from thermal source: {connection.source_info.name}")
            return True
    
    def start_streaming(self, source_id: str) -> bool:
        """
        Start streaming from a thermal source.
        
        Args:
            source_id: ID of the source to stream from
            
        Returns:
            True if streaming started, False if not connected
        """
        with self._lock:
            if source_id not in self.connections:
                logger.warning(f"Cannot start streaming: thermal source {source_id} not connected")
                return False
                
            # Get connection
            connection = self.connections[source_id]
            
            # Start streaming
            return connection.start_streaming()
    
    def stop_streaming(self, source_id: str) -> bool:
        """
        Stop streaming from a thermal source.
        
        Args:
            source_id: ID of the source to stop streaming from
            
        Returns:
            True if streaming stopped, False if not streaming
        """
        with self._lock:
            if source_id not in self.connections:
                logger.warning(f"Cannot stop streaming: thermal source {source_id} not connected")
                return False
                
            # Get connection
            connection = self.connections[source_id]
            
            # Stop streaming
            connection.stop_streaming()
            return True
    
    def get_frame(self, source_id: str, timeout: float = 1.0) -> Optional[ThermalFrame]:
        """
        Get the next available frame from a thermal source.
        
        Args:
            source_id: ID of the source to get frame from
            timeout: Maximum time to wait for a frame (seconds)
            
        Returns:
            ThermalFrame if available, None otherwise
        """
        if source_id not in self.connections:
            logger.warning(f"Cannot get frame: thermal source {source_id} not connected")
            return None
            
        # Get connection
        connection = self.connections[source_id]
        
        # Get frame
        return connection.get_frame(timeout=timeout)
    
    def get_latest_frames(self, timeout: float = 0.1) -> Dict[str, ThermalFrame]:
        """
        Get the latest available frame from each streaming source.
        
        Args:
            timeout: Maximum time to wait for each frame (seconds)
            
        Returns:
            Dictionary mapping source_id to ThermalFrame
        """
        with self._lock:
            frames = {}
            
            for source_id, connection in self.connections.items():
                frame = connection.get_frame(timeout=timeout)
                if frame is not None:
                    frames[source_id] = frame
                    
            return frames
    
    def register_frame_callback(self, callback: Callable[[str, ThermalFrame], None]) -> int:
        """
        Register a callback function to be called when new frames are available.
        
        Args:
            callback: Function to call with (source_id, frame) arguments
            
        Returns:
            Callback ID for unregistering
        """
        with self._lock:
            callback_id = len(self.frame_callbacks)
            self.frame_callbacks.append(callback)
            logger.debug(f"Registered frame callback {callback_id}")
            return callback_id
    
    def unregister_frame_callback(self, callback_id: int) -> bool:
        """
        Unregister a previously registered frame callback.
        
        Args:
            callback_id: ID returned from register_frame_callback
            
        Returns:
            True if callback was unregistered, False if not found
        """
        with self._lock:
            if 0 <= callback_id < len(self.frame_callbacks):
                self.frame_callbacks.pop(callback_id)
                logger.debug(f"Unregistered frame callback {callback_id}")
                return True
            else:
                logger.warning(f"Callback ID {callback_id} not found")
                return False
    
    def calibrate_source(self, source_id: str, reference_temp: float = None) -> bool:
        """
        Calibrate temperature readings for a thermal source.
        
        Args:
            source_id: ID of the source to calibrate
            reference_temp: Optional reference temperature in Celsius
            
        Returns:
            True if calibration successful, False otherwise
        """
        with self._lock:
            if source_id not in self.connections:
                logger.warning(f"Cannot calibrate: thermal source {source_id} not connected")
                return False
                
            # Get connection
            connection = self.connections[source_id]
            
            # Perform calibration
            return connection.calibrate(reference_temp=reference_temp)
    
    def get_source_status(self, source_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status information about a thermal source connection.
        
        Args:
            source_id: ID of the source to get status for
            
        Returns:
            Dictionary with status information, or None if not connected
        """
        with self._lock:
            if source_id not in self.connections:
                return None
                
            # Get connection
            connection = self.connections[source_id]
            
            # Get status
            return connection.get_status()
    
    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all connected thermal sources.
        
        Returns:
            Dictionary mapping source_id to status information
        """
        with self._lock:
            return {
                source_id: connection.get_status()
                for source_id, connection in self.connections.items()
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics for the thermal collection manager.
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            # Update frame counts from connections
            total_frames = 0
            dropped_frames = 0
            
            for connection in self.connections.values():
                total_frames += connection.frame_count
                dropped_frames += connection.dropped_frames
                
            self.stats["total_frames"] = total_frames
            self.stats["dropped_frames"] = dropped_frames
            
            # Add derived statistics
            stats = self.stats.copy()
            stats["uptime"] = time.time() - stats["start_time"]
            stats["fps"] = total_frames / stats["uptime"] if stats["uptime"] > 0 else 0
            
            return stats
    
    def _process_frames_loop(self):
        """Background thread for processing frames and calling callbacks."""
        logger.debug("Frame processing loop started")
        
        executor = ThreadPoolExecutor(max_workers=max(1, os.cpu_count() // 2))
        
        while not self._stop_event.is_set():
            try:
                # Get latest frames from all sources
                frames = self.get_latest_frames(timeout=0.05)
                
                # Skip if no frames or callbacks
                if not frames or not self.frame_callbacks:
                    time.sleep(0.01)  # Small sleep to avoid CPU spinning
                    continue
                    
                # Process frames and call callbacks
                for source_id, frame in frames.items():
                    # Submit callback tasks to thread pool
                    for callback in self.frame_callbacks:
                        executor.submit(callback, source_id, frame)
                        
            except Exception as e:
                logger.error(f"Error in frame processing loop: {str(e)}", exc_info=True)
                time.sleep(1)  # Longer sleep after error
                
        # Shutdown executor
        executor.shutdown(wait=False)
        logger.debug("Frame processing loop ended")
