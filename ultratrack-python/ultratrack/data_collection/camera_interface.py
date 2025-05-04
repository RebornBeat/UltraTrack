"""
UltraTrack Camera Interface Module

This module provides interfaces for connecting to and managing diverse camera networks:
- Public, private, government, and commercial camera systems
- Various connection protocols (RTSP, HTTP, etc.)
- Camera discovery and registration
- Video stream processing
- Quality monitoring and management

Copyright (c) 2025 Your Organization
"""

import cv2
import os
import logging
import threading
import time
import uuid
import ipaddress
import json
import queue
import numpy as np
import requests
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse

# Setup module logger
logger = logging.getLogger(__name__)

# Define enumerations
class CameraType(Enum):
    """Camera type classification."""
    PUBLIC = auto()         # Public safety, traffic cameras
    PRIVATE = auto()        # Private security systems
    GOVERNMENT = auto()     # Government/secured facilities
    COMMERCIAL = auto()     # Commercial establishments
    RESIDENTIAL = auto()    # Home security systems
    MOBILE = auto()         # Body cameras, dashcams
    DRONE = auto()          # Aerial drones
    SATELLITE = auto()      # Satellite imagery
    SPECIALIZED = auto()    # Specialized imaging systems
    UNKNOWN = auto()        # Unclassified sources

class StreamProtocol(Enum):
    """Camera streaming protocol."""
    RTSP = auto()           # Real Time Streaming Protocol
    RTMP = auto()           # Real Time Messaging Protocol
    HTTP = auto()           # HTTP streams (HLS, DASH)
    HTTPS = auto()          # Secure HTTP streams
    ONVIF = auto()          # ONVIF standards
    SDK = auto()            # Vendor-specific SDK
    FILE = auto()           # File-based input
    SPECIALIZED = auto()    # Specialized protocols
    MJPEG = auto()          # Motion JPEG
    CUSTOM = auto()         # Custom streaming protocol

class CameraStatus(Enum):
    """Camera connection status."""
    ONLINE = auto()         # Connected and streaming
    OFFLINE = auto()        # Disconnected
    RECONNECTING = auto()   # Attempting to reconnect
    ERROR = auto()          # Error state
    DEGRADED = auto()       # Connected but with quality issues
    INITIALIZING = auto()   # Connection being established
    DISABLED = auto()       # Administratively disabled
    UNKNOWN = auto()        # Status cannot be determined

class PTZCapability(Enum):
    """Pan-Tilt-Zoom capabilities."""
    NONE = auto()           # No PTZ capability
    PAN_TILT = auto()       # Pan and tilt only
    ZOOM_ONLY = auto()      # Zoom only
    FULL_PTZ = auto()       # Full pan, tilt, zoom
    PRESET = auto()         # Supports position presets
    PATROL = auto()         # Supports patrol routes
    ADVANCED = auto()       # Advanced PTZ features

# Data structures
@dataclass
class GeographicLocation:
    """Geographic location information."""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    accuracy_meters: Optional[float] = None
    timestamp: Optional[datetime] = None
    address: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None

@dataclass
class CameraMetadata:
    """Camera metadata information."""
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    firmware_version: Optional[str] = None
    installation_date: Optional[datetime] = None
    last_maintenance: Optional[datetime] = None
    resolution: Optional[Tuple[int, int]] = None
    fps: Optional[int] = None
    field_of_view: Optional[float] = None
    focal_length: Optional[float] = None
    ptz_capability: PTZCapability = PTZCapability.NONE
    infrared_capability: bool = False
    audio_capability: bool = False
    storage_capability: bool = False
    connection_info: Optional[Dict[str, Any]] = None
    additional_info: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        result = {
            'manufacturer': self.manufacturer,
            'model': self.model,
            'firmware_version': self.firmware_version,
            'resolution': self.resolution,
            'fps': self.fps,
            'field_of_view': self.field_of_view,
            'focal_length': self.focal_length,
            'ptz_capability': self.ptz_capability.name if self.ptz_capability else None,
            'infrared_capability': self.infrared_capability,
            'audio_capability': self.audio_capability,
            'storage_capability': self.storage_capability,
        }
        
        # Add installation date if exists
        if self.installation_date:
            result['installation_date'] = self.installation_date.isoformat()
            
        # Add last maintenance if exists
        if self.last_maintenance:
            result['last_maintenance'] = self.last_maintenance.isoformat()
            
        # Add connection info if exists
        if self.connection_info:
            # Filter out sensitive information
            filtered_conn = {k: v for k, v in self.connection_info.items() 
                            if k not in ('password', 'token', 'api_key', 'secret')}
            result['connection_info'] = filtered_conn
            
        # Add additional info if exists
        if self.additional_info:
            result['additional_info'] = self.additional_info
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraMetadata':
        """Create metadata from dictionary."""
        # Process special fields
        if 'installation_date' in data and data['installation_date']:
            data['installation_date'] = datetime.fromisoformat(data['installation_date'])
            
        if 'last_maintenance' in data and data['last_maintenance']:
            data['last_maintenance'] = datetime.fromisoformat(data['last_maintenance'])
            
        if 'ptz_capability' in data and data['ptz_capability']:
            data['ptz_capability'] = PTZCapability[data['ptz_capability']]
            
        return cls(**data)

@dataclass
class CameraCredentials:
    """Secure camera access credentials."""
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    api_key: Optional[str] = None
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    auth_method: Optional[str] = None
    
    def validate(self) -> bool:
        """Validate that sufficient credentials are provided."""
        # Username/password authentication
        if self.username and self.password:
            return True
            
        # Token-based authentication
        if self.token:
            return True
            
        # API key authentication
        if self.api_key:
            return True
            
        # Certificate-based authentication
        if self.certificate_path and self.private_key_path:
            # Verify files exist
            if not os.path.exists(self.certificate_path):
                logger.error(f"Certificate file not found: {self.certificate_path}")
                return False
                
            if not os.path.exists(self.private_key_path):
                logger.error(f"Private key file not found: {self.private_key_path}")
                return False
                
            return True
            
        # No valid credentials
        return False

@dataclass
class CameraCalibration:
    """Camera calibration information."""
    intrinsic_matrix: Optional[np.ndarray] = None
    distortion_coefficients: Optional[np.ndarray] = None
    extrinsic_matrix: Optional[np.ndarray] = None
    calibration_date: Optional[datetime] = None
    pixel_to_meter_ratio: Optional[float] = None
    calibration_quality: Optional[float] = None
    reference_objects: Optional[List[Dict[str, Any]]] = None
    
    def is_calibrated(self) -> bool:
        """Check if camera is calibrated."""
        return self.intrinsic_matrix is not None and self.distortion_coefficients is not None

@dataclass
class CameraSource:
    """Camera source definition."""
    id: str
    name: str
    url: str
    type: CameraType
    protocol: StreamProtocol
    status: CameraStatus = CameraStatus.UNKNOWN
    location: Optional[GeographicLocation] = None
    metadata: Optional[CameraMetadata] = None
    credentials: Optional[CameraCredentials] = None
    calibration: Optional[CameraCalibration] = None
    coverage_area: Optional[List[Dict[str, float]]] = None
    access_level: str = "public"
    tags: List[str] = field(default_factory=list)
    last_connected: Optional[datetime] = None
    last_status_change: Optional[datetime] = None
    connection_attempts: int = 0
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert camera source to dictionary for storage."""
        result = {
            'id': self.id,
            'name': self.name,
            'url': self.url,
            'type': self.type.name,
            'protocol': self.protocol.name,
            'status': self.status.name,
            'access_level': self.access_level,
            'tags': self.tags,
            'enabled': self.enabled,
            'connection_attempts': self.connection_attempts
        }
        
        # Add location if exists
        if self.location:
            result['location'] = {
                'latitude': self.location.latitude,
                'longitude': self.location.longitude,
                'altitude': self.location.altitude,
                'accuracy_meters': self.location.accuracy_meters,
                'address': self.location.address,
                'region': self.location.region,
                'country': self.location.country
            }
            
        # Add metadata if exists
        if self.metadata:
            result['metadata'] = self.metadata.to_dict()
            
        # Add credentials reference (not the actual credentials)
        if self.credentials:
            result['has_credentials'] = True
            
        # Add coverage area if exists
        if self.coverage_area:
            result['coverage_area'] = self.coverage_area
            
        # Add timestamps if they exist
        if self.last_connected:
            result['last_connected'] = self.last_connected.isoformat()
            
        if self.last_status_change:
            result['last_status_change'] = self.last_status_change.isoformat()
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], credentials: Optional[CameraCredentials] = None) -> 'CameraSource':
        """Create camera source from dictionary."""
        # Process enums
        data['type'] = CameraType[data['type']]
        data['protocol'] = StreamProtocol[data['protocol']]
        data['status'] = CameraStatus[data['status']]
        
        # Process location
        if 'location' in data and data['location']:
            data['location'] = GeographicLocation(**data['location'])
            
        # Process metadata
        if 'metadata' in data and data['metadata']:
            data['metadata'] = CameraMetadata.from_dict(data['metadata'])
            
        # Process timestamps
        if 'last_connected' in data and data['last_connected']:
            data['last_connected'] = datetime.fromisoformat(data['last_connected'])
            
        if 'last_status_change' in data and data['last_status_change']:
            data['last_status_change'] = datetime.fromisoformat(data['last_status_change'])
            
        # Remove has_credentials flag
        if 'has_credentials' in data:
            del data['has_credentials']
            
        # Add provided credentials
        data['credentials'] = credentials
            
        return cls(**data)

@dataclass
class VideoFrame:
    """Video frame data with metadata."""
    frame: np.ndarray
    timestamp: datetime
    camera_id: str
    frame_number: int
    quality: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def copy(self) -> 'VideoFrame':
        """Create a copy of the video frame."""
        return VideoFrame(
            frame=self.frame.copy(),
            timestamp=self.timestamp,
            camera_id=self.camera_id,
            frame_number=self.frame_number,
            quality=self.quality,
            metadata=self.metadata.copy()
        )

@dataclass
class FrameBatch:
    """Batch of video frames from multiple cameras."""
    frames: List[VideoFrame]
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    creation_time: datetime = field(default_factory=datetime.now)
    
    def get_frames_by_camera(self, camera_id: str) -> List[VideoFrame]:
        """Get all frames from a specific camera."""
        return [frame for frame in self.frames if frame.camera_id == camera_id]
    
    def get_unique_camera_ids(self) -> Set[str]:
        """Get the set of unique camera IDs in this batch."""
        return {frame.camera_id for frame in self.frames}

class CameraConnection:
    """
    Manages the connection to a single camera.
    
    This class handles video capture, frame processing, and connection management
    for a single camera source.
    """
    
    def __init__(self, camera_source: CameraSource, frame_buffer_size: int = 30, 
                 processing_fps: Optional[int] = None):
        """
        Initialize camera connection.
        
        Args:
            camera_source: Camera source definition
            frame_buffer_size: Maximum number of frames to buffer
            processing_fps: Target FPS for processing (None = as fast as possible)
        """
        self.camera_source = camera_source
        self.frame_buffer_size = frame_buffer_size
        self.processing_fps = processing_fps
        
        # Frame handling
        self.frame_buffer = queue.Queue(maxsize=frame_buffer_size)
        self.current_frame_number = 0
        
        # Connection state
        self.capture = None
        self.running = False
        self.status = CameraStatus.INITIALIZING
        self.last_frame_time = None
        self.connection_thread = None
        self.frame_processing_thread = None
        self.last_error = None
        self.fps_stats = []  # Track actual FPS
        
        # Callbacks
        self.on_status_change = None
        self.on_new_frame = None
        self.on_error = None
        
        # Performance metrics
        self.metrics = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'connection_attempts': 0,
            'connection_failures': 0,
            'avg_fps': 0.0,
            'last_connect_time': None,
            'total_uptime': timedelta(0),
            'last_frame_processing_time': 0.0,
        }
        
    def set_status(self, new_status: CameraStatus):
        """
        Update connection status and trigger callback.
        
        Args:
            new_status: New connection status
        """
        if new_status != self.status:
            old_status = self.status
            self.status = new_status
            self.camera_source.status = new_status
            self.camera_source.last_status_change = datetime.now()
            
            logger.info(f"Camera {self.camera_source.id} status changed: {old_status.name} -> {new_status.name}")
            
            # Trigger callback if defined
            if self.on_status_change:
                try:
                    self.on_status_change(self.camera_source, old_status, new_status)
                except Exception as e:
                    logger.error(f"Error in status change callback: {str(e)}")
    
    def start(self):
        """Start camera connection and frame processing."""
        if self.running:
            logger.warning(f"Camera {self.camera_source.id} already running")
            return
            
        self.running = True
        self.connection_thread = threading.Thread(
            target=self._connection_thread,
            name=f"camera-connection-{self.camera_source.id}",
            daemon=True
        )
        self.connection_thread.start()
        
        self.frame_processing_thread = threading.Thread(
            target=self._frame_processing_thread,
            name=f"camera-processing-{self.camera_source.id}",
            daemon=True
        )
        self.frame_processing_thread.start()
        
        logger.info(f"Started camera connection for {self.camera_source.id}")
    
    def stop(self):
        """Stop camera connection and frame processing."""
        if not self.running:
            return
            
        logger.info(f"Stopping camera connection for {self.camera_source.id}")
        self.running = False
        
        # Wait for threads to terminate
        if self.connection_thread and self.connection_thread.is_alive():
            self.connection_thread.join(timeout=5.0)
            
        if self.frame_processing_thread and self.frame_processing_thread.is_alive():
            self.frame_processing_thread.join(timeout=5.0)
            
        # Release capture
        if self.capture:
            self.capture.release()
            self.capture = None
            
        # Clear buffer
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break
                
        self.set_status(CameraStatus.OFFLINE)
        logger.info(f"Camera connection stopped for {self.camera_source.id}")
    
    def _connection_thread(self):
        """Thread for maintaining camera connection and capturing frames."""
        while self.running:
            try:
                if self.capture is None or not self.capture.isOpened():
                    self._connect_to_camera()
                
                if not self.capture or not self.capture.isOpened():
                    # Connection failed, wait and retry
                    time.sleep(5)
                    continue
                    
                # Read frame
                ret, frame = self.capture.read()
                
                if not ret or frame is None:
                    self._handle_read_failure()
                    continue
                    
                # Process captured frame
                self._process_captured_frame(frame)
                
                # Adjust rate if needed
                if self.processing_fps:
                    time.sleep(max(0, 1.0/self.processing_fps - 
                                   (time.time() - (self.last_frame_time or 0))))
                    
            except Exception as e:
                self.last_error = str(e)
                logger.error(f"Error in camera connection thread for {self.camera_source.id}: {str(e)}")
                
                if self.on_error:
                    try:
                        self.on_error(self.camera_source, str(e))
                    except Exception as cb_error:
                        logger.error(f"Error in error callback: {str(cb_error)}")
                        
                self.set_status(CameraStatus.ERROR)
                
                # Close connection and wait before retry
                if self.capture:
                    self.capture.release()
                    self.capture = None
                    
                time.sleep(5)
                
        # Thread exit cleanup
        if self.capture:
            self.capture.release()
            self.capture = None
            
    def _connect_to_camera(self):
        """Establish connection to the camera."""
        self.metrics['connection_attempts'] += 1
        self.camera_source.connection_attempts += 1
        self.set_status(CameraStatus.INITIALIZING)
        
        logger.info(f"Connecting to camera {self.camera_source.id} at {self.camera_source.url}")
        
        try:
            # Handle different connection protocols
            if self.camera_source.protocol == StreamProtocol.RTSP:
                # RTSP options
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
                
                self.capture = cv2.VideoCapture(self.camera_source.url, cv2.CAP_FFMPEG)
                
                # Set additional RTSP options
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Small buffer to reduce latency
                
            elif self.camera_source.protocol == StreamProtocol.HTTP or self.camera_source.protocol == StreamProtocol.HTTPS:
                self.capture = cv2.VideoCapture(self.camera_source.url)
                
            elif self.camera_source.protocol == StreamProtocol.FILE:
                self.capture = cv2.VideoCapture(self.camera_source.url)
                
                # Loop video file if it's a file source
                self.capture.set(cv2.CAP_PROP_LOOP, 1)
                
            else:
                logger.error(f"Unsupported protocol {self.camera_source.protocol.name} for camera {self.camera_source.id}")
                self.set_status(CameraStatus.ERROR)
                self.metrics['connection_failures'] += 1
                return
                
            # Check if connection successful
            if not self.capture.isOpened():
                logger.error(f"Failed to open camera {self.camera_source.id}")
                self.set_status(CameraStatus.ERROR)
                self.metrics['connection_failures'] += 1
                return
                
            # Read first frame to confirm connection is working
            ret, frame = self.capture.read()
            if not ret or frame is None:
                logger.error(f"Failed to read first frame from camera {self.camera_source.id}")
                self.set_status(CameraStatus.ERROR)
                self.metrics['connection_failures'] += 1
                self.capture.release()
                self.capture = None
                return
                
            # Get camera properties and update metadata
            if self.camera_source.metadata is None:
                self.camera_source.metadata = CameraMetadata()
                
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            
            self.camera_source.metadata.resolution = (width, height)
            self.camera_source.metadata.fps = int(fps) if fps > 0 else None
            
            # Update connection status
            self.set_status(CameraStatus.ONLINE)
            self.camera_source.last_connected = datetime.now()
            self.metrics['last_connect_time'] = datetime.now()
            
            logger.info(f"Successfully connected to camera {self.camera_source.id} "
                       f"({width}x{height} @ {fps}fps)")
                       
        except Exception as e:
            logger.error(f"Error connecting to camera {self.camera_source.id}: {str(e)}")
            self.set_status(CameraStatus.ERROR)
            self.metrics['connection_failures'] += 1
            self.last_error = str(e)
            
            if self.capture:
                self.capture.release()
                self.capture = None
                
    def _handle_read_failure(self):
        """Handle frame read failure."""
        # Increment failures counter
        self.metrics['frames_dropped'] += 1
        
        if self.status == CameraStatus.ONLINE:
            # First failure, set to degraded
            self.set_status(CameraStatus.DEGRADED)
        else:
            # Subsequent failures
            read_failures = self.metrics['frames_dropped']
            
            if read_failures % 30 == 0:  # Log every 30 failures
                logger.warning(f"Continued frame read failures for camera {self.camera_source.id}: {read_failures} total")
                
            if read_failures > 100:
                # Too many failures, reset connection
                logger.error(f"Too many read failures for camera {self.camera_source.id}, resetting connection")
                self.set_status(CameraStatus.RECONNECTING)
                
                if self.capture:
                    self.capture.release()
                    self.capture = None
                    
    def _process_captured_frame(self, frame):
        """Process a successfully captured frame."""
        current_time = datetime.now()
        self.current_frame_number += 1
        
        # Update FPS tracking
        if self.last_frame_time is not None:
            elapsed = (current_time - self.last_frame_time).total_seconds()
            instantaneous_fps = 1.0 / elapsed if elapsed > 0 else 0
            
            # Keep moving average of last 10 frames for FPS statistics
            self.fps_stats.append(instantaneous_fps)
            if len(self.fps_stats) > 10:
                self.fps_stats.pop(0)
                
            self.metrics['avg_fps'] = sum(self.fps_stats) / len(self.fps_stats)
            
        self.last_frame_time = current_time
        
        # Update counters
        self.metrics['frames_captured'] += 1
        
        # If connected but status is degraded, reset to online
        if self.status == CameraStatus.DEGRADED:
            self.set_status(CameraStatus.ONLINE)
            
        # Calculate frame quality (placeholder for more sophisticated quality metrics)
        frame_quality = 1.0  # Default high quality
        
        # Create video frame object
        video_frame = VideoFrame(
            frame=frame,
            timestamp=current_time,
            camera_id=self.camera_source.id,
            frame_number=self.current_frame_number,
            quality=frame_quality,
            metadata={
                'resolution': (frame.shape[1], frame.shape[0]),
                'camera_name': self.camera_source.name,
                'camera_type': self.camera_source.type.name,
            }
        )
        
        # Add to frame buffer, dropping oldest frame if full
        try:
            if self.frame_buffer.full():
                # Buffer full, remove oldest frame
                try:
                    self.frame_buffer.get_nowait()
                    self.metrics['frames_dropped'] += 1
                except queue.Empty:
                    pass
                    
            self.frame_buffer.put_nowait(video_frame)
            
        except Exception as e:
            logger.error(f"Error adding frame to buffer for camera {self.camera_source.id}: {str(e)}")
            
    def _frame_processing_thread(self):
        """Thread for processing frames from the buffer."""
        while self.running:
            try:
                # Get next frame from buffer
                video_frame = self.frame_buffer.get(timeout=1.0)
                
                process_start = time.time()
                
                # Process the frame
                if self.on_new_frame:
                    try:
                        self.on_new_frame(video_frame)
                    except Exception as e:
                        logger.error(f"Error in frame processing callback: {str(e)}")
                        
                # Track processing time
                self.metrics['last_frame_processing_time'] = time.time() - process_start
                
                # Mark task as done
                self.frame_buffer.task_done()
                
            except queue.Empty:
                # No frames available, nothing to do
                pass
            except Exception as e:
                logger.error(f"Error in frame processing thread: {str(e)}")
                time.sleep(0.1)  # Prevent tight loop in case of persistent errors

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics for this camera connection."""
        current_time = datetime.now()
        
        stats = self.metrics.copy()
        
        # Calculate uptime if online
        if self.status in (CameraStatus.ONLINE, CameraStatus.DEGRADED) and self.metrics['last_connect_time']:
            current_session_uptime = current_time - self.metrics['last_connect_time']
            stats['total_uptime'] = self.metrics['total_uptime'] + current_session_uptime
        else:
            stats['total_uptime'] = self.metrics['total_uptime']
            
        # Add current status and frame buffer state
        stats['status'] = self.status.name
        stats['buffer_used'] = self.frame_buffer.qsize()
        stats['buffer_capacity'] = self.frame_buffer_size
        stats['last_error'] = self.last_error
        
        return stats
        
    def get_latest_frame(self) -> Optional[VideoFrame]:
        """Get the latest frame without removing it from the buffer."""
        if self.frame_buffer.empty():
            return None
            
        # We can't peek at the queue in Python, so this is a workaround
        all_frames = []
        latest_frame = None
        
        # Empty the queue and keep track of everything
        while not self.frame_buffer.empty():
            try:
                frame = self.frame_buffer.get_nowait()
                all_frames.append(frame)
                
                # Keep track of the latest frame by timestamp
                if latest_frame is None or frame.timestamp > latest_frame.timestamp:
                    latest_frame = frame
                    
                self.frame_buffer.task_done()
            except queue.Empty:
                break
                
        # Put everything back in the queue
        for frame in all_frames:
            try:
                self.frame_buffer.put_nowait(frame)
            except queue.Full:
                # If queue is full, frames will be dropped
                pass
                
        return latest_frame.copy() if latest_frame else None

    def send_ptz_command(self, command: str, params: Dict[str, Any] = None) -> bool:
        """
        Send PTZ command to camera.
        
        Args:
            command: Command type (e.g., 'pan', 'tilt', 'zoom', 'preset')
            params: Command parameters
            
        Returns:
            bool: True if command was sent successfully
        """
        if not params:
            params = {}
            
        # Check if camera has PTZ capability
        if (not self.camera_source.metadata or 
            self.camera_source.metadata.ptz_capability == PTZCapability.NONE):
            logger.warning(f"Camera {self.camera_source.id} does not have PTZ capability")
            return False
            
        # Check if camera is online
        if self.status not in (CameraStatus.ONLINE, CameraStatus.DEGRADED):
            logger.warning(f"Cannot send PTZ command to offline camera {self.camera_source.id}")
            return False
            
        # PTZ control depends on camera protocol and would require integration
        # with specific camera APIs. This is a placeholder for actual implementation.
        logger.info(f"PTZ command '{command}' with params {params} for camera {self.camera_source.id}")
        
        # In a real implementation, this would communicate with the camera's PTZ API
        return False  # Not implemented

class CameraRegistry:
    """
    Registry of all available camera sources.
    
    This class maintains a database of all camera sources and their metadata,
    supporting persistence, searching, and categorization.
    """
    
    def __init__(self, storage_path: str):
        """
        Initialize camera registry.
        
        Args:
            storage_path: Path to registry storage directory
        """
        self.storage_path = storage_path
        self.cameras: Dict[str, CameraSource] = {}
        self.credentials_path = os.path.join(storage_path, "credentials")
        self.registry_path = os.path.join(storage_path, "registry.json")
        
        # Ensure directories exist
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(self.credentials_path, exist_ok=True)
        
        # Load registry
        self._load_registry()
        
        logger.info(f"Camera registry initialized with {len(self.cameras)} cameras")
        
    def _load_registry(self):
        """Load camera registry from storage."""
        try:
            if not os.path.exists(self.registry_path):
                logger.info(f"Registry file not found at {self.registry_path}, creating new registry")
                self.cameras = {}
                return
                
            with open(self.registry_path, 'r') as f:
                registry_data = json.load(f)
                
            for camera_id, camera_data in registry_data.items():
                # Load credentials if available
                credentials = None
                if camera_data.get('has_credentials', False):
                    credentials_path = os.path.join(self.credentials_path, f"{camera_id}.json")
                    if os.path.exists(credentials_path):
                        try:
                            with open(credentials_path, 'r') as f:
                                credentials_data = json.load(f)
                                credentials = CameraCredentials(**credentials_data)
                        except Exception as e:
                            logger.error(f"Error loading credentials for camera {camera_id}: {str(e)}")
                
                # Create camera source
                camera = CameraSource.from_dict(camera_data, credentials)
                self.cameras[camera_id] = camera
                
            logger.info(f"Loaded {len(self.cameras)} cameras from registry")
            
        except Exception as e:
            logger.error(f"Error loading camera registry: {str(e)}")
            self.cameras = {}
            
    def save_registry(self):
        """Save camera registry to storage."""
        try:
            # Convert cameras to serializable dict
            registry_data = {}
            for camera_id, camera in self.cameras.items():
                registry_data[camera_id] = camera.to_dict()
                
            # Save registry file
            with open(self.registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
            logger.info(f"Saved {len(self.cameras)} cameras to registry")
            
        except Exception as e:
            logger.error(f"Error saving camera registry: {str(e)}")
            
    def save_credentials(self, camera_id: str, credentials: CameraCredentials):
        """
        Save camera credentials securely.
        
        Args:
            camera_id: Camera ID
            credentials: Camera credentials
        """
        try:
            credentials_path = os.path.join(self.credentials_path, f"{camera_id}.json")
            
            # Convert credentials to serializable dict
            credentials_data = {
                'username': credentials.username,
                'password': credentials.password,
                'token': credentials.token,
                'api_key': credentials.api_key,
                'certificate_path': credentials.certificate_path,
                'private_key_path': credentials.private_key_path,
                'auth_method': credentials.auth_method
            }
            
            # Save credentials file
            with open(credentials_path, 'w') as f:
                json.dump(credentials_data, f, indent=2)
                
            # Secure the file (works on Unix-like systems)
            try:
                os.chmod(credentials_path, 0o600)  # Read/write for owner only
            except Exception as e:
                logger.warning(f"Could not set secure permissions on credentials file: {str(e)}")
                
            logger.info(f"Saved credentials for camera {camera_id}")
            
        except Exception as e:
            logger.error(f"Error saving credentials for camera {camera_id}: {str(e)}")
            
    def add_camera(self, camera: CameraSource) -> bool:
        """
        Add camera to registry.
        
        Args:
            camera: Camera source to add
            
        Returns:
            bool: True if added successfully
        """
        if camera.id in self.cameras:
            logger.warning(f"Camera with ID {camera.id} already exists in registry")
            return False
            
        # Add to registry
        self.cameras[camera.id] = camera
        
        # Save credentials if available
        if camera.credentials and camera.credentials.validate():
            self.save_credentials(camera.id, camera.credentials)
            
        # Save registry
        self.save_registry()
        
        logger.info(f"Added camera {camera.id} to registry")
        return True
        
    def update_camera(self, camera: CameraSource) -> bool:
        """
        Update camera in registry.
        
        Args:
            camera: Updated camera source
            
        Returns:
            bool: True if updated successfully
        """
        if camera.id not in self.cameras:
            logger.warning(f"Camera with ID {camera.id} not found in registry")
            return False
            
        # Update registry
        self.cameras[camera.id] = camera
        
        # Save credentials if available
        if camera.credentials and camera.credentials.validate():
            self.save_credentials(camera.id, camera.credentials)
            
        # Save registry
        self.save_registry()
        
        logger.info(f"Updated camera {camera.id} in registry")
        return True
        
    def remove_camera(self, camera_id: str) -> bool:
        """
        Remove camera from registry.
        
        Args:
            camera_id: ID of camera to remove
            
        Returns:
            bool: True if removed successfully
        """
        if camera_id not in self.cameras:
            logger.warning(f"Camera with ID {camera_id} not found in registry")
            return False
            
        # Remove from registry
        del self.cameras[camera_id]
        
        # Remove credentials if available
        credentials_path = os.path.join(self.credentials_path, f"{camera_id}.json")
        if os.path.exists(credentials_path):
            try:
                os.remove(credentials_path)
            except Exception as e:
                logger.error(f"Error removing credentials for camera {camera_id}: {str(e)}")
                
        # Save registry
        self.save_registry()
        
        logger.info(f"Removed camera {camera_id} from registry")
        return True
        
    def get_camera(self, camera_id: str) -> Optional[CameraSource]:
        """
        Get camera by ID.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            Optional[CameraSource]: Camera source if found, None otherwise
        """
        return self.cameras.get(camera_id)
        
    def get_all_cameras(self) -> List[CameraSource]:
        """
        Get all cameras in registry.
        
        Returns:
            List[CameraSource]: List of all camera sources
        """
        return list(self.cameras.values())
        
    def search_cameras(self, 
                     name: str = None, 
                     camera_type: CameraType = None,
                     status: CameraStatus = None,
                     tags: List[str] = None,
                     location_radius: Tuple[float, float, float] = None) -> List[CameraSource]:
        """
        Search for cameras matching criteria.
        
        Args:
            name: Name to search for (case-insensitive substring match)
            camera_type: Camera type to filter by
            status: Camera status to filter by
            tags: Tags to filter by (all tags must match)
            location_radius: (latitude, longitude, radius_km) to search within
            
        Returns:
            List[CameraSource]: List of matching camera sources
        """
        results = []
        
        for camera in self.cameras.values():
            # Check name
            if name and name.lower() not in camera.name.lower():
                continue
                
            # Check type
            if camera_type and camera.type != camera_type:
                continue
                
            # Check status
            if status and camera.status != status:
                continue
                
            # Check tags (all must match)
            if tags and not all(tag in camera.tags for tag in tags):
                continue
                
            # Check location radius
            if location_radius and camera.location:
                lat, lon, radius_km = location_radius
                
                # Calculate distance using Haversine formula
                if not self._is_within_radius(
                    camera.location.latitude, camera.location.longitude,
                    lat, lon, radius_km):
                    continue
                    
            # All criteria matched
            results.append(camera)
            
        return results
        
    def _is_within_radius(self, lat1, lon1, lat2, lon2, radius_km):
        """
        Check if a point is within a radius of another point using Haversine formula.
        
        Args:
            lat1, lon1: Coordinates of first point
            lat2, lon2: Coordinates of second point
            radius_km: Radius in kilometers
            
        Returns:
            bool: True if within radius
        """
        # Convert coordinates to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Earth radius in kilometers
        earth_radius = 6371.0
        
        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        distance = earth_radius * c
        
        return distance <= radius_km

class NetworkDiscoveryService:
    """
    Service for discovering cameras on the network.
    
    This class implements methods for discovering cameras using various
    protocols, including ONVIF, UPnP, and Bonjour/Zeroconf.
    """
    
    def __init__(self, registry: CameraRegistry):
        """
        Initialize network discovery service.
        
        Args:
            registry: Camera registry to update with discovered cameras
        """
        self.registry = registry
        self.discovery_threads = {}
        self.running = False
        
        # Discovery configurations
        self.discovery_methods = {
            'onvif': True,
            'upnp': True,
            'bonjour': True,
            'ip_scan': False,  # Disabled by default as it can be intrusive
        }
        
        # Discovery statistics
        self.discovery_stats = {
            'last_discovery_time': None,
            'cameras_discovered': 0,
            'cameras_added': 0,
            'discovery_duration': 0,
        }
        
        logger.info("Network discovery service initialized")
        
    def start_discovery(self, 
                      methods: List[str] = None, 
                      ip_ranges: List[str] = None,
                      callback: Callable[[List[CameraSource]], None] = None) -> str:
        """
        Start camera discovery process.
        
        Args:
            methods: Discovery methods to use (default: all enabled methods)
            ip_ranges: IP ranges to scan (for IP scan method)
            callback: Function to call when discovery completes
            
        Returns:
            str: Discovery ID
        """
        discovery_id = str(uuid.uuid4())
        
        # Determine which methods to use
        if methods:
            active_methods = {method: True for method in methods if method in self.discovery_methods}
        else:
            active_methods = {method: enabled for method, enabled in self.discovery_methods.items() if enabled}
            
        # Start discovery thread
        thread = threading.Thread(
            target=self._discovery_thread,
            args=(discovery_id, active_methods, ip_ranges, callback),
            name=f"camera-discovery-{discovery_id}",
            daemon=True
        )
        
        self.discovery_threads[discovery_id] = {
            'thread': thread,
            'status': 'starting',
            'methods': active_methods,
            'ip_ranges': ip_ranges,
            'callback': callback,
            'start_time': datetime.now(),
            'discovered_cameras': [],
            'added_cameras': 0,
        }
        
        thread.start()
        logger.info(f"Started camera discovery {discovery_id} with methods: {', '.join(active_methods.keys())}")
        
        return discovery_id
        
    def get_discovery_status(self, discovery_id: str) -> Dict[str, Any]:
        """
        Get status of a discovery process.
        
        Args:
            discovery_id: Discovery ID
            
        Returns:
            Dict: Discovery status
        """
        if discovery_id not in self.discovery_threads:
            return {'status': 'not_found'}
            
        discovery = self.discovery_threads[discovery_id]
        
        # Calculate progress based on active methods
        methods_count = len(discovery['methods'])
        completed_methods = sum(1 for method, status in discovery.get('method_status', {}).items() 
                               if status == 'completed')
        progress = (completed_methods / methods_count) if methods_count > 0 else 0
        
        return {
            'status': discovery['status'],
            'methods': list(discovery['methods'].keys()),
            'progress': progress,
            'start_time': discovery['start_time'].isoformat(),
            'elapsed_time': (datetime.now() - discovery['start_time']).total_seconds(),
            'discovered_cameras': len(discovery['discovered_cameras']),
            'added_cameras': discovery['added_cameras'],
        }
        
    def _discovery_thread(self, 
                         discovery_id: str, 
                         methods: Dict[str, bool], 
                         ip_ranges: List[str],
                         callback: Callable[[List[CameraSource]], None]):
        """
        Thread for running camera discovery.
        
        Args:
            discovery_id: Discovery ID
            methods: Discovery methods to use
            ip_ranges: IP ranges to scan
            callback: Function to call when discovery completes
        """
        logger.info(f"Discovery thread {discovery_id} started")
        
        try:
            discovery = self.discovery_threads[discovery_id]
            discovery['status'] = 'running'
            discovery['method_status'] = {}
            
            discovered_cameras = []
            
            # Run each enabled discovery method
            for method, enabled in methods.items():
                if not enabled:
                    continue
                    
                discovery['method_status'][method] = 'running'
                
                try:
                    if method == 'onvif':
                        cameras = self._discover_onvif_cameras()
                    elif method == 'upnp':
                        cameras = self._discover_upnp_cameras()
                    elif method == 'bonjour':
                        cameras = self._discover_bonjour_cameras()
                    elif method == 'ip_scan':
                        cameras = self._discover_ip_scan_cameras(ip_ranges)
                    else:
                        logger.warning(f"Unknown discovery method: {method}")
                        cameras = []
                        
                    # Update discovered cameras
                    discovered_cameras.extend(cameras)
                    discovery['discovered_cameras'].extend(cameras)
                    
                    discovery['method_status'][method] = 'completed'
                    logger.info(f"Discovery method {method} completed, found {len(cameras)} cameras")
                    
                except Exception as e:
                    discovery['method_status'][method] = 'error'
                    logger.error(f"Error in discovery method {method}: {str(e)}")
                    
            # De-duplicate cameras by URL
            unique_cameras = {}
            for camera in discovered_cameras:
                if camera.url not in unique_cameras:
                    unique_cameras[camera.url] = camera
                    
            discovered_cameras = list(unique_cameras.values())
            
            # Add discovered cameras to registry
            added_count = 0
            for camera in discovered_cameras:
                # Generate ID if not present
                if not camera.id:
                    camera.id = str(uuid.uuid4())
                    
                # Check if camera already exists by URL
                existing_cameras = [c for c in self.registry.get_all_cameras() if c.url == camera.url]
                
                if existing_cameras:
                    # Camera already exists, update if needed
                    existing = existing_cameras[0]
                    
                    # Update metadata if discovered version has more info
                    if camera.metadata and (not existing.metadata or 
                                           len(camera.metadata.to_dict()) > len(existing.metadata.to_dict())):
                        existing.metadata = camera.metadata
                        self.registry.update_camera(existing)
                        
                else:
                    # New camera, add to registry
                    self.registry.add_camera(camera)
                    added_count += 1
                    
            # Update discovery statistics
            discovery['status'] = 'completed'
            discovery['added_cameras'] = added_count
            discovery['end_time'] = datetime.now()
            
            # Update global stats
            self.discovery_stats['last_discovery_time'] = datetime.now()
            self.discovery_stats['cameras_discovered'] = len(discovered_cameras)
            self.discovery_stats['cameras_added'] = added_count
            self.discovery_stats['discovery_duration'] = (
                discovery['end_time'] - discovery['start_time']
            ).total_seconds()
            
            logger.info(f"Discovery {discovery_id} completed: found {len(discovered_cameras)} cameras, "
                       f"added {added_count} new cameras")
                       
            # Call callback if provided
            if callback:
                try:
                    callback(discovered_cameras)
                except Exception as e:
                    logger.error(f"Error in discovery callback: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error in discovery thread {discovery_id}: {str(e)}")
            
            if discovery_id in self.discovery_threads:
                self.discovery_threads[discovery_id]['status'] = 'error'
                
        finally:
            # Clean up old discovery records after 1 hour
            current_time = datetime.now()
            expired_discoveries = [
                disc_id for disc_id, disc in self.discovery_threads.items()
                if (disc['status'] in ('completed', 'error') and 
                    'end_time' in disc and 
                    (current_time - disc['end_time']).total_seconds() > 3600)
            ]
            
            for disc_id in expired_discoveries:
                if disc_id in self.discovery_threads:
                    del self.discovery_threads[disc_id]
                    
    def _discover_onvif_cameras(self) -> List[CameraSource]:
        """
        Discover cameras using ONVIF protocol.
        
        Returns:
            List[CameraSource]: Discovered cameras
        """
        # This is a placeholder. In a real implementation, this would use the ONVIF
        # protocol to discover cameras on the network.
        logger.info("ONVIF camera discovery started")
        
        # Simulate discovery
        time.sleep(2)
        
        # In a real implementation, this would return actual discovered cameras
        return []
        
    def _discover_upnp_cameras(self) -> List[CameraSource]:
        """
        Discover cameras using UPnP protocol.
        
        Returns:
            List[CameraSource]: Discovered cameras
        """
        # This is a placeholder. In a real implementation, this would use the UPnP
        # protocol to discover cameras on the network.
        logger.info("UPnP camera discovery started")
        
        # Simulate discovery
        time.sleep(2)
        
        # In a real implementation, this would return actual discovered cameras
        return []
        
    def _discover_bonjour_cameras(self) -> List[CameraSource]:
        """
        Discover cameras using Bonjour/Zeroconf protocol.
        
        Returns:
            List[CameraSource]: Discovered cameras
        """
        # This is a placeholder. In a real implementation, this would use the
        # Bonjour/Zeroconf protocol to discover cameras on the network.
        logger.info("Bonjour camera discovery started")
        
        # Simulate discovery
        time.sleep(2)
        
        # In a real implementation, this would return actual discovered cameras
        return []
        
    def _discover_ip_scan_cameras(self, ip_ranges: List[str] = None) -> List[CameraSource]:
        """
        Discover cameras by scanning IP ranges.
        
        Args:
            ip_ranges: IP ranges to scan (e.g., ['192.168.1.0/24'])
            
        Returns:
            List[CameraSource]: Discovered cameras
        """
        if not ip_ranges:
            logger.warning("No IP ranges specified for IP scan discovery")
            return []
            
        # This is a placeholder. In a real implementation, this would scan the
        # specified IP ranges for cameras on common ports.
        logger.info(f"IP scan camera discovery started for ranges: {', '.join(ip_ranges)}")
        
        # Simulate discovery
        time.sleep(3)
        
        # In a real implementation, this would return actual discovered cameras
        return []

class CameraManager:
    """
    Manages the collection of camera feeds and their processing.
    
    This class is the main entry point for the camera subsystem, managing
    camera connections, frame processing, and integration with the rest
    of the system.
    """
    
    def __init__(self, config=None):
        """
        Initialize camera manager.
        
        Args:
            config: Configuration for the camera manager
        """
        self.config = config or {}
        
        # Extract configuration parameters with defaults
        self.storage_path = self.config.get('local_storage_path', '/var/lib/ultratrack/camera_registry')
        self.max_connections = self.config.get('max_connections', 1000)
        self.connection_timeout = self.config.get('connection_timeout', 10)
        self.reconnect_interval = self.config.get('reconnect_interval', 30)
        self.frame_buffer_size = self.config.get('frame_buffer_size', 30)
        self.discovery_enabled = self.config.get('discovery_enabled', True)
        
        # Create registry and discovery service
        self.registry = CameraRegistry(self.storage_path)
        self.discovery_service = NetworkDiscoveryService(self.registry)
        
        # Active camera connections
        self.connections: Dict[str, CameraConnection] = {}
        
        # Frame processing
        self.frame_handlers: List[Callable[[VideoFrame], None]] = []
        self.batch_handlers: List[Callable[[FrameBatch], None]] = []
        self.batch_interval = 0.1  # seconds
        self.batch_size = 10  # max frames per batch
        self.batch_buffer = []
        self.batch_lock = threading.Lock()
        self.batch_thread = None
        self.running = False
        
        # Statistics
        self.stats = {
            'total_frames_processed': 0,
            'total_batches_processed': 0,
            'active_connections': 0,
            'last_stats_time': datetime.now(),
            'frames_per_second': 0,
            'batches_per_second': 0,
        }
        
        logger.info("Camera manager initialized")
        
    def start(self):
        """Start camera manager and connect to cameras."""
        if self.running:
            logger.warning("Camera manager already running")
            return
            
        self.running = True
        
        # Start batch processing thread
        self.batch_thread = threading.Thread(
            target=self._batch_processing_thread,
            name="camera-batch-processor",
            daemon=True
        )
        self.batch_thread.start()
        
        # Connect to cameras
        self.connect_all_cameras()
        
        logger.info("Camera manager started")
        
    def stop(self):
        """Stop camera manager and disconnect from cameras."""
        if not self.running:
            return
            
        logger.info("Stopping camera manager")
        self.running = False
        
        # Disconnect from all cameras
        for camera_id in list(self.connections.keys()):
            self.disconnect_camera(camera_id)
            
        # Wait for batch thread to terminate
        if self.batch_thread and self.batch_thread.is_alive():
            self.batch_thread.join(timeout=5.0)
            
        logger.info("Camera manager stopped")
        
    def connect_all_cameras(self):
        """Connect to all enabled cameras in the registry."""
        cameras = self.registry.get_all_cameras()
        
        for camera in cameras:
            if camera.enabled:
                self.connect_camera(camera.id)
                
    def connect_camera(self, camera_id: str) -> bool:
        """
        Connect to a specific camera.
        
        Args:
            camera_id: Camera ID to connect to
            
        Returns:
            bool: True if connection was initiated successfully
        """
        # Check if already connected
        if camera_id in self.connections:
            logger.warning(f"Camera {camera_id} already connected")
            return True
            
        # Check connection limit
        if len(self.connections) >= self.max_connections:
            logger.error(f"Cannot connect to camera {camera_id}: maximum connections reached")
            return False
            
        # Get camera from registry
        camera = self.registry.get_camera(camera_id)
        if not camera:
            logger.error(f"Camera {camera_id} not found in registry")
            return False
            
        # Create connection
        connection = CameraConnection(
            camera_source=camera,
            frame_buffer_size=self.frame_buffer_size,
            processing_fps=camera.metadata.fps if camera.metadata and camera.metadata.fps else None
        )
        
        # Set callbacks
        connection.on_new_frame = self._handle_new_frame
        connection.on_status_change = self._handle_status_change
        connection.on_error = self._handle_connection_error
        
        # Start connection
        connection.start()
        
        # Add to active connections
        self.connections[camera_id] = connection
        self.stats['active_connections'] = len(self.connections)
        
        logger.info(f"Camera {camera_id} connection initiated")
        return True
        
    def disconnect_camera(self, camera_id: str) -> bool:
        """
        Disconnect from a specific camera.
        
        Args:
            camera_id: Camera ID to disconnect from
            
        Returns:
            bool: True if disconnection was successful
        """
        if camera_id not in self.connections:
            logger.warning(f"Camera {camera_id} not connected")
            return False
            
        # Stop connection
        connection = self.connections[camera_id]
        connection.stop()
        
        # Remove from active connections
        del self.connections[camera_id]
        self.stats['active_connections'] = len(self.connections)
        
        logger.info(f"Camera {camera_id} disconnected")
        return True
        
    def get_camera_status(self, camera_id: str) -> Dict[str, Any]:
        """
        Get status of a specific camera.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            Dict: Camera status information
        """
        # Check if connected
        if camera_id in self.connections:
            connection = self.connections[camera_id]
            
            # Get camera source info
            camera = connection.camera_source
            
            # Get connection stats
            connection_stats = connection.get_stats()
            
            return {
                'id': camera.id,
                'name': camera.name,
                'status': connection_stats['status'],
                'connected': connection.status in (CameraStatus.ONLINE, CameraStatus.DEGRADED),
                'frames_captured': connection_stats['frames_captured'],
                'frames_dropped': connection_stats['frames_dropped'],
                'avg_fps': connection_stats['avg_fps'],
                'buffer_used': connection_stats['buffer_used'],
                'buffer_capacity': connection_stats['buffer_capacity'],
                'last_error': connection_stats['last_error'],
                'uptime': str(connection_stats['total_uptime']),
            }
        else:
            # Get camera from registry
            camera = self.registry.get_camera(camera_id)
            if not camera:
                return {'error': 'Camera not found'}
                
            return {
                'id': camera.id,
                'name': camera.name,
                'status': camera.status.name,
                'connected': False,
                'enabled': camera.enabled,
            }
            
    def get_all_camera_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all cameras.
        
        Returns:
            Dict: Camera statuses keyed by camera ID
        """
        statuses = {}
        
        # Get all cameras from registry
        all_cameras = self.registry.get_all_cameras()
        
        for camera in all_cameras:
            statuses[camera.id] = self.get_camera_status(camera.id)
            
        return statuses
        
    def register_frame_handler(self, handler: Callable[[VideoFrame], None]):
        """
        Register a handler for individual frames.
        
        Args:
            handler: Function to call for each frame
        """
        if handler not in self.frame_handlers:
            self.frame_handlers.append(handler)
            logger.info(f"Registered frame handler {handler.__name__}")
            
    def unregister_frame_handler(self, handler: Callable[[VideoFrame], None]):
        """
        Unregister a frame handler.
        
        Args:
            handler: Handler to unregister
        """
        if handler in self.frame_handlers:
            self.frame_handlers.remove(handler)
            logger.info(f"Unregistered frame handler {handler.__name__}")
            
    def register_batch_handler(self, handler: Callable[[FrameBatch], None]):
        """
        Register a handler for frame batches.
        
        Args:
            handler: Function to call for each batch
        """
        if handler not in self.batch_handlers:
            self.batch_handlers.append(handler)
            logger.info(f"Registered batch handler {handler.__name__}")
            
    def unregister_batch_handler(self, handler: Callable[[FrameBatch], None]):
        """
        Unregister a batch handler.
        
        Args:
            handler: Handler to unregister
        """
        if handler in self.batch_handlers:
            self.batch_handlers.remove(handler)
            logger.info(f"Unregistered batch handler {handler.__name__}")
            
    def get_latest_frame(self, camera_id: str) -> Optional[VideoFrame]:
        """
        Get the latest frame from a specific camera.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            Optional[VideoFrame]: Latest frame if available
        """
        if camera_id not in self.connections:
            logger.warning(f"Camera {camera_id} not connected")
            return None
            
        return self.connections[camera_id].get_latest_frame()
        
    def get_camera_frame(self, camera_id: str) -> Optional[VideoFrame]:
        """
        Get a frame from a specific camera, connecting if necessary.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            Optional[VideoFrame]: Camera frame if available
        """
        # Check if connected
        if camera_id not in self.connections:
            # Try to connect
            if not self.connect_camera(camera_id):
                return None
                
            # Wait for connection to establish
            time.sleep(1.0)
            
        # Try to get frame
        return self.get_latest_frame(camera_id)
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get camera manager statistics.
        
        Returns:
            Dict: Statistics
        """
        current_time = datetime.now()
        elapsed = (current_time - self.stats['last_stats_time']).total_seconds()
        
        # Calculate FPS and BPS if sufficient time has passed
        if elapsed >= 1.0:
            frames_since_last = self.stats['total_frames_processed'] - self.stats.get('last_frames_processed', 0)
            batches_since_last = self.stats['total_batches_processed'] - self.stats.get('last_batches_processed', 0)
            
            self.stats['frames_per_second'] = frames_since_last / elapsed
            self.stats['batches_per_second'] = batches_since_last / elapsed
            
            self.stats['last_frames_processed'] = self.stats['total_frames_processed']
            self.stats['last_batches_processed'] = self.stats['total_batches_processed']
            self.stats['last_stats_time'] = current_time
            
        # Update active connections count
        self.stats['active_connections'] = len(self.connections)
        
        return self.stats
        
    def start_discovery(self, 
                      methods: List[str] = None, 
                      ip_ranges: List[str] = None,
                      callback: Callable[[List[CameraSource]], None] = None) -> str:
        """
        Start camera discovery process.
        
        Args:
            methods: Discovery methods to use
            ip_ranges: IP ranges to scan
            callback: Function to call when discovery completes
            
        Returns:
            str: Discovery ID
        """
        if not self.discovery_enabled:
            logger.warning("Camera discovery is disabled")
            return None
            
        return self.discovery_service.start_discovery(methods, ip_ranges, callback)
        
    def get_discovery_status(self, discovery_id: str) -> Dict[str, Any]:
        """
        Get status of a discovery process.
        
        Args:
            discovery_id: Discovery ID
            
        Returns:
            Dict: Discovery status
        """
        return self.discovery_service.get_discovery_status(discovery_id)
        
    def _handle_new_frame(self, frame: VideoFrame):
        """
        Handle a new frame from a camera.
        
        Args:
            frame: Video frame
        """
        # Call individual frame handlers
        for handler in self.frame_handlers:
            try:
                handler(frame)
            except Exception as e:
                logger.error(f"Error in frame handler {handler.__name__}: {str(e)}")
                
        # Add to batch buffer
        with self.batch_lock:
            self.batch_buffer.append(frame)
            
        # Update statistics
        self.stats['total_frames_processed'] += 1
        
    def _handle_status_change(self, camera: CameraSource, old_status: CameraStatus, new_status: CameraStatus):
        """
        Handle camera status change.
        
        Args:
            camera: Camera source
            old_status: Old status
            new_status: New status
        """
        # Update registry if status changed
        if old_status != new_status:
            self.registry.update_camera(camera)
            
    def _handle_connection_error(self, camera: CameraSource, error: str):
        """
        Handle camera connection error.
        
        Args:
            camera: Camera source
            error: Error message
        """
        logger.error(f"Camera {camera.id} connection error: {error}")
        
        # Update registry with error status
        camera.status = CameraStatus.ERROR
        self.registry.update_camera(camera)
        
    def _batch_processing_thread(self):
        """Thread for processing frame batches."""
        last_batch_time = time.time()
        
        while self.running:
            current_time = time.time()
            elapsed = current_time - last_batch_time
            
            # Check if it's time to process a batch
            if elapsed >= self.batch_interval or len(self.batch_buffer) >= self.batch_size:
                # Extract frames from buffer
                with self.batch_lock:
                    if not self.batch_buffer:
                        time.sleep(0.01)  # No frames, wait a bit
                        continue
                        
                    frames = self.batch_buffer
                    self.batch_buffer = []
                    
                # Create batch
                batch = FrameBatch(frames=frames)
                
                # Call batch handlers
                for handler in self.batch_handlers:
                    try:
                        handler(batch)
                    except Exception as e:
                        logger.error(f"Error in batch handler {handler.__name__}: {str(e)}")
                        
                # Update statistics
                self.stats['total_batches_processed'] += 1
                last_batch_time = current_time
                
            else:
                # Wait a bit to avoid tight loop
                time.sleep(0.01)
                
    def shutdown(self):
        """Shutdown the camera manager."""
        self.stop()

# Allow module to be run directly for testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create camera manager
    manager = CameraManager()
    
    # Start camera manager
    manager.start()
    
    try:
        # Run for a while
        time.sleep(60)
    finally:
        # Stop camera manager
        manager.stop()
