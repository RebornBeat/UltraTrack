"""
Camera capture module for the Traffic Monitoring System.
Provides interfaces for camera hardware interaction and video stream management.
"""

from .camera_interface import CameraInterface, CameraType, CameraConfig
from .video_stream import VideoStream, FrameBuffer, StreamStatus

__all__ = [
    'CameraInterface', 
    'CameraType', 
    'CameraConfig',
    'VideoStream', 
    'FrameBuffer', 
    'StreamStatus'
]
