"""
Processing module for the Traffic Monitoring System.
Provides functionality for video processing, object detection, and feature extraction.
"""

from .background_subtraction import (
    BackgroundSubtractor, SubtractionMethod, MotionDetector
)
from .vehicle_detection import (
    VehicleDetector, DetectionModel, VehicleType, Detection
)
from .license_plate_recognition import (
    LicensePlateDetector, LicensePlateRecognizer, PlateDetection
)

__all__ = [
    'BackgroundSubtractor',
    'SubtractionMethod',
    'MotionDetector',
    'VehicleDetector',
    'DetectionModel',
    'VehicleType',
    'Detection',
    'LicensePlateDetector',
    'LicensePlateRecognizer',
    'PlateDetection'
]
