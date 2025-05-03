"""
License plate recognition module for detecting and reading license plates in vehicles.
Implements plate detection, preprocessing, and OCR for various plate formats.
"""

import os
import re
import time
import logging
import threading
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

# Configure logger for this module
logger = logging.getLogger(__name__)

# Configure Tesseract path if not in PATH
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


@dataclass
class PlateDetection:
    """Class for storing license plate detection results."""
    box: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    plate_text: str
    plate_image: np.ndarray
    vehicle_id: Optional[int] = None
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'box': self.box,
            'confidence': self.confidence,
            'plate_text': self.plate_text,
            'vehicle_id': self.vehicle_id,
            'timestamp': self.timestamp
        }


class LicensePlateDetector:
    """
    License plate detector for locating plates in images.
    """
    
    def __init__(
        self,
        cascade_path: Optional[str] = None,
        use_haar: bool = True,
        confidence_threshold: float = 0.5,
        min_plate_width: int = 60,
        min_plate_height: int = 20,
        max_plate_width: int = 300,
        max_plate_height: int = 100,
        plate_aspect_ratio_min: float = 1.0,  # width/height
        plate_aspect_ratio_max: float = 6.0
    ):
        """
        Initialize license plate detector.
        
        Args:
            cascade_path: Path to Haar cascade XML file or None for default
            use_haar: Whether to use Haar cascade (if False, use edge-based detection)
            confidence_threshold: Minimum confidence for YOLO detections
            min_plate_width: Minimum plate width in pixels
            min_plate_height: Minimum plate height in pixels
            max_plate_width: Maximum plate width in pixels
            max_plate_height: Maximum plate height in pixels
            plate_aspect_ratio_min: Minimum plate aspect ratio (width/height)
            plate_aspect_ratio_max: Maximum plate aspect ratio (width/height)
        """
        self.use_haar = use_haar
        self.confidence_threshold = confidence_threshold
        self.min_plate_width = min_plate_width
        self.min_plate_height = min_plate_height
        self.max_plate_width = max_plate_width
        self.max_plate_height = max_plate_height
        self.plate_aspect_ratio_min = plate_aspect_ratio_min
        self.plate_aspect_ratio_max = plate_aspect_ratio_max
        
        # Load Haar cascade classifier
        if use_haar:
            if cascade_path is None:
                # Use default cascade path
                current_dir = os.path.dirname(os.path.abspath(__file__))
                cascade_path = os.path.join(current_dir, "../models/haarcascades/haarcascade_russian_plate_number.xml")
                
                # If file doesn't exist, try to use built-in OpenCV cascades
                if not os.path.exists(cascade_path):
                    cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            
            # Load cascade classifier
            self.cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.cascade.empty():
                logger.error(f"Failed to load cascade classifier from {cascade_path}")
                raise RuntimeError(f"Failed to load cascade classifier from {cascade_path}")
            
            logger.info(f"Loaded license plate cascade classifier from {cascade_path}")
        else:
            self.cascade = None
            logger.info("Using edge-based license plate detection")
        
        # Initialize YOLO model for license plate detection if available
        self.yolo_model = None
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            yolo_dir = os.path.join(current_dir, "../models/yolo_plate")
            
            cfg_path = os.path.join(yolo_dir, "yolov4-plate.cfg")
            weights_path = os.path.join(yolo_dir, "yolov4-plate.weights")
            
            if os.path.exists(cfg_path) and os.path.exists(weights_path):
                self.yolo_model = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
                logger.info("Loaded YOLO model for license plate detection")
                
                # Get output layer names
                self.yolo_output_layers = self.yolo_model.getUnconnectedOutLayersNames()
            
        except Exception as e:
            logger.warning(f"Could not load YOLO model for license plate detection: {str(e)}")
            self.yolo_model = None
        
        # Performance metrics
        self.frame_count = 0
        self.plate_count = 0
        self.process_times = []
        self.avg_process_time = 0
    
    def detect(self, frame: np.ndarray, vehicle_boxes: Optional[List[Tuple[int, int, int, int]]] = None) -> List[Tuple[int, int, int, int]]:
        """
        Detect license plates in a frame.
        
        Args:
            frame: Input video frame
            vehicle_boxes: Optional list of vehicle bounding boxes to search within
        
        Returns:
            List of license plate bounding boxes (x, y, w, h)
        """
        start_time = time.time()
        self.frame_count += 1
        
        plate_boxes = []
        
        # If vehicle boxes are provided, search within each vehicle
        if vehicle_boxes and len(vehicle_boxes) > 0:
            for vbox in vehicle_boxes:
                vx, vy, vw, vh = vbox
                
                # Extract vehicle ROI with margin
                margin = 10
                roi_x = max(0, vx - margin)
                roi_y = max(0, vy - margin)
                roi_w = min(frame.shape[1] - roi_x, vw
