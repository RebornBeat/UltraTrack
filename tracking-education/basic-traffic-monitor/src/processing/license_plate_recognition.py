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
                roi_w = min(frame.shape[1] - roi_x, vw + 2 * margin)
                roi_h = min(frame.shape[0] - roi_y, vh + 2 * margin)
                
                # Extract vehicle ROI
                vehicle_roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                
                if vehicle_roi.size > 0:
                    # Detect plates in vehicle ROI
                    roi_plates = self._detect_plates_in_image(vehicle_roi)
                    
                    # Adjust coordinates to full frame
                    for rx, ry, rw, rh in roi_plates:
                        plate_boxes.append((rx + roi_x, ry + roi_y, rw, rh))
        else:
            # No vehicle boxes provided, search in full frame
            plate_boxes = self._detect_plates_in_image(frame)
        
        # Update metrics
        self.plate_count += len(plate_boxes)
        
        process_time = time.time() - start_time
        self.process_times.append(process_time)
        
        # Keep only the last 100 processing times
        if len(self.process_times) > 100:
            self.process_times.pop(0)
        
        self.avg_process_time = sum(self.process_times) / len(self.process_times)
        
        return plate_boxes
    
    def _detect_plates_in_image(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect license plates in an image using selected method.
        
        Args:
            image: Input image
        
        Returns:
            List of license plate bounding boxes (x, y, w, h)
        """
        if self.yolo_model is not None:
            # Try YOLO detection first
            plate_boxes = self._detect_plates_yolo(image)
            
            # If YOLO found plates or use_haar is False, return those results
            if plate_boxes or not self.use_haar:
                return plate_boxes
        
        # If YOLO not available or found no plates, use Haar or edge detection
        if self.use_haar:
            return self._detect_plates_haar(image)
        else:
            return self._detect_plates_edge(image)
    
    def _detect_plates_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect license plates using Haar cascade.
        
        Args:
            image: Input image
        
        Returns:
            List of license plate bounding boxes
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect potential license plates
        plates = self.cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(self.min_plate_width, self.min_plate_height),
            maxSize=(self.max_plate_width, self.max_plate_height)
        )
        
        # Filter plates by aspect ratio
        filtered_plates = []
        for (x, y, w, h) in plates:
            aspect_ratio = w / h
            
            if (self.plate_aspect_ratio_min <= aspect_ratio <= self.plate_aspect_ratio_max):
                filtered_plates.append((x, y, w, h))
        
        return filtered_plates
    
    def _detect_plates_edge(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect license plates using edge-based detection.
        
        Args:
            image: Input image
        
        Returns:
            List of license plate bounding boxes
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Sobel edge detection
        grad_x = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur, cv2.CV_8U, 0, 1, ksize=3)
        
        # Combine gradients
        grad = cv2.addWeighted(grad_x, 1, grad_y, 1, 0)
        
        # Apply threshold
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find potential license plates
        plate_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if (self.min_plate_width <= w <= self.max_plate_width and 
                self.min_plate_height <= h <= self.max_plate_height):
                
                # Filter by aspect ratio
                aspect_ratio = w / h
                if self.plate_aspect_ratio_min <= aspect_ratio <= self.plate_aspect_ratio_max:
                    plate_boxes.append((x, y, w, h))
        
        return plate_boxes
    
    def _detect_plates_yolo(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect license plates using YOLO model.
        
        Args:
            image: Input image
        
        Returns:
            List of license plate bounding boxes
        """
        if self.yolo_model is None:
            return []
        
        # Get image dimensions
        height, width, _ = image.shape
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Set input and perform forward pass
        self.yolo_model.setInput(blob)
        outputs = self.yolo_model.forward(self.yolo_output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # YOLO returns center, width, height
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
        
        # Apply non-maximum suppression
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
            
            # Convert indices to the correct format based on OpenCV version
            if len(indices) > 0:
                # Check if indices is nested (older OpenCV versions)
                if isinstance(indices[0], (list, tuple, np.ndarray)):
                    indices = [i[0] for i in indices]
                
                # Get final boxes
                plate_boxes = []
                for i in indices:
                    box = tuple(boxes[i])
                    plate_boxes.append(box)
                
                return plate_boxes
        
        return []
    
    def extract_plate_images(self, frame: np.ndarray, plate_boxes: List[Tuple[int, int, int, int]], 
                           margin_percent: float = 0.1) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract license plate images from frame.
        
        Args:
            frame: Input frame
            plate_boxes: List of plate bounding boxes
            margin_percent: Margin to add around the plate as a percentage of dimensions
        
        Returns:
            List of tuples containing (plate_image, original_box)
        """
        plate_images = []
        
        for box in plate_boxes:
            x, y, w, h = box
            
            # Add margin
            margin_x = int(w * margin_percent)
            margin_y = int(h * margin_percent)
            
            # Calculate new coordinates with margin
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(frame.shape[1], x + w + margin_x)
            y2 = min(frame.shape[0], y + h + margin_y)
            
            # Extract plate image
            plate_image = frame[y1:y2, x1:x2].copy()
            
            if plate_image.size > 0:  # Check if plate image is not empty
                plate_images.append((plate_image, box))
        
        return plate_images
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get detector statistics.
        
        Returns:
            Dictionary of detector statistics
        """
        return {
            'detection_method': 'Haar' if self.use_haar else 'Edge-based',
            'yolo_available': self.yolo_model is not None,
            'frames_processed': self.frame_count,
            'plates_detected': self.plate_count,
            'plates_per_frame': round(self.plate_count / max(1, self.frame_count), 2),
            'avg_process_time_ms': round(self.avg_process_time * 1000, 2),
            'fps_capacity': round(1.0 / max(0.001, self.avg_process_time), 2)
        }


class LicensePlateRecognizer:
    """
    License plate recognizer for reading text from license plate images.
    """
    
    def __init__(
        self,
        tesseract_config: str = '--oem 1 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
        preprocess_resize: Optional[Tuple[int, int]] = (250, 50),
        confidence_threshold: float = 0.6,
        max_workers: int = 4,
        plate_patterns: Optional[List[str]] = None
    ):
        """
        Initialize license plate recognizer.
        
        Args:
            tesseract_config: Tesseract OCR configuration string
            preprocess_resize: Size to resize plates to before OCR, or None to keep original
            confidence_threshold: Minimum confidence threshold for plate text
            max_workers: Maximum number of parallel OCR workers
            plate_patterns: List of regex patterns for license plate validation
        """
        self.tesseract_config = tesseract_config
        self.preprocess_resize = preprocess_resize
        self.confidence_threshold = confidence_threshold
        self.max_workers = max_workers
        
        # Set default plate patterns if none provided
        if plate_patterns is None:
            self.plate_patterns = [
                r'^[A-Z]{3}\d{3,4}$',            # ABC123 or ABC1234
                r'^[A-Z]{2}\d{3,4}[A-Z]{1,2}$',  # AB123C or AB1234CD
                r'^\d{3,4}[A-Z]{3}$',            # 123ABC or 1234ABC
                r'^[A-Z]\d{3,4}[A-Z]{2}$',       # A123BC or A1234BC
                r'^\d{1,3}-[A-Z]{3}-\d{1,3}$'    # 1-ABC-123 or 123-ABC-1
            ]
        else:
            self.plate_patterns = plate_patterns
        
        # Compile regex patterns
        self.compiled_patterns = [re.compile(pattern) for pattern in self.plate_patterns]
        
        # Processing queue and thread pool
        self._processing_queue = []
        self._processing_lock = threading.Lock()
        self._processing_threads = []
        self._stop_event = threading.Event()
        
        # Start worker threads
        self._start_workers()
        
        # Performance metrics
        self.processed_count = 0
        self.recognized_count = 0
        self.process_times = []
        self.avg_process_time = 0
        
        logger.info("License plate recognizer initialized")
    
    def _start_workers(self):
        """Start worker threads for OCR processing."""
        for _ in range(self.max_workers):
            thread = threading.Thread(target=self._worker_process, daemon=True)
            thread.start()
            self._processing_threads.append(thread)
        
        logger.debug(f"Started {self.max_workers} OCR worker threads")
    
    def _worker_process(self):
        """Worker thread process for OCR."""
        while not self._stop_event.is_set():
            # Get next plate from queue
            plate_item = None
            with self._processing_lock:
                if self._processing_queue:
                    plate_item = self._processing_queue.pop(0)
            
            if plate_item:
                plate_image, callback = plate_item
                
                try:
                    # Process the plate
                    start_time = time.time()
                    plate_text, confidence = self._process_plate_image(plate_image)
                    process_time = time.time() - start_time
                    
                    # Update performance metrics
                    with self._processing_lock:
                        self.processed_count += 1
                        if plate_text:
                            self.recognized_count += 1
                        
                        self.process_times.append(process_time)
                        if len(self.process_times) > 100:
                            self.process_times.pop(0)
                        
                        self.avg_process_time = sum(self.process_times) / len(self.process_times)
                    
                    # Call the callback with the result
                    if callback:
                        callback(plate_text, confidence)
                
                except Exception as e:
                    logger.error(f"Error processing plate in worker: {str(e)}")
            
            # Sleep briefly to avoid busy waiting
            time.sleep(0.01)
    
    def _preprocess_plate_image(self, plate_image: np.ndarray) -> np.ndarray:
        """
        Preprocess license plate image for OCR.
        
        Args:
            plate_image: License plate image
        
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # Resize if needed
        if self.preprocess_resize:
            gray = cv2.resize(gray, self.preprocess_resize)
        
        # Apply bilateral filter to remove noise while keeping edges
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Potential image enhancement transformations can be added based on testing
        
        return morph
    
    def _process_plate_image(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        Process a license plate image and extract text.
        
        Args:
            plate_image: License plate image
        
        Returns:
            Tuple of (plate_text, confidence)
        """
        # Preprocess the plate image
        processed_img = self._preprocess_plate_image(plate_image)
        
        try:
            # Perform OCR using pytesseract
            ocr_result = pytesseract.image_to_data(
                processed_img, 
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and confidence
            text_parts = []
            confidences = []
            
            for i, conf in enumerate(ocr_result['conf']):
                if conf > 0:  # Filter out unrecognized parts
                    text = ocr_result['text'][i].strip()
                    if text:
                        text_parts.append(text)
                        confidences.append(conf)
            
            if not text_parts:
                return "", 0.0
            
            # Join text parts and calculate average confidence
            plate_text = ''.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) / 100.0  # Convert to 0-1 range
            
            # Clean up the text
            plate_text = self._clean_plate_text(plate_text)
            
            # Validate against patterns if confidence is above threshold
            if avg_confidence >= self.confidence_threshold:
                plate_text = self._validate_plate_format(plate_text)
            else:
                plate_text = ""
            
            return plate_text, avg_confidence
            
        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            return "", 0.0
    
    def _clean_plate_text(self, text: str) -> str:
        """
        Clean license plate text.
        
        Args:
            text: Raw plate text
        
        Returns:
            Cleaned text
        """
        # Remove spaces
        text = text.replace(" ", "")
        
        # Remove special characters except dashes
        text = re.sub(r'[^A-Z0-9\-]', '', text.upper())
        
        return text
    
    def _validate_plate_format(self, plate_text: str) -> str:
        """
        Validate plate against known formats.
        
        Args:
            plate_text: Plate text to validate
        
        Returns:
            Validated plate text or empty string if invalid
        """
        # Check against patterns
        for pattern in self.compiled_patterns:
            if pattern.match(plate_text):
                return plate_text
        
        # No pattern matched, try to fix common OCR errors
        fixed_text = self._fix_common_ocr_errors(plate_text)
        
        # Check fixed text against patterns
        for pattern in self.compiled_patterns:
            if pattern.match(fixed_text):
                return fixed_text
        
        # If no pattern matches, return empty to indicate invalid plate
        return ""
    
    def _fix_common_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR errors in license plate text.
        
        Args:
            text: Plate text with potential errors
        
        Returns:
            Corrected plate text
        """
        # Common OCR substitution errors
        substitutions = {
            '0': 'O', 'O': '0',
            '1': 'I', 'I': '1',
            '5': 'S', 'S': '5',
            '8': 'B', 'B': '8',
            '2': 'Z', 'Z': '2',
            '6': 'G', 'G': '6',
            '7': 'T', 'T': '7'
        }
        
        # Try each possible substitution
        for i, char in enumerate(text):
            if char in substitutions:
                # Create new text with substitution
                new_text = text[:i] + substitutions[char] + text[i+1:]
                
                # Check if new text matches any pattern
                for pattern in self.compiled_patterns:
                    if pattern.match(new_text):
                        return new_text
        
        return text
    
    def process_plate(self, plate_image: np.ndarray, callback=None) -> Optional[Tuple[str, float]]:
        """
        Process a license plate image asynchronously.
        
        Args:
            plate_image: License plate image
            callback: Optional callback function to receive results
        
        Returns:
            None if processing asynchronously, or Tuple of (plate_text, confidence) if processing synchronously
        """
        if callback:
            # Process asynchronously
            with self._processing_lock:
                self._processing_queue.append((plate_image, callback))
            return None
        else:
            # Process synchronously
            return self._process_plate_image(plate_image)
    
    def process_plate_batch(self, plate_images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Process a batch of license plate images synchronously.
        
        Args:
            plate_images: List of license plate images
        
        Returns:
            List of tuples with (plate_text, confidence)
        """
        results = []
        
        for plate_image in plate_images:
            text, confidence = self._process_plate_image(plate_image)
            results.append((text, confidence))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get recognizer statistics.
        
        Returns:
            Dictionary of recognizer statistics
        """
        with self._processing_lock:
            recognition_rate = 0
            if self.processed_count > 0:
                recognition_rate = self.recognized_count / self.processed_count
                
            return {
                'processed_count': self.processed_count,
                'recognized_count': self.recognized_count,
                'recognition_rate': round(recognition_rate, 2),
                'avg_process_time_ms': round(self.avg_process_time * 1000, 2),
                'queue_size': len(self._processing_queue),
                'worker_count': len(self._processing_threads)
            }
    
    def stop(self):
        """Stop processing and clean up resources."""
        self._stop_event.set()
        
        # Wait for all threads to finish
        for thread in self._processing_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        self._processing_threads = []
        self._processing_queue = []
        
        logger.info("License plate recognizer stopped")
    
    def __del__(self):
        """Clean up resources when object is deleted."""
        self.stop()


# Functions for manual testing and visualization

def draw_plates(frame: np.ndarray, plate_detections: List[PlateDetection]) -> np.ndarray:
    """
    Draw license plate detections on a frame.
    
    Args:
        frame: Input video frame
        plate_detections: List of plate detections
    
    Returns:
        Frame with drawn plate boxes and text
    """
    # Make a copy of the frame
    output = frame.copy()
    
    # Draw each plate detection
    for plate in plate_detections:
        x, y, w, h = plate.box
        
        # Draw bounding box
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw plate text
        if plate.plate_text:
            # Prepare label text
            label = f"{plate.plate_text} ({plate.confidence:.2f})"
            
            # Draw label background
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(output, (x, y - text_size[1] - 10), (x + text_size[0], y), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(output, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return output
