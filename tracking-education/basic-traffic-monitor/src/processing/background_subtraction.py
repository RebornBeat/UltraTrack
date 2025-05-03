"""
Background subtraction module for detecting moving objects in video streams.
Provides different methods for background modeling and motion detection.
"""

import logging
import time
from enum import Enum
from typing import Tuple, List, Dict, Any, Optional

import cv2
import numpy as np

# Configure logger for this module
logger = logging.getLogger(__name__)

class SubtractionMethod(Enum):
    """Enumeration of background subtraction methods."""
    MOG2 = 'MOG2'
    KNN = 'KNN'
    CNT = 'CNT'
    GMG = 'GMG'
    LSBP = 'LSBP'


class BackgroundSubtractor:
    """
    Background subtraction for detecting moving objects in video frames.
    """
    
    def __init__(
        self,
        method: SubtractionMethod = SubtractionMethod.MOG2,
        history: int = 500,
        learning_rate: float = 0.01,
        detect_shadows: bool = True,
        threshold: int = 25,
        kernel_size: int = 3,
        min_area: int = 100,
        max_area: Optional[int] = None,
        roi: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Initialize background subtractor.
        
        Args:
            method: Background subtraction algorithm to use
            history: Number of frames to keep in history
            learning_rate: Learning rate for background model update (0-1)
            detect_shadows: Whether to detect and mark shadows
            threshold: Threshold for foreground/background separation
            kernel_size: Size of morphological operation kernel
            min_area: Minimum contour area to consider
            max_area: Maximum contour area to consider (None for no limit)
            roi: Region of interest as (x, y, width, height) or None for full frame
        """
        self.method = method
        self.history = history
        self.learning_rate = learning_rate
        self.detect_shadows = detect_shadows
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.min_area = min_area
        self.max_area = max_area
        self.roi = roi
        
        # Create morphological kernels
        self.kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size * 2 + 1, kernel_size * 2 + 1))
        
        # Create background subtractor based on selected method
        self._create_subtractor()
        
        # Performance metrics
        self.frame_count = 0
        self.process_times = []
        self.avg_process_time = 0
        
        logger.info(f"Background subtractor initialized: {method.value}")
    
    def _create_subtractor(self):
        """Create the appropriate background subtractor based on method."""
        if self.method == SubtractionMethod.MOG2:
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.threshold,
                detectShadows=self.detect_shadows
            )
        elif self.method == SubtractionMethod.KNN:
            self.subtractor = cv2.createBackgroundSubtractorKNN(
                history=self.history,
                dist2Threshold=self.threshold,
                detectShadows=self.detect_shadows
            )
        elif self.method == SubtractionMethod.CNT:
            self.subtractor = cv2.bgsegm.createBackgroundSubtractorCNT(
                minPixelStability=15,
                useHistory=True,
                maxPixelStability=15 * 60,
                isParallel=True
            )
        elif self.method == SubtractionMethod.GMG:
            self.subtractor = cv2.bgsegm.createBackgroundSubtractorGMG(
                initializationFrames=self.history,
                decisionThreshold=self.threshold
            )
        elif self.method == SubtractionMethod.LSBP:
            self.subtractor = cv2.bgsegm.createBackgroundSubtractorLSBP()
        else:
            # Default to MOG2 if method not recognized
            logger.warning(f"Unknown subtraction method: {self.method}, using MOG2 instead")
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.threshold,
                detectShadows=self.detect_shadows
            )
    
    def apply(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Apply background subtraction to a frame.
        
        Args:
            frame: Input video frame
        
        Returns:
            Tuple of (foreground mask, list of bounding boxes)
        """
        start_time = time.time()
        
        # Extract ROI if specified
        if self.roi:
            x, y, w, h = self.roi
            roi_frame = frame[y:y+h, x:x+w]
        else:
            roi_frame = frame
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply background subtraction
        mask = self.subtractor.apply(blurred, learningRate=self.learning_rate)
        
        # Threshold to obtain binary mask
        if self.detect_shadows:
            # If shadows are detected (gray values around 127), convert to binary
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        else:
            binary_mask = mask
        
        # Apply morphological operations to clean up mask
        # Opening (erosion followed by dilation) removes small noise
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self.kernel_open)
        
        # Closing (dilation followed by erosion) closes small holes
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel_close)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and create bounding boxes
        bounding_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area:
                continue
                
            if self.max_area is not None and area > self.max_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Adjust coordinates if ROI is used
            if self.roi:
                roi_x, roi_y, _, _ = self.roi
                x += roi_x
                y += roi_y
            
            bounding_boxes.append((x, y, w, h))
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Update performance metrics
        self.frame_count += 1
        self.process_times.append(process_time)
        
        # Keep only the last 100 processing times for average calculation
        if len(self.process_times) > 100:
            self.process_times.pop(0)
        
        self.avg_process_time = sum(self.process_times) / len(self.process_times)
        
        # Return original mask and bounding boxes
        return mask, bounding_boxes
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get subtractor statistics.
        
        Returns:
            Dictionary of performance statistics
        """
        return {
            'method': self.method.value,
            'frames_processed': self.frame_count,
            'avg_process_time_ms': round(self.avg_process_time * 1000, 2),
            'fps_capacity': round(1.0 / max(0.001, self.avg_process_time), 2)
        }
    
    def reset(self):
        """Reset the background subtractor."""
        self._create_subtractor()
        self.frame_count = 0
        self.process_times = []
        self.avg_process_time = 0
        logger.info("Background subtractor reset")


class MotionDetector:
    """
    Motion detector using background subtraction and temporal analysis.
    """
    
    def __init__(
        self,
        subtractor: BackgroundSubtractor,
        min_motion_frames: int = 3,
        max_motion_frames: int = 30,
        cooldown_frames: int = 10
    ):
        """
        Initialize motion detector.
        
        Args:
            subtractor: Background subtractor object
            min_motion_frames: Minimum consecutive frames with motion to trigger detection
            max_motion_frames: Maximum frames to consider a single motion event
            cooldown_frames: Frames to wait after motion stops before resetting
        """
        self.subtractor = subtractor
        self.min_motion_frames = min_motion_frames
        self.max_motion_frames = max_motion_frames
        self.cooldown_frames = cooldown_frames
        
        # Tracking state
        self.motion_frames = 0
        self.cooldown_counter = 0
        self.is_motion_detected = False
        self.motion_boxes = []
        
        # Motion history for temporal filtering
        self.motion_history = []
        self.history_max_len = 5
        
        # Statistics
        self.total_detections = 0
        self.last_detection_time = 0
        
        logger.info("Motion detector initialized")
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
        """
        Detect motion in a frame.
        
        Args:
            frame: Input video frame
        
        Returns:
            Tuple of (motion detected flag, list of motion bounding boxes)
        """
        # Apply background subtraction
        _, boxes = self.subtractor.apply(frame)
        
        # Update motion history
        self.motion_history.append(boxes)
        if len(self.motion_history) > self.history_max_len:
            self.motion_history.pop(0)
        
        # Check if there is any motion
        has_motion = len(boxes) > 0
        
        # Temporal filtering to reduce false positives
        if has_motion:
            # Reset cooldown if new motion detected
            self.cooldown_counter = 0
            
            # Increment motion frame counter
            self.motion_frames += 1
            
            # Check if motion duration threshold reached
            if self.motion_frames >= self.min_motion_frames:
                if not self.is_motion_detected:
                    # New motion event
                    self.is_motion_detected = True
                    self.total_detections += 1
                    self.last_detection_time = time.time()
                    logger.debug("Motion detected")
                
                # Update motion boxes with current detection
                self.motion_boxes = self._merge_overlapping_boxes(boxes)
                
                # Check if motion event has been active too long
                if self.motion_frames > self.max_motion_frames:
                    # Reset motion to avoid locking in motion state
                    logger.debug(f"Motion event lasted too long ({self.motion_frames} frames), resetting")
                    self.motion_frames = self.min_motion_frames
            
        else:
            # No motion detected, increment cooldown
            self.cooldown_counter += 1
            
            # Check if cooldown period reached
            if self.cooldown_counter >= self.cooldown_frames:
                # Reset motion state after cooldown
                if self.is_motion_detected:
                    logger.debug("Motion stopped")
                
                self.is_motion_detected = False
                self.motion_frames = 0
                self.motion_boxes = []
        
        return self.is_motion_detected, self.motion_boxes
    
    def _merge_overlapping_boxes(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Merge overlapping bounding boxes to reduce duplicates.
        
        Args:
            boxes: List of bounding boxes as (x, y, w, h)
        
        Returns:
            List of merged bounding boxes
        """
        if not boxes:
            return []
        
        # Convert to format [x1, y1, x2, y2]
        boxes_xyxy = []
        for x, y, w, h in boxes:
            boxes_xyxy.append([x, y, x + w, y + h])
        
        # Sort by x coordinate
        boxes_xyxy.sort(key=lambda box: box[0])
        
        merged_boxes = []
        current_box = boxes_xyxy[0]
        
        for box in boxes_xyxy[1:]:
            # Check if boxes overlap
            current_x2 = current_box[2]
            current_y2 = current_box[3]
            x1, y1, x2, y2 = box
            
            # Compute IoU (Intersection over Union)
            x_overlap = max(0, min(current_x2, x2) - max(current_box[0], x1))
            y_overlap = max(0, min(current_y2, y2) - max(current_box[1], y1))
            
            intersection = x_overlap * y_overlap
            current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            box_area = (x2 - x1) * (y2 - y1)
            union = current_area + box_area - intersection
            
            iou = intersection / max(1e-6, union)
            
            if iou > 0.3:  # Threshold for considering boxes as overlapping
                # Merge boxes
                current_box = [
                    min(current_box[0], x1),
                    min(current_box[1], y1),
                    max(current_box[2], x2),
                    max(current_box[3], y2)
                ]
            else:
                # Add current box to merged list and start a new one
                merged_boxes.append(current_box)
                current_box = box
        
        # Add the last box
        merged_boxes.append(current_box)
        
        # Convert back to (x, y, w, h) format
        result = []
        for x1, y1, x2, y2 in merged_boxes:
            result.append((x1, y1, x2 - x1, y2 - y1))
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get detector statistics.
        
        Returns:
            Dictionary of performance statistics
        """
        return {
            'total_detections': self.total_detections,
            'is_motion_detected': self.is_motion_detected,
            'current_motion_frames': self.motion_frames,
            'current_cooldown': self.cooldown_counter,
            'last_detection_time': self.last_detection_time,
            'time_since_last_detection': round(time.time() - self.last_detection_time, 2) if self.last_detection_time > 0 else None
        }
    
    def reset(self):
        """Reset the motion detector state."""
        self.motion_frames = 0
        self.cooldown_counter = 0
        self.is_motion_detected = False
        self.motion_boxes = []
        self.motion_history = []
        logger.info("Motion detector reset")
