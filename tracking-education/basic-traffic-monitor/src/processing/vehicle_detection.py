"""
Vehicle detection module for identifying and classifying vehicles in video frames.
Provides object detection using various models and tracking across frames.
"""

import os
import time
import logging
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass

import cv2
import numpy as np

# Configure logger for this module
logger = logging.getLogger(__name__)

class DetectionModel(Enum):
    """Enumeration of vehicle detection models."""
    YOLO = 'yolo'
    SSD = 'ssd'
    HAAR = 'haar'
    HOG = 'hog'


class VehicleType(Enum):
    """Enumeration of vehicle types."""
    UNKNOWN = 0
    CAR = 1
    TRUCK = 2
    BUS = 3
    MOTORCYCLE = 4
    BICYCLE = 5
    PEDESTRIAN = 6


@dataclass
class Detection:
    """Class for storing vehicle detection results."""
    box: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    vehicle_type: VehicleType
    id: Optional[int] = None  # Tracking ID if tracked across frames
    
    def center(self) -> Tuple[int, int]:
        """Get the center point of the bounding box."""
        x, y, w, h = self.box
        return (x + w // 2, y + h // 2)
    
    def area(self) -> int:
        """Get the area of the bounding box."""
        _, _, w, h = self.box
        return w * h
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary."""
        return {
            'box': self.box,
            'confidence': self.confidence,
            'vehicle_type': self.vehicle_type.name,
            'id': self.id
        }


class VehicleDetector:
    """
    Vehicle detector using various object detection models.
    """
    
    def __init__(
        self, 
        model_type: DetectionModel = DetectionModel.YOLO,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_width: int = 416,
        input_height: int = 416,
        use_gpu: bool = False
    ):
        """
        Initialize vehicle detector.
        
        Args:
            model_type: Detection model type
            model_path: Path to model files or None for default
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            input_width: Input width for the model
            input_height: Input height for the model
            use_gpu: Whether to use GPU acceleration if available
        """
        self.model_type = model_type
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_width = input_width
        self.input_height = input_height
        self.use_gpu = use_gpu
        
        # Detection model
        self.model = None
        self.class_names = []
        
        # Configure GPU usage if requested
        if use_gpu:
            try:
                cv2.setUseOptimized(True)
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    cv2.cuda.setDevice(0)
                    logger.info("GPU acceleration enabled")
                else:
                    logger.warning("GPU acceleration requested but no CUDA devices found")
                    self.use_gpu = False
            except:
                logger.warning("GPU acceleration requested but OpenCV built without CUDA support")
                self.use_gpu = False
        
        # Performance tracking
        self.frame_count = 0
        self.process_times = []
        self.avg_process_time = 0
        
        # Load the model
        self._load_model()
        
        logger.info(f"Vehicle detector initialized: {model_type.value}")
    
    def _load_model(self):
        """Load the appropriate detection model based on type."""
        if self.model_type == DetectionModel.YOLO:
            self._load_yolo_model()
        elif self.model_type == DetectionModel.SSD:
            self._load_ssd_model()
        elif self.model_type == DetectionModel.HAAR:
            self._load_haar_cascades()
        elif self.model_type == DetectionModel.HOG:
            self._load_hog_detector()
        else:
            logger.error(f"Unsupported model type: {self.model_type}")
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _load_yolo_model(self):
        """Load YOLO model for object detection."""
        try:
            # Default model paths
            if self.model_path is None:
                # Use current directory
                current_dir = os.path.dirname(os.path.abspath(__file__))
                base_path = os.path.join(current_dir, "../models/yolo")
                
                cfg_path = os.path.join(base_path, "yolov4.cfg")
                weights_path = os.path.join(base_path, "yolov4.weights")
                names_path = os.path.join(base_path, "coco.names")
            else:
                # Use provided path
                base_path = self.model_path
                cfg_path = os.path.join(base_path, "yolov4.cfg")
                weights_path = os.path.join(base_path, "yolov4.weights")
                names_path = os.path.join(base_path, "coco.names")
            
            # Check if files exist
            if not os.path.exists(cfg_path) or not os.path.exists(weights_path):
                logger.error(f"YOLO model files not found at {base_path}")
                raise FileNotFoundError(f"YOLO model files not found at {base_path}")
            
            # Load class names
            if os.path.exists(names_path):
                with open(names_path, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            else:
                # Default COCO class names relevant to vehicles
                self.class_names = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
            
            # Load YOLO network
            self.model = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
            
            # Configure model
            if self.use_gpu:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Get output layer names
            self.output_layers = self.model.getUnconnectedOutLayersNames()
            
            # Map COCO class names to vehicle types
            self.class_map = {
                'car': VehicleType.CAR,
                'truck': VehicleType.TRUCK,
                'bus': VehicleType.BUS,
                'motorcycle': VehicleType.MOTORCYCLE,
                'bicycle': VehicleType.BICYCLE,
                'person': VehicleType.PEDESTRIAN
            }
            
            logger.info("YOLO model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            raise
    
    def _load_ssd_model(self):
        """Load SSD MobileNet model for object detection."""
        try:
            # Default model paths
            if self.model_path is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                base_path = os.path.join(current_dir, "../models/ssd")
                
                proto_path = os.path.join(base_path, "ssd_mobilenet_v2_coco.pbtxt")
                model_path = os.path.join(base_path, "ssd_mobilenet_v2_coco.pb")
            else:
                # Use provided path
                base_path = self.model_path
                proto_path = os.path.join(base_path, "ssd_mobilenet_v2_coco.pbtxt")
                model_path = os.path.join(base_path, "ssd_mobilenet_v2_coco.pb")
            
            # Check if files exist
            if not os.path.exists(proto_path) or not os.path.exists(model_path):
                logger.error(f"SSD model files not found at {base_path}")
                raise FileNotFoundError(f"SSD model files not found at {base_path}")
            
            # Load SSD model
            self.model = cv2.dnn.readNetFromTensorflow(model_path, proto_path)
            
            # Configure model
            if self.use_gpu:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # COCO class IDs for vehicles
            self.class_map = {
                1: VehicleType.PEDESTRIAN,  # person
                2: VehicleType.BICYCLE,     # bicycle
                3: VehicleType.CAR,         # car
                4: VehicleType.MOTORCYCLE,  # motorcycle
                6: VehicleType.BUS,         # bus
                8: VehicleType.TRUCK        # truck
            }
            
            logger.info("SSD model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading SSD model: {str(e)}")
            raise
    
    def _load_haar_cascades(self):
        """Load Haar cascade classifiers for vehicle detection."""
        try:
            # Default cascade paths
            if self.model_path is None:
                # Use OpenCV's built-in cascades
                self.car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
            else:
                # Use provided path
                cascade_path = os.path.join(self.model_path, "haarcascade_car.xml")
                if not os.path.exists(cascade_path):
                    logger.error(f"Haar cascade file not found at {cascade_path}")
                    raise FileNotFoundError(f"Haar cascade file not found at {cascade_path}")
                
                self.car_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Check if cascade loaded successfully
            if self.car_cascade.empty():
                logger.error("Failed to load Haar cascade classifier")
                raise RuntimeError("Failed to load Haar cascade classifier")
            
            self.model = self.car_cascade  # for consistency
            
            logger.info("Haar cascade classifier loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Haar cascade classifier: {str(e)}")
            raise
    
    def _load_hog_detector(self):
        """Load HOG-based person detector."""
        try:
            # HOG detector is built into OpenCV
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            self.model = self.hog  # for consistency
            
            logger.info("HOG detector loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading HOG detector: {str(e)}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect vehicles in a frame.
        
        Args:
            frame: Input video frame
        
        Returns:
            List of Detection objects
        """
        start_time = time.time()
        
        if self.model_type == DetectionModel.YOLO:
            detections = self._detect_yolo(frame)
        elif self.model_type == DetectionModel.SSD:
            detections = self._detect_ssd(frame)
        elif self.model_type == DetectionModel.HAAR:
            detections = self._detect_haar(frame)
        elif self.model_type == DetectionModel.HOG:
            detections = self._detect_hog(frame)
        else:
            logger.error(f"Unsupported model type for detection: {self.model_type}")
            return []
        
        # Update performance metrics
        process_time = time.time() - start_time
        self.frame_count += 1
        self.process_times.append(process_time)
        
        # Keep only the last 100 processing times
        if len(self.process_times) > 100:
            self.process_times.pop(0)
        
        self.avg_process_time = sum(self.process_times) / len(self.process_times)
        
        return detections
    
    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect vehicles using YOLO model.
        
        Args:
            frame: Input video frame
        
        Returns:
            List of Detection objects
        """
        # Get image dimensions
        height, width, _ = frame.shape
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (self.input_width, self.input_height), 
            swapRB=True, crop=False
        )
        
        # Set input and perform forward pass
        self.model.setInput(blob)
        outputs = self.model.forward(self.output_layers)
        
        # Initialize lists for detected bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []
        
        # Process each output layer
        for output in outputs:
            # Process each detection
            for detection in output:
                # Extract class scores
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter by confidence and vehicle classes
                if confidence > self.confidence_threshold and class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    
                    # Check if this is a vehicle class
                    if class_name in self.class_map:
                        # YOLO returns center, width, height
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Calculate top-left corner
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        # Add to lists
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
        
        # Apply non-maximum suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        # Create Detection objects for final detections
        detections = []
        
        # Convert indices to the correct format based on OpenCV version
        if len(indices) > 0:
            # Check if indices is nested (older OpenCV versions)
            if isinstance(indices[0], (list, tuple, np.ndarray)):
                indices = [i[0] for i in indices]
                
            for i in indices:
                box = tuple(boxes[i])
                confidence = confidences[i]
                class_name = self.class_names[class_ids[i]]
                vehicle_type = self.class_map.get(class_name, VehicleType.UNKNOWN)
                
                detections.append(Detection(box, confidence, vehicle_type))
        
        return detections
    
    def _detect_ssd(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect vehicles using SSD model.
        
        Args:
            frame: Input video frame
        
        Returns:
            List of Detection objects
        """
        # Get image dimensions
        height, width, _ = frame.shape
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (self.input_width, self.input_height), 
            (127.5, 127.5, 127.5), swapRB=True, crop=False
        )
        
        # Set input and perform forward pass
        self.model.setInput(blob)
        outputs = self.model.forward()
        
        # Create list for detections
        detections = []
        
        # Process detections
        for i in range(outputs.shape[2]):
            confidence = outputs[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                class_id = int(outputs[0, 0, i, 1])
                
                # Check if this is a vehicle class
                if class_id in self.class_map:
                    # SSD returns normalized coordinates
                    x1 = int(outputs[0, 0, i, 3] * width)
                    y1 = int(outputs[0, 0, i, 4] * height)
                    x2 = int(outputs[0, 0, i, 5] * width)
                    y2 = int(outputs[0, 0, i, 6] * height)
                    
                    # Convert to (x, y, w, h) format
                    x = max(0, x1)
                    y = max(0, y1)
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    
                    vehicle_type = self.class_map.get(class_id, VehicleType.UNKNOWN)
                    
                    detections.append(Detection((x, y, w, h), confidence, vehicle_type))
        
        return detections
    
    def _detect_haar(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect vehicles using Haar cascade classifier.
        
        Args:
            frame: Input video frame
        
        Returns:
            List of Detection objects
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect vehicles
        vehicles = self.car_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Create list for detections
        detections = []
        
        # Process detections
        for (x, y, w, h) in vehicles:
            # Haar cascades don't provide confidence scores
            # Use a fixed confidence or calculate based on detection size
            confidence = 0.7
            
            detections.append(Detection((x, y, w, h), confidence, VehicleType.CAR))
        
        return detections
    
    def _detect_hog(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect people using HOG detector.
        
        Args:
            frame: Input video frame
        
        Returns:
            List of Detection objects
        """
        # Detect people
        boxes, weights = self.hog.detectMultiScale(
            frame, 
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        
        # Create list for detections
        detections = []
        
        # Process detections
        for i, (x, y, w, h) in enumerate(boxes):
            confidence = float(weights[i])
            
            if confidence > self.confidence_threshold:
                detections.append(Detection((x, y, w, h), confidence, VehicleType.PEDESTRIAN))
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw detection boxes on the frame.
        
        Args:
            frame: Input video frame
            detections: List of detections
        
        Returns:
            Frame with drawn detections
        """
        # Make a copy of the frame
        output = frame.copy()
        
        # Define colors for different vehicle types
        colors = {
            VehicleType.CAR: (0, 255, 0),        # Green
            VehicleType.TRUCK: (0, 0, 255),      # Red
            VehicleType.BUS: (255, 0, 0),        # Blue
            VehicleType.MOTORCYCLE: (255, 0, 255), # Magenta
            VehicleType.BICYCLE: (255, 255, 0),  # Cyan
            VehicleType.PEDESTRIAN: (0, 255, 255), # Yellow
            VehicleType.UNKNOWN: (128, 128, 128) # Gray
        }
        
        # Draw each detection
        for detection in detections:
            x, y, w, h = detection.box
            
            # Get color for vehicle type
            color = colors.get(detection.vehicle_type, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label text
            label = f"{detection.vehicle_type.name}: {detection.confidence:.2f}"
            if detection.id is not None:
                label = f"ID {detection.id}: {label}"
            
            # Draw label background
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(output, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get detector statistics.
        
        Returns:
            Dictionary of performance statistics
        """
        return {
            'model_type': self.model_type.value,
            'frames_processed': self.frame_count,
            'avg_process_time_ms': round(self.avg_process_time * 1000, 2),
            'fps_capacity': round(1.0 / max(0.001, self.avg_process_time), 2),
            'gpu_enabled': self.use_gpu
        }


class VehicleTracker:
    """
    Track vehicles across multiple frames using their detections.
    """
    
    def __init__(
        self, 
        max_distance: float = 100.0,
        max_frames_to_skip: int = 10,
        max_trace_length: int = 20,
        min_detection_confidence: float = 0.5
    ):
        """
        Initialize vehicle tracker.
        
        Args:
            max_distance: Maximum distance between detections to consider them the same object
            max_frames_to_skip: Maximum number of frames a track can be lost before removing
            max_trace_length: Maximum length of track history to maintain
            min_detection_confidence: Minimum confidence for detections to be tracked
        """
        self.max_distance = max_distance
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.min_detection_confidence = min_detection_confidence
        
        # Tracking state
        self.tracks = []  # List of active tracks
        self.next_track_id = 1
        self.frame_count = 0
        
        # Track statistics
        self.total_tracks = 0
        self.active_tracks = 0
        
        logger.info("Vehicle tracker initialized")
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of new detections
        
        Returns:
            List of updated detections with tracking IDs
        """
        self.frame_count += 1
        
        # Filter detections by confidence
        confident_detections = [d for d in detections if d.confidence >= self.min_detection_confidence]
        
        # If first frame, initialize tracks
        if not self.tracks:
            for detection in confident_detections:
                self._add_new_track(detection)
            
            # Return detections with assigned IDs
            return confident_detections
        
        # Get centroids of current detections
        detection_centroids = [d.center() for d in confident_detections]
        
        # If no detections, update all tracks with missed detection
        if not detection_centroids:
            for track in self.tracks:
                track['frames_since_detection'] += 1
            
            # Remove tracks that have been lost for too long
            self._remove_lost_tracks()
            
            return []
        
        # Calculate distance matrix between existing tracks and new detections
        distance_matrix = np.zeros((len(self.tracks), len(detection_centroids)))
        
        for i, track in enumerate(self.tracks):
            for j, centroid in enumerate(detection_centroids):
                distance_matrix[i, j] = self._calculate_distance(track['centroids'][-1], centroid)
        
        # Hungarian algorithm for assignment
        from scipy.optimize import linear_sum_assignment
        track_indices, detection_indices = linear_sum_assignment(distance_matrix)
        
        # Mark all tracks as unmatched initially
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(confident_detections)))
        
        # Update tracks with matched detections
        for track_idx, detection_idx in zip(track_indices, detection_indices):
            # Check if the distance is within threshold
            if distance_matrix[track_idx, detection_idx] <= self.max_distance:
                self._update_track(track_idx, confident_detections[detection_idx])
                unmatched_tracks.discard(track_idx)
                unmatched_detections.discard(detection_idx)
        
        # Update unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx]['frames_since_detection'] += 1
        
        # Add new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self._add_new_track(confident_detections[detection_idx])
        
        # Remove tracks that have been lost for too long
        self._remove_lost_tracks()
        
        # Update active track count
        self.active_tracks = len(self.tracks)
        
        # Return updated detections with tracking IDs
        updated_detections = confident_detections.copy()
        
        # Assign track IDs to detections
        for track in self.tracks:
            if track['detection_index'] is not None:
                updated_detections[track['detection_index']].id = track['id']
        
        return updated_detections
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
        
        Returns:
            Euclidean distance
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _add_new_track(self, detection: Detection):
        """
        Add a new track for the detection.
        
        Args:
            detection: New detection to track
        """
        centroid = detection.center()
        
        track = {
            'id': self.next_track_id,
            'centroids': [centroid],
            'boxes': [detection.box],
            'vehicle_type': detection.vehicle_type,
            'confidence': detection.confidence,
            'frames_since_detection': 0,
            'total_tracked_frames': 1,
            'first_detection_time': time.time(),
            'last_detection_time': time.time(),
            'detection_index': None
        }
        
        self.tracks.append(track)
        self.next_track_id += 1
        self.total_tracks += 1
        
        # Assign ID to detection
        detection.id = track['id']
    
    def _update_track(self, track_idx: int, detection: Detection):
        """
        Update existing track with new detection.
        
        Args:
            track_idx: Index of track to update
            detection: New detection data
        """
        track = self.tracks[track_idx]
        centroid = detection.center()
        
        # Update track data
        track['centroids'].append(centroid)
        track['boxes'].append(detection.box)
        track['frames_since_detection'] = 0
        track['total_tracked_frames'] += 1
        track['last_detection_time'] = time.time()
        track['detection_index'] = self.frame_count
        
        # Limit trace length
        if len(track['centroids']) > self.max_trace_length:
            track['centroids'] = track['centroids'][-self.max_trace_length:]
            track['boxes'] = track['boxes'][-self.max_trace_length:]
    
    def _remove_lost_tracks(self):
        """Remove tracks that have been lost for too long."""
        self.tracks = [track for track in self.tracks 
                      if track['frames_since_detection'] <= self.max_frames_to_skip]
    
    def get_active_tracks(self) -> List[Dict[str, Any]]:
        """
        Get list of active tracks.
        
        Returns:
            List of active track dictionaries
        """
        return [
            {
                'id': track['id'],
                'vehicle_type': track['vehicle_type'].name,
                'confidence': track['confidence'],
                'last_position': track['centroids'][-1],
                'last_box': track['boxes'][-1],
                'frames_tracked': track['total_tracked_frames'],
                'frames_since_detection': track['frames_since_detection'],
                'tracking_duration': time.time() - track['first_detection_time']
            }
            for track in self.tracks
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get tracker statistics.
        
        Returns:
            Dictionary of tracker statistics
        """
        return {
            'total_tracks': self.total_tracks,
            'active_tracks': self.active_tracks,
            'frames_processed': self.frame_count
        }
    
    def draw_tracks(self, frame: np.ndarray, max_trail_length: int = 10) -> np.ndarray:
        """
        Draw tracking trails on the frame.
        
        Args:
            frame: Input video frame
            max_trail_length: Maximum length of trails to draw
        
        Returns:
            Frame with drawn tracks
        """
        # Make a copy of the frame
        output = frame.copy()
        
        # Draw each track
        for track in self.tracks:
            # Skip tracks that have been lost for a while
            if track['frames_since_detection'] > 3:
                continue
            
            # Get centroids for trail
            centroids = track['centroids'][-max_trail_length:]
            
            # Get color based on track ID
            color = self._get_color_for_id(track['id'])
            
            # Draw trail
            for i in range(1, len(centroids)):
                thickness = max(1, min(3, i // 2))
                cv2.line(output, centroids[i-1], centroids[i], color, thickness)
            
            # Draw last box
            if track['boxes']:
                last_box = track['boxes'][-1]
                x, y, w, h = last_box
                
                # Draw bounding box
                cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
                
                # Draw ID
                cv2.putText(output, f"ID: {track['id']}", (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output
    
    def _get_color_for_id(self, track_id: int) -> Tuple[int, int, int]:
        """
        Generate a consistent color based on track ID.
        
        Args:
            track_id: Track identifier
        
        Returns:
            RGB color tuple
        """
        # Use golden ratio to generate well-distributed colors
        hue = (track_id * 0.618033988749895) % 1.0
        
        # Convert HSV to RGB
        h = hue * 360
        s = 0.95
        v = 0.95
        
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        r = int((r + m) * 255)
        g = int((g + m) * 255)
        b = int((b + m) * 255)
        
        return (b, g, r)  # OpenCV uses BGR format
    
    def reset(self):
        """Reset the tracker state."""
        self.tracks = []
        self.frame_count = 0
        self.active_tracks = 0
        # Keep total_tracks and next_track_id to maintain unique IDs
        logger.info("Vehicle tracker reset")
