"""
Speed estimator for calculating vehicle speeds from video frames.
Implements camera calibration and speed measurement functionality.
"""

import time
import math
import logging
import threading
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set

import cv2
import numpy as np
from shapely.geometry import Point, LineString

from ..processing.vehicle_detection import Detection, VehicleType

# Configure logger for this module
logger = logging.getLogger(__name__)


class CalibrationMethod(Enum):
    """Enumeration of speed calibration methods."""
    PIXEL_DISTANCE = 'pixel_distance'  # Calibrate using pixel to meter ratio
    REFERENCE_LINE = 'reference_line'  # Calibrate using a reference line of known length
    HOMOGRAPHY = 'homography'          # Calibrate using homography matrix
    GPS_COORDS = 'gps_coords'          # Calibrate using GPS coordinates


@dataclass
class SpeedMeasurement:
    """Class representing a vehicle speed measurement."""
    vehicle_id: int
    timestamp: float
    speed: float  # in km/h
    position: Tuple[int, int]  # (x, y) in pixels
    vehicle_type: VehicleType
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'vehicle_id': self.vehicle_id,
            'timestamp': self.timestamp,
            'speed': self.speed,
            'position': self.position,
            'vehicle_type': self.vehicle_type.name,
            'confidence': self.confidence
        }


class SpeedEstimator:
    """
    Speed estimator for calculating vehicle speeds from video frames.
    """
    
    def __init__(
        self,
        fps: float,
        calibration_method: CalibrationMethod = CalibrationMethod.PIXEL_DISTANCE,
        calibration_params: Dict[str, Any] = None,
        min_tracking_points: int = 5,
        max_tracking_points: int = 30,
        smoothing_factor: float = 0.3,
        min_speed_confidence: float = 0.6,
        speed_lines: Optional[List[Tuple[List[Tuple[int, int]], float]]] = None,
        store_history: bool = True,
        max_history_items: int = 1000
    ):
        """
        Initialize speed estimator.
        
        Args:
            fps: Video frames per second
            calibration_method: Method to use for calibration
            calibration_params: Calibration parameters (specific to method)
            min_tracking_points: Minimum tracking points needed for speed calculation
            max_tracking_points: Maximum tracking points to store
            smoothing_factor: Smoothing factor for speed calculation (0-1)
            min_speed_confidence: Minimum confidence threshold for speed measurements
            speed_lines: Optional list of (line points, distance) tuples for line-based speed measurement
            store_history: Whether to store speed measurement history
            max_history_items: Maximum speed measurements to store
        """
        self.fps = fps
        self.calibration_method = calibration_method
        self.calibration_params = calibration_params or {}
        self.min_tracking_points = min_tracking_points
        self.max_tracking_points = max_tracking_points
        self.smoothing_factor = smoothing_factor
        self.min_speed_confidence = min_speed_confidence
        self.speed_lines = speed_lines
        self.store_history = store_history
        self.max_history_items = max_history_items
        
        # Check if parameters are valid for the selected calibration method
        self._validate_calibration_params()
        
        # Internal tracking state
        self.tracking_points = {}  # vehicle_id -> list of (timestamp, x, y) tuples
        self.vehicle_speeds = {}   # vehicle_id -> list of (timestamp, speed) tuples
        self.last_crossing = {}    # vehicle_id -> {line_index: timestamp}
        self.current_speeds = {}   # vehicle_id -> current speed estimate
        
        # History
        self.history = [] if store_history else None
        
        # Prepare speed lines if provided
        self.line_strings = None
        if speed_lines:
            self.line_strings = []
            for line_points, _ in speed_lines:
                self.line_strings.append(LineString(line_points))
        
        # Calibration
        self.meters_per_pixel = self._calculate_meters_per_pixel()
        
        # Statistics
        self.measurements_count = 0
        self.frames_processed = 0
        
        # Create lock for thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Speed estimator initialized with {calibration_method.value} calibration")
    
    def _validate_calibration_params(self):
        """Validate calibration parameters for the selected method."""
        if self.calibration_method == CalibrationMethod.PIXEL_DISTANCE:
            if 'meters_per_pixel' not in self.calibration_params:
                raise ValueError("meters_per_pixel parameter is required for PIXEL_DISTANCE calibration")
        
        elif self.calibration_method == CalibrationMethod.REFERENCE_LINE:
            if 'reference_line' not in self.calibration_params or 'real_length' not in self.calibration_params:
                raise ValueError("reference_line and real_length parameters are required for REFERENCE_LINE calibration")
        
        elif self.calibration_method == CalibrationMethod.HOMOGRAPHY:
            if 'homography_matrix' not in self.calibration_params:
                raise ValueError("homography_matrix parameter is required for HOMOGRAPHY calibration")
        
        elif self.calibration_method == CalibrationMethod.GPS_COORDS:
            if 'frame_corners_gps' not in self.calibration_params or 'frame_size' not in self.calibration_params:
                raise ValueError("frame_corners_gps and frame_size parameters are required for GPS_COORDS calibration")
    
    def _calculate_meters_per_pixel(self) -> float:
        """
        Calculate the meters per pixel ratio based on calibration method.
        
        Returns:
            Meters per pixel ratio
        """
        if self.calibration_method == CalibrationMethod.PIXEL_DISTANCE:
            return self.calibration_params['meters_per_pixel']
        
        elif self.calibration_method == CalibrationMethod.REFERENCE_LINE:
            # Calculate meters per pixel from reference line
            ref_line = self.calibration_params['reference_line']
            real_length = self.calibration_params['real_length']
            
            # Calculate pixel length
            pixel_length = math.sqrt((ref_line[1][0] - ref_line[0][0])**2 + 
                                    (ref_line[1][1] - ref_line[0][1])**2)
            
            return real_length / pixel_length
        
        elif self.calibration_method == CalibrationMethod.HOMOGRAPHY:
            # With homography, we convert coordinates for each point
            # For average value, use a point in the middle of the frame
            frame_size = self.calibration_params.get('frame_size', (1920, 1080))
            center_x, center_y = frame_size[0] // 2, frame_size[1] // 2
            
            # Define a 1-meter horizontal line in the middle of the frame
            p1 = np.array([center_x - 10, center_y, 1])
            p2 = np.array([center_x + 10, center_y, 1])
            
            # Apply homography to get real-world coordinates
            h = np.array(self.calibration_params['homography_matrix'])
            p1_real = h.dot(p1)
            p2_real = h.dot(p2)
            
            # Convert to 2D coordinates
            p1_real = p1_real[:2] / p1_real[2]
            p2_real = p2_real[:2] / p2_real[2]
            
            # Calculate distance in real world
            real_dist = math.sqrt((p2_real[0] - p1_real[0])**2 + (p2_real[1] - p1_real[1])**2)
            
            # Calculate meters per 20 pixels
            return real_dist / 20
        
        elif self.calibration_method == CalibrationMethod.GPS_COORDS:
            # Calculate average meters per pixel from GPS coordinates
            corners_gps = self.calibration_params['frame_corners_gps']
            frame_size = self.calibration_params['frame_size']
            
            # Calculate horizontal distance
            top_left = corners_gps[0]
            top_right = corners_gps[1]
            h_dist_meters = self._haversine_distance(top_left[1], top_left[0], top_right[1], top_right[0]) * 1000
            h_dist_pixels = frame_size[0]
            
            # Calculate vertical distance
            bottom_left = corners_gps[3]
            v_dist_meters = self._haversine_distance(top_left[1], top_left[0], bottom_left[1], bottom_left[0]) * 1000
            v_dist_pixels = frame_size[1]
            
            # Use average of horizontal and vertical ratios
            h_meters_per_pixel = h_dist_meters / h_dist_pixels
            v_meters_per_pixel = v_dist_meters / v_dist_pixels
            
            return (h_meters_per_pixel + v_meters_per_pixel) / 2
        
        else:
            logger.error(f"Unknown calibration method: {self.calibration_method}")
            return 0.01  # Fallback default
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the haversine distance between two GPS coordinates in kilometers.
        
        Args:
            lat1: Latitude of first point in degrees
            lon1: Longitude of first point in degrees
            lat2: Latitude of second point in degrees
            lon2: Longitude of second point in degrees
        
        Returns:
            Distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        return c * r
    
    def process_frame(self, detections: List[Detection], frame_timestamp: float) -> List[SpeedMeasurement]:
        """
        Process a frame of vehicle detections to calculate speeds.
        
        Args:
            detections: List of vehicle detections with tracking IDs
            frame_timestamp: Timestamp of the current frame
        
        Returns:
            List of new speed measurements in this frame
        """
        new_measurements = []
        
        with self.lock:
            self.frames_processed += 1
            
            # Filter detections with tracking IDs
            valid_detections = [d for d in detections if d.id is not None]
            
            # Get set of all current vehicle IDs
            current_vehicles = {d.id for d in valid_detections}
            
            # Process detections
            for detection in valid_detections:
                vehicle_id = detection.id
                center_x, center_y = detection.center()
                
                # Add tracking point
                if vehicle_id in self.tracking_points:
                    self.tracking_points[vehicle_id].append((frame_timestamp, center_x, center_y))
                    
                    # Limit number of tracking points
                    if len(self.tracking_points[vehicle_id]) > self.max_tracking_points:
                        self.tracking_points[vehicle_id] = self.tracking_points[vehicle_id][-self.max_tracking_points:]
                else:
                    self.tracking_points[vehicle_id] = [(frame_timestamp, center_x, center_y)]
                    self.vehicle_speeds[vehicle_id] = []
                    self.last_crossing[vehicle_id] = {}
                
                # Check if we have enough tracking points to calculate speed
                if len(self.tracking_points[vehicle_id]) >= self.min_tracking_points:
                    # Calculate speed based on method
                    if self.line_strings:
                        # Line-based speed measurement
                        speed, confidence = self._calculate_line_speed(vehicle_id, center_x, center_y, frame_timestamp)
                    else:
                        # Trajectory-based speed measurement
                        speed, confidence = self._calculate_trajectory_speed(vehicle_id)
                    
                    # Store results if above confidence threshold
                    if confidence >= self.min_speed_confidence:
                        # Apply smoothing to speed estimate
                        if vehicle_id in self.current_speeds:
                            prev_speed = self.current_speeds[vehicle_id]
                            speed = prev_speed * (1 - self.smoothing_factor) + speed * self.smoothing_factor
                        
                        self.current_speeds[vehicle_id] = speed
                        
                        # Store speed measurement
                        self.vehicle_speeds[vehicle_id].append((frame_timestamp, speed))
                        
                        # Create measurement object
                        measurement = SpeedMeasurement(
                            vehicle_id=vehicle_id,
                            timestamp=frame_timestamp,
                            speed=speed,
                            position=(center_x, center_y),
                            vehicle_type=detection.vehicle_type,
                            confidence=confidence
                        )
                        
                        # Add to history if enabled
                        if self.store_history:
                            self.history.append(measurement)
                            
                            # Trim history if needed
                            if len(self.history) > self.max_history_items:
                                self.history = self.history[-self.max_history_items:]
                        
                        # Add to new measurements
                        new_measurements.append(measurement)
                        self.measurements_count += 1
            
            # Clean up vehicles that are no longer tracked
            vehicles_to_remove = set(self.tracking_points.keys()) - current_vehicles
            for vehicle_id in vehicles_to_remove:
                self._clean_up_vehicle(vehicle_id)
        
        return new_measurements
    
    def _calculate_trajectory_speed(self, vehicle_id: int) -> Tuple[float, float]:
        """
        Calculate speed based on trajectory history.
        
        Args:
            vehicle_id: Vehicle ID
        
        Returns:
            Tuple of (speed in km/h, confidence)
        """
        # Get tracking points for this vehicle
        points = self.tracking_points[vehicle_id]
        
        # Use last n points for calculation (more stable)
        n = min(10, len(points))
        recent_points = points[-n:]
        
        # Calculate travel distance in pixels
        total_distance = 0
        for i in range(1, len(recent_points)):
            prev_t, prev_x, prev_y = recent_points[i-1]
            curr_t, curr_x, curr_y = recent_points[i]
            
            # Calculate distance between consecutive points
            distance = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            total_distance += distance
        
        # Calculate time elapsed
        start_time = recent_points[0][0]
        end_time = recent_points[-1][0]
        time_elapsed = end_time - start_time
        
        # Avoid division by zero
        if time_elapsed <= 0.001:
            return 0.0, 0.0
        
        # Convert to real-world distance using calibration
        distance_meters = total_distance * self.meters_per_pixel
        
        # Calculate speed (meters per second)
        speed_mps = distance_meters / time_elapsed
        
        # Convert to km/h
        speed_kmh = speed_mps * 3.6
        
        # Calculate confidence based on number of points and consistency
        points_confidence = min(1.0, len(recent_points) / self.min_tracking_points)
        
        # Calculate variance of speed estimates as a measure of consistency
        if len(recent_points) >= 3:
            speeds = []
            for i in range(1, len(recent_points)):
                prev_t, prev_x, prev_y = recent_points[i-1]
                curr_t, curr_x, curr_y = recent_points[i]
                
                segment_dist = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                segment_time = curr_t - prev_t
                
                if segment_time > 0.001:
                    segment_speed = (segment_dist * self.meters_per_pixel) / segment_time * 3.6
                    speeds.append(segment_speed)
            
            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                variance = sum((s - avg_speed)**2 for s in speeds) / len(speeds)
                consistency_confidence = 1.0 / (1.0 + variance / 100.0)  # Normalize variance
            else:
                consistency_confidence = 0.5
        else:
            consistency_confidence = 0.5
        
        # Combine confidences
        confidence = points_confidence * 0.7 + consistency_confidence * 0.3
        
        return speed_kmh, confidence
    
    def _calculate_line_speed(self, vehicle_id: int, x: int, y: int, timestamp: float) -> Tuple[float, float]:
        """
        Calculate speed based on line crossing times.
        
        Args:
            vehicle_id: Vehicle ID
            x: Current x coordinate
            y: Current y coordinate
            timestamp: Current timestamp
        
        Returns:
            Tuple of (speed in km/h, confidence)
        """
        if not self.line_strings or not self.speed_lines:
            return 0.0, 0.0
        
        # Check if vehicle crosses any lines
        point = Point(x, y)
        current_speed = 0.0
        current_confidence = 0.0
        
        # Get tracking history for vehicle
        history = self.tracking_points[vehicle_id]
        if len(history) < 2:
            return 0.0, 0.0
        
        # Get previous position
        prev_t, prev_x, prev_y = history[-2]
        prev_point = Point(prev_x, prev_y)
        
        # Create segment representing vehicle movement
        vehicle_segment = LineString([(prev_x, prev_y), (x, y)])
        
        # Check each line
        for i, (line_string, line_data) in enumerate(zip(self.line_strings, self.speed_lines)):
            line_points, real_distance = line_data
            
            # Check if vehicle crosses this line
            if line_string.intersects(vehicle_segment):
                # Vehicle crossed line, record timestamp
                if i in self.last_crossing[vehicle_id]:
                    # Calculate time between crossings
                    prev_crossing = self.last_crossing[vehicle_id][i]
                    time_diff = timestamp - prev_crossing
                    
                    # Calculate speed if time difference is significant
                    if time_diff > 0.1:
                        speed = real_distance / time_diff * 3.6  # Convert to km/h
                        
                        # Apply simple validation
                        if 1.0 <= speed <= 200.0:  # Reasonable speed range
                            current_speed = speed
                            current_confidence = 0.8
                
                # Update crossing time
                self.last_crossing[vehicle_id][i] = timestamp
        
        # If no new crossings or valid speed, use trajectory method as fallback
        if current_speed <= 0.1:
            return self._calculate_trajectory_speed(vehicle_id)
        
        return current_speed, current_confidence
    
    def _clean_up_vehicle(self, vehicle_id: int):
        """
        Clean up tracking data for a vehicle.
        
        Args:
            vehicle_id: Vehicle ID to clean up
        """
        if vehicle_id in self.tracking_points:
            del self.tracking_points[vehicle_id]
        if vehicle_id in self.vehicle_speeds:
            del self.vehicle_speeds[vehicle_id]
        if vehicle_id in self.last_crossing:
            del self.last_crossing[vehicle_id]
        if vehicle_id in self.current_speeds:
            del self.current_speeds[vehicle_id]
    
    def get_current_speeds(self) -> Dict[int, float]:
        """
        Get current speed estimates for all tracked vehicles.
        
        Returns:
            Dictionary of vehicle_id -> speed (km/h)
        """
        with self.lock:
            return self.current_speeds.copy()
    
    def get_speed_history(self, vehicle_id: Optional[int] = None, 
                         max_items: Optional[int] = None,
                         time_period: Optional[Tuple[float, float]] = None) -> List[SpeedMeasurement]:
        """
        Get speed measurement history.
        
        Args:
            vehicle_id: Optional vehicle ID to filter by
            max_items: Maximum items to return
            time_period: Optional time range as (start_time, end_time) timestamps
        
        Returns:
            List of SpeedMeasurement objects
        """
        if not self.store_history:
            logger.warning("History storage is disabled")
            return []
        
        with self.lock:
            # Filter measurements
            filtered = self.history.copy()
            
            # Filter by vehicle ID if specified
            if vehicle_id is not None:
                filtered = [m for m in filtered if m.vehicle_id == vehicle_id]
            
            # Filter by time period if specified
            if time_period:
                start_time, end_time = time_period
                filtered = [m for m in filtered if start_time <= m.timestamp <= end_time]
            
            # Sort by timestamp (newest first)
            filtered.sort(key=lambda m: m.timestamp, reverse=True)
            
            # Limit number of items if specified
            if max_items:
                filtered = filtered[:max_items]
            
            return filtered
    
    def get_avg_speed(self, vehicle_type: Optional[VehicleType] = None,
                     time_period: Optional[Tuple[float, float]] = None) -> float:
        """
        Get average speed for all or specific vehicle types.
        
        Args:
            vehicle_type: Optional vehicle type to filter by
            time_period: Optional time range as (start_time, end_time) timestamps
        
        Returns:
            Average speed in km/h
        """
        if not self.store_history:
            # Use current speeds as fallback
            with self.lock:
                speeds = list(self.current_speeds.values())
                return sum(speeds) / max(1, len(speeds)) if speeds else 0.0
        
        # Get all measurements
        measurements = self.get_speed_history(time_period=time_period)
        
        # Filter by vehicle type if specified
        if vehicle_type is not None:
            measurements = [m for m in measurements if m.vehicle_type == vehicle_type]
        
        # Calculate average speed
        if not measurements:
            return 0.0
        
        return sum(m.speed for m in measurements) / len(measurements)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get estimator statistics.
        
        Returns:
            Dictionary of estimator statistics
        """
        with self.lock:
            return {
                'calibration_method': self.calibration_method.value,
                'meters_per_pixel': self.meters_per_pixel,
                'measurements_count': self.measurements_count,
                'frames_processed': self.frames_processed,
                'currently_tracked': len(self.tracking_points),
                'history_enabled': self.store_history,
                'history_items': len(self.history) if self.store_history else 0
            }
    
    def reset(self):
        """Reset all tracking data."""
        with self.lock:
            self.tracking_points = {}
            self.vehicle_speeds = {}
            self.last_crossing = {}
            self.current_speeds = {}
            
            if self.store_history:
                self.history = []
            
            self.measurements_count = 0
            
            logger.info("Speed estimator reset")
    
    def update_calibration(self, calibration_method: CalibrationMethod, calibration_params: Dict[str, Any]):
        """
        Update calibration method and parameters.
        
        Args:
            calibration_method: New calibration method
            calibration_params: New calibration parameters
        """
        with self.lock:
            self.calibration_method = calibration_method
            self.calibration_params = calibration_params
            
            # Validate new parameters
            self._validate_calibration_params()
            
            # Recalculate meters per pixel
            self.meters_per_pixel = self._calculate_meters_per_pixel()
            
            logger.info(f"Updated calibration to {calibration_method.value}")
    
    def draw_speed_info(self, frame: np.ndarray, font_scale: float = 0.6, 
                       thickness: int = 2) -> np.ndarray:
        """
        Draw speed information on frame.
        
        Args:
            frame: Input video frame
            font_scale: Font scale for text
            thickness: Line thickness
        
        Returns:
            Frame with drawn speed information
        """
        # Make a copy of the frame
        output = frame.copy()
        
        with self.lock:
            # Draw speed lines if used
            if self.line_strings and self.speed_lines:
                for i, (line_points, _) in enumerate(self.speed_lines):
                    # Convert to numpy array
                    points = np.array(line_points, dtype=np.int32).reshape((-1, 1, 2))
                    
                    # Draw the line
                    cv2.polylines(output, [points], False, (0, 255, 255), 2)
                    
                    # Draw line label
                    mid_x = sum(p[0] for p in line_points) // len(line_points)
                    mid_y = sum(p[1] for p in line_points) // len(line_points)
                    cv2.putText(output, f"Line {i+1}", (mid_x, mid_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
            
            # Draw speed for each vehicle
            for vehicle_id, speed in self.current_speeds.items():
                # Get last position
                if vehicle_id in self.tracking_points and self.tracking_points[vehicle_id]:
                    _, x, y = self.tracking_points[vehicle_id][-1]
                    
                    # Draw speed
                    cv2.putText(output, f"{speed:.1f} km/h", (int(x), int(y - 10)), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
        
        return output
