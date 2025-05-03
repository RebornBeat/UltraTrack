"""
Live display module for real-time visualization of traffic monitoring results.
Provides configurable display modes and overlay options, including traffic light visualization.
"""

import time
import logging
import threading
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

import cv2
import numpy as np

# Configure logger for this module
logger = logging.getLogger(__name__)


class DisplayMode(Enum):
    """Display mode for visualization."""
    RAW = 'raw'                     # Raw camera feed
    DETECTION = 'detection'         # Show detections
    TRACKING = 'tracking'           # Show tracking
    COUNTING = 'counting'           # Show counting zones
    SPEED = 'speed'                 # Show speed measurements
    FLOW = 'flow'                   # Show traffic flow
    LICENSE_PLATE = 'license_plate' # Show license plate recognition
    TRAFFIC_LIGHTS = 'traffic_lights' # Show traffic light states
    COMPOSITE = 'composite'         # Composite of multiple views


@dataclass
class DisplayConfig:
    """Configuration for display appearance."""
    show_boxes: bool = True
    show_labels: bool = True
    show_ids: bool = True
    show_trails: bool = True
    show_zones: bool = True
    show_speed: bool = True
    show_flow: bool = True
    show_plates: bool = True
    show_traffic_lights: bool = True
    window_name: str = "Traffic Monitoring"
    fullscreen: bool = False
    display_fps: int = 30
    overlay_alpha: float = 0.4
    font_scale: float = 0.7
    line_thickness: int = 2


class FrameProcessor:
    """
    Process frames for display with various visualization overlays.
    """
    
    def __init__(
        self,
        config: DisplayConfig = DisplayConfig()
    ):
        """
        Initialize frame processor.
        
        Args:
            config: Display configuration
        """
        self.config = config
        
        # Default colors for different vehicle types
        self.vehicle_colors = {
            'CAR': (0, 255, 0),        # Green
            'TRUCK': (0, 0, 255),      # Red
            'BUS': (255, 0, 0),        # Blue
            'MOTORCYCLE': (255, 0, 255), # Magenta
            'BICYCLE': (255, 255, 0),  # Cyan
            'PEDESTRIAN': (0, 255, 255), # Yellow
            'UNKNOWN': (128, 128, 128) # Gray
        }
        
        # Traffic light colors
        self.traffic_light_colors = {
            'red': (0, 0, 255),        # Red
            'yellow': (0, 255, 255),   # Yellow
            'green': (0, 255, 0),      # Green
            'flashing_red': (0, 0, 255),  # Red
            'flashing_yellow': (0, 255, 255),  # Yellow
            'off': (128, 128, 128)     # Gray
        }
        
        # Status display values
        self.fps = 0.0
        self.frame_count = 0
        self.processing_time = 0.0
        self.detection_count = 0
        self.tracking_count = 0
        
        # Traffic light flashing state
        self.flash_state = False
        self.last_flash_time = time.time()
        self.flash_interval = 0.5  # seconds
        
        logger.info("Frame processor initialized")
    
    def process_frame(
        self,
        frame: np.ndarray,
        mode: DisplayMode,
        detections: Optional[List[Any]] = None,
        tracks: Optional[List[Any]] = None,
        zones: Optional[Dict[str, Any]] = None,
        speeds: Optional[Dict[int, float]] = None,
        plates: Optional[List[Any]] = None,
        flow_data: Optional[Any] = None,
        traffic_lights: Optional[Dict[str, Any]] = None,
        processing_time: float = 0.0
    ) -> np.ndarray:
        """
        Process a frame for display with selected visualization mode.
        
        Args:
            frame: Input video frame
            mode: Display mode
            detections: Optional list of detection objects
            tracks: Optional list of track objects
            zones: Optional dictionary of zone objects
            speeds: Optional dictionary of vehicle_id -> speed
            plates: Optional list of license plate objects
            flow_data: Optional flow visualization data
            traffic_lights: Optional traffic light states
            processing_time: Processing time in milliseconds
        
        Returns:
            Processed frame for display
        """
        if frame is None:
            # Create blank frame if input is None
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "No Video Feed", (480, 360), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (255, 255, 255), 2)
            return frame
        
        # Make a copy of the frame for drawing
        display = frame.copy()
        
        # Update flashing state for traffic lights
        current_time = time.time()
        if current_time - self.last_flash_time > self.flash_interval:
            self.flash_state = not self.flash_state
            self.last_flash_time = current_time
        
        # Apply visualizations based on mode
        if mode == DisplayMode.RAW:
            # Just show raw frame
            pass
        
        elif mode == DisplayMode.DETECTION and detections:
            # Show detection boxes
            display = self._draw_detections(display, detections)
        
        elif mode == DisplayMode.TRACKING and tracks:
            # Show tracking visualization
            display = self._draw_tracks(display, tracks)
        
        elif mode == DisplayMode.COUNTING and zones:
            # Show counting zones
            display = self._draw_counting_zones(display, zones, tracks)
        
        elif mode == DisplayMode.SPEED and speeds:
            # Show speed measurements
            display = self._draw_speeds(display, speeds, tracks)
        
        elif mode == DisplayMode.FLOW and flow_data:
            # Show traffic flow
            display = self._draw_flow(display, flow_data)
        
        elif mode == DisplayMode.LICENSE_PLATE and plates:
            # Show license plate recognition
            display = self._draw_plates(display, plates)
        
        elif mode == DisplayMode.TRAFFIC_LIGHTS and traffic_lights:
            # Show traffic light states
            display = self._draw_traffic_lights(display, traffic_lights)
        
        elif mode == DisplayMode.COMPOSITE:
            # Show composite view with all available visualizations
            if detections and self.config.show_boxes:
                display = self._draw_detections(display, detections)
            
            if tracks and self.config.show_trails:
                display = self._draw_tracks(display, tracks)
            
            if zones and self.config.show_zones:
                display = self._draw_counting_zones(display, zones, tracks)
            
            if speeds and tracks and self.config.show_speed:
                display = self._draw_speeds(display, speeds, tracks)
            
            if flow_data and self.config.show_flow:
                display = self._draw_flow(display, flow_data)
            
            if plates and self.config.show_plates:
                display = self._draw_plates(display, plates)
                
            if traffic_lights and self.config.show_traffic_lights:
                display = self._draw_traffic_lights(display, traffic_lights)
        
        # Draw status overlay
        display = self._draw_status_overlay(display, processing_time)
        
        return display
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Any]) -> np.ndarray:
        """
        Draw detection boxes on the frame.
        
        Args:
            frame: Input video frame
            detections: List of detection objects
        
        Returns:
            Frame with drawn detections
        """
        if not self.config.show_boxes:
            return frame
        
        for detection in detections:
            # Get bounding box
            try:
                x, y, w, h = detection.box
            except AttributeError:
                # Try alternate formats
                try:
                    box = detection.box if hasattr(detection, 'box') else detection['box']
                    x, y, w, h = box
                except:
                    continue
            
            # Get vehicle type
            try:
                vehicle_type = detection.vehicle_type.name if hasattr(detection, 'vehicle_type') else 'UNKNOWN'
            except AttributeError:
                vehicle_type = 'UNKNOWN'
            
            # Get confidence
            try:
                confidence = detection.confidence if hasattr(detection, 'confidence') else 0.0
            except AttributeError:
                confidence = 0.0
            
            # Get color for vehicle type
            color = self.vehicle_colors.get(vehicle_type, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, self.config.line_thickness)
            
            # Draw label if enabled
            if self.config.show_labels:
                # Prepare label text
                label = f"{vehicle_type}"
                if confidence > 0:
                    label += f": {confidence:.2f}"
                
                # Draw label background
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                            self.config.font_scale, self.config.line_thickness)
                cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                          self.config.font_scale, (255, 255, 255), self.config.line_thickness)
        
        self.detection_count = len(detections)
        return frame
    
    def _draw_tracks(self, frame: np.ndarray, tracks: List[Any]) -> np.ndarray:
        """
        Draw tracking information on the frame.
        
        Args:
            frame: Input video frame
            tracks: List of track objects
        
        Returns:
            Frame with drawn tracks
        """
        if not tracks:
            return frame
        
        # Draw each track
        for track in tracks:
            # Get track ID
            try:
                track_id = track.id if hasattr(track, 'id') else track['id']
            except (AttributeError, KeyError):
                continue
            
            # Get box if available
            box = None
            try:
                box = track.box if hasattr(track, 'box') else track['box']
            except (AttributeError, KeyError):
                pass
            
            # Get centroids if available
            centroids = []
            try:
                centroids = track['centroids'] if isinstance(track, dict) and 'centroids' in track else []
            except:
                pass
            
            # Generate color based on ID
            color = self._get_color_for_id(track_id)
            
            # Draw box if available
            if box and self.config.show_boxes:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, self.config.line_thickness)
            
            # Draw ID if enabled
            if self.config.show_ids:
                if box:
                    x, y, w, h = box
                    text_pos = (x, y - 10)
                elif centroids and centroids[-1]:
                    text_pos = centroids[-1]
                else:
                    continue
                
                # Draw ID text
                cv2.putText(frame, f"ID: {track_id}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                          self.config.font_scale, color, self.config.line_thickness)
            
            # Draw trail if enabled and available
            if self.config.show_trails and centroids and len(centroids) >= 2:
                # Draw line connecting centroids
                for i in range(1, len(centroids)):
                    thickness = max(1, self.config.line_thickness - (len(centroids) - i) // 2)
                    cv2.line(frame, centroids[i-1], centroids[i], color, thickness)
        
        self.tracking_count = len(tracks)
        return frame
    
    def _draw_counting_zones(self, frame: np.ndarray, zones: Dict[str, Any], 
                          tracks: Optional[List[Any]] = None) -> np.ndarray:
        """
        Draw counting zones on the frame.
        
        Args:
            frame: Input video frame
            zones: Dictionary of zone objects
            tracks: Optional list of track objects
        
        Returns:
            Frame with drawn zones
        """
        if not self.config.show_zones:
            return frame
        
        # Draw each zone
        for zone_id, zone in zones.items():
            # Get zone polygon
            try:
                polygon = zone.polygon if hasattr(zone, 'polygon') else zone['polygon']
            except (AttributeError, KeyError):
                continue
            
            # Get zone name
            try:
                name = zone.name if hasattr(zone, 'name') else zone.get('name', zone_id)
            except (AttributeError, KeyError):
                name = zone_id
            
            # Get zone count if available
            count = None
            try:
                count = zone.count if hasattr(zone, 'count') else zone.get('count', None)
            except (AttributeError, KeyError):
                pass
            
            # Convert polygon to numpy array
            points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            
            # Draw zone polygon
            cv2.polylines(frame, [points], True, (0, 255, 0), self.config.line_thickness)
            
            # Draw zone label
            center_x = sum(p[0] for p in polygon) // len(polygon)
            center_y = sum(p[1] for p in polygon) // len(polygon)
            
            # Prepare label text
            label = name
            if count is not None:
                label += f": {count}"
            
            # Draw label
            cv2.putText(frame, label, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                      self.config.font_scale, (0, 255, 0), self.config.line_thickness)
        
        return frame
    
    def _draw_speeds(self, frame: np.ndarray, speeds: Dict[int, float], 
                   tracks: Optional[List[Any]] = None) -> np.ndarray:
        """
        Draw speed measurements on the frame.
        
        Args:
            frame: Input video frame
            speeds: Dictionary of vehicle_id -> speed
            tracks: Optional list of track objects
        
        Returns:
            Frame with drawn speed information
        """
        if not self.config.show_speed or not speeds:
            return frame
        
        # If we have tracks, match speeds to track positions
        if tracks:
            for track in tracks:
                # Get track ID
                try:
                    track_id = track.id if hasattr(track, 'id') else track['id']
                except (AttributeError, KeyError):
                    continue
                
                # Get speed for this track
                if track_id not in speeds:
                    continue
                
                speed = speeds[track_id]
                
                # Get position
                pos = None
                
                # Try to get box
                try:
                    box = track.box if hasattr(track, 'box') else track.get('box', None)
                    if box:
                        x, y, w, h = box
                        pos = (x + w // 2, y)
                except (AttributeError, KeyError):
                    pass
                
                # Try to get centroids if no box
                if not pos:
                    try:
                        centroids = track['centroids'] if isinstance(track, dict) and 'centroids' in track else []
                        if centroids:
                            pos = centroids[-1]
                    except:
                        pass
                
                # Draw speed if we have a position
                if pos:
                    # Draw speed text
                    cv2.putText(frame, f"{speed:.1f} km/h", pos, cv2.FONT_HERSHEY_SIMPLEX, 
                              self.config.font_scale, (255, 0, 0), self.config.line_thickness)
        
        # If no tracks or not all speeds matched to tracks, draw remaining speeds
        # in a simple list
        unmatched_speeds = speeds.copy()
        
        if tracks:
            for track in tracks:
                try:
                    track_id = track.id if hasattr(track, 'id') else track['id']
                    if track_id in unmatched_speeds:
                        del unmatched_speeds[track_id]
                except:
                    pass
        
        if unmatched_speeds:
            # Draw a list of speeds in the corner
            y_offset = 40
            for vehicle_id, speed in list(unmatched_speeds.items())[:10]:  # Limit to 10 entries
                cv2.putText(frame, f"ID {vehicle_id}: {speed:.1f} km/h", (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, (255, 0, 0), 
                          self.config.line_thickness)
                y_offset += 30
        
        return frame
    
    def _draw_flow(self, frame: np.ndarray, flow_data: Any) -> np.ndarray:
        """
        Draw traffic flow visualization on the frame.
        
        Args:
            frame: Input video frame
            flow_data: Flow visualization data
        
        Returns:
            Frame with drawn flow information
        """
        if not self.config.show_flow:
            return frame
        
        # Check what type of flow data we have
        
        # If flow data is a dictionary with density_map and flow_map
        if isinstance(flow_data, dict) and 'density_map' in flow_data and 'flow_map' in flow_data:
            density_map = flow_data['density_map']
            flow_map = flow_data['flow_map']
            
            # Draw density map overlay
            if density_map is not None and density_map.shape[:2] == frame.shape[:2]:
                # Create colored density map
                colored_density = cv2.applyColorMap(
                    (density_map * 255).astype(np.uint8), 
                    cv2.COLORMAP_JET
                )
                
                # Create mask for areas with significant density
                mask = (density_map > 0.05).astype(np.uint8) * 255
                
                # Apply mask to colored density
                masked_density = cv2.bitwise_and(colored_density, colored_density, mask=mask)
                
                # Blend with frame
                cv2.addWeighted(frame, 1.0, masked_density, self.config.overlay_alpha, 0, frame)
            
            # Draw flow vectors
            if flow_map is not None and flow_map.shape[:2] == frame.shape[:2]:
                # Draw flow vectors on a grid
                h, w = flow_map.shape[:2]
                grid_step = 20
                y, x = np.mgrid[0:h:grid_step, 0:w:grid_step].reshape(2, -1).astype(int)
                
                # Filter points outside the frame
                mask = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                x, y = x[mask], y[mask]
                
                # Get flow vectors at grid points
                fx, fy = flow_map[y, x, 0], flow_map[y, x, 1]
                
                # Calculate vector magnitudes
                magnitude = np.sqrt(fx**2 + fy**2)
                
                # Filter by threshold
                threshold = 0.05
                mask = magnitude > threshold
                x, y, fx, fy = x[mask], y[mask], fx[mask], fy[mask]
                
                # Scale for visualization
                scale = 10.0
                
                # Draw arrows
                for i in range(len(x)):
                    cv2.arrowedLine(
                        frame,
                        (int(x[i]), int(y[i])),
                        (int(x[i] + fx[i] * scale), int(y[i] + fy[i] * scale)),
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                        tipLength=0.3
                    )
        
        # If flow data is a list of flow objects
        elif isinstance(flow_data, (list, tuple)) and flow_data and hasattr(flow_data[0], 'region_id'):
            # Draw a box with flow statistics in the corner
            y_offset = 40
            
            for flow in flow_data[:5]:  # Limit to 5 entries
                # Get region ID
                region_id = flow.region_id if hasattr(flow, 'region_id') else 'unknown'
                
                # Get density
                density = flow.density.name if hasattr(flow, 'density') else 'unknown'
                
                # Get vehicle count
                count = flow.vehicle_count if hasattr(flow, 'vehicle_count') else 0
                
                # Get average speed
                avg_speed = flow.avg_speed if hasattr(flow, 'avg_speed') else 0.0
                
                # Draw flow info
                cv2.putText(frame, f"{region_id}: {density}, {count} vehicles, {avg_speed:.1f} km/h", 
                          (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, 
                          (0, 255, 255), self.config.line_thickness)
                y_offset += 30
        
        return frame
    
    def _draw_plates(self, frame: np.ndarray, plates: List[Any]) -> np.ndarray:
        """
        Draw license plate information on the frame.
        
        Args:
            frame: Input video frame
            plates: List of license plate objects
        
        Returns:
            Frame with drawn plate information
        """
        if not self.config.show_plates:
            return frame
        
        for plate in plates:
            # Get plate box
            try:
                box = plate.box if hasattr(plate, 'box') else plate['box']
                x, y, w, h = box
            except (AttributeError, KeyError):
                continue
            
            # Get plate text
            try:
                text = plate.plate_text if hasattr(plate, 'plate_text') else plate['plate_text']
            except (AttributeError, KeyError):
                text = ''
            
            # Get confidence
            try:
                confidence = plate.confidence if hasattr(plate, 'confidence') else plate.get('confidence', 0.0)
            except (AttributeError, KeyError):
                confidence = 0.0
            
            # Draw plate box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Draw plate text
            if text:
                # Prepare label
                label = f"{text}"
                if confidence > 0:
                    label += f" ({confidence:.2f})"
                
                # Draw label background
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                            self.config.font_scale, self.config.line_thickness)
                cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0], y), (0, 0, 255), -1)
                
                # Draw label text
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                          self.config.font_scale, (255, 255, 255), self.config.line_thickness)
        
        return frame
    
    def _draw_traffic_lights(self, frame: np.ndarray, traffic_lights: Dict[str, Any]) -> np.ndarray:
        """
        Draw traffic light states on the frame.
        
        Args:
            frame: Input video frame
            traffic_lights: Dictionary of intersection_id -> traffic light state
        
        Returns:
            Frame with drawn traffic light states
        """
        if not self.config.show_traffic_lights or not traffic_lights:
            return frame
        
        # Draw panel with traffic light states
        panel_width = 300
        panel_height = len(traffic_lights) * 120 + 40
        panel_x = frame.shape[1] - panel_width - 10
        panel_y = 10
        
        # Create semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                    (panel_x + panel_width, panel_y + panel_height), 
                    (32, 32, 32), -1)
        
        # Add transparency
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw panel title
        cv2.putText(frame, "Traffic Light Status", 
                  (panel_x + 10, panel_y + 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw each intersection
        y_offset = panel_y + 60
        
        for intersection_id, state in traffic_lights.items():
            # Draw intersection name
            cv2.putText(frame, f"Intersection: {intersection_id}", 
                      (panel_x + 10, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw signals
            if 'signals' in state:
                signal_x = panel_x + 15
                signal_y = y_offset + 25
                
                for signal_id, signal_data in state['signals'].items():
                    # Get signal state
                    signal_state = signal_data.get('state', 'red').lower()
                    
                    # Get color for state
                    color = self.traffic_light_colors.get(signal_state, (128, 128, 128))
                    
                    # For flashing states, handle flashing
                    if 'flashing' in signal_state and not self.flash_state:
                        color = (128, 128, 128)  # Gray when off in flash cycle
                    
                    # Draw traffic light
                    self._draw_traffic_light_icon(frame, signal_x, signal_y, color)
                    
                    # Draw signal ID
                    cv2.putText(frame, signal_id, 
                              (signal_x + 25, signal_y + 8), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Update position for next signal
                    signal_x += 70
                    if signal_x > panel_x + panel_width - 60:
                        signal_x = panel_x + 15
                        signal_y += 30
            
            # Draw active plan
            if 'active_plan_id' in state:
                cv2.putText(frame, f"Plan: {state['active_plan_id']}", 
                          (panel_x + 10, y_offset + 85), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
            
            # Update y_offset for next intersection
            y_offset += 120
        
        return frame
    
    def _draw_traffic_light_icon(self, frame: np.ndarray, x: int, y: int, color: Tuple[int, int, int]) -> None:
        """
        Draw a traffic light icon.
        
        Args:
            frame: Input video frame
            x: X position
            y: Y position
            color: Light color (BGR)
        """
        # Draw traffic light housing
        cv2.rectangle(frame, (x, y), (x + 20, y + 50), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y), (x + 20, y + 50), (100, 100, 100), 2)
        
        # Draw light (one active light in the housing)
        light_y = 0
        if np.array_equal(color, self.traffic_light_colors['red']):
            light_y = y + 10
        elif np.array_equal(color, self.traffic_light_colors['yellow']):
            light_y = y + 25
        elif np.array_equal(color, self.traffic_light_colors['green']):
            light_y = y + 40
        else:
            # For unknown states, center the light
            light_y = y + 25
        
        # Draw the active light
        cv2.circle(frame, (x + 10, light_y), 7, color, -1)
        cv2.circle(frame, (x + 10, light_y), 7, (255, 255, 255), 1)
    
    def _draw_status_overlay(self, frame: np.ndarray, processing_time: float) -> np.ndarray:
        """
        Draw status overlay on the frame.
        
        Args:
            frame: Input video frame
            processing_time: Processing time in milliseconds
        
        Returns:
            Frame with status overlay
        """
        # Draw status box in the top-left corner
        fps_text = f"FPS: {self.fps:.1f}"
        detection_text = f"Detections: {self.detection_count}"
        tracking_text = f"Tracks: {self.tracking_count}"
        time_text = f"Processing: {processing_time:.1f} ms"
        
        y_offset = 30
        cv2.putText(frame, fps_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                  self.config.font_scale, (255, 255, 255), self.config.line_thickness)
        
        y_offset += 30
        cv2.putText(frame, detection_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                  self.config.font_scale, (255, 255, 255), self.config.line_thickness)
        
        y_offset += 30
        cv2.putText(frame, tracking_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                  self.config.font_scale, (255, 255, 255), self.config.line_thickness)
        
        y_offset += 30
        cv2.putText(frame, time_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                  self.config.font_scale, (255, 255, 255), self.config.line_thickness)
        
        return frame
    
    def _get_color_for_id(self, track_id: int) -> Tuple[int, int, int]:
        """
        Generate a consistent color based on track ID.
        
        Args:
            track_id: Tracking identifier
        
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
    
    def update_stats(self, fps: float, processing_time: float):
        """
        Update performance statistics.
        
        Args:
            fps: Current frames per second
            processing_time: Processing time in milliseconds
        """
        self.fps = fps
        self.processing_time = processing_time
        self.frame_count += 1


class DisplayManager:
    """
    Manage display windows and handle user interaction.
    """
    
    def __init__(
        self,
        config: DisplayConfig = DisplayConfig(),
        enable_ui_controls: bool = True
    ):
        """
        Initialize display manager.
        
        Args:
            config: Display configuration
            enable_ui_controls: Whether to enable keyboard/mouse controls
        """
        self.config = config
        self.enable_ui_controls = enable_ui_controls
        
        # Display mode
        self.mode = DisplayMode.COMPOSITE
        
        # Frame processor
        self.processor = FrameProcessor(config)
        
        # Display state
        self.window_created = False
        self.running = False
        self.paused = False
        self.frame = None
        self.last_update_time = 0
        self.frame_rate_limiter = 1.0 / max(1, config.display_fps)
        
        # Key callback functions
        self.key_callbacks = {}
        
        # Default key bindings
        if enable_ui_controls:
            self._setup_default_key_bindings()
        
        logger.info("Display manager initialized")
    
    def _setup_default_key_bindings(self):
        """Set up default keyboard controls."""
        # Mode switching
        self.register_key_callback(ord('1'), lambda: self.set_mode(DisplayMode.RAW))
        self.register_key_callback(ord('2'), lambda: self.set_mode(DisplayMode.DETECTION))
        self.register_key_callback(ord('3'), lambda: self.set_mode(DisplayMode.TRACKING))
        self.register_key_callback(ord('4'), lambda: self.set_mode(DisplayMode.COUNTING))
        self.register_key_callback(ord('5'), lambda: self.set_mode(DisplayMode.SPEED))
        self.register_key_callback(ord('6'), lambda: self.set_mode(DisplayMode.FLOW))
        self.register_key_callback(ord('7'), lambda: self.set_mode(DisplayMode.LICENSE_PLATE))
        self.register_key_callback(ord('8'), lambda: self.set_mode(DisplayMode.TRAFFIC_LIGHTS))
        self.register_key_callback(ord('9'), lambda: self.set_mode(DisplayMode.COMPOSITE))
        
        # Display options
        self.register_key_callback(ord('b'), lambda: self.toggle_config('show_boxes'))
        self.register_key_callback(ord('l'), lambda: self.toggle_config('show_labels'))
        self.register_key_callback(ord('i'), lambda: self.toggle_config('show_ids'))
        self.register_key_callback(ord('t'), lambda: self.toggle_config('show_trails'))
        self.register_key_callback(ord('z'), lambda: self.toggle_config('show_zones'))
        self.register_key_callback(ord('s'), lambda: self.toggle_config('show_speed'))
        self.register_key_callback(ord('f'), lambda: self.toggle_config('show_flow'))
        self.register_key_callback(ord('p'), lambda: self.toggle_config('show_plates'))
        self.register_key_callback(ord('g'), lambda: self.toggle_config('show_traffic_lights'))
        
        # Playback control
        self.register_key_callback(ord(' '), lambda: self.toggle_pause())
        
        # Window control
        self.register_key_callback(ord('w'), lambda: self.toggle_fullscreen())
        
        # Exit
        self.register_key_callback(ord('q'), lambda: self.stop())
        self.register_key_callback(27, lambda: self.stop())  # ESC key
    
    def toggle_config(self, option: str):
        """
        Toggle a boolean configuration option.
        
        Args:
            option: Name of the option to toggle
        """
        if hasattr(self.config, option):
            setattr(self.config, option, not getattr(self.config, option))
            logger.debug(f"Toggled {option} to {getattr(self.config, option)}")
    
    def toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
        logger.debug(f"Display {'paused' if self.paused else 'resumed'}")
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        self.config.fullscreen = not self.config.fullscreen
        
        if self.window_created:
            # Update window properties
            if self.config.fullscreen:
                cv2.setWindowProperty(self.config.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(self.config.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        
        logger.debug(f"Fullscreen mode {'enabled' if self.config.fullscreen else 'disabled'}")
    
    def set_mode(self, mode: DisplayMode):
        """
        Set the display mode.
        
        Args:
            mode: New display mode
        """
        self.mode = mode
        logger.debug(f"Display mode set to {mode.value}")
    
    def register_key_callback(self, key: int, callback: Callable[[], None]):
        """
        Register a callback function for a keyboard key.
        
        Args:
            key: Key code
            callback: Function to call when key is pressed
        """
        self.key_callbacks[key] = callback
    
    def start(self):
        """Start the display manager."""
        if not self.running:
            self.running = True
            self.create_window()
            logger.info(f"Display manager started with mode {self.mode.value}")
    
    def stop(self):
        """Stop the display manager."""
        self.running = False
        
        if self.window_created:
            cv2.destroyWindow(self.config.window_name)
            self.window_created = False
        
        logger.info("Display manager stopped")
    
    def create_window(self):
        """Create the display window."""
        if not self.window_created:
            cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
            
            if self.config.fullscreen:
                cv2.setWindowProperty(self.config.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            self.window_created = True
            logger.debug(f"Created window '{self.config.window_name}'")
    
    def update(
        self,
        frame: np.ndarray,
        detections: Optional[List[Any]] = None,
        tracks: Optional[List[Any]] = None,
        zones: Optional[Dict[str, Any]] = None,
        speeds: Optional[Dict[int, float]] = None,
        plates: Optional[List[Any]] = None,
        flow_data: Optional[Any] = None,
        traffic_lights: Optional[Dict[str, Any]] = None,
        processing_time: float = 0.0
    ):
        """
        Update display with new frame and data.
        
        Args:
            frame: Input video frame
            detections: Optional list of detection objects
            tracks: Optional list of track objects
            zones: Optional dictionary of zone objects
            speeds: Optional dictionary of vehicle_id -> speed
            plates: Optional list of license plate objects
            flow_data: Optional flow visualization data
            traffic_lights: Optional traffic light states
            processing_time: Processing time in milliseconds
        """
        if not self.running:
            return
        
        # Check if we need to limit frame rate
        current_time = time.time()
        time_since_last_update = current_time - self.last_update_time
        
        if time_since_last_update < self.frame_rate_limiter:
            # Not enough time has passed, skip this update
            return
        
        self.last_update_time = current_time
        
        # Skip processing if paused
        if self.paused and self.frame is not None:
            # When paused, just show the last processed frame
            cv2.imshow(self.config.window_name, self.frame)
            self._handle_user_input()
            return
        
        # Process frame
        start_time = time.time()
        processed_frame = self.processor.process_frame(
            frame, self.mode, detections, tracks, zones, speeds, plates, flow_data, traffic_lights, processing_time
        )
        display_time = (time.time() - start_time) * 1000
        
        # Calculate FPS
        fps = 1.0 / max(0.001, time_since_last_update)
        
        # Update performance stats
        self.processor.update_stats(fps, processing_time)
        
        # Store processed frame
        self.frame = processed_frame
        
        # Show frame
        cv2.imshow(self.config.window_name, processed_frame)
        
        # Handle user input
        self._handle_user_input()
    
    def _handle_user_input(self):
        """Handle keyboard and mouse input."""
        if not self.enable_ui_controls:
            # Wait 1ms for any key without processing it
            cv2.waitKey(1)
            return
        
        # Check for key press with 1ms timeout
        key = cv2.waitKey(1) & 0xFF
        
        if key != 255:  # 255 means no key pressed
            # Check if we have a callback for this key
            if key in self.key_callbacks:
                self.key_callbacks[key]()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get display manager status.
        
        Returns:
            Dictionary of status information
        """
        return {
            'running': self.running,
            'paused': self.paused,
            'mode': self.mode.value,
            'fps': self.processor.fps,
            'frame_count': self.processor.frame_count,
            'detection_count': self.processor.detection_count,
            'tracking_count': self.processor.tracking_count,
            'processing_time': self.processor.processing_time
        }
    
    def __del__(self):
        """Clean up resources on deletion."""
        self.stop()
