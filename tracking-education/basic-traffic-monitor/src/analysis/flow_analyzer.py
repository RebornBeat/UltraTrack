"""
Traffic flow analyzer for analyzing vehicle movement patterns and traffic density.
Provides traffic flow visualization and analysis functionality.
"""

import time
import logging
import threading
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set, Union

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from ..processing.vehicle_detection import Detection, VehicleType

# Configure logger for this module
logger = logging.getLogger(__name__)


class TrafficDensity(Enum):
    """Enumeration of traffic density levels."""
    VERY_LOW = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    VERY_HIGH = 4
    CONGESTED = 5


class FlowDirection(Enum):
    """Enumeration of traffic flow directions."""
    UNKNOWN = 0
    NORTH = 1
    NORTHEAST = 2
    EAST = 3
    SOUTHEAST = 4
    SOUTH = 5
    SOUTHWEST = 6
    WEST = 7
    NORTHWEST = 8


@dataclass
class TrafficFlow:
    """Class representing traffic flow information for a region."""
    region_id: str
    timestamp: float
    density: TrafficDensity
    vehicle_count: int
    avg_speed: float
    dominant_direction: FlowDirection
    directional_counts: Dict[FlowDirection, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'region_id': self.region_id,
            'timestamp': self.timestamp,
            'density': self.density.name,
            'vehicle_count': self.vehicle_count,
            'avg_speed': self.avg_speed,
            'dominant_direction': self.dominant_direction.name,
            'directional_counts': {d.name: c for d, c in self.directional_counts.items()}
        }


class FlowAnalyzer:
    """
    Traffic flow analyzer for analyzing movement patterns and density.
    """
    
    def __init__(
        self,
        frame_size: Tuple[int, int],
        regions: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        grid_size: Optional[Tuple[int, int]] = None,
        density_thresholds: Optional[Dict[TrafficDensity, float]] = None,
        flow_history_length: int = 10,
        update_interval: int = 30,
        use_speed_data: bool = True,
        store_history: bool = True,
        max_history_items: int = 1000
    ):
        """
        Initialize flow analyzer.
        
        Args:
            frame_size: Size of the video frames (width, height)
            regions: Optional dictionary of region_id -> polygon points for analysis regions
            grid_size: Optional size of grid cells (width, height) for automatic region generation
            density_thresholds: Optional dictionary of density level -> vehicles per region thresholds
            flow_history_length: Number of frames to use for flow calculation
            update_interval: Number of frames between flow updates
            use_speed_data: Whether to use speed data in flow analysis
            store_history: Whether to store flow history
            max_history_items: Maximum flow history items to store
        """
        self.frame_size = frame_size
        self.flow_history_length = flow_history_length
        self.update_interval = update_interval
        self.use_speed_data = use_speed_data
        self.store_history = store_history
        self.max_history_items = max_history_items
        
        # Set default density thresholds if not provided
        if density_thresholds is None:
            self.density_thresholds = {
                TrafficDensity.VERY_LOW: 0.0,
                TrafficDensity.LOW: 0.05,
                TrafficDensity.MODERATE: 0.15,
                TrafficDensity.HIGH: 0.25,
                TrafficDensity.VERY_HIGH: 0.35,
                TrafficDensity.CONGESTED: 0.5
            }
        else:
            self.density_thresholds = density_thresholds
        
        # Set up regions
        if regions:
            self.regions = regions
        elif grid_size:
            # Create grid of regions
            self.regions = self._create_grid_regions(grid_size)
        else:
            # Default to single region covering entire frame
            self.regions = {
                'full_frame': [(0, 0), (frame_size[0], 0), 
                              (frame_size[0], frame_size[1]), (0, frame_size[1])]
            }
        
        # Initialize region data
        self.region_data = {}
        for region_id in self.regions:
            self.region_data[region_id] = {
                'vehicle_positions': [],  # List of (frame_index, vehicle_id, x, y, direction, speed) tuples
                'vehicle_counts': {},     # frame_index -> count
                'vehicle_speeds': {},     # frame_index -> avg speed
                'flow_directions': {},    # frame_index -> Dict[FlowDirection, count]
                'current_vehicles': set(),  # Set of currently tracked vehicle IDs
                'last_flow': None         # Last calculated TrafficFlow object
            }
        
        # Create density maps
        self.density_map = np.zeros(frame_size[::-1], dtype=np.float32)
        self.flow_map = np.zeros((frame_size[1], frame_size[0], 2), dtype=np.float32)
        
        # History
        self.history = [] if store_history else None
        
        # Statistics
        self.frames_processed = 0
        self.flow_updates = 0
        
        # Create lock for thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Flow analyzer initialized with {len(self.regions)} regions")
    
    def _create_grid_regions(self, grid_size: Tuple[int, int]) -> Dict[str, List[Tuple[int, int]]]:
        """
        Create grid of regions for analysis.
        
        Args:
            grid_size: Size of grid cells (width, height)
        
        Returns:
            Dictionary of region_id -> polygon points
        """
        regions = {}
        cell_width, cell_height = grid_size
        
        # Calculate number of cells in each dimension
        num_cols = self.frame_size[0] // cell_width
        num_rows = self.frame_size[1] // cell_height
        
        # Create grid cells
        for row in range(num_rows):
            for col in range(num_cols):
                # Calculate cell coordinates
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = min(x1 + cell_width, self.frame_size[0])
                y2 = min(y1 + cell_height, self.frame_size[1])
                
                # Create region polygon
                region_polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                
                # Add to regions dictionary
                region_id = f"grid_{row}_{col}"
                regions[region_id] = region_polygon
        
        return regions
    
    def _point_in_polygon(self, x: int, y: int, polygon: List[Tuple[int, int]]) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm.
        
        Args:
            x: X coordinate
            y: Y coordinate
            polygon: List of polygon vertices as (x, y) tuples
        
        Returns:
            True if point is inside polygon, False otherwise
        """
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _classify_direction(self, dx: float, dy: float) -> FlowDirection:
        """
        Classify movement direction based on delta x and y.
        
        Args:
            dx: Change in x coordinate
            dy: Change in y coordinate
        
        Returns:
            FlowDirection enum value
        """
        if abs(dx) < 0.001 and abs(dy) < 0.001:
            return FlowDirection.UNKNOWN
        
        # Calculate angle in radians
        angle = np.arctan2(-dy, dx)  # Negative dy because y increases downward in images
        
        # Convert to degrees and ensure positive angle
        angle_deg = np.degrees(angle)
        if angle_deg < 0:
            angle_deg += 360
        
        # Classify direction
        if angle_deg >= 337.5 or angle_deg < 22.5:
            return FlowDirection.EAST
        elif angle_deg >= 22.5 and angle_deg < 67.5:
            return FlowDirection.NORTHEAST
        elif angle_deg >= 67.5 and angle_deg < 112.5:
            return FlowDirection.NORTH
        elif angle_deg >= 112.5 and angle_deg < 157.5:
            return FlowDirection.NORTHWEST
        elif angle_deg >= 157.5 and angle_deg < 202.5:
            return FlowDirection.WEST
        elif angle_deg >= 202.5 and angle_deg < 247.5:
            return FlowDirection.SOUTHWEST
        elif angle_deg >= 247.5 and angle_deg < 292.5:
            return FlowDirection.SOUTH
        else:  # angle_deg >= 292.5 and angle_deg < 337.5
            return FlowDirection.SOUTHEAST
    
    def _get_dominant_direction(self, direction_counts: Dict[FlowDirection, int]) -> FlowDirection:
        """
        Get dominant flow direction from direction counts.
        
        Args:
            direction_counts: Dictionary of direction -> count
        
        Returns:
            Dominant FlowDirection
        """
        if not direction_counts:
            return FlowDirection.UNKNOWN
        
        # Remove unknown direction from consideration
        counts = {d: c for d, c in direction_counts.items() if d != FlowDirection.UNKNOWN}
        
        if not counts:
            return FlowDirection.UNKNOWN
        
        # Find direction with highest count
        return max(counts.items(), key=lambda x: x[1])[0]
    
    def _get_density_level(self, vehicle_count: int, region_area: float) -> TrafficDensity:
        """
        Determine traffic density level based on vehicle count and region area.
        
        Args:
            vehicle_count: Number of vehicles in region
            region_area: Area of region in pixels
        
        Returns:
            TrafficDensity enum value
        """
        # Calculate vehicle density as vehicles per pixel
        density = vehicle_count / max(1.0, region_area)
        
        # Classify based on thresholds
        if density >= self.density_thresholds[TrafficDensity.CONGESTED]:
            return TrafficDensity.CONGESTED
        elif density >= self.density_thresholds[TrafficDensity.VERY_HIGH]:
            return TrafficDensity.VERY_HIGH
        elif density >= self.density_thresholds[TrafficDensity.HIGH]:
            return TrafficDensity.HIGH
        elif density >= self.density_thresholds[TrafficDensity.MODERATE]:
            return TrafficDensity.MODERATE
        elif density >= self.density_thresholds[TrafficDensity.LOW]:
            return TrafficDensity.LOW
        else:
            return TrafficDensity.VERY_LOW
    
    def _calculate_region_area(self, polygon: List[Tuple[int, int]]) -> float:
        """
        Calculate area of a region polygon in pixels.
        
        Args:
            polygon: List of polygon vertices as (x, y) tuples
        
        Returns:
            Area in pixels
        """
        # Convert to numpy array
        polygon = np.array(polygon)
        
        # Calculate area using Shoelace formula
        x = polygon[:, 0]
        y = polygon[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        return area
    
    def process_frame(self, detections: List[Detection], speeds: Optional[Dict[int, float]] = None) -> Dict[str, TrafficFlow]:
        """
        Process a frame of vehicle detections to analyze traffic flow.
        
        Args:
            detections: List of vehicle detections with tracking IDs
            speeds: Optional dictionary of vehicle_id -> speed (km/h)
        
        Returns:
            Dictionary of region_id -> TrafficFlow object for regions with updated flow
        """
        updated_flows = {}
        
        with self.lock:
            frame_index = self.frames_processed
            self.frames_processed += 1
            
            # Filter detections with tracking IDs
            valid_detections = [d for d in detections if d.id is not None]
            
            # Process each region
            for region_id, region_polygon in self.regions.items():
                region_data = self.region_data[region_id]
                
                # Find vehicles in this region
                vehicles_in_region = []
                
                for detection in valid_detections:
                    vehicle_id = detection.id
                    center_x, center_y = detection.center()
                    
                    # Check if vehicle is in region
                    if self._point_in_polygon(center_x, center_y, region_polygon):
                        # Get vehicle speed if available
                        speed = speeds.get(vehicle_id, 0.0) if speeds else 0.0
                        
                        # Determine direction if vehicle was previously tracked
                        direction = FlowDirection.UNKNOWN
                        dx, dy = 0, 0
                        
                        # Find last position of this vehicle
                        prev_positions = [
                            (f, x, y) for f, vid, x, y, _, _ in region_data['vehicle_positions']
                            if vid == vehicle_id
                        ]
                        
                        if prev_positions:
                            # Sort by frame index (descending)
                            prev_positions.sort(reverse=True)
                            prev_frame, prev_x, prev_y = prev_positions[0]
                            
                            # Calculate movement vector
                            dx = center_x - prev_x
                            dy = center_y - prev_y
                            
                            # Classify direction
                            direction = self._classify_direction(dx, dy)
                        
                        # Record vehicle position
                        region_data['vehicle_positions'].append(
                            (frame_index, vehicle_id, center_x, center_y, direction, speed)
                        )
                        
                        # Add to current vehicles in region
                        vehicles_in_region.append((vehicle_id, direction, speed))
                        region_data['current_vehicles'].add(vehicle_id)
                
                # Update vehicle count for this frame
                region_data['vehicle_counts'][frame_index] = len(vehicles_in_region)
                
                # Update vehicle speeds for this frame
                speeds_in_region = [s for _, _, s in vehicles_in_region if s > 0]
                avg_speed = sum(speeds_in_region) / max(1, len(speeds_in_region)) if speeds_in_region else 0.0
                region_data['vehicle_speeds'][frame_index] = avg_speed
                
                # Update flow directions for this frame
                direction_counts = {}
                for _, direction, _ in vehicles_in_region:
                    direction_counts[direction] = direction_counts.get(direction, 0) + 1
                
                region_data['flow_directions'][frame_index] = direction_counts
                
                # Limit history to flow_history_length
                if len(region_data['vehicle_positions']) > self.flow_history_length * 10:
                    # Keep only the most recent entries
                    region_data['vehicle_positions'] = region_data['vehicle_positions'][-self.flow_history_length * 10:]
                
                # Clean up old frame data
                for key in list(region_data['vehicle_counts'].keys()):
                    if key < frame_index - self.flow_history_length:
                        del region_data['vehicle_counts'][key]
                
                for key in list(region_data['vehicle_speeds'].keys()):
                    if key < frame_index - self.flow_history_length:
                        del region_data['vehicle_speeds'][key]
                
                for key in list(region_data['flow_directions'].keys()):
                    if key < frame_index - self.flow_history_length:
                        del region_data['flow_directions'][key]
                
                # Update flow data at specified interval
                if frame_index % self.update_interval == 0:
                    flow = self._calculate_region_flow(region_id)
                    region_data['last_flow'] = flow
                    
                    if self.store_history and flow:
                        self.history.append(flow)
                        
                        # Trim history if needed
                        if len(self.history) > self.max_history_items:
                            self.history = self.history[-self.max_history_items:]
                    
                    if flow:
                        updated_flows[region_id] = flow
                        self.flow_updates += 1
            
            # Update density and flow maps
            self._update_density_map(valid_detections)
            self._update_flow_map(valid_detections, speeds)
        
        return updated_flows
    
    def _calculate_region_flow(self, region_id: str) -> Optional[TrafficFlow]:
        """
        Calculate traffic flow for a region.
        
        Args:
            region_id: Region identifier
        
        Returns:
            TrafficFlow object or None if insufficient data
        """
        region_data = self.region_data[region_id]
        region_polygon = self.regions[region_id]
        
        # Check if we have enough data
        if not region_data['vehicle_counts']:
            return None
        
        # Calculate average vehicle count
        avg_count = sum(region_data['vehicle_counts'].values()) / len(region_data['vehicle_counts'])
        
        # Calculate average speed
        if self.use_speed_data and region_data['vehicle_speeds']:
            avg_speed = sum(region_data['vehicle_speeds'].values()) / len(region_data['vehicle_speeds'])
        else:
