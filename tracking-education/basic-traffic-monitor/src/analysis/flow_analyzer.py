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
            avg_speed = 0.0
        
        # Combine direction counts from all frames
        combined_direction_counts = {}
        for direction_counts in region_data['flow_directions'].values():
            for direction, count in direction_counts.items():
                combined_direction_counts[direction] = combined_direction_counts.get(direction, 0) + count
        
        # Determine dominant direction
        dominant_direction = self._get_dominant_direction(combined_direction_counts)
        
        # Calculate region area
        region_area = self._calculate_region_area(region_polygon)
        
        # Determine density level
        density_level = self._get_density_level(avg_count, region_area)
        
        # Create flow object
        flow = TrafficFlow(
            region_id=region_id,
            timestamp=time.time(),
            density=density_level,
            vehicle_count=int(avg_count),
            avg_speed=avg_speed,
            dominant_direction=dominant_direction,
            directional_counts=combined_direction_counts
        )
        
        return flow
    
    def _update_density_map(self, detections: List[Detection]):
        """
        Update density map based on current detections.
        
        Args:
            detections: List of vehicle detections
        """
        # Create new density map
        new_density = np.zeros_like(self.density_map)
        
        # Add vehicle positions to density map
        for detection in detections:
            center_x, center_y = detection.center()
            if 0 <= center_x < self.frame_size[0] and 0 <= center_y < self.frame_size[1]:
                # Add Gaussian blob at vehicle position
                new_density[center_y, center_x] = 1.0
        
        # Apply Gaussian blur to spread density
        new_density = gaussian_filter(new_density, sigma=15)
        
        # Normalize density
        if np.max(new_density) > 0:
            new_density = new_density / np.max(new_density)
        
        # Update density map with temporal smoothing
        self.density_map = self.density_map * 0.8 + new_density * 0.2
    
    def _update_flow_map(self, detections: List[Detection], speeds: Optional[Dict[int, float]] = None):
        """
        Update flow map based on current detections and speeds.
        
        Args:
            detections: List of vehicle detections
            speeds: Optional dictionary of vehicle_id -> speed (km/h)
        """
        # Create new flow map
        new_flow = np.zeros_like(self.flow_map)
        
        # Process each region
        for region_id, region_data in self.region_data.items():
            # Get recent vehicle positions
            recent_positions = sorted(region_data['vehicle_positions'], reverse=True)
            
            # Group by vehicle ID
            positions_by_vehicle = {}
            for frame, vehicle_id, x, y, direction, speed in recent_positions:
                if vehicle_id not in positions_by_vehicle:
                    positions_by_vehicle[vehicle_id] = []
                positions_by_vehicle[vehicle_id].append((frame, x, y, direction, speed))
            
            # Calculate flow vectors for each vehicle
            for vehicle_id, positions in positions_by_vehicle.items():
                if len(positions) >= 2:
                    # Get most recent positions
                    current = positions[0]
                    previous = positions[1]
                    
                    # Calculate flow vector
                    _, curr_x, curr_y, _, curr_speed = current
                    _, prev_x, prev_y, _, _ = previous
                    
                    # Skip if positions are identical
                    if curr_x == prev_x and curr_y == prev_y:
                        continue
                    
                    # Calculate direction vector
                    dx = curr_x - prev_x
                    dy = curr_y - prev_y
                    
                    # Normalize vector
                    magnitude = np.sqrt(dx*dx + dy*dy)
                    if magnitude > 0:
                        dx = dx / magnitude
                        dy = dy / magnitude
                    
                    # Apply speed scaling if available
                    if speeds and vehicle_id in speeds:
                        speed = speeds[vehicle_id]
                        dx = dx * min(50.0, speed) / 50.0
                        dy = dy * min(50.0, speed) / 50.0
                    
                    # Add flow vector to the map
                    if 0 <= curr_x < self.frame_size[0] and 0 <= curr_y < self.frame_size[1]:
                        new_flow[int(curr_y), int(curr_x), 0] = dx
                        new_flow[int(curr_y), int(curr_x), 1] = dy
        
        # Apply Gaussian blur to spread flow vectors
        new_flow[:, :, 0] = gaussian_filter(new_flow[:, :, 0], sigma=5)
        new_flow[:, :, 1] = gaussian_filter(new_flow[:, :, 1], sigma=5)
        
        # Update flow map with temporal smoothing
        self.flow_map = self.flow_map * 0.9 + new_flow * 0.1
    
    def get_current_flow(self, region_id: Optional[str] = None) -> Union[TrafficFlow, Dict[str, TrafficFlow], None]:
        """
        Get current flow for a region or all regions.
        
        Args:
            region_id: Optional region identifier
        
        Returns:
            TrafficFlow object, dictionary of region_id -> TrafficFlow, or None if no data
        """
        with self.lock:
            if region_id:
                if region_id not in self.region_data:
                    return None
                return self.region_data[region_id]['last_flow']
            else:
                # Return all regions
                flows = {}
                for rid, data in self.region_data.items():
                    if data['last_flow']:
                        flows[rid] = data['last_flow']
                return flows if flows else None
    
    def get_flow_history(self, region_id: Optional[str] = None,
                        max_items: Optional[int] = None,
                        time_period: Optional[Tuple[float, float]] = None) -> List[TrafficFlow]:
        """
        Get flow history for a region or all regions.
        
        Args:
            region_id: Optional region identifier to filter by
            max_items: Maximum items to return
            time_period: Optional time range as (start_time, end_time) timestamps
        
        Returns:
            List of TrafficFlow objects
        """
        if not self.store_history:
            logger.warning("History storage is disabled")
            return []
        
        with self.lock:
            # Filter history
            filtered = self.history.copy()
            
            # Filter by region ID if specified
            if region_id:
                filtered = [f for f in filtered if f.region_id == region_id]
            
            # Filter by time period if specified
            if time_period:
                start_time, end_time = time_period
                filtered = [f for f in filtered if start_time <= f.timestamp <= end_time]
            
            # Sort by timestamp (newest first)
            filtered.sort(key=lambda f: f.timestamp, reverse=True)
            
            # Limit number of items if specified
            if max_items:
                filtered = filtered[:max_items]
            
            return filtered
    
    def get_density_map(self) -> np.ndarray:
        """
        Get current traffic density map.
        
        Returns:
            Numpy array representing density (0-1 range)
        """
        with self.lock:
            return self.density_map.copy()
    
    def get_flow_map(self) -> np.ndarray:
        """
        Get current traffic flow map.
        
        Returns:
            Numpy array representing flow vectors (x, y components)
        """
        with self.lock:
            return self.flow_map.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get analyzer statistics.
        
        Returns:
            Dictionary of analyzer statistics
        """
        with self.lock:
            return {
                'regions': len(self.regions),
                'frames_processed': self.frames_processed,
                'flow_updates': self.flow_updates,
                'history_enabled': self.store_history,
                'history_items': len(self.history) if self.store_history else 0
            }
    
    def reset(self):
        """Reset all flow data."""
        with self.lock:
            # Reset region data
            for region_id in self.regions:
                self.region_data[region_id] = {
                    'vehicle_positions': [],
                    'vehicle_counts': {},
                    'vehicle_speeds': {},
                    'flow_directions': {},
                    'current_vehicles': set(),
                    'last_flow': None
                }
            
            # Reset maps
            self.density_map = np.zeros_like(self.density_map)
            self.flow_map = np.zeros_like(self.flow_map)
            
            # Reset history
            if self.store_history:
                self.history = []
            
            # Reset statistics
            self.frames_processed = 0
            self.flow_updates = 0
            
            logger.info("Flow analyzer reset")
    
    def update_regions(self, regions: Dict[str, List[Tuple[int, int]]]):
        """
        Update analysis regions.
        
        Args:
            regions: Dictionary of region_id -> polygon points
        """
        with self.lock:
            # Store old data for migration
            old_data = self.region_data
            
            # Update regions
            self.regions = regions
            
            # Create new region data
            self.region_data = {}
            for region_id in self.regions:
                if region_id in old_data:
                    # Reuse existing data for this region
                    self.region_data[region_id] = old_data[region_id]
                else:
                    # Create new data structure for new region
                    self.region_data[region_id] = {
                        'vehicle_positions': [],
                        'vehicle_counts': {},
                        'vehicle_speeds': {},
                        'flow_directions': {},
                        'current_vehicles': set(),
                        'last_flow': None
                    }
            
            logger.info(f"Updated regions: now using {len(self.regions)} regions")
    
    def draw_flow_visualization(self, frame: np.ndarray, 
                              draw_density: bool = True,
                              draw_flow: bool = True,
                              draw_regions: bool = True,
                              density_alpha: float = 0.5,
                              flow_scale: float = 10.0,
                              flow_threshold: float = 0.05,
                              flow_grid_step: int = 20) -> np.ndarray:
        """
        Draw flow visualization on a frame.
        
        Args:
            frame: Input video frame
            draw_density: Whether to draw density map
            draw_flow: Whether to draw flow vectors
            draw_regions: Whether to draw region boundaries
            density_alpha: Alpha value for density overlay (0-1)
            flow_scale: Scaling factor for flow vectors
            flow_threshold: Minimum flow magnitude to draw
            flow_grid_step: Grid step size for flow vectors
        
        Returns:
            Frame with visualization overlay
        """
        # Make a copy of the frame
        output = frame.copy()
        
        with self.lock:
            # Draw density map if enabled
            if draw_density:
                # Create colored density map
                colored_density = cv2.applyColorMap(
                    (self.density_map * 255).astype(np.uint8), 
                    cv2.COLORMAP_JET
                )
                
                # Create mask for areas with significant density
                mask = (self.density_map > 0.05).astype(np.uint8) * 255
                
                # Apply mask to colored density
                masked_density = cv2.bitwise_and(colored_density, colored_density, mask=mask)
                
                # Blend with original frame
                cv2.addWeighted(output, 1.0, masked_density, density_alpha, 0, output)
            
            # Draw flow vectors if enabled
            if draw_flow:
                # Get flow map
                flow = self.flow_map
                
                # Draw flow vectors on a grid
                h, w = flow.shape[:2]
                y, x = np.mgrid[0:h:flow_grid_step, 0:w:flow_grid_step].reshape(2, -1).astype(int)
                
                # Filter points outside the frame
                mask = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                x, y = x[mask], y[mask]
                
                # Get flow vectors at grid points
                fx, fy = flow[y, x, 0], flow[y, x, 1]
                
                # Calculate vector magnitudes
                magnitude = np.sqrt(fx**2 + fy**2)
                
                # Filter by threshold
                mask = magnitude > flow_threshold
                x, y, fx, fy = x[mask], y[mask], fx[mask], fy[mask]
                
                # Draw arrows
                for i in range(len(x)):
                    cv2.arrowedLine(
                        output,
                        (int(x[i]), int(y[i])),
                        (int(x[i] + fx[i] * flow_scale), int(y[i] + fy[i] * flow_scale)),
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                        tipLength=0.3
                    )
            
            # Draw region boundaries if enabled
            if draw_regions:
                for region_id, polygon in self.regions.items():
                    # Get flow data for this region
                    flow_data = self.region_data[region_id]['last_flow']
                    
                    # Convert polygon to numpy array
                    points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                    
                    # Determine color based on density level
                    if flow_data and hasattr(flow_data, 'density'):
                        if flow_data.density == TrafficDensity.VERY_LOW:
                            color = (0, 255, 0)  # Green
                        elif flow_data.density == TrafficDensity.LOW:
                            color = (0, 255, 255)  # Yellow
                        elif flow_data.density == TrafficDensity.MODERATE:
                            color = (0, 165, 255)  # Orange
                        elif flow_data.density == TrafficDensity.HIGH:
                            color = (0, 0, 255)  # Red
                        elif flow_data.density == TrafficDensity.VERY_HIGH:
                            color = (128, 0, 255)  # Purple
                        elif flow_data.density == TrafficDensity.CONGESTED:
                            color = (0, 0, 128)  # Dark red
                        else:
                            color = (255, 255, 255)  # White
                    else:
                        color = (255, 255, 255)  # White
                    
                    # Draw polygon
                    cv2.polylines(output, [points], True, color, 2)
                    
                    # Draw region label
                    center_x = sum(p[0] for p in polygon) // len(polygon)
                    center_y = sum(p[1] for p in polygon) // len(polygon)
                    
                    # Prepare label text
                    label = region_id
                    if flow_data:
                        label += f": {flow_data.vehicle_count} vehicles"
                    
                    # Draw label
                    cv2.putText(output, label, (center_x, center_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return output
