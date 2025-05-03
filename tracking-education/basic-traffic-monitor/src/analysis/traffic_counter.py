"""
Traffic counter module for counting vehicles passing through defined zones.
Provides real-time and historical traffic counting functionality.
"""

import time
import logging
import threading
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union, Set

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from ..processing.vehicle_detection import Detection, VehicleType

# Configure logger for this module
logger = logging.getLogger(__name__)


@dataclass
class CountingZone:
    """Class representing a vehicle counting zone."""
    id: str
    name: str
    polygon: List[Tuple[int, int]]
    direction: Optional[Tuple[int, int]] = None  # Direction vector (x, y) or None for any direction
    vehicle_types: Optional[List[VehicleType]] = None  # None means count all types
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'name': self.name,
            'polygon': self.polygon,
            'direction': self.direction,
            'vehicle_types': [vt.name for vt in self.vehicle_types] if self.vehicle_types else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CountingZone':
        """Create from dictionary representation."""
        vehicle_types = None
        if data.get('vehicle_types'):
            vehicle_types = [VehicleType[vt] for vt in data['vehicle_types']]
        
        return cls(
            id=data['id'],
            name=data['name'],
            polygon=data['polygon'],
            direction=data.get('direction'),
            vehicle_types=vehicle_types
        )


@dataclass
class VehicleCount:
    """Class representing a vehicle count event."""
    zone_id: str
    timestamp: float
    vehicle_id: int
    vehicle_type: VehicleType
    direction: Optional[Tuple[float, float]] = None
    speed: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'zone_id': self.zone_id,
            'timestamp': self.timestamp,
            'vehicle_id': self.vehicle_id,
            'vehicle_type': self.vehicle_type.name,
            'direction': self.direction,
            'speed': self.speed
        }


@dataclass
class TrafficStatistics:
    """Class containing traffic statistics for a zone."""
    zone_id: str
    total_count: int
    counts_by_type: Dict[VehicleType, int]
    counts_by_hour: Dict[int, int]
    average_speed: Optional[float] = None
    peak_hour: Optional[int] = None
    peak_hour_count: int = 0


class TrafficCounter:
    """
    Traffic counter for counting vehicles through defined zones.
    """
    
    def __init__(
        self,
        zones: List[CountingZone],
        min_detection_frames: int = 3,
        min_zone_frames: int = 2,
        count_timeout: float = 5.0,
        store_history: bool = True,
        max_history_items: int = 1000
    ):
        """
        Initialize traffic counter.
        
        Args:
            zones: List of counting zones
            min_detection_frames: Minimum frames a vehicle needs to be tracked before counting
            min_zone_frames: Minimum frames a vehicle needs to be in zone to be counted
            count_timeout: Timeout in seconds before a vehicle can be counted again
            store_history: Whether to store count history
            max_history_items: Maximum count history items to store
        """
        self.zones = {zone.id: zone for zone in zones}
        self.min_detection_frames = min_detection_frames
        self.min_zone_frames = min_zone_frames
        self.count_timeout = count_timeout
        self.store_history = store_history
        self.max_history_items = max_history_items
        
        # Internal tracking state
        self.zone_trackers = {
            zone_id: {
                'vehicles_in_zone': {},       # vehicle_id -> frames in zone
                'counted_vehicles': set(),    # Set of vehicle IDs counted 
                'last_count_time': {},        # vehicle_id -> last count timestamp
                'total_count': 0,             # Total vehicles counted
                'counts_by_type': {},         # vehicle_type -> count
                'counts_by_hour': {},         # hour -> count
                'count_history': []           # List of VehicleCount objects
            }
            for zone_id in self.zones
        }
        
        # Create polygons for zone detection
        self.zone_polygons = {}
        for zone_id, zone in self.zones.items():
            self.zone_polygons[zone_id] = Polygon(zone.polygon)
        
        # Statistics
        self.total_counts = 0
        self.frames_processed = 0
        
        # Create lock for thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Traffic counter initialized with {len(zones)} zones")
    
    def process_frame(self, detections: List[Detection], frame_size: Tuple[int, int]) -> List[VehicleCount]:
        """
        Process a frame of vehicle detections.
        
        Args:
            detections: List of vehicle detections with tracking IDs
            frame_size: Size of the frame (width, height)
        
        Returns:
            List of new vehicle counts in this frame
        """
        new_counts = []
        
        with self.lock:
            self.frames_processed += 1
            
            current_time = time.time()
            current_hour = time.localtime(current_time).tm_hour
            
            # Filter detections with tracking IDs and sufficient tracking history
            valid_detections = [d for d in detections if d.id is not None]
            
            # Get set of all vehicle IDs in this frame
            current_vehicles = {d.id for d in valid_detections}
            
            # Process each zone
            for zone_id, zone_data in self.zone_trackers.items():
                zone = self.zones[zone_id]
                polygon = self.zone_polygons[zone_id]
                
                # Check which vehicles are in the zone
                vehicles_in_zone = set()
                
                for detection in valid_detections:
                    # Skip if zone has vehicle type filter and this type doesn't match
                    if zone.vehicle_types and detection.vehicle_type not in zone.vehicle_types:
                        continue
                    
                    # Get vehicle center point
                    center_x, center_y = detection.center()
                    
                    # Check if point is inside polygon
                    if polygon.contains(Point(center_x, center_y)):
                        vehicles_in_zone.add(detection.id)
                        
                        # Update frame count for this vehicle in this zone
                        if detection.id in zone_data['vehicles_in_zone']:
                            zone_data['vehicles_in_zone'][detection.id] += 1
                        else:
                            zone_data['vehicles_in_zone'][detection.id] = 1
                        
                        # Check if vehicle should be counted
                        frames_in_zone = zone_data['vehicles_in_zone'][detection.id]
                        
                        if (frames_in_zone >= self.min_zone_frames and 
                            detection.id not in zone_data['counted_vehicles']):
                            
                            # Check timeout since last count
                            last_count_time = zone_data['last_count_time'].get(detection.id, 0)
                            if current_time - last_count_time >= self.count_timeout:
                                # Count this vehicle
                                zone_data['counted_vehicles'].add(detection.id)
                                zone_data['last_count_time'][detection.id] = current_time
                                zone_data['total_count'] += 1
                                self.total_counts += 1
                                
                                # Update counts by type
                                vtype = detection.vehicle_type
                                zone_data['counts_by_type'][vtype] = zone_data['counts_by_type'].get(vtype, 0) + 1
                                
                                # Update counts by hour
                                zone_data['counts_by_hour'][current_hour] = zone_data['counts_by_hour'].get(current_hour, 0) + 1
                                
                                # Create count event
                                count = VehicleCount(
                                    zone_id=zone_id,
                                    timestamp=current_time,
                                    vehicle_id=detection.id,
                                    vehicle_type=detection.vehicle_type
                                )
                                
                                # Store in history if enabled
                                if self.store_history:
                                    zone_data['count_history'].append(count)
                                    
                                    # Trim history if needed
                                    if len(zone_data['count_history']) > self.max_history_items:
                                        zone_data['count_history'] = zone_data['count_history'][-self.max_history_items:]
                                
                                # Add to new counts for this frame
                                new_counts.append(count)
                                
                                logger.debug(f"Vehicle {detection.id} ({detection.vehicle_type.name}) counted in zone {zone_id}")
                
                # Clean up vehicles that are no longer in the zone
                vehicles_to_remove = set(zone_data['vehicles_in_zone'].keys()) - vehicles_in_zone
                for vehicle_id in vehicles_to_remove:
                    del zone_data['vehicles_in_zone'][vehicle_id]
                    
                    # Also remove from counted vehicles if they've left the zone
                    if vehicle_id in zone_data['counted_vehicles']:
                        zone_data['counted_vehicles'].remove(vehicle_id)
        
        return new_counts
    
    def get_zone_statistics(self, zone_id: str, time_period: Optional[Tuple[float, float]] = None) -> TrafficStatistics:
        """
        Get statistics for a specific zone.
        
        Args:
            zone_id: Zone identifier
            time_period: Optional time range as (start_time, end_time) timestamps
        
        Returns:
            TrafficStatistics object
        """
        if zone_id not in self.zones:
            raise ValueError(f"Invalid zone ID: {zone_id}")
            
        with self.lock:
            zone_data = self.zone_trackers[zone_id]
            
            # If time period specified and history enabled, filter counts
            if time_period and self.store_history:
                start_time, end_time = time_period
                filtered_counts = [
                    count for count in zone_data['count_history']
                    if start_time <= count.timestamp <= end_time
                ]
                
                # Calculate statistics for filtered counts
                total_count = len(filtered_counts)
                counts_by_type = {}
                counts_by_hour = {}
                
                for count in filtered_counts:
                    # Update counts by type
                    vtype = count.vehicle_type
                    counts_by_type[vtype] = counts_by_type.get(vtype, 0) + 1
                    
                    # Update counts by hour
                    hour = time.localtime(count.timestamp).tm_hour
                    counts_by_hour[hour] = counts_by_hour.get(hour, 0) + 1
                
                # Find peak hour
                peak_hour = None
                peak_hour_count = 0
                
                for hour, count in counts_by_hour.items():
                    if count > peak_hour_count:
                        peak_hour = hour
                        peak_hour_count = count
                
                stats = TrafficStatistics(
                    zone_id=zone_id,
                    total_count=total_count,
                    counts_by_type=counts_by_type,
                    counts_by_hour=counts_by_hour,
                    peak_hour=peak_hour,
                    peak_hour_count=peak_hour_count
                )
                
            else:
                # Use all-time statistics
                counts_by_type = zone_data['counts_by_type'].copy()
                counts_by_hour = zone_data['counts_by_hour'].copy()
                
                # Find peak hour
                peak_hour = None
                peak_hour_count = 0
                
                for hour, count in counts_by_hour.items():
                    if count > peak_hour_count:
                        peak_hour = hour
                        peak_hour_count = count
                
                stats = TrafficStatistics(
                    zone_id=zone_id,
                    total_count=zone_data['total_count'],
                    counts_by_type=counts_by_type,
                    counts_by_hour=counts_by_hour,
                    peak_hour=peak_hour,
                    peak_hour_count=peak_hour_count
                )
            
            return stats
    
    def get_all_statistics(self, time_period: Optional[Tuple[float, float]] = None) -> Dict[str, TrafficStatistics]:
        """
        Get statistics for all zones.
        
        Args:
            time_period: Optional time range as (start_time, end_time) timestamps
        
        Returns:
            Dictionary of zone_id -> TrafficStatistics
        """
        return {
            zone_id: self.get_zone_statistics(zone_id, time_period)
            for zone_id in self.zones
        }
    
    def get_count_history(self, zone_id: Optional[str] = None, 
                         max_items: Optional[int] = None, 
                         time_period: Optional[Tuple[float, float]] = None) -> List[VehicleCount]:
        """
        Get count history for a zone or all zones.
        
        Args:
            zone_id: Zone identifier or None for all zones
            max_items: Maximum items to return
            time_period: Optional time range as (start_time, end_time) timestamps
        
        Returns:
            List of VehicleCount objects
        """
        if not self.store_history:
            logger.warning("History storage is disabled")
            return []
        
        with self.lock:
            # Collect counts from specified zones
            all_counts = []
            
            if zone_id:
                if zone_id not in self.zones:
                    raise ValueError(f"Invalid zone ID: {zone_id}")
                zone_data = self.zone_trackers[zone_id]
                all_counts.extend(zone_data['count_history'])
            else:
                for zone_data in self.zone_trackers.values():
                    all_counts.extend(zone_data['count_history'])
            
            # Sort by timestamp (newest first)
            all_counts.sort(key=lambda c: c.timestamp, reverse=True)
            
            # Filter by time period if specified
            if time_period:
                start_time, end_time = time_period
                all_counts = [c for c in all_counts if start_time <= c.timestamp <= end_time]
            
            # Limit number of items if specified
            if max_items:
                all_counts = all_counts[:max_items]
            
            return all_counts
    
    def reset_zone(self, zone_id: str):
        """
        Reset statistics for a specific zone.
        
        Args:
            zone_id: Zone identifier
        """
        if zone_id not in self.zones:
            raise ValueError(f"Invalid zone ID: {zone_id}")
            
        with self.lock:
            self.zone_trackers[zone_id] = {
                'vehicles_in_zone': {},
                'counted_vehicles': set(),
                'last_count_time': {},
                'total_count': 0,
                'counts_by_type': {},
                'counts_by_hour': {},
                'count_history': [] if self.store_history else None
            }
            
            logger.info(f"Reset statistics for zone {zone_id}")
    
    def reset_all(self):
        """Reset all statistics."""
        with self.lock:
            for zone_id in self.zones:
                self.reset_zone(zone_id)
            
            self.total_counts = 0
            self.frames_processed = 0
            
            logger.info("Reset all statistics")
    
    def update_zones(self, zones: List[CountingZone]):
        """
        Update counting zones.
        
        Args:
            zones: New list of counting zones
        """
        with self.lock:
            # Create dictionary of new zones
            new_zones = {zone.id: zone for zone in zones}
            
            # Find zones to add, update, or remove
            current_zone_ids = set(self.zones.keys())
            new_zone_ids = set(new_zones.keys())
            
            zones_to_add = new_zone_ids - current_zone_ids
            zones_to_update = current_zone_ids.intersection(new_zone_ids)
            zones_to_remove = current_zone_ids - new_zone_ids
            
            # Update zones
            for zone_id in zones_to_update:
                self.zones[zone_id] = new_zones[zone_id]
                self.zone_polygons[zone_id] = Polygon(new_zones[zone_id].polygon)
            
            # Add new zones
            for zone_id in zones_to_add:
                self.zones[zone_id] = new_zones[zone_id]
                self.zone_polygons[zone_id] = Polygon(new_zones[zone_id].polygon)
                
                self.zone_trackers[zone_id] = {
                    'vehicles_in_zone': {},
                    'counted_vehicles': set(),
                    'last_count_time': {},
                    'total_count': 0,
                    'counts_by_type': {},
                    'counts_by_hour': {},
                    'count_history': [] if self.store_history else None
                }
            
            # Remove old zones
            for zone_id in zones_to_remove:
                del self.zones[zone_id]
                del self.zone_polygons[zone_id]
                del self.zone_trackers[zone_id]
            
            logger.info(f"Updated zones: {len(zones_to_add)} added, {len(zones_to_update)} updated, {len(zones_to_remove)} removed")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get counter statistics.
        
        Returns:
            Dictionary of counter statistics
        """
        with self.lock:
            return {
                'total_zones': len(self.zones),
                'total_counts': self.total_counts,
                'frames_processed': self.frames_processed,
                'counts_per_zone': {
                    zone_id: self.zone_trackers[zone_id]['total_count']
                    for zone_id in self.zones
                },
                'history_enabled': self.store_history,
                'history_capacity': self.max_history_items
            }
    
    def draw_zones(self, frame: np.ndarray, color_mapping: Optional[Dict[str, Tuple[int, int, int]]] = None) -> np.ndarray:
        """
        Draw counting zones on a frame.
        
        Args:
            frame: Input video frame
            color_mapping: Optional mapping of zone_id to RGB color
        
        Returns:
            Frame with drawn zones
        """
        # Make a copy of the frame
        output = frame.copy()
        
        # Use default color mapping if none provided
        if color_mapping is None:
            color_mapping = {}
            
            # Generate colors for zones without specified colors
            for i, zone_id in enumerate(self.zones):
                if zone_id not in color_mapping:
                    # Use HSV color space to generate distinct colors
                    hue = i * 137.5 % 360  # Use golden angle to generate well-distributed colors
                    color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue / 360, 0.7, 0.9))
                    color_mapping[zone_id] = (color[2], color[1], color[0])  # BGR format
        
        with self.lock:
            # Draw each zone
            for zone_id, zone in self.zones.items():
                # Get color for this zone
                color = color_mapping.get(zone_id, (0, 255, 0))  # Default to green
                
                # Convert polygon points to numpy array
                points = np.array(zone.polygon, dtype=np.int32)
                
                # Draw zone polygon
                cv2.polylines(output, [points], True, color, 2)
                
                # Draw zone name
                # Find a good position for the text (top-left point of the polygon)
                text_pos = tuple(points[0])
                
                # Get zone statistics
                total_count = self.zone_trackers[zone_id]['total_count']
                
                # Draw zone name and count
                cv2.putText(output, f"{zone.name}: {total_count}", text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return output


# Import here to avoid circular import
import colorsys
