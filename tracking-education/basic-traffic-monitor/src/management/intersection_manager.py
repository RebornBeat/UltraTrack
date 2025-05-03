"""
Intersection management module for coordinating multiple traffic intersections.
Implements network-wide coordination, green wave management, and emergency prioritization.
"""

import time
import logging
import threading
import json
import math
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
import datetime
import numpy as np
from collections import defaultdict
import heapq

from .traffic_light_controller import TrafficLightController, SignalState, SignalTiming, SignalPlan

# Configure logger for this module
logger = logging.getLogger(__name__)


class LaneDirection(Enum):
    """Direction of a lane at an intersection."""
    NORTHBOUND = 'northbound'
    SOUTHBOUND = 'southbound'
    EASTBOUND = 'eastbound'
    WESTBOUND = 'westbound'
    NORTHBOUND_LEFT = 'northbound_left'
    SOUTHBOUND_LEFT = 'southbound_left'
    EASTBOUND_LEFT = 'eastbound_left'
    WESTBOUND_LEFT = 'westbound_left'
    NORTHBOUND_RIGHT = 'northbound_right'
    SOUTHBOUND_RIGHT = 'southbound_right'
    EASTBOUND_RIGHT = 'eastbound_right'
    WESTBOUND_RIGHT = 'westbound_right'


@dataclass
class Lane:
    """
    Represents a lane at an intersection.
    """
    id: str
    direction: LaneDirection
    length: float  # Length in meters
    width: float  # Width in meters
    speed_limit: float  # Speed limit in km/h
    is_turning_lane: bool = False
    is_through_lane: bool = True
    is_shared_lane: bool = False
    vehicle_count: int = 0
    queue_length: float = 0.0  # Length of queue in meters
    lane_group_id: Optional[str] = None
    downstream_lanes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'direction': self.direction.value,
            'length': self.length,
            'width': self.width,
            'speed_limit': self.speed_limit,
            'is_turning_lane': self.is_turning_lane,
            'is_through_lane': self.is_through_lane,
            'is_shared_lane': self.is_shared_lane,
            'vehicle_count': self.vehicle_count,
            'queue_length': self.queue_length,
            'lane_group_id': self.lane_group_id,
            'downstream_lanes': self.downstream_lanes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Lane':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            direction=LaneDirection(data['direction']),
            length=data['length'],
            width=data['width'],
            speed_limit=data['speed_limit'],
            is_turning_lane=data.get('is_turning_lane', False),
            is_through_lane=data.get('is_through_lane', True),
            is_shared_lane=data.get('is_shared_lane', False),
            vehicle_count=data.get('vehicle_count', 0),
            queue_length=data.get('queue_length', 0.0),
            lane_group_id=data.get('lane_group_id'),
            downstream_lanes=data.get('downstream_lanes', [])
        )


@dataclass
class Intersection:
    """
    Represents a physical intersection with associated traffic control.
    """
    id: str
    name: str
    location: Tuple[float, float]  # (latitude, longitude)
    controller: Optional[TrafficLightController] = None
    lanes: Dict[str, Lane] = field(default_factory=dict)
    upstream_intersections: Dict[str, float] = field(default_factory=dict)  # id -> distance (m)
    downstream_intersections: Dict[str, float] = field(default_factory=dict)  # id -> distance (m)
    is_critical: bool = False
    coordination_speed: float = 40.0  # km/h
    congestion_level: float = 0.0  # 0-1 scale
    total_vehicles: int = 0
    emergency_preemption_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'location': self.location,
            'lanes': {lane_id: lane.to_dict() for lane_id, lane in self.lanes.items()},
            'upstream_intersections': self.upstream_intersections,
            'downstream_intersections': self.downstream_intersections,
            'is_critical': self.is_critical,
            'coordination_speed': self.coordination_speed,
            'congestion_level': self.congestion_level,
            'total_vehicles': self.total_vehicles,
            'emergency_preemption_active': self.emergency_preemption_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], controller: Optional[TrafficLightController] = None) -> 'Intersection':
        """Create from dictionary."""
        lanes = {}
        if 'lanes' in data:
            for lane_id, lane_data in data['lanes'].items():
                lanes[lane_id] = Lane.from_dict(lane_data)
        
        return cls(
            id=data['id'],
            name=data['name'],
            location=data['location'],
            controller=controller,
            lanes=lanes,
            upstream_intersections=data.get('upstream_intersections', {}),
            downstream_intersections=data.get('downstream_intersections', {}),
            is_critical=data.get('is_critical', False),
            coordination_speed=data.get('coordination_speed', 40.0),
            congestion_level=data.get('congestion_level', 0.0),
            total_vehicles=data.get('total_vehicles', 0),
            emergency_preemption_active=data.get('emergency_preemption_active', False)
        )
    
    def add_lane(self, lane: Lane) -> None:
        """Add a lane to the intersection."""
        self.lanes[lane.id] = lane
    
    def remove_lane(self, lane_id: str) -> bool:
        """Remove a lane from the intersection."""
        if lane_id in self.lanes:
            del self.lanes[lane_id]
            return True
        return False
    
    def update_congestion(self, flow_data: Any) -> None:
        """Update congestion level and vehicle counts from flow data."""
        if hasattr(flow_data, 'density') and flow_data.density:
            # Map density level to congestion value
            density_map = {
                'VERY_LOW': 0.0,
                'LOW': 0.2,
                'MODERATE': 0.4,
                'HIGH': 0.6,
                'VERY_HIGH': 0.8,
                'CONGESTED': 1.0
            }
            
            density_name = flow_data.density.name if hasattr(flow_data.density, 'name') else str(flow_data.density)
            self.congestion_level = density_map.get(density_name, self.congestion_level)
        
        if hasattr(flow_data, 'vehicle_count'):
            self.total_vehicles = flow_data.vehicle_count
            
            # Distribute vehicles among lanes (simplified)
            lane_count = len(self.lanes)
            if lane_count > 0:
                vehicles_per_lane = self.total_vehicles // lane_count
                remainder = self.total_vehicles % lane_count
                
                for i, lane_id in enumerate(self.lanes):
                    self.lanes[lane_id].vehicle_count = vehicles_per_lane + (1 if i < remainder else 0)
                    
                    # Estimate queue length (very simplified)
                    self.lanes[lane_id].queue_length = self.lanes[lane_id].vehicle_count * 7.0  # 7 meters per vehicle
    
    def add_upstream_intersection(self, intersection_id: str, distance: float) -> None:
        """Add an upstream intersection with distance."""
        self.upstream_intersections[intersection_id] = distance
    
    def add_downstream_intersection(self, intersection_id: str, distance: float) -> None:
        """Add a downstream intersection with distance."""
        self.downstream_intersections[intersection_id] = distance
    
    def set_emergency_preemption(self, active: bool, direction: Optional[LaneDirection] = None) -> None:
        """Set emergency preemption state."""
        self.emergency_preemption_active = active
        
        # Pass to controller if available
        if self.controller:
            direction_str = direction.value if direction else None
            self.controller.emergency_preemption(active, direction_str)
    
    def get_travel_time_to(self, other_id: str) -> float:
        """
        Get travel time to another intersection.
        
        Args:
            other_id: ID of the other intersection
        
        Returns:
            Travel time in seconds or -1 if not connected
        """
        if other_id in self.downstream_intersections:
            # Distance in meters, speed in km/h, convert to seconds
            distance = self.downstream_intersections[other_id]
            return distance / (self.coordination_speed * 1000 / 3600)
        
        if other_id in self.upstream_intersections:
            # Distance in meters, speed in km/h, convert to seconds
            distance = self.upstream_intersections[other_id]
            return distance / (self.coordination_speed * 1000 / 3600)
        
        return -1.0  # Not directly connected
    
    def set_coordination_speed(self, speed: float) -> None:
        """Set coordination speed in km/h."""
        if speed <= 0:
            raise ValueError("Coordination speed must be positive")
        
        self.coordination_speed = speed
    
    def get_lane_groups(self) -> Dict[str, List[Lane]]:
        """
        Get grouped lanes by lane_group_id.
        
        Returns:
            Dictionary of lane_group_id -> list of Lanes
        """
        groups = defaultdict(list)
        
        for lane in self.lanes.values():
            group_id = lane.lane_group_id or lane.direction.value
            groups[group_id].append(lane)
        
        return dict(groups)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the intersection.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            'id': self.id,
            'name': self.name,
            'congestion_level': self.congestion_level,
            'total_vehicles': self.total_vehicles,
            'upstream_count': len(self.upstream_intersections),
            'downstream_count': len(self.downstream_intersections)
        }
        
        # Add controller metrics if available
        if self.controller:
            controller_metrics = self.controller.get_performance_metrics()
            metrics.update({f"controller_{k}": v for k, v in controller_metrics.items()})
        
        return metrics


class IntersectionNetwork:
    """
    Manages a network of connected intersections.
    """
    
    def __init__(self):
        """Initialize intersection network."""
        self.intersections = {}  # id -> Intersection
        self.critical_intersections = set()  # Set of critical intersection IDs
        self.coordination_groups = {}  # group_id -> set of intersection IDs
        self.network_cycle_length = 0.0  # Common cycle length
        
        # Cache for path finding
        self.path_cache = {}  # (source, dest) -> path
        
        # Network structure
        self.adjacency_list = defaultdict(dict)  # id -> {neighbor_id -> distance}
        
        logger.info("Intersection network initialized")
    
    def add_intersection(self, intersection: Intersection) -> None:
        """
        Add an intersection to the network.
        
        Args:
            intersection: Intersection to add
        """
        if intersection.id in self.intersections:
            logger.warning(f"Replacing existing intersection: {intersection.id}")
        
        self.intersections[intersection.id] = intersection
        
        # Update adjacency list
        for upstream_id, distance in intersection.upstream_intersections.items():
            self.adjacency_list[intersection.id][upstream_id] = distance
            self.adjacency_list[upstream_id][intersection.id] = distance
        
        for downstream_id, distance in intersection.downstream_intersections.items():
            self.adjacency_list[intersection.id][downstream_id] = distance
            self.adjacency_list[downstream_id][intersection.id] = distance
        
        # Mark as critical if needed
        if intersection.is_critical:
            self.critical_intersections.add(intersection.id)
        
        logger.info(f"Added intersection {intersection.id} to network")
    
    def remove_intersection(self, intersection_id: str) -> bool:
        """
        Remove an intersection from the network.
        
        Args:
            intersection_id: ID of intersection to remove
        
        Returns:
            True if intersection was removed, False if not found
        """
        if intersection_id not in self.intersections:
            return False
        
        # Remove from adjacency list
        for neighbor in list(self.adjacency_list[intersection_id].keys()):
            del self.adjacency_list[neighbor][intersection_id]
        
        del self.adjacency_list[intersection_id]
        
        # Remove from critical intersections
        if intersection_id in self.critical_intersections:
            self.critical_intersections.remove(intersection_id)
        
        # Remove from coordination groups
        for group_id, intersections in list(self.coordination_groups.items()):
            if intersection_id in intersections:
                intersections.remove(intersection_id)
                if not intersections:
                    del self.coordination_groups[group_id]
        
        # Remove the intersection
        del self.intersections[intersection_id]
        
        # Clear path cache
        self.path_cache = {}
        
        logger.info(f"Removed intersection {intersection_id} from network")
        return True
    
    def add_connection(self, from_id: str, to_id: str, distance: float) -> bool:
        """
        Add a connection between intersections.
        
        Args:
            from_id: Source intersection ID
            to_id: Destination intersection ID
            distance: Distance in meters
        
        Returns:
            True if connection was added, False if any intersection was not found
        """
        if from_id not in self.intersections or to_id not in self.intersections:
            return False
        
        # Update adjacency list
        self.adjacency_list[from_id][to_id] = distance
        self.adjacency_list[to_id][from_id] = distance
        
        # Update intersection connections
        self.intersections[from_id].add_downstream_intersection(to_id, distance)
        self.intersections[to_id].add_upstream_intersection(from_id, distance)
        
        # Clear path cache
        self.path_cache = {}
        
        logger.info(f"Added connection: {from_id} -> {to_id} ({distance}m)")
        return True
    
    def remove_connection(self, from_id: str, to_id: str) -> bool:
        """
        Remove a connection between intersections.
        
        Args:
            from_id: Source intersection ID
            to_id: Destination intersection ID
        
        Returns:
            True if connection was removed, False if not found
        """
        if from_id not in self.adjacency_list or to_id not in self.adjacency_list[from_id]:
            return False
        
        # Remove from adjacency list
        del self.adjacency_list[from_id][to_id]
        del self.adjacency_list[to_id][from_id]
        
        # Remove from intersection connections
        if to_id in self.intersections[from_id].downstream_intersections:
            del self.intersections[from_id].downstream_intersections[to_id]
        
        if from_id in self.intersections[to_id].upstream_intersections:
            del self.intersections[to_id].upstream_intersections[from_id]
        
        # Clear path cache
        self.path_cache = {}
        
        logger.info(f"Removed connection: {from_id} -> {to_id}")
        return True
    
    def create_coordination_group(self, group_id: str, intersection_ids: List[str]) -> bool:
        """
        Create a coordination group.
        
        Args:
            group_id: Group identifier
            intersection_ids: List of intersection IDs in the group
        
        Returns:
            True if group was created, False if any intersection was not found
        """
        # Verify all intersections exist
        for intersection_id in intersection_ids:
            if intersection_id not in self.intersections:
                return False
        
        # Create group
        self.coordination_groups[group_id] = set(intersection_ids)
        
        logger.info(f"Created coordination group {group_id} with {len(intersection_ids)} intersections")
        return True
    
    def add_to_coordination_group(self, group_id: str, intersection_id: str) -> bool:
        """
        Add an intersection to a coordination group.
        
        Args:
            group_id: Group identifier
            intersection_id: Intersection ID to add
        
        Returns:
            True if added, False if group doesn't exist or intersection not found
        """
        if group_id not in self.coordination_groups or intersection_id not in self.intersections:
            return False
        
        self.coordination_groups[group_id].add(intersection_id)
        logger.info(f"Added intersection {intersection_id} to coordination group {group_id}")
        return True
    
    def calculate_network_cycle_length(self) -> float:
        """
        Calculate common cycle length for network coordination.
        
        Returns:
            Calculated cycle length
        """
        # Strategy: Find the largest required cycle length among critical intersections
        max_cycle = 0.0
        
        for intersection_id in self.critical_intersections:
            if intersection_id in self.intersections:
                intersection = self.intersections[intersection_id]
                if intersection.controller and intersection.controller.current_result:
                    cycle = intersection.controller.current_result.cycle_length
                    max_cycle = max(max_cycle, cycle)
        
        # If no critical intersections, use average of all intersections
        if max_cycle == 0:
            cycles = []
            for intersection in self.intersections.values():
                if intersection.controller and intersection.controller.current_result:
                    cycles.append(intersection.controller.current_result.cycle_length)
            
            if cycles:
                max_cycle = sum(cycles) / len(cycles)
        
        # Use default if still no value
        if max_cycle < 30:
            max_cycle = 120.0
        
        # Round to nearest 5 seconds
        cycle_length = round(max_cycle / 5) * 5
        
        # Store and return
        self.network_cycle_length = cycle_length
        logger.info(f"Calculated network cycle length: {cycle_length}s")
        
        return cycle_length
    
    def optimize_coordination(self, group_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize signal coordination for a group or the entire network.
        
        Args:
            group_id: Optional group identifier, if None, optimize all groups
        
        Returns:
            Dictionary with optimization results
        """
        # Calculate common cycle length
        cycle_length = self.calculate_network_cycle_length()
        
        # Groups to optimize
        groups_to_optimize = []
        
        if group_id:
            if group_id in self.coordination_groups:
                groups_to_optimize.append((group_id, self.coordination_groups[group_id]))
        else:
            groups_to_optimize = list(self.coordination_groups.items())
        
        # Initialize results
        results = {
            'cycle_length': cycle_length,
            'groups': {}
        }
        
        # Optimize each group
        for gid, intersections in groups_to_optimize:
            group_results = self._optimize_group(gid, intersections, cycle_length)
            results['groups'][gid] = group_results
        
        logger.info(f"Optimized coordination for {len(results['groups'])} groups")
        return results
    
    def _optimize_group(self, group_id: str, intersection_ids: Set[str], cycle_length: float) -> Dict[str, Any]:
        """
        Optimize coordination for a group of intersections.
        
        Args:
            group_id: Group identifier
            intersection_ids: Set of intersection IDs in the group
            cycle_length: Common cycle length
        
        Returns:
            Dictionary with group optimization results
        """
        # Find critical path within the group
        critical_path = self._find_critical_path(intersection_ids)
        
        # Calculate offsets along the critical path
        path_offsets = self._calculate_path_offsets(critical_path, cycle_length)
        
        # Apply offsets to controllers
        for intersection_id, offset in path_offsets.items():
            if intersection_id in self.intersections:
                intersection = self.intersections[intersection_id]
                if intersection.controller:
                    # Set active plan with network cycle and calculated offset
                    current_active_plan = intersection.controller.active_plan_id
                    
                    # Create coordination data
                    coordination_data = {
                        'enabled': True,
                        'cycle_length': cycle_length,
                        'offset': offset,
                        'group_id': group_id
                    }
                    
                    # Update controller with coordination data
                    if intersection.controller.running:
                        # If running, update with current state
                        if current_active_plan in intersection.controller.plans:
                            current_plan = intersection.controller.plans[current_active_plan]
                            
                            # Update plan parameters
                            current_plan.cycle_length = cycle_length
                            current_plan.offset = offset
                            current_plan.coordination_mode = True
        
        # Calculate bandwidth
        bandwidth = self._calculate_bandwidth(critical_path, path_offsets, cycle_length)
        
        # Create result
        result = {
            'critical_path': critical_path,
            'offsets': path_offsets,
            'bandwidth': bandwidth,
            'cycle_length': cycle_length
        }
        
        return result
    
    def _find_critical_path(self, intersection_ids: Set[str]) -> List[str]:
        """
        Find the critical path through a group of intersections.
        
        Args:
            intersection_ids: Set of intersection IDs
        
        Returns:
            List of intersection IDs forming the critical path
        """
        # For simplicity, find the longest path through the group
        # In a real system, this would consider traffic volumes
        
        # Build subgraph for the group
        subgraph = defaultdict(dict)
        
        for i_id in intersection_ids:
            for j_id in self.adjacency_list[i_id]:
                if j_id in intersection_ids:
                    subgraph[i_id][j_id] = self.adjacency_list[i_id][j_id]
        
        # Find longest path
        longest_path = []
        longest_length = 0
        
        for start in intersection_ids:
            for end in intersection_ids:
                if start != end:
                    path = self._find_shortest_path(start, end, subgraph)
                    
                    if path:
                        # Calculate total path length
                        path_length = 0
                        for i in range(len(path) - 1):
                            path_length += subgraph[path[i]][path[i+1]]
                        
                        if path_length > longest_length:
                            longest_length = path_length
                            longest_path = path
        
        # If no path found, use any intersection
        if not longest_path and intersection_ids:
            longest_path = [next(iter(intersection_ids))]
        
        return longest_path
    
    def _find_shortest_path(self, start: str, end: str, graph: Dict[str, Dict[str, float]]) -> List[str]:
        """
        Find shortest path between two intersections using Dijkstra's algorithm.
        
        Args:
            start: Starting intersection ID
            end: Ending intersection ID
            graph: Graph as adjacency list
        
        Returns:
            List of intersection IDs forming the path
        """
        # Check cache
        cache_key = (start, end)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Initialize
        distances = {node: float('infinity') for node in graph}
        distances[start] = 0
        previous = {node: None for node in graph}
        unvisited = list(graph.keys())
        
        while unvisited:
            # Find node with minimum distance
            current = min(unvisited, key=lambda node: distances[node])
            
            # Stop if we've reached the end or if current distance is infinity
            if current == end or distances[current] == float('infinity'):
                break
            
            # Remove current from unvisited
            unvisited.remove(current)
            
            # Check neighbors
            for neighbor, distance in graph[current].items():
                if neighbor in unvisited:
                    new_distance = distances[current] + distance
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current
        
        # Build path
        path = []
        current = end
        
        while current:
            path.append(current)
            current = previous[current]
        
        # Reverse path
        path.reverse()
        
        # Check if path is valid
        if path[0] != start:
            path = []
        
        # Cache and return
        self.path_cache[cache_key] = path
        return path
    
    def _calculate_path_offsets(self, path: List[str], cycle_length: float) -> Dict[str, float]:
        """
        Calculate offsets for intersections along a path.
        
        Args:
            path: List of intersection IDs
            cycle_length: Cycle length in seconds
        
        Returns:
            Dictionary of intersection ID -> offset
        """
        offsets = {}
        
        # For the first intersection, offset is 0
        if not path:
            return offsets
        
        offsets[path[0]] = 0.0
        
        # Calculate offsets along the path
        for i in range(len(path) - 1):
            from_id = path[i]
            to_id = path[i+1]
            
            # Get intersections
            from_intersection = self.intersections.get(from_id)
            to_intersection = self.intersections.get(to_id)
            
            if not from_intersection or not to_intersection:
                continue
            
            # Calculate travel time
            travel_time = from_intersection.get_travel_time_to(to_id)
            
            if travel_time < 0:
                continue
            
            # Calculate offset
            offsets[to_id] = (offsets[from_id] + travel_time) % cycle_length
        
        return offsets
    
    def _calculate_bandwidth(self, path: List[str], offsets: Dict[str, float], cycle_length: float) -> float:
        """
        Calculate bandwidth for a coordinated path.
        
        Args:
            path: List of intersection IDs
            offsets: Dictionary of intersection ID -> offset
            cycle_length: Cycle length in seconds
        
        Returns:
            Bandwidth as percentage of cycle length
        """
        if len(path) <= 1:
            return 1.0  # Single intersection has full bandwidth
        
        # Get green times for each intersection
        green_times = {}
        
        for i_id in path:
            if i_id in self.intersections:
                intersection = self.intersections[i_id]
                if intersection.controller and intersection.controller.current_result:
                    # Find coordinated phases
                    coordinated_phases = set()
                    for phase_id, phase in intersection.controller.plans.items():
                        if hasattr(phase, 'is_coordinated') and phase.is_coordinated:
                            coordinated_phases.add(phase_id)
                    
                    # Calculate total green time for coordinated phases
                    green = 0.0
                    for phase_id, duration in intersection.controller.current_result.phase_durations.items():
                        if phase_id in coordinated_phases:
                            # Assume effective green is 80% of phase duration
                            green += duration * 0.8
                    
                    green_times[i_id] = green
                else:
                    # Default assumption
                    green_times[i_id] = cycle_length * 0.4
            else:
                green_times[i_id] = cycle_length * 0.4
        
        # Calculate bandwidth (simplified method)
        min_green = min(green_times.values())
        bandwidth = min_green / cycle_length
        
        return bandwidth
    
    def create_green_wave(self, path: List[str], speed: float) -> Dict[str, Any]:
        """
        Create a green wave along a path of intersections.
        
        Args:
            path: List of intersection IDs
            speed: Desired progression speed in km/h
        
        Returns:
            Dictionary with green wave configuration
        """
        # Calculate common cycle length
        cycle_length = self.calculate_network_cycle_length()
        
        # Create results structure
        results = {
            'path': path,
            'speed': speed,
            'cycle_length': cycle_length,
            'offsets': {},
            'success': True
        }
        
        # Calculate offsets for green wave
        current_offset = 0.0
        
        for i in range(len(path) - 1):
            from_id = path[i]
            to_id = path[i+1]
            
            # Store current offset
            results['offsets'][from_id] = current_offset
            
            # Get distance between intersections
            distance = 0.0
            
            if from_id in self.adjacency_list and to_id in self.adjacency_list[from_id]:
                distance = self.adjacency_list[from_id][to_id]
            else:
                # Path is not contiguous
                results['success'] = False
                logger.warning(f"Green wave path is not contiguous: {from_id} -> {to_id}")
                continue
            
            # Calculate travel time at desired speed
            speed_m_s = speed * 1000 / 3600  # Convert km/h to m/s
            travel_time = distance / speed_m_s
            
            # Update offset for next intersection
            current_offset = (current_offset + travel_time) % cycle_length
        
        # Add last intersection
        if path:
            results['offsets'][path[-1]] = current_offset
        
        # Apply offsets to controllers
        for intersection_id, offset in results['offsets'].items():
            if intersection_id in self.intersections:
                intersection = self.intersections[intersection_id]
                if intersection.controller:
                    # Set coordination speed
                    intersection.set_coordination_speed(speed)
                    
                    # Set active plan with network cycle and calculated offset
                    current_active_plan = intersection.controller.active_plan_id
                    
                    # Update controller with coordination data
                    if intersection.controller.running:
                        # If running, update with current state
                        if current_active_plan in intersection.controller.plans:
                            current_plan = intersection.controller.plans[current_active_plan]
                            
                            # Update plan parameters
                            current_plan.cycle_length = cycle_length
                            current_plan.offset = offset
                            current_plan.coordination_mode = True
        
        logger.info(f"Created green wave along path with {len(path)} intersections at {speed} km/h")
        return results
    
    def handle_emergency_vehicle(self, vehicle_location: Tuple[float, float], route: List[str], 
                               clear_time_sec: float = 30.0) -> Dict[str, Any]:
        """
        Handle emergency vehicle prioritization.
        
        Args:
            vehicle_location: Current (lat, lon) location of emergency vehicle
            route: List of intersection IDs on the route
            clear_time_sec: Time in seconds to clear ahead of vehicle
        
        Returns:
            Dictionary with prioritization results
        """
        # Calculate intersections to prioritize
        prioritized = []
        
        # Calculate estimated arrival times
        arrival_times = {}
        current_time = 0.0
        
        # Find closest intersection to current location
        closest_id = None
        closest_dist = float('infinity')
        
        for intersection_id, intersection in self.intersections.items():
            # Calculate distance (simple approximation)
            lat_diff = intersection.location[0] - vehicle_location[0]
            lon_diff = intersection.location[1] - vehicle_location[1]
            distance = math.sqrt(lat_diff**2 + lon_diff**2) * 111000  # Rough conversion to meters
            
            if distance < closest_dist:
                closest_dist = distance
                closest_id = intersection_id
        
        # If we found a starting intersection and it's in the route
        if closest_id and closest_id in route:
            # Calculate time to arrival (assuming 80 km/h for emergency vehicle)
            emergency_speed = 80.0  # km/h
            emergency_speed_m_s = emergency_speed * 1000 / 3600  # m/s
            
            current_time = closest_dist / emergency_speed_m_s
            
            # Get index in route
            start_index = route.index(closest_id)
            
            # Calculate arrival times for subsequent intersections
            arrival_times[closest_id] = current_time
            
            for i in range(start_index, len(route) - 1):
                from_id = route[i]
                to_id = route[i+1]
                
                # Get distance
                distance = 0.0
                if from_id in self.adjacency_list and to_id in self.adjacency_list[from_id]:
                    distance = self.adjacency_list[from_id][to_id]
                
                # Calculate travel time
                travel_time = distance / emergency_speed_m_s
                current_time += travel_time
                arrival_times[to_id] = current_time
            
            # Prioritize intersections that need to be cleared in advance
            for intersection_id, arrival_time in arrival_times.items():
                if arrival_time <= clear_time_sec:
                    # Immediate prioritization needed
                    if intersection_id in self.intersections:
                        direction = self._get_direction_from_route(intersection_id, route)
                        
                        # Set preemption
                        self.intersections[intersection_id].set_emergency_preemption(True, direction)
                        prioritized.append(intersection_id)
        
        # Create result
        result = {
            'prioritized_intersections': prioritized,
            'arrival_times': arrival_times,
            'clear_time_sec': clear_time_sec,
            'emergency_route': route
        }
        
        logger.info(f"Prioritized {len(prioritized)} intersections for emergency vehicle")
        return result
    
    def _get_direction_from_route(self, intersection_id: str, route: List[str]) -> Optional[LaneDirection]:
        """
        Determine emergency vehicle approach direction from route.
        
        Args:
            intersection_id: Intersection ID
            route: List of intersection IDs on the route
        
        Returns:
            LaneDirection or None if unknown
        """
        try:
            # Find position in route
            index = route.index(intersection_id)
            
            # If first in route, can't determine approach
            if index == 0:
                return None
            
            # Get previous intersection
            prev_id = route[index - 1]
            
            # Get next intersection if available
            next_id = route[index + 1] if index < len(route) - 1 else None
            
            # Get intersection
            intersection = self.intersections.get(intersection_id)
            if not intersection:
                return None
            
            # Get previous intersection
            prev_intersection = self.intersections.get(prev_id)
            if not prev_intersection:
                return None
            
            # Calculate rough approach direction
            lat_diff = intersection.location[0] - prev_intersection.location[0]
            lon_diff = intersection.location[1] - prev_intersection.location[1]
            
            # Determine direction based on larger difference
            if abs(lat_diff) > abs(lon_diff):
                # North-South direction
                if lat_diff > 0:
                    direction = LaneDirection.NORTHBOUND
                else:
                    direction = LaneDirection.SOUTHBOUND
            else:
                # East-West direction
                if lon_diff > 0:
                    direction = LaneDirection.EASTBOUND
                else:
                    direction = LaneDirection.WESTBOUND
            
            # Refine with next intersection if available
            if next_id:
                next_intersection = self.intersections.get(next_id)
                if next_intersection:
                    # Calculate exit direction
                    exit_lat_diff = next_intersection.location[0] - intersection.location[0]
                    exit_lon_diff = next_intersection.location[1] - intersection.location[1]
                    
                    # Check if this is a left turn
                    if direction == LaneDirection.NORTHBOUND and exit_lon_diff < 0:
                        direction = LaneDirection.NORTHBOUND_LEFT
                    elif direction == LaneDirection.SOUTHBOUND and exit_lon_diff > 0:
                        direction = LaneDirection.SOUTHBOUND_LEFT
                    elif direction == LaneDirection.EASTBOUND and exit_lat_diff > 0:
                        direction = LaneDirection.EASTBOUND_LEFT
                    elif direction == LaneDirection.WESTBOUND and exit_lat_diff < 0:
                        direction = LaneDirection.WESTBOUND_LEFT
                    
                    # Check if this is a right turn
                    elif direction == LaneDirection.NORTHBOUND and exit_lon_diff > 0:
                        direction = LaneDirection.NORTHBOUND_RIGHT
                    elif direction == LaneDirection.SOUTHBOUND and exit_lon_diff < 0:
                        direction = LaneDirection.SOUTHBOUND_RIGHT
                    elif direction == LaneDirection.EASTBOUND and exit_lat_diff < 0:
                        direction = LaneDirection.EASTBOUND_RIGHT
                    elif direction == LaneDirection.WESTBOUND and exit_lat_diff > 0:
                        direction = LaneDirection.WESTBOUND_RIGHT
            
            return direction
            
        except (ValueError, IndexError):
            return None
    
    def clear_emergency_preemption(self, route: List[str]) -> None:
        """
        Clear emergency preemption for a route.
        
        Args:
            route: List of intersection IDs to clear
        """
        for intersection_id in route:
            if intersection_id in self.intersections:
                self.intersections[intersection_id].set_emergency_preemption(False)
        
        logger.info(f"Cleared emergency preemption for {len(route)} intersections")
    
    def update_traffic_data(self, intersection_id: str, flow_data: Any) -> None:
        """
        Update traffic data for an intersection.
        
        Args:
            intersection_id: Intersection ID
            flow_data: Traffic flow data
        """
        if intersection_id in self.intersections:
            # Update intersection congestion
            self.intersections[intersection_id].update_congestion(flow_data)
            
            # Update controller
            if self.intersections[intersection_id].controller:
                self.intersections[intersection_id].controller.update(flow_data)
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for the entire network.
        
        Returns:
            Dictionary with network statistics
        """
        stats = {
            'intersection_count': len(self.intersections),
            'critical_intersection_count': len(self.critical_intersections),
            'coordination_group_count': len(self.coordination_groups),
            'network_cycle_length': self.network_cycle_length,
            'average_congestion': 0.0,
            'total_vehicles': 0,
            'intersections': {}
        }
        
        # Calculate aggregate statistics
        if self.intersections:
            congestion_sum = sum(i.congestion_level for i in self.intersections.values())
            stats['average_congestion'] = congestion_sum / len(self.intersections)
            
            stats['total_vehicles'] = sum(i.total_vehicles for i in self.intersections.values())
        
        # Individual intersection stats
        for intersection_id, intersection in self.intersections.items():
            stats['intersections'][intersection_id] = {
                'name': intersection.name,
                'congestion_level': intersection.congestion_level,
                'total_vehicles': intersection.total_vehicles,
                'lane_count': len(intersection.lanes),
                'is_critical': intersection_id in self.critical_intersections,
                'coordinated': any(intersection_id in group for group in self.coordination_groups.values())
            }
        
        return stats
    
    def save_to_file(self, file_path: str) -> bool:
        """
        Save network configuration to file.
        
        Args:
            file_path: Path to save file
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create serializable data
            data = {
                'intersections': {
                    intersection_id: intersection.to_dict()
                    for intersection_id, intersection in self.intersections.items()
                },
                'critical_intersections': list(self.critical_intersections),
                'coordination_groups': {
                    group_id: list(intersections)
                    for group_id, intersections in self.coordination_groups.items()
                },
                'network_cycle_length': self.network_cycle_length
            }
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Network configuration saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save network configuration: {str(e)}")
            return False
    
    @classmethod
    def load_from_file(cls, file_path: str, controllers: Optional[Dict[str, TrafficLightController]] = None) -> 'IntersectionNetwork':
        """
        Load network configuration from file.
        
        Args:
            file_path: Path to load file
            controllers: Optional dictionary of intersection ID -> TrafficLightController
        
        Returns:
            IntersectionNetwork instance
        """
        try:
            # Read from file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Create network
            network = cls()
            
            # Add intersections
            for intersection_id, intersection_data in data['intersections'].items():
                controller = None
                if controllers and intersection_id in controllers:
                    controller = controllers[intersection_id]
                
                intersection = Intersection.from_dict(intersection_data, controller)
                network.add_intersection(intersection)
            
            # Set critical intersections
            network.critical_intersections = set(data['critical_intersections'])
            
            # Set coordination groups
            for group_id, intersection_ids in data['coordination_groups'].items():
                network.coordination_groups[group_id] = set(intersection_ids)
            
            # Set network cycle length
            network.network_cycle_length = data['network_cycle_length']
            
            logger.info(f"Network configuration loaded from {file_path}")
            return network
            
        except Exception as e:
            logger.error(f"Failed to load network configuration: {str(e)}")
            raise


class IntersectionManager:
    """
    Manager for handling intersection control and coordination.
    """
    
    def __init__(self, network: Optional[IntersectionNetwork] = None):
        """
        Initialize intersection manager.
        
        Args:
            network: Optional existing intersection network
        """
        self.network = network or IntersectionNetwork()
        self.controllers = {}  # intersection_id -> TrafficLightController
        self.running = False
        self.stop_event = threading.Event()
        self.update_thread = None
        self.update_interval = 5.0  # seconds
        
        # Performance metrics
        self.last_update_time = 0.0
        self.update_count = 0
        
        logger.info("Intersection manager initialized")
    
    def add_controller(self, intersection_id: str, controller: TrafficLightController) -> bool:
        """
        Add a traffic light controller for an intersection.
        
        Args:
            intersection_id: Intersection ID
            controller: TrafficLightController instance
        
        Returns:
            True if added, False if intersection not found
        """
        if intersection_id not in self.network.intersections:
            return False
        
        self.controllers[intersection_id] = controller
        self.network.intersections[intersection_id].controller = controller
        
        logger.info(f"Added controller for intersection {intersection_id}")
        return True
    
    def add_intersection(self, intersection: Intersection, controller: Optional[TrafficLightController] = None) -> None:
        """
        Add an intersection with optional controller.
        
        Args:
            intersection: Intersection to add
            controller: Optional controller for the intersection
        """
        # Add to network
        self.network.add_intersection(intersection)
        
        # Add controller if provided
        if controller:
            self.controllers[intersection.id] = controller
            intersection.controller = controller
            
            logger.info(f"Added intersection {intersection.id} with controller")
        else:
            logger.info(f"Added intersection {intersection.id} without controller")
    
    def remove_intersection(self, intersection_id: str) -> bool:
        """
        Remove an intersection and its controller.
        
        Args:
            intersection_id: Intersection ID to remove
        
        Returns:
            True if removed, False if not found
        """
        # Remove from network
        result = self.network.remove_intersection(intersection_id)
        
        # Remove controller if exists
        if intersection_id in self.controllers:
            del self.controllers[intersection_id]
        
        return result
    
    def start(self) -> bool:
        """
        Start the intersection manager and all controllers.
        
        Returns:
            True if started successfully
        """
        if self.running:
            logger.warning("Intersection manager is already running")
            return True
        
        logger.info("Starting intersection manager")
        
        # Reset stop event
        self.stop_event.clear()
        
        # Start controllers
        started_count = 0
        
        for intersection_id, controller in self.controllers.items():
            if controller.start():
                started_count += 1
            else:
                logger.warning(f"Failed to start controller for intersection {intersection_id}")
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        self.running = True
        logger.info(f"Intersection manager started with {started_count} controllers")
        
        return True
    
    def stop(self) -> None:
        """Stop the intersection manager and all controllers."""
        if not self.running:
            return
        
        logger.info("Stopping intersection manager")
        
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=3.0)
        
        # Stop controllers
        for intersection_id, controller in self.controllers.items():
            controller.stop()
        
        self.running = False
        logger.info("Intersection manager stopped")
    
    def _update_loop(self) -> None:
        """Update loop for coordination."""
        logger.info("Update loop started")
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Update at specified interval
                if current_time - self.last_update_time >= self.update_interval:
                    self._perform_coordination_update()
                    self.last_update_time = current_time
                    self.update_count += 1
                
                # Sleep to avoid busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}")
                time.sleep(1.0)
    
    def _perform_coordination_update(self) -> None:
        """Perform coordination update."""
        # Calculate network cycle length
        self.network.calculate_network_cycle_length()
        
        # Optimize coordination for each group
        self.network.optimize_coordination()
    
    def update_traffic_data(self, intersection_id: str, flow_data: Any) -> bool:
        """
        Update traffic data for an intersection.
        
        Args:
            intersection_id: Intersection ID
            flow_data: Traffic flow data
        
        Returns:
            True if updated, False if intersection not found
        """
        if intersection_id not in self.network.intersections:
            return False
        
        # Update network data
        self.network.update_traffic_data(intersection_id, flow_data)
        
        return True
    
    def create_green_wave(self, start_id: str, end_id: str, speed: float) -> Dict[str, Any]:
        """
        Create a green wave between two intersections.
        
        Args:
            start_id: Starting intersection ID
            end_id: Ending intersection ID
            speed: Desired progression speed in km/h
        
        Returns:
            Result dictionary
        """
        # Find path between intersections
        path = []
        
        # Check if intersections exist
        if start_id not in self.network.intersections or end_id not in self.network.intersections:
            return {'success': False, 'error': 'Intersection not found'}
        
        # Find shortest path
        path = self.network._find_shortest_path(start_id, end_id, self.network.adjacency_list)
        
        if not path:
            return {'success': False, 'error': 'No path found between intersections'}
        
        # Create green wave
        result = self.network.create_green_wave(path, speed)
        
        return result
    
    def handle_emergency_vehicle(self, vehicle_location: Tuple[float, float], 
                               destination: Tuple[float, float]) -> Dict[str, Any]:
        """
        Handle emergency vehicle route and prioritization.
        
        Args:
            vehicle_location: Current (lat, lon) location of emergency vehicle
            destination: Destination (lat, lon) location
        
        Returns:
            Result dictionary
        """
        # Find nearest intersections to start and end
        start_id = self._find_nearest_intersection(vehicle_location)
        end_id = self._find_nearest_intersection(destination)
        
        if not start_id or not end_id:
            return {'success': False, 'error': 'Could not find nearby intersections'}
        
        # Find route
        route = self.network._find_shortest_path(start_id, end_id, self.network.adjacency_list)
        
        if not route:
            return {'success': False, 'error': 'No route found'}
        
        # Handle emergency vehicle
        result = self.network.handle_emergency_vehicle(vehicle_location, route)
        
        # Add additional info
        result['start_intersection'] = start_id
        result['end_intersection'] = end_id
        result['success'] = True
        
        return result
    
    def _find_nearest_intersection(self, location: Tuple[float, float]) -> Optional[str]:
        """
        Find the nearest intersection to a location.
        
        Args:
            location: (lat, lon) location
        
        Returns:
            Nearest intersection ID or None if none found
        """
        nearest_id = None
        min_distance = float('infinity')
        
        for intersection_id, intersection in self.network.intersections.items():
            # Calculate distance (simple approximation)
            lat_diff = intersection.location[0] - location[0]
            lon_diff = intersection.location[1] - location[1]
            distance = math.sqrt(lat_diff**2 + lon_diff**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_id = intersection_id
        
        return nearest_id
    
    def get_intersection_status(self, intersection_id: str) -> Dict[str, Any]:
        """
        Get status of an intersection.
        
        Args:
            intersection_id: Intersection ID
        
        Returns:
            Status dictionary
        """
        if intersection_id not in self.network.intersections:
            return {'error': 'Intersection not found'}
        
        intersection = self.network.intersections[intersection_id]
        
        # Get controller state if available
        controller_state = {}
        if intersection.controller:
            controller_state = intersection.controller.get_current_state()
        
        # Build status
        status = {
            'id': intersection_id,
            'name': intersection.name,
            'location': intersection.location,
            'congestion_level': intersection.congestion_level,
            'total_vehicles': intersection.total_vehicles,
            'lanes': {lane_id: lane.to_dict() for lane_id, lane in intersection.lanes.items()},
            'upstream_intersections': intersection.upstream_intersections,
            'downstream_intersections': intersection.downstream_intersections,
            'is_critical': intersection_id in self.network.critical_intersections,
            'coordinated': any(intersection_id in group for group in self.network.coordination_groups.values()),
            'controller': controller_state,
            'emergency_preemption_active': intersection.emergency_preemption_active
        }
        
        return status
    
    def get_network_status(self) -> Dict[str, Any]:
        """
        Get status of the entire network.
        
        Returns:
            Network status dictionary
        """
        # Get network statistics
        stats = self.network.get_network_statistics()
        
        # Add manager info
        status = {
            'running': self.running,
            'controller_count': len(self.controllers),
            'update_count': self.update_count,
            'update_interval': self.update_interval,
            'last_update_time': self.last_update_time,
            'network': stats
        }
        
        return status
    
    def save_configuration(self, file_path: str) -> bool:
        """
        Save manager configuration to file.
        
        Args:
            file_path: Path to save file
        
        Returns:
            True if saved successfully
        """
        return self.network.save_to_file(file_path)
    
    @classmethod
    def load_configuration(cls, file_path: str) -> 'IntersectionManager':
        """
        Load manager configuration from file.
        
        Args:
            file_path: Path to load file
        
        Returns:
            IntersectionManager instance
        """
        # Load network
        network = IntersectionNetwork.load_from_file(file_path)
        
        # Create manager
        manager = cls(network)
        
        # Transfer controllers from network to manager
        for intersection_id, intersection in network.intersections.items():
            if intersection.controller:
                manager.controllers[intersection_id] = intersection.controller
        
        return manager
