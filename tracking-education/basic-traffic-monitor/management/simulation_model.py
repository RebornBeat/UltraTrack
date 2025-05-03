"""
Traffic light simulation model for virtual traffic generation and testing.
Implements vehicle generation, movement, and response to traffic signals.
"""

import time
import logging
import random
import threading
import math
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime, timedelta

# Configure logger for this module
logger = logging.getLogger(__name__)


class VehicleType(Enum):
    """Vehicle type enumeration."""
    CAR = 'car'
    TRUCK = 'truck'
    BUS = 'bus'
    MOTORCYCLE = 'motorcycle'
    BICYCLE = 'bicycle'
    EMERGENCY = 'emergency'


class MovementType(Enum):
    """Traffic movement type enumeration."""
    THROUGH = 'through'
    LEFT = 'left'
    RIGHT = 'right'
    U_TURN = 'u_turn'


class VehicleStatus(Enum):
    """Vehicle status enumeration."""
    APPROACHING = 'approaching'
    QUEUED = 'queued'
    MOVING = 'moving'
    CLEARED = 'cleared'
    BLOCKED = 'blocked'


@dataclass
class SimulatedVehicle:
    """Class representing a simulated vehicle."""
    id: int
    vehicle_type: VehicleType
    origin_lane: str
    destination_lane: str
    movement_type: MovementType
    position: float  # Distance from stop line in meters (negative = before stop line)
    speed: float  # Current speed in m/s
    max_speed: float  # Maximum speed in m/s
    acceleration: float  # Current acceleration in m/s²
    length: float  # Vehicle length in meters
    creation_time: float  # Time vehicle was created
    status: VehicleStatus = VehicleStatus.APPROACHING
    waiting_time: float = 0.0  # Total time spent waiting (speed < 0.1 m/s)
    is_emergency: bool = False
    signal_compliance: float = 1.0  # Compliance with traffic signals (0-1)
    reaction_time: float = 1.0  # Driver reaction time in seconds
    following_distance: float = 2.0  # Minimum following distance in meters
    
    # Additional parameters
    last_update_time: float = field(default_factory=time.time)
    in_dilemma_zone: bool = False
    has_priority: bool = False
    
    def update(self, dt: float, signal_state: str, vehicles_ahead: List['SimulatedVehicle']) -> None:
        """
        Update vehicle position and speed.
        
        Args:
            dt: Time step in seconds
            signal_state: Current signal state ('red', 'yellow', 'green')
            vehicles_ahead: List of vehicles ahead of this one
        """
        current_time = time.time()
        self.last_update_time = current_time
        
        # Check if vehicle is in dilemma zone (typically 2.5-5 seconds from stop line)
        time_to_stop_line = abs(self.position) / max(0.1, self.speed)
        self.in_dilemma_zone = 0 < time_to_stop_line < 5.0 and self.position < 0
        
        # Calculate desired speed based on conditions
        desired_speed = self._calculate_desired_speed(signal_state, vehicles_ahead)
        
        # Calculate acceleration
        target_acceleration = self._calculate_acceleration(desired_speed)
        
        # Apply acceleration limits
        max_accel = 2.0 if self.vehicle_type == VehicleType.CAR else 1.0
        max_decel = -4.0 if self.vehicle_type == VehicleType.CAR else -2.0
        
        # Emergency vehicles can accelerate and decelerate faster
        if self.is_emergency:
            max_accel *= 1.5
            max_decel *= 1.2
        
        # Apply limits
        self.acceleration = max(max_decel, min(max_accel, target_acceleration))
        
        # Update speed
        new_speed = self.speed + self.acceleration * dt
        self.speed = max(0.0, min(self.max_speed, new_speed))
        
        # Update position
        self.position += self.speed * dt
        
        # Update waiting time if speed is very low
        if self.speed < 0.1:
            self.waiting_time += dt
            
            # Update status if waiting
            if self.status == VehicleStatus.APPROACHING:
                self.status = VehicleStatus.QUEUED
        else:
            # Reset status if moving
            if self.status == VehicleStatus.QUEUED:
                self.status = VehicleStatus.MOVING
    
    def _calculate_desired_speed(
        self, 
        signal_state: str, 
        vehicles_ahead: List['SimulatedVehicle']
    ) -> float:
        """
        Calculate desired speed based on traffic conditions.
        
        Args:
            signal_state: Current signal state ('red', 'yellow', 'green')
            vehicles_ahead: List of vehicles ahead of this one
        
        Returns:
            Desired speed in m/s
        """
        # Default desired speed is max speed
        desired_speed = self.max_speed
        
        # Check for signal compliance
        if signal_state == 'red' or signal_state == 'yellow':
            # Only stop for red/yellow if before stop line and compliant
            if self.position < 0 and random.random() < self.signal_compliance:
                # Calculate stopping distance
                stopping_distance = -0.5 * self.speed**2 / -4.0  # s = v²/2a
                
                # If we can stop before stop line, reduce speed
                if abs(self.position) > stopping_distance:
                    # Gradual deceleration
                    desired_speed = max(0.0, self.speed - 2.0)
                elif self.in_dilemma_zone:
                    # In dilemma zone with yellow, decide to go or stop
                    if signal_state == 'yellow' and self.speed > 5.0:
                        # Proceed through
                        desired_speed = self.speed
                    else:
                        # Try to stop
                        desired_speed = 0.0
                else:
                    # Too close to stop, proceed through
                    desired_speed = self.speed
            elif self.position >= 0:
                # Already past stop line, proceed
                desired_speed = self.max_speed
        
        # Check for vehicles ahead
        if vehicles_ahead:
            nearest_vehicle = vehicles_ahead[0]
            distance = nearest_vehicle.position - self.position - nearest_vehicle.length
            
            # Calculate safe following distance
            safe_distance = max(
                self.following_distance,
                self.speed * self.reaction_time
            )
            
            if distance < safe_distance:
                # Adjust speed based on car following model
                if distance < 1.0:
                    # Very close, stop
                    desired_speed = 0.0
                else:
                    # Adjust based on time headway
                    desired_speed = min(
                        desired_speed,
                        nearest_vehicle.speed * (distance / safe_distance)
                    )
        
        return desired_speed
    
    def _calculate_acceleration(self, desired_speed: float) -> float:
        """
        Calculate acceleration using Intelligent Driver Model (IDM).
        
        Args:
            desired_speed: Desired speed in m/s
        
        Returns:
            Acceleration in m/s²
        """
        # Simple acceleration model
        speed_difference = desired_speed - self.speed
        
        if abs(speed_difference) < 0.1:
            return 0.0
        elif speed_difference > 0:
            # Accelerate (slower when near desired speed)
            return 2.0 * (1.0 - self.speed / (desired_speed + 0.01))
        else:
            # Decelerate
            return speed_difference
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'vehicle_type': self.vehicle_type.value,
            'origin_lane': self.origin_lane,
            'destination_lane': self.destination_lane,
            'movement_type': self.movement_type.value,
            'position': self.position,
            'speed': self.speed,
            'status': self.status.value,
            'waiting_time': self.waiting_time,
            'is_emergency': self.is_emergency
        }


@dataclass
class TrafficGenerator:
    """Configuration for traffic generation."""
    flow_rate: float  # Vehicles per hour
    composition: Dict[VehicleType, float]  # Vehicle type -> percentage
    movement_distribution: Dict[MovementType, float]  # Movement type -> percentage
    generate_emergency: bool = False
    emergency_frequency: float = 0.01  # Frequency of emergency vehicles (0-1)
    time_pattern: Dict[int, float] = field(default_factory=dict)  # Hour -> flow multiplier
    random_seed: Optional[int] = None
    max_vehicles: int = 1000  # Maximum vehicles to generate


@dataclass
class LaneConfig:
    """Configuration for a traffic lane."""
    id: str
    origin_direction: str  # N, S, E, W
    destination_direction: str  # N, S, E, W
    length: float  # Lane length in meters
    capacity: int  # Maximum vehicles per lane
    allowed_movements: List[MovementType]
    signal_ids: List[str]  # IDs of signals controlling this lane
    saturation_flow: float = 1800.0  # Vehicles per hour of green
    start_offset: float = 100.0  # Distance from stop line to start of lane


@dataclass
class TrafficStats:
    """Traffic statistics for simulation."""
    vehicles_created: int = 0
    vehicles_cleared: int = 0
    total_wait_time: float = 0.0
    total_travel_time: float = 0.0
    max_queue_length: int = 0
    avg_speed: float = 0.0
    throughput: float = 0.0
    current_queue_length: int = 0
    total_stops: int = 0
    total_emissions: float = 0.0
    
    def reset(self):
        """Reset statistics."""
        self.vehicles_created = 0
        self.vehicles_cleared = 0
        self.total_wait_time = 0.0
        self.total_travel_time = 0.0
        self.max_queue_length = 0
        self.avg_speed = 0.0
        self.throughput = 0.0
        self.current_queue_length = 0
        self.total_stops = 0
        self.total_emissions = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'vehicles_created': self.vehicles_created,
            'vehicles_cleared': self.vehicles_cleared,
            'total_wait_time': self.total_wait_time,
            'avg_wait_time': self.total_wait_time / max(1, self.vehicles_cleared),
            'total_travel_time': self.total_travel_time,
            'avg_travel_time': self.total_travel_time / max(1, self.vehicles_cleared),
            'max_queue_length': self.max_queue_length,
            'current_queue_length': self.current_queue_length,
            'avg_speed': self.avg_speed,
            'throughput': self.throughput,
            'total_stops': self.total_stops,
            'stops_per_vehicle': self.total_stops / max(1, self.vehicles_cleared),
            'total_emissions': self.total_emissions
        }


class SimulationModel:
    """
    Traffic simulation model for traffic lights and vehicle movement.
    """
    
    def __init__(
        self,
        lanes: Dict[str, LaneConfig],
        generators: Dict[str, TrafficGenerator],
        time_step: float = 0.1,  # Simulation time step in seconds
        speed_limit: float = 14.0,  # Speed limit in m/s (approximately 50 km/h)
        random_seed: Optional[int] = None,
        real_time: bool = True,  # Whether to run in real-time or fast simulation
        simulation_speed: float = 1.0  # Simulation speed multiplier for non-real-time mode
    ):
        """
        Initialize simulation model.
        
        Args:
            lanes: Dictionary of lane ID -> LaneConfig
            generators: Dictionary of lane ID -> TrafficGenerator
            time_step: Simulation time step in seconds
            speed_limit: Speed limit in m/s
            random_seed: Optional random seed for reproducibility
            real_time: Whether to run in real-time or fast simulation
            simulation_speed: Simulation speed multiplier for non-real-time mode
        """
        self.lanes = lanes
        self.generators = generators
        self.time_step = time_step
        self.speed_limit = speed_limit
        self.real_time = real_time
        self.simulation_speed = simulation_speed
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Vehicle storage
        self.vehicles: Dict[str, List[SimulatedVehicle]] = {lane_id: [] for lane_id in lanes}
        self.vehicle_count = 0
        self.next_vehicle_id = 1
        
        # Signal states
        self.signal_states: Dict[str, str] = {}
        
        # Current simulation time
        self.simulation_time = 0.0
        self.start_time = 0.0
        self.last_generation_time = 0.0
        
        # Statistical tracking
        self.stats = TrafficStats()
        self.lane_stats: Dict[str, TrafficStats] = {lane_id: TrafficStats() for lane_id in lanes}
        
        # Run state
        self.running = False
        self.paused = False
        self.simulation_thread = None
        self.stop_event = threading.Event()
        
        # Event callbacks
        self.vehicle_creation_callback = None
        self.vehicle_clearance_callback = None
        
        logger.info(f"Simulation model initialized with {len(lanes)} lanes")
    
    def start(self) -> bool:
        """
        Start the simulation.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Simulation is already running")
            return True
        
        # Reset state
        self.reset_simulation()
        
        # Start simulation thread
        self.stop_event.clear()
        self.simulation_thread = threading.Thread(
            target=self._simulation_loop,
            daemon=True
        )
        self.simulation_thread.start()
        
        self.running = True
        logger.info("Simulation started")
        
        return True
    
    def stop(self):
        """Stop the simulation."""
        if not self.running:
            return
        
        logger.info("Stopping simulation")
        
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.simulation_thread is not None and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=3.0)
        
        self.running = False
        logger.info("Simulation stopped")
    
    def pause(self):
        """Pause the simulation."""
        if self.running:
            self.paused = True
            logger.info("Simulation paused")
    
    def resume(self):
        """Resume the simulation."""
        if self.running:
            self.paused = False
            logger.info("Simulation resumed")
    
    def reset_simulation(self):
        """Reset the simulation state."""
        # Clear vehicles
        for lane_id in self.vehicles:
            self.vehicles[lane_id] = []
        
        # Reset counters
        self.vehicle_count = 0
        self.next_vehicle_id = 1
        self.simulation_time = 0.0
        self.start_time = time.time()
        self.last_generation_time = 0.0
        
        # Reset statistics
        self.stats.reset()
        for lane_id in self.lane_stats:
            self.lane_stats[lane_id].reset()
        
        logger.info("Simulation reset")
    
    def set_signal_state(self, signal_id: str, state: str):
        """
        Set the state of a traffic signal.
        
        Args:
            signal_id: ID of the signal to set
            state: Signal state ('red', 'yellow', 'green')
        """
        self.signal_states[signal_id] = state.lower()
    
    def get_lane_signal_state(self, lane_id: str) -> str:
        """
        Get the effective signal state for a lane.
        
        Args:
            lane_id: Lane identifier
        
        Returns:
            Effective signal state for the lane
        """
        if lane_id not in self.lanes:
            return 'red'  # Default to red if lane not found
        
        lane_config = self.lanes[lane_id]
        
        # Check if any controlling signal is green
        for signal_id in lane_config.signal_ids:
            if self.signal_states.get(signal_id, 'red') == 'green':
                return 'green'
        
        # Check if any controlling signal is yellow
        for signal_id in lane_config.signal_ids:
            if self.signal_states.get(signal_id, 'red') == 'yellow':
                return 'yellow'
        
        # Otherwise, lane is red
        return 'red'
    
    def _simulation_loop(self):
        """Main simulation loop."""
        last_time = time.time()
        
        while not self.stop_event.is_set():
            # Skip processing if paused
            if self.paused:
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            
            # Calculate time step
            if self.real_time:
                dt = (current_time - last_time) * self.simulation_speed
            else:
                dt = self.time_step
            
            # Update simulation time
            self.simulation_time += dt
            
            # Generate new vehicles
            self._generate_vehicles()
            
            # Update all vehicles
            self._update_vehicles(dt)
            
            # Update statistics
            self._update_statistics()
            
            # Update last time
            last_time = current_time
            
            # Sleep to maintain real-time simulation
            if self.real_time:
                time.sleep(max(0.0, self.time_step / self.simulation_speed))
    
    def _generate_vehicles(self):
        """Generate new vehicles based on traffic generators."""
        current_time = time.time()
        sim_hour = (datetime.now().hour + self.simulation_time / 3600.0) % 24
        
        # Process each lane's generator
        for lane_id, generator in self.generators.items():
            if lane_id not in self.lanes:
                continue
            
            lane_config = self.lanes[lane_id]
            
            # Apply time-of-day pattern if available
            hour = int(sim_hour)
            flow_multiplier = generator.time_pattern.get(hour, 1.0)
            effective_flow = generator.flow_rate * flow_multiplier
            
            # Calculate expected vehicles per second
            vehicles_per_second = effective_flow / 3600.0
            
            # Calculate expected vehicles since last generation
            time_since_last = self.simulation_time - self.last_generation_time
            expected_vehicles = vehicles_per_second * time_since_last
            
            # Generate vehicles probabilistically
            while expected_vehicles > 0:
                if random.random() < expected_vehicles:
                    # Generate a vehicle
                    self._create_vehicle(lane_id, lane_config, generator)
                    expected_vehicles -= 1
                else:
                    break
        
        # Update last generation time
        self.last_generation_time = self.simulation_time
    
    def _create_vehicle(self, lane_id: str, lane_config: LaneConfig, generator: TrafficGenerator):
        """
        Create a new vehicle in the specified lane.
        
        Args:
            lane_id: Lane identifier
            lane_config: Lane configuration
            generator: Traffic generator for this lane
        """
        # Check if lane is at capacity
        if len(self.vehicles[lane_id]) >= lane_config.capacity:
            return
        
        # Check if maximum vehicles has been reached
        if self.vehicle_count >= generator.max_vehicles:
            return
        
        # Determine vehicle type
        vehicle_type = self._select_weighted(generator.composition)
        
        # Determine movement type from allowed movements
        allowed_movements = lane_config.allowed_movements
        movement_weights = {m: generator.movement_distribution.get(m, 0.0) for m in allowed_movements}
        
        # Normalize weights
        total_weight = sum(movement_weights.values())
        if total_weight <= 0:
            return
        
        normalized_weights = {m: w / total_weight for m, w in movement_weights.items()}
        movement_type = self._select_weighted(normalized_weights)
        
        # Determine destination lane based on movement
        destination_lane = self._get_destination_lane(lane_id, lane_config, movement_type)
        
        # Create position at start of lane
        position = -lane_config.length
        
        # Determine if this is an emergency vehicle
        is_emergency = generator.generate_emergency and random.random() < generator.emergency_frequency
        
        # Set vehicle type to EMERGENCY if it's an emergency vehicle
        if is_emergency:
            vehicle_type = VehicleType.EMERGENCY
        
        # Set parameters based on vehicle type
        if vehicle_type == VehicleType.CAR:
            length = random.uniform(4.0, 5.0)
            max_speed = random.uniform(13.0, 15.0)  # ~45-55 km/h
            reaction_time = random.uniform(0.8, 1.2)
        elif vehicle_type == VehicleType.TRUCK:
            length = random.uniform(8.0, 12.0)
            max_speed = random.uniform(11.0, 13.0)  # ~40-45 km/h
            reaction_time = random.uniform(1.0, 1.5)
        elif vehicle_type == VehicleType.BUS:
            length = random.uniform(10.0, 14.0)
            max_speed = random.uniform(10.0, 12.0)  # ~35-45 km/h
            reaction_time = random.uniform(1.0, 1.5)
        elif vehicle_type == VehicleType.MOTORCYCLE:
            length = random.uniform(1.5, 2.5)
            max_speed = random.uniform(13.0, 16.0)  # ~45-60 km/h
            reaction_time = random.uniform(0.6, 1.0)
        elif vehicle_type == VehicleType.BICYCLE:
            length = random.uniform(1.0, 2.0)
            max_speed = random.uniform(5.0, 8.0)  # ~20-30 km/h
            reaction_time = random.uniform(0.5, 0.9)
        elif vehicle_type == VehicleType.EMERGENCY:
            length = random.uniform(5.0, 6.0)
            max_speed = random.uniform(16.0, 19.0)  # ~60-70 km/h
            reaction_time = random.uniform(0.6, 0.8)
        else:
            length = 5.0
            max_speed = 14.0
            reaction_time = 1.0
        
        # Create vehicle
        vehicle = SimulatedVehicle(
            id=self.next_vehicle_id,
            vehicle_type=vehicle_type,
            origin_lane=lane_id,
            destination_lane=destination_lane,
            movement_type=movement_type,
            position=position,
            speed=max_speed / 2.0,  # Start at half max speed
            max_speed=max_speed,
            acceleration=0.0,
            length=length,
            creation_time=self.simulation_time,
            status=VehicleStatus.APPROACHING,
            waiting_time=0.0,
            is_emergency=is_emergency,
            signal_compliance=random.uniform(0.9, 1.0) if not is_emergency else 0.8,
            reaction_time=reaction_time,
            following_distance=random.uniform(1.5, 3.0)
        )
        
        # Add to lane
        self.vehicles[lane_id].append(vehicle)
        
        # Update counters
        self.next_vehicle_id += 1
        self.vehicle_count += 1
        self.stats.vehicles_created += 1
        self.lane_stats[lane_id].vehicles_created += 1
        
        # Invoke callback if registered
        if self.vehicle_creation_callback:
            try:
                self.vehicle_creation_callback(vehicle)
            except Exception as e:
                logger.error(f"Error in vehicle creation callback: {str(e)}")
    
    def _select_weighted(self, weights: Dict[Any, float]) -> Any:
        """
        Select an item based on weights.
        
        Args:
            weights: Dictionary of item -> weight
        
        Returns:
            Selected item
        """
        if not weights:
            return None
        
        items = list(weights.keys())
        weights_list = [weights[item] for item in items]
        
        # Normalize weights
        total = sum(weights_list)
        if total <= 0:
            return random.choice(items)
        
        normalized_weights = [w / total for w in weights_list]
        
        # Select item
        return np.random.choice(items, p=normalized_weights)
    
    def _get_destination_lane(self, lane_id: str, lane_config: LaneConfig, movement_type: MovementType) -> str:
        """
        Determine destination lane based on movement type.
        
        Args:
            lane_id: Origin lane ID
            lane_config: Lane configuration
            movement_type: Movement type
        
        Returns:
            Destination lane ID
        """
        # Simple destination logic based on direction
        origin = lane_config.origin_direction
        
        if movement_type == MovementType.THROUGH:
            # Through movement goes to opposite direction
            opposite_dir = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
            dest_dir = opposite_dir.get(origin, origin)
        elif movement_type == MovementType.LEFT:
            # Left turn based on compass directions
            left_dir = {'N': 'W', 'S': 'E', 'E': 'N', 'W': 'S'}
            dest_dir = left_dir.get(origin, origin)
        elif movement_type == MovementType.RIGHT:
            # Right turn based on compass directions
            right_dir = {'N': 'E', 'S': 'W', 'E': 'S', 'W': 'N'}
            dest_dir = right_dir.get(origin, origin)
        elif movement_type == MovementType.U_TURN:
            # U-turn returns to same direction
            dest_dir = origin
        else:
            dest_dir = origin
        
        # Find matching destination lane (simplified)
        dest_lanes = [l_id for l_id, l in self.lanes.items() 
                     if l.origin_direction == dest_dir]
        
        if dest_lanes:
            return random.choice(dest_lanes)
        
        # Default to same lane if no match found
        return lane_id
    
    def _update_vehicles(self, dt: float):
        """
        Update all vehicles in the simulation.
        
        Args:
            dt: Time step in seconds
        """
        # Process each lane
        for lane_id, lane_vehicles in self.vehicles.items():
            if not lane_vehicles:
                continue
            
            # Get signal state for this lane
            signal_state = self.get_lane_signal_state(lane_id)
            
            # Sort vehicles by position (descending) to process from front to back
            lane_vehicles.sort(key=lambda v: v.position, reverse=True)
            
            # Update each vehicle
            for i, vehicle in enumerate(lane_vehicles):
                # Determine vehicles ahead (if any)
                vehicles_ahead = lane_vehicles[:i] if i > 0 else []
                
                # Update vehicle
                vehicle.update(dt, signal_state, vehicles_ahead)
                
                # Check if vehicle has cleared the intersection
                lane_length = self.lanes[lane_id].length
                if vehicle.position > lane_length * 1.5:  # Well past the stop line
                    vehicle.status = VehicleStatus.CLEARED
            
            # Remove cleared vehicles
            cleared_vehicles = [v for v in lane_vehicles if v.status == VehicleStatus.CLEARED]
            for vehicle in cleared_vehicles:
                lane_vehicles.remove(vehicle)
                
                # Update statistics
                self.stats.vehicles_cleared += 1
                self.lane_stats[lane_id].vehicles_cleared += 1
                
                # Calculate travel time
                travel_time = self.simulation_time - vehicle.creation_time
                self.stats.total_travel_time += travel_time
                self.lane_stats[lane_id].total_travel_time += travel_time
                
                # Add wait time
                self.stats.total_wait_time += vehicle.waiting_time
                self.lane_stats[lane_id].total_wait_time += vehicle.waiting_time
                
                # Invoke callback if registered
                if self.vehicle_clearance_callback:
                    try:
                        self.vehicle_clearance_callback(vehicle)
                    except Exception as e:
                        logger.error(f"Error in vehicle clearance callback: {str(e)}")
    
    def _update_statistics(self):
        """Update simulation statistics."""
        queued_vehicles = 0
        
        # Process each lane
        for lane_id, lane_vehicles in self.vehicles.items():
            # Count queued vehicles
            lane_queued = sum(1 for v in lane_vehicles if v.status == VehicleStatus.QUEUED)
            queued_vehicles += lane_queued
            
            # Update lane statistics
            self.lane_stats[lane_id].current_queue_length = lane_queued
            self.lane_stats[lane_id].max_queue_length = max(
                self.lane_stats[lane_id].max_queue_length, lane_queued
            )
            
            # Calculate average speed
            if lane_vehicles:
                avg_speed = sum(v.speed for v in lane_vehicles) / len(lane_vehicles)
                self.lane_stats[lane_id].avg_speed = avg_speed
            
            # Calculate throughput (vehicles per hour)
            if self.simulation_time > 0:
                throughput = self.lane_stats[lane_id].vehicles_cleared / self.simulation_time * 3600
                self.lane_stats[lane_id].throughput = throughput
        
        # Update global statistics
        self.stats.current_queue_length = queued_vehicles
        self.stats.max_queue_length = max(self.stats.max_queue_length, queued_vehicles)
        
        # Calculate global average speed
        total_vehicles = sum(len(vehicles) for vehicles in self.vehicles.values())
        if total_vehicles > 0:
            total_speed = sum(sum(v.speed for v in vehicles) for vehicles in self.vehicles.values())
            self.stats.avg_speed = total_speed / total_vehicles
        
        # Calculate global throughput
        if self.simulation_time > 0:
            throughput = self.stats.vehicles_cleared / self.simulation_time * 3600
            self.stats.throughput = throughput
    
    def register_vehicle_callbacks(
        self,
        creation_callback: Optional[Callable[[SimulatedVehicle], None]] = None,
        clearance_callback: Optional[Callable[[SimulatedVehicle], None]] = None
    ):
        """
        Register callbacks for vehicle events.
        
        Args:
            creation_callback: Called when a vehicle is created
            clearance_callback: Called when a vehicle clears the intersection
        """
        self.vehicle_creation_callback = creation_callback
        self.vehicle_clearance_callback = clearance_callback
    
    def get_queue_lengths(self) -> Dict[str, int]:
        """
        Get current queue lengths for all lanes.
        
        Returns:
            Dictionary of lane ID -> queue length
        """
        return {
            lane_id: sum(1 for v in vehicles if v.status == VehicleStatus.QUEUED)
            for lane_id, vehicles in self.vehicles.items()
        }
    
    def get_vehicles_in_dilemma_zone(self) -> Dict[str, List[SimulatedVehicle]]:
        """
        Get vehicles currently in the dilemma zone.
        
        Returns:
            Dictionary of lane ID -> list of vehicles in dilemma zone
        """
        return {
            lane_id: [v for v in vehicles if v.in_dilemma_zone]
            for lane_id, vehicles in self.vehicles.items()
        }
    
    def get_lane_vehicle_counts(self) -> Dict[str, int]:
        """
        Get vehicle counts for all lanes.
        
        Returns:
            Dictionary of lane ID -> vehicle count
        """
        return {lane_id: len(vehicles) for lane_id, vehicles in self.vehicles.items()}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get simulation statistics.
        
        Returns:
            Dictionary with simulation statistics
        """
        return {
            'global_stats': self.stats.to_dict(),
            'lane_stats': {lane_id: stats.to_dict() for lane_id, stats in self.lane_stats.items()},
            'simulation_time': self.simulation_time,
            'real_time_elapsed': time.time() - self.start_time,
            'vehicle_count': self.vehicle_count,
            'active_vehicles': sum(len(vehicles) for vehicles in self.vehicles.values())
        }
    
    def get_all_vehicles(self) -> List[Dict[str, Any]]:
        """
        Get information for all vehicles in the simulation.
        
        Returns:
            List of vehicle dictionaries
        """
        all_vehicles = []
        
        for lane_id, vehicles in self.vehicles.items():
            for vehicle in vehicles:
                vehicle_info = vehicle.to_dict()
                vehicle_info['lane_id'] = lane_id
                all_vehicles.append(vehicle_info)
        
        return all_vehicles
    
    def set_flow_rate(self, lane_id: str, flow_rate: float):
        """
        Set flow rate for a lane.
        
        Args:
            lane_id: Lane identifier
            flow_rate: New flow rate in vehicles per hour
        """
        if lane_id in self.generators:
            self.generators[lane_id].flow_rate = flow_rate
    
    def set_time_pattern(self, lane_id: str, hour: int, multiplier: float):
        """
        Set time-of-day flow pattern for a lane.
        
        Args:
            lane_id: Lane identifier
            hour: Hour of day (0-23)
            multiplier: Flow multiplier for this hour
        """
        if lane_id in self.generators:
            self.generators[lane_id].time_pattern[hour] = multiplier
    
    def change_lane_capacity(self, lane_id: str, capacity: int):
        """
        Change capacity of a lane.
        
        Args:
            lane_id: Lane identifier
            capacity: New lane capacity
        """
        if lane_id in self.lanes:
            self.lanes[lane_id].capacity = capacity
    
    def inject_emergency_vehicle(self, lane_id: str):
        """
        Inject an emergency vehicle into a lane.
        
        Args:
            lane_id: Lane identifier
        """
        if lane_id not in self.lanes or lane_id not in self.generators:
            return
        
        lane_config = self.lanes[lane_id]
        generator = self.generators[lane_id]
        
        # Force creation of emergency vehicle
        generator.generate_emergency = True
        generator.emergency_frequency = 1.0
        self._create_vehicle(lane_id, lane_config, generator)
        
        # Reset emergency settings
        generator.generate_emergency = False
        generator.emergency_frequency = 0.01
