"""
Signal optimization module for traffic light timing optimization.
Implements various optimization strategies for different traffic conditions.
"""

import time
import logging
import math
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Set
from dataclasses import dataclass
import datetime
import numpy as np
from collections import defaultdict

# Configure logger for this module
logger = logging.getLogger(__name__)


@dataclass
class TrafficDemand:
    """Traffic demand data for a movement."""
    count: int                  # Number of vehicles
    queue_length: float         # Estimated queue length in vehicles
    flow_rate: float            # Flow rate in vehicles per hour
    saturation_rate: float      # Saturation flow rate (veh/hour of green)
    heavy_vehicle_percentage: float  # Percentage of heavy vehicles
    arrival_rate: float         # Vehicle arrival rate (vehicles per second)
    peak_hour_factor: float     # Peak hour factor (0-1)
    density: float              # Traffic density value (0-1)
    platoon_ratio: float = 1.0  # Platoon ratio for coordination


@dataclass
class MovementData:
    """Data for a specific traffic movement."""
    id: str                     # Movement identifier (e.g., "northbound_through")
    signals: List[str]          # List of signal IDs controlling this movement
    demand: TrafficDemand       # Traffic demand data
    capacity: float             # Movement capacity in vehicles per hour
    saturation_flow: float      # Base saturation flow in vehicles per hour
    min_green: float            # Minimum green time in seconds
    max_green: float            # Maximum green time in seconds
    yellow: float               # Yellow time in seconds
    red_clearance: float        # All-red clearance time in seconds
    startup_lost_time: float    # Startup lost time in seconds
    lane_group_id: str = ""     # Lane group identifier for coordinated movements


@dataclass
class PhaseData:
    """Data for a signal phase."""
    id: str                     # Phase identifier
    movements: List[str]        # List of movement IDs in this phase
    min_green: float            # Minimum green time
    max_green: float            # Maximum green time
    yellow: float               # Yellow time
    red_clearance: float        # All-red clearance time
    pedestrian_calls: int       # Number of pedestrian calls
    vehicle_calls: int          # Number of vehicle calls
    is_coordinated: bool = False  # Whether this phase is coordinated
    priority: int = 1           # Phase priority (higher = more important)


@dataclass
class OptimizationResult:
    """Result of signal timing optimization."""
    cycle_length: float                     # Optimized cycle length in seconds
    phase_durations: Dict[str, float]       # Dictionary of phase ID -> duration
    phase_sequence: List[str]               # Optimized phase sequence
    offset: float = 0.0                     # Offset for coordination in seconds
    performance_index: float = 0.0          # Performance index of optimization
    delay_reduction: float = 0.0            # Estimated delay reduction in percent
    optimization_method: str = ""           # Method used for optimization
    timestamp: float = 0.0                  # Timestamp of optimization
    
    def __post_init__(self):
        """Initialize after creation."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cycle_length': self.cycle_length,
            'phase_durations': self.phase_durations,
            'phase_sequence': self.phase_sequence,
            'offset': self.offset,
            'performance_index': self.performance_index,
            'delay_reduction': self.delay_reduction,
            'optimization_method': self.optimization_method,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationResult':
        """Create an OptimizationResult from a dictionary."""
        return cls(
            cycle_length=data['cycle_length'],
            phase_durations=data['phase_durations'],
            phase_sequence=data['phase_sequence'],
            offset=data.get('offset', 0.0),
            performance_index=data.get('performance_index', 0.0),
            delay_reduction=data.get('delay_reduction', 0.0),
            optimization_method=data.get('optimization_method', ''),
            timestamp=data.get('timestamp', time.time())
        )


class SignalOptimizer:
    """
    Base class for traffic signal optimization.
    """
    
    def __init__(
        self,
        min_cycle: float = 30.0,
        max_cycle: float = 180.0,
        lost_time_per_phase: float = 4.0,
        target_v_c_ratio: float = 0.9,
        update_interval: float = 300.0,
        critical_intersection: bool = False
    ):
        """
        Initialize signal optimizer.
        
        Args:
            min_cycle: Minimum cycle length in seconds
            max_cycle: Maximum cycle length in seconds
            lost_time_per_phase: Lost time per phase in seconds
            target_v_c_ratio: Target volume-to-capacity ratio
            update_interval: Time between optimization updates in seconds
            critical_intersection: Whether this is a critical intersection
        """
        self.min_cycle = min_cycle
        self.max_cycle = max_cycle
        self.lost_time_per_phase = lost_time_per_phase
        self.target_v_c_ratio = target_v_c_ratio
        self.update_interval = update_interval
        self.critical_intersection = critical_intersection
        
        # Internal state
        self.last_update_time = 0.0
        self.current_result = None
        self.historical_results = []
        
        logger.info(f"Signal optimizer initialized with min_cycle={min_cycle}, max_cycle={max_cycle}")
    
    def optimize(
        self,
        movements: Dict[str, MovementData],
        phases: Dict[str, PhaseData],
        current_timings: Dict[str, float],
        coordination_data: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Optimize signal timings based on current traffic conditions.
        
        Args:
            movements: Dictionary of movement ID -> MovementData
            phases: Dictionary of phase ID -> PhaseData
            current_timings: Dictionary of phase ID -> current duration
            coordination_data: Optional data for coordination with adjacent intersections
        
        Returns:
            OptimizationResult with optimized timings
        """
        # Check if update is needed
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval and self.current_result:
            logger.debug("Skipping optimization, using cached result")
            return self.current_result
        
        # Implement a basic optimization in the base class
        # Specific optimization algorithms should be implemented in subclasses
        
        # Determine critical movements
        critical_movements = self._find_critical_movements(movements, phases)
        
        # Calculate optimal cycle length
        cycle_length = self._calculate_webster_cycle(critical_movements, phases)
        
        # Adjust to min/max bounds
        cycle_length = max(self.min_cycle, min(self.max_cycle, cycle_length))
        
        # Allocate green time proportionally
        phase_durations = self._allocate_green_time(cycle_length, movements, phases, critical_movements)
        
        # Create phase sequence
        phase_sequence = list(phases.keys())  # Default to same sequence
        
        # Create optimization result
        result = OptimizationResult(
            cycle_length=cycle_length,
            phase_durations=phase_durations,
            phase_sequence=phase_sequence,
            optimization_method="webster",
            performance_index=0.0,  # Placeholder, calculate actual PI in subclasses
            delay_reduction=0.0      # Placeholder
        )
        
        # Update internal state
        self.last_update_time = current_time
        self.current_result = result
        self.historical_results.append(result)
        
        # Keep history limited
        if len(self.historical_results) > 100:
            self.historical_results = self.historical_results[-100:]
        
        logger.info(f"Optimized signal timing: cycle={cycle_length:.1f}s")
        return result
    
    def _find_critical_movements(
        self, 
        movements: Dict[str, MovementData],
        phases: Dict[str, PhaseData]
    ) -> Dict[str, List[str]]:
        """
        Find critical movements for each phase.
        
        Args:
            movements: Dictionary of movement ID -> MovementData
            phases: Dictionary of phase ID -> PhaseData
        
        Returns:
            Dictionary of phase ID -> list of critical movement IDs
        """
        critical_movements = {}
        
        for phase_id, phase in phases.items():
            if not phase.movements:
                critical_movements[phase_id] = []
                continue
            
            # Find movement with highest v/c ratio in this phase
            max_flow_ratio = 0.0
            critical_movement = None
            
            for movement_id in phase.movements:
                if movement_id in movements:
                    movement = movements[movement_id]
                    if movement.saturation_flow > 0:
                        flow_ratio = movement.demand.flow_rate / movement.saturation_flow
                        if flow_ratio > max_flow_ratio:
                            max_flow_ratio = flow_ratio
                            critical_movement = movement_id
            
            critical_movements[phase_id] = [critical_movement] if critical_movement else []
        
        return critical_movements
    
    def _calculate_webster_cycle(
        self,
        critical_movements: Dict[str, List[str]],
        phases: Dict[str, PhaseData]
    ) -> float:
        """
        Calculate optimal cycle length using Webster's method.
        
        Args:
            critical_movements: Dictionary of phase ID -> list of critical movement IDs
            phases: Dictionary of phase ID -> PhaseData
        
        Returns:
            Optimal cycle length in seconds
        """
        # Calculate total lost time
        total_lost_time = sum(
            self.lost_time_per_phase for phase_id in phases if critical_movements.get(phase_id)
        )
        
        # Calculate sum of critical flow ratios
        y_critical_sum = sum(
            max(movement.demand.flow_rate / movement.saturation_flow 
                for movement_id in critical_movements[phase_id]
                for movement in [self.movements[movement_id]])
            for phase_id in critical_movements
            if critical_movements[phase_id] and critical_movements[phase_id][0] in self.movements
        )
        
        # Handle case where y_critical_sum is too high or zero
        if y_critical_sum >= 0.95 or y_critical_sum <= 0.0:
            return self.max_cycle if y_critical_sum >= 0.95 else self.min_cycle
        
        # Webster's formula: C = (1.5L + 5) / (1 - Y)
        # where L is total lost time and Y is sum of critical flow ratios
        cycle_length = (1.5 * total_lost_time + 5) / (1 - y_critical_sum)
        
        return cycle_length
    
    def _allocate_green_time(
        self,
        cycle_length: float,
        movements: Dict[str, MovementData],
        phases: Dict[str, PhaseData],
        critical_movements: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Allocate green time to phases based on critical movements.
        
        Args:
            cycle_length: Cycle length in seconds
            movements: Dictionary of movement ID -> MovementData
            phases: Dictionary of phase ID -> PhaseData
            critical_movements: Dictionary of phase ID -> list of critical movement IDs
        
        Returns:
            Dictionary of phase ID -> duration
        """
        # Calculate effective green time (cycle length minus total lost time)
        total_lost_time = sum(
            self.lost_time_per_phase for phase_id in phases if critical_movements.get(phase_id)
        )
        effective_green = cycle_length - total_lost_time
        
        # Calculate sum of critical flow ratios
        y_critical_values = {}
        y_critical_sum = 0.0
        
        for phase_id, critical_ids in critical_movements.items():
            if not critical_ids:
                y_critical_values[phase_id] = 0.0
                continue
            
            critical_id = critical_ids[0]
            if critical_id in movements:
                movement = movements[critical_id]
                if movement.saturation_flow > 0:
                    y_critical = movement.demand.flow_rate / movement.saturation_flow
                    y_critical_values[phase_id] = y_critical
                    y_critical_sum += y_critical
                else:
                    y_critical_values[phase_id] = 0.0
            else:
                y_critical_values[phase_id] = 0.0
        
        # Allocate green time proportionally to critical flow ratios
        phase_durations = {}
        
        for phase_id, phase in phases.items():
            if y_critical_sum > 0 and phase_id in y_critical_values:
                # Calculate green time proportionally
                green_ratio = y_critical_values[phase_id] / y_critical_sum if y_critical_sum > 0 else 0
                green_time = effective_green * green_ratio
                
                # Add lost time components
                phase_duration = green_time + phase.yellow + phase.red_clearance
                
                # Apply min/max constraints
                phase_duration = max(phase.min_green + phase.yellow + phase.red_clearance, 
                                    min(phase.max_green + phase.yellow + phase.red_clearance, 
                                        phase_duration))
                
                phase_durations[phase_id] = phase_duration
            else:
                # Assign minimum green time for phases without critical movements
                phase_durations[phase_id] = phase.min_green + phase.yellow + phase.red_clearance
        
        # Normalize durations to match cycle length
        total_duration = sum(phase_durations.values())
        if total_duration > 0:
            scale_factor = cycle_length / total_duration
            for phase_id in phase_durations:
                phase_durations[phase_id] *= scale_factor
        
        return phase_durations
    
    def _calculate_performance_index(
        self,
        cycle_length: float,
        phase_durations: Dict[str, float],
        movements: Dict[str, MovementData],
        phases: Dict[str, PhaseData]
    ) -> float:
        """
        Calculate performance index for optimization result.
        
        Args:
            cycle_length: Cycle length in seconds
            phase_durations: Dictionary of phase ID -> duration
            movements: Dictionary of movement ID -> MovementData
            phases: Dictionary of phase ID -> PhaseData
        
        Returns:
            Performance index value (lower is better)
        """
        # Base performance index on total delay
        total_delay = 0.0
        total_vehicles = 0
        
        for movement_id, movement in movements.items():
            # Find phase containing this movement
            phase_id = next((pid for pid, phase in phases.items() 
                           if movement_id in phase.movements), None)
            
            if not phase_id:
                continue
            
            # Calculate green time for this movement
            green_time = phase_durations[phase_id] - phases[phase_id].yellow - phases[phase_id].red_clearance
            
            # Calculate capacity
            capacity = movement.saturation_flow * green_time / cycle_length
            
            # Calculate delay using Webster's delay formula
            x = min(0.99, movement.demand.flow_rate / max(1, capacity))  # v/c ratio capped at 0.99
            
            delay_term1 = cycle_length * (1 - green_time/cycle_length)**2 / (2 * (1 - x * green_time/cycle_length))
            delay_term2 = x**2 / (2 * movement.demand.flow_rate * (1 - x))
            
            avg_delay = 0.9 * (delay_term1 + delay_term2)  # Scale factor for calibration
            
            # Add to total delay
            movement_vehicles = movement.demand.count
            total_delay += avg_delay * movement_vehicles
            total_vehicles += movement_vehicles
        
        # Calculate average delay per vehicle as performance index
        if total_vehicles > 0:
            performance_index = total_delay / total_vehicles
        else:
            performance_index = 0.0
        
        return performance_index
    
    def get_historical_results(self) -> List[OptimizationResult]:
        """
        Get historical optimization results.
        
        Returns:
            List of historical OptimizationResult objects
        """
        return self.historical_results
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze performance of current optimization.
        
        Returns:
            Dictionary with performance analysis
        """
        if not self.historical_results:
            return {'status': 'No optimization history available'}
        
        # Get latest and historical results
        latest = self.historical_results[-1]
        
        # Calculate statistics from historical data
        cycle_lengths = [result.cycle_length for result in self.historical_results]
        avg_cycle = sum(cycle_lengths) / len(cycle_lengths)
        min_cycle = min(cycle_lengths)
        max_cycle = max(cycle_lengths)
        
        # Analyze phase durations
        phase_stats = {}
        for phase_id in latest.phase_durations:
            phase_durations = [result.phase_durations.get(phase_id, 0) for result in self.historical_results 
                              if phase_id in result.phase_durations]
            
            if phase_durations:
                phase_stats[phase_id] = {
                    'avg': sum(phase_durations) / len(phase_durations),
                    'min': min(phase_durations),
                    'max': max(phase_durations),
                    'current': latest.phase_durations[phase_id]
                }
        
        # Create analysis report
        return {
            'status': 'OK',
            'optimization_method': latest.optimization_method,
            'current_cycle': latest.cycle_length,
            'avg_cycle': avg_cycle,
            'min_cycle': min_cycle,
            'max_cycle': max_cycle,
            'cycle_stability': 1.0 - (max_cycle - min_cycle) / max(1.0, avg_cycle),  # Higher is more stable
            'phase_stats': phase_stats,
            'latest_performance_index': latest.performance_index,
            'latest_delay_reduction': latest.delay_reduction,
            'update_frequency': len(self.historical_results) / 
                               (self.historical_results[-1].timestamp - self.historical_results[0].timestamp 
                                if len(self.historical_results) > 1 else 1.0)
        }


class FixedTimeOptimizer(SignalOptimizer):
    """
    Fixed-time signal optimizer using Webster's method.
    """
    
    def __init__(
        self,
        min_cycle: float = 30.0,
        max_cycle: float = 120.0,
        lost_time_per_phase: float = 4.0,
        target_v_c_ratio: float = 0.9,
        update_interval: float = 1800.0,  # 30 minutes between updates
        critical_intersection: bool = False,
        periodic_update: bool = True
    ):
        """
        Initialize fixed-time optimizer.
        
        Args:
            min_cycle: Minimum cycle length in seconds
            max_cycle: Maximum cycle length in seconds
            lost_time_per_phase: Lost time per phase in seconds
            target_v_c_ratio: Target volume-to-capacity ratio
            update_interval: Time between optimization updates in seconds
            critical_intersection: Whether this is a critical intersection
            periodic_update: Whether to update timings periodically
        """
        super().__init__(
            min_cycle=min_cycle,
            max_cycle=max_cycle,
            lost_time_per_phase=lost_time_per_phase,
            target_v_c_ratio=target_v_c_ratio,
            update_interval=update_interval,
            critical_intersection=critical_intersection
        )
        self.periodic_update = periodic_update
        self.movements = {}  # Cache for movements
        
        logger.info("Fixed-time signal optimizer initialized")
    
    def optimize(
        self,
        movements: Dict[str, MovementData],
        phases: Dict[str, PhaseData],
        current_timings: Dict[str, float],
        coordination_data: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Optimize signal timings using Webster's method.
        
        Args:
            movements: Dictionary of movement ID -> MovementData
            phases: Dictionary of phase ID -> PhaseData
            current_timings: Dictionary of phase ID -> current duration
            coordination_data: Optional data for coordination with adjacent intersections
        
        Returns:
            OptimizationResult with optimized timings
        """
        # Check if update is needed
        current_time = time.time()
        if self.periodic_update and current_time - self.last_update_time < self.update_interval and self.current_result:
            logger.debug("Skipping optimization, using cached result")
            return self.current_result
        
        # Cache movements for use in internal methods
        self.movements = movements
        
        # Find critical movements
        critical_movements = self._find_critical_movements(movements, phases)
        
        # Calculate optimal cycle length
        cycle_length = self._calculate_webster_cycle(critical_movements, phases)
        
        # Adjust to min/max bounds
        cycle_length = max(self.min_cycle, min(self.max_cycle, cycle_length))
        
        # Allocate green time proportionally
        phase_durations = self._allocate_green_time(cycle_length, movements, phases, critical_movements)
        
        # Create phase sequence - fixed time uses the same sequence
        phase_sequence = list(phases.keys())
        
        # Calculate performance index
        performance_index = self._calculate_performance_index(cycle_length, phase_durations, movements, phases)
        
        # Calculate delay reduction
        delay_reduction = self._calculate_delay_reduction(cycle_length, phase_durations, 
                                                        current_timings, movements, phases)
        
        # Process coordination if provided
        offset = 0.0
        if coordination_data and coordination_data.get('enabled', False):
            offset = self._calculate_coordination_offset(cycle_length, phase_durations, 
                                                         coordination_data, movements, phases)
        
        # Create optimization result
        result = OptimizationResult(
            cycle_length=cycle_length,
            phase_durations=phase_durations,
            phase_sequence=phase_sequence,
            offset=offset,
            optimization_method="webster_fixed",
            performance_index=performance_index,
            delay_reduction=delay_reduction
        )
        
        # Update internal state
        self.last_update_time = current_time
        self.current_result = result
        self.historical_results.append(result)
        
        # Keep history limited
        if len(self.historical_results) > 100:
            self.historical_results = self.historical_results[-100:]
        
        logger.info(f"Fixed-time optimization: cycle={cycle_length:.1f}s, PI={performance_index:.2f}")
        return result
    
    def _calculate_delay_reduction(
        self,
        cycle_length: float,
        phase_durations: Dict[str, float],
        current_timings: Dict[str, float],
        movements: Dict[str, MovementData],
        phases: Dict[str, PhaseData]
    ) -> float:
        """
        Calculate delay reduction compared to current timings.
        
        Args:
            cycle_length: Optimized cycle length in seconds
            phase_durations: Dictionary of phase ID -> optimized duration
            current_timings: Dictionary of phase ID -> current duration
            movements: Dictionary of movement ID -> MovementData
            phases: Dictionary of phase ID -> PhaseData
        
        Returns:
            Delay reduction percentage (0-100)
        """
        # Calculate current cycle length
        current_cycle = sum(current_timings.values())
        
        # Calculate delay with current timings
        current_pi = self._calculate_performance_index(current_cycle, current_timings, movements, phases)
        
        # Calculate delay with optimized timings
        optimized_pi = self._calculate_performance_index(cycle_length, phase_durations, movements, phases)
        
        # Calculate reduction percentage
        if current_pi > 0:
            delay_reduction = max(0, (current_pi - optimized_pi) / current_pi * 100)
        else:
            delay_reduction = 0.0
        
        return delay_reduction
    
    def _calculate_coordination_offset(
        self,
        cycle_length: float,
        phase_durations: Dict[str, float],
        coordination_data: Dict[str, Any],
        movements: Dict[str, MovementData],
        phases: Dict[str, PhaseData]
    ) -> float:
        """
        Calculate signal offset for coordination.
        
        Args:
            cycle_length: Optimized cycle length in seconds
            phase_durations: Dictionary of phase ID -> optimized duration
            coordination_data: Coordination data
            movements: Dictionary of movement ID -> MovementData
            phases: Dictionary of phase ID -> PhaseData
        
        Returns:
            Offset value in seconds
        """
        # Get coordination parameters
        travel_time = coordination_data.get('travel_time', 0.0)
        reference_offset = coordination_data.get('reference_offset', 0.0)
        direction = coordination_data.get('direction', 'two_way')
        
        # Identify coordinated phase
        coordinated_phase = next((phase_id for phase_id, phase in phases.items() 
                                if phase.is_coordinated), None)
        
        if not coordinated_phase:
            return 0.0
        
        # Calculate offset
        if direction == 'one_way':
            # One-way coordination: offset = travel time
            offset = travel_time % cycle_length
        else:
            # Two-way coordination: offset = half cycle
            offset = (cycle_length / 2) % cycle_length
        
        # Adjust for reference offset
        offset = (offset + reference_offset) % cycle_length
        
        return offset


class AdaptiveOptimizer(SignalOptimizer):
    """
    Adaptive signal optimizer that responds to real-time traffic conditions.
    """
    
    def __init__(
        self,
        min_cycle: float = 30.0,
        max_cycle: float = 150.0,
        lost_time_per_phase: float = 4.0,
        target_v_c_ratio: float = 0.9,
        update_interval: float = 180.0,  # 3 minutes between updates
        critical_intersection: bool = False,
        adaptation_rate: float = 0.3,
        queue_weight: float = 0.5,
        historical_weight: float = 0.2,
        min_green_extension: float = 2.0,
        max_green_extension: float = 20.0
    ):
        """
        Initialize adaptive optimizer.
        
        Args:
            min_cycle: Minimum cycle length in seconds
            max_cycle: Maximum cycle length in seconds
            lost_time_per_phase: Lost time per phase in seconds
            target_v_c_ratio: Target volume-to-capacity ratio
            update_interval: Time between optimization updates in seconds
            critical_intersection: Whether this is a critical intersection
            adaptation_rate: Rate of adaptation to new conditions (0-1)
            queue_weight: Weight for queue length in decision making
            historical_weight: Weight for historical patterns
            min_green_extension: Minimum green extension time
            max_green_extension: Maximum green extension time
        """
        super().__init__(
            min_cycle=min_cycle,
            max_cycle=max_cycle,
            lost_time_per_phase=lost_time_per_phase,
            target_v_c_ratio=target_v_c_ratio,
            update_interval=update_interval,
            critical_intersection=critical_intersection
        )
        self.adaptation_rate = adaptation_rate
        self.queue_weight = queue_weight
        self.historical_weight = historical_weight
        self.min_green_extension = min_green_extension
        self.max_green_extension = max_green_extension
        
        # Historical data for adaptation
        self.historical_demands = {}  # time_of_day -> movement_id -> demand
        self.historical_timings = {}  # time_of_day -> cycle_length, phase_durations
        self.movements = {}  # Cache for movements
        
        # Store past demand data
        self.past_demands = []  # List of (timestamp, movement_id, demand) tuples
        
        logger.info("Adaptive signal optimizer initialized")
    
    def optimize(
        self,
        movements: Dict[str, MovementData],
        phases: Dict[str, PhaseData],
        current_timings: Dict[str, float],
        coordination_data: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Optimize signal timings using adaptive approach.
        
        Args:
            movements: Dictionary of movement ID -> MovementData
            phases: Dictionary of phase ID -> PhaseData
            current_timings: Dictionary of phase ID -> current duration
            coordination_data: Optional data for coordination with adjacent intersections
        
        Returns:
            OptimizationResult with optimized timings
        """
        # Cache movements for use in internal methods
        self.movements = movements
        
        # Get current time of day
        current_time = time.time()
        time_of_day = datetime.datetime.now().strftime("%H:%M")
        
        # Update historical data
        self._update_historical_data(time_of_day, movements, current_timings)
        
        # Find critical movements
        critical_movements = self._find_critical_movements(movements, phases)
        
        # Calculate base cycle length using Webster's method
        base_cycle = self._calculate_webster_cycle(critical_movements, phases)
        
        # Adjust cycle length based on historical patterns
        historical_cycle = self._get_historical_cycle(time_of_day)
        
        # Blend current and historical cycles
        if historical_cycle > 0:
            cycle_length = (1 - self.historical_weight) * base_cycle + self.historical_weight * historical_cycle
        else:
            cycle_length = base_cycle
        
        # Adjust to min/max bounds
        cycle_length = max(self.min_cycle, min(self.max_cycle, cycle_length))
        
        # Allocate green time proportionally but consider queues
        phase_durations = self._adaptive_green_allocation(cycle_length, movements, phases, critical_movements)
        
        # Create phase sequence - may reorder phases based on demand
        phase_sequence = self._optimize_phase_sequence(phases, movements)
        
        # Calculate performance index
        performance_index = self._calculate_performance_index(cycle_length, phase_durations, movements, phases)
        
        # Calculate delay reduction
        delay_reduction = self._calculate_delay_reduction(cycle_length, phase_durations, 
                                                        current_timings, movements, phases)
        
        # Process coordination if provided
        offset = 0.0
        if coordination_data and coordination_data.get('enabled', False):
            offset = self._calculate_coordination_offset(cycle_length, phase_durations, 
                                                         coordination_data, movements, phases)
        
        # Create optimization result
        result = OptimizationResult(
            cycle_length=cycle_length,
            phase_durations=phase_durations,
            phase_sequence=phase_sequence,
            offset=offset,
            optimization_method="adaptive",
            performance_index=performance_index,
            delay_reduction=delay_reduction
        )
        
        # Adapt to current result based on adaptation rate
        if self.current_result:
            # Blend previous and current results
            blended_result = self._blend_results(self.current_result, result)
            result = blended_result
        
        # Update internal state
        self.last_update_time = current_time
        self.current_result = result
        self.historical_results.append(result)
        
        # Keep history limited
        if len(self.historical_results) > 100:
            self.historical_results = self.historical_results[-100:]
        
        logger.info(f"Adaptive optimization: cycle={cycle_length:.1f}s, PI={performance_index:.2f}")
        return result
    
    def _update_historical_data(
        self,
        time_of_day: str,
        movements: Dict[str, MovementData],
        current_timings: Dict[str, float]
    ):
        """
        Update historical data for time of day.
        
        Args:
            time_of_day: Current time of day (HH:MM format)
            movements: Dictionary of movement ID -> MovementData
            current_timings: Dictionary of phase ID -> current duration
        """
        # Round time to nearest 15 minutes for binning
        hour, minute = map(int, time_of_day.split(':'))
        minute = (minute // 15) * 15
        time_bin = f"{hour:02d}:{minute:02d}"
        
        # Initialize time bin if not exists
        if time_bin not in self.historical_demands:
            self.historical_demands[time_bin] = {}
        
        if time_bin not in self.historical_timings:
            self.historical_timings[time_bin] = {
                'cycle_length': sum(current_timings.values()),
                'phase_durations': current_timings.copy(),
                'count': 0
            }
        
        # Update movement demands
        for movement_id, movement in movements.items():
            if movement_id not in self.historical_demands[time_bin]:
                self.historical_demands[time_bin][movement_id] = {
                    'count': 0,
                    'flow_rate': 0,
                    'queue_length': 0
                }
            
            # Update with exponential moving average
            hist = self.historical_demands[time_bin][movement_id]
            alpha = 1.0 / (hist['count'] + 1) if hist['count'] < 10 else 0.1  # Adaptation rate
            
            hist['flow_rate'] = (1 - alpha) * hist['flow_rate'] + alpha * movement.demand.flow_rate
            hist['queue_length'] = (1 - alpha) * hist['queue_length'] + alpha * movement.demand.queue_length
            hist['count'] += 1
        
        # Update timing history
        hist_timing = self.historical_timings[time_bin]
        cycle_length = sum(current_timings.values())
        
        # Update with exponential moving average
        alpha = 1.0 / (hist_timing['count'] + 1) if hist_timing['count'] < 10 else 0.1
        
        hist_timing['cycle_length'] = (1 - alpha) * hist_timing['cycle_length'] + alpha * cycle_length
        
        for phase_id, duration in current_timings.items():
            if phase_id not in hist_timing['phase_durations']:
                hist_timing['phase_durations'][phase_id] = duration
            else:
                hist_timing['phase_durations'][phase_id] = (
                    (1 - alpha) * hist_timing['phase_durations'][phase_id] + alpha * duration
                )
        
        hist_timing['count'] += 1
        
        # Store demand data for time-series analysis
        timestamp = time.time()
        for movement_id, movement in movements.items():
            self.past_demands.append((timestamp, movement_id, {
                'flow_rate': movement.demand.flow_rate,
                'queue_length': movement.demand.queue_length,
                'time_of_day': time_of_day
            }))
        
        # Keep history limited
        if len(self.past_demands) > 1000:
            self.past_demands = self.past_demands[-1000:]
    
    def _get_historical_cycle(self, time_of_day: str) -> float:
        """
        Get historical cycle length for time of day.
        
        Args:
            time_of_day: Current time of day (HH:MM format)
        
        Returns:
            Historical cycle length or 0 if no history
        """
        # Round time to nearest 15 minutes for binning
        hour, minute = map(int, time_of_day.split(':'))
        minute = (minute // 15) * 15
        time_bin = f"{hour:02d}:{minute:02d}"
        
        if time_bin in self.historical_timings:
            return self.historical_timings[time_bin]['cycle_length']
        
        # Try adjacent time bins if exact match not found
        adjacent_bins = []
        
        # Previous 15-minute bin
        prev_minute = minute - 15
        prev_hour = hour
        if prev_minute < 0:
            prev_minute = 45
            prev_hour = (hour - 1) % 24
        adjacent_bins.append(f"{prev_hour:02d}:{prev_minute:02d}")
        
        # Next 15-minute bin
        next_minute = minute + 15
        next_hour = hour
        if next_minute >= 60:
            next_minute = 0
            next_hour = (hour + 1) % 24
        adjacent_bins.append(f"{next_hour:02d}:{next_minute:02d}")
        
        # Check adjacent bins
        for bin_time in adjacent_bins:
            if bin_time in self.historical_timings:
                return self.historical_timings[bin_time]['cycle_length']
        
        return 0.0  # No historical data found
    
    def _adaptive_green_allocation(
        self,
        cycle_length: float,
        movements: Dict[str, MovementData],
        phases: Dict[str, PhaseData],
        critical_movements: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Allocate green time adaptively based on flow and queues.
        
        Args:
            cycle_length: Cycle length in seconds
            movements: Dictionary of movement ID -> MovementData
            phases: Dictionary of phase ID -> PhaseData
            critical_movements: Dictionary of phase ID -> list of critical movement IDs
        
        Returns:
            Dictionary of phase ID -> duration
        """
        # Calculate lost time
        total_lost_time = sum(
            phase.yellow + phase.red_clearance for phase in phases.values()
        )
        
        # Calculate effective green time
        effective_green = cycle_length - total_lost_time
        
        # Calculate base green times using flow ratios
        flow_green_times = {}
        total_flow_ratio = 0.0
        
        for phase_id, phase in phases.items():
            # Calculate flow ratio for this phase
            phase_flow_ratio = 0.0
            
            for movement_id in phase.movements:
                if movement_id in movements:
                    movement = movements[movement_id]
                    if movement.saturation_flow > 0:
                        flow_ratio = movement.demand.flow_rate / movement.saturation_flow
                        phase_flow_ratio = max(phase_flow_ratio, flow_ratio)
            
            flow_green_times[phase_id] = phase_flow_ratio
            total_flow_ratio += phase_flow_ratio
        
        # Normalize flow ratios to allocate green time
        if total_flow_ratio > 0:
            for phase_id in flow_green_times:
                flow_green_times[phase_id] = (
                    effective_green * flow_green_times[phase_id] / total_flow_ratio
                )
        
        # Calculate queue-based green times
        queue_green_times = {}
        total_queue_length = 0.0
        
        for phase_id, phase in phases.items():
            # Calculate total queue for this phase
            phase_queue = 0.0
            
            for movement_id in phase.movements:
                if movement_id in movements:
                    phase_queue += movements[movement_id].demand.queue_length
            
            queue_green_times[phase_id] = phase_queue
            total_queue_length += phase_queue
        
        # Normalize queue lengths to allocate green time
        if total_queue_length > 0:
            for phase_id in queue_green_times:
                queue_green_times[phase_id] = (
                    effective_green * queue_green_times[phase_id] / total_queue_length
                )
        
        # Blend flow-based and queue-based allocations
        phase_durations = {}
        
        for phase_id, phase in phases.items():
            # Blend allocations using queue weight
            green_time = (
                (1 - self.queue_weight) * flow_green_times.get(phase_id, 0) +
                self.queue_weight * queue_green_times.get(phase_id, 0)
            )
            
            # Add yellow and red clearance
            duration = green_time + phase.yellow + phase.red_clearance
            
            # Apply min/max constraints
            min_duration = phase.min_green + phase.yellow + phase.red_clearance
            max_duration = phase.max_green + phase.yellow + phase.red_clearance
            
            duration = max(min_duration, min(max_duration, duration))
            
            phase_durations[phase_id] = duration
        
        # Normalize durations to match cycle length
        total_duration = sum(phase_durations.values())
        if total_duration > 0:
            scale_factor = cycle_length / total_duration
            for phase_id in phase_durations:
                phase_durations[phase_id] *= scale_factor
        
        return phase_durations
    
    def _optimize_phase_sequence(
        self,
        phases: Dict[str, PhaseData],
        movements: Dict[str, MovementData]
    ) -> List[str]:
        """
        Optimize phase sequence based on demand.
        
        Args:
            phases: Dictionary of phase ID -> PhaseData
            movements: Dictionary of movement ID -> MovementData
        
        Returns:
            Optimized list of phase IDs
        """
        # Default to original sequence
        default_sequence = list(phases.keys())
        
        # For adaptive control, we might want to prioritize phases with higher demand
        # However, changing sequence too often can disrupt traffic flow
        # So we prioritize but maintain compatibility with coordination
        
        # Calculate demand score for each phase
        phase_scores = {}
        
        for phase_id, phase in phases.items():
            # Base score on total flow and queue length
            flow_sum = 0.0
            queue_sum = 0.0
            
            for movement_id in phase.movements:
                if movement_id in movements:
                    flow_sum += movements[movement_id].demand.flow_rate
                    queue_sum += movements[movement_id].demand.queue_length
            
            # Combine into score, weighted by queue
            score = (1 - self.queue_weight) * flow_sum + self.queue_weight * queue_sum
            
            # Prioritize coordinated phases
            if phase.is_coordinated:
                score *= 1.5
            
            # Apply priority multiplier
            score *= phase.priority
            
            phase_scores[phase_id] = score
        
        # For non-coordinated phases, sort by score
        coord_phases = [p_id for p_id, p in phases.items() if p.is_coordinated]
        non_coord_phases = [p_id for p_id in default_sequence if p_id not in coord_phases]
        
        # Sort non-coordinated phases by score
        non_coord_phases.sort(key=lambda p_id: phase_scores.get(p_id, 0), reverse=True)
        
        # Rebuild sequence maintaining coordinated phases in original positions
        optimized_sequence = []
        non_coord_index = 0
        
        for phase_id in default_sequence:
            if phase_id in coord_phases:
                optimized_sequence.append(phase_id)
            else:
                if non_coord_index < len(non_coord_phases):
                    optimized_sequence.append(non_coord_phases[non_coord_index])
                    non_coord_index += 1
        
        return optimized_sequence
    
    def _blend_results(
        self,
        previous: OptimizationResult,
        current: OptimizationResult
    ) -> OptimizationResult:
        """
        Blend previous and current optimization results.
        
        Args:
            previous: Previous optimization result
            current: Current optimization result
        
        Returns:
            Blended OptimizationResult
        """
        # Blend cycle length
        cycle_length = (
            (1 - self.adaptation_rate) * previous.cycle_length +
            self.adaptation_rate * current.cycle_length
        )
        
        # Blend phase durations
        phase_durations = {}
        all_phases = set(list(previous.phase_durations.keys()) + list(current.phase_durations.keys()))
        
        for phase_id in all_phases:
            prev_duration = previous.phase_durations.get(phase_id, 0)
            curr_duration = current.phase_durations.get(phase_id, 0)
            
            if prev_duration == 0:
                # New phase, use current duration
                phase_durations[phase_id] = curr_duration
            elif curr_duration == 0:
                # Removed phase, use zero
                phase_durations[phase_id] = 0
            else:
                # Blend durations
                phase_durations[phase_id] = (
                    (1 - self.adaptation_rate) * prev_duration +
                    self.adaptation_rate * curr_duration
                )
        
        # Use the current phase sequence
        phase_sequence = current.phase_sequence
        
        # Blend offset
        offset = (
            (1 - self.adaptation_rate) * previous.offset +
            self.adaptation_rate * current.offset
        )
        
        # Use the better performance index
        performance_index = min(previous.performance_index, current.performance_index)
        
        # Use the current delay reduction
        delay_reduction = current.delay_reduction
        
        # Create blended result
        result = OptimizationResult(
            cycle_length=cycle_length,
            phase_durations=phase_durations,
            phase_sequence=phase_sequence,
            offset=offset,
            optimization_method="adaptive_blended",
            performance_index=performance_index,
            delay_reduction=delay_reduction
        )
        
        return result
    
    def calculate_green_extension(
        self,
        phase_id: str,
        current_green_time: float,
        vehicles_in_dilemma_zone: int,
        seconds_since_last_call: float
    ) -> float:
        """
        Calculate green extension time for adaptive control.
        
        Args:
            phase_id: Current phase ID
            current_green_time: Current green time so far
            vehicles_in_dilemma_zone: Number of vehicles in dilemma zone
            seconds_since_last_call: Seconds since last vehicle call
        
        Returns:
            Extension time in seconds or 0 if no extension
        """
        if not self.current_result:
            return 0.0
        
        # Get max green time for this phase
        phase_max_green = 0.0
        for phase_id_result, duration in self.current_result.phase_durations.items():
            if phase_id_result == phase_id and phase_id in self.movements:
                # Assume yellow and red clearance take about 20% of phase
                phase_max_green = duration * 0.8
                break
        
        # If we've exceeded max green, don't extend
        if current_green_time >= phase_max_green:
            return 0.0
        
        # Calculate extension based on vehicles in dilemma zone
        if vehicles_in_dilemma_zone > 0:
            # Extend based on number of vehicles
            extension = min(self.max_green_extension, 
                            vehicles_in_dilemma_zone * self.min_green_extension)
            
            # Limit extension to not exceed max green
            return min(extension, phase_max_green - current_green_time)
        
        # If gap is too large, don't extend
        if seconds_since_last_call > 2.0:
            return 0.0
        
        # Default small extension
        return min(self.min_green_extension, phase_max_green - current_green_time)
    
    def analyze_traffic_patterns(self) -> Dict[str, Any]:
        """
        Analyze historical traffic patterns.
        
        Returns:
            Dictionary with traffic pattern analysis
        """
        if not self.past_demands:
            return {'status': 'No historical data available'}
        
        # Group by time of day and movement
        time_patterns = defaultdict(lambda: defaultdict(list))
        movement_patterns = defaultdict(list)
        
        for timestamp, movement_id, demand in self.past_demands:
            # Extract hour
            time_of_day = demand['time_of_day']
            hour = int(time_of_day.split(':')[0])
            
            # Add to time patterns
            time_patterns[hour][movement_id].append(demand['flow_rate'])
            
            # Add to movement patterns
            movement_patterns[movement_id].append((timestamp, demand['flow_rate']))
        
        # Calculate hourly averages
        hourly_averages = {}
        for hour, movements in time_patterns.items():
            hourly_averages[hour] = {}
            for movement_id, flow_rates in movements.items():
                hourly_averages[hour][movement_id] = sum(flow_rates) / len(flow_rates)
        
        # Identify peak hours
        total_hourly_flow = {hour: sum(avgs.values()) for hour, avgs in hourly_averages.items()}
        peak_hours = sorted(total_hourly_flow.keys(), key=lambda h: total_hourly_flow[h], reverse=True)[:3]
        
        # Identify critical movements
        movement_averages = {m_id: sum(flow for _, flow in flows) / len(flows) 
                            for m_id, flows in movement_patterns.items()}
        critical_movements = sorted(movement_averages.keys(), 
                                   key=lambda m: movement_averages[m], reverse=True)[:3]
        
        # Create analysis report
        return {
            'status': 'OK',
            'peak_hours': peak_hours,
            'peak_hour_flows': {hour: total_hourly_flow[hour] for hour in peak_hours},
            'critical_movements': critical_movements,
            'critical_movement_flows': {m_id: movement_averages[m_id] for m_id in critical_movements},
            'data_points': len(self.past_demands),
            'time_coverage_hours': (self.past_demands[-1][0] - self.past_demands[0][0]) / 3600.0
        }


class PredictiveOptimizer(SignalOptimizer):
    """
    Predictive signal optimizer that forecasts traffic patterns.
    """
    
    def __init__(
        self,
        min_cycle: float = 30.0,
        max_cycle: float = 150.0,
        lost_time_per_phase: float = 4.0,
        target_v_c_ratio: float = 0.9,
        update_interval: float = 300.0,  # 5 minutes between updates
        critical_intersection: bool = False,
        prediction_horizon: int = 3,  # Prediction steps ahead
        historical_periods: int = 7,  # Days of historical data to use
        seasonal_patterns: bool = True,  # Consider time-of-day and day-of-week
        use_ml_model: bool = False,  # Use machine learning model for prediction
        prediction_weight: float = 0.5  # Weight of prediction vs current conditions
    ):
        """
        Initialize predictive optimizer.
        
        Args:
            min_cycle: Minimum cycle length in seconds
            max_cycle: Maximum cycle length in seconds
            lost_time_per_phase: Lost time per phase in seconds
            target_v_c_ratio: Target volume-to-capacity ratio
            update_interval: Time between optimization updates in seconds
            critical_intersection: Whether this is a critical intersection
            prediction_horizon: Number of time steps to predict ahead
            historical_periods: Number of historical periods to consider
            seasonal_patterns: Whether to consider seasonal patterns
            use_ml_model: Whether to use machine learning model for prediction
            prediction_weight: Weight of prediction vs current conditions
        """
        super().__init__(
            min_cycle=min_cycle,
            max_cycle=max_cycle,
            lost_time_per_phase=lost_time_per_phase,
            target_v_c_ratio=target_v_c_ratio,
            update_interval=update_interval,
            critical_intersection=critical_intersection
        )
        self.prediction_horizon = prediction_horizon
        self.historical_periods = historical_periods
        self.seasonal_patterns = seasonal_patterns
        self.use_ml_model = use_ml_model
        self.prediction_weight = prediction_weight
        
        # Historical data storage
        self.historical_data = []  # List of (timestamp, day_of_week, time_of_day, movement_data) tuples
        self.predicted_movements = {}  # Cache for predicted movement data
        self.movements = {}  # Cache for current movements
        
        # Prediction model parameters
        self.trained_model = False
        self.model_features = []
        self.model_weights = {}
        
        logger.info("Predictive signal optimizer initialized")
    
    def optimize(
        self,
        movements: Dict[str, MovementData],
        phases: Dict[str, PhaseData],
        current_timings: Dict[str, float],
        coordination_data: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Optimize signal timings using predictive approach.
        
        Args:
            movements: Dictionary of movement ID -> MovementData
            phases: Dictionary of phase ID -> PhaseData
            current_timings: Dictionary of phase ID -> current duration
            coordination_data: Optional data for coordination with adjacent intersections
        
        Returns:
            OptimizationResult with optimized timings
        """
        # Cache movements for use in internal methods
        self.movements = movements
        
        # Check if update is needed
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval and self.current_result:
            logger.debug("Skipping optimization, using cached result")
            return self.current_result
        
        # Store current data for future predictions
        self._store_historical_data(movements)
        
        # If we have enough data, train prediction model
        if len(self.historical_data) > 24 * self.historical_periods and not self.trained_model:
            self._train_prediction_model()
        
        # Predict future traffic conditions
        predicted_movements = self._predict_future_movements(movements)
        
        # Blend current and predicted movements
        blended_movements = self._blend_movements(movements, predicted_movements)
        
        # Find critical movements
        critical_movements = self._find_critical_movements(blended_movements, phases)
        
        # Calculate optimal cycle length
        cycle_length = self._calculate_webster_cycle(critical_movements, phases)
        
        # Adjust to min/max bounds
        cycle_length = max(self.min_cycle, min(self.max_cycle, cycle_length))
        
        # Allocate green time
        phase_durations = self._allocate_green_time(cycle_length, blended_movements, phases, critical_movements)
        
        # Optimize phase sequence
        phase_sequence = self._optimize_phase_sequence(phases, blended_movements)
        
        # Calculate performance index
        performance_index = self._calculate_performance_index(cycle_length, phase_durations, blended_movements, phases)
        
        # Calculate delay reduction
        delay_reduction = self._calculate_delay_reduction(cycle_length, phase_durations, 
                                                        current_timings, blended_movements, phases)
        
        # Process coordination if provided
        offset = 0.0
        if coordination_data and coordination_data.get('enabled', False):
            offset = self._calculate_coordination_offset(cycle_length, phase_durations, 
                                                         coordination_data, blended_movements, phases)
        
        # Create optimization result
        result = OptimizationResult(
            cycle_length=cycle_length,
            phase_durations=phase_durations,
            phase_sequence=phase_sequence,
            offset=offset,
            optimization_method="predictive",
            performance_index=performance_index,
            delay_reduction=delay_reduction
        )
        
        # Update internal state
        self.last_update_time = current_time
        self.current_result = result
        self.historical_results.append(result)
        
        # Keep history limited
        if len(self.historical_results) > 100:
            self.historical_results = self.historical_results[-100:]
        
        logger.info(f"Predictive optimization: cycle={cycle_length:.1f}s, PI={performance_index:.2f}")
        return result
    
    def _store_historical_data(self, movements: Dict[str, MovementData]):
        """
        Store current traffic data for future predictions.
        
        Args:
            movements: Dictionary of movement ID -> MovementData
        """
        # Get current time information
        current_time = time.time()
        now = datetime.datetime.now()
        day_of_week = now.weekday()  # 0=Monday, 6=Sunday
        time_of_day = now.strftime("%H:%M")
        
        # Store movement data
        movement_data = {}
        for movement_id, movement in movements.items():
            # Store relevant metrics
            movement_data[movement_id] = {
                'flow_rate': movement.demand.flow_rate,
                'queue_length': movement.demand.queue_length,
                'saturation_rate': movement.demand.saturation_rate,
                'arrival_rate': movement.demand.arrival_rate,
                'density': movement.demand.density
            }
        
        # Add to historical data
        self.historical_data.append((current_time, day_of_week, time_of_day, movement_data))
        
        # Keep history limited but sufficient for predictions
        max_history = 24 * 7 * self.historical_periods  # hourly data for N weeks
        if len(self.historical_data) > max_history:
            self.historical_data = self.historical_data[-max_history:]
    
    def _train_prediction_model(self):
        """Train prediction model using historical data."""
        logger.info("Training prediction model with historical data")
        
        if self.use_ml_model:
            self._train_ml_model()
        else:
            self._train_simple_model()
        
        self.trained_model = True
    
    def _train_simple_model(self):
        """Train a simple time-series model for predictions."""
        # For each movement, build a model of flow rate by time of day and day of week
        self.model_weights = {}
        
        # Extract all movement IDs from historical data
        all_movement_ids = set()
        for _, _, _, movement_data in self.historical_data:
            all_movement_ids.update(movement_data.keys())
        
        # For each movement, calculate average flow by time and day
        for movement_id in all_movement_ids:
            # Group by day and time
            day_time_flows = {}
            
            for _, day_of_week, time_of_day, movement_data in self.historical_data:
                if movement_id in movement_data:
                    day_time_key = (day_of_week, self._round_time(time_of_day))
                    
                    if day_time_key not in day_time_flows:
                        day_time_flows[day_time_key] = []
                    
                    day_time_flows[day_time_key].append(movement_data[movement_id]['flow_rate'])
            
            # Calculate averages
            flow_averages = {}
            for day_time_key, flows in day_time_flows.items():
                flow_averages[day_time_key] = sum(flows) / len(flows)
            
            self.model_weights[movement_id] = flow_averages
    
    def _train_ml_model(self):
        """Train a machine learning model for predictions."""
        # This would normally use a proper ML model
        # For this implementation, we'll use a simplified approach
        
        # Create feature list
        self.model_features = ['hour', 'minute', 'day_of_week', 'is_weekend']
        
        # For each movement, build a regression model
        self.model_weights = {}
        
        # Extract all movement IDs from historical data
        all_movement_ids = set()
        for _, _, _, movement_data in self.historical_data:
            all_movement_ids.update(movement_data.keys())
        
        # For each movement, train a simple model
        for movement_id in all_movement_ids:
            # Prepare training data
            X = []  # Features
            y = []  # Target (flow rate)
            
            for _, day_of_week, time_of_day, movement_data in self.historical_data:
                if movement_id in movement_data:
                    # Extract features
                    hour, minute = map(int, time_of_day.split(':'))
                    is_weekend = 1 if day_of_week >= 5 else 0
                    
                    X.append([hour, minute, day_of_week, is_weekend])
                    y.append(movement_data[movement_id]['flow_rate'])
            
            # Fit linear regression model
            weights = self._fit_linear_regression(X, y)
            self.model_weights[movement_id] = weights
    
    def _fit_linear_regression(self, X: List[List[float]], y: List[float]) -> Dict[str, float]:
        """
        Fit a simple linear regression model.
        
        Args:
            X: Feature matrix
            y: Target values
        
        Returns:
            Dictionary of feature weights
        """
        # This is a simplified implementation of linear regression
        # In a real system, a proper ML library would be used
        
        # Convert to numpy arrays
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Add constant term
        X_with_const = np.column_stack([np.ones(X_array.shape[0]), X_array])
        
        # Solve for weights using normal equation
        try:
            weights = np.linalg.inv(X_with_const.T.dot(X_with_const)).dot(X_with_const.T).dot(y_array)
            
            # Convert to dictionary
            weight_dict = {'intercept': weights[0]}
            for i, feature in enumerate(self.model_features):
                weight_dict[feature] = weights[i + 1]
            
            return weight_dict
            
        except np.linalg.LinAlgError:
            logger.warning("Linear algebra error in regression, using fallback")
            return {'intercept': np.mean(y_array)}
    
    def _predict_future_movements(self, current_movements: Dict[str, MovementData]) -> Dict[str, MovementData]:
        """
        Predict future traffic conditions.
        
        Args:
            current_movements: Dictionary of movement ID -> MovementData
        
        Returns:
            Dictionary of movement ID -> predicted MovementData
        """
        # Get current time information
        now = datetime.datetime.now()
        
        # Predict for next time step (usual prediction horizon = 5 minutes)
        future_time = now + datetime.timedelta(minutes=5 * self.prediction_horizon)
        
        future_day_of_week = future_time.weekday()
        future_time_of_day = future_time.strftime("%H:%M")
        
        # Make predictions for each movement
        predicted_movements = {}
        
        for movement_id, current_movement in current_movements.items():
            if not self.trained_model:
                # If model not trained, use current values
                predicted_movements[movement_id] = current_movement
                continue
            
            # Predict flow rate
            predicted_flow = self._predict_flow_rate(movement_id, future_day_of_week, future_time_of_day)
            
            # Create predicted movement data
            # Copy the current movement data but update the demand
            predicted_demand = TrafficDemand(
                count=int(predicted_flow / 3600 * 300),  # Convert hourly flow to 5-minute count
                queue_length=current_movement.demand.queue_length,
                flow_rate=predicted_flow,
                saturation_rate=current_movement.demand.saturation_rate,
                heavy_vehicle_percentage=current_movement.demand.heavy_vehicle_percentage,
                arrival_rate=predicted_flow / 3600,  # Convert to per-second
                peak_hour_factor=current_movement.demand.peak_hour_factor,
                density=current_movement.demand.density,
                platoon_ratio=current_movement.demand.platoon_ratio
            )
            
            # Create new movement with predicted demand
            predicted_movements[movement_id] = MovementData(
                id=current_movement.id,
                signals=current_movement.signals,
                demand=predicted_demand,
                capacity=current_movement.capacity,
                saturation_flow=current_movement.saturation_flow,
                min_green=current_movement.min_green,
                max_green=current_movement.max_green,
                yellow=current_movement.yellow,
                red_clearance=current_movement.red_clearance,
                startup_lost_time=current_movement.startup_lost_time,
                lane_group_id=current_movement.lane_group_id
            )
        
        # Cache predicted movements
        self.predicted_movements = predicted_movements
        
        return predicted_movements
    
    def _predict_flow_rate(self, movement_id: str, day_of_week: int, time_of_day: str) -> float:
        """
        Predict flow rate for a specific movement and time.
        
        Args:
            movement_id: Movement identifier
            day_of_week: Day of week (0=Monday)
            time_of_day: Time of day (HH:MM format)
        
        Returns:
            Predicted flow rate
        """
        if movement_id not in self.model_weights:
            # No model for this movement, use average flow from historical data
            flows = [m[movement_id]['flow_rate'] for _, _, _, m in self.historical_data 
                    if movement_id in m]
            return sum(flows) / len(flows) if flows else 0.0
        
        if self.use_ml_model:
            # Use ML model for prediction
            # Extract features
            hour, minute = map(int, time_of_day.split(':'))
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Get model weights
            weights = self.model_weights[movement_id]
            
            # Make prediction
            prediction = weights['intercept']
            prediction += weights.get('hour', 0) * hour
            prediction += weights.get('minute', 0) * minute
            prediction += weights.get('day_of_week', 0) * day_of_week
            prediction += weights.get('is_weekend', 0) * is_weekend
            
            return max(0.0, prediction)
        else:
            # Use simple time-based model
            rounded_time = self._round_time(time_of_day)
            day_time_key = (day_of_week, rounded_time)
            
            # Get average flow for this day and time
            if day_time_key in self.model_weights[movement_id]:
                return self.model_weights[movement_id][day_time_key]
            
            # If no exact match, try nearby times
            nearby_flows = []
            
            # Check same day, nearby times
            hour, minute = map(int, rounded_time.split(':'))
            for offset in [-15, 15, -30, 30, -45, 45, -60, 60]:
                adjusted_time = datetime.datetime(2000, 1, 1, hour, minute) + datetime.timedelta(minutes=offset)
                adjusted_key = (day_of_week, adjusted_time.strftime("%H:%M"))
                
                if adjusted_key in self.model_weights[movement_id]:
                    nearby_flows.append(self.model_weights[movement_id][adjusted_key])
            
            # If we have nearby flows, use their average
            if nearby_flows:
                return sum(nearby_flows) / len(nearby_flows)
            
            # Otherwise, use average for this day
            day_flows = [flow for key, flow in self.model_weights[movement_id].items() 
                        if key[0] == day_of_week]
            
            if day_flows:
                return sum(day_flows) / len(day_flows)
            
            # Last resort: use average of all flows
            all_flows = list(self.model_weights[movement_id].values())
            return sum(all_flows) / len(all_flows) if all_flows else 0.0
    
    def _round_time(self, time_of_day: str) -> str:
        """
        Round time to nearest 15 minutes.
        
        Args:
            time_of_day: Time in HH:MM format
        
        Returns:
            Rounded time in HH:MM format
        """
        hour, minute = map(int, time_of_day.split(':'))
        
        # Round to nearest 15 minutes
        minute = ((minute + 7) // 15) * 15
        
        # Handle overflow
        if minute >= 60:
            minute = 0
            hour = (hour + 1) % 24
        
        return f"{hour:02d}:{minute:02d}"
    
    def _blend_movements(
        self,
        current: Dict[str, MovementData],
        predicted: Dict[str, MovementData]
    ) -> Dict[str, MovementData]:
        """
        Blend current and predicted movement data.
        
        Args:
            current: Dictionary of movement ID -> current MovementData
            predicted: Dictionary of movement ID -> predicted MovementData
        
        Returns:
            Dictionary of movement ID -> blended MovementData
        """
        blended_movements = {}
        
        for movement_id, current_movement in current.items():
            if movement_id not in predicted:
                blended_movements[movement_id] = current_movement
                continue
            
            predicted_movement = predicted[movement_id]
            
            # Blend flow rates and other demand metrics
            blended_flow = (
                (1 - self.prediction_weight) * current_movement.demand.flow_rate +
                self.prediction_weight * predicted_movement.demand.flow_rate
            )
            
            blended_queue = (
                (1 - self.prediction_weight) * current_movement.demand.queue_length +
                self.prediction_weight * predicted_movement.demand.queue_length
            )
            
            # Create blended demand
            blended_demand = TrafficDemand(
                count=int(blended_flow / 3600 * 300),  # Convert hourly flow to 5-minute count
                queue_length=blended_queue,
                flow_rate=blended_flow,
                saturation_rate=current_movement.demand.saturation_rate,
                heavy_vehicle_percentage=current_movement.demand.heavy_vehicle_percentage,
                arrival_rate=blended_flow / 3600,  # Convert to per-second
                peak_hour_factor=current_movement.demand.peak_hour_factor,
                density=max(current_movement.demand.density, predicted_movement.demand.density),
                platoon_ratio=current_movement.demand.platoon_ratio
            )
            
            # Create blended movement
            blended_movements[movement_id] = MovementData(
                id=current_movement.id,
                signals=current_movement.signals,
                demand=blended_demand,
                capacity=current_movement.capacity,
                saturation_flow=current_movement.saturation_flow,
                min_green=current_movement.min_green,
                max_green=current_movement.max_green,
                yellow=current_movement.yellow,
                red_clearance=current_movement.red_clearance,
                startup_lost_time=current_movement.startup_lost_time,
                lane_group_id=current_movement.lane_group_id
            )
        
        return blended_movements
    
    def _optimize_phase_sequence(
        self,
        phases: Dict[str, PhaseData],
        movements: Dict[str, MovementData]
    ) -> List[str]:
        """
        Optimize phase sequence based on predicted demand.
        
        Args:
            phases: Dictionary of phase ID -> PhaseData
            movements: Dictionary of movement ID -> MovementData
        
        Returns:
            Optimized list of phase IDs
        """
        # Default sequence
        default_sequence = list(phases.keys())
        
        # In predictive control, we might want to adapt the sequence
        # based on predicted demand patterns
        
        # For simplicity, just use the default sequence
        # A real implementation would use a more sophisticated algorithm
        # considering predicted arrival patterns and coordination
        
        return default_sequence
    
    def _calculate_coordination_offset(
        self,
        cycle_length: float,
        phase_durations: Dict[str, float],
        coordination_data: Dict[str, Any],
        movements: Dict[str, MovementData],
        phases: Dict[str, PhaseData]
    ) -> float:
        """
        Calculate signal offset for coordination using prediction.
        
        Args:
            cycle_length: Optimized cycle length in seconds
            phase_durations: Dictionary of phase ID -> optimized duration
            coordination_data: Coordination data
            movements: Dictionary of movement ID -> MovementData
            phases: Dictionary of phase ID -> PhaseData
        
        Returns:
            Offset value in seconds
        """
        # Get coordination parameters
        travel_time = coordination_data.get('travel_time', 0.0)
        reference_offset = coordination_data.get('reference_offset', 0.0)
        direction = coordination_data.get('direction', 'two_way')
        
        # Identify coordinated phase
        coordinated_phase = next((phase_id for phase_id, phase in phases.items() 
                                if phase.is_coordinated), None)
        
        if not coordinated_phase:
            return 0.0
        
        # Base offset calculation (similar to fixed-time)
        if direction == 'one_way':
            # One-way coordination: offset = travel time
            base_offset = travel_time % cycle_length
        else:
            # Two-way coordination: offset = half cycle
            base_offset = (cycle_length / 2) % cycle_length
        
        # Adjust for reference offset
        base_offset = (base_offset + reference_offset) % cycle_length
        
        # Predictive enhancement: adjust based on predicted platoon arrival
        platoon_adjustment = 0.0
        
        # Get platoon arrival time prediction if available
        if coordination_data.get('platoon_info'):
            platoon_data = coordination_data['platoon_info']
            predicted_arrival = platoon_data.get('predicted_arrival', 0.0)
            
            if predicted_arrival > 0:
                # Calculate when the coordinated phase will be green
                phase_start_time = 0.0
                for phase_id in self.current_result.phase_sequence:
                    if phase_id == coordinated_phase:
                        break
                    phase_start_time += phase_durations.get(phase_id, 0.0)
                
                # Adjust offset to align green window with predicted platoon arrival
                platoon_adjustment = (predicted_arrival - phase_start_time) % cycle_length
        
        # Calculate final offset
        offset = (base_offset + platoon_adjustment) % cycle_length
        
        return offset
    
    def evaluate_predictions(self) -> Dict[str, Any]:
        """
        Evaluate prediction accuracy.
        
        Returns:
            Dictionary with prediction evaluation
        """
        if not self.predicted_movements or not self.trained_model:
            return {'status': 'No prediction data available'}
        
        # For each movement, compare last prediction with actual value
        accuracy_metrics = {}
        total_error = 0.0
        total_movements = 0
        
        for movement_id, current_movement in self.movements.items():
            if movement_id not in self.predicted_movements:
                continue
            
            predicted_movement = self.predicted_movements[movement_id]
            
            # Calculate error
            current_flow = current_movement.demand.flow_rate
            predicted_flow = predicted_movement.demand.flow_rate
            
            if current_flow > 0:
                relative_error = abs(predicted_flow - current_flow) / current_flow
                accuracy = max(0.0, 1.0 - relative_error)
            else:
                accuracy = 1.0 if predicted_flow == 0.0 else 0.0
            
            accuracy_metrics[movement_id] = {
                'actual_flow': current_flow,
                'predicted_flow': predicted_flow,
                'accuracy': accuracy
            }
            
            total_error += abs(predicted_flow - current_flow)
            total_movements += 1
        
        # Calculate overall metrics
        overall_rmse = math.sqrt(total_error**2 / max(1, total_movements))
        overall_accuracy = sum(m['accuracy'] for m in accuracy_metrics.values()) / max(1, len(accuracy_metrics))
        
        return {
            'status': 'OK',
            'overall_accuracy': overall_accuracy,
            'overall_rmse': overall_rmse,
            'movement_metrics': accuracy_metrics,
            'model_type': 'machine_learning' if self.use_ml_model else 'time_series',
            'historical_data_points': len(self.historical_data)
        }


class CoordinationOptimizer(SignalOptimizer):
    """
    Signal optimizer that focuses on coordinating multiple intersections.
    """
    
    def __init__(
        self,
        min_cycle: float = 60.0,
        max_cycle: float = 180.0,
        lost_time_per_phase: float = 4.0,
        target_v_c_ratio: float = 0.9,
        update_interval: float = 900.0,  # 15 minutes between updates
        critical_intersection: bool = True,
        network_cycle_length: Optional[float] = None,
        coordination_speed: float = 40.0,  # km/h
        bandwidth_efficiency: float = 0.4,
        arterial_priority: float = 0.7,
        coordination_type: str = 'two_way'
    ):
        """
        Initialize coordination optimizer.
        
        Args:
            min_cycle: Minimum cycle length in seconds
            max_cycle: Maximum cycle length in seconds
            lost_time_per_phase: Lost time per phase in seconds
            target_v_c_ratio: Target volume-to-capacity ratio
            update_interval: Time between optimization updates in seconds
            critical_intersection: Whether this is a critical intersection
            network_cycle_length: Network-wide common cycle length or None for local optimization
            coordination_speed: Coordination speed in km/h
            bandwidth_efficiency: Target bandwidth efficiency (0-1)
            arterial_priority: Priority weight for arterial direction (0-1)
            coordination_type: Type of coordination ('one_way' or 'two_way')
        """
        super().__init__(
            min_cycle=min_cycle,
            max_cycle=max_cycle,
            lost_time_per_phase=lost_time_per_phase,
            target_v_c_ratio=target_v_c_ratio,
            update_interval=update_interval,
            critical_intersection=critical_intersection
        )
        self.network_cycle_length = network_cycle_length
        self.coordination_speed = coordination_speed
        self.bandwidth_efficiency = bandwidth_efficiency
        self.arterial_priority = arterial_priority
        self.coordination_type = coordination_type
        
        # Network information
        self.upstream_intersections = []
        self.downstream_intersections = []
        self.distance_to_upstream = {}
        self.distance_to_downstream = {}
        
        # Coordination parameters
        self.coordinated_phases = set()
        self.fixed_offsets = {}
        self.movements = {}  # Cache for movements
        
        logger.info("Coordination optimizer initialized")
    
    def set_network_information(
        self,
        upstream: List[str],
        downstream: List[str],
        distances_upstream: Dict[str, float],
        distances_downstream: Dict[str, float]
    ):
        """
        Set network information for coordination.
        
        Args:
            upstream: List of upstream intersection IDs
            downstream: List of downstream intersection IDs
            distances_upstream: Dictionary of upstream ID -> distance (meters)
            distances_downstream: Dictionary of downstream ID -> distance (meters)
        """
        self.upstream_intersections = upstream
        self.downstream_intersections = downstream
        self.distance_to_upstream = distances_upstream
        self.distance_to_downstream = distances_downstream
        
        logger.info(f"Network information set: {len(upstream)} upstream, {len(downstream)} downstream intersections")
    
    def set_coordinated_phases(self, phase_ids: List[str], fixed_offsets: Optional[Dict[str, float]] = None):
        """
        Set phases that should be coordinated.
        
        Args:
            phase_ids: List of phase IDs to coordinate
            fixed_offsets: Optional dictionary of phase ID -> fixed offset
        """
        self.coordinated_phases = set(phase_ids)
        self.fixed_offsets = fixed_offsets or {}
        
        logger.info(f"Coordinated phases set: {', '.join(phase_ids)}")
    
    def optimize(
        self,
        movements: Dict[str, MovementData],
        phases: Dict[str, PhaseData],
        current_timings: Dict[str, float],
        coordination_data: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Optimize signal timings for coordination.
        
        Args:
            movements: Dictionary of movement ID -> MovementData
            phases: Dictionary of phase ID -> PhaseData
            current_timings: Dictionary of phase ID -> current duration
            coordination_data: Optional data for coordination with adjacent intersections
        
        Returns:
            OptimizationResult with optimized timings
        """
        # Cache movements for use in internal methods
        self.movements = movements
        
        # Check if update is needed
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval and self.current_result:
            logger.debug("Skipping optimization, using cached result")
            return self.current_result
        
        # Determine cycle length
        if self.network_cycle_length:
            # Use network-wide cycle length if provided
            cycle_length = self.network_cycle_length
        else:
            # Calculate optimal cycle length locally
            critical_movements = self._find_critical_movements(movements, phases)
            cycle_length = self._calculate_webster_cycle(critical_movements, phases)
            
            # Adjust to min/max bounds
            cycle_length = max(self.min_cycle, min(self.max_cycle, cycle_length))
        
        # Allocate green time to phases
        critical_movements = self._find_critical_movements(movements, phases)
        phase_durations = self._allocate_green_time(cycle_length, movements, phases, critical_movements)
        
        # Optimize phase sequence for coordination
        phase_sequence = self._optimize_for_coordination(phases)
        
        # Calculate timing offsets for coordination
        offset = self._calculate_coordination_offset(coordination_data)
        
        # Calculate performance index
        performance_index = self._calculate_performance_index(cycle_length, phase_durations, movements, phases)
        
        # Calculate delay reduction
        delay_reduction = self._calculate_delay_reduction(cycle_length, phase_durations, 
                                                        current_timings, movements, phases)
        
        # Create optimization result
        result = OptimizationResult(
            cycle_length=cycle_length,
            phase_durations=phase_durations,
            phase_sequence=phase_sequence,
            offset=offset,
            optimization_method="coordination",
            performance_index=performance_index,
            delay_reduction=delay_reduction
        )
        
        # Update internal state
        self.last_update_time = current_time
        self.current_result = result
        self.historical_results.append(result)
        
        # Keep history limited
        if len(self.historical_results) > 100:
            self.historical_results = self.historical_results[-100:]
        
        logger.info(f"Coordination optimization: cycle={cycle_length:.1f}s, offset={offset:.1f}s")
        return result
    
    def _optimize_for_coordination(self, phases: Dict[str, PhaseData]) -> List[str]:
        """
        Optimize phase sequence for signal coordination.
        
        Args:
            phases: Dictionary of phase ID -> PhaseData
        
        Returns:
            Optimized phase sequence
        """
        # For coordination, we typically want coordinated phases to appear
        # at a specific point in the sequence to facilitate progression
        
        # First, identify coordinated phases
        coord_phases = [phase_id for phase_id in phases if phase_id in self.coordinated_phases]
        non_coord_phases = [phase_id for phase_id in phases if phase_id not in self.coordinated_phases]
        
        # For two-way coordination, put coordinated phases at the beginning and end
        if self.coordination_type == 'two_way' and len(coord_phases) >= 2:
            # Split coordinated phases
            first_coord = coord_phases[0]
            last_coord = coord_phases[-1]
            middle_coord = coord_phases[1:-1] if len(coord_phases) > 2 else []
            
            # Build sequence
            sequence = [first_coord] + non_coord_phases + middle_coord + [last_coord]
            
        # For one-way coordination, put all coordinated phases together
        else:
            sequence = coord_phases + non_coord_phases
        
        return sequence
    
    def _calculate_coordination_offset(self, coordination_data: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate offset for signal coordination.
        
        Args:
            coordination_data: Optional coordination data
        
        Returns:
            Offset in seconds
        """
        if not coordination_data:
            return 0.0
        
        # Get cycle length
        cycle_length = coordination_data.get('cycle_length', self.network_cycle_length or 120.0)
        
        # If we have fixed offsets for coordinated phases, use those
        if self.coordinated_phases and self.fixed_offsets:
            first_coord_phase = next(iter(self.coordinated_phases))
            if first_coord_phase in self.fixed_offsets:
                return self.fixed_offsets[first_coord_phase]
        
        # Otherwise, calculate based on distance and speed
        travel_time = 0.0
        
        # Use reference intersection information
        reference_id = coordination_data.get('reference_intersection')
        reference_offset = coordination_data.get('reference_offset', 0.0)
        
        if reference_id:
            # Calculate travel time based on distance and speed
            if reference_id in self.distance_to_upstream:
                distance = self.distance_to_upstream[reference_id]  # meters
                speed = self.coordination_speed * 1000 / 3600  # Convert km/h to m/s
                travel_time = distance / speed  # seconds
            elif reference_id in self.distance_to_downstream:
                distance = self.distance_to_downstream[reference_id]  # meters
                speed = self.coordination_speed * 1000 / 3600  # Convert km/h to m/s
                travel_time = distance / speed  # seconds
        
        # For two-way coordination, we typically use half-cycle offset
        if self.coordination_type == 'two_way':
            offset = (cycle_length / 2) % cycle_length
        else:
            # For one-way, offset based on travel time
            offset = travel_time % cycle_length
        
        # Adjust by reference offset
        offset = (offset + reference_offset) % cycle_length
        
        return offset
    
    def _calculate_delay_reduction(
        self,
        cycle_length: float,
        phase_durations: Dict[str, float],
        current_timings: Dict[str, float],
        movements: Dict[str, MovementData],
        phases: Dict[str, PhaseData]
    ) -> float:
        """
        Calculate delay reduction with coordination.
        
        Args:
            cycle_length: Optimized cycle length in seconds
            phase_durations: Dictionary of phase ID -> optimized duration
            current_timings: Dictionary of phase ID -> current duration
            movements: Dictionary of movement ID -> MovementData
            phases: Dictionary of phase ID -> PhaseData
        
        Returns:
            Delay reduction percentage (0-100)
        """
        # For coordination, we need to consider not just individual intersection delay
        # but also the progression quality through the corridor
        
        # First, calculate standard delay reduction
        current_cycle = sum(current_timings.values())
        
        # Calculate delay with current timings
        current_pi = self._calculate_performance_index(current_cycle, current_timings, movements, phases)
        
        # Calculate delay with optimized timings
        optimized_pi = self._calculate_performance_index(cycle_length, phase_durations, movements, phases)
        
        # Calculate reduction percentage
        if current_pi > 0:
            base_reduction = max(0, (current_pi - optimized_pi) / current_pi * 100)
        else:
            base_reduction = 0.0
        
        # Add coordination benefit (simplified calculation)
        # In reality, this would involve a more sophisticated progression quality measure
        coordination_benefit = 0.0
        
        if self.coordinated_phases:
            # Calculate green time for coordinated phases
            coord_green = sum(
                phase_durations.get(phase_id, 0) * 0.8  # Assuming 80% of phase duration is effective green
                for phase_id in self.coordinated_phases
            )
            
            # Calculate bandwidth efficiency (simplified)
            bandwidth = coord_green / cycle_length
            
            # Estimate coordination benefit based on bandwidth
            if bandwidth > 0:
                coordination_benefit = min(20.0, bandwidth * 40.0)  # Max 20% additional benefit
        
        # Combine reductions
        total_reduction = min(100.0, base_reduction + coordination_benefit)
        
        return total_reduction
    
    def estimate_progression_quality(self, coordination_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Estimate progression quality through the coordinated corridor.
        
        Args:
            coordination_data: Optional coordination data
        
        Returns:
            Dictionary with progression quality metrics
        """
        if not self.current_result or not coordination_data:
            return {'status': 'No coordination data available'}
        
        # Get current optimization result
        cycle_length = self.current_result.cycle_length
        offset = self.current_result.offset
        
        # Calculate progression metrics
        cycle_match = 1.0
        if self.network_cycle_length and abs(cycle_length - self.network_cycle_length) > 1.0:
            cycle_match = self.network_cycle_length / cycle_length if cycle_length > self.network_cycle_length else cycle_length / self.network_cycle_length
        
        # Calculate bandwidth (simplified)
        bandwidth = 0.0
        
        if self.coordinated_phases:
            # Get green time for coordinated phases
            coord_green = sum(
                self.current_result.phase_durations.get(phase_id, 0) * 0.8  # Assuming 80% of phase is effective green
                for phase_id in self.coordinated_phases
            )
            
            bandwidth = coord_green / cycle_length
        
        # Calculate progression quality metrics
        return {
            'status': 'OK',
            'cycle_length': cycle_length,
            'network_cycle': self.network_cycle_length or cycle_length,
            'cycle_match': cycle_match,
            'offset': offset,
            'bandwidth': bandwidth,
            'bandwidth_efficiency': bandwidth / max(0.01, self.bandwidth_efficiency),
            'progression_speed': self.coordination_speed,
            'coordination_type': self.coordination_type,
            'coordinated_phases': list(self.coordinated_phases),
            'upstream_intersections': self.upstream_intersections,
            'downstream_intersections': self.downstream_intersections
        }
