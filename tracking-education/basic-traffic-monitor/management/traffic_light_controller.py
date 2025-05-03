"""
Core traffic light controller module for managing signal states and timings.
Implements traffic signal management, state transitions, and timing controls.
"""

import time
import logging
import threading
import json
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
import datetime
import numpy as np

# Configure logger for this module
logger = logging.getLogger(__name__)


class SignalState(Enum):
    """Traffic signal state enumeration."""
    RED = 'red'
    YELLOW = 'yellow'
    GREEN = 'green'
    FLASHING_RED = 'flashing_red'
    FLASHING_YELLOW = 'flashing_yellow'
    OFF = 'off'


@dataclass
class SignalTiming:
    """Configuration for traffic signal timings."""
    min_green_time: float       # Minimum green time in seconds
    max_green_time: float       # Maximum green time in seconds
    yellow_time: float          # Yellow time in seconds
    all_red_time: float         # All-red clearance time in seconds
    pedestrian_time: float      # Additional time for pedestrian crossings
    extension_time: float       # Extension time per vehicle detected
    gap_time: float             # Maximum gap time to extend green
    
    def validate(self) -> None:
        """Validate timing values are within acceptable ranges."""
        if self.min_green_time < 5.0:
            raise ValueError(f"Minimum green time too short: {self.min_green_time}s, should be >= 5s")
        
        if self.max_green_time > 180.0:
            raise ValueError(f"Maximum green time too long: {self.max_green_time}s, should be <= 180s")
        
        if self.min_green_time > self.max_green_time:
            raise ValueError(f"Minimum green time ({self.min_green_time}s) cannot exceed maximum green time ({self.max_green_time}s)")
        
        if self.yellow_time < 3.0 or self.yellow_time > 6.0:
            raise ValueError(f"Yellow time out of range: {self.yellow_time}s, should be 3-6s")
        
        if self.all_red_time < 1.0:
            raise ValueError(f"All-red time too short: {self.all_red_time}s, should be >= 1s")
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'SignalTiming':
        """Create a SignalTiming from a dictionary."""
        timing = cls(
            min_green_time=data.get('min_green_time', 10.0),
            max_green_time=data.get('max_green_time', 60.0),
            yellow_time=data.get('yellow_time', 4.0),
            all_red_time=data.get('all_red_time', 2.0),
            pedestrian_time=data.get('pedestrian_time', 15.0),
            extension_time=data.get('extension_time', 2.0),
            gap_time=data.get('gap_time', 3.0)
        )
        timing.validate()
        return timing
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'min_green_time': self.min_green_time,
            'max_green_time': self.max_green_time,
            'yellow_time': self.yellow_time,
            'all_red_time': self.all_red_time,
            'pedestrian_time': self.pedestrian_time,
            'extension_time': self.extension_time,
            'gap_time': self.gap_time
        }


@dataclass
class SignalPlan:
    """Plan for signal phase timings."""
    cycle_length: float                     # Total cycle length in seconds
    phases: List[Dict[str, Any]]            # List of phase configurations
    offset: float = 0.0                     # Offset from coordination reference in seconds
    coordination_mode: bool = False         # Whether this plan is part of coordination
    plan_id: str = "default"                # Identifier for this plan
    description: str = ""                   # Description of the plan
    active_days: List[int] = None           # Days of week this plan is active (0=Monday)
    active_periods: List[Tuple[str, str]] = None  # Time periods this plan is active as (start, end)
    vehicle_actuated: bool = False          # Whether phases can be extended by vehicle actuation
    adaptive: bool = False                  # Whether plan can be adaptively modified
    
    def __post_init__(self):
        """Initialize after creation."""
        if self.active_days is None:
            self.active_days = list(range(7))  # Default to all days
        
        if self.active_periods is None:
            self.active_periods = [("00:00", "23:59")]  # Default to all day
    
    def validate(self) -> None:
        """Validate the signal plan."""
        if not self.phases:
            raise ValueError("Signal plan must have at least one phase")
        
        # Calculate sum of minimum phase times
        min_cycle_time = sum(phase.get('min_time', 0) for phase in self.phases)
        
        if self.cycle_length < min_cycle_time:
            raise ValueError(f"Cycle length ({self.cycle_length}s) less than sum of minimum phase times ({min_cycle_time}s)")
        
        # Validate each phase
        for i, phase in enumerate(self.phases):
            if 'signals' not in phase:
                raise ValueError(f"Phase {i} missing 'signals' configuration")
            
            if not isinstance(phase['signals'], dict):
                raise ValueError(f"Phase {i} 'signals' must be a dictionary mapping signal IDs to states")
    
    def is_active_now(self) -> bool:
        """Check if this plan should be active at the current time."""
        now = datetime.datetime.now()
        
        # Check day of week (0=Monday in our system)
        weekday = (now.weekday()) % 7
        if weekday not in self.active_days:
            return False
        
        # Check time of day
        current_time_str = now.strftime("%H:%M")
        
        for start_time, end_time in self.active_periods:
            if start_time <= current_time_str <= end_time:
                return True
            
            # Handle overnight periods
            if end_time < start_time and (current_time_str >= start_time or current_time_str <= end_time):
                return True
        
        return False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalPlan':
        """Create a SignalPlan from a dictionary."""
        active_periods = None
        if 'active_periods' in data:
            active_periods = [(p['start'], p['end']) for p in data['active_periods']]
        
        plan = cls(
            cycle_length=data['cycle_length'],
            phases=data['phases'],
            offset=data.get('offset', 0.0),
            coordination_mode=data.get('coordination_mode', False),
            plan_id=data.get('plan_id', 'default'),
            description=data.get('description', ''),
            active_days=data.get('active_days'),
            active_periods=active_periods,
            vehicle_actuated=data.get('vehicle_actuated', False),
            adaptive=data.get('adaptive', False)
        )
        plan.validate()
        return plan
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        active_periods_dict = None
        if self.active_periods:
            active_periods_dict = [{'start': start, 'end': end} for start, end in self.active_periods]
            
        return {
            'cycle_length': self.cycle_length,
            'phases': self.phases,
            'offset': self.offset,
            'coordination_mode': self.coordination_mode,
            'plan_id': self.plan_id,
            'description': self.description,
            'active_days': self.active_days,
            'active_periods': active_periods_dict,
            'vehicle_actuated': self.vehicle_actuated,
            'adaptive': self.adaptive
        }


class TrafficSignal:
    """
    Class representing a single traffic signal.
    """
    
    def __init__(self, signal_id: str, initial_state: SignalState = SignalState.RED):
        """
        Initialize traffic signal.
        
        Args:
            signal_id: Unique identifier for this signal
            initial_state: Initial signal state
        """
        self.signal_id = signal_id
        self.state = initial_state
        self.previous_state = initial_state
        self.state_start_time = time.time()
        self.state_duration = 0.0
        self.pedestrian_call = False
        self.vehicle_calls = 0
        
        # Performance metrics
        self.state_history = []  # List of (timestamp, state, duration) tuples
        self.total_green_time = 0.0
        self.total_red_time = 0.0
        self.total_yellow_time = 0.0
        
        logger.debug(f"Traffic signal {signal_id} initialized with state {initial_state.value}")
    
    def set_state(self, new_state: SignalState) -> None:
        """
        Change the signal state.
        
        Args:
            new_state: New signal state
        """
        if new_state == self.state:
            return
        
        current_time = time.time()
        state_duration = current_time - self.state_start_time
        
        # Update performance metrics
        if self.state == SignalState.GREEN:
            self.total_green_time += state_duration
        elif self.state == SignalState.RED:
            self.total_red_time += state_duration
        elif self.state == SignalState.YELLOW:
            self.total_yellow_time += state_duration
        
        # Record history
        self.state_history.append((current_time, self.state, state_duration))
        
        # Keep history limited to reasonable size
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]
        
        # Update state
        self.previous_state = self.state
        self.state = new_state
        self.state_start_time = current_time
        self.state_duration = 0.0
        
        logger.debug(f"Traffic signal {self.signal_id} changed from {self.previous_state.value} to {new_state.value}")
    
    def update(self) -> None:
        """Update signal timing information."""
        current_time = time.time()
        self.state_duration = current_time - self.state_start_time
    
    def register_vehicle_call(self) -> None:
        """Register a vehicle call (detection) for this signal."""
        self.vehicle_calls += 1
    
    def register_pedestrian_call(self) -> None:
        """Register a pedestrian call for this signal."""
        self.pedestrian_call = True
    
    def clear_calls(self) -> None:
        """Clear all calls after they've been serviced."""
        self.vehicle_calls = 0
        self.pedestrian_call = False
    
    def get_efficiency_metrics(self) -> Dict[str, float]:
        """
        Get signal efficiency metrics.
        
        Returns:
            Dictionary of metrics
        """
        total_time = self.total_green_time + self.total_red_time + self.total_yellow_time
        
        if total_time == 0:
            # Avoid division by zero
            return {
                'green_ratio': 0.0,
                'red_ratio': 0.0,
                'yellow_ratio': 0.0,
                'avg_green_duration': 0.0,
                'avg_red_duration': 0.0
            }
        
        # Count state transitions
        green_periods = 0
        red_periods = 0
        
        for _, state, _ in self.state_history:
            if state == SignalState.GREEN:
                green_periods += 1
            elif state == SignalState.RED:
                red_periods += 1
        
        # Calculate metrics
        return {
            'green_ratio': self.total_green_time / total_time,
            'red_ratio': self.total_red_time / total_time,
            'yellow_ratio': self.total_yellow_time / total_time,
            'avg_green_duration': self.total_green_time / max(1, green_periods),
            'avg_red_duration': self.total_red_time / max(1, red_periods)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert signal to dictionary representation.
        
        Returns:
            Dictionary with signal data
        """
        return {
            'signal_id': self.signal_id,
            'state': self.state.value,
            'previous_state': self.previous_state.value,
            'state_duration': self.state_duration,
            'pedestrian_call': self.pedestrian_call,
            'vehicle_calls': self.vehicle_calls,
            'metrics': self.get_efficiency_metrics()
        }


class TrafficLightController:
    """
    Controller for managing traffic lights at an intersection.
    """
    
    def __init__(
        self, 
        intersection_id: str, 
        signals: Dict[str, SignalState],
        timing_params: Dict[str, float],
        plans: List[Dict[str, Any]],
        optimization_mode: str = "fixed",
        coordination_enabled: bool = False,
        emergency_mode_enabled: bool = True,
        fail_safe_mode: str = "flashing_red",
        startup_sequence: str = "all_red"
    ):
        """
        Initialize traffic light controller.
        
        Args:
            intersection_id: Unique identifier for the intersection
            signals: Dictionary of signal_id -> initial_state
            timing_params: Parameters for signal timing
            plans: List of signal timing plans
            optimization_mode: Optimization strategy ('fixed', 'adaptive', 'predictive')
            coordination_enabled: Whether coordination with other intersections is enabled
            emergency_mode_enabled: Whether emergency vehicle preemption is enabled
            fail_safe_mode: Fail-safe mode in case of system failure
            startup_sequence: Sequence to use when starting up
        """
        self.intersection_id = intersection_id
        self.coordination_enabled = coordination_enabled
        self.emergency_mode_enabled = emergency_mode_enabled
        self.optimization_mode = optimization_mode
        self.fail_safe_mode = fail_safe_mode
        self.startup_sequence = startup_sequence
        
        # Initialize timing parameters
        self.timing = SignalTiming.from_dict(timing_params)
        
        # Initialize signals
        self.signals = {}
        for signal_id, initial_state in signals.items():
            self.signals[signal_id] = TrafficSignal(signal_id, initial_state)
        
        # Initialize signal plans
        self.plans = {}
        for plan_data in plans:
            plan = SignalPlan.from_dict(plan_data)
            self.plans[plan.plan_id] = plan
        
        # Ensure we have at least a default plan
        if not self.plans:
            logger.warning(f"No signal plans provided for {intersection_id}, creating default plan")
            self._create_default_plan()
        
        # Current state
        self.active_plan_id = next(iter(self.plans.keys()))  # First plan as default
        self.current_phase_index = 0
        self.phase_start_time = time.time()
        self.phase_duration = 0.0
        self.cycle_start_time = time.time()
        self.running = False
        self.emergency_mode_active = False
        self.coordination_offset_active = False
        self.manual_control_active = False
        
        # Performance metrics
        self.total_cycles = 0
        self.total_vehicles = 0
        self.avg_wait_time = 0.0
        self.congestion_level = 0.0
        
        # Control thread
        self.control_thread = None
        self.stop_event = threading.Event()
        
        # Flow data
        self.current_flow_data = None
        self.historical_flow_data = []
        
        logger.info(f"Traffic light controller initialized for intersection {intersection_id}")
    
    def _create_default_plan(self):
        """Create a default signal plan if none was provided."""
        # Group signals by direction (assuming naming convention like "north_main", "south_left", etc.)
        directions = {}
        for signal_id in self.signals:
            prefix = signal_id.split('_')[0]
            if prefix in ['north', 'south', 'east', 'west']:
                if prefix not in directions:
                    directions[prefix] = []
                directions[prefix].append(signal_id)
        
        # Create a simple two-phase plan
        phases = []
        
        # North-South phase
        if 'north' in directions or 'south' in directions:
            ns_signals = {}
            for direction in ['north', 'south']:
                for signal_id in directions.get(direction, []):
                    ns_signals[signal_id] = SignalState.GREEN.value
            
            for direction in ['east', 'west']:
                for signal_id in directions.get(direction, []):
                    ns_signals[signal_id] = SignalState.RED.value
            
            phases.append({
                'name': 'North-South',
                'min_time': 20.0,
                'max_time': 60.0,
                'signals': ns_signals
            })
        
        # East-West phase
        if 'east' in directions or 'west' in directions:
            ew_signals = {}
            for direction in ['east', 'west']:
                for signal_id in directions.get(direction, []):
                    ew_signals[signal_id] = SignalState.GREEN.value
            
            for direction in ['north', 'south']:
                for signal_id in directions.get(direction, []):
                    ew_signals[signal_id] = SignalState.RED.value
            
            phases.append({
                'name': 'East-West',
                'min_time': 20.0,
                'max_time': 60.0,
                'signals': ew_signals
            })
        
        # If we couldn't create a direction-based plan, create a all-red/all-green alternating plan
        if not phases:
            all_signal_ids = list(self.signals.keys())
            half = len(all_signal_ids) // 2
            
            phase1_signals = {}
            phase2_signals = {}
            
            for i, signal_id in enumerate(all_signal_ids):
                if i < half:
                    phase1_signals[signal_id] = SignalState.GREEN.value
                    phase2_signals[signal_id] = SignalState.RED.value
                else:
                    phase1_signals[signal_id] = SignalState.RED.value
                    phase2_signals[signal_id] = SignalState.GREEN.value
            
            phases = [
                {
                    'name': 'Phase 1',
                    'min_time': 20.0,
                    'max_time': 60.0,
                    'signals': phase1_signals
                },
                {
                    'name': 'Phase 2',
                    'min_time': 20.0,
                    'max_time': 60.0,
                    'signals': phase2_signals
                }
            ]
        
        # Create the default plan
        default_plan = SignalPlan(
            cycle_length=sum(phase['min_time'] for phase in phases) + len(phases) * self.timing.yellow_time,
            phases=phases,
            plan_id='default',
            description='Default generated plan'
        )
        
        self.plans['default'] = default_plan
    
    def start(self) -> bool:
        """
        Start the traffic light control.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            logger.warning(f"Traffic light controller for {self.intersection_id} is already running")
            return True
        
        # Reset state
        self.stop_event.clear()
        self.cycle_start_time = time.time()
        self.phase_start_time = time.time()
        self.current_phase_index = 0
        
        # Execute startup sequence
        self._execute_startup_sequence()
        
        # Start control thread
        self.control_thread = threading.Thread(
            target=self._control_loop,
            daemon=True
        )
        self.control_thread.start()
        
        self.running = True
        logger.info(f"Traffic light controller started for intersection {self.intersection_id}")
        
        return True
    
    def stop(self):
        """Stop the traffic light control."""
        if not self.running:
            return
        
        logger.info(f"Stopping traffic light controller for intersection {self.intersection_id}")
        
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.control_thread is not None and self.control_thread.is_alive():
            self.control_thread.join(timeout=3.0)
        
        # Set all signals to fail-safe mode
        self._set_fail_safe_mode()
        
        self.running = False
        logger.info(f"Traffic light controller stopped for intersection {self.intersection_id}")
    
    def _execute_startup_sequence(self):
        """Execute the startup sequence for signals."""
        logger.info(f"Executing {self.startup_sequence} startup sequence")
        
        if self.startup_sequence == "all_red":
            # Set all signals to red
            for signal in self.signals.values():
                signal.set_state(SignalState.RED)
            
            # Wait for all-red clearance time
            time.sleep(self.timing.all_red_time)
            
        elif self.startup_sequence == "flashing_yellow":
            # Set all signals to flashing yellow
            for signal in self.signals.values():
                signal.set_state(SignalState.FLASHING_YELLOW)
            
            # Flash for a few seconds
            for _ in range(5):
                time.sleep(0.5)
                for signal in self.signals.values():
                    if signal.state == SignalState.FLASHING_YELLOW:
                        signal.set_state(SignalState.OFF)
                    else:
                        signal.set_state(SignalState.FLASHING_YELLOW)
            
            # Set all to red
            for signal in self.signals.values():
                signal.set_state(SignalState.RED)
            
            # Wait for all-red clearance time
            time.sleep(self.timing.all_red_time)
            
        else:  # Default to all-red
            for signal in self.signals.values():
                signal.set_state(SignalState.RED)
            time.sleep(self.timing.all_red_time)
    
    def _set_fail_safe_mode(self):
        """Set all signals to fail-safe mode."""
        if self.fail_safe_mode == "flashing_red":
            for signal in self.signals.values():
                signal.set_state(SignalState.FLASHING_RED)
        elif self.fail_safe_mode == "flashing_yellow":
            for signal in self.signals.values():
                signal.set_state(SignalState.FLASHING_YELLOW)
        else:  # Default to all-red
            for signal in self.signals.values():
                signal.set_state(SignalState.RED)
    
    def _control_loop(self):
        """Main control loop for traffic light operation."""
        logger.debug("Control loop started")
        
        while not self.stop_event.is_set():
            try:
                # Check if we should change the active plan
                self._check_plan_schedule()
                
                # Get the active plan
                active_plan = self.plans[self.active_plan_id]
                
                # Handle emergency mode if active
                if self.emergency_mode_active:
                    self._handle_emergency_mode()
                    continue
                
                # Handle manual control if active
                if self.manual_control_active:
                    # Just update timing information but don't change states
                    self._update_timing_information()
                    time.sleep(0.1)
                    continue
                
                # Normal operation
                self._process_normal_operation(active_plan)
                
                # Small delay to prevent CPU hogging
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in control loop: {str(e)}")
                # Set to fail-safe mode on error
                self._set_fail_safe_mode()
                time.sleep(1.0)
    
    def _check_plan_schedule(self):
        """Check if we should change the active plan based on schedule."""
        # Find plans that should be active now
        active_plans = [plan_id for plan_id, plan in self.plans.items() if plan.is_active_now()]
        
        if active_plans and self.active_plan_id not in active_plans:
            # Change to the first active plan
            new_plan_id = active_plans[0]
            logger.info(f"Changing active plan from {self.active_plan_id} to {new_plan_id} based on schedule")
            self.set_active_plan(new_plan_id)
    
    def _handle_emergency_mode(self):
        """Handle emergency vehicle preemption."""
        # This would normally involve setting specific signals to green
        # based on the emergency vehicle's approach direction
        # For now, just a simple implementation that turns main signals green
        
        # Find main signals (assuming naming convention)
        main_signals = [signal_id for signal_id in self.signals.keys() if 'main' in signal_id]
        
        # Set main signals to green, others to red
        for signal_id, signal in self.signals.items():
            if signal_id in main_signals:
                if signal.state != SignalState.GREEN:
                    if signal.state == SignalState.RED:
                        signal.set_state(SignalState.GREEN)
                    elif signal.state == SignalState.YELLOW and signal.state_duration >= self.timing.yellow_time:
                        signal.set_state(SignalState.GREEN)
            else:
                if signal.state != SignalState.RED:
                    if signal.state == SignalState.GREEN:
                        signal.set_state(SignalState.YELLOW)
                    elif signal.state == SignalState.YELLOW and signal.state_duration >= self.timing.yellow_time:
                        signal.set_state(SignalState.RED)
        
        # Update timing information
        self._update_timing_information()
        
        # Wait a bit before checking again
        time.sleep(0.2)
    
    def _process_normal_operation(self, active_plan: SignalPlan):
        """Process normal traffic light operation based on the active plan."""
        # Update timing information
        self._update_timing_information()
        
        # Check if current phase has completed
        current_phase = active_plan.phases[self.current_phase_index]
        min_phase_time = current_phase.get('min_time', self.timing.min_green_time)
        max_phase_time = current_phase.get('max_time', self.timing.max_green_time)
        
        # Get phase signals
        phase_signals = current_phase['signals']
        
        # Check if we need to initialize this phase
        if self.phase_duration == 0.0:
            self._initialize_phase(phase_signals)
        
        # Determine if phase should end
        phase_complete = False
        
        # 1. Check minimum time requirement
        if self.phase_duration >= min_phase_time:
            # 2. Check for phase extension based on vehicle actuation
            if active_plan.vehicle_actuated and self.phase_duration < max_phase_time:
                # Check if any green signal has vehicle calls
                extend_phase = False
                for signal_id, state in phase_signals.items():
                    if state == SignalState.GREEN.value and signal_id in self.signals:
                        if self.signals[signal_id].vehicle_calls > 0:
                            extend_phase = True
                            break
                
                # If no extension needed or max time reached, complete the phase
                if not extend_phase or self.phase_duration >= max_phase_time:
                    phase_complete = True
            else:
                # Fixed time mode or max time reached
                if self.phase_duration >= max_phase_time:
                    phase_complete = True
        
        # Process phase completion if needed
        if phase_complete:
            self._complete_phase(phase_signals)
        
        # Process coordination if enabled
        if active_plan.coordination_mode and self.coordination_enabled:
            self._process_coordination(active_plan)
    
    def _initialize_phase(self, phase_signals: Dict[str, str]):
        """Initialize a new signal phase."""
        # Set signal states according to phase configuration
        for signal_id, state_value in phase_signals.items():
            if signal_id in self.signals:
                # Convert state string to enum
                try:
                    state = SignalState(state_value)
                    # Only go from RED directly to GREEN
                    # If current state is not RED, transition through YELLOW first
                    current_signal = self.signals[signal_id]
                    if state == SignalState.GREEN and current_signal.state != SignalState.RED:
                        # Need to transition through yellow first
                        if current_signal.state != SignalState.YELLOW:
                            current_signal.set_state(SignalState.YELLOW)
                    else:
                        current_signal.set_state(state)
                except ValueError:
                    logger.warning(f"Invalid signal state: {state_value}")
        
        logger.debug(f"Initialized phase {self.current_phase_index} of plan {self.active_plan_id}")
    
    def _complete_phase(self, phase_signals: Dict[str, str]):
        """Complete the current phase and prepare for the next one."""
        # Transition GREEN signals to YELLOW
        yellow_transition_needed = False
        
        for signal_id, state_value in phase_signals.items():
            if signal_id in self.signals and state_value == SignalState.GREEN.value:
                if self.signals[signal_id].state == SignalState.GREEN:
                    self.signals[signal_id].set_state(SignalState.YELLOW)
                    yellow_transition_needed = True
        
        # If yellow transition was initiated, wait for yellow time
        if yellow_transition_needed:
            # Note: we don't actually sleep here as that would block the control loop
            # Instead, we'll check in the next iteration if yellow time has elapsed
            # For signals that are now yellow, check if they've been yellow long enough
            all_yellow_complete = True
            for signal_id in phase_signals:
                if signal_id in self.signals and self.signals[signal_id].state == SignalState.YELLOW:
                    if self.signals[signal_id].state_duration < self.timing.yellow_time:
                        all_yellow_complete = False
                        break
            
            if not all_yellow_complete:
                return  # Continue with yellow phase
            
            # Yellow complete, transition to all-red
            for signal_id in self.signals:
                if self.signals[signal_id].state == SignalState.YELLOW:
                    self.signals[signal_id].set_state(SignalState.RED)
            
            # Wait for all-red time
            time.sleep(self.timing.all_red_time)
        
        # Move to next phase
        self.current_phase_index = (self.current_phase_index + 1) % len(self.plans[self.active_plan_id].phases)
        self.phase_start_time = time.time()
        self.phase_duration = 0.0
        
        # If we've completed a full cycle, update cycle metrics
        if self.current_phase_index == 0:
            self.cycle_start_time = time.time()
            self.total_cycles += 1
            logger.debug(f"Completed traffic cycle {self.total_cycles}")
    
    def _process_coordination(self, active_plan: SignalPlan):
        """Process signal coordination with adjacent intersections."""
        # Calculate time within cycle
        current_time = time.time()
        cycle_time = (current_time - self.cycle_start_time) % active_plan.cycle_length
        
        # Check if we need to adjust for coordination offset
        if not self.coordination_offset_active and self.current_phase_index == 0:
            # Apply coordination offset if we're at the start of the cycle
            cycle_offset = active_plan.offset
            
            if cycle_offset > 0 and cycle_time < cycle_offset:
                # We need to delay this cycle start to maintain offset
                logger.debug(f"Adjusting for coordination offset: {cycle_offset} seconds")
                self.coordination_offset_active = True
                
                # Extend the all-red time to achieve the offset
                time.sleep(cycle_offset - cycle_time)
                
                # Reset cycle start time
                self.cycle_start_time = time.time()
                self.coordination_offset_active = False
    
    def _update_timing_information(self):
        """Update timing information for all signals and phases."""
        current_time = time.time()
        
        # Update phase duration
        self.phase_duration = current_time - self.phase_start_time
        
        # Update signal timing
        for signal in self.signals.values():
            signal.update()
    
    def update(self, traffic_flow_data: Any) -> bool:
        """
        Update the controller with new traffic flow data.
        
        Args:
            traffic_flow_data: Traffic flow data from the analysis module
        
        Returns:
            True if update was processed, False otherwise
        """
        # Store the flow data
        self.current_flow_data = traffic_flow_data
        self.historical_flow_data.append((time.time(), traffic_flow_data))
        
        # Limit history size
        if len(self.historical_flow_data) > 1000:
            self.historical_flow_data = self.historical_flow_data[-1000:]
        
        # Extract vehicle counts from flow data if available
        try:
            # Update vehicle counts for signals based on flow data
            if hasattr(traffic_flow_data, 'dominant_direction') and traffic_flow_data.dominant_direction:
                direction = traffic_flow_data.dominant_direction.name.lower()
                # Map direction to signal IDs (assuming naming convention)
                for signal_id, signal in self.signals.items():
                    if direction in signal_id:
                        # Register vehicle calls for signals in dominant flow direction
                        signal.register_vehicle_call()
            
            # Update congestion level if available
            if hasattr(traffic_flow_data, 'density') and traffic_flow_data.density:
                density_name = traffic_flow_data.density.name
                # Map density level to congestion value
                density_map = {
                    'VERY_LOW': 0.0,
                    'LOW': 0.2,
                    'MODERATE': 0.4,
                    'HIGH': 0.6,
                    'VERY_HIGH': 0.8,
                    'CONGESTED': 1.0
                }
                self.congestion_level = density_map.get(density_name, 0.0)
            
            # Update total vehicles count if available
            if hasattr(traffic_flow_data, 'vehicle_count'):
                self.total_vehicles += traffic_flow_data.vehicle_count
        
        except Exception as e:
            logger.warning(f"Error processing traffic flow data: {str(e)}")
            return False
        
        # Process emergency vehicle detection
        if self.emergency_mode_enabled:
            self._check_emergency_vehicle(traffic_flow_data)
        
        return True
    
    def _check_emergency_vehicle(self, traffic_flow_data: Any) -> None:
        """
        Check for emergency vehicles in traffic flow data.
        
        Args:
            traffic_flow_data: Traffic flow data
        """
        # This would normally check for emergency vehicle detection
        # For demonstration, just a placeholder
        emergency_detected = False
        
        # Update emergency mode status
        if emergency_detected and not self.emergency_mode_active:
            logger.info("Emergency vehicle detected, activating emergency mode")
            self.emergency_mode_active = True
        elif not emergency_detected and self.emergency_mode_active:
            logger.info("Emergency vehicle no longer detected, deactivating emergency mode")
            self.emergency_mode_active = False
    
    def set_active_plan(self, plan_id: str) -> bool:
        """
        Set the active signal plan.
        
        Args:
            plan_id: ID of the plan to activate
        
        Returns:
            True if plan was activated, False if plan not found
        """
        if plan_id not in self.plans:
            logger.warning(f"Signal plan {plan_id} not found")
            return False
        
        logger.info(f"Activating signal plan: {plan_id}")
        
        # Complete the current phase if needed
        if self.running:
            active_plan = self.plans[self.active_plan_id]
            current_phase = active_plan.phases[self.current_phase_index]
            self._complete_phase(current_phase['signals'])
        
        # Set new active plan
        self.active_plan_id = plan_id
        self.current_phase_index = 0
        self.phase_start_time = time.time()
        self.phase_duration = 0.0
        self.cycle_start_time = time.time()
        self.coordination_offset_active = False
        
        return True
    
    def set_manual_control(self, enabled: bool) -> None:
        """
        Enable or disable manual control mode.
        
        Args:
            enabled: Whether to enable manual control
        """
        self.manual_control_active = enabled
        logger.info(f"Manual control {'enabled' if enabled else 'disabled'}")
    
    def set_signal_state(self, signal_id: str, state: SignalState) -> bool:
        """
        Manually set a signal state (only in manual control mode).
        
        Args:
            signal_id: ID of the signal to set
            state: New signal state
        
        Returns:
            True if state was set, False otherwise
        """
        if not self.manual_control_active:
            logger.warning("Cannot set signal state: manual control not active")
            return False
        
        if signal_id not in self.signals:
            logger.warning(f"Signal {signal_id} not found")
            return False
        
        self.signals[signal_id].set_state(state)
        logger.info(f"Manually set signal {signal_id} to {state.value}")
        return True
    
    def add_signal_plan(self, plan: Dict[str, Any]) -> bool:
        """
        Add a new signal plan.
        
        Args:
            plan: Signal plan configuration
        
        Returns:
            True if plan was added, False otherwise
        """
        try:
            signal_plan = SignalPlan.from_dict(plan)
            self.plans[signal_plan.plan_id] = signal_plan
            logger.info(f"Added signal plan: {signal_plan.plan_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding signal plan: {str(e)}")
            return False
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current controller state.
        
        Returns:
            Dictionary with current state information
        """
        return {
            'intersection_id': self.intersection_id,
            'running': self.running,
            'active_plan_id': self.active_plan_id,
            'current_phase_index': self.current_phase_index,
            'phase_duration': self.phase_duration,
            'total_cycles': self.total_cycles,
            'emergency_mode_active': self.emergency_mode_active,
            'manual_control_active': self.manual_control_active,
            'signals': {signal_id: signal.to_dict() for signal_id, signal in self.signals.items()},
            'congestion_level': self.congestion_level
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the controller.
        
        Returns:
            Dictionary with performance metrics
        """
        # Calculate metrics for all signals
        signal_metrics = {}
        for signal_id, signal in self.signals.items():
            signal_metrics[signal_id] = signal.get_efficiency_metrics()
        
        # Calculate average metrics
        avg_green_ratio = np.mean([m['green_ratio'] for m in signal_metrics.values()]) if signal_metrics else 0.0
        avg_red_ratio = np.mean([m['red_ratio'] for m in signal_metrics.values()]) if signal_metrics else 0.0
        
        # Return metrics
        return {
            'intersection_id': self.intersection_id,
            'total_cycles': self.total_cycles,
            'total_vehicles': self.total_vehicles,
            'avg_green_ratio': avg_green_ratio,
            'avg_red_ratio': avg_red_ratio,
            'congestion_level': self.congestion_level,
            'signal_metrics': signal_metrics,
            'uptime': time.time() - self.cycle_start_time if self.running else 0.0
        }
    
    def emergency_preemption(self, activate: bool, direction: Optional[str] = None) -> None:
        """
        Activate or deactivate emergency preemption.
        
        Args:
            activate: Whether to activate emergency preemption
            direction: Optional approach direction for the emergency vehicle
        """
        if not self.emergency_mode_enabled:
            logger.warning("Emergency preemption not enabled for this controller")
            return
        
        self.emergency_mode_active = activate
        
        logger.info(f"Emergency preemption {'activated' for 'approach from {direction}'' if activate else 'deactivated'}")
    
    def save_configuration(self, file_path: str) -> bool:
        """
        Save controller configuration to file.
        
        Args:
            file_path: Path to save configuration
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create configuration dictionary
            config = {
                'intersection_id': self.intersection_id,
                'signals': {signal_id: signal.state.value for signal_id, signal in self.signals.items()},
                'timing_params': self.timing.to_dict(),
                'plans': [plan.to_dict() for plan in self.plans.values()],
                'optimization_mode': self.optimization_mode,
                'coordination_enabled': self.coordination_enabled,
                'emergency_mode_enabled': self.emergency_mode_enabled,
                'fail_safe_mode': self.fail_safe_mode,
                'startup_sequence': self.startup_sequence
            }
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Configuration saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    @classmethod
    def load_configuration(cls, file_path: str) -> 'TrafficLightController':
        """
        Load controller configuration from file.
        
        Args:
            file_path: Path to configuration file
        
        Returns:
            TrafficLightController instance
        """
        try:
            # Read configuration
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            # Convert signal states to enums
            signals = {}
            for signal_id, state_value in config['signals'].items():
                signals[signal_id] = SignalState(state_value)
            
            # Create controller
            controller = cls(
                intersection_id=config['intersection_id'],
                signals=signals,
                timing_params=config['timing_params'],
                plans=config['plans'],
                optimization_mode=config.get('optimization_mode', 'fixed'),
                coordination_enabled=config.get('coordination_enabled', False),
                emergency_mode_enabled=config.get('emergency_mode_enabled', True),
                fail_safe_mode=config.get('fail_safe_mode', 'flashing_red'),
                startup_sequence=config.get('startup_sequence', 'all_red')
            )
            
            logger.info(f"Configuration loaded from {file_path}")
            return controller
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
