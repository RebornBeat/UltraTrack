"""
Traffic management module for the Traffic Monitoring System.
Provides functionality for traffic light control, optimization, and simulation.
"""

from .traffic_light_controller import (
    TrafficLightController, 
    TrafficSignal, 
    SignalState, 
    SignalTiming, 
    SignalPlan
)
from .signal_optimizer import (
    SignalOptimizer, 
    FixedTimeOptimizer, 
    AdaptiveOptimizer, 
    PredictiveOptimizer,
    CoordinationOptimizer
)
from .simulation_model import (
    SimulationModel, 
    VehicleGenerator, 
    SimulatedVehicle, 
    TrafficStats
)
from .intersection_manager import (
    IntersectionManager, 
    Intersection, 
    Lane,
    LaneDirection, 
    IntersectionNetwork
)

__all__ = [
    'TrafficLightController',
    'TrafficSignal',
    'SignalState',
    'SignalTiming',
    'SignalPlan',
    'SignalOptimizer',
    'FixedTimeOptimizer',
    'AdaptiveOptimizer',
    'PredictiveOptimizer',
    'CoordinationOptimizer',
    'SimulationModel',
    'VehicleGenerator',
    'SimulatedVehicle',
    'TrafficStats',
    'IntersectionManager',
    'Intersection',
    'Lane',
    'LaneDirection',
    'IntersectionNetwork'
]
