"""
Analysis module for the Traffic Monitoring System.
Provides traffic counting, speed estimation, and flow analysis functionality.
"""

from .traffic_counter import (
    TrafficCounter, CountingZone, VehicleCount, TrafficStatistics
)
from .speed_estimator import (
    SpeedEstimator, SpeedMeasurement, CalibrationMethod
)
from .flow_analyzer import (
    FlowAnalyzer, TrafficDensity, FlowDirection, TrafficFlow
)

__all__ = [
    'TrafficCounter',
    'CountingZone',
    'VehicleCount',
    'TrafficStatistics',
    'SpeedEstimator',
    'SpeedMeasurement',
    'CalibrationMethod',
    'FlowAnalyzer',
    'TrafficDensity',
    'FlowDirection',
    'TrafficFlow'
]
