"""
Database module for the Traffic Monitoring System.
Provides database connectivity and data model functionality.
"""

from .db_connect import (
    DatabaseConnection, ConnectionParams, execute_query, execute_transaction
)
from .data_models import (
    Base, Vehicle, DetectionEvent, CountEvent, SpeedMeasurement, TrafficFlow
)

__all__ = [
    'DatabaseConnection',
    'ConnectionParams',
    'execute_query',
    'execute_transaction',
    'Base',
    'Vehicle',
    'DetectionEvent',
    'CountEvent',
    'SpeedMeasurement',
    'TrafficFlow'
]
