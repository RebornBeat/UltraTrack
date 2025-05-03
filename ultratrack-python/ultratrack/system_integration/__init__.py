"""
UltraTrack System Integration Module

This module provides components for integrating tracking across the system:
- Core tracking engine
- Geospatial mapping
- Trajectory analysis
- Alert generation
- Distributed tracking coordination
- Network handoff management
- Blind spot navigation
- Transition management

Copyright (c) 2025 Your Organization
"""

import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Import key components for easier access
from ultratrack.system_integration.tracking_engine import (
    TrackingEngine, Track, TrackStatus, TrackConfidence,
    TrackingParameters, TrackingMetrics
)
from ultratrack.system_integration.geospatial_mapper import (
    GeospatialMapper, GeographicCoordinate, Projection, 
    MapLayer, SpatialIndex, GeofenceRegion
)
from ultratrack.system_integration.trajectory_analyzer import (
    TrajectoryAnalyzer, Trajectory, PathSegment, MovementPattern,
    SpeedProfile, DirectionChange
)
from ultratrack.system_integration.alert_system import (
    AlertSystem, Alert, AlertPriority, AlertType, 
    NotificationChannel, AlertTemplate
)
from ultratrack.system_integration.inter_node_communication import (
    InterNodeCommunication, NodeMessage, CommunicationChannel, 
    MessagePriority, NodeEndpoint
)
from ultratrack.system_integration.handoff_manager import (
    HandoffManager, HandoffRequest, HandoffStatus, HandoffMetrics,
    ZoneBoundary, HandoffProtocol
)
from ultratrack.system_integration.distributed_tracking import (
    DistributedTracking, GlobalTrackIdentifier, TrackFragment,
    NetworkTopology, DistributedConsensus
)
from ultratrack.system_integration.vehicle_transition_tracker import (
    VehicleTransitionTracker, VehicleTransition, TransitionConfidence,
    VehicleType, TransitionEvent
)
from ultratrack.system_integration.building_interior_connector import (
    BuildingInteriorConnector, InteriorMap, BuildingTransition,
    FloorPlan, AccessPoint, TransitionProbability
)
from ultratrack.system_integration.cross_network_resolver import (
    CrossNetworkResolver, NetworkBoundary, IdentityCorrelation,
    CrossNetworkLink, ResolverStrategy
)
from ultratrack.system_integration.blind_spot_analyzer import (
    BlindSpotAnalyzer, BlindSpot, InferredPath, Constraint,
    LocationProbability, PathConstraint
)
from ultratrack.system_integration.transition_predictor import (
    TransitionPredictor, PredictedTransition, TransitionPattern,
    TransitionTiming, PredictionConfidence
)
from ultratrack.system_integration.global_coordination import (
    GlobalCoordinator, GlobalTrackRegistry, JurisdictionBoundary,
    GlobalTrackingMetrics, CrossJurisdictionProtocol
)

# Export public API
__all__ = [
    # Tracking engine interfaces
    'TrackingEngine', 'Track', 'TrackStatus', 'TrackConfidence',
    'TrackingParameters', 'TrackingMetrics',
    
    # Geospatial mapping interfaces
    'GeospatialMapper', 'GeographicCoordinate', 'Projection', 
    'MapLayer', 'SpatialIndex', 'GeofenceRegion',
    
    # Trajectory analysis interfaces
    'TrajectoryAnalyzer', 'Trajectory', 'PathSegment', 'MovementPattern',
    'SpeedProfile', 'DirectionChange',
    
    # Alert system interfaces
    'AlertSystem', 'Alert', 'AlertPriority', 'AlertType', 
    'NotificationChannel', 'AlertTemplate',
    
    # Inter-node communication interfaces
    'InterNodeCommunication', 'NodeMessage', 'CommunicationChannel', 
    'MessagePriority', 'NodeEndpoint',
    
    # Handoff management interfaces
    'HandoffManager', 'HandoffRequest', 'HandoffStatus', 'HandoffMetrics',
    'ZoneBoundary', 'HandoffProtocol',
    
    # Distributed tracking interfaces
    'DistributedTracking', 'GlobalTrackIdentifier', 'TrackFragment',
    'NetworkTopology', 'DistributedConsensus',
    
    # Vehicle transition interfaces
    'VehicleTransitionTracker', 'VehicleTransition', 'TransitionConfidence',
    'VehicleType', 'TransitionEvent',
    
    # Building interior interfaces
    'BuildingInteriorConnector', 'InteriorMap', 'BuildingTransition',
    'FloorPlan', 'AccessPoint', 'TransitionProbability',
    
    # Cross-network interfaces
    'CrossNetworkResolver', 'NetworkBoundary', 'IdentityCorrelation',
    'CrossNetworkLink', 'ResolverStrategy',
    
    # Blind spot analysis interfaces
    'BlindSpotAnalyzer', 'BlindSpot', 'InferredPath', 'Constraint',
    'LocationProbability', 'PathConstraint',
    
    # Transition prediction interfaces
    'TransitionPredictor', 'PredictedTransition', 'TransitionPattern',
    'TransitionTiming', 'PredictionConfidence',
    
    # Global coordination interfaces
    'GlobalCoordinator', 'GlobalTrackRegistry', 'JurisdictionBoundary',
    'GlobalTrackingMetrics', 'CrossJurisdictionProtocol',
]

logger.debug("UltraTrack system integration module initialized")
