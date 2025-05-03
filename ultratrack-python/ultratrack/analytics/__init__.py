"""
UltraTrack Analytics Module

This module provides components for analyzing tracking data:
- Behavior analysis
- Historical tracking reconstruction
- Reporting and visualization
- Pattern recognition
- Anomaly detection
- Predictive analytics
- Relationship mapping
- Location analysis

Copyright (c) 2025 Your Organization
"""

import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Import key components for easier access
from ultratrack.analytics.behavior_analysis import (
    BehaviorAnalyzer, BehaviorPattern, ActivityProfile, Routine,
    BehaviorMetrics, BehavioralSignature
)
from ultratrack.analytics.historical_tracking import (
    HistoricalTrackingEngine, TimelineReconstruction, TrackHistory,
    HistoricalQuery, TimeRange, HistoricalMetadata
)
from ultratrack.analytics.reporting import (
    ReportGenerator, ReportTemplate, ReportFormat, ReportParameters,
    ReportSchedule, ReportDistribution
)
from ultratrack.analytics.metrics import (
    AnalyticsMetrics, MetricType, MetricValue, MetricTrend,
    MetricThreshold, MetricVisualization
)
from ultratrack.analytics.pattern_recognition import (
    PatternRecognizer, RecognizedPattern, PatternSignificance,
    PatternRule, PatternCategory
)
from ultratrack.analytics.anomaly_detection import (
    AnomalyDetector, Anomaly, AnomalyType, AnomalySeverity,
    AnomalyEvidence, DetectionRule
)
from ultratrack.analytics.predictive_tracking import (
    PredictiveEngine, TrackingPrediction, PredictionConfidence,
    ForecastHorizon, PredictionModel
)
from ultratrack.analytics.associate_mapper import (
    AssociateMapper, Relationship, RelationshipType, RelationshipStrength,
    SocialNetwork, ConnectionEvidence
)
from ultratrack.analytics.location_frequency_analyzer import (
    LocationFrequencyAnalyzer, FrequentedLocation, VisitPattern,
    LocationSignificance, GeospatialHotspot
)
from ultratrack.analytics.schedule_predictor import (
    SchedulePredictor, PredictedSchedule, RoutineEvent, 
    TemporalPattern, ScheduleConfidence
)
from ultratrack.analytics.relationship_inference import (
    RelationshipInference, InferredRelationship, RelationshipEvidence,
    RelationshipCategory, NetworkPosition
)
from ultratrack.analytics.probability_heatmap import (
    ProbabilityHeatmap, SpatialProbability, HeatmapLayer,
    ProbabilityGradient, TimeBasedHeatmap
)
from ultratrack.analytics.lifestyle_analyzer import (
    LifestyleAnalyzer, LifestyleProfile, ActivityCategory,
    LifestylePatterns, ProfileMetrics
)

# Export public API
__all__ = [
    # Behavior analysis interfaces
    'BehaviorAnalyzer', 'BehaviorPattern', 'ActivityProfile', 'Routine',
    'BehaviorMetrics', 'BehavioralSignature',
    
    # Historical tracking interfaces
    'HistoricalTrackingEngine', 'TimelineReconstruction', 'TrackHistory',
    'HistoricalQuery', 'TimeRange', 'HistoricalMetadata',
    
    # Reporting interfaces
    'ReportGenerator', 'ReportTemplate', 'ReportFormat', 'ReportParameters',
    'ReportSchedule', 'ReportDistribution',
    
    # Metrics interfaces
    'AnalyticsMetrics', 'MetricType', 'MetricValue', 'MetricTrend',
    'MetricThreshold', 'MetricVisualization',
    
    # Pattern recognition interfaces
    'PatternRecognizer', 'RecognizedPattern', 'PatternSignificance',
    'PatternRule', 'PatternCategory',
    
    # Anomaly detection interfaces
    'AnomalyDetector', 'Anomaly', 'AnomalyType', 'AnomalySeverity',
    'AnomalyEvidence', 'DetectionRule',
    
    # Predictive tracking interfaces
    'PredictiveEngine', 'TrackingPrediction', 'PredictionConfidence',
    'ForecastHorizon', 'PredictionModel',
    
    # Associate mapping interfaces
    'AssociateMapper', 'Relationship', 'RelationshipType', 'RelationshipStrength',
    'SocialNetwork', 'ConnectionEvidence',
    
    # Location frequency interfaces
    'LocationFrequencyAnalyzer', 'FrequentedLocation', 'VisitPattern',
    'LocationSignificance', 'GeospatialHotspot',
    
    # Schedule prediction interfaces
    'SchedulePredictor', 'PredictedSchedule', 'RoutineEvent', 
    'TemporalPattern', 'ScheduleConfidence',
    
    # Relationship inference interfaces
    'RelationshipInference', 'InferredRelationship', 'RelationshipEvidence',
    'RelationshipCategory', 'NetworkPosition',
    
    # Probability heatmap interfaces
    'ProbabilityHeatmap', 'SpatialProbability', 'HeatmapLayer',
    'ProbabilityGradient', 'TimeBasedHeatmap',
    
    # Lifestyle analysis interfaces
    'LifestyleAnalyzer', 'LifestyleProfile', 'ActivityCategory',
    'LifestylePatterns', 'ProfileMetrics',
]

logger.debug("UltraTrack analytics module initialized")
