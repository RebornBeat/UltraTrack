"""
UltraTrack Dashboard Module

This module provides the real-time tracking dashboard:
- User interface for tracking visualization
- Multi-camera viewing
- Analytics visualization
- Alert management
- Timeline viewing
- Relationship graph visualization

Copyright (c) 2025 Your Organization
"""

import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Import key components for easier access
from ultratrack.frontend.dashboard.app import (
    DashboardApp, AppConfig, UIState, EventHandler
)
from ultratrack.frontend.dashboard.authentication import (
    AuthenticationUI, LoginForm, SessionManager, UserProfile
)
from ultratrack.frontend.dashboard.visualization import (
    Visualization, MapView, DataVisualization, ChartType,
    VisualizationTheme, InteractiveElement
)
from ultratrack.frontend.dashboard.tracking_view import (
    TrackingView, TrackDisplayMode, TrackFilter, TrackSelection,
    VisualizationOptions, TrackingControls
)
from ultratrack.frontend.dashboard.analytics_view import (
    AnalyticsView, AnalyticsDisplayMode, DataFilter, AnalyticsSelection,
    ReportGeneration, AnalyticsExport
)
from ultratrack.frontend.dashboard.alert_panel import (
    AlertPanel, AlertDisplayMode, AlertFilter, AlertPriority,
    NotificationSettings, AlertHandling
)
from ultratrack.frontend.dashboard.multi_camera_viewer import (
    MultiCameraViewer, ViewLayout, CameraSelection, StreamControl,
    ViewSynchronization, PTZControl
)
from ultratrack.frontend.dashboard.predictive_analytics import (
    PredictiveAnalyticsView, PredictionVisualization, ScenarioBuilder,
    ForecastOptions, ConfidenceDisplay
)
from ultratrack.frontend.dashboard.timeline_view import (
    TimelineView, TimelineDisplayMode, TimeRange, EventFilter,
    PlaybackControl, TemporalNavigation
)
from ultratrack.frontend.dashboard.relationship_graph import (
    RelationshipGraph, GraphDisplayMode, NodeType, EdgeType,
    GraphLayout, InteractionOptions
)

# Export public API
__all__ = [
    # Application interfaces
    'DashboardApp', 'AppConfig', 'UIState', 'EventHandler',
    
    # Authentication interfaces
    'AuthenticationUI', 'LoginForm', 'SessionManager', 'UserProfile',
    
    # Visualization interfaces
    'Visualization', 'MapView', 'DataVisualization', 'ChartType',
    'VisualizationTheme', 'InteractiveElement',
    
    # Tracking view interfaces
    'TrackingView', 'TrackDisplayMode', 'TrackFilter', 'TrackSelection',
    'VisualizationOptions', 'TrackingControls',
    
    # Analytics view interfaces
    'AnalyticsView', 'AnalyticsDisplayMode', 'DataFilter', 'AnalyticsSelection',
    'ReportGeneration', 'AnalyticsExport',
    
    # Alert panel interfaces
    'AlertPanel', 'AlertDisplayMode', 'AlertFilter', 'AlertPriority',
    'NotificationSettings', 'AlertHandling',
    
    # Multi-camera viewer interfaces
    'MultiCameraViewer', 'ViewLayout', 'CameraSelection', 'StreamControl',
    'ViewSynchronization', 'PTZControl',
    
    # Predictive analytics interfaces
    'PredictiveAnalyticsView', 'PredictionVisualization', 'ScenarioBuilder',
    'ForecastOptions', 'ConfidenceDisplay',
    
    # Timeline view interfaces
    'TimelineView', 'TimelineDisplayMode', 'TimeRange', 'EventFilter',
    'PlaybackControl', 'TemporalNavigation',
    
    # Relationship graph interfaces
    'RelationshipGraph', 'GraphDisplayMode', 'NodeType', 'EdgeType',
    'GraphLayout', 'InteractionOptions',
]

logger.debug("UltraTrack dashboard module initialized")
