"""
UltraTrack Behavior Analysis Module

This module provides behavior analysis capabilities:
- Movement pattern analysis
- Micro-expression analysis
- Interaction mapping
- Routine detection
- Anomaly detection

Copyright (c) 2025 Your Organization
"""

import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Import key components for easier access
from ultratrack.ai_models.behavior_analysis.movement_patterns import (
    MovementPatternAnalyzer, MovementPattern, AnalysisParameters,
    PatternSignificance, MotionSignature
)
from ultratrack.ai_models.behavior_analysis.micro_expression_analyzer import (
    MicroExpressionAnalyzer, MicroExpression, ExpressionParameters,
    ExpressionSignificance, FacialMicromovement
)
from ultratrack.ai_models.behavior_analysis.interaction_mapper import (
    InteractionMapper, Interaction, MappingParameters,
    InteractionSignificance, SocialDynamics
)
from ultratrack.ai_models.behavior_analysis.routine_detector import (
    RoutineDetector, Routine, DetectionParameters,
    RoutineSignificance, TemporalPattern
)
from ultratrack.ai_models.behavior_analysis.anomaly_detector import (
    AnomalyDetector, Anomaly, DetectionParameters,
    AnomalySignificance, BehavioralBaseline
)

# Consolidated behavior analysis model
class BehaviorAnalysisModel:
    """
    Unified interface for behavior analysis capabilities.
    
    This class provides a high-level interface to all behavior analysis
    capabilities, including movement pattern analysis, micro-expression
    analysis, interaction mapping, routine detection, and anomaly detection.
    """
    
    def __init__(self, config=None):
        """Initialize the behavior analysis model with the given configuration."""
        self.movement_analyzer = MovementPatternAnalyzer(config)
        self.expression_analyzer = MicroExpressionAnalyzer(config)
        self.interaction_mapper = InteractionMapper(config)
        self.routine_detector = RoutineDetector(config)
        self.anomaly_detector = AnomalyDetector(config)
        
        logger.info("Behavior analysis model initialized")
    
    def analyze_behavior(self, tracking_data, parameters=None):
        """
        Perform complete behavior analysis.
        
        Args:
            tracking_data: Input tracking data sequence
            parameters: Optional analysis parameters
            
        Returns:
            Behavior analysis results
        """
        # Implementation details would go here
        pass

# Export public API
__all__ = [
    # Main model class
    'BehaviorAnalysisModel',
    
    # Movement pattern interfaces
    'MovementPatternAnalyzer', 'MovementPattern', 'AnalysisParameters',
    'PatternSignificance', 'MotionSignature',
    
    # Micro-expression interfaces
    'MicroExpressionAnalyzer', 'MicroExpression', 'ExpressionParameters',
    'ExpressionSignificance', 'FacialMicromovement',
    
    # Interaction mapping interfaces
    'InteractionMapper', 'Interaction', 'MappingParameters',
    'InteractionSignificance', 'SocialDynamics',
    
    # Routine detection interfaces
    'RoutineDetector', 'Routine', 'DetectionParameters',
    'RoutineSignificance', 'TemporalPattern',
    
    # Anomaly detection interfaces
    'AnomalyDetector', 'Anomaly', 'DetectionParameters',
    'AnomalySignificance', 'BehavioralBaseline',
]

logger.debug("UltraTrack behavior analysis module initialized")
