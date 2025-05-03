"""
UltraTrack Thermal Analysis Module

This module provides thermal analysis capabilities:
- Heat signature extraction
- Thermal profile matching
- Environmental compensation

Copyright (c) 2025 Your Organization
"""

import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Import key components for easier access
from ultratrack.ai_models.thermal_analysis.heat_signature_extractor import (
    HeatSignatureExtractor, ThermalSignature, ExtractionParameters,
    SignatureQuality, TemperatureMap
)
from ultratrack.ai_models.thermal_analysis.thermal_matcher import (
    ThermalMatcher, MatchResult, MatchingParameters, MatchConfidence,
    SignatureSimilarity, TemporalConsistency
)
from ultratrack.ai_models.thermal_analysis.environment_compensation import (
    EnvironmentCompensation, AmbientConditions, TemperatureNormalization,
    ActivityCompensation, BaselineEstimation
)

# Consolidated thermal analysis model
class ThermalAnalysisModel:
    """
    Unified interface for thermal analysis capabilities.
    
    This class provides a high-level interface to all thermal analysis
    capabilities, including heat signature extraction, thermal matching,
    and environmental compensation.
    """
    
    def __init__(self, config=None):
        """Initialize the thermal analysis model with the given configuration."""
        self.extractor = HeatSignatureExtractor(config)
        self.matcher = ThermalMatcher(config)
        self.environment_compensation = EnvironmentCompensation(config)
        
        logger.info("Thermal analysis model initialized")
    
    def extract_and_match(self, thermal_image, parameters=None):
        """
        Perform complete thermal signature extraction and matching.
        
        Args:
            thermal_image: Input thermal image or video frame
            parameters: Optional processing parameters
            
        Returns:
            Thermal analysis results with identity matches
        """
        # Implementation details would go here
        pass

# Export public API
__all__ = [
    # Main model class
    'ThermalAnalysisModel',
    
    # Extractor interfaces
    'HeatSignatureExtractor', 'ThermalSignature', 'ExtractionParameters',
    'SignatureQuality', 'TemperatureMap',
    
    # Matcher interfaces
    'ThermalMatcher', 'MatchResult', 'MatchingParameters', 'MatchConfidence',
    'SignatureSimilarity', 'TemporalConsistency',
    
    # Environmental compensation interfaces
    'EnvironmentCompensation', 'AmbientConditions', 'TemperatureNormalization',
    'ActivityCompensation', 'BaselineEstimation',
]

logger.debug("UltraTrack thermal analysis module initialized")
