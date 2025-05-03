"""
UltraTrack Biometric Processors Module

This module provides additional biometric processing capabilities:
- Ear recognition
- Iris recognition
- Fingerprint enhancement
- Vein pattern recognition
- Heart rhythm analysis
- Remote vital signs monitoring

Copyright (c) 2025 Your Organization
"""

import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Import key components for easier access
from ultratrack.ai_models.biometric_processors.ear_recognition import (
    EarRecognizer, EarFeatures, MatchResult, RecognitionParameters,
    EarDetection, MatchConfidence
)
from ultratrack.ai_models.biometric_processors.iris_recognition import (
    IrisRecognizer, IrisFeatures, IrisMatchResult, IrisParameters,
    IrisQuality, IrisTemplateFormat
)
from ultratrack.ai_models.biometric_processors.fingerprint_enhancement import (
    FingerprintEnhancer, EnhancementParameters, EnhancedImage,
    QualityAssessment, FeatureExtraction
)
from ultratrack.ai_models.biometric_processors.vein_pattern_recognition import (
    VeinPatternRecognizer, VeinFeatures, VeinMatchResult, PatternParameters,
    VeinType, PatternQuality
)
from ultratrack.ai_models.biometric_processors.heart_rhythm_analysis import (
    HeartRhythmAnalyzer, RhythmSignature, MatchResult, AnalysisParameters,
    HeartBeatFeatures, SignalQuality
)
from ultratrack.ai_models.biometric_processors.remote_vital_signs import (
    RemoteVitalSigns, VitalSignType, MeasurementConfidence, SignalParameters,
    TemporalPattern, PhysiologicalFeatures
)

# Consolidated biometric processors model
class BiometricProcessorsModel:
    """
    Unified interface for additional biometric processing capabilities.
    
    This class provides a high-level interface to all biometric processors,
    including ear recognition, iris recognition, fingerprint enhancement,
    vein pattern recognition, heart rhythm analysis, and remote vital signs.
    """
    
    def __init__(self, config=None):
        """Initialize the biometric processors model with the given configuration."""
        self.ear_recognizer = EarRecognizer(config)
        self.iris_recognizer = IrisRecognizer(config)
        self.fingerprint_enhancer = FingerprintEnhancer(config)
        self.vein_pattern_recognizer = VeinPatternRecognizer(config)
        self.heart_rhythm_analyzer = HeartRhythmAnalyzer(config)
        self.remote_vital_signs = RemoteVitalSigns(config)
        
        logger.info("Biometric processors model initialized")
    
    def process_and_match(self, biometric_data, biometric_type, parameters=None):
        """
        Perform biometric processing and matching.
        
        Args:
            biometric_data: Input biometric data
            biometric_type: Type of biometric data
            parameters: Optional processing parameters
            
        Returns:
            Biometric processing results with matching
        """
        # Implementation details would go here
        pass

# Export public API
__all__ = [
    # Main model class
    'BiometricProcessorsModel',
    
    # Ear recognition interfaces
    'EarRecognizer', 'EarFeatures', 'MatchResult', 'RecognitionParameters',
    'EarDetection', 'MatchConfidence',
    
    # Iris recognition interfaces
    'IrisRecognizer', 'IrisFeatures', 'IrisMatchResult', 'IrisParameters',
    'IrisQuality', 'IrisTemplateFormat',
    
    # Fingerprint interfaces
    'FingerprintEnhancer', 'EnhancementParameters', 'EnhancedImage',
    'QualityAssessment', 'FeatureExtraction',
    
    # Vein pattern interfaces
    'VeinPatternRecognizer', 'VeinFeatures', 'VeinMatchResult', 'PatternParameters',
    'VeinType', 'PatternQuality',
    
    # Heart rhythm interfaces
    'HeartRhythmAnalyzer', 'RhythmSignature', 'MatchResult', 'AnalysisParameters',
    'HeartBeatFeatures', 'SignalQuality',
    
    # Remote vital signs interfaces
    'RemoteVitalSigns', 'VitalSignType', 'MeasurementConfidence', 'SignalParameters',
    'TemporalPattern', 'PhysiologicalFeatures',
]

logger.debug("UltraTrack biometric processors module initialized")
