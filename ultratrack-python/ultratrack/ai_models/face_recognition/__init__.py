"""
UltraTrack Face Recognition Module

This module provides face detection and recognition capabilities:
- Face detection in images and video
- Feature extraction and matching
- Anti-spoofing protection
- Aging compensation
- Disguise penetration

Copyright (c) 2025 Your Organization
"""

import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Import key components for easier access
from ultratrack.ai_models.face_recognition.detector import (
    FaceDetector, DetectedFace, DetectionParameters, FaceQuality,
    FaceAlignment, FaceAngles
)
from ultratrack.ai_models.face_recognition.recognizer import (
    FaceRecognizer, FaceFeatures, MatchResult, RecognitionParameters,
    IdentityConfidence, FacialAttributes
)
from ultratrack.ai_models.face_recognition.anti_spoofing import (
    AntiSpoofing, SpoofType, LivenessScore, SpoofingDetection,
    PresentationAttack, AntiSpoofingParameters
)
from ultratrack.ai_models.face_recognition.aging_compensation import (
    AgingCompensation, AgeFactor, AgeProgression, AgeRegression,
    TemporalConsistency, AgingModel
)
from ultratrack.ai_models.face_recognition.disguise_penetration import (
    DisguisePenetration, DisguiseType, DisguiseConfidence, 
    UnderlyingFeatures, ObscurationLevel
)
from ultratrack.ai_models.face_recognition.model_loader import (
    FaceModelLoader, ModelConfiguration, ModelPerformance, 
    ModelRequirements, ModelCapabilities
)

# Consolidated face recognition model
class FaceRecognitionModel:
    """
    Unified interface for face recognition capabilities.
    
    This class provides a high-level interface to all face recognition
    capabilities, combining detection, recognition, anti-spoofing,
    aging compensation, and disguise penetration.
    """
    
    def __init__(self, config=None):
        """Initialize the face recognition model with the given configuration."""
        self.detector = FaceDetector(config)
        self.recognizer = FaceRecognizer(config)
        self.anti_spoofing = AntiSpoofing(config)
        self.aging_compensation = AgingCompensation(config)
        self.disguise_penetration = DisguisePenetration(config)
        self.model_loader = FaceModelLoader(config)
        
        logger.info("Face recognition model initialized")
    
    def detect_and_recognize(self, image, parameters=None):
        """
        Perform complete face detection and recognition.
        
        Args:
            image: Input image or video frame
            parameters: Optional processing parameters
            
        Returns:
            List of recognized faces with identity matches
        """
        # Implementation details would go here
        pass

# Export public API
__all__ = [
    # Main model class
    'FaceRecognitionModel',
    
    # Detector interfaces
    'FaceDetector', 'DetectedFace', 'DetectionParameters', 'FaceQuality',
    'FaceAlignment', 'FaceAngles',
    
    # Recognizer interfaces
    'FaceRecognizer', 'FaceFeatures', 'MatchResult', 'RecognitionParameters',
    'IdentityConfidence', 'FacialAttributes',
    
    # Anti-spoofing interfaces
    'AntiSpoofing', 'SpoofType', 'LivenessScore', 'SpoofingDetection',
    'PresentationAttack', 'AntiSpoofingParameters',
    
    # Aging compensation interfaces
    'AgingCompensation', 'AgeFactor', 'AgeProgression', 'AgeRegression',
    'TemporalConsistency', 'AgingModel',
    
    # Disguise penetration interfaces
    'DisguisePenetration', 'DisguiseType', 'DisguiseConfidence', 
    'UnderlyingFeatures', 'ObscurationLevel',
    
    # Model loader interfaces
    'FaceModelLoader', 'ModelConfiguration', 'ModelPerformance', 
    'ModelRequirements', 'ModelCapabilities',
]

logger.debug("UltraTrack face recognition module initialized")
