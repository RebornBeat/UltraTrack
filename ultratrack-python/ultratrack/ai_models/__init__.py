"""
UltraTrack AI Models Module

This module provides access to the AI models used for identification and tracking:
- Face recognition
- Person re-identification
- Gait analysis
- Voice recognition
- Object detection
- Specialized biometrics
- Thermal and RF analysis
- License plate recognition
- Behavior analysis
- Multi-modal fusion

Copyright (c) 2025 Your Organization
"""

import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Import key components for easier access
from ultratrack.ai_models.model_registry import (
    ModelRegistry, ModelInfo, ModelVersion, ModelStatus,
    ModelEvaluation, ModelMetrics
)
from ultratrack.ai_models.model_versioning import (
    VersionManager, Version, ModelRollback, DeploymentStage,
    ABTest, VersionComparisonResult
)

# High-level models interface classes
from ultratrack.ai_models.face_recognition import FaceRecognitionModel
from ultratrack.ai_models.person_reid import PersonReIDModel
from ultratrack.ai_models.gait_analysis import GaitAnalysisModel
from ultratrack.ai_models.voice_recognition import VoiceRecognitionModel
from ultratrack.ai_models.object_detection import ObjectDetectionModel
from ultratrack.ai_models.biometric_processors import BiometricProcessorsModel
from ultratrack.ai_models.thermal_analysis import ThermalAnalysisModel
from ultratrack.ai_models.rf_analysis import RFAnalysisModel
from ultratrack.ai_models.lpr import LPRModel
from ultratrack.ai_models.behavior_analysis import BehaviorAnalysisModel
from ultratrack.ai_models.multi_modal import MultiModalFusionModel

# Export public API
__all__ = [
    # Model management
    'ModelRegistry', 'ModelInfo', 'ModelVersion', 'ModelStatus',
    'ModelEvaluation', 'ModelMetrics',
    
    # Version management
    'VersionManager', 'Version', 'ModelRollback', 'DeploymentStage',
    'ABTest', 'VersionComparisonResult',
    
    # High-level model interfaces
    'FaceRecognitionModel', 'PersonReIDModel', 'GaitAnalysisModel',
    'VoiceRecognitionModel', 'ObjectDetectionModel', 'BiometricProcessorsModel',
    'ThermalAnalysisModel', 'RFAnalysisModel', 'LPRModel',
    'BehaviorAnalysisModel', 'MultiModalFusionModel',
]

# Initialize model registry at import time
logger.debug("Initializing AI model registry...")

logger.debug("UltraTrack AI models module initialized")
