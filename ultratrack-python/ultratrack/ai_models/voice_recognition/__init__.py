"""
UltraTrack Voice Recognition Module

This module provides voice recognition capabilities:
- Voice detection in audio streams
- Speaker identification
- Voice disguise detection
- Conversation analysis

Copyright (c) 2025 Your Organization
"""

import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Import key components for easier access
from ultratrack.ai_models.voice_recognition.voice_detector import (
    VoiceDetector, VoiceSegment, DetectionParameters, VoiceActivity,
    SpeechQuality, NoiseProfile
)
from ultratrack.ai_models.voice_recognition.speaker_recognizer import (
    SpeakerRecognizer, VoiceFeatures, MatchResult, RecognitionParameters,
    MatchConfidence, VoiceAttributes
)
from ultratrack.ai_models.voice_recognition.voice_disguise_detector import (
    DisguiseDetector, DisguiseType, DisguiseConfidence, 
    VoiceModification, NaturalVoiceEstimation
)
from ultratrack.ai_models.voice_recognition.conversation_analyzer import (
    ConversationAnalyzer, Conversation, SpeakerDiarization,
    SpeakerTurn, ConversationPattern
)

# Consolidated voice recognition model
class VoiceRecognitionModel:
    """
    Unified interface for voice recognition capabilities.
    
    This class provides a high-level interface to all voice recognition
    capabilities, including detection, speaker recognition, disguise detection,
    and conversation analysis.
    """
    
    def __init__(self, config=None):
        """Initialize the voice recognition model with the given configuration."""
        self.detector = VoiceDetector(config)
        self.recognizer = SpeakerRecognizer(config)
        self.disguise_detector = DisguiseDetector(config)
        self.conversation_analyzer = ConversationAnalyzer(config)
        
        logger.info("Voice recognition model initialized")
    
    def detect_and_recognize(self, audio, parameters=None):
        """
        Perform complete voice detection and speaker recognition.
        
        Args:
            audio: Input audio signal
            parameters: Optional processing parameters
            
        Returns:
            Voice recognition results with speaker identity matches
        """
        # Implementation details would go here
        pass

# Export public API
__all__ = [
    # Main model class
    'VoiceRecognitionModel',
    
    # Detector interfaces
    'VoiceDetector', 'VoiceSegment', 'DetectionParameters', 'VoiceActivity',
    'SpeechQuality', 'NoiseProfile',
    
    # Recognizer interfaces
    'SpeakerRecognizer', 'VoiceFeatures', 'MatchResult', 'RecognitionParameters',
    'MatchConfidence', 'VoiceAttributes',
    
    # Disguise detector interfaces
    'DisguiseDetector', 'DisguiseType', 'DisguiseConfidence', 
    'VoiceModification', 'NaturalVoiceEstimation',
    
    # Conversation analyzer interfaces
    'ConversationAnalyzer', 'Conversation', 'SpeakerDiarization',
    'SpeakerTurn', 'ConversationPattern',
]

logger.debug("UltraTrack voice recognition module initialized")
