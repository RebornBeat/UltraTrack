"""
Configuration management for UltraTrack system.

This module provides a robust configuration system with support for:
- Loading from environment variables
- Loading from YAML/JSON configuration files
- Validation of configuration parameters
- Environment-specific configurations (dev, test, prod)
- Secure handling of sensitive information
"""

import os
import sys
import json
import yaml
import logging
import socket
import ipaddress
import threading
import functools
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime, timedelta

# Setup logging
logger = logging.getLogger(__name__)

# Define available environments
class Environment(Enum):
    DEVELOPMENT = "dev"
    TESTING = "test"
    STAGING = "staging"
    PRODUCTION = "prod"
    
    @classmethod
    def from_string(cls, value: str) -> 'Environment':
        """Convert string to Environment enum."""
        value = value.lower()
        for env in cls:
            if env.value == value:
                return env
        raise ValueError(f"Invalid environment: {value}")


# Configuration validation exception
class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


# Configuration base class
@dataclass
class ConfigSection:
    """Base class for configuration sections."""
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ConfigValidationError: If validation fails
        """
        pass
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigSection':
        """
        Create a configuration section from a dictionary.
        
        Args:
            data: Dictionary containing configuration values
            
        Returns:
            ConfigSection: Initialized configuration section
        """
        # Filter out keys that aren't in the dataclass fields
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        
        # Handle nested sections
        for field_name, field_value in filtered_data.items():
            field_type = cls.__annotations__.get(field_name, None)
            
            # Check if field type is a ConfigSection subclass
            if (field_type and isinstance(field_value, dict) and 
                issubclass(field_type, ConfigSection)):
                filtered_data[field_name] = field_type.from_dict(field_value)
                
        return cls(**filtered_data)


# Database configuration
@dataclass
class DatabaseConfig(ConfigSection):
    """Database connection configuration."""
    host: str = "localhost"
    port: int = 5432
    username: str = "ultratrack"
    password: str = field(default="", repr=False)  # Sensitive - don't include in repr
    database: str = "ultratrack"
    pool_size: int = 20
    max_overflow: int = 10
    connect_timeout: int = 30
    ssl_mode: str = "prefer"
    
    def validate(self) -> None:
        """Validate database configuration."""
        if not self.host:
            raise ConfigValidationError("Database host cannot be empty")
        
        if not isinstance(self.port, int) or self.port < 1 or self.port > 65535:
            raise ConfigValidationError(f"Invalid database port: {self.port}")
        
        if not self.username:
            raise ConfigValidationError("Database username cannot be empty")
        
        if not self.database:
            raise ConfigValidationError("Database name cannot be empty")
        
        if self.pool_size < 1:
            raise ConfigValidationError(f"Invalid pool size: {self.pool_size}")
        
        if self.max_overflow < 0:
            raise ConfigValidationError(f"Invalid max overflow: {self.max_overflow}")
        
        if self.connect_timeout < 1:
            raise ConfigValidationError(f"Invalid connection timeout: {self.connect_timeout}")
        
        valid_ssl_modes = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
        if self.ssl_mode not in valid_ssl_modes:
            raise ConfigValidationError(f"Invalid SSL mode: {self.ssl_mode}")


# Data collection configuration sections
@dataclass
class CameraConfig(ConfigSection):
    """Camera integration configuration."""
    enabled: bool = True
    discovery_enabled: bool = True
    max_connections: int = 1000
    connection_timeout: int = 10
    reconnect_interval: int = 30
    frame_buffer_size: int = 30
    default_fps: int = 15
    rtsp_transport: str = "tcp"
    public_registry_url: str = "https://registry.cameras.ultratrack.org/api/v1"
    registry_api_key: str = field(default="", repr=False)
    local_storage_path: str = "/var/lib/ultratrack/camera_registry"
    
    def validate(self) -> None:
        """Validate camera configuration."""
        if self.max_connections < 1:
            raise ConfigValidationError(f"Invalid max connections: {self.max_connections}")
        
        if self.connection_timeout < 1:
            raise ConfigValidationError(f"Invalid connection timeout: {self.connection_timeout}")
        
        if self.reconnect_interval < 1:
            raise ConfigValidationError(f"Invalid reconnect interval: {self.reconnect_interval}")
        
        if self.frame_buffer_size < 1:
            raise ConfigValidationError(f"Invalid frame buffer size: {self.frame_buffer_size}")
        
        if self.default_fps < 1 or self.default_fps > 60:
            raise ConfigValidationError(f"Invalid default FPS: {self.default_fps}")
        
        if self.rtsp_transport not in {"tcp", "udp", "http", "https"}:
            raise ConfigValidationError(f"Invalid RTSP transport: {self.rtsp_transport}")
        
        if not os.path.isdir(os.path.dirname(self.local_storage_path)):
            try:
                os.makedirs(os.path.dirname(self.local_storage_path), exist_ok=True)
            except Exception as e:
                raise ConfigValidationError(f"Cannot create local storage path: {str(e)}")


@dataclass
class AudioConfig(ConfigSection):
    """Audio collection configuration."""
    enabled: bool = True
    sample_rate: int = 16000
    channels: int = 1
    format: str = "pcm"
    buffer_duration_ms: int = 500
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    noise_reduction_enabled: bool = True
    max_connections: int = 500
    
    def validate(self) -> None:
        """Validate audio configuration."""
        valid_sample_rates = {8000, 16000, 22050, 24000, 44100, 48000}
        if self.sample_rate not in valid_sample_rates:
            raise ConfigValidationError(f"Invalid sample rate: {self.sample_rate}")
        
        if self.channels < 1 or self.channels > 8:
            raise ConfigValidationError(f"Invalid channel count: {self.channels}")
        
        valid_formats = {"pcm", "wav", "mp3", "aac", "opus"}
        if self.format not in valid_formats:
            raise ConfigValidationError(f"Invalid audio format: {self.format}")
        
        if self.buffer_duration_ms < 10 or self.buffer_duration_ms > 5000:
            raise ConfigValidationError(f"Invalid buffer duration: {self.buffer_duration_ms}")
        
        if self.vad_threshold < 0.0 or self.vad_threshold > 1.0:
            raise ConfigValidationError(f"Invalid VAD threshold: {self.vad_threshold}")
        
        if self.max_connections < 1:
            raise ConfigValidationError(f"Invalid max connections: {self.max_connections}")


@dataclass
class ThermalConfig(ConfigSection):
    """Thermal imaging configuration."""
    enabled: bool = True
    temperature_range_min_c: float = -20.0
    temperature_range_max_c: float = 150.0
    resolution_normalization: bool = True
    target_resolution: Tuple[int, int] = (640, 480)
    frame_rate: int = 9
    color_palette: str = "iron"
    temperature_compensation: bool = True
    max_connections: int = 100
    
    def validate(self) -> None:
        """Validate thermal configuration."""
        if self.temperature_range_min_c >= self.temperature_range_max_c:
            raise ConfigValidationError("Minimum temperature must be less than maximum temperature")
        
        if self.target_resolution[0] < 32 or self.target_resolution[1] < 32:
            raise ConfigValidationError(f"Invalid target resolution: {self.target_resolution}")
        
        if self.frame_rate < 1 or self.frame_rate > 60:
            raise ConfigValidationError(f"Invalid frame rate: {self.frame_rate}")
        
        valid_palettes = {"iron", "rainbow", "white_hot", "black_hot", "arctic", "medical"}
        if self.color_palette not in valid_palettes:
            raise ConfigValidationError(f"Invalid color palette: {self.color_palette}")
        
        if self.max_connections < 1:
            raise ConfigValidationError(f"Invalid max connections: {self.max_connections}")


@dataclass
class RFConfig(ConfigSection):
    """RF signal collection configuration."""
    enabled: bool = True
    frequency_ranges_mhz: List[Tuple[float, float]] = field(
        default_factory=lambda: [(2400, 2500), (5150, 5850), (900, 930)]
    )
    collection_interval_ms: int = 100
    signal_threshold_dbm: float = -90.0
    device_fingerprinting: bool = True
    mac_address_collection: bool = True
    signal_strength_tracking: bool = True
    triangulation_enabled: bool = True
    min_receivers_for_location: int = 3
    max_connections: int = 50
    
    def validate(self) -> None:
        """Validate RF configuration."""
        for freq_range in self.frequency_ranges_mhz:
            if len(freq_range) != 2 or freq_range[0] >= freq_range[1]:
                raise ConfigValidationError(f"Invalid frequency range: {freq_range}")
            
            if freq_range[0] < 1 or freq_range[1] > 6000:
                raise ConfigValidationError(f"Frequency range out of bounds: {freq_range}")
        
        if self.collection_interval_ms < 10 or self.collection_interval_ms > 10000:
            raise ConfigValidationError(f"Invalid collection interval: {self.collection_interval_ms}")
        
        if self.signal_threshold_dbm > 0:
            raise ConfigValidationError(f"Signal threshold should be negative: {self.signal_threshold_dbm}")
        
        if self.min_receivers_for_location < 2:
            raise ConfigValidationError(f"At least 2 receivers needed for triangulation: {self.min_receivers_for_location}")
        
        if self.max_connections < 1:
            raise ConfigValidationError(f"Invalid max connections: {self.max_connections}")


@dataclass
class DataCollectionConfig(ConfigSection):
    """Data collection master configuration."""
    cameras: CameraConfig = field(default_factory=CameraConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    
    def validate(self) -> None:
        """Validate all data collection configurations."""
        self.cameras.validate()
        self.audio.validate()
        self.thermal.validate()
        self.rf.validate()


# AI model configurations
@dataclass
class ModelConfig(ConfigSection):
    """Base configuration for AI models."""
    enabled: bool = True
    model_path: str = ""
    precision: str = "fp16"
    batch_size: int = 16
    max_batch_latency_ms: int = 100
    device: str = "cuda:0"
    threads: int = 4
    cache_size: int = 1024
    
    def validate(self) -> None:
        """Validate model configuration."""
        if self.model_path and not os.path.exists(self.model_path):
            logger.warning(f"Model path does not exist: {self.model_path}")
        
        valid_precisions = {"fp32", "fp16", "int8", "int4"}
        if self.precision not in valid_precisions:
            raise ConfigValidationError(f"Invalid precision: {self.precision}")
        
        if self.batch_size < 1:
            raise ConfigValidationError(f"Invalid batch size: {self.batch_size}")
        
        if self.max_batch_latency_ms < 1:
            raise ConfigValidationError(f"Invalid max batch latency: {self.max_batch_latency_ms}")
        
        if not re.match(r'^(cpu|cuda(:\d+)?)$', self.device):
            raise ConfigValidationError(f"Invalid device specification: {self.device}")
        
        if self.threads < 1:
            raise ConfigValidationError(f"Invalid thread count: {self.threads}")
        
        if self.cache_size < 0:
            raise ConfigValidationError(f"Invalid cache size: {self.cache_size}")


@dataclass
class FaceRecognitionConfig(ModelConfig):
    """Face recognition model configuration."""
    detector_model: str = "ultrafast_face_detector_v2"
    recognition_model: str = "ultratrack_faceid_v3"
    detection_threshold: float = 0.65
    recognition_threshold: float = 0.75
    anti_spoofing_enabled: bool = True
    anti_spoofing_threshold: float = 0.95
    disguise_penetration_enabled: bool = True
    aging_compensation_enabled: bool = True
    max_yaw_angle: int = 45
    max_pitch_angle: int = 30
    min_face_size: int = 40
    
    def validate(self) -> None:
        """Validate face recognition configuration."""
        super().validate()
        
        if self.detection_threshold < 0.0 or self.detection_threshold > 1.0:
            raise ConfigValidationError(f"Invalid detection threshold: {self.detection_threshold}")
        
        if self.recognition_threshold < 0.0 or self.recognition_threshold > 1.0:
            raise ConfigValidationError(f"Invalid recognition threshold: {self.recognition_threshold}")
        
        if self.anti_spoofing_threshold < 0.0 or self.anti_spoofing_threshold > 1.0:
            raise ConfigValidationError(f"Invalid anti-spoofing threshold: {self.anti_spoofing_threshold}")
        
        if self.max_yaw_angle < 0 or self.max_yaw_angle > 90:
            raise ConfigValidationError(f"Invalid max yaw angle: {self.max_yaw_angle}")
        
        if self.max_pitch_angle < 0 or self.max_pitch_angle > 90:
            raise ConfigValidationError(f"Invalid max pitch angle: {self.max_pitch_angle}")
        
        if self.min_face_size < 10 or self.min_face_size > 1000:
            raise ConfigValidationError(f"Invalid min face size: {self.min_face_size}")


@dataclass
class PersonReIDConfig(ModelConfig):
    """Person re-identification model configuration."""
    model_name: str = "ultratrack_reid_v2"
    feature_dim: int = 2048
    matching_threshold: float = 0.70
    clothing_change_detection: bool = True
    clothing_change_threshold: float = 0.85
    partial_matching_enabled: bool = True
    partial_matching_min_overlap: float = 0.3
    temporal_consistency_weight: float = 0.4
    gallery_size: int = 10000
    
    def validate(self) -> None:
        """Validate person re-identification configuration."""
        super().validate()
        
        if self.feature_dim < 128 or self.feature_dim > 4096:
            raise ConfigValidationError(f"Invalid feature dimension: {self.feature_dim}")
        
        if self.matching_threshold < 0.0 or self.matching_threshold > 1.0:
            raise ConfigValidationError(f"Invalid matching threshold: {self.matching_threshold}")
        
        if self.clothing_change_threshold < 0.0 or self.clothing_change_threshold > 1.0:
            raise ConfigValidationError(f"Invalid clothing change threshold: {self.clothing_change_threshold}")
        
        if self.partial_matching_min_overlap < 0.0 or self.partial_matching_min_overlap > 1.0:
            raise ConfigValidationError(f"Invalid partial matching min overlap: {self.partial_matching_min_overlap}")
        
        if self.temporal_consistency_weight < 0.0 or self.temporal_consistency_weight > 1.0:
            raise ConfigValidationError(f"Invalid temporal consistency weight: {self.temporal_consistency_weight}")
        
        if self.gallery_size < 100:
            raise ConfigValidationError(f"Gallery size too small: {self.gallery_size}")


@dataclass
class GaitAnalysisConfig(ModelConfig):
    """Gait analysis model configuration."""
    model_name: str = "ultratrack_gait_v2"
    sequence_length: int = 30
    matching_threshold: float = 0.80
    min_sequence_quality: float = 0.6
    multi_angle_synthesis: bool = True
    view_angles: List[int] = field(default_factory=lambda: [0, 30, 60, 90, 120, 150, 180])
    stride_normalization: bool = True
    speed_invariant: bool = True
    injury_compensation: bool = True
    
    def validate(self) -> None:
        """Validate gait analysis configuration."""
        super().validate()
        
        if self.sequence_length < 10 or self.sequence_length > 120:
            raise ConfigValidationError(f"Invalid sequence length: {self.sequence_length}")
        
        if self.matching_threshold < 0.0 or self.matching_threshold > 1.0:
            raise ConfigValidationError(f"Invalid matching threshold: {self.matching_threshold}")
        
        if self.min_sequence_quality < 0.0 or self.min_sequence_quality > 1.0:
            raise ConfigValidationError(f"Invalid min sequence quality: {self.min_sequence_quality}")
        
        for angle in self.view_angles:
            if angle < 0 or angle > 180:
                raise ConfigValidationError(f"Invalid view angle: {angle}")


@dataclass
class VoiceRecognitionConfig(ModelConfig):
    """Voice recognition model configuration."""
    model_name: str = "ultratrack_voice_v1"
    sample_rate: int = 16000
    feature_type: str = "mfcc"
    num_features: int = 40
    matching_threshold: float = 0.75
    min_speech_duration_ms: int = 1000
    voice_activity_detection_enabled: bool = True
    voice_disguise_detection: bool = True
    disguise_detection_threshold: float = 0.8
    
    def validate(self) -> None:
        """Validate voice recognition configuration."""
        super().validate()
        
        valid_sample_rates = {8000, 16000, 22050, 24000, 44100, 48000}
        if self.sample_rate not in valid_sample_rates:
            raise ConfigValidationError(f"Invalid sample rate: {self.sample_rate}")
        
        valid_feature_types = {"mfcc", "fbank", "spectrogram", "melspectrogram"}
        if self.feature_type not in valid_feature_types:
            raise ConfigValidationError(f"Invalid feature type: {self.feature_type}")
        
        if self.num_features < 10 or self.num_features > 128:
            raise ConfigValidationError(f"Invalid number of features: {self.num_features}")
        
        if self.matching_threshold < 0.0 or self.matching_threshold > 1.0:
            raise ConfigValidationError(f"Invalid matching threshold: {self.matching_threshold}")
        
        if self.min_speech_duration_ms < 100 or self.min_speech_duration_ms > 10000:
            raise ConfigValidationError(f"Invalid min speech duration: {self.min_speech_duration_ms}")
        
        if self.disguise_detection_threshold < 0.0 or self.disguise_detection_threshold > 1.0:
            raise ConfigValidationError(f"Invalid disguise detection threshold: {self.disguise_detection_threshold}")


@dataclass
class ThermalAnalysisConfig(ModelConfig):
    """Thermal analysis model configuration."""
    model_name: str = "ultratrack_thermal_v1"
    detection_threshold: float = 0.65
    matching_threshold: float = 0.75
    temperature_normalization: bool = True
    baseline_temperature_c: float = 37.0
    temperature_variance_threshold_c: float = a2.0
    environment_compensation: bool = True
    
    def validate(self) -> None:
        """Validate thermal analysis configuration."""
        super().validate()
        
        if self.detection_threshold < 0.0 or self.detection_threshold > 1.0:
            raise ConfigValidationError(f"Invalid detection threshold: {self.detection_threshold}")
        
        if self.matching_threshold < 0.0 or self.matching_threshold > 1.0:
            raise ConfigValidationError(f"Invalid matching threshold: {self.matching_threshold}")
        
        if self.baseline_temperature_c < 25.0 or self.baseline_temperature_c > 45.0:
            raise ConfigValidationError(f"Invalid baseline temperature: {self.baseline_temperature_c}")
        
        if self.temperature_variance_threshold_c < 0.1 or self.temperature_variance_threshold_c > 10.0:
            raise ConfigValidationError(f"Invalid temperature variance threshold: {self.temperature_variance_threshold_c}")


@dataclass
class MultiModalFusionConfig(ModelConfig):
    """Multi-modal fusion configuration."""
    fusion_strategy: str = "weighted"
    modality_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "face": 1.0,
            "reid": 0.8,
            "gait": 0.7,
            "voice": 0.6,
            "thermal": 0.5,
            "rf": 0.4
        }
    )
    min_confidence_threshold: float = 0.65
    temporal_consistency_window: int = 5
    spatial_consistency_radius_m: float = 2.0
    conflict_resolution_strategy: str = "highest_confidence"
    evasion_countermeasures_enabled: bool = True
    suspicious_transition_threshold: float = 0.7
    
    def validate(self) -> None:
        """Validate multi-modal fusion configuration."""
        super().validate()
        
        valid_strategies = {"weighted", "average", "maximum", "bayesian", "learned"}
        if self.fusion_strategy not in valid_strategies:
            raise ConfigValidationError(f"Invalid fusion strategy: {self.fusion_strategy}")
        
        weight_sum = sum(self.modality_weights.values())
        if weight_sum <= 0:
            raise ConfigValidationError(f"Sum of modality weights must be positive: {weight_sum}")
        
        for modality, weight in self.modality_weights.items():
            if weight < 0.0:
                raise ConfigValidationError(f"Negative weight for modality {modality}: {weight}")
        
        if self.min_confidence_threshold < 0.0 or self.min_confidence_threshold > 1.0:
            raise ConfigValidationError(f"Invalid min confidence threshold: {self.min_confidence_threshold}")
        
        if self.temporal_consistency_window < 1:
            raise ConfigValidationError(f"Invalid temporal consistency window: {self.temporal_consistency_window}")
        
        if self.spatial_consistency_radius_m < 0.1 or self.spatial_consistency_radius_m > 100.0:
            raise ConfigValidationError(f"Invalid spatial consistency radius: {self.spatial_consistency_radius_m}")
        
        valid_conflict_strategies = {"highest_confidence", "majority_vote", "weighted_average", "most_recent"}
        if self.conflict_resolution_strategy not in valid_conflict_strategies:
            raise ConfigValidationError(f"Invalid conflict resolution strategy: {self.conflict_resolution_strategy}")
        
        if self.suspicious_transition_threshold < 0.0 or self.suspicious_transition_threshold > 1.0:
            raise ConfigValidationError(f"Invalid suspicious transition threshold: {self.suspicious_transition_threshold}")


@dataclass
class AIModelsConfig(ConfigSection):
    """AI models master configuration."""
    face_recognition: FaceRecognitionConfig = field(default_factory=FaceRecognitionConfig)
    person_reid: PersonReIDConfig = field(default_factory=PersonReIDConfig)
    gait_analysis: GaitAnalysisConfig = field(default_factory=GaitAnalysisConfig)
    voice_recognition: VoiceRecognitionConfig = field(default_factory=VoiceRecognitionConfig)
    thermal_analysis: ThermalAnalysisConfig = field(default_factory=ThermalAnalysisConfig)
    multi_modal_fusion: MultiModalFusionConfig = field(default_factory=MultiModalFusionConfig)
    model_cache_size_mb: int = 4096
    model_versioning_enabled: bool = True
    auto_update_models: bool = False
    model_repository_url: str = "https://models.ultratrack.org/api/v1"
    model_registry_path: str = "/var/lib/ultratrack/models"
    
    def validate(self) -> None:
        """Validate all AI model configurations."""
        self.face_recognition.validate()
        self.person_reid.validate()
        self.gait_analysis.validate()
        self.voice_recognition.validate()
        self.thermal_analysis.validate()
        self.multi_modal_fusion.validate()
        
        if self.model_cache_size_mb < 512:
            raise ConfigValidationError(f"Model cache size too small: {self.model_cache_size_mb}")
        
        if not os.path.isdir(os.path.dirname(self.model_registry_path)):
            try:
                os.makedirs(os.path.dirname(self.model_registry_path), exist_ok=True)
            except Exception as e:
                raise ConfigValidationError(f"Cannot create model registry path: {str(e)}")


# Tracking and integration configurations
@dataclass
class TrackingConfig(ConfigSection):
    """Tracking engine configuration."""
    max_active_tracks: int = 10000
    track_retention_period_s: int = 3600
    inactive_track_timeout_s: int = 300
    position_prediction_enabled: bool = True
    prediction_max_gap_s: float = 5.0
    spatial_indexing_enabled: bool = True
    spatial_index_cell_size_m: float = 50.0
    confidence_threshold: float = 0.6
    min_detections_for_track: int = 3
    max_position_uncertainty_m: float = 100.0
    blind_spot_inference: bool = True
    vehicle_transition_detection: bool = True
    building_transition_detection: bool = True
    transition_confidence_threshold: float = 0.8
    track_merge_threshold: float = 0.9
    track_persistence_strategy: str = "identity_based"
    
    def validate(self) -> None:
        """Validate tracking configuration."""
        if self.max_active_tracks < 100:
            raise ConfigValidationError(f"Max active tracks too small: {self.max_active_tracks}")
        
        if self.track_retention_period_s < 60:
            raise ConfigValidationError(f"Track retention period too short: {self.track_retention_period_s}")
        
        if self.inactive_track_timeout_s < 10:
            raise ConfigValidationError(f"Inactive track timeout too short: {self.inactive_track_timeout_s}")
        
        if self.prediction_max_gap_s < 0.1 or self.prediction_max_gap_s > 60.0:
            raise ConfigValidationError(f"Invalid prediction max gap: {self.prediction_max_gap_s}")
        
        if self.spatial_index_cell_size_m < 1.0 or self.spatial_index_cell_size_m > 1000.0:
            raise ConfigValidationError(f"Invalid spatial index cell size: {self.spatial_index_cell_size_m}")
        
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ConfigValidationError(f"Invalid confidence threshold: {self.confidence_threshold}")
        
        if self.min_detections_for_track < 1:
            raise ConfigValidationError(f"Min detections for track too small: {self.min_detections_for_track}")
        
        if self.max_position_uncertainty_m < 1.0:
            raise ConfigValidationError(f"Invalid max position uncertainty: {self.max_position_uncertainty_m}")
        
        if self.transition_confidence_threshold < 0.0 or self.transition_confidence_threshold > 1.0:
            raise ConfigValidationError(f"Invalid transition confidence threshold: {self.transition_confidence_threshold}")
        
        if self.track_merge_threshold < 0.0 or self.track_merge_threshold > 1.0:
            raise ConfigValidationError(f"Invalid track merge threshold: {self.track_merge_threshold}")
        
        valid_persistence_strategies = {"identity_based", "time_based", "hybrid"}
        if self.track_persistence_strategy not in valid_persistence_strategies:
            raise ConfigValidationError(f"Invalid track persistence strategy: {self.track_persistence_strategy}")


@dataclass
class GeospatialConfig(ConfigSection):
    """Geospatial mapping configuration."""
    coordinate_system: str = "wgs84"
    map_provider: str = "internal"
    map_tiles_url: str = "https://maps.ultratrack.org/tiles/{z}/{x}/{y}.png"
    map_api_key: str = field(default="", repr=False)
    map_cache_size_mb: int = 1024
    map_cache_path: str = "/var/cache/ultratrack/map_tiles"
    default_center_lat: float = 0.0
    default_center_lon: float = 0.0
    default_zoom_level: int = 10
    enable_3d: bool = True
    building_data_enabled: bool = True
    building_data_source: str = "internal"
    routing_engine_enabled: bool = True
    
    def validate(self) -> None:
        """Validate geospatial configuration."""
        valid_coordinate_systems = {"wgs84", "web_mercator", "utm", "local"}
        if self.coordinate_system not in valid_coordinate_systems:
            raise ConfigValidationError(f"Invalid coordinate system: {self.coordinate_system}")
        
        valid_map_providers = {"internal", "openstreetmap", "google", "bing", "mapbox", "custom"}
        if self.map_provider not in valid_map_providers:
            raise ConfigValidationError(f"Invalid map provider: {self.map_provider}")
        
        if self.map_cache_size_mb < 10:
            raise ConfigValidationError(f"Map cache size too small: {self.map_cache_size_mb}")
        
        if self.default_zoom_level < 1 or self.default_zoom_level > 20:
            raise ConfigValidationError(f"Invalid default zoom level: {self.default_zoom_level}")
        
        valid_building_data_sources = {"internal", "openstreetmap", "google", "custom"}
        if self.building_data_enabled and self.building_data_source not in valid_building_data_sources:
            raise ConfigValidationError(f"Invalid building data source: {self.building_data_source}")
        
        if not os.path.isdir(os.path.dirname(self.map_cache_path)):
            try:
                os.makedirs(os.path.dirname(self.map_cache_path), exist_ok=True)
            except Exception as e:
                raise ConfigValidationError(f"Cannot create map cache path: {str(e)}")


@dataclass
class HandoffConfig(ConfigSection):
    """Tracking handoff configuration."""
    enabled: bool = True
    protocol_version: str = "1.2"
    authentication_required: bool = True
    encryption_enabled: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_interval_hours: int = 24
    handoff_timeout_s: int = a30
    retry_interval_s: int = 5
    max_retries: int = 3
    connection_pool_size: int = 20
    trusted_nodes: List[str] = field(default_factory=list)
    
    def validate(self) -> None:
        """Validate handoff configuration."""
        valid_protocol_versions = {"1.0", "1.1", "1.2"}
        if self.protocol_version not in valid_protocol_versions:
            raise ConfigValidationError(f"Invalid protocol version: {self.protocol_version}")
        
        valid_encryption_algorithms = {"AES-256-GCM", "ChaCha20-Poly1305"}
        if self.encryption_enabled and self.encryption_algorithm not in valid_encryption_algorithms:
            raise ConfigValidationError(f"Invalid encryption algorithm: {self.encryption_algorithm}")
        
        if self.key_rotation_interval_hours < 1:
            raise ConfigValidationError(f"Key rotation interval too short: {self.key_rotation_interval_hours}")
        
        if self.handoff_timeout_s < 1:
            raise ConfigValidationError(f"Handoff timeout too short: {self.handoff_timeout_s}")
        
        if self.retry_interval_s < 1:
            raise ConfigValidationError(f"Retry interval too short: {self.retry_interval_s}")
        
        if self.max_retries < 0:
            raise ConfigValidationError(f"Invalid max retries: {self.max_retries}")
        
        if self.connection_pool_size < 1:
            raise ConfigValidationError(f"Connection pool size too small: {self.connection_pool_size}")
        
        for node in self.trusted_nodes:
            try:
                host, port = node.split(":")
                ipaddress.ip_address(host)  # Will raise ValueError if not a valid IP
                port_num = int(port)
                if port_num < 1 or port_num > 65535:
                    raise ValueError(f"Invalid port number: {port}")
            except ValueError:
                raise ConfigValidationError(f"Invalid trusted node address: {node}")


@dataclass
class SystemIntegrationConfig(ConfigSection):
    """System integration master configuration."""
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    geospatial: GeospatialConfig = field(default_factory=GeospatialConfig)
    handoff: HandoffConfig = field(default_factory=HandoffConfig)
    distributed_coordination_enabled: bool = True
    coordination_mode: str = "consensus"
    node_id: str = socket.gethostname()
    node_role: str = "worker"
    node_region: str = "default"
    global_coordinator_url: str = "https://coordinator.ultratrack.org/api/v1"
    coordinator_api_key: str = field(default="", repr=False)
    
    def validate(self) -> None:
        """Validate system integration configuration."""
        self.tracking.validate()
        self.geospatial.validate()
        self.handoff.validate()
        
        valid_coordination_modes = {"consensus", "leader_follower", "gossip", "hierarchical"}
        if self.distributed_coordination_enabled and self.coordination_mode not in valid_coordination_modes:
            raise ConfigValidationError(f"Invalid coordination mode: {self.coordination_mode}")
        
        valid_node_roles = {"worker", "coordinator", "edge", "storage", "analytics", "master"}
        if self.node_role not in valid_node_roles:
            raise ConfigValidationError(f"Invalid node role: {self.node_role}")
        
        if not self.node_id:
            raise ConfigValidationError("Node ID cannot be empty")
        
        if not self.node_region:
            raise ConfigValidationError("Node region cannot be empty")


# Compliance and security configurations
@dataclass
class AuditConfig(ConfigSection):
    """Audit logging configuration."""
    enabled: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    log_path: str = "/var/log/ultratrack/audit"
    max_log_size_mb: int = 100
    max_log_files: int = 30
    log_rotation_interval_hours: int = 24
    secure_logging: bool = True
    tamper_detection: bool = True
    include_user_info: bool = True
    include_source_ip: bool = True
    include_timestamp: bool = True
    include_operation: bool = True
    include_resource: bool = True
    include_outcome: bool = True
    
    def validate(self) -> None:
        """Validate audit configuration."""
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_log_levels:
            raise ConfigValidationError(f"Invalid log level: {self.log_level}")
        
        valid_log_formats = {"json", "syslog", "csv", "xml"}
        if self.log_format not in valid_log_formats:
            raise ConfigValidationError(f"Invalid log format: {self.log_format}")
        
        if self.max_log_size_mb < 1:
            raise ConfigValidationError(f"Max log size too small: {self.max_log_size_mb}")
        
        if self.max_log_files < 1:
            raise ConfigValidationError(f"Max log files too small: {self.max_log_files}")
        
        if self.log_rotation_interval_hours < 1:
            raise ConfigValidationError(f"Log rotation interval too short: {self.log_rotation_interval_hours}")
        
        if not os.path.isdir(os.path.dirname(self.log_path)):
            try:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            except Exception as e:
                raise ConfigValidationError(f"Cannot create log directory: {str(e)}")


@dataclass
class AuthorizationConfig(ConfigSection):
    """Authorization configuration."""
    required: bool = True
    token_expiration_hours: int = 12
    refresh_token_expiration_days: int = 30
    max_sessions_per_user: int = 5
    multi_factor_required: bool = True
    multi_factor_methods: List[str] = field(
        default_factory=lambda: ["totp", "email", "sms"]
    )
    password_min_length: int = 12
    password_require_mixed_case: bool = True
    password_require_numbers: bool = True
    password_require_special: bool = True
    password_max_age_days: int = 90
    account_lockout_threshold: int = 5
    account_lockout_duration_minutes: int = 30
    role_based_access: bool = True
    
    def validate(self) -> None:
        """Validate authorization configuration."""
        if self.token_expiration_hours < 1:
            raise ConfigValidationError(f"Token expiration too short: {self.token_expiration_hours}")
        
        if self.refresh_token_expiration_days < 1:
            raise ConfigValidationError(f"Refresh token expiration too short: {self.refresh_token_expiration_days}")
        
        if self.max_sessions_per_user < 1:
            raise ConfigValidationError(f"Max sessions per user too small: {self.max_sessions_per_user}")
        
        valid_mfa_methods = {"totp", "email", "sms", "push", "hardware_token", "biometric"}
        for method in self.multi_factor_methods:
            if method not in valid_mfa_methods:
                raise ConfigValidationError(f"Invalid multi-factor method: {method}")
        
        if self.password_min_length < 8:
            raise ConfigValidationError(f"Password min length too small: {self.password_min_length}")
        
        if self.password_max_age_days < 1:
            raise ConfigValidationError(f"Password max age too short: {self.password_max_age_days}")
        
        if self.account_lockout_threshold < 1:
            raise ConfigValidationError(f"Account lockout threshold too small: {self.account_lockout_threshold}")
        
        if self.account_lockout_duration_minutes < 1:
            raise ConfigValidationError(f"Account lockout duration too short: {self.account_lockout_duration_minutes}")


@dataclass
class ComplianceConfig(ConfigSection):
    """Compliance master configuration."""
    audit: AuditConfig = field(default_factory=AuditConfig)
    authorization: AuthorizationConfig = field(default_factory=AuthorizationConfig)
    data_retention_days: int = 90
    automatic_deletion_enabled: bool = True
    automatic_anonymization_enabled: bool = True
    purpose_validation_required: bool = True
    jurisdiction_aware: bool = True
    warrant_validation_required: bool = True
    warrant_database_path: str = "/var/lib/ultratrack/warrants"
    privacy_by_default: bool = True
    privacy_impact_assessment_required: bool = True
    
    def validate(self) -> None:
        """Validate compliance configuration."""
        self.audit.validate()
        self.authorization.validate()
        
        if self.data_retention_days < 1:
            raise ConfigValidationError(f"Data retention days too small: {self.data_retention_days}")
        
        if not os.path.isdir(os.path.dirname(self.warrant_database_path)):
            try:
                os.makedirs(os.path.dirname(self.warrant_database_path), exist_ok=True)
            except Exception as e:
                raise ConfigValidationError(f"Cannot create warrant database path: {str(e)}")


# API configurations
@dataclass
class RESTAPIConfig(ConfigSection):
    """REST API configuration."""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    base_path: str = "/api/v1"
    ssl_enabled: bool = True
    ssl_cert_path: str = "/etc/ultratrack/ssl/server.crt"
    ssl_key_path: str = "/etc/ultratrack/ssl/server.key"
    workers: int = 4
    request_timeout_s: int = 30
    max_request_size_mb: int = 10
    rate_limiting_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window_s: int = 60
    cors_enabled: bool = True
    cors_allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    
    def validate(self) -> None:
        """Validate REST API configuration."""
        if not self.host:
            raise ConfigValidationError("API host cannot be empty")
        
        if not isinstance(self.port, int) or self.port < 1 or self.port > 65535:
            raise ConfigValidationError(f"Invalid API port: {self.port}")
        
        if self.ssl_enabled:
            if not os.path.exists(self.ssl_cert_path):
                raise ConfigValidationError(f"SSL certificate file not found: {self.ssl_cert_path}")
            
            if not os.path.exists(self.ssl_key_path):
                raise ConfigValidationError(f"SSL key file not found: {self.ssl_key_path}")
        
        if self.workers < 1:
            raise ConfigValidationError(f"Invalid worker count: {self.workers}")
        
        if self.request_timeout_s < 1:
            raise ConfigValidationError(f"Request timeout too short: {self.request_timeout_s}")
        
        if self.max_request_size_mb < 1:
            raise ConfigValidationError(f"Max request size too small: {self.max_request_size_mb}")
        
        if self.rate_limiting_enabled:
            if self.rate_limit_requests < 1:
                raise ConfigValidationError(f"Rate limit requests too small: {self.rate_limit_requests}")
            
            if self.rate_limit_window_s < 1:
                raise ConfigValidationError(f"Rate limit window too short: {self.rate_limit_window_s}")


@dataclass
class GRPCConfig(ConfigSection):
    """gRPC API configuration."""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 10
    max_message_size_mb: int = 100
    ssl_enabled: bool = True
    ssl_cert_path: str = "/etc/ultratrack/ssl/server.crt"
    ssl_key_path: str = "/etc/ultratrack/ssl/server.key"
    reflection_enabled: bool = True
    compression_enabled: bool = True
    keepalive_time_ms: int = 7200000  # 2 hours
    keepalive_timeout_ms: int = 20000  # 20 seconds
    
    def validate(self) -> None:
        """Validate gRPC configuration."""
        if not self.host:
            raise ConfigValidationError("gRPC host cannot be empty")
        
        if not isinstance(self.port, int) or self.port < 1 or self.port > 65535:
            raise ConfigValidationError(f"Invalid gRPC port: {self.port}")
        
        if self.max_workers < 1:
            raise ConfigValidationError(f"Max workers too small: {self.max_workers}")
        
        if self.max_message_size_mb < 1:
            raise ConfigValidationError(f"Max message size too small: {self.max_message_size_mb}")
        
        if self.ssl_enabled:
            if not os.path.exists(self.ssl_cert_path):
                raise ConfigValidationError(f"SSL certificate file not found: {self.ssl_cert_path}")
            
            if not os.path.exists(self.ssl_key_path):
                raise ConfigValidationError(f"SSL key file not found: {self.ssl_key_path}")
        
        if self.keepalive_time_ms < 1000:
            raise ConfigValidationError(f"Keepalive time too short: {self.keepalive_time_ms}")
        
        if self.keepalive_timeout_ms < 1000:
            raise ConfigValidationError(f"Keepalive timeout too short: {self.keepalive_timeout_ms}")


@dataclass
class APIConfig(ConfigSection):
    """API master configuration."""
    rest: RESTAPIConfig = field(default_factory=RESTAPIConfig)
    grpc: GRPCConfig = field(default_factory=GRPCConfig)
    
    def validate(self) -> None:
        """Validate API configurations."""
        self.rest.validate()
        self.grpc.validate()
        
        # Check for port conflicts
        if self.rest.enabled and self.grpc.enabled and self.rest.port == self.grpc.port:
            raise ConfigValidationError(f"REST and gRPC APIs cannot use the same port: {self.rest.port}")


# Logging configuration
@dataclass
class LoggingConfig(ConfigSection):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_enabled: bool = True
    file_path: str = "/var/log/ultratrack/system.log"
    file_max_size_mb: int = 100
    file_backup_count: int = 10
    console_enabled: bool = True
    console_level: str = "INFO"
    json_format: bool = False
    syslog_enabled: bool = False
    syslog_facility: str = "local0"
    syslog_address: str = "/dev/log"
    
    def validate(self) -> None:
        """Validate logging configuration."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level not in valid_levels:
            raise ConfigValidationError(f"Invalid logging level: {self.level}")
        
        if self.console_enabled and self.console_level not in valid_levels:
            raise ConfigValidationError(f"Invalid console logging level: {self.console_level}")
        
        if self.file_enabled:
            if self.file_max_size_mb < 1:
                raise ConfigValidationError(f"File max size too small: {self.file_max_size_mb}")
            
            if self.file_backup_count < 1:
                raise ConfigValidationError(f"File backup count too small: {self.file_backup_count}")
            
            log_dir = os.path.dirname(self.file_path)
            if not os.path.isdir(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except Exception as e:
                    raise ConfigValidationError(f"Cannot create log directory: {str(e)}")
        
        if self.syslog_enabled:
            valid_facilities = {
                "auth", "authpriv", "cron", "daemon", "ftp", "kern",
                "lpr", "mail", "news", "syslog", "user", "uucp",
                "local0", "local1", "local2", "local3",
                "local4", "local5", "local6", "local7"
            }
            if self.syslog_facility not in valid_facilities:
                raise ConfigValidationError(f"Invalid syslog facility: {self.syslog_facility}")


# System configuration class
@dataclass
class SystemConfig(ConfigSection):
    """Complete system configuration."""
    environment: Environment = Environment.PRODUCTION
    node_name: str = socket.gethostname()
    data_directory: str = "/var/lib/ultratrack"
    temp_directory: str = "/tmp/ultratrack"
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    data_collection: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    ai_models: AIModelsConfig = field(default_factory=AIModelsConfig)
    system_integration: SystemIntegrationConfig = field(default_factory=SystemIntegrationConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def validate(self) -> None:
        """Validate complete system configuration."""
        if not self.node_name:
            raise ConfigValidationError("Node name cannot be empty")
        
        if not os.path.isdir(self.data_directory):
            try:
                os.makedirs(self.data_directory, exist_ok=True)
            except Exception as e:
                raise ConfigValidationError(f"Cannot create data directory: {str(e)}")
        
        if not os.path.isdir(self.temp_directory):
            try:
                os.makedirs(self.temp_directory, exist_ok=True)
            except Exception as e:
                raise ConfigValidationError(f"Cannot create temp directory: {str(e)}")
        
        # Validate all sections
        self.database.validate()
        self.data_collection.validate()
        self.ai_models.validate()
        self.system_integration.validate()
        self.compliance.validate()
        self.api.validate()
        self.logging.validate()


# Configuration manager
class ConfigManager:
    """Manages loading and validation of system configuration."""
    
    _instance = None
    _lock = threading.RLock()
    _config = None
    
    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        """
        Get the singleton instance of ConfigManager.
        
        Returns:
            ConfigManager: The singleton instance
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    @classmethod
    def load(cls, config_path: Optional[str] = None, 
             environment: Optional[str] = None) -> SystemConfig:
        """
        Load configuration from file and/or environment variables.
        
        Args:
            config_path: Path to configuration file
            environment: Environment name (dev, test, prod)
            
        Returns:
            SystemConfig: Loaded and validated configuration
            
        Raises:
            ConfigValidationError: If configuration validation fails
        """
        instance = cls.get_instance()
        instance._config = instance._load_config(config_path, environment)
        return instance._config
    
    @classmethod
    def get_config(cls) -> SystemConfig:
        """
        Get the current configuration.
        
        Returns:
            SystemConfig: Current configuration
            
        Raises:
            RuntimeError: If configuration has not been loaded
        """
        instance = cls.get_instance()
        if instance._config is None:
            raise RuntimeError("Configuration has not been loaded")
        return instance._config
    
    def _load_config(self, config_path: Optional[str] = None,
                    environment: Optional[str] = None) -> SystemConfig:
        """
        Load configuration from file and/or environment variables.
        
        Args:
            config_path: Path to configuration file
            environment: Environment name (dev, test, prod)
            
        Returns:
            SystemConfig: Loaded and validated configuration
            
        Raises:
            ConfigValidationError: If configuration validation fails
        """
        # Start with default configuration
        config = SystemConfig()
        
        # Set environment if specified
        if environment:
            config.environment = Environment.from_string(environment)
        
        # Load from configuration file if specified
        if config_path:
            file_config = self._load_from_file(config_path)
            config = self._merge_configs(config, file_config)
        
        # Load from environment variables
        env_config = self._load_from_env()
        config = self._merge_configs(config, env_config)
        
        # Environment-specific overrides
        config = self._apply_environment_overrides(config)
        
        # Validate configuration
        config.validate()
        
        return config
    
    def _load_from_file(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            dict: Loaded configuration
            
        Raises:
            ConfigValidationError: If file cannot be loaded
        """
        if not os.path.exists(config_path):
            raise ConfigValidationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    return json.load(f)
                else:
                    raise ConfigValidationError(
                        f"Unsupported configuration file format: {config_path}"
                    )
        except Exception as e:
            raise ConfigValidationError(f"Error loading configuration file: {str(e)}")
    
    def _load_from_env(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Returns:
            dict: Loaded configuration
        """
        # Environment variables should be prefixed with ULTRATRACK_
        prefix = "ULTRATRACK_"
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                key = key[len(prefix):].lower()
                
                # Split by double underscore to create nested dictionaries
                parts = key.split('__')
                
                # Navigate to the correct nested dictionary
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set the value, converting to appropriate type
                current[parts[-1]] = self._convert_env_value(value)
        
        return config
    
    def _convert_env_value(self, value: str) -> Any:
        """
        Convert environment variable string to appropriate type.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Converted value of appropriate type
        """
        # Try to convert to int
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Handle boolean values
        if value.lower() in {'true', 'yes', '1'}:
            return True
        if value.lower() in {'false', 'no', '0'}:
            return False
        
        # Handle lists (comma-separated values)
        if ',' in value:
            return [self._convert_env_value(v.strip()) for v in value.split(',')]
        
        # Default to string
        return value
    
    def _merge_configs(self, base: Any, override: Dict[str, Any]) -> Any:
        """
        Recursively merge configurations.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        if isinstance(base, ConfigSection):
            # Convert to dictionary for merging
            base_dict = asdict(base)
            # Merge dictionaries
            merged = self._merge_configs(base_dict, override)
            # Convert back to the original type
            return type(base).from_dict(merged)
        
        elif isinstance(base, dict) and isinstance(override, dict):
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], (dict, ConfigSection)):
                    result[key] = self._merge_configs(result[key], value)
                else:
                    result[key] = value
            return result
        
        else:
            # Override base with override
            return override
    
    def _apply_environment_overrides(self, config: SystemConfig) -> SystemConfig:
        """
        Apply environment-specific overrides.
        
        Args:
            config: Current configuration
            
        Returns:
            Updated configuration
        """
        # Development environment adjustments
        if config.environment == Environment.DEVELOPMENT:
            # Use smaller cache sizes in development
            config.ai_models.model_cache_size_mb = min(config.ai_models.model_cache_size_mb, 1024)
            config.system_integration.geospatial.map_cache_size_mb = min(
                config.system_integration.geospatial.map_cache_size_mb, 256
            )
            
            # Shorter token expiration for faster testing
            config.compliance.authorization.token_expiration_hours = 1
            
            # More verbose logging
            config.logging.level = "DEBUG"
            config.logging.console_level = "DEBUG"
            
            # Disable SSL for easier local development
            config.api.rest.ssl_enabled = False
            config.api.grpc.ssl_enabled = False
        
        # Testing environment adjustments
        elif config.environment == Environment.TESTING:
            # More verbose logging
            config.logging.level = "DEBUG"
            
            # Shorter retention periods
            config.compliance.data_retention_days = 7
            
            # Relaxed security for testing
            config.compliance.authorization.multi_factor_required = False
            
            # Faster timeout for tests
            config.database.connect_timeout = 5
            config.api.rest.request_timeout_s = 5
        
        # Production environment adjustments
        elif config.environment == Environment.PRODUCTION:
            # Ensure security features are enabled
            config.api.rest.ssl_enabled = True
            config.api.grpc.ssl_enabled = True
            config.compliance.authorization.multi_factor_required = True
            config.system_integration.handoff.encryption_enabled = True
        
        return config


# Helper function to get dataclass fields
def fields(cls):
    """Get fields of a dataclass."""
    return getattr(cls, '__dataclass_fields__', {}).values()
