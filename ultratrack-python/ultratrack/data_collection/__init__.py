"""
UltraTrack Data Collection Module

This module provides interfaces for collecting data from various sources:
- Camera networks (public, private, government, commercial)
- Audio collection systems
- Thermal imaging devices
- RF signal collectors
- Mobile data sources
- IoT devices
- Access control systems
- Satellite feeds

Copyright (c) 2025 Your Organization
"""

import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Import key components for easier access
from ultratrack.data_collection.camera_interface import (
    CameraManager, CameraSource, CameraType, StreamProtocol, 
    VideoFrame, FrameBatch
)
from ultratrack.data_collection.lpr_collector import (
    LPRCollector, LicensePlateDetection, LPRSource
)
from ultratrack.data_collection.iot_integrator import (
    IoTIntegrator, IoTDevice, IoTReading, IoTDeviceType
)
from ultratrack.data_collection.audio_collector import (
    AudioCollectionManager, AudioSource, AudioFrame, AudioSegment
)
from ultratrack.data_collection.thermal_collector import (
    ThermalCollectionManager, ThermalSource, ThermalFrame, TemperatureReading
)
from ultratrack.data_collection.mobile_data_connector import (
    MobileDataConnector, MobileDataType, AnonymizedLocation
)
from ultratrack.data_collection.network_discovery import (
    NetworkDiscovery, DiscoveredDevice, NetworkScan
)
from ultratrack.data_collection.rf_collector import (
    RFCollectionManager, RFSource, RFReading, SignalType
)
from ultratrack.data_collection.social_media_connector import (
    SocialMediaConnector, PublicProfile, SocialMediaPlatform
)
from ultratrack.data_collection.financial_data_connector import (
    FinancialDataConnector, TransactionType, AnonymizedTransaction
)
from ultratrack.data_collection.access_control_integrator import (
    AccessControlIntegrator, AccessEvent, AccessPoint
)
from ultratrack.data_collection.satellite_feed_connector import (
    SatelliteFeedConnector, SatelliteImage, ImageMetadata
)
from ultratrack.data_collection.data_validator import (
    DataValidator, ValidationRule, DataQualityMetrics, ValidationResult
)

# Export public API
__all__ = [
    # Camera interfaces
    'CameraManager', 'CameraSource', 'CameraType', 'StreamProtocol', 
    'VideoFrame', 'FrameBatch',
    
    # LPR interfaces
    'LPRCollector', 'LicensePlateDetection', 'LPRSource',
    
    # IoT interfaces
    'IoTIntegrator', 'IoTDevice', 'IoTReading', 'IoTDeviceType',
    
    # Audio interfaces
    'AudioCollectionManager', 'AudioSource', 'AudioFrame', 'AudioSegment',
    
    # Thermal interfaces
    'ThermalCollectionManager', 'ThermalSource', 'ThermalFrame', 'TemperatureReading',
    
    # Mobile data interfaces
    'MobileDataConnector', 'MobileDataType', 'AnonymizedLocation',
    
    # Network discovery interfaces
    'NetworkDiscovery', 'DiscoveredDevice', 'NetworkScan',
    
    # RF interfaces
    'RFCollectionManager', 'RFSource', 'RFReading', 'SignalType',
    
    # Social media interfaces
    'SocialMediaConnector', 'PublicProfile', 'SocialMediaPlatform',
    
    # Financial data interfaces
    'FinancialDataConnector', 'TransactionType', 'AnonymizedTransaction',
    
    # Access control interfaces
    'AccessControlIntegrator', 'AccessEvent', 'AccessPoint',
    
    # Satellite feed interfaces
    'SatelliteFeedConnector', 'SatelliteImage', 'ImageMetadata',
    
    # Data validation interfaces
    'DataValidator', 'ValidationRule', 'DataQualityMetrics', 'ValidationResult',
]

logger.debug("UltraTrack data collection module initialized")
