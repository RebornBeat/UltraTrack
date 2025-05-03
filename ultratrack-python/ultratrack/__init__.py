"""
UltraTrack - Advanced Omnipresence Tracking System

This package provides a comprehensive tracking system capable of tracking
individuals, vehicles, and objects across unlimited distances using
multi-modal identification methods and global coordination.

Copyright (c) 2025 Your Organization
"""

__version__ = "1.0.0"
__author__ = "Your Organization"
__license__ = "MIT"

import logging
import os
import sys

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Import key components for easier access at package level
from ultratrack.config import ConfigManager, SystemConfig
from ultratrack.data_collection.camera_interface import CameraManager
from ultratrack.data_collection.audio_collector import AudioCollectionManager
from ultratrack.data_collection.thermal_collector import ThermalCollectionManager
from ultratrack.data_collection.rf_collector import RFCollectionManager
from ultratrack.ai_models.model_registry import ModelRegistry
from ultratrack.system_integration.tracking_engine import TrackingEngine
from ultratrack.compliance.audit_logger import AuditLogger
from ultratrack.api.rest.server import RESTAPIServer
from ultratrack.api.grpc.server import GRPCServer

# Setup tracking configuration
logger = logging.getLogger(__name__)

# Initialize package resources
_package_root = os.path.dirname(os.path.abspath(__file__))
_resource_path = os.path.join(_package_root, "resources")

# Export important symbols
__all__ = [
    "ConfigManager",
    "SystemConfig",
    "CameraManager",
    "AudioCollectionManager",
    "ThermalCollectionManager",
    "RFCollectionManager",
    "ModelRegistry",
    "TrackingEngine",
    "AuditLogger",
    "RESTAPIServer",
    "GRPCServer",
    "initialize_system",
    "shutdown_system"
]

# System lifecycle management
_initialized = False
_system_components = []

def initialize_system(config_path=None, environment=None):
    """
    Initialize the UltraTrack system with specified configuration.
    
    Args:
        config_path (str, optional): Path to configuration file
        environment (str, optional): Environment name (dev, test, prod)
        
    Returns:
        bool: True if initialization successful
    """
    global _initialized, _system_components
    
    if _initialized:
        logger.warning("System already initialized")
        return True
    
    try:
        # Load configuration
        config = ConfigManager.load(config_path, environment)
        
        # Initialize audit logging first
        audit_logger = AuditLogger(config.compliance.audit)
        _system_components.append(audit_logger)
        audit_logger.log_system_event("system_initialization_started", 
                                     {"environment": environment or "default"})
                                     
        # Initialize model registry
        model_registry = ModelRegistry(config.ai_models)
        _system_components.append(model_registry)
        
        # Initialize data collection subsystems
        camera_manager = CameraManager(config.data_collection.cameras)
        _system_components.append(camera_manager)
        
        audio_manager = AudioCollectionManager(config.data_collection.audio)
        _system_components.append(audio_manager)
        
        thermal_manager = ThermalCollectionManager(config.data_collection.thermal)
        _system_components.append(thermal_manager)
        
        rf_manager = RFCollectionManager(config.data_collection.rf)
        _system_components.append(rf_manager)
        
        # Initialize the tracking engine
        tracking_engine = TrackingEngine(
            config.system_integration.tracking,
            model_registry=model_registry,
            camera_manager=camera_manager,
            audio_manager=audio_manager,
            thermal_manager=thermal_manager,
            rf_manager=rf_manager
        )
        _system_components.append(tracking_engine)
        
        # Initialize API servers if enabled
        if config.api.rest.enabled:
            rest_server = RESTAPIServer(
                config.api.rest,
                tracking_engine=tracking_engine
            )
            _system_components.append(rest_server)
            
        if config.api.grpc.enabled:
            grpc_server = GRPCServer(
                config.api.grpc,
                tracking_engine=tracking_engine
            )
            _system_components.append(grpc_server)
            
        # Mark system as initialized
        _initialized = True
        logger.info("UltraTrack system initialized successfully")
        audit_logger.log_system_event("system_initialization_completed", 
                                     {"status": "success"})
        return True
        
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}", exc_info=True)
        # Attempt to shut down any components that did start
        shutdown_system()
        return False

def shutdown_system():
    """
    Gracefully shut down the UltraTrack system.
    
    Returns:
        bool: True if shutdown successful
    """
    global _initialized, _system_components
    
    if not _initialized:
        logger.warning("System not initialized, nothing to shut down")
        return True
    
    logger.info("Shutting down UltraTrack system...")
    
    # Shut down components in reverse order
    shutdown_successful = True
    for component in reversed(_system_components):
        try:
            if hasattr(component, 'shutdown'):
                logger.info(f"Shutting down {component.__class__.__name__}")
                component.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down {component.__class__.__name__}: {str(e)}", 
                         exc_info=True)
            shutdown_successful = False
    
    # Clear component list
    _system_components = []
    _initialized = False
    
    if shutdown_successful:
        logger.info("UltraTrack system shutdown completed successfully")
    else:
        logger.warning("UltraTrack system shutdown completed with errors")
    
    return shutdown_successful
