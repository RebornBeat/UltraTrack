#!/usr/bin/env python3
"""
UltraTrack - Main Application Entry Point

This is the main entry point for the UltraTrack system, responsible for:
1. Parsing command-line arguments
2. Loading configuration
3. Setting up logging
4. Initializing system components
5. Starting services
6. Handling signals and graceful shutdown

Copyright (c) 2025 Your Organization
"""

import argparse
import logging
import os
import signal
import sys
import time
import threading
import traceback
from typing import Dict, List, Any, Optional, Set
from pathlib import Path

# Import early to ensure proper initialization
from ultratrack.config import ConfigManager, SystemConfig, Environment, ConfigValidationError

# Configure logging - will be reconfigured later with proper settings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Import remaining components after basic setup
from ultratrack import initialize_system, shutdown_system
from ultratrack.data_collection.camera_interface import CameraManager
from ultratrack.data_collection.audio_collector import AudioCollectionManager
from ultratrack.data_collection.thermal_collector import ThermalCollectionManager
from ultratrack.data_collection.rf_collector import RFCollectionManager
from ultratrack.ai_models.model_registry import ModelRegistry
from ultratrack.system_integration.tracking_engine import TrackingEngine
from ultratrack.compliance.audit_logger import AuditLogger
from ultratrack.api.rest.server import RESTAPIServer
from ultratrack.api.grpc.server import GRPCServer
from ultratrack.security.authentication import AuthenticationManager
from ultratrack.infrastructure.distributed_processing import DistributedProcessingManager
from ultratrack.infrastructure.node_health_monitor import NodeHealthMonitor

# Global state
running = True
exit_code = 0
services = []


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="UltraTrack Advanced Tracking System")
    
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--environment",
        "-e",
        type=str,
        choices=["dev", "test", "staging", "prod"],
        help="Environment (dev, test, staging, prod)"
    )
    
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    
    parser.add_argument(
        "--pid-file",
        type=str,
        help="PID file path"
    )
    
    parser.add_argument(
        "--role",
        type=str,
        choices=["worker", "coordinator", "edge", "storage", "analytics", "master"],
        help="Node role in the distributed system"
    )
    
    parser.add_argument(
        "--region",
        type=str,
        help="Geographic region for this node"
    )
    
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        help="Show version information and exit"
    )
    
    return parser.parse_args()


def setup_logging(config):
    """
    Set up logging based on configuration.
    
    Args:
        config: System configuration
    """
    # Get logging configuration
    log_config = config.logging
    
    # Determine log level
    log_level = getattr(logging, log_config.level)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        log_config.format,
        datefmt=log_config.date_format
    )
    
    # Add console handler if enabled
    if log_config.console_enabled:
        console_level = getattr(logging, log_config.console_level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if log_config.file_enabled:
        try:
            # Ensure directory exists
            log_dir = os.path.dirname(log_config.file_path)
            os.makedirs(log_dir, exist_ok=True)
            
            # Create rotating file handler
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_config.file_path,
                maxBytes=log_config.file_max_size_mb * 1024 * 1024,
                backupCount=log_config.file_backup_count
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"Failed to set up file logging: {str(e)}")
    
    # Add syslog handler if enabled
    if log_config.syslog_enabled:
        try:
            from logging.handlers import SysLogHandler
            syslog_handler = SysLogHandler(
                address=log_config.syslog_address,
                facility=getattr(SysLogHandler, f"LOG_{log_config.syslog_facility.upper()}")
            )
            syslog_handler.setLevel(log_level)
            syslog_handler.setFormatter(formatter)
            root_logger.addHandler(syslog_handler)
        except Exception as e:
            logger.error(f"Failed to set up syslog logging: {str(e)}")
    
    logger.info(f"Logging configured with level {log_config.level}")


def write_pid_file(pid_file_path):
    """
    Write PID to file.
    
    Args:
        pid_file_path: Path to PID file
    """
    try:
        with open(pid_file_path, 'w') as f:
            f.write(str(os.getpid()))
        logger.info(f"PID written to {pid_file_path}")
    except Exception as e:
        logger.error(f"Failed to write PID file: {str(e)}")


def remove_pid_file(pid_file_path):
    """
    Remove PID file.
    
    Args:
        pid_file_path: Path to PID file
    """
    try:
        if os.path.exists(pid_file_path):
            os.remove(pid_file_path)
            logger.info(f"PID file {pid_file_path} removed")
    except Exception as e:
        logger.error(f"Failed to remove PID file: {str(e)}")


def load_configuration(args):
    """
    Load system configuration from file and/or environment variables.
    
    Args:
        args: Command-line arguments
        
    Returns:
        SystemConfig: Loaded configuration
        
    Raises:
        ConfigValidationError: If configuration validation fails
    """
    try:
        # Load configuration
        config = ConfigManager.load(args.config, args.environment)
        
        # Override with command-line arguments
        if args.log_level:
            config.logging.level = args.log_level
            config.logging.console_level = args.log_level
        
        if args.log_file:
            config.logging.file_enabled = True
            config.logging.file_path = args.log_file
        
        if args.role:
            config.system_integration.node_role = args.role
        
        if args.region:
            config.system_integration.node_region = args.region
        
        # Validate configuration
        config.validate()
        
        return config
        
    except ConfigValidationError as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise ConfigValidationError(f"Configuration loading error: {str(e)}")


def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def handle_signal(sig, frame):
        global running
        if sig == signal.SIGINT or sig == signal.SIGTERM:
            logger.info(f"Received signal {sig}, initiating graceful shutdown...")
            running = False
        else:
            logger.info(f"Received signal {sig}, ignoring")
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGHUP, handle_signal)


def start_services(config):
    """
    Start system services.
    
    Args:
        config: System configuration
        
    Returns:
        list: Started services
    """
    global services
    
    logger.info("Starting UltraTrack services...")
    
    try:
        # Initialize audit logging first for proper tracking
        audit_logger = AuditLogger(config.compliance.audit)
        services.append(audit_logger)
        audit_logger.start()
        audit_logger.log_system_event("system_startup", 
                                     {"environment": config.environment.value})
        
        # Initialize authentication
        auth_manager = AuthenticationManager(config.compliance.authorization)
        services.append(auth_manager)
        auth_manager.start()
        
        # Initialize model registry
        model_registry = ModelRegistry(config.ai_models)
        services.append(model_registry)
        model_registry.start()
        
        # Initialize data collection services
        data_collection_services = []
        
        if config.data_collection.cameras.enabled:
            camera_manager = CameraManager(config.data_collection.cameras)
            services.append(camera_manager)
            data_collection_services.append(camera_manager)
            camera_manager.start()
        
        if config.data_collection.audio.enabled:
            audio_manager = AudioCollectionManager(config.data_collection.audio)
            services.append(audio_manager)
            data_collection_services.append(audio_manager)
            audio_manager.start()
        
        if config.data_collection.thermal.enabled:
            thermal_manager = ThermalCollectionManager(config.data_collection.thermal)
            services.append(thermal_manager)
            data_collection_services.append(thermal_manager)
            thermal_manager.start()
        
        if config.data_collection.rf.enabled:
            rf_manager = RFCollectionManager(config.data_collection.rf)
            services.append(rf_manager)
            data_collection_services.append(rf_manager)
            rf_manager.start()
        
        # Initialize tracking engine
        tracking_engine = TrackingEngine(
            config.system_integration.tracking,
            model_registry=model_registry,
            data_collection_services=data_collection_services
        )
        services.append(tracking_engine)
        tracking_engine.start()
        
        # Initialize distributed processing if enabled
        if config.system_integration.distributed_coordination_enabled:
            dist_processing = DistributedProcessingManager(
                config.system_integration,
                tracking_engine=tracking_engine
            )
            services.append(dist_processing)
            dist_processing.start()
        
        # Initialize health monitoring
        health_monitor = NodeHealthMonitor(
            config,
            services=services
        )
        services.append(health_monitor)
        health_monitor.start()
        
        # Initialize API servers if enabled
        if config.api.rest.enabled:
            rest_server = RESTAPIServer(
                config.api.rest,
                tracking_engine=tracking_engine,
                auth_manager=auth_manager
            )
            services.append(rest_server)
            rest_server.start()
        
        if config.api.grpc.enabled:
            grpc_server = GRPCServer(
                config.api.grpc,
                tracking_engine=tracking_engine,
                auth_manager=auth_manager
            )
            services.append(grpc_server)
            grpc_server.start()
        
        logger.info(f"Started {len(services)} services successfully")
        audit_logger.log_system_event("services_started", 
                                     {"service_count": len(services)})
        
        return services
        
    except Exception as e:
        logger.error(f"Failed to start services: {str(e)}", exc_info=True)
        shutdown_services(services)
        raise


def shutdown_services(services):
    """
    Shutdown services gracefully.
    
    Args:
        services: List of services to shut down
    """
    if not services:
        return
    
    logger.info(f"Shutting down {len(services)} services...")
    
    # Shutdown in reverse order
    for service in reversed(services):
        try:
            service_name = service.__class__.__name__
            logger.info(f"Shutting down {service_name}...")
            
            if hasattr(service, 'shutdown'):
                service.shutdown()
            elif hasattr(service, 'stop'):
                service.stop()
            else:
                logger.warning(f"No shutdown method found for {service_name}")
                
            logger.info(f"{service_name} shut down successfully")
            
        except Exception as e:
            logger.error(f"Error shutting down {service.__class__.__name__}: {str(e)}", 
                         exc_info=True)
    
    logger.info("All services shut down")


def main():
    """Main application entry point."""
    global running, exit_code, services
    
    args = parse_arguments()
    pid_file = args.pid_file
    
    try:
        # Show version if requested
        if args.version:
            from ultratrack import __version__
            print(f"UltraTrack version {__version__}")
            return 0
        
        # Load configuration
        config = load_configuration(args)
        
        # Exit after validation if requested
        if args.validate_config:
            print("Configuration validated successfully")
            return 0
        
        # Set up logging
        setup_logging(config)
        
        # Write PID file
        if pid_file:
            write_pid_file(pid_file)
        
        # Set up signal handlers
        setup_signal_handlers()
        
        logger.info("Starting UltraTrack system...")
        logger.info(f"Environment: {config.environment.value}")
        logger.info(f"Node name: {config.node_name}")
        logger.info(f"Node role: {config.system_integration.node_role}")
        logger.info(f"Node region: {config.system_integration.node_region}")
        
        # Start services
        services = start_services(config)
        
        logger.info("UltraTrack system started successfully")
        
        # Main loop
        while running:
            time.sleep(1)
        
        logger.info("Shutting down UltraTrack system...")
        
    except ConfigValidationError as e:
        logger.error(f"Configuration error: {str(e)}")
        exit_code = 1
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        exit_code = 1
    finally:
        # Shutdown services
        shutdown_services(services)
        
        # Remove PID file
        if pid_file:
            remove_pid_file(pid_file)
        
        logger.info(f"UltraTrack system shutdown complete, exit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
