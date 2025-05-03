"""
UltraTrack gRPC API Module

This module provides gRPC API capabilities:
- High-performance API for system integration
- Protocol buffer definitions
- Service implementations
- Authentication and authorization

Copyright (c) 2025 Your Organization
"""

import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Import key components for easier access
from ultratrack.api.grpc.server import (
    GRPCServer, ServerConfig, ServerStatus, ServiceRegistration
)
from ultratrack.api.grpc.services import (
    TrackingService, AnalyticsService, AdminService,
    SearchService, NotificationService
)

# Export public API
__all__ = [
    # Server interfaces
    'GRPCServer', 'ServerConfig', 'ServerStatus', 'ServiceRegistration',
    
    # Service interfaces
    'TrackingService', 'AnalyticsService', 'AdminService',
    'SearchService', 'NotificationService',
]

logger.debug("UltraTrack gRPC API module initialized")
