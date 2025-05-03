"""
UltraTrack API Module

This module provides external interfaces to the UltraTrack system:
- REST API for web/mobile clients
- gRPC API for high-performance integration
- API versioning and documentation

Copyright (c) 2025 Your Organization
"""

import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Import key components for easier access
from ultratrack.api.rest import RESTAPIServer
from ultratrack.api.grpc import GRPCServer

# Export public API
__all__ = [
    'RESTAPIServer',
    'GRPCServer',
]

logger.debug("UltraTrack API module initialized")
