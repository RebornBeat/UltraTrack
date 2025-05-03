"""
UltraTrack REST API Module

This module provides REST API capabilities:
- HTTP-based API for web and mobile clients
- Authentication and authorization
- API versioning
- Input validation
- Rate limiting

Copyright (c) 2025 Your Organization
"""

import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Import key components for easier access
from ultratrack.api.rest.server import (
    RESTAPIServer, ServerConfig, ServerStatus, APIMetrics
)
from ultratrack.api.rest.auth import (
    AuthHandler, AuthenticationResult, TokenValidation, 
    PermissionCheck, SessionManager
)
from ultratrack.api.rest.routes import (
    APIRouter, RouteRegistration, EndpointHandler,
    ResponseFormatter, ErrorHandler
)
from ultratrack.api.rest.validators import (
    RequestValidator, ValidationResult, SchemaValidator,
    InputSanitizer, ValidationError
)
from ultratrack.api.rest.rate_limiter import (
    RateLimiter, RateLimitResult, RateLimitPolicy,
    RateLimitQuota, QuotaRefreshStrategy
)

# Export public API
__all__ = [
    # Server interfaces
    'RESTAPIServer', 'ServerConfig', 'ServerStatus', 'APIMetrics',
    
    # Auth interfaces
    'AuthHandler', 'AuthenticationResult', 'TokenValidation', 
    'PermissionCheck', 'SessionManager',
    
    # Route interfaces
    'APIRouter', 'RouteRegistration', 'EndpointHandler',
    'ResponseFormatter', 'ErrorHandler',
    
    # Validator interfaces
    'RequestValidator', 'ValidationResult', 'SchemaValidator',
    'InputSanitizer', 'ValidationError',
    
    # Rate limiter interfaces
    'RateLimiter', 'RateLimitResult', 'RateLimitPolicy',
    'RateLimitQuota', 'QuotaRefreshStrategy',
]

logger.debug("UltraTrack REST API module initialized")
