"""
UltraTrack Frontend Module

This module provides user interface components:
- Dashboard for real-time tracking
- Administrative interfaces
- Visualization tools
- User management

Copyright (c) 2025 Your Organization
"""

import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Import key components for easier access
from ultratrack.frontend.dashboard import DashboardApp
from ultratrack.frontend.admin import AdminApp

# Export public API
__all__ = [
    'DashboardApp',
    'AdminApp',
]

logger.debug("UltraTrack frontend module initialized")
