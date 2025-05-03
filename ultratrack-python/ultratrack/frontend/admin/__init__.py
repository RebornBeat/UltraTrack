"""
UltraTrack Admin Module

This module provides administrative interfaces:
- User and role management
- System configuration
- Network management
- Audit review
- Data source management
- Compliance monitoring
- Access approval
- System health monitoring

Copyright (c) 2025 Your Organization
"""

import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Import key components for easier access
from ultratrack.frontend.admin.app import (
    AdminApp, AppConfig, UIState, EventHandler
)
from ultratrack.frontend.admin.user_management import (
    UserManagementUI, UserEditor, RoleEditor, PermissionAssignment,
    UserBulkOperations, AccessControl
)
from ultratrack.frontend.admin.system_config import (
    SystemConfigUI, ConfigEditor, ParameterValidation, ConfigProfile,
    ConfigHistory, ConfigDeployment
)
from ultratrack.frontend.admin.network_management import (
    NetworkManagementUI, NodeEditor, TopologyVisualization, ConnectionEditor,
    NetworkDiagnostics, NetworkMonitoring
)
from ultratrack.frontend.admin.audit_review import (
    AuditReviewUI, AuditViewer, AuditFilter, ComplianceReporting,
    AuditExport, AnomalyHighlighting
)
from ultratrack.frontend.admin.data_source_manager import (
    DataSourceManagerUI, SourceEditor, ConnectionTest, CredentialManager,
    DataPreview, BatchImport
)
from ultratrack.frontend.admin.compliance_dashboard import (
    ComplianceDashboardUI, ComplianceMetrics, RequirementStatus,
    ViolationReporting, RemediationTracking
)
from ultratrack.frontend.admin.access_approval import (
    AccessApprovalUI, RequestViewer, ApprovalWorkflow, JustificationReview,
    ApprovalLog, NotificationSettings
)
from ultratrack.frontend.admin.system_health import (
    SystemHealthUI, HealthDashboard, AlertConfiguration, PerfMetricsView,
    ResourceUtilization, MaintenanceScheduler
)

# Export public API
__all__ = [
    # Application interfaces
    'AdminApp', 'AppConfig', 'UIState', 'EventHandler',
    
    # User management interfaces
    'UserManagementUI', 'UserEditor', 'RoleEditor', 'PermissionAssignment',
    'UserBulkOperations', 'AccessControl',
    
    # System config interfaces
    'SystemConfigUI', 'ConfigEditor', 'ParameterValidation', 'ConfigProfile',
    'ConfigHistory', 'ConfigDeployment',
    
    # Network management interfaces
    'NetworkManagementUI', 'NodeEditor', 'TopologyVisualization', 'ConnectionEditor',
    'NetworkDiagnostics', 'NetworkMonitoring',
    
    # Audit review interfaces
    'AuditReviewUI', 'AuditViewer', 'AuditFilter', 'ComplianceReporting',
    'AuditExport', 'AnomalyHighlighting',
    
    # Data source interfaces
    'DataSourceManagerUI', 'SourceEditor', 'ConnectionTest', 'CredentialManager',
    'DataPreview', 'BatchImport',
    
    # Compliance dashboard interfaces
    'ComplianceDashboardUI', 'ComplianceMetrics', 'RequirementStatus',
    'ViolationReporting', 'RemediationTracking',
    
    # Access approval interfaces
    'AccessApprovalUI', 'RequestViewer', 'ApprovalWorkflow', 'JustificationReview',
    'ApprovalLog', 'NotificationSettings',
    
    # System health interfaces
    'SystemHealthUI', 'HealthDashboard', 'AlertConfiguration', 'PerfMetricsView',
    'ResourceUtilization', 'MaintenanceScheduler',
]

logger.debug("UltraTrack admin module initialized")
