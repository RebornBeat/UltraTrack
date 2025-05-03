"""
UltraTrack Compliance Module

This module provides components for ensuring proper system usage:
- Privacy protection
- Comprehensive audit logging
- Consent management
- Data lifecycle enforcement
- Legal compliance
- Authorization verification
- Ethical oversight

Copyright (c) 2025 Your Organization
"""

import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Import key components for easier access
from ultratrack.compliance.privacy_manager import (
    PrivacyManager, PrivacyPolicy, DataCategory, ProcessingPurpose,
    PrivacyControl, DataProtectionImpactAssessment
)
from ultratrack.compliance.audit_logger import (
    AuditLogger, AuditRecord, AuditEventType, EventSeverity,
    AuditQuery, AuditExport
)
from ultratrack.compliance.consent_manager import (
    ConsentManager, ConsentRecord, ConsentType, ConsentVerification,
    ConsentScope, ConsentRevocation
)
from ultratrack.compliance.data_lifecycle import (
    DataLifecycleManager, RetentionPolicy, DeletionVerification,
    LifecycleStage, RetentionSchedule
)
from ultratrack.compliance.legal_compliance import (
    LegalComplianceManager, LegalFramework, ComplianceRequirement,
    JurisdictionRule, RegulationUpdate
)
from ultratrack.compliance.authorized_purpose import (
    AuthorizedPurposeManager, Purpose, PurposeVerification,
    AuthorizationScope, PurposeRegistry
)
from ultratrack.compliance.ethical_oversight import (
    EthicalOversightManager, EthicalReview, OversightCommittee,
    EthicalImpactAssessment, UserAppeal
)
from ultratrack.compliance.jurisdiction_manager import (
    JurisdictionManager, Jurisdiction, JurisdictionBoundary,
    LegalAuthority, CrossJurisdictionTransfer
)
from ultratrack.compliance.access_justification import (
    AccessJustificationManager, AccessJustification, JustificationType,
    JustificationEvidence, JustificationReview
)
from ultratrack.compliance.warrant_validation import (
    WarrantValidator, WarrantRecord, WarrantType, WarrantScope,
    LegalAuthorization, WarrantExpiration
)
from ultratrack.compliance.compliance_reporting import (
    ComplianceReporter, ComplianceReport, ReportingRequirement,
    ComplianceMetric, RegulatorReport
)

# Export public API
__all__ = [
    # Privacy management interfaces
    'PrivacyManager', 'PrivacyPolicy', 'DataCategory', 'ProcessingPurpose',
    'PrivacyControl', 'DataProtectionImpactAssessment',
    
    # Audit logging interfaces
    'AuditLogger', 'AuditRecord', 'AuditEventType', 'EventSeverity',
    'AuditQuery', 'AuditExport',
    
    # Consent management interfaces
    'ConsentManager', 'ConsentRecord', 'ConsentType', 'ConsentVerification',
    'ConsentScope', 'ConsentRevocation',
    
    # Data lifecycle interfaces
    'DataLifecycleManager', 'RetentionPolicy', 'DeletionVerification',
    'LifecycleStage', 'RetentionSchedule',
    
    # Legal compliance interfaces
    'LegalComplianceManager', 'LegalFramework', 'ComplianceRequirement',
    'JurisdictionRule', 'RegulationUpdate',
    
    # Authorized purpose interfaces
    'AuthorizedPurposeManager', 'Purpose', 'PurposeVerification',
    'AuthorizationScope', 'PurposeRegistry',
    
    # Ethical oversight interfaces
    'EthicalOversightManager', 'EthicalReview', 'OversightCommittee',
    'EthicalImpactAssessment', 'UserAppeal',
    
    # Jurisdiction management interfaces
    'JurisdictionManager', 'Jurisdiction', 'JurisdictionBoundary',
    'LegalAuthority', 'CrossJurisdictionTransfer',
    
    # Access justification interfaces
    'AccessJustificationManager', 'AccessJustification', 'JustificationType',
    'JustificationEvidence', 'JustificationReview',
    
    # Warrant validation interfaces
    'WarrantValidator', 'WarrantRecord', 'WarrantType', 'WarrantScope',
    'LegalAuthorization', 'WarrantExpiration',
    
    # Compliance reporting interfaces
    'ComplianceReporter', 'ComplianceReport', 'ReportingRequirement',
    'ComplianceMetric', 'RegulatorReport',
]

logger.debug("UltraTrack compliance module initialized")
