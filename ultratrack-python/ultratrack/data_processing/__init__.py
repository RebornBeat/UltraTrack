"""
UltraTrack Data Processing Module

This module provides components for processing collected data:
- Preprocessing and enhancing raw data
- Managing storage and retrieval
- Fusing data from multiple sources
- Anonymizing sensitive information
- Batch and stream processing

Copyright (c) 2025 Your Organization
"""

import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Import key components for easier access
from ultratrack.data_processing.preprocessing import (
    Preprocessor, VideoPreprocessor, AudioPreprocessor,
    ThermalPreprocessor, RFPreprocessor, EnhancementPipeline
)
from ultratrack.data_processing.storage_manager import (
    StorageManager, StorageTier, DataSegment, SegmentMetadata,
    QueryOptions, StorageTransaction
)
from ultratrack.data_processing.data_fusion import (
    DataFusion, FusionStrategy, SourceReliability, DataAlignment,
    TemporalAlignment, SpatialAlignment, FusionResult
)
from ultratrack.data_processing.anonymization import (
    Anonymizer, AnonymizationLevel, PrivacyPreservingTransform,
    DifferentialPrivacy, PseudonymizationStrategy
)
from ultratrack.data_processing.batch_processor import (
    BatchProcessor, BatchJob, ProcessingPriority, JobScheduler,
    JobStatus, ProcessingMetrics
)
from ultratrack.data_processing.streaming_processor import (
    StreamProcessor, StreamPipeline, ProcessingNode, DataStream,
    StreamWindow, WindowFunction
)
from ultratrack.data_processing.metadata_extractor import (
    MetadataExtractor, MetadataSchema, ExtractionRule, MetadataField
)
from ultratrack.data_processing.historical_integrator import (
    HistoricalIntegrator, HistoricalDataSource, ArchiveFormat, 
    MigrationStrategy
)
from ultratrack.data_processing.temporal_reconciliation import (
    TemporalReconciliation, TimeseriesAlignment, SequenceMatching,
    EventOrdering, TimeSynchronization
)

# Export public API
__all__ = [
    # Preprocessing interfaces
    'Preprocessor', 'VideoPreprocessor', 'AudioPreprocessor',
    'ThermalPreprocessor', 'RFPreprocessor', 'EnhancementPipeline',
    
    # Storage interfaces
    'StorageManager', 'StorageTier', 'DataSegment', 'SegmentMetadata',
    'QueryOptions', 'StorageTransaction',
    
    # Data fusion interfaces
    'DataFusion', 'FusionStrategy', 'SourceReliability', 'DataAlignment',
    'TemporalAlignment', 'SpatialAlignment', 'FusionResult',
    
    # Anonymization interfaces
    'Anonymizer', 'AnonymizationLevel', 'PrivacyPreservingTransform',
    'DifferentialPrivacy', 'PseudonymizationStrategy',
    
    # Batch processing interfaces
    'BatchProcessor', 'BatchJob', 'ProcessingPriority', 'JobScheduler',
    'JobStatus', 'ProcessingMetrics',
    
    # Stream processing interfaces
    'StreamProcessor', 'StreamPipeline', 'ProcessingNode', 'DataStream',
    'StreamWindow', 'WindowFunction',
    
    # Metadata interfaces
    'MetadataExtractor', 'MetadataSchema', 'ExtractionRule', 'MetadataField',
    
    # Historical data interfaces
    'HistoricalIntegrator', 'HistoricalDataSource', 'ArchiveFormat',
    'MigrationStrategy',
    
    # Temporal reconciliation interfaces
    'TemporalReconciliation', 'TimeseriesAlignment', 'SequenceMatching',
    'EventOrdering', 'TimeSynchronization',
]

logger.debug("UltraTrack data processing module initialized")
