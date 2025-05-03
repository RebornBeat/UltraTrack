"""
UltraTrack Infrastructure Module

This module provides system infrastructure components:
- Distributed processing framework
- Automatic scaling
- GPU resource scheduling
- Service discovery
- Node health monitoring
- Load balancing

Copyright (c) 2025 Your Organization
"""

import logging

# Setup module logger
logger = logging.getLogger(__name__)

# Import key components for easier access
from ultratrack.infrastructure.distributed_processing import (
    DistributedProcessingManager, WorkerNode, TaskDistributor,
    ProcessingTask, TaskResult
)
from ultratrack.infrastructure.scaling_manager import (
    ScalingManager, ScalingPolicy, ResourceMetrics, ScalingAction,
    CapacityPlan, ScalingThreshold
)
from ultratrack.infrastructure.gpu_scheduler import (
    GPUScheduler, GPUResource, ComputeTask, ExecutionPriority,
    BatchExecution, MemoryOptimization
)
from ultratrack.infrastructure.service_discovery import (
    ServiceDiscovery, ServiceRegistry, ServiceEndpoint, 
    HealthCheck, ServiceMetadata
)
from ultratrack.infrastructure.node_health_monitor import (
    NodeHealthMonitor, HealthStatus, PerformanceMetrics,
    FailureDetector, RecoveryAction
)
from ultratrack.infrastructure.load_balancer import (
    LoadBalancer, BalancingStrategy, RoutingTable,
    RequestDistribution, LoadMetric
)
from ultratrack.infrastructure.global_synchronization import (
    GlobalSynchronizer, SyncProtocol, TimeServer,
    SynchronizationEvent, ClockSkew
)
from ultratrack.infrastructure.edge_processing import (
    EdgeProcessingManager, EdgeNode, LocalProcessing,
    DataFilteringPolicy, EdgeToCore
)
from ultratrack.infrastructure.network_optimization import (
    NetworkOptimizer, BandwidthAllocation, QualityOfService,
    CongestionControl, RouteOptimization
)
from ultratrack.infrastructure.cloud_hybrid_manager import (
    CloudHybridManager, CloudResource, OnPremisesResource,
    ResourceAllocation, DeploymentStrategy
)
from ultratrack.infrastructure.resource_forecaster import (
    ResourceForecaster, ResourceForecast, UsagePattern,
    CapacityPrediction, GrowthTrend
)
from ultratrack.infrastructure.data_pipeline_optimizer import (
    DataPipelineOptimizer, PipelineStage, Bottleneck,
    ThroughputOptimization, LatencyReduction
)

# Export public API
__all__ = [
    # Distributed processing interfaces
    'DistributedProcessingManager', 'WorkerNode', 'TaskDistributor',
    'ProcessingTask', 'TaskResult',
    
    # Scaling interfaces
    'ScalingManager', 'ScalingPolicy', 'ResourceMetrics', 'ScalingAction',
    'CapacityPlan', 'ScalingThreshold',
    
    # GPU scheduling interfaces
    'GPUScheduler', 'GPUResource', 'ComputeTask', 'ExecutionPriority',
    'BatchExecution', 'MemoryOptimization',
    
    # Service discovery interfaces
    'ServiceDiscovery', 'ServiceRegistry', 'ServiceEndpoint', 
    'HealthCheck', 'ServiceMetadata',
    
    # Node health interfaces
    'NodeHealthMonitor', 'HealthStatus', 'PerformanceMetrics',
    'FailureDetector', 'RecoveryAction',
    
    # Load balancing interfaces
    'LoadBalancer', 'BalancingStrategy', 'RoutingTable',
    'RequestDistribution', 'LoadMetric',
    
    # Global synchronization interfaces
    'GlobalSynchronizer', 'SyncProtocol', 'TimeServer',
    'SynchronizationEvent', 'ClockSkew',
    
    # Edge processing interfaces
    'EdgeProcessingManager', 'EdgeNode', 'LocalProcessing',
    'DataFilteringPolicy', 'EdgeToCore',
    
    # Network optimization interfaces
    'NetworkOptimizer', 'BandwidthAllocation', 'QualityOfService',
    'CongestionControl', 'RouteOptimization',
    
    # Cloud hybrid interfaces
    'CloudHybridManager', 'CloudResource', 'OnPremisesResource',
    'ResourceAllocation', 'DeploymentStrategy',
    
    # Resource forecasting interfaces
    'ResourceForecaster', 'ResourceForecast', 'UsagePattern',
    'CapacityPrediction', 'GrowthTrend',
    
    # Data pipeline interfaces
    'DataPipelineOptimizer', 'PipelineStage', 'Bottleneck',
    'ThroughputOptimization', 'LatencyReduction',
]

logger.debug("UltraTrack infrastructure module initialized")
