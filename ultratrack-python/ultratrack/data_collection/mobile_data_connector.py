"""
UltraTrack Mobile Data Connector

This module provides interfaces for collecting anonymized mobile device data from authorized
sources. It includes capabilities for:
- Connecting to mobile network operator data
- Retrieving anonymized location data
- Processing cellular tower triangulation data
- Integrating with mobile device tracking APIs
- Managing privacy controls and data anonymization
- Correlating mobile signals with other tracking modalities

The module enforces strict privacy controls and legal authorization checks for all data access.

Copyright (c) 2025 Your Organization
"""

import logging
import time
import uuid
import json
import hashlib
import ipaddress
import socket
import threading
import queue
import asyncio
import aiohttp
import backoff
import enum
from typing import Dict, List, Optional, Set, Tuple, Union, Any, AsyncIterator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import numpy as np
from shapely.geometry import Point, Polygon

from ultratrack.config import ConfigManager
from ultratrack.compliance.audit_logger import AuditLogger
from ultratrack.compliance.privacy_manager import PrivacyManager, DataCategory
from ultratrack.compliance.authorized_purpose import AuthorizedPurposeManager, Purpose
from ultratrack.compliance.warrant_validation import WarrantValidator
from ultratrack.security.encryption import EncryptionManager
from ultratrack.data_processing.anonymization import Anonymizer, AnonymizationLevel
from ultratrack.data_collection.data_validator import DataValidator, ValidationRule


# Module logger
logger = logging.getLogger(__name__)


class MobileDataType(enum.Enum):
    """Types of mobile data that can be collected."""
    
    CELL_TOWER_TRIANGULATION = "cell_tower_triangulation"
    WIFI_POSITIONING = "wifi_positioning"
    GPS_LOCATION = "gps_location"
    BLUETOOTH_PROXIMITY = "bluetooth_proximity"
    APP_LOCATION_SERVICES = "app_location_services"
    NETWORK_CONNECTIONS = "network_connections"
    DEVICE_ACTIVITY = "device_activity"
    SIGNAL_STRENGTH = "signal_strength"


class LocationAccuracy(enum.Enum):
    """Accuracy levels for location data."""
    
    HIGH = "high"       # 0-10 meters
    MEDIUM = "medium"   # 10-50 meters
    LOW = "low"         # 50-500 meters
    VERY_LOW = "very_low"  # 500+ meters


class PrivacyTier(enum.Enum):
    """Privacy tiers for different levels of data anonymization."""
    
    TIER_1 = 1  # Highly anonymized, coarse location (neighborhood/district level)
    TIER_2 = 2  # Moderately anonymized, approximate location (block level)
    TIER_3 = 3  # Minimally anonymized, precise location (with legal authorization)
    TIER_4 = 4  # Raw data (requires special authorization and auditing)


@dataclass
class MobileDataProvider:
    """Information about a mobile data provider."""
    
    id: str
    name: str
    api_endpoint: str
    api_key: str = field(repr=False)
    api_version: str = "v1"
    rate_limit: int = 100  # Requests per minute
    coverage_countries: List[str] = field(default_factory=list)
    supports_real_time: bool = False
    supports_historical: bool = True
    historical_data_days: int = 30
    privacy_tier: PrivacyTier = PrivacyTier.TIER_2
    anonymization_method: str = "k-anonymity"
    enabled: bool = True
    

@dataclass
class AnonymizedLocation:
    """Anonymized location data from a mobile device."""
    
    # Unique identifier for this location record
    record_id: str
    
    # Anonymized device identifier (consistent for same device)
    device_id: str
    
    # Approximate coordinates (anonymized)
    latitude: float
    longitude: float
    
    # Accuracy radius in meters
    accuracy_radius: float
    
    # Timestamp (UTC)
    timestamp: datetime
    
    # Source of the location data
    data_type: MobileDataType
    
    # Confidence score (0.0-1.0)
    confidence: float
    
    # Privacy tier applied to this data
    privacy_tier: PrivacyTier
    
    # Optional altitude information (meters above sea level)
    altitude: Optional[float] = None
    
    # Optional heading information (degrees from north)
    heading: Optional[float] = None
    
    # Optional speed information (meters per second)
    speed: Optional[float] = None
    
    # Optional provider information
    provider_id: Optional[str] = None
    
    # Optional area name (neighborhood, district, etc.)
    area_name: Optional[str] = None
    
    # Optional metadata dictionary
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "record_id": self.record_id,
            "device_id": self.device_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "accuracy_radius": self.accuracy_radius,
            "timestamp": self.timestamp.isoformat(),
            "data_type": self.data_type.value,
            "confidence": self.confidence,
            "privacy_tier": self.privacy_tier.value,
        }
        
        # Add optional fields if they exist
        if self.altitude is not None:
            result["altitude"] = self.altitude
        
        if self.heading is not None:
            result["heading"] = self.heading
        
        if self.speed is not None:
            result["speed"] = self.speed
        
        if self.provider_id:
            result["provider_id"] = self.provider_id
        
        if self.area_name:
            result["area_name"] = self.area_name
        
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnonymizedLocation":
        """Create an AnonymizedLocation from a dictionary."""
        # Parse required fields
        record_id = data["record_id"]
        device_id = data["device_id"]
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        accuracy_radius = float(data["accuracy_radius"])
        timestamp = datetime.fromisoformat(data["timestamp"])
        data_type = MobileDataType(data["data_type"])
        confidence = float(data["confidence"])
        privacy_tier = PrivacyTier(int(data["privacy_tier"]))
        
        # Parse optional fields
        altitude = data.get("altitude")
        heading = data.get("heading")
        speed = data.get("speed")
        provider_id = data.get("provider_id")
        area_name = data.get("area_name")
        metadata = data.get("metadata", {})
        
        return cls(
            record_id=record_id,
            device_id=device_id,
            latitude=latitude,
            longitude=longitude,
            accuracy_radius=accuracy_radius,
            timestamp=timestamp,
            data_type=data_type,
            confidence=confidence,
            privacy_tier=privacy_tier,
            altitude=altitude,
            heading=heading,
            speed=speed,
            provider_id=provider_id,
            area_name=area_name,
            metadata=metadata
        )


@dataclass
class QueryParameters:
    """Parameters for querying mobile data."""
    
    # Geographic area to query (can be a polygon, circle, or bounding box)
    area: Union[Polygon, Tuple[float, float, float], Tuple[float, float, float, float]]
    
    # Time range to query
    start_time: datetime
    end_time: datetime
    
    # Minimum data quality (0.0-1.0)
    min_confidence: float = 0.6
    
    # Maximum number of results to return
    max_results: int = 1000
    
    # Types of data to include
    data_types: Set[MobileDataType] = field(default_factory=lambda: {
        MobileDataType.CELL_TOWER_TRIANGULATION,
        MobileDataType.WIFI_POSITIONING,
        MobileDataType.GPS_LOCATION
    })
    
    # Provider IDs to query
    provider_ids: Set[str] = field(default_factory=set)
    
    # Device IDs to filter by (if empty, all devices are included)
    device_ids: Set[str] = field(default_factory=set)
    
    # Privacy tier for results
    privacy_tier: PrivacyTier = PrivacyTier.TIER_2
    
    # Include additional metadata
    include_metadata: bool = False
    
    # Sort order for results
    sort_by: str = "timestamp"
    sort_direction: str = "asc"
    
    def validate(self) -> bool:
        """Validate query parameters."""
        # Check time range
        if self.start_time >= self.end_time:
            logger.error("Start time must be before end time")
            return False
        
        # Check confidence
        if not 0.0 <= self.min_confidence <= 1.0:
            logger.error("Confidence must be between 0.0 and 1.0")
            return False
        
        # Check max results
        if self.max_results <= 0:
            logger.error("Max results must be positive")
            return False
        
        # Check data types
        if not self.data_types:
            logger.error("At least one data type must be specified")
            return False
        
        return True


class AuthorizationError(Exception):
    """Exception raised when unauthorized access is attempted."""
    pass


class ConnectionError(Exception):
    """Exception raised when connection to a provider fails."""
    pass


class QueryError(Exception):
    """Exception raised when a query fails."""
    pass


class MobileDataConnector:
    """
    Connector for mobile device location data from authorized sources.
    
    This class provides methods for retrieving anonymized mobile location data
    from mobile network operators, app providers, and other authorized sources.
    It enforces strict privacy controls and legal authorization requirements.
    """
    
    def __init__(self, config=None):
        """
        Initialize the mobile data connector.
        
        Args:
            config: Configuration for the connector
        """
        self.config = config or ConfigManager.get_config().data_collection.mobile_data_connector
        self.providers: Dict[str, MobileDataProvider] = {}
        self.anonymizer = Anonymizer(self.config.anonymization)
        self.audit_logger = AuditLogger.get_instance()
        self.privacy_manager = PrivacyManager.get_instance()
        self.purpose_manager = AuthorizedPurposeManager.get_instance()
        self.warrant_validator = WarrantValidator.get_instance()
        self.encryption_manager = EncryptionManager.get_instance()
        self.data_validator = DataValidator(self.config.validation)
        
        self._request_queue = queue.Queue()
        self._response_cache = {}
        self._cache_lock = threading.RLock()
        self._rate_limiters = {}
        self._last_stats_time = time.time()
        self._request_count = 0
        self._error_count = 0
        self._worker_thread = None
        self._running = False
        
        # Thread-local HTTP session for async operations
        self._session = None
        self._loop = None
        
        # Initialize providers
        self._init_providers()
        
        logger.info(f"Mobile data connector initialized with {len(self.providers)} providers")
    
    def _init_providers(self):
        """Initialize configured mobile data providers."""
        for provider_config in self.config.providers:
            try:
                provider = MobileDataProvider(
                    id=provider_config["id"],
                    name=provider_config["name"],
                    api_endpoint=provider_config["api_endpoint"],
                    api_key=self.encryption_manager.decrypt(provider_config["encrypted_api_key"]),
                    api_version=provider_config.get("api_version", "v1"),
                    rate_limit=provider_config.get("rate_limit", 100),
                    coverage_countries=provider_config.get("coverage_countries", []),
                    supports_real_time=provider_config.get("supports_real_time", False),
                    supports_historical=provider_config.get("supports_historical", True),
                    historical_data_days=provider_config.get("historical_data_days", 30),
                    privacy_tier=PrivacyTier(provider_config.get("privacy_tier", 2)),
                    anonymization_method=provider_config.get("anonymization_method", "k-anonymity"),
                    enabled=provider_config.get("enabled", True)
                )
                
                if provider.enabled:
                    self.providers[provider.id] = provider
                    self._rate_limiters[provider.id] = {
                        "limit": provider.rate_limit,
                        "remaining": provider.rate_limit,
                        "reset_time": time.time() + 60
                    }
                    
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_config.get('id', 'unknown')}: {str(e)}")
    
    def start(self):
        """Start the mobile data connector service."""
        if self._running:
            logger.warning("Mobile data connector is already running")
            return
        
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="MobileDataConnector-Worker",
            daemon=True
        )
        self._worker_thread.start()
        
        logger.info("Mobile data connector service started")
    
    def stop(self):
        """Stop the mobile data connector service."""
        if not self._running:
            logger.warning("Mobile data connector is not running")
            return
        
        self._running = False
        
        # Wait for worker thread to complete
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        
        # Close any open sessions
        if self._session and not self._session.closed:
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(self._session.close(), self._loop)
        
        logger.info("Mobile data connector service stopped")
    
    def _worker_loop(self):
        """Background worker thread for processing requests."""
        # Create event loop for this thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        # Create HTTP session
        self._session = aiohttp.ClientSession(loop=self._loop)
        
        try:
            while self._running:
                # Process rate limiter resets
                current_time = time.time()
                for provider_id, rate_limiter in self._rate_limiters.items():
                    if current_time >= rate_limiter["reset_time"]:
                        rate_limiter["remaining"] = rate_limiter["limit"]
                        rate_limiter["reset_time"] = current_time + 60
                
                # Log statistics periodically
                if current_time - self._last_stats_time >= 300:  # Every 5 minutes
                    self._log_stats()
                    self._last_stats_time = current_time
                
                # Process request queue
                try:
                    request = self._request_queue.get(timeout=1.0)
                    self._process_request(request)
                    self._request_queue.task_done()
                except queue.Empty:
                    pass
                
                # Clear expired cache entries
                self._cleanup_cache()
                
                # Small sleep to prevent tight loop
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in mobile data connector worker thread: {str(e)}")
        finally:
            # Close session
            if self._session and not self._session.closed:
                self._loop.run_until_complete(self._session.close())
    
    def _process_request(self, request):
        """Process a queued request."""
        query_id = request["query_id"]
        params = request["params"]
        callback = request["callback"]
        
        try:
            # Check rate limits
            provider_ids = params.provider_ids or set(self.providers.keys())
            for provider_id in provider_ids:
                if provider_id in self._rate_limiters:
                    rate_limiter = self._rate_limiters[provider_id]
                    if rate_limiter["remaining"] <= 0:
                        raise QueryError(f"Rate limit exceeded for provider {provider_id}")
                    rate_limiter["remaining"] -= 1
            
            # Execute query
            results = self._loop.run_until_complete(
                self._execute_query(params)
            )
            
            # Cache results
            cache_key = self._get_cache_key(params)
            with self._cache_lock:
                self._response_cache[cache_key] = {
                    "data": results,
                    "timestamp": time.time(),
                    "expiry": time.time() + self.config.cache_ttl_seconds
                }
            
            # Call callback with results
            if callback:
                callback(query_id, results, None)
                
            self._request_count += 1
                
        except Exception as e:
            logger.error(f"Error processing query {query_id}: {str(e)}")
            self._error_count += 1
            
            # Call callback with error
            if callback:
                callback(query_id, None, str(e))
    
    async def _execute_query(self, params: QueryParameters) -> List[AnonymizedLocation]:
        """
        Execute a query against mobile data providers.
        
        Args:
            params: Query parameters
            
        Returns:
            List of anonymized location records
        """
        all_results = []
        
        # Determine which providers to query
        provider_ids = params.provider_ids or set(self.providers.keys())
        
        # Create tasks for each provider
        tasks = []
        for provider_id in provider_ids:
            if provider_id in self.providers:
                provider = self.providers[provider_id]
                
                # Skip providers that don't support the required time range
                if not provider.supports_historical and params.start_time < datetime.now() - timedelta(hours=1):
                    logger.debug(f"Skipping provider {provider_id} that doesn't support historical data")
                    continue
                
                # Skip providers with insufficient privacy tier
                if provider.privacy_tier.value > params.privacy_tier.value:
                    logger.debug(f"Skipping provider {provider_id} with insufficient privacy tier")
                    continue
                
                # Create task for this provider
                task = self._query_provider(provider, params)
                tasks.append(task)
        
        # Execute queries in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Provider query failed: {str(result)}")
                    continue
                
                all_results.extend(result)
        
        # Apply sorting
        if params.sort_by:
            reverse = params.sort_direction.lower() == "desc"
            all_results.sort(
                key=lambda x: getattr(x, params.sort_by), 
                reverse=reverse
            )
        
        # Apply limit
        if params.max_results > 0 and len(all_results) > params.max_results:
            all_results = all_results[:params.max_results]
        
        return all_results
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=30
    )
    async def _query_provider(
        self, 
        provider: MobileDataProvider, 
        params: QueryParameters
    ) -> List[AnonymizedLocation]:
        """
        Query a specific mobile data provider.
        
        Args:
            provider: Provider to query
            params: Query parameters
            
        Returns:
            List of anonymized location records
        """
        # Prepare request parameters
        request_params = {
            "start_time": params.start_time.isoformat(),
            "end_time": params.end_time.isoformat(),
            "min_confidence": params.min_confidence,
            "data_types": [dt.value for dt in params.data_types],
            "privacy_tier": params.privacy_tier.value,
            "include_metadata": params.include_metadata,
        }
        
        # Add area parameter based on type
        if isinstance(params.area, Polygon):
            # Convert shapely polygon to coordinates
            coords = list(params.area.exterior.coords)
            request_params["area_type"] = "polygon"
            request_params["area_coordinates"] = coords
        elif len(params.area) == 3:
            # Circle: (lat, lon, radius_meters)
            request_params["area_type"] = "circle"
            request_params["latitude"] = params.area[0]
            request_params["longitude"] = params.area[1]
            request_params["radius"] = params.area[2]
        elif len(params.area) == 4:
            # Bounding box: (min_lat, min_lon, max_lat, max_lon)
            request_params["area_type"] = "bbox"
            request_params["min_latitude"] = params.area[0]
            request_params["min_longitude"] = params.area[1]
            request_params["max_latitude"] = params.area[2]
            request_params["max_longitude"] = params.area[3]
        
        # Add device IDs if specified
        if params.device_ids:
            request_params["device_ids"] = list(params.device_ids)
        
        # Construct URL
        url = f"{provider.api_endpoint}/api/{provider.api_version}/query"
        
        # Set headers with authentication
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"UltraTrack/1.0 MobileDataConnector/{self.config.version}"
        }
        
        # Make request
        async with self._session.post(
            url,
            json=request_params,
            headers=headers,
            timeout=self.config.request_timeout_seconds
        ) as response:
            # Check status
            if response.status != 200:
                error_text = await response.text()
                raise QueryError(f"Provider {provider.id} returned error {response.status}: {error_text}")
            
            # Parse response
            response_data = await response.json()
            
            # Validate response
            if "locations" not in response_data:
                raise QueryError(f"Invalid response from provider {provider.id}: missing 'locations' field")
            
            # Parse location records
            results = []
            for location_data in response_data["locations"]:
                try:
                    # Add provider ID to the record
                    location_data["provider_id"] = provider.id
                    
                    # Convert to AnonymizedLocation
                    location = AnonymizedLocation.from_dict(location_data)
                    
                    # Validate data
                    if self.data_validator.validate_location(location):
                        results.append(location)
                    else:
                        logger.warning(f"Invalid location data from provider {provider.id}")
                        
                except Exception as e:
                    logger.error(f"Error parsing location data: {str(e)}")
            
            return results
    
    def _get_cache_key(self, params: QueryParameters) -> str:
        """Generate a cache key for the given query parameters."""
        # Create a hashable representation of the query
        query_dict = {
            "start_time": params.start_time.isoformat(),
            "end_time": params.end_time.isoformat(),
            "min_confidence": params.min_confidence,
            "data_types": sorted([dt.value for dt in params.data_types]),
            "provider_ids": sorted(list(params.provider_ids)) if params.provider_ids else [],
            "device_ids": sorted(list(params.device_ids)) if params.device_ids else [],
            "privacy_tier": params.privacy_tier.value,
        }
        
        # Add area information
        if isinstance(params.area, Polygon):
            # Use the WKT representation of the polygon
            query_dict["area"] = params.area.wkt
        else:
            # Use the tuple representation
            query_dict["area"] = str(params.area)
        
        # Convert to JSON and hash
        query_json = json.dumps(query_dict, sort_keys=True)
        return hashlib.sha256(query_json.encode()).hexdigest()
    
    def _cleanup_cache(self):
        """Remove expired entries from the response cache."""
        current_time = time.time()
        with self._cache_lock:
            expired_keys = [
                key for key, entry in self._response_cache.items()
                if current_time > entry["expiry"]
            ]
            
            for key in expired_keys:
                del self._response_cache[key]
    
    def _log_stats(self):
        """Log statistics about the connector's activity."""
        logger.info(
            f"Mobile data connector stats: "
            f"requests={self._request_count}, "
            f"errors={self._error_count}, "
            f"cache_size={len(self._response_cache)}, "
            f"queue_size={self._request_queue.qsize()}"
        )
    
    def _check_authorization(self, purpose_id: str, area: Any) -> bool:
        """
        Check if the current context is authorized to access mobile data.
        
        Args:
            purpose_id: Purpose identifier for the query
            area: Geographic area for the query
            
        Returns:
            True if authorized, False otherwise
            
        Raises:
            AuthorizationError: If authorization fails
        """
        # Check if a valid purpose exists
        if not self.purpose_manager.verify_purpose(purpose_id, DataCategory.LOCATION_DATA):
            raise AuthorizationError(f"Invalid purpose: {purpose_id}")
        
        # For higher privacy tiers, check if there's a valid warrant
        purpose = self.purpose_manager.get_purpose(purpose_id)
        if purpose.privacy_tier.value > PrivacyTier.TIER_2.value:
            # Need to validate warrant for higher privacy tiers
            if not self.warrant_validator.validate_for_location(purpose_id, area):
                raise AuthorizationError(
                    f"Valid warrant required for privacy tier {purpose.privacy_tier.name}"
                )
        
        # Record authorization check in audit log
        self.audit_logger.log_authorization_check(
            data_category=DataCategory.LOCATION_DATA,
            purpose_id=purpose_id,
            authorized=True
        )
        
        return True
    
    def query_locations(
        self, 
        params: QueryParameters,
        purpose_id: str,
        callback=None
    ) -> str:
        """
        Query mobile location data with the given parameters.
        
        This method is asynchronous and returns a query ID. Results will be
        delivered via the provided callback function when available.
        
        Args:
            params: Query parameters
            purpose_id: Authorized purpose identifier
            callback: Optional callback function to receive results
            
        Returns:
            Query ID
            
        Raises:
            AuthorizationError: If access is not authorized
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if not params.validate():
            raise ValueError("Invalid query parameters")
        
        # Check authorization
        if not self._check_authorization(purpose_id, params.area):
            raise AuthorizationError(f"Unauthorized access: {purpose_id}")
        
        # Check cache first
        cache_key = self._get_cache_key(params)
        with self._cache_lock:
            if cache_key in self._response_cache:
                cache_entry = self._response_cache[cache_key]
                if time.time() <= cache_entry["expiry"]:
                    logger.debug(f"Cache hit for query {cache_key}")
                    
                    # Call callback with cached results
                    if callback:
                        callback(cache_key, cache_entry["data"], None)
                    
                    return cache_key
        
        # Generate query ID
        query_id = str(uuid.uuid4())
        
        # Queue request for processing
        self._request_queue.put({
            "query_id": query_id,
            "params": params,
            "callback": callback,
            "purpose_id": purpose_id,
            "timestamp": time.time()
        })
        
        logger.debug(f"Queued query {query_id} with {len(params.provider_ids or set(self.providers.keys()))} providers")
        
        return query_id
    
    def query_locations_sync(
        self, 
        params: QueryParameters,
        purpose_id: str,
        timeout: float = 30.0
    ) -> List[AnonymizedLocation]:
        """
        Query mobile location data synchronously with the given parameters.
        
        This method blocks until results are available or the timeout expires.
        
        Args:
            params: Query parameters
            purpose_id: Authorized purpose identifier
            timeout: Maximum time to wait for results in seconds
            
        Returns:
            List of anonymized location records
            
        Raises:
            AuthorizationError: If access is not authorized
            ValueError: If parameters are invalid
            TimeoutError: If the query times out
        """
        # Create synchronization primitive
        result_event = threading.Event()
        result_data = {"results": None, "error": None}
        
        # Define callback
        def query_callback(query_id, results, error):
            result_data["results"] = results
            result_data["error"] = error
            result_event.set()
        
        # Submit query
        query_id = self.query_locations(params, purpose_id, query_callback)
        
        # Wait for results
        if not result_event.wait(timeout):
            raise TimeoutError(f"Query {query_id} timed out after {timeout} seconds")
        
        # Check for errors
        if result_data["error"]:
            raise QueryError(f"Query {query_id} failed: {result_data['error']}")
        
        return result_data["results"]
    
    def get_device_history(
        self,
        device_id: str,
        start_time: datetime,
        end_time: datetime,
        purpose_id: str,
        privacy_tier: PrivacyTier = PrivacyTier.TIER_2
    ) -> List[AnonymizedLocation]:
        """
        Get historical location data for a specific device.
        
        Args:
            device_id: Anonymized device identifier
            start_time: Start time for the query
            end_time: End time for the query
            purpose_id: Authorized purpose identifier
            privacy_tier: Privacy tier for the results
            
        Returns:
            List of anonymized location records
            
        Raises:
            AuthorizationError: If access is not authorized
        """
        # Create query parameters
        params = QueryParameters(
            # Use world bounding box as area
            area=(-90, -180, 90, 180),
            start_time=start_time,
            end_time=end_time,
            device_ids={device_id},
            privacy_tier=privacy_tier,
            sort_by="timestamp",
            sort_direction="asc"
        )
        
        # Execute query synchronously
        return self.query_locations_sync(params, purpose_id)

    def get_area_activity(
        self,
        area: Union[Polygon, Tuple[float, float, float], Tuple[float, float, float, float]],
        start_time: datetime,
        end_time: datetime,
        purpose_id: str,
        min_confidence: float = 0.7,
        privacy_tier: PrivacyTier = PrivacyTier.TIER_2
    ) -> List[AnonymizedLocation]:
        """
        Get activity in a specific geographic area.
        
        Args:
            area: Geographic area (polygon, circle, or bounding box)
            start_time: Start time for the query
            end_time: End time for the query
            purpose_id: Authorized purpose identifier
            min_confidence: Minimum confidence for results
            privacy_tier: Privacy tier for the results
            
        Returns:
            List of anonymized location records
            
        Raises:
            AuthorizationError: If access is not authorized
        """
        # Create query parameters
        params = QueryParameters(
            area=area,
            start_time=start_time,
            end_time=end_time,
            min_confidence=min_confidence,
            privacy_tier=privacy_tier,
            sort_by="timestamp",
            sort_direction="asc"
        )
        
        # Execute query synchronously
        return self.query_locations_sync(params, purpose_id)

    def get_providers(self) -> List[Dict[str, Any]]:
        """
        Get information about available data providers.
        
        Returns:
            List of provider information dictionaries
        """
        return [
            {
                "id": p.id,
                "name": p.name,
                "supports_real_time": p.supports_real_time,
                "supports_historical": p.supports_historical,
                "historical_data_days": p.historical_data_days,
                "privacy_tier": p.privacy_tier.value,
                "anonymization_method": p.anonymization_method,
                "coverage_countries": p.coverage_countries,
            }
            for p in self.providers.values()
        ]

    def anonymize_device_id(self, raw_device_id: str, privacy_tier: PrivacyTier) -> str:
        """
        Anonymize a raw device identifier.
        
        Args:
            raw_device_id: Raw device identifier
            privacy_tier: Privacy tier to apply
            
        Returns:
            Anonymized device identifier
        """
        if privacy_tier == PrivacyTier.TIER_1:
            # Highly anonymized - group devices by region
            # Use first 3 characters of hash
            hash_obj = hashlib.sha256(raw_device_id.encode())
            return f"anon_t1_{hash_obj.hexdigest()[:3]}"
            
        elif privacy_tier == PrivacyTier.TIER_2:
            # Moderately anonymized - stable identifier but not reversible
            hash_obj = hashlib.sha256(raw_device_id.encode())
            return f"anon_t2_{hash_obj.hexdigest()}"
            
        elif privacy_tier == PrivacyTier.TIER_3:
            # Minimally anonymized - reversible with proper authorization
            key = self.encryption_manager.get_key("device_id_tier3")
            cipher = Fernet(key)
            return f"anon_t3_{cipher.encrypt(raw_device_id.encode()).decode()}"
            
        elif privacy_tier == PrivacyTier.TIER_4:
            # Raw data - special authorization required
            return raw_device_id
            
        else:
            # Default to TIER_2 if unrecognized privacy tier
            hash_obj = hashlib.sha256(raw_device_id.encode())
            return f"anon_t2_{hash_obj.hexdigest()}"

    def anonymize_location(
        self, 
        latitude: float, 
        longitude: float, 
        privacy_tier: PrivacyTier
    ) -> Tuple[float, float]:
        """
        Anonymize a location based on privacy tier.
        
        Args:
            latitude: Raw latitude
            longitude: Raw longitude
            privacy_tier: Privacy tier to apply
            
        Returns:
            Tuple of (anonymized latitude, anonymized longitude)
        """
        if privacy_tier == PrivacyTier.TIER_1:
            # Highly anonymized - neighborhood level
            # Round to ~1 km precision
            return (
                round(latitude * 100) / 100,
                round(longitude * 100) / 100
            )
            
        elif privacy_tier == PrivacyTier.TIER_2:
            # Moderately anonymized - block level
            # Round to ~100 m precision
            return (
                round(latitude * 1000) / 1000,
                round(longitude * 1000) / 1000
            )
            
        elif privacy_tier == PrivacyTier.TIER_3:
            # Minimally anonymized - approximate location
            # Add small random noise
            random_lat = (np.random.random() - 0.5) * 0.0002  # ~20m
            random_lon = (np.random.random() - 0.5) * 0.0002  # ~20m
            return (
                latitude + random_lat,
                longitude + random_lon
            )
            
        elif privacy_tier == PrivacyTier.TIER_4:
            # Raw data - special authorization required
            return (latitude, longitude)
            
        else:
            # Default to TIER_2 if unrecognized privacy tier
            return (
                round(latitude * 1000) / 1000,
                round(longitude * 1000) / 1000
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the mobile data connector.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "providers": len(self.providers),
            "request_count": self._request_count,
            "error_count": self._error_count,
            "cache_size": len(self._response_cache),
            "queue_size": self._request_queue.qsize(),
            "uptime_seconds": time.time() - self._last_stats_time + 300,
        }
