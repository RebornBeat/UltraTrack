"""
License Plate Recognition (LPR) Collection Module

This module provides interfaces for collecting license plate recognition data
from various sources, including dedicated LPR systems, traffic cameras,
and third-party providers. It handles connection management, data validation,
and standardization across different LPR system types.

Copyright (c) 2025 Your Organization
"""

import logging
import threading
import queue
import time
import uuid
import json
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any, Union, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
import requests
import numpy as np
import cv2

from ultratrack.config import ConfigManager
from ultratrack.data_collection.data_validator import DataValidator, ValidationRule
from ultratrack.data_processing.storage_manager import StorageManager

# Configure module logger
logger = logging.getLogger(__name__)


class LPRSourceType(Enum):
    """Types of LPR data sources."""
    DEDICATED_LPR = auto()  # Dedicated LPR system (e.g., Genetec AutoVu)
    TRAFFIC_CAMERA = auto()  # Traffic camera with integrated LPR
    MOBILE_LPR = auto()  # Mobile LPR units (e.g., police vehicles)
    TOLLBOOTH = auto()  # Toll collection systems
    PARKING = auto()  # Parking lot/garage systems
    THIRD_PARTY_API = auto()  # Third-party LPR data provider
    CUSTOM = auto()  # Custom LPR implementation


class PlateConfidenceLevel(Enum):
    """Confidence levels for license plate detections."""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5


class PlateJurisdiction(Enum):
    """Common jurisdictions for license plates."""
    UNKNOWN = 0
    # US States
    ALABAMA = 1
    ALASKA = 2
    ARIZONA = 3
    # ... (would include all US states)
    CALIFORNIA = 6
    # ... (other jurisdictions)
    INTERNATIONAL = 999
    CUSTOM = 1000


@dataclass
class LPRSourceCredentials:
    """Credentials for connecting to an LPR source."""
    username: str = ""
    password: str = ""
    api_key: str = ""
    certificate_path: str = ""
    token: str = ""
    token_expiration: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if credentials are valid."""
        # For API key authentication
        if self.api_key:
            return len(self.api_key) > 8
        
        # For username/password authentication
        if self.username and self.password:
            return len(self.username) > 0 and len(self.password) > 0
        
        # For token-based authentication
        if self.token:
            if not self.token_expiration:
                return True  # No expiration set
            return datetime.now() < self.token_expiration
        
        # For certificate authentication
        if self.certificate_path:
            # Check if certificate file exists
            import os
            return os.path.exists(self.certificate_path)
        
        return False
    
    def is_expired(self) -> bool:
        """Check if token-based credentials have expired."""
        if not self.token or not self.token_expiration:
            return False
        return datetime.now() >= self.token_expiration
    
    def __str__(self) -> str:
        """String representation with sensitive information masked."""
        if self.api_key:
            return f"API Key: {self.api_key[:3]}...{self.api_key[-3:] if len(self.api_key) > 6 else ''}"
        if self.token:
            return f"Token: {self.token[:5]}... (expires: {self.token_expiration})"
        if self.username and self.password:
            return f"Username: {self.username}, Password: ****"
        if self.certificate_path:
            return f"Certificate: {self.certificate_path}"
        return "No credentials provided"


@dataclass
class LPRSource:
    """Represents a source of LPR data."""
    id: str
    name: str
    source_type: LPRSourceType
    endpoint: str  # URL, IP address, or connection string
    credentials: LPRSourceCredentials
    location: Optional[Tuple[float, float]] = None  # lat, lon
    jurisdiction: Set[PlateJurisdiction] = field(default_factory=lambda: {PlateJurisdiction.UNKNOWN})
    active: bool = True
    rate_limit: Optional[int] = None  # Requests per minute
    connection_timeout: int = 30  # seconds
    last_successful_connection: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and initialize source after creation."""
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Validate endpoint format
        if self.source_type == LPRSourceType.THIRD_PARTY_API:
            if not self.endpoint.startswith(('http://', 'https://')):
                self.endpoint = f"https://{self.endpoint}"
        elif self.source_type in [LPRSourceType.DEDICATED_LPR, LPRSourceType.TRAFFIC_CAMERA]:
            # Try to validate IP address for camera-based systems
            try:
                # Extract IP if in format like rtsp://192.168.1.1:554/stream
                if '://' in self.endpoint:
                    parts = self.endpoint.split('://')
                    if len(parts) == 2:
                        ip_part = parts[1].split('/')[0]
                        if ':' in ip_part:
                            ip_part = ip_part.split(':')[0]
                        ipaddress.ip_address(ip_part)
                else:
                    # Try direct IP validation
                    ipaddress.ip_address(self.endpoint.split(':')[0])
            except ValueError:
                logger.warning(f"Source {self.name} has a non-IP endpoint: {self.endpoint}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "id": self.id,
            "name": self.name,
            "source_type": self.source_type.name,
            "endpoint": self.endpoint,
            "active": self.active,
            "connection_timeout": self.connection_timeout,
        }
        
        if self.location:
            result["location"] = {"latitude": self.location[0], "longitude": self.location[1]}
        
        if self.jurisdiction:
            result["jurisdiction"] = [j.name for j in self.jurisdiction]
        
        if self.rate_limit:
            result["rate_limit"] = self.rate_limit
        
        if self.last_successful_connection:
            result["last_successful_connection"] = self.last_successful_connection.isoformat()
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LPRSource':
        """Create from dictionary representation."""
        source_type = LPRSourceType[data.get("source_type", "CUSTOM")]
        
        location = None
        if "location" in data:
            location = (data["location"].get("latitude"), data["location"].get("longitude"))
        
        jurisdiction = set()
        if "jurisdiction" in data:
            for j in data["jurisdiction"]:
                try:
                    jurisdiction.add(PlateJurisdiction[j])
                except KeyError:
                    jurisdiction.add(PlateJurisdiction.UNKNOWN)
        
        if not jurisdiction:
            jurisdiction = {PlateJurisdiction.UNKNOWN}
        
        last_connection = None
        if "last_successful_connection" in data:
            try:
                last_connection = datetime.fromisoformat(data["last_successful_connection"])
            except ValueError:
                pass
        
        credentials = LPRSourceCredentials()
        if "credentials" in data:
            cred_data = data["credentials"]
            credentials = LPRSourceCredentials(
                username=cred_data.get("username", ""),
                password=cred_data.get("password", ""),
                api_key=cred_data.get("api_key", ""),
                certificate_path=cred_data.get("certificate_path", ""),
                token=cred_data.get("token", "")
            )
            if "token_expiration" in cred_data:
                try:
                    credentials.token_expiration = datetime.fromisoformat(cred_data["token_expiration"])
                except ValueError:
                    pass
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Unnamed LPR Source"),
            source_type=source_type,
            endpoint=data.get("endpoint", ""),
            credentials=credentials,
            location=location,
            jurisdiction=jurisdiction,
            active=data.get("active", True),
            rate_limit=data.get("rate_limit"),
            connection_timeout=data.get("connection_timeout", 30),
            last_successful_connection=last_connection,
            metadata=data.get("metadata", {})
        )


@dataclass
class LicensePlateDetection:
    """Represents a single license plate detection."""
    id: str
    plate_number: str
    source_id: str
    timestamp: datetime
    confidence: float  # 0.0 to 1.0
    jurisdiction: PlateJurisdiction = PlateJurisdiction.UNKNOWN
    location: Optional[Tuple[float, float]] = None  # lat, lon
    confidence_level: PlateConfidenceLevel = PlateConfidenceLevel.MEDIUM
    vehicle_type: Optional[str] = None
    vehicle_color: Optional[str] = None
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    direction_of_travel: Optional[float] = None  # Degrees (0-359.9)
    estimated_speed: Optional[float] = None  # KPH
    image_data: Optional[bytes] = None
    image_quality: Optional[float] = None  # 0.0 to 1.0
    ocr_confidence: Optional[Dict[str, float]] = None  # Character-by-character confidence
    plate_type: Optional[str] = None  # Standard, temporary, commercial, etc.
    raw_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and initialize detection after creation."""
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Normalize plate number (remove spaces, standardize characters)
        self.plate_number = self._normalize_plate_number(self.plate_number)
        
        # Assign confidence level based on confidence score
        if self.confidence <= 0.3:
            self.confidence_level = PlateConfidenceLevel.VERY_LOW
        elif self.confidence <= 0.5:
            self.confidence_level = PlateConfidenceLevel.LOW
        elif self.confidence <= 0.7:
            self.confidence_level = PlateConfidenceLevel.MEDIUM
        elif self.confidence <= 0.9:
            self.confidence_level = PlateConfidenceLevel.HIGH
        else:
            self.confidence_level = PlateConfidenceLevel.VERY_HIGH
    
    def _normalize_plate_number(self, plate: str) -> str:
        """Normalize a license plate number."""
        # Convert to uppercase
        plate = plate.upper()
        
        # Remove spaces and special characters
        plate = ''.join(c for c in plate if c.isalnum())
        
        # Convert similar characters
        replacements = {
            'O': '0',  # Letter O to digit 0
            'I': '1',  # Letter I to digit 1
            'Z': '2',  # Sometimes misread
            'S': '5',  # Sometimes misread
            'B': '8',  # Sometimes misread
        }
        
        # Only apply replacements if confidence is low
        if self.confidence < 0.6:
            for old, new in replacements.items():
                plate = plate.replace(old, new)
        
        return plate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "id": self.id,
            "plate_number": self.plate_number,
            "source_id": self.source_id,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.name,
            "jurisdiction": self.jurisdiction.name,
        }
        
        if self.location:
            result["location"] = {"latitude": self.location[0], "longitude": self.location[1]}
        
        if self.vehicle_type:
            result["vehicle_type"] = self.vehicle_type
        
        if self.vehicle_color:
            result["vehicle_color"] = self.vehicle_color
        
        if self.vehicle_make:
            result["vehicle_make"] = self.vehicle_make
        
        if self.vehicle_model:
            result["vehicle_model"] = self.vehicle_model
        
        if self.direction_of_travel is not None:
            result["direction_of_travel"] = self.direction_of_travel
        
        if self.estimated_speed is not None:
            result["estimated_speed"] = self.estimated_speed
        
        if self.image_quality is not None:
            result["image_quality"] = self.image_quality
        
        if self.ocr_confidence:
            result["ocr_confidence"] = self.ocr_confidence
        
        if self.plate_type:
            result["plate_type"] = self.plate_type
            
        # Exclude image_data and raw_data due to size
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LicensePlateDetection':
        """Create from dictionary representation."""
        try:
            timestamp = datetime.fromisoformat(data["timestamp"])
        except (KeyError, ValueError):
            timestamp = datetime.now()
        
        try:
            jurisdiction = PlateJurisdiction[data.get("jurisdiction", "UNKNOWN")]
        except KeyError:
            jurisdiction = PlateJurisdiction.UNKNOWN
        
        try:
            confidence_level = PlateConfidenceLevel[data.get("confidence_level", "MEDIUM")]
        except KeyError:
            confidence_level = PlateConfidenceLevel.MEDIUM
        
        location = None
        if "location" in data:
            loc_data = data["location"]
            if isinstance(loc_data, dict) and "latitude" in loc_data and "longitude" in loc_data:
                location = (loc_data["latitude"], loc_data["longitude"])
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            plate_number=data["plate_number"],
            source_id=data["source_id"],
            timestamp=timestamp,
            confidence=data.get("confidence", 0.5),
            confidence_level=confidence_level,
            jurisdiction=jurisdiction,
            location=location,
            vehicle_type=data.get("vehicle_type"),
            vehicle_color=data.get("vehicle_color"),
            vehicle_make=data.get("vehicle_make"),
            vehicle_model=data.get("vehicle_model"),
            direction_of_travel=data.get("direction_of_travel"),
            estimated_speed=data.get("estimated_speed"),
            image_quality=data.get("image_quality"),
            ocr_confidence=data.get("ocr_confidence"),
            plate_type=data.get("plate_type"),
            raw_data=data.get("raw_data"),
        )


class LPRAdapter:
    """Base adapter class for different LPR systems."""
    
    def __init__(self, source: LPRSource):
        """Initialize the adapter with the source configuration."""
        self.source = source
        self.connected = False
        self.last_error = None
        self.last_connect_time = None
        self.session = None
    
    async def connect(self) -> bool:
        """Establish connection to the LPR source."""
        raise NotImplementedError("Subclasses must implement connect()")
    
    async def disconnect(self) -> bool:
        """Disconnect from the LPR source."""
        raise NotImplementedError("Subclasses must implement disconnect()")
    
    async def get_detections(self, since: Optional[datetime] = None, limit: int = 100) -> List[LicensePlateDetection]:
        """Retrieve license plate detections from the source."""
        raise NotImplementedError("Subclasses must implement get_detections()")
    
    async def get_detection_stream(self, callback: Callable[[LicensePlateDetection], None]):
        """Set up a real-time stream of detections."""
        raise NotImplementedError("Subclasses must implement get_detection_stream()")
    
    async def is_available(self) -> bool:
        """Check if the LPR source is available."""
        try:
            return await self.connect()
        except Exception as e:
            self.last_error = str(e)
            return False
        finally:
            await self.disconnect()


class DedicatedLPRAdapter(LPRAdapter):
    """Adapter for dedicated LPR systems (e.g., Genetec AutoVu)."""
    
    async def connect(self) -> bool:
        """Establish connection to the dedicated LPR system."""
        self.last_connect_time = datetime.now()
        
        try:
            # Create HTTP session for API requests
            self.session = requests.Session()
            
            # Set up authentication based on credential type
            if self.source.credentials.api_key:
                self.session.headers.update({"X-API-Key": self.source.credentials.api_key})
            elif self.source.credentials.token:
                self.session.headers.update({"Authorization": f"Bearer {self.source.credentials.token}"})
            elif self.source.credentials.username and self.source.credentials.password:
                # Most dedicated systems use token auth, so first get a token
                auth_endpoint = f"{self.source.endpoint}/auth"
                auth_data = {
                    "username": self.source.credentials.username,
                    "password": self.source.credentials.password
                }
                
                response = self.session.post(
                    auth_endpoint, 
                    json=auth_data, 
                    timeout=self.source.connection_timeout
                )
                
                if response.status_code == 200:
                    auth_result = response.json()
                    if "token" in auth_result:
                        self.source.credentials.token = auth_result["token"]
                        if "expiration" in auth_result:
                            self.source.credentials.token_expiration = datetime.fromisoformat(auth_result["expiration"])
                        
                        self.session.headers.update({"Authorization": f"Bearer {self.source.credentials.token}"})
                    else:
                        raise ConnectionError(f"Authentication succeeded but no token returned: {auth_result}")
                else:
                    raise ConnectionError(f"Authentication failed with status {response.status_code}: {response.text}")
            
            # Test connection with a simple request
            test_endpoint = f"{self.source.endpoint}/system/status"
            response = self.session.get(test_endpoint, timeout=self.source.connection_timeout)
            
            if response.status_code == 200:
                self.connected = True
                self.source.last_successful_connection = datetime.now()
                logger.info(f"Successfully connected to dedicated LPR system: {self.source.name}")
                return True
            else:
                self.connected = False
                self.last_error = f"Connection test failed with status {response.status_code}: {response.text}"
                logger.error(self.last_error)
                return False
                
        except Exception as e:
            self.connected = False
            self.last_error = str(e)
            logger.error(f"Failed to connect to dedicated LPR system {self.source.name}: {str(e)}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from the dedicated LPR system."""
        if self.session:
            try:
                # Some systems have explicit logout endpoints
                if self.source.credentials.token:
                    try:
                        logout_endpoint = f"{self.source.endpoint}/auth/logout"
                        self.session.post(logout_endpoint, timeout=self.source.connection_timeout)
                    except Exception as e:
                        logger.warning(f"Error during logout from {self.source.name}: {str(e)}")
                
                self.session.close()
                self.session = None
                self.connected = False
                return True
            except Exception as e:
                logger.error(f"Error disconnecting from {self.source.name}: {str(e)}")
                return False
        
        return True
    
    async def get_detections(self, since: Optional[datetime] = None, limit: int = 100) -> List[LicensePlateDetection]:
        """Retrieve license plate detections from the dedicated LPR system."""
        if not self.connected and not await self.connect():
            raise ConnectionError(f"Not connected to LPR system: {self.last_error}")
        
        try:
            # Construct query parameters
            params = {"limit": limit}
            if since:
                params["since"] = since.isoformat()
            
            # Fetch detections
            detections_endpoint = f"{self.source.endpoint}/detections"
            response = self.session.get(
                detections_endpoint,
                params=params,
                timeout=self.source.connection_timeout
            )
            
            if response.status_code != 200:
                raise ConnectionError(f"Failed to fetch detections: {response.status_code} - {response.text}")
            
            # Parse response
            detection_data = response.json()
            if not isinstance(detection_data, list):
                detection_data = detection_data.get("detections", [])
            
            # Convert to LicensePlateDetection objects
            result = []
            for data in detection_data:
                try:
                    # Normalize data structure based on the LPR system's format
                    normalized = self._normalize_detection_data(data)
                    detection = LicensePlateDetection(
                        id=normalized.get("id", str(uuid.uuid4())),
                        plate_number=normalized["plate_number"],
                        source_id=self.source.id,
                        timestamp=normalized["timestamp"],
                        confidence=normalized["confidence"],
                        jurisdiction=normalized.get("jurisdiction", PlateJurisdiction.UNKNOWN),
                        location=normalized.get("location"),
                        vehicle_type=normalized.get("vehicle_type"),
                        vehicle_color=normalized.get("vehicle_color"),
                        vehicle_make=normalized.get("vehicle_make"),
                        vehicle_model=normalized.get("vehicle_model"),
                        direction_of_travel=normalized.get("direction_of_travel"),
                        estimated_speed=normalized.get("estimated_speed"),
                        image_quality=normalized.get("image_quality"),
                        plate_type=normalized.get("plate_type"),
                        raw_data=data
                    )
                    
                    # Fetch image if available and not included in detection data
                    if "image_url" in normalized and not normalized.get("image_data"):
                        try:
                            img_response = self.session.get(normalized["image_url"], timeout=self.source.connection_timeout)
                            if img_response.status_code == 200:
                                detection.image_data = img_response.content
                        except Exception as img_err:
                            logger.warning(f"Failed to fetch detection image: {str(img_err)}")
                    
                    result.append(detection)
                except KeyError as e:
                    logger.warning(f"Invalid detection data format: {str(e)} - {data}")
                except Exception as e:
                    logger.warning(f"Error processing detection: {str(e)} - {data}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching detections from {self.source.name}: {str(e)}")
            raise
    
    async def get_detection_stream(self, callback: Callable[[LicensePlateDetection], None]):
        """Set up a real-time stream of detections using polling or websockets."""
        if not self.connected and not await self.connect():
            raise ConnectionError(f"Not connected to LPR system: {self.last_error}")
        
        try:
            # Check if the system supports websocket streaming
            ws_supported = False
            system_info_endpoint = f"{self.source.endpoint}/system/info"
            
            try:
                response = self.session.get(system_info_endpoint, timeout=self.source.connection_timeout)
                if response.status_code == 200:
                    system_info = response.json()
                    ws_supported = system_info.get("websocket_streaming", False)
            except Exception:
                ws_supported = False
            
            if ws_supported:
                # Using websocket for real-time streaming
                try:
                    import websockets
                    import json
                    import asyncio
                    
                    # Get websocket URL and token
                    ws_endpoint_response = self.session.get(
                        f"{self.source.endpoint}/streaming/token",
                        timeout=self.source.connection_timeout
                    )
                    
                    if ws_endpoint_response.status_code != 200:
                        raise ConnectionError(f"Failed to get websocket token: {ws_endpoint_response.status_code}")
                    
                    ws_data = ws_endpoint_response.json()
                    ws_url = ws_data.get("websocket_url")
                    ws_token = ws_data.get("token")
                    
                    if not ws_url or not ws_token:
                        raise ValueError("Missing websocket URL or token")
                    
                    # Connect to websocket
                    async with websockets.connect(
                        f"{ws_url}?token={ws_token}",
                        extra_headers=self.session.headers
                    ) as websocket:
                        logger.info(f"Established websocket connection to {self.source.name}")
                        
                        # Subscribe to detections
                        await websocket.send(json.dumps({
                            "action": "subscribe",
                            "channel": "detections"
                        }))
                        
                        # Process incoming messages
                        async for message in websocket:
                            try:
                                data = json.loads(message)
                                normalized = self._normalize_detection_data(data)
                                detection = LicensePlateDetection(
                                    id=normalized.get("id", str(uuid.uuid4())),
                                    plate_number=normalized["plate_number"],
                                    source_id=self.source.id,
                                    timestamp=normalized["timestamp"],
                                    confidence=normalized["confidence"],
                                    jurisdiction=normalized.get("jurisdiction", PlateJurisdiction.UNKNOWN),
                                    location=normalized.get("location"),
                                    vehicle_type=normalized.get("vehicle_type"),
                                    vehicle_color=normalized.get("vehicle_color"),
                                    vehicle_make=normalized.get("vehicle_make"),
                                    vehicle_model=normalized.get("vehicle_model"),
                                    direction_of_travel=normalized.get("direction_of_travel"),
                                    estimated_speed=normalized.get("estimated_speed"),
                                    image_quality=normalized.get("image_quality"),
                                    plate_type=normalized.get("plate_type"),
                                    raw_data=data
                                )
                                callback(detection)
                            except Exception as e:
                                logger.error(f"Error processing websocket message: {str(e)}")
                
                except (ImportError, Exception) as e:
                    logger.error(f"Failed to set up websocket streaming: {str(e)}")
                    ws_supported = False
            
            if not ws_supported:
                # Fall back to polling if websockets not supported
                logger.info(f"Using polling for detection streaming from {self.source.name}")
                
                last_check = datetime.now() - timedelta(seconds=10)
                while self.connected:
                    try:
                        detections = await self.get_detections(since=last_check)
                        if detections:
                            for detection in detections:
                                callback(detection)
                            # Update last check time to most recent detection
                            newest_detection = max(detections, key=lambda d: d.timestamp)
                            last_check = newest_detection.timestamp
                        
                        # Rate limiting
                        if self.source.rate_limit:
                            await asyncio.sleep(60 / self.source.rate_limit)
                        else:
                            await asyncio.sleep(1)
                    except Exception as e:
                        logger.error(f"Error during detection polling: {str(e)}")
                        await asyncio.sleep(5)  # Wait before retry
        
        except Exception as e:
            logger.error(f"Error setting up detection stream for {self.source.name}: {str(e)}")
            raise
    
    def _normalize_detection_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize detection data from the specific LPR system format."""
        normalized = {}
        
        # Extract standard fields with different possible names
        
        # ID
        for id_field in ["id", "detection_id", "read_id", "uid"]:
            if id_field in data:
                normalized["id"] = str(data[id_field])
                break
        else:
            normalized["id"] = str(uuid.uuid4())
        
        # Plate number
        for plate_field in ["plate_number", "plate", "plate_text", "license_plate", "plate_string"]:
            if plate_field in data:
                normalized["plate_number"] = str(data[plate_field])
                break
        else:
            raise KeyError("No plate number field found in detection data")
        
        # Timestamp
        timestamp = None
        for ts_field in ["timestamp", "date", "date_time", "capture_time", "read_time"]:
            if ts_field in data:
                ts_value = data[ts_field]
                if isinstance(ts_value, (int, float)):
                    # Unix timestamp (seconds or milliseconds)
                    if ts_value > 1e11:  # Milliseconds
                        timestamp = datetime.fromtimestamp(ts_value / 1000)
                    else:  # Seconds
                        timestamp = datetime.fromtimestamp(ts_value)
                else:
                    # String format
                    try:
                        timestamp = datetime.fromisoformat(ts_value.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        try:
                            # Try common formats
                            for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"]:
                                try:
                                    timestamp = datetime.strptime(ts_value, fmt)
                                    break
                                except ValueError:
                                    continue
                        except Exception:
                            pass
                
                if timestamp:
                    break
        
        if not timestamp:
            timestamp = datetime.now()
        
        normalized["timestamp"] = timestamp
        
        # Confidence
        confidence = 0.5  # Default
        for conf_field in ["confidence", "confidence_score", "ocr_confidence", "read_confidence"]:
            if conf_field in data:
                conf_value = data[conf_field]
                if isinstance(conf_value, (int, float)):
                    # Normalize to 0-1 range
                    if conf_value > 0 and conf_value <= 1:
                        confidence = conf_value
                    elif conf_value > 1 and conf_value <= 100:
                        confidence = conf_value / 100
                    break
        
        normalized["confidence"] = confidence
        
        # Location
        location = None
        if "location" in data:
            loc_data = data["location"]
            if isinstance(loc_data, dict):
                lat = loc_data.get("latitude", loc_data.get("lat"))
                lon = loc_data.get("longitude", loc_data.get("lon", loc_data.get("lng")))
                if lat is not None and lon is not None:
                    location = (float(lat), float(lon))
        elif all(k in data for k in ["latitude", "longitude"]):
            location = (float(data["latitude"]), float(data["longitude"]))
        elif all(k in data for k in ["lat", "lon"]) or all(k in data for k in ["lat", "lng"]):
            lat = data["lat"]
            lon = data.get("lon", data.get("lng"))
            location = (float(lat), float(lon))
        
        if location:
            normalized["location"] = location
        
        # Jurisdiction/State
        jurisdiction = PlateJurisdiction.UNKNOWN
        for juris_field in ["jurisdiction", "state", "province", "country"]:
            if juris_field in data:
                juris_value = data[juris_field]
                try:
                    # Try to match with enum values
                    jurisdiction = PlateJurisdiction[juris_value]
                    break
                except (KeyError, TypeError):
                    # Handle using custom logic based on common abbreviations
                    if isinstance(juris_value, str):
                        if juris_value.upper() == "CA":
                            jurisdiction = PlateJurisdiction.CALIFORNIA
                        # Add more state/province mappings as needed
                        else:
                            jurisdiction = PlateJurisdiction.CUSTOM
                            # Store original value in metadata
                            if "metadata" not in normalized:
                                normalized["metadata"] = {}
                            normalized["metadata"]["original_jurisdiction"] = juris_value
        
        normalized["jurisdiction"] = jurisdiction
        
        # Vehicle details
        for field in ["vehicle_type", "vehicle_color", "vehicle_make", "vehicle_model"]:
            alt_names = {
                "vehicle_type": ["vehicle_category", "type", "category"],
                "vehicle_color": ["color", "vehicle_colour"],
                "vehicle_make": ["make", "manufacturer"],
                "vehicle_model": ["model"]
            }
            
            for name in [field] + alt_names.get(field, []):
                if name in data:
                    normalized[field] = data[name]
                    break
        
        # Direction and speed
        for field, alt_names in {
            "direction_of_travel": ["direction", "heading", "travel_direction", "angle"],
            "estimated_speed": ["speed", "vehicle_speed"]
        }.items():
            for name in [field] + alt_names:
                if name in data:
                    value = data[name]
                    if isinstance(value, (int, float)):
                        normalized[field] = float(value)
                        break
                    elif isinstance(value, str):
                        try:
                            normalized[field] = float(value)
                            break
                        except ValueError:
                            pass
        
        # Image URL
        for img_field in ["image_url", "plate_image_url", "context_image_url", "image_link"]:
            if img_field in data:
                url = data[img_field]
                if url:
                    # Make relative URLs absolute
                    if url.startswith('/'):
                        base_url = self.source.endpoint
                        if base_url.endswith('/'):
                            base_url = base_url[:-1]
                        url = f"{base_url}{url}"
                    
                    normalized["image_url"] = url
                    break
        
        # Image data (if included directly)
        if "image_data" in data and data["image_data"]:
            img_data = data["image_data"]
            if isinstance(img_data, str) and img_data.startswith(('data:image', 'base64:')):
                import base64
                try:
                    # Extract base64 data
                    if ',' in img_data:
                        img_data = img_data.split(',', 1)[1]
                    elif img_data.startswith('base64:'):
                        img_data = img_data[7:]
                    
                    normalized["image_data"] = base64.b64decode(img_data)
                except Exception as e:
                    logger.error(f"Error decoding base64 image data: {str(e)}")
            elif isinstance(img_data, bytes):
                normalized["image_data"] = img_data
        
        return normalized


# Additional adapter implementations for other types of LPR sources would go here:
# - TrafficCameraAdapter
# - MobileLPRAdapter
# - TollboothAdapter
# - ParkingLPRAdapter
# - ThirdPartyAPIAdapter


class LPRCollector:
    """
    Main collector for license plate recognition data.
    
    This class manages connections to various LPR sources,
    collects and normalizes plate detections, and provides
    interfaces for real-time and historical access.
    """
    
    def __init__(self, config=None):
        """Initialize the LPR collector."""
        self.config = config or ConfigManager.get_config().data_collection.lpr_collector
        self.sources: Dict[str, LPRSource] = {}
        self.adapters: Dict[str, LPRAdapter] = {}
        self.active_streams: Dict[str, bool] = {}
        self.detection_queue = queue.Queue(maxsize=10000)
        self.validator = DataValidator([
            ValidationRule("plate_number", lambda p: len(p) >= 2 and len(p) <= 10),
            ValidationRule("confidence", lambda c: 0 <= c <= 1),
            ValidationRule("source_id", lambda s: s in self.sources)
        ])
        self.storage_manager = None
        self.running = False
        self.processing_thread = None
        self.stream_tasks = {}
        
        # Load sources from configuration
        self._load_sources()
        
        logger.info(f"LPR Collector initialized with {len(self.sources)} sources")
    
    def _load_sources(self):
        """Load LPR sources from configuration."""
        source_configs = getattr(self.config, 'sources', [])
        
        if not source_configs and hasattr(self.config, 'sources_file'):
            # Load from file if provided
            try:
                sources_path = self.config.sources_file
                with open(sources_path, 'r') as f:
                    if sources_path.endswith('.json'):
                        source_configs = json.load(f)
                    elif sources_path.endswith(('.yaml', '.yml')):
                        import yaml
                        source_configs = yaml.safe_load(f)
                    else:
                        logger.error(f"Unsupported sources file format: {sources_path}")
            except Exception as e:
                logger.error(f"Failed to load LPR sources from file: {str(e)}")
        
        # Process source configurations
        for source_data in source_configs:
            try:
                source = LPRSource.from_dict(source_data)
                self.sources[source.id] = source
            except Exception as e:
                logger.error(f"Error creating LPR source: {str(e)}")
    
    def add_source(self, source: LPRSource) -> str:
        """Add a new LPR source."""
        self.sources[source.id] = source
        logger.info(f"Added LPR source: {source.name} ({source.id})")
        return source.id
    
    def remove_source(self, source_id: str) -> bool:
        """Remove an LPR source."""
        if source_id in self.sources:
            # Stop any active streams
            self.stop_stream(source_id)
            
            # Remove adapter if exists
            if source_id in self.adapters:
                adapter = self.adapters.pop(source_id)
                asyncio.run(adapter.disconnect())
            
            # Remove source
            source = self.sources.pop(source_id)
            logger.info(f"Removed LPR source: {source.name} ({source_id})")
            return True
        
        return False
    
    def get_source(self, source_id: str) -> Optional[LPRSource]:
        """Get an LPR source by ID."""
        return self.sources.get(source_id)
    
    def list_sources(self) -> List[LPRSource]:
        """List all LPR sources."""
        return list(self.sources.values())
    
    def _get_adapter(self, source_id: str) -> LPRAdapter:
        """Get or create an adapter for the source."""
        if source_id not in self.adapters:
            source = self.sources.get(source_id)
            if not source:
                raise ValueError(f"Unknown source ID: {source_id}")
            
            # Create adapter based on source type
            if source.source_type == LPRSourceType.DEDICATED_LPR:
                adapter = DedicatedLPRAdapter(source)
            elif source.source_type == LPRSourceType.TRAFFIC_CAMERA:
                # This would be implemented separately
                adapter = DedicatedLPRAdapter(source)  # Placeholder
            elif source.source_type == LPRSourceType.MOBILE_LPR:
                adapter = DedicatedLPRAdapter(source)  # Placeholder
            elif source.source_type == LPRSourceType.TOLLBOOTH:
                adapter = DedicatedLPRAdapter(source)  # Placeholder
            elif source.source_type == LPRSourceType.PARKING:
                adapter = DedicatedLPRAdapter(source)  # Placeholder
            elif source.source_type == LPRSourceType.THIRD_PARTY_API:
                adapter = DedicatedLPRAdapter(source)  # Placeholder
            else:  # CUSTOM or undefined
                adapter = DedicatedLPRAdapter(source)  # Placeholder
            
            self.adapters[source_id] = adapter
        
        return self.adapters[source_id]
    
    async def get_detections(self, source_id: str, since: Optional[datetime] = None, limit: int = 100) -> List[LicensePlateDetection]:
        """Get historical detections from a source."""
        adapter = self._get_adapter(source_id)
        detections = await adapter.get_detections(since=since, limit=limit)
        
        # Apply validation
        valid_detections = []
        for detection in detections:
            if self.validator.validate(detection):
                valid_detections.append(detection)
            else:
                logger.warning(f"Invalid detection from {source_id}: {detection.plate_number}")
        
        return valid_detections
    
    def start_stream(self, source_id: str) -> bool:
        """Start streaming detections from a source."""
        if source_id not in self.sources:
            logger.error(f"Unknown source ID: {source_id}")
            return False
        
        if source_id in self.active_streams and self.active_streams[source_id]:
            logger.warning(f"Stream already active for source: {source_id}")
            return True
        
        try:
            # Set up asynchronous task for the stream
            async def start_stream_task():
                adapter = self._get_adapter(source_id)
                
                # Connect to the source
                if not await adapter.connect():
                    logger.error(f"Failed to connect to source: {source_id}")
                    return False
                
                # Start the detection stream
                try:
                    await adapter.get_detection_stream(
                        lambda detection: self.detection_queue.put(detection, block=False)
                    )
                except Exception as e:
                    logger.error(f"Error in detection stream for {source_id}: {str(e)}")
                finally:
                    await adapter.disconnect()
                    self.active_streams[source_id] = False
            
            # Create and start task
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            task = loop.create_task(start_stream_task())
            self.stream_tasks[source_id] = (loop, task)
            
            # Mark as active
            self.active_streams[source_id] = True
            
            logger.info(f"Started detection stream for source: {source_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start stream for {source_id}: {str(e)}")
            return False
    
    def stop_stream(self, source_id: str) -> bool:
        """Stop streaming detections from a source."""
        if source_id not in self.active_streams or not self.active_streams[source_id]:
            logger.warning(f"No active stream for source: {source_id}")
            return True
        
        try:
            # Cancel the task
            if source_id in self.stream_tasks:
                loop, task = self.stream_tasks[source_id]
                if not task.done():
                    task.cancel()
                self.stream_tasks.pop(source_id)
            
            # Mark as inactive
            self.active_streams[source_id] = False
            
            logger.info(f"Stopped detection stream for source: {source_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping stream for {source_id}: {str(e)}")
            return False
    
    def start(self):
        """Start the LPR collector, including processing thread."""
        if self.running:
            logger.warning("LPR Collector is already running")
            return
        
        try:
            # Initialize storage if needed
            if not self.storage_manager and hasattr(self.config, 'storage_enabled') and self.config.storage_enabled:
                from ultratrack.data_processing.storage_manager import StorageManager
                storage_config = getattr(self.config, 'storage_config', None)
                self.storage_manager = StorageManager(storage_config)
            
            # Start processing thread
            self.running = True
            self.processing_thread = threading.Thread(
                target=self._process_detections,
                daemon=True,
                name="LPRCollectorProcessor"
            )
            self.processing_thread.start()
            
            # Start streams for active sources
            for source_id, source in self.sources.items():
                if source.active:
                    self.start_stream(source_id)
            
            logger.info("LPR Collector started")
            
        except Exception as e:
            self.running = False
            logger.error(f"Error starting LPR Collector: {str(e)}")
            raise
    
    def stop(self):
        """Stop the LPR collector and all streams."""
        if not self.running:
            logger.warning("LPR Collector is not running")
            return
        
        try:
            # Stop all streams
            for source_id in list(self.active_streams.keys()):
                if self.active_streams[source_id]:
                    self.stop_stream(source_id)
            
            # Stop processing thread
            self.running = False
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
            
            # Clean up adapters
            for source_id, adapter in list(self.adapters.items()):
                asyncio.run(adapter.disconnect())
            self.adapters.clear()
            
            logger.info("LPR Collector stopped")
            
        except Exception as e:
            logger.error(f"Error stopping LPR Collector: {str(e)}")
    
    def _process_detections(self):
        """Process detections from the queue."""
        logger.info("Detection processing thread started")
        
        # Set up callback handler
        callbacks = []
        
        while self.running:
            try:
                # Get detection from queue with timeout
                try:
                    detection = self.detection_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Validate detection
                if not self.validator.validate(detection):
                    logger.warning(f"Invalid detection rejected: {detection.plate_number}")
                    continue
                
                # Store detection if storage is enabled
                if self.storage_manager:
                    try:
                        self.storage_manager.store_detection(detection)
                    except Exception as e:
                        logger.error(f"Error storing detection: {str(e)}")
                
                # Process callbacks
                for callback in callbacks:
                    try:
                        callback(detection)
                    except Exception as e:
                        logger.error(f"Error in detection callback: {str(e)}")
                
                # Mark queue item as processed
                self.detection_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing detection: {str(e)}")
                time.sleep(1.0)  # Prevent tight error loop
        
        logger.info("Detection processing thread stopped")
    
    def add_detection_callback(self, callback: Callable[[LicensePlateDetection], None]):
        """Add a callback to be notified of new detections."""
        self.callbacks.append(callback)
        return len(self.callbacks) - 1  # Return callback ID
    
    def remove_detection_callback(self, callback_id: int) -> bool:
        """Remove a detection callback by ID."""
        if 0 <= callback_id < len(self.callbacks):
            self.callbacks.pop(callback_id)
            return True
        return False
    
    async def check_source_status(self, source_id: str) -> Dict[str, Any]:
        """Check the status of an LPR source."""
        if source_id not in self.sources:
            raise ValueError(f"Unknown source ID: {source_id}")
        
        source = self.sources[source_id]
        adapter = self._get_adapter(source_id)
        
        # Build result with basic info
        result = {
            "id": source.id,
            "name": source.name,
            "source_type": source.source_type.name,
            "active": source.active,
            "streaming": self.active_streams.get(source_id, False),
            "connected": adapter.connected,
            "last_error": adapter.last_error,
            "last_successful_connection": source.last_successful_connection,
        }
        
        # Check availability if not already connected
        if not adapter.connected:
            try:
                available = await adapter.is_available()
                result["available"] = available
            except Exception as e:
                result["available"] = False
                result["check_error"] = str(e)
        else:
            result["available"] = True
        
        return result
    
    def search_detections(
        self,
        plate_number: Optional[str] = None,
        source_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        location: Optional[Tuple[float, float, float]] = None,  # lat, lon, radius_km
        limit: int = 100
    ) -> List[LicensePlateDetection]:
        """
        Search for license plate detections matching criteria.
        
        Args:
            plate_number: Full or partial plate number to match
            source_ids: List of source IDs to include
            start_time: Start of time range
            end_time: End of time range
            location: Tuple of (latitude, longitude, radius_km)
            limit: Maximum number of results
            
        Returns:
            List of matching detections
        """
        if not self.storage_manager:
            raise RuntimeError("Storage manager not initialized")
        
        # Build search query
        query = {}
        
        if plate_number:
            # Normalize plate number
            plate_number = ''.join(c for c in plate_number.upper() if c.isalnum())
            query["plate_number"] = plate_number
        
        if source_ids:
            query["source_ids"] = source_ids
        
        if start_time:
            query["start_time"] = start_time
        
        if end_time:
            query["end_time"] = end_time
        
        if location:
            query["location"] = location
        
        # Execute search
        return self.storage_manager.search_detections(query, limit=limit)
    
    def get_statistics(self, source_id: Optional[str] = None, time_period: str = "day") -> Dict[str, Any]:
        """
        Get statistics about LPR detections.
        
        Args:
            source_id: Optional source ID to filter
            time_period: 'hour', 'day', 'week', or 'month'
            
        Returns:
            Statistics about detections
        """
        if not self.storage_manager:
            raise RuntimeError("Storage manager not initialized")
        
        # Calculate time range
        end_time = datetime.now()
        if time_period == "hour":
            start_time = end_time - timedelta(hours=1)
        elif time_period == "day":
            start_time = end_time - timedelta(days=1)
        elif time_period == "week":
            start_time = end_time - timedelta(weeks=1)
        elif time_period == "month":
            start_time = end_time - timedelta(days=30)
        else:
            raise ValueError(f"Invalid time period: {time_period}")
        
        # Build query
        query = {
            "start_time": start_time,
            "end_time": end_time
        }
        
        if source_id:
            query["source_ids"] = [source_id]
        
        # Get statistics
        return self.storage_manager.get_detection_statistics(query)
