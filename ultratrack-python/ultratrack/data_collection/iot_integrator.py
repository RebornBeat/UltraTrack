"""
UltraTrack IoT Integrator Module

This module provides comprehensive integration with IoT devices for the UltraTrack system.
It handles connection management, data collection, and processing from various IoT sources:
- Bluetooth beacons
- Wi-Fi access points
- RFID readers
- Smart sensors
- Connected devices
- Smart city infrastructure

The module implements secure communication protocols, robust error handling,
and supports both push and pull data collection models.

Copyright (c) 2025 Your Organization
"""

import asyncio
import base64
import datetime
import hashlib
import hmac
import json
import logging
import os
import queue
import re
import socket
import ssl
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Third-party imports for IoT protocols
import paho.mqtt.client as mqtt
from bluepy.btle import Scanner, DefaultDelegate, BTLEException
from bleak import BleakClient, BleakScanner
import zigpy.application
import zigpy.device
import coap
import modbus_tk
import modbus_tk.defines as modbus_defines
import modbus_tk.modbus_tcp as modbus_tcp
import pywifi
from pywifi import const as wifi_const
from zeroconf import ServiceBrowser, ServiceStateChange, Zeroconf

# UltraTrack imports
from ultratrack.data_collection.data_validator import DataValidator, ValidationRule
from ultratrack.security.encryption import EncryptionManager
from ultratrack.compliance.privacy_manager import PrivacyManager, DataCategory
from ultratrack.infrastructure.service_discovery import ServiceDiscovery

# Module logger
logger = logging.getLogger(__name__)


class IoTDeviceType(Enum):
    """Enumeration of supported IoT device types."""
    BLUETOOTH_BEACON = auto()
    WIFI_AP = auto()
    RFID_READER = auto()
    ENVIRONMENT_SENSOR = auto()
    MOTION_SENSOR = auto()
    PRESENCE_DETECTOR = auto()
    SMART_LOCK = auto()
    ACCESS_CONTROL = auto()
    CAMERA_CONTROLLER = auto()
    SMART_LIGHT = auto()
    SMART_THERMOSTAT = auto()
    GPS_TRACKER = auto()
    TRAFFIC_SENSOR = auto()
    PARKING_SENSOR = auto()
    CROWD_COUNTER = auto()
    CUSTOM = auto()


class ConnectionProtocol(Enum):
    """Enumeration of supported communication protocols."""
    MQTT = auto()
    COAP = auto()
    BLE = auto()
    ZIGBEE = auto()
    ZWAVE = auto()
    WIFI_DIRECT = auto()
    HTTP = auto()
    HTTPS = auto()
    WS = auto()
    WSS = auto()
    MODBUS = auto()
    BACNET = auto()
    LORA = auto()
    SIGFOX = auto()
    CUSTOM = auto()


class ConnectionStatus(Enum):
    """Enumeration of device connection statuses."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    FAILED = auto()
    RECONNECTING = auto()
    STANDBY = auto()


class DataFormat(Enum):
    """Enumeration of supported data formats."""
    JSON = auto()
    XML = auto()
    BINARY = auto()
    TEXT = auto()
    CBOR = auto()
    PROTOBUF = auto()
    CSV = auto()
    KEY_VALUE = auto()
    CUSTOM = auto()


@dataclass
class IoTDeviceCredential:
    """Secure credential storage for IoT device authentication."""
    device_id: str
    credential_type: str  # "password", "token", "certificate", "key", etc.
    value: str = field(repr=False)  # Sensitive - don't include in repr
    expiration: Optional[datetime.datetime] = None
    
    def is_valid(self) -> bool:
        """Check if the credential is still valid (not expired)."""
        if not self.expiration:
            return True
        return datetime.datetime.now() < self.expiration
    
    @classmethod
    def create_token(cls, device_id: str, secret_key: str, 
                     expiration_hours: int = 24) -> 'IoTDeviceCredential':
        """
        Create a new time-limited token for device authentication.
        
        Args:
            device_id: Unique device identifier
            secret_key: Secret key for signing
            expiration_hours: Token validity period in hours
            
        Returns:
            IoTDeviceCredential with a signed token
        """
        expiration = datetime.datetime.now() + datetime.timedelta(hours=expiration_hours)
        expiration_str = expiration.isoformat()
        
        # Create payload
        payload = f"{device_id}:{expiration_str}"
        
        # Sign payload
        signature = hmac.new(
            secret_key.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Combine into token
        token = f"{payload}:{signature}"
        
        return cls(
            device_id=device_id,
            credential_type="token",
            value=token,
            expiration=expiration
        )


@dataclass
class IoTDeviceConfig:
    """Configuration for an IoT device connection."""
    device_id: str
    device_type: IoTDeviceType
    protocol: ConnectionProtocol
    address: str  # IP address, MAC address, URL, etc.
    port: Optional[int] = None
    credentials: Optional[IoTDeviceCredential] = None
    data_format: DataFormat = DataFormat.JSON
    connection_timeout_sec: int = 30
    retry_interval_sec: int = 60
    max_retries: int = 5
    encryption_enabled: bool = True
    compression_enabled: bool = False
    batch_size: int = 100
    polling_interval_sec: Optional[int] = None  # None for push-based devices
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate the configuration parameters."""
        # Basic validation
        if not self.device_id:
            logger.error("Device ID cannot be empty")
            return False
            
        if self.port and (self.port < 1 or self.port > 65535):
            logger.error(f"Invalid port number: {self.port}")
            return False
            
        if self.connection_timeout_sec < 1:
            logger.error(f"Invalid connection timeout: {self.connection_timeout_sec}")
            return False
            
        if self.retry_interval_sec < 1:
            logger.error(f"Invalid retry interval: {self.retry_interval_sec}")
            return False
            
        if self.max_retries < 0:
            logger.error(f"Invalid max retries: {self.max_retries}")
            return False
            
        if self.polling_interval_sec is not None and self.polling_interval_sec < 1:
            logger.error(f"Invalid polling interval: {self.polling_interval_sec}")
            return False
            
        if self.batch_size < 1:
            logger.error(f"Invalid batch size: {self.batch_size}")
            return False
            
        # Protocol-specific validation
        if self.protocol == ConnectionProtocol.MQTT and not self.port:
            self.port = 1883  # Default MQTT port
            
        if self.protocol == ConnectionProtocol.COAP and not self.port:
            self.port = 5683  # Default CoAP port
            
        if self.protocol == ConnectionProtocol.HTTP and not self.port:
            self.port = 80  # Default HTTP port
            
        if self.protocol == ConnectionProtocol.HTTPS and not self.port:
            self.port = 443  # Default HTTPS port
            
        # Credential validation if provided
        if self.credentials and not self.credentials.is_valid():
            logger.error(f"Credentials for device {self.device_id} have expired")
            return False
            
        return True


@dataclass
class IoTReading:
    """Represents a single reading from an IoT device."""
    device_id: str
    timestamp: datetime.datetime
    reading_type: str
    value: Any
    unit: Optional[str] = None
    sequence_number: Optional[int] = None
    quality: Optional[float] = None  # 0.0 to 1.0, higher is better
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reading to dictionary format."""
        return {
            "device_id": self.device_id,
            "timestamp": self.timestamp.isoformat(),
            "reading_type": self.reading_type,
            "value": self.value,
            "unit": self.unit,
            "sequence_number": self.sequence_number,
            "quality": self.quality,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IoTReading':
        """Create reading from dictionary format."""
        # Parse timestamp
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.datetime.fromisoformat(timestamp)
        
        return cls(
            device_id=data.get("device_id"),
            timestamp=timestamp,
            reading_type=data.get("reading_type"),
            value=data.get("value"),
            unit=data.get("unit"),
            sequence_number=data.get("sequence_number"),
            quality=data.get("quality"),
            metadata=data.get("metadata", {})
        )
    
    def validate(self) -> bool:
        """Validate reading data."""
        if not self.device_id:
            return False
            
        if not self.timestamp:
            return False
            
        if not self.reading_type:
            return False
            
        if self.value is None:
            return False
            
        # Ensure timestamp is not in the future
        now = datetime.datetime.now()
        if self.timestamp > now + datetime.timedelta(seconds=1):  # Allow 1s tolerance
            logger.warning(f"Reading from device {self.device_id} has future timestamp")
            return False
            
        # Validate quality if provided
        if self.quality is not None and (self.quality < 0.0 or self.quality > 1.0):
            logger.warning(f"Reading from device {self.device_id} has invalid quality value")
            return False
            
        return True


@dataclass
class IoTDevice:
    """Represents a connected IoT device."""
    config: IoTDeviceConfig
    status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    connection_object: Any = None  # Protocol-specific connection object
    last_connected: Optional[datetime.datetime] = None
    last_reading: Optional[datetime.datetime] = None
    error_count: int = 0
    readings_buffer: deque = field(default_factory=lambda: deque(maxlen=1000))
    lock: threading.RLock = field(default_factory=threading.RLock)
    
    def __post_init__(self):
        """Initialize device after creation."""
        if not isinstance(self.readings_buffer, deque):
            self.readings_buffer = deque(maxlen=1000)
            
    def update_status(self, new_status: ConnectionStatus) -> None:
        """Thread-safe update of device status."""
        with self.lock:
            old_status = self.status
            self.status = new_status
            
            if new_status == ConnectionStatus.CONNECTED:
                self.last_connected = datetime.datetime.now()
                self.error_count = 0
                
            # Log status change
            if old_status != new_status:
                logger.info(f"Device {self.config.device_id} status changed from {old_status} to {new_status}")
    
    def add_reading(self, reading: IoTReading) -> bool:
        """
        Add a new reading to the device's buffer.
        
        Args:
            reading: The reading to add
            
        Returns:
            bool: True if reading was added successfully
        """
        with self.lock:
            if not reading.validate():
                logger.warning(f"Invalid reading from device {self.config.device_id}")
                return False
                
            self.readings_buffer.append(reading)
            self.last_reading = datetime.datetime.now()
            return True
    
    def get_readings(self, max_count: int = 100) -> List[IoTReading]:
        """
        Get readings from the device buffer.
        
        Args:
            max_count: Maximum number of readings to retrieve
            
        Returns:
            List of readings
        """
        readings = []
        with self.lock:
            count = min(max_count, len(self.readings_buffer))
            for _ in range(count):
                readings.append(self.readings_buffer.popleft())
        return readings
    
    def reading_count(self) -> int:
        """Get the number of readings in the buffer."""
        with self.lock:
            return len(self.readings_buffer)
    
    def increment_error(self) -> int:
        """
        Increment the error count and return the new value.
        
        Returns:
            Current error count
        """
        with self.lock:
            self.error_count += 1
            return self.error_count
    
    def is_active(self) -> bool:
        """
        Check if the device is actively sending data.
        
        Returns:
            True if the device has sent data recently
        """
        if not self.last_reading:
            return False
            
        # Define "recent" based on polling interval if available
        threshold = datetime.timedelta(minutes=5)  # Default
        
        if self.config.polling_interval_sec:
            # Use 3x polling interval as threshold
            threshold = datetime.timedelta(seconds=self.config.polling_interval_sec * 3)
            
        return datetime.datetime.now() - self.last_reading < threshold


class BLEDelegate(DefaultDelegate):
    """Delegate for handling BLE device notifications."""
    
    def __init__(self, device_handler):
        """
        Initialize the BLE delegate.
        
        Args:
            device_handler: Callback to handle discovered devices
        """
        DefaultDelegate.__init__(self)
        self.device_handler = device_handler
        
    def handleDiscovery(self, dev, isNewDev, isNewData):
        """
        Handle discovered BLE devices.
        
        Args:
            dev: Discovered device
            isNewDev: True if this is a new device
            isNewData: True if this is new data for an existing device
        """
        if isNewDev or isNewData:
            self.device_handler(dev)


class IoTProtocolHandler:
    """Base class for protocol-specific handlers."""
    
    def __init__(self, integrator: 'IoTIntegrator'):
        """
        Initialize the protocol handler.
        
        Args:
            integrator: Parent IoT integrator
        """
        self.integrator = integrator
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def connect(self, device: IoTDevice) -> bool:
        """
        Connect to an IoT device.
        
        Args:
            device: Device to connect to
            
        Returns:
            True if connection successful
        """
        raise NotImplementedError("Subclasses must implement connect()")
        
    def disconnect(self, device: IoTDevice) -> bool:
        """
        Disconnect from an IoT device.
        
        Args:
            device: Device to disconnect from
            
        Returns:
            True if disconnection successful
        """
        raise NotImplementedError("Subclasses must implement disconnect()")
        
    def read_data(self, device: IoTDevice) -> List[IoTReading]:
        """
        Read data from an IoT device.
        
        Args:
            device: Device to read from
            
        Returns:
            List of readings
        """
        raise NotImplementedError("Subclasses must implement read_data()")
        
    def send_command(self, device: IoTDevice, command: str, params: Dict[str, Any] = None) -> bool:
        """
        Send a command to an IoT device.
        
        Args:
            device: Device to send command to
            command: Command name
            params: Command parameters
            
        Returns:
            True if command was sent successfully
        """
        raise NotImplementedError("Subclasses must implement send_command()")
        
    def configure(self, device: IoTDevice, config: Dict[str, Any]) -> bool:
        """
        Configure an IoT device.
        
        Args:
            device: Device to configure
            config: Configuration parameters
            
        Returns:
            True if configuration was successful
        """
        raise NotImplementedError("Subclasses must implement configure()")
    
    def parse_data(self, device: IoTDevice, raw_data: Any) -> List[IoTReading]:
        """
        Parse raw data from an IoT device.
        
        Args:
            device: Source device
            raw_data: Raw data from device
            
        Returns:
            List of parsed readings
        """
        try:
            # Handle different data formats
            if device.config.data_format == DataFormat.JSON:
                return self._parse_json(device, raw_data)
            elif device.config.data_format == DataFormat.XML:
                return self._parse_xml(device, raw_data)
            elif device.config.data_format == DataFormat.BINARY:
                return self._parse_binary(device, raw_data)
            elif device.config.data_format == DataFormat.TEXT:
                return self._parse_text(device, raw_data)
            elif device.config.data_format == DataFormat.KEY_VALUE:
                return self._parse_key_value(device, raw_data)
            else:
                self.logger.warning(f"Unsupported data format: {device.config.data_format}")
                return []
        except Exception as e:
            self.logger.error(f"Error parsing data from device {device.config.device_id}: {str(e)}")
            return []
    
    def _parse_json(self, device: IoTDevice, raw_data: Union[str, bytes]) -> List[IoTReading]:
        """Parse JSON data from device."""
        if isinstance(raw_data, bytes):
            raw_data = raw_data.decode('utf-8')
            
        try:
            data = json.loads(raw_data)
            readings = []
            
            # Handle both single readings and arrays
            if isinstance(data, list):
                for item in data:
                    reading = self._create_reading_from_json(device, item)
                    if reading:
                        readings.append(reading)
            else:
                reading = self._create_reading_from_json(device, data)
                if reading:
                    readings.append(reading)
                    
            return readings
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from device {device.config.device_id}: {str(e)}")
            return []
    
    def _create_reading_from_json(self, device: IoTDevice, data: Dict[str, Any]) -> Optional[IoTReading]:
        """Create a reading from JSON data."""
        try:
            # Handle different JSON structures
            if 'reading_type' in data and 'value' in data:
                # Standard format
                return IoTReading(
                    device_id=device.config.device_id,
                    timestamp=datetime.datetime.now(),
                    reading_type=data.get('reading_type'),
                    value=data.get('value'),
                    unit=data.get('unit'),
                    sequence_number=data.get('sequence_number'),
                    quality=data.get('quality'),
                    metadata=data.get('metadata', {})
                )
            elif len(data) == 1:
                # Simple key-value format
                reading_type = list(data.keys())[0]
                value = data[reading_type]
                return IoTReading(
                    device_id=device.config.device_id,
                    timestamp=datetime.datetime.now(),
                    reading_type=reading_type,
                    value=value,
                    unit=None,
                    sequence_number=None,
                    quality=None,
                    metadata={}
                )
            elif 'data' in data and isinstance(data['data'], dict):
                # Nested format
                readings = []
                for reading_type, value in data['data'].items():
                    unit = None
                    if isinstance(value, dict) and 'value' in value and 'unit' in value:
                        unit = value['unit']
                        value = value['value']
                        
                    reading = IoTReading(
                        device_id=device.config.device_id,
                        timestamp=datetime.datetime.now(),
                        reading_type=reading_type,
                        value=value,
                        unit=unit,
                        sequence_number=data.get('sequence_number'),
                        quality=data.get('quality'),
                        metadata=data.get('metadata', {})
                    )
                    readings.append(reading)
                return readings
            else:
                # Try to handle unknown format
                self.logger.warning(f"Unknown JSON format from device {device.config.device_id}")
                reading_type = "unknown"
                return IoTReading(
                    device_id=device.config.device_id,
                    timestamp=datetime.datetime.now(),
                    reading_type=reading_type,
                    value=data,
                    unit=None,
                    sequence_number=None,
                    quality=None,
                    metadata={}
                )
        except Exception as e:
            self.logger.error(f"Error creating reading from JSON for device {device.config.device_id}: {str(e)}")
            return None
    
    def _parse_xml(self, device: IoTDevice, raw_data: Union[str, bytes]) -> List[IoTReading]:
        """Parse XML data from device."""
        # Implementation would use xml.etree.ElementTree or similar
        # This is a placeholder for the implementation
        self.logger.warning(f"XML parsing not fully implemented for device {device.config.device_id}")
        return []
    
    def _parse_binary(self, device: IoTDevice, raw_data: bytes) -> List[IoTReading]:
        """Parse binary data from device."""
        # Implementation would depend on device-specific binary format
        # This is a placeholder for the implementation
        self.logger.warning(f"Binary parsing not fully implemented for device {device.config.device_id}")
        return []
    
    def _parse_text(self, device: IoTDevice, raw_data: Union[str, bytes]) -> List[IoTReading]:
        """Parse text data from device."""
        if isinstance(raw_data, bytes):
            raw_data = raw_data.decode('utf-8')
            
        lines = raw_data.strip().split('\n')
        readings = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to parse as key=value format
            if '=' in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    reading_type = parts[0].strip()
                    value = parts[1].strip()
                    
                    # Try to convert value to numeric if possible
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # Keep as string
                        
                    reading = IoTReading(
                        device_id=device.config.device_id,
                        timestamp=datetime.datetime.now(),
                        reading_type=reading_type,
                        value=value,
                        unit=None,
                        sequence_number=None,
                        quality=None,
                        metadata={}
                    )
                    readings.append(reading)
            else:
                # Try to parse as CSV or other formats
                self.logger.warning(f"Unrecognized text format from device {device.config.device_id}")
                
        return readings
    
    def _parse_key_value(self, device: IoTDevice, raw_data: Union[str, bytes, Dict[str, Any]]) -> List[IoTReading]:
        """Parse key-value data from device."""
        if isinstance(raw_data, bytes):
            raw_data = raw_data.decode('utf-8')
            
        if isinstance(raw_data, str):
            # Try to parse as JSON first
            try:
                raw_data = json.loads(raw_data)
            except json.JSONDecodeError:
                # Not JSON, try other formats
                return self._parse_text(device, raw_data)
                
        if isinstance(raw_data, dict):
            readings = []
            
            for key, value in raw_data.items():
                reading = IoTReading(
                    device_id=device.config.device_id,
                    timestamp=datetime.datetime.now(),
                    reading_type=key,
                    value=value,
                    unit=None,
                    sequence_number=None,
                    quality=None,
                    metadata={}
                )
                readings.append(reading)
                
            return readings
        
        return []


class MQTTProtocolHandler(IoTProtocolHandler):
    """Handler for MQTT protocol."""
    
    def __init__(self, integrator: 'IoTIntegrator'):
        """Initialize the MQTT handler."""
        super().__init__(integrator)
        self.clients = {}  # Maps device_id to MQTT client
        self.topics = defaultdict(set)  # Maps device_id to set of subscribed topics
        
    def connect(self, device: IoTDevice) -> bool:
        """Connect to an MQTT device."""
        try:
            client_id = f"ultratrack-{uuid.uuid4().hex[:8]}"
            client = mqtt.Client(client_id=client_id)
            
            # Set up callbacks
            client.on_connect = lambda client, userdata, flags, rc: self._on_connect(client, device, flags, rc)
            client.on_disconnect = lambda client, userdata, rc: self._on_disconnect(client, device, rc)
            client.on_message = lambda client, userdata, msg: self._on_message(client, device, msg)
            
            # Set up authentication if provided
            if device.config.credentials:
                if device.config.credentials.credential_type == "password":
                    username, password = device.config.credentials.value.split(':', 1)
                    client.username_pw_set(username, password)
                elif device.config.credentials.credential_type == "token":
                    client.username_pw_set(device.config.device_id, device.config.credentials.value)
                    
            # Set up TLS if using secure connection
            if device.config.encryption_enabled:
                client.tls_set(
                    ca_certs=None,  # Could specify CA cert path
                    certfile=None,  # Could specify client cert path
                    keyfile=None,   # Could specify client key path
                    tls_version=ssl.PROTOCOL_TLS,
                    ciphers=None
                )
                
            # Connect to broker
            host = device.config.address
            port = device.config.port or 1883
            
            device.update_status(ConnectionStatus.CONNECTING)
            client.connect_async(host, port, keepalive=60)
            client.loop_start()
            
            # Store client in map
            self.clients[device.config.device_id] = client
            device.connection_object = client
            
            # Wait for connection to complete
            timeout = device.config.connection_timeout_sec
            start_time = time.time()
            while device.status != ConnectionStatus.CONNECTED:
                if time.time() - start_time > timeout:
                    client.loop_stop()
                    device.update_status(ConnectionStatus.FAILED)
                    return False
                time.sleep(0.1)
                
            return device.status == ConnectionStatus.CONNECTED
            
        except Exception as e:
            self.logger.error(f"Error connecting to MQTT device {device.config.device_id}: {str(e)}")
            device.update_status(ConnectionStatus.FAILED)
            return False
            
    def disconnect(self, device: IoTDevice) -> bool:
        """Disconnect from an MQTT device."""
        try:
            client = self.clients.get(device.config.device_id)
            if not client:
                return True  # Already disconnected
                
            client.loop_stop()
            client.disconnect()
            
            device.update_status(ConnectionStatus.DISCONNECTED)
            del self.clients[device.config.device_id]
            self.topics.pop(device.config.device_id, None)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from MQTT device {device.config.device_id}: {str(e)}")
            return False
            
    def read_data(self, device: IoTDevice) -> List[IoTReading]:
        """
        Read data from an MQTT device.
        
        Note: MQTT is primarily a push-based protocol. This method checks if
        any messages have been received and processed for the device.
        """
        # For MQTT, reading is done via the message callback
        # Just return any readings in the buffer
        return device.get_readings()
            
    def send_command(self, device: IoTDevice, command: str, params: Dict[str, Any] = None) -> bool:
        """Send a command to an MQTT device."""
        try:
            client = self.clients.get(device.config.device_id)
            if not client or device.status != ConnectionStatus.CONNECTED:
                self.logger.error(f"Cannot send command to disconnected device: {device.config.device_id}")
                return False
                
            # Construct command topic
            base_topic = device.config.metadata.get("base_topic", f"devices/{device.config.device_id}")
            command_topic = f"{base_topic}/commands/{command}"
            
            # Construct payload
            payload = json.dumps(params) if params else "{}"
            
            # Publish command
            result = client.publish(command_topic, payload, qos=1)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
            
        except Exception as e:
            self.logger.error(f"Error sending command to MQTT device {device.config.device_id}: {str(e)}")
            return False
            
    def configure(self, device: IoTDevice, config: Dict[str, Any]) -> bool:
        """Configure an MQTT device."""
        # For MQTT devices, configuration is typically done by sending a command
        return self.send_command(device, "configure", config)
        
    def subscribe(self, device: IoTDevice, topic: str) -> bool:
        """
        Subscribe to an MQTT topic.
        
        Args:
            device: Device to subscribe for
            topic: Topic to subscribe to
            
        Returns:
            True if subscription was successful
        """
        try:
            client = self.clients.get(device.config.device_id)
            if not client or device.status != ConnectionStatus.CONNECTED:
                self.logger.error(f"Cannot subscribe for disconnected device: {device.config.device_id}")
                return False
                
            result = client.subscribe(topic, qos=1)
            if result[0] == mqtt.MQTT_ERR_SUCCESS:
                self.topics[device.config.device_id].add(topic)
                return True
            else:
                self.logger.error(f"Failed to subscribe to topic {topic} for device {device.config.device_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error subscribing to topic {topic} for device {device.config.device_id}: {str(e)}")
            return False
            
    def unsubscribe(self, device: IoTDevice, topic: str) -> bool:
        """
        Unsubscribe from an MQTT topic.
        
        Args:
            device: Device to unsubscribe
            topic: Topic to unsubscribe from
            
        Returns:
            True if unsubscription was successful
        """
        try:
            client = self.clients.get(device.config.device_id)
            if not client:
                return True  # Already disconnected
                
            result = client.unsubscribe(topic)
            if result[0] == mqtt.MQTT_ERR_SUCCESS:
                self.topics[device.config.device_id].discard(topic)
                return True
            else:
                self.logger.error(f"Failed to unsubscribe from topic {topic} for device {device.config.device_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error unsubscribing from topic {topic} for device {device.config.device_id}: {str(e)}")
            return False
            
    def _on_connect(self, client, device: IoTDevice, flags, rc):
        """Handle MQTT connection event."""
        if rc == 0:
            self.logger.info(f"Connected to MQTT broker for device {device.config.device_id}")
            device.update_status(ConnectionStatus.CONNECTED)
            
            # Subscribe to device data topics
            base_topic = device.config.metadata.get("base_topic", f"devices/{device.config.device_id}")
            data_topic = f"{base_topic}/data/#"
            self.subscribe(device, data_topic)
            
        else:
            self.logger.error(f"Failed to connect to MQTT broker for device {device.config.device_id}, rc={rc}")
            device.update_status(ConnectionStatus.FAILED)
            
    def _on_disconnect(self, client, device: IoTDevice, rc):
        """Handle MQTT disconnection event."""
        self.logger.info(f"Disconnected from MQTT broker for device {device.config.device_id}, rc={rc}")
        
        if rc != 0:
            # Unexpected disconnect
            device.update_status(ConnectionStatus.RECONNECTING)
            device.increment_error()
        else:
            # Expected disconnect
            device.update_status(ConnectionStatus.DISCONNECTED)
            
    def _on_message(self, client, device: IoTDevice, msg):
        """Handle MQTT message event."""
        try:
            topic = msg.topic
            payload = msg.payload
            
            self.logger.debug(f"Received MQTT message on topic {topic} for device {device.config.device_id}")
            
            # Parse data based on format
            readings = self.parse_data(device, payload)
            
            # Add readings to device buffer
            for reading in readings:
                device.add_reading(reading)
                
            # Notify integrator of new data
            self.integrator.on_data_received(device, readings)
            
        except Exception as e:
            self.logger.error(f"Error handling MQTT message for device {device.config.device_id}: {str(e)}")


class BLEProtocolHandler(IoTProtocolHandler):
    """Handler for Bluetooth Low Energy protocol."""
    
    def __init__(self, integrator: 'IoTIntegrator'):
        """Initialize the BLE handler."""
        super().__init__(integrator)
        self.scanner = None
        self.connections = {}  # Maps device_id to BleakClient
        self.characteristics = {}  # Maps device_id to dict of characteristics
        self.scanning_thread = None
        self.scanning_active = False
        self.discovered_devices = {}  # Maps address to discovered device info
        
    def start_scanning(self):
        """Start scanning for BLE devices."""
        if self.scanning_thread and self.scanning_thread.is_alive():
            return  # Already scanning
            
        self.scanning_active = True
        self.scanning_thread = threading.Thread(target=self._scan_task)
        self.scanning_thread.daemon = True
        self.scanning_thread.start()
        
    def stop_scanning(self):
        """Stop scanning for BLE devices."""
        self.scanning_active = False
        if self.scanning_thread:
            self.scanning_thread.join(timeout=5.0)
            
    def _scan_task(self):
        """Background task for BLE scanning."""
        try:
            self.logger.info("Starting BLE scan")
            scanner = Scanner().withDelegate(BLEDelegate(self._handle_discovered_device))
            
            while self.scanning_active:
                try:
                    devices = scanner.scan(timeout=5.0)
                    self.logger.debug(f"Scanned {len(devices)} BLE devices")
                except BTLEException as e:
                    self.logger.error(f"Error during BLE scan: {str(e)}")
                    time.sleep(1.0)
                    
        except Exception as e:
            self.logger.error(f"BLE scanning thread error: {str(e)}")
            
    def _handle_discovered_device(self, device):
        """Handle a discovered BLE device."""
        try:
            addr = device.addr
            name = device.getValueText(9) or "Unknown"
            rssi = device.rssi
            
            # Update device information
            self.discovered_devices[addr] = {
                "address": addr,
                "name": name,
                "rssi": rssi,
                "last_seen": datetime.datetime.now(),
                "manufacturer_data": device.getValueText(255),
                "service_data": {}
            }
            
            # Extract service data
            for adtype, desc, value in device.getScanData():
                if adtype == 22:  # Service Data
                    service_uuid = desc.split()[0]
                    self.discovered_devices[addr]["service_data"][service_uuid] = value
                    
            # Notify integrator of discovered device
            self.integrator.on_device_discovered("ble", addr, name, self.discovered_devices[addr])
            
        except Exception as e:
            self.logger.error(f"Error handling discovered BLE device: {str(e)}")
            
    def connect(self, device: IoTDevice) -> bool:
        """Connect to a BLE device."""
        try:
            address = device.config.address
            
            device.update_status(ConnectionStatus.CONNECTING)
            
            # For async operation, we run this in a separate thread
            client = BleakClient(address)
            
            # Connect with timeout
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            connected = loop.run_until_complete(client.connect(timeout=device.config.connection_timeout_sec))
            
            if not connected:
                device.update_status(ConnectionStatus.FAILED)
                return False
                
            # Discover services and characteristics
            services = loop.run_until_complete(client.get_services())
            
            # Store characteristics
            characteristics = {}
            for service in services:
                for char in service.characteristics:
                    characteristics[str(char.uuid)] = char
                    
            # Store connection
            self.connections[device.config.device_id] = client
            self.characteristics[device.config.device_id] = characteristics
            device.connection_object = client
            device.update_status(ConnectionStatus.CONNECTED)
            
            # Subscribe to notifications if configured
            if "notify_characteristics" in device.config.metadata:
                for uuid in device.config.metadata["notify_characteristics"]:
                    if uuid in characteristics:
                        loop.run_until_complete(
                            client.start_notify(
                                uuid,
                                lambda sender, data: self._handle_notification(device, uuid, data)
                            )
                        )
                        
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to BLE device {device.config.device_id}: {str(e)}")
            device.update_status(ConnectionStatus.FAILED)
            return False
            
    def disconnect(self, device: IoTDevice) -> bool:
        """Disconnect from a BLE device."""
        try:
            client = self.connections.get(device.config.device_id)
            if not client:
                return True  # Already disconnected
                
            # For async operation, we run this in a separate thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(client.disconnect())
            
            device.update_status(ConnectionStatus.DISCONNECTED)
            del self.connections[device.config.device_id]
            self.characteristics.pop(device.config.device_id, None)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from BLE device {device.config.device_id}: {str(e)}")
            return False
            
    def read_data(self, device: IoTDevice) -> List[IoTReading]:
        """Read data from a BLE device."""
        try:
            client = self.connections.get(device.config.device_id)
            if not client or not client.is_connected:
                self.logger.error(f"Cannot read from disconnected device: {device.config.device_id}")
                return []
                
            # For async operation, we run this in a separate thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            readings = []
            
            # Read characteristics
            if "read_characteristics" in device.config.metadata:
                for uuid in device.config.metadata["read_characteristics"]:
                    try:
                        data = loop.run_until_complete(client.read_gatt_char(uuid))
                        
                        # Parse data
                        if "parsers" in device.config.metadata and uuid in device.config.metadata["parsers"]:
                            parser_name = device.config.metadata["parsers"][uuid]
                            parsed = self._parse_characteristic(device, uuid, data, parser_name)
                            readings.extend(parsed)
                        else:
                            # Default parsing
                            parsed = self.parse_data(device, data)
                            readings.extend(parsed)
                            
                    except Exception as e:
                        self.logger.error(f"Error reading characteristic {uuid} for device {device.config.device_id}: {str(e)}")
                        
            # Add readings to device buffer
            for reading in readings:
                device.add_reading(reading)
                
            return readings
            
        except Exception as e:
            self.logger.error(f"Error reading data from BLE device {device.config.device_id}: {str(e)}")
            return []
            
    def send_command(self, device: IoTDevice, command: str, params: Dict[str, Any] = None) -> bool:
        """Send a command to a BLE device."""
        try:
            client = self.connections.get(device.config.device_id)
            if not client or not client.is_connected:
                self.logger.error(f"Cannot send command to disconnected device: {device.config.device_id}")
                return False
                
            # For async operation, we run this in a separate thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Convert command and params to bytes
            if "command_characteristic" not in device.config.metadata:
                self.logger.error(f"No command characteristic specified for device {device.config.device_id}")
                return False
                
            uuid = device.config.metadata["command_characteristic"]
            
            # Format may vary by device
            if "command_format" in device.config.metadata:
                cmd_format = device.config.metadata["command_format"]
                if cmd_format == "json":
                    command_data = json.dumps({"command": command, "params": params or {}}).encode('utf-8')
                elif cmd_format == "binary":
                    # Binary format would depend on device
                    self.logger.warning(f"Binary command format not fully implemented for device {device.config.device_id}")
                    return False
                else:
                    self.logger.error(f"Unknown command format {cmd_format} for device {device.config.device_id}")
                    return False
            else:
                # Default format
                command_data = command.encode('utf-8')
                
            # Write command
            loop.run_until_complete(client.write_gatt_char(uuid, command_data))
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending command to BLE device {device.config.device_id}: {str(e)}")
            return False
            
    def configure(self, device: IoTDevice, config: Dict[str, Any]) -> bool:
        """Configure a BLE device."""
        # For BLE devices, configuration is typically done by writing to characteristics
        return self.send_command(device, "configure", config)
        
    def _handle_notification(self, device: IoTDevice, uuid: str, data: bytes):
        """Handle BLE notification."""
        try:
            self.logger.debug(f"Received BLE notification from {uuid} for device {device.config.device_id}")
            
            # Parse data
            if "parsers" in device.config.metadata and uuid in device.config.metadata["parsers"]:
                parser_name = device.config.metadata["parsers"][uuid]
                readings = self._parse_characteristic(device, uuid, data, parser_name)
            else:
                # Default parsing
                readings = self.parse_data(device, data)
                
            # Add readings to device buffer
            for reading in readings:
                device.add_reading(reading)
                
            # Notify integrator of new data
            self.integrator.on_data_received(device, readings)
            
        except Exception as e:
            self.logger.error(f"Error handling BLE notification for device {device.config.device_id}: {str(e)}")
            
    def _parse_characteristic(self, device: IoTDevice, uuid: str, data: bytes, parser_name: str) -> List[IoTReading]:
        """
        Parse BLE characteristic data using a named parser.
        
        This allows device-specific parsing logic for different characteristics.
        """
        try:
            if parser_name == "temperature_humidity":
                # Example parser for temperature and humidity sensor
                if len(data) < 4:
                    return []
                    
                temp = int.from_bytes(data[0:2], byteorder='little') / 100.0
                humidity = int.from_bytes(data[2:4], byteorder='little') / 100.0
                
                readings = [
                    IoTReading(
                        device_id=device.config.device_id,
                        timestamp=datetime.datetime.now(),
                        reading_type="temperature",
                        value=temp,
                        unit="°C",
                        sequence_number=None,
                        quality=None,
                        metadata={"characteristic": uuid}
                    ),
                    IoTReading(
                        device_id=device.config.device_id,
                        timestamp=datetime.datetime.now(),
                        reading_type="humidity",
                        value=humidity,
                        unit="%",
                        sequence_number=None,
                        quality=None,
                        metadata={"characteristic": uuid}
                    )
                ]
                return readings
                
            elif parser_name == "accelerometer":
                # Example parser for accelerometer
                if len(data) < 6:
                    return []
                    
                x = int.from_bytes(data[0:2], byteorder='little', signed=True) / 1000.0
                y = int.from_bytes(data[2:4], byteorder='little', signed=True) / 1000.0
                z = int.from_bytes(data[4:6], byteorder='little', signed=True) / 1000.0
                
                reading = IoTReading(
                    device_id=device.config.device_id,
                    timestamp=datetime.datetime.now(),
                    reading_type="accelerometer",
                    value={"x": x, "y": y, "z": z},
                    unit="g",
                    sequence_number=None,
                    quality=None,
                    metadata={"characteristic": uuid}
                )
                return [reading]
                
            elif parser_name == "battery":
                # Example parser for battery level
                if len(data) < 1:
                    return []
                    
                level = data[0]
                
                reading = IoTReading(
                    device_id=device.config.device_id,
                    timestamp=datetime.datetime.now(),
                    reading_type="battery",
                    value=level,
                    unit="%",
                    sequence_number=None,
                    quality=None,
                    metadata={"characteristic": uuid}
                )
                return [reading]
                
            elif parser_name == "beacon":
                # Example parser for beacon data
                if len(data) < 23:  # iBeacon format
                    return []
                    
                # Parse iBeacon format
                uuid_bytes = data[2:18]
                uuid_str = '-'.join([
                    uuid_bytes.hex()[0:8],
                    uuid_bytes.hex()[8:12],
                    uuid_bytes.hex()[12:16],
                    uuid_bytes.hex()[16:20],
                    uuid_bytes.hex()[20:32]
                ])
                
                major = int.from_bytes(data[18:20], byteorder='big')
                minor = int.from_bytes(data[20:22], byteorder='big')
                power = data[22]
                
                reading = IoTReading(
                    device_id=device.config.device_id,
                    timestamp=datetime.datetime.now(),
                    reading_type="beacon",
                    value={
                        "uuid": uuid_str,
                        "major": major,
                        "minor": minor,
                        "power": power
                    },
                    unit=None,
                    sequence_number=None,
                    quality=None,
                    metadata={"characteristic": uuid}
                )
                return [reading]
                
            else:
                self.logger.warning(f"Unknown parser {parser_name} for device {device.config.device_id}")
                return self.parse_data(device, data)
                
        except Exception as e:
            self.logger.error(f"Error parsing characteristic {uuid} for device {device.config.device_id}: {str(e)}")
            return []


# Add other protocol handlers as needed (CoAP, Zigbee, etc.)


class DataProcessor:
    """Processes data from IoT devices."""
    
    def __init__(self, integrator: 'IoTIntegrator'):
        """
        Initialize the data processor.
        
        Args:
            integrator: Parent IoT integrator
        """
        self.integrator = integrator
        self.logger = logging.getLogger(f"{__name__}.DataProcessor")
        self.validator = DataValidator()
        self.privacy_manager = None
        self.batch_queue = queue.Queue()
        self.processing_thread = None
        self.processing_active = False
        
    def start(self):
        """Start the data processor."""
        if self.processing_thread and self.processing_thread.is_alive():
            return  # Already running
            
        # Initialize components
        self.privacy_manager = PrivacyManager()
        
        # Set up validation rules
        self._setup_validation_rules()
        
        # Start processing thread
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self._processing_task)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("Data processor started")
        
    def stop(self):
        """Stop the data processor."""
        self.processing_active = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
            self.processing_thread = None
            
        self.logger.info("Data processor stopped")
        
    def process_readings(self, device: IoTDevice, readings: List[IoTReading]) -> List[IoTReading]:
        """
        Process readings from an IoT device.
        
        Args:
            device: Source device
            readings: Readings to process
            
        Returns:
            Processed readings
        """
        if not readings:
            return []
            
        processed = []
        
        for reading in readings:
            # Validate reading
            if not reading.validate():
                self.logger.warning(f"Invalid reading from device {device.config.device_id}: {reading}")
                continue
                
            # Apply data validation rules
            validation_result = self.validator.validate(reading)
            if not validation_result.is_valid:
                self.logger.warning(f"Reading failed validation for device {device.config.device_id}: {validation_result.reason}")
                continue
                
            # Apply privacy controls
            if self.privacy_manager:
                # Determine data category
                data_category = self._determine_data_category(reading)
                
                # Apply privacy transformations if needed
                if data_category != DataCategory.PUBLIC:
                    reading = self.privacy_manager.apply_privacy_controls(reading, data_category)
                    
            # Apply any additional processing
            reading = self._apply_custom_processing(device, reading)
            
            # Add to processed readings
            processed.append(reading)
            
        return processed
        
    def queue_batch_processing(self, device: IoTDevice, readings: List[IoTReading]):
        """
        Queue readings for batch processing.
        
        Args:
            device: Source device
            readings: Readings to process
        """
        self.batch_queue.put((device, readings))
        
    def _processing_task(self):
        """Background task for batch processing."""
        try:
            while self.processing_active:
                try:
                    # Get batch with timeout
                    device, readings = self.batch_queue.get(timeout=1.0)
                    
                    # Process readings
                    processed = self.process_readings(device, readings)
                    
                    # Forward to data handlers
                    self.integrator.on_data_processed(device, processed)
                    
                    # Mark task as done
                    self.batch_queue.task_done()
                    
                except queue.Empty:
                    # No data to process, just continue
                    pass
                except Exception as e:
                    self.logger.error(f"Error in batch processing: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Processing thread error: {str(e)}")
            
    def _setup_validation_rules(self):
        """Set up data validation rules."""
        # Add common validation rules
        self.validator.add_rule(
            ValidationRule(
                "timestamp_future",
                lambda reading: reading.timestamp <= (datetime.datetime.now() + datetime.timedelta(seconds=5)),
                "Reading timestamp is in the future"
            )
        )
        
        self.validator.add_rule(
            ValidationRule(
                "timestamp_too_old",
                lambda reading: reading.timestamp >= (datetime.datetime.now() - datetime.timedelta(days=1)),
                "Reading timestamp is too old"
            )
        )
        
        # Add type-specific validation rules
        self.validator.add_rule(
            ValidationRule(
                "temperature_range",
                lambda reading: reading.reading_type != "temperature" or 
                               (-50.0 <= float(reading.value) <= 100.0),
                "Temperature value out of range"
            )
        )
        
        self.validator.add_rule(
            ValidationRule(
                "humidity_range",
                lambda reading: reading.reading_type != "humidity" or 
                               (0.0 <= float(reading.value) <= 100.0),
                "Humidity value out of range"
            )
        )
        
        self.validator.add_rule(
            ValidationRule(
                "battery_range",
                lambda reading: reading.reading_type != "battery" or 
                               (0.0 <= float(reading.value) <= 100.0),
                "Battery value out of range"
            )
        )
        
    def _determine_data_category(self, reading: IoTReading) -> DataCategory:
        """
        Determine the data category for a reading.
        
        Args:
            reading: Reading to categorize
            
        Returns:
            Data category
        """
        # Determine based on reading type
        if reading.reading_type in {"temperature", "humidity", "pressure", "light", "noise"}:
            return DataCategory.PUBLIC
            
        elif reading.reading_type in {"presence", "motion", "occupancy", "count"}:
            return DataCategory.PSEUDONYMOUS
            
        elif reading.reading_type in {"mac_address", "device_id", "beacon_id"}:
            return DataCategory.PERSONAL
            
        elif reading.reading_type in {"face_id", "voice_id", "biometric"}:
            return DataCategory.SENSITIVE
            
        # Default to pseudonymous
        return DataCategory.PSEUDONYMOUS
        
    def _apply_custom_processing(self, device: IoTDevice, reading: IoTReading) -> IoTReading:
        """
        Apply custom processing to a reading.
        
        Args:
            device: Source device
            reading: Reading to process
            
        Returns:
            Processed reading
        """
        # Check for device-specific processing
        if "processors" in device.config.metadata:
            processor_name = device.config.metadata["processors"].get(reading.reading_type)
            if processor_name:
                return self._apply_named_processor(reading, processor_name)
                
        # Apply default processing based on reading type
        if reading.reading_type == "temperature" and reading.unit == "°F":
            # Convert Fahrenheit to Celsius
            reading.value = (float(reading.value) - 32) * 5 / 9
            reading.unit = "°C"
            
        elif reading.reading_type == "location" and isinstance(reading.value, dict):
            # Validate and normalize location data
            if "latitude" in reading.value and "longitude" in reading.value:
                lat = float(reading.value["latitude"])
                lon = float(reading.value["longitude"])
                
                # Validate coordinates
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    # Normalize precision
                    reading.value["latitude"] = round(lat, 6)
                    reading.value["longitude"] = round(lon, 6)
                    
        return reading
        
    def _apply_named_processor(self, reading: IoTReading, processor_name: str) -> IoTReading:
        """
        Apply a named processor to a reading.
        
        Args:
            reading: Reading to process
            processor_name: Name of processor to apply
            
        Returns:
            Processed reading
        """
        if processor_name == "motion_analysis":
            # Example processor for motion data
            if reading.reading_type == "motion" and isinstance(reading.value, bool):
                # Add timestamp to metadata for motion events
                if reading.value:  # If motion detected
                    reading.metadata["detection_time"] = reading.timestamp.isoformat()
                    
        elif processor_name == "presence_counter":
            # Example processor for presence counting
            if reading.reading_type == "presence" and isinstance(reading.value, int):
                # Add previous count if available
                if "previous_count" in reading.metadata:
                    reading.metadata["change"] = reading.value - reading.metadata["previous_count"]
                
                # Update previous count
                reading.metadata["previous_count"] = reading.value
                
        return reading


class IoTIntegrator:
    """
    Main class for integrating IoT devices with the UltraTrack system.
    
    This class manages the collection and processing of data from various IoT devices
    using different communication protocols.
    """
    
    def __init__(self, config=None):
        """
        Initialize the IoT integrator.
        
        Args:
            config: Optional configuration
        """
        self.logger = logging.getLogger(__name__)
        self.devices = {}  # Maps device_id to IoTDevice
        self.protocol_handlers = {}  # Maps protocol type to handler
        self.data_processor = DataProcessor(self)
        self.discovery_service = None
        self.encryption_manager = None
        self.polling_thread = None
        self.polling_active = False
        self.event_callbacks = defaultdict(list)
        self.config = config or {}
        
    def initialize(self):
        """Initialize the integrator and its components."""
        try:
            self.logger.info("Initializing IoT integrator")
            
            # Initialize protocol handlers
            self._init_protocol_handlers()
            
            # Initialize components
            self.data_processor.start()
            
            # Initialize discovery service if available
            try:
                self.discovery_service = ServiceDiscovery()
            except Exception as e:
                self.logger.warning(f"Failed to initialize service discovery: {str(e)}")
                
            # Initialize encryption manager if available
            try:
                self.encryption_manager = EncryptionManager()
            except Exception as e:
                self.logger.warning(f"Failed to initialize encryption manager: {str(e)}")
                
            # Start polling thread
            self.polling_active = True
            self.polling_thread = threading.Thread(target=self._polling_task)
            self.polling_thread.daemon = True
            self.polling_thread.start()
            
            self.logger.info("IoT integrator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize IoT integrator: {str(e)}")
            self.shutdown()
            return False
            
    def shutdown(self):
        """Shut down the integrator and release resources."""
        self.logger.info("Shutting down IoT integrator")
        
        # Stop polling
        self.polling_active = False
        if self.polling_thread:
            self.polling_thread.join(timeout=5.0)
            
        # Disconnect devices
        for device_id in list(self.devices.keys()):
            self.disconnect_device(device_id)
            
        # Stop data processor
        if self.data_processor:
            self.data_processor.stop()
            
        # Release resources
        if self.discovery_service:
            # Clean up discovery service
            pass
            
        # Clear data structures
        self.devices.clear()
        self.protocol_handlers.clear()
        self.event_callbacks.clear()
        
        self.logger.info("IoT integrator shutdown completed")
        
    def _init_protocol_handlers(self):
        """Initialize protocol handlers."""
        # Initialize handlers for supported protocols
        self.protocol_handlers[ConnectionProtocol.MQTT] = MQTTProtocolHandler(self)
        self.protocol_handlers[ConnectionProtocol.BLE] = BLEProtocolHandler(self)
        
        # Start BLE scanning if configured
        if self.config.get("enable_ble_scanning", False):
            ble_handler = self.protocol_handlers.get(ConnectionProtocol.BLE)
            if ble_handler:
                ble_handler.start_scanning()
                
        # Initialize other handlers as needed
        # self.protocol_handlers[ConnectionProtocol.COAP] = CoAPProtocolHandler(self)
        # self.protocol_handlers[ConnectionProtocol.ZIGBEE] = ZigbeeProtocolHandler(self)
        # etc.
        
    def add_device(self, config: IoTDeviceConfig) -> bool:
        """
        Add a new IoT device.
        
        Args:
            config: Device configuration
            
        Returns:
            True if device was added successfully
        """
        try:
            # Validate configuration
            if not config.validate():
                self.logger.error(f"Invalid device configuration for device {config.device_id}")
                return False
                
            # Check if device already exists
            if config.device_id in self.devices:
                self.logger.warning(f"Device {config.device_id} already exists, updating configuration")
                # Update existing device configuration
                self.devices[config.device_id].config = config
                return True
                
            # Create new device
            device = IoTDevice(config=config)
            self.devices[config.device_id] = device
            
            self.logger.info(f"Added device {config.device_id} of type {config.device_type}")
            
            # Try to connect if device is not polling-based
            if config.polling_interval_sec is None:
                self.connect_device(config.device_id)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding device {config.device_id}: {str(e)}")
            return False
            
    def remove_device(self, device_id: str) -> bool:
        """
        Remove an IoT device.
        
        Args:
            device_id: ID of device to remove
            
        Returns:
            True if device was removed successfully
        """
        try:
            # Check if device exists
            if device_id not in self.devices:
                self.logger.warning(f"Device {device_id} does not exist")
                return False
                
            # Disconnect device if connected
            self.disconnect_device(device_id)
            
            # Remove device from map
            del self.devices[device_id]
            
            self.logger.info(f"Removed device {device_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing device {device_id}: {str(e)}")
            return False
            
    def connect_device(self, device_id: str) -> bool:
        """
        Connect to an IoT device.
        
        Args:
            device_id: ID of device to connect to
            
        Returns:
            True if connection was successful
        """
        try:
            # Check if device exists
            if device_id not in self.devices:
                self.logger.error(f"Device {device_id} does not exist")
                return False
                
            device = self.devices[device_id]
            
            # Check if already connected
            if device.status == ConnectionStatus.CONNECTED:
                return True
                
            # Get protocol handler
            handler = self.protocol_handlers.get(device.config.protocol)
            if not handler:
                self.logger.error(f"Unsupported protocol {device.config.protocol} for device {device_id}")
                return False
                
            # Connect to device
            success = handler.connect(device)
            if success:
                self.logger.info(f"Connected to device {device_id}")
                self._trigger_event("device_connected", device)
            else:
                self.logger.error(f"Failed to connect to device {device_id}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error connecting to device {device_id}: {str(e)}")
            return False
            
    def disconnect_device(self, device_id: str) -> bool:
        """
        Disconnect from an IoT device.
        
        Args:
            device_id: ID of device to disconnect from
            
        Returns:
            True if disconnection was successful
        """
        try:
            # Check if device exists
            if device_id not in self.devices:
                self.logger.error(f"Device {device_id} does not exist")
                return False
                
            device = self.devices[device_id]
            
            # Check if already disconnected
            if device.status == ConnectionStatus.DISCONNECTED:
                return True
                
            # Get protocol handler
            handler = self.protocol_handlers.get(device.config.protocol)
            if not handler:
                self.logger.error(f"Unsupported protocol {device.config.protocol} for device {device_id}")
                return False
                
            # Disconnect from device
            success = handler.disconnect(device)
            if success:
                self.logger.info(f"Disconnected from device {device_id}")
                self._trigger_event("device_disconnected", device)
            else:
                self.logger.error(f"Failed to disconnect from device {device_id}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from device {device_id}: {str(e)}")
            return False
            
    def read_device_data(self, device_id: str) -> List[IoTReading]:
        """
        Read data from an IoT device.
        
        Args:
            device_id: ID of device to read from
            
        Returns:
            List of readings
        """
        try:
            # Check if device exists
            if device_id not in self.devices:
                self.logger.error(f"Device {device_id} does not exist")
                return []
                
            device = self.devices[device_id]
            
            # Check if connected
            if device.status != ConnectionStatus.CONNECTED:
                self.logger.error(f"Device {device_id} is not connected")
                return []
                
            # Get protocol handler
            handler = self.protocol_handlers.get(device.config.protocol)
            if not handler:
                self.logger.error(f"Unsupported protocol {device.config.protocol} for device {device_id}")
                return []
                
            # Read data
            readings = handler.read_data(device)
            if readings:
                self.logger.debug(f"Read {len(readings)} readings from device {device_id}")
                
                # Process readings
                processed = self.data_processor.process_readings(device, readings)
                
                # Trigger event
                self._trigger_event("data_received", device, processed)
                
            return readings
            
        except Exception as e:
            self.logger.error(f"Error reading data from device {device_id}: {str(e)}")
            return []
            
    def send_device_command(self, device_id: str, command: str, params: Dict[str, Any] = None) -> bool:
        """
        Send a command to an IoT device.
        
        Args:
            device_id: ID of device to send command to
            command: Command name
            params: Command parameters
            
        Returns:
            True if command was sent successfully
        """
        try:
            # Check if device exists
            if device_id not in self.devices:
                self.logger.error(f"Device {device_id} does not exist")
                return False
                
            device = self.devices[device_id]
            
            # Check if connected
            if device.status != ConnectionStatus.CONNECTED:
                self.logger.error(f"Device {device_id} is not connected")
                return False
                
            # Get protocol handler
            handler = self.protocol_handlers.get(device.config.protocol)
            if not handler:
                self.logger.error(f"Unsupported protocol {device.config.protocol} for device {device_id}")
                return False
                
            # Send command
            success = handler.send_command(device, command, params)
            if success:
                self.logger.info(f"Sent command {command} to device {device_id}")
                self._trigger_event("command_sent", device, command, params)
            else:
                self.logger.error(f"Failed to send command {command} to device {device_id}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending command {command} to device {device_id}: {str(e)}")
            return False
            
    def get_device_status(self, device_id: str) -> Optional[ConnectionStatus]:
        """
        Get the status of an IoT device.
        
        Args:
            device_id: ID of device to get status for
            
        Returns:
            Device status or None if device does not exist
        """
        # Check if device exists
        if device_id not in self.devices:
            self.logger.error(f"Device {device_id} does not exist")
            return None
            
        return self.devices[device_id].status
        
    def get_device_readings(self, device_id: str, max_count: int = 100) -> List[IoTReading]:
        """
        Get readings from an IoT device buffer.
        
        Args:
            device_id: ID of device to get readings from
            max_count: Maximum number of readings to retrieve
            
        Returns:
            List of readings
        """
        # Check if device exists
        if device_id not in self.devices:
            self.logger.error(f"Device {device_id} does not exist")
            return []
            
        return self.devices[device_id].get_readings(max_count)
        
    def discover_devices(self, protocol_type: ConnectionProtocol = None) -> Dict[str, Any]:
        """
        Discover IoT devices.
        
        Args:
            protocol_type: Optional protocol to restrict discovery to
            
        Returns:
            Dictionary of discovered devices
        """
        discovered = {}
        
        # Check if specific protocol requested
        if protocol_type:
            handler = self.protocol_handlers.get(protocol_type)
            if handler and hasattr(handler, 'discovered_devices'):
                return handler.discovered_devices.copy()
            else:
                return {}
                
        # Combine discovered devices from all handlers
        for protocol, handler in self.protocol_handlers.items():
            if hasattr(handler, 'discovered_devices'):
                for addr, device_info in handler.discovered_devices.items():
                    discovered[f"{protocol.name.lower()}:{addr}"] = {
                        "protocol": protocol,
                        "info": device_info
                    }
                    
        return discovered
        
    def register_event_callback(self, event_type: str, callback: Callable):
        """
        Register a callback for an event.
        
        Args:
            event_type: Type of event to register for
            callback: Callback function
        """
        self.event_callbacks[event_type].append(callback)
        
    def unregister_event_callback(self, event_type: str, callback: Callable) -> bool:
        """
        Unregister a callback for an event.
        
        Args:
            event_type: Type of event to unregister for
            callback: Callback function
            
        Returns:
            True if callback was unregistered
        """
        if event_type in self.event_callbacks and callback in self.event_callbacks[event_type]:
            self.event_callbacks[event_type].remove(callback)
            return True
        return False
        
    def on_data_received(self, device: IoTDevice, readings: List[IoTReading]):
        """
        Handle data received from an IoT device.
        
        Args:
            device: Source device
            readings: Received readings
        """
        # Queue for processing
        self.data_processor.queue_batch_processing(device, readings)
        
    def on_data_processed(self, device: IoTDevice, readings: List[IoTReading]):
        """
        Handle processed data from an IoT device.
        
        Args:
            device: Source device
            readings: Processed readings
        """
        # Trigger event
        self._trigger_event("data_processed", device, readings)
        
    def on_device_discovered(self, protocol_type: str, address: str, name: str, info: Dict[str, Any]):
        """
        Handle discovered IoT device.
        
        Args:
            protocol_type: Protocol type
            address: Device address
            name: Device name
            info: Additional device information
        """
        # Trigger event
        self._trigger_event("device_discovered", protocol_type, address, name, info)
        
    def _polling_task(self):
        """Background task for polling devices."""
        try:
            while self.polling_active:
                start_time = time.time()
                
                # Get devices that need polling
                polling_devices = {
                    device_id: device for device_id, device in self.devices.items()
                    if device.config.polling_interval_sec is not None
                }
                
                # Check if each device needs polling
                for device_id, device in polling_devices.items():
                    try:
                        # Check if it's time to poll
                        if device.last_reading is None:
                            # Never polled, do it now
                            self._poll_device(device)
                        else:
                            # Check if polling interval has elapsed
                            elapsed = (datetime.datetime.now() - device.last_reading).total_seconds()
                            if elapsed >= device.config.polling_interval_sec:
                                self._poll_device(device)
                    except Exception as e:
                        self.logger.error(f"Error polling device {device_id}: {str(e)}")
                        
                # Sleep for a bit (100ms)
                time.sleep(0.1)
                
                # Limit polling rate
                elapsed = time.time() - start_time
                if elapsed < 1.0:
                    time.sleep(1.0 - elapsed)
                    
        except Exception as e:
            self.logger.error(f"Polling thread error: {str(e)}")
            
    def _poll_device(self, device: IoTDevice):
        """
        Poll a device for data.
        
        Args:
            device: Device to poll
        """
        # Check if connected
        if device.status != ConnectionStatus.CONNECTED:
            # Try to connect
            if not self.connect_device(device.config.device_id):
                return
                
        # Read data
        self.read_device_data(device.config.device_id)
        
    def _trigger_event(self, event_type: str, *args):
        """
        Trigger an event.
        
        Args:
            event_type: Type of event to trigger
            *args: Event arguments
        """
        for callback in self.event_callbacks.get(event_type, []):
            try:
                callback(*args)
            except Exception as e:
                self.logger.error(f"Error in event callback for {event_type}: {str(e)}")
    
    # Additional utility methods
    
    def get_active_devices(self) -> List[str]:
        """
        Get a list of active device IDs.
        
        Returns:
            List of active device IDs
        """
        return [
            device_id for device_id, device in self.devices.items()
            if device.is_active()
        ]
        
    def get_devices_by_type(self, device_type: IoTDeviceType) -> List[str]:
        """
        Get a list of device IDs by type.
        
        Args:
            device_type: Device type to filter by
            
        Returns:
            List of matching device IDs
        """
        return [
            device_id for device_id, device in self.devices.items()
            if device.config.device_type == device_type
        ]
        
    def get_device_info(self, device_id: str) -> Dict[str, Any]:
        """
        Get information about an IoT device.
        
        Args:
            device_id: ID of device to get information for
            
        Returns:
            Dictionary of device information or empty dict if device does not exist
        """
        # Check if device exists
        if device_id not in self.devices:
            return {}
            
        device = self.devices[device_id]
        
        # Build info dictionary
        info = {
            "device_id": device.config.device_id,
            "device_type": device.config.device_type.name,
            "protocol": device.config.protocol.name,
            "address": device.config.address,
            "status": device.status.name,
            "last_connected": device.last_connected.isoformat() if device.last_connected else None,
            "last_reading": device.last_reading.isoformat() if device.last_reading else None,
            "error_count": device.error_count,
            "readings_buffered": device.reading_count(),
            "is_active": device.is_active(),
            "metadata": device.config.metadata
        }
        
        return info
        
    def configure_device(self, device_id: str, config: Dict[str, Any]) -> bool:
        """
        Configure an IoT device.
        
        Args:
            device_id: ID of device to configure
            config: Configuration parameters
            
        Returns:
            True if configuration was successful
        """
        try:
            # Check if device exists
            if device_id not in self.devices:
                self.logger.error(f"Device {device_id} does not exist")
                return False
                
            device = self.devices[device_id]
            
            # Check if connected
            if device.status != ConnectionStatus.CONNECTED:
                self.logger.error(f"Device {device_id} is not connected")
                return False
                
            # Get protocol handler
            handler = self.protocol_handlers.get(device.config.protocol)
            if not handler:
                self.logger.error(f"Unsupported protocol {device.config.protocol} for device {device_id}")
                return False
                
            # Configure device
            success = handler.configure(device, config)
            if success:
                self.logger.info(f"Configured device {device_id}")
                self._trigger_event("device_configured", device, config)
            else:
                self.logger.error(f"Failed to configure device {device_id}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error configuring device {device_id}: {str(e)}")
            return False
            
    def batch_add_devices(self, configs: List[IoTDeviceConfig]) -> Dict[str, bool]:
        """
        Add multiple IoT devices.
        
        Args:
            configs: List of device configurations
            
        Returns:
            Dictionary mapping device IDs to success/failure
        """
        results = {}
        
        for config in configs:
            results[config.device_id] = self.add_device(config)
            
        return results
