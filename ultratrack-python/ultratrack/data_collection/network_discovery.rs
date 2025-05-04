//! Network Discovery Module
//!
//! This module provides capabilities for discovering and integrating various data sources
//! on the network:
//! - Automatic camera discovery via multiple protocols (ONVIF, RTSP, etc.)
//! - Network scanning for IoT and RF devices
//! - Service discovery for integration with other tracking systems
//! - Continuous monitoring for new devices
//! - Device capability detection
//!
//! The discovery system maintains a registry of available sources and provides
//! events when new sources are discovered or existing ones change.

use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::{future::join_all, stream::FuturesUnordered, StreamExt};
use log::{debug, error, info, trace, warn};
use serde::{Deserialize, Serialize};
use tokio::net::UdpSocket;
use tokio::sync::{broadcast, mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio::time::{self, sleep};
use uuid::Uuid;

use crate::common::errors::{Error, Result};
use crate::common::models::{Capability, Coordinates, DeviceCredentials, NetworkProtocol};
use crate::config::NetworkDiscoveryConfig;
use crate::data_collection::camera_interface::{CameraInfo, CameraType, StreamProtocol};
use crate::security::encryption::EncryptionManager;
use crate::storage::persistent::PersistentStorage;

/// Network discovery interval in seconds
const DEFAULT_DISCOVERY_INTERVAL: u64 = 300; // 5 minutes
/// Default scan timeout in seconds
const DEFAULT_SCAN_TIMEOUT: u64 = 10;
/// Default number of discovery retries
const DEFAULT_DISCOVERY_RETRIES: u8 = 3;
/// Default TTL for discovered devices in seconds
const DEFAULT_DEVICE_TTL: u64 = 86400; // 24 hours
/// ONVIF discovery multicast address
const ONVIF_MULTICAST_ADDR: &str = "239.255.255.250:3702";
/// SSDP discovery multicast address for IP cameras
const SSDP_MULTICAST_ADDR: &str = "239.255.255.250:1900";
/// mDNS discovery multicast address
const MDNS_MULTICAST_ADDR: &str = "224.0.0.251:5353";

/// Events published by the NetworkDiscovery system
#[derive(Clone, Debug)]
pub enum DiscoveryEvent {
    /// A new device has been discovered
    DeviceDiscovered(DiscoveredDevice),
    /// An existing device has been updated
    DeviceUpdated(DiscoveredDevice),
    /// A device is no longer available
    DeviceRemoved(DeviceId),
    /// A network scan has been completed
    ScanCompleted(NetworkScan),
    /// A network scan has failed
    ScanFailed(Error),
}

/// Unique identifier for a discovered device
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId(Uuid);

impl DeviceId {
    /// Create a new random device ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    /// Create a device ID from a unique identifier
    pub fn from_unique_id(id: &str) -> Self {
        let bytes = md5::compute(id.as_bytes());
        Self(Uuid::from_bytes(bytes.0))
    }
    
    /// Get the ID as a string
    pub fn as_str(&self) -> String {
        self.0.to_string()
    }
}

impl Default for DeviceId {
    fn default() -> Self {
        Self::new()
    }
}

/// Types of discoverable devices
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    /// Camera device
    Camera(CameraType),
    /// IoT sensor device
    IoTSensor(String),
    /// RF signal source
    RFSource(String),
    /// Access control system
    AccessControl(String),
    /// Audio capture device
    AudioCapture(String),
    /// Thermal imaging device
    ThermalImaging(String),
    /// Network Video Recorder system
    NVR(String),
    /// License Plate Recognition system
    LPR(String),
    /// Other device type
    Other(String),
}

/// Information about a discovered device
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiscoveredDevice {
    /// Unique device identifier
    pub id: DeviceId,
    /// Device type
    pub device_type: DeviceType,
    /// Device manufacturer
    pub manufacturer: String,
    /// Device model
    pub model: String,
    /// Device name or description
    pub name: String,
    /// IP address of the device
    pub ip_address: IpAddr,
    /// Port number
    pub port: u16,
    /// MAC address if available
    pub mac_address: Option<String>,
    /// Serial number if available
    pub serial_number: Option<String>,
    /// Firmware version
    pub firmware_version: Option<String>,
    /// Physical location if known
    pub location: Option<Coordinates>,
    /// Supported network protocols
    pub protocols: Vec<NetworkProtocol>,
    /// Device capabilities
    pub capabilities: Vec<Capability>,
    /// Device online status
    pub online: bool,
    /// First discovery time
    pub first_discovered: DateTime<Utc>,
    /// Last seen time
    pub last_seen: DateTime<Utc>,
    /// Metadata associated with the device
    pub metadata: HashMap<String, String>,
}

impl DiscoveredDevice {
    /// Create a new discovered device
    pub fn new(
        device_type: DeviceType,
        manufacturer: String,
        model: String,
        name: String,
        ip_address: IpAddr,
        port: u16,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: DeviceId::new(),
            device_type,
            manufacturer,
            model,
            name,
            ip_address,
            port,
            mac_address: None,
            serial_number: None,
            firmware_version: None,
            location: None,
            protocols: Vec::new(),
            capabilities: Vec::new(),
            online: true,
            first_discovered: now,
            last_seen: now,
            metadata: HashMap::new(),
        }
    }
    
    /// Convert to CameraInfo if applicable
    pub fn to_camera_info(&self) -> Option<CameraInfo> {
        match &self.device_type {
            DeviceType::Camera(camera_type) => {
                let streams = self.protocols.iter()
                    .filter_map(|p| {
                        if let NetworkProtocol::Stream(stream_protocol) = p {
                            match stream_protocol {
                                StreamProtocol::RTSP => {
                                    Some((StreamProtocol::RTSP, format!("rtsp://{}:{}/stream", self.ip_address, self.port)))
                                },
                                StreamProtocol::HTTP => {
                                    Some((StreamProtocol::HTTP, format!("http://{}:{}/video", self.ip_address, self.port)))
                                },
                                StreamProtocol::HTTPS => {
                                    Some((StreamProtocol::HTTPS, format!("https://{}:{}/video", self.ip_address, self.port)))
                                },
                                StreamProtocol::RTMP => {
                                    Some((StreamProtocol::RTMP, format!("rtmp://{}:{}/stream", self.ip_address, self.port)))
                                },
                                StreamProtocol::HLS => {
                                    Some((StreamProtocol::HLS, format!("http://{}:{}/hls/stream.m3u8", self.ip_address, self.port)))
                                },
                                _ => None
                            }
                        } else {
                            None
                        }
                    })
                    .collect::<HashMap<_, _>>();
                
                Some(CameraInfo {
                    id: self.id.clone(),
                    name: self.name.clone(),
                    camera_type: camera_type.clone(),
                    manufacturer: self.manufacturer.clone(),
                    model: self.model.clone(),
                    ip_address: self.ip_address,
                    mac_address: self.mac_address.clone(),
                    location: self.location.clone(),
                    streams,
                    capabilities: self.capabilities.clone(),
                    online: self.online,
                    metadata: self.metadata.clone(),
                })
            },
            _ => None
        }
    }
    
    /// Update from a newer device record
    pub fn update_from(&mut self, other: &DiscoveredDevice) {
        // Don't update the ID, first_discovered time, or device_type
        self.manufacturer = other.manufacturer.clone();
        self.model = other.model.clone();
        self.name = other.name.clone();
        self.ip_address = other.ip_address;
        self.port = other.port;
        
        if other.mac_address.is_some() {
            self.mac_address = other.mac_address.clone();
        }
        
        if other.serial_number.is_some() {
            self.serial_number = other.serial_number.clone();
        }
        
        if other.firmware_version.is_some() {
            self.firmware_version = other.firmware_version.clone();
        }
        
        if other.location.is_some() {
            self.location = other.location.clone();
        }
        
        if !other.protocols.is_empty() {
            self.protocols = other.protocols.clone();
        }
        
        if !other.capabilities.is_empty() {
            self.capabilities = other.capabilities.clone();
        }
        
        self.online = other.online;
        self.last_seen = other.last_seen;
        
        // Merge metadata
        for (key, value) in &other.metadata {
            self.metadata.insert(key.clone(), value.clone());
        }
    }
}

/// Network scan information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkScan {
    /// Unique scan identifier
    pub id: Uuid,
    /// Scan start time
    pub start_time: DateTime<Utc>,
    /// Scan end time
    pub end_time: Option<DateTime<Utc>>,
    /// Number of devices discovered
    pub devices_discovered: usize,
    /// Number of new devices discovered
    pub new_devices: usize,
    /// Number of updated devices
    pub updated_devices: usize,
    /// IP address ranges scanned
    pub scanned_ranges: Vec<String>,
    /// Protocols used for scanning
    pub protocols: Vec<NetworkProtocol>,
    /// Scan status
    pub status: ScanStatus,
    /// Error message if scan failed
    pub error_message: Option<String>,
}

/// Network scan status
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScanStatus {
    /// Scan is in progress
    InProgress,
    /// Scan completed successfully
    Completed,
    /// Scan failed
    Failed,
    /// Scan was cancelled
    Cancelled,
}

/// Network discovery method trait
#[async_trait]
pub trait DiscoveryMethod: Send + Sync {
    /// Get the name of the discovery method
    fn name(&self) -> &'static str;
    
    /// Get the protocols supported by this discovery method
    fn supported_protocols(&self) -> Vec<NetworkProtocol>;
    
    /// Discover devices using this method
    async fn discover(&self, config: &ScanConfig) -> Result<Vec<DiscoveredDevice>>;
    
    /// Check if this method can handle the given protocol
    fn supports_protocol(&self, protocol: &NetworkProtocol) -> bool {
        self.supported_protocols().contains(protocol)
    }
}

/// Configuration for network scanning
#[derive(Clone, Debug)]
pub struct ScanConfig {
    /// IP address ranges to scan
    pub ip_ranges: Vec<String>,
    /// Protocols to use for scanning
    pub protocols: Vec<NetworkProtocol>,
    /// Network scan timeout in seconds
    pub timeout: u64,
    /// Number of retries for discovery attempts
    pub retries: u8,
    /// Additional discovery parameters
    pub parameters: HashMap<String, String>,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            ip_ranges: vec!["192.168.1.0/24".to_string()],
            protocols: vec![
                NetworkProtocol::Stream(StreamProtocol::RTSP),
                NetworkProtocol::Discovery(String::from("ONVIF")),
                NetworkProtocol::Discovery(String::from("SSDP")),
            ],
            timeout: DEFAULT_SCAN_TIMEOUT,
            retries: DEFAULT_DISCOVERY_RETRIES,
            parameters: HashMap::new(),
        }
    }
}

/// ONVIF discovery method implementation
pub struct OnvifDiscovery {
    client: Option<Arc<onvif::Client>>,
}

impl OnvifDiscovery {
    /// Create a new ONVIF discovery instance
    pub fn new() -> Self {
        Self {
            client: None,
        }
    }
    
    /// Initialize the ONVIF client if needed
    async fn initialize_client(&mut self) -> Result<Arc<onvif::Client>> {
        if self.client.is_none() {
            let client = Arc::new(onvif::Client::new().map_err(|e| Error::ExternalServiceError(format!("Failed to create ONVIF client: {}", e)))?);
            self.client = Some(client.clone());
            Ok(client)
        } else {
            Ok(self.client.as_ref().unwrap().clone())
        }
    }
}

#[async_trait]
impl DiscoveryMethod for OnvifDiscovery {
    fn name(&self) -> &'static str {
        "ONVIF"
    }
    
    fn supported_protocols(&self) -> Vec<NetworkProtocol> {
        vec![
            NetworkProtocol::Discovery(String::from("ONVIF")),
            NetworkProtocol::Stream(StreamProtocol::RTSP),
        ]
    }
    
    async fn discover(&self, config: &ScanConfig) -> Result<Vec<DiscoveredDevice>> {
        // Initialize client if needed
        let client = match self.initialize_client().await {
            Ok(client) => client,
            Err(e) => {
                error!("Failed to initialize ONVIF client: {}", e);
                return Err(e);
            }
        };
        
        // Create a socket for discovery
        let socket = UdpSocket::bind("0.0.0.0:0").await
            .map_err(|e| Error::NetworkError(format!("Failed to bind UDP socket: {}", e)))?;
        
        // Enable broadcast
        socket.set_broadcast(true)
            .map_err(|e| Error::NetworkError(format!("Failed to set broadcast: {}", e)))?;
        
        // Send ONVIF discovery message
        let discovery_msg = onvif::WS_DISCOVERY_PROBE_MESSAGE;
        let multicast_addr = ONVIF_MULTICAST_ADDR.parse::<SocketAddr>()
            .map_err(|e| Error::NetworkError(format!("Failed to parse multicast address: {}", e)))?;
        
        socket.send_to(discovery_msg.as_bytes(), multicast_addr).await
            .map_err(|e| Error::NetworkError(format!("Failed to send discovery message: {}", e)))?;
        
        // Collect responses
        let mut buffer = vec![0u8; 8192];
        let mut devices = Vec::new();
        let timeout = Duration::from_secs(config.timeout);
        let start_time = Instant::now();
        
        while start_time.elapsed() < timeout {
            // Set socket timeout
            let remaining_time = timeout.checked_sub(start_time.elapsed()).unwrap_or(Duration::from_millis(100));
            if remaining_time.is_zero() {
                break;
            }
            
            // Try to receive a response with timeout
            let result = tokio::time::timeout(
                remaining_time,
                socket.recv_from(&mut buffer)
            ).await;
            
            match result {
                Ok(Ok((size, addr))) => {
                    trace!("Received {} bytes from {}", size, addr);
                    let response = String::from_utf8_lossy(&buffer[..size]);
                    
                    // Extract device information from response
                    if let Some(device) = self.parse_onvif_response(&response, addr.ip()) {
                        trace!("Discovered ONVIF device: {:?}", device);
                        devices.push(device);
                    }
                },
                Ok(Err(e)) => {
                    warn!("Error receiving ONVIF discovery response: {}", e);
                    continue;
                },
                Err(_) => {
                    // Timeout reached
                    break;
                }
            }
        }
        
        // If any devices were found, try to get more details
        if !devices.is_empty() {
            let mut enhanced_devices = Vec::with_capacity(devices.len());
            
            for device in devices {
                match self.enhance_device_info(client.clone(), device).await {
                    Ok(enhanced_device) => {
                        enhanced_devices.push(enhanced_device);
                    },
                    Err(e) => {
                        warn!("Failed to enhance device info: {}", e);
                        enhanced_devices.push(device);
                    }
                }
            }
            
            Ok(enhanced_devices)
        } else {
            Ok(devices)
        }
    }
}

impl OnvifDiscovery {
    /// Parse ONVIF discovery response
    fn parse_onvif_response(&self, response: &str, ip: IpAddr) -> Option<DiscoveredDevice> {
        // Extract ONVIF device information from XML response
        // Real implementation would use proper XML parsing
        
        // Extract XAddrs (service addresses)
        let xaddrs_start = response.find("<d:XAddrs>")?;
        let xaddrs_end = response.find("</d:XAddrs>")?;
        let xaddrs = &response[xaddrs_start + 10..xaddrs_end];
        
        // Extract device endpoint from XAddrs
        let endpoint = xaddrs.split_whitespace().next()?;
        
        // Parse endpoint to get port
        let uri = url::Url::parse(endpoint).ok()?;
        let port = uri.port().unwrap_or(80);
        
        // Extract device UUID
        let uuid_start = response.find("urn:uuid:")?;
        let uuid_end = uuid_start + 9 + 36; // "urn:uuid:" + UUID length
        let uuid_str = &response[uuid_start..uuid_end];
        
        // Create a device ID from the UUID
        let device_id = DeviceId::from_unique_id(uuid_str);
        
        // Get current timestamp
        let now = Utc::now();
        
        // Create a basic discovered device
        Some(DiscoveredDevice {
            id: device_id,
            device_type: DeviceType::Camera(CameraType::IPCamera),
            manufacturer: "Unknown".to_string(), // Will be enhanced later
            model: "Unknown".to_string(),        // Will be enhanced later
            name: format!("ONVIF Camera ({})", ip),
            ip_address: ip,
            port,
            mac_address: None,                   // Will be enhanced later
            serial_number: None,                 // Will be enhanced later
            firmware_version: None,              // Will be enhanced later
            location: None,
            protocols: vec![
                NetworkProtocol::Discovery(String::from("ONVIF")),
                NetworkProtocol::Stream(StreamProtocol::RTSP),
            ],
            capabilities: Vec::new(),            // Will be enhanced later
            online: true,
            first_discovered: now,
            last_seen: now,
            metadata: HashMap::from([
                ("endpoint".to_string(), endpoint.to_string()),
                ("uuid".to_string(), uuid_str.to_string()),
            ]),
        })
    }
    
    /// Enhance device information with more details
    async fn enhance_device_info(&self, client: Arc<onvif::Client>, mut device: DiscoveredDevice) -> Result<DiscoveredDevice> {
        // Get device endpoint from metadata
        let endpoint = match device.metadata.get("endpoint") {
            Some(endpoint) => endpoint,
            None => return Ok(device),
        };
        
        // Connect to the device
        let device_info = client.connect(endpoint).await
            .map_err(|e| Error::ExternalServiceError(format!("Failed to connect to ONVIF device: {}", e)))?;
        
        // Update device information
        device.manufacturer = device_info.manufacturer.clone();
        device.model = device_info.model.clone();
        device.name = device_info.name.clone();
        device.firmware_version = Some(device_info.firmware_version.clone());
        device.serial_number = Some(device_info.serial_number.clone());
        
        // Get device capabilities
        if let Ok(capabilities) = client.get_capabilities().await {
            // Convert ONVIF capabilities to our internal representation
            let mut device_capabilities = Vec::new();
            
            if capabilities.ptz {
                device_capabilities.push(Capability::PanTiltZoom);
            }
            
            if capabilities.events {
                device_capabilities.push(Capability::EventStreaming);
            }
            
            if capabilities.imaging {
                device_capabilities.push(Capability::ImageSettings);
            }
            
            if capabilities.media {
                device_capabilities.push(Capability::MultipleStreams);
            }
            
            if capabilities.analytics {
                device_capabilities.push(Capability::VideoAnalytics);
            }
            
            device.capabilities = device_capabilities;
        }
        
        // Get device stream URIs
        if let Ok(profiles) = client.get_profiles().await {
            for profile in profiles {
                if let Ok(uri) = client.get_stream_uri(&profile.token).await {
                    device.metadata.insert(format!("stream_uri_{}", profile.token), uri.clone());
                    
                    // Add RTSP as supported protocol if not already present
                    if !device.protocols.contains(&NetworkProtocol::Stream(StreamProtocol::RTSP)) {
                        device.protocols.push(NetworkProtocol::Stream(StreamProtocol::RTSP));
                    }
                }
            }
        }
        
        Ok(device)
    }
}

/// SSDP discovery method implementation
pub struct SSDPDiscovery;

impl SSDPDiscovery {
    /// Create a new SSDP discovery instance
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DiscoveryMethod for SSDPDiscovery {
    fn name(&self) -> &'static str {
        "SSDP"
    }
    
    fn supported_protocols(&self) -> Vec<NetworkProtocol> {
        vec![
            NetworkProtocol::Discovery(String::from("SSDP")),
            NetworkProtocol::Stream(StreamProtocol::HTTP),
            NetworkProtocol::Stream(StreamProtocol::HTTPS),
            NetworkProtocol::Stream(StreamProtocol::RTSP),
            NetworkProtocol::Stream(StreamProtocol::HLS),
        ]
    }
    
    async fn discover(&self, config: &ScanConfig) -> Result<Vec<DiscoveredDevice>> {
        // Create a socket for discovery
        let socket = UdpSocket::bind("0.0.0.0:0").await
            .map_err(|e| Error::NetworkError(format!("Failed to bind UDP socket: {}", e)))?;
        
        // Enable broadcast
        socket.set_broadcast(true)
            .map_err(|e| Error::NetworkError(format!("Failed to set broadcast: {}", e)))?;
        
        // Send M-SEARCH request
        let search_types = vec![
            "urn:schemas-upnp-org:device:MediaServer:1",
            "urn:schemas-upnp-org:service:ContentDirectory:1",
            "urn:schemas-upnp-org:device:Basic:1",
            "upnp:rootdevice",
            "ssdp:all",
        ];
        
        let multicast_addr = SSDP_MULTICAST_ADDR.parse::<SocketAddr>()
            .map_err(|e| Error::NetworkError(format!("Failed to parse multicast address: {}", e)))?;
        
        for search_type in search_types {
            let message = format!(
                "M-SEARCH * HTTP/1.1\r\n\
                HOST: {}\r\n\
                MAN: \"ssdp:discover\"\r\n\
                MX: 5\r\n\
                ST: {}\r\n\r\n",
                SSDP_MULTICAST_ADDR, search_type
            );
            
            socket.send_to(message.as_bytes(), multicast_addr).await
                .map_err(|e| Error::NetworkError(format!("Failed to send discovery message: {}", e)))?;
        }
        
        // Collect responses
        let mut buffer = vec![0u8; 8192];
        let mut devices = Vec::new();
        let timeout = Duration::from_secs(config.timeout);
        let start_time = Instant::now();
        
        // Track already discovered device locations to avoid duplicates
        let mut discovered_locations = HashSet::new();
        
        while start_time.elapsed() < timeout {
            // Set socket timeout
            let remaining_time = timeout.checked_sub(start_time.elapsed()).unwrap_or(Duration::from_millis(100));
            if remaining_time.is_zero() {
                break;
            }
            
            // Try to receive a response with timeout
            let result = tokio::time::timeout(
                remaining_time,
                socket.recv_from(&mut buffer)
            ).await;
            
            match result {
                Ok(Ok((size, addr))) => {
                    trace!("Received {} bytes from {}", size, addr);
                    let response = String::from_utf8_lossy(&buffer[..size]);
                    
                    // Extract LOCATION header
                    if let Some(location) = self.extract_header(&response, "LOCATION") {
                        // Skip if we've already seen this location
                        if !discovered_locations.insert(location.clone()) {
                            continue;
                        }
                        
                        // Parse device information from the location URL
                        if let Some(device) = self.fetch_device_description(&location, addr.ip()).await {
                            trace!("Discovered SSDP device: {:?}", device);
                            devices.push(device);
                        }
                    }
                },
                Ok(Err(e)) => {
                    warn!("Error receiving SSDP discovery response: {}", e);
                    continue;
                },
                Err(_) => {
                    // Timeout reached
                    break;
                }
            }
        }
        
        Ok(devices)
    }
}

impl SSDPDiscovery {
    /// Extract a header value from an HTTP-like response
    fn extract_header<'a>(&self, response: &'a str, header: &str) -> Option<&'a str> {
        let header_prefix = format!("{}: ", header);
        for line in response.lines() {
            if line.starts_with(&header_prefix) {
                return Some(line[header_prefix.len()..].trim());
            }
        }
        None
    }
    
    /// Fetch device description from the location URL
    async fn fetch_device_description(&self, location: &str, ip: IpAddr) -> Option<DiscoveredDevice> {
        // Parse URL
        let url = url::Url::parse(location).ok()?;
        let host = url.host_str()?;
        let port = url.port().unwrap_or(80);
        
        // Create HTTP client
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .ok()?;
        
        // Fetch device description
        let response = match client.get(location).send().await {
            Ok(response) => response,
            Err(e) => {
                warn!("Failed to fetch device description from {}: {}", location, e);
                return None;
            }
        };
        
        // Check if the response is successful
        if !response.status().is_success() {
            warn!("Failed to fetch device description from {}: {}", location, response.status());
            return None;
        }
        
        // Get response text
        let text = match response.text().await {
            Ok(text) => text,
            Err(e) => {
                warn!("Failed to read device description from {}: {}", location, e);
                return None;
            }
        };
        
        // Parse XML to extract device information
        // A real implementation would use a proper XML parser
        let device_type = self.extract_xml_field(&text, "deviceType")?;
        let friendly_name = self.extract_xml_field(&text, "friendlyName").unwrap_or_else(|| format!("Device at {}", ip));
        let manufacturer = self.extract_xml_field(&text, "manufacturer").unwrap_or_else(|| "Unknown".to_string());
        let model_name = self.extract_xml_field(&text, "modelName").unwrap_or_else(|| "Unknown".to_string());
        let model_number = self.extract_xml_field(&text, "modelNumber").unwrap_or_default();
        let serial_number = self.extract_xml_field(&text, "serialNumber");
        let udn = self.extract_xml_field(&text, "UDN")?;
        
        // Determine device type
        let device_type = if device_type.contains("MediaServer") {
            DeviceType::NVR(model_name.clone())
        } else if device_type.contains("BasicDevice") || device_type.contains("Webcam") {
            DeviceType::Camera(CameraType::IPCamera)
        } else {
            DeviceType::Other(device_type)
        };
        
        // Create device ID
        let device_id = match udn.strip_prefix("uuid:") {
            Some(uuid) => DeviceId::from_unique_id(uuid),
            None => DeviceId::from_unique_id(&udn),
        };
        
        // Extract services
        let mut protocols = vec![NetworkProtocol::Discovery(String::from("SSDP"))];
        
        // Check for media-related services that might indicate streaming capability
        if text.contains("AVTransport") || text.contains("RenderingControl") {
            protocols.push(NetworkProtocol::Stream(StreamProtocol::HTTP));
        }
        
        // Current timestamp
        let now = Utc::now();
        
        // Create the discovered device
        Some(DiscoveredDevice {
            id: device_id,
            device_type,
            manufacturer,
            model: model_name,
            name: friendly_name,
            ip_address: ip,
            port,
            mac_address: None,
            serial_number,
            firmware_version: None,
            location: None,
            protocols,
            capabilities: Vec::new(),
            online: true,
            first_discovered: now,
            last_seen: now,
            metadata: HashMap::from([
                ("location".to_string(), location.to_string()),
                ("UDN".to_string(), udn),
                ("model_number".to_string(), model_number),
            ]),
        })
    }
    
    /// Extract a field from an XML document
    fn extract_xml_field(&self, xml: &str, field_name: &str) -> Option<String> {
        let start_tag = format!("<{}>", field_name);
        let end_tag = format!("</{}>", field_name);
        
        let start_idx = xml.find(&start_tag)?;
        let end_idx = xml.find(&end_tag)?;
        
        if start_idx < end_idx {
            let start_pos = start_idx + start_tag.len();
            Some(xml[start_pos..end_idx].trim().to_string())
        } else {
            None
        }
    }
}

/// mDNS discovery method implementation
pub struct MDNSDiscovery;

impl MDNSDiscovery {
    /// Create a new mDNS discovery instance
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DiscoveryMethod for MDNSDiscovery {
    fn name(&self) -> &'static str {
        "mDNS"
    }
    
    fn supported_protocols(&self) -> Vec<NetworkProtocol> {
        vec![
            NetworkProtocol::Discovery(String::from("mDNS")),
            NetworkProtocol::API(String::from("REST")),
            NetworkProtocol::Stream(StreamProtocol::HTTP),
            NetworkProtocol::Stream(StreamProtocol::HTTPS),
        ]
    }
    
    async fn discover(&self, config: &ScanConfig) -> Result<Vec<DiscoveredDevice>> {
        // Create a socket for discovery
        let socket = UdpSocket::bind("0.0.0.0:5353").await
            .map_err(|e| Error::NetworkError(format!("Failed to bind mDNS socket: {}", e)))?;
        
        // Join multicast group
        let multicast_addr = MDNS_MULTICAST_ADDR.parse::<SocketAddr>()
            .map_err(|e| Error::NetworkError(format!("Failed to parse multicast address: {}", e)))?;
        
        #[cfg(unix)]
        {
            use socket2::{Domain, Protocol, Socket, Type};
            use std::os::unix::io::{AsRawFd, FromRawFd};
            
            let socket2 = unsafe {
                Socket::from_raw_fd(socket.as_raw_fd())
            };
            
            socket2.join_multicast_v4(
                &multicast_addr.ip().to_string().parse::<Ipv4Addr>().unwrap(),
                &Ipv4Addr::UNSPECIFIED,
            ).map_err(|e| Error::NetworkError(format!("Failed to join multicast group: {}", e)))?;
            
            std::mem::forget(socket2); // Avoid closing the fd
        }
        
        // Send mDNS queries for different service types
        let service_types = vec![
            "_http._tcp.local",
            "_https._tcp.local",
            "_rtsp._tcp.local",
            "_camera._tcp.local",
            "_axis-video._tcp.local",
            "_dahua-dss._tcp.local",
            "_hikvision._tcp.local",
            "_nvr._tcp.local",
        ];
        
        for service_type in &service_types {
            let query = self.create_mdns_query(service_type);
            socket.send_to(&query, multicast_addr).await
                .map_err(|e| Error::NetworkError(format!("Failed to send mDNS query: {}", e)))?;
        }
        
        // Collect responses
        let mut buffer = vec![0u8; 8192];
        let mut devices = Vec::new();
        let timeout = Duration::from_secs(config.timeout);
        let start_time = Instant::now();
        
        // Track already discovered services to avoid duplicates
        let mut discovered_services = HashSet::new();
        
        while start_time.elapsed() < timeout {
            // Set socket timeout
            let remaining_time = timeout.checked_sub(start_time.elapsed()).unwrap_or(Duration::from_millis(100));
            if remaining_time.is_zero() {
                break;
            }
            
            // Try to receive a response with timeout
            let result = tokio::time::timeout(
                remaining_time,
                socket.recv_from(&mut buffer)
            ).await;
            
            match result {
                Ok(Ok((size, addr))) => {
                    trace!("Received {} bytes from {}", size, addr);
                    
                    // Parse mDNS response
                    // A real implementation would use a proper DNS parser library
                    if let Some(services) = self.parse_mdns_response(&buffer[..size]) {
                        for service in services {
                            // Skip if we've already seen this service
                            if !discovered_services.insert(service.clone()) {
                                continue;
                            }
                            
                            // Resolve service to get device information
                            if let Some(device) = self.resolve_mdns_service(&service, addr.ip()).await {
                                trace!("Discovered mDNS device: {:?}", device);
                                devices.push(device);
                            }
                        }
                    }
                },
                Ok(Err(e)) => {
                    warn!("Error receiving mDNS discovery response: {}", e);
                    continue;
                },
                Err(_) => {
                    // Timeout reached
                    break;
                }
            }
        }
        
        Ok(devices)
    }
}

impl MDNSDiscovery {
    /// Create an mDNS query packet
    fn create_mdns_query(&self, service_type: &str) -> Vec<u8> {
        // DNS header
        let mut packet = vec![
            0x00, 0x00, // Transaction ID (0 for mDNS)
            0x00, 0x00, // Flags (standard query)
            0x00, 0x01, // Questions (1)
            0x00, 0x00, // Answer RRs (0)
            0x00, 0x00, // Authority RRs (0)
            0x00, 0x00, // Additional RRs (0)
        ];
        
        // Encode query name
        for part in service_type.split('.') {
            packet.push(part.len() as u8);
            packet.extend_from_slice(part.as_bytes());
        }
        packet.push(0); // Terminator
        
        // Query type (PTR) and class (IN)
        packet.extend_from_slice(&[0x00, 0x0c, 0x00, 0x01]);
        
        packet
    }
    
    /// Parse mDNS response packet
    fn parse_mdns_response(&self, packet: &[u8]) -> Option<Vec<String>> {
        // Simple parsing of mDNS response to extract service names
        // A real implementation would use a proper DNS parser library
        
        if packet.len() < 12 {
            return None;
        }
        
        // Extract counts from DNS header
        let answers = ((packet[6] as u16) << 8) | packet[7] as u16;
        let authorities = ((packet[8] as u16) << 8) | packet[9] as u16;
        let additionals = ((packet[10] as u16) << 8) | packet[11] as u16;
        
        if answers + authorities + additionals == 0 {
            return None;
        }
        
        // Simple extraction of service names
        // This is a simplified approach - a real implementation would properly parse DNS records
        let mut services = Vec::new();
        let mut pos = 12; // Skip DNS header
        
        // Skip questions
        let questions = ((packet[4] as u16) << 8) | packet[5] as u16;
        for _ in 0..questions {
            // Skip name
            while pos < packet.len() {
                let len = packet[pos] as usize;
                if len == 0 {
                    pos += 1;
                    break;
                }
                pos += 1 + len;
                if pos >= packet.len() {
                    return None;
                }
            }
            
            // Skip type and class
            pos += 4;
            if pos > packet.len() {
                return None;
            }
        }
        
        // Process answers, authorities, and additionals to extract service names
        for _ in 0..(answers + authorities + additionals) {
            if pos + 10 > packet.len() {
                break;
            }
            
            // Extract name
            let mut service_name = String::new();
            let mut name_pos = pos;
            
            loop {
                if name_pos >= packet.len() {
                    break;
                }
                
                let len = packet[name_pos] as usize;
                if len == 0 {
                    name_pos += 1;
                    break;
                }
                
                // Check for DNS name compression (pointer)
                if (len & 0xc0) == 0xc0 {
                    // Skip compressed name for simplicity
                    name_pos += 2;
                    break;
                }
                
                name_pos += 1;
                if name_pos + len > packet.len() {
                    break;
                }
                
                if !service_name.is_empty() {
                    service_name.push('.');
                }
                
                service_name.push_str(
                    std::str::from_utf8(&packet[name_pos..name_pos + len]).unwrap_or_default()
                );
                
                name_pos += len;
            }
            
            if !service_name.is_empty() {
                services.push(service_name);
            }
            
            // Skip to next record
            pos = name_pos;
            if pos + 10 > packet.len() {
                break;
            }
            
            // Skip type, class, TTL, and data length
            let data_len = ((packet[pos + 8] as u16) << 8) | packet[pos + 9] as u16;
            pos += 10 + data_len as usize;
            
            if pos > packet.len() {
                break;
            }
        }
        
        if services.is_empty() {
            None
        } else {
            Some(services)
        }
    }
    
    /// Resolve mDNS service to get device information
    async fn resolve_mdns_service(&self, service: &str, ip: IpAddr) -> Option<DiscoveredDevice> {
        // Parse service name to determine device type
        let service_type = if service.contains("_camera") {
            DeviceType::Camera(CameraType::IPCamera)
        } else if service.contains("_rtsp") {
            DeviceType::Camera(CameraType::IPCamera)
        } else if service.contains("_nvr") {
            DeviceType::NVR(String::from("Unknown"))
        } else if service.contains("_axis") {
            DeviceType::Camera(CameraType::IPCamera)
        } else if service.contains("_dahua") {
            DeviceType::Camera(CameraType::IPCamera)
        } else if service.contains("_hikvision") {
            DeviceType::Camera(CameraType::IPCamera)
        } else {
            DeviceType::Other(String::from("mDNS Device"))
        };
        
        // Extract manufacturer from service name if possible
        let manufacturer = if service.contains("axis") {
            "Axis"
        } else if service.contains("dahua") {
            "Dahua"
        } else if service.contains("hikvision") {
            "Hikvision"
        } else {
            "Unknown"
        }.to_string();
        
        // Determine protocols based on service type
        let mut protocols = vec![NetworkProtocol::Discovery(String::from("mDNS"))];
        
        if service.contains("_http") {
            protocols.push(NetworkProtocol::Stream(StreamProtocol::HTTP));
            protocols.push(NetworkProtocol::API(String::from("REST")));
        }
        
        if service.contains("_https") {
            protocols.push(NetworkProtocol::Stream(StreamProtocol::HTTPS));
            protocols.push(NetworkProtocol::API(String::from("REST")));
        }
        
        if service.contains("_rtsp") {
            protocols.push(NetworkProtocol::Stream(StreamProtocol::RTSP));
        }
        
        // Create device ID
        let device_id = DeviceId::from_unique_id(&format!("mdns:{}", service));
        
        // Current timestamp
        let now = Utc::now();
        
        // Create the discovered device
        Some(DiscoveredDevice {
            id: device_id,
            device_type,
            manufacturer,
            model: "Unknown".to_string(),
            name: service.to_string(),
            ip_address: ip,
            port: 80, // Default port, would be extracted from SRV record in a real implementation
            mac_address: None,
            serial_number: None,
            firmware_version: None,
            location: None,
            protocols,
            capabilities: Vec::new(),
            online: true,
            first_discovered: now,
            last_seen: now,
            metadata: HashMap::from([
                ("service".to_string(), service.to_string()),
            ]),
        })
    }
}

/// Network port scanning method
pub struct PortScanner;

impl PortScanner {
    /// Create a new port scanner instance
    pub fn new() -> Self {
        Self
    }
    
    /// Scan common camera ports on the given IP address
    async fn scan_ip(&self, ip: IpAddr, timeout: Duration) -> Result<Option<DiscoveredDevice>> {
        // Common ports for cameras and NVRs
        let camera_ports = vec![
            (80, NetworkProtocol::Stream(StreamProtocol::HTTP)),
            (443, NetworkProtocol::Stream(StreamProtocol::HTTPS)),
            (554, NetworkProtocol::Stream(StreamProtocol::RTSP)),
            (1935, NetworkProtocol::Stream(StreamProtocol::RTMP)),
            (8000, NetworkProtocol::Stream(StreamProtocol::HTTP)), // Hikvision
            (8554, NetworkProtocol::Stream(StreamProtocol::RTSP)),
            (8080, NetworkProtocol::Stream(StreamProtocol::HTTP)), // Common alternative HTTP
            (9000, NetworkProtocol::Stream(StreamProtocol::HTTP)), // Dahua
            (37777, NetworkProtocol::Stream(StreamProtocol::HTTP)), // Dahua
        ];
        
        let mut open_ports = Vec::new();
        let mut futures = FuturesUnordered::new();
        
        // Create a future for each port scan
        for (port, protocol) in camera_ports {
            let ip_clone = ip;
            let protocol_clone = protocol.clone();
            
            futures.push(async move {
                match tokio::net::TcpStream::connect(format!("{}:{}", ip_clone, port)).await {
                    Ok(_) => Some((port, protocol_clone)),
                    Err(_) => None,
                }
            });
        }
        
        // Wait for all port scans to complete or timeout
        let start_time = Instant::now();
        
        while let Some(result) = futures.next().await {
            if start_time.elapsed() > timeout {
                break;
            }
            
            if let Some((port, protocol)) = result {
                open_ports.push((port, protocol));
            }
        }
        
        // If no open ports, return None
        if open_ports.is_empty() {
            return Ok(None);
        }
        
        // Create protocols list from open ports
        let protocols = open_ports.iter().map(|(_, protocol)| protocol.clone()).collect();
        
        // Probe HTTP port if available to get more device information
        let http_port = open_ports.iter()
            .find(|(_, protocol)| matches!(protocol, NetworkProtocol::Stream(StreamProtocol::HTTP)))
            .map(|(port, _)| *port);
        
        let name = match http_port {
            Some(port) => {
                match self.probe_http_port(ip, port).await {
                    Ok(Some((device_name, manufacturer, model))) => {
                        // Current timestamp
                        let now = Utc::now();
                        
                        // Create a more detailed device
                        return Ok(Some(DiscoveredDevice {
                            id: DeviceId::from_unique_id(&format!("port:{}:{}", ip, port)),
                            device_type: DeviceType::Camera(CameraType::IPCamera),
                            manufacturer,
                            model,
                            name: device_name,
                            ip_address: ip,
                            port,
                            mac_address: None,
                            serial_number: None,
                            firmware_version: None,
                            location: None,
                            protocols,
                            capabilities: Vec::new(),
                            online: true,
                            first_discovered: now,
                            last_seen: now,
                            metadata: HashMap::from([
                                ("discovery_method".to_string(), "port_scan".to_string()),
                            ]),
                        }));
                    },
                    _ => format!("IP Camera ({})", ip),
                }
            },
            None => format!("IP Camera ({})", ip),
        };
        
        // Current timestamp
        let now = Utc::now();
        
        // Create a basic device with the open ports
        let device = DiscoveredDevice {
            id: DeviceId::from_unique_id(&format!("port:{}", ip)),
            device_type: DeviceType::Camera(CameraType::IPCamera),
            manufacturer: "Unknown".to_string(),
            model: "Unknown".to_string(),
            name,
            ip_address: ip,
            port: open_ports[0].0, // Use first open port
            mac_address: None,
            serial_number: None,
            firmware_version: None,
            location: None,
            protocols,
            capabilities: Vec::new(),
            online: true,
            first_discovered: now,
            last_seen: now,
            metadata: HashMap::from([
                ("discovery_method".to_string(), "port_scan".to_string()),
            ]),
        };
        
        Ok(Some(device))
    }
    
    /// Probe HTTP port to get more device information
    async fn probe_http_port(&self, ip: IpAddr, port: u16) -> Result<Option<(String, String, String)>> {
        // Create HTTP client with short timeout
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .map_err(|e| Error::NetworkError(format!("Failed to create HTTP client: {}", e)))?;
        
        // Try to get device information from common paths
        let paths = vec![
            "/", 
            "/info", 
            "/device_info", 
            "/system/deviceInfo", 
            "/onvif-http/snapshot"
        ];
        
        for path in paths {
            let url = format!("http://{}:{}{}", ip, port, path);
            
            match client.get(&url).send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        // Try to parse response
                        if let Ok(text) = response.text().await {
                            // Look for common patterns in the response to identify the device
                            let device_name = if text.contains("<title>") && text.contains("</title>") {
                                let start = text.find("<title>").unwrap() + 7;
                                let end = text.find("</title>").unwrap();
                                if start < end {
                                    text[start..end].trim().to_string()
                                } else {
                                    format!("Web Camera ({})", ip)
                                }
                            } else {
                                format!("Web Camera ({})", ip)
                            };
                            
                            // Extract manufacturer
                            let manufacturer = if text.to_lowercase().contains("hikvision") {
                                "Hikvision".to_string()
                            } else if text.to_lowercase().contains("dahua") {
                                "Dahua".to_string()
                            } else if text.to_lowercase().contains("axis") {
                                "Axis".to_string()
                            } else if text.to_lowercase().contains("samsung") {
                                "Samsung".to_string()
                            } else if text.to_lowercase().contains("sony") {
                                "Sony".to_string()
                            } else if text.to_lowercase().contains("panasonic") {
                                "Panasonic".to_string()
                            } else {
                                "Unknown".to_string()
                            };
                            
                            // Simplified model detection - a real implementation would use more sophisticated parsing
                            let model = "Unknown".to_string();
                            
                            return Ok(Some((device_name, manufacturer, model)));
                        }
                    }
                },
                Err(_) => {
                    // Continue to next path on error
                    continue;
                }
            }
        }
        
        Ok(None)
    }
}

#[async_trait]
impl DiscoveryMethod for PortScanner {
    fn name(&self) -> &'static str {
        "Port Scanner"
    }
    
    fn supported_protocols(&self) -> Vec<NetworkProtocol> {
        vec![
            NetworkProtocol::Stream(StreamProtocol::HTTP),
            NetworkProtocol::Stream(StreamProtocol::HTTPS),
            NetworkProtocol::Stream(StreamProtocol::RTSP),
            NetworkProtocol::Stream(StreamProtocol::RTMP),
        ]
    }
    
    async fn discover(&self, config: &ScanConfig) -> Result<Vec<DiscoveredDevice>> {
        let mut devices = Vec::new();
        let timeout = Duration::from_secs(config.timeout);
        
        // Process each IP range
        for range_str in &config.ip_ranges {
            // Parse IP range - simplified approach for common formats
            let ips = match self.parse_ip_range(range_str) {
                Ok(ips) => ips,
                Err(e) => {
                    warn!("Failed to parse IP range {}: {}", range_str, e);
                    continue;
                }
            };
            
            // Scan IPs in parallel with concurrency limit
            let mut futures = FuturesUnordered::new();
            let max_concurrent = 100; // Limit concurrent scans
            
            for ip in ips {
                if futures.len() >= max_concurrent {
                    if let Some(result) = futures.next().await {
                        match result {
                            Ok(Some(device)) => devices.push(device),
                            Ok(None) => {},
                            Err(e) => warn!("Error scanning IP: {}", e),
                        }
                    }
                }
                
                let timeout_clone = timeout;
                futures.push(self.scan_ip(ip, timeout_clone));
            }
            
            // Complete remaining futures
            while let Some(result) = futures.next().await {
                match result {
                    Ok(Some(device)) => devices.push(device),
                    Ok(None) => {},
                    Err(e) => warn!("Error scanning IP: {}", e),
                }
            }
        }
        
        Ok(devices)
    }
}

impl PortScanner {
    /// Parse IP range string into a list of IP addresses
    fn parse_ip_range(&self, range_str: &str) -> Result<Vec<IpAddr>> {
        if range_str.contains('/') {
            // CIDR notation
            self.parse_cidr(range_str)
        } else if range_str.contains('-') {
            // Range notation (e.g., 192.168.1.1-192.168.1.10)
            self.parse_range(range_str)
        } else {
            // Single IP
            match range_str.parse::<IpAddr>() {
                Ok(ip) => Ok(vec![ip]),
                Err(e) => Err(Error::InvalidInput(format!("Invalid IP address: {}", e))),
            }
        }
    }
    
    /// Parse CIDR notation into a list of IP addresses
    fn parse_cidr(&self, cidr: &str) -> Result<Vec<IpAddr>> {
        // Use ipnetwork crate for proper CIDR parsing
        let network = ipnetwork::IpNetwork::from_str(cidr)
            .map_err(|e| Error::InvalidInput(format!("Invalid CIDR notation: {}", e)))?;
        
        // For IPv4, generate all IPs; for IPv6, limit to first 1000 for performance
        let hosts = match network {
            ipnetwork::IpNetwork::V4(net) => {
                if net.prefix() < 16 {
                    // Too large, limit to first 1000
                    net.iter().take(1000).map(IpAddr::V4).collect()
                } else {
                    net.iter().map(IpAddr::V4).collect()
                }
            },
            ipnetwork::IpNetwork::V6(net) => {
                if net.prefix() < 112 {
                    // Too large, limit to first 1000
                    net.iter().take(1000).map(IpAddr::V6).collect()
                } else {
                    net.iter().map(IpAddr::V6).collect()
                }
            },
        };
        
        Ok(hosts)
    }
    
    /// Parse IP range notation into a list of IP addresses
    fn parse_range(&self, range: &str) -> Result<Vec<IpAddr>> {
        let parts: Vec<&str> = range.split('-').collect();
        if parts.len() != 2 {
            return Err(Error::InvalidInput(format!("Invalid IP range: {}", range)));
        }
        
        let start_ip = parts[0].trim().parse::<Ipv4Addr>()
            .map_err(|e| Error::InvalidInput(format!("Invalid start IP: {}", e)))?;
        
        let end_str = parts[1].trim();
        let end_ip = if end_str.contains('.') {
            // Full IP address
            end_str.parse::<Ipv4Addr>()
                .map_err(|e| Error::InvalidInput(format!("Invalid end IP: {}", e)))?
        } else {
            // Just the last octet
            let mut octets = start_ip.octets();
            let last_octet = end_str.parse::<u8>()
                .map_err(|e| Error::InvalidInput(format!("Invalid end octet: {}", e)))?;
            octets[3] = last_octet;
            Ipv4Addr::new(octets[0], octets[1], octets[2], octets[3])
        };
        
        let start_u32 = u32::from(start_ip);
        let end_u32 = u32::from(end_ip);
        
        if end_u32 < start_u32 {
            return Err(Error::InvalidInput(format!("End IP is before start IP: {}", range)));
        }
        
        if end_u32 - start_u32 > 1000 {
            // Limit to 1000 IPs for performance
            return Err(Error::InvalidInput(format!("IP range too large: {}", range)));
        }
        
        let ips = (start_u32..=end_u32)
            .map(|ip| IpAddr::V4(Ipv4Addr::from(ip)))
            .collect();
        
        Ok(ips)
    }
}

/// The NetworkDiscovery struct manages the discovery of devices on the network
pub struct NetworkDiscovery {
    /// Configuration for network discovery
    config: Arc<NetworkDiscoveryConfig>,
    /// Available discovery methods
    methods: Arc<Vec<Box<dyn DiscoveryMethod>>>,
    /// Registry of discovered devices
    device_registry: Arc<RwLock<HashMap<DeviceId, DiscoveredDevice>>>,
    /// Event sender to publish discovery events
    event_sender: broadcast::Sender<DiscoveryEvent>,
    /// Storage manager for persistence
    storage: Arc<dyn PersistentStorage>,
    /// Encryption manager for credentials
    encryption: Arc<EncryptionManager>,
    /// Discovery task handle
    discovery_task: Arc<Mutex<Option<JoinHandle<()>>>>,
    /// Running status
    running: Arc<RwLock<bool>>,
}

impl NetworkDiscovery {
    /// Create a new NetworkDiscovery instance
    pub fn new(
        config: NetworkDiscoveryConfig,
        storage: Arc<dyn PersistentStorage>,
        encryption: Arc<EncryptionManager>,
    ) -> Self {
        // Create discovery methods
        let methods: Vec<Box<dyn DiscoveryMethod>> = vec![
            Box::new(OnvifDiscovery::new()),
            Box::new(SSDPDiscovery::new()),
            Box::new(MDNSDiscovery::new()),
            Box::new(PortScanner::new()),
        ];
        
        // Create event channel
        let (event_sender, _) = broadcast::channel(1000);
        
        Self {
            config: Arc::new(config),
            methods: Arc::new(methods),
            device_registry: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            storage,
            encryption,
            discovery_task: Arc::new(Mutex::new(None)),
            running: Arc::new(RwLock::new(false)),
        }
    }
    
    /// Start the discovery process
    pub async fn start(&self) -> Result<()> {
        let mut running = self.running.write().unwrap();
        if *running {
            return Err(Error::InvalidState("Network discovery already running".to_string()));
        }
        
        *running = true;
        drop(running);
        
        // Load previously discovered devices from storage
        self.load_devices().await?;
        
        // Start the discovery task
        let config = self.config.clone();
        let methods = self.methods.clone();
        let device_registry = self.device_registry.clone();
        let event_sender = self.event_sender.clone();
        let running = self.running.clone();
        
        let task = tokio::spawn(async move {
            let discovery_interval = Duration::from_secs(
                config.discovery_interval.unwrap_or(DEFAULT_DISCOVERY_INTERVAL)
            );
            
            info!("Starting network discovery with interval of {} seconds", discovery_interval.as_secs());
            
            // Initial scan
            if let Err(e) = Self::run_discovery_scan(config.clone(), methods.clone(), device_registry.clone(), event_sender.clone()).await {
                error!("Initial discovery scan failed: {}", e);
            }
            
            // Periodic scans
            let mut interval = time::interval(discovery_interval);
            
            loop {
                interval.tick().await;
                
                // Check if still running
                let is_running = { *running.read().unwrap() };
                if !is_running {
                    break;
                }
                
                // Run discovery scan
                if let Err(e) = Self::run_discovery_scan(config.clone(), methods.clone(), device_registry.clone(), event_sender.clone()).await {
                    error!("Discovery scan failed: {}", e);
                }
            }
        });
        
        // Store task handle
        let mut discovery_task = self.discovery_task.lock().await;
        *discovery_task = Some(task);
        
        Ok(())
    }
    
    /// Run a discovery scan using all available methods
    async fn run_discovery_scan(
        config: Arc<NetworkDiscoveryConfig>,
        methods: Arc<Vec<Box<dyn DiscoveryMethod>>>,
        device_registry: Arc<RwLock<HashMap<DeviceId, DiscoveredDevice>>>,
        event_sender: broadcast::Sender<DiscoveryEvent>,
    ) -> Result<()> {
        // Create scan configuration
        let scan_config = ScanConfig {
            ip_ranges: config.ip_ranges.clone(),
            protocols: config.protocols.clone(),
            timeout: config.scan_timeout.unwrap_or(DEFAULT_SCAN_TIMEOUT),
            retries: config.discovery_retries.unwrap_or(DEFAULT_DISCOVERY_RETRIES),
            parameters: config.parameters.clone(),
        };
        
        // Create scan record
        let scan_id = Uuid::new_v4();
        let mut scan = NetworkScan {
            id: scan_id,
            start_time: Utc::now(),
            end_time: None,
            devices_discovered: 0,
            new_devices: 0,
            updated_devices: 0,
            scanned_ranges: scan_config.ip_ranges.clone(),
            protocols: scan_config.protocols.clone(),
            status: ScanStatus::InProgress,
            error_message: None,
        };
        
        info!("Starting network scan with ID {}", scan_id);
        
        // Run discovery methods in parallel
        let mut tasks = Vec::new();
        
        for method in methods.iter() {
            let config_clone = scan_config.clone();
            
            // Check if method supports any of the requested protocols
            let supported = scan_config.protocols.iter().any(|protocol| method.supports_protocol(protocol));
            if !supported {
                continue;
            }
            
            let method_name = method.name();
            let method_task = method.discover(&config_clone);
            
            tasks.push(async move {
                match method_task.await {
                    Ok(devices) => {
                        info!("{} discovered {} devices", method_name, devices.len());
                        Ok(devices)
                    },
                    Err(e) => {
                        error!("{} discovery failed: {}", method_name, e);
                        Err(e)
                    }
                }
            });
        }
        
        // Wait for all tasks to complete
        let results = join_all(tasks).await;
        
        // Process results
        let mut device_count = 0;
        let mut new_devices = 0;
        let mut updated_devices = 0;
        let mut success = true;
        let mut error_message = None;
        
        // Update device registry
        {
            let mut registry = device_registry.write().unwrap();
            
            for result in results {
                match result {
                    Ok(devices) => {
                        device_count += devices.len();
                        
                        for device in devices {
                            // Check if device already exists
                            let existing = registry.get_mut(&device.id);
                            
                            if let Some(existing_device) = existing {
                                // Update existing device
                                let updated_device = device.clone();
                                existing_device.update_from(&updated_device);
                                updated_devices += 1;
                                
                                // Send device updated event
                                let _ = event_sender.send(DiscoveryEvent::DeviceUpdated(existing_device.clone()));
                            } else {
                                // Add new device
                                new_devices += 1;
                                registry.insert(device.id.clone(), device.clone());
                                
                                // Send device discovered event
                                let _ = event_sender.send(DiscoveryEvent::DeviceDiscovered(device));
                            }
                        }
                    },
                    Err(e) => {
                        success = false;
                        error_message = Some(e.to_string());
                    }
                }
            }
        }
        
        // Update scan record
        scan.end_time = Some(Utc::now());
        scan.devices_discovered = device_count;
        scan.new_devices = new_devices;
        scan.updated_devices = updated_devices;
        scan.status = if success { ScanStatus::Completed } else { ScanStatus::Failed };
        scan.error_message = error_message;
        
        // Send scan completed event
        let event = if success {
            DiscoveryEvent::ScanCompleted(scan.clone())
        } else {
            DiscoveryEvent::ScanFailed(Error::DiscoveryError(
                scan.error_message.clone().unwrap_or_else(|| "Unknown error".to_string())
            ))
        };
        
        let _ = event_sender.send(event);
        
        if success {
            info!("Network scan completed: {} devices discovered ({} new, {} updated)",
                  device_count, new_devices, updated_devices);
        } else {
            error!("Network scan failed: {}",
                   scan.error_message.as_deref().unwrap_or("Unknown error"));
        }
        
        Ok(())
    }
    
    /// Stop the discovery process
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().unwrap();
        if !*running {
            return Err(Error::InvalidState("Network discovery not running".to_string()));
        }
        
        *running = false;
        drop(running);
        
        // Stop the discovery task
        let mut discovery_task = self.discovery_task.lock().await;
        if let Some(task) = discovery_task.take() {
            task.abort();
        }
        
        // Save discovered devices
        self.save_devices().await?;
        
        info!("Network discovery stopped");
        Ok(())
    }
    
    /// Subscribe to discovery events
    pub fn subscribe(&self) -> broadcast::Receiver<DiscoveryEvent> {
        self.event_sender.subscribe()
    }
    
    /// Get all discovered devices
    pub fn get_devices(&self) -> Vec<DiscoveredDevice> {
        let registry = self.device_registry.read().unwrap();
        registry.values().cloned().collect()
    }
    
    /// Get a specific device by ID
    pub fn get_device(&self, id: &DeviceId) -> Option<DiscoveredDevice> {
        let registry = self.device_registry.read().unwrap();
        registry.get(id).cloned()
    }
    
    /// Get devices by type
    pub fn get_devices_by_type(&self, device_type: &DeviceType) -> Vec<DiscoveredDevice> {
        let registry = self.device_registry.read().unwrap();
        registry.values()
            .filter(|device| {
                match (device_type, &device.device_type) {
                    (DeviceType::Camera(_), DeviceType::Camera(_)) => true,
                    (DeviceType::IoTSensor(_), DeviceType::IoTSensor(_)) => true,
                    (DeviceType::RFSource(_), DeviceType::RFSource(_)) => true,
                    (DeviceType::AccessControl(_), DeviceType::AccessControl(_)) => true,
                    (DeviceType::AudioCapture(_), DeviceType::AudioCapture(_)) => true,
                    (DeviceType::ThermalImaging(_), DeviceType::ThermalImaging(_)) => true,
                    (DeviceType::NVR(_), DeviceType::NVR(_)) => true,
                    (DeviceType::LPR(_), DeviceType::LPR(_)) => true,
                    (DeviceType::Other(_), DeviceType::Other(_)) => true,
                    _ => false,
                }
            })
            .cloned()
            .collect()
    }
    
    /// Add a device manually
    pub fn add_device(&self, device: DiscoveredDevice) -> Result<()> {
        let mut registry = self.device_registry.write().unwrap();
        
        // Check if device already exists
        if registry.contains_key(&device.id) {
            return Err(Error::DuplicateEntity(format!("Device with ID {} already exists", device.id.as_str())));
        }
        
        // Add device
        registry.insert(device.id.clone(), device.clone());
        
        // Send device discovered event
        let _ = self.event_sender.send(DiscoveryEvent::DeviceDiscovered(device));
        
        Ok(())
    }
    
    /// Update a device
    pub fn update_device(&self, device: DiscoveredDevice) -> Result<()> {
        let mut registry = self.device_registry.write().unwrap();
        
        // Check if device exists
        if !registry.contains_key(&device.id) {
            return Err(Error::EntityNotFound(format!("Device with ID {} not found", device.id.as_str())));
        }
        
        // Update device
        registry.insert(device.id.clone(), device.clone());
        
        // Send device updated event
        let _ = self.event_sender.send(DiscoveryEvent::DeviceUpdated(device));
        
        Ok(())
    }
    
    /// Remove a device
    pub fn remove_device(&self, id: &DeviceId) -> Result<()> {
        let mut registry = self.device_registry.write().unwrap();
        
        // Check if device exists
        if !registry.contains_key(id) {
            return Err(Error::EntityNotFound(format!("Device with ID {} not found", id.as_str())));
        }
        
        // Remove device
        registry.remove(id);
        
        // Send device removed event
        let _ = self.event_sender.send(DiscoveryEvent::DeviceRemoved(id.clone()));
        
        Ok(())
    }
    
    /// Set device credentials
    pub fn set_device_credentials(&self, id: &DeviceId, credentials: DeviceCredentials) -> Result<()> {
        let mut registry = self.device_registry.write().unwrap();
        
        // Check if device exists
        let device = match registry.get_mut(id) {
            Some(device) => device,
            None => return Err(Error::EntityNotFound(format!("Device with ID {} not found", id.as_str()))),
        };
        
        // Encrypt credentials
        let encrypted_username = self.encryption.encrypt(&credentials.username)?;
        let encrypted_password = self.encryption.encrypt(&credentials.password)?;
        
        // Store encrypted credentials in metadata
        device.metadata.insert("encrypted_username".to_string(), encrypted_username);
        device.metadata.insert("encrypted_password".to_string(), encrypted_password);
        
        // Send device updated event
        let _ = self.event_sender.send(DiscoveryEvent::DeviceUpdated(device.clone()));
        
        Ok(())
    }
    
    /// Get device credentials
    pub fn get_device_credentials(&self, id: &DeviceId) -> Result<Option<DeviceCredentials>> {
        let registry = self.device_registry.read().unwrap();
        
        // Check if device exists
        let device = match registry.get(id) {
            Some(device) => device,
            None => return Err(Error::EntityNotFound(format!("Device with ID {} not found", id.as_str()))),
        };
        
        // Check if device has credentials
        let encrypted_username = match device.metadata.get("encrypted_username") {
            Some(username) => username,
            None => return Ok(None),
        };
        
        let encrypted_password = match device.metadata.get("encrypted_password") {
            Some(password) => password,
            None => return Ok(None),
        };
        
        // Decrypt credentials
        let username = self.encryption.decrypt(encrypted_username)?;
        let password = self.encryption.decrypt(encrypted_password)?;
        
        Ok(Some(DeviceCredentials { username, password }))
    }
    
    /// Test device connectivity
    pub async fn test_device(&self, id: &DeviceId) -> Result<bool> {
        let registry = self.device_registry.read().unwrap();
        
        // Check if device exists
        let device = match registry.get(id) {
            Some(device) => device,
            None => return Err(Error::EntityNotFound(format!("Device with ID {} not found", id.as_str()))),
        };
        
        // Test connectivity based on device type and protocols
        match &device.device_type {
            DeviceType::Camera(_) => {
                // Test camera connectivity
                let stream_protocols: Vec<_> = device.protocols.iter()
                    .filter_map(|p| {
                        if let NetworkProtocol::Stream(sp) = p {
                            Some(sp)
                        } else {
                            None
                        }
                    })
                    .collect();
                
                // Try to connect to the camera using available protocols
                for protocol in stream_protocols {
                    match protocol {
                        StreamProtocol::HTTP | StreamProtocol::HTTPS => {
                            // Try to connect using HTTP(S)
                            let scheme = if *protocol == StreamProtocol::HTTPS { "https" } else { "http" };
                            let url = format!("{}://{}:{}/", scheme, device.ip_address, device.port);
                            
                            match reqwest::Client::builder()
                                .timeout(Duration::from_secs(5))
                                .build() {
                                Ok(client) => {
                                    match client.get(&url).send().await {
                                        Ok(response) => {
                                            if response.status().is_success() {
                                                return Ok(true);
                                            }
                                        },
                                        Err(_) => {
                                            // Try next protocol
                                            continue;
                                        }
                                    }
                                },
                                Err(e) => {
                                    warn!("Failed to create HTTP client: {}", e);
                                    continue;
                                }
                            }
                        },
                        StreamProtocol::RTSP => {
                            // Try to connect using RTSP
                            let url = format!("rtsp://{}:{}/", device.ip_address, device.port);
                            
                            // Simple check if the port is open
                            match tokio::net::TcpStream::connect(format!("{}:{}", device.ip_address, device.port)).await {
                                Ok(_) => {
                                    // Port is open, assume RTSP is available
                                    return Ok(true);
                                },
                                Err(_) => {
                                    // Try next protocol
                                    continue;
                                }
                            }
                        },
                        _ => {
                            // Unsupported protocol for testing
                            continue;
                        }
                    }
                }
                
                // All protocols failed
                Ok(false)
            },
            DeviceType::NVR(_) => {
                // Test NVR connectivity similar to cameras
                let url = format!("http://{}:{}/", device.ip_address, device.port);
                
                match reqwest::Client::builder()
                    .timeout(Duration::from_secs(5))
                    .build() {
                    Ok(client) => {
                        match client.get(&url).send().await {
                            Ok(response) => Ok(response.status().is_success()),
                            Err(_) => Ok(false),
                        }
                    },
                    Err(_) => Ok(false),
                }
            },
            _ => {
                // Simple ping test for other device types
                match tokio::net::TcpStream::connect(format!("{}:{}", device.ip_address, device.port)).await {
                    Ok(_) => Ok(true),
                    Err(_) => Ok(false),
                }
            }
        }
    }
    
    /// Load devices from storage
    async fn load_devices(&self) -> Result<()> {
        // Load devices from storage
        let devices: Vec<DiscoveredDevice> = match self.storage.load("network_devices").await {
            Ok(Some(data)) => {
                match serde_json::from_slice(&data) {
                    Ok(devices) => devices,
                    Err(e) => {
                        warn!("Failed to deserialize devices: {}", e);
                        Vec::new()
                    }
                }
            },
            Ok(None) => {
                // No stored devices
                Vec::new()
            },
            Err(e) => {
                warn!("Failed to load devices from storage: {}", e);
                Vec::new()
            }
        };
        
        // Add devices to registry
        let mut registry = self.device_registry.write().unwrap();
        let now = Utc::now();
        let ttl = self.config.device_ttl.unwrap_or(DEFAULT_DEVICE_TTL);
        
        for mut device in devices {
            // Check if device is expired
            let age = now.signed_duration_since(device.last_seen);
            if age.num_seconds() > ttl as i64 {
                // Mark as offline but keep in registry
                device.online = false;
            }
            
            registry.insert(device.id.clone(), device);
        }
        
        info!("Loaded {} devices from storage", registry.len());
        Ok(())
    }
    
    /// Save devices to storage
    async fn save_devices(&self) -> Result<()> {
        // Get devices from registry
        let registry = self.device_registry.read().unwrap();
        let devices: Vec<DiscoveredDevice> = registry.values().cloned().collect();
        
        // Serialize devices
        let data = serde_json::to_vec(&devices)
            .map_err(|e| Error::SerializationError(format!("Failed to serialize devices: {}", e)))?;
        
        // Save to storage
        self.storage.store("network_devices", &data).await?;
        
        info!("Saved {} devices to storage", devices.len());
        Ok(())
    }
    
    /// Run a manual network scan
    pub async fn run_scan(&self, config: ScanConfig) -> Result<NetworkScan> {
        // Create channel for scan result
        let (tx, mut rx) = mpsc::channel(1);
        
        // Run scan in background
        let methods = self.methods.clone();
        let device_registry = self.device_registry.clone();
        let event_sender = self.event_sender.clone();
        
        tokio::spawn(async move {
            // Create scan record
            let scan_id = Uuid::new_v4();
            let scan = NetworkScan {
                id: scan_id,
                start_time: Utc::now(),
                end_time: None,
                devices_discovered: 0,
                new_devices: 0,
                updated_devices: 0,
                scanned_ranges: config.ip_ranges.clone(),
                protocols: config.protocols.clone(),
                status: ScanStatus::InProgress,
                error_message: None,
            };
            
            // Run discovery methods in parallel
            let mut tasks = Vec::new();
            
            for method in methods.iter() {
                let config_clone = config.clone();
                
                // Check if method supports any of the requested protocols
                let supported = config.protocols.iter().any(|protocol| method.supports_protocol(protocol));
                if !supported {
                    continue;
                }
                
                let method_name = method.name();
                let method_task = method.discover(&config_clone);
                
                tasks.push(async move {
                    match method_task.await {
                        Ok(devices) => {
                            info!("{} discovered {} devices", method_name, devices.len());
                            Ok(devices)
                        },
                        Err(e) => {
                            error!("{} discovery failed: {}", method_name, e);
                            Err(e)
                        }
                    }
                });
            }
            
            // Wait for all tasks to complete
            let results = join_all(tasks).await;
            
            // Process results
            let mut device_count = 0;
            let mut new_devices = 0;
            let mut updated_devices = 0;
            let mut success = true;
            let mut error_message = None;
            
            // Update device registry
            {
                let mut registry = device_registry.write().unwrap();
                
                for result in results {
                    match result {
                        Ok(devices) => {
                            device_count += devices.len();
                            
                            for device in devices {
                                // Check if device already exists
                                let existing = registry.get_mut(&device.id);
                                
                                if let Some(existing_device) = existing {
                                    // Update existing device
                                    let updated_device = device.clone();
                                    existing_device.update_from(&updated_device);
                                    updated_devices += 1;
                                    
                                    // Send device updated event
                                    let _ = event_sender.send(DiscoveryEvent::DeviceUpdated(existing_device.clone()));
                                } else {
                                    // Add new device
                                    new_devices += 1;
                                    registry.insert(device.id.clone(), device.clone());
                                    
                                    // Send device discovered event
                                    let _ = event_sender.send(DiscoveryEvent::DeviceDiscovered(device));
                                }
                            }
                        },
                        Err(e) => {
                            success = false;
                            error_message = Some(e.to_string());
                        }
                    }
                }
            }
            
            // Update scan record
            let mut scan = scan;
            scan.end_time = Some(Utc::now());
            scan.devices_discovered = device_count;
            scan.new_devices = new_devices;
            scan.updated_devices = updated_devices;
            scan.status = if success { ScanStatus::Completed } else { ScanStatus::Failed };
            scan.error_message = error_message;
            
            // Send scan completed event
            let event = if success {
                DiscoveryEvent::ScanCompleted(scan.clone())
            } else {
                DiscoveryEvent::ScanFailed(Error::DiscoveryError(
                    scan.error_message.clone().unwrap_or_else(|| "Unknown error".to_string())
                ))
            };
            
            let _ = event_sender.send(event);
            let _ = tx.send(scan).await;
        });
        
        // Wait for scan to complete
        match rx.recv().await {
            Some(scan) => Ok(scan),
            None => Err(Error::DiscoveryError("Scan task failed".to_string())),
        }
    }
}

/// Initialize the onvif module
/// 
/// This is a placeholder for the actual ONVIF client implementation
mod onvif {
    use std::error::Error as StdError;
    use std::fmt;
    
    /// ONVIF client error
    #[derive(Debug)]
    pub struct Error(String);
    
    impl fmt::Display for Error {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    
    impl StdError for Error {}
    
    /// ONVIF discovery message
    pub const WS_DISCOVERY_PROBE_MESSAGE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing">
  <s:Header>
    <a:Action s:mustUnderstand="1">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</a:Action>
    <a:MessageID>uuid:5504ce10-4a69-44ae-9974-9d1138ced156</a:MessageID>
    <a:ReplyTo>
      <a:Address>http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous</a:Address>
    </a:ReplyTo>
    <a:To s:mustUnderstand="1">urn:schemas-xmlsoap-org:ws:2005:04:discovery</a:To>
  </s:Header>
  <s:Body>
    <Probe xmlns="http://schemas.xmlsoap.org/ws/2005/04/discovery">
      <d:Types xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" xmlns:dp0="http://www.onvif.org/ver10/network/wsdl">dp0:NetworkVideoTransmitter</d:Types>
    </Probe>
  </s:Body>
</s:Envelope>"#;
    
    /// ONVIF device information
    #[derive(Debug, Clone)]
    pub struct DeviceInfo {
        pub manufacturer: String,
        pub model: String,
        pub firmware_version: String,
        pub serial_number: String,
        pub hardware_id: String,
        pub name: String,
    }
    
    /// ONVIF device capabilities
    #[derive(Debug, Clone)]
    pub struct DeviceCapabilities {
        pub ptz: bool,
        pub events: bool,
        pub imaging: bool,
        pub media: bool,
        pub analytics: bool,
    }
    
    /// ONVIF device profile
    #[derive(Debug, Clone)]
    pub struct DeviceProfile {
        pub token: String,
        pub name: String,
        pub video_source_token: String,
        pub video_encoder_token: String,
        pub ptz_token: Option<String>,
    }
    
    /// ONVIF client
    #[derive(Debug)]
    pub struct Client;
    
    impl Client {
        /// Create a new ONVIF client
        pub fn new() -> Result<Self, Error> {
            Ok(Self)
        }
        
        /// Connect to an ONVIF device
        pub async fn connect(&self, endpoint: &str) -> Result<DeviceInfo, Error> {
            // Simulate a connection to the device
            // In a real implementation, this would send a GetDeviceInformation request
            
            // Extract device details from the endpoint for simulation
            let parts: Vec<&str> = endpoint.split('/').collect();
            let mut manufacturer = "Generic";
            
            if endpoint.contains("axis") {
                manufacturer = "Axis";
            } else if endpoint.contains("hikvision") {
                manufacturer = "Hikvision";
            } else if endpoint.contains("dahua") {
                manufacturer = "Dahua";
            }
            
            Ok(DeviceInfo {
                manufacturer: manufacturer.to_string(),
                model: format!("Camera-{}", parts[2].split(':').next().unwrap_or("Unknown")),
                firmware_version: "1.0.0".to_string(),
                serial_number: format!("SN-{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap_or("Unknown")),
                hardware_id: format!("HW-{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap_or("Unknown")),
                name: format!("ONVIF Camera ({})", parts[2].split(':').next().unwrap_or("Unknown")),
            })
        }
        
        /// Get device capabilities
        pub async fn get_capabilities(&self) -> Result<DeviceCapabilities, Error> {
            // Simulate getting device capabilities
            Ok(DeviceCapabilities {
                ptz: true,
                events: true,
                imaging: true,
                media: true,
                analytics: false,
            })
        }
        
        /// Get device profiles
        pub async fn get_profiles(&self) -> Result<Vec<DeviceProfile>, Error> {
            // Simulate getting device profiles
            Ok(vec![
                DeviceProfile {
                    token: "profile_1".to_string(),
                    name: "Main Stream".to_string(),
                    video_source_token: "video_source_1".to_string(),
                    video_encoder_token: "video_encoder_1".to_string(),
                    ptz_token: Some("ptz_1".to_string()),
                },
                DeviceProfile {
                    token: "profile_2".to_string(),
                    name: "Sub Stream".to_string(),
                    video_source_token: "video_source_1".to_string(),
                    video_encoder_token: "video_encoder_2".to_string(),
                    ptz_token: Some("ptz_1".to_string()),
                },
            ])
        }
        
        /// Get stream URI for a profile
        pub async fn get_stream_uri(&self, profile_token: &str) -> Result<String, Error> {
            // Simulate getting stream URI
            Ok(format!("rtsp://username:password@camera.example.com:554/onvif/{}", profile_token))
        }
    }
}
