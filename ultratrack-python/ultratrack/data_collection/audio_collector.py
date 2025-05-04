"""
UltraTrack Audio Collection Module

This module provides robust audio collection capabilities for speaker recognition
and other audio-based tracking mechanisms:
- Connection to various audio sources (microphones, IP audio, telephony)
- Real-time processing and buffering of audio streams
- Voice activity detection (VAD) to isolate speech
- Noise reduction and audio enhancement
- Audio segmentation for optimal processing
- Stream synchronization with other data sources

Copyright (c) 2025 Your Organization
"""

import logging
import os
import threading
import time
import queue
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Set, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import socket
import enum
from concurrent.futures import ThreadPoolExecutor
import wave

# Local imports
from ultratrack.config import ConfigManager
from ultratrack.data_processing.preprocessing import AudioPreprocessor
from ultratrack.compliance.audit_logger import AuditLogger
from ultratrack.compliance.privacy_manager import PrivacyManager
from ultratrack.data_collection.data_validator import DataValidator, ValidationResult

# Set up module logger
logger = logging.getLogger(__name__)


class AudioSourceType(enum.Enum):
    """Types of audio sources supported by the system."""
    MICROPHONE = "microphone"
    IP_AUDIO = "ip_audio"
    TELEPHONY = "telephony"
    FILE = "file"
    STREAMING = "streaming_service"
    RADIO = "radio_receiver"
    VOIP = "voip"
    OTHER = "other"


class AudioFormat(enum.Enum):
    """Supported audio format types."""
    PCM = "pcm"
    WAV = "wav"
    MP3 = "mp3"
    AAC = "aac"
    OPUS = "opus"
    FLAC = "flac"
    G711A = "g711a"
    G711U = "g711u"


class AudioChannelLayout(enum.Enum):
    """Audio channel layouts."""
    MONO = "mono"
    STEREO = "stereo"
    SURROUND_5_1 = "5.1_surround"
    SURROUND_7_1 = "7.1_surround"
    BINAURAL = "binaural"


class ConnectionStatus(enum.Enum):
    """Status of audio source connections."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"
    RECOVERING = "recovering"
    DISABLED = "disabled"


@dataclass
class AudioQualityMetrics:
    """Metrics for assessing audio quality for speaker recognition."""
    snr_db: float = 0.0  # Signal-to-noise ratio in decibels
    clipping_percentage: float = 0.0  # Percentage of clipped samples
    dc_offset: float = 0.0  # DC offset level
    energy_level: float = 0.0  # Average energy level
    noise_level: float = 0.0  # Estimated noise level
    speech_presence_probability: float = 0.0  # Probability that speech is present
    reverberation_level: float = 0.0  # Estimated level of reverberation
    distortion_level: float = 0.0  # Estimated level of distortion
    timestamp: datetime = field(default_factory=datetime.now)  # When metrics were computed


@dataclass
class AudioSourceMetadata:
    """Metadata associated with an audio source."""
    source_id: str
    source_type: AudioSourceType
    location: Optional[str] = None  # Physical/geographic location
    description: Optional[str] = None
    owner: Optional[str] = None
    sample_rate: int = 16000  # Hz
    bit_depth: int = 16  # bits
    channels: int = 1
    format: AudioFormat = AudioFormat.PCM
    channel_layout: AudioChannelLayout = AudioChannelLayout.MONO
    encryption_enabled: bool = False
    encryption_type: Optional[str] = None
    last_maintenance: Optional[datetime] = None
    installation_date: Optional[datetime] = None
    firmware_version: Optional[str] = None
    hardware_model: Optional[str] = None
    additional_properties: Dict[str, Any] = field(default_factory=dict)
    legal_authorization: Optional[str] = None  # Reference to legal authorization document


@dataclass
class AudioFrame:
    """Single frame of audio data with associated metadata."""
    data: np.ndarray  # Actual audio samples
    source_id: str  # ID of the source that produced this frame
    timestamp: datetime  # When this frame was captured
    frame_number: int  # Sequential frame number in the stream
    duration_ms: int  # Duration of the frame in milliseconds
    sample_rate: int  # Sampling rate in Hz
    channels: int  # Number of audio channels
    bit_depth: int  # Bit depth
    speech_detected: bool = False  # Whether speech was detected in this frame
    quality_metrics: Optional[AudioQualityMetrics] = None  # Quality metrics if computed
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class AudioSegment:
    """Segment of audio data representing a continuous speech or sound event."""
    frames: List[AudioFrame]  # Sequence of frames in this segment
    segment_id: str  # Unique identifier for this segment
    source_id: str  # ID of the source that produced this segment
    start_time: datetime  # When this segment starts
    end_time: datetime  # When this segment ends
    duration_ms: int  # Duration of the segment in milliseconds
    speech_confidence: float = 0.0  # Confidence that segment contains speech (0-1)
    speaker_count_estimate: int = 1  # Estimated number of speakers
    is_complete: bool = False  # Whether this segment is complete or still being built
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


class AudioProcessingError(Exception):
    """Exception raised for errors in audio processing."""
    pass


class AudioSourceConnectionError(Exception):
    """Exception raised for errors connecting to audio sources."""
    pass


class AudioSource(ABC):
    """Abstract base class for all audio sources."""
    
    def __init__(self, source_id: str, config: Dict[str, Any]):
        """
        Initialize the audio source.
        
        Args:
            source_id: Unique identifier for this source
            config: Configuration for this source
        """
        self.source_id = source_id
        self.config = config
        self.status = ConnectionStatus.DISCONNECTED
        self.error_message = None
        self.last_connection_attempt = None
        self.last_successful_connection = None
        self.connection_attempts = 0
        self.metadata = AudioSourceMetadata(
            source_id=source_id,
            source_type=self._get_source_type(),
            sample_rate=config.get("sample_rate", 16000),
            channels=config.get("channels", 1),
            bit_depth=config.get("bit_depth", 16),
            format=AudioFormat(config.get("format", "pcm")),
            channel_layout=AudioChannelLayout(config.get("channel_layout", "mono")),
            location=config.get("location"),
            description=config.get("description"),
            owner=config.get("owner"),
            encryption_enabled=config.get("encryption_enabled", False),
            encryption_type=config.get("encryption_type"),
            legal_authorization=config.get("legal_authorization")
        )
        self._lock = threading.RLock()
        self._frame_count = 0
        logger.debug(f"Initialized audio source {source_id} of type {self._get_source_type().value}")
    
    @abstractmethod
    def _get_source_type(self) -> AudioSourceType:
        """Get the type of this audio source."""
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the audio source.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the audio source.
        
        Returns:
            bool: True if disconnection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def read_frame(self) -> Optional[AudioFrame]:
        """
        Read a single frame of audio data from the source.
        
        Returns:
            AudioFrame or None: The audio frame if available, None otherwise
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the audio source is available.
        
        Returns:
            bool: True if the source is available, False otherwise
        """
        pass
    
    def get_status(self) -> Tuple[ConnectionStatus, Optional[str]]:
        """
        Get the current status of the audio source.
        
        Returns:
            Tuple of ConnectionStatus and optional error message
        """
        with self._lock:
            return self.status, self.error_message
    
    def set_status(self, status: ConnectionStatus, error_message: Optional[str] = None):
        """
        Set the current status of the audio source.
        
        Args:
            status: New status
            error_message: Optional error message
        """
        with self._lock:
            self.status = status
            self.error_message = error_message
            if status == ConnectionStatus.CONNECTED:
                self.last_successful_connection = datetime.now()
            elif status == ConnectionStatus.CONNECTING:
                self.last_connection_attempt = datetime.now()
                self.connection_attempts += 1
    
    def get_metadata(self) -> AudioSourceMetadata:
        """
        Get metadata for this audio source.
        
        Returns:
            AudioSourceMetadata: Metadata for this source
        """
        with self._lock:
            return self.metadata
    
    def update_metadata(self, updates: Dict[str, Any]):
        """
        Update metadata for this audio source.
        
        Args:
            updates: Dictionary of metadata fields to update
        """
        with self._lock:
            for key, value in updates.items():
                if hasattr(self.metadata, key):
                    setattr(self.metadata, key, value)
                else:
                    self.metadata.additional_properties[key] = value


class MicrophoneAudioSource(AudioSource):
    """Audio source for local microphones."""
    
    def __init__(self, source_id: str, config: Dict[str, Any]):
        """
        Initialize a microphone audio source.
        
        Args:
            source_id: Unique identifier for this source
            config: Configuration for this source
        """
        super().__init__(source_id, config)
        self.device_index = config.get("device_index", 0)
        self._stream = None
        self._pyaudio = None
        
        # Import PyAudio here to avoid requiring it as a dependency
        # if not using microphone sources
        try:
            import pyaudio
            self._pyaudio_module = pyaudio
            self._import_error = None
        except ImportError as e:
            self._import_error = str(e)
            logger.warning(f"PyAudio not available for MicrophoneAudioSource {source_id}: {e}")
    
    def _get_source_type(self) -> AudioSourceType:
        return AudioSourceType.MICROPHONE
    
    def connect(self) -> bool:
        """Connect to the microphone."""
        with self._lock:
            if self._import_error:
                self.set_status(ConnectionStatus.ERROR, f"PyAudio not available: {self._import_error}")
                return False
            
            try:
                self.set_status(ConnectionStatus.CONNECTING)
                
                if self._stream and self._stream.is_active():
                    logger.debug(f"Microphone {self.source_id} already connected")
                    self.set_status(ConnectionStatus.CONNECTED)
                    return True
                
                if not self._pyaudio:
                    self._pyaudio = self._pyaudio_module.PyAudio()
                
                sample_rate = self.metadata.sample_rate
                channels = self.metadata.channels
                format_mapping = {
                    8: self._pyaudio_module.paInt8,
                    16: self._pyaudio_module.paInt16,
                    24: self._pyaudio_module.paInt24,
                    32: self._pyaudio_module.paInt32,
                    -32: self._pyaudio_module.paFloat32
                }
                audio_format = format_mapping.get(self.metadata.bit_depth, self._pyaudio_module.paInt16)
                
                self._stream = self._pyaudio.open(
                    format=audio_format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=int(sample_rate * self.config.get("buffer_duration_ms", 100) / 1000)
                )
                
                if self._stream.is_active():
                    self.set_status(ConnectionStatus.CONNECTED)
                    logger.info(f"Connected to microphone {self.source_id} (device index {self.device_index})")
                    return True
                else:
                    self.set_status(ConnectionStatus.ERROR, "Failed to activate microphone stream")
                    return False
                
            except Exception as e:
                self.set_status(ConnectionStatus.ERROR, str(e))
                logger.error(f"Error connecting to microphone {self.source_id}: {e}")
                return False
    
    def disconnect(self) -> bool:
        """Disconnect from the microphone."""
        with self._lock:
            try:
                if self._stream:
                    self._stream.stop_stream()
                    self._stream.close()
                    self._stream = None
                
                self.set_status(ConnectionStatus.DISCONNECTED)
                logger.info(f"Disconnected from microphone {self.source_id}")
                return True
                
            except Exception as e:
                self.set_status(ConnectionStatus.ERROR, str(e))
                logger.error(f"Error disconnecting from microphone {self.source_id}: {e}")
                return False
    
    def read_frame(self) -> Optional[AudioFrame]:
        """Read a frame of audio data from the microphone."""
        with self._lock:
            if not self._stream or not self._stream.is_active():
                return None
            
            try:
                sample_rate = self.metadata.sample_rate
                channels = self.metadata.channels
                frame_duration_ms = self.config.get("frame_duration_ms", 30)
                frame_size = int(sample_rate * frame_duration_ms / 1000)
                
                raw_data = self._stream.read(frame_size, exception_on_overflow=False)
                
                if not raw_data:
                    return None
                
                # Convert to numpy array
                if self.metadata.bit_depth == -32:  # Float32
                    audio_data = np.frombuffer(raw_data, dtype=np.float32)
                else:
                    audio_data = np.frombuffer(raw_data, dtype=np.int16)
                
                # Reshape for multi-channel audio
                if channels > 1:
                    audio_data = audio_data.reshape(-1, channels)
                
                self._frame_count += 1
                
                return AudioFrame(
                    data=audio_data,
                    source_id=self.source_id,
                    timestamp=datetime.now(),
                    frame_number=self._frame_count,
                    duration_ms=frame_duration_ms,
                    sample_rate=sample_rate,
                    channels=channels,
                    bit_depth=self.metadata.bit_depth
                )
                
            except Exception as e:
                logger.error(f"Error reading from microphone {self.source_id}: {e}")
                return None
    
    def is_available(self) -> bool:
        """Check if the microphone is available."""
        if self._import_error:
            return False
        
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            device_count = pa.get_device_count()
            pa.terminate()
            return self.device_index < device_count
        except Exception as e:
            logger.error(f"Error checking microphone availability: {e}")
            return False


class IPAudioSource(AudioSource):
    """Audio source for IP-based audio streams (e.g. VoIP, IP cameras with audio)."""
    
    def __init__(self, source_id: str, config: Dict[str, Any]):
        """
        Initialize an IP audio source.
        
        Args:
            source_id: Unique identifier for this source
            config: Configuration for this source
        """
        super().__init__(source_id, config)
        self.url = config.get("url")
        self.username = config.get("username")
        self.password = config.get("password")
        self.protocol = config.get("protocol", "rtsp")
        self._stream = None
        self._ffmpeg_process = None
        self._buffer = queue.Queue(maxsize=100)
        self._running = False
        self._thread = None
    
    def _get_source_type(self) -> AudioSourceType:
        return AudioSourceType.IP_AUDIO
    
    def connect(self) -> bool:
        """Connect to the IP audio source."""
        with self._lock:
            if self._running:
                logger.debug(f"IP audio source {self.source_id} already connected")
                return True
            
            try:
                self.set_status(ConnectionStatus.CONNECTING)
                
                # Import ffmpeg-python here to avoid requiring it as a dependency
                # if not using IP audio sources
                try:
                    import ffmpeg
                except ImportError:
                    self.set_status(ConnectionStatus.ERROR, "ffmpeg-python not available")
                    logger.error(f"ffmpeg-python not available for IPAudioSource {self.source_id}")
                    return False
                
                # Construct the full URL with authentication if needed
                full_url = self.url
                if self.username and self.password and '://' in self.url:
                    protocol, rest = self.url.split('://', 1)
                    full_url = f"{protocol}://{self.username}:{self.password}@{rest}"
                
                # Configure ffmpeg process
                sample_rate = self.metadata.sample_rate
                channels = self.metadata.channels
                
                if self.metadata.bit_depth == 8:
                    output_format = "u8"
                elif self.metadata.bit_depth == 16:
                    output_format = "s16le"
                elif self.metadata.bit_depth == 24:
                    output_format = "s24le"
                elif self.metadata.bit_depth == 32:
                    output_format = "s32le"
                elif self.metadata.bit_depth == -32:
                    output_format = "f32le"
                else:
                    output_format = "s16le"
                
                # Build the ffmpeg command
                process = (
                    ffmpeg
                    .input(full_url, rtsp_transport=self.config.get("rtsp_transport", "tcp"))
                    .output('pipe:', format='wav', acodec='pcm_' + output_format, ar=sample_rate, ac=channels)
                    .global_args('-loglevel', 'quiet', '-nostdin')
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )
                
                self._ffmpeg_process = process
                self._running = True
                
                # Start a thread to read from the process
                self._thread = threading.Thread(
                    target=self._read_stream_thread,
                    daemon=True,
                    name=f"IPAudio-{self.source_id}"
                )
                self._thread.start()
                
                # Wait a short time to see if the thread encounters an error
                time.sleep(0.5)
                
                if self._running:
                    self.set_status(ConnectionStatus.CONNECTED)
                    logger.info(f"Connected to IP audio source {self.source_id} at {self.url}")
                    return True
                else:
                    self.set_status(ConnectionStatus.ERROR, "Failed to start stream reading thread")
                    return False
                
            except Exception as e:
                self.set_status(ConnectionStatus.ERROR, str(e))
                logger.error(f"Error connecting to IP audio source {self.source_id}: {e}")
                return False
    
    def _read_stream_thread(self):
        """Thread function to read from the ffmpeg process."""
        try:
            # Skip the WAV header (44 bytes)
            self._ffmpeg_process.stdout.read(44)
            
            sample_rate = self.metadata.sample_rate
            channels = self.metadata.channels
            frame_duration_ms = self.config.get("frame_duration_ms", 30)
            frame_size_bytes = int(sample_rate * frame_duration_ms / 1000) * channels * (abs(self.metadata.bit_depth) // 8)
            
            while self._running:
                data = self._ffmpeg_process.stdout.read(frame_size_bytes)
                
                if not data:
                    logger.warning(f"End of stream for IP audio source {self.source_id}")
                    self._running = False
                    break
                
                # Convert to numpy array
                if self.metadata.bit_depth == 8:
                    audio_data = np.frombuffer(data, dtype=np.uint8)
                    # Convert to float in [-1, 1] range
                    audio_data = (audio_data.astype(np.float32) - 128) / 128.0
                elif self.metadata.bit_depth == 16:
                    audio_data = np.frombuffer(data, dtype=np.int16)
                elif self.metadata.bit_depth == 24:
                    # Handle 24-bit audio (not directly supported by numpy)
                    audio_data = np.zeros(len(data) // 3, dtype=np.int32)
                    for i in range(len(data) // 3):
                        audio_data[i] = int.from_bytes(data[i*3:i*3+3], byteorder='little', signed=True)
                elif self.metadata.bit_depth == 32:
                    audio_data = np.frombuffer(data, dtype=np.int32)
                elif self.metadata.bit_depth == -32:
                    audio_data = np.frombuffer(data, dtype=np.float32)
                else:
                    # Default to 16-bit
                    audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Reshape for multi-channel audio
                if channels > 1:
                    audio_data = audio_data.reshape(-1, channels)
                
                self._frame_count += 1
                
                frame = AudioFrame(
                    data=audio_data,
                    source_id=self.source_id,
                    timestamp=datetime.now(),
                    frame_number=self._frame_count,
                    duration_ms=frame_duration_ms,
                    sample_rate=sample_rate,
                    channels=channels,
                    bit_depth=self.metadata.bit_depth
                )
                
                try:
                    # Add to buffer with timeout to avoid blocking indefinitely
                    self._buffer.put(frame, timeout=1)
                except queue.Full:
                    # If buffer is full, remove oldest frame
                    try:
                        self._buffer.get_nowait()
                        self._buffer.put_nowait(frame)
                    except (queue.Empty, queue.Full):
                        pass
        
        except Exception as e:
            logger.error(f"Error in stream reading thread for {self.source_id}: {e}")
            self.set_status(ConnectionStatus.ERROR, str(e))
            self._running = False
    
    def disconnect(self) -> bool:
        """Disconnect from the IP audio source."""
        with self._lock:
            try:
                self._running = False
                
                if self._ffmpeg_process:
                    self._ffmpeg_process.kill()
                    self._ffmpeg_process = None
                
                if self._thread and self._thread.is_alive():
                    self._thread.join(timeout=2)
                    self._thread = None
                
                # Clear buffer
                while not self._buffer.empty():
                    try:
                        self._buffer.get_nowait()
                    except queue.Empty:
                        break
                
                self.set_status(ConnectionStatus.DISCONNECTED)
                logger.info(f"Disconnected from IP audio source {self.source_id}")
                return True
                
            except Exception as e:
                self.set_status(ConnectionStatus.ERROR, str(e))
                logger.error(f"Error disconnecting from IP audio source {self.source_id}: {e}")
                return False
    
    def read_frame(self) -> Optional[AudioFrame]:
        """Read a frame of audio data from the IP audio source."""
        if not self._running:
            return None
        
        try:
            return self._buffer.get(block=False)
        except queue.Empty:
            return None
    
    def is_available(self) -> bool:
        """Check if the IP audio source is available."""
        try:
            # Try to connect to the host to check availability
            if '://' in self.url:
                host = self.url.split('://', 1)[1].split('/', 1)[0].split(':', 1)[0]
                port = 554  # Default RTSP port
                
                if ':' in host:
                    host, port_str = host.split(':', 1)
                    port = int(port_str)
                
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)
                s.connect((host, port))
                s.close()
                return True
        except Exception:
            return False
        
        return False


class FileAudioSource(AudioSource):
    """Audio source for reading from audio files."""
    
    def __init__(self, source_id: str, config: Dict[str, Any]):
        """
        Initialize a file audio source.
        
        Args:
            source_id: Unique identifier for this source
            config: Configuration for this source
        """
        super().__init__(source_id, config)
        self.file_path = config.get("file_path")
        self._wave_file = None
        self._frame_size = 0
        self._running = False
        self._thread = None
        self._buffer = queue.Queue(maxsize=100)
        self._loop_playback = config.get("loop", False)
        self._playback_speed = config.get("playback_speed", 1.0)
        self._current_position = 0
    
    def _get_source_type(self) -> AudioSourceType:
        return AudioSourceType.FILE
    
    def connect(self) -> bool:
        """Connect to the audio file."""
        with self._lock:
            if self._running:
                logger.debug(f"File audio source {self.source_id} already connected")
                return True
            
            try:
                self.set_status(ConnectionStatus.CONNECTING)
                
                if not self.file_path or not os.path.exists(self.file_path):
                    self.set_status(ConnectionStatus.ERROR, f"File not found: {self.file_path}")
                    return False
                
                # Open the wave file
                self._wave_file = wave.open(self.file_path, 'rb')
                
                # Update metadata from the file
                channels = self._wave_file.getnchannels()
                sample_rate = self._wave_file.getframerate()
                bit_depth = self._wave_file.getsampwidth() * 8
                
                self.metadata.channels = channels
                self.metadata.sample_rate = sample_rate
                self.metadata.bit_depth = bit_depth
                
                if channels == 1:
                    self.metadata.channel_layout = AudioChannelLayout.MONO
                elif channels == 2:
                    self.metadata.channel_layout = AudioChannelLayout.STEREO
                elif channels == 6:
                    self.metadata.channel_layout = AudioChannelLayout.SURROUND_5_1
                elif channels == 8:
                    self.metadata.channel_layout = AudioChannelLayout.SURROUND_7_1
                
                # Calculate frame size
                frame_duration_ms = self.config.get("frame_duration_ms", 30)
                self._frame_size = int(sample_rate * frame_duration_ms / 1000)
                
                # Start the reading thread
                self._running = True
                self._thread = threading.Thread(
                    target=self._read_file_thread,
                    daemon=True,
                    name=f"FileAudio-{self.source_id}"
                )
                self._thread.start()
                
                self.set_status(ConnectionStatus.CONNECTED)
                logger.info(f"Connected to audio file {self.source_id} at {self.file_path}")
                return True
                
            except Exception as e:
                self.set_status(ConnectionStatus.ERROR, str(e))
                logger.error(f"Error connecting to audio file {self.source_id}: {e}")
                return False
    
    def _read_file_thread(self):
        """Thread function to read from the audio file."""
        try:
            frame_duration_ms = self.config.get("frame_duration_ms", 30)
            sample_rate = self.metadata.sample_rate
            channels = self.metadata.channels
            bit_depth = self.metadata.bit_depth
            bytes_per_sample = bit_depth // 8
            
            # Calculate sleep time based on playback speed
            sleep_time = frame_duration_ms / 1000.0 / self._playback_speed
            
            while self._running:
                # Read a frame
                raw_data = self._wave_file.readframes(self._frame_size)
                
                # If we reached the end of the file
                if not raw_data or len(raw_data) < self._frame_size * bytes_per_sample * channels:
                    if self._loop_playback:
                        # Reset to beginning of file
                        self._wave_file.rewind()
                        self._current_position = 0
                        continue
                    else:
                        logger.info(f"End of file for audio source {self.source_id}")
                        self._running = False
                        break
                
                # Convert to numpy array
                if bit_depth == 8:
                    audio_data = np.frombuffer(raw_data, dtype=np.uint8)
                    # Convert to float in [-1, 1] range
                    audio_data = (audio_data.astype(np.float32) - 128) / 128.0
                elif bit_depth == 16:
                    audio_data = np.frombuffer(raw_data, dtype=np.int16)
                elif bit_depth == 24:
                    # Handle 24-bit audio (not directly supported by numpy)
                    audio_data = np.zeros(len(raw_data) // 3, dtype=np.int32)
                    for i in range(len(raw_data) // 3):
                        audio_data[i] = int.from_bytes(raw_data[i*3:i*3+3], byteorder='little', signed=True)
                elif bit_depth == 32:
                    audio_data = np.frombuffer(raw_data, dtype=np.int32)
                else:
                    # Default to 16-bit
                    audio_data = np.frombuffer(raw_data, dtype=np.int16)
                
                # Reshape for multi-channel audio
                if channels > 1:
                    audio_data = audio_data.reshape(-1, channels)
                
                self._current_position += len(audio_data)
                self._frame_count += 1
                
                frame = AudioFrame(
                    data=audio_data,
                    source_id=self.source_id,
                    timestamp=datetime.now(),
                    frame_number=self._frame_count,
                    duration_ms=frame_duration_ms,
                    sample_rate=sample_rate,
                    channels=channels,
                    bit_depth=bit_depth
                )
                
                try:
                    # Add to buffer with timeout to avoid blocking indefinitely
                    self._buffer.put(frame, timeout=1)
                except queue.Full:
                    # If buffer is full, remove oldest frame
                    try:
                        self._buffer.get_nowait()
                        self._buffer.put_nowait(frame)
                    except (queue.Empty, queue.Full):
                        pass
                
                # Sleep to simulate real-time playback
                time.sleep(sleep_time)
        
        except Exception as e:
            logger.error(f"Error in file reading thread for {self.source_id}: {e}")
            self.set_status(ConnectionStatus.ERROR, str(e))
            self._running = False
    
    def disconnect(self) -> bool:
        """Disconnect from the audio file."""
        with self._lock:
            try:
                self._running = False
                
                if self._thread and self._thread.is_alive():
                    self._thread.join(timeout=2)
                    self._thread = None
                
                if self._wave_file:
                    self._wave_file.close()
                    self._wave_file = None
                
                # Clear buffer
                while not self._buffer.empty():
                    try:
                        self._buffer.get_nowait()
                    except queue.Empty:
                        break
                
                self.set_status(ConnectionStatus.DISCONNECTED)
                logger.info(f"Disconnected from audio file {self.source_id}")
                return True
                
            except Exception as e:
                self.set_status(ConnectionStatus.ERROR, str(e))
                logger.error(f"Error disconnecting from audio file {self.source_id}: {e}")
                return False
    
    def read_frame(self) -> Optional[AudioFrame]:
        """Read a frame of audio data from the file."""
        if not self._running:
            return None
        
        try:
            return self._buffer.get(block=False)
        except queue.Empty:
            return None
    
    def is_available(self) -> bool:
        """Check if the audio file is available."""
        return os.path.exists(self.file_path) if self.file_path else False


class VoiceActivityDetector:
    """Detects speech/voice activity in audio frames."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the voice activity detector.
        
        Args:
            config: Configuration for the detector
        """
        self.enabled = config.get("vad_enabled", True)
        self.threshold = config.get("vad_threshold", 0.5)
        self.min_speech_duration_ms = config.get("min_speech_duration_ms", 100)
        self.frame_duration_ms = config.get("frame_duration_ms", 30)
        self.frames_per_min_speech = self.min_speech_duration_ms // self.frame_duration_ms
        self.history_buffer = []
        self.history_buffer_size = 10  # Number of frames to keep for context
        
        # Initialize with optional external VAD model
        self.use_external_model = config.get("use_external_vad_model", False)
        self.external_model = None
        
        if self.use_external_model:
            try:
                # This would be replaced with actual model loading code
                # For example, loading a TensorFlow or PyTorch model
                logger.info("Initializing external VAD model")
                # self.external_model = load_model(config.get("vad_model_path"))
            except Exception as e:
                logger.error(f"Failed to load external VAD model: {e}")
                self.use_external_model = False
    
    def detect(self, frame: AudioFrame) -> bool:
        """
        Detect voice activity in an audio frame.
        
        Args:
            frame: Audio frame to process
            
        Returns:
            bool: True if voice activity is detected, False otherwise
        """
        if not self.enabled:
            return True  # If VAD is disabled, assume everything is speech
        
        # Use external model if available
        if self.use_external_model and self.external_model:
            # This would be replaced with actual model inference code
            # return self.external_model.predict(frame.data) > self.threshold
            pass
        
        # Simple energy-based VAD
        audio_data = frame.data.astype(np.float32)
        
        # Normalize based on bit depth
        if frame.bit_depth == 16:
            audio_data = audio_data / 32768.0
        elif frame.bit_depth == 24:
            audio_data = audio_data / 8388608.0
        elif frame.bit_depth == 32:
            audio_data = audio_data / 2147483648.0
        
        # Calculate energy
        energy = np.mean(audio_data ** 2)
        
        # Simple noise estimation using the history buffer
        self.history_buffer.append(energy)
        if len(self.history_buffer) > self.history_buffer_size:
            self.history_buffer.pop(0)
        
        # Sort energies and take the lower half as noise estimate
        sorted_energies = sorted(self.history_buffer)
        noise_estimate = np.mean(sorted_energies[:len(sorted_energies)//2]) if sorted_energies else 0
        
        # Calculate SNR
        snr = energy / (noise_estimate + 1e-10)
        
        # Calculate speech probability
        speech_probability = min(1.0, max(0.0, (snr - 1.0) / 10.0))
        
        # Store speech probability in frame metadata
        frame.metadata["speech_probability"] = speech_probability
        
        # Set speech_detected flag
        speech_detected = speech_probability > self.threshold
        frame.speech_detected = speech_detected
        
        return speech_detected
    
    def get_segment_speech_confidence(self, segment: AudioSegment) -> float:
        """
        Calculate overall speech confidence for a segment.
        
        Args:
            segment: Audio segment to analyze
            
        Returns:
            float: Speech confidence value between 0 and 1
        """
        if not segment.frames:
            return 0.0
        
        # Get speech probabilities for all frames
        speech_probs = [
            frame.metadata.get("speech_probability", 0.0)
            for frame in segment.frames
        ]
        
        # Calculate average speech probability
        return sum(speech_probs) / len(speech_probs)


class NoiseReduction:
    """Reduces noise in audio frames."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the noise reduction module.
        
        Args:
            config: Configuration for noise reduction
        """
        self.enabled = config.get("noise_reduction_enabled", True)
        self.method = config.get("noise_reduction_method", "spectral_subtraction")
        self.noise_floor_db = config.get("noise_floor_db", -60)
        self.smoothing_factor = config.get("smoothing_factor", 0.7)
        
        # Noise profile estimation
        self.noise_profile = None
        self.noise_estimate_frames = config.get("noise_estimate_frames", 10)
        self.frame_counter = 0
        self.in_calibration = True
        
        # FFT parameters
        self.fft_size = config.get("fft_size", 2048)
        self.hop_size = config.get("hop_size", 512)
        
        # Initialize with optional external noise reduction model
        self.use_external_model = config.get("use_external_nr_model", False)
        self.external_model = None
        
        if self.use_external_model:
            try:
                # This would be replaced with actual model loading code
                logger.info("Initializing external noise reduction model")
                # self.external_model = load_model(config.get("nr_model_path"))
            except Exception as e:
                logger.error(f"Failed to load external noise reduction model: {e}")
                self.use_external_model = False
    
    def process(self, frame: AudioFrame) -> AudioFrame:
        """
        Apply noise reduction to an audio frame.
        
        Args:
            frame: Audio frame to process
            
        Returns:
            AudioFrame: Processed audio frame
        """
        if not self.enabled:
            return frame
        
        # Use external model if available
        if self.use_external_model and self.external_model:
            # This would be replaced with actual model inference code
            # return self.external_model.process(frame)
            return frame
        
        # Convert to float for processing
        audio_data = frame.data.astype(np.float32)
        
        # Normalize based on bit depth
        if frame.bit_depth == 16:
            audio_data = audio_data / 32768.0
        elif frame.bit_depth == 24:
            audio_data = audio_data / 8388608.0
        elif frame.bit_depth == 32:
            audio_data = audio_data / 2147483648.0
        
        # For multi-channel audio, process each channel separately
        if len(audio_data.shape) > 1:
            processed_channels = []
            for channel in range(frame.channels):
                processed_channel = self._process_channel(audio_data[:, channel])
                processed_channels.append(processed_channel)
            processed_audio = np.column_stack(processed_channels)
        else:
            processed_audio = self._process_channel(audio_data)
        
        # Convert back to original data type
        if frame.bit_depth == 16:
            processed_audio = (processed_audio * 32768.0).astype(np.int16)
        elif frame.bit_depth == 24 or frame.bit_depth == 32:
            processed_audio = (processed_audio * 2147483648.0).astype(np.int32)
        
        # Create new frame with processed audio
        processed_frame = AudioFrame(
            data=processed_audio,
            source_id=frame.source_id,
            timestamp=frame.timestamp,
            frame_number=frame.frame_number,
            duration_ms=frame.duration_ms,
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            bit_depth=frame.bit_depth,
            speech_detected=frame.speech_detected,
            quality_metrics=frame.quality_metrics,
            metadata=frame.metadata.copy()
        )
        
        # Add noise reduction info to metadata
        processed_frame.metadata["noise_reduction_applied"] = True
        processed_frame.metadata["noise_reduction_method"] = self.method
        
        return processed_frame
    
    def _process_channel(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Process a single audio channel.
        
        Args:
            audio_data: Audio data for one channel
            
        Returns:
            np.ndarray: Processed audio data
        """
        # If we're still calibrating, update noise profile
        if self.in_calibration:
            self._update_noise_profile(audio_data)
            self.frame_counter += 1
            
            if self.frame_counter >= self.noise_estimate_frames:
                self.in_calibration = False
            
            return audio_data  # Return unmodified during calibration
        
        # Apply spectral subtraction for noise reduction
        if self.method == "spectral_subtraction" and self.noise_profile is not None:
            # Split into overlapping frames
            frames = []
            for i in range(0, len(audio_data) - self.fft_size + self.hop_size, self.hop_size):
                if i + self.fft_size <= len(audio_data):
                    frames.append(audio_data[i:i+self.fft_size])
            
            processed_frames = []
            
            for frame_data in frames:
                # Apply window function
                windowed = frame_data * np.hanning(self.fft_size)
                
                # Compute FFT
                spectrum = np.fft.rfft(windowed)
                magnitude = np.abs(spectrum)
                phase = np.angle(spectrum)
                
                # Apply spectral subtraction
                subtracted = np.maximum(
                    magnitude - self.smoothing_factor * self.noise_profile,
                    10 ** (self.noise_floor_db / 20) * magnitude
                )
                
                # Reconstruct signal
                processed_spectrum = subtracted * np.exp(1j * phase)
                processed_frame = np.fft.irfft(processed_spectrum)
                
                # Truncate to original size (in case of rounding issues)
                processed_frame = processed_frame[:self.fft_size]
                
                processed_frames.append(processed_frame)
            
            # Overlap-add to reconstruct the signal
            result = np.zeros(len(audio_data))
            for i, frame_data in enumerate(processed_frames):
                start = i * self.hop_size
                end = min(start + self.fft_size, len(result))
                result[start:end] += frame_data[:end-start] * np.hanning(self.fft_size)[:end-start]
            
            # Normalize to avoid clipping
            result = np.clip(result, -1.0, 1.0)
            
            return result
        
        return audio_data
    
    def _update_noise_profile(self, audio_data: np.ndarray):
        """
        Update the noise profile from audio data.
        
        Args:
            audio_data: Audio data to use for noise profile update
        """
        # Split into overlapping frames
        frames = []
        for i in range(0, len(audio_data) - self.fft_size + self.hop_size, self.hop_size):
            if i + self.fft_size <= len(audio_data):
                frames.append(audio_data[i:i+self.fft_size])
        
        if not frames:
            return
        
        # Compute average magnitude spectrum
        magnitude_spectra = []
        
        for frame_data in frames:
            # Apply window function
            windowed = frame_data * np.hanning(self.fft_size)
            
            # Compute FFT
            spectrum = np.fft.rfft(windowed)
            magnitude = np.abs(spectrum)
            magnitude_spectra.append(magnitude)
        
        avg_magnitude = np.mean(magnitude_spectra, axis=0)
        
        # Update noise profile
        if self.noise_profile is None:
            self.noise_profile = avg_magnitude
        else:
            self.noise_profile = (self.noise_profile * (self.frame_counter) + avg_magnitude) / (self.frame_counter + 1)


class AudioQualityAnalyzer:
    """Analyzes audio quality for better recognition performance."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the audio quality analyzer.
        
        Args:
            config: Configuration for the analyzer
        """
        self.enabled = config.get("quality_analysis_enabled", True)
        self.min_snr_db = config.get("min_snr_db", 10.0)
        self.max_clipping_percentage = config.get("max_clipping_percentage", 0.05)
        self.analysis_window_ms = config.get("analysis_window_ms", 500)
    
    def analyze(self, frame: AudioFrame) -> AudioQualityMetrics:
        """
        Analyze audio quality of a frame.
        
        Args:
            frame: Audio frame to analyze
            
        Returns:
            AudioQualityMetrics: Quality metrics for the frame
        """
        if not self.enabled:
            return AudioQualityMetrics()
        
        # Convert to float for analysis
        audio_data = frame.data.astype(np.float32)
        
        # Normalize based on bit depth
        if frame.bit_depth == 16:
            audio_data = audio_data / 32768.0
        elif frame.bit_depth == 24:
            audio_data = audio_data / 8388608.0
        elif frame.bit_depth == 32:
            audio_data = audio_data / 2147483648.0
        
        # For multi-channel audio, average all channels for analysis
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Calculate metrics
        energy_level = np.mean(audio_data ** 2)
        
        # Estimate noise level from quietest sections
        sorted_energies = []
        frame_size = min(int(frame.sample_rate * 0.02), len(audio_data))  # 20ms frames
        
        for i in range(0, len(audio_data) - frame_size, frame_size):
            frame_energy = np.mean(audio_data[i:i+frame_size] ** 2)
            sorted_energies.append(frame_energy)
        
        sorted_energies.sort()
        noise_level = np.mean(sorted_energies[:max(1, len(sorted_energies)//5)])
        
        # Calculate SNR
        snr = 10 * np.log10(energy_level / (noise_level + 1e-10)) if noise_level > 0 else 100.0
        
        # Calculate clipping percentage
        clipping_threshold = 0.95  # Close to maximum value
        clipping_count = np.sum(np.abs(audio_data) > clipping_threshold)
        clipping_percentage = clipping_count / len(audio_data) if len(audio_data) > 0 else 0.0
        
        # Calculate DC offset
        dc_offset = np.mean(audio_data)
        
        # Use speech probability from VAD if available
        speech_presence_probability = frame.metadata.get("speech_probability", 0.0)
        
        # Estimate reverberation level (simplified)
        reverberation_level = 0.0
        
        # Estimate distortion level (simplified)
        distortion_level = 0.0
        
        return AudioQualityMetrics(
            snr_db=snr,
            clipping_percentage=clipping_percentage,
            dc_offset=dc_offset,
            energy_level=energy_level,
            noise_level=noise_level,
            speech_presence_probability=speech_presence_probability,
            reverberation_level=reverberation_level,
            distortion_level=distortion_level,
            timestamp=frame.timestamp
        )


class AudioSegmentBuilder:
    """Builds audio segments from individual frames based on voice activity."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the audio segment builder.
        
        Args:
            config: Configuration for the segment builder
        """
        self.max_segment_duration_ms = config.get("max_segment_duration_ms", 10000)
        self.min_segment_duration_ms = config.get("min_segment_duration_ms", 100)
        self.speech_padding_ms = config.get("speech_padding_ms", 300)
        self.frame_duration_ms = config.get("frame_duration_ms", 30)
        
        self.padding_frames = self.speech_padding_ms // self.frame_duration_ms
        self.min_segment_frames = self.min_segment_duration_ms // self.frame_duration_ms
        self.max_segment_frames = self.max_segment_duration_ms // self.frame_duration_ms
        
        self.current_segments = {}  # source_id -> current segment being built
        self.silence_counters = {}  # source_id -> count of silence frames
    
    def process_frame(self, frame: AudioFrame) -> Optional[AudioSegment]:
        """
        Process a frame and potentially produce a completed segment.
        
        Args:
            frame: Audio frame to process
            
        Returns:
            Optional[AudioSegment]: Completed segment if available, None otherwise
        """
        source_id = frame.source_id
        
        # Initialize state for new sources
        if source_id not in self.current_segments:
            self.current_segments[source_id] = None
            self.silence_counters[source_id] = 0
        
        # Get current state
        current_segment = self.current_segments[source_id]
        silence_counter = self.silence_counters[source_id]
        
        # Handle speech frame
        if frame.speech_detected:
            # Reset silence counter
            self.silence_counters[source_id] = 0
            
            # Create new segment if none exists
            if current_segment is None:
                segment_id = f"{source_id}_{uuid.uuid4()}"
                current_segment = AudioSegment(
                    frames=[],
                    segment_id=segment_id,
                    source_id=source_id,
                    start_time=frame.timestamp,
                    end_time=frame.timestamp + timedelta(milliseconds=frame.duration_ms),
                    duration_ms=0,
                    speech_confidence=frame.metadata.get("speech_probability", 0.5),
                    is_complete=False
                )
                self.current_segments[source_id] = current_segment
            
            # Add frame to segment
            current_segment.frames.append(frame)
            current_segment.end_time = frame.timestamp + timedelta(milliseconds=frame.duration_ms)
            current_segment.duration_ms += frame.duration_ms
            
            # Check if segment exceeds maximum duration
            if len(current_segment.frames) >= self.max_segment_frames:
                # Mark segment as complete
                current_segment.is_complete = True
                
                # Create result and reset state
                result = current_segment
                self.current_segments[source_id] = None
                
                return result
            
            return None
        
        # Handle silence frame
        else:
            # No active segment, nothing to do
            if current_segment is None:
                return None
            
            # Add frame during padding period
            if silence_counter < self.padding_frames:
                current_segment.frames.append(frame)
                current_segment.end_time = frame.timestamp + timedelta(milliseconds=frame.duration_ms)
                current_segment.duration_ms += frame.duration_ms
                self.silence_counters[source_id] += 1
                return None
            
            # Padding period over, finalize segment if long enough
            if len(current_segment.frames) >= self.min_segment_frames:
                current_segment.is_complete = True
                result = current_segment
                self.current_segments[source_id] = None
                self.silence_counters[source_id] = 0
                return result
            else:
                # Segment too short, discard it
                self.current_segments[source_id] = None
                self.silence_counters[source_id] = 0
                return None
    
    def flush_segment(self, source_id: str) -> Optional[AudioSegment]:
        """
        Flush any in-progress segment for a source.
        
        Args:
            source_id: Source ID to flush
            
        Returns:
            Optional[AudioSegment]: Completed segment if available, None otherwise
        """
        if source_id not in self.current_segments or self.current_segments[source_id] is None:
            return None
        
        current_segment = self.current_segments[source_id]
        
        # Only return segment if it's long enough
        if len(current_segment.frames) >= self.min_segment_frames:
            current_segment.is_complete = True
            result = current_segment
            self.current_segments[source_id] = None
            self.silence_counters[source_id] = 0
            return result
        else:
            # Segment too short, discard it
            self.current_segments[source_id] = None
            self.silence_counters[source_id] = 0
            return None


class SpeakerCountEstimator:
    """Estimates the number of speakers in an audio segment."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the speaker count estimator.
        
        Args:
            config: Configuration for the estimator
        """
        self.enabled = config.get("speaker_count_estimation_enabled", True)
        self.method = config.get("speaker_count_method", "clustering")
        self.min_segment_duration_ms = config.get("min_segment_duration_ms", 1000)
        
        # Initialize with optional external speaker counting model
        self.use_external_model = config.get("use_external_sc_model", False)
        self.external_model = None
        
        if self.use_external_model:
            try:
                # This would be replaced with actual model loading code
                logger.info("Initializing external speaker count model")
                # self.external_model = load_model(config.get("sc_model_path"))
            except Exception as e:
                logger.error(f"Failed to load external speaker count model: {e}")
                self.use_external_model = False
    
    def estimate(self, segment: AudioSegment) -> int:
        """
        Estimate the number of speakers in a segment.
        
        Args:
            segment: Audio segment to analyze
            
        Returns:
            int: Estimated number of speakers
        """
        if not self.enabled:
            return 1
        
        # Only process segments of sufficient duration
        if segment.duration_ms < self.min_segment_duration_ms:
            return 1
        
        # Use external model if available
        if self.use_external_model and self.external_model:
            # This would be replaced with actual model inference code
            # return self.external_model.estimate(segment)
            return 1
        
        # Simplified speaker count estimation (placeholder)
        # In a production system, this would be replaced with a more sophisticated algorithm
        # such as spectral clustering of speaker embeddings, diarization, etc.
        
        # For now, just return 1 (single speaker)
        return 1


class AudioCollectionManager:
    """
    Manages the collection of audio from various sources for tracking and identification.
    
    This class serves as the main entry point for the audio collection subsystem,
    handling source registration, connection management, audio processing,
    and providing processed audio segments to other system components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the audio collection manager.
        
        Args:
            config: Configuration for audio collection
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        
        if not self.enabled:
            logger.info("Audio collection manager is disabled")
            return
        
        # Initialize source management
        self.sources = {}  # source_id -> AudioSource
        self.source_configs = {}  # source_id -> config
        self.source_statuses = {}  # source_id -> status info
        
        # Initialize processing components
        self.vad = VoiceActivityDetector(config)
        self.noise_reduction = NoiseReduction(config)
        self.quality_analyzer = AudioQualityAnalyzer(config)
        self.segment_builder = AudioSegmentBuilder(config)
        self.speaker_count_estimator = SpeakerCountEstimator(config)
        
        # Set up processing queues and threads
        self.frame_queue = queue.Queue(maxsize=1000)
        self.segment_queue = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
        
        # Thread management
        self.collection_threads = {}  # source_id -> thread
        self.processing_thread = None
        
        # Performance monitoring
        self.performance_metrics = {
            "frames_processed": 0,
            "segments_created": 0,
            "processing_time_ms": 0,
            "dropped_frames": 0,
            "source_errors": {},
            "last_reset_time": datetime.now()
        }
        
        # Status flags
        self.is_running = False
        self._lock = threading.RLock()
        
        # Thread pool for parallel operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.get("max_worker_threads", 4),
            thread_name_prefix="AudioWorker"
        )
        
        # Check for privacy and audit components
        self.privacy_manager = None
        try:
            self.privacy_manager = PrivacyManager()
        except Exception as e:
            logger.warning(f"Could not initialize privacy manager: {e}")
        
        self.audit_logger = None
        try:
            self.audit_logger = AuditLogger()
        except Exception as e:
            logger.warning(f"Could not initialize audit logger: {e}")
        
        # Initialize data validator
        self.data_validator = DataValidator()
        
        logger.info("Audio collection manager initialized")
    
    def start(self):
        """Start the audio collection manager."""
        with self._lock:
            if not self.enabled:
                logger.info("Audio collection manager is disabled, not starting")
                return
            
            if self.is_running:
                logger.warning("Audio collection manager is already running")
                return
            
            logger.info("Starting audio collection manager")
            
            # Reset stop event
            self.stop_event.clear()
            
            # Start the processing thread
            self.processing_thread = threading.Thread(
                target=self._processing_thread_func,
                name="AudioProcessing",
                daemon=True
            )
            self.processing_thread.start()
            
            # Start collection for all registered sources
            for source_id in self.sources:
                self._start_collection_thread(source_id)
            
            self.is_running = True
            
            if self.audit_logger:
                self.audit_logger.log_system_event(
                    "audio_collection_started",
                    {"source_count": len(self.sources)}
                )
            
            logger.info(f"Audio collection manager started with {len(self.sources)} sources")
    
    def stop(self):
        """Stop the audio collection manager."""
        with self._lock:
            if not self.is_running:
                return
            
            logger.info("Stopping audio collection manager")
            
            # Signal threads to stop
            self.stop_event.set()
            
            # Stop collection threads
            for thread in self.collection_threads.values():
                if thread.is_alive():
                    thread.join(timeout=2)
            
            # Stop processing thread
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2)
            
            # Clear thread references
            self.collection_threads = {}
            self.processing_thread = None
            
            # Shut down thread pool
            self.thread_pool.shutdown(wait=True)
            
            # Close all sources
            for source in self.sources.values():
                source.disconnect()
            
            self.is_running = False
            
            if self.audit_logger:
                self.audit_logger.log_system_event(
                    "audio_collection_stopped",
                    {
                        "frames_processed": self.performance_metrics["frames_processed"],
                        "segments_created": self.performance_metrics["segments_created"]
                    }
                )
            
            logger.info("Audio collection manager stopped")
    
    def register_source(self, source_id: str, source_type: str, config: Dict[str, Any]) -> bool:
        """
        Register a new audio source.
        
        Args:
            source_id: Unique identifier for the source
            source_type: Type of audio source
            config: Configuration for the source
            
        Returns:
            bool: True if registration was successful, False otherwise
        """
        with self._lock:
            if source_id in self.sources:
                logger.warning(f"Source {source_id} already registered")
                return False
            
            try:
                # Create source instance based on type
                if source_type == "microphone":
                    source = MicrophoneAudioSource(source_id, config)
                elif source_type == "ip_audio":
                    source = IPAudioSource(source_id, config)
                elif source_type == "file":
                    source = FileAudioSource(source_id, config)
                else:
                    logger.error(f"Unsupported source type: {source_type}")
                    return False
                
                # Check if source is available
                if not source.is_available():
                    logger.warning(f"Source {source_id} is not available")
                    return False
                
                # Store source and config
                self.sources[source_id] = source
                self.source_configs[source_id] = config
                self.source_statuses[source_id] = {
                    "status": "registered",
                    "last_update": datetime.now(),
                    "error": None
                }
                
                # Start collection if manager is running
                if self.is_running:
                    self._start_collection_thread(source_id)
                
                if self.audit_logger:
                    self.audit_logger.log_system_event(
                        "audio_source_registered",
                        {
                            "source_id": source_id,
                            "source_type": source_type
                        }
                    )
                
                logger.info(f"Registered audio source {source_id} of type {source_type}")
                return True
                
            except Exception as e:
                logger.error(f"Error registering source {source_id}: {e}")
                return False
    
    def unregister_source(self, source_id: str) -> bool:
        """
        Unregister an audio source.
        
        Args:
            source_id: Identifier of the source to unregister
            
        Returns:
            bool: True if unregistration was successful, False otherwise
        """
        with self._lock:
            if source_id not in self.sources:
                logger.warning(f"Source {source_id} not registered")
                return False
            
            try:
                # Stop collection thread if running
                if source_id in self.collection_threads:
                    thread = self.collection_threads.pop(source_id)
                    if thread.is_alive():
                        thread.join(timeout=2)
                
                # Disconnect source
                source = self.sources.pop(source_id)
                source.disconnect()
                
                # Remove from other dictionaries
                self.source_configs.pop(source_id, None)
                self.source_statuses.pop(source_id, None)
                
                if self.audit_logger:
                    self.audit_logger.log_system_event(
                        "audio_source_unregistered",
                        {"source_id": source_id}
                    )
                
                logger.info(f"Unregistered audio source {source_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error unregistering source {source_id}: {e}")
                return False
    
    def get_source_status(self, source_id: str) -> Dict[str, Any]:
        """
        Get the status of an audio source.
        
        Args:
            source_id: Identifier of the source
            
        Returns:
            Dict: Status information for the source
        """
        with self._lock:
            if source_id not in self.sources:
                return {"status": "not_registered"}
            
            source = self.sources[source_id]
            status, error = source.get_status()
            
            result = {
                "status": status.value,
                "error": error,
                "last_update": self.source_statuses[source_id]["last_update"],
                "metadata": source.get_metadata().__dict__
            }
            
            return result
    
    def get_all_source_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the status of all audio sources.
        
        Returns:
            Dict: Status information for all sources
        """
        with self._lock:
            statuses = {}
            
            for source_id, source in self.sources.items():
                status, error = source.get_status()
                
                statuses[source_id] = {
                    "status": status.value,
                    "error": error,
                    "last_update": self.source_statuses[source_id]["last_update"],
                    "metadata": source.get_metadata().__dict__
                }
            
            return statuses
    
    def get_next_segment(self, timeout: Optional[float] = None) -> Optional[AudioSegment]:
        """
        Get the next available audio segment.
        
        Args:
            timeout: Maximum time to wait for a segment, in seconds
            
        Returns:
            Optional[AudioSegment]: Audio segment if available, None otherwise
        """
        try:
            return self.segment_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the audio collection subsystem.
        
        Returns:
            Dict: Performance metrics
        """
        with self._lock:
            # Calculate rates
            now = datetime.now()
            time_since_reset = (now - self.performance_metrics["last_reset_time"]).total_seconds()
            
            if time_since_reset > 0:
                frames_per_second = self.performance_metrics["frames_processed"] / time_since_reset
                segments_per_second = self.performance_metrics["segments_created"] / time_since_reset
                avg_processing_time = self.performance_metrics["processing_time_ms"] / max(1, self.performance_metrics["frames_processed"])
            else:
                frames_per_second = 0
                segments_per_second = 0
                avg_processing_time = 0
            
            # Copy metrics and add calculated values
            metrics = self.performance_metrics.copy()
            metrics.update({
                "frames_per_second": frames_per_second,
                "segments_per_second": segments_per_second,
                "avg_processing_time_ms": avg_processing_time,
                "frame_queue_size": self.frame_queue.qsize(),
                "segment_queue_size": self.segment_queue.qsize(),
                "uptime_seconds": time_since_reset
            })
            
            return metrics
    
    def reset_performance_metrics(self):
        """Reset all performance metrics."""
        with self._lock:
            self.performance_metrics = {
                "frames_processed": 0,
                "segments_created": 0,
                "processing_time_ms": 0,
                "dropped_frames": 0,
                "source_errors": {},
                "last_reset_time": datetime.now()
            }
    
    def _start_collection_thread(self, source_id: str):
        """
        Start the collection thread for a source.
        
        Args:
            source_id: Identifier of the source
        """
        if source_id in self.collection_threads and self.collection_threads[source_id].is_alive():
            return
        
        thread = threading.Thread(
            target=self._collection_thread_func,
            args=(source_id,),
            name=f"AudioCollection-{source_id}",
            daemon=True
        )
        
        self.collection_threads[source_id] = thread
        thread.start()
    
    def _collection_thread_func(self, source_id: str):
        """
        Thread function for collecting audio from a source.
        
        Args:
            source_id: Identifier of the source
        """
        source = self.sources[source_id]
        reconnect_interval = self.config.get("reconnect_interval_s", 5)
        last_reconnect_attempt = None
        
        logger.info(f"Starting collection from source {source_id}")
        
        while not self.stop_event.is_set():
            try:
                # Check if source is connected, try to connect if not
                status, _ = source.get_status()
                
                if status != ConnectionStatus.CONNECTED and status != ConnectionStatus.STREAMING:
                    # Rate-limit reconnection attempts
                    now = datetime.now()
                    if (last_reconnect_attempt is None or 
                        (now - last_reconnect_attempt).total_seconds() >= reconnect_interval):
                        
                        logger.info(f"Connecting to source {source_id}")
                        if source.connect():
                            self.source_statuses[source_id] = {
                                "status": "connected",
                                "last_update": now,
                                "error": None
                            }
                        else:
                            self.source_statuses[source_id] = {
                                "status": "connection_failed",
                                "last_update": now,
                                "error": source.error_message
                            }
                            
                            # Record error in metrics
                            self.performance_metrics["source_errors"][source_id] = {
                                "last_error": source.error_message,
                                "last_error_time": now,
                                "error_count": self.performance_metrics["source_errors"].get(source_id, {}).get("error_count", 0) + 1
                            }
                        
                        last_reconnect_attempt = now
                    
                    # Sleep before trying again
                    time.sleep(0.1)
                    continue
                
                # Read a frame from the source
                frame = source.read_frame()
                
                if frame is not None:
                    # Update source status
                    self.source_statuses[source_id] = {
                        "status": "streaming",
                        "last_update": datetime.now(),
                        "error": None
                    }
                    
                    # Validate frame
                    validation_result = self.data_validator.validate_audio_frame(frame)
                    
                    if validation_result.is_valid:
                        # Put frame in queue for processing
                        try:
                            self.frame_queue.put(frame, timeout=0.1)
                        except queue.Full:
                            self.performance_metrics["dropped_frames"] += 1
                    else:
                        logger.warning(f"Invalid frame from source {source_id}: {validation_result.error}")
                
                # Short sleep to avoid busy-waiting
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in collection thread for source {source_id}: {e}")
                
                # Update source status
                self.source_statuses[source_id] = {
                    "status": "error",
                    "last_update": datetime.now(),
                    "error": str(e)
                }
                
                # Record error in metrics
                self.performance_metrics["source_errors"][source_id] = {
                    "last_error": str(e),
                    "last_error_time": datetime.now(),
                    "error_count": self.performance_metrics["source_errors"].get(source_id, {}).get("error_count", 0) + 1
                }
                
                # Sleep before trying again
                time.sleep(0.5)
        
        # Clean up source when thread exits
        try:
            source.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting source {source_id}: {e}")
        
        logger.info(f"Collection thread for source {source_id} stopped")
    
    def _processing_thread_func(self):
        """Thread function for processing audio frames."""
        logger.info("Audio processing thread started")
        
        while not self.stop_event.is_set():
            try:
                # Get a frame from the queue
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                start_time = time.time()
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                if processed_frame:
                    # Try to build a segment from this frame
                    segment = self.segment_builder.process_frame(processed_frame)
                    
                    if segment:
                        # Finalize segment
                        segment = self._finalize_segment(segment)
                        
                        # Put segment in output queue
                        try:
                            self.segment_queue.put(segment, timeout=0.1)
                            self.performance_metrics["segments_created"] += 1
                        except queue.Full:
                            logger.warning("Segment queue full, dropping segment")
                
                # Update performance metrics
                self.performance_metrics["frames_processed"] += 1
                self.performance_metrics["processing_time_ms"] += (time.time() - start_time) * 1000
                
            except Exception as e:
                logger.error(f"Error in audio processing thread: {e}")
                time.sleep(0.1)
        
        # Finalize any in-progress segments
        try:
            self._flush_segments()
        except Exception as e:
            logger.error(f"Error flushing segments: {e}")
        
        logger.info("Audio processing thread stopped")
    
    def _process_frame(self, frame: AudioFrame) -> Optional[AudioFrame]:
        """
        Process an audio frame.
        
        Args:
            frame: Audio frame to process
            
        Returns:
            Optional[AudioFrame]: Processed frame if successful, None otherwise
        """
        try:
            # Detect voice activity
            frame.speech_detected = self.vad.detect(frame)
            
            # Apply noise reduction
            processed_frame = self.noise_reduction.process(frame)
            
            # Analyze quality
            quality_metrics = self.quality_analyzer.analyze(processed_frame)
            processed_frame.quality_metrics = quality_metrics
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Error processing frame from source {frame.source_id}: {e}")
            return None
    
    def _finalize_segment(self, segment: AudioSegment) -> AudioSegment:
        """
        Finalize an audio segment before sending it to output.
        
        Args:
            segment: Audio segment to finalize
            
        Returns:
            AudioSegment: Finalized segment
        """
        # Calculate speech confidence
        segment.speech_confidence = self.vad.get_segment_speech_confidence(segment)
        
        # Estimate speaker count
        segment.speaker_count_estimate = self.speaker_count_estimator.estimate(segment)
        
        # Apply privacy controls if available
        if self.privacy_manager:
            # This would apply any privacy-related transformations
            # For example, redacting sensitive information
            pass
        
        # Log audit event if enabled
        if self.audit_logger:
            self.audit_logger.log_system_event(
                "audio_segment_created",
                {
                    "segment_id": segment.segment_id,
                    "source_id": segment.source_id,
                    "duration_ms": segment.duration_ms,
                    "speech_confidence": segment.speech_confidence,
                    "speaker_count": segment.speaker_count_estimate
                }
            )
        
        return segment
    
    def _flush_segments(self):
        """Flush any in-progress segments for all sources."""
        for source_id in self.sources:
            segment = self.segment_builder.flush_segment(source_id)
            
            if segment:
                # Finalize segment
                segment = self._finalize_segment(segment)
                
                # Put segment in output queue
                try:
                    self.segment_queue.put(segment, timeout=0.1)
                    self.performance_metrics["segments_created"] += 1
                except queue.Full:
                    logger.warning("Segment queue full when flushing, dropping segment")
    
    def shutdown(self):
        """Shut down the audio collection manager and release resources."""
        logger.info("Shutting down audio collection manager")
        self.stop()
        
        with self._lock:
            # Clear all collections
            self.sources = {}
            self.source_configs = {}
            self.source_statuses = {}
            
            # Clear queues
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            while not self.segment_queue.empty():
                try:
                    self.segment_queue.get_nowait()
                except queue.Empty:
                    break
        
        logger.info("Audio collection manager shut down")


# Factory function to create pre-configured manager
def create_audio_collection_manager(config_path: Optional[str] = None) -> AudioCollectionManager:
    """
    Create and configure an audio collection manager.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        AudioCollectionManager: Configured manager instance
    """
    # Get configuration
    if config_path:
        # Load from specified file
        config = ConfigManager.load(config_path).data_collection.audio
    else:
        # Load from default configuration
        config = ConfigManager.get_config().data_collection.audio
    
    # Create manager
    manager = AudioCollectionManager(config)
    
    # Auto-register sources from config if specified
    auto_register_sources = config.get("auto_register_sources", [])
    for source_config in auto_register_sources:
        source_id = source_config.get("id")
        source_type = source_config.get("type")
        
        if source_id and source_type:
            manager.register_source(source_id, source_type, source_config)
    
    return manager
