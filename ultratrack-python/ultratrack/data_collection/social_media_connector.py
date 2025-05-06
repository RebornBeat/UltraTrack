"""
UltraTrack Social Media Connector

This module provides interfaces for collecting publicly available data from social media platforms
while respecting privacy regulations and platform terms of service. It only collects information
that is legally accessible and relevant for tracking purposes.

Key features:
- Multi-platform integration (most major social networks)
- Authentication and rate limit management
- Privacy-compliant data collection
- Data standardization across platforms
- Resilient connection handling
- Comprehensive logging and audit trail

Copyright (c) 2025 Your Organization
"""

import asyncio
import base64
import enum
import hashlib
import hmac
import json
import logging
import time
import typing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlencode

import httpx
import oauthlib.oauth2
import tenacity
from cryptography.fernet import Fernet

from ultratrack.compliance.audit_logger import AuditLogger
from ultratrack.compliance.authorized_purpose import AuthorizedPurposeManager, Purpose
from ultratrack.compliance.privacy_manager import PrivacyManager, DataCategory
from ultratrack.config import ConfigManager
from ultratrack.data_processing.anonymization import Anonymizer, AnonymizationLevel
from ultratrack.data_processing.data_fusion import DataAlignment
from ultratrack.security.encryption import EncryptionManager

# Configure module logger
logger = logging.getLogger(__name__)


class SocialMediaPlatform(enum.Enum):
    """Enumeration of supported social media platforms."""
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    MASTODON = "mastodon"
    THREADS = "threads"
    VK = "vk"
    WEIBO = "weibo"


class ContentType(enum.Enum):
    """Types of content available on social media platforms."""
    POST = "post"
    PROFILE = "profile"
    COMMENT = "comment"
    MEDIA = "media"
    LOCATION = "location"
    CONTACT = "contact"
    EVENT = "event"
    GROUP = "group"
    HASHTAG = "hashtag"
    MENTION = "mention"


class ConnectionStatus(enum.Enum):
    """Status of the connection to a social media platform."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    UNAUTHORIZED = "unauthorized"
    PENDING = "pending"


class AuthType(enum.Enum):
    """Authentication types for social media platforms."""
    OAUTH = "oauth"
    API_KEY = "api_key"
    APP_TOKEN = "app_token"
    JWT = "jwt"
    BASIC = "basic"
    NONE = "none"


@dataclass
class PlatformCredentials:
    """Secure storage for platform API credentials."""
    platform: SocialMediaPlatform
    auth_type: AuthType
    app_id: Optional[str] = None
    app_secret: Optional[str] = field(default=None, repr=False)
    api_key: Optional[str] = field(default=None, repr=False)
    api_secret: Optional[str] = field(default=None, repr=False)
    access_token: Optional[str] = field(default=None, repr=False)
    refresh_token: Optional[str] = field(default=None, repr=False)
    token_expiry: Optional[datetime] = None
    bearer_token: Optional[str] = field(default=None, repr=False)
    endpoint_url: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if credentials are valid and not expired."""
        if self.auth_type == AuthType.OAUTH and self.access_token:
            if self.token_expiry and self.token_expiry > datetime.now() + timedelta(minutes=5):
                return True
            return False
        
        if self.auth_type == AuthType.API_KEY and self.api_key:
            return True
        
        if self.auth_type == AuthType.APP_TOKEN and self.app_id and self.app_secret:
            return True
        
        if self.auth_type == AuthType.JWT and self.bearer_token:
            return True
        
        if self.auth_type == AuthType.BASIC and self.app_id and self.app_secret:
            return True
        
        if self.auth_type == AuthType.NONE:
            return True
        
        return False
    
    def needs_refresh(self) -> bool:
        """Check if OAuth tokens need refreshing."""
        if self.auth_type != AuthType.OAUTH:
            return False
        
        if not self.token_expiry:
            return True
        
        return self.token_expiry <= datetime.now() + timedelta(minutes=15)


@dataclass
class ApiSettings:
    """API settings for a specific platform."""
    rate_limit: int = 60  # requests per minute
    concurrent_requests: int = 5
    timeout_seconds: int = 30
    retry_count: int = 3
    retry_backoff_factor: float = 1.5
    retry_max_backoff: float = 60.0
    user_agent: str = "UltraTrack/1.0"
    api_version: str = "v1"


@dataclass
class PlatformConfig:
    """Configuration for a social media platform."""
    platform: SocialMediaPlatform
    enabled: bool = True
    credentials: Optional[PlatformCredentials] = None
    api_settings: ApiSettings = field(default_factory=ApiSettings)
    base_url: str = ""
    cache_ttl_seconds: int = 300
    webhook_url: Optional[str] = None
    proxy_url: Optional[str] = None
    allowed_content_types: Set[ContentType] = field(default_factory=lambda: {
        ContentType.POST, ContentType.PROFILE, ContentType.LOCATION
    })


@dataclass
class PublicProfile:
    """Representation of a public social media profile."""
    platform: SocialMediaPlatform
    platform_id: str
    username: Optional[str] = None
    display_name: Optional[str] = None
    bio: Optional[str] = None
    profile_url: Optional[str] = None
    profile_image_url: Optional[str] = None
    location: Optional[str] = None
    follower_count: Optional[int] = None
    following_count: Optional[int] = None
    post_count: Optional[int] = None
    verified: bool = False
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None
    data_collected_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "platform": self.platform.value,
            "platform_id": self.platform_id,
            "username": self.username,
            "display_name": self.display_name,
            "bio": self.bio,
            "profile_url": self.profile_url,
            "profile_image_url": self.profile_image_url,
            "location": self.location,
            "follower_count": self.follower_count,
            "following_count": self.following_count,
            "post_count": self.post_count,
            "verified": self.verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "data_collected_at": self.data_collected_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PublicProfile':
        """Create profile from dictionary."""
        # Convert string dates to datetime objects
        for date_field in ['created_at', 'last_active', 'data_collected_at']:
            if data.get(date_field) and isinstance(data[date_field], str):
                data[date_field] = datetime.fromisoformat(data[date_field])
        
        # Convert platform string to enum
        if 'platform' in data and isinstance(data['platform'], str):
            data['platform'] = SocialMediaPlatform(data['platform'])
        
        return cls(**data)


@dataclass
class LocationInfo:
    """Geographic location information from social media."""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    place_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    accuracy_meters: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def has_coordinates(self) -> bool:
        """Check if location has valid coordinates."""
        return (self.latitude is not None and 
                self.longitude is not None and
                -90 <= self.latitude <= 90 and
                -180 <= self.longitude <= 180)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "place_name": self.place_name,
            "address": self.address,
            "city": self.city,
            "region": self.region,
            "country": self.country,
            "accuracy_meters": self.accuracy_meters,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass
class ContentMetadata:
    """Metadata for social media content."""
    content_type: ContentType
    platform: SocialMediaPlatform
    content_id: str
    author_id: str
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    location: Optional[LocationInfo] = None
    engagement_count: Optional[int] = None
    permalink: Optional[str] = None
    mentions: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    media_urls: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content_type": self.content_type.value,
            "platform": self.platform.value,
            "content_id": self.content_id,
            "author_id": self.author_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "location": self.location.to_dict() if self.location else None,
            "engagement_count": self.engagement_count,
            "permalink": self.permalink,
            "mentions": self.mentions,
            "hashtags": self.hashtags,
            "media_urls": self.media_urls
        }


@dataclass
class SocialMediaContent:
    """Content from social media platforms."""
    metadata: ContentMetadata
    text_content: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "text_content": self.text_content
            # Exclude raw_data from dictionary conversion for security/privacy
        }


@dataclass
class SearchParameters:
    """Parameters for searching social media."""
    query: str
    platforms: List[SocialMediaPlatform] = field(default_factory=list)
    content_types: List[ContentType] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    max_results: int = 100
    include_replies: bool = False
    include_reposts: bool = False
    location_radius_km: Optional[float] = None
    location_center: Optional[Tuple[float, float]] = None
    sort_by: str = "recency"  # "recency" or "relevance"
    language: Optional[str] = None


class PlatformAuthManager:
    """Manages authentication with social media platforms."""
    
    def __init__(self, 
                 encryption_manager: EncryptionManager,
                 audit_logger: AuditLogger):
        """
        Initialize the platform authentication manager.
        
        Args:
            encryption_manager: System encryption manager
            audit_logger: System audit logger
        """
        self.encryption_manager = encryption_manager
        self.audit_logger = audit_logger
        self.oauth_clients: Dict[SocialMediaPlatform, Any] = {}
        self.token_cache: Dict[str, Dict[str, Any]] = {}
        logger.info("Platform authentication manager initialized")
    
    def configure_platform(self, credentials: PlatformCredentials) -> bool:
        """
        Configure authentication for a platform.
        
        Args:
            credentials: Platform credentials
            
        Returns:
            bool: True if configuration successful
        """
        platform = credentials.platform
        auth_type = credentials.auth_type
        
        try:
            logger.info(f"Configuring authentication for {platform.value}")
            
            if auth_type == AuthType.OAUTH:
                # Create OAuth client
                if not credentials.app_id or not credentials.app_secret:
                    logger.error(f"Missing OAuth credentials for {platform.value}")
                    return False
                
                # Set up OAuth client based on platform
                if platform == SocialMediaPlatform.TWITTER:
                    self._configure_twitter_oauth(credentials)
                elif platform == SocialMediaPlatform.FACEBOOK:
                    self._configure_facebook_oauth(credentials)
                # Add other platforms as needed
                
            # Encrypt sensitive information for storage
            self._secure_credentials(credentials)
            
            self.audit_logger.log_security_event(
                "platform_auth_configured",
                {"platform": platform.value, "auth_type": auth_type.value}
            )
            
            logger.info(f"Authentication configured for {platform.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure {platform.value} authentication: {str(e)}")
            return False
    
    def _configure_twitter_oauth(self, credentials: PlatformCredentials) -> None:
        """
        Configure Twitter OAuth client.
        
        Args:
            credentials: Twitter API credentials
        """
        # Twitter uses OAuth 1.0a or OAuth 2.0 depending on API version
        client = oauthlib.oauth2.BackendApplicationClient(client_id=credentials.app_id)
        self.oauth_clients[credentials.platform] = client
    
    def _configure_facebook_oauth(self, credentials: PlatformCredentials) -> None:
        """
        Configure Facebook OAuth client.
        
        Args:
            credentials: Facebook API credentials
        """
        # Facebook uses OAuth 2.0
        client = oauthlib.oauth2.BackendApplicationClient(client_id=credentials.app_id)
        self.oauth_clients[credentials.platform] = client
    
    def _secure_credentials(self, credentials: PlatformCredentials) -> None:
        """
        Encrypt and secure sensitive credential information.
        
        Args:
            credentials: Platform credentials to secure
        """
        # This method would use the encryption_manager to encrypt
        # sensitive fields like app_secret, api_key, access_token, etc.
        # Implementation details depend on the encryption_manager functionality
        pass
    
    def refresh_token(self, credentials: PlatformCredentials) -> PlatformCredentials:
        """
        Refresh OAuth access token if expired.
        
        Args:
            credentials: Platform credentials with refresh token
            
        Returns:
            Updated credentials with new access token
            
        Raises:
            ValueError: If token refresh fails
        """
        if not credentials.needs_refresh():
            return credentials
        
        logger.info(f"Refreshing access token for {credentials.platform.value}")
        
        try:
            platform = credentials.platform
            client = self.oauth_clients.get(platform)
            
            if not client or not credentials.refresh_token:
                raise ValueError(f"Missing OAuth client or refresh token for {platform.value}")
            
            # Refresh token logic depends on the platform
            if platform == SocialMediaPlatform.TWITTER:
                return self._refresh_twitter_token(credentials, client)
            elif platform == SocialMediaPlatform.FACEBOOK:
                return self._refresh_facebook_token(credentials, client)
            # Add other platforms as needed
            
            raise ValueError(f"Token refresh not implemented for {platform.value}")
            
        except Exception as e:
            logger.error(f"Token refresh failed for {credentials.platform.value}: {str(e)}")
            raise ValueError(f"Token refresh failed: {str(e)}")
    
    def _refresh_twitter_token(self, 
                              credentials: PlatformCredentials, 
                              client: Any) -> PlatformCredentials:
        """
        Refresh Twitter access token.
        
        Args:
            credentials: Twitter credentials
            client: OAuth client
            
        Returns:
            Updated credentials
        """
        # Implementation would use Twitter's token refresh endpoint
        # This is a simplified example
        # In reality, would use proper OAuth token refresh flow
        
        async def _refresh():
            async with httpx.AsyncClient() as http_client:
                response = await http_client.post(
                    "https://api.twitter.com/oauth2/token",
                    auth=(credentials.app_id, credentials.app_secret),
                    data={"grant_type": "refresh_token", "refresh_token": credentials.refresh_token}
                )
                response.raise_for_status()
                data = response.json()
                
                new_credentials = PlatformCredentials(
                    platform=credentials.platform,
                    auth_type=credentials.auth_type,
                    app_id=credentials.app_id,
                    app_secret=credentials.app_secret,
                    access_token=data.get("access_token"),
                    refresh_token=data.get("refresh_token", credentials.refresh_token),
                    token_expiry=datetime.now() + timedelta(seconds=data.get("expires_in", 3600)),
                    endpoint_url=credentials.endpoint_url
                )
                return new_credentials
        
        # Run async function in thread
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(_refresh())
        loop.close()
        
        return result
    
    def _refresh_facebook_token(self, 
                               credentials: PlatformCredentials, 
                               client: Any) -> PlatformCredentials:
        """
        Refresh Facebook access token.
        
        Args:
            credentials: Facebook credentials
            client: OAuth client
            
        Returns:
            Updated credentials
        """
        # Implementation would use Facebook's token refresh endpoint
        # This is a simplified example
        
        async def _refresh():
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(
                    "https://graph.facebook.com/oauth/access_token",
                    params={
                        "client_id": credentials.app_id,
                        "client_secret": credentials.app_secret,
                        "grant_type": "fb_exchange_token",
                        "fb_exchange_token": credentials.access_token
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                new_credentials = PlatformCredentials(
                    platform=credentials.platform,
                    auth_type=credentials.auth_type,
                    app_id=credentials.app_id,
                    app_secret=credentials.app_secret,
                    access_token=data.get("access_token"),
                    refresh_token=credentials.refresh_token,
                    token_expiry=datetime.now() + timedelta(seconds=data.get("expires_in", 3600)),
                    endpoint_url=credentials.endpoint_url
                )
                return new_credentials
        
        # Run async function in thread
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(_refresh())
        loop.close()
        
        return result


class PlatformAdapter:
    """Base adapter for social media platform integration."""
    
    def __init__(self, 
                 config: PlatformConfig,
                 auth_manager: PlatformAuthManager,
                 audit_logger: AuditLogger):
        """
        Initialize the platform adapter.
        
        Args:
            config: Platform configuration
            auth_manager: Authentication manager
            audit_logger: Audit logger
        """
        self.platform = config.platform
        self.config = config
        self.auth_manager = auth_manager
        self.audit_logger = audit_logger
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.api_settings.timeout_seconds),
            proxies=config.proxy_url,
            headers={"User-Agent": config.api_settings.user_agent}
        )
        self.status = ConnectionStatus.DISCONNECTED
        self.last_request_time = 0.0
        self.request_count = 0
        self.error_count = 0
        self.rate_limit_reset = 0
        
        logger.info(f"Initialized adapter for {self.platform.value}")
    
    async def connect(self) -> bool:
        """
        Establish connection to the platform.
        
        Returns:
            bool: True if connection successful
        """
        try:
            logger.info(f"Connecting to {self.platform.value}")
            
            if not self.config.enabled:
                logger.warning(f"{self.platform.value} is disabled in configuration")
                return False
            
            if not self.config.credentials or not self.config.credentials.is_valid():
                logger.error(f"Invalid credentials for {self.platform.value}")
                self.status = ConnectionStatus.UNAUTHORIZED
                return False
            
            # Refresh token if needed
            if self.config.credentials.needs_refresh():
                self.config.credentials = self.auth_manager.refresh_token(self.config.credentials)
            
            # Test connection
            await self._test_connection()
            
            self.status = ConnectionStatus.CONNECTED
            self.audit_logger.log_system_event(
                "platform_connected",
                {"platform": self.platform.value}
            )
            
            logger.info(f"Successfully connected to {self.platform.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.platform.value}: {str(e)}")
            self.status = ConnectionStatus.ERROR
            self.error_count += 1
            return False
    
    async def disconnect(self) -> None:
        """Close connection to the platform."""
        logger.info(f"Disconnecting from {self.platform.value}")
        
        try:
            await self.client.aclose()
            self.status = ConnectionStatus.DISCONNECTED
            self.audit_logger.log_system_event(
                "platform_disconnected",
                {"platform": self.platform.value}
            )
            
        except Exception as e:
            logger.error(f"Error during {self.platform.value} disconnection: {str(e)}")
    
    async def _test_connection(self) -> None:
        """
        Test connection to the platform.
        
        Raises:
            Exception: If connection test fails
        """
        # This should be implemented by platform-specific adapters
        raise NotImplementedError(f"Test connection not implemented for {self.platform.value}")
    
    async def _make_request(self, 
                           method: str, 
                           endpoint: str, 
                           params: Optional[Dict[str, Any]] = None,
                           data: Optional[Dict[str, Any]] = None,
                           headers: Optional[Dict[str, Any]] = None,
                           auth: Optional[Any] = None) -> httpx.Response:
        """
        Make a request to the platform API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body
            headers: HTTP headers
            auth: Authentication
            
        Returns:
            HTTP response
            
        Raises:
            httpx.HTTPError: If request fails
        """
        # Check connection status
        if self.status == ConnectionStatus.DISCONNECTED:
            await self.connect()
        
        if self.status == ConnectionStatus.RATE_LIMITED:
            if time.time() < self.rate_limit_reset:
                wait_time = self.rate_limit_reset - time.time()
                logger.warning(f"Rate limited on {self.platform.value}, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # Respect rate limits
        now = time.time()
        if self.request_count >= self.config.api_settings.rate_limit:
            elapsed = now - self.last_request_time
            min_interval = 60.0 / self.config.api_settings.rate_limit
            if elapsed < min_interval:
                wait_time = min_interval - elapsed
                logger.debug(f"Rate limiting {self.platform.value}, waiting {wait_time:.3f}s")
                await asyncio.sleep(wait_time)
        
        # Update request tracking
        self.last_request_time = time.time()
        self.request_count += 1
        
        # Apply platform-specific auth if not provided
        if not auth and self.config.credentials:
            auth = self._get_auth_for_request()
        
        # Combine endpoint with base URL
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Make request with retries
        retry_strategy = tenacity.retry(
            retry=tenacity.retry_if_exception_type(httpx.HTTPError),
            stop=tenacity.stop_after_attempt(self.config.api_settings.retry_count),
            wait=tenacity.wait_exponential(
                multiplier=self.config.api_settings.retry_backoff_factor,
                max=self.config.api_settings.retry_max_backoff
            ),
            before_sleep=lambda retry_state: logger.warning(
                f"Retrying {self.platform.value} request (attempt {retry_state.attempt_number})"
            )
        )
        
        @retry_strategy
        async def _request():
            req_headers = {"User-Agent": self.config.api_settings.user_agent}
            if headers:
                req_headers.update(headers)
            
            logger.debug(f"Making {method} request to {url}")
            
            response = await self.client.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=req_headers,
                auth=auth
            )
            
            # Check for rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "60"))
                self.rate_limit_reset = time.time() + retry_after
                self.status = ConnectionStatus.RATE_LIMITED
                logger.warning(f"{self.platform.value} rate limited, reset in {retry_after}s")
            
            # Check for auth errors
            if response.status_code == 401:
                self.status = ConnectionStatus.UNAUTHORIZED
                logger.error(f"{self.platform.value} authentication failed")
            
            response.raise_for_status()
            return response
        
        try:
            return await _request()
        except Exception as e:
            self.error_count += 1
            if self.error_count >= 5:
                self.status = ConnectionStatus.ERROR
            raise
    
    def _get_auth_for_request(self) -> Optional[Any]:
        """
        Get authentication for a request based on platform and auth type.
        
        Returns:
            Authentication object or None
        """
        if not self.config.credentials:
            return None
        
        creds = self.config.credentials
        
        if creds.auth_type == AuthType.BASIC:
            return httpx.BasicAuth(creds.app_id, creds.app_secret)
        
        if creds.auth_type == AuthType.API_KEY:
            # Return as header dict, will be added in the request
            return None
        
        if creds.auth_type == AuthType.OAUTH:
            # OAuth handling depends on platform
            return None
        
        return None
    
    async def get_profile(self, username: str) -> Optional[PublicProfile]:
        """
        Get a public social media profile.
        
        Args:
            username: Username or identifier
            
        Returns:
            Public profile if found, None otherwise
        """
        # This should be implemented by platform-specific adapters
        raise NotImplementedError(f"Get profile not implemented for {self.platform.value}")
    
    async def search(self, params: SearchParameters) -> List[SocialMediaContent]:
        """
        Search for content on the platform.
        
        Args:
            params: Search parameters
            
        Returns:
            List of matching content items
        """
        # This should be implemented by platform-specific adapters
        raise NotImplementedError(f"Search not implemented for {self.platform.value}")
    
    async def get_location_history(self, 
                                  username: str, 
                                  start_date: Optional[datetime] = None, 
                                  end_date: Optional[datetime] = None) -> List[LocationInfo]:
        """
        Get location history for a user from public posts.
        
        Args:
            username: Username or identifier
            start_date: Starting date for history
            end_date: Ending date for history
            
        Returns:
            List of location information
        """
        # This should be implemented by platform-specific adapters
        raise NotImplementedError(f"Location history not implemented for {self.platform.value}")
    
    async def get_connections(self, 
                             username: str, 
                             connection_type: str = "followers",
                             limit: int = 100) -> List[str]:
        """
        Get public connections for a user.
        
        Args:
            username: Username or identifier
            connection_type: Type of connection (followers, following)
            limit: Maximum number of connections to return
            
        Returns:
            List of connected usernames
        """
        # This should be implemented by platform-specific adapters
        raise NotImplementedError(f"Get connections not implemented for {self.platform.value}")


class TwitterAdapter(PlatformAdapter):
    """Twitter-specific platform adapter."""
    
    async def _test_connection(self) -> None:
        """Test connection to Twitter API."""
        response = await self._make_request("GET", "2/users/me")
        
        if response.status_code != 200:
            raise Exception(f"Twitter connection test failed: {response.status_code}")
        
        logger.info("Twitter connection test successful")
    
    async def get_profile(self, username: str) -> Optional[PublicProfile]:
        """
        Get a public Twitter profile.
        
        Args:
            username: Twitter username (without @)
            
        Returns:
            Public profile if found, None otherwise
        """
        try:
            # First get user ID from username
            response = await self._make_request(
                "GET", 
                "2/users/by/username/" + username,
                params={"user.fields": "created_at,description,location,profile_image_url,public_metrics,verified"}
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to get Twitter user {username}: {response.status_code}")
                return None
            
            data = response.json()
            if "data" not in data:
                logger.warning(f"No data returned for Twitter user {username}")
                return None
            
            user_data = data["data"]
            
            # Map Twitter API response to PublicProfile
            profile = PublicProfile(
                platform=SocialMediaPlatform.TWITTER,
                platform_id=user_data["id"],
                username=user_data["username"],
                display_name=user_data["name"],
                bio=user_data.get("description"),
                profile_url=f"https://twitter.com/{user_data['username']}",
                profile_image_url=user_data.get("profile_image_url"),
                location=user_data.get("location"),
                follower_count=user_data.get("public_metrics", {}).get("followers_count"),
                following_count=user_data.get("public_metrics", {}).get("following_count"),
                post_count=user_data.get("public_metrics", {}).get("tweet_count"),
                verified=user_data.get("verified", False),
                created_at=datetime.fromisoformat(user_data["created_at"].replace("Z", "+00:00"))
                if "created_at" in user_data else None,
                data_collected_at=datetime.now()
            )
            
            self.audit_logger.log_data_event(
                "profile_collected",
                {"platform": "twitter", "username": username}
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"Error getting Twitter profile for {username}: {str(e)}")
            return None
    
    async def search(self, params: SearchParameters) -> List[SocialMediaContent]:
        """
        Search for content on Twitter.
        
        Args:
            params: Search parameters
            
        Returns:
            List of matching tweets
        """
        try:
            query = params.query
            api_params = {
                "query": query,
                "max_results": min(params.max_results, 100),  # Twitter API limit
                "tweet.fields": "created_at,geo,public_metrics,entities",
                "expansions": "author_id,geo.place_id"
            }
            
            if params.start_date:
                api_params["start_time"] = params.start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            if params.end_date:
                api_params["end_time"] = params.end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            response = await self._make_request(
                "GET", 
                "2/tweets/search/recent",
                params=api_params
            )
            
            if response.status_code != 200:
                logger.warning(f"Twitter search failed: {response.status_code}")
                return []
            
            data = response.json()
            tweets = data.get("data", [])
            users = {user["id"]: user for user in data.get("includes", {}).get("users", [])}
            places = {place["id"]: place for place in data.get("includes", {}).get("places", [])}
            
            results = []
            
            for tweet in tweets:
                author_id = tweet.get("author_id")
                location = None
                
                # Extract location if available
                if "geo" in tweet and "place_id" in tweet["geo"]:
                    place_id = tweet["geo"]["place_id"]
                    place = places.get(place_id)
                    
                    if place:
                        # Extract coordinates if available
                        coords = None
                        if "geo" in place and "coordinates" in place["geo"]:
                            coords = place["geo"]["coordinates"]
                        
                        location = LocationInfo(
                            place_name=place.get("name"),
                            city=place.get("name"),
                            region=place.get("full_name", "").split(", ")[-1] if ", " in place.get("full_name", "") else None,
                            country=place.get("country"),
                            timestamp=datetime.fromisoformat(tweet["created_at"].replace("Z", "+00:00"))
                        )
                        
                        # Add coordinates if available (Twitter provides as [long, lat])
                        if coords and len(coords) == 2:
                            location.longitude = coords[0]
                            location.latitude = coords[1]
                
                # Extract hashtags and mentions
                hashtags = []
                mentions = []
                
                if "entities" in tweet:
                    if "hashtags" in tweet["entities"]:
                        hashtags = [tag["tag"] for tag in tweet["entities"]["hashtags"]]
                    
                    if "mentions" in tweet["entities"]:
                        mentions = [mention["username"] for mention in tweet["entities"]["mentions"]]
                
                # Create metadata
                metadata = ContentMetadata(
                    content_type=ContentType.POST,
                    platform=SocialMediaPlatform.TWITTER,
                    content_id=tweet["id"],
                    author_id=author_id,
                    created_at=datetime.fromisoformat(tweet["created_at"].replace("Z", "+00:00")),
                    location=location,
                    engagement_count=sum(tweet.get("public_metrics", {}).values()),
                    permalink=f"https://twitter.com/user/status/{tweet['id']}",
                    mentions=mentions,
                    hashtags=hashtags
                )
                
                # Create content
                content = SocialMediaContent(
                    metadata=metadata,
                    text_content=tweet.get("text"),
                    raw_data=tweet
                )
                
                results.append(content)
            
            self.audit_logger.log_data_event(
                "search_performed",
                {"platform": "twitter", "results_count": len(results)}
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Twitter: {str(e)}")
            return []


class FacebookAdapter(PlatformAdapter):
    """Facebook-specific platform adapter."""
    
    async def _test_connection(self) -> None:
        """Test connection to Facebook API."""
        response = await self._make_request("GET", "me", params={"fields": "id,name"})
        
        if response.status_code != 200:
            raise Exception(f"Facebook connection test failed: {response.status_code}")
        
        logger.info("Facebook connection test successful")
    
    async def get_profile(self, username: str) -> Optional[PublicProfile]:
        """
        Get a public Facebook profile.
        
        Args:
            username: Facebook username or page ID
            
        Returns:
            Public profile if found, None otherwise
        """
        try:
            # Facebook Graph API request
            response = await self._make_request(
                "GET", 
                username,
                params={"fields": "id,name,about,cover,picture,link,location,fan_count,created_time,verification_status"}
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to get Facebook user {username}: {response.status_code}")
                return None
            
            data = response.json()
            
            # Determine if this is a page or a user
            is_page = "fan_count" in data
            
            # Map Facebook API response to PublicProfile
            profile = PublicProfile(
                platform=SocialMediaPlatform.FACEBOOK,
                platform_id=data["id"],
                username=username,
                display_name=data.get("name"),
                bio=data.get("about"),
                profile_url=data.get("link", f"https://facebook.com/{username}"),
                profile_image_url=data.get("picture", {}).get("data", {}).get("url") if "picture" in data else None,
                location=data.get("location", {}).get("name") if "location" in data else None,
                follower_count=data.get("fan_count") if is_page else None,
                verified=data.get("verification_status") == "blue_verified",
                created_at=datetime.fromisoformat(data["created_time"].replace("Z", "+00:00"))
                if "created_time" in data else None,
                data_collected_at=datetime.now()
            )
            
            self.audit_logger.log_data_event(
                "profile_collected",
                {"platform": "facebook", "username": username}
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"Error getting Facebook profile for {username}: {str(e)}")
            return None


class SocialMediaConnector:
    """
    Main connector for social media platforms.
    
    This class provides a unified interface for collecting publicly available
    information from social media platforms while respecting privacy regulations
    and platform terms of service.
    """
    
    def __init__(self):
        """Initialize the social media connector."""
        # Load system components
        self.config = ConfigManager.get_instance().get_config()
        self.audit_logger = AuditLogger(self.config.compliance.audit)
        self.privacy_manager = PrivacyManager(self.config.compliance)
        self.purpose_manager = AuthorizedPurposeManager(self.config.compliance.authorized_purpose)
        self.encryption_manager = EncryptionManager(self.config.security.encryption)
        self.anonymizer = Anonymizer(self.config.data_processing.anonymization)
        
        # Initialize platform configurations
        self.platform_configs: Dict[SocialMediaPlatform, PlatformConfig] = {}
        self._load_platform_configs()
        
        # Initialize auth manager
        self.auth_manager = PlatformAuthManager(
            encryption_manager=self.encryption_manager,
            audit_logger=self.audit_logger
        )
        
        # Initialize platform adapters
        self.adapters: Dict[SocialMediaPlatform, PlatformAdapter] = {}
        self._initialize_adapters()
        
        # Connection state
        self.connected = False
        
        logger.info("Social media connector initialized")
    
    def _load_platform_configs(self) -> None:
        """Load platform configurations from system config."""
        # This would typically load from a configuration file or database
        # For now, we'll create some example configurations
        
        # Twitter configuration
        twitter_config = PlatformConfig(
            platform=SocialMediaPlatform.TWITTER,
            enabled=True,
            base_url="https://api.twitter.com/",
            credentials=PlatformCredentials(
                platform=SocialMediaPlatform.TWITTER,
                auth_type=AuthType.OAUTH,
                app_id=os.environ.get("TWITTER_APP_ID", ""),
                app_secret=os.environ.get("TWITTER_APP_SECRET", ""),
                access_token=os.environ.get("TWITTER_ACCESS_TOKEN", ""),
                refresh_token=os.environ.get("TWITTER_REFRESH_TOKEN", ""),
                token_expiry=datetime.now() + timedelta(hours=1)  # Example expiry
            ),
            api_settings=ApiSettings(
                rate_limit=300,  # 300 requests per 15 minutes = 20 per minute
                concurrent_requests=10,
                timeout_seconds=30,
                retry_count=3
            )
        )
        self.platform_configs[SocialMediaPlatform.TWITTER] = twitter_config
        
        # Facebook configuration
        facebook_config = PlatformConfig(
            platform=SocialMediaPlatform.FACEBOOK,
            enabled=True,
            base_url="https://graph.facebook.com/v18.0/",
            credentials=PlatformCredentials(
                platform=SocialMediaPlatform.FACEBOOK,
                auth_type=AuthType.OAUTH,
                app_id=os.environ.get("FACEBOOK_APP_ID", ""),
                app_secret=os.environ.get("FACEBOOK_APP_SECRET", ""),
                access_token=os.environ.get("FACEBOOK_ACCESS_TOKEN", ""),
                token_expiry=datetime.now() + timedelta(days=60)  # Example expiry
            ),
            api_settings=ApiSettings(
                rate_limit=200,  # Example rate limit
                concurrent_requests=5,
                timeout_seconds=30,
                retry_count=3
            )
        )
        self.platform_configs[SocialMediaPlatform.FACEBOOK] = facebook_config
        
        # Add other platform configurations as needed
    
    def _initialize_adapters(self) -> None:
        """Initialize platform adapters based on configurations."""
        platform_adapter_map = {
            SocialMediaPlatform.TWITTER: TwitterAdapter,
            SocialMediaPlatform.FACEBOOK: FacebookAdapter,
            # Add other platforms here
        }
        
        for platform, config in self.platform_configs.items():
            if not config.enabled:
                continue
            
            adapter_class = platform_adapter_map.get(platform)
            if not adapter_class:
                logger.warning(f"No adapter available for {platform.value}")
                continue
            
            try:
                adapter = adapter_class(
                    config=config,
                    auth_manager=self.auth_manager,
                    audit_logger=self.audit_logger
                )
                self.adapters[platform] = adapter
                logger.info(f"Initialized adapter for {platform.value}")
                
            except Exception as e:
                logger.error(f"Failed to initialize adapter for {platform.value}: {str(e)}")
    
    async def connect(self, platforms: Optional[List[SocialMediaPlatform]] = None) -> bool:
        """
        Connect to specified social media platforms or all enabled platforms.
        
        Args:
            platforms: List of platforms to connect to, or None for all
            
        Returns:
            bool: True if all connections successful
        """
        if not platforms:
            platforms = list(self.adapters.keys())
        
        logger.info(f"Connecting to {len(platforms)} platforms")
        
        connection_tasks = []
        for platform in platforms:
            adapter = self.adapters.get(platform)
            if adapter:
                connection_tasks.append(adapter.connect())
        
        # Connect to all platforms concurrently
        if connection_tasks:
            results = await asyncio.gather(*connection_tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            
            self.connected = success_count > 0
            
            if success_count == len(connection_tasks):
                logger.info(f"Successfully connected to all {len(platforms)} platforms")
            else:
                logger.warning(f"Connected to {success_count}/{len(platforms)} platforms")
            
            return success_count == len(connection_tasks)
        
        return False
    
    async def disconnect(self) -> None:
        """Disconnect from all platforms."""
        if not self.connected:
            return
        
        logger.info("Disconnecting from all platforms")
        
        disconnect_tasks = []
        for adapter in self.adapters.values():
            disconnect_tasks.append(adapter.disconnect())
        
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        self.connected = False
        
        logger.info("Disconnected from all platforms")
    
    async def get_profile(self, 
                         username: str, 
                         platform: SocialMediaPlatform,
                         purpose: Purpose) -> Optional[PublicProfile]:
        """
        Get a public profile from a specific platform.
        
        Args:
            username: Username or identifier
            platform: Social media platform
            purpose: Authorized purpose for collection
            
        Returns:
            Public profile if found, None otherwise
        """
        # Validate purpose
        if not self.purpose_manager.validate_purpose(purpose, DataCategory.SOCIAL_MEDIA_PROFILE):
            logger.warning(f"Unauthorized purpose {purpose} for profile collection")
            return None
        
        adapter = self.adapters.get(platform)
        if not adapter:
            logger.warning(f"No adapter available for {platform.value}")
            return None
        
        # Ensure connected
        if not self.connected:
            await self.connect([platform])
        
        # Get profile
        profile = await adapter.get_profile(username)
        
        if profile:
            self.audit_logger.log_data_event(
                "profile_accessed",
                {
                    "platform": platform.value,
                    "username": username,
                    "purpose": purpose.value
                }
            )
        
        return profile
    
    async def search_platform(self,
                             params: SearchParameters,
                             platform: SocialMediaPlatform,
                             purpose: Purpose) -> List[SocialMediaContent]:
        """
        Search for content on a specific platform.
        
        Args:
            params: Search parameters
            platform: Social media platform
            purpose: Authorized purpose for collection
            
        Returns:
            List of matching content
        """
        # Validate purpose
        if not self.purpose_manager.validate_purpose(purpose, DataCategory.SOCIAL_MEDIA_CONTENT):
            logger.warning(f"Unauthorized purpose {purpose} for content search")
            return []
        
        adapter = self.adapters.get(platform)
        if not adapter:
            logger.warning(f"No adapter available for {platform.value}")
            return []
        
        # Ensure connected
        if not self.connected:
            await self.connect([platform])
        
        # Search platform
        results = await adapter.search(params)
        
        if results:
            self.audit_logger.log_data_event(
                "content_searched",
                {
                    "platform": platform.value,
                    "query": params.query,
                    "result_count": len(results),
                    "purpose": purpose.value
                }
            )
        
        return results
    
    async def search(self, 
                    params: SearchParameters,
                    purpose: Purpose) -> Dict[SocialMediaPlatform, List[SocialMediaContent]]:
        """
        Search for content across multiple platforms.
        
        Args:
            params: Search parameters
            purpose: Authorized purpose for collection
            
        Returns:
            Dictionary of platforms to content lists
        """
        # Determine which platforms to search
        platforms = params.platforms
        if not platforms:
            platforms = list(self.adapters.keys())
        
        # Validate purpose
        if not self.purpose_manager.validate_purpose(purpose, DataCategory.SOCIAL_MEDIA_CONTENT):
            logger.warning(f"Unauthorized purpose {purpose} for content search")
            return {}
        
        # Ensure connected
        if not self.connected:
            await self.connect(platforms)
        
        # Search all platforms concurrently
        search_tasks = {}
        for platform in platforms:
            adapter = self.adapters.get(platform)
            if adapter:
                search_tasks[platform] = adapter.search(params)
        
        # Wait for all searches to complete
        results = {}
        for platform, task in search_tasks.items():
            try:
                platform_results = await task
                results[platform] = platform_results
                logger.info(f"Found {len(platform_results)} results on {platform.value}")
                
            except Exception as e:
                logger.error(f"Error searching {platform.value}: {str(e)}")
                results[platform] = []
        
        # Log search event
        total_results = sum(len(r) for r in results.values())
        self.audit_logger.log_data_event(
            "multi_platform_search",
            {
                "query": params.query,
                "platforms": [p.value for p in platforms],
                "total_results": total_results,
                "purpose": purpose.value
            }
        )
        
        return results
    
    async def get_location_history(self,
                                  username: str,
                                  platforms: List[SocialMediaPlatform],
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None,
                                  purpose: Purpose = None) -> Dict[SocialMediaPlatform, List[LocationInfo]]:
        """
        Get location history for a user from public posts across platforms.
        
        Args:
            username: Username or identifier (assumed to be the same across platforms)
            platforms: List of platforms to search
            start_date: Starting date for history
            end_date: Ending date for history
            purpose: Authorized purpose for collection
            
        Returns:
            Dictionary of platforms to location lists
        """
        # Validate purpose
        if not self.purpose_manager.validate_purpose(purpose, DataCategory.LOCATION_DATA):
            logger.warning(f"Unauthorized purpose {purpose} for location history")
            return {}
        
        # Ensure connected
        if not self.connected:
            await self.connect(platforms)
        
        # Get location history from all platforms concurrently
        history_tasks = {}
        for platform in platforms:
            adapter = self.adapters.get(platform)
            if adapter and hasattr(adapter, "get_location_history"):
                try:
                    history_tasks[platform] = adapter.get_location_history(
                        username, start_date, end_date
                    )
                except NotImplementedError:
                    logger.info(f"Location history not implemented for {platform.value}")
        
        # Wait for all tasks to complete
        results = {}
        for platform, task in history_tasks.items():
            try:
                platform_results = await task
                results[platform] = platform_results
                logger.info(f"Found {len(platform_results)} locations on {platform.value}")
                
            except Exception as e:
                logger.error(f"Error getting locations from {platform.value}: {str(e)}")
                results[platform] = []
        
        # Log event
        total_locations = sum(len(r) for r in results.values())
        self.audit_logger.log_data_event(
            "location_history_accessed",
            {
                "username": username,
                "platforms": [p.value for p in platforms],
                "total_locations": total_locations,
                "purpose": purpose.value
            }
        )
        
        return results
    
    def get_status(self) -> Dict[SocialMediaPlatform, ConnectionStatus]:
        """
        Get connection status for all platforms.
        
        Returns:
            Dictionary of platforms to connection status
        """
        status = {}
        for platform, adapter in self.adapters.items():
            status[platform] = adapter.status
        
        return status
    
    async def validate_usernames(self,
                               username: str,
                               platforms: List[SocialMediaPlatform]) -> Dict[SocialMediaPlatform, bool]:
        """
        Check if a username exists across multiple platforms.
        
        Args:
            username: Username to check
            platforms: List of platforms to check
            
        Returns:
            Dictionary of platforms to existence boolean
        """
        # Ensure connected
        if not self.connected:
            await self.connect(platforms)
        
        # Check all platforms concurrently
        check_tasks = {}
        for platform in platforms:
            adapter = self.adapters.get(platform)
            if adapter:
                check_tasks[platform] = adapter.get_profile(username)
        
        # Wait for all checks to complete
        results = {}
        for platform, task in check_tasks.items():
            try:
                profile = await task
                results[platform] = profile is not None
                
            except Exception as e:
                logger.error(f"Error checking username on {platform.value}: {str(e)}")
                results[platform] = False
        
        return results
