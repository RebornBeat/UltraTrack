"""
Database models for the Traffic Monitoring System.
Provides SQLAlchemy models for storing and retrieving data.
"""

import time
import datetime
import logging
from typing import Dict, List, Optional, Any, Union

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Enum, ForeignKey, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, scoped_session
from sqlalchemy.dialects.postgresql import JSONB

# Configure logger for this module
logger = logging.getLogger(__name__)

# Create the SQLAlchemy base class
Base = declarative_base()


class Vehicle(Base):
    """Vehicle model for storing vehicle information."""
    __tablename__ = 'vehicles'
    
    id = Column(Integer, primary_key=True)
    vehicle_type = Column(String(50), nullable=False)
    first_seen = Column(DateTime, default=datetime.datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    plate_number = Column(String(20), nullable=True, index=True)
    metadata = Column(JSONB, nullable=True)
    
    # Relationships
    detections = relationship("DetectionEvent", back_populates="vehicle")
    counts = relationship("CountEvent", back_populates="vehicle")
    speed_measurements = relationship("SpeedMeasurement", back_populates="vehicle")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'vehicle_type': self.vehicle_type,
            'first_seen': self.first_seen.isoformat() if self.first_seen else None,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'plate_number': self.plate_number,
            'metadata': self.metadata
        }


class DetectionEvent(Base):
    """Detection event model for storing vehicle detections."""
    __tablename__ = 'detection_events'
    
    id = Column(Integer, primary_key=True)
    vehicle_id = Column(Integer, ForeignKey('vehicles.id'), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    x = Column(Integer, nullable=False)
    y = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    camera_id = Column(String(50), nullable=False, index=True)
    frame_index = Column(Integer, nullable=True)
    has_plate = Column(Boolean, default=False)
    
    # Relationships
    vehicle = relationship("Vehicle", back_populates="detections")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'vehicle_id': self.vehicle_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'position': (self.x, self.y, self.width, self.height),
            'confidence': self.confidence,
            'camera_id': self.camera_id,
            'frame_index': self.frame_index,
            'has_plate': self.has_plate
        }


class CountEvent(Base):
    """Count event model for storing vehicle count events."""
    __tablename__ = 'count_events'
    
    id = Column(Integer, primary_key=True)
    vehicle_id = Column(Integer, ForeignKey('vehicles.id'), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    zone_id = Column(String(50), nullable=False, index=True)
    direction = Column(String(50), nullable=True)
    speed = Column(Float, nullable=True)
    camera_id = Column(String(50), nullable=False, index=True)
    
    # Relationships
    vehicle = relationship("Vehicle", back_populates="counts")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'vehicle_id': self.vehicle_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'zone_id': self.zone_id,
            'direction': self.direction,
            'speed': self.speed,
            'camera_id': self.camera_id
        }


class SpeedMeasurement(Base):
    """Speed measurement model for storing vehicle speed measurements."""
    __tablename__ = 'speed_measurements'
    
    id = Column(Integer, primary_key=True)
    vehicle_id = Column(Integer, ForeignKey('vehicles.id'), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    speed = Column(Float, nullable=False)
    x = Column(Integer, nullable=False)
    y = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    camera_id = Column(String(50), nullable=False, index=True)
    method = Column(String(50), nullable=True)
    
    # Relationships
    vehicle = relationship("Vehicle", back_populates="speed_measurements")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'vehicle_id': self.vehicle_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'speed': self.speed,
            'position': (self.x, self.y),
            'confidence': self.confidence,
            'camera_id': self.camera_id,
            'method': self.method
        }


class TrafficFlow(Base):
    """Traffic flow model for storing traffic flow data."""
    __tablename__ = 'traffic_flows'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    region_id = Column(String(50), nullable=False, index=True)
    density = Column(String(50), nullable=False)
    vehicle_count = Column(Integer, nullable=False)
    avg_speed = Column(Float, nullable=True)
    dominant_direction = Column(String(50), nullable=True)
    directional_counts = Column(JSONB, nullable=True)
    camera_id = Column(String(50), nullable=False, index=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'region_id': self.region_id,
            'density': self.density,
            'vehicle_count': self.vehicle_count,
            'avg_speed': self.avg_speed,
            'dominant_direction': self.dominant_direction,
            'directional_counts': self.directional_counts,
            'camera_id': self.camera_id
        }


# Database connection and session management
_engine = None
_session_factory = None
_Session = None


def init_database(
    connection_string: str,
    create_tables: bool = False,
    echo: bool = False
) -> None:
    """
    Initialize the database connection.
    
    Args:
        connection_string: SQLAlchemy connection string
        create_tables: Whether to create tables if they don't exist
        echo: Whether to echo SQL statements
    """
    global _engine, _session_factory, _Session
    
    try:
        # Create engine
        _engine = create_engine(connection_string, echo=echo)
        
        # Create session factory
        _session_factory = sessionmaker(bind=_engine)
        _Session = scoped_session(_session_factory)
        
        # Create tables if requested
        if create_tables:
            Base.metadata.create_all(_engine)
            logger.info("Created database tables")
        
        logger.info(f"Initialized database connection to {connection_string}")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        _engine = None
        _session_factory = None
        _Session = None
        raise


def get_session():
    """
    Get a database session.
    
    Returns:
        SQLAlchemy session
    """
    global _Session
    
    if not _Session:
        raise RuntimeError("Database not initialized")
    
    return _Session()


def close_sessions():
    """Close all database sessions."""
    global _Session
    
    if _Session:
        _Session.remove()
        logger.info("Closed all database sessions")
