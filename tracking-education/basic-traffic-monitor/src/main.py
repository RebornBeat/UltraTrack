#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the Traffic Monitoring System.
Initializes all components, starts processing pipeline, and handles program lifecycle.
"""

import os
import sys
import time
import json
import signal
import logging
import argparse
import threading
from typing import Dict, List, Optional, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('traffic_monitor.log')
    ]
)
logger = logging.getLogger(__name__)

# Import system components
from capture.camera_interface import CameraInterface, CameraType, CameraConfig
from capture.video_stream import VideoStream, StreamStatus

from processing.background_subtraction import BackgroundSubtractor, SubtractionMethod, MotionDetector
from processing.vehicle_detection import VehicleDetector, DetectionModel, VehicleTracker
from processing.license_plate_recognition import LicensePlateDetector, LicensePlateRecognizer, PlateDetection

from analysis.traffic_counter import TrafficCounter, CountingZone
from analysis.speed_estimator import SpeedEstimator, CalibrationMethod
from analysis.flow_analyzer import FlowAnalyzer, TrafficDensity

from visualization.live_display import DisplayManager, DisplayMode, DisplayConfig
from visualization.traffic_report import ReportGenerator, ReportType, ChartType

from database.db_connect import init_db_connection, ConnectionParams
from database.data_models import init_database

class TrafficMonitoringSystem:
    """Main class for the Traffic Monitoring System."""

    def __init__(self, config_path: str):
        """
        Initialize the traffic monitoring system.
        
        Args:
            config_path: Path to configuration file directory
        """
        self.running = False
        self.config_path = config_path
        self.cameras = {}
        self.video_streams = {}
        self.processors = {}
        self.analyzers = {}
        
        # Threading and synchronization
        self.processing_threads = []
        self.stop_event = threading.Event()
        
        # Components
        self.display_manager = None
        self.db_connection = None
        
        # Load configuration
        self.load_configuration()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("Traffic Monitoring System initialized")
    
    def load_configuration(self):
        """Load system configuration files."""
        try:
            # Load camera configuration
            camera_config_path = os.path.join(self.config_path, "camera_config.json")
            with open(camera_config_path, 'r') as f:
                self.camera_config = json.load(f)
            
            # Load system settings
            system_settings_path = os.path.join(self.config_path, "system_settings.json")
            with open(system_settings_path, 'r') as f:
                self.system_settings = json.load(f)
                
            logger.info("Configuration loaded successfully")
            
            # Set log level from configuration
            log_level = self.system_settings.get('system', {}).get('log_level', 'INFO')
            numeric_level = getattr(logging, log_level, None)
            if isinstance(numeric_level, int):
                logging.getLogger().setLevel(numeric_level)
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def initialize_components(self):
        """Initialize system components based on configuration."""
        try:
            # Initialize cameras
            self.initialize_cameras()
            
            # Initialize database connection
            self.initialize_database()
            
            # Initialize processing pipeline
            self.initialize_processing()
            
            # Initialize analysis components
            self.initialize_analysis()
            
            # Initialize visualization
            self.initialize_visualization()
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            return False
    
    def initialize_cameras(self):
        """Initialize camera interfaces and video streams."""
        if 'cameras' not in self.camera_config:
            logger.error("No cameras defined in configuration")
            return
            
        for camera_info in self.camera_config['cameras']:
            if not camera_info.get('enabled', True):
                continue
                
            camera_id = camera_info['camera_id']
            logger.info(f"Initializing camera: {camera_id}")
            
            try:
                # Create camera configuration
                camera_type = CameraType[camera_info.get('camera_type', 'USB')]
                
                camera_config = CameraConfig(
                    camera_id=camera_id,
                    camera_type=camera_type,
                    width=camera_info.get('width', 1280),
                    height=camera_info.get('height', 720),
                    fps=camera_info.get('fps', 30),
                    rotation=camera_info.get('rotation', 0),
                    url=camera_info.get('url'),
                    username=camera_info.get('username'),
                    password=camera_info.get('password'),
                    exposure_mode=camera_info.get('exposure_mode', 'auto'),
                    white_balance=camera_info.get('white_balance', 'auto'),
                    flip_horizontal=camera_info.get('flip_horizontal', False),
                    flip_vertical=camera_info.get('flip_vertical', False),
                    buffer_size=camera_info.get('buffer_size', 5)
                )
                
                # Create camera interface
                camera = CameraInterface(camera_config)
                
                # Create video stream
                stream = VideoStream(camera, buffer_size=camera_config.buffer_size)
                
                # Store objects
                self.cameras[camera_id] = camera
                self.video_streams[camera_id] = stream
                
                logger.info(f"Camera {camera_id} initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize camera {camera_id}: {str(e)}")
    
    def initialize_database(self):
        """Initialize database connection and models."""
        db_settings = self.system_settings.get('storage', {}).get('database', {})
        
        if not db_settings.get('enabled', True):
            logger.info("Database storage disabled in configuration")
            return
            
        try:
            # Create connection parameters
            connection_params = ConnectionParams(
                host=db_settings.get('host', 'localhost'),
                port=db_settings.get('port', 5432),
                database=db_settings.get('database', 'traffic_monitor'),
                user=db_settings.get('user', 'traffic_user'),
                password=db_settings.get('password', '')
            )
            
            # Initialize database connection
            self.db_connection = init_db_connection(
                connection_params,
                min_connections=db_settings.get('min_connections', 1),
                max_connections=db_settings.get('max_connections', 10)
            )
            
            # Initialize database models
            db_string = f"postgresql://{connection_params.user}:{connection_params.password}@{connection_params.host}:{connection_params.port}/{connection_params.database}"
            init_database(db_string, create_tables=True)
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            logger.warning("Continuing without database connection")
    
    def initialize_processing(self):
        """Initialize processing components for each camera."""
        proc_settings = self.system_settings.get('processing', {})
        
        for camera_id, stream in self.video_streams.items():
            logger.info(f"Initializing processing pipeline for camera {camera_id}")
            
            camera_info = next((c for c in self.camera_config['cameras'] if c['camera_id'] == camera_id), None)
            if not camera_info:
                continue
                
            try:
                # Initialize processors dictionary for this camera
                self.processors[camera_id] = {}
                
                # Background subtraction
                bg_settings = proc_settings.get('background_subtraction', {})
                bg_subtractor = BackgroundSubtractor(
                    method=SubtractionMethod[bg_settings.get('method', 'MOG2')],
                    history=bg_settings.get('history', 500),
                    learning_rate=bg_settings.get('learning_rate', 0.01),
                    detect_shadows=bg_settings.get('detect_shadows', True),
                    threshold=bg_settings.get('threshold', 16),
                    kernel_size=bg_settings.get('kernel_size', 3)
                )
                
                motion_detector = MotionDetector(
                    subtractor=bg_subtractor,
                    min_motion_frames=proc_settings.get('min_motion_frames', 3),
                    max_motion_frames=proc_settings.get('max_motion_frames', 30),
                    cooldown_frames=proc_settings.get('cooldown_frames', 10)
                )
                
                # Vehicle detection and tracking
                det_settings = proc_settings.get('detection', {})
                vehicle_detector = VehicleDetector(
                    model_type=DetectionModel[det_settings.get('model_type', 'YOLO')],
                    model_path=det_settings.get('model_path'),
                    confidence_threshold=det_settings.get('confidence_threshold', 0.6),
                    nms_threshold=det_settings.get('nms_threshold', 0.4),
                    input_width=det_settings.get('input_width', 416),
                    input_height=det_settings.get('input_height', 416),
                    use_gpu=det_settings.get('use_gpu', True)
                )
                
                track_settings = proc_settings.get('tracking', {})
                vehicle_tracker = VehicleTracker(
                    max_distance=track_settings.get('max_distance', 100.0),
                    max_frames_to_skip=track_settings.get('max_frames_to_skip', 30),
                    max_trace_length=track_settings.get('max_trace_length', 50),
                    min_detection_confidence=track_settings.get('min_detection_confidence', 0.6)
                )
                
                # License plate recognition
                if proc_settings.get('license_plate', {}).get('enabled', True):
                    lpr_settings = proc_settings.get('license_plate', {})
                    plate_detector = LicensePlateDetector(
                        cascade_path=lpr_settings.get('detector_cascade_path'),
                        use_haar=True,
                        min_plate_width=lpr_settings.get('min_plate_width', 60),
                        min_plate_height=lpr_settings.get('min_plate_height', 20)
                    )
                    
                    plate_recognizer = LicensePlateRecognizer(
                        tesseract_config=lpr_settings.get('ocr_config', '--oem 1 --psm 8'),
                        confidence_threshold=lpr_settings.get('confidence_threshold', 0.7),
                        max_workers=lpr_settings.get('max_workers', 4)
                    )
                else:
                    plate_detector = None
                    plate_recognizer = None
                
                # Store processors
                self.processors[camera_id]['bg_subtractor'] = bg_subtractor
                self.processors[camera_id]['motion_detector'] = motion_detector
                self.processors[camera_id]['vehicle_detector'] = vehicle_detector
                self.processors[camera_id]['vehicle_tracker'] = vehicle_tracker
                self.processors[camera_id]['plate_detector'] = plate_detector
                self.processors[camera_id]['plate_recognizer'] = plate_recognizer
                
                logger.info(f"Processing pipeline initialized for camera {camera_id}")
                
            except Exception as e:
                logger.error(f"Failed to initialize processing for camera {camera_id}: {str(e)}")
    
    def initialize_analysis(self):
        """Initialize analysis components for each camera."""
        for camera_id, stream in self.video_streams.items():
            logger.info(f"Initializing analysis components for camera {camera_id}")
            
            camera_info = next((c for c in self.camera_config['cameras'] if c['camera_id'] == camera_id), None)
            if not camera_info:
                continue
                
            try:
                # Initialize analyzers dictionary for this camera
                self.analyzers[camera_id] = {}
                
                # Traffic counting
                if 'zones' in camera_info:
                    counting_zones = []
                    for zone_info in camera_info['zones']:
                        zone = CountingZone(
                            id=zone_info['id'],
                            name=zone_info['name'],
                            polygon=zone_info['polygon'],
                            direction=zone_info.get('direction'),
                            vehicle_types=[VehicleType[vt] for vt in zone_info.get('vehicle_types', [])] if 'vehicle_types' in zone_info else None
                        )
                        counting_zones.append(zone)
                    
                    count_settings = self.system_settings.get('analysis', {}).get('counting', {})
                    traffic_counter = TrafficCounter(
                        zones=counting_zones,
                        min_detection_frames=count_settings.get('min_detection_frames', 3),
                        min_zone_frames=count_settings.get('min_zone_frames', 2),
                        count_timeout=count_settings.get('count_timeout', 5.0),
                        store_history=count_settings.get('store_history', True),
                        max_history_items=count_settings.get('max_history_items', 10000)
                    )
                else:
                    traffic_counter = None
                
                # Speed estimation
                speed_settings = self.system_settings.get('analysis', {}).get('speed_estimation', {})
                calibration = camera_info.get('calibration', {})
                speed_lines = camera_info.get('speed_lines', [])
                
                if calibration and 'method' in calibration:
                    speed_estimator = SpeedEstimator(
                        fps=camera_info.get('fps', 30),
                        calibration_method=CalibrationMethod[calibration.get('method', 'PIXEL_DISTANCE')],
                        calibration_params=calibration,
                        min_tracking_points=speed_settings.get('min_tracking_points', 5),
                        smoothing_factor=speed_settings.get('smoothing_factor', 0.3),
                        min_speed_confidence=speed_settings.get('min_speed_confidence', 0.7),
                        speed_lines=[(line['points'], line['distance_meters']) for line in speed_lines] if speed_lines else None,
                        store_history=speed_settings.get('store_history', True),
                        max_history_items=speed_settings.get('max_history_items', 10000)
                    )
                else:
                    speed_estimator = None
                
                # Traffic flow analysis
                flow_settings = self.system_settings.get('analysis', {}).get('flow', {})
                frame_size = (camera_info.get('width', 1280), camera_info.get('height', 720))
                
                flow_analyzer = FlowAnalyzer(
                    frame_size=frame_size,
                    regions={zone['id']: zone['polygon'] for zone in camera_info.get('zones', [])} if 'zones' in camera_info else None,
                    flow_history_length=flow_settings.get('flow_history_length', 150),
                    update_interval=flow_settings.get('update_interval', 30),
                    store_history=flow_settings.get('store_history', True),
                    max_history_items=flow_settings.get('max_history_items', 5000)
                )
                
                # Store analyzers
                self.analyzers[camera_id]['traffic_counter'] = traffic_counter
                self.analyzers[camera_id]['speed_estimator'] = speed_estimator
                self.analyzers[camera_id]['flow_analyzer'] = flow_analyzer
                
                logger.info(f"Analysis components initialized for camera {camera_id}")
                
            except Exception as e:
                logger.error(f"Failed to initialize analysis for camera {camera_id}: {str(e)}")
    
    def initialize_visualization(self):
        """Initialize visualization components."""
        vis_settings = self.system_settings.get('visualization', {})
        
        if vis_settings.get('display', {}).get('enabled', True):
            display_config = DisplayConfig(
                show_boxes=vis_settings.get('display', {}).get('show_boxes', True),
                show_labels=vis_settings.get('display', {}).get('show_labels', True),
                show_ids=vis_settings.get('display', {}).get('show_ids', True),
                show_trails=vis_settings.get('display', {}).get('show_trails', True),
                show_zones=vis_settings.get('display', {}).get('show_zones', True),
                show_speed=vis_settings.get('display', {}).get('show_speed', True),
                show_flow=vis_settings.get('display', {}).get('show_flow', True),
                show_plates=vis_settings.get('display', {}).get('show_plates', True),
                window_name=vis_settings.get('display', {}).get('window_name', "Traffic Monitoring System"),
                fullscreen=vis_settings.get('display', {}).get('fullscreen', False),
                display_fps=vis_settings.get('display', {}).get('display_fps', 30),
                overlay_alpha=vis_settings.get('display', {}).get('overlay_alpha', 0.4),
                font_scale=vis_settings.get('display', {}).get('font_scale', 0.7),
                line_thickness=vis_settings.get('display', {}).get('line_thickness', 2)
            )
            
            self.display_manager = DisplayManager(config=display_config)
            logger.info("Display manager initialized")
    
    def start(self):
        """Start the traffic monitoring system."""
        if self.running:
            logger.warning("System is already running")
            return False
            
        logger.info("Starting Traffic Monitoring System")
        
        # Initialize all components
        if not self.initialize_components():
            logger.error("Failed to initialize components, system not started")
            return False
        
        # Start display manager if configured
        if self.display_manager:
            self.display_manager.start()
        
        # Start video streams
        for camera_id, stream in self.video_streams.items():
            if not stream.start():
                logger.error(f"Failed to start video stream for camera {camera_id}")
            else:
                logger.info(f"Started video stream for camera {camera_id}")
        
        # Start processing threads
        for camera_id in self.video_streams:
            thread = threading.Thread(
                target=self.process_camera_feed,
                args=(camera_id,),
                daemon=True
            )
            thread.start()
            self.processing_threads.append(thread)
            logger.info(f"Started processing thread for camera {camera_id}")
        
        self.running = True
        logger.info("Traffic Monitoring System started")
        return True
    
    def process_camera_feed(self, camera_id):
        """
        Process video feed from a camera.
        
        Args:
            camera_id: Camera identifier
        """
        logger.info(f"Processing thread started for camera {camera_id}")
        
        stream = self.video_streams[camera_id]
        processors = self.processors[camera_id]
        analyzers = self.analyzers[camera_id]
        
        bg_subtractor = processors.get('bg_subtractor')
        motion_detector = processors.get('motion_detector')
        vehicle_detector = processors.get('vehicle_detector')
        vehicle_tracker = processors.get('vehicle_tracker')
        plate_detector = processors.get('plate_detector')
        plate_recognizer = processors.get('plate_recognizer')
        
        traffic_counter = analyzers.get('traffic_counter')
        speed_estimator = analyzers.get('speed_estimator')
        flow_analyzer = analyzers.get('flow_analyzer')
        
        frame_count = 0
        detections = []
        tracked_objects = []
        speed_measurements = {}
        plate_detections = []
        
        while not self.stop_event.is_set() and stream.status == StreamStatus.RUNNING:
            # Read frame from stream
            ret, frame, timestamp = stream.read()
            
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            start_time = time.time()
            frame_count += 1
            
            try:
                # Motion detection
                if bg_subtractor and motion_detector:
                    motion_detected, motion_boxes = motion_detector.detect(frame)
                    
                    # Skip further processing if no motion detected and we have recent data
                    if not motion_detected and frame_count % 10 != 0 and tracked_objects:
                        if self.display_manager:
                            self.display_manager.update(
                                frame, 
                                detections=detections, 
                                tracks=tracked_objects,
                                zones=traffic_counter.zones if traffic_counter else None,
                                speeds=speed_measurements,
                                plates=plate_detections,
                                processing_time=(time.time() - start_time) * 1000
                            )
                        continue
                
                # Vehicle detection
                if vehicle_detector:
                    detections = vehicle_detector.detect(frame)
                
                # Vehicle tracking
                if vehicle_tracker:
                    tracked_objects = vehicle_tracker.update(detections)
                
                # License plate recognition
                plate_detections = []
                if plate_detector and plate_recognizer:
                    # Only run license plate detection every few frames to save processing time
                    if frame_count % 5 == 0:
                        plate_boxes = plate_detector.detect(frame, [(d.box if hasattr(d, 'box') else d['box']) for d in tracked_objects])
                        if plate_boxes:
                            plate_images = plate_detector.extract_plate_images(frame, plate_boxes)
                            for plate_image, box in plate_images:
                                text, confidence = plate_recognizer.process_plate(plate_image)
                                if text:
                                    plate_detections.append(PlateDetection(box, confidence, text, plate_image, timestamp=timestamp))
                
                # Traffic counting
                if traffic_counter:
                    new_counts = traffic_counter.process_frame(tracked_objects, (frame.shape[1], frame.shape[0]))
                
                # Speed estimation
                if speed_estimator:
                    speeds = speed_estimator.process_frame(tracked_objects, timestamp)
                    speed_measurements = speed_estimator.get_current_speeds()
                
                # Traffic flow analysis
                if flow_analyzer:
                    flow_data = flow_analyzer.process_frame(tracked_objects, speed_measurements)
                
                # Update visualization
                if self.display_manager:
                    self.display_manager.update(
                        frame, 
                        detections=detections, 
                        tracks=tracked_objects,
                        zones=traffic_counter.zones if traffic_counter else None,
                        speeds=speed_measurements,
                        plates=plate_detections,
                        flow_data=flow_analyzer.get_current_flow() if flow_analyzer else None,
                        processing_time=(time.time() - start_time) * 1000
                    )
            
            except Exception as e:
                logger.error(f"Error processing frame for camera {camera_id}: {str(e)}")
        
        logger.info(f"Processing thread stopped for camera {camera_id}")
    
    def stop(self):
        """Stop the traffic monitoring system."""
        if not self.running:
            return
            
        logger.info("Stopping Traffic Monitoring System")
        
        # Signal threads to stop
        self.stop_event.set()
        
        # Stop video streams
        for camera_id, stream in self.video_streams.items():
            stream.stop()
            logger.info(f"Stopped video stream for camera {camera_id}")
        
        # Wait for processing threads to finish
        for thread in self.processing_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        # Stop display manager
        if self.display_manager:
            self.display_manager.stop()
        
        # Close database connection
        if self.db_connection:
            self.db_connection.close()
        
        self.running = False
        logger.info("Traffic Monitoring System stopped")
    
    def signal_handler(self, sig, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {sig}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def run(self):
        """Run the system until stopped."""
        if not self.start():
            return
            
        try:
            # Keep main thread alive until stopped
            while self.running and not self.stop_event.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
        finally:
            self.stop()


def main():
    """Main entry point for the traffic monitoring system."""
    parser = argparse.ArgumentParser(description='Traffic Monitoring System')
    parser.add_argument('--config', type=str, default='config',
                      help='Path to configuration directory')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set log level based on arguments
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run traffic monitoring system
    system = TrafficMonitoringSystem(args.config)
    system.run()


if __name__ == "__main__":
    main()
