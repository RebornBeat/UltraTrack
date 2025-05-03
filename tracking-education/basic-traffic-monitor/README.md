# Basic Traffic Monitoring System

A student-focused educational project for building a simple traffic monitoring system using a Raspberry Pi, camera module, and computer vision techniques.

## Overview

This project demonstrates how to build a basic traffic monitoring system capable of:
- Detecting vehicles in video streams
- Counting traffic flow
- Estimating vehicle speed
- Basic license plate recognition
- Storing traffic data in a database
- Visualizing traffic patterns

## Educational Objectives

- Learn fundamentals of computer vision
- Understand basic object detection techniques
- Implement simple tracking algorithms
- Work with camera hardware and Raspberry Pi
- Store and analyze time-series data
- Create visualization dashboards

## Components

The system consists of several modular components:

1. **Camera Capture Module**: Interfaces with various camera sources including USB webcams, IP cameras, RTSP streams, Raspberry Pi cameras, or video files.
2. **Processing Module**: Performs background subtraction, vehicle detection, tracking, and license plate recognition.
3. **Analysis Module**: Analyzes traffic flow, counts vehicles, and estimates speeds.
4. **Database Module**: Stores and retrieves traffic data for historical analysis.
5. **Visualization Module**: Provides real-time display and generates reports.

## Components Required

- Raspberry Pi 4 (2GB+ RAM) or any computer with Python support
- Camera (options include):
  - Raspberry Pi Camera Module
  - USB webcam
  - IP camera
  - RTSP camera
  - Video file (for testing)
- MicroSD card (16GB+)
- Power supply
- Optional: Weather-resistant case for outdoor deployment

## Installation

### Prerequisites

- Python 3.8 or higher
- PostgreSQL database (for data storage)
- Tesseract OCR (for license plate recognition)

### Dependencies Installation

```bash
# Install required Python packages
pip install -r requirements.txt

# Install PostgreSQL on Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# Install Tesseract OCR for license plate recognition
sudo apt-get install tesseract-ocr
```

### Database Setup

```bash
# Create database and user
sudo -u postgres psql

postgres=# CREATE DATABASE traffic_monitor;
postgres=# CREATE USER traffic_user WITH PASSWORD 'your_secure_password';
postgres=# GRANT ALL PRIVILEGES ON DATABASE traffic_monitor TO traffic_user;
postgres=# \q
```

Don't forget to update the password in `src/config/system_settings.json`.

### Model Files Setup

Download the required model files:

1. For vehicle detection:
   - Download YOLOv4 weights and config from: https://github.com/AlexeyAB/darknet/releases
   - Place in `src/models/yolo/` directory:
     - `yolov4.weights`
     - `yolov4.cfg`
     - `coco.names`

2. For license plate detection:
   - Download Haar cascade classifier from: https://github.com/opencv/opencv/tree/master/data/haarcascades
   - Place in `src/models/haarcascades/` directory:
     - `haarcascade_russian_plate_number.xml`

### Directory Structure Setup

Create the necessary directories if they don't exist:

```bash
mkdir -p src/models/yolo
mkdir -p src/models/haarcascades
mkdir -p data/recordings
mkdir -p data/exports
mkdir -p data/reports
```

## Configuration

### Camera Configuration

The system supports multiple camera configurations as demonstrated in `src/config/camera_config.json`. You can use a single camera or multiple cameras based on your needs.

Options include:
- USB webcam
- Raspberry Pi Camera
- IP cameras
- RTSP streams
- Video files (for testing)

For testing without physical cameras, modify the camera configuration to use a video file:

```json
{
  "camera_id": "test_camera",
  "name": "Test Camera",
  "camera_type": "FILE",
  "url": "path/to/test/video.mp4",
  "width": 1280,
  "height": 720,
  "fps": 30,
  "enabled": true
}
```

### System Settings

The `src/config/system_settings.json` file contains configurations for:
- Processing parameters
- Analysis settings
- Database connection
- Visualization options

Adjust these settings as needed for your hardware and requirements.

## Usage

1. Position the camera to overlook a traffic area
2. Configure the system parameters in `config/system_settings.json` and `config/camera_config.json`
3. Run the system:

```bash
python -m src.main
```

4. Access the visualization dashboard at http://localhost:8050 (if web dashboard is enabled)

### Command Line Arguments

```bash
python -m src.main --config config --debug
```

- `--config`: Path to configuration directory (default: 'config')
- `--debug`: Enable debug logging

## Project Extensions

- Add traffic light timing optimization
- Implement cloud data backup
- Create time-of-day traffic pattern analysis
- Add vehicle classification (car, truck, motorcycle, etc.)
- Implement multi-camera integration

## Exploration Questions

1. How does the positioning of the camera affect detection accuracy?
2. How do weather conditions affect the system's performance?
3. What privacy considerations should be taken into account when deploying such a system?
4. How could this system be integrated with traffic light control?
5. What is the maximum distance at which the system can reliably detect vehicles?
6. How do lighting conditions affect performance throughout the day?
7. Can the system be trained to recognize specific vehicle types or models?
8. How could multiple camera systems be coordinated to track vehicles across larger areas?
9. What methods could be used to anonymize license plate data while still tracking flow?
10. How accurate is speed estimation, and what factors improve or reduce accuracy?
11. What is the computational bottleneck in the system, and how could it be optimized?
12. How could this system be integrated with existing traffic management infrastructure?
13. What privacy and ethical considerations arise when implementing traffic monitoring?

## Troubleshooting

### Camera Issues
- Ensure you have the correct permissions to access camera devices
- For USB cameras, check the device index (usually 0 for the first camera)
- For IP/RTSP cameras, verify the URL, username, and password
- For Raspberry Pi cameras, ensure the camera is enabled with `sudo raspi-config`

### Database Issues
- Verify PostgreSQL is running: `sudo systemctl status postgresql`
- Check database connection parameters in `system_settings.json`
- Ensure the database user has proper permissions

### Processing Issues
- Verify model files are in the correct location
- Check CPU/GPU utilization with `htop` or similar tools
- Adjust processing parameters for lower-powered hardware

## Ethical Guidelines

This project is designed for educational purposes only. When deploying any camera system:
- Ensure you have permission to monitor the area
- Do not store personally identifiable information
- Consider privacy implications and local regulations
- Use data only for educational analysis

## License

MIT License - See LICENSE file for details.
