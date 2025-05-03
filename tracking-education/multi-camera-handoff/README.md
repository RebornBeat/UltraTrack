# Multi-Camera Object Tracking System

An educational project that teaches the fundamentals of tracking objects across multiple camera views with seamless handoff - a critical concept in advanced surveillance and computer vision systems.

## Overview

This project demonstrates how to build a system capable of:
- Tracking objects across multiple camera views
- Maintaining consistent object identity between cameras
- Handling camera view overlaps and transitions
- Creating a unified coordinate system across cameras
- Visualizing object trajectories in a combined view

## Educational Objectives

- Learn camera calibration techniques
- Understand object re-identification across views
- Implement tracking handoff algorithms
- Create unified spatial coordinate systems
- Design multi-view visualization systems
- Develop predictive tracking capabilities

## Components Required

- 3+ Raspberry Pi 4 devices (4GB+ RAM recommended)
- 3+ Raspberry Pi Camera Modules or USB webcams
- Network switch for camera communication
- MicroSD cards (16GB+)
- Power supplies
- Optional: Server/PC for central processing

## Installation

Follow the [Installation Guide](docs/installation_guide.md) for detailed setup instructions including network configuration.

## Usage

1. Position cameras with overlapping fields of view
2. Run the calibration process: `python -m src.calibration.calibrate_cameras`
3. Start the tracking system: `python -m src.main`
4. Access the visualization dashboard at http://server-ip:8050

## Key Features

- **Camera Calibration**: Automatic detection of spatial relationships between cameras
- **Seamless Handoff**: Maintains object identity as objects move between camera views
- **Consistent Tracking IDs**: Objects retain the same ID across the entire camera network
- **Global Coordinate System**: Maps all camera views to a unified spatial representation
- **Trajectory Visualization**: Shows complete object paths across multiple cameras
- **Blind Spot Inference**: Predicts object location in non-covered areas

## Project Extensions

- Implement appearance-based re-identification
- Add predictive trajectory analysis
- Integrate with other sensor types (infrared, depth)
- Develop 3D space reconstruction from multiple views
- Create an alert system for tracked objects of interest

## Exploration Questions

1. How does camera placement affect the success rate of tracking handoffs?
2. What features are most reliable for re-identifying objects across different camera views?
3. How can the system handle blind spots between camera coverages?
4. What methods improve tracking accuracy in crowded scenes?
5. How does the number of cameras affect system performance and accuracy?
6. What algorithms are most effective for predicting object trajectories?

## Ethical Guidelines

- This system should only be used with appropriate permissions in controlled environments
- Focus on tracking non-personal objects for educational purposes
- When tracking people, ensure informed consent and anonymization
- Consider privacy implications of any tracking system deployment
- Use only for educational and research purposes

## License

MIT License - See LICENSE file for details.
