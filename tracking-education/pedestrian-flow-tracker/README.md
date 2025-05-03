# Pedestrian Flow Tracker

An educational project for tracking pedestrian movement patterns while maintaining privacy. Ideal for learning about computer vision, tracking algorithms, and ethical considerations in surveillance systems.

## Overview

This system enables students to build a privacy-respecting pedestrian tracking system capable of:
- Detecting and counting people in video streams
- Tracking movement patterns and flow directions
- Measuring dwell time in defined zones
- Generating heat maps of pedestrian activity
- Analyzing crowd density and distribution
- Respecting privacy through automatic anonymization

## Educational Objectives

- Learn human detection techniques in computer vision
- Implement tracking algorithms for maintaining identity
- Understand privacy-preserving techniques in surveillance
- Design data visualization for movement patterns
- Analyze time-series movement data
- Consider ethical implications of people-tracking systems

## Components Required

- Raspberry Pi 4 (4GB+ RAM recommended)
- Raspberry Pi Camera Module V2 or USB webcam
- MicroSD card (16GB+)
- Power supply
- Optional: Additional cameras for multi-angle coverage
- Optional: Neural compute stick for improved performance

## Installation

Follow the [Installation Guide](docs/installation_guide.md) for step-by-step setup instructions.

## Usage

1. Mount cameras in the areas to be monitored
2. Configure zones of interest in `config/zones_config.json`
3. Start the system: `python -m src.main`
4. Access the dashboard at http://localhost:8050

## Privacy Features

This project implements several privacy-preserving features:
- Automatic face blurring in all stored images and video
- No collection of biometric or identifying information
- Analysis of aggregate movement rather than individuals
- Configurable privacy zones where no tracking occurs
- All data stored locally, with automatic deletion

## Project Extensions

- Implement multi-camera tracking across spaces
- Add path prediction capabilities
- Develop congestion prediction algorithms
- Integrate with building management systems
- Create an alert system for unusual crowd density

## Exploration Questions

1. How does camera placement affect the accuracy of pedestrian detection?
2. What techniques can improve tracking when people are partially occluded?
3. How do different anonymization methods impact tracking performance?
4. What patterns emerge in pedestrian movement in different environments?
5. How could this system be used to improve building design or emergency evacuation?

## Ethical Guidelines

This project emphasizes ethical considerations in tracking systems:
- Always inform people they are entering an area with movement tracking
- Never use the system for identifying or profiling individuals
- Ensure all stored data is anonymized and protected
- Consider accessibility impacts when analyzing movement patterns
- Use findings only for educational purposes and space improvement

## License

MIT License - See LICENSE file for details.
