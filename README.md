# HMAP (Halo's Main Autonomy Piece)

[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-NVIDIA%20Drive%20AGX%20Orin-green.svg)](https://developer.nvidia.com/drive/hardware)
[![API](https://img.shields.io/badge/API-DriveWorks%205.x-blue.svg)](https://developer.nvidia.com/driveworks)

## Overview

HMAP (Halo's Main Autonomy Piece) is Halo's proprietary autonomous driving software platform, designed specifically for next-generation self-driving vehicles. This repository contains the source code archive for core perception modules, proprietary drivers, and application components that power Halo's autonomous driving capabilities on NVIDIA Drive AGX Orin SOC platforms.

This repository serves as the primary source code storage and reference for HMAP's perception system, containing all proprietary modules and drivers developed for autonomous driving applications.

**Repository:** [https://github.com/halo-drive/HMAP-Perception-Source](https://github.com/halo-drive/HMAP-Perception-Source)

## About Halo

Halo is a cutting-edge startup focused on developing advanced autonomous driving solutions. Our mission is to accelerate the deployment of safe, reliable, and scalable autonomous vehicles through innovative software and AI technologies.

## Features

### Core Capabilities
- **Real-time Perception Processing** - Advanced computer vision and sensor fusion algorithms
- **Multi-Modal Sensor Integration** - Support for cameras, LiDAR, radar, and IMU sensors
- **High-Performance Computing** - Optimized for NVIDIA Drive AGX Orin architecture
- **Modular Architecture** - Scalable and configurable perception pipeline
- **Safety-Critical Design** - ASIL-compliant development practices

### Key Modules
- **Vision Processing Engine** - Real-time object detection and tracking
- **Sensor Fusion Module** - Multi-sensor data integration and correlation
- **Mapping & Localization** - SLAM and HD map integration
- **Prediction & Planning Interface** - Seamless integration with planning systems
- **Hardware Abstraction Layer** - Unified interface for various sensor configurations

## System Requirements

### Hardware Prerequisites
- **Platform:** NVIDIA Drive AGX Orin SOC
- **Memory:** Minimum 32GB RAM
- **Storage:** 256GB+ NVMe SSD
- **Sensors:** Compatible camera, LiDAR, and radar sensors

### Software Dependencies
- **Operating System:** NVIDIA Drive OS 6.0+
- **DriveWorks SDK:** Version 5.20+
- **CUDA:** Version 11.4
- **TensorRT:** Version 8.6.13

## Repository Structure

This repository serves as the primary source code storage for HMAP's perception modules and proprietary drivers. The codebase is organized for reference and development purposes.

### Directory Structure
- **`/dataandrig/`** - cotnains jsons for sensor and vehcile suite with their respective transformations 
- **`/logs/`** - application specific log preservation directory 
- **`/modules/`** - Public API headers and interfaces for each module 
- **`/plugins/`** - Drivers and plugins for custom sensors and applications 
- **`/py-postprocessed/`** - contains skeleton implimentations of post-postprocessing needed to be ported into native APIs (for debug purposed on DL models)
- **`/standaloneapps/`** - Unit applications in each sensor category suite (full functional application in native APIs)

## Source Code Documentation

### Module Architecture
The source code is organized into distinct modules that handle specific aspects of the autonomous driving perception stack:

- **Vision Processing** - Computer vision algorithms and object detection
- **Sensor Fusion** - Multi-modal sensor data integration
- **Localization** - SLAM and positioning algorithms  
- **Hardware Abstraction** - Driver interfaces for various sensors


## Performance Benchmarks

| Configuration | Latency | Throughput | Power Consumption |
|---------------|---------|------------|-------------------|
| 4x Cameras    | <50ms   | 30 FPS     | 45W               |
| Full Sensor Suite | <100ms | 15 FPS | 75W               |

## Architecture Overview

HMAP's perception system is built on a modular architecture optimized for real-time processing on NVIDIA Drive AGX Orin hardware. The source code implements advanced computer vision algorithms and sensor fusion techniques using DriveWorks APIs for maximum performance and reliability.

## Contributing
Contributions are accepted from Halo engineers forking the main branch and pushing code to their native branches, the API mis-matches are verified by @atharv-sharma and acccepted to the production by @sudoashwin, this cycle is performed every 10th of the month. 

### Development Guidelines
- Follow Halo's coding standards (see [CODING_STANDARDS.md](CODING_STANDARDS.md) (to be provided))
- All commits must pass CI/CD pipeline
- Safety-critical code requires additional review
- Performance regressions are not acceptable

## Security & Compliance

This repository contains proprietary and confidential information. Access is restricted to authorized Halo personnel and partners under appropriate NDAs.

- **ASIL Compliance:** Developed according to ISO 26262 standards
- **Cybersecurity:** Implements automotive cybersecurity best practices
- **Data Protection:** Adheres to privacy regulations for autonomous vehicle data

## Support & Contact

### Maintainers
- **Engineer:** [@sudoashwin](ashwin@drivehalo.com)

## License

This software is proprietary and confidential to Halo corporation. Unauthorized reproduction, distribution, or reverse engineering is strictly prohibited.

**Copyright © 2025 Halo. All rights reserved.**

---

## Version History

| Version | Release Date | Key Features |
|---------|-------------|--------------|
| v2.2.0  | 2025-11-04  | Core perceptin algorithms are ready for realtime detections |
| v2.1.0  | 2024-09-15  | Enhanced LiDAR integration, performance optimizations |
| v2.0.0  | 2025-07-01  | Major architecture refactor, DriveWorks 6.0.10 support |
| v1.5.0  | 2025-03-15  | Multi-camera support, improved latency |
| v1.0.0  | 2024-08-01  | Initial production release |

---

**Built by the Halo Engineering Team**
