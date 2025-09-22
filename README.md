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
- **`/src/`** - Core perception module source code
- **`/drivers/`** - Hardware abstraction layer and sensor drivers  
- **`/include/`** - Public API headers and interfaces
- **`/config/`** - Configuration templates and schemas
- **`/docs/`** - Technical documentation and specifications
- **`/tests/`** - Unit and integration test suites

## Source Code Documentation

### Core Components
- **Perception API** - Main interface definitions for perception processing
- **Sensor Manager** - Hardware sensor abstraction and management layers
- **Data Pipeline** - Streaming data processing and buffering implementations
- **Calibration API** - Sensor calibration and alignment utilities

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

### Development Guidelines
- Follow Halo's coding standards (see [CODING_STANDARDS.md](CODING_STANDARDS.md) (to be provided))
- All commits must pass CI/CD pipeline
- Safety-critical code requires additional review
- Performance regressions are not acceptable

### Contribution Process
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## Security & Compliance

This repository contains proprietary and confidential information. Access is restricted to authorized Halo personnel and partners under appropriate NDAs.

- **ASIL Compliance:** Developed according to ISO 26262 standards
- **Cybersecurity:** Implements automotive cybersecurity best practices
- **Data Protection:** Adheres to privacy regulations for autonomous vehicle data

## Support & Contact

### Maintainers
- **Engineer:** [@sudoashwin](ashwin@drivehalo.com)

## License

This software is proprietary and confidential to Halo. Unauthorized reproduction, distribution, or reverse engineering is strictly prohibited.

**Copyright © 2024 Halo. All rights reserved.**

---

## Version History

| Version | Release Date | Key Features |
|---------|-------------|--------------|
| v2.1.0  | 2024-09-15  | Enhanced LiDAR integration, performance optimizations |
| v2.0.0  | 2024-08-01  | Major architecture refactor, DriveWorks 5.8 support |
| v1.5.0  | 2024-06-15  | Multi-camera support, improved latency |
| v1.0.0  | 2024-04-01  | Initial production release |

---

**Built by the Halo Engineering Team**
