# SLAM Pipeline - Fast-LIO on NVIDIA DriveWorks

This repository contains a SLAM (Simultaneous Localization and Mapping) pipeline implementation using LiDAR + IMU + GPS/RTK sensors, built around the Fast-LIO algorithm integrated with NVIDIA DriveWorks SDK.

## Overview

The SLAM system provides real-time mapping and localization capabilities for autonomous vehicles and robotics applications. It combines:
- **LiDAR** for high-resolution 3D point cloud data
- **IMU** for high-frequency motion estimation
- **GPS/RTK** for global positioning and loop closure

The implementation features a pure DriveWorks-native solution without external Python or ZeroCM dependencies, making it suitable for production deployment on NVIDIA DRIVE platforms.

## Features

- ✅ Real-time LiDAR-inertial odometry using Fast-LIO algorithm
- ✅ Multi-sensor fusion (LiDAR, IMU, GPS/RTK)
- ✅ Thread-safe data processing with mutex locks
- ✅ DriveWorks-native visualization
- ✅ Map building and saving capabilities
- ✅ Localization mode (load pre-built maps)
- ✅ Type conversion layer between Fast-LIO and DriveWorks datatypes

## Dependencies

### Required Libraries
- **NVIDIA DriveWorks SDK** - Core framework for sensor handling and visualization
- **PCL (Point Cloud Library)** 1.10+ - Point cloud processing
- **Eigen3** - Linear algebra and transformations
- **VTK** 7.1+ - Visualization support (via PCL)
- **Boost** - Required by PCL
- **GLFW** - Window management for visualization


## Building

### Prerequisites
Ensure all dependencies are installed and DriveWorks SDK is properly configured in your environment.

### Build Instructions

The build system automatically detects the architecture (x86_64 or aarch64) and configures paths accordingly.


### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--lidar-protocol` | LiDAR sensor protocol (e.g., `lidar.virtual`) | `lidar.virtual` |
| `--lidar-params` | LiDAR sensor parameters (e.g., `file=/path/to/lidar.bin`) | Required |
| `--imu-protocol` | IMU sensor protocol (e.g., `imu.virtual`) | `imu.virtual` |
| `--imu-params` | IMU sensor parameters (e.g., `file=/path/to/imu.bin`) | Required |
| `--gps-protocol` | GPS/RTK sensor protocol | Optional |
| `--gps-params` | GPS/RTK sensor parameters | Optional |
| `--lidar-imu-extrinsics` | Extrinsic calibration between LiDAR and IMU | Optional |
| `--voxel-size` | Voxel grid filter size for map downsampling | `0.5` |
| `--map-file` | Path to pre-built map file for localization mode | Optional |
| `--offscreen` | Enable offscreen rendering | Disabled |

### Example

```bash
# Mapping mode (build new map)
./sample_dw_fastlio_slam \
    --lidar-protocol=lidar.virtual \
    --lidar-params="file=/data/lidar.bin" \
    --imu-protocol=imu.virtual \
    --imu-params="file=/data/imu.bin" \
    --voxel-size=0.5

# Localization mode (use existing map)
./sample_dw_fastlio_slam \
    --lidar-protocol=lidar.virtual \
    --lidar-params="file=/data/lidar.bin" \
    --imu-protocol=imu.virtual \
    --imu-params="file=/data/imu.bin" \
    --map-file=/data/slam_map.pcd
```

### Interactive Controls

During runtime, the following keyboard controls are available:

- **M** - Toggle between mapping and localization mode
- **S** - Save current map to file (timestamped `.pcd` file)
- **L** - Load map from file (requires `--map-file` argument)

## Configuration

### SLAM Parameters

The SLAM system can be configured through the `DWFastLIOConfig` structure:

- **Extrinsics**: LiDAR-to-IMU transformation matrix
- **Filter Parameters**: Point cloud filtering and processing settings
- **Mapping Parameters**: Keyframe selection thresholds
- **Optimization**: Graph optimization solver settings

### Sensor Calibration

Proper sensor calibration is critical for accurate SLAM performance:
- **LiDAR-IMU Extrinsics**: Must be calibrated and provided via `--lidar-imu-extrinsics`
- **IMU Intrinsics**: Configured through DriveWorks sensor parameters
- **GPS/RTK**: Optional but recommended for global consistency


## Current Status

### ✅ Completed
- DriveWorks application compilation with Fast-LIO as dependency
- Data type conversion layer between Fast-LIO library and DriveWorks datatypes
- Mutex locks to prevent deadlocks and data overwriting

### 🔄 In Progress
- Identifying and fixing IMU/LiDAR data synchronization issues (sway correction)
- CMake build system cleanup
- Improving scan overlap for consecutive LiDAR frames

### ⚠️ Known Issues
- **IMU/LiDAR Data Sway**: Consecutive scans don't overlap properly due to sensor data synchronization issues. This is actively being worked on.
- **CPU-Only Processing**: Currently runs on CPU only. GPU acceleration is planned for future work.

## Future Work

- **Localization Pipeline**: Build localization logic within  a saved map
- **GPU Acceleration**: CUDA-accelerated point cloud processing and optimization
- **Location-Based Map Management**: Automatic saving and loading of maps based on current GPS location
- **Improved Sensor Fusion**: Better handling of sensor data synchronization and calibration
- **Performance Optimization**: Real-time performance improvements for larger environments
