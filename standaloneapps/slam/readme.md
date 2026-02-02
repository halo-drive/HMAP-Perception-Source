# SLAM Pipeline - Fast-LIO on NVIDIA DriveWorks

This repository contains a SLAM (Simultaneous Localization and Mapping) pipeline implementation using LiDAR + IMU + GPS/RTK sensors, built around the Fast-LIO algorithm integrated with NVIDIA DriveWorks SDK.

## Overview

The SLAM system provides real-time mapping and localization capabilities for autonomous vehicles and robotics applications. It combines:
- **LiDAR** for high-resolution 3D point cloud data
- **IMU** for high-frequency motion estimation
- **GPS/RTK** for global positioning and loop closure

The implementation features a pure DriveWorks-native solution without external Python or ZeroCM dependencies, making it suitable for production deployment on NVIDIA DRIVE platforms.

## Features

-  Real-time LiDAR-inertial odometry using Fast-LIO algorithm [ done ]
-  Multi-sensor fusion (LiDAR, IMU, GPS/RTK) [ done ]
-  Thread-safe data processing with mutex locks [ done ]
-  DriveWorks-native visualization [ done ]
-  Map building and saving capabilities [ pending ]
-  Localization mode (load pre-built maps) [ pending ]
-  Type conversion layer between Fast-LIO and DriveWorks datatypes [ done ]

## Dependencies

### Essential Libraries
- **PCL (Point Cloud Library)** 1.10+ - Point cloud processing
- **Eigen3** - Linear algebra and transformations
- **Boost** - Required by PCL


## Building

### Prerequisites
Ensure all dependencies are installed and DriveWorks SDK is properly configured in your environment.

### Build Instructions

The build system automatically detects the architecture (x86_64 or aarch64) and configures paths accordingly.


### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--rig` | LiDAR sensor protocol (e.g., `lidar.virtual`) | `lidar.virtual` |
| `--voxel-size` | Voxel grid filter size for map downsampling | `0.5` |
| `--map-file` | Path to pre-built map file for localization mode | Optional |
| `--offscreen` | Enable offscreen rendering | Disabled |

### Example

```bash
# Mapping mode (build new map)
./sample_dw_fastlio_slam \
    --rig=<path to rig.json> \
    --voxel-size=0.5

# Localization mode (use existing map)
./sample_dw_fastlio_slam \
    --rig = <path to rig.json>\
    --map-file=/data/slam_map.pcd
```

### Interactive Controls

During runtime, the following keyboard controls are available:

- **M** - Toggle between mapping and localization mode
- **S** - Save current map to file (timestamped `.pcd` file)
- **L** - Load map from file (requires `--map-file` argument)


## Current Status

###  Completed
- DriveWorks application compilation with Fast-LIO as dependency
- Data type conversion layer between Fast-LIO library and DriveWorks datatypes
- Mutex locks to prevent deadlocks and data overwriting
- Run sensors using sensor manager and rig file 

### In Progress
- test on moving car
- Solve dependencies for boost, pcl and g2o for orin. [ couldnt install then using sudo apt ]
- Implement map saving and loading feature.

### Known Issues
- **CPU-Only Processing**: Currently runs on CPU only. GPU acceleration is planned for future work.

## Future Work
- **Localization Pipeline**: Build localization logic within  a saved map
- **GPU Acceleration**: CUDA-accelerated point cloud processing and optimization
- **Location-Based Map Management**: Automatic saving and loading of maps based on current GPS location
- **Improved Sensor Fusion**: Better handling of sensor data synchronization and calibration
- **Performance Optimization**: Real-time performance improvements for larger environments
