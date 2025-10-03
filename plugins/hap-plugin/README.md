# Livox HAP LiDAR Visualization Application

This application provides real-time visualization of Livox HAP (High Accuracy Positioning) LiDAR sensor data using NVIDIA DriveWorks.

## Overview

The Livox HAP visualization application is designed to display point cloud data from the Livox HAP sensor in real-time. It supports:

- Real-time point cloud visualization
- Configurable point accumulation
- Multiple data format support (High Precision Cartesian, Low Precision Cartesian, Spherical)
- Interactive 3D rendering with mouse controls
- Point cloud data export

## Prerequisites

- NVIDIA DriveWorks SDK
- Livox SDK2
- Livox HAP sensor
- Network connection to the sensor
- Compiled Livox HAP plugin (`liblivox_hap_plugin.so`)

## Building

1. **Navigate to the application directory**:
   ```bash
   cd driveworks/samples/src/sensors/lidar/lidar_livox_hap_replay
   ```

2. **Build the application**:
   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```

## Configuration

### Network Configuration

Update the `livox_hap_config.json` file with your HAP sensor settings:

```json
{
    "master_sdk": true,
    "lidar_log_enable": true,
    "lidar_log_cache_size_MB": 100,
    "lidar_log_path": "/path/to/log/directory",
    
    "HAP": {
      "lidar_net_info": {
        "cmd_data_port": 57000,
        "push_msg_port": 57100,
        "point_data_port": 57200,
        "log_data_port": 57300
      },
      "host_net_info": [
        {
          "lidar_ip": ["192.168.1.3"],
          "lidar_broadcast_code": ["YOUR_HAP_BROADCAST_CODE"],
          "host_ip": "192.168.1.100",
          "cmd_data_port": 57001,
          "push_msg_port": 57101,
          "point_data_port": 57201,
          "log_data_port": 57301
        }
      ]
    }
}
```

### Required Parameters

- `lidar_ip`: IP address of your HAP sensor
- `lidar_broadcast_code`: Broadcast code of your HAP sensor
- `host_ip`: IP address of your host machine

## Usage

### Basic Usage

```bash
./lidar_replay_livox_hap \
    --protocol=lidar.custom \
    --params="decoder-path=/path/to/liblivox_hap_plugin.so,ip=192.168.1.3,host-ip=192.168.1.100,broadcast-code=YOUR_CODE,sdk-config-path=/path/to/livox_hap_config.json" \
    --max-points=20000 \
    --show-intensity=true
```

### Advanced Usage

```bash
./lidar_replay_livox_hap \
    --protocol=lidar.custom \
    --params="decoder-path=/usr/local/lib/liblivox_hap_plugin.so,ip=192.168.1.3,host-ip=192.168.1.100,broadcast-code=0TFDG3B006H2Z11,sdk-config-path=/usr/local/driveworks/samples/src/sensors/lidar/lidar_livox_hap_replay/livox_hap_config.json" \
    --max-points=50000 \
    --show-intensity=true \
    --output-dir=/tmp/hap_data
```

### Command Line Arguments

- `--protocol`: Sensor protocol (use `lidar.custom` for HAP)
- `--params`: Comma-separated parameter string
- `--max-points`: Maximum number of points to accumulate (default: 20000)
- `--show-intensity`: Show intensity values (true/false, default: true)
- `--output-dir`: Directory to save point cloud data (optional)

### Parameter String Format

The `--params` argument should include:

```
decoder-path=/path/to/liblivox_hap_plugin.so,
ip=192.168.1.3,
host-ip=192.168.1.100,
broadcast-code=YOUR_HAP_BROADCAST_CODE,
sdk-config-path=/path/to/livox_hap_config.json
```

## Controls

- **Mouse**: Rotate, pan, and zoom the 3D view
- **R**: Reset the sensor and visualization
- **ESC**: Exit the application

## Features

### Point Cloud Visualization

- Real-time 3D point cloud rendering
- Color-coded by position or intensity
- Configurable point accumulation
- Interactive camera controls

### Data Export

When `--output-dir` is specified, the application saves point cloud data as binary files:

- Filename format: `{timestamp}.bin`
- Data format: Raw point cloud data in binary format
- Each file contains one frame of accumulated points

### Performance Monitoring

The application displays real-time statistics:

- Host and sensor timestamps
- Packets per scan
- Points per scan
- Frame rate
- Device information
- Accumulated point count

## Troubleshooting

### Common Issues

1. **No Point Cloud Data**:
   - Check network connectivity
   - Verify broadcast code
   - Ensure sensor is powered on
   - Check firewall settings

2. **Plugin Loading Error**:
   - Verify plugin path in `--params`
   - Ensure plugin is compiled and installed
   - Check file permissions

3. **Network Connection Issues**:
   - Verify IP addresses in config file
   - Check network cable connection
   - Ensure ports are not blocked by firewall

### Debug Information

The application provides detailed console output:

- Packet reception statistics
- Point accumulation progress
- Frame completion status
- Error messages and warnings

## Performance Considerations

- **Point Accumulation**: Higher `--max-points` values provide denser point clouds but may impact performance
- **Network Bandwidth**: HAP sensors can generate high data rates; ensure adequate network capacity
- **Memory Usage**: Large point clouds require significant memory; adjust `--max-points` accordingly

## Limitations

- Maximum 100,000 points per frame (configurable)
- Single sensor per application instance
- Requires Livox SDK2 compatibility
- No IMU data support (HAP sensor is LiDAR-only)

## Support

For issues and questions:
- Check the troubleshooting section
- Review the Livox HAP documentation
- Contact the development team 




/home/nvidia/build-x86_64-linux-gnu/install/usr/local/driveworks/samples/bin/lidar_livox_hap_replay \
    --protocol=lidar.custom \
    --params="decoder-path=/home/nvidia/build-x86_64-linux-gnu/install/usr/local/driveworks/samples/lib/liblivox_hap_plugin.so,ip=192.168.1.100,host-ip=192.168.1.123,broadcast-code=5CWD239T4106BV,sdk-config-path=/usr/local/driveworks/samples/src/sensors/lidar/lidar_livox_hap_replay/livox_hap_config.json" \
    --max-points=20000 \
    --show-intensity=true


 ./lidar_livox_hap_replay \
    --protocol=lidar.custom \
    --params="decoder-path=/home/nvidia/build-x86_64-linux-gnu/install/usr/local/driveworks/samples/lib/liblivox_hap_plugin.so,ip=192.168.1.100,host-ip=192.168.1.123,broadcast-code=5CWD239T4106BV,sdk-config-path=/usr/local/driveworks/samples/src/sensors/lidar/lidar_livox_hap_replay/livox_hap_config.json" \
    --max-points=20000 \
    --show-intensity=true 2>&1 | tee /usr/local/driveworks/samples/src/sensors/lidar/lidar_livox_hap_replay/hap_visualization_logs.txt