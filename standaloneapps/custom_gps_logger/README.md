# Custom GPS Logger

This is a modified version of the NVIDIA DriveWorks GPS sample that logs GPS data to JSON files and supports data comparison.

this was used to understand rtk and feasibility to check if the vehicle is follwoing same recorded path or not.


## Features

1. **Rig File Based Initialization**: GPS sensor is initialized from a rig configuration file instead of hardcoded parameters
2. **JSON Data Logging**: All GPS data is saved to a timestamped JSON file
3. **Data Comparison**: Optional comparison mode to compare current GPS data against reference data
4. **Simple Command Line Interface**: Only two parameters needed: rig file and optional input file

## Usage

### Basic GPS Logging
```bash
./custom_gps_logger --rig=example_rig.json
```

### GPS Logging with Comparison
```bash
./custom_gps_logger --rig=example_rig.json --input-file=reference_gps.json
```

## Command Line Arguments

- `--rig=rig_file.json`: **Required** - Path to rig configuration file containing GPS sensor setup
- `--input-file=reference.json`: **Optional** - Path to reference GPS data file for comparison

## Output

### Output Files

#### GPS Data File
The application always saves GPS data to a timestamped JSON file with the following naming convention:
- **Main mode**: `gps_log_main_[timestamp].json` (when no comparison is performed)
- **Compare mode**: `gps_log_compare_[timestamp].json` (when comparing against reference data)

Example structure:

```json
{
  "gps_data": [
    {
      "timestamp_us": 1703123456789000,
      "latitude": 37.7749295,
      "longitude": -122.4194155,
      "altitude": 52.5,
      "course": 180.0,
      "speed": 15.2,
      "climb": 0.1,
      "hdop": 1.2,
      "vdop": 1.5,
      "pdop": 1.9,
      "hacc": 2.1,
      "vacc": 3.2,
      "utcTimeUs": 1703123456789000,
      "satelliteCount": 8,
      "fixStatus": 1,
      "timestampQuality": 1,
      "mode": 3,
      "errors": 0
    }
  ]
}
```

#### Comparison Data File (when using --input-file)
When using comparison mode, the application saves detailed comparison data to a separate JSON file (e.g., `gps_comparison_1703123456.json`) with the following structure:

```json
{
  "main_file": "/path/to/reference.json",
  "path_comparison_analysis": [
    {
      "timestamp_us": 1703123456789000,
      "current_position": {
        "latitude": 37.7749295,
        "longitude": -122.4194155
      },
      "reference_position": {
        "latitude": 37.7749300,
        "longitude": -122.4194160
      },
      "errors": {
        "position_error_m": 2.345,
        "altitude_error_m": 1.2,
        "speed_error_ms": 0.5,
        "course_error_deg": 5.2
      },
      "path_analysis": {
        "closest_reference_idx": 156,
        "distance_along_path_m": 45.234,
        "is_on_path": true
      }
    }
  ]
}
```

#### Path Analysis File (when using --input-file)
When using comparison mode, the application also saves a comprehensive path analysis report to a TXT file (e.g., `path_analysis_1703123456.txt`) with the following content:

```
========================================
PATH FOLLOWING ANALYSIS REPORT
========================================

Reference File: /path/to/reference.json
Current File: /path/to/gps_log_compare_1703123456.json
Analysis Date: 1703123456

PATH STATISTICS:
---------------
Reference path points: 1000
Current path points: 950
Total distance traveled: 1250.50 meters

ERROR ANALYSIS:
--------------
Average position error: 3.245 meters
Maximum position error: 12.800 meters

PATH FOLLOWING PERFORMANCE:
---------------------------
Path following accuracy: 87.5%
Points on path (within 10m): 831/950

RESULT:
-------
SUCCESS: Successfully following the same path!

========================================
```

### Console Output
During execution, the application displays:
- GPS coordinates and sensor data
- Error messages if any
- Comparison results (if comparison mode is enabled)
- Summary statistics at the end

### Path Following Analysis
When `--input-file` is provided, the application:
- Loads reference GPS data from the JSON file
- Performs **spatial path comparison** (not temporal) to handle different start times and durations
- For each current GPS point, finds the closest point on the reference path
- Displays real-time path analysis including:
  - Position error (meters) from closest reference point
  - Distance along the reference path
  - Whether the current point is "on path" (within 10m of reference)
- Provides comprehensive path following analysis at the end:
  - Path following accuracy percentage
  - Average and maximum position errors
  - Total distance traveled
  - Clear result indicating if the same path is being followed

## Rig File Format

The rig file should contain a GPS sensor configuration. See `example_rig.json` for a sample configuration.

### Example Rig File Structure
```json
{
  "rig": {
    "sensors": [
      {
        "name": "gps",
        "protocol": "gps.virtual",
        "parameter": "fifo-size=1024",
        "nominalSensor2Rig": {
          "quaternion": [0.0, 0.0, 0.0, 1.0],
          "t": [0.0, 0.0, 0.0]
        },
        "sensor2Rig": {
          "quaternion": [0.0, 0.0, 0.0, 1.0],
          "t": [0.0, 0.0, 0.0]
        }
      }
    ]
  },
  "version": 2
}
```

## Building

The application uses the standard DriveWorks CMake build system:

```bash
cd /usr/local/driveworks-5.20/samples/src/sensors/custom_gps_logger
mkdir build && cd build
cmake ..
make
```

## Requirements

- NVIDIA DriveWorks SDK 5.20
- Valid rig configuration file
- GPS sensor hardware or virtual GPS driver

## Notes

- The application automatically generates timestamped output filenames
- GPS data is collected until the application is stopped (Ctrl+C)
- All GPS fields are logged, including validity information
- Comparison mode requires reference data in the same JSON format as the output
