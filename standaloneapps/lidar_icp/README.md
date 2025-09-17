### Inter-Lidar ICP + Ground Plane Extraction (Sample)

**Purpose**: Align two LiDAR point clouds in real time using ICP, then extract and visualize the ground plane.

### Run
- From this directory or your workspace root, run the sample binary with the options below.
- Show help:

```bash
./lidar_icp --rigFile='/usr/local/driveworks-5.20/samples/src/sensors/lidar/lidar_icp/rig.json' --verbose=true
```

### Required
- **--rigFile=FILE**: Path to rig configuration JSON (must define exactly 2 LiDARs and transforms).


### Output
- 4-panel visualization: individual lidars, ICP alignment view, stitched result
- Console logs for ICP and ground plane detection
- Optional PLY files when enabled
