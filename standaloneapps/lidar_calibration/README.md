# Lidar Calibration Tool

Simple tool to compute ICP corrections for multi-lidar setups with real-time visualization. The calibration values can be used for correcting rotation values for recordings.

## Purpose

This tool:
1. Loads sensors from rig file
2. Runs ICP alignment between two lidars
3. Visualizes the alignment in real-time
4. Outputs correction values (translation + roll/pitch/yaw) for the rig file

## Usage

```bash
./lidar_calibration --rig=../rigFile/rig.json --num-frames=100 --max-iters=50 --verbose=true
```

## Parameters

- `--rig`: Path to rig configuration file (required)
- `--num-frames`: Number of frames to process for calibration (default: 100)
- `--max-iters`: Maximum ICP iterations per frame (default: 50)
- `--verbose`: Enable verbose logging (default: false)

## Output

The tool outputs correction values that you can copy directly into your rig file:

```
=== CALIBRATION RESULTS ===
Successful ICP: 95/100

ICP Correction for: lidar:rear:vlp16hr
-----------------------------------
Translation (meters): [0.005123, -0.002345, 0.000876]
Rotation (degrees):   [0.234567, -0.123456, 0.345678]

=== UPDATE YOUR RIG FILE ===
For sensor: lidar:rear:vlp16hr

"correction_rig_T": [0.005123, -0.002345, 0.000876],
"correction_sensor_R_FLU": {
    "roll-pitch-yaw": [0.234567, -0.123456, 0.345678]
}
```

## Visualization

The tool provides real-time 3D visualization:

- **Green points**: Front lidar (reference)
- **Red points**: Rear lidar after ICP alignment
- **On-screen stats**: Frame count, ICP success rate, average correction values

This helps you:
- Verify that calibration is working correctly
- See how well the point clouds align
- Identify if there are any issues with the sensor data or alignment

## How It Works

1. Initializes both lidars from rig configuration
2. Accumulates point clouds from each lidar
3. Applies nominal rig transformations
4. Runs ICP between the two point clouds
5. Visualizes the aligned point clouds in real-time
6. Accumulates the correction over multiple frames
7. Outputs the averaged correction values

## Calibration Workflow

### **Recommended: Start with Rough Corrections for t values**

For the **first calibration**, set corrections to zero in your rig file:

```json
"correction_rig_T": [0.0, 0.0, 0.0],
"correction_sensor_R_FLU": {
    "roll-pitch-yaw": [0.0, 0.0, 0.0]
}
```

**Why?**
- Shows the **complete deviation** from your nominal setup
- Gives you the **full correction values** to apply directly
- Simplest and most accurate approach
- You can see the total misalignment in the visualization

**Steps:**
1. Set corrections to `[0.0, 0.0, 0.0]` in your rig file
2. Run the calibration tool
3. Copy the output correction values directly into your rig file
4. Run your application - the point clouds should now align well!


## Notes

- Requires exactly 2 lidars in the rig file
- Uses depth map ICP (requires organized point clouds)
- The second lidar is aligned to the first (reference)
- Apply the output values to the second lidar's correction fields in your rig file
- Corrections are applied as: `sensor2Rig = [correctionT] * sensor2RigNominal * [correctionR]`

