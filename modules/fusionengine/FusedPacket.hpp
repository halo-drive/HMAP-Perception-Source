////////////////////////////////////////////////////////////////////////////////
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS"
// NVIDIA MAKES NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR
// OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED
// WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR
// PURPOSE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef FUSIONENGINE_FUSEDPACKET_HPP
#define FUSIONENGINE_FUSEDPACKET_HPP

#include <cstdint>
#include <vector>
#include <string>
#include <array>

#include <dw/core/base/Types.h>

namespace fusionengine {

//------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------
static constexpr uint32_t MAX_LIDAR_POINTS      = 100000;
static constexpr uint32_t MAX_LIDAR_DETECTIONS  = 100;
static constexpr uint32_t MAX_CAMERA_DETECTIONS = 100;
static constexpr uint32_t MAX_FUSED_DETECTIONS  = 200;
static constexpr uint32_t MAX_FREE_SPACE_POINTS = 360;
static constexpr uint32_t MAX_CAMERAS           = 4;

//------------------------------------------------------------------------------
// LiDAR Detection Bounding Box (from lidardetseg_server)
//------------------------------------------------------------------------------
struct LidarBoundingBox
{
    float x, y, z;                  // Center coordinates in LiDAR frame (meters)
    float width, length, height;    // 3D dimensions (meters)
    float rotation;                 // Yaw rotation around Z-axis (radians)
    float confidence;               // Detection confidence [0.0, 1.0]
    int32_t classId;                // 0: Vehicle, 1: Pedestrian, 2: Cyclist
};

//------------------------------------------------------------------------------
// Ground Plane Data (from lidardetseg_server)
//------------------------------------------------------------------------------
struct GroundPlaneData
{
    float normalX, normalY, normalZ;  // Ground plane normal vector
    float offset;                      // Plane offset (ax + by + cz + d = 0)
    bool valid;                        // Validity flag
};

//------------------------------------------------------------------------------
// Free Space Data (from lidardetseg_server)
//------------------------------------------------------------------------------
struct FreeSpaceData
{
    uint32_t numPoints;
    float points[MAX_FREE_SPACE_POINTS * 3];  // x, y, intensity per angle
};

//------------------------------------------------------------------------------
// Camera 2D Detection Box (from driveseg_object)
//------------------------------------------------------------------------------
struct Camera2DBox
{
    float x, y, width, height;    // Bounding box in image coordinates (640x640)
    char label[64];               // Class label string
    uint32_t cameraIndex;         // Source camera [0-3]
};

//------------------------------------------------------------------------------
// Camera 3D Detection Box (from driveseg_object Stage2)
//------------------------------------------------------------------------------
struct Camera3DBox
{
    float depth;                  // Estimated depth (meters)
    float height, width, length;  // Estimated dimensions (meters)
    float rotation;               // Estimated yaw rotation (radians)
    float iouScore;               // Quality/confidence score
    uint32_t cameraIndex;         // Source camera [0-3]
    dwRectf box2D;                // Associated 2D box
};

//------------------------------------------------------------------------------
// Fused Detection (combined LiDAR + Camera)
//------------------------------------------------------------------------------
struct FusedDetection
{
    // Position in world/vehicle frame
    float x, y, z;

    // 3D dimensions
    float width, length, height;

    // Orientation
    float rotation;

    // Classification
    int32_t classId;
    char label[64];

    // Confidence scores
    float lidarConfidence;
    float cameraConfidence;
    float fusedConfidence;

    // Source information
    bool hasLidarSource;
    bool hasCameraSource;
    uint32_t cameraSourceMask;    // Bitmask of contributing cameras

    // Tracking ID (for temporal association)
    uint32_t trackId;
};

//------------------------------------------------------------------------------
// Per-Camera Frame Data
//------------------------------------------------------------------------------
struct CameraFrameData
{
    uint32_t cameraIndex;
    uint64_t frameId;
    dwTime_t timestamp;

    // Image data (640x640 RGBA)
    uint32_t width;
    uint32_t height;
    std::vector<uint8_t> rgbaPixels;

    // 2D detections
    std::vector<Camera2DBox> boxes2D;

    // 3D detections (from Stage2 network)
    std::vector<Camera3DBox> boxes3D;

    // Statistics
    uint32_t segCount;
    uint32_t detCount;
    float avgSegMs;
    float avgDetMs;
    float avgStage2Ms;

    bool valid;
};

//------------------------------------------------------------------------------
// LiDAR Frame Data
//------------------------------------------------------------------------------
struct LidarFrameData
{
    dwTime_t timestamp;
    uint32_t frameNumber;
    bool icpAligned;

    // Point cloud
    uint32_t numPoints;
    std::vector<float> points;    // x, y, z, intensity per point

    // Detections
    std::vector<LidarBoundingBox> detections;

    // Ground plane
    GroundPlaneData groundPlane;

    // Free space
    FreeSpaceData freeSpace;

    bool valid;
};

//------------------------------------------------------------------------------
// Fused Output Packet
//------------------------------------------------------------------------------
struct FusedPacket
{
    // Timestamps
    dwTime_t lidarTimestamp;
    dwTime_t cameraTimestamps[MAX_CAMERAS];
    dwTime_t fusionTimestamp;

    // Frame identifiers
    uint32_t lidarFrameNumber;
    uint64_t cameraFrameIds[MAX_CAMERAS];
    uint32_t fusionFrameNumber;

    // Raw sensor data (optional, can be large)
    LidarFrameData lidarData;
    std::array<CameraFrameData, MAX_CAMERAS> cameraData;

    // Fused detections
    std::vector<FusedDetection> fusedDetections;

    // Fusion quality metrics
    float temporalAlignmentError;   // Max timestamp difference (ms)
    uint32_t numLidarOnlyDetections;
    uint32_t numCameraOnlyDetections;
    uint32_t numFusedDetections;

    // Processing statistics
    float lidarProcessingMs;
    float cameraProcessingMs;
    float fusionProcessingMs;
    float totalProcessingMs;

    bool valid;
};

//------------------------------------------------------------------------------
// Sensor status enumeration
//------------------------------------------------------------------------------
enum class SensorStatus
{
    DISCONNECTED = 0,
    CONNECTING,
    CONNECTED,
    RECEIVING,
    ERROR
};

} // namespace fusionengine

#endif // FUSIONENGINE_FUSEDPACKET_HPP
