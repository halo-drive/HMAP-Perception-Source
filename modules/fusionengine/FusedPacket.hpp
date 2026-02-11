////////////////////////////////////////////////////////////////////////////////
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS"
// NVIDIA MAKES NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR
// OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED
// WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR
// PURPOSE.
//
// FusedPacket.hpp - Data structures for sensor fusion
// Defines common structures used by LidarIPCClient, CameraIPCClient, and
// SensorSynchronizer for multi-sensor fusion pipeline.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef FUSIONENGINE_FUSEDPACKET_HPP
#define FUSIONENGINE_FUSEDPACKET_HPP

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <dw/core/base/Types.h>

namespace fusionengine {

//------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------
static constexpr uint32_t MAX_LIDAR_POINTS       = 100000;
static constexpr uint32_t MAX_LIDAR_DETECTIONS   = 100;
static constexpr uint32_t MAX_CAMERA_DETECTIONS  = 100;
static constexpr uint32_t MAX_FUSED_DETECTIONS   = 200;
static constexpr uint32_t MAX_FREE_SPACE_POINTS  = 360;   // Downsampled for fusion
static constexpr uint32_t MAX_CAMERAS            = 4;

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

//------------------------------------------------------------------------------
// LiDAR Detection Bounding Box
// Matches LidarIPCClient::LidarDetectionPacket::DetectionBoundingBox
//------------------------------------------------------------------------------
struct LidarBoundingBox
{
    float x{0.0f}, y{0.0f}, z{0.0f};           // Center in LiDAR frame (meters)
    float width{0.0f}, length{0.0f}, height{0.0f};  // 3D dimensions (meters)
    float rotation{0.0f};                       // Yaw around Z-axis (radians)
    float confidence{0.0f};                     // Detection confidence [0.0, 1.0]
    int32_t classId{0};                         // 0: Vehicle, 1: Pedestrian, 2: Cyclist
};

//------------------------------------------------------------------------------
// Ground Plane Data
// Matches LidarIPCClient::LidarDetectionPacket::GroundPlane
//------------------------------------------------------------------------------
struct GroundPlaneData
{
    float normalX{0.0f}, normalY{0.0f}, normalZ{1.0f};  // Plane normal vector
    float offset{0.0f};                                  // Plane offset (d in ax+by+cz+d=0)
    bool valid{false};                                   // Validity flag
};

//------------------------------------------------------------------------------
// Free Space Data (downsampled from LiDAR server's 50000 points)
// Used for driveable area visualization
//------------------------------------------------------------------------------
struct FreeSpaceData
{
    uint32_t numPoints{0};
    float points[MAX_FREE_SPACE_POINTS * 3]{};  // x, y, distance per angle bin

    void clear()
    {
        numPoints = 0;
        std::memset(points, 0, sizeof(points));
    }
};

//------------------------------------------------------------------------------
// Camera 2D Detection Box
// Populated from CameraIPCClient::CameraDetectionBox
//------------------------------------------------------------------------------
struct Camera2DBox
{
    float x{0.0f}, y{0.0f};                // Top-left corner in image coords
    float width{0.0f}, height{0.0f};       // Box dimensions in pixels
    char label[64]{};                       // Class label string
    uint32_t cameraIndex{0};               // Source camera [0-3]

    Camera2DBox()
    {
        std::memset(label, 0, sizeof(label));
    }
};

//------------------------------------------------------------------------------
// Camera 3D Detection Box
// Populated from CameraIPCClient::CameraDetection3DBox
//------------------------------------------------------------------------------
struct Camera3DBox
{
    float depth{0.0f};                     // Estimated depth (meters)
    float height{0.0f}, width{0.0f}, length{0.0f};  // 3D dimensions (meters)
    float rotation{0.0f};                  // Estimated yaw rotation (radians)
    float iouScore{0.0f};                  // Quality/confidence score
    uint32_t cameraIndex{0};               // Source camera [0-3]
    dwRectf box2D{0.0f, 0.0f, 0.0f, 0.0f}; // Associated 2D bounding box
};

//------------------------------------------------------------------------------
// Fused Detection (combined LiDAR + Camera)
// Output of the sensor fusion algorithm
//------------------------------------------------------------------------------
struct FusedDetection
{
    // Position in world/vehicle frame (meters)
    float x{0.0f}, y{0.0f}, z{0.0f};

    // 3D dimensions (meters)
    float width{0.0f}, length{0.0f}, height{0.0f};

    // Orientation
    float rotation{0.0f};                  // Yaw rotation (radians)

    // Classification
    int32_t classId{0};                    // 0: Vehicle, 1: Pedestrian, 2: Cyclist
    char label[64]{};                      // Class label string

    // Confidence scores
    float lidarConfidence{0.0f};
    float cameraConfidence{0.0f};
    float fusedConfidence{0.0f};

    // Source information
    bool hasLidarSource{false};
    bool hasCameraSource{false};
    uint32_t cameraSourceMask{0};          // Bitmask of contributing cameras

    // Tracking ID (for temporal association)
    uint32_t trackId{0};

    FusedDetection()
    {
        std::memset(label, 0, sizeof(label));
    }
};

//------------------------------------------------------------------------------
// Per-Camera Frame Data
// Output of CameraIPCClient::receive()
//------------------------------------------------------------------------------
struct CameraFrameData
{
    uint32_t cameraIndex{0};
    uint64_t frameId{0};
    dwTime_t timestamp{0};

    // Image dimensions
    uint32_t width{0};
    uint32_t height{0};

    // RGBA pixel data (width * height * 4 bytes)
    std::vector<uint8_t> rgbaPixels;

    // 2D detections from object detector
    std::vector<Camera2DBox> boxes2D;

    // 3D detections from Stage2 depth network
    std::vector<Camera3DBox> boxes3D;

    // Processing statistics from server
    uint32_t segCount{0};                  // Segmentation inference count
    uint32_t detCount{0};                  // Detection inference count
    float avgSegMs{0.0f};                  // Average segmentation time (ms)
    float avgDetMs{0.0f};                  // Average detection time (ms)
    float avgStage2Ms{0.0f};               // Average Stage2 depth time (ms)

    bool valid{false};

    void clear()
    {
        cameraIndex = 0;
        frameId = 0;
        timestamp = 0;
        width = 0;
        height = 0;
        rgbaPixels.clear();
        boxes2D.clear();
        boxes3D.clear();
        segCount = 0;
        detCount = 0;
        avgSegMs = 0.0f;
        avgDetMs = 0.0f;
        avgStage2Ms = 0.0f;
        valid = false;
    }
};

//------------------------------------------------------------------------------
// LiDAR Frame Data
// Output of LidarIPCClient::receive()
//------------------------------------------------------------------------------
struct LidarFrameData
{
    dwTime_t timestamp{0};
    uint32_t frameNumber{0};
    bool icpAligned{false};

    // Point cloud (x, y, z, intensity per point)
    uint32_t numPoints{0};
    std::vector<float> points;             // Size: numPoints * 4

    // 3D detections from LiDAR object detector
    std::vector<LidarBoundingBox> detections;

    // Ground plane estimation
    GroundPlaneData groundPlane;

    // Free space (driveable area boundary)
    FreeSpaceData freeSpace;

    bool valid{false};

    void clear()
    {
        timestamp = 0;
        frameNumber = 0;
        icpAligned = false;
        numPoints = 0;
        points.clear();
        detections.clear();
        groundPlane = GroundPlaneData{};
        freeSpace.clear();
        valid = false;
    }
};

//------------------------------------------------------------------------------
// Fused Output Packet
// Combined output from SensorSynchronizer and fusion algorithm
//------------------------------------------------------------------------------
struct FusedPacket
{
    // Timestamps
    dwTime_t lidarTimestamp{0};
    dwTime_t cameraTimestamps[MAX_CAMERAS]{};
    dwTime_t fusionTimestamp{0};

    // Frame identifiers
    uint32_t lidarFrameNumber{0};
    uint64_t cameraFrameIds[MAX_CAMERAS]{};
    uint32_t fusionFrameNumber{0};

    // Raw sensor data (may be large - ~2MB for LiDAR, ~1.6MB per camera)
    LidarFrameData lidarData;
    std::array<CameraFrameData, MAX_CAMERAS> cameraData;

    // Fused detections (output of fusion algorithm)
    std::vector<FusedDetection> fusedDetections;

    // Fusion quality metrics
    float temporalAlignmentError{0.0f};    // Max timestamp difference (ms)
    uint32_t numLidarOnlyDetections{0};
    uint32_t numCameraOnlyDetections{0};
    uint32_t numFusedDetections{0};

    // Processing statistics
    float lidarProcessingMs{0.0f};
    float cameraProcessingMs{0.0f};
    float fusionProcessingMs{0.0f};
    float totalProcessingMs{0.0f};

    bool valid{false};

    void clear()
    {
        lidarTimestamp = 0;
        std::memset(cameraTimestamps, 0, sizeof(cameraTimestamps));
        fusionTimestamp = 0;
        lidarFrameNumber = 0;
        std::memset(cameraFrameIds, 0, sizeof(cameraFrameIds));
        fusionFrameNumber = 0;
        lidarData.clear();
        for (auto& cam : cameraData)
        {
            cam.clear();
        }
        fusedDetections.clear();
        temporalAlignmentError = 0.0f;
        numLidarOnlyDetections = 0;
        numCameraOnlyDetections = 0;
        numFusedDetections = 0;
        lidarProcessingMs = 0.0f;
        cameraProcessingMs = 0.0f;
        fusionProcessingMs = 0.0f;
        totalProcessingMs = 0.0f;
        valid = false;
    }
};

//------------------------------------------------------------------------------
// Utility: Convert class ID to label string
//------------------------------------------------------------------------------
inline const char* classIdToLabel(int32_t classId)
{
    switch (classId)
    {
    case 0:
        return "Vehicle";
    case 1:
        return "Pedestrian";
    case 2:
        return "Cyclist";
    default:
        return "Unknown";
    }
}

//------------------------------------------------------------------------------
// Utility: Convert SensorStatus to string
//------------------------------------------------------------------------------
inline const char* sensorStatusToString(SensorStatus status)
{
    switch (status)
    {
    case SensorStatus::DISCONNECTED:
        return "DISCONNECTED";
    case SensorStatus::CONNECTING:
        return "CONNECTING";
    case SensorStatus::CONNECTED:
        return "CONNECTED";
    case SensorStatus::RECEIVING:
        return "RECEIVING";
    case SensorStatus::ERROR:
        return "ERROR";
    default:
        return "UNKNOWN";
    }
}

} // namespace fusionengine

#endif // FUSIONENGINE_FUSEDPACKET_HPP
