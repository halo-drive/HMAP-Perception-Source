#ifndef DETECTION_PACKET_HPP
#define DETECTION_PACKET_HPP

#include <cstdint>
#include <cstddef>
#include <dw/core/base/Types.h>
#include <dwcgf/channel/ChannelPacketTypes.hpp>

// Maximum number of bounding boxes per frame
constexpr uint32_t MAX_DETECTIONS = 100;

// Bounding box structure (matches InterLidarICP.hpp)
struct DetectionBoundingBox {
    float x, y, z;           // Center coordinates
    float width, length, height;  // Dimensions
    float rotation;          // Rotation around Z-axis
    float confidence;        // Detection confidence
    int32_t classId;         // Object class (0: Vehicle, 1: Pedestrian, 2: Cyclist)
};

// Ground plane data
struct GroundPlaneData {
    float normalX, normalY, normalZ;  // Normal vector
    float offset;                      // Plane offset
    bool valid;                        // Whether plane is valid
};

// Free space data
struct FreeSpaceData {
    uint32_t numPoints;                // Number of free space points
    float points[360 * 3];             // x, y, intensity triplets (max 360 angles)
};

// Detection packet containing all data needed for visualization
struct DetectionPacket {
    dwTime_t timestamp;                // Frame timestamp
    
    // Point cloud data (stitched)
    static constexpr uint32_t MAX_POINTS = 100000;  // Maximum points per frame
    uint32_t numPoints;                // Number of points
    float points[MAX_POINTS * 4];      // x, y, z, intensity (max points per frame)
    
    // Detection results
    uint32_t numDetections;            // Number of detected objects
    DetectionBoundingBox boxes[MAX_DETECTIONS];
    
    // Ground plane
    GroundPlaneData groundPlane;
    
    // Free space
    FreeSpaceData freeSpace;
    
    // Frame metadata
    uint32_t frameNumber;
    bool icpAligned;                   // Whether ICP alignment is complete
};

// Declare packet type ID
constexpr dw::framework::ChannelPacketTypeID DetectionPacketTypeID = dw::framework::DWFRAMEWORK_MAX_INTERNAL_TYPES + 1;

// Declare as POD type for socket transmission
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(DetectionPacket);

#endif // DETECTION_PACKET_HPP

