/////////////////////////////////////////////////////////////////////////////////////////
// Livox HAP Plugin for NVIDIA DriveWorks
//
// Copyright (c) 2025 - All rights reserved.
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef LIVOX_HAP_PROPERTIES_HPP
#define LIVOX_HAP_PROPERTIES_HPP

#include <cmath>
#include <cstdint>

#include <dw/core/base/Types.h>

namespace dw
{
namespace plugin
{
namespace lidar
{

//################################################################################
//################ Sensor specific parameters and data structures ################
//################################################################################

// GENERAL - HAP specific parameters
static const uint32_t MAX_POINTS_PER_FRAME    = 120000;  // Maximum points per frame (1.2M points/s ÷ 10Hz)
static const uint32_t POINT_STRIDE            = 4U;     // x, y, z and intensity (each 4 bytes)
static const uint16_t MAX_UDP_PAYLOAD_SIZE    = 1500;   // Standard Ethernet MTU size
static const uint32_t PACKETS_PER_FRAME       = 900;    // Approximation for Livox HAP at 10Hz
static const uint16_t MAX_FRAME_RATE          = 20;     // Maximum frame rate
static const uint32_t MAX_POINTS_PER_PACKET   = 100;    // Typical points per Livox packet
static const uint32_t MAX_PAYLOAD_SIZE        = PACKETS_PER_FRAME * MAX_UDP_PAYLOAD_SIZE;
static const uint16_t SLOT_SIZE               = 30;     // Buffer pool size
static const uint32_t PAYLOAD_OFFSET          = sizeof(uint32_t) + sizeof(dwTime_t);
static const bool FLIP_X_AXIS = false;  // Set to true if X axis needs to be flipped
static const bool FLIP_Y_AXIS = false;  // Set to true if Y axis needs to be flipped
static const bool FLIP_Z_AXIS = false;  // Set to true if Z axis needs to be flipped
static const float ROTATION_Z = 0.0f;   // Rotation around Z axis in radians

static const bool ENABLE_DIRECT_PROCESSING = false;  // Whether to process packets immediately

// Point validation parameters
static const float MAX_VALID_DISTANCE_M       = 200.0f;  // Maximum valid distance in meters
static const int32_t MAX_VALID_DISTANCE_MM    = 200000;  // Maximum valid distance in millimeters
static const int16_t MAX_VALID_DISTANCE_CM    = 20000;   // Maximum valid distance in centimeters

// Queue management 
static const size_t MAX_QUEUE_SIZE = 200;  // Maximum queue size before dropping packets
static const uint32_t MIN_POINTS_FOR_FRAME = 1000;  // Minimum points needed for a complete frame

// HAP specific constants
static const uint8_t HAP_DEVICE_TYPE = 15;  // kLivoxLidarTypeHAP from livox_lidar_def.h
static const uint8_t HAP_INDUSTRIAL_DEVICE_TYPE = 10;  // kLivoxLidarTypeIndustrialHAP from livox_lidar_def.h
static const uint16_t HAP_DISCOVERY_PORT = 56000;  // HAP discovery port
static const uint16_t HAP_CONTROL_PORT = 57000;    // HAP control port

// DATA STRUCTURES
#pragma pack(1)
typedef struct
{
    uint8_t rawData[MAX_UDP_PAYLOAD_SIZE + PAYLOAD_OFFSET];
} RawPacket;

// Points in Cartesian coordinates (XYZ + intensity)
typedef struct
{
    float x;           // 4 Bytes [m]
    float y;           // 4       [m]
    float z;           // 4       [m]
    float intensity;   // 4       [0.0-1.0]
} LivoxPointXYZI;      // 16 bytes total

// Points in Polar coordinates (radius, theta, phi + intensity)
typedef struct
{
    float radius;      // 4 Bytes [m]
    float theta;       // 4       [rad]
    float phi;         // 4       [rad]
    float intensity;   // 4       [0.0-1.0]
} LivoxPointRTHI;      // 16 bytes total

// Structure to hold processed point cloud packet
typedef struct
{
    bool scan_complete;                               // 1  byte
    dwTime_t sensor_timestamp;                        // 8  bytes
    uint32_t max_points;                              // 4  bytes
    uint32_t n_points;                                // 4  bytes
    LivoxPointXYZI xyzi[MAX_POINTS_PER_PACKET];      // Array of XYZI points
    LivoxPointRTHI rthi[MAX_POINTS_PER_PACKET];      // Array of RTHI points
} LivoxLidarPacket;
#pragma pack()

// HAP specific data structures based on the communication protocol
#pragma pack(1)
// HAP Ethernet packet header structure
typedef struct {
    uint8_t version;
    uint8_t slot;
    uint8_t id;
    uint8_t reserved;
    uint32_t timestamp_type;
    uint64_t timestamp;
    uint8_t data_type;
} HapEthPacketHeader;

// HAP point data structure (Type 1 - High precision Cartesian)
typedef struct {
    int32_t x;            // X axis, Unit: mm
    int32_t y;            // Y axis, Unit: mm
    int32_t z;            // Z axis, Unit: mm
    uint8_t reflectivity; // Reflectivity
    uint8_t tag;          // Tag
} HapCartesianHighPoint;

// HAP point data structure (Type 2 - Low precision Cartesian)
typedef struct {
    int16_t x;            // X axis, Unit: cm
    int16_t y;            // Y axis, Unit: cm
    int16_t z;            // Z axis, Unit: cm
    uint8_t reflectivity; // Reflectivity
    uint8_t tag;          // Tag
} HapCartesianLowPoint;

// HAP point data structure (Type 3 - Spherical)
typedef struct {
    uint32_t depth;       // Depth, Unit: mm
    uint16_t theta;       // Theta, Unit: 0.01°
    uint16_t phi;         // Phi, Unit: 0.01°
    uint8_t reflectivity; // Reflectivity
    uint8_t tag;          // Tag
} HapSphericalPoint;
#pragma pack()

} // namespace lidar
} // namespace plugin
} // namespace dw

#endif // LIVOX_HAP_PROPERTIES_HPP 