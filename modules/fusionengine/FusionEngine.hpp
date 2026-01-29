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

#ifndef FUSIONENGINE_LIDARIPCCLIENT_HPP
#define FUSIONENGINE_LIDARIPCCLIENT_HPP

#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

// DriveWorks headers
#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/comms/socketipc/SocketClientServer.h>

#include <framework/Checks.hpp>

#include "FusedPacket.hpp"

namespace fusionengine {

//------------------------------------------------------------------------------
// LiDAR Detection Packet
// Layout must match DetectionPacket in
// lidar_object_detection_interprocess_communiation/DetectionPacket.hpp exactly
// (same sizes and field order) so that sizeof(LidarDetectionPacket) equals
// the producer's packet size and the stream stays in sync.
//------------------------------------------------------------------------------
struct LidarDetectionPacket
{
    dwTime_t timestamp;

    static constexpr uint32_t MAX_POINTS = 100000;
    uint32_t numPoints;
    float points[MAX_POINTS * 4];  // x, y, z, intensity

    static constexpr uint32_t MAX_DETECTIONS = 100;
    uint32_t numDetections;

    struct DetectionBoundingBox
    {
        float x, y, z;
        float width, length, height;
        float rotation;
        float confidence;
        int32_t classId;
    };
    DetectionBoundingBox boxes[MAX_DETECTIONS];

    struct GroundPlane
    {
        float normalX, normalY, normalZ;
        float offset;
        bool valid;
    };
    GroundPlane groundPlane;

    // Must match producer: FreeSpaceData has MAX_FREE_SPACE_POINTS = 50000
    static constexpr uint32_t MAX_FREE_SPACE_POINTS = 50000;
    struct FreeSpace
    {
        uint32_t numPoints;
        float points[MAX_FREE_SPACE_POINTS * 3];
    };
    FreeSpace freeSpace;

    uint32_t frameNumber;
    bool icpAligned;
};

//------------------------------------------------------------------------------
// LiDAR IPC Client Class
//------------------------------------------------------------------------------
class LidarIPCClient
{
public:
    LidarIPCClient() = default;
    ~LidarIPCClient()
    {
        release();
    }

    // Non-copyable
    LidarIPCClient(const LidarIPCClient&) = delete;
    LidarIPCClient& operator=(const LidarIPCClient&) = delete;

    //--------------------------------------------------------------------------
    // Initialize the LiDAR IPC client
    //--------------------------------------------------------------------------
    bool initialize(dwContextHandle_t context,
                    const std::string& serverIP = "127.0.0.1",
                    uint16_t port = 40002)
    {
        if (m_socketClient != DW_NULL_HANDLE)
        {
            std::cerr << "[LidarIPCClient] Already initialized" << std::endl;
            return true;
        }

        m_context = context;
        m_serverIP = serverIP;
        m_port = port;

        std::cout << "[LidarIPCClient] Initializing..." << std::endl;
        std::cout << "  Server: " << m_serverIP << ":" << m_port << std::endl;

        // Initialize socket client
        dwStatus status = dwSocketClient_initialize(&m_socketClient, 1, m_context);
        if (status != DW_SUCCESS)
        {
            std::cerr << "[LidarIPCClient] Failed to initialize socket client: "
                      << dwGetStatusName(status) << std::endl;
            return false;
        }

        m_status = SensorStatus::CONNECTING;
        return true;
    }

    //--------------------------------------------------------------------------
    // Connect to the LiDAR server
    //--------------------------------------------------------------------------
    bool connect(uint32_t maxRetries = 30, uint32_t timeoutUs = 1000000)
    {
        if (m_socketClient == DW_NULL_HANDLE)
        {
            std::cerr << "[LidarIPCClient] Not initialized" << std::endl;
            return false;
        }

        std::cout << "[LidarIPCClient] Connecting to server..." << std::endl;

        dwStatus status = DW_TIME_OUT;
        uint32_t retryCount = 0;

        while (status == DW_TIME_OUT && retryCount < maxRetries)
        {
            status = dwSocketClient_connect(&m_socketConnection,
                                            m_serverIP.c_str(),
                                            m_port,
                                            timeoutUs,
                                            m_socketClient);
            if (status == DW_TIME_OUT)
            {
                retryCount++;
                std::cout << "[LidarIPCClient] Connection attempt "
                          << retryCount << "/" << maxRetries << std::endl;
            }
        }

        if (status == DW_SUCCESS)
        {
            m_connected = true;
            m_status = SensorStatus::CONNECTED;
            std::cout << "[LidarIPCClient] Connected!" << std::endl;
            return true;
        }

        std::cerr << "[LidarIPCClient] Failed to connect: "
                  << dwGetStatusName(status) << std::endl;
        m_status = SensorStatus::ERROR;
        return false;
    }

    //--------------------------------------------------------------------------
    // Receive detection packet from LiDAR server
    //--------------------------------------------------------------------------
    bool receive(LidarFrameData& frameData)
    {
        if (!m_connected)
        {
            return false;
        }

        // Receive packet in chunks (TCP may fragment large packets)
        const size_t totalSize = sizeof(LidarDetectionPacket);
        size_t bytesReceived = 0;
        uint8_t* buffer = reinterpret_cast<uint8_t*>(&m_receiveBuffer);

        while (bytesReceived < totalSize)
        {
            size_t chunkSize = totalSize - bytesReceived;
            dwStatus status = dwSocketConnection_read(
                buffer + bytesReceived,
                &chunkSize,
                m_receiveTimeoutUs,
                m_socketConnection);

            if (status == DW_END_OF_STREAM)
            {
                std::cerr << "[LidarIPCClient] Server disconnected" << std::endl;
                m_connected = false;
                m_status = SensorStatus::DISCONNECTED;
                return false;
            }
            else if (status == DW_TIME_OUT)
            {
                // Timeout is non-fatal, just no data available
                return false;
            }
            else if (status != DW_SUCCESS)
            {
                m_receiveErrors++;
                if (m_receiveErrors <= 5)
                {
                    std::cerr << "[LidarIPCClient] Receive error: "
                              << dwGetStatusName(status) << std::endl;
                }
                return false;
            }

            bytesReceived += chunkSize;
        }

        // Successfully received full packet - convert to LidarFrameData
        m_status = SensorStatus::RECEIVING;
        convertToFrameData(m_receiveBuffer, frameData);
        m_packetsReceived++;

        return true;
    }

    //--------------------------------------------------------------------------
    // Release resources
    //--------------------------------------------------------------------------
    void release()
    {
        if (m_socketConnection != DW_NULL_HANDLE)
        {
            dwSocketConnection_release(m_socketConnection);
            m_socketConnection = DW_NULL_HANDLE;
        }

        if (m_socketClient != DW_NULL_HANDLE)
        {
            dwSocketClient_release(m_socketClient);
            m_socketClient = DW_NULL_HANDLE;
        }

        m_connected = false;
        m_status = SensorStatus::DISCONNECTED;
    }

    //--------------------------------------------------------------------------
    // Status accessors
    //--------------------------------------------------------------------------
    bool isConnected() const { return m_connected; }
    SensorStatus getStatus() const { return m_status; }
    uint64_t getPacketsReceived() const { return m_packetsReceived; }
    uint64_t getReceiveErrors() const { return m_receiveErrors; }

private:
    //--------------------------------------------------------------------------
    // Convert raw packet to LidarFrameData structure
    //--------------------------------------------------------------------------
    void convertToFrameData(const LidarDetectionPacket& packet,
                            LidarFrameData& frameData)
    {
        frameData.timestamp = packet.timestamp;
        frameData.frameNumber = packet.frameNumber;
        frameData.icpAligned = packet.icpAligned;

        // Copy point cloud
        frameData.numPoints = packet.numPoints;
        frameData.points.resize(packet.numPoints * 4);
        std::memcpy(frameData.points.data(),
                    packet.points,
                    packet.numPoints * 4 * sizeof(float));

        // Copy detections
        frameData.detections.resize(packet.numDetections);
        for (uint32_t i = 0; i < packet.numDetections; ++i)
        {
            const auto& src = packet.boxes[i];
            auto& dst = frameData.detections[i];

            dst.x = src.x;
            dst.y = src.y;
            dst.z = src.z;
            dst.width = src.width;
            dst.length = src.length;
            dst.height = src.height;
            dst.rotation = src.rotation;
            dst.confidence = src.confidence;
            dst.classId = src.classId;
        }

        // Copy ground plane
        frameData.groundPlane.normalX = packet.groundPlane.normalX;
        frameData.groundPlane.normalY = packet.groundPlane.normalY;
        frameData.groundPlane.normalZ = packet.groundPlane.normalZ;
        frameData.groundPlane.offset = packet.groundPlane.offset;
        frameData.groundPlane.valid = packet.groundPlane.valid;

        // Copy free space (FusedPacket uses MAX_FREE_SPACE_POINTS=360; truncate if needed)
        const uint32_t maxFreeSpace = static_cast<uint32_t>(
            sizeof(frameData.freeSpace.points) / (sizeof(float) * 3));
        frameData.freeSpace.numPoints =
            std::min(packet.freeSpace.numPoints, maxFreeSpace);
        if (frameData.freeSpace.numPoints > 0)
        {
            std::memcpy(frameData.freeSpace.points,
                        packet.freeSpace.points,
                        frameData.freeSpace.numPoints * 3 * sizeof(float));
        }

        frameData.valid = true;
    }

private:
    // DriveWorks context
    dwContextHandle_t m_context{DW_NULL_HANDLE};

    // Socket IPC handles
    dwSocketClientHandle_t m_socketClient{DW_NULL_HANDLE};
    dwSocketConnectionHandle_t m_socketConnection{DW_NULL_HANDLE};

    // Connection parameters
    std::string m_serverIP{"127.0.0.1"};
    uint16_t m_port{40002};

    // Receive configuration
    uint32_t m_receiveTimeoutUs{1000000};  // 1 second

    // State
    std::atomic<bool> m_connected{false};
    std::atomic<SensorStatus> m_status{SensorStatus::DISCONNECTED};

    // Statistics
    std::atomic<uint64_t> m_packetsReceived{0};
    std::atomic<uint64_t> m_receiveErrors{0};

    // Receive buffer (large POD structure)
    LidarDetectionPacket m_receiveBuffer{};
};

} // namespace fusionengine

#endif // FUSIONENGINE_LIDARIPCCLIENT_HPP
