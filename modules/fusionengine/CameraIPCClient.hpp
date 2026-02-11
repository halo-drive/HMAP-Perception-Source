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

#ifndef FUSIONENGINE_CAMERAIPCCLIENT_HPP
#define FUSIONENGINE_CAMERAIPCCLIENT_HPP

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
// Camera IPC Data Structures (must match driveseg_object/drivesegserver.cu)
//------------------------------------------------------------------------------
static constexpr uint32_t CAMERA_FRAME_HEADER_MAGIC = 0xDEADBEEF;

#pragma pack(push, 1)
struct CameraFrameHeader
{
    uint32_t magic;             // 0xDEADBEEF
    uint32_t cameraIndex;
    uint64_t frameId;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t imageDataSize;
    uint32_t numBoxes;
    uint32_t segCount;
    uint32_t detCount;
    float avgSegMs;
    float avgDetMs;
    float avgStage2Ms;
    uint32_t num3DBoxes;
};

struct CameraDetectionBox
{
    float x, y, width, height;
    char label[64];
};

struct CameraDetection3DBox
{
    float depth;        // meters
    float height;       // meters
    float width;        // meters
    float length;       // meters
    float rotation;     // radians
    float iouScore;     // quality score
};
#pragma pack(pop)

//------------------------------------------------------------------------------
// Camera IPC Client Class (handles single camera connection)
//------------------------------------------------------------------------------
class CameraIPCClient
{
public:
    CameraIPCClient() = default;
    ~CameraIPCClient()
    {
        release();
    }

    // Non-copyable
    CameraIPCClient(const CameraIPCClient&) = delete;
    CameraIPCClient& operator=(const CameraIPCClient&) = delete;

    //--------------------------------------------------------------------------
    // Initialize the camera IPC client
    //--------------------------------------------------------------------------
    bool initialize(dwContextHandle_t context,
                    uint32_t cameraIndex,
                    const std::string& serverIP = "127.0.0.1",
                    uint16_t basePort = 49252)
    {
        if (m_socketClient != DW_NULL_HANDLE)
        {
            std::cerr << "[CameraIPCClient:" << cameraIndex
                      << "] Already initialized" << std::endl;
            return true;
        }

        m_context = context;
        m_cameraIndex = cameraIndex;
        m_serverIP = serverIP;
        m_port = basePort + cameraIndex;

        std::cout << "[CameraIPCClient:" << m_cameraIndex << "] Initializing..."
                  << std::endl;
        std::cout << "  Server: " << m_serverIP << ":" << m_port << std::endl;

        // Initialize socket client
        dwStatus status = dwSocketClient_initialize(&m_socketClient, 1, m_context);
        if (status != DW_SUCCESS)
        {
            std::cerr << "[CameraIPCClient:" << m_cameraIndex
                      << "] Failed to initialize socket client: "
                      << dwGetStatusName(status) << std::endl;
            return false;
        }

        m_status = SensorStatus::CONNECTING;
        return true;
    }

    //--------------------------------------------------------------------------
    // Connect to the camera DNN server
    //--------------------------------------------------------------------------
    bool connect(uint32_t maxRetries = 30, uint32_t timeoutUs = 1000000)
    {
        if (m_socketClient == DW_NULL_HANDLE)
        {
            std::cerr << "[CameraIPCClient:" << m_cameraIndex
                      << "] Not initialized" << std::endl;
            return false;
        }

        std::cout << "[CameraIPCClient:" << m_cameraIndex
                  << "] Connecting to server..." << std::endl;

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
                std::cout << "[CameraIPCClient:" << m_cameraIndex
                          << "] Connection attempt " << retryCount << "/"
                          << maxRetries << std::endl;
            }
        }

        if (status == DW_SUCCESS)
        {
            m_connected = true;
            m_status = SensorStatus::CONNECTED;
            std::cout << "[CameraIPCClient:" << m_cameraIndex
                      << "] Connected!" << std::endl;
            return true;
        }

        std::cerr << "[CameraIPCClient:" << m_cameraIndex
                  << "] Failed to connect: " << dwGetStatusName(status)
                  << std::endl;
        m_status = SensorStatus::ERROR;
        return false;
    }

    //--------------------------------------------------------------------------
    // Receive frame from camera DNN server
    //--------------------------------------------------------------------------
    bool receive(CameraFrameData& frameData)
    {
        if (!m_connected)
        {
            return false;
        }

        // Step 1: Peek header to validate magic
        CameraFrameHeader header{};
        if (!peekAndValidateHeader(header))
        {
            return false;
        }

        // Step 2: Read full header
        if (!readHeader(header))
        {
            return false;
        }

        // Step 3: Read RGBA image data
        std::vector<uint8_t> imageData(header.imageDataSize);
        if (!readImageData(imageData, header.imageDataSize))
        {
            return false;
        }

        // Step 4: Read 2D detection boxes
        std::vector<CameraDetectionBox> boxes2D(header.numBoxes);
        if (!readDetection2DBoxes(boxes2D, header.numBoxes))
        {
            return false;
        }

        // Step 5: Read 3D detection boxes
        std::vector<CameraDetection3DBox> boxes3D(header.num3DBoxes);
        if (!readDetection3DBoxes(boxes3D, header.num3DBoxes))
        {
            return false;
        }

        // Successfully received frame - convert to CameraFrameData
        m_status = SensorStatus::RECEIVING;
        convertToFrameData(header, imageData, boxes2D, boxes3D, frameData);
        m_framesReceived++;

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
    uint32_t getCameraIndex() const { return m_cameraIndex; }
    uint64_t getFramesReceived() const { return m_framesReceived; }
    uint64_t getReceiveErrors() const { return m_receiveErrors; }

private:
    //--------------------------------------------------------------------------
    // Peek header and validate magic number
    //--------------------------------------------------------------------------
    bool peekAndValidateHeader(CameraFrameHeader& header)
    {
        size_t peekSize = sizeof(CameraFrameHeader);
        dwStatus status = dwSocketConnection_peek(
            reinterpret_cast<uint8_t*>(&header),
            &peekSize,
            m_peekTimeoutUs,
            m_socketConnection);

        if (status == DW_TIME_OUT)
        {
            return false;  // No data available
        }

        if (status != DW_SUCCESS)
        {
            handleReceiveError(status);
            return false;
        }

        // Validate magic number
        if (header.magic != CAMERA_FRAME_HEADER_MAGIC)
        {
            // Try to resync
            if (!resyncToMagic())
            {
                return false;
            }
            // Re-peek after resync
            return peekAndValidateHeader(header);
        }

        return true;
    }

    //--------------------------------------------------------------------------
    // Resync stream to find valid magic number
    // Discards bytes one at a time and peeks for valid header.
    // This ensures magic bytes remain in stream for subsequent readHeader().
    //--------------------------------------------------------------------------
    bool resyncToMagic()
    {
        std::cout << "[CameraIPCClient:" << m_cameraIndex
                  << "] Attempting resync..." << std::endl;

        const uint32_t maxAttempts = 1000;

        for (uint32_t i = 0; i < maxAttempts; ++i)
        {
            // Discard 1 byte
            uint8_t discard;
            size_t discardSize = 1;
            dwStatus status = dwSocketConnection_read(
                &discard,
                &discardSize,
                m_receiveTimeoutUs,
                m_socketConnection);

            if (status != DW_SUCCESS || discardSize != 1)
            {
                std::cerr << "[CameraIPCClient:" << m_cameraIndex
                          << "] Resync read failed: " << dwGetStatusName(status)
                          << std::endl;
                return false;
            }

            // Peek for valid header
            CameraFrameHeader peekHeader{};
            size_t peekSize = sizeof(CameraFrameHeader);
            status = dwSocketConnection_peek(
                reinterpret_cast<uint8_t*>(&peekHeader),
                &peekSize,
                1000,  // Short timeout for peek
                m_socketConnection);

            if (status == DW_SUCCESS &&
                peekSize == sizeof(CameraFrameHeader) &&
                peekHeader.magic == CAMERA_FRAME_HEADER_MAGIC)
            {
                std::cout << "[CameraIPCClient:" << m_cameraIndex
                          << "] Resynced after discarding " << (i + 1)
                          << " bytes" << std::endl;
                return true;
            }
        }

        std::cerr << "[CameraIPCClient:" << m_cameraIndex
                  << "] Failed to resync after " << maxAttempts
                  << " attempts" << std::endl;
        return false;
    }

    //--------------------------------------------------------------------------
    // Read full header
    //--------------------------------------------------------------------------
    bool readHeader(CameraFrameHeader& header)
    {
        size_t headerSize = sizeof(CameraFrameHeader);
        dwStatus status = dwSocketConnection_read(
            reinterpret_cast<uint8_t*>(&header),
            &headerSize,
            m_receiveTimeoutUs,
            m_socketConnection);

        if (status != DW_SUCCESS || headerSize != sizeof(CameraFrameHeader))
        {
            handleReceiveError(status);
            return false;
        }

        return true;
    }

    //--------------------------------------------------------------------------
    // Read image data with retry logic
    //--------------------------------------------------------------------------
    bool readImageData(std::vector<uint8_t>& imageData, uint32_t totalSize)
    {
        size_t bytesRead = 0;
        uint32_t retries = 0;
        const uint32_t maxRetries = 5;

        while (bytesRead < totalSize && retries < maxRetries)
        {
            size_t chunkSize = totalSize - bytesRead;
            dwStatus status = dwSocketConnection_read(
                imageData.data() + bytesRead,
                &chunkSize,
                m_imageReadTimeoutUs,
                m_socketConnection);

            if (status == DW_END_OF_STREAM)
            {
                std::cerr << "[CameraIPCClient:" << m_cameraIndex
                          << "] Server disconnected during image read"
                          << std::endl;
                m_connected = false;
                m_status = SensorStatus::DISCONNECTED;
                return false;
            }
            else if (status == DW_TIME_OUT)
            {
                retries++;
                continue;
            }
            else if (status != DW_SUCCESS)
            {
                handleReceiveError(status);
                return false;
            }

            bytesRead += chunkSize;
        }

        return bytesRead == totalSize;
    }

    //--------------------------------------------------------------------------
    // Read 2D detection boxes
    //--------------------------------------------------------------------------
    bool readDetection2DBoxes(std::vector<CameraDetectionBox>& boxes,
                               uint32_t numBoxes)
    {
        if (numBoxes == 0)
        {
            return true;
        }

        for (uint32_t i = 0; i < numBoxes; ++i)
        {
            size_t boxSize = sizeof(CameraDetectionBox);
            dwStatus status = dwSocketConnection_read(
                reinterpret_cast<uint8_t*>(&boxes[i]),
                &boxSize,
                m_receiveTimeoutUs,
                m_socketConnection);

            if (status != DW_SUCCESS)
            {
                handleReceiveError(status);
                return false;
            }
        }

        return true;
    }

    //--------------------------------------------------------------------------
    // Read 3D detection boxes
    //--------------------------------------------------------------------------
    bool readDetection3DBoxes(std::vector<CameraDetection3DBox>& boxes,
                               uint32_t numBoxes)
    {
        if (numBoxes == 0)
        {
            return true;
        }

        for (uint32_t i = 0; i < numBoxes; ++i)
        {
            size_t boxSize = sizeof(CameraDetection3DBox);
            dwStatus status = dwSocketConnection_read(
                reinterpret_cast<uint8_t*>(&boxes[i]),
                &boxSize,
                m_receiveTimeoutUs,
                m_socketConnection);

            if (status != DW_SUCCESS)
            {
                handleReceiveError(status);
                return false;
            }
        }

        return true;
    }

    //--------------------------------------------------------------------------
    // Handle receive errors
    //--------------------------------------------------------------------------
    void handleReceiveError(dwStatus status)
    {
        m_receiveErrors++;

        if (status == DW_END_OF_STREAM)
        {
            std::cerr << "[CameraIPCClient:" << m_cameraIndex
                      << "] Server disconnected" << std::endl;
            m_connected = false;
            m_status = SensorStatus::DISCONNECTED;
        }
        else if (m_receiveErrors <= 5)
        {
            std::cerr << "[CameraIPCClient:" << m_cameraIndex
                      << "] Receive error: " << dwGetStatusName(status)
                      << std::endl;
        }
    }

    //--------------------------------------------------------------------------
    // Convert raw data to CameraFrameData structure
    //--------------------------------------------------------------------------
    void convertToFrameData(const CameraFrameHeader& header,
                            const std::vector<uint8_t>& imageData,
                            const std::vector<CameraDetectionBox>& boxes2D,
                            const std::vector<CameraDetection3DBox>& boxes3D,
                            CameraFrameData& frameData)
    {
        frameData.cameraIndex = header.cameraIndex;
        frameData.frameId = header.frameId;
        // Use current system time since server header doesn't include timestamp
        frameData.timestamp = getCurrentTimestamp();

        // Image data
        frameData.width = header.width;
        frameData.height = header.height;
        frameData.rgbaPixels = imageData;

        // 2D detections
        frameData.boxes2D.resize(boxes2D.size());
        for (size_t i = 0; i < boxes2D.size(); ++i)
        {
            const auto& src = boxes2D[i];
            auto& dst = frameData.boxes2D[i];

            dst.x = src.x;
            dst.y = src.y;
            dst.width = src.width;
            dst.height = src.height;
            std::strncpy(dst.label, src.label, sizeof(dst.label) - 1);
            dst.label[sizeof(dst.label) - 1] = '\0';
            dst.cameraIndex = header.cameraIndex;
        }

        // 3D detections
        frameData.boxes3D.resize(boxes3D.size());
        for (size_t i = 0; i < boxes3D.size(); ++i)
        {
            const auto& src = boxes3D[i];
            auto& dst = frameData.boxes3D[i];

            dst.depth = src.depth;
            dst.height = src.height;
            dst.width = src.width;
            dst.length = src.length;
            dst.rotation = src.rotation;
            dst.iouScore = src.iouScore;
            dst.cameraIndex = header.cameraIndex;

            // Associate with 2D box if available
            if (i < boxes2D.size())
            {
                dst.box2D.x = boxes2D[i].x;
                dst.box2D.y = boxes2D[i].y;
                dst.box2D.width = boxes2D[i].width;
                dst.box2D.height = boxes2D[i].height;
            }
        }

        // Statistics
        frameData.segCount = header.segCount;
        frameData.detCount = header.detCount;
        frameData.avgSegMs = header.avgSegMs;
        frameData.avgDetMs = header.avgDetMs;
        frameData.avgStage2Ms = header.avgStage2Ms;

        frameData.valid = true;
    }

    //--------------------------------------------------------------------------
    // Get current timestamp in microseconds
    //--------------------------------------------------------------------------
    static dwTime_t getCurrentTimestamp()
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now.time_since_epoch();
        return static_cast<dwTime_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(duration).count());
    }

private:
    // DriveWorks context
    dwContextHandle_t m_context{DW_NULL_HANDLE};

    // Socket IPC handles
    dwSocketClientHandle_t m_socketClient{DW_NULL_HANDLE};
    dwSocketConnectionHandle_t m_socketConnection{DW_NULL_HANDLE};

    // Connection parameters
    uint32_t m_cameraIndex{0};
    std::string m_serverIP{"127.0.0.1"};
    uint16_t m_port{49252};

    // Receive configuration
    uint32_t m_peekTimeoutUs{5000000};       // 5 seconds for peek
    uint32_t m_receiveTimeoutUs{5000000};    // 5 seconds for header/boxes
    uint32_t m_imageReadTimeoutUs{20000000}; // 20 seconds for image data

    // State
    std::atomic<bool> m_connected{false};
    std::atomic<SensorStatus> m_status{SensorStatus::DISCONNECTED};

    // Statistics
    std::atomic<uint64_t> m_framesReceived{0};
    std::atomic<uint64_t> m_receiveErrors{0};
};

//------------------------------------------------------------------------------
// Multi-Camera IPC Client Manager
//------------------------------------------------------------------------------
class MultiCameraIPCClient
{
public:
    MultiCameraIPCClient() = default;
    ~MultiCameraIPCClient() = default;

    //--------------------------------------------------------------------------
    // Initialize all camera clients
    //--------------------------------------------------------------------------
    bool initialize(dwContextHandle_t context,
                    uint32_t numCameras = 4,
                    const std::string& serverIP = "127.0.0.1",
                    uint16_t basePort = 49252)
    {
        m_numCameras = std::min(numCameras, MAX_CAMERAS);

        std::cout << "[MultiCameraIPCClient] Initializing " << m_numCameras
                  << " camera clients..." << std::endl;

        for (uint32_t i = 0; i < m_numCameras; ++i)
        {
            if (!m_clients[i].initialize(context, i, serverIP, basePort))
            {
                std::cerr << "[MultiCameraIPCClient] Failed to initialize "
                          << "camera " << i << std::endl;
                return false;
            }
        }

        return true;
    }

    //--------------------------------------------------------------------------
    // Connect all camera clients
    //--------------------------------------------------------------------------
    bool connectAll(uint32_t maxRetries = 30, uint32_t timeoutUs = 1000000)
    {
        bool allConnected = true;

        for (uint32_t i = 0; i < m_numCameras; ++i)
        {
            if (!m_clients[i].connect(maxRetries, timeoutUs))
            {
                std::cerr << "[MultiCameraIPCClient] Failed to connect "
                          << "camera " << i << std::endl;
                allConnected = false;
            }
        }

        return allConnected;
    }

    //--------------------------------------------------------------------------
    // Receive from specific camera
    //--------------------------------------------------------------------------
    bool receive(uint32_t cameraIndex, CameraFrameData& frameData)
    {
        if (cameraIndex >= m_numCameras)
        {
            return false;
        }

        return m_clients[cameraIndex].receive(frameData);
    }

    //--------------------------------------------------------------------------
    // Receive from all cameras (non-blocking)
    //--------------------------------------------------------------------------
    uint32_t receiveAll(std::array<CameraFrameData, MAX_CAMERAS>& frames)
    {
        uint32_t receivedCount = 0;

        for (uint32_t i = 0; i < m_numCameras; ++i)
        {
            if (m_clients[i].receive(frames[i]))
            {
                receivedCount++;
            }
        }

        return receivedCount;
    }

    //--------------------------------------------------------------------------
    // Release all camera clients
    //--------------------------------------------------------------------------
    void release()
    {
        for (uint32_t i = 0; i < m_numCameras; ++i)
        {
            m_clients[i].release();
        }
    }

    //--------------------------------------------------------------------------
    // Status accessors
    //--------------------------------------------------------------------------
    uint32_t getNumCameras() const { return m_numCameras; }
    bool isConnected(uint32_t cameraIndex) const
    {
        return cameraIndex < m_numCameras &&
               m_clients[cameraIndex].isConnected();
    }

    CameraIPCClient& getClient(uint32_t cameraIndex)
    {
        return m_clients[cameraIndex];
    }

private:
    uint32_t m_numCameras{0};
    std::array<CameraIPCClient, MAX_CAMERAS> m_clients;
};

} // namespace fusionengine

#endif // FUSIONENGINE_CAMERAIPCCLIENT_HPP
