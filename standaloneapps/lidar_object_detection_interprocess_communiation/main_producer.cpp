#include <cstdint>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <unistd.h>
#include <thread>
#include <atomic>

#include <framework/ProgramArguments.hpp>
#include <framework/SamplesDataPath.hpp>
#include <framework/Checks.hpp>

#include <dw/core/base/Version.h>
#include <dw/core/logger/Logger.h>
#include <dw/comms/socketipc/SocketClientServer.h>

#include "InterLidarICP.hpp"
#include "DetectionPacket.hpp"

// Producer application: gathers data, runs inference, sends results via socket
class DetectionProducer : public InterLidarICP
{
private:
    // Socket IPC components
    dwSocketServerHandle_t m_socketServer{DW_NULL_HANDLE};
    dwSocketConnectionHandle_t m_socketConnection{DW_NULL_HANDLE};
    
    // IPC parameters
    uint16_t m_port{40002};
    bool m_ipcEnabled{true};
    std::atomic<bool> m_connected{false};
    
public:
    DetectionProducer(const ProgramArguments& args) : InterLidarICP(args)
    {
        // Parse IPC parameters
        m_port = static_cast<uint16_t>(std::stoul(args.get("port")));
        m_ipcEnabled = args.get("ipc") != "false";
    }
    
    bool initializeIPC()
    {
        if (!m_ipcEnabled) {
            std::cout << "IPC disabled, running in standalone mode" << std::endl;
            return true;
        }
        
        // Guard against double initialization
        if (m_socketServer != DW_NULL_HANDLE) {
            std::cout << "IPC producer already initialized, skipping..." << std::endl;
            return true;
        }
        
        std::cout << "Initializing IPC producer (Socket Server)..." << std::endl;
        std::cout << "  Port: " << m_port << std::endl;
        
        // Initialize socket server
        dwContextHandle_t ctx = this->getContext();
        CHECK_DW_ERROR(dwSocketServer_initialize(&m_socketServer, m_port, 1, ctx));
        
        std::cout << "Socket server initialized, waiting for consumer to connect..." << std::endl;
        
        // Try to accept connection (non-blocking initially)
        dwStatus status = dwSocketServer_accept(&m_socketConnection, 100000, m_socketServer);  // 100ms timeout
        if (status == DW_SUCCESS) {
            m_connected = true;
            std::cout << "Producer: Consumer connected!" << std::endl;
        } else if (status == DW_TIME_OUT) {
            std::cout << "Producer: No consumer yet, will accept during processing" << std::endl;
        } else {
            std::cerr << "Producer: Error accepting connection: " << dwGetStatusName(status) << std::endl;
        }
        
        return true;
    }
    
    void sendDetectionResults()
    {
        if (!m_ipcEnabled) {
            return;
        }
        
        // Try to accept connection if not connected yet
        if (!m_connected && m_socketServer != DW_NULL_HANDLE) {
            dwStatus status = dwSocketServer_accept(&m_socketConnection, 100, m_socketServer);  // 100µs timeout
            if (status == DW_SUCCESS) {
                m_connected = true;
                std::cout << "Producer: Consumer connected!" << std::endl;
            }
            
            if (!m_connected) {
                static uint32_t checkCount = 0;
                if (++checkCount % 1000 == 0) {
                    std::cout << "Producer: Waiting for consumer... (count: " << checkCount << ")" << std::endl;
                }
                return;
            }
        }
        
        if (!m_connected) {
            return;
        }
        
        // Create and fill packet
        DetectionPacket packet{};
        packet.timestamp = getClockRealtime();
        packet.frameNumber = this->getFrameNum();
        auto alignmentState = this->getAlignmentState();
        packet.icpAligned = (alignmentState == InterLidarICP::AlignmentState::ALIGNED || 
                            alignmentState == InterLidarICP::AlignmentState::REALIGNMENT);
        
        // Copy point cloud data
        const dwPointCloud& stitchedHost = this->getStitchedPointsHost();
        packet.numPoints = std::min(stitchedHost.size, 
                                    static_cast<uint32_t>(DetectionPacket::MAX_POINTS));
        const dwVector4f* points = static_cast<const dwVector4f*>(stitchedHost.points);
        for (uint32_t i = 0; i < packet.numPoints; ++i) {
            packet.points[i * 4 + 0] = points[i].x;
            packet.points[i * 4 + 1] = points[i].y;
            packet.points[i * 4 + 2] = points[i].z;
            packet.points[i * 4 + 3] = points[i].w;  // intensity
        }
        
        // Copy detection boxes
        const std::vector<BoundingBox>& boxes = this->getCurrentBoxes();
        packet.numDetections = std::min(static_cast<uint32_t>(boxes.size()), MAX_DETECTIONS);
        for (uint32_t i = 0; i < packet.numDetections; ++i) {
            packet.boxes[i].x = boxes[i].x;
            packet.boxes[i].y = boxes[i].y;
            packet.boxes[i].z = boxes[i].z;
            packet.boxes[i].width = boxes[i].width;
            packet.boxes[i].length = boxes[i].length;
            packet.boxes[i].height = boxes[i].height;
            packet.boxes[i].rotation = boxes[i].rotation;
            packet.boxes[i].confidence = boxes[i].confidence;
            packet.boxes[i].classId = boxes[i].classId;
        }
        
        // Copy ground plane data
        if (this->isFilteredGroundPlaneValid()) {
            const dwPointCloudExtractedPlane& plane = this->getFilteredGroundPlane();
            packet.groundPlane.normalX = plane.normal.x;
            packet.groundPlane.normalY = plane.normal.y;
            packet.groundPlane.normalZ = plane.normal.z;
            packet.groundPlane.offset = plane.offset;
            packet.groundPlane.valid = true;
        } else {
            packet.groundPlane.valid = false;
        }
        
        // Copy free space data
        if (this->isFreeSpaceEnabled()) {
            const std::vector<float32_t>& freeSpacePts = this->getFreeSpacePoints();
            if (!freeSpacePts.empty()) {
                packet.freeSpace.numPoints = std::min(static_cast<uint32_t>(freeSpacePts.size() / 3), 360u);
                for (uint32_t i = 0; i < packet.freeSpace.numPoints * 3; ++i) {
                    packet.freeSpace.points[i] = freeSpacePts[i];
                }
            } else {
                packet.freeSpace.numPoints = 0;
            }
        } else {
            packet.freeSpace.numPoints = 0;
        }
        
        // Send packet over socket (handle large packet fragmentation)
        const size_t totalSize = sizeof(DetectionPacket);
        size_t bytesSent = 0;
        const uint8_t* buffer = reinterpret_cast<const uint8_t*>(&packet);
        
        while (bytesSent < totalSize) {
            size_t chunkSize = totalSize - bytesSent;
            dwStatus sendStatus = dwSocketConnection_write(const_cast<uint8_t*>(buffer + bytesSent), 
                                                           &chunkSize, 
                                                           1000000,  // 1 second timeout per chunk
                                                           m_socketConnection);
            
            if (sendStatus == DW_END_OF_STREAM) {
                std::cerr << "Producer: Consumer disconnected during send" << std::endl;
                m_connected = false;
                dwSocketConnection_release(m_socketConnection);
                m_socketConnection = DW_NULL_HANDLE;
                return;
            } else if (sendStatus != DW_SUCCESS) {
                static uint32_t sendErrorCount = 0;
                if (++sendErrorCount <= 5) {
                    std::cerr << "Producer: Failed to send chunk: " << dwGetStatusName(sendStatus) << std::endl;
                }
                return;
            }
            
            bytesSent += chunkSize;
        }
        
        // Validate that we sent everything
        if (bytesSent != totalSize) {
            std::cerr << "Producer: Incomplete send (expected " << totalSize 
                      << " bytes, sent " << bytesSent << " bytes)" << std::endl;
            return;
        }
        
        // Log successful send
        static uint32_t sendCount = 0;
        sendCount++;
        if (sendCount == 1 || sendCount % 30 == 0) {
            std::cout << "Producer: Sent packet #" << sendCount 
                      << " (frame=" << packet.frameNumber 
                      << ", points=" << packet.numPoints 
                      << ", detections=" << packet.numDetections 
                      << ", size=" << (sizeof(DetectionPacket) / 1024) << "KB)" << std::endl;
        }
    }
    
    dwTime_t getClockRealtime()
    {
        struct timespec time;
        clock_gettime(CLOCK_REALTIME, &time);
        return time.tv_sec * 1000000000LL + time.tv_nsec;
    }
    
    // Override onInitialize to add IPC initialization
    bool onInitialize() override
    {
        bool result = InterLidarICP::onInitialize();
        if (result && m_ipcEnabled) {
            result = initializeIPC();
        }
        return result;
    }
    
    // Override onProcess to send data after processing
    void onProcess() override
    {
        // Call parent processing
        InterLidarICP::onProcess();
        
        // Send results via IPC
        if (m_ipcEnabled) {
            sendDetectionResults();
        }
    }
    
    // Override onRelease to clean up IPC
    void onRelease() override
    {
        if (m_ipcEnabled) {
            if (m_socketConnection != DW_NULL_HANDLE) {
                dwSocketConnection_release(m_socketConnection);
                m_socketConnection = DW_NULL_HANDLE;
            }
            if (m_socketServer != DW_NULL_HANDLE) {
                dwSocketServer_release(m_socketServer);
                m_socketServer = DW_NULL_HANDLE;
            }
        }
        
        InterLidarICP::onRelease();
    }
    
    // Simple running flag for standalone mode
    bool isRunning() {
        // Check if base class says we should run (handles stop() calls)
        if (!this->shouldRun()) {
            return false;
        }
        // Also check frame limits
        uint32_t frameNum = this->getFrameNum();
        uint32_t numFrames = this->getNumFrames();
        return (numFrames == 0) || (frameNum < numFrames);
    }
};

//------------------------------------------------------------------------------
int main(int argc, const char **argv)
{
    ProgramArguments args({
        // Rig configuration
        ProgramArguments::Option_t("rigFile", (dw_samples::SamplesDataPath::get() + "/samples/lidar/lidar_object_detection_interprocess_communiation/rig.json").c_str(), "Path to rig configuration file"),
        
        // ICP parameters
        ProgramArguments::Option_t("maxIters", "50", "Number of ICP iterations"),
        ProgramArguments::Option_t("numFrames", "0", "Number of frames to process (0 = unlimited)"),
        ProgramArguments::Option_t("skipFrames", "5", "Number of initial frames to skip"),
        
        // Object detection parameters
        ProgramArguments::Option_t("tensorRTEngine", "", "Path to TensorRT engine file"),
        ProgramArguments::Option_t("objectDetection", "true", "Enable object detection"),
        ProgramArguments::Option_t("minPoints", "120", "Minimum points for valid detection"),
        
        // Ground plane and free space
        ProgramArguments::Option_t("groundPlane", "false", "Enable ground plane extraction"),
        ProgramArguments::Option_t("freeSpace", "true", "Enable free space detection"),
        
        // IPC parameters
        ProgramArguments::Option_t("port", "40002", "Port for socket communication"),
        ProgramArguments::Option_t("ipc", "true", "Enable IPC communication"),
        
        // Debug options
        ProgramArguments::Option_t("verbose", "false", "Enable verbose logging")
    });

    if (!args.parse(argc, argv)) {
        std::cerr << "Failed to parse command line arguments" << std::endl;
        return -1;
    }

    if (args.has("help")) {
        return 0;
    }

    std::cout << "\n=== Detection Producer (Socket Server) ===" << std::endl;
    std::cout << "Port: " << args.get("port") << std::endl;
    std::cout << "==========================================\n" << std::endl;

    DetectionProducer app(args);
    
    if (!app.onInitialize()) {
        std::cerr << "Failed to initialize application" << std::endl;
        app.onRelease();
        return -1;
    }

    app.initializeWindow("Detection Producer", 1, 1, args.enabled("offscreen"));

    // Run processing loop
    while (app.isRunning()) {
        app.onProcess();
        usleep(1000);  // Small sleep to prevent busy-waiting
    }

    app.onRelease();
    
    return 0;
}
