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

#ifndef FUSIONENGINE_FUSIONENGINE_HPP
#define FUSIONENGINE_FUSIONENGINE_HPP

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// DriveWorks headers
#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>

// Local headers
#include "FusedPacket.hpp"
#include "LidarIPCClient.hpp"
#include "CameraIPCClient.hpp"
#include "SensorSynchronizer.hpp"

namespace fusionengine {

//------------------------------------------------------------------------------
// Fusion Engine Configuration
//------------------------------------------------------------------------------
struct FusionEngineConfig
{
    // LiDAR IPC settings
    bool enableLidar{true};
    std::string lidarServerIP{"127.0.0.1"};
    uint16_t lidarServerPort{40002};

    // Camera IPC settings
    bool enableCameras{true};
    uint32_t numCameras{4};
    std::string cameraServerIP{"127.0.0.1"};
    uint16_t cameraServerBasePort{49252};

    // Connection settings
    uint32_t connectionRetries{30};
    uint32_t connectionTimeoutUs{1000000};

    // Synchronization settings
    SynchronizerConfig syncConfig;

    // Fusion settings
    float iouThreshold{0.3f};           // IoU threshold for detection matching
    float confidenceThreshold{0.5f};    // Minimum confidence for fusion
    bool projectCameraToLidar{true};    // Project camera 3D boxes to LiDAR frame

    // Threading
    bool asyncReceive{true};            // Use separate threads for receiving
};

//------------------------------------------------------------------------------
// Fusion callback type
//------------------------------------------------------------------------------
using FusionCallback = std::function<void(const FusedPacket&)>;

//------------------------------------------------------------------------------
// Fusion Engine Class
//------------------------------------------------------------------------------
class FusionEngine
{
public:
    FusionEngine();
    ~FusionEngine();

    // Non-copyable
    FusionEngine(const FusionEngine&) = delete;
    FusionEngine& operator=(const FusionEngine&) = delete;

    //--------------------------------------------------------------------------
    // Lifecycle
    //--------------------------------------------------------------------------

    /**
     * Initialize the fusion engine with given configuration
     * @param context DriveWorks context handle
     * @param config Configuration parameters
     * @return true if successful
     */
    bool initialize(dwContextHandle_t context, const FusionEngineConfig& config);

    /**
     * Connect to all configured sensor servers
     * @return true if all connections successful
     */
    bool connect();

    /**
     * Start processing (launches receive threads if async mode)
     */
    void start();

    /**
     * Stop processing
     */
    void stop();

    /**
     * Release all resources
     */
    void release();

    //--------------------------------------------------------------------------
    // Processing
    //--------------------------------------------------------------------------

    /**
     * Process one cycle (synchronous mode)
     * Call this in your main loop if asyncReceive is false
     */
    void process();

    /**
     * Try to get a synchronized/fused packet
     * @param fusedPacket Output fused packet
     * @return true if a fused packet is available
     */
    bool tryGetFusedPacket(FusedPacket& fusedPacket);

    /**
     * Register callback for fused packets
     * @param callback Function to call when fused data is available
     */
    void setFusionCallback(FusionCallback callback);

    //--------------------------------------------------------------------------
    // Status
    //--------------------------------------------------------------------------

    bool isInitialized() const { return m_initialized; }
    bool isConnected() const;
    bool isRunning() const { return m_running; }

    // Sensor status
    SensorStatus getLidarStatus() const;
    SensorStatus getCameraStatus(uint32_t cameraIndex) const;

    // Statistics
    uint64_t getLidarPacketsReceived() const;
    uint64_t getCameraFramesReceived(uint32_t cameraIndex) const;
    uint64_t getFusedPacketsProduced() const { return m_fusedPacketsProduced; }

    // Get configuration
    const FusionEngineConfig& getConfig() const { return m_config; }

private:
    //--------------------------------------------------------------------------
    // Internal methods
    //--------------------------------------------------------------------------

    // Receive thread functions
    void lidarReceiveThread();
    void cameraReceiveThread(uint32_t cameraIndex);

    // Fusion processing
    void fusionThread();
    void performFusion(FusedPacket& fusedPacket);

    // Detection fusion algorithms
    void fuseDetections(FusedPacket& fusedPacket);
    float computeIoU3D(const LidarBoundingBox& lidarBox,
                       const Camera3DBox& cameraBox);
    void projectCameraBoxToLidarFrame(const Camera3DBox& cameraBox,
                                       uint32_t cameraIndex,
                                       LidarBoundingBox& projectedBox);

private:
    // Configuration
    FusionEngineConfig m_config;

    // DriveWorks context
    dwContextHandle_t m_context{DW_NULL_HANDLE};

    // IPC clients
    std::unique_ptr<LidarIPCClient> m_lidarClient;
    std::unique_ptr<MultiCameraIPCClient> m_cameraClient;

    // Sensor synchronizer
    SensorSynchronizer m_synchronizer;

    // State
    std::atomic<bool> m_initialized{false};
    std::atomic<bool> m_running{false};

    // Threads
    std::unique_ptr<std::thread> m_lidarThread;
    std::array<std::unique_ptr<std::thread>, MAX_CAMERAS> m_cameraThreads;
    std::unique_ptr<std::thread> m_fusionThread;

    // Output
    std::mutex m_outputMutex;
    FusedPacket m_latestFusedPacket;
    bool m_hasFusedPacket{false};

    // Callback
    FusionCallback m_fusionCallback;

    // Statistics
    std::atomic<uint64_t> m_fusedPacketsProduced{0};
};

} // namespace fusionengine

#endif // FUSIONENGINE_FUSIONENGINE_HPP
