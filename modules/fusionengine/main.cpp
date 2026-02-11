////////////////////////////////////////////////////////////////////////////////
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS"
// NVIDIA MAKES NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR
// OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED
// WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR
// PURPOSE.
//
// FusionEngine - Multi-Sensor Fusion Pipeline Test
// Tests receiving and synchronizing camera and LiDAR frames for fusion.
// Camera server: driveseg_object (ports 49252-49255)
// LiDAR server: lidar_object_detection_interprocess_communication (port 40002)
//
////////////////////////////////////////////////////////////////////////////////

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// DriveWorks Core
#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>

// Sample Framework
#include <framework/ProgramArguments.hpp>

// FusionEngine components
#include "CameraIPCClient.hpp"
#include "LidarIPCClient.hpp"
#include "SensorSynchronizer.hpp"

using namespace fusionengine;

//------------------------------------------------------------------------------
// Global state for signal handling
//------------------------------------------------------------------------------
static std::atomic<bool> g_running{true};

extern "C" void signalHandler(int)
{
    std::cout << "\n[Main] Shutdown signal received" << std::endl;
    g_running = false;
}

//------------------------------------------------------------------------------
// Camera receive thread function
//------------------------------------------------------------------------------
void cameraReceiveThread(CameraIPCClient* client,
                         SensorSynchronizer* synchronizer,
                         std::atomic<uint64_t>* frameCount)
{
    uint32_t camIndex = client->getCameraIndex();
    std::cout << "[Thread] Camera " << camIndex << " receive thread started"
              << std::endl;

    while (g_running)
    {
        if (!client->isConnected())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        CameraFrameData frameData;
        if (client->receive(frameData))
        {
            // Push to synchronizer
            synchronizer->pushCameraData(camIndex, frameData);
            (*frameCount)++;

            // Print periodic updates
            uint64_t count = frameCount->load();
            if (count <= 3 || count % 100 == 0)
            {
                std::cout << "[Cam" << camIndex << "] Frame #" << count
                          << " | ID:" << frameData.frameId
                          << " | Size:" << frameData.width << "x"
                          << frameData.height
                          << " | 2D:" << frameData.boxes2D.size()
                          << " | 3D:" << frameData.boxes3D.size()
                          << std::endl;
            }
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    std::cout << "[Thread] Camera " << camIndex << " receive thread stopped"
              << std::endl;
}

//------------------------------------------------------------------------------
// LiDAR receive thread function
//------------------------------------------------------------------------------
void lidarReceiveThread(LidarIPCClient* client,
                        SensorSynchronizer* synchronizer,
                        std::atomic<uint64_t>* frameCount)
{
    std::cout << "[Thread] LiDAR receive thread started" << std::endl;

    while (g_running)
    {
        if (!client->isConnected())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        LidarFrameData frameData;
        if (client->receive(frameData))
        {
            // Push to synchronizer
            synchronizer->pushLidarData(frameData);
            (*frameCount)++;

            // Print periodic updates
            uint64_t count = frameCount->load();
            if (count <= 3 || count % 100 == 0)
            {
                std::cout << "[LiDAR] Frame #" << count
                          << " | Pts:" << frameData.numPoints
                          << " | Det:" << frameData.detections.size()
                          << " | ICP:" << (frameData.icpAligned ? "Y" : "N")
                          << " | GndPlane:"
                          << (frameData.groundPlane.valid ? "Y" : "N")
                          << std::endl;
            }
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    std::cout << "[Thread] LiDAR receive thread stopped" << std::endl;
}

//------------------------------------------------------------------------------
// Fusion thread function - processes synchronized packets
//------------------------------------------------------------------------------
void fusionThread(SensorSynchronizer* synchronizer,
                  std::atomic<uint64_t>* fusedCount)
{
    std::cout << "[Thread] Fusion thread started" << std::endl;

    while (g_running)
    {
        FusedPacket fusedPacket;
        if (synchronizer->trySync(fusedPacket))
        {
            (*fusedCount)++;

            // Count contributing sensors
            uint32_t camCount = 0;
            for (uint32_t i = 0; i < MAX_CAMERAS; ++i)
            {
                if (fusedPacket.cameraData[i].valid)
                {
                    camCount++;
                }
            }

            // Print periodic updates
            uint64_t count = fusedCount->load();
            if (count <= 5 || count % 50 == 0)
            {
                std::cout << "[Fusion] Packet #" << count
                          << " | LiDAR:" << (fusedPacket.lidarData.valid ? "Y" : "N")
                          << " | Cams:" << camCount
                          << " | AlignErr:" << fusedPacket.temporalAlignmentError
                          << "ms" << std::endl;

                if (fusedPacket.lidarData.valid)
                {
                    std::cout << "         LiDAR det: "
                              << fusedPacket.lidarData.detections.size()
                              << std::endl;
                }
            }
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    std::cout << "[Thread] Fusion thread stopped" << std::endl;
}

//------------------------------------------------------------------------------
// Print statistics
//------------------------------------------------------------------------------
void printStats(uint32_t numCameras,
                const std::vector<std::atomic<uint64_t>*>& camCounts,
                const std::atomic<uint64_t>& lidarCount,
                const std::atomic<uint64_t>& fusedCount,
                const std::vector<std::unique_ptr<CameraIPCClient>>& camClients,
                const LidarIPCClient& lidarClient,
                const SensorSynchronizer& synchronizer)
{
    std::cout << "\n========== Statistics ==========\n";

    // Camera stats
    for (uint32_t i = 0; i < numCameras; ++i)
    {
        std::cout << "Camera " << i << ": "
                  << camCounts[i]->load() << " frames, "
                  << camClients[i]->getReceiveErrors() << " errors, "
                  << "Q:" << synchronizer.getCameraQueueSize(i) << ", "
                  << "Status: " << sensorStatusToString(camClients[i]->getStatus())
                  << std::endl;
    }

    // LiDAR stats
    std::cout << "LiDAR:    "
              << lidarCount.load() << " frames, "
              << lidarClient.getReceiveErrors() << " errors, "
              << "Q:" << synchronizer.getLidarQueueSize() << ", "
              << "Status: " << sensorStatusToString(lidarClient.getStatus())
              << std::endl;

    // Fusion stats
    std::cout << "Fusion:   "
              << fusedCount.load() << " synced packets, "
              << synchronizer.getDroppedLidarFrames() << " dropped LiDAR, "
              << synchronizer.getDroppedCameraFrames() << " dropped cam"
              << std::endl;

    std::cout << "================================\n" << std::endl;
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main(int argc, const char** argv)
{
    // Parse arguments
    ProgramArguments args(argc, argv,
                          {
                              ProgramArguments::Option_t{
                                  "cam-ip", "127.0.0.1", "Camera server IP"},
                              ProgramArguments::Option_t{
                                  "cam-port", "49252", "Camera server base port"},
                              ProgramArguments::Option_t{
                                  "lidar-ip", "127.0.0.1", "LiDAR server IP"},
                              ProgramArguments::Option_t{
                                  "lidar-port", "40002", "LiDAR server port"},
                              ProgramArguments::Option_t{
                                  "num-cameras", "4", "Number of cameras (1-4)"},
                              ProgramArguments::Option_t{
                                  "timeout", "30", "Connection timeout (seconds)"},
                              ProgramArguments::Option_t{
                                  "sync-policy", "latest",
                                  "Sync policy: lidar, camera, nearest, latest"},
                              ProgramArguments::Option_t{
                                  "max-time-diff", "50000",
                                  "Max timestamp diff for sync (microseconds)"},
                          },
                          "FusionEngine - Multi-Sensor Fusion Pipeline Test");

    std::string camIP = args.get("cam-ip");
    uint16_t camBasePort = static_cast<uint16_t>(std::stoul(args.get("cam-port")));
    std::string lidarIP = args.get("lidar-ip");
    uint16_t lidarPort = static_cast<uint16_t>(std::stoul(args.get("lidar-port")));
    uint32_t numCameras = static_cast<uint32_t>(std::stoul(args.get("num-cameras")));
    uint32_t timeoutSec = static_cast<uint32_t>(std::stoul(args.get("timeout")));
    std::string syncPolicyStr = args.get("sync-policy");
    dwTime_t maxTimeDiffUs = std::stoull(args.get("max-time-diff"));

    numCameras = std::min(numCameras, MAX_CAMERAS);

    // Parse sync policy
    SynchronizerConfig::FusionPolicy syncPolicy;
    if (syncPolicyStr == "lidar")
    {
        syncPolicy = SynchronizerConfig::FusionPolicy::LIDAR_MASTER;
    }
    else if (syncPolicyStr == "camera")
    {
        syncPolicy = SynchronizerConfig::FusionPolicy::CAMERA_MASTER;
    }
    else if (syncPolicyStr == "nearest")
    {
        syncPolicy = SynchronizerConfig::FusionPolicy::NEAREST_TIMESTAMP;
    }
    else
    {
        syncPolicy = SynchronizerConfig::FusionPolicy::LATEST_AVAILABLE;
    }

    std::cout << "========================================\n";
    std::cout << " FusionEngine - Multi-Sensor Fusion    \n";
    std::cout << "========================================\n";
    std::cout << "Cameras: " << camIP << ":" << camBasePort << "-"
              << (camBasePort + numCameras - 1) << " (" << numCameras << " cams)\n";
    std::cout << "LiDAR:   " << lidarIP << ":" << lidarPort << "\n";
    std::cout << "Timeout: " << timeoutSec << "s\n";
    std::cout << "Sync:    " << syncPolicyStr << " (max diff: "
              << maxTimeDiffUs << " us)\n";
    std::cout << std::endl;

    // Setup signal handler
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // Initialize DriveWorks context
    std::cout << "[Main] Initializing DriveWorks..." << std::endl;

    dwContextHandle_t context = DW_NULL_HANDLE;
    dwContextParameters contextParams{};

    dwStatus status = dwInitialize(&context, DW_VERSION, &contextParams);
    if (status != DW_SUCCESS)
    {
        std::cerr << "[Main] Failed to initialize DriveWorks: "
                  << dwGetStatusName(status) << std::endl;
        return -1;
    }

    dwVersion version{};
    dwGetVersion(&version);
    std::cout << "[Main] DriveWorks SDK v" << version.major << "."
              << version.minor << "." << version.patch << std::endl;

    // Create sensor synchronizer
    SensorSynchronizer synchronizer;
    SynchronizerConfig syncConfig;
    syncConfig.maxTimeDifferenceUs = maxTimeDiffUs;
    syncConfig.numCameras = numCameras;
    syncConfig.enableLidar = true;
    syncConfig.enableCameras = true;
    syncConfig.policy = syncPolicy;
    syncConfig.verbose = true;
    synchronizer.initialize(syncConfig);

    // Create camera clients
    std::vector<std::unique_ptr<CameraIPCClient>> camClients(numCameras);
    std::vector<std::atomic<uint64_t>> camCounts(numCameras);
    std::vector<std::atomic<uint64_t>*> camCountPtrs(numCameras);

    for (uint32_t i = 0; i < numCameras; ++i)
    {
        camClients[i] = std::make_unique<CameraIPCClient>();
        camCounts[i] = 0;
        camCountPtrs[i] = &camCounts[i];
    }

    // Create LiDAR client
    LidarIPCClient lidarClient;
    std::atomic<uint64_t> lidarCount{0};
    std::atomic<uint64_t> fusedCount{0};

    // Initialize camera clients
    std::cout << "\n[Main] Initializing camera clients..." << std::endl;
    for (uint32_t i = 0; i < numCameras; ++i)
    {
        if (!camClients[i]->initialize(context, i, camIP, camBasePort))
        {
            std::cerr << "[Main] Failed to initialize camera " << i << std::endl;
            dwRelease(context);
            return -1;
        }
    }

    // Initialize LiDAR client
    std::cout << "[Main] Initializing LiDAR client..." << std::endl;
    if (!lidarClient.initialize(context, lidarIP, lidarPort))
    {
        std::cerr << "[Main] Failed to initialize LiDAR client" << std::endl;
        dwRelease(context);
        return -1;
    }

    // Connect camera clients
    std::cout << "\n[Main] Connecting to camera servers..." << std::endl;
    uint32_t connectedCams = 0;
    for (uint32_t i = 0; i < numCameras; ++i)
    {
        if (camClients[i]->connect(timeoutSec, 1000000))
        {
            connectedCams++;
        }
        else
        {
            std::cerr << "[Main] Warning: Camera " << i << " not connected"
                      << std::endl;
        }
    }

    // Connect LiDAR client
    std::cout << "[Main] Connecting to LiDAR server..." << std::endl;
    bool lidarConnected = lidarClient.connect(timeoutSec, 1000000);
    if (!lidarConnected)
    {
        std::cerr << "[Main] Warning: LiDAR not connected" << std::endl;
    }

    if (connectedCams == 0 && !lidarConnected)
    {
        std::cerr << "[Main] Error: No sensors connected" << std::endl;
        dwRelease(context);
        return -1;
    }

    std::cout << "\n[Main] Connected sensors: " << connectedCams << " cameras, "
              << (lidarConnected ? "1 LiDAR" : "0 LiDAR") << std::endl;

    // Start receive threads
    std::cout << "\n[Main] Starting receive threads..." << std::endl;
    std::vector<std::thread> threads;

    // Camera threads
    for (uint32_t i = 0; i < numCameras; ++i)
    {
        threads.emplace_back(cameraReceiveThread,
                             camClients[i].get(),
                             &synchronizer,
                             camCountPtrs[i]);
    }

    // LiDAR thread
    threads.emplace_back(lidarReceiveThread,
                         &lidarClient,
                         &synchronizer,
                         &lidarCount);

    // Fusion thread
    threads.emplace_back(fusionThread, &synchronizer, &fusedCount);

    std::cout << "\n[Main] Running. Press Ctrl+C to stop.\n" << std::endl;

    // Main loop - print periodic statistics
    auto lastStatTime = std::chrono::steady_clock::now();
    while (g_running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           now - lastStatTime)
                           .count();

        if (elapsed >= 5)
        {
            printStats(numCameras, camCountPtrs, lidarCount, fusedCount,
                       camClients, lidarClient, synchronizer);
            lastStatTime = now;
        }
    }

    // Cleanup
    std::cout << "\n[Main] Stopping..." << std::endl;

    // Join threads
    for (auto& t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    // Print final statistics
    printStats(numCameras, camCountPtrs, lidarCount, fusedCount,
               camClients, lidarClient, synchronizer);

    // Release clients
    for (uint32_t i = 0; i < numCameras; ++i)
    {
        camClients[i]->release();
    }
    lidarClient.release();

    // Release DriveWorks
    dwRelease(context);

    std::cout << "[Main] Done." << std::endl;
    return 0;
}
