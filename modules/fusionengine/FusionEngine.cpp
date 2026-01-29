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

#include "FusionEngine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace fusionengine {

//------------------------------------------------------------------------------
// Constructor / Destructor
//------------------------------------------------------------------------------

FusionEngine::FusionEngine()
    : m_lidarClient(std::make_unique<LidarIPCClient>())
    , m_cameraClient(std::make_unique<MultiCameraIPCClient>())
{
}

FusionEngine::~FusionEngine()
{
    release();
}

//------------------------------------------------------------------------------
// Lifecycle
//------------------------------------------------------------------------------

bool FusionEngine::initialize(dwContextHandle_t context,
                               const FusionEngineConfig& config)
{
    if (m_initialized)
    {
        std::cerr << "[FusionEngine] Already initialized" << std::endl;
        return true;
    }

    m_context = context;
    m_config = config;

    std::cout << "[FusionEngine] Initializing..." << std::endl;
    std::cout << "  LiDAR enabled: " << (config.enableLidar ? "yes" : "no")
              << std::endl;
    std::cout << "  Cameras enabled: " << (config.enableCameras ? "yes" : "no")
              << std::endl;
    std::cout << "  Num cameras: " << config.numCameras << std::endl;
    std::cout << "  Async receive: " << (config.asyncReceive ? "yes" : "no")
              << std::endl;

    // Initialize LiDAR client
    if (config.enableLidar)
    {
        if (!m_lidarClient->initialize(context,
                                        config.lidarServerIP,
                                        config.lidarServerPort))
        {
            std::cerr << "[FusionEngine] Failed to initialize LiDAR client"
                      << std::endl;
            return false;
        }
    }

    // Initialize camera clients
    if (config.enableCameras && config.numCameras > 0)
    {
        if (!m_cameraClient->initialize(context,
                                         config.numCameras,
                                         config.cameraServerIP,
                                         config.cameraServerBasePort))
        {
            std::cerr << "[FusionEngine] Failed to initialize camera clients"
                      << std::endl;
            return false;
        }
    }

    // Initialize synchronizer
    m_synchronizer.initialize(config.syncConfig);

    m_initialized = true;
    std::cout << "[FusionEngine] Initialization complete" << std::endl;

    return true;
}

bool FusionEngine::connect()
{
    if (!m_initialized)
    {
        std::cerr << "[FusionEngine] Not initialized" << std::endl;
        return false;
    }

    std::cout << "[FusionEngine] Connecting to servers..." << std::endl;

    bool allConnected = true;

    // Connect to LiDAR server
    if (m_config.enableLidar)
    {
        if (!m_lidarClient->connect(m_config.connectionRetries,
                                     m_config.connectionTimeoutUs))
        {
            std::cerr << "[FusionEngine] Failed to connect to LiDAR server"
                      << std::endl;
            allConnected = false;
        }
    }

    // Connect to camera servers
    if (m_config.enableCameras && m_config.numCameras > 0)
    {
        if (!m_cameraClient->connectAll(m_config.connectionRetries,
                                         m_config.connectionTimeoutUs))
        {
            std::cerr << "[FusionEngine] Failed to connect to all camera servers"
                      << std::endl;
            allConnected = false;
        }
    }

    if (allConnected)
    {
        std::cout << "[FusionEngine] All connections established" << std::endl;
    }

    return allConnected;
}

void FusionEngine::start()
{
    if (!m_initialized)
    {
        std::cerr << "[FusionEngine] Not initialized" << std::endl;
        return;
    }

    if (m_running)
    {
        std::cout << "[FusionEngine] Already running" << std::endl;
        return;
    }

    m_running = true;

    std::cout << "[FusionEngine] Starting..." << std::endl;

    if (m_config.asyncReceive)
    {
        // Launch LiDAR receive thread
        if (m_config.enableLidar)
        {
            m_lidarThread = std::make_unique<std::thread>(
                &FusionEngine::lidarReceiveThread, this);
        }

        // Launch camera receive threads (only if cameras are enabled)
        if (m_config.enableCameras)
        {
            for (uint32_t i = 0; i < m_config.numCameras; ++i)
            {
                m_cameraThreads[i] = std::make_unique<std::thread>(
                    &FusionEngine::cameraReceiveThread, this, i);
            }

            // Launch fusion thread only when camera data is part of fusion
            m_fusionThread = std::make_unique<std::thread>(
                &FusionEngine::fusionThread, this);
        }
    }

    std::cout << "[FusionEngine] Started" << std::endl;
}

void FusionEngine::stop()
{
    if (!m_running)
    {
        return;
    }

    std::cout << "[FusionEngine] Stopping..." << std::endl;

    m_running = false;

    // Join all threads
    if (m_lidarThread && m_lidarThread->joinable())
    {
        m_lidarThread->join();
    }

    for (auto& thread : m_cameraThreads)
    {
        if (thread && thread->joinable())
        {
            thread->join();
        }
    }

    if (m_fusionThread && m_fusionThread->joinable())
    {
        m_fusionThread->join();
    }

    std::cout << "[FusionEngine] Stopped" << std::endl;
}

void FusionEngine::release()
{
    stop();

    if (m_lidarClient)
    {
        m_lidarClient->release();
    }

    if (m_cameraClient)
    {
        m_cameraClient->release();
    }

    m_initialized = false;
}

//------------------------------------------------------------------------------
// Processing
//------------------------------------------------------------------------------

void FusionEngine::process()
{
    if (!m_initialized || !m_running)
    {
        return;
    }

    // In synchronous mode, receive from all sources
    if (!m_config.asyncReceive)
    {
        // Receive LiDAR data
        if (m_config.enableLidar && m_lidarClient->isConnected())
        {
            LidarFrameData lidarData;
            if (m_lidarClient->receive(lidarData))
            {
                m_synchronizer.pushLidarData(lidarData);
            }
        }

        // Receive camera data
        if (m_config.enableCameras)
        {
            for (uint32_t i = 0; i < m_config.numCameras; ++i)
            {
                if (m_cameraClient->isConnected(i))
                {
                    CameraFrameData cameraData;
                    if (m_cameraClient->receive(i, cameraData))
                    {
                        m_synchronizer.pushCameraData(i, cameraData);
                    }
                }
            }
        }

        // Try to produce fused output
        FusedPacket fusedPacket;
        if (m_synchronizer.trySync(fusedPacket))
        {
            performFusion(fusedPacket);

            // Store output
            {
                std::lock_guard<std::mutex> lock(m_outputMutex);
                m_latestFusedPacket = fusedPacket;
                m_hasFusedPacket = true;
            }

            m_fusedPacketsProduced++;

            // Invoke callback
            if (m_fusionCallback)
            {
                m_fusionCallback(fusedPacket);
            }
        }
    }
}

bool FusionEngine::tryGetFusedPacket(FusedPacket& fusedPacket)
{
    std::lock_guard<std::mutex> lock(m_outputMutex);

    if (!m_hasFusedPacket)
    {
        return false;
    }

    fusedPacket = m_latestFusedPacket;
    m_hasFusedPacket = false;

    return true;
}

void FusionEngine::setFusionCallback(FusionCallback callback)
{
    m_fusionCallback = callback;
}

//------------------------------------------------------------------------------
// Status
//------------------------------------------------------------------------------

bool FusionEngine::isConnected() const
{
    bool connected = true;

    if (m_config.enableLidar)
    {
        connected = connected && m_lidarClient->isConnected();
    }

    if (m_config.enableCameras)
    {
        for (uint32_t i = 0; i < m_config.numCameras; ++i)
        {
            connected = connected && m_cameraClient->isConnected(i);
        }
    }

    return connected;
}

SensorStatus FusionEngine::getLidarStatus() const
{
    return m_lidarClient->getStatus();
}

SensorStatus FusionEngine::getCameraStatus(uint32_t cameraIndex) const
{
    if (cameraIndex >= m_config.numCameras)
    {
        return SensorStatus::DISCONNECTED;
    }
    return m_cameraClient->getClient(cameraIndex).getStatus();
}

uint64_t FusionEngine::getLidarPacketsReceived() const
{
    return m_lidarClient->getPacketsReceived();
}

uint64_t FusionEngine::getCameraFramesReceived(uint32_t cameraIndex) const
{
    if (cameraIndex >= m_config.numCameras)
    {
        return 0;
    }
    return m_cameraClient->getClient(cameraIndex).getFramesReceived();
}

//------------------------------------------------------------------------------
// Thread functions
//------------------------------------------------------------------------------

void FusionEngine::lidarReceiveThread()
{
    std::cout << "[FusionEngine] LiDAR receive thread started" << std::endl;

    uint64_t localFrameCount = 0;

    while (m_running)
    {
        if (!m_lidarClient->isConnected())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        LidarFrameData lidarData;
        if (m_lidarClient->receive(lidarData))
        {
            ++localFrameCount;
            if (localFrameCount <= 5)
            {
                std::cout << "[FusionEngine] Received LiDAR frame "
                          << localFrameCount
                          << " ts=" << lidarData.timestamp
                          << " numPoints=" << lidarData.numPoints
                          << " numDetections=" << lidarData.detections.size()
                          << std::endl;
            }

            // If cameras are enabled, go through the normal synchronizer + fusion path.
            // If cameras are disabled, produce a simple LiDAR-only fused packet here.
            if (m_config.enableCameras)
            {
                m_synchronizer.pushLidarData(lidarData);
            }
            else
            {
                FusedPacket fused{};

                fused.lidarData       = lidarData;
                fused.lidarTimestamp  = lidarData.timestamp;
                fused.lidarFrameNumber = lidarData.frameNumber;

                // No cameras, so leave camera-related fields at defaults.
                fused.numLidarOnlyDetections   = static_cast<uint32_t>(lidarData.detections.size());
                fused.numCameraOnlyDetections  = 0;
                fused.numFusedDetections       = 0;

                // Basic timing metadata
                auto now = std::chrono::high_resolution_clock::now();
                auto us  = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
                fused.fusionTimestamp = static_cast<dwTime_t>(us);
                fused.fusionProcessingMs = 0.0f;
                fused.totalProcessingMs = 0.0f;
                fused.temporalAlignmentError = 0.0f;
                fused.valid = true;

                // Update statistics counter and assign a fusion frame number
                m_fusedPacketsProduced++;
                fused.fusionFrameNumber =
                    static_cast<uint32_t>(m_fusedPacketsProduced.load());

                // Store latest packet
                {
                    std::lock_guard<std::mutex> lock(m_outputMutex);
                    m_latestFusedPacket = fused;
                    m_hasFusedPacket    = true;
                }

                // Invoke callback for visualization
                if (m_fusionCallback)
                {
                    if (localFrameCount <= 5)
                    {
                        std::cout << "[FusionEngine] Invoking fusion callback for frame "
                                  << fused.lidarFrameNumber
                                  << " (points=" << fused.lidarData.numPoints
                                  << ", detections=" << fused.lidarData.detections.size()
                                  << ")" << std::endl;
                    }
                    m_fusionCallback(fused);
                }
            }
        }
        else
        {
            // Small sleep to prevent busy-waiting on timeout
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    std::cout << "[FusionEngine] LiDAR receive thread stopped" << std::endl;
}

void FusionEngine::cameraReceiveThread(uint32_t cameraIndex)
{
    std::cout << "[FusionEngine] Camera " << cameraIndex
              << " receive thread started" << std::endl;

    while (m_running)
    {
        if (!m_cameraClient->isConnected(cameraIndex))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        CameraFrameData cameraData;
        if (m_cameraClient->receive(cameraIndex, cameraData))
        {
            m_synchronizer.pushCameraData(cameraIndex, cameraData);
        }
        else
        {
            // Small sleep to prevent busy-waiting on timeout
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    std::cout << "[FusionEngine] Camera " << cameraIndex
              << " receive thread stopped" << std::endl;
}

void FusionEngine::fusionThread()
{
    std::cout << "[FusionEngine] Fusion thread started" << std::endl;

    while (m_running)
    {
        FusedPacket fusedPacket;
        if (m_synchronizer.trySync(fusedPacket))
        {
            performFusion(fusedPacket);

            // Store output
            {
                std::lock_guard<std::mutex> lock(m_outputMutex);
                m_latestFusedPacket = fusedPacket;
                m_hasFusedPacket = true;
            }

            m_fusedPacketsProduced++;

            // Invoke callback
            if (m_fusionCallback)
            {
                m_fusionCallback(fusedPacket);
            }
        }
        else
        {
            // No synchronized data available, sleep briefly
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    std::cout << "[FusionEngine] Fusion thread stopped" << std::endl;
}

//------------------------------------------------------------------------------
// Fusion algorithms
//------------------------------------------------------------------------------

void FusionEngine::performFusion(FusedPacket& fusedPacket)
{
    auto startTime = std::chrono::high_resolution_clock::now();

    // Fuse detections from LiDAR and cameras
    fuseDetections(fusedPacket);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        endTime - startTime);

    fusedPacket.fusionProcessingMs = duration.count() / 1000.0f;
}

void FusionEngine::fuseDetections(FusedPacket& fusedPacket)
{
    fusedPacket.fusedDetections.clear();

    // Track which detections have been fused
    std::vector<bool> lidarMatched(fusedPacket.lidarData.detections.size(),
                                    false);
    std::vector<std::vector<bool>> cameraMatched(MAX_CAMERAS);

    for (uint32_t i = 0; i < m_config.numCameras; ++i)
    {
        cameraMatched[i].resize(fusedPacket.cameraData[i].boxes3D.size(), false);
    }

    // Step 1: Match LiDAR detections with camera 3D detections
    for (size_t lidarIdx = 0;
         lidarIdx < fusedPacket.lidarData.detections.size();
         ++lidarIdx)
    {
        const auto& lidarBox = fusedPacket.lidarData.detections[lidarIdx];

        FusedDetection fusedDet{};
        fusedDet.x = lidarBox.x;
        fusedDet.y = lidarBox.y;
        fusedDet.z = lidarBox.z;
        fusedDet.width = lidarBox.width;
        fusedDet.length = lidarBox.length;
        fusedDet.height = lidarBox.height;
        fusedDet.rotation = lidarBox.rotation;
        fusedDet.classId = lidarBox.classId;
        fusedDet.lidarConfidence = lidarBox.confidence;
        fusedDet.hasLidarSource = true;
        fusedDet.cameraSourceMask = 0;

        // Try to match with camera detections
        float bestIoU = 0.0f;
        int32_t bestCameraIdx = -1;
        int32_t bestBoxIdx = -1;

        for (uint32_t camIdx = 0; camIdx < m_config.numCameras; ++camIdx)
        {
            if (!fusedPacket.cameraData[camIdx].valid)
            {
                continue;
            }

            for (size_t boxIdx = 0;
                 boxIdx < fusedPacket.cameraData[camIdx].boxes3D.size();
                 ++boxIdx)
            {
                if (cameraMatched[camIdx][boxIdx])
                {
                    continue;
                }

                const auto& cameraBox =
                    fusedPacket.cameraData[camIdx].boxes3D[boxIdx];
                float iou = computeIoU3D(lidarBox, cameraBox);

                if (iou > bestIoU && iou > m_config.iouThreshold)
                {
                    bestIoU = iou;
                    bestCameraIdx = static_cast<int32_t>(camIdx);
                    bestBoxIdx = static_cast<int32_t>(boxIdx);
                }
            }
        }

        // Fuse with best matching camera detection
        if (bestCameraIdx >= 0 && bestBoxIdx >= 0)
        {
            const auto& camBox =
                fusedPacket.cameraData[bestCameraIdx].boxes3D[bestBoxIdx];

            // Mark as matched
            lidarMatched[lidarIdx] = true;
            cameraMatched[bestCameraIdx][bestBoxIdx] = true;

            // Update fused detection with camera information
            fusedDet.cameraConfidence = camBox.iouScore;
            fusedDet.hasCameraSource = true;
            fusedDet.cameraSourceMask |= (1 << bestCameraIdx);

            // Fused confidence (simple weighted average)
            fusedDet.fusedConfidence =
                0.6f * fusedDet.lidarConfidence +
                0.4f * fusedDet.cameraConfidence;

            // Copy label from camera detection if available
            if (bestBoxIdx < static_cast<int32_t>(
                    fusedPacket.cameraData[bestCameraIdx].boxes2D.size()))
            {
                std::strncpy(
                    fusedDet.label,
                    fusedPacket.cameraData[bestCameraIdx]
                        .boxes2D[bestBoxIdx].label,
                    sizeof(fusedDet.label) - 1);
            }

            fusedPacket.numFusedDetections++;
        }
        else
        {
            // LiDAR-only detection
            fusedDet.fusedConfidence = fusedDet.lidarConfidence;
            fusedDet.hasCameraSource = false;
            fusedPacket.numLidarOnlyDetections++;

            // Set label based on class ID
            switch (fusedDet.classId)
            {
            case 0:
                std::strncpy(fusedDet.label, "Vehicle",
                             sizeof(fusedDet.label) - 1);
                break;
            case 1:
                std::strncpy(fusedDet.label, "Pedestrian",
                             sizeof(fusedDet.label) - 1);
                break;
            case 2:
                std::strncpy(fusedDet.label, "Cyclist",
                             sizeof(fusedDet.label) - 1);
                break;
            default:
                std::strncpy(fusedDet.label, "Unknown",
                             sizeof(fusedDet.label) - 1);
                break;
            }
        }

        fusedDet.trackId = 0;  // TODO: Implement tracking

        fusedPacket.fusedDetections.push_back(fusedDet);
    }

    // Step 2: Add unmatched camera detections as camera-only
    for (uint32_t camIdx = 0; camIdx < m_config.numCameras; ++camIdx)
    {
        if (!fusedPacket.cameraData[camIdx].valid)
        {
            continue;
        }

        for (size_t boxIdx = 0;
             boxIdx < fusedPacket.cameraData[camIdx].boxes3D.size();
             ++boxIdx)
        {
            if (cameraMatched[camIdx][boxIdx])
            {
                continue;
            }

            const auto& camBox = fusedPacket.cameraData[camIdx].boxes3D[boxIdx];

            // Project camera box to LiDAR frame
            LidarBoundingBox projectedBox;
            projectCameraBoxToLidarFrame(camBox, camIdx, projectedBox);

            FusedDetection fusedDet{};
            fusedDet.x = projectedBox.x;
            fusedDet.y = projectedBox.y;
            fusedDet.z = projectedBox.z;
            fusedDet.width = projectedBox.width;
            fusedDet.length = projectedBox.length;
            fusedDet.height = projectedBox.height;
            fusedDet.rotation = camBox.rotation;
            fusedDet.cameraConfidence = camBox.iouScore;
            fusedDet.fusedConfidence = camBox.iouScore;
            fusedDet.hasLidarSource = false;
            fusedDet.hasCameraSource = true;
            fusedDet.cameraSourceMask = (1 << camIdx);
            fusedDet.trackId = 0;

            // Copy label from 2D detection if available
            if (boxIdx < fusedPacket.cameraData[camIdx].boxes2D.size())
            {
                std::strncpy(
                    fusedDet.label,
                    fusedPacket.cameraData[camIdx].boxes2D[boxIdx].label,
                    sizeof(fusedDet.label) - 1);
            }

            fusedPacket.fusedDetections.push_back(fusedDet);
            fusedPacket.numCameraOnlyDetections++;
        }
    }
}

float FusionEngine::computeIoU3D(const LidarBoundingBox& lidarBox,
                                  const Camera3DBox& cameraBox)
{
    // Simplified 3D IoU calculation
    // In a full implementation, this would compute proper 3D box overlap
    // considering rotations

    // For now, use a simple approximation based on center distance and sizes
    // This assumes boxes are axis-aligned for simplicity

    // Project camera box position (relative depth) to approximate world coords
    // This is a placeholder - real implementation needs camera calibration
    float camX = 0.0f;  // Would be computed from depth + camera position
    float camY = cameraBox.depth;  // Approximate forward distance
    float camZ = 0.0f;

    // Compute center distance
    float dx = lidarBox.x - camX;
    float dy = lidarBox.y - camY;
    float dz = lidarBox.z - camZ;
    float centerDist = std::sqrt(dx * dx + dy * dy + dz * dz);

    // Average box size
    float avgSize = (lidarBox.width + lidarBox.length + lidarBox.height +
                     cameraBox.width + cameraBox.length + cameraBox.height) /
                    6.0f;

    // Simple IoU approximation based on distance
    if (centerDist > avgSize * 2.0f)
    {
        return 0.0f;  // Too far apart
    }

    // Linear falloff based on distance
    float iou = std::max(0.0f, 1.0f - centerDist / (avgSize * 2.0f));

    return iou;
}

void FusionEngine::projectCameraBoxToLidarFrame(const Camera3DBox& cameraBox,
                                                 uint32_t cameraIndex,
                                                 LidarBoundingBox& projectedBox)
{
    // Placeholder projection - real implementation needs camera extrinsics
    // This assumes a simplified camera configuration

    // Approximate camera positions relative to LiDAR (center)
    // Camera 0: Front, Camera 1: Left, Camera 2: Rear, Camera 3: Right
    float cameraOffsets[4][2] = {
        {0.0f, 2.0f},   // Front: +Y
        {-2.0f, 0.0f},  // Left: -X
        {0.0f, -2.0f},  // Rear: -Y
        {2.0f, 0.0f}    // Right: +X
    };

    float camAngle[4] = {0.0f, M_PI / 2.0f, M_PI, -M_PI / 2.0f};

    if (cameraIndex >= 4)
    {
        cameraIndex = 0;
    }

    // Project camera-relative position to world frame
    float depth = cameraBox.depth;
    float angle = camAngle[cameraIndex];

    projectedBox.x = cameraOffsets[cameraIndex][0] + depth * std::sin(angle);
    projectedBox.y = cameraOffsets[cameraIndex][1] + depth * std::cos(angle);
    projectedBox.z = 0.0f;  // Ground level assumption

    projectedBox.width = cameraBox.width;
    projectedBox.length = cameraBox.length;
    projectedBox.height = cameraBox.height;
    projectedBox.rotation = cameraBox.rotation + angle;
    projectedBox.confidence = cameraBox.iouScore;
    projectedBox.classId = 0;  // Unknown from camera
}

} // namespace fusionengine
