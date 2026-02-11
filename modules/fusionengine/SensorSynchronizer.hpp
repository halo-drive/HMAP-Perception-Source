////////////////////////////////////////////////////////////////////////////////
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS"
// NVIDIA MAKES NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR
// OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED
// WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR
// PURPOSE.
//
// SensorSynchronizer.hpp - Multi-sensor temporal synchronization
// Buffers and synchronizes LiDAR and camera data based on timestamps to
// produce temporally-aligned FusedPackets for sensor fusion.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef FUSIONENGINE_SENSORSYNCHRONIZER_HPP
#define FUSIONENGINE_SENSORSYNCHRONIZER_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iostream>
#include <mutex>

#include <dw/core/base/Types.h>

#include "FusedPacket.hpp"

namespace fusionengine {

//------------------------------------------------------------------------------
// Timestamped data wrapper for queue management
//------------------------------------------------------------------------------
template <typename T>
struct TimestampedData
{
    dwTime_t timestamp{0};
    T data;
    bool valid{false};
};

//------------------------------------------------------------------------------
// Sensor Synchronizer Configuration
//------------------------------------------------------------------------------
struct SynchronizerConfig
{
    // Maximum time difference (microseconds) between LiDAR and camera timestamps
    // for them to be considered synchronized
    dwTime_t maxTimeDifferenceUs{50000};  // 50ms default tolerance

    // Maximum queue depth for each sensor (prevents unbounded memory growth)
    uint32_t maxQueueDepth{10};

    // Enable/disable specific sensors
    bool enableLidar{true};
    bool enableCameras{true};
    uint32_t numCameras{4};

    // Fusion policy determines how frames are matched
    enum class FusionPolicy
    {
        LIDAR_MASTER,       // Use LiDAR timestamp as reference, find nearest cameras
        CAMERA_MASTER,      // Use oldest camera timestamp as reference
        NEAREST_TIMESTAMP,  // Find globally nearest matching timestamps
        LATEST_AVAILABLE    // Use latest available from each sensor (no strict sync)
    };
    FusionPolicy policy{FusionPolicy::LATEST_AVAILABLE};

    // Debug output
    bool verbose{false};
};

//------------------------------------------------------------------------------
// Sensor Synchronizer Class
// Thread-safe synchronization of multi-sensor data streams
//------------------------------------------------------------------------------
class SensorSynchronizer
{
public:
    SensorSynchronizer() = default;
    ~SensorSynchronizer() = default;

    // Non-copyable (contains mutexes)
    SensorSynchronizer(const SensorSynchronizer&) = delete;
    SensorSynchronizer& operator=(const SensorSynchronizer&) = delete;

    //--------------------------------------------------------------------------
    // Initialize with configuration
    //--------------------------------------------------------------------------
    void initialize(const SynchronizerConfig& config)
    {
        m_config = config;

        // Clear all queues
        {
            std::lock_guard<std::mutex> lock(m_lidarMutex);
            m_lidarQueue.clear();
        }

        for (uint32_t i = 0; i < MAX_CAMERAS; ++i)
        {
            std::lock_guard<std::mutex> lock(m_cameraMutex[i]);
            m_cameraQueues[i].clear();
        }

        m_fusionFrameNumber = 0;
        m_syncedPackets = 0;
        m_droppedLidarFrames = 0;
        m_droppedCameraFrames = 0;

        if (m_config.verbose)
        {
            std::cout << "[SensorSynchronizer] Initialized with policy: "
                      << policyToString(m_config.policy) << std::endl;
            std::cout << "  Max time diff: " << m_config.maxTimeDifferenceUs
                      << " us" << std::endl;
            std::cout << "  LiDAR enabled: " << (m_config.enableLidar ? "yes" : "no")
                      << std::endl;
            std::cout << "  Cameras enabled: " << (m_config.enableCameras ? "yes" : "no")
                      << ", count: " << m_config.numCameras << std::endl;
        }
    }

    //--------------------------------------------------------------------------
    // Push LiDAR data to synchronization queue
    //--------------------------------------------------------------------------
    void pushLidarData(const LidarFrameData& data)
    {
        if (!m_config.enableLidar)
        {
            return;
        }

        std::lock_guard<std::mutex> lock(m_lidarMutex);

        TimestampedData<LidarFrameData> entry;
        entry.timestamp = data.timestamp;
        entry.data = data;
        entry.valid = data.valid;

        m_lidarQueue.push_back(std::move(entry));

        // Maintain queue depth
        while (m_lidarQueue.size() > m_config.maxQueueDepth)
        {
            m_lidarQueue.pop_front();
            m_droppedLidarFrames++;
        }
    }

    //--------------------------------------------------------------------------
    // Push camera data to synchronization queue
    //--------------------------------------------------------------------------
    void pushCameraData(uint32_t cameraIndex, const CameraFrameData& data)
    {
        if (!m_config.enableCameras || cameraIndex >= m_config.numCameras)
        {
            return;
        }

        std::lock_guard<std::mutex> lock(m_cameraMutex[cameraIndex]);

        TimestampedData<CameraFrameData> entry;
        entry.timestamp = data.timestamp;
        entry.data = data;
        entry.valid = data.valid;

        m_cameraQueues[cameraIndex].push_back(std::move(entry));

        // Maintain queue depth
        while (m_cameraQueues[cameraIndex].size() > m_config.maxQueueDepth)
        {
            m_cameraQueues[cameraIndex].pop_front();
            m_droppedCameraFrames++;
        }
    }

    //--------------------------------------------------------------------------
    // Try to synchronize and produce a fused packet
    // Returns true if a synchronized packet was produced
    //--------------------------------------------------------------------------
    bool trySync(FusedPacket& fusedPacket)
    {
        bool result = false;

        switch (m_config.policy)
        {
        case SynchronizerConfig::FusionPolicy::LIDAR_MASTER:
            result = trySyncLidarMaster(fusedPacket);
            break;

        case SynchronizerConfig::FusionPolicy::CAMERA_MASTER:
            result = trySyncCameraMaster(fusedPacket);
            break;

        case SynchronizerConfig::FusionPolicy::NEAREST_TIMESTAMP:
            result = trySyncNearestTimestamp(fusedPacket);
            break;

        case SynchronizerConfig::FusionPolicy::LATEST_AVAILABLE:
        default:
            result = trySyncLatestAvailable(fusedPacket);
            break;
        }

        if (result)
        {
            m_syncedPackets++;
        }

        return result;
    }

    //--------------------------------------------------------------------------
    // Statistics accessors
    //--------------------------------------------------------------------------
    uint32_t getLidarQueueSize() const
    {
        std::lock_guard<std::mutex> lock(m_lidarMutex);
        return static_cast<uint32_t>(m_lidarQueue.size());
    }

    uint32_t getCameraQueueSize(uint32_t cameraIndex) const
    {
        if (cameraIndex >= MAX_CAMERAS)
        {
            return 0;
        }
        std::lock_guard<std::mutex> lock(m_cameraMutex[cameraIndex]);
        return static_cast<uint32_t>(m_cameraQueues[cameraIndex].size());
    }

    uint64_t getSyncedPacketCount() const { return m_syncedPackets.load(); }
    uint64_t getDroppedLidarFrames() const { return m_droppedLidarFrames.load(); }
    uint64_t getDroppedCameraFrames() const { return m_droppedCameraFrames.load(); }

    //--------------------------------------------------------------------------
    // Configuration accessor
    //--------------------------------------------------------------------------
    const SynchronizerConfig& getConfig() const { return m_config; }

private:
    //--------------------------------------------------------------------------
    // LIDAR_MASTER: Use LiDAR timestamp as reference, find nearest camera frames
    //--------------------------------------------------------------------------
    bool trySyncLidarMaster(FusedPacket& fusedPacket)
    {
        // Get oldest LiDAR frame
        TimestampedData<LidarFrameData> lidarEntry;
        {
            std::lock_guard<std::mutex> lock(m_lidarMutex);
            if (m_lidarQueue.empty())
            {
                return false;
            }
            lidarEntry = m_lidarQueue.front();
            m_lidarQueue.pop_front();
        }

        if (!lidarEntry.valid)
        {
            return false;
        }

        // Initialize fused packet with LiDAR data
        fusedPacket.clear();
        fusedPacket.lidarTimestamp = lidarEntry.timestamp;
        fusedPacket.lidarData = std::move(lidarEntry.data);
        fusedPacket.lidarFrameNumber = fusedPacket.lidarData.frameNumber;

        // Find matching camera frames within tolerance
        dwTime_t maxTimeDiff = 0;
        for (uint32_t i = 0; i < m_config.numCameras; ++i)
        {
            auto cameraEntry = findAndRemoveNearestCameraFrame(i, lidarEntry.timestamp);
            if (cameraEntry.valid)
            {
                fusedPacket.cameraData[i] = std::move(cameraEntry.data);
                fusedPacket.cameraTimestamps[i] = cameraEntry.timestamp;
                fusedPacket.cameraFrameIds[i] = fusedPacket.cameraData[i].frameId;

                dwTime_t timeDiff = static_cast<dwTime_t>(std::abs(
                    static_cast<int64_t>(cameraEntry.timestamp) -
                    static_cast<int64_t>(lidarEntry.timestamp)));
                maxTimeDiff = std::max(maxTimeDiff, timeDiff);
            }
        }

        // Finalize packet
        finalizePacket(fusedPacket, maxTimeDiff);
        return true;
    }

    //--------------------------------------------------------------------------
    // CAMERA_MASTER: Use oldest camera timestamp as reference
    //--------------------------------------------------------------------------
    bool trySyncCameraMaster(FusedPacket& fusedPacket)
    {
        // Find the camera with the oldest frame (as reference)
        dwTime_t oldestTimestamp = UINT64_MAX;
        uint32_t refCamera = 0;
        bool foundCamera = false;

        for (uint32_t i = 0; i < m_config.numCameras; ++i)
        {
            std::lock_guard<std::mutex> lock(m_cameraMutex[i]);
            if (!m_cameraQueues[i].empty() &&
                m_cameraQueues[i].front().valid &&
                m_cameraQueues[i].front().timestamp < oldestTimestamp)
            {
                oldestTimestamp = m_cameraQueues[i].front().timestamp;
                refCamera = i;
                foundCamera = true;
            }
        }

        if (!foundCamera)
        {
            return false;
        }

        // Get reference camera frame
        TimestampedData<CameraFrameData> refEntry;
        {
            std::lock_guard<std::mutex> lock(m_cameraMutex[refCamera]);
            if (m_cameraQueues[refCamera].empty())
            {
                return false;
            }
            refEntry = m_cameraQueues[refCamera].front();
            m_cameraQueues[refCamera].pop_front();
        }

        // Initialize fused packet
        fusedPacket.clear();
        fusedPacket.cameraData[refCamera] = std::move(refEntry.data);
        fusedPacket.cameraTimestamps[refCamera] = refEntry.timestamp;
        fusedPacket.cameraFrameIds[refCamera] = fusedPacket.cameraData[refCamera].frameId;

        // Find matching LiDAR frame
        dwTime_t maxTimeDiff = 0;
        if (m_config.enableLidar)
        {
            auto lidarEntry = findAndRemoveNearestLidarFrame(refEntry.timestamp);
            if (lidarEntry.valid)
            {
                fusedPacket.lidarData = std::move(lidarEntry.data);
                fusedPacket.lidarTimestamp = lidarEntry.timestamp;
                fusedPacket.lidarFrameNumber = fusedPacket.lidarData.frameNumber;

                dwTime_t timeDiff = static_cast<dwTime_t>(std::abs(
                    static_cast<int64_t>(lidarEntry.timestamp) -
                    static_cast<int64_t>(refEntry.timestamp)));
                maxTimeDiff = std::max(maxTimeDiff, timeDiff);
            }
        }

        // Find matching frames from other cameras
        for (uint32_t i = 0; i < m_config.numCameras; ++i)
        {
            if (i == refCamera)
            {
                continue;
            }

            auto cameraEntry = findAndRemoveNearestCameraFrame(i, refEntry.timestamp);
            if (cameraEntry.valid)
            {
                fusedPacket.cameraData[i] = std::move(cameraEntry.data);
                fusedPacket.cameraTimestamps[i] = cameraEntry.timestamp;
                fusedPacket.cameraFrameIds[i] = fusedPacket.cameraData[i].frameId;

                dwTime_t timeDiff = static_cast<dwTime_t>(std::abs(
                    static_cast<int64_t>(cameraEntry.timestamp) -
                    static_cast<int64_t>(refEntry.timestamp)));
                maxTimeDiff = std::max(maxTimeDiff, timeDiff);
            }
        }

        // Finalize packet
        finalizePacket(fusedPacket, maxTimeDiff);
        return true;
    }

    //--------------------------------------------------------------------------
    // NEAREST_TIMESTAMP: Find best matching set across all sensors
    //--------------------------------------------------------------------------
    bool trySyncNearestTimestamp(FusedPacket& fusedPacket)
    {
        // Collect candidate timestamps from all sensors
        std::vector<dwTime_t> candidates;

        {
            std::lock_guard<std::mutex> lock(m_lidarMutex);
            for (const auto& entry : m_lidarQueue)
            {
                if (entry.valid)
                {
                    candidates.push_back(entry.timestamp);
                }
            }
        }

        for (uint32_t i = 0; i < m_config.numCameras; ++i)
        {
            std::lock_guard<std::mutex> lock(m_cameraMutex[i]);
            for (const auto& entry : m_cameraQueues[i])
            {
                if (entry.valid)
                {
                    candidates.push_back(entry.timestamp);
                }
            }
        }

        if (candidates.empty())
        {
            return false;
        }

        // Find median timestamp as reference
        std::sort(candidates.begin(), candidates.end());
        dwTime_t refTimestamp = candidates[candidates.size() / 2];

        // Build packet using reference timestamp
        fusedPacket.clear();
        dwTime_t maxTimeDiff = 0;

        // Get nearest LiDAR
        if (m_config.enableLidar)
        {
            auto lidarEntry = findAndRemoveNearestLidarFrame(refTimestamp);
            if (lidarEntry.valid)
            {
                fusedPacket.lidarData = std::move(lidarEntry.data);
                fusedPacket.lidarTimestamp = lidarEntry.timestamp;
                fusedPacket.lidarFrameNumber = fusedPacket.lidarData.frameNumber;

                dwTime_t timeDiff = static_cast<dwTime_t>(std::abs(
                    static_cast<int64_t>(lidarEntry.timestamp) -
                    static_cast<int64_t>(refTimestamp)));
                maxTimeDiff = std::max(maxTimeDiff, timeDiff);
            }
        }

        // Get nearest cameras
        for (uint32_t i = 0; i < m_config.numCameras; ++i)
        {
            auto cameraEntry = findAndRemoveNearestCameraFrame(i, refTimestamp);
            if (cameraEntry.valid)
            {
                fusedPacket.cameraData[i] = std::move(cameraEntry.data);
                fusedPacket.cameraTimestamps[i] = cameraEntry.timestamp;
                fusedPacket.cameraFrameIds[i] = fusedPacket.cameraData[i].frameId;

                dwTime_t timeDiff = static_cast<dwTime_t>(std::abs(
                    static_cast<int64_t>(cameraEntry.timestamp) -
                    static_cast<int64_t>(refTimestamp)));
                maxTimeDiff = std::max(maxTimeDiff, timeDiff);
            }
        }

        // Check if we got any data
        bool hasData = fusedPacket.lidarData.valid;
        for (uint32_t i = 0; i < m_config.numCameras && !hasData; ++i)
        {
            hasData = fusedPacket.cameraData[i].valid;
        }

        if (!hasData)
        {
            return false;
        }

        finalizePacket(fusedPacket, maxTimeDiff);
        return true;
    }

    //--------------------------------------------------------------------------
    // LATEST_AVAILABLE: Use latest data from each sensor (no strict sync)
    // Good for visualization where freshness matters more than sync
    //--------------------------------------------------------------------------
    bool trySyncLatestAvailable(FusedPacket& fusedPacket)
    {
        fusedPacket.clear();
        bool hasData = false;

        // Get latest LiDAR frame (non-destructive peek, then pop)
        if (m_config.enableLidar)
        {
            std::lock_guard<std::mutex> lock(m_lidarMutex);
            if (!m_lidarQueue.empty())
            {
                // Find latest valid entry
                for (auto it = m_lidarQueue.rbegin(); it != m_lidarQueue.rend(); ++it)
                {
                    if (it->valid)
                    {
                        fusedPacket.lidarData = it->data;
                        fusedPacket.lidarTimestamp = it->timestamp;
                        fusedPacket.lidarFrameNumber = it->data.frameNumber;
                        hasData = true;
                        break;
                    }
                }
                // Clear old entries, keep only latest
                if (m_lidarQueue.size() > 1)
                {
                    auto latest = m_lidarQueue.back();
                    m_lidarQueue.clear();
                    m_lidarQueue.push_back(std::move(latest));
                }
            }
        }

        // Get latest camera frames
        for (uint32_t i = 0; i < m_config.numCameras; ++i)
        {
            std::lock_guard<std::mutex> lock(m_cameraMutex[i]);
            if (!m_cameraQueues[i].empty())
            {
                // Find latest valid entry
                for (auto it = m_cameraQueues[i].rbegin();
                     it != m_cameraQueues[i].rend(); ++it)
                {
                    if (it->valid)
                    {
                        fusedPacket.cameraData[i] = it->data;
                        fusedPacket.cameraTimestamps[i] = it->timestamp;
                        fusedPacket.cameraFrameIds[i] = it->data.frameId;
                        hasData = true;
                        break;
                    }
                }
                // Clear old entries, keep only latest
                if (m_cameraQueues[i].size() > 1)
                {
                    auto latest = m_cameraQueues[i].back();
                    m_cameraQueues[i].clear();
                    m_cameraQueues[i].push_back(std::move(latest));
                }
            }
        }

        if (!hasData)
        {
            return false;
        }

        // For LATEST_AVAILABLE, temporal alignment is not strictly enforced
        finalizePacket(fusedPacket, 0);
        return true;
    }

    //--------------------------------------------------------------------------
    // Find and remove nearest camera frame to reference timestamp
    //--------------------------------------------------------------------------
    TimestampedData<CameraFrameData> findAndRemoveNearestCameraFrame(
        uint32_t cameraIndex,
        dwTime_t refTimestamp)
    {
        TimestampedData<CameraFrameData> result{};

        std::lock_guard<std::mutex> lock(m_cameraMutex[cameraIndex]);

        if (m_cameraQueues[cameraIndex].empty())
        {
            return result;
        }

        dwTime_t minDiff = UINT64_MAX;
        size_t bestIndex = 0;

        for (size_t i = 0; i < m_cameraQueues[cameraIndex].size(); ++i)
        {
            const auto& entry = m_cameraQueues[cameraIndex][i];
            if (!entry.valid)
            {
                continue;
            }

            dwTime_t diff = static_cast<dwTime_t>(std::abs(
                static_cast<int64_t>(entry.timestamp) -
                static_cast<int64_t>(refTimestamp)));

            if (diff < minDiff)
            {
                minDiff = diff;
                bestIndex = i;
            }
        }

        // Check if within tolerance
        if (minDiff <= m_config.maxTimeDifferenceUs)
        {
            result = m_cameraQueues[cameraIndex][bestIndex];

            // Remove this and older entries to prevent reuse
            m_cameraQueues[cameraIndex].erase(
                m_cameraQueues[cameraIndex].begin(),
                m_cameraQueues[cameraIndex].begin() +
                    static_cast<std::ptrdiff_t>(bestIndex) + 1);
        }

        return result;
    }

    //--------------------------------------------------------------------------
    // Find and remove nearest LiDAR frame to reference timestamp
    //--------------------------------------------------------------------------
    TimestampedData<LidarFrameData> findAndRemoveNearestLidarFrame(
        dwTime_t refTimestamp)
    {
        TimestampedData<LidarFrameData> result{};

        std::lock_guard<std::mutex> lock(m_lidarMutex);

        if (m_lidarQueue.empty())
        {
            return result;
        }

        dwTime_t minDiff = UINT64_MAX;
        size_t bestIndex = 0;

        for (size_t i = 0; i < m_lidarQueue.size(); ++i)
        {
            const auto& entry = m_lidarQueue[i];
            if (!entry.valid)
            {
                continue;
            }

            dwTime_t diff = static_cast<dwTime_t>(std::abs(
                static_cast<int64_t>(entry.timestamp) -
                static_cast<int64_t>(refTimestamp)));

            if (diff < minDiff)
            {
                minDiff = diff;
                bestIndex = i;
            }
        }

        // Check if within tolerance
        if (minDiff <= m_config.maxTimeDifferenceUs)
        {
            result = m_lidarQueue[bestIndex];

            // Remove this and older entries to prevent reuse
            m_lidarQueue.erase(
                m_lidarQueue.begin(),
                m_lidarQueue.begin() + static_cast<std::ptrdiff_t>(bestIndex) + 1);
        }

        return result;
    }

    //--------------------------------------------------------------------------
    // Finalize fused packet with metadata
    //--------------------------------------------------------------------------
    void finalizePacket(FusedPacket& fusedPacket, dwTime_t maxTimeDiff)
    {
        fusedPacket.fusionTimestamp = getCurrentTimestamp();
        fusedPacket.fusionFrameNumber = m_fusionFrameNumber++;
        fusedPacket.temporalAlignmentError =
            static_cast<float>(maxTimeDiff) / 1000.0f;  // Convert us to ms
        fusedPacket.valid = true;
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

    //--------------------------------------------------------------------------
    // Convert policy to string for logging
    //--------------------------------------------------------------------------
    static const char* policyToString(SynchronizerConfig::FusionPolicy policy)
    {
        switch (policy)
        {
        case SynchronizerConfig::FusionPolicy::LIDAR_MASTER:
            return "LIDAR_MASTER";
        case SynchronizerConfig::FusionPolicy::CAMERA_MASTER:
            return "CAMERA_MASTER";
        case SynchronizerConfig::FusionPolicy::NEAREST_TIMESTAMP:
            return "NEAREST_TIMESTAMP";
        case SynchronizerConfig::FusionPolicy::LATEST_AVAILABLE:
            return "LATEST_AVAILABLE";
        default:
            return "UNKNOWN";
        }
    }

private:
    SynchronizerConfig m_config;

    // LiDAR queue with mutex
    mutable std::mutex m_lidarMutex;
    std::deque<TimestampedData<LidarFrameData>> m_lidarQueue;

    // Camera queues with per-camera mutex
    mutable std::array<std::mutex, MAX_CAMERAS> m_cameraMutex;
    std::array<std::deque<TimestampedData<CameraFrameData>>, MAX_CAMERAS>
        m_cameraQueues;

    // Fusion frame counter
    std::atomic<uint32_t> m_fusionFrameNumber{0};

    // Statistics
    std::atomic<uint64_t> m_syncedPackets{0};
    std::atomic<uint64_t> m_droppedLidarFrames{0};
    std::atomic<uint64_t> m_droppedCameraFrames{0};
};

} // namespace fusionengine

#endif // FUSIONENGINE_SENSORSYNCHRONIZER_HPP
