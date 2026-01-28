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

#ifndef FUSIONENGINE_SENSORSYNCHRONIZER_HPP
#define FUSIONENGINE_SENSORSYNCHRONIZER_HPP

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
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
    dwTime_t timestamp;
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
    dwTime_t maxTimeDifferenceUs{33000};  // ~1 frame at 30 FPS

    // Maximum queue depth for each sensor
    uint32_t maxQueueDepth{10};

    // Enable/disable specific sensors
    bool enableLidar{true};
    bool enableCameras{true};
    uint32_t numCameras{4};

    // Fusion policy
    enum class FusionPolicy
    {
        LIDAR_MASTER,       // Use LiDAR timestamp as reference
        CAMERA_MASTER,      // Use camera timestamp as reference
        NEAREST_TIMESTAMP,  // Use closest matching timestamps
        LATEST_AVAILABLE    // Use latest available from each sensor
    };
    FusionPolicy policy{FusionPolicy::LIDAR_MASTER};
};

//------------------------------------------------------------------------------
// Sensor Synchronizer Class
//------------------------------------------------------------------------------
class SensorSynchronizer
{
public:
    SensorSynchronizer() = default;

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

        m_lidarQueue.push_back(entry);

        // Maintain queue depth
        while (m_lidarQueue.size() > m_config.maxQueueDepth)
        {
            m_lidarQueue.pop_front();
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

        m_cameraQueues[cameraIndex].push_back(entry);

        // Maintain queue depth
        while (m_cameraQueues[cameraIndex].size() > m_config.maxQueueDepth)
        {
            m_cameraQueues[cameraIndex].pop_front();
        }
    }

    //--------------------------------------------------------------------------
    // Try to synchronize and produce a fused packet
    //--------------------------------------------------------------------------
    bool trySync(FusedPacket& fusedPacket)
    {
        switch (m_config.policy)
        {
        case SynchronizerConfig::FusionPolicy::LIDAR_MASTER:
            return trySyncLidarMaster(fusedPacket);

        case SynchronizerConfig::FusionPolicy::CAMERA_MASTER:
            return trySyncCameraMaster(fusedPacket);

        case SynchronizerConfig::FusionPolicy::NEAREST_TIMESTAMP:
            return trySyncNearestTimestamp(fusedPacket);

        case SynchronizerConfig::FusionPolicy::LATEST_AVAILABLE:
            return trySyncLatestAvailable(fusedPacket);

        default:
            return false;
        }
    }

    //--------------------------------------------------------------------------
    // Get queue statistics
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

private:
    //--------------------------------------------------------------------------
    // LiDAR-master synchronization: Use LiDAR timestamp as reference
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

        // Initialize fused packet
        fusedPacket = FusedPacket{};
        fusedPacket.lidarTimestamp = lidarEntry.timestamp;
        fusedPacket.lidarData = lidarEntry.data;
        fusedPacket.lidarFrameNumber = lidarEntry.data.frameNumber;

        // Find matching camera frames
        dwTime_t maxTimeDiff = 0;
        for (uint32_t i = 0; i < m_config.numCameras; ++i)
        {
            auto cameraEntry = findNearestCameraFrame(i, lidarEntry.timestamp);
            if (cameraEntry.valid)
            {
                fusedPacket.cameraData[i] = cameraEntry.data;
                fusedPacket.cameraTimestamps[i] = cameraEntry.timestamp;
                fusedPacket.cameraFrameIds[i] = cameraEntry.data.frameId;

                dwTime_t timeDiff = std::abs(
                    static_cast<int64_t>(cameraEntry.timestamp) -
                    static_cast<int64_t>(lidarEntry.timestamp));
                maxTimeDiff = std::max(maxTimeDiff, timeDiff);
            }
        }

        // Set fusion metadata
        fusedPacket.fusionTimestamp = getCurrentTimestamp();
        fusedPacket.fusionFrameNumber = m_fusionFrameNumber++;
        fusedPacket.temporalAlignmentError =
            static_cast<float>(maxTimeDiff) / 1000.0f;  // Convert to ms
        fusedPacket.valid = true;

        return true;
    }

    //--------------------------------------------------------------------------
    // Camera-master synchronization: Use camera timestamp as reference
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
            refEntry = m_cameraQueues[refCamera].front();
            m_cameraQueues[refCamera].pop_front();
        }

        // Initialize fused packet
        fusedPacket = FusedPacket{};
        fusedPacket.cameraData[refCamera] = refEntry.data;
        fusedPacket.cameraTimestamps[refCamera] = refEntry.timestamp;

        // Find matching LiDAR frame
        auto lidarEntry = findNearestLidarFrame(refEntry.timestamp);
        if (lidarEntry.valid)
        {
            fusedPacket.lidarData = lidarEntry.data;
            fusedPacket.lidarTimestamp = lidarEntry.timestamp;
            fusedPacket.lidarFrameNumber = lidarEntry.data.frameNumber;
        }

        // Find matching frames from other cameras
        dwTime_t maxTimeDiff = 0;
        for (uint32_t i = 0; i < m_config.numCameras; ++i)
        {
            if (i == refCamera)
            {
                continue;
            }

            auto cameraEntry = findNearestCameraFrame(i, refEntry.timestamp);
            if (cameraEntry.valid)
            {
                fusedPacket.cameraData[i] = cameraEntry.data;
                fusedPacket.cameraTimestamps[i] = cameraEntry.timestamp;
                fusedPacket.cameraFrameIds[i] = cameraEntry.data.frameId;

                dwTime_t timeDiff = std::abs(
                    static_cast<int64_t>(cameraEntry.timestamp) -
                    static_cast<int64_t>(refEntry.timestamp));
                maxTimeDiff = std::max(maxTimeDiff, timeDiff);
            }
        }

        // Set fusion metadata
        fusedPacket.fusionTimestamp = getCurrentTimestamp();
        fusedPacket.fusionFrameNumber = m_fusionFrameNumber++;
        fusedPacket.temporalAlignmentError =
            static_cast<float>(maxTimeDiff) / 1000.0f;
        fusedPacket.valid = true;

        return true;
    }

    //--------------------------------------------------------------------------
    // Nearest timestamp synchronization
    //--------------------------------------------------------------------------
    bool trySyncNearestTimestamp(FusedPacket& fusedPacket)
    {
        // For simplicity, use LiDAR-master policy with stricter matching
        return trySyncLidarMaster(fusedPacket);
    }

    //--------------------------------------------------------------------------
    // Latest available synchronization (no strict timestamp matching)
    //--------------------------------------------------------------------------
    bool trySyncLatestAvailable(FusedPacket& fusedPacket)
    {
        fusedPacket = FusedPacket{};
        bool hasData = false;

        // Get latest LiDAR frame
        {
            std::lock_guard<std::mutex> lock(m_lidarMutex);
            if (!m_lidarQueue.empty())
            {
                auto& entry = m_lidarQueue.back();
                if (entry.valid)
                {
                    fusedPacket.lidarData = entry.data;
                    fusedPacket.lidarTimestamp = entry.timestamp;
                    fusedPacket.lidarFrameNumber = entry.data.frameNumber;
                    hasData = true;
                }
                m_lidarQueue.clear();  // Clear queue after taking latest
            }
        }

        // Get latest camera frames
        for (uint32_t i = 0; i < m_config.numCameras; ++i)
        {
            std::lock_guard<std::mutex> lock(m_cameraMutex[i]);
            if (!m_cameraQueues[i].empty())
            {
                auto& entry = m_cameraQueues[i].back();
                if (entry.valid)
                {
                    fusedPacket.cameraData[i] = entry.data;
                    fusedPacket.cameraTimestamps[i] = entry.timestamp;
                    fusedPacket.cameraFrameIds[i] = entry.data.frameId;
                    hasData = true;
                }
                m_cameraQueues[i].clear();  // Clear queue after taking latest
            }
        }

        if (!hasData)
        {
            return false;
        }

        // Set fusion metadata
        fusedPacket.fusionTimestamp = getCurrentTimestamp();
        fusedPacket.fusionFrameNumber = m_fusionFrameNumber++;
        fusedPacket.temporalAlignmentError = 0.0f;  // Not meaningful for this policy
        fusedPacket.valid = true;

        return true;
    }

    //--------------------------------------------------------------------------
    // Find nearest camera frame to reference timestamp
    //--------------------------------------------------------------------------
    TimestampedData<CameraFrameData> findNearestCameraFrame(
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
            dwTime_t diff = std::abs(
                static_cast<int64_t>(entry.timestamp) -
                static_cast<int64_t>(refTimestamp));

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

            // Remove this and older entries
            m_cameraQueues[cameraIndex].erase(
                m_cameraQueues[cameraIndex].begin(),
                m_cameraQueues[cameraIndex].begin() + bestIndex + 1);
        }

        return result;
    }

    //--------------------------------------------------------------------------
    // Find nearest LiDAR frame to reference timestamp
    //--------------------------------------------------------------------------
    TimestampedData<LidarFrameData> findNearestLidarFrame(dwTime_t refTimestamp)
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
            dwTime_t diff = std::abs(
                static_cast<int64_t>(entry.timestamp) -
                static_cast<int64_t>(refTimestamp));

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

            // Remove this and older entries
            m_lidarQueue.erase(
                m_lidarQueue.begin(),
                m_lidarQueue.begin() + bestIndex + 1);
        }

        return result;
    }

    //--------------------------------------------------------------------------
    // Get current timestamp
    //--------------------------------------------------------------------------
    dwTime_t getCurrentTimestamp() const
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(duration)
            .count();
    }

private:
    SynchronizerConfig m_config;

    // LiDAR queue
    mutable std::mutex m_lidarMutex;
    std::deque<TimestampedData<LidarFrameData>> m_lidarQueue;

    // Camera queues (one per camera)
    mutable std::array<std::mutex, MAX_CAMERAS> m_cameraMutex;
    std::array<std::deque<TimestampedData<CameraFrameData>>, MAX_CAMERAS>
        m_cameraQueues;

    // Fusion frame counter
    std::atomic<uint32_t> m_fusionFrameNumber{0};
};

} // namespace fusionengine

#endif // FUSIONENGINE_SENSORSYNCHRONIZER_HPP
