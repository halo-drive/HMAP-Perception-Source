/////////////////////////////////////////////////////////////////////////////////////////
// Livox HAP Plugin for NVIDIA DriveWorks
//
// Copyright (c) 2025 - All rights reserved.
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef LIVOX_HAP_PLUGIN_HPP
#define LIVOX_HAP_PLUGIN_HPP

#include <cstdint>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <dw/core/base/Types.h>
#include <dw/core/context/Context.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/lidar/Lidar.h>
#include <dw/sensors/legacy/plugins/lidar/LidarPlugin.h>

// Include Livox SDK headers
#include "livox_lidar_def.h"
#include "livox_lidar_api.h"

#include "BufferPool.hpp"
#include "LivoxHapProperties.hpp"

namespace dw
{
namespace plugin
{
namespace lidar
{

class LivoxHapPlugin
{
public:
    // Static container for all plugin instances
    static std::vector<std::unique_ptr<LivoxHapPlugin>> g_pluginInstances;

    // Constructor/Destructor
    explicit LivoxHapPlugin(dwContextHandle_t ctx);
    ~LivoxHapPlugin();

    // Common Sensor Plugin Functions
    dwStatus createSensor(dwSALHandle_t sal, const char* params);
    dwStatus startSensor();
    dwStatus stopSensor();
    dwStatus resetSensor();
    dwStatus releaseSensor();

    dwStatus readRawData(uint8_t const** data,
                         size_t* size,
                         dwTime_t* timestamp,
                         dwTime_t timeout_us);

    dwStatus returnRawData(uint8_t const* data);
    dwStatus pushData(size_t* lenPushed, uint8_t const* data, size_t const size);

    // Lidar Specific Plugin Functions
    dwStatus parseDataBuffer(dwLidarDecodedPacket* output, const dwTime_t hostTimestamp);
    dwStatus getConstants(_dwSensorLidarDecoder_constants* constants);

    // Helper functions
    bool isVirtualSensor() const;
    dwStatus getParameter(std::string& val, const std::string& param, const char* params);

private:
    // DriveWorks context
    dwContextHandle_t m_ctx;
    dwSALHandle_t m_sal = nullptr;

    // Sensor operation mode
    bool m_isVirtual = true;

    // Livox specific members
    std::atomic<bool> m_initialized{false};
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_connected{false};
    std::thread m_deviceThread;
    uint32_t m_deviceHandle{0}; // Livox SDK uses uint32_t for handles
    // m_broadcastCode removed - HAP uses IP-based discovery from config file
    char m_ipAddress[16] = {0}; // Livox device IP address
    std::string m_sdkConfigPath;
    // m_hostIpAddress removed - host IP is specified in config file
    std::vector<dwLidarPointXYZI> pointCloud_;
    std::vector<dwLidarPointXYZI> pointCloudBuffer_; 
    std::mutex m_pointCloudMutex; 
    
    // Buffer handling
    dw::plugins::common::BufferPool<RawPacket> m_dataBuffer{SLOT_SIZE};
    std::unordered_map<uint8_t*, RawPacket*> m_bufferMap;
    std::queue<RawPacket> m_packetQueue;
    std::mutex m_queueMutex;
    std::condition_variable m_queueCondition;
    
    // Parsed data
    LivoxLidarPacket m_lidarOutput;
    bool m_packetReady{false};
    std::mutex m_packetMutex;
    
    // Decoder constants
    _dwSensorLidarDecoder_constants m_constants{};
    bool m_constantsInitialized{false};
    uint16_t m_frameRate{10}; // Default 10Hz frame rate for HAP
    
    // Livox callbacks
    static void pointCloudCallback(const uint32_t handle, const uint8_t dev_type, 
                                   LivoxLidarEthernetPacket* data, void* client_data);
                                   
    static void infoChangeCallback(const uint32_t handle, const LivoxLidarInfo* info, void* client_data);
    static void controlCallback(livox_status status, uint32_t handle,
                               LivoxLidarAsyncControlResponse* response, void* client_data);
    static void queryDeviceStateCallback(livox_status status, uint32_t handle,
                                        LivoxLidarDiagInternalInfoResponse* response, void* client_data);
    
    // Livox data processing
    void processLivoxPointCloud(uint32_t handle, const LivoxLidarEthernetPacket* data);
    bool isScanComplete();
    void deviceManagementThread();
    bool findAndConnectDevice();
    void dumpPacketInfo(const LivoxLidarEthernetPacket* packet);
    void dumpPoint(const dwLidarPointXYZI& point, int index);
    
    // Format conversion
    void convertLivoxToDriveWorksFormat(const LivoxLidarEthernetPacket* packet);

    void dumpRawPacketData(const uint8_t* data, size_t size, const char* label);
    void dumpDetailedPointData(const LivoxLidarEthernetPacket* packet);
    void analyzePointStatistics(const LivoxLidarEthernetPacket* packet);
    void dumpConvertedPoints(const char* stage, int count);
    void analyzePacketMemoryLayout(const LivoxLidarEthernetPacket* packet);
    void checkOutputBufferIntegrity();
    void analyzeBinaryPacket(const uint8_t* data, size_t size);
    void processCartesianHighData(const uint8_t* pointData, uint16_t dotNum, uint8_t frameCount);

    // HAP specific methods
    void processHapPointCloud(uint32_t handle, const LivoxLidarEthernetPacket* data);
    bool isHapDevice(const LivoxLidarInfo* info) const;
    dwStatus configureHapDevice(uint32_t handle);

    // Performance monitoring
    std::atomic<uint64_t> m_totalPacketsReceived{0};
    std::atomic<uint64_t> m_totalPointsReceived{0};
    std::atomic<uint64_t> m_droppedPackets{0};
    std::atomic<uint64_t> m_invalidPackets{0};
    std::chrono::time_point<std::chrono::steady_clock> m_statsTimer{std::chrono::steady_clock::now()};
    
    // Callback tracking
    static std::atomic<uint64_t> s_callbackCallCount;

};

} // namespace lidar
} // namespace plugin
} // namespace dw

#endif // LIVOX_HAP_PLUGIN_HPP 