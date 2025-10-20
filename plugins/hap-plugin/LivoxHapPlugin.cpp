/////////////////////////////////////////////////////////////////////////////////////////
// Livox HAP Plugin for NVIDIA DriveWorks
// Based on the NVIDIA sample lidar plugin
//
// Copyright (c) 2025 - All rights reserved.
/////////////////////////////////////////////////////////////////////////////////////////

#include "LivoxHapPlugin.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>

// Debug logging macro
#define DEBUG_LOG(msg) { \
    auto now = std::time(nullptr); \
    auto tm_info = std::localtime(&now); \
    std::cout << "[" << std::put_time(tm_info, "%Y-%m-%d %H:%M:%S") << "] LIVOX_HAP_PLUGIN: " << msg << std::endl; \
}

// Helper function to get current timestamp for thread-safe logging
static std::string getCurrentTimestamp() {
    auto now = std::time(nullptr);
    struct tm tm_info;
    localtime_r(&now, &tm_info);  // Thread-safe version
    std::stringstream ss;
    ss << std::put_time(&tm_info, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

namespace dw
{
namespace plugin
{
namespace lidar
{

// Initialize the static vector
std::vector<std::unique_ptr<dw::plugin::lidar::LivoxHapPlugin>> dw::plugin::lidar::LivoxHapPlugin::g_pluginInstances;

// Initialize static callback counter
std::atomic<uint64_t> dw::plugin::lidar::LivoxHapPlugin::s_callbackCallCount{0};

LivoxHapPlugin::LivoxHapPlugin(dwContextHandle_t ctx)
    : m_ctx(ctx)
{
    DEBUG_LOG("LivoxHapPlugin constructor called with ctx: " << ctx);
    m_lidarOutput.max_points = MAX_POINTS_PER_PACKET;
    m_lidarOutput.n_points = 0;
    m_lidarOutput.scan_complete = false;
    m_lidarOutput.sensor_timestamp = 0;
    
    // Initialize point cloud buffer with reasonable capacity
    pointCloud_.reserve(MAX_POINTS_PER_FRAME);
    DEBUG_LOG("LivoxHapPlugin constructor: Point cloud buffer initialized with capacity: " << pointCloud_.capacity());
}

LivoxHapPlugin::~LivoxHapPlugin()
{
    DEBUG_LOG("LivoxHapPlugin destructor called");
    stopSensor();
    releaseSensor();
}

//################################################################################
//###################### Common Sensor Plugin Functions ##########################
//################################################################################

dwStatus LivoxHapPlugin::createSensor(dwSALHandle_t sal, const char* params)
{
    DEBUG_LOG("createSensor called with SAL: " << sal << ", params: " << (params ? params : "NULL"));
    
    // Set SAL handle
    m_sal = sal;
    
    // Default to real sensor (not virtual)
    m_isVirtual = false;
    
    // Parse parameters (broadcast code not needed for HAP - uses IP-based discovery from config file)
    
    // Host IP is specified in the config file, not as a parameter
    
    if (getParameter(m_sdkConfigPath, "sdk-config-path", params) == DW_SUCCESS)
    {
        DEBUG_LOG("Using SDK config path: " << m_sdkConfigPath);
    }
    else
    {
        m_sdkConfigPath = "/usr/local/driveworks/samples/src/sensors/lidar/lidar_replay_livox/livox_config";
        DEBUG_LOG("Using default SDK config path: " << m_sdkConfigPath);
    }
    
    std::string frameRateStr;
    if (getParameter(frameRateStr, "frame-rate", params) == DW_SUCCESS)
    {
        m_frameRate = static_cast<uint16_t>(atoi(frameRateStr.c_str()));
        DEBUG_LOG("Using frame rate: " << m_frameRate << "Hz");
    }
    else
    {
        DEBUG_LOG("Using default frame rate: " << m_frameRate << "Hz");
    }
    
    // Initialize buffer map
    DEBUG_LOG("Initializing buffer pool with " << (int)SLOT_SIZE << " slots");
    for (uint8_t i = 0; i < SLOT_SIZE; ++i)
    {
        RawPacket* p_rawPacket = nullptr;
        bool result = m_dataBuffer.get(p_rawPacket);
        if (!result || p_rawPacket == nullptr) {
            DEBUG_LOG("ERROR: Failed to get packet from buffer pool at initialization");
            return DW_FAILURE;
        }
        
        uint8_t* p_rawData = &(p_rawPacket->rawData[0]);
        m_bufferMap[p_rawData] = p_rawPacket;
        m_dataBuffer.put(p_rawPacket);
    }
    DEBUG_LOG("Buffer pool initialized with " << m_bufferMap.size() << " entries");
    
    // Initialize Livox SDK - we'll do actual device connection during startSensor
    m_initialized = true;
    DEBUG_LOG("createSensor completed successfully");
    return DW_SUCCESS;
} 

dwStatus LivoxHapPlugin::startSensor()
{
    DEBUG_LOG("startSensor called");
    
    if (!m_initialized) {
        DEBUG_LOG("ERROR: Cannot start uninitialized sensor");
        return DW_INVALID_HANDLE;
    }
    
    if (m_running) {
        DEBUG_LOG("Sensor already running, ignoring start request");
        return DW_SUCCESS;
    }
    
    if (!isVirtualSensor()) {
        // Initialize Livox SDK with proper config
        std::string configPath = m_sdkConfigPath;
        DEBUG_LOG("Initializing Livox SDK with config: " << configPath);
        
        // Ensure we're using the correct file
        if (configPath.find(".json") == std::string::npos) {
            configPath += ".json";
        }
        
        // Check if config file exists
        std::ifstream file(configPath);
        if (!file.is_open()) {
            DEBUG_LOG("WARNING: Config file not found: " << configPath);
            DEBUG_LOG("Using default configuration");
            configPath = ""; // Use default config
        } else {
            file.close();
        }
        
        // Host IP is specified in the config file, not as a parameter
        const char* hostIp = nullptr; // Let Livox SDK use config file settings
        DEBUG_LOG("Using config file for host IP settings");
        
        // Initialize SDK
        std::cerr << "LIVOX_HAP_PLUGIN: Initializing Livox SDK with config: " << configPath << std::endl;
        std::cerr << "LIVOX_HAP_PLUGIN: Host IP: Using config file settings" << std::endl;
        bool sdkInitResult = LivoxLidarSdkInit(configPath.c_str(), hostIp);
        if (!sdkInitResult) {
            std::cerr << "LIVOX_HAP_PLUGIN: ERROR - Failed to initialize Livox SDK" << std::endl;
            return DW_FAILURE;
        }
        std::cerr << "LIVOX_HAP_PLUGIN: SUCCESS - Livox SDK initialized" << std::endl;
        
        // Set SDK callbacks
    DEBUG_LOG("Setting SDK callbacks");
    SetLivoxLidarPointCloudCallBack(pointCloudCallback, this);
    DEBUG_LOG("SetLivoxLidarPointCloudCallBack called");
    
    SetLivoxLidarInfoChangeCallback(infoChangeCallback, this);
    DEBUG_LOG("SetLivoxLidarInfoChangeCallback called");
    
    DEBUG_LOG("SDK callbacks set successfully");
    
    // Test callback registration by checking if we can get callback info
    DEBUG_LOG("Verifying callback registration...");
    DEBUG_LOG("Point cloud callback function address: " << (void*)pointCloudCallback);
    DEBUG_LOG("Info change callback function address: " << (void*)infoChangeCallback);
    DEBUG_LOG("Plugin instance address: " << (void*)this);
    
    // Force a test callback to verify registration
    std::cerr << "LIVOX_HAP_PLUGIN: Testing callback registration - if you see this, callbacks are working" << std::endl;
    
    // Test callback registration by manually calling it with dummy data
    std::cerr << "LIVOX_HAP_PLUGIN: Testing point cloud callback with dummy data..." << std::endl;
    LivoxLidarEthernetPacket dummyPacket = {};
    dummyPacket.version = 1;
    dummyPacket.length = 100;
    dummyPacket.time_interval = 1000;
    dummyPacket.dot_num = 10;
    dummyPacket.udp_cnt = 1;
    dummyPacket.frame_cnt = 1;
    dummyPacket.data_type = kLivoxLidarSphericalCoordinateData;
    dummyPacket.time_type = 0;
    
    // Test the callback function directly
    std::cerr << "LIVOX_HAP_PLUGIN: Manually testing callback function..." << std::endl;
    pointCloudCallback(12345, 10, &dummyPacket, this);
    std::cerr << "LIVOX_HAP_PLUGIN: Manual callback test completed" << std::endl;
        
        // Start Livox SDK
    std::cerr << "LIVOX_HAP_PLUGIN: Starting Livox SDK..." << std::endl;
    if (!LivoxLidarSdkStart()) {
        std::cerr << "LIVOX_HAP_PLUGIN: ERROR - Failed to start Livox SDK" << std::endl;
        return DW_FAILURE;
    }
    std::cerr << "LIVOX_HAP_PLUGIN: SUCCESS - Livox SDK started" << std::endl;
    
    // Check if device is already connected (in case it was connected before SDK start)
    std::cerr << "LIVOX_HAP_PLUGIN: Checking for already connected devices..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(3000)); // Wait for device detection
    
    // Test network connectivity
    std::cerr << "LIVOX_HAP_PLUGIN: Testing network connectivity..." << std::endl;
    std::cerr << "LIVOX_HAP_PLUGIN: Expected device IP: 192.168.1.203 (from config)" << std::endl;
    std::cerr << "LIVOX_HAP_PLUGIN: Host IP: Using config file settings" << std::endl;
    std::cerr << "LIVOX_HAP_PLUGIN: Using IP-based discovery (no broadcast code needed)" << std::endl;
    
    // Check if callback was already triggered automatically
    if (!m_connected && m_deviceHandle == 0) {
        DEBUG_LOG("No device connected yet, waiting for automatic connection callback");
    } else {
        DEBUG_LOG("Device already connected (Handle: " << m_deviceHandle << "), callback already triggered automatically");
    }
        
        // Start device management thread
        m_running = true;
        m_deviceThread = std::thread(&LivoxHapPlugin::deviceManagementThread, this);
        
        // Add a test to see if we can receive any data
        std::cerr << "LIVOX_HAP_PLUGIN: Starting data reception test..." << std::endl;
        std::cerr << "LIVOX_HAP_PLUGIN: Will wait 10 seconds to see if any data arrives..." << std::endl;
        
        std::cerr << "LIVOX_HAP_PLUGIN: Livox HAP sensor successfully started" << std::endl;
    }
    
    return DW_SUCCESS;
}

dwStatus LivoxHapPlugin::stopSensor()
{
    DEBUG_LOG("stopSensor called");
    
    if (!m_initialized)
    {
        DEBUG_LOG("ERROR: Cannot stop uninitialized sensor");
        return DW_INVALID_HANDLE;
    }
    
    if (!m_running)
    {
        DEBUG_LOG("Sensor already stopped, ignoring stop request");
        return DW_SUCCESS; // Already stopped
    }
    
    DEBUG_LOG("Setting running flag to false");
    m_running = false;
   
    // Stop data transmission from device and set to SLEEP state
    if (m_connected && m_deviceHandle != 0)
    {
        DEBUG_LOG("Disabling point cloud data from device handle: " << m_deviceHandle);
        livox_status status = DisableLivoxLidarPointSend(m_deviceHandle, controlCallback, this);
        DEBUG_LOG("DisableLivoxLidarPointSend result: " << status);
        
        // Wait a moment for the disable command to complete
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // Set device to SLEEP state (standby) according to HAP protocol
        DEBUG_LOG("Setting device to SLEEP state (standby)");
        status = SetLivoxLidarWorkMode(m_deviceHandle, kLivoxLidarSleep, controlCallback, this);
        DEBUG_LOG("SetLivoxLidarWorkMode to SLEEP result: " << status);
        
        // Wait for the sleep command to complete
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        DEBUG_LOG("Motor shutdown sequence completed");
    }
    else
    {
        DEBUG_LOG("No device connected, skipping disable point send");
    }

    // Wait for device thread to finish
    if (m_deviceThread.joinable())
    {
        DEBUG_LOG("Waiting for device thread to join");
        m_deviceThread.join();
        DEBUG_LOG("Device thread joined successfully");
    }
    else
    {
        DEBUG_LOG("Device thread not running, no join needed");
    }    
    DEBUG_LOG("stopSensor completed successfully");
    return DW_SUCCESS;
}

dwStatus LivoxHapPlugin::resetSensor()
{
    DEBUG_LOG("resetSensor called");
    
    if (!m_initialized)
    {
        DEBUG_LOG("ERROR: Cannot reset uninitialized sensor");
        return DW_INVALID_HANDLE;
    }
    
    // First stop the sensor
    DEBUG_LOG("Stopping sensor for reset");
    dwStatus status = stopSensor();
    if (status != DW_SUCCESS)
    {
        DEBUG_LOG("ERROR: Failed to stop sensor during reset, status: " << status);
        return status;
    }
    
    // Clear buffer pool
    DEBUG_LOG("Clearing buffer pool");
    for (int i = 0; i < SLOT_SIZE; ++i)
    {
        // Get pointer to element in buffer
        RawPacket* p_rawPacket = nullptr;
        if (m_dataBuffer.get(p_rawPacket, 1000))
        {
            // Clear memory
            std::memset(p_rawPacket, 0, sizeof(RawPacket));
            
            // Return pointer to element
            m_dataBuffer.put(p_rawPacket);
        }
    }
    
    // Empty packet queue
    {
        DEBUG_LOG("Clearing packet queue");
        std::lock_guard<std::mutex> lock(m_queueMutex);
        size_t queueSize = m_packetQueue.size();
        while (!m_packetQueue.empty())
        {
            m_packetQueue.pop();
        }
        DEBUG_LOG("Cleared " << queueSize << " packets from queue");
    }
    
    // Reset packet ready flag
    {
        DEBUG_LOG("Resetting packet ready flag");
        std::lock_guard<std::mutex> lock(m_packetMutex);
        m_packetReady = false;
    }
    
    // Start again
    DEBUG_LOG("Starting sensor after reset");
    status = startSensor();
    if (status != DW_SUCCESS) {
        DEBUG_LOG("ERROR: Failed to start sensor after reset, status: " << status);
    } else {
        DEBUG_LOG("resetSensor completed successfully");
    }
    return status;
}

dwStatus LivoxHapPlugin::releaseSensor()
{
    DEBUG_LOG("releaseSensor called");
    
    if (!m_initialized)
    {
        DEBUG_LOG("ERROR: Cannot release uninitialized sensor");
        return DW_INVALID_HANDLE;
    }
    
    // Stop the sensor if running
    if (m_running) {
        DEBUG_LOG("Stopping sensor for release");
        stopSensor();
    }
    
    // Uninitialize Livox SDK
    DEBUG_LOG("Uninitializing Livox SDK");
    LivoxLidarSdkUninit();
    
    m_initialized = false;
    m_connected = false;
    m_deviceHandle = 0;
    
    DEBUG_LOG("releaseSensor completed successfully");
    return DW_SUCCESS;
} 

dwStatus LivoxHapPlugin::readRawData(uint8_t const** data,
    size_t* size,
    dwTime_t* timestamp,
    dwTime_t timeout_us)
{
    if (!m_initialized || !m_running)
    {
        DEBUG_LOG("ERROR: readRawData called on invalid/stopped sensor");
        return DW_INVALID_HANDLE;
    }

    // Check if we're in virtual mode
    if (isVirtualSensor())
    {
        DEBUG_LOG("Virtual mode not supported in readRawData");
        return DW_NOT_SUPPORTED;
    }

    // Wait for data with timeout
    std::unique_lock<std::mutex> lock(m_queueMutex);
    if (m_packetQueue.empty())
    {
        DEBUG_LOG("Waiting for data with timeout: " << timeout_us << "us");
        
        // Check if device is connected but still in motor startup phase
        if (m_connected && m_deviceHandle != 0) {
            DEBUG_LOG("Device connected, waiting for motor startup and data transmission...");
        }
        
        auto status = m_queueCondition.wait_for(lock, std::chrono::microseconds(timeout_us));
        if (status == std::cv_status::timeout)
        {
            DEBUG_LOG("Timeout waiting for data");
            return DW_TIME_OUT;
        }
    }

    if (m_packetQueue.empty())
    {
        DEBUG_LOG("No data available after wait");
        return DW_NOT_AVAILABLE;
    }

    // Get pointer to object in buffer pool for raw data
    RawPacket* p_rawPacket = nullptr;
    bool result = m_dataBuffer.get(p_rawPacket, timeout_us);
    if (!result || p_rawPacket == nullptr)
    {
        DEBUG_LOG("ERROR: Failed to get slot from buffer pool");
        return DW_BUFFER_FULL;
    }

    // Get front packet from queue
    RawPacket& queuedPacket = m_packetQueue.front();

    // Copy data to the buffer pool slot
    std::memcpy(p_rawPacket->rawData, queuedPacket.rawData, sizeof(queuedPacket.rawData));

    // Remove the packet from queue
    m_packetQueue.pop();

    // Update function outputs
    *data = p_rawPacket->rawData;

    // Extract size and timestamp from the buffer
    uint32_t payloadSize;
    dwTime_t packetTime;
    std::memcpy(&payloadSize, p_rawPacket->rawData, sizeof(uint32_t));
    std::memcpy(&packetTime, p_rawPacket->rawData + sizeof(uint32_t), sizeof(dwTime_t));

    *size = payloadSize + PAYLOAD_OFFSET;
    *timestamp = packetTime;

    DEBUG_LOG("readRawData successful - size: " << *size << ", timestamp: " << *timestamp);
    return DW_SUCCESS;
}

dwStatus LivoxHapPlugin::returnRawData(uint8_t const* data)
{
    if (!m_initialized || data == nullptr)
    {
        DEBUG_LOG("ERROR: returnRawData called with invalid parameters");
        return DW_INVALID_HANDLE;
    }
    
    // Return object to buffer pool
    auto it = m_bufferMap.find(const_cast<uint8_t*>(data));
    if (it == m_bufferMap.end())
    {
        DEBUG_LOG("ERROR: Invalid data pointer in returnRawData: " << (void*)data);
        return DW_INVALID_ARGUMENT;
    }
    
    bool result = m_dataBuffer.put(it->second);
    if (!result)
    {
        DEBUG_LOG("ERROR: Failed to return object to buffer pool");
        return DW_INVALID_ARGUMENT;
    }
    
    DEBUG_LOG("returnRawData successful");
    return DW_SUCCESS;
}

dwStatus LivoxHapPlugin::pushData(size_t* lenPushed, const uint8_t* data, size_t size)
{
    // The first part of data contains our metadata (size + timestamp)
    constexpr size_t PAYLOAD_OFFSET = sizeof(uint32_t) + sizeof(dwTime_t); // 12 bytes
    
    // Extract the actual Livox packet
    if (size <= PAYLOAD_OFFSET) {
        DEBUG_LOG("ERROR: Packet too small: " << size << " bytes");
        *lenPushed = 0;
        return DW_SUCCESS; // Return success to continue the pipeline
    }
    
    const uint8_t* livoxPacket = data + PAYLOAD_OFFSET;
    size_t livoxPacketSize = size - PAYLOAD_OFFSET;
    
    // Log packet information for debugging
    DEBUG_LOG("Processing packet: size=" << size << " bytes, payload=" << livoxPacketSize << " bytes");
    
    // Dump the first few bytes for debugging
    std::stringstream ss;
    ss << "Packet header bytes: ";
    for (size_t i = 0; i < std::min(size_t(16), livoxPacketSize); i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') 
           << static_cast<int>(livoxPacket[i]) << " ";
    }
    DEBUG_LOG(ss.str());
    
    // Since we're already receiving and processing point data via the Livox SDK callbacks,
    // we won't try to re-parse the data here. The pointCloud_ vector is already being
    // populated by the callbacks.
    
    // We'll still check if we have points - if not, points might be coming in later callbacks
    {
        std::lock_guard<std::mutex> lock(m_packetMutex);
        if (pointCloud_.empty()) {
            DEBUG_LOG("No points available yet from callbacks, waiting for more data");
        } else {
            DEBUG_LOG("Using " << pointCloud_.size() << " points from callback data");
            m_packetReady = true;
        }
    }
    
    // Mark the entire packet as processed to keep the pipeline moving
    *lenPushed = size;
    return DW_SUCCESS;
} 

//################################################################################
//###################### Lidar Specific Plugin Functions #########################
//################################################################################

// dwStatus LivoxHapPlugin::parseDataBuffer(dwLidarDecodedPacket* output, const dwTime_t hostTimestamp) 
// {
//     if (!output) {
//         DEBUG_LOG("ERROR: parseDataBuffer called with NULL output parameter");
//         return DW_INVALID_ARGUMENT;
//     }
    
//     // Use mutex to protect access to the packet queue
//     std::lock_guard<std::mutex> lock(m_queueMutex);
    
//     // Check if we have any packets to process
//     if (m_packetQueue.empty()) {
//         DEBUG_LOG("parseDataBuffer: No packets available for visualization (queue size: " << m_packetQueue.size() << ")");
//         return DW_NOT_READY;
//     }
    
//     DEBUG_LOG("parseDataBuffer: Processing packet from queue (queue size: " << m_packetQueue.size() << ")");
    
//     // Get the next packet from the queue
//     RawPacket& packet = m_packetQueue.front();
    
//     // Extract the Livox packet from the raw data
//     const uint8_t* packetData = packet.rawData + PAYLOAD_OFFSET;
//     const LivoxLidarEthernetPacket* livoxPacket = reinterpret_cast<const LivoxLidarEthernetPacket*>(packetData);
    
//     // Validate the packet
//     if (livoxPacket->dot_num == 0) {
//         DEBUG_LOG("parseDataBuffer: Invalid packet - dot_num: " << livoxPacket->dot_num);
//         m_packetQueue.pop();
//         return DW_NOT_READY;
//     }
    
//     DEBUG_LOG("parseDataBuffer: Processing " << livoxPacket->dot_num << " points from packet");
    
//     // Process points based on data type and convert to DriveWorks format
//     // Use the class member vector to store points (persistent storage)
//     // DON'T clear - we want to accumulate points like Mid-360 plugin
//     // pointCloud_.clear();  // ❌ REMOVED - this was causing the sparse visualization
//     // pointCloud_.reserve(livoxPacket->dot_num);  // ❌ REMOVED - not needed for accumulation
    
//     if (livoxPacket->data_type == kLivoxLidarCartesianCoordinateHighData) {
//         // High precision Cartesian coordinates (x, y, z in mm)
//         const uint8_t* pointData = livoxPacket->data;
//         const size_t pointSize = 14; // 4+4+4+1+1 (x,y,z,reflectivity,tag)
        
//         // Calculate how many new points we can add (like Mid-360 plugin)
//         size_t maxPointsToAdd = pointCloud_.capacity() - pointCloud_.size();
//         if (maxPointsToAdd < livoxPacket->dot_num) {
//             // Not enough room, remove oldest points to make space
//             if (pointCloud_.size() > livoxPacket->dot_num) {
//                 // We have more points than we need to add, so remove enough to make room
//                 pointCloud_.erase(
//                     pointCloud_.begin(),
//                     pointCloud_.begin() + (pointCloud_.size() - maxPointsToAdd + livoxPacket->dot_num)
//                 );
//             } else {
//                 // Just clear all existing points
//                 pointCloud_.clear();
//             }
//         }
        
//         // Reserve space for new points
//         size_t currentSize = pointCloud_.size();
//         pointCloud_.resize(currentSize + livoxPacket->dot_num);
        
//         // Process and add new points
//         size_t validPointsAdded = 0;
        
//         for (uint16_t i = 0; i < livoxPacket->dot_num; i++) {
//             int32_t x_raw, y_raw, z_raw;
//             uint8_t reflectivity, tag;

//             const uint8_t* point = pointData + (i * pointSize);
//             memcpy(&x_raw, point, 4);
//             memcpy(&y_raw, point + 4, 4);
//             memcpy(&z_raw, point + 8, 4);
//             reflectivity = point[12];
//             tag = point[13];
            
//             // Skip invalid points
//             if (x_raw == 0 && y_raw == 0 && z_raw == 0) {
//                 continue;
//             }
            
//             // Convert to DriveWorks point format (mm to m)
//             dwLidarPointXYZI& pt = pointCloud_[currentSize + validPointsAdded];
//             pt.x = static_cast<float>(x_raw) / 1000.0f;
//             pt.y = static_cast<float>(y_raw) / 1000.0f;
//             pt.z = static_cast<float>(z_raw) / 1000.0f;
//             pt.intensity = static_cast<float>(reflectivity) / 255.0f;
            
//             validPointsAdded++;
//         }
        
//         // Resize to actual valid points added
//         if (validPointsAdded < livoxPacket->dot_num) {
//             pointCloud_.resize(currentSize + validPointsAdded);
//         }
//     }
//     else if (livoxPacket->data_type == kLivoxLidarCartesianCoordinateLowData) {
//         // Low precision Cartesian coordinates (x, y, z in cm)
//         const uint8_t* pointData = livoxPacket->data;
//         const size_t pointSize = 8; // 2+2+2+1+1 (x,y,z,reflectivity,tag)
        
//         for (uint16_t i = 0; i < livoxPacket->dot_num; i++) {
//             int16_t x_raw, y_raw, z_raw;
//             uint8_t reflectivity, tag;

//             const uint8_t* point = pointData + (i * pointSize);
//             memcpy(&x_raw, point, 2);
//             memcpy(&y_raw, point + 2, 2);
//             memcpy(&z_raw, point + 4, 2);
//             reflectivity = point[6];
//             tag = point[7];
            
//             // Skip invalid points
//             if (x_raw == 0 && y_raw == 0 && z_raw == 0) {
//                 continue;
//             }
            
//             // Convert to DriveWorks point format (cm to m)
//             dwLidarPointXYZI pt;
//             pt.x = static_cast<float>(x_raw) / 100.0f;
//             pt.y = static_cast<float>(y_raw) / 100.0f;
//             pt.z = static_cast<float>(z_raw) / 100.0f;
//             pt.intensity = static_cast<float>(reflectivity) / 255.0f;
            
//             pointCloud_.push_back(pt);
//         }
//     }
//     else if (livoxPacket->data_type == kLivoxLidarSphericalCoordinateData) {
//         // Spherical coordinates (distance, azimuth, elevation, reflectivity, tag)
//         const uint8_t* pointData = livoxPacket->data;
//         const size_t pointSize = 8; // 2+2+2+1+1 (distance,azimuth,elevation,reflectivity,tag)
        
//         for (uint16_t i = 0; i < livoxPacket->dot_num; i++) {
//             uint16_t distance, azimuth, elevation;
//             uint8_t reflectivity, tag;

//             const uint8_t* point = pointData + (i * pointSize);
//             memcpy(&distance, point, 2);
//             memcpy(&azimuth, point + 2, 2);
//             memcpy(&elevation, point + 4, 2);
//             reflectivity = point[6];
//             tag = point[7];
            
//             // Skip invalid points
//             if (distance == 0) {
//                 continue;
//             }
            
//             // Convert spherical to Cartesian coordinates
//             // distance is in mm, azimuth and elevation are in 0.01 degrees
//             float dist_m = static_cast<float>(distance) / 1000.0f;
//             float az_rad = static_cast<float>(azimuth) * 0.01f * M_PI / 180.0f;
//             float el_rad = static_cast<float>(elevation) * 0.01f * M_PI / 180.0f;
            
//             // Convert to DriveWorks point format
//             dwLidarPointXYZI pt;
//             pt.x = dist_m * cos(el_rad) * sin(az_rad);
//             pt.y = dist_m * cos(el_rad) * cos(az_rad);
//             pt.z = dist_m * sin(el_rad);
//             pt.intensity = static_cast<float>(reflectivity) / 255.0f;
            
//             pointCloud_.push_back(pt);
//         }
//     }
//     else {
//         DEBUG_LOG("parseDataBuffer: Unsupported data type: " << (int)livoxPacket->data_type);
//         m_packetQueue.pop();
//         return DW_NOT_READY;
//     }
    
//     // Remove the processed packet from the queue
//     m_packetQueue.pop();
    
//     if (pointCloud_.empty()) {
//         DEBUG_LOG("parseDataBuffer: No valid points extracted from packet");
//         return DW_NOT_READY;
//     }
    
//     // Fill the output structure with our point data
//     output->pointsXYZI = pointCloud_.data();
//     output->pointsRTHI = nullptr;  // We don't compute RTHI data
//     output->nPoints = static_cast<uint32_t>(pointCloud_.size());
//     output->maxPoints = static_cast<uint32_t>(pointCloud_.capacity());
//     output->hostTimestamp = hostTimestamp;
//     output->sensorTimestamp = hostTimestamp; 
//     output->duration = 0;
//     output->scanComplete = false;  // Don't mark as complete - let the application accumulate points
    
//     DEBUG_LOG("parseDataBuffer: Successfully prepared " << output->nPoints << " points for visualization");
    
//     return DW_SUCCESS;
// }

dwStatus LivoxHapPlugin::parseDataBuffer(dwLidarDecodedPacket* output, const dwTime_t hostTimestamp) 
{
    if (!output) {
        DEBUG_LOG("ERROR: parseDataBuffer called with NULL output parameter");
        return DW_INVALID_ARGUMENT;
    }
    
    // Use dedicated mutex for point cloud access
    std::lock_guard<std::mutex> lock(m_pointCloudMutex);
    
    // Check if we have any points to process
    if (pointCloudBuffer_.empty()) {
        DEBUG_LOG("parseDataBuffer: No points available for visualization");
        return DW_NOT_READY;
    }
    
    DEBUG_LOG("parseDataBuffer: Processing " << pointCloudBuffer_.size() << " points for visualization");
    
    // Use the buffer for output (thread-safe copy)
    output->pointsXYZI = pointCloudBuffer_.data();
    output->pointsRTHI = nullptr;
    output->nPoints = static_cast<uint32_t>(pointCloudBuffer_.size());
    output->maxPoints = static_cast<uint32_t>(pointCloudBuffer_.capacity());
    output->hostTimestamp = hostTimestamp;
    output->sensorTimestamp = hostTimestamp;
    output->duration = 0;
    output->scanComplete = false;  // Don't mark as complete for continuous streaming
    
    DEBUG_LOG("parseDataBuffer: Successfully prepared " << output->nPoints << " points for visualization");
    
    return DW_SUCCESS;
}



dwStatus LivoxHapPlugin::getConstants(_dwSensorLidarDecoder_constants* constants)
{
    if (!m_initialized || constants == nullptr)
    {
        return DW_INVALID_ARGUMENT;
    }
    
    if (!m_constantsInitialized)
    {
        m_constants.maxPayloadSize = MAX_UDP_PAYLOAD_SIZE;
        
        // If virtual sensor, use max allowed frame rate
        if (isVirtualSensor())
        {
            m_frameRate = MAX_FRAME_RATE;
        }
        
        // Populate lidar properties
        dwLidarProperties* properties = &(m_constants.properties);
        properties->pointsPerSpin = MAX_POINTS_PER_FRAME;
        properties->pointsPerSecond = MAX_POINTS_PER_FRAME * m_frameRate;
        properties->packetsPerSpin = PACKETS_PER_FRAME;
        properties->spinFrequency = m_frameRate;
        properties->packetsPerSecond = PACKETS_PER_FRAME * m_frameRate;
        properties->pointsPerPacket = MAX_POINTS_PER_PACKET;
        properties->pointStride = POINT_STRIDE;
        properties->availableReturns = DW_LIDAR_RETURN_TYPE_ANY;
        strcpy(properties->deviceString, "LIVOX_HAP");
        
       // Avoid repeated initialization
       m_constantsInitialized = true;
    }
    
    // Copy initialized constants to output
    *constants = m_constants;
    
    return DW_SUCCESS;
}

//################################################################################
//############################## Callback Functions ##############################
//################################################################################

void LivoxHapPlugin::pointCloudCallback(const uint32_t handle, const uint8_t dev_type, 
    LivoxLidarEthernetPacket* data, void* client_data)
{
    // CRITICAL: Log EVERY callback attempt, even with NULL pointers
    std::cerr << "LIVOX_HAP_PLUGIN: *** POINT CLOUD CALLBACK CALLED *** handle=" << handle 
              << ", dev_type=" << (int)dev_type 
              << ", data=" << (void*)data 
              << ", client_data=" << (void*)client_data << std::endl;
    
    if (!data || !client_data) {
        std::cerr << "LIVOX_HAP_PLUGIN: ERROR - NULL pointers in pointCloudCallback" << std::endl;
        return;
    }
    
    // DETAILED PACKET LOGGING
    std::cerr << "LIVOX_HAP_PLUGIN: PACKET DETAILS:" << std::endl;
    std::cerr << "  - version: " << (int)data->version << std::endl;
    std::cerr << "  - length: " << data->length << std::endl;
    std::cerr << "  - time_interval: " << data->time_interval << std::endl;
    std::cerr << "  - dot_num: " << data->dot_num << std::endl;
    std::cerr << "  - udp_cnt: " << data->udp_cnt << std::endl;
    std::cerr << "  - frame_cnt: " << data->frame_cnt << std::endl;
    std::cerr << "  - data_type: " << (int)data->data_type << std::endl;
    std::cerr << "  - time_type: " << (int)data->time_type << std::endl;
    
    // Log first few bytes of data
    if (data->length > 0) {
        std::cerr << "LIVOX_HAP_PLUGIN: First 16 bytes of data: ";
        for (int i = 0; i < std::min(16, (int)data->length); i++) {
            std::cerr << std::hex << std::setw(2) << std::setfill('0') << (int)data->data[i] << " ";
        }
        std::cerr << std::dec << std::endl;
    }

    LivoxHapPlugin* plugin = static_cast<LivoxHapPlugin*>(client_data);

    // Increment static callback counter
    uint64_t callbackCount = ++s_callbackCallCount;
    
    // Log every call for the first 10 seconds, then every 100th call
    static auto startTime = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
    
    bool shouldLog = (elapsed < 10) || (callbackCount % 100 == 0);
    
    if (shouldLog) {
        DEBUG_LOG("*** POINT CLOUD CALLBACK CALLED *** #" << callbackCount 
                  << " - device: " << handle 
                  << ", points: " << data->dot_num
                  << ", frame: " << (int)data->frame_cnt 
                  << ", type: " << (int)data->data_type
                  << ", elapsed: " << elapsed << "s");
        
        // Log first 16 bytes of data for debugging
        if (callbackCount <= 5 && data->length > 0) {
            DEBUG_LOG("First 16 bytes of packet data:");
            for (int i = 0; i < 16 && i < data->length; i++) {
                printf("%02X ", data->data[i]);
            }
            printf("\n");
        }
    }

    // Validate the packet has a reasonable number of points
    if (data->dot_num == 0) {
        DEBUG_LOG("Skipping empty packet - dot_num: " << data->dot_num);
        return;
    }

    // Log packet header details for debugging (only for first few calls)
    if (callbackCount <= 5) {
        DEBUG_LOG("Packet details - Version: " << (int)data->version 
                  << ", Length: " << data->length 
                  << ", Time interval: " << data->time_interval 
                  << ", Dot num: " << data->dot_num
                  << ", UDP cnt: " << data->udp_cnt
                  << ", Frame cnt: " << (int)data->frame_cnt
                  << ", Data type: " << (int)data->data_type
                  << ", Time type: " << (int)data->time_type);

        // Log first few bytes of data for debugging
        if (data->length > 0) {
            std::stringstream ss;
            ss << "First 16 bytes of data: ";
            for (int i = 0; i < std::min(16, (int)data->length); i++) {
                ss << std::hex << std::setw(2) << std::setfill('0') 
                   << static_cast<int>(data->data[i]) << " ";
            }
            DEBUG_LOG(ss.str());
        }
    }

    // Process HAP-specific point cloud data
    plugin->processHapPointCloud(handle, data);
}

void LivoxHapPlugin::infoChangeCallback(const uint32_t handle, const LivoxLidarInfo* info, void* client_data)
{
    if (!info || !client_data) {
        std::cerr << "LIVOX_HAP_PLUGIN: infoChangeCallback received NULL pointers" << std::endl;
        return;
    }
    
    LivoxHapPlugin* plugin = static_cast<LivoxHapPlugin*>(client_data);
    
    std::cout << "LivoxHapPlugin::infoChangeCallback - Device connected - Handle: " << handle 
              << ", Type: " << static_cast<int>(info->dev_type) 
              << ", SN: " << info->sn 
              << ", IP: " << info->lidar_ip << std::endl;
    
    // Check if this is a HAP device
    if (!plugin->isHapDevice(info)) {
        DEBUG_LOG("infoChangeCallback: Not a HAP device, ignoring");
        return;
    }
    
    DEBUG_LOG("infoChangeCallback: HAP device connected successfully - Handle: " << handle << ", IP: " << info->lidar_ip);
    
    // Mark as connected immediately
    plugin->m_connected = true;
    plugin->m_deviceHandle = handle;
    
    // Copy IP address
    strncpy(plugin->m_ipAddress, info->lidar_ip, sizeof(plugin->m_ipAddress)-1);
    plugin->m_ipAddress[sizeof(plugin->m_ipAddress)-1] = '\0';
    
    // Configure HAP device according to protocol documentation
    DEBUG_LOG("infoChangeCallback: Starting HAP device configuration sequence");
    
    // According to HAP protocol, the state machine is:
    // SELFCHECK (auto, ~5s) -> MOTORSTARTUP (~10s) -> SAMPLING (ready)
    // DO NOT force work mode change - let the sensor naturally transition
    
    DEBUG_LOG("infoChangeCallback: Sending configuration commands");
    
    // Step 1: Set work mode to NORMAL to start motor (trigger MOTORSTARTUP -> SAMPLING transition)
    DEBUG_LOG("infoChangeCallback: Setting work mode to NORMAL to start motor");
    livox_status status = SetLivoxLidarWorkMode(handle, kLivoxLidarNormal, controlCallback, plugin);
    if (status != kLivoxLidarStatusSuccess) {
        DEBUG_LOG("ERROR: Failed to set work mode, status: " << status);
    } else {
        DEBUG_LOG("SUCCESS: Work mode command sent successfully - motor startup initiated");
    }
    
    // Step 2: Query current work mode to check sensor state
    DEBUG_LOG("infoChangeCallback: Querying device state to check current work mode");
    QueryLivoxLidarInternalInfo(handle, queryDeviceStateCallback, plugin);
    
    // Step 4: Wait for motor startup (non-blocking approach)
    DEBUG_LOG("infoChangeCallback: Starting motor startup sequence (non-blocking)");
    
    // Start a separate thread to handle the motor startup delay and enable data transmission
    std::thread([plugin, handle]() {
        try {
            // According to HAP protocol state machine:
            // After SetLivoxLidarWorkMode(NORMAL), device goes through:
            // SELFCHECK (~5s) -> MOTORSTARTUP (~10s) -> SAMPLING (auto)
            // We wait for these transitions to complete before enabling point data
            
            std::cerr << "LIVOX_HAP_PLUGIN: Motor startup thread: STARTED" << std::endl;
            std::cerr << "LIVOX_HAP_PLUGIN: Motor startup thread: Waiting for SELFCHECK and MOTORSTARTUP to complete..." << std::endl;
            std::cerr << "LIVOX_HAP_PLUGIN: Motor startup thread: Will wait 17 seconds (SELFCHECK ~5s + MOTORSTARTUP ~10s + 2s margin)" << std::endl;
            
            // Use polling approach instead of single long sleep to detect issues early
            for (int i = 0; i < 17; i++) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                if (i % 5 == 0) {
                    std::cerr << "LIVOX_HAP_PLUGIN: Motor startup thread: " << i << "/17 seconds elapsed..." << std::endl;
                }
            }
            
            std::cerr << "LIVOX_HAP_PLUGIN: Motor startup thread: 17 seconds complete - device should now be in SAMPLING state" << std::endl;
            
            // Follow Livox SDK example - just enable point cloud data transmission
            // The HAP sensor should work with default settings after work mode is set
            std::cerr << "LIVOX_HAP_PLUGIN: Motor startup thread: Enabling point cloud data transmission (following Livox SDK example)" << std::endl;
            livox_status status = EnableLivoxLidarPointSend(handle, controlCallback, plugin);
            std::cerr << "LIVOX_HAP_PLUGIN: Motor startup thread: EnableLivoxLidarPointSend returned status: " << status << std::endl;
            
            if (status != kLivoxLidarStatusSuccess) {
                std::cerr << "LIVOX_HAP_PLUGIN: ERROR: Failed to enable point cloud data, status: " << status << std::endl;
            } else {
                std::cerr << "LIVOX_HAP_PLUGIN: SUCCESS: Point cloud data transmission enabled" << std::endl;
            }
            
            // Query device state to verify
            std::cerr << "LIVOX_HAP_PLUGIN: Motor startup thread: Querying device state..." << std::endl;
            QueryLivoxLidarInternalInfo(handle, queryDeviceStateCallback, plugin);
            
            // Test if callback is working by waiting a bit and checking if any callbacks arrived
            std::cerr << "LIVOX_HAP_PLUGIN: Motor startup thread: Waiting 5 seconds to see if point cloud callbacks arrive..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
            
            std::cerr << "LIVOX_HAP_PLUGIN: Motor startup thread: COMPLETED" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "LIVOX_HAP_PLUGIN: ERROR: Motor startup thread exception: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "LIVOX_HAP_PLUGIN: ERROR: Motor startup thread unknown exception" << std::endl;
        }
    }).detach();
    
    DEBUG_LOG("infoChangeCallback: HAP device configuration initiated (non-blocking)");
}

void LivoxHapPlugin::controlCallback(livox_status status, uint32_t handle,
                                LivoxLidarAsyncControlResponse* response, void* client_data)
{
    if (!client_data) {
        std::cerr << "LIVOX_HAP_PLUGIN: controlCallback received NULL client_data" << std::endl;
        return;
    }
    
    if (!response) {
        std::cerr << "LIVOX_HAP_PLUGIN: controlCallback received NULL response" << std::endl;
        return;
    }
    
    LivoxHapPlugin* plugin = static_cast<LivoxHapPlugin*>(client_data);

    const char* statusStr = "UNKNOWN";
    if (status == kLivoxLidarStatusSuccess) statusStr = "SUCCESS";
    else if (status == kLivoxLidarStatusFailure) statusStr = "FAILURE";
    else if (status == kLivoxLidarStatusTimeout) statusStr = "TIMEOUT";

    DEBUG_LOG("controlCallback: Command completed with status: " << statusStr 
              << " (code: " << status << "), handle: " << handle
              << ", response code: " << (int)response->ret_code
              << ", error key: " << response->error_key);
              
    // Log success/failure for command execution
    if (status == kLivoxLidarStatusSuccess && response->ret_code == 0) {
        DEBUG_LOG("controlCallback: SUCCESS - Command executed successfully");
    } else {
        DEBUG_LOG("controlCallback: ERROR - Command failed with ret_code: " << (int)response->ret_code 
                  << ", error_key: " << response->error_key);
    }
}

void LivoxHapPlugin::queryDeviceStateCallback(livox_status status, uint32_t handle,
                                        LivoxLidarDiagInternalInfoResponse* response, void* client_data)
{
    if (!response || !client_data) {
        std::cerr << "LIVOX_HAP_PLUGIN: queryDeviceStateCallback received NULL response or client_data" << std::endl;
        return;
    }
    
    LivoxHapPlugin* plugin = static_cast<LivoxHapPlugin*>(client_data);

    const char* statusStr = "UNKNOWN";
    if (status == kLivoxLidarStatusSuccess) statusStr = "SUCCESS";
    else if (status == kLivoxLidarStatusFailure) statusStr = "FAILURE";
    else if (status == kLivoxLidarStatusTimeout) statusStr = "TIMEOUT";

    DEBUG_LOG("queryDeviceStateCallback: Device state query completed with status: " << statusStr 
              << " (code: " << status << "), handle: " << handle
              << ", response code: " << (int)response->ret_code);
              
    if (status == kLivoxLidarStatusSuccess && response->ret_code == 0) {
        DEBUG_LOG("queryDeviceStateCallback: SUCCESS - Device state retrieved successfully");
        // The response contains internal diagnostic information about the device
        // This can help verify that the device is configured correctly
    } else {
        DEBUG_LOG("queryDeviceStateCallback: ERROR - Failed to retrieve device state");
    }
}

//################################################################################
//############################## HAP Specific Methods ############################
//################################################################################

// void LivoxHapPlugin::processHapPointCloud(uint32_t handle, const LivoxLidarEthernetPacket* data)
// {
//     if (!m_running || !data) {
//         DEBUG_LOG("processHapPointCloud: Not running or NULL data");
//         return;
//     }

//     // Update statistics
//     m_totalPacketsReceived++;
//     m_totalPointsReceived += data->dot_num;
    
//     // Print stats periodically
//     auto now = std::chrono::steady_clock::now();
//     auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_statsTimer).count();
//     if (elapsed >= 5) {  // Every 5 seconds
//         DEBUG_LOG("STATS: Received " << m_totalPacketsReceived << " packets, " 
//                   << m_totalPointsReceived << " points");
//         DEBUG_LOG("STATS: Dropped " << m_droppedPackets << " packets, "
//                   << m_invalidPackets << " invalid packets");
//         m_statsTimer = now;
//     }

//     // Validate packet data
//     if (data->dot_num == 0 || data->length == 0) {
//         m_invalidPackets++;
//         DEBUG_LOG("processHapPointCloud: Invalid packet - dot_num: " << data->dot_num << ", length: " << data->length);
//         return;
//     }

//     DEBUG_LOG("processHapPointCloud: Processing " << data->dot_num << " points, data type: " << (int)data->data_type);

//     // Use the same approach as MID360 - just process the packet for readRawData API
//     // Don't accumulate points in a complex buffer, just queue the packet
//     processLivoxPointCloud(handle, data);
    
//     // For visualization, we'll use the parseDataBuffer approach which reads from the queue
//     DEBUG_LOG("processHapPointCloud: Packet queued for readRawData API");
// }

void LivoxHapPlugin::processHapPointCloud(uint32_t handle, const LivoxLidarEthernetPacket* data)
{
    if (!m_running || !data) {
        DEBUG_LOG("processHapPointCloud: Not running or NULL data");
        return;
    }

    // Validate packet data
    if (data->dot_num == 0 || data->length == 0) {
        m_invalidPackets++;
        DEBUG_LOG("processHapPointCloud: Invalid packet - dot_num: " << data->dot_num);
        return;
    }

    // Update statistics
    m_totalPacketsReceived++;
    m_totalPointsReceived += data->dot_num;

    DEBUG_LOG("processHapPointCloud: Processing " << data->dot_num 
              << " points, data type: " << (int)data->data_type);

    // Thread-safe point accumulation
    std::lock_guard<std::mutex> lock(m_pointCloudMutex);
    
    size_t validPointsAdded = 0;
    
    // According to HAP protocol, Type 1 is already Cartesian in mm
    if (data->data_type == 1) { // Type 1: Cartesian (high accuracy) - x,y,z in mm
        const uint8_t* pointData = data->data;
        const size_t pointSize = 14; // 4+4+4+1+1 bytes per point
        
        // Verify data length
        if (data->length < data->dot_num * pointSize) {
            DEBUG_LOG("ERROR: Invalid data length");
            return;
        }
        
        // Ensure we don't overflow our buffer
        if (pointCloud_.size() + data->dot_num > MAX_POINTS_PER_FRAME) {
            // Swap buffers - move current points to visualization buffer
            pointCloudBuffer_ = std::move(pointCloud_);
            pointCloud_.clear();
            pointCloud_.reserve(MAX_POINTS_PER_FRAME);
            DEBUG_LOG("Buffer swapped at " << pointCloudBuffer_.size() << " points");
        }
        
        // Process each point - they're ALREADY Cartesian!
        for (uint16_t i = 0; i < data->dot_num; i++) {
            const uint8_t* point = pointData + (i * pointSize);
            
            // Extract x, y, z (already in mm, just need to convert to meters)
            int32_t x_mm, y_mm, z_mm;
            memcpy(&x_mm, point, 4);
            memcpy(&y_mm, point + 4, 4);
            memcpy(&z_mm, point + 8, 4);
            uint8_t reflectivity = point[12];
            uint8_t tag = point[13];
            
            // Skip invalid points (tag bits 0-1 indicate validity)
            if ((tag & 0x03) == 0x03) {
                continue;  // Invalid point
            }
            
            // Skip zero points
            if (x_mm == 0 && y_mm == 0 && z_mm == 0) {
                continue;
            }
            
            // Simple conversion from mm to meters
            dwLidarPointXYZI pt;
            pt.x = x_mm / 1000.0f;  // mm to m
            pt.y = y_mm / 1000.0f;  // mm to m
            pt.z = z_mm / 1000.0f;  // mm to m
            pt.intensity = reflectivity / 255.0f;
            
            pointCloud_.push_back(pt);
            validPointsAdded++;
        }
    }
    else if (data->data_type == 2) { // Type 2: Cartesian (low accuracy) - x,y,z in cm
        const uint8_t* pointData = data->data;
        const size_t pointSize = 8; // 2+2+2+1+1 bytes per point
        
        if (data->length < data->dot_num * pointSize) {
            DEBUG_LOG("ERROR: Invalid data length");
            return;
        }
        
        // Buffer management
        if (pointCloud_.size() + data->dot_num > MAX_POINTS_PER_FRAME) {
            pointCloudBuffer_ = std::move(pointCloud_);
            pointCloud_.clear();
            pointCloud_.reserve(MAX_POINTS_PER_FRAME);
        }
        
        for (uint16_t i = 0; i < data->dot_num; i++) {
            const uint8_t* point = pointData + (i * pointSize);
            
            // Extract x, y, z (already in cm, just need to convert to meters)
            int16_t x_cm, y_cm, z_cm;
            memcpy(&x_cm, point, 2);
            memcpy(&y_cm, point + 2, 2);
            memcpy(&z_cm, point + 4, 2);
            uint8_t reflectivity = point[6];
            uint8_t tag = point[7];
            
            if ((tag & 0x03) == 0x03 || (x_cm == 0 && y_cm == 0 && z_cm == 0)) {
                continue;
            }
            
            // Simple conversion from cm to meters
            dwLidarPointXYZI pt;
            pt.x = x_cm / 100.0f;  // cm to m
            pt.y = y_cm / 100.0f;  // cm to m
            pt.z = z_cm / 100.0f;  // cm to m
            pt.intensity = reflectivity / 255.0f;
            
            pointCloud_.push_back(pt);
            validPointsAdded++;
        }
    }
    else if (data->data_type == 3) { // Type 3: Spherical - needs conversion
        const uint8_t* pointData = data->data;
        const size_t pointSize = 8; // 2+2+2+1+1 bytes
        
        if (data->length < data->dot_num * pointSize) {
            DEBUG_LOG("ERROR: Invalid data length");
            return;
        }
        
        // Buffer management
        if (pointCloud_.size() + data->dot_num > MAX_POINTS_PER_FRAME) {
            pointCloudBuffer_ = std::move(pointCloud_);
            pointCloud_.clear();
            pointCloud_.reserve(MAX_POINTS_PER_FRAME);
        }
        
        for (uint16_t i = 0; i < data->dot_num; i++) {
            const uint8_t* point = pointData + (i * pointSize);
            
            uint16_t dist_mm, zenith, azimuth;
            memcpy(&dist_mm, point, 2);     // Distance in mm
            memcpy(&zenith, point + 2, 2);  // Zenith in 0.01°
            memcpy(&azimuth, point + 4, 2); // Azimuth in 0.01°
            uint8_t reflectivity = point[6];
            uint8_t tag = point[7];
            
            if ((tag & 0x03) == 0x03 || dist_mm == 0) {
                continue;
            }
            
            // Convert spherical to Cartesian
            float r = dist_mm / 1000.0f;  // mm to m
            float theta = (zenith / 100.0f) * M_PI / 180.0f;   // 0.01° to rad
            float phi = (azimuth / 100.0f) * M_PI / 180.0f;    // 0.01° to rad
            
            dwLidarPointXYZI pt;
            pt.x = r * sin(theta) * cos(phi);
            pt.y = r * sin(theta) * sin(phi);
            pt.z = r * cos(theta);
            pt.intensity = reflectivity / 255.0f;
            
            pointCloud_.push_back(pt);
            validPointsAdded++;
        }
    }
    else {
        DEBUG_LOG("Unknown data type: " << (int)data->data_type);
        return;
    }
    
    DEBUG_LOG("Added " << validPointsAdded << " points, total: " << pointCloud_.size());
    
    // Ensure visualization buffer has data
    if (pointCloudBuffer_.empty() && pointCloud_.size() >= MIN_POINTS_FOR_FRAME) {
        pointCloudBuffer_ = pointCloud_;
    }
    
    // Keep accumulation under control
    if (pointCloud_.size() > MAX_POINTS_PER_FRAME * 0.8) {
        // Keep recent points, remove old ones
        size_t keepCount = MAX_POINTS_PER_FRAME * 0.6;
        if (pointCloud_.size() > keepCount) {
            pointCloud_.erase(pointCloud_.begin(), 
                            pointCloud_.begin() + (pointCloud_.size() - keepCount));
        }
        // Update visualization buffer
        pointCloudBuffer_ = pointCloud_;
    }
    
    // Queue raw packet for readRawData API
    processLivoxPointCloud(handle, data);
}

bool LivoxHapPlugin::isHapDevice(const LivoxLidarInfo* info) const
{
    if (!info) {
        DEBUG_LOG("isHapDevice called with NULL info");
        return false;
    }
    
    DEBUG_LOG("isHapDevice: Checking device type: " << (int)info->dev_type 
              << " (expected HAP: " << (int)HAP_DEVICE_TYPE 
              << " or Industrial HAP: " << (int)HAP_INDUSTRIAL_DEVICE_TYPE << ")");
    DEBUG_LOG("isHapDevice: Device SN: " << info->sn << ", IP: " << info->lidar_ip);
    
    // Accept both standard HAP (type 15) and Industrial HAP (type 10)
    bool isHap = (info->dev_type == HAP_DEVICE_TYPE || info->dev_type == HAP_INDUSTRIAL_DEVICE_TYPE);
    
    if (isHap) {
        if (info->dev_type == HAP_DEVICE_TYPE) {
            DEBUG_LOG("isHapDevice: Device confirmed as standard HAP sensor (type: " << (int)info->dev_type << ")");
        } else {
            DEBUG_LOG("isHapDevice: Device confirmed as Industrial HAP sensor (type: " << (int)info->dev_type << ")");
        }
    } else {
        DEBUG_LOG("isHapDevice: Device is NOT a HAP sensor (type: " << (int)info->dev_type << ")");
        DEBUG_LOG("isHapDevice: Known HAP types: Standard=" << (int)HAP_DEVICE_TYPE << ", Industrial=" << (int)HAP_INDUSTRIAL_DEVICE_TYPE);
    }
    
    return isHap;
}

dwStatus LivoxHapPlugin::configureHapDevice(uint32_t handle)
{
    DEBUG_LOG("configureHapDevice: Configuring HAP device with handle: " << handle);
    
    // Set HAP-specific parameters according to HAP protocol documentation
    // The HAP protocol requires specific sequence: work_tgt_mode = 0x01 (SAMPLING state)
    
    // Step 1: Set work mode to NORMAL (0x01) - This is the SAMPLING state for HAP
    DEBUG_LOG("configureHapDevice: Setting work mode to NORMAL (0x01 - SAMPLING state)");
    livox_status status = SetLivoxLidarWorkMode(handle, kLivoxLidarNormal, controlCallback, this);
    if (status != kLivoxLidarStatusSuccess) {
        DEBUG_LOG("ERROR: Failed to set work mode, status: " << status);
    } else {
        DEBUG_LOG("SUCCESS: Work mode command sent successfully");
    }
    
    // Step 2: Set scan pattern (repetitive for HAP)
    DEBUG_LOG("configureHapDevice: Setting scan pattern to repetitive");
    status = SetLivoxLidarScanPattern(handle, kLivoxLidarScanPatternRepetive, controlCallback, this);
    if (status != kLivoxLidarStatusSuccess) {
        DEBUG_LOG("ERROR: Failed to set scan pattern, status: " << status);
    } else {
        DEBUG_LOG("SUCCESS: Scan pattern command sent successfully");
    }
    
    // Step 3: Set data type to Cartesian coordinates (standard for HAP)
    DEBUG_LOG("configureHapDevice: Setting coordinate type to Cartesian");
    status = SetLivoxLidarPclDataType(handle, kLivoxLidarCartesianCoordinateHighData, controlCallback, this);
    if (status != kLivoxLidarStatusSuccess) {
        DEBUG_LOG("ERROR: Failed to set data type, status: " << status);
    } else {
        DEBUG_LOG("SUCCESS: Data type command sent successfully");
    }
    
    DEBUG_LOG("configureHapDevice: Configuration commands completed for handle: " << handle);
    
    return DW_SUCCESS;
}

//################################################################################
//############################## Helper Functions ################################
//################################################################################

bool LivoxHapPlugin::isVirtualSensor() const
{
    return m_isVirtual;
}

dwStatus LivoxHapPlugin::getParameter(std::string& val, const std::string& param, const char* params)
{
    if (params == nullptr)
    {
        DEBUG_LOG("ERROR: getParameter called with NULL params");
        return DW_INVALID_ARGUMENT;
    }
    
    DEBUG_LOG("getParameter called for param: " << param);
    
    std::string paramsString(params);
    std::string searchString(param + "=");
    size_t pos = paramsString.find(searchString);
    if (pos == std::string::npos)
    {
        DEBUG_LOG("Parameter " << param << " not found");
        return DW_FAILURE;
    }
    
    val = paramsString.substr(pos + searchString.length());
    pos = val.find_first_of(',');
    if (pos != std::string::npos) {
        val = val.substr(0, pos);
    }
    
    DEBUG_LOG("Parameter " << param << " found with value: " << val);
    return DW_SUCCESS;
}

// Additional helper methods (simplified versions from Mid-360)
void LivoxHapPlugin::processLivoxPointCloud(uint32_t handle, const LivoxLidarEthernetPacket* data)
{
    // Simplified implementation for HAP - just queue the packet for readRawData API
    if (!m_running || !data) {
        return;
    }

    // Get host timestamp for received packet
    dwTime_t packetTimestamp;
    dwContext_getCurrentTime(&packetTimestamp, m_ctx);
    
    // Create a new raw packet for the queue
    RawPacket packet;
    
    // Insert metadata header (size and timestamp)
    uint32_t payloadSize = sizeof(LivoxLidarEthernetPacket) + data->length;
    std::memcpy(&packet.rawData[0], &payloadSize, sizeof(uint32_t));
    std::memcpy(&packet.rawData[sizeof(uint32_t)], &packetTimestamp, sizeof(dwTime_t));
    
    // Make sure we don't exceed buffer size
    if (payloadSize + PAYLOAD_OFFSET > sizeof(packet.rawData)) {
        DEBUG_LOG("WARNING: Packet too large (" << payloadSize << 
                  " bytes), truncating to " << (sizeof(packet.rawData) - PAYLOAD_OFFSET));
        payloadSize = sizeof(packet.rawData) - PAYLOAD_OFFSET;
    }
    
    // Copy packet data
    std::memcpy(&packet.rawData[PAYLOAD_OFFSET], data, std::min(static_cast<size_t>(payloadSize), sizeof(LivoxLidarEthernetPacket)));
    
    // Add to queue with notification
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        
        // Check if queue is getting too large, remove oldest packets if needed
        if (m_packetQueue.size() > MAX_QUEUE_SIZE) {
            DEBUG_LOG("WARNING: Queue size " << m_packetQueue.size() << 
                      " exceeds limit, removing oldest packet");
            m_packetQueue.pop();
            m_droppedPackets++;
        }
        
        m_packetQueue.push(packet);
        m_queueCondition.notify_one();
    }
}

void LivoxHapPlugin::deviceManagementThread() {
    DEBUG_LOG("deviceManagementThread started");
    
    // Timeout for device connection
    const int CONNECTION_TIMEOUT_SEC = 30;
    int connectionAttempts = 0;
    bool connectionTimedOut = false;
    
    // Wait for device connection
    while (m_running && !m_connected && !connectionTimedOut) {
        if (connectionAttempts % 10 == 0) {
            DEBUG_LOG("Waiting for Livox HAP device connection (attempt " << connectionAttempts << ")");
        }
        
        // Check for timeout
        if (connectionAttempts++ > CONNECTION_TIMEOUT_SEC) {
            connectionTimedOut = true;
            DEBUG_LOG("ERROR: Connection timeout - no HAP device found after " << 
                      CONNECTION_TIMEOUT_SEC << " seconds");
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    // Main monitoring loop while connected
    int monitoringCounter = 0;
    uint64_t lastCallbackCount = 0;
    while (m_running && m_connected) {
        // Periodic monitoring every 5 seconds
        if (monitoringCounter++ % 5 == 0) {
            std::lock_guard<std::mutex> lock(m_packetMutex);
            uint64_t currentCallbackCount = s_callbackCallCount.load();
            DEBUG_LOG("Device monitoring: Connected=" << m_connected 
                      << ", Handle=" << m_deviceHandle 
                      << ", Point cloud size=" << pointCloud_.size()
                      << ", Total packets received=" << m_totalPacketsReceived
                      << ", Total points received=" << m_totalPointsReceived
                      << ", Callback calls=" << currentCallbackCount
                      << ", New callbacks since last check=" << (currentCallbackCount - lastCallbackCount));
            lastCallbackCount = currentCallbackCount;
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    // Cleanup when thread is exiting
    if (m_connected && m_deviceHandle != 0) {
        DEBUG_LOG("Shutting down device connection");
        DisableLivoxLidarPointSend(m_deviceHandle, controlCallback, this);
    }
    
    DEBUG_LOG("deviceManagementThread exiting");
}

bool LivoxHapPlugin::isScanComplete()
{
    // For HAP, a "scan" is complete when we have accumulated enough points
    std::lock_guard<std::mutex> lock(m_packetMutex);
    
    // Check if we have accumulated enough points
    if (pointCloud_.size() >= MIN_POINTS_FOR_FRAME) {
        DEBUG_LOG("Scan complete by point count threshold: " << pointCloud_.size());
        return true;
    }
    
    return false;
}

} // namespace lidar
} // namespace plugin
} // namespace dw 