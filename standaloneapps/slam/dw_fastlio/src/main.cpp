/////////////////////////////////////////////////////////////////////////////////////////
// DriveWorks SLAM Sample - Fast-LIO with DriveWorks Sensors
// 
// This sample demonstrates:
// 1. Building a map using Fast-LIO with DriveWorks sensors
// 2. Localizing within a pre-built map
// 3. DriveWorks-native visualization
/////////////////////////////////////////////////////////////////////////////////////////

#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <chrono>

#include <thread>

// DriveWorks
#include <dw/core/base/Version.h>
#include <dw/core/base/Status.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/lidar/Lidar.h>
#include <dw/sensors/imu/IMU.h>
#include <dw/sensors/gps/GPS.h>
#include <dw/rig/Rig.h>
#include <dw/sensors/sensormanager/SensorManager.h>

// DriveWorks Visualization
#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Visualization.h>
#include <framework/DriveWorksSample.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/MouseView3D.hpp>
#include <framework/Checks.hpp>

// GLFW for key constants
#include <GLFW/glfw3.h>

// Standard library
#include <cmath>
#include <ctime>
#include <algorithm>

// Our SLAM implementation
#include "DWFastLIO.hpp"

using namespace dw_samples::common;

//------------------------------------------------------------------------------
// Global variables
//------------------------------------------------------------------------------
static bool gRun = true;

//------------------------------------------------------------------------------
void sig_int_handler(int) {
    gRun = false;
}

//------------------------------------------------------------------------------
// DriveWorks SLAM Sample Class
//------------------------------------------------------------------------------
class DWFastLIOSample : public DriveWorksSample {
private:
    // DriveWorks SDK
    dwContextHandle_t m_context = DW_NULL_HANDLE;
    dwSALHandle_t m_sal = DW_NULL_HANDLE;
    
    // Sensors
    dwSensorHandle_t m_lidarSensor = DW_NULL_HANDLE;
    dwSensorHandle_t m_imuSensor = DW_NULL_HANDLE;
    dwSensorHandle_t m_gpsSensor = DW_NULL_HANDLE;
    
    dwLidarProperties m_lidarProperties{};
    
    // Rig configuration
    dwRigHandle_t m_rigConfig = DW_NULL_HANDLE;
    dwSensorManagerHandle_t m_sensorManager = DW_NULL_HANDLE;
    
    // Visualization
    dwVisualizationContextHandle_t m_visualizationContext = DW_NULL_HANDLE;
    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
    uint32_t m_trajectoryBuffer = 0;
    uint32_t m_mapBuffer = 0;
    uint32_t m_currentScanBuffer = 0;
    
    // SLAM
    std::unique_ptr<dw_slam::DWFastLIO> m_slam;
    dw_slam::DWFastLIOConfig m_slamConfig;
    
    // Temporary point cloud storage
    std::unique_ptr<dwLidarPointXYZI[]> m_pointCloudBuffer;
    size_t m_pointCloudSize = 0;
    size_t m_pointCloudCapacity = 0;
    
    // IMU data
    dwIMUFrame m_latestIMU{};
    bool m_hasIMU = false;
    
    // Messages
    std::string m_statusMessage;
    bool m_mappingMode = true;

public:
    DWFastLIOSample(const ProgramArguments& args)
        : DriveWorksSample(args) {}
    
    bool onInitialize() override {
        // Initialize logger - reduce noise from DriveWorks internal warnings
        CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
        CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_WARN)); // Changed from VERBOSE to WARN to reduce noise
        
        // Initialize SDK context
        dwContextParameters sdkParams = {};
        CHECK_DW_ERROR(dwInitialize(&m_context, DW_VERSION, &sdkParams));
        CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_context));
        
        // Initialize sensors
        if (!initializeSensors()) {
            return false;
        }
        
        // Initialize SLAM
        if (!initializeSLAM()) {
            return false;
        }
        
        // Initialize visualization
        if (!initializeVisualization()) {
            return false;
        }
        
        // Set process rate to 5-6 FPS to prevent blocking and ensure smooth operation
        setProcessRate(6);  // 6 FPS for processing
        setRenderRate(30);  // 30 FPS for rendering (smooth visualization)
        
        std::cout << "[DWFastLIO] System initialized successfully" << std::endl;
        if (m_gpsSensor == DW_NULL_HANDLE) {
            std::cout << "[DWFastLIO] GPS not available (no GPS in rig, or --gps-params/--gps-protocol not set); RTK will not be fed to Fast-LIO" << std::endl;
        } else {
            std::cout << "[DWFastLIO] GPS available; RTK will be fed to Fast-LIO when valid" << std::endl;
        }
        std::cout << "[DWFastLIO] Mode: MAPPING" << std::endl;
        std::cout << "[DWFastLIO] Process rate: 6 FPS, Render rate: 30 FPS" << std::endl;
        std::cout << "[DWFastLIO] Press 'M' to toggle MAPPING/LOCALIZATION mode" << std::endl;
        std::cout << "[DWFastLIO] Press 'S' to save map" << std::endl;
        std::cout << "[DWFastLIO] Press 'L' to load map" << std::endl;
        
        return true;
    }
    
    bool initializeSensors() {
        // Check if rig file is provided
        std::string rigFile = getArgument("rig-file");
        
        if (!rigFile.empty()) {
            // Initialize from rig file
            return initializeSensorsFromRig(rigFile);
        } else {
            // Fall back to command-line parameters
            return initializeSensorsFromParams();
        }
    }
    
    bool initializeSensorsFromRig(const std::string& rigFile) {
        // Initialize rig configuration
        dwStatus status = dwRig_initializeFromFile(&m_rigConfig, m_context, rigFile.c_str());
        if (status != DW_SUCCESS) {
            std::cerr << "[DWFastLIO] ERROR: Failed to load rig file: " << rigFile 
                      << " - " << dwGetStatusName(status) << std::endl;
            return false;
        }
        
        std::cout << "[DWFastLIO] Loaded rig file: " << rigFile << std::endl;
        
        // Initialize sensor manager from rig
        status = dwSensorManager_initializeFromRig(&m_sensorManager, m_rigConfig,
                                                    DW_SENSORMANGER_MAX_NUM_SENSORS, m_sal);
        if (status != DW_SUCCESS) {
            std::cerr << "[DWFastLIO] ERROR: Failed to initialize sensor manager from rig: " 
                      << dwGetStatusName(status) << std::endl;
            return false;
        }
        
        // Start sensor manager
        CHECK_DW_ERROR(dwSensorManager_start(m_sensorManager));
        
        // Get LiDAR sensor
        uint32_t lidarCount = 0;
        CHECK_DW_ERROR(dwSensorManager_getNumSensors(&lidarCount, DW_SENSOR_LIDAR, m_sensorManager));
        
        if (lidarCount == 0) {
            std::cerr << "[DWFastLIO] ERROR: No LiDAR sensors found in rig file!" << std::endl;
            return false;
        }
        
        if (lidarCount > 1) {
            std::cout << "[DWFastLIO] WARNING: Multiple LiDAR sensors found, using first one" << std::endl;
        }
        
        uint32_t lidarSensorIndex;
        CHECK_DW_ERROR(dwSensorManager_getSensorIndex(&lidarSensorIndex, DW_SENSOR_LIDAR, 0, m_sensorManager));
        CHECK_DW_ERROR(dwSensorManager_getSensorHandle(&m_lidarSensor, lidarSensorIndex, m_sensorManager));
        CHECK_DW_ERROR(dwSensorLidar_getProperties(&m_lidarProperties, m_lidarSensor));
        
        m_pointCloudCapacity = m_lidarProperties.pointsPerSecond * 2;
        m_pointCloudBuffer.reset(new dwLidarPointXYZI[m_pointCloudCapacity]);
        
        std::cout << "[DWFastLIO] LiDAR initialized from rig: " << m_lidarProperties.deviceString << std::endl;
        
        // Get IMU sensor
        uint32_t imuCount = 0;
        CHECK_DW_ERROR(dwSensorManager_getNumSensors(&imuCount, DW_SENSOR_IMU, m_sensorManager));
        
        if (imuCount == 0) {
            std::cerr << "[DWFastLIO] ERROR: No IMU sensors found in rig file!" << std::endl;
            return false;
        }
        
        if (imuCount > 1) {
            std::cout << "[DWFastLIO] WARNING: Multiple IMU sensors found, using first one" << std::endl;
        }
        
        uint32_t imuSensorIndex;
        CHECK_DW_ERROR(dwSensorManager_getSensorIndex(&imuSensorIndex, DW_SENSOR_IMU, 0, m_sensorManager));
        CHECK_DW_ERROR(dwSensorManager_getSensorHandle(&m_imuSensor, imuSensorIndex, m_sensorManager));
        
        std::cout << "[DWFastLIO] IMU initialized from rig" << std::endl;
        
        // Get GPS sensor (optional)
        uint32_t gpsCount = 0;
        CHECK_DW_ERROR(dwSensorManager_getNumSensors(&gpsCount, DW_SENSOR_GPS, m_sensorManager));
        
        if (gpsCount > 0) {
            uint32_t gpsSensorIndex;
            CHECK_DW_ERROR(dwSensorManager_getSensorIndex(&gpsSensorIndex, DW_SENSOR_GPS, 0, m_sensorManager));
            CHECK_DW_ERROR(dwSensorManager_getSensorHandle(&m_gpsSensor, gpsSensorIndex, m_sensorManager));
            std::cout << "[DWFastLIO] GPS initialized from rig" << std::endl;
        }
        
        // Extract LiDAR-to-IMU transform from rig
        dwTransformation3f lidarToRig, imuToRig;
        CHECK_DW_ERROR(dwRig_getSensorToRigTransformation(&lidarToRig, lidarSensorIndex, m_rigConfig));
        CHECK_DW_ERROR(dwRig_getSensorToRigTransformation(&imuToRig, imuSensorIndex, m_rigConfig));
        
        // Compute lidar-to-IMU transform: T_lidar_imu = T_rig_imu^-1 * T_rig_lidar
        // Convert to Eigen for easier computation
        Eigen::Matrix4d T_rig_lidar = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d T_rig_imu = Eigen::Matrix4d::Identity();
        
        // Extract translation and rotation from DriveWorks transforms
        // DriveWorks uses column-major: item(row,col) = array[row + col*4]
        // Translation is in column 3, rows 0-2 (indices 12, 13, 14)
        T_rig_lidar(0, 3) = lidarToRig.array[0 + 3*4];  // row 0, col 3
        T_rig_lidar(1, 3) = lidarToRig.array[1 + 3*4];  // row 1, col 3
        T_rig_lidar(2, 3) = lidarToRig.array[2 + 3*4];  // row 2, col 3
        
        // Rotation matrix is in rows 0-2, columns 0-2
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                T_rig_lidar(i, j) = lidarToRig.array[i + j*4];  // row i, col j (column-major)
            }
        }
        
        T_rig_imu(0, 3) = imuToRig.array[0 + 3*4];
        T_rig_imu(1, 3) = imuToRig.array[1 + 3*4];
        T_rig_imu(2, 3) = imuToRig.array[2 + 3*4];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                T_rig_imu(i, j) = imuToRig.array[i + j*4];
            }
        }
        
        // Compute lidar-to-IMU: T_lidar_imu = T_rig_imu^-1 * T_rig_lidar
        m_slamConfig.lidar_to_imu_transform = T_rig_imu.inverse() * T_rig_lidar;
        
        std::cout << "[DWFastLIO] LiDAR-to-IMU transform from rig:" << std::endl;
        std::cout << "[DWFastLIO]   Translation: [" 
                  << m_slamConfig.lidar_to_imu_transform(0, 3) << ", "
                  << m_slamConfig.lidar_to_imu_transform(1, 3) << ", "
                  << m_slamConfig.lidar_to_imu_transform(2, 3) << "]" << std::endl;
        
        return (m_lidarSensor != DW_NULL_HANDLE) && (m_imuSensor != DW_NULL_HANDLE);
    }
    
    bool initializeSensorsFromParams() {
        // Initialize LiDAR
        std::string lidarParams = getArgument("lidar-params");
        std::string lidarProtocol = getArgument("lidar-protocol");
        
        if (!lidarParams.empty() && !lidarProtocol.empty()) {
            dwSensorParams params{};
            params.protocol = lidarProtocol.c_str();
            params.parameters = lidarParams.c_str();
            
            CHECK_DW_ERROR(dwSAL_createSensor(&m_lidarSensor, params, m_sal));
            CHECK_DW_ERROR(dwSensorLidar_getProperties(&m_lidarProperties, m_lidarSensor));
            
            m_pointCloudCapacity = m_lidarProperties.pointsPerSecond * 2;
            m_pointCloudBuffer.reset(new dwLidarPointXYZI[m_pointCloudCapacity]);
            
            CHECK_DW_ERROR(dwSensor_start(m_lidarSensor));
            std::cout << "[DWFastLIO] LiDAR initialized: " << m_lidarProperties.deviceString << std::endl;
        }
        
        // Initialize IMU
        std::string imuParams = getArgument("imu-params");
        std::string imuProtocol = getArgument("imu-protocol");
        
        if (!imuParams.empty() && !imuProtocol.empty()) {
            dwSensorParams params{};
            params.protocol = imuProtocol.c_str();
            params.parameters = imuParams.c_str();
            
            CHECK_DW_ERROR(dwSAL_createSensor(&m_imuSensor, params, m_sal));
            CHECK_DW_ERROR(dwSensor_start(m_imuSensor));
            std::cout << "[DWFastLIO] IMU initialized" << std::endl;
        }
        
        // Initialize GPS (optional)
        std::string gpsParams = getArgument("gps-params");
        std::string gpsProtocol = getArgument("gps-protocol");
        
        if (!gpsParams.empty() && !gpsProtocol.empty()) {
            dwSensorParams params{};
            params.protocol = gpsProtocol.c_str();
            params.parameters = gpsParams.c_str();
            
            CHECK_DW_ERROR(dwSAL_createSensor(&m_gpsSensor, params, m_sal));
            CHECK_DW_ERROR(dwSensor_start(m_gpsSensor));
            std::cout << "[DWFastLIO] GPS initialized" << std::endl;
        }
        
        // Use identity transform if no extrinsics provided
        m_slamConfig.lidar_to_imu_transform = Eigen::Matrix4d::Identity();
        
        return (m_lidarSensor != DW_NULL_HANDLE) && (m_imuSensor != DW_NULL_HANDLE);
    }
    
    bool initializeSLAM() {
        // Configure SLAM
        m_slamConfig.scan_period = 1.0 / m_lidarProperties.spinFrequency;
        m_slamConfig.voxel_size = std::stof(getArgument("voxel-size"));
        
        // Parse extrinsics if provided (lidar to IMU transform)
        // Format: "tx,ty,tz,qw,qx,qy,qz"
        std::string extrinsics = getArgument("lidar-imu-extrinsics");
        if (!extrinsics.empty()) {
            // Parse extrinsics string
            // For now, use identity
            m_slamConfig.lidar_to_imu_transform = Eigen::Matrix4d::Identity();
        }
        
        // Create SLAM object
        m_slam.reset(new dw_slam::DWFastLIO());
        
        // Check if we should load a map
        std::string mapFile = getArgument("map-file");
        if (!mapFile.empty()) {
            if (m_slam->loadMap(mapFile)) {
                m_mappingMode = false;
                m_statusMessage = "Mode: LOCALIZATION";
            }
        } else {
            m_mappingMode = true;
            m_statusMessage = "Mode: MAPPING";
        }
        
        return m_slam->initialize(m_slamConfig);
    }
    
    
    bool initializeVisualization() {
        // Initialize visualization context
        CHECK_DW_ERROR(dwVisualizationInitialize(&m_visualizationContext, m_context));
        
        // Initialize render engine
        dwRenderEngineParams renderParams{};
        CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&renderParams,
                                                        getWindowWidth(),
                                                        getWindowHeight()));
        renderParams.defaultTile.backgroundColor = {0.0f, 0.0f, 0.1f, 1.0f};
        CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &renderParams, m_visualizationContext));
        
        // Create buffers (map uses 4 floats per point for x,y,z,intensity for colored-by-intensity rendering)
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_mapBuffer,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                   4 * sizeof(float32_t), 0, 10000000, // 10M points
                                                   m_renderEngine));
                                                   
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_currentScanBuffer,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                   sizeof(dwVector3f), 0, 1000000, // 1M points
                                                   m_renderEngine));
                                                   
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_trajectoryBuffer,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINESTRIP_3D,
                                                   sizeof(dwVector3f), 0, 100000, // trajectory
                                                   m_renderEngine));
        
        return true;
    }
    
    void onProcess() override {
        // When using rig: single acquireNextEvent() loop (same as egomotion_pomo).
        // One Sensor Manager → one connection; IMU and GPS from same port are demultiplexed by the plugin.
        if (m_sensorManager != DW_NULL_HANDLE) {
            processEventsFromSensorManager();
            return;
        }
        // Params path: no Sensor Manager, use per-sensor readFrame/readPacket.
        processSensorsDirect();
    }

    // Rig path: single event loop (IMU + GPS from same port via one connection).
    void processEventsFromSensorManager() {
        constexpr int MAX_PROCESS_TIME_MS = 50;
        constexpr int MAX_EVENTS_PER_FRAME = 256;
        auto processStart = std::chrono::steady_clock::now();
        int eventsProcessed = 0;

        while (eventsProcessed < MAX_EVENTS_PER_FRAME) {
            auto elapsed = std::chrono::steady_clock::now() - processStart;
            if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > MAX_PROCESS_TIME_MS)
                break;

            const dwSensorEvent* acquiredEvent = nullptr;
            dwStatus status = dwSensorManager_acquireNextEvent(&acquiredEvent, 0, m_sensorManager);

            if (status == DW_TIME_OUT)
                break;
            if (status != DW_SUCCESS) {
                if (status == DW_END_OF_STREAM)
                    std::cout << "[DWFastLIO] Sensor manager end of stream" << std::endl;
                break;
            }

            switch (acquiredEvent->type) {
                case DW_SENSOR_IMU: {
                    const dwIMUFrame& imu = acquiredEvent->imuFrame;
                    m_latestIMU = imu;
                    m_hasIMU = true;
                    if (m_slam)
                        m_slam->feedIMU(imu, imu.hostTimestamp);
                    break;
                }
                case DW_SENSOR_GPS: {
                    if (m_hasIMU && m_slam) {
                        const dwGPSFrame& gps = acquiredEvent->gpsFrame;
                        m_slam->feedGPS(gps, m_latestIMU, gps.timestamp_us);
                    }
                    break;
                }
                case DW_SENSOR_LIDAR: {
                    const dwLidarDecodedPacket* packet = acquiredEvent->lidFrame;
                    if (!packet) break;
                    // Copy packet data before releasing event
                    if (m_pointCloudSize + packet->nPoints <= m_pointCloudCapacity) {
                        memcpy(&m_pointCloudBuffer[m_pointCloudSize], packet->pointsXYZI,
                               packet->nPoints * sizeof(dwLidarPointXYZI));
                        m_pointCloudSize += packet->nPoints;
                    } else {
                        m_pointCloudSize = 0;
                        dwSensorManager_releaseAcquiredEvent(acquiredEvent, m_sensorManager);
                        acquiredEvent = nullptr;
                        break;
                    }
                    bool isScanComplete = packet->scanComplete;
                    dwTime_t packetTimestamp = packet->hostTimestamp;
                    dwSensorManager_releaseAcquiredEvent(acquiredEvent, m_sensorManager);
                    acquiredEvent = nullptr;

                    if (isScanComplete && m_pointCloudSize > 0 && m_slam && m_pointCloudBuffer) {
                        try {
                            m_slam->feedLiDAR(m_pointCloudBuffer.get(), m_pointCloudSize, packetTimestamp);
                        } catch (const std::exception& e) {
                            std::cerr << "[DWFastLIO] feedLiDAR exception: " << e.what() << std::endl;
                        } catch (...) {
                            std::cerr << "[DWFastLIO] feedLiDAR unknown exception" << std::endl;
                        }
                        m_pointCloudSize = 0;
                    }
                    break;
                }
                default:
                    break;
            }

            if (acquiredEvent)
                dwSensorManager_releaseAcquiredEvent(acquiredEvent, m_sensorManager);
            ++eventsProcessed;
        }
    }

    // Params path: per-sensor readFrame/readPacket (used when not using rig).
    void processSensorsDirect() {
        if (m_imuSensor != DW_NULL_HANDLE) {
            for (int i = 0; i < 10; ++i) {
                dwIMUFrame imuFrame;
                dwStatus status = dwSensorIMU_readFrame(&imuFrame, 10000, m_imuSensor);
                if (status == DW_SUCCESS) {
                    m_latestIMU = imuFrame;
                    m_hasIMU = true;
                    if (m_slam) m_slam->feedIMU(imuFrame, imuFrame.hostTimestamp);
                } else if (status == DW_TIME_OUT || status == DW_NOT_READY || status == DW_NOT_AVAILABLE)
                    break;
                else break;
            }
        }
        if (m_gpsSensor != DW_NULL_HANDLE && m_hasIMU && m_slam) {
            for (int i = 0; i < 2; ++i) {
                dwGPSFrame gpsFrame;
                if (dwSensorGPS_readFrame(&gpsFrame, 0, m_gpsSensor) != DW_SUCCESS)
                    break;
                m_slam->feedGPS(gpsFrame, m_latestIMU, gpsFrame.timestamp_us);
            }
        }
        if (m_lidarSensor != DW_NULL_HANDLE) {
            const auto MAX_PROCESS_TIME_MS = 50;
            auto processStart = std::chrono::steady_clock::now();
            while (true) {
                if (std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - processStart).count() > MAX_PROCESS_TIME_MS)
                    break;
                const dwLidarDecodedPacket* packet = nullptr;
                dwStatus status = dwSensorLidar_readPacket(&packet, 10000, m_lidarSensor);
                if (status != DW_SUCCESS) break;
                if (m_pointCloudSize + packet->nPoints <= m_pointCloudCapacity) {
                    memcpy(&m_pointCloudBuffer[m_pointCloudSize], packet->pointsXYZI,
                           packet->nPoints * sizeof(dwLidarPointXYZI));
                    m_pointCloudSize += packet->nPoints;
                } else {
                    m_pointCloudSize = 0;
                    dwSensorLidar_returnPacket(packet, m_lidarSensor);
                    break;
                }
                bool isScanComplete = packet->scanComplete;
                dwTime_t packetTimestamp = packet->hostTimestamp;
                dwSensorLidar_returnPacket(packet, m_lidarSensor);
                if (isScanComplete && m_pointCloudSize > 0 && m_slam && m_pointCloudBuffer) {
                    try {
                        m_slam->feedLiDAR(m_pointCloudBuffer.get(), m_pointCloudSize, packetTimestamp);
                    } catch (const std::exception& e) {
                        std::cerr << "[DWFastLIO] feedLiDAR exception: " << e.what() << std::endl;
                    } catch (...) {
                        std::cerr << "[DWFastLIO] feedLiDAR unknown exception" << std::endl;
                    }
                    m_pointCloudSize = 0;
                }
                if (isScanComplete) break;
            }
        }
    }
    
    void onRender() override {
        if (!m_renderEngine || !m_slam || m_mapBuffer == 0) {
            return; // Not initialized or already released
        }
        
        dwRenderEngine_reset(m_renderEngine);
        dwRenderEngine_setTile(0, m_renderEngine);
        dwRenderEngine_setModelView(getMouseView().getModelView(), m_renderEngine);
        dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine);
        
        // Get pose once (thread-safe, fast)
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity(); // Default identity
        try {
            pose = m_slam->getCurrentPose();
        } catch (...) {
            pose = Eigen::Matrix4d::Identity();
        }
        // Detect invalid pose (e.g. timestamp-unit drift: position in billions)
        const double kMaxReasonablePos = 1e6;
        bool poseValid = std::isfinite(pose(0,3)) && std::isfinite(pose(1,3)) && std::isfinite(pose(2,3))
                         && std::fabs(pose(0,3)) < kMaxReasonablePos
                         && std::fabs(pose(1,3)) < kMaxReasonablePos
                         && std::fabs(pose(2,3)) < kMaxReasonablePos;
        
        // Render map - colored by intensity (like Fast-LIO repo: yellow/green/white gradient)
        struct PointXYZI { float32_t x, y, z, intensity; };
        auto mapCloud = m_slam->getMapCloud();
        if (mapCloud && !mapCloud->empty()) {
            const size_t MAX_RENDER_POINTS = 50000; // Limit render points for performance
            const size_t step = std::max(size_t(1), mapCloud->size() / MAX_RENDER_POINTS);
            
            std::vector<PointXYZI> mapPoints;
            mapPoints.reserve(std::min(mapCloud->size() / step, MAX_RENDER_POINTS));
            
            for (size_t i = 0; i < mapCloud->size(); i += step) {
                if (mapPoints.size() >= MAX_RENDER_POINTS) break;
                const auto& pt = mapCloud->points[i];
                mapPoints.push_back({pt.x, pt.y, pt.z, pt.intensity});
            }
            
            if (!mapPoints.empty()) {
                dwRenderEngine_setBuffer(m_mapBuffer,
                                        DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                        mapPoints.data(),
                                        4 * sizeof(float32_t), 0,
                                        mapPoints.size(),
                                        m_renderEngine);
                dwRenderEngine_setColorByValue(DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_INTENSITY, 1.0f, m_renderEngine);
                dwRenderEngine_renderBuffer(m_mapBuffer, mapPoints.size(), m_renderEngine);
            }
        }
        // Reset to solid color so next buffers (3 components) don't trigger "color dimension too small"
        dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_RED, m_renderEngine);
        
        // Render current scan - limit to reasonable size
        if (m_pointCloudSize > 0) {
            const size_t MAX_SCAN_POINTS = 10000; // Limit scan rendering
            const size_t step = std::max(size_t(1), m_pointCloudSize / MAX_SCAN_POINTS);
            
            std::vector<dwVector3f> scanPoints;
            scanPoints.reserve(std::min(m_pointCloudSize / step, MAX_SCAN_POINTS));
            
            for (size_t i = 0; i < m_pointCloudSize; i += step) {
                if (scanPoints.size() >= MAX_SCAN_POINTS) break;
                // Transform to world frame
                Eigen::Vector4d pt(m_pointCloudBuffer[i].x, m_pointCloudBuffer[i].y, m_pointCloudBuffer[i].z, 1.0);
                Eigen::Vector4d transformed = pose * pt;
                scanPoints.push_back({static_cast<float32_t>(transformed.x()),
                                    static_cast<float32_t>(transformed.y()),
                                    static_cast<float32_t>(transformed.z())});
            }
            
            if (!scanPoints.empty()) {
                dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_RED, m_renderEngine);
                dwRenderEngine_setBuffer(m_currentScanBuffer,
                                        DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                        scanPoints.data(),
                                        sizeof(dwVector3f), 0,
                                        scanPoints.size(),
                                        m_renderEngine);
                dwRenderEngine_renderBuffer(m_currentScanBuffer, scanPoints.size(), m_renderEngine);
            }
        }
        dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_GREEN, m_renderEngine);
        
        // Render trajectory (green line) - only when pose is valid; otherwise trajectory is huge and draws a single long line
        auto trajectory = m_slam->getTrajectory();
        if (poseValid && !trajectory.empty()) {
            const size_t MAX_TRAJ_POINTS = 1000; // Limit trajectory rendering
            size_t startIdx = trajectory.size() > MAX_TRAJ_POINTS ? trajectory.size() - MAX_TRAJ_POINTS : 0;
            
            std::vector<dwVector3f> trajPoints;
            trajPoints.reserve(trajectory.size() - startIdx);
            for (size_t i = startIdx; i < trajectory.size(); ++i) {
                double x = trajectory[i](0,3), y = trajectory[i](1,3), z = trajectory[i](2,3);
                if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)
                    && std::fabs(x) < kMaxReasonablePos && std::fabs(y) < kMaxReasonablePos && std::fabs(z) < kMaxReasonablePos) {
                    trajPoints.push_back({static_cast<float32_t>(x), static_cast<float32_t>(y), static_cast<float32_t>(z)});
                }
            }
            
            if (!trajPoints.empty()) {
                dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_GREEN, m_renderEngine);
                dwRenderEngine_setBuffer(m_trajectoryBuffer,
                                        DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINESTRIP_3D,
                                        trajPoints.data(),
                                        sizeof(dwVector3f), 0,
                                        trajPoints.size(),
                                        m_renderEngine);
                dwRenderEngine_renderBuffer(m_trajectoryBuffer, trajPoints.size(), m_renderEngine);
            }
        }
        
        // Render text overlay
        dwRenderEngine_setProjection(&DW_IDENTITY_MATRIX4F, m_renderEngine);
        dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
        dwVector2f range{static_cast<float32_t>(getWindowWidth()),
                         static_cast<float32_t>(getWindowHeight())};
        dwRenderEngine_setCoordinateRange2D(range, m_renderEngine);
        dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_renderEngine);
        dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_GREEN, m_renderEngine);
        
        std::string poseText = poseValid
            ? ("Position: " + std::to_string(pose(0,3)) + ", " + std::to_string(pose(1,3)) + ", " + std::to_string(pose(2,3)))
            : "Position: (invalid - check timestamp units)";
        
        dwRenderEngine_renderText2D(m_statusMessage.c_str(), {20.f, static_cast<float32_t>(getWindowHeight()) - 30.f}, m_renderEngine);
        dwRenderEngine_renderText2D(poseText.c_str(), {20.f, static_cast<float32_t>(getWindowHeight()) - 50.f}, m_renderEngine);
        
        if (mapCloud && !mapCloud->empty()) {
            std::string mapText = "Map points: " + std::to_string(mapCloud->size());
            dwRenderEngine_renderText2D(mapText.c_str(), {20.f, static_cast<float32_t>(getWindowHeight()) - 70.f}, m_renderEngine);
        }
    }
    
    void onKeyDown(int key, int scancode, int mods) override {
        (void)scancode;
        (void)mods;
        
        if (key == GLFW_KEY_M) {
            // Toggle mapping/localization mode
            m_mappingMode = !m_mappingMode;
            m_slam->setMappingMode(m_mappingMode);
            m_statusMessage = m_mappingMode ? "Mode: MAPPING" : "Mode: LOCALIZATION";
        }
        else if (key == GLFW_KEY_S) {
            // Save map: keyframe directory if we have keyframes from mapping, else single PCD
            std::string base = "slam_map_" + std::to_string(std::time(nullptr));
            std::string path = m_slam->hasKeyframesToSave() ? base : (base + ".pcd");
            if (m_slam->saveMap(path)) {
                m_statusMessage = "Map saved: " + path;
            }
        }
        else if (key == GLFW_KEY_L) {
            // Load map (would need file dialog in production)
            std::string filename = getArgument("map-file");
            if (!filename.empty() && m_slam->loadMap(filename)) {
                m_mappingMode = false;
                m_statusMessage = "Map loaded: " + filename;
            }
        }
        else if (key == GLFW_KEY_R) {
            // Request global re-localization (only when keyframe map is loaded)
            if (m_slam->hasKeyframeMap()) {
                m_slam->requestRelocalization();
                m_statusMessage = "Re-localization requested (next scan)";
                std::cout << "[dw_fastlio] Re-localization requested. Next LiDAR scan will run ScanContext + GICP." << std::endl;
            } else {
                m_statusMessage = "Reloc requires keyframe map (load dir with S or --map-file=dir)";
                std::cout << "[dw_fastlio] Re-localization requires a keyframe map. Load a map directory (saved with S), not a single .pcd." << std::endl;
            }
        }
    }
    
    void onRelease() override {
        // 1) Stop sensor manager first so no more events are delivered (avoids feedLiDAR after SLAM is gone).
        //    When using a rig, sensor handles are owned by the manager - do NOT release them individually.
        if (m_sensorManager != DW_NULL_HANDLE) {
            dwSensorManager_stop(m_sensorManager);
            std::this_thread::sleep_for(std::chrono::milliseconds(150)); // let plugin threads exit
            dwSensorManager_release(m_sensorManager);
            m_sensorManager = DW_NULL_HANDLE;
            m_lidarSensor = DW_NULL_HANDLE;
            m_imuSensor = DW_NULL_HANDLE;
            m_gpsSensor = DW_NULL_HANDLE;
        } else {
            // Params path: we created sensors with dwSAL_createSensor - release them.
            if (m_lidarSensor != DW_NULL_HANDLE) {
                dwSAL_releaseSensor(m_lidarSensor);
                m_lidarSensor = DW_NULL_HANDLE;
            }
            if (m_imuSensor != DW_NULL_HANDLE) {
                dwSAL_releaseSensor(m_imuSensor);
                m_imuSensor = DW_NULL_HANDLE;
            }
            if (m_gpsSensor != DW_NULL_HANDLE) {
                dwSAL_releaseSensor(m_gpsSensor);
                m_gpsSensor = DW_NULL_HANDLE;
            }
        }
        if (m_rigConfig != DW_NULL_HANDLE) {
            dwRig_release(m_rigConfig);
            m_rigConfig = DW_NULL_HANDLE;
        }

        // 2) Release SLAM (stops processing thread in destructor)
        m_slam.reset();
        
        // 3) Release buffers (invalidate handles so late onRender is a no-op)
        if (m_mapBuffer != 0 && m_renderEngine != DW_NULL_HANDLE) {
            dwRenderEngine_destroyBuffer(m_mapBuffer, m_renderEngine);
            m_mapBuffer = 0;
        }
        if (m_currentScanBuffer != 0 && m_renderEngine != DW_NULL_HANDLE) {
            dwRenderEngine_destroyBuffer(m_currentScanBuffer, m_renderEngine);
            m_currentScanBuffer = 0;
        }
        if (m_trajectoryBuffer != 0 && m_renderEngine != DW_NULL_HANDLE) {
            dwRenderEngine_destroyBuffer(m_trajectoryBuffer, m_renderEngine);
            m_trajectoryBuffer = 0;
        }
        
        // 4) Release visualization
        if (m_renderEngine != DW_NULL_HANDLE) {
            dwRenderEngine_release(m_renderEngine);
            m_renderEngine = DW_NULL_HANDLE;
        }
        if (m_visualizationContext != DW_NULL_HANDLE) {
            dwVisualizationRelease(m_visualizationContext);
            m_visualizationContext = DW_NULL_HANDLE;
        }
        
        // Release DriveWorks
        if (m_sal != DW_NULL_HANDLE) {
            dwSAL_release(m_sal);
        }
        if (m_context != DW_NULL_HANDLE) {
            dwRelease(m_context);
        }
        
        dwLogger_release();
    }
    
    void onResizeWindow(int width, int height) override {
        dwRectf bounds{0.0f, 0.0f, static_cast<float32_t>(width), static_cast<float32_t>(height)};
        dwRenderEngine_setBounds(bounds, m_renderEngine);
    }
};

//------------------------------------------------------------------------------
int main(int argc, const char** argv) {
    // Setup signal handlers
    struct sigaction action = {};
    action.sa_handler = sig_int_handler;
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGTERM, &action, NULL);
    
    // Program arguments
    ProgramArguments args(argc, argv, {
        ProgramArguments::Option_t("rig-file", "", "Path to rig.json configuration file (alternative to individual sensor params)"),
        ProgramArguments::Option_t("lidar-protocol", "lidar.virtual"),
        ProgramArguments::Option_t("lidar-params", "file=/path/to/lidar.bin"),
        ProgramArguments::Option_t("imu-protocol", "imu.virtual"),
        ProgramArguments::Option_t("imu-params", "file=/path/to/imu.bin"),
        ProgramArguments::Option_t("gps-protocol", ""),
        ProgramArguments::Option_t("gps-params", ""),
        ProgramArguments::Option_t("lidar-imu-extrinsics", ""),
        ProgramArguments::Option_t("voxel-size", "0.5"),
        ProgramArguments::Option_t("map-file", ""),
    });
    
    // Run sample
    DWFastLIOSample app(args);
    
    if (args.enabled("offscreen")) {
        app.initializeWindow("DriveWorks Fast-LIO SLAM", 1280, 720, args.enabled("offscreen"));
    } else {
        app.initializeWindow("DriveWorks Fast-LIO SLAM", 1280, 720, false);
    }
    
    return app.run();
}

