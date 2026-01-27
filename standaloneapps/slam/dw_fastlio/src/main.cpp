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

// DriveWorks
#include <dw/core/base/Version.h>
#include <dw/core/base/Status.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/lidar/Lidar.h>
#include <dw/sensors/imu/IMU.h>
#include <dw/sensors/gps/GPS.h>

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
        std::cout << "[DWFastLIO] Mode: MAPPING" << std::endl;
        std::cout << "[DWFastLIO] Process rate: 6 FPS, Render rate: 30 FPS" << std::endl;
        std::cout << "[DWFastLIO] Press 'M' to toggle MAPPING/LOCALIZATION mode" << std::endl;
        std::cout << "[DWFastLIO] Press 'S' to save map" << std::endl;
        std::cout << "[DWFastLIO] Press 'L' to load map" << std::endl;
        
        return true;
    }
    
    bool initializeSensors() {
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
        
        // Create buffers
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_mapBuffer, 
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                   sizeof(dwVector3f), 0, 10000000, // 10M points
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
        // Ultra-conservative processing: Only process minimal data per frame
        // Target: 5-6 FPS, so we have ~166ms per frame, but keep processing very light
        // IMPORTANT: feedLiDAR() conversion can block - we need to limit scan size or make it async
        
        // Read IMU - use reasonable timeout (DriveWorks samples use 10ms)
        if (m_imuSensor != DW_NULL_HANDLE) {
            static int imu_read_attempts = 0;
            static int imu_read_success = 0;
            
            // Read multiple IMU frames per cycle to drain buffer
            for (int i = 0; i < 10; ++i) { // Process up to 10 IMU frames max
                dwIMUFrame imuFrame;
                // Use 10ms timeout like DriveWorks IMU sample - IMU data comes at high frequency
                dwStatus status = dwSensorIMU_readFrame(&imuFrame, 10000, m_imuSensor); // 10ms timeout
                
                imu_read_attempts++;
                
                if (status == DW_SUCCESS) {
                    imu_read_success++;
                    m_latestIMU = imuFrame;
                    m_hasIMU = true;
                    m_slam->feedIMU(imuFrame, imuFrame.hostTimestamp);
                    
                    // Log first few successful reads
                    if (imu_read_success <= 5) {
                        std::cout << "[DWFastLIO] IMU read SUCCESS #" << imu_read_success 
                                  << " (attempts: " << imu_read_attempts << ")" << std::endl;
                    }
                } else if (status == DW_TIME_OUT || status == DW_NOT_READY || status == DW_NOT_AVAILABLE) {
                    // Normal - no more IMU data available right now
                    break;
                } else {
                    // Other error - log it
                    if (imu_read_attempts <= 5) {
                        std::cout << "[DWFastLIO] IMU read error: " << dwGetStatusName(status) 
                                  << " (attempt #" << imu_read_attempts << ")" << std::endl;
                    }
                    break;
                }
            }
        } else {
            static bool warned = false;
            if (!warned) {
                std::cout << "[DWFastLIO] WARNING: IMU sensor handle is NULL!" << std::endl;
                warned = true;
            }
        }
        
        // Read GPS - only process a few frames per cycle
        if (m_gpsSensor != DW_NULL_HANDLE && m_hasIMU) {
            for (int i = 0; i < 2; ++i) { // Process only 2 GPS frames max
                dwGPSFrame gpsFrame;
                dwStatus status = dwSensorGPS_readFrame(&gpsFrame, 0, m_gpsSensor); // Non-blocking
                
                if (status == DW_SUCCESS) {
                    m_slam->feedGPS(gpsFrame, m_latestIMU, gpsFrame.timestamp_us);
                } else {
                    break; // No more data available
                }
            }
        }
        
        // Read LiDAR packets - accumulate complete 360° scans (like DriveWorks lidar_replay sample)
        // CRITICAL: Return packets IMMEDIATELY after copying to avoid queue overflow
        // Fast-LIO expects complete scans, not individual packets
        // Using CPU memory (dwLidarPointXYZI) - no GPU-CPU copy overhead
        if (m_lidarSensor != DW_NULL_HANDLE) {
            static int totalPacketsRead = 0;
            static int totalScansComplete = 0;
            static int packetsInCurrentScan = 0;
            
            // Time budget: onProcess() should return quickly (target: <50ms for 6 FPS)
            auto processStart = std::chrono::steady_clock::now();
            const auto MAX_PROCESS_TIME_MS = 50;
            
            // Read packets in a loop until we get a complete scan or timeout
            // Use shorter timeout (10ms) to avoid blocking too long
            while (true) {
                // Check time budget - must return quickly to avoid queue overflow
                auto elapsed = std::chrono::steady_clock::now() - processStart;
                if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > MAX_PROCESS_TIME_MS) {
                    if (totalPacketsRead <= 20) {
                        std::cout << "[DWFastLIO] Time budget exceeded, returning from onProcess()" << std::endl;
                    }
                    break; // Return to avoid blocking
                }
                
                const dwLidarDecodedPacket* packet;
                // Use 10ms timeout - short enough to avoid blocking, long enough for packets
                dwStatus status = dwSensorLidar_readPacket(&packet, 10000, m_lidarSensor); // 10ms timeout
                
                if (status == DW_SUCCESS) {
                    totalPacketsRead++;
                    packetsInCurrentScan++;
                    
                    // Log first few packets
                    if (totalPacketsRead <= 15) {
                        std::cout << "[DWFastLIO] Read packet #" << totalPacketsRead 
                                  << ": " << packet->nPoints << " points, scanComplete=" 
                                  << packet->scanComplete << ", accumulated=" << m_pointCloudSize << std::endl;
                    }
                    
                    // Accumulate points into CPU buffer (no GPU-CPU copy - already CPU memory)
                    if (m_pointCloudSize + packet->nPoints <= m_pointCloudCapacity) {
                        // Direct memcpy - CPU to CPU, no GPU involved
                        memcpy(&m_pointCloudBuffer[m_pointCloudSize], packet->pointsXYZI,
                               packet->nPoints * sizeof(dwLidarPointXYZI));
                        m_pointCloudSize += packet->nPoints;
                    } else {
                        // Buffer full - this shouldn't happen, but handle it
                        std::cout << "[DWFastLIO] ERROR: Point cloud buffer full! Dropping scan." << std::endl;
                        m_pointCloudSize = 0; // Reset
                        packetsInCurrentScan = 0;
                        dwSensorLidar_returnPacket(packet, m_lidarSensor);
                        break;
                    }
                    
                    // CRITICAL: Return packet IMMEDIATELY after copying to avoid queue overflow
                    // Don't wait for scanComplete to return the packet!
                    bool isScanComplete = packet->scanComplete;
                    dwSensorLidar_returnPacket(packet, m_lidarSensor);
                    
                    // When scan complete, feed COMPLETE 360° scan to SLAM
                    // Fast-LIO expects complete scans, not individual packets
                    if (isScanComplete && m_pointCloudSize > 0) {
                        totalScansComplete++;
                        std::cout << "[DWFastLIO] ===== Complete 360° scan #" << totalScansComplete 
                                  << ": " << m_pointCloudSize << " points from " << packetsInCurrentScan 
                                  << " packets - feeding to SLAM =====" << std::flush << std::endl;
                        
                        // Check if SLAM is initialized
                        if (!m_slam) {
                            std::cout << "[DWFastLIO] ERROR: m_slam is NULL! Cannot feed scan." << std::endl;
                            m_pointCloudSize = 0;
                            packetsInCurrentScan = 0;
                            break;
                        }
                        
                        std::cout << "[DWFastLIO] [MAIN] About to call feedLiDAR() with " << m_pointCloudSize 
                                  << " points, timestamp=" << packet->hostTimestamp << std::flush << std::endl;
                        
                        // Validate inputs before calling
                        if (!m_pointCloudBuffer) {
                            std::cerr << "[DWFastLIO] [MAIN] ERROR: m_pointCloudBuffer is NULL!" << std::endl;
                            m_pointCloudSize = 0;
                            packetsInCurrentScan = 0;
                            break;
                        }
                        if (m_pointCloudSize == 0) {
                            std::cerr << "[DWFastLIO] [MAIN] ERROR: m_pointCloudSize is 0!" << std::endl;
                            m_pointCloudSize = 0;
                            packetsInCurrentScan = 0;
                            break;
                        }
                        
                        std::cout << "[DWFastLIO] [MAIN] Validated inputs, calling feedLiDAR()..." << std::flush << std::endl;
                        std::cerr << "[DWFastLIO] [MAIN] CRITICAL: About to call feedLiDAR, m_slam=" 
                                  << (void*)m_slam.get() << std::flush << std::endl;
                        
                        // WARNING: feedLiDAR will convert points synchronously - this can block!
                        // For large scans (15K+ points), this conversion takes time
                        auto feedStart = std::chrono::steady_clock::now();
                        std::cerr << "[DWFastLIO] [MAIN] CRITICAL: Entering try block..." << std::flush << std::endl;
                        try {
                            std::cerr << "[DWFastLIO] [MAIN] CRITICAL: Testing object callability..." << std::flush << std::endl;
                        m_slam->testFunction();
                        std::cerr << "[DWFastLIO] [MAIN] CRITICAL: testFunction() returned, calling feedLiDAR()..." << std::flush << std::endl;
                            m_slam->feedLiDAR(m_pointCloudBuffer.get(), m_pointCloudSize, packet->hostTimestamp);
                            std::cerr << "[DWFastLIO] [MAIN] CRITICAL: feedLiDAR() returned!" << std::flush << std::endl;
                            std::cout << "[DWFastLIO] [MAIN] feedLiDAR() returned successfully" << std::flush << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "[DWFastLIO] [MAIN] ERROR: Exception in feedLiDAR(): " << e.what() << std::endl;
                        } catch (...) {
                            std::cerr << "[DWFastLIO] [MAIN] ERROR: Unknown exception in feedLiDAR()" << std::endl;
                        }
                        std::cerr << "[DWFastLIO] [MAIN] CRITICAL: Exited try block" << std::flush << std::endl;
                        auto feedTime = std::chrono::steady_clock::now() - feedStart;
                        auto feedMs = std::chrono::duration_cast<std::chrono::milliseconds>(feedTime).count();
                        
                        std::cout << "[DWFastLIO] [MAIN] feedLiDAR() completed in " << feedMs << "ms" << std::flush << std::endl;
                        
                        if (feedMs > 50) {
                            std::cout << "[DWFastLIO] [MAIN] WARNING: feedLiDAR took " << feedMs 
                                      << "ms (this blocks the main thread!)" << std::endl;
                        }
                        
                        m_pointCloudSize = 0; // Reset for next scan
                        packetsInCurrentScan = 0;
                        break; // Got complete scan, exit loop
                    }
                } else if (status == DW_TIME_OUT) {
                    // Timeout - no more packets available right now
                    // If we have accumulated points but scan not complete, keep them for next frame
                    // This is normal - packets arrive asynchronously
                    break;
                } else if (status == DW_END_OF_STREAM) {
                    // End of stream
                    std::cout << "[DWFastLIO] LiDAR end of stream" << std::endl;
                    break;
                } else if (status == DW_NOT_READY || status == DW_NOT_AVAILABLE) {
                    // Not ready - normal, just exit
                    break;
                } else {
                    // Other error - log it
                    static int error_count = 0;
                    if (++error_count <= 5) {
                        std::cout << "[DWFastLIO] LiDAR read error: " << dwGetStatusName(status) << std::endl;
                    }
                    break;
                }
            }
        }
    }
    
    void onRender() override {
        if (!m_renderEngine || !m_slam) {
            return; // Not initialized yet
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
            // If getCurrentPose fails, use identity
            pose = Eigen::Matrix4d::Identity();
        }
        
        // Render map - only render subset of points for performance (max 50K points)
        auto mapCloud = m_slam->getMapCloud();
        if (mapCloud && !mapCloud->empty()) {
            const size_t MAX_RENDER_POINTS = 50000; // Limit render points for performance
            const size_t step = std::max(size_t(1), mapCloud->size() / MAX_RENDER_POINTS);
            
            std::vector<dwVector3f> mapPoints;
            mapPoints.reserve(std::min(mapCloud->size() / step, MAX_RENDER_POINTS));
            
            for (size_t i = 0; i < mapCloud->size(); i += step) {
                if (mapPoints.size() >= MAX_RENDER_POINTS) break;
                const auto& pt = mapCloud->points[i];
                mapPoints.push_back({pt.x, pt.y, pt.z});
            }
            
            if (!mapPoints.empty()) {
                dwRenderEngine_setBuffer(m_mapBuffer,
                                        DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                        mapPoints.data(),
                                        sizeof(dwVector3f), 0,
                                        mapPoints.size(),
                                        m_renderEngine);
                
                dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_WHITE, m_renderEngine);
                dwRenderEngine_renderBuffer(m_mapBuffer, mapPoints.size(), m_renderEngine);
            }
        }
        
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
                dwRenderEngine_setBuffer(m_currentScanBuffer,
                                        DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                        scanPoints.data(),
                                        sizeof(dwVector3f), 0,
                                        scanPoints.size(),
                                        m_renderEngine);
                
                dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_RED, m_renderEngine);
                dwRenderEngine_renderBuffer(m_currentScanBuffer, scanPoints.size(), m_renderEngine);
            }
        }
        
        // Render trajectory - limit to recent points
        auto trajectory = m_slam->getTrajectory();
        if (!trajectory.empty()) {
            const size_t MAX_TRAJ_POINTS = 1000; // Limit trajectory rendering
            size_t startIdx = trajectory.size() > MAX_TRAJ_POINTS ? trajectory.size() - MAX_TRAJ_POINTS : 0;
            
            std::vector<dwVector3f> trajPoints;
            trajPoints.reserve(trajectory.size() - startIdx);
            
            for (size_t i = startIdx; i < trajectory.size(); ++i) {
                trajPoints.push_back({static_cast<float32_t>(trajectory[i](0,3)),
                                     static_cast<float32_t>(trajectory[i](1,3)),
                                     static_cast<float32_t>(trajectory[i](2,3))});
            }
            
            if (!trajPoints.empty()) {
                dwRenderEngine_setBuffer(m_trajectoryBuffer,
                                        DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINESTRIP_3D,
                                        trajPoints.data(),
                                        sizeof(dwVector3f), 0,
                                        trajPoints.size(),
                                        m_renderEngine);
                
                dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_GREEN, m_renderEngine);
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
        
        std::string poseText = "Position: " + 
                              std::to_string(pose(0,3)) + ", " +
                              std::to_string(pose(1,3)) + ", " +
                              std::to_string(pose(2,3));
        
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
            // Save map
            std::string filename = "slam_map_" + std::to_string(std::time(nullptr)) + ".pcd";
            if (m_slam->saveMap(filename)) {
                m_statusMessage = "Map saved: " + filename;
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
    }
    
    void onRelease() override {
        // Release SLAM
        m_slam.reset();
        
        // Release buffers
        if (m_mapBuffer != 0) {
            dwRenderEngine_destroyBuffer(m_mapBuffer, m_renderEngine);
        }
        if (m_currentScanBuffer != 0) {
            dwRenderEngine_destroyBuffer(m_currentScanBuffer, m_renderEngine);
        }
        if (m_trajectoryBuffer != 0) {
            dwRenderEngine_destroyBuffer(m_trajectoryBuffer, m_renderEngine);
        }
        
        // Release visualization
        if (m_renderEngine != DW_NULL_HANDLE) {
            dwRenderEngine_release(m_renderEngine);
        }
        if (m_visualizationContext != DW_NULL_HANDLE) {
            dwVisualizationRelease(m_visualizationContext);
        }
        
        // Release sensors
        if (m_lidarSensor != DW_NULL_HANDLE) {
            dwSAL_releaseSensor(m_lidarSensor);
        }
        if (m_imuSensor != DW_NULL_HANDLE) {
            dwSAL_releaseSensor(m_imuSensor);
        }
        if (m_gpsSensor != DW_NULL_HANDLE) {
            dwSAL_releaseSensor(m_gpsSensor);
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

