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
        // Initialize logger
        CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
        CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));
        
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
        
        std::cout << "[DWFastLIO] System initialized successfully" << std::endl;
        std::cout << "[DWFastLIO] Mode: MAPPING" << std::endl;
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
        // Read IMU
        if (m_imuSensor != DW_NULL_HANDLE) {
            dwIMUFrame imuFrame;
            dwStatus status = dwSensorIMU_readFrame(&imuFrame, 0, m_imuSensor); // Non-blocking
            
            if (status == DW_SUCCESS) {
                m_latestIMU = imuFrame;
                m_hasIMU = true;
                m_slam->feedIMU(imuFrame, imuFrame.hostTimestamp);
            }
        }
        
        // Read GPS
        if (m_gpsSensor != DW_NULL_HANDLE && m_hasIMU) {
            dwGPSFrame gpsFrame;
            dwStatus status = dwSensorGPS_readFrame(&gpsFrame, 0, m_gpsSensor); // Non-blocking
            
            if (status == DW_SUCCESS) {
                m_slam->feedGPS(gpsFrame, m_latestIMU, gpsFrame.timestamp_us);
            }
        }
        
        // Read LiDAR packets and accumulate full spin
        if (m_lidarSensor != DW_NULL_HANDLE) {
            const dwLidarDecodedPacket* packet;
            dwStatus status = dwSensorLidar_readPacket(&packet, 100000, m_lidarSensor);
            
            if (status == DW_SUCCESS) {
                // Accumulate points
                if (m_pointCloudSize + packet->nPoints <= m_pointCloudCapacity) {
                    memcpy(&m_pointCloudBuffer[m_pointCloudSize], packet->pointsXYZI,
                           packet->nPoints * sizeof(dwLidarPointXYZI));
                    m_pointCloudSize += packet->nPoints;
                }
                
                // When scan complete, feed to SLAM
                if (packet->scanComplete && m_pointCloudSize > 0) {
                    m_slam->feedLiDAR(m_pointCloudBuffer.get(), m_pointCloudSize, packet->hostTimestamp);
                    m_pointCloudSize = 0; // Reset for next scan
                }
                
                dwSensorLidar_returnPacket(packet, m_lidarSensor);
            } else if (status == DW_END_OF_STREAM) {
                // For recorded data, reset and loop
                dwSensor_reset(m_lidarSensor);
                m_pointCloudSize = 0;
            }
        }
    }
    
    void onRender() override {
        dwRenderEngine_reset(m_renderEngine);
        dwRenderEngine_setTile(0, m_renderEngine);
        dwRenderEngine_setModelView(getMouseView().getModelView(), m_renderEngine);
        dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine);
        
        // Render map
        auto mapCloud = m_slam->getMapCloud();
        if (mapCloud && !mapCloud->empty()) {
            std::vector<dwVector3f> mapPoints;
            mapPoints.reserve(std::min(mapCloud->size(), size_t(1000000)));
            
            for (const auto& pt : mapCloud->points) {
                if (mapPoints.size() >= 1000000) break; // Limit for performance
                mapPoints.push_back({pt.x, pt.y, pt.z});
            }
            
            dwRenderEngine_setBuffer(m_mapBuffer,
                                    DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                    mapPoints.data(),
                                    sizeof(dwVector3f), 0,
                                    mapPoints.size(),
                                    m_renderEngine);
            
            dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_WHITE, m_renderEngine);
            dwRenderEngine_renderBuffer(m_mapBuffer, mapPoints.size(), m_renderEngine);
        }
        
        // Render current scan
        if (m_pointCloudSize > 0) {
            std::vector<dwVector3f> scanPoints(m_pointCloudSize);
            auto pose = m_slam->getCurrentPose();
            
            for (size_t i = 0; i < m_pointCloudSize; ++i) {
                // Transform to world frame
                Eigen::Vector4d pt(m_pointCloudBuffer[i].x, m_pointCloudBuffer[i].y, m_pointCloudBuffer[i].z, 1.0);
                Eigen::Vector4d transformed = pose * pt;
                scanPoints[i] = {static_cast<float32_t>(transformed.x()),
                                static_cast<float32_t>(transformed.y()),
                                static_cast<float32_t>(transformed.z())};
            }
            
            dwRenderEngine_setBuffer(m_currentScanBuffer,
                                    DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                    scanPoints.data(),
                                    sizeof(dwVector3f), 0,
                                    scanPoints.size(),
                                    m_renderEngine);
            
            dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_RED, m_renderEngine);
            dwRenderEngine_renderBuffer(m_currentScanBuffer, scanPoints.size(), m_renderEngine);
        }
        
        // Render trajectory
        auto trajectory = m_slam->getTrajectory();
        if (!trajectory.empty()) {
            std::vector<dwVector3f> trajPoints;
            trajPoints.reserve(trajectory.size());
            
            for (const auto& pose : trajectory) {
                trajPoints.push_back({static_cast<float32_t>(pose(0,3)),
                                     static_cast<float32_t>(pose(1,3)),
                                     static_cast<float32_t>(pose(2,3))});
            }
            
            dwRenderEngine_setBuffer(m_trajectoryBuffer,
                                    DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINESTRIP_3D,
                                    trajPoints.data(),
                                    sizeof(dwVector3f), 0,
                                    trajPoints.size(),
                                    m_renderEngine);
            
            dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_GREEN, m_renderEngine);
            dwRenderEngine_renderBuffer(m_trajectoryBuffer, trajPoints.size(), m_renderEngine);
        }
        
        // Render text overlay
        dwRenderEngine_setProjection(&DW_IDENTITY_MATRIX4F, m_renderEngine);
        dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
        dwVector2f range{static_cast<float32_t>(getWindowWidth()),
                         static_cast<float32_t>(getWindowHeight())};
        dwRenderEngine_setCoordinateRange2D(range, m_renderEngine);
        dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_renderEngine);
        dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_GREEN, m_renderEngine);
        
        auto pose = m_slam->getCurrentPose();
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

