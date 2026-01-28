////////////////////////////////////////////////////////////////////////////////
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS"
// NVIDIA MAKES NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR
// OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED
// WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR
// PURPOSE.
//
// FusionEngine - Multi-sensor fusion consumer application
// Consumes LiDAR detection data (port 40002) and camera DNN data (ports 49252-49255)
// and performs sensor fusion to produce combined detection outputs.
//
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

// CUDA
#include <cuda_runtime.h>

// DriveWorks Core
#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>

// DriveWorks Visualization
#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Renderer.h>
#include <dwvisualization/core/Visualization.h>

// DriveWorks Sample Framework
#include <framework/Checks.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/Log.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/WindowGLFW.hpp>
#include <framework/Mat4.hpp>
#include <framework/MouseView3D.hpp>

// FusionEngine headers
#include "FusionEngine.hpp"
#include "FusedPacket.hpp"

using namespace dw_samples::common;
using namespace fusionengine;

//------------------------------------------------------------------------------
// FusionEngine Application Class
//------------------------------------------------------------------------------
class FusionEngineApp : public DriveWorksSample
{
public:
    explicit FusionEngineApp(const ProgramArguments& args);

    // DriveWorksSample interface
    bool onInitialize() override;
    void onRelease() override;
    void onProcess() override;
    void onRender() override;
    void onKeyDown(int key, int scancode, int mods) override;
    void onMouseMove(float x, float y) override;
    void onMouseDown(int button, float x, float y) override;
    void onMouseUp(int button, float x, float y) override;
    void onMouseWheel(float x, float y) override;

private:
    // Initialization helpers
    bool initDriveWorks();
    bool initVisualization();
    bool initFusionEngine();
    bool initRenderBuffers();

    // Rendering helpers
    void renderLidarPointCloud();
    void renderDetections();
    void renderStatusOverlay();
    void render3DBox(const FusedDetection& det, const float* color);

    // Parse command-line arguments
    void parseArguments();

private:
    // DriveWorks context
    dwContextHandle_t m_context{DW_NULL_HANDLE};

    // Visualization
    dwVisualizationContextHandle_t m_viz{DW_NULL_HANDLE};
    dwRenderEngineHandle_t m_renderEngine{DW_NULL_HANDLE};
    uint32_t m_mainTile{0};

    // Camera view control
    std::unique_ptr<MouseView3D> m_mouseView;

    // Render buffers
    uint32_t m_pointCloudBufferId{0};
    uint32_t m_detectionLineBufferId{0};

    // Fusion engine
    std::unique_ptr<FusionEngine> m_fusionEngine;
    FusionEngineConfig m_fusionConfig;

    // Current fused data
    FusedPacket m_currentPacket{};
    std::mutex m_packetMutex;
    bool m_hasPacket{false};

    // Configuration from arguments
    std::string m_lidarIP{"127.0.0.1"};
    uint16_t m_lidarPort{40002};
    std::string m_cameraIP{"127.0.0.1"};
    uint16_t m_cameraBasePort{49252};
    uint32_t m_numCameras{4};
    bool m_enableLidar{true};
    bool m_enableCameras{true};
    bool m_asyncMode{true};

    // Display settings
    uint32_t m_windowWidth{1920};
    uint32_t m_windowHeight{1080};
    float m_pointSize{2.0f};

    // Statistics
    std::atomic<uint64_t> m_frameCount{0};
    std::chrono::steady_clock::time_point m_lastStatTime;
    float m_fps{0.0f};
};

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
FusionEngineApp::FusionEngineApp(const ProgramArguments& args)
    : DriveWorksSample(args)
    , m_fusionEngine(std::make_unique<FusionEngine>())
{
    parseArguments();
}

//------------------------------------------------------------------------------
// Parse command-line arguments
//------------------------------------------------------------------------------
void FusionEngineApp::parseArguments()
{
    // LiDAR settings
    m_lidarIP = getArgument("lidar-ip");
    if (m_lidarIP.empty())
    {
        m_lidarIP = "127.0.0.1";
    }

    std::string lidarPortStr = getArgument("lidar-port");
    if (!lidarPortStr.empty())
    {
        m_lidarPort = static_cast<uint16_t>(std::stoul(lidarPortStr));
    }

    // Camera settings
    m_cameraIP = getArgument("camera-ip");
    if (m_cameraIP.empty())
    {
        m_cameraIP = "127.0.0.1";
    }

    std::string cameraPortStr = getArgument("camera-port");
    if (!cameraPortStr.empty())
    {
        m_cameraBasePort = static_cast<uint16_t>(std::stoul(cameraPortStr));
    }

    std::string numCamerasStr = getArgument("num-cameras");
    if (!numCamerasStr.empty())
    {
        m_numCameras = static_cast<uint32_t>(std::stoul(numCamerasStr));
    }

    // Enable/disable sensors
    std::string enableLidarStr = getArgument("enable-lidar");
    if (!enableLidarStr.empty())
    {
        m_enableLidar = (enableLidarStr == "1" || enableLidarStr == "true");
    }

    std::string enableCamerasStr = getArgument("enable-cameras");
    if (!enableCamerasStr.empty())
    {
        m_enableCameras = (enableCamerasStr == "1" || enableCamerasStr == "true");
    }

    // Processing mode
    std::string asyncStr = getArgument("async");
    if (!asyncStr.empty())
    {
        m_asyncMode = (asyncStr == "1" || asyncStr == "true");
    }

    // Display settings
    std::string widthStr = getArgument("width");
    if (!widthStr.empty())
    {
        m_windowWidth = static_cast<uint32_t>(std::stoul(widthStr));
    }

    std::string heightStr = getArgument("height");
    if (!heightStr.empty())
    {
        m_windowHeight = static_cast<uint32_t>(std::stoul(heightStr));
    }
}

//------------------------------------------------------------------------------
// Initialize
//------------------------------------------------------------------------------
bool FusionEngineApp::onInitialize()
{
    std::cout << "========================================" << std::endl;
    std::cout << " FusionEngine - Multi-Sensor Fusion    " << std::endl;
    std::cout << "========================================" << std::endl;

    if (!initDriveWorks())
    {
        std::cerr << "Failed to initialize DriveWorks" << std::endl;
        return false;
    }

    if (!initVisualization())
    {
        std::cerr << "Failed to initialize visualization" << std::endl;
        return false;
    }

    if (!initRenderBuffers())
    {
        std::cerr << "Failed to initialize render buffers" << std::endl;
        return false;
    }

    if (!initFusionEngine())
    {
        std::cerr << "Failed to initialize fusion engine" << std::endl;
        return false;
    }

    m_lastStatTime = std::chrono::steady_clock::now();

    std::cout << "Initialization complete" << std::endl;
    return true;
}

//------------------------------------------------------------------------------
// Initialize DriveWorks
//------------------------------------------------------------------------------
bool FusionEngineApp::initDriveWorks()
{
    std::cout << "Initializing DriveWorks..." << std::endl;

    // Print SDK version
    int32_t major, minor, patch;
    dwGetVersionNumbers(&major, &minor, &patch);
    std::cout << "  DriveWorks SDK version: " << major << "." << minor << "."
              << patch << std::endl;

    // Create context
    dwContextParameters contextParams{};
    CHECK_DW_ERROR(dwInitialize(&m_context, DW_VERSION, &contextParams));

    return true;
}

//------------------------------------------------------------------------------
// Initialize Visualization
//------------------------------------------------------------------------------
bool FusionEngineApp::initVisualization()
{
    std::cout << "Initializing visualization..." << std::endl;

    // Initialize visualization context
    CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_context));

    // Create render engine
    dwRenderEngineParams renderParams{};
    renderParams.defaultTile.lineWidth = 2.0f;
    renderParams.defaultTile.font = DW_RENDER_ENGINE_FONT_VERDANA_16;
    renderParams.bounds.x = 0;
    renderParams.bounds.y = 0;
    renderParams.bounds.width = m_windowWidth;
    renderParams.bounds.height = m_windowHeight;

    CHECK_DW_ERROR(
        dwRenderEngine_initDefaultParams(&renderParams, m_windowWidth, m_windowHeight));
    CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &renderParams, m_viz));

    // Add main tile for 3D view
    dwRenderEngineTileState tileState{};
    dwRenderEngine_initTileState(&tileState);
    tileState.projectionType = DW_RENDER_ENGINE_TILE_PROJECTION_TYPE_3D;

    dwRenderEngineTileLayout tileLayout{};
    tileLayout.positionLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_ABSOLUTE;
    tileLayout.sizeLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_ABSOLUTE;
    tileLayout.positionType.absolutePosition = {0, 0};
    tileLayout.sizeType.absoluteSize = {m_windowWidth, m_windowHeight};

    CHECK_DW_ERROR(dwRenderEngine_addTile(&m_mainTile, &tileState, &tileLayout,
                                          m_renderEngine));

    // Initialize mouse view for camera control
    m_mouseView = std::make_unique<MouseView3D>();
    m_mouseView->setWindowAspect(
        static_cast<float>(m_windowWidth) / static_cast<float>(m_windowHeight));

    // Set initial camera position (bird's eye view)
    m_mouseView->setCenter(0.0f, 0.0f, 0.0f);
    m_mouseView->setAngleAndRadius(0.0f, -60.0f, 50.0f);

    return true;
}

//------------------------------------------------------------------------------
// Initialize Render Buffers
//------------------------------------------------------------------------------
bool FusionEngineApp::initRenderBuffers()
{
    std::cout << "Initializing render buffers..." << std::endl;

    // Point cloud buffer (for LiDAR points)
    dwRenderEngineParams params{};
    dwRenderEngine_initDefaultParams(&params, m_windowWidth, m_windowHeight);

    CHECK_DW_ERROR(dwRenderEngine_createBuffer(
        &m_pointCloudBufferId,
        DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
        MAX_LIDAR_POINTS,
        m_mainTile,
        m_renderEngine));

    // Detection bounding box lines buffer
    // Each 3D box = 12 edges, each edge = 2 vertices
    // Max detections * 12 edges * 2 = max lines
    CHECK_DW_ERROR(dwRenderEngine_createBuffer(
        &m_detectionLineBufferId,
        DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
        MAX_FUSED_DETECTIONS * 24,
        m_mainTile,
        m_renderEngine));

    return true;
}

//------------------------------------------------------------------------------
// Initialize Fusion Engine
//------------------------------------------------------------------------------
bool FusionEngineApp::initFusionEngine()
{
    std::cout << "Initializing fusion engine..." << std::endl;
    std::cout << "  LiDAR server: " << m_lidarIP << ":" << m_lidarPort
              << (m_enableLidar ? " (enabled)" : " (disabled)") << std::endl;
    std::cout << "  Camera server: " << m_cameraIP << ":" << m_cameraBasePort
              << "-" << (m_cameraBasePort + m_numCameras - 1)
              << (m_enableCameras ? " (enabled)" : " (disabled)") << std::endl;

    // Configure fusion engine
    m_fusionConfig.enableLidar = m_enableLidar;
    m_fusionConfig.lidarServerIP = m_lidarIP;
    m_fusionConfig.lidarServerPort = m_lidarPort;

    m_fusionConfig.enableCameras = m_enableCameras;
    m_fusionConfig.numCameras = m_numCameras;
    m_fusionConfig.cameraServerIP = m_cameraIP;
    m_fusionConfig.cameraServerBasePort = m_cameraBasePort;

    m_fusionConfig.asyncReceive = m_asyncMode;
    m_fusionConfig.connectionRetries = 30;
    m_fusionConfig.connectionTimeoutUs = 1000000;

    // Synchronization settings
    m_fusionConfig.syncConfig.enableLidar = m_enableLidar;
    m_fusionConfig.syncConfig.enableCameras = m_enableCameras;
    m_fusionConfig.syncConfig.numCameras = m_numCameras;
    m_fusionConfig.syncConfig.maxTimeDifferenceUs = 50000;  // 50ms
    m_fusionConfig.syncConfig.policy =
        SynchronizerConfig::FusionPolicy::LATEST_AVAILABLE;

    // Fusion settings
    m_fusionConfig.iouThreshold = 0.3f;
    m_fusionConfig.confidenceThreshold = 0.5f;

    // Initialize
    if (!m_fusionEngine->initialize(m_context, m_fusionConfig))
    {
        std::cerr << "Failed to initialize fusion engine" << std::endl;
        return false;
    }

    // Connect to servers
    std::cout << "Connecting to sensor servers..." << std::endl;
    if (!m_fusionEngine->connect())
    {
        std::cerr << "Warning: Failed to connect to all servers" << std::endl;
        // Continue anyway - some servers might come online later
    }

    // Set fusion callback
    m_fusionEngine->setFusionCallback([this](const FusedPacket& packet) {
        std::lock_guard<std::mutex> lock(m_packetMutex);
        m_currentPacket = packet;
        m_hasPacket = true;
    });

    // Start processing
    m_fusionEngine->start();

    return true;
}

//------------------------------------------------------------------------------
// Release
//------------------------------------------------------------------------------
void FusionEngineApp::onRelease()
{
    std::cout << "Releasing resources..." << std::endl;

    // Stop and release fusion engine
    if (m_fusionEngine)
    {
        m_fusionEngine->release();
    }

    // Destroy render buffers
    if (m_renderEngine != DW_NULL_HANDLE)
    {
        dwRenderEngine_destroyBuffer(m_pointCloudBufferId, m_renderEngine);
        dwRenderEngine_destroyBuffer(m_detectionLineBufferId, m_renderEngine);
    }

    // Release render engine
    if (m_renderEngine != DW_NULL_HANDLE)
    {
        dwRenderEngine_release(m_renderEngine);
    }

    // Release visualization
    if (m_viz != DW_NULL_HANDLE)
    {
        dwVisualizationRelease(m_viz);
    }

    // Release DriveWorks context
    if (m_context != DW_NULL_HANDLE)
    {
        dwRelease(m_context);
    }

    std::cout << "Release complete" << std::endl;
}

//------------------------------------------------------------------------------
// Process
//------------------------------------------------------------------------------
void FusionEngineApp::onProcess()
{
    // In synchronous mode, call process() to drive the fusion engine
    if (!m_asyncMode)
    {
        m_fusionEngine->process();
    }

    // Update FPS counter
    m_frameCount++;
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       now - m_lastStatTime)
                       .count();

    if (elapsed >= 1000)
    {
        m_fps = static_cast<float>(m_frameCount) * 1000.0f /
                static_cast<float>(elapsed);
        m_frameCount = 0;
        m_lastStatTime = now;
    }
}

//------------------------------------------------------------------------------
// Render
//------------------------------------------------------------------------------
void FusionEngineApp::onRender()
{
    // Get latest fused packet
    FusedPacket packet;
    {
        std::lock_guard<std::mutex> lock(m_packetMutex);
        if (m_hasPacket)
        {
            packet = m_currentPacket;
        }
    }

    // Begin rendering
    CHECK_DW_ERROR(dwRenderEngine_setTile(m_mainTile, m_renderEngine));

    // Set camera transformation from mouse view
    dwMatrix4f modelView = m_mouseView->getModelView();
    dwMatrix4f projection = m_mouseView->getProjection();
    CHECK_DW_ERROR(
        dwRenderEngine_setModelView(&modelView, m_renderEngine));
    CHECK_DW_ERROR(
        dwRenderEngine_setProjection(&projection, m_renderEngine));

    // Clear background
    CHECK_DW_ERROR(dwRenderEngine_setBackgroundColor(
        {0.1f, 0.1f, 0.1f, 1.0f}, m_renderEngine));

    // Render coordinate axes (small reference at origin)
    float axisLength = 2.0f;
    float axisVerts[] = {
        // X axis (red)
        0.0f, 0.0f, 0.0f, axisLength, 0.0f, 0.0f,
        // Y axis (green)
        0.0f, 0.0f, 0.0f, 0.0f, axisLength, 0.0f,
        // Z axis (blue)
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, axisLength};

    // Render LiDAR point cloud
    if (packet.valid && packet.lidarData.valid)
    {
        renderLidarPointCloud();
    }

    // Render fused detections
    if (packet.valid)
    {
        renderDetections();
    }

    // Render status overlay
    renderStatusOverlay();
}

//------------------------------------------------------------------------------
// Render LiDAR Point Cloud
//------------------------------------------------------------------------------
void FusionEngineApp::renderLidarPointCloud()
{
    FusedPacket packet;
    {
        std::lock_guard<std::mutex> lock(m_packetMutex);
        packet = m_currentPacket;
    }

    if (!packet.lidarData.valid || packet.lidarData.numPoints == 0)
    {
        return;
    }

    // Map point cloud buffer
    float* pointBuffer = nullptr;
    uint32_t maxPoints = 0;
    uint32_t stride = 0;

    CHECK_DW_ERROR(dwRenderEngine_mapBuffer(
        m_pointCloudBufferId,
        reinterpret_cast<void**>(&pointBuffer),
        0,
        &maxPoints,
        &stride,
        m_renderEngine));

    // Copy points (x, y, z only - skip intensity)
    uint32_t numPoints = std::min(packet.lidarData.numPoints, maxPoints);
    for (uint32_t i = 0; i < numPoints; ++i)
    {
        pointBuffer[i * 3 + 0] = packet.lidarData.points[i * 4 + 0];  // x
        pointBuffer[i * 3 + 1] = packet.lidarData.points[i * 4 + 1];  // y
        pointBuffer[i * 3 + 2] = packet.lidarData.points[i * 4 + 2];  // z
    }

    CHECK_DW_ERROR(dwRenderEngine_unmapBuffer(
        m_pointCloudBufferId, numPoints, m_renderEngine));

    // Render points in green
    CHECK_DW_ERROR(dwRenderEngine_setColor({0.0f, 1.0f, 0.0f, 0.8f},
                                           m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_setPointSize(m_pointSize, m_renderEngine));
    CHECK_DW_ERROR(
        dwRenderEngine_renderBuffer(m_pointCloudBufferId, numPoints, m_renderEngine));
}

//------------------------------------------------------------------------------
// Render Detections
//------------------------------------------------------------------------------
void FusionEngineApp::renderDetections()
{
    FusedPacket packet;
    {
        std::lock_guard<std::mutex> lock(m_packetMutex);
        packet = m_currentPacket;
    }

    if (packet.fusedDetections.empty())
    {
        return;
    }

    // Color coding by source type
    const float colorLidarOnly[] = {1.0f, 0.0f, 0.0f, 1.0f};   // Red
    const float colorCameraOnly[] = {0.0f, 0.0f, 1.0f, 1.0f};  // Blue
    const float colorFused[] = {1.0f, 1.0f, 0.0f, 1.0f};       // Yellow

    for (const auto& det : packet.fusedDetections)
    {
        const float* color;
        if (det.hasLidarSource && det.hasCameraSource)
        {
            color = colorFused;
        }
        else if (det.hasLidarSource)
        {
            color = colorLidarOnly;
        }
        else
        {
            color = colorCameraOnly;
        }

        render3DBox(det, color);
    }
}

//------------------------------------------------------------------------------
// Render 3D Bounding Box
//------------------------------------------------------------------------------
void FusionEngineApp::render3DBox(const FusedDetection& det, const float* color)
{
    // Box corners in local frame (centered at origin)
    float hw = det.width / 2.0f;
    float hl = det.length / 2.0f;
    float hh = det.height / 2.0f;

    // 8 corners: bottom 4, top 4
    float corners[8][3] = {
        {-hw, -hl, 0.0f},  // 0: bottom-left-front
        {hw, -hl, 0.0f},   // 1: bottom-right-front
        {hw, hl, 0.0f},    // 2: bottom-right-back
        {-hw, hl, 0.0f},   // 3: bottom-left-back
        {-hw, -hl, hh * 2},    // 4: top-left-front
        {hw, -hl, hh * 2},     // 5: top-right-front
        {hw, hl, hh * 2},      // 6: top-right-back
        {-hw, hl, hh * 2}      // 7: top-left-back
    };

    // Apply rotation around Z axis
    float cosR = std::cos(det.rotation);
    float sinR = std::sin(det.rotation);

    float transformed[8][3];
    for (int i = 0; i < 8; ++i)
    {
        float x = corners[i][0];
        float y = corners[i][1];

        transformed[i][0] = det.x + x * cosR - y * sinR;
        transformed[i][1] = det.y + x * sinR + y * cosR;
        transformed[i][2] = det.z + corners[i][2];
    }

    // 12 edges: 4 bottom, 4 top, 4 vertical
    int edges[12][2] = {
        // Bottom face
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        // Top face
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        // Vertical edges
        {0, 4}, {1, 5}, {2, 6}, {3, 7}};

    // Build line vertices
    float lineVerts[12 * 6];  // 12 edges * 2 vertices * 3 coords
    for (int i = 0; i < 12; ++i)
    {
        int v0 = edges[i][0];
        int v1 = edges[i][1];

        lineVerts[i * 6 + 0] = transformed[v0][0];
        lineVerts[i * 6 + 1] = transformed[v0][1];
        lineVerts[i * 6 + 2] = transformed[v0][2];
        lineVerts[i * 6 + 3] = transformed[v1][0];
        lineVerts[i * 6 + 4] = transformed[v1][1];
        lineVerts[i * 6 + 5] = transformed[v1][2];
    }

    // Map buffer and copy
    float* buffer = nullptr;
    uint32_t maxVerts = 0;
    uint32_t stride = 0;

    CHECK_DW_ERROR(dwRenderEngine_mapBuffer(
        m_detectionLineBufferId,
        reinterpret_cast<void**>(&buffer),
        0,
        &maxVerts,
        &stride,
        m_renderEngine));

    std::memcpy(buffer, lineVerts, sizeof(lineVerts));

    CHECK_DW_ERROR(
        dwRenderEngine_unmapBuffer(m_detectionLineBufferId, 24, m_renderEngine));

    // Render
    CHECK_DW_ERROR(dwRenderEngine_setColor(
        {color[0], color[1], color[2], color[3]}, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_setLineWidth(2.0f, m_renderEngine));
    CHECK_DW_ERROR(
        dwRenderEngine_renderBuffer(m_detectionLineBufferId, 24, m_renderEngine));
}

//------------------------------------------------------------------------------
// Render Status Overlay
//------------------------------------------------------------------------------
void FusionEngineApp::renderStatusOverlay()
{
    FusedPacket packet;
    {
        std::lock_guard<std::mutex> lock(m_packetMutex);
        packet = m_currentPacket;
    }

    // Build status text
    char statusText[1024];
    int offset = 0;

    offset += snprintf(statusText + offset, sizeof(statusText) - offset,
                       "FusionEngine Status\n");
    offset += snprintf(statusText + offset, sizeof(statusText) - offset,
                       "FPS: %.1f\n", m_fps);
    offset += snprintf(statusText + offset, sizeof(statusText) - offset,
                       "Fused Packets: %lu\n",
                       m_fusionEngine->getFusedPacketsProduced());

    // Sensor status
    if (m_enableLidar)
    {
        const char* lidarStatus = "Unknown";
        switch (m_fusionEngine->getLidarStatus())
        {
        case SensorStatus::DISCONNECTED:
            lidarStatus = "Disconnected";
            break;
        case SensorStatus::CONNECTING:
            lidarStatus = "Connecting";
            break;
        case SensorStatus::CONNECTED:
            lidarStatus = "Connected";
            break;
        case SensorStatus::RECEIVING:
            lidarStatus = "Receiving";
            break;
        case SensorStatus::ERROR:
            lidarStatus = "Error";
            break;
        }
        offset += snprintf(statusText + offset, sizeof(statusText) - offset,
                           "LiDAR: %s (pkts: %lu)\n", lidarStatus,
                           m_fusionEngine->getLidarPacketsReceived());
    }

    if (m_enableCameras)
    {
        for (uint32_t i = 0; i < m_numCameras; ++i)
        {
            const char* camStatus = "Unknown";
            switch (m_fusionEngine->getCameraStatus(i))
            {
            case SensorStatus::DISCONNECTED:
                camStatus = "Disc";
                break;
            case SensorStatus::CONNECTING:
                camStatus = "Conn...";
                break;
            case SensorStatus::CONNECTED:
                camStatus = "Conn";
                break;
            case SensorStatus::RECEIVING:
                camStatus = "Recv";
                break;
            case SensorStatus::ERROR:
                camStatus = "Err";
                break;
            }
            offset +=
                snprintf(statusText + offset, sizeof(statusText) - offset,
                         "Cam%d: %s (frames: %lu)\n", i, camStatus,
                         m_fusionEngine->getCameraFramesReceived(i));
        }
    }

    // Current packet info
    if (packet.valid)
    {
        offset += snprintf(statusText + offset, sizeof(statusText) - offset,
                           "\nCurrent Packet:\n");
        offset += snprintf(statusText + offset, sizeof(statusText) - offset,
                           "  Frame: %u\n", packet.fusionFrameNumber);
        offset += snprintf(statusText + offset, sizeof(statusText) - offset,
                           "  LiDAR Points: %u\n", packet.lidarData.numPoints);
        offset += snprintf(statusText + offset, sizeof(statusText) - offset,
                           "  Detections: %zu\n", packet.fusedDetections.size());
        offset += snprintf(statusText + offset, sizeof(statusText) - offset,
                           "    Fused: %u\n", packet.numFusedDetections);
        offset += snprintf(statusText + offset, sizeof(statusText) - offset,
                           "    LiDAR-only: %u\n", packet.numLidarOnlyDetections);
        offset += snprintf(statusText + offset, sizeof(statusText) - offset,
                           "    Camera-only: %u\n", packet.numCameraOnlyDetections);
    }

    // Render text
    CHECK_DW_ERROR(dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_renderText2D(statusText, {10.0f, 30.0f}, m_renderEngine));

    // Legend
    char legendText[256];
    snprintf(legendText, sizeof(legendText),
             "Legend:\n"
             "  Red: LiDAR-only\n"
             "  Blue: Camera-only\n"
             "  Yellow: Fused");

    CHECK_DW_ERROR(dwRenderEngine_renderText2D(
        legendText, {static_cast<float>(m_windowWidth) - 150.0f, 30.0f}, m_renderEngine));
}

//------------------------------------------------------------------------------
// Input Handlers
//------------------------------------------------------------------------------
void FusionEngineApp::onKeyDown(int key, int /*scancode*/, int /*mods*/)
{
    if (key == GLFW_KEY_ESCAPE)
    {
        stop();
    }
    else if (key == GLFW_KEY_R)
    {
        // Reset camera view
        m_mouseView->setCenter(0.0f, 0.0f, 0.0f);
        m_mouseView->setAngleAndRadius(0.0f, -60.0f, 50.0f);
    }
    else if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD)
    {
        m_pointSize = std::min(m_pointSize + 0.5f, 10.0f);
    }
    else if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT)
    {
        m_pointSize = std::max(m_pointSize - 0.5f, 1.0f);
    }
}

void FusionEngineApp::onMouseMove(float x, float y)
{
    m_mouseView->mouseMove(static_cast<int>(x), static_cast<int>(y));
}

void FusionEngineApp::onMouseDown(int button, float x, float y)
{
    m_mouseView->mouseDown(button, static_cast<int>(x), static_cast<int>(y));
}

void FusionEngineApp::onMouseUp(int button, float x, float y)
{
    m_mouseView->mouseUp(button, static_cast<int>(x), static_cast<int>(y));
}

void FusionEngineApp::onMouseWheel(float /*x*/, float y)
{
    m_mouseView->mouseWheel(static_cast<int>(y * 10.0f));
}

//------------------------------------------------------------------------------
// Main Entry Point
//------------------------------------------------------------------------------
int main(int argc, const char** argv)
{
    // Define program arguments
    ProgramArguments args(argc, argv,
    {
        // LiDAR settings
        ProgramArguments::Option_t{"lidar-ip", "127.0.0.1",
            "LiDAR server IP address"},
        ProgramArguments::Option_t{"lidar-port", "40002",
            "LiDAR server port"},
        ProgramArguments::Option_t{"enable-lidar", "1",
            "Enable LiDAR input (0/1)"},

        // Camera settings
        ProgramArguments::Option_t{"camera-ip", "127.0.0.1",
            "Camera server IP address"},
        ProgramArguments::Option_t{"camera-port", "49252",
            "Camera server base port"},
        ProgramArguments::Option_t{"num-cameras", "4",
            "Number of cameras (1-4)"},
        ProgramArguments::Option_t{"enable-cameras", "1",
            "Enable camera input (0/1)"},

        // Processing settings
        ProgramArguments::Option_t{"async", "1",
            "Async receive mode (0/1)"},

        // Display settings
        ProgramArguments::Option_t{"width", "1920",
            "Window width"},
        ProgramArguments::Option_t{"height", "1080",
            "Window height"},
        ProgramArguments::Option_t{"offscreen", "0",
            "Offscreen rendering (0/1)"}
    });

    // Create and run application
    FusionEngineApp app(args);

    app.initializeWindow("FusionEngine - Multi-Sensor Fusion",
                         std::stoul(args.get("width")),
                         std::stoul(args.get("height")),
                         args.get("offscreen") == "1");

    return app.run();
}
