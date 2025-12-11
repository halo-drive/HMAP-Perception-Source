#include <cstdint>
#include <cstdlib>
#include <string>
#include <iostream>
#include <vector>
#include <sstream>
#include <cmath>

#include <framework/ProgramArguments.hpp>
#include <framework/WindowGLFW.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/Checks.hpp>
#include <framework/MouseView3D.hpp>

#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/logger/Logger.h>
#include <dw/core/context/Context.h>
#include <dw/image/Image.h>
#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Visualization.h>
#include <dwcgf/channel/ChannelFactory.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwframework/dwnodes/common/factories/DWChannelFactory.hpp>

#include "DetectionPacket.hpp"
#include "DetectionPacket.cpp"
#include <framework/Mat4.hpp>

using namespace dw_samples::common;

// Consumer application: receives data via socket and visualizes
class DetectionConsumer : public DriveWorksSample
{
private:
    dwContextHandle_t m_context = DW_NULL_HANDLE;
    dwVisualizationContextHandle_t m_viz = DW_NULL_HANDLE;
    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
    
    // Channel components for IPC
    std::unique_ptr<dw::framework::DWChannelFactory> m_channelFactory{};
    std::unique_ptr<dw::framework::PortInput<DetectionPacket>> m_inputPort{};
    std::shared_ptr<dw::framework::ChannelObject> m_channel{};
    
    // IPC parameters
    std::string m_ipAddr{"127.0.0.1"};
    std::string m_portId{"40002"};
    
    // Received data
    DetectionPacket m_currentPacket{};
    bool m_hasData{false};
    
    // Rendering buffers
    uint32_t m_pointCloudRenderBufferId{0};
    uint32_t m_groundPlaneRenderBufferId{0};
    uint32_t m_boxLineBuffer{0};
    uint32_t m_freeSpaceRenderBufferId{0};
    
    // Window tile
    uint32_t m_tileId{0};
    
public:
    DetectionConsumer(const ProgramArguments& args) : DriveWorksSample(args)
    {
        m_ipAddr = args.get("ip");
        m_portId = args.get("port");
    }
    
    bool initializeIPC()
    {
        std::cout << "Initializing IPC consumer..." << std::endl;
        std::cout << "  IP: " << m_ipAddr << std::endl;
        std::cout << "  Port: " << m_portId << std::endl;
        
        // Create channel factory
        m_channelFactory = std::make_unique<dw::framework::DWChannelFactory>(m_context);
        
        // Create channel parameters
        std::stringstream channelParam;
        channelParam << "role=consumer"
                     << ",type=SOCKET"
                     << ",ip=" << m_ipAddr
                     << ",id=" << m_portId
                     << ",uid=1"
                     << ",fifo-size=10"
                     << ",connect-timeout=5000";
        
        // Create channel
        m_channel = m_channelFactory->makeChannel(channelParam.str().c_str());
        
        // Create input port with specimen
        DetectionPacket specimen{};
        m_inputPort = std::make_unique<dw::framework::PortInput<DetectionPacket>>(specimen);
        
        // Bind port to channel
        CHECK_DW_ERROR(m_inputPort->bindChannel(m_channel.get()));
        
        // Start channel services
        m_channelFactory->startServices();
        
        std::cout << "IPC consumer initialized, waiting for connection..." << std::endl;
        return true;
    }
    
    bool receiveDetectionResults()
    {
        if (!m_inputPort || !m_inputPort->isBound()) {
            return false;
        }
        
        // Wait for data with timeout
        dwStatus status = m_inputPort->wait(100000);  // 100ms timeout
        if (status != DW_SUCCESS) {
            return false;
        }
        
        // Receive packet
        std::shared_ptr<DetectionPacket> packet = m_inputPort->recv();
        if (packet == nullptr) {
            return false;
        }
        
        // Copy to current packet
        m_currentPacket = *packet;
        m_hasData = true;
        
        return true;
    }
    
    void initRendering()
    {
        dwRenderEngineParams params{};
        CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&params,
                                                        static_cast<uint32_t>(getWindowWidth()),
                                                        static_cast<uint32_t>(getWindowHeight())));
        CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_viz));
        
        CHECK_DW_ERROR(dwRenderEngine_initTileState(&params.defaultTile));
        
        dwRenderEngineTileState tileParam = params.defaultTile;
        tileParam.layout.viewport = {0.f, 0.f, 1.0f, 1.0f};
        tileParam.layout.positionType = DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_LEFT;
        CHECK_DW_ERROR(dwRenderEngine_addTile(&m_tileId, &tileParam, m_renderEngine));
        
        // Create render buffers
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_pointCloudRenderBufferId,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                   sizeof(dwVector4f),
                                                   0,
                                                   DetectionPacket::MAX_POINTS,
                                                   m_renderEngine));
        
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_groundPlaneRenderBufferId,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
                                                   sizeof(dwVector3f),
                                                   0,
                                                   50 * 50 * 6,  // Grid size
                                                   m_renderEngine));
        
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_boxLineBuffer,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                                   sizeof(dwVector3f),
                                                   0,
                                                   100 * 24,  // Max boxes * vertices per box
                                                   m_renderEngine));
        
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_freeSpaceRenderBufferId,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                   sizeof(dwVector3f),
                                                   0,
                                                   50000,
                                                   m_renderEngine));
        
        std::cout << "Rendering initialized" << std::endl;
    }
    
    void renderPointCloud()
    {
        if (!m_hasData || m_currentPacket.numPoints == 0) {
            return;
        }
        
        dwVector4f* vertices = nullptr;
        uint32_t bufferSize = m_currentPacket.numPoints * sizeof(dwVector4f);
        
        CHECK_DW_ERROR(dwRenderEngine_mapBuffer(m_pointCloudRenderBufferId,
                                                reinterpret_cast<void**>(&vertices),
                                                0,
                                                bufferSize,
                                                DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                m_renderEngine));
        
        for (uint32_t i = 0; i < m_currentPacket.numPoints; ++i) {
            vertices[i].x = m_currentPacket.points[i * 4 + 0];
            vertices[i].y = m_currentPacket.points[i * 4 + 1];
            vertices[i].z = m_currentPacket.points[i * 4 + 2];
            vertices[i].w = m_currentPacket.points[i * 4 + 3];  // intensity
        }
        
        CHECK_DW_ERROR(dwRenderEngine_unmapBuffer(m_pointCloudRenderBufferId,
                                                  DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                  m_renderEngine));
        
        dwRenderEngine_setColor({0.0f, 1.0f, 0.0f, 1.0f}, m_renderEngine);
        dwRenderEngine_setPointSize(1.0f, m_renderEngine);
        dwRenderEngine_renderBuffer(m_pointCloudRenderBufferId, m_currentPacket.numPoints, m_renderEngine);
    }
    
    void renderBoundingBoxes()
    {
        if (!m_hasData || m_currentPacket.numDetections == 0) {
            return;
        }
        
        dwVector3f* vertices = nullptr;
        uint32_t expectedVertices = m_currentPacket.numDetections * 24;  // 12 edges * 2 vertices
        uint32_t bufferSize = expectedVertices * sizeof(dwVector3f);
        
        CHECK_DW_ERROR(dwRenderEngine_mapBuffer(m_boxLineBuffer,
                                                reinterpret_cast<void**>(&vertices),
                                                0,
                                                bufferSize,
                                                DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                                m_renderEngine));
        
        uint32_t vertexIndex = 0;
        for (uint32_t b = 0; b < m_currentPacket.numDetections; ++b) {
            const auto& box = m_currentPacket.boxes[b];
            
            float halfWidth = box.width / 2.0f;
            float halfLength = box.length / 2.0f;
            float halfHeight = box.height / 2.0f;
            
            float cosYaw = cos(box.rotation);
            float sinYaw = sin(box.rotation);
            
            // 8 corners (local coordinates)
            struct LocalCorner { float x, y, z; };
            LocalCorner localCorners[8] = {
                {-halfWidth, -halfLength, -halfHeight},
                { halfWidth, -halfLength, -halfHeight},
                { halfWidth,  halfLength, -halfHeight},
                {-halfWidth,  halfLength, -halfHeight},
                {-halfWidth, -halfLength,  halfHeight},
                { halfWidth, -halfLength,  halfHeight},
                { halfWidth,  halfLength,  halfHeight},
                {-halfWidth,  halfLength,  halfHeight}
            };
            
            // Apply rotation and translation to get world coordinates
            dwVector3f worldCorners[8];
            for (int i = 0; i < 8; ++i) {
                float rx = localCorners[i].x * cosYaw - localCorners[i].y * sinYaw;
                float ry = localCorners[i].x * sinYaw + localCorners[i].y * cosYaw;
                worldCorners[i] = {box.x + rx, box.y + ry, box.z + localCorners[i].z};
            }
            
            // 12 edges
            int edges[12][2] = {
                {0,1}, {1,2}, {2,3}, {3,0},  // Bottom
                {4,5}, {5,6}, {6,7}, {7,4},  // Top
                {0,4}, {1,5}, {2,6}, {3,7}   // Vertical
            };
            
            for (int i = 0; i < 12; ++i) {
                vertices[vertexIndex++] = worldCorners[edges[i][0]];
                vertices[vertexIndex++] = worldCorners[edges[i][1]];
            }
        }
        
        CHECK_DW_ERROR(dwRenderEngine_unmapBuffer(m_boxLineBuffer,
                                                  DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                                  m_renderEngine));
        
        // Render with class-specific colors
        dwRenderEngine_setLineWidth(2.0f, m_renderEngine);
        
        // Render each box with its class color
        uint32_t vertexOffset = 0;
        for (uint32_t b = 0; b < m_currentPacket.numDetections; ++b) {
            dwRenderEngineColorRGBA color;
            switch (m_currentPacket.boxes[b].classId) {
                case 0: color = {1.0f, 0.0f, 0.0f, 1.0f}; break;  // Red for vehicles
                case 1: color = {0.0f, 1.0f, 0.0f, 1.0f}; break;  // Green for pedestrians
                case 2: color = {0.0f, 0.0f, 1.0f, 1.0f}; break;  // Blue for cyclists
                default: color = {1.0f, 1.0f, 1.0f, 1.0f}; break;
            }
            dwRenderEngine_setColor(color, m_renderEngine);
            // Render 24 vertices per box (12 edges * 2 vertices)
            dwRenderEngine_renderBuffer(m_boxLineBuffer, 24, m_renderEngine);
            vertexOffset += 24;
        }
    }
    
    bool onInitialize() override
    {
        // Initialize DriveWorks
        CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
        CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_INFO));
        
        dwContextParameters sdkParams = {};
        CHECK_DW_ERROR(dwInitialize(&m_context, DW_VERSION, &sdkParams));
        
        CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_context));
        
        // Initialize IPC
        if (!initializeIPC()) {
            return false;
        }
        
        // Initialize rendering
        initRendering();
        
        return true;
    }
    
    void onProcess() override
    {
        // Try to receive new data
        receiveDetectionResults();
    }
    
    void onRender() override
    {
        dwRenderEngine_reset(m_renderEngine);
        dwRenderEngine_setTile(m_tileId, m_renderEngine);
        
        // Set up 3D view
        dwMatrix4f modelView;
        const dwMatrix4f* mv = getMouseView().getModelView();
        Mat4_AxB(modelView.array, mv->array, DW_IDENTITY_TRANSFORMATION3F.array);
        dwRenderEngine_setModelView(&modelView, m_renderEngine);
        const dwMatrix4f* proj = getMouseView().getProjection();
        dwRenderEngine_setProjection(proj, m_renderEngine);
        dwRenderEngine_setBackgroundColor({0.1f, 0.1f, 0.1f, 1.0f}, m_renderEngine);
        
        // Render point cloud
        renderPointCloud();
        
        // Render bounding boxes
        renderBoundingBoxes();
        
        // Render text info
        dwRenderEngine_setTile(m_tileId, m_renderEngine);
        dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
        dwRenderEngine_setCoordinateRange2D({1.0f, 1.0f}, m_renderEngine);
        
        if (m_hasData) {
            char info[256];
            snprintf(info, sizeof(info), "Frame: %u | Points: %u | Detections: %u | Aligned: %s",
                     m_currentPacket.frameNumber,
                     m_currentPacket.numPoints,
                     m_currentPacket.numDetections,
                     m_currentPacket.icpAligned ? "YES" : "NO");
            
            dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine);
            dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, m_renderEngine);
            dwRenderEngine_renderText2D(info, {0.1f, 0.9f}, m_renderEngine);
        } else {
            dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine);
            dwRenderEngine_setColor({1.0f, 0.0f, 0.0f, 1.0f}, m_renderEngine);
            dwRenderEngine_renderText2D("Waiting for data from producer...", {0.1f, 0.9f}, m_renderEngine);
        }
    }
    
    void onRelease() override
    {
        if (m_inputPort) {
            m_inputPort.reset();
        }
        if (m_channelFactory) {
            m_channelFactory->stopServices();
            m_channelFactory.reset();
        }
        
        if (m_renderEngine) {
            dwRenderEngine_destroyBuffer(m_pointCloudRenderBufferId, m_renderEngine);
            dwRenderEngine_destroyBuffer(m_groundPlaneRenderBufferId, m_renderEngine);
            dwRenderEngine_destroyBuffer(m_boxLineBuffer, m_renderEngine);
            dwRenderEngine_destroyBuffer(m_freeSpaceRenderBufferId, m_renderEngine);
            dwRenderEngine_release(m_renderEngine);
        }
        
        if (m_viz) {
            dwVisualizationRelease(m_viz);
        }
        if (m_context) {
            dwRelease(m_context);
        }
        
        dwLogger_release();
    }
};

//------------------------------------------------------------------------------
int32_t main(int32_t argc, const char** argv)
{
    typedef ProgramArguments::Option_t opt;

    ProgramArguments args(argc, argv,
                          {
                              // IPC parameters
                              opt("ip", "127.0.0.1", "IP address for socket communication"),
                              opt("port", "40002", "Port ID for socket communication"),
                              
                              // Display parameters
                              opt("displayWindowHeight", "900", "Display window height"),
                              opt("displayWindowWidth", "1500", "Display window width")
                          });

    if (!args.parse(argc, argv)) {
        std::cerr << "Failed to parse command line arguments" << std::endl;
        return -1;
    }

    if (args.has("help")) {
        std::cout << "Detection Consumer Application" << std::endl;
        std::cout << "Receives detection results via socket and visualizes them" << std::endl;
        std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
        std::cout << std::endl;
        std::cout << "IPC Options:" << std::endl;
        std::cout << "  --ip=ADDRESS      IP address for socket (default: 127.0.0.1)" << std::endl;
        std::cout << "  --port=ID         Port ID for socket (default: 40002)" << std::endl;
        return 0;
    }

    std::cout << "=== Detection Consumer (Visualization) ===" << std::endl;
    std::cout << "IP: " << args.get("ip") << std::endl;
    std::cout << "Port: " << args.get("port") << std::endl;
    std::cout << "===========================================" << std::endl;

    try {
        DetectionConsumer app(args);
        
        int32_t width = std::atoi(args.get("displayWindowWidth").c_str());
        int32_t height = std::atoi(args.get("displayWindowHeight").c_str());
        
        app.initializeWindow("Detection Consumer", width, height, args.enabled("offscreen"));
        
        if (!app.onInitialize()) {
            std::cerr << "Failed to initialize application" << std::endl;
            return -1;
        }
        
        std::cout << "Starting consumer visualization..." << std::endl;
        std::cout << "Waiting for data from producer..." << std::endl;
        
        return app.run();
        
    } catch (const std::exception& e) {
        std::cerr << "Application error: " << e.what() << std::endl;
        return -1;
    }
}

