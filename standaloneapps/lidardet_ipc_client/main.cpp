#include <cstdint>
#include <cstdlib>
#include <string>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <unistd.h>
#include <csignal>

#include <framework/ProgramArguments.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/Checks.hpp>
#include <framework/WindowGLFW.hpp>
#include <framework/Mat4.hpp>

#include <dw/core/base/Version.h>
#include <dw/core/logger/Logger.h>
#include <dw/core/context/Context.h>
#include <dw/comms/socketipc/SocketClientServer.h>

#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Visualization.h>

#include <cuda_runtime_api.h>

#include "../lidar_object_detection_interprocess_communiation/DetectionPacket.hpp"

using namespace dw_samples::common;

// Consumer application: receives detection data and visualizes it
class DetectionConsumer : public DriveWorksSample
{
private:
    // DriveWorks context
    dwContextHandle_t m_context{DW_NULL_HANDLE};
    
    // Socket IPC components
    dwSocketClientHandle_t m_socketClient{DW_NULL_HANDLE};
    dwSocketConnectionHandle_t m_socketConnection{DW_NULL_HANDLE};
    
    // IPC parameters
    std::string m_serverIP{"127.0.0.1"};
    uint16_t m_port{40002};
    std::atomic<bool> m_connected{false};
    
    // Received data
    DetectionPacket m_currentPacket{};
    bool m_hasData{false};
    uint32_t m_packetCount{0};
    
    // Rendering components
    dwVisualizationContextHandle_t m_viz{DW_NULL_HANDLE};
    dwRenderEngineHandle_t m_renderEngine{DW_NULL_HANDLE};
    uint32_t m_tile{0};
    
    // Render buffers
    uint32_t m_pointCloudBufferId{0};
    uint32_t m_boxLineBufferId{0};
    uint32_t m_freeSpaceBufferId{0};
    uint32_t m_groundPlaneBufferId{0};

public:
    DetectionConsumer(const ProgramArguments& args) : DriveWorksSample(args)
    {
        m_serverIP = args.get("ip");
        m_port = static_cast<uint16_t>(std::stoul(args.get("port")));
    }
    
    bool initializeIPC()
    {
        // Guard against double initialization
        if (m_socketClient != DW_NULL_HANDLE) {
            std::cout << "IPC consumer already initialized, skipping..." << std::endl;
            return true;
        }
        
        std::cout << "Initializing IPC consumer (Socket Client)..." << std::endl;
        std::cout << "  Server IP: " << m_serverIP << std::endl;
        std::cout << "  Port: " << m_port << std::endl;
        
        // Initialize socket client
        CHECK_DW_ERROR(dwSocketClient_initialize(&m_socketClient, 1, m_context));
        
        std::cout << "Attempting to connect to producer..." << std::endl;
        
        // Try to connect (with timeout)
        dwStatus status = DW_TIME_OUT;
        int retryCount = 0;
        while (status == DW_TIME_OUT && retryCount < 30) {
            status = dwSocketClient_connect(&m_socketConnection, m_serverIP.c_str(), 
                                           m_port, 1000000, m_socketClient);
            if (status == DW_TIME_OUT) {
                retryCount++;
                std::cout << "Connection attempt " << retryCount << "/30..." << std::endl;
            }
        }
        
        if (status == DW_SUCCESS) {
            m_connected = true;
            std::cout << "Consumer: Connected to producer!" << std::endl;
            return true;
        } else {
            std::cerr << "Consumer: Failed to connect: " << dwGetStatusName(status) << std::endl;
            return false;
        }
    }
    
    bool receiveDetectionResults()
    {
        if (!m_connected) {
            return false;
        }
        
        // Receive packet in chunks (TCP may break large packets)
        const size_t totalSize = sizeof(DetectionPacket);
        size_t bytesReceived = 0;
        uint8_t* buffer = reinterpret_cast<uint8_t*>(&m_currentPacket);
        
        while (bytesReceived < totalSize) {
            size_t chunkSize = totalSize - bytesReceived;
            dwStatus recvStatus = dwSocketConnection_read(buffer + bytesReceived, 
                                                          &chunkSize, 
                                                          1000000,  // 1 second timeout per chunk
                                                          m_socketConnection);
            
            if (recvStatus == DW_END_OF_STREAM) {
                std::cerr << "Consumer: Producer disconnected" << std::endl;
                m_connected = false;
                stop();
                return false;
            } else if (recvStatus == DW_TIME_OUT) {
                static uint32_t timeoutCount = 0;
                if (++timeoutCount == 1 || timeoutCount % 100 == 0) {
                    std::cout << "Consumer: Timeout waiting for data (count: " << timeoutCount << ")" << std::endl;
                }
                return false;
            } else if (recvStatus != DW_SUCCESS) {
                static uint32_t recvErrorCount = 0;
                if (++recvErrorCount <= 5) {
                    std::cerr << "Consumer: Failed to receive chunk: " << dwGetStatusName(recvStatus) << std::endl;
                }
                return false;
            }
            
            bytesReceived += chunkSize;
            
            // Log progress for large packets
            if (bytesReceived < totalSize) {
                static uint32_t progressCount = 0;
                if (++progressCount % 10 == 0) {
                    std::cout << "Consumer: Receiving packet... " << bytesReceived << "/" << totalSize 
                              << " bytes (" << (bytesReceived * 100 / totalSize) << "%)" << std::endl;
                }
            }
        }
        
        // Validate received data
        if (bytesReceived != totalSize) {
            std::cerr << "Consumer: Incomplete packet (expected " << totalSize 
                      << " bytes, got " << bytesReceived << " bytes)" << std::endl;
            return false;
        }
        
        // Log successful receive
        m_packetCount++;
        if (m_packetCount == 1 || m_packetCount % 30 == 0) {
            std::cout << "Consumer: Received packet #" << m_packetCount 
                      << " (frame=" << m_currentPacket.frameNumber 
                      << ", points=" << m_currentPacket.numPoints 
                      << ", detections=" << m_currentPacket.numDetections 
                      << ", size=" << (totalSize / 1024) << "KB)" << std::endl;
        }
        
        m_hasData = true;
        return true;
    }
    
    // DriveWorksSample overrides
    bool onInitialize() override
    {
        // Initialize DW context
        CHECK_DW_ERROR(dwInitialize(&m_context, DW_VERSION, nullptr));
        
        // Initialize visualization
        CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_context));
        
        // Initialize render engine
        dwRenderEngineParams params{};
        CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&params,
                                                        static_cast<uint32_t>(getWindowWidth()),
                                                        static_cast<uint32_t>(getWindowHeight())));
        CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_viz));
        
        // Initialize tile
        CHECK_DW_ERROR(dwRenderEngine_initTileState(&params.defaultTile));
        dwRenderEngineTileState tileParam = params.defaultTile;
        tileParam.layout.viewport = {0.f, 0.f, 1.f, 1.f};
        tileParam.layout.sizeLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_RELATIVE;
        tileParam.layout.positionLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_RELATIVE;
        tileParam.layout.positionType = DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_LEFT;
        CHECK_DW_ERROR(dwRenderEngine_addTile(&m_tile, &tileParam, m_renderEngine));
        
        // Create render buffers
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_pointCloudBufferId,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                   sizeof(dwVector4f),
                                                   0,
                                                   DetectionPacket::MAX_POINTS,
                                                   m_renderEngine));
        
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_boxLineBufferId,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                                   sizeof(dwVector3f),
                                                   0,
                                                   MAX_DETECTIONS * 24,  // 12 edges * 2 vertices per box
                                                   m_renderEngine));
        
        // Free space buffer (MUCH larger to handle all free space points)
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_freeSpaceBufferId,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                   sizeof(dwVector3f),
                                                   0,
                                                   50000,  // Match original capacity
                                                   m_renderEngine));
        
        // Ground plane buffer (grid mesh)
        const uint32_t GRID_SIZE = 21;
        const uint32_t gridVertices = (GRID_SIZE - 1) * (GRID_SIZE - 1) * 6;  // 2 triangles per cell
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_groundPlaneBufferId,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
                                                   sizeof(dwVector3f),
                                                   0,
                                                   gridVertices,
                                                   m_renderEngine));
        
        // Initialize mouse view for camera control (inherited from DriveWorksSample)
        getMouseView().setCenter(0.0f, 0.0f, 0.0f);
        
        std::cout << "Rendering initialized successfully" << std::endl;
        
        // Initialize IPC
        if (!initializeIPC()) {
            return false;
        }
        
        return true;
    }
    
    void onProcess() override
    {
        // Receive new data
        receiveDetectionResults();
    }
    
    void onRender() override
    {
        if (!m_hasData) {
            return;
        }
        
        // Clear buffer and set center (EXACT as original)
        dwRenderEngine_reset(m_renderEngine);
        getMouseView().setCenter(0.0f, 0.0f, 0.0f);
        
        // Set up rendering with mouse-controlled camera (EXACT as original)
        dwRenderEngine_setTile(m_tile, m_renderEngine);
        dwRenderEngine_setBackgroundColor({0.1f, 0.1f, 0.1f, 1.0f}, m_renderEngine);
        
        dwMatrix4f modelView;
        Mat4_AxB(modelView.array, getMouseView().getModelView()->array, DW_IDENTITY_TRANSFORMATION3F.array);
        dwRenderEngine_setModelView(&modelView, m_renderEngine);
        dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine);
        
        // Render point cloud first (EXACT order as original)
        if (m_currentPacket.numPoints > 0) {
            renderPointCloud();
        }
        
        // Render ground plane (EXACT as original - set tile and camera before rendering)
        if (m_currentPacket.groundPlane.valid) {
            dwRenderEngine_setTile(m_tile, m_renderEngine);
            Mat4_AxB(modelView.array, getMouseView().getModelView()->array, DW_IDENTITY_TRANSFORMATION3F.array);
            dwRenderEngine_setModelView(&modelView, m_renderEngine);
            dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine);
            renderGroundPlane();
        }
        
        // Render bounding boxes (EXACT as original - set tile before rendering)
        if (m_currentPacket.numDetections > 0) {
            dwRenderEngine_setTile(m_tile, m_renderEngine);
            renderBoundingBoxes();
        }
        
        // Render free space
        if (m_currentPacket.freeSpace.numPoints > 0) {
            dwRenderEngine_setTile(m_tile, m_renderEngine);
            Mat4_AxB(modelView.array, getMouseView().getModelView()->array, DW_IDENTITY_TRANSFORMATION3F.array);
            dwRenderEngine_setModelView(&modelView, m_renderEngine);
            dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine);
            renderFreeSpace();
        }
        
        // Render status text
        renderStatusText();
        
        // CUDA sync at end of frame (like original)
        #ifdef __aarch64__
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        #endif
    }
    
    void onRelease() override
    {
        // Clean up IPC
        if (m_socketConnection != DW_NULL_HANDLE) {
            dwSocketConnection_release(m_socketConnection);
            m_socketConnection = DW_NULL_HANDLE;
        }
        if (m_socketClient != DW_NULL_HANDLE) {
            dwSocketClient_release(m_socketClient);
            m_socketClient = DW_NULL_HANDLE;
        }
        
        // Clean up rendering
        if (m_renderEngine != DW_NULL_HANDLE) {
            if (m_pointCloudBufferId != 0) {
                dwRenderEngine_destroyBuffer(m_pointCloudBufferId, m_renderEngine);
            }
            if (m_boxLineBufferId != 0) {
                dwRenderEngine_destroyBuffer(m_boxLineBufferId, m_renderEngine);
            }
            if (m_freeSpaceBufferId != 0) {
                dwRenderEngine_destroyBuffer(m_freeSpaceBufferId, m_renderEngine);
            }
            if (m_groundPlaneBufferId != 0) {
                dwRenderEngine_destroyBuffer(m_groundPlaneBufferId, m_renderEngine);
            }
            dwRenderEngine_release(m_renderEngine);
        }
        
        if (m_viz != DW_NULL_HANDLE) {
            dwVisualizationRelease(m_viz);
        }
        
        // Release DW context
        if (m_context != DW_NULL_HANDLE) {
            dwRelease(m_context);
        }
    }
    
private:
    void renderPointCloud()
    {
        // Use exact same technique as original with CUDA sync
        uint32_t sizeInBytes = m_currentPacket.numPoints * sizeof(dwVector4f);
        dwVector4f* dataToRender = nullptr;
        
        // CUDA synchronization (like original)
        #ifdef __aarch64__
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        #endif
        
        CHECK_DW_ERROR(dwRenderEngine_mapBuffer(m_pointCloudBufferId,
                                               reinterpret_cast<void**>(&dataToRender),
                                               0,
                                               sizeInBytes,
                                               DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                               m_renderEngine));
        
        // Clear buffer first on Orin (like original)
        #ifdef __aarch64__
        memset(dataToRender, 0, sizeInBytes);
        #endif
        
        // Copy points with intensity (EXACT as original)
        for (uint32_t i = 0; i < m_currentPacket.numPoints; ++i) {
            dataToRender[i].x = m_currentPacket.points[i * 4 + 0];
            dataToRender[i].y = m_currentPacket.points[i * 4 + 1];
            dataToRender[i].z = m_currentPacket.points[i * 4 + 2];
            dataToRender[i].w = m_currentPacket.points[i * 4 + 3];  // Keep original intensity
        }
        
        CHECK_DW_ERROR(dwRenderEngine_unmapBuffer(m_pointCloudBufferId,
                                                  DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                  m_renderEngine));
        
        // Render in RED (like original aligned/stitched points)
        dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_RED, m_renderEngine);
        dwRenderEngine_setPointSize(1.0f, m_renderEngine);
        CHECK_DW_ERROR(dwRenderEngine_renderBuffer(m_pointCloudBufferId, m_currentPacket.numPoints, m_renderEngine));
    }
    
    void renderBoundingBoxes()
    {
        // Prepare text labels (EXACT as original)
        std::vector<std::pair<dwVector3f, std::string>> labelTexts;    // Above box - class & confidence
        std::vector<std::pair<dwVector3f, std::string>> debugTexts;    // Below box - point count & distance
        
        // Group boxes by class (EXACT as original)
        std::vector<DetectionBoundingBox> vehicleBoxes;
        std::vector<DetectionBoundingBox> pedestrianBoxes;
        std::vector<DetectionBoundingBox> cyclistBoxes;
        
        for (uint32_t i = 0; i < m_currentPacket.numDetections; ++i) {
            const auto& box = m_currentPacket.boxes[i];
            switch (box.classId) {
                case 0: vehicleBoxes.push_back(box); break;
                case 1: pedestrianBoxes.push_back(box); break;
                case 2: cyclistBoxes.push_back(box); break;
            }
            
            // Prepare text labels (EXACT as original)
            float halfHeight = box.height / 2.0f;
            dwVector3f labelPos = {box.x, box.y, box.z + halfHeight + 0.3f};
            dwVector3f debugPos = {box.x, box.y, box.z - halfHeight - 0.3f};
            
            // Class name
            std::string className;
            switch (box.classId) {
                case 0: className = "Vehicle"; break;
                case 1: className = "Pedestrian"; break;
                case 2: className = "Cyclist"; break;
                default: className = "Unknown"; break;
            }
            
            // Count points and distance
            int pointCount = countPointsInBox(box);
            float distance = std::sqrt(box.x * box.x + box.y * box.y);
            
            // Label text
            char labelText[64];
            snprintf(labelText, sizeof(labelText), "%s (%.2f)", className.c_str(), box.confidence);
            labelTexts.emplace_back(labelPos, std::string(labelText));
            
            // Debug text
            char debugText[64];
            snprintf(debugText, sizeof(debugText), "Pts: %d, Dist: %.1fm", pointCount, distance);
            debugTexts.emplace_back(debugPos, std::string(debugText));
        }
        
        // Render each class with its color (EXACT as original with proper buffer clearing)
        auto renderBoxGroup = [&](const std::vector<DetectionBoundingBox>& boxes, dwRenderEngineColorRGBA color) {
            if (boxes.empty()) return;
            
            uint32_t expectedVertices = boxes.size() * 24;  // 12 edges × 2 vertices per box
            uint32_t bufferSizeBytes = expectedVertices * sizeof(dwVector3f);
            
            // CUDA sync before mapping (like original)
            #ifdef __aarch64__
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            #endif
            
            // Map buffer (EXACT as original)
            dwVector3f* vertices = nullptr;
            CHECK_DW_ERROR(dwRenderEngine_mapBuffer(m_boxLineBufferId,
                                                   reinterpret_cast<void**>(&vertices),
                                                   0,
                                                   bufferSizeBytes,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                                   m_renderEngine));
            
            // Clear buffer on Orin (like original - prevents weird lines)
            #ifdef __aarch64__
            memset(vertices, 0, bufferSizeBytes);
            #endif
            
            uint32_t vertexIndex = 0;
            
            for (const auto& box : boxes) {
                float hw = box.width / 2.0f;
                float hl = box.length / 2.0f;
                float hh = box.height / 2.0f;
                
                float cosYaw = std::cos(box.rotation);
                float sinYaw = std::sin(box.rotation);
                
                float localCorners[8][3] = {
                    {-hw, -hl, -hh}, {hw, -hl, -hh}, {hw, hl, -hh}, {-hw, hl, -hh},
                    {-hw, -hl, hh}, {hw, -hl, hh}, {hw, hl, hh}, {-hw, hl, hh}
                };
                
                dwVector3f corners[8];
                for (int i = 0; i < 8; ++i) {
                    float x = localCorners[i][0];
                    float y = localCorners[i][1];
                    float rotatedX = x * cosYaw - y * sinYaw;
                    float rotatedY = x * sinYaw + y * cosYaw;
                    corners[i] = {box.x + rotatedX, box.y + rotatedY, box.z + localCorners[i][2]};
                }
                
                int edges[12][2] = {
                    {0,1}, {1,2}, {2,3}, {3,0}, {4,5}, {5,6}, {6,7}, {7,4},
                    {0,4}, {1,5}, {2,6}, {3,7}
                };
                
                for (int i = 0; i < 12; ++i) {
                    if (vertexIndex + 2 > expectedVertices) break;  // Safety check
                    vertices[vertexIndex++] = corners[edges[i][0]];
                    vertices[vertexIndex++] = corners[edges[i][1]];
                }
                
                // Render yellow points inside box (EXACT as original)
                renderPointsInBox(box);
            }
            
            // CUDA sync before unmap (like original)
            #ifdef __aarch64__
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            #endif
            
            CHECK_DW_ERROR(dwRenderEngine_unmapBuffer(m_boxLineBufferId,
                                                     DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                                     m_renderEngine));
            
            if (vertexIndex > 0) {
                dwRenderEngine_setColor(color, m_renderEngine);
                dwRenderEngine_setLineWidth(2.0f, m_renderEngine);
                CHECK_DW_ERROR(dwRenderEngine_renderBuffer(m_boxLineBufferId, vertexIndex, m_renderEngine));
            }
        };
        
        // Red for vehicles, Green for pedestrians, Blue for cyclists (EXACT as original)
        renderBoxGroup(vehicleBoxes, {1.0f, 0.0f, 0.0f, 1.0f});
        renderBoxGroup(pedestrianBoxes, {0.0f, 1.0f, 0.0f, 1.0f});
        renderBoxGroup(cyclistBoxes, {0.0f, 0.0f, 1.0f, 1.0f});
        
        // Render 3D text labels (EXACT as original)
        dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_renderEngine);
        dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, m_renderEngine); // White
        
        for (const auto& text : labelTexts) {
            dwRenderEngine_renderText3D(text.second.c_str(), text.first, m_renderEngine);
        }
        
        // Render debug texts in yellow (EXACT as original)
        dwRenderEngine_setColor({1.0f, 1.0f, 0.0f, 1.0f}, m_renderEngine); // Yellow
        
        for (const auto& text : debugTexts) {
            dwRenderEngine_renderText3D(text.second.c_str(), text.first, m_renderEngine);
        }
    }
    
    int countPointsInBox(const DetectionBoundingBox& box)
    {
        int count = 0;
        float xMin = box.x - box.width/2;
        float xMax = box.x + box.width/2;
        float yMin = box.y - box.length/2;
        float yMax = box.y + box.length/2;
        float zMin = box.z - box.height/2;
        float zMax = box.z + box.height/2;
        
        for (uint32_t i = 0; i < m_currentPacket.numPoints; ++i) {
            float x = m_currentPacket.points[i * 4 + 0];
            float y = m_currentPacket.points[i * 4 + 1];
            float z = m_currentPacket.points[i * 4 + 2];
            
            if (x >= xMin && x <= xMax && y >= yMin && y <= yMax && z >= zMin && z <= zMax) {
                count++;
            }
        }
        return count;
    }
    
    void renderPointsInBox(const DetectionBoundingBox& box)
    {
        // Find points inside bounding box (EXACT as original)
        std::vector<dwVector3f> pointsInBox;
        float xMin = box.x - box.width/2;
        float xMax = box.x + box.width/2;
        float yMin = box.y - box.length/2;
        float yMax = box.y + box.length/2;
        float zMin = box.z - box.height/2;
        float zMax = box.z + box.height/2;
        
        for (uint32_t i = 0; i < m_currentPacket.numPoints; ++i) {
            float x = m_currentPacket.points[i * 4 + 0];
            float y = m_currentPacket.points[i * 4 + 1];
            float z = m_currentPacket.points[i * 4 + 2];
            
            if (x >= xMin && x <= xMax && y >= yMin && y <= yMax && z >= zMin && z <= zMax) {
                pointsInBox.push_back({x, y, z});
            }
        }
        
        if (!pointsInBox.empty()) {
            // YELLOW highlighting with bigger points (EXACT as original)
            dwRenderEngine_setPointSize(5.0f, m_renderEngine);
            dwRenderEngine_setColor({1.0f, 1.0f, 0.0f, 1.0f}, m_renderEngine);
            
            dwRenderEngine_render(
                DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                pointsInBox.data(),
                sizeof(dwVector3f),
                0,
                pointsInBox.size(),
                m_renderEngine
            );
            
            dwRenderEngine_setPointSize(1.0f, m_renderEngine);
        }
    }
    
    void renderFreeSpace()
    {
        // Use exact same technique as original with CUDA sync
        uint32_t numPoints = std::min(m_currentPacket.freeSpace.numPoints, 50000u);  // Match buffer capacity
        uint32_t sizeInBytes = numPoints * sizeof(dwVector3f);
        dwVector3f* vertices = nullptr;
        
        // CUDA sync (like original)
        #ifdef __aarch64__
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        #endif
        
        CHECK_DW_ERROR(dwRenderEngine_mapBuffer(m_freeSpaceBufferId,
                                               reinterpret_cast<void**>(&vertices),
                                               0,
                                               sizeInBytes,
                                               DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                               m_renderEngine));
        
        // Clear buffer on Orin (like original)
        #ifdef __aarch64__
        memset(vertices, 0, sizeInBytes);
        #endif
        
        // Convert free space points to 3D vertices (on ground level)
        for (uint32_t i = 0; i < numPoints; ++i) {
            vertices[i].x = m_currentPacket.freeSpace.points[i * 3 + 0];
            vertices[i].y = m_currentPacket.freeSpace.points[i * 3 + 1];
            vertices[i].z = 0.0f;  // Free space at ground level
        }
        
        CHECK_DW_ERROR(dwRenderEngine_unmapBuffer(m_freeSpaceBufferId,
                                                  DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                  m_renderEngine));
        
        // Render free space in bright green with transparency (like original)
        dwRenderEngine_setColor({0.0f, 1.0f, 0.0f, 0.6f}, m_renderEngine);
        dwRenderEngine_setPointSize(3.0f, m_renderEngine);
        CHECK_DW_ERROR(dwRenderEngine_renderBuffer(m_freeSpaceBufferId, numPoints, m_renderEngine));
        dwRenderEngine_setPointSize(1.0f, m_renderEngine);  // Reset
    }
    
    void renderGroundPlane()
    {
        const GroundPlaneData& plane = m_currentPacket.groundPlane;
        
        // Map render buffer
        dwVector3f* vertices = nullptr;
        const uint32_t GRID_SIZE = 21;
        const uint32_t gridVertices = (GRID_SIZE - 1) * (GRID_SIZE - 1) * 6;
        uint32_t bufferSize = gridVertices * sizeof(dwVector3f);
        
        CHECK_DW_ERROR(dwRenderEngine_mapBuffer(m_groundPlaneBufferId,
                                               reinterpret_cast<void**>(&vertices),
                                               0,
                                               bufferSize,
                                               DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
                                               m_renderEngine));
        
        // Generate ground plane mesh
        const float gridSpacing = 10.0f;  // 10m spacing
        const float gridStart = -100.0f;
        uint32_t vertexIndex = 0;
        
        for (uint32_t i = 0; i < GRID_SIZE - 1; ++i) {
            for (uint32_t j = 0; j < GRID_SIZE - 1; ++j) {
                float x0 = gridStart + i * gridSpacing;
                float y0 = gridStart + j * gridSpacing;
                float x1 = x0 + gridSpacing;
                float y1 = y0 + gridSpacing;
                
                // Calculate Z using plane equation: nx*x + ny*y + nz*z = -offset
                auto calcZ = [&](float x, float y) {
                    if (std::abs(plane.normalZ) > 0.01f) {
                        return (-plane.offset - plane.normalX * x - plane.normalY * y) / plane.normalZ;
                    }
                    return 0.0f;
                };
                
                // First triangle
                vertices[vertexIndex++] = {x0, y0, calcZ(x0, y0)};
                vertices[vertexIndex++] = {x1, y0, calcZ(x1, y0)};
                vertices[vertexIndex++] = {x1, y1, calcZ(x1, y1)};
                
                // Second triangle
                vertices[vertexIndex++] = {x0, y0, calcZ(x0, y0)};
                vertices[vertexIndex++] = {x1, y1, calcZ(x1, y1)};
                vertices[vertexIndex++] = {x0, y1, calcZ(x0, y1)};
            }
        }
        
        CHECK_DW_ERROR(dwRenderEngine_unmapBuffer(m_groundPlaneBufferId,
                                                  DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
                                                  m_renderEngine));
        
        // Render with earth-tone color (like original)
        dwRenderEngine_setColor({0.4f, 0.3f, 0.2f, 0.7f}, m_renderEngine);
        CHECK_DW_ERROR(dwRenderEngine_renderBuffer(m_groundPlaneBufferId, vertexIndex, m_renderEngine));
    }
    
    void renderStatusText()
    {
        // Set up 2D text rendering (EXACT as original)
        dwRenderEngine_setTile(0, m_renderEngine);
        dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
        dwRenderEngine_setCoordinateRange2D({1.0f, 1.0f}, m_renderEngine);
        dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine);
        dwRenderEngine_setColor({0.0f, 0.6f, 1.0f, 1.0f}, m_renderEngine);
        
        char text[256];
        snprintf(text, sizeof(text), "Packets: %u | Frame: %u | Points: %u | Detections: %u",
                 m_packetCount, m_currentPacket.frameNumber, m_currentPacket.numPoints, m_currentPacket.numDetections);
        
        dwRenderEngine_renderText2D(text, {0.05f, 0.05f}, m_renderEngine);
    }
};

//------------------------------------------------------------------------------
int main(int argc, const char **argv)
{
    ProgramArguments args({
        ProgramArguments::Option_t("ip", "127.0.0.1", "Server IP address to connect to"),
        ProgramArguments::Option_t("port", "40002", "Port for socket communication")
    });

    if (!args.parse(argc, argv)) {
        std::cerr << "Failed to parse command line arguments" << std::endl;
        return -1;
    }

    if (args.has("help")) {
        return 0;
    }

    std::cout << "\n=== Detection Consumer (Socket Client) ===" << std::endl;
    std::cout << "Server IP: " << args.get("ip") << std::endl;
    std::cout << "Port: " << args.get("port") << std::endl;
    std::cout << "==========================================\n" << std::endl;

    DetectionConsumer app(args);
    app.initializeWindow("Lidar Object Detection Consumer", 1280, 800, false);
    
    return app.run();
}
