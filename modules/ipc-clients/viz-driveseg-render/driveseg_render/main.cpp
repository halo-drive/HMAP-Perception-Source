// DriveSeg4CamClient.cpp
// Uses CUDA images for GL streaming compatibility

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

// ---------------- DriveWorks ----------------
#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>

#include <dw/image/Image.h>
#include <dw/interop/streamer/ImageStreamer.h>

// Socket IPC
#include <dw/comms/socketipc/SocketClientServer.h>

// Visualization
#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Renderer.h>
#include <dwvisualization/core/Visualization.h>
#include <dwvisualization/image/Image.h>
#include <dwvisualization/interop/ImageStreamer.h>

// ------------- DW Sample Framework ----------
#include <framework/Checks.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/Log.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/ScreenshotHelper.hpp>
#include <framework/WindowGLFW.hpp>

using namespace dw_samples::common;

#define MAX_CAMS 4

// ===============================================================
// IPC DATA STRUCTURES (must match server)
// ===============================================================

#pragma pack(push, 1)
struct FrameHeader {
    uint32_t magic;           // 0xDEADBEEF
    uint32_t cameraIndex;
    uint64_t frameId;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t imageDataSize;
    uint32_t numBoxes;
    uint32_t segCount;
    uint32_t detCount;
    float avgSegMs;
    float avgDetMs;
    float avgStage2Ms;
    uint32_t num3DBoxes;      
};

struct DetectionBox {
    float x, y, width, height;
    char label[64];
};

struct Detection3DBox {
    float depth;        // in meters
    float height;       // in meters
    float width;        // in meters
    float length;       // in meters
    float rotation;     // in radians
    float iouScore;     // quality score
};
#pragma pack(pop)

struct ReceivedFrame {
    uint32_t cameraIndex;
    uint64_t frameId;
    
    // Image
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    std::vector<uint8_t> rgbaPixels;
    
    // Detections
    std::vector<dwRectf> boxes;
    std::vector<std::string> labels;
    
    std::vector<Detection3DBox> boxes3D;

    // Stats
    uint32_t segCount;
    uint32_t detCount;
    float avgSegMs;
    float avgDetMs;
    float avgStage2Ms;
    
    bool valid{false};
};

// ===============================================================
// RENDER CLIENT APPLICATION
// ===============================================================

class DriveSeg4CamClient : public DriveWorksSample
{
public:
    explicit DriveSeg4CamClient(const ProgramArguments& args);

    bool onInitialize() override;
    void onRelease() override;
    void onProcess() override;
    void onRender() override;
    void onKeyDown(int key, int scancode, int mods) override;

private:
    // Core DW
    dwContextHandle_t m_ctx{DW_NULL_HANDLE};

    // Visualization
    dwVisualizationContextHandle_t m_viz{DW_NULL_HANDLE};
    dwRenderEngineHandle_t m_re{DW_NULL_HANDLE};
    dwRenderEngineParams m_reParams{};
    uint32_t m_tiles[MAX_CAMS]{};
    std::unique_ptr<ScreenshotHelper> m_screenshot;

    // Images for rendering (CUDA, like original)
    uint32_t m_numCameras{4};
    dwImageHandle_t m_imgCUDA[MAX_CAMS]{DW_NULL_HANDLE};
    dwImageStreamerHandle_t m_streamerToGL[MAX_CAMS]{DW_NULL_HANDLE};

    // IPC Socket Client
    dwSocketClientHandle_t m_socketClients[MAX_CAMS]{DW_NULL_HANDLE};
    dwSocketConnectionHandle_t m_connections[MAX_CAMS]{DW_NULL_HANDLE};
    std::mutex m_frameMutex[MAX_CAMS];
    ReceivedFrame m_latestFrames[MAX_CAMS];

    // Connection parameters
    std::string m_serverIP;
    uint16_t m_serverPort{49252};

    // Reception threads
    std::vector<std::thread> m_receiveThreads;
    std::atomic<bool> m_running{true};

    // Stats
    std::atomic<uint64_t> m_frameCounter{0};
    std::chrono::high_resolution_clock::time_point m_lastStats{};

private:
    // Init
    void initDW();
    void initRenderer();
    void initImages();
    void initSocketClient();

    // Reception
    void receiveThreadFunc(uint32_t camIndex);
    bool receiveFrame(uint32_t camIndex, ReceivedFrame& frame);

    // Rendering
    void renderCamera(uint32_t i);
    void renderDetectionBoxes(uint32_t i, const ReceivedFrame& frame);

    // Utilities
    void printStats();

    struct CameraIntrinsics {
        float fx, fy, cx, cy;
    };CameraIntrinsics m_intrinsics;
    void initIntrinsics();
};

// ===============================================================
// IMPLEMENTATION
// ===============================================================

DriveSeg4CamClient::DriveSeg4CamClient(const ProgramArguments& args)
    : DriveWorksSample(args)
{
    m_serverIP = args.get("ip");
    m_serverPort = static_cast<uint16_t>(std::stoul(args.get("port")));
    m_numCameras = static_cast<uint32_t>(std::stoul(args.get("num-cameras")));
}

void DriveSeg4CamClient::initIntrinsics()
    {
        // Rectified camera intrinsics from calibration (3848x2168)
        // Scaled to transmitted resolution (640x640)
        const float origW = 3848.0f;
        const float origH = 2168.0f;
        const float transW = 640.0f;
        const float transH = 640.0f;
        
        const float scaleX = transW / origW;
        const float scaleY = transH / origH;
        
        // Original rectified intrinsics
        const float fx_orig = 1655.2066956340523f;
        const float fy_orig = 1656.334760722568f;
        const float cx_orig = 1819.2548585234283f;
        const float cy_orig = 1019.9403399106654f;
        
        // Scale to 640x640
        m_intrinsics.fx = fx_orig * scaleX;  // ~275.3
        m_intrinsics.fy = fy_orig * scaleY;  // ~489.0
        m_intrinsics.cx = cx_orig * scaleX;  // ~302.5
        m_intrinsics.cy = cy_orig * scaleY;  // ~301.1
        
        std::cout << "[Intrinsics] Scaled to " << transW << "x" << transH << ":\n";
        std::cout << "  fx=" << m_intrinsics.fx << ", fy=" << m_intrinsics.fy << "\n";
        std::cout << "  cx=" << m_intrinsics.cx << ", cy=" << m_intrinsics.cy << "\n";
    }

bool DriveSeg4CamClient::onInitialize()
{
    std::cout << "[Client] Initializing...\n";

    initDW();
    initRenderer();
    initImages();
    initIntrinsics();
    initSocketClient();


    // Start reception threads
    for (uint32_t i = 0; i < m_numCameras; ++i) {
        m_receiveThreads.emplace_back(&DriveSeg4CamClient::receiveThreadFunc, this, i);
    }

    m_lastStats = std::chrono::high_resolution_clock::now();
    std::cout << "[Client] Initialized for " << m_numCameras << " camera(s)\n";
    std::cout << "[Client] Connected to server " << m_serverIP << ":" << m_serverPort << "\n";

    return true;
}

void DriveSeg4CamClient::onRelease()
{
    std::cout << "[Client] Releasing...\n";
    m_running = false;

    // Join reception threads
    for (auto& thread : m_receiveThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Release IPC
    for (uint32_t i = 0; i < MAX_CAMS; ++i) {
        if (m_connections[i]) {
            dwSocketConnection_release(m_connections[i]);
        }
        if (m_socketClients[i]) {
            dwSocketClient_release(m_socketClients[i]);
        }
    }

    // Release images
    for (uint32_t i = 0; i < m_numCameras; ++i) {
        if (m_streamerToGL[i]) dwImageStreamerGL_release(m_streamerToGL[i]);
        if (m_imgCUDA[i]) dwImage_destroy(m_imgCUDA[i]);
    }

    // Release renderer
    if (m_re) dwRenderEngine_release(m_re);
    if (m_viz) dwVisualizationRelease(m_viz);
    if (m_ctx) dwRelease(m_ctx);

    std::cout << "[Client] Released.\n";
}

void DriveSeg4CamClient::onProcess()
{
    // Processing happens in reception threads
}

void DriveSeg4CamClient::onRender()
{
    for (uint32_t i = 0; i < m_numCameras; ++i) {
        renderCamera(i);
    }

    m_frameCounter++;
    auto now = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - m_lastStats).count() >= 2) {
        printStats();
        m_lastStats = now;
    }
}

void DriveSeg4CamClient::onKeyDown(int key, int, int)
{
    if (key == GLFW_KEY_P) {
        printStats();
    }
}

void DriveSeg4CamClient::initDW()
{
    CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
    CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));
    dwContextParameters p{};
#ifdef VIBRANTE
    p.eglDisplay = getEGLDisplay();
#endif
    CHECK_DW_ERROR(dwInitialize(&m_ctx, DW_VERSION, &p));
}

void DriveSeg4CamClient::initRenderer()
{
    CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_ctx));

    CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&m_reParams, getWindowWidth(), getWindowHeight()));
    m_reParams.defaultTile.lineWidth = 2.f;
    m_reParams.maxBufferCount = 1;
    m_reParams.bounds = {0, 0, (float)getWindowWidth(), (float)getWindowHeight()};
    CHECK_DW_ERROR(dwRenderEngine_initialize(&m_re, &m_reParams, m_viz));

    const uint32_t tilesPerRow = (m_numCameras <= 1) ? 1 : 2;
    dwRenderEngineTileState states[MAX_CAMS];
    for (uint32_t i = 0; i < m_numCameras; ++i) {
        dwRenderEngine_initTileState(&states[i]);
        states[i].font = DW_RENDER_ENGINE_FONT_VERDANA_16;
        states[i].modelViewMatrix = DW_IDENTITY_MATRIX4F;
    }
    CHECK_DW_ERROR(dwRenderEngine_addTilesByCount(m_tiles, m_numCameras, tilesPerRow, states, m_re));
}

void DriveSeg4CamClient::initImages()
{
    // Create CUDA images (matching original application approach)
    dwImageProperties cudaProps{};
    cudaProps.type = DW_IMAGE_CUDA;
    cudaProps.format = DW_IMAGE_FORMAT_RGBA_UINT8;
    cudaProps.width = 1920;  // Default, will be updated on first frame
    cudaProps.height = 1208;

    for (uint32_t i = 0; i < m_numCameras; ++i) {
        CHECK_DW_ERROR(dwImage_create(&m_imgCUDA[i], cudaProps, m_ctx));
        
        // Initialize CUDA→GL streamer (same as original)
        CHECK_DW_ERROR(dwImageStreamerGL_initialize(&m_streamerToGL[i], &cudaProps, DW_IMAGE_GL, m_ctx));
    }
    
    std::cout << "[Client] CUDA images initialized\n";
}


void DriveSeg4CamClient::initSocketClient()
{
    std::cout << "[IPC] Connecting to server " << m_serverIP << "...\n";

    for (uint32_t i = 0; i < m_numCameras; ++i) {
        uint16_t port = m_serverPort + i;
        
        CHECK_DW_ERROR(dwSocketClient_initialize(&m_socketClients[i], 1, m_ctx));
        
        std::cout << "[IPC] Connecting camera " << i << " to port " << port << "...\n";
        
        dwStatus status = DW_TIME_OUT;
        int attempts = 0;
        
        while (status == DW_TIME_OUT && attempts < 30) {
            status = dwSocketClient_connect(&m_connections[i], m_serverIP.c_str(), 
                                           port, 1000, m_socketClients[i]);
            attempts++;
        }
        
        if (status == DW_SUCCESS) {
            std::cout << "[IPC] Camera " << i << " connected to port " << port << "\n";
        } else {
            throw std::runtime_error("Failed to connect camera " + std::to_string(i));
        }
    }
}

// ===============================================================
// FRAME RECEPTION (runs in separate threads)
// ===============================================================

void DriveSeg4CamClient::receiveThreadFunc(uint32_t camIndex)
{
    std::cout << "[Receive Thread " << camIndex << "] Started\n";

    while (m_running) {
        ReceivedFrame frame;
        if (receiveFrame(camIndex, frame)) {
            std::lock_guard<std::mutex> lock(m_frameMutex[camIndex]);
            m_latestFrames[camIndex] = std::move(frame);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    std::cout << "[Receive Thread " << camIndex << "] Stopped\n";
}



bool DriveSeg4CamClient::receiveFrame(uint32_t camIndex, ReceivedFrame& frame)
{
    if (!m_connections[camIndex]) return false;

    // ========================================
    // STEP 1: Peek at header to verify magic
    // ========================================
    FrameHeader header{};
    size_t peekSize = sizeof(FrameHeader);
    
    dwStatus status = dwSocketConnection_peek(
        reinterpret_cast<uint8_t*>(&header),
        &peekSize,
        5000,
        m_connections[camIndex]
    );

    if (status != DW_SUCCESS) {
        if (status != DW_TIME_OUT) {
            std::cout << "[Client] Peek header failed for cam " << camIndex 
                      << ": " << dwGetStatusName(status) << "\n";
        }
        return false;
    }

    if (peekSize != sizeof(FrameHeader)) {
        std::cout << "[Client] Incomplete header peek for cam " << camIndex 
                  << " (got " << peekSize << " bytes)\n";
        return false;
    }

    // ========================================
    // STEP 2: Check if header is valid
    // ========================================
    if (header.magic != 0xDEADBEEF) {
        std::cout << "[Client] Invalid magic during peek for cam " << camIndex 
                  << " (got 0x" << std::hex << header.magic << std::dec << ")\n";
        
        // Try to resync
        std::cout << "[Client] Attempting resync for cam " << camIndex << "...\n";
        
        for (int attempts = 0; attempts < 1000; ++attempts) {
            uint8_t discard;
            size_t discardSize = 1;
            dwSocketConnection_read(&discard, &discardSize, 1000, m_connections[camIndex]);
            
            peekSize = sizeof(FrameHeader);
            status = dwSocketConnection_peek(
                reinterpret_cast<uint8_t*>(&header),
                &peekSize,
                1000,
                m_connections[camIndex]
            );
            
            if (status == DW_SUCCESS && peekSize == sizeof(FrameHeader) && header.magic == 0xDEADBEEF) {
                std::cout << "[Client] Resync successful for cam " << camIndex 
                          << " after " << attempts << " bytes\n";
                break;
            }
        }
        
        if (header.magic != 0xDEADBEEF) {
            std::cout << "[Client] Resync failed for cam " << camIndex << "\n";
            return false;
        }
    }

    // ========================================
    // STEP 3: Now actually READ the header
    // ========================================
    size_t headerSize = sizeof(FrameHeader);
    status = dwSocketConnection_read(
        reinterpret_cast<uint8_t*>(&header),
        &headerSize,
        5000,
        m_connections[camIndex]
    );

    if (status != DW_SUCCESS || headerSize != sizeof(FrameHeader)) {
        std::cout << "[Client] Failed to read verified header for cam " << camIndex << "\n";
        return false;
    }

    std::cout << "[Client] Cam " << camIndex << " frame " << header.frameId 
              << " | Image: " << header.imageDataSize << " bytes"
              << " | 2D: " << header.numBoxes 
              << " | 3D: " << header.num3DBoxes << "\n";

    // Populate frame metadata
    frame.cameraIndex = header.cameraIndex;
    frame.frameId = header.frameId;
    frame.width = header.width;
    frame.height = header.height;
    frame.stride = header.stride;
    frame.segCount = header.segCount;
    frame.detCount = header.detCount;
    frame.avgSegMs = header.avgSegMs;
    frame.avgDetMs = header.avgDetMs;
    frame.avgStage2Ms = header.avgStage2Ms;  // 

    // ========================================
    // STEP 4: Receive image data
    // ========================================
    frame.rgbaPixels.resize(header.imageDataSize);
    size_t totalReceived = 0;
    uint8_t* dataPtr = frame.rgbaPixels.data();
    
    int maxRetries = 5;
    int retryCount = 0;
    
    while (totalReceived < header.imageDataSize && retryCount < maxRetries) {
        size_t remaining = header.imageDataSize - totalReceived;
        size_t bytesRead = remaining;
        
        status = dwSocketConnection_read(
            dataPtr + totalReceived,
            &bytesRead,
            20000,
            m_connections[camIndex]
        );
        
        if (status == DW_SUCCESS) {
            totalReceived += bytesRead;
            retryCount = 0;
            
            if (bytesRead == 0) {
                std::cout << "[Client] Cam " << camIndex << " connection closed\n";
                return false;
            }
        } else if (status == DW_TIME_OUT) {
            retryCount++;
            std::cout << "[Client] Cam " << camIndex << " image read timeout (attempt " 
                      << retryCount << "/" << maxRetries << ")\n";
            
            if (retryCount >= maxRetries) {
                std::cout << "[Client] Cam " << camIndex << " failed after retries\n";
                return false;
            }
        } else {
            std::cout << "[Client] Cam " << camIndex << " image read failed: " 
                      << dwGetStatusName(status) << "\n";
            return false;
        }
    }
    
    if (totalReceived != header.imageDataSize) {
        std::cout << "[Client] Cam " << camIndex << " incomplete image\n";
        return false;
    }

    // ========================================
    // STEP 5: Receive 2D detection boxes
    // ========================================
    frame.boxes.clear();
    frame.labels.clear();
    frame.boxes.reserve(header.numBoxes);
    frame.labels.reserve(header.numBoxes);

    for (uint32_t i = 0; i < header.numBoxes; ++i) {
        DetectionBox box{};
        size_t boxSize = sizeof(DetectionBox);
        status = dwSocketConnection_read(
            reinterpret_cast<uint8_t*>(&box),
            &boxSize,
            5000,
            m_connections[camIndex]
        );

        if (status != DW_SUCCESS || boxSize != sizeof(DetectionBox)) {
            std::cout << "[Client] Failed to receive 2D box " << i << " for cam " << camIndex << "\n";
            return false;
        }

        dwRectf rectf{box.x, box.y, box.width, box.height};
        frame.boxes.push_back(rectf);
        frame.labels.push_back(std::string(box.label));
    }

    // ========================================
    //  STEP 6: Receive 3D detection boxes (NEW!)
    // ========================================
    frame.boxes3D.clear();
    frame.boxes3D.reserve(header.num3DBoxes);

    for (uint32_t i = 0; i < header.num3DBoxes; ++i) {
        Detection3DBox box3D{};
        size_t box3DSize = sizeof(Detection3DBox);
        status = dwSocketConnection_read(
            reinterpret_cast<uint8_t*>(&box3D),
            &box3DSize,
            5000,
            m_connections[camIndex]
        );

        if (status != DW_SUCCESS || box3DSize != sizeof(Detection3DBox)) {
            std::cout << "[Client] Failed to receive 3D box " << i << " for cam " << camIndex << "\n";
            return false;
        }

        frame.boxes3D.push_back(box3D);
    }

    frame.valid = true;
    return true;
}

// ===============================================================
// RENDERING
// ===============================================================

void DriveSeg4CamClient::renderCamera(uint32_t i)
{
    ReceivedFrame frame;
    {
        std::lock_guard<std::mutex> lock(m_frameMutex[i]);
        if (!m_latestFrames[i].valid) return;
        frame = m_latestFrames[i];  // Copy frame data
    }

    CHECK_DW_ERROR(dwRenderEngine_setTile(m_tiles[i], m_re));
    CHECK_DW_ERROR(dwRenderEngine_resetTile(m_re));

    // Get CUDA image
    dwImageCUDA* cudaImg{};
    CHECK_DW_ERROR(dwImage_getCUDA(&cudaImg, m_imgCUDA[i]));

    // Check if we need to resize the image
    if (cudaImg->prop.width != frame.width || cudaImg->prop.height != frame.height) {
        std::cout << "[Client] Resizing camera " << i << " to " 
                  << frame.width << "x" << frame.height << "\n";
        
        // Recreate image with correct dimensions
        dwImageStreamerGL_release(m_streamerToGL[i]);
        dwImage_destroy(m_imgCUDA[i]);

        dwImageProperties props{};
        props.type = DW_IMAGE_CUDA;
        props.format = DW_IMAGE_FORMAT_RGBA_UINT8;
        props.width = frame.width;
        props.height = frame.height;

        CHECK_DW_ERROR(dwImage_create(&m_imgCUDA[i], props, m_ctx));
        CHECK_DW_ERROR(dwImageStreamerGL_initialize(&m_streamerToGL[i], &props, DW_IMAGE_GL, m_ctx));

        CHECK_DW_ERROR(dwImage_getCUDA(&cudaImg, m_imgCUDA[i]));
    }

    // Copy received pixel data from CPU to CUDA
    const size_t rowBytes = frame.width * 4;  // RGBA
    cudaMemcpy2D(
        cudaImg->dptr[0],           // dst
        cudaImg->pitch[0],          // dst pitch
        frame.rgbaPixels.data(),    // src
        frame.stride,               // src pitch
        rowBytes,                   // width in bytes
        frame.height,               // height
        cudaMemcpyHostToDevice
    );

    // Stream CUDA image to GL (same as original)
    CHECK_DW_ERROR(dwImageStreamerGL_producerSend(m_imgCUDA[i], m_streamerToGL[i]));
    
    dwImageHandle_t frameGL{};
    CHECK_DW_ERROR(dwImageStreamerGL_consumerReceive(&frameGL, 33000, m_streamerToGL[i]));
    
    dwImageGL* imageGL{};
    CHECK_DW_ERROR(dwImage_getGL(&imageGL, frameGL));

    // Render image
    dwVector2f range{(float)imageGL->prop.width, (float)imageGL->prop.height};
    CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_re));
    CHECK_DW_ERROR(dwRenderEngine_renderImage2D(imageGL, {0, 0, range.x, range.y}, m_re));

    // Render detection boxes
    renderDetectionBoxes(i, frame);

    // Render stats
    CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_re));
    CHECK_DW_ERROR(dwRenderEngine_setColor({1, 1, 1, 1}, m_re));
    char buf[256];
    std::snprintf(buf, sizeof(buf), "Cam %u | Frame:%lu Det:%u(%.1fms) Seg:%u(%.1fms) Boxes:%zu",
                  i, frame.frameId, frame.detCount, frame.avgDetMs, 
                  frame.segCount, frame.avgSegMs, frame.boxes.size(), frame.avgStage2Ms, frame.boxes.size());
    CHECK_DW_ERROR(dwRenderEngine_renderText2D(buf, {16, 28}, m_re));

    // Return GL image (same as original)
    CHECK_DW_ERROR(dwImageStreamerGL_consumerReturn(&frameGL, m_streamerToGL[i]));
    CHECK_DW_ERROR(dwImageStreamerGL_producerReturn(nullptr, 32000, m_streamerToGL[i]));
}



void DriveSeg4CamClient::renderDetectionBoxes(uint32_t i, const ReceivedFrame& frame)
{
    if (frame.boxes.empty()) return;

    const float fx = m_intrinsics.fx;
    const float fy = m_intrinsics.fy;
    const float cx = m_intrinsics.cx;
    const float cy = m_intrinsics.cy;

    // ========================================
    // Render 2D boxes (red)
    // ========================================
    CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_RED, m_re));
    CHECK_DW_ERROR(dwRenderEngine_setLineWidth(2.0f, m_re));
    CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D,
                                         frame.boxes.data(), sizeof(dwRectf), 0,
                                         frame.boxes.size(), m_re));

    // ========================================
    // Render 2D labels
    // ========================================
    CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_WHITE, m_re));
    CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_8, m_re));
    for (size_t j = 0; j < frame.labels.size(); ++j) {
        dwVector2f labelPos = {frame.boxes[j].x + 2, frame.boxes[j].y - 2};
        CHECK_DW_ERROR(dwRenderEngine_renderText2D(frame.labels[j].c_str(), labelPos, m_re));
    }

    // ========================================
    // Render 3D boxes with camera projection
    // ========================================
    if (!frame.boxes3D.empty()) {
        size_t num3D = std::min(frame.boxes.size(), frame.boxes3D.size());
        
        for (size_t j = 0; j < num3D; ++j) {
            const auto& box3D = frame.boxes3D[j];
            const auto& box2D = frame.boxes[j];
            
            // Skip invalid detections
            if (box3D.depth <= 0.1f || box3D.depth > 150.0f) continue;
            
            // ========================================
            // Reconstruct 3D position from 2D box + depth
            // Use bottom-center of 2D box as reference
            // ========================================
            float x_2d = box2D.x + box2D.width / 2.0f;
            float y_2d = box2D.y + box2D.height;  // Bottom edge
            float z = box3D.depth;
            
            // Unproject to camera coordinates
            float x_cam = (x_2d - cx) * z / fx;
            float y_cam_bottom = (y_2d - cy) * z / fy;
            
            // Shift to geometric center
            float y_cam = y_cam_bottom - box3D.height / 2.0f;
            float z_cam = z;
            
            float h = box3D.height;
            float w = box3D.width;
            float l = box3D.length;
            float ry = box3D.rotation;
            
            // ========================================
            // 8 corners in object frame (KITTI convention)
            // Y is down in camera frame
            // ========================================
            float half_l = l / 2.0f;
            float half_w = w / 2.0f;
            float half_h = h / 2.0f;
            
            // Corners: [x, y, z] in object frame
            // Bottom = +y, Top = -y (camera Y points down)
            float corners_obj[8][3] = {
                {-half_l,  half_h, -half_w},  // 0: back-left-bottom
                { half_l,  half_h, -half_w},  // 1: back-right-bottom
                { half_l,  half_h,  half_w},  // 2: front-right-bottom
                {-half_l,  half_h,  half_w},  // 3: front-left-bottom
                {-half_l, -half_h, -half_w},  // 4: back-left-top
                { half_l, -half_h, -half_w},  // 5: back-right-top
                { half_l, -half_h,  half_w},  // 6: front-right-top
                {-half_l, -half_h,  half_w},  // 7: front-left-top
            };
            
            // ========================================
            // Rotate around Y and translate to camera frame
            // ========================================
            float cos_ry = std::cos(ry);
            float sin_ry = std::sin(ry);
            
            dwVector2f corners_2d[8];
            bool valid = true;
            
            for (int c = 0; c < 8; ++c) {
                // Rotate around Y axis
                float x_rot =  corners_obj[c][0] * cos_ry + corners_obj[c][2] * sin_ry;
                float y_rot =  corners_obj[c][1];
                float z_rot = -corners_obj[c][0] * sin_ry + corners_obj[c][2] * cos_ry;
                
                // Translate to camera frame
                float x_c = x_rot + x_cam;
                float y_c = y_rot + y_cam;
                float z_c = z_rot + z_cam;
                
                // Behind camera check
                if (z_c <= 0.1f) {
                    valid = false;
                    break;
                }
                
                // Project to 2D
                corners_2d[c].x = fx * x_c / z_c + cx;
                corners_2d[c].y = fy * y_c / z_c + cy;
            }
            
            if (!valid) continue;
            
            // ========================================
            // Draw wireframe
            // ========================================
            CHECK_DW_ERROR(dwRenderEngine_setLineWidth(2.0f, m_re));
            
            // Bottom face (cyan)
            CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_CYAN, m_re));
            for (int c = 0; c < 4; ++c) {
                dwVector2f line[2] = {corners_2d[c], corners_2d[(c + 1) % 4]};
                CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_2D,
                                                     line, sizeof(dwVector2f), 0, 2, m_re));
            }
            
            // Top face (yellow)
            CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_YELLOW, m_re));
            for (int c = 4; c < 8; ++c) {
                int next = 4 + ((c - 4 + 1) % 4);
                dwVector2f line[2] = {corners_2d[c], corners_2d[next]};
                CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_2D,
                                                     line, sizeof(dwVector2f), 0, 2, m_re));
            }
            
            // Vertical edges (green)
            CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_GREEN, m_re));
            for (int c = 0; c < 4; ++c) {
                dwVector2f line[2] = {corners_2d[c], corners_2d[c + 4]};
                CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_2D,
                                                     line, sizeof(dwVector2f), 0, 2, m_re));
            }
            
            // Front face edges (red, thicker) - indicates heading
            CHECK_DW_ERROR(dwRenderEngine_setLineWidth(3.0f, m_re));
            CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_RED, m_re));
            dwVector2f frontEdges[4][2] = {
                {corners_2d[2], corners_2d[3]},  // front bottom
                {corners_2d[6], corners_2d[7]},  // front top
                {corners_2d[3], corners_2d[7]},  // front left vertical
                {corners_2d[2], corners_2d[6]},  // front right vertical
            };
            for (int e = 0; e < 4; ++e) {
                CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_2D,
                                                     frontEdges[e], sizeof(dwVector2f), 0, 2, m_re));
            }
            
            // ========================================
            // Depth label
            // ========================================
            CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_8, m_re));
            CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_WHITE, m_re));
            
            float minX = corners_2d[0].x, maxX = corners_2d[0].x, maxY = corners_2d[0].y;
            for (int c = 1; c < 8; ++c) {
                minX = std::min(minX, corners_2d[c].x);
                maxX = std::max(maxX, corners_2d[c].x);
                maxY = std::max(maxY, corners_2d[c].y);
            }
            
            char depthText[32];
            std::snprintf(depthText, sizeof(depthText), "%.1fm", box3D.depth);
            dwVector2f textPos = {(minX + maxX) / 2.0f - 12.0f, maxY + 10.0f};
            CHECK_DW_ERROR(dwRenderEngine_renderText2D(depthText, textPos, m_re));
        }
    }
}


void DriveSeg4CamClient::printStats()
{
    std::cout << "=== Client Stats ===\n";
    std::cout << "Render frames: " << m_frameCounter.load() << "\n";
    for (uint32_t i = 0; i < m_numCameras; ++i) {
        std::lock_guard<std::mutex> lock(m_frameMutex[i]);
        if (m_latestFrames[i].valid) {
            std::cout << "  Cam" << i 
                      << " | Latest frame: " << m_latestFrames[i].frameId
                      << " | 2D Boxes: " << m_latestFrames[i].boxes.size() 
                      << " | 3D Boxes: " << m_latestFrames[i].boxes3D.size()
                      << "\n";
        } else {
            std::cout << "  Cam" << i << " | No frames received yet\n";
        }
    }
}

// ===============================================================
// SIGNAL HANDLER
// ===============================================================

static std::atomic<bool> g_run{true};

extern "C" void sig_int_handler(int)
{
    g_run = false;
    std::cout << "\n[Client] Shutdown signal received\n";
}

// ===============================================================
// MAIN
// ===============================================================

int main(int argc, const char** argv)
{
    ProgramArguments args(argc, argv, {
        ProgramArguments::Option_t(
            "ip",
            "127.0.0.1",
            "Server IP address"
        ),
        ProgramArguments::Option_t(
            "port",
            "49252",
            "Server port"
        ),
        ProgramArguments::Option_t(
            "num-cameras",
            "4",
            "Number of cameras to receive"
        ),
    },
    "Four-camera perception render client");

    // Setup signal handlers
    std::signal(SIGINT, sig_int_handler);

    try {
        DriveSeg4CamClient app(args);
        app.initializeWindow("DriveSeg4Cam Client", 1280, 800, args.enabled("offscreen"));
        if (!args.enabled("offscreen")) app.setProcessRate(30);
        
        return app.run();
        
    } catch (const std::exception& e) {
        std::cerr << "[Client] Exception: " << e.what() << "\n";
        return -1;
    }
}
