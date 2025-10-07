// DepthRenderer.cpp
#include "DepthRenderer.hpp"
#include <framework/Checks.hpp>
#include <framework/Log.hpp>
#include <dwvisualization/interop/ImageStreamer.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <cmath>

namespace depth_pipeline {

// Forward declaration of CUDA kernel launcher (implemented in depth_visualization_kernels.cu)
extern void launchDepthColorizationKernel(
    const float* depthMap,
    uint8_t* outputRGBA,
    uint32_t width,
    uint32_t height,
    size_t pitch,
    float minDepth,
    float maxDepth,
    int colorScheme,
    cudaStream_t stream
);

DepthRenderer::DepthRenderer(dwContextHandle_t context)
    : m_context(context)
    , m_viz(DW_NULL_HANDLE)
    , m_renderEngine(DW_NULL_HANDLE)
    , m_colorScheme(ColorScheme::JET_COLORMAP)
    , m_showStats(true)
{
    // Initialize all camera contexts to safe defaults
    for (uint32_t i = 0; i < 4; ++i) {
        m_cameraContexts[i].cudaStream = nullptr;
        m_cameraContexts[i].streamerGL = DW_NULL_HANDLE;
        m_cameraContexts[i].depthImageRGBA = DW_NULL_HANDLE;
        m_cameraContexts[i].tileId = 0;
    }
    
    // Initialize state
    for (uint32_t i = 0; i < 4; ++i) {
        m_state.dataAvailable[i] = false;
        m_state.depthRange[i][0] = 0.0f;
        m_state.depthRange[i][1] = 1.0f;
        m_state.lastUpdateTime[i] = std::chrono::microseconds(0);
    }
}


DepthRenderer::~DepthRenderer()
{
    std::cout << "Releasing depth renderer resources..." << std::endl;
    
    for (uint32_t i = 0; i < m_config.numCameras && i < m_cameraContexts.size(); ++i) {
        CameraRenderContext& ctx = m_cameraContexts[i];
        
        if (ctx.cudaStream) {
            cudaStreamDestroy(ctx.cudaStream);
        }
        
        if (ctx.streamerGL != DW_NULL_HANDLE) {
            dwImageStreamerGL_release(ctx.streamerGL);
        }
        
        if (ctx.depthImageRGBA != DW_NULL_HANDLE) {
            dwImage_destroy(ctx.depthImageRGBA);
        }
    }
    
    // Release visualization state images
    for (uint32_t i = 0; i < 4; ++i) {
        if (m_state.depthVisualization[i] != DW_NULL_HANDLE) {
            m_state.depthVisualization[i] = DW_NULL_HANDLE;  // Clear reference (owned by context)
        }
    }
    
    // Release render engine and visualization context
    if (m_renderEngine != DW_NULL_HANDLE) {
        dwRenderEngine_release(m_renderEngine);
    }
    
    if (m_viz != DW_NULL_HANDLE) {
        dwVisualizationRelease(m_viz);
    }
    
    std::cout << "Depth renderer released" << std::endl;
}


dwStatus DepthRenderer::initialize(const RenderConfig& config)
{
    m_config = config;
    m_colorScheme = config.colorScheme;
    m_showStats = config.showPerformanceStats;
    
    std::cout << "Initializing depth renderer..." << std::endl;
    std::cout << "  Window: " << config.windowWidth << "×" << config.windowHeight << std::endl;
    std::cout << "  Cameras: " << config.numCameras << std::endl;
    
    // Initialize visualization context
    dwStatus status = dwVisualizationInitialize(&m_viz, m_context);
    if (status != DW_SUCCESS) {
        std::cerr << "ERROR: Failed to initialize visualization context: " 
                 << dwGetStatusName(status) << std::endl;
        return status;
    }
    
    // Initialize render engine (SIMPLIFIED - matching standalone pattern)
    dwRenderEngineParams engineParams;
    CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&engineParams, 
                                                     config.windowWidth, 
                                                     config.windowHeight));
    
    engineParams.defaultTile.lineWidth = 2.0f;
    engineParams.defaultTile.font = DW_RENDER_ENGINE_FONT_VERDANA_20;
    engineParams.maxBufferCount = 1;
    
    float32_t windowSize[2] = {static_cast<float32_t>(config.windowWidth),
                               static_cast<float32_t>(config.windowHeight)};
    engineParams.bounds = {0, 0, windowSize[0], windowSize[1]};
    
    status = dwRenderEngine_initialize(&m_renderEngine, &engineParams, m_viz);
    if (status != DW_SUCCESS) {
        std::cerr << "ERROR: Failed to initialize render engine: " 
                 << dwGetStatusName(status) << std::endl;
        return status;
    }
    
    // Configure tile layout (matching standalone pattern)
    uint32_t tilesPerRow = 1;
    if (config.numCameras == 2) {
        tilesPerRow = 2;
    } else if (config.numCameras >= 3 && config.numCameras <= 4) {
        tilesPerRow = 2;
    } else if (config.numCameras > 4) {
        tilesPerRow = 4;
    }
    
    // Create render tiles
    std::vector<dwRenderEngineTileState> tileStates(config.numCameras);
    std::vector<uint32_t> tileIds(config.numCameras);
    
    for (uint32_t i = 0; i < config.numCameras; ++i) {
        dwRenderEngine_initTileState(&tileStates[i]);
        tileStates[i].modelViewMatrix = DW_IDENTITY_MATRIX4F;
        tileStates[i].font = DW_RENDER_ENGINE_FONT_VERDANA_20;
    }
    
    CHECK_DW_ERROR(dwRenderEngine_addTilesByCount(tileIds.data(), config.numCameras, 
                                                   tilesPerRow, tileStates.data(), 
                                                   m_renderEngine));
    
    // Initialize per-camera rendering resources
    dwImageProperties depthVizProps{};
    depthVizProps.type = DW_IMAGE_CUDA;
    depthVizProps.format = DW_IMAGE_FORMAT_RGBA_UINT8;
    depthVizProps.width = DepthInferenceEngine::DEPTH_MODEL_WIDTH;
    depthVizProps.height = DepthInferenceEngine::DEPTH_MODEL_HEIGHT;
    
    for (uint32_t i = 0; i < config.numCameras; ++i) {
        CameraRenderContext& ctx = m_cameraContexts[i];
        
        // Create CUDA stream
        cudaError_t cudaStatus = cudaStreamCreate(&ctx.cudaStream);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "ERROR: Failed to create CUDA stream for camera " << i 
                     << ": " << cudaGetErrorString(cudaStatus) << std::endl;
            return DW_CUDA_ERROR;
        }
        
        // Create depth visualization image (CUDA)
        CHECK_DW_ERROR(dwImage_create(&ctx.depthImageRGBA, depthVizProps, m_context));
        
        // Create CUDA→GL streamer
        CHECK_DW_ERROR(dwImageStreamerGL_initialize(&ctx.streamerGL, &depthVizProps, 
                                                    DW_IMAGE_GL, m_context));
        
        // Store tile ID
        ctx.tileId = tileIds[i];
        
        // Create state visualization image reference
        m_state.depthVisualization[i] = ctx.depthImageRGBA;
        
        std::cout << "  Camera " << i << " render context initialized (tile " 
                  << ctx.tileId << ")" << std::endl;
    }
    
    std::cout << "Depth renderer initialization complete" << std::endl;
    
    return DW_SUCCESS;
}

dwStatus DepthRenderer::updateDepthVisualization(const DepthInferenceEngine::InferenceResult& result)
{
    if (result.cameraId >= m_config.numCameras) {
        std::cerr << "ERROR: Invalid camera ID " << result.cameraId << std::endl;
        return DW_INVALID_ARGUMENT;
    }
    
    if (result.depthTensorDevice == DW_NULL_HANDLE) {
        std::cerr << "ERROR: Null depth tensor for camera " << result.cameraId << std::endl;
        return DW_INVALID_HANDLE;
    }
    
    CameraRenderContext& ctx = m_cameraContexts[result.cameraId];
    
    // Create GPU visualization from depth tensor
    dwStatus status = createVisualizationGPU(result.depthTensorDevice, ctx.depthImageRGBA,
                                            result.minDepth, result.maxDepth, ctx.cudaStream);
    if (status != DW_SUCCESS) {
        std::cerr << "ERROR: Failed to create visualization for camera " << result.cameraId 
                 << ": " << dwGetStatusName(status) << std::endl;
        return status;
    }
    
    // Update state
    m_state.dataAvailable[result.cameraId] = true;
    m_state.depthRange[result.cameraId][0] = result.minDepth;
    m_state.depthRange[result.cameraId][1] = result.maxDepth;
    m_state.lastUpdateTime[result.cameraId] = result.inferenceTime;
    
    return DW_SUCCESS;
}

dwStatus DepthRenderer::render()
{
    // Render each camera's depth visualization
    for (uint32_t i = 0; i < m_config.numCameras; ++i) {
        if (!m_state.dataAvailable[i]) {
            continue; // No depth data available for this camera yet
        }
        
        dwStatus status = renderCameraTile(i);
        if (status != DW_SUCCESS) {
            std::cerr << "WARNING: Failed to render tile for camera " << i 
                     << ": " << dwGetStatusName(status) << std::endl;
        }
    }
    
    // Render performance overlay
    if (m_showStats) {
        renderPerformanceOverlay();
    }
    
    return DW_SUCCESS;
}

dwStatus DepthRenderer::createVisualizationGPU(dwDNNTensorHandle_t depthTensor,
                                              dwImageHandle_t outputImage,
                                              float minDepth, float maxDepth,
                                              cudaStream_t stream)
{
    // Lock depth tensor to access GPU pointer
    void* depthData = nullptr;
    dwStatus status = dwDNNTensor_lock(&depthData, depthTensor);
    if (status != DW_SUCCESS || depthData == nullptr) {
        std::cerr << "ERROR: Failed to lock depth tensor: " << dwGetStatusName(status) << std::endl;
        return status;
    }
    
    // Get output CUDA image
    dwImageCUDA* outputCuda = nullptr;
    status = dwImage_getCUDA(&outputCuda, outputImage);
    if (status != DW_SUCCESS) {
        dwDNNTensor_unlock(depthTensor);
        std::cerr << "ERROR: Failed to get CUDA image: " << dwGetStatusName(status) << std::endl;
        return status;
    }
    
    // Launch GPU colorization kernel
    launchDepthColorizationKernel(
        reinterpret_cast<const float*>(depthData),
        reinterpret_cast<uint8_t*>(outputCuda->dptr[0]),
        DepthInferenceEngine::DEPTH_MODEL_WIDTH,
        DepthInferenceEngine::DEPTH_MODEL_HEIGHT,
        outputCuda->pitch[0],
        minDepth,
        maxDepth,
        static_cast<int>(m_colorScheme),
        stream
    );
    
    // Check for CUDA errors
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        dwDNNTensor_unlock(depthTensor);
        std::cerr << "ERROR: CUDA kernel launch failed: " << cudaGetErrorString(cudaErr) << std::endl;
        return DW_CUDA_ERROR;
    }
    
    dwDNNTensor_unlock(depthTensor);
    
    return DW_SUCCESS;
}

dwStatus DepthRenderer::renderCameraTile(uint32_t cameraIdx)
{
    CameraRenderContext& ctx = m_cameraContexts[cameraIdx];
    
    // Set active tile
    CHECK_DW_ERROR(dwRenderEngine_setTile(ctx.tileId, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_resetTile(m_renderEngine));
    
    // Stream CUDA image to GL
    CHECK_DW_ERROR(dwImageStreamerGL_producerSend(ctx.depthImageRGBA, ctx.streamerGL));
    
    dwImageHandle_t depthGL = DW_NULL_HANDLE;
    CHECK_DW_ERROR(dwImageStreamerGL_consumerReceive(&depthGL, 33000, ctx.streamerGL));
    
    dwImageGL* depthImageGL = nullptr;
    CHECK_DW_ERROR(dwImage_getGL(&depthImageGL, depthGL));
    
    // Set coordinate system
    dwVector2f range{};
    range.x = static_cast<float32_t>(depthImageGL->prop.width);
    range.y = static_cast<float32_t>(depthImageGL->prop.height);
    CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));
    
    // Render depth image
    dwRectf renderRect = {0.0f, 0.0f, range.x, range.y};
    CHECK_DW_ERROR(dwRenderEngine_renderImage2D(depthImageGL, renderRect, m_renderEngine));
    
    // Return GL image
    CHECK_DW_ERROR(dwImageStreamerGL_consumerReturn(&depthGL, ctx.streamerGL));
    CHECK_DW_ERROR(dwImageStreamerGL_producerReturn(nullptr, 32000, ctx.streamerGL));
    
    // Render text overlay with depth range
    std::stringstream ss;
    ss << "Camera " << cameraIdx;
    ss << " | Depth: [" << std::fixed << std::setprecision(2) 
       << m_state.depthRange[cameraIdx][0] << ", " 
       << m_state.depthRange[cameraIdx][1] << "]";
    ss << " | " << (m_state.lastUpdateTime[cameraIdx].count() / 1000.0f) << "ms";
    
    CHECK_DW_ERROR(dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_renderText2D(ss.str().c_str(), {10.0f, 20.0f}, m_renderEngine));
    
    return DW_SUCCESS;
}

void DepthRenderer::renderPerformanceOverlay()
{
    // Performance overlay rendered on first tile
    CHECK_DW_ERROR(dwRenderEngine_setTile(m_cameraContexts[0].tileId, m_renderEngine));
    
    // Render simple status text
    std::string status = "Depth Rendering Active";
    
    CHECK_DW_ERROR(dwRenderEngine_setColor({0.0f, 1.0f, 0.0f, 1.0f}, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_renderText2D(status.c_str(), {10.0f, 50.0f}, m_renderEngine));
}

} // namespace depth_pipeline