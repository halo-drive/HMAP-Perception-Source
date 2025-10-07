#pragma once

#include <dw/core/context/Context.h>
#include <dw/image/Image.h>
#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Visualization.h>
#include <dw/interop/streamer/ImageStreamer.h>

#include "DepthInferenceEngine.hpp"

#include <memory>
#include <array>
#include <EGL/egl.h>

namespace depth_pipeline {

/**
 * @brief Depth visualization renderer - GPU-accelerated depth map rendering
 * 
 * Responsibilities:
 * - Convert depth tensors to RGBA visualizations (GPU-only)
 * - Stream depth images to OpenGL for display
 * - Multi-tile rendering for 4 cameras
 * - Performance overlay rendering
 */
class DepthRenderer {
public:
    enum class ColorScheme {
        GRAYSCALE,
        JET_COLORMAP,
        TURBO_COLORMAP
    };
    
    struct RenderConfig {
        uint32_t windowWidth;
        uint32_t windowHeight;
        uint32_t numCameras;
        ColorScheme colorScheme;
        bool showPerformanceStats;
    };
    
    struct VisualizationState {
        dwImageHandle_t depthVisualization[4];  // RGBA depth images
        bool dataAvailable[4];
        float depthRange[4][2];  // [min, max] per camera
        std::chrono::microseconds lastUpdateTime[4];
    };

    /**
     * @brief Constructor
     * @param context DriveWorks context handle
     */
    explicit DepthRenderer(dwContextHandle_t context);
    
    /**
     * @brief Destructor
     */
    ~DepthRenderer();
    
    /**
     * @brief Initialize rendering resources
     * @param config Render configuration
     * @return DW_SUCCESS on success
     */
    dwStatus initialize(const RenderConfig& config);
    
    /**
     * @brief Update depth visualization from inference result
     * @param result Inference result containing GPU depth tensor
     * @return DW_SUCCESS on success
     */
    dwStatus updateDepthVisualization(const DepthInferenceEngine::InferenceResult& result);
    
    /**
     * @brief Render all depth visualizations to window
     * @return DW_SUCCESS on success
     */
    dwStatus render();
    
    /**
     * @brief Set color scheme for depth visualization
     */
    void setColorScheme(ColorScheme scheme) { m_colorScheme = scheme; }
    
    /**
     * @brief Toggle performance stats overlay
     */
    void togglePerformanceStats() { m_showStats = !m_showStats; }
    
    /**
     * @brief Get current visualization state
     */
    const VisualizationState& getState() const { return m_state; }

private:
    dwContextHandle_t m_context;
    dwVisualizationContextHandle_t m_viz;
    dwRenderEngineHandle_t m_renderEngine;
    
    RenderConfig m_config;
    ColorScheme m_colorScheme;
    bool m_showStats;
    
    // Per-camera rendering resources
    struct CameraRenderContext {
        dwImageHandle_t depthImageRGBA;    // Visualization image (CUDA)
        dwImageStreamerHandle_t streamerGL; // CUDA->GL streamer
        uint32_t tileId;                    // RenderEngine tile ID
        cudaStream_t cudaStream;            // Dedicated stream for visualization
    };
    
    std::array<CameraRenderContext, 4> m_cameraContexts;
    VisualizationState m_state;
    
    /**
     * @brief Create depth visualization on GPU (CUDA kernel dispatch)
     * @param depthTensor Input depth tensor (GPU)
     * @param outputImage Output RGBA image (GPU)
     * @param minDepth Minimum depth value
     * @param maxDepth Maximum depth value
     * @param stream CUDA stream
     */
    dwStatus createVisualizationGPU(dwDNNTensorHandle_t depthTensor,
                                   dwImageHandle_t outputImage,
                                   float minDepth, float maxDepth,
                                   cudaStream_t stream);
    
    /**
     * @brief Render single camera tile
     */
    dwStatus renderCameraTile(uint32_t cameraIdx);
    
    /**
     * @brief Render performance overlay
     */
    void renderPerformanceOverlay();
    
    // Non-copyable
    DepthRenderer(const DepthRenderer&) = delete;
    DepthRenderer& operator=(const DepthRenderer&) = delete;
};

} // namespace depth_pipeline