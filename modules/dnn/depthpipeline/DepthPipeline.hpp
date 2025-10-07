#pragma once

#include "CameraCapture.hpp"
#include "DepthInferenceEngine.hpp"
#include "DepthRenderer.hpp"
#include "MemoryManager.hpp"

#include <dw/core/context/Context.h>
#include <dw/sensors/common/Sensors.h>

#include <atomic>
#include <thread>
#include <memory>

class WindowGLFW;

namespace depth_pipeline {

/**
 * @brief Main depth estimation pipeline orchestrator
 * 
 * Responsibilities:
 * - Coordinate all pipeline stages
 * - Manage pipeline lifecycle
 * - Handle synchronization between stages
 * - Provide unified performance metrics
 */
class DepthPipeline {
public:
    enum class PipelineMode {
        REAL_TIME,      // 30fps target, prioritize latency
        HIGH_QUALITY,   // Best quality, may drop frames
        DATA_COLLECTION // No visualization, maximum throughput
    };
    
    struct PipelineConfig {
        std::string rigPath;
        std::string depthModelPath;
        PipelineMode mode;
        uint32_t targetFPS;
        bool enableVisualization;
        uint32_t windowWidth;
        uint32_t windowHeight;
    };
    
    struct PipelineStatistics {
        uint32_t totalFramesCaptured;
        uint32_t totalInferencesCompleted;
        uint32_t droppedFrames;
        float currentFPS;
        float avgLatencyMs;
        DepthInferenceEngine::PerformanceMetrics inferenceMetrics;
    };

    /**
     * @brief Constructor
     */
    DepthPipeline();
    
    /**
     * @brief Destructor
     */
    ~DepthPipeline();
    
    /**
     * @brief Initialize pipeline
     * @param config Pipeline configuration
     * @return DW_SUCCESS on success
     */
    dwStatus initialize(const PipelineConfig& config);
    
    /**
     * @brief Start pipeline execution
     * @return DW_SUCCESS on success
     */
    dwStatus start();
    
    /**
     * @brief Stop pipeline execution
     */
    void stop();
    
    /**
     * @brief Execute single pipeline iteration (for external loop control)
     * @return DW_SUCCESS on success
     */
    dwStatus processFrame();
    
    bool isRunning() const { return m_running.load(); }
    bool shouldRender() const;
    void swapBuffers();

    /**
     * @brief Get pipeline statistics
     */
    PipelineStatistics getStatistics() const;


private:
    // DriveWorks core handles
    dwContextHandle_t m_context;
    dwSALHandle_t m_sal;
    
    std::unique_ptr<WindowGLFW> m_window;
    // Pipeline modules
    std::unique_ptr<CameraCapture> m_cameraCapture;
    std::unique_ptr<DepthInferenceEngine> m_inferenceEngine;
    std::unique_ptr<DepthRenderer> m_renderer;
    std::unique_ptr<MemoryManager> m_memoryManager;
    
    PipelineConfig m_config;
    std::atomic<bool> m_running;
    std::atomic<uint32_t> m_frameCounter;
    
    // Performance tracking
    std::chrono::high_resolution_clock::time_point m_pipelineStartTime;
    std::chrono::high_resolution_clock::time_point m_lastStatsUpdate;
    
    /**
     * @brief Initialize DriveWorks context
     */
    dwStatus initializeContext();
    
    /**
     * @brief Pipeline stage: capture frames from cameras
     */
    dwStatus stageCaptureFrames(CameraCapture::CaptureResult& result);
    
    /**
     * @brief Pipeline stage: submit frames for inference
     */
    dwStatus stageSubmitInference(const CameraCapture::CaptureResult& captureResult);
    
    /**
     * @brief Pipeline stage: collect inference results
     */
    dwStatus stageCollectResults(std::vector<DepthInferenceEngine::InferenceResult>& results);
    
    /**
     * @brief Pipeline stage: visualize depth maps
     */
    dwStatus stageVisualize(const std::vector<DepthInferenceEngine::InferenceResult>& results);
    
    // Non-copyable
    DepthPipeline(const DepthPipeline&) = delete;
    DepthPipeline& operator=(const DepthPipeline&) = delete;
};

} // namespace depth_pipeline