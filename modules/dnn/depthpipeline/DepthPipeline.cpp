
#include "DepthPipeline.hpp"
#include <framework/Checks.hpp>
#include <framework/Log.hpp>
#include <framework/WindowGLFW.hpp>

#ifdef VIBRANTE
#include <EGL/egl.h>  
#endif

#include <iostream>
#include <sstream>


namespace depth_pipeline {

DepthPipeline::DepthPipeline()
    : m_context(DW_NULL_HANDLE)
    , m_sal(DW_NULL_HANDLE)
    , m_running(false)
    , m_frameCounter(0)
    , m_pipelineStartTime(std::chrono::high_resolution_clock::now())
    , m_lastStatsUpdate(std::chrono::high_resolution_clock::now())
{
}

DepthPipeline::~DepthPipeline()
{
    if (m_running.load()) {
        stop();
    }
    
    // Release modules in reverse initialization order
    m_renderer.reset();
    m_inferenceEngine.reset();
    m_cameraCapture.reset();
    m_memoryManager.reset();
    
    // Release DriveWorks resources
    if (m_sal != DW_NULL_HANDLE) {
        dwSAL_release(m_sal);
    }
    
    if (m_context != DW_NULL_HANDLE) {
        dwRelease(m_context);
        dwLogger_release();
    }
}

dwStatus DepthPipeline::initialize(const PipelineConfig& config)
{
    m_config = config;
    
    std::cout << "=== Initializing Depth Pipeline ===" << std::endl;
    std::cout << "Mode: " << (config.mode == PipelineMode::REAL_TIME ? "REAL_TIME" :
                              config.mode == PipelineMode::HIGH_QUALITY ? "HIGH_QUALITY" :
                              "DATA_COLLECTION") << std::endl;
    std::cout << "Target FPS: " << config.targetFPS << std::endl;
    std::cout << "Visualization: " << (config.enableVisualization ? "enabled" : "disabled") << std::endl;
    
    // Step 1: Initialize DriveWorks context
    dwStatus status = initializeContext();
    if (status != DW_SUCCESS) {
        std::cerr << "ERROR: Failed to initialize DriveWorks context" << std::endl;
        return status;
    }
    
    // Step 2: Initialize SAL (Sensor Abstraction Layer)
    std::cout << "Initializing SAL..." << std::endl;
    status = dwSAL_initialize(&m_sal, m_context);
    if (status != DW_SUCCESS) {
        std::cerr << "ERROR: Failed to initialize SAL: " << dwGetStatusName(status) << std::endl;
        return status;
    }
    
    // Step 3: Initialize camera capture module
    std::cout << "Initializing camera capture..." << std::endl;
    m_cameraCapture.reset(new CameraCapture(m_context, m_sal));
    
    status = m_cameraCapture->initialize(config.rigPath);
    if (status != DW_SUCCESS) {
        std::cerr << "ERROR: Failed to initialize camera capture" << std::endl;
        return status;
    }
    
    uint32_t numCameras = m_cameraCapture->getCameraCount();
    std::cout << "Camera capture initialized: " << numCameras << " cameras" << std::endl;
    
    // Step 4: Initialize depth inference engine
    std::cout << "Initializing depth inference engine..." << std::endl;
    m_inferenceEngine.reset(new DepthInferenceEngine(m_context));
    
    status = m_inferenceEngine->initialize(config.depthModelPath, numCameras);
    if (status != DW_SUCCESS) {
        std::cerr << "ERROR: Failed to initialize depth inference engine" << std::endl;
        return status;
    }
    
    std::cout << "Depth inference engine initialized" << std::endl;
    
    // Step 5: Initialize renderer (if visualization enabled)
    if (config.enableVisualization) {
        std::cout << "Initializing depth renderer..." << std::endl;
        m_renderer.reset(new DepthRenderer(m_context));
        
        DepthRenderer::RenderConfig renderConfig;
        renderConfig.windowWidth = config.windowWidth;
        renderConfig.windowHeight = config.windowHeight;
        renderConfig.numCameras = numCameras;
        renderConfig.colorScheme = DepthRenderer::ColorScheme::JET_COLORMAP;
        renderConfig.showPerformanceStats = true;
        
        status = m_renderer->initialize(renderConfig);
        if (status != DW_SUCCESS) {
            std::cerr << "ERROR: Failed to initialize depth renderer" << std::endl;
            return status;
        }
        
        std::cout << "Depth renderer initialized" << std::endl;
    }
    
    // Step 6: Initialize memory manager (optional)
    m_memoryManager.reset(new MemoryManager(m_context));
    
    std::cout << "=== Pipeline Initialization Complete ===" << std::endl;
    
    return DW_SUCCESS;
}

dwStatus DepthPipeline::start()
{
    if (m_running.load()) {
        std::cerr << "WARNING: Pipeline already running" << std::endl;
        return DW_SUCCESS;
    }
    
    std::cout << "Starting pipeline..." << std::endl;
    
    // Start camera sensors
    dwStatus status = m_cameraCapture->start();
    if (status != DW_SUCCESS) {
        std::cerr << "ERROR: Failed to start camera capture" << std::endl;
        return status;
    }
    
    // Reset performance counters
    m_frameCounter.store(0);
    m_pipelineStartTime = std::chrono::high_resolution_clock::now();
    m_lastStatsUpdate = m_pipelineStartTime;
    
    m_inferenceEngine->resetMetrics();
    
    m_running.store(true);
    
    std::cout << "Pipeline started successfully" << std::endl;
    
    return DW_SUCCESS;
}

void DepthPipeline::stop()
{
    if (!m_running.load()) {
        return;
    }
    
    std::cout << "Stopping pipeline..." << std::endl;
    
    m_running.store(false);
    
    // Stop camera capture
    if (m_cameraCapture) {
        m_cameraCapture->stop();
    }
    
    std::cout << "Pipeline stopped" << std::endl;
}

dwStatus DepthPipeline::processFrame()
{
    if (!m_running.load()) {
        return DW_NOT_READY;
    }
    
    dwStatus overallStatus = DW_SUCCESS;
    
    // Stage 1: Capture frames from all cameras
    CameraCapture::CaptureResult captureResult;
    dwStatus status = stageCaptureFrames(captureResult);
    if (status != DW_SUCCESS) {
        if (status == DW_TIME_OUT) {
            // Timeout acceptable - retry next iteration
            return DW_TIME_OUT;
        }
        std::cerr << "ERROR: Frame capture stage failed: " << dwGetStatusName(status) << std::endl;
        return status;
    }
    
    // Stage 2: Submit frames for depth inference
    status = stageSubmitInference(captureResult);
    if (status != DW_SUCCESS && status != DW_BUFFER_FULL) {
        std::cerr << "WARNING: Inference submission stage failed: " << dwGetStatusName(status) << std::endl;
        overallStatus = status;
    }
    
    // Stage 3: Collect completed inference results (non-blocking poll)
    std::vector<DepthInferenceEngine::InferenceResult> inferenceResults;
    status = stageCollectResults(inferenceResults);
    if (status != DW_SUCCESS) {
        std::cerr << "WARNING: Result collection stage failed: " << dwGetStatusName(status) << std::endl;
    }
    
    // Stage 4: Visualize depth maps (if enabled and results available)
    if (m_config.enableVisualization && !inferenceResults.empty()) {
        status = stageVisualize(inferenceResults);
        if (status != DW_SUCCESS) {
            std::cerr << "WARNING: Visualization stage failed: " << dwGetStatusName(status) << std::endl;
        }
    }
    
    // Release inference results back to engine
    for (const auto& result : inferenceResults) {
        m_inferenceEngine->releaseResult(result);
    }
    
    // Return captured frames to cameras
    m_cameraCapture->returnFrames(captureResult);
    
    // Increment frame counter
    m_frameCounter.fetch_add(1);
    
    return overallStatus;
}


bool DepthPipeline::shouldRender() const
{
    return m_config.enableVisualization && m_window;
}

void DepthPipeline::swapBuffers()
{
    if (m_window) {
        m_window->swapBuffers();
    }
}

DepthPipeline::PipelineStatistics DepthPipeline::getStatistics() const
{
    PipelineStatistics stats;
    
    stats.totalFramesCaptured = m_frameCounter.load();
    
    // Get inference metrics
    auto inferenceMetrics = m_inferenceEngine->getMetrics();
    stats.inferenceMetrics = inferenceMetrics;
    
    // Compute total inferences
    stats.totalInferencesCompleted = 0;
    for (uint32_t i = 0; i < DepthInferenceEngine::MAX_STREAMS; ++i) {
        stats.totalInferencesCompleted += inferenceMetrics.completedInferences[i];
    }
    
    // Estimate dropped frames (captured but not inferred)
    if (stats.totalFramesCaptured > stats.totalInferencesCompleted) {
        stats.droppedFrames = stats.totalFramesCaptured - stats.totalInferencesCompleted;
    } else {
        stats.droppedFrames = 0;
    }
    
    // Compute pipeline FPS
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(
        currentTime - m_pipelineStartTime
    ).count();
    
    if (elapsedSeconds > 0) {
        stats.currentFPS = static_cast<float>(stats.totalFramesCaptured) / elapsedSeconds;
    } else {
        stats.currentFPS = 0.0f;
    }
    
    // Compute average latency (approximate - from inference time)
    float totalInferenceTime = 0.0f;
    uint32_t inferenceCount = 0;
    for (uint32_t i = 0; i < DepthInferenceEngine::MAX_STREAMS; ++i) {
        if (inferenceMetrics.completedInferences[i] > 0) {
            totalInferenceTime += inferenceMetrics.avgInferenceMs[i] * inferenceMetrics.completedInferences[i];
            inferenceCount += inferenceMetrics.completedInferences[i];
        }
    }
    
    if (inferenceCount > 0) {
        stats.avgLatencyMs = totalInferenceTime / inferenceCount;
    } else {
        stats.avgLatencyMs = 0.0f;
    }
    
    return stats;
}


dwStatus DepthPipeline::initializeContext()
{
    std::cout << "Initializing DriveWorks context..." << std::endl;
    
    // Step 1: Create window FIRST (before DriveWorks context)
    if (m_config.enableVisualization) {
        std::cout << "Creating rendering window..." << std::endl;
        
        try {
            m_window.reset(new WindowGLFW(
                "Depth Pipeline Visualization",
                m_config.windowWidth,
                m_config.windowHeight,
                false  // not offscreen
            ));
            
            if (!m_window) {
                std::cerr << "ERROR: Failed to allocate window" << std::endl;
                return DW_FAILURE;
            }
            
            std::cout << "Window created successfully" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "ERROR: Window creation exception: " << e.what() << std::endl;
            return DW_FAILURE;
        }
    }
    
    // Step 2: Initialize logger
    dwStatus status = dwLogger_initialize(getConsoleLoggerCallback(true));
    if (status != DW_SUCCESS) {
        std::cerr << "ERROR: Failed to initialize logger: " << dwGetStatusName(status) << std::endl;
        return status;
    }
    
    dwLoggerVerbosity logLevel = DW_LOG_VERBOSE;
    if (m_config.mode == PipelineMode::DATA_COLLECTION) {
        logLevel = DW_LOG_WARN;
    }
    dwLogger_setLogLevel(logLevel);
    
    // Step 3: Initialize DriveWorks context WITH EGL display
    dwContextParameters sdkParams{};
    memset(&sdkParams, 0, sizeof(dwContextParameters));
    
#ifdef VIBRANTE
    if (m_window) {
        sdkParams.eglDisplay = m_window->getEGLDisplay();  // GET FROM WINDOW
    }
#endif
    
    status = dwInitialize(&m_context, DW_VERSION, &sdkParams);
    if (status != DW_SUCCESS) {
        std::cerr << "ERROR: Failed to initialize DriveWorks: " << dwGetStatusName(status) << std::endl;
        dwLogger_release();
        return status;
    }
    
    // Step 4: Query GPU information (unchanged)
    int32_t currentGPU = 0;
    dwGPUDeviceProperties gpuProps{};
    
    status = dwContext_getGPUDeviceCurrent(&currentGPU, m_context);
    if (status == DW_SUCCESS) {
        status = dwContext_getGPUProperties(&gpuProps, currentGPU, m_context);
        if (status == DW_SUCCESS) {
            std::cout << "GPU Information:" << std::endl;
            std::cout << "  Compute Capability: " << gpuProps.major << "." << gpuProps.minor << std::endl;
            std::cout << "  Architecture: " << (gpuProps.major == 8 && gpuProps.minor == 7 ? "Ampere (AGX Orin)" : "Other") << std::endl;
            std::cout << "  Memory Bus Width: " << gpuProps.memoryBusWidth << " bits" << std::endl;
            std::cout << "  Type: " << (gpuProps.integrated ? "Integrated" : "Discrete") << std::endl;
        }
    }
    
    std::cout << "DriveWorks SDK initialized" << std::endl;
    
    return DW_SUCCESS;
}

dwStatus DepthPipeline::stageCaptureFrames(CameraCapture::CaptureResult& result)
{
    // Compute timeout based on target FPS
    uint32_t timeoutUs = (1000000 / m_config.targetFPS) + 5000; // +5ms tolerance
    
    dwStatus status = m_cameraCapture->captureFrames(result, timeoutUs);
    
    if (status != DW_SUCCESS) {
        if (status == DW_TIME_OUT && result.validCameraCount > 0) {
            // Partial capture - acceptable
            return DW_SUCCESS;
        }
        return status;
    }
    
    return DW_SUCCESS;
}

dwStatus DepthPipeline::stageSubmitInference(const CameraCapture::CaptureResult& captureResult)
{
    dwStatus overallStatus = DW_SUCCESS;
    
    // Submit inference request for each captured camera frame
    for (uint32_t i = 0; i < captureResult.validCameraCount; ++i) {
        if (captureResult.imagesRGBA[i] == DW_NULL_HANDLE) {
            continue;
        }
        
        DepthInferenceEngine::InferenceRequest request;
        request.cameraId = i;
        request.frameId = captureResult.frameId;
        request.imageRGBA = captureResult.imagesRGBA[i];
        request.submitTime = std::chrono::high_resolution_clock::now();
        
        dwStatus status = m_inferenceEngine->submitInference(request);
        
        if (status == DW_BUFFER_FULL) {
            // Inference engine busy - frame will be dropped
            // This is acceptable for real-time operation
            continue;
        }
        
        if (status != DW_SUCCESS) {
            std::cerr << "ERROR: Failed to submit inference for camera " << i 
                     << ": " << dwGetStatusName(status) << std::endl;
            overallStatus = status;
        }
    }
    
    return overallStatus;
}

dwStatus DepthPipeline::stageCollectResults(std::vector<DepthInferenceEngine::InferenceResult>& results)
{
    // Poll for up to MAX_STREAMS results (non-blocking)
    uint32_t numResults = m_inferenceEngine->pollResults(results, DepthInferenceEngine::MAX_STREAMS);
    
    // Results vector now contains completed inferences (if any)
    // Empty vector is acceptable - no completed inferences this iteration
    
    return DW_SUCCESS;
}

dwStatus DepthPipeline::stageVisualize(const std::vector<DepthInferenceEngine::InferenceResult>& results)
{
    dwStatus overallStatus = DW_SUCCESS;
    
    // Update depth visualizations with inference results
    for (const auto& result : results) {
        if (result.status != DW_SUCCESS) {
            continue; // Skip failed inferences
        }
        
        dwStatus status = m_renderer->updateDepthVisualization(result);
        if (status != DW_SUCCESS) {
            std::cerr << "WARNING: Failed to update depth visualization for camera " 
                     << result.cameraId << ": " << dwGetStatusName(status) << std::endl;
            overallStatus = status;
        }
    }
    
    // Render all depth maps to display
    dwStatus status = m_renderer->render();
    if (status != DW_SUCCESS) {
        std::cerr << "WARNING: Render failed: " << dwGetStatusName(status) << std::endl;
        overallStatus = status;
    }
    
    return overallStatus;
}

} // namespace depth_pipeline