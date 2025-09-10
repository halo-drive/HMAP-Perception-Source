#ifndef MULTICAM_DNN_APP_HPP_
#define MULTICAM_DNN_APP_HPP_

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <future>
#include <async>
#include <thread>
#include <chrono>

// DriveWorks Core Framework
#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>

// Sensor Management
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/camera/Camera.h>
#include <dw/rig/Rig.h>

// Image Processing
#include <dw/image/Image.h>
#include <dw/interop/streamer/ImageStreamer.h>

// Visualization Framework
#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Visualization.h>
#include <dwvisualization/interop/ImageStreamer.h>

// Sample Framework
#include <framework/Checks.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/Log.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/SamplesDataPath.hpp>
#include <framework/ScreenshotHelper.hpp>
#include <framework/WindowGLFW.hpp>

// DNN Processing Module
#include "DNNProcessor.hpp"

using namespace dw_samples::common;
using namespace multicam_dnn;

class MultiCameraDNNApp : public DriveWorksSample
{
private:
    // ============================================
    // CAMERA PIPELINE CONSTANTS
    // ============================================
    static constexpr uint32_t MAX_PORTS_COUNT = 4U;
    static constexpr uint32_t MAX_CAMS_PER_PORT = 4U;
    static constexpr uint32_t MAX_CAMERAS = MAX_PORTS_COUNT * MAX_CAMS_PER_PORT;
    static constexpr uint32_t FIRST_CAMERA_IDX = 0U;
    static constexpr uint32_t DEFAULT_FIFO_SIZE = 4U;
    static constexpr uint32_t FRAME_TIMEOUT_US = 333333U; // 333ms timeout

    // ============================================
    // DRIVEWORKS CORE HANDLES
    // ============================================
    dwContextHandle_t m_context = DW_NULL_HANDLE;
    dwSALHandle_t m_sal = DW_NULL_HANDLE;
    dwVisualizationContextHandle_t m_viz = DW_NULL_HANDLE;
    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
    dwRigHandle_t m_rigConfig = DW_NULL_HANDLE;

    // ============================================
    // CAMERA SYSTEM STATE
    // ============================================
    dwSensorHandle_t m_camera[MAX_CAMERAS] = {DW_NULL_HANDLE};
    uint32_t m_totalCameras = 0U;
    bool m_enableRender[MAX_CAMERAS] = {true, true, true, true};
    
    // Camera output format flags
    bool m_useRaw = false;
    bool m_useProcessed = false;
    bool m_useProcessed1 = false;
    bool m_useProcessed2 = false;
    uint32_t m_fifoSize = DEFAULT_FIFO_SIZE;

    // ============================================
    // IMAGE BUFFER MANAGEMENT
    // ============================================
    dwImageHandle_t m_imageRGBA[MAX_CAMERAS] = {DW_NULL_HANDLE};
    dwImageStreamerHandle_t m_streamerToGL[MAX_CAMERAS] = {DW_NULL_HANDLE};
    uint32_t m_imageWidth = 0U;
    uint32_t m_imageHeight = 0U;

    // ============================================
    // RENDERING SYSTEM STATE
    // ============================================
    dwRenderEngineParams m_renderParams{};
    dwRenderEngineColorRGBA m_colorPerPort[MAX_PORTS_COUNT];
    uint32_t m_tileVideo[MAX_CAMERAS];
    dwRectf m_renderRanges[MAX_CAMERAS] = {{0.0f, 0.0f, 0.0f, 0.0f}};

    // ============================================
    // DNN PROCESSING INFRASTRUCTURE
    // ============================================
    std::unique_ptr<DNNProcessor> m_dnnProcessor;
    InferenceResult m_inferenceResults[MAX_CAMERAS];
    
    // ============================================
    // PROCESSING STATE
    // ============================================
    uint32_t m_frameCount = 0U;
    uint32_t m_currentCamera = 0U; // Round-robin camera index
    
    // Screenshot utility
    std::unique_ptr<ScreenshotHelper> m_screenshot;

public:
    // ============================================
    // CONSTRUCTOR & FRAMEWORK INTERFACE
    // ============================================
    MultiCameraDNNApp(const ProgramArguments& args);
    ~MultiCameraDNNApp() override = default;

    // DriveWorksSample interface
    bool onInitialize() override;
    void onProcess() override;
    void onRender() override;
    void onRelease() override;
    void onKeyDown(int key, int scancode, int mods) override;

private:
    // ============================================
    // INITIALIZATION METHODS
    // ============================================
    void initializeDriveWorks();
    void initializeMultiCamera();
    void initializeRenderer();
    void initializeDNNProcessor();
    
    // ============================================
    // CAMERA PIPELINE METHODS
    // ============================================
    void acquireFrames(dwCameraFrameHandle_t frames[MAX_CAMERAS]);
    void processCurrentCamera(dwCameraFrameHandle_t frame);
    void renderCameraFrame(dwCameraFrameHandle_t frame, uint32_t cameraIndex);
    void returnFrames(dwCameraFrameHandle_t frames[MAX_CAMERAS]);
    
    // ============================================
    // DNN INTEGRATION METHODS
    // ============================================
    void runDNNInference(uint32_t cameraIndex);
    void renderDetectionResults(uint32_t cameraIndex);
    void renderSegmentationOverlays(uint32_t cameraIndex);
    
    // ============================================
    // UTILITY METHODS
    // ============================================
    void setupCameraProperties();
    void setupRenderingLayout();
    void advanceCurrentCamera();
    void handleFrameTimeout(uint32_t cameraIndex);
    void logCameraStatus() const;
    void logDNNStatus() const;
};

#endif // MULTICAM_DNN_APP_HPP_