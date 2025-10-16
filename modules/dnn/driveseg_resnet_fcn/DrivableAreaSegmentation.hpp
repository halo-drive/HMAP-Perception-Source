/////////////////////////////////////////////////////////////////////////////////////////
// Drivable Area Segmentation - Single Camera via Rig Configuration
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DRIVABLE_AREA_SEGMENTATION_HPP
#define DRIVABLE_AREA_SEGMENTATION_HPP

#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/image/Image.h>
#include <dw/interop/streamer/ImageStreamer.h>
#include <dw/rig/Rig.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/camera/Camera.h>

// DNN Headers
#include <dw/core/platform/GPUProperties.h>
#include <dw/dnn/DNN.h>
#include <dw/dnn/dataconditioner/DataConditioner.h>
#include <dw/dnn/tensor/Tensor.h>

#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Renderer.h>
#include <dwvisualization/core/Visualization.h>

#include <framework/Checks.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/Log.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/SamplesDataPath.hpp>
#include <framework/WindowGLFW.hpp>

#include <memory>
#include <atomic>
#include <chrono>
#include <string>

using namespace dw_samples::common;

#define NUM_SEGMENTATION_CLASSES 3

class DrivableAreaSegmentation : public DriveWorksSample
{
public:
    explicit DrivableAreaSegmentation(const ProgramArguments& args);
    ~DrivableAreaSegmentation() override = default;

    bool onInitialize() override;
    void onProcess() override final {}
    void onRender() override final;
    void onRelease() override final;
    void onKeyDown(int key, int scancode, int mods) override;

private:
    // Initialization (following reference pattern)
    void initializeDriveWorks(dwContextHandle_t& context) const;
    void initializeCameraFromRig();
    void initializeRenderer();
    void initializeDNN();
    void validateTensorProperties();
    
    // Processing
    void processFrame(dwCameraFrameHandle_t frame);
    void preprocessFrame(dwCameraFrameHandle_t frame);
    void runInference();
    void postprocessSegmentation();
    

    // In private section, add:
    void generateColoredMask(const float* outputLogits);
    dwImageHandle_t m_segmentationOverlay = DW_NULL_HANDLE;
    dwImageStreamerHandle_t m_overlayStreamerToGL = DW_NULL_HANDLE;
    std::unique_ptr<uint8_t[]> m_coloredMaskHost;
    std::unique_ptr<float[]> m_outputLogitsHost;  // Host copy of output tensor

    // Rendering
    void onRenderHelper(dwCameraFrameHandle_t frame);
    void renderPerformanceStats();
    
    // Utility
    std::string getPlatformPrefix();
    void printPerformanceMetrics();
    
    // DriveWorks Context
    dwContextHandle_t m_context = DW_NULL_HANDLE;
    dwVisualizationContextHandle_t m_viz = DW_NULL_HANDLE;
    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
    dwSALHandle_t m_sal = DW_NULL_HANDLE;
    
    // Rig configuration (following reference)
    dwRigHandle_t m_rigConfig = DW_NULL_HANDLE;
    
    // Camera Resources
    dwSensorHandle_t m_camera = DW_NULL_HANDLE;
    dwImageHandle_t m_imageRGBA = DW_NULL_HANDLE;
    dwImageStreamerHandle_t m_streamerToGL = DW_NULL_HANDLE;
    uint32_t m_imageWidth = 0;
    uint32_t m_imageHeight = 0;
    uint32_t m_tile = 0;
    
    // DNN Resources
    dwDNNHandle_t m_dnn = DW_NULL_HANDLE;
    dwDataConditionerHandle_t m_dataConditioner = DW_NULL_HANDLE;
    cudaStream_t m_cudaStream = 0;
    dwDNNTensorHandle_t m_dnnInput = DW_NULL_HANDLE;
    dwDNNTensorHandle_t m_dnnOutput = DW_NULL_HANDLE;
    
    // Tensor properties for validation
    dwDNNTensorProperties m_inputProps;
    dwDNNTensorProperties m_outputProps;
    
    // Processing ROI
    dwRect m_roi;
    
    // Performance
    std::atomic<uint32_t> m_frameCount{0};
    std::chrono::high_resolution_clock::time_point m_lastStatsTime;
    float m_avgInferenceMs = 0.0f;
    uint32_t m_processedFrameCount = 0;
    
    // Config
    std::string m_modelPath;
};

#endif // DRIVABLE_AREA_SEGMENTATION_HPP