/////////////////////////////////////////////////////////////////////////////////////////
// Multi-Stream Camera Depth Estimation Application
// Per-camera CUDA streams with DepthAnythingV2 integration
/////////////////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <mutex>
#include <thread>

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
#include <dw/interop/streamer/TensorStreamer.h>

#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Renderer.h>
#include <dwvisualization/core/Visualization.h>
#include <dwvisualization/image/Image.h>
#include <dwvisualization/interop/ImageStreamer.h>

#include <framework/Checks.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/Log.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/SamplesDataPath.hpp>
#include <framework/ScreenshotHelper.hpp>
#include <framework/WindowGLFW.hpp>

using namespace dw_samples::common;

#define MAX_PORTS_COUNT 4 
#define MAX_CAMS_PER_PORT 4
#define MAX_CAMS MAX_PORTS_COUNT * MAX_CAMS_PER_PORT

// Depth model constants
static constexpr uint32_t DEPTH_MODEL_WIDTH = 924;
static constexpr uint32_t DEPTH_MODEL_HEIGHT = 518;

// Per-Camera Depth Processing Context
struct CameraDepthContext {
    // CUDA stream for this camera
    cudaStream_t cudaStream;
    
    // Depth DNN resources per camera
    dwDNNTensorHandle_t depthInput;
    dwDNNTensorHandle_t depthOutputDevice;
    dwDataConditionerHandle_t depthDataConditioner;
    
    // Tensor streamer for depth output (GPU->CPU transfer)
    dwDNNTensorStreamerHandle_t depthOutputStreamer;
    dwDNNTensorHandle_t depthOutputHost;
    
    // Processing state
    std::atomic<bool> inferenceInProgress;
    std::atomic<uint64_t> frameId;
    std::atomic<bool> resultsReady;
    
    // CUDA synchronization
    cudaEvent_t frameReadyEvent;
    cudaEvent_t inferenceCompleteEvent;
    
    // Depth map storage (thread-safe access needed)
    std::vector<float32_t> depthMap;  // Stores 518×924 depth values
    uint32_t depthWidth;
    uint32_t depthHeight;
    float32_t minDepth;  // For normalization/visualization
    float32_t maxDepth;
    mutable std::mutex depthMutex;
    
    // Depth map for visualization (RGBA format)
    dwImageHandle_t depthVisualizationImage;
    
    // Performance metrics
    std::chrono::high_resolution_clock::time_point lastProcessTime;
    float avgInferenceTimeMs;
    uint32_t processedFrameCount;
    
    CameraDepthContext() : 
        cudaStream(0),
        depthInput(DW_NULL_HANDLE),
        depthOutputDevice(DW_NULL_HANDLE),
        depthDataConditioner(DW_NULL_HANDLE),
        depthOutputStreamer(DW_NULL_HANDLE),
        depthOutputHost(DW_NULL_HANDLE),
        inferenceInProgress(false),
        frameId(0),
        resultsReady(false),
        frameReadyEvent(nullptr),
        inferenceCompleteEvent(nullptr),
        depthVisualizationImage(DW_NULL_HANDLE),
        depthWidth(DEPTH_MODEL_WIDTH),
        depthHeight(DEPTH_MODEL_HEIGHT),
        minDepth(0.0f),
        maxDepth(1.0f),
        avgInferenceTimeMs(0.0f),
        processedFrameCount(0)
    {
        depthMap.resize(DEPTH_MODEL_WIDTH * DEPTH_MODEL_HEIGHT, 0.0f);
    }
};

class MultiStreamDepthEstimationApp : public DriveWorksSample
{
private:
    // ============================================
    // MULTI-CAMERA SECTION 
    // ============================================
    dwContextHandle_t m_context           = DW_NULL_HANDLE;
    dwVisualizationContextHandle_t m_viz  = DW_NULL_HANDLE;
    dwRendererHandle_t m_renderer         = DW_NULL_HANDLE;
    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
    dwRenderEngineParams params{};

    dwSALHandle_t m_sal             = DW_NULL_HANDLE;
    dwSensorHandle_t m_cameraMaster = DW_NULL_HANDLE;
    uint32_t m_totalCameras;

    dwRigHandle_t m_rigConfig{};

    // Per camera resources
    dwSensorHandle_t m_camera[MAX_CAMS];
    dwImageStreamerHandle_t m_streamerToGL[MAX_CAMS] = {DW_NULL_HANDLE};
    dwImageStreamerHandle_t m_depthStreamerToGL[MAX_CAMS] = {DW_NULL_HANDLE};
    uint32_t m_tileVideo[MAX_CAMS];
    bool m_enableRender[MAX_CAMS] = {true};

    bool m_useProcessed  = false;
    uint32_t m_fifoSize  = 0U;
    std::atomic<uint32_t> m_frameCount{0};

    // ============================================
    // DEPTH ESTIMATION SECTION
    // ============================================
    
    // Shared depth DNN model (read-only after initialization)
    dwDNNHandle_t m_depthDNN = DW_NULL_HANDLE;
    
    // Per-camera depth processing contexts
    std::unique_ptr<CameraDepthContext[]> m_cameraContexts;
    
    // Shared image resources
    dwImageHandle_t m_imageRGBA[MAX_CAMS];
    uint32_t m_imageWidth;
    uint32_t m_imageHeight;
    
    // Visualization mode
    enum DepthVisualizationMode {
        DEPTH_VIZ_GRAYSCALE,
        DEPTH_VIZ_HEATMAP,
        DEPTH_VIZ_OVERLAY
    };
    DepthVisualizationMode m_depthVizMode = DEPTH_VIZ_HEATMAP;

    std::unique_ptr<ScreenshotHelper> m_screenshot;
    
    // Performance monitoring
    std::chrono::high_resolution_clock::time_point m_lastStatsTime;
    uint32_t m_totalInferencesPerSecond = 0;

public:
    MultiStreamDepthEstimationApp(const ProgramArguments& args);

    void initializeDriveWorks(dwContextHandle_t& context) const;
    
    // Sample framework
    void onProcess() override final {}
    void onRender() override final;
    void onRelease() override final;
    void onKeyDown(int key, int scancode, int mods) override;
    bool onInitialize() override;

private:
    // Initialization methods
    void initializeMultiCamera();
    void initializeRenderer();
    void initializeDepthDNN();
    void initializeCameraDepthContext(uint32_t cameraIdx);
    
    // Multi-stream processing methods  
    void processAllCamerasParallel(dwCameraFrameHandle_t frames[]);
    bool startDepthInferenceAsync(uint32_t cameraIdx, dwCameraFrameHandle_t frame);
    void checkAndCollectDepthResults(uint32_t cameraIdx);
    
    // Depth processing methods
    void prepareDepthInput(uint32_t cameraIdx, dwCameraFrameHandle_t frame, CameraDepthContext& ctx);
    void doDepthInferenceAsync(uint32_t cameraIdx, CameraDepthContext& ctx);
    void collectDepthResults(uint32_t cameraIdx, CameraDepthContext& ctx);
    void processDepthOutput(uint32_t cameraIdx, CameraDepthContext& ctx);
    
    // Visualization methods
    void onRenderHelper(dwCameraFrameHandle_t frame, uint8_t cameraIndex);
    void visualizeDepthMap(uint32_t cameraIdx);
    void createDepthVisualization(uint32_t cameraIdx, CameraDepthContext& ctx);
    void renderPerformanceStats(uint32_t cameraIdx);
    
    // Utility methods
    std::string getPlatformPrefix();
    dwVector4f depthToColor(float normalizedDepth);
    
    // Performance monitoring
    void updatePerformanceStats();
    void printPerformanceMetrics();
};

//#######################################################################################
MultiStreamDepthEstimationApp::MultiStreamDepthEstimationApp(const ProgramArguments& args)
    : DriveWorksSample(args)
{
    m_cameraContexts.reset(new CameraDepthContext[MAX_CAMS]);
    m_lastStatsTime = std::chrono::high_resolution_clock::now();
}

void MultiStreamDepthEstimationApp::initializeDriveWorks(dwContextHandle_t& context) const
{
    CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
    CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));

    dwContextParameters sdkParams = {};
#ifdef VIBRANTE
    sdkParams.eglDisplay = getEGLDisplay();
#endif
    CHECK_DW_ERROR(dwInitialize(&context, DW_VERSION, &sdkParams));
}

void MultiStreamDepthEstimationApp::initializeMultiCamera()
{
    m_totalCameras = 0;
    CHECK_DW_ERROR(dwRig_initializeFromFile(&m_rigConfig, m_context,
                                           getArgument("rig").c_str()));

    uint32_t cnt = 0;
    CHECK_DW_ERROR(dwRig_getSensorCountOfType(&cnt, DW_SENSOR_CAMERA, m_rigConfig));

    dwSensorParams paramsClient[MAX_CAMS] = {};
    for (uint32_t i = 0; i < cnt; i++)
    {
        uint32_t cameraSensorIdx = 0;
        CHECK_DW_ERROR(dwRig_findSensorByTypeIndex(&cameraSensorIdx, DW_SENSOR_CAMERA, i, m_rigConfig));

        const char* protocol = nullptr;
        CHECK_DW_ERROR(dwRig_getSensorProtocol(&protocol, cameraSensorIdx, m_rigConfig));
        const char* params = nullptr;
        CHECK_DW_ERROR(dwRig_getSensorParameterUpdatedPath(&params, cameraSensorIdx, m_rigConfig));
        paramsClient[i].protocol   = protocol;
        paramsClient[i].parameters = params;

        m_enableRender[i] = true;

        std::cout << "Initializing camera " << i << " with params: " << params << std::endl;
        CHECK_DW_ERROR(dwSAL_createSensor(&m_camera[m_totalCameras], paramsClient[i], m_sal));

        m_totalCameras++;

        m_useProcessed = std::string::npos != std::string(params).find("processed");
        if (!m_useProcessed)
            m_useProcessed = true;

        // Parse FIFO size
        auto posFifoSize = std::string(params).find("fifo-size=");
        if (posFifoSize != std::string::npos)
        {
            static constexpr uint32_t COMMA_INCLUDED_STR_MAX_SIZE = 5;
            auto fifoSizeStr = std::string(params).substr(posFifoSize + std::string("fifo-size=").size(), COMMA_INCLUDED_STR_MAX_SIZE);
            auto fifoSizeStrDataPos = fifoSizeStr.find(",");
            if ((fifoSizeStrDataPos != std::string::npos) && (fifoSizeStr.size() != 0))
            {
                m_fifoSize = std::stoi(fifoSizeStr.substr(0, fifoSizeStrDataPos));
            }
        }

        if (m_fifoSize == 0)
        {
            m_fifoSize = 4;
        }
    }

    // Get image properties from first camera
    dwImageProperties imageProperties{};
    CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&imageProperties, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, m_camera[0]));
    m_imageWidth = imageProperties.width;
    m_imageHeight = imageProperties.height;

    std::cout << "Detected camera resolution: " << m_imageWidth << "x" << m_imageHeight << std::endl;
    std::cout << "Depth model resolution: " << DEPTH_MODEL_WIDTH << "x" << DEPTH_MODEL_HEIGHT << std::endl;

    // Create image and streaming resources for each camera
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        CHECK_DW_ERROR(dwImage_create(&m_imageRGBA[i], imageProperties, m_context));
        CHECK_DW_ERROR(dwImageStreamerGL_initialize(&m_streamerToGL[i], &imageProperties, DW_IMAGE_GL, m_context));
        
        // Create streamer for depth visualization
        dwImageProperties depthVizProps = imageProperties;
        depthVizProps.width = DEPTH_MODEL_WIDTH;
        depthVizProps.height = DEPTH_MODEL_HEIGHT;
        depthVizProps.type = DW_IMAGE_CUDA;
        depthVizProps.format = DW_IMAGE_FORMAT_RGBA_UINT8;
        CHECK_DW_ERROR(dwImageStreamerGL_initialize(&m_depthStreamerToGL[i], &depthVizProps, DW_IMAGE_GL, m_context));
    }
}


void MultiStreamDepthEstimationApp::initializeDepthDNN()
{
    std::string depthModel = getArgument("depth_model");
    if (depthModel.empty())
    {
        depthModel = dw_samples::SamplesDataPath::get() + "/samples/detector/";
        depthModel += getPlatformPrefix();
        depthModel += "/depth_anything_v2_fp32.bin";
    }

    std::cout << "Loading Depth model: " << depthModel << std::endl;
    
    // Initialize shared depth DNN model
    CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFile(&m_depthDNN, depthModel.c_str(), nullptr,
                                                    DW_PROCESSOR_TYPE_GPU, m_context));

    // Get depth model tensor properties
    dwDNNTensorProperties depthInputProps;
    dwDNNTensorProperties depthOutputProps;

    CHECK_DW_ERROR(dwDNN_getInputTensorProperties(&depthInputProps, 0U, m_depthDNN));

    std::cout << "Model expects input type: " << depthInputProps.dataType << std::endl;
    std::cout << "  DW_TYPE_FLOAT16 = " << DW_TYPE_FLOAT16 << std::endl;
    std::cout << "  DW_TYPE_FLOAT32 = " << DW_TYPE_FLOAT32 << std::endl;

    CHECK_DW_ERROR(dwDNN_getOutputTensorProperties(&depthOutputProps, 0U, m_depthDNN));

    // Print raw dimensions for debugging
    std::cout << "Depth Model loaded successfully" << std::endl;
    std::cout << "Input tensor properties:" << std::endl;
    std::cout << "  Number of dimensions: " << depthInputProps.numDimensions << std::endl;
    std::cout << "  Tensor layout: " << depthInputProps.tensorLayout << std::endl;
    std::cout << "  Dimensions (in reverse layout order): [";
    for (uint32_t i = 0; i < depthInputProps.numDimensions; ++i)
    {
        std::cout << depthInputProps.dimensionSize[i];
        if (i < depthInputProps.numDimensions - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Data type: " << (depthInputProps.dataType == DW_TYPE_FLOAT32 ? "FP32" : 
                                     depthInputProps.dataType == DW_TYPE_FLOAT16 ? "FP16" : "OTHER") << std::endl;
    
    std::cout << "Output tensor properties:" << std::endl;
    std::cout << "  Number of dimensions: " << depthOutputProps.numDimensions << std::endl;
    std::cout << "  Tensor layout: " << depthOutputProps.tensorLayout << std::endl;
    std::cout << "  Dimensions (in reverse layout order): [";
    for (uint32_t i = 0; i < depthOutputProps.numDimensions; ++i)
    {
        std::cout << depthOutputProps.dimensionSize[i];
        if (i < depthOutputProps.numDimensions - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Extract actual dimensions based on tensor layout
    // DriveWorks documentation states: dimensions are in REVERSE order of the layout suffix
    // For NCHW: dimensionSize[0]=W, [1]=H, [2]=C, [3]=N
    
    uint32_t actualBatch, actualChannels, actualHeight, actualWidth;
    uint32_t outputBatch, outputHeight, outputWidth;
    
    // Interpret input dimensions (should be NCHW format)
    if (depthInputProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NCHW && depthInputProps.numDimensions == 4)
    {
        // Reverse order: [0]=W, [1]=H, [2]=C, [3]=N
        actualWidth = depthInputProps.dimensionSize[0];
        actualHeight = depthInputProps.dimensionSize[1];
        actualChannels = depthInputProps.dimensionSize[2];
        actualBatch = depthInputProps.dimensionSize[3];
    }
    else if (depthInputProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NHWC && depthInputProps.numDimensions == 4)
    {
        // Reverse order: [0]=C, [1]=W, [2]=H, [3]=N
        actualChannels = depthInputProps.dimensionSize[0];
        actualWidth = depthInputProps.dimensionSize[1];
        actualHeight = depthInputProps.dimensionSize[2];
        actualBatch = depthInputProps.dimensionSize[3];
    }
    else
    {
        std::stringstream ss;
        ss << "Unsupported input tensor layout: " << depthInputProps.tensorLayout 
           << " with " << depthInputProps.numDimensions << " dimensions";
        throw std::runtime_error(ss.str());
    }
    
    // Interpret output dimensions
    // For 3D tensors, deduce dimensions by their values
    // Expected: one dimension is 1 (batch or channels), others are 518 and 924
    if (depthOutputProps.numDimensions == 3)
    {
        // Initialize to invalid values
        outputBatch = 0;
        outputHeight = 0;
        outputWidth = 0;
        
        // Find dimensions by their characteristic values
        for (uint32_t i = 0; i < 3; ++i)
        {
            if (depthOutputProps.dimensionSize[i] == 1)
            {
                outputBatch = depthOutputProps.dimensionSize[i];
            }
            else if (depthOutputProps.dimensionSize[i] == 518)
            {
                outputHeight = depthOutputProps.dimensionSize[i];
            }
            else if (depthOutputProps.dimensionSize[i] == 924)
            {
                outputWidth = depthOutputProps.dimensionSize[i];
            }
        }
        
        // Verify we found all dimensions
        if (outputBatch == 0 || outputHeight == 0 || outputWidth == 0)
        {
            std::stringstream ss;
            ss << "Could not identify output dimensions from values: ["
               << depthOutputProps.dimensionSize[0] << ", "
               << depthOutputProps.dimensionSize[1] << ", "
               << depthOutputProps.dimensionSize[2] << "]";
            throw std::runtime_error(ss.str());
        }
    }
    else if (depthOutputProps.numDimensions == 4)
    {
        // Handle 4D output case (unlikely but possible)
        if (depthOutputProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NCHW)
        {
            outputWidth = depthOutputProps.dimensionSize[0];
            outputHeight = depthOutputProps.dimensionSize[1];
            // Skip channels dimension
            outputBatch = depthOutputProps.dimensionSize[3];
        }
        else if (depthOutputProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NHWC)
        {
            // Skip channels dimension
            outputWidth = depthOutputProps.dimensionSize[1];
            outputHeight = depthOutputProps.dimensionSize[2];
            outputBatch = depthOutputProps.dimensionSize[3];
        }
    }
    else
    {
        std::stringstream ss;
        ss << "Unsupported output tensor dimensions: " << depthOutputProps.numDimensions;
        throw std::runtime_error(ss.str());
    }
    
    std::cout << "\nInterpreted dimensions:" << std::endl;
    std::cout << "  Input: Batch=" << actualBatch << ", Channels=" << actualChannels 
              << ", Height=" << actualHeight << ", Width=" << actualWidth << std::endl;
    std::cout << "  Output: Batch/Channels=" << outputBatch << ", Height=" << outputHeight 
              << ", Width=" << outputWidth << std::endl;

    // Verify dimensions match expectations
    if (actualHeight != DEPTH_MODEL_HEIGHT || actualWidth != DEPTH_MODEL_WIDTH)
    {
        std::stringstream ss;
        ss << "Depth model input dimensions mismatch. Expected " << DEPTH_MODEL_HEIGHT << "x" << DEPTH_MODEL_WIDTH
           << " but got " << actualHeight << "x" << actualWidth;
        throw std::runtime_error(ss.str());
    }
    
    if (outputHeight != DEPTH_MODEL_HEIGHT || outputWidth != DEPTH_MODEL_WIDTH)
    {
        std::stringstream ss;
        ss << "Depth output dimensions mismatch. Expected " << DEPTH_MODEL_HEIGHT << "x" << DEPTH_MODEL_WIDTH
           << " but got " << outputHeight << "x" << outputWidth;
        throw std::runtime_error(ss.str());
    }

    std::cout << "Dimension verification passed!" << std::endl;

    // Initialize per-camera depth contexts
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        initializeCameraDepthContext(i);
    }

    std::cout << "Multi-stream depth estimation initialized for " << m_totalCameras << " cameras" << std::endl;
}

void MultiStreamDepthEstimationApp::initializeCameraDepthContext(uint32_t cameraIdx)
{
    CameraDepthContext& ctx = m_cameraContexts[cameraIdx];
    
    std::cout << "Initializing depth context for camera " << cameraIdx << std::endl;
    
    // Create dedicated CUDA stream
    cudaError_t cudaStatus = cudaStreamCreate(&ctx.cudaStream);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream for camera " + std::to_string(cameraIdx));
    }
    
    // Create CUDA events for synchronization
    cudaStatus = cudaEventCreate(&ctx.frameReadyEvent);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Failed to create frame ready event for camera " + std::to_string(cameraIdx));
    }
    
    cudaStatus = cudaEventCreate(&ctx.inferenceCompleteEvent);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Failed to create inference complete event for camera " + std::to_string(cameraIdx));
    }
    
    // Get tensor properties
    dwDNNTensorProperties depthInputProps;
    dwDNNTensorProperties depthOutputProps;
    
    CHECK_DW_ERROR(dwDNN_getInputTensorProperties(&depthInputProps, 0U, m_depthDNN));
    CHECK_DW_ERROR(dwDNN_getOutputTensorProperties(&depthOutputProps, 0U, m_depthDNN));
    
    // Create per-camera depth tensors
    CHECK_DW_ERROR(dwDNNTensor_create(&ctx.depthInput, &depthInputProps, m_context));
    CHECK_DW_ERROR(dwDNNTensor_create(&ctx.depthOutputDevice, &depthOutputProps, m_context));
    
    // Create tensor streamer for GPU->CPU transfer of depth output
    dwDNNTensorProperties hostProps = depthOutputProps;
    hostProps.tensorType = DW_DNN_TENSOR_TYPE_CPU;
    CHECK_DW_ERROR(dwDNNTensorStreamer_initialize(&ctx.depthOutputStreamer,
                                                  &depthOutputProps,
                                                  hostProps.tensorType, m_context));
    
    // Create per-camera data conditioner
    dwDNNMetaData metadata;
    CHECK_DW_ERROR(dwDNN_getMetaData(&metadata, m_depthDNN));
    
    // Verify data conditioner params (should be DINOv2 ImageNet preprocessing)
    std::cout << "  Data conditioner mean: [" 
              << metadata.dataConditionerParams.meanValue[0] << ", "
              << metadata.dataConditionerParams.meanValue[1] << ", "
              << metadata.dataConditionerParams.meanValue[2] << "]" << std::endl;
    std::cout << "  Data conditioner std: ["
              << metadata.dataConditionerParams.stdev[0] << ", "
              << metadata.dataConditionerParams.stdev[1] << ", "
              << metadata.dataConditionerParams.stdev[2] << "]" << std::endl;
    
    CHECK_DW_ERROR(dwDataConditioner_initializeFromTensorProperties(&ctx.depthDataConditioner, 
                                                                   &depthInputProps, 1U,
                                                                   &metadata.dataConditionerParams, 
                                                                   ctx.cudaStream,
                                                                   m_context));
    
    // Create depth visualization image
    dwImageProperties depthVizProps{};
    depthVizProps.type = DW_IMAGE_CUDA;
    depthVizProps.format = DW_IMAGE_FORMAT_RGBA_UINT8;
    depthVizProps.width = DEPTH_MODEL_WIDTH;
    depthVizProps.height = DEPTH_MODEL_HEIGHT;
    CHECK_DW_ERROR(dwImage_create(&ctx.depthVisualizationImage, depthVizProps, m_context));
    
    std::cout << "Depth context " << cameraIdx << " initialized successfully" << std::endl;
}

void MultiStreamDepthEstimationApp::initializeRenderer()
{
    CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_context));

    std::cout << "Total cameras for rendering: " << m_totalCameras << std::endl;

    CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&params, getWindowWidth(), getWindowHeight()));
    params.defaultTile.lineWidth = 2.0f;
    params.defaultTile.font      = DW_RENDER_ENGINE_FONT_VERDANA_20;
    params.maxBufferCount        = 1;

    float32_t windowSize[2] = {static_cast<float32_t>(getWindowWidth()), static_cast<float32_t>(getWindowHeight())};
    params.bounds = {0, 0, 0, 0};

    uint32_t tilesPerRow = 1;
    params.bounds.width  = windowSize[0];
    params.bounds.height = windowSize[1];
    
    switch (m_totalCameras)
    {
    case 1:
        tilesPerRow = 1;
        break;
    case 2:
        params.bounds.height = (windowSize[1] / 2);
        params.bounds.y      = (windowSize[1] / 2);
        tilesPerRow          = 2;
        break;
    case 3:
    case 4:
        tilesPerRow = 2;
        break;
    default:
        tilesPerRow = 4;
        break;
    }

    CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_viz));

    dwRenderEngineTileState paramList[MAX_CAMS];
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        dwRenderEngine_initTileState(&paramList[i]);
        paramList[i].modelViewMatrix = DW_IDENTITY_MATRIX4F;
        paramList[i].font            = DW_RENDER_ENGINE_FONT_VERDANA_20;
    }

    CHECK_DW_ERROR(dwRenderEngine_addTilesByCount(m_tileVideo, m_totalCameras, tilesPerRow, paramList, m_renderEngine));
}

bool MultiStreamDepthEstimationApp::onInitialize()
{
    initializeDriveWorks(m_context);
    CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_context));

    initializeMultiCamera();
    CHECK_DW_ERROR(dwSAL_start(m_sal));
    initializeRenderer();
    initializeDepthDNN();

    m_screenshot.reset(new ScreenshotHelper(m_context, m_sal, getWindowWidth(), getWindowHeight(), "MultiStreamDepth"));

    std::cout << "Starting all camera sensors..." << std::endl;
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        CHECK_DW_ERROR(dwSensor_start(m_camera[i]));
    }

    std::cout << "Multi-stream depth estimation application initialized successfully" << std::endl;
    return true;
}

void MultiStreamDepthEstimationApp::onRender()
{
    // Capture frames from all cameras
    dwCameraFrameHandle_t frames[MAX_CAMS];
    
    // Read all camera frames
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        dwStatus status = DW_NOT_READY;
        uint32_t countFailure = 0;
        
        while ((status == DW_NOT_READY) || (status == DW_END_OF_STREAM))
        {
            status = dwSensorCamera_readFrame(&frames[i], 33333, m_camera[i]);
            countFailure++;
            if (countFailure == 30000)
            {
                std::cout << "Camera " << i << " timeout - skipping frame" << std::endl;
                break;
            }

            if (status == DW_END_OF_STREAM)
            {
                dwSensor_reset(m_camera[i]);
                std::cout << "Camera " << i << " reached end of stream" << std::endl;
            }
        }

        if (status == DW_TIME_OUT || status != DW_SUCCESS)
        {
            continue;
        }
    }

    // Process all cameras in parallel for depth estimation
    processAllCamerasParallel(frames);

    // Render all cameras with their depth maps
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        if (m_useProcessed && m_enableRender[i])
        {
            onRenderHelper(frames[i], i);
        }
    }

    // Return all frames
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        CHECK_DW_ERROR(dwSensorCamera_returnFrame(&frames[i]));
    }

    m_frameCount++;
    updatePerformanceStats();
}

void MultiStreamDepthEstimationApp::processAllCamerasParallel(dwCameraFrameHandle_t frames[])
{
    // Step 1: Start asynchronous depth inference on all cameras that aren't currently processing
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        CameraDepthContext& ctx = m_cameraContexts[i];
        
        if (!ctx.inferenceInProgress.load())
        {
            if (startDepthInferenceAsync(i, frames[i]))
            {
                ctx.frameId.store(m_frameCount.load());
                ctx.inferenceInProgress.store(true);
            }
        }
    }
    
    // Step 2: Check for completed inferences and collect depth results
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        checkAndCollectDepthResults(i);
    }
}

bool MultiStreamDepthEstimationApp::startDepthInferenceAsync(uint32_t cameraIdx, dwCameraFrameHandle_t frame)
{
    CameraDepthContext& ctx = m_cameraContexts[cameraIdx];
    cudaError_t cudaStatus;
    
    try
    {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Prepare input frame (resize to 518x924 and preprocess)
        prepareDepthInput(cameraIdx, frame, ctx);
        
        // Record event to mark frame preparation complete
        cudaStatus = cudaEventRecord(ctx.frameReadyEvent, ctx.cudaStream);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("Failed to record frame ready event for camera " + std::to_string(cameraIdx));
        }
        
        // Start asynchronous depth inference
        doDepthInferenceAsync(cameraIdx, ctx);
        
        // Record event to mark inference dispatch complete
        cudaStatus = cudaEventRecord(ctx.inferenceCompleteEvent, ctx.cudaStream);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("Failed to record inference complete event for camera " + std::to_string(cameraIdx));
        }
        
        ctx.lastProcessTime = startTime;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cout << "Error starting depth inference for camera " << cameraIdx << ": " << e.what() << std::endl;
        ctx.inferenceInProgress.store(false);
        return false;
    }
}

void MultiStreamDepthEstimationApp::checkAndCollectDepthResults(uint32_t cameraIdx)
{
    CameraDepthContext& ctx = m_cameraContexts[cameraIdx];
    
    if (!ctx.inferenceInProgress.load())
        return;
    
    // Check if inference is complete (non-blocking)
    cudaError_t eventStatus = cudaEventQuery(ctx.inferenceCompleteEvent);
    
    if (eventStatus == cudaSuccess)
    {
        // Inference is complete - collect depth results
        try
        {
            collectDepthResults(cameraIdx, ctx);
            
            // Update performance metrics
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - ctx.lastProcessTime);
            float inferenceTimeMs = duration.count() / 1000.0f;
            
            ctx.avgInferenceTimeMs = (ctx.avgInferenceTimeMs * ctx.processedFrameCount + inferenceTimeMs) / (ctx.processedFrameCount + 1);
            ctx.processedFrameCount++;
            
            ctx.resultsReady.store(true);
            ctx.inferenceInProgress.store(false);
            
            m_totalInferencesPerSecond++;
        }
        catch (const std::exception& e)
        {
            std::cout << "Error collecting depth results for camera " << cameraIdx << ": " << e.what() << std::endl;
            ctx.inferenceInProgress.store(false);
        }
    }
    else if (eventStatus != cudaErrorNotReady)
    {
        std::cout << "CUDA error in camera " << cameraIdx << " depth inference: " << cudaGetErrorString(eventStatus) << std::endl;
        ctx.inferenceInProgress.store(false);
    }
}

void MultiStreamDepthEstimationApp::prepareDepthInput(uint32_t cameraIdx, dwCameraFrameHandle_t frame, CameraDepthContext& ctx)
{
    dwImageHandle_t img = DW_NULL_HANDLE;
    CHECK_DW_ERROR(dwSensorCamera_getImage(&img, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, frame));
    CHECK_DW_ERROR(dwImage_copyConvert(m_imageRGBA[cameraIdx], img, m_context));

    dwRect roi;
    roi.x = 0;
    roi.y = 0;
    roi.width = m_imageWidth;
    roi.height = m_imageHeight;

    CHECK_DW_ERROR(dwDataConditioner_prepareData(ctx.depthInput, &m_imageRGBA[cameraIdx], 1, 
                                                 &roi,
                                                 cudaAddressModeClamp, ctx.depthDataConditioner));
}


void MultiStreamDepthEstimationApp::doDepthInferenceAsync(uint32_t cameraIdx, CameraDepthContext& ctx)
{
    // Synchronize stream before setting it on DNN
    cudaStreamSynchronize(ctx.cudaStream);
    
    CHECK_DW_ERROR(dwDNN_setCUDAStream(ctx.cudaStream, m_depthDNN));
    
    dwConstDNNTensorHandle_t inputs[1U] = {ctx.depthInput};
    dwDNNTensorHandle_t outputs[1U] = {ctx.depthOutputDevice};
    
    dwStatus status = dwDNN_infer(outputs, 1U, inputs, 1U, m_depthDNN);
    if (status != DW_SUCCESS)
    {
        std::cout << "ERROR: Depth inference failed for camera " << cameraIdx 
                  << ": " << dwGetStatusName(status) << std::endl;
        return;
    }
    
    status = dwDNNTensorStreamer_producerSend(ctx.depthOutputDevice, ctx.depthOutputStreamer);
    if (status != DW_SUCCESS)
    {
        std::cout << "ERROR: Failed to send depth output for camera " << cameraIdx 
                  << ": " << dwGetStatusName(status) << std::endl;
    }
}

void MultiStreamDepthEstimationApp::collectDepthResults(uint32_t cameraIdx, CameraDepthContext& ctx)
{
    dwStatus status = dwDNNTensorStreamer_consumerReceive(&ctx.depthOutputHost, 1000, ctx.depthOutputStreamer);
    if (status != DW_SUCCESS)
    {
        std::cout << "ERROR: Failed to receive depth output for camera " << cameraIdx << std::endl;
        return;
    }
    
    if (ctx.depthOutputHost == DW_NULL_HANDLE)
    {
        std::cout << "ERROR: Received null depth tensor for camera " << cameraIdx << std::endl;
        return;
    }

    {
        std::lock_guard<std::mutex> lock(ctx.depthMutex);
        processDepthOutput(cameraIdx, ctx);
        
        // TEMPORARILY DISABLE VISUALIZATION
        // if (ctx.minDepth < ctx.maxDepth)
        // {
        //     createDepthVisualization(cameraIdx, ctx);
        // }
    }

    CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReturn(&ctx.depthOutputHost, ctx.depthOutputStreamer));
    CHECK_DW_ERROR(dwDNNTensorStreamer_producerReturn(nullptr, 1000, ctx.depthOutputStreamer));
}

void MultiStreamDepthEstimationApp::processDepthOutput(uint32_t cameraIdx, CameraDepthContext& ctx)
{
    // Lock depth tensor and read values
    void* depthData;
    dwStatus status = dwDNNTensor_lock(&depthData, ctx.depthOutputHost);
    if (status != DW_SUCCESS || depthData == nullptr)
    {
        std::cout << "ERROR: Failed to lock depth tensor for camera " << cameraIdx 
                  << ": " << dwGetStatusName(status) << std::endl;
        return;
    }
    
    // Get tensor properties to understand layout
    dwDNNTensorProperties depthProps;
    CHECK_DW_ERROR(dwDNNTensor_getProperties(&depthProps, ctx.depthOutputHost));
    
    // Calculate total elements
    uint32_t totalElements = 1;
    for (uint32_t i = 0; i < depthProps.numDimensions; ++i)
    {
        totalElements *= depthProps.dimensionSize[i];
    }
    
    std::cout << "Camera " << cameraIdx << " depth tensor: "
              << totalElements << " elements, type=" << depthProps.dataType << std::endl;
    
    // Process based on data type
    if (depthProps.dataType == DW_TYPE_FLOAT32)
    {
        float32_t* depthValues = reinterpret_cast<float32_t*>(depthData);
        
        // Find min/max and check for validity
        ctx.minDepth = std::numeric_limits<float32_t>::max();
        ctx.maxDepth = std::numeric_limits<float32_t>::lowest();
        uint32_t validCount = 0;
        uint32_t nanCount = 0;
        uint32_t infCount = 0;
        
        uint32_t pixelsToProcess = std::min(totalElements, static_cast<uint32_t>(ctx.depthMap.size()));
        
        for (uint32_t i = 0; i < pixelsToProcess; ++i)
        {
            float32_t depth = depthValues[i];
            
            // Check for invalid values
            if (std::isnan(depth))
            {
                nanCount++;
                ctx.depthMap[i] = 0.0f;
                continue;
            }
            if (std::isinf(depth))
            {
                infCount++;
                ctx.depthMap[i] = 0.0f;
                continue;
            }
            
            ctx.depthMap[i] = depth;
            ctx.minDepth = std::min(ctx.minDepth, depth);
            ctx.maxDepth = std::max(ctx.maxDepth, depth);
            validCount++;
        }
        
        if (cameraIdx == 0)
        {
            std::cout << "Camera " << cameraIdx << " depth stats:" << std::endl;
            std::cout << "  Valid: " << validCount << ", NaN: " << nanCount << ", Inf: " << infCount << std::endl;
            std::cout << "  First 10 values: ";
            for (uint32_t i = 0; i < std::min(10u, pixelsToProcess); ++i)
            {
                std::cout << depthValues[i] << " ";
            }
            std::cout << std::endl;
        }
        
        // If no valid values, set defaults
        if (validCount == 0)
        {
            std::cout << "WARNING: No valid depth values for camera " << cameraIdx << std::endl;
            ctx.minDepth = 0.0f;
            ctx.maxDepth = 1.0f;
        }
    }
    else if (depthProps.dataType == DW_TYPE_FLOAT16)
    {
        std::cout << "ERROR: FP16 depth output not yet supported" << std::endl;
        ctx.minDepth = 0.0f;
        ctx.maxDepth = 1.0f;
    }
    
    CHECK_DW_ERROR(dwDNNTensor_unlock(ctx.depthOutputHost));
}

void MultiStreamDepthEstimationApp::createDepthVisualization(uint32_t cameraIdx, CameraDepthContext& ctx)
{
    // Get CUDA image properties
    dwImageCUDA* depthVizCuda;
    dwStatus status = dwImage_getCUDA(&depthVizCuda, ctx.depthVisualizationImage);
    if (status != DW_SUCCESS || depthVizCuda == nullptr)
    {
        std::cout << "ERROR: Failed to get CUDA image for camera " << cameraIdx << std::endl;
        return;
    }
    
    // Check if we have valid depth data
    if (ctx.minDepth >= ctx.maxDepth)
    {
        std::cout << "WARNING: Invalid depth range for camera " << cameraIdx << std::endl;
        return;
    }
    
    // Allocate host memory for visualization
    std::vector<uint8_t> hostPixels(DEPTH_MODEL_WIDTH * DEPTH_MODEL_HEIGHT * 4);
    
    // Create visualization on CPU
    for (uint32_t y = 0; y < DEPTH_MODEL_HEIGHT; ++y)
    {
        for (uint32_t x = 0; x < DEPTH_MODEL_WIDTH; ++x)
        {
            uint32_t depthIdx = y * DEPTH_MODEL_WIDTH + x;
            float32_t depth = ctx.depthMap[depthIdx];
            
            // Normalize depth to [0, 1]
            float32_t normalizedDepth = (depth - ctx.minDepth) / (ctx.maxDepth - ctx.minDepth + 1e-6f);
            normalizedDepth = std::max(0.0f, std::min(1.0f, normalizedDepth));
            
            // Invert so closer = brighter
            normalizedDepth = 1.0f - normalizedDepth;
            
            dwVector4f color = depthToColor(normalizedDepth);
            
            uint32_t pixelIdx = (y * DEPTH_MODEL_WIDTH + x) * 4;
            hostPixels[pixelIdx + 0] = static_cast<uint8_t>(color.x * 255);  // R
            hostPixels[pixelIdx + 1] = static_cast<uint8_t>(color.y * 255);  // G
            hostPixels[pixelIdx + 2] = static_cast<uint8_t>(color.z * 255);  // B
            hostPixels[pixelIdx + 3] = 255;  // A
        }
    }
    
    // Copy from host to CUDA device
    cudaError_t cudaStatus = cudaMemcpy2DAsync(
        depthVizCuda->dptr[0],
        depthVizCuda->pitch[0],
        hostPixels.data(),
        DEPTH_MODEL_WIDTH * 4,
        DEPTH_MODEL_WIDTH * 4,
        DEPTH_MODEL_HEIGHT,
        cudaMemcpyHostToDevice,
        ctx.cudaStream
    );
    
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "ERROR: Failed to copy depth visualization to CUDA: " 
                  << cudaGetErrorString(cudaStatus) << std::endl;
    }
}

dwVector4f MultiStreamDepthEstimationApp::depthToColor(float normalizedDepth)
{
    dwVector4f color;
    
    switch (m_depthVizMode)
    {
    case DEPTH_VIZ_GRAYSCALE:
        color.x = normalizedDepth;
        color.y = normalizedDepth;
        color.z = normalizedDepth;
        color.w = 1.0f;
        break;
        
    case DEPTH_VIZ_HEATMAP:
        // Jet colormap: blue (far) -> cyan -> green -> yellow -> red (near)
        if (normalizedDepth < 0.25f)
        {
            color.x = 0.0f;
            color.y = normalizedDepth * 4.0f;
            color.z = 1.0f;
        }
        else if (normalizedDepth < 0.5f)
        {
            color.x = 0.0f;
            color.y = 1.0f;
            color.z = 1.0f - (normalizedDepth - 0.25f) * 4.0f;
        }
        else if (normalizedDepth < 0.75f)
        {
            color.x = (normalizedDepth - 0.5f) * 4.0f;
            color.y = 1.0f;
            color.z = 0.0f;
        }
        else
        {
            color.x = 1.0f;
            color.y = 1.0f - (normalizedDepth - 0.75f) * 4.0f;
            color.z = 0.0f;
        }
        color.w = 1.0f;
        break;
        
    case DEPTH_VIZ_OVERLAY:
        // Semi-transparent heatmap for overlay
        color = depthToColor(normalizedDepth); // Recursive call with HEATMAP
        color.w = 0.6f; // transparency
        break;
    }
    
    return color;
}

void MultiStreamDepthEstimationApp::onRenderHelper(dwCameraFrameHandle_t frame, uint8_t cameraIndex)
{
    dwImageHandle_t img = DW_NULL_HANDLE;

    CHECK_DW_ERROR(dwRenderEngine_setTile(m_tileVideo[cameraIndex], m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_resetTile(m_renderEngine));

    CHECK_DW_ERROR(dwSensorCamera_getImage(&img, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, frame));

    // Stream camera image to GL domain
    CHECK_DW_ERROR(dwImageStreamerGL_producerSend(img, m_streamerToGL[cameraIndex]));
    dwImageHandle_t frameGL;
    CHECK_DW_ERROR(dwImageStreamerGL_consumerReceive(&frameGL, 33000, m_streamerToGL[cameraIndex]));

    dwImageGL* imageGL;
    CHECK_DW_ERROR(dwImage_getGL(&imageGL, frameGL));

    // Render camera image
    dwVector2f range{};
    range.x = imageGL->prop.width;
    range.y = imageGL->prop.height;
    CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_renderImage2D(imageGL, {0, 0, range.x, range.y}, m_renderEngine));

    CHECK_DW_ERROR(dwImageStreamerGL_consumerReturn(&frameGL, m_streamerToGL[cameraIndex]));
    CHECK_DW_ERROR(dwImageStreamerGL_producerReturn(nullptr, 32000, m_streamerToGL[cameraIndex]));

    // Render depth map visualization (overlay or side-by-side)
    visualizeDepthMap(cameraIndex);
    
    // Render performance stats
    renderPerformanceStats(cameraIndex);
}

void MultiStreamDepthEstimationApp::visualizeDepthMap(uint32_t cameraIdx)
{
    return ; // TEMPORARILY DISABLE VISUALIZATION
    CameraDepthContext& ctx = m_cameraContexts[cameraIdx];
    
    if (!ctx.resultsReady.load())
        return;
    
    std::lock_guard<std::mutex> lock(ctx.depthMutex);
    
    // Stream depth visualization to GL
    CHECK_DW_ERROR(dwImageStreamerGL_producerSend(ctx.depthVisualizationImage, m_depthStreamerToGL[cameraIdx]));
    
    dwImageHandle_t depthGL;
    CHECK_DW_ERROR(dwImageStreamerGL_consumerReceive(&depthGL, 33000, m_depthStreamerToGL[cameraIdx]));
    
    dwImageGL* depthImageGL;
    CHECK_DW_ERROR(dwImage_getGL(&depthImageGL, depthGL));
    
    // Render depth map in bottom-right corner (picture-in-picture style)
    dwVector2f range{};
    range.x = m_imageWidth;
    range.y = m_imageHeight;
    
    float32_t depthWidth = range.x * 0.3f;   // 30% of camera image width
    float32_t depthHeight = range.y * 0.3f;  // 30% of camera image height
    float32_t depthX = range.x - depthWidth - 10;  // 10px margin
    float32_t depthY = range.y - depthHeight - 10;
    
    dwRectf depthRect = {depthX, depthY, depthWidth, depthHeight};
    
    CHECK_DW_ERROR(dwRenderEngine_renderImage2D(depthImageGL, depthRect, m_renderEngine));
    
    // Draw border around depth map
    CHECK_DW_ERROR(dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_setLineWidth(2.0f, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D, &depthRect,
                                         sizeof(dwRectf), 0, 1, m_renderEngine));
    
    CHECK_DW_ERROR(dwImageStreamerGL_consumerReturn(&depthGL, m_depthStreamerToGL[cameraIdx]));
    CHECK_DW_ERROR(dwImageStreamerGL_producerReturn(nullptr, 32000, m_depthStreamerToGL[cameraIdx]));
}

void MultiStreamDepthEstimationApp::renderPerformanceStats(uint32_t cameraIdx)
{
    CameraDepthContext& ctx = m_cameraContexts[cameraIdx];
    
    std::string statusStr = "Cam:" + std::to_string(cameraIdx);
    statusStr += " Depth Inf:" + std::to_string(static_cast<int>(ctx.avgInferenceTimeMs)) + "ms";
    statusStr += " Range:[" + std::to_string(ctx.minDepth).substr(0, 4) + "," + std::to_string(ctx.maxDepth).substr(0, 4) + "]";
    statusStr += " Frm:" + std::to_string(ctx.frameId.load());
    
    if (ctx.inferenceInProgress.load())
    {
        statusStr += " [DEPTH PROCESSING]";
        CHECK_DW_ERROR(dwRenderEngine_setColor({1.0f, 1.0f, 0.0f, 1.0f}, m_renderEngine));
    }
    else if (ctx.resultsReady.load())
    {
        statusStr += " [READY]";
        CHECK_DW_ERROR(dwRenderEngine_setColor({0.0f, 1.0f, 0.0f, 1.0f}, m_renderEngine));
    }
    else
    {
        statusStr += " [IDLE]";
        CHECK_DW_ERROR(dwRenderEngine_setColor({0.7f, 0.7f, 0.7f, 1.0f}, m_renderEngine));
    }

    CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_renderText2D(statusStr.c_str(), {25, 60}, m_renderEngine));
}

void MultiStreamDepthEstimationApp::updatePerformanceStats()
{
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto timeSinceLastUpdate = std::chrono::duration_cast<std::chrono::seconds>(currentTime - m_lastStatsTime);
    
    if (timeSinceLastUpdate.count() >= 2)
    {
        printPerformanceMetrics();
        m_lastStatsTime = currentTime;
        m_totalInferencesPerSecond = 0;
    }
}

void MultiStreamDepthEstimationApp::printPerformanceMetrics()
{
    std::cout << "=== Depth Estimation Performance ===" << std::endl;
    std::cout << "Total Frame Count: " << m_frameCount.load() << std::endl;
    std::cout << "Depth Inferences/sec (last 2s): " << m_totalInferencesPerSecond / 2 << std::endl;
    
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        CameraDepthContext& ctx = m_cameraContexts[i];
        std::cout << "Camera " << i << ": "
                  << "Avg Depth Inference: " << ctx.avgInferenceTimeMs << "ms, "
                  << "Processed: " << ctx.processedFrameCount << " frames, "
                  << "Depth Range: [" << ctx.minDepth << ", " << ctx.maxDepth << "], "
                  << "Status: " << (ctx.inferenceInProgress.load() ? "PROCESSING" : "IDLE")
                  << std::endl;
    }
}

void MultiStreamDepthEstimationApp::onRelease()
{
    std::cout << "Releasing multi-stream depth estimation application..." << std::endl;
    
    // Stop all camera sensors
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        if (m_camera[i])
        {
            CHECK_DW_ERROR(dwSensor_stop(m_camera[i]));
        }
    }

    // Wait for all in-progress inferences to complete
    bool allComplete = false;
    int maxWaitIterations = 100;
    int waitCount = 0;
    
    while (!allComplete && waitCount < maxWaitIterations)
    {
        allComplete = true;
        for (uint32_t i = 0; i < m_totalCameras; ++i)
        {
            if (m_cameraContexts[i].inferenceInProgress.load())
            {
                allComplete = false;
                break;
            }
        }
        if (!allComplete)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            waitCount++;
        }
    }

    // Release per-camera depth resources
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        CameraDepthContext& ctx = m_cameraContexts[i];
        
        // Destroy CUDA events
        if (ctx.frameReadyEvent)
            cudaEventDestroy(ctx.frameReadyEvent);
        if (ctx.inferenceCompleteEvent)
            cudaEventDestroy(ctx.inferenceCompleteEvent);
        
        // Destroy CUDA stream
        if (ctx.cudaStream)
            cudaStreamDestroy(ctx.cudaStream);
        
        // Release depth DNN tensors
        if (ctx.depthInput)
            CHECK_DW_ERROR(dwDNNTensor_destroy(ctx.depthInput));
        if (ctx.depthOutputDevice)
            CHECK_DW_ERROR(dwDNNTensor_destroy(ctx.depthOutputDevice));
        if (ctx.depthOutputStreamer)
            CHECK_DW_ERROR(dwDNNTensorStreamer_release(ctx.depthOutputStreamer));
        
        // Release data conditioner
        if (ctx.depthDataConditioner)
            CHECK_DW_ERROR(dwDataConditioner_release(ctx.depthDataConditioner));
        
        // Release visualization image
        if (ctx.depthVisualizationImage)
            CHECK_DW_ERROR(dwImage_destroy(ctx.depthVisualizationImage));
        
        // Release image and streaming resources
        if (m_streamerToGL[i])
            dwImageStreamerGL_release(m_streamerToGL[i]);
        if (m_depthStreamerToGL[i])
            dwImageStreamerGL_release(m_depthStreamerToGL[i]);
        if (m_camera[i])
            dwSAL_releaseSensor(m_camera[i]);
        if (m_imageRGBA[i])
            CHECK_DW_ERROR(dwImage_destroy(m_imageRGBA[i]));
    }

    // Release shared depth DNN model
    if (m_depthDNN)
        CHECK_DW_ERROR(dwDNN_release(m_depthDNN));

    // Release framework resources
    m_screenshot.reset();
    if (m_rigConfig)
        dwRig_release(m_rigConfig);
    if (m_cameraMaster)
        dwSAL_releaseSensor(m_cameraMaster);
    if (m_sal)
        dwSAL_release(m_sal);
    if (m_renderEngine)
        dwRenderEngine_release(m_renderEngine);
    if (m_renderer)
        dwRenderer_release(m_renderer);
    if (m_viz)
        dwVisualizationRelease(m_viz);
    if (m_context)
        dwRelease(m_context);
        
    std::cout << "Multi-stream depth estimation application released successfully" << std::endl;
}

void MultiStreamDepthEstimationApp::onKeyDown(int key, int scancode, int mods)
{
    (void)scancode;
    (void)mods;

    if (key == GLFW_KEY_S)
    {
        m_screenshot->triggerScreenshot();
    }
    else if (key == GLFW_KEY_P)
    {
        printPerformanceMetrics();
    }
    else if (key == GLFW_KEY_V)
    {
        // Cycle through visualization modes
        m_depthVizMode = static_cast<DepthVisualizationMode>((m_depthVizMode + 1) % 3);
        std::cout << "Depth visualization mode: " 
                  << (m_depthVizMode == DEPTH_VIZ_GRAYSCALE ? "GRAYSCALE" :
                      m_depthVizMode == DEPTH_VIZ_HEATMAP ? "HEATMAP" : "OVERLAY")
                  << std::endl;
    }
}

std::string MultiStreamDepthEstimationApp::getPlatformPrefix()
{
    static const int32_t CUDA_AMPERE_MAJOR_COMPUTE_CAPABILITY = 8;
    static const int32_t CUDA_TURING_VOLTA_MAJOR_COMPUTE_CAPABILITY = 7;
    static const int32_t CUDA_VOLTA_DISCRETE_MINOR_COMPUTE_CAPABILITY = 0;
    static const int32_t CUDA_VOLTA_INTEGRATED_MINOR_COMPUTE_CAPABILITY = 2;
    static const int32_t CUDA_TURING_DISCRETE_MINOR_COMPUTE_CAPABILITY = 5;

    std::string path;
    int32_t currentGPU;
    dwGPUDeviceProperties gpuProp{};

    CHECK_DW_ERROR(dwContext_getGPUDeviceCurrent(&currentGPU, m_context));
    CHECK_DW_ERROR(dwContext_getGPUProperties(&gpuProp, currentGPU, m_context));

    if (gpuProp.major == CUDA_AMPERE_MAJOR_COMPUTE_CAPABILITY)
    {
        path = gpuProp.integrated ? "ampere-integrated" : "ampere-discrete";
    }
    else if (gpuProp.major == CUDA_TURING_VOLTA_MAJOR_COMPUTE_CAPABILITY)
    {
        if (gpuProp.minor == CUDA_TURING_DISCRETE_MINOR_COMPUTE_CAPABILITY)
            path = "turing";
        else if (gpuProp.minor == CUDA_VOLTA_INTEGRATED_MINOR_COMPUTE_CAPABILITY)
            path = "volta-integrated";
        else if (gpuProp.minor == CUDA_VOLTA_DISCRETE_MINOR_COMPUTE_CAPABILITY)
            path = "volta-discrete";
    }
    else
    {
        path = "pascal";
    }

    return path;
}

//#######################################################################################
int main(int argc, const char** argv)
{
    ProgramArguments args(argc, argv,
                          {
                              ProgramArguments::Option_t("rig", (dw_samples::SamplesDataPath::get() + "/samples/sensors/camera/camera/rig_4cam.json").c_str(), "Rig configuration for 4 cameras"),
                              ProgramArguments::Option_t("depth_model", "", (std::string("path to depth TensorRT model file. By default: ") + dw_samples::SamplesDataPath::get() + "/samples/detector/<gpu-architecture>/depth_anything_v2_fp32.bin").c_str())},
                          "Multi-Stream 4-Camera Depth Estimation with DepthAnythingV2.");

    MultiStreamDepthEstimationApp app(args);
    app.initializeWindow("Multi-Stream Depth Estimation", 1280, 800, args.enabled("offscreen"));
    
    if (!args.enabled("offscreen"))
        app.setProcessRate(30);

    return app.run();
}