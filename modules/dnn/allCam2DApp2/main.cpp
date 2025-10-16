/////////////////////////////////////////////////////////////////////////////////////////
// Multi-Stream Camera DNN Application - Strategy 1 Implementation
// Per-camera CUDA streams with dedicated tensor resources
/////////////////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <set>
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
#define FIRST_CAMERA_IDX 0

// Per-Camera Processing Context
struct CameraProcessingContext {
    // CUDA stream for this camera
    cudaStream_t cudaStream;
    
    // DNN resources per camera
    dwDNNTensorHandle_t dnnInput;
    dwDNNTensorHandle_t dnnOutputsDevice[2]; // coverage + bbox
    dwDataConditionerHandle_t dataConditioner;
    
    // Tensor streamers per camera  
    dwDNNTensorStreamerHandle_t dnnOutputStreamers[2];
    std::unique_ptr<dwDNNTensorHandle_t[]> outputsHost;
    
    // Processing state
    std::atomic<bool> inferenceInProgress;
    std::atomic<uint64_t> frameId;
    std::atomic<bool> resultsReady;
    
    // CUDA synchronization
    cudaEvent_t frameReadyEvent;
    cudaEvent_t inferenceCompleteEvent;
    
    // Detection results (thread-safe access needed)
    std::vector<dwBox2D> detectedBoxList;
    std::vector<dwRectf> detectedBoxListFloat;  
    std::vector<std::string> labels;
    mutable std::mutex resultsMutex;
    
    // Performance metrics
    std::chrono::high_resolution_clock::time_point lastProcessTime;
    float avgInferenceTimeMs;
    uint32_t processedFrameCount;
    
    CameraProcessingContext() : 
        cudaStream(0),
        dnnInput(DW_NULL_HANDLE),
        dataConditioner(DW_NULL_HANDLE),
        inferenceInProgress(false),
        frameId(0),
        resultsReady(false),
        frameReadyEvent(nullptr),
        inferenceCompleteEvent(nullptr),
        avgInferenceTimeMs(0.0f),
        processedFrameCount(0)
    {
        dnnOutputsDevice[0] = DW_NULL_HANDLE;
        dnnOutputsDevice[1] = DW_NULL_HANDLE;
        dnnOutputStreamers[0] = DW_NULL_HANDLE;
        dnnOutputStreamers[1] = DW_NULL_HANDLE;
        
        detectedBoxList.reserve(1000);
        detectedBoxListFloat.reserve(1000);
        labels.reserve(1000);
    }
};

class MultiStreamCameraDNNApp : public DriveWorksSample
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
    dwRenderEngineColorRGBA m_colorPerPort[MAX_PORTS_COUNT];

    dwSALHandle_t m_sal             = DW_NULL_HANDLE;
    dwSensorHandle_t m_cameraMaster = DW_NULL_HANDLE;
    uint32_t m_totalCameras;

    dwRigHandle_t m_rigConfig{};

    // Per camera resources
    dwSensorHandle_t m_camera[MAX_CAMS];
    dwImageStreamerHandle_t m_streamerToGL[MAX_CAMS] = {DW_NULL_HANDLE};
    uint32_t m_tileVideo[MAX_CAMS];
    bool m_enableRender[MAX_CAMS] = {true};

    bool m_useRaw        = false;
    bool m_useProcessed  = false;
    bool m_useProcessed1 = false;
    bool m_useProcessed2 = false;
    uint32_t m_fifoSize  = 0U;
    std::atomic<uint32_t> m_frameCount{0};

    // ============================================
    // MULTI-STREAM DNN SECTION
    // ============================================
    static constexpr float32_t COVERAGE_THRESHOLD = 0.6f;
    static constexpr float32_t CONFIDENCE_THRESHOLD = 0.45f;
    static constexpr float32_t SCORE_THRESHOLD = 0.25f;
    static constexpr uint32_t NUM_OUTPUT_TENSORS = 2U;
    
    const uint32_t m_maxDetections = 1000U;
    const float32_t m_nonMaxSuppressionOverlapThreshold = 0.5;
    uint32_t m_numOutputTensors = 1U;

    // Shared DNN model (read-only after initialization)
    dwDNNHandle_t m_dnn = DW_NULL_HANDLE;
    
    // Per-camera processing contexts
    std::unique_ptr<CameraProcessingContext[]> m_cameraContexts;
    
    // DNN model parameters
    uint32_t m_cellSize = 1U;
    uint32_t m_cvgIdx;
    uint32_t m_bboxIdx;
    dwRect m_detectionRegion[MAX_CAMS];

    bool m_usecuDLA = false;
    uint32_t m_dlaEngineNo = 0;

    // Shared image resources
    dwImageHandle_t m_imageRGBA[MAX_CAMS];
    uint32_t m_imageWidth;
    uint32_t m_imageHeight;

    // YOLO class definitions
    const std::string YOLO_CLASS_NAMES[80] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
        "toothbrush"
    };
    
    std::set<std::string> m_automotiveClasses = {
        "person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "stop sign"
    };

    typedef struct YoloScoreRect {
        dwRectf rectf;
        float32_t score;
        uint16_t classIndex;
    } YoloScoreRect;

    std::unique_ptr<ScreenshotHelper> m_screenshot;
    
    // Performance monitoring
    std::chrono::high_resolution_clock::time_point m_lastStatsTime;
    uint32_t m_totalInferencesPerSecond = 0;

public:
    MultiStreamCameraDNNApp(const ProgramArguments& args);

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
    void initializeMultiStreamDNN();
    void initializeCameraContext(uint32_t cameraIdx);
    
    // Multi-stream processing methods  
    void processAllCamerasParallel(dwCameraFrameHandle_t frames[]);
    bool startInferenceAsync(uint32_t cameraIdx, dwCameraFrameHandle_t frame);
    void checkAndCollectResults(uint32_t cameraIdx);
    
    // DNN processing methods
    void prepareInputFrame(uint32_t cameraIdx, dwCameraFrameHandle_t frame, CameraProcessingContext& ctx);
    void doInferenceAsync(uint32_t cameraIdx, CameraProcessingContext& ctx);
    void collectInferenceResults(uint32_t cameraIdx, CameraProcessingContext& ctx);
    void interpretOutput(dwDNNTensorHandle_t outConf, dwDNNTensorHandle_t outBBox, 
                        const dwRect* const roi, uint32_t cameraIdx, CameraProcessingContext& ctx);
    void interpretOutput(dwDNNTensorHandle_t outConf, const dwRect* const roi, 
                        uint32_t cameraIdx, CameraProcessingContext& ctx);
    
    // Rendering methods
    void onRenderHelper(dwCameraFrameHandle_t frame, uint8_t cameraIndex);
    void renderDetectionResults(uint32_t cameraIdx);
    void renderPerformanceStats(uint32_t cameraIdx);
    
    // Utility methods
    std::string getPlatformPrefix();
    float32_t calculateIouOfBoxes(dwRectf box1, dwRectf box2);
    std::vector<YoloScoreRect> doNmsForYoloOutputBoxes(std::vector<YoloScoreRect>& boxes, float32_t threshold);
    static bool sort_score(YoloScoreRect box1, YoloScoreRect box2);
    
    // Performance monitoring
    void updatePerformanceStats();
    void printPerformanceMetrics();
};

//#######################################################################################
MultiStreamCameraDNNApp::MultiStreamCameraDNNApp(const ProgramArguments& args)
    : DriveWorksSample(args)
{
    m_cameraContexts.reset(new CameraProcessingContext[MAX_CAMS]);
    m_lastStatsTime = std::chrono::high_resolution_clock::now();
}

void MultiStreamCameraDNNApp::initializeDriveWorks(dwContextHandle_t& context) const
{
    CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
    CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));

    dwContextParameters sdkParams = {};
#ifdef VIBRANTE
    sdkParams.eglDisplay = getEGLDisplay();
#endif
    CHECK_DW_ERROR(dwInitialize(&context, DW_VERSION, &sdkParams));
}

void MultiStreamCameraDNNApp::initializeMultiCamera()
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

        // Parse output format parameters
        m_useRaw        = std::string::npos != std::string(params).find("raw");
        m_useProcessed  = std::string::npos != std::string(params).find("processed");
        m_useProcessed1 = std::string::npos != std::string(params).find("processed1");
        m_useProcessed2 = std::string::npos != std::string(params).find("processed2");

        if (!m_useProcessed && !m_useRaw)
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

    std::cout << "Detected image resolution: " << m_imageWidth << "x" << m_imageHeight << std::endl;

    // Create image and streaming resources for each camera
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        CHECK_DW_ERROR(dwImage_create(&m_imageRGBA[i], imageProperties, m_context));
        CHECK_DW_ERROR(dwImageStreamerGL_initialize(&m_streamerToGL[i], &imageProperties, DW_IMAGE_GL, m_context));
    }
}

void MultiStreamCameraDNNApp::initializeMultiStreamDNN()
{
#ifdef VIBRANTE
    m_usecuDLA    = getArgument("cudla").compare("1") == 0;
    m_dlaEngineNo = std::atoi(getArgument("dla-engine").c_str());
#endif

    std::string tensorRTModel = getArgument("tensorRT_model");
    if (tensorRTModel.empty())
    {
        tensorRTModel = dw_samples::SamplesDataPath::get() + "/samples/detector/";
        tensorRTModel += getPlatformPrefix();
        tensorRTModel += "/tensorRT_model";
        if (m_usecuDLA)
            tensorRTModel += ".dla";
        tensorRTModel += ".bin";
    }

    std::cout << "Loading DNN model: " << tensorRTModel << std::endl;
    
    // Initialize shared DNN model (read-only after this point)
    CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFileWithEngineId(&m_dnn, tensorRTModel.c_str(), nullptr,
                                                               m_usecuDLA ? DW_PROCESSOR_TYPE_CUDLA : DW_PROCESSOR_TYPE_GPU,
                                                               m_dlaEngineNo, m_context));

    m_numOutputTensors = m_usecuDLA ? NUM_OUTPUT_TENSORS : 1U;

    // Get DNN metadata and tensor properties (shared across all cameras)
    dwDNNTensorProperties inputProps;
    dwDNNTensorProperties outputProps[NUM_OUTPUT_TENSORS];

    CHECK_DW_ERROR(dwDNN_getInputTensorProperties(&inputProps, 0U, m_dnn));
    for (uint32_t outputIdx = 0U; outputIdx < m_numOutputTensors; ++outputIdx)
    {
        CHECK_DW_ERROR(dwDNN_getOutputTensorProperties(&outputProps[outputIdx], outputIdx, m_dnn));
    }

    // Get output blob indices
    if (m_usecuDLA)
    {
        const char* coverageBlobName    = "coverage";
        const char* boundingBoxBlobName = "bboxes";
        CHECK_DW_ERROR(dwDNN_getOutputIndex(&m_cvgIdx, coverageBlobName, m_dnn));
        CHECK_DW_ERROR(dwDNN_getOutputIndex(&m_bboxIdx, boundingBoxBlobName, m_dnn));
    }
    else
    {
        const char* coverageBlobName = "output0";
        CHECK_DW_ERROR(dwDNN_getOutputIndex(&m_cvgIdx, coverageBlobName, m_dnn));
    }

    dwDNNMetaData metadata;
    CHECK_DW_ERROR(dwDNN_getMetaData(&metadata, m_dnn));

    uint32_t gridW = outputProps[0].dimensionSize[0];
    m_cellSize = inputProps.dimensionSize[0] / gridW;

    std::cout << "DNN Model loaded successfully" << std::endl;
    std::cout << "Input dimensions: " << inputProps.dimensionSize[0] << "x" << inputProps.dimensionSize[1] << std::endl;
    std::cout << "Cell size: " << m_cellSize << std::endl;
    std::cout << "Using " << m_numOutputTensors << " output tensors" << std::endl;

    // Initialize per-camera processing contexts
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        initializeCameraContext(i);
        
        // Setup detection regions
        m_detectionRegion[i].width = m_imageWidth;
        m_detectionRegion[i].height = m_imageHeight;
        m_detectionRegion[i].x = 0;
        m_detectionRegion[i].y = 0;
    }

    std::cout << "Multi-stream DNN initialization complete for " << m_totalCameras << " cameras" << std::endl;
}

void MultiStreamCameraDNNApp::initializeCameraContext(uint32_t cameraIdx)
{
    CameraProcessingContext& ctx = m_cameraContexts[cameraIdx];
    
    std::cout << "Initializing camera context " << cameraIdx << std::endl;
    
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
    dwDNNTensorProperties inputProps;
    dwDNNTensorProperties outputProps[NUM_OUTPUT_TENSORS];
    
    CHECK_DW_ERROR(dwDNN_getInputTensorProperties(&inputProps, 0U, m_dnn));
    for (uint32_t outputIdx = 0U; outputIdx < m_numOutputTensors; ++outputIdx)
    {
        CHECK_DW_ERROR(dwDNN_getOutputTensorProperties(&outputProps[outputIdx], outputIdx, m_dnn));
    }
    
    // Create per-camera DNN tensors
    CHECK_DW_ERROR(dwDNNTensor_create(&ctx.dnnInput, &inputProps, m_context));
    
    for (uint32_t outputIdx = 0U; outputIdx < m_numOutputTensors; ++outputIdx)
    {
        CHECK_DW_ERROR(dwDNNTensor_create(&ctx.dnnOutputsDevice[outputIdx], &outputProps[outputIdx], m_context));
        
        // Create tensor streamers for GPU->CPU transfer
        dwDNNTensorProperties hostProps = outputProps[outputIdx];
        hostProps.tensorType = DW_DNN_TENSOR_TYPE_CPU;
        CHECK_DW_ERROR(dwDNNTensorStreamer_initialize(&ctx.dnnOutputStreamers[outputIdx],
                                                      &outputProps[outputIdx],
                                                      hostProps.tensorType, m_context));
    }
    
    // Initialize host output tensor array
    ctx.outputsHost.reset(new dwDNNTensorHandle_t[m_numOutputTensors]);
    
    // Create per-camera data conditioner
    dwDNNMetaData metadata;
    CHECK_DW_ERROR(dwDNN_getMetaData(&metadata, m_dnn));
    CHECK_DW_ERROR(dwDataConditioner_initializeFromTensorProperties(&ctx.dataConditioner, &inputProps, 1U,
                                                                   &metadata.dataConditionerParams, ctx.cudaStream,
                                                                   m_context));
    
    std::cout << "Camera context " << cameraIdx << " initialized successfully" << std::endl;
}

void MultiStreamCameraDNNApp::initializeRenderer()
{
    CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_context));
    m_colorPerPort[0] = {1, 0, 0, 1};
    m_colorPerPort[1] = {0, 1, 0, 1};
    m_colorPerPort[2] = {0, 0, 1, 1};
    m_colorPerPort[3] = {0, 0, 0, 1};

    std::cout << "Total cameras for rendering: " << m_totalCameras << std::endl;

    CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&params, getWindowWidth(), getWindowHeight()));
    params.defaultTile.lineWidth = 2.0f;
    params.defaultTile.font      = DW_RENDER_ENGINE_FONT_VERDANA_24;
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
        tilesPerRow = 2;
        break;
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
        paramList[i].font            = DW_RENDER_ENGINE_FONT_VERDANA_24;
    }

    CHECK_DW_ERROR(dwRenderEngine_addTilesByCount(m_tileVideo, m_totalCameras, tilesPerRow, paramList, m_renderEngine));
}

bool MultiStreamCameraDNNApp::onInitialize()
{
    initializeDriveWorks(m_context);
    CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_context));

    initializeMultiCamera();
    CHECK_DW_ERROR(dwSAL_start(m_sal));
    initializeRenderer();
    initializeMultiStreamDNN();

    m_screenshot.reset(new ScreenshotHelper(m_context, m_sal, getWindowWidth(), getWindowHeight(), "MultiStreamCameraDNN"));

    std::cout << "Starting all camera sensors..." << std::endl;
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        CHECK_DW_ERROR(dwSensor_start(m_camera[i]));
    }

    std::cout << "Multi-stream camera DNN application initialized successfully" << std::endl;
    return true;
}

void MultiStreamCameraDNNApp::onRender()
{
    // Capture frames from all cameras
    dwCameraFrameHandle_t frames[MAX_CAMS];
    
    // Read all camera frames in parallel (non-blocking where possible)
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        dwStatus status = DW_NOT_READY;
        uint32_t countFailure = 0;
        
        while ((status == DW_NOT_READY) || (status == DW_END_OF_STREAM))
        {
            status = dwSensorCamera_readFrame(&frames[i], 33333, m_camera[i]); // ~30fps timeout
            countFailure++;
            if (countFailure == 30000) // Reduced timeout for responsiveness
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

        if (status == DW_TIME_OUT)
        {
            std::cout << "Camera " << i << " frame timeout" << std::endl;
            continue;
        }

        if (status != DW_SUCCESS)
        {
            std::cout << "Camera " << i << " frame read error: " << dwGetStatusName(status) << std::endl;
            continue;
        }
    }

    // CRITICAL: Process all cameras in parallel instead of round-robin
    processAllCamerasParallel(frames);

    // Render all cameras with their current detection results
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
    
    // Update performance statistics periodically
    updatePerformanceStats();
}

void MultiStreamCameraDNNApp::processAllCamerasParallel(dwCameraFrameHandle_t frames[])
{
    // Step 1: Start asynchronous inference on all cameras that aren't currently processing
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        CameraProcessingContext& ctx = m_cameraContexts[i];
        
        // Check if this camera is ready for new inference
        if (!ctx.inferenceInProgress.load())
        {
            if (startInferenceAsync(i, frames[i]))
            {
                ctx.frameId.store(m_frameCount.load());
                ctx.inferenceInProgress.store(true);
            }
        }
    }
    
    // Step 2: Check for completed inferences and collect results
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        checkAndCollectResults(i);
    }
}

bool MultiStreamCameraDNNApp::startInferenceAsync(uint32_t cameraIdx, dwCameraFrameHandle_t frame)
{
    CameraProcessingContext& ctx = m_cameraContexts[cameraIdx];
    cudaError_t cudaStatus;
    
    try
    {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Prepare input frame on camera's dedicated CUDA stream
        prepareInputFrame(cameraIdx, frame, ctx);
        
        // Record event to mark frame preparation complete
        cudaStatus = cudaEventRecord(ctx.frameReadyEvent, ctx.cudaStream);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("Failed to record frame ready event for camera " + std::to_string(cameraIdx));
        }
        
        // Start asynchronous inference
        doInferenceAsync(cameraIdx, ctx);
        
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
        std::cout << "Error starting inference for camera " << cameraIdx << ": " << e.what() << std::endl;
        ctx.inferenceInProgress.store(false);
        return false;
    }
}

void MultiStreamCameraDNNApp::checkAndCollectResults(uint32_t cameraIdx)
{
    CameraProcessingContext& ctx = m_cameraContexts[cameraIdx];
    
    if (!ctx.inferenceInProgress.load())
        return;
    
    // Check if inference is complete (non-blocking)
    cudaError_t eventStatus = cudaEventQuery(ctx.inferenceCompleteEvent);
    
    if (eventStatus == cudaSuccess)
    {
        // Inference is complete - collect results
        try
        {
            collectInferenceResults(cameraIdx, ctx);
            
            // Update performance metrics
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - ctx.lastProcessTime);
            float inferenceTimeMs = duration.count() / 1000.0f;
            
            // Update rolling average
            ctx.avgInferenceTimeMs = (ctx.avgInferenceTimeMs * ctx.processedFrameCount + inferenceTimeMs) / (ctx.processedFrameCount + 1);
            ctx.processedFrameCount++;
            
            ctx.resultsReady.store(true);
            ctx.inferenceInProgress.store(false);
            
            m_totalInferencesPerSecond++;
        }
        catch (const std::exception& e)
        {
            std::cout << "Error collecting results for camera " << cameraIdx << ": " << e.what() << std::endl;
            ctx.inferenceInProgress.store(false);
        }
    }
    else if (eventStatus != cudaErrorNotReady)
    {
        // Actual error occurred
        std::cout << "CUDA error in camera " << cameraIdx << " inference: " << cudaGetErrorString(eventStatus) << std::endl;
        ctx.inferenceInProgress.store(false);
    }
    // else: inference still in progress, check next frame
}

void MultiStreamCameraDNNApp::prepareInputFrame(uint32_t cameraIdx, dwCameraFrameHandle_t frame, CameraProcessingContext& ctx)
{
    dwImageHandle_t img = DW_NULL_HANDLE;
    CHECK_DW_ERROR(dwSensorCamera_getImage(&img, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, frame));
    CHECK_DW_ERROR(dwImage_copyConvert(m_imageRGBA[cameraIdx], img, m_context));

    // Run data conditioner to prepare input for the network on this camera's stream
    CHECK_DW_ERROR(dwDataConditioner_prepareData(ctx.dnnInput, &m_imageRGBA[cameraIdx], 1, 
                                                 &m_detectionRegion[cameraIdx],
                                                 cudaAddressModeClamp, ctx.dataConditioner));
}

void MultiStreamCameraDNNApp::doInferenceAsync(uint32_t cameraIdx, CameraProcessingContext& ctx)
{
    // Set the DNN to use this camera's CUDA stream
    CHECK_DW_ERROR(dwDNN_setCUDAStream(ctx.cudaStream, m_dnn));
    
    // Run inference on camera's dedicated stream
    dwConstDNNTensorHandle_t inputs[1U] = {ctx.dnnInput};
    CHECK_DW_ERROR(dwDNN_infer(ctx.dnnOutputsDevice, m_numOutputTensors, inputs, 1U, m_dnn));
    
    // Start async transfer of results to host memory
    for (uint32_t outputIdx = 0U; outputIdx < m_numOutputTensors; ++outputIdx)
    {
        dwDNNTensorStreamerHandle_t streamer = ctx.dnnOutputStreamers[outputIdx];
        CHECK_DW_ERROR(dwDNNTensorStreamer_producerSend(ctx.dnnOutputsDevice[outputIdx], streamer));
    }
}

void MultiStreamCameraDNNApp::collectInferenceResults(uint32_t cameraIdx, CameraProcessingContext& ctx)
{
    // Receive streamed results
    for (uint32_t outputIdx = 0U; outputIdx < m_numOutputTensors; ++outputIdx)
    {
        dwDNNTensorStreamerHandle_t streamer = ctx.dnnOutputStreamers[outputIdx];
        CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReceive(&ctx.outputsHost[outputIdx], 1000, streamer));
    }

    // Interpret results and update detection lists (thread-safe)
    {
        std::lock_guard<std::mutex> lock(ctx.resultsMutex);
        
        if (m_usecuDLA)
        {
            interpretOutput(ctx.outputsHost[m_cvgIdx], ctx.outputsHost[m_bboxIdx], 
                          &m_detectionRegion[cameraIdx], cameraIdx, ctx);
        }
        else
        {
            interpretOutput(ctx.outputsHost[m_cvgIdx], &m_detectionRegion[cameraIdx], cameraIdx, ctx);
        }
    }

    // Return streamed tensors
    for (uint32_t outputIdx = 0U; outputIdx < m_numOutputTensors; ++outputIdx)
    {
        dwDNNTensorStreamerHandle_t streamer = ctx.dnnOutputStreamers[outputIdx];
        CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReturn(&ctx.outputsHost[outputIdx], streamer));
        CHECK_DW_ERROR(dwDNNTensorStreamer_producerReturn(nullptr, 1000, streamer));
    }
}

void MultiStreamCameraDNNApp::onRenderHelper(dwCameraFrameHandle_t frame, uint8_t cameraIndex)
{
    dwImageHandle_t img = DW_NULL_HANDLE;
    dwCameraOutputType outputType = DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8;

    CHECK_DW_ERROR(dwRenderEngine_setTile(m_tileVideo[cameraIndex], m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_resetTile(m_renderEngine));

    CHECK_DW_ERROR(dwSensorCamera_getImage(&img, outputType, frame));

    // Stream image to GL domain
    CHECK_DW_ERROR(dwImageStreamerGL_producerSend(img, m_streamerToGL[cameraIndex]));

    // Receive streamed image
    dwImageHandle_t frameGL;
    CHECK_DW_ERROR(dwImageStreamerGL_consumerReceive(&frameGL, 33000, m_streamerToGL[cameraIndex]));

    dwImageGL* imageGL;
    CHECK_DW_ERROR(dwImage_getGL(&imageGL, frameGL));

    // Render image
    {
        dwVector2f range{};
        range.x = imageGL->prop.width;
        range.y = imageGL->prop.height;
        CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_renderImage2D(imageGL, {0, 0, range.x, range.y}, m_renderEngine));

        // Render detection region (yellow outline)
        dwRectf detectionRegionFloat;
        detectionRegionFloat.x = m_detectionRegion[cameraIndex].x;
        detectionRegionFloat.y = m_detectionRegion[cameraIndex].y;
        detectionRegionFloat.width = m_detectionRegion[cameraIndex].width;
        detectionRegionFloat.height = m_detectionRegion[cameraIndex].height;

        CHECK_DW_ERROR(dwRenderEngine_setColor({1.0f, 1.0f, 0.0f, 1.0f}, m_renderEngine)); // Yellow
        CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D, &detectionRegionFloat,
                                             sizeof(dwRectf), 0, 1, m_renderEngine));

        // Render current detection results
        renderDetectionResults(cameraIndex);
        
        // Render performance stats
        renderPerformanceStats(cameraIndex);
    }

    CHECK_DW_ERROR(dwImageStreamerGL_consumerReturn(&frameGL, m_streamerToGL[cameraIndex]));
    CHECK_DW_ERROR(dwImageStreamerGL_producerReturn(nullptr, 32000, m_streamerToGL[cameraIndex]));
}

void MultiStreamCameraDNNApp::renderDetectionResults(uint32_t cameraIdx)
{
    CameraProcessingContext& ctx = m_cameraContexts[cameraIdx];
    
    // Thread-safe access to detection results
    std::lock_guard<std::mutex> lock(ctx.resultsMutex);
    
    if (!ctx.detectedBoxListFloat.empty())
    {
        // Render detection boxes (red for cars, blue for people, etc.)
        CHECK_DW_ERROR(dwRenderEngine_setColor({1.0f, 0.0f, 0.0f, 1.0f}, m_renderEngine)); // Red
        CHECK_DW_ERROR(dwRenderEngine_setLineWidth(2.0f, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D,
                                             ctx.detectedBoxListFloat.data(), sizeof(dwRectf), 0,
                                             ctx.detectedBoxListFloat.size(), m_renderEngine));
        
        // Render labels if available
        if (ctx.labels.size() == ctx.detectedBoxListFloat.size())
        {
            CHECK_DW_ERROR(dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, m_renderEngine)); // White text
            CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_12, m_renderEngine));
            
            for (size_t i = 0; i < ctx.labels.size(); ++i)
            {
                const dwRectf& box = ctx.detectedBoxListFloat[i];
                dwVector2f labelPos = {box.x + 2, box.y - 5}; // Slightly above box
                CHECK_DW_ERROR(dwRenderEngine_renderText2D(ctx.labels[i].c_str(), labelPos, m_renderEngine));
            }
        }
    }
}

void MultiStreamCameraDNNApp::renderPerformanceStats(uint32_t cameraIdx)
{
    CameraProcessingContext& ctx = m_cameraContexts[cameraIdx];
    
    dwTime_t timestamp;
    dwImageHandle_t img = DW_NULL_HANDLE;
    // Get timestamp from current frame (we need to get this from the frame)
    
    std::string statusStr = "Cam:" + std::to_string(cameraIdx);
    statusStr += " Det:" + std::to_string(ctx.detectedBoxListFloat.size());
    statusStr += " Inf:" + std::to_string(static_cast<int>(ctx.avgInferenceTimeMs)) + "ms";
    statusStr += " Frm:" + std::to_string(ctx.frameId.load());
    
    if (ctx.inferenceInProgress.load())
    {
        statusStr += " [PROCESSING]";
        CHECK_DW_ERROR(dwRenderEngine_setColor({1.0f, 1.0f, 0.0f, 1.0f}, m_renderEngine)); // Yellow for processing
    }
    else if (ctx.resultsReady.load())
    {
        statusStr += " [READY]";
        CHECK_DW_ERROR(dwRenderEngine_setColor({0.0f, 1.0f, 0.0f, 1.0f}, m_renderEngine)); // Green for ready
    }
    else
    {
        statusStr += " [IDLE]";
        CHECK_DW_ERROR(dwRenderEngine_setColor({0.7f, 0.7f, 0.7f, 1.0f}, m_renderEngine)); // Gray for idle
    }

    CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_renderText2D(statusStr.c_str(), {25, 80}, m_renderEngine));
}

void MultiStreamCameraDNNApp::updatePerformanceStats()
{
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto timeSinceLastUpdate = std::chrono::duration_cast<std::chrono::seconds>(currentTime - m_lastStatsTime);
    
    if (timeSinceLastUpdate.count() >= 2) // Update every 2 seconds
    {
        printPerformanceMetrics();
        m_lastStatsTime = currentTime;
        m_totalInferencesPerSecond = 0; // Reset counter
    }
}

void MultiStreamCameraDNNApp::printPerformanceMetrics()
{
    std::cout << "=== Performance Metrics ===" << std::endl;
    std::cout << "Total Frame Count: " << m_frameCount.load() << std::endl;
    std::cout << "Inferences/sec (last 2s): " << m_totalInferencesPerSecond / 2 << std::endl;
    
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        CameraProcessingContext& ctx = m_cameraContexts[i];
        std::cout << "Camera " << i << ": "
                  << "Avg Inference: " << ctx.avgInferenceTimeMs << "ms, "
                  << "Processed: " << ctx.processedFrameCount << " frames, "
                  << "Status: " << (ctx.inferenceInProgress.load() ? "PROCESSING" : "IDLE")
                  << std::endl;
    }
}

// DNN Output Interpretation Methods (Updated for per-camera contexts)
void MultiStreamCameraDNNApp::interpretOutput(dwDNNTensorHandle_t outConf, dwDNNTensorHandle_t outBBox, 
                                             const dwRect* const roi, uint32_t cameraIdx, CameraProcessingContext& ctx)
{
    ctx.detectedBoxList.clear();
    ctx.detectedBoxListFloat.clear();
    uint32_t numBBoxes = 0U;

    void* tmpConf;
    CHECK_DW_ERROR(dwDNNTensor_lock(&tmpConf, outConf));
    dwFloat16_t* confData = reinterpret_cast<dwFloat16_t*>(tmpConf);

    void* tmpBbox;
    CHECK_DW_ERROR(dwDNNTensor_lock(&tmpBbox, outBBox));
    dwFloat16_t* bboxData = reinterpret_cast<dwFloat16_t*>(tmpBbox);

    size_t offsetY;
    size_t strideY;
    size_t height;
    uint32_t indices[4] = {0, 0, 0, 0};
    CHECK_DW_ERROR(dwDNNTensor_getLayoutView(&offsetY, &strideY, &height, indices, 4U, 1U, outConf));

    for (uint32_t gridY = 0U; gridY < height; ++gridY)
    {
        size_t offsetX;
        size_t strideX;
        size_t width;
        uint32_t subIndices[4] = {0, gridY, 0, 0};
        CHECK_DW_ERROR(dwDNNTensor_getLayoutView(&offsetX, &strideX, &width, subIndices, 4U, 0U, outConf));
        
        for (uint32_t gridX = 0U; gridX < width; gridX += strideX)
        {
            dwFloat16_t conf = confData[offsetX + gridX * strideX];
            if (conf > COVERAGE_THRESHOLD && numBBoxes < m_maxDetections)
            {
                float32_t imageX = (float32_t)gridX * (float32_t)m_cellSize;
                float32_t imageY = (float32_t)gridY * (float32_t)m_cellSize;

                size_t bboxOffset;
                size_t bboxStride;
                size_t numDimensions;
                uint32_t bboxIndices[4] = {gridX, gridY, 0, 0};
                CHECK_DW_ERROR(dwDNNTensor_getLayoutView(&bboxOffset, &bboxStride, &numDimensions, bboxIndices, 4U, 2U, outBBox));
                dwFloat16_t* bboxOut = &(bboxData[bboxOffset + 0 * bboxStride]);

                float32_t boxX1, boxY1, boxX2, boxY2;

                dwDataConditioner_outputPositionToInput(&boxX1, &boxY1, bboxOut[0 * bboxStride] + imageX,
                                                        bboxOut[1 * bboxStride] + imageY, roi, ctx.dataConditioner);
                dwDataConditioner_outputPositionToInput(&boxX2, &boxY2, bboxOut[2 * bboxStride] + imageX,
                                                        bboxOut[3 * bboxStride] + imageY, roi, ctx.dataConditioner);

                dwRectf bboxFloat{boxX1, boxY1, boxX2 - boxX1, boxY2 - boxY1};
                dwBox2D bbox;
                bbox.width  = static_cast<int32_t>(std::round(bboxFloat.width));
                bbox.height = static_cast<int32_t>(std::round(bboxFloat.height));
                bbox.x      = static_cast<int32_t>(std::round(bboxFloat.x));
                bbox.y      = static_cast<int32_t>(std::round(bboxFloat.y));

                ctx.detectedBoxList.push_back(bbox);
                ctx.detectedBoxListFloat.push_back(bboxFloat);
                numBBoxes++;
            }
        }
    }

    CHECK_DW_ERROR(dwDNNTensor_unlock(outConf));
    CHECK_DW_ERROR(dwDNNTensor_unlock(outBBox));
}

void MultiStreamCameraDNNApp::interpretOutput(dwDNNTensorHandle_t outConf, const dwRect* const roi, 
                                             uint32_t cameraIdx, CameraProcessingContext& ctx)
{
    ctx.detectedBoxList.clear();
    ctx.detectedBoxListFloat.clear();
    ctx.labels.clear();

    uint32_t numBBoxes = 0U;

    void* tmpConf;
    CHECK_DW_ERROR(dwDNNTensor_lock(&tmpConf, outConf));
    float32_t* confData = reinterpret_cast<float32_t*>(tmpConf);

    size_t offsetY;
    size_t strideY;
    size_t height;
    uint32_t indices[4] = {0, 0, 0, 0};
    CHECK_DW_ERROR(dwDNNTensor_getLayoutView(&offsetY, &strideY, &height, indices, 4U, 1U, outConf));

    std::vector<YoloScoreRect> tmpRes;
    for (uint16_t gridY = 0U; gridY < height; ++gridY)
    {
        const float32_t* outConfRow = &confData[gridY * strideY];
        if (outConfRow[4] < CONFIDENCE_THRESHOLD || numBBoxes >= 100)
        {
            continue;
        }
        
        uint16_t maxIndex  = 0;
        float32_t maxScore = 0;
        for (uint16_t i = 5; i < 85; i++)
        { 
            if (outConfRow[i] > maxScore)
            {
                maxScore = outConfRow[i];
                maxIndex = i;
            }
        }

        if (maxScore > SCORE_THRESHOLD)
        {
            float32_t imageX = (float32_t)outConfRow[0];
            float32_t imageY = (float32_t)outConfRow[1];
            float32_t bboxW  = (float32_t)outConfRow[2];
            float32_t bboxH  = (float32_t)outConfRow[3];

            float32_t boxX1Tmp = (float32_t)(imageX - 0.5 * bboxW);
            float32_t boxY1Tmp = (float32_t)(imageY - 0.5 * bboxH);
            float32_t boxX2Tmp = (float32_t)(imageX + 0.5 * bboxW);
            float32_t boxY2Tmp = (float32_t)(imageY + 0.5 * bboxH);

            float32_t boxX1, boxY1, boxX2, boxY2;

            dwDataConditioner_outputPositionToInput(&boxX1, &boxY1, boxX1Tmp, boxY1Tmp, roi, ctx.dataConditioner);
            dwDataConditioner_outputPositionToInput(&boxX2, &boxY2, boxX2Tmp, boxY2Tmp, roi, ctx.dataConditioner);
            
            dwRectf bboxFloat{boxX1, boxY1, boxX2 - boxX1, boxY2 - boxY1};
            tmpRes.push_back({bboxFloat, maxScore, (uint16_t)(maxIndex - 5)});
            numBBoxes++;
        }
    }

    CHECK_DW_ERROR(dwDNNTensor_unlock(outConf));

    // Apply NMS
    std::vector<YoloScoreRect> tmpResAfterNMS = doNmsForYoloOutputBoxes(tmpRes, float32_t(0.45));
    
    for (uint32_t i = 0; i < tmpResAfterNMS.size(); i++)
    {
        YoloScoreRect box = tmpResAfterNMS[i];
        
        // Filter for automotive-relevant classes only
        if (m_automotiveClasses.find(YOLO_CLASS_NAMES[box.classIndex]) != m_automotiveClasses.end())
        {
            dwRectf bboxFloat = box.rectf;
            dwBox2D bbox;
            bbox.width  = static_cast<int32_t>(std::round(bboxFloat.width));
            bbox.height = static_cast<int32_t>(std::round(bboxFloat.height));
            bbox.x      = static_cast<int32_t>(std::round(bboxFloat.x));
            bbox.y      = static_cast<int32_t>(std::round(bboxFloat.y));

            ctx.detectedBoxList.push_back(bbox);
            ctx.detectedBoxListFloat.push_back(bboxFloat);
            ctx.labels.push_back(YOLO_CLASS_NAMES[box.classIndex]);
        }
    }
}

void MultiStreamCameraDNNApp::onRelease()
{
    std::cout << "Releasing multi-stream camera DNN application..." << std::endl;
    
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

    // Release per-camera resources
    for (uint32_t i = 0; i < m_totalCameras; ++i)
    {
        CameraProcessingContext& ctx = m_cameraContexts[i];
        
        // Destroy CUDA events
        if (ctx.frameReadyEvent)
        {
            cudaError_t status = cudaEventDestroy(ctx.frameReadyEvent);
            if (status != cudaSuccess) {
                std::cout << "Warning: Failed to destroy frame ready event for camera " << i << std::endl;
            }
        }
        if (ctx.inferenceCompleteEvent)
        {
            cudaError_t status = cudaEventDestroy(ctx.inferenceCompleteEvent);
            if (status != cudaSuccess) {
                std::cout << "Warning: Failed to destroy inference complete event for camera " << i << std::endl;
            }
        }
        
        // Destroy CUDA stream
        if (ctx.cudaStream)
        {
            cudaError_t status = cudaStreamDestroy(ctx.cudaStream);
            if (status != cudaSuccess) {
                std::cout << "Warning: Failed to destroy CUDA stream for camera " << i << std::endl;
            }
        }
        
        // Release DNN tensors
        if (ctx.dnnInput)
        {
            CHECK_DW_ERROR(dwDNNTensor_destroy(ctx.dnnInput));
        }
        
        for (uint32_t outputIdx = 0U; outputIdx < NUM_OUTPUT_TENSORS; ++outputIdx)
        {
            if (ctx.dnnOutputsDevice[outputIdx])
            {
                CHECK_DW_ERROR(dwDNNTensor_destroy(ctx.dnnOutputsDevice[outputIdx]));
            }
            if (ctx.dnnOutputStreamers[outputIdx])
            {
                CHECK_DW_ERROR(dwDNNTensorStreamer_release(ctx.dnnOutputStreamers[outputIdx]));
            }
        }
        
        // Release data conditioner
        if (ctx.dataConditioner)
        {
            CHECK_DW_ERROR(dwDataConditioner_release(ctx.dataConditioner));
        }
        
        // Release image and streaming resources
        if (m_streamerToGL[i])
        {
            dwImageStreamerGL_release(m_streamerToGL[i]);
        }
        if (m_camera[i])
        {
            dwSAL_releaseSensor(m_camera[i]);
        }
        if (m_imageRGBA[i])
        {
            CHECK_DW_ERROR(dwImage_destroy(m_imageRGBA[i]));
        }
    }

    // Release shared DNN model
    if (m_dnn)
    {
        CHECK_DW_ERROR(dwDNN_release(m_dnn));
    }

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
        
    std::cout << "Multi-stream camera DNN application released successfully" << std::endl;
}

void MultiStreamCameraDNNApp::onKeyDown(int key, int scancode, int mods)
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
}

// Utility Methods (unchanged from original)
std::string MultiStreamCameraDNNApp::getPlatformPrefix()
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

    if (m_usecuDLA)
    {
        path = "cudla";
    }
    else if (gpuProp.major == CUDA_AMPERE_MAJOR_COMPUTE_CAPABILITY)
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

bool MultiStreamCameraDNNApp::sort_score(YoloScoreRect box1, YoloScoreRect box2)
{
    return box1.score > box2.score;
}

float32_t MultiStreamCameraDNNApp::calculateIouOfBoxes(dwRectf box1, dwRectf box2)
{
    float32_t x1 = std::max(box1.x, box2.x);
    float32_t y1 = std::max(box1.y, box2.y);
    float32_t x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float32_t y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    float32_t w = std::max(0.0f, x2 - x1);
    float32_t h = std::max(0.0f, y2 - y1);
    float32_t over_area = w * h;
    return float32_t(over_area) / float32_t(box1.width * box1.height + box2.width * box2.height - over_area);
}

std::vector<MultiStreamCameraDNNApp::YoloScoreRect> MultiStreamCameraDNNApp::doNmsForYoloOutputBoxes(
    std::vector<YoloScoreRect>& boxes, float32_t threshold)
{
    std::vector<YoloScoreRect> results;
    std::sort(boxes.begin(), boxes.end(), sort_score);
    
    while (boxes.size() > 0)
    {
        results.push_back(boxes[0]);
        uint32_t index = 1;
        while (index < boxes.size())
        {
            float32_t iou_value = calculateIouOfBoxes(boxes[0].rectf, boxes[index].rectf);
            if (iou_value > threshold)
            {
                boxes.erase(boxes.begin() + index);
            }
            else
            {
                index++;
            }
        }
        boxes.erase(boxes.begin());
    }
    return results;
}

//#######################################################################################
int main(int argc, const char** argv)
{
    ProgramArguments args(argc, argv,
                          {
                              ProgramArguments::Option_t("rig", (dw_samples::SamplesDataPath::get() + "/samples/sensors/camera/camera/rig_4cam.json").c_str(), "Rig configuration for 4 cameras"),
#ifdef VIBRANTE
                              ProgramArguments::Option_t("cudla", "0", "run inference on cudla"),
                              ProgramArguments::Option_t("dla-engine", "0", "dla engine number to run on if --cudla=1"),
#endif
                              ProgramArguments::Option_t("tensorRT_model", "", (std::string("path to TensorRT model file. By default: ") + dw_samples::SamplesDataPath::get() + "/samples/detector/<gpu-architecture>/tensorRT_model.bin").c_str())},
                          "Multi-Stream 4-Camera DNN application with parallel object detection.");

    MultiStreamCameraDNNApp app(args);
    app.initializeWindow("Multi-Stream Camera DNN", 1280, 800, args.enabled("offscreen"));
    
    if (!args.enabled("offscreen"))
        app.setProcessRate(30);

    return app.run();
}