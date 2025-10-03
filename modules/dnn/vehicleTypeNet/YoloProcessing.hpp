/////////////////////////////////////////////////////////////////////////////////////////
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA CORPORATION & AFFILIATES assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA CORPORATION & AFFILIATES. No third party distribution is allowed unless
// expressly authorized by NVIDIA. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA CORPORATION & AFFILIATES products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA CORPORATION & AFFILIATES.
//
// SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <set>
#include <atomic>
#include <mutex>

#include <driver_types.h>
#include <texture_types.h>

#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/core/platform/GPUProperties.h>
#include <dw/dnn/DNN.h>
#include <dw/dnn/dataconditioner/DataConditioner.h>
#include <dw/dnn/tensor/Tensor.h>
#include <dw/image/Image.h>
#include <dw/interop/streamer/TensorStreamer.h>
#include <dw/interop/streamer/ImageStreamer.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/camera/Camera.h>
#include <dw/rig/Rig.h>

#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Visualization.h>
#include <dwvisualization/image/Image.h>
#include <dwvisualization/interop/ImageStreamer.h>

#include <framework/Checks.hpp>
#include <framework/CudaTimer.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/Log.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/SamplesDataPath.hpp>
#include <framework/ScreenshotHelper.hpp>
#include <framework/WindowGLFW.hpp>


#include "vehicleTypeNetProcessing.hpp"

using namespace dw_samples::common;

#define MAX_PORTS_COUNT 4 
#define MAX_CAMS_PER_PORT 4
#define MAX_CAMS MAX_PORTS_COUNT * MAX_CAMS_PER_PORT

// Per-Camera Processing Context
struct CameraProcessingContext {
    // CUDA stream for this camera
    cudaStream_t cudaStream;
    
    // DNN resources per camera
    dwDNNTensorHandle_t dnnInput;
    dwDNNTensorHandle_t dnnOutputsDevice[2];
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
    
    // VehicleTypeNet classification results
    std::vector<VehicleTypeNetProcessor::ClassificationResult> vehicleTypes;
    std::vector<size_t> vehicleDetectionIndices;
    
    mutable std::mutex resultsMutex;
    
    // Performance metrics
    std::chrono::high_resolution_clock::time_point lastProcessTime;
    float avgInferenceTimeMs;
    float avgClassificationTimeMs;
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
        avgClassificationTimeMs(0.0f),
        processedFrameCount(0)
    {
        dnnOutputsDevice[0] = DW_NULL_HANDLE;
        dnnOutputsDevice[1] = DW_NULL_HANDLE;
        dnnOutputStreamers[0] = DW_NULL_HANDLE;
        dnnOutputStreamers[1] = DW_NULL_HANDLE;
        
        detectedBoxList.reserve(1000);
        detectedBoxListFloat.reserve(1000);
        labels.reserve(1000);
        vehicleTypes.reserve(100);
        vehicleDetectionIndices.reserve(100);
    }
};

class DNNTensorSample : public DriveWorksSample
{
private:
    // ============================================
    // MULTI-CAMERA SECTION 
    // ============================================
    dwContextHandle_t m_sdk = DW_NULL_HANDLE;
    dwVisualizationContextHandle_t m_viz = DW_NULL_HANDLE;
    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
    dwSALHandle_t m_sal = DW_NULL_HANDLE;
    
    uint32_t m_totalCameras;
    dwRigHandle_t m_rigConfig{};
    
    // Per camera resources
    dwSensorHandle_t m_camera[MAX_CAMS];
    dwImageStreamerHandle_t m_streamerToGL[MAX_CAMS] = {DW_NULL_HANDLE};
    uint32_t m_tileVideo[MAX_CAMS];
    bool m_enableRender[MAX_CAMS] = {true};
    
    bool m_useRaw = false;
    bool m_useProcessed = false;
    bool m_useProcessed1 = false;
    bool m_useProcessed2 = false;
    uint32_t m_fifoSize = 0U;
    std::atomic<uint32_t> m_frameCount{0};
    
    // ============================================
    // MULTI-STREAM DNN SECTION
    // ============================================
    static constexpr float32_t COVERAGE_THRESHOLD = 0.6f;
    static constexpr float32_t CONFIDENCE_THRESHOLD = 0.45f;
    static constexpr float32_t SCORE_THRESHOLD = 0.25f;
    static constexpr uint32_t NUM_OUTPUT_TENSORS = 2U;
    
    const uint32_t m_maxDetections = 1000U;
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
    
    // VehicleTypeNet processor (shared across all cameras)
    std::unique_ptr<VehicleTypeNetProcessor> m_vtProcessor;
    bool m_enableVehicleClassification = true;
    
    // Performance monitoring
    std::chrono::high_resolution_clock::time_point m_lastStatsTime;
    uint32_t m_totalInferencesPerSecond = 0;
    
    std::unique_ptr<ScreenshotHelper> m_screenshot;

public:
    DNNTensorSample(const ProgramArguments& args)
        : DriveWorksSample(args)
    {
        m_cameraContexts.reset(new CameraProcessingContext[MAX_CAMS]);
        m_lastStatsTime = std::chrono::high_resolution_clock::now();
    }
    
    bool onInitialize() override
    {
        initDWAndSAL();
        initializeMultiCamera();
        CHECK_DW_ERROR(dwSAL_start(m_sal));
        initRender();
        initDNN();
        
        m_screenshot.reset(new ScreenshotHelper(m_sdk, m_sal, getWindowWidth(), getWindowHeight(), "VehicleTypeNet"));
        
        std::cout << "Starting all camera sensors..." << std::endl;
        for (uint32_t i = 0; i < m_totalCameras; ++i)
        {
            CHECK_DW_ERROR(dwSensor_start(m_camera[i]));
        }
        
        std::cout << "Multi-camera VehicleTypeNet application initialized" << std::endl;
        return true;
    }
    
    void onProcess() override {}
    
    void onRender() override
    {
        // Capture frames from all cameras
        dwCameraFrameHandle_t frames[MAX_CAMS];
        
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
                    std::cout << "Camera " << i << " timeout" << std::endl;
                    break;
                }
                
                if (status == DW_END_OF_STREAM)
                {
                    dwSensor_reset(m_camera[i]);
                }
            }
            
            if (status != DW_SUCCESS && status != DW_NOT_READY && status != DW_TIME_OUT)
            {
                std::cout << "Camera " << i << " error: " << dwGetStatusName(status) << std::endl;
            }
        }
        
        // Process all cameras in parallel
        processAllCamerasParallel(frames);
        
        // Render all cameras
        for (uint32_t i = 0; i < m_totalCameras; ++i)
        {
            if (m_useProcessed && m_enableRender[i])
            {
                renderCameraFrame(frames[i], i);
            }
        }
        
        // Return frames
        for (uint32_t i = 0; i < m_totalCameras; ++i)
        {
            CHECK_DW_ERROR(dwSensorCamera_returnFrame(&frames[i]));
        }
        
        m_frameCount++;
        updatePerformanceStats();
    }
    
    void onRelease() override
    {
        // Stop sensors
        for (uint32_t i = 0; i < m_totalCameras; ++i)
        {
            if (m_camera[i])
            {
                CHECK_DW_ERROR(dwSensor_stop(m_camera[i]));
            }
        }
        
        // Wait for in-progress inferences
        bool allComplete = false;
        int maxWait = 100;
        int waitCount = 0;
        
        while (!allComplete && waitCount < maxWait)
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
            
            if (ctx.frameReadyEvent)
                cudaEventDestroy(ctx.frameReadyEvent);
            if (ctx.inferenceCompleteEvent)
                cudaEventDestroy(ctx.inferenceCompleteEvent);
            if (ctx.cudaStream)
                cudaStreamDestroy(ctx.cudaStream);
            if (ctx.dnnInput)
                CHECK_DW_ERROR(dwDNNTensor_destroy(ctx.dnnInput));
            
            for (uint32_t outputIdx = 0U; outputIdx < NUM_OUTPUT_TENSORS; ++outputIdx)
            {
                if (ctx.dnnOutputsDevice[outputIdx])
                    CHECK_DW_ERROR(dwDNNTensor_destroy(ctx.dnnOutputsDevice[outputIdx]));
                if (ctx.dnnOutputStreamers[outputIdx])
                    CHECK_DW_ERROR(dwDNNTensorStreamer_release(ctx.dnnOutputStreamers[outputIdx]));
            }
            
            if (ctx.dataConditioner)
                CHECK_DW_ERROR(dwDataConditioner_release(ctx.dataConditioner));
            if (m_streamerToGL[i])
                dwImageStreamerGL_release(m_streamerToGL[i]);
            if (m_camera[i])
                dwSAL_releaseSensor(m_camera[i]);
            if (m_imageRGBA[i])
                CHECK_DW_ERROR(dwImage_destroy(m_imageRGBA[i]));
        }
        
        // Release shared resources
        if (m_dnn)
            CHECK_DW_ERROR(dwDNN_release(m_dnn));
        
        m_vtProcessor.reset();
        m_screenshot.reset();
        
        if (m_rigConfig)
            dwRig_release(m_rigConfig);
        if (m_sal)
            dwSAL_release(m_sal);
        if (m_renderEngine)
            dwRenderEngine_release(m_renderEngine);
        if (m_viz)
            dwVisualizationRelease(m_viz);
        if (m_sdk)
            dwRelease(m_sdk);
    }
    
    void onReset() override
    {
        CHECK_DW_ERROR(dwDNN_reset(m_dnn));
        for (uint32_t i = 0; i < m_totalCameras; ++i)
        {
            if (m_cameraContexts[i].dataConditioner)
            {
                CHECK_DW_ERROR(dwDataConditioner_reset(m_cameraContexts[i].dataConditioner));
            }
        }
    }
    
    void onResizeWindow(int width, int height) override
    {
        CHECK_DW_ERROR(dwRenderEngine_reset(m_renderEngine));
        dwRectf rect;
        rect.width = width;
        rect.height = height;
        rect.x = 0;
        rect.y = 0;
        CHECK_DW_ERROR(dwRenderEngine_setBounds(rect, m_renderEngine));
    }
    
    void onKeyDown(int key, int /*scancode*/, int /*mods*/) override
    {
        if (key == GLFW_KEY_S)
        {
            m_screenshot->triggerScreenshot();
        }
        else if (key == GLFW_KEY_P)
        {
            printPerformanceMetrics();
        }
    }

protected:
    dwContextHandle_t getSDKContext()
    {
        return m_sdk;
    }

private:
    void initDWAndSAL()
    {
        CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
        CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));
        
        dwContextParameters sdkParams = {};
#ifdef VIBRANTE
        sdkParams.eglDisplay = getEGLDisplay();
#endif
        CHECK_DW_ERROR(dwInitialize(&m_sdk, DW_VERSION, &sdkParams));
        CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_sdk));
    }
    
    void initializeMultiCamera()
    {
        m_totalCameras = 0;
        CHECK_DW_ERROR(dwRig_initializeFromFile(&m_rigConfig, m_sdk,
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
            paramsClient[i].protocol = protocol;
            paramsClient[i].parameters = params;
            
            m_enableRender[i] = true;
            
            std::cout << "Initializing camera " << i << " with params: " << params << std::endl;
            CHECK_DW_ERROR(dwSAL_createSensor(&m_camera[m_totalCameras], paramsClient[i], m_sal));
            
            m_totalCameras++;
            
            // Parse output format
            m_useRaw = std::string::npos != std::string(params).find("raw");
            m_useProcessed = std::string::npos != std::string(params).find("processed");
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
            CHECK_DW_ERROR(dwImage_create(&m_imageRGBA[i], imageProperties, m_sdk));
            CHECK_DW_ERROR(dwImageStreamerGL_initialize(&m_streamerToGL[i], &imageProperties, DW_IMAGE_GL, m_sdk));
        }
    }
    
    void initRender()
    {
        CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_sdk));
        
        dwRenderEngineParams params{};
        CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&params, getWindowWidth(), getWindowHeight()));
        params.defaultTile.lineWidth = 2.0f;
        params.defaultTile.font = DW_RENDER_ENGINE_FONT_VERDANA_20;
        params.maxBufferCount = 1;
        
        float32_t windowSize[2] = {static_cast<float32_t>(getWindowWidth()), static_cast<float32_t>(getWindowHeight())};
        params.bounds = {0, 0, windowSize[0], windowSize[1]};
        
        uint32_t tilesPerRow = 1;
        switch (m_totalCameras)
        {
        case 1: tilesPerRow = 1; break;
        case 2: 
            params.bounds.height = (windowSize[1] / 2);
            params.bounds.y = (windowSize[1] / 2);
            tilesPerRow = 2;
            break;
        case 3:
        case 4: tilesPerRow = 2; break;
        default: tilesPerRow = 4; break;
        }
        
        CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_viz));
        
        dwRenderEngineTileState paramList[MAX_CAMS];
        for (uint32_t i = 0; i < m_totalCameras; ++i)
        {
            dwRenderEngine_initTileState(&paramList[i]);
            paramList[i].modelViewMatrix = DW_IDENTITY_MATRIX4F;
            paramList[i].font = DW_RENDER_ENGINE_FONT_VERDANA_20;
        }
        
        CHECK_DW_ERROR(dwRenderEngine_addTilesByCount(m_tileVideo, m_totalCameras, tilesPerRow, paramList, m_renderEngine));
    }
    
    void initDNN()
    {
#ifdef VIBRANTE
        m_usecuDLA = getArgument("cudla").compare("1") == 0;
        m_dlaEngineNo = std::atoi(getArgument("dla-engine").c_str());
#endif
        
        std::string yoloModel = getArgument("tensorRT_model");
        if (yoloModel.empty())
        {
            yoloModel = dw_samples::SamplesDataPath::get() + "/samples/detector/";
            yoloModel += getPlatformPrefix();
            yoloModel += "/yolov3_640x640";
            if (m_usecuDLA)
                yoloModel += ".dla";
            yoloModel += ".bin";
        }
        
        std::cout << "Loading YOLO model: " << yoloModel << std::endl;
        
        // Initialize shared DNN model
        CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFileWithEngineId(&m_dnn, yoloModel.c_str(), nullptr,
                                                                   m_usecuDLA ? DW_PROCESSOR_TYPE_CUDLA : DW_PROCESSOR_TYPE_GPU,
                                                                   m_dlaEngineNo, m_sdk));
        
        m_numOutputTensors = m_usecuDLA ? NUM_OUTPUT_TENSORS : 1U;
        
        // Get DNN metadata and tensor properties
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
            const char* coverageBlobName = "coverage";
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
        
        std::cout << "YOLO Model loaded successfully" << std::endl;
        
        // Initialize per-camera processing contexts
        for (uint32_t i = 0; i < m_totalCameras; ++i)
        {
            initializeCameraContext(i, inputProps, outputProps, metadata);
            
            m_detectionRegion[i].width = m_imageWidth;
            m_detectionRegion[i].height = m_imageHeight;
            m_detectionRegion[i].x = 0;
            m_detectionRegion[i].y = 0;
        }
        
        // Initialize VehicleTypeNet
        if (m_enableVehicleClassification)
        {
            std::string vtModelPath = dw_samples::SamplesDataPath::get() + "/samples/detector/";
            vtModelPath += getPlatformPrefix();
            vtModelPath += "/vehicletypenet";
            if (m_usecuDLA)
                vtModelPath += ".dla";
            vtModelPath += ".bin";
            
            std::cout << "Loading VehicleTypeNet model: " << vtModelPath << std::endl;
            
            try {
                m_vtProcessor.reset(new VehicleTypeNetProcessor(
                    m_sdk,
                    nullptr,  // Stream set per-call
                    vtModelPath,
                    m_usecuDLA ? DW_PROCESSOR_TYPE_CUDLA : DW_PROCESSOR_TYPE_GPU,
                    m_dlaEngineNo));
                
                std::cout << "VehicleTypeNet loaded successfully" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to load VehicleTypeNet: " << e.what() << std::endl;
                m_enableVehicleClassification = false;
            }
        }
        
        std::cout << "Multi-stream DNN initialization complete for " << m_totalCameras << " cameras" << std::endl;
    }
    
    void initializeCameraContext(uint32_t cameraIdx, const dwDNNTensorProperties& inputProps, 
                                const dwDNNTensorProperties outputProps[NUM_OUTPUT_TENSORS],
                                const dwDNNMetaData& metadata)
    {
        CameraProcessingContext& ctx = m_cameraContexts[cameraIdx];
        
        std::cout << "Initializing camera context " << cameraIdx << std::endl;
        
        // Create dedicated CUDA stream
        cudaError_t cudaStatus = cudaStreamCreate(&ctx.cudaStream);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream for camera " + std::to_string(cameraIdx));
        }
        
        // Create CUDA events
        CHECK_CUDA_ERROR(cudaEventCreate(&ctx.frameReadyEvent));
        CHECK_CUDA_ERROR(cudaEventCreate(&ctx.inferenceCompleteEvent));
        
        // Create per-camera DNN tensors
        CHECK_DW_ERROR(dwDNNTensor_create(&ctx.dnnInput, &inputProps, m_sdk));
        
        for (uint32_t outputIdx = 0U; outputIdx < m_numOutputTensors; ++outputIdx)
        {
            CHECK_DW_ERROR(dwDNNTensor_create(&ctx.dnnOutputsDevice[outputIdx], &outputProps[outputIdx], m_sdk));
            
            dwDNNTensorProperties hostProps = outputProps[outputIdx];
            hostProps.tensorType = DW_DNN_TENSOR_TYPE_CPU;
            CHECK_DW_ERROR(dwDNNTensorStreamer_initialize(&ctx.dnnOutputStreamers[outputIdx],
                                                          &outputProps[outputIdx],
                                                          hostProps.tensorType, m_sdk));
        }
        
        ctx.outputsHost.reset(new dwDNNTensorHandle_t[m_numOutputTensors]);
        
        // Create per-camera data conditioner
        CHECK_DW_ERROR(dwDataConditioner_initializeFromTensorProperties(&ctx.dataConditioner, &inputProps, 1U,
                                                                       &metadata.dataConditionerParams, ctx.cudaStream,
                                                                       m_sdk));
        
        std::cout << "Camera context " << cameraIdx << " initialized" << std::endl;
    }
    
    void processAllCamerasParallel(dwCameraFrameHandle_t frames[])
    {
        // Start async inference on all idle cameras
        for (uint32_t i = 0; i < m_totalCameras; ++i)
        {
            CameraProcessingContext& ctx = m_cameraContexts[i];
            
            if (!ctx.inferenceInProgress.load())
            {
                if (startInferenceAsync(i, frames[i]))
                {
                    ctx.frameId.store(m_frameCount.load());
                    ctx.inferenceInProgress.store(true);
                }
            }
        }
        
        // Check for completed inferences
        for (uint32_t i = 0; i < m_totalCameras; ++i)
        {
            checkAndCollectResults(i);
        }
    }
    
    bool startInferenceAsync(uint32_t cameraIdx, dwCameraFrameHandle_t frame)
    {
        CameraProcessingContext& ctx = m_cameraContexts[cameraIdx];
        
        try
        {
            auto startTime = std::chrono::high_resolution_clock::now();
            
            // Prepare input
            prepareInputFrame(cameraIdx, frame, ctx);
            
            CHECK_CUDA_ERROR(cudaEventRecord(ctx.frameReadyEvent, ctx.cudaStream));
            
            // Run inference
            doInferenceAsync(cameraIdx, ctx);
            
            CHECK_CUDA_ERROR(cudaEventRecord(ctx.inferenceCompleteEvent, ctx.cudaStream));
            
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
    
    void checkAndCollectResults(uint32_t cameraIdx)
    {
        CameraProcessingContext& ctx = m_cameraContexts[cameraIdx];
        
        if (!ctx.inferenceInProgress.load())
            return;
        
        cudaError_t eventStatus = cudaEventQuery(ctx.inferenceCompleteEvent);
        
        if (eventStatus == cudaSuccess)
        {
            try
            {
                collectInferenceResults(cameraIdx, ctx);
                
                // Classify vehicles on detected cars
                classifyVehiclesAsync(cameraIdx, ctx);
                
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
                std::cout << "Error collecting results for camera " << cameraIdx << ": " << e.what() << std::endl;
                ctx.inferenceInProgress.store(false);
            }
        }
        else if (eventStatus != cudaErrorNotReady)
        {
            std::cout << "CUDA error in camera " << cameraIdx << ": " << cudaGetErrorString(eventStatus) << std::endl;
            ctx.inferenceInProgress.store(false);
        }
    }
    
    void prepareInputFrame(uint32_t cameraIdx, dwCameraFrameHandle_t frame, CameraProcessingContext& ctx)
    {
        dwImageHandle_t img = DW_NULL_HANDLE;
        CHECK_DW_ERROR(dwSensorCamera_getImage(&img, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, frame));
        CHECK_DW_ERROR(dwImage_copyConvert(m_imageRGBA[cameraIdx], img, m_sdk));
        
        CHECK_DW_ERROR(dwDataConditioner_prepareData(ctx.dnnInput, &m_imageRGBA[cameraIdx], 1, 
                                                     &m_detectionRegion[cameraIdx],
                                                     cudaAddressModeClamp, ctx.dataConditioner));
    }
    
    void doInferenceAsync(uint32_t cameraIdx, CameraProcessingContext& ctx)
    {
        CHECK_DW_ERROR(dwDNN_setCUDAStream(ctx.cudaStream, m_dnn));
        
        dwConstDNNTensorHandle_t inputs[1U] = {ctx.dnnInput};
        CHECK_DW_ERROR(dwDNN_infer(ctx.dnnOutputsDevice, m_numOutputTensors, inputs, 1U, m_dnn));
        
        for (uint32_t outputIdx = 0U; outputIdx < m_numOutputTensors; ++outputIdx)
        {
            dwDNNTensorStreamerHandle_t streamer = ctx.dnnOutputStreamers[outputIdx];
            CHECK_DW_ERROR(dwDNNTensorStreamer_producerSend(ctx.dnnOutputsDevice[outputIdx], streamer));
        }
    }
    
    void collectInferenceResults(uint32_t cameraIdx, CameraProcessingContext& ctx)
    {
        for (uint32_t outputIdx = 0U; outputIdx < m_numOutputTensors; ++outputIdx)
        {
            dwDNNTensorStreamerHandle_t streamer = ctx.dnnOutputStreamers[outputIdx];
            CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReceive(&ctx.outputsHost[outputIdx], 1000, streamer));
        }
        
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
        
        for (uint32_t outputIdx = 0U; outputIdx < m_numOutputTensors; ++outputIdx)
        {
            dwDNNTensorStreamerHandle_t streamer = ctx.dnnOutputStreamers[outputIdx];
            CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReturn(&ctx.outputsHost[outputIdx], streamer));
            CHECK_DW_ERROR(dwDNNTensorStreamer_producerReturn(nullptr, 1000, streamer));
        }
    }
    
    void classifyVehiclesAsync(uint32_t cameraIdx, CameraProcessingContext& ctx)
    {
        if (!m_enableVehicleClassification || !m_vtProcessor)
            return;
        
        auto classifyStart = std::chrono::high_resolution_clock::now();
        
        std::lock_guard<std::mutex> lock(ctx.resultsMutex);
        
        ctx.vehicleTypes.clear();
        ctx.vehicleDetectionIndices.clear();
        
        for (size_t i = 0; i < ctx.labels.size(); ++i)
        {
            if (ctx.labels[i] == "car" || ctx.labels[i] == "truck" || ctx.labels[i] == "bus")
            {
                const dwRectf& bbox = ctx.detectedBoxListFloat[i];
                
                dwRect cropRegion;
                cropRegion.x = std::max(0, static_cast<int32_t>(bbox.x));
                cropRegion.y = std::max(0, static_cast<int32_t>(bbox.y));
                cropRegion.width = std::min(static_cast<uint32_t>(bbox.width), m_imageWidth - cropRegion.x);
                cropRegion.height = std::min(static_cast<uint32_t>(bbox.height), m_imageHeight - cropRegion.y);
                
                if (cropRegion.width < 50 || cropRegion.height < 50)
                    continue;
                
                try {
                    // CRITICAL: Pass camera's CUDA stream to VehicleTypeNet
                    auto result = m_vtProcessor->classify(m_imageRGBA[cameraIdx], cropRegion, ctx.cudaStream);
                    ctx.vehicleTypes.push_back(result);
                    ctx.vehicleDetectionIndices.push_back(i);
                } catch (const std::exception& e) {
                    std::cerr << "Classification error for camera " << cameraIdx << ": " << e.what() << std::endl;
                }
            }
        }
        
        auto classifyEnd = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(classifyEnd - classifyStart);
        float classifyTimeMs = duration.count() / 1000.0f;
        
        ctx.avgClassificationTimeMs = (ctx.avgClassificationTimeMs * ctx.processedFrameCount + classifyTimeMs) / 
                                      (ctx.processedFrameCount + 1);
    }
    
    void renderCameraFrame(dwCameraFrameHandle_t frame, uint8_t cameraIndex)
    {
        dwImageHandle_t img = DW_NULL_HANDLE;
        dwCameraOutputType outputType = DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8;
        
        CHECK_DW_ERROR(dwRenderEngine_setTile(m_tileVideo[cameraIndex], m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_resetTile(m_renderEngine));
        
        CHECK_DW_ERROR(dwSensorCamera_getImage(&img, outputType, frame));
        
        CHECK_DW_ERROR(dwImageStreamerGL_producerSend(img, m_streamerToGL[cameraIndex]));
        
        dwImageHandle_t frameGL;
        CHECK_DW_ERROR(dwImageStreamerGL_consumerReceive(&frameGL, 33000, m_streamerToGL[cameraIndex]));
        
        dwImageGL* imageGL;
        CHECK_DW_ERROR(dwImage_getGL(&imageGL, frameGL));
        
        {
            dwVector2f range{};
            range.x = imageGL->prop.width;
            range.y = imageGL->prop.height;
            CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));
            CHECK_DW_ERROR(dwRenderEngine_renderImage2D(imageGL, {0, 0, range.x, range.y}, m_renderEngine));
            
            renderDetectionResults(cameraIndex);
            renderPerformanceStats(cameraIndex);
        }
        
        CHECK_DW_ERROR(dwImageStreamerGL_consumerReturn(&frameGL, m_streamerToGL[cameraIndex]));
        CHECK_DW_ERROR(dwImageStreamerGL_producerReturn(nullptr, 32000, m_streamerToGL[cameraIndex]));
    }
    
    void renderDetectionResults(uint32_t cameraIdx)
    {
        CameraProcessingContext& ctx = m_cameraContexts[cameraIdx];
        
        std::lock_guard<std::mutex> lock(ctx.resultsMutex);
        
        if (!ctx.detectedBoxListFloat.empty())
        {
            // Render detection boxes
            CHECK_DW_ERROR(dwRenderEngine_setColor({1.0f, 0.0f, 0.0f, 1.0f}, m_renderEngine));
            CHECK_DW_ERROR(dwRenderEngine_setLineWidth(2.0f, m_renderEngine));
            CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D,
                                                 ctx.detectedBoxListFloat.data(), sizeof(dwRectf), 0,
                                                 ctx.detectedBoxListFloat.size(), m_renderEngine));
            
            // Render YOLO labels
            if (ctx.labels.size() == ctx.detectedBoxListFloat.size())
            {
                CHECK_DW_ERROR(dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, m_renderEngine));
                CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_12, m_renderEngine));
                
                for (size_t i = 0; i < ctx.labels.size(); ++i)
                {
                    const dwRectf& box = ctx.detectedBoxListFloat[i];
                    dwVector2f labelPos = {box.x + 2, box.y - 5};
                    CHECK_DW_ERROR(dwRenderEngine_renderText2D(ctx.labels[i].c_str(), labelPos, m_renderEngine));
                }
            }
            
            // Render VehicleTypeNet classifications
            if (!ctx.vehicleTypes.empty())
            {
                CHECK_DW_ERROR(dwRenderEngine_setColor({0.0f, 1.0f, 0.0f, 1.0f}, m_renderEngine));
                CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_12, m_renderEngine));
                
                for (size_t i = 0; i < ctx.vehicleTypes.size(); ++i)
                {
                    if (ctx.vehicleTypes[i].valid && i < ctx.vehicleDetectionIndices.size())
                    {
                        size_t detectionIdx = ctx.vehicleDetectionIndices[i];
                        if (detectionIdx < ctx.detectedBoxListFloat.size())
                        {
                            const dwRectf& box = ctx.detectedBoxListFloat[detectionIdx];
                            std::string vtLabel = std::string(VehicleTypeNetProcessor::VEHICLE_TYPE_NAMES[ctx.vehicleTypes[i].classIndex]) +
                                                " " + std::to_string(static_cast<int>(ctx.vehicleTypes[i].confidence * 100)) + "%";
                            
                            dwVector2f vtLabelPos = {box.x + 2, box.y + box.height + 15};
                            CHECK_DW_ERROR(dwRenderEngine_renderText2D(vtLabel.c_str(), vtLabelPos, m_renderEngine));
                        }
                    }
                }
            }
        }
    }
    
    void renderPerformanceStats(uint32_t cameraIdx)
    {
        CameraProcessingContext& ctx = m_cameraContexts[cameraIdx];
        
        std::string statusStr = "Cam:" + std::to_string(cameraIdx);
        statusStr += " Det:" + std::to_string(ctx.detectedBoxListFloat.size());
        statusStr += " Veh:" + std::to_string(ctx.vehicleTypes.size());
        statusStr += " Inf:" + std::to_string(static_cast<int>(ctx.avgInferenceTimeMs)) + "ms";
        statusStr += " Cls:" + std::to_string(static_cast<int>(ctx.avgClassificationTimeMs)) + "ms";
        
        if (ctx.inferenceInProgress.load())
        {
            statusStr += " [PROCESSING]";
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
        CHECK_DW_ERROR(dwRenderEngine_renderText2D(statusStr.c_str(), {25, 80}, m_renderEngine));
    }
    
    void updatePerformanceStats()
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
    
    void printPerformanceMetrics()
    {
        std::cout << "=== Performance Metrics ===" << std::endl;
        std::cout << "Total Frame Count: " << m_frameCount.load() << std::endl;
        std::cout << "Inferences/sec (last 2s): " << m_totalInferencesPerSecond / 2 << std::endl;
        
        for (uint32_t i = 0; i < m_totalCameras; ++i)
        {
            CameraProcessingContext& ctx = m_cameraContexts[i];
            std::cout << "Camera " << i << ": "
                      << "Avg Inference: " << ctx.avgInferenceTimeMs << "ms, "
                      << "Avg Classification: " << ctx.avgClassificationTimeMs << "ms, "
                      << "Processed: " << ctx.processedFrameCount << " frames"
                      << std::endl;
        }
    }
    
    // YOLO interpretation methods
    void interpretOutput(dwDNNTensorHandle_t outConf, dwDNNTensorHandle_t outBBox, 
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
        
        size_t offsetY, strideY, height;
        uint32_t indices[4] = {0, 0, 0, 0};
        CHECK_DW_ERROR(dwDNNTensor_getLayoutView(&offsetY, &strideY, &height, indices, 4U, 1U, outConf));
        
        for (uint32_t gridY = 0U; gridY < height; ++gridY)
        {
            size_t offsetX, strideX, width;
            uint32_t subIndices[4] = {0, gridY, 0, 0};
            CHECK_DW_ERROR(dwDNNTensor_getLayoutView(&offsetX, &strideX, &width, subIndices, 4U, 0U, outConf));
            
            for (uint32_t gridX = 0U; gridX < width; gridX += strideX)
            {
                dwFloat16_t conf = confData[offsetX + gridX * strideX];
                if (conf > COVERAGE_THRESHOLD && numBBoxes < m_maxDetections)
                {
                    float32_t imageX = (float32_t)gridX * (float32_t)m_cellSize;
                    float32_t imageY = (float32_t)gridY * (float32_t)m_cellSize;
                    
                    size_t bboxOffset, bboxStride, numDimensions;
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
                    bbox.width = static_cast<int32_t>(std::round(bboxFloat.width));
                    bbox.height = static_cast<int32_t>(std::round(bboxFloat.height));
                    bbox.x = static_cast<int32_t>(std::round(bboxFloat.x));
                    bbox.y = static_cast<int32_t>(std::round(bboxFloat.y));
                    
                    ctx.detectedBoxList.push_back(bbox);
                    ctx.detectedBoxListFloat.push_back(bboxFloat);
                    numBBoxes++;
                }
            }
        }
        
        CHECK_DW_ERROR(dwDNNTensor_unlock(outConf));
        CHECK_DW_ERROR(dwDNNTensor_unlock(outBBox));
    }
    
    void interpretOutput(dwDNNTensorHandle_t outConf, const dwRect* const roi, 
                        uint32_t cameraIdx, CameraProcessingContext& ctx)
    {
        ctx.detectedBoxList.clear();
        ctx.detectedBoxListFloat.clear();
        ctx.labels.clear();
        
        uint32_t numBBoxes = 0U;
        
        void* tmpConf;
        CHECK_DW_ERROR(dwDNNTensor_lock(&tmpConf, outConf));
        float32_t* confData = reinterpret_cast<float32_t*>(tmpConf);
        
        size_t offsetY, strideY, height;
        uint32_t indices[4] = {0, 0, 0, 0};
        CHECK_DW_ERROR(dwDNNTensor_getLayoutView(&offsetY, &strideY, &height, indices, 4U, 1U, outConf));
        
        std::vector<YoloScoreRect> tmpRes;
        for (uint16_t gridY = 0U; gridY < height; ++gridY)
        {
            const float32_t* outConfRow = &confData[gridY * strideY];
            if (outConfRow[4] < CONFIDENCE_THRESHOLD || numBBoxes >= 100)
                continue;
            
            uint16_t maxIndex = 0;
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
                float32_t bboxW = (float32_t)outConfRow[2];
                float32_t bboxH = (float32_t)outConfRow[3];
                
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
        
        std::vector<YoloScoreRect> tmpResAfterNMS = doNmsForYoloOutputBoxes(tmpRes, float32_t(0.45));
        
        for (uint32_t i = 0; i < tmpResAfterNMS.size(); i++)
        {
            YoloScoreRect box = tmpResAfterNMS[i];
            
            if (m_automotiveClasses.find(YOLO_CLASS_NAMES[box.classIndex]) != m_automotiveClasses.end())
            {
                dwRectf bboxFloat = box.rectf;
                dwBox2D bbox;
                bbox.width = static_cast<int32_t>(std::round(bboxFloat.width));
                bbox.height = static_cast<int32_t>(std::round(bboxFloat.height));
                bbox.x = static_cast<int32_t>(std::round(bboxFloat.x));
                bbox.y = static_cast<int32_t>(std::round(bboxFloat.y));
                
                ctx.detectedBoxList.push_back(bbox);
                ctx.detectedBoxListFloat.push_back(bboxFloat);
                ctx.labels.push_back(YOLO_CLASS_NAMES[box.classIndex]);
            }
        }
    }
    
    // Utility methods
    std::string getPlatformPrefix()
    {
        static const int32_t CUDA_AMPERE_MAJOR_COMPUTE_CAPABILITY = 8;
        static const int32_t CUDA_TURING_VOLTA_MAJOR_COMPUTE_CAPABILITY = 7;
        static const int32_t CUDA_VOLTA_DISCRETE_MINOR_COMPUTE_CAPABILITY = 0;
        static const int32_t CUDA_VOLTA_INTEGRATED_MINOR_COMPUTE_CAPABILITY = 2;
        static const int32_t CUDA_TURING_DISCRETE_MINOR_COMPUTE_CAPABILITY = 5;
        
        std::string path;
        int32_t currentGPU;
        dwGPUDeviceProperties gpuProp{};
        
        CHECK_DW_ERROR(dwContext_getGPUDeviceCurrent(&currentGPU, m_sdk));
        CHECK_DW_ERROR(dwContext_getGPUProperties(&gpuProp, currentGPU, m_sdk));
        
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
    
    static bool sort_score(YoloScoreRect box1, YoloScoreRect box2)
    {
        return box1.score > box2.score;
    }
    
    float32_t calculateIouOfBoxes(dwRectf box1, dwRectf box2)
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
    
    std::vector<YoloScoreRect> doNmsForYoloOutputBoxes(std::vector<YoloScoreRect>& boxes, float32_t threshold)
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
};