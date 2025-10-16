/////////////////////////////////////////////////////////////////////////////////////////
// Drivable Area Segmentation - Implementation Following Reference Pattern
/////////////////////////////////////////////////////////////////////////////////////////

#include "DrivableAreaSegmentation.hpp"

#include <iostream>
#include <sstream>

DrivableAreaSegmentation::DrivableAreaSegmentation(const ProgramArguments& args)
    : DriveWorksSample(args)
{
    m_lastStatsTime = std::chrono::high_resolution_clock::now();
}

bool DrivableAreaSegmentation::onInitialize()
{
    initializeDriveWorks(m_context);
    CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_context));
    
    initializeCameraFromRig();
    initializeRenderer();
    initializeDNN();
    
    std::cout << "\n==================================" << std::endl;
    std::cout << "Initialization Complete" << std::endl;
    std::cout << "Camera: " << m_imageWidth << "x" << m_imageHeight << std::endl;
    std::cout << "==================================" << std::endl;
    
    return true;
}

void DrivableAreaSegmentation::initializeDriveWorks(dwContextHandle_t& context) const
{
    CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
    CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));

    dwContextParameters sdkParams = {};
#ifdef VIBRANTE
    sdkParams.eglDisplay = getEGLDisplay();
#endif
    CHECK_DW_ERROR(dwInitialize(&context, DW_VERSION, &sdkParams));
    
    std::cout << "✓ DriveWorks initialized" << std::endl;
}

void DrivableAreaSegmentation::initializeCameraFromRig()
{
    // Initialize rig configuration (following reference pattern)
    CHECK_DW_ERROR(dwRig_initializeFromFile(&m_rigConfig, m_context,
                                           getArgument("rig").c_str()));
    
    uint32_t cameraCount = 0;
    CHECK_DW_ERROR(dwRig_getSensorCountOfType(&cameraCount, DW_SENSOR_CAMERA, m_rigConfig));
    
    if (cameraCount == 0)
    {
        throw std::runtime_error("No cameras found in rig configuration");
    }
    
    std::cout << "Found " << cameraCount << " camera(s) in rig, using first camera" << std::endl;
    
    // Get first camera sensor parameters from rig
    uint32_t cameraSensorIdx = 0;
    CHECK_DW_ERROR(dwRig_findSensorByTypeIndex(&cameraSensorIdx, DW_SENSOR_CAMERA, 0, m_rigConfig));
    
    const char* protocol = nullptr;
    CHECK_DW_ERROR(dwRig_getSensorProtocol(&protocol, cameraSensorIdx, m_rigConfig));
    
    const char* params = nullptr;
    CHECK_DW_ERROR(dwRig_getSensorParameterUpdatedPath(&params, cameraSensorIdx, m_rigConfig));
    
    std::cout << "Initializing camera with protocol: " << protocol << std::endl;
    std::cout << "Camera parameters: " << params << std::endl;
    
    dwSensorParams sensorParams{};
    sensorParams.protocol = protocol;
    sensorParams.parameters = params;
    
    CHECK_DW_ERROR(dwSAL_createSensor(&m_camera, sensorParams, m_sal));
    
    // Get image properties
    dwImageProperties imageProperties{};
    CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&imageProperties, 
                                                     DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, 
                                                     m_camera));
    m_imageWidth = imageProperties.width;
    m_imageHeight = imageProperties.height;
    
    CHECK_DW_ERROR(dwImage_create(&m_imageRGBA, imageProperties, m_context));
    CHECK_DW_ERROR(dwImageStreamerGL_initialize(&m_streamerToGL, &imageProperties, 
                                               DW_IMAGE_GL, m_context));
    
    CHECK_DW_ERROR(dwSensor_start(m_camera));
    
    std::cout << "✓ Camera initialized: " << m_imageWidth << "x" << m_imageHeight << std::endl;
}

void DrivableAreaSegmentation::initializeRenderer()
{
    CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_context));
    
    dwRenderEngineParams params{};
    CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&params, getWindowWidth(), getWindowHeight()));
    params.defaultTile.lineWidth = 2.0f;
    params.defaultTile.font = DW_RENDER_ENGINE_FONT_VERDANA_20;
    params.maxBufferCount = 1;
    params.bounds.width = static_cast<float32_t>(getWindowWidth());
    params.bounds.height = static_cast<float32_t>(getWindowHeight());
    params.bounds.x = 0;
    params.bounds.y = 0;
    
    CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_viz));
    
    dwRenderEngineTileState tileState{};
    dwRenderEngine_initTileState(&tileState);
    tileState.modelViewMatrix = DW_IDENTITY_MATRIX4F;
    tileState.font = DW_RENDER_ENGINE_FONT_VERDANA_20;
    
    CHECK_DW_ERROR(dwRenderEngine_addTilesByCount(&m_tile, 1, 1, &tileState, m_renderEngine));
    
    std::cout << "✓ Renderer initialized" << std::endl;
}

void DrivableAreaSegmentation::initializeDNN()
{
    m_modelPath = getArgument("model");
    if (m_modelPath.empty())
    {
        m_modelPath = dw_samples::SamplesDataPath::get() + "/samples/detector/";
        m_modelPath += getPlatformPrefix();
        m_modelPath += "/resnet34_fcn_gpu_fp16.bin";
    }
    
    std::cout << "Loading model: " << m_modelPath << std::endl;
    
    // Create CUDA stream
    cudaError_t cudaStatus = cudaStreamCreate(&m_cudaStream);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream");
    }
    
    // Initialize DNN
    CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFile(&m_dnn, m_modelPath.c_str(), 
                                                    nullptr, DW_PROCESSOR_TYPE_GPU, 
                                                    m_context));
    CHECK_DW_ERROR(dwDNN_setCUDAStream(m_cudaStream, m_dnn));
    
    // Validate tensor properties
    validateTensorProperties();
    
    // Create tensors
    CHECK_DW_ERROR(dwDNNTensor_create(&m_dnnInput, &m_inputProps, m_context));
    CHECK_DW_ERROR(dwDNNTensor_create(&m_dnnOutput, &m_outputProps, m_context));
    
    // Initialize DataConditioner
    dwDNNMetaData metadata;
    CHECK_DW_ERROR(dwDNN_getMetaData(&metadata, m_dnn));
    
    std::cout << "Data conditioner parameters:" << std::endl;
    std::cout << "  Mean: [" << metadata.dataConditionerParams.meanValue[0] << ", "
              << metadata.dataConditionerParams.meanValue[1] << ", "
              << metadata.dataConditionerParams.meanValue[2] << "]" << std::endl;
    std::cout << "  Std:  [" << metadata.dataConditionerParams.stdev[0] << ", "
              << metadata.dataConditionerParams.stdev[1] << ", "
              << metadata.dataConditionerParams.stdev[2] << "]" << std::endl;
    
    CHECK_DW_ERROR(dwDataConditioner_initializeFromTensorProperties(&m_dataConditioner, 
                                                                   &m_inputProps, 1,
                                                                   &metadata.dataConditionerParams, 
                                                                   m_cudaStream, 
                                                                   m_context));
    
    // Setup ROI - full frame
    m_roi.x = 0;
    m_roi.y = 0;
    m_roi.width = static_cast<int32_t>(m_imageWidth);
    m_roi.height = static_cast<int32_t>(m_imageHeight);
    
    std::cout << "✓ DNN initialized" << std::endl;

    // ADD: Allocate colored mask buffer (RGBA)
    size_t maskSize = 1280 * 720 * 4;  // width * height * RGBA
    m_coloredMaskHost.reset(new uint8_t[maskSize]);
    memset(m_coloredMaskHost.get(), 0, maskSize);
    
    // ADD: Create overlay image for visualization
    dwImageProperties overlayProps{};
    overlayProps.type = DW_IMAGE_CUDA;
    overlayProps.format = DW_IMAGE_FORMAT_RGBA_UINT8;
    overlayProps.width = 1280;
    overlayProps.height = 720;
    CHECK_DW_ERROR(dwImage_create(&m_segmentationOverlay, overlayProps, m_context));
    CHECK_DW_ERROR(dwImageStreamerGL_initialize(&m_overlayStreamerToGL, &overlayProps, 
                                               DW_IMAGE_GL, m_context));

    uint32_t outputTensorSize = 1280 * 720 * 3;  // W * H * C
    m_outputLogitsHost.reset(new float[outputTensorSize]);

}

void DrivableAreaSegmentation::validateTensorProperties()
{
    // Get tensor properties
    CHECK_DW_ERROR(dwDNN_getInputTensorProperties(&m_inputProps, 0, m_dnn));
    CHECK_DW_ERROR(dwDNN_getOutputTensorProperties(&m_outputProps, 0, m_dnn));
    
    std::cout << "\nTensor Properties Validation:" << std::endl;
    std::cout << "Input tensor:" << std::endl;
    std::cout << "  Dimensions: " << m_inputProps.numDimensions << "D [";
    for (uint32_t i = 0; i < m_inputProps.numDimensions; ++i) {
        std::cout << m_inputProps.dimensionSize[i];
        if (i < m_inputProps.numDimensions - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Layout: " << (m_inputProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NCHW ? "NCHW" : 
                                  m_inputProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NHWC ? "NHWC" : "OTHER") << std::endl;
    std::cout << "  Data type: " << (m_inputProps.dataType == DW_TYPE_FLOAT32 ? "FP32" : 
                                     m_inputProps.dataType == DW_TYPE_FLOAT16 ? "FP16" : "OTHER") << std::endl;
    
    std::cout << "Output tensor:" << std::endl;
    std::cout << "  Dimensions: " << m_outputProps.numDimensions << "D [";
    for (uint32_t i = 0; i < m_outputProps.numDimensions; ++i) {
        std::cout << m_outputProps.dimensionSize[i];
        if (i < m_outputProps.numDimensions - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Layout: " << (m_outputProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NCHW ? "NCHW" : 
                                  m_outputProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NHWC ? "NHWC" : "OTHER") << std::endl;
    std::cout << "  Data type: " << (m_outputProps.dataType == DW_TYPE_FLOAT32 ? "FP32" : 
                                     m_outputProps.dataType == DW_TYPE_FLOAT16 ? "FP16" : "OTHER") << std::endl;
    
    // Validate dimensions
    if (m_inputProps.numDimensions != 4 || m_outputProps.numDimensions != 4)
    {
        throw std::runtime_error("Expected 4D tensors for segmentation model");
    }
    
    // Extract spatial dimensions based on layout (DriveWorks uses reverse order)
    uint32_t inputWidth, inputHeight, outputWidth, outputHeight;
    
    if (m_inputProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NCHW)
    {
        // Reverse order: [0]=W, [1]=H, [2]=C, [3]=N
        inputWidth = m_inputProps.dimensionSize[0];
        inputHeight = m_inputProps.dimensionSize[1];
    }
    else if (m_inputProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NHWC)
    {
        // Reverse order: [0]=C, [1]=W, [2]=H, [3]=N
        inputWidth = m_inputProps.dimensionSize[1];
        inputHeight = m_inputProps.dimensionSize[2];
    }
    else
    {
        throw std::runtime_error("Unsupported input tensor layout");
    }
    
    if (m_outputProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NCHW)
    {
        outputWidth = m_outputProps.dimensionSize[0];
        outputHeight = m_outputProps.dimensionSize[1];
        uint32_t numClasses = m_outputProps.dimensionSize[2];
        
        std::cout << "  Interpreted: [N=1, C=" << numClasses << ", H=" << outputHeight << ", W=" << outputWidth << "]" << std::endl;
        
        if (numClasses != NUM_SEGMENTATION_CLASSES)
        {
            std::cout << "WARNING: Expected " << NUM_SEGMENTATION_CLASSES << " classes, got " << numClasses << std::endl;
        }
    }
    else if (m_outputProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NHWC)
    {
        uint32_t numClasses = m_outputProps.dimensionSize[0];
        outputWidth = m_outputProps.dimensionSize[1];
        outputHeight = m_outputProps.dimensionSize[2];
        
        std::cout << "  Interpreted: [N=1, H=" << outputHeight << ", W=" << outputWidth << ", C=" << numClasses << "]" << std::endl;
        
        if (numClasses != NUM_SEGMENTATION_CLASSES)
        {
            std::cout << "WARNING: Expected " << NUM_SEGMENTATION_CLASSES << " classes, got " << numClasses << std::endl;
        }
    }
    
    std::cout << "Model resolution: " << inputWidth << "x" << inputHeight << std::endl;
    std::cout << "✓ Tensor properties validated" << std::endl;
}

void DrivableAreaSegmentation::onRender()
{
    CHECK_DW_ERROR(dwRenderEngine_setTile(m_tile, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_resetTile(m_renderEngine));
    
    dwCameraFrameHandle_t frame;
    dwStatus status = dwSensorCamera_readFrame(&frame, 33333, m_camera);
    
    if (status == DW_END_OF_STREAM)
    {
        dwSensor_reset(m_camera);
        return;
    }
    
    if (status == DW_SUCCESS)
    {
        processFrame(frame);
        onRenderHelper(frame);
        renderPerformanceStats();
        CHECK_DW_ERROR(dwSensorCamera_returnFrame(&frame));
        m_frameCount++;
    }
}

void DrivableAreaSegmentation::processFrame(dwCameraFrameHandle_t frame)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    // Preprocessing
    preprocessFrame(frame);
    
    // Inference
    runInference();
    
    // Postprocessing
    postprocessSegmentation();
    
    auto end = std::chrono::high_resolution_clock::now();
    float inferenceMs = std::chrono::duration<float, std::milli>(end - start).count();
    
    m_avgInferenceMs = (m_avgInferenceMs * m_processedFrameCount + inferenceMs) / 
                       (m_processedFrameCount + 1);
    m_processedFrameCount++;
}

void DrivableAreaSegmentation::preprocessFrame(dwCameraFrameHandle_t frame)
{
    // Get CUDA RGBA image
    dwImageHandle_t img = DW_NULL_HANDLE;
    CHECK_DW_ERROR(dwSensorCamera_getImage(&img, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, frame));
    CHECK_DW_ERROR(dwImage_copyConvert(m_imageRGBA, img, m_context));
    
    // Run data conditioner (resize + normalize + format conversion)
    CHECK_DW_ERROR(dwDataConditioner_prepareData(m_dnnInput, &m_imageRGBA, 1, 
                                                 &m_roi, cudaAddressModeClamp, 
                                                 m_dataConditioner));
}

void DrivableAreaSegmentation::runInference()
{
    // Run inference on GPU
    dwConstDNNTensorHandle_t inputs[1] = {m_dnnInput};
    CHECK_DW_ERROR(dwDNN_infer(&m_dnnOutput, 1, inputs, 1, m_dnn));
    
    // Synchronize stream
    cudaStreamSynchronize(m_cudaStream);
}

void DrivableAreaSegmentation::postprocessSegmentation()
{
    // Lock output tensor (GPU device memory)
    void* outputDataDevice = nullptr;
    dwStatus lockStatus = dwDNNTensor_lock(&outputDataDevice, m_dnnOutput);
    
    if (lockStatus != DW_SUCCESS || outputDataDevice == nullptr)
    {
        std::cerr << "Failed to lock output tensor" << std::endl;
        return;
    }
    
    // CRITICAL: Copy from device (GPU) to host (CPU)
    const uint32_t OUTPUT_SIZE = 1280 * 720 * 3;  // W * H * C
    cudaError_t cudaStatus = cudaMemcpy(
        m_outputLogitsHost.get(),
        outputDataDevice,
        OUTPUT_SIZE * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    
    CHECK_DW_ERROR(dwDNNTensor_unlock(m_dnnOutput));
    
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "Failed to copy output tensor to host: " 
                  << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }
    
    // Now process on CPU using host-accessible memory
    generateColoredMask(m_outputLogitsHost.get());
    
    // Upload colored mask to CUDA image
    dwImageCUDA* overlayImageCUDA;
    CHECK_DW_ERROR(dwImage_getCUDA(&overlayImageCUDA, m_segmentationOverlay));
    
    cudaStatus = cudaMemcpy2D(
        overlayImageCUDA->dptr[0],
        overlayImageCUDA->pitch[0],
        m_coloredMaskHost.get(),
        1280 * 4,  // source pitch
        1280 * 4,  // width in bytes
        720,       // height
        cudaMemcpyHostToDevice
    );
    
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "Failed to upload overlay: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
}

void DrivableAreaSegmentation::generateColoredMask(const float* outputLogits)
{
    // Color map (RGBA): [Background, Direct Drivable, Alternative Drivable]
    const uint8_t COLORS[3][4] = {
        {0, 0, 0, 0},        // Background - transparent
        {0, 255, 0, 180},    // Direct drivable - green (70% opacity)
        {255, 255, 0, 150}   // Alternative drivable - yellow (60% opacity)
    };
    
    const uint32_t WIDTH = 1280;
    const uint32_t HEIGHT = 720;
    const uint32_t NUM_CLASSES = 3;
    const uint32_t SPATIAL_SIZE = WIDTH * HEIGHT;
    
    // NCHW layout: outputLogits[c * SPATIAL_SIZE + h * WIDTH + w]
    for (uint32_t h = 0; h < HEIGHT; ++h)
    {
        for (uint32_t w = 0; w < WIDTH; ++w)
        {
            const uint32_t pixelIdx = h * WIDTH + w;
            
            // Argmax across channel dimension
            float maxVal = outputLogits[pixelIdx];  // Class 0
            uint8_t maxClass = 0;
            
            for (uint32_t c = 1; c < NUM_CLASSES; ++c)
            {
                float val = outputLogits[c * SPATIAL_SIZE + pixelIdx];
                if (val > maxVal)
                {
                    maxVal = val;
                    maxClass = c;
                }
            }
            
            // Write RGBA to output buffer
            const uint32_t outIdx = pixelIdx * 4;
            m_coloredMaskHost[outIdx + 0] = COLORS[maxClass][0];  // R
            m_coloredMaskHost[outIdx + 1] = COLORS[maxClass][1];  // G
            m_coloredMaskHost[outIdx + 2] = COLORS[maxClass][2];  // B
            m_coloredMaskHost[outIdx + 3] = COLORS[maxClass][3];  // A
        }
    }
}

void DrivableAreaSegmentation::onRenderHelper(dwCameraFrameHandle_t frame)
{
    dwImageHandle_t img = DW_NULL_HANDLE;
    CHECK_DW_ERROR(dwSensorCamera_getImage(&img, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, frame));
    
    CHECK_DW_ERROR(dwImageStreamerGL_producerSend(img, m_streamerToGL));
    
    dwImageHandle_t frameGL;
    CHECK_DW_ERROR(dwImageStreamerGL_consumerReceive(&frameGL, 33000, m_streamerToGL));
    
    dwImageGL* imageGL;
    CHECK_DW_ERROR(dwImage_getGL(&imageGL, frameGL));
    
    dwVector2f range{};
    range.x = static_cast<float32_t>(m_imageWidth);
    range.y = static_cast<float32_t>(m_imageHeight);
    CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_renderImage2D(imageGL, {0, 0, range.x, range.y}, m_renderEngine));
    
    CHECK_DW_ERROR(dwImageStreamerGL_consumerReturn(&frameGL, m_streamerToGL));
    CHECK_DW_ERROR(dwImageStreamerGL_producerReturn(nullptr, 32000, m_streamerToGL));
    
    // Render segmentation overlay
    CHECK_DW_ERROR(dwImageStreamerGL_producerSend(m_segmentationOverlay, m_overlayStreamerToGL));
    
    dwImageHandle_t overlayGL;
    CHECK_DW_ERROR(dwImageStreamerGL_consumerReceive(&overlayGL, 33000, m_overlayStreamerToGL));
    
    dwImageGL* overlayImageGL;
    CHECK_DW_ERROR(dwImage_getGL(&overlayImageGL, overlayGL));
    
    // Render overlay scaled to camera resolution (alpha blending automatic)
    dwRectf overlayRect = {0, 0, static_cast<float32_t>(m_imageWidth), static_cast<float32_t>(m_imageHeight)};
    
    // REMOVED: dwRenderEngine_setColorByValue - not needed, alpha blending is automatic
    CHECK_DW_ERROR(dwRenderEngine_renderImage2D(overlayImageGL, overlayRect, m_renderEngine));
    
    CHECK_DW_ERROR(dwImageStreamerGL_consumerReturn(&overlayGL, m_overlayStreamerToGL));
    CHECK_DW_ERROR(dwImageStreamerGL_producerReturn(nullptr, 32000, m_overlayStreamerToGL));
}

void DrivableAreaSegmentation::renderPerformanceStats()
{
    float fps = (m_avgInferenceMs > 0) ? (1000.0f / m_avgInferenceMs) : 0.0f;
    
    std::stringstream ss;
    ss << "FPS: " << static_cast<int>(fps) 
       << " | Inference: " << static_cast<int>(m_avgInferenceMs) << "ms"
       << " | Frame: " << m_frameCount.load();
    
    CHECK_DW_ERROR(dwRenderEngine_setColor({0.0f, 1.0f, 0.0f, 1.0f}, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_renderText2D(ss.str().c_str(), {25, 50}, m_renderEngine));
}

void DrivableAreaSegmentation::printPerformanceMetrics()
{
    float fps = (m_avgInferenceMs > 0) ? (1000.0f / m_avgInferenceMs) : 0.0f;
    
    std::cout << "\n=== Performance Metrics ===" << std::endl;
    std::cout << "Frames Processed: " << m_processedFrameCount << std::endl;
    std::cout << "Avg Inference: " << m_avgInferenceMs << " ms" << std::endl;
    std::cout << "Avg FPS: " << fps << std::endl;
    std::cout << "==========================" << std::endl;
}

void DrivableAreaSegmentation::onRelease()
{
    std::cout << "\nReleasing resources..." << std::endl;
    
    if (m_camera) dwSensor_stop(m_camera);
    if (m_dnnInput) dwDNNTensor_destroy(m_dnnInput);
    if (m_dnnOutput) dwDNNTensor_destroy(m_dnnOutput);
    if (m_dataConditioner) dwDataConditioner_release(m_dataConditioner);
    if (m_dnn) dwDNN_release(m_dnn);
    if (m_cudaStream) cudaStreamDestroy(m_cudaStream);
    if (m_streamerToGL) dwImageStreamerGL_release(m_streamerToGL);
    if (m_camera) dwSAL_releaseSensor(m_camera);
    if (m_imageRGBA) dwImage_destroy(m_imageRGBA);
    if (m_renderEngine) dwRenderEngine_release(m_renderEngine);
    if (m_viz) dwVisualizationRelease(m_viz);
    if (m_rigConfig) dwRig_release(m_rigConfig);
    if (m_sal) dwSAL_release(m_sal);
    if (m_context) dwRelease(m_context);
    if (m_overlayStreamerToGL) dwImageStreamerGL_release(m_overlayStreamerToGL);
    if (m_segmentationOverlay) dwImage_destroy(m_segmentationOverlay);
    
    std::cout << "✓ Resources released" << std::endl;
}

void DrivableAreaSegmentation::onKeyDown(int key, int scancode, int mods)
{
    (void)scancode;
    (void)mods;
    
    if (key == GLFW_KEY_P) {
        printPerformanceMetrics();
    }
}

std::string DrivableAreaSegmentation::getPlatformPrefix()
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