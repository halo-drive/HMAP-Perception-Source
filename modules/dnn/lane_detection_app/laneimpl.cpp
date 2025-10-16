#include "lanepomo.hpp"

// ============================================================================
// INITIALIZATION METHODS
// ============================================================================
static inline void dbg(const char* tag, const std::string& msg)
{
    std::cout << "[DBG-" << tag << "] " << msg << std::endl;
}

bool LaneDetectionApplication::onInitialize() {
    initDWAndSAL();
    initRender();
    
    if (!initializeEnhancedProcessingPipeline()) {
        return false;
    }
    
    bool ret = initSensors();
    if (!ret) return false;
    
    if (!initTensorRT()) {
        return false;
    }
    
    allocateProcessingBuffers();
    
    return true;
}

bool LaneDetectionApplication::initializeEnhancedProcessingPipeline() {
    // Initialize cuBLAS handle
    cublasStatus_t cublasStatus = cublasCreate(&m_cublasHandle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        logError("Failed to create cuBLAS handle: %d\n", cublasStatus);
        return false;
    }
    
    cublasStatus = cublasSetStream(m_cublasHandle, m_cudaStream);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        logError("Failed to set cuBLAS stream: %d\n", cublasStatus);
        return false;
    }
    
    // Initialize image transformation engine
    dwImageTransformationParameters transformParams{false};
    CHECK_DW_ERROR(dwImageTransformation_initialize(&m_imageTransformationEngine, 
                                                    transformParams, m_sdk));
    
    return true;
}

bool LaneDetectionApplication::initTensorRT() {
    std::string tensorRTModel = getArgument("tensorRT_model");
    if (tensorRTModel.empty()) {
        tensorRTModel = dw_samples::SamplesDataPath::get() + "/models/yolov8n-seg-lane-train2_fp16.plan";
    }
    
    std::cout << "Loading TensorRT engine: " << tensorRTModel << std::endl;
    
    if (!m_tensorrtEngine.loadEngine(tensorRTModel)) {
        logError("Failed to load TensorRT engine from: %s\n", tensorRTModel.c_str());
        return false;
    }
    
    // Validate engine properties
    nvinfer1::Dims inputDims = m_tensorrtEngine.getInputDims();
    int numOutputs = m_tensorrtEngine.getNumOutputs();
    
    std::cout << "Engine loaded successfully:" << std::endl;
    std::cout << "  Input dimensions: ";
    for (int i = 0; i < inputDims.nbDims; ++i) {
        std::cout << inputDims.d[i];
        if (i < inputDims.nbDims - 1) std::cout << "x";
    }
    std::cout << std::endl;
    std::cout << "  Number of outputs: " << numOutputs << std::endl;
    
    for (int i = 0; i < numOutputs; ++i) {
        nvinfer1::Dims outputDims = m_tensorrtEngine.getOutputDims(i);
        std::cout << "  Output " << i << " dimensions: ";
        for (int j = 0; j < outputDims.nbDims; ++j) {
            std::cout << outputDims.d[j];
            if (j < outputDims.nbDims - 1) std::cout << "x";
        }
        std::cout << std::endl;
    }
    
    // Validate expected input dimensions (should be 1x3x640x640 for YOLOv8)
    if (inputDims.nbDims != 4 || inputDims.d[0] != 1 || inputDims.d[1] != 3 ||
        inputDims.d[2] != INPUT_HEIGHT || inputDims.d[3] != INPUT_WIDTH) {
        logError("Unexpected input dimensions. Expected [1,3,640,640], got [%d,%d,%d,%d]\n",
                inputDims.d[0], inputDims.d[1], inputDims.d[2], inputDims.d[3]);
        return false;
    }
    
    if (numOutputs != 2) {
        logError("Expected 2 outputs (detection + segmentation), got %d\n", numOutputs);
        return false;
    }

    {
    std::ostringstream oss;
    oss << "TRT input dims = [";
    for (int i = 0; i < inputDims.nbDims; ++i)
        oss << inputDims.d[i] << (i < inputDims.nbDims - 1 ? "," : "");
    oss << "]";
    dbg("TRT", oss.str());
    }

    return true;
}

void LaneDetectionApplication::allocateProcessingBuffers() {
    std::cout << "\n=== BUFFER ALLOCATION START ===" << std::endl;
    
    // Set detection region to match input dimensions
    m_detectionRegionWidth = INPUT_WIDTH;
    m_detectionRegionHeight = INPUT_HEIGHT;
    
    const uint32_t detectionHW = m_detectionRegionWidth * m_detectionRegionHeight;
    
    std::cout << "Detection Region: " << m_detectionRegionWidth << "x" << m_detectionRegionHeight 
              << " (" << detectionHW << " pixels)" << std::endl;
    
    try {
        // Allocate input preprocessing buffer
        const uint32_t inputElements = 1 * 3 * INPUT_HEIGHT * INPUT_WIDTH; // NCHW format
        m_inputBuffer.allocate(inputElements);
        if (!m_inputBuffer.isValid()) {
            throw std::runtime_error("Failed to allocate input buffer");
        }
        
        // Allocate prototype processing buffers
        m_prototypeBuffer.allocate(detectionHW * NUM_MASK_PROTOTYPES);
        if (!m_prototypeBuffer.isValid()) {
            throw std::runtime_error("Failed to allocate prototype buffer");
        }
        
        m_maskCoeffsBuffer.allocate(m_maxDetections * NUM_MASK_PROTOTYPES);
        if (!m_maskCoeffsBuffer.isValid()) {
            throw std::runtime_error("Failed to allocate mask coefficients buffer");
        }
        
        m_linearMasksBuffer.allocate(m_maxDetections * detectionHW);
        if (!m_linearMasksBuffer.isValid()) {
            throw std::runtime_error("Failed to allocate linear masks buffer");
        }
        
        m_binaryMasksBuffer.allocate(m_maxDetections * detectionHW);
        m_combinedMaskBuffer.allocate(detectionHW);
        m_leftBoundsBuffer.allocate(m_detectionRegionHeight);
        m_rightBoundsBuffer.allocate(m_detectionRegionHeight);
        m_leftMaskBuffer.allocate(detectionHW);
        m_rightMaskBuffer.allocate(detectionHW);
        m_areaMaskBuffer.allocate(detectionHW);
        
        if (!m_binaryMasksBuffer.isValid() || !m_combinedMaskBuffer.isValid() ||
            !m_leftBoundsBuffer.isValid() || !m_rightBoundsBuffer.isValid() ||
            !m_leftMaskBuffer.isValid() || !m_rightMaskBuffer.isValid() ||
            !m_areaMaskBuffer.isValid()) {
            throw std::runtime_error("Failed to allocate lane processing buffers");
        }
        
        std::cout << "=== BUFFER ALLOCATION SUCCESS ===" << std::endl;
        
    } catch (const std::exception& e) {
        logError("Buffer allocation failed: %s\n", e.what());
        throw;
    }
}

void LaneDetectionApplication::initDWAndSAL() {
    CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
    CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));

    dwContextParameters sdkParams = {};

#ifdef VIBRANTE
    sdkParams.eglDisplay = getEGLDisplay();
#endif

    CHECK_DW_ERROR(dwInitialize(&m_sdk, DW_VERSION, &sdkParams));
    CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_sdk));
}

void LaneDetectionApplication::initRender() {
    CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_sdk));

    dwRenderEngineParams params{};
    CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&params, getWindowWidth(), getWindowHeight()));
    params.defaultTile.lineWidth = 0.2f;
    params.defaultTile.font = DW_RENDER_ENGINE_FONT_VERDANA_20;
    CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_viz));

    // --- lane-area R8 image that will be streamed to GL ------------------------
    {
    dwImageProperties p{};
    p.type   = DW_IMAGE_CUDA;
    p.format = DW_IMAGE_FORMAT_R_UINT8;   // 1 byte / pixel
    p.width  = INPUT_WIDTH;               // 640
    p.height = INPUT_HEIGHT;

    CHECK_DW_ERROR(dwImage_create(&m_laneAreaImg, p, m_sdk));
    CHECK_DW_ERROR(dwImage_getCUDA(&m_laneAreaCUDA, m_laneAreaImg));

    m_streamerCUDA2GL_Area.reset(new SimpleImageStreamerGL<>(p, 1000, m_sdk));
    }

}


bool LaneDetectionApplication::initSensors() {
#ifdef VIBRANTE
    m_useVirtualVideo = getArgument("input-type") == "video";
#else
    m_useVirtualVideo = true; 
#endif

    if (m_useVirtualVideo) {
        std::string videoPath = getArgument("video");
        if (videoPath.empty()) {
            logError("Video file path required\n");
            return false;
        }

        dwSensorParams sensorParams{};
        sensorParams.protocol = "camera.virtual";
        std::string paramStr = "video=" + videoPath;
        sensorParams.parameters = paramStr.c_str();

        std::cout << "Initializing video sensor with: " << paramStr << std::endl;

        CHECK_DW_ERROR(dwSAL_createSensor(&m_cameraSensor, sensorParams, m_sal));
        CHECK_DW_ERROR(dwSensor_start(m_cameraSensor));

        dwCameraProperties cameraProps;
        CHECK_DW_ERROR(dwSensorCamera_getSensorProperties(&cameraProps, m_cameraSensor));

        std::cout << "Video loaded with " << cameraProps.resolution.x << "x"
                  << cameraProps.resolution.y << " at " << cameraProps.framerate << " FPS" << std::endl;

        dwImageProperties displayProperties;
        CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&displayProperties, 
                                                        DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, 
                                                        m_cameraSensor));

        CHECK_DW_ERROR(dwImage_create(&m_imageRGBA, displayProperties, m_sdk));
        m_streamerCUDA2GL.reset(new SimpleImageStreamerGL<>(displayProperties, 1000, m_sdk));

        m_rcbProperties = displayProperties;
        m_imageWidth = displayProperties.width;
        m_imageHeight = displayProperties.height;

        // Set detection region to center of image
        m_detectionRegion.width = std::min(INPUT_WIDTH, m_imageWidth);
        m_detectionRegion.height = std::min(INPUT_HEIGHT, m_imageHeight);
        m_detectionRegion.x = (m_imageWidth - m_detectionRegion.width) / 2;
        m_detectionRegion.y = (m_imageHeight - m_detectionRegion.height) / 2;
        
        dbg("CAM",
        "video frame = " + std::to_string(m_imageWidth)  + " × "
                     + std::to_string(m_imageHeight));

        return true;
    } 
    else {
        std::string rigPath = getArgument("rig");
        if (rigPath.empty()) {
            logError("Rig configuration file path required\n");
            return false;
        }

        CHECK_DW_ERROR(dwRig_initializeFromFile(&m_rigConfig, m_sdk, rigPath.c_str()));

        uint32_t cameraCount = 0;
        CHECK_DW_ERROR(dwRig_getSensorCountOfType(&cameraCount, DW_SENSOR_CAMERA, m_rigConfig));
        
        if (cameraCount == 0) {
            logError("No cameras found in rig configuration\n");
            return false;
        }

        m_selectedCameraIndex = std::stoi(getArgument("camera-index"));
        if (m_selectedCameraIndex >= cameraCount) {
            logError("Camera index %d exceeds available cameras %d\n", m_selectedCameraIndex, cameraCount);
            return false;
        }

        uint32_t cameraSensorIdx = 0;
        CHECK_DW_ERROR(dwRig_findSensorByTypeIndex(&cameraSensorIdx, DW_SENSOR_CAMERA, 
                                                  m_selectedCameraIndex, m_rigConfig));

        const char* protocol = nullptr;
        const char* params = nullptr;
        CHECK_DW_ERROR(dwRig_getSensorProtocol(&protocol, cameraSensorIdx, m_rigConfig));
        CHECK_DW_ERROR(dwRig_getSensorParameterUpdatedPath(&params, cameraSensorIdx, m_rigConfig));

        dwSensorParams sensorParams{};
        sensorParams.protocol = protocol;
        sensorParams.parameters = params;

        std::cout << "Initializing camera with rig params: " << params << std::endl;

        CHECK_DW_ERROR(dwSAL_createSensor(&m_cameraSensor, sensorParams, m_sal));
        CHECK_DW_ERROR(dwSensor_start(m_cameraSensor));

        dwCameraProperties cameraProps;
        CHECK_DW_ERROR(dwSensorCamera_getSensorProperties(&cameraProps, m_cameraSensor));

        std::cout << "Camera image with " << cameraProps.resolution.x << "x"
                  << cameraProps.resolution.y << " at " << cameraProps.framerate << " FPS" << std::endl;

        dwImageProperties displayProperties;
        CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&displayProperties, 
                                                        DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, 
                                                        m_cameraSensor));

        CHECK_DW_ERROR(dwImage_create(&m_imageRGBA, displayProperties, m_sdk));
        m_streamerCUDA2GL.reset(new SimpleImageStreamerGL<>(displayProperties, 1000, m_sdk));

        m_rcbProperties = displayProperties;
        m_imageWidth = displayProperties.width;
        m_imageHeight = displayProperties.height;

        // Set detection region to center of image
        m_detectionRegion.width = std::min(INPUT_WIDTH, m_imageWidth);
        m_detectionRegion.height = std::min(INPUT_HEIGHT, m_imageHeight);
        m_detectionRegion.x = (m_imageWidth - m_detectionRegion.width) / 2;
        m_detectionRegion.y = (m_imageHeight - m_detectionRegion.height) / 2;
        
        dbg("ROI",
        "crop = [" + std::to_string(m_detectionRegion.x) + ", "
                   + std::to_string(m_detectionRegion.y) + "]  "
                   + std::to_string(m_detectionRegion.width)  + " × "
                   + std::to_string(m_detectionRegion.height));

        return true;
    }
}

// ============================================================================
// PROCESSING METHODS
// ============================================================================

void LaneDetectionApplication::onProcess() {
    prepareInputFrame();
    doInference();
    processInferenceResults();
}

void LaneDetectionApplication::prepareInputFrame() {
    dwImageCUDA* yuvImage = nullptr;
    getNextFrame(&yuvImage, &m_imgGl);
    std::this_thread::yield();
    while (yuvImage == nullptr) {
        if (should_AutoExit()) {
            log("AutoExit was set, stopping the sample because reached the end of the data stream\n");
            stop();
            return;
        }
        onReset();

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        getNextFrame(&yuvImage, &m_imgGl);
    }

    // Preprocess the image for TensorRT
    preprocessImage(yuvImage);
}

// ============================================================================
// PRE-PROCESS THE NEXT FRAME (crop → 640×640 RGB float tensor)
// ============================================================================

bool LaneDetectionApplication::preprocessImage(const dwImageCUDA* inputImage)
{
    if (!inputImage || !m_inputBuffer.isValid())
    {
        logError("Invalid input image or buffer\n");
        return false;
    }

    // ── 1. point the source pointer at the top-left pixel of the crop ─────────
    const uint8_t* srcRGBA =
        static_cast<const uint8_t*>(inputImage->dptr[0])
      + m_detectionRegion.y * inputImage->pitch[0]           // skip rows
      + m_detectionRegion.x * 4;                             // skip columns (RGBA → 4 bytes)

    // ── 2. TensorRT input buffer ──────────────────────────────────────────────
    void* tensorInput = m_tensorrtEngine.getInputBuffer();
    if (!tensorInput)
    {
        logError("Failed to get TensorRT input buffer\n");
        return false;
    }

    // ── 3. launch CUDA kernel (crop → planar RGB, NCHW, normalised) ───────────
    dbg("PRE",
    "launchPreprocessImageKernel( srcPitch = " + std::to_string(inputImage->pitch[0]) +
    ", dstPitch = " + std::to_string(INPUT_WIDTH * 3 * sizeof(float)) + " )");

    launchPreprocessImageKernel(
        srcRGBA,                                             // cropped start
        static_cast<float*>(tensorInput),
        INPUT_WIDTH, INPUT_HEIGHT,                           // always 640×640
        inputImage->pitch[0],                                // pitch of *full* frame
        INPUT_WIDTH * 3 * sizeof(float),                     // output pitch
        INPUT_SCALE,
        MEAN_R, MEAN_G, MEAN_B,
        STD_R,  STD_G,  STD_B,
        m_cudaStream);

    cudaError_t err = cudaStreamSynchronize(m_cudaStream);

    dbg("PRE",
    "tensor filled OK  →  feeding 640 × 640 to network");

    if (err != cudaSuccess)
    {
        logError("Pre-processing kernel failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}


bool LaneDetectionApplication::doInference() {
    // Execute TensorRT inference
    if (!m_tensorrtEngine.infer(m_cudaStream)) {
        logError("TensorRT inference failed\n");
        return false;
    }

    cudaError_t err = cudaStreamSynchronize(m_cudaStream);
    if (err != cudaSuccess) {
        logError("Inference stream synchronization failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    return true;
}

void LaneDetectionApplication::processInferenceResults() {
   // --- device pointers ----------------------------------------------------
    float* d_prototypes = static_cast<float*>(
        m_tensorrtEngine.getOutputBuffer(SEGMENTATION_OUTPUT_IDX));   // 0
    float* d_detections = static_cast<float*>(
        m_tensorrtEngine.getOutputBuffer(DETECTION_OUTPUT_IDX));      // 1

    if (!d_prototypes || !d_detections)
    {
        logError("Failed to get TensorRT output buffers\n");
        return;
    }

    // --- copy detection tensor to host -------------------------------------
    const size_t detBytes = m_tensorrtEngine.getOutputSize(DETECTION_OUTPUT_IDX);
    static std::vector<float> h_detections(detBytes / sizeof(float));   // reuse

    cudaError_t cpyErr = cudaMemcpyAsync(
        h_detections.data(),            // dst (host)
        d_detections,                   // src (device)
        detBytes,
        cudaMemcpyDeviceToHost,
        m_cudaStream);

    if (cpyErr != cudaSuccess)
    {
        logError("cudaMemcpy detections → host failed: %s\n",
                 cudaGetErrorString(cpyErr));
        return;
    }
    cudaStreamSynchronize(m_cudaStream);
    // Process the segmentation output
    processSegmentationOutputOptimized(h_detections.data(), d_prototypes);
}

void LaneDetectionApplication::processSegmentationOutputOptimized(const float* detectionOutput, 
                                                                 const float* segmentationOutput) {
    m_detectedBoxList.clear();
    m_detectedBoxListFloat.clear();
    
    std::vector<LaneDetection> validDetections;
    extractValidDetections(detectionOutput, validDetections);
    
    if (!validDetections.empty()) {
        processValidDetectionsOptimized(validDetections, segmentationOutput);
    }
    
    convertToRenderingFormat(validDetections);
}

void LaneDetectionApplication::extractValidDetections(const float32_t* detection, 
                                                     std::vector<LaneDetection>& validDetections) {
    validDetections.clear();
    validDetections.reserve(m_maxDetections);
    
    for (uint32_t i = 0; i < NUM_PREDICTIONS; ++i) {
        float32_t confidence = detection[4 * NUM_PREDICTIONS + i];
        
        if (confidence > CONFIDENCE_THRESHOLD) {
            LaneDetection det;
            
            float32_t center_x = detection[0 * NUM_PREDICTIONS + i];
            float32_t center_y = detection[1 * NUM_PREDICTIONS + i];
            float32_t width = detection[2 * NUM_PREDICTIONS + i];
            float32_t height = detection[3 * NUM_PREDICTIONS + i];
            
            det.bbox.x = center_x - width * 0.5f;
            det.bbox.y = center_y - height * 0.5f;
            det.bbox.width = width;
            det.bbox.height = height;
            det.confidence = confidence;
            
            for (uint32_t c = 0; c < NUM_MASK_PROTOTYPES; ++c) {
                det.maskCoeffs[c] = detection[(5 + c) * NUM_PREDICTIONS + i];
            }
            
            validDetections.push_back(det);
        }
    }
}

void LaneDetectionApplication::processValidDetectionsOptimized(const std::vector<LaneDetection>& validDetections,
                                                             const float32_t* prototypes) {
    const uint32_t N = static_cast<uint32_t>(validDetections.size());
    
    std::cout << "\n=== PROCESSING START ===" << std::endl;
    std::cout << "Input detections: " << N << std::endl;
    
    if (N == 0) {
        log("No valid detections found, skipping mask processing\n");
        return;
    }
    
    uint32_t processN = std::min(N, m_maxDetections);
    std::cout << "Processing detections: " << processN << std::endl;
    
    try {
        processValidDetectionsOptimizedImpl(validDetections, prototypes, processN);
        std::cout << "=== PROCESSING SUCCESS ===" << std::endl;
    } catch (const std::exception& e) {
        logError("Processing failed: %s\n", e.what());
        std::cout << "=== PROCESSING FAILED ===" << std::endl;
    }
}

void LaneDetectionApplication::processValidDetectionsOptimizedImpl(const std::vector<LaneDetection>& validDetections,
                                                                  const float32_t* prototypes,
                                                                  uint32_t N) {
    const uint32_t detectionHW = m_detectionRegionWidth * m_detectionRegionHeight;
    
    std::cout << "=== PROCESSING START ===" << std::endl;
    std::cout << "Detection region: " << m_detectionRegionWidth << "x" << m_detectionRegionHeight 
              << " = " << detectionHW << " pixels" << std::endl;
    
    // Create processing stream
    cudaStream_t kernelStream;
    cudaError_t streamErr = cudaStreamCreate(&kernelStream);
    if (streamErr != cudaSuccess) {
        logError("Failed to create kernel stream: %s\n", cudaGetErrorString(streamErr));
        return;
    }
    
    // Step 1: Resize prototypes from 160x160 to 640x640
    std::cout << "Step 1: Launching resize prototypes kernel..." << std::endl;
    
    launchResizePrototypesKernel(
        prototypes, m_prototypeBuffer.get(),
        NUM_MASK_PROTOTYPES,    // C=32
        PROTOTYPE_SIZE,         // H=160  
        PROTOTYPE_SIZE,         // W=160
        m_detectionRegionHeight,// H_det=640
        m_detectionRegionWidth, // W_det=640
        kernelStream
    );
    
    cudaError_t kernelErr = cudaStreamSynchronize(kernelStream);
    if (kernelErr != cudaSuccess) {
        logError("Kernel synchronization failed: %s\n", cudaGetErrorString(kernelErr));
        cudaStreamDestroy(kernelStream);
        return;
    }
    
    std::cout << "Prototype resize kernel completed successfully" << std::endl;
    
    // Step 2: Prepare mask coefficients
    std::cout << "Step 2: Preparing mask coefficients..." << std::endl;
    
    const size_t coeffsSize = N * NUM_MASK_PROTOTYPES;
    const size_t coeffsBytes = coeffsSize * sizeof(float32_t);
    
    std::vector<float32_t> hostCoeffs(coeffsSize);
    for (uint32_t i = 0; i < N; ++i) {
        const size_t destOffset = i * NUM_MASK_PROTOTYPES;
        std::memcpy(hostCoeffs.data() + destOffset, 
                   validDetections[i].maskCoeffs, 
                   NUM_MASK_PROTOTYPES * sizeof(float32_t));
    }
    
    cudaError_t copyErr = cudaMemcpy(m_maskCoeffsBuffer.get(), hostCoeffs.data(), 
                                    coeffsBytes, cudaMemcpyHostToDevice);
    if (copyErr != cudaSuccess) {
        logError("Memory copy failed: %s\n", cudaGetErrorString(copyErr));
        cudaStreamDestroy(kernelStream);
        return;
    }
    
    // Step 3: Execute cuBLAS GEMM
    std::cout << "Step 3: Executing cuBLAS GEMM..." << std::endl;
    
    cublasStatus_t cublasErr = cublasSetStream(m_cublasHandle, m_cudaStream);
    if (cublasErr != CUBLAS_STATUS_SUCCESS) {
        logError("Failed to set cuBLAS stream: %d\n", cublasErr);
        cudaStreamDestroy(kernelStream);
        return;
    }
    
    const float alpha = 1.0f, beta = 0.0f;
    
    cublasStatus_t status = cublasSgemm(
        m_cublasHandle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, detectionHW, NUM_MASK_PROTOTYPES,
        &alpha,
        m_maskCoeffsBuffer.get(), NUM_MASK_PROTOTYPES,
        m_prototypeBuffer.get(), NUM_MASK_PROTOTYPES,
        &beta,
        m_linearMasksBuffer.get(), N
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        logError("cuBLAS GEMM operation failed: %d\n", status);
        cudaStreamDestroy(kernelStream);
        return;
    }
    
    cudaError_t syncErr = cudaStreamSynchronize(m_cudaStream);
    if (syncErr != cudaSuccess) {
        logError("cuBLAS stream synchronization failed: %s\n", cudaGetErrorString(syncErr));
        cudaStreamDestroy(kernelStream);
        return;
    }
    
    std::cout << "cuBLAS GEMM completed successfully" << std::endl;
    
    // Step 4: Complete processing pipeline
    std::cout << "Step 4: Completing processing pipeline..." << std::endl;
    
    launchSigmoidThresholdKernel(
        m_linearMasksBuffer.get(), m_binaryMasksBuffer.get(),
        N, detectionHW, SIGMOID_THRESHOLD, kernelStream
    );
    
    launchOrReduceMasksKernel(
        m_binaryMasksBuffer.get(), m_combinedMaskBuffer.get(),
        N, m_detectionRegionHeight, m_detectionRegionWidth, kernelStream
    );
    
    launchRowBoundaryDetectionKernel(
        m_combinedMaskBuffer.get(), m_leftBoundsBuffer.get(), m_rightBoundsBuffer.get(),
        m_detectionRegionHeight, m_detectionRegionWidth, kernelStream
    );
    
    launchBuildLaneMasksKernel(
        m_leftBoundsBuffer.get(), m_rightBoundsBuffer.get(),
        m_leftMaskBuffer.get(), m_rightMaskBuffer.get(), m_areaMaskBuffer.get(),
        m_detectionRegionHeight, m_detectionRegionWidth, kernelStream
    );
    
    kernelErr = cudaStreamSynchronize(kernelStream);
    if (kernelErr != cudaSuccess) {
        logError("Final pipeline synchronization failed: %s\n", cudaGetErrorString(kernelErr));
        cudaStreamDestroy(kernelStream);
        return;
    }
    
    if (m_areaMaskBuffer.isValid() && m_laneAreaCUDA) {
        cudaMemcpy2DAsync(
        /*dst*/ m_laneAreaCUDA->dptr[0], m_laneAreaCUDA->pitch[0],
        /*src*/ m_areaMaskBuffer.get(),  m_detectionRegionWidth,   // 640 bytes/row
        m_detectionRegionWidth, m_detectionRegionHeight,          // w , h
        cudaMemcpyDeviceToDevice, kernelStream);
    }

    cudaStreamDestroy(kernelStream);
    std::cout << "=== PROCESSING COMPLETED SUCCESSFULLY ===" << std::endl;
}

// ============================================================================
// CONVERT DETECTIONS TO RENDER-ENGINE FORMAT
// ============================================================================

void LaneDetectionApplication::convertToRenderingFormat(
        const std::vector<LaneDetection>& detections)
{
    m_detectedBoxList.clear();
    m_detectedBoxListFloat.clear();
    m_label.clear();

    const float32_t scaleX = static_cast<float32_t>(m_imageWidth)  / INPUT_WIDTH;
    const float32_t scaleY = static_cast<float32_t>(m_imageHeight) / INPUT_HEIGHT;

    for (const auto& det : detections)
    {
        // map 640×640 → full frame, **including crop offset**
        dwRectf bboxF;
        bboxF.x      = det.bbox.x      * scaleX + m_detectionRegion.x;
        bboxF.y      = det.bbox.y      * scaleY + m_detectionRegion.y;
        bboxF.width  = det.bbox.width  * scaleX;
        bboxF.height = det.bbox.height * scaleY;

        dwBox2D bboxI;
        bboxI.x      = static_cast<int32_t>(std::round(bboxF.x));
        bboxI.y      = static_cast<int32_t>(std::round(bboxF.y));
        bboxI.width  = static_cast<uint32_t>(std::round(bboxF.width));
        bboxI.height = static_cast<uint32_t>(std::round(bboxF.height));

        m_detectedBoxList.push_back(bboxI);
        m_detectedBoxListFloat.push_back(bboxF);
        m_label.push_back("lane");
    }
}




void LaneDetectionApplication::getNextFrame(dwImageCUDA** nextFrameCUDA, dwImageGL** nextFrameGL) {
    if (m_currentFrame != DW_NULL_HANDLE) {
        dwSensorCamera_returnFrame(&m_currentFrame);
        m_currentFrame = DW_NULL_HANDLE;
    }

    dwStatus status = dwSensorCamera_readFrame(&m_currentFrame, 33000, m_cameraSensor);
    
    if (status == DW_NOT_READY) {
        *nextFrameCUDA = nullptr;
        *nextFrameGL = nullptr;
        return;
    }
    
    if (status == DW_END_OF_STREAM) {
        if (m_useVirtualVideo) {
            dwSensor_reset(m_cameraSensor);
            std::cout << "Video reached end of stream, resetting" << std::endl;
        }
        *nextFrameCUDA = nullptr;
        *nextFrameGL = nullptr;
        return;
    }
    
    if (status == DW_TIME_OUT) {
        logError("Timeout waiting for frame\n");
        *nextFrameCUDA = nullptr;
        *nextFrameGL = nullptr;
        return;
    }
    
    CHECK_DW_ERROR(status);

    dwImageHandle_t imageHandle;
    CHECK_DW_ERROR(dwSensorCamera_getImage(&imageHandle, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, m_currentFrame));
    
    CHECK_DW_ERROR(dwImage_getCUDA(nextFrameCUDA, imageHandle));
    
    CHECK_DW_ERROR(dwImage_copyConvert(m_imageRGBA, imageHandle, m_sdk));
    
    dwImageHandle_t frameGL = m_streamerCUDA2GL->post(m_imageRGBA);
    dwImage_getGL(nextFrameGL, frameGL);
}

// ============================================================================
// RENDERING METHODS
// ============================================================================

// ────────────────────────────────────────────────────────────────────────────
// helper – draw one boundary, automatically breaking at gaps
// ────────────────────────────────────────────────────────────────────────────
void LaneDetectionApplication::renderBoundaryStrip(const int32_t* bounds,
                                                   uint32_t        height,
                                                   float           roiX,
                                                   float           roiY,
                                                   dwRenderEngineColorRGBA colour)
{
    std::vector<dwVector2f> strip;
    strip.reserve(height);

    auto flush = [&]()
    {
        if (strip.size() >= 2)
        {
            CHECK_DW_ERROR(dwRenderEngine_setColor(colour, m_renderEngine));
            CHECK_DW_ERROR(dwRenderEngine_setLineWidth(2.f, m_renderEngine));
            CHECK_DW_ERROR(dwRenderEngine_render(
                DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINESTRIP_2D,
                strip.data(), sizeof(dwVector2f), 0,
                strip.size(),  m_renderEngine));
        }
        strip.clear();
    };

    for (uint32_t y = 0; y < height; ++y)
    {
        const int32_t x = bounds[y];

        /* valid pixel?  – sentinel produced by rowMinMaxKernel is  W (==width) */
        if (x >= 0 && x < static_cast<int32_t>(m_detectionRegionWidth))
        {
            strip.push_back({ roiX + static_cast<float>(x),
                              roiY + static_cast<float>(y) });
        }
        else
        {   /* gap → finish the current segment */
            flush();
        }
    }
    flush();                   // last segment (if any)
}




void LaneDetectionApplication::onRender()
{   
    dwRectf roiF{static_cast<float>(m_detectionRegion.x),
                 static_cast<float>(m_detectionRegion.y),
                 static_cast<float>(m_detectionRegion.width),
                 static_cast<float>(m_detectionRegion.height)};

    dbg("REN",
        "GL frame = "  + std::to_string(m_imgGl->prop.width)  + " × "
                       + std::to_string(m_imgGl->prop.height) +
        " | ROI = ["   + std::to_string(roiF.x)  + "," + std::to_string(roiF.y) +
        "] "           + std::to_string(roiF.width) + " × "
                       + std::to_string(roiF.height));

    // ─── 0. Camera frame ──────────────────────────────────────────────
    CHECK_DW_ERROR(dwRenderEngine_setTile(0, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_resetTile(m_renderEngine));

    dwVector2f range{static_cast<float>(m_imgGl->prop.width),
                     static_cast<float>(m_imgGl->prop.height)};
    CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_renderImage2D(
        m_imgGl, {0.f, 0.f, range.x, range.y}, m_renderEngine));

    // ─── 0·bis  green translucent drivable-area overlay ──────────────────────
    if (m_laneAreaImg)
    {
        dwImageHandle_t maskHandle = m_streamerCUDA2GL_Area->post(m_laneAreaImg);

        dwImageGL* maskGL = nullptr;
        dwImage_getGL(&maskGL, maskHandle);

        // 40 %-opaque green
        dwRenderEngineColorRGBA green40{0.f, 1.f, 0.f, 0.40f};
        CHECK_DW_ERROR(dwRenderEngine_setColor(green40, m_renderEngine));

        dwRectf dst{
        static_cast<float>(m_detectionRegion.x),
        static_cast<float>(m_detectionRegion.y),
        static_cast<float>(m_detectionRegion.width),
        static_cast<float>(m_detectionRegion.height)
        };
        
        CHECK_DW_ERROR(dwRenderEngine_renderImage2D(
            maskGL, dst, m_renderEngine));
    }

    // ─── 1. Yellow ROI box ────────────────────────────────────────────
    CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_YELLOW,
                                           m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_render(
        DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D,
        &roiF, sizeof(dwRectf), 0, 1, m_renderEngine));

    // ─── 2. Lane boundaries (green / cyan, gap-safe) ───────────────────────────
    if (m_leftBoundsBuffer.isValid() && m_rightBoundsBuffer.isValid())
    {
        static std::vector<int32_t> hLeft(m_detectionRegionHeight);
        static std::vector<int32_t> hRight(m_detectionRegionHeight);

        cudaMemcpy(hLeft.data(),  m_leftBoundsBuffer.get(),
                hLeft.size()*sizeof(int32_t),  cudaMemcpyDeviceToHost);
        cudaMemcpy(hRight.data(), m_rightBoundsBuffer.get(),
                hRight.size()*sizeof(int32_t), cudaMemcpyDeviceToHost);

        const float roiX = static_cast<float>(m_detectionRegion.x);
        const float roiY = static_cast<float>(m_detectionRegion.y);

        renderBoundaryStrip(hLeft.data(),  m_detectionRegionHeight,
                            roiX, roiY, DW_RENDER_ENGINE_COLOR_GREEN);

        renderBoundaryStrip(hRight.data(), m_detectionRegionHeight,
                            roiX, roiY, DW_RENDER_ENGINE_COLOR_CYAN);
    }

    // ─── 3. Red YOLO boxes ────────────────────────────────────────────
    if (!m_detectedBoxListFloat.empty())
    {
        CHECK_DW_ERROR(dwRenderEngine_setColor(
            DW_RENDER_ENGINE_COLOR_RED, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_setLineWidth(2.f, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_render(
            DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D,
            m_detectedBoxListFloat.data(), sizeof(dwRectf), 0,
            m_detectedBoxListFloat.size(), m_renderEngine));
    }
}


// ============================================================================
// CLEANUP METHODS
// ============================================================================

void LaneDetectionApplication::onRelease() {
    if (m_cublasHandle) {
        cublasDestroy(m_cublasHandle);
        m_cublasHandle = nullptr;
    }
    if (m_streamerCUDA2GL_Area) {
    m_streamerCUDA2GL_Area.reset();
    }
    if (m_laneAreaImg != DW_NULL_HANDLE) {
        CHECK_DW_ERROR(dwImage_destroy(m_laneAreaImg));
        m_laneAreaImg = DW_NULL_HANDLE;
        m_laneAreaCUDA = nullptr;
    }
    if (m_imageTransformationEngine != DW_NULL_HANDLE) {
        dwImageTransformation_release(m_imageTransformationEngine);
    }

    if (m_currentFrame != DW_NULL_HANDLE) {
        dwSensorCamera_returnFrame(&m_currentFrame);
        m_currentFrame = DW_NULL_HANDLE;
    }
    
    if (m_cameraSensor != DW_NULL_HANDLE) {
        dwSensor_stop(m_cameraSensor);
        dwSAL_releaseSensor(m_cameraSensor);
    }
    
    if (m_rigConfig != DW_NULL_HANDLE && !m_useVirtualVideo) {
        dwRig_release(m_rigConfig);
    }

    releaseImage();
    releaseTensorRT();
    releaseRender();
    releaseDWAndSAL();
}

void LaneDetectionApplication::onReset() {
    m_tensorrtEngine.destroy();
    if (!initTensorRT()) {
        logError("Failed to reinitialize TensorRT on reset\n");
    }

    if (m_cameraSensor != DW_NULL_HANDLE) {
        CHECK_DW_ERROR(dwSensor_reset(m_cameraSensor));
    }
}

void LaneDetectionApplication::onResizeWindow(int width, int height) {
    CHECK_DW_ERROR(dwRenderEngine_reset(m_renderEngine));
    dwRectf rect;
    rect.width = width;
    rect.height = height;
    rect.x = 0;
    rect.y = 0;
    CHECK_DW_ERROR(dwRenderEngine_setBounds(rect, m_renderEngine));
}

void LaneDetectionApplication::releaseRender() {
    CHECK_DW_ERROR(dwRenderEngine_release(m_renderEngine));
    CHECK_DW_ERROR(dwVisualizationRelease(m_viz));
}

void LaneDetectionApplication::releaseDWAndSAL() {
    CHECK_DW_ERROR(dwSAL_release(m_sal));
    CHECK_DW_ERROR(dwRelease(m_sdk));
}

void LaneDetectionApplication::releaseTensorRT() {
    m_tensorrtEngine.destroy();
}

void LaneDetectionApplication::releaseImage() {
    m_streamerCUDA2GL.reset();
    if (m_imageRGBA) {
        CHECK_DW_ERROR(dwImage_destroy(m_imageRGBA));
    }
}

// ============================================================================
// UTILITY METHODS
// ============================================================================

bool LaneDetectionApplication::sort_score(YoloScoreRect box1, YoloScoreRect box2) {
    return box1.score > box2.score ? true : false;
}

float32_t LaneDetectionApplication::calculateIouOfBoxes(dwRectf box1, dwRectf box2) {
    float32_t x1 = std::max(box1.x, box2.x);
    float32_t y1 = std::max(box1.y, box2.y);
    float32_t x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float32_t y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    float32_t w = std::max(0.0f, x2 - x1);
    float32_t h = std::max(0.0f, y2 - y1);
    float32_t over_area = w * h;
    return float32_t(over_area) / float32_t(box1.width * box1.height + box2.width * box2.height - over_area);
}

std::vector<LaneDetectionApplication::YoloScoreRect> 
LaneDetectionApplication::doNmsForYoloOutputBoxes(std::vector<YoloScoreRect>& boxes, float32_t threshold) {
    std::vector<YoloScoreRect> results;
    std::sort(boxes.begin(), boxes.end(), sort_score);
    while (boxes.size() > 0) {
        results.push_back(boxes[0]);
        uint32_t index = 1;
        while (index < boxes.size()) {
            float32_t iou_value = calculateIouOfBoxes(boxes[0].rectf, boxes[index].rectf);
            if (iou_value > threshold) {
                boxes.erase(boxes.begin() + index);
            } else {
                index++;
            }
        }
        boxes.erase(boxes.begin());
    }
    return results;
}

float32_t LaneDetectionApplication::overlap(const dwRectf& boxA, const dwRectf& boxB) {
    int32_t overlapWidth = std::min(boxA.x + boxA.width, boxB.x + boxB.width) - std::max(boxA.x, boxB.x);
    int32_t overlapHeight = std::min(boxA.y + boxA.height, boxB.y + boxB.height) - std::max(boxA.y, boxB.y);
    return (overlapWidth < 0 || overlapHeight < 0) ? 0.0f : (overlapWidth * overlapHeight);
}

std::string LaneDetectionApplication::getPlatformPrefix() {
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

    if (gpuProp.major == CUDA_AMPERE_MAJOR_COMPUTE_CAPABILITY) {
        if (gpuProp.integrated) {
            path = "ampere-integrated";
        } else {
            path = "ampere-discrete";
        }
    } else if (gpuProp.major == CUDA_TURING_VOLTA_MAJOR_COMPUTE_CAPABILITY) {
        if (gpuProp.minor == CUDA_TURING_DISCRETE_MINOR_COMPUTE_CAPABILITY) {
            path = "turing";
        } else if (gpuProp.minor == CUDA_VOLTA_INTEGRATED_MINOR_COMPUTE_CAPABILITY) {
            path = "volta-integrated";
        } else if (gpuProp.minor == CUDA_VOLTA_DISCRETE_MINOR_COMPUTE_CAPABILITY) {
            path = "volta-discrete";
        }
    } else {
        path = "pascal";
    }

    return path;
}