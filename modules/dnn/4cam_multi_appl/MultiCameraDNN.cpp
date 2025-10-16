#include "MultiCameraDNN.hpp"

//#######################################################################################
// CONSTRUCTOR & FRAMEWORK INTERFACE
//#######################################################################################

MultiCameraDNNApp::MultiCameraDNNApp(const ProgramArguments& args)
    : DriveWorksSample(args)
{
    // Initialize color coding per port for visual identification
    m_colorPerPort[0] = {1.0f, 0.0f, 0.0f, 1.0f}; // Red
    m_colorPerPort[1] = {0.0f, 1.0f, 0.0f, 1.0f}; // Green
    m_colorPerPort[2] = {0.0f, 0.0f, 1.0f, 1.0f}; // Blue
    m_colorPerPort[3] = {1.0f, 1.0f, 0.0f, 1.0f}; // Yellow
}

bool MultiCameraDNNApp::onInitialize()
{
    try {
        // Initialize DriveWorks framework components
        initializeDriveWorks();
        
        // Initialize multi-camera system
        initializeMultiCamera();
        
        // Start sensor abstraction layer
        CHECK_DW_ERROR(dwSAL_start(m_sal));
        
        // Initialize rendering system
        initializeRenderer();
        
        // Initialize DNN processor
        initializeDNNProcessor();
        
        // Initialize screenshot utility
        m_screenshot.reset(new ScreenshotHelper(m_context, m_sal, getWindowWidth(), 
                                               getWindowHeight(), "MultiCameraDNN"));
        
        // Start all cameras
        log("Starting %u cameras...\n", m_totalCameras);
        for (uint32_t i = 0; i < m_totalCameras; ++i) {
            CHECK_DW_ERROR(dwSensor_start(m_camera[i]));
        }
        
        logCameraStatus();
        logDNNStatus();
        return true;
        
    } catch (const std::exception& e) {
        log("Initialization failed: %s\n", e.what());
        return false;
    }
}

void MultiCameraDNNApp::onProcess()
{
    // Process screenshot trigger
    m_screenshot->processScreenshotTrig();
}

void MultiCameraDNNApp::onRender()
{
    dwCameraFrameHandle_t frames[MAX_CAMERAS];
    
    try {
        // Acquire frames from all cameras
        acquireFrames(frames);
        
        // Process current camera for DNN (round-robin)
        processCurrentCamera(frames[m_currentCamera]);
        
        // Render all camera frames
        for (uint32_t i = 0; i < m_totalCameras; ++i) {
            if (m_useProcessed && m_enableRender[i]) {
                renderCameraFrame(frames[i], i);
            }
        }
        
        // Return frames to sensor system
        returnFrames(frames);
        
        // Advance to next camera for DNN processing
        advanceCurrentCamera();
        m_frameCount++;
        log("Render state: useProcessed=%s, enableRender=[%s,%s,%s,%s]\n",
            m_useProcessed ? "true" : "false",
            m_enableRender[0] ? "T" : "F", m_enableRender[1] ? "T" : "F",
            m_enableRender[2] ? "T" : "F", m_enableRender[3] ? "T" : "F");
        
    } catch (const std::exception& e) {
        log("Render cycle failed: %s\n", e.what());
        // Ensure frames are returned even on error
        for (uint32_t i = 0; i < m_totalCameras; ++i) {
            if (frames[i] != DW_NULL_HANDLE) {
                dwSensorCamera_returnFrame(&frames[i]);
            }
        }
    }
}

void MultiCameraDNNApp::onRelease()
{
    // Stop cameras first
    for (uint32_t i = 0; i < m_totalCameras; ++i) {
        if (m_camera[i] != DW_NULL_HANDLE) {
            CHECK_DW_ERROR(dwSensor_stop(m_camera[i]));
        }
    }
    
    // Release image streamers and buffers
    for (uint32_t i = 0; i < m_totalCameras; ++i) {
        if (m_streamerToGL[i] != DW_NULL_HANDLE) {
            CHECK_DW_ERROR(dwImageStreamerGL_release(m_streamerToGL[i]));
        }
        if (m_imageRGBA[i] != DW_NULL_HANDLE) {
            CHECK_DW_ERROR(dwImage_destroy(m_imageRGBA[i]));
        }
        if (m_camera[i] != DW_NULL_HANDLE) {
            CHECK_DW_ERROR(dwSAL_releaseSensor(m_camera[i]));
        }
    }
    
    // Release screenshot helper
    m_screenshot.reset();
    
    // Release DNN processor
    if (m_dnnProcessor) {
        m_dnnProcessor.reset();
        log("DNN processor released\n");
    }
    
    // Release core DriveWorks components in reverse order
    if (m_rigConfig != DW_NULL_HANDLE) {
        CHECK_DW_ERROR(dwRig_release(m_rigConfig));
    }
    if (m_sal != DW_NULL_HANDLE) {
        CHECK_DW_ERROR(dwSAL_release(m_sal));
    }
    if (m_renderEngine != DW_NULL_HANDLE) {
        CHECK_DW_ERROR(dwRenderEngine_release(m_renderEngine));
    }
    if (m_viz != DW_NULL_HANDLE) {
        CHECK_DW_ERROR(dwVisualizationRelease(m_viz));
    }
    if (m_context != DW_NULL_HANDLE) {
        CHECK_DW_ERROR(dwRelease(m_context));
    }
}

void MultiCameraDNNApp::onKeyDown(int key, int scancode, int mods)
{
    (void)scancode;
    (void)mods;
    
    switch (key) {
        case GLFW_KEY_S:
            m_screenshot->triggerScreenshot();
            break;
        case GLFW_KEY_SPACE:
            log("Current processing camera: %u, Frame count: %u\n", m_currentCamera, m_frameCount);
            break;
        default:
            DriveWorksSample::onKeyDown(key, scancode, mods);
            break;
    }
}

//#######################################################################################
// INITIALIZATION METHODS
//#######################################################################################

void MultiCameraDNNApp::initializeDriveWorks()
{
    // Initialize logger with verbose output
    CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
    CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));
    
    // Setup context parameters
    dwContextParameters sdkParams = {};
#ifdef VIBRANTE
    sdkParams.eglDisplay = getEGLDisplay();
#endif
    
    // Initialize DriveWorks context
    CHECK_DW_ERROR(dwInitialize(&m_context, DW_VERSION, &sdkParams));
    
    // Initialize sensor abstraction layer
    CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_context));
    
    // Initialize visualization context
    CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_context));
    
    log("DriveWorks SDK initialized successfully\n");
}

void MultiCameraDNNApp::initializeMultiCamera()
{
    // Load rig configuration
    std::string rigPath = getArgument("rig");
    if (rigPath.empty()) {
        rigPath = dw_samples::SamplesDataPath::get() + "/samples/sensors/camera/camera/rig_4cam.json";
    }
    
    log("Loading rig configuration: %s\n", rigPath.c_str());
    CHECK_DW_ERROR(dwRig_initializeFromFile(&m_rigConfig, m_context, rigPath.c_str()));
    
    // Query number of cameras in rig
    uint32_t rigCameraCount = 0;
    CHECK_DW_ERROR(dwRig_getSensorCountOfType(&rigCameraCount, DW_SENSOR_CAMERA, m_rigConfig));
    
    if (rigCameraCount == 0) {
        throw std::runtime_error("No cameras found in rig configuration");
    }
    
    m_totalCameras = std::min(rigCameraCount, MAX_CAMERAS);
    log("Found %u cameras in rig, using %u\n", rigCameraCount, m_totalCameras);
    
    // Initialize each camera sensor
    dwSensorParams cameraParams[MAX_CAMERAS] = {};
    for (uint32_t i = 0; i < m_totalCameras; ++i) {
        // Get sensor configuration from rig
        uint32_t sensorIdx = 0;
        CHECK_DW_ERROR(dwRig_findSensorByTypeIndex(&sensorIdx, DW_SENSOR_CAMERA, i, m_rigConfig));
        
        const char* protocol = nullptr;
        const char* parameters = nullptr;
        CHECK_DW_ERROR(dwRig_getSensorProtocol(&protocol, sensorIdx, m_rigConfig));
        CHECK_DW_ERROR(dwRig_getSensorParameterUpdatedPath(&parameters, sensorIdx, m_rigConfig));
        
        cameraParams[i].protocol = protocol;
        cameraParams[i].parameters = parameters;
        
        log("Camera %u: %s with params: %s\n", i, protocol, parameters);
        
        // Create camera sensor
        CHECK_DW_ERROR(dwSAL_createSensor(&m_camera[i], cameraParams[i], m_sal));
        
        // Parse output format flags from parameters
        std::string paramStr(parameters);
        m_useRaw = paramStr.find("raw") != std::string::npos;
        m_useProcessed = paramStr.find("processed") != std::string::npos;
        m_useProcessed1 = paramStr.find("processed1") != std::string::npos;
        m_useProcessed2 = paramStr.find("processed2") != std::string::npos;
        
        // Default to processed if not specified
        if (!m_useProcessed && !m_useRaw) {
            m_useProcessed = true;
        }
        
        // Parse FIFO size if specified
        auto fifoPos = paramStr.find("fifo-size=");
        if (fifoPos != std::string::npos) {
            auto fifoStr = paramStr.substr(fifoPos + 10, 5);
            auto commaPos = fifoStr.find(",");
            if (commaPos != std::string::npos) {
                m_fifoSize = std::stoi(fifoStr.substr(0, commaPos));
            }
        }
    }
    
    // Setup camera properties and image buffers
    setupCameraProperties();
    
    log("Multi-camera initialization completed\n");
}

void MultiCameraDNNApp::initializeRenderer()
{
    // Initialize render engine parameters
    CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&m_renderParams, getWindowWidth(), getWindowHeight()));
    
    m_renderParams.defaultTile.lineWidth = 2.0f;
    m_renderParams.defaultTile.font = DW_RENDER_ENGINE_FONT_VERDANA_24;
    m_renderParams.maxBufferCount = 1;
    
    // Create render engine
    CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &m_renderParams, m_viz));
    
    // Setup camera layout
    setupRenderingLayout();
    
    log("Rendering system initialized\n");
}

void MultiCameraDNNApp::initializeDNNProcessor()
{
    log("Initializing pomo-drivenet DNN processor...\n");
    
    // Create DNN processor instance
    m_dnnProcessor = std::make_unique<DNNProcessor>(m_context, m_imageWidth, m_imageHeight);
    
    // Get DNN configuration from arguments
    std::string modelPath = getArgument("tensorRT_model");
    bool useCuDLA = getArgument("cudla") == "1";
    uint32_t dlaEngine = static_cast<uint32_t>(std::stoi(getArgument("dla-engine")));
    
    // Initialize DNN processor
    bool success = m_dnnProcessor->initialize(modelPath, useCuDLA, dlaEngine);
    if (!success) {
        throw std::runtime_error("Failed to initialize pomo-drivenet DNN processor");
    }
    
    // Initialize inference results for all cameras
    for (uint32_t i = 0; i < m_totalCameras; ++i) {
        m_inferenceResults[i].isValid = false;
        m_inferenceResults[i].detections.clear();
        m_inferenceResults[i].segmentation.drivePixelCount = 0;
        m_inferenceResults[i].segmentation.lanePixelCount = 0;
    }
    
    log("pomo-drivenet DNN processor initialized successfully\n");
}

//#######################################################################################
// CAMERA PIPELINE METHODS
//#######################################################################################

void MultiCameraDNNApp::acquireFrames(dwCameraFrameHandle_t frames[MAX_CAMERAS])
{
    // Initialize all frames to null
    std::fill_n(frames, m_totalCameras, DW_NULL_HANDLE);
    
    // Parallel frame acquisition for better performance
    std::vector<std::future<std::pair<uint32_t, dwStatus >>> acquisitionFutures;
    acquisitionFutures.reserve(m_totalCameras);
    
    for (uint32_t i = 0; i < m_totalCameras; ++i) {
        acquisitionFutures.emplace_back(
            std::async(std::launch::async, [this, i, &frames]() -> std::pair<uint32_t, dwStatus> {
                dwStatus status = dwSensorCamera_readFrame(&frames[i], FRAME_TIMEOUT_US, m_camera[i]);
                return {i, status};
            })
        );
    }
    // Collect results
    uint32_t successCount = 0;
    for (auto& future : acquisitionFutures) {
        auto [cameraIdx, status] = future.get();
        if (status == DW_SUCCESS) {
            successCount++;

        }
    }
}

void MultiCameraDNNApp::processCurrentCamera(dwCameraFrameHandle_t frame)
{
    // Get RGBA image from current camera frame
    dwImageHandle_t imageHandle = DW_NULL_HANDLE;
    CHECK_DW_ERROR(dwSensorCamera_getImage(&imageHandle, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, frame));
    
    // Copy to our managed buffer for processing
    CHECK_DW_ERROR(dwImage_copyConvert(m_imageRGBA[m_currentCamera], imageHandle, m_context));
    
    // Execute DNN inference on current camera
    runDNNInference(m_currentCamera);
    
    // Log processing activity periodically
    if (m_frameCount % 30 == 0) { // Log every 30 frames to avoid spam
        const auto& result = m_inferenceResults[m_currentCamera];
        log("Processing camera %u (frame %u): %zu detections, %s\n", 
            m_currentCamera, m_frameCount, 
            result.detections.size(),
            result.isValid ? "valid" : "invalid");
    }
}

void MultiCameraDNNApp::renderCameraFrame(dwCameraFrameHandle_t frame, uint32_t cameraIndex)
{
    // Set rendering tile for this camera
    CHECK_DW_ERROR(dwRenderEngine_setTile(m_tileVideo[cameraIndex], m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_resetTile(m_renderEngine));
    
    // Get image from frame
    dwImageHandle_t imageHandle = DW_NULL_HANDLE;
    CHECK_DW_ERROR(dwSensorCamera_getImage(&imageHandle, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, frame));
    
    // Stream image to GL domain for rendering
    CHECK_DW_ERROR(dwImageStreamerGL_producerSend(imageHandle, m_streamerToGL[cameraIndex]));
    
    // Receive streamed GL image
    dwImageHandle_t imageGL = DW_NULL_HANDLE;
    CHECK_DW_ERROR(dwImageStreamerGL_consumerReceive(&imageGL, 33000, m_streamerToGL[cameraIndex]));
    
    // Get GL image properties
    dwImageGL* glImage = nullptr;
    CHECK_DW_ERROR(dwImage_getGL(&glImage, imageGL));
    
    // Setup coordinate system and render image
    dwVector2f range{static_cast<float>(glImage->prop.width), static_cast<float>(glImage->prop.height)};
    CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_renderImage2D(glImage, {0, 0, range.x, range.y}, m_renderEngine));
    
    // Render camera information overlay
    dwTime_t timestamp;
    CHECK_DW_ERROR(dwImage_getTimestamp(&timestamp, imageHandle));
    
    std::string cameraInfo = "Camera " + std::to_string(cameraIndex) + 
                            " | Time: " + std::to_string(timestamp);
    
    if (cameraIndex == m_currentCamera) {
        cameraInfo += " [PROCESSING]";
    }
    
    CHECK_DW_ERROR(dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_renderText2D(cameraInfo.c_str(), {25, 25}, m_renderEngine));
    
    // Render DNN results
    renderDetectionResults(cameraIndex);
    renderSegmentationOverlays(cameraIndex);
    
    // Return GL image
    CHECK_DW_ERROR(dwImageStreamerGL_consumerReturn(&imageGL, m_streamerToGL[cameraIndex]));
    CHECK_DW_ERROR(dwImageStreamerGL_producerReturn(nullptr, 32000, m_streamerToGL[cameraIndex]));
}

void MultiCameraDNNApp::returnFrames(dwCameraFrameHandle_t frames[MAX_CAMERAS])
{
    for (uint32_t i = 0; i < m_totalCameras; ++i) {
        if (frames[i] != DW_NULL_HANDLE) {
            CHECK_DW_ERROR(dwSensorCamera_returnFrame(&frames[i]));
        }
    }
}

//#######################################################################################
// UTILITY METHODS
//#######################################################################################

void MultiCameraDNNApp::setupCameraProperties()
{
    // Get image properties from first camera
    dwImageProperties imageProps{};
    CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&imageProps, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, m_camera[0]));
    
    m_imageWidth = imageProps.width;
    m_imageHeight = imageProps.height;
    
    log("Camera resolution: %ux%u\n", m_imageWidth, m_imageHeight);
    
    // Create image buffers and streamers for each camera
    for (uint32_t i = 0; i < m_totalCameras; ++i) {
        // Create CUDA RGBA image for processing
        CHECK_DW_ERROR(dwImage_create(&m_imageRGBA[i], imageProps, m_context));
        
        // Create image streamer for CUDA to GL transfer
        CHECK_DW_ERROR(dwImageStreamerGL_initialize(&m_streamerToGL[i], &imageProps, DW_IMAGE_GL, m_context));
    }
}

void MultiCameraDNNApp::setupRenderingLayout()
{
    float windowWidth = static_cast<float>(getWindowWidth());
    float windowHeight = static_cast<float>(getWindowHeight());
    
    // Calculate tiles per row based on camera count
    uint32_t tilesPerRow = 1;
    switch (m_totalCameras) {
        case 1: tilesPerRow = 1; break;
        case 2: tilesPerRow = 2; break;
        case 3: 
        case 4: tilesPerRow = 2; break;
        default: tilesPerRow = 4; break;
    }
    
    // Initialize tile state for each camera
    dwRenderEngineTileState tileStates[MAX_CAMERAS];
    for (uint32_t i = 0; i < m_totalCameras; ++i) {
        CHECK_DW_ERROR(dwRenderEngine_initTileState(&tileStates[i]));
        tileStates[i].modelViewMatrix = DW_IDENTITY_MATRIX4F;
        tileStates[i].font = DW_RENDER_ENGINE_FONT_VERDANA_16;
    }
    
    // Add tiles to render engine
    CHECK_DW_ERROR(dwRenderEngine_addTilesByCount(m_tileVideo, m_totalCameras, tilesPerRow, 
                                                  tileStates, m_renderEngine));
    
    log("Rendering layout: %u cameras in %u tiles per row\n", m_totalCameras, tilesPerRow);
}

void MultiCameraDNNApp::advanceCurrentCamera()
{
    m_currentCamera = (m_currentCamera + 1) % m_totalCameras;
}

void MultiCameraDNNApp::handleFrameTimeout(uint32_t cameraIndex)
{
    static uint32_t timeoutCount[MAX_CAMERAS] = {0};
    timeoutCount[cameraIndex]++;
    
    if (timeoutCount[cameraIndex] % 30 == 0) { // Log every 30 timeouts
        logWarn("Camera %u: %u frame timeouts\n", cameraIndex, timeoutCount[cameraIndex]);
    }
}

void MultiCameraDNNApp::logDNNStatus() const
{
    log("=== DNN System Status ===\n");
    if (m_dnnProcessor && m_dnnProcessor->isInitialized()) {
        log("pomo-drivenet processor: initialized\n");
        log("Total inferences: %u\n", m_dnnProcessor->getInferenceCount());
        
        // Log recent results summary
        uint32_t validResults = 0;
        uint32_t totalDetections = 0;
        for (uint32_t i = 0; i < m_totalCameras; ++i) {
            if (m_inferenceResults[i].isValid) {
                validResults++;
                totalDetections += static_cast<uint32_t>(m_inferenceResults[i].detections.size());
            }
        }
        log("Valid results: %u/%u cameras\n", validResults, m_totalCameras);
        log("Total detections: %u\n", totalDetections);
    } else {
        log("pomo-drivenet processor: not initialized\n");
    }
    log("========================\n");
}

void MultiCameraDNNApp::logCameraStatus() const
{
    log("=== Camera System Status ===\n");
    log("Total cameras: %u\n", m_totalCameras);
    log("Image resolution: %ux%u\n", m_imageWidth, m_imageHeight);
    log("Output formats - Raw: %s, Processed: %s\n", 
        m_useRaw ? "Yes" : "No", m_useProcessed ? "Yes" : "No");
    log("FIFO buffer size: %u\n", m_fifoSize);
    log("Current processing camera: %u\n", m_currentCamera);
    log("===========================\n");
}

//#######################################################################################
// DNN INTEGRATION METHODS
//#######################################################################################

void MultiCameraDNNApp::runDNNInference(uint32_t cameraIndex){
    if (!m_dnnProcessor || cameraIndex >= MAX_CAMERAS) {
        return;
    }
    
    // Get RGBA image for DNN processing
    dwImageHandle_t rgbaImage = m_imageRGBA[cameraIndex];
    if (rgbaImage == DW_NULL_HANDLE) {
        logError("No RGBA image available for camera %u\n", cameraIndex);
        return;
    }
    
    // Process image through DNN
    bool success = m_dnnProcessor->processImage(rgbaImage, m_inferenceResults[cameraIndex]);
    if (!success) {
        logError("DNN inference failed for camera %u\n", cameraIndex);
    } else {
        logInfo("Camera %u: Found %zu detections, drive_area: %s, lane_lines: %s\n",
                cameraIndex, 
                m_inferenceResults[cameraIndex].detections.size(),
                m_inferenceResults[cameraIndex].hasDriveAreaMask ? "yes" : "no",
                m_inferenceResults[cameraIndex].hasLaneLineMask ? "yes" : "no");
    }
}


void MultiCameraDNNApp::renderDetectionResults(uint32_t cameraIndex)
{
    const auto& result = m_inferenceResults[cameraIndex];
    
    if (!result.isValid || result.detections.empty()) {
        return;
    }
    
    // Render detection bounding boxes in red
    CHECK_DW_ERROR(dwRenderEngine_setColor({1.0f, 0.0f, 0.0f, 1.0f}, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_setLineWidth(2.0f, m_renderEngine));
    
    for (const auto& detection : result.detections) {
        CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D,
                                           &detection.bbox, sizeof(dwRectf), 0, 1, m_renderEngine));
    }
    
    // Render detection count
    std::string detectionInfo = "Detections: " + std::to_string(result.detections.size());
    CHECK_DW_ERROR(dwRenderEngine_setColor({1.0f, 1.0f, 0.0f, 1.0f}, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_renderText2D(detectionInfo.c_str(), {25, 50}, m_renderEngine));
}

void MultiCameraDNNApp::renderSegmentationOverlays(uint32_t cameraIndex)
{
    if (cameraIndex >= MAX_CAMERAS) return;
    
    const auto& results = m_inferenceResults[cameraIndex];
    
    // Render drive area segmentation overlay
    if (results.hasDriveAreaMask) {
        dwRenderEngineColorRGBA driveAreaColor = {0.0f, 1.0f, 0.0f, 0.3f}; // Green with transparency
        CHECK_DW_ERROR(dwRenderEngine_setColor(driveAreaColor, m_renderEngine));
        // Implementation would render segmentation mask as overlay
        // This requires converting mask to renderable format
    }
    
    // Render lane line segmentation overlay
    if (results.hasLaneLineMask) {
        dwRenderEngineColorRGBA laneLineColor = {0.0f, 0.0f, 1.0f, 0.5f}; // Blue with transparency
        CHECK_DW_ERROR(dwRenderEngine_setColor(laneLineColor, m_renderEngine));
        // Implementation would render lane line mask as overlay
    }
}
