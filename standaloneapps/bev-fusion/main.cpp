/////////////////////////////////////////////////////////////////////////////////////////
//
// ./sample_bevfusion_driveworks \    --hostpc=1 \    --bevfusion-model=/usr/local/Lidar_AI_Solution/CUDA-BEVFusion/model/resnet50int8 \    --bevfusion-precision=int8 \    --example-data-path=/usr/local/Lidar_AI_Solution/CUDA-BEVFusion/example-data \    --enable-logging=1 \    --lidar-ip=192.168.2.201 \    --lidar-port=2368 \    > /usr/local/driveworks/samples/src/sensors/sensor_fusion/bevfusion_logs.txt 2>&1
//
/////////////////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <dlfcn.h>

// BEVFusion includes
#include "bevfusion/bevfusion.hpp"
#include "common/check.hpp"
#include "common/tensor.hpp"
#include "common/timer.hpp"
#include "common/visualize.hpp"

#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/image/Image.h>
#include <dw/imageprocessing/geometry/imagetransformation/ImageTransformation.h>
#include <dw/interop/streamer/ImageStreamer.h>
#include <dw/rig/Rig.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/camera/Camera.h>
#include <dw/sensors/lidar/Lidar.h>
#include <dw/sensors/sensormanager/SensorManager.h>

#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Renderer.h>
#include <dwvisualization/core/Visualization.h>
#include <dwvisualization/image/Image.h>
#include <dwvisualization/interop/ImageStreamer.h>

#include <framework/Checks.hpp>
#include <framework/CudaTimer.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/Log.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/RenderUtils.hpp>
#include <framework/SamplesDataPath.hpp>
#include <framework/ScreenshotHelper.hpp>
#include <framework/WindowGLFW.hpp>

using namespace dw_samples::common;

#define MAX_CAMERAS 6
#define MAX_LIDARS 1
#define BEVFUSION_IMAGE_WIDTH 1600
#define BEVFUSION_IMAGE_HEIGHT 900
#define BEVFUSION_OUTPUT_WIDTH 704
#define BEVFUSION_OUTPUT_HEIGHT 256

// Structure to hold camera data
struct CameraData {
    dwSensorHandle_t sensor = DW_NULL_HANDLE;
    dwImageStreamerHandle_t streamerToCPU = DW_NULL_HANDLE;
    dwImageHandle_t currentFrame = DW_NULL_HANDLE;
    unsigned char* imageData = nullptr;
    bool isBlackImage = false;
    std::string name;
    dwTransformation3f sensorToRig;
};

    // Structure to hold LiDAR data
struct LidarData {
    dwSensorHandle_t sensor = DW_NULL_HANDLE;
    std::unique_ptr<float32_t[]> pointCloud;        // For BEVFusion (XYZI + timestamp)
    std::unique_ptr<float32_t[]> pointCloudViz;     // For DriveWorks visualization (XYZI)
    uint32_t pointCount = 0;
    uint32_t maxPoints = 250000; // Increased for full LiDAR scans (BEVFusion expects ~242k points)
    std::string name;
    dwTransformation3f sensorToRig;
    
    // Default constructor
    LidarData() = default;
    
    // Move constructor
    LidarData(LidarData&& other) = default;
    
    // Move assignment
    LidarData& operator=(LidarData&& other) = default;
    
    // Delete copy constructor and assignment
    LidarData(const LidarData&) = delete;
    LidarData& operator=(const LidarData&) = delete;
};

class BEVFusionDriveWorksApp : public DriveWorksSample
{
    private:
    // DriveWorks components
    dwContextHandle_t m_context = DW_NULL_HANDLE;
    dwVisualizationContextHandle_t m_viz = DW_NULL_HANDLE;
    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
    dwSALHandle_t m_sal = DW_NULL_HANDLE;
    dwRigHandle_t m_rigConfig = DW_NULL_HANDLE;
    dwSensorManagerHandle_t m_sensorManager = DW_NULL_HANDLE;

    // Camera and LiDAR data
    std::vector<CameraData> m_cameras;
    std::vector<LidarData> m_lidars;
    
    // BEVFusion components
    std::shared_ptr<bevfusion::Core> m_bevfusionCore;
    cudaStream_t m_cudaStream;
    
    // Data synchronization
    std::mutex m_dataMutex;
    std::condition_variable m_dataCV;
    bool m_newDataAvailable = false;
    
    // Configuration
    std::string m_rigFile;
    std::string m_bevfusionModelPath;
    std::string m_bevfusionPrecision;
    std::string m_exampleDataPath;
    bool m_useBlackImages[MAX_CAMERAS] = {false, false, false, false, false, false};
    bool m_enableLogging = false;
    bool m_hostPC = false;
    bool m_cameraOnly = false;  // New flag for camera-only mode
    // Runtime tuning
    float m_processRateHz = 0.0f;       // 0 = uncapped
    uint32_t m_stabilizeSkipFrames = 0; // 0 = disable
    std::string m_lidarIP;
    uint32_t m_lidarPort;
    
    // Rendering
    std::unique_ptr<ScreenshotHelper> m_screenshot;
    dwImageStreamerHandle_t m_streamerToGL[MAX_CAMERAS] = {DW_NULL_HANDLE};
    uint32_t m_tileVideo = 0;
    uint32_t m_tileCameras[MAX_CAMERAS] = {0};
    
    // Camera rendering images
    dwImageHandle_t m_cameraRGBAImages[MAX_CAMERAS] = {DW_NULL_HANDLE};
    dwImageHandle_t m_cameraRGBImages[MAX_CAMERAS] = {DW_NULL_HANDLE};  // For BEVFusion processing
    
    // Camera projection data for 3D→2D bounding box projection
    // Note: We use the existing m_lidar2image tensor that's already loaded by loadTransformationMatrices()
    
    // Performance tracking
    CudaTimer m_timer;
    float m_inferenceTime = 0.0f;
    uint32_t m_frameCount = 0;
    
    // BEVFusion transformation matrices
    nv::Tensor m_camera2lidar;
    nv::Tensor m_cameraIntrinsics;
    nv::Tensor m_lidar2image;
    nv::Tensor m_imgAugMatrix;
    
    // Image transformation for resizing
    dwImageTransformationHandle_t m_imageTransformer = DW_NULL_HANDLE;
    
    // Visualization
    uint32_t m_pointCloudBuffer = 0;
    uint32_t m_pointCloudBufferSize = 0;
    uint32_t m_pointCloudBufferCapacity = 0;
    std::unique_ptr<float32_t[]> m_pointCloudViz;
    std::vector<bevfusion::head::transbbox::BoundingBox> m_lastBboxes;
    
    // 3D Bounding Box rendering
    uint32_t m_boxLineBuffer = 0;
    static constexpr int VERTICES_PER_BOX = 16;

public:
    BEVFusionDriveWorksApp(const ProgramArguments& args)
    : DriveWorksSample(args) {}

    void initializeDriveWorks(dwContextHandle_t& context)
    {
    log("=== Initializing DriveWorks ===\n");
    
    // Initialize logger
    log("Initializing logger...\n");
    CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
    CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));
    log("Logger initialized successfully\n");

    // Initialize SDK context
    log("Initializing SDK context...\n");
    dwContextParameters sdkParams = {};
    #ifdef VIBRANTE
    sdkParams.eglDisplay = getEGLDisplay();
    log("EGL display configured for Vibrante\n");
    #endif
    CHECK_DW_ERROR(dwInitialize(&context, DW_VERSION, &sdkParams));
    log("SDK context initialized successfully\n");

    // Initialize SAL
    log("Initializing SAL...\n");
    CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_context));
    log("SAL initialized successfully\n");

    // Get configuration first (including host PC mode and camera-only mode)
    m_bevfusionModelPath = getArgument("bevfusion-model");
    m_bevfusionPrecision = getArgument("bevfusion-precision");
    m_exampleDataPath = getArgument("example-data-path");
    m_enableLogging = (std::stoi(getArgument("enable-logging")) > 0);
    m_hostPC = (std::stoi(getArgument("hostpc")) > 0);
    m_cameraOnly = (std::stoi(getArgument("camera-only")) > 0);
    // Optional runtime tuning
    try {
        m_processRateHz = std::stof(getArgument("process-rate"));
    } catch (...) { m_processRateHz = 0.0f; }
    try {
        m_stabilizeSkipFrames = static_cast<uint32_t>(std::stoul(getArgument("stabilize-skip")));
    } catch (...) { m_stabilizeSkipFrames = 0; }
    m_lidarIP = getArgument("lidar-ip");
    m_lidarPort = std::stoi(getArgument("lidar-port"));

    // Initialize rig configuration (only if not in host PC mode)
    if (!m_hostPC) {
    m_rigFile = getArgument("rig");
    CHECK_DW_ERROR_MSG(dwRig_initializeFromFile(&m_rigConfig, m_context, m_rigFile.c_str()),
    "Could not initialize Rig from file");
    log("Rig file loaded successfully: %s\n", m_rigFile.c_str());
    } else {
    // Host PC mode: no rig file needed
    log("Host PC mode: no rig file needed\n");
    }
    
    // Parse black image configuration
    std::string blackImages = getArgument("black-images");
    if (!blackImages.empty()) {
    std::istringstream iss(blackImages);
    std::string token;
    while (std::getline(iss, token, ',')) {
    int idx = std::stoi(token);
    if (idx >= 0 && idx < MAX_CAMERAS) {
    m_useBlackImages[idx] = true;
    }
    }
    }
    
    // In host PC mode, all cameras use black images
    if (m_hostPC) {
    for (int i = 0; i < MAX_CAMERAS; i++) {
    m_useBlackImages[i] = true;
    }
    }
    
    // Log configuration
    if (m_enableLogging) {
    log("=== BEVFusion DriveWorks Configuration ===\n");
    log("Host PC Mode: %s\n", m_hostPC ? "true" : "false");
    log("Camera Only Mode: %s\n", m_cameraOnly ? "true" : "false");
    log("Rig File: %s\n", m_rigFile.c_str());
    log("BEVFusion Model Path: %s\n", m_bevfusionModelPath.c_str());
    log("BEVFusion Precision: %s\n", m_bevfusionPrecision.c_str());
    log("Example Data Path: %s\n", m_exampleDataPath.c_str());
    log("Output Path: Disabled (GUI visualization only)\n");
    log("Enable Logging: %s\n", m_enableLogging ? "true" : "false");
    log("LiDAR IP: %s\n", m_lidarIP.c_str());
    log("LiDAR Port: %d\n", m_lidarPort);
    log("Black Images: %s\n", blackImages.c_str());
    log("==========================================\n");
    }
    }

    void initializeSensors()
    {
    if (m_cameraOnly) {
    initializeSensorsCameraOnly();
    } else if (m_hostPC) {
    initializeSensorsHostPC();
    } else {
    initializeSensorsFromRig();
    }
    }
    
    void initializeSensorsHostPC()
    {
    log("Initializing sensors in Host PC mode\n");
    
    // Create 6 black image cameras
    for (int i = 0; i < MAX_CAMERAS; i++) {
    CameraData camera;
    camera.sensor = DW_NULL_HANDLE; // No real sensor
    camera.isBlackImage = true;
    camera.name = "black_camera_" + std::to_string(i);
    
    // Allocate black image
    size_t imageSize = BEVFUSION_IMAGE_WIDTH * BEVFUSION_IMAGE_HEIGHT * 3;
    camera.imageData = new unsigned char[imageSize];
    memset(camera.imageData, 0, imageSize);
    
    m_cameras.push_back(camera);
    log("Created black image camera %d\n", i);
    }
    
    // Initialize LiDAR manually (Host PC mode)
    initializeLidarHostPC();
    
    log("Host PC sensor initialization complete\n");
    }
    
    void initializeLidarHostPC()
    {
    log("Initializing LiDAR in Host PC mode\n");
    
    // Create LiDAR sensor manually using direct SAL approach
    dwSensorParams lidarParams = {};
    lidarParams.protocol = "lidar.socket";
    
    // Create parameters string following the NVIDIA format (only supported parameters)
    std::string paramString = "ip=" + m_lidarIP + ",port=" + std::to_string(m_lidarPort) + ",device=VELO_VLP16,scan-frequency=5.0,protocol=udp";
    lidarParams.parameters = paramString.c_str();
    
        LidarData lidar;
    dwStatus status = dwSAL_createSensor(&lidar.sensor, lidarParams, m_sal);
    
    if (status == DW_SUCCESS) {
        lidar.name = "host_lidar";
        lidar.pointCloud.reset(new float32_t[lidar.maxPoints * 5]);      // BEVFusion format (XYZI + timestamp)
        lidar.pointCloudViz.reset(new float32_t[lidar.maxPoints * 4]);   // DriveWorks format (XYZI)
        
        m_lidars.emplace_back(std::move(lidar));
        log("LiDAR initialized successfully: %s:%d\n", m_lidarIP.c_str(), m_lidarPort);
    } else {
        logError("Failed to initialize LiDAR: %s:%d\n", m_lidarIP.c_str(), m_lidarPort);
    }
    
    // Start SAL (following NVIDIA camera sample approach)
    CHECK_DW_ERROR(dwSAL_start(m_sal));
    log("SAL started successfully\n");
    
    // CRITICAL FIX: Start the LiDAR sensor after SAL is started
    if (!m_lidars.empty() && m_lidars[0].sensor != DW_NULL_HANDLE) {
        CHECK_DW_ERROR(dwSensor_start(m_lidars[0].sensor));
        log("LiDAR sensor started successfully\n");
    }
    }
    
    void initializeSensorsCameraOnly()
    {
    log("=== CAMERA-ONLY MODE: Initializing cameras using NVIDIA camera sample pipeline ===\n");
    log("Camera-only mode: initializing cameras from rig file (no LiDAR, no BEVFusion)\n");
    
    // Initialize rig configuration (following NVIDIA camera sample)
    m_rigFile = getArgument("rig");
    CHECK_DW_ERROR_MSG(dwRig_initializeFromFile(&m_rigConfig, m_context, m_rigFile.c_str()),
    "Could not initialize Rig from file");
    
    // Get camera count from rig (following NVIDIA camera sample)
    uint32_t cameraCount = 0;
    CHECK_DW_ERROR(dwRig_getSensorCountOfType(&cameraCount, DW_SENSOR_CAMERA, m_rigConfig));
    
    log("Found %d cameras in rig file\n", cameraCount);
    
    // Initialize cameras using NVIDIA camera sample approach
    dwSensorParams paramsClient[MAX_CAMERAS] = {};
    for (uint32_t i = 0; i < cameraCount && i < MAX_CAMERAS; i++) {
    CameraData camera;
    
    // Get camera sensor index from rig (following NVIDIA camera sample)
    uint32_t cameraSensorIdx = 0;
    CHECK_DW_ERROR(dwRig_findSensorByTypeIndex(&cameraSensorIdx, DW_SENSOR_CAMERA, i, m_rigConfig));
    
    // Get rig parsed protocol from dwRig (following NVIDIA camera sample)
    const char* protocol = nullptr;
    CHECK_DW_ERROR(dwRig_getSensorProtocol(&protocol, cameraSensorIdx, m_rigConfig));
    const char* params = nullptr;
    CHECK_DW_ERROR(dwRig_getSensorParameterUpdatedPath(&params, cameraSensorIdx, m_rigConfig));
    
    paramsClient[i].protocol = protocol;
    paramsClient[i].parameters = params;
    
    // Check if this camera should use black images based on command line arguments
    bool useBlackImage = m_useBlackImages[i];
    
    if (!useBlackImage) {
    // Create real camera sensor (following NVIDIA camera sample)
    log("Creating camera.gmsl with params: %s\n", params);
    CHECK_DW_ERROR(dwSAL_createSensor(&camera.sensor, paramsClient[i], m_sal));
    
    // Get sensor to rig transformation
    CHECK_DW_ERROR(dwRig_getSensorToRigTransformation(&camera.sensorToRig, cameraSensorIdx, m_rigConfig));
    
    // Get camera name
    const char* cameraName;
    CHECK_DW_ERROR(dwRig_getSensorName(&cameraName, cameraSensorIdx, m_rigConfig));
    camera.name = std::string(cameraName);
    camera.isBlackImage = false;
    
    log("Camera %d (%s): live camera (camera-only mode)\n", i, camera.name.c_str());
    } else {
    // Create black image camera
    camera.sensor = DW_NULL_HANDLE;
    camera.isBlackImage = true;
    camera.name = "black_camera_" + std::to_string(i);
    
    size_t imageSize = BEVFUSION_IMAGE_WIDTH * BEVFUSION_IMAGE_HEIGHT * 3;
    camera.imageData = new unsigned char[imageSize];
    memset(camera.imageData, 0, imageSize);
    
    log("Camera %d (%s): black image (camera-only mode)\n", i, camera.name.c_str());
    }
    
    m_cameras.push_back(camera);
    }
    
    // Fill remaining cameras with black images
    for (uint32_t i = cameraCount; i < MAX_CAMERAS; i++) {
    CameraData camera;
    camera.sensor = DW_NULL_HANDLE;
    camera.isBlackImage = true;
    camera.name = "black_camera_" + std::to_string(i);
    
    size_t imageSize = BEVFUSION_IMAGE_WIDTH * BEVFUSION_IMAGE_HEIGHT * 3;
    camera.imageData = new unsigned char[imageSize];
    memset(camera.imageData, 0, imageSize);
    
    m_cameras.push_back(camera);
    log("Added black image camera %d (camera-only mode)\n", i);
    }
    
    // Start SAL (following NVIDIA camera sample)
    CHECK_DW_ERROR(dwSAL_start(m_sal));
    log("SAL started successfully\n");
    
    log("Camera-only initialization completed (no LiDAR, no BEVFusion)\n");
    }
    
    void initializeSensorsFromRig()
    {
    log("Initializing sensors from rig file using NVIDIA camera sample approach\n");
    
    // Initialize cameras using NVIDIA camera sample approach
    uint32_t cameraCount = 0;
    CHECK_DW_ERROR(dwRig_getSensorCountOfType(&cameraCount, DW_SENSOR_CAMERA, m_rigConfig));
    
    log("Found %d cameras in rig file\n", cameraCount);
    
    // Initialize cameras using NVIDIA camera sample approach
    dwSensorParams paramsClient[MAX_CAMERAS] = {};
    for (uint32_t i = 0; i < cameraCount && i < MAX_CAMERAS; i++) {
    CameraData camera;
    
    // Get camera sensor index from rig (following NVIDIA camera sample)
    uint32_t cameraSensorIdx = 0;
    CHECK_DW_ERROR(dwRig_findSensorByTypeIndex(&cameraSensorIdx, DW_SENSOR_CAMERA, i, m_rigConfig));
    
    // Get rig parsed protocol from dwRig (following NVIDIA camera sample)
    const char* protocol = nullptr;
    CHECK_DW_ERROR(dwRig_getSensorProtocol(&protocol, cameraSensorIdx, m_rigConfig));
    const char* params = nullptr;
    CHECK_DW_ERROR(dwRig_getSensorParameterUpdatedPath(&params, cameraSensorIdx, m_rigConfig));
    
    paramsClient[i].protocol = protocol;
    paramsClient[i].parameters = params;
    
    // Check if this camera should use black images based on command line arguments
    bool useBlackImage = m_useBlackImages[i];
    
        if (!useBlackImage) {
        // Create real camera sensor (following NVIDIA camera sample)
        log("Creating camera.gmsl with params: %s\n", params);
        CHECK_DW_ERROR(dwSAL_createSensor(&camera.sensor, paramsClient[i], m_sal));
        
        // Get sensor to rig transformation
        CHECK_DW_ERROR(dwRig_getSensorToRigTransformation(&camera.sensorToRig, cameraSensorIdx, m_rigConfig));
        
        // Get camera name
        const char* cameraName;
        CHECK_DW_ERROR(dwRig_getSensorName(&cameraName, cameraSensorIdx, m_rigConfig));
        camera.name = std::string(cameraName);
        camera.isBlackImage = false;
        
        // CRITICAL FIX: Initialize CPU streamer for BEVFusion processing with correct format
        dwImageProperties imageProperties;
        imageProperties.type = DW_IMAGE_CUDA;
        imageProperties.format = DW_IMAGE_FORMAT_RGB_UINT8;  // RGB for BEVFusion
        imageProperties.width = BEVFUSION_IMAGE_WIDTH;       // Use BEVFusion dimensions
        imageProperties.height = BEVFUSION_IMAGE_HEIGHT;
        CHECK_DW_ERROR(dwImageStreamer_initialize(&camera.streamerToCPU, &imageProperties, DW_IMAGE_CPU, m_context));
        
        log("Camera %d (%s): live camera with CPU streamer initialized\n", i, camera.name.c_str());
    } else {
    // Create black image camera
    camera.sensor = DW_NULL_HANDLE;
    camera.isBlackImage = true;
    camera.name = "black_camera_" + std::to_string(i);
    
    // CRITICAL FIX: Initialize CPU streamer for black image cameras too (following NVIDIA pattern)
    dwImageProperties imageProperties;
    imageProperties.type = DW_IMAGE_CUDA;
    imageProperties.format = DW_IMAGE_FORMAT_RGB_UINT8;
    imageProperties.width = BEVFUSION_IMAGE_WIDTH;
    imageProperties.height = BEVFUSION_IMAGE_HEIGHT;
    CHECK_DW_ERROR(dwImageStreamer_initialize(&camera.streamerToCPU, &imageProperties, DW_IMAGE_CPU, m_context));
    
    size_t imageSize = BEVFUSION_IMAGE_WIDTH * BEVFUSION_IMAGE_HEIGHT * 3;
    camera.imageData = new unsigned char[imageSize];
    memset(camera.imageData, 0, imageSize);
    
    log("Camera %d (%s): black image with CPU streamer\n", i, camera.name.c_str());
    }
    
    m_cameras.push_back(camera);
    }
    
    // Fill remaining cameras with black images if needed
    for (uint32_t i = cameraCount; i < MAX_CAMERAS; i++) {
    CameraData camera;
    camera.sensor = DW_NULL_HANDLE;
    camera.isBlackImage = true;
    camera.name = "black_camera_" + std::to_string(i);
    
    size_t imageSize = BEVFUSION_IMAGE_WIDTH * BEVFUSION_IMAGE_HEIGHT * 3;
    camera.imageData = new unsigned char[imageSize];
    memset(camera.imageData, 0, imageSize);
    
    m_cameras.push_back(camera);
    log("Added black image camera %d (no real camera available)\n", i);
    }
    
    // Log rig camera indices and names for verification
    log("=== Rig camera indices and names ===\n");
    for (size_t i = 0; i < m_cameras.size(); ++i) {
        log("rig[%zu] name=%s (black=%s)\n", i, m_cameras[i].name.c_str(), m_cameras[i].isBlackImage ? "true" : "false");
    }
    log("====================================\n");
    
    // Log intended BEVFusion mapping (BEV indices -> rig indices)
    log("=== Intended Camera Mapping for BEVFusion (index -> rig[name]) ===\n");
    auto name_or_black = [&](int rigIdx)->const char*{
        if (rigIdx < 0 || rigIdx >= (int)m_cameras.size()) return "BLACK IMAGE";
        return m_cameras[rigIdx].isBlackImage ? "BLACK IMAGE" : m_cameras[rigIdx].name.c_str();
    };
    // Target mapping: 0=-1, 1=1, 2=0, 3=-1, 4=2, 5=3
    log("0 FRONT       -> %s\n", name_or_black(-1));
    log("1 FRONT_RIGHT -> %s\n", name_or_black(1));
    log("2 FRONT_LEFT  -> %s\n", name_or_black(0));
    log("3 BACK        -> %s\n", name_or_black(-1));
    log("4 BACK_LEFT   -> %s\n", name_or_black(2));
    log("5 BACK_RIGHT  -> %s\n", name_or_black(3));
    log("==============================================================\n");
    
    // Initialize LiDAR using rig file approach
    uint32_t lidarCount = 0;
    CHECK_DW_ERROR(dwRig_getSensorCountOfType(&lidarCount, DW_SENSOR_LIDAR, m_rigConfig));
    
    log("Found %d LiDAR sensors in rig file\n", lidarCount);
    
    for (uint32_t i = 0; i < lidarCount && i < MAX_LIDARS; i++) {
    LidarData lidar;
    
    // Get LiDAR sensor index from rig (following same approach as cameras)
    uint32_t lidarSensorIdx = 0;
    CHECK_DW_ERROR(dwRig_findSensorByTypeIndex(&lidarSensorIdx, DW_SENSOR_LIDAR, i, m_rigConfig));
    
    // Get rig parsed protocol from dwRig
    const char* protocol = nullptr;
    CHECK_DW_ERROR(dwRig_getSensorProtocol(&protocol, lidarSensorIdx, m_rigConfig));
    const char* params = nullptr;
    CHECK_DW_ERROR(dwRig_getSensorParameterUpdatedPath(&params, lidarSensorIdx, m_rigConfig));
    
    dwSensorParams lidarParams = {};
    lidarParams.protocol = protocol;
    lidarParams.parameters = params;
    
    // Create LiDAR sensor
    log("Creating LiDAR sensor with params: %s\n", params);
    CHECK_DW_ERROR(dwSAL_createSensor(&lidar.sensor, lidarParams, m_sal));
    
    // Get sensor to rig transformation
    CHECK_DW_ERROR(dwRig_getSensorToRigTransformation(&lidar.sensorToRig, lidarSensorIdx, m_rigConfig));
    
    // Get LiDAR name
    const char* lidarName;
    CHECK_DW_ERROR(dwRig_getSensorName(&lidarName, lidarSensorIdx, m_rigConfig));
    lidar.name = std::string(lidarName);
    
    // Allocate point cloud buffers
    lidar.pointCloud.reset(new float32_t[lidar.maxPoints * 5]);      // BEVFusion format (XYZI + timestamp)
    lidar.pointCloudViz.reset(new float32_t[lidar.maxPoints * 4]);   // DriveWorks format (XYZI)
    
    m_lidars.emplace_back(std::move(lidar));
    log("LiDAR %d (%s): initialized\n", i, lidar.name.c_str());
    }
    
    // Start SAL (following NVIDIA camera sample)
    CHECK_DW_ERROR(dwSAL_start(m_sal));
    log("SAL started successfully\n");
    
    // CRITICAL FIX: Start individual camera sensors after SAL is started
    for (size_t i = 0; i < m_cameras.size(); i++) {
        if (m_cameras[i].sensor != DW_NULL_HANDLE && !m_cameras[i].isBlackImage) {
            CHECK_DW_ERROR(dwSensor_start(m_cameras[i].sensor));
            log("Camera %zu (%s): sensor started successfully\n", i, m_cameras[i].name.c_str());
        }
    }
    
    // CRITICAL FIX: Start individual LiDAR sensors after SAL is started
    for (size_t i = 0; i < m_lidars.size(); i++) {
        if (m_lidars[i].sensor != DW_NULL_HANDLE) {
            CHECK_DW_ERROR(dwSensor_start(m_lidars[i].sensor));
            log("LiDAR %zu (%s): sensor started successfully\n", i, m_lidars[i].name.c_str());
        }
    }
    }

    void initializeBEVFusion()
    {
    if (m_cameraOnly) {
    log("=== CAMERA-ONLY MODE ENABLED ===\n");
    log("Skipping LiDAR and BEVFusion initialization for debugging\n");
    return;
    }
    
    log("=== Initializing BEVFusion ===\n");
    
    // Load custom layernorm plugin
    log("Loading custom layernorm plugin...\n");
    void* handle = dlopen("libcustom_layernorm.so", RTLD_NOW);
    if (!handle) {
    logWarn("Failed to load libcustom_layernorm.so: %s\n", dlerror());
    log("Continuing without custom layernorm plugin...\n");
    } else {
    log("Custom layernorm plugin loaded successfully\n");
    }
    
    // Create BEVFusion core
    log("Creating BEVFusion core...\n");
    m_bevfusionCore = createBEVFusionCore(m_bevfusionModelPath, m_bevfusionPrecision);
    if (!m_bevfusionCore) {
    logError("Failed to create BEVFusion core\n");
    throw std::runtime_error("Failed to create BEVFusion core");
    }
    log("BEVFusion core created successfully\n");
    
    // Create CUDA stream
    log("Creating CUDA stream...\n");
    cudaError_t cudaStatus = cudaStreamCreate(&m_cudaStream);
    if (cudaStatus != cudaSuccess) {
    logError("Failed to create CUDA stream: %s\n", cudaGetErrorString(cudaStatus));
    throw std::runtime_error("Failed to create CUDA stream");
    }
    log("CUDA stream created successfully\n");
    
    // Load transformation matrices from BEVFusion example data
    log("Loading transformation matrices...\n");
    loadTransformationMatrices();
    
    // Update BEVFusion with transformation matrices
    log("Updating BEVFusion with transformation matrices...\n");
    m_bevfusionCore->update(m_camera2lidar.ptr<float>(), 
    m_cameraIntrinsics.ptr<float>(), 
    m_lidar2image.ptr<float>(), 
    m_imgAugMatrix.ptr<float>(),
    m_cudaStream);
    log("BEVFusion updated with transformation matrices\n");
    
    log("Printing BEVFusion configuration...\n");
    m_bevfusionCore->print();
    m_bevfusionCore->set_timer(true);
    
    log("BEVFusion initialized successfully\n");
    }

    std::shared_ptr<bevfusion::Core> createBEVFusionCore(const std::string& model, const std::string& precision)
    {
    log("Creating BEVFusion core with model: %s, precision: %s\n", model.c_str(), precision.c_str());
    
    bevfusion::camera::NormalizationParameter normalization;
    normalization.image_width = BEVFUSION_IMAGE_WIDTH;
    normalization.image_height = BEVFUSION_IMAGE_HEIGHT;
    normalization.output_width = BEVFUSION_OUTPUT_WIDTH;
    normalization.output_height = BEVFUSION_OUTPUT_HEIGHT;
    normalization.num_camera = MAX_CAMERAS;
    normalization.resize_lim = 0.48f;
    normalization.interpolation = bevfusion::camera::Interpolation::Bilinear;

    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};
    normalization.method = bevfusion::camera::NormMethod::mean_std(mean, std, 1 / 255.0f, 0.0f);

    bevfusion::lidar::VoxelizationParameter voxelization;
    voxelization.min_range = nvtype::Float3(-54.0f, -54.0f, -5.0);
    voxelization.max_range = nvtype::Float3(+54.0f, +54.0f, +3.0);
    voxelization.voxel_size = nvtype::Float3(0.075f, 0.075f, 0.2f);
    voxelization.grid_size = voxelization.compute_grid_size(voxelization.max_range, voxelization.min_range, voxelization.voxel_size);
    voxelization.max_points_per_voxel = 10;
    voxelization.max_points = 300000;
    voxelization.max_voxels = 160000;
    voxelization.num_feature = 5;

    bevfusion::lidar::SCNParameter scn;
    scn.voxelization = voxelization;
    scn.model = nv::format("%s/lidar.backbone.xyz.onnx", model.c_str());
    scn.order = bevfusion::lidar::CoordinateOrder::XYZ;

    if (precision == "int8") {
    scn.precision = bevfusion::lidar::Precision::Int8;
    } else {
    scn.precision = bevfusion::lidar::Precision::Float16;
    }

    bevfusion::camera::GeometryParameter geometry;
    geometry.xbound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
    geometry.ybound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
    geometry.zbound = nvtype::Float3(-10.0f, 10.0f, 20.0f);
    geometry.dbound = nvtype::Float3(1.0, 60.0f, 0.5f);
    geometry.image_width = BEVFUSION_OUTPUT_WIDTH;
    geometry.image_height = BEVFUSION_OUTPUT_HEIGHT;
    geometry.feat_width = 88;
    geometry.feat_height = 32;
    geometry.num_camera = MAX_CAMERAS;
    geometry.geometry_dim = nvtype::Int3(360, 360, 80);

    bevfusion::head::transbbox::TransBBoxParameter transbbox;
    transbbox.out_size_factor = 8;
    transbbox.pc_range = {-54.0f, -54.0f};
    transbbox.post_center_range_start = {-61.2, -61.2, -10.0};
    transbbox.post_center_range_end = {61.2, 61.2, 10.0};
    transbbox.voxel_size = {0.075, 0.075};    transbbox.model = nv::format("%s/build/head.bbox.layernormplugin.plan", model.c_str());
    transbbox.confidence_threshold = 0.12f;
    transbbox.sorted_bboxes = true;


    bevfusion::CoreParameter param;
    param.camera_model = nv::format("%s/build/camera.backbone.plan", model.c_str());
    param.normalize = normalization;
    param.lidar_scn = scn;
    param.geometry = geometry;
    param.transfusion = nv::format("%s/build/fuser.plan", model.c_str());
    param.transbbox = transbbox;
    param.camera_vtransform = nv::format("%s/build/camera.vtransform.plan", model.c_str());
    
    return bevfusion::create_core(param);
    }

    void loadTransformationMatrices()
    {
    // Load transformation matrices from specified example data path
    if (m_exampleDataPath.empty()) {
    logError("Example data path not specified. Using identity matrices.\n");
    initializeIdentityMatrices();
    return;
    }
    
    if (m_enableLogging) {
    log("Loading transformation matrices from: %s\n", m_exampleDataPath.c_str());
    }
    
    try {
    // Load transformation matrices
    m_camera2lidar = nv::Tensor::load(nv::format("%s/camera2lidar.tensor", m_exampleDataPath.c_str()), false);
    m_cameraIntrinsics = nv::Tensor::load(nv::format("%s/camera_intrinsics.tensor", m_exampleDataPath.c_str()), false);
    m_lidar2image = nv::Tensor::load(nv::format("%s/lidar2image.tensor", m_exampleDataPath.c_str()), false);
    m_imgAugMatrix = nv::Tensor::load(nv::format("%s/img_aug_matrix.tensor", m_exampleDataPath.c_str()), false);
    
    log("Transformation matrices loaded successfully\n");
    
    if (m_enableLogging) {
    // Log tensor information
    log("Camera2Lidar: shape=[%ld, %ld, %ld, %ld], dtype=%s, size=%ld bytes\n", 
    m_camera2lidar.shape[0], m_camera2lidar.shape[1], m_camera2lidar.shape[2], m_camera2lidar.shape[3],
    nv::dtype_string(m_camera2lidar.dtype()), m_camera2lidar.bytes());
    
    log("Camera Intrinsics: shape=[%ld, %ld, %ld, %ld], dtype=%s, size=%ld bytes\n",
    m_cameraIntrinsics.shape[0], m_cameraIntrinsics.shape[1], m_cameraIntrinsics.shape[2], m_cameraIntrinsics.shape[3],
    nv::dtype_string(m_cameraIntrinsics.dtype()), m_cameraIntrinsics.bytes());
    
    log("LiDAR2Image: shape=[%ld, %ld, %ld, %ld], dtype=%s, size=%ld bytes\n",
    m_lidar2image.shape[0], m_lidar2image.shape[1], m_lidar2image.shape[2], m_lidar2image.shape[3],
    nv::dtype_string(m_lidar2image.dtype()), m_lidar2image.bytes());
    
    log("Img Aug Matrix: shape=[%ld, %ld, %ld, %ld], dtype=%s, size=%ld bytes\n",
    m_imgAugMatrix.shape[0], m_imgAugMatrix.shape[1], m_imgAugMatrix.shape[2], m_imgAugMatrix.shape[3],
    nv::dtype_string(m_imgAugMatrix.dtype()), m_imgAugMatrix.bytes());
    }
    
    } catch (const std::exception& e) {
    logError("Failed to load transformation matrices from %s: %s\n", m_exampleDataPath.c_str(), e.what());
    logError("Using identity matrices as fallback\n");
    
    // Fallback to identity matrices
    initializeIdentityMatrices();
    }
    }
    
    void initializeIdentityMatrices()
    {
    // Initialize identity transformation matrices as fallback
    std::vector<float> camera2lidar_data(MAX_CAMERAS * 4 * 4, 0.0f);
    std::vector<float> camera_intrinsics_data(MAX_CAMERAS * 4 * 4, 0.0f);
    std::vector<float> lidar2image_data(MAX_CAMERAS * 4 * 4, 0.0f);
    std::vector<float> img_aug_matrix_data(MAX_CAMERAS * 4 * 4, 0.0f);
    
    // Set identity matrices for each camera
    for (int i = 0; i < MAX_CAMERAS; i++) {
    // Camera to LiDAR transformation (identity for now)
    camera2lidar_data[i * 16 + 0] = 1.0f;
    camera2lidar_data[i * 16 + 5] = 1.0f;
    camera2lidar_data[i * 16 + 10] = 1.0f;
    camera2lidar_data[i * 16 + 15] = 1.0f;
    
    // Camera intrinsics (placeholder values)
    camera_intrinsics_data[i * 16 + 0] = 1266.417f; // fx
    camera_intrinsics_data[i * 16 + 5] = 1266.417f; // fy
    camera_intrinsics_data[i * 16 + 10] = 1.0f;
    camera_intrinsics_data[i * 16 + 15] = 1.0f;
    
    // LiDAR to image projection (identity for now)
    lidar2image_data[i * 16 + 0] = 1.0f;
    lidar2image_data[i * 16 + 5] = 1.0f;
    lidar2image_data[i * 16 + 10] = 1.0f;
    lidar2image_data[i * 16 + 15] = 1.0f;
    
    // Image augmentation matrix (identity for now)
    img_aug_matrix_data[i * 16 + 0] = 1.0f;
    img_aug_matrix_data[i * 16 + 5] = 1.0f;
    img_aug_matrix_data[i * 16 + 10] = 1.0f;
    img_aug_matrix_data[i * 16 + 15] = 1.0f;
    }
    
    // Create tensors
    m_camera2lidar = nv::Tensor(std::vector<int>{1, MAX_CAMERAS, 4, 4}, nv::DataType::Float32);
    m_cameraIntrinsics = nv::Tensor(std::vector<int>{1, MAX_CAMERAS, 4, 4}, nv::DataType::Float32);
    m_lidar2image = nv::Tensor(std::vector<int>{1, MAX_CAMERAS, 4, 4}, nv::DataType::Float32);
    m_imgAugMatrix = nv::Tensor(std::vector<int>{1, MAX_CAMERAS, 4, 4}, nv::DataType::Float32);
    
    // Copy data to tensors
    memcpy(m_camera2lidar.ptr<float>(), camera2lidar_data.data(), camera2lidar_data.size() * sizeof(float));
    memcpy(m_cameraIntrinsics.ptr<float>(), camera_intrinsics_data.data(), camera_intrinsics_data.size() * sizeof(float));
    memcpy(m_lidar2image.ptr<float>(), lidar2image_data.data(), lidar2image_data.size() * sizeof(float));
    // Follow CUDA-BEVFusion: img_aug_matrix performs model-space resize 1600x900 -> 704x256
    // Scale: sx = 704/1600, sy = 256/900. Do not apply in GUI projections.
    {
        std::vector<float> aug_data(img_aug_matrix_data.size(), 0.0f);
        const float sx = 704.0f / 1600.0f;
        const float sy = 256.0f / 900.0f;
        for (int i = 0; i < MAX_CAMERAS; ++i) {
            const int base = i * 16;
            aug_data[base + 0]  = sx;   // (0,0)
            aug_data[base + 5]  = sy;   // (1,1)
            aug_data[base + 10] = 1.0f; // (2,2)
            aug_data[base + 15] = 1.0f; // (3,3)
        }
        memcpy(m_imgAugMatrix.ptr<float>(), aug_data.data(), aug_data.size() * sizeof(float));
    }
    
    log("Transformation matrices initialized (img_aug set to 704x256 scale for model)\n");
    }
    
    // Note: saveDetectionResults function removed as requested - visualization is handled in GUI
    
    void updateCameraImages()
    {
    if (m_enableLogging && m_frameCount % 30 == 0) {
    log("Updating camera images for rendering...\n");
    }
    
    for (size_t i = 0; i < MAX_CAMERAS; i++) {
    // Get CUDA image data
    dwImageCUDA* imgCUDA;
    CHECK_DW_ERROR(dwImage_getCUDA(&imgCUDA, m_cameraRGBAImages[i]));
    
    if (i < m_cameras.size() && !m_cameras[i].isBlackImage && m_cameras[i].sensor != DW_NULL_HANDLE) {
    // Try to get real camera data (following NVIDIA camera sample approach)
    dwCameraFrameHandle_t frame;
    dwStatus status = dwSensorCamera_readFrame(&frame, 333333, m_cameras[i].sensor); // 333ms timeout like NVIDIA sample
    
    if (status == DW_SUCCESS) {
    // Get RGBA image from camera (following NVIDIA camera sample)
    dwImageHandle_t rgbaImage;
    dwStatus imgStatus = dwSensorCamera_getImage(&rgbaImage, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, frame);
    
    if (imgStatus == DW_SUCCESS) {
    // Copy camera image to our render image
    dwImageCUDA* cameraImgCUDA;
    CHECK_DW_ERROR(dwImage_getCUDA(&cameraImgCUDA, rgbaImage));
    
    // CRITICAL FIX: Use proper DriveWorks image transformation (following NVIDIA approach)
    if (cameraImgCUDA->prop.width != imgCUDA->prop.width || cameraImgCUDA->prop.height != imgCUDA->prop.height) {
        // Step 1: Resize RGBA to BEVFusion dimensions using DriveWorks API
        CHECK_DW_ERROR(dwImageTransformation_copy(m_cameraRGBAImages[i], rgbaImage, nullptr, nullptr, m_imageTransformer));
        
        // Step 2: Convert RGBA to RGB using DriveWorks API with proper synchronization
        CHECK_DW_ERROR(dwImage_copyConvertAsync(m_cameraRGBImages[i], m_cameraRGBAImages[i], m_cudaStream, m_context));
        
        if (m_enableLogging && m_frameCount % 60 == 0) {
            log("Camera %zu: GPU pipeline - Resized and converted %dx%d→%dx%d\n", i, 
                cameraImgCUDA->prop.width, cameraImgCUDA->prop.height,
                imgCUDA->prop.width, imgCUDA->prop.height);
        }
    } else {
        // Same dimensions: copy and convert using DriveWorks APIs
        // Use dwImageTransformation_copy for same-size copy (following NVIDIA approach)
        CHECK_DW_ERROR(dwImageTransformation_copy(m_cameraRGBAImages[i], rgbaImage, nullptr, nullptr, m_imageTransformer));
        CHECK_DW_ERROR(dwImage_copyConvertAsync(m_cameraRGBImages[i], m_cameraRGBAImages[i], m_cudaStream, m_context));
        
        if (m_enableLogging && m_frameCount % 60 == 0) {
            log("Camera %zu: Direct copy and convert\n", i);
        }
    }
    } else {
    // Create colored placeholder for this camera
    uint8_t color = (i + 1) * 40;
    cudaMemset(imgCUDA->dptr[0], color, imgCUDA->pitch[0] * imgCUDA->prop.height);
    }
    
    // CRITICAL: Always return the frame (following NVIDIA camera sample)
    CHECK_DW_ERROR(dwSensorCamera_returnFrame(&frame));
    } else if (status == DW_END_OF_STREAM) {
    // Reset the sensor to support loopback (following NVIDIA camera sample)
    CHECK_DW_ERROR(dwSensor_reset(m_cameras[i].sensor));
    if (m_enableLogging && m_frameCount % 60 == 0) {
    log("Camera %zu: Video reached end of stream, reset sensor\n", i);
    }
    // Create colored placeholder
    uint8_t color = (i + 1) * 40;
    cudaMemset(imgCUDA->dptr[0], color, imgCUDA->pitch[0] * imgCUDA->prop.height);
    } else {
    // No frame available, create colored placeholder
    uint8_t color = (i + 1) * 40;
    cudaMemset(imgCUDA->dptr[0], color, imgCUDA->pitch[0] * imgCUDA->prop.height);
    if (m_enableLogging && m_frameCount % 60 == 0) {
    log("Camera %zu: No frame available (status: %d), using colored placeholder\n", i, status);
    }
    }
    } else {
    // Black image or no camera available - use proper black initialization
    cudaMemset(imgCUDA->dptr[0], 0, imgCUDA->pitch[0] * imgCUDA->prop.height);
    
    // Also ensure RGB buffer is black for BEVFusion
    dwImageCUDA* rgbImgCUDA;
    CHECK_DW_ERROR(dwImage_getCUDA(&rgbImgCUDA, m_cameraRGBImages[i]));
    cudaMemset(rgbImgCUDA->dptr[0], 0, rgbImgCUDA->pitch[0] * rgbImgCUDA->prop.height);
    }
    }
    
    // CRITICAL: Synchronize CUDA stream after all operations (following NVIDIA pattern)
    cudaStreamSynchronize(m_cudaStream);
    
    if (m_enableLogging && m_frameCount % 30 == 0) {
    log("Camera images updated\n");
    }
    }
    
    void updatePointCloudBufferFromLidar()
    {
    if (m_lidars.empty()) {
    m_pointCloudBufferSize = 0;
    if (m_enableLogging) {
    log("No LiDAR sensors available for point cloud update\n");
    }
    return;
    }
    
    LidarData& lidar = m_lidars[0];
    uint32_t pointCount = lidar.pointCount;
    
    if (pointCount == 0) {
    m_pointCloudBufferSize = 0;
    if (m_enableLogging) {
    log("No points available from LiDAR sensor\n");
    }
    return;
    }
    
    if (m_enableLogging) {
    log("Updating point cloud buffer with %d points from DriveWorks format\n", pointCount);
    }
    
    // Safety check: limit point count to prevent buffer overflow
    if (pointCount > 100000) {
    logWarn("Point count too large (%d), limiting to 100000\n", pointCount);
    pointCount = 100000;
    }
    
    // Ensure we have enough capacity
    if (pointCount > m_pointCloudBufferCapacity) {
    m_pointCloudBufferCapacity = pointCount * 2;
    m_pointCloudViz.reset(new float32_t[m_pointCloudBufferCapacity * 4]);
    if (m_enableLogging) {
    log("Resized point cloud buffer to %d points capacity\n", m_pointCloudBufferCapacity);
    }
    }
    
    // Safety check: ensure buffer is valid
    if (!m_pointCloudViz) {
    logError("Point cloud buffer is null after allocation attempt\n");
    return;
    }
    
    // Simple copy from the DriveWorks format buffer (no conversion needed!)
    if (lidar.pointCloudViz) {
    memcpy(m_pointCloudViz.get(), lidar.pointCloudViz.get(), pointCount * 4 * sizeof(float32_t));
    m_pointCloudBufferSize = pointCount;
    
    if (m_enableLogging) {
    log("Successfully updated point cloud buffer: %d points copied directly\n", pointCount);
    if (pointCount > 0) {
    log("First point: X=%.2f, Y=%.2f, Z=%.2f, I=%.2f\n", 
    m_pointCloudViz[0], m_pointCloudViz[1], m_pointCloudViz[2], m_pointCloudViz[3]);
    }
    }
    } else {
    logError("LiDAR visualization buffer is null\n");
    m_pointCloudBufferSize = 0;
    }
    }
    
    void renderBoundingBoxes()
    {
    // Always render ego vehicle first (independent of detections)
    renderEgoVehicleAlways();
    
    if (m_lastBboxes.empty()) {
    if (m_enableLogging && m_frameCount % 30 == 0) {
    log("No bounding boxes to render\n");
    }
    return;
    }
    
    if (m_enableLogging && m_frameCount % 30 == 0) {
    log("Rendering %zu bounding boxes\n", m_lastBboxes.size());
    }
    
    // Safety check: ensure render engine is valid
    if (m_renderEngine == DW_NULL_HANDLE) {
    logError("Render engine is null\n");
    return;
    }
    
    // Safety check: ensure box line buffer is valid
    if (m_boxLineBuffer == 0) {
    logError("Box line buffer is not initialized\n");
    return;
    }
    
    // Set up 3D transformation for rendering
    dwMatrix4f modelView = *getMouseView().getModelView();
    dwRenderEngine_setModelView(&modelView, m_renderEngine);
    dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine);
    
    // Group boxes by class for separate rendering
    std::vector<bevfusion::head::transbbox::BoundingBox> vehicleBoxes;
    std::vector<bevfusion::head::transbbox::BoundingBox> pedestrianBoxes;
    std::vector<bevfusion::head::transbbox::BoundingBox> cyclistBoxes;
    
    for (const auto& box : m_lastBboxes) {
    // Classify boxes by type based on BEVFusion's actual class IDs
    switch (box.id) {
    case 0: vehicleBoxes.push_back(box); break;      // Vehicle
    case 1: pedestrianBoxes.push_back(box); break;   // Pedestrian  
    case 2: cyclistBoxes.push_back(box); break;      // Cyclist
    default: vehicleBoxes.push_back(box); break;     // Unknown -> treat as vehicle
    }
    }
    
    // Helper function to render a group of boxes with the same color
    auto renderBoxGroup = [&](const std::vector<bevfusion::head::transbbox::BoundingBox>& boxes, dwRenderEngineColorRGBA color, const char* className) {
    if (boxes.empty()) return;
    
    if (m_enableLogging && m_frameCount % 30 == 0) {
    log("Rendering %zu %s boxes\n", boxes.size(), className);
    }
    
    // Safety check: limit number of boxes to prevent buffer overflow
    uint32_t maxBoxes = std::min(static_cast<uint32_t>(boxes.size()), 50u);
    
    // Calculate exact buffer size needed
    uint32_t expectedVertices = maxBoxes * 24; // 12 edges × 2 vertices per edge
    uint32_t bufferSizeBytes = expectedVertices * sizeof(dwVector3f);
    
    // Safety check: ensure buffer size is reasonable
    if (bufferSizeBytes > 1024 * 1024) { // 1MB limit
    logWarn("Buffer size too large (%u bytes), limiting %s boxes to 10\n", bufferSizeBytes, className);
    maxBoxes = 10;
    expectedVertices = maxBoxes * 24;
    bufferSizeBytes = expectedVertices * sizeof(dwVector3f);
    }
    
    // Map the render buffer for this group
    dwVector3f* vertices = nullptr;
    dwStatus status = dwRenderEngine_mapBuffer(m_boxLineBuffer,
    reinterpret_cast<void**>(&vertices),
    0,
    bufferSizeBytes,
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
    m_renderEngine);
    
    if (status != DW_SUCCESS) {
    logError("Failed to map buffer for %s bounding boxes, status: %d\n", className, status);
    return;
    }
    
    if (!vertices) {
    logError("Mapped buffer returned null pointer for %s boxes\n", className);
    return;
    }
    
    uint32_t vertexIndex = 0;
    for (uint32_t i = 0; i < maxBoxes; i++) {
    const auto& box = boxes[i];
    
    // Validate box parameters
    if (!std::isfinite(box.position.x) || !std::isfinite(box.position.y) || !std::isfinite(box.position.z) ||
    !std::isfinite(box.size.w) || !std::isfinite(box.size.l) || !std::isfinite(box.size.h) ||
    !std::isfinite(box.z_rotation)) {
    if (m_enableLogging) {
    logWarn("Skipping invalid %s box %d with non-finite values\n", className, i);
    }
    continue;
    }
    
    // Convert BEVFusion bbox to 3D box vertices
    float x = box.position.x;
    float y = box.position.y;
    float z = box.position.z;
    float w = box.size.w;
    float l = box.size.l;
    float h = box.size.h;
    float yaw = box.z_rotation;
    
    // Skip boxes that are too small or too large
    if (w < 0.1f || l < 0.1f || h < 0.1f || w > 50.0f || l > 50.0f || h > 10.0f) {
    if (m_enableLogging && m_frameCount % 30 == 0) {
    logWarn("Skipping %s box %d with invalid size: w=%.2f, l=%.2f, h=%.2f\n", className, i, w, l, h);
    }
    continue;
    }
    
    // Calculate box corners
    float cos_yaw = cos(yaw);
    float sin_yaw = sin(yaw);
    
    // 8 corners of the box
    std::vector<dwVector3f> corners = {
    {x - l/2*cos_yaw - w/2*sin_yaw, y - l/2*sin_yaw + w/2*cos_yaw, z - h/2},
    {x + l/2*cos_yaw - w/2*sin_yaw, y + l/2*sin_yaw + w/2*cos_yaw, z - h/2},
    {x + l/2*cos_yaw + w/2*sin_yaw, y + l/2*sin_yaw - w/2*cos_yaw, z - h/2},
    {x - l/2*cos_yaw + w/2*sin_yaw, y - l/2*sin_yaw - w/2*cos_yaw, z - h/2},
    {x - l/2*cos_yaw - w/2*sin_yaw, y - l/2*sin_yaw + w/2*cos_yaw, z + h/2},
    {x + l/2*cos_yaw - w/2*sin_yaw, y + l/2*sin_yaw + w/2*cos_yaw, z + h/2},
    {x + l/2*cos_yaw + w/2*sin_yaw, y + l/2*sin_yaw - w/2*cos_yaw, z + h/2},
    {x - l/2*cos_yaw + w/2*sin_yaw, y - l/2*sin_yaw - w/2*cos_yaw, z + h/2}
    };
    
    // Define 12 edges of the box
    std::vector<std::pair<int, int>> edges = {
    {0,1}, {1,2}, {2,3}, {3,0}, // Bottom face
    {4,5}, {5,6}, {6,7}, {7,4}, // Top face
    {0,4}, {1,5}, {2,6}, {3,7} // Vertical edges
    };
    
    // Add edges to vertex buffer with bounds checking
    for (const auto& edge : edges) {
    if (vertexIndex + 1 < expectedVertices) {
    vertices[vertexIndex++] = corners[edge.first];
    vertices[vertexIndex++] = corners[edge.second];
    } else {
    if (m_enableLogging) {
    logWarn("Vertex buffer overflow prevented for %s box %d\n", className, i);
    }
    break;
    }
    }
    }
    
    // Unmap buffer
    dwStatus unmapStatus = dwRenderEngine_unmapBuffer(m_boxLineBuffer, DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D, m_renderEngine);
    if (unmapStatus != DW_SUCCESS) {
    logError("Failed to unmap buffer for %s boxes, status: %d\n", className, unmapStatus);
    return;
    }
    
    // Render the boxes
    dwRenderEngine_setColor(color, m_renderEngine);
    dwStatus renderStatus = dwRenderEngine_renderBuffer(m_boxLineBuffer, vertexIndex, m_renderEngine);
    if (renderStatus != DW_SUCCESS) {
    logError("Failed to render %s boxes, status: %d\n", className, renderStatus);
    } else if (m_enableLogging && m_frameCount % 30 == 0) {
    log("Successfully rendered %d vertices for %s boxes\n", vertexIndex, className);
    }
    };
    
      // Render boxes by class with nuScenes colors (CUDA-BEVFusion defaults)
      // car (vehicleBoxes grouped include id=0): RGB(255,158,0)
      renderBoxGroup(vehicleBoxes, {255.0f/255.0f, 158.0f/255.0f, 0.0f/255.0f, 1.0f}, "vehicle");
      // pedestrian: RGB(0,0,230)
      renderBoxGroup(pedestrianBoxes, {0.0f/255.0f, 0.0f/255.0f, 230.0f/255.0f, 1.0f}, "pedestrian");
      // cyclist (bicycle proxy): RGB(220,20,60)
      renderBoxGroup(cyclistBoxes, {220.0f/255.0f, 20.0f/255.0f, 60.0f/255.0f, 1.0f}, "cyclist");
    } // End of renderBoundingBoxes()
    
    // Always render ego vehicle (independent of detections)
    void renderEgoVehicleAlways() {
        // Ego vehicle dimensions (matching BEVFusion: 1.5m width × 3.0m length × 2.0m height)
        float length = 3.0f;  // Car length (Y-axis)
        float width = 1.5f;   // Car width (X-axis)  
        float height = 2.0f;  // Car height (Z-axis)
        float x = 0.0f, y = 0.0f, z = 0.0f; // At origin (our car's position)
        
        // Calculate 8 corners of the ego vehicle box
        std::vector<dwVector3f> corners = {
            {x - width/2, y - length/2, z - height/2}, // Bottom face
            {x + width/2, y - length/2, z - height/2},
            {x + width/2, y + length/2, z - height/2},
            {x - width/2, y + length/2, z - height/2},
            {x - width/2, y - length/2, z + height/2}, // Top face
            {x + width/2, y - length/2, z + height/2},
            {x + width/2, y + length/2, z + height/2},
            {x - width/2, y + length/2, z + height/2}
        };
        
        // Define 12 edges of the box (same as BEVFusion)
        std::vector<std::pair<int, int>> edges = {
            {0,1}, {1,2}, {2,3}, {3,0}, // Bottom face
            {4,5}, {5,6}, {6,7}, {7,4}, // Top face
            {0,4}, {1,5}, {2,6}, {3,7}  // Vertical edges
        };
        
        // Map render buffer for ego vehicle
        dwVector3f* vertices = nullptr;
        uint32_t expectedVertices = 24; // 12 edges × 2 vertices per edge
        uint32_t bufferSizeBytes = expectedVertices * sizeof(dwVector3f);
        
        dwStatus status = dwRenderEngine_mapBuffer(m_boxLineBuffer,
                                                 reinterpret_cast<void**>(&vertices),
                                                 0,
                                                 bufferSizeBytes,
                                                 DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                                 m_renderEngine);
        
        if (status != DW_SUCCESS) {
            logError("Failed to map buffer for ego vehicle, status: %d\n", status);
            return;
        }
        
        // Add edges to vertex buffer
        uint32_t vertexIndex = 0;
        for (const auto& edge : edges) {
            vertices[vertexIndex++] = corners[edge.first];
            vertices[vertexIndex++] = corners[edge.second];
        }
        
        // Unmap buffer
        dwRenderEngine_unmapBuffer(m_boxLineBuffer, DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D, m_renderEngine);
        
        // Render ego vehicle in bright green (matching BEVFusion)
        dwRenderEngine_setColor({0.0f, 1.0f, 0.0f, 1.0f}, m_renderEngine);
        dwRenderEngine_renderBuffer(m_boxLineBuffer, vertexIndex, m_renderEngine);
        
        if (m_enableLogging && m_frameCount % 120 == 0) {
            log("Rendered ego vehicle at origin (%.1f × %.1f × %.1f m)\n", width, length, height);
        }
    }
    
    void onRelease() override
    {
    // Release BEVFusion core first
    if (m_bevfusionCore) {
    m_bevfusionCore.reset();
    }
    
    // Release image transformation
    if (m_imageTransformer != DW_NULL_HANDLE) {
    dwImageTransformation_release(m_imageTransformer);
    m_imageTransformer = DW_NULL_HANDLE;
    }
    
    // Release CUDA stream
    if (m_cudaStream) {
    checkRuntime(cudaStreamDestroy(m_cudaStream));
    m_cudaStream = 0;
    }
    
    // Release camera resources
    for (auto& camera : m_cameras) {
    if (camera.streamerToCPU != DW_NULL_HANDLE) {
    dwImageStreamer_release(camera.streamerToCPU);
    camera.streamerToCPU = DW_NULL_HANDLE;
    }
    if (camera.currentFrame != DW_NULL_HANDLE) {
    // Note: dwImage handles are managed by DriveWorks, no explicit release needed
    camera.currentFrame = DW_NULL_HANDLE;
    }
    if (camera.imageData) {
    delete[] camera.imageData;
    camera.imageData = nullptr;
    }
    }
    
    // Release LiDAR resources
    for (auto& lidar : m_lidars) {
    if (lidar.sensor != DW_NULL_HANDLE) {
    dwSAL_releaseSensor(lidar.sensor);
    lidar.sensor = DW_NULL_HANDLE;
    }
    }
    
    // Release camera resources
    for (uint32_t i = 0; i < MAX_CAMERAS; i++) {
    if (m_streamerToGL[i] != DW_NULL_HANDLE) {
    dwImageStreamerGL_release(m_streamerToGL[i]);
    m_streamerToGL[i] = DW_NULL_HANDLE;
    }
    if (m_cameraRGBAImages[i] != DW_NULL_HANDLE) {
    dwImage_destroy(m_cameraRGBAImages[i]);
    m_cameraRGBAImages[i] = DW_NULL_HANDLE;
    }
    if (m_cameraRGBImages[i] != DW_NULL_HANDLE) {
    dwImage_destroy(m_cameraRGBImages[i]);
    m_cameraRGBImages[i] = DW_NULL_HANDLE;
    }
    }
    
    if (m_sensorManager != DW_NULL_HANDLE) {
    dwSensorManager_stop(m_sensorManager);
    dwSensorManager_release(m_sensorManager);
    m_sensorManager = DW_NULL_HANDLE;
    }
    
    if (m_rigConfig != DW_NULL_HANDLE) {
    dwRig_release(m_rigConfig);
    m_rigConfig = DW_NULL_HANDLE;
    }
    
    if (m_sal != DW_NULL_HANDLE) {
    dwSAL_release(m_sal);
    m_sal = DW_NULL_HANDLE;
    }
    
    if (m_renderEngine != DW_NULL_HANDLE) {
    // Release visualization buffers
    if (m_pointCloudBuffer != 0) {
    dwRenderEngine_destroyBuffer(m_pointCloudBuffer, m_renderEngine);
    m_pointCloudBuffer = 0;
    }
    if (m_boxLineBuffer != 0) {
    dwRenderEngine_destroyBuffer(m_boxLineBuffer, m_renderEngine);
    m_boxLineBuffer = 0;
    }
    
    dwRenderEngine_release(m_renderEngine);
    m_renderEngine = DW_NULL_HANDLE;
    }
    
    if (m_viz != DW_NULL_HANDLE) {
    dwVisualizationRelease(m_viz);
    m_viz = DW_NULL_HANDLE;
    }
    
    if (m_context != DW_NULL_HANDLE) {
    dwRelease(m_context);
    m_context = DW_NULL_HANDLE;
    }
    
    m_screenshot.reset();
    
    // Release logger
    dwLogger_release();
    }

    bool onInitialize() override
    {
    log("=== Initializing BEVFusion DriveWorks Application ===\n");
    
    try {
    log("Step 1: Initializing DriveWorks...\n");
    initializeDriveWorks(m_context);
    log("Step 1 completed successfully\n");
    
    log("Step 2: Initializing sensors...\n");
    initializeSensors();
    log("Step 2 completed successfully\n");
    
    if (m_cameraOnly) {
    log("Step 3: Skipping BEVFusion (camera-only mode)...\n");
    log("Step 3 skipped\n");
    } else {
    log("Step 3: Initializing BEVFusion...\n");
    initializeBEVFusion();
    log("Step 3 completed successfully\n");
    }
    
    // Initialize visualization
    log("Step 4: Initializing visualization...\n");
    CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_context));
    log("Visualization context initialized\n");
    
    log("Initializing render engine...\n");
    dwRenderEngineParams renderParams;
    CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&renderParams, getWindowWidth(), getWindowHeight()));
    renderParams.defaultTile.backgroundColor = {0.0f, 0.0f, 0.1f, 1.0f};
    CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &renderParams, m_viz));
    log("Render engine initialized\n");
    
    // Create render tiles - main 3D view and 6 camera tiles
    log("Creating render tiles...\n");
    
    // Main 3D LiDAR view tile (center column)
    dwRenderEngineTileState tileState;
    dwRenderEngine_initTileState(&tileState);
    tileState.layout.viewport = {0.33f, 0.0f, 0.34f, 1.0f}; // Center 34% of screen
    CHECK_DW_ERROR(dwRenderEngine_addTile(&m_tileVideo, &tileState, m_renderEngine));
    log("Main 3D view tile: Center (0.33, 0.0, 0.34, 1.0)\n");
    
    // Manual camera tile placement
    // Camera 0 - Top Left Corner
    dwRenderEngine_initTileState(&tileState);
    tileState.layout.sizeLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_RELATIVE;
    tileState.layout.positionLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_RELATIVE;
    tileState.layout.positionType = DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_LEFT;
    tileState.layout.viewport = {0.0f, 0.0f, 0.33f, 0.33f}; // x, y, width, height
    CHECK_DW_ERROR(dwRenderEngine_addTile(&m_tileCameras[0], &tileState, m_renderEngine));
    log("Camera 0 tile: Top Left Corner (0.0, 0.0, 0.33, 0.33)\n");

    // Camera 1 - Middle Left
    dwRenderEngine_initTileState(&tileState);
    tileState.layout.sizeLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_RELATIVE;
    tileState.layout.positionLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_RELATIVE;
    tileState.layout.positionType = DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_LEFT;
    tileState.layout.viewport = {0.0f, 0.33f, 0.33f, 0.33f};
    CHECK_DW_ERROR(dwRenderEngine_addTile(&m_tileCameras[1], &tileState, m_renderEngine));
    log("Camera 1 tile: Middle Left (0.0, 0.33, 0.33, 0.33)\n");

    // Camera 2 - Bottom Left
    dwRenderEngine_initTileState(&tileState);
    tileState.layout.sizeLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_RELATIVE;
    tileState.layout.positionLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_RELATIVE;
    tileState.layout.positionType = DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_LEFT;
    tileState.layout.viewport = {0.0f, 0.67f, 0.33f, 0.33f};
    CHECK_DW_ERROR(dwRenderEngine_addTile(&m_tileCameras[2], &tileState, m_renderEngine));
    log("Camera 2 tile: Bottom Left (0.0, 0.67, 0.33, 0.33)\n");

    // Camera 3 - Top Right
    dwRenderEngine_initTileState(&tileState);
    tileState.layout.sizeLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_RELATIVE;
    tileState.layout.positionLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_RELATIVE;
    tileState.layout.positionType = DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_LEFT;
    tileState.layout.viewport = {0.67f, 0.0f, 0.33f, 0.33f};
    CHECK_DW_ERROR(dwRenderEngine_addTile(&m_tileCameras[3], &tileState, m_renderEngine));
    log("Camera 3 tile: Top Right (0.67, 0.0, 0.33, 0.33)\n");

    // Camera 4 - Middle Right
    dwRenderEngine_initTileState(&tileState);
    tileState.layout.sizeLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_RELATIVE;
    tileState.layout.positionLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_RELATIVE;
    tileState.layout.positionType = DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_LEFT;
    tileState.layout.viewport = {0.67f, 0.33f, 0.33f, 0.33f};
    CHECK_DW_ERROR(dwRenderEngine_addTile(&m_tileCameras[4], &tileState, m_renderEngine));
    log("Camera 4 tile: Middle Right (0.67, 0.33, 0.33, 0.33)\n");

    // Camera 5 - Bottom Right
    dwRenderEngine_initTileState(&tileState);
    tileState.layout.sizeLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_RELATIVE;
    tileState.layout.positionLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_RELATIVE;
    tileState.layout.positionType = DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_LEFT;
    tileState.layout.viewport = {0.67f, 0.67f, 0.33f, 0.33f};
    CHECK_DW_ERROR(dwRenderEngine_addTile(&m_tileCameras[5], &tileState, m_renderEngine));
    log("Camera 5 tile: Bottom Right (0.67, 0.67, 0.33, 0.33)\n");
    
    // Initialize point cloud visualization buffer
    log("Initializing point cloud visualization buffer...\n");
    m_pointCloudBufferCapacity = 500000; // Large enough for LiDAR data
    m_pointCloudViz.reset(new float32_t[m_pointCloudBufferCapacity * 4]); // XYZI format
    
    // Note: Transformation matrices (including lidar2image) are already loaded by loadTransformationMatrices()
    log("Point cloud buffer allocated with capacity: %d points\n", m_pointCloudBufferCapacity);
    
    // Initialize bounding box rendering buffer
    log("Initializing bounding box rendering buffer...\n");
    // Each box has 12 edges, each edge has 2 vertices
    uint32_t maxBoxes = 50; // Maximum number of boxes we expect to render
    uint32_t verticesPerBox = 24; // 12 edges * 2 vertices per edge
    uint32_t totalVertices = maxBoxes * verticesPerBox;
    
    CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_boxLineBuffer, 
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
    sizeof(dwVector3f), 
    0, 
    totalVertices, 
    m_renderEngine));
    log("Bounding box buffer created with capacity: %d vertices\n", totalVertices);
    
    // Initialize image transformation for resizing
    log("Initializing image transformation...\n");
    dwImageTransformationParameters imgTransformParams{};
    CHECK_DW_ERROR(dwImageTransformation_initialize(&m_imageTransformer, imgTransformParams, m_context));
    CHECK_DW_ERROR(dwImageTransformation_setCUDAStream(m_cudaStream, m_imageTransformer));
    log("Image transformation initialized\n");
    if (m_enableLogging) {
    log("ImageTransformation ready: default params, stream set.\n");
    }
    
    // Initialize screenshot helper
    log("Initializing screenshot helper...\n");
    m_screenshot.reset(new ScreenshotHelper(m_context, m_sal, getWindowWidth(), getWindowHeight(), "BEVFusionDriveWorks"));
    log("Screenshot helper initialized\n");
    
    // Initialize camera image streamers (following NVIDIA camera sample)
    log("Initializing camera streamers...\n");
    if (m_cameraOnly) {
    // Camera-only mode: use NVIDIA camera sample approach
    dwImageProperties imageProperties{};
    for (uint32_t i = 0; i < m_cameras.size(); i++) {
    if (m_cameras[i].sensor != DW_NULL_HANDLE) {
    log("Getting image properties for camera %d\n", i);
    CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&imageProperties, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, m_cameras[i].sensor));
    
    log("Initializing GL streamer for camera %d\n", i);
    CHECK_DW_ERROR(dwImageStreamerGL_initialize(&m_streamerToGL[i], &imageProperties, DW_IMAGE_GL, m_context));
    
    // Create RGBA image for camera rendering
    dwImageProperties cudaProps = {};
    cudaProps.type = DW_IMAGE_CUDA;
    cudaProps.format = DW_IMAGE_FORMAT_RGBA_UINT8;
    cudaProps.width = imageProperties.width;
    cudaProps.height = imageProperties.height;
    CHECK_DW_ERROR(dwImage_create(&m_cameraRGBAImages[i], cudaProps, m_context));
    if (m_enableLogging) {
    log("Camera %d RGBA buffer created: %dx%d\n", i, (int)cudaProps.width, (int)cudaProps.height);
    }
    
    log("Camera %d streamer initialized with %dx%d\n", i, imageProperties.width, imageProperties.height);
    } else {
    // Black image camera
    dwImageProperties imageProps = {};
    imageProps.type = DW_IMAGE_CUDA;
    imageProps.format = DW_IMAGE_FORMAT_RGBA_UINT8;
    imageProps.width = BEVFUSION_IMAGE_WIDTH;
    imageProps.height = BEVFUSION_IMAGE_HEIGHT;
    
    CHECK_DW_ERROR(dwImage_create(&m_cameraRGBAImages[i], imageProps, m_context));
    if (m_enableLogging) {
    log("Camera %d RGBA buffer created (black): %dx%d\n", i, (int)imageProps.width, (int)imageProps.height);
    }
    CHECK_DW_ERROR(dwImageStreamerGL_initialize(&m_streamerToGL[i], &imageProps, DW_IMAGE_GL, m_context));
    
    log("Camera %d (black image) streamer initialized with %dx%d\n", i, imageProps.width, imageProps.height);
    }
    }
    } else {
    // BEVFusion mode: use consistent BEVFusion dimensions for all cameras
    for (uint32_t i = 0; i < MAX_CAMERAS; i++) {
    dwImageProperties imageProps = {};
    imageProps.type = DW_IMAGE_CUDA;
    imageProps.format = DW_IMAGE_FORMAT_RGBA_UINT8;
    
    // Use BEVFusion dimensions for all cameras (consistent with display pipeline)
    imageProps.width = BEVFUSION_IMAGE_WIDTH;
    imageProps.height = BEVFUSION_IMAGE_HEIGHT;
    log("Camera %d: Using BEVFusion dimensions %dx%d (BEVFusion mode)\n", i, imageProps.width, imageProps.height);
    
    // Check if this is a black image camera
    bool isBlackImageCamera = (i < m_cameras.size()) ? m_cameras[i].isBlackImage : true;
    
    CHECK_DW_ERROR(dwImage_create(&m_cameraRGBAImages[i], imageProps, m_context));
    if (m_enableLogging) {
    log("Camera %d RGBA buffer created: %dx%d (black=%s)\n", i, (int)imageProps.width, (int)imageProps.height,
        isBlackImageCamera ? "true" : "false");
    }
    
    // Create RGB image for BEVFusion processing
    dwImageProperties rgbProps = imageProps;
    rgbProps.format = DW_IMAGE_FORMAT_RGB_UINT8;
    CHECK_DW_ERROR(dwImage_create(&m_cameraRGBImages[i], rgbProps, m_context));
    if (m_enableLogging) {
    log("Camera %d RGB buffer created: %dx%d\n", i, (int)rgbProps.width, (int)rgbProps.height);
    }
    
    // Initialize GL streamer
    CHECK_DW_ERROR(dwImageStreamerGL_initialize(&m_streamerToGL[i], &imageProps, DW_IMAGE_GL, m_context));
    
    // CRITICAL FIX: Initialize black image buffers properly for black image cameras
    if (isBlackImageCamera) {
        // Fill RGBA buffer with black pixels
        dwImageCUDA* rgbaImgCUDA;
        CHECK_DW_ERROR(dwImage_getCUDA(&rgbaImgCUDA, m_cameraRGBAImages[i]));
        cudaMemset(rgbaImgCUDA->dptr[0], 0, rgbaImgCUDA->pitch[0] * rgbaImgCUDA->prop.height);
        
        // Fill RGB buffer with black pixels  
        dwImageCUDA* rgbImgCUDA;
        CHECK_DW_ERROR(dwImage_getCUDA(&rgbImgCUDA, m_cameraRGBImages[i]));
        cudaMemset(rgbImgCUDA->dptr[0], 0, rgbImgCUDA->pitch[0] * rgbImgCUDA->prop.height);
        
        // Synchronize to ensure initialization is complete
        cudaStreamSynchronize(m_cudaStream);
        
        if (m_enableLogging) {
            log("Camera %d: Initialized black image buffers\n", i);
        }
    }
    
    log("Camera %d streamer initialized with %dx%d\n", i, imageProps.width, imageProps.height);
    }
    }
    
    log("=== BEVFusion DriveWorks Application initialized successfully ===\n");
    
    // Apply process rate if provided
    if (m_processRateHz > 0.0f) {
    setProcessRate(m_processRateHz);
    log("Process rate set to %.1f Hz\n", m_processRateHz);
    } else {
    // Uncapped - rely on render loop
    log("Process rate uncapped (0.0 Hz)\n");
    }
    
    // Start cameras (following NVIDIA camera sample)
    if (m_cameraOnly) {
    log("Starting cameras (camera-only mode)...\n");
    for (uint32_t i = 0; i < m_cameras.size(); i++) {
    if (m_cameras[i].sensor != DW_NULL_HANDLE) {
    CHECK_DW_ERROR(dwSensor_start(m_cameras[i].sensor));
    log("Camera %d started\n", i);
    }
    }
    if (m_stabilizeSkipFrames > 0) {
    log("Stabilization: Skipping first %u frames for sensor stability\n", m_stabilizeSkipFrames);
    }
    }
    
    return true;
    
    } catch (const std::exception& e) {
    logError("Exception during initialization: %s\n", e.what());
    return false;
    } catch (...) {
    logError("Unknown exception during initialization\n");
    return false;
    }
    }

    void onProcess() override
    {
    // Process sensor data and run BEVFusion inference
    processSensorData();
    }

    void processSensorData()
    {
    if (m_cameraOnly) {
    // Optional stabilization skip controlled by CLI
    if (m_stabilizeSkipFrames > 0 && m_frameCount < m_stabilizeSkipFrames) {
    if (m_enableLogging && (m_frameCount % 10 == 0)) {
    log("Stabilization: Skipping frame %d/%u\n", m_frameCount + 1, m_stabilizeSkipFrames);
    }
    m_frameCount++;
    return;
    }
    
    // Camera-only mode: just update camera images for rendering
    updateCameraImages();
    m_frameCount++;
    return;
    }
    
    if (m_enableLogging) {
    log("=== Starting sensor data processing for frame %d ===\n", m_frameCount + 1);
    }
    
    // Clear previous camera projections at start of new frame processing
    for (auto& cameraProj : m_cameraProjections) {
        cameraProj.clear();
    }
    
    std::vector<unsigned char*> cameraImages(MAX_CAMERAS, nullptr);
    nv::Tensor lidarPoints;
    
    // Get camera images and reorder for BEVFusion
    if (m_enableLogging) {
    log("Processing camera data and reordering for BEVFusion...\n");
    }
    
    // BEVFusion expects: 0=FRONT, 1=FRONT_RIGHT, 2=FRONT_LEFT, 3=BACK, 4=BACK_LEFT, 5=BACK_RIGHT
    // Simple reordering: map your rig cameras to BEVFusion indices
    std::vector<std::unique_ptr<unsigned char[]>> tempBlackImages;
    size_t imageSize = BEVFUSION_IMAGE_WIDTH * BEVFUSION_IMAGE_HEIGHT * 3;
    
    // Create black images for missing cameras
    for (int i = 0; i < 6; i++) {
        auto blackImage = std::make_unique<unsigned char[]>(imageSize);
        memset(blackImage.get(), 0, imageSize);
        tempBlackImages.push_back(std::move(blackImage));
    }
    
    // BEVFusion camera mapping based on your 4camera_1lidar.json:
    // Your rig: 0=front_left, 1=front_right, 2=back_left, 3=back_right
    
    // BEVFusion index 0 (FRONT) - black image (not in your rig)
    cameraImages[0] = tempBlackImages[0].get();
    
    // BEVFusion index 1 (FRONT_RIGHT) <- rig 1 (front_right)
    cameraImages[1] = (m_cameras.size() > 1 && !m_cameras[1].isBlackImage) ? getCameraFrame(1) : tempBlackImages[1].get();
    // BEVFusion index 2 (FRONT_LEFT)  <- rig 0 (front_left)
    cameraImages[2] = (m_cameras.size() > 0 && !m_cameras[0].isBlackImage) ? getCameraFrame(0) : tempBlackImages[2].get();
    // BEVFusion index 3 (BACK) <- black
    cameraImages[3] = tempBlackImages[3].get();
    // BEVFusion index 4 (BACK_LEFT)   <- rig 2 (back_left)
    cameraImages[4] = (m_cameras.size() > 2 && !m_cameras[2].isBlackImage) ? getCameraFrame(2) : tempBlackImages[4].get();
    // BEVFusion index 5 (BACK_RIGHT)  <- rig 3 (back_right)
    cameraImages[5] = (m_cameras.size() > 3 && !m_cameras[3].isBlackImage) ? getCameraFrame(3) : tempBlackImages[5].get();
    
    // Get LiDAR data
    if (m_enableLogging) {
    log("Processing LiDAR data...\n");
    }
    
    if (!m_lidars.empty()) {
    lidarPoints = getLidarPoints();
    if (lidarPoints.size(0) == 0) {
    logWarn("LiDAR returned 0 points, using empty tensor\n");
    lidarPoints = nv::Tensor(std::vector<int>{0, 5}, nv::DataType::Float16);
    } else {
    if (m_enableLogging) {
    log("LiDAR: Successfully got %ld points\n", lidarPoints.size(0));
    }
    }
    } else {
    // Create empty LiDAR tensor
    lidarPoints = nv::Tensor(std::vector<int>{0, 5}, nv::DataType::Float16);
    if (m_enableLogging) {
    log("LiDAR: No LiDAR available, using empty tensor\n");
    }
    }
    
    // Run BEVFusion inference
    if (m_enableLogging) {
    log("Running BEVFusion inference...\n");
    for (size_t ci = 0; ci < MAX_CAMERAS; ++ci) {
    dwImageCUDA* dbgRGBA = nullptr;
    dwImageCUDA* dbgRGB = nullptr;
    dwImage_getCUDA(&dbgRGBA, m_cameraRGBAImages[ci]);
    dwImage_getCUDA(&dbgRGB, m_cameraRGBImages[ci]);
    if (dbgRGBA && dbgRGB) {
    log("Cam%zu buffers: RGBA %dx%d pitch=%d | RGB %dx%d pitch=%d\n", ci,
        dbgRGBA->prop.width, dbgRGBA->prop.height, dbgRGBA->pitch[0],
        dbgRGB->prop.width, dbgRGB->prop.height, dbgRGB->pitch[0]);
    }
    }
    }
    
    m_timer.start();
    
    // Safety check: ensure all camera images are valid
    for (size_t i = 0; i < MAX_CAMERAS; i++) {
    if (cameraImages[i] == nullptr) {
    logError("Camera %zu image is null, cannot proceed with inference\n", i);
    return;
    }
    }
    if (m_enableLogging) {
    log("All %d camera inputs present. Lidar points: %ld. Proceeding.\n", MAX_CAMERAS, lidarPoints.size(0));
    }
    
        if (m_enableLogging) {
        log("All camera images validated, proceeding with BEVFusion forward pass...\n");
    }
    
    auto bboxes = m_bevfusionCore->forward((const unsigned char**)cameraImages.data(), 
                                          lidarPoints.ptr<nvtype::half>(), 
                                          lidarPoints.size(0), 
                                          m_cudaStream);
    m_timer.stop();
    m_inferenceTime = m_timer.getTime();
    
    if (m_enableLogging) {
    log("BEVFusion inference completed in %.2f ms, detected %zu objects\n", m_inferenceTime, bboxes.size());
    }
    
    // Store bounding boxes for visualization
    m_lastBboxes = bboxes;
    
    // RAW MODEL VALUES OUTPUT (following NVIDIA BEVFusion style)
    if (m_enableLogging && !bboxes.empty()) {
        log("RAW MODEL VALUES: bboxes.size()=%zu\n", bboxes.size());
        for (size_t i = 0; i < bboxes.size(); i++) {
            const auto& bbox = bboxes[i];
            log("RAW MODEL VALUES [%zu]: pos.x=%.6f, pos.y=%.6f, pos.z=%.6f, size.w=%.6f, size.l=%.6f, size.h=%.6f, z_rotation=%.6f, score=%.6f, id=%d, vel.vx=%.6f, vel.vy=%.6f\n", 
                i, bbox.position.x, bbox.position.y, bbox.position.z, 
                bbox.size.w, bbox.size.l, bbox.size.h, 
                bbox.z_rotation, bbox.score, bbox.id, 
                bbox.velocity.vx, bbox.velocity.vy);
        }
    }
    
    // Project to all cameras
    for (int i = 0; i < MAX_CAMERAS; i++) {
        if (m_enableLogging && !bboxes.empty()) {
            log("About to call projectBoundingBoxesToCamera for camera %d with %zu bboxes\n", i, bboxes.size());
        }
        projectBoundingBoxesToCamera(i, bboxes);
    }
    
    // Update point cloud buffer for visualization using the safe DriveWorks format
    if (m_enableLogging) {
    log("Updating point cloud buffer from LiDAR DriveWorks format...\n");
    }
    updatePointCloudBufferFromLidar();
    
    // Update camera images for rendering
    updateCameraImages();
    
    // Clean up temporary black images (automatic with unique_ptr)
    
    m_frameCount++;
    
    // Log performance every 30 frames
    if (m_frameCount % 30 == 0) {
    log("Frame %d: BEVFusion inference time: %.2f ms\n", m_frameCount, m_inferenceTime);
    }
    
    // Note: Data saving removed as requested - visualization is handled in GUI
    }

    unsigned char* getCameraFrame(size_t cameraIndex)
    {
    if (cameraIndex >= m_cameras.size()) {
        return nullptr;
    }
    
    CameraData& camera = m_cameras[cameraIndex];
    
    // If this is a black image camera or no real sensor, return black image
    if (camera.isBlackImage || camera.sensor == DW_NULL_HANDLE) {
    return camera.imageData;
    }
    
    // CRITICAL FIX: Use proper DriveWorks streaming pattern (following NVIDIA camera sample)
    // Don't read frames here - frames are already read in updateCameraImages()
    // Just return the processed CPU data using proper streaming
    
    // Allocate camera image data if needed
    if (!camera.imageData) {
    size_t imageSize = BEVFUSION_IMAGE_WIDTH * BEVFUSION_IMAGE_HEIGHT * 3;
    camera.imageData = new unsigned char[imageSize];
    }
    
    // Use proper DriveWorks streaming pattern to get CPU data (following NVIDIA approach)
    if (camera.streamerToCPU != DW_NULL_HANDLE) {
        // Stream RGB image to CPU domain (following NVIDIA camera sample)
        dwStatus streamStatus = dwImageStreamer_producerSend(m_cameraRGBImages[cameraIndex], camera.streamerToCPU);
        if (streamStatus == DW_SUCCESS) {
            // Receive streamed image as CPU handle
            dwImageHandle_t frameCPU;
            dwStatus receiveStatus = dwImageStreamer_consumerReceive(&frameCPU, 33000, camera.streamerToCPU);
            if (receiveStatus == DW_SUCCESS) {
                // Get CPU image data (following NVIDIA camera sample)
                dwImageCPU* imgCPU;
                dwStatus getCPUStatus = dwImage_getCPU(&imgCPU, frameCPU);
                if (getCPUStatus == DW_SUCCESS && imgCPU->data[0]) {
                    // Copy CPU data safely (following NVIDIA approach)
                    size_t rowSize = BEVFUSION_IMAGE_WIDTH * 3; // RGB: 3 bytes per pixel
                    for (uint32_t row = 0; row < BEVFUSION_IMAGE_HEIGHT; row++) {
                        unsigned char* srcRow = static_cast<unsigned char*>(imgCPU->data[0]) + (row * imgCPU->pitch[0]);
                        unsigned char* dstRow = camera.imageData + (row * rowSize);
                        memcpy(dstRow, srcRow, rowSize);
                    }
                    
                    if (m_enableLogging && m_frameCount % 30 == 0) {
                        log("getCameraFrame(%zu): CPU streaming successful\n", cameraIndex);
                    }
                }
                
                // Return CPU image (following NVIDIA camera sample)
                dwImageStreamer_consumerReturn(&frameCPU, camera.streamerToCPU);
            }
            
            // Return to producer (following NVIDIA camera sample)
            dwImageStreamer_producerReturn(nullptr, 33000, camera.streamerToCPU);
        }
    } else {
        // Fallback: fill with black data
        size_t imageSize = BEVFUSION_IMAGE_WIDTH * BEVFUSION_IMAGE_HEIGHT * 3;
        memset(camera.imageData, 0, imageSize);
        if (m_enableLogging && m_frameCount % 30 == 0) {
            logWarn("getCameraFrame(%zu): No CPU streamer available, using black fallback\n", cameraIndex);
        }
    }
    
    return camera.imageData;
    }

    nv::Tensor getLidarPoints()
    {
    if (m_enableLogging && m_frameCount % 30 == 0) {
    log("Getting LiDAR points...\n");
    }
    
    if (m_lidars.empty()) {
    if (m_enableLogging) {
    log("No LiDAR sensors available\n");
    }
    return nv::Tensor(std::vector<int>{0, 5}, nv::DataType::Float16);
    }
    
    LidarData& lidar = m_lidars[0];
    
    if (m_enableLogging && m_frameCount % 30 == 0) {
    log("Processing LiDAR data from sensor: %s\n", lidar.name.c_str());
    }
    
    // Accumulate packets until we have a complete scan (like lidar_live_replay)
    uint32_t accumulatedPoints = 0;
    uint32_t packetCount = 0;
    const dwLidarDecodedPacket* packet;
    
    while (true) {
    dwStatus status = dwSensorLidar_readPacket(&packet, 100000, lidar.sensor);
    
    if (status == DW_SUCCESS) {
    packetCount++;
    
    // Append the packet to both buffers simultaneously
    float32_t* bevMap = &lidar.pointCloud[accumulatedPoints * 5];      // BEVFusion buffer
    float32_t* vizMap = &lidar.pointCloudViz[accumulatedPoints * 4];   // Visualization buffer
    
    // Convert XYZI to both BEVFusion format (XYZI + timestamp) and DriveWorks format (XYZI)
    for (uint32_t i = 0; i < packet->nPoints; i++) {
    const dwLidarPointXYZI& srcPoint = packet->pointsXYZI[i];
    
    // BEVFusion format (5 elements)
    float32_t* bevPoint = &bevMap[i * 5];
    bevPoint[0] = srcPoint.x;
    bevPoint[1] = srcPoint.y;
    bevPoint[2] = srcPoint.z;
    bevPoint[3] = srcPoint.intensity;
    bevPoint[4] = 0.0f; // timestamp placeholder
    
    // DriveWorks visualization format (4 elements)
    float32_t* vizPoint = &vizMap[i * 4];
    vizPoint[0] = srcPoint.x;
    vizPoint[1] = srcPoint.y;
    vizPoint[2] = srcPoint.z;
    vizPoint[3] = srcPoint.intensity;
    }
    
    accumulatedPoints += packet->nPoints;
    
    // If we go beyond a full spin, process the accumulated data
    if (packet->scanComplete) {
    if (m_enableLogging && m_frameCount % 10 == 0) {
    log("LiDAR: Complete scan with %d points from %d packets\n", accumulatedPoints, packetCount);
    }
    
    // Return packet
    CHECK_DW_ERROR(dwSensorLidar_returnPacket(packet, lidar.sensor));
    
    // Limit to max points if needed
    if (accumulatedPoints > lidar.maxPoints) {
    if (m_enableLogging) {
    logWarn("LiDAR: Point count %d exceeds max capacity %d, limiting\n", accumulatedPoints, lidar.maxPoints);
    }
    accumulatedPoints = lidar.maxPoints;
    }
    
    if (accumulatedPoints == 0) {
    if (m_enableLogging) {
    logWarn("LiDAR: Complete scan but no points accumulated\n");
    }
    return nv::Tensor(std::vector<int>{0, 5}, nv::DataType::Float16);
    }
    
    // Create tensor
    if (m_enableLogging && m_frameCount % 30 == 0) {
    log("Creating tensor with %d points\n", accumulatedPoints);
    }
    
    try {
    // Store point count for visualization
    lidar.pointCount = accumulatedPoints;
    
    // Create BEVFusion tensor from the BEVFusion buffer
    nv::Tensor points(std::vector<int>{static_cast<int>(accumulatedPoints), 5}, nv::DataType::Float16);
    points.copy_from_host(lidar.pointCloud.get(), m_cudaStream);
    
    if (m_enableLogging && m_frameCount % 30 == 0) {
    log("Successfully created LiDAR tensor with %d points (both formats ready)\n", accumulatedPoints);
    }
    
    return points;
    } catch (const std::exception& e) {
    logError("Exception creating LiDAR tensor: %s\n", e.what());
    lidar.pointCount = 0;
    return nv::Tensor(std::vector<int>{0, 5}, nv::DataType::Float16);
    }
    }
    
    // Return packet and continue accumulating
    CHECK_DW_ERROR(dwSensorLidar_returnPacket(packet, lidar.sensor));
    }
    else if (status == DW_END_OF_STREAM) {
    if (m_enableLogging) {
    log("LiDAR: End of stream, processing %d accumulated points\n", accumulatedPoints);
    }
    
    // For recorded data, start over at the end of the file
    dwSensor_reset(lidar.sensor);
    
    if (accumulatedPoints > 0) {
    // Store point count for visualization
    lidar.pointCount = accumulatedPoints;
    
    // Create tensor with accumulated points
    nv::Tensor points(std::vector<int>{static_cast<int>(accumulatedPoints), 5}, nv::DataType::Float16);
    points.copy_from_host(lidar.pointCloud.get(), m_cudaStream);
    return points;
    } else {
    lidar.pointCount = 0;
    return nv::Tensor(std::vector<int>{0, 5}, nv::DataType::Float16);
    }
    }
    else if (status == DW_TIME_OUT) {
    if (m_enableLogging) {
    log("LiDAR: Read timeout, processing %d accumulated points\n", accumulatedPoints);
    }
    
    if (accumulatedPoints > 0) {
    // Store point count for visualization
    lidar.pointCount = accumulatedPoints;
    
    // Create tensor with accumulated points
    nv::Tensor points(std::vector<int>{static_cast<int>(accumulatedPoints), 5}, nv::DataType::Float16);
    points.copy_from_host(lidar.pointCloud.get(), m_cudaStream);
    return points;
    } else {
    lidar.pointCount = 0;
    return nv::Tensor(std::vector<int>{0, 5}, nv::DataType::Float16);
    }
    }
    else {
    logError("LiDAR: Failed to read packet, status: %d\n", status);
    lidar.pointCount = 0;
    return nv::Tensor(std::vector<int>{0, 5}, nv::DataType::Float16);
    }
    }
    }

    void onRender() override
    {
    if (m_enableLogging && m_frameCount % 30 == 0) {
    log("=== Rendering frame %d ===\n", m_frameCount);
    }
    
    dwRenderEngine_reset(m_renderEngine);
    dwRenderEngine_setTile(m_tileVideo, m_renderEngine);
    
    // Set 3D view for point cloud visualization
    dwRenderEngine_setModelView(getMouseView().getModelView(), m_renderEngine);
    dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine);
    
    // Set background color
    dwRenderEngine_setBackgroundColor({0.0f, 0.0f, 0.1f, 1.0f}, m_renderEngine);
    
    // Render point cloud if available
    if (m_pointCloudBufferSize > 0 && m_pointCloudViz) {
    if (m_enableLogging && m_frameCount % 30 == 0) {
    log("Rendering point cloud with %d points\n", m_pointCloudBufferSize);
    }
    
    try {
    // Create point cloud buffer if not exists
    if (m_pointCloudBuffer == 0) {
    if (m_enableLogging) {
    log("Creating point cloud render buffer with capacity %d\n", m_pointCloudBufferCapacity);
    }
    CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_pointCloudBuffer,
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
    sizeof(float32_t) * 4, // XYZI stride
    0,
    m_pointCloudBufferCapacity,
    m_renderEngine));
    if (m_enableLogging) {
    log("Point cloud render buffer created successfully\n");
    }
    }
    
    // Update buffer with current data
    CHECK_DW_ERROR(dwRenderEngine_setBuffer(m_pointCloudBuffer,
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
    m_pointCloudViz.get(),
    sizeof(float32_t) * 4, // XYZI stride
    0,
    m_pointCloudBufferSize,
    m_renderEngine));
    
    // Render point cloud with color by intensity or height
    dwRenderEngine_setColorByValue(DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_Z, 10.0f, m_renderEngine);
    dwRenderEngine_renderBuffer(m_pointCloudBuffer, m_pointCloudBufferSize, m_renderEngine);
    
    if (m_enableLogging && m_frameCount % 30 == 0) {
    log("Point cloud rendered successfully\n");
    }
    } catch (const std::exception& e) {
    logError("Error rendering point cloud: %s\n", e.what());
    }
    } else {
    if (m_enableLogging && m_frameCount % 30 == 0) {
    log("Skipping point cloud render: bufferSize=%d, vizBuffer=%s\n", 
    m_pointCloudBufferSize, m_pointCloudViz ? "valid" : "null");
    }
    }
    
    // Render bounding boxes if available
    if (m_enableLogging && m_frameCount % 30 == 0) {
    log("Attempting to render bounding boxes...\n");
    }
    
    try {
    renderBoundingBoxes();
    if (m_enableLogging && m_frameCount % 30 == 0) {
    log("Bounding boxes rendered successfully\n");
    }
    } catch (const std::exception& e) {
    logError("Error rendering bounding boxes: %s\n", e.what());
    }
    
    // Switch to 2D overlay for text
    dwRenderEngine_setProjection(&DW_IDENTITY_MATRIX4F, m_renderEngine);
    dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
    dwRenderEngine_setCoordinateRange2D({static_cast<float32_t>(getWindowWidth()), 
    static_cast<float32_t>(getWindowHeight())}, m_renderEngine);
    
    // Render text information
    dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_renderEngine);
    dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_GREEN, m_renderEngine);
    
    std::string infoText = "BEVFusion DriveWorks Application";
    dwRenderEngine_renderText2D(infoText.c_str(), {20.0f, getWindowHeight() - 30.0f}, m_renderEngine);
    
    std::string perfText = "Inference Time: " + std::to_string(m_inferenceTime) + " ms";
    dwRenderEngine_renderText2D(perfText.c_str(), {20.0f, getWindowHeight() - 50.0f}, m_renderEngine);
    
    std::string frameText = "Frame: " + std::to_string(m_frameCount);
    dwRenderEngine_renderText2D(frameText.c_str(), {20.0f, getWindowHeight() - 70.0f}, m_renderEngine);
    
    std::string cameraText = "Cameras: " + std::to_string(m_cameras.size()) + "/" + std::to_string(MAX_CAMERAS);
    dwRenderEngine_renderText2D(cameraText.c_str(), {20.0f, getWindowHeight() - 90.0f}, m_renderEngine);
    
    std::string lidarText = "LiDARs: " + std::to_string(m_lidars.size());
    dwRenderEngine_renderText2D(lidarText.c_str(), {20.0f, getWindowHeight() - 110.0f}, m_renderEngine);
    
    std::string pointCloudText = "Point Cloud: " + std::to_string(m_pointCloudBufferSize) + " points";
    dwRenderEngine_renderText2D(pointCloudText.c_str(), {20.0f, getWindowHeight() - 130.0f}, m_renderEngine);
    
    std::string bboxText = "Detections: " + std::to_string(m_lastBboxes.size());
    dwRenderEngine_renderText2D(bboxText.c_str(), {20.0f, getWindowHeight() - 150.0f}, m_renderEngine);
    
    if (!m_lastBboxes.empty()) {
    std::string bestText = "Best: ID=" + std::to_string(m_lastBboxes[0].id) + 
    " Score=" + std::to_string(m_lastBboxes[0].score);
    dwRenderEngine_renderText2D(bestText.c_str(), {20.0f, getWindowHeight() - 170.0f}, m_renderEngine);
    }
    
    renderutils::renderFPS(m_renderEngine, getCurrentFPS());
    
    // Render camera feeds in their tiles
    renderCameraFeeds();
    }
    

    
    // Project 3D bounding boxes to 2D camera coordinates using BEVFusion's approach
    void projectBoundingBoxesToCamera(int cameraIndex, const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes) {
        if (m_enableLogging) {
            log("PROJECTION DEBUG: camera=%d, bboxes=%zu\n", cameraIndex, bboxes.size());
        }
        
        if (bboxes.empty() || cameraIndex >= MAX_CAMERAS) {
            if (m_enableLogging) {
                log("PROJECTION DEBUG: Skipping projection: bboxes.empty()=%s, cameraIndex=%d >= MAX_CAMERAS=%d\n", 
                    bboxes.empty() ? "true" : "false", cameraIndex, MAX_CAMERAS);
            }
            return;
        }
        
        // Check if lidar2image tensor is loaded (it should be loaded by loadTransformationMatrices)
        if (m_lidar2image.empty()) {
            if (m_enableLogging) {
                log("PROJECTION DEBUG: ERROR: lidar2image tensor not loaded, skipping camera projection\n");
            }
            return;
        }
        
        if (m_enableLogging && cameraIndex == 0) {
            log("PROJECTION DEBUG: lidar2image tensor loaded: shape=[%ld, %ld, %ld, %ld]\n", 
                m_lidar2image.shape[0], m_lidar2image.shape[1], m_lidar2image.shape[2], m_lidar2image.shape[3]);
        }
        
        // Get camera image properties
        dwImageCUDA* imgCUDA = nullptr;
        dwImage_getCUDA(&imgCUDA, m_cameraRGBAImages[cameraIndex]);
        if (!imgCUDA) return;
        
        int imageWidth = imgCUDA->prop.width;
        int imageHeight = imgCUDA->prop.height;
        
        // Get camera transformation matrix (4×4) directly from existing tensor
        const float* tensorData = m_lidar2image.ptr<float>();
        const float* matrix = tensorData + (cameraIndex * 16); // Each camera has 16 floats (4×4)
        
        // Transform each bounding box using BEVFusion's method
        for (size_t i = 0; i < std::min(bboxes.size(), size_t(20)); ++i) {
            const auto& bbox = bboxes[i];
            
            if (m_enableLogging && m_frameCount % 60 == 0 && cameraIndex == 0) {
                log("Camera %d: Processing bbox %zu score=%.3f (no additional filtering)\n", cameraIndex, i, bbox.score);
            }
            
            // No additional confidence filtering - BEVFusion model already filtered at 0.12
            
            // Calculate 8 corners of 3D bounding box (BEVFusion approach)
            std::vector<std::array<float, 3>> corners;
            corners.reserve(8);
            
            float cos_rotation = cos(bbox.z_rotation);
            float sin_rotation = sin(bbox.z_rotation);
            
            // 8 corner offsets: {-1,-1,-1}, {+1,-1,-1}, {+1,+1,-1}, {-1,+1,-1}, 
            //                   {-1,-1,+1}, {+1,-1,+1}, {+1,+1,+1}, {-1,+1,+1}
            std::array<std::array<float, 3>, 8> offset_corners = {{
                {{-1, -1, -1}}, {{+1, -1, -1}}, {{+1, +1, -1}}, {{-1, +1, -1}},
                {{-1, -1, +1}}, {{+1, -1, +1}}, {{+1, +1, +1}}, {{-1, +1, +1}}
            }};
            
            std::vector<std::array<float, 2>> projectedCorners;
            bool validProjection = true;
            
            for (const auto& offset : offset_corners) {
                // Calculate 3D corner position
                float std_corner_x = bbox.size.w * offset[0] * 0.5f;
                float std_corner_y = bbox.size.l * offset[1] * 0.5f;
                float std_corner_z = bbox.size.h * offset[2] * 0.5f;
                
                // Apply rotation
                float corner_x = bbox.position.x + std_corner_x * cos_rotation + std_corner_y * sin_rotation;
                float corner_y = bbox.position.y + std_corner_x * (-sin_rotation) + std_corner_y * cos_rotation;
                float corner_z = bbox.position.z + std_corner_z;
                
                // Apply 4×4 transformation matrix
                float image_x = corner_x * matrix[0] + corner_y * matrix[1] + corner_z * matrix[2] + matrix[3];
                float image_y = corner_x * matrix[4] + corner_y * matrix[5] + corner_z * matrix[6] + matrix[7];
                float weight   = corner_x * matrix[8] + corner_y * matrix[9] + corner_z * matrix[10] + matrix[11];
                
                // Check validity
                if (weight <= 0 || image_x <= 0 || image_y <= 0) {
                    validProjection = false;
                    break;
                }
                
                // Perspective division
                weight = std::max(1e-5f, std::min(1e5f, weight));
                float pixel_x = image_x / weight;
                float pixel_y = image_y / weight;
                
                projectedCorners.push_back({{pixel_x, pixel_y}});
            }
            
            if (!validProjection || projectedCorners.size() != 8) {
                if (m_enableLogging) {
                    log("PROJECTION DEBUG: Camera %d: Invalid projection for bbox %zu: valid=%s, corners=%zu\n", 
                        cameraIndex, i, validProjection ? "true" : "false", projectedCorners.size());
                }
                continue;
            }
            
            // Store projected bounding box for rendering
            storeCameraProjection(cameraIndex, i, projectedCorners, bbox);
            
            if (m_enableLogging) {
                log("PROJECTION DEBUG: Camera %d: Projected bbox %zu (%.1f,%.1f,%.1f) score=%.2f, 8 corners valid, STORED\n",
                    cameraIndex, i, bbox.position.x, bbox.position.y, bbox.position.z, bbox.score);
            }
        }
        
        if (m_enableLogging && m_frameCount % 90 == 0) {
            log("Camera %d: Processed %zu bounding box projections using BEVFusion matrices\n", 
                cameraIndex, std::min(bboxes.size(), size_t(20)));
        }
    }
    
    // Store projected bounding box data for rendering
    struct CameraProjection {
        std::vector<std::array<float, 2>> corners;
        float score;
        int label;
        dwVector4f color;
    };
    
    std::vector<std::vector<CameraProjection>> m_cameraProjections; // [camera][projection]
    
    void storeCameraProjection(int cameraIndex, int bboxIndex, 
                              const std::vector<std::array<float, 2>>& corners,
                              const bevfusion::head::transbbox::BoundingBox& bbox) {
        if (m_cameraProjections.size() <= cameraIndex) {
            m_cameraProjections.resize(MAX_CAMERAS);
        }
        
        CameraProjection projection;
        projection.corners = corners;
        projection.score = bbox.score;
        projection.label = bbox.id;
        
        if (m_enableLogging) {
            log("PROJECTION DEBUG: storeCameraProjection: camera=%d, bbox=%d, corners=%zu, score=%.2f\n", 
                cameraIndex, bboxIndex, corners.size(), bbox.score);
        }
        
        // Set color based on object class using CUDA-BEVFusion nuScenes palette
        switch (bbox.id) {
            case 0: /* car */
                projection.color = {255.0f/255.0f, 158.0f/255.0f, 0.0f/255.0f, 1.0f};
                break;
            case 1: /* pedestrian */
                projection.color = {0.0f/255.0f, 0.0f/255.0f, 230.0f/255.0f, 1.0f};
                break;
            case 2: /* cyclist (bicycle) */
                projection.color = {220.0f/255.0f, 20.0f/255.0f, 60.0f/255.0f, 1.0f};
                break;
            case 3: /* motorcycle */
                projection.color = {255.0f/255.0f, 61.0f/255.0f, 99.0f/255.0f, 1.0f};
                break;
            case 4: /* truck */
                projection.color = {255.0f/255.0f, 99.0f/255.0f, 71.0f/255.0f, 1.0f};
                break;
            case 5: /* bus */
                projection.color = {255.0f/255.0f, 69.0f/255.0f, 0.0f/255.0f, 1.0f};
                break;
            case 6: /* trailer */
                projection.color = {255.0f/255.0f, 140.0f/255.0f, 0.0f/255.0f, 1.0f};
                break;
            case 7: /* construction_vehicle */
                projection.color = {233.0f/255.0f, 150.0f/255.0f, 70.0f/255.0f, 1.0f};
                break;
            case 8: /* barrier */
                projection.color = {112.0f/255.0f, 128.0f/255.0f, 144.0f/255.0f, 1.0f};
                break;
            case 9: /* traffic_cone */
                projection.color = {47.0f/255.0f, 79.0f/255.0f, 79.0f/255.0f, 1.0f};
                break;
            default: /* unknown */
                projection.color = {1.0f, 1.0f, 0.0f, 1.0f};
                break;
        }
        
        m_cameraProjections[cameraIndex].push_back(projection);
    }
    
    // Render projected bounding boxes on camera feed using DriveWorks 2D rendering
    void renderCameraBoundingBoxes(int cameraIndex, const dwVector2f& imageRange) {
        if (m_enableLogging) {
            log("RENDER DEBUG: renderCameraBoundingBoxes: camera=%d, projections_size=%zu, has_projections=%s\n", 
                cameraIndex, 
                (cameraIndex < m_cameraProjections.size()) ? m_cameraProjections[cameraIndex].size() : 0,
                (cameraIndex < m_cameraProjections.size() && !m_cameraProjections[cameraIndex].empty()) ? "true" : "false");
        }
        
        if (cameraIndex >= m_cameraProjections.size() || m_cameraProjections[cameraIndex].empty()) {
            if (m_enableLogging) {
                log("RENDER DEBUG: Camera %d: No projections to render (size=%zu, empty=%s)\n", 
                    cameraIndex,
                    (cameraIndex < m_cameraProjections.size()) ? m_cameraProjections[cameraIndex].size() : 0,
                    (cameraIndex < m_cameraProjections.size()) ? (m_cameraProjections[cameraIndex].empty() ? "true" : "false") : "out_of_bounds");
            }
            return;
        }
        
        const auto& projections = m_cameraProjections[cameraIndex];
        
        // Define 12 edges of 3D bounding box (same as BEVFusion)
        const int edges[][2] = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0}, // Bottom face
            {4, 5}, {5, 6}, {6, 7}, {7, 4}, // Top face
            {0, 4}, {1, 5}, {2, 6}, {3, 7}  // Vertical edges
        };
        
        for (const auto& projection : projections) {
            // Set color based on object class
            dwRenderEngine_setColor(projection.color, m_renderEngine);
            dwRenderEngine_setLineWidth(2.0f, m_renderEngine);
            
            // Collect all line segments for this bounding box
            std::vector<dwVector2f> linePoints;
            linePoints.reserve(24); // 12 edges × 2 points per edge
            
            // Draw 12 edges of the projected 3D bounding box
            for (const auto& edge : edges) {
                int idx0 = edge[0];
                int idx1 = edge[1];
                
                if (idx0 < projection.corners.size() && idx1 < projection.corners.size()) {
                    const auto& p0 = projection.corners[idx0];
                    const auto& p1 = projection.corners[idx1];
                    
                    // Clamp coordinates to image bounds
                    float x0 = std::max(0.0f, std::min(p0[0], imageRange.x));
                    float y0 = std::max(0.0f, std::min(p0[1], imageRange.y));
                    float x1 = std::max(0.0f, std::min(p1[0], imageRange.x));
                    float y1 = std::max(0.0f, std::min(p1[1], imageRange.y));
                    
                    // Add line segment (start point, end point)
                    linePoints.push_back({x0, y0});
                    linePoints.push_back({x1, y1});
                }
            }
            
            // Render all lines for this bounding box at once (NVIDIA's approach)
            if (!linePoints.empty()) {
                CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_2D,
                                                   linePoints.data(), sizeof(dwVector2f), 0, 
                                                   linePoints.size() / 2, m_renderEngine));
            }
            
            // Add confidence score text for all detections
            // Find minimum corner position for text placement
            float minX = imageRange.x, minY = imageRange.y;
            for (const auto& corner : projection.corners) {
                minX = std::min(minX, corner[0]);
                minY = std::min(minY, corner[1]);
            }
            
            minX = std::max(5.0f, std::min(minX, imageRange.x - 50.0f));
            minY = std::max(20.0f, std::min(minY, imageRange.y - 5.0f));
            
            // Create confidence text
            char scoreText[32];
            snprintf(scoreText, sizeof(scoreText), "%.2f", projection.score);
            
            dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, m_renderEngine); // White text
            dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_12, m_renderEngine);
            dwRenderEngine_renderText2D(scoreText, {minX, minY}, m_renderEngine);
        }
        
        if (m_enableLogging && m_frameCount % 60 == 0 && !projections.empty()) {
            log("Camera %d: Rendered %zu projected bounding boxes\n", cameraIndex, projections.size());
        }
    }
    
    void renderCameraFeeds()
    {
    if (m_enableLogging && m_frameCount % 30 == 0) {
    log("Rendering camera feeds...\n");
    }
    
    // Note: Don't clear camera projections here - they contain current frame's bounding boxes
    
    // BEVFusion order mapping for display
    // BEVFusion index -> Rig camera index (based on your 4camera_1lidar.json)
    // Your rig: 0=front_left, 1=front_right, 2=back_left, 3=back_right
    // BEVFusion: 0=FRONT, 1=FRONT_RIGHT, 2=FRONT_LEFT, 3=BACK, 4=BACK_LEFT, 5=BACK_RIGHT
    int rigCameraMapping[MAX_CAMERAS] = {-1, 0, 3, -1, 2, 1}; 
    // -1 = black image (no camera in rig)
    // BEVFusion FRONT(0) → black image
    // BEVFusion FRONT_RIGHT(1) → rig camera:front_right(1)  
    // BEVFusion FRONT_LEFT(2) → rig camera:front_left(0)
    // BEVFusion BACK(3) → black image
    // BEVFusion BACK_LEFT(4) → rig camera:back_left(2)
    // BEVFusion BACK_RIGHT(5) → rig camera:back_right(3)
    
    // Friendly camera names for visualization (BEVFusion order)
    // IMPORTANT: This order must match BEVFusion example and your tensor files
    // BEVFusion expects: 0=FRONT, 1=FRONT_RIGHT, 2=FRONT_LEFT, 3=BACK, 4=BACK_LEFT, 5=BACK_RIGHT
    // For 4-camera setup: map your rig cameras to indices 1,2,4,5 and use black images for 0,3
    static const char* kFriendlyNames[MAX_CAMERAS] = {
        "FRONT",         // 0 - BEVFusion index 0 (use black image for 4-camera setup)
        "FRONT_RIGHT",   // 1 - BEVFusion index 1 (your rig camera 1)
        "FRONT_LEFT",    // 2 - BEVFusion index 2 (your rig camera 0) 
        "BACK",          // 3 - BEVFusion index 3 (use black image for 4-camera setup)
        "BACK_LEFT",     // 4 - BEVFusion index 4 (your rig camera 2)
        "BACK_RIGHT"     // 5 - BEVFusion index 5 (your rig camera 3)
    };
    
    for (uint32_t i = 0; i < MAX_CAMERAS; i++) {
    try {
    // Set camera tile
    dwRenderEngine_setTile(m_tileCameras[i], m_renderEngine);
    dwRenderEngine_resetTile(m_renderEngine);
    
    // Get the rig camera index for this BEVFusion index
    int rigCameraIndex = rigCameraMapping[i];
    
    // Project 3D bounding boxes to 2D camera coordinates (use rig camera index for consistency)
    if (m_enableLogging && m_frameCount % 60 == 0 && i == 0) {
        log("About to call projectBoundingBoxesToCamera for BEVFusion camera %d (rig camera %d) with %zu bboxes\n", i, rigCameraIndex, m_lastBboxes.size());
    }
    
    // CRITICAL FIX: Project to ALL cameras (including black image cameras)
    // Use BEVFusion camera index for projection (consistent with transformation matrices)
    projectBoundingBoxesToCamera(i, m_lastBboxes);
    
    // Stream camera image to GL (use rig camera index for display)
    if (rigCameraIndex >= 0 && rigCameraIndex < static_cast<int>(m_cameras.size())) {
        if (m_enableLogging && m_frameCount % 60 == 0) {
            const char* rigName = m_cameras[rigCameraIndex].name.c_str();
            log("GUI tile %u label=%s -> rig[%d]=%s (actual physical position)\n", i,
                (i < MAX_CAMERAS ? kFriendlyNames[i] : "unknown"),
                rigCameraIndex, rigName);
        }
        // Convert RGB to RGBA for display (GPU operation)
        CHECK_DW_ERROR(dwImage_copyConvertAsync(m_cameraRGBAImages[rigCameraIndex], m_cameraRGBImages[rigCameraIndex], m_cudaStream, m_context));
        
        CHECK_DW_ERROR(dwImageStreamerGL_producerSend(m_cameraRGBAImages[rigCameraIndex], m_streamerToGL[rigCameraIndex]));
        
        // Receive GL image
        dwImageHandle_t frameGL;
        CHECK_DW_ERROR(dwImageStreamerGL_consumerReceive(&frameGL, 33000, m_streamerToGL[rigCameraIndex]));
        
        // Get GL image data
        dwImageGL* imageGL;
        CHECK_DW_ERROR(dwImage_getGL(&imageGL, frameGL));
        
        // Render the camera image
        dwVector2f range{};
        range.x = imageGL->prop.width;
        range.y = imageGL->prop.height;
        CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_renderImage2D(imageGL, {0, 0, range.x, range.y}, m_renderEngine));
        
        // Render projected bounding boxes on camera feed (use BEVFusion index for consistency)
        renderCameraBoundingBoxes(i, range);
        
        // Return GL image
        CHECK_DW_ERROR(dwImageStreamerGL_consumerReturn(&frameGL, m_streamerToGL[rigCameraIndex]));
        CHECK_DW_ERROR(dwImageStreamerGL_producerReturn(nullptr, 33000, m_streamerToGL[rigCameraIndex]));
    } else {
        // For missing cameras (indices 0 and 3), show black image
        // Use BEVFusion dimensions since all streamers now use consistent dimensions
        if (m_enableLogging && m_frameCount % 60 == 0) {
            log("GUI tile %u label=%s -> BLACK IMAGE\n", i,
                (i < MAX_CAMERAS ? kFriendlyNames[i] : "unknown"));
        }
        dwImageProperties blackProps = {};
        blackProps.type = DW_IMAGE_CUDA;
        blackProps.format = DW_IMAGE_FORMAT_RGBA_UINT8;
        blackProps.width = BEVFUSION_IMAGE_WIDTH;
        blackProps.height = BEVFUSION_IMAGE_HEIGHT;
        
        dwImageHandle_t blackImage;
        CHECK_DW_ERROR(dwImage_create(&blackImage, blackProps, m_context));
        
        // Fill with black
        dwImageCUDA* blackImgCUDA;
        CHECK_DW_ERROR(dwImage_getCUDA(&blackImgCUDA, blackImage));
        cudaMemset(blackImgCUDA->dptr[0], 0, blackImgCUDA->pitch[0] * blackImgCUDA->prop.height);
        
        // Stream to GL
        CHECK_DW_ERROR(dwImageStreamerGL_producerSend(blackImage, m_streamerToGL[i]));
        
        // Receive GL image
        dwImageHandle_t frameGL;
        CHECK_DW_ERROR(dwImageStreamerGL_consumerReceive(&frameGL, 33000, m_streamerToGL[i]));
        
        // Get GL image data
        dwImageGL* imageGL;
        CHECK_DW_ERROR(dwImage_getGL(&imageGL, frameGL));
        
        // Render the black image
        dwVector2f range{};
        range.x = imageGL->prop.width;
        range.y = imageGL->prop.height;
        CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_renderImage2D(imageGL, {0, 0, range.x, range.y}, m_renderEngine));
        
        // CRITICAL FIX: Render projected bounding boxes on black image cameras too
        renderCameraBoundingBoxes(i, range);
        
        // Return GL image
        CHECK_DW_ERROR(dwImageStreamerGL_consumerReturn(&frameGL, m_streamerToGL[i]));
        CHECK_DW_ERROR(dwImageStreamerGL_producerReturn(nullptr, 33000, m_streamerToGL[i]));
        
        // Clean up
        dwImage_destroy(blackImage);
    }
    
    // Add camera label
    dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, m_renderEngine);
    dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_12, m_renderEngine);
    
    std::string cameraLabel;
    const char* friendly = (i < MAX_CAMERAS) ? kFriendlyNames[i] : "unknown";
    if (i < m_cameras.size()) {
    cameraLabel = std::string("Cam ") + std::to_string(i) + ": " + friendly;
    } else {
    cameraLabel = std::string("Cam ") + std::to_string(i) + ": " + friendly;
    }
    
    // Move label slightly downward to avoid clipping in the tile
    CHECK_DW_ERROR(dwRenderEngine_renderText2D(cameraLabel.c_str(), {5.0f, 60.0f}, m_renderEngine));
    
    } catch (const std::exception& e) {
    logError("Error rendering camera %d: %s\n", i, e.what());
    }
    }
    
    if (m_enableLogging && m_frameCount % 30 == 0) {
    log("Camera feeds rendered\n");
    }
    }

    void onKeyDown(int key, int scancode, int mods) override
    {
    (void)scancode;
    (void)mods;
    
    if (key == GLFW_KEY_S) {
    m_screenshot->triggerScreenshot();
    }
    
    if (key == GLFW_KEY_R) {
    // Reset sensors
    dwSensorManager_stop(m_sensorManager);
    dwSensorManager_start(m_sensorManager);
    log("Sensors reset\n");
    }
    }
    };

    int main(int argc, const char** argv)
    {
    ProgramArguments args(argc, argv, {
    ProgramArguments::Option_t("rig", (dw_samples::SamplesDataPath::get() + "/samples/sensors/sensor_fusion/lab_sensors_rig.json").c_str(), 
    "Path to rig configuration file"),
    ProgramArguments::Option_t("bevfusion-model", "model/resnet50int8", 
    "Path to BEVFusion model directory"),
    ProgramArguments::Option_t("bevfusion-precision", "int8", 
    "BEVFusion precision (int8 or fp16)"),
    ProgramArguments::Option_t("example-data-path", "", 
    "Path to BEVFusion example data directory containing tensor files"),
    ProgramArguments::Option_t("enable-logging", "0", 
    "Enable detailed logging (1 for enabled, 0 for disabled)"),
    ProgramArguments::Option_t("hostpc", "0", 
    "Host PC mode - bypass rig file, use black images for cameras (1 for enabled, 0 for disabled)"),
    ProgramArguments::Option_t("camera-only", "0", 
    "Camera-only mode - skip LiDAR and BEVFusion, show only camera feed (1 for enabled, 0 for disabled)"),
    ProgramArguments::Option_t("process-rate", "0", 
    "Target process rate in Hz (0 = uncapped)"),
    ProgramArguments::Option_t("stabilize-skip", "0", 
    "Number of initial frames to skip for sensor stabilization (0 = disable)"),
    ProgramArguments::Option_t("lidar-ip", "192.168.2.201", 
    "LiDAR IP address (used in host PC mode, format: ip=X.X.X.X)"),
    ProgramArguments::Option_t("lidar-port", "2368", 
    "LiDAR port (used in host PC mode, format: port=XXXX)"),
    ProgramArguments::Option_t("black-images", "", 
    "Comma-separated list of camera indices to use black images (e.g., 0,3)"),
    }, "BEVFusion DriveWorks Application - Real-time sensor fusion with BEVFusion");
    
    BEVFusionDriveWorksApp app(args);
    app.initializeWindow("BEVFusion DriveWorks", 1280, 720, args.enabled("offscreen"));
    return app.run();
    }
