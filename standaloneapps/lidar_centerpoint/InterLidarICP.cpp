#include "InterLidarICP.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <numeric>
#include <cstdio>
#include <unistd.h>

#include <cuda_runtime_api.h>

#include <dw/core/base/Version.h>
#include <dw/core/logger/Logger.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/sensormanager/SensorManagerConstants.h>

#include <framework/Checks.hpp>
#include <framework/Log.hpp>
#include <framework/Mat4.hpp>
#include <framework/MathUtils.hpp>
#include <framework/MouseView3D.hpp>
#include <framework/RenderUtils.hpp>
#include <framework/WindowGLFW.hpp>

/////////////////////////////////////////////////////////////////////////////////////////////////
InterLidarICP::InterLidarICP(const ProgramArguments& args)
    : DriveWorksSample(args)
    , m_stream(cudaStreamDefault)
{
    m_rigFile = getArgument("rigFile");
    m_tensorRTEngine = getArgument("tensorRTEngine");
    m_pfeEnginePath = getArgument("pfeEngine");
    m_rpnEnginePath = getArgument("rpnEngine");
    m_maxIters = static_cast<uint32_t>(atoi(getArgument("maxIters").c_str()));
    m_numFrames = static_cast<uint32_t>(atoi(getArgument("numFrames").c_str()));
    m_skipFrames = static_cast<uint32_t>(atoi(getArgument("skipFrames").c_str()));
    m_verbose = getArgument("verbose") == "true";
    m_savePointClouds = getArgument("savePointClouds") == "true";
    m_objectDetectionEnabled = getArgument("objectDetection") == "true";
    m_groundPlaneEnabled = getArgument("groundPlane") == "true";
    m_groundPlaneVisualizationEnabled = getArgument("groundPlaneVisualization") == "true";
    m_minPointsThreshold = static_cast<uint32_t>(atoi(getArgument("minPoints").c_str()));
    m_bevVisualizationEnabled = getArgument("bevVisualization") == "true";
    m_heatmapVisualizationEnabled = getArgument("heatmapVisualization") == "true";

    m_freeSpaceEnabled = getArgument("freeSpace") == "true";
    m_freeSpaceVisualizationEnabled = getArgument("freeSpaceVisualization") == "true";



    if (m_maxIters > 100) {
        std::cerr << "`--maxIters` too large, set to " << (m_maxIters = 100) << std::endl;
    }

    if (m_numFrames == 0) {
        m_numFrames = static_cast<uint32_t>(-1);
    }

    // Initialize statistics
    m_lastICPStats = {};
    
    // Initialize ground plane
    m_groundPlane = {};
    m_groundPlaneValid = false;
    m_filteredGroundPlane = {};
    m_filteredGroundPlaneValid = false;
    
    // Initialize CUDA-native memory coherency resources
    m_memoryCoherencyStream = nullptr;
    m_cpuWriteCompleteEvent = nullptr;
    m_gpuReadReadyEvent = nullptr;
    
    // Initialize state management
    m_alignmentState = AlignmentState::INITIALIZING;
    m_consecutiveSuccessfulICP = 0;
    m_lidarReadyAnnounced = false;
    m_lastPeriodicICP = std::chrono::steady_clock::now();
    
    // Initialize object detection statistics
    m_totalDetections = 0;
    m_vehicleDetections = 0;
    m_pedestrianDetections = 0;
    m_cyclistDetections = 0;
    
    std::cout << "Inter-Lidar ICP with Object Detection initialized:" << std::endl;
    std::cout << "  Max ICP Iterations: " << m_maxIters << std::endl;
    std::cout << "  Verbose Logging: " << (m_verbose ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "  Save Point Clouds: " << (m_savePointClouds ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "  Object Detection: " << (m_objectDetectionEnabled ? "ENABLED" : "DISABLED") << std::endl;
    if (m_objectDetectionEnabled) {
        std::cout << "    CenterPoint PFE Engine: " << m_pfeEnginePath << std::endl;
        std::cout << "    CenterPoint RPN Engine: " << m_rpnEnginePath << std::endl;
    }
    std::cout << "  Ground Plane Extraction: " << (m_groundPlaneEnabled ? "ENABLED" : "DISABLED") << " (CUDA Pipeline)" << std::endl;
    std::cout << "  Ground Plane Visualization: " << (m_groundPlaneVisualizationEnabled ? "ENABLED" : "DISABLED") << std::endl;
    
    std::cout << "  Free Space Detection: " << (m_freeSpaceEnabled ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "  Free Space Visualization: " << (m_freeSpaceVisualizationEnabled ? "ENABLED" : "DISABLED") << std::endl;
    
    
    std::cout << "  Minimum Points Threshold: " << m_minPointsThreshold << " points" << std::endl;
    std::cout << "  State Management: ENABLED" << std::endl;
    std::cout << "    - Initial Alignment: Continuous ICP until aligned (TIGHTENED CRITERIA)" << std::endl;
    std::cout << "    - Periodic ICP: Every " << PERIODIC_ICP_INTERVAL_SECONDS << " seconds after alignment" << std::endl;
    std::cout << "    - Ground Plane: Only after initial alignment complete" << std::endl;
    std::cout << "    - Object Detection: Only after initial alignment complete" << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
InterLidarICP::~InterLidarICP()
{
    // Cleanup CUDA-native memory coherency resources
    if (m_memoryCoherencyStream) {
        cudaStreamDestroy(m_memoryCoherencyStream);
    }
    if (m_cpuWriteCompleteEvent) {
        cudaEventDestroy(m_cpuWriteCompleteEvent);
    }
    if (m_gpuReadReadyEvent) {
        cudaEventDestroy(m_gpuReadReadyEvent);
    }
    
    // Cleanup existing CUDA stream (was missing before)
    if (m_stream) {
        cudaStreamDestroy(m_stream);
    }
    
    if (m_logFile.is_open()) {
        m_logFile.close();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
std::string InterLidarICP::getStateString() const
{
    switch (m_alignmentState) {
        case AlignmentState::INITIALIZING: return "INITIALIZING";
        case AlignmentState::INITIAL_ALIGNMENT: return "INITIAL_ALIGNMENT";
        case AlignmentState::ALIGNED: return "ALIGNED";
        case AlignmentState::REALIGNMENT: return "REALIGNMENT";
        default: return "UNKNOWN";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::updateAlignmentState()
{
    switch (m_alignmentState) {
        case AlignmentState::INITIALIZING:
            // Move to initial alignment once we start processing frames
            if (m_frameNum > 0) {
                m_alignmentState = AlignmentState::INITIAL_ALIGNMENT;
                // std::cout << "STATE CHANGE: INITIALIZING -> INITIAL_ALIGNMENT" << std::endl;
                // std::cout << "Starting continuous ICP until complete alignment..." << std::endl;
            }
            break;
            
        case AlignmentState::INITIAL_ALIGNMENT:
            // Check if initial alignment is complete
            if (isInitialAlignmentComplete()) {
                m_alignmentState = AlignmentState::ALIGNED;
                m_lastPeriodicICP = std::chrono::steady_clock::now();
                announceLibarReady();
                // std::cout << "STATE CHANGE: INITIAL_ALIGNMENT -> ALIGNED" << std::endl;
                // std::cout << "Lidars successfully aligned! Switching to periodic ICP mode." << std::endl;
                // std::cout << "Ground plane extraction will now start continuously." << std::endl;
            }
            break;
            
        case AlignmentState::ALIGNED:
            // Check if periodic ICP is due
            if (isPeriodicICPDue()) {
                m_alignmentState = AlignmentState::REALIGNMENT;
                // std::cout << "STATE CHANGE: ALIGNED -> REALIGNMENT" << std::endl;
                // std::cout << "Performing periodic ICP re-alignment..." << std::endl;
            }
            break;
            
        case AlignmentState::REALIGNMENT:
            // Return to aligned state after performing ICP
            m_alignmentState = AlignmentState::ALIGNED;
            m_lastPeriodicICP = std::chrono::steady_clock::now();
            // if (m_verbose) {
            //     std::cout << "STATE CHANGE: REALIGNMENT -> ALIGNED" << std::endl;
            //     std::cout << "Periodic ICP complete, returning to aligned state." << std::endl;
            // }
            break;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
bool InterLidarICP::shouldPerformICP() const
{
    return (m_alignmentState == AlignmentState::INITIAL_ALIGNMENT || 
            m_alignmentState == AlignmentState::REALIGNMENT);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
bool InterLidarICP::shouldPerformGroundPlaneExtraction() const
{
    return (m_groundPlaneEnabled && 
            (m_alignmentState == AlignmentState::ALIGNED || 
             m_alignmentState == AlignmentState::REALIGNMENT));
}

/////////////////////////////////////////////////////////////////////////////////////////////////
bool InterLidarICP::shouldPerformObjectDetection() const
{
    // Only perform object detection when lidars are aligned and object detection is enabled
    return (m_objectDetectionEnabled && 
            (m_alignmentState == AlignmentState::ALIGNED || 
             m_alignmentState == AlignmentState::REALIGNMENT));
}

/////////////////////////////////////////////////////////////////////////////////////////////////
bool InterLidarICP::isInitialAlignmentComplete() const
{
    // Check multiple criteria for stable alignment
    if (m_consecutiveSuccessfulICP < MIN_SUCCESSFUL_ICP_FOR_ALIGNMENT) {
        return false;
    }
    
    if (!m_lastICPStats.successful) {
        return false;
    }
    
    if (m_lastICPStats.rmsCost > MAX_RMS_COST_FOR_ALIGNMENT) {
        return false;
    }
    
    if (m_lastICPStats.inlierFraction < MIN_INLIER_FRACTION_FOR_ALIGNMENT) {
        return false;
    }
    
    // Check if transform change is small (stable alignment)
    float32_t transformChange = calculateTransformChange(m_icpTransform, m_previousICPTransform);
    if (transformChange > MAX_TRANSFORM_CHANGE_FOR_ALIGNMENT) {
        return false;
    }
    
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
bool InterLidarICP::isPeriodicICPDue() const
{
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_lastPeriodicICP);
    return elapsed.count() >= PERIODIC_ICP_INTERVAL_SECONDS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
float32_t InterLidarICP::calculateTransformChange(const dwTransformation3f& current, const dwTransformation3f& previous) const
{
    // Calculate translation change
    float32_t translationChange = sqrt(
        pow(current.array[12] - previous.array[12], 2) +
        pow(current.array[13] - previous.array[13], 2) +
        pow(current.array[14] - previous.array[14], 2)
    );
    
    // Calculate rotation change (simplified using matrix difference)
    float32_t rotationChange = 0.0f;
    for (int i = 0; i < 9; i++) {
        rotationChange += pow(current.array[i] - previous.array[i], 2);
    }
    rotationChange = sqrt(rotationChange);
    
    // Combine translation and rotation changes (weighted)
    return translationChange + rotationChange * 0.1f;  // Weight rotation less
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::announceLibarReady()
{
    if (!m_lidarReadyAnnounced) {
        // std::cout << "\n" << std::string(50, '=') << std::endl;
        // std::cout << "           *** LIDAR READY ***" << std::endl;
        // std::cout << "   Initial alignment complete!" << std::endl;
        // std::cout << "   - ICP Success Rate: " << (100.0f * m_successfulICPCount / m_frameNum) << "%" << std::endl;
        // std::cout << "   - Final RMS Cost: " << (m_lastICPStats.rmsCost*1000) << " mm" << std::endl;
        // std::cout << "   - Inlier Fraction: " << (m_lastICPStats.inlierFraction*100) << "%" << std::endl;
        // std::cout << "   - Consecutive Successful ICPs: " << m_consecutiveSuccessfulICP << std::endl;
        // std::cout << "   Now performing periodic ICP every " << PERIODIC_ICP_INTERVAL_SECONDS << " seconds" << std::endl;
        // std::cout << "   Ground plane extraction enabled" << std::endl;
        // std::cout << std::string(50, '=') << "\n" << std::endl;
        
        m_lidarReadyAnnounced = true;
        
        // Log to file if enabled
        if (m_logFile.is_open()) {
            m_logFile << "LIDAR_READY," << m_frameNum << "," 
                      << (100.0f * m_successfulICPCount / m_frameNum) << "," 
                      << (m_lastICPStats.rmsCost*1000) << "," 
                      << (m_lastICPStats.inlierFraction*100) << "," 
                      << m_consecutiveSuccessfulICP << std::endl;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::initDriveWorks()
{
    // Initialize logger to print verbose message on console in color
    CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
    CHECK_DW_ERROR(dwLogger_setLogLevel(m_verbose ? DW_LOG_VERBOSE : DW_LOG_INFO));

    // Initialize SDK context
    dwContextParameters sdkParams = {};
#ifdef VIBRANTE
    sdkParams.eglDisplay = getEGLDisplay();
#endif

    CHECK_DW_ERROR(dwInitialize(&m_context, DW_VERSION, &sdkParams));
    
    // Initialize SAL
    CHECK_DW_ERROR_MSG(dwSAL_initialize(&m_sal, m_context), "Cannot initialize SAL");
    
    // Initialize CUDA stream
    CHECK_CUDA_ERROR(cudaStreamCreate(&m_stream));
    
    // Initialize CUDA-native memory coherency resources (replaces compiler barriers)
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&m_memoryCoherencyStream, cudaStreamNonBlocking));
    CHECK_CUDA_ERROR(cudaEventCreate(&m_cpuWriteCompleteEvent));
    CHECK_CUDA_ERROR(cudaEventCreate(&m_gpuReadReadyEvent));
    
    #ifdef __aarch64__
    std::cout << "[CUDA-NATIVE] Initialized memory coherency stream and events for unified memory" << std::endl;
    #endif
    
    // Initialize visualization
    CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_context));
    
    std::cout << "DriveWorks SDK initialized successfully" << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::initSensors()
{
    // Initialize Rig configuration
    CHECK_DW_ERROR_MSG(dwRig_initializeFromFile(&m_rigConfig, m_context, m_rigFile.c_str()),
                       "Could not initialize Rig from File");

    // Initialize Sensor Manager
    CHECK_DW_ERROR(dwSensorManager_initializeFromRig(&m_sensorManager, m_rigConfig, 
                                                     DW_SENSORMANGER_MAX_NUM_SENSORS, m_sal));

    // Get sensor counts
    uint32_t lidarCount = 0;
    CHECK_DW_ERROR(dwSensorManager_getNumSensors(&lidarCount, DW_SENSOR_LIDAR, m_sensorManager));

    std::cout << "Found " << lidarCount << " LiDAR sensors" << std::endl;

    if (lidarCount != NUM_LIDARS) {
        logError("This sample requires exactly %d LiDAR sensors, found %d", NUM_LIDARS, lidarCount);
        throw std::runtime_error("Incorrect number of LiDAR sensors");
    }

    m_lidarCount = lidarCount;

    // Start sensor manager
    CHECK_DW_ERROR(dwSensorManager_start(m_sensorManager));

    // Get LiDAR properties and transformations
    for (uint32_t i = 0; i < m_lidarCount; i++) {
        uint32_t lidarSensorIndex;
        dwSensorHandle_t lidarHandle;
        
        CHECK_DW_ERROR(dwSensorManager_getSensorIndex(&lidarSensorIndex, DW_SENSOR_LIDAR, i, m_sensorManager));
        CHECK_DW_ERROR(dwSensorManager_getSensorHandle(&lidarHandle, lidarSensorIndex, m_sensorManager));
        CHECK_DW_ERROR(dwSensorLidar_getProperties(&m_lidarProps[i], lidarHandle));
        CHECK_DW_ERROR(dwRig_getSensorToRigTransformation(&m_sensorToRigs[i], lidarSensorIndex, m_rigConfig));
        
        checkDeviceType(m_lidarProps[i], i);
        
        std::cout << "LiDAR " << i << " (" << m_lidarProps[i].deviceString << "):" << std::endl;
        std::cout << "  Points per spin: " << m_lidarProps[i].pointsPerSpin << std::endl;
        std::cout << "  Packets per spin: " << m_lidarProps[i].packetsPerSpin << std::endl;
        
        if (m_verbose) {
            dwTransformation3f& transform = m_sensorToRigs[i];
            std::cout << "  Sensor to Rig Transform:" << std::endl;
            std::cout << "    Translation: [" << transform.array[12] << ", " << transform.array[13] << ", " << transform.array[14] << "]" << std::endl;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::checkDeviceType(const dwLidarProperties& prop, uint32_t lidarIndex)
{
    std::string deviceString = prop.deviceString;
    if (deviceString != "VELO_VLP16" && 
        deviceString != "VELO_VLP16HR" && 
        deviceString != "VELO_HDL32E" && 
        deviceString != "VELO_HDL64E") {
        logWarn("LiDAR %d device is %s, this sample is optimized for Velodyne devices.\n", 
                lidarIndex, prop.deviceString);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::initBuffers()
{
    dwMemoryType memoryType = DW_MEMORY_TYPE_CUDA;

    // Initialize point cloud buffers for each lidar
    for (uint32_t i = 0; i < m_lidarCount; i++) {
        // Accumulated points buffer (organized for depth map ICP)
        m_accumulatedPoints[i].capacity = m_lidarProps[i].pointsPerSpin;
        m_accumulatedPoints[i].type = memoryType;
        m_accumulatedPoints[i].format = DW_POINTCLOUD_FORMAT_XYZI;
        m_accumulatedPoints[i].organized = true;  // CRITICAL: Mark as organized for depth map ICP
        CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_accumulatedPoints[i]));

        // Rig transformed points buffer (also organized)
        m_rigTransformedPoints[i].capacity = m_lidarProps[i].pointsPerSpin;
        m_rigTransformedPoints[i].type = memoryType;
        m_rigTransformedPoints[i].format = DW_POINTCLOUD_FORMAT_XYZI;
        m_rigTransformedPoints[i].organized = true;  // Keep organized structure
        CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_rigTransformedPoints[i]));

        // ICP aligned points buffer (organized)
        m_icpAlignedPoints[i].capacity = m_lidarProps[i].pointsPerSpin;
        m_icpAlignedPoints[i].type = memoryType;
        m_icpAlignedPoints[i].format = DW_POINTCLOUD_FORMAT_XYZI;
        m_icpAlignedPoints[i].organized = true;  // Keep organized structure
        CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_icpAlignedPoints[i]));
    }

    // Stitched point cloud buffer (can be unorganized for final output)
    uint32_t totalCapacity = m_lidarProps[0].pointsPerSpin + m_lidarProps[1].pointsPerSpin;
    m_stitchedPoints.capacity = totalCapacity;
    m_stitchedPoints.type = memoryType;
    m_stitchedPoints.format = DW_POINTCLOUD_FORMAT_XYZI;
    m_stitchedPoints.organized = false;  // Final stitched output doesn't need to be organized
    CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_stitchedPoints));
    
    // Host copy for ground plane extraction
    m_stitchedPointsHost.capacity = totalCapacity;
    m_stitchedPointsHost.type = DW_MEMORY_TYPE_CPU;
    m_stitchedPointsHost.format = DW_POINTCLOUD_FORMAT_XYZI;
    m_stitchedPointsHost.organized = false;
    CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_stitchedPointsHost));

    std::cout << "Point cloud buffers initialized (organized=true for ICP compatibility)" << std::endl;
}




void InterLidarICP::initFreeSpaceRendering()
{
    // Create render buffer for free space points
    // Assume max 50,000 free space points (100m x 100m at 0.2m resolution)
    uint32_t maxFreeSpacePoints = 50000;
    
    CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_freeSpaceRenderBufferId,
                                               DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                               sizeof(dwVector3f),
                                               0,
                                               maxFreeSpacePoints,
                                               m_renderEngine));
    
    std::cout << "Free space render buffer initialized with " << maxFreeSpacePoints << " points" << std::endl;
}


/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::performFreeSpaceDetection()
{
    if (!m_freeSpaceEnabled) {
        return;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Generate free space from stitched point cloud
    m_freeSpacePoints.clear();
    bool success = m_freeSpace.GenerateFreeSpace(m_stitchedPoints, m_freeSpacePoints);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    if (success && m_verbose) {
        uint32_t numGround, numObstacle, numFreeSpace;
        m_freeSpace.GetStatistics(numGround, numObstacle, numFreeSpace);
        
        std::cout << "Free Space Detection completed in " << duration.count() << " ms" << std::endl;
        std::cout << "  Ground points: " << numGround << std::endl;
        std::cout << "  Obstacle points: " << numObstacle << std::endl;
        std::cout << "  Free space points: " << numFreeSpace << std::endl;
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::renderFreeSpace()
{
    if (!m_freeSpaceVisualizationEnabled || m_freeSpacePoints.empty()) {
        return;
    }
    
    // Map render buffer
    dwVector3f* vertices = nullptr;
    uint32_t numPoints = m_freeSpacePoints.size() / 3; // x, y, intensity triplets
    uint32_t bufferSize = numPoints * sizeof(dwVector3f);
    
    CHECK_DW_ERROR(dwRenderEngine_mapBuffer(m_freeSpaceRenderBufferId,
                                            reinterpret_cast<void**>(&vertices),
                                            0,
                                            bufferSize,
                                            DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                            m_renderEngine));
    
    // Convert free space points to 3D vertices (on ground level)
    for (uint32_t i = 0; i < numPoints; i++) {
        vertices[i].x = m_freeSpacePoints[i * 3 + 0];
        vertices[i].y = m_freeSpacePoints[i * 3 + 1];
        vertices[i].z = 0.0f; // Free space at ground level
    }
    
    CHECK_DW_ERROR(dwRenderEngine_unmapBuffer(m_freeSpaceRenderBufferId,
                                              DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                              m_renderEngine));
    
    // Render free space in bright green with transparency
    dwRenderEngine_setColor({0.0f, 1.0f, 0.0f, 0.6f}, m_renderEngine);
    dwRenderEngine_setPointSize(3.0f, m_renderEngine);
    dwRenderEngine_renderBuffer(m_freeSpaceRenderBufferId, numPoints, m_renderEngine);
    dwRenderEngine_setPointSize(1.0f, m_renderEngine); // Reset
}


/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::initAccumulation()
{
    for (uint32_t i = 0; i < m_lidarCount; i++) {
        dwPointCloudAccumulatorParams params{};
        CHECK_DW_ERROR(dwPointCloudAccumulator_getDefaultParams(&params));
        
        // For depth map ICP, we need organized point clouds
        params.organized = true;  // Changed to true for depth map ICP
        params.memoryType = DW_MEMORY_TYPE_CUDA;
        params.enableMotionCompensation = false;  // Disable motion compensation for simplicity
        params.egomotion = DW_NULL_HANDLE;
        
        // Get sensor transformation
        uint32_t lidarSensorIndex;
        CHECK_DW_ERROR(dwSensorManager_getSensorIndex(&lidarSensorIndex, DW_SENSOR_LIDAR, i, m_sensorManager));
        CHECK_DW_ERROR(dwRig_getSensorToRigTransformation(&params.sensorTransformation, lidarSensorIndex, m_rigConfig));
        
        CHECK_DW_ERROR(dwPointCloudAccumulator_initialize(&m_accumulator[i], &params, &m_lidarProps[i], m_context));
        CHECK_DW_ERROR(dwPointCloudAccumulator_bindOutput(&m_accumulatedPoints[i], m_accumulator[i]));
        CHECK_DW_ERROR(dwPointCloudAccumulator_setCUDAStream(m_stream, m_accumulator[i]));
        
        std::cout << "LiDAR " << i << " accumulator initialized (organized=true for depth map ICP)" << std::endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::initICP()
{
    dwPointCloudICPParams params{};
    CHECK_DW_ERROR(dwPointCloudICP_getDefaultParams(&params));
    
    // Configure ICP parameters - only depth map type is available
    params.maxIterations = m_maxIters;
    params.icpType = DW_POINT_CLOUD_ICP_TYPE_DEPTH_MAP;
    
    // Configure depth map parameters to match VLP16 structure
    // VLP16 has 16 vertical beams, and your data shows 29,184 points per spin
    // 29,184 / 16 = 1,824 horizontal samples per beam
    uint32_t horizontalSamples = m_lidarProps[0].pointsPerSpin / 16;  // Calculate from actual data
    uint32_t verticalBeams = 16;  // VLP16 has 16 beams
    
    params.depthmapSize.x = horizontalSamples;  // ~1824 for VLP16
    params.depthmapSize.y = verticalBeams;      // 16 for VLP16
    
    params.maxPoints = params.depthmapSize.x * params.depthmapSize.y;
    
    // Verify the calculated size matches actual point cloud size
    if (params.maxPoints != m_lidarProps[0].pointsPerSpin) {
        logWarn("Depth map size mismatch! Calculated: %d, Actual: %d", 
                params.maxPoints, m_lidarProps[0].pointsPerSpin);
        
        // Fallback: use actual point count and calculate closest rectangular dimensions
        uint32_t totalPoints = m_lidarProps[0].pointsPerSpin;
        params.depthmapSize.y = 16;  // Keep 16 beams for VLP16
        params.depthmapSize.x = totalPoints / params.depthmapSize.y;
        params.maxPoints = params.depthmapSize.x * params.depthmapSize.y;
        
        std::cout << "Adjusted depth map size to: " << params.depthmapSize.x << "x" << params.depthmapSize.y 
                  << " (total: " << params.maxPoints << ")" << std::endl;
    }
    
    // *** ULTRA-TIGHT CONVERGENCE CRITERIA FOR PRECISE ALIGNMENT ***
    params.distanceConvergenceTol = 0.0005f;  // 0.5mm (was 2mm) - ultra-tight!
    params.angleConvergenceTol = 0.005f;      // ~0.3° (was 0.6°) - ultra-tight!
    
    CHECK_DW_ERROR(dwPointCloudICP_initialize(&m_icp, &params, m_context));
    CHECK_DW_ERROR(dwPointCloudICP_setCUDAStream(m_stream, m_icp));
    
    std::cout << "ICP initialized with ULTRA-TIGHT parameters:" << std::endl;
    std::cout << "  Type: DEPTH_MAP" << std::endl;
    std::cout << "  Max iterations: " << params.maxIterations << std::endl;
    std::cout << "  Depthmap size: " << params.depthmapSize.x << "x" << params.depthmapSize.y << std::endl;
    std::cout << "  Distance tolerance: " << params.distanceConvergenceTol*1000 << " mm (ULTRA-TIGHT)" << std::endl;
    std::cout << "  Angle tolerance: " << params.angleConvergenceTol << " rad (~" << (params.angleConvergenceTol*57.3) << "°)" << std::endl;
    std::cout << "  Max points: " << params.maxPoints << std::endl;
    std::cout << "  Expected VLP16 points: " << m_lidarProps[0].pointsPerSpin << std::endl;
    std::cout << "  Target: Sub-millimeter alignment accuracy" << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::initStitching()
{
    // Initialize coordinate converters for individual transformations
    for (uint32_t i = 0; i < m_lidarCount; i++) {
        CHECK_DW_ERROR(dwPointCloudStitcher_initialize(&m_coordinateConverter[i], m_context));
        CHECK_DW_ERROR(dwPointCloudStitcher_bindOutput(&m_rigTransformedPoints[i], m_coordinateConverter[i]));
        CHECK_DW_ERROR(dwPointCloudStitcher_setCUDAStream(m_stream, m_coordinateConverter[i]));
        
        std::cout << "Coordinate converter " << i << " initialized" << std::endl;
    }
    
    // Initialize ICP transformer for applying ICP transformation to LiDAR B
    CHECK_DW_ERROR(dwPointCloudStitcher_initialize(&m_icpTransformer, m_context));
    CHECK_DW_ERROR(dwPointCloudStitcher_bindOutput(&m_icpAlignedPoints[LIDAR_B_INDEX], m_icpTransformer));
    CHECK_DW_ERROR(dwPointCloudStitcher_setCUDAStream(m_stream, m_icpTransformer));
    std::cout << "ICP transformer initialized" << std::endl;
    
    // Initialize main stitcher for final stitching
    CHECK_DW_ERROR(dwPointCloudStitcher_initialize(&m_stitcher, m_context));
    CHECK_DW_ERROR(dwPointCloudStitcher_bindOutput(&m_stitchedPoints, m_stitcher));
    CHECK_DW_ERROR(dwPointCloudStitcher_setCUDAStream(m_stream, m_stitcher));
    
    std::cout << "Point cloud stitcher initialized" << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::initGroundPlaneExtraction()
{
    dwPointCloudPlaneExtractorParams params{};
    CHECK_DW_ERROR(dwPCPlaneExtractor_getDefaultParameters(&params));
    
    // Enable CUDA pipeline - this means we won't get inliers/outliers
    params.cudaPipelineEnabled = true;
    
    // Configure for ground plane detection
    params.maxInputPointCount = m_stitchedPoints.capacity;
    params.minInlierFraction = 0.22f;  // Tighter fit; reject non-ground clusters
    params.ransacIterationCount = 100;  // Maximum allowed by DriveWorks
    params.optimizerIterationCount = 10;  // Non-linear optimization iterations
    params.maxInlierDistance = 0.12f;  // CRITICAL: Missing parameter for proper ground detection!
    
    // Box filter parameters - focus on the region around the ground plane in rig coordinates
    // With correct rig mounting heights (sensors at z>0), the ground is near z ≈ 0
    params.boxFilterParams.maxPointCount = 20000;  // Keep enough points for robust fitting
    params.boxFilterParams.box.center = {3.0f, 0.0f, 0.0f};  // Center at ground level in rig frame
    params.boxFilterParams.box.halfAxisXYZ = {20.0f, 10.0f, 1.2f};  // 16m x 16m x 1.6m vertical span
    
    // Initial guess - ground plane should be roughly horizontal
    params.rotation = DW_IDENTITY_MATRIX3F;
    
    CHECK_DW_ERROR(dwPCPlaneExtractor_initialize(&m_planeExtractor, &params, m_context));
    CHECK_DW_ERROR(dwPCPlaneExtractor_setCUDAStream(m_stream, m_planeExtractor));
    
    std::cout << "Ground plane extractor initialized:" << std::endl;
    std::cout << "  CUDA Pipeline: ENABLED" << std::endl;
    std::cout << "  Max input points: " << params.maxInputPointCount << std::endl;
    std::cout << "  Min inlier fraction: " << params.minInlierFraction << " (increased for stability)" << std::endl;
    std::cout << "  RANSAC iterations: " << params.ransacIterationCount << " (max allowed: 100)" << std::endl;
    std::cout << "  Optimizer iterations: " << params.optimizerIterationCount << std::endl;
    std::cout << "  Max inlier distance: " << params.maxInlierDistance << " m (CRITICAL parameter)" << std::endl;
    std::cout << "  Box filter center: [" << params.boxFilterParams.box.center.x << ", " 
              << params.boxFilterParams.box.center.y << ", " << params.boxFilterParams.box.center.z << "] m" << std::endl;
    std::cout << "  Box filter size: " << params.boxFilterParams.box.halfAxisXYZ.x*2 << "x" 
              << params.boxFilterParams.box.halfAxisXYZ.y*2 << "x" 
              << params.boxFilterParams.box.halfAxisXYZ.z*2 << " m (focused on ground)" << std::endl;
    std::cout << "  Expected ground level: ~0.0m (rig origin at ground)" << std::endl;
    std::cout << "  NOTE: Ground plane extraction will only start after initial ICP alignment is complete" << std::endl;
    
    // Print the actual parameters being used
    std::cout << "\n=== GROUND PLANE EXTRACTION PARAMETERS ===" << std::endl;
    std::cout << "Box Filter:" << std::endl;
    std::cout << "  Center: [" << params.boxFilterParams.box.center.x << ", " 
              << params.boxFilterParams.box.center.y << ", " << params.boxFilterParams.box.center.z << "]" << std::endl;
    std::cout << "  Half Axis: [" << params.boxFilterParams.box.halfAxisXYZ.x << ", " 
              << params.boxFilterParams.box.halfAxisXYZ.y << ", " << params.boxFilterParams.box.halfAxisXYZ.z << "]" << std::endl;
    std::cout << "  Max Points: " << params.boxFilterParams.maxPointCount << std::endl;
    std::cout << "RANSAC:" << std::endl;
    std::cout << "  Min Inlier Fraction: " << params.minInlierFraction << std::endl;
    std::cout << "  Max Iterations: " << params.ransacIterationCount << std::endl;
    std::cout << "  Optimizer Iterations: " << params.optimizerIterationCount << std::endl;
    std::cout << "  Max Inlier Distance: " << params.maxInlierDistance << " m" << std::endl;
    std::cout << "=========================================" << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// void InterLidarICP::initGroundPlaneRenderBuffer()
// {
//     // Create render buffer for ground plane visualization
//     // We'll create a grid of triangles to represent the ground plane
//     uint32_t gridVertices = GROUND_PLANE_GRID_SIZE * GROUND_PLANE_GRID_SIZE * 6; // 2 triangles per cell, 3 vertices per triangle
    
//     CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_groundPlaneRenderBufferId,
//                                                DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
//                                                sizeof(dwVector3f),
//                                                0,
//                                                gridVertices,
//                                                m_renderEngine));
    
//     std::cout << "Ground plane render buffer initialized with " << gridVertices << " vertices" << std::endl;
// }

void InterLidarICP::initGroundPlaneRenderBuffer()
{
    // CORRECT: Calculate exact vertices needed for the mesh
    // For a grid of GROUND_PLANE_GRID_SIZE x GROUND_PLANE_GRID_SIZE,
    // we generate (SIZE-1) x (SIZE-1) cells, each with 2 triangles (6 vertices)
    uint32_t gridVertices = (GROUND_PLANE_GRID_SIZE - 1) * (GROUND_PLANE_GRID_SIZE - 1) * 6;
    
    std::cout << "Ground plane buffer calculation:" << std::endl;
    std::cout << "  Grid size: " << GROUND_PLANE_GRID_SIZE << "x" << GROUND_PLANE_GRID_SIZE << std::endl;
    std::cout << "  Cells: " << (GROUND_PLANE_GRID_SIZE - 1) << "x" << (GROUND_PLANE_GRID_SIZE - 1) << std::endl;
    std::cout << "  Vertices per cell: 6 (2 triangles)" << std::endl;
    std::cout << "  Total vertices: " << gridVertices << std::endl;
    
    CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_groundPlaneRenderBufferId,
                                               DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
                                               sizeof(dwVector3f),
                                               0,
                                               gridVertices,  // FIXED: Use correct size
                                               m_renderEngine));
    
    std::cout << "Ground plane render buffer initialized with EXACT " << gridVertices << " vertices" << std::endl;
    
    // AARCH64-SPECIFIC: CUDA-native buffer initialization (replaces compiler barriers)
    #ifdef __aarch64__
    dwVector3f* vertices = nullptr;
    uint32_t bufferSizeBytes = gridVertices * sizeof(dwVector3f);
    
    std::cout << "[CUDA-NATIVE] Performing initial buffer clear with CUDA-synchronization..." << std::endl;
    
    // Map buffer immediately after creation
    CHECK_DW_ERROR(dwRenderEngine_mapBuffer(m_groundPlaneRenderBufferId,
                                            reinterpret_cast<void**>(&vertices),
                                            0,
                                            bufferSizeBytes,
                                            DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
                                            m_renderEngine));
    
    // AARCH64-SPECIFIC: Enhanced buffer clearing for cache coherency
    #ifdef __aarch64__
    // Step 1: CPU clears the mapped buffer (only CPU operations work on mapped buffers)
    memset(vertices, 0, bufferSizeBytes);
    
    // Step 2: Write sentinel pattern for debugging
    for (uint32_t i = 0; i < gridVertices; i++) {
        vertices[i] = {0.0f, 0.0f, -999.0f};  // Sentinel value for debugging
    }
    
    // Step 3: CUDA-native memory barrier (ensures CPU writes complete before GPU reads)
    CHECK_CUDA_ERROR(cudaEventRecord(m_cpuWriteCompleteEvent, m_memoryCoherencyStream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_memoryCoherencyStream));
    
    std::cout << "  Buffer cleared and initialized with CUDA-native synchronization" << std::endl;
    #else
    // Standard initialization for x86
    memset(vertices, 0, bufferSizeBytes);
    for (uint32_t i = 0; i < gridVertices; i++) {
        vertices[i] = {0.0f, 0.0f, -999.0f};  // Sentinel value for debugging
    }
    #endif
    
    CHECK_DW_ERROR(dwRenderEngine_unmapBuffer(m_groundPlaneRenderBufferId,
                                              DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
                                              m_renderEngine));
    
    std::cout << "[CUDA-NATIVE] Buffer initialization complete with unified memory prefetching" << std::endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::initRendering()
{
    dwRenderEngineParams params{};
    CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&params,
                                                    static_cast<uint32_t>(getWindowWidth()),
                                                    static_cast<uint32_t>(getWindowHeight())));
    CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_viz));

    CHECK_DW_ERROR(dwRenderEngine_initTileState(&params.defaultTile));

    dwRenderEngineTileState tileParam = params.defaultTile;
    tileParam.layout.sizeLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_RELATIVE;
    tileParam.layout.positionLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_RELATIVE;

    // COMMENTED OUT: Individual LiDAR tiles - only showing ICP alignment view
    // Create tiles: 2x2 layout
    // Top-left: LiDAR A
    // tileParam.layout.viewport = {0.f, 0.f, 0.5f, 0.5f};
    // tileParam.layout.positionType = DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_LEFT;
    // CHECK_DW_ERROR(dwRenderEngine_addTile(&m_lidarTiles[LIDAR_A_INDEX].tileId, &tileParam, m_renderEngine));

    // COMMENTED OUT: Individual LiDAR tiles - only showing ICP alignment view
    // Top-right: LiDAR B
    // tileParam.layout.viewport = {0.5f, 0.f, 0.5f, 0.5f};
    // CHECK_DW_ERROR(dwRenderEngine_addTile(&m_lidarTiles[LIDAR_B_INDEX].tileId, &tileParam, m_renderEngine));

    // COMMENTED OUT: Individual LiDAR tiles - only showing ICP alignment view
    // Bottom-left: ICP alignment view
    // tileParam.layout.viewport = {0.f, 0.5f, 0.5f, 0.5f};
    // CHECK_DW_ERROR(dwRenderEngine_addTile(&m_icpTile.tileId, &tileParam, m_renderEngine));

    // MODIFIED: Single full-screen ICP alignment view
    tileParam.layout.viewport = {0.f, 0.f, 1.0f, 1.0f};
    tileParam.layout.positionType = DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_LEFT;
    CHECK_DW_ERROR(dwRenderEngine_addTile(&m_icpTile.tileId, &tileParam, m_renderEngine));

    // Bottom-right: Stitched view (commented out - redundant with post-ICP view)
    // tileParam.layout.viewport = {0.5f, 0.5f, 0.5f, 0.5f};
    // CHECK_DW_ERROR(dwRenderEngine_addTile(&m_stitchedTile.tileId, &tileParam, m_renderEngine));

    // COMMENTED OUT: Individual LiDAR render buffers - only showing ICP alignment view
    // Create render buffers
    // for (uint32_t i = 0; i < m_lidarCount; i++) {
    //     CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_lidarTiles[i].renderBufferId,
    //                                                DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
    //                                                sizeof(dwVector4f),
    //                                                0,
    //                                                m_lidarProps[i].pointsPerSpin,
    //                                                m_renderEngine));
    // }

    CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_icpTile.renderBufferId,
                                               DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                               sizeof(dwVector4f),
                                               0,
                                               m_stitchedPoints.capacity,
                                               m_renderEngine));

    // Stitched view buffer creation commented out - redundant with post-ICP view
    // CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_stitchedTile.renderBufferId,
    //                                            DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
    //                                            sizeof(dwVector4f),
    //                                            0,
    //                                            m_stitchedPoints.capacity,
    //                                            m_renderEngine));

    // Initialize ground plane render buffer
    initGroundPlaneRenderBuffer();

    // Initialize bounding box render buffer for object detection
    initBoundingBoxRenderBuffer();

    // Initialize BEV and heatmap visualization tiles (if enabled)
    if (m_bevVisualizationEnabled) {
        tileParam.layout.viewport = {0.0f, 0.0f, 0.5f, 0.5f};
        tileParam.layout.positionType = DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_LEFT;
        CHECK_DW_ERROR(dwRenderEngine_addTile(&m_bevTile.tileId, &tileParam, m_renderEngine));
        std::cout << "BEV visualization tile initialized" << std::endl;
    }

    if (m_heatmapVisualizationEnabled) {
        tileParam.layout.viewport = {0.5f, 0.0f, 0.5f, 0.5f};
        tileParam.layout.positionType = DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_LEFT;
        CHECK_DW_ERROR(dwRenderEngine_addTile(&m_heatmapTile.tileId, &tileParam, m_renderEngine));
        std::cout << "Heatmap visualization tile initialized" << std::endl;
    }

    std::cout << "Rendering system initialized" << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::initLogging()
{
    if (m_verbose) {
        std::string logFileName = "inter_lidar_icp_log.txt";
        m_logFile.open(logFileName);
        if (m_logFile.is_open()) {
            m_logFile << "Inter-Lidar ICP Log with State Management" << std::endl;
            m_logFile << "=========================================" << std::endl;
            m_logFile << "Max ICP Iterations: " << m_maxIters << std::endl;
            m_logFile << "State Management: ENABLED" << std::endl;
            m_logFile << "Initial Alignment Criteria:" << std::endl;
            m_logFile << "  - Min Consecutive Successful ICPs: " << MIN_SUCCESSFUL_ICP_FOR_ALIGNMENT << std::endl;
            m_logFile << "  - Max RMS Cost: " << (MAX_RMS_COST_FOR_ALIGNMENT*1000) << " mm" << std::endl;
            m_logFile << "  - Min Inlier Fraction: " << (MIN_INLIER_FRACTION_FOR_ALIGNMENT*100) << "%" << std::endl;
            m_logFile << "  - Max Transform Change: " << MAX_TRANSFORM_CHANGE_FOR_ALIGNMENT << std::endl;
            m_logFile << "Periodic ICP Interval: " << PERIODIC_ICP_INTERVAL_SECONDS << " seconds" << std::endl;
            m_logFile << "Frame,State,ICP_Performed,ICP_Success,Iterations,Correspondences,RMS_Cost,Inlier_Fraction,Processing_Time_ms,Ground_Plane_Valid,Ground_Plane_Filtered,Consecutive_Successful" << std::endl;
            std::cout << "Logging to " << logFileName << std::endl;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::calculateInputRequirements() {
    // Set up point thresholds for different distance ranges
    m_pointThresholds[10] = 1000;    // Within 10m: need 1000 points
    m_pointThresholds[20] = 500;     // Within 20m: need 500 points
    m_pointThresholds[30] = 300;     // Within 30m: need 300 points
    m_pointThresholds[50] = 200;     // Within 50m: need 200 points
    m_pointThresholds[100] = 100;    // Within 100m: need 100 points
    
    // Use constants from working implementation
    m_maxPoints = FIXED_NUM_POINTS;      // 204800 - what the model expects
    m_numFeatures = POINT_FEATURES;      // 4 - x, y, z, intensity
    m_requiredPoints = REALTIME_NUM_POINTS;  // 29000 - minimum for real-time
    m_maxModelPoints = FIXED_NUM_POINTS; // 204800 - maximum model can process
    
    m_inputSize = m_maxModelPoints * m_numFeatures;  // 204800 * 4
    m_outputSize = 393216 * 9;  // From working implementation: 393216 max detections with 9 values each
    
    std::cout << "Object detection input requirements calculated:" << std::endl;
    std::cout << "  Max points: " << m_maxPoints << std::endl;
    std::cout << "  Features per point: " << m_numFeatures << std::endl;
    std::cout << "  Required points (real-time): " << m_requiredPoints << std::endl;
    std::cout << "  Input size: " << m_inputSize << std::endl;
    std::cout << "  Output size: " << m_outputSize << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
bool InterLidarICP::initializeTensorRT() {
    if (!m_objectDetectionEnabled) {
        return true; // Skip initialization if object detection is disabled
    }
    
    if (m_tensorRTEngine.empty()) {
        std::cerr << "TensorRT engine file path not specified" << std::endl;
        return false;
    }
    
    std::cout << "Initializing TensorRT for object detection..." << std::endl;
    
    // Initialize TensorRT plugins (CRITICAL for models with custom plugins like VoxelGeneratorPlugin)
    initLibNvInferPlugins(&m_logger, "");
    std::cout << "TensorRT plugins initialized" << std::endl;
    
    // Create TensorRT runtime
    m_runtime.reset(nvinfer1::createInferRuntime(m_logger));
    if (!m_runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }
    
    // Load engine from file
    std::ifstream file(m_tensorRTEngine, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open TensorRT engine file: " << m_tensorRTEngine << std::endl;
        return false;
    }
    
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();
    
    // Deserialize engine
    m_engine.reset(m_runtime->deserializeCudaEngine(engineData.data(), size));
    if (!m_engine) {
        std::cerr << "Failed to deserialize TensorRT engine" << std::endl;
        return false;
    }
    
    // Create execution context
    m_executionContext.reset(m_engine->createExecutionContext());
    if (!m_executionContext) {
        std::cerr << "Failed to create TensorRT execution context" << std::endl;
        return false;
    }
    
    // Allocate buffers
    m_hostInputBuffer.resize(m_inputSize);
    m_hostOutputBuffer.resize(m_outputSize);
    
    // Allocate device buffers for all required tensors
    CHECK_CUDA_ERROR(cudaMalloc(&m_deviceInputBuffer, m_inputSize * sizeof(float)));  // points tensor
    CHECK_CUDA_ERROR(cudaMalloc(&m_deviceNumPointsBuffer, sizeof(int32_t)));  // num_points tensor
    CHECK_CUDA_ERROR(cudaMalloc(&m_deviceOutputBuffer, m_outputSize * sizeof(float)));  // output_boxes tensor
    CHECK_CUDA_ERROR(cudaMalloc(&m_deviceOutputCountBuffer, sizeof(int32_t)));  // num_boxes tensor
    
    std::cout << "TensorRT initialized successfully" << std::endl;
    std::cout << "  Engine: " << m_tensorRTEngine << std::endl;
    std::cout << "  Input buffer size: " << m_inputSize << " floats" << std::endl;
    std::cout << "  Output buffer size: " << m_outputSize << " floats" << std::endl;
    std::cout << "  Model expects " << m_maxModelPoints << " points with " << m_numFeatures << " features each" << std::endl;
    
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
bool InterLidarICP::initializeObjectDetection() {
    if (!m_objectDetectionEnabled) {
        std::cout << "Object detection disabled, skipping initialization" << std::endl;
        return true;
    }

    std::cout << "Initializing CenterPoint object detection..." << std::endl;

    if (m_pfeEnginePath.empty() || m_rpnEnginePath.empty()) {
        std::cerr << "CenterPoint engine paths (--pfeEngine, --rpnEngine) must be provided when object detection is enabled"
                  << std::endl;
        m_objectDetectionEnabled = false;
        return false;
    }

    CenterPointDW::Config cfg;
    cfg.pfeEnginePath = m_pfeEnginePath;
    cfg.rpnEnginePath = m_rpnEnginePath;
    cfg.fp16 = true;   // Engines are typically built as FP16 as in CenterPoint project
    cfg.dlaCore = -1;  // Use GPU by default

    m_centerPointDetector = std::make_unique<CenterPointDW>(cfg);
    if (!m_centerPointDetector || !m_centerPointDetector->isInitialized()) {
        std::cerr << "Failed to initialize CenterPoint detector" << std::endl;
        m_objectDetectionEnabled = false;
        return false;
    }

    // Initialize SORT-style tracker
    // Parameters: IOU threshold, max age (frames), min hits
    // maxAge: How many frames a track can survive without a match (higher = more persistent)
    // Increasing from 3 to 7 helps with temporary occlusions or missed detections
    m_tracker = std::make_unique<SimpleTracker>(0.3f, 7, 1);  // IOU threshold 0.3, max age 7, min hits 1
    std::cout << "SORT-style tracker initialized (maxAge=7 frames)" << std::endl;

    std::cout << "CenterPoint object detection initialized successfully" << std::endl;
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::initBoundingBoxRenderBuffer()
{
    if (!m_objectDetectionEnabled) {
        return;
    }
    
    // Create render buffer for bounding box lines
    // Each box has 12 edges, each edge has 2 vertices
    uint32_t maxBoxes = 100;  // Maximum number of boxes we expect to render
    uint32_t verticesPerBox = 24;  // 12 edges * 2 vertices per edge
    uint32_t totalVertices = maxBoxes * verticesPerBox;
    
    CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_boxLineBuffer,
                                               DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                               sizeof(dwVector3f),
                                               0,
                                               totalVertices,
                                               m_renderEngine));
    
    std::cout << "Bounding box render buffer initialized with " << totalVertices << " vertices" << std::endl;
    
    // AARCH64-SPECIFIC: CUDA-native line buffer initialization
    #ifdef __aarch64__
    dwVector3f* vertices = nullptr;
    uint32_t bufferSizeBytes = totalVertices * sizeof(dwVector3f);
    
    std::cout << "[BBOX-CUDA-NATIVE] Performing initial line buffer clear..." << std::endl;
    
    // Map buffer immediately after creation
    CHECK_DW_ERROR(dwRenderEngine_mapBuffer(m_boxLineBuffer,
                                            reinterpret_cast<void**>(&vertices),
                                            0,
                                            bufferSizeBytes,
                                            DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                            m_renderEngine));
    
    // AARCH64-SPECIFIC: Enhanced buffer clearing for cache coherency
    // Step 1: CPU clears the mapped buffer
    memset(vertices, 0, bufferSizeBytes);
    
    // Step 2: Write sentinel pattern for debugging
    for (uint32_t i = 0; i < totalVertices; i++) {
        vertices[i] = {0.0f, 0.0f, -999.0f};  // Sentinel value for debugging
    }
    
    // Step 3: CUDA-native memory barrier
    CHECK_CUDA_ERROR(cudaEventRecord(m_cpuWriteCompleteEvent, m_memoryCoherencyStream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_memoryCoherencyStream));
    
    std::cout << "  Line buffer cleared and initialized with CUDA-native synchronization" << std::endl;
    
    CHECK_DW_ERROR(dwRenderEngine_unmapBuffer(m_boxLineBuffer,
                                              DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                              m_renderEngine));
    
    std::cout << "[BBOX-CUDA-NATIVE] Line buffer initialization complete" << std::endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////
bool InterLidarICP::onInitialize()
{
    try {
        initDriveWorks();
        initSensors();
        initBuffers();
        initAccumulation();
        initICP();
        initStitching();
        initGroundPlaneExtraction();
        initRendering();
        initLogging();

        inspectRenderBuffers();  // Debug buffer state
        
        // Initialize object detection
        if (!initializeObjectDetection()) {
            std::cerr << "Failed to initialize object detection" << std::endl;
            return false;
        }

        if (m_freeSpaceEnabled) {
            initFreeSpaceRendering();
        }

        // Skip initial frames for sensor stabilization
        std::cout << "Skipping " << m_skipFrames << " initial frames for sensor stabilization..." << std::endl;
        for (uint32_t i = 0; i < m_skipFrames; ++i) {
            if (!getSpinFromBothLidars()) {
                logWarn("Could not get initial frame %d", i);
                break;
            }
        }
        
        std::cout << "Initialization complete!" << std::endl;
        std::cout << "Starting with STATE: " << getStateString() << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
bool InterLidarICP::getSpinFromBothLidars()
{
    // Reset accumulation state
    for (uint32_t i = 0; i < m_lidarCount; i++) {
        m_lidarAccumulated[i] = false;
        m_lidarOverflowCount[i] = 0;
    }

    uint32_t numLidarsAccumulated = 0;
    auto startTime = std::chrono::steady_clock::now();
    
    while (numLidarsAccumulated < m_lidarCount) {
        // Timeout after 5 seconds
        auto currentTime = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count() > 5) {
            logWarn("Timeout waiting for LiDAR data");
            return false;
        }

        const dwSensorEvent* acquiredEvent = nullptr;
        dwStatus status = dwSensorManager_acquireNextEvent(&acquiredEvent, 1000, m_sensorManager);

        if (status != DW_SUCCESS) {
            if (status == DW_END_OF_STREAM) {
                std::cout << "End of stream reached" << std::endl;
                return false;
            } else if (status == DW_TIME_OUT) {
                continue;
            } else {
                std::cerr << "Unable to acquire sensor event: " << dwGetStatusName(status) << std::endl;
                return false;
            }
        }

        if (acquiredEvent->type == DW_SENSOR_LIDAR) {
            const dwLidarDecodedPacket* packet = acquiredEvent->lidFrame;
            const uint32_t& lidarIndex = acquiredEvent->sensorTypeIndex;
            
            if (lidarIndex >= m_lidarCount) {
                logWarn("Invalid lidar index: %d", lidarIndex);
                CHECK_DW_ERROR(dwSensorManager_releaseAcquiredEvent(acquiredEvent, m_sensorManager));
                continue;
            }

            if (m_lidarAccumulated[lidarIndex]) {
                m_lidarOverflowCount[lidarIndex]++;
            } else {
                CHECK_DW_ERROR(dwPointCloudAccumulator_addLidarPacket(packet, m_accumulator[lidarIndex]));

                bool ready = false;
                CHECK_DW_ERROR(dwPointCloudAccumulator_isReady(&ready, m_accumulator[lidarIndex]));

                if (ready) {
                    m_registrationTime[lidarIndex] = packet->hostTimestamp;
                    m_lidarAccumulated[lidarIndex] = true;
                    CHECK_DW_ERROR(dwPointCloudAccumulator_process(m_accumulator[lidarIndex]));
                    numLidarsAccumulated++;
                    
                    if (m_verbose) {
                        std::cout << "LiDAR " << lidarIndex << " accumulated, points: " 
                                  << m_accumulatedPoints[lidarIndex].size << std::endl;
                    }
                }
            }
        }

        CHECK_DW_ERROR(dwSensorManager_releaseAcquiredEvent(acquiredEvent, m_sensorManager));
    }

    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::applyRigTransformations()
{
    // Transform each point cloud using their respective sensor-to-rig transformations
    for (uint32_t i = 0; i < m_lidarCount; i++) {
        // Bind input point cloud and transformation to the individual coordinate converter
        CHECK_DW_ERROR(dwPointCloudStitcher_bindInput(DW_BIND_SLOT_1,
                                                      &m_accumulatedPoints[i],
                                                      &m_sensorToRigs[i],
                                                      m_coordinateConverter[i]));
        
        // Process the transformation
        CHECK_DW_ERROR(dwPointCloudStitcher_process(m_coordinateConverter[i]));
        
        // IMPORTANT: Preserve organized structure for depth map ICP
        m_rigTransformedPoints[i].organized = m_accumulatedPoints[i].organized;
        
        if (m_verbose && i == 0) {
            std::cout << "Applied rig transformation for LiDAR " << i 
                      << " (organized=" << (m_rigTransformedPoints[i].organized ? "true" : "false") << ")" << std::endl;
        }
    }
    
    if (m_verbose) {
        std::cout << "Applied rig transformations to both lidars" << std::endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
bool InterLidarICP::performICP()
{
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Use LiDAR A as target (reference), LiDAR B as source (to be aligned)
    const dwPointCloud* targetPointCloud = &m_rigTransformedPoints[LIDAR_A_INDEX];
    const dwPointCloud* sourcePointCloud = &m_rigTransformedPoints[LIDAR_B_INDEX];
    
    // Debug: Verify point clouds are organized
    if (m_verbose) {
        std::cout << "ICP Debug Info:" << std::endl;
        std::cout << "  Target (LiDAR A) organized: " << (targetPointCloud->organized ? "true" : "false") << std::endl;
        std::cout << "  Source (LiDAR B) organized: " << (sourcePointCloud->organized ? "true" : "false") << std::endl;
        std::cout << "  Target points: " << targetPointCloud->size << std::endl;
        std::cout << "  Source points: " << sourcePointCloud->size << std::endl;
    }
    
    // Verify both point clouds are organized before proceeding
    if (!targetPointCloud->organized || !sourcePointCloud->organized) {
        logError("ICP requires organized point clouds! Target: %s, Source: %s",
                targetPointCloud->organized ? "organized" : "unorganized",
                sourcePointCloud->organized ? "organized" : "unorganized");
        return false;
    }
    
    // *** IMPROVED INITIAL GUESS - Use previous ICP result for better convergence ***
    dwTransformation3f initialGuess;
    if (m_frameNum > 1 && m_lastICPStats.successful) {
        // Use previous successful ICP transformation as initial guess
        initialGuess = m_icpTransform;
        if (m_verbose) {
            std::cout << "  Using previous ICP result as initial guess" << std::endl;
        }
    } else {
        // First frame or previous failed - use identity
        initialGuess = DW_IDENTITY_TRANSFORMATION3F;
        if (m_verbose) {
            std::cout << "  Using identity transform as initial guess" << std::endl;
        }
    }
    
    // Store previous transform for change calculation
    m_previousICPTransform = m_icpTransform;
    
    // Bind ICP inputs and outputs
    CHECK_DW_ERROR(dwPointCloudICP_bindInput(sourcePointCloud, targetPointCloud, &initialGuess, m_icp));
    CHECK_DW_ERROR(dwPointCloudICP_bindOutput(&m_icpTransform, m_icp));
    
    // Perform ICP
    dwStatus icpStatus = dwPointCloudICP_process(m_icp);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    m_lastICPStats.processingTime = duration.count() / 1000.0f; // Convert to milliseconds
    
    if (icpStatus == DW_SUCCESS) {
        // Get ICP statistics
        dwPointCloudICPResultStats icpStats{};
        CHECK_DW_ERROR(dwPointCloudICP_getLastResultStats(&icpStats, m_icp));
        
        m_lastICPStats.successful = true;
        m_lastICPStats.iterations = icpStats.actualNumIterations;
        m_lastICPStats.correspondences = icpStats.numCorrespondences;
        m_lastICPStats.rmsCost = icpStats.rmsCost;
        m_lastICPStats.inlierFraction = icpStats.inlierFraction;
        
        // Update cumulative transform
        m_cumulativeICPTransform *= m_icpTransform;
        Mat4_RenormR(m_cumulativeICPTransform.array);
        
        m_successfulICPCount++;
        m_consecutiveSuccessfulICP++;
        
        if (m_verbose) {
            std::cout << "ICP SUCCESS:" << std::endl;
            std::cout << "  Iterations: " << m_lastICPStats.iterations << std::endl;
            std::cout << "  Correspondences: " << m_lastICPStats.correspondences << std::endl;
            std::cout << "  RMS Cost: " << (m_lastICPStats.rmsCost*1000) << " mm" << std::endl;  // Show in mm
            std::cout << "  Inlier Fraction: " << (m_lastICPStats.inlierFraction*100) << "%" << std::endl;
            std::cout << "  Processing Time: " << m_lastICPStats.processingTime << " ms" << std::endl;
            std::cout << "  Consecutive Successful: " << m_consecutiveSuccessfulICP << std::endl;
        }
        
        return true;
    } else {
        m_lastICPStats.successful = false;
        m_lastICPStats.iterations = 0;
        m_lastICPStats.correspondences = 0;
        m_lastICPStats.rmsCost = -1.0f;
        m_lastICPStats.inlierFraction = -1.0f;
        
        m_failedICPCount++;
        m_consecutiveSuccessfulICP = 0;  // Reset consecutive count on failure
        
        if (m_verbose) {
            std::cout << "ICP FAILED: " << dwGetStatusName(icpStatus) << std::endl;
            std::cout << "  Consecutive successful ICPs reset to 0" << std::endl;
        }
        
        // Use identity transform when ICP fails
        m_icpTransform = DW_IDENTITY_TRANSFORMATION3F;
        return false;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::stitchPointClouds()
{
    // Copy LiDAR A points directly (they are the reference)
    // Just copy the buffer data since LiDAR A doesn't need ICP transformation
    CHECK_CUDA_ERROR(cudaMemcpy(m_icpAlignedPoints[LIDAR_A_INDEX].points, 
                                m_rigTransformedPoints[LIDAR_A_INDEX].points,
                                sizeof(dwVector4f) * m_rigTransformedPoints[LIDAR_A_INDEX].size,
                                cudaMemcpyDeviceToDevice));
    m_icpAlignedPoints[LIDAR_A_INDEX].size = m_rigTransformedPoints[LIDAR_A_INDEX].size;
    
    // Apply ICP transformation to LiDAR B points using dedicated transformer
    CHECK_DW_ERROR(dwPointCloudStitcher_bindInput(DW_BIND_SLOT_1,
                                                  &m_rigTransformedPoints[LIDAR_B_INDEX],
                                                  &m_icpTransform,
                                                  m_icpTransformer));
    CHECK_DW_ERROR(dwPointCloudStitcher_process(m_icpTransformer));
    
    // Now stitch both aligned point clouds together using the main stitcher
    CHECK_DW_ERROR(dwPointCloudStitcher_bindInput(DW_BIND_SLOT_1,
                                                  &m_icpAlignedPoints[LIDAR_A_INDEX],
                                                  &DW_IDENTITY_TRANSFORMATION3F,
                                                  m_stitcher));
    CHECK_DW_ERROR(dwPointCloudStitcher_bindInput(DW_BIND_SLOT_2,
                                                  &m_icpAlignedPoints[LIDAR_B_INDEX],
                                                  &DW_IDENTITY_TRANSFORMATION3F,
                                                  m_stitcher));
    CHECK_DW_ERROR(dwPointCloudStitcher_process(m_stitcher));
    
    // Copy to host for ground plane extraction
    CHECK_CUDA_ERROR(cudaMemcpy(m_stitchedPointsHost.points, 
                                m_stitchedPoints.points,
                                sizeof(dwVector4f) * m_stitchedPoints.size,
                                cudaMemcpyDeviceToHost));
    m_stitchedPointsHost.size = m_stitchedPoints.size;
    
    if (m_verbose) {
        std::cout << "Stitched point cloud created with " << m_stitchedPoints.size << " points" << std::endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::performGroundPlaneExtraction()
{
    // Debug: Print point cloud statistics before extraction
    std::cout << "\n=== POINT CLOUD STATISTICS ===" << std::endl;
    std::cout << "Total stitched points: " << m_stitchedPoints.size << std::endl;
    std::cout << "Point cloud capacity: " << m_stitchedPoints.capacity << std::endl;
    
    // Bind input point cloud (use device memory for CUDA pipeline)
    CHECK_DW_ERROR(dwPCPlaneExtractor_bindInput(&m_stitchedPoints, m_planeExtractor));
    
    // Bind output - only ground plane (no inliers/outliers in CUDA pipeline)
    CHECK_DW_ERROR(dwPCPlaneExtractor_bindOutput(nullptr, nullptr, &m_groundPlane, m_planeExtractor));
    
    // Process ground plane extraction
    dwStatus status = dwPCPlaneExtractor_process(m_planeExtractor);
    
    if (status == DW_SUCCESS) {
        m_groundPlaneValid = m_groundPlane.valid;
        
        // *** ALWAYS PRINT RAW GROUND PLANE DATA FOR DEBUGGING ***
        std::cout << "\n=== RAW GROUND PLANE DATA FROM DRIVEWORKS API ===" << std::endl;
        std::cout << "Status: " << dwGetStatusName(status) << std::endl;
        std::cout << "Valid: " << (m_groundPlane.valid ? "YES" : "NO") << std::endl;
        std::cout << "Raw Normal Vector: [" << m_groundPlane.normal.x << ", " 
                  << m_groundPlane.normal.y << ", " << m_groundPlane.normal.z << "]" << std::endl;
        std::cout << "Raw Offset: " << m_groundPlane.offset << std::endl;
        
        // Calculate and print plane equation: ax + by + cz + d = 0
        std::cout << "Plane Equation: " << m_groundPlane.normal.x << "x + " 
                  << m_groundPlane.normal.y << "y + " << m_groundPlane.normal.z << "z + " 
                  << m_groundPlane.offset << " = 0" << std::endl;
        
        // Print additional plane properties if available
        std::cout << "Normal Magnitude: " << sqrt(m_groundPlane.normal.x * m_groundPlane.normal.x +
                                                  m_groundPlane.normal.y * m_groundPlane.normal.y +
                                                  m_groundPlane.normal.z * m_groundPlane.normal.z) << std::endl;
        
        // Print ground height relative to sensor origin
        if (fabs(m_groundPlane.normal.z) > 0.001f) {
            float groundHeight = -m_groundPlane.offset / m_groundPlane.normal.z;
            std::cout << "Ground Height (z = " << groundHeight << "m) relative to sensor origin" << std::endl;
        }
        
        // Print plane angle from horizontal
        float angleFromHorizontal = acos(fabs(m_groundPlane.normal.z)) * 180.0f / M_PI;
        std::cout << "Plane Angle from Horizontal: " << angleFromHorizontal << " degrees" << std::endl;
        std::cout << "================================================" << std::endl;
        
        // *** GROUND PLANE VALIDATION ***
        if (m_groundPlaneValid) {
            // Check if plane is too steep (more than 20 degrees from horizontal)
            if (angleFromHorizontal > 20.0f) {
                std::cout << "*** GROUND PLANE REJECTED: Too steep (" << angleFromHorizontal 
                          << "° > 20° threshold) - likely a wall or vertical surface ***" << std::endl;
                m_groundPlaneValid = false;
            }
            
            // Check if ground height is reasonable (between -2m and -0.5m relative to sensor)
            // Lidars are at 1.26m height, so ground should be around -1.26m
            // if (fabs(m_groundPlane.normal.z) > 0.001f) {
            //     float groundHeight = -m_groundPlane.offset / m_groundPlane.normal.z;
            //     if (groundHeight < -2.0f || groundHeight > -0.5f) {
            //         std::cout << "*** GROUND PLANE REJECTED: Unreasonable height (" << groundHeight 
            //                   << "m, expected -2m to -0.5m, lidars at 1.26m) ***" << std::endl;
            //         m_groundPlaneValid = false;
            //     }
            // }
            
            // Check if normal vector is pointing mostly downward (positive z component in DriveWorks coordinate system)
            if (m_groundPlane.normal.z < 0.1f) {
                std::cout << "*** GROUND PLANE REJECTED: Normal not pointing downward (z=" 
                          << m_groundPlane.normal.z << ", expected > 0.1) ***" << std::endl;
                m_groundPlaneValid = false;
            }
        }
        
        if (m_groundPlaneValid) {
            // Apply temporal filtering for stability
            if (!m_filteredGroundPlaneValid) {
                // First valid detection - initialize filtered plane
                m_filteredGroundPlane = m_groundPlane;
                m_filteredGroundPlaneValid = true;
                std::cout << "*** FIRST VALID GROUND PLANE - INITIALIZED FILTERED PLANE ***" << std::endl;
            } else {
                // Apply exponential moving average filter
                float32_t alpha = GROUND_PLANE_FILTER_ALPHA;
                
                // Filter normal vector
                m_filteredGroundPlane.normal.x = alpha * m_filteredGroundPlane.normal.x + (1.0f - alpha) * m_groundPlane.normal.x;
                m_filteredGroundPlane.normal.y = alpha * m_filteredGroundPlane.normal.y + (1.0f - alpha) * m_groundPlane.normal.y;
                m_filteredGroundPlane.normal.z = alpha * m_filteredGroundPlane.normal.z + (1.0f - alpha) * m_groundPlane.normal.z;
                
                // Filter offset
                m_filteredGroundPlane.offset = alpha * m_filteredGroundPlane.offset + (1.0f - alpha) * m_groundPlane.offset;
                
                // Renormalize normal vector
                float32_t normalMag = sqrt(m_filteredGroundPlane.normal.x * m_filteredGroundPlane.normal.x +
                                          m_filteredGroundPlane.normal.y * m_filteredGroundPlane.normal.y +
                                          m_filteredGroundPlane.normal.z * m_filteredGroundPlane.normal.z);
                if (normalMag > 0.001f) {
                    m_filteredGroundPlane.normal.x /= normalMag;
                    m_filteredGroundPlane.normal.y /= normalMag;
                    m_filteredGroundPlane.normal.z /= normalMag;
                }
                
                m_filteredGroundPlane.valid = true;
                
                std::cout << "*** FILTERED GROUND PLANE UPDATED ***" << std::endl;
                std::cout << "Filtered Normal: [" << m_filteredGroundPlane.normal.x << ", " 
                          << m_filteredGroundPlane.normal.y << ", " << m_filteredGroundPlane.normal.z << "]" << std::endl;
                std::cout << "Filtered Offset: " << m_filteredGroundPlane.offset << std::endl;
            }
        } else {
            std::cout << "*** GROUND PLANE NOT VALID - SKIPPING FILTERING ***" << std::endl;
        }
    } else {
        m_groundPlaneValid = false;
        std::cout << "\n=== GROUND PLANE EXTRACTION FAILED ===" << std::endl;
        std::cout << "Status: " << dwGetStatusName(status) << std::endl;
        std::cout << "Error: " << dwGetStatusName(status) << std::endl;
        std::cout << "=====================================" << std::endl;
    }
}



/////////////////////////////////////////////////////////////////////////////////////////////////
// void InterLidarICP::performInference()
// {
//     if (!m_objectDetectionEnabled || !m_executionContext) {
//         return;
//     }
    
//     // Get actual number of points we have
//     uint32_t actualPoints = m_stitchedPointsHost.size;
    
//     // Create formatted input (pad to model size if needed)
//     std::vector<float> formattedInput(m_maxModelPoints * POINT_FEATURES, 0.0f);
    
//     // Copy available points 
//     const dwVector4f* points = static_cast<const dwVector4f*>(m_stitchedPointsHost.points);
//     uint32_t pointsToCopy = std::min(actualPoints, static_cast<uint32_t>(m_maxModelPoints));
    
//     for (uint32_t i = 0; i < pointsToCopy; ++i) {
//         uint32_t baseIdx = i * POINT_FEATURES;
//         formattedInput[baseIdx + 0] = points[i].x;     // x
//         formattedInput[baseIdx + 1] = points[i].y;     // y  
//         formattedInput[baseIdx + 2] = points[i].z;     // z
//         formattedInput[baseIdx + 3] = points[i].w;     // intensity
//     }
    
//     // Copy formatted input to GPU
//     CHECK_CUDA_ERROR(cudaMemcpy(m_deviceInputBuffer, formattedInput.data(),
//                                 m_maxModelPoints * POINT_FEATURES * sizeof(float),
//                                 cudaMemcpyHostToDevice));
    
//     // Set number of points
//     int32_t numPoints = static_cast<int32_t>(pointsToCopy);
//     CHECK_CUDA_ERROR(cudaMemcpy(m_deviceNumPointsBuffer, &numPoints, sizeof(int32_t),
//                                 cudaMemcpyHostToDevice));
    
//     // Set up bindings in the correct order (must match model's expected input/output order)
//     void* bindings[] = {
//         m_deviceInputBuffer,           // points tensor (input)
//         m_deviceNumPointsBuffer,       // num_points tensor (input)
//         m_deviceOutputBuffer,          // output_boxes tensor (output)
//         m_deviceOutputCountBuffer      // num_boxes tensor (output)
//     };
    
//     // Execute inference
//     bool success = m_executionContext->executeV2(bindings);
//     if (!success) {
//         std::cerr << "TensorRT inference execution failed" << std::endl;
//         return;
//     }
    
//     // Synchronize to ensure inference is complete
//     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
//     if (m_verbose) {
//         std::cout << "Object detection inference completed successfully" << std::endl;
//         std::cout << "  Input points: " << pointsToCopy << "/" << m_maxModelPoints << std::endl;
//     }
// }

/////////////////////////////////////////////////////////////////////////////////////////////////
// void InterLidarICP::processDetections()
// {
//     if (!m_objectDetectionEnabled) {
//         return;
//     }
    
//     // Get number of detected boxes from the model
//     int32_t numDetections;
//     CHECK_CUDA_ERROR(cudaMemcpy(&numDetections, m_deviceOutputCountBuffer, sizeof(int32_t),
//                                 cudaMemcpyDeviceToHost));
    
//     if (m_verbose) {
//         std::cout << "Number of detections from model: " << numDetections << std::endl;
//     }
    
//     if (numDetections <= 0) {
//         m_currentBoxes.clear();
//         return;
//     }
    
//     // Resize host buffer to accommodate actual detections (9 values per box)
//     std::vector<float> detectionData(numDetections * 9);
    
//     // Copy only the relevant box data from device
//     CHECK_CUDA_ERROR(cudaMemcpy(detectionData.data(), m_deviceOutputBuffer,
//                                 numDetections * 9 * sizeof(float),
//                                 cudaMemcpyDeviceToHost));
    
//     // Print raw detection data to understand what each value represents
//     if (m_verbose && numDetections > 0) {
//         std::cout << "\n=== RAW DETECTION DATA ANALYSIS ===" << std::endl;
//         std::cout << "Number of detections: " << numDetections << std::endl;
//         std::cout << "Each detection has 9 values - let's see what they are:" << std::endl;
        
//         // Show first 2 detections with labeled values
//         for (int i = 0; i < std::min(numDetections, 2); ++i) {
//             int baseIdx = i * 9;
//             std::cout << "\nDetection " << i << " - All 9 values:" << std::endl;
//             std::cout << "  [0] = " << detectionData[baseIdx + 0] << " (value 1)" << std::endl;
//             std::cout << "  [1] = " << detectionData[baseIdx + 1] << " (value 2)" << std::endl;
//             std::cout << "  [2] = " << detectionData[baseIdx + 2] << " (value 3)" << std::endl;
//             std::cout << "  [3] = " << detectionData[baseIdx + 3] << " (value 4)" << std::endl;
//             std::cout << "  [4] = " << detectionData[baseIdx + 4] << " (value 5)" << std::endl;
//             std::cout << "  [5] = " << detectionData[baseIdx + 5] << " (value 6)" << std::endl;
//             std::cout << "  [6] = " << detectionData[baseIdx + 6] << " (value 7)" << std::endl;
//             std::cout << "  [7] = " << detectionData[baseIdx + 7] << " (value 8)" << std::endl;
//             std::cout << "  [8] = " << detectionData[baseIdx + 8] << " (value 9)" << std::endl;
//         }
//         std::cout << "\n=== END RAW DETECTION DATA ANALYSIS ===\n" << std::endl;
//     }
    
//     std::vector<BoundingBox> rawDetections;
    
//     // Parse detection results from model output (9 values per detection)
//     // Format: [x, y, z, width, length, height, rotation, class, confidence] (CORRECTED ORDER)
//     const float confidenceThreshold = 0.0f;
    
//     for (int32_t i = 0; i < numDetections; ++i) {
//         int32_t baseIdx = i * 9;
        
//         // CORRECTED: Swap confidence and class parsing
//         int classId = static_cast<int>(detectionData[baseIdx + 7]);      // Class is at position 7
//         float confidence = detectionData[baseIdx + 8];                   // Confidence is at position 8
        
//         if (confidence > confidenceThreshold && classId >= 0 && classId <= 2) {
//             BoundingBox box;
//             box.x = detectionData[baseIdx + 0];
//             box.y = detectionData[baseIdx + 1];
//             box.z = detectionData[baseIdx + 2];
//             box.width = detectionData[baseIdx + 3];
//             box.length = detectionData[baseIdx + 4];
//             box.height = detectionData[baseIdx + 5];
//             box.rotation = detectionData[baseIdx + 6];
//             box.confidence = confidence;
//             box.classId = classId;
            
//             rawDetections.push_back(box);
//         }
//     }
    
//     if (m_verbose) {
//         std::cout << "Raw detections before filtering: " << rawDetections.size() << std::endl;
        
//         // Count detections by class
//         int vehicleCount = 0, pedestrianCount = 0, cyclistCount = 0;
//         for (const auto& box : rawDetections) {
//             switch (box.classId) {
//                 case 0: vehicleCount++; break;
//                 case 1: pedestrianCount++; break;
//                 case 2: cyclistCount++; break;
//             }
//         }
//         std::cout << "  Class distribution: Vehicles=" << vehicleCount << ", Pedestrians=" << pedestrianCount 
//                   << ", Cyclists=" << cyclistCount << std::endl;
        
//         // Debug: Show ALL pedestrian detections to help find the real person
//         std::cout << "  === ALL PEDESTRIAN DETECTIONS ===" << std::endl;
//         int pedestrianIndex = 0;
//         for (size_t i = 0; i < rawDetections.size(); ++i) {
//             const auto& box = rawDetections[i];
//             if (box.classId == 1) { // Pedestrian
//                 float distance = sqrt(box.x * box.x + box.y * box.y);
//                 int pointCount = countPointsInBox(box);
//                 float volume = box.width * box.length * box.height;
//                 float pointDensity = pointCount / std::max(volume, 0.1f);
                
//                 std::cout << "    Pedestrian " << pedestrianIndex++ << ": [" << box.width << "x" << box.length << "x" << box.height 
//                           << "] at (" << box.x << "," << box.y << "," << box.z << ") distance=" << distance 
//                           << "m conf=" << box.confidence << " points=" << pointCount << " density=" << pointDensity << " pts/m³" << std::endl;
                
//                 // Check if this could be a real person based on reasonable dimensions
//                 bool reasonableSize = (box.width >= 0.3f && box.width <= 1.2f && 
//                                       box.length >= 0.3f && box.length <= 1.2f && 
//                                       box.height >= 1.4f && box.height <= 2.2f);
//                 bool reasonableDistance = (distance >= 1.0f && distance <= 15.0f);
//                 bool reasonableConfidence = (box.confidence >= 0.05f);
                
//                 if (reasonableSize && reasonableDistance && reasonableConfidence) {
//                     std::cout << "      *** POTENTIAL REAL PERSON ***" << std::endl;
//                 }
//             }
//         }
        
//         // Debug: Show first few raw detection dimensions to help tune filtering
//         std::cout << "  === FIRST 5 RAW DETECTIONS ===" << std::endl;
//         for (size_t i = 0; i < std::min(rawDetections.size(), size_t(5)); ++i) {
//             const auto& box = rawDetections[i];
//             const char* className = (box.classId == 0) ? "Vehicle" : 
//                                    (box.classId == 1) ? "Pedestrian" : "Cyclist";
//             float distance = sqrt(box.x * box.x + box.y * box.y);
//             std::cout << "  Raw detection " << i << ": [" << box.width << "x" << box.length << "x" << box.height 
//                       << "] at (" << box.x << "," << box.y << "," << box.z << ") distance=" << distance 
//                       << "m conf=" << box.confidence << " class=" << box.classId << " (" << className << ")" << std::endl;
//         }
//     }
    
//     // Apply all filters to clean up detections
//     m_currentBoxes = applyAllFilters(rawDetections);
    
//     // Update detection statistics
//     m_totalDetections += m_currentBoxes.size();
//     for (const auto& box : m_currentBoxes) {
//         switch (box.classId) {
//             case 0: m_vehicleDetections++; break;
//             case 1: m_pedestrianDetections++; break;
//             case 2: m_cyclistDetections++; break;
//         }
//     }
    
//     if (m_verbose) {
//         std::cout << "\n=== FILTERING PIPELINE SUMMARY ===" << std::endl;
//         std::cout << "Raw detections: " << rawDetections.size() << std::endl;
//         std::cout << "After class filtering: " << m_currentBoxes.size() << std::endl;
//         std::cout << "Final detections: " << m_currentBoxes.size() << std::endl;
        
//         // Show detection quality metrics
//         if (!m_currentBoxes.empty()) {
//             std::cout << "\n=== DETECTION QUALITY ANALYSIS ===" << std::endl;
//             float avgConfidence = 0.0f;
//             float avgDistance = 0.0f;
//             int totalPoints = 0;
            
//             for (const auto& box : m_currentBoxes) {
//                 avgConfidence += box.confidence;
//                 avgDistance += sqrt(box.x * box.x + box.y * box.y);
//                 totalPoints += countPointsInBox(box);
//             }
            
//             avgConfidence /= m_currentBoxes.size();
//             avgDistance /= m_currentBoxes.size();
            
//             std::cout << "Average confidence: " << avgConfidence << std::endl;
//             std::cout << "Average distance: " << avgDistance << "m" << std::endl;
//             std::cout << "Total points across all detections: " << totalPoints << std::endl;
//         }
        
//         std::cout << "\n=== FINAL DETECTIONS ===" << std::endl;
//         for (const auto& box : m_currentBoxes) {
//             const char* className = (box.classId == 0) ? "Vehicle" : 
//                                    (box.classId == 1) ? "Pedestrian" : "Cyclist";
            
//             // Calculate distance and point count for detailed analysis
//             float distance = sqrt(box.x * box.x + box.y * box.y);
//             int pointCount = countPointsInBox(box);
//             float volume = box.width * box.length * box.height;
//             float pointDensity = pointCount / std::max(volume, 0.1f);
            
//             std::cout << "  " << className << " at (" << box.x << ", " << box.y << ", " << box.z 
//                       << ") conf=" << box.confidence << std::endl;
//             std::cout << "    Distance: " << distance << "m, Size: [" << box.width << "x" << box.length << "x" << box.height 
//                       << "]m, Points: " << pointCount << ", Density: " << pointDensity << " pts/m³" << std::endl;
            
//             // Add ground plane information for context
//             if (m_filteredGroundPlaneValid) {
//                 float groundHeight = -m_filteredGroundPlane.offset;
//                 float objectBottom = box.z - box.height/2.0f;
//                 float heightAboveGround = objectBottom - groundHeight;
//                 std::cout << "    Ground plane at: " << groundHeight << "m, Object bottom: " << objectBottom 
//                           << "m, Height above ground: " << heightAboveGround << "m" << std::endl;
//             }
            
//             // Additional analysis for potential false positives
//             if (box.classId == 1) { // Pedestrian
//                 std::cout << "    Pedestrian Analysis:" << std::endl;
                
//                 // Check if dimensions are realistic for a pedestrian
//                 if (box.width > 1.2f || box.length > 1.2f) {
//                     std::cout << "      WARNING: Dimensions too large for pedestrian!" << std::endl;
//                 }
//                 if (box.height < 1.0f || box.height > 2.5f) {
//                     std::cout << "      WARNING: Height unrealistic for pedestrian!" << std::endl;
//                 }
                
//                 // Check point distribution
//                 if (pointDensity > 500.0f) {
//                     std::cout << "      WARNING: Very high point density - likely static object!" << std::endl;
//                 }
//                 if (pointCount < 20) {
//                     std::cout << "      WARNING: Very few points - likely noise!" << std::endl;
//                 }
                
//                 // Check aspect ratios
//                 float widthToHeightRatio = box.width / std::max(box.height, 0.1f);
//                 float lengthToHeightRatio = box.length / std::max(box.height, 0.1f);
//                 if (widthToHeightRatio > 1.5f || lengthToHeightRatio > 1.5f) {
//                     std::cout << "      WARNING: Aspect ratios suggest non-human object!" << std::endl;
//                 }
                
//                 // Check if it's too close to ground (likely ground clutter)
//                 if (m_filteredGroundPlaneValid) {
//                     float groundHeight = -m_filteredGroundPlane.offset;
//                     float objectBottom = box.z - box.height/2.0f;
//                     float heightAboveGround = objectBottom - groundHeight;
                    
//                     if (heightAboveGround < 0.3f) {
//                         std::cout << "      WARNING: Too close to ground (height above ground: " << heightAboveGround << "m) - likely ground clutter!" << std::endl;
//                     }
//                 } else {
//                     if (box.z - box.height/2.0f < 0.3f) {
//                         std::cout << "      WARNING: Too close to ground - likely ground clutter!" << std::endl;
//                     }
//                 }
                
//                 // Check if it's at a suspicious distance
//                 if (distance > 15.0f && pointCount < 50) {
//                     std::cout << "      WARNING: Far distance with few points - likely false positive!" << std::endl;
//                 }
//             }
//         }
//     }
// }

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::performObjectDetection()
{
    if (!shouldPerformObjectDetection() || !m_centerPointDetector) {
        return;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    // Run CenterPoint on the stitched host point cloud
    std::vector<CenterPointDW::DWBoundingBox> cpBoxes;
    bool ok = m_centerPointDetector->inferOnPointCloud(m_stitchedPointsHost, cpBoxes);
    if (!ok) {
        if (m_verbose) {
            std::cout << "CenterPoint inference failed for this frame" << std::endl;
        }
        return;
    }

    // Convert to InterLidarICP::BoundingBox and apply existing filter pipeline
    std::vector<BoundingBox> rawBoxes;
    rawBoxes.reserve(cpBoxes.size());
    for (const auto& b : cpBoxes) {
        BoundingBox bb;
        bb.x = b.x;
        bb.y = b.y;
        bb.z = b.z;
        bb.width = b.width;
        bb.length = b.length;
        bb.height = b.height;
        bb.rotation = b.rotation;
        bb.confidence = b.confidence;
        bb.classId = b.classId;
        bb.trackId = -1;
        rawBoxes.push_back(bb);
    }

    m_currentBoxes = applyAllFilters(rawBoxes);

    // Update tracker and assign track IDs
    if (m_tracker) {
        std::cout << "[InterLidarICP] Updating tracker with " << m_currentBoxes.size() 
                  << " filtered detections" << std::endl;
        int numTracks = m_tracker->update(m_currentBoxes);
        std::cout << "[InterLidarICP] Tracker update complete: " << numTracks 
                  << " active tracks, " << m_currentBoxes.size() << " detections" << std::endl;
        
        // Log track IDs assigned
        for (const auto& box : m_currentBoxes) {
            std::cout << "[InterLidarICP]   Detection class=" << box.classId 
                      << " at (" << box.x << "," << box.y << "," << box.z 
                      << ") assigned trackId=" << box.trackId << std::endl;
        }
    } else {
        std::cout << "[InterLidarICP] WARNING: Tracker is null, skipping track ID assignment" << std::endl;
    }

    // Update detection statistics
    m_totalDetections += m_currentBoxes.size();
    for (const auto& box : m_currentBoxes) {
        switch (box.classId) {
            case 0: m_vehicleDetections++; break;
            case 1: m_pedestrianDetections++; break;
            case 2: m_cyclistDetections++; break;
            default: break;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    if (m_verbose) {
        std::cout << "CenterPoint object detection completed in " << duration.count()
                  << " ms, detections after filtering: " << m_currentBoxes.size() << std::endl;
    }

    // Fetch BEV and heatmap data for visualization (if enabled)
    if (m_bevVisualizationEnabled || m_heatmapVisualizationEnabled) {
        if (m_centerPointDetector) {
            if (m_bevVisualizationEnabled) {
                m_centerPointDetector->getBEVFeatureMap(m_bevFeatureMap);
            }
            if (m_heatmapVisualizationEnabled) {
                m_centerPointDetector->getHeatmap(m_heatmapData);
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
float InterLidarICP::computeIoU(const BoundingBox& box1, const BoundingBox& box2) {
    // Simplified 2D IoU calculation for now (considering x,y dimensions only)
    float x1_min = box1.x - box1.width / 2;
    float x1_max = box1.x + box1.width / 2;
    float y1_min = box1.y - box1.length / 2;
    float y1_max = box1.y + box1.length / 2;
    
    float x2_min = box2.x - box2.width / 2;
    float x2_max = box2.x + box2.width / 2;
    float y2_min = box2.y - box2.length / 2;
    float y2_max = box2.y + box2.length / 2;
    
    float intersect_x_min = std::max(x1_min, x2_min);
    float intersect_x_max = std::min(x1_max, x2_max);
    float intersect_y_min = std::max(y1_min, y2_min);
    float intersect_y_max = std::min(y1_max, y2_max);
    
    if (intersect_x_max <= intersect_x_min || intersect_y_max <= intersect_y_min) {
        return 0.0f;
    }
    
    float intersection = (intersect_x_max - intersect_x_min) * (intersect_y_max - intersect_y_min);
    float area1 = box1.width * box1.length;
    float area2 = box2.width * box2.length;
    float unionArea = area1 + area2 - intersection;
    
    return (unionArea > 0) ? intersection / unionArea : 0.0f;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<BoundingBox> InterLidarICP::applyNMS(const std::vector<BoundingBox>& boxes, float iouThreshold) {
    std::vector<BoundingBox> result;
    std::vector<bool> suppressed(boxes.size(), false);
    
    // Sort by confidence
    std::vector<size_t> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&boxes](size_t a, size_t b) {
        return boxes[a].confidence > boxes[b].confidence;
    });
    
    for (size_t i = 0; i < indices.size(); ++i) {
        if (suppressed[indices[i]]) continue;
        
        result.push_back(boxes[indices[i]]);
        
        for (size_t j = i + 1; j < indices.size(); ++j) {
            if (suppressed[indices[j]]) continue;
            
            if (computeIoU(boxes[indices[i]], boxes[indices[j]]) > iouThreshold) {
                suppressed[indices[j]] = true;
            }
        }
    }
    
    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<BoundingBox> InterLidarICP::filterBoundingBoxesBySize(const std::vector<BoundingBox>& boxes) {
    std::vector<BoundingBox> filtered;
    
    struct SizeConstraint {
        float minWidth;
        float maxWidth;
        float minLength;
        float maxLength;
        float minHeight;
        float maxHeight;
    };
    
    // Define size constraints for each class
    std::vector<SizeConstraint> constraints = {
        // Vehicle (class 0)
        {1.5f, 6.0f, 3.0f, 12.0f, 1.2f, 3.5f},
        // Pedestrian (class 1)  
        {0.3f, 1.2f, 0.3f, 1.2f, 1.0f, 2.2f},
        // Cyclist (class 2)
        {0.5f, 2.5f, 1.0f, 3.0f, 1.0f, 2.5f}
    };
    
    for (const auto& box : boxes) {
        if (box.classId >= 0 && box.classId < 3) {
            const auto& constraint = constraints[box.classId];
            
            if (box.width >= constraint.minWidth && box.width <= constraint.maxWidth &&
                box.length >= constraint.minLength && box.length <= constraint.maxLength &&
                box.height >= constraint.minHeight && box.height <= constraint.maxHeight) {
                filtered.push_back(box);
            }
        }
    }
    
    return filtered;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
int InterLidarICP::countPointsInBox(const BoundingBox& box) {
    if (!m_stitchedPointsHost.points || m_stitchedPointsHost.size == 0) {
        return 0;
    }
    
    const dwVector4f* points = static_cast<const dwVector4f*>(m_stitchedPointsHost.points);
    int count = 0;
    
    float halfWidth = box.width / 2.0f;
    float halfLength = box.length / 2.0f;
    float halfHeight = box.height / 2.0f;
    
    for (uint32_t i = 0; i < m_stitchedPointsHost.size; ++i) {
        float dx = points[i].x - box.x;
        float dy = points[i].y - box.y;
        float dz = points[i].z - box.z;
        
        if (std::abs(dx) <= halfWidth && 
            std::abs(dy) <= halfLength && 
            std::abs(dz) <= halfHeight) {
            count++;
        }
    }
    
    return count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<BoundingBox> InterLidarICP::filterByPointDensity(const std::vector<BoundingBox>& boxes) {
    std::vector<BoundingBox> filtered;
    
    for (const auto& box : boxes) {
        int pointCount = countPointsInBox(box);
        float distance = sqrt(box.x * box.x + box.y * box.y);
        
        // Define point count ranges for each object type based on distance
        int minPoints = 0;
        int maxPoints = 0;
        
        switch (box.classId) {
            case 0: // Vehicle
                if (distance < 10.0f) {
                    minPoints = 80;   // Close vehicles need more points
                    maxPoints = 800;
                } else if (distance < 20.0f) {
                    minPoints = 50;
                    maxPoints = 600;
                } else if (distance < 30.0f) {
                    minPoints = 30;
                    maxPoints = 400;
                } else if (distance < 50.0f) {
                    minPoints = 20;
                    maxPoints = 300;
                } else {
                    minPoints = 15;
                    maxPoints = 200;
                }
                break;
                
            case 1: // Pedestrian
                if (distance < 10.0f) {
                    minPoints = 10;   // Very permissive for close objects
                    maxPoints = 500;  // Higher max to allow real people with many points
                } else if (distance < 20.0f) {
                    minPoints = 8;    // Very permissive for medium distance
                    maxPoints = 400;  // Higher max to allow real people
                } else if (distance < 30.0f) {
                    minPoints = 5;    // Very permissive for far objects
                    maxPoints = 300;  // Higher max to allow real people
                } else if (distance < 50.0f) {
                    minPoints = 3;
                    maxPoints = 200;
                } else {
                    minPoints = 2;
                    maxPoints = 150;
                }
                break;
                
            case 2: // Cyclist
                if (distance < 10.0f) {
                    minPoints = 50;
                    maxPoints = 400;
                } else if (distance < 20.0f) {
                    minPoints = 30;
                    maxPoints = 300;
                } else if (distance < 30.0f) {
                    minPoints = 20;
                    maxPoints = 250;
                } else if (distance < 50.0f) {
                    minPoints = 15;
                    maxPoints = 200;
                } else {
                    minPoints = 10;
                    maxPoints = 150;
                }
                break;
                
            default:
                // Unknown class - use conservative defaults
                minPoints = 30;
                maxPoints = 400;
                break;
        }
        
        // Check if point count is within the acceptable range
        if (pointCount >= minPoints && pointCount <= maxPoints) {
            filtered.push_back(box);
            
            if (m_verbose) {
                const char* className = (box.classId == 0) ? "Vehicle" : 
                                       (box.classId == 1) ? "Pedestrian" : "Cyclist";
                std::cout << "  " << className << " at " << distance << "m: " 
                          << pointCount << " points [" << minPoints << "-" << maxPoints << "]" << std::endl;
            }
        } else if (m_verbose) {
            const char* className = (box.classId == 0) ? "Vehicle" : 
                                   (box.classId == 1) ? "Pedestrian" : "Cyclist";
            std::string reason = (pointCount < minPoints) ? "TOO FEW POINTS" : "TOO MANY POINTS";
            std::cout << "  REJECTED " << className << " at " << distance << "m: " 
                      << pointCount << " points [" << minPoints << "-" << maxPoints << "] - " << reason << std::endl;
            
            // Check if this rejected detection could be a real person
            if (box.classId == 1) { // Pedestrian
                bool reasonableSize = (box.width >= 0.3f && box.width <= 1.2f && 
                                      box.length >= 0.3f && box.length <= 1.2f && 
                                      box.height >= 1.4f && box.height <= 2.2f);
                bool reasonableDistance = (distance >= 1.0f && distance <= 15.0f);
                bool reasonableConfidence = (box.confidence >= 0.05f);
                
                if (reasonableSize && reasonableDistance && reasonableConfidence) {
                    std::cout << "    *** WARNING: REJECTED POTENTIAL REAL PERSON ***" << std::endl;
                    std::cout << "    Consider adjusting point density ranges for distance " << distance << "m" << std::endl;
                }
            }
        }
    }
    
    return filtered;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<BoundingBox> InterLidarICP::filterByGroundPlane(const std::vector<BoundingBox>& boxes) {
    if (!m_filteredGroundPlaneValid) {
        return boxes; // No ground plane filtering if plane not detected
    }
    
    std::vector<BoundingBox> filtered;
    
    for (const auto& box : boxes) {
        // Calculate distance from box bottom to ground plane
        float boxBottom = box.z - box.height / 2.0f;
        
        // Distance from point to plane: |ax + by + cz + d| / sqrt(a² + b² + c²)
        float distanceToGround = std::abs(
            m_filteredGroundPlane.normal.x * box.x +
            m_filteredGroundPlane.normal.y * box.y +
            m_filteredGroundPlane.normal.z * boxBottom +
            m_filteredGroundPlane.offset
        );
        
        // Objects should be reasonably close to the ground plane
        if (distanceToGround < 2.0f) { // Within 2 meters of ground
            filtered.push_back(box);
        }
    }
    
    return filtered;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<BoundingBox> InterLidarICP::applyAllFilters(const std::vector<BoundingBox>& rawBoxes) {
    std::vector<BoundingBox> filtered = rawBoxes;
    
    if (filtered.empty()) {
        return filtered;
    }
    
    // 1. Filter by allowed classes (Vehicle, Pedestrian, Cyclist)
    std::vector<int> allowedClasses = {0,1};  // Vehicle, Pedestrian, Cyclist (re-enabled cyclists)
    size_t beforeClassFilter = filtered.size();
    filtered = filterByClasses(filtered, allowedClasses);
    if (m_verbose) {
        std::cout << "After class filtering: " << filtered.size() << " boxes (removed " 
                  << (beforeClassFilter - filtered.size()) << " non-pedestrian detections)" << std::endl;
    }
    
    // 2. Filter by confidence (now that confidence parsing is fixed)
    // filtered = filterByDistanceBasedConfidence(filtered);
    // if (m_verbose) std::cout << "After confidence filtering: " << filtered.size() << " boxes" << std::endl;
    
    // 2.5. Confidence filter for vehicles and pedestrians
    size_t beforeConfidence = filtered.size();
    std::vector<BoundingBox> highConfidenceFiltered;
    for (const auto& box : filtered) {
        if (box.classId == 0) { // Vehicle
            // Only keep vehicles with reasonable confidence
            if (box.confidence >= 0.20f) {  // 20% confidence threshold for vehicles
                highConfidenceFiltered.push_back(box);
            } else if (m_verbose) {
                std::cout << "  REJECTED: Low confidence vehicle (" << box.confidence << ")" << std::endl;
            }
        } else if (box.classId == 1) { // Pedestrian
            // Only keep pedestrians with reasonable confidence
            if (box.confidence >= 0.20f) {  // Lowered threshold to catch more real people
                highConfidenceFiltered.push_back(box);
            } else if (m_verbose) {
                std::cout << "  REJECTED: Low confidence pedestrian (" << box.confidence << ")" << std::endl;
            }
        } else {
            // Keep other classes (cyclists, etc.) as-is
            highConfidenceFiltered.push_back(box);
        }
    }
    filtered = highConfidenceFiltered;
    if (m_verbose) {
        std::cout << "After confidence filtering: " << filtered.size() << " boxes (removed " 
                  << (beforeConfidence - filtered.size()) << " low confidence detections)" << std::endl;
    }
    
    // 3. Filter by minimum point count (configurable threshold)
    size_t beforeMinPointFilter = filtered.size();
    std::vector<BoundingBox> minPointFiltered;
    for (const auto& box : filtered) {
        int pointCount = countPointsInBox(box);
        if (pointCount >= static_cast<int>(m_minPointsThreshold)) {
            minPointFiltered.push_back(box);
        } else if (m_verbose) {
            const char* className = (box.classId == 0) ? "Vehicle" : 
                                   (box.classId == 1) ? "Pedestrian" : "Cyclist";
            float distance = sqrt(box.x * box.x + box.y * box.y);
            std::cout << "  REJECTED " << className << " at " << distance << "m: " 
                      << pointCount << " points (< " << m_minPointsThreshold << " minimum)" << std::endl;
        }
    }
    filtered = minPointFiltered;
    if (m_verbose) {
        std::cout << "After minimum point count filtering (" << m_minPointsThreshold << "): " << filtered.size() << " boxes (removed " 
                  << (beforeMinPointFilter - filtered.size()) << " detections with < " << m_minPointsThreshold << " points)" << std::endl;
    }
    
    // 4. Filter by point density (remove sparse detections)
    size_t beforePointDensity = filtered.size();
    filtered = filterByPointDensity(filtered);
    if (m_verbose) {
        std::cout << "After point density filtering: " << filtered.size() << " boxes (removed " 
                  << (beforePointDensity - filtered.size()) << " sparse/dense detections)" << std::endl;
    }
    
    // 5. Filter out walls and large structures
    // size_t beforeWallFilter = filtered.size();
    // filtered = filterWalls(filtered);
    // if (m_verbose) {
    //     std::cout << "After wall filtering: " << filtered.size() << " boxes (removed " 
    //               << (beforeWallFilter - filtered.size()) << " wall sections)" << std::endl;
    // }
    
    // 6. Additional pedestrian false positive filtering
    // size_t beforePedestrianFilter = filtered.size();
    // filtered = filterPedestrianFalsePositives(filtered);
    // if (m_verbose) {
    //     std::cout << "After pedestrian false positive filtering: " << filtered.size() << " boxes (removed " 
    //               << (beforePedestrianFilter - filtered.size()) << " false positives)" << std::endl;
    // }
    
    // Size filtering still disabled - needs tuning based on model output
    // filtered = filterBoundingBoxesBySize(filtered);
    // if (m_verbose) std::cout << "After size filtering: " << filtered.size() << " boxes" << std::endl;
    
    // 7. Re-enable NMS to remove duplicates (now that confidence is correct)
    size_t beforeNMS = filtered.size();
    filtered = applyNMS(filtered, 0.3f);
    if (m_verbose) {
        std::cout << "After NMS: " << filtered.size() << " boxes (removed " 
                  << (beforeNMS - filtered.size()) << " duplicate detections)" << std::endl;
    }
    
    return filtered;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<BoundingBox> InterLidarICP::crossClassNMS(const std::vector<BoundingBox>& detections, float iouThreshold) {
    return applyNMS(detections, iouThreshold);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<BoundingBox> InterLidarICP::classAwareNMS(const std::vector<BoundingBox>& detections, float iouThreshold) {
    std::vector<BoundingBox> result;
    
    // Group by class
    std::vector<std::vector<BoundingBox>> classBuckets(3);
    for (const auto& box : detections) {
        if (box.classId >= 0 && box.classId < 3) {
            classBuckets[box.classId].push_back(box);
        }
    }
    
    // Apply NMS within each class
    for (auto& bucket : classBuckets) {
        auto filtered = applyNMS(bucket, iouThreshold);
        result.insert(result.end(), filtered.begin(), filtered.end());
    }
    
    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<BoundingBox> InterLidarICP::mergeNearbyDetections(const std::vector<BoundingBox>& detections, float mergeThreshold) {
    // Simple implementation - can be expanded for more sophisticated merging
    return detections;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<BoundingBox> InterLidarICP::filterWalls(const std::vector<BoundingBox>& boxes) {
    std::vector<BoundingBox> filtered;
    
    for (const auto& box : boxes) {
        bool isWall = false;
        
        // Method 1: Check aspect ratios - walls are typically very wide/tall and thin
        float widthToHeightRatio = box.width / std::max(box.height, 0.1f);
        float lengthToHeightRatio = box.length / std::max(box.height, 0.1f);
        float widthToLengthRatio = box.width / std::max(box.length, 0.1f);
        
        // Method 2: Check for partial wall detections (slim, tall sections)
        // These are often misclassified as pedestrians
        if (box.classId == 1) { // Pedestrian
            // Check for slim, tall structures that are likely wall sections
            if ((box.width < 1.5f && box.height > 2.5f) || 
                (box.length < 1.5f && box.height > 2.5f)) {
                isWall = true;
            }
            
            // Check for very thin but wide structures (wall slices)
            if ((box.width < 0.8f && box.length > 3.0f) || 
                (box.length < 0.8f && box.width > 3.0f)) {
                isWall = true;
            }
            
            // Check aspect ratios for wall-like characteristics - much less aggressive
            if (widthToHeightRatio > 5.0f || lengthToHeightRatio > 5.0f) {  // Only very wide objects
                isWall = true;
            }
            
            // Additional wall detection: very thin and tall structures (much more restrictive)
            if ((widthToHeightRatio < 0.2f && lengthToHeightRatio < 0.2f) && box.height > 3.0f) {
                isWall = true;
            }
            
            // Check if dimensions are too large for a pedestrian
            if (box.width > 3.0f || box.length > 3.0f || box.height > 3.5f) {
                isWall = true;
            }
        }
        
        // Method 3: Check point distribution for wall-like patterns
        if (!isWall) {
            int pointCount = countPointsInBox(box);
            if (pointCount > 0) {
                // Calculate point density (points per cubic meter)
                float volume = box.width * box.length * box.height;
                float pointDensity = pointCount / std::max(volume, 0.1f);
                
                // Wall sections often have very high point density - much less aggressive
                if (pointDensity > 1200.0f) {  // Much higher threshold - only extremely dense walls
                    isWall = true;
                }
                
                // Check if points are distributed in a wall-like pattern
                // (high density in a thin slice) - much more restrictive
                if (pointCount > 500 && (box.width < 0.5f || box.length < 0.5f)) {  // Much more restrictive
                    isWall = true;
                }
                
                // Additional wall detection: very high point count with reasonable dimensions
                if (pointCount > 800 && pointDensity > 800.0f) {  // Much more restrictive
                    isWall = true;
                }
            }
        }
        
        // Method 4: Check distance and size relationship
        float distance = sqrt(box.x * box.x + box.y * box.y);
        if (distance > 20.0f) {
            // Far objects that are slim and tall are likely wall sections
            if (box.classId == 1 && (box.width < 1.2f || box.length < 1.2f) && box.height > 2.0f) {
                isWall = true;
            }
        }
        
        // Method 5: Check for absolute wall dimensions (full walls)
        if (box.width > 20.0f || box.length > 20.0f) {
            isWall = true;
        }
        
        // Method 6: Check for thin, tall structures (like walls)
        if (box.width < 2.0f && box.height > 3.0f && box.length > 10.0f) {
            isWall = true;
        }
        if (box.length < 2.0f && box.height > 3.0f && box.width > 10.0f) {
            isWall = true;
        }
        
        if (!isWall) {
            filtered.push_back(box);
        } else if (m_verbose) {
            const char* className = (box.classId == 0) ? "Vehicle" : 
                                   (box.classId == 1) ? "Pedestrian" : "Cyclist";
            std::cout << "  REJECTED WALL SECTION: " << className << " at (" << box.x << ", " << box.y << ", " << box.z 
                      << ") size [" << box.width << "x" << box.length << "x" << box.height 
                      << "] ratios [w/h:" << widthToHeightRatio << " l/h:" << lengthToHeightRatio << "]" << std::endl;
            
            // Check if this could be a real person that was incorrectly rejected
            if (box.classId == 1) { // Pedestrian
                float distance = sqrt(box.x * box.x + box.y * box.y);
                bool reasonableSize = (box.width >= 0.3f && box.width <= 1.2f && 
                                      box.length >= 0.3f && box.length <= 1.2f && 
                                      box.height >= 1.4f && box.height <= 2.2f);
                bool reasonableDistance = (distance >= 1.0f && distance <= 15.0f);
                bool reasonableConfidence = (box.confidence >= 0.05f);
                
                if (reasonableSize && reasonableDistance && reasonableConfidence) {
                    std::cout << "    *** WARNING: WALL FILTER MAY HAVE REJECTED REAL PERSON ***" << std::endl;
                    std::cout << "    Consider adjusting wall filter thresholds" << std::endl;
                }
            }
        }
    }
    
    return filtered;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<BoundingBox> InterLidarICP::filterPedestrianFalsePositives(const std::vector<BoundingBox>& boxes) {
    std::vector<BoundingBox> filtered;
    
    for (const auto& box : boxes) {
        bool isFalsePositive = false;
        
        // Only apply this filter to pedestrian detections
        if (box.classId == 1) { // Pedestrian
            float distance = sqrt(box.x * box.x + box.y * box.y);
            int pointCount = countPointsInBox(box);
            float volume = box.width * box.length * box.height;
            float pointDensity = pointCount / std::max(volume, 0.1f);
            
            // Check 1: Unrealistic dimensions for pedestrians
            if (box.width > 1.0f || box.length > 1.0f || box.height < 1.2f || box.height > 2.3f) {
                isFalsePositive = true;
                if (m_verbose) {
                    std::cout << "    REJECTED: Unrealistic pedestrian dimensions [" 
                              << box.width << "x" << box.length << "x" << box.height << "]" << std::endl;
                    
                    // Check if this might actually be a reasonable person
                    bool reasonableSize = (box.width >= 0.3f && box.width <= 1.2f && 
                                          box.length >= 0.3f && box.length <= 1.2f && 
                                          box.height >= 1.4f && box.height <= 2.2f);
                    if (reasonableSize) {
                        std::cout << "    *** WARNING: DIMENSION FILTER MAY HAVE REJECTED REAL PERSON ***" << std::endl;
                    }
                }
            }
            
            // Check 2: Very high point density (likely static objects)
            if (pointDensity > 800.0f) {  // Much higher threshold
                isFalsePositive = true;
                if (m_verbose) {
                    std::cout << "    REJECTED: High point density (" << pointDensity << " pts/m³) - likely static object" << std::endl;
                }
            }
            
            // Check 2.5: Too many points for a pedestrian (likely a large object misclassified)
            if (pointCount > 240) {
                isFalsePositive = true;
                if (m_verbose) {
                    std::cout << "    REJECTED: Too many points (" << pointCount << ") for pedestrian - likely misclassified object" << std::endl;
                }
            }
            
            // Check 3: Too close to ground (ground clutter) - much less aggressive
            // Use ground plane height for proper reference
            if (m_filteredGroundPlaneValid) {
                float groundHeight = -m_filteredGroundPlane.offset;  // Ground plane height relative to sensor
                float objectBottom = box.z - box.height/2.0f;
                float heightAboveGround = objectBottom - groundHeight;
                
                if (heightAboveGround < 0.1f) {  // Much less aggressive - only reject if very close to ground
                    isFalsePositive = true;
                    if (m_verbose) {
                        std::cout << "    REJECTED: Too close to ground (bottom at " << objectBottom
                                  << "m, ground at " << groundHeight << "m, height above ground: " << heightAboveGround << "m)" << std::endl;
                    }
                }
            } else {
                // Fallback: use sensor origin as ground reference
                if (box.z - box.height/2.0f < -0.2f) {  // Much less aggressive
                    isFalsePositive = true;
                    if (m_verbose) {
                        std::cout << "    REJECTED: Too close to ground (bottom at " << (box.z - box.height/2.0f) << "m)" << std::endl;
                    }
                }
            }
            
            // Check 4: Far distance with few points
            if (distance > 12.0f && pointCount < 40) {
                isFalsePositive = true;
                if (m_verbose) {
                    std::cout << "    REJECTED: Far distance (" << distance << "m) with few points (" << pointCount << ")" << std::endl;
                }
            }
            
            // Check 5: Aspect ratios that don't match human proportions
            float widthToHeightRatio = box.width / std::max(box.height, 0.1f);
            float lengthToHeightRatio = box.length / std::max(box.height, 0.1f);
            if (widthToHeightRatio > 1.2f || lengthToHeightRatio > 1.2f) {
                isFalsePositive = true;
                if (m_verbose) {
                    std::cout << "    REJECTED: Non-human aspect ratios [w/h:" << widthToHeightRatio 
                              << " l/h:" << lengthToHeightRatio << "]" << std::endl;
                }
            }
            
            // Check 6: Very low confidence with suspicious characteristics
            if (box.confidence < 0.10f && (pointCount < 20 || distance > 15.0f)) {  // Less aggressive thresholds
                isFalsePositive = true;
                if (m_verbose) {
                    std::cout << "    REJECTED: Low confidence (" << box.confidence 
                              << ") with suspicious characteristics" << std::endl;
                }
            }
            
            // Check 7: Check if it's in a suspicious location (e.g., middle of road)
            // This is a simple heuristic - you might want to make this more sophisticated
            if (std::abs(box.y) < 1.0f && distance > 5.0f) {
                isFalsePositive = true;
                if (m_verbose) {
                    std::cout << "    REJECTED: Suspicious location (y=" << box.y << "m, likely in road)" << std::endl;
                }
            }
        }
        
        if (!isFalsePositive) {
            filtered.push_back(box);
            if (m_verbose && box.classId == 1) {
                float distance = sqrt(box.x * box.x + box.y * box.y);
                int pointCount = countPointsInBox(box);
                float volume = box.width * box.length * box.height;
                float pointDensity = pointCount / std::max(volume, 0.1f);
                std::cout << "    ACCEPTED: Pedestrian at (" << box.x << ", " << box.y << ", " << box.z 
                          << ") conf=" << box.confidence << " points=" << pointCount 
                          << " density=" << pointDensity << " pts/m³" << std::endl;
            }
        }
    }
    
    if (m_verbose) {
        std::cout << "    Pedestrian False Positive Filter Summary: " << filtered.size() 
                  << " accepted out of " << boxes.size() << " input detections" << std::endl;
    }
    
    return filtered;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<BoundingBox> InterLidarICP::filterByClasses(const std::vector<BoundingBox>& boxes, const std::vector<int>& allowedClasses) {
    std::vector<BoundingBox> filtered;
    
    for (const auto& box : boxes) {
        if (std::find(allowedClasses.begin(), allowedClasses.end(), box.classId) != allowedClasses.end()) {
            filtered.push_back(box);
        }
    }
    
    return filtered;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<BoundingBox> InterLidarICP::applyTemporalFiltering(const std::vector<BoundingBox>& currentBoxes) {
    // Simple implementation - can be expanded for sophisticated tracking
    return currentBoxes;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<BoundingBox> InterLidarICP::filterByDistanceBasedConfidence(const std::vector<BoundingBox>& boxes) {
    std::vector<BoundingBox> filtered;
    
    for (const auto& box : boxes) {
        float distance = sqrt(box.x * box.x + box.y * box.y);
        float adjustedThreshold = 0.3f + (distance * 0.01f); // Increase threshold with distance
        
        if (box.confidence >= adjustedThreshold) {
            filtered.push_back(box);
        }
    }
    
    return filtered;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<BoundingBox> InterLidarICP::filterByAspectRatios(const std::vector<BoundingBox>& boxes) {
    std::vector<BoundingBox> filtered;
    
    for (const auto& box : boxes) {
        float aspectRatio = box.length / box.width;
        
        // Define reasonable aspect ratios for each class
        bool validAspectRatio = false;
        switch (box.classId) {
            case 0: // Vehicle
                validAspectRatio = (aspectRatio >= 0.5f && aspectRatio <= 4.0f);
                break;
            case 1: // Pedestrian
                validAspectRatio = (aspectRatio >= 0.5f && aspectRatio <= 2.0f);
                break;
            case 2: // Cyclist
                validAspectRatio = (aspectRatio >= 0.5f && aspectRatio <= 3.0f);
                break;
        }
        
        if (validAspectRatio) {
            filtered.push_back(box);
        }
    }
    
    return filtered;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::adjustBoxToFitPoints(BoundingBox& box) {
    // This is a placeholder - in a real implementation, you would
    // analyze the points within the box and adjust its size/position
    // to better fit the actual object
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::printObjectDetectionStatistics() {
    if (!m_objectDetectionEnabled) {
        return;
    }
    
    std::cout << "\nObject Detection Statistics:" << std::endl;
    std::cout << "  Total Detections: " << m_totalDetections << std::endl;
    std::cout << "  Vehicle Detections: " << m_vehicleDetections << std::endl;
    std::cout << "  Pedestrian Detections: " << m_pedestrianDetections << std::endl;
    std::cout << "  Cyclist Detections: " << m_cyclistDetections << std::endl;
    if (m_frameNum > 0) {
        std::cout << "  Average Detections per Frame: " << (float)m_totalDetections / m_frameNum << std::endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// TODO: OG
// void InterLidarICP::renderGroundPlane()
// {
//     // Use filtered ground plane for more stable visualization
//     if (!m_filteredGroundPlaneValid || !m_filteredGroundPlane.valid) {
//         if (m_verbose) {
//             std::cout << "Ground plane rendering skipped - not valid" << std::endl;
//         }
//         return;
//     }
    
//     // Debug: Print what we're rendering
//     if (m_verbose) {
//         std::cout << "Rendering ground plane with:" << std::endl;
//         std::cout << "  Normal: [" << m_filteredGroundPlane.normal.x << ", " 
//                   << m_filteredGroundPlane.normal.y << ", " << m_filteredGroundPlane.normal.z << "]" << std::endl;
//         std::cout << "  Offset: " << m_filteredGroundPlane.offset << std::endl;
//         std::cout << "  Grid size: " << GROUND_PLANE_GRID_SIZE << "x" << GROUND_PLANE_GRID_SIZE << std::endl;
//         std::cout << "  Cell size: " << GROUND_PLANE_CELL_SIZE << "m" << std::endl;
//     }
    
//     // CLEAR BUFFER EXPLICITLY ON ORIN PLATFORM to avoid stale mesh artifacts
//     #ifdef __aarch64__  // ARM64/Orin specific
//     // Force GPU-CPU memory coherency on Tegra before rendering
//     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
//     #endif

//     // Map render buffer for ground plane
//     dwVector3f* vertices = nullptr;
//     uint32_t maxVerts = 0;
//     uint32_t stride = 0;
    
//     CHECK_DW_ERROR(dwRenderEngine_mapBuffer(m_groundPlaneRenderBufferId,
//                                             reinterpret_cast<void**>(&vertices),
//                                             0,
//                                             GROUND_PLANE_GRID_SIZE * GROUND_PLANE_GRID_SIZE * 6 * sizeof(dwVector3f),
//                                             DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
//                                             m_renderEngine));
    
//     // ZERO OUT BUFFER FIRST ON ORIN to clear any stale mesh data
//     #ifdef __aarch64__
//     uint32_t totalVertsCapacity = GROUND_PLANE_GRID_SIZE * GROUND_PLANE_GRID_SIZE * 6;
//     memset(vertices, 0, totalVertsCapacity * sizeof(dwVector3f));
//     std::cout << "[ORIN-DEBUG] Cleared ground plane buffer (" << totalVertsCapacity << " vertices)" << std::endl;
//     #endif
    
//     // Generate ground plane mesh based on filtered plane
//     uint32_t vertexIndex = 0;
//     float32_t halfSize = GROUND_PLANE_GRID_SIZE * GROUND_PLANE_CELL_SIZE * 0.5f;
    
//     // Get plane normal and offset from the filtered ground plane
//     dwVector3f normal = m_filteredGroundPlane.normal;
//     float32_t offset = m_filteredGroundPlane.offset;
    
//     // Normalize the normal vector if needed
//     float32_t normalMagnitude = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
//     if (normalMagnitude > 0.001f) {
//         normal.x /= normalMagnitude;
//         normal.y /= normalMagnitude;
//         normal.z /= normalMagnitude;
//     }
    
//     // Debug: Print normalized normal and offset
//     if (m_verbose) {
//         std::cout << "  Normalized Normal: [" << normal.x << ", " << normal.y << ", " << normal.z << "]" << std::endl;
//         std::cout << "  Normal Magnitude: " << normalMagnitude << std::endl;
//         std::cout << "  Half Grid Size: " << halfSize << "m" << std::endl;
//         std::cout << "  Grid Range: [" << -halfSize << ", " << halfSize << "] x [" << -halfSize << ", " << halfSize << "]" << std::endl;
//     }
    
//     // Check if plane is too vertical (normal.z close to 0)
//     if (fabs(normal.z) < 0.01f) {
//         if (m_verbose) {
//             std::cout << "Ground plane too vertical, skipping visualization" << std::endl;
//         }
//         CHECK_DW_ERROR(dwRenderEngine_unmapBuffer(m_groundPlaneRenderBufferId,
//                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
//                                                   m_renderEngine));
//         return;
//     }
    
//     // Generate grid vertices
//     uint32_t debugVertexCount = 0;
//     for (uint32_t i = 0; i < GROUND_PLANE_GRID_SIZE - 1; i++) {
//         for (uint32_t j = 0; j < GROUND_PLANE_GRID_SIZE - 1; j++) {
//             float32_t x1 = -halfSize + i * GROUND_PLANE_CELL_SIZE;
//             float32_t y1 = -halfSize + j * GROUND_PLANE_CELL_SIZE;
//             float32_t x2 = x1 + GROUND_PLANE_CELL_SIZE;
//             float32_t y2 = y1 + GROUND_PLANE_CELL_SIZE;
            
//             // Calculate Z coordinates based on plane equation: normal.x*x + normal.y*y + normal.z*z + offset = 0
//             // Solving for z: z = -(normal.x*x + normal.y*y + offset) / normal.z
//             float32_t z1 = -(normal.x * x1 + normal.y * y1 + offset) / normal.z;
//             float32_t z2 = -(normal.x * x2 + normal.y * y1 + offset) / normal.z;
//             float32_t z3 = -(normal.x * x1 + normal.y * y2 + offset) / normal.z;
//             float32_t z4 = -(normal.x * x2 + normal.y * y2 + offset) / normal.z;
            
//             // Debug: Show first few vertices
//             if (debugVertexCount < 4 && m_verbose) {
//                 std::cout << "  Vertex " << debugVertexCount << ": (" << x1 << ", " << y1 << ", " << z1 << ")" << std::endl;
//                 debugVertexCount++;
//             }
            
//             // First triangle
//             vertices[vertexIndex++] = {x1, y1, z1};
//             vertices[vertexIndex++] = {x2, y1, z2};
//             vertices[vertexIndex++] = {x1, y2, z3};
            
//             // Second triangle
//             vertices[vertexIndex++] = {x2, y1, z2};
//             vertices[vertexIndex++] = {x2, y2, z4};
//             vertices[vertexIndex++] = {x1, y2, z3};
//         }
//     }
    
//     if (m_verbose) {
//         std::cout << "  Generated " << vertexIndex << " vertices for ground plane mesh" << std::endl;
//         std::cout << "  Expected vertices: " << (GROUND_PLANE_GRID_SIZE - 1) * (GROUND_PLANE_GRID_SIZE - 1) * 6 << std::endl;
//     }
    
//     CHECK_DW_ERROR(dwRenderEngine_unmapBuffer(m_groundPlaneRenderBufferId,
//                                               DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
//                                               m_renderEngine));
    
//     // Render the ground plane with a pleasant earth-tone color
//     dwRenderEngine_setColor({0.4f, 0.3f, 0.2f, 0.8f}, m_renderEngine);  // Earth brown with transparency
//     dwRenderEngine_renderBuffer(m_groundPlaneRenderBufferId, vertexIndex, m_renderEngine);
// }

void InterLidarICP::renderGroundPlane()
{
    // Check if ground plane visualization is enabled
    if (!m_groundPlaneVisualizationEnabled) {
        if (m_verbose) {
            std::cout << "Ground plane rendering skipped - visualization disabled" << std::endl;
        }
        return;
    }

    // Use filtered ground plane for more stable visualization
    if (!m_filteredGroundPlaneValid || !m_filteredGroundPlane.valid) {
        if (m_verbose) {
            std::cout << "Ground plane rendering skipped - not valid" << std::endl;
        }
        return;
    }
    
    // Calculate exact buffer size needed
    uint32_t expectedVertices = (GROUND_PLANE_GRID_SIZE - 1) * (GROUND_PLANE_GRID_SIZE - 1) * 6;
    uint32_t bufferSizeBytes = expectedVertices * sizeof(dwVector3f);
    
    if (m_verbose) {
        std::cout << "Rendering ground plane with:" << std::endl;
        std::cout << "  Normal: [" << m_filteredGroundPlane.normal.x << ", " 
                  << m_filteredGroundPlane.normal.y << ", " << m_filteredGroundPlane.normal.z << "]" << std::endl;
        std::cout << "  Offset: " << m_filteredGroundPlane.offset << std::endl;
        std::cout << "  Expected vertices: " << expectedVertices << std::endl;
        std::cout << "  Buffer size: " << bufferSizeBytes << " bytes" << std::endl;
    }
    
    // ORIN PLATFORM-SPECIFIC SYNCHRONIZATION
    #ifdef __aarch64__
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    #endif

    // Map render buffer - use EXACT size needed
    dwVector3f* vertices = nullptr;
    CHECK_DW_ERROR(dwRenderEngine_mapBuffer(m_groundPlaneRenderBufferId,
                                            reinterpret_cast<void**>(&vertices),
                                            0,
                                            bufferSizeBytes,  // FIXED: Use exact size
                                            DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
                                            m_renderEngine));
    
    // AARCH64-SPECIFIC: CUDA-native buffer management (DriveWorks mapped buffers)
    #ifdef __aarch64__
    // Record that we're about to modify the buffer
    CHECK_CUDA_ERROR(cudaEventRecord(m_cpuWriteCompleteEvent, m_memoryCoherencyStream));
    
    // DEBUG: Check for sentinel values (should be -999.0f if properly initialized)
    if (m_verbose) {
        std::cout << "[CUDA-NATIVE] Buffer content before clear:" << std::endl;
        for (uint32_t i = 0; i < std::min(5u, expectedVertices); i++) {
            std::cout << "  Vertex[" << i << "]: (" << vertices[i].x << ", " 
                      << vertices[i].y << ", " << vertices[i].z << ")" << std::endl;
        }
        
        // Check for stale data patterns
        uint32_t sentinelCount = 0;
        uint32_t zeroCount = 0;
        uint32_t randomCount = 0;
        
        for (uint32_t i = 0; i < expectedVertices; i++) {
            if (vertices[i].z == -999.0f) {
                sentinelCount++;
            } else if (vertices[i].x == 0.0f && vertices[i].y == 0.0f && vertices[i].z == 0.0f) {
                zeroCount++;
            } else {
                randomCount++;
            }
        }
        
        std::cout << "[BUFFER-ANALYSIS] Sentinel: " << sentinelCount 
                  << ", Zero: " << zeroCount << ", Random/Stale: " << randomCount << std::endl;
        
        if (randomCount > 0) {
            std::cout << "[WARNING] Found " << randomCount 
                      << " vertices with potentially stale data!" << std::endl;
        }
    }
    
    // CUDA-native buffer clearing (CPU operations on mapped buffers)
    memset(vertices, 0, bufferSizeBytes);
    
    // CUDA-native memory barrier to ensure coherency
    CHECK_CUDA_ERROR(cudaEventRecord(m_cpuWriteCompleteEvent, m_memoryCoherencyStream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_memoryCoherencyStream));
    
    if (m_verbose) {
        std::cout << "[CUDA-NATIVE] Cleared buffer with memset + CUDA synchronization (" << expectedVertices 
                  << " vertices, " << bufferSizeBytes << " bytes)" << std::endl;
    }
    #else
    // Standard CPU buffer clearing for x86
    memset(vertices, 0, bufferSizeBytes);
    #endif
    
    // Generate ground plane mesh
    uint32_t vertexIndex = 0;
    float32_t halfSize = GROUND_PLANE_GRID_SIZE * GROUND_PLANE_CELL_SIZE * 0.5f;
    
    // Get plane normal and offset from the filtered ground plane
    dwVector3f normal = m_filteredGroundPlane.normal;
    float32_t offset = m_filteredGroundPlane.offset;
    
    // Normalize the normal vector with enhanced precision
    double normalMagnitude = sqrt(static_cast<double>(normal.x) * normal.x + 
                                static_cast<double>(normal.y) * normal.y + 
                                static_cast<double>(normal.z) * normal.z);
    if (normalMagnitude > 0.001) {
        normal.x = static_cast<float32_t>(normal.x / normalMagnitude);
        normal.y = static_cast<float32_t>(normal.y / normalMagnitude);
        normal.z = static_cast<float32_t>(normal.z / normalMagnitude);
    }
    
    // Check if plane is too vertical
    if (fabs(normal.z) < 0.1f) {
        if (m_verbose) {
            std::cout << "Ground plane too vertical, skipping visualization" << std::endl;
        }
        CHECK_DW_ERROR(dwRenderEngine_unmapBuffer(m_groundPlaneRenderBufferId,
                                                  DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
                                                  m_renderEngine));
        return;
    }
    
    // Generate mesh vertices
    for (uint32_t i = 0; i < GROUND_PLANE_GRID_SIZE - 1; i++) {
        for (uint32_t j = 0; j < GROUND_PLANE_GRID_SIZE - 1; j++) {
            // Calculate grid positions
            float32_t x1 = -halfSize + i * GROUND_PLANE_CELL_SIZE;
            float32_t y1 = -halfSize + j * GROUND_PLANE_CELL_SIZE;
            float32_t x2 = x1 + GROUND_PLANE_CELL_SIZE;
            float32_t y2 = y1 + GROUND_PLANE_CELL_SIZE;
            
            // Calculate Z coordinates with higher precision
            double z1_precise = -(static_cast<double>(normal.x) * x1 + 
                                static_cast<double>(normal.y) * y1 + 
                                static_cast<double>(offset)) / static_cast<double>(normal.z);
            double z2_precise = -(static_cast<double>(normal.x) * x2 + 
                                static_cast<double>(normal.y) * y1 + 
                                static_cast<double>(offset)) / static_cast<double>(normal.z);
            double z3_precise = -(static_cast<double>(normal.x) * x1 + 
                                static_cast<double>(normal.y) * y2 + 
                                static_cast<double>(offset)) / static_cast<double>(normal.z);
            double z4_precise = -(static_cast<double>(normal.x) * x2 + 
                                static_cast<double>(normal.y) * y2 + 
                                static_cast<double>(offset)) / static_cast<double>(normal.z);
            
            // Convert back to float with bounds checking
            const double maxReasonableZ = 5.0;
            float32_t z1 = static_cast<float32_t>(std::max(-maxReasonableZ, std::min(maxReasonableZ, z1_precise)));
            float32_t z2 = static_cast<float32_t>(std::max(-maxReasonableZ, std::min(maxReasonableZ, z2_precise)));
            float32_t z3 = static_cast<float32_t>(std::max(-maxReasonableZ, std::min(maxReasonableZ, z3_precise)));
            float32_t z4 = static_cast<float32_t>(std::max(-maxReasonableZ, std::min(maxReasonableZ, z4_precise)));
            
            // CRITICAL: Verify we don't exceed buffer bounds
            if (vertexIndex + 6 > expectedVertices) {
                logError("Ground plane buffer overflow! VertexIndex: %d, Expected: %d", vertexIndex, expectedVertices);
                break;
            }
            
            // Generate triangles with correct winding order
            // First triangle (counter-clockwise)
            vertices[vertexIndex++] = {x1, y1, z1};
            vertices[vertexIndex++] = {x2, y1, z2};
            vertices[vertexIndex++] = {x1, y2, z3};
            
            // Second triangle (counter-clockwise)
            vertices[vertexIndex++] = {x2, y1, z2};
            vertices[vertexIndex++] = {x2, y2, z4};
            vertices[vertexIndex++] = {x1, y2, z3};
        }
    }
    
    // CRITICAL: Verify exact vertex count match
    if (vertexIndex != expectedVertices) {
        logError("CRITICAL: Vertex count mismatch! Generated: %d, Expected: %d", vertexIndex, expectedVertices);
        logError("This indicates a serious buffer management bug!");
    } else if (m_verbose) {
        std::cout << "  Generated " << vertexIndex << " vertices - EXACT MATCH!" << std::endl;
    }
    
    // AARCH64-SPECIFIC: CUDA-native coherency management before GPU rendering
    #ifdef __aarch64__
    if (m_verbose) {
        std::cout << "[CUDA-NATIVE] Final buffer validation:" << std::endl;
        for (uint32_t i = 0; i < std::min(3u, vertexIndex); i++) {
            std::cout << "  Final Vertex[" << i << "]: (" << vertices[i].x << ", " 
                      << vertices[i].y << ", " << vertices[i].z << ")" << std::endl;
        }
    }
    
    // CUDA-native: Ensure all operations complete before GPU rendering
    CHECK_CUDA_ERROR(cudaEventRecord(m_cpuWriteCompleteEvent, m_memoryCoherencyStream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_memoryCoherencyStream));
    #endif
    
    CHECK_DW_ERROR(dwRenderEngine_unmapBuffer(m_groundPlaneRenderBufferId,
                                              DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
                                              m_renderEngine));
    
    // CUDA-native: Final rendering synchronization
    #ifdef __aarch64__
    // Record that GPU is ready to render
    CHECK_CUDA_ERROR(cudaEventRecord(m_gpuReadReadyEvent, m_memoryCoherencyStream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_memoryCoherencyStream));
    
    if (m_verbose) {
        std::cout << "[CUDA-NATIVE] Unified memory coherency complete - GPU ready to render" << std::endl;
    }
    #endif
    
    // Render with exact vertex count
    dwRenderEngine_setColor({0.4f, 0.3f, 0.2f, 0.7f}, m_renderEngine);
    dwRenderEngine_renderBuffer(m_groundPlaneRenderBufferId, vertexIndex, m_renderEngine);
    
    if (m_verbose) {
        std::cout << "Ground plane rendered with EXACT " << vertexIndex << " vertices (CUDA-native coherency)" << std::endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::renderBoundingBoxes()
{
    if (!m_objectDetectionEnabled || m_currentBoxes.empty()) {
        return;
    }
    
    // Set up 3D transformation for rendering
    dwMatrix4f modelView;
    Mat4_AxB(modelView.array, getMouseView().getModelView()->array, DW_IDENTITY_TRANSFORMATION3F.array);
    dwRenderEngine_setModelView(&modelView, m_renderEngine);
    dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine);
    
    // Prepare text labels for rendering
    std::vector<std::pair<dwVector3f, std::string>> labelTexts;    // Above box - class & confidence
    std::vector<std::pair<dwVector3f, std::string>> debugTexts;    // Below box - point count & distance
    
    // Group boxes by class for separate rendering
    std::vector<BoundingBox> vehicleBoxes;
    std::vector<BoundingBox> pedestrianBoxes;
    std::vector<BoundingBox> cyclistBoxes;
    
    for (const auto& box : m_currentBoxes) {
        // Classify boxes by type
        switch (box.classId) {
            case 0: vehicleBoxes.push_back(box); break;
            case 1: pedestrianBoxes.push_back(box); break;
            case 2: cyclistBoxes.push_back(box); break;
            default: vehicleBoxes.push_back(box); break; // Unknown -> Vehicle
        }
        
        // Prepare text labels
        float halfHeight = box.height / 2.0f;
        
        // Above box - class & confidence
        dwVector3f labelPos = {box.x, box.y, box.z + halfHeight + 0.3f};
        
        // Below box - point count & distance  
        dwVector3f debugPos = {box.x, box.y, box.z - halfHeight - 0.3f};
        
        // Class name mapping
        std::string className;
        switch (box.classId) {
            case 0: className = "Vehicle"; break;
            case 1: className = "Pedestrian"; break;
            case 2: className = "Cyclist"; break;
            default: className = "Unknown"; break;
        }
        
        // Count points in box and calculate distance
        int pointCount = countPointsInBox(box);
        float distance = std::sqrt(box.x * box.x + box.y * box.y);
        
        // Class, track ID (if available) & confidence text
        char labelText[64];
        if (box.trackId >= 0)
        {
            // Example: "Vehicle#12 (0.85)"
            snprintf(labelText, sizeof(labelText), "%s#%d (%.2f)",
                     className.c_str(), box.trackId, box.confidence);
        }
        else
        {
            // Fallback: no ID available
            snprintf(labelText, sizeof(labelText), "%s (%.2f)",
                     className.c_str(), box.confidence);
        }
        labelTexts.emplace_back(labelPos, std::string(labelText));
        
        // Debug text
        char debugText[64];
        snprintf(debugText, sizeof(debugText), "Pts: %d, Dist: %.1fm", pointCount, distance);
        debugTexts.emplace_back(debugPos, std::string(debugText));
        
        // Highlight points within this bounding box in yellow
        renderPointsInBox(box, {box.x, box.y, box.z}, 1.2f);
    }
    
    // Helper function to render a group of boxes with the same color
    auto renderBoxGroup = [&](const std::vector<BoundingBox>& boxes, dwRenderEngineColorRGBA color) {
        if (boxes.empty()) return;
        
        // Calculate exact buffer size needed
        uint32_t expectedVertices = boxes.size() * 24;  // 12 edges × 2 vertices per edge
        uint32_t bufferSizeBytes = expectedVertices * sizeof(dwVector3f);
        
        // ORIN PLATFORM-SPECIFIC SYNCHRONIZATION
        #ifdef __aarch64__
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        if (m_verbose) {
            std::cout << "[BBOX-DEBUG] Rendering " << boxes.size() << " boxes, " 
                      << expectedVertices << " vertices, " << bufferSizeBytes << " bytes" << std::endl;
        }
        #endif
        
        // Map the render buffer for this group - use EXACT size needed
        dwVector3f* vertices = nullptr;
        CHECK_DW_ERROR(dwRenderEngine_mapBuffer(m_boxLineBuffer,
                                                reinterpret_cast<void**>(&vertices),
                                                0,
                                                bufferSizeBytes,  // FIXED: Use exact size
                                                DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                                m_renderEngine));
        
        // AARCH64-SPECIFIC: CUDA-native buffer management for bounding boxes
        #ifdef __aarch64__
        // Record that we're about to modify the buffer
        CHECK_CUDA_ERROR(cudaEventRecord(m_cpuWriteCompleteEvent, m_memoryCoherencyStream));
        
        // DEBUG: Check for stale data in line buffer
        if (m_verbose) {
            std::cout << "[BBOX-BUFFER] Buffer content before clear:" << std::endl;
            uint32_t zeroCount = 0, randomCount = 0;
            
            for (uint32_t i = 0; i < std::min(expectedVertices, 10u); i++) {
                if (vertices[i].x == 0.0f && vertices[i].y == 0.0f && vertices[i].z == 0.0f) {
                    zeroCount++;
                } else {
                    randomCount++;
                    if (i < 3) {  // Show first few non-zero vertices
                        std::cout << "  Vertex[" << i << "]: (" << vertices[i].x << ", " 
                                  << vertices[i].y << ", " << vertices[i].z << ")" << std::endl;
                    }
                }
            }
            
            std::cout << "[BBOX-ANALYSIS] Zero: " << zeroCount << ", Stale: " << randomCount << std::endl;
            
            if (randomCount > 0) {
                std::cout << "[WARNING] Found " << randomCount 
                          << " vertices with potentially stale line data!" << std::endl;
            }
        }
        
        // CUDA-native buffer clearing (CPU operations on mapped buffers)
        memset(vertices, 0, bufferSizeBytes);
        
        // CUDA-native memory barrier to ensure coherency
        CHECK_CUDA_ERROR(cudaEventRecord(m_cpuWriteCompleteEvent, m_memoryCoherencyStream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(m_memoryCoherencyStream));
        
        if (m_verbose) {
            std::cout << "[BBOX-CUDA-NATIVE] Cleared line buffer with CUDA synchronization (" 
                      << expectedVertices << " vertices)" << std::endl;
        }
        #else
        // Standard CPU buffer clearing for x86
        memset(vertices, 0, bufferSizeBytes);
        #endif
        
        uint32_t vertexIndex = 0;
        
        for (const auto& box : boxes) {
            // Calculate box corners
            float halfWidth = box.width / 2.0f;
            float halfLength = box.length / 2.0f;
            float halfHeight = box.height / 2.0f;
            
            // Apply rotation around Z-axis (yaw)
            float cosYaw = cos(box.rotation);
            float sinYaw = sin(box.rotation);
            
            // Debug: Print rotation information for first few boxes
            if (m_verbose && vertexIndex == 0) {
                std::cout << "[ROTATION-DEBUG] Box rotation: " << box.rotation 
                          << " radians (" << (box.rotation * 180.0f / 3.14159f) << " degrees)" << std::endl;
                std::cout << "[ROTATION-DEBUG] cos(yaw)=" << cosYaw << ", sin(yaw)=" << sinYaw << std::endl;
            }
            
            // Define 8 corners of the bounding box with rotation applied
            // Create local coordinates first, then apply rotation and translation
            float localCorners[8][3] = {
                {-halfWidth, -halfLength, -halfHeight}, // Bottom corners
                { halfWidth, -halfLength, -halfHeight},
                { halfWidth,  halfLength, -halfHeight},
                {-halfWidth,  halfLength, -halfHeight},
                {-halfWidth, -halfLength,  halfHeight}, // Top corners
                { halfWidth, -halfLength,  halfHeight},
                { halfWidth,  halfLength,  halfHeight},
                {-halfWidth,  halfLength,  halfHeight}
            };
            
            // Apply rotation and translation to get world coordinates
            dwVector3f corners[8];
            for (int i = 0; i < 8; ++i) {
                float x = localCorners[i][0];
                float y = localCorners[i][1];
                float z = localCorners[i][2];
                
                // Apply rotation around Z-axis (yaw)
                float rotatedX = x * cosYaw - y * sinYaw;
                float rotatedY = x * sinYaw + y * cosYaw;
                float rotatedZ = z; // No rotation around Z for height
                
                // Apply translation (move to world position)
                corners[i] = {
                    box.x + rotatedX,
                    box.y + rotatedY,
                    box.z + rotatedZ
                };
            }
            
            // Define the 12 edges of the box (each edge needs 2 vertices)
            int edges[12][2] = {
                // Bottom face edges
                {0, 1}, {1, 2}, {2, 3}, {3, 0},
                // Top face edges
                {4, 5}, {5, 6}, {6, 7}, {7, 4},
                // Vertical edges
                {0, 4}, {1, 5}, {2, 6}, {3, 7}
            };
            
            // Add vertices for each edge
            for (int i = 0; i < 12; ++i) {
                // CRITICAL: Verify we don't exceed buffer bounds
                if (vertexIndex + 2 > expectedVertices) {
                    logError("Bounding box buffer overflow! VertexIndex: %d, Expected: %d", vertexIndex, expectedVertices);
                    break;
                }
                
                vertices[vertexIndex++] = corners[edges[i][0]];
                vertices[vertexIndex++] = corners[edges[i][1]];
            }
        }
        
        // CRITICAL: Verify exact vertex count match
        if (vertexIndex != expectedVertices) {
            logError("CRITICAL: Bounding box vertex count mismatch! Generated: %d, Expected: %d", vertexIndex, expectedVertices);
            logError("This indicates a serious line buffer management bug!");
        } else if (m_verbose) {
            std::cout << "[BBOX-DEBUG] Generated " << vertexIndex << " line vertices - EXACT MATCH!" << std::endl;
        }
        
        // AARCH64-SPECIFIC: CUDA-native coherency management before GPU rendering
        #ifdef __aarch64__
        if (m_verbose) {
            std::cout << "[BBOX-CUDA-NATIVE] Final line buffer validation:" << std::endl;
            for (uint32_t i = 0; i < std::min(6u, vertexIndex); i++) {
                std::cout << "  Final LineVertex[" << i << "]: (" << vertices[i].x << ", " 
                          << vertices[i].y << ", " << vertices[i].z << ")" << std::endl;
            }
        }
        
        // CUDA-native: Ensure all operations complete before GPU rendering
        CHECK_CUDA_ERROR(cudaEventRecord(m_cpuWriteCompleteEvent, m_memoryCoherencyStream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(m_memoryCoherencyStream));
        #endif
        
        CHECK_DW_ERROR(dwRenderEngine_unmapBuffer(m_boxLineBuffer,
                                                  DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                                  m_renderEngine));
        
        // CUDA-native: Final rendering synchronization for line buffer
        #ifdef __aarch64__
        // Record that GPU is ready to render lines
        CHECK_CUDA_ERROR(cudaEventRecord(m_gpuReadReadyEvent, m_memoryCoherencyStream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(m_memoryCoherencyStream));
        
        if (m_verbose) {
            std::cout << "[BBOX-CUDA-NATIVE] Line buffer unified memory coherency complete - GPU ready to render" << std::endl;
        }
        #endif
        
        // Set color and render this group with exact vertex count
        dwRenderEngine_setColor(color, m_renderEngine);
        dwRenderEngine_setLineWidth(2.0f, m_renderEngine);
        dwRenderEngine_renderBuffer(m_boxLineBuffer, vertexIndex, m_renderEngine);
        
        if (m_verbose) {
            std::cout << "[BBOX-DEBUG] Rendered " << boxes.size() << " boxes with EXACT " 
                      << vertexIndex << " line vertices (CUDA-native coherency)" << std::endl;
        }
    };
    
    // Render each class with its specific color
    renderBoxGroup(vehicleBoxes, {1.0f, 0.0f, 0.0f, 1.0f});     // Red for vehicles
    renderBoxGroup(pedestrianBoxes, {0.0f, 1.0f, 0.0f, 1.0f});  // Green for pedestrians
    renderBoxGroup(cyclistBoxes, {0.0f, 0.0f, 1.0f, 1.0f});     // Blue for cyclists
    
    // Render label texts (class & confidence) in white
    dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_renderEngine);
    dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, m_renderEngine); // White
    
    for (const auto& text : labelTexts) {
        dwRenderEngine_renderText3D(text.second.c_str(), text.first, m_renderEngine);
    }
    
    // Render debug texts (point count & distance) in yellow
    dwRenderEngine_setColor({1.0f, 1.0f, 0.0f, 1.0f}, m_renderEngine); // Yellow
    
    for (const auto& text : debugTexts) {
        dwRenderEngine_renderText3D(text.second.c_str(), text.first, m_renderEngine);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::renderPointsInBox(const BoundingBox& box, const dwVector3f& position, float lidarHeight)
{
    if (!m_stitchedPointsHost.points || m_stitchedPointsHost.size == 0) {
        return;
    }
    
    // Calculate bounding box boundaries
    float xMin = box.x - box.width/2;
    float xMax = box.x + box.width/2;
    float yMin = box.y - box.length/2;
    float yMax = box.y + box.length/2;
    float zMin = box.z - box.height/2;
    float zMax = box.z + box.height/2;
    
    // Find points inside this bounding box
    std::vector<dwVector3f> pointsInBox;
    const dwVector4f* points = static_cast<const dwVector4f*>(m_stitchedPointsHost.points);
    
    for (uint32_t i = 0; i < m_stitchedPointsHost.size; ++i) {
        float x = points[i].x;
        float y = points[i].y;
        float z = points[i].z;
        
        if (x >= xMin && x <= xMax && 
            y >= yMin && y <= yMax && 
            z >= zMin && z <= zMax) {
            pointsInBox.push_back({x, y, z});
        }
    }
    
    if (!pointsInBox.empty()) {
        // Set up for highlighting points
        dwRenderEngine_setPointSize(5.0f, m_renderEngine); // Make points bigger for visibility
        dwRenderEngine_setColor({1.0f, 1.0f, 0.0f, 1.0f}, m_renderEngine); // Yellow for emphasis
        
        // Render highlighted points inside the box
        dwRenderEngine_render(
            DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
            pointsInBox.data(),
            sizeof(dwVector3f),
            0,
            pointsInBox.size(),
            m_renderEngine
        );
        
        // Reset point size to default
        dwRenderEngine_setPointSize(1.0f, m_renderEngine);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::convertBEVToImage(const std::vector<float>& bevData, std::vector<uint8_t>& outImage)
{
    // BEV shape: C x H x W = PFE_OUTPUT_DIM x BEV_H x BEV_W
    // Convert to grayscale by taking mean across channels, then normalize to 0-255
    const int C = PFE_OUTPUT_DIM;
    const int H = BEV_H;
    const int W = BEV_W;
    
    outImage.resize(H * W * 3); // RGB image
    
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            // Compute mean across channels for this spatial location
            float sum = 0.0f;
            for (int c = 0; c < C; ++c) {
                int idx = c * H * W + y * W + x;
                sum += bevData[idx];
            }
            float mean = sum / C;
            
            // Normalize to 0-1 range (assuming features are roughly in [-1, 1] or [0, 1])
            // Apply tanh to squash to [-1, 1], then map to [0, 1]
            float normalized = (std::tanh(mean) + 1.0f) * 0.5f;
            
            // Convert to 0-255 and apply colormap (grayscale for now)
            uint8_t gray = static_cast<uint8_t>(normalized * 255.0f);
            int imgIdx = (y * W + x) * 3;
            outImage[imgIdx + 0] = gray;     // R
            outImage[imgIdx + 1] = gray;     // G
            outImage[imgIdx + 2] = gray;     // B
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::convertHeatmapToImage(const std::vector<float>& heatmapData, std::vector<uint8_t>& outImage)
{
    // Heatmap shape: H x W = OUTPUT_H x OUTPUT_W
    // Normalize scores to 0-255 and apply colormap (jet/hot colormap)
    const int H = OUTPUT_H;
    const int W = OUTPUT_W;
    
    outImage.resize(H * W * 3); // RGB image
    
    // Find min/max for normalization
    float minVal = *std::min_element(heatmapData.begin(), heatmapData.end());
    float maxVal = *std::max_element(heatmapData.begin(), heatmapData.end());
    
    // Debug output (first frame only)
    static bool firstFrame = true;
    if (firstFrame && m_verbose) {
        std::cout << "[Heatmap] Min=" << minVal << ", Max=" << maxVal << std::endl;
        firstFrame = false;
    }
    
    // Use percentile-based normalization to handle outliers
    // Find 95th percentile for better visualization
    std::vector<float> sorted = heatmapData;
    std::sort(sorted.begin(), sorted.end());
    float p95 = sorted[static_cast<size_t>(sorted.size() * 0.95f)];
    float p5 = sorted[static_cast<size_t>(sorted.size() * 0.05f)];
    
    // Clamp to reasonable range (scores are typically 0-1, but might have outliers)
    float displayMax = std::max(0.1f, std::min(1.0f, p95)); // At least 0.1, cap at 1.0
    float displayMin = std::max(0.0f, p5);
    float range = std::max(0.001f, displayMax - displayMin);
    
    // Simple jet colormap: blue -> cyan -> green -> yellow -> red
    auto jetColormap = [](float t) -> std::tuple<uint8_t, uint8_t, uint8_t> {
        t = std::max(0.0f, std::min(1.0f, t));
        float r, g, b;
        if (t < 0.25f) {
            b = 1.0f;
            g = t * 4.0f;
            r = 0.0f;
        } else if (t < 0.5f) {
            b = 1.0f - (t - 0.25f) * 4.0f;
            g = 1.0f;
            r = 0.0f;
        } else if (t < 0.75f) {
            b = 0.0f;
            g = 1.0f;
            r = (t - 0.5f) * 4.0f;
        } else {
            b = 0.0f;
            g = 1.0f - (t - 0.75f) * 4.0f;
            r = 1.0f;
        }
        return {static_cast<uint8_t>(r * 255.0f),
                static_cast<uint8_t>(g * 255.0f),
                static_cast<uint8_t>(b * 255.0f)};
    };
    
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int idx = y * W + x;
            // Clamp and normalize
            float clamped = std::max(displayMin, std::min(displayMax, heatmapData[idx]));
            float normalized = (clamped - displayMin) / range;
            auto [r, g, b] = jetColormap(normalized);
            
            int imgIdx = (y * W + x) * 3;
            outImage[imgIdx + 0] = r;
            outImage[imgIdx + 1] = g;
            outImage[imgIdx + 2] = b;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::renderBEVFeatureMap()
{
    if (!m_bevVisualizationEnabled || m_bevFeatureMap.empty()) {
        return;
    }
    
    // Convert BEV to image data (refresh each frame)
    convertBEVToImage(m_bevFeatureMap, m_bevImageData);
    
    // Set tile for BEV visualization
    dwRenderEngine_setTile(m_bevTile.tileId, m_renderEngine);
    dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
    
    // Set 2D coordinate range to match image dimensions
    dwVector2f range{static_cast<float>(BEV_W), static_cast<float>(BEV_H)};
    dwRenderEngine_setCoordinateRange2D(range, m_renderEngine);
    
    // OPTIMIZED: Use point rendering instead of triangles (much faster)
    // Render every 4th pixel as a colored point
    const int step = 4;
    std::vector<dwVector2f> points;
    std::vector<dwRenderEngineColorRGBA> colors;
    points.reserve((BEV_H / step) * (BEV_W / step));
    colors.reserve((BEV_H / step) * (BEV_W / step));
    
    for (int y = 0; y < BEV_H; y += step) {
        for (int x = 0; x < BEV_W; x += step) {
            int idx = (y * BEV_W + x) * 3;
            if (idx + 2 < static_cast<int>(m_bevImageData.size())) {
                uint8_t r = m_bevImageData[idx];
                uint8_t g = m_bevImageData[idx + 1];
                uint8_t b = m_bevImageData[idx + 2];
                
                points.push_back({static_cast<float>(x), static_cast<float>(y)});
                colors.push_back({r / 255.0f, g / 255.0f, b / 255.0f, 1.0f});
            }
        }
    }
    
    // Render all points at once (much faster than individual triangles)
    if (!points.empty()) {
        dwRenderEngine_setPointSize(static_cast<float>(step), m_renderEngine);
        for (size_t i = 0; i < points.size(); ++i) {
            dwRenderEngine_setColor(colors[i], m_renderEngine);
            dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_2D,
                                 &points[i], sizeof(dwVector2f), 0, 1, m_renderEngine);
        }
        dwRenderEngine_setPointSize(1.0f, m_renderEngine);
    }
    
    // Add label
    dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine);
    dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, m_renderEngine);
    dwRenderEngine_renderText2D("BEV Feature Map", {10.0f, 30.0f}, m_renderEngine);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::renderHeatmap()
{
    if (!m_heatmapVisualizationEnabled || m_heatmapData.empty()) {
        return;
    }
    
    // Convert heatmap to image data (refresh each frame)
    convertHeatmapToImage(m_heatmapData, m_heatmapImageData);
    
    // Set tile for heatmap visualization
    dwRenderEngine_setTile(m_heatmapTile.tileId, m_renderEngine);
    dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
    
    // Set 2D coordinate range to match image dimensions
    dwVector2f range{static_cast<float>(OUTPUT_W), static_cast<float>(OUTPUT_H)};
    dwRenderEngine_setCoordinateRange2D(range, m_renderEngine);
    
    // OPTIMIZED: Use point rendering instead of triangles (much faster)
    // Render every 2nd pixel as a colored point
    const int step = 2;
    std::vector<dwVector2f> points;
    std::vector<dwRenderEngineColorRGBA> colors;
    points.reserve((OUTPUT_H / step) * (OUTPUT_W / step));
    colors.reserve((OUTPUT_H / step) * (OUTPUT_W / step));
    
    for (int y = 0; y < OUTPUT_H; y += step) {
        for (int x = 0; x < OUTPUT_W; x += step) {
            int idx = (y * OUTPUT_W + x) * 3;
            if (idx + 2 < static_cast<int>(m_heatmapImageData.size())) {
                uint8_t r = m_heatmapImageData[idx];
                uint8_t g = m_heatmapImageData[idx + 1];
                uint8_t b = m_heatmapImageData[idx + 2];
                
                points.push_back({static_cast<float>(x), static_cast<float>(y)});
                colors.push_back({r / 255.0f, g / 255.0f, b / 255.0f, 1.0f});
            }
        }
    }
    
    // Render all points at once (much faster than individual triangles)
    if (!points.empty()) {
        dwRenderEngine_setPointSize(static_cast<float>(step), m_renderEngine);
        for (size_t i = 0; i < points.size(); ++i) {
            dwRenderEngine_setColor(colors[i], m_renderEngine);
            dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_2D,
                                 &points[i], sizeof(dwVector2f), 0, 1, m_renderEngine);
        }
        dwRenderEngine_setPointSize(1.0f, m_renderEngine);
    }
    
    // Add label
    dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine);
    dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, m_renderEngine);
    dwRenderEngine_renderText2D("CenterPoint Heatmap", {10.0f, 30.0f}, m_renderEngine);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::logICPResults()
{
    if (m_logFile.is_open()) {
        m_logFile << m_frameNum << ","
                  << getStateString() << ","
                  << (shouldPerformICP() ? "1" : "0") << ","
                  << (m_lastICPStats.successful ? "1" : "0") << ","
                  << m_lastICPStats.iterations << ","
                  << m_lastICPStats.correspondences << ","
                  << m_lastICPStats.rmsCost << ","
                  << m_lastICPStats.inlierFraction << ","
                  << m_lastICPStats.processingTime << ","
                  << (m_groundPlaneValid ? "1" : "0") << ","
                  << (m_filteredGroundPlaneValid ? "1" : "0") << ","
                  << m_consecutiveSuccessfulICP << std::endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::savePointCloudToPLY(const dwPointCloud& pointCloud, const std::string& filename)
{
    if (!m_savePointClouds) return;
    
    // This is a simplified PLY writer - you would implement full PLY writing here
    std::ofstream file(filename);
    if (!file.is_open()) {
        logWarn("Could not open PLY file: %s", filename.c_str());
        return;
    }
    
    // Write PLY header
    file << "ply" << std::endl;
    file << "format ascii 1.0" << std::endl;
    file << "element vertex " << pointCloud.size << std::endl;
    file << "property float x" << std::endl;
    file << "property float y" << std::endl;
    file << "property float z" << std::endl;
    file << "property float intensity" << std::endl;
    file << "end_header" << std::endl;
    
    // Copy point cloud to CPU for writing
    dwPointCloud hostPointCloud;
    hostPointCloud.capacity = pointCloud.capacity;
    hostPointCloud.size = pointCloud.size;
    hostPointCloud.type = DW_MEMORY_TYPE_CPU;
    hostPointCloud.format = DW_POINTCLOUD_FORMAT_XYZI;
    CHECK_DW_ERROR(dwPointCloud_createBuffer(&hostPointCloud));
    
    CHECK_CUDA_ERROR(cudaMemcpy(hostPointCloud.points, pointCloud.points,
                                sizeof(dwVector4f) * pointCloud.size,
                                cudaMemcpyDeviceToHost));
    
    // Write points
    const dwVector4f* points = static_cast<const dwVector4f*>(hostPointCloud.points);
    for (uint32_t i = 0; i < hostPointCloud.size; ++i) {
        file << points[i].x << " " << points[i].y << " " << points[i].z << " " << points[i].w << std::endl;
    }
    
    file.close();
    dwPointCloud_destroyBuffer(&hostPointCloud);
    
    if (m_verbose) {
        std::cout << "Saved point cloud to " << filename << std::endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::onProcess()
{
    // Check if paused
    if (m_paused) {
        return;
    }

    if (m_frameNum++ >= m_numFrames) {
        stop();
        return;
    }

    // std::cout << "========== Frame: " << m_frameNum << " | State: " << getStateString() << " ==========" << std::endl;

    // Step 1: Get point clouds from both lidars (ALWAYS)
    if (!getSpinFromBothLidars()) {
        logError("Failed to get data from both lidars");
        stop();
        return;
    }

    // Step 2: Apply rig transformations (ALWAYS)
    applyRigTransformations();

    // Step 3: Perform ICP alignment (CONDITIONAL based on state)
    bool icpPerformed = false;
    bool icpSuccess = false;
    
    if (shouldPerformICP()) {
        icpPerformed = true;
        icpSuccess = performICP();
        // if (m_verbose) {
        //     std::cout << "ICP performed in state: " << getStateString() 
        //               << " | Success: " << (icpSuccess ? "YES" : "NO") << std::endl;
        // }
    } else {
        // if (m_verbose && m_alignmentState == AlignmentState::ALIGNED) {
        //     auto timeUntilNext = PERIODIC_ICP_INTERVAL_SECONDS - 
        //         std::chrono::duration_cast<std::chrono::seconds>(
        //             std::chrono::steady_clock::now() - m_lastPeriodicICP).count();
        //     std::cout << "ICP skipped (aligned state) | Next ICP in: " << timeUntilNext << " seconds" << std::endl;
        // }
    }

    // Step 4: Stitch point clouds together (ALWAYS - regardless of ICP success)
    stitchPointClouds();

    // Step 5: Perform ground plane extraction (CONDITIONAL - only after initial alignment)
    bool groundPlanePerformed = false;
    if (shouldPerformGroundPlaneExtraction()) {
        groundPlanePerformed = true;
        performGroundPlaneExtraction();
        if (m_verbose) {
            std::cout << "Ground plane extraction performed" << std::endl;
        }
    } else {
        if (m_verbose) {
            std::cout << "Ground plane extraction skipped (waiting for initial alignment)" << std::endl;
        }
    }

    // Step 6: Perform object detection (CONDITIONAL - only after initial alignment)
    bool objectDetectionPerformed = false;
    if (shouldPerformObjectDetection()) {
        objectDetectionPerformed = true;
        performObjectDetection();
        if (m_verbose) {
            std::cout << "Object detection performed, detected " << m_currentBoxes.size() << " objects" << std::endl;
        }
    } else {
        if (m_verbose && m_objectDetectionEnabled) {
            std::cout << "Object detection skipped (waiting for initial alignment)" << std::endl;
        }
    }


    bool freeSpacePerformed = false;
    if (m_freeSpaceEnabled && shouldPerformObjectDetection()) {
        freeSpacePerformed = true;
        performFreeSpaceDetection();
        if (m_verbose) {
            std::cout << "Free space detection performed" << std::endl;
        }
    } else {
        if (m_verbose && m_freeSpaceEnabled) {
            std::cout << "Free space detection skipped (waiting for initial alignment)" << std::endl;
        }
    }








    // Step 7: Update state management
    updateAlignmentState();

    // Step 8: Log results
    logICPResults();

    // Step 9: Save point clouds if enabled
    if (m_savePointClouds && m_frameNum % 10 == 0) { // Save every 10th frame
        std::stringstream ss;
        ss << "frame_" << std::setfill('0') << std::setw(4) << m_frameNum << "_stitched.ply";
        savePointCloudToPLY(m_stitchedPoints, ss.str());
    }

    // Print comprehensive summary
    // std::cout << "Frame " << m_frameNum << " processed:" << std::endl;
    // std::cout << "  State: " << getStateString() << std::endl;
    // std::cout << "  LiDAR A points: " << m_accumulatedPoints[LIDAR_A_INDEX].size << std::endl;
    // std::cout << "  LiDAR B points: " << m_accumulatedPoints[LIDAR_B_INDEX].size << std::endl;
    // std::cout << "  ICP: " << (icpPerformed ? (icpSuccess ? "PERFORMED & SUCCESS" : "PERFORMED & FAILED") : "SKIPPED") << std::endl;
    // if (icpPerformed) {
    //     std::cout << "  Consecutive Successful: " << m_consecutiveSuccessfulICP << std::endl;
    // }
    // std::cout << "  Stitched points: " << m_stitchedPoints.size << std::endl;
    // std::cout << "  Ground plane: " << (groundPlanePerformed ? 
    //     (m_filteredGroundPlaneValid ? "FILTERED (stable)" : 
    //      (m_groundPlaneValid ? "RAW ONLY (unstable)" : "NOT DETECTED")) : "SKIPPED") << std::endl;
    // std::cout << "  Object detection: " << (objectDetectionPerformed ? 
    //     ("PERFORMED (" + std::to_string(m_currentBoxes.size()) + " objects)") : "SKIPPED") << std::endl;
    // std::cout << "  Total ICP Success Rate: " << (100.0f * m_successfulICPCount / m_frameNum) << "%" << std::endl;
    
    // Show alignment progress if in initial alignment phase
    // if (m_alignmentState == AlignmentState::INITIAL_ALIGNMENT) {
    //             std::cout << "  Alignment Progress: " << m_consecutiveSuccessfulICP << "/" << MIN_SUCCESSFUL_ICP_FOR_ALIGNMENT 
    //                << " consecutive successful ICPs" << std::endl;
    //     if (m_lastICPStats.successful) {
    //         std::cout << "    RMS Cost: " << (m_lastICPStats.rmsCost*1000) << " mm (target: <" 
    //                   << (MAX_RMS_COST_FOR_ALIGNMENT*1000) << " mm)" << std::endl;
    //         std::cout << "    Inlier Fraction: " << (m_lastICPStats.inlierFraction*100) << "% (target: >" 
    //                   << (MIN_INLIER_FRACTION_FOR_ALIGNMENT*100) << "%)" << std::endl;
    //     }
    // }
    
    // PLATFORM-SPECIFIC SYNCHRONIZATION
    #ifdef __aarch64__  // Orin platform
    // Force GPU-CPU memory coherency on Tegra
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    std::cout << "[ORIN-DEBUG] Forced GPU sync at end of frame " << m_frameNum << std::endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::copyToRenderBuffer(uint32_t renderBufferId, uint32_t offset, const dwPointCloud& pointCloud)
{
    // Copy only actual points; copying capacity could leave stale vertices visible
    uint32_t sizeInBytes = pointCloud.size * sizeof(dwVector4f);
    dwVector4f* dataToRender = nullptr;

        // ENSURE PROPER BUFFER SYNCHRONIZATION
    #ifdef __aarch64__  // Orin-specific sync
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    #endif

    dwRenderEngine_mapBuffer(renderBufferId,
                             reinterpret_cast<void**>(&dataToRender),
                             offset,
                             sizeInBytes,
                             DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                             m_renderEngine);

    // ZERO OUT BUFFER FIRST ON ORIN
    #ifdef __aarch64__
    memset(dataToRender, 0, sizeInBytes);
    std::cout << "[ORIN-DEBUG] Cleared point cloud buffer (" << pointCloud.size << " points, " << sizeInBytes << " bytes)" << std::endl;
    #endif

    CHECK_CUDA_ERROR(cudaMemcpy(dataToRender, pointCloud.points,
                                sizeInBytes, cudaMemcpyDeviceToHost));

    dwRenderEngine_unmapBuffer(renderBufferId,
                               DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                               m_renderEngine);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::renderPointCloud(uint32_t renderBufferId,
                                     uint32_t tileId,
                                     uint32_t offset,
                                     dwRenderEngineColorRGBA color,
                                     const dwPointCloud& pointCloud)
{
    // Set tile
    dwRenderEngine_setTile(tileId, m_renderEngine);

    // Model view and projection
    dwMatrix4f modelView;
    Mat4_AxB(modelView.array, getMouseView().getModelView()->array, DW_IDENTITY_TRANSFORMATION3F.array);
    dwRenderEngine_setModelView(&modelView, m_renderEngine);
    dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine);

    // Color and size
    dwRenderEngine_setBackgroundColor({0.1f, 0.1f, 0.1f, 1.0f}, m_renderEngine);
    dwRenderEngine_setColor(color, m_renderEngine);
    dwRenderEngine_setPointSize(1.f, m_renderEngine);

    // Transfer to GL buffer
    copyToRenderBuffer(renderBufferId, offset, pointCloud);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::renderTexts(const char* msg, const dwVector2f& location)
{
    // Store previous tile
    uint32_t previousTile = 0;
    CHECK_DW_ERROR(dwRenderEngine_getTile(&previousTile, m_renderEngine));

    // Select default tile
    CHECK_DW_ERROR(dwRenderEngine_setTile(0, m_renderEngine));

    // Get default tile state
    dwRenderEngineTileState previousDefaultState{};
    CHECK_DW_ERROR(dwRenderEngine_getState(&previousDefaultState, m_renderEngine));

    // Set text render settings
    CHECK_DW_ERROR(dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D({1.0f, 1.0f}, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_setColor({0.0f, 0.6f, 1.0f, 1.0f}, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine));

    // Render
    CHECK_DW_ERROR(dwRenderEngine_renderText2D(msg, location, m_renderEngine));

    // Restore previous settings
    CHECK_DW_ERROR(dwRenderEngine_setState(&previousDefaultState, m_renderEngine));
    CHECK_DW_ERROR(dwRenderEngine_setTile(previousTile, m_renderEngine));
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// void InterLidarICP::onRender()
// {
//     // Clear buffer
//     dwRenderEngine_reset(m_renderEngine);

//     getMouseView().setCenter(0.f, 0.f, 0.f);

//     // Render LiDAR A (top-left, green)
//     renderPointCloud(m_lidarTiles[LIDAR_A_INDEX].renderBufferId,
//                      m_lidarTiles[LIDAR_A_INDEX].tileId,
//                      0,
//                      DW_RENDER_ENGINE_COLOR_GREEN,
//                      m_rigTransformedPoints[LIDAR_A_INDEX]);
//     dwRenderEngine_renderBuffer(m_lidarTiles[LIDAR_A_INDEX].renderBufferId,
//                                 m_rigTransformedPoints[LIDAR_A_INDEX].size,
//                                 m_renderEngine);

//     // Render LiDAR B (top-right, orange)
//     renderPointCloud(m_lidarTiles[LIDAR_B_INDEX].renderBufferId,
//                      m_lidarTiles[LIDAR_B_INDEX].tileId,
//                      0,
//                      DW_RENDER_ENGINE_COLOR_ORANGE,
//                      m_rigTransformedPoints[LIDAR_B_INDEX]);
//     dwRenderEngine_renderBuffer(m_lidarTiles[LIDAR_B_INDEX].renderBufferId,
//                                 m_rigTransformedPoints[LIDAR_B_INDEX].size,
//                                 m_renderEngine);

//     // Render ICP alignment view (bottom-left, both point clouds)
//     // Show LiDAR A in green and aligned LiDAR B in red
//     renderPointCloud(m_icpTile.renderBufferId,
//                      m_icpTile.tileId,
//                      0,
//                      DW_RENDER_ENGINE_COLOR_GREEN,
//                      m_icpAlignedPoints[LIDAR_A_INDEX]);
    
//     // Render aligned LiDAR B points on top
//     copyToRenderBuffer(m_icpTile.renderBufferId, 
//                        m_icpAlignedPoints[LIDAR_A_INDEX].size * sizeof(dwVector4f),
//                        m_icpAlignedPoints[LIDAR_B_INDEX]);
    
//     dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_RED, m_renderEngine);
//     dwRenderEngine_renderBuffer(m_icpTile.renderBufferId,
//                                 m_icpAlignedPoints[LIDAR_A_INDEX].size + m_icpAlignedPoints[LIDAR_B_INDEX].size,
//                                 m_renderEngine);

//     // Render ground plane in ICP view if detected and aligned
//     if (shouldPerformGroundPlaneExtraction() && m_filteredGroundPlaneValid) {
//         renderGroundPlane();
//     }

//     // Render bounding boxes in ICP view if object detection is enabled and aligned
//     if (shouldPerformObjectDetection() && !m_currentBoxes.empty()) {
//         renderBoundingBoxes();
//     }

//     // Stitched view rendering commented out - redundant with post-ICP view
//     // renderPointCloud(m_stitchedTile.renderBufferId,
//     //                  m_stitchedTile.tileId,
//     //                  0,
//     //                  DW_RENDER_ENGINE_COLOR_LIGHTBLUE,
//     //                  m_stitchedPoints);
//     // dwRenderEngine_renderBuffer(m_stitchedTile.renderBufferId,
//     //                             m_stitchedPoints.size,
//     //                             m_renderEngine);

//     // // Render ground plane in stitched view if detected and aligned
//     // if (shouldPerformGroundPlaneExtraction() && m_filteredGroundPlaneValid) {
//     //     dwRenderEngine_setTile(m_stitchedTile.tileId, m_renderEngine);
//     //     renderGroundPlane();
//     // }

//     // // Render bounding boxes in stitched view if object detection is enabled and aligned
//     // if (shouldPerformObjectDetection() && !m_currentBoxes.empty()) {
//     //     dwRenderEngine_setTile(m_stitchedTile.tileId, m_renderEngine);
//     //     renderBoundingBoxes();
//     // }

//     // Add text overlays with state information - render in each individual tile
//     // LiDAR A tile (top-left)
//     dwRenderEngine_setTile(m_lidarTiles[LIDAR_A_INDEX].tileId, m_renderEngine);
//     dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
//     dwRenderEngine_setCoordinateRange2D({1.0f, 1.0f}, m_renderEngine);
//     dwRenderEngine_setColor({0.0f, 0.6f, 1.0f, 1.0f}, m_renderEngine);
//     dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine);
//     dwRenderEngine_renderText2D("LiDAR A (Reference)", {0.1f, 0.9f}, m_renderEngine);
    
//     // LiDAR B tile (top-right)
//     dwRenderEngine_setTile(m_lidarTiles[LIDAR_B_INDEX].tileId, m_renderEngine);
//     dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
//     dwRenderEngine_setCoordinateRange2D({1.0f, 1.0f}, m_renderEngine);
//     dwRenderEngine_setColor({0.0f, 0.6f, 1.0f, 1.0f}, m_renderEngine);
//     dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine);
//     dwRenderEngine_renderText2D("LiDAR B (Source)", {0.1f, 0.9f}, m_renderEngine);
    
//     // ICP alignment tile (bottom-left)
//     dwRenderEngine_setTile(m_icpTile.tileId, m_renderEngine);
//     dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
//     dwRenderEngine_setCoordinateRange2D({1.0f, 1.0f}, m_renderEngine);
//     dwRenderEngine_setColor({0.0f, 0.6f, 1.0f, 1.0f}, m_renderEngine);
//     dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine);
    
//     std::string icpText = "ICP Alignment (";
//     icpText += getStateString();
//     icpText += ")";
//     if (shouldPerformICP()) {
//         icpText += m_lastICPStats.successful ? " SUCCESS" : " FAILED";
//     } else {
//         icpText += " SKIPPED";
//     }
//     if (shouldPerformGroundPlaneExtraction()) {
//         icpText += " + Ground Plane ";
//         if (m_filteredGroundPlaneValid) {
//             icpText += "(FILTERED)";
//         } else if (m_groundPlaneValid) {
//             icpText += "(RAW)";
//         } else {
//             icpText += "(NOT DETECTED)";
//         }
//     } else {
//         icpText += " [Ground Plane: WAITING]";
//     }
//     if (shouldPerformObjectDetection()) {
//         icpText += " + Objects (" + std::to_string(m_currentBoxes.size()) + ")";
//     } else if (m_objectDetectionEnabled) {
//         icpText += " [Objects: WAITING]";
//     }
//     dwRenderEngine_renderText2D(icpText.c_str(), {0.1f, 0.9f}, m_renderEngine);
    
//     // Stitched result tile text commented out - no longer showing stitched view
//     // dwRenderEngine_setTile(m_stitchedTile.tileId, m_renderEngine);
//     // dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
//     // dwRenderEngine_setCoordinateRange2D({1.0f, 1.0f}, m_renderEngine);
//     // dwRenderEngine_setColor({0.0f, 0.6f, 1.0f, 1.0f}, m_renderEngine);
//     // dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine);
    
//     // std::string stitchedText = "Stitched Result (" + std::to_string(m_stitchedPoints.size) + " points)";
//     // if (shouldPerformGroundPlaneExtraction()) {
//     //     if (m_filteredGroundPlaneValid) {
//     //         stitchedText += " + Filtered Ground Plane";
//     //     } else if (m_groundPlaneValid) {
//     //         stitchedText += " + Raw Ground Plane";
//     //     }
//     // }
//     // if (shouldPerformObjectDetection()) {
//     //     stitchedText += " + " + std::to_string(m_currentBoxes.size()) + " Objects";
//     //     if (!m_currentBoxes.empty()) {
//     //         uint32_t vehicles = 0, pedestrians = 0, cyclists = 0;
//     //         for (const auto& box : m_currentBoxes) {
//     //             switch (box.classId) {
//     //                 case 0: vehicles++; break;
//     //                 case 1: pedestrians++; break;
//     //                 case 2: cyclists++; break;
//     //             }
//     //         }
//     //         stitchedText += " (V:" + std::to_string(vehicles) + 
//     //                        " P:" + std::to_string(pedestrians) + 
//     //                        " C:" + std::to_string(cyclists) + ")";
//     //     }
//     // }
//     // dwRenderEngine_renderText2D(stitchedText.c_str(), {0.1f, 0.9f}, m_renderEngine);

//     // FPS and comprehensive statistics
//     renderutils::renderFPS(m_renderEngine, getCurrentFPS());
    
//     std::string statsText = "Frame: " + std::to_string(m_frameNum) + 
//                            " | State: " + getStateString() +
//                            " | ICP Success Rate: " + std::to_string(int(100.0f * m_successfulICPCount / std::max(1u, m_frameNum))) + "%";
    
//     if (m_alignmentState == AlignmentState::INITIAL_ALIGNMENT) {
//         statsText += " | Progress: " + std::to_string(m_consecutiveSuccessfulICP) + "/" + std::to_string(MIN_SUCCESSFUL_ICP_FOR_ALIGNMENT);
//     } else if (m_alignmentState == AlignmentState::ALIGNED) {
//         auto timeUntilNext = PERIODIC_ICP_INTERVAL_SECONDS - 
//             std::chrono::duration_cast<std::chrono::seconds>(
//                 std::chrono::steady_clock::now() - m_lastPeriodicICP).count();
//         statsText += " | Next ICP: " + std::to_string(timeUntilNext) + "s";
//     }
    
//     if (m_lidarReadyAnnounced) {
//         statsText += " | *** LIDAR READY ***";
//     }
    
//     renderTexts(statsText.c_str(), {0.5f, 0.05f});
// }

void InterLidarICP::onRender()
{
    
    // Clear buffer
    dwRenderEngine_reset(m_renderEngine);

    getMouseView().setCenter(0.f, 0.f, 0.f);

        // DIAGNOSTIC: Add buffer state logging
        #ifdef __aarch64__
        if (m_verbose) {
            std::cout << "[DIAGNOSTIC] === RENDER FRAME START ===" << std::endl;
            std::cout << "Ground plane buffer ID: " << m_groundPlaneRenderBufferId << std::endl;
            std::cout << "ICP tile buffer ID: " << m_icpTile.renderBufferId << std::endl;
            std::cout << "LiDAR A buffer ID: " << m_lidarTiles[LIDAR_A_INDEX].renderBufferId << std::endl;
            std::cout << "LiDAR B buffer ID: " << m_lidarTiles[LIDAR_B_INDEX].renderBufferId << std::endl;
            std::cout << "Box line buffer ID: " << m_boxLineBuffer << std::endl;
        }
        #endif
    

    // COMMENTED OUT: Individual LiDAR point cloud rendering
    // Render LiDAR A (top-left, green)
    // renderPointCloud(m_lidarTiles[LIDAR_A_INDEX].renderBufferId,
    //                  m_lidarTiles[LIDAR_A_INDEX].tileId,
    //                  0,
    //                  DW_RENDER_ENGINE_COLOR_GREEN,
    //                  m_rigTransformedPoints[LIDAR_A_INDEX]);
    // dwRenderEngine_renderBuffer(m_lidarTiles[LIDAR_A_INDEX].renderBufferId,
    //                             m_rigTransformedPoints[LIDAR_A_INDEX].size,
    //                             m_renderEngine);

    // COMMENTED OUT: Individual LiDAR point cloud rendering
    // Render LiDAR B (top-right, orange)
    // renderPointCloud(m_lidarTiles[LIDAR_B_INDEX].renderBufferId,
    //                  m_lidarTiles[LIDAR_B_INDEX].tileId,
    //                  0,
    //                  DW_RENDER_ENGINE_COLOR_ORANGE,
    //                  m_rigTransformedPoints[LIDAR_B_INDEX]);
    // dwRenderEngine_renderBuffer(m_lidarTiles[LIDAR_B_INDEX].renderBufferId,
    //                             m_rigTransformedPoints[LIDAR_B_INDEX].size,
    //                             m_renderEngine);

    // Render ICP alignment view (bottom-left, both point clouds)
    // Show LiDAR A in green and aligned LiDAR B in red
    renderPointCloud(m_icpTile.renderBufferId,
                     m_icpTile.tileId,
                     0,
                     DW_RENDER_ENGINE_COLOR_GREEN,
                     m_icpAlignedPoints[LIDAR_A_INDEX]);
    
    // Render aligned LiDAR B points on top
    copyToRenderBuffer(m_icpTile.renderBufferId, 
                       m_icpAlignedPoints[LIDAR_A_INDEX].size * sizeof(dwVector4f),
                       m_icpAlignedPoints[LIDAR_B_INDEX]);
    
    dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_RED, m_renderEngine);
    dwRenderEngine_renderBuffer(m_icpTile.renderBufferId,
                                m_icpAlignedPoints[LIDAR_A_INDEX].size + m_icpAlignedPoints[LIDAR_B_INDEX].size,
                                m_renderEngine);

    // 🔧 FIXED: Properly set tile BEFORE rendering ground plane
    if (shouldPerformGroundPlaneExtraction() && m_filteredGroundPlaneValid) {
        // Set the ICP tile for ground plane rendering
        dwRenderEngine_setTile(m_icpTile.tileId, m_renderEngine);
        
        // Set proper 3D rendering state for ground plane
        dwMatrix4f modelView;
        Mat4_AxB(modelView.array, getMouseView().getModelView()->array, DW_IDENTITY_TRANSFORMATION3F.array);
        dwRenderEngine_setModelView(&modelView, m_renderEngine);
        dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine);
        
        // Render ground plane in ICP tile
        renderGroundPlane();
        
        if (m_verbose) {
            std::cout << "[RENDER-DEBUG] Ground plane rendered in ICP tile" << std::endl;
        }
    }

    // 🔧 FIXED: Properly set tile BEFORE rendering bounding boxes
    if (shouldPerformObjectDetection() && !m_currentBoxes.empty()) {
        // Set the ICP tile for bounding box rendering
        dwRenderEngine_setTile(m_icpTile.tileId, m_renderEngine);
        
        // Render bounding boxes in ICP tile
        renderBoundingBoxes();
        
        if (m_verbose) {
            std::cout << "[RENDER-DEBUG] Bounding boxes rendered in ICP tile" << std::endl;
        }
    }



    if (m_freeSpaceEnabled && shouldPerformObjectDetection()) {
        dwRenderEngine_setTile(m_icpTile.tileId, m_renderEngine);
        
        // Set proper 3D rendering state
        dwMatrix4f modelView;
        Mat4_AxB(modelView.array, getMouseView().getModelView()->array, DW_IDENTITY_TRANSFORMATION3F.array);
        dwRenderEngine_setModelView(&modelView, m_renderEngine);
        dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine);
        
        renderFreeSpace();
        
        if (m_verbose) {
            std::cout << "[RENDER-DEBUG] Free space rendered in ICP tile" << std::endl;
        }
    }

    // Render BEV feature map visualization
    if (m_bevVisualizationEnabled && !m_bevFeatureMap.empty()) {
        renderBEVFeatureMap();
    }

    // Render heatmap visualization
    if (m_heatmapVisualizationEnabled && !m_heatmapData.empty()) {
        renderHeatmap();
    }






    // COMMENTED OUT: Individual tile text overlays
    // Add text overlays with state information - render in each individual tile
    // LiDAR A tile (top-left)
    // dwRenderEngine_setTile(m_lidarTiles[LIDAR_A_INDEX].tileId, m_renderEngine);
    // dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
    // dwRenderEngine_setCoordinateRange2D({1.0f, 1.0f}, m_renderEngine);
    // dwRenderEngine_setColor({0.0f, 0.6f, 1.0f, 1.0f}, m_renderEngine);
    // dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine);
    // dwRenderEngine_renderText2D("LiDAR A (Reference)", {0.1f, 0.9f}, m_renderEngine);
    
    // COMMENTED OUT: Individual tile text overlays
    // LiDAR B tile (top-right)
    // dwRenderEngine_setTile(m_lidarTiles[LIDAR_B_INDEX].tileId, m_renderEngine);
    // dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
    // dwRenderEngine_setCoordinateRange2D({1.0f, 1.0f}, m_renderEngine);
    // dwRenderEngine_setColor({0.0f, 0.6f, 1.0f, 1.0f}, m_renderEngine);
    // dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine);
    // dwRenderEngine_renderText2D("LiDAR B (Source)", {0.1f, 0.9f}, m_renderEngine);
    
    // ICP alignment tile (bottom-left)
    // dwRenderEngine_setTile(m_icpTile.tileId, m_renderEngine);
    // dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
    // dwRenderEngine_setCoordinateRange2D({1.0f, 1.0f}, m_renderEngine);
    // dwRenderEngine_setColor({0.0f, 0.6f, 1.0f, 1.0f}, m_renderEngine);
    // dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine);
    
    // std::string icpText = "ICP Alignment (";
    // icpText += getStateString();
    // icpText += ")";
    // if (shouldPerformICP()) {
    //     icpText += m_lastICPStats.successful ? " SUCCESS" : " FAILED";
    // } else {
    //     icpText += " SKIPPED";
    // }
    // if (shouldPerformGroundPlaneExtraction()) {
    //     icpText += " + Ground Plane ";
    //     if (m_filteredGroundPlaneValid) {
    //         icpText += "(FILTERED)";
    //     } else if (m_groundPlaneValid) {
    //         icpText += "(RAW)";
    //     } else {
    //         icpText += "(NOT DETECTED)";
    //     }
    // } else {
    //     icpText += " [Ground Plane: WAITING]";
    // }
    // if (shouldPerformObjectDetection()) {
    //     icpText += " + Objects (" + std::to_string(m_currentBoxes.size()) + ")";
    // } else if (m_objectDetectionEnabled) {
    //     icpText += " [Objects: WAITING]";
    // }
    // dwRenderEngine_renderText2D(icpText.c_str(), {0.1f, 0.9f}, m_renderEngine);

    // FPS and comprehensive statistics
    // renderutils::renderFPS(m_renderEngine, getCurrentFPS());
    
    // std::string statsText = "Frame: " + std::to_string(m_frameNum) + 
    //                        " | State: " + getStateString() +
    //                        " | ICP Success Rate: " + std::to_string(int(100.0f * m_successfulICPCount / std::max(1u, m_frameNum))) + "%";
    
    // if (m_alignmentState == AlignmentState::INITIAL_ALIGNMENT) {
    //     statsText += " | Progress: " + std::to_string(m_consecutiveSuccessfulICP) + "/" + std::to_string(MIN_SUCCESSFUL_ICP_FOR_ALIGNMENT);
    // } else if (m_alignmentState == AlignmentState::ALIGNED) {
    //     auto timeUntilNext = PERIODIC_ICP_INTERVAL_SECONDS - 
    //         std::chrono::duration_cast<std::chrono::seconds>(
    //             std::chrono::steady_clock::now() - m_lastPeriodicICP).count();
    //     statsText += " | Next ICP: " + std::to_string(timeUntilNext) + "s";
    // }
    
    // if (m_lidarReadyAnnounced) {
    //     statsText += " | *** LIDAR READY ***";
    // }
    
    // renderTexts(statsText.c_str(), {0.5f, 0.05f});
    
    // 🔧 CRITICAL: Reset rendering state at end of frame
    #ifdef __aarch64__  // Orin-specific cleanup
    dwRenderEngine_setTile(0, m_renderEngine);  // Reset to default tile
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());   // Final GPU sync
    if (m_verbose) {
        std::cout << "[ORIN-DEBUG] Rendering state reset at end of frame" << std::endl;
    }
    #endif
}


void InterLidarICP::inspectRenderBuffers()
{
    #ifdef __aarch64__
    std::cout << "\n=== RENDER BUFFER INSPECTION ===" << std::endl;
    std::cout << "Ground plane buffer ID: " << m_groundPlaneRenderBufferId << std::endl;
    std::cout << "ICP tile buffer ID: " << m_icpTile.renderBufferId << std::endl;
    std::cout << "LiDAR A buffer ID: " << m_lidarTiles[LIDAR_A_INDEX].renderBufferId << std::endl;
    std::cout << "LiDAR B buffer ID: " << m_lidarTiles[LIDAR_B_INDEX].renderBufferId << std::endl;
    std::cout << "Box line buffer ID: " << m_boxLineBuffer << std::endl;
    
    // Check for buffer ID conflicts
    std::vector<uint32_t> bufferIds = {
        m_groundPlaneRenderBufferId,
        m_icpTile.renderBufferId,
        m_lidarTiles[LIDAR_A_INDEX].renderBufferId,
        m_lidarTiles[LIDAR_B_INDEX].renderBufferId,
        m_boxLineBuffer
    };
    
    std::sort(bufferIds.begin(), bufferIds.end());
    for (size_t i = 1; i < bufferIds.size(); i++) {
        if (bufferIds[i] == bufferIds[i-1]) {
            std::cout << "*** BUFFER ID CONFLICT DETECTED: " << bufferIds[i] << " ***" << std::endl;
        }
    }
    std::cout << "=================================" << std::endl;
    #endif
}


/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::onResizeWindow(int width, int height)
{
    dwRenderEngine_reset(m_renderEngine);
    dwRectf rect;
    rect.width = width;
    rect.height = height;
    rect.x = 0;
    rect.y = 0;
    dwRenderEngine_setBounds(rect, m_renderEngine);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::onKeyDown(int32_t key, int32_t scancode, int32_t mods)
{
    (void)scancode;
    (void)mods;

    if (key == GLFW_KEY_SPACE) {
        m_paused = !m_paused;
        std::cout << (m_paused ? "Paused" : "Resumed") << " | Current State: " << getStateString() << std::endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::onRelease()
{
    // Stop sensors
    if (m_sensorManager) {
        dwSensorManager_stop(m_sensorManager);
        dwSensorManager_release(m_sensorManager);
    }

    // Release rig config
    if (m_rigConfig) {
        dwRig_release(m_rigConfig);
    }

    // Release point cloud buffers
    for (uint32_t i = 0; i < m_lidarCount; i++) {
        dwPointCloud_destroyBuffer(&m_accumulatedPoints[i]);
        dwPointCloud_destroyBuffer(&m_rigTransformedPoints[i]);
        dwPointCloud_destroyBuffer(&m_icpAlignedPoints[i]);
    }
    dwPointCloud_destroyBuffer(&m_stitchedPoints);
    dwPointCloud_destroyBuffer(&m_stitchedPointsHost);

    // Release accumulators
    for (uint32_t i = 0; i < m_lidarCount; i++) {
        if (m_accumulator[i]) {
            dwPointCloudAccumulator_release(m_accumulator[i]);
        }
    }

    // Release ICP and stitchers
    if (m_icp) {
        dwPointCloudICP_release(m_icp);
    }
    if (m_stitcher) {
        dwPointCloudStitcher_release(m_stitcher);
    }
    if (m_icpTransformer) {
        dwPointCloudStitcher_release(m_icpTransformer);
    }
    for (uint32_t i = 0; i < m_lidarCount; i++) {
        if (m_coordinateConverter[i]) {
            dwPointCloudStitcher_release(m_coordinateConverter[i]);
        }
    }

    // Release ground plane extractor
    if (m_planeExtractor) {
        dwPCPlaneExtractor_release(m_planeExtractor);
    }

    // Release object detection resources
    if (m_objectDetectionEnabled) {
        if (m_deviceInputBuffer) {
            cudaFree(m_deviceInputBuffer);
            m_deviceInputBuffer = nullptr;
        }
        if (m_deviceOutputBuffer) {
            cudaFree(m_deviceOutputBuffer);
            m_deviceOutputBuffer = nullptr;
        }
        if (m_deviceNumPointsBuffer) {
            cudaFree(m_deviceNumPointsBuffer);
            m_deviceNumPointsBuffer = nullptr;
        }
        if (m_deviceOutputCountBuffer) {
            cudaFree(m_deviceOutputCountBuffer);
            m_deviceOutputCountBuffer = nullptr;
        }
        
        // TensorRT resources are released automatically by smart pointers
        m_executionContext.reset();
        m_engine.reset();
        m_runtime.reset();
    }

    // COMMENTED OUT: Individual LiDAR render buffer destruction - only showing ICP alignment view
    // Release render buffers
    // for (uint32_t i = 0; i < m_lidarCount; i++) {
    //     dwRenderEngine_destroyBuffer(m_lidarTiles[i].renderBufferId, m_renderEngine);
    // }
    dwRenderEngine_destroyBuffer(m_icpTile.renderBufferId, m_renderEngine);
    // Stitched view buffer destruction commented out - buffer was not created
    // dwRenderEngine_destroyBuffer(m_stitchedTile.renderBufferId, m_renderEngine);
    dwRenderEngine_destroyBuffer(m_groundPlaneRenderBufferId, m_renderEngine);
    
    if (m_objectDetectionEnabled && m_boxLineBuffer != 0) {
        dwRenderEngine_destroyBuffer(m_boxLineBuffer, m_renderEngine);
    }

    if (m_freeSpaceEnabled && m_freeSpaceRenderBufferId != 0) {
        dwRenderEngine_destroyBuffer(m_freeSpaceRenderBufferId, m_renderEngine);
    }

    // Release render engine
    if (m_renderEngine) {
        dwRenderEngine_release(m_renderEngine);
    }

    // Release DriveWorks and SAL
    if (m_sal) {
        dwSAL_release(m_sal);
    }
    if (m_viz) {
        dwVisualizationRelease(m_viz);
    }
    if (m_context) {
        dwRelease(m_context);
    }
    
    dwLogger_release();

    // Close log file
    if (m_logFile.is_open()) {
        m_logFile.close();
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Inter-Lidar ICP Statistics with State Management and Object Detection:" << std::endl;
    std::cout << "  Total Frames: " << m_frameNum << std::endl;
    std::cout << "  Successful ICP: " << m_successfulICPCount << std::endl;
    std::cout << "  Failed ICP: " << m_failedICPCount << std::endl;
    std::cout << "  Success Rate: " << (100.0f * m_successfulICPCount / std::max(1u, m_frameNum)) << "%" << std::endl;
    std::cout << "  Final State: " << getStateString() << std::endl;
    std::cout << "  Lidar Ready Announced: " << (m_lidarReadyAnnounced ? "YES" : "NO") << std::endl;
    
    if (m_groundPlaneEnabled) {
        if (m_lidarReadyAnnounced) {
            std::cout << "  Ground Plane Extraction: ENABLED (after alignment)" << std::endl;
            std::cout << "  Ground Plane Visualization: " << (m_groundPlaneVisualizationEnabled ? "ENABLED" : "DISABLED") << std::endl;
        } else {
            std::cout << "  Ground Plane Extraction: ENABLED (waiting for alignment)" << std::endl;
            std::cout << "  Ground Plane Visualization: " << (m_groundPlaneVisualizationEnabled ? "ENABLED" : "DISABLED") << std::endl;
        }
    } else {
        std::cout << "  Ground Plane Extraction: DISABLED (user disabled)" << std::endl;
        std::cout << "  Ground Plane Visualization: DISABLED (extraction disabled)" << std::endl;
    }

    // Free space statistics
    if (m_freeSpaceEnabled) {
        std::cout << "  Free Space Detection: ENABLED" << std::endl;
        std::cout << "  Free Space Visualization: " << (m_freeSpaceVisualizationEnabled ? "ENABLED" : "DISABLED") << std::endl;
    } else {
        std::cout << "  Free Space Detection: DISABLED" << std::endl;
    }
    
    // Object detection statistics
    if (m_objectDetectionEnabled) {
        std::cout << "  Object Detection: ENABLED" << std::endl;
        std::cout << "    Total Detections: " << m_totalDetections << std::endl;
        std::cout << "    Vehicle Detections: " << m_vehicleDetections << std::endl;
        std::cout << "    Pedestrian Detections: " << m_pedestrianDetections << std::endl;
        std::cout << "    Cyclist Detections: " << m_cyclistDetections << std::endl;
        if (m_frameNum > 0) {
            std::cout << "    Average Detections per Frame: " << (float)m_totalDetections / m_frameNum << std::endl;
        }
    } else {
        std::cout << "  Object Detection: DISABLED" << std::endl;
    }
    
    std::cout << std::string(60, '=') << std::endl;
}