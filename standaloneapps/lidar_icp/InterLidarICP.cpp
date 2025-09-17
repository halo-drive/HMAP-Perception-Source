#include "InterLidarICP.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <thread>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>

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
    m_maxIters = static_cast<uint32_t>(atoi(getArgument("maxIters").c_str()));
    m_numFrames = static_cast<uint32_t>(atoi(getArgument("numFrames").c_str()));
    m_skipFrames = static_cast<uint32_t>(atoi(getArgument("skipFrames").c_str()));
    m_verbose = getArgument("verbose") == "true";
    m_savePointClouds = getArgument("savePointClouds") == "true";
    m_stitchOnly = getArgument("stitchOnly") == "true";

    if (m_maxIters > 50) {
        std::cerr << "`--maxIters` too large, set to " << (m_maxIters = 50) << std::endl;
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
    
    // Initialize state management
    m_alignmentState = AlignmentState::INITIALIZING;
    m_consecutiveSuccessfulICP = 0;
    m_lidarReadyAnnounced = false;
    m_lastPeriodicICP = std::chrono::steady_clock::now();
    
    std::cout << "Inter-Lidar ICP initialized with STATE MANAGEMENT:" << std::endl;
    std::cout << "  Max ICP Iterations: " << m_maxIters << std::endl;
    std::cout << "  Verbose Logging: " << (m_verbose ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "  Save Point Clouds: " << (m_savePointClouds ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "  Ground Plane Extraction: " << (m_stitchOnly ? "DISABLED (stitchOnly)" : "ENABLED (CUDA Pipeline)") << std::endl;
    std::cout << "  State Management: ENABLED" << std::endl;
    std::cout << "    - Initial Alignment: Continuous ICP until aligned" << std::endl;
    std::cout << "    - Periodic ICP: Every " << PERIODIC_ICP_INTERVAL_SECONDS << " seconds after alignment" << std::endl;
    std::cout << "    - Ground Plane: Only after initial alignment complete" << std::endl;
    if (m_stitchOnly) {
        std::cout << "  MODE: STITCH-ONLY (ICP disabled, ground plane disabled)" << std::endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
InterLidarICP::~InterLidarICP()
{
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
                std::cout << "STATE CHANGE: INITIALIZING -> INITIAL_ALIGNMENT" << std::endl;
                std::cout << "Starting continuous ICP until complete alignment..." << std::endl;
            }
            break;
            
        case AlignmentState::INITIAL_ALIGNMENT:
            // Check if initial alignment is complete
            if (isInitialAlignmentComplete()) {
                m_alignmentState = AlignmentState::ALIGNED;
                m_lastPeriodicICP = std::chrono::steady_clock::now();
                announceLibarReady();
                std::cout << "STATE CHANGE: INITIAL_ALIGNMENT -> ALIGNED" << std::endl;
                std::cout << "Lidars successfully aligned! Switching to periodic ICP mode." << std::endl;
                std::cout << "Ground plane extraction will now start continuously." << std::endl;
            }
            break;
            
        case AlignmentState::ALIGNED:
            // Check if periodic ICP is due
            if (isPeriodicICPDue()) {
                m_alignmentState = AlignmentState::REALIGNMENT;
                std::cout << "STATE CHANGE: ALIGNED -> REALIGNMENT" << std::endl;
                std::cout << "Performing periodic ICP re-alignment..." << std::endl;
            }
            break;
            
        case AlignmentState::REALIGNMENT:
            // Return to aligned state after performing ICP
            m_alignmentState = AlignmentState::ALIGNED;
            m_lastPeriodicICP = std::chrono::steady_clock::now();
            if (m_verbose) {
                std::cout << "STATE CHANGE: REALIGNMENT -> ALIGNED" << std::endl;
                std::cout << "Periodic ICP complete, returning to aligned state." << std::endl;
            }
            break;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
bool InterLidarICP::shouldPerformICP() const
{
    if (m_stitchOnly) return false;
    return (m_alignmentState == AlignmentState::INITIAL_ALIGNMENT || 
            m_alignmentState == AlignmentState::REALIGNMENT);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
bool InterLidarICP::shouldPerformGroundPlaneExtraction() const
{
    if (m_stitchOnly) return false;
    return (m_alignmentState == AlignmentState::ALIGNED || 
            m_alignmentState == AlignmentState::REALIGNMENT);
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
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "           *** LIDAR READY ***" << std::endl;
        std::cout << "   Initial alignment complete!" << std::endl;
        std::cout << "   - ICP Success Rate: " << (100.0f * m_successfulICPCount / m_frameNum) << "%" << std::endl;
        std::cout << "   - Final RMS Cost: " << (m_lastICPStats.rmsCost*1000) << " mm" << std::endl;
        std::cout << "   - Inlier Fraction: " << (m_lastICPStats.inlierFraction*100) << "%" << std::endl;
        std::cout << "   - Consecutive Successful ICPs: " << m_consecutiveSuccessfulICP << std::endl;
        std::cout << "   Now performing periodic ICP every " << PERIODIC_ICP_INTERVAL_SECONDS << " seconds" << std::endl;
        std::cout << "   Ground plane extraction enabled" << std::endl;
        std::cout << std::string(50, '=') << "\n" << std::endl;
        
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
        // COMMENTED OUT: Do not apply sensor-to-rig at accumulation stage to avoid double-transform
        // CHECK_DW_ERROR(dwRig_getSensorToRigTransformation(&params.sensorTransformation, lidarSensorIndex, m_rigConfig));
        // Ensure accumulator uses identity transform; rig transform will be applied in applyRigTransformations()
        params.sensorTransformation = DW_IDENTITY_TRANSFORMATION3F;
        
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
    
    // *** TIGHTENED CONVERGENCE CRITERIA FOR BETTER ALIGNMENT ***
    params.distanceConvergenceTol = 0.002f;  // 2mm (was 1cm) - much tighter!
    params.angleConvergenceTol = 0.01f;      // ~0.6° (was ~3°) - much tighter!
    
    CHECK_DW_ERROR(dwPointCloudICP_initialize(&m_icp, &params, m_context));
    CHECK_DW_ERROR(dwPointCloudICP_setCUDAStream(m_stream, m_icp));
    
    std::cout << "ICP initialized with TIGHTENED parameters:" << std::endl;
    std::cout << "  Type: DEPTH_MAP" << std::endl;
    std::cout << "  Max iterations: " << params.maxIterations << std::endl;
    std::cout << "  Depthmap size: " << params.depthmapSize.x << "x" << params.depthmapSize.y << std::endl;
    std::cout << "  Distance tolerance: " << params.distanceConvergenceTol*1000 << " mm (TIGHTENED)" << std::endl;
    std::cout << "  Angle tolerance: " << params.angleConvergenceTol << " rad (~" << (params.angleConvergenceTol*57.3) << "°)" << std::endl;
    std::cout << "  Max points: " << params.maxPoints << std::endl;
    std::cout << "  Expected VLP16 points: " << m_lidarProps[0].pointsPerSpin << std::endl;
    std::cout << "  Target: Sub-centimeter alignment accuracy" << std::endl;
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
    params.minInlierFraction = 0.35f;  // Tighter fit; reject non-ground clusters
    params.ransacIterationCount = 100;  // Maximum allowed by DriveWorks
    params.optimizerIterationCount = 10;  // Non-linear optimization iterations
    
    // Box filter parameters - focus on the region around the ground plane in rig coordinates
    // With correct rig mounting heights (sensors at z>0), the ground is near z ≈ 0
    params.boxFilterParams.maxPointCount = 20000;  // Keep enough points for robust fitting
    params.boxFilterParams.box.center = {0.0f, 0.0f, 0.0f};  // Center at ground level in rig frame
    params.boxFilterParams.box.halfAxisXYZ = {8.0f, 8.0f, 0.8f};  // 16m x 16m x 1.6m vertical span
    
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
    std::cout << "  Box filter center: [" << params.boxFilterParams.box.center.x << ", " 
              << params.boxFilterParams.box.center.y << ", " << params.boxFilterParams.box.center.z << "] m" << std::endl;
    std::cout << "  Box filter size: " << params.boxFilterParams.box.halfAxisXYZ.x*2 << "x" 
              << params.boxFilterParams.box.halfAxisXYZ.y*2 << "x" 
              << params.boxFilterParams.box.halfAxisXYZ.z*2 << " m (focused on ground)" << std::endl;
    std::cout << "  Expected ground level: ~0.0m (rig origin at ground)" << std::endl;
    std::cout << "  NOTE: Ground plane extraction will only start after initial ICP alignment is complete" << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::initGroundPlaneRenderBuffer()
{
    // Create render buffer for ground plane visualization
    // We'll create a grid of triangles to represent the ground plane
    uint32_t gridVertices = GROUND_PLANE_GRID_SIZE * GROUND_PLANE_GRID_SIZE * 6; // 2 triangles per cell, 3 vertices per triangle
    
    CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_groundPlaneRenderBufferId,
                                               DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
                                               sizeof(dwVector3f),
                                               0,
                                               gridVertices,
                                               m_renderEngine));
    
    std::cout << "Ground plane render buffer initialized with " << gridVertices << " vertices" << std::endl;
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

    // Create tiles: 2x2 layout
    // Top-left: LiDAR A
    tileParam.layout.viewport = {0.f, 0.f, 0.5f, 0.5f};
    tileParam.layout.positionType = DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_LEFT;
    CHECK_DW_ERROR(dwRenderEngine_addTile(&m_lidarTiles[LIDAR_A_INDEX].tileId, &tileParam, m_renderEngine));

    // Top-right: LiDAR B
    tileParam.layout.viewport = {0.5f, 0.f, 0.5f, 0.5f};
    CHECK_DW_ERROR(dwRenderEngine_addTile(&m_lidarTiles[LIDAR_B_INDEX].tileId, &tileParam, m_renderEngine));

    // Bottom-left: ICP alignment view
    tileParam.layout.viewport = {0.f, 0.5f, 0.5f, 0.5f};
    CHECK_DW_ERROR(dwRenderEngine_addTile(&m_icpTile.tileId, &tileParam, m_renderEngine));

    // Bottom-right: Stitched view
    tileParam.layout.viewport = {0.5f, 0.5f, 0.5f, 0.5f};
    CHECK_DW_ERROR(dwRenderEngine_addTile(&m_stitchedTile.tileId, &tileParam, m_renderEngine));

    // Create render buffers
    for (uint32_t i = 0; i < m_lidarCount; i++) {
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_lidarTiles[i].renderBufferId,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                   sizeof(dwVector4f),
                                                   0,
                                                   m_lidarProps[i].pointsPerSpin,
                                                   m_renderEngine));
    }

    CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_icpTile.renderBufferId,
                                               DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                               sizeof(dwVector4f),
                                               0,
                                               m_stitchedPoints.capacity,
                                               m_renderEngine));

    CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_stitchedTile.renderBufferId,
                                               DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                               sizeof(dwVector4f),
                                               0,
                                               m_stitchedPoints.capacity,
                                               m_renderEngine));

    // Initialize ground plane render buffer
    initGroundPlaneRenderBuffer();

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
    // Bind input point cloud (use device memory for CUDA pipeline)
    CHECK_DW_ERROR(dwPCPlaneExtractor_bindInput(&m_stitchedPoints, m_planeExtractor));
    
    // Bind output - only ground plane (no inliers/outliers in CUDA pipeline)
    CHECK_DW_ERROR(dwPCPlaneExtractor_bindOutput(nullptr, nullptr, &m_groundPlane, m_planeExtractor));
    
    // Process ground plane extraction
    dwStatus status = dwPCPlaneExtractor_process(m_planeExtractor);
    
    if (status == DW_SUCCESS) {
        m_groundPlaneValid = m_groundPlane.valid;
        
        if (m_groundPlaneValid) {
            // Apply temporal filtering for stability
            if (!m_filteredGroundPlaneValid) {
                // First valid detection - initialize filtered plane
                m_filteredGroundPlane = m_groundPlane;
                m_filteredGroundPlaneValid = true;
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
            }
            
            if (m_verbose) {
                std::cout << "Ground plane extracted successfully:" << std::endl;
                std::cout << "  Raw Normal: [" << m_groundPlane.normal.x << ", " 
                          << m_groundPlane.normal.y << ", " << m_groundPlane.normal.z << "]" << std::endl;
                std::cout << "  Raw Offset: " << m_groundPlane.offset << std::endl;
                std::cout << "  Filtered Normal: [" << m_filteredGroundPlane.normal.x << ", " 
                          << m_filteredGroundPlane.normal.y << ", " << m_filteredGroundPlane.normal.z << "]" << std::endl;
                std::cout << "  Filtered Offset: " << m_filteredGroundPlane.offset << std::endl;
                // Height of ground relative to the current frame follows n·x + d = 0.
                // For n.z > 0 and rig origin at ground, print the signed distance d clearly.
                std::cout << "  Filtered Distance d: " << m_filteredGroundPlane.offset << " m (plane n·x + d = 0)" << std::endl;
            }
        }
    } else {
        m_groundPlaneValid = false;
        if (m_verbose) {
            std::cout << "Ground plane extraction failed: " << dwGetStatusName(status) << std::endl;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::renderGroundPlane()
{
    // Use filtered ground plane for more stable visualization
    if (!m_filteredGroundPlaneValid || !m_filteredGroundPlane.valid) {
        return;
    }
    
    // Map render buffer for ground plane
    dwVector3f* vertices = nullptr;
    uint32_t maxVerts = 0;
    uint32_t stride = 0;
    
    CHECK_DW_ERROR(dwRenderEngine_mapBuffer(m_groundPlaneRenderBufferId,
                                            reinterpret_cast<void**>(&vertices),
                                            0,
                                            GROUND_PLANE_GRID_SIZE * GROUND_PLANE_GRID_SIZE * 6 * sizeof(dwVector3f),
                                            DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
                                            m_renderEngine));
    
    // Generate ground plane mesh based on filtered plane
    uint32_t vertexIndex = 0;
    float32_t halfSize = GROUND_PLANE_GRID_SIZE * GROUND_PLANE_CELL_SIZE * 0.5f;
    
    // Get plane normal and offset from the filtered ground plane
    dwVector3f normal = m_filteredGroundPlane.normal;
    float32_t offset = m_filteredGroundPlane.offset;
    
    // Normalize the normal vector if needed
    float32_t normalMagnitude = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (normalMagnitude > 0.001f) {
        normal.x /= normalMagnitude;
        normal.y /= normalMagnitude;
        normal.z /= normalMagnitude;
    }
    
    // Check if plane is too vertical (normal.z close to 0)
    if (fabs(normal.z) < 0.01f) {
        if (m_verbose) {
            std::cout << "Ground plane too vertical, skipping visualization" << std::endl;
        }
        CHECK_DW_ERROR(dwRenderEngine_unmapBuffer(m_groundPlaneRenderBufferId,
                                                  DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
                                                  m_renderEngine));
        return;
    }
    
    // Generate grid vertices
    for (uint32_t i = 0; i < GROUND_PLANE_GRID_SIZE - 1; i++) {
        for (uint32_t j = 0; j < GROUND_PLANE_GRID_SIZE - 1; j++) {
            float32_t x1 = -halfSize + i * GROUND_PLANE_CELL_SIZE;
            float32_t y1 = -halfSize + j * GROUND_PLANE_CELL_SIZE;
            float32_t x2 = x1 + GROUND_PLANE_CELL_SIZE;
            float32_t y2 = y1 + GROUND_PLANE_CELL_SIZE;
            
            // Calculate Z coordinates based on plane equation: normal.x*x + normal.y*y + normal.z*z + offset = 0
            // Solving for z: z = -(normal.x*x + normal.y*y + offset) / normal.z
            float32_t z1 = -(normal.x * x1 + normal.y * y1 + offset) / normal.z;
            float32_t z2 = -(normal.x * x2 + normal.y * y1 + offset) / normal.z;
            float32_t z3 = -(normal.x * x1 + normal.y * y2 + offset) / normal.z;
            float32_t z4 = -(normal.x * x2 + normal.y * y2 + offset) / normal.z;
            
            // First triangle
            vertices[vertexIndex++] = {x1, y1, z1};
            vertices[vertexIndex++] = {x2, y1, z2};
            vertices[vertexIndex++] = {x1, y2, z3};
            
            // Second triangle
            vertices[vertexIndex++] = {x2, y1, z2};
            vertices[vertexIndex++] = {x2, y2, z4};
            vertices[vertexIndex++] = {x1, y2, z3};
        }
    }
    
    CHECK_DW_ERROR(dwRenderEngine_unmapBuffer(m_groundPlaneRenderBufferId,
                                              DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
                                              m_renderEngine));
    
    // Render the ground plane with a pleasant earth-tone color
    dwRenderEngine_setColor({0.4f, 0.3f, 0.2f, 0.8f}, m_renderEngine);  // Earth brown with transparency
    dwRenderEngine_renderBuffer(m_groundPlaneRenderBufferId, vertexIndex, m_renderEngine);
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

    std::cout << "========== Frame: " << m_frameNum << " | State: " << getStateString() << " ==========" << std::endl;

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
        if (m_verbose) {
            std::cout << "ICP performed in state: " << getStateString() 
                      << " | Success: " << (icpSuccess ? "YES" : "NO") << std::endl;
        }
    } else {
        // In stitch-only mode, force identity; otherwise keep last ICP transform
        if (m_stitchOnly) {
            m_icpTransform = DW_IDENTITY_TRANSFORMATION3F;
        }
        if (m_verbose && m_alignmentState == AlignmentState::ALIGNED) {
            auto timeUntilNext = PERIODIC_ICP_INTERVAL_SECONDS - 
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - m_lastPeriodicICP).count();
            std::cout << "ICP skipped (aligned state) | Next ICP in: " << timeUntilNext << " seconds" << std::endl;
        }
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

    // Step 6: Update state management
    updateAlignmentState();

    // Step 7: Log results
    logICPResults();

    // Step 8: Save point clouds if enabled
    if (m_savePointClouds && m_frameNum % 10 == 0) { // Save every 10th frame
        std::stringstream ss;
        ss << "frame_" << std::setfill('0') << std::setw(4) << m_frameNum << "_stitched.ply";
        savePointCloudToPLY(m_stitchedPoints, ss.str());
    }

    // Print comprehensive summary
    std::cout << "Frame " << m_frameNum << " processed:" << std::endl;
    std::cout << "  State: " << getStateString() << std::endl;
    std::cout << "  LiDAR A points: " << m_accumulatedPoints[LIDAR_A_INDEX].size << std::endl;
    std::cout << "  LiDAR B points: " << m_accumulatedPoints[LIDAR_B_INDEX].size << std::endl;
    std::cout << "  ICP: " << (icpPerformed ? (icpSuccess ? "PERFORMED & SUCCESS" : "PERFORMED & FAILED") : "SKIPPED") << std::endl;
    if (icpPerformed) {
        std::cout << "  Consecutive Successful: " << m_consecutiveSuccessfulICP << std::endl;
    }
    std::cout << "  Stitched points: " << m_stitchedPoints.size << std::endl;
    std::cout << "  Ground plane: " << (groundPlanePerformed ? 
        (m_filteredGroundPlaneValid ? "FILTERED (stable)" : 
         (m_groundPlaneValid ? "RAW ONLY (unstable)" : "NOT DETECTED")) : "SKIPPED") << std::endl;
    std::cout << "  Total ICP Success Rate: " << (100.0f * m_successfulICPCount / m_frameNum) << "%" << std::endl;
    
    // Show alignment progress if in initial alignment phase
    if (m_alignmentState == AlignmentState::INITIAL_ALIGNMENT) {
        std::cout << "  Alignment Progress: " << m_consecutiveSuccessfulICP << "/" << MIN_SUCCESSFUL_ICP_FOR_ALIGNMENT 
                  << " consecutive successful ICPs" << std::endl;
        if (m_lastICPStats.successful) {
            std::cout << "    RMS Cost: " << (m_lastICPStats.rmsCost*1000) << " mm (target: <" 
                      << (MAX_RMS_COST_FOR_ALIGNMENT*1000) << " mm)" << std::endl;
            std::cout << "    Inlier Fraction: " << (m_lastICPStats.inlierFraction*100) << "% (target: >" 
                      << (MIN_INLIER_FRACTION_FOR_ALIGNMENT*100) << "%)" << std::endl;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
void InterLidarICP::copyToRenderBuffer(uint32_t renderBufferId, uint32_t offset, const dwPointCloud& pointCloud)
{
    uint32_t sizeInBytes = pointCloud.capacity * sizeof(dwVector4f);
    dwVector4f* dataToRender = nullptr;

    dwRenderEngine_mapBuffer(renderBufferId,
                             reinterpret_cast<void**>(&dataToRender),
                             offset,
                             sizeInBytes,
                             DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                             m_renderEngine);
    
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
void InterLidarICP::onRender()
{
    // Clear buffer
    dwRenderEngine_reset(m_renderEngine);

    getMouseView().setCenter(0.f, 0.f, 0.f);

    // Render LiDAR A (top-left, green)
    renderPointCloud(m_lidarTiles[LIDAR_A_INDEX].renderBufferId,
                     m_lidarTiles[LIDAR_A_INDEX].tileId,
                     0,
                     DW_RENDER_ENGINE_COLOR_GREEN,
                     m_rigTransformedPoints[LIDAR_A_INDEX]);
    dwRenderEngine_renderBuffer(m_lidarTiles[LIDAR_A_INDEX].renderBufferId,
                                m_rigTransformedPoints[LIDAR_A_INDEX].size,
                                m_renderEngine);

    // Render LiDAR B (top-right, orange)
    renderPointCloud(m_lidarTiles[LIDAR_B_INDEX].renderBufferId,
                     m_lidarTiles[LIDAR_B_INDEX].tileId,
                     0,
                     DW_RENDER_ENGINE_COLOR_ORANGE,
                     m_rigTransformedPoints[LIDAR_B_INDEX]);
    dwRenderEngine_renderBuffer(m_lidarTiles[LIDAR_B_INDEX].renderBufferId,
                                m_rigTransformedPoints[LIDAR_B_INDEX].size,
                                m_renderEngine);

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

    // Render ground plane in ICP view if detected and aligned
    if (shouldPerformGroundPlaneExtraction() && m_filteredGroundPlaneValid) {
        renderGroundPlane();
    }

    // Render stitched view (bottom-right, cyan)
    renderPointCloud(m_stitchedTile.renderBufferId,
                     m_stitchedTile.tileId,
                     0,
                     DW_RENDER_ENGINE_COLOR_LIGHTBLUE,
                     m_stitchedPoints);
    dwRenderEngine_renderBuffer(m_stitchedTile.renderBufferId,
                                m_stitchedPoints.size,
                                m_renderEngine);

    // Render ground plane in stitched view if detected and aligned
    if (shouldPerformGroundPlaneExtraction() && m_filteredGroundPlaneValid) {
        dwRenderEngine_setTile(m_stitchedTile.tileId, m_renderEngine);
        renderGroundPlane();
    }

    // Add text overlays with state information - render in each individual tile
    // LiDAR A tile (top-left)
    dwRenderEngine_setTile(m_lidarTiles[LIDAR_A_INDEX].tileId, m_renderEngine);
    dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
    dwRenderEngine_setCoordinateRange2D({1.0f, 1.0f}, m_renderEngine);
    dwRenderEngine_setColor({0.0f, 0.6f, 1.0f, 1.0f}, m_renderEngine);
    dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine);
    dwRenderEngine_renderText2D("LiDAR A (Reference)", {0.1f, 0.9f}, m_renderEngine);
    
    // LiDAR B tile (top-right)
    dwRenderEngine_setTile(m_lidarTiles[LIDAR_B_INDEX].tileId, m_renderEngine);
    dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
    dwRenderEngine_setCoordinateRange2D({1.0f, 1.0f}, m_renderEngine);
    dwRenderEngine_setColor({0.0f, 0.6f, 1.0f, 1.0f}, m_renderEngine);
    dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine);
    dwRenderEngine_renderText2D("LiDAR B (Source)", {0.1f, 0.9f}, m_renderEngine);
    
    // ICP alignment tile (bottom-left)
    dwRenderEngine_setTile(m_icpTile.tileId, m_renderEngine);
    dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
    dwRenderEngine_setCoordinateRange2D({1.0f, 1.0f}, m_renderEngine);
    dwRenderEngine_setColor({0.0f, 0.6f, 1.0f, 1.0f}, m_renderEngine);
    dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine);
    
    std::string icpText = "ICP Alignment (";
    icpText += getStateString();
    icpText += ")";
    if (shouldPerformICP()) {
        icpText += m_lastICPStats.successful ? " SUCCESS" : " FAILED";
    } else {
        icpText += " SKIPPED";
    }
    if (shouldPerformGroundPlaneExtraction()) {
        icpText += " + Ground Plane ";
        if (m_filteredGroundPlaneValid) {
            icpText += "(FILTERED)";
        } else if (m_groundPlaneValid) {
            icpText += "(RAW)";
        } else {
            icpText += "(NOT DETECTED)";
        }
    } else {
        icpText += " [Ground Plane: WAITING]";
    }
    dwRenderEngine_renderText2D(icpText.c_str(), {0.1f, 0.9f}, m_renderEngine);
    
    // Stitched result tile (bottom-right)
    dwRenderEngine_setTile(m_stitchedTile.tileId, m_renderEngine);
    dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
    dwRenderEngine_setCoordinateRange2D({1.0f, 1.0f}, m_renderEngine);
    dwRenderEngine_setColor({0.0f, 0.6f, 1.0f, 1.0f}, m_renderEngine);
    dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_20, m_renderEngine);
    
    std::string stitchedText = "Stitched Result (" + std::to_string(m_stitchedPoints.size) + " points)";
    if (shouldPerformGroundPlaneExtraction()) {
        if (m_filteredGroundPlaneValid) {
            stitchedText += " + Filtered Ground Plane";
        } else if (m_groundPlaneValid) {
            stitchedText += " + Raw Ground Plane";
        }
    }
    dwRenderEngine_renderText2D(stitchedText.c_str(), {0.1f, 0.9f}, m_renderEngine);

    // FPS and comprehensive statistics
    renderutils::renderFPS(m_renderEngine, getCurrentFPS());
    
    std::string statsText = "Frame: " + std::to_string(m_frameNum) + 
                           " | State: " + getStateString() +
                           " | ICP Success Rate: " + std::to_string(int(100.0f * m_successfulICPCount / std::max(1u, m_frameNum))) + "%";
    
    if (m_alignmentState == AlignmentState::INITIAL_ALIGNMENT) {
        statsText += " | Progress: " + std::to_string(m_consecutiveSuccessfulICP) + "/" + std::to_string(MIN_SUCCESSFUL_ICP_FOR_ALIGNMENT);
    } else if (m_alignmentState == AlignmentState::ALIGNED) {
        auto timeUntilNext = PERIODIC_ICP_INTERVAL_SECONDS - 
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - m_lastPeriodicICP).count();
        statsText += " | Next ICP: " + std::to_string(timeUntilNext) + "s";
    }
    
    if (m_lidarReadyAnnounced) {
        statsText += " | *** LIDAR READY ***";
    }
    
    renderTexts(statsText.c_str(), {0.5f, 0.05f});
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

    // Release render buffers
    for (uint32_t i = 0; i < m_lidarCount; i++) {
        dwRenderEngine_destroyBuffer(m_lidarTiles[i].renderBufferId, m_renderEngine);
    }
    dwRenderEngine_destroyBuffer(m_icpTile.renderBufferId, m_renderEngine);
    dwRenderEngine_destroyBuffer(m_stitchedTile.renderBufferId, m_renderEngine);
    dwRenderEngine_destroyBuffer(m_groundPlaneRenderBufferId, m_renderEngine);

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
    std::cout << "Inter-Lidar ICP Statistics with State Management:" << std::endl;
    std::cout << "  Total Frames: " << m_frameNum << std::endl;
    std::cout << "  Successful ICP: " << m_successfulICPCount << std::endl;
    std::cout << "  Failed ICP: " << m_failedICPCount << std::endl;
    std::cout << "  Success Rate: " << (100.0f * m_successfulICPCount / std::max(1u, m_frameNum)) << "%" << std::endl;
    std::cout << "  Final State: " << getStateString() << std::endl;
    std::cout << "  Lidar Ready Announced: " << (m_lidarReadyAnnounced ? "YES" : "NO") << std::endl;
    if (m_lidarReadyAnnounced) {
        std::cout << "  Ground Plane Extraction: ENABLED (after alignment)" << std::endl;
    } else {
        std::cout << "  Ground Plane Extraction: DISABLED (alignment incomplete)" << std::endl;
    }
    std::cout << std::string(60, '=') << std::endl;
}