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
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>

#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/pointcloudprocessing/accumulator/PointCloudAccumulator.h>
#include <dw/pointcloudprocessing/icp/PointCloudICP.h>
#include <dw/pointcloudprocessing/pointcloud/PointCloud.h>
#include <dw/pointcloudprocessing/stitcher/PointCloudStitcher.h>
#include <dw/rig/Rig.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/lidar/Lidar.h>
#include <dw/sensors/sensormanager/SensorManager.h>

#include <framework/Checks.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/Log.hpp>
#include <framework/ProgramArguments.hpp>

using namespace dw_samples::common;

//------------------------------------------------------------------------------
// Lidar Recorder with ICP Correction
// Records combined and synchronized point clouds from multiple lidars
//------------------------------------------------------------------------------
class LidarRecorderSample : public DriveWorksSample
{
private:
    static const uint32_t MAX_LIDARS = 4;
    
    // DriveWorks handles
    dwContextHandle_t m_context = DW_NULL_HANDLE;
    dwSALHandle_t m_sal = DW_NULL_HANDLE;
    dwRigHandle_t m_rigConfig = DW_NULL_HANDLE;
    dwSensorManagerHandle_t m_sensorManager = DW_NULL_HANDLE;
    
    // Point cloud processing handles
    dwPointCloudAccumulatorHandle_t m_accumulator[MAX_LIDARS] = {DW_NULL_HANDLE};
    dwPointCloudICPHandle_t m_icp = DW_NULL_HANDLE;
    dwPointCloudStitcherHandle_t m_coordinateConverter[MAX_LIDARS] = {DW_NULL_HANDLE};
    dwPointCloudStitcherHandle_t m_icpTransformer = DW_NULL_HANDLE;
    dwPointCloudStitcherHandle_t m_stitcher = DW_NULL_HANDLE;
    
    // Point cloud buffers
    dwPointCloud m_accumulatedPoints[MAX_LIDARS];
    dwPointCloud m_rigTransformedPoints[MAX_LIDARS];
    dwPointCloud m_icpAlignedPoints[MAX_LIDARS];
    dwPointCloud m_stitchedPoints;
    dwPointCloud m_stitchedPointsHost;  // Host copy for recording
    
    // Sensor information
    uint32_t m_lidarCount = 0;
    dwLidarProperties m_lidarProps[MAX_LIDARS];
    dwTransformation3f m_sensorToRigs[MAX_LIDARS];
    dwTime_t m_registrationTime[MAX_LIDARS];
    
    // Sensor state tracking
    bool m_lidarAccumulated[MAX_LIDARS];
    
    // ICP state
    dwTransformation3f m_icpTransform = DW_IDENTITY_TRANSFORMATION3F;
    bool m_icpInitialized = false;
    uint32_t m_icpInitializationFrames = 0;
    static constexpr uint32_t ICP_INIT_FRAMES = 5;  // Frames to wait before using ICP
    
    // Alignment quality criteria
    static constexpr uint32_t MIN_SUCCESSFUL_ICP_FOR_ALIGNMENT = 5;  // Need 5 consecutive successful ICPs
    static constexpr float32_t MAX_RMS_COST_FOR_ALIGNMENT = 0.1f;   // 100mm max RMS cost
    static constexpr float32_t MIN_INLIER_FRACTION_FOR_ALIGNMENT = 0.75f; // 75% inlier fraction
    
    // Alignment tracking
    uint32_t m_consecutiveSuccessfulICP = 0;
    bool m_alignmentReady = false;  // True when alignment is good enough to record
    dwPointCloudICPResultStats m_lastICPStats{};
    
    // Recording
    std::string m_outputDir;
    uint32_t m_frameCount = 0;
    uint32_t m_recordedFrameCount = 0;
    bool m_recordEnabled = true;
    
    // Configuration
    std::string m_rigFile;
    uint32_t m_maxIters;
    bool m_verbose;
    bool m_enableICP;
    
    // CUDA stream
    cudaStream_t m_stream;

public:
    LidarRecorderSample(const ProgramArguments& args)
        : DriveWorksSample(args) {}

    void initializeDriveWorks(dwContextHandle_t& context)
    {
        CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
        CHECK_DW_ERROR(dwLogger_setLogLevel(m_verbose ? DW_LOG_VERBOSE : DW_LOG_INFO));

        dwContextParameters sdkParams = {};
#ifdef VIBRANTE
        sdkParams.eglDisplay = getEGLDisplay();
#endif
        CHECK_DW_ERROR(dwInitialize(&context, DW_VERSION, &sdkParams));
        CHECK_DW_ERROR(dwSAL_initialize(&m_sal, context));
        
        CHECK_CUDA_ERROR(cudaStreamCreate(&m_stream));
    }

    bool onInitialize() override
    {
        m_rigFile = getArgument("rig-file");
        m_outputDir = getArgument("output-dir");
        m_maxIters = static_cast<uint32_t>(atoi(getArgument("max-icp-iters").c_str()));
        m_verbose = getArgument("verbose") == "true";
        m_enableICP = getArgument("enable-icp") != "false";  // Default enabled
        
        if (m_rigFile.empty()) {
            logError("Rig file must be specified with --rig-file");
            return false;
        }
        
        if (m_outputDir.empty()) {
            logError("Output directory must be specified with --output-dir");
            return false;
        }

        initializeDriveWorks(m_context);
        initSensors();
        initBuffers();
        initAccumulation();
        if (m_enableICP) {
            initICP();
        }
        initStitching();
        initRecording();

        // If ICP is disabled, start recording immediately
        if (!m_enableICP || m_lidarCount < 2) {
            m_alignmentReady = true;
            log("ICP disabled or only one lidar - recording all frames");
        }

        log("Lidar Recorder initialized:");
        log("  Rig file: %s", m_rigFile.c_str());
        log("  Output directory: %s", m_outputDir.c_str());
        log("  Lidars: %d", m_lidarCount);
        log("  ICP enabled: %s", m_enableICP ? "YES" : "NO");
        if (m_enableICP && m_lidarCount >= 2) {
            log("  Max ICP iterations: %d", m_maxIters);
            log("  Alignment criteria: RMS < %.3fmm, inliers > %.1f%%, consecutive: %d",
                MAX_RMS_COST_FOR_ALIGNMENT * 1000.0f,
                MIN_INLIER_FRACTION_FOR_ALIGNMENT * 100.0f,
                MIN_SUCCESSFUL_ICP_FOR_ALIGNMENT);
        }

        return true;
    }

    void onReset() override
    {
        // Reset sensors
        if (m_sensorManager) {
            dwSensorManager_reset(m_sensorManager);
        }
        
        // Reset accumulators
        for (uint32_t i = 0; i < m_lidarCount; i++) {
            if (m_accumulator[i]) {
                dwPointCloudAccumulator_reset(m_accumulator[i]);
            }
        }
        
        m_icpInitialized = false;
        m_icpInitializationFrames = 0;
        m_icpTransform = DW_IDENTITY_TRANSFORMATION3F;
        m_frameCount = 0;
        m_recordedFrameCount = 0;
        m_consecutiveSuccessfulICP = 0;
        m_alignmentReady = false;
    }

    void onRelease() override
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
            if (m_accumulator[i]) {
                dwPointCloudAccumulator_release(m_accumulator[i]);
            }
            if (m_coordinateConverter[i]) {
                dwPointCloudStitcher_release(m_coordinateConverter[i]);
            }
        }
        dwPointCloud_destroyBuffer(&m_stitchedPoints);
        dwPointCloud_destroyBuffer(&m_stitchedPointsHost);

        // Release ICP and stitchers
        if (m_icp) {
            dwPointCloudICP_release(m_icp);
        }
        if (m_icpTransformer) {
            dwPointCloudStitcher_release(m_icpTransformer);
        }
        if (m_stitcher) {
            dwPointCloudStitcher_release(m_stitcher);
        }

        // Release DriveWorks
        if (m_sal) {
            dwSAL_release(m_sal);
        }
        if (m_context) {
            dwRelease(m_context);
        }
        
        if (m_stream) {
            cudaStreamDestroy(m_stream);
        }
        
        dwLogger_release();

        log("Processed %d frames, recorded %d aligned frames to %s", 
            m_frameCount, m_recordedFrameCount, m_outputDir.c_str());
    }

    void onProcess() override
    {
        // Get spins from all lidars
        if (!getSpinFromAllLidars()) {
            logWarn("Failed to get data from all lidars");
            return;
        }

        // Apply rig transformations
        applyRigTransformations();

        // Perform ICP correction if enabled
        if (m_enableICP && m_lidarCount >= 2) {
            performICP();
        }

        // Stitch point clouds
        stitchPointClouds();

        // Record the stitched point cloud only if alignment is ready
        if (m_alignmentReady || !m_enableICP) {
            recordPointCloud();
            m_recordedFrameCount++;
        } else {
            if (m_verbose && m_frameCount % 10 == 0) {
                log("Waiting for alignment (frame %d, consecutive ICP: %d/%d)", 
                    m_frameCount, m_consecutiveSuccessfulICP, MIN_SUCCESSFUL_ICP_FOR_ALIGNMENT);
            }
        }

        m_frameCount++;
        
        if (m_verbose && m_alignmentReady && m_frameCount % 10 == 0) {
            log("Recorded frame %d: %d points (total recorded: %d)", 
                m_frameCount, m_stitchedPoints.size, m_recordedFrameCount);
        }
    }

private:
    void initSensors()
    {
        // Initialize Rig configuration
        CHECK_DW_ERROR_MSG(dwRig_initializeFromFile(&m_rigConfig, m_context, m_rigFile.c_str()),
                           "Could not initialize Rig from File");

        // Initialize Sensor Manager
        // NOTE: Host Time Synchronization
        // All sensors initialized via dwSensorManager_initializeFromRig() share the same context.
        // According to DriveWorks documentation: "Timestamps within the same context are guaranteed to be in sync."
        // Host time uses CLOCK_MONOTONIC time source from Xavier, adjusted to UNIX epoch time.
        // For lidars, timestamps are taken when Ethernet packets are read by the host.
        // This provides synchronized timestamps across all lidars without requiring a Time Sensor.
        // If rig file specifies output-timestamp=synced without a Time Sensor, it falls back to host time.
        CHECK_DW_ERROR(dwSensorManager_initializeFromRig(&m_sensorManager, m_rigConfig, 
                                                         DW_SENSORMANGER_MAX_NUM_SENSORS, m_sal));

        // Get lidar count
        CHECK_DW_ERROR(dwSensorManager_getNumSensors(&m_lidarCount, DW_SENSOR_LIDAR, m_sensorManager));
        
        if (m_lidarCount == 0) {
            throw std::runtime_error("No lidar sensors found in rig file");
        }
        
        if (m_lidarCount > MAX_LIDARS) {
            logWarn("Found %d lidars, but only %d are supported. Using first %d.", 
                    m_lidarCount, MAX_LIDARS, MAX_LIDARS);
            m_lidarCount = MAX_LIDARS;
        }

        log("Found %d LiDAR sensors", m_lidarCount);

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
            
            log("LiDAR %d (%s): %d points/spin, %d packets/spin", 
                i, m_lidarProps[i].deviceString, 
                m_lidarProps[i].pointsPerSpin, 
                m_lidarProps[i].packetsPerSpin);
        }
    }

    void initBuffers()
    {
        dwMemoryType memoryType = DW_MEMORY_TYPE_CUDA;

        // Initialize point cloud buffers for each lidar
        for (uint32_t i = 0; i < m_lidarCount; i++) {
            // Accumulated points buffer (organized for depth map ICP)
            m_accumulatedPoints[i].capacity = m_lidarProps[i].pointsPerSpin;
            m_accumulatedPoints[i].type = memoryType;
            m_accumulatedPoints[i].format = DW_POINTCLOUD_FORMAT_XYZI;
            m_accumulatedPoints[i].organized = true;
            CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_accumulatedPoints[i]));

            // Rig transformed points buffer
            m_rigTransformedPoints[i].capacity = m_lidarProps[i].pointsPerSpin;
            m_rigTransformedPoints[i].type = memoryType;
            m_rigTransformedPoints[i].format = DW_POINTCLOUD_FORMAT_XYZI;
            m_rigTransformedPoints[i].organized = true;
            CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_rigTransformedPoints[i]));

            // ICP aligned points buffer
            m_icpAlignedPoints[i].capacity = m_lidarProps[i].pointsPerSpin;
            m_icpAlignedPoints[i].type = memoryType;
            m_icpAlignedPoints[i].format = DW_POINTCLOUD_FORMAT_XYZI;
            m_icpAlignedPoints[i].organized = true;
            CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_icpAlignedPoints[i]));
        }

        // Stitched point cloud buffer
        uint32_t totalCapacity = 0;
        for (uint32_t i = 0; i < m_lidarCount; i++) {
            totalCapacity += m_lidarProps[i].pointsPerSpin;
        }
        
        m_stitchedPoints.capacity = totalCapacity;
        m_stitchedPoints.type = memoryType;
        m_stitchedPoints.format = DW_POINTCLOUD_FORMAT_XYZI;
        m_stitchedPoints.organized = false;
        CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_stitchedPoints));
        
        // Host copy for recording
        m_stitchedPointsHost.capacity = totalCapacity;
        m_stitchedPointsHost.type = DW_MEMORY_TYPE_CPU;
        m_stitchedPointsHost.format = DW_POINTCLOUD_FORMAT_XYZI;
        m_stitchedPointsHost.organized = false;
        CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_stitchedPointsHost));
    }

    void initAccumulation()
    {
        for (uint32_t i = 0; i < m_lidarCount; i++) {
            dwPointCloudAccumulatorParams params{};
            CHECK_DW_ERROR(dwPointCloudAccumulator_getDefaultParams(&params));
            
            params.organized = true;
            params.memoryType = DW_MEMORY_TYPE_CUDA;
            params.enableMotionCompensation = false;
            params.egomotion = DW_NULL_HANDLE;
            
            // Get sensor transformation
            uint32_t lidarSensorIndex;
            CHECK_DW_ERROR(dwSensorManager_getSensorIndex(&lidarSensorIndex, DW_SENSOR_LIDAR, i, m_sensorManager));
            CHECK_DW_ERROR(dwRig_getSensorToRigTransformation(&params.sensorTransformation, lidarSensorIndex, m_rigConfig));
            
            CHECK_DW_ERROR(dwPointCloudAccumulator_initialize(&m_accumulator[i], &params, &m_lidarProps[i], m_context));
            CHECK_DW_ERROR(dwPointCloudAccumulator_bindOutput(&m_accumulatedPoints[i], m_accumulator[i]));
            CHECK_DW_ERROR(dwPointCloudAccumulator_setCUDAStream(m_stream, m_accumulator[i]));
        }
    }

    void initICP()
    {
        if (m_lidarCount < 2) {
            logWarn("ICP requires at least 2 lidars, but only %d found. ICP disabled.", m_lidarCount);
            m_enableICP = false;
            return;
        }

        dwPointCloudICPParams params{};
        CHECK_DW_ERROR(dwPointCloudICP_getDefaultParams(&params));
        
        params.maxIterations = m_maxIters;
        params.icpType = DW_POINT_CLOUD_ICP_TYPE_DEPTH_MAP;
        
        // Configure depth map parameters
        uint32_t horizontalSamples = m_lidarProps[0].pointsPerSpin / 16;  // VLP16 has 16 beams
        uint32_t verticalBeams = 16;
        
        params.depthmapSize.x = horizontalSamples;
        params.depthmapSize.y = verticalBeams;
        params.maxPoints = params.depthmapSize.x * params.depthmapSize.y;
        
        params.distanceConvergenceTol = 0.0005f;
        params.angleConvergenceTol = 0.005f;
        
        CHECK_DW_ERROR(dwPointCloudICP_initialize(&m_icp, &params, m_context));
        CHECK_DW_ERROR(dwPointCloudICP_setCUDAStream(m_stream, m_icp));
        
        log("ICP initialized: depthmap %dx%d, max iterations %d", 
            params.depthmapSize.x, params.depthmapSize.y, params.maxIterations);
    }

    void initStitching()
    {
        // Initialize coordinate converters for individual transformations
        for (uint32_t i = 0; i < m_lidarCount; i++) {
            CHECK_DW_ERROR(dwPointCloudStitcher_initialize(&m_coordinateConverter[i], m_context));
            CHECK_DW_ERROR(dwPointCloudStitcher_bindOutput(&m_rigTransformedPoints[i], m_coordinateConverter[i]));
            CHECK_DW_ERROR(dwPointCloudStitcher_setCUDAStream(m_stream, m_coordinateConverter[i]));
        }
        
        // Initialize ICP transformer for applying ICP transformation
        if (m_enableICP && m_lidarCount >= 2) {
            CHECK_DW_ERROR(dwPointCloudStitcher_initialize(&m_icpTransformer, m_context));
            CHECK_DW_ERROR(dwPointCloudStitcher_bindOutput(&m_icpAlignedPoints[1], m_icpTransformer));
            CHECK_DW_ERROR(dwPointCloudStitcher_setCUDAStream(m_stream, m_icpTransformer));
        }
        
        // Initialize main stitcher
        CHECK_DW_ERROR(dwPointCloudStitcher_initialize(&m_stitcher, m_context));
        CHECK_DW_ERROR(dwPointCloudStitcher_bindOutput(&m_stitchedPoints, m_stitcher));
        CHECK_DW_ERROR(dwPointCloudStitcher_setCUDAStream(m_stream, m_stitcher));
    }

    void initRecording()
    {
        // Create output directory if it doesn't exist
        auto dirExist = [](const std::string& dir) -> bool {
            struct stat buffer;
            return (stat(dir.c_str(), &buffer) == 0);
        };

        if (!dirExist(m_outputDir)) {
            if (mkdir(m_outputDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
                throw std::runtime_error("Failed to create output directory: " + m_outputDir);
            }
        }
        
        log("Recording to directory: %s", m_outputDir.c_str());
    }

    bool getSpinFromAllLidars()
    {
        // Synchronization Strategy:
        // 1. All lidars share the same context, so their timestamps are synchronized to host time
        // 2. We wait for complete spins from all lidars (spin-based synchronization)
        // 3. Each lidar's timestamp (m_registrationTime) is in the synchronized host time domain
        // 4. The stitched point cloud uses these synchronized timestamps
        
        // Reset accumulation state
        for (uint32_t i = 0; i < m_lidarCount; i++) {
            m_lidarAccumulated[i] = false;
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

            // dwSensorManager_acquireNextEvent() provides events from all sensors
            // with timestamps synchronized to the same host time context
            const dwSensorEvent* acquiredEvent = nullptr;
            dwStatus status = dwSensorManager_acquireNextEvent(&acquiredEvent, 1000, m_sensorManager);

            if (status != DW_SUCCESS) {
                if (status == DW_END_OF_STREAM) {
                    log("End of stream reached");
                    return false;
                } else if (status == DW_TIME_OUT) {
                    continue;
                } else {
                    logError("Unable to acquire sensor event: %s", dwGetStatusName(status));
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

                if (!m_lidarAccumulated[lidarIndex]) {
                    CHECK_DW_ERROR(dwPointCloudAccumulator_addLidarPacket(packet, m_accumulator[lidarIndex]));

                    bool ready = false;
                    CHECK_DW_ERROR(dwPointCloudAccumulator_isReady(&ready, m_accumulator[lidarIndex]));

                    if (ready) {
                        m_registrationTime[lidarIndex] = packet->hostTimestamp;
                        m_lidarAccumulated[lidarIndex] = true;
                        CHECK_DW_ERROR(dwPointCloudAccumulator_process(m_accumulator[lidarIndex]));
                        numLidarsAccumulated++;
                    }
                }
            }

            CHECK_DW_ERROR(dwSensorManager_releaseAcquiredEvent(acquiredEvent, m_sensorManager));
        }

        return true;
    }

    void applyRigTransformations()
    {
        // Transform each point cloud using their respective sensor-to-rig transformations
        for (uint32_t i = 0; i < m_lidarCount; i++) {
            CHECK_DW_ERROR(dwPointCloudStitcher_bindInput(DW_BIND_SLOT_1,
                                                          &m_accumulatedPoints[i],
                                                          &m_sensorToRigs[i],
                                                          m_coordinateConverter[i]));
            CHECK_DW_ERROR(dwPointCloudStitcher_process(m_coordinateConverter[i]));
            m_rigTransformedPoints[i].organized = m_accumulatedPoints[i].organized;
        }
    }

    bool performICP()
    {
        if (m_lidarCount < 2 || !m_icp) {
            return false;
        }

        // Wait a few frames before starting ICP to allow accumulation
        if (!m_icpInitialized) {
            m_icpInitializationFrames++;
            if (m_icpInitializationFrames < ICP_INIT_FRAMES) {
                return false;
            }
            m_icpInitialized = true;
            log("ICP initialization complete, starting alignment");
        }

        // Use first lidar as target, second as source
        const dwPointCloud* targetPointCloud = &m_rigTransformedPoints[0];
        const dwPointCloud* sourcePointCloud = &m_rigTransformedPoints[1];
        
        if (!targetPointCloud->organized || !sourcePointCloud->organized) {
            logWarn("ICP requires organized point clouds");
            return false;
        }
        
        // Use previous ICP result as initial guess if available
        dwTransformation3f initialGuess = m_icpTransform;
        
        // Bind ICP inputs and outputs
        CHECK_DW_ERROR(dwPointCloudICP_bindInput(sourcePointCloud, targetPointCloud, &initialGuess, m_icp));
        CHECK_DW_ERROR(dwPointCloudICP_bindOutput(&m_icpTransform, m_icp));
        
        // Perform ICP
        dwStatus icpStatus = dwPointCloudICP_process(m_icp);
        
        if (icpStatus == DW_SUCCESS) {
            CHECK_DW_ERROR(dwPointCloudICP_getLastResultStats(&m_lastICPStats, m_icp));
            
            // Check if ICP result meets quality criteria
            bool icpGood = (m_lastICPStats.rmsCost <= MAX_RMS_COST_FOR_ALIGNMENT) &&
                          (m_lastICPStats.inlierFraction >= MIN_INLIER_FRACTION_FOR_ALIGNMENT);
            
            if (icpGood) {
                m_consecutiveSuccessfulICP++;
                
                // Check if we have enough consecutive good ICPs
                if (!m_alignmentReady && m_consecutiveSuccessfulICP >= MIN_SUCCESSFUL_ICP_FOR_ALIGNMENT) {
                    m_alignmentReady = true;
                    log("*** Alignment ready! Starting to record frames ***");
                    log("  RMS cost: %.3fmm (max: %.3fmm)", 
                        m_lastICPStats.rmsCost * 1000.0f, 
                        MAX_RMS_COST_FOR_ALIGNMENT * 1000.0f);
                    log("  Inlier fraction: %.1f%% (min: %.1f%%)", 
                        m_lastICPStats.inlierFraction * 100.0f,
                        MIN_INLIER_FRACTION_FOR_ALIGNMENT * 100.0f);
                }
            } else {
                // Reset counter if ICP quality drops
                if (m_consecutiveSuccessfulICP > 0) {
                    if (m_verbose) {
                        logWarn("ICP quality dropped: RMS=%.3fmm (max: %.3fmm), inliers=%.1f%% (min: %.1f%%)",
                               m_lastICPStats.rmsCost * 1000.0f, MAX_RMS_COST_FOR_ALIGNMENT * 1000.0f,
                               m_lastICPStats.inlierFraction * 100.0f, MIN_INLIER_FRACTION_FOR_ALIGNMENT * 100.0f);
                    }
                    m_consecutiveSuccessfulICP = 0;
                    m_alignmentReady = false;
                }
            }
            
            if (m_verbose && m_frameCount % 10 == 0) {
                log("ICP: iterations=%d, RMS=%.3fmm, inliers=%.1f%%, consecutive=%d, ready=%s", 
                    m_lastICPStats.actualNumIterations, 
                    m_lastICPStats.rmsCost * 1000.0f,
                    m_lastICPStats.inlierFraction * 100.0f,
                    m_consecutiveSuccessfulICP,
                    m_alignmentReady ? "YES" : "NO");
            }
            return true;
        } else {
            // Reset on ICP failure
            if (m_consecutiveSuccessfulICP > 0) {
                if (m_verbose) {
                    logWarn("ICP failed: %s", dwGetStatusName(icpStatus));
                }
                m_consecutiveSuccessfulICP = 0;
                m_alignmentReady = false;
            }
            // Use identity transform when ICP fails
            m_icpTransform = DW_IDENTITY_TRANSFORMATION3F;
            return false;
        }
    }

    void stitchPointClouds()
    {
        // Copy first lidar points directly (reference)
        CHECK_CUDA_ERROR(cudaMemcpy(m_icpAlignedPoints[0].points, 
                                    m_rigTransformedPoints[0].points,
                                    sizeof(dwVector4f) * m_rigTransformedPoints[0].size,
                                    cudaMemcpyDeviceToDevice));
        m_icpAlignedPoints[0].size = m_rigTransformedPoints[0].size;
        
        // Apply ICP transformation to second lidar if ICP is enabled
        if (m_enableICP && m_lidarCount >= 2) {
            CHECK_DW_ERROR(dwPointCloudStitcher_bindInput(DW_BIND_SLOT_1,
                                                          &m_rigTransformedPoints[1],
                                                          &m_icpTransform,
                                                          m_icpTransformer));
            CHECK_DW_ERROR(dwPointCloudStitcher_process(m_icpTransformer));
        } else if (m_lidarCount >= 2) {
            // No ICP, just copy
            CHECK_CUDA_ERROR(cudaMemcpy(m_icpAlignedPoints[1].points, 
                                        m_rigTransformedPoints[1].points,
                                        sizeof(dwVector4f) * m_rigTransformedPoints[1].size,
                                        cudaMemcpyDeviceToDevice));
            m_icpAlignedPoints[1].size = m_rigTransformedPoints[1].size;
        }
        
        // Stitch all point clouds together
        CHECK_DW_ERROR(dwPointCloudStitcher_bindInput(DW_BIND_SLOT_1,
                                                      &m_icpAlignedPoints[0],
                                                      &DW_IDENTITY_TRANSFORMATION3F,
                                                      m_stitcher));
        
        if (m_lidarCount >= 2) {
            CHECK_DW_ERROR(dwPointCloudStitcher_bindInput(DW_BIND_SLOT_2,
                                                          &m_icpAlignedPoints[1],
                                                          &DW_IDENTITY_TRANSFORMATION3F,
                                                          m_stitcher));
        }
        
        // Add additional lidars if present
        for (uint32_t i = 2; i < m_lidarCount; i++) {
            // Copy additional lidars directly (no ICP for them)
            CHECK_CUDA_ERROR(cudaMemcpy(m_icpAlignedPoints[i].points, 
                                        m_rigTransformedPoints[i].points,
                                        sizeof(dwVector4f) * m_rigTransformedPoints[i].size,
                                        cudaMemcpyDeviceToDevice));
            m_icpAlignedPoints[i].size = m_rigTransformedPoints[i].size;
            
            // Note: DriveWorks stitcher supports up to 2 inputs, so for more lidars
            // we would need to stitch iteratively or use a different approach
            logWarn("More than 2 lidars not fully supported in stitching");
        }
        
        CHECK_DW_ERROR(dwPointCloudStitcher_process(m_stitcher));
    }

    void recordPointCloud()
    {
        if (m_outputDir.empty()) {
            return;
        }

        // Copy to host for recording
        CHECK_CUDA_ERROR(cudaMemcpy(m_stitchedPointsHost.points, 
                                    m_stitchedPoints.points,
                                    sizeof(dwVector4f) * m_stitchedPoints.size,
                                    cudaMemcpyDeviceToHost));
        m_stitchedPointsHost.size = m_stitchedPoints.size;

        // Use first lidar timestamp as filename (same as lidar_replay sample)
        dwTime_t timestamp = m_registrationTime[0];
        
        // Create filename: <timestamp>.bin
        std::string filename = m_outputDir + "/" + std::to_string(timestamp) + ".bin";
        
        // Write point cloud data to file (same format as lidar_replay)
        std::ofstream fout;
        fout.open(filename, std::ios::binary | std::ios::out);
        if (!fout.is_open()) {
            logWarn("Failed to open file for writing: %s", filename.c_str());
            return;
        }
        
        const dwVector4f* points = static_cast<const dwVector4f*>(m_stitchedPointsHost.points);
        fout.write(reinterpret_cast<const char*>(points), 
                   sizeof(dwVector4f) * m_stitchedPointsHost.size);
        fout.close();
    }
};

//------------------------------------------------------------------------------
int main(int argc, const char** argv)
{
    ProgramArguments args(argc, argv,
                          {
                              ProgramArguments::Option_t("rig-file", ""),
                              ProgramArguments::Option_t("output-dir", ""),
                              ProgramArguments::Option_t("max-icp-iters", "50"),
                              ProgramArguments::Option_t("enable-icp", "true"),
                              ProgramArguments::Option_t("verbose", "false")
                          },
                          "Lidar Recorder with ICP Correction\n"
                          "Records combined and synchronized point clouds from multiple lidars\n"
                          "Creates one .bin file per spin, named with timestamp\n\n"
                          "Required arguments:\n"
                          "  --rig-file=<path>          : Path to rig configuration file\n"
                          "  --output-dir=<path>        : Path to output directory for recorded files\n\n"
                          "Optional arguments:\n"
                          "  --max-icp-iters=<N>        : Maximum ICP iterations (default: 50)\n"
                          "  --enable-icp=<true|false>  : Enable ICP correction (default: true)\n"
                          "  --verbose=<true|false>     : Enable verbose logging (default: false)\n");

    LidarRecorderSample app(args);
    return app.run();
}

