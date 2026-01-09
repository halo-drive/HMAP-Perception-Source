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
// SPDX-FileCopyrightText: Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

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

#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Visualization.h>

#include <framework/Checks.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/Log.hpp>
#include <framework/Mat4.hpp>
#include <framework/MathUtils.hpp>
#include <framework/MouseView3D.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/RenderUtils.hpp>

using namespace dw_samples::common;

//------------------------------------------------------------------------------
// Lidar Calibration - Run ICP and output correction values for rig file
//------------------------------------------------------------------------------
class LidarCalibration : public DriveWorksSample
{
private:
    static const uint32_t NUM_LIDARS = 2;
    static const uint32_t LIDAR_A_INDEX = 0;
    static const uint32_t LIDAR_B_INDEX = 1;

    // Configuration
    std::string m_rigFile;
    uint32_t m_numFrames;
    uint32_t m_maxIters;
    bool m_verbose;

    // DriveWorks Handles
    dwContextHandle_t m_context = DW_NULL_HANDLE;
    dwSALHandle_t m_sal = DW_NULL_HANDLE;
    dwRigHandle_t m_rigConfig = DW_NULL_HANDLE;
    dwSensorManagerHandle_t m_sensorManager = DW_NULL_HANDLE;
    
    // Visualization
    dwVisualizationContextHandle_t m_viz = DW_NULL_HANDLE;
    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;

    // Point Cloud Processing Handles
    dwPointCloudAccumulatorHandle_t m_accumulator[NUM_LIDARS] = {DW_NULL_HANDLE};
    dwPointCloudICPHandle_t m_icp = DW_NULL_HANDLE;
    dwPointCloudStitcherHandle_t m_coordinateConverter[NUM_LIDARS] = {DW_NULL_HANDLE};

    // Sensor Information
    uint32_t m_lidarCount = 0;
    dwLidarProperties m_lidarProps[NUM_LIDARS];
    dwTransformation3f m_sensorToRigs[NUM_LIDARS];
    std::string m_lidarNames[NUM_LIDARS];

    // Point Cloud Buffers
    dwPointCloud m_accumulatedPoints[NUM_LIDARS];
    dwPointCloud m_rigTransformedPoints[NUM_LIDARS];

    // ICP State
    dwTransformation3f m_icpTransform = DW_IDENTITY_TRANSFORMATION3F;
    
    // Accumulate corrections for averaging
    dwVector3f m_totalTranslation = {0.0f, 0.0f, 0.0f};
    dwVector3f m_totalRotation = {0.0f, 0.0f, 0.0f};  // roll, pitch, yaw in radians

    // Frame counter
    uint32_t m_frameNum = 0;
    uint32_t m_successfulICPCount = 0;
    
    // Render buffers
    uint32_t m_lidarABufferId = 0;
    uint32_t m_lidarBBufferId = 0;
    uint32_t m_alignedBufferId = 0;
    
    // Host point clouds for rendering (updated in onProcess, rendered in onRender)
    std::unique_ptr<dwVector4f[]> m_hostPointsA;
    std::unique_ptr<dwVector4f[]> m_hostPointsB;
    uint32_t m_hostPointsASize = 0;
    uint32_t m_hostPointsBSize = 0;

public:
    LidarCalibration(const ProgramArguments& args)
        : DriveWorksSample(args)
    {
        m_rigFile = getArgument("rig");
        m_numFrames = static_cast<uint32_t>(atoi(getArgument("num-frames").c_str()));
        m_maxIters = static_cast<uint32_t>(atoi(getArgument("max-iters").c_str()));
        m_verbose = getArgument("verbose") == "true";

        if (m_numFrames == 0)
            m_numFrames = 100;

        std::cout << "=== Lidar Calibration Tool ===" << std::endl;
        std::cout << "Rig File: " << m_rigFile << std::endl;
        std::cout << "Frames: " << m_numFrames << std::endl;
        std::cout << "Max ICP iterations: " << m_maxIters << std::endl;
    }

    bool onInitialize() override
    {
        // Initialize DriveWorks
        CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
        CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_WARN));
        
        dwContextParameters sdkParams = {};
        CHECK_DW_ERROR(dwInitialize(&m_context, DW_VERSION, &sdkParams));
        CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_context));

        // Load rig configuration
        CHECK_DW_ERROR(dwRig_initializeFromFile(&m_rigConfig, m_context, m_rigFile.c_str()));

        // Initialize sensor manager from rig
        CHECK_DW_ERROR(dwSensorManager_initializeFromRig(&m_sensorManager, m_rigConfig, 1024, m_sal));

        // Get lidar count and properties
        CHECK_DW_ERROR(dwRig_getSensorCountOfType(&m_lidarCount, DW_SENSOR_LIDAR, m_rigConfig));

        if (m_lidarCount != NUM_LIDARS)
        {
            logError("Expected %u lidars, found %u\n", NUM_LIDARS, m_lidarCount);
            return false;
        }

        std::cout << "Found " << m_lidarCount << " lidars" << std::endl;

        // Initialize visualization
        CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_context));
        
        // Initialize render engine
        dwRenderEngineParams renderParams{};
        CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&renderParams, getWindowWidth(), getWindowHeight()));
        renderParams.defaultTile.backgroundColor = {0.0f, 0.0f, 0.0f, 1.0f};
        CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &renderParams, m_viz));
        
        std::cout << "Visualization initialized" << std::endl;

        // Get sensor information from rig
        for (uint32_t i = 0; i < NUM_LIDARS; ++i)
        {
            uint32_t sensorIndex;
            CHECK_DW_ERROR(dwRig_findSensorByTypeIndex(&sensorIndex, DW_SENSOR_LIDAR, i, m_rigConfig));

            dwSensorHandle_t sensorHandle;
            CHECK_DW_ERROR(dwSensorManager_getSensorHandle(&sensorHandle, sensorIndex, m_sensorManager));
            CHECK_DW_ERROR(dwSensorLidar_getProperties(&m_lidarProps[i], sensorHandle));
            CHECK_DW_ERROR(dwRig_getSensorToRigTransformation(&m_sensorToRigs[i], sensorIndex, m_rigConfig));

            const char* sensorName;
            CHECK_DW_ERROR(dwRig_getSensorName(&sensorName, sensorIndex, m_rigConfig));
            m_lidarNames[i] = std::string(sensorName);

            std::cout << "  " << m_lidarNames[i] << ": " << m_lidarProps[i].pointsPerSpin << " points/spin" << std::endl;
        }

        // Allocate point cloud buffers
        for (uint32_t i = 0; i < NUM_LIDARS; ++i)
        {
            // Accumulated points buffer
            m_accumulatedPoints[i].capacity = m_lidarProps[i].pointsPerSpin;
            m_accumulatedPoints[i].type = DW_MEMORY_TYPE_CUDA;
            m_accumulatedPoints[i].format = DW_POINTCLOUD_FORMAT_XYZI;
            m_accumulatedPoints[i].organized = true;
            CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_accumulatedPoints[i]));

            // Rig transformed points buffer
            m_rigTransformedPoints[i].capacity = m_lidarProps[i].pointsPerSpin;
            m_rigTransformedPoints[i].type = DW_MEMORY_TYPE_CUDA;
            m_rigTransformedPoints[i].format = DW_POINTCLOUD_FORMAT_XYZI;
            m_rigTransformedPoints[i].organized = true;
            CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_rigTransformedPoints[i]));
        }

        // Initialize accumulators
        for (uint32_t i = 0; i < NUM_LIDARS; ++i)
        {
            dwPointCloudAccumulatorParams params{};
            CHECK_DW_ERROR(dwPointCloudAccumulator_getDefaultParams(&params));
            
            params.organized = true;
            params.memoryType = DW_MEMORY_TYPE_CUDA;
            params.enableMotionCompensation = false;
            params.egomotion = DW_NULL_HANDLE;

            uint32_t sensorIndex;
            CHECK_DW_ERROR(dwRig_findSensorByTypeIndex(&sensorIndex, DW_SENSOR_LIDAR, i, m_rigConfig));
            CHECK_DW_ERROR(dwRig_getSensorToRigTransformation(&params.sensorTransformation, sensorIndex, m_rigConfig));

            CHECK_DW_ERROR(dwPointCloudAccumulator_initialize(&m_accumulator[i], &params, &m_lidarProps[i], m_context));
            CHECK_DW_ERROR(dwPointCloudAccumulator_bindOutput(&m_accumulatedPoints[i], m_accumulator[i]));
        }

        // Initialize coordinate converters
        for (uint32_t i = 0; i < NUM_LIDARS; ++i)
        {
            CHECK_DW_ERROR(dwPointCloudStitcher_initialize(&m_coordinateConverter[i], m_context));
            CHECK_DW_ERROR(dwPointCloudStitcher_bindOutput(&m_rigTransformedPoints[i], m_coordinateConverter[i]));
        }

        // Initialize ICP
        dwPointCloudICPParams icpParams{};
        CHECK_DW_ERROR(dwPointCloudICP_getDefaultParams(&icpParams));
        
        icpParams.maxIterations = m_maxIters;
        icpParams.icpType = DW_POINT_CLOUD_ICP_TYPE_DEPTH_MAP;
        
        uint32_t horizontalSamples = m_lidarProps[0].pointsPerSpin / 16;
        icpParams.depthmapSize.x = horizontalSamples;
        icpParams.depthmapSize.y = 16;
        icpParams.maxPoints = icpParams.depthmapSize.x * icpParams.depthmapSize.y;

        CHECK_DW_ERROR(dwPointCloudICP_initialize(&m_icp, &icpParams, m_context));

        // Create render buffers for point clouds
        uint32_t maxPoints = std::max(m_lidarProps[0].pointsPerSpin, m_lidarProps[1].pointsPerSpin);
        
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_lidarABufferId,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                   sizeof(dwVector4f), 0, maxPoints, m_renderEngine));
        
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_lidarBBufferId,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                   sizeof(dwVector4f), 0, maxPoints, m_renderEngine));
        
        // Allocate host memory for point clouds
        m_hostPointsA.reset(new dwVector4f[maxPoints]);
        m_hostPointsB.reset(new dwVector4f[maxPoints]);
        
        std::cout << "Render buffers created" << std::endl;

        // Start sensors
        CHECK_DW_ERROR(dwSensorManager_start(m_sensorManager));

        std::cout << "Starting calibration...\n" << std::endl;
        return true;
    }

    void onRelease() override
    {
        // Output results
        outputCalibrationResults();

        // Release render buffers
        if (m_lidarABufferId != 0)
            CHECK_DW_ERROR(dwRenderEngine_destroyBuffer(m_lidarABufferId, m_renderEngine));
        if (m_lidarBBufferId != 0)
            CHECK_DW_ERROR(dwRenderEngine_destroyBuffer(m_lidarBBufferId, m_renderEngine));
        if (m_alignedBufferId != 0)
            CHECK_DW_ERROR(dwRenderEngine_destroyBuffer(m_alignedBufferId, m_renderEngine));

        // Release render engine and visualization
        if (m_renderEngine != DW_NULL_HANDLE)
            CHECK_DW_ERROR(dwRenderEngine_release(m_renderEngine));
        if (m_viz != DW_NULL_HANDLE)
            CHECK_DW_ERROR(dwVisualizationRelease(m_viz));

        // Release resources
        if (m_icp != DW_NULL_HANDLE)
            CHECK_DW_ERROR(dwPointCloudICP_release(m_icp));

        for (uint32_t i = 0; i < NUM_LIDARS; ++i)
        {
            if (m_coordinateConverter[i] != DW_NULL_HANDLE)
                CHECK_DW_ERROR(dwPointCloudStitcher_release(m_coordinateConverter[i]));
            if (m_accumulator[i] != DW_NULL_HANDLE)
                CHECK_DW_ERROR(dwPointCloudAccumulator_release(m_accumulator[i]));
            
            // Release point cloud buffers
            dwPointCloud_destroyBuffer(&m_accumulatedPoints[i]);
            dwPointCloud_destroyBuffer(&m_rigTransformedPoints[i]);
        }

        if (m_sensorManager != DW_NULL_HANDLE)
        {
            dwSensorManager_stop(m_sensorManager);
            CHECK_DW_ERROR(dwSensorManager_release(m_sensorManager));
        }

        if (m_rigConfig != DW_NULL_HANDLE)
            CHECK_DW_ERROR(dwRig_release(m_rigConfig));

        if (m_sal != DW_NULL_HANDLE)
            CHECK_DW_ERROR(dwSAL_release(m_sal));

        if (m_context != DW_NULL_HANDLE)
            CHECK_DW_ERROR(dwRelease(m_context));

        CHECK_DW_ERROR(dwLogger_release());
    }

    void onProcess() override
    {
        if (m_frameNum >= m_numFrames)
        {
            stop();
            return;
        }

        if (!getSpinFromBothLidars())
        {
            stop();
            return;
        }

        applyRigTransformations();

        if (performICP())
            m_successfulICPCount++;

        // Update render buffers (like lidar_replay does in updateFrame)
        updateRenderBuffers();

        m_frameNum++;

        if (m_frameNum % 10 == 0)
            std::cout << "Processed " << m_frameNum << "/" << m_numFrames 
                      << " (" << m_successfulICPCount << " successful)" << std::endl;
    }
    
    void onRender() override
    {
        dwRenderEngine_reset(m_renderEngine);
        getMouseView().setCenter(0.0f, 0.0f, 0.0f);
        
        dwRenderEngine_setTile(0, m_renderEngine);
        dwRenderEngine_setBackgroundColor({0.1f, 0.1f, 0.1f, 1.0f}, m_renderEngine);
        
        // Set up 3D rendering state
        dwMatrix4f modelView;
        Mat4_AxB(modelView.array, getMouseView().getModelView()->array, DW_IDENTITY_TRANSFORMATION3F.array);
        dwRenderEngine_setModelView(&modelView, m_renderEngine);
        dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine);
        
        // Render aligned point clouds (after ICP)
        renderAlignedPointClouds();
        
        // Render statistics
        renderStatistics();
    }
    
    void onResizeWindow(int width, int height) override
    {
        dwRectf bounds{.x = 0.0f, .y = 0.0f, .width = static_cast<float>(width), .height = static_cast<float>(height)};
        dwRenderEngine_setBounds(bounds, m_renderEngine);
    }

private:
    bool getSpinFromBothLidars()
    {
        bool gotSpin[NUM_LIDARS] = {false, false};
        bool lidarAccumulated[NUM_LIDARS] = {false, false};

        while (!(gotSpin[0] && gotSpin[1]))
        {
            const dwSensorEvent* event = nullptr;
            dwStatus status = dwSensorManager_acquireNextEvent(&event, 100000, m_sensorManager);

            if (status != DW_SUCCESS)
            {
                if (status == DW_END_OF_STREAM)
                    return false;
                else if (status == DW_TIME_OUT)
                    continue;
                else
                    return false;
            }

            if (event->type == DW_SENSOR_LIDAR)
            {
                const dwLidarDecodedPacket* packet = event->lidFrame;
                const uint32_t idx = event->sensorTypeIndex;

                if (idx < NUM_LIDARS && !lidarAccumulated[idx])
                {
                    CHECK_DW_ERROR(dwPointCloudAccumulator_addLidarPacket(packet, m_accumulator[idx]));

                    bool ready = false;
                    CHECK_DW_ERROR(dwPointCloudAccumulator_isReady(&ready, m_accumulator[idx]));

                    if (ready)
                    {
                        CHECK_DW_ERROR(dwPointCloudAccumulator_process(m_accumulator[idx]));
                        lidarAccumulated[idx] = true;
                        gotSpin[idx] = true;
                    }
                }
            }

            CHECK_DW_ERROR(dwSensorManager_releaseAcquiredEvent(event, m_sensorManager));
        }

        return true;
    }

    void applyRigTransformations()
    {
        for (uint32_t i = 0; i < NUM_LIDARS; ++i)
        {
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
        // Always use identity as initial guess for consistent per-frame measurements
        dwTransformation3f initialGuess = DW_IDENTITY_TRANSFORMATION3F;

        CHECK_DW_ERROR(dwPointCloudICP_bindInput(&m_rigTransformedPoints[LIDAR_B_INDEX],
                                                 &m_rigTransformedPoints[LIDAR_A_INDEX],
                                                 &initialGuess, m_icp));
        CHECK_DW_ERROR(dwPointCloudICP_bindOutput(&m_icpTransform, m_icp));

        dwStatus status = dwPointCloudICP_process(m_icp);

        if (status == DW_SUCCESS)
        {
            dwPointCloudICPResultStats stats{};
            CHECK_DW_ERROR(dwPointCloudICP_getLastResultStats(&stats, m_icp));

            // Extract translation and rotation from this frame
            dwVector3f translation = getTranslation(m_icpTransform);
            float32_t roll, pitch, yaw;
            transformationToRollPitchYaw(m_icpTransform, roll, pitch, yaw);

            // Accumulate for averaging
            m_totalTranslation.x += translation.x;
            m_totalTranslation.y += translation.y;
            m_totalTranslation.z += translation.z;
            
            m_totalRotation.x += roll;
            m_totalRotation.y += pitch;
            m_totalRotation.z += yaw;

            if (m_verbose)
            {
                std::cout << "Frame " << m_frameNum << ": iter=" << stats.actualNumIterations
                          << " rms=" << stats.rmsCost << " inlier=" << (stats.inlierFraction * 100) << "%"
                          << " | t=[" << translation.x << "," << translation.y << "," << translation.z << "]"
                          << " r=[" << RAD2DEG(roll) << "," << RAD2DEG(pitch) << "," << RAD2DEG(yaw) << "]°"
                          << std::endl;
            }

            return true;
        }

        return false;
    }

    dwVector3f getTranslation(const dwTransformation3f& T)
    {
        return {T.array[12], T.array[13], T.array[14]};
    }
    
    void transformationToRollPitchYaw(const dwTransformation3f& T, float32_t& roll, float32_t& pitch, float32_t& yaw)
    {
        // Extract rotation matrix from transformation
        float32_t r11 = T.array[0], r21 = T.array[1], r31 = T.array[2];
        float32_t r12 = T.array[4], r22 = T.array[5], r32 = T.array[6];
        float32_t r13 = T.array[8], r23 = T.array[9], r33 = T.array[10];

        // Convert to Euler angles (ZYX convention)
        float32_t sy = std::sqrt(r11 * r11 + r21 * r21);
        
        if (sy > 1e-6)
        {
            roll = std::atan2(r32, r33);
            pitch = std::atan2(-r31, sy);
            yaw = std::atan2(r21, r11);
        }
        else
        {
            roll = std::atan2(-r23, r22);
            pitch = std::atan2(-r31, sy);
            yaw = 0;
        }
    }

    void outputCalibrationResults()
    {
        std::cout << "\n=== CALIBRATION RESULTS ===" << std::endl;
        std::cout << "Successful ICP: " << m_successfulICPCount << "/" << m_frameNum << std::endl;

        if (m_successfulICPCount == 0)
        {
            std::cout << "No successful ICP! Cannot generate calibration values." << std::endl;
            return;
        }

        // Average the corrections over all successful frames
        float32_t avgTx = m_totalTranslation.x / m_successfulICPCount;
        float32_t avgTy = m_totalTranslation.y / m_successfulICPCount;
        float32_t avgTz = m_totalTranslation.z / m_successfulICPCount;
        
        float32_t avgRoll = m_totalRotation.x / m_successfulICPCount;
        float32_t avgPitch = m_totalRotation.y / m_successfulICPCount;
        float32_t avgYaw = m_totalRotation.z / m_successfulICPCount;

        float32_t avgRollDeg = RAD2DEG(avgRoll);
        float32_t avgPitchDeg = RAD2DEG(avgPitch);
        float32_t avgYawDeg = RAD2DEG(avgYaw);

        std::cout << "\nAverage ICP Correction for: " << m_lidarNames[LIDAR_B_INDEX] << std::endl;
        std::cout << "-----------------------------------" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Translation (meters): [" << avgTx << ", " << avgTy << ", " << avgTz << "]" << std::endl;
        std::cout << "Rotation (degrees):   [" << avgRollDeg << ", " << avgPitchDeg << ", " << avgYawDeg << "]" << std::endl;
        std::cout << "Rotation (radians):   [" << avgRoll << ", " << avgPitch << ", " << avgYaw << "]" << std::endl;

        // Show magnitude for quick assessment
        float32_t translationMag = std::sqrt(avgTx*avgTx + avgTy*avgTy + avgTz*avgTz);
        float32_t rotationMag = std::sqrt(avgRoll*avgRoll + avgPitch*avgPitch + avgYaw*avgYaw);
        
        std::cout << "\nCorrection Magnitude:" << std::endl;
        std::cout << "  Translation: " << (translationMag * 1000.0f) << " mm" << std::endl;
        std::cout << "  Rotation:    " << RAD2DEG(rotationMag) << " degrees" << std::endl;

        std::cout << "\n=== UPDATE YOUR RIG FILE ===" << std::endl;
        std::cout << "For sensor: " << m_lidarNames[LIDAR_B_INDEX] << std::endl;
        std::cout << "\n\"correction_rig_T\": [" << avgTx << ", " << avgTy << ", " << avgTz << "]," << std::endl;
        std::cout << "\"correction_sensor_R_FLU\": {" << std::endl;
        std::cout << "    \"roll-pitch-yaw\": [" << avgRollDeg << ", " << avgPitchDeg << ", " << avgYawDeg << "]" << std::endl;
        std::cout << "}" << std::endl;
        std::cout << "\n================================" << std::endl;
    }
    
    void updateRenderBuffers()
    {
        // Copy point clouds from GPU to CPU and update render buffers
        if (m_rigTransformedPoints[LIDAR_A_INDEX].size == 0 ||
            m_rigTransformedPoints[LIDAR_B_INDEX].size == 0)
        {
            return;
        }

        // Copy LiDAR A points from GPU to host
        m_hostPointsASize = m_rigTransformedPoints[LIDAR_A_INDEX].size;
        CHECK_CUDA_ERROR(cudaMemcpy(m_hostPointsA.get(),
                                    m_rigTransformedPoints[LIDAR_A_INDEX].points,
                                    sizeof(dwVector4f) * m_hostPointsASize,
                                    cudaMemcpyDeviceToHost));

        // Copy LiDAR B points from GPU to host
        m_hostPointsBSize = m_rigTransformedPoints[LIDAR_B_INDEX].size;
        CHECK_CUDA_ERROR(cudaMemcpy(m_hostPointsB.get(),
                                    m_rigTransformedPoints[LIDAR_B_INDEX].points,
                                    sizeof(dwVector4f) * m_hostPointsBSize,
                                    cudaMemcpyDeviceToHost));

        // Apply AVERAGE ICP transformation to LiDAR B points (for stable visualization)
        if (m_successfulICPCount > 0)
        {
            // Calculate average transformation
            float avgTx = m_totalTranslation.x / m_successfulICPCount;
            float avgTy = m_totalTranslation.y / m_successfulICPCount;
            float avgTz = m_totalTranslation.z / m_successfulICPCount;
            
            float avgRoll = m_totalRotation.x / m_successfulICPCount;
            float avgPitch = m_totalRotation.y / m_successfulICPCount;
            float avgYaw = m_totalRotation.z / m_successfulICPCount;
            
            // Build average transformation matrix from roll-pitch-yaw
            float cr = cos(avgRoll), sr = sin(avgRoll);
            float cp = cos(avgPitch), sp = sin(avgPitch);
            float cy = cos(avgYaw), sy = sin(avgYaw);
            
            // ZYX rotation order (yaw-pitch-roll)
            float r00 = cy * cp;
            float r01 = cy * sp * sr - sy * cr;
            float r02 = cy * sp * cr + sy * sr;
            float r10 = sy * cp;
            float r11 = sy * sp * sr + cy * cr;
            float r12 = sy * sp * cr - cy * sr;
            float r20 = -sp;
            float r21 = cp * sr;
            float r22 = cp * cr;
            
            // Apply average transformation to LiDAR B points
            for (uint32_t i = 0; i < m_hostPointsBSize; ++i)
            {
                float x = m_hostPointsB[i].x;
                float y = m_hostPointsB[i].y;
                float z = m_hostPointsB[i].z;
                
                // Apply average ICP transformation
                float tx = r00 * x + r01 * y + r02 * z + avgTx;
                float ty = r10 * x + r11 * y + r12 * z + avgTy;
                float tz = r20 * x + r21 * y + r22 * z + avgTz;
                
                m_hostPointsB[i].x = tx;
                m_hostPointsB[i].y = ty;
                m_hostPointsB[i].z = tz;
                // Keep intensity (w) unchanged
            }
        }

        // Update render buffers (like lidar_replay does)
        CHECK_DW_ERROR(dwRenderEngine_setBuffer(m_lidarABufferId,
                                               DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                               m_hostPointsA.get(),
                                               sizeof(dwVector4f),
                                               0,
                                               m_hostPointsASize,
                                               m_renderEngine));

        CHECK_DW_ERROR(dwRenderEngine_setBuffer(m_lidarBBufferId,
                                               DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                               m_hostPointsB.get(),
                                               sizeof(dwVector4f),
                                               0,
                                               m_hostPointsBSize,
                                               m_renderEngine));
    }
    
    void renderAlignedPointClouds()
    {
        // Just render the buffers that were updated in onProcess
        if (m_hostPointsASize == 0 || m_hostPointsBSize == 0)
        {
            return;
        }

        // Render LiDAR A (green)
        dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_GREEN, m_renderEngine);
        dwRenderEngine_setPointSize(2.0f, m_renderEngine);
        dwRenderEngine_renderBuffer(m_lidarABufferId, m_hostPointsASize, m_renderEngine);

        // Render LiDAR B (red, already transformed by ICP)
        dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_RED, m_renderEngine);
        dwRenderEngine_setPointSize(2.0f, m_renderEngine);
        dwRenderEngine_renderBuffer(m_lidarBBufferId, m_hostPointsBSize, m_renderEngine);
    }
    
    void renderStatistics()
    {
        dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
        dwRenderEngine_setCoordinateRange2D({static_cast<float>(getWindowWidth()),
                                             static_cast<float>(getWindowHeight())}, m_renderEngine);
        dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_renderEngine);
        dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_WHITE, m_renderEngine);
        
        // Title
        std::string title = "Lidar Calibration - Frame " + std::to_string(m_frameNum) + "/" + std::to_string(m_numFrames);
        dwRenderEngine_renderText2D(title.c_str(), {20.f, getWindowHeight() - 30.f}, m_renderEngine);
        
        // Success rate
        float successRate = (m_frameNum > 0) ? (100.0f * m_successfulICPCount / m_frameNum) : 0.0f;
        std::string success = "ICP Success: " + std::to_string(m_successfulICPCount) + "/" + 
                             std::to_string(m_frameNum) + " (" + 
                             std::to_string(static_cast<int>(successRate)) + "%)";
        dwRenderEngine_renderText2D(success.c_str(), {20.f, getWindowHeight() - 55.f}, m_renderEngine);
        
        // Current average correction
        if (m_successfulICPCount > 0)
        {
            float avgTx = m_totalTranslation.x / m_successfulICPCount;
            float avgTy = m_totalTranslation.y / m_successfulICPCount;
            float avgTz = m_totalTranslation.z / m_successfulICPCount;
            
            float avgRoll = RAD2DEG(m_totalRotation.x / m_successfulICPCount);
            float avgPitch = RAD2DEG(m_totalRotation.y / m_successfulICPCount);
            float avgYaw = RAD2DEG(m_totalRotation.z / m_successfulICPCount);
            
            char transBuf[128];
            snprintf(transBuf, sizeof(transBuf), "Avg Translation: [%.3f, %.3f, %.3f] m", avgTx, avgTy, avgTz);
            dwRenderEngine_renderText2D(transBuf, {20.f, getWindowHeight() - 80.f}, m_renderEngine);
            
            char rotBuf[128];
            snprintf(rotBuf, sizeof(rotBuf), "Avg Rotation: [%.2f, %.2f, %.2f] deg", avgRoll, avgPitch, avgYaw);
            dwRenderEngine_renderText2D(rotBuf, {20.f, getWindowHeight() - 105.f}, m_renderEngine);
        }
        
        // Legend
        dwRenderEngine_setColor({0.0f, 1.0f, 0.0f, 1.0f}, m_renderEngine);
        dwRenderEngine_renderText2D("Green: Front Lidar (Reference)", {20.f, 50.f}, m_renderEngine);
        
        dwRenderEngine_setColor({1.0f, 0.0f, 0.0f, 1.0f}, m_renderEngine);
        dwRenderEngine_renderText2D("Red: Rear Lidar (Aligned with ICP)", {20.f, 25.f}, m_renderEngine);
        
        // FPS
        renderutils::renderFPS(m_renderEngine, getCurrentFPS());
    }
};

//------------------------------------------------------------------------------
int main(int argc, const char** argv)
{
    ProgramArguments args(argc, argv,
                          {
                              ProgramArguments::Option_t("rig", ""),
                              ProgramArguments::Option_t("num-frames", "100"),
                              ProgramArguments::Option_t("max-iters", "50"),
                              ProgramArguments::Option_t("verbose", "false")
                          },
                          "Lidar Calibration Tool - Compute ICP corrections for rig file");

    if (std::string(args.get("rig")).empty())
    {
        std::cerr << "ERROR: --rig parameter required!" << std::endl;
        std::cerr << "Usage: " << argv[0] << " --rig=<rig.json> [--num-frames=100] [--max-iters=50]" << std::endl;
        return -1;
    }

    LidarCalibration app(args);
    
    // Initialize window for visualization
    app.initializeWindow("Lidar Calibration Tool", 1280, 720, false);
    
    return app.run();
}
