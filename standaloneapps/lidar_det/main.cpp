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

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <ratio>
#include <string>

#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/lidar/Lidar.h>

#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Visualization.h>

#include <framework/Checks.hpp>
#include <framework/CudaTimer.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/Log.hpp>
#include <framework/MouseView3D.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/RenderUtils.hpp>
#include <framework/SamplesDataPath.hpp>

// PointPillars integration
#include "./include/pointpillar.hpp"

using namespace dw_samples::common;

//------------------------------------------------------------------------------
// Lidar replay sample with PointPillars 3D object detection integration
//------------------------------------------------------------------------------
class LidarPointPillarsSample : public DriveWorksSample
{
private:
    std::string m_outputDir = "";
    std::string m_modelPath = "";

    dwContextHandle_t m_context = DW_NULL_HANDLE;
    dwSALHandle_t m_sal         = DW_NULL_HANDLE;

    dwSensorHandle_t m_lidarSensor = DW_NULL_HANDLE;
    dwLidarProperties m_lidarProperties{};
    bool m_recordedLidar = false;
    std::unique_ptr<float32_t[]> m_pointCloud;

    // PointPillars inference engine
    std::unique_ptr<PointPillar> m_pointPillarsEngine = nullptr;
    std::vector<Bndbox> m_detections;
    dw_samples::common::CudaTimer m_inferenceTimer;
    float32_t m_inferenceTimeMs = 0.0f;
    uint32_t m_frameCount = 0;

    // Rendering
    dwVisualizationContextHandle_t m_visualizationContext = DW_NULL_HANDLE;

    dwRenderEngineColorByValueMode m_colorByValueMode = DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_XY;

    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
    uint32_t m_gridBuffer                 = 0;
    uint32_t m_gridBufferPrimitiveCount   = 0;
    uint32_t m_pointCloudBuffer           = 0;
    uint32_t m_pointCloudBufferCapacity   = 0;
    uint32_t m_pointCloudBufferSize       = 0;

    // Detection visualization buffers
    uint32_t m_detectionBoxBuffer         = 0;
    uint32_t m_detectionBoxBufferCapacity = 0;
    uint32_t m_detectionBoxBufferSize     = 0;

    std::string m_message1;
    std::string m_message2;
    std::string m_message3;
    std::string m_message4;
    std::string m_message5;
    std::string m_message6;
    std::string m_message7;
    std::string m_message8; // For inference info

public:
    LidarPointPillarsSample(const ProgramArguments& args)
        : DriveWorksSample(args) {}

    void initializeDriveWorks(dwContextHandle_t& context)
    {
        // initialize logger to print verbose message on console in color
        CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
        CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));

        // initialize SDK context, using data folder
        dwContextParameters sdkParams = {};

#ifdef VIBRANTE
        sdkParams.eglDisplay = getEGLDisplay();
#endif

        CHECK_DW_ERROR(dwInitialize(&context, DW_VERSION, &sdkParams));

        CHECK_DW_ERROR(dwSAL_initialize(&m_sal, context));

        std::string parameterString;
        std::string protocolString;
        dwSensorParams params{};
        if (strcmp(getArgument("protocol").c_str(), "") != 0)
        {
            protocolString  = getArgument("protocol");
            m_recordedLidar = (protocolString == "lidar.virtual") ? true : false;
        }

        if (strcmp(getArgument("params").c_str(), "") != 0)
            parameterString = getArgument("params");

        std::string showIntensity = getArgument("show-intensity");
        std::transform(showIntensity.begin(), showIntensity.end(), showIntensity.begin(), ::tolower);
        if (showIntensity.compare("true") == 0)
            m_colorByValueMode = DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_INTENSITY;

        if (protocolString.empty() || parameterString.empty())
        {
            logError("INVALID PARAMETERS\n");
            exit(-1);
        }

        params.protocol   = protocolString.c_str();
        params.parameters = parameterString.c_str();
        CHECK_DW_ERROR(dwSAL_createSensor(&m_lidarSensor, params, m_sal));
        
        // Get lidar properties
        CHECK_DW_ERROR(dwSensorLidar_getProperties(&m_lidarProperties, m_lidarSensor));

        // Allocate bigger buffer in case certain spin exceeds the pointsPerSpin in lidar property
        m_pointCloudBufferCapacity = m_lidarProperties.pointsPerSecond;
        m_pointCloud.reset(new float32_t[m_pointCloudBufferCapacity * m_lidarProperties.pointStride]);
    }

    void initializePointPillars()
    {
        m_modelPath = getArgument("pointpillars-model");
        
        if (m_modelPath.empty())
        {
            log("PointPillars model not specified, running without object detection\n");
            return;
        }

        try
        {
            log("Initializing PointPillars with model: %s\n", m_modelPath.c_str());
            m_pointPillarsEngine = std::make_unique<PointPillar>(m_modelPath, 0);
            log("PointPillars initialized successfully\n");
        }
        catch (const std::exception& e)
        {
            logError("Failed to initialize PointPillars: %s\n", e.what());
            m_pointPillarsEngine.reset();
        }
    }

    void preparePointCloudDumps()
    {
        m_outputDir = getArgument("output-dir");

        if (!m_outputDir.empty())
        {
            auto dirExist = [](const std::string& dir) -> bool {
                struct stat buffer;
                return (stat(dir.c_str(), &buffer) == 0);
            };

            if (dirExist(m_outputDir))
            {
                rmdir(m_outputDir.c_str());
            }

            mkdir(m_outputDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

            if (chdir(m_outputDir.c_str()))
            {
                logError("Unable to change to output directory: %s\n", m_outputDir.c_str());
                exit(-1);
            }
        }
    }

    /// -----------------------------
    /// Initialize everything of a sample here incl. SDK components
    /// -----------------------------
    bool onInitialize() override
    {
        log("Starting PointPillars Lidar Detection Sample...\n");

        preparePointCloudDumps();
        initializeDriveWorks(m_context);
        initializePointPillars();

        CHECK_DW_ERROR(dwVisualizationInitialize(&m_visualizationContext, m_context));

        // -----------------------------
        // Initialize RenderEngine
        // -----------------------------
        dwRenderEngineParams renderEngineParams{};
        CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&renderEngineParams,
                                                        getWindowWidth(),
                                                        getWindowHeight()));
        renderEngineParams.defaultTile.backgroundColor = {0.0f, 0.0f, 0.0f, 1.0f};
        CHECK_DW_ERROR_MSG(dwRenderEngine_initialize(&m_renderEngine, &renderEngineParams, m_visualizationContext),
                           "Cannot initialize Render Engine, maybe no GL context available?");

        // Point cloud buffer
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_pointCloudBuffer, DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                   sizeof(dwVector3f), 0, m_pointCloudBufferCapacity, m_renderEngine));

        // Grid buffer
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_gridBuffer, DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                                   sizeof(dwVector3f), 0, 10000, m_renderEngine));

        dwMatrix4f identity = DW_IDENTITY_MATRIX4F;
        CHECK_DW_ERROR(dwRenderEngine_setBufferPlanarGrid3D(m_gridBuffer, {0.f, 0.f, 100.f, 100.f},
                                                            5.0f, 5.0f,
                                                            &identity, m_renderEngine));

        CHECK_DW_ERROR(dwRenderEngine_getBufferMaxPrimitiveCount(&m_gridBufferPrimitiveCount, m_gridBuffer, m_renderEngine));

        // Detection visualization buffer
        if (m_pointPillarsEngine)
        {
            m_detectionBoxBufferCapacity = 1000; // Max lines for bounding boxes
            CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_detectionBoxBuffer, DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                                       sizeof(dwVector3f), 0, m_detectionBoxBufferCapacity, m_renderEngine));
        }

        CHECK_DW_ERROR(dwSensor_start(m_lidarSensor));

        return true;
    }

    ///------------------------------------------------------------------------------
    /// This method is executed when user presses `R`, it indicates that sample has to reset
    ///------------------------------------------------------------------------------
    void onReset() override
    {
        CHECK_DW_ERROR(dwSensor_reset(m_lidarSensor));
        CHECK_DW_ERROR(dwRenderEngine_reset(m_renderEngine));
        
        m_frameCount = 0;
        m_inferenceTimeMs = 0.0f;
        m_detections.clear();
    }

    ///------------------------------------------------------------------------------
    /// This method is executed on release, free up used memory here
    ///------------------------------------------------------------------------------
    void onRelease() override
    {
        // Release PointPillars
        m_pointPillarsEngine.reset();

        // Release render buffers
        if (m_pointCloudBuffer != 0)
        {
            CHECK_DW_ERROR(dwRenderEngine_destroyBuffer(m_pointCloudBuffer, m_renderEngine));
        }
        if (m_gridBuffer != 0)
        {
            CHECK_DW_ERROR(dwRenderEngine_destroyBuffer(m_gridBuffer, m_renderEngine));
        }
        if (m_detectionBoxBuffer != 0)
        {
            CHECK_DW_ERROR(dwRenderEngine_destroyBuffer(m_detectionBoxBuffer, m_renderEngine));
        }

        if (m_renderEngine != DW_NULL_HANDLE)
        {
            CHECK_DW_ERROR(dwRenderEngine_release(m_renderEngine));
        }

        if (m_visualizationContext != DW_NULL_HANDLE)
        {
            CHECK_DW_ERROR(dwVisualizationRelease(m_visualizationContext));
        }

        if (m_lidarSensor != DW_NULL_HANDLE)
        {
            CHECK_DW_ERROR(dwSAL_releaseSensor(m_lidarSensor));
        }

        // Release DriveWorks context and SAL
        CHECK_DW_ERROR(dwSAL_release(m_sal));

        if (m_context != DW_NULL_HANDLE)
        {
            CHECK_DW_ERROR(dwRelease(m_context));
        }

        CHECK_DW_ERROR(dwLogger_release());
    }

    ///------------------------------------------------------------------------------
    /// Change renderer properties when main rendering window is resized
    ///------------------------------------------------------------------------------
    void onResizeWindow(int width, int height) override
    {
        dwRectf bounds{.x = 0.0f, .y = 0.0f};
        bounds.width  = width;
        bounds.height = height;
        CHECK_DW_ERROR(dwRenderEngine_setBounds(bounds, m_renderEngine));
    }

    void runPointPillarsInference(uint32_t accumulatedPoints)
    {
        if (!m_pointPillarsEngine || accumulatedPoints == 0)
            return;

        m_inferenceTimer.start();

        // Convert DriveWorks point cloud format to PointPillars input format
        std::vector<float> inferencePoints;
        convertPointCloudForInference(accumulatedPoints, inferencePoints);

        // Run PointPillars inference
        m_detections.clear();
        try
        {
            m_pointPillarsEngine->doinfer(inferencePoints.data(), 
                                        inferencePoints.size() / 4, 
                                        m_detections);
        }
        catch (const std::exception& e)
        {
            logError("PointPillars inference failed: %s\n", e.what());
            return;
        }

        m_inferenceTimer.stop();
        m_inferenceTimeMs = m_inferenceTimer.getElapsedTime();

        // Update detection visualization
        updateDetectionVisualization();
    }

    void convertPointCloudForInference(uint32_t pointCount, std::vector<float>& inferencePoints)
    {
        inferencePoints.clear();
        inferencePoints.reserve(pointCount * 4);

        const dwLidarPointXYZI* dwPoints = reinterpret_cast<const dwLidarPointXYZI*>(m_pointCloud.get());

        for (uint32_t i = 0; i < pointCount; i++)
        {
            inferencePoints.push_back(dwPoints[i].x);
            inferencePoints.push_back(dwPoints[i].y);
            inferencePoints.push_back(dwPoints[i].z);
            inferencePoints.push_back(static_cast<float>(dwPoints[i].intensity) / 255.0f);
        }
    }

    void updateDetectionVisualization()
    {
        if (!m_pointPillarsEngine || m_detections.empty())
        {
            m_detectionBoxBufferSize = 0;
            return;
        }

        std::vector<dwVector3f> boxLines;
        boxLines.reserve(m_detections.size() * 24); // 12 lines per box * 2 points per line

        for (const auto& detection : m_detections)
        {
            createBoundingBoxLines(detection, boxLines);
        }

        m_detectionBoxBufferSize = static_cast<uint32_t>(boxLines.size());
        
        if (m_detectionBoxBufferSize > 0 && m_detectionBoxBufferSize <= m_detectionBoxBufferCapacity)
        {
            CHECK_DW_ERROR(dwRenderEngine_setBuffer(m_detectionBoxBuffer,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                                   boxLines.data(),
                                                   sizeof(dwVector3f),
                                                   0,
                                                   m_detectionBoxBufferSize,
                                                   m_renderEngine));
        }
    }

    void createBoundingBoxLines(const Bndbox& detection, std::vector<dwVector3f>& lines)
    {
        // Calculate 8 corners of the 3D bounding box
        float32_t cx = detection.x, cy = detection.y, cz = detection.z;
        float32_t w = detection.w, l = detection.l, h = detection.h;
        float32_t yaw = detection.rt;

        float32_t cos_yaw = std::cos(yaw);
        float32_t sin_yaw = std::sin(yaw);

        // 8 corners in local coordinates
        std::array<std::array<float32_t, 3>, 8> localCorners = {{
            {{ w/2,  l/2, -h/2}}, {{-w/2,  l/2, -h/2}},
            {{-w/2, -l/2, -h/2}}, {{ w/2, -l/2, -h/2}},
            {{ w/2,  l/2,  h/2}}, {{-w/2,  l/2,  h/2}},
            {{-w/2, -l/2,  h/2}}, {{ w/2, -l/2,  h/2}}
        }};

        // Transform to global coordinates
        std::array<dwVector3f, 8> corners;
        for (int i = 0; i < 8; i++)
        {
            float32_t lx = localCorners[i][0];
            float32_t ly = localCorners[i][1];
            float32_t lz = localCorners[i][2];

            corners[i].x = cx + lx * cos_yaw - ly * sin_yaw;
            corners[i].y = cy + lx * sin_yaw + ly * cos_yaw;
            corners[i].z = cz + lz;
        }

        // Bottom face (4 lines)
        for (int i = 0; i < 4; i++)
        {
            lines.push_back(corners[i]);
            lines.push_back(corners[(i + 1) % 4]);
        }

        // Top face (4 lines)
        for (int i = 4; i < 8; i++)
        {
            lines.push_back(corners[i]);
            lines.push_back(corners[4 + ((i - 4 + 1) % 4)]);
        }

        // Vertical lines (4 lines)
        for (int i = 0; i < 4; i++)
        {
            lines.push_back(corners[i]);
            lines.push_back(corners[i + 4]);
        }
    }

    void updateFrame(uint32_t accumulatedPoints, uint32_t packetCount,
                     dwTime_t hostTimestamp, dwTime_t sensorTimestamp)
    {
        m_pointCloudBufferSize = accumulatedPoints;
        m_frameCount++;

        // Get updated properties
        CHECK_DW_ERROR(dwSensorLidar_getProperties(&m_lidarProperties, m_lidarSensor));

        // Dump lidar frame if requested
        if (!m_outputDir.empty())
        {
            dumpLidarFrame(accumulatedPoints, hostTimestamp);
        }

        // Run PointPillars inference
        runPointPillarsInference(accumulatedPoints);

        // Update point cloud buffer for rendering
        CHECK_DW_ERROR(dwRenderEngine_setBuffer(m_pointCloudBuffer,
                                               DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                               m_pointCloud.get(),
                                               sizeof(dwLidarPointXYZI),
                                               0,
                                               m_pointCloudBufferSize,
                                               m_renderEngine));

        // Update UI messages
        updateUIMessages(accumulatedPoints, packetCount, hostTimestamp, sensorTimestamp);
    }

    void updateUIMessages(uint32_t accumulatedPoints, uint32_t packetCount,
                         dwTime_t hostTimestamp, dwTime_t sensorTimestamp)
    {
        m_message1 = "Host timestamp    (us) " + std::to_string(hostTimestamp);
        m_message2 = "Sensor timestamp (us) " + std::to_string(sensorTimestamp);
        m_message3 = "Packets per scan         " + std::to_string(packetCount);
        m_message4 = "Points per scan           " + std::to_string(accumulatedPoints);
        m_message5 = "Frequency (Hz)           " + std::to_string(m_lidarProperties.spinFrequency);
        m_message6 = "Lidar Device               " + std::string{m_lidarProperties.deviceString};
        m_message7 = "Press ESC to exit";
        
        if (m_pointPillarsEngine)
        {
            m_message8 = "PointPillars: " + std::to_string(m_detections.size()) + 
                        " detections (" + std::to_string(static_cast<int>(m_inferenceTimeMs)) + "ms)";
        }
        else
        {
            m_message8 = "PointPillars: Not initialized";
        }
    }

    void dumpLidarFrame(uint32_t accumulatedPoints, dwTime_t timestamp)
    {
        const std::string lidarFilename = std::to_string(timestamp) + ".bin";
        std::ofstream fout;
        fout.open(lidarFilename, std::ios::binary | std::ios::out);
        fout.write(reinterpret_cast<char*>(m_pointCloud.get()), 
                   accumulatedPoints * m_lidarProperties.pointStride * sizeof(float32_t));
        fout.close();
    }

    void computeSpin()
    {
        const dwLidarDecodedPacket* nextPacket;
        static uint32_t packetCount       = 0;
        static uint32_t accumulatedPoints = 0;
        static bool endOfSpin             = false;
        static auto tStart                = std::chrono::high_resolution_clock::now();
        static auto tEnd                  = tStart;

        // For recorded data throttling
        if (m_recordedLidar && endOfSpin)
        {
            tEnd                = std::chrono::high_resolution_clock::now();
            float64_t duration  = std::chrono::duration<float64_t, std::milli>(tEnd - tStart).count();
            float64_t sleepTime = 1000.0 / m_lidarProperties.spinFrequency - duration;

            if (sleepTime > 0.0)
                return;
            else
                endOfSpin = false;
        }

        dwStatus status = DW_NOT_AVAILABLE;
        dwTime_t hostTimestamp   = 0;
        dwTime_t sensorTimestamp = 0;

        tStart = std::chrono::high_resolution_clock::now();
        while (1)
        {
            status = dwSensorLidar_readPacket(&nextPacket, 100000, m_lidarSensor);
            if (status == DW_SUCCESS)
            {
                packetCount++;
                hostTimestamp   = nextPacket->hostTimestamp;
                sensorTimestamp = nextPacket->sensorTimestamp;

                // Append the packet to the buffer
                float32_t* map = &m_pointCloud[accumulatedPoints * m_lidarProperties.pointStride];
                memcpy(map, nextPacket->pointsXYZI, nextPacket->nPoints * sizeof(dwLidarPointXYZI));

                accumulatedPoints += nextPacket->nPoints;

                // If we go beyond a full spin, update the render data then return
                if (nextPacket->scanComplete)
                {
                    updateFrame(accumulatedPoints, packetCount, hostTimestamp, sensorTimestamp);

                    accumulatedPoints = 0;
                    packetCount       = 0;
                    endOfSpin         = true;
                    CHECK_DW_ERROR(dwSensorLidar_returnPacket(nextPacket, m_lidarSensor));
                    return;
                }

                CHECK_DW_ERROR(dwSensorLidar_returnPacket(nextPacket, m_lidarSensor));
            }
            else if (status == DW_END_OF_STREAM)
            {
                updateFrame(accumulatedPoints, packetCount, hostTimestamp, sensorTimestamp);

                // For recorded data, start over at the end of the file
                CHECK_DW_ERROR(dwSensor_reset(m_lidarSensor));
                accumulatedPoints = 0;
                packetCount       = 0;
                endOfSpin         = true;

                return;
            }
            else if (status == DW_TIME_OUT)
            {
                std::cout << "Read lidar packet: timeout" << std::endl;
            }
            else
            {
                stop();
                return;
            }
        }
    }

    void onProcess() override
    {
        computeSpin();
    }

    void onRender() override
    {
        CHECK_DW_ERROR(dwRenderEngine_reset(m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_setTile(0, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_setModelView(getMouseView().getModelView(), m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine));

        // Render background and grid
        CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_DARKGREY, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_setBackgroundColor({0.0f, 0.0f, 0.0f, 1.0f}, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_renderBuffer(m_gridBuffer, m_gridBufferPrimitiveCount, m_renderEngine));

        // Render point cloud
        CHECK_DW_ERROR(dwRenderEngine_setColorByValue(m_colorByValueMode, 130.0f, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_renderBuffer(m_pointCloudBuffer, m_pointCloudBufferSize, m_renderEngine));

        // Render detection bounding boxes
        if (m_detectionBoxBufferSize > 0)
        {
            CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_RED, m_renderEngine));
            CHECK_DW_ERROR(dwRenderEngine_renderBuffer(m_detectionBoxBuffer, m_detectionBoxBufferSize, m_renderEngine));
        }

        // Render UI text
        CHECK_DW_ERROR(dwRenderEngine_setProjection(&DW_IDENTITY_MATRIX4F, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine));
        dwVector2f range{static_cast<float32_t>(getWindowWidth()),
                         static_cast<float32_t>(getWindowHeight())};
        CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_GREEN, m_renderEngine));
        
        CHECK_DW_ERROR(dwRenderEngine_renderText2D(m_message1.c_str(), {20.f, getWindowHeight() - 30.f}, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_renderText2D(m_message2.c_str(), {20.f, getWindowHeight() - 50.f}, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_renderText2D(m_message3.c_str(), {20.f, getWindowHeight() - 70.f}, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_renderText2D(m_message4.c_str(), {20.f, getWindowHeight() - 90.f}, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_renderText2D(m_message5.c_str(), {20.f, getWindowHeight() - 110.f}, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_renderText2D(m_message6.c_str(), {20.f, getWindowHeight() - 130.f}, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_renderText2D(m_message7.c_str(), {20.f, 20.f}, m_renderEngine));
        
        // PointPillars inference info
        CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_YELLOW, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_renderText2D(m_message8.c_str(), {20.f, getWindowHeight() - 150.f}, m_renderEngine));

        renderutils::renderFPS(m_renderEngine, getCurrentFPS());
    }
};

//------------------------------------------------------------------------------
int main(int argc, const char** argv)
{
    ProgramArguments args(argc, argv,
                          {
                              ProgramArguments::Option_t("protocol", "lidar.virtual"),
                              ProgramArguments::Option_t("params", ("file=" + dw_samples::SamplesDataPath::get() + "/samples/sensors/lidar/sample.bin").c_str()),
                              ProgramArguments::Option_t("output-dir", ""),
                              ProgramArguments::Option_t("show-intensity", "false"),
                              ProgramArguments::Option_t("pointpillars-model", "/usr/local/driveworks/data/models/pointpillar.plan")
                          });

    LidarPointPillarsSample app(args);

    app.initializeWindow("Lidar PointPillars Detection Sample", 1024, 800, args.enabled("offscreen"));

    return app.run();
}