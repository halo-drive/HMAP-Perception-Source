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
// SPDX-FileCopyrightText: Copyright (c) 2016-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <driver_types.h>
#include <cuda_runtime.h>

#include <dw/control/vehicleio/VehicleIO.h>
#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/egomotion/base/Egomotion.h>
#include <dw/image/Image.h>
#include <dw/imageprocessing/featuredetector/FeatureDetector.h>
#include <dw/imageprocessing/features/FeatureList.h>
#include <dw/imageprocessing/pyramid/Pyramid.h>
#include <dw/imageprocessing/sfm/SFM.h>
#include <dw/imageprocessing/tracking/featuretracker/FeatureTracker.h>
#include <dw/interop/streamer/ImageStreamer.h>
#include <dw/rig/Rig.h>
#include <dw/rig/Vehicle.h>
#include <dw/sensors/camera/Camera.h>
#include <dw/sensors/sensormanager/SensorManager.h>
#include <dw/sensors/common/Sensors.h>

#include <dwvisualization/core/Renderer.h>
#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Visualization.h>
#include <dwvisualization/gl/GL.h>
#include <dwvisualization/image/Image.h>
#include <dwvisualization/interop/ImageStreamer.h>

#include <framework/Checks.hpp>
#include <framework/ChecksExt.hpp>
#include <framework/CudaTimer.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/Log.hpp>
#include <framework/Mat4.hpp>
#include <framework/MouseView3D.hpp>
#include <framework/ProfilerCUDA.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/RenderUtils.hpp>
#include <framework/SamplesDataPath.hpp>
#include <framework/WindowGLFW.hpp>

using namespace dw_samples::common;

//------------------------------------------------------------------------------
// SfM sample
// Demonstrates the usage of feature tracking, 3d triangulation, and car pose estimation
// using four cameras.
//------------------------------------------------------------------------------
class SfmSample : public DriveWorksSample
{
private:
    static constexpr size_t CAMERA_COUNT = 4;

    // ------------------------------------------------
    // SAL
    // ------------------------------------------------
    dwContextHandle_t m_context           = DW_NULL_HANDLE;
    dwVisualizationContextHandle_t m_viz  = DW_NULL_HANDLE;
    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
    dwSALHandle_t sal                     = DW_NULL_HANDLE;

    dwSensorManagerHandle_t sensorManager = DW_NULL_HANDLE;
    dwRigHandle_t rigConfig               = DW_NULL_HANDLE;

    // ------------------------------------------------
    // Sample specific variables
    // ------------------------------------------------

    // Egomotion
    dwVehicle const* vehicle;
    dwEgomotionHandle_t egomotion           = DW_NULL_HANDLE;
    dwVehicleIOHandle_t vehicleIO           = DW_NULL_HANDLE;
    dwTime_t lastEgoUpdate                  = 0;
    dwTransformation3f egoRig2World         = DW_IDENTITY_TRANSFORMATION3F;
    dwTransformation3f previousEgoRig2World = DW_IDENTITY_TRANSFORMATION3F;

    // Feature tracker
    uint32_t maxFeatureCount          = 2000;
    const size_t FEATURE_HISTORY_SIZE = 60;

    // Reconstructor
    dwReconstructorHandle_t reconstructor = DW_NULL_HANDLE;

    // Per-camera data
    struct CameraData
    {
        // Sensor
        dwSensorHandle_t camera            = DW_NULL_HANDLE;
        dwImageStreamerHandle_t image2GL   = DW_NULL_HANDLE;
        dwCameraProperties cameraProps     = {};
        dwImageProperties cameraImageProps = {};
        dwCameraFrameHandle_t frame        = DW_NULL_HANDLE;

        // Image
        dwImageHandle_t frameCudaRgba = DW_NULL_HANDLE; // Only used if the camera produces CUDA images

        dwImageHandle_t frameCudaYuv = DW_NULL_HANDLE; // This is the image used for processing. Could be same as frameCameraCudaYuv or coming from nvm2cuda streamer

        dwImageHandle_t frameGL = DW_NULL_HANDLE;

        // Features
        dwPyramidImage pyramidCurrent  = {};
        dwPyramidImage pyramidPrevious = {};

        dwFeature2DDetectorHandle_t detector = DW_NULL_HANDLE;
        dwFeature2DTrackerHandle_t tracker   = DW_NULL_HANDLE;

        // These point into the buffers of g_featureList
        dwFeatureHistoryArray featureHistoryCPU = {};
        dwFeatureHistoryArray featureHistoryGPU = {};
        dwFeatureArray featureDetectedGPU       = {};

        // Predicted points
        dwVector2f* d_predictedFeatureLocations = {nullptr};
        std::unique_ptr<dwVector2f[]> predictedFeatureLocations;

        // Projected points
        std::unique_ptr<dwVector2f[]> projectedLocations;
        dwVector2f* d_projectedLocations = {nullptr};

        // Reconstruction data
        dwVector4f* d_worldPoints = nullptr;
        std::unique_ptr<dwVector4f[]> worldPoints;

        // Render
        uint32_t tileId;
    };
    std::array<CameraData, CAMERA_COUNT> cameras;

    // Arrays containing pointers for all cameras
    uint32_t* d_allFeatureCount[CAMERA_COUNT]             = {nullptr};
    dwFeature2DStatus* d_allFeatureStatuses[CAMERA_COUNT] = {nullptr};
    dwVector4f* d_allWorldPoints[CAMERA_COUNT]            = {nullptr};
    dwVector2f* d_allProjections[CAMERA_COUNT]            = {nullptr};

    dwTransformation3f currentRig2World = DW_IDENTITY_TRANSFORMATION3F;

    // Pose rendering
    dwVector4f poseColor = DW_RENDERER_COLOR_DARKGREEN;
    std::vector<dwVector3f> positionsCAN;
    std::vector<dwVector3f> positionsRefined;
    std::vector<dwVector3f> positionsDifference;

    // UI control
    bool doPoseRefinement    = true;
    bool doFeaturePrediction = true;
    uint32_t worldTileId     = 0;

    const uint32_t MAX_FRAME_COUNT = 2000;
    uint32_t maxRenderBufferCount;

public:
    SfmSample(const ProgramArguments& args)
        : DriveWorksSample(args) {}

    /// -----------------------------
    /// Initialize Logger and DriveWorks context
    /// -----------------------------
    void initializeDriveWorks(dwContextHandle_t& context) const
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
    }

    /// -----------------------------
    /// Initialize Renderer, Sensors, Image Streamers and Tracker
    /// -----------------------------
    bool onInitialize() override
    {
        // -----------------------------------------
        // Get values from command line
        // -----------------------------------------
        maxFeatureCount      = std::stoi(getArgument("maxFeatureCount"));
        maxRenderBufferCount = std::max(MAX_FRAME_COUNT, maxFeatureCount);

        uint32_t useHalf                = std::stoi(getArgument("useHalf"));
        uint32_t enableAdaptiveWindow   = std::stoi(getArgument("enableAdaptiveWindow"));
        float32_t displacementThreshold = std::stof(getArgument("displacementThreshold"));

        dwFeature2DTrackerAlgorithm trackMode =
            (getArgument("trackMode") == "0" ? DW_FEATURE2D_TRACKER_ALGORITHM_STD : DW_FEATURE2D_TRACKER_ALGORITHM_EX_FAST);
        // -----------------------------------------
        // Initialize DriveWorks context and SAL
        // -----------------------------------------
        {
            initializeDriveWorks(m_context);
            CHECK_DW_ERROR(dwSAL_initialize(&sal, m_context));
            CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_context));
        }

        // -----------------------------
        // initialize sensors
        // -----------------------------
        {

            // Need to change directories because the rig.json file contains
            // relative paths to the sensors
            int res;
            res = chdir(getArgument("baseDir").c_str());
            (void)res;
            CHECK_DW_ERROR(dwRig_initializeFromFile(&rigConfig, m_context, getArgument("rig").c_str()));

            // Sensor manager
            dwSensorManagerParams sensorManagerParams    = {};
            sensorManagerParams.singleVirtualCameraGroup = true; // we are expecting all cameras to be provided with a single SM event
            CHECK_DW_ERROR(dwSensorManager_initializeFromRigWithParams(&sensorManager, rigConfig, &sensorManagerParams, 10, sal));

            // Add can sensor

            for (size_t k = 0; k < CAMERA_COUNT; k++)
            {
                auto& camera = cameras[k];

                uint32_t cameraIndex{};
                CHECK_DW_ERROR(dwSensorManager_getSensorIndex(&cameraIndex, DW_SENSOR_CAMERA, k, sensorManager));
                CHECK_DW_ERROR(dwSensorManager_getSensorHandle(&camera.camera, cameraIndex, sensorManager));
                CHECK_DW_ERROR(dwSensorCamera_getSensorProperties(&camera.cameraProps, camera.camera));
                CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&camera.cameraImageProps, DW_CAMERA_OUTPUT_NATIVE_PROCESSED, camera.camera));

                // Pyramids
                CHECK_DW_ERROR(dwPyramid_create(&camera.pyramidCurrent, 3, camera.cameraImageProps.width,
                                                camera.cameraImageProps.height, DW_TYPE_UINT8, m_context));
                CHECK_DW_ERROR(dwPyramid_create(&camera.pyramidPrevious, 3, camera.cameraImageProps.width,
                                                camera.cameraImageProps.height, DW_TYPE_UINT8, m_context));

                // Feature list
                CHECK_DW_ERROR(dwFeatureHistoryArray_createNew(&camera.featureHistoryCPU, maxFeatureCount,
                                                               FEATURE_HISTORY_SIZE, DW_MEMORY_TYPE_CPU, nullptr, m_context));
                CHECK_DW_ERROR(dwFeatureHistoryArray_createNew(&camera.featureHistoryGPU, maxFeatureCount,
                                                               FEATURE_HISTORY_SIZE, DW_MEMORY_TYPE_CUDA, nullptr, m_context));
                CHECK_DW_ERROR(dwFeatureArray_createNew(&camera.featureDetectedGPU, maxFeatureCount,
                                                        DW_MEMORY_TYPE_CUDA, nullptr, m_context));

                // -----------------------------
                // Feature tracker
                // -----------------------------
                dwFeature2DDetectorConfig detectorConfig = {};
                dwFeature2DDetector_initDefaultParams(&detectorConfig);
                detectorConfig.type            = DW_FEATURE2D_DETECTOR_TYPE_STD;
                detectorConfig.imageWidth      = camera.cameraImageProps.width;
                detectorConfig.imageHeight     = camera.cameraImageProps.height;
                detectorConfig.maxFeatureCount = maxFeatureCount;
                CHECK_DW_ERROR(dwFeature2DDetector_initialize(&camera.detector, &detectorConfig, cudaStream_t(0), m_context));

                dwFeature2DTrackerConfig trackerConfig = {};
                dwFeature2DTracker_initDefaultParams(&trackerConfig);
                trackerConfig.algorithm                  = trackMode;
                trackerConfig.imageWidth                 = camera.cameraImageProps.width;
                trackerConfig.imageHeight                = camera.cameraImageProps.height;
                trackerConfig.maxFeatureCount            = maxFeatureCount;
                trackerConfig.historyCapacity            = FEATURE_HISTORY_SIZE;
                trackerConfig.pyramidLevelCount          = camera.pyramidCurrent.levelCount;
                trackerConfig.numLevelTranslationOnly    = trackerConfig.pyramidLevelCount - 1;
                trackerConfig.detectorType               = detectorConfig.type;
                trackerConfig.useHalf                    = useHalf;
                trackerConfig.enableAdaptiveWindowSizeLK = enableAdaptiveWindow;
                trackerConfig.displacementThreshold      = displacementThreshold;
                trackerConfig.enableSparseOutput         = 1;
                CHECK_DW_ERROR(dwFeature2DTracker_initialize(&camera.tracker, &trackerConfig, cudaStream_t(0), m_context));

                CHECK_CUDA_ERROR(cudaMalloc(&camera.d_predictedFeatureLocations, maxFeatureCount * sizeof(dwVector2f)));
                CHECK_CUDA_ERROR(cudaMalloc(&camera.d_worldPoints, maxFeatureCount * sizeof(dwVector4f)));
                CHECK_CUDA_ERROR(cudaMalloc(&camera.d_projectedLocations, maxFeatureCount * sizeof(dwVector2f)));
                camera.predictedFeatureLocations.reset(new dwVector2f[maxFeatureCount]);
                camera.worldPoints.reset(new dwVector4f[maxFeatureCount]);
                camera.projectedLocations.reset(new dwVector2f[maxFeatureCount]);

                CHECK_CUDA_ERROR(cudaMemset(camera.d_worldPoints, 0, maxFeatureCount * sizeof(dwVector4f)));

                d_allFeatureCount[k]    = camera.featureHistoryGPU.featureCount;
                d_allFeatureStatuses[k] = camera.featureHistoryGPU.statuses;
                d_allProjections[k]     = camera.d_projectedLocations;
                d_allWorldPoints[k]     = camera.d_worldPoints;

                // Streamer pipeline
                dwImageProperties displayImageProps{};
                CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&displayImageProps, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, camera.camera));

                // image2GL is either from NvMedia or CUDA, depending on camera
                CHECK_DW_ERROR(dwImageStreamerGL_initialize(&camera.image2GL,
                                                            &displayImageProps,
                                                            DW_IMAGE_GL,
                                                            m_context));
            }

            // we would like the application run as fast as the original video
            setProcessRate(cameras[0].cameraProps.framerate);
        }

        // -----------------------------
        // Egomotion
        // -----------------------------
        CHECK_DW_ERROR(dwRig_getVehicle(&vehicle, rigConfig));

        dwEgomotionParameters egoparams{};
        egoparams.vehicle         = *vehicle;
        egoparams.motionModel     = DW_EGOMOTION_ODOMETRY;
        egoparams.automaticUpdate = true;
        CHECK_DW_ERROR(dwEgomotion_initialize(&egomotion, &egoparams, m_context));

        //VehicleIO
        CHECK_DW_ERROR(dwVehicleIO_initializeFromRig(&vehicleIO, rigConfig, m_context));

        // -----------------------------
        // Reconstructor
        // -----------------------------
        dwReconstructorConfig reconstructorConfig;
        dwReconstructor_initConfig(&reconstructorConfig);
        reconstructorConfig.maxFeatureCount = maxFeatureCount;
        reconstructorConfig.rig             = rigConfig;
        // Log default config values to understand triangulation requirements
        log("Reconstructor config: maxFeatureCount=%u, minTriangulationEntries=%u, maxPoseHistoryLength=%u\n",
            reconstructorConfig.maxFeatureCount, 
            reconstructorConfig.minTriangulationEntries,
            reconstructorConfig.maxPoseHistoryLength);

        dwStatus status = dwReconstructor_initialize(&reconstructor, &reconstructorConfig, cudaStream_t(0), m_context);
        if (status != DW_SUCCESS) {
            logError("Reconstructor initialization failed with status: %d\n", status);
            CHECK_DW_ERROR(status);
        } else {
            log("Reconstructor initialized successfully with maxFeatureCount: %u\n", maxFeatureCount);
        }

        // -----------------------------
        // Initialize Renderer
        // -----------------------------
        {
            float32_t videoAspect     = static_cast<float32_t>(cameras[0].cameraImageProps.width) / cameras[0].cameraImageProps.height;
            float32_t videoTileHeight = getWindowHeight() / CAMERA_COUNT;
            float32_t videoTileWidth  = videoTileHeight * videoAspect;

            dwRenderEngineParams params;
            dwRenderEngine_initDefaultParams(&params, getWindowWidth(), getWindowHeight());
            // This should be the size of the positions because they grow as the program
            // continues to run.
            params.bufferSize = maxRenderBufferCount * sizeof(dwVector4f);

            // 1st tile (default) - 3d world rendering
            {
                dwRenderEngine_initTileState(&params.defaultTile);
                params.defaultTile.layout.positionType    = DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_LEFT;
                params.defaultTile.layout.positionLayout  = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_ABSOLUTE;
                params.defaultTile.layout.sizeLayout      = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_ABSOLUTE;
                params.defaultTile.layout.viewport.x      = videoTileWidth;
                params.defaultTile.layout.viewport.y      = 0;
                params.defaultTile.layout.viewport.width  = getWindowWidth() - videoTileWidth;
                params.defaultTile.layout.viewport.height = getWindowHeight();
            }

            CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_viz));

            // Video tiles
            for (size_t k = 0; k < CAMERA_COUNT; k++)
            {
                dwRenderEngineTileState tileParams = params.defaultTile;
                tileParams.projectionMatrix        = DW_IDENTITY_MATRIX4F;
                tileParams.modelViewMatrix         = DW_IDENTITY_MATRIX4F;
                tileParams.layout.positionLayout   = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_ABSOLUTE;
                tileParams.layout.sizeLayout       = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_ABSOLUTE;
                tileParams.layout.viewport         = {0.f,
                                              k * videoTileHeight,
                                              videoTileWidth,
                                              videoTileHeight};
                tileParams.layout.positionType = DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_LEFT;

                CHECK_DW_ERROR(dwRenderEngine_addTile(&cameras[k].tileId, &tileParams, m_renderEngine));
            }
        }

        // -----------------------------
        // Start Sensors
        // -----------------------------
        CHECK_DW_ERROR(dwSensorManager_start(sensorManager));

        return true;
    }

    ///------------------------------------------------------------------------------
    /// When user requested a reset we playback the video from beginning
    ///------------------------------------------------------------------------------
    void onReset() override
    {
        dwSensorManager_stop(sensorManager);
        dwSensorManager_reset(sensorManager);
        dwSensorManager_start(sensorManager);
        dwEgomotion_reset(egomotion);
        dwVehicleIO_reset(vehicleIO);
        for (auto& camera : cameras)
        {
            dwFeatureHistoryArray_reset(&camera.featureHistoryGPU, cudaStream_t(0));
            dwFeatureArray_reset(&camera.featureDetectedGPU, cudaStream_t(0));

            dwFeature2DDetector_reset(camera.detector);
            dwFeature2DTracker_reset(camera.tracker);
        }

        dwReconstructor_reset(reconstructor);

        positionsCAN.clear();
        positionsRefined.clear();
        positionsDifference.clear();
        currentRig2World = DW_IDENTITY_TRANSFORMATION3F;
    }

    ///------------------------------------------------------------------------------
    /// Release acquired memory
    ///------------------------------------------------------------------------------
    void onRelease() override
    {
        // stop sensor
        dwSensorManager_stop(sensorManager);

        dwSensorManager_release(sensorManager);
        dwRig_release(rigConfig);

        if (m_renderEngine != DW_NULL_HANDLE)
        {
            CHECK_DW_ERROR(dwRenderEngine_release(m_renderEngine));
        }

        dwEgomotion_release(egomotion);
        dwVehicleIO_release(vehicleIO);

        dwReconstructor_release(reconstructor);

        // Per-camera data
        for (auto& camera : cameras)
        {
            releaseFrame(camera);

            dwImageStreamerGL_release(camera.image2GL);
            dwPyramid_destroy(camera.pyramidCurrent);
            dwPyramid_destroy(camera.pyramidPrevious);

            dwFeature2DDetector_release(camera.detector);
            dwFeature2DTracker_release(camera.tracker);

            dwFeatureHistoryArray_destroy(camera.featureHistoryCPU);
            dwFeatureHistoryArray_destroy(camera.featureHistoryGPU);
            dwFeatureArray_destroy(camera.featureDetectedGPU);

            cudaFree(camera.d_predictedFeatureLocations);
            cudaFree(camera.d_projectedLocations);
            cudaFree(camera.d_worldPoints);
        };

        // -----------------------------------------
        // Release DriveWorks handles, context and SAL
        // -----------------------------------------
        dwSAL_release(sal);
        CHECK_DW_ERROR(dwVisualizationRelease(m_viz));
        CHECK_DW_ERROR(dwRelease(m_context));
        CHECK_DW_ERROR(dwLogger_release());
    }

    void onKeyDown(int key, int /* scancode*/, int /*mods*/) override
    {
        if (key == GLFW_KEY_Q)
        {
            if (worldTileId == 0)
                worldTileId = 4;
            else
                worldTileId--;
        }
        else if (key == GLFW_KEY_V)
        {
            doPoseRefinement = !doPoseRefinement;
        }
        else if (key == GLFW_KEY_F)
        {
            doFeaturePrediction = !doFeaturePrediction;
        }
    }
    void onMouseDown(int button, float x, float y, int mods) override
    {
        getMouseView().mouseDown(button, x, y);
        (void)mods;
    }
    void onMouseUp(int button, float x, float y, int mods) override
    {
        getMouseView().mouseUp(button, x, y);
        (void)mods;
    }
    void onMouseMove(float x, float y) override
    {
        getMouseView().mouseMove(x, y);
    }
    void onMouseWheel(float x, float y) override
    {
        getMouseView().mouseWheel(x, y);
    }

    ///------------------------------------------------------------------------------
    /// Change renderer properties when main rendering window is resized
    ///------------------------------------------------------------------------------
    void onResizeWindow(int width, int height) override
    {
        {
            //dwRenderEngine_reset(m_renderEngine);
            dwRectf rect;
            rect.width  = width;
            rect.height = height;
            rect.x      = 0;
            rect.y      = 0;
            dwRenderEngine_setBounds(rect, m_renderEngine);
        }

        float32_t videoAspect     = static_cast<float32_t>(cameras[0].cameraImageProps.width) / cameras[0].cameraImageProps.height;
        float32_t videoTileHeight = getWindowHeight() / CAMERA_COUNT;
        float32_t videoTileWidth  = videoTileHeight * videoAspect;

        dwRenderEngine_setTile(0, m_renderEngine);
        dwRenderEngine_setViewport({videoTileWidth, 0, static_cast<float32_t>(width - videoTileWidth), static_cast<float32_t>(height)}, m_renderEngine);

        for (size_t k = 0; k < CAMERA_COUNT; k++)
        {
            dwRenderEngine_setTile(k + 1, m_renderEngine);
            dwRenderEngine_setViewport({0, k * videoTileHeight, videoTileWidth, videoTileHeight}, m_renderEngine);
        }
    }

    void checkBufferSizeAgainst(size_t size)
    {
        if (size > maxRenderBufferCount)
        {
            std::cerr << "Cannot render point count over "
                      << maxRenderBufferCount
                      << ". Try increasing MAX_FRAME_COUNT or maxFeatureCount."
                      << std::endl;
            throw std::runtime_error("Requested render size greater than size of render buffer.");
        }
    }

    ///------------------------------------------------------------------------------
    /// Render the window
    ///------------------------------------------------------------------------------
    void onRender() override
    {
        static uint32_t renderCounter = 0;
        if (renderCounter % 30 == 0) {
            log("=== onRender START (frame %u) ===\n", renderCounter);
        }
        renderCounter++;
        
        if (isOffscreen())
            return;

        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        for (size_t tileId = 0; tileId < 5; tileId++)
        {
            if (renderCounter % 30 == 0 && tileId == 0) {
                log("Rendering tile %zu, worldTileId: %u\n", tileId, worldTileId);
            }
            dwRenderEngine_setTile(tileId, m_renderEngine);

            if (tileId == worldTileId) {
                if (renderCounter % 30 == 0) {
                    log("Rendering 3D world...\n");
                }
                render3D();
                if (renderCounter % 30 == 0) {
                    log("3D world rendered\n");
                }
            }
            else
            {
                size_t cameraIdx;
                if (tileId > worldTileId)
                    cameraIdx = tileId - worldTileId - 1;
                else
                    cameraIdx = tileId + 4 - worldTileId;

                if (renderCounter % 30 == 0) {
                    log("Rendering camera %zu...\n", cameraIdx);
                }
                renderCamera(cameras[cameraIdx]);
            }
        }

        renderutils::renderFPS(m_renderEngine, getCurrentFPS());
        CHECK_GL_ERROR();
        if (renderCounter % 30 == 0) {
            log("=== onRender END ===\n");
        }
    }

    void render3D()
    {
        log("render3D: Setting up view...\n");
        getMouseView().setWindowAspect(static_cast<float32_t>(getWindowWidth()) / getWindowHeight());

        dwMatrix4f pi;
        Mat4_IsoInv(pi.array, currentRig2World.array);

        dwMatrix4f t;
        Mat4_AxB(t.array, getMouseView().getModelView()->array, pi.array);

        dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine);
        dwRenderEngine_setModelView(&t, m_renderEngine);
        log("render3D: View setup complete\n");

        dwRenderEngine_setColor(DW_RENDERER_COLOR_DARKGREY, m_renderEngine);
        dwRenderEngine_renderPlanarGrid3D({-300, -300, 600, 600}, 5, 5, &DW_IDENTITY_MATRIX4F, m_renderEngine);

        checkBufferSizeAgainst(positionsCAN.size());
        checkBufferSizeAgainst(positionsRefined.size());
        checkBufferSizeAgainst(positionsDifference.size());

        // Render poses
        dwRenderEngine_setColor(DW_RENDERER_COLOR_DARKGREEN, m_renderEngine);
        dwRenderEngine_setLineWidth(1.0f, m_renderEngine);
        dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D, positionsDifference.data(), sizeof(dwVector3f), 0, positionsDifference.size() / 2, m_renderEngine);

        dwRenderEngine_setColor(DW_RENDERER_COLOR_DARKGREEN, m_renderEngine);
        dwRenderEngine_setPointSize(2.0f, m_renderEngine);
        dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D, positionsCAN.data(), sizeof(dwVector3f), 0, positionsCAN.size(), m_renderEngine);

        dwRenderEngine_setColor(poseColor, m_renderEngine);
        dwRenderEngine_setLineWidth(2.0f, m_renderEngine);
        dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINESTRIP_3D, positionsRefined.data(), sizeof(dwVector3f), 0, positionsRefined.size(), m_renderEngine);

        // Render points
        dwVector4f colors[4] = {DW_RENDERER_COLOR_RED,
                                DW_RENDERER_COLOR_GREEN,
                                DW_RENDERER_COLOR_BLUE,
                                DW_RENDERER_COLOR_YELLOW};
        dwRenderEngine_setPointSize(2.0f, m_renderEngine);
        for (size_t k = 0; k < CAMERA_COUNT; k++)
        {
            auto& camera = cameras[k];

            // Static buffers so allocation only happens for the first frame
            static std::vector<dwVector3f> pointsBuffer;

            log("render3D: Processing camera %zu points...\n", k);
            pointsBuffer.clear();
            if (camera.featureHistoryCPU.featureCount == nullptr) {
                logError("Camera %zu: featureHistoryCPU.featureCount is NULL!\n", k);
                continue;
            }
            if (camera.worldPoints == nullptr) {
                logError("Camera %zu: worldPoints is NULL!\n", k);
                continue;
            }
            
            uint32_t totalFeatures = *camera.featureHistoryCPU.featureCount;
            uint32_t validPoints = 0;
            log("render3D: Camera %zu has %u total features, checking world points...\n", k, totalFeatures);
            for (size_t i = 0; i < totalFeatures; i++)
            {
                auto& p = camera.worldPoints[i];
                if (p.w != 0.0f) {
                    pointsBuffer.push_back(dwVector3f{p.x, p.y, p.z});
                    validPoints++;
                }
            }
            log("render3D: Camera %zu has %u valid points to render\n", k, validPoints);
            if (validPoints > 0 || totalFeatures > 0) {
                static uint32_t logCounter = 0;
                if (logCounter % 30 == 0) { // Log every 30 frames to avoid spam
                    log("Camera %zu: Rendering %u valid points out of %u total features\n", k, validPoints, totalFeatures);
                }
                logCounter++;
            }
            dwRenderEngine_setColor(colors[k], m_renderEngine);
            checkBufferSizeAgainst(pointsBuffer.size());
            log("render3D: Rendering %zu points for camera %zu...\n", pointsBuffer.size(), k);
            dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D, pointsBuffer.data(), sizeof(dwVector3f), 0, pointsBuffer.size(), m_renderEngine);
            log("render3D: Camera %zu rendering completed\n", k);
        }
    }

    void renderCamera(CameraData& camera)
    {
        // Static buffers so allocation only happens for the first frame
        static std::vector<dwVector2f> featuresBuffer;
        static std::vector<dwVector2f> reprojectionsBuffer;
        static std::vector<dwVector2f> predictionPointsBuffer;
        static std::vector<dwVector2f> predictionLinesBuffer;

        if (!camera.frameGL)
            return;

        dwRenderEngine_setProjection(&DW_IDENTITY_MATRIX4F, m_renderEngine);
        dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);

        //////////////
        // Draw image
        dwVector2f range{};
        dwImageGL* frameGL;
        dwImage_getGL(&frameGL, camera.frameGL);
        range.x = frameGL->prop.width;
        range.y = frameGL->prop.height;
        dwRenderEngine_setCoordinateRange2D(range, m_renderEngine);
        dwRenderEngine_renderImage2D(frameGL, {0.0f, 0.0f, range.x, range.y}, m_renderEngine);

        //////////////
        // Draw features
        dwFeatureArray curFeatures{};
        dwFeatureHistoryArray_getCurrent(&curFeatures, &camera.featureHistoryCPU);
        const dwVector2f* locations = curFeatures.locations;

        featuresBuffer.clear();
        reprojectionsBuffer.clear();
        predictionPointsBuffer.clear();
        predictionLinesBuffer.clear();
        for (size_t i = 0; i < *curFeatures.featureCount; i++)
        {
            if (curFeatures.statuses[i] != DW_FEATURE2D_STATUS_TRACKED)
                continue;

            featuresBuffer.push_back(locations[i]);
            predictionPointsBuffer.push_back(camera.predictedFeatureLocations[i]);
            predictionLinesBuffer.push_back(camera.predictedFeatureLocations[i]);
            predictionLinesBuffer.push_back(locations[i]);

            if (camera.worldPoints[i].w != 0.0f)
                reprojectionsBuffer.push_back(camera.projectedLocations[i]);
        }

        dwRenderEngine_setPointSize(2.0f, m_renderEngine);
        dwRenderEngine_setColor(DW_RENDERER_COLOR_GREEN, m_renderEngine);
        checkBufferSizeAgainst(featuresBuffer.size() / 2);
        dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_2D, featuresBuffer.data(), sizeof(dwVector2f), 0, featuresBuffer.size(), m_renderEngine);

        dwRenderEngine_setPointSize(4.0f, m_renderEngine);
        dwRenderEngine_setColor(DW_RENDERER_COLOR_DARKRED, m_renderEngine);
        checkBufferSizeAgainst(reprojectionsBuffer.size() / 2);
        dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_2D, reprojectionsBuffer.data(), sizeof(dwVector2f), 0, reprojectionsBuffer.size(), m_renderEngine);

        dwRenderEngine_setLineWidth(1.0f, m_renderEngine);
        dwRenderEngine_setColor(DW_RENDERER_COLOR_YELLOW, m_renderEngine);
        checkBufferSizeAgainst(predictionLinesBuffer.size() / 4);
        dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_2D, predictionLinesBuffer.data(), sizeof(dwVector2f), 0, predictionLinesBuffer.size() / 2, m_renderEngine);
    }

    ///------------------------------------------------------------------------------
    /// Main processing of the sample
    ///     - grab a frame from the camera
    ///     - convert frame to RGB
    ///     - push frame through the streamer to convert it into GL
    ///     - track the features in the frame
    ///------------------------------------------------------------------------------
    void onProcess() override
    {
        log("=== onProcess START ===\n");
        ProfileCUDASection s(getProfilerCUDA(), "ProcessFrame");

        log("Releasing frames...\n");
        for (auto& camera : cameras)
            releaseFrame(camera);
        log("Frames released\n");

        log("Waiting for cameras to be ready...\n");
        while (!isAllCamerasReady())
        {
            log("Acquiring next event...\n");
            const dwSensorEvent* event;
            {
                dwStatus status;
                status = dwSensorManager_acquireNextEvent(&event, 0, sensorManager);
                log("acquireNextEvent returned status: %d\n", status);
                if (status == DW_END_OF_STREAM)
                {
                    if (should_AutoExit())
                    {
                        log("AutoExit was set, stopping the sample because reached the end of the data stream\n");
                        stop();
                        return;
                    }
                    log("Resetting...\n");
                    reset();
                    return;
                }
                if (status != DW_SUCCESS) {
                    logError("acquireNextEvent failed with status: %d\n", status);
                    return;
                }
            }

            log("Event acquired, type: %d\n", event->type);
            // Process event
            switch (event->type)
            {
            case DW_SENSOR_CAMERA:
            {
                log("Processing camera event...\n");
                processCamera(*event);
                log("Camera event processed\n");
                break;
            }
            case DW_SENSOR_CAN:
            {
                log("Processing CAN event...\n");
                processCan(*event);
                log("CAN event processed\n");
                break;
            }
            case DW_SENSOR_LIDAR:
            case DW_SENSOR_GPS:
            case DW_SENSOR_IMU:
            case DW_SENSOR_RADAR:
            case DW_SENSOR_TIME:
            case DW_SENSOR_ULTRASONIC:
            case DW_SENSOR_COUNT:
            case DW_SENSOR_DATA:
            default:
                logError("Unexpected event type: %d\n", event->type);
                throw std::runtime_error("Unexpected event type");
            }

            log("Releasing acquired event...\n");
            dwStatus releaseStatus = dwSensorManager_releaseAcquiredEvent(event, sensorManager);
            if (releaseStatus != DW_SUCCESS) {
                logError("releaseAcquiredEvent failed with status: %d\n", releaseStatus);
            }
            log("Event released\n");
        }

        log("All cameras ready, calling updatePose...\n");
        updatePose();
        log("updatePose completed\n");

        log("Collecting timers...\n");
        getProfilerCUDA()->collectTimers();
        log("=== onProcess END ===\n");
    }

    void processCamera(const dwSensorEvent& event)
    {
        log("processCamera: numCamFrames=%u, CAMERA_COUNT=%zu\n", event.numCamFrames, CAMERA_COUNT);
        if (event.numCamFrames != CAMERA_COUNT)
            throw std::runtime_error("All cameras should come in simultaneously");

        for (size_t k = 0; k < event.numCamFrames; k++)
        {
            log("Processing camera %zu...\n", k);
            auto& camera = cameras[k];

            // Get CUDA & GL images
            log("Getting CUDA YUV image for camera %zu...\n", k);
            dwStatus status = dwSensorCamera_getImage(&camera.frameCudaYuv, DW_CAMERA_OUTPUT_CUDA_YUV420_UINT8_SEMIPLANAR, event.camFrames[k]);
            if (status != DW_SUCCESS) {
                logError("dwSensorCamera_getImage YUV failed for camera %zu: %d\n", k, status);
                CHECK_DW_ERROR(status);
            }
            
            log("Getting CUDA RGBA image for camera %zu...\n", k);
            status = dwSensorCamera_getImage(&camera.frameCudaRgba, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, event.camFrames[k]);
            if (status != DW_SUCCESS) {
                logError("dwSensorCamera_getImage RGBA failed for camera %zu: %d\n", k, status);
                CHECK_DW_ERROR(status);
            }

            log("Sending to GL streamer for camera %zu...\n", k);
            status = dwImageStreamerGL_producerSend(camera.frameCudaRgba, camera.image2GL);
            if (status != DW_SUCCESS) {
                logError("dwImageStreamerGL_producerSend failed for camera %zu: %d\n", k, status);
                CHECK_DW_ERROR(status);
            }
            
            log("Receiving from GL streamer for camera %zu...\n", k);
            status = dwImageStreamerGL_consumerReceive(&camera.frameGL, 30000, camera.image2GL);
            if (status != DW_SUCCESS) {
                logError("dwImageStreamerGL_consumerReceive failed for camera %zu: %d\n", k, status);
                CHECK_DW_ERROR(status);
            }
            log("Camera %zu processed successfully\n", k);
        }
    }

    dwFeatureArray trackFrame(size_t cameraIdx, const dwTransformation3f& predictedRig2World)
    {
        log("trackFrame: Starting for camera %zu...\n", cameraIdx);
        log("trackFrame: Camera index %zu, cameras array size: %zu\n", cameraIdx, cameras.size());
        if (cameraIdx >= cameras.size()) {
            logError("trackFrame: Camera index %zu out of bounds!\n", cameraIdx);
            throw std::runtime_error("Camera index out of bounds");
        }
        log("trackFrame: Accessing camera reference...\n");
        // Use pointer to safely check validity
        CameraData* cameraPtr = &cameras[cameraIdx];
        log("trackFrame: Camera pointer obtained: %p\n", (void*)cameraPtr);
        if (cameraPtr == nullptr) {
            logError("trackFrame: Camera pointer is NULL!\n");
            throw std::runtime_error("Camera pointer is NULL");
        }
        auto& camera = *cameraPtr;
        log("trackFrame: Camera reference obtained\n");
        
        // Verify camera structure is accessible
        log("trackFrame: Verifying camera structure accessibility...\n");
        void* cameraAddr = &camera;
        log("trackFrame: Camera address: %p\n", cameraAddr);
        
        log("trackFrame: Swapping pyramids...\n");
        std::swap(camera.pyramidCurrent, camera.pyramidPrevious);

        if (doFeaturePrediction)
        {
            log("trackFrame: Predicting feature positions...\n");
            //Predict
            ProfileCUDASection s(getProfilerCUDA(), "Predict");

            dwStatus status = dwReconstructor_predictFeaturePosition(camera.d_predictedFeatureLocations,
                                                                  cameraIdx,
                                                                  &currentRig2World,
                                                                  &predictedRig2World,
                                                                  camera.featureDetectedGPU.featureCount,
                                                                  camera.featureDetectedGPU.statuses,
                                                                  camera.featureDetectedGPU.locations,
                                                                  camera.d_worldPoints,
                                                                  reconstructor);
            if (status != DW_SUCCESS) {
                logError("trackFrame: Camera %zu predictFeaturePosition failed: %d\n", cameraIdx, status);
                CHECK_DW_ERROR(status);
            }
        }
        else
        {
            log("trackFrame: Copying detected locations (no prediction)...\n");
            cudaError_t cudaStatus = cudaMemcpyAsync(camera.d_predictedFeatureLocations,
                                             camera.featureDetectedGPU.locations,
                                             maxFeatureCount * sizeof(dwVector2f),
                                             cudaMemcpyDeviceToDevice,
                                             cudaStream_t(0));
            if (cudaStatus != cudaSuccess) {
                logError("trackFrame: CUDA memcpy failed: %s\n", cudaGetErrorString(cudaStatus));
                CHECK_CUDA_ERROR(cudaStatus);
            }
        }

        //Build pyramid
        {
            log("trackFrame: Building pyramid for camera %zu...\n", cameraIdx);
            ProfileCUDASection s(getProfilerCUDA(), "Pyramid");

            dwImageCUDA* img;
            dwStatus status = dwImage_getCUDA(&img, camera.frameCudaYuv);
            if (status != DW_SUCCESS) {
                logError("trackFrame: dwImage_getCUDA failed for camera %zu: %d\n", cameraIdx, status);
                CHECK_DW_ERROR(status);
            }
            status = dwImageFilter_computePyramid(&camera.pyramidCurrent, img, 0, m_context);
            if (status != DW_SUCCESS) {
                logError("trackFrame: dwImageFilter_computePyramid failed for camera %zu: %d\n", cameraIdx, status);
                CHECK_DW_ERROR(status);
            }
            log("trackFrame: Pyramid built for camera %zu\n", cameraIdx);
        }

        dwFeatureArray featuresTracked{};
        {
            log("trackFrame: Tracking features for camera %zu...\n", cameraIdx);
            
            // Validate all pointers before calling trackFeatures
            log("trackFrame: Validating pointers for camera %zu...\n", cameraIdx);
            // Skip tracker validation as it's causing crashes - let API handle it
            log("trackFrame: Skipping tracker validation (will be checked by API)\n");
            
            log("trackFrame: Checking d_predictedFeatureLocations...\n");
            if (camera.d_predictedFeatureLocations == nullptr) {
                logError("trackFrame: Camera %zu d_predictedFeatureLocations is NULL!\n", cameraIdx);
                throw std::runtime_error("d_predictedFeatureLocations is NULL");
            }
            log("trackFrame: d_predictedFeatureLocations OK\n");
            
            log("trackFrame: Checking featureHistoryGPU.featureCount...\n");
            if (camera.featureHistoryGPU.featureCount == nullptr) {
                logError("trackFrame: Camera %zu featureHistoryGPU.featureCount is NULL!\n", cameraIdx);
                throw std::runtime_error("featureHistoryGPU.featureCount is NULL");
            }
            log("trackFrame: featureHistoryGPU.featureCount OK\n");
            
            log("trackFrame: Checking featureHistoryGPU.statuses...\n");
            if (camera.featureHistoryGPU.statuses == nullptr) {
                logError("trackFrame: Camera %zu featureHistoryGPU.statuses is NULL!\n", cameraIdx);
                throw std::runtime_error("featureHistoryGPU.statuses is NULL");
            }
            log("trackFrame: featureHistoryGPU.statuses OK\n");
            
            log("trackFrame: Checking featureHistoryGPU.locationHistory...\n");
            if (camera.featureHistoryGPU.locationHistory == nullptr) {
                logError("trackFrame: Camera %zu featureHistoryGPU.locationHistory is NULL!\n", cameraIdx);
                throw std::runtime_error("featureHistoryGPU.locationHistory is NULL");
            }
            log("trackFrame: featureHistoryGPU.locationHistory OK\n");
            
            log("trackFrame: Checking featureDetectedGPU.featureCount...\n");
            if (camera.featureDetectedGPU.featureCount == nullptr) {
                logError("trackFrame: Camera %zu featureDetectedGPU.featureCount is NULL!\n", cameraIdx);
                throw std::runtime_error("featureDetectedGPU.featureCount is NULL");
            }
            log("trackFrame: featureDetectedGPU.featureCount OK\n");
            
            log("trackFrame: Checking featureDetectedGPU.locations...\n");
            if (camera.featureDetectedGPU.locations == nullptr) {
                logError("trackFrame: Camera %zu featureDetectedGPU.locations is NULL!\n", cameraIdx);
                throw std::runtime_error("featureDetectedGPU.locations is NULL");
            }
            log("trackFrame: featureDetectedGPU.locations OK\n");
            
            // Note: Cannot dereference GPU pointers from CPU - they point to GPU memory
            // The featureCount pointers are valid but point to GPU memory, not CPU accessible memory
            log("trackFrame: All pointers validated (GPU pointers cannot be dereferenced from CPU)\n");
            log("trackFrame: Camera %zu - pyramidPrevious levelCount: %u\n", cameraIdx, camera.pyramidPrevious.levelCount);
            log("trackFrame: Camera %zu - pyramidCurrent levelCount: %u\n", cameraIdx, camera.pyramidCurrent.levelCount);
            
            log("trackFrame: Calling dwFeature2DTracker_trackFeatures for camera %zu...\n", cameraIdx);
            ProfileCUDASection s(getProfilerCUDA(), "Tracking");
            dwStatus status = dwFeature2DTracker_trackFeatures(&camera.featureHistoryGPU, &featuresTracked,
                                                            nullptr, &camera.featureDetectedGPU,
                                                            camera.d_predictedFeatureLocations,
                                                            &camera.pyramidPrevious, &camera.pyramidCurrent,
                                                            camera.tracker);
            log("trackFrame: dwFeature2DTracker_trackFeatures returned status: %d for camera %zu\n", status, cameraIdx);
            if (status != DW_SUCCESS) {
                logError("Camera %zu: Tracking failed with status: %d\n", cameraIdx, status);
            } else {
                // Note: featuresTracked.featureCount is a GPU pointer, cannot dereference from CPU
                if (featuresTracked.featureCount != nullptr) {
                    log("Camera %zu: Tracking succeeded (featureCount pointer is valid, but GPU memory - cannot read from CPU)\n", cameraIdx);
                } else {
                    logError("Camera %zu: Tracking succeeded but featuresTracked.featureCount is NULL!\n", cameraIdx);
                }
            }
        }

        log("trackFrame: Completed for camera %zu\n", cameraIdx);
        return featuresTracked;
    }

    void copyDataFromGPU(CameraData& camera)
    {
        //Get tracked feature info to CPU
        ProfileCUDASection s(getProfilerCUDA(), "downloadToCPU");
        dwStatus status = dwFeatureHistoryArray_copyAsync(&camera.featureHistoryCPU, &camera.featureHistoryGPU, 0);
        if (status != DW_SUCCESS) {
            logError("Feature history copy failed with status: %d\n", status);
        }

        cudaError_t cudaStatus;
        cudaStatus = cudaMemcpyAsync(camera.predictedFeatureLocations.get(), camera.d_predictedFeatureLocations, maxFeatureCount * sizeof(dwVector2f), cudaMemcpyDeviceToHost, cudaStream_t(0));
        if (cudaStatus != cudaSuccess) logError("CUDA memcpy predicted locations failed: %s\n", cudaGetErrorString(cudaStatus));
        
        cudaStatus = cudaMemcpyAsync(camera.projectedLocations.get(), camera.d_projectedLocations, maxFeatureCount * sizeof(dwVector2f), cudaMemcpyDeviceToHost, cudaStream_t(0));
        if (cudaStatus != cudaSuccess) logError("CUDA memcpy projected locations failed: %s\n", cudaGetErrorString(cudaStatus));
        
        cudaStatus = cudaMemcpyAsync(camera.worldPoints.get(), camera.d_worldPoints, maxFeatureCount * sizeof(dwVector4f), cudaMemcpyDeviceToHost, cudaStream_t(0));
        if (cudaStatus != cudaSuccess) logError("CUDA memcpy world points failed: %s\n", cudaGetErrorString(cudaStatus));
        
        cudaStreamSynchronize(cudaStream_t(0));
    }

    void processCan(const dwSensorEvent& event)
    {
        log("processCan: consuming CAN frame...\n");
        dwStatus status = dwVehicleIO_consumeCANFrame(&event.canFrame, 0, vehicleIO);
        if (status != DW_SUCCESS) {
            logError("dwVehicleIO_consumeCANFrame failed: %d\n", status);
            CHECK_DW_ERROR(status);
        }

        log("Getting vehicle IO states...\n");
        dwVehicleIOSafetyState vehicleIOSafeState{};
        dwVehicleIONonSafetyState vehicleIONonSafeState{};
        dwVehicleIOActuationFeedback vehicleIOActuationFeedbackState{};
        
        status = dwVehicleIO_getVehicleSafetyState(&vehicleIOSafeState, vehicleIO);
        if (status != DW_SUCCESS) {
            logError("dwVehicleIO_getVehicleSafetyState failed: %d\n", status);
            CHECK_DW_ERROR(status);
        }
        
        status = dwVehicleIO_getVehicleNonSafetyState(&vehicleIONonSafeState, vehicleIO);
        if (status != DW_SUCCESS) {
            logError("dwVehicleIO_getVehicleNonSafetyState failed: %d\n", status);
            CHECK_DW_ERROR(status);
        }
        
        status = dwVehicleIO_getVehicleActuationFeedback(&vehicleIOActuationFeedbackState, vehicleIO);
        if (status != DW_SUCCESS) {
            logError("dwVehicleIO_getVehicleActuationFeedback failed: %d\n", status);
            CHECK_DW_ERROR(status);
        }
        
        // Log CAN data values
        static uint32_t canLogCounter = 0;
        if (canLogCounter % 30 == 0) {
            log("CAN Data - Speed: %.3f m/s, Steering: %.3f rad, SpeedTimestamp: %lu, SteeringTimestamp: %lu\n",
                vehicleIONonSafeState.speedESC,
                vehicleIOSafeState.steeringWheelAngle,
                vehicleIONonSafeState.speedESCTimestamp,
                vehicleIOSafeState.timestamp_us);
        }
        canLogCounter++;
        
        log("Adding vehicle IO state to egomotion...\n");
        status = dwEgomotion_addVehicleIOState(&vehicleIOSafeState, &vehicleIONonSafeState, &vehicleIOActuationFeedbackState, egomotion);
        if (status != DW_SUCCESS) {
            logError("dwEgomotion_addVehicleIOState failed: %d\n", status);
            CHECK_DW_ERROR(status);
        }
        log("CAN processing completed\n");
    }

    bool isAllCamerasReady()
    {
        for (auto& camera : cameras)
        {
            if (!camera.frameCudaYuv)
                return false;
        }
        return true;
    }
    void updatePose()
    {
        log("=== updatePose START ===\n");
        dwTime_t now;
        log("Getting timestamp from camera 0...\n");
        dwStatus status = dwImage_getTimestamp(&now, cameras[0].frameCudaYuv);
        if (status != DW_SUCCESS) {
            logError("dwImage_getTimestamp failed: %d\n", status);
            CHECK_DW_ERROR(status);
        }
        log("Timestamp: %lu, lastEgoUpdate: %lu\n", now, lastEgoUpdate);

        // update current absolute estimate of the pose using relative motion between now and last time
        {
            log("Computing relative transformation...\n");
            dwTransformation3f rigLast2rigNow;
            status = dwEgomotion_computeRelativeTransformation(&rigLast2rigNow, nullptr, lastEgoUpdate, now, egomotion);
            log("dwEgomotion_computeRelativeTransformation status: %d\n", status);
            if (DW_SUCCESS == status)
            {
                // Log relative transformation values
                static uint32_t egoLogCounter = 0;
                if (egoLogCounter % 30 == 0) {
                    log("Relative transformation - translation: (%.3f, %.3f, %.3f), rotation matrix:\n",
                        rigLast2rigNow.array[12], rigLast2rigNow.array[13], rigLast2rigNow.array[14]);
                    log("  [%.3f %.3f %.3f]\n  [%.3f %.3f %.3f]\n  [%.3f %.3f %.3f]\n",
                        rigLast2rigNow.array[0], rigLast2rigNow.array[1], rigLast2rigNow.array[2],
                        rigLast2rigNow.array[4], rigLast2rigNow.array[5], rigLast2rigNow.array[6],
                        rigLast2rigNow.array[8], rigLast2rigNow.array[9], rigLast2rigNow.array[10]);
                }
                egoLogCounter++;
                
                log("Applying relative transformation...\n");
                // compute absolute pose given the relative motion between two last estimates
                dwTransformation3f rigNow2World;
                dwEgomotion_applyRelativeTransformation(&rigNow2World, &rigLast2rigNow, &egoRig2World);
                egoRig2World = rigNow2World;
                
                // Log absolute ego pose
                static uint32_t egoPoseLogCounter = 0;
                if (egoPoseLogCounter % 30 == 0) {
                    log("Ego pose updated - position: (%.3f, %.3f, %.3f)\n",
                        egoRig2World.array[12], egoRig2World.array[13], egoRig2World.array[14]);
                }
                egoPoseLogCounter++;
            } else {
                log("Relative transformation not available (this is normal at start)\n");
            }
        }
        lastEgoUpdate = now;

        log("Computing transformations...\n");
        dwTransformation3f invPreviousEgoRig2World;
        Mat4_IsoInv(invPreviousEgoRig2World.array, previousEgoRig2World.array);

        dwTransformation3f egoLastRig2PredictedRig;
        Mat4_AxB(egoLastRig2PredictedRig.array, invPreviousEgoRig2World.array, egoRig2World.array);

        dwTransformation3f predictedRig2World;
        Mat4_AxB(predictedRig2World.array, currentRig2World.array, egoLastRig2PredictedRig.array);
        
        // Log predicted pose
        static uint32_t poseLogCounter = 0;
        if (poseLogCounter % 30 == 0) {
            log("Predicted pose: (%.3f, %.3f, %.3f), Current pose: (%.3f, %.3f, %.3f)\n",
                predictedRig2World.array[12], predictedRig2World.array[13], predictedRig2World.array[14],
                currentRig2World.array[12], currentRig2World.array[13], currentRig2World.array[14]);
        }
        poseLogCounter++;
        log("Transformations computed\n");

        // Features
        log("Tracking features for all cameras...\n");
        static bool isFirstFrame = true;
        if (isFirstFrame) {
            log("First frame detected - initializing pyramids for all cameras...\n");
            for (size_t k = 0; k < CAMERA_COUNT; k++) {
                auto& camera = cameras[k];
                // Build initial pyramid for previous frame (will be used as previous in next frame)
                dwImageCUDA* img;
                dwStatus status = dwImage_getCUDA(&img, camera.frameCudaYuv);
                if (status == DW_SUCCESS) {
                    status = dwImageFilter_computePyramid(&camera.pyramidPrevious, img, 0, m_context);
                    if (status != DW_SUCCESS) {
                        logError("Failed to build initial pyramid for camera %zu: %d\n", k, status);
                    } else {
                        log("Initial pyramid built for camera %zu\n", k);
                    }
                }
            }
            isFirstFrame = false;
        }
        
        const dwVector2f* d_trackedLocations[CAMERA_COUNT];
        for (size_t k = 0; k < CAMERA_COUNT; k++)
        {
            log("Tracking frame for camera %zu...\n", k);
            dwFeatureArray featuresTracked = trackFrame(k, predictedRig2World);
            d_trackedLocations[k]          = featuresTracked.locations;
            // Note: featuresTracked.featureCount is a GPU pointer, cannot dereference from CPU
            if (featuresTracked.featureCount != nullptr) {
                log("Camera %zu tracking completed (featureCount pointer valid, GPU memory - cannot read from CPU)\n", k);
            } else {
                logError("Camera %zu tracking completed but featureCount is NULL!\n", k);
            }
        }
        log("All cameras tracked\n");

        // Refine
        if (doPoseRefinement)
        {
            poseColor = DW_RENDERER_COLOR_DARKRED;

            {
                ProfileCUDASection s(getProfilerCUDA(), "PoseEstimation");

                dwStatus status = dwReconstructor_estimatePoseAsync(&currentRig2World,
                                                                 &predictedRig2World,
                                                                 CAMERA_COUNT,
                                                                 d_allFeatureCount,
                                                                 d_allFeatureStatuses,
                                                                 d_trackedLocations,
                                                                 d_allWorldPoints,
                                                                 reconstructor);
                if (status != DW_SUCCESS) {
                    logError("Pose estimation failed with status: %d\n", status);
                } else {
                    log("Pose estimation started successfully\n");
                }
            }

            //Blocking memcpy
            {
                log("Getting estimated pose from reconstructor...\n");
                ProfileCUDASection s(getProfilerCUDA(), "copyPose2CPU");
                log("About to call dwReconstructor_getEstimatedPose...\n");
                dwStatus status = dwReconstructor_getEstimatedPose(&currentRig2World, reconstructor);
                log("dwReconstructor_getEstimatedPose returned status: %d\n", status);
                if (status != DW_SUCCESS) {
                    logError("Get estimated pose failed with status: %d\n", status);
                } else {
                    log("Estimated pose retrieved successfully - position: (%.3f, %.3f, %.3f)\n",
                        currentRig2World.array[12], currentRig2World.array[13], currentRig2World.array[14]);
                }
            }
        }
        else
        {
            poseColor        = DW_RENDERER_COLOR_DARKGREEN;
            currentRig2World = predictedRig2World;
            log("Pose refinement disabled - using predicted pose: (%.3f, %.3f, %.3f)\n",
                currentRig2World.array[12], currentRig2World.array[13], currentRig2World.array[14]);
        }

        // Store pose
        log("Storing pose...\n");
        recordPose(predictedRig2World, currentRig2World);
        log("Pose stored\n");

        previousEgoRig2World = egoRig2World;

        //Update feature history
        {
            log("Updating feature history...\n");
            ProfileCUDASection s(getProfilerCUDA(), "History3D");
            int32_t currentPoseIdx = -1;
            log("About to call dwReconstructor_updateHistory with pose: (%.3f, %.3f, %.3f)\n", 
                currentRig2World.array[12], currentRig2World.array[13], currentRig2World.array[14]);
            dwStatus status = dwReconstructor_updateHistory(&currentPoseIdx,
                                                         &currentRig2World,
                                                         CAMERA_COUNT,
                                                         d_allFeatureCount,
                                                         d_trackedLocations,
                                                         reconstructor);
            log("dwReconstructor_updateHistory returned status: %d, poseIdx: %d\n", status, currentPoseIdx);
            if (status != DW_SUCCESS) {
                logError("Update history failed with status: %d\n", status);
            } else {
                if (currentPoseIdx >= 0) {
                    log("Feature history updated - NEW pose added to history at index %d\n", currentPoseIdx);
                } else {
                    static uint32_t skipLogCounter = 0;
                    if (skipLogCounter % 30 == 0) {
                        log("Feature history updated - pose NOT added (insufficient movement/rotation or history full)\n");
                    }
                    skipLogCounter++;
                }
            }
        }

        for (size_t k = 0; k < CAMERA_COUNT; k++)
        {
            auto& camera = cameras[k];

            //Triangulate
            ProfileCUDASection s(getProfilerCUDA(), "Triangulation");
            // Note: Cannot dereference GPU pointer from CPU - featureCount is in GPU memory
            log("Camera %zu: Triangulating features (featureCount is GPU pointer, cannot read from CPU)\n", k);
            
            // Check world points before triangulation (they should all be zero)
            static uint32_t triangLogCounter = 0;
            if (triangLogCounter % 60 == 0 && k == 0) {
                // Sample a few world points from CPU copy to see their state
                uint32_t zeroCount = 0;
                for (size_t i = 0; i < std::min(static_cast<size_t>(5), static_cast<size_t>(maxFeatureCount)); i++) {
                    auto& p = camera.worldPoints[i];
                    if (p.w == 0.0f) zeroCount++;
                }
                log("Camera %zu: Before triangulation - %u out of 5 sample points have w==0\n", k, zeroCount);
            }
            
            dwStatus status = dwReconstructor_triangulateFeatures(
                camera.d_worldPoints,
                camera.featureHistoryGPU.statuses,
                camera.featureHistoryGPU.featureCount,
                k, reconstructor);
            
            log("Camera %zu: dwReconstructor_triangulateFeatures returned status: %d\n", k, status);
            if (status != DW_SUCCESS) {
                logError("Camera %zu: Triangulation failed with status: %d\n", k, status);
            } else {
                log("Camera %zu: Triangulation completed successfully\n", k);
                // Note: World points are still on GPU at this point, will be copied to CPU later
            }
            triangLogCounter++;
        }

        //Project back onto camera for display
        {
            log("Reprojecting features...\n");
            ProfileCUDASection s(getProfilerCUDA(), "Reproject");
            dwStatus status = dwReconstructor_project(d_allProjections,
                                                   &currentRig2World,
                                                   d_allFeatureCount,
                                                   d_allWorldPoints,
                                                   reconstructor);
            log("dwReconstructor_project returned status: %d\n", status);
            if (status != DW_SUCCESS) {
                logError("Reprojection failed with status: %d\n", status);
            }
        }

        log("Copying data from GPU and compacting...\n");
        for (size_t k = 0; k < CAMERA_COUNT; k++)
        {
            log("Processing camera %zu: copying from GPU...\n", k);
            copyDataFromGPU(cameras[k]);
            log("Camera %zu: GPU copy completed\n", k);
            
            // Log sample world points for debugging - check after GPU copy
            static uint32_t worldPointLogCounter = 0;
            uint32_t totalFeatures = 0;
            if (cameras[k].featureHistoryCPU.featureCount != nullptr) {
                totalFeatures = *cameras[k].featureHistoryCPU.featureCount;
            }
            
            if (totalFeatures > 0) {
                uint32_t validCount = 0;
                uint32_t nonZeroWCount = 0;
                uint32_t sampleCount = 0;
                
                // Check first 20 points to see their state
                for (size_t i = 0; i < std::min(static_cast<size_t>(20), static_cast<size_t>(totalFeatures)); i++) {
                    auto& p = cameras[k].worldPoints[i];
                    if (p.w != 0.0f) {
                        nonZeroWCount++;
                        validCount++;
                        if (sampleCount < 5) {
                            log("Camera %zu: Valid world point[%zu] = (%.3f, %.3f, %.3f, w=%.3f)\n", 
                                k, i, p.x, p.y, p.z, p.w);
                            sampleCount++;
                        }
                    }
                }
                
                // Check all points for valid count
                for (size_t i = 0; i < totalFeatures; i++) {
                    auto& p = cameras[k].worldPoints[i];
                    if (p.w != 0.0f) {
                        validCount++;
                    }
                }
                
                // Always log if no valid points, otherwise log every 30 frames
                if (validCount == 0 && totalFeatures > 0) {
                    logError("Camera %zu: WARNING - No valid triangulated points! All %u points have w==0\n", k, totalFeatures);
                    // Log a few sample points to see their values
                    for (size_t i = 0; i < std::min(static_cast<size_t>(10), static_cast<size_t>(totalFeatures)); i++) {
                        auto& p = cameras[k].worldPoints[i];
                        log("Camera %zu: Sample point[%zu] = (%.6f, %.6f, %.6f, w=%.6f)\n", 
                            k, i, p.x, p.y, p.z, p.w);
                    }
                } else if (worldPointLogCounter % 30 == 0) {
                    log("Camera %zu: After GPU copy - %u valid world points (w!=0) out of %u total features\n", 
                        k, validCount, totalFeatures);
                }
            }
            worldPointLogCounter++;
            
            log("Camera %zu: compacting and detecting...\n", k);
            compactAndDetect(k);
            log("Camera %zu: compact and detect completed\n", k);
        }
        log("=== updatePose END ===\n");
    }

    void releaseFrame(CameraData& camera)
    {
        if (camera.frameGL)
        {
            dwStatus status = dwImageStreamerGL_consumerReturn(&camera.frameGL, camera.image2GL);
            if (status != DW_SUCCESS) {
                logError("dwImageStreamerGL_consumerReturn failed: %d\n", status);
            }
            status = dwImageStreamerGL_producerReturn(nullptr, 33000, camera.image2GL);
            if (status != DW_SUCCESS) {
                logError("dwImageStreamerGL_producerReturn failed: %d\n", status);
            }
        }

        camera.frameCudaYuv = nullptr;
        camera.frameGL      = nullptr;
        camera.frame        = nullptr;
    }

    void compactAndDetect(size_t cameraIdx)
    {
        log("compactAndDetect: Starting for camera %zu...\n", cameraIdx);
        ProfileCUDASection s(getProfilerCUDA(), "compactFeature");

        auto& camera = cameras[cameraIdx];

        // Determine which features to throw away
        {
            log("compactAndDetect: Compacting feature history for camera %zu...\n", cameraIdx);
            ProfileCUDASection s(getProfilerCUDA(), "compactFeatureHistory");
            dwStatus status = dwFeature2DTracker_compact(&camera.featureHistoryGPU, camera.tracker);
            if (status != DW_SUCCESS) {
                logError("compactAndDetect: dwFeature2DTracker_compact failed for camera %zu: %d\n", cameraIdx, status);
                CHECK_DW_ERROR(status);
            }
        }

        // Compact SFM data
        {
            log("compactAndDetect: Compacting SFM data for camera %zu...\n", cameraIdx);
            ProfileCUDASection s(getProfilerCUDA(), "compactSFM");
            dwStatus status = dwReconstructor_compactFeatureHistory(
                cameraIdx, camera.featureHistoryGPU.featureCount,
                camera.featureHistoryGPU.newToOldMap, reconstructor);
            if (status != DW_SUCCESS) {
                logError("compactAndDetect: dwReconstructor_compactFeatureHistory failed for camera %zu: %d\n", cameraIdx, status);
                CHECK_DW_ERROR(status);
            }
            
            status = dwReconstructor_compactWorldPoints(
                camera.d_worldPoints, camera.featureHistoryGPU.featureCount,
                camera.featureHistoryGPU.newToOldMap, reconstructor);
            if (status != DW_SUCCESS) {
                logError("compactAndDetect: dwReconstructor_compactWorldPoints failed for camera %zu: %d\n", cameraIdx, status);
                CHECK_DW_ERROR(status);
            }
        }

        // Detect new features
        {
            log("compactAndDetect: Detecting new features for camera %zu...\n", cameraIdx);
            ProfileCUDASection s(getProfilerCUDA(), "detectNewFeatures");
            dwFeatureArray lastTrackedFeatures{};
            dwStatus status = dwFeatureHistoryArray_getCurrent(&lastTrackedFeatures, &camera.featureHistoryGPU);
            if (status != DW_SUCCESS) {
                logError("compactAndDetect: dwFeatureHistoryArray_getCurrent failed for camera %zu: %d\n", cameraIdx, status);
                CHECK_DW_ERROR(status);
            }
            
            log("About to call dwFeature2DDetector_detectFromPyramid...\n");
            status = dwFeature2DDetector_detectFromPyramid(
                &camera.featureDetectedGPU, &camera.pyramidCurrent,
                &lastTrackedFeatures, nullptr, camera.detector);
            log("dwFeature2DDetector_detectFromPyramid returned status: %d\n", status);
            if (status != DW_SUCCESS) {
                logError("Camera %zu: Feature detection failed with status: %d\n", cameraIdx, status);
            } else {
                // Note: featureDetectedGPU.featureCount is a GPU pointer, cannot dereference from CPU
                static uint32_t detectLogCounter = 0;
                if (detectLogCounter % 30 == 0) {
                    log("Camera %zu: Feature detection succeeded (featureCount is GPU pointer, cannot read from CPU)\n", cameraIdx);
                }
                detectLogCounter++;
            }
        }
        log("compactAndDetect: Completed for camera %zu\n", cameraIdx);
    }

    //------------------------------------------------------------------------------
    void recordPose(const dwTransformation3f& poseCAN, const dwTransformation3f& poseRefined)
    {
        dwVector3f positionCAN     = {poseCAN.array[0 + 3 * 4], poseCAN.array[1 + 3 * 4], poseCAN.array[2 + 3 * 4]};
        dwVector3f positionRefined = {poseRefined.array[0 + 3 * 4], poseRefined.array[1 + 3 * 4], poseRefined.array[2 + 3 * 4]};

        positionsCAN.push_back(positionCAN);
        positionsRefined.push_back(positionRefined);
        positionsDifference.push_back(positionCAN);
        positionsDifference.push_back(positionRefined);
    }
};

//------------------------------------------------------------------------------
int main(int argc, const char** argv)
{

    // -------------------
    // define all arguments used by the application
    ProgramArguments args(argc, argv,
                          {ProgramArguments::Option_t("baseDir", (dw_samples::SamplesDataPath::get() + "/samples/sfm/triangulation").c_str()),
                           ProgramArguments::Option_t("rig", "rig.json"),
                           ProgramArguments::Option_t("maxFeatureCount", "2000"),
                           ProgramArguments::Option_t("trackMode", "0"),
                           ProgramArguments::Option_t("useHalf", "0"),
                           ProgramArguments::Option_t("displacementThreshold", "0.001"),
                           ProgramArguments::Option_t("enableAdaptiveWindow", "0")},
                          "SfM sample.");

    // -------------------
    // initialize and start a window application
    SfmSample app(args);

    app.initializeWindow("SfM Sample", 1280, 800, args.enabled("offscreen"));

    return app.run();
}
