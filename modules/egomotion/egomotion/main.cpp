/////////////////////////////////////////////////////////////////////////////////////////
//
// Real-time Egomotion Sample with Synchronized CAN Processing
// Extended from NVIDIA DriveWorks sample for live vehicle data processing
//
/////////////////////////////////////////////////////////////////////////////////////////

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <atomic>

#include <dw/control/vehicleio/VehicleIO.h>
#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/egomotion/base/Egomotion.h>
#include <dw/egomotion/global/GlobalEgomotion.h>
#include <dw/image/Image.h>
#include <dw/rig/Rig.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/camera/Camera.h>
#include <dw/sensors/gps/GPS.h>
#include <dw/sensors/imu/IMU.h>
#include <dw/sensors/sensormanager/SensorManager.h>

#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Renderer.h>
#include <dwvisualization/core/Visualization.h>
#include <dwvisualization/gl/GL.h>
#include <dwvisualization/image/Image.h>

#include <framework/Checks.hpp>
#include <framework/CudaTimer.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/Log.hpp>
#include <framework/Mat4.hpp>
#include <framework/MathUtils.hpp>
#include <framework/MouseView3D.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/RenderUtils.hpp>
#include <framework/SamplesDataPath.hpp>
#include <framework/SimpleStreamer.hpp>
#include <framework/WindowGLFW.hpp>

#include "TrajectoryLogger.hpp"
#include "sygnalpomoparser.hpp"

using namespace dw_samples::common;

static inline bool allowEvery(dwTime_t now, dwTime_t &last, uint64_t period_us) {
    if (now - last < period_us) return false;
    last = now; return true;
}


///------------------------------------------------------------------------------
/// Real-time Egomotion sample with synchronized CAN processing
///------------------------------------------------------------------------------
class EgomotionSample : public DriveWorksSample
{
private:
    // ------------------------------------------------
    // DriveWorks Context and SAL
    // ------------------------------------------------
    dwContextHandle_t m_context                 = DW_NULL_HANDLE;
    dwRenderEngineHandle_t m_renderEngine       = DW_NULL_HANDLE;
    dwEgomotionHandle_t m_egomotion             = DW_NULL_HANDLE;
    dwGlobalEgomotionHandle_t m_globalEgomotion = DW_NULL_HANDLE;
    dwSALHandle_t m_sal                         = DW_NULL_HANDLE;
    dwSensorManagerHandle_t m_sensorManager     = DW_NULL_HANDLE;
    dwRigHandle_t m_rigConfig                   = DW_NULL_HANDLE;
    dwVisualizationContextHandle_t m_vizCtx     = DW_NULL_HANDLE;
    
    dwImageHandle_t m_lastGLFrame = DW_NULL_HANDLE; 

    // Real-time CAN Parser (replaces VehicleIO)
    std::unique_ptr<SygnalPomoParser> m_canParser;

    // ------------------------------------------------
    // Renderer related
    // ------------------------------------------------
    dwRendererHandle_t m_renderer = DW_NULL_HANDLE;
    bool m_shallRender            = false;
    uint32_t m_tileGrid           = 0;
    uint32_t m_tileVideo          = 1;
    uint32_t m_tileRollPlot       = 2;
    uint32_t m_tilePitchPlot      = 3;
    uint32_t m_tileAltitudePlot   = 4;

    enum RenderingMode
    {
        STICK_TO_VEHICLE,
        ON_VEHICLE_STICK_TO_WORLD,
    } m_renderingMode = ON_VEHICLE_STICK_TO_WORLD;

    // ------------------------------------------------
    // Camera and video visualization
    // ------------------------------------------------
    dwImageHandle_t m_convertedImageRGBA = DW_NULL_HANDLE;
    std::unique_ptr<SimpleImageStreamerGL<>> m_streamerInput2GL;
    dwImageGL* m_currentGlFrame = nullptr;

    // ------------------------------------------------
    // Current sensor states
    // ------------------------------------------------
    dwEgomotionParameters m_egomotionParameters{};
    dwGlobalEgomotionParameters m_globalEgomotionParameters{};

    dwIMUFrame m_currentIMUFrame = {};
    dwGPSFrame m_currentGPSFrame = {};
    uint32_t m_imuSensorIdx      = 0;
    uint32_t m_vehicleSensorIdx  = 0;
    uint32_t m_gpsSensorIdx      = 0;

    const dwSensorEvent* acquiredEvent = nullptr;

    // ------------------------------------------------
    // Real-time processing variables
    // ------------------------------------------------
    const dwTime_t POSE_SAMPLE_PERIOD = 100000; // 100ms
    const size_t MAX_BUFFER_POINTS    = 100000;
    
    // State commit timing management
    std::atomic<dwTime_t> m_lastCommitAttempt{0};
    std::atomic<dwTime_t> m_lastSuccessfulCommit{0};
    std::atomic<uint32_t> m_commitAttempts{0};
    std::atomic<uint32_t> m_commitSuccesses{0};

    struct Pose
    {
        dwTime_t timestamp                 = 0;
        dwTransformation3f rig2world       = {};
        dwEgomotionUncertainty uncertainty = {};
        float32_t rpy[3]                   = {};
    };

    std::vector<Pose> m_poseHistory;

    dwQuaternionf m_orientationENU = DW_IDENTITY_QUATERNIONF;
    bool m_hasOrientationENU       = false;

    FILE* m_outputFile = nullptr;

    dwTime_t m_elapsedTime         = 0;
    dwTime_t m_lastSampleTimestamp = 0;
    dwTime_t m_firstTimestamp      = 0;

    TrajectoryLogger m_trajectoryLog;

public:
    EgomotionSample(const ProgramArguments& args)
        : DriveWorksSample(args)
    {
        if (getArgument("output").length() > 0)
        {
            m_outputFile = fopen(getArgument("output").c_str(), "wt");
            log("Real-time output file opened: %s\n", getArgument("output").c_str());
        }

        getMouseView().setRadiusFromCenter(25.0f);
    }

    void initializeDriveWorks(dwContextHandle_t& context) const
    {
        CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
        CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_INFO));

        dwContextParameters sdkParams = {};
#ifdef VIBRANTE
        sdkParams.eglDisplay = getEGLDisplay();
#endif
        CHECK_DW_ERROR(dwInitialize(&context, DW_VERSION, &sdkParams));
    }

    bool onInitialize() override
    {
        dwSensorManagerParams smParams{};
        dwSensorType vehicleSensorType{};

        // Initialize DriveWorks SDK context and SAL
        {
            initializeDriveWorks(m_context);
            CHECK_DW_ERROR(dwVisualizationInitialize(&m_vizCtx, m_context));
        }

        // Read Rig file to extract vehicle properties
        {
            dwStatus ret = dwRig_initializeFromFile(&m_rigConfig, m_context, getArgument("rig").c_str());
            if (ret != DW_SUCCESS)
                throw std::runtime_error("Error reading rig config for real-time processing");

            std::string imuSensorName     = getArgument("imu-sensor-name");
            std::string vehicleSensorName = getArgument("vehicle-sensor-name");
            std::string gpsSensorName     = getArgument("gps-sensor-name");
            std::string cameraSensorName  = getArgument("camera-sensor-name");

            // Extract sensor names from rig config
            {
                uint32_t cnt;
                dwRig_getSensorCount(&cnt, m_rigConfig);
                char buffer[256]; // Declare buffer once for the entire loop scope
                
                for (uint32_t i = 0; i < cnt; i++)
                {
                    dwSensorType type;
                    const char* name;
                    dwRig_getSensorType(&type, i, m_rigConfig);
                    dwRig_getSensorName(&name, i, m_rigConfig);

                    if (type == DW_SENSOR_IMU && imuSensorName.length() == 0)
                    {
                        imuSensorName = name;
                        sprintf(buffer, "[SENSORS IN RIG] IMU Sensor found: %s\n", name);
                        printColored(stdout, COLOR_GREEN, buffer);
                    }
                    if ((type == DW_SENSOR_CAN || type == DW_SENSOR_DATA) && vehicleSensorName.length() == 0)
                    {
                        vehicleSensorName = name;
                        sprintf(buffer, "[SENSORS IN RIG] Vehicle CAN interface found: %s\n", name);
                        printColored(stdout, COLOR_GREEN, buffer);
                    }
                    if (type == DW_SENSOR_GPS && gpsSensorName.length() == 0)
                    {
                        gpsSensorName = name;
                        sprintf(buffer, "[SENSORS IN RIG] GPS Sensor found: %s\n", name);
                        printColored(stdout, COLOR_GREEN, buffer);
                    }
                    if (type == DW_SENSOR_CAMERA && cameraSensorName.length() == 0)
                    {
                        cameraSensorName = name;
                        sprintf(buffer, "[SENSORS IN RIG] Camera Sensor found, Rig intake name: %s\n", name);
                        printColored(stdout, COLOR_GREEN, buffer);
                    }
                }
            }

            // Get sensor indices
            CHECK_DW_ERROR_MSG(dwRig_findSensorByName(&m_imuSensorIdx, imuSensorName.c_str(), m_rigConfig), 
                              "Cannot find IMU sensor for real-time processing");
            CHECK_DW_ERROR_MSG(dwRig_findSensorByName(&m_vehicleSensorIdx, vehicleSensorName.c_str(), m_rigConfig), 
                              "Cannot find vehicle sensor for real-time processing");

            smParams.enableSensors[smParams.numEnableSensors++] = m_vehicleSensorIdx;
            smParams.enableSensors[smParams.numEnableSensors++] = m_imuSensorIdx;

            CHECK_DW_ERROR_MSG(dwRig_getSensorType(&vehicleSensorType, m_vehicleSensorIdx, m_rigConfig), 
                              "Cannot determine vehicle sensor type");

            // Initialize egomotion parameters
            dwEgomotion_initParamsFromRig(&m_egomotionParameters, m_rigConfig, imuSensorName.c_str(), nullptr);

            // GPS sensor (optional)
            if (dwRig_findSensorByName(&m_gpsSensorIdx, gpsSensorName.c_str(), m_rigConfig) == DW_SUCCESS)
            {
                dwGlobalEgomotion_initParamsFromRig(&m_globalEgomotionParameters, m_rigConfig, gpsSensorName.c_str());
                smParams.enableSensors[smParams.numEnableSensors++] = m_gpsSensorIdx;
            }
            else
            {
                logWarn("GPS sensor not found - global egomotion will not be available\n");
            }

            // Camera sensor (optional)
            uint32_t cameraSensorId = 0;
            if (dwRig_findSensorByName(&cameraSensorId, cameraSensorName.c_str(), m_rigConfig) == DW_SUCCESS)
            {
                smParams.enableSensors[smParams.numEnableSensors++] = cameraSensorId;
            }
            else
            {
                logWarn("Camera sensor not found - no video display will be available\n");
            }
        }
        

        // Initialize Egomotion module
        {
            if (getArgument("mode") == "0")
                m_egomotionParameters.motionModel = DW_EGOMOTION_ODOMETRY;
            else if (getArgument("mode") == "1")
            {
                m_egomotionParameters.motionModel = DW_EGOMOTION_IMU_ODOMETRY;
                m_egomotionParameters.estimateInitialOrientation = true;
            }
            else
            {
                logError("Invalid mode %s for real-time processing\n", getArgument("mode").c_str());
                return false;
            }

            m_egomotionParameters.automaticUpdate = true;
            auto speedType = std::stoi(getArgument("speed-measurement-type"));
            m_egomotionParameters.speedMeasurementType = dwEgomotionSpeedMeasurementType(speedType);

            if (getArgument("enable-suspension") == "1")
            {
                if (m_egomotionParameters.motionModel == DW_EGOMOTION_IMU_ODOMETRY)
                {
                    m_egomotionParameters.suspension.model = DW_EGOMOTION_SUSPENSION_TORSIONAL_SPRING_MODEL;
                }
                else
                {
                    logError("Suspension model requires Odometry+IMU mode (--mode=1)\n");
                    return false;
                }
            }

            dwStatus status = dwEgomotion_initialize(&m_egomotion, &m_egomotionParameters, m_context);
            if (status != DW_SUCCESS)
            {
                logError("Error initializing real-time egomotion: %s\n", dwGetStatusName(status));
                return false;
            }
        }

        // Initialize Global Egomotion module
        {
            dwStatus status = dwGlobalEgomotion_initialize(&m_globalEgomotion, &m_globalEgomotionParameters, m_context);
            if (status != DW_SUCCESS)
            {
                logError("Error initializing global egomotion: %s\n", dwGetStatusName(status));
                return false;
            }
        }

        // Initialize Sensors and Real-time CAN Parser
        {
            CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_context));

            dwStatus status = dwSensorManager_initializeFromRigWithParams(&m_sensorManager, m_rigConfig, &smParams, 64, m_sal);
            if (status != DW_SUCCESS)
            {
                logError("Error initializing SensorManager for real-time processing: %s\n", dwGetStatusName(status));
                return false;
            }

            // Initialize Real-time CAN Parser (replaces VehicleIO)
            {
                m_canParser = std::make_unique<SygnalPomoParser>();
                
                // Load Hyundai/Kia configuration for real-time processing
                SygnalPomoParser::VehicleCANConfiguration realtimeConfig;
                // Use default Hyundai/Kia CAN IDs and scaling factors
                
                if (!m_canParser->loadVehicleConfiguration(realtimeConfig)) {
                    logError("Failed to initialize real-time CAN parser\n");
                    return false;
                }
                
                // Configure speed measurement type
                auto speedType = std::stoi(getArgument("speed-measurement-type"));
                m_canParser->configureSpeedMeasurementType(
                    static_cast<dwEgomotionSpeedMeasurementType>(speedType));
                
                log("SYGNALPOMO CAN PARSER initialized successfully\n");
                log("CAN-ID Config: Speed=0x%03X, Steering=0x%03X, Wheels=0x%03X\n", 
                    realtimeConfig.speedCANId, realtimeConfig.steeringWheelAngleCANId, realtimeConfig.wheelSpeedCANId);
            }

            // Initialize camera processing (if available)
            uint32_t cnt;
            dwSensorManager_getNumSensors(&cnt, DW_SENSOR_CAMERA, m_sensorManager);
            if (cnt == 1)
            {
                uint32_t cameraSensorIndex{};
                CHECK_DW_ERROR(dwSensorManager_getSensorIndex(&cameraSensorIndex, DW_SENSOR_CAMERA, 0, m_sensorManager));
                dwSensorHandle_t cameraSensor = DW_NULL_HANDLE;
                CHECK_DW_ERROR(dwSensorManager_getSensorHandle(&cameraSensor, cameraSensorIndex, m_sensorManager));

                dwCameraProperties cameraProperties{};
                dwImageProperties outputProperties{};

                CHECK_DW_ERROR(dwSensorCamera_getSensorProperties(&cameraProperties, cameraSensor));
                // Request CUDA RGBA format directly - no manual conversion needed
                CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&outputProperties, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, cameraSensor));

                std::cout << "Camera image with " << cameraProperties.resolution.x << "x"
                        << cameraProperties.resolution.y << " at " << cameraProperties.framerate << " FPS" << std::endl;

                // Remove conversion image creation - not needed
                m_streamerInput2GL.reset(new SimpleImageStreamerGL<>(outputProperties, 12, m_context));
            }

            // Start sensor manager for real-time processing
            if (dwSensorManager_start(m_sensorManager) != DW_SUCCESS)
            {
                logError("Failed to start SensorManager for real-time processing\n");
                dwSensorManager_release(m_sensorManager);
                return false;
            }
        }

        // Initialize rendering subsystem
        {
            CHECK_DW_ERROR(dwRenderer_initialize(&m_renderer, m_vizCtx));

            dwRect rect;
            rect.width = getWindowWidth();
            rect.height = getWindowHeight();
            rect.x = 0;
            rect.y = 0;
            dwRenderer_setRect(rect, m_renderer);

            dwRenderEngineParams params{};
            params.bufferSize = sizeof(Pose) * MAX_BUFFER_POINTS;
            params.bounds = {0, 0, static_cast<float32_t>(getWindowWidth()), static_cast<float32_t>(getWindowHeight())};

            {
                dwRenderEngine_initTileState(&params.defaultTile);
                params.defaultTile.layout.viewport = params.bounds;
            }
            CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_vizCtx));

            dwRenderEngineTileState tileParams = params.defaultTile;
            tileParams.projectionMatrix = DW_IDENTITY_MATRIX4F;
            tileParams.modelViewMatrix = DW_IDENTITY_MATRIX4F;
            tileParams.layout.positionLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_ABSOLUTE;
            tileParams.layout.sizeLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_ABSOLUTE;

            // Video tile
            {
                tileParams.layout.viewport = {0.f, 0.f, getWindowWidth() / 5.0f, getWindowHeight() / 5.0f};
                tileParams.layout.positionType = DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_LEFT;
                dwRenderEngine_addTile(&m_tileVideo, &tileParams, m_renderEngine);
            }

            // Plot tiles
            {
                const float32_t plotWidth = getWindowWidth() / 4.0f;
                const float32_t plotHeight = getWindowHeight() / 4.0f;

                tileParams.layout.viewport = {0.f, 2 * plotHeight, plotWidth, plotHeight};
                tileParams.layout.positionType = DW_RENDER_ENGINE_TILE_POSITION_TYPE_BOTTOM_RIGHT;
                dwRenderEngine_addTile(&m_tileRollPlot, &tileParams, m_renderEngine);

                tileParams.layout.viewport = {0.f, plotHeight, plotWidth, plotHeight};
                dwRenderEngine_addTile(&m_tilePitchPlot, &tileParams, m_renderEngine);

                tileParams.layout.viewport = {0.f, 0.f, plotWidth, plotHeight};
                dwRenderEngine_addTile(&m_tileAltitudePlot, &tileParams, m_renderEngine);
            }
        }

        // Initialize trajectory logger
        {
            m_trajectoryLog.addTrajectory("GPS", TrajectoryLogger::Color::GREEN);
            m_trajectoryLog.addTrajectory("Egomotion", TrajectoryLogger::Color::RED);
        }

        log("Real-time Egomotion sample initialized successfully\n");
        return true;
    }

    void onRelease() override
    {
        if (acquiredEvent)
            dwSensorManager_releaseAcquiredEvent(acquiredEvent, m_sensorManager);

        if (getArgument("outputkml").length())
            m_trajectoryLog.writeKML(getArgument("outputkml"));

        if (m_outputFile)
            fclose(m_outputFile);

        if (m_convertedImageRGBA)
            dwImage_destroy(m_convertedImageRGBA);

        // Release real-time CAN parser
        if (m_canParser) {
            m_canParser.reset();
        }

        dwSensorManager_stop(m_sensorManager);
        dwSensorManager_release(m_sensorManager);
        if (m_streamerInput2GL && m_lastGLFrame != DW_NULL_HANDLE) {
            m_streamerInput2GL->release();
            m_lastGLFrame = DW_NULL_HANDLE;
        }
        m_streamerInput2GL.reset();

        dwGlobalEgomotion_release(m_globalEgomotion);
        dwEgomotion_release(m_egomotion);
        dwRig_release(m_rigConfig);

        if (m_renderer)
            dwRenderer_release(m_renderer);

        if (m_renderEngine != DW_NULL_HANDLE)
            CHECK_DW_ERROR(dwRenderEngine_release(m_renderEngine));

        dwSAL_release(m_sal);

        CHECK_DW_ERROR(dwVisualizationRelease(m_vizCtx));
        CHECK_DW_ERROR(dwRelease(m_context));
        CHECK_DW_ERROR(dwLogger_release());

        log("Real-time Egomotion sample released\n");
    }

    void onResizeWindow(int width, int height) override
    {
        {
            dwRect rect;
            rect.width = width;
            rect.height = height;
            rect.x = 0;
            rect.y = 0;
            dwRenderer_setRect(rect, m_renderer);
        }

        {
            dwRenderEngine_reset(m_renderEngine);
            dwRectf rect;
            rect.width = width;
            rect.height = height;
            rect.x = 0;
            rect.y = 0;
            dwRenderEngine_setBounds(rect, m_renderEngine);
        }

        log("Real-time window resized to %dx%d\n", width, height);
    }

    void render3DGrid()
    {
        if (m_poseHistory.empty())
            return;

        dwRenderEngine_setTile(m_tileGrid, m_renderEngine);

        const Pose& currentPose = m_poseHistory.back();
        auto currentRig2World = currentPose.rig2world;

        dwMatrix4f modelView;
        {
            if (m_renderingMode == RenderingMode::STICK_TO_VEHICLE)
            {
                dwTransformation3f world2rig;
                Mat4_IsoInv(world2rig.array, currentRig2World.array);
                Mat4_AxB(modelView.array, getMouseView().getModelView()->array, world2rig.array);
            }
            else if (m_renderingMode == RenderingMode::ON_VEHICLE_STICK_TO_WORLD)
            {
                float32_t center[3] = {currentRig2World.array[0 + 3 * 4],
                                       currentRig2World.array[1 + 3 * 4],
                                       currentRig2World.array[2 + 3 * 4]};

                getMouseView().setCenter(center[0], center[1], center[2]);
                Mat4_AxB(modelView.array, getMouseView().getModelView()->array, DW_IDENTITY_TRANSFORMATION3F.array);
            }
        }

        dwRenderEngine_setModelView(&modelView, m_renderEngine);
        dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine);
        dwRenderEngine_setBackgroundColor({0.0f, 0.0f, 0.0f, 1.0f}, m_renderEngine);

        // Render trajectory path
        dwRenderEngine_setColor({0.0f, 1.0f, 0.0f, 1.0f}, m_renderEngine);
        dwRenderEngine_setLineWidth(2.f, m_renderEngine);
        dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINESTRIP_3D,
                              m_poseHistory.data(),
                              sizeof(Pose),
                              offsetof(Pose, rig2world) + 3 * 4 * sizeof(float32_t),
                              m_poseHistory.size(),
                              m_renderEngine);

        // Render vehicle coordinate system
        {
            dwRenderEngine_setLineWidth(2.0f, m_renderEngine);

            for (int i = 0; i < 3; i++)
            {
                float32_t localAxis[3] = {
                    i == 0 ? 1.f : 0.f,
                    i == 1 ? 1.f : 0.f,
                    i == 2 ? 1.f : 0.f};
                dwVector3f arrow[2];

                arrow[0].x = currentRig2World.array[0 + 3 * 4];
                arrow[0].y = currentRig2World.array[1 + 3 * 4];
                arrow[0].z = currentRig2World.array[2 + 3 * 4];

                Mat4_Axp(reinterpret_cast<float32_t*>(&arrow[1]), currentRig2World.array, localAxis);

                dwRenderEngine_setColor({localAxis[0], localAxis[1], localAxis[2], 1.0f}, m_renderEngine);
                dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_ARROWS_3D,
                                      arrow,
                                      sizeof(dwVector3f) * 2,
                                      0,
                                      1,
                                      m_renderEngine);
            }
        }

        // Render GPS north reference (if available)
        if (m_hasOrientationENU)
        {
            dwTransformation3f rig2Enu = rigidTransformation(m_orientationENU, {0.f, 0.f, 0.f});

            dwVector3f arrow[2];
            {
                dwVector3f arrowENU[2];
                dwVector3f arrowRig[2];

                arrowENU[0] = {0, 0, 0};
                arrowENU[1] = {0, 2, 0};

                Mat4_Rtxp(reinterpret_cast<float32_t*>(&arrowRig[0]), rig2Enu.array, reinterpret_cast<float32_t*>(&arrowENU[0]));
                Mat4_Rtxp(reinterpret_cast<float32_t*>(&arrowRig[1]), rig2Enu.array, reinterpret_cast<float32_t*>(&arrowENU[1]));

                Mat4_Axp(reinterpret_cast<float32_t*>(&arrow[0]), currentRig2World.array, reinterpret_cast<float32_t*>(&arrowRig[0]));
                Mat4_Axp(reinterpret_cast<float32_t*>(&arrow[1]), currentRig2World.array, reinterpret_cast<float32_t*>(&arrowRig[1]));
            }

            const char* labels[] = {"GPS NORTH"};

            dwRenderEngine_setLineWidth(2.0f, m_renderEngine);
            dwRenderEngine_setColor({0.8f, 0.3f, 0.05f, 1.0f}, m_renderEngine);
            dwRenderEngine_renderWithLabels(DW_RENDER_ENGINE_PRIMITIVE_TYPE_ARROWS_3D,
                                            arrow,
                                            sizeof(dwVector3f) * 2,
                                            0,
                                            labels,
                                            1,
                                            m_renderEngine);
        }

        // Render world and local grids
        {
            dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 0.3f}, m_renderEngine);
            dwRenderEngine_setLineWidth(0.5f, m_renderEngine);
            dwRenderEngine_renderPlanarGrid3D({0, 0, 1100.0f, 1100.0f}, 10.f, 10.f, &DW_IDENTITY_MATRIX4F, m_renderEngine);

            dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 0.1f}, m_renderEngine);
            dwRenderEngine_setLineWidth(0.25f, m_renderEngine);
            dwRenderEngine_renderPlanarGrid3D({0, 0, 1100.0f, 1100.0f}, 1.f, 1.f, &DW_IDENTITY_MATRIX4F, m_renderEngine);

            dwMatrix4f modelView = dwMakeMatrix4f(currentRig2World);
            dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, m_renderEngine);
            dwRenderEngine_setLineWidth(2.f, m_renderEngine);
            dwRenderEngine_renderPlanarGrid3D({0, 0, 2.1f, 1.0f}, 2.1f, 1.0f, &modelView, m_renderEngine);
        }
    }

    void renderText()
    {
        const dwVector2i origin = {10, 500};
        char sbuffer[256];

        dwRenderer_setFont(DW_RENDER_FONT_VERDANA_20, m_renderer);

        // Sample information
        {
            dwRenderer_renderText(origin.x, origin.y, "REAL-TIME EGOMOTION SAMPLE", m_renderer);
            dwRenderer_renderText(origin.x, origin.y - 30, "F1 - camera on rig", m_renderer);
            dwRenderer_renderText(origin.x, origin.y - 50, "F2 - camera following rig in world", m_renderer);
            dwRenderer_renderText(origin.x, origin.y - 70, "SPACE - pause", m_renderer);
            dwRenderer_renderText(origin.x, origin.y - 85, "__________________________", m_renderer);
        }

        // Motion model information
        {
            dwMotionModel motionModel;
            dwEgomotion_getMotionModel(&motionModel, m_egomotion);
            if (motionModel == DW_EGOMOTION_ODOMETRY)
                dwRenderer_renderText(origin.x, origin.y - 120, "Motion model: ODOMETRY", m_renderer);
            else if (motionModel == DW_EGOMOTION_IMU_ODOMETRY)
                dwRenderer_renderText(origin.x, origin.y - 120, "Motion model: ODOMETRY+IMU", m_renderer);
        }

        // Speed measurement type
        {
            if (m_egomotionParameters.speedMeasurementType == DW_EGOMOTION_FRONT_SPEED)
                dwRenderer_renderText(origin.x, origin.y - 140, "Speed measurement: front linear speed", m_renderer);
            else if (m_egomotionParameters.speedMeasurementType == DW_EGOMOTION_REAR_SPEED)
                dwRenderer_renderText(origin.x, origin.y - 140, "Speed measurement: rear linear speed", m_renderer);
            else if (m_egomotionParameters.speedMeasurementType == DW_EGOMOTION_REAR_WHEEL_SPEED)
                dwRenderer_renderText(origin.x, origin.y - 140, "Speed measurement: rear wheel angular speed", m_renderer);
        }

        // Suspension model status
        {
            if (m_egomotionParameters.suspension.model == DW_EGOMOTION_SUSPENSION_TORSIONAL_SPRING_MODEL)
                dwRenderer_renderText(origin.x, origin.y - 160, "Suspension modeling: enabled", m_renderer);
            else
                dwRenderer_renderText(origin.x, origin.y - 160, "Suspension modeling: disabled", m_renderer);
        }

        // Real-time egomotion state
        {
            dwEgomotionResult state{};
            dwEgomotion_getEstimation(&state, m_egomotion);

            dwEgomotionUncertainty uncertainty{};
            dwEgomotion_getUncertainty(&uncertainty, m_egomotion);

            sprintf(sbuffer, "Real-time processing time: %.1f s", m_elapsedTime * 1e-6);
            dwRenderer_renderText(origin.x, origin.y - 195, sbuffer, m_renderer);

            static constexpr uint32_t VEL = DW_EGOMOTION_LIN_VEL_X;
            if ((state.validFlags & VEL) == VEL)
            {
                static float32_t oldSpeed = std::numeric_limits<float32_t>::max();
                static float32_t olddVdt = 0;
                static dwTime_t oldTimestamp = 0;

                float32_t speed = sqrt(state.linearVelocity[0] * state.linearVelocity[0] + 
                                     state.linearVelocity[1] * state.linearVelocity[1]);
                float32_t dVdt = 0;

                if (state.timestamp != oldTimestamp)
                {
                    dVdt = (speed - oldSpeed) / (static_cast<float32_t>(state.timestamp - oldTimestamp) / 1000000.f);
                    oldTimestamp = state.timestamp;
                    oldSpeed = speed;
                    olddVdt = dVdt;
                }

                sprintf(sbuffer, "Speed: %.2f m/s (%.2f km/h), rate: %.2f m/s^2", speed, speed * 3.6, olddVdt);
                dwRenderer_renderText(origin.x, origin.y - 220, sbuffer, m_renderer);

                // Display velocity components
                const auto printLinear = [this, &sbuffer](int32_t x, int32_t y, const char* name,
                                                          float32_t value, float32_t rate,
                                                          bool printLinear, bool printRate) {
                    int32_t len = 0;
                    len += sprintf(sbuffer, "%s: ", name);

                    if (printLinear)
                        len += sprintf(sbuffer + len, "%.2f m/s, ", value);

                    if (printRate)
                        len += sprintf(sbuffer + len, "rate: %.2f m/s^2", rate);

                    dwRenderer_renderText(x, y, sbuffer, m_renderer);
                };

                printLinear(origin.x, origin.y - 260, "V_x",
                           state.linearVelocity[0], state.linearAcceleration[0],
                           (state.validFlags & DW_EGOMOTION_LIN_VEL_X) != 0,
                           (state.validFlags & DW_EGOMOTION_LIN_ACC_X) != 0);

                printLinear(origin.x, origin.y - 280, "V_y",
                           state.linearVelocity[1], state.linearAcceleration[1],
                           (state.validFlags & DW_EGOMOTION_LIN_VEL_Y) != 0,
                           (state.validFlags & DW_EGOMOTION_LIN_ACC_Y) != 0);

                if (state.validFlags & DW_EGOMOTION_LIN_VEL_Z)
                {
                    printLinear(origin.x, origin.y - 300, "V_z",
                               state.linearVelocity[2], state.linearAcceleration[2],
                               (state.validFlags & DW_EGOMOTION_LIN_VEL_Z) != 0,
                               (state.validFlags & DW_EGOMOTION_LIN_ACC_Z) != 0);
                }
            }
            else
                dwRenderer_renderText(origin.x, origin.y - 220, "Speed: not supported", m_renderer);

            // Display orientation information
            const auto printAngle = [this, &sbuffer](int32_t x, int32_t y, const char* name,
                                                     float32_t value, float32_t std, float32_t rate,
                                                     bool printAngle, bool printStd, bool printRate) {
                int32_t len = 0;
                len += sprintf(sbuffer, "%s: ", name);

                if (printAngle)
                {
                    if (printStd)
                        len += sprintf(sbuffer + len, "%.2f +/- %.2f deg, ", RAD2DEG(value), RAD2DEG(std));
                    else
                        len += sprintf(sbuffer + len, "%.2f deg, ", RAD2DEG(value));
                }

                if (printRate)
                    len += sprintf(sbuffer + len, "rate: %.2f deg/s", RAD2DEG(rate));

                dwRenderer_renderText(x, y, sbuffer, m_renderer);
            };

            float32_t roll, pitch, yaw;
            quaternionToEulerAngles(state.rotation, roll, pitch, yaw);

            printAngle(origin.x, origin.y - 320, "Roll",
                      roll, std::sqrt(uncertainty.rotation.array[0]), state.angularVelocity[0],
                      (state.validFlags & DW_EGOMOTION_ROTATION) != 0,
                      (uncertainty.validFlags & DW_EGOMOTION_ROTATION) != 0,
                      (state.validFlags & DW_EGOMOTION_ANG_VEL_X) != 0);

            printAngle(origin.x, origin.y - 340, "Pitch",
                      pitch, std::sqrt(uncertainty.rotation.array[3 + 1]), state.angularVelocity[1],
                      (state.validFlags & DW_EGOMOTION_ROTATION) != 0,
                      (uncertainty.validFlags & DW_EGOMOTION_ROTATION) != 0,
                      (state.validFlags & DW_EGOMOTION_ANG_VEL_Y) != 0);

            printAngle(origin.x, origin.y - 360, "Yaw",
                      yaw, 0, state.angularVelocity[2],
                      (state.validFlags & DW_EGOMOTION_ROTATION) != 0,
                      false,
                      (state.validFlags & DW_EGOMOTION_ANG_VEL_Z) != 0);

            if ((state.validFlags & DW_EGOMOTION_ROTATION) != 0)
            {
                sprintf(sbuffer, "Rotation relative to starting pose (t=0)");
            }

            dwRenderer_renderText(origin.x, origin.y - 400, sbuffer, m_renderer);

            // GPS information
            dwGlobalEgomotionResult globalResult{};
            dwGlobalEgomotion_getEstimate(&globalResult, nullptr, m_globalEgomotion);

            if (globalResult.validPosition)
            {
                sprintf(sbuffer, "Longitude: %.5f deg", globalResult.position.lon);
                dwRenderer_renderText(origin.x, origin.y - 440, sbuffer, m_renderer);

                sprintf(sbuffer, "Latitude: %.5f deg", globalResult.position.lat);
                dwRenderer_renderText(origin.x, origin.y - 460, sbuffer, m_renderer);

                sprintf(sbuffer, "Altitude: %.2f m", globalResult.position.height);
                dwRenderer_renderText(origin.x, origin.y - 480, sbuffer, m_renderer);
            }
            else
            {
                sprintf(sbuffer, "GPS: not available");
                dwRenderer_renderText(origin.x, origin.y - 440, sbuffer, m_renderer);
            }
        }
        
        // Real-time CAN parser status
        if (m_canParser && m_canParser->isInitialized()) 
        {
            const auto& diagnostics = m_canParser->getDiagnostics();
            const auto& config = m_canParser->getConfiguration();
            
            sprintf(sbuffer, "Real-time CAN Parser Status:");
            dwRenderer_renderText(origin.x, origin.y - 520, sbuffer, m_renderer);
            
            sprintf(sbuffer, "State Valid: %s", m_canParser->hasValidState() ? "YES" : "NO");
            dwRenderer_renderText(origin.x, origin.y - 540, sbuffer, m_renderer);
            
            sprintf(sbuffer, "Messages: %u processed, %u rejected, %.1f Hz", 
                    diagnostics.validCANMessagesProcessed.load(), 
                    diagnostics.invalidCANMessagesRejected.load(),
                    diagnostics.averageMessageRate.load());
            dwRenderer_renderText(origin.x, origin.y - 560, sbuffer, m_renderer);
            
            sprintf(sbuffer, "State Commits: %u successful, %u failed, %.1f Hz",
                    diagnostics.stateCommitsSuccessful.load(),
                    diagnostics.stateCommitsFailed.load(),
                    diagnostics.averageCommitRate.load());
            dwRenderer_renderText(origin.x, origin.y - 580, sbuffer, m_renderer);
            
            if (diagnostics.speedMessageTimeout.load() || diagnostics.steeringMessageTimeout.load() || 
                diagnostics.wheelSpeedMessageTimeout.load()) {
                dwRenderer_renderText(origin.x, origin.y - 600, "⚠ MESSAGE TIMEOUTS DETECTED", m_renderer);
            }
            
            // Show current vehicle state
            if (m_canParser->hasValidState()) {
                const auto& safetyState = m_canParser->getSafetyState();
                const auto& nonSafetyState = m_canParser->getNonSafetyState();
                
                sprintf(sbuffer, "Vehicle: Speed=%.2f m/s, Steering=%.1f°", 
                        nonSafetyState.speedESC, 
                        safetyState.steeringWheelAngle * 180.0f / M_PI);
                dwRenderer_renderText(origin.x, origin.y - 620, sbuffer, m_renderer);
            }
        }
    }

    void renderPlots()
    {
        // Implementation similar to reference but optimized for real-time display
        if (!m_poseHistory.empty())
        {
            std::vector<dwVector2f> roll, rollUncertaintyPlus, rollUncertaintyMinus;
            std::vector<dwVector2f> pitch, pitchUncertaintyPlus, pitchUncertaintyMinus;
            std::vector<dwVector2f> altitude;

            float32_t negInf = -std::numeric_limits<float32_t>::infinity();
            float32_t posInf = std::numeric_limits<float32_t>::infinity();

            dwTime_t startTime = m_poseHistory.front().timestamp;
            dwTime_t lastTime = m_poseHistory.back().timestamp;

            for (const auto& pose : m_poseHistory)
            {
                float32_t dt = float32_t((pose.timestamp - startTime) * 1e-6);

                if (lastTime - pose.timestamp < 240 * 1e6)
                {
                    altitude.push_back({dt, float32_t(pose.rig2world.array[2 + 3 * 4])});
                }

                if (lastTime - pose.timestamp < 5 * 1e6)
                {
                    roll.push_back({dt, RAD2DEG(pose.rpy[0])});
                    pitch.push_back({dt, RAD2DEG(pose.rpy[1])});

                    if (pose.uncertainty.validFlags & DW_EGOMOTION_ROTATION)
                    {
                        rollUncertaintyPlus.push_back({dt, RAD2DEG(pose.rpy[0]) + RAD2DEG(std::sqrt(pose.uncertainty.rotation.array[0]))});
                        rollUncertaintyMinus.push_back({dt, RAD2DEG(pose.rpy[0]) - RAD2DEG(std::sqrt(pose.uncertainty.rotation.array[0]))});
                        pitchUncertaintyPlus.push_back({dt, RAD2DEG(pose.rpy[1]) + RAD2DEG(std::sqrt(pose.uncertainty.rotation.array[3 + 1]))});
                        pitchUncertaintyMinus.push_back({dt, RAD2DEG(pose.rpy[1]) - RAD2DEG(std::sqrt(pose.uncertainty.rotation.array[3 + 1]))});
                    }
                }
            }

            // Roll plot
            if (!roll.empty()) {
                dwRenderEnginePlotType types[] = {DW_RENDER_ENGINE_PLOT_TYPE_LINESTRIP, DW_RENDER_ENGINE_PLOT_TYPE_LINESTRIP, DW_RENDER_ENGINE_PLOT_TYPE_LINESTRIP};
                const void* data[] = {roll.data(), rollUncertaintyPlus.data(), rollUncertaintyMinus.data()};
                uint32_t strides[] = {sizeof(dwVector2f), sizeof(dwVector2f), sizeof(dwVector2f)};
                uint32_t offsets[] = {0, 0, 0};
                uint32_t counts[] = {uint32_t(roll.size()), uint32_t(rollUncertaintyPlus.size()), uint32_t(rollUncertaintyMinus.size())};
                dwRenderEngineColorRGBA colors[] = {{1.0f, 0.0f, 0.0f, 1.0f},
                                                    {1.0f, 0.0f, 0.0f, 0.5f},
                                                    {1.0f, 0.0f, 0.0f, 0.5f}};
                float32_t widths[] = {2.0f, 1.0f, 1.0f};
                const char* labels[] = {"roll", "", ""};

                dwRenderEngine_setTile(m_tileRollPlot, m_renderEngine);
                dwRenderEngine_setLineWidth(1.0f, m_renderEngine);
                dwRenderEngine_renderPlots2D(types,
                                             data, strides, offsets, counts,
                                             colors, widths, labels,
                                             counts[1] > 0 ? 3 : 1,
                                             {negInf, -10.f, posInf, 10.f},
                                             {0.0f, 0.0f, 1.0f, 1.0f},
                                             {0.5f, 0.4f, 0.2f, 1.0f},
                                             1.f,
                                             "", " time", "[deg]",
                                             m_renderEngine);
            }

            // Pitch plot  
            if (!pitch.empty()) {
                dwRenderEnginePlotType types[] = {DW_RENDER_ENGINE_PLOT_TYPE_LINESTRIP, DW_RENDER_ENGINE_PLOT_TYPE_LINESTRIP, DW_RENDER_ENGINE_PLOT_TYPE_LINESTRIP};
                const void* data[] = {pitch.data(), pitchUncertaintyPlus.data(), pitchUncertaintyMinus.data()};
                uint32_t strides[] = {sizeof(dwVector2f), sizeof(dwVector2f), sizeof(dwVector2f)};
                uint32_t offsets[] = {0, 0, 0};
                uint32_t counts[] = {uint32_t(pitch.size()), uint32_t(pitchUncertaintyPlus.size()), uint32_t(pitchUncertaintyMinus.size())};
                dwRenderEngineColorRGBA colors[] = {{0.0f, 1.0f, 0.0f, 1.0f},
                                                    {0.0f, 1.0f, 0.0f, 0.5f},
                                                    {0.0f, 1.0f, 0.0f, 0.5f}};
                float32_t widths[] = {2.0f, 1.0f, 1.0f};
                const char* labels[] = {"pitch", "", ""};

                dwRenderEngine_setTile(m_tilePitchPlot, m_renderEngine);
                dwRenderEngine_setLineWidth(1.0f, m_renderEngine);
                dwRenderEngine_renderPlots2D(types,
                                             data, strides, offsets, counts,
                                             colors, widths, labels,
                                             counts[1] > 0 ? 3 : 1,
                                             {negInf, -10.f, posInf, 10.f},
                                             {0.0f, 0.0f, 1.0f, 1.0f},
                                             {0.5f, 0.4f, 0.2f, 1.0f},
                                             1.f,
                                             "", " time", "[deg]",
                                             m_renderEngine);
            }

            // Altitude plot
            if (!altitude.empty()) {
                dwRenderEngine_setTile(m_tileAltitudePlot, m_renderEngine);
                dwRenderEngine_setLineWidth(1.0f, m_renderEngine);
                dwRenderEngine_renderPlot2D(DW_RENDER_ENGINE_PLOT_TYPE_LINESTRIP,
                                            altitude.data(),
                                            sizeof(dwVector2f),
                                            0,
                                            altitude.size(),
                                            "Altitude",
                                            {negInf, altitude.back().y - 5.f, posInf, altitude.back().y + 5.f},
                                            {0.0f, 0.0f, 1.0f, 1.0f},
                                            {0.5f, 0.4f, 0.2f, 1.0f},
                                            1.f,
                                            "", " time", "[m]",
                                            m_renderEngine);
            }
        }

        dwRenderEngine_setTile(m_tileGrid, m_renderEngine);
    }

    void onRender() override
    {
        if (!isPaused() && !m_shallRender)
            return;

        m_shallRender = false;

        if (isOffscreen())
            return;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        render3DGrid();

        if (m_lastGLFrame != DW_NULL_HANDLE)
            {
                dwImageGL* glFrame = nullptr;
                dwImage_getGL(&glFrame, m_lastGLFrame);   // borrow pointer for draw

                if (glFrame)
                {
                    dwRenderEngine_setTile(m_tileVideo, m_renderEngine);
                    dwVector2f range{float(glFrame->prop.width), float(glFrame->prop.height)};
                    dwRenderEngine_setCoordinateRange2D(range, m_renderEngine);
                    dwRenderEngine_renderImage2D(glFrame, {0.0f, 0.0f, range.x, range.y}, m_renderEngine);
                }

                // Return GL frame to the streamer so the pool doesn’t fill up
                m_streamerInput2GL->release();
                m_lastGLFrame = DW_NULL_HANDLE;
            }

        renderText();
        renderPlots();

        renderutils::renderFPS(m_renderEngine, getCurrentFPS());
    }

    /// Real-time CAN message processing with synchronized state commits
    void onProcess() override
{
    // Drain events for a short budget each tick so we never starve subscribers
    auto budgetEnd = std::chrono::steady_clock::now() + std::chrono::milliseconds(2);
    int processed = 0;
    while (!isPaused())
    {
        if (processed >= 64 || std::chrono::steady_clock::now() >= budgetEnd)
            break;

        dwStatus status = dwSensorManager_acquireNextEvent(&acquiredEvent, 0, m_sensorManager);

        if (status == DW_TIME_OUT)
            break;

        if (status != DW_SUCCESS)
        {
            if (status != DW_END_OF_STREAM)
                logError("Real-time sensor error: %s\n", dwGetStatusName(status));
            else if (should_AutoExit())
            {
                log("End of stream reached, stopping real-time processing\n");
                stop();
                return;
            }
            pause();
            if (isOffscreen())
                stop();

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            break;
        }

        dwTime_t timestamp = acquiredEvent->timestamp_us;

        if (m_firstTimestamp == 0)
        {
            m_firstTimestamp = timestamp;
            m_lastSampleTimestamp = timestamp;
        }

        // Process sensor events for real-time data
        switch (acquiredEvent->type)
        {
        case DW_SENSOR_CAN:
        {
            if (acquiredEvent->sensorIndex != m_vehicleSensorIdx)
                break;
            
            m_canParser->processCANFrame(acquiredEvent->canFrame); // Feed raw frame to parser

            // Periodic timeout check
            static dwTime_t lastTimeoutCheck = 0;
            if (timestamp - lastTimeoutCheck > 50000) {
                m_canParser->checkMessageTimeouts(timestamp);
                lastTimeoutCheck = timestamp;
            }

            // Check for synchronized state (like VehicleIO.getState())
            dwVehicleIOSafetyState safetyState;
            dwVehicleIONonSafetyState nonSafetyState;

            if (m_canParser->getTemporallySynchronizedState(&safetyState, &nonSafetyState)) {
                // Validate before feeding to egomotion
                if (nonSafetyState.speedESC >= 0.0f && nonSafetyState.speedESC < 100.0f) {
                    CHECK_DW_ERROR(dwEgomotion_addVehicleIOState(&safetyState, &nonSafetyState, nullptr, m_egomotion));
                    
                    static dwTime_t lastCommitLog = 0;
                    if (allowEvery(timestamp, lastCommitLog, 200000)) {
                        char buf[256];
                        sprintf(buf, "✓ Synchronized state fed to egomotion: Speed=%.2f m/s, Steering=%.1f°\n",
                                nonSafetyState.speedESC, safetyState.steeringWheelAngle * 180.0f / M_PI);
                        printColored(stdout, COLOR_GREEN, buf);
                    }
                }
            }

            m_elapsedTime = timestamp - m_firstTimestamp;
            break;
        }

        case DW_SENSOR_IMU:
        {
            if (acquiredEvent->sensorIndex != m_imuSensorIdx)
                break;

            m_currentIMUFrame = acquiredEvent->imuFrame;

            // ---- RATE-LIMITED IMU LOG (5 Hz) ----
            static dwTime_t lastImuLog = 0;
            if (allowEvery(timestamp, lastImuLog, 200000)) {
                char buffer[512];
                sprintf(buffer, "IMU Data: Accel=[%.3f, %.3f, %.3f] m/s², Gyro=[%.3f, %.3f, %.3f] rad/s, Timestamp=%lu\n",
                        m_currentIMUFrame.acceleration[0], m_currentIMUFrame.acceleration[1], m_currentIMUFrame.acceleration[2],
                        m_currentIMUFrame.turnrate[0], m_currentIMUFrame.turnrate[1], m_currentIMUFrame.turnrate[2],
                        m_currentIMUFrame.timestamp_us);
                printColored(stdout, COLOR_DEFAULT, buffer);
            }

            if (m_egomotionParameters.motionModel != DW_EGOMOTION_ODOMETRY)
            {
                dwEgomotion_addIMUMeasurement(&m_currentIMUFrame, m_egomotion);
            }
            m_elapsedTime = timestamp - m_firstTimestamp;
            break;
        }

        case DW_SENSOR_GPS:
        {
            if (acquiredEvent->sensorIndex != m_gpsSensorIdx)
                break;

            m_currentGPSFrame = acquiredEvent->gpsFrame;

            // ---- RATE-LIMITED GPS LOG (5 Hz) ----
            static dwTime_t lastGpsLog = 0;
            if (allowEvery(timestamp, lastGpsLog, 200000)) {
                char buffer[512];
                sprintf(buffer, "GPS Data: Lat=%.6f°, Lon=%.6f°, Alt=%.2fm, Speed=%.2fm/s, Course=%.1f°, Timestamp=%lu\n",
                        m_currentGPSFrame.latitude, m_currentGPSFrame.longitude, m_currentGPSFrame.altitude,
                        m_currentGPSFrame.speed, m_currentGPSFrame.course, m_currentGPSFrame.timestamp_us);
                printColored(stdout, COLOR_GREEN, buffer);
            }

            m_trajectoryLog.addWGS84("GPS", m_currentGPSFrame);
            CHECK_DW_ERROR(dwGlobalEgomotion_addGPSMeasurement(&m_currentGPSFrame, m_globalEgomotion));
            m_elapsedTime = timestamp - m_firstTimestamp;
            break;
        }

        case DW_SENSOR_CAMERA:
        {
            // Keep at most one GL frame in flight; drop if renderer hasn’t caught up
            if (m_lastGLFrame == DW_NULL_HANDLE)
            {
                dwImageHandle_t nextFrame = DW_NULL_HANDLE;
                dwSensorCamera_getImage(&nextFrame, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, acquiredEvent->camFrames[0]);

                m_lastGLFrame = m_streamerInput2GL->post(nextFrame);

                // Destroy wrapper handle (not the underlying buffer)
                dwImage_destroy(nextFrame);

                m_shallRender = true;
            }
            break;
        }

        default: break;
        }

        // Return event to SensorManager
        dwSensorManager_releaseAcquiredEvent(acquiredEvent, m_sensorManager);
        acquiredEvent = nullptr;

        // Pose estimation & trajectory
        dwEgomotionResult estimate;
        dwEgomotionUncertainty uncertainty;
        if (dwEgomotion_getEstimation(&estimate, m_egomotion) == DW_SUCCESS &&
            dwEgomotion_getUncertainty(&uncertainty, m_egomotion) == DW_SUCCESS)
        {
            dwGlobalEgomotion_addRelativeMotion(&estimate, &uncertainty, m_globalEgomotion);

            if (estimate.timestamp >= m_lastSampleTimestamp + POSE_SAMPLE_PERIOD)
            {
                dwTransformation3f rigLast2rigNow;
                dwStatus st = dwEgomotion_computeRelativeTransformation(&rigLast2rigNow, nullptr,
                                                                        m_lastSampleTimestamp, estimate.timestamp, m_egomotion);
                if (st == DW_SUCCESS)
                {
                    Pose pose{};
                    quaternionToEulerAngles(estimate.rotation, pose.rpy[0], pose.rpy[1], pose.rpy[2]);

                    dwTransformation3f rigLast2world = DW_IDENTITY_TRANSFORMATION3F;
                    if (!m_poseHistory.empty())
                        rigLast2world = m_poseHistory.back().rig2world;
                    else if ((estimate.validFlags & DW_EGOMOTION_ROTATION) != 0)
                    {
                        dwMatrix3f rot{};
                        getRotationMatrix(&rot, RAD2DEG(pose.rpy[0]), RAD2DEG(pose.rpy[1]), 0);
                        rotationToTransformMatrix(rigLast2world.array, rot.array);
                    }

                    dwTransformation3f rigNow2World;
                    dwEgomotion_applyRelativeTransformation(&rigNow2World, &rigLast2rigNow, &rigLast2world);

                    pose.rig2world = rigNow2World;
                    pose.timestamp = estimate.timestamp;

                    if (m_poseHistory.size() > MAX_BUFFER_POINTS)
                    {
                        decltype(m_poseHistory) tmp;
                        tmp.assign(++m_poseHistory.begin(), m_poseHistory.end());
                        std::swap(tmp, m_poseHistory);
                    }

                    if (m_outputFile)
                        fprintf(m_outputFile, "%lu,%.2f,%.2f,%.2f\n", estimate.timestamp,
                                rigNow2World.array[0 + 3 * 4], rigNow2World.array[1 + 3 * 4], rigNow2World.array[2 + 3 * 4]);

                    dwGlobalEgomotionResult absoluteEstimate{};
                    if (dwGlobalEgomotion_getEstimate(&absoluteEstimate, nullptr, m_globalEgomotion) == DW_SUCCESS &&
                        absoluteEstimate.timestamp == estimate.timestamp && absoluteEstimate.validOrientation)
                    {
                        m_orientationENU = absoluteEstimate.orientation;
                        m_hasOrientationENU = true;

                        if (m_trajectoryLog.size("Egomotion") == 0)
                            m_trajectoryLog.addWGS84("Egomotion", m_currentGPSFrame);

                        m_trajectoryLog.addWGS84("Egomotion", absoluteEstimate.position);
                    }
                    else
                    {
                        m_hasOrientationENU = false;
                    }

                    dwEgomotion_getUncertainty(&pose.uncertainty, m_egomotion);
                    m_poseHistory.push_back(pose);
                    m_shallRender = true;
                }
                m_lastSampleTimestamp = estimate.timestamp;
            }
        }

        ++processed;
    }
}


    void onKeyDown(int key, int scancode, int mods) override
    {
        (void)scancode;
        (void)mods;

        if (key == GLFW_KEY_F1)
        {
            getMouseView().setCenter(0, 0, 0);
            m_renderingMode = RenderingMode::STICK_TO_VEHICLE;
        }

        if (key == GLFW_KEY_F2)
        {
            m_renderingMode = RenderingMode::ON_VEHICLE_STICK_TO_WORLD;
        }
    }
};

//------------------------------------------------------------------------------
int main(int argc, const char* argv[])
{
    const std::string samplePath = dw_samples::SamplesDataPath::get() + "/samples/recordings/cloverleaf/";

    ProgramArguments args(argc, argv,
                          {
                              ProgramArguments::Option_t("vehicle-sensor-name", "", "[optional] Name of the vehicle sensor in rig file for real-time processing."),
                              ProgramArguments::Option_t("imu-sensor-name", "", "[optional] Name of the IMU sensor in rig file for real-time processing."),
                              ProgramArguments::Option_t("gps-sensor-name", "", "[optional] Name of the GPS sensor in rig file for real-time processing."),
                              ProgramArguments::Option_t("camera-sensor-name", "", "[optional] Name of the camera sensor in rig file for real-time processing."),

                              ProgramArguments::Option_t("rig", (samplePath + "rig-nominal-intrinsics.json").c_str(),
                                                         "Rig file containing sensor and vehicle configuration for real-time processing."),

                              ProgramArguments::Option_t("output", "", "If specified, real-time trajectory will be output to this file."),
                              ProgramArguments::Option_t("outputkml", "", "If specified, real-time GPS and estimated trajectories will be output to this KML file"),
                              ProgramArguments::Option_t("mode", "1", "0=Ackerman motion, 1=IMU+Odometry+GPS for real-time processing"),
                              ProgramArguments::Option_t("speed-measurement-type", "1", "Speed measurement type for real-time processing, refer to dwEgomotionSpeedMeasurementType"),
                              ProgramArguments::Option_t("enable-suspension", "0", "If 1, enables egomotion suspension modeling for real-time processing (requires Odometry+IMU [--mode=1]), otherwise disabled."),
                          },
                          "DriveWorks real-time egomotion sample with synchronized CAN processing");

    EgomotionSample app(args);

    app.initializeWindow("Real-time Egomotion Sample", 1920, 1080, args.enabled("offscreen"));

    if (!args.enabled("offscreen"))
        app.setProcessRate(240);
    
    log("Starting real-time egomotion processing...\n");
    return app.run();
}