/////////////////////////////////////////////////////////////////////////////////////////
//
// PointPillar LiDAR 3D Object Detection Sample
// Combines VLP-16 LiDAR replay with real-time 3D object detection
//
/////////////////////////////////////////////////////////////////////////////////////////

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
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

#include "PointPillarDetector.hpp"

using namespace dw_samples::common;
using namespace dw_samples::pointpillar;

//------------------------------------------------------------------------------
// PointPillar LiDAR Detection Sample
//------------------------------------------------------------------------------
class PointPillarLidarSample : public DriveWorksSample
{
private:
    std::string m_outputDir = "";

    dwContextHandle_t m_context = DW_NULL_HANDLE;
    dwSALHandle_t m_sal         = DW_NULL_HANDLE;

    // LiDAR components
    dwSensorHandle_t m_lidarSensor = DW_NULL_HANDLE;
    dwLidarProperties m_lidarProperties{};
    bool m_recordedLidar = false;
    std::unique_ptr<float32_t[]> m_pointCloud;

    // PointPillar detector
    std::unique_ptr<PointPillarDetector> m_detector;
    std::vector<BoundingBox3D> m_detections;
    cudaStream_t m_cudaStream = 0;

    // Rendering
    dwVisualizationContextHandle_t m_visualizationContext = DW_NULL_HANDLE;
    dwRenderEngineColorByValueMode m_colorByValueMode = DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_XY;

    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
    uint32_t m_gridBuffer                 = 0;
    uint32_t m_gridBufferPrimitiveCount   = 0;
    uint32_t m_pointCloudBuffer           = 0;
    uint32_t m_pointCloudBufferCapacity   = 0;
    uint32_t m_pointCloudBufferSize       = 0;
    uint32_t m_boundingBoxBuffer          = 0;
    uint32_t m_boundingBoxBufferCapacity  = 0;

    // Statistics
    std::string m_message1;
    std::string m_message2;
    std::string m_message3;
    std::string m_message4;
    std::string m_message5;
    std::string m_message6;
    std::string m_message7;
    std::string m_message8;
    std::string m_message9;

public:
    PointPillarLidarSample(const ProgramArguments& args)
        : DriveWorksSample(args) {}

    void initializeDriveWorks(dwContextHandle_t& context)
    {
        // Initialize logger
        CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
        CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));

        // Initialize SDK context
        dwContextParameters sdkParams = {};

#ifdef VIBRANTE
        sdkParams.eglDisplay = getEGLDisplay();
#endif

        CHECK_DW_ERROR(dwInitialize(&context, DW_VERSION, &sdkParams));
        CHECK_DW_ERROR(dwSAL_initialize(&m_sal, context));

        // Setup LiDAR sensor
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
        CHECK_DW_ERROR(dwSensorLidar_getProperties(&m_lidarProperties, m_lidarSensor));

        // Allocate point cloud buffer
        m_pointCloudBufferCapacity = m_lidarProperties.pointsPerSecond;
        m_pointCloud.reset(new float32_t[m_pointCloudBufferCapacity * m_lidarProperties.pointStride]);
    }

    void initializeDetector()
    {
        log("Initializing PointPillar Detector...\n");

        // Create CUDA stream
        cudaStreamCreate(&m_cudaStream);

        // Configure detector
        PointPillarDetector::Config config;
        config.modelPath = dw_samples::SamplesDataPath::get() + 
                          "/samples/detector/ampere-integrated/pointpillars_deployable_fp32.bin";
        config.confidenceThreshold = 0.3f;  // 30% confidence threshold
        config.nmsIouThreshold = 0.01f;     // NMS threshold from NGC
        config.maxDetections = 100;
        config.maxInputPoints = 204800;

        // Override with command line arguments if provided
        std::string modelPath = getArgument("model");
        if (!modelPath.empty()) {
            config.modelPath = modelPath;
        }

        std::string thresholdStr = getArgument("threshold");
        if (!thresholdStr.empty()) {
            config.confidenceThreshold = std::stof(thresholdStr);
        }

        // Create detector
        m_detector.reset(new PointPillarDetector(config, m_context, m_cudaStream));

        log("PointPillar Detector initialized.\n");
    }

    /// -----------------------------
    /// Initialize everything
    /// -----------------------------
    bool onInitialize() override
    {
        log("Starting PointPillar LiDAR Detection Sample...\n");

        initializeDriveWorks(m_context);
        initializeDetector();

        dwVisualizationInitialize(&m_visualizationContext, m_context);

        // Initialize RenderEngine
        dwRenderEngineParams renderEngineParams{};
        CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&renderEngineParams,
                                                        getWindowWidth(),
                                                        getWindowHeight()));
        renderEngineParams.defaultTile.backgroundColor = {0.0f, 0.0f, 0.1f, 1.0f};
        CHECK_DW_ERROR_MSG(dwRenderEngine_initialize(&m_renderEngine, &renderEngineParams, m_visualizationContext),
                           "Cannot initialize Render Engine");

        // Create point cloud buffer
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_pointCloudBuffer, 
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                   sizeof(dwVector3f), 0, m_pointCloudBufferCapacity, 
                                                   m_renderEngine));

        // Create grid buffer
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_gridBuffer, 
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                                   sizeof(dwVector3f), 0, 10000, m_renderEngine));

        dwMatrix4f identity = DW_IDENTITY_MATRIX4F;
        CHECK_DW_ERROR(dwRenderEngine_setBufferPlanarGrid3D(m_gridBuffer, {0.f, 0.f, 100.f, 100.f},
                                                            5.0f, 5.0f,
                                                            &identity, m_renderEngine));

        dwRenderEngine_getBufferMaxPrimitiveCount(&m_gridBufferPrimitiveCount, m_gridBuffer, m_renderEngine);

        // Create bounding box buffer (12 lines per box * 100 boxes * 2 vertices per line)
        m_boundingBoxBufferCapacity = 100 * 12 * 2;
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_boundingBoxBuffer,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                                   sizeof(dwVector3f), 0, m_boundingBoxBufferCapacity,
                                                   m_renderEngine));

        dwSensor_start(m_lidarSensor);

        return true;
    }

    ///------------------------------------------------------------------------------
    /// Reset
    ///------------------------------------------------------------------------------
    void onReset() override
    {
        dwSensor_reset(m_lidarSensor);
        dwRenderEngine_reset(m_renderEngine);
        if (m_detector) {
            m_detector->reset();
        }
    }

    ///------------------------------------------------------------------------------
    /// Release resources
    ///------------------------------------------------------------------------------
    void onRelease() override
    {
        if (m_boundingBoxBuffer != 0) {
            CHECK_DW_ERROR(dwRenderEngine_destroyBuffer(m_boundingBoxBuffer, m_renderEngine));
        }
        if (m_pointCloudBuffer != 0) {
            CHECK_DW_ERROR(dwRenderEngine_destroyBuffer(m_pointCloudBuffer, m_renderEngine));
        }
        if (m_gridBuffer != 0) {
            CHECK_DW_ERROR(dwRenderEngine_destroyBuffer(m_gridBuffer, m_renderEngine));
        }

        if (m_renderEngine != DW_NULL_HANDLE) {
            CHECK_DW_ERROR(dwRenderEngine_release(m_renderEngine));
        }

        if (m_visualizationContext != DW_NULL_HANDLE) {
            CHECK_DW_ERROR(dwVisualizationRelease(m_visualizationContext));
        }

        // Release detector
        m_detector.reset();

        if (m_cudaStream) {
            cudaStreamDestroy(m_cudaStream);
        }

        if (m_lidarSensor != DW_NULL_HANDLE) {
            CHECK_DW_ERROR(dwSAL_releaseSensor(m_lidarSensor));
        }

        dwSAL_release(m_sal);

        if (m_context != DW_NULL_HANDLE) {
            CHECK_DW_ERROR(dwRelease(m_context));
        }

        CHECK_DW_ERROR(dwLogger_release());
    }

    ///------------------------------------------------------------------------------
    /// Resize window
    ///------------------------------------------------------------------------------
    void onResizeWindow(int width, int height) override
    {
        dwRectf bounds{.x = 0.0f, .y = 0.0f};
        bounds.width  = width;
        bounds.height = height;
        dwRenderEngine_setBounds(bounds, m_renderEngine);
    }

    ///------------------------------------------------------------------------------
    /// Update frame with detections
    ///------------------------------------------------------------------------------
    void updateFrame(uint32_t accumulatedPoints, uint32_t packetCount,
                     dwTime_t hostTimestamp, dwTime_t sensorTimestamp)
    {
        m_pointCloudBufferSize = accumulatedPoints;
        dwSensorLidar_getProperties(&m_lidarProperties, m_lidarSensor);

        // Update point cloud rendering
        dwRenderEngine_setBuffer(m_pointCloudBuffer,
                                 DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                 m_pointCloud.get(),
                                 sizeof(dwLidarPointXYZI),
                                 0,
                                 m_pointCloudBufferSize,
                                 m_renderEngine);

        // Run detection on the point cloud
        runDetection(accumulatedPoints);

        // Update display messages
        m_message1 = "Host timestamp    (us) " + std::to_string(hostTimestamp);
        m_message2 = "Sensor timestamp (us) " + std::to_string(sensorTimestamp);
        m_message3 = "Packets per scan         " + std::to_string(packetCount);
        m_message4 = "Points per scan           " + std::to_string(accumulatedPoints);
        m_message5 = "Frequency (Hz)           " + std::to_string(m_lidarProperties.spinFrequency);
        m_message6 = "Lidar Device               " + std::string{m_lidarProperties.deviceString};
        m_message7 = "Detections                 " + std::to_string(m_detections.size());
        m_message8 = "Inference Time (ms)     " + std::to_string(m_detector->getAverageInferenceTime());
        m_message9 = "Press ESC to exit";
    }

    ///------------------------------------------------------------------------------
    /// Run PointPillar detection
    ///------------------------------------------------------------------------------
    void runDetection(uint32_t pointCount)
    {
        if (!m_detector || pointCount == 0) {
            m_detections.clear();
            return;
        }

        // Convert point cloud to dwLidarPointXYZI format
        dwLidarPointXYZI* points = reinterpret_cast<dwLidarPointXYZI*>(m_pointCloud.get());

        // Run inference
        dwStatus status = m_detector->runInference(points, pointCount, m_detections);

        if (status != DW_SUCCESS) {
            logError("Detection failed with status: %d\n", status);
            m_detections.clear();
            return;
        }

        // Update bounding box rendering
        updateBoundingBoxes();

        // Log detections
        for (const auto& det : m_detections) {
            const char* className = PointPillarDetector::CLASS_NAMES[det.classId];
            log("%s: conf=%.2f, pos=(%.2f, %.2f, %.2f), size=(%.2f, %.2f, %.2f), yaw=%.2f\n",
                className, det.confidence,
                det.x, det.y, det.z,
                det.length, det.width, det.height,
                det.yaw);
        }
    }

    ///------------------------------------------------------------------------------
    /// Generate 3D bounding box lines for rendering
    ///------------------------------------------------------------------------------
    void updateBoundingBoxes()
    {
        if (m_detections.empty()) {
            return;
        }

        // Each box needs 12 lines (edges), each line needs 2 vertices
        std::vector<dwVector3f> boxLines;
        boxLines.reserve(m_detections.size() * 12 * 2);

        for (const auto& box : m_detections) {
            // Calculate 8 corners of the box
            float32_t halfL = box.length / 2.0f;
            float32_t halfW = box.width / 2.0f;
            float32_t halfH = box.height / 2.0f;

            float32_t cosYaw = std::cos(box.yaw);
            float32_t sinYaw = std::sin(box.yaw);

            // Define 8 corners in local frame (before rotation)
            dwVector3f corners[8];
            corners[0] = {-halfL, -halfW, -halfH};  // Bottom-back-left
            corners[1] = { halfL, -halfW, -halfH};  // Bottom-front-left
            corners[2] = { halfL,  halfW, -halfH};  // Bottom-front-right
            corners[3] = {-halfL,  halfW, -halfH};  // Bottom-back-right
            corners[4] = {-halfL, -halfW,  halfH};  // Top-back-left
            corners[5] = { halfL, -halfW,  halfH};  // Top-front-left
            corners[6] = { halfL,  halfW,  halfH};  // Top-front-right
            corners[7] = {-halfL,  halfW,  halfH};  // Top-back-right

            // Rotate and translate corners to world frame
            for (int i = 0; i < 8; i++) {
                float32_t localX = corners[i].x;
                float32_t localY = corners[i].y;
                
                // Rotate around Z-axis
                corners[i].x = localX * cosYaw - localY * sinYaw + box.x;
                corners[i].y = localX * sinYaw + localY * cosYaw + box.y;
                corners[i].z += box.z;
            }

            // Define 12 edges of the box
            int edges[12][2] = {
                {0, 1}, {1, 2}, {2, 3}, {3, 0},  // Bottom face
                {4, 5}, {5, 6}, {6, 7}, {7, 4},  // Top face
                {0, 4}, {1, 5}, {2, 6}, {3, 7}   // Vertical edges
            };

            // Add edges to line buffer
            for (int i = 0; i < 12; i++) {
                boxLines.push_back(corners[edges[i][0]]);
                boxLines.push_back(corners[edges[i][1]]);
            }
        }

        // Update render buffer
        if (!boxLines.empty()) {
            CHECK_DW_ERROR(dwRenderEngine_setBuffer(
                m_boundingBoxBuffer,
                DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                boxLines.data(),
                sizeof(dwVector3f),
                0,
                boxLines.size(),
                m_renderEngine
            ));
        }
    }

    ///------------------------------------------------------------------------------
    /// Compute spin and run detection
    ///------------------------------------------------------------------------------
    void computeSpin()
    {
        const dwLidarDecodedPacket* nextPacket;
        static uint32_t packetCount       = 0;
        static uint32_t accumulatedPoints = 0;
        static bool endOfSpin             = false;
        static auto tStart                = std::chrono::high_resolution_clock::now();
        static auto tEnd                  = tStart;

        // Throttling for recorded data
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

                // Append packet to buffer
                float32_t* map = &m_pointCloud[accumulatedPoints * m_lidarProperties.pointStride];
                memcpy(map, nextPacket->pointsXYZI, nextPacket->nPoints * sizeof(dwLidarPointXYZI));

                accumulatedPoints += nextPacket->nPoints;

                // If scan complete, update frame and run detection
                if (nextPacket->scanComplete)
                {
                    updateFrame(accumulatedPoints, packetCount, hostTimestamp, sensorTimestamp);

                    accumulatedPoints = 0;
                    packetCount       = 0;
                    endOfSpin         = true;
                    dwSensorLidar_returnPacket(nextPacket, m_lidarSensor);
                    return;
                }

                dwSensorLidar_returnPacket(nextPacket, m_lidarSensor);
            }
            else if (status == DW_END_OF_STREAM)
            {
                updateFrame(accumulatedPoints, packetCount, hostTimestamp, sensorTimestamp);

                // Reset for recorded data
                dwSensor_reset(m_lidarSensor);
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
        dwRenderEngine_reset(m_renderEngine);
        dwRenderEngine_setTile(0, m_renderEngine);
        dwRenderEngine_setModelView(getMouseView().getModelView(), m_renderEngine);
        dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine);

        // Render grid
        dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_DARKGREY, m_renderEngine);
        dwRenderEngine_setBackgroundColor({0.0f, 0.0f, 0.0f, 1.0f}, m_renderEngine);
        dwRenderEngine_renderBuffer(m_gridBuffer, m_gridBufferPrimitiveCount, m_renderEngine);

        // Render point cloud
        dwRenderEngine_setColorByValue(m_colorByValueMode, 130.0f, m_renderEngine);
        dwRenderEngine_renderBuffer(m_pointCloudBuffer, m_pointCloudBufferSize, m_renderEngine);

        // Render 3D bounding boxes
        if (!m_detections.empty()) {
            dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_RED, m_renderEngine);
            dwRenderEngine_setLineWidth(2.0f, m_renderEngine);
            dwRenderEngine_renderBuffer(m_boundingBoxBuffer, m_detections.size() * 12 * 2, m_renderEngine);
        }

        // Render text overlay
        dwRenderEngine_setProjection(&DW_IDENTITY_MATRIX4F, m_renderEngine);
        dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);
        dwVector2f range{static_cast<float32_t>(getWindowWidth()),
                         static_cast<float32_t>(getWindowHeight())};
        dwRenderEngine_setCoordinateRange2D(range, m_renderEngine);
        dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_renderEngine);
        dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_GREEN, m_renderEngine);
        
        dwRenderEngine_renderText2D(m_message1.c_str(), {20.f, getWindowHeight() - 30.f}, m_renderEngine);
        dwRenderEngine_renderText2D(m_message2.c_str(), {20.f, getWindowHeight() - 50.f}, m_renderEngine);
        dwRenderEngine_renderText2D(m_message3.c_str(), {20.f, getWindowHeight() - 70.f}, m_renderEngine);
        dwRenderEngine_renderText2D(m_message4.c_str(), {20.f, getWindowHeight() - 90.f}, m_renderEngine);
        dwRenderEngine_renderText2D(m_message5.c_str(), {20.f, getWindowHeight() - 110.f}, m_renderEngine);
        dwRenderEngine_renderText2D(m_message6.c_str(), {20.f, getWindowHeight() - 130.f}, m_renderEngine);
        
        // Detection info in cyan
        dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_CYAN, m_renderEngine);
        dwRenderEngine_renderText2D(m_message7.c_str(), {20.f, getWindowHeight() - 150.f}, m_renderEngine);
        dwRenderEngine_renderText2D(m_message8.c_str(), {20.f, getWindowHeight() - 170.f}, m_renderEngine);
        
        dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_YELLOW, m_renderEngine);
        dwRenderEngine_renderText2D(m_message9.c_str(), {20.f, 20.f}, m_renderEngine);

        renderutils::renderFPS(m_renderEngine, getCurrentFPS());
    }
};

//------------------------------------------------------------------------------
int main(int argc, const char** argv)
{
    ProgramArguments args(argc, argv,
                          {
                              ProgramArguments::Option_t("protocol", "lidar.socket", "LiDAR protocol (lidar.socket or lidar.virtual)"),
                              ProgramArguments::Option_t("params", "device=VELO_VLP16,ip=192.168.1.201,port=2368,scan-frequency=10", "LiDAR parameters"),
                              ProgramArguments::Option_t("show-intensity", "false", "Color by intensity instead of XY"),
                              ProgramArguments::Option_t("model", "", "Path to PointPillar TensorRT model (default: auto-detect)"),
                              ProgramArguments::Option_t("threshold", "0.3", "Confidence threshold for detections (0.0-1.0)")
                          },
                          "PointPillar LiDAR 3D Object Detection Sample\n"
                          "Detects vehicles, pedestrians, and cyclists from VLP-16 LiDAR data");

    PointPillarLidarSample app(args);

    app.initializeWindow("PointPillar LiDAR Detection", 1280, 800, args.enabled("offscreen"));

    return app.run();
}