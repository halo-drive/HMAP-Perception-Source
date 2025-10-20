#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <dlfcn.h>


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

// Debug logging macro
#define DEBUG_LOG(msg) { \
    auto now = std::time(nullptr); \
    auto tm_info = std::localtime(&now); \
    std::cout << "[" << std::put_time(tm_info, "%Y-%m-%d %H:%M:%S") << "] LIVOX_HAP_PLUGIN: " << msg << std::endl; \
}



using namespace dw_samples::common;

//------------------------------------------------------------------------------
// Template of a sample. Put some description what the sample does here
//------------------------------------------------------------------------------
class LidarHapReplaySample : public DriveWorksSample
{
private:
    std::string m_outputDir = "";

    dwContextHandle_t m_context = DW_NULL_HANDLE;
    dwSALHandle_t m_sal         = DW_NULL_HANDLE;

    dwSensorHandle_t m_lidarSensor = DW_NULL_HANDLE;
    dwLidarProperties m_lidarProperties{};
    bool m_recordedLidar = false;
    std::unique_ptr<float32_t[]> m_pointCloud;

    // Rendering
    dwVisualizationContextHandle_t m_visualizationContext = DW_NULL_HANDLE;

    dwRenderEngineColorByValueMode m_colorByValueMode = DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_XY;

    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
    uint32_t m_gridBuffer                 = 0;
    uint32_t m_gridBufferPrimitiveCount   = 0;
    uint32_t m_pointCloudBuffer           = 0;
    uint32_t m_pointCloudBufferCapacity   = 0; // max storage
    uint32_t m_pointCloudBufferSize       = 0; // actual size

    const bool m_isNonSpinningLidar = true;  // Set based on lidar type
    const int MIN_POINTS_FOR_DISPLAY = 1000; // Minimum points needed
    const int MAX_PACKETS_PER_FRAME = 50;    // Maximum packets per frame
    const int FRAME_TIMEOUT_MS = 100;        // Timeout for frame collection

    std::vector<dwLidarPointXYZI> m_accumulatedPoints;  // Storage for accumulated points
    uint32_t m_maxAccumulatedPoints = 20000;           // Maximum points to accumulate

    std::string m_message1;
    std::string m_message2;
    std::string m_message3;
    std::string m_message4;
    std::string m_message5;
    std::string m_message6;
    std::string m_message7;


public:
    LidarHapReplaySample(const ProgramArguments& args)
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
        CHECK_DW_ERROR(dwSAL_createSensor(&m_lidarSensor, params, m_sal)); // error here


        // Get lidar properties
        CHECK_DW_ERROR(dwSensorLidar_getProperties(&m_lidarProperties, m_lidarSensor));

        // Allocate bigger buffer in case certain spin exceeds the pointsPerSpin in lidar property
        // m_pointCloudBufferCapacity = m_lidarProperties.pointsPerSecond;
        // m_pointCloud.reset(new float32_t[m_pointCloudBufferCapacity * m_lidarProperties.pointStride]);
        m_pointCloudBufferCapacity = std::max(m_lidarProperties.pointsPerSecond, (uint32_t)(m_lidarProperties.pointsPerSpin * 1.5));
        // m_pointCloud.reset(new float32_t[m_pointCloudBufferCapacity * m_lidarProperties.pointStride]);
        m_pointCloud.reset(reinterpret_cast<float32_t*>(new dwLidarPointXYZI[m_pointCloudBufferCapacity]));

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

    void setMaxAccumulatedPoints(uint32_t maxPoints) {
        m_maxAccumulatedPoints = maxPoints;
        std::cout << "Maximum accumulated points set to: " << m_maxAccumulatedPoints << std::endl;
        
        // Clear current accumulated points to ensure clean start
        m_accumulatedPoints.clear();
        
        // Reserve space for efficiency
        m_accumulatedPoints.reserve(m_maxAccumulatedPoints);
    }

    /// -----------------------------
    /// Initialize everything of a sample here incl. SDK components
    /// -----------------------------
    bool onInitialize() override
    {
        log("Starting Livox HAP visualization application...\n");

        preparePointCloudDumps();

        initializeDriveWorks(m_context);

        dwVisualizationInitialize(&m_visualizationContext, m_context);

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

        // Initialize the accumulated points collection with our target size
        m_accumulatedPoints.clear();
        m_accumulatedPoints.reserve(m_maxAccumulatedPoints);
        
        // For Livox HAP sensors, we need a larger buffer capacity to handle accumulated points
        m_pointCloudBufferCapacity = std::max(m_lidarProperties.pointsPerSecond, 
                                            m_maxAccumulatedPoints);
        m_pointCloud.reset(reinterpret_cast<float32_t*>(new dwLidarPointXYZI[m_pointCloudBufferCapacity]));

        log("Creating point cloud buffer with capacity: %d points\n", m_pointCloudBufferCapacity);
        
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_pointCloudBuffer, DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                sizeof(dwVector3f), 0, m_pointCloudBufferCapacity, m_renderEngine));

        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_gridBuffer, DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
                                                sizeof(dwVector3f), 0, 10000, m_renderEngine));

        dwMatrix4f identity = DW_IDENTITY_MATRIX4F;
        CHECK_DW_ERROR(dwRenderEngine_setBufferPlanarGrid3D(m_gridBuffer, {0.f, 0.f, 100.f, 100.f},
                                                        5.0f, 5.0f,
                                                        &identity, m_renderEngine));

        dwRenderEngine_getBufferMaxPrimitiveCount(&m_gridBufferPrimitiveCount, m_gridBuffer, m_renderEngine);

        dwSensor_start(m_lidarSensor);

        log("Livox HAP visualization ready. Accumulating up to %d points.\n", m_maxAccumulatedPoints);

        return true;
    }

    ///------------------------------------------------------------------------------
    /// This method is executed when user presses `R`, it indicates that sample has to reset
    ///------------------------------------------------------------------------------
    void onReset() override
    {
        dwSensor_reset(m_lidarSensor);
        dwRenderEngine_reset(m_renderEngine);
    }

    ///------------------------------------------------------------------------------
    /// This method is executed on release, free up used memory here
    ///------------------------------------------------------------------------------
    void onRelease() override
    {
        if (m_pointCloudBuffer != 0)
        {
            CHECK_DW_ERROR(dwRenderEngine_destroyBuffer(m_pointCloudBuffer, m_renderEngine));
        }
        if (m_gridBuffer != 0)
        {
            CHECK_DW_ERROR(dwRenderEngine_destroyBuffer(m_gridBuffer, m_renderEngine));
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

        // -----------------------------------------
        // Release DriveWorks context and SAL
        // -----------------------------------------
        dwSAL_release(m_sal);

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
        dwRenderEngine_setBounds(bounds, m_renderEngine);
    }

    void updateFrame(uint32_t accumulatedPoints, uint32_t packetCount,
                     dwTime_t hostTimestamp, dwTime_t sensorTimestamp)
    {
        std::cout << "[VIS] Rendering " << accumulatedPoints << " points" << std::endl;

        m_pointCloudBufferSize = accumulatedPoints;
        // Grab properties, in case they were changed while running
        dwSensorLidar_getProperties(&m_lidarProperties, m_lidarSensor);

        if (!m_outputDir.empty())
        {
            dumpLidarFrame(accumulatedPoints, hostTimestamp);
        }

        dwRenderEngine_setBuffer(m_pointCloudBuffer,
                                 DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                 m_pointCloud.get(),
                                 sizeof(dwLidarPointXYZI),
                                 0,
                                 m_pointCloudBufferSize,
                                 m_renderEngine);

        m_message1 = "Host timestamp    (us) " + std::to_string(hostTimestamp);
        m_message2 = "Sensor timestamp (us) " + std::to_string(sensorTimestamp);
        m_message3 = "Packets per scan         " + std::to_string(packetCount);
        m_message4 = "Points per scan           " + std::to_string(accumulatedPoints);
        m_message5 = "Frequency (Hz)           " + std::to_string(m_lidarProperties.spinFrequency);
        m_message6 = "Lidar Device               " + std::string{m_lidarProperties.deviceString};
        // m_message7 = "Press ESC to exit";
        m_message7 = "Accumulated points: " + std::to_string(m_accumulatedPoints.size()) + 
             " of " + std::to_string(m_maxAccumulatedPoints) + " max";
    }

    void dumpLidarFrame(uint32_t accumulatedPoints, dwTime_t timestamp)
    {
        const std::string lidarFilename = std::to_string(timestamp) + ".bin";
        std::ofstream fout;
        fout.open(lidarFilename, std::ios::binary | std::ios::out);
        fout.write(reinterpret_cast<char*>(m_pointCloud.get()), accumulatedPoints * m_lidarProperties.pointStride * sizeof(float32_t));
        fout.close();
    }

    // void computeSpin()
    // {
    //     // Use the correct DriveWorks LiDAR API
    //     const dwLidarDecodedPacket* nextPacket;
        
    //     // Configurable parameters for point cloud accumulation
    //     static const int FRAME_TIMEOUT_MS = 100;     // Timeout for packet reading
    //     static uint32_t packetCount = 0;
    //     static dwTime_t lastHostTimestamp = 0;
    //     static dwTime_t lastSensorTimestamp = 0;
        
    //     dwStatus status = dwSensorLidar_readPacket(&nextPacket, FRAME_TIMEOUT_MS * 1000, m_lidarSensor);
        
    //     if (status == DW_SUCCESS) {
    //         packetCount++;
    //         lastHostTimestamp = nextPacket->hostTimestamp;
    //         lastSensorTimestamp = nextPacket->sensorTimestamp;
            
    //         std::cout << "Packet received: points=" << nextPacket->nPoints 
    //                 << ", timestamp=" << nextPacket->hostTimestamp 
    //                 << ", packet #" << packetCount << std::endl;
            
    //         // Process points if there are any
    //         if (nextPacket->nPoints > 0) {
    //             std::cout << "Processing " << nextPacket->nPoints << " points from packet" << std::endl;
                
    //             // Add new points to the accumulated buffer
    //             for (uint32_t i = 0; i < nextPacket->nPoints; i++) {
    //                 // Check if adding these points would exceed the maximum
    //                 if (m_accumulatedPoints.size() >= m_maxAccumulatedPoints) {
    //                     // If we've reached our target point count, we're ready to visualize
    //                     std::cout << "Reached target point count: " << m_accumulatedPoints.size() << std::endl;
    //                     break;
    //                 }
                    
    //                 // Add the point
    //                 m_accumulatedPoints.push_back(nextPacket->pointsXYZI[i]);
    //             }
                
    //             std::cout << "Added " << nextPacket->nPoints << " points, total accumulated: " << m_accumulatedPoints.size() << std::endl;
                
    //             // Return the packet buffer
    //             dwSensorLidar_returnPacket(nextPacket, m_lidarSensor);
                
    //             // Decide if we should update the frame based on our target point count
    //             bool frameComplete = (m_accumulatedPoints.size() >= m_maxAccumulatedPoints);
                
    //             if (frameComplete && !m_accumulatedPoints.empty()) {
    //                 std::cout << "Frame complete: " << m_accumulatedPoints.size() 
    //                         << " accumulated points in " << packetCount << " packets" << std::endl;
                            
    //                 // Copy accumulated points to rendering buffer
    //                 dwLidarPointXYZI* map = reinterpret_cast<dwLidarPointXYZI*>(m_pointCloud.get());
    //                 memcpy(map, m_accumulatedPoints.data(), m_accumulatedPoints.size() * sizeof(dwLidarPointXYZI));
                    
    //                 // Update frame with the accumulated points
    //                 updateFrame(m_accumulatedPoints.size(), packetCount, lastHostTimestamp, lastSensorTimestamp);
                    
    //                 // Don't clear the accumulated points - keep some to maintain continuity
    //                 if (m_accumulatedPoints.size() > m_maxAccumulatedPoints * 0.8) {
    //                     // Remove 20% of the oldest points to make room for new ones
    //                     size_t pointsToRemove = m_accumulatedPoints.size() * 0.2;
    //                     m_accumulatedPoints.erase(
    //                         m_accumulatedPoints.begin(),
    //                         m_accumulatedPoints.begin() + pointsToRemove
    //                     );
    //                     std::cout << "Removed " << pointsToRemove << " old points, keeping " 
    //                             << m_accumulatedPoints.size() << std::endl;
    //                 }
                    
    //                 packetCount = 0;
    //                 return;
    //             }
    //         } else {
    //             std::cout << "Packet has no points" << std::endl;
    //             // Return the packet buffer even if no points
    //             dwSensorLidar_returnPacket(nextPacket, m_lidarSensor);
    //         }
    //     }
    //     else if (status == DW_END_OF_STREAM) {
    //         std::cout << "End of stream reached" << std::endl;
    //         if (!m_accumulatedPoints.empty()) {
    //             // Copy accumulated points to rendering buffer
    //             dwLidarPointXYZI* map = reinterpret_cast<dwLidarPointXYZI*>(m_pointCloud.get());
    //             memcpy(map, m_accumulatedPoints.data(), m_accumulatedPoints.size() * sizeof(dwLidarPointXYZI));
                
    //             updateFrame(m_accumulatedPoints.size(), packetCount, lastHostTimestamp, lastSensorTimestamp);
    //         }
    //         dwSensor_reset(m_lidarSensor);
    //         packetCount = 0;
    //         return;
    //     }
    //     else if (status == DW_TIME_OUT) {
    //         // If we have any accumulated points, display them even on timeout
    //         if (!m_accumulatedPoints.empty()) {
    //             std::cout << "Timeout with " << m_accumulatedPoints.size() << " accumulated points" << std::endl;
                
    //             // Copy accumulated points to rendering buffer
    //             dwLidarPointXYZI* map = reinterpret_cast<dwLidarPointXYZI*>(m_pointCloud.get());
    //             memcpy(map, m_accumulatedPoints.data(), m_accumulatedPoints.size() * sizeof(dwLidarPointXYZI));
                
    //             updateFrame(m_accumulatedPoints.size(), packetCount, lastHostTimestamp, lastSensorTimestamp);
    //             packetCount = 0;
    //         } else {
    //             std::cout << "Timeout with no accumulated points" << std::endl;
    //         }
    //         return;
    //     }
    //     else {
    //         std::cout << "Error reading packet: " << status << std::endl;
    //         stop();
    //         return;
    //     }
    // }

    void computeSpin()
{
    const dwLidarDecodedPacket* nextPacket;
    static uint32_t frameCount = 0;
    
    dwStatus status = dwSensorLidar_readPacket(&nextPacket, 100000, m_lidarSensor); // 100ms timeout
    
    if (status == DW_SUCCESS) {
        if (nextPacket->nPoints > 0) {
            frameCount++;
            
            // For HAP, we get continuous stream of points
            // Don't accumulate too much, just display what we have
            dwLidarPointXYZI* map = reinterpret_cast<dwLidarPointXYZI*>(m_pointCloud.get());
            
            // Copy points directly (HAP already provides accumulated points)
            size_t copyCount = std::min((size_t)nextPacket->nPoints, 
                                       (size_t)m_pointCloudBufferCapacity);
            memcpy(map, nextPacket->pointsXYZI, copyCount * sizeof(dwLidarPointXYZI));
            
            updateFrame(copyCount, frameCount, 
                       nextPacket->hostTimestamp, 
                       nextPacket->sensorTimestamp);
        }
        
        dwSensorLidar_returnPacket(nextPacket, m_lidarSensor);
    }
    else if (status == DW_TIME_OUT) {
        // Normal timeout, HAP might be in motor startup
        DEBUG_LOG("Timeout waiting for packets - sensor may be starting up");
    }
    else if (status == DW_END_OF_STREAM) {
        dwSensor_reset(m_lidarSensor);
    }
}


    void onProcess() override
    {
        computeSpin();    
    }

    void onRender() override
    {
        // render text in the middle of the window
        dwRenderEngine_reset(m_renderEngine);
        dwRenderEngine_setTile(0, m_renderEngine);
        dwRenderEngine_setModelView(getMouseView().getModelView(), m_renderEngine);
        dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine);

        dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_DARKGREY, m_renderEngine);

        dwRenderEngine_setBackgroundColor({0.0f, 0.0f, 0.0f, 1.0f}, m_renderEngine);

        dwRenderEngine_renderBuffer(m_gridBuffer, m_gridBufferPrimitiveCount, m_renderEngine);

        dwRenderEngine_setColorByValue(m_colorByValueMode, 130.0f, m_renderEngine);

        dwRenderEngine_renderBuffer(m_pointCloudBuffer, m_pointCloudBufferSize, m_renderEngine);

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
        dwRenderEngine_renderText2D(m_message7.c_str(), {20.f, 20.f}, m_renderEngine);

        renderutils::renderFPS(m_renderEngine, getCurrentFPS());

    }

};


int main(int argc, const char** argv)
{
    // Define all arguments used by the application
    ProgramArguments args(argc, argv,
                        {
                            ProgramArguments::Option_t("protocol", "lidar.custom"),
                            ProgramArguments::Option_t("params", ""),
                            ProgramArguments::Option_t("max-points", "20000"),
                            ProgramArguments::Option_t("output-dir", ""),
                            ProgramArguments::Option_t("show-intensity", "true")
                        });

    // Validate required parameters
    std::string paramsStr = args.get("params");
    if (paramsStr.empty()) {
        std::cerr << "ERROR: You must provide the 'params' argument for the Livox HAP LiDAR." << std::endl;
        std::cerr << "Example: --params=decoder-path=/path/to/liblivox_hap_plugin.so,ip=192.168.1.3,host-ip=192.168.1.108,broadcast-code=YOUR_CODE,sdk-config-path=/path/to/livox_hap_config.json" << std::endl;
        return -1;
    }

    // Initialize the LiDAR replay application
    LidarHapReplaySample app(args);

    // Parse max points parameter
    uint32_t maxPoints = 20000;  // Default value
    std::string maxPointsStr = args.get("max-points");
    if (!maxPointsStr.empty()) {
        try {
            maxPoints = std::stoi(maxPointsStr);
            if (maxPoints < 1000) {
                std::cout << "WARNING: max-points value too small, using minimum of 1000" << std::endl;
                maxPoints = 1000;
            }
            if (maxPoints > 100000) {
                std::cout << "WARNING: max-points value too large, using maximum of 100000" << std::endl;
                maxPoints = 100000;
            }
        } catch (const std::exception& e) {
            std::cout << "WARNING: Invalid max-points value, using default of 20000" << std::endl;
        }
    }
    
    std::cout << "Initializing HAP visualization with max point count: " << maxPoints << std::endl;

    // Initialize the window for visualization
    app.initializeWindow("Livox HAP LiDAR Visualization", 1280, 800, args.enabled("offscreen"));
    
    // Configure the point accumulation limit
    app.setMaxAccumulatedPoints(maxPoints);
    
    // Run the application
    return app.run();
} 