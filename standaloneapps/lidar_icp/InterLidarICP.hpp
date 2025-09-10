/////////////////////////////////////////////////////////////////////////////////////////
// Inter-Lidar ICP Header
// Simple prototype for aligning two lidars using ICP with ground plane extraction
// Now includes state management for initial alignment and periodic re-alignment
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef INTER_LIDAR_ICP_HPP
#define INTER_LIDAR_ICP_HPP

#include <cstdint>
#include <string>
#include <fstream>
#include <chrono>

#include <driver_types.h>

#include <dw/core/base/Types.h>
#include <dw/core/context/Context.h>
#include <dw/image/Image.h>
#include <dw/pointcloudprocessing/accumulator/PointCloudAccumulator.h>
#include <dw/pointcloudprocessing/icp/PointCloudICP.h>
#include <dw/pointcloudprocessing/pointcloud/PointCloud.h>
#include <dw/pointcloudprocessing/planeextractor/PointCloudPlaneExtractor.h>
#include <dw/pointcloudprocessing/stitcher/PointCloudStitcher.h>
#include <dw/rig/Rig.h>
#include <dw/rig/Vehicle.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/lidar/Lidar.h>
#include <dw/sensors/sensormanager/SensorManager.h>

#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Visualization.h>

#include <framework/DriveWorksSample.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/WindowGLFW.hpp>

using namespace dw_samples::common;

//------------------------------------------------------------------------------
// Inter-Lidar ICP Processor
// Aligns two lidars using ICP and stitches them together with ground plane extraction
// Includes state management for initial alignment and periodic re-alignment
//------------------------------------------------------------------------------
class InterLidarICP : public dw_samples::common::DriveWorksSample
{
private:
    // ------------------------------------------------
    // Global Constants
    // ------------------------------------------------
    static const uint32_t NUM_LIDARS = 2;
    static const uint32_t LIDAR_A_INDEX = 0;
    static const uint32_t LIDAR_B_INDEX = 1;
    static const uint32_t MAX_POINTS_TO_RENDER = 500000;
    
    // Ground plane visualization constants
    static constexpr uint32_t GROUND_PLANE_GRID_SIZE = 50;
    static constexpr float32_t GROUND_PLANE_CELL_SIZE = 0.5f;
    
    // ------------------------------------------------
    // State Management
    // ------------------------------------------------
    enum class AlignmentState {
        INITIALIZING,           // System starting up
        INITIAL_ALIGNMENT,      // Performing continuous ICP until aligned
        ALIGNED,               // Lidars aligned, periodic ICP + continuous ground plane
        REALIGNMENT            // Performing ICP re-alignment
    };
    
    AlignmentState m_alignmentState = AlignmentState::INITIALIZING;
    std::string getStateString() const;
    
    // Initial alignment criteria
    static constexpr uint32_t MIN_SUCCESSFUL_ICP_FOR_ALIGNMENT = 5;  // Need 5 consecutive successful ICPs
    static constexpr float32_t MAX_RMS_COST_FOR_ALIGNMENT = 0.060f;   // 60mm max RMS cost (increased from 5mm)
    static constexpr float32_t MIN_INLIER_FRACTION_FOR_ALIGNMENT = 0.7f; // 70% inlier fraction
    static constexpr float32_t MAX_TRANSFORM_CHANGE_FOR_ALIGNMENT = 0.01f; // Max transform change between iterations
    
    // Periodic ICP timing
    static constexpr float32_t PERIODIC_ICP_INTERVAL_SECONDS = 10.0f;
    std::chrono::steady_clock::time_point m_lastPeriodicICP;
    
    // Alignment tracking
    uint32_t m_consecutiveSuccessfulICP = 0;
    bool m_lidarReadyAnnounced = false;
    dwTransformation3f m_previousICPTransform = DW_IDENTITY_TRANSFORMATION3F;
    
    // ------------------------------------------------
    // Configuration Parameters
    // ------------------------------------------------
    std::string m_rigFile;
    uint32_t m_maxIters;
    uint32_t m_numFrames;
    uint32_t m_skipFrames;
    uint32_t m_frameNum = 0;
    bool m_verbose;
    bool m_savePointClouds;
    bool m_stitchOnly = false;  // If true, disable ICP and ground plane
    bool m_paused = false;  // Simple pause mechanism
    
    // ------------------------------------------------
    // DriveWorks Handles
    // ------------------------------------------------
    dwContextHandle_t m_context = DW_NULL_HANDLE;
    dwVisualizationContextHandle_t m_viz = DW_NULL_HANDLE;
    dwSALHandle_t m_sal = DW_NULL_HANDLE;
    dwRigHandle_t m_rigConfig = DW_NULL_HANDLE;
    dwSensorManagerHandle_t m_sensorManager = DW_NULL_HANDLE;
    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
    
    // Point Cloud Processing Handles
    dwPointCloudAccumulatorHandle_t m_accumulator[NUM_LIDARS] = {DW_NULL_HANDLE};
    dwPointCloudICPHandle_t m_icp = DW_NULL_HANDLE;
    dwPointCloudStitcherHandle_t m_stitcher = DW_NULL_HANDLE;
    dwPointCloudStitcherHandle_t m_coordinateConverter[NUM_LIDARS] = {DW_NULL_HANDLE}; // Individual transformers
    dwPointCloudStitcherHandle_t m_icpTransformer = DW_NULL_HANDLE; // For applying ICP transform
    
    // Ground Plane Extraction Handle
    dwPointCloudPlaneExtractorHandle_t m_planeExtractor = DW_NULL_HANDLE;
    dwPointCloudExtractedPlane m_groundPlane;
    bool m_groundPlaneValid = false;
    
    // Ground plane temporal filtering for stability
    dwPointCloudExtractedPlane m_filteredGroundPlane;
    bool m_filteredGroundPlaneValid = false;
    static constexpr float32_t GROUND_PLANE_FILTER_ALPHA = 0.8f;  // Smoothing factor (0.8 = slow change)
    
    // ------------------------------------------------
    // Sensor Information
    // ------------------------------------------------
    uint32_t m_lidarCount = 0;
    dwLidarProperties m_lidarProps[NUM_LIDARS];
    dwTransformation3f m_sensorToRigs[NUM_LIDARS];
    dwTime_t m_registrationTime[NUM_LIDARS];
    
    // Sensor state tracking
    bool m_lidarAccumulated[NUM_LIDARS];
    uint32_t m_lidarOverflowCount[NUM_LIDARS];
    
    // ------------------------------------------------
    // Point Cloud Buffers
    // ------------------------------------------------
    dwPointCloud m_accumulatedPoints[NUM_LIDARS];     // Raw accumulated points from each lidar
    dwPointCloud m_rigTransformedPoints[NUM_LIDARS];  // Points after rig transformation
    dwPointCloud m_icpAlignedPoints[NUM_LIDARS];      // Points after ICP alignment (only lidar B gets transformed)
    dwPointCloud m_stitchedPoints;                    // Final stitched point cloud
    dwPointCloud m_stitchedPointsHost;                // Host copy for ground plane extraction
    
    // ------------------------------------------------
    // ICP State
    // ------------------------------------------------
    dwTransformation3f m_icpTransform = DW_IDENTITY_TRANSFORMATION3F;  // Current ICP transformation
    dwTransformation3f m_cumulativeICPTransform = DW_IDENTITY_TRANSFORMATION3F;  // Cumulative ICP over time
    
    // ICP statistics
    struct ICPStats {
        bool successful;
        uint32_t iterations;
        uint32_t correspondences;
        float32_t rmsCost;
        float32_t inlierFraction;
        float32_t processingTime;
    } m_lastICPStats;
    
    // ------------------------------------------------
    // Rendering
    // ------------------------------------------------
    typedef struct WindowTile {
        uint32_t tileId;
        uint32_t renderBufferId;
    } WindowTile;
    
    WindowTile m_lidarTiles[NUM_LIDARS];  // Individual lidar views
    WindowTile m_stitchedTile;            // Stitched view
    WindowTile m_icpTile;                 // ICP alignment view
    
    // Ground plane rendering
    uint32_t m_groundPlaneRenderBufferId = 0;
    
    // CUDA/GL
    cudaStream_t m_stream;
    
    // ------------------------------------------------
    // Logging and Debug
    // ------------------------------------------------
    std::ofstream m_logFile;
    uint32_t m_successfulICPCount = 0;
    uint32_t m_failedICPCount = 0;
    
    // ------------------------------------------------
    // Member Functions
    // ------------------------------------------------
    
    // Initialization functions
    void initDriveWorks();
    void initSensors();
    void initBuffers();
    void initAccumulation();
    void initICP();
    void initStitching();
    void initGroundPlaneExtraction();
    void initRendering();
    void initLogging();
    
    // Processing functions
    bool getSpinFromBothLidars();
    bool getSpinFromLidar(uint32_t lidarIndex);
    void applyRigTransformations();
    bool performICP();
    void stitchPointClouds();
    void performGroundPlaneExtraction();
    void logICPResults();
    
    // State management functions
    void updateAlignmentState();
    bool shouldPerformICP() const;
    bool shouldPerformGroundPlaneExtraction() const;
    bool isInitialAlignmentComplete() const;
    bool isPeriodicICPDue() const;
    float32_t calculateTransformChange(const dwTransformation3f& current, const dwTransformation3f& previous) const;
    void announceLibarReady();
    
    // Utility functions
    void checkDeviceType(const dwLidarProperties& prop, uint32_t lidarIndex);
    void savePointCloudToPLY(const dwPointCloud& pointCloud, const std::string& filename);
    void printConfiguration();
    void printICPStatistics();
    
    // Rendering functions
    void renderPointCloud(uint32_t renderBufferId,
                          uint32_t tileId,
                          uint32_t offset,
                          dwRenderEngineColorRGBA color,
                          const dwPointCloud& pointCloud);
    void copyToRenderBuffer(uint32_t renderBufferId, uint32_t offset, const dwPointCloud& pointCloud);
    void renderTexts(const char* msg, const dwVector2f& location);
    void renderGroundPlane();
    void initGroundPlaneRenderBuffer();
    
public:
    ///------------------------------------------------------------------------------
    /// Initialize sample
    ///------------------------------------------------------------------------------
    InterLidarICP(const ProgramArguments& args);
    
    ///------------------------------------------------------------------------------
    /// Destructor
    ///------------------------------------------------------------------------------
    ~InterLidarICP();

    ///------------------------------------------------------------------------------
    /// Initialize DriveWorks components
    ///------------------------------------------------------------------------------
    bool onInitialize() override;

    ///------------------------------------------------------------------------------
    /// Release acquired memory
    ///------------------------------------------------------------------------------
    void onRelease() override;

    ///------------------------------------------------------------------------------
    /// Main processing loop
    ///------------------------------------------------------------------------------
    void onProcess() override;

    ///------------------------------------------------------------------------------
    /// Render loop
    ///------------------------------------------------------------------------------
    void onRender() override;
    
    ///------------------------------------------------------------------------------
    /// Handle window resize
    ///------------------------------------------------------------------------------
    void onResizeWindow(int width, int height) override;
    
    ///------------------------------------------------------------------------------
    /// Handle key input
    ///------------------------------------------------------------------------------
    void onKeyDown(int32_t key, int32_t scancode, int32_t mods) override;
};

#endif // INTER_LIDAR_ICP_HPP