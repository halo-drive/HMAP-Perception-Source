#ifndef INTER_LIDAR_ICP_HPP
#define INTER_LIDAR_ICP_HPP

#include <cstdint>
#include <string>
#include <fstream>
#include <chrono>
#include <map>
#include <vector>
#include <unordered_map>

#include <driver_types.h>

// DriveWorks includes
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
#include "FreeSpaceDW.hpp"
#include "CenterPointDW.hpp"
#include "SimpleTracker.hpp"

// TensorRT includes for object detection
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <framework/DriveWorksSample.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/WindowGLFW.hpp>

using namespace dw_samples::common;

//------------------------------------------------------------------------------
// Object Detection Components
//------------------------------------------------------------------------------

// Logger class for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

// Bounding box structure for detected objects
struct BoundingBox {
    float x, y, z;           // Center coordinates
    float width, length, height;  // Dimensions
    float rotation;          // Rotation around Z-axis
    float confidence;        // Detection confidence
    int classId;            // Object class (0: Vehicle, 1: Pedestrian, 2: Cyclist)
    int trackId = -1;       // Tracking ID for temporal consistency
};

//------------------------------------------------------------------------------
// Inter-Lidar ICP Processor with Object Detection
// Aligns two lidars using ICP, stitches them together with ground plane extraction,
// and performs object detection on the stitched point cloud
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
    
    // Object detection constants
    static constexpr int VERTICES_PER_BOX = 16;
    static constexpr int HISTORY_LENGTH = 3;
    
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
    
    // Initial alignment criteria - REALISTIC FOR REAL-WORLD CONDITIONS
    static constexpr uint32_t MIN_SUCCESSFUL_ICP_FOR_ALIGNMENT = 8;  // Need 8 consecutive successful ICPs
    static constexpr float32_t MAX_RMS_COST_FOR_ALIGNMENT = 0.08f;   // 80mm max RMS cost (realistic for real-world)
    static constexpr float32_t MIN_INLIER_FRACTION_FOR_ALIGNMENT = 0.8f; // 80% inlier fraction
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
    std::string m_tensorRTEngine;  // Legacy single-engine TensorRT path (unused with CenterPoint)
    std::string m_pfeEnginePath;   // CenterPoint PFE engine
    std::string m_rpnEnginePath;   // CenterPoint RPN engine
    uint32_t m_maxIters;
    uint32_t m_numFrames;
    uint32_t m_skipFrames;
    uint32_t m_frameNum = 0;
    bool m_verbose;
    bool m_savePointClouds;
    bool m_objectDetectionEnabled = true;
    bool m_groundPlaneEnabled = true;
    bool m_groundPlaneVisualizationEnabled = true;
    bool m_bevVisualizationEnabled = false;  // BEV feature map visualization
    bool m_heatmapVisualizationEnabled = false;  // Heatmap visualization
    bool m_paused = false;  // Simple pause mechanism
    uint32_t m_minPointsThreshold = 120;  // Minimum points required for valid detection



    FreeSpaceDW m_freeSpace;
    std::vector<float32_t> m_freeSpacePoints;
    uint32_t m_freeSpaceRenderBufferId = 0;
    bool m_freeSpaceEnabled = true;
    bool m_freeSpaceVisualizationEnabled = true;
    
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
    static constexpr float32_t GROUND_PLANE_FILTER_ALPHA = 0.9f;  // Smoothing factor (0.9 = very slow change, more stable)
    // Debugging: track when the ground plane was last updated to catch stale renders
    uint32_t m_groundPlaneLastUpdateFrame = 0;
    dwVector3f m_lastRenderedPlaneNormal{0.0f, 0.0f, 1.0f};
    float32_t m_lastRenderedPlaneOffset = 0.0f;
    
    // ------------------------------------------------
    // Object Detection Components
    // ------------------------------------------------
    
    // Object detection backend: CenterPoint two-stage detector (PFE + RPN).
    Logger m_logger;
    std::unique_ptr<CenterPointDW> m_centerPointDetector;

    // Legacy single-engine TensorRT members are kept for compatibility but are no longer used
    // when CenterPoint is enabled.
    std::unique_ptr<nvinfer1::IRuntime> m_runtime{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> m_executionContext{nullptr};
    
    // Object detection buffers (legacy path)
    void* m_deviceInputBuffer{nullptr};
    void* m_deviceOutputBuffer{nullptr};
    void* m_deviceNumPointsBuffer{nullptr};
    void* m_deviceOutputCountBuffer{nullptr};
    std::vector<float> m_hostInputBuffer;
    std::vector<float> m_hostOutputBuffer;
    size_t m_inputSize{0};
    size_t m_outputSize{0};
    
    // Object detection parameters (legacy path)
    int m_maxPoints{0};      // Maximum number of points the model expects
    int m_numFeatures{0};    // Number of features per point (usually 4 for x,y,z,intensity)
    uint32_t m_requiredPoints{0};  // Number of points to collect before inference
    uint32_t m_maxModelPoints{0};  // Maximum points the model can handle
    
    // Model-specific constants (legacy single-engine implementation)
    static constexpr int FIXED_NUM_POINTS = 204800;   // Model requirement
    static constexpr int REALTIME_NUM_POINTS = 29000; // Optimized for real-time performance
    static constexpr int POINT_FEATURES = 4;
    
    // Detection results and tracking
    std::vector<BoundingBox> m_currentBoxes;
    std::map<int, std::vector<BoundingBox>> m_trackingHistory;  // ID -> history of boxes
    int m_nextTrackID = 0;
    std::unique_ptr<SimpleTracker> m_tracker;  // SORT-style tracker
    
    // BEV and heatmap visualization data
    std::vector<float> m_bevFeatureMap;  // BEV feature map data (C x H x W)
    std::vector<float> m_heatmapData;     // Heatmap data (H x W)
    std::vector<uint8_t> m_bevImageData;   // BEV RGB image data for rendering
    std::vector<uint8_t> m_heatmapImageData; // Heatmap RGB image data for rendering
    
    // Point cloud for object detection
    std::unique_ptr<float32_t[]> m_pointCloudForDetection;
    
    // Point thresholds for different ranges
    std::unordered_map<int32_t, int32_t> m_pointThresholds;
    
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
    dwPointCloud m_stitchedPointsHost;                // Host copy for ground plane extraction and object detection
    
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
    WindowTile m_bevTile;                 // BEV feature map visualization tile
    WindowTile m_heatmapTile;             // Heatmap visualization tile
    
    // Ground plane rendering
    uint32_t m_groundPlaneRenderBufferId = 0;
    
    // Object detection rendering
    uint32_t m_boxLineBuffer = 0;
    
    // CUDA/GL
    cudaStream_t m_stream;

    // CUDA-native memory coherency management (replaces compiler-based barriers)
    cudaStream_t m_memoryCoherencyStream;   // Dedicated stream for memory synchronization
    cudaEvent_t m_cpuWriteCompleteEvent;    // Event: CPU writes to buffer complete
    cudaEvent_t m_gpuReadReadyEvent;        // Event: GPU ready to read from buffer

    // Debug counters (per-frame)
    uint32_t m_renderedGroundPlanesThisFrame = 0;
    
    // ------------------------------------------------
    // Logging and Debug
    // ------------------------------------------------
    std::ofstream m_logFile;
    uint32_t m_successfulICPCount = 0;
    uint32_t m_failedICPCount = 0;
    
    // Object detection statistics
    uint32_t m_totalDetections = 0;
    uint32_t m_vehicleDetections = 0;
    uint32_t m_pedestrianDetections = 0;
    uint32_t m_cyclistDetections = 0;
    
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
    
    // Object detection initialization
    bool initializeObjectDetection();
    bool initializeTensorRT();
    void calculateInputRequirements();

    void inspectRenderBuffers();

    void initFreeSpaceRendering();
    void performFreeSpaceDetection();
    void renderFreeSpace();
    
    // Processing functions
    bool getSpinFromBothLidars();
    bool getSpinFromLidar(uint32_t lidarIndex);
    void applyRigTransformations();
    bool performICP();
    void stitchPointClouds();
    void performGroundPlaneExtraction();
    void logICPResults();
    
    // Object detection processing
    void performObjectDetection();
    
    // Object detection filtering and processing
    std::vector<BoundingBox> crossClassNMS(const std::vector<BoundingBox>& detections, float iouThreshold = 0.3f);
    std::vector<BoundingBox> classAwareNMS(const std::vector<BoundingBox>& detections, float iouThreshold = 0.3f);
    std::vector<BoundingBox> filterBoundingBoxesBySize(const std::vector<BoundingBox>& boxes);
    std::vector<BoundingBox> mergeNearbyDetections(const std::vector<BoundingBox>& detections, float mergeThreshold = 1.0f);
    std::vector<BoundingBox> filterWalls(const std::vector<BoundingBox>& boxes);
    std::vector<BoundingBox> filterPedestrianFalsePositives(const std::vector<BoundingBox>& boxes);
    std::vector<BoundingBox> applyNMS(const std::vector<BoundingBox>& boxes, float iouThreshold = 0.3f);
    std::vector<BoundingBox> filterByClasses(const std::vector<BoundingBox>& boxes, const std::vector<int>& allowedClasses);
    std::vector<BoundingBox> applyTemporalFiltering(const std::vector<BoundingBox>& currentBoxes);
    std::vector<BoundingBox> filterByPointDensity(const std::vector<BoundingBox>& boxes);
    std::vector<BoundingBox> filterByGroundPlane(const std::vector<BoundingBox>& boxes);
    std::vector<BoundingBox> filterByDistanceBasedConfidence(const std::vector<BoundingBox>& boxes);
    std::vector<BoundingBox> filterByAspectRatios(const std::vector<BoundingBox>& boxes);
    std::vector<BoundingBox> applyAllFilters(const std::vector<BoundingBox>& rawBoxes);
    
    // Object detection utility functions
    float computeIoU(const BoundingBox& box1, const BoundingBox& box2);
    int countPointsInBox(const BoundingBox& box);
    void adjustBoxToFitPoints(BoundingBox& box);
    
    // State management functions
    void updateAlignmentState();
    bool shouldPerformICP() const;
    bool shouldPerformGroundPlaneExtraction() const;
    bool shouldPerformObjectDetection() const;
    bool isInitialAlignmentComplete() const;
    bool isPeriodicICPDue() const;
    float32_t calculateTransformChange(const dwTransformation3f& current, const dwTransformation3f& previous) const;
    void announceLibarReady();
    
    // Utility functions
    void checkDeviceType(const dwLidarProperties& prop, uint32_t lidarIndex);
    void savePointCloudToPLY(const dwPointCloud& pointCloud, const std::string& filename);
    void printConfiguration();
    void printICPStatistics();
    void printObjectDetectionStatistics();
    
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
    
    // Object detection rendering
    void renderBoundingBoxes();
    void renderPointsInBox(const BoundingBox& box, const dwVector3f& position, float lidarHeight);
    void initBoundingBoxRenderBuffer();
    
    // BEV and heatmap visualization
    void renderBEVFeatureMap();
    void renderHeatmap();
    void convertBEVToImage(const std::vector<float>& bevData, std::vector<uint8_t>& outImage);
    void convertHeatmapToImage(const std::vector<float>& heatmapData, std::vector<uint8_t>& outImage);
    
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




// ./tensorRT_optimization \
//   --modelType=onnx \
//   --onnxFile=/usr/local/driveworks-5.20/samples/src/sensors/lidar/lidar_object_detection/pointpillars_deployable_fixed.onnx \
//   --useDLA \
//   --half2=1 \
//   --workspaceSize=2048 \
//   --verbose=1 \
//   --out=/usr/local/driveworks-5.20/samples/src/sensors/lidar/lidar_object_detection/pp_dla_fp16.bin



