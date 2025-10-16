/////////////////////////////////////////////////////////////////////////////////////////
//
// PointPillar LiDAR 3D Object Detection for NVIDIA DriveWorks
// Integrates NGC PointPillarNet model with VLP-16 LiDAR data
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef POINTPILLAR_DETECTOR_HPP_
#define POINTPILLAR_DETECTOR_HPP_

#include <memory>
#include <string>
#include <vector>

#include <dw/core/base/Types.h>
#include <dw/core/context/Context.h>
#include <dw/dnn/DNN.h>
#include <dw/dnn/tensor/Tensor.h>
#include <dw/sensors/lidar/Lidar.h>
#include <dw/interop/streamer/TensorStreamer.h>

namespace dw_samples {
namespace pointpillar {

/// 3D Bounding Box structure matching TAO recipe output format
struct BoundingBox3D {
    float32_t x;              // center x (meters)
    float32_t y;              // center y (meters)
    float32_t z;              // center z (meters)
    float32_t length;         // box length (meters)
    float32_t width;          // box width (meters)
    float32_t height;         // box height (meters)
    float32_t yaw;            // rotation around z-axis (radians)
    int32_t classId;          // 0=Vehicle, 1=Pedestrian, 2=Cyclist
    float32_t confidence;     // detection confidence score
    
    BoundingBox3D()
        : x(0.0f), y(0.0f), z(0.0f)
        , length(0.0f), width(0.0f), height(0.0f)
        , yaw(0.0f), classId(-1), confidence(0.0f) {}
    
    BoundingBox3D(float32_t x_, float32_t y_, float32_t z_,
                  float32_t l_, float32_t w_, float32_t h_,
                  float32_t yaw_, int32_t id_, float32_t score_)
        : x(x_), y(y_), z(z_)
        , length(l_), width(w_), height(h_)
        , yaw(yaw_), classId(id_), confidence(score_) {}
};

/// PointPillar 3D Object Detector
class PointPillarDetector {
public:
    /// Configuration parameters
    struct Config {
        std::string modelPath;           // Path to TensorRT .bin model
        float32_t confidenceThreshold;   // Min confidence to keep detection (default: 0.3)
        float32_t nmsIouThreshold;       // NMS IoU threshold (default: 0.01)
        int32_t maxDetections;           // Max detections to return (default: 100)
        uint32_t maxInputPoints;         // Max points capacity (default: 204800)
        
        Config()
            : modelPath("")
            , confidenceThreshold(0.3f)
            , nmsIouThreshold(0.01f)
            , maxDetections(100)
            , maxInputPoints(204800) {}
    };
    
    /// Class names
    static constexpr const char* CLASS_NAMES[3] = {
        "Vehicle", "Pedestrian", "Cyclist"
    };

public:
    /**
     * Constructor
     * @param config Configuration parameters
     * @param context DriveWorks context
     * @param cudaStream CUDA stream for async operations
     */
    PointPillarDetector(const Config& config,
                       dwContextHandle_t context,
                       cudaStream_t cudaStream);
    
    /// Destructor
    ~PointPillarDetector();
    
    // Disable copy/move
    PointPillarDetector(const PointPillarDetector&) = delete;
    PointPillarDetector& operator=(const PointPillarDetector&) = delete;
    
    /**
     * Run inference on VLP-16 point cloud
     * @param points Input point cloud from VLP-16
     * @param pointCount Number of valid points in the cloud
     * @param detections Output vector of detected 3D boxes
     * @return DW_SUCCESS on success
     */
    dwStatus runInference(const dwLidarPointXYZI* points,
                         uint32_t pointCount,
                         std::vector<BoundingBox3D>& detections);
    
    /**
     * Reset the detector state
     */
    dwStatus reset();
    
    /**
     * Get the maximum number of input points supported
     */
    uint32_t getMaxInputPoints() const { return m_maxInputPoints; }
    
    /**
     * Get average inference time in milliseconds
     */
    float32_t getAverageInferenceTime() const { return m_avgInferenceTime; }

private:
    /// Initialize DNN model
    dwStatus initializeDNN();
    
    /// Allocate tensors for input/output
    dwStatus allocateTensors();
    
    /// Prepare input tensors from VLP-16 point cloud
    dwStatus prepareInputTensors(const dwLidarPointXYZI* points, uint32_t pointCount);
    
    /// Parse output tensors into bounding boxes
    dwStatus parseOutputTensors(std::vector<BoundingBox3D>& detections);
    
    /// Apply NMS (Non-Maximum Suppression) to detections
    void applyNMS(std::vector<BoundingBox3D>& detections);
    
    /// Calculate 3D IoU between two oriented boxes
    float32_t calculate3DIoU(const BoundingBox3D& box1, const BoundingBox3D& box2);

private:
    // Configuration
    Config m_config;
    dwContextHandle_t m_context;
    cudaStream_t m_cudaStream;
    
    // DNN components
    dwDNNHandle_t m_dnn;
    
    // Input tensors (2 inputs: points and num_points)
    dwDNNTensorHandle_t m_inputTensorPoints;      // Shape: (1, maxPoints, 4)
    dwDNNTensorHandle_t m_inputTensorNumPoints;   // Shape: (1)
    
    // Output tensors (2 outputs: boxes and num_boxes)
    dwDNNTensorHandle_t m_outputTensorBoxes;      // Shape: (1, 393216, 9)
    dwDNNTensorHandle_t m_outputTensorNumBoxes;   // Shape: (1)
    
    // Streamers for GPU->CPU transfer
    dwDNNTensorStreamerHandle_t m_streamerBoxes;
    dwDNNTensorStreamerHandle_t m_streamerNumBoxes;
    
    // Host-side output tensors
    dwDNNTensorHandle_t m_outputTensorBoxesHost;
    dwDNNTensorHandle_t m_outputTensorNumBoxesHost;
    
    // CPU buffers for tensor data access
    float32_t* m_inputPointsBuffer;    // CPU-accessible buffer for input points
    uint32_t* m_inputNumPointsBuffer;  // CPU-accessible buffer for num_points
    
    // Model dimensions
    uint32_t m_maxInputPoints;         // 204800
    uint32_t m_maxOutputBoxes;         // 393216
    
    // Performance tracking
    float32_t m_avgInferenceTime;
    uint32_t m_inferenceCount;
};

} // namespace pointpillar
} // namespace dw_samples

#endif // POINTPILLAR_DETECTOR_HPP_