#ifndef DNN_PROCESSOR_HPP_
#define DNN_PROCESSOR_HPP_

#include <cstdint>
#include <vector>
#include <string>
#include <memory>

// DriveWorks Core Framework
#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>

// DNN Processing
#include <dw/dnn/DNN.h>
#include <dw/dnn/tensor/Tensor.h>
#include <dw/dnn/dataconditioner/DataConditioner.h>
#include <dw/interop/streamer/TensorStreamer.h>

// Image Processing
#include <dw/image/Image.h>

// CUDA Runtime
#include <cuda_runtime.h>

// Framework Utilities
#include <framework/Checks.hpp>
#include <framework/Log.hpp>
#include <framework/SamplesDataPath.hpp>

namespace multicam_dnn {

/**
 * @brief DriveNet detection result structure
 */
struct DetectionResult {
    dwRectf bbox;           // Bounding box in image coordinates
    float32_t confidence;   // Detection confidence score
    uint32_t classId;       // DriveNet class identifier
};

/**
 * @brief Segmentation result structure for binary mask processing
 */
struct SegmentationResult {
    std::vector<uint8_t> driveAreaMask;    // Binary mask for drivable area [640x640]
    std::vector<uint8_t> laneLineMask;     // Binary mask for lane lines [640x640]
    uint32_t drivePixelCount;              // Number of drivable pixels
    uint32_t lanePixelCount;               // Number of lane line pixels
};

/**
 * @brief Complete inference result for pomo-drivenet model
 */
struct InferenceResult {
    std::vector<DetectionResult> detections;
    SegmentationResult segmentation;
    bool isValid;                          // Processing success flag
    uint32_t processingTimeMs;             // Inference timing
};

/**
 * @brief DriveNet 3-output DNN processor for pomo-drivenet.onnx model
 * 
 * Processes detection + dual segmentation outputs using pure DriveWorks APIs.
 * Optimized for automotive embedded deployment with cuDLA acceleration support.
 */
class DNNProcessor {
private:
    // ============================================
    // POMO-DRIVENET MODEL CONSTANTS
    // ============================================
    static constexpr uint32_t NUM_OUTPUT_TENSORS = 3U;
    static constexpr uint32_t DETECTION_GRID_SIZE = 25200U;  // Detection grid points
    static constexpr uint32_t DETECTION_FEATURES = 6U;       // Detection feature vector
    static constexpr uint32_t SEGMENTATION_SIZE = 640 * 640; // Segmentation mask size
    static constexpr uint32_t INPUT_WIDTH = 640U;
    static constexpr uint32_t INPUT_HEIGHT = 640U;
    static constexpr uint32_t INPUT_CHANNELS = 3U;
    
    // DriveNet processing thresholds
    static constexpr float32_t CONFIDENCE_THRESHOLD = 0.25f;  // DriveNet confidence threshold
    static constexpr float32_t NMS_THRESHOLD = 0.5f;        // Non-maximum suppression
    
    // ============================================
    // DRIVEWORKS HANDLES
    // ============================================
    dwContextHandle_t m_context;
    dwDNNHandle_t m_dnn;
    dwDataConditionerHandle_t m_dataConditioner;
    
    // Tensor infrastructure
    dwDNNTensorHandle_t m_dnnInput;
    dwDNNTensorHandle_t m_dnnOutputsDevice[NUM_OUTPUT_TENSORS];
    dwDNNTensorHandle_t m_dnnOutputsHost[NUM_OUTPUT_TENSORS];
    dwDNNTensorStreamerHandle_t m_outputStreamers[NUM_OUTPUT_TENSORS];
    
    // ============================================
    // PROCESSING CONFIGURATION
    // ============================================
    cudaStream_t m_cudaStream;
    uint32_t m_imageWidth;
    uint32_t m_imageHeight;
    dwRect m_processingRegion;
    
    // Platform configuration
    bool m_useCuDLA;
    uint32_t m_dlaEngineNo;
    
    // Performance tracking
    uint32_t m_inferenceCount;

public:
    // ============================================
    // LIFECYCLE MANAGEMENT
    // ============================================
    
    /**
     * @brief Constructor
     * @param context DriveWorks context handle
     * @param imageWidth Input image width
     * @param imageHeight Input image height
     */
    // Dynamic tensor layout detection
    struct TensorLayoutDetector {
        static bool detectInputLayout(const dwDNNTensorProperties& props, TensorLayout& layout);
        static bool detectDetectionLayout(const dwDNNTensorProperties& props, DetectionLayout& layout);
        static bool detectSegmentationLayout(const dwDNNTensorProperties& props, SegmentationLayout& layout);
    };

    // Enhanced layout structures
    struct TensorLayout {
        uint32_t height_idx = UINT32_MAX;
        uint32_t width_idx = UINT32_MAX;
        uint32_t channels_idx = UINT32_MAX;
        uint32_t batch_idx = UINT32_MAX;
        bool isValid = false;
    };

    struct DetectionLayout {
        uint32_t grid_idx = UINT32_MAX;
        uint32_t features_idx = UINT32_MAX;
        uint32_t batch_idx = UINT32_MAX;
        bool isValid = false;
    };

    struct SegmentationLayout {
    uint32_t height_idx = UINT32_MAX;
    uint32_t width_idx = UINT32_MAX;
    uint32_t classes_idx = UINT32_MAX;
    bool isValid = false;
    };

    DNNProcessor(dwContextHandle_t context, uint32_t imageWidth, uint32_t imageHeight);
    
    /**
     * @brief Destructor - releases all DriveWorks resources
     */
    ~DNNProcessor();
    
    // Non-copyable
    DNNProcessor(const DNNProcessor&) = delete;
    DNNProcessor& operator=(const DNNProcessor&) = delete;
    
    // ============================================
    // INITIALIZATION INTERFACE
    // ============================================
    
    /**
     * @brief Initialize DNN processing pipeline
     * @param modelPath Path to pomo-drivenet TensorRT model file
     * @param useCuDLA Enable cuDLA acceleration
     * @param dlaEngine DLA engine number if enabled
     * @return true on success
     */
    bool initialize(const std::string& modelPath, bool useCuDLA = false, uint32_t dlaEngine = 0);
    
    /**
     * @brief Check if processor is properly initialized
     * @return true if ready for inference
     */
    bool isInitialized() const;
    
    // ============================================
    // INFERENCE INTERFACE
    // ============================================
    
    /**
     * @brief Process image through pomo-drivenet 3-output pipeline
     * @param inputImage CUDA RGBA image handle
     * @param result Output structure for detection and segmentation results
     * @return true on successful processing
     */
    bool processImage(dwImageHandle_t inputImage, InferenceResult& result);
    
    /**
     * @brief Get processing statistics
     * @return Number of successful inferences performed
     */
    uint32_t getInferenceCount() const { return m_inferenceCount; }

private:
    // ============================================
    // INTERNAL PROCESSING METHODS
    // ============================================
    // ADD to private section after line ~120:

    // ADD member variables:
    TensorLayout m_inputLayout;
    DetectionLayout m_detectionLayout;
    SegmentationLayout m_segmentationLayout;
    /**
     * @brief Load and initialize pomo-drivenet TensorRT model
     */
    bool initializeDNNModel(const std::string& modelPath);
    
    /**
     * @brief Setup 3-tensor output infrastructure
     */
    bool initializeTensors();
    
    /**
     * @brief Initialize data conditioning pipeline
     */
    bool initializeDataConditioner();
    
    /**
     * @brief Prepare input image for inference
     */
    bool prepareInput(dwImageHandle_t inputImage);
    
    /**
     * @brief Execute DNN inference
     */
    bool runInference();
    
    /**
     * @brief Stream output tensors from device to host
     */
    bool streamOutputsToHost();
    
    /**
     * @brief Return streamed tensors back to device
     */
    bool returnStreamedOutputs();
    
    // ============================================
    // OUTPUT INTERPRETATION METHODS
    // ============================================
    
    /**
     * @brief Interpret detection output tensor [1, 25200, 6]
     * @param detections Output detection vector
     * @return true on successful interpretation
     */
    bool interpretDetectionOutput(std::vector<DetectionResult>& detections);
    
    /**
     * @brief Interpret segmentation output tensors [1, 2, 640, 640]
     * @param segResult Output segmentation structure
     * @return true on successful interpretation
     */
    bool interpretSegmentationOutputs(SegmentationResult& segResult);
    
    /**
     * @brief Apply Non-Maximum Suppression to detection results
     */
    std::vector<DetectionResult> applyNMS(std::vector<DetectionResult>& detections, float32_t threshold);
    
    /**
     * @brief Convert normalized coordinates to image coordinates
     */
    dwRectf convertToImageCoordinates(float32_t x, float32_t y, float32_t w, float32_t h);
    
    /**
     * @brief Calculate IoU between two bounding boxes
     */
    float32_t calculateIoU(const dwRectf& box1, const dwRectf& box2);
    
    /**
     * @brief Convert binary segmentation to mask using argmax
     */
    void convertSegmentationToMask(float32_t* segData, std::vector<uint8_t>& mask);
    
    // ============================================
    // UTILITY METHODS
    // ============================================
    
    /**
     * @brief Get platform-specific model path prefix
     */
    std::string getPlatformPrefix();
    
    /**
     * @brief Validate tensor properties match pomo-drivenet model
     */
    bool validateTensorProperties();
    
    /**
     * @brief Release all allocated resources
     */
    void releaseResources();
};

} // namespace multicam_dnn

#endif // DNN_PROCESSOR_HPP_