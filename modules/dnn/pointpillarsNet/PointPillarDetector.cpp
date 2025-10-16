/////////////////////////////////////////////////////////////////////////////////////////
//
// PointPillar LiDAR 3D Object Detection - Implementation (Fixed for DW 5.20)
//
/////////////////////////////////////////////////////////////////////////////////////////

#include "PointPillarDetector.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#include <framework/Checks.hpp>
#include <framework/Log.hpp>
#include <NvInferPlugin.h>

namespace dw_samples {
namespace pointpillar {

constexpr const char* PointPillarDetector::CLASS_NAMES[3];
// Anonymous namespace for internal helpers
namespace {

/// Sort comparator for bounding boxes by confidence (descending)
bool compareByConfidence(const BoundingBox3D& a, const BoundingBox3D& b) {
    return a.confidence > b.confidence;
}

/// Calculate 2D intersection area of two oriented boxes (Bird's Eye View)
float32_t calculate2DIntersection(const BoundingBox3D& boxA, const BoundingBox3D& boxB) {
    // Simplified 2D IoU calculation - using axis-aligned approximation
    float32_t aMinX = boxA.x - boxA.length / 2.0f;
    float32_t aMaxX = boxA.x + boxA.length / 2.0f;
    float32_t aMinY = boxA.y - boxA.width / 2.0f;
    float32_t aMaxY = boxA.y + boxA.width / 2.0f;
    
    float32_t bMinX = boxB.x - boxB.length / 2.0f;
    float32_t bMaxX = boxB.x + boxB.length / 2.0f;
    float32_t bMinY = boxB.y - boxB.width / 2.0f;
    float32_t bMaxY = boxB.y + boxB.width / 2.0f;
    
    float32_t interMinX = std::max(aMinX, bMinX);
    float32_t interMaxX = std::min(aMaxX, bMaxX);
    float32_t interMinY = std::max(aMinY, bMinY);
    float32_t interMaxY = std::min(aMaxY, bMaxY);
    
    if (interMinX >= interMaxX || interMinY >= interMaxY) {
        return 0.0f;
    }
    
    return (interMaxX - interMinX) * (interMaxY - interMinY);
}

} // anonymous namespace

//------------------------------------------------------------------------------
PointPillarDetector::PointPillarDetector(
    const Config& config,
    dwContextHandle_t context,
    cudaStream_t cudaStream)
    : m_config(config)
    , m_context(context)
    , m_cudaStream(cudaStream)
    , m_dnn(DW_NULL_HANDLE)
    , m_inputTensorPoints(DW_NULL_HANDLE)
    , m_inputTensorNumPoints(DW_NULL_HANDLE)
    , m_outputTensorBoxes(DW_NULL_HANDLE)
    , m_outputTensorNumBoxes(DW_NULL_HANDLE)
    , m_streamerBoxes(DW_NULL_HANDLE)
    , m_streamerNumBoxes(DW_NULL_HANDLE)
    , m_outputTensorBoxesHost(DW_NULL_HANDLE)
    , m_outputTensorNumBoxesHost(DW_NULL_HANDLE)
    , m_inputPointsBuffer(nullptr)
    , m_inputNumPointsBuffer(nullptr)
    , m_maxInputPoints(config.maxInputPoints)
    , m_maxOutputBoxes(393216)
    , m_avgInferenceTime(0.0f)
    , m_inferenceCount(0)
{
    log("Initializing PointPillar Detector...\n");
    
    CHECK_DW_ERROR(initializeDNN());
    CHECK_DW_ERROR(allocateTensors());
    
    log("PointPillar Detector initialized successfully.\n");
}

//------------------------------------------------------------------------------
PointPillarDetector::~PointPillarDetector()
{
    // Release streamers
    if (m_streamerBoxes != DW_NULL_HANDLE) {
        dwDNNTensorStreamer_release(m_streamerBoxes);
    }
    if (m_streamerNumBoxes != DW_NULL_HANDLE) {
        dwDNNTensorStreamer_release(m_streamerNumBoxes);
    }
    
    // Release tensors
    if (m_inputTensorPoints != DW_NULL_HANDLE) {
        dwDNNTensor_destroy(m_inputTensorPoints);
    }
    if (m_inputTensorNumPoints != DW_NULL_HANDLE) {
        dwDNNTensor_destroy(m_inputTensorNumPoints);
    }
    if (m_outputTensorBoxes != DW_NULL_HANDLE) {
        dwDNNTensor_destroy(m_outputTensorBoxes);
    }
    if (m_outputTensorNumBoxes != DW_NULL_HANDLE) {
        dwDNNTensor_destroy(m_outputTensorNumBoxes);
    }
    
    // Release DNN
    if (m_dnn != DW_NULL_HANDLE) {
        dwDNN_release(m_dnn);
    }
}

//------------------------------------------------------------------------------
dwStatus PointPillarDetector::initializeDNN()
{
    log("Loading PointPillar model: %s\n", m_config.modelPath.c_str());
    
    
    bool pluginInitSuccess = initLibNvInferPlugins(nullptr, "");
    if (!pluginInitSuccess) {
        logError("Failed to initialize TensorRT plugin library\n");
        return DW_INTERNAL_ERROR;
    }
    log("TensorRT plugin library initialized successfully\n");

    // Initialize DNN from TensorRT file
    dwStatus status = dwDNN_initializeTensorRTFromFile(
        &m_dnn,
        m_config.modelPath.c_str(),
        nullptr,  // No plugin configuration
        DW_PROCESSOR_TYPE_GPU,
        m_context
    );
    
    if (status != DW_SUCCESS) {
        logError("Failed to initialize DNN from file: %s\n", m_config.modelPath.c_str());
        return status;
    }
    
    // Set CUDA stream for async operations
    CHECK_DW_ERROR(dwDNN_setCUDAStream(m_cudaStream, m_dnn));
    
    // Verify input/output count using correct API
    uint32_t inputCount = 0;
    uint32_t outputCount = 0;
    CHECK_DW_ERROR(dwDNN_getInputBlobCount(&inputCount, m_dnn));
    CHECK_DW_ERROR(dwDNN_getOutputBlobCount(&outputCount, m_dnn));
    
    if (inputCount != 2) {
        logError("Expected 2 inputs (points, num_points), got %u\n", inputCount);
        return DW_INVALID_ARGUMENT;
    }
    
    if (outputCount != 2) {
        logError("Expected 2 outputs (boxes, num_boxes), got %u\n", outputCount);
        return DW_INVALID_ARGUMENT;
    }
    
    log("DNN loaded successfully. Inputs: %u, Outputs: %u\n", inputCount, outputCount);
    
    return DW_SUCCESS;
}

//------------------------------------------------------------------------------
dwStatus PointPillarDetector::allocateTensors()
{
    log("Allocating tensors...\n");
    
    // Get input tensor properties
    dwDNNTensorProperties inputPropsPoints;
    dwDNNTensorProperties inputPropsNumPoints;
    CHECK_DW_ERROR(dwDNN_getInputTensorProperties(&inputPropsPoints, 0, m_dnn));
    CHECK_DW_ERROR(dwDNN_getInputTensorProperties(&inputPropsNumPoints, 1, m_dnn));
    
    log("Input 0 (points): dims=%u, shape=[%u, %u, %u]\n",
        inputPropsPoints.numDimensions,
        inputPropsPoints.dimensionSize[0],
        inputPropsPoints.dimensionSize[1],
        inputPropsPoints.dimensionSize[2]);
    
    log("Input 1 (num_points): dims=%u, shape=[%u]\n",
        inputPropsNumPoints.numDimensions,
        inputPropsNumPoints.dimensionSize[0]);
    
    // Allocate input tensors
    CHECK_DW_ERROR(dwDNNTensor_create(&m_inputTensorPoints, &inputPropsPoints, m_context));
    CHECK_DW_ERROR(dwDNNTensor_create(&m_inputTensorNumPoints, &inputPropsNumPoints, m_context));
    
    // Get output tensor properties
    dwDNNTensorProperties outputPropsBoxes;
    dwDNNTensorProperties outputPropsNumBoxes;
    CHECK_DW_ERROR(dwDNN_getOutputTensorProperties(&outputPropsBoxes, 0, m_dnn));
    CHECK_DW_ERROR(dwDNN_getOutputTensorProperties(&outputPropsNumBoxes, 1, m_dnn));
    
    log("Output 0 (boxes): dims=%u, shape=[%u, %u, %u]\n",
        outputPropsBoxes.numDimensions,
        outputPropsBoxes.dimensionSize[0],
        outputPropsBoxes.dimensionSize[1],
        outputPropsBoxes.dimensionSize[2]);
    
    log("Output 1 (num_boxes): dims=%u, shape=[%u]\n",
        outputPropsNumBoxes.numDimensions,
        outputPropsNumBoxes.dimensionSize[0]);
    
    // Allocate device output tensors
    CHECK_DW_ERROR(dwDNNTensor_create(&m_outputTensorBoxes, &outputPropsBoxes, m_context));
    CHECK_DW_ERROR(dwDNNTensor_create(&m_outputTensorNumBoxes, &outputPropsNumBoxes, m_context));
    
    // Create host-side output tensors for CPU access
    dwDNNTensorProperties hostPropsBoxes = outputPropsBoxes;
    hostPropsBoxes.tensorType = DW_DNN_TENSOR_TYPE_CPU;
    
    dwDNNTensorProperties hostPropsNumBoxes = outputPropsNumBoxes;
    hostPropsNumBoxes.tensorType = DW_DNN_TENSOR_TYPE_CPU;
    
    // Initialize streamers for GPU->CPU transfer
    CHECK_DW_ERROR(dwDNNTensorStreamer_initialize(
        &m_streamerBoxes,
        &outputPropsBoxes,
        DW_DNN_TENSOR_TYPE_CPU,
        m_context
    ));
    
    CHECK_DW_ERROR(dwDNNTensorStreamer_initialize(
        &m_streamerNumBoxes,
        &outputPropsNumBoxes,
        DW_DNN_TENSOR_TYPE_CPU,
        m_context
    ));
    
    log("Tensors allocated successfully.\n");
    
    return DW_SUCCESS;
}

//------------------------------------------------------------------------------
dwStatus PointPillarDetector::prepareInputTensors(
    const dwLidarPointXYZI* points,
    uint32_t pointCount)
{
    // Clamp point count to max capacity
    uint32_t actualPoints = std::min(pointCount, m_maxInputPoints);
    
    // Lock input tensor for points
    void* pointsBuffer = nullptr;
    CHECK_DW_ERROR(dwDNNTensor_lock(&pointsBuffer, m_inputTensorPoints));
    float32_t* pointsData = static_cast<float32_t*>(pointsBuffer);
    
    // Copy VLP-16 points directly
    for (uint32_t i = 0; i < actualPoints; i++) {
        pointsData[i * 4 + 0] = points[i].x;
        pointsData[i * 4 + 1] = points[i].y;
        pointsData[i * 4 + 2] = points[i].z;
        pointsData[i * 4 + 3] = points[i].intensity;
    }
    
    // Zero-pad remaining points
    for (uint32_t i = actualPoints; i < m_maxInputPoints; i++) {
        pointsData[i * 4 + 0] = 0.0f;
        pointsData[i * 4 + 1] = 0.0f;
        pointsData[i * 4 + 2] = 0.0f;
        pointsData[i * 4 + 3] = 0.0f;
    }
    
    CHECK_DW_ERROR(dwDNNTensor_unlock(m_inputTensorPoints));
    
    // Set num_points - NOW AS 2D TENSOR (1, 1)
    void* numPointsBuffer = nullptr;
    CHECK_DW_ERROR(dwDNNTensor_lock(&numPointsBuffer, m_inputTensorNumPoints));
    uint32_t* numPointsData = static_cast<uint32_t*>(numPointsBuffer);
    
    // Write to first element of 2D tensor
    numPointsData[0] = actualPoints;
    
    CHECK_DW_ERROR(dwDNNTensor_unlock(m_inputTensorNumPoints));
    
    return DW_SUCCESS;
}

//------------------------------------------------------------------------------
dwStatus PointPillarDetector::runInference(
    const dwLidarPointXYZI* points,
    uint32_t pointCount,
    std::vector<BoundingBox3D>& detections)
{
    detections.clear();
    
    if (points == nullptr || pointCount == 0) {
        return DW_INVALID_ARGUMENT;
    }
    
    // Prepare input tensors
    CHECK_DW_ERROR(prepareInputTensors(points, pointCount));
    
    // Run inference
    dwConstDNNTensorHandle_t inputs[2] = {m_inputTensorPoints, m_inputTensorNumPoints};
    dwDNNTensorHandle_t outputs[2] = {m_outputTensorBoxes, m_outputTensorNumBoxes};
    
    CHECK_DW_ERROR(dwDNN_infer(outputs, 2, inputs, 2, m_dnn));
    
    // Stream outputs to CPU
    CHECK_DW_ERROR(dwDNNTensorStreamer_producerSend(m_outputTensorBoxes, m_streamerBoxes));
    CHECK_DW_ERROR(dwDNNTensorStreamer_producerSend(m_outputTensorNumBoxes, m_streamerNumBoxes));
    
    CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReceive(&m_outputTensorBoxesHost, 1000, m_streamerBoxes));
    CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReceive(&m_outputTensorNumBoxesHost, 1000, m_streamerNumBoxes));
    
    // Parse outputs
    CHECK_DW_ERROR(parseOutputTensors(detections));
    
    // Return streamed outputs
    CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReturn(&m_outputTensorBoxesHost, m_streamerBoxes));
    CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReturn(&m_outputTensorNumBoxesHost, m_streamerNumBoxes));
    
    CHECK_DW_ERROR(dwDNNTensorStreamer_producerReturn(nullptr, 1000, m_streamerBoxes));
    CHECK_DW_ERROR(dwDNNTensorStreamer_producerReturn(nullptr, 1000, m_streamerNumBoxes));
    
    // Apply NMS
    applyNMS(detections);
    
    return DW_SUCCESS;
}

//------------------------------------------------------------------------------
dwStatus PointPillarDetector::parseOutputTensors(std::vector<BoundingBox3D>& detections)
{
    void* boxesBuffer = nullptr;
    void* numBoxesBuffer = nullptr;
    
    CHECK_DW_ERROR(dwDNNTensor_lock(&boxesBuffer, m_outputTensorBoxesHost));
    CHECK_DW_ERROR(dwDNNTensor_lock(&numBoxesBuffer, m_outputTensorNumBoxesHost));
    
    float32_t* boxesData = static_cast<float32_t*>(boxesBuffer);
    int32_t* numBoxesData = static_cast<int32_t*>(numBoxesBuffer);
    
    // Read from first element of 2D tensor
    int32_t numDetections = numBoxesData[0];
    
    log("Raw detections from model: %d\n", numDetections);
    
    // Parse each detection (same as before)
    for (int32_t i = 0; i < numDetections && i < m_config.maxDetections; i++) {
        float32_t x = boxesData[i * 9 + 0];
        float32_t y = boxesData[i * 9 + 1];
        float32_t z = boxesData[i * 9 + 2];
        float32_t length = boxesData[i * 9 + 3];
        float32_t width = boxesData[i * 9 + 4];
        float32_t height = boxesData[i * 9 + 5];
        float32_t yaw = boxesData[i * 9 + 6];
        int32_t classId = static_cast<int32_t>(boxesData[i * 9 + 7]);
        float32_t confidence = boxesData[i * 9 + 8];
        
        if (confidence >= m_config.confidenceThreshold) {
            detections.emplace_back(x, y, z, length, width, height, yaw, classId, confidence);
        }
    }
    
    CHECK_DW_ERROR(dwDNNTensor_unlock(m_outputTensorBoxesHost));
    CHECK_DW_ERROR(dwDNNTensor_unlock(m_outputTensorNumBoxesHost));
    
    log("Detections after threshold: %zu\n", detections.size());
    
    return DW_SUCCESS;
}
//------------------------------------------------------------------------------
void PointPillarDetector::applyNMS(std::vector<BoundingBox3D>& detections)
{
    if (detections.empty()) {
        return;
    }
    
    // Sort by confidence (descending)
    std::sort(detections.begin(), detections.end(), compareByConfidence);
    
    std::vector<bool> suppressed(detections.size(), false);
    std::vector<BoundingBox3D> filtered;
    
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) {
            continue;
        }
        
        filtered.push_back(detections[i]);
        
        // Suppress overlapping boxes
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) {
                continue;
            }
            
            float32_t iou = calculate3DIoU(detections[i], detections[j]);
            
            if (iou >= m_config.nmsIouThreshold) {
                suppressed[j] = true;
            }
        }
    }
    
    detections = filtered;
    log("Detections after NMS: %zu\n", detections.size());
}

//------------------------------------------------------------------------------
float32_t PointPillarDetector::calculate3DIoU(
    const BoundingBox3D& boxA,
    const BoundingBox3D& boxB)
{
    // Calculate 2D IoU in Bird's Eye View (simplified)
    float32_t areaA = boxA.length * boxA.width;
    float32_t areaB = boxB.length * boxB.width;
    
    float32_t intersection = calculate2DIntersection(boxA, boxB);
    float32_t unionArea = areaA + areaB - intersection;
    
    if (unionArea < 1e-8f) {
        return 0.0f;
    }
    
    return intersection / unionArea;
}

//------------------------------------------------------------------------------
dwStatus PointPillarDetector::reset()
{
    CHECK_DW_ERROR(dwDNN_reset(m_dnn));
    m_avgInferenceTime = 0.0f;
    m_inferenceCount = 0;
    return DW_SUCCESS;
}

} // namespace pointpillar
} // namespace dw_samples