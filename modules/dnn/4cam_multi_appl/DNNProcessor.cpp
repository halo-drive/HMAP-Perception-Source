#include "DNNProcessor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

#include <dw/core/platform/GPUProperties.h>

using namespace dw_samples::common;

namespace multicam_dnn {

//#######################################################################################
// LIFECYCLE MANAGEMENT
//#######################################################################################

DNNProcessor::DNNProcessor(dwContextHandle_t context, uint32_t imageWidth, uint32_t imageHeight)
    : m_context(context)
    , m_dnn(DW_NULL_HANDLE)
    , m_dataConditioner(DW_NULL_HANDLE)
    , m_dnnInput(DW_NULL_HANDLE)
    , m_cudaStream(0)
    , m_imageWidth(imageWidth)
    , m_imageHeight(imageHeight)
    , m_useCuDLA(false)
    , m_dlaEngineNo(0)
    , m_inferenceCount(0)
    , m_inputLayout{}
    , m_detectionLayout{}
    , m_segmentationLayout{}
{
    // Initialize tensor handles
    for (uint32_t i = 0; i < NUM_OUTPUT_TENSORS; ++i) {
        m_dnnOutputsDevice[i] = DW_NULL_HANDLE;
        m_dnnOutputsHost[i] = DW_NULL_HANDLE;
        m_outputStreamers[i] = DW_NULL_HANDLE;
    }
    
    // Setup processing region (full image)
    m_processingRegion.x = 0;
    m_processingRegion.y = 0;
    m_processingRegion.width = m_imageWidth;
    m_processingRegion.height = m_imageHeight;
    
    log("DNNProcessor created for pomo-drivenet model: %ux%u images\n", m_imageWidth, m_imageHeight);
}

DNNProcessor::~DNNProcessor()
{
    releaseResources();
}

//#######################################################################################
// INITIALIZATION INTERFACE
//#######################################################################################

bool DNNProcessor::initialize(const std::string& modelPath, bool useCuDLA, uint32_t dlaEngine)
{
    m_useCuDLA = useCuDLA;
    m_dlaEngineNo = dlaEngine;
    
    log("Initializing pomo-drivenet DNN processor...\n");
    log("Model path: %s\n", modelPath.c_str());
    log("cuDLA enabled: %s\n", m_useCuDLA ? "Yes" : "No");
    if (m_useCuDLA) {
        log("DLA engine: %u\n", m_dlaEngineNo);
    }
    
    try {
        // Initialize DNN model
        if (!initializeDNNModel(modelPath)) {
            log("ERROR: Failed to initialize pomo-drivenet model\n");
            return false;
        }
        
        // Initialize 3-tensor infrastructure
        if (!initializeTensors()) {
            log("ERROR: Failed to initialize tensor infrastructure\n");
            return false;
        }
        
        // Initialize data conditioner
        if (!initializeDataConditioner()) {
            log("ERROR: Failed to initialize data conditioner\n");
            return false;
        }
        
        // Validate tensor properties against pomo-drivenet model
        if (!validateTensorProperties()) {
            log("ERROR: Tensor validation failed for pomo-drivenet model\n");
            return false;
        }
        
        log("pomo-drivenet DNN processor initialization completed successfully\n");
        return true;
        
    } catch (const std::exception& e) {
        log("ERROR: DNN initialization exception: %s\n", e.what());
        return false;
    }
}

bool DNNProcessor::isInitialized() const
{
    return (m_dnn != DW_NULL_HANDLE && 
            m_dataConditioner != DW_NULL_HANDLE &&
            m_dnnInput != DW_NULL_HANDLE);
}

//#######################################################################################
// INFERENCE INTERFACE
//#######################################################################################

bool DNNProcessor::processImage(dwImageHandle_t inputImage, InferenceResult& result)
{
    if (!isInitialized()) {
        log("ERROR: pomo-drivenet processor not initialized\n");
        result.isValid = false;
        return false;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        // Clear previous results
        result.detections.clear();
        result.segmentation.driveAreaMask.assign(SEGMENTATION_SIZE, 0);
        result.segmentation.laneLineMask.assign(SEGMENTATION_SIZE, 0);
        result.isValid = false;
        
        // Prepare input tensor
        if (!prepareInput(inputImage)) {
            log("ERROR: Input preparation failed\n");
            return false;
        }
        
        // Execute inference
        if (!runInference()) {
            log("ERROR: Inference execution failed\n");
            return false;
        }
        
        // Stream outputs to host memory
        if (!streamOutputsToHost()) {
            log("ERROR: Output streaming failed\n");
            return false;
        }
        
        // Interpret detection output (tensor 0: det_out)
        if (!interpretDetectionOutput(result.detections)) {
            log("ERROR: Detection interpretation failed\n");
            return false;
        }
        
        // Interpret segmentation outputs (tensors 1 & 2: drive_area_seg, lane_line_seg)
        if (!interpretSegmentationOutputs(result.segmentation)) {
            log("ERROR: Segmentation interpretation failed\n");
            return false;
        }
        
        // Return streamed outputs
        if (!returnStreamedOutputs()) {
            log("ERROR: Output return failed\n");
            return false;
        }
        
        // Calculate processing time
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        result.processingTimeMs = static_cast<uint32_t>(duration.count());
        
        result.isValid = true;
        m_inferenceCount++;
        
        // Log processing statistics periodically
        if (m_inferenceCount % 100 == 0) {
            log("pomo-drivenet processed %u frames, avg time: %ums\n", m_inferenceCount, result.processingTimeMs);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        log("ERROR: Process image exception: %s\n", e.what());
        result.isValid = false;
        return false;
    }
}

//#######################################################################################
// INTERNAL PROCESSING METHODS
//#######################################################################################

bool DNNProcessor::initializeDNNModel(const std::string& modelPath)
{
    std::string finalModelPath = modelPath;
    
    // Auto-detect model path if not provided
    if (finalModelPath.empty()) {
        finalModelPath = dw_samples::SamplesDataPath::get() + "/samples/detector/";
        finalModelPath += getPlatformPrefix();
        finalModelPath += "/pomo_drivenet";
        if (m_useCuDLA) {
            finalModelPath += ".dla";
        }
        finalModelPath += ".bin";
    }
    
    log("Loading pomo-drivenet TensorRT model: %s\n", finalModelPath.c_str());
    
    // Initialize DNN with platform-specific acceleration
    dwStatus status = dwDNN_initializeTensorRTFromFileWithEngineId(
        &m_dnn, 
        finalModelPath.c_str(), 
        nullptr,
        m_useCuDLA ? DW_PROCESSOR_TYPE_CUDLA : DW_PROCESSOR_TYPE_GPU,
        m_dlaEngineNo, 
        m_context
    );
    
    if (status != DW_SUCCESS) {
        log("ERROR: Failed to load pomo-drivenet model from %s\n", finalModelPath.c_str());
        return false;
    }
    
    // Set CUDA stream for asynchronous processing
    CHECK_DW_ERROR(dwDNN_setCUDAStream(m_cudaStream, m_dnn));
    
    log("pomo-drivenet model loaded successfully\n");
    return true;
}

//#######################################################################################
// TENSOR LAYOUT DETECTOR IMPLEMENTATION
//#######################################################################################

bool DNNProcessor::TensorLayoutDetector::detectInputLayout(const dwDNNTensorProperties& props, TensorLayout& layout)
{
    // Search for height=640, width=640, channels=3, batch=1 in any order
    std::vector<uint32_t> dimensions = {props.dimensionSize[0], props.dimensionSize[1], 
                                       props.dimensionSize[2], props.dimensionSize[3]};
    
    layout.isValid = false;
    
    for (uint32_t i = 0; i < 4; ++i) {
        if (dimensions[i] == 640) {
            if (layout.height_idx == UINT32_MAX) {
                layout.height_idx = i;
            } else if (layout.width_idx == UINT32_MAX) {
                layout.width_idx = i;
            }
        } else if (dimensions[i] == 3) {
            layout.channels_idx = i;
        } else if (dimensions[i] == 1) {
            layout.batch_idx = i;
        }
    }
    
    // Validate all required dimensions found
    if (layout.height_idx != UINT32_MAX && layout.width_idx != UINT32_MAX && 
        layout.channels_idx != UINT32_MAX && layout.batch_idx != UINT32_MAX) {
        layout.isValid = true;
        return true;
    }
    
    return false;
}

bool DNNProcessor::TensorLayoutDetector::detectDetectionLayout(const dwDNNTensorProperties& props, DetectionLayout& layout)
{
    layout.isValid = false;
    
    // Search for DETECTION_GRID_SIZE (25200) and DETECTION_FEATURES (6) in any dimension
    for (uint32_t i = 0; i < 4; ++i) {
        for (uint32_t j = 0; j < 4; ++j) {
            if (i != j && 
                props.dimensionSize[i] == DETECTION_GRID_SIZE && 
                props.dimensionSize[j] == DETECTION_FEATURES) {
                layout.grid_idx = i;
                layout.features_idx = j;
                layout.isValid = true;
                return true;
            }
        }
    }
    
    return false;
}

bool DNNProcessor::TensorLayoutDetector::detectSegmentationLayout(const dwDNNTensorProperties& props, SegmentationLayout& layout)
{
    layout.isValid = false;
    uint32_t spatialDims = 0;
    
    // Search for 640x640 spatial dimensions and 2 classes
    for (uint32_t i = 0; i < 4; ++i) {
        uint32_t dim = props.dimensionSize[i];
        if (dim == 640) {
            if (layout.height_idx == UINT32_MAX) {
                layout.height_idx = i;
            } else if (layout.width_idx == UINT32_MAX) {
                layout.width_idx = i;
            }
            spatialDims++;
        } else if (dim == 2) {
            layout.classes_idx = i;
        }
    }
    
    if (spatialDims == 2 && layout.classes_idx != UINT32_MAX) {
        layout.isValid = true;
        return true;
    }
    
    return false;
}




bool DNNProcessor::initializeTensors()
{
    // Get input tensor properties (images: [1, 3, 640, 640])
    dwDNNTensorProperties inputProps;
    CHECK_DW_ERROR(dwDNN_getInputTensorProperties(&inputProps, 0U, m_dnn));
    
    // Create input tensor
    CHECK_DW_ERROR(dwDNNTensor_create(&m_dnnInput, &inputProps, m_context));
    
    log("Input tensor (images): [%u, %u, %u, %u]\n", 
        inputProps.dimensionSize[0], inputProps.dimensionSize[1],
        inputProps.dimensionSize[2], inputProps.dimensionSize[3]);
    
    // Initialize 3 output tensors for pomo-drivenet model
    for (uint32_t i = 0; i < NUM_OUTPUT_TENSORS; ++i) {
        dwDNNTensorProperties outputProps;
        CHECK_DW_ERROR(dwDNN_getOutputTensorProperties(&outputProps, i, m_dnn));
        
        // Create device tensor
        CHECK_DW_ERROR(dwDNNTensor_create(&m_dnnOutputsDevice[i], &outputProps, m_context));
        
        // Create host tensor properties
        dwDNNTensorProperties hostProps = outputProps;
        hostProps.tensorType = DW_DNN_TENSOR_TYPE_CPU;
        
        // Create tensor streamer for GPU->CPU transfer
        CHECK_DW_ERROR(dwDNNTensorStreamer_initialize(&m_outputStreamers[i],
                                                      &outputProps, hostProps.tensorType, m_context));
        
        const char* outputName = "";
        switch (i) {
            case 0: outputName = "det_out"; break;
            case 1: outputName = "drive_area_seg"; break;
            case 2: outputName = "lane_line_seg"; break;
        }
        
        log("Output tensor %u (%s): [%u, %u, %u, %u]\n", i, outputName,
            outputProps.dimensionSize[0], outputProps.dimensionSize[1],
            outputProps.dimensionSize[2], outputProps.dimensionSize[3]);
    }
    
    log("pomo-drivenet tensor infrastructure initialized\n");
    return true;
}


bool DNNProcessor::initializeDataConditioner()
{
    // Get DNN metadata for data conditioning parameters
    dwDNNMetaData metadata;
    CHECK_DW_ERROR(dwDNN_getMetaData(&metadata, m_dnn));
    
    // Get input tensor properties
    dwDNNTensorProperties inputProps;
    CHECK_DW_ERROR(dwDNN_getInputTensorProperties(&inputProps, 0U, m_dnn));
    
    // Configure YOLO-compatible preprocessing parameters
    dwDataConditionerParams yoloParams = metadata.dataConditionerParams;
    
    // Set ImageNet normalization parameters for YOLO models
    // Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    yoloParams.meanValue[0] = 0.485f;
    yoloParams.meanValue[1] = 0.456f; 
    yoloParams.meanValue[2] = 0.406f;
    
    yoloParams.stdDev[0] = 0.229f;
    yoloParams.stdDev[1] = 0.224f;
    yoloParams.stdDev[2] = 0.225f;
    
    // Configure value range for proper normalization
    yoloParams.pixelRange[0] = 0.0f;
    yoloParams.pixelRange[1] = 1.0f;
    
    // Enable proper channel ordering for RGB input
    yoloParams.channelOrder = DW_IMAGE_CHANNEL_ORDER_RGB;
    
    // Initialize data conditioner with YOLO parameters
    CHECK_DW_ERROR(dwDataConditioner_initializeFromTensorProperties(
        &m_dataConditioner, 
        &inputProps, 
        1U,
        &yoloParams, 
        m_cudaStream,
        m_context
    ));
    
    log("Data conditioner initialized with YOLO-compatible preprocessing\n");
    log("Normalization - Mean: [%.3f, %.3f, %.3f], Std: [%.3f, %.3f, %.3f]\n",
        yoloParams.meanValue[0], yoloParams.meanValue[1], yoloParams.meanValue[2],
        yoloParams.stdDev[0], yoloParams.stdDev[1], yoloParams.stdDev[2]);
    
    return true;
}



bool DNNProcessor::prepareInput(dwImageHandle_t inputImage)
{
    // Use data conditioner to prepare input for the network
    CHECK_DW_ERROR(dwDataConditioner_prepareData(
        m_dnnInput, 
        &inputImage, 
        1, 
        &m_processingRegion,
        cudaAddressModeClamp, 
        m_dataConditioner
    ));
    
    return true;
}

bool DNNProcessor::runInference()
{
    // Prepare input array
    dwConstDNNTensorHandle_t inputs[1] = {m_dnnInput};
    
    // Execute inference on pomo-drivenet 3-output model
    CHECK_DW_ERROR(dwDNN_infer(m_dnnOutputsDevice, NUM_OUTPUT_TENSORS, inputs, 1U, m_dnn));
    
    return true;
}

bool DNNProcessor::streamOutputsToHost()
{
    for (uint32_t i = 0; i < NUM_OUTPUT_TENSORS; ++i) {
        // Send device tensor to streamer
        CHECK_DW_ERROR(dwDNNTensorStreamer_producerSend(m_dnnOutputsDevice[i], m_outputStreamers[i]));
        
        // Receive on host side
        CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReceive(&m_dnnOutputsHost[i], 1000, m_outputStreamers[i]));
    }
    
    return true;
}

bool DNNProcessor::returnStreamedOutputs()
{
    for (uint32_t i = 0; i < NUM_OUTPUT_TENSORS; ++i) {
        // Return host tensor
        CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReturn(&m_dnnOutputsHost[i], m_outputStreamers[i]));
        
        // Return to producer
        CHECK_DW_ERROR(dwDNNTensorStreamer_producerReturn(nullptr, 1000, m_outputStreamers[i]));
    }
    
    return true;
}

//#######################################################################################
// OUTPUT INTERPRETATION METHODS
//#######################################################################################
bool DNNProcessor::interpretDetectionOutput(std::vector<DetectionResult>& detections)
{
    // Lock detection output tensor
    void* outputData;
    CHECK_DW_ERROR(dwDNNTensor_lock(&outputData, m_dnnOutputsHost[0]));
    float32_t* detectionData = reinterpret_cast<float32_t*>(outputData);
    
    std::vector<DetectionResult> rawDetections;
    
    // Get tensor properties for stride calculation
    dwDNNTensorProperties detectionProps;
    CHECK_DW_ERROR(dwDNN_getOutputTensorProperties(&detectionProps, 0U, m_dnn));
    
    // Use validated layout for interpretation
    if (!m_detectionLayout.isValid) {
        log("ERROR: Detection layout not validated\n");
        CHECK_DW_ERROR(dwDNNTensor_unlock(m_dnnOutputsHost[0]));
        return false;
    }
    
    // Process detections with correct layout
    for (uint32_t i = 0; i < DETECTION_GRID_SIZE; ++i) {
        float32_t x, y, w, h, confidence, classId;
        
        // Calculate indices based on detected layout
        if (m_detectionLayout.grid_idx < m_detectionLayout.features_idx) {
            // Layout: [GRID, FEATURES, ?, ?]
            uint32_t baseIdx = i * detectionProps.dimensionSize[m_detectionLayout.features_idx];
            x = detectionData[baseIdx + 0];
            y = detectionData[baseIdx + 1];
            w = detectionData[baseIdx + 2];
            h = detectionData[baseIdx + 3];
            confidence = detectionData[baseIdx + 4];
            classId = detectionData[baseIdx + 5];
        } else {
            // Layout: [FEATURES, GRID, ?, ?]
            uint32_t stride = detectionProps.dimensionSize[m_detectionLayout.grid_idx];
            x = detectionData[0 * stride + i];
            y = detectionData[1 * stride + i];
            w = detectionData[2 * stride + i];
            h = detectionData[3 * stride + i];
            confidence = detectionData[4 * stride + i];
            classId = detectionData[5 * stride + i];
        }
        
        // Filter by confidence threshold (now 0.25f)
        if (confidence > CONFIDENCE_THRESHOLD) {
            DetectionResult detection;
            detection.bbox = convertToImageCoordinates(x, y, w, h);
            detection.confidence = confidence;
            detection.classId = static_cast<uint32_t>(classId);
            
            rawDetections.push_back(detection);
        }
    }
    
    CHECK_DW_ERROR(dwDNNTensor_unlock(m_dnnOutputsHost[0]));
    
    // Apply Non-Maximum Suppression
    detections = applyNMS(rawDetections, NMS_THRESHOLD);
    
    log("Detection interpretation: found %zu raw detections, %zu after NMS (threshold: %.2f)\n", 
        rawDetections.size(), detections.size(), CONFIDENCE_THRESHOLD);
    
    return true;
}


bool DNNProcessor::interpretSegmentationOutputs(SegmentationResult& segResult)
{
    // Process drive area segmentation (output 1: drive_area_seg [1, 2, 640, 640])
    void* driveAreaData;
    CHECK_DW_ERROR(dwDNNTensor_lock(&driveAreaData, m_dnnOutputsHost[1]));
    float32_t* driveAreaSeg = reinterpret_cast<float32_t*>(driveAreaData);
    
    convertSegmentationToMask(driveAreaSeg, segResult.driveAreaMask);
    segResult.drivePixelCount = std::count_if(segResult.driveAreaMask.begin(), 
                                             segResult.driveAreaMask.end(), 
                                             [](uint8_t p) { return p > 0; });
    
    CHECK_DW_ERROR(dwDNNTensor_unlock(m_dnnOutputsHost[1]));
    
    // Process lane line segmentation (output 2: lane_line_seg [1, 2, 640, 640])
    void* laneLineData;
    CHECK_DW_ERROR(dwDNNTensor_lock(&laneLineData, m_dnnOutputsHost[2]));
    float32_t* laneLineSeg = reinterpret_cast<float32_t*>(laneLineData);
    
    convertSegmentationToMask(laneLineSeg, segResult.laneLineMask);
    segResult.lanePixelCount = std::count_if(segResult.laneLineMask.begin(), 
                                            segResult.laneLineMask.end(), 
                                            [](uint8_t p) { return p > 0; });
    
    CHECK_DW_ERROR(dwDNNTensor_unlock(m_dnnOutputsHost[2]));
    
    return true;
}

std::vector<DetectionResult> DNNProcessor::applyNMS(std::vector<DetectionResult>& detections, float32_t threshold)
{
    if (detections.empty()) return detections;
    
    // Sort by confidence (highest first)
    std::sort(detections.begin(), detections.end(), 
        [](const DetectionResult& a, const DetectionResult& b) {
            return a.confidence > b.confidence;
        });
    
    std::vector<DetectionResult> nmsResults;
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        
        nmsResults.push_back(detections[i]);
        
        // Suppress overlapping detections of same class
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (!suppressed[j] && detections[i].classId == detections[j].classId) {
                float32_t iou = calculateIoU(detections[i].bbox, detections[j].bbox);
                if (iou > threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    
    return nmsResults;
}

//#######################################################################################
// UTILITY METHODS
//#######################################################################################

dwRectf DNNProcessor::convertToImageCoordinates(float32_t x, float32_t y, float32_t w, float32_t h)
{
    // Convert normalized coordinates to image pixel coordinates
    float32_t pixelX = x * static_cast<float32_t>(m_imageWidth);
    float32_t pixelY = y * static_cast<float32_t>(m_imageHeight);
    float32_t pixelW = w * static_cast<float32_t>(m_imageWidth);
    float32_t pixelH = h * static_cast<float32_t>(m_imageHeight);
    
    // Convert center coordinates to corner coordinates
    return {pixelX - pixelW/2.0f, pixelY - pixelH/2.0f, pixelW, pixelH};
}

float32_t DNNProcessor::calculateIoU(const dwRectf& box1, const dwRectf& box2)
{
    float32_t x1 = std::max(box1.x, box2.x);
    float32_t y1 = std::max(box1.y, box2.y);
    float32_t x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float32_t y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    float32_t intersectionWidth = std::max(0.0f, x2 - x1);
    float32_t intersectionHeight = std::max(0.0f, y2 - y1);
    float32_t intersectionArea = intersectionWidth * intersectionHeight;
    
    float32_t box1Area = box1.width * box1.height;
    float32_t box2Area = box2.width * box2.height;
    float32_t unionArea = box1Area + box2Area - intersectionArea;
    
    return (unionArea > 0.0f) ? (intersectionArea / unionArea) : 0.0f;
}


void DNNProcessor::convertSegmentationToMask(float32_t* segData, std::vector<uint8_t>& mask)
{
    mask.resize(SEGMENTATION_SIZE);
    
    // Get segmentation tensor properties to determine layout
    dwDNNTensorProperties segProps;
    CHECK_DW_ERROR(dwDNN_getOutputTensorProperties(&segProps, 1, m_dnn));
    
    // Use detected layout from validation
    if (!m_segmentationLayout.isValid) {
        log("WARNING: Using fallback segmentation layout\n");
        // Fallback to assumed layout
        m_segmentationLayout.height_idx = 0;
        m_segmentationLayout.width_idx = 1; 
        m_segmentationLayout.classes_idx = 2;
    }
    
    // Calculate strides based on detected tensor layout
    uint32_t heightStride = 1;
    uint32_t widthStride = 1;
    uint32_t classStride = 1;
    
    for (uint32_t i = 0; i < 4; ++i) {
        if (i > m_segmentationLayout.height_idx) heightStride *= segProps.dimensionSize[i];
        if (i > m_segmentationLayout.width_idx) widthStride *= segProps.dimensionSize[i];
        if (i > m_segmentationLayout.classes_idx) classStride *= segProps.dimensionSize[i];
    }
    
    // Process segmentation with numerically stable softmax
    for (uint32_t h = 0; h < 640; ++h) {
        for (uint32_t w = 0; w < 640; ++w) {
            // Calculate indices based on detected layout
            uint32_t baseIdx = h * heightStride + w * widthStride;
            
            // Get class logits
            float32_t class0_logit = segData[baseIdx];
            float32_t class1_logit = segData[baseIdx + classStride];
            
            // Numerically stable softmax computation
            float32_t maxLogit = std::max(class0_logit, class1_logit);
            float32_t exp0 = std::exp(class0_logit - maxLogit);
            float32_t exp1 = std::exp(class1_logit - maxLogit);
            float32_t sumExp = exp0 + exp1;
            
            // Calculate probabilities
            float32_t prob0 = exp0 / sumExp;
            float32_t prob1 = exp1 / sumExp;
            
            // Convert to binary mask with confidence threshold
            uint32_t pixelIdx = h * 640 + w;
            mask[pixelIdx] = (prob1 > 0.5f) ? 255 : 0;
        }
    }
}


std::string DNNProcessor::getPlatformPrefix()
{
    static const int32_t CUDA_AMPERE_MAJOR = 8;
    static const int32_t CUDA_TURING_VOLTA_MAJOR = 7;
    static const int32_t CUDA_VOLTA_DISCRETE_MINOR = 0;
    static const int32_t CUDA_VOLTA_INTEGRATED_MINOR = 2;
    static const int32_t CUDA_TURING_DISCRETE_MINOR = 5;

    if (m_useCuDLA) {
        return "cudla";
    }

    int32_t currentGPU;
    dwGPUDeviceProperties gpuProp{};
    CHECK_DW_ERROR(dwContext_getGPUDeviceCurrent(&currentGPU, m_context));
    CHECK_DW_ERROR(dwContext_getGPUProperties(&gpuProp, currentGPU, m_context));

    if (gpuProp.major == CUDA_AMPERE_MAJOR) {
        return gpuProp.integrated ? "ampere-integrated" : "ampere-discrete";
    } else if (gpuProp.major == CUDA_TURING_VOLTA_MAJOR) {
        if (gpuProp.minor == CUDA_TURING_DISCRETE_MINOR) {
            return "turing";
        } else if (gpuProp.minor == CUDA_VOLTA_INTEGRATED_MINOR) {
            return "volta-integrated";
        } else if (gpuProp.minor == CUDA_VOLTA_DISCRETE_MINOR) {
            return "volta-discrete";
        }
    }
    
    return "pascal";
}

bool DNNProcessor::validateTensorProperties()
{
    // Validate input tensor using dynamic layout detection
    dwDNNTensorProperties inputProps;
    CHECK_DW_ERROR(dwDNN_getInputTensorProperties(&inputProps, 0U, m_dnn));
    
    if (!TensorLayoutDetector::detectInputLayout(inputProps, m_inputLayout)) {
        log("ERROR: Input tensor layout incompatible with pomo-drivenet. Expected 640x640x3x1 in any order, got [%u, %u, %u, %u]\n",
            inputProps.dimensionSize[0], inputProps.dimensionSize[1],
            inputProps.dimensionSize[2], inputProps.dimensionSize[3]);
        return false;
    }
    
    log("Input tensor layout detected: [%u, %u, %u, %u] - H:%u, W:%u, C:%u, B:%u\n",
        inputProps.dimensionSize[0], inputProps.dimensionSize[1],
        inputProps.dimensionSize[2], inputProps.dimensionSize[3],
        m_inputLayout.height_idx, m_inputLayout.width_idx,
        m_inputLayout.channels_idx, m_inputLayout.batch_idx);
    
    // Validate output tensor count
    uint32_t numOutputs = 0;
    dwDNN_getOutputBlobCount(&numOutputs, m_dnn);
    if (numOutputs != NUM_OUTPUT_TENSORS) {
        log("ERROR: pomo-drivenet expected %u output tensors, found %u\n", NUM_OUTPUT_TENSORS, numOutputs);
        return false;
    }
    
    // Validate detection output using dynamic detection
    dwDNNTensorProperties detectionProps;
    CHECK_DW_ERROR(dwDNN_getOutputTensorProperties(&detectionProps, 0U, m_dnn));
    
    if (!TensorLayoutDetector::detectDetectionLayout(detectionProps, m_detectionLayout)) {
        log("ERROR: Detection tensor layout incompatible. Expected %ux%u in any dimension order, got [%u, %u, %u, %u]\n",
            DETECTION_GRID_SIZE, DETECTION_FEATURES,
            detectionProps.dimensionSize[0], detectionProps.dimensionSize[1], 
            detectionProps.dimensionSize[2], detectionProps.dimensionSize[3]);
        return false;
    }
    
    log("Detection tensor layout: grid_points at dim[%u]=%u, features at dim[%u]=%u\n",
        m_detectionLayout.grid_idx, detectionProps.dimensionSize[m_detectionLayout.grid_idx],
        m_detectionLayout.features_idx, detectionProps.dimensionSize[m_detectionLayout.features_idx]);
    
    // Validate segmentation outputs using dynamic detection
    for (uint32_t i = 1; i < NUM_OUTPUT_TENSORS; ++i) {
        dwDNNTensorProperties segProps;
        CHECK_DW_ERROR(dwDNN_getOutputTensorProperties(&segProps, i, m_dnn));
        
        SegmentationLayout tempLayout;
        if (!TensorLayoutDetector::detectSegmentationLayout(segProps, tempLayout)) {
            log("ERROR: Segmentation tensor %u layout incompatible. Expected 640x640x2 in any order, got [%u, %u, %u, %u]\n",
                i, segProps.dimensionSize[0], segProps.dimensionSize[1], 
                segProps.dimensionSize[2], segProps.dimensionSize[3]);
            return false;
        }
        
        // Store layout for first segmentation tensor (both should be identical)
        if (i == 1) {
            m_segmentationLayout = tempLayout;
        }
        
        log("Segmentation tensor %u layout: classes at dim[%u]=%u, spatial dims 640x640\n",
            i, tempLayout.classes_idx, segProps.dimensionSize[tempLayout.classes_idx]);
    }
    
    log("pomo-drivenet tensor validation passed with dynamic layout detection\n");
    return true;
}

void DNNProcessor::releaseResources()
{
    log("Releasing pomo-drivenet DNN processor resources...\n");
    
    // Release output tensors and streamers
    for (uint32_t i = 0; i < NUM_OUTPUT_TENSORS; ++i) {
        if (m_dnnOutputsHost[i] != DW_NULL_HANDLE) {
            dwDNNTensor_destroy(m_dnnOutputsHost[i]);
        }
        if (m_dnnOutputsDevice[i] != DW_NULL_HANDLE) {
            dwDNNTensor_destroy(m_dnnOutputsDevice[i]);
        }
        if (m_outputStreamers[i] != DW_NULL_HANDLE) {
            dwDNNTensorStreamer_release(m_outputStreamers[i]);
        }
    }
    
    // Release input tensor
    if (m_dnnInput != DW_NULL_HANDLE) {
        dwDNNTensor_destroy(m_dnnInput);
    }
    
    // Release data conditioner
    if (m_dataConditioner != DW_NULL_HANDLE) {
        dwDataConditioner_release(m_dataConditioner);
    }
    
    // Release DNN
    if (m_dnn != DW_NULL_HANDLE) {
        dwDNN_release(m_dnn);
    }
    
    log("pomo-drivenet DNN processor resources released\n");
}

} // namespace multicam_dnn