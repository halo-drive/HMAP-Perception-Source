/////////////////////////////////////////////////////////////////////////////////////////
// VehicleTypeNetProcessing.cpp 
/////////////////////////////////////////////////////////////////////////////////////////

#include "vehicleTypeNetProcessing.hpp"
#include <cstring>
#include <sstream>

constexpr const char* VehicleTypeNetProcessor::VEHICLE_TYPE_NAMES[6];

VehicleTypeNetProcessor::VehicleTypeNetProcessor(
    dwContextHandle_t sdk,
    cudaStream_t stream,
    const std::string& modelPath,
    dwProcessorType processorType,
    uint32_t dlaEngineNo)
    : m_sdk(sdk)
    , m_cudaStream(stream)
    , m_processorType(processorType)
    , m_dlaEngineNo(dlaEngineNo)
    , m_initialized(false)
    , m_dnn(DW_NULL_HANDLE)
    , m_dataConditioner(DW_NULL_HANDLE)
    , m_inputTensor(DW_NULL_HANDLE)
    , m_outputDeviceTensor(DW_NULL_HANDLE)
    , m_outputHostTensor(DW_NULL_HANDLE)
    , m_outputStreamer(DW_NULL_HANDLE)
{
    try {
        initializeDNN(modelPath);
        m_initialized = true;
    } catch (...) {
        releaseDNN();
        throw;
    }
}

VehicleTypeNetProcessor::~VehicleTypeNetProcessor()
{
    releaseDNN();
}

void VehicleTypeNetProcessor::initializeDNN(const std::string& modelPath)
{
    // Initialize DNN from TensorRT engine
    if (m_processorType == DW_PROCESSOR_TYPE_CUDLA) {
        CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFileWithEngineId(
            &m_dnn, 
            modelPath.c_str(), 
            nullptr,
            DW_PROCESSOR_TYPE_CUDLA,
            m_dlaEngineNo, 
            m_sdk));
    } else {
        CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFile(
            &m_dnn, 
            modelPath.c_str(), 
            nullptr,
            DW_PROCESSOR_TYPE_GPU,
            m_sdk));
    }

    // Set CUDA stream
    CHECK_DW_ERROR(dwDNN_setCUDAStream(m_cudaStream, m_dnn));

    // Get tensor properties
    CHECK_DW_ERROR(dwDNN_getInputTensorProperties(&m_inputProps, 0U, m_dnn));
    CHECK_DW_ERROR(dwDNN_getOutputTensorProperties(&m_outputProps, 0U, m_dnn));

    // === DIAGNOSTIC: Inspect actual tensor properties ===
    std::cout << "\n=== VehicleTypeNet Input Tensor ===" << std::endl;
    std::cout << "numDimensions: " << m_inputProps.numDimensions << std::endl;
    std::cout << "tensorLayout: ";
    switch(m_inputProps.tensorLayout) {
        case DW_DNN_TENSOR_LAYOUT_NCHW: std::cout << "NCHW"; break;
        case DW_DNN_TENSOR_LAYOUT_NHWC: std::cout << "NHWC"; break;
        default: std::cout << "UNKNOWN(" << static_cast<int>(m_inputProps.tensorLayout) << ")"; break;
    }
    std::cout << std::endl;
    std::cout << "dimensionSize array (reverse order of layout):" << std::endl;
    for (uint32_t i = 0; i < m_inputProps.numDimensions; ++i) {
        std::cout << "  [" << i << "] = " << m_inputProps.dimensionSize[i];
        if (m_inputProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NCHW && m_inputProps.numDimensions == 4) {
            const char* labels[4] = {"Width", "Height", "Channels", "Batch"};
            std::cout << " (" << labels[i] << ")";
        }
        std::cout << std::endl;
    }

    std::cout << "\n=== VehicleTypeNet Output Tensor ===" << std::endl;
    std::cout << "numDimensions: " << m_outputProps.numDimensions << std::endl;
    std::cout << "dimensionSize array:" << std::endl;
    for (uint32_t i = 0; i < m_outputProps.numDimensions; ++i) {
        std::cout << "  [" << i << "] = " << m_outputProps.dimensionSize[i] << std::endl;
    }
    std::cout << "===================================\n" << std::endl;

    // Validation based on REVERSE-ordered dimensionSize for NCHW
    if (m_inputProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NCHW) {
        if (m_inputProps.numDimensions != 4) {
            throw std::runtime_error("VehicleTypeNet: Expected 4D NCHW tensor");
        }
        // For NCHW: [0]=W, [1]=H, [2]=C, [3]=N
        if (m_inputProps.dimensionSize[0] != 224 ||   // Width
            m_inputProps.dimensionSize[1] != 224) {   // Height
            throw std::runtime_error("VehicleTypeNet: Expected 224x224 input");
        }
        if (m_inputProps.dimensionSize[2] != 3) {     // Channels
            throw std::runtime_error("VehicleTypeNet: Expected 3-channel RGB input");
        }
    } else if (m_inputProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NHWC) {
        // For NHWC: [0]=C, [1]=W, [2]=H, [3]=N
        if (m_inputProps.dimensionSize[1] != 224 ||   // Width
            m_inputProps.dimensionSize[2] != 224) {   // Height
            throw std::runtime_error("VehicleTypeNet: Expected 224x224 input");
        }
    }

    // Output validation - handle 4D tensor [1,1,6,1] (NCHW-like with classes in dim[2])
    if (m_outputProps.numDimensions == 4) {
        // Shape: [width=1, height=6, channels=1, batch=1] in reverse order
        // Classes are at dimensionSize[2] (height dimension in reverse NCHW)
        if (m_outputProps.dimensionSize[2] != 6) {
            std::stringstream ss;
            ss << "VehicleTypeNet: Expected 6 classes at dim[1], got " 
               << m_outputProps.dimensionSize[2];
            throw std::runtime_error(ss.str());
        }
    } else if (m_outputProps.numDimensions == 2) {
        // 2D fallback: [classes, batch] in reverse
        if (m_outputProps.dimensionSize[0] != 6) {
            throw std::runtime_error("VehicleTypeNet: Expected 6-class output");
        }
    } else {
        std::stringstream ss;
        ss << "VehicleTypeNet: Unexpected output dimensions: " 
           << m_outputProps.numDimensions;
        throw std::runtime_error(ss.str());
    }
    

    // Allocate input tensor
    CHECK_DW_ERROR(dwDNNTensor_create(&m_inputTensor, &m_inputProps, m_sdk));

    // Allocate output tensors (device and host)
    CHECK_DW_ERROR(dwDNNTensor_create(&m_outputDeviceTensor, &m_outputProps, m_sdk));
    
    dwDNNTensorProperties hostProps = m_outputProps;
    hostProps.tensorType = DW_DNN_TENSOR_TYPE_CPU;
    
    // Create streamer for device→host transfer
    CHECK_DW_ERROR(dwDNNTensorStreamer_initialize(
        &m_outputStreamer,
        &m_outputProps,
        hostProps.tensorType,
        m_sdk));

    // Initialize data conditioner
    dwDNNMetaData metadata;
    CHECK_DW_ERROR(dwDNN_getMetaData(&metadata, m_dnn));
    
    CHECK_DW_ERROR(dwDataConditioner_initializeFromTensorProperties(
        &m_dataConditioner,
        &m_inputProps, 1U,
        &metadata.dataConditionerParams,
        m_cudaStream,
        m_sdk));
}

VehicleTypeNetProcessor::ClassificationResult 
VehicleTypeNetProcessor::classify(
    dwImageHandle_t sourceImage,
    const dwRect& cropRegion,
    cudaStream_t stream)
{
    ClassificationResult result;

    try {
        
        // Set CUDA stream for this classification operation
        if (stream != 0 && stream != m_cudaStream)
        {
            CHECK_DW_ERROR(dwDNN_setCUDAStream(stream, m_dnn));
            m_cudaStream = stream;  // Update cached stream
        }
        
        // Prepare cropped input through data conditioner
        CHECK_DW_ERROR(dwDataConditioner_prepareData(
            m_inputTensor,
            &sourceImage, 1,
            &cropRegion,
            cudaAddressModeClamp,
            m_dataConditioner));

        // Run inference
        dwConstDNNTensorHandle_t inputs[1] = {m_inputTensor};
        CHECK_DW_ERROR(dwDNN_infer(
            &m_outputDeviceTensor, 1U,
            inputs, 1U,
            m_dnn));

        // Stream output to host
        CHECK_DW_ERROR(dwDNNTensorStreamer_producerSend(
            m_outputDeviceTensor,
            m_outputStreamer));
        
        CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReceive(
            &m_outputHostTensor, 1000,
            m_outputStreamer));

        // Parse classification result
        void* outputData;
        CHECK_DW_ERROR(dwDNNTensor_lock(&outputData, m_outputHostTensor));
        result = parseOutput(outputData);
        CHECK_DW_ERROR(dwDNNTensor_unlock(m_outputHostTensor));

        // Return streamed tensor
        CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReturn(
            &m_outputHostTensor,
            m_outputStreamer));
        CHECK_DW_ERROR(dwDNNTensorStreamer_producerReturn(
            nullptr, 1000,
            m_outputStreamer));

        result.valid = true;

    } catch (...) {
        result.valid = false;
    }

    return result;
}

VehicleTypeNetProcessor::ClassificationResult 
VehicleTypeNetProcessor::parseOutput(void* outputData)
{
    ClassificationResult result;
    
    // For shape [1,1,6,1], the 6 class values are contiguous in memory
    // regardless of dimension interpretation
    
    if (m_processorType == DW_PROCESSOR_TYPE_CUDLA) {
        // cuDLA outputs FP16
        dwFloat16_t* probs = reinterpret_cast<dwFloat16_t*>(outputData);
        
        float32_t maxProb = static_cast<float32_t>(probs[0]);
        uint8_t maxIdx = 0;
        
        for (uint8_t i = 1; i < 6; ++i) {
            float32_t prob = static_cast<float32_t>(probs[i]);
            if (prob > maxProb) {
                maxProb = prob;
                maxIdx = i;
            }
        }
        
        result.classIndex = maxIdx;
        result.confidence = maxProb;
        
    } else {
        // GPU outputs FP32
        float32_t* probs = reinterpret_cast<float32_t*>(outputData);
        
        float32_t maxProb = probs[0];
        uint8_t maxIdx = 0;
        
        for (uint8_t i = 1; i < 6; ++i) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIdx = i;
            }
        }
        
        result.classIndex = maxIdx;
        result.confidence = maxProb;
    }
    
    return result;
}

std::vector<VehicleTypeNetProcessor::ClassificationResult> 
VehicleTypeNetProcessor::classifyBatch(
    dwImageHandle_t sourceImage,
    const std::vector<dwRect>& cropRegions)
{
    // Sequential implementation (batch optimization future work)
    std::vector<ClassificationResult> results;
    results.reserve(cropRegions.size());
    
    for (const auto& region : cropRegions) {
        results.push_back(classify(sourceImage, region));
    }
    
    return results;
}

void VehicleTypeNetProcessor::reset()
{
    if (m_dnn != DW_NULL_HANDLE) {
        CHECK_DW_ERROR(dwDNN_reset(m_dnn));
    }
    if (m_dataConditioner != DW_NULL_HANDLE) {
        CHECK_DW_ERROR(dwDataConditioner_reset(m_dataConditioner));
    }
}

void VehicleTypeNetProcessor::releaseDNN()
{
    if (m_inputTensor != DW_NULL_HANDLE) {
        dwDNNTensor_destroy(m_inputTensor);
        m_inputTensor = DW_NULL_HANDLE;
    }
    
    if (m_outputDeviceTensor != DW_NULL_HANDLE) {
        dwDNNTensor_destroy(m_outputDeviceTensor);
        m_outputDeviceTensor = DW_NULL_HANDLE;
    }
    
    if (m_outputStreamer != DW_NULL_HANDLE) {
        dwDNNTensorStreamer_release(m_outputStreamer);
        m_outputStreamer = DW_NULL_HANDLE;
    }
    
    if (m_dataConditioner != DW_NULL_HANDLE) {
        dwDataConditioner_release(m_dataConditioner);
        m_dataConditioner = DW_NULL_HANDLE;
    }
    
    if (m_dnn != DW_NULL_HANDLE) {
        dwDNN_release(m_dnn);
        m_dnn = DW_NULL_HANDLE;
    }
}