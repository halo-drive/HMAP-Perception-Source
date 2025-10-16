/////////////////////////////////////////////////////////////////////////////////////////
// vehicleTypeNetProcessing.hpp
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef VEHICLETYPENETPROCESSING_HPP
#define VEHICLETYPENETPROCESSING_HPP

#include <dw/core/context/Context.h>
#include <dw/dnn/DNN.h>
#include <dw/dnn/dataconditioner/DataConditioner.h>
#include <dw/dnn/tensor/Tensor.h>
#include <dw/image/Image.h>
#include <dw/interop/streamer/TensorStreamer.h>

#include <framework/Checks.hpp>

#include <memory>
#include <string>
#include <vector>

class VehicleTypeNetProcessor
{
public:
    // Classification result structure
    struct ClassificationResult
    {
        uint8_t classIndex;      // 0-5 vehicle type class
        float32_t confidence;    // Softmax probability
        bool valid;              // False if inference failed
        
        ClassificationResult() 
            : classIndex(0), confidence(0.0f), valid(false) {}
    };

    // Vehicle type class names (configure based on actual model)
    static constexpr const char* VEHICLE_TYPE_NAMES[6] = {
        "coupe", "largevehicle", "sedan", "suv", "truck", "van"
    };

    /**
     * @brief Constructor - initializes DNN resources
     * 
     * @param sdk              DriveWorks context handle
     * @param stream           CUDA stream for async operations
     * @param modelPath        Path to TensorRT engine (.bin file)
     * @param processorType    DW_PROCESSOR_TYPE_GPU or DW_PROCESSOR_TYPE_CUDLA
     * @param dlaEngineNo      DLA engine index (0 or 1), ignored if GPU
     */
    VehicleTypeNetProcessor(
        dwContextHandle_t sdk,
        cudaStream_t stream,
        const std::string& modelPath,
        dwProcessorType processorType = DW_PROCESSOR_TYPE_GPU,
        uint32_t dlaEngineNo = 0);

    /**
     * @brief Destructor - releases all DNN resources
     */
    ~VehicleTypeNetProcessor();

    /**
     * @brief Classify a single vehicle crop
     * 
     * @param sourceImage  Full frame image (dwImageHandle_t with CUDA or CPU data)
     * @param cropRegion   ROI to extract and classify
     * @return Classification result with class and confidence
     */
    ClassificationResult classify(
        dwImageHandle_t sourceImage,
        const dwRect& cropRegion,
        cudaStream_t stream =0);

    /**
     * @brief Batch classification (future optimization)
     * 
     * @param sourceImage   Full frame image
     * @param cropRegions   Vector of ROIs to classify
     * @return Vector of classification results (same size as cropRegions)
     */
    std::vector<ClassificationResult> classifyBatch(
        dwImageHandle_t sourceImage,
        const std::vector<dwRect>& cropRegions);

    /**
     * @brief Reset internal state (call between frames if needed)
     */
    void reset();

    /**
     * @brief Get the processor type being used
     */
    dwProcessorType getProcessorType() const { return m_processorType; }

    /**
     * @brief Check if processor was initialized successfully
     */
    bool isInitialized() const { return m_initialized; }

private:
    // Resource handles
    dwContextHandle_t m_sdk;
    cudaStream_t m_cudaStream;
    dwProcessorType m_processorType;
    uint32_t m_dlaEngineNo;
    bool m_initialized;

    // DNN pipeline components
    dwDNNHandle_t m_dnn;
    dwDataConditionerHandle_t m_dataConditioner;
    
    // Tensor handles
    dwDNNTensorHandle_t m_inputTensor;
    dwDNNTensorHandle_t m_outputDeviceTensor;
    dwDNNTensorHandle_t m_outputHostTensor;

    
    // Device-to-host streamer
    dwDNNTensorStreamerHandle_t m_outputStreamer;

    // Tensor properties (cached from initialization)
    dwDNNTensorProperties m_inputProps;
    dwDNNTensorProperties m_outputProps;

    // Helper methods
    void initializeDNN(const std::string& modelPath);
    void releaseDNN();
    ClassificationResult parseOutput(void* outputData);
};

#endif // VEHICLETYPENETPROCESSING_HPP