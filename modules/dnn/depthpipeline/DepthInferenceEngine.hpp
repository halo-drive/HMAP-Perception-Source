#pragma once

#include <dw/core/context/Context.h>
#include <dw/dnn/DNN.h>
#include <dw/dnn/tensor/Tensor.h>
#include <dw/dnn/dataconditioner/DataConditioner.h>
#include <dw/image/Image.h>

#include <cuda_runtime.h>
#include <atomic>
#include <memory>
#include <queue>
#include <mutex>
#include <chrono>

namespace depth_pipeline {

class DepthInferenceEngine {
public:
    static constexpr uint32_t DEPTH_MODEL_WIDTH = 924;
    static constexpr uint32_t DEPTH_MODEL_HEIGHT = 518;
    static constexpr uint32_t MAX_STREAMS = 4;
    
    struct InferenceRequest {
        uint32_t cameraId;
        uint64_t frameId;
        dwImageHandle_t imageRGBA;
        std::chrono::high_resolution_clock::time_point submitTime;
    };
    
    struct InferenceResult {
        uint32_t cameraId;
        uint64_t frameId;
        dwDNNTensorHandle_t depthTensorDevice;
        float minDepth;
        float maxDepth;
        std::chrono::microseconds inferenceTime;
        dwStatus status;
    };
    
    struct PerformanceMetrics {
        float avgInferenceMs[MAX_STREAMS];
        uint32_t completedInferences[MAX_STREAMS];
        uint32_t droppedRequests[MAX_STREAMS];
        float currentThroughputFPS;
    };

    explicit DepthInferenceEngine(dwContextHandle_t context);
    ~DepthInferenceEngine();
    
    dwStatus initialize(const std::string& modelPath, uint32_t numStreams = 4);
    dwStatus submitInference(const InferenceRequest& request);
    uint32_t pollResults(std::vector<InferenceResult>& results, uint32_t maxResults = 4);
    void releaseResult(const InferenceResult& result);
    PerformanceMetrics getMetrics() const;
    void resetMetrics();

private:
    struct StreamContext {
        // Per-stream DNN handle (ADDED - was missing)
        dwDNNHandle_t dnnHandle;
        
        cudaStream_t cudaStream;
        dwDNNTensorHandle_t inputTensor;
        dwDNNTensorHandle_t outputTensorDevice;
        dwDataConditionerHandle_t dataConditioner;
        
        cudaEvent_t inferenceComplete;
        std::atomic<bool> busy;
        
        InferenceRequest currentRequest;
        std::chrono::high_resolution_clock::time_point inferenceStartTime;
        
        uint32_t completedCount;
        float totalInferenceTimeMs;
    };
    
    dwContextHandle_t m_context;
    
    uint32_t m_numStreams;
    std::unique_ptr<StreamContext[]> m_streams;
    
    std::atomic<uint32_t> m_nextStreamIdx;
    
    std::queue<InferenceResult> m_completedResults;
    std::mutex m_resultsMutex;
    
    std::chrono::high_resolution_clock::time_point m_lastMetricsUpdate;
    uint32_t m_inferencesSinceLastUpdate;
    
    uint32_t findAvailableStream();
    
    // FIXED SIGNATURE: removed request parameter (stored in ctx already)
    dwStatus processInference(uint32_t streamIdx);
    
    void checkStreamCompletion(uint32_t streamIdx);
    
    // FIXED SIGNATURE: changed from (dwDNNTensorHandle_t, float*, float*, cudaStream_t)
    void computeDepthStats(dwDNNTensorHandle_t depthTensor,
                          float* minDepth, float* maxDepth,
                          cudaStream_t stream);
    
    DepthInferenceEngine(const DepthInferenceEngine&) = delete;
    DepthInferenceEngine& operator=(const DepthInferenceEngine&) = delete;
};

} // namespace depth_pipeline