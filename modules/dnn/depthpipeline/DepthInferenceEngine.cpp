// DepthInferenceEngine.cpp
#include "DepthInferenceEngine.hpp"
#include <framework/Checks.hpp>
#include <framework/Log.hpp>

#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>

namespace depth_pipeline {

// Forward declaration of CUDA kernel launcher (implemented in depth_kernels.cu)
extern void launchMinMaxReductionKernel(const float* input, uint32_t numElements,
                                        float* minResult, float* maxResult,
                                        cudaStream_t stream);

DepthInferenceEngine::DepthInferenceEngine(dwContextHandle_t context)
    : m_context(context)
    , m_numStreams(0)
    , m_nextStreamIdx(0)
    , m_lastMetricsUpdate(std::chrono::high_resolution_clock::now())
    , m_inferencesSinceLastUpdate(0)
{
}

DepthInferenceEngine::~DepthInferenceEngine()
{
    std::cout << "Releasing depth inference engine resources..." << std::endl;
    
    if (m_streams) {
        for (uint32_t i = 0; i < m_numStreams; ++i) {
            StreamContext& ctx = m_streams[i];
            
            if (ctx.inferenceComplete) {
                cudaEventDestroy(ctx.inferenceComplete);
            }
            
            if (ctx.dnnHandle != DW_NULL_HANDLE) {
                dwStatus status = dwDNN_release(ctx.dnnHandle);
                if (status != DW_SUCCESS) {
                    std::cerr << "WARNING: Failed to release DNN " << i 
                             << ": " << dwGetStatusName(status) << std::endl;
                }
            }
            
            if (ctx.inputTensor != DW_NULL_HANDLE) {
                dwDNNTensor_destroy(ctx.inputTensor);
            }
            if (ctx.outputTensorDevice != DW_NULL_HANDLE) {
                dwDNNTensor_destroy(ctx.outputTensorDevice);
            }
            
            if (ctx.dataConditioner != DW_NULL_HANDLE) {
                dwDataConditioner_release(ctx.dataConditioner);
            }
            
            if (ctx.cudaStream) {
                cudaStreamDestroy(ctx.cudaStream);
            }
        }
    }
    
    std::cout << "Depth inference engine released" << std::endl;
}

dwStatus DepthInferenceEngine::initialize(const std::string& modelPath, uint32_t numStreams)
{
    if (numStreams == 0 || numStreams > MAX_STREAMS) {
        std::cerr << "ERROR: Invalid number of streams: " << numStreams 
                 << " (must be 1-" << MAX_STREAMS << ")" << std::endl;
        return DW_INVALID_ARGUMENT;
    }
    
    m_numStreams = numStreams;
    m_streams.reset(new StreamContext[m_numStreams]);
    
    std::cout << "Initializing depth inference engine with " << m_numStreams 
             << " parallel streams" << std::endl;
    
    std::string baseModelPath = modelPath;
    size_t extensionPos = baseModelPath.rfind(".bin");
    std::string modelBaseName, modelExtension;
    
    if (extensionPos != std::string::npos) {
        modelBaseName = baseModelPath.substr(0, extensionPos);
        modelExtension = ".bin";
    } else {
        modelBaseName = baseModelPath;
        modelExtension = "";
    }
    
    std::cout << "Model base path: " << modelBaseName << std::endl;
    
    for (uint32_t i = 0; i < m_numStreams; ++i) {
        StreamContext& ctx = m_streams[i];
        
        std::cout << "\nInitializing stream " << i << "..." << std::endl;
        
        std::stringstream modelPathStream;
        if (m_numStreams == 1) {
            modelPathStream << baseModelPath;
        } else {
            modelPathStream << modelBaseName << "_cam" << i << modelExtension;
        }
        std::string cameraModelPath = modelPathStream.str();
        
        std::cout << "  Loading model: " << cameraModelPath << std::endl;
        
        dwStatus status = dwDNN_initializeTensorRTFromFile(
            &ctx.dnnHandle,
            cameraModelPath.c_str(),
            nullptr,
            DW_PROCESSOR_TYPE_GPU,
            m_context
        );
        
        if (status != DW_SUCCESS) {
            std::cerr << "ERROR: Failed to load DNN model for stream " << i 
                     << " from " << cameraModelPath << ": " 
                     << dwGetStatusName(status) << std::endl;
            
            if (m_numStreams > 1) {
                std::cerr << "\nHINT: For multi-camera operation, create copies of the model:" << std::endl;
                std::cerr << "  cp " << baseModelPath << " " << modelBaseName << "_cam0.bin" << std::endl;
                std::cerr << "  cp " << baseModelPath << " " << modelBaseName << "_cam1.bin" << std::endl;
                std::cerr << "  cp " << baseModelPath << " " << modelBaseName << "_cam2.bin" << std::endl;
                std::cerr << "  cp " << baseModelPath << " " << modelBaseName << "_cam3.bin" << std::endl;
            }
            
            return status;
        }
        
        dwDNNTensorProperties inputProps, outputProps;
        CHECK_DW_ERROR(dwDNN_getInputTensorProperties(&inputProps, 0, ctx.dnnHandle));
        CHECK_DW_ERROR(dwDNN_getOutputTensorProperties(&outputProps, 0, ctx.dnnHandle));
        
        if (i == 0) {
            std::cout << "  Model input dimensions: ";
            for (uint32_t d = 0; d < inputProps.numDimensions; ++d) {
                std::cout << inputProps.dimensionSize[d];
                if (d < inputProps.numDimensions - 1) std::cout << "×";
            }
            std::cout << " (layout: " << inputProps.tensorLayout << ")" << std::endl;
            
            bool validDimensions = false;
            if (inputProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NCHW && inputProps.numDimensions == 4) {
                uint32_t width = inputProps.dimensionSize[0];
                uint32_t height = inputProps.dimensionSize[1];
                if (width == DEPTH_MODEL_WIDTH && height == DEPTH_MODEL_HEIGHT) {
                    validDimensions = true;
                }
            }
            
            if (!validDimensions) {
                std::cerr << "ERROR: Model dimensions do not match expected " 
                         << DEPTH_MODEL_WIDTH << "×" << DEPTH_MODEL_HEIGHT << std::endl;
                return DW_INVALID_ARGUMENT;
            }
        }
        
        cudaError_t cudaStatus = cudaStreamCreateWithFlags(&ctx.cudaStream, cudaStreamNonBlocking);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "ERROR: Failed to create CUDA stream " << i 
                     << ": " << cudaGetErrorString(cudaStatus) << std::endl;
            return DW_CUDA_ERROR;
        }
        
        CHECK_DW_ERROR(dwDNN_setCUDAStream(ctx.cudaStream, ctx.dnnHandle));
        
        CHECK_DW_ERROR(dwDNNTensor_create(&ctx.inputTensor, &inputProps, m_context));
        CHECK_DW_ERROR(dwDNNTensor_create(&ctx.outputTensorDevice, &outputProps, m_context));
        
        dwDNNMetaData metadata;
        CHECK_DW_ERROR(dwDNN_getMetaData(&metadata, ctx.dnnHandle));
        
        CHECK_DW_ERROR(dwDataConditioner_initializeFromTensorProperties(
            &ctx.dataConditioner,
            &inputProps,
            1U,
            &metadata.dataConditionerParams,
            ctx.cudaStream,
            m_context
        ));
        
        cudaStatus = cudaEventCreateWithFlags(&ctx.inferenceComplete, cudaEventDisableTiming);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "ERROR: Failed to create completion event for stream " << i 
                     << ": " << cudaGetErrorString(cudaStatus) << std::endl;
            return DW_CUDA_ERROR;
        }
        
        ctx.busy.store(false);
        ctx.completedCount = 0;
        ctx.totalInferenceTimeMs = 0.0f;
        
        std::cout << "Stream " << i << " initialization complete" << std::endl;
    }
    
    std::cout << "\n=== Depth Inference Engine Ready ===" << std::endl;
    
    return DW_SUCCESS;
}

dwStatus DepthInferenceEngine::submitInference(const InferenceRequest& request)
{
    if (request.cameraId >= m_numStreams) {
        std::cerr << "ERROR: Invalid camera ID " << request.cameraId 
                 << " (max: " << m_numStreams - 1 << ")" << std::endl;
        return DW_INVALID_ARGUMENT;
    }
    
    if (request.imageRGBA == DW_NULL_HANDLE) {
        std::cerr << "ERROR: Null image handle in inference request" << std::endl;
        return DW_INVALID_HANDLE;
    }
    
    StreamContext& ctx = m_streams[request.cameraId];
    
    if (ctx.busy.load(std::memory_order_acquire)) {
        return DW_BUFFER_FULL;
    }
    
    ctx.busy.store(true, std::memory_order_release);
    ctx.currentRequest = request;
    ctx.inferenceStartTime = std::chrono::high_resolution_clock::now();
    
    dwStatus status = processInference(request.cameraId);
    if (status != DW_SUCCESS) {
        ctx.busy.store(false, std::memory_order_release);
        return status;
    }
    
    return DW_SUCCESS;
}

uint32_t DepthInferenceEngine::pollResults(std::vector<InferenceResult>& results, uint32_t maxResults)
{
    uint32_t collectedCount = 0;
    
    for (uint32_t i = 0; i < m_numStreams && collectedCount < maxResults; ++i) {
        StreamContext& ctx = m_streams[i];
        
        if (!ctx.busy.load(std::memory_order_acquire)) {
            continue;
        }
        
        cudaError_t eventStatus = cudaEventQuery(ctx.inferenceComplete);
        
        if (eventStatus == cudaSuccess) {
            InferenceResult result;
            result.cameraId = ctx.currentRequest.cameraId;
            result.frameId = ctx.currentRequest.frameId;
            result.depthTensorDevice = ctx.outputTensorDevice;
            result.status = DW_SUCCESS;
            
            auto endTime = std::chrono::high_resolution_clock::now();
            result.inferenceTime = std::chrono::duration_cast<std::chrono::microseconds>(
                endTime - ctx.inferenceStartTime
            );
            
            computeDepthStats(ctx.outputTensorDevice, &result.minDepth, &result.maxDepth, ctx.cudaStream);
            
            float inferenceMs = result.inferenceTime.count() / 1000.0f;
            ctx.totalInferenceTimeMs += inferenceMs;
            ctx.completedCount++;
            
            results.push_back(result);
            collectedCount++;
            
        } else if (eventStatus != cudaErrorNotReady) {
            std::cerr << "ERROR: CUDA error in stream " << i 
                     << ": " << cudaGetErrorString(eventStatus) << std::endl;
            
            InferenceResult errorResult;
            errorResult.cameraId = ctx.currentRequest.cameraId;
            errorResult.frameId = ctx.currentRequest.frameId;
            errorResult.depthTensorDevice = DW_NULL_HANDLE;
            errorResult.minDepth = 0.0f;
            errorResult.maxDepth = 0.0f;
            errorResult.inferenceTime = std::chrono::microseconds(0);
            errorResult.status = DW_CUDA_ERROR;
            
            results.push_back(errorResult);
            collectedCount++;
            
            ctx.busy.store(false, std::memory_order_release);
        }
    }
    
    return collectedCount;
}

void DepthInferenceEngine::releaseResult(const InferenceResult& result)
{
    if (result.cameraId >= m_numStreams) {
        std::cerr << "WARNING: Cannot release result - invalid camera ID " 
                 << result.cameraId << std::endl;
        return;
    }
    
    StreamContext& ctx = m_streams[result.cameraId];
    cudaStreamSynchronize(ctx.cudaStream);
    ctx.busy.store(false, std::memory_order_release);
}

DepthInferenceEngine::PerformanceMetrics DepthInferenceEngine::getMetrics() const
{
    PerformanceMetrics metrics;
    
    uint32_t totalCompleted = 0;
    
    for (uint32_t i = 0; i < m_numStreams; ++i) {
        const StreamContext& ctx = m_streams[i];
        
        if (ctx.completedCount > 0) {
            metrics.avgInferenceMs[i] = ctx.totalInferenceTimeMs / ctx.completedCount;
        } else {
            metrics.avgInferenceMs[i] = 0.0f;
        }
        
        metrics.completedInferences[i] = ctx.completedCount;
        metrics.droppedRequests[i] = 0;
        
        totalCompleted += ctx.completedCount;
    }
    
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(
        currentTime - m_lastMetricsUpdate
    ).count();
    
    if (elapsedSeconds > 0 && totalCompleted > 0) {
        metrics.currentThroughputFPS = static_cast<float>(totalCompleted) / elapsedSeconds;
    } else {
        metrics.currentThroughputFPS = 0.0f;
    }
    
    return metrics;
}

void DepthInferenceEngine::resetMetrics()
{
    for (uint32_t i = 0; i < m_numStreams; ++i) {
        m_streams[i].completedCount = 0;
        m_streams[i].totalInferenceTimeMs = 0.0f;
    }
    
    m_lastMetricsUpdate = std::chrono::high_resolution_clock::now();
    m_inferencesSinceLastUpdate = 0;
}

dwStatus DepthInferenceEngine::processInference(uint32_t streamIdx)
{
    StreamContext& ctx = m_streams[streamIdx];
    const InferenceRequest& request = ctx.currentRequest;
    
    dwRect roi;
    roi.x = 0;
    roi.y = 0;
    
    dwImageProperties srcProps;
    dwStatus status = dwImage_getProperties(&srcProps, request.imageRGBA);
    if (status != DW_SUCCESS) {
        std::cerr << "ERROR: Failed to get source image properties: " 
                 << dwGetStatusName(status) << std::endl;
        return status;
    }
    
    roi.width = srcProps.width;
    roi.height = srcProps.height;
    
    status = dwDataConditioner_prepareData(
        ctx.inputTensor,
        &request.imageRGBA,
        1,
        &roi,
        cudaAddressModeClamp,
        ctx.dataConditioner
    );
    
    if (status != DW_SUCCESS) {
        std::cerr << "ERROR: Data conditioning failed for stream " << streamIdx 
                 << ": " << dwGetStatusName(status) << std::endl;
        return status;
    }
    
    dwConstDNNTensorHandle_t inputs[1] = {ctx.inputTensor};
    dwDNNTensorHandle_t outputs[1] = {ctx.outputTensorDevice};
    
    status = dwDNN_infer(outputs, 1, inputs, 1, ctx.dnnHandle);
    if (status != DW_SUCCESS) {
        std::cerr << "ERROR: Inference failed for stream " << streamIdx 
                 << ": " << dwGetStatusName(status) << std::endl;
        return status;
    }
    
    cudaError_t cudaStatus = cudaEventRecord(ctx.inferenceComplete, ctx.cudaStream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "ERROR: Failed to record completion event for stream " << streamIdx 
                 << ": " << cudaGetErrorString(cudaStatus) << std::endl;
        return DW_CUDA_ERROR;
    }
    
    return DW_SUCCESS;
}

void DepthInferenceEngine::computeDepthStats(dwDNNTensorHandle_t depthTensor,
                                             float* minDepth, float* maxDepth,
                                             cudaStream_t stream)
{
    void* tensorData = nullptr;
    dwStatus status = dwDNNTensor_lock(&tensorData, depthTensor);
    if (status != DW_SUCCESS || tensorData == nullptr) {
        *minDepth = 0.0f;
        *maxDepth = 1.0f;
        return;
    }
    
    dwDNNTensorProperties props;
    dwDNNTensor_getProperties(&props, depthTensor);
    
    uint32_t numElements = 1;
    for (uint32_t i = 0; i < props.numDimensions; ++i) {
        numElements *= props.dimensionSize[i];
    }
    
    static float* d_minResult = nullptr;
    static float* d_maxResult = nullptr;
    if (d_minResult == nullptr) {
        cudaMalloc(&d_minResult, sizeof(float));
        cudaMalloc(&d_maxResult, sizeof(float));
    }
    
    launchMinMaxReductionKernel(
        reinterpret_cast<const float*>(tensorData),
        numElements,
        d_minResult,
        d_maxResult,
        stream
    );
    
    cudaMemcpyAsync(minDepth, d_minResult, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(maxDepth, d_maxResult, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    dwDNNTensor_unlock(depthTensor);
}

} // namespace depth_pipeline