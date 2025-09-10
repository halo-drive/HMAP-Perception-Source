#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <driver_types.h>
#include <texture_types.h>
#include <cublas_v2.h>

// TensorRT includes
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

// DriveWorks core includes (keeping only non-DNN related)
#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/base/GeometricTypes.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/core/platform/GPUProperties.h>
#include <dw/image/Image.h>
#include <dw/imageprocessing/geometry/imagetransformation/ImageTransformation.h>
#include <dw/interop/streamer/ImageStreamer.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/camera/Camera.h>
#include <dw/rig/Rig.h>

#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Visualization.h>
#include <dwvisualization/image/Image.h>

#include <framework/Checks.hpp>
#include <framework/CudaTimer.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/Log.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/SimpleStreamer.hpp>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

using namespace dw_samples::common;


// ============================================================================
// TENSORRT LOGGER CLASS
// ============================================================================
class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "[TRT INTERNAL ERROR] " << msg << std::endl;
                break;
            case Severity::kERROR:
                std::cerr << "[TRT ERROR] " << msg << std::endl;
                break;
            case Severity::kWARNING:
                std::cout << "[TRT WARNING] " << msg << std::endl;
                break;
            case Severity::kINFO:
                std::cout << "[TRT INFO] " << msg << std::endl;
                break;
            case Severity::kVERBOSE:
                std::cout << "[TRT VERBOSE] " << msg << std::endl;
                break;
        }
    }
};

// ============================================================================
// CUDA KERNEL DECLARATIONS
// ============================================================================

__global__ void resizePrototypesKernelChannelFirst(
    const float* __restrict__ prototypes,
    float* __restrict__ resized,
    int C, int H_proto, int W_proto,
    int H_det, int W_det);

__global__ void resizePrototypesKernelChannelLast(
    const float* __restrict__ prototypes,
    float* __restrict__ resized,
    int C, int H_proto, int W_proto,
    int H_det, int W_det);

__global__ void preprocessImageKernel(
    const uint8_t* input_rgba,
    float* output_rgb,
    int width, int height,
    int input_pitch, int output_pitch,
    float scale, float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b);

extern "C" {
    // Enhanced prototype processing kernels
    void launchResizePrototypesKernel(
        const float* prototypes, float* resized,
        int C, int H_proto, int W_proto, int H_det, int W_det,
        cudaStream_t stream);
        
    void launchSigmoidThresholdKernel(
        const float* linear_masks, uint8_t* binary_masks,
        int N, int HW, float threshold, cudaStream_t stream);
    
    // Mask analysis and combination kernels
    void launchOrReduceMasksKernel(
        const uint8_t* masks, uint8_t* combined,
        int N, int H, int W, cudaStream_t stream);
    
    void launchRowBoundaryDetectionKernel(
        const uint8_t* combined_mask, int32_t* left_bounds, int32_t* right_bounds,
        int H, int W, cudaStream_t stream);
    
    void launchBuildLaneMasksKernel(
        const int32_t* left_bounds, const int32_t* right_bounds,
        uint8_t* left_mask, uint8_t* right_mask, uint8_t* area_mask,
        int H, int W, cudaStream_t stream);
    
    // Image preprocessing kernel
    void launchPreprocessImageKernel(
        const uint8_t* input_rgba, float* output_rgb,
        int width, int height, int input_pitch, int output_pitch,
        float scale, float mean_r, float mean_g, float mean_b,
        float std_r, float std_g, float std_b, cudaStream_t stream);
    
    // Legacy kernel compatibility
    void launchMaxCombineKernel(uint8_t* dest, size_t destPitch,
                               const uint8_t* src, size_t srcPitch,
                               uint32_t width, uint32_t height,
                               cudaStream_t stream);
    
    void launchMorphologyKernel(uint8_t* dest, size_t destPitch,
                               const uint8_t* src, size_t srcPitch,
                               uint32_t width, uint32_t height,
                               int kernelSize, bool isDilation,
                               cudaStream_t stream);
}

// ============================================================================
// GPU MEMORY MANAGEMENT HELPER
// ============================================================================

template<typename T>
class CudaBuffer {
private:
    T* m_ptr = nullptr;
    size_t m_size = 0;
    size_t m_count = 0;
    std::string m_name;
    
public:
    CudaBuffer() = default;
    
    explicit CudaBuffer(size_t count, const std::string& name = "unnamed") 
        : m_ptr(nullptr), m_size(0), m_count(0), m_name(name) {
        if (count > 0) {
            allocate(count);
        }
    }
    
    ~CudaBuffer() {
        free();
    }
    
    CudaBuffer(CudaBuffer&& other) noexcept 
        : m_ptr(other.m_ptr), m_size(other.m_size), m_count(other.m_count), m_name(std::move(other.m_name)) {
        other.m_ptr = nullptr;
        other.m_size = 0;
        other.m_count = 0;
    }
    
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            free();
            m_ptr = other.m_ptr;
            m_size = other.m_size;
            m_count = other.m_count;
            m_name = std::move(other.m_name);
            other.m_ptr = nullptr;
            other.m_size = 0;
            other.m_count = 0;
        }
        return *this;
    }
    
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    
    void allocate(size_t count) {
        free();
        
        if (count == 0) {
            std::cout << "WARNING: " << m_name << " - Attempting to allocate 0 elements" << std::endl;
            return;
        }
        
        m_count = count;
        m_size = count * sizeof(T);
        
        std::cout << "ALLOC " << m_name << ": " << count << " elements (" 
                  << m_size << " bytes)" << std::endl;
        
        cudaError_t err = cudaMalloc(&m_ptr, m_size);
        if (err != cudaSuccess) {
            logError("CudaBuffer::allocate '%s': Failed to allocate %zu bytes: %s\n", 
                    m_name.c_str(), m_size, cudaGetErrorString(err));
            m_ptr = nullptr;
            m_size = 0;
            m_count = 0;
            CHECK_CUDA_ERROR(err);
            return;
        }
        
        std::cout << "ALLOC " << m_name << ": SUCCESS - ptr=" << (void*)m_ptr << std::endl;
        
        cudaError_t initErr = cudaMemset(m_ptr, 0, m_size);
        if (initErr != cudaSuccess) {
            logError("CudaBuffer::allocate '%s': Memory initialization failed: %s\n", 
                    m_name.c_str(), cudaGetErrorString(initErr));
            cudaFree(m_ptr);
            m_ptr = nullptr;
            m_size = 0;
            m_count = 0;
            CHECK_CUDA_ERROR(initErr);
        }
    }
    
    void free() {
        if (m_ptr) {
            std::cout << "FREE " << m_name << ": ptr=" << (void*)m_ptr << std::endl;
            cudaFree(m_ptr);
            m_ptr = nullptr;
            m_size = 0;
            m_count = 0;
        }
    }
    
    T* get() const { return m_ptr; }
    size_t size() const { return m_size; }
    size_t count() const { return m_count; }
    bool isValid() const { return m_ptr != nullptr && m_size > 0; }
    const std::string& name() const { return m_name; }
    
    void zero() {
        if (m_ptr && m_size > 0) {
            std::cout << "ZERO " << m_name << ": " << m_size << " bytes" << std::endl;
            CHECK_CUDA_ERROR(cudaMemset(m_ptr, 0, m_size));
        }
    }
};

// ============================================================================
// TENSORRT ENGINE WRAPPER
// ============================================================================
class TensorRTEngine {
private:
    TensorRTLogger m_logger;
    nvinfer1::IRuntime* m_runtime = nullptr;
    nvinfer1::ICudaEngine* m_engine = nullptr;
    nvinfer1::IExecutionContext* m_context = nullptr;
    
    std::vector<void*> m_bindings;
    std::vector<size_t> m_bindingSizes;
    std::vector<nvinfer1::Dims> m_bindingDims;
    std::vector<nvinfer1::DataType> m_bindingDataTypes;
    std::vector<std::string> m_bindingNames;
    
    int m_inputIndex = -1;
    std::vector<int> m_outputIndices;
    
public:
    TensorRTEngine() = default;
    ~TensorRTEngine() { destroy(); }
    
    TensorRTEngine(const TensorRTEngine&) = delete;
    TensorRTEngine& operator=(const TensorRTEngine&) = delete;
    
    bool loadEngine(const std::string& enginePath) {
        std::ifstream file(enginePath, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Failed to open engine file: " << enginePath << std::endl;
            return false;
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> engineData(size);
        file.read(engineData.data(), size);
        file.close();
        
        m_runtime = nvinfer1::createInferRuntime(m_logger);
        if (!m_runtime) {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }
        
        m_engine = m_runtime->deserializeCudaEngine(engineData.data(), size);
        if (!m_engine) {
            std::cerr << "Failed to deserialize CUDA engine" << std::endl;
            return false;
        }
        
        m_context = m_engine->createExecutionContext();
        if (!m_context) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }
        
        return setupBindings();
    }
    
    bool setupBindings() {
        int numBindings = m_engine->getNbBindings();
        m_bindings.resize(numBindings);
        m_bindingSizes.resize(numBindings);
        m_bindingDims.resize(numBindings);
        m_bindingDataTypes.resize(numBindings);
        m_bindingNames.resize(numBindings);
        
        for (int i = 0; i < numBindings; ++i) {
            const char* name = m_engine->getBindingName(i);
            m_bindingNames[i] = name;
            m_bindingDims[i] = m_engine->getBindingDimensions(i);
            m_bindingDataTypes[i] = m_engine->getBindingDataType(i);
            
            size_t elementSize = 0;
            switch (m_bindingDataTypes[i]) {
                case nvinfer1::DataType::kFLOAT:
                    elementSize = sizeof(float);
                    break;
                case nvinfer1::DataType::kHALF:
                    elementSize = sizeof(__half);
                    break;
                case nvinfer1::DataType::kINT8:
                    elementSize = sizeof(int8_t);
                    break;
                case nvinfer1::DataType::kINT32:
                    elementSize = sizeof(int32_t);
                    break;
                default:
                    std::cerr << "Unsupported data type for binding " << i << std::endl;
                    return false;
            }
            
            size_t volume = 1;
            for (int j = 0; j < m_bindingDims[i].nbDims; ++j) {
                volume *= m_bindingDims[i].d[j];
            }
            
            m_bindingSizes[i] = volume * elementSize;
            
            cudaError_t err = cudaMalloc(&m_bindings[i], m_bindingSizes[i]);
            if (err != cudaSuccess) {
                std::cerr << "Failed to allocate GPU memory for binding " << i 
                          << ": " << cudaGetErrorString(err) << std::endl;
                return false;
            }
            
            if (m_engine->bindingIsInput(i)) {
                m_inputIndex = i;
                std::cout << "Input binding: " << name << " [" << i << "]" << std::endl;
            } else {
                m_outputIndices.push_back(i);
                std::cout << "Output binding: " << name << " [" << i << "]" << std::endl;
            }
            
            std::cout << "Binding " << i << " (" << name << "): ";
            for (int j = 0; j < m_bindingDims[i].nbDims; ++j) {
                std::cout << m_bindingDims[i].d[j];
                if (j < m_bindingDims[i].nbDims - 1) std::cout << "x";
            }
            std::cout << " (" << m_bindingSizes[i] << " bytes)" << std::endl;
        }
        
        return true;
    }
    
    bool infer(cudaStream_t stream = 0) {
        return m_context->enqueueV2(m_bindings.data(), stream, nullptr);
    }
    
    void* getInputBuffer() const {
        return (m_inputIndex >= 0) ? m_bindings[m_inputIndex] : nullptr;
    }
    
    void* getOutputBuffer(int outputIndex) const {
        if (outputIndex < 0 || outputIndex >= static_cast<int>(m_outputIndices.size())) {
            return nullptr;
        }
        return m_bindings[m_outputIndices[outputIndex]];
    }
    
    nvinfer1::Dims getInputDims() const {
        return (m_inputIndex >= 0) ? m_bindingDims[m_inputIndex] : nvinfer1::Dims{};
    }
    
    nvinfer1::Dims getOutputDims(int outputIndex) const {
        if (outputIndex < 0 || outputIndex >= static_cast<int>(m_outputIndices.size())) {
            return nvinfer1::Dims{};
        }
        return m_bindingDims[m_outputIndices[outputIndex]];
    }
    
    int getNumOutputs() const {
        return static_cast<int>(m_outputIndices.size());
    }
    
    size_t getInputSize() const {
        return (m_inputIndex >= 0) ? m_bindingSizes[m_inputIndex] : 0;
    }
    
    size_t getOutputSize(int outputIndex) const {
        if (outputIndex < 0 || outputIndex >= static_cast<int>(m_outputIndices.size())) {
            return 0;
        }
        return m_bindingSizes[m_outputIndices[outputIndex]];
    }
    
    void destroy() {
        for (void* binding : m_bindings) {
            if (binding) {
                cudaFree(binding);
            }
        }
        m_bindings.clear();
        
        if (m_context) {
            m_context->destroy();
            m_context = nullptr;
        }
        
        if (m_engine) {
            m_engine->destroy();
            m_engine = nullptr;
        }
        
        if (m_runtime) {
            m_runtime->destroy();
            m_runtime = nullptr;
        }
    }
};

class LaneDetectionApplication : public DriveWorksSample
{
private:
    // ================================================
    // DRIVEWORKS CONTEXT AND SAL
    // ================================================
    dwContextHandle_t m_sdk              = DW_NULL_HANDLE;
    dwVisualizationContextHandle_t m_viz = DW_NULL_HANDLE;
    dwSALHandle_t m_sal                  = DW_NULL_HANDLE;

    // ================================================
    // ENHANCED TENSORRT CONFIGURATION
    // ================================================
    typedef std::pair<dwRectf, float32_t> BBoxConf;
    static constexpr float32_t COVERAGE_THRESHOLD       = 0.5f;
    static constexpr uint32_t NUM_OUTPUT_TENSORS        = 2U;
    static constexpr uint32_t DETECTION_OUTPUT_IDX      = 1U;
    static constexpr uint32_t SEGMENTATION_OUTPUT_IDX   = 0U;   

    // Network architecture constants
    static constexpr uint32_t PROTOTYPE_SIZE = 160;
    static constexpr uint32_t NUM_MASK_PROTOTYPES = 32;
    static constexpr uint32_t DETECTION_FEATURES = 37;
    static constexpr uint32_t NUM_PREDICTIONS = 8400;
    static constexpr float32_t CONFIDENCE_THRESHOLD = 0.45f;
    static constexpr float32_t SCORE_THRESHOLD = 0.25f;
    static constexpr float32_t SIGMOID_THRESHOLD = 0.5f;
    
    // Input preprocessing constants
    static constexpr uint32_t INPUT_WIDTH = 640;
    static constexpr uint32_t INPUT_HEIGHT = 640;
    static constexpr float32_t INPUT_SCALE = 1.0f / 255.0f;
    static constexpr float32_t MEAN_R = 0.485f;
    static constexpr float32_t MEAN_G = 0.456f;
    static constexpr float32_t MEAN_B = 0.406f;
    static constexpr float32_t STD_R = 0.229f;
    static constexpr float32_t STD_G = 0.224f;
    static constexpr float32_t STD_B = 0.225f;

    // Enhanced detection structure
    struct LaneDetection {
        dwRectf bbox;
        float32_t confidence;
        float32_t maskCoeffs[NUM_MASK_PROTOTYPES];
        
        LaneDetection() : confidence(0.0f) {
            bbox = {0.0f, 0.0f, 0.0f, 0.0f};
            std::fill(maskCoeffs, maskCoeffs + NUM_MASK_PROTOTYPES, 0.0f);
        }
    }; 

    typedef struct YoloScoreRect {
        dwRectf rectf;
        float32_t score;
        uint16_t classIndex;
    } YoloScoreRect;

    // ================================================
    // CAMERA AND SENSOR INFRASTRUCTURE
    // ================================================
    dwSensorHandle_t m_cameraSensor = DW_NULL_HANDLE;
    dwRigHandle_t m_rigConfig = DW_NULL_HANDLE;
    dwCameraFrameHandle_t m_currentFrame = DW_NULL_HANDLE;
    uint32_t m_selectedCameraIndex = 0;
    bool m_useProcessed = true;
    bool m_useVirtualVideo = false;

    // ================================================
    // TENSORRT INFERENCE ENGINE
    // ================================================
    TensorRTEngine m_tensorrtEngine;
    
    // ================================================
    // ENHANCED GPU PROCESSING PIPELINE
    // ================================================
    cublasHandle_t m_cublasHandle = nullptr;
    
    // Input preprocessing buffers
    CudaBuffer<float> m_inputBuffer;
    
    // GPU memory management for enhanced processing
    CudaBuffer<float> m_prototypeBuffer;
    CudaBuffer<float> m_maskCoeffsBuffer;
    CudaBuffer<float> m_linearMasksBuffer;
    CudaBuffer<uint8_t> m_binaryMasksBuffer;
    CudaBuffer<uint8_t> m_combinedMaskBuffer;
    CudaBuffer<int32_t> m_leftBoundsBuffer;
    CudaBuffer<int32_t> m_rightBoundsBuffer;
    CudaBuffer<uint8_t> m_leftMaskBuffer;
    CudaBuffer<uint8_t> m_rightMaskBuffer;
    CudaBuffer<uint8_t> m_areaMaskBuffer;
    
    uint32_t m_detectionRegionWidth = 0;
    uint32_t m_detectionRegionHeight = 0;
    
    dwImageTransformationHandle_t m_imageTransformationEngine = DW_NULL_HANDLE;
    dwImageStreamerHandle_t m_imageStreamer2GL = DW_NULL_HANDLE;

    dwImageHandle_t m_laneAreaImg = DW_NULL_HANDLE;
    dwImageCUDA* m_laneAreaCUDA = nullptr;
    std::unique_ptr<SimpleImageStreamerGL<>> m_streamerCUDA2GL_Area;

    std::vector<LaneDetection> m_validDetections;
    
    // Processing pipeline constants
    const uint32_t m_maxDetections = 1000U;
    const float32_t m_nonMaxSuppressionOverlapThreshold = 0.5f;
    
    std::vector<dwBox2D> m_detectedBoxList;
    std::vector<dwRectf> m_detectedBoxListFloat;
    std::vector<std::string> m_label;

    uint32_t m_cellSize = 1U;
    dwRect m_detectionRegion;

    const std::string YOLO_CLASS_NAMES[1] = {"lane"};

    // ================================================
    // RENDERER
    // ================================================
    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
    dwImageHandle_t m_imageRGBA;
    std::unique_ptr<SimpleImageStreamerGL<>> m_streamerCUDA2GL;
    cudaStream_t m_cudaStream = 0;

    // ================================================
    // CAMERA IMAGE PROPERTIES
    // ================================================
    dwImageGL* m_imgGl;
    dwImageProperties m_rcbProperties;
    uint32_t m_imageWidth;
    uint32_t m_imageHeight;
    bool m_isRaw = false;

public:
    LaneDetectionApplication(const ProgramArguments& args)
        : DriveWorksSample(args) {}

    bool initializeEnhancedProcessingPipeline();
    void allocateProcessingBuffers();
    bool onInitialize() override;
    void onProcess() override;
    void renderBoundaryStrip(const int32_t* bounds,
                                                   uint32_t        height,
                                                   float           roiX,
                                                   float           roiY,
                                                   dwRenderEngineColorRGBA colour);
    void onRender() override;
    void onRelease() override;
    void onReset() override;
    void onResizeWindow(int width, int height) override;

private:
    // TensorRT-specific methods
    bool initTensorRT();
    bool preprocessImage(const dwImageCUDA* inputImage);
    bool doInference();
    void processInferenceResults();
    
    // Processing pipeline methods
    void processSegmentationOutputOptimized(const float* detectionOutput, 
                                           const float* segmentationOutput);
    void extractValidDetections(const float32_t* detection, 
                               std::vector<LaneDetection>& validDetections);
    void processValidDetectionsOptimized(const std::vector<LaneDetection>& validDetections,
                                       const float32_t* prototypes);
    void processValidDetectionsOptimizedImpl(const std::vector<LaneDetection>& validDetections,
                                           const float32_t* prototypes, uint32_t N);
    void convertToRenderingFormat(const std::vector<LaneDetection>& detections);
    
    // Camera and sensor methods
    void initDWAndSAL();
    void initRender();
    bool initSensors();
    void prepareInputFrame();
    void getNextFrame(dwImageCUDA** nextFrameCUDA, dwImageGL** nextFrameGL);
    
    // Cleanup methods
    void releaseRender();
    void releaseDWAndSAL();
    void releaseTensorRT();
    void releaseImage();
    
    // Utility methods
    static bool sort_score(YoloScoreRect box1, YoloScoreRect box2);
    float32_t calculateIouOfBoxes(dwRectf box1, dwRectf box2);
    std::vector<YoloScoreRect> doNmsForYoloOutputBoxes(std::vector<YoloScoreRect>& boxes, float32_t threshold);
    float32_t overlap(const dwRectf& boxA, const dwRectf& boxB);
    std::string getPlatformPrefix();
    
protected:
    dwContextHandle_t getSDKContext() { return m_sdk; }
};