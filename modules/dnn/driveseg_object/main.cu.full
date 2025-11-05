// main.cu
// Four-camera semantic segmentation with alpha-blended masks.
// Single file: includes main(). No external headers beyond DW sample framework.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <GLFW/glfw3.h> // for GLFW_KEY_*

// ---------------- DriveWorks ----------------
#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/core/platform/GPUProperties.h>

#include <dw/dnn/DNN.h>
#include <dw/dnn/dataconditioner/DataConditioner.h>
#include <dw/dnn/tensor/Tensor.h>
#include <dw/interop/streamer/TensorStreamer.h>

#include <dw/image/Image.h>
#include <dw/interop/streamer/ImageStreamer.h>

#include <dw/rig/Rig.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/camera/Camera.h>

#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Renderer.h>
#include <dwvisualization/core/Visualization.h>
#include <dwvisualization/image/Image.h>
#include <dwvisualization/interop/ImageStreamer.h>

// ------------- DW Sample Framework ----------
#include <framework/Checks.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/Log.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/SamplesDataPath.hpp>
#include <framework/ScreenshotHelper.hpp>
#include <framework/WindowGLFW.hpp>

using namespace dw_samples::common;

#define MAX_PORTS_COUNT 4
#define MAX_CAMS_PER_PORT 4
#define MAX_CAMS (MAX_PORTS_COUNT * MAX_CAMS_PER_PORT)
static_assert(MAX_CAMS >= 4, "MAX_CAMS must be at least 4");

constexpr float kOverlayAlpha = 0.35f;

// ===============================================================
// CUDA: blend a uint8 mask onto an RGBA8 image (CUDA).
// ===============================================================
__global__ void blendMaskKernel(
    uint8_t* rgba, int imgW, int imgH, int imgStrideBytes,
    const uint8_t* mask, int maskW, int maskH, int maskStrideBytes,
    float4 overlayColor, float alpha,
    bool linearScaleOnly,
    float sx, float sy, float ox, float oy)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= imgW || y >= imgH) return;

    float mx_f, my_f;
    if (linearScaleOnly) {
        mx_f = (float)x * ((float)maskW / (float)imgW);
        my_f = (float)y * ((float)maskH / (float)imgH);
    } else {
        mx_f = sx * x + ox;
        my_f = sy * y + oy;
    }
    const int mx = (int)mx_f;
    const int my = (int)my_f;
    if (mx < 0 || my < 0 || mx >= maskW || my >= maskH) return;

    const uint8_t* maskRow = mask + my * maskStrideBytes;
    const uint8_t m = maskRow[mx];
    if (m == 0) return;

    uint8_t* p = rgba + y * imgStrideBytes + x * 4;
    float r = p[0] / 255.f;
    float g = p[1] / 255.f;
    float b = p[2] / 255.f;

    const float invA = 1.0f - alpha;
    r = invA * r + alpha * overlayColor.x;
    g = invA * g + alpha * overlayColor.y;
    b = invA * b + alpha * overlayColor.z;

    p[0] = (uint8_t)(255.f * r + 0.5f);
    p[1] = (uint8_t)(255.f * g + 0.5f);
    p[2] = (uint8_t)(255.f * b + 0.5f);
    // preserve p[3]
}
// ===============================================================
// CUDA: GPU-accelerated argmax for segmentation (NCHW layout)
// Input: [C, H, W] FP32 logits (reverse-order: [W, H, C, N])
// Output: [H, W] uint8 class indices
// ===============================================================
__global__ void argmaxNCHW_kernel(
    uint8_t* classIndices,      // Output: [H*W] class map
    const float* logits,        // Input: [C*H*W] logits (NCHW)
    int W, int H, int C)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= W || y >= H) return;
    
    const int spatialIdx = y * W + x;
    const int spatialSize = W * H;
    
    // Find max across channel dimension
    float maxVal = logits[spatialIdx];  // Class 0
    uint8_t maxClass = 0;
    
    #pragma unroll
    for (int c = 1; c < C; ++c) {
        float val = logits[c * spatialSize + spatialIdx];
        if (val > maxVal) {
            maxVal = val;
            maxClass = c;
        }
    }
    
    classIndices[spatialIdx] = maxClass;
}

// ===============================================================
// CUDA: Colorize class indices and blend in single pass
// ===============================================================
__global__ void colorizeAndBlend_kernel(
    uint8_t* rgba,              // Input/Output: RGBA image
    const uint8_t* classIndices, // Input: class map
    int imgW, int imgH, int imgStrideBytes,
    int maskW, int maskH,
    const uint8_t* colorLUT,    // [numClasses * 4] RGBA colors
    float alpha)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= imgW || y >= imgH) return;
    
    // Bilinear scale from mask to image coordinates
    const float mx_f = (float)x * ((float)maskW / (float)imgW);
    const float my_f = (float)y * ((float)maskH / (float)imgH);
    const int mx = (int)mx_f;
    const int my = (int)my_f;
    
    if (mx >= maskW || my >= maskH) return;
    
    const uint8_t classIdx = classIndices[my * maskW + mx];
    
    // Skip background (class 0)
    if (classIdx == 0) return;
    
    // Lookup color for this class
    const uint8_t* color = &colorLUT[classIdx * 4];
    const uint8_t colorA = color[3];
    
    if (colorA == 0) return;  // Fully transparent
    
    // Blend color into image
    uint8_t* pixel = rgba + y * imgStrideBytes + x * 4;
    
    const float finalAlpha = (colorA / 255.0f) * alpha;
    const float invAlpha = 1.0f - finalAlpha;
    
    pixel[0] = (uint8_t)(invAlpha * pixel[0] + finalAlpha * color[0]);
    pixel[1] = (uint8_t)(invAlpha * pixel[1] + finalAlpha * color[1]);
    pixel[2] = (uint8_t)(invAlpha * pixel[2] + finalAlpha * color[2]);
    // pixel[3] unchanged
}

// ===============================================================
// App
// ===============================================================
class DriveSeg4CamApp : public DriveWorksSample
{
public:
    explicit DriveSeg4CamApp(const ProgramArguments& args)
        : DriveWorksSample(args) {}

    bool onInitialize() override;
    void onRelease() override;
    void onProcess() override {}
    void onRender() override;
    void onKeyDown(int key, int scancode, int mods) override;

private:
    // Core
    dwContextHandle_t m_ctx{DW_NULL_HANDLE};
    dwSALHandle_t m_sal{DW_NULL_HANDLE};
    dwRigHandle_t m_rig{DW_NULL_HANDLE};

    // Viz
    dwVisualizationContextHandle_t m_viz{DW_NULL_HANDLE};
    dwRenderEngineHandle_t m_re{DW_NULL_HANDLE};
    dwRenderEngineParams m_reParams{};
    uint32_t m_tiles[MAX_CAMS]{};
    std::unique_ptr<ScreenshotHelper> m_screenshot;

    // Cameras
    uint32_t m_numCameras{0};
    dwSensorHandle_t m_cam[MAX_CAMS]{DW_NULL_HANDLE};
    dwRect m_roi[MAX_CAMS]{};
    dwImageHandle_t m_imgRGBA[MAX_CAMS]{DW_NULL_HANDLE};
    dwImageStreamerHandle_t m_streamerToGL[MAX_CAMS]{DW_NULL_HANDLE};

    // Segmentation DNN
    dwDNNHandle_t m_dnn{DW_NULL_HANDLE};
    uint32_t m_outIdx{0};
    bool m_outputIsFP16{false};
    bool m_outputIsNCHW{true};
    dwDNNTensorProperties m_inProps{};
    dwDNNTensorProperties m_outProps{};
    
    // Detection DNN
    dwDNNHandle_t m_detDnn{DW_NULL_HANDLE};
    dwDNNTensorProperties m_detInProps{};
    dwDNNTensorProperties m_detOutProps{};

    // YOLO class definitions
    const std::string YOLO_CLASS_NAMES[80] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
        "toothbrush"
    };

    std::set<std::string> m_automotiveClasses = {
        "person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "stop sign"
    };

    typedef struct YoloScoreRect {
        dwRectf rectf;
        float32_t score;
        uint16_t classIndex;
    } YoloScoreRect;

    struct CamDNN {
        // === SEGMENTATION ===
        cudaStream_t segStream{nullptr};
        cudaEvent_t  segInferDone{nullptr};
        dwDataConditionerHandle_t segConditioner{DW_NULL_HANDLE};
        dwDNNTensorHandle_t segInTensor{DW_NULL_HANDLE};
        dwDNNTensorHandle_t segOutTensorDev{DW_NULL_HANDLE};
        dwDNNTensorStreamerHandle_t segOutStreamer{DW_NULL_HANDLE};
        dwDNNTensorHandle_t segOutTensorHost{DW_NULL_HANDLE};
        bool segRunning{false};
        
        // === OBJECT DETECTION ===
        cudaStream_t detStream{nullptr};
        cudaEvent_t  detInferDone{nullptr};
        dwDataConditionerHandle_t detConditioner{DW_NULL_HANDLE};
        dwDNNTensorHandle_t detInTensor{DW_NULL_HANDLE};
        dwDNNTensorHandle_t detOutTensorDev{DW_NULL_HANDLE};
        dwDNNTensorStreamerHandle_t detOutStreamer{DW_NULL_HANDLE};
        dwDNNTensorHandle_t detOutTensorHost{DW_NULL_HANDLE};
        bool detRunning{false};
        
        // Detection results
        std::vector<dwRectf> detBoxes;
        std::vector<std::string> detLabels;
        std::mutex detMutex;
        
        // === SHARED ===
        uint64_t frameId{0};
        float avgSegMs{0.f};
        float avgDetMs{0.f};
        uint32_t segCount{0};
        uint32_t detCount{0};
        
        std::chrono::high_resolution_clock::time_point segStartTime;
        std::chrono::high_resolution_clock::time_point detStartTime;
    } m_dnnCtx[MAX_CAMS];

    // Stats
    std::atomic<uint64_t> m_frameCounter{0};
    std::chrono::high_resolution_clock::time_point m_lastStats{};
    uint32_t m_recentInferences{0};

private:
    // Init
    void initDW();
    void initCameras();
    void initRenderer();
    void initDNN();
    void initPerCameraDNN(uint32_t i);

    // Per-frame
    void grabFrames(dwCameraFrameHandle_t frames[]);
    void startAllInferences(dwCameraFrameHandle_t frames[]);
    void maybeCollect(uint32_t i);
    void renderCamera(uint32_t i, dwCameraFrameHandle_t frame);

    // DNN steps
    void prepareInput(uint32_t i, dwCameraFrameHandle_t frame);
    void runInference(uint32_t i);
    void collectAndOverlay(uint32_t i);

    // Misc
    std::string platformPrefix();
    void printStats();
    
    // detection DNN helpers
    void initDetectionDNN();
    void initPerCameraDetection(uint32_t i);
    void runDetectionInference(uint32_t i);
    void collectDetectionResults(uint32_t i);
    void renderDetectionBoxes(uint32_t i);

    static bool sort_score(YoloScoreRect box1, YoloScoreRect box2);
    float32_t calculateIouOfBoxes(dwRectf box1, dwRectf box2);
    std::vector<YoloScoreRect> doNmsForYoloOutputBoxes(std::vector<YoloScoreRect>& boxes, float32_t threshold);
};

// ---------------- Init ----------------
bool DriveSeg4CamApp::onInitialize()
{
    initDW();
    CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_ctx));

    initCameras();
    CHECK_DW_ERROR(dwSAL_start(m_sal));
    initRenderer();

    initDNN();
    initDetectionDNN();

     for (uint32_t i = 0; i < m_numCameras; ++i) {
        initPerCameraDNN(i);        // Segmentation
        initPerCameraDetection(i);  // Detection
    }
    m_screenshot.reset(new ScreenshotHelper(m_ctx, m_sal, getWindowWidth(), getWindowHeight(), "DriveSeg4Cam"));

    for (uint32_t i = 0; i < m_numCameras; ++i) {
        CHECK_DW_ERROR(dwSensor_start(m_cam[i]));
    }

    m_lastStats = std::chrono::high_resolution_clock::now();
    std::cout << "[DriveSeg4Cam] Initialized for " << m_numCameras << " camera(s)\n";
    return true;
}

void DriveSeg4CamApp::onRelease()
{
    std::cout << "[DriveSeg4Cam] Releasing...\n";

    for (uint32_t i = 0; i < m_numCameras; ++i) {
        if (m_cam[i]) dwSensor_stop(m_cam[i]);
    }

    for (uint32_t i = 0; i < m_numCameras; ++i) {
    auto &c = m_dnnCtx[i];
    
        // Segmentation cleanup
        if (c.segInferDone) cudaEventDestroy(c.segInferDone);
        if (c.segStream) cudaStreamDestroy(c.segStream);
        if (c.segInTensor) dwDNNTensor_destroy(c.segInTensor);
        if (c.segOutTensorDev) dwDNNTensor_destroy(c.segOutTensorDev);
        if (c.segOutStreamer) dwDNNTensorStreamer_release(c.segOutStreamer);
        if (c.segConditioner) dwDataConditioner_release(c.segConditioner);
        
        // Detection cleanup
        if (c.detInferDone) cudaEventDestroy(c.detInferDone);
        if (c.detStream) cudaStreamDestroy(c.detStream);
        if (c.detInTensor) dwDNNTensor_destroy(c.detInTensor);
        if (c.detOutTensorDev) dwDNNTensor_destroy(c.detOutTensorDev);
        if (c.detOutStreamer) dwDNNTensorStreamer_release(c.detOutStreamer);
        if (c.detConditioner) dwDataConditioner_release(c.detConditioner);
    }

    if (m_dnn) dwDNN_release(m_dnn);
    if (m_detDnn) dwDNN_release(m_detDnn);

        for (uint32_t i = 0; i < m_numCameras; ++i) {
            if (m_streamerToGL[i]) dwImageStreamerGL_release(m_streamerToGL[i]);
            if (m_imgRGBA[i]) dwImage_destroy(m_imgRGBA[i]);
            if (m_cam[i]) dwSAL_releaseSensor(m_cam[i]);
        }

        if (m_re) dwRenderEngine_release(m_re);
        if (m_viz) dwVisualizationRelease(m_viz);
        if (m_rig) dwRig_release(m_rig);
        if (m_sal) dwSAL_release(m_sal);
        if (m_ctx) dwRelease(m_ctx);

        std::cout << "[DriveSeg4Cam] Released.\n";
}

void DriveSeg4CamApp::onRender()
{
    dwCameraFrameHandle_t frames[MAX_CAMS]{};
    grabFrames(frames);
    startAllInferences(frames);

    for (uint32_t i = 0; i < m_numCameras; ++i) {
        maybeCollect(i);
    }

    for (uint32_t i = 0; i < m_numCameras; ++i) {
        renderCamera(i, frames[i]);
    }

    for (uint32_t i = 0; i < m_numCameras; ++i) {
        if (frames[i]) dwSensorCamera_returnFrame(&frames[i]);
    }

    m_frameCounter++;
    auto now = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - m_lastStats).count() >= 2) {
        printStats();
        m_lastStats = now;
        m_recentInferences = 0;
    }
}

void DriveSeg4CamApp::onKeyDown(int key, int, int)
{
    if (key == GLFW_KEY_S) {
        if (m_screenshot) m_screenshot->triggerScreenshot();
    } else if (key == GLFW_KEY_P) {
        printStats();
    }
}

void DriveSeg4CamApp::initDW()
{
    CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
    CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));
    dwContextParameters p{};
#ifdef VIBRANTE
    p.eglDisplay = getEGLDisplay();
#endif
    CHECK_DW_ERROR(dwInitialize(&m_ctx, DW_VERSION, &p));
}

void DriveSeg4CamApp::initCameras()
{
    CHECK_DW_ERROR(dwRig_initializeFromFile(&m_rig, m_ctx, getArgument("rig").c_str()));

    uint32_t countType = 0;
    CHECK_DW_ERROR(dwRig_getSensorCountOfType(&countType, DW_SENSOR_CAMERA, m_rig));
    if (countType == 0) throw std::runtime_error("No cameras in rig");

    m_numCameras = std::min<uint32_t>(countType, 4);

    for (uint32_t i = 0; i < m_numCameras; ++i) {
        uint32_t camIdx = 0;
        CHECK_DW_ERROR(dwRig_findSensorByTypeIndex(&camIdx, DW_SENSOR_CAMERA, i, m_rig));

        const char* proto = nullptr;
        const char* params = nullptr;
        CHECK_DW_ERROR(dwRig_getSensorProtocol(&proto, camIdx, m_rig));
        CHECK_DW_ERROR(dwRig_getSensorParameterUpdatedPath(&params, camIdx, m_rig));

        dwSensorParams sp{};
        sp.protocol = proto;
        sp.parameters = params;

        std::cout << "[Cam" << i << "] " << proto << " params: " << params << "\n";
        CHECK_DW_ERROR(dwSAL_createSensor(&m_cam[i], sp, m_sal));
    }

    dwImageProperties rgbaProps{};
    CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&rgbaProps, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, m_cam[0]));

    for (uint32_t i = 0; i < m_numCameras; ++i) {
        m_roi[i] = {0, 0, (int32_t)rgbaProps.width, (int32_t)rgbaProps.height};
        CHECK_DW_ERROR(dwImage_create(&m_imgRGBA[i], rgbaProps, m_ctx));
        CHECK_DW_ERROR(dwImageStreamerGL_initialize(&m_streamerToGL[i], &rgbaProps, DW_IMAGE_GL, m_ctx));
    }
}

void DriveSeg4CamApp::initRenderer()
{
    CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_ctx));

    CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&m_reParams, getWindowWidth(), getWindowHeight()));
    m_reParams.defaultTile.lineWidth = 2.f;
    m_reParams.maxBufferCount = 1;
    m_reParams.bounds = {0, 0, (float)getWindowWidth(), (float)getWindowHeight()};
    CHECK_DW_ERROR(dwRenderEngine_initialize(&m_re, &m_reParams, m_viz));

    const uint32_t tilesPerRow = (m_numCameras <= 1) ? 1 : 2;
    dwRenderEngineTileState states[MAX_CAMS];
    for (uint32_t i = 0; i < m_numCameras; ++i) {
        dwRenderEngine_initTileState(&states[i]);
        states[i].font = DW_RENDER_ENGINE_FONT_VERDANA_16;
        states[i].modelViewMatrix = DW_IDENTITY_MATRIX4F;
    }
    CHECK_DW_ERROR(dwRenderEngine_addTilesByCount(m_tiles, m_numCameras, tilesPerRow, states, m_re));
}

void DriveSeg4CamApp::initDNN()
{
#ifdef VIBRANTE
    bool useDLA = (getArgument("cudla") == "1");
    uint32_t dlaEngine = (uint32_t)std::atoi(getArgument("dla-engine").c_str());
#endif

    std::string trtPath = getArgument("tensorRT_model");
    if (trtPath.empty()) {
        trtPath = dw_samples::SamplesDataPath::get() + std::string("/samples/detector/");
        trtPath += platformPrefix();
        trtPath += "/resnet34_fcn_gpu_fp16.bin"; // adjust name if needed
    }
    std::cout << "[DNN] Loading: " << trtPath << "\n";

#ifdef VIBRANTE
    CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFileWithEngineId(
        &m_dnn, trtPath.c_str(), nullptr,
        useDLA ? DW_PROCESSOR_TYPE_CUDLA : DW_PROCESSOR_TYPE_GPU,
        useDLA ? (int32_t)dlaEngine : 0, m_ctx));
#else
    CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFile(
        &m_dnn, trtPath.c_str(), nullptr, DW_PROCESSOR_TYPE_GPU, m_ctx));
#endif

    CHECK_DW_ERROR(dwDNN_getInputTensorProperties(&m_inProps, 0, m_dnn));
    CHECK_DW_ERROR(dwDNN_getOutputTensorProperties(&m_outProps, 0, m_dnn));

    // ---- FP16 detection using dwTrivialDataType from header
#if defined(DW_TRIVIAL_DATA_TYPE_FLOAT16)
    #define DW_FP16_ENUM DW_TRIVIAL_DATA_TYPE_FLOAT16
#elif defined(DW_TRIVIAL_DATA_TYPE_F16)
    #define DW_FP16_ENUM DW_TRIVIAL_DATA_TYPE_F16
#elif defined(DW_TYPE_FLOAT16)
    #define DW_FP16_ENUM DW_TYPE_FLOAT16
#elif defined(DW_TYPE_F16)
    #define DW_FP16_ENUM DW_TYPE_F16
#else
    #define DW_FP16_ENUM ((dwTrivialDataType)-1)
#endif
    m_outputIsFP16 = (m_outProps.dataType == DW_FP16_ENUM);

    // ---- Layout via tensorLayout
    m_outputIsNCHW = (m_outProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NCHW);
    m_outIdx = 0; // using first output

    // Log the tensor layout obeying DNNTensor header comment (dimension order)
    std::cout << "Tensor Properties Validation:\n";
    std::cout << "Input tensor:\n  Dimensions: " << m_inProps.numDimensions << "D [";
    for (uint32_t i=0;i<m_inProps.numDimensions;++i)
        std::cout << m_inProps.dimensionSize[i] << (i+1<m_inProps.numDimensions?", ":"");
    std::cout << "]\n  Layout: " 
              << (m_inProps.tensorLayout==DW_DNN_TENSOR_LAYOUT_NCHW?"NCHW":
                  m_inProps.tensorLayout==DW_DNN_TENSOR_LAYOUT_NHWC?"NHWC":"NCHWx")
              << "\n  Data type: " << (m_outputIsFP16?"FP16":"FP32") << "\n";

    std::cout << "Output tensor:\n  Dimensions: " << m_outProps.numDimensions << "D [";
    for (uint32_t i=0;i<m_outProps.numDimensions;++i)
        std::cout << m_outProps.dimensionSize[i] << (i+1<m_outProps.numDimensions?", ":"");
    std::cout << "]\n  Layout: " 
              << (m_outProps.tensorLayout==DW_DNN_TENSOR_LAYOUT_NCHW?"NCHW":
                  m_outProps.tensorLayout==DW_DNN_TENSOR_LAYOUT_NHWC?"NHWC":"NCHWx")
              << "\n  Data type: " << (m_outputIsFP16?"FP16":"FP32") << "\n";

    int N=1,C=1,H=1,W=1;
    if (m_outProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NCHW) {
        W = (int)m_outProps.dimensionSize[0];
        H = (int)m_outProps.dimensionSize[1];
        C = (int)m_outProps.dimensionSize[2];
        N = (int)m_outProps.dimensionSize[3];
    } else if (m_outProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NHWC) {
        C = (int)m_outProps.dimensionSize[0];
        W = (int)m_outProps.dimensionSize[1];
        H = (int)m_outProps.dimensionSize[2];
        N = (int)m_outProps.dimensionSize[3];
    }
    std::cout << "  Interpreted: [N="<<N<<", C="<<C<<", H="<<H<<", W="<<W<<"]\n";
    std::cout << "Model resolution: " << W << "x" << H << "\n";
    std::cout << "✓ Tensor properties validated\n";

    dwDNNMetaData meta{};
    CHECK_DW_ERROR(dwDNN_getMetaData(&meta, m_dnn));
    std::cout << "Data conditioner parameters:\n";
    std::cout << "  Mean: [" << meta.dataConditionerParams.meanValue[0] << ", "
                              << meta.dataConditionerParams.meanValue[1] << ", "
                              << meta.dataConditionerParams.meanValue[2] << "]\n";
    // NOTE: Do not print stdDev — not present in your headers.
    std::cout << "✓ DNN initialized\n";
}

void DriveSeg4CamApp::initPerCameraDNN(uint32_t i)
{
    auto &c = m_dnnCtx[i];

    cudaStreamCreate(&c.segStream);        
    cudaEventCreate(&c.segInferDone);       

    dwDNNMetaData meta{};
    CHECK_DW_ERROR(dwDNN_getMetaData(&meta, m_dnn));
    CHECK_DW_ERROR(dwDataConditioner_initializeFromTensorProperties(
        &c.segConditioner, &m_inProps, 1, &meta.dataConditionerParams, c.segStream, m_ctx));

    CHECK_DW_ERROR(dwDNNTensor_create(&c.segInTensor, &m_inProps, m_ctx));
    CHECK_DW_ERROR(dwDNNTensor_create(&c.segOutTensorDev, &m_outProps, m_ctx));

    dwDNNTensorProperties hostProps = m_outProps;
    hostProps.tensorType = DW_DNN_TENSOR_TYPE_CPU;
    CHECK_DW_ERROR(dwDNNTensorStreamer_initialize(&c.segOutStreamer, &m_outProps, hostProps.tensorType, m_ctx));
}

void DriveSeg4CamApp::initDetectionDNN()
{
#ifdef VIBRANTE
    bool useDLA = (getArgument("cudla") == "1");
    uint32_t dlaEngine = (uint32_t)std::atoi(getArgument("dla-engine").c_str());
#endif

    std::string detModelPath = getArgument("detection_model");
    if (detModelPath.empty()) {
        detModelPath = dw_samples::SamplesDataPath::get() + std::string("/samples/detector/");
        detModelPath += platformPrefix();
        detModelPath += "/yolov3_640x640.bin";  // YOLO model
    }
    
    std::cout << "[Detection DNN] Loading: " << detModelPath << "\n";

#ifdef VIBRANTE
    CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFileWithEngineId(
        &m_detDnn, detModelPath.c_str(), nullptr,
        useDLA ? DW_PROCESSOR_TYPE_CUDLA : DW_PROCESSOR_TYPE_GPU,
        useDLA ? (int32_t)dlaEngine : 0, m_ctx));
#else
    CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFile(
        &m_detDnn, detModelPath.c_str(), nullptr, DW_PROCESSOR_TYPE_GPU, m_ctx));
#endif

    CHECK_DW_ERROR(dwDNN_getInputTensorProperties(&m_detInProps, 0, m_detDnn));
    CHECK_DW_ERROR(dwDNN_getOutputTensorProperties(&m_detOutProps, 0, m_detDnn));
    
    std::cout << "✓ Detection DNN initialized\n";
}

void DriveSeg4CamApp::initPerCameraDetection(uint32_t i)
{
    auto &c = m_dnnCtx[i];

    cudaStreamCreate(&c.detStream);
    cudaEventCreate(&c.detInferDone);

    dwDNNMetaData meta{};
    CHECK_DW_ERROR(dwDNN_getMetaData(&meta, m_detDnn));
    CHECK_DW_ERROR(dwDataConditioner_initializeFromTensorProperties(
        &c.detConditioner, &m_detInProps, 1, &meta.dataConditionerParams, c.detStream, m_ctx));

    CHECK_DW_ERROR(dwDNNTensor_create(&c.detInTensor, &m_detInProps, m_ctx));
    CHECK_DW_ERROR(dwDNNTensor_create(&c.detOutTensorDev, &m_detOutProps, m_ctx));

    dwDNNTensorProperties hostProps = m_detOutProps;
    hostProps.tensorType = DW_DNN_TENSOR_TYPE_CPU;
    CHECK_DW_ERROR(dwDNNTensorStreamer_initialize(&c.detOutStreamer, &m_detOutProps, hostProps.tensorType, m_ctx));
    
    c.detBoxes.reserve(100);
    c.detLabels.reserve(100);
}

// ---------------- Per-frame ----------------
void DriveSeg4CamApp::grabFrames(dwCameraFrameHandle_t frames[])
{
    for (uint32_t i = 0; i < m_numCameras; ++i) {
        dwStatus s = DW_NOT_READY;
        int spins = 0;
        while (s == DW_NOT_READY) {
            s = dwSensorCamera_readFrame(&frames[i], 33333, m_cam[i]); // ~30fps timeout
            if (++spins > 10) break;
        }
        if (s == DW_TIME_OUT) {
            frames[i] = nullptr;
            std::cout << "[Cam" << i << "] TIMEOUT\n";
        } else if (s != DW_SUCCESS) {
            frames[i] = nullptr;
            std::cout << "[Cam" << i << "] read error: " << dwGetStatusName(s) << "\n";
        }
    }
}

void DriveSeg4CamApp::runDetectionInference(uint32_t i)
{
    auto &c = m_dnnCtx[i];
    CHECK_DW_ERROR(dwDNN_setCUDAStream(c.detStream, m_detDnn));

    dwConstDNNTensorHandle_t in[1] = {c.detInTensor};
    CHECK_DW_ERROR(dwDNN_infer(&c.detOutTensorDev, 1, in, 1, m_detDnn));

    CHECK_DW_ERROR(dwDNNTensorStreamer_producerSend(c.detOutTensorDev, c.detOutStreamer));
    cudaEventRecord(c.detInferDone, c.detStream);
}

void DriveSeg4CamApp::startAllInferences(dwCameraFrameHandle_t frames[])
{
    for (uint32_t i = 0; i < m_numCameras; ++i) {
        auto &c = m_dnnCtx[i];
        if (frames[i] == nullptr) continue;

        // ✓ STEP 1: Start DETECTION first (changed order)
        if (!c.detRunning) {
            try {
                c.detStartTime = std::chrono::high_resolution_clock::now();
                
                // Prepare detection input FIRST
                dwImageHandle_t src = DW_NULL_HANDLE;
                CHECK_DW_ERROR(dwSensorCamera_getImage(&src, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, frames[i]));
                CHECK_DW_ERROR(dwImage_copyConvert(m_imgRGBA[i], src, m_ctx));
                
                CHECK_DW_ERROR(dwDataConditioner_prepareData(
                    c.detInTensor, &m_imgRGBA[i], 1, &m_roi[i], 
                    cudaAddressModeClamp, c.detConditioner));
                
                runDetectionInference(i);
                c.detRunning = true;
            } catch (const std::exception& e) {
                std::cout << "[Cam" << i << "] det inference failed: " << e.what() << "\n";
            }
        }
        
        // ✓ STEP 2: Start SEGMENTATION second (changed order)
        if (!c.segRunning) {
            try {
                c.segStartTime = std::chrono::high_resolution_clock::now();
                prepareInput(i, frames[i]);  // Segmentation prep
                runInference(i);             // Segmentation inference
                c.segRunning = true;
                c.frameId = m_frameCounter.load();
            } catch (const std::exception& e) {
                std::cout << "[Cam" << i << "] seg inference failed: " << e.what() << "\n";
            }
        }
    }
}


void DriveSeg4CamApp::collectDetectionResults(uint32_t i)
{
    auto &c = m_dnnCtx[i];
    CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReceive(&c.detOutTensorHost, 1000, c.detOutStreamer));

    void* hptr = nullptr;
    CHECK_DW_ERROR(dwDNNTensor_lock(&hptr, c.detOutTensorHost));
    float32_t* confData = reinterpret_cast<float32_t*>(hptr);

    std::lock_guard<std::mutex> lock(c.detMutex);
    c.detBoxes.clear();
    c.detLabels.clear();

    // STEP 1: Collect ALL detections with scores (before NMS)
    std::vector<YoloScoreRect> tmpRes;
    size_t offsetY, strideY, height;
    uint32_t indices[4] = {0, 0, 0, 0};
    CHECK_DW_ERROR(dwDNNTensor_getLayoutView(&offsetY, &strideY, &height, indices, 4U, 1U, c.detOutTensorHost));

    for (uint16_t row = 0; row < height; ++row) {
        const float32_t* detection = &confData[row * strideY];
        float conf = detection[4];
        
        if (conf < 0.45f) continue;  // Confidence threshold
        
        // Find best class
        uint16_t maxClass = 0;
        float maxScore = 0;
        for (uint16_t cls = 0; cls < 80; ++cls) {
            if (detection[5 + cls] > maxScore) {
                maxScore = detection[5 + cls];
                maxClass = cls;
            }
        }
        
        if (maxScore < 0.25f) continue;  // Class score threshold
        
        // Extract box in model space (center format)
        float32_t imageX = detection[0];
        float32_t imageY = detection[1];
        float32_t bboxW  = detection[2];
        float32_t bboxH  = detection[3];

        // Convert center format to corner format (still in model space)
        float32_t boxX1Tmp = imageX - 0.5f * bboxW;
        float32_t boxY1Tmp = imageY - 0.5f * bboxH;
        float32_t boxX2Tmp = imageX + 0.5f * bboxW;
        float32_t boxY2Tmp = imageY + 0.5f * bboxH;

        // Transform from model space to camera image space
        float32_t boxX1, boxY1, boxX2, boxY2;
        dwDataConditioner_outputPositionToInput(&boxX1, &boxY1, boxX1Tmp, boxY1Tmp, 
                                                &m_roi[i], c.detConditioner);
        dwDataConditioner_outputPositionToInput(&boxX2, &boxY2, boxX2Tmp, boxY2Tmp, 
                                                &m_roi[i], c.detConditioner);
        
        // Store detection with score for NMS
        dwRectf bboxFloat{boxX1, boxY1, boxX2 - boxX1, boxY2 - boxY1};
        tmpRes.push_back({bboxFloat, maxScore, maxClass});
    }

    CHECK_DW_ERROR(dwDNNTensor_unlock(c.detOutTensorHost));

    // STEP 2: Apply NMS to remove overlapping boxes
    std::vector<YoloScoreRect> tmpResAfterNMS = doNmsForYoloOutputBoxes(tmpRes, 0.45f);
    
    // STEP 3: Filter for automotive classes and add to final results
    for (uint32_t j = 0; j < tmpResAfterNMS.size(); j++)
    {
        YoloScoreRect box = tmpResAfterNMS[j];
        
        // Filter for automotive classes only
        const std::string& className = YOLO_CLASS_NAMES[box.classIndex];
        if (m_automotiveClasses.find(className) != m_automotiveClasses.end())
        {
            c.detBoxes.push_back(box.rectf);
            c.detLabels.push_back(className);
        }
    }

    CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReturn(&c.detOutTensorHost, c.detOutStreamer));
    CHECK_DW_ERROR(dwDNNTensorStreamer_producerReturn(nullptr, 1000, c.detOutStreamer));
}

void DriveSeg4CamApp::maybeCollect(uint32_t i)
{
    auto &c = m_dnnCtx[i];
    
    // Check segmentation
    if (c.segRunning && cudaEventQuery(c.segInferDone) == cudaSuccess) {
        collectAndOverlay(i);  // Existing
        
        auto endTime = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(endTime - c.segStartTime).count();
        c.avgSegMs = (c.avgSegMs * c.segCount + ms) / (c.segCount + 1);
        
        c.segRunning = false;
        c.segCount++;
        m_recentInferences++;
    }
    
    // Check detection (NEW)
    if (c.detRunning && cudaEventQuery(c.detInferDone) == cudaSuccess) {
        collectDetectionResults(i);  // NEW
        
        auto endTime = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(endTime - c.detStartTime).count();
        c.avgDetMs = (c.avgDetMs * c.detCount + ms) / (c.detCount + 1);
        
        c.detRunning = false;
        c.detCount++;
    }
}

void DriveSeg4CamApp::renderDetectionBoxes(uint32_t i)
{
    auto &c = m_dnnCtx[i];
    std::lock_guard<std::mutex> lock(c.detMutex);
    
    if (c.detBoxes.empty()) return;
    
    // Render boxes (red)
    CHECK_DW_ERROR(dwRenderEngine_setColor({1.0f, 0.0f, 0.0f, 1.0f}, m_re));
    CHECK_DW_ERROR(dwRenderEngine_setLineWidth(2.0f, m_re));
    CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D,
                                         c.detBoxes.data(), sizeof(dwRectf), 0,
                                         c.detBoxes.size(), m_re));
    
    // Render labels (white text)
    CHECK_DW_ERROR(dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, m_re));
    CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_12, m_re));
    for (size_t j = 0; j < c.detLabels.size(); ++j) {
        dwVector2f labelPos = {c.detBoxes[j].x + 2, c.detBoxes[j].y - 5};
        CHECK_DW_ERROR(dwRenderEngine_renderText2D(c.detLabels[j].c_str(), labelPos, m_re));
    }
}


void DriveSeg4CamApp::renderCamera(uint32_t i, dwCameraFrameHandle_t frame)
{
    if (!frame) return;

    CHECK_DW_ERROR(dwRenderEngine_setTile(m_tiles[i], m_re));
    CHECK_DW_ERROR(dwRenderEngine_resetTile(m_re));

    // Render camera image (with segmentation overlay already blended)
    CHECK_DW_ERROR(dwImageStreamerGL_producerSend(m_imgRGBA[i], m_streamerToGL[i]));
    dwImageHandle_t frameGL{};
    CHECK_DW_ERROR(dwImageStreamerGL_consumerReceive(&frameGL, 33000, m_streamerToGL[i]));
    dwImageGL* imageGL{};
    CHECK_DW_ERROR(dwImage_getGL(&imageGL, frameGL));

    dwVector2f range{(float)imageGL->prop.width, (float)imageGL->prop.height};
    CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_re));
    CHECK_DW_ERROR(dwRenderEngine_renderImage2D(imageGL, {0, 0, range.x, range.y}, m_re));

    // NEW: Render detection boxes
    renderDetectionBoxes(i);

    // Render stats
    CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_re));
    CHECK_DW_ERROR(dwRenderEngine_setColor({1,1,1,1}, m_re));
    char buf[256];
    auto &c = m_dnnCtx[i];
    std::snprintf(buf, sizeof(buf), "Cam %u | Det:%u(%.1fms) Seg:%u(%.1fms) Boxes:%zu",  // Swapped order
              i, c.detCount, c.avgDetMs, c.segCount, c.avgSegMs, c.detBoxes.size());
    CHECK_DW_ERROR(dwRenderEngine_renderText2D(buf, {16, 28}, m_re));

    CHECK_DW_ERROR(dwImageStreamerGL_consumerReturn(&frameGL, m_streamerToGL[i]));
    CHECK_DW_ERROR(dwImageStreamerGL_producerReturn(nullptr, 32000, m_streamerToGL[i]));
}
// --------------- DNN steps -----------------

void DriveSeg4CamApp::prepareInput(uint32_t i, dwCameraFrameHandle_t frame)
{
    dwImageHandle_t src = DW_NULL_HANDLE;
    CHECK_DW_ERROR(dwSensorCamera_getImage(&src, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, frame));
    CHECK_DW_ERROR(dwImage_copyConvert(m_imgRGBA[i], src, m_ctx));

    auto &c = m_dnnCtx[i];
    CHECK_DW_ERROR(dwDataConditioner_prepareData(
        c.segInTensor, &m_imgRGBA[i], 1, &m_roi[i], cudaAddressModeClamp, c.segConditioner));  // ✓
}


void DriveSeg4CamApp::runInference(uint32_t i)
{
    auto &c = m_dnnCtx[i];
    CHECK_DW_ERROR(dwDNN_setCUDAStream(c.segStream, m_dnn));  // ✓

    dwConstDNNTensorHandle_t in[1] = {c.segInTensor};  // ✓
    CHECK_DW_ERROR(dwDNN_infer(&c.segOutTensorDev, 1, in, 1, m_dnn));  // ✓

    CHECK_DW_ERROR(dwDNNTensorStreamer_producerSend(c.segOutTensorDev, c.segOutStreamer));  // ✓
    cudaEventRecord(c.segInferDone, c.segStream);  // ✓
}

void DriveSeg4CamApp::collectAndOverlay(uint32_t i)
{
    auto &c = m_dnnCtx[i];
    CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReceive(&c.segOutTensorHost, 1000, c.segOutStreamer));

    void* hptr = nullptr;
    CHECK_DW_ERROR(dwDNNTensor_lock(&hptr, c.segOutTensorHost));

    // Get output dimensions (NCHW: [W, H, C, N])
    const int W = (int)m_outProps.dimensionSize[0];
    const int H = (int)m_outProps.dimensionSize[1];
    const int C = (int)m_outProps.dimensionSize[2];
    
    const float* h_logits = reinterpret_cast<const float*>(hptr);
    
    // ========== GPU ARGMAX ==========
    
    // Allocate GPU memory for logits and class map
    float* d_logits = nullptr;
    uint8_t* d_classMap = nullptr;
    const size_t logitsBytes = W * H * C * sizeof(float);
    const size_t classMapBytes = W * H * sizeof(uint8_t);
    
    cudaMalloc(&d_logits, logitsBytes);
    cudaMalloc(&d_classMap, classMapBytes);
    
    // Upload logits to GPU
    cudaMemcpyAsync(d_logits, h_logits, logitsBytes, cudaMemcpyHostToDevice, c.segStream);  
    
    // Launch argmax kernel
    dim3 argmaxBlock(16, 16);
    dim3 argmaxGrid((W + 15) / 16, (H + 15) / 16);
    
    argmaxNCHW_kernel<<<argmaxGrid, argmaxBlock, 0, c.segStream>>>(  
        d_classMap, d_logits, W, H, C
    );
    
    CHECK_DW_ERROR(dwDNNTensor_unlock(c.segOutTensorHost));  
    CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReturn(&c.segOutTensorHost, c.segOutStreamer));  
    CHECK_DW_ERROR(dwDNNTensorStreamer_producerReturn(nullptr, 1000, c.segOutStreamer));        
    
    // ========== COLORIZE & BLEND ==========
    
    // Color LUT: [Background, Drivable, Alternative] (device memory)
    static uint8_t h_colorLUT[3 * 4] = {
        0, 0, 0, 0,          // Class 0: transparent
        0, 255, 0, 180,      // Class 1: green 70% opacity
        255, 255, 0, 150     // Class 2: yellow 60% opacity
    };
    
    static uint8_t* d_colorLUT = nullptr;
    if (!d_colorLUT) {
        cudaMalloc(&d_colorLUT, sizeof(h_colorLUT));
        cudaMemcpy(d_colorLUT, h_colorLUT, sizeof(h_colorLUT), cudaMemcpyHostToDevice);
    }
    
    // Get camera RGBA image
    dwImageCUDA* cudaImg{};
    CHECK_DW_ERROR(dwImage_getCUDA(&cudaImg, m_imgRGBA[i]));
    uint8_t* d_rgba = reinterpret_cast<uint8_t*>(cudaImg->dptr[0]);
    
    const int imgW = cudaImg->prop.width;
    const int imgH = cudaImg->prop.height;
    const int imgStride = cudaImg->pitch[0];
    
    // Launch colorize+blend kernel
    dim3 blendBlock(16, 16);
    dim3 blendGrid((imgW + 15) / 16, (imgH + 15) / 16);
    
    colorizeAndBlend_kernel<<<blendGrid, blendBlock, 0, c.segStream>>>(         d_rgba, d_classMap,
        imgW, imgH, imgStride,
        W, H,
        d_colorLUT, kOverlayAlpha
    );
    
    cudaStreamSynchronize(c.segStream);  
    
    // Cleanup
    cudaFree(d_logits);
    cudaFree(d_classMap);
}

std::string DriveSeg4CamApp::platformPrefix()
{
    int32_t gpu{};
    dwGPUDeviceProperties prop{};
    CHECK_DW_ERROR(dwContext_getGPUDeviceCurrent(&gpu, m_ctx));
    CHECK_DW_ERROR(dwContext_getGPUProperties(&prop, gpu, m_ctx));

    if (prop.major >= 8) return "ampere-integrated";
    if (prop.major == 7) return prop.integrated ? "volta-integrated" : "turing";
    return "pascal";
}

void DriveSeg4CamApp::printStats()
{
    std::cout << "=== Stats ===\n";
    std::cout << "Frames: " << m_frameCounter.load() << "\n";
    std::cout << "Seg Inferences in last 2s: " << (m_recentInferences/2) << " per sec\n";
    for (uint32_t i=0; i<m_numCameras; ++i) {
        auto &c = m_dnnCtx[i];
        std::cout << "  Cam" << i 
          << " | Det: " << c.detCount << " frames, " << c.avgDetMs << "ms"  // Detection first
          << " | Seg: " << c.segCount << " frames, " << c.avgSegMs << "ms"  // Segmentation second
          << " | Boxes: " << c.detBoxes.size() << "\n";
    }
}

bool DriveSeg4CamApp::sort_score(YoloScoreRect box1, YoloScoreRect box2)
{
    return box1.score > box2.score;
}

float32_t DriveSeg4CamApp::calculateIouOfBoxes(dwRectf box1, dwRectf box2)
{
    float32_t x1 = std::max(box1.x, box2.x);
    float32_t y1 = std::max(box1.y, box2.y);
    float32_t x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float32_t y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    float32_t w = std::max(0.0f, x2 - x1);
    float32_t h = std::max(0.0f, y2 - y1);
    float32_t over_area = w * h;
    return float32_t(over_area) / float32_t(box1.width * box1.height + box2.width * box2.height - over_area);
}

std::vector<DriveSeg4CamApp::YoloScoreRect> DriveSeg4CamApp::doNmsForYoloOutputBoxes(
    std::vector<YoloScoreRect>& boxes, float32_t threshold)
{
    std::vector<YoloScoreRect> results;
    std::sort(boxes.begin(), boxes.end(), sort_score);
    
    while (boxes.size() > 0)
    {
        results.push_back(boxes[0]);
        uint32_t index = 1;
        while (index < boxes.size())
        {
            float32_t iou_value = calculateIouOfBoxes(boxes[0].rectf, boxes[index].rectf);
            if (iou_value > threshold)
            {
                boxes.erase(boxes.begin() + index);
            }
            else
            {
                index++;
            }
        }
        boxes.erase(boxes.begin());
    }
    return results;
}

// ---------------- main ----------------
int main(int argc, const char** argv)
{
    ProgramArguments args(argc, argv,
        {
            ProgramArguments::Option_t(
                "rig",
                (dw_samples::SamplesDataPath::get() + "/samples/sensors/camera/camera/rig_4cam.json").c_str(),
                "Rig configuration (expects 4 cameras)"
            ),
#ifdef VIBRANTE
            ProgramArguments::Option_t("cudla", "0", "Run DNN on CuDLA (1=yes)"),
            ProgramArguments::Option_t("dla-engine", "0", "DLA engine index if --cudla=1"),
#endif
            ProgramArguments::Option_t(
                "tensorRT_model",
                "",
                ("Path to TensorRT engine. Default: " + dw_samples::SamplesDataPath::get() + "/samples/detector/ampere-integrated/resnet34_fcn_gpu_fp16.bin").c_str()
            ),
            ProgramArguments::Option_t(
                "detection_model",
                "",
                ("Path to detection TensorRT engine. Default: " + dw_samples::SamplesDataPath::get() + "/samples/detector/ampere-integrated/yolov3_640x640.bin").c_str()
            ),
        },
        "Four-camera segmentation with alpha-blended masks over original frames."
    );

    DriveSeg4CamApp app(args);
    app.initializeWindow("DriveSeG (Blended)", 1280, 800, args.enabled("offscreen"));
    if (!args.enabled("offscreen")) app.setProcessRate(30);
    return app.run();
}
