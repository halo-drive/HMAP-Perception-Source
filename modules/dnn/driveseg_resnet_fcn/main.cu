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

    // DNN
    dwDNNHandle_t m_dnn{DW_NULL_HANDLE};
    uint32_t m_outIdx{0};
    bool m_outputIsFP16{false};
    bool m_outputIsNCHW{true};
    dwDNNTensorProperties m_inProps{};
    dwDNNTensorProperties m_outProps{};

    struct CamDNN {
        cudaStream_t stream{nullptr};
        cudaEvent_t  inferDone{nullptr};
        dwDataConditionerHandle_t conditioner{DW_NULL_HANDLE};
        dwDNNTensorHandle_t inTensor{DW_NULL_HANDLE};
        dwDNNTensorHandle_t outTensorDev{DW_NULL_HANDLE};
        dwDNNTensorStreamerHandle_t outStreamer{DW_NULL_HANDLE};
        dwDNNTensorHandle_t outTensorHost{DW_NULL_HANDLE};
        bool running{false};
        uint64_t frameId{0};
        float avgMs{0.f};
        uint32_t count{0};
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
    for (uint32_t i = 0; i < m_numCameras; ++i) initPerCameraDNN(i);

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
        if (c.inferDone) cudaEventDestroy(c.inferDone);
        if (c.stream) cudaStreamDestroy(c.stream);
        if (c.inTensor) dwDNNTensor_destroy(c.inTensor);
        if (c.outTensorDev) dwDNNTensor_destroy(c.outTensorDev);
        if (c.outStreamer) dwDNNTensorStreamer_release(c.outStreamer);
        if (c.conditioner) dwDataConditioner_release(c.conditioner);
    }

    if (m_dnn) dwDNN_release(m_dnn);

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

    cudaStreamCreate(&c.stream);
    cudaEventCreate(&c.inferDone);

    dwDNNMetaData meta{};
    CHECK_DW_ERROR(dwDNN_getMetaData(&meta, m_dnn));
    CHECK_DW_ERROR(dwDataConditioner_initializeFromTensorProperties(
        &c.conditioner, &m_inProps, 1, &meta.dataConditionerParams, c.stream, m_ctx));

    CHECK_DW_ERROR(dwDNNTensor_create(&c.inTensor, &m_inProps, m_ctx));
    CHECK_DW_ERROR(dwDNNTensor_create(&c.outTensorDev, &m_outProps, m_ctx));

    dwDNNTensorProperties hostProps = m_outProps;
    hostProps.tensorType = DW_DNN_TENSOR_TYPE_CPU;
    CHECK_DW_ERROR(dwDNNTensorStreamer_initialize(&c.outStreamer, &m_outProps, hostProps.tensorType, m_ctx));
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

void DriveSeg4CamApp::startAllInferences(dwCameraFrameHandle_t frames[])
{
    for (uint32_t i = 0; i < m_numCameras; ++i) {
        auto &c = m_dnnCtx[i];
        if (c.running || frames[i] == nullptr) continue;

        try {
            prepareInput(i, frames[i]);
            runInference(i);
            c.running = true;
            c.frameId = m_frameCounter.load();
        } catch (const std::exception& e) {
            std::cout << "[Cam" << i << "] start inference failed: " << e.what() << "\n";
            c.running = false;
        }
    }
}

void DriveSeg4CamApp::maybeCollect(uint32_t i)
{
    auto &c = m_dnnCtx[i];
    if (!c.running) return;

    cudaError_t st = cudaEventQuery(c.inferDone);
    if (st == cudaSuccess) {
        collectAndOverlay(i);
        c.running = false;
        c.count++;
        m_recentInferences++;
    } else if (st != cudaErrorNotReady) {
        std::cout << "[Cam" << i << "] CUDA error: " << cudaGetErrorString(st) << "\n";
        c.running = false;
    }
}

void DriveSeg4CamApp::renderCamera(uint32_t i, dwCameraFrameHandle_t frame)
{
    if (!frame) return;

    CHECK_DW_ERROR(dwRenderEngine_setTile(m_tiles[i], m_re));
    CHECK_DW_ERROR(dwRenderEngine_resetTile(m_re));

    CHECK_DW_ERROR(dwImageStreamerGL_producerSend(m_imgRGBA[i], m_streamerToGL[i]));

    dwImageHandle_t frameGL{};
    CHECK_DW_ERROR(dwImageStreamerGL_consumerReceive(&frameGL, 33000, m_streamerToGL[i]));

    dwImageGL* imageGL{};
    CHECK_DW_ERROR(dwImage_getGL(&imageGL, frameGL));

    dwVector2f range{(float)imageGL->prop.width, (float)imageGL->prop.height};
    CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_re));
    CHECK_DW_ERROR(dwRenderEngine_renderImage2D(imageGL, {0, 0, range.x, range.y}, m_re));

    CHECK_DW_ERROR(dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_re));
    CHECK_DW_ERROR(dwRenderEngine_setColor({1,1,1,1}, m_re));
    char buf[128];
    auto &c = m_dnnCtx[i];
    std::snprintf(buf, sizeof(buf), "Cam %u  frames:%u  avg(ms):%.1f", i, c.count, c.avgMs);
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
        c.inTensor, &m_imgRGBA[i], 1, &m_roi[i], cudaAddressModeClamp, c.conditioner));
}

void DriveSeg4CamApp::runInference(uint32_t i)
{
    auto &c = m_dnnCtx[i];
    CHECK_DW_ERROR(dwDNN_setCUDAStream(c.stream, m_dnn));

    dwConstDNNTensorHandle_t in[1] = {c.inTensor};
    CHECK_DW_ERROR(dwDNN_infer(&c.outTensorDev, 1, in, 1, m_dnn));

    CHECK_DW_ERROR(dwDNNTensorStreamer_producerSend(c.outTensorDev, c.outStreamer));
    cudaEventRecord(c.inferDone, c.stream);
}

void DriveSeg4CamApp::collectAndOverlay(uint32_t i)
{
    auto &c = m_dnnCtx[i];
    CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReceive(&c.outTensorHost, 1000, c.outStreamer));

    void* hptr = nullptr;
    CHECK_DW_ERROR(dwDNNTensor_lock(&hptr, c.outTensorHost));

    // Interpret output dims using DNNTensor header rule:
    // NCHW => dimensionSize = [W, H, C, N]; NHWC => [C, W, H, N]
    int C=1,H=1,W=1;
    if (m_outProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NCHW) {
        W = (int)m_outProps.dimensionSize[0];
        H = (int)m_outProps.dimensionSize[1];
        C = (int)m_outProps.dimensionSize[2];
    } else if (m_outProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NHWC) {
        C = (int)m_outProps.dimensionSize[0];
        W = (int)m_outProps.dimensionSize[1];
        H = (int)m_outProps.dimensionSize[2];
    } else {
        W = (int)m_outProps.dimensionSize[0];
        H = (int)m_outProps.dimensionSize[1];
        C = (int)m_outProps.dimensionSize[2];
    }

    std::vector<uint8_t> hostMask((size_t)H * (size_t)W, 0);
    auto mark_drivable = [&](int x, int y) { hostMask[(size_t)y * (size_t)W + (size_t)x] = 255; };

    if (m_outputIsFP16) {
        const dwFloat16_t* f16 = reinterpret_cast<const dwFloat16_t*>(hptr);
        if (C == 1) {
            for (int y=0; y<H; ++y)
                for (int x=0; x<W; ++x)
                    if ((float)f16[y*W + x] > 0.5f) mark_drivable(x,y);
        } else {
            for (int y=0; y<H; ++y) {
                for (int x=0; x<W; ++x) {
                    int best = 0; float bestv = -1e9f;
                    for (int cix=0; cix<C; ++cix) {
                        float v = (float)f16[cix*H*W + y*W + x];
                        if (v > bestv) { bestv = v; best = cix; }
                    }
                    if (best == 1) mark_drivable(x,y); // assume class 1 = drivable
                }
            }
        }
    } else {
        const float* f32 = reinterpret_cast<const float*>(hptr);
        if (C == 1) {
            for (int y=0; y<H; ++y)
                for (int x=0; x<W; ++x)
                    if (f32[y*W + x] > 0.5f) mark_drivable(x,y);
        } else {
            for (int y=0; y<H; ++y) {
                for (int x=0; x<W; ++x) {
                    int best = 0; float bestv = -1e9f;
                    for (int cix=0; cix<C; ++cix) {
                        float v = f32[cix*H*W + y*W + x];
                        if (v > bestv) { bestv = v; best = cix; }
                    }
                    if (best == 1) mark_drivable(x,y);
                }
            }
        }
    }

    CHECK_DW_ERROR(dwDNNTensor_unlock(c.outTensorHost));
    CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReturn(&c.outTensorHost, c.outStreamer));
    CHECK_DW_ERROR(dwDNNTensorStreamer_producerReturn(nullptr, 1000, c.outStreamer));

    // Upload mask and blend
    dwImageCUDA* cudaImg{};
    CHECK_DW_ERROR(dwImage_getCUDA(&cudaImg, m_imgRGBA[i]));
    uint8_t* d_rgba = reinterpret_cast<uint8_t*>(cudaImg->dptr[0]);
    const int imgW = cudaImg->prop.width;
    const int imgH = cudaImg->prop.height;
    const int imgStride = cudaImg->pitch[0];

    uint8_t* d_mask = nullptr;
    const size_t maskStride = (size_t)W;
    const size_t maskBytes = (size_t)H * maskStride;
    cudaMalloc(&d_mask, maskBytes);
    cudaMemcpy(d_mask, hostMask.data(), maskBytes, cudaMemcpyHostToDevice);

    dim3 blk(16,16);
    dim3 grd((imgW + blk.x - 1)/blk.x, (imgH + blk.y - 1)/blk.y);
    const float4 green = make_float4(0.f, 1.f, 0.f, 1.f);

    auto &cctx = m_dnnCtx[i];
    blendMaskKernel<<<grd, blk, 0, cctx.stream>>>(
        d_rgba, imgW, imgH, imgStride,
        d_mask, W, H, (int)maskStride,
        green, kOverlayAlpha,
        /*linearScaleOnly=*/true,
        /*sx=*/0.f, /*sy=*/0.f, /*ox=*/0.f, /*oy=*/0.f);

    cudaStreamSynchronize(cctx.stream);
    cudaFree(d_mask);
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
    std::cout << "Inferences in last 2s: " << (m_recentInferences/2) << " per sec\n";
    for (uint32_t i=0; i<m_numCameras; ++i) {
        auto &c = m_dnnCtx[i];
        std::cout << "  Cam" << i << " frames:" << c.count << " avg(ms):" << c.avgMs << "\n";
    }
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
        },
        "Four-camera segmentation with alpha-blended masks over original frames."
    );

    DriveSeg4CamApp app(args);
    app.initializeWindow("DriveSeg 4-Cam (Blended)", 1280, 800, args.enabled("offscreen"));
    if (!args.enabled("offscreen")) app.setProcessRate(30);
    return app.run();
}
