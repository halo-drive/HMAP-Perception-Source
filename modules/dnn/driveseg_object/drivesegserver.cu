// Four-camera semantic segmentation + object detection SERVER
// Headless inference server sending results via socket IPC

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
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
#include <queue>              
#include <condition_variable> 
#include <set>                
#include <NvInferPlugin.h>
#include <cuda_runtime.h>

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

// Socket IPC
#include <dw/comms/socketipc/SocketClientServer.h>

// ------------- DW Sample Framework ----------
#include <framework/Checks.hpp>
#include <framework/Log.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/SamplesDataPath.hpp>

using namespace dw_samples::common;

#define MAX_PORTS_COUNT 4
#define MAX_CAMS_PER_PORT 4
#define MAX_CAMS (MAX_PORTS_COUNT * MAX_CAMS_PER_PORT)
static_assert(MAX_CAMS >= 4, "MAX_CAMS must be at least 4");

constexpr float kOverlayAlpha = 0.35f;

// ===============================================================
// CUDA KERNELS 
// ===============================================================

__global__ void argmaxNCHW_kernel(
    uint8_t* classIndices,
    const float* logits,
    int W, int H, int C)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= W || y >= H) return;
    
    const int spatialIdx = y * W + x;
    const int spatialSize = W * H;
    
    float maxVal = logits[spatialIdx];
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

__global__ void colorizeAndBlend_kernel(
    uint8_t* rgba,
    const uint8_t* classIndices,
    int imgW, int imgH, int imgStrideBytes,
    int maskW, int maskH,
    const uint8_t* colorLUT,
    float alpha )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= imgW || y >= imgH) return;
    
    const float mx_f = (float)x * ((float)maskW / (float)imgW);
    const float my_f = (float)y * ((float)maskH / (float)imgH);
    const int mx = (int)mx_f;
    const int my = (int)my_f;
    
    if (mx >= maskW || my >= maskH) return;
    
    const uint8_t classIdx = classIndices[my * maskW + mx];
    if (classIdx == 0) return;
    
    const uint8_t* color = &colorLUT[classIdx * 4];
    const uint8_t colorA = color[3];
    if (colorA == 0) return;
    
    uint8_t* pixel = rgba + y * imgStrideBytes + x * 4;
    
    const float finalAlpha = (colorA / 255.0f) * alpha;
    const float invAlpha = 1.0f - finalAlpha;
    
    pixel[0] = (uint8_t)(invAlpha * pixel[0] + finalAlpha * color[0]);
    pixel[1] = (uint8_t)(invAlpha * pixel[1] + finalAlpha * color[1]);
    pixel[2] = (uint8_t)(invAlpha * pixel[2] + finalAlpha * color[2]);
}

// ===============================================================
// STAGE 2: CUDA PREPROCESSING KERNEL
// ===============================================================

__global__ void preprocessImageForStage2(
    float* output,      // [1, 3, H, W] NCHW normalized float32
    const uint8_t* rgba, // [H, W, 4] uint8 RGBA
    int W, int H, int pitch,
    float meanR, float meanG, float meanB,
    float stdR, float stdG, float stdB)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= W || y >= H) return;
    
    const uint8_t* pixel = rgba + y * pitch + x * 4;
    const int spatialIdx = y * W + x;
    const int spatialSize = W * H;
    
    // Convert RGBA uint8 → RGB float32 normalized (NCHW layout)
    // output[c, y, x] = output[c * H * W + y * W + x]
    output[0 * spatialSize + spatialIdx] = ((pixel[0] / 255.0f) - meanR) / stdR; // R
    output[1 * spatialSize + spatialIdx] = ((pixel[1] / 255.0f) - meanG) / stdG; // G
    output[2 * spatialSize + spatialIdx] = ((pixel[2] / 255.0f) - meanB) / stdB; // B
}



// ===============================================================
// IPC DATA STRUCTURES
// ===============================================================

#pragma pack(push, 1)
struct FrameHeader {
    uint32_t magic;           // 0xDEADBEEF
    uint32_t cameraIndex;
    uint64_t frameId;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t imageDataSize;
    uint32_t numBoxes;
    uint32_t segCount;
    uint32_t detCount;
    float avgSegMs;
    float avgDetMs;
    float avgStage2Ms;
    uint32_t num3DBoxes;
};

struct DetectionBox {
    float x, y, width, height;
    char label[64];
};

struct Detection3DBox {
    float depth;        // in meters
    float height;       // in meters
    float width;        // in meters
    float length;      // in meters
    float rotation;     // in radians
    float iouScore;     // quality score
};
#pragma pack(pop)

struct SerializableFrame {
    uint32_t cameraIndex;
    uint64_t frameId;
    
    // Image
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    std::vector<uint8_t> rgbaPixels;
    
    // Detections
    std::vector<dwRectf> boxes;
    std::vector<std::string> labels;
    
    //3d detections 
    std::vector<float> depths;       // [N] depth in meters
    std::vector<float> dimensions;   // [N*3] h,w,l in meters
    std::vector<float> rotations;    // [N] rotation in radians
    std::vector<float> iouScores;    // [N] quality scores

    // Stats
    uint32_t segCount;
    uint32_t detCount;
    float avgSegMs;
    float avgDetMs;
    float avgStage2Ms;
};

// ===============================================================
// SERVER APPLICATION
// ===============================================================

__global__ void downscaleKernel(
    uint8_t* dst,
    const uint8_t* src,
    int dstW, int dstH,
    int srcW, int srcH,
    int srcPitch,
    float scaleX, float scaleY)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dstW || y >= dstH) return;
    
    // Bilinear sampling
    const float srcX = x * scaleX;
    const float srcY = y * scaleY;
    
    const int x0 = (int)srcX;
    const int y0 = (int)srcY;
    const int x1 = min(x0 + 1, srcW - 1);
    const int y1 = min(y0 + 1, srcH - 1);
    
    const float fx = srcX - x0;
    const float fy = srcY - y0;
    
    // Sample 4 neighbors
    const uint8_t* p00 = src + y0 * srcPitch + x0 * 4;
    const uint8_t* p10 = src + y0 * srcPitch + x1 * 4;
    const uint8_t* p01 = src + y1 * srcPitch + x0 * 4;
    const uint8_t* p11 = src + y1 * srcPitch + x1 * 4;
    
    uint8_t* out = dst + y * (dstW * 4) + x * 4;
    
    // Bilinear interpolation for each channel
    for (int c = 0; c < 4; ++c) {
        float v0 = p00[c] * (1 - fx) + p10[c] * fx;
        float v1 = p01[c] * (1 - fx) + p11[c] * fx;
        out[c] = (uint8_t)(v0 * (1 - fy) + v1 * fy);
    }
}

class DriveSeg4CamServer
{
public:
    explicit DriveSeg4CamServer(const ProgramArguments& args);
    ~DriveSeg4CamServer();

    bool initialize();
    void run();
    void release();

private:
    // Core DW
    ProgramArguments m_args;
    dwContextHandle_t m_ctx{DW_NULL_HANDLE};
    dwSALHandle_t m_sal{DW_NULL_HANDLE};
    dwRigHandle_t m_rig{DW_NULL_HANDLE};

    // Cameras
    uint32_t m_numCameras{0};
    dwSensorHandle_t m_cam[MAX_CAMS]{DW_NULL_HANDLE};
    dwRect m_roi[MAX_CAMS]{};
    dwImageHandle_t m_imgRGBA[MAX_CAMS]{DW_NULL_HANDLE};

    // Segmentation DNN
    dwDNNHandle_t m_dnn{DW_NULL_HANDLE};
    dwDNNTensorProperties m_inProps{};
    dwDNNTensorProperties m_outProps{};
    bool m_outputIsFP16{false};
    bool m_outputIsNCHW{true};
    
    // Detection DNN
    dwDNNHandle_t m_detDnn{DW_NULL_HANDLE};
    dwDNNTensorProperties m_detInProps{};
    dwDNNTensorProperties m_detOutProps{};

    //Stage2 Detection DNN  
    dwDNNHandle_t m_stage2Dnn[MAX_CAMS]{DW_NULL_HANDLE};
    dwDNNTensorProperties m_stage2InPropsImage{};    // images input
    dwDNNTensorProperties m_stage2InPropsBoxes{};    // boxes_2d input
    dwDNNTensorProperties m_stage2OutProps[7]{};     // 7 outputs
    uint32_t m_stage2OutputCount{0};

    static constexpr uint32_t STAGE2_MAX_BOXES = 100;  // Must match ONNX export
    //IPC isolaters 
    std::vector<std::thread> m_sendThreads;
    std::queue<SerializableFrame> m_sendQueue[MAX_CAMS];
    std::mutex m_sendMutex[MAX_CAMS];
    std::condition_variable m_sendCV[MAX_CAMS];

    void sendThreadFunc(uint32_t camIndex);


    // YOLO classes
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
        dwRectf rectf640;
        float32_t score;
        uint16_t classIndex;
    } YoloScoreRect;

    struct CamDNN {
        // Segmentation
        cudaStream_t segStream{nullptr};
        cudaEvent_t segInferDone{nullptr};
        dwDataConditionerHandle_t segConditioner{DW_NULL_HANDLE};
        dwDNNTensorHandle_t segInTensor{DW_NULL_HANDLE};
        dwDNNTensorHandle_t segOutTensorDev{DW_NULL_HANDLE};
        dwDNNTensorStreamerHandle_t segOutStreamer{DW_NULL_HANDLE};
        dwDNNTensorHandle_t segOutTensorHost{DW_NULL_HANDLE};
        bool segRunning{false};
        
        // Detection
        cudaStream_t detStream{nullptr};
        cudaEvent_t detInferDone{nullptr};
        dwDataConditionerHandle_t detConditioner{DW_NULL_HANDLE};
        dwDNNTensorHandle_t detInTensor{DW_NULL_HANDLE};
        dwDNNTensorHandle_t detOutTensorDev{DW_NULL_HANDLE};
        dwDNNTensorStreamerHandle_t detOutStreamer{DW_NULL_HANDLE};
        dwDNNTensorHandle_t detOutTensorHost{DW_NULL_HANDLE};
        bool detRunning{false};
        
        std::vector<dwRectf> detBoxes;
        std::vector<dwRectf> detBoxes640;
        std::vector<std::string> detLabels;
        std::mutex detMutex;
        
        uint64_t frameId{0};
        float avgSegMs{0.f};
        float avgDetMs{0.f};
        uint32_t segCount{0};
        uint32_t detCount{0};
        
        std::chrono::high_resolution_clock::time_point segStartTime;
        std::chrono::high_resolution_clock::time_point detStartTime;
        

        //stage 2 detection
        dwDNNTensorHandle_t stage2InTensorImage{DW_NULL_HANDLE};
        dwDNNTensorHandle_t stage2InTensorBoxes{DW_NULL_HANDLE};
        dwDNNTensorHandle_t stage2OutTensorsDev[7]{DW_NULL_HANDLE};
        
        dwDNNTensorStreamerHandle_t stage2OutStreamers[7]{DW_NULL_HANDLE};
        dwDNNTensorHandle_t stage2OutTensorsHost[7]{DW_NULL_HANDLE};
        
        cudaStream_t stage2Stream{nullptr};
        cudaEvent_t stage2InferDone{nullptr};

        bool stage2Running{false};

        float* stage2HostDepth{nullptr};
        float* stage2HostDims{nullptr};
        float* stage2HostRotBins{nullptr};
        float* stage2HostRotRes{nullptr};

        // 3D results
        std::vector<float> depths;       // [N] depth in meters
        std::vector<float> dimensions;   // [N*3] h,w,l in meters  
        std::vector<float> rotations;    // [N] rotation in radians
        std::vector<float> iouScores;    // [N] quality scores
        std::mutex stage2Mutex;
        
        std::chrono::high_resolution_clock::time_point stage2StartTime;
        float avgStage2Ms{0.f};
        uint32_t stage2Count{0};
    
    } m_dnnCtx[MAX_CAMS];

    // IPC Socket Server
    dwSocketServerHandle_t m_socketServers[MAX_CAMS]{DW_NULL_HANDLE};
    dwSocketConnectionHandle_t m_connections[MAX_CAMS]{DW_NULL_HANDLE};
    std::mutex m_socketMutex;
    uint16_t m_serverPort{49252};

    // Stats
    std::atomic<bool> m_running{true};
    std::atomic<uint64_t> m_frameCounter{0};
    std::chrono::high_resolution_clock::time_point m_lastStats{};
    uint32_t m_recentInferences{0};

private:
    // Init
    void initDW();
    void initCameras();
    void initDNN();
    void initDetectionDNN();
    void initPerCameraDNN(uint32_t i);
    void initPerCameraDetection(uint32_t i);
    void initSocketServer();

    // Per-frame processing
    void processFrame();
    void grabFrames(dwCameraFrameHandle_t frames[]);
    void startAllInferences(dwCameraFrameHandle_t frames[]);
    void maybeCollect(uint32_t i);

    // DNN steps
    void prepareInput(uint32_t i, dwCameraFrameHandle_t frame);
    void runInference(uint32_t i);
    void collectAndOverlay(uint32_t i);
    void runDetectionInference(uint32_t i);
    void collectDetectionResults(uint32_t i);

    // Frame extraction & transmission
    SerializableFrame extractFrameData(uint32_t i);
    bool sendFrameToClients(const SerializableFrame& frame);

    // Utilities
    std::string platformPrefix();
    void printStats();
    static bool sort_score(YoloScoreRect box1, YoloScoreRect box2);
    float32_t calculateIouOfBoxes(dwRectf box1, dwRectf box2);
    std::vector<YoloScoreRect> doNmsForYoloOutputBoxes(std::vector<YoloScoreRect>& boxes, float32_t threshold);

    // stage 2 
    void initStage2DNN();
    void initPerCameraStage2(uint32_t i);
    void runStage2Inference(uint32_t i);
    void collectStage2Results(uint32_t i);

};

// ===============================================================
// IMPLEMENTATION
// ===============================================================

DriveSeg4CamServer::DriveSeg4CamServer(const ProgramArguments& args)
    : m_args(args)
{
}

DriveSeg4CamServer::~DriveSeg4CamServer()
{
    release();
}

bool DriveSeg4CamServer::initialize()
{
    std::cout << "[Server] Initializing...\n";

    initDW();
    CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_ctx));

    initCameras();
    CHECK_DW_ERROR(dwSAL_start(m_sal));

    initDNN();
    initDetectionDNN();
    initStage2DNN();

    for (uint32_t i = 0; i < m_numCameras; ++i) {
        initPerCameraDNN(i);
        initPerCameraDetection(i);
        initPerCameraStage2(i);
    }

    initSocketServer();

    for (uint32_t i = 0; i < m_numCameras; ++i) {
        m_sendThreads.emplace_back(&DriveSeg4CamServer::sendThreadFunc, this, i);
    }

    for (uint32_t i = 0; i < m_numCameras; ++i) {
        CHECK_DW_ERROR(dwSensor_start(m_cam[i]));
    }

    m_lastStats = std::chrono::high_resolution_clock::now();
    std::cout << "[Server] Initialized for " << m_numCameras << " camera(s)\n";
    std::cout << "[Server] Listening on port " << m_serverPort << "\n";
    
    return true;
}


void DriveSeg4CamServer::release()
{
    std::cout << "[Server] Releasing...\n";
    m_running = false;

    for (uint32_t i = 0; i < m_numCameras; ++i) {
        m_sendCV[i].notify_all();
    }
    
    for (auto& thread : m_sendThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Stop cameras
    for (uint32_t i = 0; i < m_numCameras; ++i) {
        if (m_cam[i]) dwSensor_stop(m_cam[i]);
    }

    // Release IPC
    for (uint32_t i = 0; i < MAX_CAMS; ++i) {
        if (m_connections[i]) {
            dwSocketConnection_release(m_connections[i]);
        }
    }
    
    for (uint32_t i = 0; i < m_numCameras; ++i) {
        if (m_socketServers[i]) {
            dwSocketServer_release(m_socketServers[i]);
        }
    }

    // Release DNN contexts
    for (uint32_t i = 0; i < m_numCameras; ++i) {
        auto &c = m_dnnCtx[i];
        
        if (c.segInferDone) cudaEventDestroy(c.segInferDone);
        if (c.segStream) cudaStreamDestroy(c.segStream);
        if (c.segInTensor) dwDNNTensor_destroy(c.segInTensor);
        if (c.segOutTensorDev) dwDNNTensor_destroy(c.segOutTensorDev);
        if (c.segOutStreamer) dwDNNTensorStreamer_release(c.segOutStreamer);
        if (c.segConditioner) dwDataConditioner_release(c.segConditioner);
        
        if (c.detInferDone) cudaEventDestroy(c.detInferDone);
        if (c.detStream) cudaStreamDestroy(c.detStream);
        if (c.detInTensor) dwDNNTensor_destroy(c.detInTensor);
        if (c.detOutTensorDev) dwDNNTensor_destroy(c.detOutTensorDev);
        if (c.detOutStreamer) dwDNNTensorStreamer_release(c.detOutStreamer);
        if (c.detConditioner) dwDataConditioner_release(c.detConditioner);
        
        if (c.stage2InferDone) cudaEventDestroy(c.stage2InferDone);
        if (c.stage2Stream) cudaStreamDestroy(c.stage2Stream);
        if (c.stage2InTensorImage) dwDNNTensor_destroy(c.stage2InTensorImage);
        if (c.stage2InTensorBoxes) dwDNNTensor_destroy(c.stage2InTensorBoxes);
        
        for (uint32_t j = 0; j < m_stage2OutputCount; ++j) {
            if (c.stage2OutTensorsDev[j]) dwDNNTensor_destroy(c.stage2OutTensorsDev[j]);
            if (c.stage2OutStreamers[j]) dwDNNTensorStreamer_release(c.stage2OutStreamers[j]);
        }
    }

    // Release shared DNNs (segmentation and detection)
    if (m_dnn) dwDNN_release(m_dnn);
    if (m_detDnn) dwDNN_release(m_detDnn);
    
    // Release per-camera Stage 2 DNNs 
    for (uint32_t i = 0; i < m_numCameras; ++i) {
        if (m_stage2Dnn[i]) {
            dwDNN_release(m_stage2Dnn[i]);
            m_stage2Dnn[i] = DW_NULL_HANDLE;
        }
    }

    if (m_rig) dwRig_release(m_rig);
    if (m_sal) dwSAL_release(m_sal);
    if (m_ctx) dwRelease(m_ctx);

    std::cout << "[Server] Released.\n";
}


void DriveSeg4CamServer::run()
{
    // Single frame processing (no internal loop)
    processFrame();

    auto now = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - m_lastStats).count() >= 2) {
        printStats();
        m_lastStats = now;
        m_recentInferences = 0;
    }
}

void DriveSeg4CamServer::initDW()
{
    CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
    CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));
    dwContextParameters p{};
    CHECK_DW_ERROR(dwInitialize(&m_ctx, DW_VERSION, &p));
}

void DriveSeg4CamServer::initCameras()
{
    CHECK_DW_ERROR(dwRig_initializeFromFile(&m_rig, m_ctx, m_args.get("rig").c_str()));

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
    }
}

void DriveSeg4CamServer::initDNN()
{
#ifdef VIBRANTE
    bool useDLA = (m_args.get("cudla") == "1");
    uint32_t dlaEngine = (uint32_t)std::atoi(m_args.get("dla-engine").c_str());
#endif

    std::string trtPath = m_args.get("tensorRT_model");
    if (trtPath.empty()) {
        trtPath = dw_samples::SamplesDataPath::get() + std::string("/samples/detector/");
        trtPath += platformPrefix();
        trtPath += "/resnet34_fcn_gpu_fp16.bin";
    }
    std::cout << "[DNN] Loading segmentation model: " << trtPath << "\n";

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
    m_outputIsNCHW = (m_outProps.tensorLayout == DW_DNN_TENSOR_LAYOUT_NCHW);

    std::cout << "[DNN] Segmentation model initialized\n";
}

void DriveSeg4CamServer::initDetectionDNN()
{
#ifdef VIBRANTE
    bool useDLA = (m_args.get("cudla") == "1");
    uint32_t dlaEngine = (uint32_t)std::atoi(m_args.get("dla-engine").c_str());
#endif

    std::string detModelPath = m_args.get("detection_model");
    if (detModelPath.empty()) {
        detModelPath = dw_samples::SamplesDataPath::get() + std::string("/samples/detector/");
        detModelPath += platformPrefix();
        detModelPath += "/yolov3_640x640.bin";
    }
    
    std::cout << "[DNN] Loading detection model: " << detModelPath << "\n";

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
    
    std::cout << "[DNN] Detection model initialized\n";
}

void DriveSeg4CamServer::initStage2DNN()
{
    // 1. Register TensorRT plugins FIRST
    if (!initLibNvInferPlugins(nullptr, "")) {
        throw std::runtime_error("Failed to initialize TensorRT plugins");
    }
    std::cout << "[DNN] TensorRT plugins registered\n";

#ifdef VIBRANTE
    bool useDLA = (m_args.get("cudla") == "1");
    uint32_t dlaEngine = (uint32_t)std::atoi(m_args.get("dla-engine").c_str());
#endif

    // 2. Load the model
    std::string stage2Path = m_args.get("stage2_model");
    if (stage2Path.empty()) {
        stage2Path = dw_samples::SamplesDataPath::get() + std::string("/samples/detector/");
        stage2Path += platformPrefix();
        stage2Path += "/stage2_3d_heads.bin";
    }
    
    std::cout << "[DNN] Loading Stage 2 (3D detection) model: " << stage2Path << "\n";

    // Load model for first camera (we'll query properties from this)
#ifdef VIBRANTE
    CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFileWithEngineId(
        &m_stage2Dnn[0], stage2Path.c_str(), nullptr,
        useDLA ? DW_PROCESSOR_TYPE_CUDLA : DW_PROCESSOR_TYPE_GPU,
        useDLA ? (int32_t)dlaEngine : 0, m_ctx));
#else
    CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFile(
        &m_stage2Dnn[0], stage2Path.c_str(), nullptr, DW_PROCESSOR_TYPE_GPU, m_ctx));
#endif

    // Get actual output count and store it
    CHECK_DW_ERROR(dwDNN_getOutputBlobCount(&m_stage2OutputCount, m_stage2Dnn[0]));
    std::cout << "[DNN] Stage 2 has " << m_stage2OutputCount << " outputs\n";
    
    if (m_stage2OutputCount > 7) {
        std::cout << "[DNN] WARNING: Stage 2 has more outputs than expected, clamping to 7\n";
        m_stage2OutputCount = 7;
    }

    // Get input properties
    dwDNNTensorProperties imgProps{};
    CHECK_DW_ERROR(dwDNN_getInputTensorProperties(&imgProps, 0, m_stage2Dnn[0]));
    
    dwDNNTensorProperties boxProps{};
    CHECK_DW_ERROR(dwDNN_getInputTensorProperties(&boxProps, 1, m_stage2Dnn[0]));
    
    m_stage2InPropsImage = imgProps;
    m_stage2InPropsBoxes = boxProps;
    
    // Get output properties - only for actual outputs
    for (uint32_t i = 0; i < m_stage2OutputCount; ++i) {
        CHECK_DW_ERROR(dwDNN_getOutputTensorProperties(&m_stage2OutProps[i], i, m_stage2Dnn[0]));
    }
    
    // Load model for remaining cameras
    for (uint32_t i = 1; i < m_numCameras; ++i) {
#ifdef VIBRANTE
        CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFileWithEngineId(
            &m_stage2Dnn[i], stage2Path.c_str(), nullptr,
            useDLA ? DW_PROCESSOR_TYPE_CUDLA : DW_PROCESSOR_TYPE_GPU,
            useDLA ? (int32_t)dlaEngine : 0, m_ctx));
#else
        CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFile(
            &m_stage2Dnn[i], stage2Path.c_str(), nullptr, DW_PROCESSOR_TYPE_GPU, m_ctx));
#endif
        std::cout << "[DNN] Stage 2 model initialized for camera " << i << "\n";
    }
    
    std::cout << "[DNN] Stage 2 (3D detection) initialization complete\n";
    std::cout << "  Max boxes: " << STAGE2_MAX_BOXES << "\n";
    std::cout << "  Output count: " << m_stage2OutputCount << "\n";
}




void DriveSeg4CamServer::initPerCameraDNN(uint32_t i)
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


void DriveSeg4CamServer::initPerCameraDetection(uint32_t i)
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


void DriveSeg4CamServer::initPerCameraStage2(uint32_t i)
{
    auto &c = m_dnnCtx[i];
    
    cudaStreamCreate(&c.stage2Stream);
    cudaEventCreate(&c.stage2InferDone);
    
    // NO tensors, NO DataConditioner, NO streamers
    // Using dwDNN_inferRaw with raw GPU buffers
    
    c.depths.reserve(STAGE2_MAX_BOXES);
    c.dimensions.reserve(STAGE2_MAX_BOXES * 3);
    c.rotations.reserve(STAGE2_MAX_BOXES);
    c.iouScores.reserve(STAGE2_MAX_BOXES);
    
    std::cout << "[Stage2] Per-camera context initialized for cam " << i << " (using inferRaw)\n";
}


void DriveSeg4CamServer::initSocketServer()
{
    m_serverPort = static_cast<uint16_t>(std::stoul(m_args.get("port")));
    
    // Create separate server for each camera
    for (uint32_t i = 0; i < m_numCameras; ++i) {
        uint16_t port = m_serverPort + i;
        CHECK_DW_ERROR(dwSocketServer_initialize(&m_socketServers[i], port, 1, m_ctx));
        std::cout << "[IPC] Socket server " << i << " initialized on port " << port << "\n";
    }
    
    std::cout << "[IPC] Will accept connections in send threads...\n";
}


// ===============================================================
// FRAME PROCESSING
// ===============================================================
void DriveSeg4CamServer::processFrame()
{
    dwCameraFrameHandle_t frames[MAX_CAMS]{};
    grabFrames(frames);
    startAllInferences(frames);

    for (uint32_t i = 0; i < m_numCameras; ++i) {
        maybeCollect(i);
    }


    for (uint32_t i = 0; i < m_numCameras; ++i) {
        if (frames[i]) dwSensorCamera_returnFrame(&frames[i]);
    }

    m_frameCounter++;
}

void DriveSeg4CamServer::sendThreadFunc(uint32_t camIndex)
{
    std::cout << "[Send Thread " << camIndex << "] Started\n";
    
    std::cout << "[Send Thread " << camIndex << "] Waiting for client on port " 
              << (m_serverPort + camIndex) << "...\n";
    
    dwStatus status = DW_TIME_OUT;
    while (m_running && status == DW_TIME_OUT) {
        status = dwSocketServer_accept(&m_connections[camIndex], 1000, m_socketServers[camIndex]);
    }
    
    if (status != DW_SUCCESS) {
        std::cout << "[Send Thread " << camIndex << "] Failed to accept client\n";
        return;
    }
    
    std::cout << "[Send Thread " << camIndex << "] Client connected!\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "[Send Thread " << camIndex << "] Starting transmission...\n";
    
    while (m_running) {
        SerializableFrame frame;
        
        {
            std::unique_lock<std::mutex> lock(m_sendMutex[camIndex]);
            m_sendCV[camIndex].wait_for(lock, std::chrono::milliseconds(100), 
                [this, camIndex] { 
                    return !m_sendQueue[camIndex].empty() || !m_running; 
                });
            
            if (!m_running) break;
            if (m_sendQueue[camIndex].empty()) continue;
            
            frame = std::move(m_sendQueue[camIndex].front());
            m_sendQueue[camIndex].pop();
        }
        
        auto sendStart = std::chrono::high_resolution_clock::now();
        
        if (!sendFrameToClients(frame)) {
            std::cout << "[Send Thread " << camIndex << "] Send failed\n";
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        auto sendEnd = std::chrono::high_resolution_clock::now();
        float sendMs = std::chrono::duration<float, std::milli>(sendEnd - sendStart).count();
        
        if (frame.frameId % 30 == 0) {  // Log every 30 frames
            std::cout << "[Send Thread " << camIndex << "] Frame " << frame.frameId 
                      << " transmitted in " << sendMs << "ms\n";
        }
    }
    
    std::cout << "[Send Thread " << camIndex << "] Stopped\n";
}


void DriveSeg4CamServer::grabFrames(dwCameraFrameHandle_t frames[])
{
    for (uint32_t i = 0; i < m_numCameras; ++i) {
        dwStatus s = DW_NOT_READY;
        int spins = 0;
        while (s == DW_NOT_READY) {
            s = dwSensorCamera_readFrame(&frames[i], 33333, m_cam[i]);
            if (++spins > 10) break;
        }
        if (s == DW_TIME_OUT) {
            frames[i] = nullptr;
        } else if (s != DW_SUCCESS) {
            frames[i] = nullptr;
        }
    }
}

void DriveSeg4CamServer::startAllInferences(dwCameraFrameHandle_t frames[])
{
    for (uint32_t i = 0; i < m_numCameras; ++i) {
        auto &c = m_dnnCtx[i];
        if (frames[i] == nullptr) continue;

        // Start detection
        if (!c.detRunning) {
            try {
                c.detStartTime = std::chrono::high_resolution_clock::now();
                
                dwImageHandle_t src = DW_NULL_HANDLE;
                CHECK_DW_ERROR(dwSensorCamera_getImage(&src, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, frames[i]));
                CHECK_DW_ERROR(dwImage_copyConvert(m_imgRGBA[i], src, m_ctx));
                
                CHECK_DW_ERROR(dwDataConditioner_prepareData(
                    c.detInTensor, &m_imgRGBA[i], 1, &m_roi[i], 
                    cudaAddressModeClamp, c.detConditioner));
                
                runDetectionInference(i);
                c.detRunning = true;
            } catch (const std::exception& e) {
                std::cout << "[Cam" << i << "] Detection inference failed: " << e.what() << "\n";
            }
        }
        
        // Start segmentation
        if (!c.segRunning) {
            try {
                c.segStartTime = std::chrono::high_resolution_clock::now();
                prepareInput(i, frames[i]);
                runInference(i);
                c.segRunning = true;
                c.frameId = m_frameCounter.load();
            } catch (const std::exception& e) {
                std::cout << "[Cam" << i << "] Segmentation inference failed: " << e.what() << "\n";
            }
        }
    }
}

void DriveSeg4CamServer::maybeCollect(uint32_t i)
{
    auto &c = m_dnnCtx[i];
    
    // Check Stage 2 completion FIRST (before starting new one)
    if (c.stage2Running && cudaEventQuery(c.stage2InferDone) == cudaSuccess) {
        collectStage2Results(i);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(endTime - c.stage2StartTime).count();
        c.avgStage2Ms = (c.avgStage2Ms * c.stage2Count + ms) / (c.stage2Count + 1);
        
        c.stage2Running = false;
        c.stage2Count++;
        
        std::cout << "[Cam" << i << "] Stage 2 complete: " 
                  << c.depths.size() << " 3D predictions in " << ms << "ms\n";
    }
    
    // Check segmentation
    if (c.segRunning && cudaEventQuery(c.segInferDone) == cudaSuccess) {
        collectAndOverlay(i);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(endTime - c.segStartTime).count();
        c.avgSegMs = (c.avgSegMs * c.segCount + ms) / (c.segCount + 1);
        
        c.segRunning = false;
        c.segCount++;
        m_recentInferences++;
        
        SerializableFrame frameData = extractFrameData(i);
        {
            std::lock_guard<std::mutex> lock(m_sendMutex[i]);
            if (m_sendQueue[i].size() < 2) {
                m_sendQueue[i].push(std::move(frameData));
                m_sendCV[i].notify_one();
            }
        }
    }
    
    // Check 2D detection - only start Stage 2 if NOT already running
    if (c.detRunning && cudaEventQuery(c.detInferDone) == cudaSuccess) {
        collectDetectionResults(i);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(endTime - c.detStartTime).count();
        c.avgDetMs = (c.avgDetMs * c.detCount + ms) / (c.detCount + 1);
        
        c.detRunning = false;
        c.detCount++;
        
        // Only start Stage 2 if we have boxes AND Stage 2 is not already running
        if (!c.detBoxes.empty() && !c.stage2Running) {
            c.stage2StartTime = std::chrono::high_resolution_clock::now();
            try {
                runStage2Inference(i);
                c.stage2Running = true;
            } catch (const std::exception& e) {
                std::cout << "[Cam" << i << "] Stage 2 inference failed: " << e.what() << "\n";
            }
        }
    }
}

void DriveSeg4CamServer::prepareInput(uint32_t i, dwCameraFrameHandle_t frame)
{
    dwImageHandle_t src = DW_NULL_HANDLE;
    CHECK_DW_ERROR(dwSensorCamera_getImage(&src, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, frame));
    CHECK_DW_ERROR(dwImage_copyConvert(m_imgRGBA[i], src, m_ctx));

    auto &c = m_dnnCtx[i];
    CHECK_DW_ERROR(dwDataConditioner_prepareData(
        c.segInTensor, &m_imgRGBA[i], 1, &m_roi[i], cudaAddressModeClamp, c.segConditioner));
}

void DriveSeg4CamServer::runInference(uint32_t i)
{
    auto &c = m_dnnCtx[i];
    CHECK_DW_ERROR(dwDNN_setCUDAStream(c.segStream, m_dnn));

    dwConstDNNTensorHandle_t in[1] = {c.segInTensor};
    CHECK_DW_ERROR(dwDNN_infer(&c.segOutTensorDev, 1, in, 1, m_dnn));

    CHECK_DW_ERROR(dwDNNTensorStreamer_producerSend(c.segOutTensorDev, c.segOutStreamer));
    cudaEventRecord(c.segInferDone, c.segStream);
}

void DriveSeg4CamServer::runDetectionInference(uint32_t i)
{
    auto &c = m_dnnCtx[i];
    CHECK_DW_ERROR(dwDNN_setCUDAStream(c.detStream, m_detDnn));

    dwConstDNNTensorHandle_t in[1] = {c.detInTensor};
    CHECK_DW_ERROR(dwDNN_infer(&c.detOutTensorDev, 1, in, 1, m_detDnn));

    CHECK_DW_ERROR(dwDNNTensorStreamer_producerSend(c.detOutTensorDev, c.detOutStreamer));
    cudaEventRecord(c.detInferDone, c.detStream);
}


void DriveSeg4CamServer::runStage2Inference(uint32_t i)
{
    auto &c = m_dnnCtx[i];
    
    // ========== RESET DNN ==========
    CHECK_DW_ERROR(dwDNN_reset(m_stage2Dnn[i]));
    CHECK_DW_ERROR(dwDNN_setCUDAStream(c.stage2Stream, m_stage2Dnn[i]));
    
    // ========== GET ORIGINAL IMAGE ==========
    dwImageCUDA* cudaImg{};
    CHECK_DW_ERROR(dwImage_getCUDA(&cudaImg, m_imgRGBA[i]));
    const uint32_t origWidth = cudaImg->prop.width;
    const uint32_t origHeight = cudaImg->prop.height;
    const uint32_t origPitch = cudaImg->pitch[0];
    uint8_t* d_rgba = reinterpret_cast<uint8_t*>(cudaImg->dptr[0]);
    
    const uint32_t stage2Width = 640;
    const uint32_t stage2Height = 640;
    
    // ========== 1. ALLOCATE RAW GPU BUFFERS (once) ==========
    // These are raw float* buffers, not DW tensors
    static float* d_inputImage = nullptr;      // [1, 3, 640, 640]
    static float* d_inputBoxes = nullptr;      // [100, 5]
    static float* d_outputDepth = nullptr;     // [100, 3]
    static float* d_outputDims = nullptr;      // [100, 3]
    static float* d_outputRotBins = nullptr;   // [100, 12]
    static float* d_outputRotRes = nullptr;    // [100, 12]
    static uint8_t* d_resizedRGBA = nullptr;   // Temporary for resize
    

    // Allocate host buffers (static to avoid repeated allocation)
    static std::vector<float> h_depth(STAGE2_MAX_BOXES * 3);
    static std::vector<float> h_dims(STAGE2_MAX_BOXES * 3);
    static std::vector<float> h_rotBins(STAGE2_MAX_BOXES * 12);
    static std::vector<float> h_rotRes(STAGE2_MAX_BOXES * 12);


    if (!d_inputImage) {
        cudaMalloc(&d_inputImage, 1 * 3 * stage2Height * stage2Width * sizeof(float));
        cudaMalloc(&d_inputBoxes, STAGE2_MAX_BOXES * 5 * sizeof(float));
        cudaMalloc(&d_outputDepth, STAGE2_MAX_BOXES * 3 * sizeof(float));
        cudaMalloc(&d_outputDims, STAGE2_MAX_BOXES * 3 * sizeof(float));
        cudaMalloc(&d_outputRotBins, STAGE2_MAX_BOXES * 12 * sizeof(float));
        cudaMalloc(&d_outputRotRes, STAGE2_MAX_BOXES * 12 * sizeof(float));
        cudaMalloc(&d_resizedRGBA, stage2Width * stage2Height * 4);
        std::cout << "[Stage2] Allocated raw GPU buffers\n";
    }
    
    // ========== 2. PREPROCESS IMAGE (CUDA kernels) ==========
    float scaleX = (float)origWidth / stage2Width;
    float scaleY = (float)origHeight / stage2Height;
    
    dim3 block(16, 16);
    dim3 grid((stage2Width + 15) / 16, (stage2Height + 15) / 16);
    
    // Resize to 640x640
    downscaleKernel<<<grid, block, 0, c.stage2Stream>>>(
        d_resizedRGBA, d_rgba,
        stage2Width, stage2Height,
        origWidth, origHeight,
        origPitch,
        scaleX, scaleY
    );
    
    // Normalize with ImageNet stats → NCHW float
    const float meanR = 0.485f, meanG = 0.456f, meanB = 0.406f;
    const float stdR = 0.229f, stdG = 0.224f, stdB = 0.225f;
    
    preprocessImageForStage2<<<grid, block, 0, c.stage2Stream>>>(
        d_inputImage, d_resizedRGBA,
        stage2Width, stage2Height, stage2Width * 4,
        meanR, meanG, meanB,
        stdR, stdG, stdB
    );
    
    // ========== 3. PREPARE BOXES ==========
    std::vector<float> boxesData(STAGE2_MAX_BOXES * 5, 0.0f);
    size_t numActualBoxes = std::min<size_t>(c.detBoxes.size(), STAGE2_MAX_BOXES);
    
    for (size_t j = 0; j < numActualBoxes; ++j) {
        const auto& box = c.detBoxes[j];
        
        // Convert original coords → 640×640 stretched coords
        float x1 = (box.x / origWidth) * stage2Width;
        float y1 = (box.y / origHeight) * stage2Height;
        float x2 = ((box.x + box.width) / origWidth) * stage2Width;
        float y2 = ((box.y + box.height) / origHeight) * stage2Height;
        
        // Clamp
        x1 = std::max(0.0f, std::min(x1, (float)stage2Width - 1.0f));
        y1 = std::max(0.0f, std::min(y1, (float)stage2Height - 1.0f));
        x2 = std::max(x1 + 1.0f, std::min(x2, (float)stage2Width));
        y2 = std::max(y1 + 1.0f, std::min(y2, (float)stage2Height));
        
        boxesData[j * 5 + 0] = 0.0f;  // batch_idx
        boxesData[j * 5 + 1] = x1;
        boxesData[j * 5 + 2] = y1;
        boxesData[j * 5 + 3] = x2;
        boxesData[j * 5 + 4] = y2;
    }
    
    // Copy boxes to GPU
    cudaMemcpyAsync(d_inputBoxes, boxesData.data(),
                    STAGE2_MAX_BOXES * 5 * sizeof(float),
                    cudaMemcpyHostToDevice, c.stage2Stream);
    
    // ========== 4. SYNCHRONIZE BEFORE INFERENCE ==========
    cudaStreamSynchronize(c.stage2Stream);
    
    // ========== 5. RUN INFERENCE WITH dwDNN_inferRaw ==========
    // Input pointers array
    const float* inputPtrs[2] = { d_inputImage, d_inputBoxes };
    
    // Output pointers array
    float* outputPtrs[4] = { d_outputDepth, d_outputDims, d_outputRotBins, d_outputRotRes };
    
    CHECK_DW_ERROR(dwDNN_inferRaw(
        outputPtrs,           // float* const* const dOutput
        inputPtrs,            // const float* const* const dInput
        1,                    // batchsize
        m_stage2Dnn[i]        // network
    ));
    
    // ========== 6. COPY OUTPUTS TO HOST ==========
    
    cudaMemcpyAsync(h_depth.data(), d_outputDepth, 
                    STAGE2_MAX_BOXES * 3 * sizeof(float),
                    cudaMemcpyDeviceToHost, c.stage2Stream);
    cudaMemcpyAsync(h_dims.data(), d_outputDims,
                    STAGE2_MAX_BOXES * 3 * sizeof(float),
                    cudaMemcpyDeviceToHost, c.stage2Stream);
    cudaMemcpyAsync(h_rotBins.data(), d_outputRotBins,
                    STAGE2_MAX_BOXES * 12 * sizeof(float),
                    cudaMemcpyDeviceToHost, c.stage2Stream);
    cudaMemcpyAsync(h_rotRes.data(), d_outputRotRes,
                    STAGE2_MAX_BOXES * 12 * sizeof(float),
                    cudaMemcpyDeviceToHost, c.stage2Stream);
    
    
    // Store host pointers for collectStage2Results to use
    c.stage2HostDepth = h_depth.data();
    c.stage2HostDims = h_dims.data();
    c.stage2HostRotBins = h_rotBins.data();
    c.stage2HostRotRes = h_rotRes.data();

    cudaEventRecord(c.stage2InferDone, c.stage2Stream);
}




void DriveSeg4CamServer::collectAndOverlay(uint32_t i)
{
    auto &c = m_dnnCtx[i];
    CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReceive(&c.segOutTensorHost, 1000, c.segOutStreamer));

    void* hptr = nullptr;
    CHECK_DW_ERROR(dwDNNTensor_lock(&hptr, c.segOutTensorHost));

    const int W = (int)m_outProps.dimensionSize[0];
    const int H = (int)m_outProps.dimensionSize[1];
    const int C = (int)m_outProps.dimensionSize[2];
    
    const float* h_logits = reinterpret_cast<const float*>(hptr);
    
    // GPU argmax
    float* d_logits = nullptr;
    uint8_t* d_classMap = nullptr;
    const size_t logitsBytes = W * H * C * sizeof(float);
    const size_t classMapBytes = W * H * sizeof(uint8_t);
    
    cudaMalloc(&d_logits, logitsBytes);
    cudaMalloc(&d_classMap, classMapBytes);
    
    cudaMemcpyAsync(d_logits, h_logits, logitsBytes, cudaMemcpyHostToDevice, c.segStream);
    
    dim3 argmaxBlock(16, 16);
    dim3 argmaxGrid((W + 15) / 16, (H + 15) / 16);
    
    argmaxNCHW_kernel<<<argmaxGrid, argmaxBlock, 0, c.segStream>>>(
        d_classMap, d_logits, W, H, C
    );
    
    CHECK_DW_ERROR(dwDNNTensor_unlock(c.segOutTensorHost));
    CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReturn(&c.segOutTensorHost, c.segOutStreamer));
    CHECK_DW_ERROR(dwDNNTensorStreamer_producerReturn(nullptr, 1000, c.segOutStreamer));
    
    // Colorize and blend
    static uint8_t h_colorLUT[3 * 4] = {
        0, 0, 0, 0,
        0, 255, 0, 180,
        255, 255, 0, 150
    };
    
    static uint8_t* d_colorLUT = nullptr;
    if (!d_colorLUT) {
        cudaMalloc(&d_colorLUT, sizeof(h_colorLUT));
        cudaMemcpy(d_colorLUT, h_colorLUT, sizeof(h_colorLUT), cudaMemcpyHostToDevice);
    }
    
    dwImageCUDA* cudaImg{};
    CHECK_DW_ERROR(dwImage_getCUDA(&cudaImg, m_imgRGBA[i]));
    uint8_t* d_rgba = reinterpret_cast<uint8_t*>(cudaImg->dptr[0]);
    
    const int imgW = cudaImg->prop.width;
    const int imgH = cudaImg->prop.height;
    const int imgStride = cudaImg->pitch[0];
    
    dim3 blendBlock(16, 16);
    dim3 blendGrid((imgW + 15) / 16, (imgH + 15) / 16);
    
    colorizeAndBlend_kernel<<<blendGrid, blendBlock, 0, c.segStream>>>(
        d_rgba, d_classMap,
        imgW, imgH, imgStride,
        W, H,
        d_colorLUT, kOverlayAlpha
    );
    
    cudaStreamSynchronize(c.segStream);
    
    cudaFree(d_logits);
    cudaFree(d_classMap);
}
void DriveSeg4CamServer::collectDetectionResults(uint32_t i)
{
    auto &c = m_dnnCtx[i];
    CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReceive(&c.detOutTensorHost, 1000, c.detOutStreamer));

    void* hptr = nullptr;
    CHECK_DW_ERROR(dwDNNTensor_lock(&hptr, c.detOutTensorHost));
    float32_t* confData = reinterpret_cast<float32_t*>(hptr);

    std::lock_guard<std::mutex> lock(c.detMutex);
    c.detBoxes.clear();
    c.detLabels.clear();
    c.detBoxes640.clear();  // NEW: boxes in 640×640 YOLO output space

    std::vector<YoloScoreRect> tmpRes;
    size_t offsetY, strideY, height;
    uint32_t indices[4] = {0, 0, 0, 0};
    CHECK_DW_ERROR(dwDNNTensor_getLayoutView(&offsetY, &strideY, &height, indices, 4U, 1U, c.detOutTensorHost));

    for (uint16_t row = 0; row < height; ++row) {
        const float32_t* detection = &confData[row * strideY];
        float conf = detection[4];
        
        if (conf < 0.45f) continue;
        
        uint16_t maxClass = 0;
        float maxScore = 0;
        for (uint16_t cls = 0; cls < 80; ++cls) {
            if (detection[5 + cls] > maxScore) {
                maxScore = detection[5 + cls];
                maxClass = cls;
            }
        }
        
        if (maxScore < 0.25f) continue;
        
        // Raw YOLO output (in 640×640 network output space)
        float32_t imageX = detection[0];
        float32_t imageY = detection[1];
        float32_t bboxW  = detection[2];
        float32_t bboxH  = detection[3];

        // Box in YOLO 640×640 output space (center format → corner format)
        float32_t boxX1_640 = imageX - 0.5f * bboxW;
        float32_t boxY1_640 = imageY - 0.5f * bboxH;
        float32_t boxX2_640 = imageX + 0.5f * bboxW;
        float32_t boxY2_640 = imageY + 0.5f * bboxH;

        // Transform to original image coordinates (for display/IPC)
        float32_t boxX1, boxY1, boxX2, boxY2;
        dwDataConditioner_outputPositionToInput(&boxX1, &boxY1, boxX1_640, boxY1_640, 
                                                &m_roi[i], c.detConditioner);
        dwDataConditioner_outputPositionToInput(&boxX2, &boxY2, boxX2_640, boxY2_640, 
                                                &m_roi[i], c.detConditioner);
        
        dwRectf bboxFloat{boxX1, boxY1, boxX2 - boxX1, boxY2 - boxY1};
        dwRectf bboxFloat640{boxX1_640, boxY1_640, boxX2_640 - boxX1_640, boxY2_640 - boxY1_640};
        
        tmpRes.push_back({bboxFloat, bboxFloat640, maxScore, maxClass});
    }

    CHECK_DW_ERROR(dwDNNTensor_unlock(c.detOutTensorHost));

    std::vector<YoloScoreRect> tmpResAfterNMS = doNmsForYoloOutputBoxes(tmpRes, 0.45f);
    
    for (uint32_t j = 0; j < tmpResAfterNMS.size(); j++) {
        YoloScoreRect box = tmpResAfterNMS[j];
        const std::string& className = YOLO_CLASS_NAMES[box.classIndex];
        if (m_automotiveClasses.find(className) != m_automotiveClasses.end()) {
            c.detBoxes.push_back(box.rectf);        // Original coords (for display)
            c.detBoxes640.push_back(box.rectf640);  // 640×640 coords (for Stage2)
            c.detLabels.push_back(className);
        }
    }

    CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReturn(&c.detOutTensorHost, c.detOutStreamer));
    CHECK_DW_ERROR(dwDNNTensorStreamer_producerReturn(nullptr, 1000, c.detOutStreamer));
}


void DriveSeg4CamServer::collectStage2Results(uint32_t i)
{
    auto &c = m_dnnCtx[i];
    
    // Data already on host from runStage2Inference
    float* depthData = c.stage2HostDepth;
    float* dimsData = c.stage2HostDims;
    float* rotBinsData = c.stage2HostRotBins;
    float* rotResData = c.stage2HostRotRes;
    
    if (!depthData || !dimsData || !rotBinsData || !rotResData) {
        std::cout << "[Stage2] ERROR: Host pointers not set for cam " << i << "\n";
        return;
    }
    
    std::lock_guard<std::mutex> lock(c.stage2Mutex);
    c.depths.clear();
    c.dimensions.clear();
    c.rotations.clear();
    c.iouScores.clear();
    
    size_t numActualBoxes = std::min<size_t>(c.detBoxes.size(), STAGE2_MAX_BOXES);
    
    const float PI = 3.14159265358979f;
    const float binSize = 2.0f * PI / 12.0f;
    
    for (size_t n = 0; n < numActualBoxes; ++n) {
        // DEPTH: depth + offset
        float depth = depthData[n * 3 + 0] + depthData[n * 3 + 2];
        
        if (depth <= 0.1f || depth > 200.0f) {
            c.depths.push_back(0.0f);
            c.dimensions.push_back(0.0f);
            c.dimensions.push_back(0.0f);
            c.dimensions.push_back(0.0f);
            c.rotations.push_back(0.0f);
            c.iouScores.push_back(0.0f);
            continue;
        }
        
        c.depths.push_back(depth);
        
        // DIMENSIONS
        c.dimensions.push_back(dimsData[n * 3 + 0]);  // h
        c.dimensions.push_back(dimsData[n * 3 + 1]);  // w
        c.dimensions.push_back(dimsData[n * 3 + 2]);  // l
        
        // ROTATION
        float maxVal = rotBinsData[n * 12];
        int maxBin = 0;
        for (int b = 1; b < 12; ++b) {
            if (rotBinsData[n * 12 + b] > maxVal) {
                maxVal = rotBinsData[n * 12 + b];
                maxBin = b;
            }
        }
        
        float binCenter = (maxBin + 0.5f) * binSize;
        float residual = rotResData[n * 12 + maxBin];
        float rotation = binCenter + residual;
        
        // Normalize to [-π, π]
        rotation = fmodf(rotation + PI, 2.0f * PI);
        if (rotation < 0) rotation += 2.0f * PI;
        rotation -= PI;
        
        c.rotations.push_back(rotation);
        c.iouScores.push_back(1.0f);
    }
    
    std::cout << "[Stage2] Cam " << i << " decoded " << c.depths.size() << " 3D boxes\n";
}

// ===============================================================
// FRAME EXTRACTION & IPC TRANSMISSION
// ===============================================================



SerializableFrame DriveSeg4CamServer::extractFrameData(uint32_t i)
{
    SerializableFrame frame;
    frame.cameraIndex = i;
    
    // Extract source image (existing code - unchanged)
    dwImageCUDA* cudaImg = nullptr;
    CHECK_DW_ERROR(dwImage_getCUDA(&cudaImg, m_imgRGBA[i]));
    
    const uint32_t srcWidth = cudaImg->prop.width;
    const uint32_t srcHeight = cudaImg->prop.height;
    const uint32_t dstWidth = 640;
    const uint32_t dstHeight = 640;
    
    frame.width = dstWidth;
    frame.height = dstHeight;
    frame.stride = dstWidth * 4;
    
    const size_t dstImageBytes = dstWidth * dstHeight * 4;
    frame.rgbaPixels.resize(dstImageBytes);
    
    // Downscale image (existing code - unchanged)
    uint8_t* d_downscaled = nullptr;
    cudaMalloc(&d_downscaled, dstImageBytes);
    
    const float scaleX = (float)srcWidth / dstWidth;
    const float scaleY = (float)srcHeight / dstHeight;
    
    dim3 block(16, 16);
    dim3 grid((dstWidth + 15) / 16, (dstHeight + 15) / 16);
    
    downscaleKernel<<<grid, block>>>(
        d_downscaled,
        reinterpret_cast<uint8_t*>(cudaImg->dptr[0]),
        dstWidth, dstHeight,
        srcWidth, srcHeight,
        cudaImg->pitch[0],
        scaleX, scaleY
    );
    
    cudaDeviceSynchronize();
    cudaMemcpy(frame.rgbaPixels.data(), d_downscaled, dstImageBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_downscaled);
    
    // Extract 2D detection data (existing code - unchanged)
    auto &c = m_dnnCtx[i];
    {
        std::lock_guard<std::mutex> lock(c.detMutex);
        frame.boxes.reserve(c.detBoxes.size());
        frame.labels = c.detLabels;
        
        const float coordScaleX = (float)dstWidth / srcWidth;
        const float coordScaleY = (float)dstHeight / srcHeight;
        
        for (const auto& box : c.detBoxes) {
            dwRectf scaledBox;
            scaledBox.x = box.x * coordScaleX;
            scaledBox.y = box.y * coordScaleY;
            scaledBox.width = box.width * coordScaleX;
            scaledBox.height = box.height * coordScaleY;
            frame.boxes.push_back(scaledBox);
        }
    }
    
    // Extract 3D predictions
    {
        std::lock_guard<std::mutex> lock(c.stage2Mutex);
        frame.depths = c.depths;
        frame.dimensions = c.dimensions;
        frame.rotations = c.rotations;
        frame.iouScores = c.iouScores;
    }
    
    // Extract stats
    frame.segCount = c.segCount;
    frame.detCount = c.detCount;
    frame.avgSegMs = c.avgSegMs;
    frame.avgDetMs = c.avgDetMs;
    frame.avgStage2Ms = c.avgStage2Ms;  
    frame.frameId = c.frameId;
    
    return frame;
}


bool DriveSeg4CamServer::sendFrameToClients(const SerializableFrame& frame)
{
    std::lock_guard<std::mutex> lock(m_socketMutex);
    
    if (!m_connections[frame.cameraIndex]) {
        std::cout << "[IPC] ERROR: No connection for camera " << frame.cameraIndex << "\n";
        return false;
    }
    
    // Prepare header
    FrameHeader header{};
    header.magic = 0xDEADBEEF;
    header.cameraIndex = frame.cameraIndex;
    header.frameId = frame.frameId;
    header.width = frame.width;
    header.height = frame.height;
    header.stride = frame.stride;
    header.imageDataSize = frame.rgbaPixels.size();
    header.numBoxes = frame.boxes.size();
    header.segCount = frame.segCount;
    header.detCount = frame.detCount;
    header.avgSegMs = frame.avgSegMs;
    header.avgDetMs = frame.avgDetMs;
    header.avgStage2Ms = frame.avgStage2Ms; 
    header.num3DBoxes = frame.depths.size();

    std::cout << "[IPC] Sending frame " << frame.frameId << " to cam " << frame.cameraIndex 
              << " | Size: " << frame.rgbaPixels.size() << " bytes | Boxes: " << frame.boxes.size() << "\n";
    
    // Send header
    size_t headerSize = sizeof(FrameHeader);
    size_t headerSent = headerSize;  // Track actual bytes sent
    
    dwStatus status = dwSocketConnection_write(
        reinterpret_cast<uint8_t*>(&header), 
        &headerSent,  // This gets updated with actual bytes written
        30000,
        m_connections[frame.cameraIndex]
    );
    
    if (status != DW_SUCCESS) {
        std::cout << "[IPC] ERROR: Header write failed with status: " << dwGetStatusName(status) 
                  << " (code " << status << ")"
                  << " | Expected: " << headerSize << " bytes"
                  << " | Actual: " << headerSent << " bytes\n";
        return false;
    }
    
    std::cout << "[IPC] Header sent successfully (" << headerSent << " bytes)\n";
    
    // Send image
    size_t imageSize = frame.rgbaPixels.size();
    size_t imageSent = imageSize;
    
    status = dwSocketConnection_write(
        const_cast<uint8_t*>(frame.rgbaPixels.data()),
        &imageSent,
        30000,
        m_connections[frame.cameraIndex]
    );
    
    if (status != DW_SUCCESS) {
        std::cout << "[IPC] ERROR: Image write failed with status: " << dwGetStatusName(status)
                  << " (code " << status << ")"
                  << " | Expected: " << imageSize << " bytes"
                  << " | Actual: " << imageSent << " bytes\n";
        return false;
    }
    
    std::cout << "[IPC] Image sent successfully (" << imageSent << " bytes)\n";
    
    // Send boxes (keep existing code but add logging on failure)
    for (size_t i = 0; i < frame.boxes.size(); ++i) {
        DetectionBox box{};
        box.x = frame.boxes[i].x;
        box.y = frame.boxes[i].y;
        box.width = frame.boxes[i].width;
        box.height = frame.boxes[i].height;
        strncpy(box.label, frame.labels[i].c_str(), sizeof(box.label) - 1);
        
        size_t boxSize = sizeof(DetectionBox);
        status = dwSocketConnection_write(
            reinterpret_cast<uint8_t*>(&box),
            &boxSize,
            30000,
            m_connections[frame.cameraIndex]
        );
        
        if (status != DW_SUCCESS) {
            std::cout << "[IPC] ERROR: Box " << i << " write failed: " << dwGetStatusName(status) << "\n";
            return false;
        }
    }
    
    for (size_t i = 0; i < frame.depths.size(); ++i) {
        Detection3DBox det3d{};
        det3d.depth = frame.depths[i];
        det3d.height = frame.dimensions[i * 3 + 0];
        det3d.width = frame.dimensions[i * 3 + 1];
        det3d.length = frame.dimensions[i * 3 + 2];
        det3d.rotation = frame.rotations[i];
        det3d.iouScore = frame.iouScores[i];
        
        size_t det3dSize = sizeof(Detection3DBox);
        status = dwSocketConnection_write(
            reinterpret_cast<uint8_t*>(&det3d), &det3dSize,
            30000, m_connections[frame.cameraIndex]);
        
        if (status != DW_SUCCESS) return false;
    }

    std::cout << "[IPC] Frame " << frame.frameId << " transmitted successfully\n";
    return true;
}


// ===============================================================
// UTILITIES
// ===============================================================

std::string DriveSeg4CamServer::platformPrefix()
{
    int32_t gpu{};
    dwGPUDeviceProperties prop{};
    CHECK_DW_ERROR(dwContext_getGPUDeviceCurrent(&gpu, m_ctx));
    CHECK_DW_ERROR(dwContext_getGPUProperties(&prop, gpu, m_ctx));

    if (prop.major >= 8) return "ampere-integrated";
    if (prop.major == 7) return prop.integrated ? "volta-integrated" : "turing";
    return "pascal";
}

void DriveSeg4CamServer::printStats()
{
    std::cout << "=== Server Stats ===\n";
    std::cout << "Frames: " << m_frameCounter.load() << "\n";
    std::cout << "Inferences/sec: " << (m_recentInferences / 2) << "\n";
    for (uint32_t i = 0; i < m_numCameras; ++i) {
        auto &c = m_dnnCtx[i];
        std::cout << "  Cam" << i 
                  << " | Det: " << c.detCount << " (" << c.avgDetMs << "ms)"
                  << " | Seg: " << c.segCount << " (" << c.avgSegMs << "ms)"
                  << " | 3D: " << c.stage2Count << " (" << c.avgStage2Ms << "ms)"
                  << " | Boxes: " << c.detBoxes.size() << "\n";
    }
}

bool DriveSeg4CamServer::sort_score(YoloScoreRect box1, YoloScoreRect box2)
{
    return box1.score > box2.score;
}

float32_t DriveSeg4CamServer::calculateIouOfBoxes(dwRectf box1, dwRectf box2)
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

std::vector<DriveSeg4CamServer::YoloScoreRect> DriveSeg4CamServer::doNmsForYoloOutputBoxes(
    std::vector<YoloScoreRect>& boxes, float32_t threshold)
{
    std::vector<YoloScoreRect> results;
    std::sort(boxes.begin(), boxes.end(), sort_score);
    
    while (boxes.size() > 0) {
        results.push_back(boxes[0]);
        uint32_t index = 1;
        while (index < boxes.size()) {
            float32_t iou_value = calculateIouOfBoxes(boxes[0].rectf, boxes[index].rectf);
            if (iou_value > threshold) {
                boxes.erase(boxes.begin() + index);
            } else {
                index++;
            }
        }
        boxes.erase(boxes.begin());
    }
    return results;
}

// ===============================================================
// SIGNAL HANDLER
// ===============================================================

static std::atomic<bool> g_run{true};

extern "C" void sig_int_handler(int)
{
    g_run = false;
    std::cout << "\n[Server] Shutdown signal received\n";
}


// ===============================================================
// MAIN
// ===============================================================

int main(int argc, const char** argv)
{
    ProgramArguments args(argc, argv, {
        ProgramArguments::Option_t(
            "rig",
            (dw_samples::SamplesDataPath::get() + "/samples/sensors/camera/camera/rig_4cam.json").c_str(),
            "Rig configuration (expects 4 cameras)"
        ),
        ProgramArguments::Option_t(
            "port",
            "49252",
            "Socket server port"
        ),
        ProgramArguments::Option_t(
            "stage2_model",
            "",
            "Path to Stage 2 (3D detection) TensorRT engine"
        ),
#ifdef VIBRANTE
        ProgramArguments::Option_t("cudla", "0", "Run DNN on CuDLA (1=yes)"),
        ProgramArguments::Option_t("dla-engine", "0", "DLA engine index"),
#endif
        ProgramArguments::Option_t(
            "tensorRT_model",
            "",
            "Path to segmentation TensorRT engine"
        ),
        ProgramArguments::Option_t(
            "detection_model",
            "",
            "Path to detection TensorRT engine"
        ),
    },
    "Four-camera perception server with socket IPC");

    // Setup signal handlers
    std::signal(SIGHUP, sig_int_handler);
    std::signal(SIGINT, sig_int_handler);
    std::signal(SIGQUIT, sig_int_handler);
    std::signal(SIGABRT, sig_int_handler);
    std::signal(SIGTERM, sig_int_handler);

    try {
        DriveSeg4CamServer server(args);
        
        if (!server.initialize()) {
            std::cerr << "[Server] Initialization failed\n";
            return -1;
        }
        
        std::cout << "[Server] Starting main loop (Ctrl+C to stop)...\n";
        
    
        while (g_run) {
            server.run();  // Processes one frame
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        std::cout << "[Server] Received shutdown signal...\n";
        server.release();
        
    } catch (const std::exception& e) {
        std::cerr << "[Server] Exception: " << e.what() << "\n";
        return -1;
    }

    std::cout << "[Server] Shutdown complete\n";
    return 0;
}
