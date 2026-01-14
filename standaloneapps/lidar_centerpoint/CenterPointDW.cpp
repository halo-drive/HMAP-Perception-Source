#include "CenterPointDW.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <cuda_runtime_api.h>

#include <dw/pointcloudprocessing/pointcloud/PointCloud.h>

// Bring TensorRT samples alias into scope
using samplesCommon::SampleUniquePtr;

// Utility macro (same as CenterPoint's DIVUP)
#ifndef DIVUP
#define DIVUP(m, n) (((m) / (n)) + (((m) % (n)) > 0))
#endif

// -------------------------------------------------------------------------------------------------
// Helper to load a serialized TensorRT engine from file (adapted from CenterPoint::buildFromSerializedEngine)
// -------------------------------------------------------------------------------------------------
static std::shared_ptr<nvinfer1::ICudaEngine> loadSerializedEngine(const std::string& serializedEngineFile)
{
    std::vector<char> trtModelStream;
    size_t size{0};

    std::ifstream file(serializedEngineFile, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = static_cast<size_t>(file.tellg());
        file.seekg(0, file.beg);
        trtModelStream.resize(size);
        file.read(trtModelStream.data(), size);
        file.close();
    }
    else
    {
        sample::gLogError << "Failed to read serialized engine: " << serializedEngineFile << std::endl;
        return nullptr;
    }

    auto runtime = SampleUniquePtr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!runtime)
    {
        sample::gLogError << "Failed to create TensorRT runtime" << std::endl;
        return nullptr;
    }

    std::shared_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(trtModelStream.data(), size),
        samplesCommon::InferDeleter());

    if (!engine)
    {
        sample::gLogError << "Failed to deserialize TensorRT engine: " << serializedEngineFile << std::endl;
        return nullptr;
    }

    return engine;
}

// -------------------------------------------------------------------------------------------------
// CenterPointDW implementation
// -------------------------------------------------------------------------------------------------

CenterPointDW::CenterPointDW(const Config& cfg)
    : m_cfg(cfg)
{
    // Fill Params similar to samplecenterpoint.cpp
    m_params.pfeSerializedEnginePath = m_cfg.pfeEnginePath;
    m_params.rpnSerializedEnginePath = m_cfg.rpnEnginePath;
    m_params.fp16 = m_cfg.fp16;
    m_params.dlaCore = m_cfg.dlaCore;
    m_params.batch_size = 1;

    // Tensor names follow the Waymo CenterPoint config from samplecenterpoint.cpp
    m_params.pfeInputTensorNames.push_back("input.1");
    m_params.rpnInputTensorNames.push_back("input.1");
    m_params.pfeOutputTensorNames.push_back("47");

    m_params.rpnOutputTensorNames["regName"]     = {"246"};
    m_params.rpnOutputTensorNames["rotName"]     = {"258"};
    m_params.rpnOutputTensorNames["heightName"]  = {"250"};
    m_params.rpnOutputTensorNames["dimName"]     = {"264"};
    m_params.rpnOutputTensorNames["scoreName"]   = {"265"};
    m_params.rpnOutputTensorNames["clsName"]     = {"266"};

    // Initialize engines and buffers
    if (!initializeEngines())
    {
        sample::gLogError << "CenterPointDW: failed to initialize engines" << std::endl;
        return;
    }
    if (!allocateBuffers())
    {
        sample::gLogError << "CenterPointDW: failed to allocate buffers" << std::endl;
        return;
    }

    // Scatter module (BEV feature scatter)
    m_scatterCuda = SampleUniquePtr<ScatterCuda>(
        new ScatterCuda(PFE_OUTPUT_DIM, PFE_OUTPUT_DIM, BEV_W, BEV_H));

    m_initialized = true;
}

CenterPointDW::~CenterPointDW()
{
    if (m_devPoints)
        cudaFree(m_devPoints);
    if (m_devIndices)
        cudaFree(m_devIndices);
    if (m_devScoreIdx)
        cudaFree(m_devScoreIdx);

    if (m_pBevIdx)
        cudaFree(m_pBevIdx);
    if (m_pPointNumAssigned)
        cudaFree(m_pPointNumAssigned);
    if (m_pMask)
        cudaFree(m_pMask);
    if (m_bevVoxelIdx)
        cudaFree(m_bevVoxelIdx);
    if (m_vPointSum)
        cudaFree(m_vPointSum);
    if (m_vRange)
        cudaFree(m_vRange);
    if (m_vPointNum)
        cudaFree(m_vPointNum);

    if (m_maskCpu)
        cudaFreeHost(m_maskCpu);
    if (m_remvCpu)
        cudaFreeHost(m_remvCpu);
    if (m_hostScoreIdx)
        cudaFreeHost(m_hostScoreIdx);
    if (m_hostKeepData)
        cudaFreeHost(m_hostKeepData);
    if (m_hostBoxes)
        cudaFreeHost(m_hostBoxes);
    if (m_hostLabel)
        cudaFreeHost(m_hostLabel);
}

bool CenterPointDW::initializeEngines()
{
    if (m_cfg.pfeEnginePath.empty() || m_cfg.rpnEnginePath.empty())
    {
        sample::gLogError << "CenterPointDW: PFE / RPN engine paths must be provided" << std::endl;
        return false;
    }

    sample::gLogInfo << "CenterPointDW: loading PFE engine from " << m_cfg.pfeEnginePath << std::endl;
    m_pfeEngine = loadSerializedEngine(m_cfg.pfeEnginePath);
    if (!m_pfeEngine)
        return false;

    sample::gLogInfo << "CenterPointDW: loading RPN engine from " << m_cfg.rpnEnginePath << std::endl;
    m_rpnEngine = loadSerializedEngine(m_cfg.rpnEnginePath);
    if (!m_rpnEngine)
        return false;

    // Create BufferManagers and execution contexts (similar to CenterPoint::infer())
    m_pfeBuffers.reset(new samplesCommon::BufferManager(m_pfeEngine));
    m_rpnBuffers.reset(new samplesCommon::BufferManager(m_rpnEngine));

    m_pfeContext = SampleUniquePtr<nvinfer1::IExecutionContext>(m_pfeEngine->createExecutionContext());
    m_rpnContext = SampleUniquePtr<nvinfer1::IExecutionContext>(m_rpnEngine->createExecutionContext());

    if (!m_pfeContext || !m_rpnContext)
    {
        sample::gLogError << "CenterPointDW: failed to create execution contexts" << std::endl;
        return false;
    }

    return true;
}

bool CenterPointDW::allocateBuffers()
{
    // Device buffers for raw points and indices
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_devPoints), MAX_POINTS * POINT_DIM * sizeof(float)));
    GPU_CHECK(cudaMemset(m_devPoints, 0, MAX_POINTS * POINT_DIM * sizeof(float)));

    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_devIndices), MAX_PILLARS * sizeof(int)));
    GPU_CHECK(cudaMemset(m_devIndices, 0, MAX_PILLARS * sizeof(int)));

    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_devScoreIdx), OUTPUT_W * OUTPUT_H * sizeof(int)));
    GPU_CHECK(cudaMemset(m_devScoreIdx, -1, OUTPUT_W * OUTPUT_H * sizeof(int)));

    // Preprocess helpers
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_pBevIdx), MAX_POINTS * sizeof(int)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_pPointNumAssigned), MAX_POINTS * sizeof(int)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_pMask), MAX_POINTS * sizeof(bool)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_bevVoxelIdx), BEV_H * BEV_W * sizeof(int)));

    GPU_CHECK(cudaMemset(m_pBevIdx, 0, MAX_POINTS * sizeof(int)));
    GPU_CHECK(cudaMemset(m_pPointNumAssigned, 0, MAX_POINTS * sizeof(int)));
    GPU_CHECK(cudaMemset(m_pMask, 0, MAX_POINTS * sizeof(bool)));
    GPU_CHECK(cudaMemset(m_bevVoxelIdx, 0, BEV_H * BEV_W * sizeof(int)));

    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_vPointSum), MAX_PILLARS * 3 * sizeof(float)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_vRange), MAX_PILLARS * sizeof(int)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_vPointNum), MAX_PILLARS * sizeof(int)));

    GPU_CHECK(cudaMemset(m_vRange, 0, MAX_PILLARS * sizeof(int)));
    GPU_CHECK(cudaMemset(m_vPointSum, 0, MAX_PILLARS * 3 * sizeof(float)));
    GPU_CHECK(cudaMemset(m_vPointNum, 0, MAX_PILLARS * sizeof(int)));

    // Postprocess host buffers (pinned)
    GPU_CHECK(cudaMallocHost(reinterpret_cast<void**>(&m_maskCpu),
                             INPUT_NMS_MAX_SIZE * DIVUP(INPUT_NMS_MAX_SIZE, THREADS_PER_BLOCK_NMS) *
                                 sizeof(unsigned long long)));
    GPU_CHECK(cudaMemset(m_maskCpu, 0, INPUT_NMS_MAX_SIZE * DIVUP(INPUT_NMS_MAX_SIZE, THREADS_PER_BLOCK_NMS) *
                                       sizeof(unsigned long long)));

    GPU_CHECK(cudaMallocHost(reinterpret_cast<void**>(&m_remvCpu),
                             THREADS_PER_BLOCK_NMS * sizeof(unsigned long long)));
    GPU_CHECK(cudaMemset(m_remvCpu, 0, THREADS_PER_BLOCK_NMS * sizeof(unsigned long long)));

    GPU_CHECK(cudaMallocHost(reinterpret_cast<void**>(&m_hostScoreIdx),
                             OUTPUT_W * OUTPUT_H * sizeof(int)));
    GPU_CHECK(cudaMemset(m_hostScoreIdx, -1, OUTPUT_W * OUTPUT_H * sizeof(int)));

    GPU_CHECK(cudaMallocHost(reinterpret_cast<void**>(&m_hostKeepData),
                             INPUT_NMS_MAX_SIZE * sizeof(long)));
    GPU_CHECK(cudaMemset(m_hostKeepData, -1, INPUT_NMS_MAX_SIZE * sizeof(long)));

    GPU_CHECK(cudaMallocHost(reinterpret_cast<void**>(&m_hostBoxes),
                             OUTPUT_NMS_MAX_SIZE * 9 * sizeof(float)));
    GPU_CHECK(cudaMemset(m_hostBoxes, 0, OUTPUT_NMS_MAX_SIZE * 9 * sizeof(float)));

    GPU_CHECK(cudaMallocHost(reinterpret_cast<void**>(&m_hostLabel),
                             OUTPUT_NMS_MAX_SIZE * sizeof(int)));
    GPU_CHECK(cudaMemset(m_hostLabel, -1, OUTPUT_NMS_MAX_SIZE * sizeof(int)));

    return true;
}

CenterPointDW::DWBoundingBox CenterPointDW::convertBox(const Box& b)
{
    DWBoundingBox out{};
    out.x = b.x;
    out.y = b.y;
    out.z = b.z;

    // CenterPoint uses (l: length along x, w: width along y, h: height).
    // InterLidarICP::BoundingBox uses (width, length, height).
    out.length = b.l;
    out.height = b.h;
    out.width = b.w;

    out.rotation = b.theta;
    out.confidence = b.score;

    // Keep class id as-is; higher-level filters can remap if needed.
    out.classId = b.cls;
    return out;
}

bool CenterPointDW::inferOnPointCloud(const dwPointCloud& stitchedHost,
                                      std::vector<DWBoundingBox>& outBoxes)
{
    if (!m_initialized)
    {
        return false;
    }

    if (stitchedHost.type != DW_MEMORY_TYPE_CPU ||
        stitchedHost.format != DW_POINTCLOUD_FORMAT_XYZI ||
        stitchedHost.points == nullptr ||
        stitchedHost.size == 0)
    {
        sample::gLogError << "CenterPointDW::inferOnPointCloud expects host XYZI point cloud" << std::endl;
        return false;
    }

    // Clamp point count to MAX_POINTS
    uint32_t pointNum = std::min<uint32_t>(stitchedHost.size, MAX_POINTS);
    const dwVector4f* pts = static_cast<const dwVector4f*>(stitchedHost.points);

    // Pack DriveWorks XYZI into CenterPoint's 5D layout [x,y,z,intensity,time_lag]
    std::vector<float> hostPoints(pointNum * POINT_DIM);
    for (uint32_t i = 0; i < pointNum; ++i)
    {
        const dwVector4f& p = pts[i];
        hostPoints[i * POINT_DIM + 0] = p.x;
        hostPoints[i * POINT_DIM + 1] = p.y;
        hostPoints[i * POINT_DIM + 2] = p.z;
        hostPoints[i * POINT_DIM + 3] = p.w;     // intensity
        hostPoints[i * POINT_DIM + 4] = 0.0f;    // time_lag (not available -> 0)
    }

    // Copy to device
    GPU_CHECK(cudaMemcpy(m_devPoints, hostPoints.data(),
                         pointNum * POINT_DIM * sizeof(float),
                         cudaMemcpyHostToDevice));

    // PFE input tensor
    float* devicePillars =
        static_cast<float*>(m_pfeBuffers->getDeviceBuffer(m_params.pfeInputTensorNames[0]));

    // Zero PFE input and derived feature grid
    GPU_CHECK(cudaMemset(devicePillars, 0,
                         MAX_PILLARS * MAX_PIONT_IN_PILLARS * FEATURE_NUM * sizeof(float)));

    // GPU preprocess: point cloud -> pillar features (voxels)
    preprocessGPU(m_devPoints,
                  devicePillars,
                  m_devIndices,
                  m_pMask,
                  m_pBevIdx,
                  m_pPointNumAssigned,
                  m_bevVoxelIdx,
                  m_vPointSum,
                  m_vRange,
                  m_vPointNum,
                  static_cast<int>(pointNum),
                  POINT_DIM);

    // Run PFE
    bool status = m_pfeContext->executeV2(m_pfeBuffers->getDeviceBindings().data());
    if (!status)
    {
        sample::gLogError << "CenterPointDW: PFE execution failed" << std::endl;
        return false;
    }

    // Scatter pillar features into BEV tensor for RPN
    float* devScatteredFeature =
        static_cast<float*>(m_rpnBuffers->getDeviceBuffer(m_params.rpnInputTensorNames[0]));
    GPU_CHECK(cudaMemset(devScatteredFeature, 0,
                         PFE_OUTPUT_DIM * BEV_W * BEV_H * sizeof(float)));

    float* pfeOutput =
        static_cast<float*>(m_pfeBuffers->getDeviceBuffer(m_params.pfeOutputTensorNames[0]));

    m_scatterCuda->doScatterCuda(MAX_PILLARS,
                                 m_devIndices,
                                 pfeOutput,
                                 devScatteredFeature);

    // Run RPN
    status = m_rpnContext->executeV2(m_rpnBuffers->getDeviceBindings().data());
    if (!status)
    {
        sample::gLogError << "CenterPointDW: RPN execution failed" << std::endl;
        return false;
    }

    // Postprocess on GPU (NMS + box decoding) and gather results on host
    std::vector<Box> centerBoxes;
    postprocessGPU(*m_rpnBuffers,
                   centerBoxes,
                   m_params.rpnOutputTensorNames,
                   m_devScoreIdx,
                   m_maskCpu,
                   m_remvCpu,
                   m_hostScoreIdx,
                   m_hostKeepData,
                   m_hostBoxes,
                   m_hostLabel);

    // Convert to DriveWorks style bounding boxes and append
    for (const Box& b : centerBoxes)
    {
        outBoxes.push_back(convertBox(b));
    }

    return true;
}


