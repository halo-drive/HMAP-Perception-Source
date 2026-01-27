/*
 * CenterPointDW.hpp
 *
 * Lightweight wrapper to run the CenterPoint TensorRT pipeline (PFE + RPN)
 * directly on a DriveWorks stitched point cloud.
 *
 * This class reuses the original CenterPoint TensorRT implementation from
 * the standalone CenterPoint project but exposes a simple API suitable for
 * DriveWorks samples:
 *
 *   - Initialize from two serialized engine files (PFE, RPN)
 *   - For each frame, take a host-side dwPointCloud (XYZI) and return
 *     a list of InterLidarICP::BoundingBox objects in rig coordinates.
 */

 #ifndef CENTERPOINT_DW_HPP
 #define CENTERPOINT_DW_HPP
 
 #include <memory>
 #include <string>
 #include <vector>
 #include <map>
 
 #include <cuda_runtime_api.h>
 
 #include <dw/pointcloudprocessing/pointcloud/PointCloud.h>
 
 // TensorRT includes
 #include <NvInfer.h>
 #include <NvInferRuntime.h>
 
 // TensorRT sample utilities (from CenterPoint/include/common/) - include first
 // so that postprocess.h can find them when it includes "buffers.h" and "common.h"
 #include "common/buffers.h"
 #include "common/common.h"
 #include "common/logger.h"
 
 // CenterPoint headers - avoid centerpoint.h to prevent pulling in TensorRT sample dependencies
 #include "config.h"        // CenterPoint configuration constants
 #include "preprocess.h"    // preprocessGPU function
 #include "postprocess.h"   // Box struct and postprocessGPU function (includes buffers.h, common.h)
 #include "scatter_cuda.h"  // ScatterCuda class
 
 // Forward-declare InterLidarICP::BoundingBox to avoid cyclic include
 class InterLidarICP;
 
 class CenterPointDW
 {
     template <typename T>
     using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
 
 public:
     struct Config
     {
         std::string pfeEnginePath;
         std::string rpnEnginePath;
         bool fp16{false};
         int dlaCore{-1};
     };
 
     // BoundingBox is defined inside InterLidarICP; we just forward-declare here
     struct DWBoundingBox
     {
         float x, y, z;
         float width, length, height;
         float rotation;
         float confidence;
         int classId;
     };
 
     CenterPointDW(const Config& cfg);
     ~CenterPointDW();
 
     bool isInitialized() const { return m_initialized; }
 
     // Run CenterPoint on a stitched host-side point cloud (XYZI) and append
     // detections into outBoxes (DriveWorks-format bounding boxes).
     bool inferOnPointCloud(const dwPointCloud& stitchedHost,
                            std::vector<DWBoundingBox>& outBoxes);
 
     // Get BEV feature map for visualization/sharing with other models
     // Returns host-side copy of BEV features (C x H x W = PFE_OUTPUT_DIM x BEV_H x BEV_W)
     // Caller must allocate outBEV with size PFE_OUTPUT_DIM * BEV_H * BEV_W
     bool getBEVFeatureMap(std::vector<float>& outBEV);
 
     // Get heatmap (score map) for visualization/sharing with other models
     // Returns host-side copy of score heatmap (H x W = OUTPUT_H x OUTPUT_W)
     // Caller must allocate outHeatmap with size OUTPUT_H * OUTPUT_W
     bool getHeatmap(std::vector<float>& outHeatmap);
 
 private:
     bool initializeEngines();
     bool allocateBuffers();
 
     // Map CenterPoint Box to DriveWorks-style bounding box
     static DWBoundingBox convertBox(const Box& b);
 
 private:
     Config m_cfg;
     bool m_initialized{false};
 
     // TensorRT engines and execution contexts
     std::shared_ptr<nvinfer1::ICudaEngine> m_pfeEngine;
     std::shared_ptr<nvinfer1::ICudaEngine> m_rpnEngine;
     SampleUniquePtr<nvinfer1::IExecutionContext> m_pfeContext;
     SampleUniquePtr<nvinfer1::IExecutionContext> m_rpnContext;
 
     // Buffer managers for PFE and RPN engines
     std::unique_ptr<samplesCommon::BufferManager> m_pfeBuffers;
     std::unique_ptr<samplesCommon::BufferManager> m_rpnBuffers;
 
     // Minimal Params structure (replaces centerpoint.h Params to avoid dependencies)
     struct Params {
         std::string pfeSerializedEnginePath;
         std::string rpnSerializedEnginePath;
         std::vector<std::string> pfeInputTensorNames;
         std::vector<std::string> rpnInputTensorNames;
         std::vector<std::string> pfeOutputTensorNames;
         std::map<std::string, std::vector<std::string>> rpnOutputTensorNames;
         bool fp16{false};
         int dlaCore{-1};
         int batch_size{1};
     };
     Params m_params{};
 
     // Device pointers for CenterPoint preprocess / scatter
     float* m_devPoints{nullptr};          // MAX_POINTS x POINT_DIM
     int* m_devIndices{nullptr};           // MAX_PILLARS
     int* m_devScoreIdx{nullptr};          // OUTPUT_W * OUTPUT_H
     SampleUniquePtr<ScatterCuda> m_scatterCuda;
 
     // Preprocess helpers
     int* m_pBevIdx{nullptr};
     int* m_pPointNumAssigned{nullptr};
     bool* m_pMask{nullptr};
     int* m_bevVoxelIdx{nullptr};
     float* m_vPointSum{nullptr};
     int* m_vRange{nullptr};
     int* m_vPointNum{nullptr};
 
     // Postprocess host/device helper buffers
     unsigned long long* m_maskCpu{nullptr};
     unsigned long long* m_remvCpu{nullptr};
     int* m_hostScoreIdx{nullptr};
     long* m_hostKeepData{nullptr};
     float* m_hostBoxes{nullptr};
     int* m_hostLabel{nullptr};
 };
 
 #endif // CENTERPOINT_DW_HPP
 
 
 