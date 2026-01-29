/////////////////////////////////////////////////////////////////////////////////////////
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA CORPORATION & AFFILIATES assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA CORPORATION & AFFILIATES. No third party distribution is allowed unless
// expressly authorized by NVIDIA. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA CORPORATION & AFFILIATES products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA CORPORATION & AFFILIATES.
//
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>

#include <dw/pointcloudprocessing/icp/PointCloudICP.h>
#include <dw/pointcloudprocessing/pointcloud/PointCloud.h>
#include <dw/pointcloudprocessing/stitcher/PointCloudStitcher.h>
#include <dw/rig/Rig.h>

#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Visualization.h>

#include <framework/Checks.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/Log.hpp>
#include <framework/MathUtils.hpp>
#include <framework/MouseView3D.hpp>
#include <framework/ProgramArguments.hpp>

using namespace dw_samples::common;

//------------------------------------------------------------------------------
// Offline visualizer for two recorded lidar streams.
// - Reads .bin point clouds (XYZI floats) from two folders (basic + hr)
// - Applies rig transformations from a DriveWorks rig.json
// - Performs ICP correction for fine alignment refinement
// - Visualizes both clouds overlaid with different colors
// - Iterates until one of the streams reaches its end
// - Prints start/end timestamps for both folders
//
// Processing pipeline:
//   1. Load point clouds from .bin files
//   2. Apply rig transformations (sensor-to-rig from rig.json)
//   3. Perform ICP alignment refinement (corrects for rig calibration errors)
//   4. Apply ICP transform to HR lidar
//   5. Visualize synchronized and aligned point clouds
//------------------------------------------------------------------------------
class RecordingVisualizerSample : public DriveWorksSample
{
private:
    // DriveWorks context
    dwContextHandle_t m_context = DW_NULL_HANDLE;

    // Visualization
    dwVisualizationContextHandle_t m_viz = DW_NULL_HANDLE;
    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;

    // Rig
    dwRigHandle_t m_rig = DW_NULL_HANDLE;
    dwTransformation3f m_frontSensorToRig = DW_IDENTITY_TRANSFORMATION3F;
    dwTransformation3f m_rearSensorToRig  = DW_IDENTITY_TRANSFORMATION3F;
    
    // ICP for alignment refinement
    dwPointCloudICPHandle_t m_icp = DW_NULL_HANDLE;
    dwTransformation3f m_icpTransform = DW_IDENTITY_TRANSFORMATION3F;
    bool m_icpInitialized = false;
    uint32_t m_icpInitFrames = 0;
    static constexpr uint32_t ICP_INIT_FRAMES = 3;  // Wait a few frames before starting ICP
    bool m_lastICPSuccessful = false;
    uint32_t m_icpMaxPoints = 0;  // Calculated from actual point cloud size
    dwVector2ui m_icpDepthmapSize = {0, 0};  // Calculated from actual point cloud size
    bool m_enableICP = true;  // Set to false to disable ICP (use rig transforms only)
    
    // Point cloud stitchers for transformations (preserves organized structure)
    dwPointCloudStitcherHandle_t m_basicRigTransformer = DW_NULL_HANDLE;
    dwPointCloudStitcherHandle_t m_hrRigTransformer = DW_NULL_HANDLE;
    dwPointCloudStitcherHandle_t m_icpTransformer = DW_NULL_HANDLE;

    // File lists
    std::string m_basicDir;
    std::string m_hrDir;
    std::vector<std::string> m_basicFiles;
    std::vector<std::string> m_hrFiles;
    size_t m_currentBasicIndex = 0;
    size_t m_currentHrIndex = 0;  // Track HR index separately for timestamp matching
    static constexpr int64_t MAX_TIMESTAMP_DIFF_US = 100000;  // 100ms max difference for pairing
    int64_t m_timestampOffset = 0;  // Offset between HR and Basic start times

    // Point cloud buffers (CPU)
    std::vector<dwVector4f> m_basicPoints;
    std::vector<dwVector4f> m_hrPoints;
    
    // Point cloud buffers (CUDA)
    dwPointCloud m_basicInput;           // Basic lidar input (before rig transform)
    dwPointCloud m_hrInput;              // HR lidar input (before rig transform)
    dwPointCloud m_basicRigTransformed;  // Basic after rig transform
    dwPointCloud m_hrRigTransformed;     // HR after rig transform
    dwPointCloud m_hrICPTransformed;     // HR after ICP transform

    // Render buffers
    uint32_t m_basicBufferId = 0;
    uint32_t m_hrBufferId    = 0;
    uint32_t m_basicPointCount = 0;
    uint32_t m_hrPointCount    = 0;
    
    // Timing control for real-time playback
    int64_t m_lastFrameTimestamp = 0;
    std::chrono::steady_clock::time_point m_lastFrameTime;
    bool m_firstFrame = true;

public:
    RecordingVisualizerSample(const ProgramArguments& args)
        : DriveWorksSample(args)
    {}

    //------------------------------------------------------------------------------
    // Utility: list .bin files in directory, sorted
    //------------------------------------------------------------------------------
    static std::vector<std::string> listBinFiles(const std::string& dir)
    {
        std::vector<std::string> files;

        DIR* dp = opendir(dir.c_str());
        if (!dp)
        {
            std::cerr << "Could not open directory: " << dir << std::endl;
            return files;
        }

        struct dirent* ep;
        while ((ep = readdir(dp)) != nullptr)
        {
            if (ep->d_type == DT_REG)
            {
                std::string name(ep->d_name);
                if (name.size() > 4 && name.substr(name.size() - 4) == ".bin")
                {
                    files.push_back(name);
                }
            }
        }
        closedir(dp);

        std::sort(files.begin(), files.end());
        return files;
    }

    //------------------------------------------------------------------------------
    // Utility: extract timestamp (us) from filename "<timestamp>.bin"
    //------------------------------------------------------------------------------
    static bool filenameToTimestamp(const std::string& name, int64_t& ts)
    {
        if (name.size() <= 4 || name.substr(name.size() - 4) != ".bin")
            return false;
        std::string base = name.substr(0, name.size() - 4);
        try
        {
            ts = std::stoll(base);
            return true;
        }
        catch (...)
        {
            return false;
        }
    }


    //------------------------------------------------------------------------------
    void initializeDriveWorks()
    {
        CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
        CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_INFO));

        dwContextParameters params{};
        CHECK_DW_ERROR(dwInitialize(&m_context, DW_VERSION, &params));

        CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_context));

        // Render engine
        dwRenderEngineParams renderParams{};
        CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&renderParams,
                                                        getWindowWidth(),
                                                        getWindowHeight()));
        renderParams.defaultTile.backgroundColor = {0.f, 0.f, 0.f, 1.f};
        CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine,
                                                 &renderParams,
                                                 m_viz));
    }

    //------------------------------------------------------------------------------
    bool loadRig(const std::string& rigFile)
    {
        CHECK_DW_ERROR_MSG(dwRig_initializeFromFile(&m_rig, m_context, rigFile.c_str()),
                           "Could not initialize Rig from file");

        uint32_t sensorCount = 0;
        CHECK_DW_ERROR(dwRig_getSensorCount(&sensorCount, m_rig));
        if (sensorCount < 2)
        {
            logError("Rig file must contain at least 2 lidar sensors");
            return false;
        }

        // For simplicity, assume sensor 0 is front, 1 is rear (matches rigFile/rig.json).
        // If you need name-based lookup, dwRig_getSensorName() can be used here.
        uint32_t frontIndex = 0;
        uint32_t rearIndex  = 1;

        CHECK_DW_ERROR(dwRig_getSensorToRigTransformation(&m_frontSensorToRig,
                                                          frontIndex,
                                                          m_rig));
        CHECK_DW_ERROR(dwRig_getSensorToRigTransformation(&m_rearSensorToRig,
                                                          rearIndex,
                                                          m_rig));

        return true;
    }

    //------------------------------------------------------------------------------
    bool loadPointCloud(const std::string& fullPath, std::vector<dwVector4f>& outPoints)
    {
        std::ifstream fin(fullPath, std::ios::binary | std::ios::ate);
        if (!fin.is_open())
        {
            logWarn("Failed to open point cloud file: %s", fullPath.c_str());
            return false;
        }

        std::streamsize size = fin.tellg();
        fin.seekg(0, std::ios::beg);

        if (size <= 0 || size % static_cast<std::streamsize>(sizeof(dwVector4f)) != 0)
        {
            logWarn("Invalid point cloud file size: %s", fullPath.c_str());
            return false;
        }

        size_t numPoints = static_cast<size_t>(size / sizeof(dwVector4f));
        outPoints.resize(numPoints);

        if (!fin.read(reinterpret_cast<char*>(outPoints.data()), size))
        {
            logWarn("Failed to read point cloud data: %s", fullPath.c_str());
            return false;
        }

        return true;
    }

    //------------------------------------------------------------------------------
    bool onInitialize() override
    {
        std::string rigFile   = getArgument("rig-file");
        m_basicDir            = getArgument("basic-dir");
        m_hrDir               = getArgument("hr-dir");
        
        // Optional: enable/disable ICP (default: enabled)
        std::string enableICPStr = getArgument("enable-icp");
        m_enableICP = (enableICPStr == "true" || enableICPStr == "1" || enableICPStr == "yes");

        if (rigFile.empty() || m_basicDir.empty() || m_hrDir.empty())
        {
            logError("Missing required arguments. Use --rig-file, --basic-dir and --hr-dir.");
            return false;
        }

        // Normalize directory paths: remove trailing '/'
        if (!m_basicDir.empty() && m_basicDir.back() == '/')
            m_basicDir.pop_back();
        if (!m_hrDir.empty() && m_hrDir.back() == '/')
            m_hrDir.pop_back();

        initializeDriveWorks();

        if (!loadRig(rigFile))
        {
            return false;
        }

        // List files
        m_basicFiles = listBinFiles(m_basicDir);
        m_hrFiles    = listBinFiles(m_hrDir);

        if (m_basicFiles.empty() || m_hrFiles.empty())
        {
            logError("No .bin files found in basic-dir or hr-dir");
            return false;
        }

        // Print start/end timestamps
        int64_t basicStartTs = 0, basicEndTs = 0;
        int64_t hrStartTs    = 0, hrEndTs    = 0;

        filenameToTimestamp(m_basicFiles.front(), basicStartTs);
        filenameToTimestamp(m_basicFiles.back(),  basicEndTs);
        filenameToTimestamp(m_hrFiles.front(),    hrStartTs);
        filenameToTimestamp(m_hrFiles.back(),     hrEndTs);

        std::cout << "Basic lidar: start=" << basicStartTs
                  << " end=" << basicEndTs << std::endl;
        std::cout << "HR lidar:    start=" << hrStartTs
                  << " end=" << hrEndTs << std::endl;

        double basicStartSec = static_cast<double>(basicStartTs) * 1e-6;
        double hrStartSec    = static_cast<double>(hrStartTs) * 1e-6;
        m_timestampOffset = hrStartTs - basicStartTs;  // Store offset for pairing
        std::cout << "Start time difference (HR - Basic) [s]: "
                  << (hrStartSec - basicStartSec) << std::endl;
        std::cout << "Timestamp offset (us): " << m_timestampOffset << std::endl;

        // Initialize render buffers
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_basicBufferId,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                   sizeof(dwVector3f),
                                                   0,
                                                   1000000, // capacity hint, will be resized as needed
                                                   m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_createBuffer(&m_hrBufferId,
                                                   DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                   sizeof(dwVector3f),
                                                   0,
                                                   1000000,
                                                   m_renderEngine));

        // Initialize stitchers (ICP will be initialized after first point cloud is loaded)
        initStitchers();

        return true;
    }
    
    //------------------------------------------------------------------------------
    void initICP()
    {
        // ICP will be initialized after first point cloud is loaded
        // to calculate correct depth map size from actual data (like InterLidarICP)
        log("ICP initialization deferred - will calculate from first point cloud");
    }
    
    //------------------------------------------------------------------------------
    void initICPFromPointCount(uint32_t actualPointCount)
    {
        // Only initialize once
        if (m_icp != DW_NULL_HANDLE)
            return;
        
        dwPointCloudICPParams params{};
        CHECK_DW_ERROR(dwPointCloudICP_getDefaultParams(&params));
        
        params.maxIterations = 50;
        params.icpType = DW_POINT_CLOUD_ICP_TYPE_DEPTH_MAP;
        
        // Calculate depth map size from actual point count (like InterLidarICP)
        // VLP16 has 16 vertical beams
        uint32_t verticalBeams = 16;
        uint32_t horizontalSamples = actualPointCount / verticalBeams;
        
        params.depthmapSize.x = horizontalSamples;
        params.depthmapSize.y = verticalBeams;
        params.maxPoints = params.depthmapSize.x * params.depthmapSize.y;
        
        // Verify the calculated size matches actual point cloud size
        if (params.maxPoints != actualPointCount)
        {
            logWarn("Depth map size mismatch! Calculated: %d, Actual: %d", 
                    params.maxPoints, actualPointCount);
            
            // Fallback: use actual point count and calculate closest rectangular dimensions
            params.depthmapSize.y = 16;  // Keep 16 beams for VLP16
            params.depthmapSize.x = actualPointCount / params.depthmapSize.y;
            params.maxPoints = params.depthmapSize.x * params.depthmapSize.y;
            
            log("Adjusted depth map size to: %dx%d (total: %d)", 
                params.depthmapSize.x, params.depthmapSize.y, params.maxPoints);
        }
        
        params.distanceConvergenceTol = 0.0005f;  // 0.5mm
        params.angleConvergenceTol = 0.005f;      // ~0.3°
        
        CHECK_DW_ERROR(dwPointCloudICP_initialize(&m_icp, &params, m_context));
        
        // Store calculated parameters
        m_icpMaxPoints = params.maxPoints;
        m_icpDepthmapSize = params.depthmapSize;
        
        // Initialize point cloud buffers for ICP (CUDA, organized)
        // Use calculated maxPoints as capacity (same as InterLidarICP)
        m_basicRigTransformed.capacity = m_icpMaxPoints;
        m_basicRigTransformed.type = DW_MEMORY_TYPE_CUDA;
        m_basicRigTransformed.format = DW_POINTCLOUD_FORMAT_XYZI;
        m_basicRigTransformed.organized = true;
        CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_basicRigTransformed));
        
        m_hrRigTransformed.capacity = m_icpMaxPoints;
        m_hrRigTransformed.type = DW_MEMORY_TYPE_CUDA;
        m_hrRigTransformed.format = DW_POINTCLOUD_FORMAT_XYZI;
        m_hrRigTransformed.organized = true;
        CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_hrRigTransformed));
        
        // Update ICP transformer output buffer capacity
        m_hrICPTransformed.capacity = m_icpMaxPoints;
        m_hrICPTransformed.type = DW_MEMORY_TYPE_CUDA;
        m_hrICPTransformed.format = DW_POINTCLOUD_FORMAT_XYZI;
        m_hrICPTransformed.organized = m_enableICP;  // Organized if ICP enabled
        CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_hrICPTransformed));
        CHECK_DW_ERROR(dwPointCloudStitcher_bindOutput(&m_hrICPTransformed, m_icpTransformer));
        
        log("ICP initialized from actual data: depthmap %dx%d, maxPoints=%d", 
            params.depthmapSize.x, params.depthmapSize.y, params.maxPoints);
    }
    
    //------------------------------------------------------------------------------
    void initStitchers()
    {
        // Always initialize point cloud buffers for rig transformations
        // If ICP is disabled, use a reasonable default capacity
        // If ICP is enabled, buffers will be resized in initICPFromPointCount()
        const uint32_t defaultCapacity = 50000;  // Enough for most lidar scans
        
        // Create INPUT buffers (separate from output!)
        m_basicInput.capacity = defaultCapacity;
        m_basicInput.size = 0;
        m_basicInput.type = DW_MEMORY_TYPE_CUDA;
        m_basicInput.format = DW_POINTCLOUD_FORMAT_XYZI;
        m_basicInput.organized = false;  // Raw .bin files are unorganized
        CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_basicInput));
        
        m_hrInput.capacity = defaultCapacity;
        m_hrInput.size = 0;
        m_hrInput.type = DW_MEMORY_TYPE_CUDA;
        m_hrInput.format = DW_POINTCLOUD_FORMAT_XYZI;
        m_hrInput.organized = false;  // Raw .bin files are unorganized
        CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_hrInput));
        
        // Create OUTPUT buffers (will receive transformed points)
        m_basicRigTransformed.capacity = defaultCapacity;
        m_basicRigTransformed.size = 0;
        m_basicRigTransformed.type = DW_MEMORY_TYPE_CUDA;
        m_basicRigTransformed.format = DW_POINTCLOUD_FORMAT_XYZI;
        m_basicRigTransformed.organized = false;  // Output is also unorganized
        CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_basicRigTransformed));
        
        m_hrRigTransformed.capacity = defaultCapacity;
        m_hrRigTransformed.size = 0;
        m_hrRigTransformed.type = DW_MEMORY_TYPE_CUDA;
        m_hrRigTransformed.format = DW_POINTCLOUD_FORMAT_XYZI;
        m_hrRigTransformed.organized = false;  // Output is also unorganized
        CHECK_DW_ERROR(dwPointCloud_createBuffer(&m_hrRigTransformed));
        
        // Initialize stitchers for rig transformations
        CHECK_DW_ERROR(dwPointCloudStitcher_initialize(&m_basicRigTransformer, m_context));
        CHECK_DW_ERROR(dwPointCloudStitcher_bindOutput(&m_basicRigTransformed, m_basicRigTransformer));
        
        CHECK_DW_ERROR(dwPointCloudStitcher_initialize(&m_hrRigTransformer, m_context));
        CHECK_DW_ERROR(dwPointCloudStitcher_bindOutput(&m_hrRigTransformed, m_hrRigTransformer));
        
        // Initialize stitcher for ICP transform application (only used if ICP enabled)
        CHECK_DW_ERROR(dwPointCloudStitcher_initialize(&m_icpTransformer, m_context));
        // Output buffer will be created and bound in initICPFromPointCount() if ICP enabled
        
        log("Point cloud stitchers initialized for transformations (ICP: %s)", 
            m_enableICP ? "enabled" : "disabled");
    }
    
    //------------------------------------------------------------------------------
    // Copy CPU point cloud to CUDA point cloud (same as InterLidarICP flow)
    // If ICP is enabled, ensures point cloud size matches ICP maxPoints exactly
    //------------------------------------------------------------------------------
    void copyToCudaPointCloud(const std::vector<dwVector4f>& cpuPoints, dwPointCloud& cudaCloud)
    {
        if (cpuPoints.empty())
        {
            cudaCloud.size = 0;
            return;
        }
        
        // If ICP is disabled, just copy what we have
        if (!m_enableICP || m_icpMaxPoints == 0)
        {
            uint32_t copySize = static_cast<uint32_t>(cpuPoints.size());
            
            // Buffer should already be created in initStitchers()
            // Just ensure we don't exceed capacity (truncate if needed)
            if (copySize > cudaCloud.capacity)
            {
                logWarn("Point cloud size (%u) exceeds buffer capacity (%u), truncating", 
                        copySize, cudaCloud.capacity);
                copySize = cudaCloud.capacity;
            }
            
            if (copySize > 0)
            {
                CHECK_CUDA_ERROR(cudaMemcpy(cudaCloud.points, cpuPoints.data(),
                                            sizeof(dwVector4f) * copySize,
                                            cudaMemcpyHostToDevice));
            }
            cudaCloud.size = copySize;
            // Stitcher can work with unorganized clouds (InterLidarICP output is organized=false)
            // Raw .bin files don't have depth map structure, so mark as unorganized
            cudaCloud.organized = false;
            return;
        }
        
        // ICP enabled: ensure exactly maxPoints points (pad with zeros if needed, truncate if too many)
        uint32_t targetSize = m_icpMaxPoints;
        uint32_t actualSize = static_cast<uint32_t>(cpuPoints.size());
        uint32_t copySize = std::min(actualSize, targetSize);
        
        // Copy available points
        CHECK_CUDA_ERROR(cudaMemcpy(cudaCloud.points, cpuPoints.data(),
                                    sizeof(dwVector4f) * copySize,
                                    cudaMemcpyHostToDevice));
        
        // Pad with zeros if needed to reach targetSize
        if (copySize < targetSize)
        {
            // Clear remaining buffer (invalid points)
            CHECK_CUDA_ERROR(cudaMemset(
                reinterpret_cast<char*>(cudaCloud.points) + copySize * sizeof(dwVector4f),
                0,
                (targetSize - copySize) * sizeof(dwVector4f)));
        }
        
        cudaCloud.size = targetSize;  // Always exactly maxPoints when ICP enabled
        cudaCloud.organized = true;  // Mark as organized for ICP
    }
    
    //------------------------------------------------------------------------------
    // Perform ICP alignment refinement
    // Points are reorganized into depth map format in copyToCudaPointCloud()
    //------------------------------------------------------------------------------
    bool performICP()
    {
        if (!m_icp || m_basicPoints.empty() || m_hrPoints.empty())
            return false;
        
        // Wait a few frames before starting ICP
        if (!m_icpInitialized)
        {
            m_icpInitFrames++;
            if (m_icpInitFrames < ICP_INIT_FRAMES)
                return false;
            m_icpInitialized = true;
            log("ICP initialization complete, starting alignment refinement");
        }
        
        // Copy rig-transformed points to CUDA buffers (reorganized into depth map)
        copyToCudaPointCloud(m_basicPoints, m_basicRigTransformed);
        copyToCudaPointCloud(m_hrPoints, m_hrRigTransformed);
        
        // Verify organized flag is set
        if (!m_basicRigTransformed.organized || !m_hrRigTransformed.organized)
        {
            logWarn("Point clouds not properly organized for ICP");
            return false;
        }
        
        // Use basic as target, HR as source
        // Use previous successful ICP result as initial guess (like InterLidarICP)
        dwTransformation3f initialGuess;
        if (m_currentBasicIndex > 0 && m_lastICPSuccessful)
        {
            // Use previous ICP transform as initial guess
            initialGuess = m_icpTransform;
        }
        else
        {
            // First frame or previous failed - use identity
            initialGuess = DW_IDENTITY_TRANSFORMATION3F;
        }
        
        // Bind ICP inputs and outputs
        CHECK_DW_ERROR(dwPointCloudICP_bindInput(&m_hrRigTransformed, 
                                                 &m_basicRigTransformed, 
                                                 &initialGuess, 
                                                 m_icp));
        CHECK_DW_ERROR(dwPointCloudICP_bindOutput(&m_icpTransform, m_icp));
        
        // Perform ICP
        dwStatus icpStatus = dwPointCloudICP_process(m_icp);
        
        if (icpStatus == DW_SUCCESS)
        {
            dwPointCloudICPResultStats stats{};
            CHECK_DW_ERROR(dwPointCloudICP_getLastResultStats(&stats, m_icp));
            
            m_lastICPSuccessful = true;
            
            if (m_currentBasicIndex % 10 == 0)
            {
                log("ICP: iterations=%d, RMS=%.3fmm, inliers=%.1f%%",
                    stats.actualNumIterations,
                    stats.rmsCost * 1000.0f,
                    stats.inlierFraction * 100.0f);
            }
            return true;
        }
        else
        {
            // Reset on failure
            m_lastICPSuccessful = false;
            // Don't reset transform - keep previous good result
            return false;
        }
    }

    //------------------------------------------------------------------------------
    void onReset() override
    {
        m_currentBasicIndex = 0;
        m_currentHrIndex     = 0;
        m_basicPointCount   = 0;
        m_hrPointCount      = 0;
        m_basicPoints.clear();
        m_hrPoints.clear();
        
        // Reset ICP state
        m_icpTransform = DW_IDENTITY_TRANSFORMATION3F;
        m_icpInitialized = false;
        m_icpInitFrames = 0;
        m_lastICPSuccessful = false;
    }

    //------------------------------------------------------------------------------
    void onRelease() override
    {
        if (m_basicBufferId)
        {
            dwRenderEngine_destroyBuffer(m_basicBufferId, m_renderEngine);
        }
        if (m_hrBufferId)
        {
            dwRenderEngine_destroyBuffer(m_hrBufferId, m_renderEngine);
        }

        if (m_renderEngine)
        {
            dwRenderEngine_release(m_renderEngine);
        }

        if (m_viz)
        {
            dwVisualizationRelease(m_viz);
        }

        // Release ICP resources
        if (m_icp)
        {
            dwPointCloudICP_release(m_icp);
        }
        dwPointCloud_destroyBuffer(&m_basicInput);
        dwPointCloud_destroyBuffer(&m_hrInput);
        dwPointCloud_destroyBuffer(&m_basicRigTransformed);
        dwPointCloud_destroyBuffer(&m_hrRigTransformed);
        dwPointCloud_destroyBuffer(&m_hrICPTransformed);
        
        // Release stitchers
        if (m_basicRigTransformer)
        {
            dwPointCloudStitcher_release(m_basicRigTransformer);
        }
        if (m_hrRigTransformer)
        {
            dwPointCloudStitcher_release(m_hrRigTransformer);
        }
        if (m_icpTransformer)
        {
            dwPointCloudStitcher_release(m_icpTransformer);
        }

        if (m_rig)
        {
            dwRig_release(m_rig);
        }

        if (m_context)
        {
            dwRelease(m_context);
        }

        dwLogger_release();
    }

    //------------------------------------------------------------------------------
    // Find HR file with closest timestamp to basic timestamp
    // Accounts for the initial timestamp offset between the two recordings
    //------------------------------------------------------------------------------
    size_t findClosestHrIndex(int64_t basicTs)
    {
        size_t bestIndex = m_currentHrIndex;
        int64_t bestDiff = std::numeric_limits<int64_t>::max();
        
        // Account for timestamp offset: HR started later, so we need to find
        // HR timestamp that matches: hrTs ≈ basicTs + offset
        int64_t targetHrTs = basicTs + m_timestampOffset;
        
        // Start searching from current HR index (files are sorted)
        for (size_t i = m_currentHrIndex; i < m_hrFiles.size(); ++i)
        {
            int64_t hrTs = 0;
            if (!filenameToTimestamp(m_hrFiles[i], hrTs))
                continue;
            
            // Compare against offset-adjusted target
            int64_t diff = std::abs(hrTs - targetHrTs);
            
            // Prefer HR frames that are slightly ahead (future) rather than behind (past)
            // to avoid showing "old" data that appears to lag visually
            bool isAhead = (hrTs >= targetHrTs);
            int64_t adjustedDiff = diff;
            if (!isAhead)
            {
                // Penalize frames that are behind by adding a penalty
                // This ensures we prefer frames that are at or ahead of the target
                adjustedDiff = diff + 2000;  // 2ms penalty for being behind
            }
            
            if (adjustedDiff < bestDiff)
            {
                bestDiff = adjustedDiff;
                bestIndex = i;
            }
            else if (hrTs > targetHrTs + MAX_TIMESTAMP_DIFF_US)
            {
                // Too far ahead, stop searching forward
                break;
            }
        }
        
        return bestIndex;
    }

    //------------------------------------------------------------------------------
    void onProcess() override
    {
        // Stop when basic stream ends
        if (m_currentBasicIndex >= m_basicFiles.size())
        {
            stop();
            return;
        }

        // Load current basic file
        const std::string& basicName = m_basicFiles[m_currentBasicIndex];
        std::string basicPath = m_basicDir + "/" + basicName;
        
        int64_t basicTs = 0;
        if (!filenameToTimestamp(basicName, basicTs))
        {
            std::cout << "Failed to parse timestamp from: " << basicName << std::endl;
            m_currentBasicIndex++;
            return;
        }

        // Find closest HR file by timestamp
        size_t hrIndex = findClosestHrIndex(basicTs);
        if (hrIndex >= m_hrFiles.size())
        {
            std::cout << "No matching HR file found for basic timestamp: " << basicTs << std::endl;
            stop();
            return;
        }

        const std::string& hrName = m_hrFiles[hrIndex];
        std::string hrPath = m_hrDir + "/" + hrName;
        
        int64_t hrTs = 0;
        if (!filenameToTimestamp(hrName, hrTs))
        {
            std::cout << "Failed to parse timestamp from: " << hrName << std::endl;
            m_currentBasicIndex++;
            return;
        }

        // Check if timestamps are close enough (within 100ms) after accounting for offset
        // HR started later, so compare: hrTs vs (basicTs + offset)
        int64_t offsetAdjustedBasicTs = basicTs + m_timestampOffset;
        int64_t timestampDiff = std::abs(hrTs - offsetAdjustedBasicTs);
        if (timestampDiff > MAX_TIMESTAMP_DIFF_US)
        {
            std::cout << "Warning: Timestamp mismatch too large: " << timestampDiff / 1000.0 
                      << "ms (max: " << MAX_TIMESTAMP_DIFF_US / 1000.0 << "ms)" << std::endl;
            std::cout << "  basicTs=" << basicTs << " hrTs=" << hrTs 
                      << " offset=" << m_timestampOffset << std::endl;
            std::cout << "  Skipping basic frame " << m_currentBasicIndex << std::endl;
            m_currentBasicIndex++;
            return;
        }

        // Load point clouds
        if (!loadPointCloud(basicPath, m_basicPoints) ||
            !loadPointCloud(hrPath,    m_hrPoints))
        {
            std::cout << "Failed to load pair: basic=" << basicName 
                      << " hr=" << hrName << ", stopping." << std::endl;
            stop();
            return;
        }

        // Initialize ICP from actual point count (like InterLidarICP) - only once, if enabled
        if (m_enableICP && m_icp == DW_NULL_HANDLE)
        {
            // Use the larger of the two point clouds to determine size
            uint32_t maxPointCount = std::max(static_cast<uint32_t>(m_basicPoints.size()),
                                             static_cast<uint32_t>(m_hrPoints.size()));
            initICPFromPointCount(maxPointCount);
        }

        // Step 1: Copy points to CUDA INPUT buffers (separate from output!)
        uint32_t basicInputSize = static_cast<uint32_t>(m_basicPoints.size());
        uint32_t hrInputSize = static_cast<uint32_t>(m_hrPoints.size());
        
        copyToCudaPointCloud(m_basicPoints, m_basicInput);
        copyToCudaPointCloud(m_hrPoints, m_hrInput);
        
        // Debug: Check sizes after copying to CUDA
        if (m_currentBasicIndex % 50 == 0)
        {
            std::cout << "After copyToCuda: basic input size=" << m_basicInput.size 
                      << " (expected=" << basicInputSize << "), hr input size=" << m_hrInput.size 
                      << " (expected=" << hrInputSize << ")" << std::endl;
        }
        
        // Step 2: Apply rig transformations using stitchers (INPUT -> OUTPUT buffers)
        std::cout << "[DEBUG] Stitcher input: basic size=" << m_basicInput.size 
                  << " organized=" << (m_basicInput.organized ? "true" : "false") << std::endl;
        
        CHECK_DW_ERROR(dwPointCloudStitcher_bindInput(DW_BIND_SLOT_1,
                                                      &m_basicInput,      // INPUT buffer
                                                      &m_frontSensorToRig,
                                                      m_basicRigTransformer));
        CHECK_DW_ERROR(dwPointCloudStitcher_process(m_basicRigTransformer));
        
        std::cout << "[DEBUG] After basic stitcher: output size=" << m_basicRigTransformed.size << std::endl;
        
        std::cout << "[DEBUG] Stitcher input: hr size=" << m_hrInput.size 
                  << " organized=" << (m_hrInput.organized ? "true" : "false") << std::endl;
        
        CHECK_DW_ERROR(dwPointCloudStitcher_bindInput(DW_BIND_SLOT_1,
                                                      &m_hrInput,         // INPUT buffer
                                                      &m_rearSensorToRig,
                                                      m_hrRigTransformer));
        CHECK_DW_ERROR(dwPointCloudStitcher_process(m_hrRigTransformer));
        
        std::cout << "[DEBUG] After HR stitcher: output size=" << m_hrRigTransformed.size << std::endl;
        
        // Step 3: Perform ICP correction for fine alignment (if enabled)
        if (m_enableICP && performICP())
        {
            // Apply ICP transform to HR points using stitcher (like InterLidarICP - use current transform, not cumulative)
            CHECK_DW_ERROR(dwPointCloudStitcher_bindInput(DW_BIND_SLOT_1,
                                                          &m_hrRigTransformed,
                                                          &m_icpTransform,
                                                          m_icpTransformer));
            CHECK_DW_ERROR(dwPointCloudStitcher_process(m_icpTransformer));
            m_hrICPTransformed.organized = true;
            
            // Copy ICP-transformed points back to hrRigTransformed for rendering
            CHECK_CUDA_ERROR(cudaMemcpy(m_hrRigTransformed.points, m_hrICPTransformed.points,
                                        sizeof(dwVector4f) * m_hrICPTransformed.size,
                                        cudaMemcpyDeviceToDevice));
            m_hrRigTransformed.size = m_hrICPTransformed.size;
        }
        
        // Step 4: Copy transformed points back to CPU for rendering
        // Get actual sizes from transformed point clouds
        uint32_t basicSize = m_basicRigTransformed.size;
        uint32_t hrSize = m_hrRigTransformed.size;
        
        std::cout << "[DEBUG] Before copy to CPU: basicSize=" << basicSize << " hrSize=" << hrSize << std::endl;
        
        // Debug: Check sizes before copying
        if (basicSize == 0 || hrSize == 0)
        {
            std::cout << "[ERROR] Point cloud sizes are zero after stitching! basicSize=" << basicSize 
                      << " hrSize=" << hrSize << " (ICP enabled: " << m_enableICP << ")" << std::endl;
            std::cout << "[ERROR] Using original input sizes: basic=" << basicInputSize 
                      << " hr=" << hrInputSize << std::endl;
            // Use original input sizes
            if (basicSize == 0) basicSize = basicInputSize;
            if (hrSize == 0) hrSize = hrInputSize;
        }
        
        m_basicPoints.clear();
        m_hrPoints.clear();
        
        // Copy basic points (reference, no ICP transform)
        if (basicSize > 0)
        {
            std::vector<dwVector4f> basicTmp(basicSize);
            CHECK_CUDA_ERROR(cudaMemcpy(basicTmp.data(), m_basicRigTransformed.points,
                                        sizeof(dwVector4f) * basicSize,
                                        cudaMemcpyDeviceToHost));
            m_basicPoints = basicTmp;
            
            // Sample a few points for debugging - always show for first 10 frames
            if ((m_currentBasicIndex < 10 || m_currentBasicIndex % 100 == 0) && basicTmp.size() >= 3)
            {
                std::cout << "[POINT-CHECK] Basic point samples (frame " << m_currentBasicIndex << "): " 
                          << "(" << basicTmp[0].x << "," << basicTmp[0].y << "," << basicTmp[0].z << ") "
                          << "(" << basicTmp[100].x << "," << basicTmp[100].y << "," << basicTmp[100].z << ") "
                          << "(" << basicTmp[1000].x << "," << basicTmp[1000].y << "," << basicTmp[1000].z << ")" << std::endl;
            }
        }
        
        // Copy HR points (with ICP transform applied if enabled)
        if (hrSize > 0)
        {
            std::vector<dwVector4f> hrTmp(hrSize);
            CHECK_CUDA_ERROR(cudaMemcpy(hrTmp.data(), m_hrRigTransformed.points,
                                        sizeof(dwVector4f) * hrSize,
                                        cudaMemcpyDeviceToHost));
            m_hrPoints = hrTmp;
            
            // Sample a few points for debugging - always show for first 10 frames
            if ((m_currentBasicIndex < 10 || m_currentBasicIndex % 100 == 0) && hrTmp.size() >= 3)
            {
                std::cout << "[POINT-CHECK] HR point samples (frame " << m_currentBasicIndex << "): " 
                          << "(" << hrTmp[0].x << "," << hrTmp[0].y << "," << hrTmp[0].z << ") "
                          << "(" << hrTmp[100].x << "," << hrTmp[100].y << "," << hrTmp[100].z << ") "
                          << "(" << hrTmp[1000].x << "," << hrTmp[1000].y << "," << hrTmp[1000].z << ")" << std::endl;
            }
        }

        m_basicPointCount = static_cast<uint32_t>(m_basicPoints.size());
        m_hrPointCount    = static_cast<uint32_t>(m_hrPoints.size());
        
        std::cout << "[DEBUG] Final counts for rendering: basic=" << m_basicPointCount 
                  << " hr=" << m_hrPointCount << std::endl;
        
        // Debug: Check if we have points
        if (m_basicPointCount == 0 && m_hrPointCount == 0)
        {
            std::cout << "[ERROR] No points to render! Check stitcher output." << std::endl;
        }

        // Calculate time difference from previous frame (use basic lidar as reference)
        int64_t frameDeltaUs = 0;
        if (!m_firstFrame && m_lastFrameTimestamp > 0)
        {
            frameDeltaUs = basicTs - m_lastFrameTimestamp;
        }
        else
        {
            // First frame - use typical scan period (10 Hz = 100ms)
            frameDeltaUs = 100000;  // 100ms in microseconds
        }
        
        std::cout << "Frame " << m_currentBasicIndex
                  << " | basic=" << basicTs
                  << " hr=" << hrTs
                  << " timestamp_diff_ms=" << timestampDiff / 1000.0
                  << " frame_delta_ms=" << frameDeltaUs / 1000.0
                  << std::endl;

        // Update render buffers
        uploadToRenderBuffer(m_basicBufferId, m_basicPoints, m_basicPointCount);
        uploadToRenderBuffer(m_hrBufferId,    m_hrPoints,    m_hrPointCount);

        // Throttle to match real-time scan rate
        if (!m_firstFrame)
        {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                now - m_lastFrameTime).count();
            
            int64_t sleepTimeUs = frameDeltaUs - elapsed;
            if (sleepTimeUs > 0)
            {
                std::this_thread::sleep_for(std::chrono::microseconds(sleepTimeUs));
            }
        }
        
        m_lastFrameTimestamp = basicTs;
        m_lastFrameTime = std::chrono::steady_clock::now();
        m_firstFrame = false;
        
        // Advance indices
        m_currentBasicIndex++;
        m_currentHrIndex = hrIndex + 1;  // Start next search from here (files are sorted)
    }

    //------------------------------------------------------------------------------
    void onRender() override
    {
        dwRenderEngine_reset(m_renderEngine);
        dwRenderEngine_setTile(0, m_renderEngine);

        // Set camera from MouseView3D
        dwRenderEngine_setModelView(getMouseView().getModelView(), m_renderEngine);
        dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine);

        // Draw a simple grid
        // (re-use same helper as in lidar_replay if desired; here we skip explicit grid)

        // Render basic lidar (front) in green
        dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_GREEN, m_renderEngine);
        dwRenderEngine_renderBuffer(m_basicBufferId, m_basicPointCount, m_renderEngine);

        // Render HR lidar (rear) in red
        dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_RED, m_renderEngine);
        dwRenderEngine_renderBuffer(m_hrBufferId, m_hrPointCount, m_renderEngine);

        // Render simple text overlay
        dwRenderEngine_setProjection(&DW_IDENTITY_MATRIX4F, m_renderEngine);
        dwRenderEngine_setModelView(&DW_IDENTITY_MATRIX4F, m_renderEngine);

        dwVector2f range{static_cast<float32_t>(getWindowWidth()),
                         static_cast<float32_t>(getWindowHeight())};
        dwRenderEngine_setCoordinateRange2D(range, m_renderEngine);
        dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_renderEngine);
        dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_WHITE, m_renderEngine);

        std::string msg = "Recording Visualizer - Frame " + std::to_string(m_currentBasicIndex) +
                          " (ESC to exit)";
        dwRenderEngine_renderText2D(msg.c_str(), {20.f, 20.f}, m_renderEngine);
    }

    //------------------------------------------------------------------------------
    void onResizeWindow(int width, int height) override
    {
        dwRectf bounds{.x = 0.0f, .y = 0.0f};
        bounds.width  = static_cast<float32_t>(width);
        bounds.height = static_cast<float32_t>(height);
        dwRenderEngine_setBounds(bounds, m_renderEngine);
    }

private:
    void uploadToRenderBuffer(uint32_t bufferId,
                              const std::vector<dwVector4f>& points,
                              uint32_t count)
    {
        if (count == 0)
            return;

        // Prepare XYZ positions for rendering
        std::vector<dwVector3f> positions;
        positions.resize(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            positions[i].x = points[i].x;
            positions[i].y = points[i].y;
            positions[i].z = points[i].z;
        }

        // Upload to render buffer
        CHECK_DW_ERROR(dwRenderEngine_setBuffer(bufferId,
                                                DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
                                                positions.data(),
                                                sizeof(dwVector3f),
                                                0,
                                                count,
                                                m_renderEngine));
    }
};

//------------------------------------------------------------------------------
int main(int argc, const char** argv)
{
    ProgramArguments args(argc, argv,
                          {
                              ProgramArguments::Option_t("rig-file", ""),
                              ProgramArguments::Option_t("basic-dir", ""),
                              ProgramArguments::Option_t("hr-dir", ""),
                              ProgramArguments::Option_t("enable-icp", "true", "Enable/disable ICP alignment (true/false, default: true)"),
                          },
                          "Recording Visualizer for two lidar streams\n"
                          "  - Applies rig transforms and visualizes both clouds\n"
                          "  - Expects .bin files from lidar_replay (XYZI floats)\n\n"
                          "Required arguments:\n"
                          "  --rig-file=<path>    : Path to rig configuration file (rig.json)\n"
                          "  --basic-dir=<path>   : Directory with basic lidar .bin files\n"
                          "  --hr-dir=<path>      : Directory with HR lidar .bin files\n"
                          "Optional arguments:\n"
                          "  --enable-icp=<true|false> : Enable ICP alignment refinement (default: true)\n");

    RecordingVisualizerSample app(args);

    app.initializeWindow("Recording Visualizer", 1280, 800, args.enabled("offscreen"));

    return app.run();
}


