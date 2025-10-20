#ifndef FREESPACE_DW_HPP
#define FREESPACE_DW_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

// Eigen for matrix operations (same as original Livox implementation)
#include <Eigen/Core>
#include <Eigen/Dense>

#include <dw/core/base/Types.h>
#include <dw/pointcloudprocessing/pointcloud/PointCloud.h>
#include "FreeSpaceDW.hpp"


// Grid parameters adapted for Velodyne VLP-16 range (~100m)
#define DW_GND_IMG_NX 150
#define DW_GND_IMG_NY 400
#define DW_GND_IMG_DX 0.2f
#define DW_GND_IMG_DY 0.2f
#define DW_GND_IMG_OFFX 40
#define DW_GND_IMG_OFFY 40

#define DW_DN_SAMPLE_IMG_NX 500
#define DW_DN_SAMPLE_IMG_NY 200
#define DW_DN_SAMPLE_IMG_NZ 100
#define DW_DN_SAMPLE_IMG_DX 0.4f
#define DW_DN_SAMPLE_IMG_DY 0.4f
#define DW_DN_SAMPLE_IMG_DZ 0.2f
#define DW_DN_SAMPLE_IMG_OFFX 50
#define DW_DN_SAMPLE_IMG_OFFY 40
#define DW_DN_SAMPLE_IMG_OFFZ 10

#define DW_DENOISE_IMG_NX 200
#define DW_DENOISE_IMG_NY 80
#define DW_DENOISE_IMG_NZ 100
#define DW_DENOISE_IMG_DX 1.0f
#define DW_DENOISE_IMG_DY 1.0f
#define DW_DENOISE_IMG_DZ 0.2f

#define DW_FREE_PI 3.14159265f
#define DW_FREE_SPACE_RESOLUTION 360  // Angular resolution for free space

class FreeSpaceDW
{
public:
    FreeSpaceDW();
    ~FreeSpaceDW();
    
    // Main interface - processes DriveWorks point cloud
    bool GenerateFreeSpace(const dwPointCloud& pointCloud, 
                          std::vector<float32_t>& freeSpacePoints);
    
    // Get free space as radial distances (360 angles)
    void GetFreeSpaceRadial(std::vector<float32_t>& radialDistances) const;
    
    // Get statistics
    void GetStatistics(uint32_t& numGroundPoints, 
                      uint32_t& numObstaclePoints,
                      uint32_t& numFreeSpacePoints) const;

private:
    // Core processing functions
    bool DownsampleAndDenoise(const dwPointCloud& input,
                             std::vector<dwVector4f>& downsampled);
    
    int GroundSegment(const std::vector<dwVector4f>& points,
                     std::vector<int>& labels);
    
    void ComputeFreeSpaceRadial(const std::vector<dwVector4f>& obstaclePoints,
                               std::vector<float32_t>& radialDistances);
    
    void FilterFreeSpace(const std::vector<float32_t>& radialInput,
                        std::vector<float32_t>& freeSpacePoints);
    
    // Helper functions using Eigen (same as original)
    void FitGroundPlaneEigen(const std::vector<dwVector4f>& seedPoints,
                            Eigen::Vector3f& normal, float32_t& offset);
    
    // Voxel grid for downsampling
    uint8_t* m_voxelGrid;
    
    // Statistics
    uint32_t m_numGroundPoints;
    uint32_t m_numObstaclePoints;
    uint32_t m_numFreeSpacePoints;
    
    // Free space radial distances (cached)
    std::vector<float32_t> m_radialDistances;
    
    // Filter kernels (from original implementation)
    static const int FILTER_X[28];
    static const int FILTER_Y[28];
    static const int ALL_X[89];
    static const int ALL_Y[89];
};

#endif // FREESPACE_DW_HPP