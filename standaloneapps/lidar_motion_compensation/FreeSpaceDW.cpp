#include "FreeSpaceDW.hpp"
#include <iostream>
#include <cstring>
#include <cstdlib>

// Filter kernels from original Livox implementation
const int FreeSpaceDW::FILTER_X[28] = {
    -1,0,1,-3,-2,2,3,-4,4,-4,4,-5,5,-5,5,-5,5,-1,0,1,-3,-2,2,3,-4,4,-4,4
};

const int FreeSpaceDW::FILTER_Y[28] = {
    -5,-5,-5,-4,-4,-4,-4,-3,-3,-2,-2,-1,-1,0,0,1,1,5,5,5,4,4,4,4,3,3,2,2
};

const int FreeSpaceDW::ALL_X[89] = {
    -1,0,1, 
    -3,-2,-1,0,1,2,3, 
    -4,-3,-2,-1,0,1,2,3,4, 
    -4,-3,-2,-1,0,1,2,3,4, 
    -5,-4,-3,-2,-1,0,1,2,3,4,5,
    -5,-4,-3,-2,-1,0,1,2,3,4,5,
    -5,-4,-3,-2,-1,0,1,2,3,4,5,
    -1,0,1,
    -3,-2,-1,0,1,2,3,
    -4,-3,-2,-1,0,1,2,3,4,
    -4,-3,-2,-1,0,1,2,3,4
};

const int FreeSpaceDW::ALL_Y[89] = {
    -5,-5,-5,
    -4,-4,-4,-4,-4,-4,-4,
    -3,-3,-3,-3,-3,-3,-3,-3,-3,
    -2,-2,-2,-2,-2,-2,-2,-2,-2,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,1,1,
    5,5,5,
    4,4,4,4,4,4,4,
    3,3,3,3,3,3,3,3,3,
    2,2,2,2,2,2,2,2,2
};

FreeSpaceDW::FreeSpaceDW()
    : m_voxelGrid(nullptr)
    , m_numGroundPoints(0)
    , m_numObstaclePoints(0)
    , m_numFreeSpacePoints(0)
{
    // Allocate voxel grid for downsampling
    size_t voxelSize = DW_DN_SAMPLE_IMG_NX * DW_DN_SAMPLE_IMG_NY * DW_DN_SAMPLE_IMG_NZ;
    m_voxelGrid = new uint8_t[voxelSize];
    
    // Pre-allocate radial distances
    m_radialDistances.resize(DW_FREE_SPACE_RESOLUTION, 2500.0f); // Max 50m range
}

FreeSpaceDW::~FreeSpaceDW()
{
    if (m_voxelGrid) {
        delete[] m_voxelGrid;
        m_voxelGrid = nullptr;
    }
}

bool FreeSpaceDW::GenerateFreeSpace(const dwPointCloud& pointCloud,
                                    std::vector<float32_t>& freeSpacePoints)
{
    if (pointCloud.size == 0) {
        std::cerr << "[FreeSpace] Empty point cloud" << std::endl;
        return false;
    }
    
    std::cout << "[FreeSpace] Processing " << pointCloud.size << " points" << std::endl;
    
    // Step 1: Downsample and denoise
    std::vector<dwVector4f> downsampled;
    if (!DownsampleAndDenoise(pointCloud, downsampled)) {
        std::cerr << "[FreeSpace] Downsampling failed" << std::endl;
        return false;
    }
    
    std::cout << "[FreeSpace] After downsampling: " << downsampled.size() << " points" << std::endl;
    
    // Step 2: Ground segmentation
    std::vector<int> labels(downsampled.size(), 0);
    m_numGroundPoints = GroundSegment(downsampled, labels);
    
    std::cout << "[FreeSpace] Ground points: " << m_numGroundPoints << std::endl;
    
    // Step 3: Extract obstacle points (non-ground)
    std::vector<dwVector4f> obstaclePoints;
    obstaclePoints.reserve(downsampled.size() - m_numGroundPoints);
    
    for (size_t i = 0; i < downsampled.size(); i++) {
        if (labels[i] == 0) { // Not ground
            obstaclePoints.push_back(downsampled[i]);
        }
    }
    
    m_numObstaclePoints = obstaclePoints.size();
    std::cout << "[FreeSpace] Obstacle points: " << m_numObstaclePoints << std::endl;
    
    // Step 4: Compute radial free space
    ComputeFreeSpaceRadial(obstaclePoints, m_radialDistances);
    
    // Step 5: Filter and expand free space
    FilterFreeSpace(m_radialDistances, freeSpacePoints);
    
    m_numFreeSpacePoints = freeSpacePoints.size() / 3; // x,y,intensity triplets
    std::cout << "[FreeSpace] Generated " << m_numFreeSpacePoints << " free space points" << std::endl;


    std::cout << "\n[FreeSpace DIAGNOSTIC] srcGrid Analysis:" << std::endl;
    
    const float32_t pixelSize = 0.2f;
    const int gridSize = static_cast<int>(100.0f / pixelSize);
    
    // Count filled cells in each direction from center
    int centerIdx = gridSize / 2;
    
    // We need to recompute this for diagnostics (or make srcGrid a member variable)
    // For now, let's analyze the OUTPUT instead
    
    std::cout << "  Free space points generated: " << m_numFreeSpacePoints << std::endl;
    
    // Analyze free space distribution
    std::vector<int> freeSpaceByAngle(8, 0); // 8 octants
    for (size_t i = 0; i < freeSpacePoints.size(); i += 3) {
        float x = freeSpacePoints[i];
        float y = freeSpacePoints[i+1];
        
        float angle = std::atan2(y, x) * 180.0f / DW_FREE_PI;
        int octant = static_cast<int>((angle + 202.5f) / 45.0f) % 8;
        freeSpaceByAngle[octant]++;
    }
    
    const char* octantNames[] = {"FRONT", "FRONT-LEFT", "LEFT", "BACK-LEFT", 
                                  "BACK", "BACK-RIGHT", "RIGHT", "FRONT-RIGHT"};
    
    std::cout << "  Free space distribution by octant:" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << "    " << octantNames[i] << ": " << freeSpaceByAngle[i] << " points" << std::endl;
    }

    
    return true;
}

bool FreeSpaceDW::DownsampleAndDenoise(const dwPointCloud& input,
                                       std::vector<dwVector4f>& downsampled)
{
    // Copy to host if needed
    const dwVector4f* points = nullptr;
    std::vector<dwVector4f> hostPoints;
    
    if (input.type == DW_MEMORY_TYPE_CUDA) {
        hostPoints.resize(input.size);
        cudaMemcpy(hostPoints.data(), input.points,
                  input.size * sizeof(dwVector4f),
                  cudaMemcpyDeviceToHost);
        points = hostPoints.data();
    } else {
        points = static_cast<const dwVector4f*>(input.points);
    }
    
    // First pass: count points per voxel for denoising
    size_t denoiseVoxelSize = DW_DENOISE_IMG_NX * DW_DENOISE_IMG_NY * DW_DENOISE_IMG_NZ;
    std::vector<int> voxelCount(denoiseVoxelSize, 0);
    std::vector<int> denoiseIndices(input.size, -1);
    
    // Count points per voxel
    for (uint32_t i = 0; i < input.size; i++) {
        int ix = static_cast<int>((points[i].x + DW_DN_SAMPLE_IMG_OFFX) / DW_DENOISE_IMG_DX);
        int iy = static_cast<int>((points[i].y + DW_DN_SAMPLE_IMG_OFFY) / DW_DENOISE_IMG_DY);
        int iz = static_cast<int>((points[i].z + DW_DN_SAMPLE_IMG_OFFZ) / DW_DENOISE_IMG_DZ);
        
        if (ix >= 0 && ix < DW_DENOISE_IMG_NX &&
            iy >= 0 && iy < DW_DENOISE_IMG_NY &&
            iz >= 0 && iz < DW_DENOISE_IMG_NZ) {
            int idx = iz * DW_DENOISE_IMG_NX * DW_DENOISE_IMG_NY + 
                     iy * DW_DENOISE_IMG_NX + ix;
            denoiseIndices[i] = idx;
            voxelCount[idx]++;
        }
    }
    
    // Remove isolated points (noise removal - require at least 3 points in voxel)
    for (uint32_t i = 0; i < input.size; i++) {
        if (denoiseIndices[i] > -1 && voxelCount[denoiseIndices[i]] < 3) {
            denoiseIndices[i] = -1; // Mark for removal
        }
    }
    
    // Second pass: downsample using finer grid
    size_t voxelSize = DW_DN_SAMPLE_IMG_NX * DW_DN_SAMPLE_IMG_NY * DW_DN_SAMPLE_IMG_NZ;
    std::memset(m_voxelGrid, 0, voxelSize);
    
    downsampled.reserve(input.size / 4); // Rough estimate
    
    for (uint32_t i = 0; i < input.size; i++) {
        // Skip if marked as noise
        if (denoiseIndices[i] == -1) {
            continue;
        }
        
        int ix = static_cast<int>((points[i].x + DW_DN_SAMPLE_IMG_OFFX) / DW_DN_SAMPLE_IMG_DX);
        int iy = static_cast<int>((points[i].y + DW_DN_SAMPLE_IMG_OFFY) / DW_DN_SAMPLE_IMG_DY);
        int iz = static_cast<int>((points[i].z + DW_DN_SAMPLE_IMG_OFFZ) / DW_DN_SAMPLE_IMG_DZ);
        
        if (ix >= 0 && ix < DW_DN_SAMPLE_IMG_NX &&
            iy >= 0 && iy < DW_DN_SAMPLE_IMG_NY &&
            iz >= 0 && iz < DW_DN_SAMPLE_IMG_NZ) {
            int idx = iz * DW_DN_SAMPLE_IMG_NX * DW_DN_SAMPLE_IMG_NY + 
                     iy * DW_DN_SAMPLE_IMG_NX + ix;
            
            // Take first point in each downsample voxel
            if (m_voxelGrid[idx] == 0) {
                downsampled.push_back(points[i]);
                m_voxelGrid[idx] = 1;
            }
        }
    }
    
    return !downsampled.empty();
}

int FreeSpaceDW::GroundSegment(const std::vector<dwVector4f>& points,
                               std::vector<int>& labels)
{
    // Step 1: Rule-based coarse ground filtering using height grid
    const int gridNX = 24;
    const int gridNY = 20;
    const float32_t gridDX = 4.0f;
    const float32_t gridDY = 4.0f;
    const int gridOffX = 40;
    const int gridOffY = 40;
    
    std::vector<float32_t> minHeightGrid(gridNX * gridNY, 100.0f);
    std::vector<int> gridIndices(points.size(), -1);
    
    // Find minimum height in each grid cell
    for (size_t i = 0; i < points.size(); i++) {
        int ix = static_cast<int>((points[i].x + gridOffX) / gridDX);
        int iy = static_cast<int>((points[i].y + gridOffY) / gridDY);
        
        if (ix >= 0 && ix < gridNX && iy >= 0 && iy < gridNY) {
            int idx = ix + iy * gridNX;
            gridIndices[i] = idx;
            
            if (minHeightGrid[idx] > points[i].z) {
                minHeightGrid[idx] = points[i].z;
            }
        }
    }
    
    // Collect seed points for plane fitting
    std::vector<dwVector4f> seedPoints;
    int preliminaryGroundCount = 0;
    
    for (size_t i = 0; i < points.size(); i++) {
        if (gridIndices[i] >= 0) {
            float32_t heightDiff = points[i].z - minHeightGrid[gridIndices[i]];
            
            if (heightDiff < 0.4f) {
                labels[i] = 1; // Preliminary ground candidate
                preliminaryGroundCount++;
                
                // Collect seed points (close range, low height)
                float32_t dist2 = points[i].x * points[i].x + points[i].y * points[i].y;
                if (dist2 < 100.0f && points[i].z < 1.0f) {
                    seedPoints.push_back(points[i]);
                }
            }
        }
    }
    
    std::cout << "[FreeSpace] Preliminary ground candidates: " << preliminaryGroundCount << std::endl;
    std::cout << "[FreeSpace] Seed points for plane fitting: " << seedPoints.size() << std::endl;
    
    // Step 2: Refine with Eigen-based plane fitting (like original implementation)
    if (seedPoints.size() < 100) {
        std::cout << "[FreeSpace] Not enough seed points for plane fitting, using rule-based only" << std::endl;
        return preliminaryGroundCount;
    }
    
    Eigen::Vector3f normal;
    float32_t offset;
    FitGroundPlaneEigen(seedPoints, normal, offset);
    
    std::cout << "[FreeSpace] Ground plane normal: [" << normal.x() << ", " 
              << normal.y() << ", " << normal.z() << "]" << std::endl;
    std::cout << "[FreeSpace] Ground plane offset: " << offset << std::endl;
    
    // Step 3: Re-label using fitted plane
    const float32_t threshold = 0.3f; // 30cm threshold
    int groundCount = 0;
    
    for (size_t i = 0; i < points.size(); i++) {
        float32_t dist = std::abs(normal.x() * points[i].x + 
                                 normal.y() * points[i].y + 
                                 normal.z() * points[i].z + offset);
        
        float32_t dist2 = points[i].x * points[i].x + points[i].y * points[i].y;
        
        // Points within threshold distance and reasonable range
        if (dist2 < 400.0f && dist < threshold) {
            labels[i] = 1;
            groundCount++;
        } else {
            labels[i] = 0;
        }
    }
    
    // Step 4: Additional heuristic refinements
    for (size_t i = 0; i < points.size(); i++) {
        if (labels[i] == 1) {
            // Remove high points that passed plane test
            if (points[i].z > 1.0f) {
                labels[i] = 0;
                groundCount--;
            }
            // Remove close-range elevated points
            else if (points[i].x * points[i].x + points[i].y * points[i].y < 100.0f) {
                if (points[i].z > 0.5f) {
                    labels[i] = 0;
                    groundCount--;
                }
            }
        } else {
            // Add low points that are clearly ground
            float32_t dist2 = points[i].x * points[i].x + points[i].y * points[i].y;
            if (dist2 < 400.0f && points[i].z < 0.2f) {
                labels[i] = 1;
                groundCount++;
            }
        }
    }
    
    // std::cout << "[FreeSpace] Final ground points after plane refinement: " << groundCount << std::endl;


    std::cout << "\n[FreeSpace DIAGNOSTIC] Ground Segmentation by Region:" << std::endl;
    
    int frontGround = 0, frontObstacle = 0;
    int backGround = 0, backObstacle = 0;
    int leftGround = 0, leftObstacle = 0;
    int rightGround = 0, rightObstacle = 0;
    
    for (size_t i = 0; i < points.size(); i++) {
        bool isGround = (labels[i] == 1);
        
        // Classify by quadrant
        if (points[i].x > 0) {  // Front
            if (isGround) frontGround++; else frontObstacle++;
        } else {  // Back
            if (isGround) backGround++; else backObstacle++;
        }
        
        if (points[i].y > 0) {  // Left
            if (isGround) leftGround++; else leftObstacle++;
        } else {  // Right
            if (isGround) rightGround++; else rightObstacle++;
        }
    }
    
    std::cout << "  FRONT (X>0): Ground=" << frontGround << ", Obstacles=" << frontObstacle 
              << " (" << (100.0f*frontObstacle/(frontGround+frontObstacle)) << "% obstacles)" << std::endl;
    std::cout << "  BACK  (X<0): Ground=" << backGround << ", Obstacles=" << backObstacle
              << " (" << (100.0f*backObstacle/(backGround+backObstacle)) << "% obstacles)" << std::endl;
    std::cout << "  LEFT  (Y>0): Ground=" << leftGround << ", Obstacles=" << leftObstacle
              << " (" << (100.0f*leftObstacle/(leftGround+leftObstacle)) << "% obstacles)" << std::endl;
    std::cout << "  RIGHT (Y<0): Ground=" << rightGround << ", Obstacles=" << rightObstacle
              << " (" << (100.0f*rightObstacle/(rightGround+rightObstacle)) << "% obstacles)" << std::endl;






    
    return groundCount;
}

void FreeSpaceDW::FitGroundPlaneEigen(const std::vector<dwVector4f>& seedPoints,
                                     Eigen::Vector3f& normal, float32_t& offset)
{
    // Convert to Eigen point cloud
    Eigen::MatrixXf points(seedPoints.size(), 3);
    for (size_t i = 0; i < seedPoints.size(); i++) {
        points(i, 0) = seedPoints[i].x;
        points(i, 1) = seedPoints[i].y;
        points(i, 2) = seedPoints[i].z;
    }
    
    // Compute mean
    Eigen::Vector4f pc_mean;
    pc_mean.head<3>() = points.colwise().mean();
    pc_mean(3) = 0.0f;
    
    // Compute covariance matrix
    Eigen::Matrix3f cov;
    Eigen::MatrixXf centered = points.rowwise() - points.colwise().mean();
    cov = (centered.transpose() * centered) / float(seedPoints.size());
    
    // SVD to find plane normal (eigenvector with smallest eigenvalue)
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov, Eigen::ComputeFullU);
    Eigen::MatrixXf normal_matrix = svd.matrixU().col(2);
    
    normal = normal_matrix.col(0);
    
    // Ensure normal points upward (positive z)
    if (normal.z() < 0) {
        normal = -normal;
    }
    
    // Compute offset: normal.T * [x,y,z] = -d
    Eigen::Vector3f seeds_mean = pc_mean.head<3>();
    offset = -(normal.transpose() * seeds_mean)(0, 0);
    
    std::cout << "[FreeSpace] SVD-based plane fitting complete" << std::endl;
}

// Original function
// void FreeSpaceDW::ComputeFreeSpaceRadial(const std::vector<dwVector4f>& obstaclePoints,
//                                         std::vector<float32_t>& radialDistances)
// {
//     // Initialize to maximum range
//     std::fill(radialDistances.begin(), radialDistances.end(), 2500.0f); // 50m^2
    
//     // For each obstacle point, update the minimum distance in its angular bin
//     for (const auto& p : obstaclePoints) {
//         // Skip high points (trees, buildings) and very close points (robot body)
//         if (p.z > 3.0f) continue;
//         if (std::abs(p.y) < 1.2f && std::abs(p.x) < 2.5f) continue;
        
//         float32_t distance = p.x * p.x + p.y * p.y;
//         int thetaId = static_cast<int>((std::atan2(p.y, p.x) + DW_FREE_PI) * 
//                                        180.0f / DW_FREE_PI + 0.5f);
//         thetaId = thetaId % DW_FREE_SPACE_RESOLUTION;
        
//         if (radialDistances[thetaId] > distance && distance > 1.0f) {
//             radialDistances[thetaId] = distance;
//         }
//     }
// }

void FreeSpaceDW::ComputeFreeSpaceRadial(const std::vector<dwVector4f>& obstaclePoints,
    std::vector<float32_t>& radialDistances)
{

    // ADD THIS DIAGNOSTIC BLOCK AT THE START:
    int nearFieldCount = 0;
    int nearFieldFront = 0;
    for (const auto& p : obstaclePoints) {
        float dist = std::sqrt(p.x * p.x + p.y * p.y);
        if (dist < 7.0f) {
            nearFieldCount++;
            if (p.x > 0 && std::abs(p.y) < 2.0f) { // Front 4m wide cone
                nearFieldFront++;
            }
        }
    }
    std::cout << "[DIAGNOSTIC] Near-field (<7m) obstacle points: " << nearFieldCount << std::endl;
    std::cout << "[DIAGNOSTIC] Near-field FRONT cone points: " << nearFieldFront << std::endl;
    

    // Initialize to maximum range
    std::fill(radialDistances.begin(), radialDistances.end(), 2500.0f); // 50m^2

    // Track obstacle distribution by angle
    std::vector<int> obstaclesPerAngle(DW_FREE_SPACE_RESOLUTION, 0);
    int skippedHigh = 0;
    int skippedVehicleZone = 0;
    int processedObstacles = 0;

    // For each obstacle point, update the minimum distance in its angular bin
    for (const auto& p : obstaclePoints) {
        // Skip high points (trees, buildings)
        if (p.z > 3.0f) {
        skippedHigh++;
        continue;
    }

    // Skip very close points (robot body)
    // if (std::abs(p.y) < 1.2f && std::abs(p.x) < 2.5f) {

    if (std::abs(p.y) < 2.2f && std::abs(p.x) < 3.5f) {
        skippedVehicleZone++;
        continue;
    }

float32_t distance = p.x * p.x + p.y * p.y;
int thetaId = static_cast<int>((std::atan2(p.y, p.x) + DW_FREE_PI) * 
   180.0f / DW_FREE_PI + 0.5f);
thetaId = thetaId % DW_FREE_SPACE_RESOLUTION;

obstaclesPerAngle[thetaId]++;

if (radialDistances[thetaId] > distance && distance > 1.0f) {
radialDistances[thetaId] = distance;
processedObstacles++;
}
}

// DIAGNOSTIC OUTPUT - Analyze angular sectors
std::cout << "\n[FreeSpace DIAGNOSTIC] Radial Distance Analysis:" << std::endl;
std::cout << "  Total obstacle points: " << obstaclePoints.size() << std::endl;
std::cout << "  Skipped (too high): " << skippedHigh << std::endl;
std::cout << "  Skipped (vehicle zone): " << skippedVehicleZone << std::endl;
std::cout << "  Processed obstacles: " << processedObstacles << std::endl;

// Analyze specific angular sectors
struct Sector {
std::string name;
int startAngle;
int endAngle;
};

std::vector<Sector> sectors = {
{"FRONT", 165, 195},      // -15° to +15° (forward)
{"RIGHT", 255, 285},      // 75° to 105° (right)
{"BACK", 345, 15},        // 165° to 195° (backward)
{"LEFT", 75, 105}         // -105° to -75° (left)
};

for (const auto& sector : sectors) {
int obstacleCount = 0;
float minDist = 50.0f;
float maxDist = 0.0f;
int emptyBins = 0;

for (int angle = sector.startAngle; angle <= sector.endAngle; angle++) {
int idx = angle % DW_FREE_SPACE_RESOLUTION;
obstacleCount += obstaclesPerAngle[idx];

float dist = std::sqrt(radialDistances[idx]);
if (radialDistances[idx] >= 2499.0f) {
emptyBins++;
} else {
minDist = std::min(minDist, dist);
maxDist = std::max(maxDist, dist);
}
}

std::cout << "  " << sector.name << " sector (" 
<< (sector.startAngle - 180) << "° to " << (sector.endAngle - 180) << "°):" << std::endl;
std::cout << "    Obstacles: " << obstacleCount << std::endl;
std::cout << "    Empty angular bins: " << emptyBins << "/31" << std::endl;
std::cout << "    Distance range: " << minDist << "m to " << maxDist << "m" << std::endl;
}
std::cout << std::endl;
}



void FreeSpaceDW::FilterFreeSpace(const std::vector<float32_t>& radialInput,
                                  std::vector<float32_t>& freeSpacePoints)
{
    const float32_t pixelSize = 0.2f;
    const float32_t deltaDInR = 0.13f;
    const float32_t deltaR = 0.15f;
    const int gridSize = static_cast<int>(100.0f / pixelSize);
    
    // Create occupancy grids using Eigen matrices (like original)
    Eigen::MatrixXi srcGrid = Eigen::MatrixXi::Zero(gridSize, gridSize);
    Eigen::MatrixXi dstGrid = Eigen::MatrixXi::Zero(gridSize, gridSize);
    
    // Precompute angular deltas for different radii
    std::vector<float32_t> deltaTheta;
    for (float32_t r = 0.0001f; r < 50.0f; r += deltaR) {
        deltaTheta.push_back(deltaDInR / r);
    }
    
    // Fill source grid with free space boundaries
    for (int i = 0; i < DW_FREE_SPACE_RESOLUTION; i++) {
        float32_t r = std::min({radialInput[i], 
                               radialInput[(i + 1) % DW_FREE_SPACE_RESOLUTION],
                               radialInput[(i - 1 + DW_FREE_SPACE_RESOLUTION) % DW_FREE_SPACE_RESOLUTION]});
        r = std::sqrt(r);
        
        int k = 0;
        for (float32_t j = 0.0f; j < r - 0.5f; j += deltaR) {
            if (k >= deltaTheta.size()) break;
            float32_t dt = deltaTheta[k++];
            float32_t theta = (i - 180) * DW_FREE_PI / 180.0f;
            
            for (float32_t t = theta - 0.01f; t < theta + 0.01f; t += dt) {
                float32_t x = j * std::cos(t);
                float32_t y = j * std::sin(t);
                int m = static_cast<int>((50.0f - x) / pixelSize);
                int n = static_cast<int>((50.0f - y) / pixelSize);
                
                if (m >= 0 && m < gridSize && n >= 0 && n < gridSize) {
                    srcGrid(m, n) = 1;
                }
            }
        }
    }
    
    // Apply morphological filter to expand free space
    for (int i = 0; i < DW_FREE_SPACE_RESOLUTION; i++) {
        for (float32_t j = 0.0f; j < 49.0f; j += deltaR) {
            float32_t x = j * std::cos((i - 180) * DW_FREE_PI / 180.0f);
            float32_t y = j * std::sin((i - 180) * DW_FREE_PI / 180.0f);
            int m = static_cast<int>((50.0f - x) / pixelSize);
            int n = static_cast<int>((50.0f - y) / pixelSize);
            
            if (m < 0 || m >= gridSize || n < 0 || n >= gridSize) continue;
            
            int theta = static_cast<int>(std::atan2(y, x) * 180.0f / DW_FREE_PI + 180.0f + 0.5f);
            theta = theta % DW_FREE_SPACE_RESOLUTION;
            
            float32_t r = std::min({radialInput[theta],
                                   radialInput[(theta + 1) % DW_FREE_SPACE_RESOLUTION],
                                   radialInput[(theta - 1 + DW_FREE_SPACE_RESOLUTION) % DW_FREE_SPACE_RESOLUTION]});
            
            if (r > j * j + 1.0f) {
                // Check filter
                int result = 0;
                for (int k = 0; k < 28; k++) {
                    int fm = m + FILTER_X[k];
                    int fn = n + FILTER_Y[k];
                    if (fm >= 0 && fm < gridSize && fn >= 0 && fn < gridSize) {
                        result += srcGrid(fm, fn);
                    }
                }
                
                if (result < 28) break;
                
                // Mark as free space with dilation
                for (int k = 0; k < 89; k++) {
                    int dm = m + ALL_X[k];
                    int dn = n + ALL_Y[k];
                    if (dm >= 0 && dm < gridSize && dn >= 0 && dn < gridSize) {
                        dstGrid(dm, dn) = std::max(1, dstGrid(dm, dn));
                    }
                }
                dstGrid(m, n) = 2;
            }
        }
    }
    
    // Convert grid to point cloud
    freeSpacePoints.clear();
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < gridSize; j++) {
            if (dstGrid(i, j) > 0) {
                float32_t x = (100.0f - i * pixelSize) - 50.0f;
                float32_t y = (100.0f - j * pixelSize) - 50.0f;
                freeSpacePoints.push_back(x);
                freeSpacePoints.push_back(y);
                freeSpacePoints.push_back(255.0f); // Intensity marker
            }
        }
    }
}

void FreeSpaceDW::GetFreeSpaceRadial(std::vector<float32_t>& radialDistances) const
{
    radialDistances = m_radialDistances;
}

void FreeSpaceDW::GetStatistics(uint32_t& numGroundPoints,
                               uint32_t& numObstaclePoints,
                               uint32_t& numFreeSpacePoints) const
{
    numGroundPoints = m_numGroundPoints;
    numObstaclePoints = m_numObstaclePoints;
    numFreeSpacePoints = m_numFreeSpacePoints;
}