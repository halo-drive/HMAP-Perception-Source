/////////////////////////////////////////////////////////////////////////////////////////
// DriveWorks Fast-LIO Implementation
/////////////////////////////////////////////////////////////////////////////////////////

#include "../include/DWFastLIO.hpp"
#include <iostream>
#include <fstream>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

// FastLIO API from LSD (C++ linkage)
// Matching actual signatures from libfast_lio.so
extern int  fastlio_init(std::vector<double> &extT, std::vector<double>& extR, 
                         int filter_num, int max_point_num, double scan_period, bool undistort);
extern bool fastlio_is_init();
extern void fastlio_imu_enqueue(::ImuType imu);  // Global scope ImuType
extern void fastlio_ins_enqueue(bool rtk_valid, ::RTKType ins);
extern void fastlio_pcl_enqueue(std::shared_ptr<::PointCloudAttr>& points);
extern bool fastlio_main();
extern void fastlio_odometry(Eigen::Matrix4d &odom_s, Eigen::Matrix4d &odom_e);
extern std::vector<double> fastlio_state();

namespace dw_slam {

DWFastLIO::DWFastLIO()
    : initialized_(false)
    , mapping_mode_(true)
    , thread_running_(false)
{
    current_pose_ = Eigen::Matrix4d::Identity();
    map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
}

DWFastLIO::~DWFastLIO() {
    thread_running_ = false;
    if (processing_thread_ && processing_thread_->joinable()) {
        processing_thread_->join();
    }
}

bool DWFastLIO::initialize(const DWFastLIOConfig& config) {
    config_ = config;
    
    // Extract extrinsics for Fast-LIO
    Eigen::Matrix4d& extr = config_.lidar_to_imu_transform;
    std::vector<double> extT = {extr(0, 3), extr(1, 3), extr(2, 3)};
    std::vector<double> extR = {
        extr(0, 0), extr(0, 1), extr(0, 2),
        extr(1, 0), extr(1, 1), extr(1, 2),
        extr(2, 0), extr(2, 1), extr(2, 2)
    };
    
    // Initialize Fast-LIO backend
    int result = fastlio_init(extT, extR, 
                              config_.filter_num,
                              config_.max_point_num,
                              config_.scan_period,
                              config_.enable_undistort);
    
    if (result != 0) {
        std::cerr << "[DWFastLIO] Failed to initialize Fast-LIO backend" << std::endl;
        return false;
    }
    
    initialized_ = true;
    
    // Start processing thread
    thread_running_ = true;
    processing_thread_.reset(new std::thread(&DWFastLIO::processScan, this));
    
    std::cout << "[DWFastLIO] Initialized successfully" << std::endl;
    return true;
}

void DWFastLIO::feedIMU(const dwIMUFrame& dwImu, dwTime_t timestamp) {
    if (!initialized_) return;
    
    ::ImuType lsdImu;
    dwIMUToLSD(dwImu, lsdImu, timestamp);
    
    {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        imu_buffer_.push_back({lsdImu, timestamp});
        
        // Keep buffer size reasonable (last 1000 IMU samples)
        if (imu_buffer_.size() > 1000) {
            imu_buffer_.pop_front();
        }
    }
    
    // Feed to Fast-LIO immediately for IMU integration
    fastlio_imu_enqueue(lsdImu);
}

void DWFastLIO::feedGPS(const dwGPSFrame& gpsFrame, const dwIMUFrame& imuFrame, dwTime_t timestamp) {
    if (!initialized_) return;
    
    ::RTKType lsdRTK;
    dwGPSToLSD(gpsFrame, imuFrame, lsdRTK, timestamp);
    
    {
        std::lock_guard<std::mutex> lock(gps_mutex_);
        gps_buffer_.push_back({lsdRTK, timestamp});
        
        // Keep buffer size reasonable
        if (gps_buffer_.size() > 100) {
            gps_buffer_.pop_front();
        }
    }
}

void DWFastLIO::feedLiDAR(const dwLidarPointXYZI* points, size_t numPoints, dwTime_t timestamp) {
    if (!initialized_) return;
    
    // Convert DriveWorks points to PCL
    pcl::PointCloud<pcl::PointXYZI>::Ptr pclCloud(new pcl::PointCloud<pcl::PointXYZI>());
    dwPointCloudToPCL(points, numPoints, pclCloud);
    
    // Set timestamp in microseconds
    pclCloud->header.stamp = timestamp;
    
    {
        std::lock_guard<std::mutex> lock(lidar_mutex_);
        lidar_buffer_.push_back({pclCloud, timestamp});
        
        // Keep only last few scans in buffer
        if (lidar_buffer_.size() > 10) {
            lidar_buffer_.pop_front();
        }
    }
    
    std::cout << "[DWFastLIO] Received LiDAR scan: " << numPoints << " points" << std::endl;
}

void DWFastLIO::processScan() {
    while (thread_running_) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
        dwTime_t timestamp;
        
        // Get next lidar scan
        {
            std::lock_guard<std::mutex> lock(lidar_mutex_);
            if (lidar_buffer_.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            
            auto scan = lidar_buffer_.front();
            lidar_buffer_.pop_front();
            cloud = scan.first;
            timestamp = scan.second;
        }
        
        // Check if we have IMU data synchronized
        bool hasIMU = false;
        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            hasIMU = !imu_buffer_.empty();
        }
        
        if (!hasIMU) {
            std::cout << "[DWFastLIO] Waiting for IMU data..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        // Feed point cloud to Fast-LIO
        // Create PointCloudAttr matching LSD structure
        std::shared_ptr<::PointCloudAttr> cloudAttr = std::make_shared<::PointCloudAttr>();
        cloudAttr->cloud = cloud;
        cloudAttr->T = Eigen::Matrix4d::Identity();
        
        // Enqueue to Fast-LIO
        fastlio_pcl_enqueue(cloudAttr);
        
        // Process through Fast-LIO
        bool success = fastlio_main();
        
        if (success) {
            // Get odometry
            Eigen::Matrix4d odom_start = Eigen::Matrix4d::Identity();
            Eigen::Matrix4d odom_end = Eigen::Matrix4d::Identity();
            fastlio_odometry(odom_start, odom_end);
            
            // Update current pose
            current_pose_ = odom_end;
            trajectory_.push_back(current_pose_);
            
            std::cout << "[DWFastLIO] Pose updated - Position: " 
                      << current_pose_(0,3) << ", "
                      << current_pose_(1,3) << ", "  
                      << current_pose_(2,3) << std::endl;
                      
            // Add to map if in mapping mode
            if (mapping_mode_) {
                // Transform cloud to world frame
                pcl::PointCloud<pcl::PointXYZI>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZI>());
                pcl::transformPointCloud(*cloud, *transformed, current_pose_);
                
                // Add to map with voxel filtering
                *map_cloud_ += *transformed;
                
                pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
                voxel_filter.setLeafSize(config_.voxel_size, config_.voxel_size, config_.voxel_size);
                voxel_filter.setInputCloud(map_cloud_);
                voxel_filter.filter(*map_cloud_);
            }
        }
    }
}

Eigen::Matrix4d DWFastLIO::getCurrentPose() const {
    return current_pose_;
}

std::vector<Eigen::Matrix4d> DWFastLIO::getTrajectory() const {
    return trajectory_;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr DWFastLIO::getMapCloud() const {
    return map_cloud_;
}

bool DWFastLIO::saveMap(const std::string& filepath) {
    if (!map_cloud_ || map_cloud_->empty()) {
        std::cerr << "[DWFastLIO] No map to save" << std::endl;
        return false;
    }
    
    pcl::io::savePCDFileBinary(filepath, *map_cloud_);
    std::cout << "[DWFastLIO] Saved map: " << map_cloud_->size() << " points to " << filepath << std::endl;
    return true;
}

bool DWFastLIO::loadMap(const std::string& filepath) {
    map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    
    if (pcl::io::loadPCDFile(filepath, *map_cloud_) == -1) {
        std::cerr << "[DWFastLIO] Failed to load map from " << filepath << std::endl;
        return false;
    }
    
    std::cout << "[DWFastLIO] Loaded map: " << map_cloud_->size() << " points from " << filepath << std::endl;
    mapping_mode_ = false; // Switch to localization mode
    return true;
}

void DWFastLIO::setMappingMode(bool mapping) {
    mapping_mode_ = mapping;
    std::cout << "[DWFastLIO] Mode: " << (mapping ? "MAPPING" : "LOCALIZATION") << std::endl;
}

} // namespace dw_slam

