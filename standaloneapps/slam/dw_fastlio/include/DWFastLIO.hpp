/////////////////////////////////////////////////////////////////////////////////////////
// DriveWorks Fast-LIO Wrapper
// Pure DriveWorks implementation of Fast-LIO SLAM
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_FAST_LIO_HPP_
#define DW_FAST_LIO_HPP_

#include <memory>
#include <vector>
#include <map>
#include <mutex>
#include <thread>
#include <deque>
#include <string>

#include <dw/sensors/lidar/LidarTypes.h>
#include <dw/sensors/imu/IMUTypes.h>
#include <dw/sensors/gps/GPS.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "TypeConverters.hpp"

namespace dw_slam {

/**
 * @brief Configuration for DWFastLIO
 */
struct DWFastLIOConfig {
    // Extrinsics
    Eigen::Matrix4d lidar_to_imu_transform = Eigen::Matrix4d::Identity();
    
    // Fast-LIO parameters
    int filter_num = 1;
    int max_point_num = -1;
    double scan_period = 0.1; // 10Hz lidar
    bool enable_undistort = true;
    
    // Processing parameters
    double voxel_size = 0.5;          // voxel grid filter size
    int ivox_nearby_type = 18;        // iVox search type
    double plane_threshold = 0.1;      // plane detection threshold
    
    // Mapping parameters  
    double keyframe_density = 0.2;
    double keyframe_delta_trans = 0.25; // meters
    double keyframe_delta_angle = 0.1;  // radians
    
    // G2O parameters
    std::string solver_type = "lm_var_cholmod";
    int optimization_iterations = 512;
};

/**
 * @brief Main Fast-LIO class integrating with DriveWorks sensors
 */
class DWFastLIO {
public:
    DWFastLIO();
    ~DWFastLIO();
    
    /**
     * @brief Initialize the SLAM system
     */
    bool initialize(const DWFastLIOConfig& config);
    
    /**
     * @brief Process IMU frame from DriveWorks
     */
    void feedIMU(const dwIMUFrame& imuFrame, dwTime_t timestamp);
    
    /**
     * @brief Process GPS frame from DriveWorks
     */
    void feedGPS(const dwGPSFrame& gpsFrame, const dwIMUFrame& imuFrame, dwTime_t timestamp);
    
    /**
     * @brief Process LiDAR scan from DriveWorks
     */
    void feedLiDAR(const dwLidarPointXYZI* points, size_t numPoints, dwTime_t timestamp);
    
    /**
     * @brief Get current pose estimate
     * @return 4x4 transformation matrix (world frame)
     */
    Eigen::Matrix4d getCurrentPose() const;
    
    /**
     * @brief Get odometry trajectory
     */
    std::vector<Eigen::Matrix4d> getTrajectory() const;
    
    /**
     * @brief Get map point cloud
     */
    pcl::PointCloud<pcl::PointXYZI>::Ptr getMapCloud() const;
    
    /**
     * @brief Save map to file
     */
    bool saveMap(const std::string& filepath);
    
    /**
     * @brief Load map from file (for localization)
     */
    bool loadMap(const std::string& filepath);
    
    /**
     * @brief Check if system is initialized
     */
    bool isInitialized() const { return initialized_; }
    
    /**
     * @brief Check if map building mode (true) or localization mode (false)
     */
    bool isMappingMode() const { return mapping_mode_; }
    
    /**
     * @brief Set to mapping or localization mode
     */
    void setMappingMode(bool mapping);

private:
    // Fast-LIO integration
    void processScan();
    void optimizeGraph();
    
    // Data synchronization
    void syncMeasurements();
    
    // Helper functions
    bool checkIMUInitialized();
    void addKeyframe(const Eigen::Matrix4d& pose, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);
    
private:
    // Configuration
    DWFastLIOConfig config_;
    bool initialized_;
    bool mapping_mode_;
    
    // Sensor data buffers (thread-safe)
    std::mutex imu_mutex_;
    std::deque<std::pair<::ImuType, dwTime_t>> imu_buffer_;
    
    std::mutex gps_mutex_;
    std::deque<std::pair<::RTKType, dwTime_t>> gps_buffer_;
    
    std::mutex lidar_mutex_;
    std::deque<std::pair<pcl::PointCloud<pcl::PointXYZI>::Ptr, dwTime_t>> lidar_buffer_;
    
    // State
    Eigen::Matrix4d current_pose_;
    std::vector<Eigen::Matrix4d> trajectory_;
    
    // Map
    pcl::PointCloud<pcl::PointXYZI>::Ptr map_cloud_;
    
    // Processing threads
    std::unique_ptr<std::thread> processing_thread_;
    bool thread_running_;
};

} // namespace dw_slam

#endif // DW_FAST_LIO_HPP_

