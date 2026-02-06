/////////////////////////////////////////////////////////////////////////////////////////
// DriveWorks SLAM - Type Converters
// Converts between DriveWorks and PCL data types
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_SLAM_TYPE_CONVERTERS_HPP_
#define DW_SLAM_TYPE_CONVERTERS_HPP_

#include <cmath>
#include <cstring>
#include <string>
#include <algorithm>

#include <dw/sensors/lidar/LidarTypes.h>
#include <dw/sensors/imu/IMUTypes.h>
#include <dw/sensors/gps/GPS.h>
#include <dw/core/base/Types.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

// ===== LSD-compatible types (global scope) =====
// When DW_FASTLIO_USE_GRAPH_BACKEND is set, these are defined by mapping_types.h
// (included via slam_base.h before this header). Otherwise we define them here so
// we can build without the full LSD stack and still convert DriveWorks <-> Fast-LIO.
#ifndef DW_FASTLIO_USE_GRAPH_BACKEND
struct ImuType {
    double stamp;
    Eigen::Vector3d acc;
    Eigen::Vector3d gyr;
    Eigen::Quaterniond rot;
};

struct RTKType {
    uint64_t timestamp;
    double heading, pitch, roll;
    double gyro_x, gyro_y, gyro_z;
    double acc_x, acc_y, acc_z;
    double latitude, longitude, altitude;
    double Ve, Vn, Vu;
    int status;
    std::string sensor;
    std::string state;
    int dimension;
    double precision;
    Eigen::Matrix4d T;
    Eigen::VectorXf mean;
};

struct PointAttr {
    int id;
    uint32_t stamp;
};

struct PointCloudAttr {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    std::vector<PointAttr> attr;
    Eigen::Matrix4d T;
};
#endif

namespace dw_slam {

// ===== Point Cloud Conversions =====

/**
 * @brief Convert DriveWorks LidarPointXYZI to PCL PointXYZI
 */
inline void dwPointToPCL(const dwLidarPointXYZI& dwPoint, pcl::PointXYZI& pclPoint)
{
    pclPoint.x = dwPoint.x;
    pclPoint.y = dwPoint.y;
    pclPoint.z = dwPoint.z;
    pclPoint.intensity = dwPoint.intensity;
}

/**
 * @brief Convert PCL PointXYZI to DriveWorks LidarPointXYZI
 */
inline void pclPointToDW(const pcl::PointXYZI& pclPoint, dwLidarPointXYZI& dwPoint)
{
    dwPoint.x = pclPoint.x;
    dwPoint.y = pclPoint.y;
    dwPoint.z = pclPoint.z;
    dwPoint.intensity = pclPoint.intensity;
}

/**
 * @brief Convert DriveWorks point cloud buffer to PCL PointCloud
 * @param dwPoints Array of DriveWorks points
 * @param numPoints Number of points in the array
 * @param pclCloud Output PCL point cloud
 */
inline void dwPointCloudToPCL(const dwLidarPointXYZI* dwPoints, 
                              size_t numPoints,
                              pcl::PointCloud<pcl::PointXYZI>::Ptr& pclCloud)
{
    // Pre-allocate to avoid reallocations during resize
    pclCloud->points.reserve(numPoints);
    pclCloud->points.resize(numPoints);
    pclCloud->width = numPoints;
    pclCloud->height = 1;
    pclCloud->is_dense = false;
    
    // Direct memory copy for x,y,z (12 bytes) then set intensity
    // This is faster than calling dwPointToPCL for each point
    for (size_t i = 0; i < numPoints; ++i) {
        pclCloud->points[i].x = dwPoints[i].x;
        pclCloud->points[i].y = dwPoints[i].y;
        pclCloud->points[i].z = dwPoints[i].z;
        pclCloud->points[i].intensity = dwPoints[i].intensity;
    }
}

/**
 * @brief Convert PCL PointCloud to DriveWorks point cloud buffer
 * @param pclCloud Input PCL point cloud
 * @param dwPoints Output array of DriveWorks points (must be pre-allocated)
 * @param maxPoints Maximum capacity of dwPoints array
 * @return Number of points actually converted
 */
inline size_t pclPointCloudToDW(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pclCloud,
                               dwLidarPointXYZI* dwPoints,
                               size_t maxPoints)
{
    size_t numPoints = std::min(pclCloud->points.size(), maxPoints);
    
    for (size_t i = 0; i < numPoints; ++i) {
        pclPointToDW(pclCloud->points[i], dwPoints[i]);
    }
    
    return numPoints;
}

// ===== IMU Data Conversions =====

/**
 * @brief Convert DriveWorks IMUFrame to LSD ImuType  
 */
inline void dwIMUToLSD(const dwIMUFrame& dwImu, ::ImuType& lsdImu, dwTime_t timestamp)
{
    // Timestamp: DriveWorks is documented as microseconds; if value > 1e15 assume nanoseconds.
    double stamp_sec = static_cast<double>(timestamp);
    if (stamp_sec > 1e15) {
        stamp_sec /= 1e9;  // nanoseconds -> seconds
    } else {
        stamp_sec /= 1e6;  // microseconds -> seconds
    }
    lsdImu.stamp = stamp_sec;
    
    // Acceleration (m/s²)
    lsdImu.acc = Eigen::Vector3d(dwImu.acceleration[0], 
                                  dwImu.acceleration[1], 
                                  dwImu.acceleration[2]);
    
    // Gyroscope (rad/s)
    lsdImu.gyr = Eigen::Vector3d(dwImu.turnrate[0],
                                  dwImu.turnrate[1],
                                  dwImu.turnrate[2]);
    
    // Orientation quaternion
    lsdImu.rot = Eigen::Quaterniond(dwImu.orientationQuaternion.w,
                                     dwImu.orientationQuaternion.x,
                                     dwImu.orientationQuaternion.y,
                                     dwImu.orientationQuaternion.z);
}

// ===== GPS/RTK Data Conversions =====

/**
 * @brief Convert DriveWorks GPSFrame to LSD RTKType
 */
inline void dwGPSToLSD(const dwGPSFrame& dwGPS, const dwIMUFrame& dwIMU, ::RTKType& lsdRTK, dwTime_t timestamp)
{
    lsdRTK.timestamp = timestamp;
    
    // Position
    lsdRTK.latitude = dwGPS.latitude;
    lsdRTK.longitude = dwGPS.longitude;
    lsdRTK.altitude = dwGPS.altitude;
    
    // Velocity
    lsdRTK.Ve = dwGPS.speed * std::cos(dwGPS.course);
    lsdRTK.Vn = dwGPS.speed * std::sin(dwGPS.course);
    lsdRTK.Vu = dwGPS.climb;
    
    // Orientation from IMU
    lsdRTK.heading = dwIMU.orientation[2]; // yaw
    lsdRTK.pitch = dwIMU.orientation[1];
    lsdRTK.roll = dwIMU.orientation[0];
    
    // Angular rates
    lsdRTK.gyro_x = dwIMU.turnrate[0];
    lsdRTK.gyro_y = dwIMU.turnrate[1];
    lsdRTK.gyro_z = dwIMU.turnrate[2];
    
    // Acceleration
    lsdRTK.acc_x = dwIMU.acceleration[0];
    lsdRTK.acc_y = dwIMU.acceleration[1];
    lsdRTK.acc_z = dwIMU.acceleration[2];
    
    // Status
    lsdRTK.status = (dwGPS.flags & DW_GPS_LAT) && (dwGPS.flags & DW_GPS_LON) ? 1 : 0;
    
    // Sensor source (LSD checks ins.sensor for "Wheel"; RTK/GPS use rtk_valid)
    lsdRTK.sensor = "RTK";
    
    // Transform matrix (identity for now, can be computed from orientation)
    lsdRTK.T = Eigen::Matrix4d::Identity();
    Eigen::Quaterniond q(dwIMU.orientationQuaternion.w,
                        dwIMU.orientationQuaternion.x,
                        dwIMU.orientationQuaternion.y,
                        dwIMU.orientationQuaternion.z);
    lsdRTK.T.topLeftCorner<3, 3>() = q.toRotationMatrix();
}

// ===== Pose/Transform Conversions =====

/**
 * @brief Convert Eigen::Matrix4d to dwTransformation3f
 */
inline void eigenToDWTransform(const Eigen::Matrix4d& eigen, dwTransformation3f& dwTrans)
{
    // Translation
    dwTrans.array[0] = static_cast<float32_t>(eigen(0, 3));
    dwTrans.array[1] = static_cast<float32_t>(eigen(1, 3));
    dwTrans.array[2] = static_cast<float32_t>(eigen(2, 3));
    
    // Rotation matrix (row-major in DW, column-major in Eigen)
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            dwTrans.array[3 + row * 4 + col] = static_cast<float32_t>(eigen(row, col));
        }
    }
}

/**
 * @brief Convert dwTransformation3f to Eigen::Matrix4d
 */
inline void dwTransformToEigen(const dwTransformation3f& dwTrans, Eigen::Matrix4d& eigen)
{
    eigen = Eigen::Matrix4d::Identity();
    
    // Translation
    eigen(0, 3) = dwTrans.array[0];
    eigen(1, 3) = dwTrans.array[1];
    eigen(2, 3) = dwTrans.array[2];
    
    // Rotation
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            eigen(row, col) = dwTrans.array[3 + row * 4 + col];
        }
    }
}

} // namespace dw_slam

#endif // DW_SLAM_TYPE_CONVERTERS_HPP_

