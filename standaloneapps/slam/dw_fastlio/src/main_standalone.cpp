/////////////////////////////////////////////////////////////////////////////////////////
// Standalone DriveWorks SLAM - Minimal example without GUI framework
// This version works without the DriveWorksSample framework
/////////////////////////////////////////////////////////////////////////////////////////

#include <csignal>
#include <iostream>
#include <memory>
#include <chrono>
#include <thread>

// DriveWorks
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/lidar/Lidar.h>
#include <dw/sensors/imu/IMU.h>
#include <dw/core/base/Status.h>

// Standard library
#include <iomanip>
#include <fstream>

// SLAM
#include "DWFastLIO.hpp"

#define CHECK_DW_ERROR(x) \
    do { \
        dwStatus result = x; \
        if (result != DW_SUCCESS) { \
            std::cerr << "DriveWorks Error: " << dwGetStatusName(result) << std::endl; \
            exit(1); \
        } \
    } while(0)

static bool gRun = true;

void sig_int_handler(int) {
    gRun = false;
}

// Console logger callback
void logCallback(dwContextHandle_t /*context*/, dwLoggerVerbosity /*verbosity*/,
                 dwTime_t /*timestamp*/, char const* msg)
{
    std::cout << msg << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <lidar_file> <imu_file> <voxel_size> <output_map>" << std::endl;
        std::cout << "Example: " << argv[0] << " lidar.bin imu.bin 0.5 map.pcd" << std::endl;
        return -1;
    }
    
    std::string lidarFile = argv[1];
    std::string imuFile = argv[2];
    float voxelSize = std::stof(argv[3]);
    std::string outputMap = argv[4];
    
    // Setup signal handler
    struct sigaction action = {};
    action.sa_handler = sig_int_handler;
    sigaction(SIGINT, &action, NULL);
    
    std::cout << "============================================" << std::endl;
    std::cout << "DriveWorks Fast-LIO SLAM (Standalone)" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "LiDAR file: " << lidarFile << std::endl;
    std::cout << "IMU file: " << imuFile << std::endl;
    std::cout << "Voxel size: " << voxelSize << std::endl;
    std::cout << "Output map: " << outputMap << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Initialize DriveWorks
    dwContextHandle_t context = DW_NULL_HANDLE;
    dwSALHandle_t sal = DW_NULL_HANDLE;
    
    dwLogger_initialize(logCallback);
    dwLogger_setLogLevel(DW_LOG_INFO);
    
    CHECK_DW_ERROR(dwInitialize(&context, DW_VERSION, nullptr));
    CHECK_DW_ERROR(dwSAL_initialize(&sal, context));
    
    // Create LiDAR sensor
    dwSensorHandle_t lidarSensor = DW_NULL_HANDLE;
    {
        dwSensorParams params{};
        std::string lidarParams = "file=" + lidarFile;
        params.protocol = "lidar.virtual";
        params.parameters = lidarParams.c_str();
        
        if (dwSAL_createSensor(&lidarSensor, params, sal) != DW_SUCCESS) {
            std::cerr << "Failed to create LiDAR sensor" << std::endl;
            return -1;
        }
        
        dwSensor_start(lidarSensor);
    }
    
    // Create IMU sensor
    dwSensorHandle_t imuSensor = DW_NULL_HANDLE;
    {
        dwSensorParams params{};
        std::string imuParams = "file=" + imuFile;
        params.protocol = "imu.virtual";
        params.parameters = imuParams.c_str();
        
        if (dwSAL_createSensor(&imuSensor, params, sal) != DW_SUCCESS) {
            std::cerr << "Failed to create IMU sensor" << std::endl;
            return -1;
        }
        
        dwSensor_start(imuSensor);
    }
    
    // Get LiDAR properties
    dwLidarProperties lidarProps{};
    dwSensorLidar_getProperties(&lidarProps, lidarSensor);
    
    // Initialize SLAM
    dw_slam::DWFastLIOConfig slamConfig;
    slamConfig.scan_period = 1.0 / lidarProps.spinFrequency;
    slamConfig.voxel_size = voxelSize;
    
    dw_slam::DWFastLIO slam;
    if (!slam.initialize(slamConfig)) {
        std::cerr << "Failed to initialize SLAM" << std::endl;
        return -1;
    }
    
    std::cout << "\n[SLAM] Initialized successfully" << std::endl;
    std::cout << "[SLAM] LiDAR frequency: " << lidarProps.spinFrequency << " Hz" << std::endl;
    std::cout << "[SLAM] Scan period: " << slamConfig.scan_period << " s" << std::endl;
    std::cout << "\nProcessing... Press Ctrl+C to stop and save map.\n" << std::endl;
    
    // Allocate point cloud buffer
    size_t pointCloudCapacity = lidarProps.pointsPerSecond * 2;
    std::unique_ptr<dwLidarPointXYZI[]> pointCloudBuffer(new dwLidarPointXYZI[pointCloudCapacity]);
    size_t pointCloudSize = 0;
    
    dwIMUFrame latestIMU{};
    bool hasIMU = false;
    
    int frameCount = 0;
    auto startTime = std::chrono::steady_clock::now();
    
    // Main processing loop
    while (gRun) {
        // Read IMU data
        {
            dwIMUFrame imuFrame;
            dwStatus status = dwSensorIMU_readFrame(&imuFrame, 0, imuSensor);
            
            if (status == DW_SUCCESS) {
                latestIMU = imuFrame;
                hasIMU = true;
                slam.feedIMU(imuFrame, imuFrame.hostTimestamp);
            }
        }
        
        // Read LiDAR packets
        if (hasIMU) {
            const dwLidarDecodedPacket* packet;
            dwStatus status = dwSensorLidar_readPacket(&packet, 100000, lidarSensor);
            
            if (status == DW_SUCCESS) {
                // Accumulate points
                if (pointCloudSize + packet->nPoints <= pointCloudCapacity) {
                    memcpy(&pointCloudBuffer[pointCloudSize], packet->pointsXYZI,
                           packet->nPoints * sizeof(dwLidarPointXYZI));
                    pointCloudSize += packet->nPoints;
                }
                
                // Process complete scan
                if (packet->scanComplete && pointCloudSize > 0) {
                    slam.feedLiDAR(pointCloudBuffer.get(), pointCloudSize, packet->hostTimestamp);
                    
                    frameCount++;
                    auto pose = slam.getCurrentPose();
                    
                    std::cout << "\r[Frame " << frameCount << "] "
                              << "Points: " << pointCloudSize << " | "
                              << "Position: ("
                              << std::fixed << std::setprecision(2)
                              << pose(0,3) << ", "
                              << pose(1,3) << ", "
                              << pose(2,3) << ")    " << std::flush;
                    
                    pointCloudSize = 0;
                }
                
                dwSensorLidar_returnPacket(packet, lidarSensor);
            }
            else if (status == DW_END_OF_STREAM) {
                std::cout << "\n\n[SLAM] Reached end of recording" << std::endl;
                gRun = false;
            }
        }
        
        // Small sleep to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    
    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    
    std::cout << "\n\n============================================" << std::endl;
    std::cout << "Processing complete!" << std::endl;
    std::cout << "Frames processed: " << frameCount << std::endl;
    std::cout << "Duration: " << duration << " seconds" << std::endl;
    std::cout << "Average FPS: " << (duration > 0 ? frameCount / duration : 0) << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Save map
    std::cout << "\nSaving map to: " << outputMap << std::endl;
    if (slam.saveMap(outputMap)) {
        auto mapCloud = slam.getMapCloud();
        std::cout << "Map saved successfully!" << std::endl;
        std::cout << "Total map points: " << mapCloud->size() << std::endl;
    } else {
        std::cerr << "Failed to save map" << std::endl;
    }
    
    // Save trajectory
    std::string trajFile = outputMap.substr(0, outputMap.find_last_of('.')) + "_trajectory.txt";
    std::ofstream trajOut(trajFile);
    auto trajectory = slam.getTrajectory();
    for (const auto& pose : trajectory) {
        trajOut << pose(0,3) << " " << pose(1,3) << " " << pose(2,3) << "\n";
    }
    trajOut.close();
    std::cout << "Trajectory saved to: " << trajFile << std::endl;
    
    // Cleanup
    dwSensor_stop(lidarSensor);
    dwSensor_stop(imuSensor);
    dwSAL_releaseSensor(lidarSensor);
    dwSAL_releaseSensor(imuSensor);
    dwSAL_release(sal);
    dwRelease(context);
    dwLogger_release();
    
    std::cout << "\nDone!" << std::endl;
    return 0;
}

