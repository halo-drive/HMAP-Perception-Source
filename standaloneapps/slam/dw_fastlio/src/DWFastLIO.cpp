/////////////////////////////////////////////////////////////////////////////////////////
// DriveWorks Fast-LIO Implementation
/////////////////////////////////////////////////////////////////////////////////////////

#include "../include/DWFastLIO.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cstring>   // for strlen()
#include <cstdio>    // for FILE, fprintf
#include <cmath>     // for std::isnan, std::isinf
#include <unistd.h>  // for write(), fsync()
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
    std::cout << "[DWFastLIO] Calling fastlio_init()..." << std::flush << std::endl;
    int result = fastlio_init(extT, extR, 
                              config_.filter_num,
                              config_.max_point_num,
                              config_.scan_period,
                              config_.enable_undistort);
    
    std::cout << "[DWFastLIO] fastlio_init() returned: " << result << std::flush << std::endl;
    
    if (result != 0) {
        std::cerr << "[DWFastLIO] Failed to initialize Fast-LIO backend (return code: " << result << ")" << std::endl;
        return false;
    }
    
    // Verify initialization
    bool is_init = fastlio_is_init();
    std::cout << "[DWFastLIO] fastlio_is_init() returned: " << (is_init ? "true" : "false") << std::flush << std::endl;
    
    if (!is_init) {
        std::cerr << "[DWFastLIO] WARNING: fastlio_init() succeeded but fastlio_is_init() returns false!" << std::endl;
        // Continue anyway - might be a timing issue
    }
    
    initialized_ = true;
    
    // Start processing thread
    thread_running_ = true;
    processing_thread_.reset(new std::thread(&DWFastLIO::processScan, this));
    
    std::cout << "[DWFastLIO] Initialized successfully" << std::endl;
    return true;
}

void DWFastLIO::feedIMU(const dwIMUFrame& dwImu, dwTime_t timestamp) {
    if (!initialized_) {
        static bool warned = false;
        if (!warned) {
            std::cout << "[DWFastLIO] WARNING: feedIMU called but not initialized!" << std::endl;
            warned = true;
        }
        return;
    }
    
    ::ImuType lsdImu;
    dwIMUToLSD(dwImu, lsdImu, timestamp);
    
    size_t imu_buffer_size = 0;
    {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        imu_buffer_.push_back({lsdImu, timestamp});
        
        // Aggressively limit buffer size (last 500 IMU samples)
        while (imu_buffer_.size() > 500) {
            imu_buffer_.pop_front();
        }
        imu_buffer_size = imu_buffer_.size();
    }
    
    // Log first few IMU feeds
    static int imu_count = 0;
    if (++imu_count <= 5) {
        std::cout << "[DWFastLIO] feedIMU called #" << imu_count 
                  << " (buffer size: " << imu_buffer_size << ")" << std::endl;
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
        
        // Aggressively limit buffer size (last 50 GPS samples)
        while (gps_buffer_.size() > 50) {
            gps_buffer_.pop_front();
        }
    }
}

__attribute__((noinline)) void DWFastLIO::testFunction() const {
    static int callCount = 0;
    callCount++;
    
    // Write to a file directly - this should always work
    FILE* f = fopen("/tmp/dwfastlio_debug.log", "a");
    if (f) {
        fprintf(f, "[DWFastLIO::testFunction] CALLED #%d, this=%p\n", callCount, (void*)this);
        fflush(f);
        fclose(f);
    }
    
    // Also try std::cout - this should match the grep pattern
    std::cout << "[DWFastLIO] [testFunction] CALLED #" << callCount << ", this=" << (void*)this << std::flush << std::endl;
}

__attribute__((noinline)) void DWFastLIO::feedLiDAR(const dwLidarPointXYZI* points, size_t numPoints, dwTime_t timestamp) {
    static int callCount = 0;
    callCount++;
    
    // Write to file first - this should always work
    FILE* f = fopen("/tmp/dwfastlio_debug.log", "a");
    if (f) {
        fprintf(f, "[DWFastLIO::feedLiDAR] ENTRY #%d: numPoints=%zu, this=%p\n", callCount, numPoints, (void*)this);
        fflush(f);
        fclose(f);
    }
    
    // Also log to stdout with grep-friendly format
    std::cout << "[DWFastLIO] [feedLiDAR] ENTRY #" << callCount 
              << ": numPoints=" << numPoints << ", this=" << (void*)this << std::flush << std::endl;
    
    // Log to file after ENTRY
    f = fopen("/tmp/dwfastlio_debug.log", "a");
    if (f) {
        fprintf(f, "[DWFastLIO::feedLiDAR] After ENTRY log, checking 'this' pointer\n");
        fflush(f);
        fclose(f);
    }
    
    // Check if 'this' pointer is valid
    if (this == nullptr) {
        const char* err = "[DWFastLIO::feedLiDAR] ERROR: 'this' pointer is NULL!\n";
        write(2, err, strlen(err));
        return;
    }
    
    // Log to file before input validation
    f = fopen("/tmp/dwfastlio_debug.log", "a");
    if (f) {
        fprintf(f, "[DWFastLIO::feedLiDAR] About to validate inputs\n");
        fflush(f);
        fclose(f);
    }
    
    // Validate inputs immediately
    if (!points) {
        f = fopen("/tmp/dwfastlio_debug.log", "a");
        if (f) {
            fprintf(f, "[DWFastLIO::feedLiDAR] ERROR: points is NULL!\n");
            fclose(f);
        }
        return;
    }
    if (numPoints == 0) {
        f = fopen("/tmp/dwfastlio_debug.log", "a");
        if (f) {
            fprintf(f, "[DWFastLIO::feedLiDAR] ERROR: numPoints is 0!\n");
            fclose(f);
        }
        return;
    }
    
    // Log to file after input validation
    f = fopen("/tmp/dwfastlio_debug.log", "a");
    if (f) {
        fprintf(f, "[DWFastLIO::feedLiDAR] Inputs validated, checking initialized_\n");
        fflush(f);
        fclose(f);
    }
    
    // Check initialized_ - this accesses a member variable
    f = fopen("/tmp/dwfastlio_debug.log", "a");
    if (f) {
        fprintf(f, "[DWFastLIO::feedLiDAR] About to check initialized_ (value=%d)\n", initialized_ ? 1 : 0);
        fflush(f);
        fclose(f);
    }
    
    if (!initialized_) {
        f = fopen("/tmp/dwfastlio_debug.log", "a");
        if (f) {
            fprintf(f, "[DWFastLIO::feedLiDAR] Not initialized, returning\n");
            fclose(f);
        }
        static bool warned = false;
        if (!warned) {
            std::cout << "[DWFastLIO] WARNING: feedLiDAR called but not initialized!" << std::endl;
            warned = true;
        }
        return;
    }
    
    f = fopen("/tmp/dwfastlio_debug.log", "a");
    if (f) {
        fprintf(f, "[DWFastLIO::feedLiDAR] Initialized check passed, starting conversion\n");
        fflush(f);
        fclose(f);
    }
    
    static int scanCount = 0;
    scanCount++;
    std::cout << "[DWFastLIO] [feedLiDAR] Processing scan #" << scanCount 
              << ": numPoints=" << numPoints << ", timestamp=" << timestamp << std::flush << std::endl;
    
    // Log immediately to see if function is called - flush to ensure it appears
    f = fopen("/tmp/dwfastlio_debug.log", "a");
    if (f) {
        fprintf(f, "[DWFastLIO::feedLiDAR] About to create PCL cloud\n");
        fflush(f);
        fclose(f);
    }
    
    auto convertStart = std::chrono::steady_clock::now();
    std::cout << "[DWFastLIO] [feedLiDAR] Converting " << numPoints << " points..." << std::flush << std::endl;
    
    // Convert DriveWorks points to PCL - optimized for speed
    // Pre-allocate to avoid reallocations during resize
    f = fopen("/tmp/dwfastlio_debug.log", "a");
    if (f) {
        fprintf(f, "[DWFastLIO::feedLiDAR] Creating PCL PointCloud object\n");
        fflush(f);
        fclose(f);
    }
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr pclCloud(new pcl::PointCloud<pcl::PointXYZI>());
    
    f = fopen("/tmp/dwfastlio_debug.log", "a");
    if (f) {
        fprintf(f, "[DWFastLIO::feedLiDAR] PCL cloud created, reserving %zu points\n", numPoints);
        fflush(f);
        fclose(f);
    }
    
    pclCloud->points.reserve(numPoints);
    
    f = fopen("/tmp/dwfastlio_debug.log", "a");
    if (f) {
        fprintf(f, "[DWFastLIO::feedLiDAR] About to call dwPointCloudToPCL()\n");
        fflush(f);
        fclose(f);
    }
    
    dwPointCloudToPCL(points, numPoints, pclCloud);
    
    f = fopen("/tmp/dwfastlio_debug.log", "a");
    if (f) {
        fprintf(f, "[DWFastLIO::feedLiDAR] dwPointCloudToPCL() completed\n");
        fflush(f);
        fclose(f);
    }
    
    auto convertTime = std::chrono::steady_clock::now() - convertStart;
    auto convertMs = std::chrono::duration_cast<std::chrono::milliseconds>(convertTime).count();
    std::cout << "[DWFastLIO] [feedLiDAR] Conversion done for scan #" << scanCount 
              << " (took " << convertMs << "ms)" << std::flush << std::endl;
    
    f = fopen("/tmp/dwfastlio_debug.log", "a");
    if (f) {
        fprintf(f, "[DWFastLIO::feedLiDAR] About to set timestamp\n");
        fflush(f);
        fclose(f);
    }
    
    // Set timestamp - Fast-LIO expects microseconds (it divides by 1e6 to get seconds)
    // PCL header.stamp can be in any unit, but Fast-LIO divides by 1e6, so use microseconds
    pclCloud->header.stamp = timestamp; // Use microseconds directly (DriveWorks timestamp is already in microseconds)
    
    f = fopen("/tmp/dwfastlio_debug.log", "a");
    if (f) {
        fprintf(f, "[DWFastLIO::feedLiDAR] Timestamp set, about to acquire mutex\n");
        fflush(f);
        fclose(f);
    }
    
    // Queue the complete scan - frame skipping happens in processing thread
    size_t buffer_size = 0;
    {
        f = fopen("/tmp/dwfastlio_debug.log", "a");
        if (f) {
            fprintf(f, "[DWFastLIO::feedLiDAR] About to lock lidar_mutex_\n");
            fflush(f);
            fclose(f);
        }
        
        std::lock_guard<std::mutex> lock(lidar_mutex_);
        
        f = fopen("/tmp/dwfastlio_debug.log", "a");
        if (f) {
            fprintf(f, "[DWFastLIO::feedLiDAR] Mutex acquired, pushing to buffer\n");
            fflush(f);
            fclose(f);
        }
        lidar_buffer_.push_back({pclCloud, timestamp});
        
        // Aggressively limit buffer size - only keep last 3 scans
        while (lidar_buffer_.size() > 3) {
            lidar_buffer_.pop_front();
        }
        buffer_size = lidar_buffer_.size();
    }
    
    // Always log first 10 scans, then periodically
    std::cout << "[DWFastLIO::feedLiDAR] Queued scan #" << scanCount 
              << ": " << numPoints << " points (lidar buffer size: " << buffer_size << ")" << std::flush << std::endl;
    
    std::cout << "[DWFastLIO::feedLiDAR] EXIT #" << scanCount << std::flush << std::endl;
}

void DWFastLIO::processScan() {
    std::cout << "[DWFastLIO] Processing thread started" << std::endl;
    int processed_count = 0;
    int failed_count = 0;
    int empty_buffer_count = 0;
    
    while (thread_running_) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
        dwTime_t timestamp;
        size_t buffer_size = 0;
        
        // Get next lidar scan - apply frame skipping here (only process every 5th scan)
        {
            FILE* f = fopen("/tmp/dwfastlio_debug.log", "a");
            if (f) {
                fprintf(f, "[DWFastLIO::processScan] About to acquire lidar_mutex_ (check #%d)\n", empty_buffer_count + 1);
                fflush(f);
                fclose(f);
            }
            
            std::lock_guard<std::mutex> lock(lidar_mutex_);
            
            f = fopen("/tmp/dwfastlio_debug.log", "a");
            if (f) {
                fprintf(f, "[DWFastLIO::processScan] lidar_mutex_ acquired, buffer size=%zu\n", lidar_buffer_.size());
                fflush(f);
                fclose(f);
            }
            
            buffer_size = lidar_buffer_.size();
            
            if (lidar_buffer_.empty()) {
                empty_buffer_count++;
                // Log first few times and then periodically
                if (empty_buffer_count <= 5 || empty_buffer_count % 100 == 0) {
                    std::cout << "[DWFastLIO] [PROCESS] lidar buffer empty (check #" 
                              << empty_buffer_count << ")" << std::flush << std::endl;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Longer sleep when no data
                continue;
            }
            
            // Frame skipping: only process every 5th scan for 5-6 FPS operation
            static int scanSkipCounter = 0;
            scanSkipCounter++;
            
            if (scanSkipCounter % 5 != 0) {
                // Skip this scan - remove it from buffer
                lidar_buffer_.pop_front();
                if (scanSkipCounter <= 10) {
                    std::cout << "[DWFastLIO] Skipping scan #" << scanSkipCounter 
                              << " (processing every 5th scan)" << std::endl;
                }
                continue;
            }
            
            // Process this scan
            auto scan = lidar_buffer_.front();
            lidar_buffer_.pop_front();
            cloud = scan.first;
            timestamp = scan.second;
            
            std::cout << "[DWFastLIO] Processing thread: Processing scan #" << scanSkipCounter 
                      << " (size: " << cloud->size() << " points, buffer had " << buffer_size << " scans)" << std::endl;
        }
        
        // Check if we have IMU data synchronized
        bool hasIMU = false;
        size_t imu_buffer_size = 0;
        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            hasIMU = !imu_buffer_.empty();
            imu_buffer_size = imu_buffer_.size();
        }
        
        if (!hasIMU) {
            // Log first few times to diagnose - this is likely the issue!
            static int imu_wait_count = 0;
            imu_wait_count++;
            // Always log first 20 waits to see what's happening
            if (imu_wait_count <= 20 || imu_wait_count % 50 == 0) {
                size_t lidar_buf_size = 0;
                {
                    std::lock_guard<std::mutex> lock(lidar_mutex_);
                    lidar_buf_size = lidar_buffer_.size();
                }
                std::cout << "[DWFastLIO] WAITING FOR IMU DATA (wait #" << imu_wait_count 
                          << ", lidar buffer: " << lidar_buf_size 
                          << ", imu buffer: " << imu_buffer_size << ")" << std::endl;
            }
            // Put scan back if no IMU
            {
                std::lock_guard<std::mutex> lock(lidar_mutex_);
                lidar_buffer_.push_front({cloud, timestamp});
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        
        std::cout << "[DWFastLIO] [PROCESS] IMU available (" << imu_buffer_size 
                  << " samples), processing scan with " << cloud->size() << " points..." << std::endl;
        
        // Log when we finally have IMU and can process
        static int first_process = 0;
        if (++first_process <= 10) {
            std::cout << "[DWFastLIO] [PROCESS] Got IMU data! Processing scan #" << first_process 
                      << " (IMU buffer: " << imu_buffer_size << ", cloud size: " << cloud->size() << ")" << std::endl;
        }
        
        // Feed IMU data to Fast-LIO before processing point cloud
        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            size_t imu_fed = 0;
            for (const auto& imu : imu_buffer_) {
                fastlio_imu_enqueue(imu.first);
                imu_fed++;
            }
            if (first_process <= 5) {
                std::cout << "[DWFastLIO] [PROCESS] Fed " << imu_fed << " IMU samples to Fast-LIO" << std::endl;
            }
            // Clear IMU buffer after feeding (Fast-LIO will use what it needs)
            imu_buffer_.clear();
        }
        
        // Feed point cloud to Fast-LIO
        // Create PointCloudAttr matching LSD structure
        std::cout << "[DWFastLIO] [PROCESS] Creating PointCloudAttr and enqueueing to Fast-LIO..." << std::flush << std::endl;
        std::shared_ptr<::PointCloudAttr> cloudAttr = std::make_shared<::PointCloudAttr>();
        cloudAttr->cloud = cloud;
        cloudAttr->T = Eigen::Matrix4d::Identity();
        
        // CRITICAL: Fast-LIO's velodyne_handler() accesses pl_orig->attr[i].stamp for each point
        // We must populate attr vector with one entry per point
        cloudAttr->attr.resize(cloud->size());
        for (size_t i = 0; i < cloud->size(); ++i) {
            cloudAttr->attr[i].id = static_cast<int>(i);
            cloudAttr->attr[i].stamp = cloud->header.stamp; // Use cloud timestamp for all points
        }
        
        // Ensure cloud has valid header
        if (cloud->header.stamp == 0) {
            std::cerr << "[DWFastLIO] [PROCESS] ERROR: Cloud timestamp is 0!" << std::endl;
            continue;
        }
        
        std::cout << "[DWFastLIO] [PROCESS] PointCloudAttr created: cloud size=" << cloud->size() 
                  << ", timestamp=" << cloud->header.stamp << " (microseconds)" << std::flush << std::endl;
        
        // Validate cloud before enqueueing
        if (!cloud || cloud->empty()) {
            std::cerr << "[DWFastLIO] [PROCESS] ERROR: Cloud is null or empty!" << std::endl;
            continue;
        }
        if (!cloudAttr || !cloudAttr->cloud) {
            std::cerr << "[DWFastLIO] [PROCESS] ERROR: cloudAttr or cloudAttr->cloud is null!" << std::endl;
            continue;
        }
        
        // Validate point cloud structure
        if (cloud->points.empty()) {
            std::cerr << "[DWFastLIO] [PROCESS] ERROR: Cloud points vector is empty!" << std::endl;
            continue;
        }
        
        // Ensure cloud has valid width/height
        if (cloud->width == 0 || cloud->height == 0) {
            cloud->width = cloud->points.size();
            cloud->height = 1;
        }
        
        // Log to file before enqueueing
        FILE* f = fopen("/tmp/dwfastlio_debug.log", "a");
        if (f) {
            fprintf(f, "[DWFastLIO::processScan] About to call fastlio_pcl_enqueue(), cloud size=%zu, width=%u, height=%u\n", 
                    cloud->size(), cloud->width, cloud->height);
            fflush(f);
            fclose(f);
        }
        
        // Note: fastlio_is_init() only returns true after EKF initialization with sensor data
        // We need to enqueue data to initialize it, so don't check this here
        // Fast-LIO will initialize automatically after processing enough IMU + LiDAR data
        
        // Validate first few points to ensure they're valid
        bool points_valid = true;
        for (size_t i = 0; i < std::min(size_t(10), cloud->points.size()); ++i) {
            const auto& pt = cloud->points[i];
            if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z) || 
                std::isinf(pt.x) || std::isinf(pt.y) || std::isinf(pt.z)) {
                points_valid = false;
                break;
            }
        }
        if (!points_valid) {
            std::cerr << "[DWFastLIO] [PROCESS] ERROR: Cloud contains invalid points (NaN/Inf)!" << std::endl;
            continue;
        }
        
        // Enqueue to Fast-LIO
        try {
            std::cout << "[DWFastLIO] [PROCESS] Calling fastlio_pcl_enqueue()..." << std::flush << std::endl;
            
            f = fopen("/tmp/dwfastlio_debug.log", "a");
            if (f) {
                fprintf(f, "[DWFastLIO::processScan] Calling fastlio_pcl_enqueue() now, cloudAttr=%p, cloud=%p\n", 
                        (void*)cloudAttr.get(), (void*)cloudAttr->cloud.get());
                fflush(f);
                fclose(f);
            }
            
            // Double-check pointers before calling
            if (!cloudAttr || !cloudAttr->cloud || cloudAttr->cloud->empty()) {
                std::cerr << "[DWFastLIO] [PROCESS] ERROR: Invalid cloudAttr before enqueue!" << std::endl;
                continue;
            }
            
            fastlio_pcl_enqueue(cloudAttr);
            
            f = fopen("/tmp/dwfastlio_debug.log", "a");
            if (f) {
                fprintf(f, "[DWFastLIO::processScan] fastlio_pcl_enqueue() returned\n");
                fflush(f);
                fclose(f);
            }
            
            f = fopen("/tmp/dwfastlio_debug.log", "a");
            if (f) {
                fprintf(f, "[DWFastLIO::processScan] fastlio_pcl_enqueue() succeeded\n");
                fflush(f);
                fclose(f);
            }
            
            std::cout << "[DWFastLIO] [PROCESS] fastlio_pcl_enqueue() succeeded" << std::flush << std::endl;
        } catch (const std::exception& e) {
            f = fopen("/tmp/dwfastlio_debug.log", "a");
            if (f) {
                fprintf(f, "[DWFastLIO::processScan] ERROR: Exception in fastlio_pcl_enqueue(): %s\n", e.what());
                fclose(f);
            }
            std::cerr << "[DWFastLIO] [PROCESS] ERROR: fastlio_pcl_enqueue() threw exception: " << e.what() << std::endl;
            continue;
        } catch (...) {
            f = fopen("/tmp/dwfastlio_debug.log", "a");
            if (f) {
                fprintf(f, "[DWFastLIO::processScan] ERROR: Unknown exception in fastlio_pcl_enqueue()\n");
                fclose(f);
            }
            std::cerr << "[DWFastLIO] [PROCESS] ERROR: fastlio_pcl_enqueue() threw unknown exception" << std::endl;
            continue;
        }
        std::cout << "[DWFastLIO] [PROCESS] Point cloud enqueued, calling fastlio_main()..." << std::flush << std::endl;
        
        // Process through Fast-LIO with timeout check
        auto processStart = std::chrono::steady_clock::now();
        bool success = fastlio_main();
        auto processTime = std::chrono::steady_clock::now() - processStart;
        
        processed_count++;
        auto processMs = std::chrono::duration_cast<std::chrono::milliseconds>(processTime).count();
        
        // Always log first 10 scans, then periodically
        if (processed_count <= 10 || processed_count % 10 == 0) {
            std::cout << "[DWFastLIO] [PROCESS] fastlio_main() " << (success ? "SUCCESS" : "FAILED") 
                      << " (scan #" << processed_count 
                      << ", time: " << processMs << "ms"
                      << ", cloud size: " << cloud->size() << ")" << std::flush << std::endl;
        }
        
        // Log if processing takes too long
        if (processMs > 100) {
            std::cout << "[DWFastLIO] [PROCESS] WARNING: fastlio_main() took " << processMs << "ms" << std::endl;
        }
        
        if (!success) {
            failed_count++;
            if (failed_count <= 5 || failed_count % 10 == 0) {
                std::cout << "[DWFastLIO] [PROCESS] ERROR: fastlio_main() FAILED (failure #" << failed_count << ")" << std::endl;
            }
        }
        
        if (success) {
            std::cout << "[DWFastLIO] [PROCESS] fastlio_main() succeeded, getting odometry..." << std::flush << std::endl;
            // Get odometry
            Eigen::Matrix4d odom_start = Eigen::Matrix4d::Identity();
            Eigen::Matrix4d odom_end = Eigen::Matrix4d::Identity();
            fastlio_odometry(odom_start, odom_end);
            
            if (processed_count <= 5) {
                std::cout << "[DWFastLIO] [PROCESS] Odometry: pose translation = [" 
                          << odom_end(0,3) << ", " << odom_end(1,3) << ", " << odom_end(2,3) << "]" << std::endl;
            }
            
            // Update current pose (with mutex protection)
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                current_pose_ = odom_end;
                trajectory_.push_back(current_pose_);
                
                // Limit trajectory size to prevent memory growth
                if (trajectory_.size() > 5000) { // Reduced from 10000
                    trajectory_.erase(trajectory_.begin(), trajectory_.begin() + 2500);
                }
            }
            
            static int pose_log_counter = 0;
            pose_log_counter++;
            // Log first few and then every 10th
            if (pose_log_counter <= 10 || pose_log_counter % 10 == 0) {
                size_t map_size = 0;
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    map_size = map_cloud_ ? map_cloud_->size() : 0;
                }
                std::cout << "[DWFastLIO] Pose updated #" << pose_log_counter 
                          << " - Position: " << std::fixed << std::setprecision(2)
                          << odom_end(0,3) << ", " << odom_end(1,3) << ", " << odom_end(2,3) 
                          << " (map points: " << map_size << ")" << std::endl;
            }
                      
            // Add to map if in mapping mode
            if (mapping_mode_) {
                if (processed_count <= 5) {
                    std::cout << "[DWFastLIO] [PROCESS] Mapping mode: transforming cloud to world frame..." << std::flush << std::endl;
                }
                // Transform cloud to world frame
                pcl::PointCloud<pcl::PointXYZI>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZI>());
                pcl::transformPointCloud(*cloud, *transformed, odom_end);
                
                if (processed_count <= 5) {
                    std::cout << "[DWFastLIO] [PROCESS] Transformed " << transformed->size() << " points to world frame" << std::endl;
                }
                
                // Add to map with voxel filtering (with mutex protection)
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    size_t map_size_before = map_cloud_ ? map_cloud_->size() : 0;
                    
                    if (processed_count <= 5) {
                        std::cout << "[DWFastLIO] [PROCESS] Map size before: " << map_size_before << " points" << std::endl;
                    }
                    
                    // Add transformed points to map
                    *map_cloud_ += *transformed;
                    
                    size_t map_size_after_add = map_cloud_->size();
                    
                    if (processed_count <= 5) {
                        std::cout << "[DWFastLIO] [PROCESS] Map size after adding scan: " << map_size_after_add << " points" << std::endl;
                    }
                    
                    // Only filter periodically to reduce CPU load (every 10th scan now)
                    static int filter_counter = 0;
                    if (++filter_counter % 10 == 0) { // Reduced frequency
                        if (processed_count <= 10) {
                            std::cout << "[DWFastLIO] [PROCESS] Running voxel filter (every 10th scan)..." << std::flush << std::endl;
                        }
                        auto filterStart = std::chrono::steady_clock::now();
                        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>());
                        pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
                        voxel_filter.setLeafSize(config_.voxel_size, config_.voxel_size, config_.voxel_size);
                        voxel_filter.setInputCloud(map_cloud_);
                        voxel_filter.filter(*filtered); // Filter to new cloud, don't overwrite
                        map_cloud_ = filtered; // Replace with filtered version
                        auto filterTime = std::chrono::steady_clock::now() - filterStart;
                        auto filterMs = std::chrono::duration_cast<std::chrono::milliseconds>(filterTime).count();
                        
                        std::cout << "[DWFastLIO] [PROCESS] Voxel filter: " << map_size_before << " -> " 
                                  << map_size_after_add << " -> " << map_cloud_->size()
                                  << " points (took " << filterMs << "ms)" << std::flush << std::endl;
                    } else {
                        // Log map growth occasionally
                        static int growth_log_counter = 0;
                        if (++growth_log_counter % 20 == 0) {
                            std::cout << "[DWFastLIO] Map growing: " << map_size_before 
                                      << " -> " << map_size_after_add << " points" << std::endl;
                        }
                    }
                }
            }
        } else {
            failed_count++;
            // Always log failures
            std::cout << "[DWFastLIO] ERROR: fastlio_main() returned false (failures: " << failed_count 
                      << " out of " << processed_count << " total)" << std::endl;
        }
    }
    std::cout << "[DWFastLIO] Processing thread stopped (processed: " << processed_count 
              << ", failed: " << failed_count << ")" << std::endl;
}

Eigen::Matrix4d DWFastLIO::getCurrentPose() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_pose_;
}

std::vector<Eigen::Matrix4d> DWFastLIO::getTrajectory() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return trajectory_;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr DWFastLIO::getMapCloud() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return map_cloud_;
}

bool DWFastLIO::saveMap(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!map_cloud_ || map_cloud_->empty()) {
        std::cerr << "[DWFastLIO] No map to save" << std::endl;
        return false;
    }
    
    pcl::io::savePCDFileBinary(filepath, *map_cloud_);
    std::cout << "[DWFastLIO] Saved map: " << map_cloud_->size() << " points to " << filepath << std::endl;
    return true;
}

bool DWFastLIO::loadMap(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(state_mutex_);
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

