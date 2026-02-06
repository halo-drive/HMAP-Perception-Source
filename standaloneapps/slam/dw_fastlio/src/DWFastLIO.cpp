/////////////////////////////////////////////////////////////////////////////////////////
// DriveWorks Fast-LIO Implementation
/////////////////////////////////////////////////////////////////////////////////////////

// Include LSD types first when using graph backend, so mapping_types.h defines
// ImuType, RTKType, PointCloudAttr, PointAttr once. TypeConverters.hpp then
// does not redefine them (it only defines when graph backend is off).
#ifdef DW_FASTLIO_USE_GRAPH_BACKEND
#include "slam_base.h"
#include "backend_api.h"
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_property.h>
#endif

#include "../include/DWFastLIO.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <cerrno>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

// FastLIO API from LSD (C++ linkage)
extern int  fastlio_init(std::vector<double> &extT, std::vector<double>& extR,
                         int filter_num, int max_point_num, double scan_period, bool undistort);
extern bool fastlio_is_init();
extern void fastlio_imu_enqueue(::ImuType imu);
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
    , graph_thread_running_(false)
    , backend_initialized_(false)
    , graph_origin_set_(false)
    , first_lidar_stamp_us_(0)
    , keyframes_dirty_(false)
    , request_reloc_(false)
{
    current_pose_ = Eigen::Matrix4d::Identity();
    odom_pose_    = Eigen::Matrix4d::Identity();
    pose_offset_  = Eigen::Matrix4d::Identity();
    last_keyframe_pose_ = Eigen::Matrix4d::Identity();
    map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
}

DWFastLIO::~DWFastLIO() {
    thread_running_ = false;
    graph_thread_running_ = false;
    if (graph_thread_ && graph_thread_->joinable()) {
        graph_thread_->join();
    }
    if (backend_init_thread_ && backend_init_thread_->joinable()) {
        backend_init_thread_->join();
    }
    if (processing_thread_ && processing_thread_->joinable()) {
        processing_thread_->join();
    }
#ifdef DW_FASTLIO_USE_GRAPH_BACKEND
    if (backend_initialized_) {
        deinit_backend();
        backend_initialized_ = false;
    }
#endif
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
    
    // fastlio_is_init() becomes true only after the first IMU+LiDAR frame is processed (EKF init). Expected to be false here.
    bool is_init = fastlio_is_init();
    std::cout << "[DWFastLIO] fastlio_is_init() = " << (is_init ? "true" : "false")
              << " (will become true after first scan is processed)" << std::endl;
    
    initialized_ = true;

#ifdef DW_FASTLIO_USE_GRAPH_BACKEND
    // Defer graph backend init to a background thread so we don't block the main thread.
    // That lets the app start draining sensor events immediately and avoids "queues full, losing packets"
    // while init_backend (and its solver allocation / logging) runs.
    if (mapping_mode_) {
        backend_init_thread_.reset(new std::thread(&DWFastLIO::initBackendInBackground, this));
        std::cout << "[DWFastLIO] Graph backend init deferred to background (main loop can drain sensors now)" << std::endl;
    }
#endif

    // Start processing thread
    thread_running_ = true;
    processing_thread_.reset(new std::thread(&DWFastLIO::processScan, this));

    std::cout << "[DWFastLIO] Initialized successfully" << std::endl;
    return true;
}

void DWFastLIO::initBackendInBackground() {
#ifdef DW_FASTLIO_USE_GRAPH_BACKEND
    // LSD backend requests solver "lm_var"; our g2o build registers "lm_var_cholmod". Register alias.
    {
        g2o::OptimizationAlgorithmFactory* factory = g2o::OptimizationAlgorithmFactory::instance();
        std::shared_ptr<g2o::AbstractOptimizationAlgorithmCreator> cholmod_creator;
        for (const auto& c : factory->creatorList()) {
            if (c->property().name == "lm_var_cholmod") {
                cholmod_creator = c;
                break;
            }
        }
        if (cholmod_creator) {
            class LmVarAliasCreator : public g2o::AbstractOptimizationAlgorithmCreator {
                std::shared_ptr<g2o::AbstractOptimizationAlgorithmCreator> delegate_;
            public:
                explicit LmVarAliasCreator(const std::shared_ptr<g2o::AbstractOptimizationAlgorithmCreator>& d)
                    : AbstractOptimizationAlgorithmCreator(g2o::OptimizationAlgorithmProperty(
                            "lm_var", "Levenberg: Cholesky (variable blocksize)", "CHOLMOD",
                            false, -1, -1)),
                      delegate_(d) {}
                g2o::OptimizationAlgorithm* construct() override { return delegate_->construct(); }
            };
            factory->registerSolver(std::make_shared<LmVarAliasCreator>(cholmod_creator));
            std::cout << "[DWFastLIO] Registered g2o solver alias: lm_var -> lm_var_cholmod" << std::endl;
        } else {
            std::cerr << "[DWFastLIO] lm_var_cholmod not found in g2o factory; graph solver may fail" << std::endl;
        }
    }

    InitParameter backend_param;
    backend_param.map_path = "";
    backend_param.resolution = config_.voxel_size;
    backend_param.key_frame_distance = config_.keyframe_delta_trans;
    backend_param.key_frame_degree = config_.keyframe_delta_angle * 180.0 / M_PI;
    backend_param.key_frame_range = 10.0;
    backend_param.scan_period = config_.scan_period;
    try {
        init_backend(backend_param);
        backend_initialized_ = true;
        graph_thread_running_ = true;
        graph_thread_.reset(new std::thread(&DWFastLIO::runGraphThread, this));
        std::cout << "[DWFastLIO] Graph backend initialized in background (get_odom2map, floor, GPS)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[DWFastLIO] init_backend failed: " << e.what() << " (continuing without graph)" << std::endl;
    }
#else
    (void)0;
#endif
}

void DWFastLIO::runGraphThread() {
#ifdef DW_FASTLIO_USE_GRAPH_BACKEND
    std::cout << "[DWFastLIO] Graph thread started" << std::endl;
    while (graph_thread_running_) {
        graph_optimization(graph_thread_running_);
        int sleep_ms = 100;
        int count = 0;
        while (count < 30 && graph_thread_running_) {  // ~3s total
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            count++;
        }
    }
    std::cout << "[DWFastLIO] Graph thread stopped" << std::endl;
#else
    (void)0;
#endif
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
    
    // Use same time origin as LiDAR (relative time so LSD EKF doesn't see huge dt on first frame)
    uint64_t stamp_us = static_cast<uint64_t>(timestamp);
    if (stamp_us > 1000000000000000ULL) stamp_us /= 1000;
    if (first_lidar_stamp_us_ == 0) first_lidar_stamp_us_ = stamp_us;
    
    ::ImuType lsdImu;
    dwIMUToLSD(dwImu, lsdImu, timestamp);
    lsdImu.stamp -= static_cast<double>(first_lidar_stamp_us_) / 1e6;  // relative seconds
    
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
    
    // Use same time origin as LiDAR/IMU (relative microseconds for LSD sync)
    uint64_t stamp_us = static_cast<uint64_t>(timestamp);
    if (stamp_us > 1000000000000000ULL) stamp_us /= 1000;
    if (first_lidar_stamp_us_ == 0) first_lidar_stamp_us_ = stamp_us;
    lsdRTK.timestamp = stamp_us - first_lidar_stamp_us_;
    
    // Feed into Fast-LIO backend (INS/RTK for velocity constraint and graph)
    bool rtk_valid = (lsdRTK.status != 0);
    fastlio_ins_enqueue(rtk_valid, lsdRTK);

#ifdef DW_FASTLIO_USE_GRAPH_BACKEND
    // Full pipeline: feed graph backend with GPS; set origin on first valid fix
    if (backend_initialized_) {
        std::shared_ptr<::RTKType> rtk_ptr = std::make_shared<::RTKType>(lsdRTK);
        enqueue_graph_gps(rtk_valid, rtk_ptr);
        if (rtk_valid && !graph_origin_set_) {
            graph_set_origin(lsdRTK);
            graph_origin_set_ = true;
            std::cout << "[DWFastLIO] Graph origin set from first valid GPS" << std::endl;
        }
    }
#endif

    static int gps_fed_count = 0;
    gps_fed_count++;
    if (gps_fed_count <= 5) {
        std::cout << "[DWFastLIO] GPS/RTK fed to Fast-LIO #" << gps_fed_count
                  << " valid=" << (rtk_valid ? 1 : 0)
                  << " (lat=" << lsdRTK.latitude << ", lon=" << lsdRTK.longitude << ")" << std::endl;
    } else if (rtk_valid && gps_fed_count == 6) {
        std::cout << "[DWFastLIO] GPS/RTK fed to Fast-LIO (valid fix); further feeds not logged" << std::endl;
    }
    
    {
        std::lock_guard<std::mutex> lock(gps_mutex_);
        gps_buffer_.push_back({lsdRTK, timestamp});
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
    
    // Set timestamp - Fast-LIO expects microseconds (it divides by 1e6 to get seconds).
    // DriveWorks docs say hostTimestamp is in microseconds, but some sensors/contexts give nanoseconds.
    // If value is > 1e15, it's likely nanoseconds; convert to microseconds.
    // CRITICAL: LSD's EKF uses dt = meas.lidar_beg_time - last_lidar_end_time_; on first frame last_lidar_end_time_=0,
    // so if we pass absolute time (e.g. 1.77e6 s), dt becomes huge and position explodes. Use RELATIVE time (us since first scan).
    uint64_t stamp_us = static_cast<uint64_t>(timestamp);
    if (stamp_us > 1000000000000000ULL) {  // 1e15 - likely nanoseconds
        stamp_us /= 1000;
    }
    if (first_lidar_stamp_us_ == 0) {
        first_lidar_stamp_us_ = stamp_us;
    }
    pclCloud->header.stamp = stamp_us - first_lidar_stamp_us_;
    
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
        
        // Global re-localization on request (e.g. 'R' key) when keyframe map is loaded
        if (!mapping_mode_ && request_reloc_ && global_reloc_ && global_reloc_->keyframeCount() > 0) {
            std::cout << "[DWFastLIO] Trying global relocalization (ScanContext + GICP)..." << std::endl;
            Eigen::Matrix4d reloc_pose;
            if (tryGlobalRelocalization(cloud, reloc_pose)) {
                request_reloc_ = false;
            } else {
                std::cout << "[DWFastLIO] Global relocalization failed (no ScanContext match or GICP did not converge)" << std::endl;
            }
        }
        
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
        
        // CRITICAL: Fast-LIO's velodyne_handler() accesses pl_orig->attr[i].stamp for each point.
        // PointAttr.stamp is uint32_t; full microsecond timestamp can overflow. Use 0 for single-scan
        // (scan time is in cloud->header.stamp); LSD time_buffer uses header.stamp only.
        cloudAttr->attr.resize(cloud->size());
        for (size_t i = 0; i < cloud->size(); ++i) {
            cloudAttr->attr[i].id = static_cast<int>(i);
            cloudAttr->attr[i].stamp = 0;
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
            // Get odometry (Fast-LIO "odom" frame)
            Eigen::Matrix4d odom_start = Eigen::Matrix4d::Identity();
            Eigen::Matrix4d odom_end   = Eigen::Matrix4d::Identity();
            fastlio_odometry(odom_start, odom_end);

#ifdef DW_FASTLIO_USE_GRAPH_BACKEND
            // Full pipeline: feed graph backend (frame + floor) and get pose from get_odom2map()
            if (backend_initialized_) {
                // Build frame like LSD Fast-LIO getPose: points->T = delta odom, frame.T = odom at scan start
                cloudAttr->T = (Eigen::Isometry3d(odom_start).inverse() * Eigen::Isometry3d(odom_end)).matrix();
                ::PointCloudAttrImagePose frame(cloudAttr, Eigen::Isometry3d(odom_start));
                ::PointCloudAttrImagePose keyframe;
                enqueue_graph(frame, keyframe);

                // Floor constraint: same scan -> filter -> floor -> graph
                try {
                    PointCloud::Ptr cloud_ptr = cloud;
                    PointCloud::Ptr filtered = enqueue_filter(cloud_ptr);
                    if (filtered && !filtered->empty()) {
                        FloorCoeffs floor_coeffs = enqueue_floor(filtered);
                        enqueue_graph_floor(floor_coeffs);
                    }
                } catch (const std::exception& e) {
                    (void)e;
                }
            }
#endif

            // World/map pose: from graph when backend active, else pose_offset_ * odom
            Eigen::Matrix4d world_pose;
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                odom_pose_ = odom_end;
#ifdef DW_FASTLIO_USE_GRAPH_BACKEND
                if (backend_initialized_) {
                    Eigen::Isometry3d odom2map = get_odom2map();
                    world_pose = (odom2map * Eigen::Isometry3d(odom_pose_)).matrix();
                } else
#endif
                {
                    world_pose = pose_offset_ * odom_pose_;
                }
                current_pose_ = world_pose;
                trajectory_.push_back(current_pose_);

                // Limit trajectory size to prevent memory growth
                if (trajectory_.size() > 5000) { // Reduced from 10000
                    trajectory_.erase(trajectory_.begin(), trajectory_.begin() + 2500);
                }
            }

            if (processed_count <= 5) {
                std::cout << "[DWFastLIO] [PROCESS] Odometry: pose translation = [" 
                          << world_pose(0,3) << ", " << world_pose(1,3) << ", " << world_pose(2,3) << "]" << std::endl;
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
                          << world_pose(0,3) << ", " << world_pose(1,3) << ", " << world_pose(2,3) 
                          << " (map points: " << map_size << ")" << std::endl;
            }
                      
            // Add to map if in mapping mode
            if (mapping_mode_) {
                if (processed_count <= 5) {
                    std::cout << "[DWFastLIO] [PROCESS] Mapping mode: transforming cloud to world frame..." << std::flush << std::endl;
                }
                // Transform cloud to world frame
                pcl::PointCloud<pcl::PointXYZI>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZI>());
                pcl::transformPointCloud(*cloud, *transformed, world_pose);
                
                // Voxel-downsample the new scan before adding (avoids unbounded growth and prevents
                // "vanishing" from full-map voxel replace every N scans)
                pcl::PointCloud<pcl::PointXYZI>::Ptr scan_voxeled(new pcl::PointCloud<pcl::PointXYZI>());
                pcl::VoxelGrid<pcl::PointXYZI> scan_voxel;
                scan_voxel.setLeafSize(config_.voxel_size, config_.voxel_size, config_.voxel_size);
                scan_voxel.setInputCloud(transformed);
                scan_voxel.filter(*scan_voxeled);
                
                if (processed_count <= 5) {
                    std::cout << "[DWFastLIO] [PROCESS] Transformed " << transformed->size()
                              << " -> voxeled " << scan_voxeled->size() << " points" << std::endl;
                }
                
                // Add voxel-downsampled scan to map (no full-map replace -> no visible "vanishing")
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    size_t map_size_before = map_cloud_ ? map_cloud_->size() : 0;
                    
                    *map_cloud_ += *scan_voxeled;
                    size_t map_size_after_add = map_cloud_->size();
                    
                    if (processed_count <= 5) {
                        std::cout << "[DWFastLIO] [PROCESS] Map size: " << map_size_before
                                  << " -> " << map_size_after_add << " points" << std::endl;
                    }
                    
                    // Keyframe for global re-localization: add when pose moved enough (sensor-frame scan)
                    double dt = (world_pose.block<3,1>(0,3) - last_keyframe_pose_.block<3,1>(0,3)).norm();
                    Eigen::AngleAxisd aa(last_keyframe_pose_.block<3,3>(0,0).inverse() * world_pose.block<3,3>(0,0));
                    double da = std::abs(aa.angle());
                    if (dt >= config_.keyframe_delta_trans || da >= config_.keyframe_delta_angle) {
                        addKeyframe(world_pose, cloud);
                        last_keyframe_pose_ = world_pose;
                        keyframes_dirty_ = true;
                    }
                    
                    // Optional: full-map voxel every 100 scans to bound memory (less frequent = less flicker)
                    static int filter_counter = 0;
                    if (++filter_counter % 100 == 0 && map_cloud_->size() > 500000) {
                        auto filterStart = std::chrono::steady_clock::now();
                        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>());
                        pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
                        voxel_filter.setLeafSize(config_.voxel_size, config_.voxel_size, config_.voxel_size);
                        voxel_filter.setInputCloud(map_cloud_);
                        voxel_filter.filter(*filtered);
                        map_cloud_ = filtered;
                        auto filterMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - filterStart).count();
                        std::cout << "[DWFastLIO] [PROCESS] Full-map voxel: " << map_size_after_add
                                  << " -> " << map_cloud_->size() << " points (took " << filterMs << "ms)" << std::endl;
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

void DWFastLIO::addKeyframe(const Eigen::Matrix4d& pose, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    if (!cloud || cloud->empty()) return;
    KeyframeReloc kf;
    kf.pose = pose;
    kf.cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::copyPointCloud(*cloud, *kf.cloud);
    keyframes_.push_back(std::move(kf));
    if (keyframes_.size() <= 3 || keyframes_.size() % 20 == 0) {
        std::cout << "[DWFastLIO] Keyframe #" << keyframes_.size() << " (pose [" << pose(0,3) << "," << pose(1,3) << "," << pose(2,3) << "])" << std::endl;
    }
}

static bool isDirectoryPath(const std::string& path) {
    if (path.empty()) return false;
    if (path.back() == '/') return true;
    return path.find('.') == std::string::npos || path.find('/') != std::string::npos;
}

bool DWFastLIO::saveMap(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (isDirectoryPath(filepath) && !keyframes_.empty()) {
        // Save keyframe map (LSD-compatible directory layout)
        std::string base = filepath;
        if (base.back() == '/') base.pop_back();
        std::string graphDir = base + "/graph";
        if (mkdir(base.c_str(), 0755) != 0 && errno != EEXIST) {
            std::cerr << "[DWFastLIO] Cannot create dir " << base << std::endl;
            return false;
        }
        if (mkdir(graphDir.c_str(), 0755) != 0 && errno != EEXIST) {
            std::cerr << "[DWFastLIO] Cannot create dir " << graphDir << std::endl;
            return false;
        }
        std::ofstream info(graphDir + "/map_info.txt");
        if (info) info << "version 1\n";
        for (size_t i = 0; i < keyframes_.size(); i++) {
            std::string kfDir = graphDir + "/" + std::to_string(i);
            if (mkdir(kfDir.c_str(), 0755) != 0 && errno != EEXIST) continue;
            pcl::io::savePCDFileBinary(kfDir + "/cloud.pcd", *keyframes_[i].cloud);
            std::ofstream data(kfDir + "/data");
            if (data) {
                data << "stamp 0 0\nestimate\n" << keyframes_[i].pose.format(Eigen::IOFormat(Eigen::FullPrecision, 0, " ", "\n")) << "\nid " << (long)i << "\n";
            }
        }
        std::cout << "[DWFastLIO] Saved keyframe map: " << keyframes_.size() << " keyframes to " << base << std::endl;
        return true;
    }
    if (!map_cloud_ || map_cloud_->empty()) {
        std::cerr << "[DWFastLIO] No map to save" << std::endl;
        return false;
    }
    pcl::io::savePCDFileBinary(filepath, *map_cloud_);
    std::cout << "[DWFastLIO] Saved map: " << map_cloud_->size() << " points to " << filepath << std::endl;
    if (keyframes_.empty()) {
        std::cout << "[DWFastLIO] No keyframes (pose did not move >= 0.25 m). Load this .pcd for map display only; press R after driving and saving a keyframe map for relocalization." << std::endl;
    }
    return true;
}

static std::vector<std::string> getSubdirs(const std::string& dir) {
    std::vector<std::string> out;
    DIR* d = opendir(dir.c_str());
    if (!d) return out;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        if (e->d_name[0] == '.') continue;
        std::string path = dir + "/" + e->d_name;
        struct stat st;
        if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            bool numeric = true;
            for (char* p = e->d_name; *p; p++) if (!std::isdigit(*p)) { numeric = false; break; }
            if (numeric) out.push_back(path);
        }
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

bool DWFastLIO::loadMap(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    std::string path = filepath;
    if (path.back() == '/') path.pop_back();
    std::string graphDir = path + "/graph";
    std::ifstream info(graphDir + "/map_info.txt");
    if (info.good()) {
        keyframes_.clear();
        auto subdirs = getSubdirs(graphDir);
        for (const std::string& kfDir : subdirs) {
            KeyframeReloc kf;
            kf.pose = Eigen::Matrix4d::Identity();
            kf.cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
            std::ifstream data(kfDir + "/data");
            if (!data) continue;
            std::string token;
            while (data >> token) {
                if (token == "estimate") {
                    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) data >> kf.pose(i, j);
                    break;
                }
            }
            if (pcl::io::loadPCDFile(kfDir + "/cloud.pcd", *kf.cloud) < 0) continue;
            keyframes_.push_back(std::move(kf));
        }
        if (keyframes_.empty()) {
            std::cerr << "[DWFastLIO] No keyframes loaded from " << path << std::endl;
            return false;
        }
        global_reloc_.reset(new GlobalReloc());
        global_reloc_->setDownsampleResolution(config_.voxel_size);
        global_reloc_->setKeyframes(keyframes_);
        map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        for (const auto& kf : keyframes_) {
            pcl::PointCloud<pcl::PointXYZI> world;
            pcl::transformPointCloud(*kf.cloud, world, kf.pose);
            *map_cloud_ += world;
        }
        std::cout << "[DWFastLIO] Loaded keyframe map: " << keyframes_.size() << " keyframes, " << map_cloud_->size() << " points (global reloc enabled)" << std::endl;
        std::cout << "[DWFastLIO] Press 'R' to trigger global re-localization (place recognition)" << std::endl;
        mapping_mode_ = false;
        return true;
    }
    map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    global_reloc_.reset();
    if (pcl::io::loadPCDFile(path, *map_cloud_) == -1) {
        std::cerr << "[DWFastLIO] Failed to load map from " << path << std::endl;
        return false;
    }
    std::cout << "[DWFastLIO] Loaded map: " << map_cloud_->size() << " points from " << path << std::endl;
    mapping_mode_ = false;
    return true;
}

bool DWFastLIO::tryGlobalRelocalization(const pcl::PointCloud<pcl::PointXYZI>::Ptr& scan, Eigen::Matrix4d& pose_out) {
    if (!global_reloc_ || global_reloc_->keyframeCount() == 0) return false;
    bool ok = global_reloc_->relocalize(scan, pose_out);
    if (ok) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        // Update map<-odom offset so future odometry lives in relocalized map frame.
        // We want: pose_offset_ * odom_pose_ = pose_out  =>  pose_offset_ = pose_out * odom_pose_.inverse()
        Eigen::Matrix4d odom = odom_pose_;
        if (odom.isIdentity(1e-9)) {
            // If odom is still identity (very early), just set offset = pose_out.
            pose_offset_ = pose_out;
        } else {
            pose_offset_ = pose_out * odom.inverse();
        }
        current_pose_ = pose_out;
        trajectory_.push_back(current_pose_);
        std::cout << "[DWFastLIO] Global relocalization SUCCESS - pose (x,y,z): "
                  << pose_out(0, 3) << ", " << pose_out(1, 3) << ", " << pose_out(2, 3) << std::endl;
    }
    return ok;
}

void DWFastLIO::setMappingMode(bool mapping) {
    mapping_mode_ = mapping;
    std::cout << "[DWFastLIO] Mode: " << (mapping ? "MAPPING" : "LOCALIZATION") << std::endl;
}

} // namespace dw_slam

