/////////////////////////////////////////////////////////////////////////////////////////
//
// Real-time Egomotion Sample with Synchronized CAN Processing
// Extended from NVIDIA DriveWorks sample for live vehicle data processing
//
/////////////////////////////////////////////////////////////////////////////////////////

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>

#include <dw/control/vehicleio/VehicleIO.h>
#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/egomotion/base/Egomotion.h>
#include <dw/egomotion/global/GlobalEgomotion.h>
#include <dw/image/Image.h>
#include <dw/rig/Rig.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/camera/Camera.h>
#include <dw/sensors/gps/GPS.h>
#include <dw/sensors/imu/IMU.h>
#include <dw/sensors/sensormanager/SensorManager.h>

#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Renderer.h>
#include <dwvisualization/core/Visualization.h>
#include <dwvisualization/gl/GL.h>
#include <dwvisualization/image/Image.h>

#include <framework/Checks.hpp>
#include <framework/CudaTimer.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/Log.hpp>
#include <framework/Mat4.hpp>
#include <framework/MathUtils.hpp>
#include <framework/MouseView3D.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/RenderUtils.hpp>
#include <framework/SamplesDataPath.hpp>
#include <framework/SimpleStreamer.hpp>
#include <framework/WindowGLFW.hpp>

#include "TrajectoryLogger.hpp"
#include "sygnalpomoparser.hpp"

using namespace dw_samples::common;

static inline bool allowEvery(dwTime_t now, dwTime_t &last, uint64_t period_us) {
    if (now - last < period_us) return false;
    last = now; return true;
}


///------------------------------------------------------------------------------
/// Real-time Egomotion sample with synchronized CAN processing
///------------------------------------------------------------------------------
class EgomotionSample : public DriveWorksSample
{
private:
    // ------------------------------------------------
    // DriveWorks Context and SAL
    // ------------------------------------------------
    dwContextHandle_t m_context                 = DW_NULL_HANDLE;
    dwRenderEngineHandle_t m_renderEngine       = DW_NULL_HANDLE;
    dwEgomotionHandle_t m_egomotion             = DW_NULL_HANDLE;
    dwGlobalEgomotionHandle_t m_globalEgomotion = DW_NULL_HANDLE;
    dwSALHandle_t m_sal                         = DW_NULL_HANDLE;
    dwSensorManagerHandle_t m_sensorManager     = DW_NULL_HANDLE;
    dwRigHandle_t m_rigConfig                   = DW_NULL_HANDLE;
    dwVisualizationContextHandle_t m_vizCtx     = DW_NULL_HANDLE;
    
    dwImageHandle_t m_lastGLFrame = DW_NULL_HANDLE; 

    // Real-time CAN Parser (replaces VehicleIO)
    std::unique_ptr<SygnalPomoParser> m_canParser;

    // ------------------------------------------------
    // Renderer related
    // ------------------------------------------------
    dwRendererHandle_t m_renderer = DW_NULL_HANDLE;
    bool m_shallRender            = true;
    uint32_t m_tileGrid           = 0;
    uint32_t m_tileVideo          = 1;
    uint32_t m_tileRollPlot       = 2;
    uint32_t m_tilePitchPlot      = 3;
    uint32_t m_tileAltitudePlot   = 4;

    enum RenderingMode
    {
        STICK_TO_VEHICLE,
        ON_VEHICLE_STICK_TO_WORLD,
    } m_renderingMode = ON_VEHICLE_STICK_TO_WORLD;

    // ------------------------------------------------
    //Temporal Synchronization Infrastructure  
    // ------------------------------------------------
    struct TemporalSensorBuffer {
        std::map<dwTime_t, dwGPSFrame> gpsBuffer;
        std::map<dwTime_t, dwIMUFrame> imuBuffer;
        std::map<dwTime_t, dwCANMessage> canBuffer;
        std::mutex bufferMutex;
        
        //temporal tracking
        dwTime_t lastProcessedTimestamp = 0;
        dwTime_t lastGPSTimestamp       = 0;
        dwTime_t lastIMUTimestamp       = 0;
        dwTime_t lastCANTimestamp       = 0;

        //Buffer management
        static constexpr dwTime_t BUFFER_RENTENTION_US = 1000000; //1 second

        void cleanupOldData(dwTime_t currentTimestamp) {
           dwTime_t cutoff = currentTimestamp - BUFFER_RENTENTION_US;
           
           gpsBuffer.erase(gpsBuffer.begin(), gpsBuffer.lower_bound(cutoff));
           imuBuffer.erase(imuBuffer.begin(), imuBuffer.lower_bound(cutoff));
           canBuffer.erase(canBuffer.begin(), canBuffer.lower_bound(cutoff));
        }
    } m_temporalBuffers;

    static constexpr dwTime_t TEMPORAL_WINDOW_US = 100000 ;
    static constexpr dwTime_t FUSION_RATE_US  = 50000 ;
    static constexpr dwTime_t SENSOR_TIMEOUT_US  = 50000 ; 
    dwTime_t m_lastFusionTimestamp = 0;

    // ------------------------------------------------
    // Camera and video visualization
    // ------------------------------------------------
    dwImageHandle_t m_convertedImageRGBA = DW_NULL_HANDLE;
    std::unique_ptr<SimpleImageStreamerGL<>> m_streamerInput2GL;
    dwImageGL* m_currentGlFrame = nullptr;

    // ------------------------------------------------
    // Current sensor states
    // ------------------------------------------------
    dwEgomotionParameters m_egomotionParameters{};
    dwGlobalEgomotionParameters m_globalEgomotionParameters{};

    dwIMUFrame m_currentIMUFrame = {};
    dwGPSFrame m_currentGPSFrame = {};
    uint32_t m_imuSensorIdx      = 0;
    uint32_t m_vehicleSensorIdx  = 0;
    uint32_t m_gpsSensorIdx      = 0;

    const dwSensorEvent* acquiredEvent = nullptr;
    
    struct PendingCameraFrame {
        dwImageHandle_t cudaImage;  // CUDA frame handle (not event pointer for safety)
        dwTime_t timestamp;
    };
    std::deque<PendingCameraFrame> m_pendingCameraFrames;
    std::mutex m_cameraQueueMutex;
    static constexpr size_t MAX_PENDING_CAMERA_FRAMES = 5;    

    
    // ------------------------------------------------
    // Real-time processing variables
    // ------------------------------------------------
    const dwTime_t POSE_SAMPLE_PERIOD = 100000; // 100ms
    const size_t MAX_BUFFER_POINTS    = 100000;
    
    // State commit timing management
    std::atomic<dwTime_t> m_lastCommitAttempt{0};
    std::atomic<dwTime_t> m_lastSuccessfulCommit{0};
    std::atomic<uint32_t> m_commitAttempts{0};
    std::atomic<uint32_t> m_commitSuccesses{0};

    struct Pose
    {
        dwTime_t timestamp                 = 0;
        dwTransformation3f rig2world       = {};
        dwEgomotionUncertainty uncertainty = {};
        float32_t rpy[3]                   = {};
    };

    std::vector<Pose> m_poseHistory;

    dwQuaternionf m_orientationENU = DW_IDENTITY_QUATERNIONF;
    bool m_hasOrientationENU       = false;

    FILE* m_outputFile = nullptr;

    dwTime_t m_elapsedTime         = 0;
    dwTime_t m_lastSampleTimestamp = 0;
    dwTime_t m_firstTimestamp      = 0;

    TrajectoryLogger m_trajectoryLog;

private:
    // ------------------------------------------------
    // Temporal Fusion Engine
    // ------------------------------------------------
    
    /**
     * Attempts temporal fusion of all sensor data at current timestamp
     * @param currentTime Current system timestamp in microseconds
     * @return true if fusion was successful, false if waiting for more data
     */
    bool attemptTemporalFusion(dwTime_t currentTime) {
        static uint32_t fusionCallCount = 0;
        ++fusionCallCount;
        
        fprintf(stderr, "   [fusion #%u] START: currentTime=%lu\n", fusionCallCount, currentTime);
        fflush(stderr);
        
        fprintf(stderr, "   [fusion #%u] Acquiring buffer mutex...\n", fusionCallCount);
        fflush(stderr);
        
        std::lock_guard<std::mutex> lock(m_temporalBuffers.bufferMutex);
        
        fprintf(stderr, "   [fusion #%u] Mutex acquired, finding GPS anchor...\n", fusionCallCount);
        fflush(stderr);
        
        // Find GPS anchor point (fusion cadence)
        auto gpsAnchor = findLatestValidGPS(currentTime);
        if (gpsAnchor == m_temporalBuffers.gpsBuffer.end()) {
            fprintf(stderr, "   [fusion #%u] No GPS anchor found, returning false\n", fusionCallCount);
            fflush(stderr);
            return false;
        }
        
        fprintf(stderr, "   [fusion #%u] GPS anchor found: timestamp=%lu\n", 
                fusionCallCount, gpsAnchor->first);
        fflush(stderr);
        
        // Initialize trajectory from GPS even when vehicle is idle
        if (m_poseHistory.empty()) {
            fprintf(stderr, "   [fusion #%u] Initializing trajectory (first GPS fix)...\n", fusionCallCount);
            fflush(stderr);
            
            Pose initialPose{};
            initialPose.timestamp = gpsAnchor->first;
            initialPose.rig2world = DW_IDENTITY_TRANSFORMATION3F;
            initialPose.rpy[0] = initialPose.rpy[1] = initialPose.rpy[2] = 0.0f;
            
            memset(&initialPose.uncertainty, 0, sizeof(dwEgomotionUncertainty));
            
            m_poseHistory.push_back(initialPose);
            m_shallRender = true;
            
            char buffer[256];
            sprintf(buffer, " Trajectory initialized from GPS fix:\n"
                            "  Location: [%.6f°N, %.6f°E, %.2fm ASL]\n"
                            "  Timestamp: %lu µs\n"
                            "  System ready for motion tracking.\n",
                    gpsAnchor->second.latitude, 
                    gpsAnchor->second.longitude, 
                    gpsAnchor->second.altitude,
                    gpsAnchor->first);
            printColored(stdout, COLOR_GREEN, buffer);
            
            fprintf(stderr, "   [fusion #%u] Trajectory initialized\n", fusionCallCount);
            fflush(stderr);
        }
        
        dwTime_t anchorTimestamp = gpsAnchor->first;
        
        fprintf(stderr, "   [fusion #%u] Checking timestamp (anchor=%lu, lastProcessed=%lu)\n",
                fusionCallCount, anchorTimestamp, m_temporalBuffers.lastProcessedTimestamp);
        fflush(stderr);
        
        // Skip if we've already processed this timestamp
        if (anchorTimestamp <= m_temporalBuffers.lastProcessedTimestamp) {
            fprintf(stderr, "   [fusion #%u] Already processed, returning false\n", fusionCallCount);
            fflush(stderr);
            return false;
        }
        
        fprintf(stderr, "   [fusion #%u] Finding IMU match...\n", fusionCallCount);
        fflush(stderr);
        
        // Find temporally aligned IMU data
        auto imuMatch = findClosestSensorData(m_temporalBuffers.imuBuffer, anchorTimestamp, TEMPORAL_WINDOW_US);
        if (imuMatch == m_temporalBuffers.imuBuffer.end()) {
            fprintf(stderr, "   [fusion #%u] No IMU match found, returning false\n", fusionCallCount);
            fflush(stderr);
            return false;
        }
        
        fprintf(stderr, "   [fusion #%u] IMU match found: timestamp=%lu\n", 
                fusionCallCount, imuMatch->first);
        fflush(stderr);
        
        // Extract synchronized sensor frames
        dwGPSFrame& gpsFrame = gpsAnchor->second;
        dwIMUFrame& imuFrame = imuMatch->second;

        fprintf(stderr, "   [fusion #%u] Buffer data extracted:\n", fusionCallCount);
        fprintf(stderr, "      GPS: [%.8f, %.8f, %.2fm] @ %lu\n",
                gpsFrame.latitude, gpsFrame.longitude, gpsFrame.altitude, gpsFrame.timestamp_us);
        fprintf(stderr, "      IMU: accel=[%.6f,%.6f,%.6f] gyro=[%.6f,%.6f,%.6f] @ %lu\n",
                imuFrame.acceleration[0], imuFrame.acceleration[1], imuFrame.acceleration[2],
                imuFrame.turnrate[0], imuFrame.turnrate[1], imuFrame.turnrate[2],
                imuFrame.timestamp_us);
        fflush(stderr);
        
        // Update current sensor states for rendering/logging
        m_currentGPSFrame = gpsFrame;
        m_currentIMUFrame = imuFrame;
        
        fprintf(stderr, "   [fusion #%u] Processing synchronized sensor data...\n", fusionCallCount);
        fflush(stderr);
        
        // Feed synchronized data to egomotion
        processSynchronizedSensorData(m_currentGPSFrame, m_currentIMUFrame, anchorTimestamp);
        
        fprintf(stderr, "   [fusion #%u] Sensor data processed\n", fusionCallCount);
        fflush(stderr);
        
        // Handle CAN data synchronization
        fprintf(stderr, "   [fusion #%u] Getting CAN synchronized state...\n", fusionCallCount);
        fflush(stderr);
        
        dwVehicleIOSafetyState safetyState;
        dwVehicleIONonSafetyState nonSafetyState;
        dwVehicleIOActuationFeedback actuationFeedback;

        bool canStateValid = m_canParser->getTemporallySynchronizedState(
            &safetyState, &nonSafetyState, &actuationFeedback);
        
        if (canStateValid) {
            fprintf(stderr, "   [fusion #%u] CAN input:\n", fusionCallCount);
            fprintf(stderr, "      speed=%.3f @ %lu, steering=%.3f @ %lu\n",
                    nonSafetyState.speedESC, nonSafetyState.speedESCTimestamp,
                    safetyState.steeringWheelAngle, safetyState.timestamp_us);
            fflush(stderr);
            
            dwStatus status = dwEgomotion_addVehicleIOState(&safetyState,
                                                            &nonSafetyState,
                                                            &actuationFeedback,
                                                            m_egomotion);
            if (status != DW_SUCCESS) {
                fprintf(stderr, "   [fusion #%u] Failed to add vehicle state: %d\n", 
                        fusionCallCount, status);
            } else {
                fprintf(stderr, "   [fusion #%u] Successfully fed CAN to egomotion\n", 
                        fusionCallCount);
            }
            
            fprintf(stderr, "   [fusion #%u] CAN state processed\n", fusionCallCount);
            fflush(stderr);
        }
        
        fprintf(stderr, "   [fusion #%u] Updating last processed timestamp...\n", fusionCallCount);
        fflush(stderr);
        
        m_temporalBuffers.lastProcessedTimestamp = anchorTimestamp;
        m_elapsedTime = anchorTimestamp - m_firstTimestamp;
        
        fprintf(stderr, "   [fusion #%u] Returning TRUE\n", fusionCallCount);
        fflush(stderr);
        
        return true;
    }
    
    /**
     * Processes synchronized GPS and IMU data
     */
    void processSynchronizedSensorData(const dwGPSFrame& gpsFrame, const dwIMUFrame& imuFrame, dwTime_t timestamp) {
        // GPS processing (from original immediate processing)
        m_trajectoryLog.addWGS84("GPS", gpsFrame);
            dwStatus gpsStatus = dwGlobalEgomotion_addGPSMeasurement(&gpsFrame, m_globalEgomotion);
            fprintf(stderr, "         GPS acceptance: %d (%s)\n", gpsStatus, dwGetStatusName(gpsStatus));
            fflush(stderr);

        if (m_egomotionParameters.motionModel != DW_EGOMOTION_ODOMETRY) {
            dwEgomotion_addIMUMeasurement(&imuFrame, m_egomotion);
            dwStatus imuStatus = dwEgomotion_addIMUMeasurement(&imuFrame, m_egomotion);
                fprintf(stderr, "         IMU acceptance: %d (%s)\n", imuStatus, dwGetStatusName(imuStatus));
                fflush(stderr);
        }
        
        static dwTime_t lastGpsLog = 0, lastImuLog = 0;
        
        if (allowEvery(timestamp, lastGpsLog, 200000)) {
            char buffer[512];
            sprintf(buffer, "GPS Data: Lat=%.6f°, Lon=%.6f°, Alt=%.2fm, Speed=%.2fm/s, Timestamp=%lu\n",
                    gpsFrame.latitude, gpsFrame.longitude, gpsFrame.altitude,
                    gpsFrame.speed, gpsFrame.timestamp_us);
            printColored(stdout, COLOR_GREEN, buffer);
        }
        
        if (allowEvery(timestamp, lastImuLog, 200000)) {
            char buffer[512];
            sprintf(buffer, "IMU Data: Accel=[%.3f, %.3f, %.3f] m/s², Gyro=[%.3f, %.3f, %.3f] rad/s, Timestamp=%lu\n",
                    imuFrame.acceleration[0], imuFrame.acceleration[1], imuFrame.acceleration[2],
                    imuFrame.turnrate[0], imuFrame.turnrate[1], imuFrame.turnrate[2],
                    imuFrame.timestamp_us);
            printColored(stdout, COLOR_DEFAULT, buffer);
        }
    }
    
    /**
     * Finds the most recent valid GPS measurement for temporal anchoring
     */
    std::map<dwTime_t, dwGPSFrame>::iterator findLatestValidGPS(dwTime_t currentTime) {
        if (m_temporalBuffers.gpsBuffer.empty()) {
            return m_temporalBuffers.gpsBuffer.end();
        }
        
        // Find GPS data within reasonable recency (allow up to 200ms latency)
        dwTime_t earliestAcceptable = currentTime - 200000;
        auto it = m_temporalBuffers.gpsBuffer.lower_bound(earliestAcceptable);
        
        if (it != m_temporalBuffers.gpsBuffer.end()) {
            // Return the most recent GPS measurement
            return std::prev(m_temporalBuffers.gpsBuffer.end());
        }
        
        return m_temporalBuffers.gpsBuffer.end();
    }
    
    /**
     * Generic template to find closest sensor data within temporal window
     */
    template<typename SensorMap>
    typename SensorMap::iterator findClosestSensorData(SensorMap& sensorMap, dwTime_t targetTime, dwTime_t maxWindow) {
        if (sensorMap.empty()) {
            return sensorMap.end();
        }
        
        auto targetIt = sensorMap.lower_bound(targetTime);
        typename SensorMap::iterator bestMatch = sensorMap.end();
        dwTime_t bestDistance = maxWindow + 1;
        
        // Check forward direction
        if (targetIt != sensorMap.end()) {
            dwTime_t forwardDist = std::abs(static_cast<int64_t>(targetIt->first - targetTime));
            if (forwardDist <= maxWindow && forwardDist < bestDistance) {
                bestMatch = targetIt;
                bestDistance = forwardDist;
            }
        }
        
        // Check backward direction
        if (targetIt != sensorMap.begin()) {
            auto backIt = std::prev(targetIt);
            dwTime_t backwardDist = std::abs(static_cast<int64_t>(backIt->first - targetTime));
            if (backwardDist <= maxWindow && backwardDist < bestDistance) {
                bestMatch = backIt;
                bestDistance = backwardDist;
            }
        }
        
        return bestMatch;
    }
    bool isValidIMU(const dwIMUFrame& imuFrame) {
        // Method 1: Check magnitude of acceleration (should be near gravity ~9.81 m/s²)
        float accelMagnitude = std::sqrt(
            imuFrame.acceleration[0] * imuFrame.acceleration[0] +
            imuFrame.acceleration[1] * imuFrame.acceleration[1] +
            imuFrame.acceleration[2] * imuFrame.acceleration[2]
        );
        
        // For stationary vehicle, expect accel magnitude close to 9.81 m/s² (gravity)
        // Allow range: 8.0 to 11.0 m/s² to account for vehicle tilt and sensor noise
        const float MIN_ACCEL = 8.0f;
        const float MAX_ACCEL = 11.0f;
        
        bool accelValid = (accelMagnitude >= MIN_ACCEL && accelMagnitude <= MAX_ACCEL);
        
        // Method 2: Check if all values are exactly zero (placeholder frame)
        bool isZeroFrame = (imuFrame.acceleration[0] == 0.0f &&
                        imuFrame.acceleration[1] == 0.0f &&
                        imuFrame.acceleration[2] == 0.0f &&
                        imuFrame.turnrate[0] == 0.0f &&
                        imuFrame.turnrate[1] == 0.0f &&
                        imuFrame.turnrate[2] == 0.0f);
        
        // Method 3: Check validity flags (if available)
        // bool flagsValid = (imuFrame.validityFlags & DW_IMU_ACCELERATION_VALID) != 0;
        
        return accelValid && !isZeroFrame;
    }

    bool drainSensorEventsToBuffers() {
            static uint32_t drainageCallCount = 0;
            static dwTime_t lastDrainageReport = 0;
            static dwTime_t drainageStart = 0;
            static bool catchUpMode = false;
            static uint32_t normalDrainCounter = 0;
            
            dwTime_t now = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            
            if (drainageStart == 0) {
                drainageStart = now;
            }
            
            ++drainageCallCount;

            if (now - lastDrainageReport > 1000000) {
                float32_t callRate = (drainageCallCount * 1000000.0f) / (now - drainageStart);
                char buf[256];
                sprintf(buf, " Drainage loop: %u calls, %.1f Hz\n", drainageCallCount, callRate);
                printColored(stdout, COLOR_DEFAULT, buf);
                lastDrainageReport = now;
            }
            
            static uint32_t canCount = 0, imuCount = 0, gpsCount = 0, cameraCount = 0;
            static dwTime_t lastReport = 0;
            
            if (now - lastReport > 1000000) {
                char buf[512];
                sprintf(buf, " Sensor events (last 1s): CAN=%u, IMU=%u, GPS=%u, Camera=%u\n",
                        canCount, imuCount, gpsCount, cameraCount);
                printColored(stdout, COLOR_YELLOW, buf);
                
                if (canCount == 0 && imuCount == 0 && gpsCount == 0) {
                    printColored(stdout, COLOR_RED, "WARNING: No sensor events drained!\n");
                }
                
                canCount = imuCount = gpsCount = cameraCount = 0;
                lastReport = now;
            }
            
            int totalDrained = 0;
            int passCount = 0;
            bool hitLimit = false;
            
            auto overallStart = std::chrono::steady_clock::now();
            
            int budgetUs = catchUpMode ? 8000 : 5000;
            auto totalBudgetEnd = overallStart + std::chrono::microseconds(budgetUs);
            
            while (passCount < 5 && !isPaused()) {
                if (std::chrono::steady_clock::now() >= totalBudgetEnd) {
                    break;  // Total budget exhausted
                }
                
                int drained = 0;
                std::vector<std::pair<dwTime_t, dwGPSFrame>> gpsUpdates;
                std::vector<std::pair<dwTime_t, dwIMUFrame>> imuUpdates;
                std::vector<std::pair<dwTime_t, dwCANMessage>> canUpdates;
                
                while (!isPaused() && drained < 128) {
                    if (std::chrono::steady_clock::now() >= totalBudgetEnd) {
                        break;  // Budget check
                    }
                    
                    dwStatus status = dwSensorManager_acquireNextEvent(&acquiredEvent, 0, m_sensorManager);
                    
                    if (status == DW_TIME_OUT) {
                        break;
                    }
                    
                    if (status != DW_SUCCESS) {
                        if (status != DW_END_OF_STREAM) {
                            logError("Sensor drainage error: %s\n", dwGetStatusName(status));
                        } else if (should_AutoExit()) {
                            stop();
                            return false;
                        }
                        break;
                    }
                    
                    dwTime_t timestamp = acquiredEvent->timestamp_us;
                    
                    if (m_firstTimestamp == 0) {
                        m_firstTimestamp = timestamp;
                        m_lastSampleTimestamp = timestamp;
                    }
                    
                    switch (acquiredEvent->type) {
                        case DW_SENSOR_GPS:
                            if (acquiredEvent->sensorIndex == m_gpsSensorIdx) {
                                gpsUpdates.emplace_back(timestamp, acquiredEvent->gpsFrame);
                            }
                            gpsCount++;
                            break;
                            
                        case DW_SENSOR_IMU:
                            if (acquiredEvent->sensorIndex == m_imuSensorIdx) {
                                bool isValidIMUFrame = isValidIMU(acquiredEvent->imuFrame);
                                if (isValidIMUFrame) {
                                    imuUpdates.emplace_back(timestamp, acquiredEvent->imuFrame);
                                } else {
                                    fprintf(stderr, "           → Skipping zero/invalid IMU frame\n");
                                    fflush(stderr);
                                }
                            }
                            imuCount++;
                            break;
                            
                        case DW_SENSOR_CAN:
                            if (acquiredEvent->sensorIndex == m_vehicleSensorIdx) {
                                canUpdates.emplace_back(timestamp, acquiredEvent->canFrame);
                                m_canParser->processCANFrame(acquiredEvent->canFrame);
                                
                                static dwTime_t lastTimeoutCheck = 0;
                                if (timestamp - lastTimeoutCheck > 50000) {
                                    m_canParser->checkMessageTimeouts(timestamp);
                                    lastTimeoutCheck = timestamp;
                                }
                            }
                            canCount++;
                            break;
                            
                        case DW_SENSOR_CAMERA:
                            cameraCount++;
                            break;
                            
                        default:
                            break;
                    }
                    
                    dwSensorManager_releaseAcquiredEvent(acquiredEvent, m_sensorManager);
                    acquiredEvent = nullptr;
                    ++drained;
                }
                
                // Apply updates
                if (!gpsUpdates.empty() || !imuUpdates.empty() || !canUpdates.empty()) {
                    std::lock_guard<std::mutex> lock(m_temporalBuffers.bufferMutex);
                    
                    for (auto& update : gpsUpdates) {
                        if (update.first > m_temporalBuffers.lastGPSTimestamp) {
                            m_temporalBuffers.gpsBuffer[update.first] = update.second;
                            m_temporalBuffers.lastGPSTimestamp = update.first;
                        }
                    }
                    
                    for (auto& update : imuUpdates) {
                        if (update.first > m_temporalBuffers.lastIMUTimestamp) {
                            m_temporalBuffers.imuBuffer[update.first] = update.second;
                            m_temporalBuffers.lastIMUTimestamp = update.first;
                        }
                    }
                    
                    for (auto& update : canUpdates) {
                        m_temporalBuffers.canBuffer[update.first] = update.second;
                        m_temporalBuffers.lastCANTimestamp = update.first;
                    }
                    
                    if (!gpsUpdates.empty()) {
                        m_temporalBuffers.cleanupOldData(gpsUpdates.back().first);
                    } else if (!imuUpdates.empty()) {
                        m_temporalBuffers.cleanupOldData(imuUpdates.back().first);
                    } else if (!canUpdates.empty()) {
                        m_temporalBuffers.cleanupOldData(canUpdates.back().first);
                    }
                }
                
                totalDrained += drained;
                passCount++;
                
                if (drained >= 128) {
                    hitLimit = true;
                } else {
                    break;  // No more data
                }
            }
            
            // Update catch-up state
            if (hitLimit) {
                if (!catchUpMode) {
                    catchUpMode = true;
                    fprintf(stderr, " CATCH-UP MODE ENTER\n");
                    fflush(stderr);
                }
                normalDrainCounter = 0;
            } else {
                normalDrainCounter++;
                if (catchUpMode && normalDrainCounter >= 3) {
                    catchUpMode = false;
                    normalDrainCounter = 0;
                    fprintf(stderr, " CATCH-UP MODE EXIT\n");
                    fflush(stderr);
                }
            }
            
            // Logging every 10 cycles
            static uint32_t cycleCounter = 0;
            ++cycleCounter;
            auto overallTime = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - overallStart).count();
            
            if (cycleCounter % 10 == 0 || catchUpMode) {
                fprintf(stderr, " Cycle #%u: %d events in %.1fms over %d passes%s\n",
                        cycleCounter, totalDrained, overallTime / 1000.0f, passCount,
                        catchUpMode ? " [CATCH-UP]" : "");
                fflush(stderr);
            }
            
            return catchUpMode;
        }

    void bufferGPSData(const dwGPSFrame& gpsFrame) {
        dwTime_t timestamp = gpsFrame.timestamp_us;
        
        // NVIDIA requirement: ignore non-monotonic timestamps  
        if (timestamp <= m_temporalBuffers.lastGPSTimestamp) {
            logWarn("GPS: ignoring non-monotonic timestamp %lu (last: %lu)\n", 
                    timestamp, m_temporalBuffers.lastGPSTimestamp);
            return;
        }
        
        std::lock_guard<std::mutex> lock(m_temporalBuffers.bufferMutex);
        m_temporalBuffers.gpsBuffer[timestamp] = gpsFrame;
        m_temporalBuffers.lastGPSTimestamp = timestamp;
        m_temporalBuffers.cleanupOldData(timestamp);
    }
    
    /**
     * Buffer IMU data with timestamp validation
     */
    void bufferIMUData(const dwIMUFrame& imuFrame) {
        dwTime_t timestamp = imuFrame.timestamp_us;
        
        if (timestamp <= m_temporalBuffers.lastIMUTimestamp) {
            return; // Silently ignore (IMU data is high-frequency)
        }
        
        std::lock_guard<std::mutex> lock(m_temporalBuffers.bufferMutex);
        m_temporalBuffers.imuBuffer[timestamp] = imuFrame;
        m_temporalBuffers.lastIMUTimestamp = timestamp;
        m_temporalBuffers.cleanupOldData(timestamp);
    }
    
    /**
     * Buffer CAN data and feed to parser
     */
    void bufferCANData(const dwCANMessage& canFrame) {
        if (acquiredEvent->sensorIndex != m_vehicleSensorIdx) {
            return; // Wrong sensor
        }
        
        // Feed to parser immediately (CAN parsing is fast)
        m_canParser->processCANFrame(canFrame);
        
        // Periodic timeout check
        static dwTime_t lastTimeoutCheck = 0;
        dwTime_t timestamp = canFrame.timestamp_us;
        if (timestamp - lastTimeoutCheck > 50000) {
            m_canParser->checkMessageTimeouts(timestamp);
            lastTimeoutCheck = timestamp;
        }
        
        // Optional: store in buffer for advanced temporal correlation
        std::lock_guard<std::mutex> lock(m_temporalBuffers.bufferMutex);
        m_temporalBuffers.canBuffer[timestamp] = canFrame;
        m_temporalBuffers.lastCANTimestamp = timestamp;
        m_temporalBuffers.cleanupOldData(timestamp);
    }
    
    /**
     * Immediate camera processing (unchanged)
     */
    void processCameraImmediate() {
        if (m_lastGLFrame == DW_NULL_HANDLE) {
            dwImageHandle_t nextFrame = DW_NULL_HANDLE;
            dwSensorCamera_getImage(&nextFrame, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, acquiredEvent->camFrames[0]);
            
            m_lastGLFrame = m_streamerInput2GL->post(nextFrame);
            dwImage_destroy(nextFrame);
            m_shallRender = true;
        }
    }

public:
    EgomotionSample(const ProgramArguments& args)
        : DriveWorksSample(args)
    {
        if (getArgument("output").length() > 0)
        {
            m_outputFile = fopen(getArgument("output").c_str(), "wt");
            log("Real-time output file opened: %s\n", getArgument("output").c_str());
        }

        getMouseView().setRadiusFromCenter(25.0f);
    }

    void initializeDriveWorks(dwContextHandle_t& context) const
    {
        CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
        CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_INFO));

        dwContextParameters sdkParams = {};
#ifdef VIBRANTE
        sdkParams.eglDisplay = getEGLDisplay();
#endif
        CHECK_DW_ERROR(dwInitialize(&context, DW_VERSION, &sdkParams));
    }

    bool onInitialize() override
    {
        dwSensorManagerParams smParams{};
        dwSensorType vehicleSensorType{};

        // Initialize DriveWorks SDK context and SAL
        {
            initializeDriveWorks(m_context);
            CHECK_DW_ERROR(dwVisualizationInitialize(&m_vizCtx, m_context));
        }

        // Read Rig file to extract vehicle properties
        {
            dwStatus ret = dwRig_initializeFromFile(&m_rigConfig, m_context, getArgument("rig").c_str());
            if (ret != DW_SUCCESS)
                throw std::runtime_error("Error reading rig config for real-time processing");

            std::string imuSensorName     = getArgument("imu-sensor-name");
            std::string vehicleSensorName = getArgument("vehicle-sensor-name");
            std::string gpsSensorName     = getArgument("gps-sensor-name");
            std::string cameraSensorName  = getArgument("camera-sensor-name");

            // Extract sensor names from rig config
            {
                uint32_t cnt;
                dwRig_getSensorCount(&cnt, m_rigConfig);
                char buffer[256]; // Declare buffer once for the entire loop scope
                
                for (uint32_t i = 0; i < cnt; i++)
                {
                    dwSensorType type;
                    const char* name;
                    dwRig_getSensorType(&type, i, m_rigConfig);
                    dwRig_getSensorName(&name, i, m_rigConfig);

                    if (type == DW_SENSOR_IMU && imuSensorName.length() == 0)
                    {
                        imuSensorName = name;
                        sprintf(buffer, "[SENSORS IN RIG] IMU Sensor found: %s\n", name);
                        printColored(stdout, COLOR_GREEN, buffer);
                    }
                    if ((type == DW_SENSOR_CAN || type == DW_SENSOR_DATA) && vehicleSensorName.length() == 0)
                    {
                        vehicleSensorName = name;
                        sprintf(buffer, "[SENSORS IN RIG] Vehicle CAN interface found: %s\n", name);
                        printColored(stdout, COLOR_GREEN, buffer);
                    }
                    if (type == DW_SENSOR_GPS && gpsSensorName.length() == 0)
                    {
                        gpsSensorName = name;
                        sprintf(buffer, "[SENSORS IN RIG] GPS Sensor found: %s\n", name);
                        printColored(stdout, COLOR_GREEN, buffer);
                    }
                    if (type == DW_SENSOR_CAMERA && cameraSensorName.length() == 0)
                    {
                        cameraSensorName = name;
                        sprintf(buffer, "[SENSORS IN RIG] Camera Sensor found, Rig intake name: %s\n", name);
                        printColored(stdout, COLOR_GREEN, buffer);
                    }
                }
            }

            // Get sensor indices
            CHECK_DW_ERROR_MSG(dwRig_findSensorByName(&m_imuSensorIdx, imuSensorName.c_str(), m_rigConfig), 
                              "Cannot find IMU sensor for real-time processing");
            CHECK_DW_ERROR_MSG(dwRig_findSensorByName(&m_vehicleSensorIdx, vehicleSensorName.c_str(), m_rigConfig), 
                              "Cannot find vehicle sensor for real-time processing");

            smParams.enableSensors[smParams.numEnableSensors++] = m_vehicleSensorIdx;
            smParams.enableSensors[smParams.numEnableSensors++] = m_imuSensorIdx;

            CHECK_DW_ERROR_MSG(dwRig_getSensorType(&vehicleSensorType, m_vehicleSensorIdx, m_rigConfig), 
                              "Cannot determine vehicle sensor type");

            // Initialize egomotion parameters
            dwEgomotion_initParamsFromRig(&m_egomotionParameters, m_rigConfig, imuSensorName.c_str(), nullptr);

            // GPS sensor (optional)
            if (dwRig_findSensorByName(&m_gpsSensorIdx, gpsSensorName.c_str(), m_rigConfig) == DW_SUCCESS)
            {
                dwGlobalEgomotion_initParamsFromRig(&m_globalEgomotionParameters, m_rigConfig, gpsSensorName.c_str());
                smParams.enableSensors[smParams.numEnableSensors++] = m_gpsSensorIdx;
            }
            else
            {
                logWarn("GPS sensor not found - global egomotion will not be available\n");
            }

            // Camera sensor (optional)
            uint32_t cameraSensorId = 0;
            if (dwRig_findSensorByName(&cameraSensorId, cameraSensorName.c_str(), m_rigConfig) == DW_SUCCESS)
            {
                smParams.enableSensors[smParams.numEnableSensors++] = cameraSensorId;
            }
            else
            {
                logWarn("Camera sensor not found - no video display will be available\n");
            }
        }
        

        // Initialize Egomotion module
        {
            if (getArgument("mode") == "0")
                m_egomotionParameters.motionModel = DW_EGOMOTION_ODOMETRY;
            else if (getArgument("mode") == "1")
            {
                m_egomotionParameters.motionModel = DW_EGOMOTION_IMU_ODOMETRY;
                m_egomotionParameters.estimateInitialOrientation = true;
            }
            else
            {
                logError("Invalid mode %s for real-time processing\n", getArgument("mode").c_str());
                return false;
            }

            m_egomotionParameters.automaticUpdate = true;
            auto speedType = std::stoi(getArgument("speed-measurement-type"));
            m_egomotionParameters.speedMeasurementType = dwEgomotionSpeedMeasurementType(speedType);

            if (getArgument("enable-suspension") == "1")
            {
                if (m_egomotionParameters.motionModel == DW_EGOMOTION_IMU_ODOMETRY)
                {
                    m_egomotionParameters.suspension.model = DW_EGOMOTION_SUSPENSION_TORSIONAL_SPRING_MODEL;
                }
                else
                {
                    logError("Suspension model requires Odometry+IMU mode (--mode=1)\n");
                    return false;
                }
            }

            dwStatus status = dwEgomotion_initialize(&m_egomotion, &m_egomotionParameters, m_context);
            if (status != DW_SUCCESS)
            {
                logError("Error initializing real-time egomotion: %s\n", dwGetStatusName(status));
                return false;
            }
        }

        // Initialize Global Egomotion module
        {
            dwStatus status = dwGlobalEgomotion_initialize(&m_globalEgomotion, &m_globalEgomotionParameters, m_context);
            if (status != DW_SUCCESS)
            {
                logError("Error initializing global egomotion: %s\n", dwGetStatusName(status));
                return false;
            }
        }

        // Initialize Sensors and Real-time CAN Parser
        {
            CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_context));

            dwStatus status = dwSensorManager_initializeFromRigWithParams(&m_sensorManager, m_rigConfig, &smParams, 128, m_sal);
            if (status != DW_SUCCESS)
            {
                logError("Error initializing SensorManager for real-time processing: %s\n", dwGetStatusName(status));
                return false;
            }

            // Initialize Real-time CAN Parser (replaces VehicleIO)
            {
                m_canParser = std::make_unique<SygnalPomoParser>();
                
                // Load Hyundai/Kia configuration for real-time processing
                SygnalPomoParser::VehicleCANConfiguration realtimeConfig;
                // Use default Hyundai/Kia CAN IDs and scaling factors
                
                if (!m_canParser->loadVehicleConfiguration(realtimeConfig)) {
                    logError("Failed to initialize real-time CAN parser\n");
                    return false;
                }
                
                // Configure speed measurement type
                auto speedType = std::stoi(getArgument("speed-measurement-type"));
                m_canParser->configureSpeedMeasurementType(
                    static_cast<dwEgomotionSpeedMeasurementType>(speedType));
                
                log("SYGNALPOMO CAN PARSER initialized successfully\n");
                log("CAN-ID Config: Speed=0x%03X, Steering=0x%03X, Wheels=0x%03X\n", 
                    realtimeConfig.speedCANId, realtimeConfig.steeringWheelAngleCANId, realtimeConfig.wheelSpeedCANId);
            }

            // Initialize camera processing (if available)
            uint32_t cnt;
            dwSensorManager_getNumSensors(&cnt, DW_SENSOR_CAMERA, m_sensorManager);
            if (cnt == 1)
            {
                uint32_t cameraSensorIndex{};
                CHECK_DW_ERROR(dwSensorManager_getSensorIndex(&cameraSensorIndex, DW_SENSOR_CAMERA, 0, m_sensorManager));
                dwSensorHandle_t cameraSensor = DW_NULL_HANDLE;
                CHECK_DW_ERROR(dwSensorManager_getSensorHandle(&cameraSensor, cameraSensorIndex, m_sensorManager));

                dwCameraProperties cameraProperties{};
                dwImageProperties outputProperties{};

                CHECK_DW_ERROR(dwSensorCamera_getSensorProperties(&cameraProperties, cameraSensor));
                // Request CUDA RGBA format directly - no manual conversion needed
                CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&outputProperties, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, cameraSensor));

                std::cout << "Camera image with " << cameraProperties.resolution.x << "x"
                        << cameraProperties.resolution.y << " at " << cameraProperties.framerate << " FPS" << std::endl;

                // Remove conversion image creation - not needed
                m_streamerInput2GL.reset(new SimpleImageStreamerGL<>(outputProperties, 12, m_context));
            }

            // Start sensor manager for real-time processing
            if (dwSensorManager_start(m_sensorManager) != DW_SUCCESS)
            {
                logError("Failed to start SensorManager for real-time processing\n");
                dwSensorManager_release(m_sensorManager);
                return false;
            }
        }

        // Initialize rendering subsystem
        {
            CHECK_DW_ERROR(dwRenderer_initialize(&m_renderer, m_vizCtx));

            dwRect rect;
            rect.width = getWindowWidth();
            rect.height = getWindowHeight();
            rect.x = 0;
            rect.y = 0;
            dwRenderer_setRect(rect, m_renderer);

            dwRenderEngineParams params{};
            params.bufferSize = sizeof(Pose) * MAX_BUFFER_POINTS;
            params.bounds = {0, 0, static_cast<float32_t>(getWindowWidth()), static_cast<float32_t>(getWindowHeight())};

            {
                dwRenderEngine_initTileState(&params.defaultTile);
                params.defaultTile.layout.viewport = params.bounds;
            }
            CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_vizCtx));

            dwRenderEngineTileState tileParams = params.defaultTile;
            tileParams.projectionMatrix = DW_IDENTITY_MATRIX4F;
            tileParams.modelViewMatrix = DW_IDENTITY_MATRIX4F;
            tileParams.layout.positionLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_ABSOLUTE;
            tileParams.layout.sizeLayout = DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_ABSOLUTE;

            // Video tile
            {
                tileParams.layout.viewport = {0.f, 0.f, getWindowWidth() / 5.0f, getWindowHeight() / 5.0f};
                tileParams.layout.positionType = DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_LEFT;
                dwRenderEngine_addTile(&m_tileVideo, &tileParams, m_renderEngine);
            }

            // Plot tiles
            {
                const float32_t plotWidth = getWindowWidth() / 4.0f;
                const float32_t plotHeight = getWindowHeight() / 4.0f;

                tileParams.layout.viewport = {0.f, 2 * plotHeight, plotWidth, plotHeight};
                tileParams.layout.positionType = DW_RENDER_ENGINE_TILE_POSITION_TYPE_BOTTOM_RIGHT;
                dwRenderEngine_addTile(&m_tileRollPlot, &tileParams, m_renderEngine);

                tileParams.layout.viewport = {0.f, plotHeight, plotWidth, plotHeight};
                dwRenderEngine_addTile(&m_tilePitchPlot, &tileParams, m_renderEngine);

                tileParams.layout.viewport = {0.f, 0.f, plotWidth, plotHeight};
                dwRenderEngine_addTile(&m_tileAltitudePlot, &tileParams, m_renderEngine);
            }
        }

        // Initialize trajectory logger
        {
            m_trajectoryLog.addTrajectory("GPS", TrajectoryLogger::Color::GREEN);
            m_trajectoryLog.addTrajectory("Egomotion", TrajectoryLogger::Color::RED);
        }

        log("Real-time Egomotion sample initialized successfully\n");
        return true;
    }

    void onRelease() override
    {
        if (acquiredEvent)
            dwSensorManager_releaseAcquiredEvent(acquiredEvent, m_sensorManager);

        if (getArgument("outputkml").length())
            m_trajectoryLog.writeKML(getArgument("outputkml"));

        if (m_outputFile)
            fclose(m_outputFile);

        if (m_convertedImageRGBA)
            dwImage_destroy(m_convertedImageRGBA);

        // Release real-time CAN parser
        if (m_canParser) {
            m_canParser.reset();
        }

        dwSensorManager_stop(m_sensorManager);
        dwSensorManager_release(m_sensorManager);
        if (m_streamerInput2GL && m_lastGLFrame != DW_NULL_HANDLE) {
            m_streamerInput2GL->release();
            m_lastGLFrame = DW_NULL_HANDLE;
        }
        m_streamerInput2GL.reset();

        dwGlobalEgomotion_release(m_globalEgomotion);
        dwEgomotion_release(m_egomotion);
        dwRig_release(m_rigConfig);

        if (m_renderer)
            dwRenderer_release(m_renderer);

        if (m_renderEngine != DW_NULL_HANDLE)
            CHECK_DW_ERROR(dwRenderEngine_release(m_renderEngine));

        dwSAL_release(m_sal);

        CHECK_DW_ERROR(dwVisualizationRelease(m_vizCtx));
        CHECK_DW_ERROR(dwRelease(m_context));
        CHECK_DW_ERROR(dwLogger_release());

        log("Real-time Egomotion sample released\n");
    }

    void onResizeWindow(int width, int height) override
    {
        {
            dwRect rect;
            rect.width = width;
            rect.height = height;
            rect.x = 0;
            rect.y = 0;
            dwRenderer_setRect(rect, m_renderer);
        }

        {
            dwRenderEngine_reset(m_renderEngine);
            dwRectf rect;
            rect.width = width;
            rect.height = height;
            rect.x = 0;
            rect.y = 0;
            dwRenderEngine_setBounds(rect, m_renderEngine);
        }

        log("Real-time window resized to %dx%d\n", width, height);
    }

    void render3DGrid()
    {
        if (m_poseHistory.empty())
            return;

        dwRenderEngine_setTile(m_tileGrid, m_renderEngine);

        const Pose& currentPose = m_poseHistory.back();
        auto currentRig2World = currentPose.rig2world;

        dwMatrix4f modelView;
        {
            if (m_renderingMode == RenderingMode::STICK_TO_VEHICLE)
            {
                dwTransformation3f world2rig;
                Mat4_IsoInv(world2rig.array, currentRig2World.array);
                Mat4_AxB(modelView.array, getMouseView().getModelView()->array, world2rig.array);
            }
            else if (m_renderingMode == RenderingMode::ON_VEHICLE_STICK_TO_WORLD)
            {
                float32_t center[3] = {currentRig2World.array[0 + 3 * 4],
                                       currentRig2World.array[1 + 3 * 4],
                                       currentRig2World.array[2 + 3 * 4]};

                getMouseView().setCenter(center[0], center[1], center[2]);
                Mat4_AxB(modelView.array, getMouseView().getModelView()->array, DW_IDENTITY_TRANSFORMATION3F.array);
            }
        }

        dwRenderEngine_setModelView(&modelView, m_renderEngine);
        dwRenderEngine_setProjection(getMouseView().getProjection(), m_renderEngine);
        dwRenderEngine_setBackgroundColor({0.0f, 0.0f, 0.0f, 1.0f}, m_renderEngine);

        // Render trajectory path
        dwRenderEngine_setColor({0.0f, 1.0f, 0.0f, 1.0f}, m_renderEngine);
        dwRenderEngine_setLineWidth(2.f, m_renderEngine);
        dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINESTRIP_3D,
                              m_poseHistory.data(),
                              sizeof(Pose),
                              offsetof(Pose, rig2world) + 3 * 4 * sizeof(float32_t),
                              m_poseHistory.size(),
                              m_renderEngine);

        // Render vehicle coordinate system
        {
            dwRenderEngine_setLineWidth(2.0f, m_renderEngine);

            for (int i = 0; i < 3; i++)
            {
                float32_t localAxis[3] = {
                    i == 0 ? 1.f : 0.f,
                    i == 1 ? 1.f : 0.f,
                    i == 2 ? 1.f : 0.f};
                dwVector3f arrow[2];

                arrow[0].x = currentRig2World.array[0 + 3 * 4];
                arrow[0].y = currentRig2World.array[1 + 3 * 4];
                arrow[0].z = currentRig2World.array[2 + 3 * 4];

                Mat4_Axp(reinterpret_cast<float32_t*>(&arrow[1]), currentRig2World.array, localAxis);

                dwRenderEngine_setColor({localAxis[0], localAxis[1], localAxis[2], 1.0f}, m_renderEngine);
                dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_ARROWS_3D,
                                      arrow,
                                      sizeof(dwVector3f) * 2,
                                      0,
                                      1,
                                      m_renderEngine);
            }
        }

        // Render GPS north reference (if available)
        if (m_hasOrientationENU)
        {
            dwTransformation3f rig2Enu = rigidTransformation(m_orientationENU, {0.f, 0.f, 0.f});

            dwVector3f arrow[2];
            {
                dwVector3f arrowENU[2];
                dwVector3f arrowRig[2];

                arrowENU[0] = {0, 0, 0};
                arrowENU[1] = {0, 2, 0};

                Mat4_Rtxp(reinterpret_cast<float32_t*>(&arrowRig[0]), rig2Enu.array, reinterpret_cast<float32_t*>(&arrowENU[0]));
                Mat4_Rtxp(reinterpret_cast<float32_t*>(&arrowRig[1]), rig2Enu.array, reinterpret_cast<float32_t*>(&arrowENU[1]));

                Mat4_Axp(reinterpret_cast<float32_t*>(&arrow[0]), currentRig2World.array, reinterpret_cast<float32_t*>(&arrowRig[0]));
                Mat4_Axp(reinterpret_cast<float32_t*>(&arrow[1]), currentRig2World.array, reinterpret_cast<float32_t*>(&arrowRig[1]));
            }

            const char* labels[] = {"GPS NORTH"};

            dwRenderEngine_setLineWidth(2.0f, m_renderEngine);
            dwRenderEngine_setColor({0.8f, 0.3f, 0.05f, 1.0f}, m_renderEngine);
            dwRenderEngine_renderWithLabels(DW_RENDER_ENGINE_PRIMITIVE_TYPE_ARROWS_3D,
                                            arrow,
                                            sizeof(dwVector3f) * 2,
                                            0,
                                            labels,
                                            1,
                                            m_renderEngine);
        }

        // Render world and local grids
        {
            dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 0.3f}, m_renderEngine);
            dwRenderEngine_setLineWidth(0.5f, m_renderEngine);
            dwRenderEngine_renderPlanarGrid3D({0, 0, 1100.0f, 1100.0f}, 10.f, 10.f, &DW_IDENTITY_MATRIX4F, m_renderEngine);

            dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 0.1f}, m_renderEngine);
            dwRenderEngine_setLineWidth(0.25f, m_renderEngine);
            dwRenderEngine_renderPlanarGrid3D({0, 0, 1100.0f, 1100.0f}, 1.f, 1.f, &DW_IDENTITY_MATRIX4F, m_renderEngine);

            dwMatrix4f modelView = dwMakeMatrix4f(currentRig2World);
            dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, m_renderEngine);
            dwRenderEngine_setLineWidth(2.f, m_renderEngine);
            dwRenderEngine_renderPlanarGrid3D({0, 0, 2.1f, 1.0f}, 2.1f, 1.0f, &modelView, m_renderEngine);
        }
    }

    void renderText()
    {
        const dwVector2i origin = {10, 500};
        char sbuffer[256];

        dwRenderer_setFont(DW_RENDER_FONT_VERDANA_20, m_renderer);

        // Sample information
        {
            dwRenderer_renderText(origin.x, origin.y, "REAL-TIME EGOMOTION SAMPLE", m_renderer);
            dwRenderer_renderText(origin.x, origin.y - 30, "F1 - camera on rig", m_renderer);
            dwRenderer_renderText(origin.x, origin.y - 50, "F2 - camera following rig in world", m_renderer);
            dwRenderer_renderText(origin.x, origin.y - 70, "SPACE - pause", m_renderer);
            dwRenderer_renderText(origin.x, origin.y - 85, "__________________________", m_renderer);
        }

        // Motion model information
        {
            dwMotionModel motionModel;
            dwEgomotion_getMotionModel(&motionModel, m_egomotion);
            if (motionModel == DW_EGOMOTION_ODOMETRY)
                dwRenderer_renderText(origin.x, origin.y - 120, "Motion model: ODOMETRY", m_renderer);
            else if (motionModel == DW_EGOMOTION_IMU_ODOMETRY)
                dwRenderer_renderText(origin.x, origin.y - 120, "Motion model: ODOMETRY+IMU", m_renderer);
        }

        // Speed measurement type
        {
            if (m_egomotionParameters.speedMeasurementType == DW_EGOMOTION_FRONT_SPEED)
                dwRenderer_renderText(origin.x, origin.y - 140, "Speed measurement: front linear speed", m_renderer);
            else if (m_egomotionParameters.speedMeasurementType == DW_EGOMOTION_REAR_SPEED)
                dwRenderer_renderText(origin.x, origin.y - 140, "Speed measurement: rear linear speed", m_renderer);
            else if (m_egomotionParameters.speedMeasurementType == DW_EGOMOTION_REAR_WHEEL_SPEED)
                dwRenderer_renderText(origin.x, origin.y - 140, "Speed measurement: rear wheel angular speed", m_renderer);
        }

        // Suspension model status
        {
            if (m_egomotionParameters.suspension.model == DW_EGOMOTION_SUSPENSION_TORSIONAL_SPRING_MODEL)
                dwRenderer_renderText(origin.x, origin.y - 160, "Suspension modeling: enabled", m_renderer);
            else
                dwRenderer_renderText(origin.x, origin.y - 160, "Suspension modeling: disabled", m_renderer);
        }

        // Real-time egomotion state
        {
            dwEgomotionResult state{};
            dwEgomotion_getEstimation(&state, m_egomotion);

            dwEgomotionUncertainty uncertainty{};
            dwEgomotion_getUncertainty(&uncertainty, m_egomotion);

            sprintf(sbuffer, "Real-time processing time: %.1f s", m_elapsedTime * 1e-6);
            dwRenderer_renderText(origin.x, origin.y - 195, sbuffer, m_renderer);

            static constexpr uint32_t VEL = DW_EGOMOTION_LIN_VEL_X;
            if ((state.validFlags & VEL) == VEL)
            {
                static float32_t oldSpeed = std::numeric_limits<float32_t>::max();
                static float32_t olddVdt = 0;
                static dwTime_t oldTimestamp = 0;

                float32_t speed = sqrt(state.linearVelocity[0] * state.linearVelocity[0] + 
                                     state.linearVelocity[1] * state.linearVelocity[1]);
                float32_t dVdt = 0;

                if (state.timestamp != oldTimestamp)
                {
                    dVdt = (speed - oldSpeed) / (static_cast<float32_t>(state.timestamp - oldTimestamp) / 1000000.f);
                    oldTimestamp = state.timestamp;
                    oldSpeed = speed;
                    olddVdt = dVdt;
                }

                sprintf(sbuffer, "Speed: %.2f m/s (%.2f km/h), rate: %.2f m/s^2", speed, speed * 3.6, olddVdt);
                dwRenderer_renderText(origin.x, origin.y - 220, sbuffer, m_renderer);

                // Display velocity components
                const auto printLinear = [this, &sbuffer](int32_t x, int32_t y, const char* name,
                                                          float32_t value, float32_t rate,
                                                          bool printLinear, bool printRate) {
                    int32_t len = 0;
                    len += sprintf(sbuffer, "%s: ", name);

                    if (printLinear)
                        len += sprintf(sbuffer + len, "%.2f m/s, ", value);

                    if (printRate)
                        len += sprintf(sbuffer + len, "rate: %.2f m/s^2", rate);

                    dwRenderer_renderText(x, y, sbuffer, m_renderer);
                };

                printLinear(origin.x, origin.y - 260, "V_x",
                           state.linearVelocity[0], state.linearAcceleration[0],
                           (state.validFlags & DW_EGOMOTION_LIN_VEL_X) != 0,
                           (state.validFlags & DW_EGOMOTION_LIN_ACC_X) != 0);

                printLinear(origin.x, origin.y - 280, "V_y",
                           state.linearVelocity[1], state.linearAcceleration[1],
                           (state.validFlags & DW_EGOMOTION_LIN_VEL_Y) != 0,
                           (state.validFlags & DW_EGOMOTION_LIN_ACC_Y) != 0);

                if (state.validFlags & DW_EGOMOTION_LIN_VEL_Z)
                {
                    printLinear(origin.x, origin.y - 300, "V_z",
                               state.linearVelocity[2], state.linearAcceleration[2],
                               (state.validFlags & DW_EGOMOTION_LIN_VEL_Z) != 0,
                               (state.validFlags & DW_EGOMOTION_LIN_ACC_Z) != 0);
                }
            }
            else
                dwRenderer_renderText(origin.x, origin.y - 220, "Speed: not supported", m_renderer);

            // Display orientation information
            const auto printAngle = [this, &sbuffer](int32_t x, int32_t y, const char* name,
                                                     float32_t value, float32_t std, float32_t rate,
                                                     bool printAngle, bool printStd, bool printRate) {
                int32_t len = 0;
                len += sprintf(sbuffer, "%s: ", name);

                if (printAngle)
                {
                    if (printStd)
                        len += sprintf(sbuffer + len, "%.2f +/- %.2f deg, ", RAD2DEG(value), RAD2DEG(std));
                    else
                        len += sprintf(sbuffer + len, "%.2f deg, ", RAD2DEG(value));
                }

                if (printRate)
                    len += sprintf(sbuffer + len, "rate: %.2f deg/s", RAD2DEG(rate));

                dwRenderer_renderText(x, y, sbuffer, m_renderer);
            };

            float32_t roll, pitch, yaw;
            quaternionToEulerAngles(state.rotation, roll, pitch, yaw);

            printAngle(origin.x, origin.y - 320, "Roll",
                      roll, std::sqrt(uncertainty.rotation.array[0]), state.angularVelocity[0],
                      (state.validFlags & DW_EGOMOTION_ROTATION) != 0,
                      (uncertainty.validFlags & DW_EGOMOTION_ROTATION) != 0,
                      (state.validFlags & DW_EGOMOTION_ANG_VEL_X) != 0);

            printAngle(origin.x, origin.y - 340, "Pitch",
                      pitch, std::sqrt(uncertainty.rotation.array[3 + 1]), state.angularVelocity[1],
                      (state.validFlags & DW_EGOMOTION_ROTATION) != 0,
                      (uncertainty.validFlags & DW_EGOMOTION_ROTATION) != 0,
                      (state.validFlags & DW_EGOMOTION_ANG_VEL_Y) != 0);

            printAngle(origin.x, origin.y - 360, "Yaw",
                      yaw, 0, state.angularVelocity[2],
                      (state.validFlags & DW_EGOMOTION_ROTATION) != 0,
                      false,
                      (state.validFlags & DW_EGOMOTION_ANG_VEL_Z) != 0);

            if ((state.validFlags & DW_EGOMOTION_ROTATION) != 0)
            {
                sprintf(sbuffer, "Rotation relative to starting pose (t=0)");
            }

            dwRenderer_renderText(origin.x, origin.y - 400, sbuffer, m_renderer);

            // GPS information
            dwGlobalEgomotionResult globalResult{};
            dwGlobalEgomotion_getEstimate(&globalResult, nullptr, m_globalEgomotion);

            if (globalResult.validPosition)
            {
                sprintf(sbuffer, "Longitude: %.5f deg", globalResult.position.lon);
                dwRenderer_renderText(origin.x, origin.y - 440, sbuffer, m_renderer);

                sprintf(sbuffer, "Latitude: %.5f deg", globalResult.position.lat);
                dwRenderer_renderText(origin.x, origin.y - 460, sbuffer, m_renderer);

                sprintf(sbuffer, "Altitude: %.2f m", globalResult.position.height);
                dwRenderer_renderText(origin.x, origin.y - 480, sbuffer, m_renderer);
            }
            else
            {
                sprintf(sbuffer, "GPS: not available");
                dwRenderer_renderText(origin.x, origin.y - 440, sbuffer, m_renderer);
            }
        }
        
        // Real-time CAN parser status
        if (m_canParser && m_canParser->isInitialized()) 
        {
            const auto& diagnostics = m_canParser->getDiagnostics();
            const auto& config = m_canParser->getConfiguration();
            
            sprintf(sbuffer, "Real-time CAN Parser Status:");
            dwRenderer_renderText(origin.x, origin.y - 520, sbuffer, m_renderer);
            
            sprintf(sbuffer, "State Valid: %s", m_canParser->hasValidState() ? "YES" : "NO");
            dwRenderer_renderText(origin.x, origin.y - 540, sbuffer, m_renderer);
            
            sprintf(sbuffer, "Messages: %u processed, %u rejected, %.1f Hz", 
                    diagnostics.validCANMessagesProcessed.load(), 
                    diagnostics.invalidCANMessagesRejected.load(),
                    diagnostics.averageMessageRate.load());
            dwRenderer_renderText(origin.x, origin.y - 560, sbuffer, m_renderer);
            
            sprintf(sbuffer, "State Commits: %u successful, %u failed, %.1f Hz",
                    diagnostics.stateCommitsSuccessful.load(),
                    diagnostics.stateCommitsFailed.load(),
                    diagnostics.averageCommitRate.load());
            dwRenderer_renderText(origin.x, origin.y - 580, sbuffer, m_renderer);
            
            if (diagnostics.speedMessageTimeout.load() || diagnostics.steeringMessageTimeout.load() || 
                diagnostics.wheelSpeedMessageTimeout.load()) {
                dwRenderer_renderText(origin.x, origin.y - 600, "⚠ MESSAGE TIMEOUTS DETECTED", m_renderer);
            }
            
            // Show current vehicle state
            if (m_canParser->hasValidState()) {
                const auto& safetyState = m_canParser->getSafetyState();
                const auto& nonSafetyState = m_canParser->getNonSafetyState();
                
                sprintf(sbuffer, "Vehicle: Speed=%.2f m/s, Steering=%.1f°", 
                        nonSafetyState.speedESC, 
                        safetyState.steeringWheelAngle * 180.0f / M_PI);
                dwRenderer_renderText(origin.x, origin.y - 620, sbuffer, m_renderer);
            }
        }
    }

    void renderPlots()
    {
        // Implementation similar to reference but optimized for real-time display
        if (!m_poseHistory.empty())
        {
            std::vector<dwVector2f> roll, rollUncertaintyPlus, rollUncertaintyMinus;
            std::vector<dwVector2f> pitch, pitchUncertaintyPlus, pitchUncertaintyMinus;
            std::vector<dwVector2f> altitude;

            float32_t negInf = -std::numeric_limits<float32_t>::infinity();
            float32_t posInf = std::numeric_limits<float32_t>::infinity();

            dwTime_t startTime = m_poseHistory.front().timestamp;
            dwTime_t lastTime = m_poseHistory.back().timestamp;

            for (const auto& pose : m_poseHistory)
            {
                float32_t dt = float32_t((pose.timestamp - startTime) * 1e-6);

                if (lastTime - pose.timestamp < 240 * 1e6)
                {
                    altitude.push_back({dt, float32_t(pose.rig2world.array[2 + 3 * 4])});
                }

                if (lastTime - pose.timestamp < 5 * 1e6)
                {
                    roll.push_back({dt, RAD2DEG(pose.rpy[0])});
                    pitch.push_back({dt, RAD2DEG(pose.rpy[1])});

                    if (pose.uncertainty.validFlags & DW_EGOMOTION_ROTATION)
                    {
                        rollUncertaintyPlus.push_back({dt, RAD2DEG(pose.rpy[0]) + RAD2DEG(std::sqrt(pose.uncertainty.rotation.array[0]))});
                        rollUncertaintyMinus.push_back({dt, RAD2DEG(pose.rpy[0]) - RAD2DEG(std::sqrt(pose.uncertainty.rotation.array[0]))});
                        pitchUncertaintyPlus.push_back({dt, RAD2DEG(pose.rpy[1]) + RAD2DEG(std::sqrt(pose.uncertainty.rotation.array[3 + 1]))});
                        pitchUncertaintyMinus.push_back({dt, RAD2DEG(pose.rpy[1]) - RAD2DEG(std::sqrt(pose.uncertainty.rotation.array[3 + 1]))});
                    }
                }
            }

            // Roll plot
            if (!roll.empty()) {
                dwRenderEnginePlotType types[] = {DW_RENDER_ENGINE_PLOT_TYPE_LINESTRIP, DW_RENDER_ENGINE_PLOT_TYPE_LINESTRIP, DW_RENDER_ENGINE_PLOT_TYPE_LINESTRIP};
                const void* data[] = {roll.data(), rollUncertaintyPlus.data(), rollUncertaintyMinus.data()};
                uint32_t strides[] = {sizeof(dwVector2f), sizeof(dwVector2f), sizeof(dwVector2f)};
                uint32_t offsets[] = {0, 0, 0};
                uint32_t counts[] = {uint32_t(roll.size()), uint32_t(rollUncertaintyPlus.size()), uint32_t(rollUncertaintyMinus.size())};
                dwRenderEngineColorRGBA colors[] = {{1.0f, 0.0f, 0.0f, 1.0f},
                                                    {1.0f, 0.0f, 0.0f, 0.5f},
                                                    {1.0f, 0.0f, 0.0f, 0.5f}};
                float32_t widths[] = {2.0f, 1.0f, 1.0f};
                const char* labels[] = {"roll", "", ""};

                dwRenderEngine_setTile(m_tileRollPlot, m_renderEngine);
                dwRenderEngine_setLineWidth(1.0f, m_renderEngine);
                dwRenderEngine_renderPlots2D(types,
                                             data, strides, offsets, counts,
                                             colors, widths, labels,
                                             counts[1] > 0 ? 3 : 1,
                                             {negInf, -10.f, posInf, 10.f},
                                             {0.0f, 0.0f, 1.0f, 1.0f},
                                             {0.5f, 0.4f, 0.2f, 1.0f},
                                             1.f,
                                             "", " time", "[deg]",
                                             m_renderEngine);
            }

            // Pitch plot  
            if (!pitch.empty()) {
                dwRenderEnginePlotType types[] = {DW_RENDER_ENGINE_PLOT_TYPE_LINESTRIP, DW_RENDER_ENGINE_PLOT_TYPE_LINESTRIP, DW_RENDER_ENGINE_PLOT_TYPE_LINESTRIP};
                const void* data[] = {pitch.data(), pitchUncertaintyPlus.data(), pitchUncertaintyMinus.data()};
                uint32_t strides[] = {sizeof(dwVector2f), sizeof(dwVector2f), sizeof(dwVector2f)};
                uint32_t offsets[] = {0, 0, 0};
                uint32_t counts[] = {uint32_t(pitch.size()), uint32_t(pitchUncertaintyPlus.size()), uint32_t(pitchUncertaintyMinus.size())};
                dwRenderEngineColorRGBA colors[] = {{0.0f, 1.0f, 0.0f, 1.0f},
                                                    {0.0f, 1.0f, 0.0f, 0.5f},
                                                    {0.0f, 1.0f, 0.0f, 0.5f}};
                float32_t widths[] = {2.0f, 1.0f, 1.0f};
                const char* labels[] = {"pitch", "", ""};

                dwRenderEngine_setTile(m_tilePitchPlot, m_renderEngine);
                dwRenderEngine_setLineWidth(1.0f, m_renderEngine);
                dwRenderEngine_renderPlots2D(types,
                                             data, strides, offsets, counts,
                                             colors, widths, labels,
                                             counts[1] > 0 ? 3 : 1,
                                             {negInf, -10.f, posInf, 10.f},
                                             {0.0f, 0.0f, 1.0f, 1.0f},
                                             {0.5f, 0.4f, 0.2f, 1.0f},
                                             1.f,
                                             "", " time", "[deg]",
                                             m_renderEngine);
            }

            // Altitude plot
            if (!altitude.empty()) {
                dwRenderEngine_setTile(m_tileAltitudePlot, m_renderEngine);
                dwRenderEngine_setLineWidth(1.0f, m_renderEngine);
                dwRenderEngine_renderPlot2D(DW_RENDER_ENGINE_PLOT_TYPE_LINESTRIP,
                                            altitude.data(),
                                            sizeof(dwVector2f),
                                            0,
                                            altitude.size(),
                                            "Altitude",
                                            {negInf, altitude.back().y - 5.f, posInf, altitude.back().y + 5.f},
                                            {0.0f, 0.0f, 1.0f, 1.0f},
                                            {0.5f, 0.4f, 0.2f, 1.0f},
                                            1.f,
                                            "", " time", "[m]",
                                            m_renderEngine);
            }
        }

        dwRenderEngine_setTile(m_tileGrid, m_renderEngine);
    }
    void onRender() override
    {
        // ========================================
        // DIAGNOSTIC: Render thread heartbeat - confirms render loop is alive
        // ========================================
        static uint32_t renderCount = 0;
        static dwTime_t lastRenderLog = 0;
        static dwTime_t renderStart = 0;
        
        dwTime_t now = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        
        if (renderStart == 0) {
            renderStart = now;
        }
        
        ++renderCount;
        
        if (now - lastRenderLog > 2000000) {  // Every 2 seconds
            float32_t actualFps = (renderCount * 1000000.0f) / (now - renderStart);
            char buf[256];
            sprintf(buf, "🎨 Render thread: %u frames, %.1f FPS\n", renderCount, actualFps);
            printColored(stdout, COLOR_GREEN, buf);
            lastRenderLog = now;
        }
        
        // ========================================
        // NEW: Process pending camera frames asynchronously
        // This runs in the render thread where GL context is active
        // ========================================
        {
            std::lock_guard<std::mutex> lock(m_cameraQueueMutex);
            
            // Process one camera frame per render cycle (if available and ready)
            if (!m_pendingCameraFrames.empty() && m_lastGLFrame == DW_NULL_HANDLE) {
                auto& pending = m_pendingCameraFrames.front();
                
                // GL streaming happens here (safe in render context, no time pressure)
                m_lastGLFrame = m_streamerInput2GL->post(pending.cudaImage);
                
                // Destroy the CUDA image after streaming
                dwImage_destroy(pending.cudaImage);
                
                m_pendingCameraFrames.pop_front();
                m_shallRender = true;
                
                // DIAGNOSTIC: Track camera frame processing
                static uint32_t processedFrames = 0;
                static dwTime_t lastCameraLog = 0;
                dwTime_t now = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                
                if (++processedFrames % 30 == 0 || now - lastCameraLog > 5000000) {  // Every 30 frames or 5s
                    char buf[256];
                    sprintf(buf, "📷 Camera frames: %u processed, %lu queued\n", 
                            processedFrames, m_pendingCameraFrames.size());
                    printColored(stdout, COLOR_YELLOW, buf);
                    lastCameraLog = now;
                }
            }
        }
        
        // ========================================
        // EXISTING: Render only if there's new data
        // ========================================
        if (!isPaused() && !m_shallRender)
            return;

        m_shallRender = true;

        if (isOffscreen())
            return;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        render3DGrid();

        // Render camera frame if available
        if (m_lastGLFrame != DW_NULL_HANDLE)
        {
            dwImageGL* glFrame = nullptr;
            dwImage_getGL(&glFrame, m_lastGLFrame);

            if (glFrame)
            {
                dwRenderEngine_setTile(m_tileVideo, m_renderEngine);
                dwVector2f range{float(glFrame->prop.width), float(glFrame->prop.height)};
                dwRenderEngine_setCoordinateRange2D(range, m_renderEngine);
                dwRenderEngine_renderImage2D(glFrame, {0.0f, 0.0f, range.x, range.y}, m_renderEngine);
            }

            // Return GL frame to the streamer so the pool doesn't fill up
            m_streamerInput2GL->release();
            m_lastGLFrame = DW_NULL_HANDLE;
        }

        renderText();
        renderPlots();

        renderutils::renderFPS(m_renderEngine, getCurrentFPS());
    }
    
    
    /// Real-time CAN message processing with synchronized state commits
    /// Temporal sensor fusion with synchronized egomotion processing
    void onProcess() override
{
    auto processStart = std::chrono::steady_clock::now();
    static uint32_t processCount = 0;
    static dwTime_t lastHeartbeat = 0;
    static dwTime_t heartbeatStart = 0;
    
    dwTime_t now = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    
    fprintf(stderr, " [%lu] ENTER onProcess() #%u\n", now, ++processCount);
    fflush(stderr);
    
    if (heartbeatStart == 0) {
        heartbeatStart = now;
    }
    
    if (now - lastHeartbeat > 1000000) {
        float32_t actualHz = (processCount * 1000000.0f) / (now - heartbeatStart);
        fprintf(stderr, " Main loop: %u cycles, %.1f Hz actual (target 100 Hz)\n", 
                processCount, actualHz);
        fflush(stderr);
        lastHeartbeat = now;
    }
    
    dwTime_t currentTime = now;
    
    // Phase 1: Drain sensor events
    fprintf(stderr, " [%u] Starting drainage...\n", processCount);
    fflush(stderr);
    
    auto drainStart = std::chrono::steady_clock::now();
    bool inCatchUp = drainSensorEventsToBuffers();
    auto drainTime = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - drainStart).count();
    
    fprintf(stderr, " [%u] Drainage complete: %.1fms, catchUp=%d\n", 
            processCount, drainTime / 1000.0f, inCatchUp);
    fflush(stderr);
    
    // Phase 2: Skip fusion during catch-up
    if (inCatchUp) {
        fprintf(stderr, " [%u] Skipping fusion (catch-up mode active)\n", processCount);
        fflush(stderr);
        
        auto totalTime = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - processStart).count();
        fprintf(stderr, " [%u] EXIT onProcess(): %.1fms total\n", 
                processCount, totalTime / 1000.0f);
        fflush(stderr);
        return;
    }
    
    // Phase 3: Normal fusion
    if (currentTime - m_lastFusionTimestamp >= FUSION_RATE_US) {
        fprintf(stderr, " [%u] Attempting fusion (lastFusion=%lu, current=%lu, diff=%lu)\n",
                processCount, m_lastFusionTimestamp, currentTime, 
                currentTime - m_lastFusionTimestamp);
        fflush(stderr);
        
        auto fusionStart = std::chrono::steady_clock::now();
        bool fusionSuccess = attemptTemporalFusion(currentTime);
        auto fusionTime = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - fusionStart).count();
        
        fprintf(stderr, " [%u] Fusion %s: %.1fms\n", 
                processCount, fusionSuccess ? "SUCCESS" : "FAILED", fusionTime / 1000.0f);
        fflush(stderr);
        
        if (fusionSuccess) {
            m_lastFusionTimestamp = currentTime;
            
            fprintf(stderr, " [%u] Starting pose estimation...\n", processCount);
            fflush(stderr);
            
            auto poseStart = std::chrono::steady_clock::now();
            performPoseEstimationAndTrajectory();
            auto poseTime = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - poseStart).count();
            
            fprintf(stderr, " [%u] Pose estimation complete: %.1fms\n", 
                    processCount, poseTime / 1000.0f);
            fflush(stderr);
            
            if (poseTime > 5000) {
                fprintf(stderr, "[%u] WARNING: Pose estimation took %.1fms (>5ms budget!)\n",
                        processCount, poseTime / 1000.0f);
                fflush(stderr);
            }
        }
    } else {
        fprintf(stderr, " [%u] Skipping fusion (too soon: %lu < %lu threshold)\n",
                processCount, currentTime - m_lastFusionTimestamp, FUSION_RATE_US);
        fflush(stderr);
    }
    
    auto totalTime = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - processStart).count();
    
    fprintf(stderr, " [%u] EXIT onProcess(): drain=%.1fms, total=%.1fms\n", 
            processCount, drainTime / 1000.0f, totalTime / 1000.0f);
    fflush(stderr);
    
    if (totalTime > 10000) {
        fprintf(stderr, "[%u] WARNING: onProcess exceeded 10ms budget!\n", processCount);
        fflush(stderr);
    }
}

    private:
    /**
     * Pose estimation and trajectory building (extracted from sample_egomotion onProcess)
     */
    void performPoseEstimationAndTrajectory() {
    static uint32_t poseCount = 0;
    ++poseCount;
    
    fprintf(stderr, "   [pose #%u] Getting estimation...\n", poseCount);
    fflush(stderr);
    
    auto t1 = std::chrono::steady_clock::now();
    dwEgomotionResult estimate;
    dwEgomotionUncertainty uncertainty;
    
    dwStatus estStatus = dwEgomotion_getEstimation(&estimate, m_egomotion);
    auto dt1 = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t1).count();
    fprintf(stderr, "   [pose #%u] dwEgomotion_getEstimation: %.1fms, status=%d\n", 
            poseCount, dt1 / 1000.0f, estStatus);
    fflush(stderr);
    
    if (estStatus != DW_SUCCESS) {
        fprintf(stderr, "   [pose #%u] Estimation failed, exiting\n", poseCount);
        fflush(stderr);
        return;
    }
    
    auto t2 = std::chrono::steady_clock::now();
    dwStatus uncStatus = dwEgomotion_getUncertainty(&uncertainty, m_egomotion);
    auto dt2 = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t2).count();
    fprintf(stderr, "   [pose #%u] dwEgomotion_getUncertainty: %.1fms, status=%d\n", 
            poseCount, dt2 / 1000.0f, uncStatus);
    fflush(stderr);
    
    if (uncStatus != DW_SUCCESS) {
        fprintf(stderr, "   [pose #%u] Uncertainty failed, exiting\n", poseCount);
        fflush(stderr);
        return;
    }
    
    fprintf(stderr, "   [pose #%u] Adding relative motion to global egomotion...\n", poseCount);
    fflush(stderr);
    
    auto t3 = std::chrono::steady_clock::now();
    dwGlobalEgomotion_addRelativeMotion(&estimate, &uncertainty, m_globalEgomotion);
    auto dt3 = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t3).count();
    fprintf(stderr, "   [pose #%u] dwGlobalEgomotion_addRelativeMotion: %.1fms\n", 
            poseCount, dt3 / 1000.0f);
    fflush(stderr);
    
    if (estimate.timestamp >= m_lastSampleTimestamp + POSE_SAMPLE_PERIOD)
    {
        fprintf(stderr, "   [pose #%u] Computing relative transformation...\n", poseCount);
        fflush(stderr);
        
        auto t4 = std::chrono::steady_clock::now();
        dwTransformation3f rigLast2rigNow;
        dwStatus transStatus = dwEgomotion_computeRelativeTransformation(
            &rigLast2rigNow, nullptr, m_lastSampleTimestamp, estimate.timestamp, m_egomotion);
        auto dt4 = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t4).count();
        fprintf(stderr, "   [pose #%u] dwEgomotion_computeRelativeTransformation: %.1fms, status=%d\n", 
                poseCount, dt4 / 1000.0f, transStatus);
        fflush(stderr);
        
        if (transStatus == DW_SUCCESS)
        {
            fprintf(stderr, "   [pose #%u] Building pose history entry...\n", poseCount);
            fflush(stderr);
            
            Pose pose{};
            quaternionToEulerAngles(estimate.rotation, pose.rpy[0], pose.rpy[1], pose.rpy[2]);

            dwTransformation3f rigLast2world = DW_IDENTITY_TRANSFORMATION3F;
            if (!m_poseHistory.empty())
                rigLast2world = m_poseHistory.back().rig2world;
            else if ((estimate.validFlags & DW_EGOMOTION_ROTATION) != 0)
            {
                dwMatrix3f rot{};
                getRotationMatrix(&rot, RAD2DEG(pose.rpy[0]), RAD2DEG(pose.rpy[1]), 0);
                rotationToTransformMatrix(rigLast2world.array, rot.array);
            }

            dwTransformation3f rigNow2World;
            dwEgomotion_applyRelativeTransformation(&rigNow2World, &rigLast2rigNow, &rigLast2world);

            pose.rig2world = rigNow2World;
            pose.timestamp = estimate.timestamp;

            if (m_poseHistory.size() > MAX_BUFFER_POINTS)
            {
                fprintf(stderr, "   [pose #%u] Trimming pose history (size=%lu)\n", 
                        poseCount, m_poseHistory.size());
                fflush(stderr);
                
                decltype(m_poseHistory) tmp;
                tmp.assign(++m_poseHistory.begin(), m_poseHistory.end());
                std::swap(tmp, m_poseHistory);
            }

            if (m_outputFile)
                fprintf(m_outputFile, "%lu,%.2f,%.2f,%.2f\n", estimate.timestamp,
                        rigNow2World.array[0 + 3 * 4], rigNow2World.array[1 + 3 * 4], rigNow2World.array[2 + 3 * 4]);

            fprintf(stderr, "   [pose #%u] Getting global estimate...\n", poseCount);
            fflush(stderr);
            
            auto t5 = std::chrono::steady_clock::now();
            dwGlobalEgomotionResult absoluteEstimate{};
            dwStatus globalStatus = dwGlobalEgomotion_getEstimate(&absoluteEstimate, nullptr, m_globalEgomotion);
            auto dt5 = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t5).count();
            fprintf(stderr, "   [pose #%u] dwGlobalEgomotion_getEstimate: %.1fms, status=%d\n", 
                    poseCount, dt5 / 1000.0f, globalStatus);
            fflush(stderr);
            
            if (globalStatus == DW_SUCCESS &&
                absoluteEstimate.timestamp == estimate.timestamp && absoluteEstimate.validOrientation)
            {
                m_orientationENU = absoluteEstimate.orientation;
                m_hasOrientationENU = true;

                if (m_trajectoryLog.size("Egomotion") == 0)
                    m_trajectoryLog.addWGS84("Egomotion", m_currentGPSFrame);

                m_trajectoryLog.addWGS84("Egomotion", absoluteEstimate.position);
            }
            else
            {
                m_hasOrientationENU = false;
            }

            dwEgomotion_getUncertainty(&pose.uncertainty, m_egomotion);
            
            fprintf(stderr, "   [pose #%u] Appending to pose history (new size=%lu)\n", 
                    poseCount, m_poseHistory.size() + 1);
            fflush(stderr);
            
            m_poseHistory.push_back(pose);
            m_shallRender = true;
        } else {
            fprintf(stderr, "   [pose #%u] Relative transformation failed\n", poseCount);
            fflush(stderr);
        }
        
        m_lastSampleTimestamp = estimate.timestamp;
    } else {
        fprintf(stderr, "   [pose #%u] Skipping pose (timestamp %lu < threshold %lu + %ld)\n",
                poseCount, estimate.timestamp, m_lastSampleTimestamp, POSE_SAMPLE_PERIOD);
        fflush(stderr);
    }
    
    fprintf(stderr, "   [pose #%u] Complete\n", poseCount);
    fflush(stderr);
}
    void onKeyDown(int key, int scancode, int mods) override
    {
        (void)scancode;
        (void)mods;

        if (key == GLFW_KEY_F1)
        {
            getMouseView().setCenter(0, 0, 0);
            m_renderingMode = RenderingMode::STICK_TO_VEHICLE;
        }

        if (key == GLFW_KEY_F2)
        {
            m_renderingMode = RenderingMode::ON_VEHICLE_STICK_TO_WORLD;
        }
    }
};

//------------------------------------------------------------------------------
int main(int argc, const char* argv[])
{
    const std::string samplePath = dw_samples::SamplesDataPath::get() + "/samples/recordings/cloverleaf/";

    ProgramArguments args(argc, argv,
                          {
                              ProgramArguments::Option_t("vehicle-sensor-name", "", "[optional] Name of the vehicle sensor in rig file for real-time processing."),
                              ProgramArguments::Option_t("imu-sensor-name", "", "[optional] Name of the IMU sensor in rig file for real-time processing."),
                              ProgramArguments::Option_t("gps-sensor-name", "", "[optional] Name of the GPS sensor in rig file for real-time processing."),
                              ProgramArguments::Option_t("camera-sensor-name", "", "[optional] Name of the camera sensor in rig file for real-time processing."),

                              ProgramArguments::Option_t("rig", (samplePath + "rig-nominal-intrinsics.json").c_str(),
                                                         "Rig file containing sensor and vehicle configuration for real-time processing."),

                              ProgramArguments::Option_t("output", "", "If specified, real-time trajectory will be output to this file."),
                              ProgramArguments::Option_t("outputkml", "", "If specified, real-time GPS and estimated trajectories will be output to this KML file"),
                              ProgramArguments::Option_t("mode", "1", "0=Ackerman motion, 1=IMU+Odometry+GPS for real-time processing"),
                              ProgramArguments::Option_t("speed-measurement-type", "1", "Speed measurement type for real-time processing, refer to dwEgomotionSpeedMeasurementType"),
                              ProgramArguments::Option_t("enable-suspension", "0", "If 1, enables egomotion suspension modeling for real-time processing (requires Odometry+IMU [--mode=1]), otherwise disabled."),
                          },
                          "DriveWorks real-time egomotion sample with synchronized CAN processing");

    EgomotionSample app(args);

    app.initializeWindow("Real-time Egomotion Sample", 1920, 1080, args.enabled("offscreen"));

    // ========================================
    // MODIFIED: Reduced process rate to match sensor bandwidth
    // Previous: 240 Hz (4.17ms period) - caused timing conflicts
    // Current:  100 Hz (10ms period) - matches IMU rate, safe drainage budget
    // ========================================
    if (!args.enabled("offscreen"))
        app.setProcessRate(100);  // Changed from 240
    
    log("Starting real-time egomotion processing at 100 Hz...\n");
    log("Sensor drainage budget: 5ms per cycle\n");
    log("Expected sensor rates: CAN=50Hz, IMU=100Hz, GPS=10Hz, Camera=30Hz\n");
    
    return app.run();
                        }