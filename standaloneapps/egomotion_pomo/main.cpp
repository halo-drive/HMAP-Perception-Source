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
    
    
    dwTime_t m_lastIMUTimestampFed = 0;
    static constexpr dwTime_t MIN_IMU_GAP_US = 5000;
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
        std::mutex bufferMutex;
        
        //temporal tracking
        dwTime_t lastProcessedTimestamp = 0;
        dwTime_t lastGPSTimestamp       = 0;
        dwTime_t lastIMUTimestamp       = 0;
        dwTime_t lastProcessedGPS       = 0;  
        dwTime_t lastProcessedIMU       = 0;  

        //Buffer management
        static constexpr dwTime_t BUFFER_RENTENTION_US = 1000000; //1 second

        void cleanupOldData(dwTime_t currentTimestamp) {
           dwTime_t cutoff = currentTimestamp - BUFFER_RENTENTION_US;
           
           gpsBuffer.erase(gpsBuffer.begin(), gpsBuffer.lower_bound(cutoff));
           imuBuffer.erase(imuBuffer.begin(), imuBuffer.lower_bound(cutoff));
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
    bool attemptTemporalFusion(dwTime_t currentTime) 
    {
        std::lock_guard<std::mutex> lock(m_temporalBuffers.bufferMutex);
        
        // =========================================================
        // DIAGNOSTIC: Log buffer states
        // =========================================================
        static dwTime_t lastDiagLog = 0;
        if (currentTime - lastDiagLog > 2000000) {  // Every 2 seconds
            fprintf(stderr, " [FUSION DIAG] GPS buffer: %lu entries, last=%lu, lastProcessed=%lu\n",
                    m_temporalBuffers.gpsBuffer.size(),
                    m_temporalBuffers.lastGPSTimestamp,
                    m_temporalBuffers.lastProcessedGPS);
            lastDiagLog = currentTime;
        }
        
        // =========================================================
        // 1. Find GPS anchor
        // =========================================================
        if (m_temporalBuffers.gpsBuffer.empty()) {
            static bool warnedNoGPS = false;
            if (!warnedNoGPS) {
                fprintf(stderr, " [FUSION] WARNING: No GPS data in buffer!\n");
                warnedNoGPS = true;
            }
            return false;
        }
        
        auto gpsIt = m_temporalBuffers.gpsBuffer.upper_bound(m_temporalBuffers.lastProcessedGPS);
        if (gpsIt == m_temporalBuffers.gpsBuffer.end()) {
            // No new GPS - this is normal, GPS is only ~10Hz
            return false;
        }
        
        dwTime_t anchorTimestamp = gpsIt->first;
        dwGPSFrame& gpsFrame = gpsIt->second;
        
        fprintf(stderr, " [FUSION] New GPS: lat=%.6f, lon=%.6f, timestamp=%lu\n",
                gpsFrame.latitude, gpsFrame.longitude, anchorTimestamp);
        
        // =========================================================
        // 2. Feed GPS to GlobalEgomotion
        // =========================================================
        m_currentGPSFrame = gpsFrame;
        m_trajectoryLog.addWGS84("GPS", gpsFrame);

        // Detailed GPS logging comparable to sensors/egomotion/main.cpp
        log("\n=== GPS FRAME (POMO) ===\n");
        log("  Timestamp: %lu us (%.6f s)\n", gpsFrame.timestamp_us, gpsFrame.timestamp_us * 1e-6);
        log("  Position:\n");
        log("    Latitude: %.8f deg\n", gpsFrame.latitude);
        log("    Longitude: %.8f deg\n", gpsFrame.longitude);
        log("    Altitude: %.3f m\n", gpsFrame.altitude);
        log("  Velocity:\n");
        log("    Horizontal speed: %.6f m/s\n", gpsFrame.speed);
        log("    Vertical speed (climb): %.6f m/s\n", gpsFrame.climb);
        log("  Course: %.2f deg\n", gpsFrame.course);
        log("  Accuracy:\n");
        log("    hacc: %.3f m\n", gpsFrame.hacc);
        log("    vacc: %.3f m\n", gpsFrame.vacc);
        log("  Dilution of Precision:\n");
        log("    HDOP: %.3f\n", gpsFrame.hdop);
        log("    VDOP: %.3f\n", gpsFrame.vdop);
        log("    PDOP: %.3f\n", gpsFrame.pdop);
        log("  Satellite Count: %u\n", gpsFrame.satelliteCount);
        log("  Fix Status: %d\n", gpsFrame.fixStatus);
        log("  Mode: %d\n", gpsFrame.mode);
        log("  UTC Time: %lu us\n", gpsFrame.utcTimeUs);

        dwStatus gpsStatus = dwGlobalEgomotion_addGPSMeasurement(&gpsFrame, m_globalEgomotion);
        log("  dwGlobalEgomotion_addGPSMeasurement status: %s\n", dwGetStatusName(gpsStatus));
        
        // Mark GPS as processed
        m_temporalBuffers.lastProcessedGPS = anchorTimestamp;
        m_temporalBuffers.gpsBuffer.erase(m_temporalBuffers.gpsBuffer.begin(), ++gpsIt);
        
        // =========================================================
        // 3. Check if egomotion has a valid estimate
        // =========================================================
        dwEgomotionResult estimate{};
        dwStatus estStatus = dwEgomotion_getEstimation(&estimate, m_egomotion);
        
        fprintf(stderr, " [FUSION] Egomotion status: %s, validFlags=0x%08X, timestamp=%lu\n",
                dwGetStatusName(estStatus), estimate.validFlags, estimate.timestamp);
        
        if (estStatus != DW_SUCCESS) {
            fprintf(stderr, " [FUSION] Egomotion not ready - needs vehicle motion to converge\n");
            return false;
        }
        
        // Check what's actually valid
        if (estimate.validFlags & DW_EGOMOTION_LIN_VEL_X) {
            fprintf(stderr, " [FUSION] ✓ Velocity valid: Vx=%.2f m/s\n", estimate.linearVelocity[0]);
        }
        if (estimate.validFlags & DW_EGOMOTION_ROTATION) {
            fprintf(stderr, " [FUSION] ✓ Rotation valid\n");
        }
        
        m_temporalBuffers.lastProcessedTimestamp = anchorTimestamp;
        m_elapsedTime = anchorTimestamp - m_firstTimestamp;
        
        fprintf(stderr, " [FUSION] SUCCESS! elapsedTime=%.1fs\n", m_elapsedTime * 1e-6);
        
        return true;
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
        auto it = m_temporalBuffers.gpsBuffer.rbegin();  // Reverse iterator
        while (it != m_temporalBuffers.gpsBuffer.rend()) {
            // Check if this GPS is new (not already processed)
            if (it->first > m_temporalBuffers.lastProcessedGPS &&
                it->first >= earliestAcceptable) {
                // Convert reverse iterator to forward iterator
                return std::prev(it.base());
            }
            ++it;
        }
        
        return m_temporalBuffers.gpsBuffer.end();
    }
    
    /**
     * Generic template to find closest sensor data within temporal window
     */
    template<typename SensorMap>
    typename SensorMap::iterator findClosestSensorData(
        SensorMap& sensorMap, 
        dwTime_t targetTime, 
        dwTime_t maxWindow,
        dwTime_t excludeBeforeTimestamp = 0) {  
        
        if (sensorMap.empty()) {
            return sensorMap.end();
        }
        
        auto targetIt = sensorMap.lower_bound(targetTime);
        typename SensorMap::iterator bestMatch = sensorMap.end();
        dwTime_t bestDistance = maxWindow + 1;
        
        // Check forward direction
        if (targetIt != sensorMap.end() && 
            targetIt->first > excludeBeforeTimestamp) {  
            dwTime_t forwardDist = std::abs(static_cast<int64_t>(targetIt->first - targetTime));
            if (forwardDist <= maxWindow && forwardDist < bestDistance) {
                bestMatch = targetIt;
                bestDistance = forwardDist;
            }
        }
        
        // Check backward direction
        if (targetIt != sensorMap.begin()) {
            auto backIt = std::prev(targetIt);
            if (backIt->first > excludeBeforeTimestamp) {  
                dwTime_t backwardDist = std::abs(static_cast<int64_t>(backIt->first - targetTime));
                if (backwardDist <= maxWindow && backwardDist < bestDistance) {
                    bestMatch = backIt;
                    bestDistance = backwardDist;
                }
            }
        }
        
        return bestMatch;
    }
    
    bool drainSensorEventsToBuffers() 
    {
        static uint32_t drainageCallCount = 0;
        ++drainageCallCount;
        
        dwTime_t now = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        
        static uint32_t canCount = 0, imuCount = 0, gpsCount = 0, cameraCount = 0;
        static dwTime_t lastReport = 0;
        
        if (now - lastReport > 1000000) {
            char buf[512];
            sprintf(buf, " Sensor events (last 1s): CAN=%u, IMU=%u, GPS=%u, Camera=%u\n",
                    canCount, imuCount, gpsCount, cameraCount);
            printColored(stdout, COLOR_YELLOW, buf);
            canCount = imuCount = gpsCount = cameraCount = 0;
            lastReport = now;
        }
        
        int totalDrained = 0;
        auto budgetEnd = std::chrono::steady_clock::now() + std::chrono::microseconds(5000);
        
        while (!isPaused() && totalDrained < 256) {
            if (std::chrono::steady_clock::now() >= budgetEnd) {
                break;
            }
            
            dwStatus status = dwSensorManager_acquireNextEvent(&acquiredEvent, 0, m_sensorManager);
            
            if (status == DW_TIME_OUT) {
                break;
            }
            
            if (status != DW_SUCCESS) {
                if (status == DW_END_OF_STREAM && should_AutoExit()) {
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
                case DW_SENSOR_CAN:
                case DW_SENSOR_DATA:
                {
                    if (acquiredEvent->sensorIndex != m_vehicleSensorIdx)
                        break;
                    
                    // VERIFICATION: Compare CAN timestamp with context time to verify PTP sync
                    static int canFrameCount = 0;
                    if (canFrameCount++ < 5) {  // Log first 5 frames
                        dwTime_t contextTime = 0;
                        dwContext_getCurrentTime(&contextTime, m_context);
                        int64_t timeDiff = static_cast<int64_t>(acquiredEvent->canFrame.timestamp_us) - contextTime;
                        printf("[PTP VERIFY] CAN frame #%d: CAN_ts=%lld, Context_ts=%lld, diff=%lld us\n",
                               canFrameCount, 
                               static_cast<long long>(acquiredEvent->canFrame.timestamp_us),
                               static_cast<long long>(contextTime),
                               static_cast<long long>(timeDiff));
                        fflush(stdout);
                    }
                    
                    m_canParser->processCANFrame(acquiredEvent->canFrame);
                    
                    dwVehicleIOSafetyState safetyState{};
                    dwVehicleIONonSafetyState nonSafetyState{};
                    dwVehicleIOActuationFeedback actuationFeedback{};
                    
                    m_canParser->getCurrentState(&safetyState, &nonSafetyState, &actuationFeedback);
                    
                    // ------------------------------------------------------------------
                    // FEED VEHICLE STATE INTO EGOMOTION (detailed logging, comparable
                    // to sensors/egomotion/main.cpp for side‑by‑side debugging)
                    // ------------------------------------------------------------------
                    {
                        const dwCANMessage& frame = acquiredEvent->canFrame;
                        log("\n=== VEHICLE DATA (POMO CAN) ===\n");
                        log("  CAN Frame:\n");
                        log("    ID: 0x%X\n", frame.id);
                        log("    DLC: %u\n", frame.size);
                        log("    Timestamp: %lu us (%.6f s)\n",
                            frame.timestamp_us,
                            frame.timestamp_us * 1e-6);
                        log("    Data: ");
                        for (uint32_t i = 0; i < frame.size && i < 8; i++)
                        {
                            log("%02X ", frame.data[i]);
                        }
                        log("\n");

                        log("  SafetyState:\n");
                        log("    size: %u\n", safetyState.size);
                        log("    steeringWheelAngle: %.6f rad (%.2f deg)\n",
                            safetyState.steeringWheelAngle,
                            RAD2DEG(safetyState.steeringWheelAngle));
                        log("    timestamp_us: %lu\n", safetyState.timestamp_us);
                        log("    sequenceId: %u\n", safetyState.sequenceId);

                        log("  NonSafetyState:\n");
                        log("    size: %u\n", nonSafetyState.size);
                        log("    speedESC: %.6f m/s (%.2f km/h)\n",
                            nonSafetyState.speedESC,
                            nonSafetyState.speedESC * 3.6f);
                        log("    speedDirectionESC: %d\n", nonSafetyState.speedDirectionESC);
                        log("    speedESCTimestamp: %lu\n", nonSafetyState.speedESCTimestamp);
                        log("    frontSteeringAngle: %.6f rad (%.2f deg)\n",
                            nonSafetyState.frontSteeringAngle,
                            RAD2DEG(nonSafetyState.frontSteeringAngle));
                        log("    frontSteeringTimestamp: %lu\n", nonSafetyState.frontSteeringTimestamp);
                        log("    wheelSpeed (angular) [rad/s]:\n");
                        log("      FL: %.6f\n", nonSafetyState.wheelSpeed[0]);
                        log("      FR: %.6f\n", nonSafetyState.wheelSpeed[1]);
                        log("      RL: %.6f\n", nonSafetyState.wheelSpeed[2]);
                        log("      RR: %.6f\n", nonSafetyState.wheelSpeed[3]);
                        log("    wheelTicksTimestamp:\n");
                        log("      FL: %lu\n", nonSafetyState.wheelTicksTimestamp[0]);
                        log("      FR: %lu\n", nonSafetyState.wheelTicksTimestamp[1]);
                        log("      RL: %lu\n", nonSafetyState.wheelTicksTimestamp[2]);
                        log("      RR: %lu\n", nonSafetyState.wheelTicksTimestamp[3]);
                        log("    drivePositionStatus: %d\n", nonSafetyState.drivePositionStatus);
                        log("    vehicleStopped: %d\n", nonSafetyState.vehicleStopped);

                        log("  ActuationFeedback:\n");
                        log("    size: %u\n", actuationFeedback.size);
                        log("    speedESC: %.6f m/s (%.2f km/h)\n",
                            actuationFeedback.speedESC,
                            actuationFeedback.speedESC * 3.6f);
                        log("    speedDirectionESC: %d\n", actuationFeedback.speedDirectionESC);
                        log("    steeringWheelAngle: %.6f rad (%.2f deg)\n",
                            actuationFeedback.steeringWheelAngle,
                            RAD2DEG(actuationFeedback.steeringWheelAngle));
                        log("    frontSteeringAngle: %.6f rad (%.2f deg)\n",
                            actuationFeedback.frontSteeringAngle,
                            RAD2DEG(actuationFeedback.frontSteeringAngle));
                        log("    timestamp_us: %lu\n", actuationFeedback.timestamp_us);
                        log("    sequenceId: %u\n", actuationFeedback.sequenceId);
                    }

                    // CRITICAL: Check return status
                    dwStatus vioStatus = dwEgomotion_addVehicleIOState(&safetyState, &nonSafetyState, &actuationFeedback, m_egomotion);
                    log("  dwEgomotion_addVehicleIOState status: %s\n", dwGetStatusName(vioStatus));
                    
                    // Log EVERY failure, but successes only once per second
                    static dwTime_t lastSuccessLog = 0;
                    static uint32_t failCount = 0;
                    static uint32_t successCount = 0;
                    
                    if (vioStatus != DW_SUCCESS) {
                        if (failCount++ < 10 || failCount % 100 == 0) {
                            fprintf(stderr, " VIO FEED FAILED: %s (ts=%lu, speed=%.2f, wheels=[%.1f,%.1f,%.1f,%.1f])\n",
                                    dwGetStatusName(vioStatus),
                                    nonSafetyState.timestamp_us,
                                    nonSafetyState.speedESC,
                                    nonSafetyState.wheelSpeed[0], nonSafetyState.wheelSpeed[1],
                                    nonSafetyState.wheelSpeed[2], nonSafetyState.wheelSpeed[3]);
                        }
                    } else {
                        successCount++;
                        if (acquiredEvent->timestamp_us - lastSuccessLog > 1000000) {
                            fprintf(stderr, " VIO FEED OK: %u successes, %u failures, ts=%lu\n",
                                    successCount, failCount, nonSafetyState.timestamp_us);
                            lastSuccessLog = acquiredEvent->timestamp_us;
                        }
                    }
                    
                    canCount++;
                    break;
                }
                
                case DW_SENSOR_IMU:
                {
                    if (acquiredEvent->sensorIndex != m_imuSensorIdx)
                        break;
                    
                    const dwIMUFrame& imu = acquiredEvent->imuFrame;
                    
                    // Detailed IMU logging comparable to sensors/egomotion/main.cpp
                    static uint32_t imuFedCount = 0;
                    ++imuFedCount;
                    {
                        float accelMag = std::sqrt(imu.acceleration[0]*imu.acceleration[0] +
                                                   imu.acceleration[1]*imu.acceleration[1] +
                                                   imu.acceleration[2]*imu.acceleration[2]);
                        float gyroMag  = std::sqrt(imu.turnrate[0]*imu.turnrate[0] +
                                                   imu.turnrate[1]*imu.turnrate[1] +
                                                   imu.turnrate[2]*imu.turnrate[2]);
                        log("\n=== IMU FRAME (POMO) ===\n");
                        log("  Frame #%u\n", imuFedCount);
                        log("  Timestamp: %lu us (%.6f s)\n", imu.timestamp_us, imu.timestamp_us * 1e-6);
                        log("  Acceleration [m/s^2]:\n");
                        log("    x: %.6f\n", imu.acceleration[0]);
                        log("    y: %.6f\n", imu.acceleration[1]);
                        log("    z: %.6f\n", imu.acceleration[2]);
                        log("    |a|: %.6f\n", accelMag);
                        log("  Turnrate [rad/s]:\n");
                        log("    x: %.6f (%.2f deg/s)\n", imu.turnrate[0], RAD2DEG(imu.turnrate[0]));
                        log("    y: %.6f (%.2f deg/s)\n", imu.turnrate[1], RAD2DEG(imu.turnrate[1]));
                        log("    z: %.6f (%.2f deg/s)\n", imu.turnrate[2], RAD2DEG(imu.turnrate[2]));
                        log("    |w|: %.6f (%.2f deg/s)\n", gyroMag, RAD2DEG(gyroMag));
                        log("  Magnetometer [utesla]:\n");
                        log("    x: %.6f\n", imu.magnetometer[0]);
                        log("    y: %.6f\n", imu.magnetometer[1]);
                        log("    z: %.6f\n", imu.magnetometer[2]);
                        log("  Orientation (RPY) [deg]:\n");
                        log("    roll: %.2f\n", imu.orientation[0]);
                        log("    pitch: %.2f\n", imu.orientation[1]);
                        log("    yaw: %.2f\n", imu.orientation[2]);
                        log("  Heading: %.2f deg\n", imu.heading);
                        log("============================\n");
                    }

                    if (m_egomotionParameters.motionModel != DW_EGOMOTION_ODOMETRY) {
                        dwStatus imuStatus = dwEgomotion_addIMUMeasurement(&imu, m_egomotion);
                        log("  dwEgomotion_addIMUMeasurement status: %s\n", dwGetStatusName(imuStatus));
                    }
                    
                    m_currentIMUFrame = imu;
                    m_lastIMUTimestampFed = imu.timestamp_us;  // FIX: Update the timestamp
                    imuCount++;
                    break;
                }

                
                case DW_SENSOR_GPS:
                {
                    if (acquiredEvent->sensorIndex != m_gpsSensorIdx)
                        break;
                    
                    // Buffer GPS for fusion anchoring
                    {
                        std::lock_guard<std::mutex> lock(m_temporalBuffers.bufferMutex);
                        if (timestamp > m_temporalBuffers.lastGPSTimestamp) {
                            m_temporalBuffers.gpsBuffer[timestamp] = acquiredEvent->gpsFrame;
                            m_temporalBuffers.lastGPSTimestamp = timestamp;
                        }
                    }
                    
                    gpsCount++;
                    break;
                }
                
                case DW_SENSOR_CAMERA:
                    cameraCount++;
                    break;
                    
                default:
                    break;
            }
            
            dwSensorManager_releaseAcquiredEvent(acquiredEvent, m_sensorManager);
            acquiredEvent = nullptr;
            ++totalDrained;
        }
        
        // Cleanup old buffered data
        {
            std::lock_guard<std::mutex> lock(m_temporalBuffers.bufferMutex);
            dwTime_t cutoff = now - m_temporalBuffers.BUFFER_RENTENTION_US;
            
            // Erase entries older than cutoff
            m_temporalBuffers.gpsBuffer.erase(
                m_temporalBuffers.gpsBuffer.begin(), 
                m_temporalBuffers.gpsBuffer.lower_bound(cutoff));
            m_temporalBuffers.imuBuffer.erase(
                m_temporalBuffers.imuBuffer.begin(), 
                m_temporalBuffers.imuBuffer.lower_bound(cutoff));
        }
        
        return (totalDrained >= 128);  // Return true if in catch-up mode
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
            
            // Check PTP synchronization status
            {
                printf("[PTP DEBUG] About to check PTP synchronization status...\n");
                fflush(stdout);
                
                bool isPTPSynchronized = false;
                dwStatus ptpStatus = dwContext_isTimePTPSynchronized(&isPTPSynchronized, m_context);
                
                printf("[PTP DEBUG] dwContext_isTimePTPSynchronized returned status: %s (code: %d)\n", 
                       dwGetStatusName(ptpStatus), ptpStatus);
                printf("[PTP DEBUG] isPTPSynchronized flag value: %s\n", isPTPSynchronized ? "true" : "false");
                fflush(stdout);
                
                char buf[512];
                if (ptpStatus == DW_SUCCESS) {
                    if (isPTPSynchronized) {
                        sprintf(buf, "[PTP SYNC] Context is PTP-synchronized - All sensors will use PTP timestamps\n");
                        printColored(stdout, COLOR_GREEN, buf);
                    } else {
                        sprintf(buf, "[PTP SYNC] Context is NOT PTP-synchronized - Sensors will use system clock\n");
                        printColored(stdout, COLOR_YELLOW, buf);
                        sprintf(buf, "[PTP SYNC] Note: PTP sync requires PTP daemon to be running during context creation\n");
                        printColored(stdout, COLOR_YELLOW, buf);
                    }
                } else if (ptpStatus == DW_NOT_SUPPORTED) {
                    sprintf(buf, "[PTP SYNC] PTP synchronization not supported on this platform\n");
                    printColored(stdout, COLOR_YELLOW, buf);
                } else {
                    sprintf(buf, "[PTP SYNC] Failed to check PTP sync status: %s (code: %d)\n", 
                            dwGetStatusName(ptpStatus), ptpStatus);
                    printColored(stdout, COLOR_YELLOW, buf);
                }
                fflush(stdout);
            }
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
            const dwEgomotionSpeedMeasurementType speedType = DW_EGOMOTION_REAR_WHEEL_SPEED;
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
            sprintf(buf, " Render thread: %u frames, %.1f FPS\n", renderCount, actualFps);
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
        
        dwTime_t now = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        
        // Phase 1: Drain sensor events (CAN fed immediately, IMU/GPS buffered)
        bool inCatchUp = drainSensorEventsToBuffers();
        
        // Phase 2: Skip fusion during catch-up to prevent falling behind
        if (inCatchUp) {
            return;
        }
        
        // Phase 3: Attempt fusion at configured rate
        if (now - m_lastFusionTimestamp >= FUSION_RATE_US) {
            bool fusionSuccess = attemptTemporalFusion(now);
            
            if (fusionSuccess) {
                m_lastFusionTimestamp = now;
                
                // Phase 4: Update trajectory if we have valid estimates
                performPoseEstimationAndTrajectory();
            }
        }
    }


    private:
    /**
     * Pose estimation and trajectory building (extracted from sample_egomotion onProcess)
     */
    void performPoseEstimationAndTrajectory() 
    {
        // =====================================================
        // GET ESTIMATION (egomotion updated itself via automaticUpdate)
        // =====================================================
        dwEgomotionResult estimate{};
        dwEgomotionUncertainty uncertainty{};
        dwStatus estStatus = dwEgomotion_getEstimation(&estimate, m_egomotion);
        dwStatus uncStatus = dwEgomotion_getUncertainty(&uncertainty, m_egomotion);

        log("\n=== Egomotion Query (POMO) ===\n");
        log("  dwEgomotion_getEstimation status: %s\n", dwGetStatusName(estStatus));
        log("  dwEgomotion_getUncertainty status: %s\n", dwGetStatusName(uncStatus));

        if (estStatus != DW_SUCCESS || uncStatus != DW_SUCCESS) {
            log("  Egomotion estimation or uncertainty not available\n");
            return;  // No valid estimate yet
        }

        log("  EgomotionResult:\n");
        log("    Timestamp: %lu us (%.6f s)\n", estimate.timestamp, estimate.timestamp * 1e-6);
        log("    ValidFlags: 0x%08X\n", estimate.validFlags);

        {
            float32_t roll, pitch, yaw;
            quaternionToEulerAngles(estimate.rotation, roll, pitch, yaw);
            log("    Rotation (Quaternion): w=%.6f, x=%.6f, y=%.6f, z=%.6f\n",
                estimate.rotation.w, estimate.rotation.x,
                estimate.rotation.y, estimate.rotation.z);
            log("    Rotation (Euler) [deg]: roll=%.2f, pitch=%.2f, yaw=%.2f\n",
                RAD2DEG(roll), RAD2DEG(pitch), RAD2DEG(yaw));
        }

        log("    Linear Velocity [m/s]: x=%.6f, y=%.6f, z=%.6f\n",
            estimate.linearVelocity[0],
            estimate.linearVelocity[1],
            estimate.linearVelocity[2]);
        log("    Linear Acceleration [m/s^2]: x=%.6f, y=%.6f, z=%.6f\n",
            estimate.linearAcceleration[0],
            estimate.linearAcceleration[1],
            estimate.linearAcceleration[2]);
        log("    Angular Velocity [rad/s]: x=%.6f (%.2f deg/s), y=%.6f (%.2f deg/s), z=%.6f (%.2f deg/s)\n",
            estimate.angularVelocity[0], RAD2DEG(estimate.angularVelocity[0]),
            estimate.angularVelocity[1], RAD2DEG(estimate.angularVelocity[1]),
            estimate.angularVelocity[2], RAD2DEG(estimate.angularVelocity[2]));

        log("  EgomotionUncertainty:\n");
        log("    ValidFlags: 0x%llX\n", static_cast<long long>(uncertainty.validFlags));
        log("    Rotation covariance diag [rad^2]: roll=%.6e, pitch=%.6e, yaw=%.6e\n",
            uncertainty.rotation.array[0],
            uncertainty.rotation.array[3 + 1],
            uncertainty.rotation.array[6 + 2]);
        log("    Linear Velocity StdDev [m/s]: x=%.6e, y=%.6e, z=%.6e\n",
            uncertainty.linearVelocity[0],
            uncertainty.linearVelocity[1],
            uncertainty.linearVelocity[2]);
        log("    Angular Velocity StdDev [rad/s]: x=%.6e, y=%.6e, z=%.6e\n",
            uncertainty.angularVelocity[0],
            uncertainty.angularVelocity[1],
            uncertainty.angularVelocity[2]);
        log("    Linear Acceleration StdDev [m/s^2]: x=%.6e, y=%.6e, z=%.6e\n",
            uncertainty.linearAcceleration[0],
            uncertainty.linearAcceleration[1],
            uncertainty.linearAcceleration[2]);
        
        // =====================================================
        // FEED TO GLOBAL EGOMOTION
        // =====================================================
        dwStatus globalStatus = dwGlobalEgomotion_addRelativeMotion(&estimate, &uncertainty, m_globalEgomotion);
        log("  dwGlobalEgomotion_addRelativeMotion status: %s\n", dwGetStatusName(globalStatus));
        
        // =====================================================
        // SAMPLE POSE AT CONFIGURED RATE
        // =====================================================
        if (estimate.timestamp < m_lastSampleTimestamp + POSE_SAMPLE_PERIOD) {
            return;  // Too soon to sample
        }
        
        // Compute relative transformation
        dwTransformation3f rigLast2rigNow{};
        dwStatus transStatus = dwEgomotion_computeRelativeTransformation(
            &rigLast2rigNow, nullptr, m_lastSampleTimestamp, estimate.timestamp, m_egomotion);
        log("  dwEgomotion_computeRelativeTransformation status: %s\n", dwGetStatusName(transStatus));
        
        if (transStatus != DW_SUCCESS) {
            return;
        }
        
        // Build pose
        Pose pose{};
        quaternionToEulerAngles(estimate.rotation, pose.rpy[0], pose.rpy[1], pose.rpy[2]);
        
        // Get previous world pose
        dwTransformation3f rigLast2world = DW_IDENTITY_TRANSFORMATION3F;
        if (!m_poseHistory.empty()) {
            rigLast2world = m_poseHistory.back().rig2world;
        } else if ((estimate.validFlags & DW_EGOMOTION_ROTATION) != 0) {
            dwMatrix3f rot{};
            getRotationMatrix(&rot, RAD2DEG(pose.rpy[0]), RAD2DEG(pose.rpy[1]), 0);
            rotationToTransformMatrix(rigLast2world.array, rot.array);
        }
        
        // Compute absolute pose
        dwTransformation3f rigNow2World{};
        dwEgomotion_applyRelativeTransformation(&rigNow2World, &rigLast2rigNow, &rigLast2world);

        log("  Absolute Pose (rigNow2World) matrix:\n");
        for (int i = 0; i < 4; ++i)
        {
            log("    [%.6f, %.6f, %.6f, %.6f]\n",
                rigNow2World.array[i + 0*4],
                rigNow2World.array[i + 1*4],
                rigNow2World.array[i + 2*4],
                rigNow2World.array[i + 3*4]);
        }
        log("  Translation: [%.6f, %.6f, %.6f]\n",
            rigNow2World.array[0 + 3*4],
            rigNow2World.array[1 + 3*4],
            rigNow2World.array[2 + 3*4]);
        
        pose.rig2world = rigNow2World;
        pose.timestamp = estimate.timestamp;
        pose.uncertainty = uncertainty;
        
        // Output to file
        if (m_outputFile) {
            fprintf(m_outputFile, "%lu,%.2f,%.2f,%.2f\n", estimate.timestamp,
                    rigNow2World.array[0 + 3 * 4], 
                    rigNow2World.array[1 + 3 * 4], 
                    rigNow2World.array[2 + 3 * 4]);
        }
        
        // Get global position
        dwGlobalEgomotionResult absoluteEstimate{};
        if (dwGlobalEgomotion_getEstimate(&absoluteEstimate, nullptr, m_globalEgomotion) == DW_SUCCESS &&
            absoluteEstimate.timestamp == estimate.timestamp && 
            absoluteEstimate.validOrientation) 
        {
            m_orientationENU = absoluteEstimate.orientation;
            m_hasOrientationENU = true;
            
            if (m_trajectoryLog.size("Egomotion") == 0) {
                m_trajectoryLog.addWGS84("Egomotion", m_currentGPSFrame);
            }
            m_trajectoryLog.addWGS84("Egomotion", absoluteEstimate.position);
        } else {
            m_hasOrientationENU = false;
        }
        
        // Manage buffer size
        if (m_poseHistory.size() >= MAX_BUFFER_POINTS) {
            size_t keepCount = MAX_BUFFER_POINTS / 2;
            m_poseHistory.erase(m_poseHistory.begin() + 1, 
                            m_poseHistory.end() - keepCount);
        }
        
        m_poseHistory.push_back(pose);
        m_lastSampleTimestamp = estimate.timestamp;
        m_shallRender = true;
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