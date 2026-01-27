/////////////////////////////////////////////////////////////////////////////////////////
//
// Real-time Vehicle CAN Parser for NVIDIA Drive Egomotion Integration
// Synchronized state management for live CAN data processing
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef SYGNALPOMOPARSER_HPP_
#define SYGNALPOMOPARSER_HPP_

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <dw/control/vehicleio/VehicleIO.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/rig/Rig.h>
#include <dw/egomotion/base/Egomotion.h>
#include <framework/Log.hpp>

#include <memory>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <deque>
#include <cstring>
#include <cstdint>

class SygnalPomoParser
{
public:
    struct VehicleCANConfiguration 
    {
        // CAN Message IDs (Hyundai/Kia specific)
        uint32_t speedCANId = 0x4F1;                    // CLU11
        uint32_t steeringWheelAngleCANId = 688;       // SAS11
        uint32_t wheelSpeedCANId = 902;               // WHL_SPD11 (single message for all wheels)
        uint32_t gearPositionCANId = 273;             // TCU11
        uint32_t yawRateCANId = 544;                  // ESP12
        
        // Vehicle calibration parameters
        float32_t steeringRatio = 16.0f;              // Steering wheel to front wheel angle ratio
        float32_t wheelRadius[4] = {0.31f, 0.31f, 0.31f, 0.31f}; // Wheel radius [FL, FR, RL, RR]
        float32_t wheelbase = 2.7f;                   // Distance between front and rear axles
        float32_t trackWidth = 1.5f;                  // Distance between left and right wheels
        
        // CAN message scaling factors (from DBC)
        float32_t speedScaleFactor = 0.5f;   // km/h to m/s
        float32_t steeringScaleFactor = (0.1f * M_PI / 180.0f); // 0.1 deg to radians
        float32_t wheelSpeedScaleFactor = (0.03125f / 3.6f);    // 0.03125 km/h to m/s
        float32_t yawRateScaleFactor = (0.01f * M_PI / 180.0f); // 0.01 deg/s to rad/s
        float32_t yawRateOffset = -40.95f * M_PI / 180.0f;      // Offset in radians
        
        // Real-time processing parameters
        dwTime_t velocityLatencyCompensation_us = 20000; // 20ms
        float32_t velocityFactor = 1.0f;
        dwTime_t temporalWindow_us = 500000;           // 500ms temporal coherency window
        dwTime_t stateCommitInterval_us = 10000;      // 10ms state commit interval
    };

    
    /// Parser state and diagnostics
    struct ParserDiagnostics 
    {
        std::atomic<uint32_t> totalCANMessagesReceived{0};
        std::atomic<uint32_t> validCANMessagesProcessed{0};
        std::atomic<uint32_t> invalidCANMessagesRejected{0};
        std::atomic<uint32_t> speedMessagesReceived{0};
        std::atomic<uint32_t> steeringMessagesReceived{0};
        std::atomic<uint32_t> wheelSpeedMessagesReceived{0};
        std::atomic<uint32_t> stateCommitsSuccessful{0};
        std::atomic<uint32_t> stateCommitsFailed{0};
        
        std::atomic<dwTime_t> lastSpeedMessageTimestamp{0};
        std::atomic<dwTime_t> lastSteeringMessageTimestamp{0};
        std::atomic<dwTime_t> lastWheelSpeedMessageTimestamp{0};
        std::atomic<dwTime_t> lastStateCommitTimestamp{0};
        
        std::atomic<bool> speedMessageTimeout{false};
        std::atomic<bool> steeringMessageTimeout{false};
        std::atomic<bool> wheelSpeedMessageTimeout{false};
        
        std::atomic<float32_t> averageMessageRate{0.0f};
        std::atomic<float32_t> averageCommitRate{0.0f};
    };

private:
    /// Synchronized Vehicle State Manager
    struct SynchronizedVehicleState {
        mutable std::mutex stateMutex;
        std::condition_variable stateCondition;
        
        // Current committed state (thread-safe read access)
        dwVehicleIOSafetyState committedSafetyState{};
        dwVehicleIONonSafetyState committedNonSafetyState{};
        dwTime_t committedStateTimestamp{0};
        std::atomic<bool> hasValidCommittedState{false};
        

        static constexpr uint32_t DW_VIO_SPEED_QUALITY_E_S_C_VALID = 1;
        static constexpr uint32_t DW_VIO_WHEEL_SPEED_QUALITY_VALID = 1;
        static constexpr uint32_t DW_VIO_FRONT_STEERING_ANGLE_QUALITY_VALID = 1;
        static constexpr uint32_t DW_VIO_WHEEL_TICKS_DIRECTION_FORWARD = 1;
        static constexpr uint32_t DW_VIO_WHEEL_TICKS_DIRECTION_BACKWARD = 2;
        static constexpr uint32_t DW_VIO_WHEEL_TICKS_DIRECTION_STANDSTILL = 3;
        static constexpr uint32_t DW_VIO_VEHICLE_STOPPED_STOPPED = 1;
        static constexpr uint32_t DW_VIO_VEHICLE_STOPPED_MOVING = 2;
        
        // State assembly buffer (protected by stateMutex)
        struct StateBuffer {
            dwVehicleIOSafetyState pendingSafety{};
            dwVehicleIONonSafetyState pendingNonSafety{};
            
            dwTime_t lastSpeedUpdate{0};
            dwTime_t lastSteeringUpdate{0};
            dwTime_t lastWheelSpeedUpdate{0};
            dwTime_t lastGearUpdate{0};
            
            bool hasSpeed{false};
            bool hasSteering{false};
            bool hasWheelSpeeds{false};
            bool hasGear{false};
        } stateBuffer;
        
        // Constructor
        SynchronizedVehicleState() {
            committedSafetyState.size = sizeof(dwVehicleIOSafetyState);
            committedNonSafetyState.size = sizeof(dwVehicleIONonSafetyState);
            stateBuffer.pendingSafety.size = sizeof(dwVehicleIOSafetyState);
            stateBuffer.pendingNonSafety.size = sizeof(dwVehicleIONonSafetyState);
            
            // Initialize with safe defaults
            committedNonSafetyState.speedDirectionESC = DW_VIO_SPEED_DIRECTION_E_S_C_VOID;
            committedNonSafetyState.drivePositionStatus = DW_VIO_DRIVE_POSITION_STATUS_N;
            stateBuffer.pendingNonSafety.speedDirectionESC = DW_VIO_SPEED_DIRECTION_E_S_C_VOID;
            stateBuffer.pendingNonSafety.drivePositionStatus = DW_VIO_DRIVE_POSITION_STATUS_N;
        }
        
        // Thread-safe state access
        dwVehicleIOSafetyState getSafetyState() const {
            std::lock_guard<std::mutex> lock(stateMutex);
            return committedSafetyState;
        }
        
        dwVehicleIONonSafetyState getNonSafetyState() const {
            std::lock_guard<std::mutex> lock(stateMutex);
            return committedNonSafetyState;
        }
        
        bool isTemporallyCoherent(dwTime_t referenceTime, dwTime_t temporalWindow, 
                  bool requireWheelSpeeds = false) const {
            
            bool baseCoherent = (referenceTime - stateBuffer.lastSpeedUpdate <= temporalWindow) &&
                            (referenceTime - stateBuffer.lastSteeringUpdate <= temporalWindow);
            
            
            
            if (requireWheelSpeeds) {
                
                bool result = baseCoherent && 
                    (referenceTime - stateBuffer.lastWheelSpeedUpdate <= temporalWindow);

                return result;
            }
            
            return baseCoherent;
        }


        
        bool isStateComplete() const {
            bool result = stateBuffer.hasSpeed && stateBuffer.hasSteering;
            return result;
        }
    };
    
    // Configuration and core components
    VehicleCANConfiguration m_configuration;
    std::unique_ptr<SynchronizedVehicleState> m_vehicleState;
    
    // Parser state management
    std::atomic<bool> m_isInitialized{false};
    std::atomic<bool> m_configurationLoaded{false};
    std::atomic<dwTime_t> m_initializationTimestamp{0};
    
    // Diagnostics and monitoring
    std::unique_ptr<ParserDiagnostics> m_diagnostics;
    
    // Timeout thresholds (microseconds)
    static constexpr dwTime_t DEFAULT_MESSAGE_TIMEOUT = 50000;     // 50ms
    static constexpr dwTime_t DEFAULT_TEMPORAL_WINDOW = 500000;      // 500ms
    
    // Maximum allowed values for validation
    static constexpr float32_t MAX_VEHICLE_SPEED = 100.0f;          // m/s
    static constexpr float32_t MAX_STEERING_ANGLE = 8.0f;          // radians
    static constexpr float32_t MAX_WHEEL_SPEED = 300.0f;            // rad/s

public:
    /// Constructor and destructor
    SygnalPomoParser();
    ~SygnalPomoParser();

    /// Initialization methods
    bool initializeFromRig(dwRigHandle_t rigConfig, const char* vehicleSensorName);
    bool loadVehicleConfiguration(const VehicleCANConfiguration& config);
    bool isInitialized() const { return m_isInitialized.load(); }

    /// Real-time CAN message processing
    bool processCANFrame(const dwCANMessage& frame);
    
    /// Thread-safe vehicle state access
    bool getTemporallySynchronizedState(dwVehicleIOSafetyState* safetyState, 
                                   dwVehicleIONonSafetyState* nonSafetyState,
                                    dwVehicleIOActuationFeedback* actuationFeedback = nullptr);
    dwVehicleIOSafetyState getSafetyState() const;
    dwVehicleIONonSafetyState getNonSafetyState() const;
    bool hasValidState() const;
    
    void getCurrentState(dwVehicleIOSafetyState* safety, 
                     dwVehicleIONonSafetyState* nonSafety,
                     dwVehicleIOActuationFeedback* actuation);
    /// Diagnostics and monitoring
    const ParserDiagnostics& getDiagnostics() const { return *m_diagnostics; }
    void resetDiagnostics();
    bool checkMessageTimeouts(dwTime_t currentTimestamp);
    
    /// Configuration access
    const VehicleCANConfiguration& getConfiguration() const { return m_configuration; }

private:
    /// Initialization helpers
    void initializeStructures();
    bool validateConfiguration(const VehicleCANConfiguration& config);
    bool extractParametersFromRig(dwRigHandle_t rigConfig, const char* vehicleSensorName);

    /// Real-time CAN message processing (thread-safe)
    bool processSpeedMessage(const dwCANMessage& frame);
    bool processSteeringMessage(const dwCANMessage& frame);
    bool processWheelSpeedMessage(const dwCANMessage& frame);
    bool processGearPositionMessage(const dwCANMessage& frame);
    bool processYawRateMessage(const dwCANMessage& frame);
    
    /// Message validation and parsing
    bool validateCANMessage(const dwCANMessage& frame);
    float32_t extractVehicleSpeed(const uint8_t* data, uint8_t length);
    dwVioSpeedDirectionESC extractSpeedDirection(const uint8_t* data, uint8_t length);
    float32_t extractSteeringWheelAngle(const uint8_t* data, uint8_t length);
    float32_t extractWheelSpeed(const uint8_t* data, uint8_t length, uint8_t wheelIndex);
    dwVioDrivePositionStatus extractGearPosition(const uint8_t* data, uint8_t length);
    float32_t extractYawRate(const uint8_t* data, uint8_t length);

    /// State management and latency compensation
    void applyLatencyCompensation(dwVehicleIONonSafetyState& state, dwTime_t referenceTime);
    void updateDiagnostics(const dwCANMessage& frame, bool processed);
    void updateCommitDiagnostics(bool successful);

    /// Coordinate frame and unit conversions
    float32_t convertSteeringWheelToFrontWheelAngle(float32_t steeringWheelAngle);
    float32_t convertWheelSpeedToLinearSpeed(float32_t wheelSpeed, uint8_t wheelIndex);
    bool validatePhysicalLimits(float32_t speed, float32_t steeringAngle);

    /// Utility methods
    void logParserState() const;
    void logDiagnostics() const;
};

#endif // SYGNALPOMOPARSER_HPP_