#ifndef SYGNALPOMO_CONTROLLER_HPP_
#define SYGNALPOMO_CONTROLLER_HPP_

#include <dw/core/context/Context.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/canbus/CAN.h>
#include <dw/sensors/legacy/plugins/canbus/CANPlugin.h>
#include <framework/ProgramArguments.hpp>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>


enum ControlInterface : uint8_t {
    THROTTLE = 0,  // Matches plugin ControlSubsystem::THROTTLE = 0
    BRAKE = 1,     // Matches plugin ControlSubsystem::BRAKE = 1
    STEERING = 2,  // Matches plugin ControlSubsystem::STEERING = 2
    GEAR = 3       // Matches plugin ControlSubsystem::GEAR = 3
};

// Gear positions for automated testing
enum GearPosition : int32_t {
    PARK = 0,
    REVERSE = 1,
    NEUTRAL = 2,
    DRIVE = 3
};

class SygnalPomoController {
public:
    SygnalPomoController();
    ~SygnalPomoController();
    
    // ==========================================================================
    // Core initialization and management
    // ==========================================================================
    bool initialize(const ProgramArguments& arguments,
                    dwContextHandle_t context,
                    dwSALHandle_t sal);
    void release();
    
    // ==========================================================================
    // High-Level Control Interface (Simple Commands)
    // ==========================================================================
    bool applyThrottle(int32_t percentage, float duration_sec = 0.0f);
    bool applyBrake(int32_t percentage, float duration_sec = 0.0f);
    bool setSteering(int32_t percentage);
    bool changeGear(GearPosition gearValue);
    
    // Legacy function name for compatibility
    bool executeGearChangeSequence(GearPosition gearValue);
    
    // ==========================================================================
    // Low-Level Control Interface (Direct Plugin Commands)
    // ==========================================================================
    bool sendControlCommand(ControlInterface interfaceId, int32_t value, float duration_sec = 0.0f);
    
    // ==========================================================================
    // Automated Test Sequences
    // ==========================================================================
    bool runAutomatedTests();
    bool runThrottleTest();
    bool runBrakeTest();
    bool runSteeringTest();
    bool runGearChangeTest();
    
    // ==========================================================================
    // System Management
    // ==========================================================================
    bool resetHeartbeat(uint8_t subsystemID);
    bool clearAllSubsystems();
    
    // ==========================================================================
    // Message Reading and Monitoring
    // ==========================================================================
    bool readMessages();
    
    // ==========================================================================
    // State Queries
    // ==========================================================================
    bool isSystemFaulted() const;
    uint8_t getSystemState() const;

private:
    // ==========================================================================
    // Plugin management
    // ==========================================================================
    dwStatus loadPlugin(const std::string& pluginPath);
    std::string constructPluginParams(const ProgramArguments& arguments);
    
    // ==========================================================================
    // Utility functions
    // ==========================================================================
    static const char* getInterfaceName(ControlInterface interfaceId);
    static const char* getGearName(GearPosition gear);
    
    // ==========================================================================
    // DriveWorks handles
    // ==========================================================================
    dwContextHandle_t m_context;
    dwSALHandle_t m_sal;
    dwSensorPluginSensorHandle_t m_pluginSensorHandle;
    
    // ==========================================================================
    // Plugin-related members
    // ==========================================================================
    void* m_pluginHandle;
    typedef dwStatus (*GetFunctionTableFn)(dwSensorCANPluginFunctionTable*);
    GetFunctionTableFn m_getFunctionTableFn;
    dwSensorCANPluginFunctionTable m_pluginFunctions;
    
    // ==========================================================================
    // State tracking
    // ==========================================================================
    bool m_initialized;
    uint8_t m_busAddress;
    uint8_t m_subsystemID;
    uint8_t m_controlCounter;
    uint8_t m_systemState;
    bool m_systemFaulted;
    std::atomic<bool> m_running;
    mutable std::mutex m_stateMutex;
    
    // ==========================================================================
    // Control value tracking (all supported interfaces)
    // ==========================================================================
    struct ControlValue {
        int32_t value;
        bool updated;
    };
    std::map<ControlInterface, ControlValue> m_controlValues;
    std::mutex m_controlMutex;
    std::condition_variable m_controlCV;
    
    static constexpr dwTime_t DEFAULT_TIMEOUT_US = 100000;
};

#endif // SYGNALPOMO_CONTROLLER_HPP_