#ifndef SYGNALPOMO_COMMON_HPP_
#define SYGNALPOMO_COMMON_HPP_

#include <dw/core/base/Types.h>
#include <dw/sensors/common/SensorTypes.h>
#include <dw/sensors/canbus/CANTypes.h>

namespace sygnalpomo {

//------------------------------------------------------------------------------
// Plugin Versioning
//------------------------------------------------------------------------------
constexpr uint32_t PLUGIN_VERSION_MAJOR = 1;
constexpr uint32_t PLUGIN_VERSION_MINOR = 1;  // Updated for optimized version
constexpr uint32_t PLUGIN_VERSION_PATCH = 2;
constexpr const char* PLUGIN_VERSION_STRING = "1.0.2-SygnalPomoDriver";
//------------------------------------------------------------------------------
// CAN-Specific Constants (updated based on empirical testing)
//------------------------------------------------------------------------------
constexpr uint32_t DEFAULT_CAN_TIMEOUT_US = 100000;  // Default timeout: 100 ms
constexpr size_t MAX_FILTER_COUNT = DW_SENSORS_CAN_MAX_FILTERS;

// Optimized CAN Message IDs (from successful Python logs)
constexpr uint32_t CONTROL_ENABLE_MSG_ID = 0x060;    // 96 decimal
constexpr uint32_t CONTROL_COMMAND_MSG_ID = 0x160;   // 352 decimal  
constexpr uint32_t HEARTBEAT_MSG_ID = 0x170;         // 368 decimal
constexpr uint32_t FAULT_MSG_ID = 0x020;             // 32 decimal

// Bus addresses (from successful Python logs)
constexpr uint8_t MCM_BUS_ADDRESS = 1;               // Motion Control Module
constexpr uint8_t CB_BUS_ADDRESS = 3;                // Control Box

//------------------------------------------------------------------------------
// Optimized Gear Change Parameters (empirically tested and proven)
//------------------------------------------------------------------------------
namespace OptimizedGearChange {
    constexpr int32_t BRAKE_PERCENTAGE = 40;         // 40% brake force
    constexpr float BRAKE_DURATION_SEC = 2.0f;       // 2.0 seconds total
    constexpr float GEAR_INTERVAL_SEC = 0.5f;        // 0.5 second intervals
    constexpr int TOTAL_GEAR_COMMANDS = 4;           // 4 commands total
    constexpr uint32_t INTERVAL_MS = 500;            // 500ms between commands
    
    // Gear position float values (matches Python DBC mapping)
    constexpr float GEAR_PARK_VALUE = 0.0f;          // Park
    constexpr float GEAR_REVERSE_VALUE = 56.0f;      // Reverse  
    constexpr float GEAR_NEUTRAL_VALUE = 48.0f;      // Neutral
    constexpr float GEAR_DRIVE_VALUE = 40.0f;        // Drive
}

//------------------------------------------------------------------------------
// Plugin State Tracking
//------------------------------------------------------------------------------
enum class PluginState {
    UNINITIALIZED,
    INITIALIZED,
    RUNNING,
    STOPPED,
    ERROR
};

//------------------------------------------------------------------------------
// Control Subsystem Enumeration (matches Python implementation)
//------------------------------------------------------------------------------
enum class ControlSubsystem : uint8_t {
    THROTTLE = 0,
    BRAKE = 1,
    STEERING = 2,
    GEAR = 3,
    TURN_SIGNAL = 4,
    HEADLIGHT = 5,
    WIPER = 6
};

//------------------------------------------------------------------------------
// Gear Position Enumeration
//------------------------------------------------------------------------------
enum class GearPosition : int32_t {
    PARK = 0,
    REVERSE = 1,
    NEUTRAL = 2,
    DRIVE = 3
};

//------------------------------------------------------------------------------
// CAN Message Filter Validation
//------------------------------------------------------------------------------
inline bool isValidFilterConfig(const uint32_t* ids, uint16_t numCanIDs) {
    if (!ids || numCanIDs == 0 || numCanIDs > MAX_FILTER_COUNT) {
        return false;
    }
    for (uint16_t i = 0; i < numCanIDs; ++i) {
        if (ids[i] > (1u << DW_SENSORS_CAN_MAX_ID_LEN)) {
            return false;
        }
    }
    return true;
}

//------------------------------------------------------------------------------
// Plugin State Utilities
//------------------------------------------------------------------------------
inline bool isInitialized(PluginState state) {
    return state != PluginState::UNINITIALIZED;
}

inline bool isRunning(PluginState state) {
    return state == PluginState::RUNNING;
}

//------------------------------------------------------------------------------
// Gear Position Utilities
//------------------------------------------------------------------------------
inline float getGearPositionValue(GearPosition gear) {
    switch (gear) {
        case GearPosition::PARK:    return OptimizedGearChange::GEAR_PARK_VALUE;
        case GearPosition::REVERSE: return OptimizedGearChange::GEAR_REVERSE_VALUE;
        case GearPosition::NEUTRAL: return OptimizedGearChange::GEAR_NEUTRAL_VALUE;
        case GearPosition::DRIVE:   return OptimizedGearChange::GEAR_DRIVE_VALUE;
        default:                    return OptimizedGearChange::GEAR_PARK_VALUE;
    }
}

inline const char* getGearPositionName(GearPosition gear) {
    switch (gear) {
        case GearPosition::PARK:    return "PARK";
        case GearPosition::REVERSE: return "REVERSE";
        case GearPosition::NEUTRAL: return "NEUTRAL";
        case GearPosition::DRIVE:   return "DRIVE";
        default:                    return "UNKNOWN";
    }
}

//------------------------------------------------------------------------------
// Subsystem Configuration Utilities
//------------------------------------------------------------------------------
inline uint8_t getSubsystemBusAddress(ControlSubsystem subsystem) {
    switch (subsystem) {
        case ControlSubsystem::THROTTLE:
        case ControlSubsystem::BRAKE:
        case ControlSubsystem::STEERING:
        case ControlSubsystem::TURN_SIGNAL:
        case ControlSubsystem::HEADLIGHT:
        case ControlSubsystem::WIPER:
            return MCM_BUS_ADDRESS;  // Motion Control Module
            
        case ControlSubsystem::GEAR:
            return CB_BUS_ADDRESS;   // Control Box
            
        default:
            return MCM_BUS_ADDRESS;
    }
}

inline const char* getSubsystemName(ControlSubsystem subsystem) {
    switch (subsystem) {
        case ControlSubsystem::THROTTLE:    return "THROTTLE";
        case ControlSubsystem::BRAKE:       return "BRAKE";
        case ControlSubsystem::STEERING:    return "STEERING";
        case ControlSubsystem::GEAR:        return "GEAR";
        case ControlSubsystem::TURN_SIGNAL: return "TURN_SIGNAL";
        case ControlSubsystem::HEADLIGHT:   return "HEADLIGHT";
        case ControlSubsystem::WIPER:       return "WIPER";
        default:                            return "UNKNOWN";
    }
}

} // namespace sygnalpomo

#endif // SYGNALPOMO_COMMON_HPP_