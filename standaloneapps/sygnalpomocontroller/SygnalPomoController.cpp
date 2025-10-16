#include "SygnalPomoController.hpp"
#include <framework/Log.hpp>
#include <framework/Checks.hpp>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <dlfcn.h>
#include <map>

SygnalPomoController::SygnalPomoController()
    : m_context(DW_NULL_HANDLE)
    , m_sal(DW_NULL_HANDLE)
    , m_pluginSensorHandle(nullptr)
    , m_initialized(false)
    , m_pluginHandle(nullptr)
    , m_getFunctionTableFn(nullptr)
    , m_controlCounter(0)
    , m_systemState(0)
    , m_systemFaulted(false)
    , m_running(false)
{
    std::memset(&m_pluginFunctions, 0, sizeof(m_pluginFunctions));
    
    // Initialize control values for all supported interfaces
    m_controlValues[THROTTLE] = {0, false};
    m_controlValues[BRAKE] = {0, false};
    m_controlValues[STEERING] = {0, false};
    m_controlValues[GEAR] = {0, false};
}

SygnalPomoController::~SygnalPomoController() {
    release();
}

dwStatus SygnalPomoController::loadPlugin(const std::string& pluginPath) {
    m_pluginHandle = dlopen(pluginPath.c_str(), RTLD_NOW);
    if (!m_pluginHandle) {
        std::cerr << "[CONTROLLER] Failed to load plugin: " << dlerror() << std::endl;
        return DW_NOT_AVAILABLE;
    }

    m_getFunctionTableFn = reinterpret_cast<GetFunctionTableFn>(
        dlsym(m_pluginHandle, "dwSensorCANPlugin_getFunctionTable"));
    
    if (!m_getFunctionTableFn) {
        std::cerr << "[CONTROLLER] Failed to get function table: " << dlerror() << std::endl;
        dlclose(m_pluginHandle);
        m_pluginHandle = nullptr;
        return DW_NOT_AVAILABLE;
    }

    dwStatus status = m_getFunctionTableFn(&m_pluginFunctions);
    if (status != DW_SUCCESS) {
        std::cerr << "[CONTROLLER] Failed to retrieve plugin functions: " 
                  << dwGetStatusName(status) << std::endl;
        dlclose(m_pluginHandle);
        m_pluginHandle = nullptr;
        return status;
    }

    std::cout << "[CONTROLLER] Plugin loaded successfully: " << pluginPath << std::endl;
    return DW_SUCCESS;
}

std::string SygnalPomoController::constructPluginParams(const ProgramArguments& arguments) {
    std::string params = arguments.get("params").c_str();
    
    if (!arguments.get("dbc-path").empty()) {
        params += ",dbc-path=" + std::string(arguments.get("dbc-path").c_str());
    }
    
    return params;
}

bool SygnalPomoController::initialize(const ProgramArguments& arguments,
                                    dwContextHandle_t context,
                                    dwSALHandle_t sal) {
    m_context = context;
    m_sal = sal;
    m_busAddress = 1;
    m_subsystemID = 0;

    std::string pluginPath = arguments.get("plugin-path");
    if (pluginPath.empty()) {
        std::cerr << "[CONTROLLER] Plugin path not specified!" << std::endl;
        return false;
    }

    // Load the plugin
    if (loadPlugin(pluginPath) != DW_SUCCESS) {
        return false;
    }

    // Create plugin sensor handle
    dwSensorPluginProperties props{};
    std::string paramString = constructPluginParams(arguments);
    
    dwStatus handleStatus = m_pluginFunctions.common.createHandle(
        &m_pluginSensorHandle, &props, paramString.c_str(), m_context);

    if (handleStatus != DW_SUCCESS || !m_pluginSensorHandle) {
        std::cerr << "[CONTROLLER] Failed to create plugin handle: " 
                  << dwGetStatusName(handleStatus) << std::endl;
        return false;
    }

    // Initialize the plugin sensor with SAL
    dwSALObject* salObj = reinterpret_cast<dwSALObject*>(m_sal);
    handleStatus = m_pluginFunctions.common.createSensor(
        paramString.c_str(), salObj, m_pluginSensorHandle);

    if (handleStatus != DW_SUCCESS) {
        std::cerr << "[CONTROLLER] Failed to initialize plugin sensor: " 
                  << dwGetStatusName(handleStatus) << std::endl;
        return false;
    }

    // Start the plugin sensor
    handleStatus = m_pluginFunctions.common.start(m_pluginSensorHandle);
    if (handleStatus != DW_SUCCESS) {
        std::cerr << "[CONTROLLER] Failed to start plugin sensor: " 
                  << dwGetStatusName(handleStatus) << std::endl;
        return false;
    }

    m_initialized = true;
    std::cout << "[CONTROLLER] Sygnal Pomo CAN Plugin initialized successfully" << std::endl;
    std::cout << "[CONTROLLER] Supported controls: Throttle, Brake, Steering, Gear" << std::endl;
    return true;
}

// =============================================================================
// Core Control Commands - Simple Interface
// =============================================================================

bool SygnalPomoController::sendControlCommand(ControlInterface interfaceId, int32_t value, float duration_sec) {
    if (!m_initialized) {
        std::cerr << "[CONTROLLER] Not initialized" << std::endl;
        return false;
    }

    if (!m_pluginFunctions.send) {
        std::cerr << "[CONTROLLER] Plugin send function not available!" << std::endl;
        return false;
    }

    // Create intent message for the plugin to process
    dwCANMessage msg;
    std::memset(&msg, 0, sizeof(msg));
    msg.id = 0xFFFF;  // Intent message ID that plugin recognizes
    msg.size = 8;
    msg.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    
    // Pack the intent message data as plugin expects
    msg.data[0] = static_cast<uint8_t>(interfaceId);
    std::memcpy(&msg.data[1], &value, sizeof(int32_t));
    std::memcpy(&msg.data[5], &duration_sec, sizeof(float));

    dwStatus status = m_pluginFunctions.send(&msg, DEFAULT_TIMEOUT_US, m_pluginSensorHandle);
    if (status != DW_SUCCESS) {
        std::cerr << "[CONTROLLER] Failed to send command: " << dwGetStatusName(status) << std::endl;
        return false;
    }

    // Update local state for tracking
    {
        std::unique_lock<std::mutex> lock(m_controlMutex);
        m_controlValues[interfaceId].value = value;
        m_controlValues[interfaceId].updated = true;
    }
    
    std::cout << "[CONTROLLER] Sent " << getInterfaceName(interfaceId)
              << " command: " << value;
    if (duration_sec > 0.0f) {
        std::cout << " (duration: " << duration_sec << "s)";
    }
    std::cout << std::endl;
    
    return true;
}

// Simple high-level interface functions
bool SygnalPomoController::applyThrottle(int32_t percentage, float duration_sec) {
    if (percentage < 0 || percentage > 100) {
        std::cerr << "[CONTROLLER] Invalid throttle percentage: " << percentage << std::endl;
        return false;
    }
    
    std::cout << "[CONTROLLER] Applying " << percentage << "% throttle";
    if (duration_sec > 0.0f) {
        std::cout << " for " << duration_sec << " seconds";
    }
    std::cout << std::endl;
    
    return sendControlCommand(THROTTLE, percentage, duration_sec);
}

bool SygnalPomoController::applyBrake(int32_t percentage, float duration_sec) {
    if (percentage < 0 || percentage > 100) {
        std::cerr << "[CONTROLLER] Invalid brake percentage: " << percentage << std::endl;
        return false;
    }
    
    std::cout << "[CONTROLLER] Applying " << percentage << "% brake";
    if (duration_sec > 0.0f) {
        std::cout << " for " << duration_sec << " seconds";
    }
    std::cout << std::endl;
    
    return sendControlCommand(BRAKE, percentage, duration_sec);
}

bool SygnalPomoController::setSteering(int32_t percentage) {
    if (percentage < -100 || percentage > 100) {
        std::cerr << "[CONTROLLER] Invalid steering percentage: " << percentage << std::endl;
        return false;
    }
    
    std::cout << "[CONTROLLER] Setting steering to " << percentage << "%";
    if (percentage == 0) {
        std::cout << " (center)";
    } else if (percentage > 0) {
        std::cout << " (right)";
    } else {
        std::cout << " (left)";
    }
    std::cout << std::endl;
    
    return sendControlCommand(STEERING, percentage, 0.0f);
}

bool SygnalPomoController::changeGear(GearPosition gearValue) {
    std::cout << "[CONTROLLER] Requesting gear change to " << getGearName(gearValue) << std::endl;
    std::cout << "[CONTROLLER] Plugin will execute optimized 4-phase sequence automatically" << std::endl;
    
    // Plugin handles the entire complex sequence internally
    return sendControlCommand(GEAR, static_cast<int32_t>(gearValue), 0.0f);
}

// Legacy function name for compatibility
bool SygnalPomoController::executeGearChangeSequence(GearPosition gearValue) {
    return changeGear(gearValue);
}

// =============================================================================
// Automated Test Sequences - Focus on Interface Testing
// =============================================================================

bool SygnalPomoController::runAutomatedTests() {
    std::cout << "\n[AUTOMATED TESTS] Starting comprehensive test sequence..." << std::endl;
    std::cout << "[AUTOMATED TESTS] Testing high-level controller interface" << std::endl;
    
    // Clear all subsystems first
    if (!clearAllSubsystems()) {
        std::cerr << "[AUTOMATED TESTS] Failed to clear subsystems" << std::endl;
        return false;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // Run individual tests
    bool success = true;
    success &= runThrottleTest();
    success &= runBrakeTest();
    success &= runSteeringTest();
    success &= runGearChangeTest();
    
    // Reset all controls to neutral/safe state
    std::cout << "[AUTOMATED TESTS] Resetting all controls to safe state..." << std::endl;
    applyThrottle(0);
    applyBrake(0);
    setSteering(0);
    changeGear(PARK);
    
    // Wait for final gear change to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    
    std::cout << "[AUTOMATED TESTS] Test sequence " 
              << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

bool SygnalPomoController::runThrottleTest() {
    std::cout << "\n[THROTTLE TEST] Testing throttle control..." << std::endl;
    
    // Test progressive throttle
    std::vector<int32_t> throttleValues = {10, 25, 50, 75, 100};
    
    for (int32_t throttleValue : throttleValues) {
        if (!applyThrottle(throttleValue)) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    // Test timed throttle application
    std::cout << "[THROTTLE TEST] Testing timed throttle (30% for 2 seconds)..." << std::endl;
    if (!applyThrottle(30, 2.0f)) {
        return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    
    // Release throttle
    std::cout << "[THROTTLE TEST] Releasing throttle..." << std::endl;
    applyThrottle(0);
    
    std::cout << "[THROTTLE TEST] Throttle test completed" << std::endl;
    return true;
}

bool SygnalPomoController::runBrakeTest() {
    std::cout << "\n[BRAKE TEST] Testing brake control..." << std::endl;
    
    // Test progressive braking
    std::vector<int32_t> brakeValues = {10, 25, 40, 60, 80};
    
    for (int32_t brakeValue : brakeValues) {
        if (!applyBrake(brakeValue)) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    // Test optimal brake parameters (matches plugin's optimized values)
    std::cout << "[BRAKE TEST] Testing optimized brake (40% for 2 seconds)..." << std::endl;
    if (!applyBrake(40, 2.0f)) {
        return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    
    // Release brakes
    std::cout << "[BRAKE TEST] Releasing brakes..." << std::endl;
    applyBrake(0);
    
    std::cout << "[BRAKE TEST] Brake test completed" << std::endl;
    return true;
}

bool SygnalPomoController::runSteeringTest() {
    std::cout << "\n[STEERING TEST] Testing steering control..." << std::endl;
    
    // Test steering range
    std::vector<int32_t> steeringValues = {-100, -50, -25, 0, 25, 50, 100, 0};
    
    for (int32_t steeringValue : steeringValues) {
        if (!setSteering(steeringValue)) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
    }
    
    std::cout << "[STEERING TEST] Steering test completed" << std::endl;
    return true;
}

bool SygnalPomoController::runGearChangeTest() {
    std::cout << "\n[GEAR TEST] Testing gear changes with optimized plugin sequences..." << std::endl;
    
    // Test all gear positions
    std::vector<GearPosition> gears = {PARK, REVERSE, NEUTRAL, DRIVE, PARK};
    
    for (GearPosition gear : gears) {
        std::cout << "[GEAR TEST] Shifting to " << getGearName(gear) << "..." << std::endl;
        if (!changeGear(gear)) {
            return false;
        }
        
        // Wait for plugin to complete the full optimized sequence
        // (2s brake duration + 0.5s buffer = 2.5s minimum)
        std::cout << "[GEAR TEST] Waiting for plugin sequence to complete..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    }
    
    std::cout << "[GEAR TEST] All gear changes completed successfully" << std::endl;
    return true;
}

// =============================================================================
// System Management Functions
// =============================================================================

bool SygnalPomoController::resetHeartbeat(uint8_t subsystemID) {
    dwCANMessage seedMsg;
    std::memset(&seedMsg, 0, sizeof(seedMsg));
    seedMsg.id = 112;  // HeartbeatClearSeed ID
    seedMsg.size = 8;
    
    seedMsg.data[0] = uint8_t((subsystemID << 7) | (m_busAddress & 0x7F));
    
    uint32_t seed = static_cast<uint32_t>(std::time(nullptr) * 1000) % 0xFFFFFFFF;
    seedMsg.data[1] = (seed >> 0) & 0xFF;
    seedMsg.data[2] = (seed >> 8) & 0xFF;
    seedMsg.data[3] = (seed >> 16) & 0xFF;
    seedMsg.data[4] = (seed >> 24) & 0xFF;
    
    dwStatus seedStatus = m_pluginFunctions.send(&seedMsg, DEFAULT_TIMEOUT_US, m_pluginSensorHandle);
    if (seedStatus != DW_SUCCESS) {
        std::cerr << "[CONTROLLER] Failed to send HeartbeatClearSeed: " 
                  << dwGetStatusName(seedStatus) << std::endl;
        return false;
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Send key message
    dwCANMessage keyMsg;
    std::memset(&keyMsg, 0, sizeof(keyMsg));
    keyMsg.id = 127;  // HeartbeatClearKey ID
    keyMsg.size = 8;
    keyMsg.data[0] = seedMsg.data[0];
    
    uint32_t key = seed ^ 0x12345678;
    keyMsg.data[1] = (key >> 0) & 0xFF;
    keyMsg.data[2] = (key >> 8) & 0xFF;
    keyMsg.data[3] = (key >> 16) & 0xFF;
    keyMsg.data[4] = (key >> 24) & 0xFF;
    
    dwStatus keyStatus = m_pluginFunctions.send(&keyMsg, DEFAULT_TIMEOUT_US, m_pluginSensorHandle);
    if (keyStatus == DW_SUCCESS) {
        std::cout << "[CONTROLLER] Heartbeat reset successful for subsystem " << static_cast<int>(subsystemID) << std::endl;
    }
    return (keyStatus == DW_SUCCESS);
}

bool SygnalPomoController::clearAllSubsystems() {
    std::cout << "[CONTROLLER] Clearing all subsystem states..." << std::endl;
    bool success = true;
    for (uint8_t subsystemID = 0; subsystemID <= 1; subsystemID++) {
        if (!resetHeartbeat(subsystemID)) {
            success = false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return success;
}

bool SygnalPomoController::readMessages() {
    if (!m_initialized) return false;
    
    uint8_t const* rawData;
    size_t size;
    dwTime_t timestamp;
    
    dwStatus status = m_pluginFunctions.common.readRawData(
        &rawData, &size, &timestamp, 1000, m_pluginSensorHandle); // 1ms timeout
    
    if (status == DW_SUCCESS) {
        if (size >= 12) {  // Minimum size for header
            uint32_t payloadSize;
            dwTime_t msgTimestamp;
            std::memcpy(&payloadSize, rawData, sizeof(uint32_t));
            std::memcpy(&msgTimestamp, rawData + sizeof(uint32_t), sizeof(dwTime_t));
            
            std::cout << "[CONTROLLER] Received CAN message - Size: " << payloadSize 
                      << ", Timestamp: " << msgTimestamp << std::endl;
        }

        m_pluginFunctions.common.returnRawData(rawData, m_pluginSensorHandle);
        return true;
    }
    
    return false;
}

// =============================================================================
// Helper Functions
// =============================================================================

const char* SygnalPomoController::getInterfaceName(ControlInterface interfaceId) {
    switch (interfaceId) {
        case THROTTLE: return "throttle";
        case BRAKE: return "brake";
        case STEERING: return "steering";
        case GEAR: return "gear";
        default: return "unknown";
    }
}

const char* SygnalPomoController::getGearName(GearPosition gear) {
    switch (gear) {
        case PARK: return "Park";
        case REVERSE: return "Reverse";
        case NEUTRAL: return "Neutral";
        case DRIVE: return "Drive";
        default: return "Unknown";
    }
}

uint8_t SygnalPomoController::getSystemState() const {
    std::unique_lock<std::mutex> lock(m_stateMutex);
    return m_systemState;
}

bool SygnalPomoController::isSystemFaulted() const {
    std::unique_lock<std::mutex> lock(m_stateMutex);
    return m_systemFaulted;
}

void SygnalPomoController::release() {
    m_running = false;
    
    std::cout << "[CONTROLLER] Shutting down..." << std::endl;
    
    // Stop and release plugin sensor
    if (m_pluginSensorHandle) {
        if (m_pluginFunctions.common.stop) {
            m_pluginFunctions.common.stop(m_pluginSensorHandle);
        }
        if (m_pluginFunctions.common.release) {
            m_pluginFunctions.common.release(m_pluginSensorHandle);
        }
        m_pluginSensorHandle = nullptr;
    }
    
    // Unload plugin
    if (m_pluginHandle) {
        dlclose(m_pluginHandle);
        m_pluginHandle = nullptr;
    }

    m_initialized = false;
    m_context = DW_NULL_HANDLE;
    m_sal = DW_NULL_HANDLE;
    
    std::cout << "[CONTROLLER] Shutdown complete" << std::endl;
}