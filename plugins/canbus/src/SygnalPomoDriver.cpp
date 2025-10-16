#include "SygnalPomoDriver.hpp"
#include "Common.hpp"
#include <dw/sensors/canbus/CAN.h>
#include <dw/sensors/canbus/Interpreter.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/legacy/plugins/canbus/CANPlugin.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <bitset>
#include <iomanip>
#include <vector>
#include <mutex>
#include <thread>
#include <chrono>
#include <map>
#include <sstream>


class BufferPool{
private:
    std::vector<std::vector<uint8_t>> m_pool;
    std::mutex m_mutex;
    const size_t m_bufferSize; 

public:
    explicit BufferPool(size_t buufferSize = 128, size_t poolSize = 10)
        : m_bufferSize(buufferSize) {
            for (size_t i =0; i < poolSize; i++){
                m_pool.emplace_back(buufferSize);
            }
        }
    
    std::vector<uint8_t>* acquire() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_pool.empty()) {
            m_pool.emplace_back(m_bufferSize);
        }
        auto* buffer = &m_pool.back();
        m_pool.pop_back();
        return buffer;
    }

    void release(std::vector<uint8_t>* buffer){
        std::lock_guard<std::mutex> lock(m_mutex);
        m_pool.push_back(std::move(*buffer));
    }

};
namespace {
     // CAN Message IDs
     constexpr uint32_t HEARTBEAT_MSG_ID = 0x170; // 368 in decimal
     constexpr uint32_t CONTROL_MSG_ID = 0x160;   // enable - 352 in decimal
     constexpr uint32_t FAULT_MSG_ID = 0x20;      // 32 in decimal
    
}

namespace sygnalpomo {

const std::map<ControlSubsystem, SubsystemConfig> SUBSYSTEM_CONFIG = {
    {ControlSubsystem::THROTTLE,   {1, 0, 1, true,  ValueType::PERCENTAGE, 0x160}},
    {ControlSubsystem::BRAKE,      {1, 0, 0, true,  ValueType::PERCENTAGE, 0x160}},
    {ControlSubsystem::STEERING,   {1, 0, 2, true,  ValueType::NORMALIZED, 0x160}},
    {ControlSubsystem::GEAR,       {3, 0, 0, true,  ValueType::ENUM,       0x160}},
    {ControlSubsystem::TURN_SIGNAL,{1, 0, 4, false, ValueType::ENUM,       0x165}},
    {ControlSubsystem::HEADLIGHT,  {1, 0, 5, false, ValueType::ENUM,       0x166}},
    {ControlSubsystem::WIPER,      {1, 0, 6, false, ValueType::ENUM,       0x167}}
};

SygnalPomoDriver::SygnalPomoDriver(dwContextHandle_t ctx, dwSALHandle_t sal)
    : m_sal(sal), m_ctx(ctx), m_canSensor(nullptr),
      m_isRunning(false), m_useHwTimestamps(true), m_virtualSensorFlag(false) {}

SygnalPomoDriver::~SygnalPomoDriver() {
    release();
}
//versioning calls 

dwStatus SygnalPomoDriver::getSensorInformation(dwSensorPlugin_information* information) {
    if (!information) {
        std::cerr << "[PLUGIN ERROR] Null information pointer provided" << std::endl;
        return DW_INVALID_ARGUMENT;
    }

    std::cout << "[PLUGIN DEBUG] Getting sensor information" << std::endl;

    // Clear the structure
    std::memset(information, 0, sizeof(dwSensorPlugin_information));

    // Set version information using constants from Common.hpp
    information->firmware.versionMajor = PLUGIN_VERSION_MAJOR;
    information->firmware.versionMinor = PLUGIN_VERSION_MINOR;
    information->firmware.versionPatch = PLUGIN_VERSION_PATCH;
    
    // Handle version string - store in member variable for memory management
    m_versionString = getVersionString();
    information->firmware.versionString = const_cast<char*>(m_versionString.c_str());

    std::cout << "[PLUGIN INFO] Sensor Information:" << std::endl;
    std::cout << "  Version: " << information->firmware.versionMajor 
              << "." << information->firmware.versionMinor 
              << "." << information->firmware.versionPatch << std::endl;
    std::cout << "  Version String: " << information->firmware.versionString << std::endl;

    return DW_SUCCESS;
}

std::string SygnalPomoDriver::getVersionString() const {
    std::ostringstream ss;
    ss << PLUGIN_VERSION_MAJOR << "." 
       << PLUGIN_VERSION_MINOR << "." 
       << PLUGIN_VERSION_PATCH << "-SygnalPomoDriver";
    
    // Add build info if needed
    #ifdef DEBUG
    ss << "-DEBUG";
    #else
    ss << "-RELEASE";
    #endif
    
    return ss.str();
}

void SygnalPomoDriver::setSAL(dwSALHandle_t sal) {
    std::cout << "[DRIVER] Setting SAL handle: " << sal << std::endl;
    m_sal = sal;
}

dwStatus SygnalPomoDriver::initialize(const char* params) {
    if (!params) {
        return DW_INVALID_ARGUMENT;
    }

    if (!m_sal) {
        std::cerr << "[PLUGIN CRITICAL] SAL handle is null during initialize" << std::endl;
        return DW_INVALID_HANDLE;
    }

    //creating a CAN sensor 
    dwSensorParams sensorParams{};
    sensorParams.parameters = params;
    sensorParams.protocol = "can.socket";

    dwStatus status = dwSAL_createSensor(&m_canSensor, sensorParams, m_sal);

    if (status != DW_SUCCESS || m_canSensor == nullptr) {
        std::cerr << "[PLUGIN CRITICAL] Failed to create CAN sensor: " 
                << dwGetStatusName(status) << " (handle = " << m_canSensor << ")" << std::endl;
        return status;
    }

    uint32_t monitoredCANIDs[] = {HEARTBEAT_MSG_ID, 0x60, 0x160, FAULT_MSG_ID, 112, 127};
    uint32_t canIDMask = 0x7FF;
    size_t numIDs = sizeof(monitoredCANIDs) / sizeof(monitoredCANIDs[0]);


    status = dwSensorCAN_setMessageFilter(monitoredCANIDs, &canIDMask, (uint16_t)numIDs, m_canSensor);
    if (status != DW_SUCCESS){
        std::cerr << "[PLUGIN ERROR] Failed to set CAN filter." << std::endl;
        release();
        return status;
    }

    std::cout << "[PLUGIN] CAN filter applied - Monitoring only IDs: ";
    for(uint16_t i=0; i < numIDs; i++){
        std::cout << "0x" << std::hex << monitoredCANIDs[i] << " ";
    }
    std::cout << std::dec << std::endl;

    // Ensure the DBC path is valid
    std::string dbcpath;
    std::string paramsstr = params;
    size_t dbcPos = paramsstr.find("dbc-path=");

    if (dbcPos != std::string::npos) {
        size_t dbcEnd = paramsstr.find(',', dbcPos);
        dbcpath = paramsstr.substr(dbcPos + 9, dbcEnd - (dbcPos + 9));
    }

    if (dbcpath.empty()) {
        std::cerr << "[ERROR] DBC path not provided. Use --dbc-path=<path> argument." << std::endl;
        return DW_FAILURE;
    }

    // Load DBC Configurations
    if (!loadDBCConfig(dbcpath + "/Heartbeat.dbc") ||
        !loadDBCConfig(dbcpath + "/Control.dbc")) {
        std::cerr << "[ERROR] Failed to load DBC configurations from " << dbcpath << std::endl;
        release();
        return DW_FAILURE;
    }

    m_isRunning = true;
    std::cout << "[PLUGIN] Sensor initialized successfully with parameters: " << params << std::endl;
    return DW_SUCCESS;
}

dwStatus SygnalPomoDriver::start() {
    return dwSensor_start(m_canSensor);
}

dwStatus SygnalPomoDriver::stop() {
    return dwSensor_stop(m_canSensor);
}

dwStatus SygnalPomoDriver::release() {
    if (m_canSensor) {
        dwSensor_stop(m_canSensor);
        dwSAL_releaseSensor(m_canSensor);
        m_canSensor = nullptr;
    }
    m_isRunning = false;
    return DW_SUCCESS;
}

dwStatus SygnalPomoDriver::sendMessage(dwCANMessage* msg, dwTime_t timeoutUs) {
    if (!msg) {
        return DW_INVALID_ARGUMENT;
    }
    if (!m_canSensor) {
        std::cerr << "[PLUGIN CRITICAL] Cannot send message: CAN sensor handle is null" << std::endl;
        return DW_INVALID_HANDLE;
    }
    
    if (!m_sal) {
        std::cerr << "[PLUGIN CRITICAL] Cannot send message: SAL handle is null" << std::endl;
        return DW_INVALID_HANDLE;
    }

    std::cout << "[PLUGIN DEBUG] sendMessage called with m_canSensor = " << m_canSensor << ", m_sal = " << m_sal << std::endl;
    std::cout << "[PLUGIN DIAG] Message recieved for transmission. ID: 0x"
              << std::hex << msg->id << std::dec
              << ", size: " << static_cast<int>(msg->size)
              << ", First byte: 0x" << std::hex
              << static_cast<int>(msg->data[0]) << std::dec << std::endl;

    bool isIntentMessage = (msg->id == 0xFFFF);
    if (isIntentMessage) {
        
        
        std::cout << "[PLUGIN DEBUG] Raw intent message data: ";
        for (int i = 0; i < 8; i++) {
            std::cout << std::hex << static_cast<int>(msg->data[i]) << " ";
        }
        std::cout << std::dec << std::endl;


        uint8_t subsystemID = msg->data[0];
        int32_t value;
        std::memcpy(&value, &msg->data[1], sizeof(int32_t));

        float duration = 0.0f;
        std::memcpy(&duration, &msg->data[5], sizeof(float));


        ControlSubsystem subsystem = static_cast<ControlSubsystem>(subsystemID);

        std::cout << "[PLUGIN DEBUG] Parsed intent: subsystem=" << static_cast<int>(subsystem) 
                  << ", value=" << value << ", duration=" << duration << std::endl;

        if (subsystem == ControlSubsystem::GEAR) {
            std::cout << "[PLUGIN] Detected gear change command - executing optimized sequence" << std::endl;
            std::cout << "[PLUGIN] Target gear position: " << value << std::endl;
            
            // Validate gear position
            if (value < 0 || value > 3) {
                std::cerr << "[PLUGIN] Invalid gear position: " << value << std::endl;
                return DW_INVALID_ARGUMENT;
            }
            return executeOptimizedGearSequence(value);
        } else {
            std::cout << "[PLUGIN] Processing " << getSubsystemName(subsystem) 
                      << " command: " << value;
            if (duration > 0.0f) {
                std::cout << " (duration: " << duration << "s)";
            }
            std::cout << std::endl;
            
            return sendUnifiedCommand(subsystem, value, duration);
        }
    }

    msg->data[7] = calculateStandardCRC8(msg->data, 7);

    std::cout << "[PLUGIN DIAG] Post-encoding message: ID: 0x"
              << std::hex << msg->id << std::dec 
              << ", Data: ";
    
    for (int i = 0; i < 8; i++){
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << static_cast<int>(msg->data[i]) << " ";
    }

    std::cout << std::dec << std::endl;

    return dwSensorCAN_sendMessage(msg, timeoutUs, m_canSensor);

}

dwStatus SygnalPomoDriver::readMessage(dwCANMessage* msg, dwTime_t timeoutUs) {
    if (!msg) {
        return DW_INVALID_ARGUMENT;
    }

    dwStatus status = dwSensorCAN_readMessage(msg, timeoutUs, m_canSensor);
    if (status != DW_SUCCESS) {
        std::cerr << "Failed to read CAN message. Status: " << status << std::endl;
    }
    return status;
}

dwStatus SygnalPomoDriver::setMessageFilter(const uint32_t* ids, const uint32_t* masks, uint16_t numCanIDs) {
    if(!ids || numCanIDs == 0){
        std::cerr << "[PLUGIN ERROR] Invalid filter parameters" << std::endl;
        return DW_INVALID_ARGUMENT;
    }
    return dwSensorCAN_setMessageFilter(ids, masks, numCanIDs, m_canSensor);
}

dwStatus SygnalPomoDriver::setUseHwTimestamps(bool flag) {
    m_useHwTimestamps = flag;
    return dwSensorCAN_setUseHwTimestamps(flag, m_canSensor);
}

dwStatus SygnalPomoDriver::sendUnifiedCommand(ControlSubsystem subsystem, int32_t value, float duration) {
    // Locate configuration
    const SubsystemConfig& config = SUBSYSTEM_CONFIG.at(subsystem);
    
    // Send enable message if required
    if (config.requiresEnable) {
        dwStatus enableStatus = sendEnableMessage(config.busAddress, config.interfaceID);
        if (enableStatus != DW_SUCCESS) {
            return enableStatus;
        }
    }
    
    // Create control message
    dwCANMessage msg;
    std::memset(&msg, 0, sizeof(dwCANMessage));
    msg.id = config.messageId;
    msg.size = 8;
    
    // Pack signals
    std::map<std::string, int32_t> signals = {
        {"BusAddress", config.busAddress},
        {"SubsystemID", config.subsystemID},
        {"InterfaceID", config.interfaceID},
        {"Count8", controlCounter++},
        {"Value", convertValue(subsystem, value)}
    };
    packCANSignals(msg.data, signals);
    
    // Calculate CRC
    msg.data[7] = calculateStandardCRC8(msg.data, 7);
    
    // Send message
    dwStatus status = dwSensorCAN_sendMessage(&msg, DEFAULT_CAN_TIMEOUT_US, m_canSensor);
    
    std::cout << "[PLUGIN DEBUG] sendUnifiedCommand - subsystem: " << static_cast<int>(subsystem)
          << ", status: " << dwGetStatusName(status)
          << ", duration: " << duration << std::endl;
    
    // Schedule release if needed (for brake)
    if (status == DW_SUCCESS && subsystem == ControlSubsystem::BRAKE && duration > 0.0f) {
        std::cout << "[PLUGIN DEBUG] Scheduling release" << std::endl;
        scheduleReleaseCommand(config.messageId, duration);
    }
    return status;
}



dwStatus SygnalPomoDriver::sendEnableMessage(uint8_t busAddress, uint8_t interfaceID) {
    dwCANMessage msg;
    std::memset(&msg, 0, sizeof(dwCANMessage));
    msg.id = 0x60; // ControlEnable ID (96 decimal)
    msg.size = 8;
    
    // Pack signals according to DBC
    std::map<std::string, int32_t> signals = {
        {"BusAddress", busAddress},
        {"SubsystemID", 0}, // Subsystem ID is not used in enable message
        {"InterfaceID", interfaceID},
        {"Enable", 1}
    };
    packCANSignals(msg.data, signals);
    
    // Calculate CRC
    msg.data[7] = calculateStandardCRC8(msg.data, 7);
    
    return dwSensorCAN_sendMessage(&msg, DEFAULT_CAN_TIMEOUT_US, m_canSensor);
}

bool SygnalPomoDriver::loadDBCConfig(const std::string& dbcFilePath) {
    // Check if file exists
    std::ifstream dbcFile(dbcFilePath);
    if (!dbcFile.is_open()) {
        std::cerr << "[ERROR] DBC file not found: " << dbcFilePath << std::endl;
        return false;
    }

    std::cout << "[INFO] DBC file loaded successfully: " << dbcFilePath << std::endl;
    return true;
}


//Sygnal CAN signal packing functions
void SygnalPomoDriver::packCANSignals(uint8_t* data, const std::map<std::string, int32_t>& signals) {
    std::memset(data, 0, 8);
    

    //byte 0
    // BusAddress: 0|7@1+ (bits 0-6 of byte 0)
    if (signals.count("BusAddress")) {
        data[0] |= (signals.at("BusAddress") & 0x7F);
    }

    if (signals.count("SubsystemID")){
        data[0] |= ((signals.at("SubsystemID") & 0x01) << 6); // bits 6-7 of byte 0
    }
    

    //byte 1
    // InterfaceID: 13|3@1+ (bits 5-7 of byte 1)
    if (signals.count("InterfaceID")) {
        data[1] |= ((signals.at("InterfaceID") & 0x07) << 5);
    }


    //byte 2
    
    // Count8: 16|8@1+ (entire byte 2)
    if (signals.count("Count8")) {
        data[2] = signals.at("Count8") & 0xFF;
    }
    if(signals.count("Enable")){
        data[2] |= (signals.at("Enable") & 0x01);
    }
    
    // Value: 24|32@1- (bytes 3-6)
    if (signals.count("Value")) {
        int32_t value = signals.at("Value");
        data[3] = (value >> 0) & 0xFF;
        data[4] = (value >> 8) & 0xFF;
        data[5] = (value >> 16) & 0xFF;
        data[6] = (value >> 24) & 0xFF;
    }
}

int32_t SygnalPomoDriver::convertValue(ControlSubsystem subsystem, int32_t rawValue) {
    switch (subsystem) {
        case ControlSubsystem::BRAKE:           
        case ControlSubsystem::THROTTLE:        
            // Convert percentage to direct float value (0-100 to 0.0-1.0)
            {
                float percentage = static_cast<float>(rawValue) / 100.0f;
                uint32_t floatBits;
                std::memcpy(&floatBits, &percentage, sizeof(floatBits));
                return floatBits;
            }
            
        case ControlSubsystem::STEERING:       
            // Convert percentage to normalized float (-100-100 to -1.0-1.0)
            {
                float normalized = static_cast<float>(rawValue) / 100.0f;
                uint32_t floatBits;
                std::memcpy(&floatBits, &normalized, sizeof(floatBits));
                return floatBits;
            }
            
        case ControlSubsystem::GEAR:            
            // Convert gear position to float values (matches Python exactly)
            {
            static const float gearFloatValues[4] = { 0.0f, 56.0f, 48.0f, 40.0f };
            if (rawValue >= 0 && rawValue < 4) {
                float gearFloat = gearFloatValues[rawValue];
                uint32_t floatBits;
                std::memcpy(&floatBits, &gearFloat, sizeof(floatBits));
                std::cout << "[PLUGIN] Gear " << rawValue << " -> " << gearFloat 
                        << " -> 0x" << std::hex << floatBits << std::dec << std::endl;
                return static_cast<int32_t>(floatBits);
            }
                return 0;
            }

        default:
            return rawValue;
    }
}


void SygnalPomoDriver::decodeMessageWithDBC(const dwCANMessage* msg) {

    if (!msg) {
        std::cerr << "[PLUGIN ERROR] recieved NULL CAN message, possible memory corrupt" << std::endl;
        return;
    }

    if (msg->id != HEARTBEAT_MSG_ID && msg->id != CONTROL_MSG_ID && msg->id != FAULT_MSG_ID) {
        std::cout << "[PLUGIN] Ingorning CAN message ID: 0x" << std::hex << msg->id << " (not in filter list)" << std::dec << std::endl;
        return;
    }

    // Create buffer for processed data
    std::cout << "[PLUGIN] decoding with DBC definitions called for ID: 0x" << std::hex << msg->id << " | Size: " << std::dec << msg->size << " | Data: ";
    for (uint16_t i = 0; i < msg->size; i++){
        std::cout << std::hex << static_cast<int>(msg->data[i]) << " ";
    }

    std::cout << std::dec << std::endl;

    // Process message based on ID
    switch(msg->id) {
        case HEARTBEAT_MSG_ID:
            handleHeartbeatMessage(msg, msg->data);
            break;
        case CONTROL_MSG_ID:
            handleControlMessage(msg, msg->data);
            break;
        case FAULT_MSG_ID:
            handleFaultMessage(msg, msg->data);
            break;
        default:
            break;
    }
}



void SygnalPomoDriver::scheduleReleaseCommand(uint32_t messageId, float duration_sec) {
    // Prepare the message that will be sent after the delay
    const auto& cfg = SUBSYSTEM_CONFIG.at(ControlSubsystem::BRAKE);  

    dwCANMessage releaseMsg;
    std::memset(&releaseMsg, 0, sizeof(dwCANMessage));
    releaseMsg.id = messageId;
    releaseMsg.size = 8;

    // Pack the signals for the release command (zero value)
    std::map<std::string, int32_t> signals = {
        {"BusAddress", cfg.busAddress},         
        {"SubsystemID", cfg.subsystemID},       
        {"InterfaceID", cfg.interfaceID},       
        {"Count8", controlCounter++},
        {"Value", 0}                           // Release value (zero)
    };
    packCANSignals(releaseMsg.data, signals);
    
    // Add CRC
    releaseMsg.data[7] = calculateStandardCRC8(releaseMsg.data, 7);
    
    // Schedule the task to execute after the duration
    m_taskManager.scheduleTask([this, releaseMsg]() mutable {
        std::cout << "[PLUGIN] Sending brake release command" << std::endl;
        dwSensorCAN_sendMessage(&releaseMsg, DEFAULT_CAN_TIMEOUT_US, m_canSensor);
        std::cout << "[PLUGIN] Brake release command sent" << std::endl;
    }, static_cast<dwTime_t>(duration_sec * 1000000.0f));  // Convert to microseconds
    
    std::cout << "[PLUGIN] Scheduled brake release in " << duration_sec << " seconds" << std::endl;
}



uint8_t SygnalPomoDriver::calculateStandardCRC8(const uint8_t* data, size_t length) {
    uint8_t crc = 0x00;
    for (size_t i = 0; i < length; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 0x80)
                crc = (crc << 1) ^ 0x07;
            else
                crc <<= 1;
        }
    }
    return crc;
}


std::string SygnalPomoDriver::getDBCValue(const std::string& fieldName, uint8_t value) {
    static const std::map<std::string, std::map<uint8_t, std::string>> dbcMappings = {
        // System state values from Heartbeat.dbc
        {"SystemState", {
            {0x00, "HUMAN_CONTROL"},
            {0x01, "MCM_CONTROL"},
            {0xF1, "FAIL_OP_1"},         // 241 decimal
            {0xF2, "FAIL_OP_2"},         // 242 decimal
            {0xFD, "HUMAN_OVERRIDE"},    // 253 decimal
            {0xFE, "FAIL_HARD"}          // 254 decimal
        }},
        
        // Interface state values from Heartbeat.dbc
        {"InterfaceState", {
            {0, "HUMAN_CONTROL"},
            {1, "MCM_CONTROL"}
        }},
        
        // Per-interface state mappings for all interfaces (0-6)
        {"Interface0State", {
            {0, "HUMAN_CONTROL"},
            {1, "MCM_CONTROL"}
        }},
        {"Interface1State", {
            {0, "HUMAN_CONTROL"},
            {1, "MCM_CONTROL"}
        }},
        {"Interface2State", {
            {0, "HUMAN_CONTROL"},
            {1, "MCM_CONTROL"}
        }},
        {"Interface3State", {
            {0, "HUMAN_CONTROL"},
            {1, "MCM_CONTROL"}
        }},
        {"Interface4State", {
            {0, "HUMAN_CONTROL"},
            {1, "MCM_CONTROL"}
        }},
        {"Interface5State", {
            {0, "HUMAN_CONTROL"},
            {1, "MCM_CONTROL"}
        }},
        {"Interface6State", {
            {0, "HUMAN_CONTROL"},
            {1, "MCM_CONTROL"}
        }},
        {"OverallInterfaceState", {
            {0, "HUMAN_CONTROL"},
            {1, "MCM_CONTROL"}
        }},
        
        // Enable/disable state for control messages
        {"Enable", {
            {0, "DISABLED"},
            {1, "ENABLED"}
        }},
        
        // Override state values
        {"OverrideState", {
            {0, "NO_OVERRIDE"},
            {1, "OVERRIDE_ACTIVE"}
        }},
        
        // Fault state values from Fault.dbc
        {"FaultState", {
            {0x01, "EMERGENCY_STOP"},
            {0x02, "EXTERNAL_FAULT"},
            {0x10, "INVALID_SUBSYSTEM_ID"},
            {0x11, "INVALID_CONFIGURATION_CRC"},
            {0x20, "INVALID_INPUT_SIGNAL"},
            {0x21, "INVALID_OUTPUT_SIGNAL"}
        }},
        
        // Subsystem identifiers
        {"SubsystemID", {
            {0, "SYSTEM0"},
            {1, "SYSTEM1"}
        }},
        
        // Interface identifiers for subsystems
        {"InterfaceID", {
            {0, "BRAKE"},
            {1, "ACCELERATOR"},
            {2, "STEERING"}
        }}
    };

    auto it = dbcMappings.find(fieldName);
    if (it != dbcMappings.end() && it->second.count(value)) {
        return it->second.at(value);
    }
    return "UNKNOWN";
}

void SygnalPomoDriver::handleHeartbeatMessage(const dwCANMessage* msg, const uint8_t* processedData) {
    if (!msg || !processedData || msg->size != 8) {
        std::cerr << "[PLUGIN] Invalid message parameters" << std::endl;
        return;
    }
    
    uint8_t dataCopy[8];
    std::memcpy(dataCopy, processedData, msg->size);

    // Extract values according to DBC bit field specification
    uint8_t busAddress = dataCopy[0] & 0x7F;                 // Bits 0-6
    uint8_t subsystemID = (dataCopy[0] >> 7) & 0x01;         // Bit 7 
    uint8_t interfaceID = (dataCopy[1] >> 5) & 0x07;         // Bits 13-15 (5-7 of byte 1)
    uint8_t systemState = dataCopy[2];                       // Bits 16-23 (entire byte 2)
    
    // Interface state bits in byte 3 (starting at bit 24)
    uint8_t interface0State = (dataCopy[3] >> 0) & 0x01;     // Bit 24
    uint8_t interface1State = (dataCopy[3] >> 1) & 0x01;     // Bit 25
    
    // Count16 in bytes 5-6
    uint16_t count16 = (static_cast<uint16_t>(dataCopy[5]) << 8) | dataCopy[4];  // Little-endian format

    // Fetch human-readable meanings dynamically from DBC
    std::string systemStateStr = this->getDBCValue("SystemState", systemState);
    std::string interface0StateStr = this->getDBCValue("InterfaceState", interface0State);
    std::string interface1StateStr = this->getDBCValue("InterfaceState", interface1State);

    // Print decoded values with DBC mappings
    std::cout << "Heartbeat Decoded:\n"
              << "  BusAddress: " << static_cast<int>(busAddress) << "\n"
              << "  SubsystemID: " << static_cast<int>(subsystemID) << "\n"
              << "  InterfaceID: " << static_cast<int>(interfaceID) << "\n"
              << "  System State: 0x" << std::hex << static_cast<int>(systemState) 
              << " (" << systemStateStr << ")\n"
              << "  Interface0: " << std::dec << static_cast<int>(interface0State) 
              << " (" << interface0StateStr << ")\n"
              << "  Interface1: " << static_cast<int>(interface1State) 
              << " (" << interface1StateStr << ")\n"
              << "  Count16: " << count16 << std::endl;

    // Store values in the plugin state
    m_state.heartbeat.systemState = systemState;
    m_state.heartbeat.busAddress = busAddress;
    m_state.heartbeat.subsystemID = subsystemID;
    m_state.heartbeat.interface0State = interface0State;
    m_state.heartbeat.interface1State = interface1State;
    m_state.heartbeat.count16 = count16;
    m_state.heartbeat.timestamp_us = msg->timestamp_us;
    m_state.heartbeat.newData = true;
}

void SygnalPomoDriver::handleControlMessage(const dwCANMessage* msg, const uint8_t* processedData) {
    if (!msg || !processedData || msg->size != 8) {
        std::cerr << "[PLUGIN] Invalid control message parameters" << std::endl;
        return;
    }

    uint8_t dataCopy[8];
    std::memcpy(dataCopy, processedData, msg->size);

    // Extract fields according to DBC bit field specification
    uint8_t busAddress = dataCopy[0] & 0x7F;                 // Bits 0-6
    uint8_t subsystemID = (dataCopy[0] >> 7) & 0x01;         // Bit 7
    uint8_t interfaceID = (dataCopy[1] >> 5) & 0x07;         // Bits 13-15 (5-7 of byte 1)
    uint8_t count8 = dataCopy[2];                            // Bits 16-23 (entire byte 2)
    
    // Value is 32-bit integer starting at bit 24 (little-endian in bytes)
    int32_t value;
    std::memcpy(&value, &dataCopy[3], sizeof(int32_t));
    
    // Apply endianness correction if needed
    if (msg->id == CONTROL_MSG_ID) {
        value = __builtin_bswap32(value);
    }

    // Store in state
    m_state.control.busAddress = busAddress;
    m_state.control.subsystemID = subsystemID;
    m_state.control.interfaceID = interfaceID;
    m_state.control.count8 = count8;
    m_state.control.value = value;
    m_state.control.timestamp_us = msg->timestamp_us;
    m_state.control.newData = true;

    std::cout << "Control Message Decoded:\n"
              << "  BusAddress: " << static_cast<int>(busAddress) << "\n"
              << "  SubsystemID: " << static_cast<int>(subsystemID) << "\n" 
              << "  InterfaceID: " << static_cast<int>(interfaceID) << "\n"
              << "  Count8: " << static_cast<int>(count8) << "\n"
              << "  Value: " << value << std::endl;
}

void SygnalPomoDriver::handleFaultMessage(const dwCANMessage* msg, const uint8_t* processedData) {
    if (!msg || !processedData || msg->size != 8) {
        std::cerr << "[PLUGIN] Invalid fault message parameters" << std::endl;
        return;
    }

    uint8_t dataCopy[8];
    std::memcpy(dataCopy, processedData, msg->size);

    // Extract fields according to DBC bit field specification
    uint8_t busAddress = dataCopy[0] & 0x7F;                 // Bits 0-6
    uint8_t subsystemID = (dataCopy[0] >> 7) & 0x01;         // Bit 7
    uint8_t faultState = dataCopy[2];                        // Bits 16-23 (Count8 position)
    uint8_t faultCause = dataCopy[3];                        // Bits 24-31 (Value[0] position)
    
    // FaultCount in bytes 4-5 (little-endian)
    uint16_t faultCount = (static_cast<uint16_t>(dataCopy[5]) << 8) | dataCopy[4];

    // Update state
    m_state.fault.busAddress = busAddress;
    m_state.fault.subsystemID = subsystemID;
    m_state.fault.faultState = faultState;
    m_state.fault.faultCause = faultCause;
    m_state.fault.faultCount = faultCount;
    m_state.fault.timestamp_us = msg->timestamp_us;
    m_state.fault.newData = true;

    std::cout << "Fault Message Decoded:\n"
              << "  BusAddress: " << static_cast<int>(busAddress) << "\n"
              << "  SubsystemID: " << static_cast<int>(subsystemID) << "\n"
              << "  State: 0x" << std::hex << static_cast<int>(faultState) << "\n"
              << "  Cause: 0x" << static_cast<int>(faultCause) << "\n"
              << "  Count: " << std::dec << faultCount << std::endl;
}


uint32_t SygnalPomoDriver::getNumAvailableSignals() const {
    uint32_t count = 0;
    if (m_state.heartbeat.newData) count++;
    if (m_state.control.newData) count++;
    if (m_state.fault.newData) count++;
    return count;
}

bool SygnalPomoDriver::getSignalValuef32(float32_t* value, dwTime_t* timestamp, uint32_t idx) const {
    if (idx >= getNumAvailableSignals()) return false;

    uint32_t currentIdx = 0;
    
    if (m_state.heartbeat.newData && currentIdx++ == idx) {
        // Convert system state to float if needed
        *value = static_cast<float32_t>(m_state.heartbeat.systemState);
        *timestamp = m_state.heartbeat.timestamp_us;
        return true;
    }
    
    if (m_state.control.newData && currentIdx++ == idx) {
        *value = static_cast<float32_t>(m_state.control.value);
        *timestamp = m_state.control.timestamp_us;
        return true;
    }

    if (m_state.fault.newData && currentIdx++ == idx) {
        *value = static_cast<float32_t>(m_state.fault.faultState);
        *timestamp = m_state.fault.timestamp_us;
        return true;
    }

    return false;
}

bool SygnalPomoDriver::getSignalValuei32(int32_t* value, dwTime_t* timestamp, uint32_t idx) const {
    if (idx >= getNumAvailableSignals()) return false;

    uint32_t currentIdx = 0;
    
    if (m_state.heartbeat.newData && currentIdx++ == idx) {
        *value = static_cast<int32_t>(m_state.heartbeat.systemState);
        *timestamp = m_state.heartbeat.timestamp_us;
        return true;
    }
    
    if (m_state.control.newData && currentIdx++ == idx) {
        *value = m_state.control.value;
        *timestamp = m_state.control.timestamp_us;
        return true;
    }

    if (m_state.fault.newData && currentIdx++ == idx) {
        *value = static_cast<int32_t>(m_state.fault.faultState);
        *timestamp = m_state.fault.timestamp_us;
        return true;
    }

    return false;
}



dwStatus SygnalPomoDriver::executeOptimizedGearSequence(int32_t gearValue) {
    std::cout << "[PLUGIN] Executing OPTIMIZED gear change sequence to position " << gearValue << std::endl;
    
    // Use optimized parameters from successful Python testing
    constexpr int32_t OPTIMAL_BRAKE_PERCENTAGE = 40;        // 40% brake
    constexpr float OPTIMAL_BRAKE_DURATION = 2.0f;          // 2.0 seconds  
    constexpr float OPTIMAL_GEAR_INTERVAL = 0.5f;           // 0.5 second intervals
    constexpr auto OPTIMAL_INTERVAL_MS = 500;               // 500ms between commands
    
    std::cout << "[PLUGIN] Using optimized parameters:" << std::endl;
    std::cout << "    Brake: " << OPTIMAL_BRAKE_PERCENTAGE << "% for " << OPTIMAL_BRAKE_DURATION << "s" << std::endl;
    std::cout << "    Gear interval: " << OPTIMAL_GEAR_INTERVAL << "s (" << OPTIMAL_INTERVAL_MS << "ms)" << std::endl;
    
    // Calculate number of gear commands (should be 4)
    const int numGearCommands = int(OPTIMAL_BRAKE_DURATION / OPTIMAL_GEAR_INTERVAL);
    std::cout << "    Expected gear commands: " << numGearCommands << std::endl;
    
    // Validate gear position
    if (gearValue < 0 || gearValue > 3) {
        std::cerr << "[PLUGIN] Invalid gear position: " << gearValue << std::endl;
        return DW_INVALID_ARGUMENT;
    }
    
    // Get gear configuration - ✅ FIXED: Added ControlSubsystem:: 
    const auto& gearConfig = SUBSYSTEM_CONFIG.at(ControlSubsystem::GEAR);
    const auto& brakeConfig = SUBSYSTEM_CONFIG.at(ControlSubsystem::BRAKE);
    
    // Phase 1: Send brake enable message
    std::cout << "[PLUGIN] Phase 1: Enabling brake control" << std::endl;
    dwStatus brakeEnableStatus = sendEnableMessage(brakeConfig.busAddress, brakeConfig.interfaceID);
    if (brakeEnableStatus != DW_SUCCESS) {
        std::cerr << "[PLUGIN] Failed to enable brake control" << std::endl;
        return brakeEnableStatus;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Python timing
    
    // Phase 2: Send gear enable message  
    std::cout << "[PLUGIN] Phase 2: Enabling gear control" << std::endl;
    dwStatus gearEnableStatus = sendEnableMessage(gearConfig.busAddress, gearConfig.interfaceID);
    if (gearEnableStatus != DW_SUCCESS) {
        std::cerr << "[PLUGIN] Failed to enable gear control" << std::endl;
        return gearEnableStatus;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Python timing
    
    // Phase 3: Apply brake at 40% - ✅ FIXED: Added ControlSubsystem::
    std::cout << "[PLUGIN] Phase 3: Applying " << OPTIMAL_BRAKE_PERCENTAGE << "% brake" << std::endl;
    dwStatus brakeApplyStatus = sendUnifiedCommand(ControlSubsystem::BRAKE, OPTIMAL_BRAKE_PERCENTAGE, OPTIMAL_BRAKE_DURATION);
    if (brakeApplyStatus != DW_SUCCESS) {
        std::cerr << "[PLUGIN] Failed to apply brakes" << std::endl;
        return brakeApplyStatus;
    }
    
    // Phase 4: Send gear commands repeatedly with optimal timing
    std::cout << "[PLUGIN] Phase 4: Sending " << numGearCommands << " gear commands" << std::endl;
    
    // Map gear positions to float values (matches Python exactly)
    static const float gearFloatValues[4] = { 0.0f, 56.0f, 48.0f, 40.0f };
    float gearFloat = gearFloatValues[gearValue];
    
    // Convert float to int32_t for transmission (matches Python struct.pack)
    uint32_t gearFloatBits;
    std::memcpy(&gearFloatBits, &gearFloat, sizeof(gearFloatBits));
    int32_t gearCommand = static_cast<int32_t>(gearFloatBits);
    
    std::cout << "[PLUGIN] Gear " << gearValue << " -> float " << gearFloat 
              << " -> int32 0x" << std::hex << gearCommand << std::dec << std::endl;
    
    auto startTime = std::chrono::steady_clock::now();
    
    for (int i = 0; i < numGearCommands; ++i) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - startTime).count();
            
        std::cout << "[PLUGIN] Gear command " << (i+1) << "/" << numGearCommands 
                  << " at t=" << (elapsed/1000.0f) << "s" << std::endl;
        
        // Create gear command message
        dwCANMessage gearMsg;
        std::memset(&gearMsg, 0, sizeof(dwCANMessage));
        gearMsg.id = gearConfig.messageId;  // 0x160
        gearMsg.size = 8;
        
        // Pack signals for gear command
        std::map<std::string, int32_t> gearSignals = {
            {"BusAddress", gearConfig.busAddress},    // 3 (CB)
            {"SubsystemID", gearConfig.subsystemID},  // 0
            {"InterfaceID", gearConfig.interfaceID},  // 0  
            {"Count8", controlCounter++},
            {"Value", gearCommand}  // Float bits
        };
        packCANSignals(gearMsg.data, gearSignals);
        
        // Calculate CRC
        gearMsg.data[7] = calculateStandardCRC8(gearMsg.data, 7);
        
        // Send gear message
        dwStatus gearStatus = dwSensorCAN_sendMessage(&gearMsg, DEFAULT_CAN_TIMEOUT_US, m_canSensor);
        if (gearStatus != DW_SUCCESS) {
            std::cerr << "[PLUGIN] Failed to send gear command " << (i+1) << ": " 
                      << dwGetStatusName(gearStatus) << std::endl;
            return gearStatus;
        }
        
        // Sleep for optimal interval (except last command)
        if (i < numGearCommands - 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(OPTIMAL_INTERVAL_MS));
        }
    }
    
    auto totalElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startTime).count();
    std::cout << "[PLUGIN] Completed " << numGearCommands 
              << " gear commands in " << (totalElapsed/1000.0f) << " seconds" << std::endl;
    
    std::cout << "[PLUGIN] Gear change sequence completed successfully" << std::endl;
    // Note: Brake release is handled automatically by the scheduled task
    
    return DW_SUCCESS;
}




} // namespace sygnalpomo
