#include <framework/Log.hpp>
#include <framework/Checks.hpp>
#include "SygnalPomoMonitor.hpp"
#include <iostream>
#include <iomanip>
#include <dlfcn.h>


SygnalPomoMonitor::SygnalPomoMonitor()
    : m_context(DW_NULL_HANDLE)
    , m_sal(DW_NULL_HANDLE)
    , m_canSensor(DW_NULL_HANDLE)
    , m_initialized(false)
    , m_pluginHandle(nullptr)
    , m_getFunctionTableFn(nullptr)
{
    std::memset(&m_pluginFunctions, 0, sizeof(m_pluginFunctions));
    std::memset(&m_commonFunctions, 0, sizeof(m_commonFunctions));
}


SygnalPomoMonitor::~SygnalPomoMonitor() {
    release();
}


dwStatus SygnalPomoMonitor::getFunctionTable(const std::string& pluginPath){
    //Load plugin library
    m_pluginHandle = dlopen(pluginPath.c_str(), RTLD_NOW);
    if (!m_pluginHandle){
        std::cerr << "[APP] Failed to load plugin: " << dlerror() << std::endl;
        return DW_NOT_AVAILABLE;
    }

    //get fucntion table
    m_getFunctionTableFn = reinterpret_cast<GetFunctionTableFn>(
        dlsym(m_pluginHandle, "dwSensorCANPlugin_getFunctionTable"));
    
    if (!m_getFunctionTableFn){
        std::cerr << "[APP] Failed to get function pointer: " << dlerror() <<  std::endl;
        dlclose(m_pluginHandle);
        m_pluginHandle = nullptr;
        return DW_NOT_AVAILABLE;
    }

    //get function table
    dwStatus status = m_getFunctionTableFn(&m_pluginFunctions);
    if (status != DW_SUCCESS) {
        std::cerr << "[APP] Failed to get plugin function table: " << dwGetStatusName(status) <<std::endl;
        dlclose(m_pluginHandle);
        m_pluginHandle = nullptr;
        return status;;
    } 

    return DW_SUCCESS;

}

bool SygnalPomoMonitor::initialize(const ProgramArguments& arguments,
    dwContextHandle_t context,
    dwSALHandle_t sal) {
m_context = context;
m_sal = sal;

// Validate plugin path
std::string pluginPath = arguments.get("plugin-path");
std::string dbcPath = arguments.get("dbc-path");

if (pluginPath.empty()) {
std::cerr << "[APP] Plugin path not specified" << std::endl;
return false;
}

if (dbcPath.empty()){
    std::cerr << "[APP] DBC Path not specified. use --dbc-path=<opath>" << std::endl;
    return false;
}

// Load plugin functions
CHECK_DW_ERROR_AND_RETURN(getFunctionTable(pluginPath));

// Parse CAN parameters
std::string params = arguments.get("params");
size_t portPos = params.find("port=");
if (portPos == std::string::npos) {
std::cerr << "[APP] CAN port not specified in parameters" << std::endl;
return false;
}

// Extract port name
size_t portEnd = params.find(',', portPos);
std::string portName = params.substr(portPos + 5, 
portEnd == std::string::npos ? std::string::npos : portEnd - (portPos + 5));

// Construct parameter string
dwSensorParams sensorParams{};
std::string paramString = "device=" + portName + params.substr(portEnd) + 
",decoder-path=" + pluginPath + ",dbc-path=" + dbcPath;

std::cout << "[APP] Configuring CAN with parameters: " << paramString << std::endl;

sensorParams.parameters = paramString.c_str();
sensorParams.protocol = arguments.get("driver").c_str();

CHECK_DW_ERROR_AND_RETURN(dwSAL_createSensor(&m_canSensor, sensorParams, m_sal));
CHECK_DW_ERROR_AND_RETURN(dwSensor_start(m_canSensor));

m_initialized = true;
return true;
}


bool SygnalPomoMonitor::pluginProcessMessage(const dwCANMessage& msg){
    if (!m_pluginFunctions.parseDataBuffer){
        std::cerr << "[APP] Plugin parseDataBuffer function not available" <<  std::endl;
        return false;
    }

    dwCANMessage processedMsg = msg;

    dwStatus status = m_pluginFunctions.parseDataBuffer(&processedMsg, reinterpret_cast<dwSensorPluginSensorHandle_t>(m_canSensor));

    if (status != DW_SUCCESS){
        std::cerr << "[APP] Failed to process message with plugin: " << dwGetStatusName(status) << std::endl;
        return false;
    }

    return true;
}

bool SygnalPomoMonitor::processMessages() {
    if (!m_initialized) {
        std::cerr << "[APP] Monitor not initialized" << std::endl;
        return false;
    }

    dwCANMessage msg;
    dwStatus status = dwSensorCAN_readMessage(&msg, DEFAULT_TIMEOUT_US, m_canSensor);

    if (status == DW_SUCCESS) {

        constexpr uint32_t HEARTBEAT_MSG_ID = 0x170;
        constexpr uint32_t CONTROL_MSG_ID = 0x161;
        constexpr uint32_t FAULT_MSG_ID = 0x20;

        if (msg.id == HEARTBEAT_MSG_ID || msg.id == CONTROL_MSG_ID || msg.id == FAULT_MSG_ID) {


        std::cout << "[APP] Received CAN message - ID: 0x"
                  << std::hex << msg.id << " | Data: ";
        for (uint16_t i= 0; i < msg.size; i++){
            std::cout << std::hex << static_cast<int>(msg.data[i]) << " ";
        }
        std::cout << std::dec << std::endl;

        return pluginProcessMessage(msg);

        }
        return false;
    } 
    else if (status == DW_TIME_OUT) {
        return false;
    } else {
        std::cerr << "[APP] Failed to read CAN message. Status: " << dwGetStatusName(status) << std::endl;
        return false;
    }
}

void SygnalPomoMonitor::release() {
    if (m_canSensor != DW_NULL_HANDLE) {
        // Stop sensor first
        dwStatus stopStatus = dwSensor_stop(m_canSensor);
        if (stopStatus != DW_SUCCESS) {
            std::cerr << "[APP] Warning: Failed to stop CAN sensor: " 
                      << dwGetStatusName(stopStatus) << std::endl;
        }

        // Release sensor
        dwStatus releaseStatus = dwSAL_releaseSensor(m_canSensor);
        if (releaseStatus != DW_SUCCESS) {
            std::cerr << "[APP] Warning: Failed to release CAN sensor: " 
                      << dwGetStatusName(releaseStatus) << std::endl;
        }
        
        m_canSensor = DW_NULL_HANDLE;
    }

    if (m_pluginHandle){
        dlclose(m_pluginHandle);
        m_pluginHandle = nullptr;
        m_getFunctionTableFn = nullptr;
    }

    m_initialized = false;
    m_context = DW_NULL_HANDLE;
    m_sal = DW_NULL_HANDLE;
    std::memset(&m_pluginFunctions, 0, sizeof(m_pluginFunctions));
    std::memset(&m_commonFunctions, 0, sizeof(m_commonFunctions));
}
