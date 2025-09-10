#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/canbus/CAN.h>
#include <dw/sensors/canbus/Interpreter.h>
#include <dw/sensors/legacy/plugins/canbus/CANPlugin.h>

#include "Common.hpp"
#include "SygnalPomoDriver.hpp"

namespace {

// Global driver registry
std::vector<std::unique_ptr<sygnalpomo::SygnalPomoDriver>> g_drivers;

// Validate sensor handle
bool isValidSensor(dwSensorPluginSensorHandle_t handle) {
    if (!handle) {
        std::cerr << "[PLUGIN ERROR] Null handle provided for validation" << std::endl;
        return false;
    }
    
    auto* driver = static_cast<sygnalpomo::SygnalPomoDriver*>(handle);
    bool valid = std::any_of(g_drivers.begin(), g_drivers.end(),
                       [driver](const auto& d) { return d.get() == driver; });
                       
    if (!valid) {
        std::cerr << "[PLUGIN ERROR] Invalid driver handle: " << handle 
                  << " not found in registry (size=" << g_drivers.size() << ")" << std::endl;
    }
    return valid;
}

// Create Handle
dwStatus _dwSensorPlugin_createHandle(void** handle,
                                      dwSensorPluginProperties* properties,
                                      const char* params,
                                      dwContextHandle_t ctx) {
    (void)params;
    if (!handle || !properties || !ctx) {
        return DW_INVALID_ARGUMENT;
    }
    std::cout << "[PLUGIN DEBUG] Creating driver instance with context: " << ctx << std::endl;

    // Create the driver instance with SAL and Context
    auto driver = std::make_unique<sygnalpomo::SygnalPomoDriver>(ctx, nullptr);
    sygnalpomo::SygnalPomoDriver* driverPtr = driver.get();
    //set properties
    properties->packetSize = 8; // standard can frame size
    properties->rawToDec = DW_SENSORS_RAW_DEC_ONE_TO_ONE;
    
    

    g_drivers.push_back(std::move(driver)); //// Add driver to global registry and return pointer
    *handle = driverPtr; // Return pointer to the now-managed instance
    
    std::cout << "[PLUGIN DEBUG] Created driver handle: " << driverPtr << std::endl;
    
    return DW_SUCCESS;
}

// createSensor
dwStatus _dwSensorPlugin_createSensor(const char* params,
    dwSALObject* salObject,
    void* handle) {
    if (!params || !salObject || !handle) {
        std::cerr << "[ERROR] _dwSensorPlugin_createSensor received invalid arguments." << std::endl;
        return DW_INVALID_ARGUMENT;
    }

    auto* driver = static_cast<sygnalpomo::SygnalPomoDriver*>(handle);
    dwSALHandle_t sal = reinterpret_cast<dwSALHandle_t>(salObject);

    std::cout << "[PLUGIN DEBUG] Setting SAL handle: " << sal << " for driver: " << handle << std::endl;
    
    driver->setSAL(sal);
    dwStatus status = driver->initialize(params);

    std::cout << "[PLUGIN DEBUG] Initialize result: " << dwGetStatusName(status) << std::endl;

    return status;
}


// Start Sensor
dwStatus _dwSensorPlugin_start(dwSensorPluginSensorHandle_t handle) {
    auto* driver = static_cast<sygnalpomo::SygnalPomoDriver*>(handle);
    return driver ? driver->start() : DW_INVALID_HANDLE;
}



// Stop Sensor
dwStatus _dwSensorPlugin_stop(dwSensorPluginSensorHandle_t handle) {
    if (!isValidSensor(handle)) {
        return DW_INVALID_HANDLE;
    }
    return static_cast<sygnalpomo::SygnalPomoDriver*>(handle)->stop();
}

// Reset Sensor
dwStatus _dwSensorPlugin_reset(dwSensorPluginSensorHandle_t handle) {
    if (!isValidSensor(handle)) {
        return DW_INVALID_HANDLE;
    }
    return static_cast<sygnalpomo::SygnalPomoDriver*>(handle)->release();
}

// Release Sensor
dwStatus _dwSensorPlugin_release(dwSensorPluginSensorHandle_t handle) {
    if (!isValidSensor(handle)) {
        return DW_INVALID_HANDLE;
    }

    auto* driver = static_cast<sygnalpomo::SygnalPomoDriver*>(handle);
    driver->release();
    g_drivers.erase(std::remove_if(g_drivers.begin(), g_drivers.end(),
                                   [driver](const auto& d) { return d.get() == driver; }),
                    g_drivers.end());

    return DW_SUCCESS;
}

// Send CAN Message
dwStatus _dwSensorCANPlugin_send(const dwCANMessage* msg, dwTime_t timeoutUs, 
    dwSensorPluginSensorHandle_t handle) {
    if (!msg) {
        std::cerr << "[PLUGIN ERROR] Null message provided to send" << std::endl;
        return DW_INVALID_ARGUMENT;
    }

    if (!handle) {
        std::cerr << "[PLUGIN ERROR] Null handle provided to send" << std::endl;
        return DW_INVALID_HANDLE;
    }

    if (!isValidSensor(handle)) {
        std::cerr << "[PLUGIN ERROR] Invalid driver handle in send: " << handle << std::endl;
        return DW_INVALID_HANDLE;
    }

    std::cout << "[PLUGIN DEBUG] Sending via valid driver: " << handle << std::endl;
    return static_cast<sygnalpomo::SygnalPomoDriver*>(handle)->sendMessage(
        const_cast<dwCANMessage*>(msg), timeoutUs);
}

dwStatus _dwSensorCANPlugin_parseDataBuffer(dwCANMessage* output, dwSensorPluginSensorHandle_t sensor){
    if (!output || !sensor) {
        return DW_INVALID_ARGUMENT;
    }

    auto* driver = static_cast<sygnalpomo::SygnalPomoDriver*>(sensor);

    std::cout << "[PLUGIN] Parsing CAN Message in Plugin: ID = 0x"
              << std::hex << output->id << std::dec << std::endl;

    driver->decodeMessageWithDBC(output);

    return DW_SUCCESS;
}
// Set CAN Filter
dwStatus _dwSensorPlugin_setFilter(const uint32_t* ids, const uint32_t* masks, uint16_t numCanIDs, dwSensorPluginSensorHandle_t handle) {
    if (!isValidSensor(handle)) {
        return DW_INVALID_HANDLE;
    }
    return static_cast<sygnalpomo::SygnalPomoDriver*>(handle)->setMessageFilter(ids, masks, numCanIDs);
}

// Clear CAN Filter
dwStatus _dwSensorPlugin_clearFilter(dwSensorPluginSensorHandle_t handle) {
    if (!isValidSensor(handle)) {
        return DW_INVALID_HANDLE;
    }
    return static_cast<sygnalpomo::SygnalPomoDriver*>(handle)->setMessageFilter(nullptr, nullptr, 0);
}

// Enable/Disable Hardware Timestamps
dwStatus _dwSensorPlugin_setUseHwTimestamps(bool flag, dwSensorPluginSensorHandle_t handle) {
    if (!isValidSensor(handle)) {
        return DW_INVALID_HANDLE;
    }
    return static_cast<sygnalpomo::SygnalPomoDriver*>(handle)->setUseHwTimestamps(flag);
}

dwStatus _dwSensorPlugin_readRawData(uint8_t const** data, 
                                    size_t* size,
                                    dwTime_t* timestamp,
                                    dwTime_t timeout_us, 
                                    dwSensorPluginSensorHandle_t handle) {
    if (!data || !size || !timestamp || !handle) {
        return DW_INVALID_ARGUMENT;
    }

    auto* driver = static_cast<sygnalpomo::SygnalPomoDriver*>(handle);
    
    // Read CAN message using existing mechanism
    dwCANMessage msg{};
    dwStatus status = driver->readMessage(&msg, timeout_us);
    
    if (status != DW_SUCCESS) {
        return status;
    }

    // Format buffer according to DriveWorks specification
    // [PayloadSize(4)] [Timestamp(8)] [Payload]
    size_t totalSize = sizeof(uint32_t) + sizeof(dwTime_t) + msg.size;
    uint8_t* buffer = new uint8_t[totalSize];
    
    // Write payload size
    uint32_t payloadSize = msg.size;
    std::memcpy(buffer, &payloadSize, sizeof(uint32_t));
    
    // Write timestamp
    std::memcpy(buffer + sizeof(uint32_t), &msg.timestamp_us, sizeof(dwTime_t));
    
    // Write payload
    std::memcpy(buffer + sizeof(uint32_t) + sizeof(dwTime_t), msg.data, msg.size);

    // Return values
    *data = buffer;
    *size = totalSize;
    *timestamp = msg.timestamp_us;

    return DW_SUCCESS;
}

dwStatus _dwSensorPlugin_returnRawData(uint8_t const* data,
                                      dwSensorPluginSensorHandle_t handle) {
    if (!data || !handle) {
        return DW_INVALID_ARGUMENT;
    }

    delete[] data;
    return DW_SUCCESS;
}


} // namespace

// Exported Function for DriveWorks SAL
extern "C" {
dwStatus dwSensorCANPlugin_getFunctionTable(dwSensorCANPluginFunctionTable* functions) {
    if (!functions) {
        return DW_INVALID_ARGUMENT;
    }

    functions->common.createHandle = _dwSensorPlugin_createHandle;
    functions->common.createSensor = _dwSensorPlugin_createSensor;
    functions->common.start = _dwSensorPlugin_start;
    functions->common.stop = _dwSensorPlugin_stop;
    functions->common.reset = _dwSensorPlugin_reset;
    functions->common.release = _dwSensorPlugin_release;
    functions->common.readRawData = _dwSensorPlugin_readRawData;
    functions->common.returnRawData = _dwSensorPlugin_returnRawData;
    

    functions->setFilter = _dwSensorPlugin_setFilter;
    functions->clearFilter = _dwSensorPlugin_clearFilter;
    functions->setUseHwTimestamps = _dwSensorPlugin_setUseHwTimestamps;
    functions->send = _dwSensorCANPlugin_send;
    functions->parseDataBuffer = _dwSensorCANPlugin_parseDataBuffer;
    return DW_SUCCESS;
}
}
