/////////////////////////////////////////////////////////////////////////////////////////
// Livox HAP Plugin for NVIDIA DriveWorks
//
// Copyright (c) 2025 - All rights reserved.
/////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <iostream>
#include <ctime>
#include <iomanip>

#include <dw/core/base/Types.h>
#include <dw/core/context/Context.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/lidar/Lidar.h>
#include <dw/sensors/legacy/plugins/SensorCommonPlugin.h>
#include <dw/sensors/legacy/plugins/lidar/LidarPlugin.h>

#include "LivoxHapPlugin.hpp"

// Debug logging macro
#define DEBUG_LOG(msg) { \
    auto now = std::time(nullptr); \
    auto tm_info = std::localtime(&now); \
    std::cout << "[" << std::put_time(tm_info, "%Y-%m-%d %H:%M:%S") << "] LIVOX_HAP_PLUGIN: " << msg << std::endl; \
}

//################################################################################
//############################### Helper Functions ###############################
//################################################################################
static dwStatus IsValidSensor(dw::plugin::lidar::LivoxHapPlugin* plugin)
{
    DEBUG_LOG("IsValidSensor called with plugin ptr: " << plugin);
    
    if (!plugin) {
        DEBUG_LOG("Plugin pointer is NULL");
        return DW_INVALID_HANDLE;
    }
    
    for (auto& i : dw::plugin::lidar::LivoxHapPlugin::g_pluginInstances)
    {
        if (i.get() == plugin)
        {
            DEBUG_LOG("Plugin found in instance list");
            return DW_SUCCESS;
        }
    }

    DEBUG_LOG("Plugin NOT found in instance list (size: " << dw::plugin::lidar::LivoxHapPlugin::g_pluginInstances.size() << ")");
    return DW_INVALID_HANDLE;
}

//################################################################################
//###################### Common Sensor Plugin Functions ##########################
//################################################################################

// exported functions
extern "C" {

dwStatus _dwSensorPlugin_createHandle(dwSensorPluginSensorHandle_t* sensor,
    dwSensorPluginProperties* properties,
    char const* params,
    dwContextHandle_t ctx)
{
    DEBUG_LOG("_dwSensorPlugin_createHandle called");
    DEBUG_LOG("  sensor ptr: " << sensor);
    DEBUG_LOG("  properties ptr: " << properties);
    DEBUG_LOG("  ctx ptr: " << ctx);
    DEBUG_LOG("  params: " << (params ? params : "NULL"));

    if (!sensor)
    {
        DEBUG_LOG("  Invalid argument: sensor is NULL");
        return DW_INVALID_ARGUMENT;
    }

    // Important change: create default properties if none provided
    if (properties) {
        properties->packetSize = 1024 * 1024;  // 1MB buffer size
        properties->rawToDec = DW_SENSORS_RAW_DEC_MANY_TO_ONE;  // Multiple raw packets to one decoded point cloud
        DEBUG_LOG("  Properties set successfully");
    }
    else
    {
        DEBUG_LOG("  Warning: properties is NULL, but continuing anyway");
        // Instead of returning error, we'll continue and just not populate properties
    }

    try {
        DEBUG_LOG("  Creating new LivoxHapPlugin instance");
        auto pluginInstance = std::unique_ptr<dw::plugin::lidar::LivoxHapPlugin>(
        new dw::plugin::lidar::LivoxHapPlugin(ctx));

        DEBUG_LOG("  Plugin instance created successfully");
        DEBUG_LOG("  Current plugin count: " << dw::plugin::lidar::LivoxHapPlugin::g_pluginInstances.size());

        dw::plugin::lidar::LivoxHapPlugin::g_pluginInstances.push_back(std::move(pluginInstance));
        DEBUG_LOG("  Added plugin to instances, new count: " << dw::plugin::lidar::LivoxHapPlugin::g_pluginInstances.size());

        *sensor = static_cast<dwSensorPluginSensorHandle_t>(dw::plugin::lidar::LivoxHapPlugin::g_pluginInstances.back().get());
        DEBUG_LOG("  Set sensor handle: " << *sensor);

        DEBUG_LOG("_dwSensorPlugin_createHandle completed successfully");
        return DW_SUCCESS;
    }
    catch (const std::exception& e) {
        DEBUG_LOG("  EXCEPTION in _dwSensorPlugin_createHandle: " << e.what());
        return DW_FAILURE;
    }
    catch (...) {
        DEBUG_LOG("  UNKNOWN EXCEPTION in _dwSensorPlugin_createHandle");
        return DW_FAILURE;
    }
}

dwStatus _dwSensorPlugin_release(dwSensorPluginSensorHandle_t handle)
{
    DEBUG_LOG("_dwSensorPlugin_release called with handle: " << handle);
    
    dw::plugin::lidar::LivoxHapPlugin* plugin = static_cast<dw::plugin::lidar::LivoxHapPlugin*>(handle);
    DEBUG_LOG("  Plugin ptr: " << plugin);
    
    dwStatus ret = IsValidSensor(plugin);
    if (ret != DW_SUCCESS)
    {
        DEBUG_LOG("  Invalid plugin handle");
        return ret;
    }

    DEBUG_LOG("  Finding plugin in instance list");
    // Find the plugin instance in the list
    auto iter = std::find_if(dw::plugin::lidar::LivoxHapPlugin::g_pluginInstances.begin(),
                              dw::plugin::lidar::LivoxHapPlugin::g_pluginInstances.end(),
                              [&plugin](std::unique_ptr<dw::plugin::lidar::LivoxHapPlugin>& instance) {
                                  return (instance.get() == plugin);
                              });

    // If found, release it
    if (iter != dw::plugin::lidar::LivoxHapPlugin::g_pluginInstances.end())
    {
        DEBUG_LOG("  Plugin found, stopping sensor");
        // Stop decoding process
        ret = plugin->stopSensor();
        if (ret != DW_SUCCESS)
        {
            DEBUG_LOG("  Failed to stop sensor: " << ret);
            return ret;
        }

        DEBUG_LOG("  Releasing sensor resources");
        // Release resources
        ret = plugin->releaseSensor();
        if (ret != DW_SUCCESS)
        {
            DEBUG_LOG("  Failed to release sensor: " << ret);
            return ret;
        }

        DEBUG_LOG("  Removing plugin from instance container");
        // Remove instance from container
        dw::plugin::lidar::LivoxHapPlugin::g_pluginInstances.erase(iter);
        DEBUG_LOG("  Plugin released successfully");
        return DW_SUCCESS;
    }

    // Plugin not found
    DEBUG_LOG("  Plugin not found in instance list");
    return DW_FAILURE;
}

dwStatus _dwSensorPlugin_createSensor(char const* params,
    dwSALHandle_t sal,
    dwSensorPluginSensorHandle_t handle)
{
    DEBUG_LOG("_dwSensorPlugin_createSensor called");
    DEBUG_LOG("  params: " << (params ? params : "NULL"));
    DEBUG_LOG("  sal: " << sal);
    DEBUG_LOG("  handle: " << handle);

    dw::plugin::lidar::LivoxHapPlugin* plugin = static_cast<dw::plugin::lidar::LivoxHapPlugin*>(handle);
    DEBUG_LOG("  Plugin ptr: " << plugin);

    dwStatus ret = IsValidSensor(plugin);
    if (ret != DW_SUCCESS)
    {
    DEBUG_LOG("  Invalid plugin handle");
    return ret;
    }

    DEBUG_LOG("  Calling plugin->createSensor()");
    ret = plugin->createSensor(sal, params);
    DEBUG_LOG("  createSensor result: " << ret);
    return ret;
}

dwStatus _dwSensorPlugin_start(dwSensorPluginSensorHandle_t handle)
{
    DEBUG_LOG("_dwSensorPlugin_start called with handle: " << handle);
    
    dw::plugin::lidar::LivoxHapPlugin* plugin = static_cast<dw::plugin::lidar::LivoxHapPlugin*>(handle);
    DEBUG_LOG("  Plugin ptr: " << plugin);
    
    dwStatus ret = IsValidSensor(plugin);
    if (ret != DW_SUCCESS)
    {
        DEBUG_LOG("  Invalid plugin handle");
        return ret;
    }

    DEBUG_LOG("  Calling plugin->startSensor()");
    ret = plugin->startSensor();
    DEBUG_LOG("  startSensor result: " << ret);
    return ret;
}

dwStatus _dwSensorPlugin_stop(dwSensorPluginSensorHandle_t handle)
{
    DEBUG_LOG("_dwSensorPlugin_stop called with handle: " << handle);
    
    dw::plugin::lidar::LivoxHapPlugin* plugin = static_cast<dw::plugin::lidar::LivoxHapPlugin*>(handle);
    DEBUG_LOG("  Plugin ptr: " << plugin);
    
    dwStatus ret = IsValidSensor(plugin);
    if (ret != DW_SUCCESS)
    {
        DEBUG_LOG("  Invalid plugin handle");
        return ret;
    }

    DEBUG_LOG("  Calling plugin->stopSensor()");
    ret = plugin->stopSensor();
    DEBUG_LOG("  stopSensor result: " << ret);
    return ret;
}

dwStatus _dwSensorPlugin_reset(dwSensorPluginSensorHandle_t handle)
{
    DEBUG_LOG("_dwSensorPlugin_reset called with handle: " << handle);
    
    dw::plugin::lidar::LivoxHapPlugin* plugin = static_cast<dw::plugin::lidar::LivoxHapPlugin*>(handle);
    DEBUG_LOG("  Plugin ptr: " << plugin);
    
    dwStatus ret = IsValidSensor(plugin);
    if (ret != DW_SUCCESS)
    {
        DEBUG_LOG("  Invalid plugin handle");
        return ret;
    }

    DEBUG_LOG("  Calling plugin->resetSensor()");
    ret = plugin->resetSensor();
    DEBUG_LOG("  resetSensor result: " << ret);
    return ret;
}

dwStatus _dwSensorPlugin_readRawData(uint8_t const** data,
    size_t* size,
    dwTime_t* timestamp,
    dwTime_t timeout_us,
    dwSensorPluginSensorHandle_t handle)
{
    DEBUG_LOG("_dwSensorPlugin_readRawData called with handle: " << handle);

    dw::plugin::lidar::LivoxHapPlugin* plugin = static_cast<dw::plugin::lidar::LivoxHapPlugin*>(handle);
    DEBUG_LOG("  Plugin ptr: " << plugin);

    dwStatus ret = IsValidSensor(plugin);
    if (ret != DW_SUCCESS)
    {
        DEBUG_LOG("  Invalid plugin handle");
        return ret;
    }

    DEBUG_LOG("  Calling plugin->readRawData()");
    ret = plugin->readRawData(data, size, timestamp, timeout_us);
    DEBUG_LOG("  readRawData result: " << ret);
    if (ret == DW_SUCCESS) {
        DEBUG_LOG("  Read data size: " << *size << ", timestamp: " << *timestamp);
    }
    return ret;
}

dwStatus _dwSensorPlugin_returnRawData(uint8_t const* data,
    dwSensorPluginSensorHandle_t handle)
{
    DEBUG_LOG("_dwSensorPlugin_returnRawData called with handle: " << handle);

    dw::plugin::lidar::LivoxHapPlugin* plugin = static_cast<dw::plugin::lidar::LivoxHapPlugin*>(handle);
    DEBUG_LOG("  Plugin ptr: " << plugin);

    dwStatus ret = IsValidSensor(plugin);
    if (ret != DW_SUCCESS)
    {
        DEBUG_LOG("  Invalid plugin handle");
        return ret;
    }

    DEBUG_LOG("  Calling plugin->returnRawData()");
    ret = plugin->returnRawData(data);
    DEBUG_LOG("  returnRawData result: " << ret);
    return ret;
}

dwStatus _dwSensorPlugin_pushData(size_t* lenPushed,
    uint8_t const* data,
    size_t const size,
    dwSensorPluginSensorHandle_t handle)
{
    DEBUG_LOG("_dwSensorPlugin_pushData called with handle: " << handle);
    DEBUG_LOG("  Data ptr: " << (void*)data << ", size: " << size);

    dw::plugin::lidar::LivoxHapPlugin* plugin = static_cast<dw::plugin::lidar::LivoxHapPlugin*>(handle);
    DEBUG_LOG("  Plugin ptr: " << plugin);

    dwStatus ret = IsValidSensor(plugin);
    if (ret != DW_SUCCESS)
    {
        DEBUG_LOG("  Invalid plugin handle");
        return ret;
    }

    DEBUG_LOG("  Calling plugin->pushData()");
    ret = plugin->pushData(lenPushed, data, size);
    DEBUG_LOG("  pushData result: " << ret << ", lenPushed: " << *lenPushed);
    return ret;
}

//################################################################################
//###################### Lidar Specific Plugin Functions #########################
//################################################################################

dwStatus _dwSensorLidarPlugin_parseDataBuffer(dwLidarDecodedPacket* output,
    const dwTime_t hostTimestamp,
    dwSensorPluginSensorHandle_t handle)
{
    DEBUG_LOG("_dwSensorLidarPlugin_parseDataBuffer called with handle: " << handle);
    DEBUG_LOG("  hostTimestamp: " << hostTimestamp);

    dw::plugin::lidar::LivoxHapPlugin* plugin = static_cast<dw::plugin::lidar::LivoxHapPlugin*>(handle);
    DEBUG_LOG("  Plugin ptr: " << plugin);

    dwStatus ret = IsValidSensor(plugin);
    if (ret != DW_SUCCESS)
    {
    DEBUG_LOG("  Invalid plugin handle");
    return ret;
    }

    DEBUG_LOG("  Calling plugin->parseDataBuffer()");
    ret = plugin->parseDataBuffer(output, hostTimestamp);

    if (ret == DW_SUCCESS && output) {
        DEBUG_LOG("  Parsed packet - points: " << output->nPoints << 
        ", maxPoints: " << output->maxPoints << 
        ", scanComplete: " << (output->scanComplete ? "true" : "false"));
    } else {
        DEBUG_LOG("  parseDataBuffer result: " << ret);
    }

    return ret;
}

dwStatus _dwSensorLidarPlugin_getConstants(_dwSensorLidarDecoder_constants* constants,
    dwSensorPluginSensorHandle_t handle)
   {
    DEBUG_LOG("_dwSensorLidarPlugin_getConstants called with handle: " << handle);
    
    dw::plugin::lidar::LivoxHapPlugin* plugin = static_cast<dw::plugin::lidar::LivoxHapPlugin*>(handle);
    DEBUG_LOG("  Plugin ptr: " << plugin);
    
    dwStatus ret = IsValidSensor(plugin);
    if (ret != DW_SUCCESS)
    {
    DEBUG_LOG("  Invalid plugin handle");
    return ret;
    }
    
    DEBUG_LOG("  Calling plugin->getConstants()");
    ret = plugin->getConstants(constants);
    
    if (ret == DW_SUCCESS && constants) {
        DEBUG_LOG("  Constants - maxPayloadSize: " << constants->maxPayloadSize);
        DEBUG_LOG("  Properties - pointsPerSpin: " << constants->properties.pointsPerSpin << 
        ", spinFrequency: " << constants->properties.spinFrequency);
    } else {
        DEBUG_LOG("  getConstants result: " << ret);
    }
    
    return ret;
   }

//################################################################################
//################# Required plugin entry point to get function table ############
//################################################################################

dwStatus dwSensorLidarPlugin_getFunctionTable(dwSensorLidarPluginFunctionTable* functions)
{
    DEBUG_LOG("dwSensorLidarPlugin_getFunctionTable called");
    
    if (functions == nullptr)
    {
        DEBUG_LOG("  Invalid argument: functions is NULL");
        return DW_INVALID_ARGUMENT;
    }

    DEBUG_LOG("  Initializing function table");
    
    // Initialize common functions
    functions->common = {};
    functions->common.createHandle = _dwSensorPlugin_createHandle;
    functions->common.createSensor = _dwSensorPlugin_createSensor;
    functions->common.release = _dwSensorPlugin_release;
    functions->common.start = _dwSensorPlugin_start;
    functions->common.stop = _dwSensorPlugin_stop;
    functions->common.reset = _dwSensorPlugin_reset;
    functions->common.readRawData = _dwSensorPlugin_readRawData;
    functions->common.returnRawData = _dwSensorPlugin_returnRawData;
    functions->common.pushData = _dwSensorPlugin_pushData;

    // Initialize lidar specific functions
    functions->parseDataBuffer = _dwSensorLidarPlugin_parseDataBuffer;
    functions->getDecoderConstants = _dwSensorLidarPlugin_getConstants;
    functions->sendMessage = nullptr;  // Not implemented for Livox
    
    
    DEBUG_LOG("  Function table initialized successfully");
    return DW_SUCCESS;
}

} // extern "C"