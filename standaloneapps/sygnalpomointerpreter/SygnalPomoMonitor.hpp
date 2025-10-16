#ifndef SYGNALPOMO_MONITOR_HPP_
#define SYGNALPOMO_MONITOR_HPP_

#include <dw/core/context/Context.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/canbus/CAN.h>
#include <dw/sensors/canbus/Interpreter.h>
#include <dw/sensors/legacy/plugins/canbus/CANPlugin.h>
#include <framework/ProgramArguments.hpp>

#include <vector>
#include <queue>
#include <mutex>


struct CANConfiguration {
    std::string driver;
    std::string device;
    uint32_t bitrate;
    std::string pluginPath;
    std::string dbcPath;

    static CANConfiguration fromProgramArguments(const ProgramArguments& args);
};

class SygnalPomoMonitor {
public:
    SygnalPomoMonitor();
    ~SygnalPomoMonitor();

    bool initialize(const ProgramArguments& arguments, 
                   dwContextHandle_t context,
                   dwSALHandle_t sal);
    bool processMessages();
    void release();

private:
    dwContextHandle_t m_context;
    dwSALHandle_t m_sal;
    dwSensorHandle_t m_canSensor;
    bool m_initialized;
    static constexpr dwTime_t DEFAULT_TIMEOUT_US = 100000;
    
    //Plugin dynamic loading
    void* m_pluginHandle;
    typedef dwStatus (*GetFunctionTableFn)(dwSensorCANPluginFunctionTable*);
    GetFunctionTableFn m_getFunctionTableFn;
    


    //Plugin function tables
    dwSensorCANPluginFunctionTable m_pluginFunctions;
    dwSensorCommonPluginFunctions m_commonFunctions;

   //Plugin initialization
    dwStatus getFunctionTable(const std::string& pluginpath);

    //message processing
    bool pluginProcessMessage(const dwCANMessage& msg);
    
    struct RawBuffer{
        std::vector<uint8_t> data;
        size_t size;
        dwTime_t timestamp;
    };
    std::queue<RawBuffer> m_rawBuffers;
    std::mutex m_bufferMutex;
};

#endif // SYGNALPOMO_MONITOR_HPP_