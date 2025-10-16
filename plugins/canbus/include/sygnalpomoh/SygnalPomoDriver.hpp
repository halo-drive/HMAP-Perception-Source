#ifndef SYGNALPOMO_DRIVER_HPP_
#define SYGNALPOMO_DRIVER_HPP_

#include <dw/sensors/canbus/CAN.h>
#include <dw/sensors/canbus/Interpreter.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/legacy/plugins/SensorCommonPlugin.h>
#include "Common.hpp"
#include <iostream>
#include <string>
#include <map>
#include <queue>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

namespace sygnalpomo {

// Remove duplicate ControlSubsystem enum - use the one from Common.hpp

enum ValueType {
    INTEGER,
    PERCENTAGE,
    NORMALIZED,
    ENUM
};

struct SubsystemConfig {
    uint8_t busAddress;
    uint8_t subsystemID;
    uint8_t interfaceID;
    bool requiresEnable;
    ValueType valueType;
    uint32_t messageId;
};

class TaskManager {
private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::queue<std::function<void()>> m_tasks;
    std::thread m_worker;
    bool m_running = false;

public:
    TaskManager() {
        m_running = true;
        m_worker = std::thread([this]() {
            workerThread();
        });
    }
    
    ~TaskManager() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_running = false;
        }
        m_cv.notify_all();
        if (m_worker.joinable()) {
            m_worker.join();
        }
    }
    
    void scheduleTask(std::function<void()> task, dwTime_t delayUs) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto executeTime = std::chrono::steady_clock::now() + 
                          std::chrono::microseconds(delayUs);
        
        m_tasks.push([task, executeTime]() {
            auto now = std::chrono::steady_clock::now();
            if (now < executeTime) {
                std::this_thread::sleep_until(executeTime);
            }
            task();
        });
        m_cv.notify_one();
    }
    
private:
    void workerThread() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cv.wait(lock, [this]() -> bool {
                    return !m_running || !m_tasks.empty(); 
                });
                
                if (!m_running && m_tasks.empty()) {
                    break;
                }
                
                if (!m_tasks.empty()) {
                    task = std::move(m_tasks.front());
                    m_tasks.pop();
                }
            }
            
            if (task) {
                task();
            }
        }
    }
};    
    
class SygnalPomoDriver {
public:
    SygnalPomoDriver(dwContextHandle_t ctx, dwSALHandle_t sal = nullptr);
    ~SygnalPomoDriver();
    
    void setSAL(dwSALHandle_t sal);
    bool isCANSensorValid() const {return m_canSensor != nullptr;}
    
    //--------------------------------------------------------------------------
    // Core Sensor Operations
    //--------------------------------------------------------------------------
    dwStatus initialize(const char* params);
    dwStatus start();
    dwStatus stop();
    dwStatus release();
    //versioning 
    dwStatus getSensorInformation(dwSensorPlugin_information* information);
    std::string getVersionString() const;
    //--------------------------------------------------------------------------
    // CAN Message Operations
    //--------------------------------------------------------------------------
    dwStatus readMessage(dwCANMessage* msg, dwTime_t timeoutUs);
    dwStatus sendMessage(dwCANMessage* msg, dwTime_t timeoutUs);
    dwStatus setMessageFilter(const uint32_t* ids, const uint32_t* masks, uint16_t numCanIDs);
    dwStatus setUseHwTimestamps(bool flag);

    //--------------------------------------------------------------------------
    // DBC Parsing Functions
    //--------------------------------------------------------------------------
    bool loadDBCConfig(const std::string& dbcFilePath);
    void encodeMessageWithDBC(dwCANMessage* msg);
    void decodeMessageWithDBC(const dwCANMessage* msg);
    std::string getDBCValue(const std::string& fieldName, uint8_t value);

private:
    //--------------------------------------------------------------------------
    // DriveWorks Handles
    //--------------------------------------------------------------------------
    dwSALHandle_t m_sal;
    dwContextHandle_t m_ctx;
    dwSensorHandle_t m_canSensor;
    TaskManager m_taskManager;
    
    //--------------------------------------------------------------------------
    // State Flags
    //--------------------------------------------------------------------------
    bool m_isRunning;
    bool m_useHwTimestamps;
    bool m_virtualSensorFlag;
    std::string m_dbcBasePath;
    uint8_t controlCounter = 0;
    static constexpr dwTime_t DEFAULT_CAN_TIMEOUT_US = 100000;
    
    struct MessageState {
        struct {
            uint8_t systemState;
            uint8_t busAddress;
            uint8_t subsystemID;
            bool interface0State;
            bool interface1State;
            uint16_t count16;
            dwTime_t timestamp_us;
            bool newData;
        } heartbeat;

        struct {
            int32_t value;
            uint8_t count8;
            uint8_t busAddress;
            uint8_t subsystemID;
            uint8_t interfaceID;
            dwTime_t timestamp_us;
            bool newData;
        } control;

        struct {
            uint8_t faultState;
            uint8_t faultCause;
            uint16_t faultCount;
            uint8_t busAddress;
            uint8_t subsystemID;
            dwTime_t timestamp_us;
            bool newData;
        } fault;
    } m_state;

    uint32_t getNumAvailableSignals() const;
    bool getSignalInfo(const char** name, dwTrivialDataType* type, dwCANVehicleData* data, uint32_t idx) const;
    bool getSignalValuef32(float32_t* value, dwTime_t* timestamp, uint32_t idx) const;
    bool getSignalValuei32(int32_t* value, dwTime_t* timestamp, uint32_t idx) const;
    mutable std::string m_versionString;

    // Message Handlers
    void handleHeartbeatMessage(const dwCANMessage* msg, const uint8_t* processedData);
    void handleControlMessage(const dwCANMessage* msg, const uint8_t* processedData);
    void handleFaultMessage(const dwCANMessage* msg, const uint8_t* processedData);
    void packCANSignals(uint8_t* data, const std::map<std::string, int32_t>& signals);
    int32_t convertValue(ControlSubsystem subsystem, int32_t rawValue);
    dwStatus sendUnifiedCommand(ControlSubsystem subsystem, int32_t value, float duration);
    dwStatus sendEnableMessage(uint8_t busAddress, uint8_t interfaceID);
    uint8_t calculateStandardCRC8(const uint8_t* data, size_t length);
    void scheduleReleaseCommand(uint32_t messageId, float duration_sec);
    dwStatus executeOptimizedGearSequence(int32_t gearValue);
    //--------------------------------------------------------------------------
    // Rule of 5: Prevent Copy Operations
    //--------------------------------------------------------------------------
    SygnalPomoDriver(const SygnalPomoDriver&) = delete;
    SygnalPomoDriver& operator=(const SygnalPomoDriver&) = delete;
};

} // namespace sygnalpomo

#endif // SYGNALPOMO_DRIVER_HPP_