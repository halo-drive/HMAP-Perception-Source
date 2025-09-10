#ifndef SYGNALPOMO_PLUGIN_API_HPP_
#define SYGNALPOMO_PLUGIN_API_HPP_

#include <dw/sensors/canbus/CAN.h>
#include <dw/sensors/canbus/Interpreter.h>
#include "Common.hpp"

namespace sygnalpomo {

    //------------------------------------------------------------------------------
    // Forward Declarations
    //------------------------------------------------------------------------------
    class SygnalPomoDriver;

    /**
     * @brief CAN Plugin Interface Class
     *
     * Defines the API contract for DriveWorks to interact with the custom CAN plugin.
     */
    class CANPluginAPI {
    public:
        virtual ~CANPluginAPI() = default;

        //--------------------------------------------------------------------------
        // Core Sensor Operations
        //--------------------------------------------------------------------------
        virtual dwStatus initialize(dwContextHandle_t ctx, dwSALHandle_t sal) = 0;
        virtual dwStatus createSensor(dwSensorHandle_t* sensor, const dwSensorParams* params) = 0;
        virtual dwStatus start(dwSensorHandle_t sensor) = 0;
        virtual dwStatus stop(dwSensorHandle_t sensor) = 0;
        virtual dwStatus release(dwSensorHandle_t sensor) = 0;

        //--------------------------------------------------------------------------
        // CAN-Specific Operations
        //--------------------------------------------------------------------------
        virtual dwStatus readMessage(dwCANMessage* msg, dwTime_t timeoutUs, dwSensorHandle_t sensor) = 0;
        virtual dwStatus sendMessage(const dwCANMessage* msg, dwTime_t timeoutUs, dwSensorHandle_t sensor) = 0;
        virtual dwStatus setMessageFilter(const uint32_t* ids, const uint32_t* masks, uint16_t numCanIDs, dwSensorHandle_t sensor) = 0;
        virtual dwStatus setUseHwTimestamps(bool flag, dwSensorHandle_t sensor) = 0;
    };

} // namespace sygnalpomo

#endif // SYGNALPOMO_PLUGIN_API_HPP_
