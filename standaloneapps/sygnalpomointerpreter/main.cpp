#include <csignal>
#include <iostream>
#include <string>
#include <thread>

#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/canbus/CAN.h>
#include <framework/ProgramArguments.hpp>
#include <framework/SamplesDataPath.hpp>
#include <framework/Log.hpp>
#include <framework/Checks.hpp>

#include "SygnalPomoMonitor.hpp"

namespace {
    volatile bool gRun = true;

    void sig_int_handler(int) {
        gRun = false;
    }
}

int main(int argc, const char** argv) {
    // Program arguments
    ProgramArguments arguments({
        ProgramArguments::Option_t("driver", "can.socket"),
        ProgramArguments::Option_t("params", "port=can0,bitrate=500000"),
        ProgramArguments::Option_t("dbc-path", (dw_samples::SamplesDataPath::get() + "/dbc").c_str()),
        ProgramArguments::Option_t("plugin-path", "/usr/local/driveworks/samples/bin/libsygnalpomo_can_plugin.so")
    });

    if (!arguments.parse(argc, argv)) {
        std::cout << "Usage: " << argv[0] << std::endl;
        std::cout << "\t--driver=can.socket \t\t: CAN driver to use\n";
        std::cout << "\t--params=port=can0,bitrate=500000 \t: CAN parameters\n";
        std::cout << "\t--dbc-path=/path/to/dbc \t: Path to DBC files\n";
        return -1;
    }


    std::cout << "[APP][INFO] Driver: " << arguments.get("driver") << "\n";
    std::cout << "[APP][INFO] Params: " << arguments.get("params") << "\n";
    std::cout << "[APP][INFO] Plugin Path: " << arguments.get("plugin-path") << "\n";


    // Initialize DriveWorks SDK 
    dwContextHandle_t sdk = DW_NULL_HANDLE;
    dwSALHandle_t sal = DW_NULL_HANDLE;
    
    // Set up signal handlers
    struct sigaction action = {};
    action.sa_handler = sig_int_handler;
    sigaction(SIGINT, &action, NULL);
    
    // Initialize logger
    dwLogger_initialize(getConsoleLoggerCallback(true));
    dwLogger_setLogLevel(DW_LOG_VERBOSE);

    // Initialize SDK context
    dwContextParameters sdkParams{};
    CHECK_DW_ERROR(dwInitialize(&sdk, DW_VERSION, &sdkParams));
    CHECK_DW_ERROR(dwSAL_initialize(&sal, sdk));

    // Create monitor instance
    SygnalPomoMonitor monitor;
    if (!monitor.initialize(arguments, sdk, sal)) {
        std::cerr << "[APP] Failed to initialize CAN monitor" << std::endl;
        return -1;
    }

     std::cout << "[APP][INFO] Monitoring CAN messages...\n";

    // Main loop
    while (gRun) {
        if (!monitor.processMessages()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    

    // Cleanup
    monitor.release();
    dwSAL_release(sal);
    dwRelease(sdk);
    dwLogger_release();

    return 0;
}