#include <csignal>
#include <iostream>
#include <string>
#include <thread>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/sensors/common/Sensors.h>
#include <framework/ProgramArguments.hpp>
#include <framework/SamplesDataPath.hpp>
#include <framework/Log.hpp>
#include <framework/Checks.hpp>
#include "SygnalPomoController.hpp"

namespace {
    volatile bool gRun = true;
    
    void sig_int_handler(int) {
        gRun = false;
    }
}




void runManualBrakeTest(SygnalPomoController& controller) {
    std::cout << "\n=== Manual Brake Test ===" << std::endl;
    int brake;
    float duration;
    
    std::cout << "Enter brake percentage (0-100): ";
    std::cin >> brake;
    
    if (brake < 0 || brake > 100) {
        std::cout << "Invalid brake value! Must be 0-100%" << std::endl;
        return;
    }
    
    std::cout << "Enter duration in seconds (0 for continuous): ";
    std::cin >> duration;
    
    if (duration < 0) {
        std::cout << "Invalid duration! Must be >= 0" << std::endl;
        return;
    }
    
    
    controller.applyBrake(brake, duration);
    
    if (duration > 0) {
        std::cout << "Brake applied for " << duration << " seconds (plugin will auto-release)" << std::endl;
    } else {
        std::cout << "Brake applied continuously. Send 0% to release." << std::endl;
    }
}

void runManualSteeringTest(SygnalPomoController& controller) {
    std::cout << "\n=== Manual Steering Test ===" << std::endl;
    int steering;
    
    std::cout << "Enter steering percentage (-100 to 100): ";
    std::cin >> steering;
    
    if (steering < -100 || steering > 100) {
        std::cout << "Invalid steering value! Must be -100 to 100%" << std::endl;
        return;
    }
    
    
    controller.setSteering(steering);
    
    if (steering == 0) {
        std::cout << "Steering centered" << std::endl;
    } else if (steering > 0) {
        std::cout << "Steering right " << steering << "%" << std::endl;
    } else {
        std::cout << "Steering left " << abs(steering) << "%" << std::endl;
    }
}

void runManualGearTest(SygnalPomoController& controller) {
    std::cout << "\n=== Manual Gear Change Test ===" << std::endl;
    int gear;
    
    std::cout << "Select gear:" << std::endl;
    std::cout << "0 = Park" << std::endl;
    std::cout << "1 = Reverse" << std::endl;
    std::cout << "2 = Neutral" << std::endl;
    std::cout << "3 = Drive" << std::endl;
    std::cout << "Choice: ";
    std::cin >> gear;
    
    if (gear < 0 || gear > 3) {
        std::cout << "Invalid gear selection! Must be 0-3" << std::endl;
        return;
    }
    
    GearPosition gearPos = static_cast<GearPosition>(gear);
    std::cout << "Initiating optimized gear change sequence..." << std::endl;
    
    
    if (controller.changeGear(gearPos)) {
        std::cout << "Gear change command sent successfully" << std::endl;
        std::cout << "Plugin will execute the full 4-phase sequence automatically" << std::endl;
    } else {
        std::cout << "Gear change command failed!" << std::endl;
    }
}


void runManualThrottleTest(SygnalPomoController& controller) {
    std::cout << "\n=== Manual Throttle Test ===" << std::endl;
    int throttle;
    float duration;
    
    std::cout << "Enter throttle percentage (0-100): ";
    std::cin >> throttle;
    
    if (throttle < 0 || throttle > 100) {
        std::cout << "Invalid throttle value! Must be 0-100%" << std::endl;
        return;
    }
    
    std::cout << "Enter duration in seconds (0 for continuous): ";
    std::cin >> duration;
    
    if (duration < 0) {
        std::cout << "Invalid duration! Must be >= 0" << std::endl;
        return;
    }
    
    controller.applyThrottle(throttle, duration);
    
    if (duration > 0) {
        std::cout << "Throttle applied for " << duration << " seconds" << std::endl;
    } else {
        std::cout << "Throttle applied continuously. Send 0% to release." << std::endl;
    }
}


void printManualTestMenu() {
    std::cout << "\n=== Manual Test Menu (Enhanced Controls) ===" << std::endl;
    std::cout << "1. Throttle Control Test" << std::endl;
    std::cout << "2. Brake Control Test" << std::endl;
    std::cout << "3. Steering Control Test" << std::endl;
    std::cout << "4. Gear Change Test (Optimized Sequence)" << std::endl;
    std::cout << "5. Clear All Subsystems" << std::endl;
    std::cout << "6. Return to Main Menu" << std::endl;
    std::cout << "Choice: ";
}


void runManualTestMode(SygnalPomoController& controller) {
    int choice;
    
    while (gRun) {
        printManualTestMenu();
        std::cin >> choice;
        
        switch (choice) {
            case 1:
                runManualThrottleTest(controller);
                break;
            case 2:
                runManualBrakeTest(controller);
                break;
            case 3:
                runManualSteeringTest(controller);
                break;
            case 4:
                runManualGearTest(controller);
                break;
            case 5:
                std::cout << "Clearing all subsystem states..." << std::endl;
                if (controller.clearAllSubsystems()) {
                    std::cout << "All subsystems cleared successfully" << std::endl;
                } else {
                    std::cout << "Failed to clear some subsystems" << std::endl;
                }
                break;
            case 6:
                return;
            default:
                std::cout << "Invalid choice!" << std::endl;
                break;
        }
        
        if (choice >= 1 && choice <= 5) {
            std::cout << "\nPress Enter to continue...";
            std::cin.ignore();
            std::cin.get();
        }
    }
}

void runMonitorMode(SygnalPomoController& controller) {
    std::cout << "\n=== CAN Message Monitor Mode ===" << std::endl;
    std::cout << "Monitoring CAN messages... Press Ctrl+C to exit" << std::endl;
    
    int messageCount = 0;
    while (gRun) {
        if (controller.readMessages()) {
            messageCount++;
            if (messageCount % 10 == 0) {
                std::cout << "[MONITOR] Processed " << messageCount << " messages..." << std::endl;
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    std::cout << "[MONITOR] Total messages processed: " << messageCount << std::endl;
}

void printMainMenu() {
    std::cout << "\n=== Sygnal Pomo Controller - Test Modes ===" << std::endl;
    std::cout << "1. Run Automated Test Suite" << std::endl;
    std::cout << "2. Manual Control Testing" << std::endl;
    std::cout << "3. CAN Message Monitor" << std::endl;
    std::cout << "4. Exit" << std::endl;
    std::cout << "Choice: ";
}

int main(int argc, const char** argv) {
    // Program arguments
    ProgramArguments arguments({
        ProgramArguments::Option_t("driver", "can.socket"),
        ProgramArguments::Option_t("params", "port=can0,bitrate=500000"),
        ProgramArguments::Option_t("dbc-path", (dw_samples::SamplesDataPath::get() + "/dbc").c_str()),
        ProgramArguments::Option_t("plugin-path", "/usr/local/driveworks/samples/bin/libsygnalpomo_can_plugin.so"),
        ProgramArguments::Option_t("auto-test", "false")  // Run automated tests immediately
    });
    
    if (!arguments.parse(argc, argv)) {
        std::cout << "Usage: " << argv[0] << std::endl;
        std::cout << "\t--driver=can.socket \t\t\t: CAN driver to use\n";
        std::cout << "\t--params=port=can0,bitrate=500000 \t: CAN parameters\n";
        std::cout << "\t--dbc-path=/path/to/dbc \t\t: Path to DBC files\n";
        std::cout << "\t--plugin-path=/path/to/plugin.so \t: Path to plugin library\n";
        std::cout << "\t--auto-test=true \t\t\t: Run automated tests and exit\n";
        return -1;
    }
    
    std::cout << "=== Sygnal Pomo Controller - Core Vehicle Controls ===" << std::endl;
    std::cout << "[APP] Supported Controls: Brake, Steering, Gear Change" << std::endl;
    std::cout << "[APP] Driver: " << arguments.get("driver") << std::endl;
    std::cout << "[APP] Params: " << arguments.get("params") << std::endl;
    std::cout << "[APP] Plugin: " << arguments.get("plugin-path") << std::endl;
    
    // Set up signal handlers
    struct sigaction action = {};
    action.sa_handler = sig_int_handler;
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGTERM, &action, NULL);
    
    // Initialize logger
    dwLogger_initialize(getConsoleLoggerCallback(true));
    dwLogger_setLogLevel(DW_LOG_VERBOSE);
    
    // Initialize SDK context
    dwContextHandle_t sdk = DW_NULL_HANDLE;
    dwSALHandle_t sal = DW_NULL_HANDLE;
    dwContextParameters sdkParams{};
    
    CHECK_DW_ERROR(dwInitialize(&sdk, DW_VERSION, &sdkParams));
    CHECK_DW_ERROR(dwSAL_initialize(&sal, sdk));
    
    // Create controller instance
    SygnalPomoController controller;
    if (!controller.initialize(arguments, sdk, sal)) {
        std::cerr << "[APP] Failed to initialize controller" << std::endl;
        dwSAL_release(sal);
        dwRelease(sdk);
        return -1;
    }
    
    // Check if auto-test mode
    if (arguments.get("auto-test") == "true") {
        std::cout << "\n[APP] Running in automated test mode..." << std::endl;
        bool testResult = controller.runAutomatedTests();
        
        std::cout << "\n[APP] Automated test result: " 
                  << (testResult ? "PASS" : "FAIL") << std::endl;
        
        // Cleanup and exit
        controller.release();
        dwSAL_release(sal);
        dwRelease(sdk);
        dwLogger_release();
        
        return testResult ? 0 : 1;
    }
    
    // Interactive mode
    int choice;
    while (gRun) {
        printMainMenu();
        std::cin >> choice;
        
        switch (choice) {
            case 1:
                std::cout << "\n[APP] Starting automated test suite..." << std::endl;
                if (controller.runAutomatedTests()) {
                    std::cout << "\n[APP]  All automated tests PASSED" << std::endl;
                } else {
                    std::cout << "\n[APP]  Some automated tests FAILED" << std::endl;
                }
                break;
                
            case 2:
                runManualTestMode(controller);
                break;
                
            case 3:
                runMonitorMode(controller);
                break;
                
            case 4:
                gRun = false;
                break;
                
            default:
                std::cout << "Invalid choice!" << std::endl;
                break;
        }
        
        if (choice >= 1 && choice <= 3 && gRun) {
            std::cout << "\nPress Enter to continue...";
            std::cin.ignore();
            std::cin.get();
        }
    }
    
    std::cout << "[APP] Shutting down..." << std::endl;
    
    // Cleanup
    controller.release();
    dwSAL_release(sal);
    dwRelease(sdk);
    dwLogger_release();
    
    return 0;
}