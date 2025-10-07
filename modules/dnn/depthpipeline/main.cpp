/////////////////////////////////////////////////////////////////////////////////////////
// Multi-Camera Depth Estimation Application
// Uses modular DepthPipeline architecture with integrated window management
/////////////////////////////////////////////////////////////////////////////////////////

#include "DepthPipeline.hpp"
#include "CameraCapture.hpp"
#include "DepthInferenceEngine.hpp"
#include "DepthRenderer.hpp"
#include "MemoryManager.hpp"

#include <framework/ProgramArguments.hpp>
#include <framework/SamplesDataPath.hpp>

#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <iomanip>

using namespace dw_samples::common;

/////////////////////////////////////////////////////////////////////////////////////////
// Global shutdown flag
/////////////////////////////////////////////////////////////////////////////////////////
static std::atomic<bool> g_run{true};

void signalHandler(int signal)
{
    (void)signal;
    g_run.store(false);
}

/////////////////////////////////////////////////////////////////////////////////////////
// Main Entry Point
/////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char** argv)
{
    // Install signal handlers
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // Parse command line arguments
    ProgramArguments args(argc, argv,
    {
        ProgramArguments::Option_t(
            "rig",
            (dw_samples::SamplesDataPath::get() + "/samples/sensors/camera/camera/rig.json").c_str(),
            "Rig configuration file"
        ),
        ProgramArguments::Option_t(
            "depth_model",
            (dw_samples::SamplesDataPath::get() + "/samples/detector/ampere-integrated/depth_anything_v2_fp32.bin").c_str(),
            "Depth model TensorRT file"
        ),
        ProgramArguments::Option_t(
            "mode",
            "realtime",
            "Pipeline mode: realtime | highquality | datacollection"
        ),
        ProgramArguments::Option_t(
            "fps",
            "30",
            "Target frames per second"
        )
    },

    "Multi-Camera Depth Estimation Pipeline");


    // Configure pipeline
    depth_pipeline::DepthPipeline::PipelineConfig config;
    config.rigPath = args.get("rig");
    config.depthModelPath = args.get("depth_model");
    config.targetFPS = std::stoi(args.get("fps"));
    config.enableVisualization = !args.enabled("offscreen");
    config.windowWidth = 1280;
    config.windowHeight = 800;

    // Parse mode
    std::string modeStr = args.get("mode");
    if (modeStr == "highquality") {
        config.mode = depth_pipeline::DepthPipeline::PipelineMode::HIGH_QUALITY;
    } else if (modeStr == "datacollection") {
        config.mode = depth_pipeline::DepthPipeline::PipelineMode::DATA_COLLECTION;
    } else {
        config.mode = depth_pipeline::DepthPipeline::PipelineMode::REAL_TIME;
    }

    // Print configuration
    std::cout << "=== Depth Estimation Pipeline Configuration ===\n";
    std::cout << "Rig config: " << config.rigPath << "\n";
    std::cout << "Depth model: " << config.depthModelPath << "\n";
    std::cout << "Mode: " << modeStr << "\n";
    std::cout << "Target FPS: " << config.targetFPS << "\n";
    std::cout << "Visualization: " << (config.enableVisualization ? "enabled" : "disabled") << "\n";
    std::cout << "===============================================\n\n";

    // Create and initialize pipeline
    depth_pipeline::DepthPipeline pipeline;

    dwStatus status = pipeline.initialize(config);
    if (status != DW_SUCCESS) {
        std::cerr << "ERROR: Failed to initialize pipeline: " << dwGetStatusName(status) << std::endl;
        return -1;
    }

    status = pipeline.start();
    if (status != DW_SUCCESS) {
        std::cerr << "ERROR: Failed to start pipeline: " << dwGetStatusName(status) << std::endl;
        return -1;
    }

    std::cout << "=== Pipeline Running ===\n";
    std::cout << "Press Ctrl+C to stop\n\n";

    // Performance monitoring
    auto lastStatsTime = std::chrono::high_resolution_clock::now();
    uint32_t frameCount = 0;

    // Main loop
    while (g_run.load() && pipeline.isRunning())
    {
        // Process one frame (capture → inference → render)
        status = pipeline.processFrame();
        
        if (status == DW_TIME_OUT) {
            // Timeout is acceptable - retry
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        if (status != DW_SUCCESS && status != DW_BUFFER_FULL) {
            std::cerr << "WARNING: processFrame returned " << dwGetStatusName(status) << std::endl;
        }

        // Swap buffers if rendering enabled
        if (pipeline.shouldRender()) {
            pipeline.swapBuffers();
        }

        frameCount++;

        // Print statistics every 5 seconds
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastStatsTime);
        
        if (elapsed.count() >= 5) {
            auto stats = pipeline.getStatistics();
            
            std::cout << "=== Statistics (5s interval) ===\n";
            std::cout << "Captured: " << stats.totalFramesCaptured 
                      << " | Inferences: " << stats.totalInferencesCompleted
                      << " | Dropped: " << stats.droppedFrames
                      << " | FPS: " << std::fixed << std::setprecision(1) << stats.currentFPS
                      << " | Latency: " << std::fixed << std::setprecision(1) << stats.avgLatencyMs << "ms\n";
            
            lastStatsTime = currentTime;
        }

        // Frame rate limiting
        if (config.targetFPS > 0) {
            uint32_t targetFrameTimeUs = 1000000 / config.targetFPS;
            std::this_thread::sleep_for(std::chrono::microseconds(targetFrameTimeUs));
        }
    }

    // Shutdown
    std::cout << "\n=== Shutting Down ===\n";
    pipeline.stop();

    // Final statistics
    auto finalStats = pipeline.getStatistics();
    std::cout << "\n=== Final Statistics ===\n";
    std::cout << "Total Frames Captured: " << finalStats.totalFramesCaptured << "\n";
    std::cout << "Total Inferences: " << finalStats.totalInferencesCompleted << "\n";
    std::cout << "Dropped Frames: " << finalStats.droppedFrames << "\n";
    std::cout << "Average FPS: " << std::fixed << std::setprecision(2) << finalStats.currentFPS << "\n";
    std::cout << "Average Latency: " << std::fixed << std::setprecision(2) << finalStats.avgLatencyMs << " ms\n";
    std::cout << "========================\n";

    return 0;
}