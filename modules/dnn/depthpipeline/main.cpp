#include "DepthPipeline.hpp"
#include <framework/ProgramArguments.hpp>
#include <framework/WindowGLFW.hpp>
#include <framework/SamplesDataPath.hpp>

#include <iostream>
#include <csignal>

using namespace depth_pipeline;
using namespace dw_samples::common;

static std::atomic<bool> g_run{true};

void signalHandler(int signal) {
    (void)signal;
    g_run.store(false);
}

int main(int argc, const char** argv) {
    // Register signal handler
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    // Parse command line arguments
    ProgramArguments args(argc, argv, {
        ProgramArguments::Option_t(
            "rig",
            (dw_samples::SamplesDataPath::get() + 
             "/samples/sensors/camera/camera/rig_4cam.json").c_str(),
            "Path to rig configuration file"
        ),
        ProgramArguments::Option_t(
            "depth_model",
            "",
            "Path to DepthAnythingV2 TensorRT model (auto-detected if empty)"
        ),
        ProgramArguments::Option_t(
            "mode",
            "realtime",
            "Pipeline mode: realtime, high_quality, data_collection"
        ),
        ProgramArguments::Option_t(
            "fps",
            "30",
            "Target frames per second"
        ),
        ProgramArguments::Option_t(
            "no-viz",
            "",
            "Disable visualization (depth collection only)"
        )
    }, "Real-time Multi-Camera Depth Estimation Pipeline");
    
    // Parse pipeline mode
    DepthPipeline::PipelineMode mode = DepthPipeline::PipelineMode::REAL_TIME;
    std::string modeStr = args.get("mode");
    if (modeStr == "high_quality") {
        mode = DepthPipeline::PipelineMode::HIGH_QUALITY;
    } else if (modeStr == "data_collection") {
        mode = DepthPipeline::PipelineMode::DATA_COLLECTION;
    }
    
    // Auto-detect model path if not provided
    std::string depthModelPath = args.get("depth_model");
    if (depthModelPath.empty()) {
        depthModelPath = dw_samples::SamplesDataPath::get() + 
                        "/samples/detector/ampere-integrated/depth_anything_v2_fp32.bin";
    }
    
    // Configure pipeline
    DepthPipeline::PipelineConfig config;
    config.rigPath = args.get("rig");
    config.depthModelPath = depthModelPath;
    config.mode = mode;
    config.targetFPS = std::stoi(args.get("fps"));
    config.windowWidth = 1280;
    config.windowHeight = 800;
    config.enableVisualization = !args.enabled("no-viz");
    
    std::cout << "=== Depth Estimation Pipeline Configuration ===" << std::endl;
    std::cout << "Rig config: " << config.rigPath << std::endl;
    std::cout << "Depth model: " << config.depthModelPath << std::endl;
    std::cout << "Mode: " << modeStr << std::endl;
    std::cout << "Target FPS: " << config.targetFPS << std::endl;
    std::cout << "Visualization: " << (config.enableVisualization ? "enabled" : "disabled") << std::endl;
    std::cout << "===============================================" << std::endl;
    
    try {
        // Create pipeline
        DepthPipeline pipeline;
        
        // Initialize
        dwStatus status = pipeline.initialize(config);
        if (status != DW_SUCCESS) {
            std::cerr << "ERROR: Failed to initialize pipeline: " 
                     << dwGetStatusName(status) << std::endl;
            return -1;
        }
        
        // Start pipeline
        status = pipeline.start();
        if (status != DW_SUCCESS) {
            std::cerr << "ERROR: Failed to start pipeline: " 
                     << dwGetStatusName(status) << std::endl;
            return -1;
        }
        
        std::cout << "Pipeline started successfully. Press Ctrl+C to stop." << std::endl;
        
        // Performance monitoring
        auto lastStatsTime = std::chrono::high_resolution_clock::now();
        uint32_t frameCount = 0;
        
        // Main processing loop
        while (g_run.load() && pipeline.isRunning()) {
            status = pipeline.processFrame();
            
            if (status != DW_SUCCESS && status != DW_TIME_OUT) {
                std::cerr << "ERROR: Pipeline processing failed: " 
                         << dwGetStatusName(status) << std::endl;
                break;
            }
            
            frameCount++;
            
            // Print statistics every 2 seconds
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                currentTime - lastStatsTime
            );
            
            if (elapsed.count() >= 2) {
                auto stats = pipeline.getStatistics();
                
                std::cout << "=== Pipeline Statistics ===" << std::endl;
                std::cout << "FPS: " << stats.currentFPS << std::endl;
                std::cout << "Total frames: " << stats.totalFramesCaptured << std::endl;
                std::cout << "Inferences completed: " << stats.totalInferencesCompleted << std::endl;
                std::cout << "Dropped frames: " << stats.droppedFrames << std::endl;
                std::cout << "Avg latency: " << stats.avgLatencyMs << " ms" << std::endl;
                std::cout << "=========================" << std::endl;
                
                lastStatsTime = currentTime;
            }
        }
        
        // Stop pipeline
        std::cout << "Stopping pipeline..." << std::endl;
        pipeline.stop();
        
        // Final statistics
        auto finalStats = pipeline.getStatistics();
        std::cout << "\n=== Final Statistics ===" << std::endl;
        std::cout << "Total frames processed: " << finalStats.totalFramesCaptured << std::endl;
        std::cout << "Total inferences: " << finalStats.totalInferencesCompleted << std::endl;
        std::cout << "Average FPS: " << finalStats.currentFPS << std::endl;
        std::cout << "=======================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "Pipeline terminated successfully." << std::endl;
    return 0;
}