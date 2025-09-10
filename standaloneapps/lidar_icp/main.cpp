#include <cstdint>
#include <cstdlib>
#include <string>
#include <iostream>

#include <framework/ProgramArguments.hpp>
#include <framework/SamplesDataPath.hpp>

#include "InterLidarICP.hpp"

//------------------------------------------------------------------------------
int32_t main(int32_t argc, const char** argv)
{
    typedef ProgramArguments::Option_t opt;

    // Default rig file path
    std::string defaultRigFile = "rig.json";
    
    ProgramArguments args(argc, argv,
                          {
                              // Rig configuration (REQUIRED)
                              opt("rigFile", defaultRigFile.c_str(), "Rig file containing sensor configurations"),
                              
                              // ICP parameters
                              opt("maxIters", "20", "Number of ICP iterations to run"),
                              opt("numFrames", "0", "Number of frames to process (0 = unlimited)"),
                              
                              // Processing parameters
                              opt("skipFrames", "5", "Number of initial frames to skip for sensor stabilization"),
                              
                              // Display parameters
                              opt("displayWindowHeight", "900", "Display window height"),
                              opt("displayWindowWidth", "1500", "Display window width"),
                              
                              // Debug options
                              opt("verbose", "false", "Enable verbose logging"),
                              opt("savePointClouds", "false", "Save point clouds to PLY files"),
                              // Operation modes
                              opt("stitchOnly", "false", "Disable ICP and ground plane; stitch using rig only")
                          });

    // Validate arguments
    if (!args.parse(argc, argv)) {
        std::cerr << "Failed to parse command line arguments" << std::endl;
        return -1;
    }

    // Check if help was requested
    if (args.has("help")) {
        std::cout << "Inter-Lidar ICP Sample with Ground Plane Extraction" << std::endl;
        std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
        std::cout << std::endl;
        std::cout << "Required Options:" << std::endl;
        std::cout << "  --rigFile=FILE          Path to rig configuration file (default: rig.json)" << std::endl;
        std::cout << std::endl;
        std::cout << "ICP Options:" << std::endl;
        std::cout << "  --maxIters=N            Number of ICP iterations (default: 20)" << std::endl;
        std::cout << "  --numFrames=N           Number of frames to process (0 = unlimited)" << std::endl;
        std::cout << "  --skipFrames=N          Initial frames to skip (default: 5)" << std::endl;
        std::cout << std::endl;
        std::cout << "Display Options:" << std::endl;
        std::cout << "  --displayWindowWidth=N  Window width (default: 1500)" << std::endl;
        std::cout << "  --displayWindowHeight=N Window height (default: 900)" << std::endl;
        std::cout << std::endl;
        std::cout << "Debug Options:" << std::endl;
        std::cout << "  --verbose=true          Enable verbose logging" << std::endl;
        std::cout << "  --savePointClouds=true  Save point clouds to PLY files" << std::endl;
        std::cout << std::endl;
        std::cout << "Description:" << std::endl;
        std::cout << "  This sample performs ICP alignment between two lidars in real-time" << std::endl;
        std::cout << "  with ground plane extraction and visualization." << std::endl;
        std::cout << std::endl;
        std::cout << "  The workflow is:" << std::endl;
        std::cout << "  1. Get point clouds from both lidars" << std::endl;
        std::cout << "  2. Apply rig transformations to align them roughly" << std::endl;
        std::cout << "  3. Perform ICP to fine-tune alignment" << std::endl;
        std::cout << "  4. Stitch point clouds together" << std::endl;
        std::cout << "  5. Extract ground plane using CUDA pipeline" << std::endl;
        std::cout << "  6. Visualize results with ground plane highlighted in red" << std::endl;
        std::cout << std::endl;
        std::cout << "Ground Plane Extraction:" << std::endl;
        std::cout << "  - Uses CUDA pipeline for improved performance" << std::endl;
        std::cout << "  - RANSAC-based plane fitting with non-linear optimization" << std::endl;
        std::cout << "  - Box filter applied to focus on area around vehicle" << std::endl;
        std::cout << "  - Ground plane rendered as red mesh overlay" << std::endl;
        std::cout << std::endl;
        std::cout << "Rig File Requirements:" << std::endl;
        std::cout << "  - Exactly 2 LiDAR sensors with 'lidar.socket' protocol" << std::endl;
        std::cout << "  - Proper sensor2Rig transformations defined" << std::endl;
        std::cout << std::endl;
        std::cout << "Window Layout:" << std::endl;
        std::cout << "  Top-left:     LiDAR A (Reference) - Green points" << std::endl;
        std::cout << "  Top-right:    LiDAR B (Source) - Orange points" << std::endl;
        std::cout << "  Bottom-left:  ICP Alignment - Green (A) + Red (B aligned) + Ground plane" << std::endl;
        std::cout << "  Bottom-right: Stitched Result - Cyan points + Ground plane" << std::endl;
        std::cout << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "  Mouse: Rotate and zoom the 3D view" << std::endl;
        std::cout << "  ESC: Exit application" << std::endl;
        std::cout << "  SPACE: Pause/Resume processing" << std::endl;
        std::cout << std::endl;
        std::cout << "Output:" << std::endl;
        std::cout << "  - Real-time visualization of ICP alignment and ground plane" << std::endl;
        std::cout << "  - Console output showing ICP success rate and ground plane detection" << std::endl;
        std::cout << "  - Optional PLY file output for point clouds" << std::endl;
        std::cout << "  - Optional detailed log file for analysis" << std::endl;
        return 0;
    }

    // Validate that rig file is specified
    std::string rigFile = args.get("rigFile");
    
    if (rigFile.empty()) {
        std::cerr << "Error: Rig file must be specified with --rigFile" << std::endl;
        std::cerr << "Use --help for usage information" << std::endl;
        return -1;
    }

    // Check if rig file exists
    std::ifstream file(rigFile);
    if (!file.good()) {
        std::cerr << "Error: Cannot open rig file: " << rigFile << std::endl;
        std::cerr << "Please make sure the file exists and is readable" << std::endl;
        return -1;
    }

    // Print configuration
    std::cout << "=== Inter-Lidar ICP with Ground Plane Extraction ===" << std::endl;
    std::cout << "Rig File: " << rigFile << std::endl;
    std::cout << "Max ICP Iterations: " << args.get("maxIters") << std::endl;
    std::cout << "Number of Frames: " << (args.get("numFrames") == "0" ? "Unlimited" : args.get("numFrames")) << std::endl;
    std::cout << "Skip Initial Frames: " << args.get("skipFrames") << std::endl;
    std::cout << "Window Size: " << args.get("displayWindowWidth") << "x" << args.get("displayWindowHeight") << std::endl;
    std::cout << "Verbose Logging: " << (args.get("verbose") == "true" ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "Save Point Clouds: " << (args.get("savePointClouds") == "true" ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "Ground Plane Extraction: ENABLED (CUDA Pipeline)" << std::endl;
    if (args.get("stitchOnly") == "true") {
        std::cout << "Mode: STITCH-ONLY (ICP and ground plane disabled)" << std::endl;
    }
    std::cout << "=========================================================" << std::endl;

    try {
        // Create and initialize the sample
        InterLidarICP app(args);
        
        // Get window dimensions
        int32_t width  = std::atoi(args.get("displayWindowWidth").c_str());
        int32_t height = std::atoi(args.get("displayWindowHeight").c_str());
        
        // Initialize the window
        app.initializeWindow("Inter-Lidar ICP + Ground Plane Extraction", width, height, args.enabled("offscreen"));
        
        std::cout << "Starting inter-lidar ICP processing with ground plane extraction..." << std::endl;
        std::cout << "Waiting for sensor data from both lidars..." << std::endl;
        std::cout << "Features:" << std::endl;
        std::cout << "  - Real-time ICP alignment between two lidars" << std::endl;
        std::cout << "  - Ground plane extraction using CUDA pipeline" << std::endl;
        std::cout << "  - Ground plane visualization in red" << std::endl;
        std::cout << "  - 4-panel view: Individual lidars + ICP alignment + Stitched result" << std::endl;
        std::cout << std::endl;
        std::cout << "Press SPACE to pause/resume, ESC to exit" << std::endl;
        
        // Run the application
        return app.run();
        
    } catch (const std::exception& e) {
        std::cerr << "Application error: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown application error occurred" << std::endl;
        return -1;
    }
}