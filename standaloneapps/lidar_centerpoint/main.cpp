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
                              opt("maxIters", "50", "Number of ICP iterations to run (increased for better alignment)"),
                              opt("numFrames", "0", "Number of frames to process (0 = unlimited)"),
                              
                              // Processing parameters
                              opt("skipFrames", "5", "Number of initial frames to skip for sensor stabilization"),
                              
                              // Object detection parameters (CenterPoint)
                              opt("tensorRTEngine", "", "Legacy single-engine TensorRT path (unused when CenterPoint is enabled)"),
                              opt("pfeEngine", "", "Path to CenterPoint PFE TensorRT engine file"),
                              opt("rpnEngine", "", "Path to CenterPoint RPN TensorRT engine file"),
                              opt("objectDetection", "true", "Enable object detection (requires CenterPoint engines)"),
                              opt("minPoints", "120", "Minimum points required for valid detection (default: 120)"),
                              opt("bevVisualization", "false", "Enable BEV feature map visualization"),
                              opt("heatmapVisualization", "false", "Enable CenterPoint heatmap visualization"),

                              opt("freeSpace", "false", "Enable free space detection"),
                              opt("freeSpaceVisualization", "false", "Enable free space visualization"),
                              
                              // Ground plane parameters
                              opt("groundPlane", "false", "Enable ground plane extraction"),
                              opt("groundPlaneVisualization", "false", "Enable ground plane visualization (set to false if ground plane is incorrect)"),
                              
                              // Display parameters
                              opt("displayWindowHeight", "900", "Display window height"),
                              opt("displayWindowWidth", "1500", "Display window width"),
                              
                              // Debug options
                              opt("verbose", "false", "Enable verbose logging"),
                              opt("savePointClouds", "false", "Save point clouds to PLY files")
                          });

    // Validate arguments
    if (!args.parse(argc, argv)) {
        std::cerr << "Failed to parse command line arguments" << std::endl;
        return -1;
    }

    // Check if help was requested
    if (args.has("help")) {
        std::cout << "Inter-Lidar ICP Sample with Ground Plane Extraction and Object Detection" << std::endl;
        std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
        std::cout << std::endl;
        std::cout << "Required Options:" << std::endl;
        std::cout << "  --rigFile=FILE          Path to rig configuration file (default: rig.json)" << std::endl;
        std::cout << std::endl;
        std::cout << "ICP Options:" << std::endl;
        std::cout << "  --maxIters=N            Number of ICP iterations (default: 50, increased for better alignment)" << std::endl;
        std::cout << "  --numFrames=N           Number of frames to process (0 = unlimited)" << std::endl;
        std::cout << "  --skipFrames=N          Initial frames to skip (default: 5)" << std::endl;
        std::cout << std::endl;
        std::cout << "Object Detection Options:" << std::endl;
        std::cout << "  --tensorRTEngine=FILE  Path to TensorRT engine file for object detection" << std::endl;
        std::cout << "  --objectDetection=true Enable object detection (requires TensorRT engine)" << std::endl;
        std::cout << "  --minPoints=N          Minimum points required for valid detection (default: 120)" << std::endl;
        std::cout << std::endl;
        std::cout << "Ground Plane Options:" << std::endl;
        std::cout << "  --groundPlane=true     Enable ground plane extraction (default: true)" << std::endl;
        std::cout << "  --groundPlaneVisualization=true Enable ground plane visualization (default: true)" << std::endl;
        std::cout << "                         Set to false if ground plane visualization is incorrect" << std::endl;
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
        std::cout << "  with ground plane extraction, object detection, and visualization." << std::endl;
        std::cout << std::endl;
        std::cout << "  The workflow is:" << std::endl;
        std::cout << "  1. Get point clouds from both lidars" << std::endl;
        std::cout << "  2. Apply rig transformations to align them roughly" << std::endl;
        std::cout << "  3. Perform ICP to fine-tune alignment" << std::endl;
        std::cout << "  4. Stitch point clouds together" << std::endl;
        std::cout << "  5. Extract ground plane using CUDA pipeline" << std::endl;
        std::cout << "  6. Perform object detection using TensorRT on stitched point cloud" << std::endl;
        std::cout << "  7. Visualize results with ground plane and bounding boxes" << std::endl;
        std::cout << std::endl;
        std::cout << "Free Space Options:" << std::endl;
        std::cout << "  --freeSpace=true       Enable free space detection (default: true)" << std::endl;
        std::cout << "  --freeSpaceVisualization=true Enable free space visualization (default: true)" << std::endl;
        std::cout << std::endl;
        std::cout << "Ground Plane Extraction:" << std::endl;
        std::cout << "  - Uses CUDA pipeline for improved performance" << std::endl;
        std::cout << "  - RANSAC-based plane fitting with non-linear optimization" << std::endl;
        std::cout << "  - Box filter applied to focus on area around vehicle" << std::endl;
        std::cout << "  - Ground plane rendered as red mesh overlay (can be disabled)" << std::endl;
        std::cout << std::endl;
        std::cout << "Object Detection:" << std::endl;
        std::cout << "  - Uses TensorRT for high-performance neural network inference" << std::endl;
        std::cout << "  - Detects vehicles, pedestrians, and cyclists" << std::endl;
        std::cout << "  - Applies comprehensive filtering (NMS, size, point density, minimum point count)" << std::endl;
        std::cout << "  - Bounding boxes rendered with class-specific colors:" << std::endl;
        std::cout << "    * Vehicles: Red boxes" << std::endl;
        std::cout << "    * Pedestrians: Green boxes" << std::endl;
        std::cout << "    * Cyclists: Blue boxes" << std::endl;
        std::cout << "  - Only performs detection after initial ICP alignment is complete" << std::endl;
        std::cout << std::endl;
        std::cout << "Rig File Requirements:" << std::endl;
        std::cout << "  - Exactly 2 LiDAR sensors with 'lidar.socket' protocol" << std::endl;
        std::cout << "  - Proper sensor2Rig transformations defined" << std::endl;
        std::cout << std::endl;
        std::cout << "TensorRT Engine Requirements:" << std::endl;
        std::cout << "  - Pre-trained object detection model converted to TensorRT engine" << std::endl;
        std::cout << "  - Model should accept point cloud input (x,y,z,intensity)" << std::endl;
        std::cout << "  - Model should output bounding boxes with confidence scores" << std::endl;
        std::cout << std::endl;
        std::cout << "Window Layout:" << std::endl;
        std::cout << "  Full-screen:  ICP Alignment View - Green (LiDAR A) + Red (LiDAR B aligned) + Ground plane + Objects" << std::endl;
        std::cout << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "  Mouse: Rotate and zoom the 3D view" << std::endl;
        std::cout << "  ESC: Exit application" << std::endl;
        std::cout << "  SPACE: Pause/Resume processing" << std::endl;
        std::cout << std::endl;
        std::cout << "Output:" << std::endl;
        std::cout << "  - Real-time visualization of ICP alignment, ground plane, and object detection" << std::endl;
        std::cout << "  - Console output showing ICP success rate, ground plane detection, and object counts" << std::endl;
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
    std::cout << "=== Inter-Lidar ICP with Ground Plane Extraction and Object Detection ===" << std::endl;
    std::cout << "Rig File: " << rigFile << std::endl;
    std::cout << "Max ICP Iterations: " << args.get("maxIters") << std::endl;
    std::cout << "Number of Frames: " << (args.get("numFrames") == "0" ? "Unlimited" : args.get("numFrames")) << std::endl;
    std::cout << "Skip Initial Frames: " << args.get("skipFrames") << std::endl;
    std::cout << "Window Size: " << args.get("displayWindowWidth") << "x" << args.get("displayWindowHeight") << std::endl;
    std::cout << "Verbose Logging: " << (args.get("verbose") == "true" ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "Save Point Clouds: " << (args.get("savePointClouds") == "true" ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "Ground Plane Extraction: " << (args.get("groundPlane") == "true" ? "ENABLED" : "DISABLED") << " (CUDA Pipeline)" << std::endl;
    std::cout << "Ground Plane Visualization: " << (args.get("groundPlaneVisualization") == "true" ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "Object Detection: " << (args.get("objectDetection") == "true" ? "ENABLED" : "DISABLED") << std::endl;
    if (args.get("objectDetection") == "true") {
        std::cout << "  TensorRT Engine: " << (args.get("tensorRTEngine").empty() ? "NOT SPECIFIED" : args.get("tensorRTEngine")) << std::endl;
        std::cout << "  Minimum Points Threshold: " << args.get("minPoints") << std::endl;
    }
    std::cout << "====================================================================" << std::endl;

    try {
        // Create and initialize the sample
        InterLidarICP app(args);
        
        // Get window dimensions
        int32_t width  = std::atoi(args.get("displayWindowWidth").c_str());
        int32_t height = std::atoi(args.get("displayWindowHeight").c_str());
        
        // Initialize the window
        app.initializeWindow("Inter-Lidar ICP + Ground Plane Extraction + Object Detection", width, height, args.enabled("offscreen"));
        
        std::cout << "Starting inter-lidar ICP processing with ground plane extraction and object detection..." << std::endl;
        std::cout << "Waiting for sensor data from both lidars..." << std::endl;
        std::cout << "Features:" << std::endl;
        std::cout << "  - Real-time ICP alignment between two lidars" << std::endl;
        std::cout << "  - Ground plane extraction using CUDA pipeline" << std::endl;
        std::cout << "  - Ground plane visualization in brown" << std::endl;
        std::cout << "  - Object detection using TensorRT (vehicles, pedestrians, cyclists)" << std::endl;
        std::cout << "  - Bounding box visualization with class-specific colors" << std::endl;
        std::cout << "  - Single full-screen view: ICP alignment with ground plane and object detection" << std::endl;
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