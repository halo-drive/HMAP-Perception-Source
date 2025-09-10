/////////////////////////////////////////////////////////////////////////////////////////
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA CORPORATION & AFFILIATES assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA CORPORATION & AFFILIATES. No third party distribution is allowed unless
// expressly authorized by NVIDIA. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA CORPORATION & AFFILIATES products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA CORPORATION & AFFILIATES.
//
// SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#include "MultiCameraDNN.hpp"

#include <iostream>
#include <stdexcept>

using namespace dw_samples::common;

int main(int argc, const char** argv)
{
    // Define program arguments with default values
    ProgramArguments args(argc, argv, {
        ProgramArguments::Option_t(
            "rig", 
            (dw_samples::SamplesDataPath::get() + "/samples/sensors/camera/camera/rig_4cam.json").c_str(),
            "Rig configuration file path for multi-camera setup"
        ),
        // Future DNN model arguments (to be added in next stage)
        ProgramArguments::Option_t(
            "tensorRT_model", 
            (dw_samples::SamplesDataPath::get() + "/samples/detector/pomo_drivenet.bin").c_str(),
            "Path to pomo-drivenet TensorRT model file for 3-output inference (detection + dual segmentation)"
        ),
#ifdef VIBRANTE
        // NVIDIA Drive platform specific arguments
        ProgramArguments::Option_t(
            "cudla", 
            "0", 
            "Enable cuDLA acceleration (0=disabled, 1=enabled)"
        ),
        ProgramArguments::Option_t(
            "dla-engine", 
            "0", 
            "DLA engine number to use if cuDLA is enabled"
        ),
#endif
    }, 
    "Multi-Camera pomo-drivenet DNN Application\n"
    "\n"
    "This application demonstrates multi-camera processing with pomo-drivenet 3-output DNN model.\n"
    "Implements real-time object detection, drive area segmentation, and lane line segmentation\n"
    "across multiple camera streams using round-robin DNN processing for optimal performance.\n"
    "\n"
    "Model outputs:\n"
    "  - Detection: [1, 25200, 6] object bounding boxes with confidence scores\n"
    "  - Drive area segmentation: [1, 2, 640, 640] binary drivable area mask\n"
    "  - Lane line segmentation: [1, 2, 640, 640] binary lane marking mask\n"
    "\n"
    "Controls:\n"
    "  S - Take screenshot\n"
    "  SPACE - Show processing status\n"
    "  ESC - Exit application\n"
    );

    try {
        // Create application instance
        MultiCameraDNNApp app(args);
        
        // Initialize application window
        const uint32_t windowWidth = 1280;
        const uint32_t windowHeight = 720;
        const bool isOffscreen = args.enabled("offscreen");
        
        std::cout << "Initializing Multi-Camera DNN Application..." << std::endl;
        std::cout << "Window size: " << windowWidth << "x" << windowHeight << std::endl;
        std::cout << "Offscreen mode: " << (isOffscreen ? "enabled" : "disabled") << std::endl;
        
        app.initializeWindow("Multi-Camera DNN Application", windowWidth, windowHeight, isOffscreen);
        
        // Set processing frame rate for interactive mode
        if (!isOffscreen) {
            const uint32_t targetFPS = 30;
            app.setProcessRate(targetFPS);
            std::cout << "Target frame rate: " << targetFPS << " FPS" << std::endl;
        }
        
        // Display configuration information
        std::cout << "\n=== Configuration ===" << std::endl;
        std::cout << "Rig file: " << args.get("rig") << std::endl;
        
#ifdef VIBRANTE
        if (args.get("cudla") == "1") {
            std::cout << "cuDLA acceleration: enabled (engine " << args.get("dla-engine") << ")" << std::endl;
        } else {
            std::cout << "cuDLA acceleration: disabled" << std::endl;
        }
#endif
        
        std::cout << "=====================" << std::endl;
        std::cout << "\nStarting application main loop..." << std::endl;
        
        // Run main application loop
        int exitCode = app.run();
        
        if (exitCode == 0) {
            std::cout << "Application completed successfully." << std::endl;
        } else {
            std::cout << "Application exited with code: " << exitCode << std::endl;
        }
        
        return exitCode;
        
    } catch (const std::exception& e) {
        std::cerr << "Application failed with exception: " << e.what() << std::endl;
        return -1;
        
    } catch (...) {
        std::cerr << "Application failed with unknown exception." << std::endl;
        return -2;
    }
}