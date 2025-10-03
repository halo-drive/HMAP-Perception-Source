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
// SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#include <framework/SamplesDataPath.hpp>
#include <framework/ProgramArguments.hpp>
#include "YoloProcessing.hpp"

int main(int argc, const char** argv)
{
    ProgramArguments args(argc, argv,
    {
        ProgramArguments::Option_t("rig", (dw_samples::SamplesDataPath::get() + "/samples/sensors/camera/camera/rig_4cam.json").c_str(), "Rig configuration for multi-camera setup"),
#ifdef VIBRANTE
        ProgramArguments::Option_t("cudla", "0", "run inference on cuDLA (0=GPU, 1=DLA)"),
        ProgramArguments::Option_t("dla-engine", "0", "DLA engine number (0 or 1) if cudla=1"),
#endif
        ProgramArguments::Option_t("tensorRT_model", "", (std::string("Path to TensorRT model file. Default: ") + dw_samples::SamplesDataPath::get() + "/samples/detector/<gpu-architecture>/yolov3_640x640.bin").c_str())
    },
    "Multi-Camera Vehicle Detection and Type Classification with YOLO + VehicleTypeNet");
    
    DNNTensorSample app(args);
    app.initializeWindow("Multi-Camera AutoVehicleTypeNet", 1280, 800, args.enabled("offscreen"));
    
    if (!args.enabled("offscreen"))
        app.setProcessRate(30);
    
    return app.run();
}