/////////////////////////////////////////////////////////////////////////////////////////
// Main Entry Point - Following Reference Pattern
/////////////////////////////////////////////////////////////////////////////////////////

#include "DrivableAreaSegmentation.hpp"

int main(int argc, const char** argv)
{
    ProgramArguments args(argc, argv,  // NOTE: No namespace qualifier
    {
        ProgramArguments::Option_t(
            "rig",
            (dw_samples::SamplesDataPath::get() + "/samples/sensors/camera/camera/rig.json").c_str(),
            "Rig configuration file"),
        ProgramArguments::Option_t(
            "model",
            "",
            "Path to TensorRT model (auto-detected if empty)")
    },
    "Drivable Area Segmentation - ResNet34-FCN Single Camera\n"
    "\nControls:\n"
    "  'P' - Print performance metrics\n");
    
    DrivableAreaSegmentation app(args);
    app.initializeWindow("Drivable Area Segmentation", 1280, 720, args.enabled("offscreen"));
    
    return app.run();
}