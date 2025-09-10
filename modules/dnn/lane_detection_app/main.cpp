#include <string>

#include <framework/SamplesDataPath.hpp>
#include <framework/ProgramArguments.hpp>

#include "lanepomo.hpp"

int main(int argc, const char** argv)
{
    ProgramArguments args(argc, argv,
    {
#ifdef VIBRANTE
        ProgramArguments::Option_t("camera-type", "ar0231-rccb-bae-sf3324", "camera gmsl type"),
        ProgramArguments::Option_t("camera-group", "a", "input port"),
        ProgramArguments::Option_t("camera-index", "0", "camera index within the rig 0-3"),
        ProgramArguments::Option_t("input-type", "camera", "input type either video or camera"),
        ProgramArguments::Option_t("rig", (dw_samples::SamplesDataPath::get() + "/samples/sensors/camera/camera/rig_4cam.json").c_str(), "path to rig configuration file"),
#else
        ProgramArguments::Option_t("input-type", "video", "input type either video or camera"),
#endif
        ProgramArguments::Option_t("video", (dw_samples::SamplesDataPath::get() + "/cam_vids/cam0/video_0.mp4").c_str(), "path to video"),
        ProgramArguments::Option_t("tensorRT_model", "", (std::string("path to TensorRT model file. By default: ") + dw_samples::SamplesDataPath::get() + "/models/yololane.bin").c_str()),
        ProgramArguments::Option_t("dwTracePath", "", "path to trace file (optional)")
    },
    "Lane detection application that provides masks on the lines and segments the region between them.");

    LaneDetectionApplication app(args);
    
    app.initializeWindow("Lane Detection and Segmentation", 1280, 800, args.enabled("offscreen"));
    if (!args.enabled("offscreen"))
        app.setProcessRate(30);
    
    return app.run();
}