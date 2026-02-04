/////////////////////////////////////////////////////////////////////////////////////////
// Global re-localization: ScanContext (place recognition) + GICP refinement
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_FASTLIO_GLOBAL_RELOC_HPP_
#define DW_FASTLIO_GLOBAL_RELOC_HPP_

#include <vector>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace dw_slam {

/** Lightweight keyframe: pose + point cloud (sensor frame). */
struct KeyframeReloc {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix4d pose;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
};

/** Global re-localization: build ScanContext index from keyframes, find pose for a scan. */
class GlobalReloc {
public:
    GlobalReloc();
    ~GlobalReloc();

    void setDownsampleResolution(double resolution);
    void setKeyframes(const std::vector<KeyframeReloc>& keyframes);
    bool relocalize(const pcl::PointCloud<pcl::PointXYZI>::Ptr& scan, Eigen::Matrix4d& pose_out);
    size_t keyframeCount() const { return keyframes_.size(); }

private:
    std::vector<KeyframeReloc> keyframes_;
    double downsample_resolution_;
    bool index_built_;
    struct SCManagerHolder;
    std::unique_ptr<SCManagerHolder> sc_holder_;
};

} // namespace dw_slam

#endif
