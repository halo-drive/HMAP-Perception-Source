/////////////////////////////////////////////////////////////////////////////////////////
// Global re-localization: ScanContext + GICP
/////////////////////////////////////////////////////////////////////////////////////////

#include "GlobalReloc.hpp"
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>
#include <iostream>

#include "Scancontext/Scancontext.h"

namespace dw_slam {

struct GlobalReloc::SCManagerHolder {
    SCManager manager;
};

GlobalReloc::GlobalReloc()
    : downsample_resolution_(0.2)
    , index_built_(false)
    , sc_holder_(std::make_unique<SCManagerHolder>())
{}

GlobalReloc::~GlobalReloc() = default;

void GlobalReloc::setDownsampleResolution(double resolution) {
    downsample_resolution_ = std::max(0.1, resolution);
}

void GlobalReloc::setKeyframes(const std::vector<KeyframeReloc>& keyframes) {
    keyframes_.clear();
    KeyMat polarcontext_invkeys_mat;
    std::vector<Eigen::MatrixXd> polarcontexts;
    SCManager& sc = sc_holder_->manager;
    for (size_t i = 0; i < keyframes.size(); i++) {
        const KeyframeReloc& kf = keyframes[i];
        if (!kf.cloud || kf.cloud->empty()) continue;
        keyframes_.push_back(kf);
        Eigen::MatrixXd scDesc = sc.makeScancontext(*kf.cloud);
        Eigen::MatrixXd ringkey = sc.makeRingkeyFromScancontext(scDesc);
        polarcontext_invkeys_mat.push_back(eig2stdvec(ringkey));
        polarcontexts.push_back(scDesc);
    }
    if (polarcontexts.empty()) {
        index_built_ = false;
        return;
    }
    sc.buildRingKeyKDTree(polarcontext_invkeys_mat, polarcontexts);
    index_built_ = true;
}

bool GlobalReloc::relocalize(const pcl::PointCloud<pcl::PointXYZI>::Ptr& scan,
                             Eigen::Matrix4d& pose_out) {
    if (!index_built_ || keyframes_.empty() || !scan || scan->empty()) {
        return false;
    }
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered = scan;
    if (downsample_resolution_ > 0.05) {
        filtered.reset(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::VoxelGrid<pcl::PointXYZI> voxel;
        voxel.setLeafSize(downsample_resolution_, downsample_resolution_, downsample_resolution_);
        voxel.setInputCloud(scan);
        voxel.filter(*filtered);
        if (filtered->empty()) filtered = scan;
    }
    SCManager& sc = sc_holder_->manager;
    Eigen::MatrixXd scQuery = sc.makeScancontext(*filtered);
    std::vector<float> ringkey = eig2stdvec(sc.makeRingkeyFromScancontext(scQuery));
    Eigen::MatrixXd sectorkey = sc.makeSectorkeyFromScancontext(scQuery);
    double score = 1.0;
    std::pair<int, float> match = sc.detectClosestMatch(scQuery, ringkey, sectorkey, score);
    if (match.first < 0 || score > sc.SC_DIST_THRES) {
        std::cout << "[GlobalReloc] No ScanContext match (best score=" << score
                  << ", threshold=" << sc.SC_DIST_THRES << ")" << std::endl;
        return false;
    }
    int kfIdx = match.first;
    float yawRad = match.second;
    std::cout << "[GlobalReloc] ScanContext match: keyframe " << kfIdx << ", score=" << score << std::endl;
    Eigen::Matrix4f initGuess = yaw2matrix(-yawRad);
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> gicp;
    gicp.setInputTarget(keyframes_[kfIdx].cloud);
    gicp.setInputSource(filtered);
    gicp.setMaximumIterations(50);
    pcl::PointCloud<pcl::PointXYZI> aligned;
    gicp.align(aligned, initGuess);
    if (!gicp.hasConverged()) {
        std::cout << "[GlobalReloc] GICP did not converge" << std::endl;
        return false;
    }
    double fitness = gicp.getFitnessScore(25.0);
    Eigen::Matrix4f delta = gicp.getFinalTransformation();
    if (fitness > 1.5) {
        std::cout << "[GlobalReloc] GICP fitness too high: " << fitness << " (threshold 1.5)" << std::endl;
        return false;
    }
    std::cout << "[GlobalReloc] Relocalization OK: keyframe " << kfIdx << ", fitness=" << fitness << std::endl;
    Eigen::Matrix4d delta_d = delta.cast<double>();
    pose_out = keyframes_[kfIdx].pose * delta_d;
    return true;
}

} // namespace dw_slam
