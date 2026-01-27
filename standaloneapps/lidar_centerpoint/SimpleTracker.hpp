#ifndef SIMPLE_TRACKER_HPP
#define SIMPLE_TRACKER_HPP

#include <vector>
#include <cmath>
#include <memory>

// Forward declaration - will be defined in InterLidarICP.hpp
struct BoundingBox;

/**
 * Simple SORT-style tracker for 3D bounding boxes
 * 
 * Implements a simplified version of Simple Online and Realtime Tracking (SORT)
 * adapted for 3D LiDAR detections. Uses IOU-based association and a constant
 * velocity motion model.
 */
class SimpleTracker
{
public:
    // Track structure (defined in .cpp to avoid circular dependency)
    struct Track;

    SimpleTracker(float iouThreshold = 0.3f, int maxAge = 3, int minHits = 1);
    ~SimpleTracker();

    /**
     * Update tracker with new detections
     * @param detections Input detections (will be modified with track IDs)
     * @return Number of active tracks
     */
    int update(std::vector<BoundingBox>& detections);

    /**
     * Get all active tracks
     */
    std::vector<Track> getActiveTracks() const;

    /**
     * Clear all tracks
     */
    void reset();

private:
    // Forward declaration for implementation details
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    
    /**
     * Compute 3D IOU between two bounding boxes (using BEV projection)
     */
    float computeIOU(const BoundingBox& box1, const BoundingBox& box2) const;

    /**
     * Compute cost matrix for Hungarian assignment
     */
    std::vector<std::vector<float>> computeCostMatrix(
        const std::vector<BoundingBox>& detections,
        const std::vector<Track>& tracks) const;

    /**
     * Simple Hungarian algorithm (greedy assignment for small problems)
     */
    std::vector<std::pair<int, int>> hungarianAssignment(
        const std::vector<std::vector<float>>& costMatrix) const;

    /**
     * Predict track state (constant velocity model)
     */
    void predictTrack(Track& track) const;

    /**
     * Update track with new detection
     */
    void updateTrack(Track& track, const BoundingBox& detection) const;
};

#endif // SIMPLE_TRACKER_HPP

