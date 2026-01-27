#include "SimpleTracker.hpp"
#include "InterLidarICP.hpp"
#include <limits>
#include <algorithm>

// Define Track structure here to avoid circular dependency
struct SimpleTracker::Track
{
    int id;
    BoundingBox box;           // Current state
    int age;                   // Frames since last update
    int hits;                  // Number of successful associations
    int timeSinceUpdate;       // Frames since last detection match
    bool active;               // Whether track is still active
    
    Track(int trackId, const BoundingBox& initialBox)
        : id(trackId)
        , box(initialBox)
        , age(0)
        , hits(1)
        , timeSinceUpdate(0)
        , active(true)
    {
    }
};

// Implementation structure
struct SimpleTracker::Impl
{
    float iouThreshold;
    int maxAge;
    int minHits;
    int nextId;
    std::vector<Track> tracks;
    
    Impl(float iouThresh, int maxA, int minH)
        : iouThreshold(iouThresh)
        , maxAge(maxA)
        , minHits(minH)
        , nextId(1)
    {
    }
};

SimpleTracker::SimpleTracker(float iouThreshold, int maxAge, int minHits)
    : m_impl(std::make_unique<Impl>(iouThreshold, maxAge, minHits))
{
}

SimpleTracker::~SimpleTracker() = default;

float SimpleTracker::computeIOU(const BoundingBox& box1, const BoundingBox& box2) const
{
    // Compute BEV (Bird's Eye View) IOU for 3D boxes
    // Project boxes to 2D by ignoring height and computing 2D IOU
    
    // Convert rotation to radians if needed
    float cos1 = std::cos(box1.rotation);
    float sin1 = std::sin(box1.rotation);
    float cos2 = std::cos(box2.rotation);
    float sin2 = std::sin(box2.rotation);
    
    // Get box corners in local frame (centered at origin)
    float w1 = box1.width / 2.0f;
    float l1 = box1.length / 2.0f;
    float w2 = box2.width / 2.0f;
    float l2 = box2.length / 2.0f;
    
    // Compute rotated corners for box1
    float corners1[4][2] = {
        {box1.x + w1 * cos1 - l1 * sin1, box1.y + w1 * sin1 + l1 * cos1},
        {box1.x - w1 * cos1 - l1 * sin1, box1.y - w1 * sin1 + l1 * cos1},
        {box1.x - w1 * cos1 + l1 * sin1, box1.y - w1 * sin1 - l1 * cos1},
        {box1.x + w1 * cos1 + l1 * sin1, box1.y + w1 * sin1 - l1 * cos1}
    };
    
    // Compute rotated corners for box2
    float corners2[4][2] = {
        {box2.x + w2 * cos2 - l2 * sin2, box2.y + w2 * sin2 + l2 * cos2},
        {box2.x - w2 * cos2 - l2 * sin2, box2.y - w2 * sin2 + l2 * cos2},
        {box2.x - w2 * cos2 + l2 * sin2, box2.y - w2 * sin2 - l2 * cos2},
        {box2.x + w2 * cos2 + l2 * sin2, box2.y + w2 * sin2 - l2 * cos2}
    };
    
    // Find bounding rectangle for intersection
    float minX1 = std::min({corners1[0][0], corners1[1][0], corners1[2][0], corners1[3][0]});
    float maxX1 = std::max({corners1[0][0], corners1[1][0], corners1[2][0], corners1[3][0]});
    float minY1 = std::min({corners1[0][1], corners1[1][1], corners1[2][1], corners1[3][1]});
    float maxY1 = std::max({corners1[0][1], corners1[1][1], corners1[2][1], corners1[3][1]});
    
    float minX2 = std::min({corners2[0][0], corners2[1][0], corners2[2][0], corners2[3][0]});
    float maxX2 = std::max({corners2[0][0], corners2[1][0], corners2[2][0], corners2[3][0]});
    float minY2 = std::min({corners2[0][1], corners2[1][1], corners2[2][1], corners2[3][1]});
    float maxY2 = std::max({corners2[0][1], corners2[1][1], corners2[2][1], corners2[3][1]});
    
    // Compute intersection
    float interMinX = std::max(minX1, minX2);
    float interMaxX = std::min(maxX1, maxX2);
    float interMinY = std::max(minY1, minY2);
    float interMaxY = std::min(maxY1, maxY2);
    
    float interArea = 0.0f;
    if (interMaxX > interMinX && interMaxY > interMinY)
    {
        // Approximate intersection area (simplified - uses axis-aligned bounding box)
        interArea = (interMaxX - interMinX) * (interMaxY - interMinY);
    }
    
    // Compute union (approximate using axis-aligned bounding boxes)
    float area1 = (maxX1 - minX1) * (maxY1 - minY1);
    float area2 = (maxX2 - minX2) * (maxY2 - minY2);
    float unionArea = area1 + area2 - interArea;
    
    if (unionArea < 1e-6f)
        return 0.0f;
    
    return interArea / unionArea;
}

std::vector<std::vector<float>> SimpleTracker::computeCostMatrix(
    const std::vector<BoundingBox>& detections,
    const std::vector<Track>& tracks) const
{
    std::vector<std::vector<float>> costMatrix(tracks.size());
    
    for (size_t i = 0; i < tracks.size(); ++i)
    {
        costMatrix[i].resize(detections.size());
        for (size_t j = 0; j < detections.size(); ++j)
        {
            // Cost = 1 - IOU (we want to minimize cost, maximize IOU)
            float iou = computeIOU(tracks[i].box, detections[j]);
            costMatrix[i][j] = 1.0f - iou;
        }
    }
    
    return costMatrix;
}

std::vector<std::pair<int, int>> SimpleTracker::hungarianAssignment(
    const std::vector<std::vector<float>>& costMatrix) const
{
    // Simplified greedy assignment for small problems
    // For larger problems, use a proper Hungarian algorithm implementation
    std::vector<std::pair<int, int>> assignments;
    
    if (costMatrix.empty() || costMatrix[0].empty())
        return assignments;
    
    size_t n = costMatrix.size();
    size_t m = costMatrix[0].size();
    
    std::vector<bool> trackUsed(n, false);
    std::vector<bool> detUsed(m, false);
    
    // Create list of all possible assignments with costs
    std::vector<std::tuple<float, int, int>> candidates;
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            candidates.push_back({costMatrix[i][j], static_cast<int>(i), static_cast<int>(j)});
        }
    }
    
    // Sort by cost (ascending)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
    
    // Greedy assignment
    for (const auto& candidate : candidates)
    {
        float cost = std::get<0>(candidate);
        int trackIdx = std::get<1>(candidate);
        int detIdx = std::get<2>(candidate);
        
        // Only assign if both are unused and cost is below threshold
        if (!trackUsed[trackIdx] && !detUsed[detIdx] && cost < (1.0f - m_impl->iouThreshold))
        {
            assignments.push_back({trackIdx, detIdx});
            trackUsed[trackIdx] = true;
            detUsed[detIdx] = true;
        }
    }
    
    return assignments;
}

void SimpleTracker::predictTrack(Track& track) const
{
    // Simple constant velocity prediction
    // In a full implementation, you'd use Kalman filter
    // For now, we just increment age
    track.age++;
    track.timeSinceUpdate++;
}

void SimpleTracker::updateTrack(Track& track, const BoundingBox& detection) const
{
    // Update track with new detection (simple averaging)
    // In a full implementation, use Kalman filter update step
    track.box.x = 0.7f * track.box.x + 0.3f * detection.x;
    track.box.y = 0.7f * track.box.y + 0.3f * detection.y;
    track.box.z = 0.7f * track.box.z + 0.3f * detection.z;
    
    // Update dimensions (use detection directly for now)
    track.box.width = detection.width;
    track.box.length = detection.length;
    track.box.height = detection.height;
    track.box.rotation = detection.rotation;
    track.box.confidence = detection.confidence;
    track.box.classId = detection.classId;
    
    track.hits++;
    track.timeSinceUpdate = 0;
}

int SimpleTracker::update(std::vector<BoundingBox>& detections)
{
    int numTracksBefore = static_cast<int>(m_impl->tracks.size());
    
    // Predict all tracks forward
    for (auto& track : m_impl->tracks)
    {
        if (track.active)
        {
            predictTrack(track);
        }
    }
    
    // Remove inactive tracks
    size_t removedCount = m_impl->tracks.size();
    m_impl->tracks.erase(
        std::remove_if(m_impl->tracks.begin(), m_impl->tracks.end(),
                       [this](const Track& t) {
                           return !t.active || t.timeSinceUpdate > m_impl->maxAge;
                       }),
        m_impl->tracks.end());
    removedCount -= m_impl->tracks.size();
    
    if (detections.empty())
    {
        if (removedCount > 0) {
            std::cout << "[Tracker] No detections this frame. Removed " << removedCount 
                      << " inactive tracks. Active tracks: " << m_impl->tracks.size() << std::endl;
        }
        return static_cast<int>(m_impl->tracks.size());
    }
    
    // Get active tracks
    std::vector<Track> activeTracks;
    for (const auto& track : m_impl->tracks)
    {
        if (track.active)
        {
            activeTracks.push_back(track);
        }
    }
    
    if (activeTracks.empty())
    {
        // No existing tracks, create new ones for all detections
        std::cout << "[Tracker] First frame: Creating " << detections.size() 
                  << " new tracks" << std::endl;
        for (auto& det : detections)
        {
            det.trackId = m_impl->nextId++;
            m_impl->tracks.push_back(Track(det.trackId, det));
            std::cout << "[Tracker]   Created track ID=" << det.trackId 
                      << " for class=" << det.classId << " at (" 
                      << det.x << "," << det.y << "," << det.z << ")" << std::endl;
        }
        return static_cast<int>(m_impl->tracks.size());
    }
    
    // Compute cost matrix
    auto costMatrix = computeCostMatrix(detections, activeTracks);
    
    // Hungarian assignment
    auto assignments = hungarianAssignment(costMatrix);
    
    std::cout << "[Tracker] Frame update: " << detections.size() << " detections, " 
              << activeTracks.size() << " existing tracks, " << assignments.size() 
              << " matches" << std::endl;
    
    // Update matched tracks
    std::vector<bool> detMatched(detections.size(), false);
    for (const auto& assignment : assignments)
    {
        int trackIdx = assignment.first;
        int detIdx = assignment.second;
        
        // Find track in m_impl->tracks by ID
        for (auto& track : m_impl->tracks)
        {
            if (track.id == activeTracks[trackIdx].id)
            {
                updateTrack(track, detections[detIdx]);
                detections[detIdx].trackId = track.id;
                detMatched[detIdx] = true;
                std::cout << "[Tracker]   Matched detection " << detIdx << " to track ID=" 
                          << track.id << " (hits=" << track.hits << ", age=" << track.age << ")" << std::endl;
                break;
            }
        }
    }
    
    // Create new tracks for unmatched detections
    int newTracksCreated = 0;
    for (size_t i = 0; i < detections.size(); ++i)
    {
        if (!detMatched[i])
        {
            detections[i].trackId = m_impl->nextId++;
            m_impl->tracks.push_back(Track(detections[i].trackId, detections[i]));
            newTracksCreated++;
            std::cout << "[Tracker]   Created new track ID=" << detections[i].trackId 
                      << " for unmatched detection class=" << detections[i].classId 
                      << " at (" << detections[i].x << "," << detections[i].y << "," 
                      << detections[i].z << ")" << std::endl;
        }
    }
    
    // Mark tracks as inactive if they haven't been seen recently
    int tracksDeactivated = 0;
    for (auto& track : m_impl->tracks)
    {
        if (track.timeSinceUpdate > m_impl->maxAge)
        {
            track.active = false;
            tracksDeactivated++;
            std::cout << "[Tracker]   Deactivated track ID=" << track.id 
                      << " (timeSinceUpdate=" << track.timeSinceUpdate << ")" << std::endl;
        }
    }
    
    int numTracksAfter = static_cast<int>(m_impl->tracks.size());
    std::cout << "[Tracker] Summary: " << numTracksBefore << " -> " << numTracksAfter 
              << " tracks (" << newTracksCreated << " new, " << removedCount 
              << " removed, " << tracksDeactivated << " deactivated)" << std::endl;
    
    return numTracksAfter;
}

std::vector<SimpleTracker::Track> SimpleTracker::getActiveTracks() const
{
    std::vector<Track> active;
    for (const auto& track : m_impl->tracks)
    {
        if (track.active)
        {
            active.push_back(track);
        }
    }
    return active;
}

void SimpleTracker::reset()
{
    m_impl->tracks.clear();
    m_impl->nextId = 1;
}
