#pragma once

#include <dw/core/context/Context.h>
#include <dw/sensors/camera/Camera.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/rig/Rig.h>
#include <dw/image/Image.h>

#include <atomic>
#include <memory>
#include <vector>
#include <string>

namespace depth_pipeline {

/**
 * @brief Camera capture module - manages multi-camera frame acquisition
 * 
 * Responsibilities:
 * - Initialize cameras from rig configuration
 * - Acquire frames from all cameras in parallel
 * - Provide CUDA RGBA images for downstream processing
 * - Handle camera lifecycle (start/stop/reset)
 */
class CameraCapture {
public:
    struct CameraDescriptor {
        dwSensorHandle_t sensor;
        dwImageProperties imageProps;
        uint32_t cameraId;
        std::string name;
        bool active;
    };
    
    struct CaptureResult {
        dwCameraFrameHandle_t frames[4];  // Up to 4 cameras
        dwImageHandle_t imagesRGBA[4];    // Pre-converted RGBA images
        uint64_t frameId;
        uint32_t validCameraCount;
        dwStatus status;
    };

    /**
     * @brief Constructor
     * @param context DriveWorks context handle
     * @param sal Sensor abstraction layer handle
     */
    CameraCapture(dwContextHandle_t context, dwSALHandle_t sal);
    
    /**
     * @brief Destructor - releases all camera resources
     */
    ~CameraCapture();
    
    /**
     * @brief Initialize cameras from rig configuration file
     * @param rigPath Path to rig JSON configuration
     * @return DW_SUCCESS on success, error code otherwise
     */
    dwStatus initialize(const std::string& rigPath);
    
    /**
     * @brief Start all camera sensors
     * @return DW_SUCCESS on success, error code otherwise
     */
    dwStatus start();
    
    /**
     * @brief Stop all camera sensors
     * @return DW_SUCCESS on success, error code otherwise
     */
    dwStatus stop();
    
    /**
     * @brief Capture frames from all active cameras
     * @param result Output capture result containing frames and images
     * @param timeoutUs Timeout in microseconds (default: 33333 = 30fps)
     * @return DW_SUCCESS if at least one frame captured
     */
    dwStatus captureFrames(CaptureResult& result, uint32_t timeoutUs = 33333);
    
    /**
     * @brief Return captured frames to camera sensors
     * @param result Capture result to return
     * @return DW_SUCCESS on success
     */
    dwStatus returnFrames(const CaptureResult& result);
    
    /**
     * @brief Get number of active cameras
     */
    uint32_t getCameraCount() const { return m_cameraCount; }
    
    /**
     * @brief Get camera descriptor by index
     */
    const CameraDescriptor& getCameraDescriptor(uint32_t idx) const {
        return m_cameras[idx];
    }
    
    /**
     * @brief Get image properties (same for all cameras)
     */
    const dwImageProperties& getImageProperties() const {
        return m_imageProperties;
    }

private:
    dwContextHandle_t m_context;
    dwSALHandle_t m_sal;
    dwRigHandle_t m_rig;
    
    uint32_t m_cameraCount;
    CameraDescriptor m_cameras[4];
    dwImageHandle_t m_imageRGBA[4];  // Persistent image handles
    dwImageProperties m_imageProperties;
    
    std::atomic<uint64_t> m_frameIdCounter;
    
    // Non-copyable
    CameraCapture(const CameraCapture&) = delete;
    CameraCapture& operator=(const CameraCapture&) = delete;
};

} // namespace depth_pipeline