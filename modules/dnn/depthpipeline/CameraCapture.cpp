// CameraCapture.cpp
#include "CameraCapture.hpp"
#include <framework/Checks.hpp>
#include <framework/Log.hpp>

#include <sstream>
#include <iostream>
#include <thread>      // ADD THIS
#include <chrono>      // ADD THIS

namespace depth_pipeline {

CameraCapture::CameraCapture(dwContextHandle_t context, dwSALHandle_t sal)
    : m_context(context)
    , m_sal(sal)
    , m_rig(DW_NULL_HANDLE)
    , m_cameraCount(0)
    , m_frameIdCounter(0)
{
    // Initialize camera descriptors
    for (uint32_t i = 0; i < 4; ++i) {
        m_cameras[i].sensor = DW_NULL_HANDLE;
        m_cameras[i].cameraId = i;
        m_cameras[i].active = false;
        m_imageRGBA[i] = DW_NULL_HANDLE;
    }
    
    // Initialize image properties to defaults
    memset(&m_imageProperties, 0, sizeof(dwImageProperties));
}

CameraCapture::~CameraCapture()
{
    // Release image handles
    for (uint32_t i = 0; i < m_cameraCount; ++i) {
        if (m_imageRGBA[i] != DW_NULL_HANDLE) {
            dwStatus status = dwImage_destroy(m_imageRGBA[i]);
            if (status != DW_SUCCESS) {
                std::cerr << "WARNING: Failed to destroy image " << i 
                         << ": " << dwGetStatusName(status) << std::endl;
            }
        }
    }
    
    // Release camera sensors
    for (uint32_t i = 0; i < m_cameraCount; ++i) {
        if (m_cameras[i].sensor != DW_NULL_HANDLE) {
            dwStatus status = dwSAL_releaseSensor(m_cameras[i].sensor);
            if (status != DW_SUCCESS) {
                std::cerr << "WARNING: Failed to release camera " << i 
                         << ": " << dwGetStatusName(status) << std::endl;
            }
        }
    }
    
    // Release rig configuration
    if (m_rig != DW_NULL_HANDLE) {
        dwStatus status = dwRig_release(m_rig);
        if (status != DW_SUCCESS) {
            std::cerr << "WARNING: Failed to release rig: " 
                     << dwGetStatusName(status) << std::endl;
        }
    }
}

dwStatus CameraCapture::initialize(const std::string& rigPath)
{
    dwStatus status;
    
    // Validate context and SAL handles
    if (m_context == DW_NULL_HANDLE || m_sal == DW_NULL_HANDLE) {
        std::cerr << "ERROR: Invalid context or SAL handle" << std::endl;
        return DW_INVALID_HANDLE;
    }
    
    // Load rig configuration
    std::cout << "Loading rig configuration: " << rigPath << std::endl;
    status = dwRig_initializeFromFile(&m_rig, m_context, rigPath.c_str());
    if (status != DW_SUCCESS) {
        std::cerr << "ERROR: Failed to load rig configuration: " 
                 << dwGetStatusName(status) << std::endl;
        return status;
    }
    
    // Query number of camera sensors in rig
    uint32_t cameraSensorCount = 0;
    status = dwRig_getSensorCountOfType(&cameraSensorCount, DW_SENSOR_CAMERA, m_rig);
    if (status != DW_SUCCESS) {
        std::cerr << "ERROR: Failed to get camera count from rig: " 
                 << dwGetStatusName(status) << std::endl;
        return status;
    }
    
    if (cameraSensorCount == 0) {
        std::cerr << "ERROR: No cameras found in rig configuration" << std::endl;
        return DW_INVALID_ARGUMENT;
    }
    
    if (cameraSensorCount > 4) {
        std::cerr << "WARNING: Rig contains " << cameraSensorCount 
                 << " cameras, but pipeline supports maximum 4. Using first 4." << std::endl;
        cameraSensorCount = 4;
    }
    
    std::cout << "Found " << cameraSensorCount << " cameras in rig" << std::endl;
    
    // Initialize each camera sensor
    for (uint32_t i = 0; i < cameraSensorCount; ++i) {
        CameraDescriptor& camDesc = m_cameras[i];
        
        // Get sensor index from rig
        uint32_t sensorIdx = 0;
        status = dwRig_findSensorByTypeIndex(&sensorIdx, DW_SENSOR_CAMERA, i, m_rig);
        if (status != DW_SUCCESS) {
            std::cerr << "ERROR: Failed to find camera sensor " << i 
                     << " in rig: " << dwGetStatusName(status) << std::endl;
            return status;
        }
        
        // Get sensor protocol (e.g., "camera.gmsl")
        const char* protocol = nullptr;
        status = dwRig_getSensorProtocol(&protocol, sensorIdx, m_rig);
        if (status != DW_SUCCESS) {
            std::cerr << "ERROR: Failed to get protocol for camera " << i 
                     << ": " << dwGetStatusName(status) << std::endl;
            return status;
        }
        
        // Get sensor parameters from rig (includes connection details)
        const char* parameters = nullptr;
        status = dwRig_getSensorParameterUpdatedPath(&parameters, sensorIdx, m_rig);
        if (status != DW_SUCCESS) {
            std::cerr << "ERROR: Failed to get parameters for camera " << i 
                     << ": " << dwGetStatusName(status) << std::endl;
            return status;
        }
        
        // Get sensor name for identification
        const char* sensorName = nullptr;
        status = dwRig_getSensorName(&sensorName, sensorIdx, m_rig);
        if (status == DW_SUCCESS && sensorName != nullptr) {
            camDesc.name = std::string(sensorName);
        } else {
            camDesc.name = "camera_" + std::to_string(i);
        }
        
        std::cout << "Initializing camera " << i << " (" << camDesc.name << ")" << std::endl;
        std::cout << "  Protocol: " << protocol << std::endl;
        std::cout << "  Parameters: " << parameters << std::endl;
        
        // Create sensor parameters structure
        dwSensorParams sensorParams{};
        sensorParams.protocol = protocol;
        sensorParams.parameters = parameters;
        
        // Create camera sensor
        status = dwSAL_createSensor(&camDesc.sensor, sensorParams, m_sal);
        if (status != DW_SUCCESS) {
            std::cerr << "ERROR: Failed to create camera sensor " << i 
                     << ": " << dwGetStatusName(status) << std::endl;
            return status;
        }
        
        // Query image properties from camera
        // We request CUDA RGBA format for direct GPU processing
        status = dwSensorCamera_getImageProperties(&camDesc.imageProps, 
                                                   DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8,
                                                   camDesc.sensor);
        if (status != DW_SUCCESS) {
            std::cerr << "ERROR: Failed to get image properties for camera " << i 
                     << ": " << dwGetStatusName(status) << std::endl;
            return status;
        }
        
        std::cout << "  Resolution: " << camDesc.imageProps.width 
                 << "x" << camDesc.imageProps.height << std::endl;
        std::cout << "  Format: " << camDesc.imageProps.format << std::endl;
        std::cout << "  Type: " << (camDesc.imageProps.type == DW_IMAGE_CUDA ? "CUDA" : "CPU") << std::endl;
        
        // Verify image properties are consistent across cameras
        if (i == 0) {
            // First camera defines the standard properties
            m_imageProperties = camDesc.imageProps;
        } else {
            // Subsequent cameras must match
            if (camDesc.imageProps.width != m_imageProperties.width ||
                camDesc.imageProps.height != m_imageProperties.height) {
                std::cerr << "WARNING: Camera " << i << " resolution mismatch. "
                         << "Expected " << m_imageProperties.width << "x" << m_imageProperties.height
                         << ", got " << camDesc.imageProps.width << "x" << camDesc.imageProps.height
                         << std::endl;
                // Continue anyway - inference engine will handle resizing
            }
        }
        
        // Create persistent image handle for frame conversion
        // This avoids repeated allocations during frame capture
        status = dwImage_create(&m_imageRGBA[i], camDesc.imageProps, m_context);
        if (status != DW_SUCCESS) {
            std::cerr << "ERROR: Failed to create image handle for camera " << i 
                     << ": " << dwGetStatusName(status) << std::endl;
            return status;
        }
        
        camDesc.cameraId = i;
        camDesc.active = true;
        m_cameraCount++;
    }
    
    std::cout << "Camera capture initialized with " << m_cameraCount << " cameras" << std::endl;
    std::cout << "Standard resolution: " << m_imageProperties.width 
             << "x" << m_imageProperties.height << std::endl;
    
    // Critical performance note for high-resolution cameras
    if (m_imageProperties.width > 2000 || m_imageProperties.height > 1500) {
        std::cout << "WARNING: High resolution cameras detected." << std::endl;
        std::cout << "  Native: " << m_imageProperties.width << "x" << m_imageProperties.height << std::endl;
        std::cout << "  Target: 924x518 (depth model input)" << std::endl;
        std::cout << "  Downsampling ratio: ~" 
                 << (float)(m_imageProperties.width * m_imageProperties.height) / (924.0f * 518.0f) 
                 << "x" << std::endl;
        std::cout << "  This will significantly impact preprocessing performance." << std::endl;
    }
    
    return DW_SUCCESS;
}

dwStatus CameraCapture::start()
{
    if (m_cameraCount == 0) {
        std::cerr << "ERROR: Cannot start cameras - no cameras initialized" << std::endl;
        return DW_INVALID_ARGUMENT;
    }
    
    std::cout << "Starting camera sensors..." << std::endl;
    
    // Start all camera sensors
    for (uint32_t i = 0; i < m_cameraCount; ++i) {
        if (!m_cameras[i].active) {
            continue;
        }
        
        dwStatus status = dwSensor_start(m_cameras[i].sensor);
        if (status != DW_SUCCESS) {
            std::cerr << "ERROR: Failed to start camera " << i 
                     << " (" << m_cameras[i].name << "): " 
                     << dwGetStatusName(status) << std::endl;
            
            // Mark camera as inactive but continue with others
            m_cameras[i].active = false;
            continue;
        }
        
        std::cout << "  Camera " << i << " (" << m_cameras[i].name << ") started" << std::endl;
    }
    
    // Verify at least one camera started successfully
    uint32_t activeCameras = 0;
    for (uint32_t i = 0; i < m_cameraCount; ++i) {
        if (m_cameras[i].active) {
            activeCameras++;
        }
    }
    
    if (activeCameras == 0) {
        std::cerr << "ERROR: No cameras started successfully" << std::endl;
        return DW_FAILURE;
    }
    
    std::cout << activeCameras << " of " << m_cameraCount << " cameras active" << std::endl;
    
    return DW_SUCCESS;
}

dwStatus CameraCapture::stop()
{
    std::cout << "Stopping camera sensors..." << std::endl;
    
    for (uint32_t i = 0; i < m_cameraCount; ++i) {
        if (m_cameras[i].sensor == DW_NULL_HANDLE) {
            continue;
        }
        
        dwStatus status = dwSensor_stop(m_cameras[i].sensor);
        if (status != DW_SUCCESS) {
            std::cerr << "WARNING: Failed to stop camera " << i 
                     << ": " << dwGetStatusName(status) << std::endl;
            // Continue stopping other cameras
        }
    }
    
    std::cout << "All cameras stopped" << std::endl;
    return DW_SUCCESS;
}

dwStatus CameraCapture::captureFrames(CaptureResult& result, uint32_t timeoutUs)
{
    // Initialize result structure
    result.validCameraCount = 0;
    result.frameId = m_frameIdCounter.fetch_add(1);
    result.status = DW_SUCCESS;
    
    for (uint32_t i = 0; i < 4; ++i) {
        result.frames[i] = DW_NULL_HANDLE;
        result.imagesRGBA[i] = DW_NULL_HANDLE;
    }
    
    if (m_cameraCount == 0) {
        result.status = DW_INVALID_ARGUMENT;
        return DW_INVALID_ARGUMENT;
    }
    
    // Capture frames from all active cameras
    for (uint32_t i = 0; i < m_cameraCount; ++i) {
        if (!m_cameras[i].active) {
            continue;
        }
        
        dwStatus status = DW_NOT_READY;
        uint32_t retryCount = 0;
        const uint32_t MAX_RETRIES = 3;
        
        // Retry loop for frame acquisition
        // Handle transient NOT_READY and END_OF_STREAM conditions
        while (retryCount < MAX_RETRIES) {
            status = dwSensorCamera_readFrame(&result.frames[i], timeoutUs, m_cameras[i].sensor);
            
            if (status == DW_SUCCESS) {
                break; // Frame acquired successfully
            }
            
            if (status == DW_END_OF_STREAM) {
                // For recorded data (video files), reset and retry
                std::cout << "Camera " << i << " reached end of stream, resetting..." << std::endl;
                dwStatus resetStatus = dwSensor_reset(m_cameras[i].sensor);
                if (resetStatus != DW_SUCCESS) {
                    std::cerr << "ERROR: Failed to reset camera " << i 
                             << ": " << dwGetStatusName(resetStatus) << std::endl;
                    m_cameras[i].active = false;
                    break;
                }
                retryCount++;
                continue;
            }
            
            if (status == DW_NOT_READY) {
                // Frame not yet available, retry
                retryCount++;
                std::this_thread::sleep_for(std::chrono::microseconds(1000));
                continue;
            }
            
            if (status == DW_TIME_OUT) {
                // Timeout waiting for frame
                std::cerr << "WARNING: Camera " << i << " frame timeout" << std::endl;
                break;
            }
            
            // Unrecoverable error
            std::cerr << "ERROR: Camera " << i << " readFrame failed: " 
                     << dwGetStatusName(status) << std::endl;
            m_cameras[i].active = false;
            break;
        }
        
        if (status != DW_SUCCESS) {
            // Frame acquisition failed - skip this camera for this iteration
            result.frames[i] = DW_NULL_HANDLE;
            continue;
        }
        
        // Extract CUDA RGBA image from frame
        dwImageHandle_t imageHandle = DW_NULL_HANDLE;
        status = dwSensorCamera_getImage(&imageHandle, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, 
                                        result.frames[i]);
        if (status != DW_SUCCESS) {
            std::cerr << "ERROR: Failed to get image from camera " << i 
                     << " frame: " << dwGetStatusName(status) << std::endl;
            
            // Return frame to sensor
            dwSensorCamera_returnFrame(&result.frames[i]);
            result.frames[i] = DW_NULL_HANDLE;
            continue;
        }
        
        // Copy/convert image to our persistent RGBA buffer
        // This ensures consistent format for downstream processing
        status = dwImage_copyConvert(m_imageRGBA[i], imageHandle, m_context);
        if (status != DW_SUCCESS) {
            std::cerr << "ERROR: Failed to copy/convert image from camera " << i 
                     << ": " << dwGetStatusName(status) << std::endl;
            
            dwSensorCamera_returnFrame(&result.frames[i]);
            result.frames[i] = DW_NULL_HANDLE;
            continue;
        }
        
        // Successfully captured and converted frame
        result.imagesRGBA[i] = m_imageRGBA[i];
        result.validCameraCount++;
    }
    
    // Determine overall capture status
    if (result.validCameraCount == 0) {
        result.status = DW_TIME_OUT;
        return DW_TIME_OUT;
    }
    
    // Partial success is acceptable
    result.status = DW_SUCCESS;
    return DW_SUCCESS;
}

dwStatus CameraCapture::returnFrames(const CaptureResult& result)
{
    dwStatus overallStatus = DW_SUCCESS;
    
    for (uint32_t i = 0; i < m_cameraCount; ++i) {
        if (result.frames[i] == DW_NULL_HANDLE) {
            continue;
        }
        
        // FIXED: Create mutable copy for API requirement
        dwCameraFrameHandle_t frameHandle = result.frames[i];
        dwStatus status = dwSensorCamera_returnFrame(&frameHandle);
        if (status != DW_SUCCESS) {
            std::cerr << "WARNING: Failed to return frame from camera " << i 
                     << ": " << dwGetStatusName(status) << std::endl;
            overallStatus = status;
            // Continue returning other frames despite error
        }
    }
    
    return overallStatus;
}

} // namespace depth_pipeline