/////////////////////////////////////////////////////////////////////////////////////////
// NVIDIA Proprietary Notice
// (This code is provided "AS IS" without warranty of any kind. Redistribution is not allowed.)
// SPDX-FileCopyrightText: Copyright (c) 2015-2024 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
/////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// CUDA and DriveWorks includes
#include <cuda_runtime.h>
#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/dnn/DNN.h>
#include <dw/dnn/dataconditioner/DataConditioner.h>
#include <dw/image/Image.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/camera/Camera.h>
#include <dwvisualization/core/RenderEngine.h>
#include <dwvisualization/core/Visualization.h>
#include <dwvisualization/image/Image.h>
#include <dwvisualization/interop/ImageStreamer.h>

// Framework includes
#include <framework/Checks.hpp>
#include <framework/CudaTimer.hpp>
#include <framework/DriveWorksSample.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/RenderUtils.hpp>
#include <framework/WindowGLFW.hpp>

using namespace dw_samples::common;

// USBObjectTrackerApp: Combines USB camera access (see sample for USB cameras)
// with object detection and tracking (see GMSL object tracker sample).
class USBObjectTrackerApp : public DriveWorksSample {
private:
    // ---------------------------
    // DriveWorks context and visualization
    // ---------------------------
    dwContextHandle_t                m_context       = DW_NULL_HANDLE;
    dwSALHandle_t                    m_sal           = DW_NULL_HANDLE;
    dwVisualizationContextHandle_t   m_viz           = DW_NULL_HANDLE;
    dwRenderEngineHandle_t           m_renderEngine  = DW_NULL_HANDLE;

    // ---------------------------
    // USB Camera sensor handles and streaming
    // ---------------------------
    dwSensorHandle_t                 m_camera        = DW_NULL_HANDLE;
    dwImageHandle_t                  m_cameraImage   = DW_NULL_HANDLE;
    dwImageHandle_t                  m_imageGL       = DW_NULL_HANDLE;
    dwImageStreamerHandle_t          m_streamerGL    = DW_NULL_HANDLE;
    dwCameraFrameHandle_t            m_frame         = DW_NULL_HANDLE;
    dwImageType                      m_cameraImageType = DW_IMAGE_CUDA;

    // ---------------------------
    // DNN and Data Conditioner (for object detection/tracking)
    // ---------------------------
    dwDNNHandle_t                    m_dnn           = DW_NULL_HANDLE;
    dwDataConditionerHandle_t        m_dataConditioner = DW_NULL_HANDLE;
    float*                           m_dnnInputDevice  = nullptr;
    float*                           m_dnnOutputsDevice[2] = { nullptr, nullptr };
    std::unique_ptr<float[]>         m_dnnOutputsHost[2]; // Host copies

    dwBlobSize                       m_networkInputDimensions;
    dwBlobSize                       m_networkOutputDimensions[2];
    uint32_t                         m_totalSizeInput  = 0;
    uint32_t                         m_totalSizesOutput[2] = { 0, 0 };

    // Detection region (crop from USB camera image used as network input)
    dwRect                           m_detectionRegion = {0, 0, 0, 0};

    // Object detection/tracking parameters
    static constexpr bool USE_YOLO = true;
    static constexpr float CONFIDENCE_THRESHOLD = 0.65f;  // Increased from 0.50f
    static constexpr float SCORE_THRESHOLD = 0.35f;       // Increased from 0.25f
    const std::string YOLO_CLASS_NAMES[80] = {"person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                              "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                                              "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
                                              "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                                              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                                              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                                              "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
                                              "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                                              "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                                              "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                              "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                              "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                                              "teddy bear", "hair drier", "toothbrush"};

    // Optional class filtering
    std::vector<std::string> m_activeClasses = {"person", "car", "truck", "bus", "bicycle", "motorcycle"};
    bool m_useClassFilter = true;
    
    // Data structures for storing detected boxes
    std::vector<dwBox2D>           m_detectedBoxList;
    std::vector<dwRectf>           m_detectedBoxListFloat;
    std::vector<std::string>       m_label;
    int32_t                        m_cvgIdx = 0, m_bboxIdx = 0;
    const uint32_t                 m_maxDetections = 1000U;
    static constexpr float         COVERAGE_THRESHOLD = 0.6f;

    // ---------------------------
    // CUDA stream
    // ---------------------------
    cudaStream_t                   m_cudaStream = 0;

public:
    USBObjectTrackerApp(const ProgramArguments& args)
        : DriveWorksSample(args)
    {}

    // onInitialize: Initialize DriveWorks context, USB camera sensor, DNN, and visualization.
    bool onInitialize() override {
        // Initialize DriveWorks context and SAL
        {
            CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
            CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));

            dwContextParameters sdkParams = {};
#ifdef VIBRANTE
            sdkParams.eglDisplay = getEGLDisplay();
#endif
            CHECK_DW_ERROR(dwInitialize(&m_context, DW_VERSION, &sdkParams));
            CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_context));
            CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_context));
        }

        // ---------------------------
        // Create USB Camera sensor (refer to the USB sample)
        // ---------------------------
        {
            dwSensorParams params{};
            params.protocol = "camera.usb";
            std::string parameters = "device=" + getArgument("device");
            const std::string& modeParam = getArgument("mode");
            if (!modeParam.empty()) {
                parameters += ",mode=" + modeParam;
            }
            params.parameters = parameters.c_str();

            CHECK_DW_ERROR(dwSAL_createSensor(&m_camera, params, m_sal));

            // Retrieve camera properties
            dwImageProperties cameraImageProps{};
            CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&cameraImageProps, DW_CAMERA_OUTPUT_NATIVE_PROCESSED, m_camera));
            m_cameraImageType = cameraImageProps.type;
            if (m_cameraImageType == DW_IMAGE_CUDA) {
                cameraImageProps.format = DW_IMAGE_FORMAT_RGBA_UINT8;
            }

            std::cout << "USB Camera image: " << cameraImageProps.width << "x" << cameraImageProps.height << std::endl;
            setWindowSize(cameraImageProps.width, cameraImageProps.height);

            // Initialize image streamer for OpenGL display
            CHECK_DW_ERROR(dwImageStreamerGL_initialize(&m_streamerGL, &cameraImageProps, DW_IMAGE_GL, m_context));

            // Start the sensor
            CHECK_DW_ERROR(dwSensor_start(m_camera));
        }

        // ---------------------------
        // Initialize DNN (object detection model)
        // ---------------------------
        {
            std::string tensorRTModel = getArgument("tensorRT_model");
            if (tensorRTModel.empty()) {
                // Default location if not provided
                tensorRTModel = dw_samples::SamplesDataPath::get() + "/samples/detector/usb/tensorRT_model.bin";
            }
            CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFile(&m_dnn, tensorRTModel.c_str(), nullptr,
                                                            DW_PROCESSOR_TYPE_GPU, m_context));
            CHECK_DW_ERROR(dwDNN_setCUDAStream(m_cudaStream, m_dnn));

            // Get input dimensions (e.g., 416x416 with 3 channels)
            CHECK_DW_ERROR(dwDNN_getInputSize(&m_networkInputDimensions, 0, m_dnn));
            m_totalSizeInput = m_networkInputDimensions.channels *
                               m_networkInputDimensions.height *
                               m_networkInputDimensions.width;
            CHECK_CUDA_ERROR(cudaMalloc((void**)&m_dnnInputDevice, sizeof(float) * m_totalSizeInput));

            // Get output dimensions for each output blob
            uint32_t i = 0;
            CHECK_DW_ERROR(dwDNN_getOutputSize(&m_networkOutputDimensions[i], i, m_dnn));
            m_totalSizesOutput[i] = m_networkOutputDimensions[i].channels *
                            m_networkOutputDimensions[i].height *
                            m_networkOutputDimensions[i].width;
            CHECK_CUDA_ERROR(cudaMalloc((void**)&m_dnnOutputsDevice[i], sizeof(float) * m_totalSizesOutput[i]));
            m_dnnOutputsHost[i].reset(new float[m_totalSizesOutput[i]]);
        
            
            // Initialize data conditioner
            dwDNNMetaData metadata;
            CHECK_DW_ERROR(dwDNN_getMetaData(&metadata, m_dnn));

            // Set proper data conditioner parameters based on available fields
            metadata.dataConditionerParams.ignoreAspectRatio = false;  // Preserve aspect ratio
            metadata.dataConditionerParams.scaleCoefficient = 1.0f/255.0f;  // Normalize [0-255] to [0-1]

            // Initialize all mean values to 0 (optional, already set by default)
            for (int i = 0; i < DW_MAX_IMAGE_PLANES; i++) {
                metadata.dataConditionerParams.meanValue[i] = 0.0f;
                metadata.dataConditionerParams.stdev[i] = 1.0f;  // No scaling per channel
            }

            CHECK_DW_ERROR(dwDataConditioner_initialize(&m_dataConditioner, &m_networkInputDimensions, 1,
                                           &metadata.dataConditionerParams, m_cudaStream, m_context));
        }
        
        // Improved detection region calculation with better letterboxing/pillarboxing
        float targetAspectRatio = static_cast<float>(m_networkInputDimensions.width) / 
                                  static_cast<float>(m_networkInputDimensions.height);
        float imageAspectRatio = static_cast<float>(getWindowWidth()) / 
                                 static_cast<float>(getWindowHeight());

        // Initialize the region to full frame
        m_detectionRegion.width = getWindowWidth();
        m_detectionRegion.height = getWindowHeight();
        m_detectionRegion.x = 0;
        m_detectionRegion.y = 0;

        // Apply letterboxing or pillarboxing to maintain aspect ratio
        if (targetAspectRatio > imageAspectRatio) {
            // Width-constrained - add padding to height (letterboxing)
            int targetHeight = static_cast<int>(getWindowWidth() / targetAspectRatio);
            m_detectionRegion.y = (getWindowHeight() - targetHeight) / 2;
            m_detectionRegion.height = targetHeight;
        } else {
            // Height-constrained - add padding to width (pillarboxing)
            int targetWidth = static_cast<int>(getWindowHeight() * targetAspectRatio);
            m_detectionRegion.x = (getWindowWidth() - targetWidth) / 2;
            m_detectionRegion.width = targetWidth;
        }

        // ---------------------------
        // Initialize Render Engine
        // ---------------------------
        {
            dwRenderEngineParams params{};
            CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&params, getWindowWidth(), getWindowHeight()));
            CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_viz));
        }

        return true;
    }

    // onProcess: Acquire frame from USB camera, run DNN inference, and interpret detections.
    void onProcess() override {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Read frame with timeout (similar to USB sample)
        const dwStatus result = dwSensorCamera_readFrame(&m_frame, 50000, m_camera);
        if (DW_TIME_OUT == result) {
            return;
        } else if (DW_NOT_AVAILABLE == result) {
            std::cerr << "Camera not available or not running" << std::endl;
            onRelease();
            return;
        } else if (DW_SUCCESS != result) {
            std::cerr << "Failed to get frame: " << dwGetStatusName(result) << std::endl;
            onRelease();
            return;
        }

        // Get the camera image (using CUDA format if available)
        dwCameraOutputType outputType = (m_cameraImageType == DW_IMAGE_CUDA) ?
                                        DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8 :
                                        DW_CAMERA_OUTPUT_NATIVE_PROCESSED;
        CHECK_DW_ERROR(dwSensorCamera_getImage(&m_cameraImage, outputType, m_frame));

        // Run data conditioner to prepare input for the network
        dwImageCUDA* imageCUDA = nullptr;
        CHECK_DW_ERROR(dwImage_getCUDA(&imageCUDA, m_cameraImage));
        CHECK_DW_ERROR(dwDataConditioner_prepareDataRaw(m_dnnInputDevice, &imageCUDA, 1,
                                                         &m_detectionRegion, cudaAddressModeClamp, m_dataConditioner));

        // Run DNN inference
        CHECK_DW_ERROR(dwDNN_inferRaw(m_dnnOutputsDevice, &m_dnnInputDevice, 1, m_dnn));

        // Copy output from GPU to host (using output0 for detection confidence/bounding boxes)
        // (For simplicity, this sample assumes a YOLO variant as in the GMSL sample.)
        CHECK_CUDA_ERROR(cudaMemcpy(m_dnnOutputsHost[0].get(), m_dnnOutputsDevice[0],
                                    sizeof(float) * m_totalSizesOutput[0], cudaMemcpyDeviceToHost));

        // Interpret network output to extract detected bounding boxes
        interpretOutput(m_dnnOutputsHost[0].get(), &m_detectionRegion);

        // Return the frame after processing
        if (m_frame)
            CHECK_DW_ERROR(dwSensorCamera_returnFrame(&m_frame));
    }

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int classID; 
};

float computeIoU(const Detection &a, const Detection &b)
{
    float overlapX1 = std::max(a.x1, b.x1);
    float overlapY1 = std::max(a.y1, b.y1);
    float overlapX2 = std::min(a.x2, b.x2);
    float overlapY2 = std::min(a.y2, b.y2);

    float overlapW = std::max(0.0f, overlapX2 - overlapX1);
    float overlapH = std::max(0.0f, overlapY2 - overlapY1);
    float overlapArea = overlapW * overlapH;

    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    float iou = overlapArea / (areaA + areaB - overlapArea);
    return iou;
}

std::vector<Detection> runNMS(const std::vector<Detection> &detections,
    float iouThreshold)
{
    // Sort by descending confidence
    std::vector<Detection> sorted(detections);
    std::sort(sorted.begin(), sorted.end(),
        [](const Detection &a, const Detection &b) {
        return a.score > b.score;
    });

    std::vector<Detection> result;
    // Keep track of boxes that survive
    for (size_t i = 0; i < sorted.size(); ++i)
        {
            const auto &det = sorted[i];
            bool keep = true;
        for (auto &r : result)
        {
            if (computeIoU(det, r) > iouThreshold)
                {
                keep = false;
                break;
                }   
        }
        if (keep){
            result.push_back(det);
        }
        }
    return result;
}

    // interpretOutput: Process DNN outputs (assuming YOLO format) to extract detections.
    // Enhanced interpretation of YOLOv8n outputs
    void interpretOutput(const float* outData, const dwRect* roi) {
        m_detectedBoxList.clear();
        m_detectedBoxListFloat.clear();
        m_label.clear();
    
        const int numAttrs = 84;        // 84 attributes per prediction
        const int numPredictions = 8400; // total predictions
    
        std::vector<Detection> rawDetections;  // collect them before NMS
    
        for (int i = 0; i < numPredictions; i++) {
            const float* pred = outData + i * numAttrs;
            float centerX = pred[0];
            float centerY = pred[1];
            float width   = pred[2];
            float height  = pred[3];
            float objScore = pred[4];
    
            // Find best class score among the remaining attributes
            int bestClass = -1;
            float bestScore = 0.0f;
            for (int j = 5; j < numAttrs; j++) {
                float score = pred[j];
                if (score > bestScore) {
                    bestScore = score;
                    bestClass = j - 5;  // class index starting at 0
                }
            }
    
            float confidence = objScore * bestScore;
            if (confidence > CONFIDENCE_THRESHOLD && width > 5 && height > 5) {
                // Convert center-based coords to box corners in [0,640]
                float x1 = centerX - width * 0.5f;
                float y1 = centerY - height * 0.5f;
                float x2 = centerX + width * 0.5f;
                float y2 = centerY + height * 0.5f;
    
                // Map coords from 640x640 -> original image
                float mappedX1, mappedY1, mappedX2, mappedY2;
                dwDataConditioner_outputPositionToInput(&mappedX1, &mappedY1, x1, y1, roi, m_dataConditioner);
                dwDataConditioner_outputPositionToInput(&mappedX2, &mappedY2, x2, y2, roi, m_dataConditioner);
    
                // Build detection
                Detection det;
                det.x1 = mappedX1;
                det.y1 = mappedY1;
                det.x2 = mappedX2;
                det.y2 = mappedY2;
                det.score = confidence;
                det.classID = bestClass;
                
                // Apply class filtering if enabled
                if (m_useClassFilter) {
                    bool classFound = false;
                    for (const auto& className : m_activeClasses) {
                        if (bestClass >= 0 && bestClass < 80 && 
                            className == YOLO_CLASS_NAMES[bestClass]) {
                            classFound = true;
                            break;
                        }
                    }
                    if (!classFound) continue;
                }
                
                rawDetections.push_back(det);
            }
        }
    
        // *** RUN NMS *** 
        float iouThreshold = 0.2f;  // Decreased from 0.45f for less overlap
        auto finalDetections = runNMS(rawDetections, iouThreshold);
    
        // Now build your bounding box vectors for rendering
        for (auto &det : finalDetections)
        {
            float w = det.x2 - det.x1;
            float h = det.y2 - det.y1;
            if (w <= 0 || h <= 0) continue;
    
            dwRectf bboxFloat { det.x1, det.y1, w, h };
            dwBox2D bbox;
            bbox.x = static_cast<int32_t>(std::round(bboxFloat.x));
            bbox.y = static_cast<int32_t>(std::round(bboxFloat.y));
            bbox.width  = static_cast<int32_t>(std::round(bboxFloat.width));
            bbox.height = static_cast<int32_t>(std::round(bboxFloat.height));
    
            m_detectedBoxList.push_back(bbox);
            m_detectedBoxListFloat.push_back(bboxFloat);
    
            if (det.classID >= 0 && det.classID < 80)
            {
                float pct = det.score * 100.0f;
                m_label.push_back(YOLO_CLASS_NAMES[det.classID] + " " + std::to_string(static_cast<int>(pct)) + "%");
            }
            else
            {
                m_label.push_back("unknown " + std::to_string(static_cast<int>(det.score * 100.0f)) + "%");
            }
        }
    }
    
    // onRender: Render the USB camera image with improved object detection overlays
    void onRender() override {
        if (m_cameraImage != nullptr) {
            // Send camera image to OpenGL stream
            CHECK_DW_ERROR(dwImageStreamerGL_producerSend(m_cameraImage, m_streamerGL));
            // Wait for the GL image to come out
            CHECK_DW_ERROR(dwImageStreamerGL_consumerReceive(&m_imageGL, 30000, m_streamerGL));
            if (m_imageGL) {
                CHECK_DW_ERROR(dwRenderEngine_setTile(0, m_renderEngine));
                CHECK_DW_ERROR(dwRenderEngine_resetTile(m_renderEngine));
                dwImageGL* frameGL = nullptr;
                CHECK_DW_ERROR(dwImage_getGL(&frameGL, m_imageGL));

                dwVector2f range { static_cast<float>(frameGL->prop.width), static_cast<float>(frameGL->prop.height) };
                CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));
                CHECK_DW_ERROR(dwRenderEngine_renderImage2D(frameGL, {0.0f, 0.0f, range.x, range.y}, m_renderEngine));

                // Render detection boxes with improved visualization
                if (!m_detectedBoxListFloat.empty()) {
                    size_t count = std::min(m_detectedBoxListFloat.size(), static_cast<size_t>(m_maxDetections));
                    
                    // Add semi-transparent fill for better visibility
                    CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_LIGHTGREY, m_renderEngine));
                    CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D,
                                     m_detectedBoxListFloat.data(), sizeof(dwRectf),
                                     0, count, m_renderEngine));
                    
                    // Render boxes with thicker lines and color coding
                    CHECK_DW_ERROR(dwRenderEngine_setLineWidth(3.0f, m_renderEngine));
                    
                    for (size_t i = 0; i < count; i++) {
                        // Set color based on confidence
                        float confidence = std::stof(m_label[i].substr(m_label[i].find_last_of(' ')+1));
                        confidence = confidence / 100.0f;  // Convert percentage to 0-1 range
                        
                        if (confidence > 0.85f) {
                            CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_GREEN, m_renderEngine));
                        } else if (confidence > 0.70f) {
                            CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_YELLOW, m_renderEngine));
                        } else {
                            CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_RED, m_renderEngine));
                        }
                        
                        // Render individual box with color
                        CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D,
                                                          &m_detectedBoxListFloat[i], sizeof(dwRectf),
                                                          0, 1, m_renderEngine));
                    }
                    
                    // Render class labels with improved visibility
                    for (size_t i = 0; i < count; i++) {
                        const dwRectf& box = m_detectedBoxListFloat[i];
                        const std::string& label = m_label[i];
                        
                        // Position label above the box
                        float labelX = box.x;
                        float labelY = box.y - 15.0f;
                        
                        // Draw text background
                        dwRectf textBg;
                        textBg.x = labelX - 2.0f;
                        textBg.y = labelY - 2.0f;
                        textBg.width = 8.0f * label.length();  // Approximate width
                        textBg.height = 18.0f;
                        
                        CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_BLACK, m_renderEngine));
                        CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D,
                                    &textBg, sizeof(dwRectf), 0, 1, m_renderEngine));
                        
                        // Draw text with larger font
                        CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_WHITE, m_renderEngine));
                        CHECK_DW_ERROR(dwRenderEngine_setPointSize(16.0f, m_renderEngine));
                        CHECK_DW_ERROR(dwRenderEngine_renderText2D(label.c_str(), {labelX, labelY}, m_renderEngine));
                    }
                }
            }
            // Return GL image to streamer and producer to wait for next frame.
            CHECK_DW_ERROR(dwImageStreamerGL_consumerReturn(&m_imageGL, m_streamerGL));
            CHECK_DW_ERROR(dwImageStreamerGL_producerReturn(nullptr, 32000, m_streamerGL));
        }
        renderutils::renderFPS(m_renderEngine, getCurrentFPS());
    }

    // onRelease: Clean up all resources.
    void onRelease() override {
        if (m_dnnOutputsDevice[0]) { CHECK_CUDA_ERROR(cudaFree(m_dnnOutputsDevice[0])); }
        if (m_dnnOutputsDevice[1]) { CHECK_CUDA_ERROR(cudaFree(m_dnnOutputsDevice[1])); }
        if (m_dnnInputDevice) { CHECK_CUDA_ERROR(cudaFree(m_dnnInputDevice)); }

        if (m_streamerGL != DW_NULL_HANDLE) {
            CHECK_DW_ERROR(dwImageStreamerGL_release(m_streamerGL));
        }
        if (m_camera) {
            CHECK_DW_ERROR(dwSensor_stop(m_camera));
            CHECK_DW_ERROR(dwSAL_releaseSensor(m_camera));
        }
        CHECK_DW_ERROR(dwDNN_release(m_dnn));
        CHECK_DW_ERROR(dwDataConditioner_release(m_dataConditioner));
        CHECK_DW_ERROR(dwRenderEngine_release(m_renderEngine));
        CHECK_DW_ERROR(dwSAL_release(m_sal));
        CHECK_DW_ERROR(dwVisualizationRelease(m_viz));
        CHECK_DW_ERROR(dwRelease(m_context));
        CHECK_DW_ERROR(dwLogger_release());
    }

    // onReset: Reset DNN and data conditioner.
    void onReset() override {
        CHECK_DW_ERROR(dwDNN_reset(m_dnn));
        CHECK_DW_ERROR(dwDataConditioner_reset(m_dataConditioner));
    }

    // onResizeWindow: Update render engine bounds.
    void onResizeWindow(int width, int height) override {
        CHECK_DW_ERROR(dwRenderEngine_reset(m_renderEngine));
        dwRectf rect = { 0, 0, static_cast<float>(width), static_cast<float>(height) };
        CHECK_DW_ERROR(dwRenderEngine_setBounds(rect, m_renderEngine));
    }
};

int main(int argc, const char** argv) {
    // Define command-line arguments.
    ProgramArguments args(argc, argv,
                          {
#ifdef VIBRANTE
                              ProgramArguments::Option_t("device", "1", "USB camera device id on Vibrante systems"),
#else
                              ProgramArguments::Option_t("device", "0", "USB camera device id"),
#endif
                              ProgramArguments::Option_t("mode", "0", "Camera mode (if applicable)"),
                              ProgramArguments::Option_t("tensorRT_model", "", "Path to TensorRT model binary file for object detection")
                          },
                          "USB Object Tracker: Uses a USB camera for object detection and tracking.");

    USBObjectTrackerApp app(args);
    app.initializeWindow("USB Object Tracker", 1280, 800, args.enabled("offscreen"));
    return app.run();
}
