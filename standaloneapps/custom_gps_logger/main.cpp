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
// SPDX-FileCopyrightText: Copyright (c) 2015-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <csignal>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cmath>

#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/core/signal/SignalStatus.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/gps/GPS.h>
#include <dw/sensors/sensormanager/SensorManager.h>
#include <dw/sensors/sensormanager/SensorManagerConstants.h>
#include <dw/rig/Rig.h>

#include <framework/Checks.hpp>
#include <framework/Log.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/SamplesDataPath.hpp>

//------------------------------------------------------------------------------
// Variables
//------------------------------------------------------------------------------
static bool gRun          = true;
static bool gDualMode     = false;
static bool enableDumpAll = false;

static std::unordered_map<dwSensorErrorID, std::string> gErrorStrings = {
    {DW_SENSORS_ERROR_CODE_GPS_MODE, "GPS sensor working in wrong modes"},
    {DW_SENSORS_ERROR_CODE_GPS_ACCURACY, "GPS sensor is not working in most accurate mode"},
};

//------------------------------------------------------------------------------
// GPS Data Structures
//------------------------------------------------------------------------------
struct GPSData {
    uint64_t timestamp_us;
    double latitude;
    double longitude;
    double altitude;
    double course;
    double speed;
    double climb;
    double hdop;
    double vdop;
    double pdop;
    double hacc;
    double vacc;
    uint64_t utcTimeUs;
    uint32_t satelliteCount;
    uint32_t fixStatus;
    uint32_t timestampQuality;
    uint32_t mode;
    uint32_t errors;
};

struct GPSComparisonResult {
    double positionError_m;
    double altitudeError_m;
    double speedError_ms;
    double courseError_deg;
    uint64_t timestamp_diff_us;
    size_t closestReferenceIdx;
    double distanceAlongPath_m;
    bool isOnPath;
};

struct PathAnalysis {
    double totalDistance_m;
    double averagePositionError_m;
    double maxPositionError_m;
    double pathFollowingAccuracy_percent;
    size_t totalPoints;
    size_t pointsOnPath;
    bool isStationary;
    double pathProgress_percent;
    double referencePathDistance_m;
};

struct GPSComparisonData {
    uint64_t timestamp_us;
    double current_latitude;
    double current_longitude;
    double reference_latitude;
    double reference_longitude;
    double positionError_m;
    double altitudeError_m;
    double speedError_ms;
    double courseError_deg;
    size_t closestReferenceIdx;
    double distanceAlongPath_m;
    bool isOnPath;
};

//------------------------------------------------------------------------------
void sig_int_handler(int)
{
    gRun = false;
}

void dumpHeader()
{
    std::cout << "dump_dwGPSFrame,"
              << "timestamp_us,"
              << "latitude,"
              << "longitude,"
              << "altitude,"
              << "course,"
              << "speed,"
              << "climb,"
              << "hdop,"
              << "vdop,"
              << "pdop,"
              << "hacc,"
              << "vacc,"
              << "utcTimeUs,"
              << "satelliteCount,"
              << "fixStatus,"
              << "timestampQuality,"
              << "mode,"
              << std::endl;
}

void dumpAll(const dwGPSFrame gpsFrame)
{
    std::cout << "dump_dwGPSFrame,"
              << std::fixed << std::setprecision(8)
              << gpsFrame.timestamp_us << ","
              << gpsFrame.latitude << ","
              << gpsFrame.longitude << ","
              << gpsFrame.altitude << ","
              << gpsFrame.course << ","
              << gpsFrame.speed << ","
              << gpsFrame.climb << ","
              << gpsFrame.hdop << ","
              << gpsFrame.vdop << ","
              << gpsFrame.pdop << ","
              << gpsFrame.hacc << ","
              << gpsFrame.vacc << ","
              << gpsFrame.utcTimeUs << ","
              << static_cast<uint32_t>(gpsFrame.satelliteCount) << ","
              << gpsFrame.fixStatus << ","
              << gpsFrame.timestampQuality << ","
              << gpsFrame.mode << ","
              << std::endl;
}

//------------------------------------------------------------------------------
// GPS Data Conversion and Utilities
//------------------------------------------------------------------------------
GPSData convertGPSFrame(const dwGPSFrame& frame) {
    GPSData data;
    data.timestamp_us = frame.timestamp_us;
    data.latitude = frame.latitude;
    data.longitude = frame.longitude;
    data.altitude = frame.altitude;
    data.course = frame.course;
    data.speed = frame.speed;
    data.climb = frame.climb;
    data.hdop = frame.hdop;
    data.vdop = frame.vdop;
    data.pdop = frame.pdop;
    data.hacc = frame.hacc;
    data.vacc = frame.vacc;
    data.utcTimeUs = frame.utcTimeUs;
    data.satelliteCount = static_cast<uint32_t>(frame.satelliteCount);
    data.fixStatus = frame.fixStatus;
    data.timestampQuality = frame.timestampQuality;
    data.mode = frame.mode;
    data.errors = frame.errors;
    return data;
}

// Calculate distance between two GPS coordinates in meters
double calculateDistance(double lat1, double lon1, double lat2, double lon2) {
    const double R = 6371000; // Earth radius in meters
    double dlat = (lat2 - lat1) * M_PI / 180.0;
    double dlon = (lon2 - lon1) * M_PI / 180.0;
    double a = sin(dlat/2) * sin(dlat/2) + cos(lat1 * M_PI / 180.0) * cos(lat2 * M_PI / 180.0) * sin(dlon/2) * sin(dlon/2);
    double c = 2 * atan2(sqrt(a), sqrt(1-a));
    return R * c;
}

// Save GPS data to JSON file
void saveGPSToJSON(const std::vector<GPSData>& gpsData, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
        return;
    }
    
    file << "{\n";
    file << "  \"gps_data\": [\n";
    
    for (size_t i = 0; i < gpsData.size(); ++i) {
        const GPSData& data = gpsData[i];
        file << "    {\n";
        file << "      \"timestamp_us\": " << data.timestamp_us << ",\n";
        file << "      \"latitude\": " << std::fixed << std::setprecision(10) << data.latitude << ",\n";
        file << "      \"longitude\": " << std::fixed << std::setprecision(10) << data.longitude << ",\n";
        file << "      \"altitude\": " << data.altitude << ",\n";
        file << "      \"course\": " << data.course << ",\n";
        file << "      \"speed\": " << data.speed << ",\n";
        file << "      \"climb\": " << data.climb << ",\n";
        file << "      \"hdop\": " << data.hdop << ",\n";
        file << "      \"vdop\": " << data.vdop << ",\n";
        file << "      \"pdop\": " << data.pdop << ",\n";
        file << "      \"hacc\": " << data.hacc << ",\n";
        file << "      \"vacc\": " << data.vacc << ",\n";
        file << "      \"utcTimeUs\": " << data.utcTimeUs << ",\n";
        file << "      \"satelliteCount\": " << data.satelliteCount << ",\n";
        file << "      \"fixStatus\": " << data.fixStatus << ",\n";
        file << "      \"timestampQuality\": " << data.timestampQuality << ",\n";
        file << "      \"mode\": " << data.mode << ",\n";
        file << "      \"errors\": " << data.errors << "\n";
        file << "    }";
        
        if (i < gpsData.size() - 1) {
            file << ",";
        }
        file << "\n";
    }
    
    file << "  ]\n";
    file << "}\n";
    file.close();
    
    std::cout << "GPS data saved to: " << filename << " (" << gpsData.size() << " records)" << std::endl;
}

// Save GPS comparison data to JSON file
void saveComparisonToJSON(const std::vector<GPSComparisonData>& comparisonData, const std::string& filename, const std::string& mainFile) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
        return;
    }
    
    file << "{\n";
    file << "  \"main_file\": \"" << mainFile << "\",\n";
    file << "  \"path_comparison_analysis\": [\n";
    
    for (size_t i = 0; i < comparisonData.size(); ++i) {
        const GPSComparisonData& data = comparisonData[i];
        file << "    {\n";
        file << "      \"timestamp_us\": " << data.timestamp_us << ",\n";
        file << "      \"current_position\": {\n";
        file << "        \"latitude\": " << std::fixed << std::setprecision(10) << data.current_latitude << ",\n";
        file << "        \"longitude\": " << std::fixed << std::setprecision(10) << data.current_longitude << "\n";
        file << "      },\n";
        file << "      \"reference_position\": {\n";
        file << "        \"latitude\": " << std::fixed << std::setprecision(10) << data.reference_latitude << ",\n";
        file << "        \"longitude\": " << std::fixed << std::setprecision(10) << data.reference_longitude << "\n";
        file << "      },\n";
        file << "      \"errors\": {\n";
        file << "        \"position_error_m\": " << std::fixed << std::setprecision(3) << data.positionError_m << ",\n";
        file << "        \"altitude_error_m\": " << std::fixed << std::setprecision(3) << data.altitudeError_m << ",\n";
        file << "        \"speed_error_ms\": " << std::fixed << std::setprecision(3) << data.speedError_ms << ",\n";
        file << "        \"course_error_deg\": " << std::fixed << std::setprecision(3) << data.courseError_deg << "\n";
        file << "      },\n";
        file << "      \"path_analysis\": {\n";
        file << "        \"closest_reference_idx\": " << data.closestReferenceIdx << ",\n";
        file << "        \"distance_along_path_m\": " << std::fixed << std::setprecision(3) << data.distanceAlongPath_m << ",\n";
        file << "        \"is_on_path\": " << (data.isOnPath ? "true" : "false") << "\n";
        file << "      }\n";
        file << "    }";
        
        if (i < comparisonData.size() - 1) {
            file << ",";
        }
        file << "\n";
    }
    
    file << "  ]\n";
    file << "}\n";
    file.close();
    
    std::cout << "GPS comparison data saved to: " << filename << " (" << comparisonData.size() << " records)" << std::endl;
}

// Save path analysis to TXT file
void savePathAnalysisToTXT(const PathAnalysis& analysis, const std::string& filename, const std::string& referenceFile, const std::string& currentFile) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
        return;
    }
    
    file << "========================================\n";
    file << "PATH FOLLOWING ANALYSIS REPORT\n";
    file << "========================================\n\n";
    
    file << "Reference File: " << referenceFile << "\n";
    file << "Current File: " << currentFile << "\n";
    file << "Analysis Date: " << std::time(nullptr) << "\n\n";
    
    file << "PATH STATISTICS:\n";
    file << "---------------\n";
    file << "Reference path points: " << analysis.totalPoints << "\n";
    file << "Current path points: " << analysis.totalPoints << "\n";
    file << "Reference path distance: " << std::fixed << std::setprecision(2) 
         << analysis.referencePathDistance_m << " meters\n";
    file << "Total distance traveled: " << std::fixed << std::setprecision(2) 
         << analysis.totalDistance_m << " meters\n";
    file << "Path progress: " << std::fixed << std::setprecision(1) 
         << analysis.pathProgress_percent << "%\n";
    file << "Vehicle status: " << (analysis.isStationary ? "STATIONARY" : "MOVING") << "\n\n";
    
    file << "ERROR ANALYSIS:\n";
    file << "--------------\n";
    file << "Average position error: " << std::fixed << std::setprecision(3) 
         << analysis.averagePositionError_m << " meters\n";
    file << "Maximum position error: " << std::fixed << std::setprecision(3) 
         << analysis.maxPositionError_m << " meters\n\n";
    
    file << "PATH FOLLOWING PERFORMANCE:\n";
    file << "---------------------------\n";
    file << "Path following accuracy: " << std::fixed << std::setprecision(1) 
         << analysis.pathFollowingAccuracy_percent << "%\n";
    file << "Points on path (within 10m): " << analysis.pointsOnPath << "/" 
         << analysis.totalPoints << "\n\n";
    
    file << "RESULT:\n";
    file << "-------\n";
    if (analysis.isStationary) {
        file << "STATIONARY: Vehicle is stationary - Path following analysis not applicable\n";
        file << "NOTE: To test path following, record a path while moving, then follow it\n";
    } else if (analysis.pathFollowingAccuracy_percent >= 80.0) {
        file << "SUCCESS: Successfully following the same path!\n";
    } else if (analysis.pathFollowingAccuracy_percent >= 60.0) {
        file << "PARTIAL: Partially following the path (some deviation)\n";
    } else {
        file << "FAILED: Not following the same path (significant deviation)\n";
    }
    
    file << "\n========================================\n";
    file.close();
    
    std::cout << "Path analysis saved to: " << filename << std::endl;
}

// Load GPS data from JSON file
std::vector<GPSData> loadGPSFromJSON(const std::string& filename) {
    std::vector<GPSData> gpsData;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for reading: " << filename << std::endl;
        return gpsData;
    }
    
    // Simple JSON parsing for GPS data
    std::string line;
    bool inDataArray = false;
    bool inRecord = false;
    GPSData currentData;
    
    while (std::getline(file, line)) {
        // Remove whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        if (line.find("\"gps_data\": [") != std::string::npos) {
            inDataArray = true;
            continue;
        }
        
        if (inDataArray && line == "]") {
            break;
        }
        
        if (inDataArray && line.find("{") != std::string::npos) {
            inRecord = true;
            currentData = GPSData{}; // Reset current data
            continue;
        }
        
        if (inRecord && line.find("}") != std::string::npos) {
            inRecord = false;
            gpsData.push_back(currentData);
            continue;
        }
        
        if (inRecord) {
            // Parse individual fields
            if (line.find("\"timestamp_us\":") != std::string::npos) {
                size_t pos = line.find(":");
                std::string value = line.substr(pos + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t,") + 1);
                currentData.timestamp_us = std::stoull(value);
            }
            else if (line.find("\"latitude\":") != std::string::npos) {
                size_t pos = line.find(":");
                std::string value = line.substr(pos + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t,") + 1);
                currentData.latitude = std::stod(value);
            }
            else if (line.find("\"longitude\":") != std::string::npos) {
                size_t pos = line.find(":");
                std::string value = line.substr(pos + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t,") + 1);
                currentData.longitude = std::stod(value);
            }
            else if (line.find("\"altitude\":") != std::string::npos) {
                size_t pos = line.find(":");
                std::string value = line.substr(pos + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t,") + 1);
                currentData.altitude = std::stod(value);
            }
            else if (line.find("\"speed\":") != std::string::npos) {
                size_t pos = line.find(":");
                std::string value = line.substr(pos + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t,") + 1);
                currentData.speed = std::stod(value);
            }
            else if (line.find("\"course\":") != std::string::npos) {
                size_t pos = line.find(":");
                std::string value = line.substr(pos + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t,") + 1);
                currentData.course = std::stod(value);
            }
        }
    }
    
    file.close();
    std::cout << "Loaded " << gpsData.size() << " GPS records from: " << filename << std::endl;
    return gpsData;
}

// Calculate cumulative distance along a path (filtering GPS noise)
double calculatePathDistance(const std::vector<GPSData>& path, size_t endIdx) {
    double totalDistance = 0.0;
    const double GPS_NOISE_THRESHOLD = 2.0; // Ignore movements smaller than 2 meters (GPS noise)
    
    for (size_t i = 1; i <= endIdx && i < path.size(); ++i) {
        double segmentDistance = calculateDistance(path[i-1].latitude, path[i-1].longitude,
                                                 path[i].latitude, path[i].longitude);
        
        // Only count significant movements (filter out GPS noise)
        if (segmentDistance > GPS_NOISE_THRESHOLD) {
            totalDistance += segmentDistance;
        }
    }
    return totalDistance;
}

// Find closest point on reference path (spatial, not temporal)
GPSComparisonResult compareGPSDataSpatial(const GPSData& current, const std::vector<GPSData>& reference) {
    GPSComparisonResult result;
    result.positionError_m = std::numeric_limits<double>::max();
    result.altitudeError_m = std::numeric_limits<double>::max();
    result.speedError_ms = std::numeric_limits<double>::max();
    result.courseError_deg = std::numeric_limits<double>::max();
    result.timestamp_diff_us = 0; // Not used in spatial comparison
    result.closestReferenceIdx = 0;
    result.distanceAlongPath_m = 0.0;
    result.isOnPath = false;
    
    // Find closest spatial match
    size_t closestIdx = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        double distance = calculateDistance(current.latitude, current.longitude, 
                                          reference[i].latitude, reference[i].longitude);
        
        if (distance < result.positionError_m) {
            result.positionError_m = distance;
            closestIdx = i;
        }
    }
    
    if (closestIdx < reference.size()) {
        const GPSData& ref = reference[closestIdx];
        result.closestReferenceIdx = closestIdx;
        result.distanceAlongPath_m = calculatePathDistance(reference, closestIdx);
        
        result.altitudeError_m = std::abs(current.altitude - ref.altitude);
        result.speedError_ms = std::abs(current.speed - ref.speed);
        
        double courseDiff = std::abs(current.course - ref.course);
        if (courseDiff > 180.0) {
            courseDiff = 360.0 - courseDiff;
        }
        result.courseError_deg = courseDiff;
        
        // Consider "on path" if within 10 meters
        result.isOnPath = (result.positionError_m < 10.0);
    }
    
    return result;
}

// Analyze overall path following performance
PathAnalysis analyzePathFollowing(const std::vector<GPSData>& currentPath, const std::vector<GPSData>& referencePath) {
    PathAnalysis analysis;
    analysis.totalDistance_m = 0.0;
    analysis.averagePositionError_m = 0.0;
    analysis.maxPositionError_m = 0.0;
    analysis.pathFollowingAccuracy_percent = 0.0;
    analysis.totalPoints = currentPath.size();
    analysis.pointsOnPath = 0;
    analysis.isStationary = true;
    analysis.pathProgress_percent = 0.0;
    analysis.referencePathDistance_m = 0.0;
    
    if (currentPath.empty() || referencePath.empty()) {
        return analysis;
    }
    
    double totalError = 0.0;
    
    for (size_t i = 0; i < currentPath.size(); ++i) {
        GPSComparisonResult result = compareGPSDataSpatial(currentPath[i], referencePath);
        
        totalError += result.positionError_m;
        if (result.positionError_m > analysis.maxPositionError_m) {
            analysis.maxPositionError_m = result.positionError_m;
        }
        
        if (result.isOnPath) {
            analysis.pointsOnPath++;
        }
    }
    
    // Calculate distances with GPS noise filtering
    analysis.totalDistance_m = calculatePathDistance(currentPath, currentPath.size() - 1);
    analysis.referencePathDistance_m = calculatePathDistance(referencePath, referencePath.size() - 1);
    
    // Detect if vehicle is stationary (very small movement)
    const double STATIONARY_THRESHOLD = 10.0; // Less than 10 meters total movement
    analysis.isStationary = (analysis.totalDistance_m < STATIONARY_THRESHOLD);
    
    // Calculate path progress
    if (analysis.referencePathDistance_m > 0) {
        analysis.pathProgress_percent = (analysis.totalDistance_m / analysis.referencePathDistance_m) * 100.0;
    }
    
    analysis.averagePositionError_m = totalError / currentPath.size();
    analysis.pathFollowingAccuracy_percent = (double(analysis.pointsOnPath) / currentPath.size()) * 100.0;
    
    return analysis;
}


//------------------------------------------------------------------------------
int main(int argc, const char** argv)
{
    struct sigaction action = {};
    action.sa_handler       = sig_int_handler;

    sigaction(SIGHUP, &action, NULL);  // controlling terminal closed, Ctrl-D
    sigaction(SIGINT, &action, NULL);  // Ctrl-C
    sigaction(SIGQUIT, &action, NULL); // Ctrl-\, clean quit with core dump
    sigaction(SIGABRT, &action, NULL); // abort() called.
    sigaction(SIGTERM, &action, NULL); // kill command

    gRun = true;

    ProgramArguments arguments(
        {ProgramArguments::Option_t("rig", "", "Path to rig configuration file"),
         ProgramArguments::Option_t("input-file", "", "Optional input file with reference GPS data for comparison"),
         ProgramArguments::Option_t("output-dir", "./", "Output directory for GPS data files (default: current directory)")});

    if (!arguments.parse(argc, argv) || !arguments.has("rig"))
    {
        std::cout << "Usage: " << argv[0] << std::endl;
        std::cout << "\t--rig=rig_file.json \t\t\t: Path to rig configuration file containing GPS sensor setup\n";
        std::cout << "\t--input-file=reference.json \t\t: Optional input file with reference GPS data for comparison\n";
        std::cout << "\t--output-dir=/path/to/output \t\t: Output directory for GPS data files (default: current directory)\n";
        std::cout << "\nExample:\n";
        std::cout << "\t" << argv[0] << " --rig=my_rig.json\n";
        std::cout << "\t" << argv[0] << " --rig=my_rig.json --input-file=reference_gps.json\n";
        std::cout << "\t" << argv[0] << " --rig=my_rig.json --output-dir=/home/user/gps_logs\n";

        return -1;
    }

    // Validate rig file exists
    std::string rigFile = arguments.get("rig");
    std::ifstream rigFileCheck(rigFile);
    if (!rigFileCheck.good()) {
        std::cerr << "Error: Cannot open rig file: " << rigFile << std::endl;
        return -1;
    }
    rigFileCheck.close();

    // Check if input file is provided for comparison
    std::string inputFile = arguments.get("input-file");
    std::vector<GPSData> referenceData;
    bool comparisonMode = !inputFile.empty();
    
    if (comparisonMode) {
        std::ifstream inputFileCheck(inputFile);
        if (!inputFileCheck.good()) {
            std::cerr << "Error: Cannot open input file: " << inputFile << std::endl;
            return -1;
        }
        inputFileCheck.close();
        
        referenceData = loadGPSFromJSON(inputFile);
        if (referenceData.empty()) {
            std::cerr << "Error: No reference data loaded from: " << inputFile << std::endl;
            return -1;
        }
        std::cout << "Comparison mode enabled with " << referenceData.size() << " reference records" << std::endl;
    }

    dwContextHandle_t sdk = DW_NULL_HANDLE;
    dwSALHandle_t hal     = DW_NULL_HANDLE;
    dwRigHandle_t rigConfig = DW_NULL_HANDLE;
    dwSensorManagerHandle_t sensorManager = DW_NULL_HANDLE;

    // create a Logger to log to console
    // we keep the ownership of the logger at the application level
    dwLogger_initialize(getConsoleLoggerCallback(true));
    dwLogger_setLogLevel(DW_LOG_VERBOSE);

    // instantiate Driveworks SDK context
    dwContextParameters sdkParams = {};

    CHECK_DW_ERROR(dwInitialize(&sdk, DW_VERSION, &sdkParams));

    // create HAL module of the SDK
    dwSAL_initialize(&hal, sdk);

    // Initialize rig configuration from file
    CHECK_DW_ERROR_MSG(dwRig_initializeFromFile(&rigConfig, sdk, rigFile.c_str()),
                       "Could not initialize Rig from file");
    std::cout << "Rig file loaded successfully: " << rigFile << std::endl;

    // Initialize sensor manager from rig
    dwSensorManagerParams smParams{};
    CHECK_DW_ERROR_MSG(dwSensorManager_initializeFromRigWithParams(&sensorManager, rigConfig, &smParams, 
                                                                   16, hal),
                       "Could not initialize sensor manager from rig");

    // Find GPS sensor in rig
    uint32_t sensorCount = 0;
    CHECK_DW_ERROR(dwRig_getSensorCount(&sensorCount, rigConfig));
    
    uint32_t gpsSensorIdx = 0;
    bool gpsFound = false;
    
    for (uint32_t i = 0; i < sensorCount; i++) {
        dwSensorType type;
        const char* name;
        dwRig_getSensorType(&type, i, rigConfig);
        dwRig_getSensorName(&name, i, rigConfig);
        
        if (type == DW_SENSOR_GPS) {
            gpsSensorIdx = i;
            gpsFound = true;
            std::cout << "Found GPS sensor: " << name << " at index " << i << std::endl;
            break;
        }
    }
    
    if (!gpsFound) {
        std::cerr << "Error: No GPS sensor found in rig file" << std::endl;
        dwSensorManager_release(sensorManager);
        dwRig_release(rigConfig);
        dwSAL_release(hal);
        dwRelease(sdk);
        dwLogger_release();
        return -1;
    }

    // Start sensor manager
    CHECK_DW_ERROR_MSG(dwSensorManager_start(sensorManager), "Could not start sensor manager");

    std::cout << "GPS sensor initialized successfully from rig file" << std::endl;

    // Data collection variables
    std::vector<GPSData> collectedGPSData;
    std::vector<GPSComparisonData> comparisonData;
    std::string outputDir = arguments.get("output-dir");
    std::string timestamp = std::to_string(std::time(nullptr));
    std::string outputFile, comparisonFile, analysisFile;
    
    if (comparisonMode) {
        outputFile = outputDir + "/gps_log_compare_" + timestamp + ".json";
        comparisonFile = outputDir + "/gps_comparison_" + timestamp + ".json";
        analysisFile = outputDir + "/path_analysis_" + timestamp + ".txt";
    } else {
        outputFile = outputDir + "/gps_log_main_" + timestamp + ".json";
    }

    // Main data collection loop
    while (gRun)
    {
        const dwSensorEvent* acquiredEvent = nullptr;
        dwStatus status = dwSensorManager_acquireNextEvent(&acquiredEvent, 50000, sensorManager);

            if (status == DW_END_OF_STREAM)
            {
            std::cout << "End of stream reached" << std::endl;
                break;
            }
            else if (status == DW_TIME_OUT)
        {
                continue;
        }
        else if (status != DW_SUCCESS)
        {
            std::cout << "Sensor manager error: " << dwGetStatusName(status) << std::endl;
            continue;
        }

        // Process GPS events
        if (acquiredEvent->type == DW_SENSOR_GPS && acquiredEvent->sensorIndex == gpsSensorIdx)
        {
            const dwGPSFrame& frame = acquiredEvent->gpsFrame;
            
            // Convert and store GPS data
            GPSData gpsData = convertGPSFrame(frame);
            collectedGPSData.push_back(gpsData);

            // Display GPS information
            std::cout << "GPS - " << frame.timestamp_us;
                std::cout << std::setprecision(10);

                if (dwSignal_checkSignalValidity(frame.validityInfo.latitude) == DW_SUCCESS)
                    std::cout << " lat: " << frame.latitude;

                if (dwSignal_checkSignalValidity(frame.validityInfo.longitude) == DW_SUCCESS)
                    std::cout << " lon: " << frame.longitude;

                if (dwSignal_checkSignalValidity(frame.validityInfo.altitude) == DW_SUCCESS)
                    std::cout << " alt: " << frame.altitude;

                if (dwSignal_checkSignalValidity(frame.validityInfo.course) == DW_SUCCESS)
                    std::cout << " course: " << frame.course;

                if (dwSignal_checkSignalValidity(frame.validityInfo.speed) == DW_SUCCESS)
                    std::cout << " speed: " << frame.speed;

                if (dwSignal_checkSignalValidity(frame.validityInfo.hacc) == DW_SUCCESS)
                    std::cout << " hacc: " << frame.hacc;

            if (frame.errors)
            {
                std::cout << " errorID(" << frame.errors << ")";
                    std::cout << " error messages: [";
                    for (auto& error : gErrorStrings)
                    {
                        if (error.first & frame.errors)
                        {
                            std::cout << error.second << ",";
                        }
                    }
                    std::cout << "]";
            }

            // Perform comparison if in comparison mode
            if (comparisonMode) {
                GPSComparisonResult result = compareGPSDataSpatial(gpsData, referenceData);
                std::cout << " | Path Analysis: pos_err=" << std::fixed << std::setprecision(2) 
                          << result.positionError_m << "m, closest_ref_idx=" << result.closestReferenceIdx
                          << ", dist_along_path=" << result.distanceAlongPath_m << "m"
                          << (result.isOnPath ? " [ON PATH]" : " [OFF PATH]");
                
                // Store comparison data for saving
                if (result.closestReferenceIdx < referenceData.size()) {
                    GPSComparisonData compData;
                    compData.timestamp_us = gpsData.timestamp_us;
                    compData.current_latitude = gpsData.latitude;
                    compData.current_longitude = gpsData.longitude;
                    compData.reference_latitude = referenceData[result.closestReferenceIdx].latitude;
                    compData.reference_longitude = referenceData[result.closestReferenceIdx].longitude;
                    compData.positionError_m = result.positionError_m;
                    compData.altitudeError_m = result.altitudeError_m;
                    compData.speedError_ms = result.speedError_ms;
                    compData.courseError_deg = result.courseError_deg;
                    compData.closestReferenceIdx = result.closestReferenceIdx;
                    compData.distanceAlongPath_m = result.distanceAlongPath_m;
                    compData.isOnPath = result.isOnPath;
                    
                    comparisonData.push_back(compData);
                }
            }

            std::cout << std::endl;
        }

        // Release the event
        dwSensorManager_releaseAcquiredEvent(acquiredEvent, sensorManager);
    }

    // Save collected GPS data
    if (!collectedGPSData.empty()) {
        saveGPSToJSON(collectedGPSData, outputFile);
        
        // Print summary statistics
        std::cout << "\n=== GPS Data Collection Summary ===" << std::endl;
        std::cout << "Total GPS records collected: " << collectedGPSData.size() << std::endl;
        std::cout << "GPS data saved to: " << outputFile << std::endl;
        
        // Save comparison data if in comparison mode
        if (comparisonMode && !comparisonData.empty()) {
            saveComparisonToJSON(comparisonData, comparisonFile, inputFile);
        }
        
        if (comparisonMode) {
            std::cout << "\n=== Path Following Analysis ===" << std::endl;
            PathAnalysis analysis = analyzePathFollowing(collectedGPSData, referenceData);
            
            std::cout << "Reference path points: " << referenceData.size() << std::endl;
            std::cout << "Current path points: " << analysis.totalPoints << std::endl;
            std::cout << "Reference path distance: " << std::fixed << std::setprecision(2) 
                      << analysis.referencePathDistance_m << " meters" << std::endl;
            std::cout << "Total distance traveled: " << std::fixed << std::setprecision(2) 
                      << analysis.totalDistance_m << " meters" << std::endl;
            std::cout << "Path progress: " << std::fixed << std::setprecision(1) 
                      << analysis.pathProgress_percent << "%" << std::endl;
            std::cout << "Vehicle status: " << (analysis.isStationary ? "STATIONARY" : "MOVING") << std::endl;
            std::cout << "Average position error: " << analysis.averagePositionError_m 
                      << " meters" << std::endl;
            std::cout << "Maximum position error: " << analysis.maxPositionError_m 
                      << " meters" << std::endl;
            std::cout << "Path following accuracy: " << std::fixed << std::setprecision(1) 
                      << analysis.pathFollowingAccuracy_percent << "%" << std::endl;
            std::cout << "Points on path (within 10m): " << analysis.pointsOnPath << "/" 
                      << analysis.totalPoints << std::endl;
            
            // Save analysis to TXT file
            savePathAnalysisToTXT(analysis, analysisFile, inputFile, outputFile);
            
            // Determine if following the same path
            if (analysis.isStationary) {
                std::cout << "RESULT: Vehicle is STATIONARY - Path following analysis not applicable" << std::endl;
                std::cout << "NOTE: To test path following, record a path while moving, then follow it" << std::endl;
            } else if (analysis.pathFollowingAccuracy_percent >= 80.0) {
                std::cout << "RESULT: Successfully following the same path!" << std::endl;
            } else if (analysis.pathFollowingAccuracy_percent >= 60.0) {
                std::cout << "RESULT: Partially following the path (some deviation)" << std::endl;
            } else {
                std::cout << "RESULT: Not following the same path (significant deviation)" << std::endl;
            }
        }
    } else {
        std::cout << "No GPS data collected" << std::endl;
    }

    // Stop and release sensor manager
    dwSensorManager_stop(sensorManager);
    dwSensorManager_release(sensorManager);

    // release used objects in correct order
    dwRig_release(rigConfig);
    dwSAL_release(hal);
    dwRelease(sdk);
    dwLogger_release();

    return 0;
}
