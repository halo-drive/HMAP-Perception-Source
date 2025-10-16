#-------------------------------------------------------------------------------
# Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.
#
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Sample - Lidar Replay with PointPillars Detection
#-------------------------------------------------------------------------------
project(sample_lidar_replay_pointpillars)

#-------------------------------------------------------------------------------
# Source files
#-------------------------------------------------------------------------------
set(SOURCES
    main.cpp
)

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
add_executable(${PROJECT_NAME} ${SOURCES})

#-------------------------------------------------------------------------------
# Standard DriveWorks libraries
#-------------------------------------------------------------------------------
set(LIBRARIES
    samples_framework
    samples_framework_math_utils
    samples_framework_program_arguments
    ${Driveworks_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})

#-------------------------------------------------------------------------------
# PointPillars Integration (AFTER target is created)
#-------------------------------------------------------------------------------

# PointPillars headers (in current directory)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# PointPillars library (local .so file)
set(POINTPILLARS_LIB ${CMAKE_CURRENT_SOURCE_DIR}/libpointpillar_core.so)
target_link_libraries(${PROJECT_NAME} PRIVATE ${POINTPILLARS_LIB})

# Find and link TensorRT
find_library(NVINFER_LIB nvinfer 
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu 
    PATH_SUFFIXES lib lib64)

find_library(NVINFER_PLUGIN_LIB nvinfer_plugin
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
    PATH_SUFFIXES lib lib64)

if(NVINFER_LIB AND NVINFER_PLUGIN_LIB)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${NVINFER_LIB}
        ${NVINFER_PLUGIN_LIB}
    )
    message(STATUS "TensorRT found: ${NVINFER_LIB}")
else()
    message(WARNING "TensorRT not found")
endif()

# Link CUDA libraries if available
find_package(CUDA QUIET)
if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
    )
endif()

#-------------------------------------------------------------------------------
# Compiler definitions
#-------------------------------------------------------------------------------
target_compile_definitions(${PROJECT_NAME} PRIVATE
    POINTPILLARS_DEFAULT_MODEL_PATH="/usr/local/driveworks/data/models/pointpillar.plan"
)

#-------------------------------------------------------------------------------
# Print status
#-------------------------------------------------------------------------------
message(STATUS "PointPillars integration configured for ${PROJECT_NAME}")