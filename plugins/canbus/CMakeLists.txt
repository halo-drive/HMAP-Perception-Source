################################################################################
#
# Notice
# This CAN plugin implementation is a joint development between Sygnal and
# POMO Robotics, building upon NVIDIA DriveWorks SDK.
#
# ALL DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS".
# SYGNAL AND POMO ROBOTICS MAKE NO REPRESENTATIONS, WARRANTIES, EXPRESSED,
# IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY
# DISCLAIM ANY IMPLIED WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, OR
# FITNESS FOR A PARTICULAR PURPOSE.
#
# Copyright (c) 2024:
# - SYGNAL
# - POMO ROBOTICS LTD.
#
# This implementation utilizes NVIDIA DriveWorks SDK under proper licensing.
# Portions of this code:
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# This material and its use are subject to the terms of the license agreement.
# Distribution and use in source and binary forms, with or without modification,
# are permitted provided adherence to the licensing terms and conditions.
#
################################################################################
################################################################################
# Project definition and requirements
################################################################################
cmake_minimum_required(VERSION 3.10)
project(sygnalpomo_can_plugin)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
################################################################################
# CUDA Configuration - Ensure CUDA Paths Are Found
################################################################################
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda" CACHE PATH "Path to CUDA Toolkit")
set(CUDA_INCLUDE_DIRS "/usr/local/cuda/targets/aarch64-linux/include" CACHE PATH "CUDA Include Directories")
set(CUDA_CUDART_LIBRARY "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so" CACHE FILEPATH "CUDA CUDART Library")
set(CUDA_NVCC_EXECUTABLE "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA NVCC Executable")

include_directories(${CUDA_INCLUDE_DIRS})

################################################################################
# SDK Discovery - Aligned with Toolchain Strategy
################################################################################
# Primary DriveWorks discovery -
find_package(DriveWorks REQUIRED 
    PATHS /usr/local/driveworks
    NO_DEFAULT_PATH
)

################################################################################
# Compiler Configuration
################################################################################
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -fPIC")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

################################################################################
# Include Configuration - Following Toolchain Order
################################################################################
include_directories(
    ${DriveWorks_INCLUDE_DIRS}  # Primary DriveWorks includes
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/sygnalpomoh
)

################################################################################
# Source Definition
################################################################################
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/PluginImpl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SygnalPomoDriver.cpp
)

################################################################################
# Target Configuration
################################################################################
add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${DriveWorks_LIBRARIES}
    sample_sensors_plugin_common
    dwframework

    -Wl,--enable-new-dtags
    -Wl,-rpath-link,/usr/local/driveworks/targets/aarch64-Linux/lib
)

################################################################################
# Installation Configuration - Preserving DriveWorks Layout
################################################################################
set(SDK_SAMPLE_DESTINATION "bin")
set(SDK_LIBRARY_DESTINATION "lib")

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${SDK_SAMPLE_DESTINATION}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${SDK_LIBRARY_DESTINATION}"
    INSTALL_RPATH "/usr/local/driveworks/targets/aarch64-Linux/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

sdk_install_sample_library(${PROJECT_NAME})

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/dbc/Heartbeat.dbc
    ${CMAKE_CURRENT_SOURCE_DIR}/dbc/Control.dbc
    DESTINATION ${CMAKE_BINARY_DIR}/dbc
)