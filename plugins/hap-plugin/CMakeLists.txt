################################################################################
#
# Livox HAP Plugin for NVIDIA DriveWorks
# Based on the NVIDIA sample lidar plugin
#
# Copyright (c) 2025 - All rights reserved.
#
################################################################################

project(livox_hap_plugin)

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(SOURCES
    main.cpp
    LivoxHapPlugin.cpp
    LivoxHapPlugin.hpp
    LivoxHapProperties.hpp
    BufferPool.hpp
)

#-------------------------------------------------------------------------------
# Find dependencies
#-------------------------------------------------------------------------------
# Set Livox SDK location - can be overridden by user
set(LIVOX_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../Livox-SDK2" CACHE PATH "Path to Livox SDK")

include_directories(
    ${LIVOX_SDK_ROOT}/include
    ${LIVOX_SDK_ROOT}/sdk_core/include
    /usr/local/include  # For installed headers
    ${CMAKE_CURRENT_SOURCE_DIR}
)

link_directories(
    ${LIVOX_SDK_ROOT}/build
    /usr/local/lib      # For installed libraries
)

set(LIBRARIES
    ${Driveworks_LIBRARIES}
    livox_lidar_sdk_shared
)

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})

#-------------------------------------------------------------------------------
# Install target
#-------------------------------------------------------------------------------
install(TARGETS ${PROJECT_NAME} 
        LIBRARY DESTINATION lib)

#-------------------------------------------------------------------------------
# Print configuration summary
#-------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "Livox HAP Plugin for NVIDIA DriveWorks Configuration:")
message(STATUS "  Livox SDK location: ${LIVOX_SDK_ROOT}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")