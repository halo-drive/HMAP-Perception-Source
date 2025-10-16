# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

############################################################################
# Project configuration for SygnalPomo CAN Controller
############################################################################

project(sample_sygnalpomo_controller C CXX)

#------------------------------------------------------------------
# Project files
#------------------------------------------------------------------
set(SOURCES
    main.cpp
    SygnalPomoController.cpp
    SygnalPomoController.hpp
)

#------------------------------------------------------------------
# Dependencies and Libraries
#------------------------------------------------------------------
set(LIBRARIES
    samples_framework
    samples_framework_checks
    samples_framework_data_path
    samples_framework_log
    samples_framework_program_arguments
    ${DRIVEWORKS_LIBRARIES}
)

#------------------------------------------------------------------
# Final target configuration
#------------------------------------------------------------------
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${DRIVEWORKS_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..  # Adjust to match samples directory structure
)

target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})

# Install DBC files alongside the executable
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${SDK_SAMPLE_DESTINATION}
)