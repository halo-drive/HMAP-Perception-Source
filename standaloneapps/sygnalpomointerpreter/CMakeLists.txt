################################################################################
# Project configuration for SygnalPomo CAN Interpreter
################################################################################

project(sample_sygnalpomo_interpreter C CXX)

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(SOURCES
    main.cpp
    SygnalPomoMonitor.cpp
    SygnalPomoMonitor.hpp
)

#-------------------------------------------------------------------------------
# Dependencies and Libraries
#-------------------------------------------------------------------------------
set(LIBRARIES
    samples_framework
    samples_framework_checks
    samples_framework_data_path
    samples_framework_log
    samples_framework_program_arguments
    ${Driveworks_LIBRARIES}
)

#-------------------------------------------------------------------------------
# Final target configuration
#-------------------------------------------------------------------------------
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} 
    PRIVATE
    ${Driveworks_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src    # Adjust to match samples directory structure
)

target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})

# Add sample to SDK
sdk_add_sample(${PROJECT_NAME})

#-------------------------------------------------------------------------------
# Installation configuration
#-------------------------------------------------------------------------------
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${SDK_SAMPLE_DESTINATION}
)

# Install DBC files alongside the executable