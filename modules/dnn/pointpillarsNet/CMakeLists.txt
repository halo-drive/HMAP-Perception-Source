################################################################################

project(pointpillarsNet)

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(SOURCES 
    PointPillarDetector.cpp
    main.cpp)

set(LIBRARIES
    samples_framework
    samples_framework_program_arguments
    samples_framework_simple_camera
    samples_framework_simple_streamer
    cuDNN::cuDNN
    TensorRT::TensorRT
    /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8
    ${Driveworks_LIBRARIES}
)

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})

sdk_add_sample(${PROJECT_NAME})
