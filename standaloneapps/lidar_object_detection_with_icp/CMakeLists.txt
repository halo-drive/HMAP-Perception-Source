project(lidar_object_detection_with_icp)

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(SOURCES 
main.cpp
InterLidarICP.cpp
InterLidarICP.hpp
)

find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)

set(LIBRARIES
    samples_framework
    samples_framework_program_arguments
    ${Driveworks_LIBRARIES}
    nvinfer
    nvinfer_plugin
    ${CUDA_LIBRARIES}
    ${TensorRT_LIBRARIES}
)

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})

# ------------------------------------------------------------------------------
# Install target
# ------------------------------------------------------------------------------
sdk_add_sample(${PROJECT_NAME})


# ./trtexec --onnx=/usr/local/driveworks/samples/src/sensors/lidar/lidar_engine_object_detection/pointpillars_deployable.fixed.onnx \
#         --maxShapes=points:1x204800x4,num_points:1 \
#         --minShapes=points:1x204800x4,num_points:1 \
#         --optShapes=points:1x204800x4,num_points:1 \
#         --fp16 \
#         --saveEngine=/usr/local/driveworks/samples/src/sensors/lidar/lidar_engine_object_detection/model.engine