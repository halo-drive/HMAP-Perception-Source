project(lidar_motion_compensation)

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(SOURCES 
main.cpp
InterLidarICP.cpp
InterLidarICP.hpp
FreeSpaceDW.cpp
FreeSpaceDW.hpp
sygnalpomoparser.cpp
sygnalpomoparser.hpp
)

include_directories(
    /usr/include/eigen3
)

# Add TensorRT include directory explicitly
include_directories(/usr/include/aarch64-linux-gnu)

# Use modern CUDART find_package instead of deprecated CUDA
find_package(CUDART MODULE REQUIRED)
find_package(CUDA REQUIRED)

# Add TensorRT paths for aarch64
set(TensorRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu")
set(TensorRT_LIBRARY_DIR "/usr/lib/aarch64-linux-gnu")
find_package(TensorRT REQUIRED)

# Add link directories explicitly
link_directories(
    /usr/lib/aarch64-linux-gnu
    /usr/local/cuda-11.4/targets/aarch64-linux/lib
)

set(LIBRARIES
    samples_framework
    samples_framework_program_arguments
    ${Driveworks_LIBRARIES}
    /usr/lib/aarch64-linux-gnu/libnvinfer.so
    /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so
    cuda
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