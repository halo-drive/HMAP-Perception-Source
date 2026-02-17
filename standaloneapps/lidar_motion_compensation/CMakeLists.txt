project(lidar_motion_compensation)

#-------------------------------------------------------------------------------
# Detect architecture
#-------------------------------------------------------------------------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(ARCH "aarch64")
    set(ARCH_SUFFIX "aarch64-linux-gnu")
    set(CUDA_ROOT "/usr/local/cuda-11.4")
    set(CUDA_TARGET_DIR "${CUDA_ROOT}/targets/aarch64-linux")
else()
    set(ARCH "x86_64")
    set(ARCH_SUFFIX "x86_64-linux-gnu")
    set(CUDA_ROOT "/usr/local/cuda")
    set(CUDA_TARGET_DIR "${CUDA_ROOT}")
endif()

message(STATUS "Detected architecture: ${ARCH}")
message(STATUS "Architecture suffix: ${ARCH_SUFFIX}")

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

#-------------------------------------------------------------------------------
# CUDA Configuration
#-------------------------------------------------------------------------------
if(ARCH STREQUAL "aarch64")
    # Use modern CUDART find_package for aarch64
    # CUDA language is already enabled by parent CMakeLists.txt
    find_package(CUDART MODULE REQUIRED)
    include_directories(${CUDA_TARGET_DIR}/include)
    link_directories(${CUDA_TARGET_DIR}/lib)
else()
    # Standard CUDA for x86_64
    find_package(CUDA REQUIRED)
endif()

#-------------------------------------------------------------------------------
# TensorRT Configuration
#-------------------------------------------------------------------------------
if(ARCH STREQUAL "aarch64")
    # TensorRT paths for aarch64
    set(TensorRT_INCLUDE_DIR "/usr/include/${ARCH_SUFFIX}")
    set(TensorRT_LIBRARY_DIR "/usr/lib/${ARCH_SUFFIX}")
    include_directories(${TensorRT_INCLUDE_DIR})
    link_directories(${TensorRT_LIBRARY_DIR})
    find_package(TensorRT REQUIRED)
    
    # Add link directories explicitly for aarch64
    link_directories(
        ${TensorRT_LIBRARY_DIR}
        ${CUDA_TARGET_DIR}/lib
    )
else()
    # TensorRT paths for x86_64
    find_package(TensorRT REQUIRED)
endif()

set(LIBRARIES
    samples_framework
    samples_framework_program_arguments
    ${Driveworks_LIBRARIES}
)

# TensorRT libraries (architecture-specific)
if(ARCH STREQUAL "aarch64")
    list(APPEND LIBRARIES
        ${TensorRT_LIBRARY_DIR}/libnvinfer.so
        ${TensorRT_LIBRARY_DIR}/libnvinfer_plugin.so
        cuda
        ${CUDA_LIBRARIES}
        ${TensorRT_LIBRARIES}
    )
else()
    list(APPEND LIBRARIES
        nvinfer
        nvinfer_plugin
        ${CUDA_LIBRARIES}
        ${TensorRT_LIBRARIES}
    )
endif()

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