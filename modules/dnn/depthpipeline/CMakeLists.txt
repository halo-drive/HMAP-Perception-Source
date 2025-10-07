################################################################################
project(depthpipeline C CXX)

#-------------------------------------------------------------------------------
# CUDA Cross-Compilation Configuration for Drive SDK
#-------------------------------------------------------------------------------

# Set CUDA compiler (needed for cross-compilation)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# Target GPU architecture for AGX Orin (SM_87)
set(CMAKE_CUDA_ARCHITECTURES 87)

# CUDA compilation flags for cross-compilation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_87,code=sm_87")

# CUDA include and link directories for aarch64
include_directories(/usr/local/cuda/targets/aarch64-linux/include)
link_directories(/usr/local/cuda/targets/aarch64-linux/lib)

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------

set(SOURCES
    main.cpp
    CameraCapture.cpp
    DepthInferenceEngine.cpp
    DepthPipeline.cpp
    DepthRenderer.cpp
    MemoryManager.cpp
)

set(CUDA_SOURCES
    depth_kernels.cu
    depth_visualization_kernels.cu
)

set(LIBRARIES
    samples_framework
    samples_framework_checks
    samples_framework_program_arguments
    samples_framework_simple_camera
    samples_framework_simple_streamer
    cuDNN::cuDNN
    TensorRT::TensorRT
    ${Driveworks_LIBRARIES}
    cudart
)

#-------------------------------------------------------------------------------
# CUDA source properties (disable separable compilation)
#-------------------------------------------------------------------------------

set_source_files_properties(
    ${CUDA_SOURCES}
    PROPERTIES 
    LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION OFF
)

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------

add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
)

target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})

sdk_add_sample(${PROJECT_NAME})
