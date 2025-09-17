project(sample_bevfusion_driveworks)

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(SOURCES main.cpp)

#-------------------------------------------------------------------------------
# Find BEVFusion dependencies
#-------------------------------------------------------------------------------
# Add BEVFusion include directories - works for both host and cross-compilation
include_directories(/usr/local/Lidar_AI_Solution/CUDA-BEVFusion/src)
include_directories(/usr/local/Lidar_AI_Solution/CUDA-BEVFusion/src/common)
include_directories(/usr/local/Lidar_AI_Solution/CUDA-BEVFusion/src/bevfusion)

# Add BEVFusion library path
if(CMAKE_CROSSCOMPILING)
    # For cross-compilation, use aarch64 libraries
    link_directories(/usr/local/Lidar_AI_Solution/CUDA-BEVFusion/build-aarch64)
    # Add spconv library path for aarch64
    link_directories(/usr/local/Lidar_AI_Solution/libraries/3DSparseConvolution/libspconv/lib/aarch64_cuda11.4)
else()
    # For host compilation, use x86_64 libraries
    link_directories(/usr/local/Lidar_AI_Solution/CUDA-BEVFusion/build)
    # Add spconv library path for x86_64
    link_directories(/usr/local/Lidar_AI_Solution/libraries/3DSparseConvolution/libspconv/lib/x86_64_cuda11.4)
endif()

#-------------------------------------------------------------------------------
# Libraries
#-------------------------------------------------------------------------------
set(LIBRARIES
    samples_framework
    samples_framework_checks
    samples_framework_log
    samples_framework_program_arguments
    ${Driveworks_LIBRARIES}
    # BEVFusion libraries
    bevfusion_core
    spconv
    # CUDA libraries
    cuda
    # Additional dependencies
    dl
)

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})

# Set RPATH for runtime library search on target system
if(CMAKE_CROSSCOMPILING)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        # INSTALL_RPATH "/home/benchdev2/Lidar_AI_Solution/CUDA-BEVFusion/build:/home/benchdev2/Lidar_AI_Solution/libraries/3DSparseConvolution/libspconv/lib/aarch64_cuda11.4"
        INSTALL_RPATH "$ORIGIN/bev-fusion-lib/lib"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

#-------------------------------------------------------------------------------
# Compiler flags
#-------------------------------------------------------------------------------
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Add CUDA flags
target_compile_options(${PROJECT_NAME} PRIVATE 
    -DCUDA_BEVFUSION_PATH="/usr/local/Lidar_AI_Solution/CUDA-BEVFusion"
)

#-------------------------------------------------------------------------------
# Install target
#-------------------------------------------------------------------------------
sdk_add_sample(${PROJECT_NAME})

