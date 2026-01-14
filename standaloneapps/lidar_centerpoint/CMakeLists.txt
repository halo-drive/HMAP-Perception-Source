project(lidar_centerpoint)

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(SOURCES 
main.cpp
InterLidarICP.cpp
InterLidarICP.hpp
FreeSpaceDW.cpp
FreeSpaceDW.hpp
CenterPointDW.cpp
CenterPointDW.hpp
centerpoint/preprocess.cpp
centerpoint/preprocess.cu
centerpoint/postprocess.cpp
centerpoint/scatter_cuda.cu
centerpoint/logger.cpp
centerpoint/iou3d_nms_kernel.cu
)

include_directories(
    /usr/include/eigen3
    ${CMAKE_CURRENT_SOURCE_DIR}/centerpoint
    ${CMAKE_CURRENT_SOURCE_DIR}/centerpoint/common
)

set(LIBRARIES
    samples_framework
    samples_framework_program_arguments
    ${Driveworks_LIBRARIES}
    nvinfer
    nvinfer_plugin
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

