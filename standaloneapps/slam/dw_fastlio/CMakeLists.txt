#########################################################################################
# DriveWorks Fast-LIO SLAM Sample
#########################################################################################

project(sample_dw_fastlio_slam C CXX)

#----------------------------------------------------------------------------------------
# Dependencies
#----------------------------------------------------------------------------------------

# Explicitly include PCL directories
include_directories(SYSTEM /usr/include/pcl-1.10)
include_directories(SYSTEM /usr/include/eigen3)
include_directories(SYSTEM /usr/include/vtk-7.1)
link_directories(/usr/lib/x86_64-linux-gnu)

# G2O
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/lidar-slam-detection/slam/backend/hdl_graph_slam/cmake)
find_package(G2O REQUIRED)
include_directories(SYSTEM ${G2O_INCLUDE_DIR})
link_directories(${G2O_LIBRARY_DIRS})

#----------------------------------------------------------------------------------------
# Source files
#----------------------------------------------------------------------------------------

set(SOURCES
    src/main.cpp
    src/DWFastLIO.cpp
)

set(HEADERS
    include/DWFastLIO.hpp
    include/TypeConverters.hpp
)

#----------------------------------------------------------------------------------------
# Executable
#----------------------------------------------------------------------------------------

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

#----------------------------------------------------------------------------------------
# Include directories
#----------------------------------------------------------------------------------------

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        /usr/local/lidar-slam-detection/slam/mapping/fastlio/include
        /usr/local/lidar-slam-detection/slam/common
        /usr/local/lidar-slam-detection/slam/thirdparty
)

#----------------------------------------------------------------------------------------
# Link libraries
#----------------------------------------------------------------------------------------

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        # DriveWorks
        samples_framework
        ${Driveworks_LIBRARIES}
        
        # PCL
        pcl_common
        pcl_io
        pcl_filters
        
        # G2O
        ${G2O_CORE_LIBRARY}
        ${G2O_STUFF_LIBRARY}
        ${G2O_SOLVER_CHOLMOD}
        ${G2O_SOLVER_CSPARSE}
        ${G2O_TYPES_SLAM3D}
        ${G2O_TYPES_SLAM3D_ADDONS}
        
        # Fast-LIO backend from LSD (include all dependent libraries)
        /usr/local/lib/python3.8/dist-packages/perception-1.0.0-py3.8-linux-x86_64.egg/libfast_lio.so
        /usr/local/lib/python3.8/dist-packages/perception-1.0.0-py3.8-linux-x86_64.egg/libg2o_backend.so
        /usr/local/lib/python3.8/dist-packages/perception-1.0.0-py3.8-linux-x86_64.egg/libmap_common.so
        /usr/local/lib/python3.8/dist-packages/perception-1.0.0-py3.8-linux-x86_64.egg/libfast_gicp.so
        
        # System
        pthread
)

#----------------------------------------------------------------------------------------
# Compiler settings
#----------------------------------------------------------------------------------------

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -Wno-deprecated-declarations
)

#----------------------------------------------------------------------------------------
# Install (using DriveWorks sample macro)
#----------------------------------------------------------------------------------------

sdk_add_sample(${PROJECT_NAME})

