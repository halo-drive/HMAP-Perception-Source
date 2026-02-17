#########################################################################################
# DriveWorks Fast-LIO SLAM Sample
#########################################################################################

project(sample_dw_fastlio_slam C CXX)

#----------------------------------------------------------------------------------------
# Architecture Detection
#----------------------------------------------------------------------------------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(ARCH "aarch64")
    set(ARCH_SUFFIX "aarch64-linux-gnu")
else()
    set(ARCH "x86_64")
    set(ARCH_SUFFIX "x86_64-linux-gnu")
endif()

message(STATUS "Detected architecture: ${ARCH}")
message(STATUS "Architecture suffix: ${ARCH_SUFFIX}")

#----------------------------------------------------------------------------------------
# Dependencies
#----------------------------------------------------------------------------------------

# Fix Eigen/X11 conflict - undefine Success symbol that X11 defines
# This will be applied via compiler flags to ensure it's defined before any includes

# Explicitly include PCL and system directories
include_directories(SYSTEM /usr/include/pcl-1.10)
include_directories(SYSTEM /usr/include/eigen3)
include_directories(SYSTEM /usr/include/vtk-7.1)
# Needed so PCL can see Boost headers like boost/shared_ptr.hpp
include_directories(SYSTEM /usr/include)
include_directories(SYSTEM /usr/include)

# Architecture-aware library directories
if(ARCH STREQUAL "aarch64")
    # Boost for aarch64 - you'll need to install and copy from target device
    # Typical locations: /usr/lib/aarch64-linux-gnu or sysroot
    if(EXISTS /usr/lib/${ARCH_SUFFIX})
        link_directories(/usr/lib/${ARCH_SUFFIX})
    endif()
    if(EXISTS /usr/include/${ARCH_SUFFIX})
        include_directories(SYSTEM /usr/include/${ARCH_SUFFIX})
    endif()
else()
    link_directories(/usr/lib/${ARCH_SUFFIX})
endif()

# Boost (required by PCL)
if(ARCH STREQUAL "aarch64")
    # For aarch64, try to find Boost in common locations
    # You may need to set BOOST_ROOT if Boost is installed in a custom location
    set(BOOST_ROOT ${BOOST_ROOT} CACHE PATH "Boost root directory for aarch64")
    if(EXISTS /usr/lib/${ARCH_SUFFIX})
        set(BOOST_LIBRARYDIR /usr/lib/${ARCH_SUFFIX})
    endif()
    if(EXISTS /usr/include/${ARCH_SUFFIX}/boost)
        set(BOOST_INCLUDEDIR /usr/include/${ARCH_SUFFIX})
    endif()
endif()
find_package(Boost REQUIRED COMPONENTS system filesystem)
if(Boost_FOUND)
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    message(STATUS "Found Boost: ${Boost_INCLUDE_DIRS}")
else()
    message(WARNING "Boost not found. PCL may fail to compile. For aarch64, install Boost on target device and copy headers/libs back.")
endif()

# G2O
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/lidar-slam-detection/slam/backend/hdl_graph_slam/cmake)
find_package(G2O REQUIRED)
include_directories(SYSTEM ${G2O_INCLUDE_DIR})
link_directories(${G2O_LIBRARY_DIRS})

# Graph backend = optimization + loop closure (g2o). OpenCV is required only so LSD's
# mapping_types.h compiles (it has cv::Mat in the frame type); we don't use OpenCV algorithms.
find_package(OpenCV REQUIRED)
message(STATUS "Graph backend (optimization + loop closure) enabled")

#----------------------------------------------------------------------------------------
# Source files
#----------------------------------------------------------------------------------------

set(SCANCONTEXT_DIR /usr/local/lidar-slam-detection/slam/common/Scancontext)
set(SOURCES
    src/main.cpp
    src/DWFastLIO.cpp
    src/GlobalReloc.cpp
    ${SCANCONTEXT_DIR}/Scancontext.cpp
)

set(HEADERS
    include/DWFastLIO.hpp
    include/TypeConverters.hpp
    include/GlobalReloc.hpp
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
        /usr/local/lidar-slam-detection/slam/backend
        /usr/local/lidar-slam-detection/slam/thirdparty
        /usr/local/lidar-slam-detection/third_party/readerwriterqueue
        /usr/local/lidar-slam-detection/sensor_driver/common_lib/cpp_utils
        /usr/local/lidar-slam-detection/sensor_driver/common_lib/logging
        ${SCANCONTEXT_DIR}
        ${OpenCV_INCLUDE_DIRS}
)
# Graph backend = optimization + loop closure (always on)
target_compile_definitions(${PROJECT_NAME} PRIVATE DW_FASTLIO_USE_GRAPH_BACKEND=1)

#----------------------------------------------------------------------------------------
# Link libraries
#----------------------------------------------------------------------------------------

# PCL: use absolute libs for aarch64 so cross-linker finds them
if(ARCH STREQUAL "aarch64")
    set(PCL_LIBS
        /usr/lib/aarch64-linux-gnu/libpcl_common.so
        /usr/lib/aarch64-linux-gnu/libpcl_io.so
        /usr/lib/aarch64-linux-gnu/libpcl_filters.so
        /usr/lib/aarch64-linux-gnu/libpcl_kdtree.so
        /usr/lib/aarch64-linux-gnu/libpcl_sample_consensus.so
        /usr/lib/aarch64-linux-gnu/libpcl_search.so
        /usr/lib/aarch64-linux-gnu/libpcl_registration.so
    )
else()
    set(PCL_LIBS
        pcl_common
        pcl_io
        pcl_filters
        pcl_kdtree
        pcl_sample_consensus
        pcl_search
        pcl_registration
    )
endif()

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        # DriveWorks
        samples_framework
        ${Driveworks_LIBRARIES}
        
        # PCL
        ${PCL_LIBS}
        
        # G2O
        ${G2O_CORE_LIBRARY}
        ${G2O_STUFF_LIBRARY}
        ${G2O_SOLVER_CHOLMOD}
        ${G2O_SOLVER_CSPARSE}
        ${G2O_TYPES_SLAM3D}
        ${G2O_TYPES_SLAM3D_ADDONS}
        
        # Fast-LIO backend from LSD (architecture-specific libraries)
        # NOTE: These libraries must be compiled for the target architecture
        # For aarch64, compile on target device and copy back, or use sysroot
        
)

# Add architecture-specific Fast-LIO libraries
if(ARCH STREQUAL "aarch64")
    target_link_libraries(${PROJECT_NAME} PRIVATE
        /usr/local/lib/python3.8/dist-packages/perception-1.0.0-py3.8-linux-aarch64.egg/libfast_lio.so
        /usr/local/lib/python3.8/dist-packages/perception-1.0.0-py3.8-linux-aarch64.egg/libg2o_backend.so
        /usr/local/lib/python3.8/dist-packages/perception-1.0.0-py3.8-linux-aarch64.egg/libmap_common.so
        /usr/local/lib/python3.8/dist-packages/perception-1.0.0-py3.8-linux-aarch64.egg/libfast_gicp.so
    )
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE
        /usr/local/lib/python3.8/dist-packages/perception-1.0.0-py3.8-linux-x86_64.egg/libfast_lio.so
        /usr/local/lib/python3.8/dist-packages/perception-1.0.0-py3.8-linux-x86_64.egg/libg2o_backend.so
        /usr/local/lib/python3.8/dist-packages/perception-1.0.0-py3.8-linux-x86_64.egg/libmap_common.so
        /usr/local/lib/python3.8/dist-packages/perception-1.0.0-py3.8-linux-x86_64.egg/libfast_gicp.so
    )
endif()

# Add Boost, OpenCV, and system libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    # Boost (required, so always link)
    Boost::system
    Boost::filesystem

    # OpenCV (required for LSD mapping_types.h cv::Mat; provides cv::fastFree etc. at link time)
    ${OpenCV_LIBS}

    # System
    pthread
)

#----------------------------------------------------------------------------------------
# Linker options (relax shared library undefineds on aarch64)
#----------------------------------------------------------------------------------------

if(ARCH STREQUAL "aarch64")
    # Allow undefined symbols in shared libraries at link time; they will be
    # resolved on the target (Orin) where full dependency set is installed.
    target_link_options(${PROJECT_NAME} PRIVATE
        -Wl,--allow-shlib-undefined
    )
endif()

#----------------------------------------------------------------------------------------
# Compiler settings
#----------------------------------------------------------------------------------------

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -Wno-deprecated-declarations
    # Force include fix_eigen_x11.h before any other headers to fix Eigen/X11 conflict
    -include "${CMAKE_CURRENT_SOURCE_DIR}/include/fix_eigen_x11.h"
)

#----------------------------------------------------------------------------------------
# Install (using DriveWorks sample macro)
#----------------------------------------------------------------------------------------

sdk_add_sample(${PROJECT_NAME})

