project(driveseg_object)

set(SOURCES
    drivesegserver.cu
)

set(LIBRARIES
    samples_framework
    samples_framework_checks
    samples_framework_program_arguments
    samples_framework_simple_camera
    samples_framework_simple_streamer
    ${Driveworks_LIBRARIES}
    /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so
)

include_directories(/usr/include/aarch64-linux-gnu)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})
sdk_add_sample(${PROJECT_NAME})
