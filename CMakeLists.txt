set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_minimum_required(VERSION 3.10)

project(Soteria-cpp)
include_directories("include/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(ONNX_RUNTIME_DIR "C:/tools/onnxruntime") # change to your system?
set(ONNX_RUNTIME_INCLUDE_DIR "${ONNX_RUNTIME_DIR}/include")
set(ONNX_RUNTIME_LIB_DIR "${ONNX_RUNTIME_DIR}/lib")
include_directories(${ONNX_RUNTIME_INCLUDE_DIR})

set(SPDLOG_DIR "C:/tools/spdlog")
set(SPDLOG_INCLUDE_DIR "${SPDLOG_DIR}/include")
set(SPDLOG_LIB_DIR "${SPDLOG_DIR}/lib")
include_directories(${SPDLOG_INCLUDE_DIR})
link_directories(${SPDLOG_LIB_DIR})

add_executable(camera
        include/YoloOnnxModel.cpp
        include/YoloOnnxModel.h
)
add_executable(video
        src/video.cpp
        include/YoloOnnxModel.cpp
        include/YoloOnnxModel.h
)
add_executable(rtsp
        src/live.cpp
        include/YoloOnnxModel.cpp
        include/YoloOnnxModel.h
)

target_link_libraries(camera ${OpenCV_LIBS})
target_link_libraries(video ${OpenCV_LIBS})
target_link_libraries(rtsp ${OpenCV_LIBS})

target_link_libraries(camera "${SPDLOG_LIB_DIR}/spdlogd.lib")
target_link_libraries(video "${SPDLOG_LIB_DIR}/spdlogd.lib")
target_link_libraries(rtsp "${SPDLOG_LIB_DIR}/spdlogd.lib")

if(WIN32)
    target_link_libraries(camera "${ONNX_RUNTIME_LIB_DIR}/onnxruntime.lib")
    target_link_libraries(video "${ONNX_RUNTIME_LIB_DIR}/onnxruntime.lib")
    target_link_libraries(rtsp "${ONNX_RUNTIME_LIB_DIR}/onnxruntime.lib")
else()
    target_link_libraries(camera "${ONNX_RUNTIME_LIB_DIR}/libonnxruntime.so")
    target_link_libraries(video "${ONNX_RUNTIME_LIB_DIR}/libonnxruntime.so")
    target_link_libraries(rtsp "${ONNX_RUNTIME_LIB_DIR}/libonnxruntime.so")
endif()

# Output build instructions
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "ONNX Runtime include directory: ${ONNX_RUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library directory: ${ONNX_RUNTIME_LIB_DIR}")
