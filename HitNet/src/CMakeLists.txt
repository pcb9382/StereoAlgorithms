﻿cmake_minimum_required (VERSION 3.8)

project(HitNet)
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O0 -g -rdynamic -g2 -ggdb")


SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/HitNet/lib)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)


include_directories(${CMAKE_SOURCE_DIR}/HitNet/include)
include_directories(${CMAKE_SOURCE_DIR}/common)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)

file(GLOB lib_header "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB common_header "${CMAKE_SOURCE_DIR}/common/*.h")

file(GLOB lib_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB common_src "${CMAKE_SOURCE_DIR}/common/*.cpp")


include_directories(${TensorRT_INCLUDE})
link_directories(${TensorRT_LIB})

cuda_add_library(${PROJECT_NAME} SHARED ${lib_src} ${common_src} HitNet_preprocess.cu)

target_link_libraries(${PROJECT_NAME} "/usr/local/cuda/lib64/libcudart.so")
target_link_libraries(${PROJECT_NAME} "${TensorRT_LIB}/libnvinfer.so")
target_link_libraries(${PROJECT_NAME} "${TensorRT_LIB}/libnvonnxparser.so")
target_link_libraries(${PROJECT_NAME} "${TensorRT_LIB}/libnvinfer_plugin.so")
target_link_libraries(${PROJECT_NAME} "${TensorRT_LIB}/libcudnn.so.8.2.1")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})