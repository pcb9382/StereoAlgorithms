# CMakeList.txt: NuctechPersonBagBinding 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)
project(Stereo_Calibration)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -Wall -g")

#源文件
file(GLOB test_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

#头文件
file(GLOB test_header "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

#Multi threads settings
set(CMAKE_THREAD_LIBS_INIT -lpthred)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

OPTION(DEMO "demo version" ON)
if(DEMO)
    ADD_DEFINITIONS(-DDEMO)
ENDIF(DEMO)

# 将源代码添加到此项目的可执行文件。
add_executable (${PROJECT_NAME} ${test_src} ${test_header})

#第三方库路径
find_package(OpenCV)
IF(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
ELSE()
    MESSAGE(STATUS "Cannot find opencv libs")
ENDIF()

#头文件搜索目录
include_directories(${CMAKE_SOURCE_DIR})

# TODO: 如有需要，请添加测试并安装目标。
