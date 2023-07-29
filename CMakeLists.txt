cmake_minimum_required (VERSION 3.8)
project (StereoAlgorithmDemo)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# tensorrt 目录
set(TensorRT_INCLUDE "/home/pcb/TensorRT-8.2.5.1/include" CACHE INTERNAL "TensorRT Library include location")
set(TensorRT_LIB "/home/pcb/TensorRT-8.2.5.1/lib" CACHE INTERNAL "TensorRT Library lib location")

#添加子目录
add_subdirectory("RAFTStereo")
add_subdirectory("HitNet")
add_subdirectory("CREStereo")
add_subdirectory("Stereo_Calibration")

