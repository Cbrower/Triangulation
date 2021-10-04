cmake_minimum_required(VERSION 3.2.2)
project(CUDACommon)

add_library( CUDACommon
    cudaHelpers.cu
)

target_include_directories( CUDACommon PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
