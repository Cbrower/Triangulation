cmake_minimum_required(VERSION 3.2.2)
project(Triangulators)

add_library(Triangulators
    lexTriangulator.cpp
)

target_include_directories(Triangulators PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if (USE_CUDA)
    target_link_libraries(Triangulators LINK_PUBLIC Common CUDACommon ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)
else()
    target_link_libraries(Triangulators LINK_PUBLIC Common ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)
endif()
target_compile_options(Triangulators PRIVATE -Wall -Werror)
