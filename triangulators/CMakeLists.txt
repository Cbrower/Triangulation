cmake_minimum_required(VERSION 3.2.2)
project(Triangulators)

add_library(Triangulators
    lexTriangulator.cu
)

target_include_directories(Triangulators PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Triangulators LINK_PUBLIC Common)
