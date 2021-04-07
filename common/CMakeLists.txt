cmake_minimum_required(VERSION 3.2.2)
project(Common)

add_library( Common
    common.cpp
)

target_include_directories( Common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
