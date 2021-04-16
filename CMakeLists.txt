cmake_minimum_required(VERSION 3.2.2)
project(Triangulator)

# Adding Definitions
option(USE_OPENMP "Should OpenMP be used" 0)
if (USE_OPENMP)
    add_definitions(-DUSE_OPENMP=1)
endif()
option(DO_TIMING "Should timing be done" 0)
if (DO_TIMING)
    add_definitions(-DDO_TIMING=1)
endif()
option(USE_CUDA "Should CUDA be used" 0)
if (USE_CUDA)
    add_definitions(-DUSE_CUDA=1)
endif()

# Enabling CUDA TODO Determine if this is a better way to include CUDA
#enable_language(CUDA)
#include(CheckLanguage)
#check_language(CUDA)

# Finding packages
find_package(LAPACK REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

# Include CUDA
include_directories("${CUDA_INCLUDE_DIRS}")

# Set important CMake Flags
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install")

# Call the subdirectories to build their components
add_subdirectory(common)
add_subdirectory(triangulators)

add_executable(Triangulator main.cpp)
target_link_libraries(Triangulator LINK_PUBLIC Triangulators ${LAPACK_LIBRARIES} ${CUDA_LIBRARIES} -L/usr/local/cuda/lib64 -lcublas -lcudart) # TODO Dynamically find cublas and cudart

install(TARGETS Triangulator DESTINATION bin)
install(DIRECTORY cones DESTINATION .)
