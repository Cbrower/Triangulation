cmake_minimum_required(VERSION 3.2.2)
project(Triangulator)

# Enabling CUDA
enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

# Finding LAPack
find_package(LAPACK)

# Set import CMakeFlags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install")

# Call the subdirectories to build their components
add_subdirectory(common)
add_subdirectory(triangulators)

add_executable(Triangulator main.cu)
if (LAPACK_FOUND)
  target_link_libraries(Triangulator LINK_PUBLIC Triangulators ${LAPACK_LIBRARIES})
endif() # TODO Add an else and make the compilation fail

install(TARGETS Triangulator DESTINATION bin)
install(DIRECTORY cones DESTINATION .)
