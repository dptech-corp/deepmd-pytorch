cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(test)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)

# Add deepmd_cc shared library
add_library(deepmd_cc SHARED src/DeepPot.cc src/common.cc)

# Include necessary directories
include_directories(include)

# Link against required libraries
target_link_libraries(deepmd_cc ${TORCH_LIBRARIES} Eigen3::Eigen)
