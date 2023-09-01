cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(test)
set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_STANDARD_REQUIRED ON) 

set(SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# Find required packages
find_package (Eigen3 REQUIRED)
find_package(Torch REQUIRED)

# Get the path from python command
execute_process(COMMAND python -c "import torch;print(torch.__path__[0])"
                OUTPUT_VARIABLE TORCH_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# Append the path to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PATH}/share/cmake")

# Add deepmd_cc shared library
add_library(deepmd_cc SHARED ${SOURCE_PATH}/source/api_cc/src/DeepPot.cc ${SOURCE_PATH}/source/api_cc/src/common.cc)

# Include necessary directories
include_directories(${SOURCE_PATH}/source/api_cc/include /opt/mamba/envs/dppt/include/eigen3/)

# Link against required libraries
target_link_libraries(deepmd_cc ${TORCH_LIBRARIES} Eigen3::Eigen)

