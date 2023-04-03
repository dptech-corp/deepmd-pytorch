cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test)

set(Torch_DIR /root/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

add_executable(test test_model.cpp)
target_link_libraries(test "${TORCH_LIBRARIES}")
set_property(TARGET test PROPERTY CXX_STANDARD 14)