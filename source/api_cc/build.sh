rm -rf build
mkdir build
cd build
export Eigen3_DIR=/opt/mamba/envs/dppt/share/eigen3/cmake
export CMAKE_PREFIX_PATH=$(python -c "import torch;print(torch.__path__[0])")/share/cmake:$CMAKE_PREFIX_PATH
cmake ..
make
