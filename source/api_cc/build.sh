rm -rf build 
mkdir build
cd build
export CMAKE_PREFIX_PATH=`python -c "import torch;print(torch.__path__[0])"`/share/cmake:$CMAKE_PREFIX_PATH
cmake  ..
make
