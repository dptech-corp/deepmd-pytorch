This repository is written by Hang'rui Bi based on Shaochen Shi's implementation of DeepMDKit using PyTorch.
It is supposed to offer comparable accuracy and performance to the TF implementation.

# Quick Start
```
ln -s tests/water/data data
PYTHONPATH=/root/deepmd_on_pytorch python3 -u deepmd_pt/main.py train tests/water/se_e2_a.json 2>&1
```
# Profiling
```
# you may change the number of training steps before profiling
PYTHONPATH=/root/deepmd_on_pytorch python3 -m cProfile -o profile deepmd_pt/main.py train tests/water/se_e2_a.json 2>&1
python -m pstats
```
# References
Original DeePMDKit on TensorFlow https://github.com/deepmodeling/deepmd-kit
DeepMD on PyTorch demo https://github.com/shishaochen/deepmd_on_pytorch
# Structure 
```
# model
model.py
    emebdding_net.py
        descriptor.py (differentiable part, se_a mat)
    fitting.py
    stat.py
dataset.py
    descriptor.py (non-differentiable part, env_mat)

# training & inference
main.py
    train.py
        loss.py
        learning_rate.py
    inference.py

# misc
my_random.py
env.py
```

# Deploy
first install libtorch
```
python test.py
mkdir build
cd build
cmake  -DCMAKE_PREFIX_PATH=/root/libtorch/share/cmake/Torch -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc  ..
cmake --build .
make
```
# Known Problems & TODO
1. Pass test_stat.py on the Cu dataset.