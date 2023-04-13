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
Tested with libtorch pre-CXX11 abi cu116, cuda 11.6, torch 1.13
```
python test.py
mkdir build
cd build
cmake  -DCMAKE_PREFIX_PATH=/root/libtorch_cu116/libtorch/share/cmake/Torch -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc  ..
cmake --build .
make
```
# Test
First modify TEST_CONFIG in env.py to the input config you want to test. For example, `tests/water/se_e2.json` is the config for a tiny water problem. The water dataset is contained in the repository.

The tests are aligned with deepmdkit 2.1.5, may fail with deepmdkit 2.2 or higher.

# Distributed Data Parallelism
The systems are shared by processes.
``` 
systems = [item for i, item in enumerate(systems) if i%world_size == rank]
```

## Run on local machine
```
torchrun --rdzv_endpoint=localhost:12321 --nnodes=1 --nproc_per_node=2  deepmd_pt/main.py train tests/water/se_e2_a.json
```

## Run on slurm system 
Use .sbatch file in slurm/, you may need to modify some config to run on your system

```
sbatch distributed_data_parallel_slurm_setup.sbatch
```

These files are modified from: https://github.com/lkskstlr/distributed_data_parallel_slurm_setup

# Known Problems & TODO