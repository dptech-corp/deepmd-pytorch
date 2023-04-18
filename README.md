This repository is written by Hang'rui Bi based on Shaochen Shi's implementation of DeePMD-kit using PyTorch.
It is supposed to offer comparable accuracy and performance to the TF implementation.

# Quick Start

## Install

This package requires PyTorch 2.
```bash
# PyTorch 2 recommends Python >= 3.8 .
conda create -n deepmd-pt python=3.10
conda activate deepmd-pt
# Following instructions on pytorch.org
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
git clone https://github.com/dptech-corp/deepmd-pytorch.git
pip install deepmd-pytorch

# ... or
pip install git+https://github.com/dptech-corp/deepmd-pytorch.git
```

## Run

```bash
conda activate deepmd-pt
python3 deepmd_pt/main.py train tests/water/se_e2_a.json
```

# Profiling

```bash
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

```bash
python test.py
export CMAKE_PREFIX_PATH=`python -c "import torch;print(torch.__path__[0])"`/share/cmake:$CMAKE_PREFIX_PATH
cmake -B build
cd build
cmake --build .
```

# Test
First modify TEST_CONFIG in env.py to the input config you want to test. For example, `tests/water/se_e2.json` is the config for a tiny water problem. The water dataset is contained in the repository.

The tests are aligned with deepmdkit 2.1.5, may fail with deepmdkit 2.2 or higher.

# Distributed Data Parallelism
The systems are shared by processes.

``` python
systems = [item for i, item in enumerate(systems) if i%world_size == rank]
```

## Run on local machine
We use [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html#usage) to start a training session.
To launch a DDP task, one can set `nnodes` as the number of available nodes, and `nproc_per_node` as the number of available GPUs in one node. Please make sure that every node can access the rendezvous address and port.

```bash
OMP_NUM_THREADS=4 torchrun --rdzv_endpoint=localhost:12321 --nnodes=1 --nproc_per_node=2  deepmd_pt/main.py train tests/water/se_e2_a.json
```

> **Note** for developers: `torchrun` by default passes settings as environment variables [(list here)](https://pytorch.org/docs/stable/elastic/run.html#environment-variables).

> To check forward, backward, and communication time, please set env var `TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL`. More details can be found [here](https://pytorch.org/docs/stable/distributed.html#logging).

## Run on slurm system
Use .sbatch file in slurm/, you may need to modify some config to run on your system

```bash
sbatch distributed_data_parallel_slurm_setup.sbatch
```

These files are modified from: https://github.com/lkskstlr/distributed_data_parallel_slurm_setup

# Known Problems & TODO