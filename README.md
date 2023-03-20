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

# Known Problems & TODO
1. Currently it cannnot achieve comparable accuracy to the TF version on the Cu dataset (rmse_e_val/atom 2e-3, rmse_f_val 1.3e-2)
2. test_descriptor.py fails sometimes when there are multiple neighbors of the same distance. 
This may be because of the unstable sorting algorithm and should not affect the accuracy.
3. 