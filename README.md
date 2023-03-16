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
# Known Problems
test_descriptor.py fails sometimes when there are multiple neighbors of the same distance. 
This may be because of the unstable sorting algorithm and should not affect the accuracy.