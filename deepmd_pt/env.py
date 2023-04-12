import numpy as np
import os
import torch

GLOBAL_NP_FLOAT_PRECISION = np.float64
GLOBAL_PT_FLOAT_PRECISION = torch.float64
GLOBAL_ENER_FLOAT_PRECISION = np.float64
if os.environ.get("DEVICE") == "cpu":
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda')
if os.environ.get("PREPROCESS_DEVICE") == "cpu":
    PREPROCESS_DEVICE = torch.device('cpu')
else:
    PREPROCESS_DEVICE = torch.device('cuda')
JIT = False
CACHE_PER_SYS = 5 # keep at most so many sets per sys in memory
TEST_CONFIG = 'tests/water/se_e2_a.json'
WORLD_SIZE = 1
ENERGY_BIAS_TRAINABLE = True
