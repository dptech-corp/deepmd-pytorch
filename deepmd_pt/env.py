import numpy as np
import torch

GLOBAL_NP_FLOAT_PRECISION = np.float64
GLOBAL_PT_FLOAT_PRECISION = torch.float64
GLOBAL_ENER_FLOAT_PRECISION = np.float64
DEVICE = torch.device(0)
PREPROCESS_DEVICE = torch.device(0)
JIT = False
CACHE_PER_SYS = 5 # keep at most so many sets per sys in memory
TEST_CONFIG = 'tests/LiGePS/0310.json'

ENERGY_BIAS_TRAINABLE = True