import numpy as np
import torch

GLOBAL_NP_FLOAT_PRECISION = np.float64
GLOBAL_PT_FLOAT_PRECISION = torch.float64
GLOBAL_ENER_FLOAT_PRECISION = np.float64
DISTANCE_INF = 1e8
DEVICE = torch.device(0)
JIT = True
ENERGY_BIAS_TRAINABLE = False