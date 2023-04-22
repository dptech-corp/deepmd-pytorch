import numpy as np
import os
import torch

GLOBAL_NP_FLOAT_PRECISION = np.float64
GLOBAL_PT_FLOAT_PRECISION = torch.float64
GLOBAL_ENER_FLOAT_PRECISION = np.float64

# Make sure DDP uses correct device if applicable
LOCAL_RANK = os.environ.get("LOCAL_RANK")
LOCAL_RANK = int(0 if LOCAL_RANK is None else LOCAL_RANK)

if os.environ.get("DEVICE") == "cpu" or torch.cuda.is_available() is False:
    DEVICE = torch.device('cpu')
else:
    DEVICE=torch.device(f"cuda:{LOCAL_RANK}")

if os.environ.get("PREPROCESS_DEVICE") == "gpu":
    PREPROCESS_DEVICE = torch.device(f'cuda:{LOCAL_RANK}')
else:
    PREPROCESS_DEVICE = torch.device('cpu')

JIT = False
CACHE_PER_SYS = 5 # keep at most so many sets per sys in memory
TEST_CONFIG = 'tests/water/se_e2_a.json'
ENERGY_BIAS_TRAINABLE = True
