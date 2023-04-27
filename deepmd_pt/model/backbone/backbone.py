import numpy as np
import torch

from deepmd_pt.utils import env

try:
    from typing import Final
except:
    from torch.jit import Final


class BackBone(torch.nn.Module):
    def __init__(self, **kwargs):
        """
        BackBone base method.
        """
        super(BackBone, self).__init__()

    def forward(self, **kwargs):
        """Calculate backBone.
        """
        raise NotImplementedError
