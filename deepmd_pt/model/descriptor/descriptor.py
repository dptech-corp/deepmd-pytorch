import numpy as np
import torch

from deepmd_pt.utils import env

try:
    from typing import Final
except:
    from torch.jit import Final


class Descriptor(torch.nn.Module):
    def __init__(self, **kwargs):
        """
        Descriptor base method.
        """
        super(Descriptor, self).__init__()

    @property
    def dim_out(self):
        """
        Returns the output dimension of this descriptor
        """
        return self.filter_neuron[-1] * self.axis_neuron

    def compute_input_stats(self, merged):
        """Update mean and stddev for descriptor elements.
        """
        raise NotImplementedError

    def forward(self, **kwargs):
        """Calculate descriptor.
        """
        raise NotImplementedError


def compute_std(sumv2, sumv, sumn, rcut_r):
    """Compute standard deviation."""
    if sumn == 0:
        return 1.0 / rcut_r
    val = np.sqrt(sumv2 / sumn - np.multiply(sumv / sumn, sumv / sumn))
    if np.abs(val) < 1e-2:
        val = 1e-2
    return val
