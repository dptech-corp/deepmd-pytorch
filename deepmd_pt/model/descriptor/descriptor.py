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
