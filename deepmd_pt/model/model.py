import numpy as np
import torch


class BaseModel(torch.nn.Module):

    def __init__(self):
        """Construct a basic model for different tasks.
        """
        super(BaseModel, self).__init__()

    def forward(self, coord, atype, natoms, mapping, shift, selected, box):
        """Model output.
        """
        raise NotImplementedError
