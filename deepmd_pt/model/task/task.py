import logging
import numpy as np
import torch


class TaskBaseMethod(torch.nn.Module):

    def __init__(self, **kwargs):
        """Construct a basic head for different tasks.
        """
        super(TaskBaseMethod, self).__init__()

    def forward(self, inputs, atype):
        """Task Output.
        """
        raise NotImplementedError
