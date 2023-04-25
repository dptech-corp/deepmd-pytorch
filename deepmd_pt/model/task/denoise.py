import logging
import numpy as np
import torch

from deepmd_pt.utils import env
try:
    from typing_extensions import Final
except:
    from torch.jit import Final

from deepmd_pt.model.network.network import NonLinearHead
from deepmd_pt.model.task.task import TaskBaseMethod
from IPython import embed


class DenoiseNet(TaskBaseMethod):

    def __init__(self,
                 attn_head=8,
                 activation_function="gelu",
                 **kwargs):
        """Construct a denoise net.

        Args:
        - ntypes: Element count.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neurons in each hidden layers of the fitting net.
        - bias_atom_e: Average enery per atom for each element.
        - resnet_dt: Using time-step in the ResNet construction.
        """
        super(DenoiseNet, self).__init__()
        self.attn_head = attn_head
        self.pair2coord_proj = NonLinearHead(self.attn_head, 1, activation_fn=activation_function)

    def forward(self, coord, pair_weights, diff, nlist_mask):
        """Calculate the updated coord.
        Args:
        - coord: Input noisy coord with shape [nframes, nloc, 3].
        - pair_weights: Input pair weights with shape [nframes, nloc, nnei, 3].
        - diff: Input pair relative coord list with shape [nframes, nloc, nnei, head].
        - nlist_mask: Input nlist mask with shape [nframes, nloc, nnei].

        Returns:
        - denoised_coord: Denoised updated coord with shape [nframes, nloc, 3].
        """
        # [nframes, nloc, nnei, 1]
        attn_probs = self.pair2coord_proj(pair_weights)
        coord_update = (attn_probs * diff).sum(dim=-2) / nlist_mask.sum(dim=-1).unsqueeze(-1)
        return coord + coord_update
