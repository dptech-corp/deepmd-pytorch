import logging
import numpy as np
import torch

from deepmd_pt.utils import env
try:
    from typing_extensions import Final
except:
    from torch.jit import Final

from deepmd_pt.model.network import NonLinearHead
from deepmd_pt.model.task import TaskBaseMethod


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
        
        if not isinstance(self.attn_head, list):
            self.pair2coord_proj = NonLinearHead(self.attn_head, 1, activation_fn=activation_function)
        else:
            self.pair2coord_proj = []
            self.ndescriptor = len(self.attn_head)
            for ii in range(self.ndescriptor):
                _pair2coord_proj = NonLinearHead(self.attn_head[ii], 1, activation_fn=activation_function)
                self.pair2coord_proj.append(_pair2coord_proj)

    def forward(self, coord, pair_weights, diff, nlist_mask):
        """Calculate the updated coord.
        Args:
        - coord: Input noisy coord with shape [nframes, nloc, 3].
        - pair_weights: Input pair weights with shape [nframes, nloc, nnei, head].
        - diff: Input pair relative coord list with shape [nframes, nloc, nnei, 3].
        - nlist_mask: Input nlist mask with shape [nframes, nloc, nnei].

        Returns:
        - denoised_coord: Denoised updated coord with shape [nframes, nloc, 3].
        """
        # [nframes, nloc, nnei, 1]
        if not isinstance(self.attn_head, list):
            attn_probs = self.pair2coord_proj(pair_weights)
            coord_update = (attn_probs * diff).sum(dim=-2) / nlist_mask.sum(dim=-1).unsqueeze(-1)
            return coord + coord_update
        else:
            all_coord_update = []
            assert len(pair_weights) == len(diff) == len(nlist_mask) == self.ndescriptor
            for ii in range(self.ndescriptor):
                _attn_probs = self.pair2coord_proj[ii](pair_weights[ii])
                _coord_update = (_attn_probs * diff[ii]).sum(dim=-2) / nlist_mask[ii].sum(dim=-1).unsqueeze(-1)
                all_coord_update.append(coord+_coord_update)
            out_coord = 0.5 * all_coord_update[0] + 0.5 *  all_coord_update[1]
            return out_coord
