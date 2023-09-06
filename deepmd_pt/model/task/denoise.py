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
        self.pair2coord_proj = []

        if isinstance(self.attn_head, list):
            logging.info("it's a list")
            self.nitem = len(attn_head)
            for idx in range(self.nitem):
                _pair2coord_proj = NonLinearHead(self.attn_head[idx], 1, activation_fn=activation_function)
                self.pair2coord_proj.append(_pair2coord_proj)
        else:
            logging.info("it's not a list")
            self.pair2coord_proj = NonLinearHead(self.attn_head, 1, activation_fn=activation_function)

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
        
        # [nframes, nloc, nnei, head]➡️[nframes, nloc, nnei, 1]
        if isinstance(pair_weights, list):
            logging.info("it's a list")
            all_coord_update = []
            assert len(pair_weights) == len(diff) == len(nlist_mask) == self.nitem
            for idx in range(self.nitem):
                _attn_probs = self.pair2coord_proj[idx](pair_weights[idx])
                _coord_update = (_attn_probs * diff[idx]).sum(dim=-2) / nlist_mask[idx].sum(dim=-1).unsqueeze(-1)
                all_coord_update.append(coord+_coord_update)
            logging.info(f"all_coord_update:{all_coord_update}")
            return all_coord_update
        else:
            logging.info("it's not a list")
            attn_probs = self.pair2coord_proj(pair_weights)
            coord_update = (attn_probs * diff).sum(dim=-2) / nlist_mask.sum(dim=-1).unsqueeze(-1)
            return coord + coord_update
        #logging.info(f"attn_probs*diff:{(attn_probs*diff).shape}")
        #logging.info(f"nlist_mask:{((attn_probs * diff).sum(dim=-2)).shape}")

        # diff [nframes, nloc, nnei, 3]
        # attn_prob*diff [nframes, nloc, nnei, 3]
        # nlist_mask [nframes, nloc, nnei]
        # [nframes, nloc, 3]/ [nframes, nloc, 1]      
