import numpy as np
import torch

from deepmd_pt.utils import env

try:
    from typing import Final
except:
    from torch.jit import Final

from deepmd_pt.model.backbone import BackBone
from deepmd_pt.model.network import Evoformer3bEncoder


class Evoformer3bBackBone(BackBone):
    def __init__(self,
                 **kwargs):
        """Construct an evoformer 3b backBone.

        """
        super(Evoformer3bBackBone, self).__init__()
        self.encoder = Evoformer3bEncoder(**kwargs)

    def forward(self, atomic_feature, pair=None, nlist=None, attn_mask=None, pair_mask=None):
        """Encoder the atomic and pair representations.

        Args:
        - atomic_feature: Atomic representation with shape [nframes, nloc, atomic_dim].
        - pair: Pair representation with shape [nframes, nloc, nnei, pair_dim].
        - attn_mask: Neighbor list with shape [nframes, head, nloc, nnei].
        - pair_mask: Neighbor types with shape [nframes, nloc, nnei].

        Returns:
        - atomic_feature: Atomic representation after encoder with shape [nframes, nloc, atomic_dim].
        - pair: Pair representation with shape [nframes, nloc, nnei, pair_dim].
        """
        atomic_feature, pair = self.encoder(atomic_feature, pair=pair, nlist=nlist, attn_mask=attn_mask,
                                            pair_mask=pair_mask)
        return atomic_feature, pair
