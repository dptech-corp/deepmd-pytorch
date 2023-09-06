import logging
import os
import copy
import numpy as np
import torch
from typing import Optional, List
from deepmd_pt.model.descriptor import DescrptSeUni
from deepmd_pt.model.task import DenoiseNet, TypePredictNet
from deepmd_pt.utils.stat import compute_output_stats, make_stat_input
from deepmd_pt.utils import env
from deepmd_pt.model.model import BaseModel


class DenoiseModelDPAUni(BaseModel):

    def __init__(self, model_params, sampled=None):
        """Based on components, construct a DPA-1 model for energy.

        Args:
        - model_params: The Dict-like configuration with model options.
        - sampled: The sampled dataset for stat.
        """
        super(DenoiseModelDPAUni, self).__init__()
        model_params = copy.deepcopy(model_params)
        # Descriptor + Type Embedding Net
        ntypes = len(model_params['type_map'])
        self.ntypes = ntypes
        descriptor_param = model_params.pop('descriptor')
        descriptor_param['ntypes'] = ntypes

        self.descriptor_type = descriptor_param['type']
        assert self.descriptor_type == 'se_uni', 'Only descriptor `se_uni` is supported for DPA-1!'
        self.descriptor = DescrptSeUni(**descriptor_param)

        # Statistics
        self.compute_or_load_stat(model_params, {}, ntypes, sampled=sampled)

        assert model_params.pop('fitting_net', None) is None, f'Denoise task must not have fitting_net!'
        self.coord_denoise_net = DenoiseNet(self.descriptor.dim_emb, 'tanh')
        self.type_predict_net = TypePredictNet(self.descriptor.dim_out, self.ntypes - 1)

    def forward(
        self, 
        coord, atype, natoms, 
        mapping, shift, 
        nlist,
        nlist_type,
        nlist_loc: Optional[torch.Tensor] = None,
        box: Optional[torch.Tensor] = None):
        """Return total energy of the system.
        Args:
        - coord: Atom coordinates with shape [nframes, natoms[1]*3].
        - atype: Atom types with shape [nframes, natoms[1]].
        - natoms: Atom statisics with shape [self.ntypes+2].
        - box: Simulation box with shape [nframes, 9].
        Returns:
        - energy: Energy per atom.
        - force: XYZ force per atom.
        """
        index = mapping.unsqueeze(-1).expand(-1, -1, 3)
        # index nframes x nall x 3
        # coord nframes x nloc x 3
        extended_coord = torch.gather(coord, dim=1, index=index)
        extended_coord = extended_coord - shift
        extended_coord.requires_grad_(True)
        nlist_type[nlist_type == -1] = self.ntypes
        nnei_mask = nlist != -1

        descriptor, env_mat, diff, _ = self.descriptor(
          extended_coord,
          nlist,
          atype,
          nlist_type,
          nlist_loc, 
        )

        updated_coord = self.coord_denoise_net(coord, env_mat, diff, nnei_mask)
        logits = self.type_predict_net(descriptor)
        model_predict = {'updated_coord': updated_coord,
                         'logits': logits,
                         }
        return model_predict
