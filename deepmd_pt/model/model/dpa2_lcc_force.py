import logging
import os
import copy
import numpy as np
import torch
from typing import Optional, List
from deepmd_pt.model.descriptor import DescrptGaussianLcc
from deepmd_pt.model.task import FittingNetAttenLcc
from deepmd_pt.model.network import TypeEmbedNet
from deepmd_pt.utils import env
from deepmd_pt.model.model import BaseModel


class ForceModelDPA2Lcc(BaseModel):

    def __init__(self, model_params, sampled=None):
        """Local cluster model for DPA-2 energy and force prediction.

        Args:
        - model_params: The Dict-like configuration with model options.
        - sampled: The sampled dataset for stat.
        """
        super(ForceModelDPA2Lcc, self).__init__()
        model_params = copy.deepcopy(model_params)
        # Descriptor + Type Embedding Net
        ntypes = len(model_params['type_map'])
        self.ntypes = ntypes
        descriptor_param = model_params.pop('descriptor')
        type_embedding_param = model_params.pop('type_embedding', None)
        if type_embedding_param is None:
            self.type_embedding = TypeEmbedNet(ntypes, 8)
            descriptor_param['tebd_dim'] = 8
            descriptor_param['tebd_input_mode'] = 'concat'
            self.tebd_dim = 8
        else:
            tebd_dim = type_embedding_param.get('neuron', [8])[-1]
            tebd_input_mode = type_embedding_param.get('tebd_input_mode', 'concat')
            self.type_embedding = TypeEmbedNet(ntypes, tebd_dim)
            descriptor_param['tebd_dim'] = tebd_dim
            descriptor_param['tebd_input_mode'] = tebd_input_mode
            self.tebd_dim = tebd_dim

        self.descriptor_type = descriptor_param['type']
        descriptor_param['ntypes'] = ntypes
        descriptor_param['num_pair'] = 2 * ntypes
        descriptor_param['embed_dim'] = self.tebd_dim

        assert self.descriptor_type == 'gaussian_lcc', 'Only descriptor `gaussian_lcc` is supported for DPA-1!'
        self.descriptor = DescrptGaussianLcc(**descriptor_param)

        # Fitting
        fitting_param = model_params.pop('fitting_net')
        assert fitting_param.pop('type', 'ener') == 'atten_vec_lcc', 'Only fitting net `atten_vec_lcc` is supported!'
        fitting_param['embedding_width'] = self.tebd_dim
        fitting_param['pair_embed_dim'] = self.descriptor.dim_emb
        fitting_param['attention_heads'] = self.descriptor.attention_heads

        # Statistics
        self.compute_or_load_stat(model_params, fitting_param, ntypes, sampled=sampled)

        self.fitting_net = FittingNetAttenLcc(**fitting_param)

    def forward(self, coord, atype, natoms, mapping, shift, selected, selected_type,
                selected_loc: Optional[torch.Tensor] = None, box: Optional[torch.Tensor] = None):
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
        nframes, nloc = coord.shape[:2]
        extended_coord = torch.gather(coord, dim=1, index=index)
        extended_coord = extended_coord - shift
        atype_tebd = self.type_embedding(atype)
        selected_type[selected_type == -1] = self.ntypes
        atomic_rep, pair_rep, delta_pos, _ = self.descriptor(extended_coord, selected, atype, selected_type,
                                                             nlist_loc=selected_loc, atype_tebd=atype_tebd)
        energy_out, predict_force_nloc = self.fitting_net(atomic_rep, pair_rep, delta_pos, atype, nframes, nloc)
        force_target_mask = torch.ones_like(atype).type_as(predict_force_nloc).unsqueeze(-1)

        model_predict = {'energy': energy_out,
                         'force': predict_force_nloc,
                         # 'force_target_mask': force_target_mask,
                         'updated_coord': predict_force_nloc + coord,
                         }
        return model_predict
