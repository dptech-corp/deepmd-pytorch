import logging
import os
import copy
import numpy as np
import torch
from typing import Optional, List
from deepmd_pt.model.descriptor import DescrptSeUni
from deepmd_pt.model.task import DipoleFittingNetType, EnergyFittingNetType
from deepmd_pt.model.network import TypeEmbedNet
from deepmd_pt.utils.stat import compute_output_stats, make_stat_input
from deepmd_pt.utils import env
from deepmd_pt.model.model import BaseModel


class ForceModelDPAUni(BaseModel):

    def __init__(self, model_params, sampled=None):
        """Based on components, construct a DPA-1 model for energy.

        Args:
        - model_params: The Dict-like configuration with model options.
        - sampled: The sampled dataset for stat.
        """
        super(ForceModelDPAUni, self).__init__()
        model_params = copy.deepcopy(model_params)
        # Descriptor + Type Embedding Net
        ntypes = len(model_params['type_map'])
        self.ntypes = ntypes
        descriptor_param = model_params.pop('descriptor')
        descriptor_param['ntypes'] = ntypes

        self.descriptor_type = descriptor_param['type']
        assert self.descriptor_type == 'se_uni', 'Only descriptor `se_uni` is supported for DPA-1!'
        self.descriptor = DescrptSeUni(**descriptor_param)

        # Fitting
        fitting_param = model_params.pop('fitting_net')
        supported_fitting = ['direct_force', 'direct_force_ener']
        self.fitting_type = fitting_param.pop('type')
        assert self.fitting_type in supported_fitting, f'Only support fitting net {supported_fitting}!'
        fitting_param['ntypes'] = 1
        fitting_param['embedding_width'] = self.descriptor.dim_out
        fitting_param['out_dim'] = self.descriptor.dim_emb
        fitting_param['use_tebd'] = True
        self.fitting_net_force = DipoleFittingNetType(**fitting_param)

        # Statistics
        self.compute_or_load_stat(model_params, fitting_param, ntypes, sampled=sampled)

        if self.fitting_type == 'direct_force_ener':
            self.fitting_net = EnergyFittingNetType(**fitting_param)

    def forward(
            self,
            coord, atype, natoms,
            mapping, shift,
            selected,
            selected_type,
            selected_loc: Optional[torch.Tensor] = None,
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
        nlist, nlist_type, nlist_loc = selected, selected_type, selected_loc
        index = mapping.unsqueeze(-1).expand(-1, -1, 3)
        # index nframes x nall x 3
        # coord nframes x nloc x 3
        extended_coord = torch.gather(coord, dim=1, index=index)
        extended_coord = extended_coord - shift
        # extended_coord.requires_grad_(True)
        descriptor, env_mat, _, rot_mat = self.descriptor(
            extended_coord,
            nlist,
            atype,
            nlist_type,
            nlist_loc,
        )
        force_out = self.fitting_net_force(descriptor, atype, None, rot_mat)
        model_predict = {'force': force_out}
        if self.fitting_type == 'direct_force_ener':
            atom_energy = self.fitting_net(descriptor, atype, None)
            energy = atom_energy.sum(dim=1)
            model_predict['energy'] = energy
        return model_predict
