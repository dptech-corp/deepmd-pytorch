import logging
import os
import copy
import numpy as np
import torch
from typing import Optional, List
from deepmd_pt.model.descriptor import DescrptSeAtten
from deepmd_pt.model.task import EnergyFittingNetType
from deepmd_pt.model.network import TypeEmbedNet
from deepmd_pt.utils.stat import compute_output_stats, make_stat_input
from deepmd_pt.utils import env
from deepmd_pt.model.model import BaseModel


class EnergyModelDPA1(BaseModel):

    def __init__(self, model_params, sampled=None):
        """Based on components, construct a DPA-1 model for energy.

        Args:
        - model_params: The Dict-like configuration with model options.
        - sampled: The sampled dataset for stat.
        """
        super(EnergyModelDPA1, self).__init__()
        model_params = copy.deepcopy(model_params)
        # Descriptor + Type Embedding Net
        ntypes = len(model_params['type_map'])
        self.ntypes = ntypes
        descriptor_param = model_params.pop('descriptor')
        descriptor_param['ntypes'] = ntypes
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

        assert self.descriptor_type == 'se_atten', 'Only descriptor `se_atten` is supported for DPA-1!'
        self.descriptor = DescrptSeAtten(**descriptor_param)

        # Fitting
        fitting_param = model_params.pop('fitting_net')
        assert fitting_param.pop('type', 'ener'), 'Only fitting net `ener` is supported!'
        fitting_param['ntypes'] = 1
        fitting_param['embedding_width'] = self.descriptor.dim_out + self.tebd_dim
        fitting_param['use_tebd'] = True

        # Statistics
        self.compute_or_load_stat(model_params, fitting_param, ntypes, sampled=sampled)

        self.fitting_net = EnergyFittingNetType(**fitting_param)

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
        extended_coord = torch.gather(coord, dim=1, index=index)
        extended_coord = extended_coord - shift
        extended_coord.requires_grad_(True)
        atype_tebd = self.type_embedding(atype)
        selected_type[selected_type == -1] = self.ntypes
        nlist_tebd = self.type_embedding(selected_type)

        descriptor, env_mat, _, _ = self.descriptor(extended_coord, selected, atype, selected_type,
                                                 atype_tebd=atype_tebd, nlist_tebd=nlist_tebd)
        atom_energy = self.fitting_net(descriptor, atype, atype_tebd)
        energy = atom_energy.sum(dim=1)
        faked_grad = torch.ones_like(energy)
        lst = torch.jit.annotate(List[Optional[torch.Tensor]], [faked_grad])
        extended_force = torch.autograd.grad([energy], [extended_coord], grad_outputs=lst, create_graph=True)[0]
        assert extended_force is not None
        virial = -torch.transpose(extended_coord, 1, 2) @ extended_force
        mapping = mapping.unsqueeze(-1).expand(-1, -1, 3)
        force = torch.zeros_like(coord)
        force = torch.scatter_reduce(force, 1, index=mapping, src=extended_force, reduce='sum')
        force = -force
        model_predict = {'energy': energy,
                         'force': force,
                         'virial': virial,
                         }
        return model_predict
