import logging
import os
import copy
import numpy as np
import torch
from typing import Optional, List
from deepmd_pt.model.descriptor import DescrptSeAtten, DescrptSeUni, DescrptHybrid, Descriptor
from deepmd_pt.model.task import Fitting
from deepmd_pt.model.network import TypeEmbedNet
from deepmd_pt.utils.stat import compute_output_stats, make_stat_input
from deepmd_pt.utils import env
from deepmd_pt.model.model import BaseModel


class EnergyModelHybrid(BaseModel):

    def __init__(self, model_params, sampled=None):
        """Based on components, construct a DPA-1 model for energy.

        Args:
        - model_params: The Dict-like configuration with model options.
        - sampled: The sampled dataset for stat.
        """
        super(EnergyModelHybrid, self).__init__()
        model_params = copy.deepcopy(model_params)
        # Descriptor + Type Embedding Net
        ntypes = len(model_params['type_map'])
        self.ntypes = ntypes
        type_embedding_param = model_params.pop('type_embedding', None)
        if type_embedding_param is None:
            self.type_embedding = TypeEmbedNet(ntypes, 8)
            self.tebd_dim = 8
        else:
            tebd_dim = type_embedding_param.get('neuron', [8])[-1]
            self.type_embedding = TypeEmbedNet(ntypes, tebd_dim)
            self.tebd_dim = tebd_dim

        descriptor_param = model_params.pop('descriptor')
        self.descriptor_type = descriptor_param['type']
        descriptor_param['ntypes'] = ntypes
        assert self.descriptor_type == 'hybrid', 'Only descriptor `hybrid` is supported for hybrid model!'
        if type_embedding_param is None:
            descriptor_param['tebd_dim'] = 8
            descriptor_param['tebd_input_mode'] = 'concat'
        else:
            tebd_dim = type_embedding_param.get('neuron', [8])[-1]
            tebd_input_mode = type_embedding_param.get('tebd_input_mode', 'concat')
            descriptor_param['tebd_dim'] = tebd_dim
            descriptor_param['tebd_input_mode'] = tebd_input_mode
        self.descriptor = Descriptor(**descriptor_param)

        # Fitting
        fitting_param = model_params.pop('fitting_net')
        assert fitting_param.pop('type', 'ener'), 'Only fitting net `ener` is supported!'
        fitting_param['type'] = 'ener'
        fitting_param['ntypes'] = 1
        fitting_param['embedding_width'] = self.descriptor.dim_out + self.tebd_dim
        fitting_param['use_tebd'] = True

        # Statistics
        self.compute_or_load_stat(fitting_param, ntypes,
                                  resuming=model_params.get("resuming", False),
                                  type_map=model_params['type_map'],
                                  stat_file_dir=model_params.get("stat_file_dir", None),
                                  stat_file_path=model_params.get("stat_file_path", None),
                                  sampled=sampled)

        self.fitting_net = Fitting(**fitting_param)

    def forward(self, coord, atype, natoms, mapping, shift, nlist, nlist_type: Optional[torch.Tensor] = None,
                nlist_loc: Optional[torch.Tensor] = None, box: Optional[torch.Tensor] = None):
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

        nlist_tebd = []
        for nlist_type_item in nlist_type:
            nlist_type_item[nlist_type_item == -1] = self.ntypes
            nlist_tebd.append(self.type_embedding(nlist_type_item))

        descriptor, _, _, _ = self.descriptor(extended_coord, nlist, atype, nlist_type,
                                              nlist_loc=nlist_loc, atype_tebd=atype_tebd, nlist_tebd=nlist_tebd)
        atom_energy, _ = self.fitting_net(descriptor, atype, atype_tebd)
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
                         'atom_energy': atom_energy,
                         'force': force,
                         'virial': virial,
                         }
        return model_predict
