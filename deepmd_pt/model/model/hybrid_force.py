import logging
import os
import copy
import numpy as np
import torch
from typing import Optional, List
from deepmd_pt.model.descriptor import DescrptSeAtten, DescrptSeUni, DescrptHybrid, DescrptGaussianLcc
from deepmd_pt.model.task import EnergyFittingNetType, DipoleFittingNetType, FittingNetAttenLcc
from deepmd_pt.model.network import TypeEmbedNet
from deepmd_pt.utils.stat import compute_output_stats, make_stat_input
from deepmd_pt.utils import env
from deepmd_pt.model.model import BaseModel


class ForceModelHybrid(BaseModel):

    def __init__(self, model_params, sampled=None):
        """Based on components, construct a DPA-1 model for energy.

        Args:
        - model_params: The Dict-like configuration with model options.
        - sampled: The sampled dataset for stat.
        """
        super(ForceModelHybrid, self).__init__()
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

        supported_descrpt = ['se_atten', 'se_uni', 'gaussian_lcc']
        descriptor_param = model_params.pop('descriptor')
        self.descriptor_type = descriptor_param['type']
        assert self.descriptor_type == 'hybrid', 'Only descriptor `hybrid` is supported for hybrid model!'
        descriptor_list = []
        for descriptor_param_item in descriptor_param['list']:
            descriptor_type_tmp = descriptor_param_item['type']
            assert descriptor_type_tmp in supported_descrpt, \
                f'Only descriptors in {supported_descrpt} are supported for `hybrid` descriptor!'
            descriptor_param_item['ntypes'] = ntypes
            if type_embedding_param is None:
                descriptor_param_item['tebd_dim'] = 8
                descriptor_param_item['tebd_input_mode'] = 'concat'
            else:
                tebd_dim = type_embedding_param.get('neuron', [8])[-1]
                tebd_input_mode = type_embedding_param.get('tebd_input_mode', 'concat')
                descriptor_param_item['tebd_dim'] = tebd_dim
                descriptor_param_item['tebd_input_mode'] = tebd_input_mode
            if descriptor_type_tmp == 'se_atten':
                descriptor_list.append(DescrptSeAtten(**descriptor_param_item))
            elif descriptor_type_tmp == 'se_uni':
                descriptor_list.append(DescrptSeUni(**descriptor_param_item))
            elif descriptor_type_tmp == 'gaussian_lcc':
                descriptor_param_item['num_pair'] = 2 * ntypes
                descriptor_param_item['embed_dim'] = self.tebd_dim
                descriptor_list.append(DescrptGaussianLcc(**descriptor_param_item))
            else:
                RuntimeError("Unsupported descriptor type!")
        self.descriptor = DescrptHybrid(descriptor_list, descriptor_param)

        # Fitting
        supported_fitting = ['direct_force', 'direct_force_ener', 'atten_vec_lcc']
        fitting_param = model_params.pop('fitting_net')
        fitting_type = fitting_param.pop('type')
        self.fitting_type = fitting_type
        assert fitting_type in supported_fitting, f'Only support fitting net {supported_fitting}!'
        fitting_param['ntypes'] = 1
        if self.fitting_type in ['direct_force', 'direct_force_ener']:
            fitting_param['embedding_width'] = self.descriptor.dim_out + self.tebd_dim
            fitting_param['out_dim'] = self.descriptor.dim_emb
            self.fitting_net_force = DipoleFittingNetType(**fitting_param)

            # Statistics
            self.compute_or_load_stat(model_params, fitting_param, ntypes, sampled=sampled)

            if fitting_type == 'direct_force_ener':
                self.fitting_net_ener = EnergyFittingNetType(**fitting_param)

        elif self.fitting_type == 'atten_vec_lcc':
            assert self.descriptor.hybrid_mode == 'sequential', \
                'Only sequential hybrid mode can use atten_vec_lcc fitting net'
            assert self.descriptor.descriptor_list[-1].local_cluster, \
                "The last descriptor in hybrid list must be local_cluster when using atten_vec_lcc fitting"
            fitting_param['embedding_width'] = self.descriptor.descriptor_list[-1].dim_out
            fitting_param['pair_embed_dim'] = self.descriptor.descriptor_list[-1].dim_emb
            fitting_param['attention_heads'] = self.descriptor.descriptor_list[-1].attention_heads

            # Statistics
            self.compute_or_load_stat(model_params, fitting_param, ntypes, sampled=sampled)

            self.fitting_net = FittingNetAttenLcc(**fitting_param)

    def forward(self, coord, atype, natoms, mapping, shift, selected, selected_type: Optional[torch.Tensor] = None,
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
        # extended_coord.requires_grad_(True)
        atype_tebd = self.type_embedding(atype)

        nlist_tebd = []
        for selected_type_item in selected_type:
            selected_type_item[selected_type_item == -1] = self.ntypes
            nlist_tebd.append(self.type_embedding(selected_type_item))

        atomic_rep, pair_rep, delta_pos, rot_mat = self.descriptor(extended_coord, selected, atype, selected_type,
                                                                   nlist_loc=selected_loc, atype_tebd=atype_tebd,
                                                                   nlist_tebd=nlist_tebd)
        if self.fitting_type in ['direct_force', 'direct_force_ener']:
            force_out = self.fitting_net_force(atomic_rep, atype, atype_tebd, rot_mat)
            model_predict = {'force': force_out}
            if self.fitting_type == 'direct_force_ener':
                atom_energy = self.fitting_net_ener(atomic_rep, atype, atype_tebd)
                energy = atom_energy.sum(dim=1)
                model_predict['energy'] = energy
            return model_predict

        elif self.fitting_type == 'atten_vec_lcc':
            energy_out, predict_force_nloc = self.fitting_net(atomic_rep, pair_rep, delta_pos, atype, nframes, nloc)
            force_target_mask = torch.ones_like(atype).type_as(predict_force_nloc).unsqueeze(-1)
            model_predict = {'energy': energy_out,
                             'force': predict_force_nloc,
                             # 'force_target_mask': force_target_mask,
                             'updated_coord': predict_force_nloc + coord,
                             }
            return model_predict
        else:
            raise RuntimeError
