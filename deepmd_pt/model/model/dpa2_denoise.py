import numpy as np
import torch
from typing import Optional, List
from deepmd_pt.model.descriptor import DescrptSeAtten
from deepmd_pt.model.task import DenoiseNet, TypePredictNet
from deepmd_pt.model.network import TypeEmbedNet
from deepmd_pt.model.backbone import Evoformer2bBackBone
from deepmd_pt.utils.stat import compute_output_stats, make_stat_input
from deepmd_pt.utils import env
from deepmd_pt.model.model import BaseModel


class DenoiseModelDPA2(BaseModel):

    def __init__(self, model_params, sampled=None):
        """Based on components, construct a DPA-1 model for energy.

        Args:
        - model_params: The Dict-like configuration with model options.
        - sampled: The sampled dataset for stat.
        """
        super(DenoiseModelDPA2, self).__init__()
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
            tebd_dim = type_embedding_param['neuron'][-1]
            tebd_input_mode = type_embedding_param.get('tebd_input_mode', 'concat')
            self.type_embedding = TypeEmbedNet(ntypes, tebd_dim)
            descriptor_param['tebd_dim'] = tebd_dim
            descriptor_param['tebd_input_mode'] = tebd_input_mode
            self.tebd_dim = tebd_dim

        self.descriptor_type = descriptor_param['type']

        assert self.descriptor_type == 'se_atten', 'Only descriptor `se_atten` is supported for DPA-2!'
        self.descriptor = DescrptSeAtten(**descriptor_param)

        # Statistics
        self.compute_or_load_stat(model_params, {}, ntypes, sampled=sampled)

        # BackBone
        backbone_param = model_params.pop('backbone')
        backbone_type = backbone_param.pop('type')
        backbone_param['atomic_dim'] = self.descriptor.dim_out
        backbone_param['pair_dim'] = self.descriptor.dim_emb
        backbone_param['nnei'] = self.descriptor.nnei
        if backbone_type == 'evo-2b':
            self.backbone = Evoformer2bBackBone(**backbone_param)
        else:
            NotImplementedError(f"Unknown backbone type {backbone_type}!")

        assert model_params.pop('fitting_net', None) is None, f'Denoise task must not have fitting_net!'
        # Denoise and predict
        self.coord_denoise_net = DenoiseNet(self.backbone.attn_head, self.backbone.activation_function)
        self.type_predict_net = TypePredictNet(self.backbone.feature_dim, self.ntypes - 1,
                                               # last type is `MASKED_TOKEN`
                                               self.backbone.activation_function)

    def forward(self, coord, atype, natoms, mapping, shift, selected, selected_type, selected_loc: Optional[torch.Tensor]=None, box: Optional[torch.Tensor]=None, **kwargs):
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
        # extended_coord.requires_grad_(True)
        atype_tebd = self.type_embedding(atype)
        selected_type[selected_type == -1] = self.ntypes
        nlist_tebd = self.type_embedding(selected_type)
        nnei_mask = selected != -1
        padding_selected_loc = selected_loc * nnei_mask

        descriptor, env_mat, diff = self.descriptor(extended_coord, selected, atype, selected_type, atype_tebd, nlist_tebd)
        atomic_rep, transformed_atomic_rep, pair_rep, delta_pair_rep, norm_x, norm_delta_pair_rep = \
            self.backbone(descriptor, env_mat, padding_selected_loc, selected_type, nnei_mask)

        updated_coord = self.coord_denoise_net(coord, delta_pair_rep, diff, nnei_mask)
        logits = self.type_predict_net(atomic_rep)
        model_predict = {'updated_coord': updated_coord,
                         'logits': logits,
                         'norm_x': norm_x,
                         'norm_delta_pair_rep': norm_delta_pair_rep,
                         }
        return model_predict
