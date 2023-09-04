import numpy as np
import torch
from typing import Optional, List
from deepmd_pt.model.descriptor import DescrptSeAtten, DescrptSeUni, DescrptHybrid
from deepmd_pt.model.task import DenoiseNet, TypePredictNet
from deepmd_pt.model.network import TypeEmbedNet, SimpleLinear
from deepmd_pt.model.backbone import Evoformer2bBackBone
from deepmd_pt.utils.stat import compute_output_stats, make_stat_input
from deepmd_pt.utils import env
from deepmd_pt.model.model import BaseModel

class DenoiseModelHybrid(BaseModel):

    def __init__(self, model_params, sampled=None):
        """Based on components, construct a hybrid model for denoise.

        Args:
        - model_params: The Dict-like configuration with model options.
        - sampled: The sampled dataset for stat
        """
        super(DenoiseModelHybrid, self).__init__()
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

        supported_descrpt = ['se_atten', 'se_uni']
        descriptor_param = model_params.pop('descriptor')
        self.descriptor_type = descriptor_param['type']
        assert self.descriptor_type == 'hybrid', 'Only descriptor `hybrid` is supported for hybrid model!'
        descriptor_list = []
        for descriptor_param_item in descriptor_param['list']:
            descriptor_type_tmp = descriptor_param_item['type']
            assert descriptor_type_tmp in supported_descrpt, \
                f"Only descriptors in {supported_descrpt} are support for `hybrid` descriptor!"
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
            else:
                RuntimeError("Unsupported descriptor type!")
        self.descriptor = DescrptHybrid(descriptor_list, descriptor_param)

        # Statistics
        self.compute_or_load_stat(model_params, {}, ntypes, sampled=sampled)

        # Denoise and predict
        self.activation_function = model_params.pop("activation_function", "gelu")
        self.coord_denoise_net = DenoiseNet(self.descriptor.dim_emb_list, self.activation_function)
        self.type_predict_net = TypePredictNet(self.descriptor.dim_out, self.ntypes-1, 'gelu')

    def forward(self, coord, atype, natoms, mapping, shift, nlist, nlist_type, nlist_loc: Optional[torch.Tensor]=None, box: Optional[torch.Tensor]=None):
        """Return xxx
        Args:
        - coord: Atom coordinates with shape [nframes, natoms[1], 3]
        - atype: Atom types with shape [nframes, natoms[1]]
        - natoms: Atom statisics with shape [self.ntypes+2]
        - box: Simulation box with shape [nframes, 9].
        Returns:

        """
        index = mapping.unsqueeze(-1).expand(-1, -1, 3)
        # index nframes x nall x 3
        # coord nframes x nloc x 3
        extended_coord = torch.gather(coord, dim=1, index=index)
        extended_coord = extended_coord - shift
        extended_coord.requires_grad_(True)
        atype_tebd = self.type_embedding(atype)

        nlist_tebd = []
        nnei_mask = []
        for nlist_type_item in nlist_type:
            nlist_type_item[nlist_type_item == -1] = self.ntypes
            nlist_tebd.append(self.type_embedding(nlist_type_item))
        for nlist_item in nlist:
            nnei_mask_item = nlist_item != -1
            nnei_mask.append(nnei_mask_item)
        

        descriptor, env_mat, diff, _ = self.descriptor(extended_coord, nlist, atype, nlist_type,
                                                    nlist_loc=nlist_loc, atype_tebd=atype_tebd, nlist_tebd=nlist_tebd)
        #env_mat，diff为列表
        #denoise net
        #in_proj_pair = SimpleLinear(env_mat[idx].shape[-1], self.attn_head, activate=None)
        #pair_rep = in_proj_pair(env_mat[idx])          
        updated_coord = self.coord_denoise_net(coord, env_mat, diff, nnei_mask)
        logits = self.type_predict_net(descriptor)
        #测试，非正式代码
        weight_updated_coord = 0 * updated_coord[0] + 1 * updated_coord[1]
        model_predict = {'updated_coord': weight_updated_coord,
                         'logits': logits
        }
        return model_predict

