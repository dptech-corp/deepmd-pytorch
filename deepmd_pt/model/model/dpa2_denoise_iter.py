import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List
from deepmd_pt.model.descriptor import DescrptSeAtten, DescrptGaussian
from deepmd_pt.model.task import DenoiseNet, TypePredictNet
from deepmd_pt.model.network import TypeEmbedNet, EnergyHead, Embedding, NodeTaskHead
from deepmd_pt.model.backbone import Evoformer3bBackBone
from deepmd_pt.utils.stat import compute_output_stats, make_stat_input
from deepmd_pt.utils import env
from deepmd_pt.model.model import BaseModel
from IPython import embed


class DenoiseModelDPA2Iter(BaseModel):

    def __init__(self, model_params, sampled=None):
        """Based on components, construct a DPA-1 model for energy.

        Args:
        - model_params: The Dict-like configuration with model options.
        - sampled: The sampled dataset for stat.
        """
        super(DenoiseModelDPA2Iter, self).__init__()
        # Descriptor + Type Embedding Net
        ntypes = len(model_params['type_map'])
        self.ntypes = ntypes
        descriptor_param = model_params.pop('descriptor')
        type_embedding_param = model_params.pop('type_embedding', None)
        if type_embedding_param is None:
            self.tebd_dim = 8
        else:
            tebd_dim = type_embedding_param['neuron'][-1]
            self.tebd_dim = tebd_dim
        self.descriptor_type = descriptor_param['type']
        assert self.descriptor_type in ['gaussian'], 'Only descriptor `gaussian` is supported for DPA-2-iter!'
        descriptor_param['ntypes'] = ntypes
        descriptor_param['num_pair'] = 2 * ntypes
        descriptor_param['embed_dim'] = self.tebd_dim
        self.descriptor = DescrptGaussian(**descriptor_param)
        self.atom_type_embedding = TypeEmbedNet(ntypes, self.tebd_dim)
        self.edge_type_embedding = TypeEmbedNet(ntypes * (ntypes + 1), self.descriptor.dim_emb)
        self.tag_encoder = nn.Embedding(3, self.tebd_dim)
        self.tag_encoder2 = nn.Embedding(2, self.tebd_dim)

        # BackBone
        backbone_param = model_params.pop('backbone')
        backbone_type = backbone_param.pop('type')
        backbone_param['atomic_dim'] = self.descriptor.dim_out
        backbone_param['pair_dim'] = self.descriptor.dim_emb
        backbone_param['nnei'] = self.descriptor.nnei
        self.pair_embed_dim = self.descriptor.dim_emb
        self.attention_heads = backbone_param['attn_head']
        self.num_block = backbone_param.get('num_block', 1)
        if backbone_type == 'evo-iter':
            self.backbone = Evoformer3bBackBone(**backbone_param)
        else:
            NotImplementedError(f"Unknown backbone type {backbone_type}!")

        # EnergyHead, MovementPredictionHead and ForceHead TODO
        self.engergy_proj = EnergyHead(self.tebd_dim, 1)
        self.energe_agg_factor = nn.Embedding(4, 1, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
        nn.init.normal_(self.energe_agg_factor.weight, 0, 0.01)

        self.node_proc = NodeTaskHead(self.tebd_dim, self.pair_embed_dim, self.attention_heads)
        self.node_proc.zero_init()

    def forward(self, coord, atype, natoms, mapping, shift, selected, selected_type,
                selected_loc: Optional[torch.Tensor] = None, box: Optional[torch.Tensor] = None, batch_data=None, **kwargs):
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
        print('here iter 82')
        embed()
        # nframes x nloc x 1
        tags = batch_data['tags']
        tags2 = batch_data['tags2']
        tags3 = batch_data['tags3']
        nframes, nloc = coord.shape[:-1]
        _, nall = extended_coord.shape[:-1]
        nnei = selected.shape[-1]
        # extended_coord.requires_grad_(True)
        # [nframes x nloc x tebd_dim]
        atom_feature = self.atom_type_embedding(atype)
        selected_type[selected_type == -1] = self.ntypes
        edge_type = atype.unsqueeze(-1) * (self.ntypes + 1) + selected_type
        # [nframes x nloc x nnei x pair_dim]
        edge_feature = self.edge_type_embedding(edge_type)
        # [nframes x nloc x nnei x 2]
        edge_type_2dim = torch.cat(
            [atype.view(nframes, nloc, 1, 1).expand(-1, -1, nnei, -1),
             selected_type.view(nframes, nloc, nnei, 1) + self.ntypes],
            dim=-1,
        )
        # atomic_feature: [nframes x nloc x tebd_dim]
        # pair_feature: [nframes x nloc x nnei x pair_dim]
        # diff: [nframes x nloc x nnei x 3]
        atomic_feature, pair_feature, diff = self.descriptor(extended_coord, selected, atom_feature, edge_type_2dim,
                                                             edge_feature)

        nnei_mask = selected != -1
        padding_selected_loc = selected_loc * nnei_mask
        # nframes x nloc x nnei
        attn_mask = torch.zeros_like(
            selected, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        ).masked_fill(~nnei_mask, float("-inf"))
        # nframes x head x nloc x nnei
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.attention_heads, 1, 1)
        # nframes x nloc x nnei x pair_dim
        attn_bias = pair_feature
        output, pair = self.backbone(atomic_feature, pair=attn_bias, nlist=padding_selected_loc, attn_mask=attn_mask,
                                     pair_mask=nnei_mask)

        # energy outut
        # [nframes, nloc]
        energy_out = self.engergy_proj(output).view(nframes, nloc)
        energy_factor = self.energe_agg_factor(torch.zeros_like(energy_out, dtype=torch.long)).view(nframes, nloc)
        energy_out = (energy_out * energy_factor).sum(dim=-1)

        # vector output
        predict_force = self.node_proc(output, pair, nlist=padding_selected_loc, delta_pos=diff, attn_mask=attn_mask)

        model_predict = {'energy': energy_out,
                         'force': predict_force,
                         'updated_coord': predict_force + coord,
                         }
        return model_predict
