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

    def __init__(self, model_params, sampled=None, set_zero_energy_bias=False):
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
        self.do_tag_embedding = descriptor_param.pop('do_tag_embedding', True)
        self.tag_ener_pref = descriptor_param.pop('tag_ener_pref', True)
        self.descriptor = DescrptGaussian(**descriptor_param)
        self.atom_type_embedding = nn.Embedding(ntypes + 1, self.tebd_dim, padding_idx=ntypes, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
        # self.atom_type_embedding = TypeEmbedNet(ntypes, self.tebd_dim)
        if self.do_tag_embedding:
            self.tag_encoder = nn.Embedding(3, self.tebd_dim)
            self.tag_encoder2 = nn.Embedding(2, self.tebd_dim)
            self.tag_type_embedding = TypeEmbedNet(10, self.descriptor.dim_emb)
        else:
            print('not do tag embedding!!')
        self.edge_type_embedding = nn.Embedding((ntypes + 1) * (ntypes + 1), self.descriptor.dim_emb, padding_idx=(ntypes + 1) * (ntypes + 1) - 1, dtype=env.GLOBAL_PT_FLOAT_PRECISION)

        # BackBone
        backbone_param = model_params.pop('backbone')
        backbone_type = backbone_param.pop('type')
        backbone_param['atomic_dim'] = self.descriptor.dim_out
        backbone_param['pair_dim'] = self.descriptor.dim_emb
        if self.descriptor.nnei2 is None:
            backbone_param['nnei'] = self.descriptor.nnei
        else:
            backbone_param['nnei'] = self.descriptor.nnei2
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

        energy_bias_dict = {}
        # Statistics
        self.compute_or_load_stat(model_params, energy_bias_dict, ntypes, sampled=sampled,
                                  set_zero_energy_bias=set_zero_energy_bias)
        bias_atom_e = torch.tensor(energy_bias_dict['bias_atom_e'])
        self.register_buffer('bias_atom_e', bias_atom_e)

    def forward_global(self, coord, atype, natoms, mapping, shift, selected, selected_type,
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
        # index = mapping.unsqueeze(-1).expand(-1, -1, 3)
        # # index nframes x nall x 3
        # # coord nframes x nloc x 3
        # extended_coord = torch.gather(coord, dim=1, index=index)
        # extended_coord = extended_coord - shift
        # _, nall = extended_coord.shape[:-1]
        # nframes x nloc x 1

        nframes, nloc = coord.shape[:-1]
        # nnei = selected.shape[-1]
        if self.do_tag_embedding or self.tag_ener_pref:
            tags = batch_data['tags'].type(torch.int).squeeze(-1)
            tags2 = batch_data['tags2'].type(torch.int).squeeze(-1)
        if batch_data['find_tags3']:
            tags3 = batch_data['tags3'].type(torch.int).squeeze(-1)
        else:
            tags3 = torch.ones_like(atype)
        # extended_coord.requires_grad_(True)
        # [nframes x nloc x tebd_dim]
        atom_feature = self.atom_type_embedding(atype)
        # [nframes x nloc x tebd_dim]
        if self.do_tag_embedding:
            tags2_emb = self.tag_encoder(tags2)
            tags3_emb = self.tag_encoder2(tags3)
            atom_feature = atom_feature + tags2_emb + tags3_emb


        # global TODO
        # [nframes x nloc x nloc x 2]
        edge_type_2dim = torch.cat(
            [atype.view(nframes, nloc, 1, 1).expand(-1, -1, nloc, -1),
             atype.view(nframes, 1, nloc, 1).expand(-1, nloc, -1, -1) + self.ntypes],
            dim=-1,
        )
        # [nframes x nloc x nloc]
        edge_type = atype.unsqueeze(-1) * (self.ntypes + 1) + atype.unsqueeze(-2)
        # [nframes x nloc x nloc x pair_dim]
        edge_feature = self.edge_type_embedding(edge_type)
        if self.do_tag_embedding:
            tag_pair = tags.unsqueeze(-1) * 3 + tags.unsqueeze(-2)
            tag_embedding = self.tag_type_embedding(tag_pair)
            edge_feature = edge_feature + tag_embedding

        atomic_feature, pair_feature, delta_pos = self.descriptor(coord, atom_feature, edge_type_2dim, edge_feature)
        # nframes x nloc x nnei x pair_dim
        attn_bias = pair_feature
        output, pair = self.backbone(atomic_feature, pair=attn_bias)

        # # local TODO
        # selected_type[selected_type == -1] = self.ntypes
        # edge_type = atype.unsqueeze(-1) * (self.ntypes + 1) + selected_type
        # # [nframes x nloc x nnei x pair_dim]
        # edge_feature = self.edge_type_embedding(edge_type)
        # # [nframes x nloc x nnei x 2]
        # edge_type_2dim = torch.cat(
        #     [atype.view(nframes, nloc, 1, 1).expand(-1, -1, nnei, -1),
        #      selected_type.view(nframes, nloc, nnei, 1) + self.ntypes],
        #     dim=-1,
        # )
        # # atomic_feature: [nframes x nloc x tebd_dim]
        # # pair_feature: [nframes x nloc x nnei x pair_dim]
        # # diff: [nframes x nloc x nnei x 3]
        # atomic_feature, pair_feature, diff = self.descriptor(extended_coord, selected, atom_feature, edge_type_2dim,
        #                                                      edge_feature)
        #
        # nnei_mask = selected != -1
        # padding_selected_loc = selected_loc * nnei_mask
        # # nframes x nloc x nnei
        # attn_mask = torch.zeros_like(
        #     selected, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        # ).masked_fill(~nnei_mask, float("-inf"))
        # # nframes x head x nloc x nnei
        # attn_mask = attn_mask.unsqueeze(1).repeat(1, self.attention_heads, 1, 1)
        # output, pair = self.backbone(atomic_feature, pair=attn_bias, nlist=padding_selected_loc, attn_mask=attn_mask,
        #                              pair_mask=nnei_mask)

        # energy outut
        # [nframes, nloc]
        energy_out = self.engergy_proj(output).view(nframes, nloc)

        # nframes x nloc
        if self.tag_ener_pref:
            energy_factor = self.energe_agg_factor(tags).view(nframes, nloc)
        else:
            energy_factor = self.energe_agg_factor(torch.zeros_like(tags3)).view(nframes, nloc)
        energy_out = (energy_out * energy_factor)
        energy_out *= (tags3 > 0).type_as(energy_out)
        energy_out = energy_out.sum(dim=-1)

        # vector output
        predict_force = self.node_proc(output, pair, delta_pos=delta_pos)
        # [nframes, nloc, 1]
        force_target_mask = (tags3 > 0).type_as(predict_force).unsqueeze(-1)

        model_predict = {'energy': energy_out,
                         'force': predict_force,
                         'force_target_mask': force_target_mask,
                         'updated_coord': predict_force + coord,
                         }
        return model_predict

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
        _, nall = extended_coord.shape[:-1]
        # nframes x nloc x 1
        nframes, nloc = coord.shape[:-1]
        selected_type[selected_type == -1] = self.ntypes

        # Neighborhood 1
        nnei1 = selected.shape[-1]
        # nframes x nloc x nnei1
        nnei1_mask = selected != -1

        # Neighborhood 2
        nnei2 = self.descriptor.nnei2
        # nframes x nloc x (1 + nnei2)
        selected2 = torch.cat([torch.arange(0, nloc, device=selected.device).reshape(1, nloc, 1).expand(nframes, -1, -1), selected[:, :, :nnei2]], dim=-1)
        selected_loc2 = torch.cat([torch.arange(0, nloc, device=selected_loc.device).reshape(1, nloc, 1).expand(nframes, -1, -1), selected_loc[:, :, :nnei2]], dim=-1)
        selected_type2 = torch.cat([atype.reshape(nframes, nloc, 1), selected_type[:, :, :nnei2]], dim=-1)
        nnei2_mask = selected2 != -1
        padding_mask = selected2 == -1
        selected2 = selected2 * nnei2_mask
        selected_loc2 = selected_loc2 * nnei2_mask
        # nframes x nloc x (1 + nnei2) x (1 + nnei2)
        pair_mask = nnei2_mask.unsqueeze(-1) * nnei2_mask.unsqueeze(-2)
        # nframes x nloc x (1 + nnei2) x (1 + nnei2) x head
        attn_mask = torch.zeros([nframes, nloc, 1 + nnei2, 1 + nnei2, self.attention_heads], device=selected.device, dtype=coord.dtype)
        attn_mask.masked_fill_(
            padding_mask.unsqueeze(2).unsqueeze(-1),
            float("-inf")
        )
        # (nframes x nloc) x head x (1 + nnei2) x (1 + nnei2)
        attn_mask = attn_mask.reshape(nframes * nloc, 1 + nnei2, 1 + nnei2, self.attention_heads).permute(0, 3, 1, 2).contiguous()

        # Atomic feature
        # [(nframes x nloc) x (1 + nnei2) x tebd_dim]
        atom_feature = self.atom_type_embedding(selected_type2).reshape(nframes * nloc, 1 + nnei2, self.tebd_dim)
        # Optional: GRRG or mean of gbf TODO

        atom_feature = atom_feature * nnei2_mask.reshape(nframes * nloc, 1 + nnei2, 1)

        # Pair feature
        # [(nframes x nloc) x (1 + nnei2)]
        selected_type2_reshape = selected_type2.reshape(nframes * nloc, 1 + nnei2)
        # [(nframes x nloc) x (1 + nnei2) x (1 + nnei2)]
        edge_type = selected_type2_reshape.unsqueeze(-1) * (self.ntypes + 1) + selected_type2_reshape.unsqueeze(-2)
        # [(nframes x nloc) x (1 + nnei2) x (1 + nnei2) x pair_dim]
        edge_feature = self.edge_type_embedding(edge_type)

        # [(nframes x nloc) x (1 + nnei2) x (1 + nnei2) x 2]
        edge_type_2dim = torch.cat(
            [selected_type2_reshape.view(nframes * nloc, 1 + nnei2, 1, 1).expand(-1, -1, 1 + nnei2, -1),
             selected_type2_reshape.view(nframes * nloc, 1, 1 + nnei2, 1).expand(-1, 1 + nnei2, -1, -1) + self.ntypes],
            dim=-1,
        )
        # [(nframes x nloc) x (1 + nnei2) x 3]
        coord_selected = torch.gather(extended_coord.unsqueeze(1).expand(-1, nloc, -1, -1).reshape(nframes * nloc, nall, 3),
                                      dim=1, index=selected2.reshape(nframes * nloc, 1 + nnei2, 1).expand(-1, -1, 3))


        # Update pair features (or and atomic features) with gbf features
        # delta_pos: [(nframes x nloc) x (1 + nnei2) x (1 + nnei2) x 3].
        atomic_feature, pair_feature, delta_pos = self.descriptor(coord_selected, atom_feature, edge_type_2dim, edge_feature)
        # [(nframes x nloc) x (1 + nnei2) x (1 + nnei2) x pair_dim]
        attn_bias = pair_feature

        # output: [(nframes x nloc) x (1 + nnei2) x tebd_dim]
        # pair: [(nframes x nloc) x (1 + nnei2) x (1 + nnei2) x pair_dim]
        output, pair = self.backbone(atomic_feature, pair=attn_bias, attn_mask=attn_mask, pair_mask=pair_mask, atom_mask=nnei2_mask.reshape(nframes * nloc, 1 + nnei2))

        # [nframes x nloc x tebd_dim]
        output_nloc = (output[:, 0, :]).reshape(nframes, nloc, self.tebd_dim)
        # Optional: GRRG or mean of gbf TODO

        # energy outut
        # [nframes, nloc]
        energy_out = self.engergy_proj(output_nloc).view(nframes, nloc)
        # [nframes, nloc]
        energy_factor = self.energe_agg_factor(torch.zeros_like(atype)).view(nframes, nloc)
        energy_out = (energy_out * energy_factor) + self.bias_atom_e[atype]
        energy_out = energy_out.sum(dim=-1)

        # vector output
        # predict_force: [(nframes x nloc) x (1 + nnei2) x 3]
        predict_force = self.node_proc(output, pair, delta_pos=delta_pos)
        # predict_force_nloc: [nframes x nloc x 3]
        predict_force_nloc = (predict_force[:, 0, :]).reshape(nframes, nloc, 3)
        force_target_mask = torch.ones_like(atype).type_as(predict_force).unsqueeze(-1)

        model_predict = {'energy': energy_out,
                         'force': predict_force_nloc,
                         # 'force_target_mask': force_target_mask,
                         'updated_coord': predict_force_nloc + coord,
                         }
        return model_predict
