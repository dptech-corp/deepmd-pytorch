import numpy as np
import torch

from deepmd_pt.utils import env
from deepmd_pt.model.descriptor import prod_env_mat_se_a, Descriptor, compute_std

try:
    from typing import Final
except:
    from torch.jit import Final

from deepmd_pt.model.network import TypeFilter, GaussianKernel, NonLinear
from IPython import embed


class DescrptGaussian(Descriptor):

    def __init__(self,
                 rcut,
                 kernel_num,
                 num_pair,
                 embed_dim,
                 pair_embed_dim,
                 sel,
                 ntypes: int,
                 **kwargs):
        """Construct an embedding net of type `gaussian`.

        """
        super(DescrptGaussian, self).__init__()
        self.gbf = GaussianKernel(K=kernel_num, num_pair=num_pair, stop=rcut)
        self.gbf_proj = NonLinear(kernel_num, pair_embed_dim)
        self.embed_dim = embed_dim
        self.pair_embed_dim = pair_embed_dim
        if kernel_num != self.embed_dim:
            self.edge_proj = torch.nn.Linear(kernel_num, self.embed_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
        else:
            self.edge_proj = None

        if isinstance(sel, int):
            sel = [sel]
        self.ntypes = ntypes
        self.sec = torch.tensor(sel)  # 每种元素在邻居中的位移
        self.nnei = sum(sel)  # 总的邻居数量

    @property
    def dim_out(self):
        """
        Returns the output dimension of this descriptor
        """
        return self.embed_dim

    @property
    def dim_emb(self):
        """
        Returns the output dimension of embedding
        """
        return self.pair_embed_dim

    def compute_input_stats(self, merged):
        pass

    def init_desc_stat(self, sumr, suma, sumn, sumr2, suma2):
        pass

    def forward_old_local(self, extended_coord, selected, atom_feature, edge_type_2dim, edge_feature):
        """Calculate decoded embedding for each atom.

        """
        nframes, nloc, nnei = selected.shape
        mask = selected >= 0
        selected = selected * mask
        coord_l = extended_coord[:, :nloc].view(nframes, -1, 1, 3)
        index = selected.view(nframes, -1).unsqueeze(-1).expand(-1, -1, 3)
        coord_r = torch.gather(extended_coord, 1, index)
        coord_r = coord_r.view(nframes, nloc, nnei, 3)
        diff = coord_r - coord_l
        diff = diff * mask.unsqueeze(-1)
        # [nframes, nloc, nnei]
        dist = torch.linalg.norm(diff, dim=-1)
        # [nframes, nloc, nnei, K]
        gbf_feature = self.gbf(dist, edge_type_2dim)
        edge_features = gbf_feature.masked_fill(
            ~mask.unsqueeze(-1).to(torch.bool),
            0.0,
        ) if mask is not None else gbf_feature
        nnei_num = mask.sum(-1, keepdim=True)
        # [nframes, nloc, K]
        mean_edge_features = edge_features.sum(-2) / (nnei_num + 1e-3)
        if self.edge_proj is not None:
            mean_edge_features = self.edge_proj(mean_edge_features)
        # [nframes, nloc, embed_dim]
        atom_feature = atom_feature + mean_edge_features

        # [nframes, nloc, pair_dim]
        gbf_result = self.gbf_proj(gbf_feature)
        gbf_result = gbf_result.masked_fill(
            ~mask.unsqueeze(-1).to(torch.bool),
            0.0,
        ) if mask is not None else gbf_feature

        attn_bias = gbf_result + edge_feature
        return atom_feature, attn_bias, diff

    def forward(self, coord, atom_feature, edge_type_2dim, edge_feature):
        ## global forward
        """Calculate decoded embedding for each atom.

        """
        nframes, nloc, _ = coord.shape
        # nframes x nloc x nloc x 3
        delta_pos = coord.unsqueeze(1) - coord.unsqueeze(2)
        # nframes x nloc x nloc
        dist = delta_pos.norm(dim=-1).view(-1, nloc, nloc)
        # [nframes, nloc, nloc, K]
        gbf_feature = self.gbf(dist, edge_type_2dim)
        edge_features = gbf_feature
        # [nframes, nloc, K]
        sum_edge_features = edge_features.sum(dim=-2)
        if self.edge_proj is not None:
            sum_edge_features = self.edge_proj(sum_edge_features)
        # [nframes, nloc, embed_dim]
        atom_feature = atom_feature + sum_edge_features

        # [nframes, nloc, nloc, pair_dim]
        gbf_result = self.gbf_proj(gbf_feature)

        attn_bias = gbf_result + edge_feature
        return atom_feature, attn_bias, delta_pos


def analyze_descrpt(matrix, ndescrpt, natoms, mixed_type=False, real_atype=None):
    pass
