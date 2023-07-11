import numpy as np
import torch

try:
    from typing import Final
except:
    from torch.jit import Final
from typing import (
    List, Optional, Tuple,
)
from deepmd_pt.utils import env
from deepmd_pt.utils.utils import get_activation_fn, ActivationFn
from deepmd_pt.model.descriptor import prod_env_mat_se_a, Descriptor, compute_std
from deepmd_pt.model.network import (
    TypeEmbedNet, SimpleLinear
)
from .se_atten import analyze_descrpt

mydtype = env.GLOBAL_PT_FLOAT_PRECISION
mydev = env.DEVICE


def torch_linear(*args, **kwargs):
    return torch.nn.Linear(*args, **kwargs, dtype=mydtype, device=mydev)


simple_linear = SimpleLinear
mylinear = simple_linear


class Atten2Map(torch.nn.Module):
    def __init__(
            self,
            ni,
            nd,
            nh,
    ):
        super(Atten2Map, self).__init__()
        self.ni = ni
        self.nd = nd
        self.nh = nh
        self.mapqk = mylinear(ni, nd * 2 * nh, bias=False)

    def forward(
            self,
            g2,  # nb x nloc x nnei x ng2
            h2,  # nb x nloc x nnei x 3
            nlist_mask,  # nb x nloc x nnei
    ):
        nb, nloc, nnei, _, = g2.shape
        nd, nh = self.nd, self.nh
        # nb x nloc x nnei x nd x (nh x 2)
        g2qk = self.mapqk(g2).view(nb, nloc, nnei, nd, nh * 2)
        # nb x nloc x (nh x 2) x nnei x nd
        g2qk = torch.permute(g2qk, (0, 1, 4, 2, 3))
        # nb x nloc x nh x nnei x nd
        g2q, g2k = torch.split(g2qk, nh, dim=2)
        # g2q = torch.nn.functional.normalize(g2q, dim=-1)
        # g2k = torch.nn.functional.normalize(g2k, dim=-1)
        # nb x nloc x nh x nnei x nnei
        attnw = torch.matmul(g2q, torch.transpose(g2k, -1, -2)) / nd ** 0.5
        # mask the attenmap, nb x nloc x 1 x 1 x nnei
        attnw_mask = ~nlist_mask.unsqueeze(2).unsqueeze(2)
        # mask the attenmap, nb x nloc x 1 x nnei x 1
        attnw_mask_c = ~nlist_mask.unsqueeze(2).unsqueeze(-1)
        attnw = attnw.masked_fill(attnw_mask, float("-inf"), )
        attnw = torch.softmax(attnw, dim=-1)
        attnw = attnw.masked_fill(attnw_mask, float(0.0), )
        attnw = attnw.masked_fill(attnw_mask_c, float(0.0), )
        # nb x nloc x nnei x nnei
        h2h2t = torch.matmul(h2, torch.transpose(h2, -1, -2)) / 3. ** 0.5
        # nb x nloc x nh x nnei x nnei
        ret = attnw * h2h2t[:, :, None, :, :]
        # ret = torch.softmax(g2qk, dim=-1)
        # nb x nloc x nnei x nnei x nh
        ret = torch.permute(ret, (0, 1, 3, 4, 2))
        return ret


class Atten2MultiHeadApply(torch.nn.Module):
    def __init__(
            self,
            ni,
            nh,
    ):
        super(Atten2MultiHeadApply, self).__init__()
        self.ni = ni
        self.nh = nh
        self.mapv = mylinear(ni, ni * nh, bias=False)
        self.head_map = mylinear(ni * nh, ni)

    def forward(
            self,
            AA,  # nf x nloc x nnei x nnei x nh
            g2,  # nf x nloc x nnei x ng2
    ):
        nf, nloc, nnei, ng2 = g2.shape
        nh = self.nh
        # nf x nloc x nnei x ng2 x nh
        g2v = self.mapv(g2).view(nf, nloc, nnei, ng2, nh)
        # nf x nloc x nh x nnei x ng2
        g2v = torch.permute(g2v, (0, 1, 4, 2, 3))
        # g2v = torch.nn.functional.normalize(g2v, dim=-1)
        # nf x nloc x nh x nnei x nnei
        AA = torch.permute(AA, (0, 1, 4, 2, 3))
        # nf x nloc x nh x nnei x ng2
        ret = torch.matmul(AA, g2v)
        # nf x nloc x nnei x ng2 x nh
        ret = torch.permute(ret, (0, 1, 3, 4, 2)).reshape(nf, nloc, nnei, (ng2 * nh))
        # nf x nloc x nnei x ng2
        return self.head_map(ret)


class Atten2EquiVarApply(torch.nn.Module):
    def __init__(
            self,
            ni,
            nh,
    ):
        super(Atten2EquiVarApply, self).__init__()
        self.ni = ni
        self.nh = nh
        self.head_map = mylinear(nh, 1)

    def forward(
            self,
            AA,  # nf x nloc x nnei x nnei x nh
            h2,  # nf x nloc x nnei x 3
    ):
        nf, nloc, nnei, _ = h2.shape
        nh = self.nh
        # nf x nloc x nh x nnei x nnei
        AA = torch.permute(AA, (0, 1, 4, 2, 3))
        h2m = torch.unsqueeze(h2, dim=2)
        # nf x nloc x nh x nnei x 3
        h2m = torch.tile(h2m, [1, 1, nh, 1, 1])
        # nf x nloc x nh x nnei x 3
        ret = torch.matmul(AA, h2m)
        # nf x nloc x nnei x 3 x nh
        ret = torch.permute(ret, (0, 1, 3, 4, 2)).view(nf, nloc, nnei, 3, nh)
        # nf x nloc x nnei x 3
        return torch.squeeze(self.head_map(ret), dim=-1)


class LocalAtten(torch.nn.Module):
    def __init__(
            self,
            ni,
            nd,
            nh,
    ):
        super(LocalAtten, self).__init__()
        self.ni = ni
        self.nd = nd
        self.nh = nh
        self.mapq = mylinear(ni, nd * 1 * nh, bias=False)
        self.mapkv = mylinear(ni, (nd + ni) * nh, bias=False)
        self.head_map = mylinear(ni * nh, ni)

    def forward(
            self,
            g1,  # nb x nloc x ng1
            gg1,  # nb x nloc x nnei x ng1
            nlist_mask,  # nb x nloc x nnei
    ):
        nb, nloc, nnei = nlist_mask.shape
        ni, nd, nh = self.ni, self.nd, self.nh
        assert ni == g1.shape[-1]
        assert ni == gg1.shape[-1]
        # nb x nloc x nd x nh
        g1q = self.mapq(g1).view(nb, nloc, nd, nh)
        # nb x nloc x nh x nd
        g1q = torch.permute(g1q, (0, 1, 3, 2))
        # nb x nloc x nnei x (nd+ni) x nh
        gg1kv = self.mapkv(gg1).view(nb, nloc, nnei, nd + ni, nh)
        gg1kv = torch.permute(gg1kv, (0, 1, 4, 2, 3))
        # nb x nloc x nh x nnei x nd, nb x nloc x nh x nnei x ng1
        gg1k, gg1v = torch.split(gg1kv, [nd, ni], dim=-1)

        # nb x nloc x nh x 1 x nnei
        attnw = torch.matmul(
            g1q.unsqueeze(-2),
            torch.transpose(gg1k, -1, -2)) / nd ** 0.5
        # nb x nloc x nh x nnei
        attnw = attnw.squeeze(-2)
        # mask the attenmap, nb x nloc x 1 x nnei
        attnw_mask = ~nlist_mask.unsqueeze(2)
        # nb x nloc x nh x nnei
        attnw = attnw.masked_fill(attnw_mask, float("-inf"), )
        attnw = torch.softmax(attnw, dim=-1)
        attnw = attnw.masked_fill(attnw_mask, float(0.0), )

        # nb x nloc x nh x ng1
        ret = torch.matmul(attnw.unsqueeze(-2), gg1v) \
            .squeeze(-2) \
            .view(nb, nloc, nh * ni)
        # nb x nloc x ng1
        ret = self.head_map(ret)
        return ret


def print_stat(aa, info=""):
    print(info, torch.mean(aa), torch.std(aa))


class DescrptSeUni(Descriptor):
    def __init__(
            self,
            rcut,
            rcut_smth,
            sel: int,
            ntypes: int,
            nlayers: int = 3,
            g1_dim=128,
            g2_dim=16,
            axis_dim: int = 4,
            gather_g1: bool = False,
            combine_grrg: bool = False,
            direct_dist: bool = False,
            do_bn_mode: str = 'no',
            bn_momentum: float = 0.1,
            update_g1_has_conv: bool = True,
            update_g1_has_drrd: bool = True,
            update_g1_has_grrg: bool = True,
            update_g1_has_attn: bool = True,
            update_g2_has_g1g1: bool = True,
            update_g2_has_attn: bool = True,
            update_h2: bool = False,
            attn1_hidden: int = 64,
            attn1_nhead: int = 4,
            attn2_hidden: int = 16,
            attn2_nhead: int = 4,
            attn_dotr: bool = True,
            activation: str = "tanh",
            update_style: str = "res_avg",
            set_davg_zero: bool = True,  # TODO
            **kwargs,
    ):
        super(DescrptSeUni, self).__init__()
        self.epsilon = 1e-4  # protection of 1./nnei
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.ntypes = ntypes
        self.nlayers = nlayers
        sel = [sel] if isinstance(sel, int) else sel
        self.nnei = sum(sel)  # 总的邻居数量
        assert len(sel) == 1
        self.sel = torch.tensor(sel)  # 每种元素在邻居中的位移
        self.sec = self.sel
        self.axis_dim = axis_dim
        self.set_davg_zero = set_davg_zero
        self.gather_g1 = gather_g1
        self.combine_grrg = combine_grrg
        self.update_last_g2 = self.combine_grrg
        self.direct_dist = direct_dist
        self.do_bn_mode = do_bn_mode
        self.bn_momentum = bn_momentum
        self.g1_hiddens = [g1_dim for ii in range(self.nlayers)]
        if not self.update_last_g2:
            self.g2_hiddens = [g2_dim for ii in range(self.nlayers - 1)]
        else:
            self.g2_hiddens = [g2_dim for ii in range(self.nlayers)]
        self.act = get_activation_fn(activation)
        self.update_h2 = update_h2
        self.update_g1_has_grrg = update_g1_has_grrg
        self.update_g1_has_drrd = update_g1_has_drrd
        self.update_g1_has_conv = update_g1_has_conv
        self.update_g1_has_attn = update_g1_has_attn
        self.update_g2_has_g1g1 = update_g2_has_g1g1
        self.update_g2_has_attn = update_g2_has_attn
        self.update_style = update_style

        def cal_1_dim(g1d, g2d, ax):
            ret = g1d
            if self.update_g1_has_grrg:
                ret += g2d * ax
            if self.update_g1_has_drrd:
                ret += g1d * ax
            if self.update_g1_has_conv:
                ret += g2d
            return ret

        g1_in_dims = [cal_1_dim(d1, d2, self.axis_dim) \
                      for d1, d2 in zip(self.g1_hiddens, [g2_dim] + self.g2_hiddens)]
        self.type_embd = TypeEmbedNet(self.ntypes, self.g1_hiddens[0])
        self.g2_embd = mylinear(1, self.g2_hiddens[0])
        self.linear1 = self._linear_layers(g1_in_dims, self.g1_hiddens)
        self.linear2 = self._linear_layers(self.g2_hiddens, self.g2_hiddens)
        if update_g1_has_conv:
            tmp_g2_h = self.g2_hiddens if self.update_last_g2 else [g2_dim] + self.g2_hiddens
            self.proj_g1g2 = self._linear_layers(self.g1_hiddens, tmp_g2_h, bias=False)
        if update_g2_has_g1g1:
            tmp_g1_h = self.g1_hiddens if self.update_last_g2 else self.g1_hiddens[1:]
            self.proj_g1g1g2 = self._linear_layers(tmp_g1_h, self.g2_hiddens, bias=False)
        if update_g2_has_attn:
            self.attn2g_map = torch.nn.ModuleList(
                [Atten2Map(ii, attn2_hidden, attn2_nhead) for ii in self.g2_hiddens])
            self.attn2_mh_apply = torch.nn.ModuleList(
                [Atten2MultiHeadApply(ii, attn2_nhead) for ii in self.g2_hiddens])
            self.attn2_lm = torch.nn.ModuleList(
                [torch.nn.LayerNorm(ii, elementwise_affine=True, device=mydev, dtype=mydtype)
                 for ii in self.g2_hiddens])
        if update_h2:
            self.attn2h_map = torch.nn.ModuleList(
                [Atten2Map(ii, attn2_hidden, attn2_nhead) for ii in self.g2_hiddens])
            self.attn2_ev_apply = torch.nn.ModuleList(
                [Atten2EquiVarApply(ii, attn2_nhead) for ii in self.g2_hiddens])
        if update_g1_has_attn:
            self.loc_attn = torch.nn.ModuleList(
                [LocalAtten(ii, attn1_hidden, attn1_nhead) for ii in self.g1_hiddens])
        if self.gather_g1:
            self.all_g1_proj = mylinear((self.nlayers + 1) * g1_dim, g1_dim)

        if self.do_bn_mode == 'uniform':
            self.bn1 = self._bn_layers(self.nlayers)
            self.bn2 = self._bn_layers(self.nlayers)
        elif self.do_bn_mode == 'component':
            self.bn1 = self._bn_layers(self.nlayers, nf=g1_dim)
            self.bn2 = self._bn_layers(self.nlayers, nf=g2_dim)
        elif self.do_bn_mode == 'no':
            self.bn1, self.bn2 = None, None
        else:
            raise RuntimeError(f"unknown bn_mode {self.do_bn_mode}")

        sshape = (self.ntypes, self.nnei, 4)
        mean = torch.zeros(sshape, dtype=mydtype, device=mydev)
        stddev = torch.ones(sshape, dtype=mydtype, device=mydev)
        self.register_buffer('mean', mean)
        self.register_buffer('stddev', stddev)

    @property
    def dim_out(self):
        """
    Returns the output dimension of this descriptor
    """
        add_g2_dim = self.g2_hiddens[-1] * self.axis_dim if self.combine_grrg else 0
        return self.g1_hiddens[-1] + add_g2_dim

    @property
    def dim_in(self):
        """
    Returns the atomic input dimension of this descriptor
    """
        return self.g1_hiddens[0]

    @property
    def dim_emb(self):
        """
    Returns the embedding dimension g2
    """
        return self.g2_hiddens[-1]

    def forward(self, extended_coord, nlist, atype, nlist_type, nlist_loc=None, atype_tebd=None, nlist_tebd=None,
                seq_input=None):

        """
    extended_coord:     [nb, nloc x 3]
    atype:              [nb, nloc]
    """
        nframes, nloc = nlist_loc.shape[:2]
        # nb x nloc x nnei x 4
        dmatrix, diff = prod_env_mat_se_a(
            extended_coord, nlist, atype,
            self.mean, self.stddev,
            self.rcut, self.rcut_smth)
        nlist_type[nlist_type == -1] = self.ntypes
        nlist_mask = (nlist != -1)
        masked_nlist_loc = nlist_loc * nlist_mask

        # nb x nloc x ng1
        atype_tebd = self.type_embd(atype)
        # [nframes, nloc, tebd_dim]
        if seq_input is not None:
            if seq_input.shape[0] == nframes * nloc:
                seq_input = seq_input[:, 0, :].reshape(nframes, nloc, -1)
            atype_tebd += seq_input

        g1 = self.act(atype_tebd)
        # nb x nloc x nnei x 1,  nb x nloc x nnei x 3
        if not self.direct_dist:
            g2, h2 = torch.split(dmatrix, [1, 3], dim=-1)
        else:
            g2, h2 = torch.linalg.norm(diff, dim=-1, keepdim=True), diff
            g2 = g2 / self.rcut
            h2 = h2 / self.rcut
        # nb x nloc x nnei x ng2
        g2 = self.act(self.g2_embd(g2))

        if self.gather_g1:
            all_g1 = [g1]
        for ll in range(self.nlayers):
            if self.update_last_g2:
                update_chnnl_2 = True
            else:
                update_chnnl_2 = (ll != self.nlayers - 1)
            g1, g2, h2 = self._one_layer(
                ll, g1, g2, h2,
                masked_nlist_loc,
                nlist_mask,
                update_chnnl_2=update_chnnl_2,
            )
            if self.gather_g1:
                all_g1.append(g1)
        if self.gather_g1:
            g1 = torch.cat(all_g1, dim=-1)
            g1 = self.all_g1_proj(g1)

        # nb x nloc x 3 x ng2
        h2g2 = self._cal_h2g2(g2, h2, nlist_mask)
        # (nb x nloc) x ng2 x 3
        rot_mat = torch.permute(h2g2, (0, 1, 3, 2))

        if self.combine_grrg:
            grrg = self._cal_grrg(h2g2)
            g1 = torch.cat([g1, grrg], dim=-1)

        return g1, None, diff, rot_mat.view(-1, self.dim_emb, 3)

    def _linear_layers(
            self,
            in_dims,
            out_dims,
            bias: bool = True,
    ):
        ret = []
        for ii, oo in zip(in_dims, out_dims):
            ret.append(
                mylinear(ii, oo, bias=bias))
        return torch.nn.ModuleList(ret)

    def _bn_layers(
            self,
            nlayers,
            nf: int = 1,
    ):
        ret = []
        for ii in range(nlayers):
            ret.append(torch.nn.BatchNorm1d(
                nf, eps=1e-5, momentum=self.bn_momentum, affine=False,
                track_running_stats=True, device=mydev, dtype=mydtype))
        return ret

    def _update_h2(self, ll, g2, h2, nlist_mask):
        nb, nloc, nnei, _ = g2.shape
        # # nb x nloc x nnei x nh2
        # h2_1 = self.attn2_ev_apply[ll](AA, h2)
        # h2_update.append(h2_1)
        # nb x nloc x nnei x nnei x nh
        AAh = self.attn2h_map[ll](g2, h2, nlist_mask)
        # nb x nloc x nnei x nh2
        h2_1 = self.attn2_ev_apply[ll](AAh, h2)
        return h2_1

    def _make_nei_g1(self, g1, nlist):
        nb, nloc, nnei = nlist.shape
        ng1 = g1.shape[-1]
        # nlist: nb x nloc x nnei
        # g1   : nb x nloc x ng1
        # index: nb x (nloc x nnei) x ng1
        index = nlist.view(nb, nloc * nnei).unsqueeze(-1).expand(-1, -1, ng1)
        # gg1  : nb x (nloc x nnei) x ng1
        gg1 = torch.gather(g1, dim=1, index=index)
        # gg1  : nb x nloc x nnei x ng1
        gg1 = gg1.view(nb, nloc, nnei, ng1)
        return gg1

    def _apply_nlist_mask(self, gg, nlist_mask):
        # gg:  nf x nloc x nnei x ng
        # msk: nf x nloc x nnei
        return gg.masked_fill(~nlist_mask.unsqueeze(-1), float(0.))

    def _update_g1_conv(self, ll, gg1, g2, nlist, nlist_mask):
        nb, nloc, nnei, _ = g2.shape
        ng1 = gg1.shape[-1]
        ng2 = g2.shape[-1]
        # gg1  : nb x nloc x nnei x ng2
        gg1 = self.proj_g1g2[ll](gg1).view(nb, nloc, nnei, ng2)
        # nb x nloc x nnei x ng2
        gg1 = self._apply_nlist_mask(gg1, nlist_mask)
        # nb x nloc
        invnnei = 1. / (self.epsilon + torch.sum(nlist_mask, dim=-1))
        # nb x nloc x ng2
        g1_11 = torch.sum(g2 * gg1, dim=2) * invnnei.unsqueeze(-1)
        return g1_11

    def _cal_h2g2(self, g2, h2, nlist_mask):
        # g2:  nf x nloc x nnei x ng2
        # h2:  nf x nloc x nnei x 3
        # msk: nf x nloc x nnei
        nb, nloc, nnei, _ = g2.shape
        ng2 = g2.shape[-1]
        # nb x nloc x nnei x ng2
        g2 = self._apply_nlist_mask(g2, nlist_mask)
        # nb x nloc
        invnnei = 1. / (self.epsilon + torch.sum(nlist_mask, dim=-1))
        # nb x nloc x 1 x 1
        invnnei = invnnei.unsqueeze(-1).unsqueeze(-1)
        # nb x nloc x 3 x ng2
        h2g2 = torch.matmul(
            torch.transpose(h2, -1, -2), g2) * invnnei
        return h2g2

    def _cal_grrg(self, h2g2):
        # nb x nloc x 3 x ng2
        nb, nloc, _, ng2 = h2g2.shape
        # nb x nloc x 3 x axis
        h2g2m = torch.split(h2g2, self.axis_dim, dim=-1)[0]
        # nb x nloc x axis x ng2
        g1_13 = torch.matmul(
            torch.transpose(h2g2m, -1, -2), h2g2) / (float(3.) ** 1)
        # nb x nloc x (axisxng2)
        g1_13 = g1_13.view(nb, nloc, self.axis_dim * ng2)
        return g1_13

    def _update_g1_grrg(self, ll, g2, h2, nlist_mask):
        # g2:  nf x nloc x nnei x ng2
        # h2:  nf x nloc x nnei x 3
        # msk: nf x nloc x nnei
        nb, nloc, nnei, _ = g2.shape
        ng2 = g2.shape[-1]
        # nb x nloc x 3 x ng2
        h2g2 = self._cal_h2g2(g2, h2, nlist_mask)
        # nb x nloc x (axisxng2)
        g1_13 = self._cal_grrg(h2g2)
        return g1_13

    def _update_g2_g1g1(
            self,
            g1,  # nb x nloc x ng1
            gg1,  # nb x nloc x nnei x ng1
            nlist_mask,  # nb x nloc x nnei
    ):
        ret = g1.unsqueeze(-2) * gg1
        # nb x nloc x nnei x ng1
        ret = self._apply_nlist_mask(ret, nlist_mask)
        return ret

    def _apply_bn_uni(
            self,
            bn,
            gg,
            mode='1',
    ):
        def _apply_nb_1(bn, gg):
            nb, nl, nf = gg.shape
            gg = gg.view([nb, 1, nl * nf])
            gg = bn(gg)
            return gg.view([nb, nl, nf])

        def _apply_nb_2(bn, gg):
            nb, nl, nnei, nf = gg.shape
            gg = gg.view([nb, 1, nl * nnei * nf])
            gg = bn(gg)
            return gg.view([nb, nl, nnei, nf])

        if len(gg.shape) == 3:
            return _apply_nb_1(bn, gg)
        elif len(gg.shape) == 4:
            return _apply_nb_2(bn, gg)
        else:
            raise RuntimeError(f'unsupported input shape {gg.shape}')

    def _apply_bn_comp(
            self,
            bn,
            gg,
    ):
        ss = gg.shape
        nf = ss[-1]
        gg = gg.view([-1, nf])
        gg = bn(gg).view(ss)
        return gg

    def _apply_bn(
            self,
            bn,
            gg,
    ):
        if self.do_bn_mode == 'uniform':
            return self._apply_bn_uni(bn, gg)
        elif self.do_bn_mode == 'component':
            return self._apply_bn_comp(bn, gg)
        else:
            return gg

    def _one_layer(
            self,
            ll,  #
            g1,  # nf x nloc x ng1
            g2,  # nf x nloc x nnei x ng2
            h2,  # nf x nloc x nnei x 3
            nlist,  # nf x nloc x nnei
            nlist_mask,
            update_chnnl_2: bool = True,
    ):
        update_g2_has_attn = self.update_g2_has_attn
        update_g1_has_conv = self.update_g1_has_conv
        update_g1_has_drrd = self.update_g1_has_drrd
        update_g1_has_grrg = self.update_g1_has_grrg
        update_g1_has_attn = self.update_g1_has_attn
        update_g2_has_g1g1 = self.update_g2_has_g1g1
        update_h2 = self.update_h2
        cal_gg1 = update_g1_has_drrd or update_g1_has_conv or update_g1_has_attn or update_g2_has_g1g1

        nb, nloc, nnei, _ = g2.shape
        assert (nb, nloc) == g1.shape[:2]
        assert (nb, nloc, nnei) == h2.shape[:3]
        ng1 = g1.shape[-1]
        ng2 = g2.shape[-1]
        nh2 = h2.shape[-1]

        # g1 = self.lmg1[ll](g1)
        # g2 = self.lmg2[ll](g2)
        if self.bn1 is not None:
            g1 = self._apply_bn(self.bn1[ll], g1)
        if self.bn2 is not None:
            g2 = self._apply_bn(self.bn2[ll], g2)

        g2_update = [g2]
        h2_update = [h2]
        g1_update = [g1]
        g1_mlp = [g1]

        if cal_gg1:
            gg1 = self._make_nei_g1(g1, nlist)

        if update_chnnl_2:
            # nb x nloc x nnei x ng2
            g2_1 = self.act(self.linear2[ll](g2))
            g2_update.append(g2_1)

            if update_g2_has_g1g1:
                g2_update.append(self.proj_g1g1g2[ll](
                    self._update_g2_g1g1(g1, gg1, nlist_mask)))

            if update_g2_has_attn:
                # nb x nloc x nnei x nnei x nh
                AAg = self.attn2g_map[ll](g2, h2, nlist_mask)
                # nb x nloc x nnei x ng2
                g2_2 = self.attn2_mh_apply[ll](AAg, g2)
                g2_2 = self.attn2_lm[ll](g2_2)
                g2_update.append(g2_2)

            if update_h2:
                h2_update.append(self._update_h2(ll, g2, h2, nlist_mask))

        if update_g1_has_conv:
            g1_mlp.append(self._update_g1_conv(ll, gg1, g2, nlist, nlist_mask))

        if update_g1_has_grrg:
            g1_mlp.append(self._update_g1_grrg(ll, g2, h2, nlist_mask))

        if update_g1_has_drrd:
            g1_mlp.append(self._update_g1_grrg(ll, gg1, h2, nlist_mask))

        # nb x nloc x [ng1+ng2+(axisxng2)+(axisxng1)]
        #                  conv   grrg      drrd
        g1_1 = self.act(self.linear1[ll](
            torch.cat(g1_mlp, dim=-1)
        ))
        g1_update.append(g1_1)

        if update_g1_has_attn:
            g1_update.append(self.loc_attn[ll](g1, gg1, nlist_mask))

        def list_update_res_avg(update_list):
            nitem = len(update_list)
            uu = update_list[0]
            for ii in range(1, nitem):
                uu = uu + update_list[ii]
            return uu / (float(nitem) ** 0.5)

        def list_update_res_incr(update_list):
            nitem = len(update_list)
            uu = update_list[0]
            scale = 1. / (float(nitem - 1) ** 0.5) if nitem > 1 else 0.
            for ii in range(1, nitem):
                uu = uu + scale * update_list[ii]
            return uu

        def list_update(update_list):
            if self.update_style == "res_avg":
                return list_update_res_avg(update_list)
            elif self.update_style == "res_incr":
                return list_update_res_incr(update_list)
            else:
                raise RuntimeError(f"unknown update style {self.update_style}")

        # update
        if update_chnnl_2:
            g2_new = list_update(g2_update)
            h2_new = list_update(h2_update)
        else:
            g2_new, h2_new = g2, h2
        g1_new = list_update(g1_update)
        return g1_new, g2_new, h2_new

    def compute_input_stats(self, merged):
        """Update mean and stddev for descriptor elements.
      """
        ndescrpt = self.nnei * 4  # 描述符的元素数量
        sumr = []
        suma = []
        sumn = []
        sumr2 = []
        suma2 = []
        mixed_type = 'real_natoms_vec' in merged[0]
        for system in merged:  # 逐个 system 的分析
            index = system['mapping'].unsqueeze(-1).expand(-1, -1, 3)
            extended_coord = torch.gather(system['coord'], dim=1, index=index)
            extended_coord = extended_coord - system['shift']
            env_mat, _ = prod_env_mat_se_a(
                extended_coord, system['selected'], system['atype'],
                self.mean, self.stddev,
                self.rcut, self.rcut_smth,
            )
            if not mixed_type:
                sysr, sysr2, sysa, sysa2, sysn = analyze_descrpt(
                    env_mat.detach().cpu().numpy(), ndescrpt,
                    system['natoms'])
            else:
                sysr, sysr2, sysa, sysa2, sysn = analyze_descrpt(
                    env_mat.detach().cpu().numpy(), ndescrpt,
                    system['real_natoms_vec'], mixed_type=mixed_type,
                    real_atype=system['atype'].detach().cpu().numpy())
            sumr.append(sysr)
            suma.append(sysa)
            sumn.append(sysn)
            sumr2.append(sysr2)
            suma2.append(sysa2)
        sumr = np.sum(sumr, axis=0)
        suma = np.sum(suma, axis=0)
        sumn = np.sum(sumn, axis=0)
        sumr2 = np.sum(sumr2, axis=0)
        suma2 = np.sum(suma2, axis=0)
        return sumr, suma, sumn, sumr2, suma2

    def init_desc_stat(self, sumr, suma, sumn, sumr2, suma2):
        all_davg = []
        all_dstd = []
        for type_i in range(self.ntypes):
            davgunit = [[sumr[type_i] / (sumn[type_i] + 1e-15), 0, 0, 0]]
            dstdunit = [[
                compute_std(sumr2[type_i], sumr[type_i], sumn[type_i], self.rcut),
                compute_std(suma2[type_i], suma[type_i], sumn[type_i], self.rcut),
                compute_std(suma2[type_i], suma[type_i], sumn[type_i], self.rcut),
                compute_std(suma2[type_i], suma[type_i], sumn[type_i], self.rcut)
            ]]
            davg = np.tile(davgunit, [self.nnei, 1])
            dstd = np.tile(dstdunit, [self.nnei, 1])
            all_davg.append(davg)
            all_dstd.append(dstd)
        self.sumr = sumr
        self.suma = suma
        self.sumn = sumn
        self.sumr2 = sumr2
        self.suma2 = suma2
        if not self.set_davg_zero:
            mean = np.stack(all_davg)
            self.mean.copy_(torch.tensor(mean, device=env.DEVICE))
        stddev = np.stack(all_dstd)
        self.stddev.copy_(torch.tensor(stddev, device=env.DEVICE))
