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
simple_linear  = SimpleLinear
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
      g2,       # nh x nloc x nnei x ng2
      h2,       # nh x nloc x nnei x 3
  ):
    nb, nloc, nnei, _, = g2.shape
    nd, nh = self.nd, self.nh
    # nb x nloc x nnei x nd x (nh x 2)
    g2qk = self.mapqk(g2).view(nb, nloc, nnei, nd, nh*2)
    # nb x nloc x (nh x 2) x nnei x nd
    g2qk = torch.permute(g2qk, (0, 1, 4, 2, 3))
    # nb x nloc x nh x nnei x nd
    g2q, g2k = torch.split(g2qk, nh, dim=2)
    # nb x nloc x nh x nnei x nnei
    g2qk = torch.matmul(
      g2q, torch.transpose(g2k, -1, -2)) / nd**0.5
    # nb x nloc x nnei x nnei
    h2h2t = torch.matmul(
      h2, torch.transpose(h2, -1, -2)) / 3.**0.5
    # nb x nloc x nh x nnei x nnei
    ret = torch.softmax(g2qk, dim=-1) * h2h2t[:,:,None,:,:]
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
      AA,       # nf x nloc x nnei x nnei x nh
      g2,       # nf x nloc x nnei x ng2
  ):
    nf, nloc, nnei, ng2 = g2.shape
    nh = self.nh
    # nf x nloc x nnei x ng2 x nh
    g2v = self.mapv(g2).view(nf, nloc, nnei, ng2, nh)    
    # nf x nloc x nh x nnei x ng2
    g2v = torch.permute(g2v, (0, 1, 4, 2, 3))    
    # nf x nloc x nh x nnei x nnei
    AA = torch.permute(AA, (0, 1, 4, 2, 3))
    # nf x nloc x nh x nnei x ng2
    ret = torch.matmul(AA, g2v)
    # nf x nloc x nnei x ng2 x nh
    ret = torch.permute(ret, (0, 1, 3, 4, 2)).reshape(nf, nloc, nnei, (ng2*nh))
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
      AA,       # nf x nloc x nnei x nnei x nh
      h2,       # nf x nloc x nnei x 3
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


def print_stat(aa, info=""):
  print(info, torch.mean(aa), torch.std(aa))

class DescrptSeUni(Descriptor):
  def __init__(
      self,
      rcut,
      rcut_smth,
      sel: int,
      ntypes: int,
      g1_hiddens: int = [128, 128, 128],
      g2_hiddens: int = [16, 16, 16],
      axis_dim: int = 4,
      attn1_hidden: int = 128,
      attn1_nhead: int = 4,
      attn2_hidden: int = 16,
      attn2_nhead: int = 4,
      attn_dotr: bool = True,
      activation: str = "tanh",
      set_davg_zero: bool = True, # TODO
      **kwargs,
  ):
    super(DescrptSeUni, self).__init__()
    self.rcut = rcut
    self.rcut_smth = rcut_smth
    self.ntypes = ntypes
    self.nlayers = len(g1_hiddens) - 1
    assert self.nlayers == len(g2_hiddens) - 1
    sel = [sel] if isinstance(self, int) else sel
    self.nnei = sum(sel)  # 总的邻居数量
    assert len(sel) == 1
    self.sel = torch.tensor(sel)  # 每种元素在邻居中的位移    
    self.sec = self.sel
    self.axis_dim = axis_dim
    self.set_davg_zero = set_davg_zero
    self.g1_hiddens = g1_hiddens
    self.g2_hiddens = g2_hiddens
    self.type_embd = TypeEmbedNet(self.ntypes, g1_hiddens[0])
    self.g2_embd = mylinear(1, g2_hiddens[0])
    self.act = get_activation_fn(activation)
    self.use_attn2 = attn2_nhead > 0 and attn2_hidden > 0

    cal_1_dim = lambda g1d, g2d, ax: g1d + g2d*ax
    g1_in_dims = [cal_1_dim(d1,d2,self.axis_dim) \
                  for d1,d2 in zip(g1_hiddens, g2_hiddens)]
    self.linear1 = self._linear_layers(g1_in_dims, g1_hiddens)
    self.linear2 = self._linear_layers(g2_hiddens, g2_hiddens)
    self.attn2_map = torch.nn.ModuleList(
      [Atten2Map(ii, attn2_hidden, attn2_nhead) for ii in g2_hiddens])
    self.attn2_mh_apply = torch.nn.ModuleList(
      [Atten2MultiHeadApply(ii, attn2_nhead) for ii in g2_hiddens])
    self.attn2_ev_apply = torch.nn.ModuleList(
      [Atten2EquiVarApply(ii, attn2_nhead) for ii in g2_hiddens])
    self.lmg1 = torch.nn.ModuleList(
      [torch.nn.LayerNorm(ii, dtype=mydtype) for ii in g1_hiddens])
    self.lmg2 = torch.nn.ModuleList(
      [torch.nn.LayerNorm(ii, dtype=mydtype) for ii in g2_hiddens])

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
    return self.g1_hiddens[-1]

  def forward(
      self, 
      extended_coord, 
      atype,
      nlist,
      nlist_type,
      nlist_loc,
  ):
    """
    extended_coord:     [nb, nloc x 3]
    atype:              [nb, nloc]
    """
    nframes, nloc = nlist_loc.shape[:2]
    dmatrix, diff = prod_env_mat_se_a(
      extended_coord, nlist, atype,
      self.mean, self.stddev,
      self.rcut, self.rcut_smth)
    nlist_type[nlist_type == -1] = self.ntypes
    nlist_mask = (nlist != -1)
    masked_nlist_loc = nlist_loc * nlist_mask

    # nb x nloc x ng1
    g1 = (self.type_embd(atype))
    # nb x nloc x nnei x 1,  nb x nloc x nnei x 3
    g2, h2 = torch.split(dmatrix, [1, 3], dim=-1)
    # nb x nloc x nnei x ng2
    g2 = (self.g2_embd(g2))

    for ll in range(self.nlayers):
      g1, g2, h2 = self._one_layer(
        ll, g1, g2, h2, 
        masked_nlist_loc, 
        nlist_mask, 
        update_chnnl_2=(ll!=self.nlayers-1),
        use_attn2=self.use_attn2,
      )

    return g1, None, None

  def _linear_layers(
      self,
      in_dims,
      out_dims,
      bias: bool=True,
  ):
    ret = []
    for ii, oo in zip(in_dims, out_dims):
      ret.append(
        mylinear(ii, oo, bias=bias))
    return torch.nn.ModuleList(ret)

  def _one_layer(
      self,
      ll,       # 
      g1,       # nf x nloc x ng1
      g2,       # nf x nloc x nnei x ng2
      h2,       # nf x nloc x nnei x 3
      nlist,    # nf x nloc x nnei
      nlist_mask,
      update_chnnl_2: bool=True,
      use_attn2: bool=True,
  ):
    nb, nloc, nnei, _ = g2.shape
    assert (nb, nloc) == g1.shape[:2]
    assert (nb, nloc, nnei) == h2.shape[:3]
    ng1 = g1.shape[-1]
    ng2 = g2.shape[-1]
    nh2 = h2.shape[-1]

    # g1 = self.lmg1[ll](g1)
    # g2 = self.lmg2[ll](g2)

    g2_update = [g2]
    h2_update = [h2]
    g1_update = [g1]

    if update_chnnl_2:
      # nb x nloc x nnei x ng2
      g2_1 = self.act(self.linear2[ll](g2))
      g2_update.append(g2_1)
      if use_attn2:
        # nb x nloc x nnei x nnei x nh
        AA = self.attn2_map[ll](g2, h2)
        # nb x nloc x nnei x ng2
        g2_2 = self.attn2_mh_apply[ll](AA, g2)
        g2_update.append(g2_2)
        # nb x nloc x nnei x nh2
        h2_1 = self.attn2_ev_apply[ll](AA, h2)
        h2_update.append(h2_1)
    # nb x nloc x 3 x ng2
    h2g2 = torch.matmul(
      torch.transpose(h2, -1, -2), g2) / (float(nnei)**1)
    # nb x nloc x 3 x axis
    h2g2m = torch.split(h2g2, self.axis_dim, dim=-1)[0]
    # nb x nloc x axis x ng2
    g1_13 = torch.matmul(
      torch.transpose(h2g2m, -1, -2), h2g2) / (float(3.)**1)
    # nb x nloc x (axisxng2)
    g1_13 = g1_13.view(nb, nloc, self.axis_dim*ng2)
    # nb x nloc x [ng1+(axisxng2)]
    g1_1 = self.act(self.linear1[ll](
      torch.cat((g1, g1_13), dim=-1)
    ))
    g1_update.append(g1_1)

    def list_update(update_list):
      nitem = len(update_list)
      uu = update_list[0]
      for ii in range(1,nitem):
        uu = uu + update_list[ii]
      return uu / (float(nitem) ** 0.5)

    # update
    if update_chnnl_2:
      g2_new = list_update(g2_update)
      h2_new = list_update(h2_update)
    else:
      g2_new, h2_new = None, None
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
      if not self.set_davg_zero:
          mean = np.stack(all_davg)
          self.mean.copy_(torch.tensor(mean, device=env.DEVICE))
      stddev = np.stack(all_dstd)
      self.stddev.copy_(torch.tensor(stddev, device=env.DEVICE))
