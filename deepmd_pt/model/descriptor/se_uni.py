import numpy as np
import torch
import logging
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
from .dpa2_layer import DescrptDPA2Layer

mydtype = env.GLOBAL_PT_FLOAT_PRECISION
mydev = env.DEVICE

def torch_linear(*args, **kwargs):
  return torch.nn.Linear(*args, **kwargs, dtype=mydtype, device=mydev)
simple_linear  = SimpleLinear
mylinear = simple_linear


@Descriptor.register("se_uni")
class DescrptSeUni(Descriptor):
  def __init__(
      self,
      rcut,
      rcut_smth,
      sel: int,
      ntypes: int,
      nlayers: int = 3,
      g1_dim = 128,
      g2_dim = 16,
      axis_dim: int = 4,
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
      attn2_has_gate: bool = False,
      attn_dotr: bool = True,
      activation: str = "tanh",
      update_style: str = "res_avg",
      set_davg_zero: bool = True, # TODO
      smooth: bool = True,
      add_type_ebd_to_seq: bool = False,
      **kwargs,
  ):
    """
    smooth: 
        If strictly smooth, cannot be used with update_g1_has_attn
    add_type_ebd_to_seq:
        At the presence of seq_input (optional input to forward), 
        whether or not add an type embedding to seq_input. 
        If no seq_input is given, it has no effect. 
    """
    super(DescrptSeUni, self).__init__()
    self.epsilon = 1e-4 # protection of 1./nnei
    self.rcut = rcut
    self.rcut_smth = rcut_smth
    self.ntypes = ntypes
    self.nlayers = nlayers
    sel = [sel] if isinstance(sel, int) else sel
    self.nnei = sum(sel)  # 总的邻居数量
    assert len(sel) == 1
    self.sel = sel  # 每种元素在邻居中的位移
    self.sec = self.sel
    self.split_sel = self.sel
    self.axis_dim = axis_dim
    self.set_davg_zero = set_davg_zero
    self.g1_dim = g1_dim
    self.g2_dim = g2_dim
    self.act = get_activation_fn(activation)
    self.direct_dist = direct_dist
    self.add_type_ebd_to_seq = add_type_ebd_to_seq

    self.type_embd = TypeEmbedNet(self.ntypes, self.g1_dim)
    self.g2_embd = mylinear(1, self.g2_dim)
    layers = []
    for ii in range(nlayers):
      layers.append(
        DescrptDPA2Layer(
          rcut, rcut_smth, sel, ntypes, self.g1_dim, self.g2_dim,
          axis_dim=self.axis_dim,
          combine_grrg=combine_grrg,
          update_chnnl_2=(ii != nlayers - 1),
          do_bn_mode=do_bn_mode,
          bn_momentum=bn_momentum,
          update_g1_has_conv=update_g1_has_conv,
          update_g1_has_drrd=update_g1_has_drrd,
          update_g1_has_grrg=update_g1_has_grrg,
          update_g1_has_attn=update_g1_has_attn,
          update_g2_has_g1g1=update_g2_has_g1g1,
          update_g2_has_attn=update_g2_has_attn,
          update_h2=update_h2,
          attn1_hidden=attn1_hidden,
          attn1_nhead=attn1_nhead,
          attn2_has_gate=attn2_has_gate,
          attn2_hidden=attn2_hidden,
          attn2_nhead=attn2_nhead,
          attn_dotr=attn_dotr,
          activation=activation,
          update_style=update_style,
          smooth=smooth,
        ))
    self.layers = torch.nn.ModuleList(layers)

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
    return self.g1_dim

  @property
  def dim_in(self):
    """
    Returns the atomic input dimension of this descriptor
    """
    return self.g1_dim

  @property
  def dim_emb(self):
    """
    Returns the embedding dimension g2
    """
    return self.g2_dim

  def forward(
          self,
          extended_coord,
          nlist,
          atype,
          nlist_type: Optional[torch.Tensor] = None,
          nlist_loc: Optional[torch.Tensor] = None,
          atype_tebd: Optional[torch.Tensor] = None,
          nlist_tebd: Optional[torch.Tensor] = None,
          seq_input: Optional[torch.Tensor] = None
  ):

    """
    extended_coord:     [nb, nloc x 3]
    atype:              [nb, nloc]
    """
    assert nlist_type is not None
    assert nlist_loc is not None
    nframes, nloc = nlist_loc.shape[:2]
    # nb x nloc x nnei x 4, nb x nloc x nnei x 3, nb x nloc x nnei x 1
    dmatrix, diff, sw = prod_env_mat_se_a(
      extended_coord, nlist, atype,
      self.mean, self.stddev,
      self.rcut, self.rcut_smth)
    nlist_type[nlist_type == -1] = self.ntypes
    nlist_mask = (nlist != -1)
    masked_nlist_loc = nlist_loc * nlist_mask
    sw = torch.squeeze(sw, -1)
    # beyond the cutoff sw should be 0.0
    sw = sw.masked_fill(~nlist_mask, float(0.0))

    # [nframes, nloc, tebd_dim]
    if seq_input is not None:
      if seq_input.shape[0] == nframes * nloc:
        seq_input = seq_input[:, 0, :].reshape(nframes, nloc, -1)
      if self.add_type_ebd_to_seq:
        # nb x nloc x ng1
        atype_tebd = self.type_embd(atype) + seq_input
      else:
        # nb x nloc x ng1        
        atype_tebd = seq_input
        # wasted evalueation of type_embd, 
        # since whether seq_input is None or not can only be 
        # known at runtime, we cannot decide whether create the
        # type embedding net or not at `__init__`
        foo = self.type_embd(atype)
    else:
      # nb x nloc x ng1
      atype_tebd = self.type_embd(atype)

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

    for ll in self.layers:
      g1, g2, h2 = ll.forward(
        g1, g2, h2,
        masked_nlist_loc, nlist_mask, sw,
      )
    # uses the last layer.
    # nb x nloc x 3 x ng2
    h2g2 = ll._cal_h2g2(g2, h2, nlist_mask, sw)
    # (nb x nloc) x ng2 x 3
    rot_mat = torch.permute(h2g2, (0, 1, 3, 2))

    return g1, g2, diff, rot_mat.view(-1, self.dim_emb, 3)

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
          env_mat, _, _ = prod_env_mat_se_a(
              extended_coord, system['nlist'], system['atype'],
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
