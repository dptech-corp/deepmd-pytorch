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
  TypeEmbedNet, SimpleLinear, Identity, Linear
)
from deepmd_pt.utils.nlist import build_multiple_neighbor_list

from .se_atten import analyze_descrpt
from .se_atten import DescrptBlockSeAtten
from .repformers import DescrptBlockRepformers


@Descriptor.register("dpa2")
class DescrptDPA2(Descriptor):
  def __init__(
      self,
      ntypes,
      repinit_rcut,
      repinit_rcut_smth,
      repinit_nsel,
      repformer_rcut,
      repformer_rcut_smth,
      repformer_nsel,
      # kwargs
      tebd_dim: int = 8,
      concat_output_tebd: bool = True,
      repinit_neuron: list = [25, 50, 100],
      repinit_axis_neuron: int = 16,
      repinit_set_davg_zero: bool = True,  # TODO
      repinit_activation="tanh",
      # repinit still unclear:
      # ffn, ffn_embed_dim, scaling_factor, normalize,
      repformer_nlayers : int = 3,
      repformer_g1_dim: int = 128,
      repformer_g2_dim: int = 16,
      repformer_axis_dim: int = 4,
      repformer_combine_grrg: bool = False,
      repformer_do_bn_mode: str = 'no',
      repformer_bn_momentum: float = 0.1,
      repformer_update_g1_has_conv: bool = True,
      repformer_update_g1_has_drrd: bool = True,
      repformer_update_g1_has_grrg: bool = True,
      repformer_update_g1_has_attn: bool = True,
      repformer_update_g2_has_g1g1: bool = True,
      repformer_update_g2_has_attn: bool = True,
      repformer_update_h2: bool = False,
      repformer_attn1_hidden: int = 64,
      repformer_attn1_nhead: int = 4,
      repformer_attn2_hidden: int = 16,
      repformer_attn2_nhead: int = 4,
      repformer_attn2_has_gate: bool = False,
      repformer_activation: str = "tanh",
      repformer_update_style: str = "res_avg",
      repformer_set_davg_zero: bool = True, # TODO
      repformer_add_type_ebd_to_seq: bool = False,
  ):
    super(DescrptDPA2, self).__init__()
    self.repinit = DescrptBlockSeAtten(
      repinit_rcut,
      repinit_rcut_smth,
      repinit_nsel,
      ntypes,
      attn_layer=0,
      neuron=repinit_neuron,
      axis_neuron=repinit_axis_neuron,
      tebd_dim=tebd_dim,
      tebd_input_mode='concat',
      # tebd_input_mode='dot_residual_s',
      set_davg_zero=repinit_set_davg_zero,
      activation=repinit_activation,
    )
    self.repformers = DescrptBlockRepformers(
      repformer_rcut,
      repformer_rcut_smth,
      repformer_nsel,
      ntypes,
      nlayers=repformer_nlayers,
      g1_dim=repformer_g1_dim,
      g2_dim=repformer_g2_dim,
      axis_dim=repformer_axis_dim,
      combine_grrg=repformer_combine_grrg,
      direct_dist=False,
      do_bn_mode=repformer_do_bn_mode,
      bn_momentum=repformer_bn_momentum,
      update_g1_has_conv=repformer_update_g1_has_conv,
      update_g1_has_drrd=repformer_update_g1_has_drrd,
      update_g1_has_grrg=repformer_update_g1_has_grrg,
      update_g1_has_attn=repformer_update_g1_has_attn,
      update_g2_has_g1g1=repformer_update_g2_has_g1g1,
      update_g2_has_attn=repformer_update_g2_has_attn,
      update_h2=repformer_update_h2,
      attn1_hidden=repformer_attn1_hidden,
      attn1_nhead=repformer_attn1_nhead,
      attn2_hidden=repformer_attn2_hidden,
      attn2_nhead=repformer_attn2_nhead,
      attn2_has_gate=repformer_attn2_has_gate,
      activation=repformer_activation,
      update_style=repformer_update_style,
      set_davg_zero=repformer_set_davg_zero,
      smooth=True,
      add_type_ebd_to_seq=repformer_add_type_ebd_to_seq,
    )
    self.type_embedding = TypeEmbedNet(ntypes, tebd_dim)
    if self.repinit.dim_out == self.repformers.dim_in:
      self.g1_shape_tranform = Identity()
    else:
      self.g1_shape_tranform = Linear(
        self.repinit.dim_out, 
        self.repformers.dim_in,
        bias=False, 
        init="glorot",
      )
    assert self.repinit.rcut > self.repformers.rcut
    assert self.repinit.sel[0] > self.repformers.sel[0]
    self.concat_output_tebd = concat_output_tebd
    self.tebd_dim = tebd_dim
    self.rcut = self.repinit.get_rcut()
    self.ntypes = ntypes
    self.sel = self.repinit.sel

  def get_rcut(self)->float:
    """
    Returns the cut-off radius
    """
    return self.rcut

  def get_nsel(self)->int:
    """
    Returns the number of selected atoms in the cut-off radius
    """
    return sum(self.sel)

  def get_ntype(self)->int:
    """
    Returns the number of element types
    """
    return self.ntypes

  def get_dim_out(self)->int:
    """
    Returns the output dimension of this descriptor
    """
    ret = self.dim_out
    if self.concat_output_tebd:
      ret += self.tebd_dim
    return ret

  @property
  def dim_out(self):
    return self.repformers.get_dim_out()

  @property
  def dim_emb(self):
    """
    Returns the embedding dimension g2
    """
    return self.g2_dim

  def compute_input_stats(self, merged):
      sumr, suma, sumn, sumr2, suma2 = [], [], [], [], []
      for ii, descrpt in enumerate([self.repinit, self.repformers]):
          merged_tmp = [
            {key: item[key] if not isinstance(item[key], list) else item[key][ii] \
             for key in item} \
            for item in merged
          ]
          sumr_tmp, suma_tmp, sumn_tmp, sumr2_tmp, suma2_tmp = descrpt.compute_input_stats(merged_tmp)
          sumr.append(sumr_tmp)
          suma.append(suma_tmp)
          sumn.append(sumn_tmp)
          sumr2.append(sumr2_tmp)
          suma2.append(suma2_tmp)
      return sumr, suma, sumn, sumr2, suma2
    

  def init_desc_stat(self, sumr, suma, sumn, sumr2, suma2):
      for ii, descrpt in enumerate(self.descriptor_list):
          descrpt.init_desc_stat(sumr[ii], suma[ii], sumn[ii], sumr2[ii], suma2[ii])

  def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
  ):
    nframes, nloc, nnei = nlist.shape
    nall = extended_coord.view(nframes, -1).shape[1] // 3
    # nlists
    nlist_dict = build_multiple_neighbor_list(
      extended_coord,
      nlist,
      [self.repformers.get_rcut(), self.repinit.get_rcut()],
      [self.repformers.get_nsel(), self.repinit.get_nsel()],
    )
    # repinit
    g1_ext = self.type_embedding(extended_atype)
    if self.concat_output_tebd:
      g1_inp = g1_ext[:,:nloc,:]
    g1, env_mat, diff, rot_mat, sw = self.repinit(
      nlist_dict[self.repinit.get_rcut()],
      extended_coord,
      extended_atype,
      g1_ext, mapping,
    )
    # linear to change shape
    g1 = self.g1_shape_tranform(g1)
    # mapping g1
    mapping_ext = mapping.view(nframes, nall)\
                         .unsqueeze(-1)\
                         .expand(-1, -1, g1.shape[-1])
    g1_ext = torch.gather(g1, 1, mapping_ext)
    # repformer
    g1, env_mat, diff, rot_mat, sw = self.repformers(
      nlist_dict[self.repformers.get_rcut()],
      extended_coord,
      extended_atype,
      g1_ext, mapping,
    )
    if self.concat_output_tebd:
      g1 = torch.cat([g1, g1_inp], dim=-1)
    return g1, env_mat, diff, rot_mat, sw
    