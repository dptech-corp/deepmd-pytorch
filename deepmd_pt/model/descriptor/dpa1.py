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

@Descriptor.register("dpa1")
@Descriptor.register("se_atten")
class DescrptDPA1(Descriptor):
  def __init__(
      self,
      rcut,
      rcut_smth,
      sel,
      ntypes: int,
      neuron: list = [25, 50, 100],
      axis_neuron: int = 16,
      tebd_dim: int = 8,
      tebd_input_mode: str = 'concat',
      # set_davg_zero: bool = False,
      set_davg_zero: bool = True,  # TODO
      attn: int = 128,
      attn_layer: int = 2,
      attn_dotr: bool = True,
      attn_mask: bool = False,
      post_ln=True,
      ffn=False,
      ffn_embed_dim=1024,
      activation="tanh",
      scaling_factor=1.0,
      head_num=1,
      normalize=True,
      temperature=None,
      return_rot=False,
      concat_output_tebd: bool = True,
  ):
    super(DescrptDPA1, self).__init__()
    self.se_atten = DescrptBlockSeAtten(
      rcut, rcut_smth, sel, ntypes,
      neuron=neuron,
      axis_neuron=axis_neuron,
      tebd_dim=tebd_dim,
      tebd_input_mode=tebd_input_mode,
      set_davg_zero=set_davg_zero,
      attn=attn,
      attn_layer=attn_layer,
      attn_dotr=attn_dotr,
      attn_mask=attn_mask,
      post_ln=post_ln,
      ffn=ffn,
      ffn_embed_dim=ffn_embed_dim,
      activation=activation,
      scaling_factor=scaling_factor,
      head_num=head_num,
      normalize=normalize,
      temperature=temperature,
      return_rot=return_rot,
    )
    self.type_embedding = TypeEmbedNet(ntypes, tebd_dim)
    self.tebd_dim = tebd_dim
    self.concat_output_tebd = concat_output_tebd
    
  def get_rcut(self)->float:
    """
    Returns the cut-off radius
    """
    return self.se_atten.get_rcut()

  def get_nsel(self)->int:
    """
    Returns the number of selected atoms in the cut-off radius
    """
    return self.se_atten.get_nsel()

  def get_ntype(self)->int:
    """
    Returns the number of element types
    """
    return self.se_atten.get_ntype()

  def get_dim_out(self)->int:
    """
    Returns the output dimension
    """
    ret = self.se_atten.get_dim_out()
    if self.concat_output_tebd:
      ret += self.tebd_dim
    return ret

  def compute_input_stats(self, merged):
    self.se_atten.compute_input_stats(merged)
    
  def init_desc_stat(self, sumr, suma, sumn, sumr2, suma2):
    self.se_atten.init_desc_stat(sumr, suma, sumn, sumr2, suma2)

  def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
  ):
    nframes, nloc, nnei = nlist.shape
    nall = extended_coord.view(nframes, -1).shape[1] // 3
    g1_ext = self.type_embedding(extended_atype)
    if self.concat_output_tebd:
      g1_inp = g1_ext[:,:nloc,:]
    g1, env_mat, diff, rot_mat, sw = self.se_atten(
      nlist,
      extended_coord,
      extended_atype,
      g1_ext, mapping,
    )
    if self.concat_output_tebd:
      g1 = torch.cat([g1, g1_inp], dim=-1)
    return g1, env_mat, diff, rot_mat, sw
    
    