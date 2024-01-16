import torch
import numpy as np
from typing import (
  List,
  Optional,
  Dict,
)

from deepmd_utils.model_format import (
  get_reduce_name,
  get_deriv_name,
  VariableDef,
  OutputVariableDef,
  FittingOutputDef,
)


def atomic_virial_corr(
    extended_coord: torch.Tensor, 
    atom_energy: torch.Tensor,
):
  nall = extended_coord.shape[1]
  nloc = atom_energy.shape[1]
  coord, _ = torch.split(extended_coord, [nloc, nall-nloc], dim=1)
  # no derivative with respect to the loc coord.
  coord = coord.detach()
  ce = coord * atom_energy
  sumce0, sumce1, sumce2 = torch.split(torch.sum(ce, dim=1), [1,1,1], dim=-1)
  faked_grad = torch.ones_like(sumce0)
  lst = torch.jit.annotate(List[Optional[torch.Tensor]], [faked_grad])
  extended_virial_corr0 = torch.autograd.grad([sumce0], [extended_coord], grad_outputs=lst, create_graph=True)[0]
  assert extended_virial_corr0 is not None
  extended_virial_corr1 = torch.autograd.grad([sumce1], [extended_coord], grad_outputs=lst, create_graph=True)[0]
  assert extended_virial_corr1 is not None
  extended_virial_corr2 = torch.autograd.grad([sumce2], [extended_coord], grad_outputs=lst, create_graph=True)[0]
  assert extended_virial_corr2 is not None
  extended_virial_corr = torch.concat([extended_virial_corr0.unsqueeze(-1),
                                       extended_virial_corr1.unsqueeze(-1),
                                       extended_virial_corr2.unsqueeze(-1)], dim=-1)
  return extended_virial_corr


def task_deriv_one(
    atom_energy: torch.Tensor,
    energy: torch.Tensor,
    extended_coord: torch.Tensor,
):
  faked_grad = torch.ones_like(energy)
  lst = torch.jit.annotate(List[Optional[torch.Tensor]], [faked_grad])
  extended_force = torch.autograd.grad([energy], [extended_coord], grad_outputs=lst, create_graph=True)[0]
  assert extended_force is not None
  extended_force = -extended_force
  extended_virial = extended_force.unsqueeze(-1) @ extended_coord.unsqueeze(-2)
  # the correction sums to zero, which does not contribute to global virial
  extended_virial_corr = atomic_virial_corr(extended_coord, atom_energy)
  extended_virial = extended_virial + extended_virial_corr
  return extended_force, extended_virial


def get_leading_dims(
    vv: torch.Tensor, 
    vdef: OutputVariableDef,
):
  vshape = vv.shape
  return list(vshape[:(len(vshape) - len(vdef.shape))])

def get_atom_axis(
    vdef: torch.Tensor,
):
  atom_axis = -(len(vdef.shape) + 1)
  return atom_axis

def take_deriv(
    vv: torch.Tensor,
    svv: torch.Tensor,
    vdef: OutputVariableDef,
    coord_ext: torch.Tensor,
):
  size = 1
  for ii in vdef.shape:
    size *= ii
  vv1 = vv.view(get_leading_dims(vv, vdef) + [size])
  svv1 = svv.view(get_leading_dims(svv, vdef) + [size])
  split_vv1 = torch.split(vv1, [1]*size, dim=-1)
  split_svv1 = torch.split(svv1, [1]*size, dim=-1)
  split_ff, split_avir = [], []
  for vvi, svvi in zip(split_vv1, split_svv1):
    # nf x nloc x 3, nf x nloc x 3 x 3
    ffi, aviri = task_deriv_one(vvi, svvi, coord_ext)
    # nf x nloc x 1 x 3, nf x nloc x 1 x 3 x 3
    ffi = ffi.unsqueeze(-2)
    aviri = aviri.unsqueeze(-3)
    split_ff.append(ffi)
    split_avir.append(aviri)
  # nf x nloc x v_dim x 3, nf x nloc x v_dim x 3 x 3
  ff = torch.concat(split_ff, dim=-2)
  avir = torch.concat(split_avir, dim=-3)
  return ff, avir


def fit_output_to_model_output(
    fit_ret : Dict[str, torch.Tensor],
    fit_output_def: FittingOutputDef,
    coord_ext: torch.Tensor,
) -> Dict[str, torch.Tensor]:
  model_ret = {kk:vv for kk,vv in fit_ret.items()}
  for kk, vv in fit_ret.items():
    vdef = fit_output_def[kk]
    shap = vdef.shape
    atom_axis = -(len(shap) + 1)
    if vdef.reduciable:
      kk_redu = get_reduce_name(kk)
      model_ret[kk_redu] = torch.sum(vv, dim=atom_axis)    
      if vdef.differentiable:
        kk_derv_r, kk_derv_c = get_deriv_name(kk)
        dr, dc = take_deriv(vv, model_ret[kk_redu], vdef, coord_ext)
        model_ret[kk_derv_r] = dr
        model_ret[kk_derv_c] = dc
  return model_ret
