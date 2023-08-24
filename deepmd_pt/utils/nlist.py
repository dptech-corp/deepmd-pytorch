import torch
import itertools
import numpy as np
from deepmd_pt.utils.env import (
  DEVICE,
  GLOBAL_PT_FLOAT_PRECISION,
)
from deepmd_pt.utils.region import (
  phys2inter,
  inter2phys,
  to_face_distance,
)
from typing import (
  List,
  Union,
)

def _build_neighbor_list(
    coord0 : torch.Tensor,
    coord1 : torch.Tensor,
    rcut : float,
    nsel : int,
    rmin : float = 1e-10,
    cut_overlap : bool = True,
) -> torch.Tensor:
  """build neightbor list for a single frame. keeps nsel neighbors.
  coord0 : [nloc x 3]
  coord1 : [nall x 3]

  ret: [nloc x nsel] stores indexes of coord1
  """
  nloc = coord0.shape[-1] // 3
  nall = coord1.shape[-1] // 3
  # nloc x nall x 3
  diff = coord1.view([-1,3])[None,:,:] - coord0.view([-1,3])[:,None,:]
  assert(list(diff.shape) == [nloc, nall, 3])
  # nloc x nall
  rr = torch.linalg.norm(diff, dim=-1)
  rr, nlist = torch.sort(rr, dim=-1)
  if cut_overlap:
    # nloc x (nall-1)
    rr = torch.split(rr, [1,nall-1], dim=-1)[-1]
    nlist = torch.split(nlist, [1,nall-1], dim=-1)[-1]
  # nloc x nsel
  nnei = rr.shape[1]
  rr = torch.split(rr, [nsel,nnei-nsel], dim=-1)[0]
  nlist = torch.split(nlist, [nsel,nnei-nsel], dim=-1)[0]
  nlist = nlist.masked_fill((rr > rcut), -1)
  return nlist

def build_neighbor_list_lower(
    coord0 : torch.Tensor,
    coord1 : torch.Tensor,
    atype : torch.Tensor,
    rcut : float,
    nsel : Union[int, List[int]],
    distinguish_types: bool = True,
) -> torch.Tensor:
  """build neightbor list for a single frame. keeps nsel neighbors.
  Parameters
  ----------
  coord0 : torch.Tensor
        local coordinates of shape [nloc x 3]
  coord1 : torch.Tensor
        exptended coordinates of shape [nall x 3]
  atype : torch.Tensor
        extended atomic types of shape [nall]
  rcut: float
        cut-off radius
  nsel: int or List[int]
        maximal number of neighbors (of each type).
        if distinguish_types==True, nsel should be list and 
        the length of nsel should be equal to number of 
        types.
  distinguish_types: bool
        distinguish different types. 
  
  Returns
  -------
  neighbor_list : torch.Tensor
        Neighbor list of shape [nloc x nsel], the neighbors
        are stored in an ascending order. If the number of 
        neighbors is less than nsel, the positions are masked
        with -1. The neighbor list of an atom looks like
        |------ nsel ------|
        xx xx xx xx -1 -1 -1
        if distinguish_types==True and we have two types
        |---- nsel[0] -----| |---- nsel[1] -----|
        xx xx xx xx -1 -1 -1 xx xx xx -1 -1 -1 -1

  """
  nloc = coord0.shape[0]//3
  nall = coord1.shape[0]//3
  if nloc == 0 or nall == 0:
    return None
  fake_type = torch.max(atype) + 1
  if isinstance(nsel, int): 
    nsel = [nsel]
  # nloc x nsel
  nlist = _build_neighbor_list(
    coord0, coord1, rcut, sum(nsel), cut_overlap=True)
  if not distinguish_types:
    return nlist
  else:
    ret_nlist = []
    # nloc x nall
    tmp_atype = torch.tile(atype.unsqueeze(0), [nloc,1])
    mask = (nlist == -1)    
    # nloc x s(nsel)
    tnlist = torch.gather(
      tmp_atype, 1, nlist.masked_fill(mask, 0),
    )
    tnlist = tnlist.masked_fill(mask, -1)
    snsel = tnlist.shape[1]
    for ii,ss in enumerate(nsel):
      # nloc x s(nsel)
      pick_mask = (tnlist == ii)
      # nloc x s(nsel), stable sort, nearer neighbors first
      pick_mask, imap = torch.sort(
        pick_mask, dim=-1, descending=True, stable=True)
      # nloc x s(nsel)
      inlist = torch.gather(nlist, 1, imap)
      inlist = inlist.masked_fill(~pick_mask, -1)
      # nloc x nsel[ii]
      ret_nlist.append(
        torch.split(inlist, [ss,snsel-ss], dim=-1)[0]
      )
    return torch.concat(ret_nlist, dim=-1)

build_neighbor_list = torch.vmap(
  build_neighbor_list_lower, 
  in_dims=(0,0,0,None,None), 
  out_dims=(0),
)


def extend_coord_with_ghosts(
    coord : torch.Tensor,
    atype : torch.Tensor,
    cell : torch.Tensor,
    rcut : float,
) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
  """Extend the coordinates of the atoms by appending peridoc images.
  The number of images is large enough to ensure all the neighbors
  within rcut are appended.

  Parameters
  ----------
  coord: torch.Tensor
        original coordinates of shape [-1, nloc*3].
  atype: torch.Tensor
        atom type of shape [-1, nloc].
  cell: torch.Tensor
        simulation cell tensor of shape [-1, 9].

  Returns
  -------
  extended_coord: torch.Tensor
        extended coordinates of shape [-1, nall*3].
  extended_atype: torch.Tensor
        extended atom type of shape [-1, nall].
  index_mapping: torch.Tensor
        maping extended index to the local index
  
  """
  nf, nloc = atype.shape  
  aidx = torch.tile(torch.arange(nloc).unsqueeze(0), [nf,1])
  coord = coord.view([nf, nloc, 3])
  cell = cell.view([nf, 3, 3])
  # nf x 3
  to_face = to_face_distance(cell)
  # nf x 3
  # *2: ghost copies on + and - directions
  # +1: central cell
  nbuff = torch.ceil(rcut / to_face).to(torch.long)
  # 3
  nbuff = torch.max(nbuff, dim=0, keepdim=False).values
  ncopy = nbuff * 2 + 1
  # 3
  npnbuff = nbuff.detach().numpy()
  rr = [range(-npnbuff[ii], npnbuff[ii]+1) for ii in range(3)]
  shift_idx = sorted(list(itertools.product(*rr)), key=np.linalg.norm)
  # ns x 3
  shift_idx = torch.tensor(shift_idx, dtype=GLOBAL_PT_FLOAT_PRECISION)
  ns, _ = shift_idx.shape
  nall = ns * nloc
  # nf x ns x 3
  shift_vec = torch.einsum("sd,fdk->fsk", shift_idx, cell)
  # nf x ns x nloc x 3
  extend_coord = coord[:,None,:,:] + shift_vec[:,:,None,:]
  # nf x ns x nloc
  extend_atype = torch.tile(atype.unsqueeze(-2), [1, ns, 1])
  # nf x ns x nloc
  extend_aidx = torch.tile(aidx.unsqueeze(-2), [1, ns, 1])

  return (
    extend_coord.reshape([nf, nall*3]),
    extend_atype.view([nf, nall]),
    extend_aidx.view([nf, nall]),
  )

