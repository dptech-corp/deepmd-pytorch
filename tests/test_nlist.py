import unittest
import torch
from deepmd_pt.utils.preprocess import Region3D
from deepmd_pt.utils import env
from deepmd_pt.utils.region import (
  phys2inter, 
  inter2phys,
  to_face_distance,
)
from deepmd_pt.utils.nlist import (
  extend_coord_with_ghosts,
  build_neighbor_list,
)
from deepmd_pt.utils.preprocess import(
  build_neighbor_list as legacy_build_neighbor_list,
)

dtype = torch.float64

class TestNeighList(unittest.TestCase):
  def setUp(self):
    self.nf = 3
    self.nloc = 2
    self.ns = 5 * 5 * 3
    self.nall = self.ns * self.nloc
    self.cell = torch.tensor(
      [[1, 0, 0], [0.4, 0.8, 0], [0.1, 0.3, 2.1]], dtype=dtype).to(env.DEVICE)
    self.icoord = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.1]], dtype=dtype).to(env.DEVICE)
    self.atype = torch.tensor([0, 1], dtype=torch.int).to(env.DEVICE)
    [self.cell, self.icoord, self.atype] = [
      ii.unsqueeze(0) for ii in [self.cell, self.icoord, self.atype]]
    self.coord = inter2phys(self.icoord, self.cell).view([-1, self.nloc*3])    
    self.cell = self.cell.view([-1, 9])
    [self.cell, self.coord, self.atype] = [
      torch.tile(ii, [self.nf,1]) for ii in [self.cell, self.coord, self.atype]]
    self.rcut = 1.01
    self.prec = 1e-10
    self.nsel = [10, 10]
    # genrated by preprocess.build_neighbor_list
    # ref_nlist, _, _ = legacy_build_neighbor_list(
    #   2, ecoord[0], eatype[0],
    #   self.rcut, 
    #   torch.tensor([10,20], dtype=torch.long),
    #   mapping[0], type_split=True, )
    self.ref_nlist = torch.tensor(
      [[10,  4, 20, 12, 30,  2, -1, -1, -1, -1,  5,  3, 15,  1, -1, -1, -1, -1, -1, -1],
       [10, 12, 36,  0, -1, -1, -1, -1, -1, -1,  5, 11, 21, 13, 31,  3, -1, -1, -1, -1]]
    ).to(env.DEVICE)
    
  def test_build_notype(self):
    ecoord, eatype, mapping = extend_coord_with_ghosts(
      self.coord, self.atype, self.cell, self.rcut)
    nlist = build_neighbor_list(
      ecoord, eatype, self.nloc,
      self.rcut, sum(self.nsel), distinguish_types=False)
    torch.testing.assert_close(
      nlist[0], nlist[1])
    print(torch.sort(nlist[0], dim=-1)[0])
    print(torch.sort(self.ref_nlist, dim=-1)[0])
    torch.testing.assert_close(
      torch.sort(nlist[0], dim=-1)[0],
      torch.sort(self.ref_nlist, dim=-1)[0],
    )

  def test_build_type(self):
    ecoord, eatype, mapping = extend_coord_with_ghosts(
      self.coord, self.atype, self.cell, self.rcut)
    nlist = build_neighbor_list(
      ecoord, eatype, self.nloc,
      self.rcut, self.nsel, distinguish_types=True,
    )
    torch.testing.assert_close(nlist[0], nlist[1])
    for ii in range(2):
      torch.testing.assert_close(
        torch.sort(torch.split(nlist[0], self.nsel, dim=-1)[ii], dim=-1)[0],
        torch.sort(torch.split(self.ref_nlist, self.nsel, dim=-1)[ii], dim=-1)[0],
      )

  def test_extend_coord(self):
    ecoord, eatype, mapping = extend_coord_with_ghosts(
      self.coord, self.atype, self.cell, self.rcut)
    # expected ncopy x nloc
    self.assertEqual(list(ecoord.shape), [self.nf, self.nall*3])
    self.assertEqual(list(eatype.shape), [self.nf, self.nall])
    self.assertEqual(list(mapping.shape), [self.nf, self.nall])
    # check the nloc part is identical with original coord
    torch.testing.assert_close(
      ecoord[:,:self.nloc*3], self.coord, rtol=self.prec, atol=self.prec)
    # check the shift vectors are aligned with grid
    shift_vec = \
      ecoord.view([-1, self.ns, self.nloc, 3]) - \
      self.coord.view([-1, self.nloc, 3])[:,None,:,:]
    shift_vec = shift_vec.view([-1, self.nall, 3])    
    # hack!!! assumes identical cell across frames
    shift_vec = torch.matmul(shift_vec, torch.linalg.inv(self.cell.view([self.nf,3,3])[0]))
    # nf x nall x 3
    shift_vec = torch.round(shift_vec)
    # check: identical shift vecs
    torch.testing.assert_close(
      shift_vec[0], shift_vec[1], rtol=self.prec, atol=self.prec)
    # check: shift idx aligned with grid
    mm, cc = torch.unique(shift_vec[0][:,0], dim=-1, return_counts=True)
    torch.testing.assert_close(
      mm, torch.tensor([-2,-1,0,1,2], dtype=dtype).to(env.DEVICE), rtol=self.prec, atol=self.prec)
    torch.testing.assert_close(
      cc, torch.tensor([30,30,30,30,30], dtype=torch.long).to(env.DEVICE), rtol=self.prec, atol=self.prec)
    mm, cc = torch.unique(shift_vec[1][:,1], dim=-1, return_counts=True)
    torch.testing.assert_close(
      mm, torch.tensor([-2,-1,0,1,2], dtype=dtype).to(env.DEVICE), rtol=self.prec, atol=self.prec)
    torch.testing.assert_close(
      cc, torch.tensor([30,30,30,30,30], dtype=torch.long).to(env.DEVICE), rtol=self.prec, atol=self.prec)
    mm, cc = torch.unique(shift_vec[1][:,2], dim=-1, return_counts=True)
    torch.testing.assert_close(
      mm, torch.tensor([-1,0,1], dtype=dtype).to(env.DEVICE), rtol=self.prec, atol=self.prec)
    torch.testing.assert_close(
      cc, torch.tensor([50,50,50], dtype=torch.long).to(env.DEVICE), rtol=self.prec, atol=self.prec)

