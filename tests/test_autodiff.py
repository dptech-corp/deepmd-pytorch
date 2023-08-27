import torch
import unittest
import numpy as np
from deepmd_pt.utils import env
from deepmd_pt.model.model import EnergyModelSeA, EnergyModelDPA1, EnergyModelDPA2, EnergyModelDPAUni, ForceModelDPAUni, EnergyModelHybrid, ForceModelHybrid

from deepmd_pt.utils.dataloader import DpLoaderSet
from deepmd_pt.utils.stat import make_stat_input

dtype = torch.float64

from .test_permutation import (
  model_se_e2_a,
  model_dpa1,
  model_dpau,
  infer_model,
  make_sample,
)

# from deepmd-kit repo
def finite_difference(f, x, delta=1e-6):
    in_shape = x.shape
    y0 = f(x)
    out_shape = y0.shape
    res = np.empty(out_shape + in_shape)
    for idx in np.ndindex(*in_shape):
        diff = np.zeros(in_shape)
        diff[idx] += delta
        y1p = f(x + diff)
        y1n = f(x - diff)
        res[(Ellipsis, *idx)] = (y1p - y1n) / (2 * delta)
    return res

def stretch_box(old_coord, old_box, new_box):
    ocoord = old_coord.reshape(-1, 3)
    obox = old_box.reshape(3, 3)
    nbox = new_box.reshape(3, 3)
    ncoord = ocoord @ np.linalg.inv(obox) @ nbox
    return ncoord.reshape(old_coord.shape)


class TestForce():
  def test(
      self,
  ):
    places = 8
    delta = 1e-5
    natoms = 5
    cell = torch.rand([3, 3], dtype=dtype)
    cell = (cell + cell.T) + 5. * torch.eye(3)
    coord = torch.rand([natoms, 3], dtype=dtype)
    coord = torch.matmul(coord, cell)
    atype = torch.IntTensor([0, 0, 0, 1, 1])      
    # assumes input to be numpy tensor
    coord = coord.numpy()
    def np_infer(
        coord,
    ):      
      ret = infer_model(
        self.model, 
        torch.tensor(coord), cell, atype,
        type_split=self.type_split)
      # detach
      ret = {kk: ret[kk].detach().numpy() for kk in ret}
      return ret
    ff = lambda coord: np_infer(coord)["energy"]
    fdf = -finite_difference(ff, coord, delta=delta).squeeze()
    rff = np_infer(coord)["force"]
    np.testing.assert_almost_equal(fdf, rff, decimal=places)


class TestVirial():
  def test(
      self,
  ):
    places = 8
    delta = 1e-4
    natoms = 5
    cell = torch.rand([3, 3], dtype=dtype)
    cell = (cell) + 5. * torch.eye(3)
    coord = torch.rand([natoms, 3], dtype=dtype)
    coord = torch.matmul(coord, cell)
    atype = torch.IntTensor([0, 0, 0, 1, 1])      
    # assumes input to be numpy tensor
    coord = coord.numpy()
    cell = cell.numpy()
    def np_infer(
        new_cell,
    ):      
      ret = infer_model(
        self.model,
        torch.tensor(stretch_box(coord, cell, new_cell)),
        torch.tensor(new_cell),
        atype,
        type_split=self.type_split,
      )
      # detach
      ret = {kk: ret[kk].detach().numpy() for kk in ret}
      return ret
    ff = lambda bb: np_infer(bb)["energy"]
    fdv = -(finite_difference(ff, cell, delta=delta)\
            .transpose(0, 2, 1) @ cell).squeeze()
    rfv = np_infer(cell)["virial"]
    np.testing.assert_almost_equal(fdv, rfv, decimal=places)    
    

class TestEnergyModelSeAForce(unittest.TestCase, TestForce):
  def setUp(self):
    model_params = model_se_e2_a
    sampled = make_sample(model_params)
    self.type_split = False
    self.model = EnergyModelSeA(model_params, sampled).to(env.DEVICE)

class TestEnergyModelSeAVirial(unittest.TestCase, TestVirial):
  def setUp(self):
    model_params = model_se_e2_a
    sampled = make_sample(model_params)
    self.type_split = False
    self.model = EnergyModelSeA(model_params, sampled).to(env.DEVICE)

class TestEnergyModelDPA1Force(unittest.TestCase, TestForce):
  def setUp(self):
    model_params = model_dpa1
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = EnergyModelDPA1(model_params, sampled).to(env.DEVICE)

class TestEnergyModelDPA1Virial(unittest.TestCase, TestVirial):
  def setUp(self):
    model_params = model_dpa1
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = EnergyModelDPA1(model_params, sampled).to(env.DEVICE)


class TestEnergyModelDPAUniForce(unittest.TestCase, TestForce):
  def setUp(self):
    model_params = model_dpau
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = EnergyModelDPAUni(model_params, sampled).to(env.DEVICE)

class TestEnergyModelDPAUniVirial(unittest.TestCase, TestVirial):
  def setUp(self):
    model_params = model_dpau
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = EnergyModelDPAUni(model_params, sampled).to(env.DEVICE)


