import torch, copy
import unittest
from deepmd_pt.utils.preprocess import (
  Region3D, make_env_mat,
)
from deepmd_pt.utils import env
from deepmd_pt.model.model import get_model
from deepmd_pt.utils.dataloader import DpLoaderSet
from deepmd_pt.utils.stat import make_stat_input
from .test_permutation import (
  infer_model, 
  make_sample,
  model_se_e2_a,
  model_dpa1,
  model_dpa2,
  model_dpau,
  model_hybrid,
)

dtype = torch.float64

class TestRot():
  def test(
      self,
  ):
    prec = 1e-10
    natoms = 5
    cell = 10. * torch.eye(3, dtype=dtype)
    coord = 2*torch.rand([natoms, 3], dtype=dtype)
    shift = torch.tensor([4, 4, 4], dtype=dtype)
    atype = torch.IntTensor([0, 0, 0, 1, 1])      
    from scipy.stats import special_ortho_group
    rmat = torch.tensor(special_ortho_group.rvs(3), dtype=dtype)

    # rotate only coord and shift to the center of cell
    coord_rot = torch.matmul(coord, rmat)
    ret0 = infer_model(self.model, coord + shift, cell, atype, type_split=self.type_split)
    ret1 = infer_model(self.model, coord_rot + shift, cell, atype, type_split=self.type_split)
    torch.testing.assert_close(ret0['energy'], ret1['energy'], rtol=prec, atol=prec)
    rmat = rmat.to(env.DEVICE)
    torch.testing.assert_close(torch.matmul(ret0['force'], rmat), ret1['force'], rtol=prec, atol=prec)
    if not hasattr(self, "test_virial") or self.test_virial:
      torch.testing.assert_close(
        torch.matmul(rmat.T, torch.matmul(ret0['virial'], rmat)), 
        ret1['virial'], rtol=prec, atol=prec)
    
    # rotate coord and cell
    torch.manual_seed(0)
    cell = torch.rand([3, 3], dtype=dtype)
    cell = (cell + cell.T) + 5. * torch.eye(3)
    coord = torch.rand([natoms, 3], dtype=dtype)
    coord = torch.matmul(coord, cell)
    atype = torch.IntTensor([0, 0, 0, 1, 1])
    coord_rot = torch.matmul(coord, rmat)
    cell_rot = torch.matmul(cell, rmat)
    ret0 = infer_model(self.model, coord, cell, atype, type_split=self.type_split)
    ret1 = infer_model(self.model, coord_rot, cell_rot, atype, type_split=self.type_split)
    torch.testing.assert_close(ret0['energy'], ret1['energy'], rtol=prec, atol=prec)
    rmat = rmat.to(env.DEVICE)
    torch.testing.assert_close(torch.matmul(ret0['force'], rmat), ret1['force'], rtol=prec, atol=prec)
    if not hasattr(self, "test_virial") or self.test_virial:
      torch.testing.assert_close(
        torch.matmul(rmat.T, torch.matmul(ret0['virial'], rmat)), 
        ret1['virial'], rtol=prec, atol=prec)    


class TestEnergyModelSeA(unittest.TestCase, TestRot):
  def setUp(self):
    model_params = model_se_e2_a
    sampled = make_sample(model_params)
    self.type_split = False
    self.model = get_model(model_params, sampled).to(env.DEVICE)

class TestEnergyModelDPA1(unittest.TestCase, TestRot):
  def setUp(self):
    model_params = model_dpa1
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)

# class TestEnergyModelDPA2(unittest.TestCase, TestRot):
#   def setUp(self):
#     model_params = model_dpa2
#     sampled = make_sample(model_params)
#     self.type_split = True
#     self.model = get_model(model_params, sampled).to(env.DEVICE)

class TestEnergyModelDPAUni(unittest.TestCase, TestRot):
  def setUp(self):
    model_params = model_dpau
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)

class TestForceModelDPAUni(unittest.TestCase, TestRot):
  def setUp(self):
    model_params = model_dpau
    model_params["fitting_net"]["type"] = "direct_force_ener"
    sampled = make_sample(model_params)
    self.type_split = True
    self.test_virial = False
    self.model = get_model(model_params, sampled).to(env.DEVICE)

class TestEnergyModelHybrid(unittest.TestCase, TestRot):
  def setUp(self):
    model_params = model_hybrid
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)

class TestForceModelHybrid(unittest.TestCase, TestRot):
  def setUp(self):
    model_params = model_hybrid
    model_params["fitting_net"]["type"] = "direct_force_ener"
    sampled = make_sample(model_params)
    self.type_split = True
    self.test_virial = False
    self.model = get_model(model_params, sampled).to(env.DEVICE)


if __name__ == '__main__':
    unittest.main()
