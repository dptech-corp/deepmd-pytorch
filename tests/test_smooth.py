import torch, copy
import unittest
from deepmd_pt.utils import env
from deepmd_pt.model.model import get_model
from deepmd_pt.utils.dataloader import DpLoaderSet
from deepmd_pt.utils.stat import make_stat_input
from .test_permutation import (
  make_sample,
  model_se_e2_a,
  model_dpa1,
  model_dpa2,
  model_dpau,
  model_hybrid,
)
from deepmd_pt.infer.deep_eval import eval_model

dtype = torch.float64


class TestSmooth:
  def test(
      self,
  ):
    natoms = 3
    cell = 10. * torch.eye(3, dtype=dtype).to(env.DEVICE)
    coord = torch.rand([natoms, 3], dtype=dtype).to(env.DEVICE)
    coord = torch.matmul(coord, cell)
    coord = torch.tensor([0., 0., 0.,
                          4., 0., 0.,
                          0., 4., 0., ], dtype=dtype).view([natoms, 3]).to(env.DEVICE)
    # displacement of atoms
    epsilon = .5e-4
    # required prec. relative prec is not checked.
    rprec = 0
    aprec = 1e-5

    coord0 = torch.clone(coord)
    coord1 = torch.clone(coord)
    coord1[1][0] -= epsilon
    coord2 = torch.clone(coord)
    coord2[2][1] -= epsilon
    coord3 = torch.clone(coord)
    coord3[1][0] -= epsilon
    coord3[2][1] -= epsilon
    atype = torch.IntTensor([0, 1, 2]).to(env.DEVICE)

    # print(coord0 , coord1)
    e0, f0, v0 = eval_model(self.model, coord0.unsqueeze(0), cell.unsqueeze(0), atype)
    ret0 = {'energy': e0.squeeze(0), 'force': f0.squeeze(0), 'virial': v0.squeeze(0)}
    e1, f1, v1 = eval_model(self.model, coord1.unsqueeze(0), cell.unsqueeze(0), atype)
    ret1 = {'energy': e1.squeeze(0), 'force': f1.squeeze(0), 'virial': v1.squeeze(0)}
    e2, f2, v2 = eval_model(self.model, coord2.unsqueeze(0), cell.unsqueeze(0), atype)
    ret2 = {'energy': e2.squeeze(0), 'force': f2.squeeze(0), 'virial': v2.squeeze(0)}
    e3, f3, v3 = eval_model(self.model, coord3.unsqueeze(0), cell.unsqueeze(0), atype)
    ret3 = {'energy': e3.squeeze(0), 'force': f3.squeeze(0), 'virial': v3.squeeze(0)}

    # print(ret0['energy']- ret1['energy'])
    def compare(ret0, ret1):
      torch.testing.assert_close(1. + ret0['energy'], 1. + ret1['energy'], rtol=rprec, atol=aprec)
      torch.testing.assert_close(1. + ret0['force'], 1. + ret1['force'], rtol=rprec, atol=aprec)
      if not hasattr(self, "test_virial") or self.test_virial:
        torch.testing.assert_close(1. + ret0['virial'], 1. + ret1['virial'], rtol=rprec, atol=aprec)

    compare(ret0, ret1)
    compare(ret1, ret2)
    compare(ret0, ret3)


class TestEnergyModelSeA(unittest.TestCase, TestSmooth):
  def setUp(self):
    model_params = copy.deepcopy(model_se_e2_a)
    sampled = make_sample(model_params)
    self.type_split = False
    self.model = get_model(model_params, sampled).to(env.DEVICE)


class TestEnergyModelDPA1(unittest.TestCase, TestSmooth):
  def setUp(self):
    model_params = copy.deepcopy(model_dpa1)
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)


class TestEnergyModelDPAUni(unittest.TestCase, TestSmooth):
  def setUp(self):
    model_params = copy.deepcopy(model_dpau)
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)


class TestEnergyModelDPAUni2(unittest.TestCase, TestSmooth):
  def setUp(self):
    model_params = copy.deepcopy(model_dpau)
    model_params["fitting_net"]["type"] = "ener"
    model_params["descriptor"]["combine_grrg"] = True
    sampled = make_sample(model_params)
    self.type_split = True
    self.test_virial = False
    self.model = get_model(model_params, sampled).to(env.DEVICE)


class TestEnergyModelDPAUni3(unittest.TestCase, TestSmooth):
  def setUp(self):
    model_params = copy.deepcopy(model_dpau)
    model_params["fitting_net"]["type"] = "ener"
    model_params["descriptor"]["gather_g1"] = True
    sampled = make_sample(model_params)
    self.type_split = True
    self.test_virial = False
    self.model = get_model(model_params, sampled).to(env.DEVICE)


class TestEnergyModelHybrid(unittest.TestCase, TestSmooth):
  def setUp(self):
    model_params = copy.deepcopy(model_hybrid)
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)


# class TestEnergyFoo(unittest.TestCase):
#   def test(self):
#     model_params = model_dpau
#     sampled = make_sample(model_params)
#     self.model = EnergyModelDPAUni(model_params, sampled).to(env.DEVICE)

#     natoms = 5
#     cell = torch.rand([3, 3], dtype=dtype)
#     cell = (cell + cell.T) + 5. * torch.eye(3)
#     coord = torch.rand([natoms, 3], dtype=dtype)
#     coord = torch.matmul(coord, cell)
#     atype = torch.IntTensor([0, 0, 0, 1, 1])      
#     idx_perm = [1, 0, 4, 3, 2]    
#     ret0 = infer_model(self.model, coord, cell, atype, type_split=True)


if __name__ == '__main__':
  unittest.main()
