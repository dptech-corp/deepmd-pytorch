import torch, copy
import unittest
from deepmd_pt.utils.preprocess import (
  Region3D, make_env_mat,
)
from deepmd_pt.utils import env
from deepmd_pt.model.model import get_model
from deepmd_pt.utils.dataloader import DpLoaderSet
from deepmd_pt.utils.stat import make_stat_input
from .test_permutation_denoise import (
  make_sample,
  model_dpa1_denoise,
  model_dpau_denoise,
  model_hybrid_denoise,
)
from deepmd_pt.infer.deep_eval import eval_model

dtype = torch.float64

class TestTransDenoise:
  def test(
      self,
  ):
    natoms = 5
    cell = torch.rand([3, 3], dtype=dtype).to(env.DEVICE)
    cell = (cell + cell.T) + 5. * torch.eye(3).to(env.DEVICE)
    coord = torch.rand([natoms, 3], dtype=dtype).to(env.DEVICE)
    coord = torch.matmul(coord, cell)
    atype = torch.IntTensor([0, 0, 0, 1, 1]).to(env.DEVICE)
    shift = (torch.rand([3], dtype=dtype) - .5).to(env.DEVICE) * 2.
    coord_s = torch.matmul(
      torch.remainder(
        torch.matmul(coord + shift, torch.linalg.inv(cell)), 1.), cell)
    updated_c0, logits0 = eval_model(self.model, coord.unsqueeze(0), cell.unsqueeze(0), atype, denoise=True)
    updated_c0 = updated_c0 - coord.unsqueeze(0)
    ret0 = {'updated_coord': updated_c0.squeeze(0), 'logits': logits0.squeeze(0)}
    updated_c1, logits1 = eval_model(self.model, coord_s.unsqueeze(0), cell.unsqueeze(0), atype, denoise=True)
    updated_c1 = updated_c1 - coord_s.unsqueeze(0)
    ret1 = {'updated_coord': updated_c1.squeeze(0), 'logits': logits1.squeeze(0)}
    prec = 1e-10
    torch.testing.assert_close(ret0['updated_coord'], ret1['updated_coord'], rtol=prec, atol=prec)
    torch.testing.assert_close(ret0['logits'], ret1['logits'], rtol=prec, atol=prec)

class TestDenoiseModelDPA1(unittest.TestCase, TestTransDenoise):
  def setUp(self):
    model_params = copy.deepcopy(model_dpa1_denoise)
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)

class TestDenoiseModelDPAUni(unittest.TestCase, TestTransDenoise):
  def setUp(self):
    model_params = copy.deepcopy(model_dpau_denoise)
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)

class TestDenoiseModelHybrid(unittest.TestCase, TestTransDenoise):
  def setUp(self):
    model_params = copy.deepcopy(model_hybrid_denoise)
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)

if __name__ == '__main__':
  unittest.main()