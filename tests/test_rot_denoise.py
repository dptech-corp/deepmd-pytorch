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

class TestRotDenoise():
  def test(
      self,
  ):
    prec = 1e-10
    natoms = 5
    cell = 10. * torch.eye(3, dtype=dtype).to(env.DEVICE)
    coord = 2*torch.rand([natoms, 3], dtype=dtype).to(env.DEVICE)
    shift = torch.tensor([4, 4, 4], dtype=dtype).to(env.DEVICE)
    atype = torch.IntTensor([0, 0, 0, 1, 1]).to(env.DEVICE)
    from scipy.stats import special_ortho_group
    rmat = torch.tensor(special_ortho_group.rvs(3), dtype=dtype).to(env.DEVICE)

    # rotate only coord and shift to the center of cell
    coord_rot = torch.matmul(coord, rmat)
    update_c0, logits0 = eval_model(self.model, (coord + shift).unsqueeze(0), cell.unsqueeze(0), atype, denoise=True)
    update_c0 = update_c0 - (coord + shift).unsqueeze(0)
    ret0 = {'updated_coord': update_c0.squeeze(0), 'logits': logits0.squeeze(0)}
    update_c1, logits1 = eval_model(self.model, (coord_rot + shift).unsqueeze(0), cell.unsqueeze(0), atype, denoise = True)
    update_c1 = update_c1 - (coord_rot + shift).unsqueeze(0)
    ret1 = {'updated_coord': update_c1.squeeze(0), 'logits': logits1.squeeze(0)}
    torch.testing.assert_close(torch.matmul(ret0['updated_coord'], rmat), ret1['updated_coord'], rtol=prec, atol=prec)
    torch.testing.assert_close(ret0['logits'], ret1['logits'], rtol=prec, atol=prec)
    
    '''
    # rotate coord and cell
    torch.manual_seed(0)
    cell = torch.rand([3, 3], dtype=dtype).to(env.DEVICE)
    cell = (cell + cell.T) + 5. * torch.eye(3).to(env.DEVICE)
    coord = torch.rand([natoms, 3], dtype=dtype).to(env.DEVICE)
    coord = torch.matmul(coord, cell)
    atype = torch.IntTensor([0, 0, 0, 1, 1]).to(env.DEVICE)
    coord_rot = torch.matmul(coord, rmat)
    cell_rot = torch.matmul(cell, rmat)
    e0, f0, v0 = eval_model(self.model, coord.unsqueeze(0), cell.unsqueeze(0), atype)
    ret0 = {'energy': e0.squeeze(0), 'force': f0.squeeze(0), 'virial': v0.squeeze(0)}
    e1, f1, v1 = eval_model(self.model, coord_rot.unsqueeze(0), cell_rot.unsqueeze(0), atype)
    ret1 = {'energy': e1.squeeze(0), 'force': f1.squeeze(0), 'virial': v1.squeeze(0)}
    torch.testing.assert_close(ret0['energy'], ret1['energy'], rtol=prec, atol=prec)
    torch.testing.assert_close(torch.matmul(ret0['force'], rmat), ret1['force'], rtol=prec, atol=prec)
    if not hasattr(self, "test_virial") or self.test_virial:
      torch.testing.assert_close(
        torch.matmul(rmat.T, torch.matmul(ret0['virial'], rmat)), 
        ret1['virial'], rtol=prec, atol=prec)  
    ''' 

class TestDenoiseModelDPA1(unittest.TestCase, TestRotDenoise):
  def setUp(self):
    model_params = copy.deepcopy(model_dpa1_denoise)
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)

class TestDenoiseModelDPAUni(unittest.TestCase, TestRotDenoise):
  def setUp(self):
    model_params = copy.deepcopy(model_dpau_denoise)
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)

class TestEnergyModelHybrid(unittest.TestCase, TestRotDenoise):
  def setUp(self):
    model_params = copy.deepcopy(model_hybrid_denoise)
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)

if __name__ == '__main__':
    unittest.main()