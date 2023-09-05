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
    make_sample,
    model_se_e2_a,
    model_dpa1,
    model_dpa2,
    model_dpau,
    model_hybrid,
)
from deepmd_pt.infer.deep_eval import eval_model

dtype = torch.float64


class TestTrans:
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
        e0, f0, v0 = eval_model(self.model, coord.unsqueeze(0), cell.unsqueeze(0), atype)
        ret0 = {'energy': e0.squeeze(0), 'force': f0.squeeze(0), 'virial': v0.squeeze(0)}
        e1, f1, v1 = eval_model(self.model, coord_s.unsqueeze(0), cell.unsqueeze(0), atype)
        ret1 = {'energy': e1.squeeze(0), 'force': f1.squeeze(0), 'virial': v1.squeeze(0)}
        prec = 1e-10
        torch.testing.assert_close(ret0['energy'], ret1['energy'], rtol=prec, atol=prec)
        torch.testing.assert_close(ret0['force'], ret1['force'], rtol=prec, atol=prec)
        if not hasattr(self, "test_virial") or self.test_virial:
            torch.testing.assert_close(ret0['virial'], ret1['virial'], rtol=prec, atol=prec)


class TestEnergyModelSeA(unittest.TestCase, TestTrans):
    def setUp(self):
        model_params = model_se_e2_a
        sampled = make_sample(model_params)
        self.type_split = False
        self.model = get_model(model_params, sampled).to(env.DEVICE)


class TestEnergyModelDPA1(unittest.TestCase, TestTrans):
    def setUp(self):
        model_params = model_dpa1
        sampled = make_sample(model_params)
        self.type_split = True
        self.model = get_model(model_params, sampled).to(env.DEVICE)


# class TestEnergyModelDPA2(unittest.TestCase, TestTrans):
#   def setUp(self):
#     model_params = model_dpa2
#     sampled = make_sample(model_params)
#     self.type_split = True
#     self.model = get_model(model_params, sampled).to(env.DEVICE)

class TestEnergyModelDPAUni(unittest.TestCase, TestTrans):
    def setUp(self):
        model_params = model_dpau
        sampled = make_sample(model_params)
        self.type_split = True
        self.model = get_model(model_params, sampled).to(env.DEVICE)


class TestForceModelDPAUni(unittest.TestCase, TestTrans):
    def setUp(self):
        model_params = model_dpau
        model_params["fitting_net"]["type"] = "direct_force_ener"
        sampled = make_sample(model_params)
        self.type_split = True
        self.test_virial = False
        self.model = get_model(model_params, sampled).to(env.DEVICE)


class TestEnergyModelHybrid(unittest.TestCase, TestTrans):
    def setUp(self):
        model_params = model_hybrid
        sampled = make_sample(model_params)
        self.type_split = True
        self.model = get_model(model_params, sampled).to(env.DEVICE)


class TestForceModelHybrid(unittest.TestCase, TestTrans):
    def setUp(self):
        model_params = model_hybrid
        model_params["fitting_net"]["type"] = "direct_force_ener"
        sampled = make_sample(model_params)
        self.type_split = True
        self.test_virial = False
        self.model = get_model(model_params, sampled).to(env.DEVICE)


if __name__ == '__main__':
    unittest.main()
