import torch, copy
import unittest
from deepmd_pt.utils.preprocess import (
  Region3D, make_env_mat,
)
from deepmd_pt.utils import env
from deepmd_pt.model.model import EnergyModelSeA, EnergyModelDPA1, EnergyModelDPA2, EnergyModelDPAUni
from deepmd_pt.utils.dataloader import DpLoaderSet
from deepmd_pt.utils.stat import make_stat_input
from .test_permutation import infer_model, make_sample

dtype = torch.float64

model_se_e2_a = {
  "type_map": ["O","H"],
  "descriptor": {
    "type": "se_e2_a",
    "sel": [46, 92],
    "rcut_smth": 0.50,
    "rcut": 6.00,
    "neuron": [25, 50, 100],
    "resnet_dt": False,
    "axis_neuron": 16,
    "seed": 1,
  },
  "fitting_net": {
    "neuron": [24, 24, 24],
    "resnet_dt": True,
    "seed": 1,
  },
  "data_stat_nbatch": 20,
}


model_dpau = {
  "type_map": ["O","H"],
  "descriptor": {
    "type": "se_uni",
    "sel": [11],
    "rcut_smth": 0.5,
    "rcut": 4.0,    
    "_comment": " that's all"
  },
  "fitting_net": {
    "neuron": [24, 24, 24],
    "resnet_dt": True,
    "seed": 1,
  },
}

model_dpa1 = {
  "type_map": ["O","H"],
  "descriptor": {
    "type": "se_atten",
    "sel": 40,
    "rcut_smth": 0.5,
    "rcut": 4.0,
    "neuron": [25, 50, 100],
    "resnet_dt": False,
    "axis_neuron": 16,
    "seed": 1,
    "attn": 64,
    "attn_layer": 2,
    "attn_dotr": True,
    "attn_mask": False,
    "post_ln": True,
    "ffn": False,
    "ffn_embed_dim": 512,
    "activation": "tanh",
    "scaling_factor": 1.0,
    "head_num": 1,
    "normalize": False,
    "temperature": 1.0,
    "_comment": " that's all"
  },
  "fitting_net": {
    "neuron": [24, 24, 24],
    "resnet_dt": True,
    "seed": 1,
  },
}

model_dpa2 = {
  "type_map": ["O", "H", "MASKED_TOKEN"],
  "descriptor": {
    "type": "se_atten",
    "sel": 120,
    "rcut_smth": 0.5,
    "rcut": 6.0,
    "neuron": [25, 50, 100 ],
    "resnet_dt": False,
    "axis_neuron": 6,
    "seed": 1,
    "attn": 128,
    "attn_layer": 2,
    "attn_dotr": True,
    "attn_mask": False,
    "post_ln": True,
    "ffn": False,
    "ffn_embed_dim": 1024,
    "activation": "tanh",
    "scaling_factor": 1.0,
    "head_num": 1,
    "normalize": True,
    "temperature": 1.0,
    "_comment": " that's all"
  },
  "backbone":{
    "type": "evo-2b",
    "layer_num": 6,
    "attn_head": 8,
    "feature_dim": 128,
    "ffn_dim": 1024,
    "post_ln": False,
    "final_layer_norm": True,
    "final_head_layer_norm": False,
    "emb_layer_norm": False,
    "atomic_residual": True,
    "evo_residual": True,
    "activation_function": "gelu"
  },
  "fitting_net": {
    "neuron": [24, 24, 24],
    "resnet_dt": True,
    "seed": 1,
    "_comment": " that's all"
  },
}


class TestRot():
  def test(
      self,
  ):
    natoms = 5
    cell = 10. * torch.eye(3, dtype=dtype)
    coord = 2*torch.rand([natoms, 3], dtype=dtype) + torch.tensor([4, 4, 4], dtype=dtype)
    atype = torch.IntTensor([0, 0, 0, 1, 1])      
    from scipy.stats import special_ortho_group
    rmat = torch.tensor(special_ortho_group.rvs(3), dtype=dtype)
    coord_rot = torch.matmul(coord, rmat)
    ret0 = infer_model(self.model, coord, cell, atype, type_split=self.type_split)
    ret1 = infer_model(self.model, coord_rot, cell, atype, type_split=self.type_split)
    prec = 1e-10
    torch.testing.assert_close(ret0['energy'], ret1['energy'], rtol=prec, atol=prec)
    torch.testing.assert_close(torch.matmul(ret0['force'], rmat), ret1['force'], rtol=prec, atol=prec)
    torch.testing.assert_close(
      torch.matmul(rmat.T, torch.matmul(ret0['virial'], rmat)), 
      ret1['virial'], rtol=prec, atol=prec)

class TestEnergyModelSeA(unittest.TestCase, TestRot):
  def setUp(self):
    model_params = model_se_e2_a
    data = make_sample(model_params)
    self.type_split = False
    self.model = EnergyModelSeA(model_params, data, sampled = True).to(env.DEVICE)

class TestEnergyModelDPA1(unittest.TestCase, TestRot):
  def setUp(self):
    model_params = model_dpa1
    data = make_sample(model_params)
    self.type_split = True
    self.model = EnergyModelDPA1(model_params, data, sampled = True).to(env.DEVICE)

class TestEnergyModelDPA2(unittest.TestCase, TestRot):
  def setUp(self):
    model_params = model_dpa2
    data = make_sample(model_params)
    self.type_split = True
    self.model = EnergyModelDPA2(model_params, data, sampled = True).to(env.DEVICE)


class TestEnergyModelDPAUni(unittest.TestCase, TestRot):
  def setUp(self):
    model_params = model_dpau
    data = make_sample(model_params)
    self.type_split = True
    self.model = EnergyModelDPAUni(model_params, data, sampled = True).to(env.DEVICE)



if __name__ == '__main__':
    unittest.main()
