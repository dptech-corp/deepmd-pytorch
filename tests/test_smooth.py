import torch, copy
import unittest
from deepmd_pt.utils.preprocess import (
  Region3D, make_env_mat,
)
from deepmd_pt.utils import env
from deepmd_pt.model.model import EnergyModelSeA, EnergyModelDPA1, EnergyModelDPA2, EnergyModelDPAUni, ForceModelDPAUni, EnergyModelHybrid, ForceModelHybrid
from deepmd_pt.utils.dataloader import DpLoaderSet
from deepmd_pt.utils.stat import make_stat_input
from .test_permutation import infer_model, make_sample

dtype = torch.float64

model_se_e2_a = {
  "type_map": ["O","H","B"],
  "descriptor": {
    "type": "se_e2_a",
    "sel": [46, 92, 4],
    "rcut_smth": 3.50,
    "rcut": 4.00,
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
  "type_map": ["O","H", "B"],
  "descriptor": {
    "type": "se_uni",
    "sel": [11],
    "rcut_smth": 3.5,
    "rcut": 4.0,
    "nlayers": 3,
    "update_g1_has_conv": True,
    "update_g1_has_drrd": True,
    "update_g1_has_grrg": True,
    "update_g1_has_attn": True,
    "update_g2_has_g1g1": True,
    "update_g2_has_attn": True,
    "update_h2": True,
    "gather_g1": True,
    "combine_grrg": False,
    "attn2_has_gate" : True,
    "smooth" : True,
    "_comment": " that's all"
  },
  "fitting_net": {
    "neuron": [24, 24, 24],
    "resnet_dt": True,
    "seed": 1,
  },
}

model_dpa1 = {
  "type_map": ["O","H","B"],
  "descriptor": {
    "type": "se_atten",
    "sel": 40,
    "rcut_smth": 3.5,
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
    "rcut_smth": 3.5,
    "rcut": 4.0,
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

class TestSmooth():
  def test(
      self,
  ):
    natoms = 3
    cell = 10. * torch.eye(3, dtype=dtype)
    coord = torch.rand([natoms, 3], dtype=dtype)
    coord = torch.matmul(coord, cell)
    coord = torch.tensor([0., 0., 0.,
                          4., 0., 0.,
                          0., 4., 0.,], dtype=dtype).view([natoms, 3])
    # displacement of atoms
    epsilon = .5e-4
    # required prec. relative prec is not checked.
    rprec = 0
    aprec = 1e-6

    coord0 = torch.clone(coord)
    coord1 = torch.clone(coord)
    coord1[1][0] -= epsilon
    coord2 = torch.clone(coord)
    coord2[2][1] -= epsilon
    coord3 = torch.clone(coord)
    coord3[1][0] -= epsilon
    coord3[2][1] -= epsilon
    atype = torch.IntTensor([0, 1, 2])

    # print(coord0 , coord1)
    ret0 = infer_model(self.model, coord0, cell, atype, type_split=self.type_split)
    ret1 = infer_model(self.model, coord1, cell, atype, type_split=self.type_split)
    ret2 = infer_model(self.model, coord2, cell, atype, type_split=self.type_split)
    ret3 = infer_model(self.model, coord3, cell, atype, type_split=self.type_split)
    # print(ret0['energy']- ret1['energy'])
    def compare(ret0, ret1):
      torch.testing.assert_close(1.+ret0['energy'], 1.+ret1['energy'], rtol=rprec, atol=aprec)
      torch.testing.assert_close(1.+ret0['force'], 1.+ret1['force'], rtol=rprec, atol=aprec)
      if not hasattr(self, "test_virial") or self.test_virial:
        torch.testing.assert_close(1.+ret0['virial'], 1.+ret1['virial'], rtol=rprec, atol=aprec)
    compare(ret0, ret1)
    compare(ret1, ret2)
    compare(ret0, ret3)


class TestEnergyModelSeA(unittest.TestCase, TestSmooth):
  def setUp(self):
    model_params = model_se_e2_a
    sampled = make_sample(model_params)
    self.type_split = False
    self.model = EnergyModelSeA(model_params, sampled).to(env.DEVICE)

class TestEnergyModelDPA1(unittest.TestCase, TestSmooth):
  def setUp(self):
    model_params = model_dpa1
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = EnergyModelDPA1(model_params, sampled).to(env.DEVICE)

@unittest.skip("not sure why the output is nan")
class TestEnergyModelDPA2(unittest.TestCase, TestSmooth):
  def setUp(self):
    model_params = model_dpa2
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = EnergyModelDPA2(model_params, sampled).to(env.DEVICE)

class TestEnergyModelDPAUni(unittest.TestCase, TestSmooth):
  def setUp(self):
    model_params = model_dpau
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = EnergyModelDPAUni(model_params, sampled).to(env.DEVICE)

class TestEnergyModelDPAUni2(unittest.TestCase, TestSmooth):
  def setUp(self):
    model_params = model_dpau
    model_params["fitting_net"]["type"] = "ener"
    model_params["descriptor"]["combine_grrg"] = True
    sampled = make_sample(model_params)
    self.type_split = True
    self.test_virial = False
    self.model = EnergyModelDPAUni(model_params, sampled).to(env.DEVICE)

class TestEnergyModelDPAUni3(unittest.TestCase, TestSmooth):
  def setUp(self):
    model_params = model_dpau
    model_params["fitting_net"]["type"] = "ener"
    model_params["descriptor"]["gather_g1"] = True
    sampled = make_sample(model_params)
    self.type_split = True
    self.test_virial = False
    self.model = EnergyModelDPAUni(model_params, sampled).to(env.DEVICE)


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
