import torch, copy
import unittest
from deepmd_pt.utils import env
from deepmd_pt.model.model import get_model
from deepmd_pt.utils.dataloader import DpLoaderSet
from deepmd_pt.utils.stat import make_stat_input
from deepmd_pt.infer.deep_eval import eval_model

dtype = torch.float64

model_se_e2_a = {
  "type_map": ["O", "H", "B"],
  "descriptor": {
    "type": "se_e2_a",
    "sel": [46, 92, 4],
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
  "type_map": ["O", "H", "B"],
  "descriptor": {
    "type": "se_uni",
    "sel": 40,
    "rcut_smth": 0.5,
    "rcut": 4.0,
    "nlayers": 2,
    "g1_dim": 10,
    "g2_dim": 5,
    "attn2_hidden": 10,
    "attn2_nhead": 2,
    "attn1_hidden": 10,
    "attn1_nhead": 2,
    "axis_dim": 4,
    "update_h2": False,
    "update_g1_has_conv": True,
    "update_g1_has_drrd": True,
    "update_g1_has_grrg": True,
    "update_g1_has_attn": True,
    "update_g2_has_g1g1": True,
    "update_g2_has_attn": True,
    "attn2_has_gate": True,
    "smooth": True,
    "do_bn_mode": "uniform"
  },
  "fitting_net": {
    "neuron": [24, 24],
    "resnet_dt": True,
    "seed": 1,
  },
}

model_dpa1 = {
  "type_map": ["O", "H", "B"],
  "descriptor": {
    "type": "se_atten",
    "sel": 40,
    "rcut_smth": 0.5,
    "rcut": 4.0,
    "neuron": [25, 50, 100],
    "axis_neuron": 16,
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
    "set_davg_zero": True
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
    "neuron": [25, 50, 100],
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
    "temperature": 1.0
  },
  "backbone": {
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
    "seed": 1
  },
}

model_hybrid = {
  "type_map": [
    "O",
    "H",
    "B"
  ],
  "descriptor": {
    "type": "hybrid",
    "list": [
      {
        "type": "se_atten",
        "sel": 120,
        "rcut_smth": 0.5,
        "rcut": 6.0,
        "neuron": [
          25,
          50,
          100
        ],
        "axis_neuron": 16,
        "attn": 128,
        "attn_layer": 0,
        "attn_dotr": True,
        "attn_mask": False,
        "post_ln": True,
        "ffn": False,
        "ffn_embed_dim": 1024,
        "activation": "tanh",
        "scaling_factor": 1.0,
        "head_num": 1,
        "normalize": True,
        "temperature": 1.0
      },
      {
        "type": "se_uni",
        "sel": 40,
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "nlayers": 2,
        "g1_dim": 10,
        "g2_dim": 5,
        "attn2_hidden": 10,
        "attn2_nhead": 2,
        "attn1_hidden": 10,
        "attn1_nhead": 2,
        "axis_dim": 4,
        "update_h2": False,
        "update_g1_has_conv": True,
        "update_g1_has_drrd": True,
        "update_g1_has_grrg": True,
        "update_g1_has_attn": True,
        "update_g2_has_g1g1": True,
        "update_g2_has_attn": True,
        "attn2_has_gate": True,
        "add_type_ebd_to_seq": False,
        "smooth": True,
        "do_bn_mode": "uniform"
      },
    ]
  },
  "fitting_net": {
    "neuron": [
      240,
      240,
      240
    ],
    "resnet_dt": True,
    "seed": 1,
    "_comment": " that's all"
  },
  "_comment": " that's all"
}


def make_sample(model_params):
  training_systems = ["tests/water/data/data_0", ]
  data_stat_nbatch = model_params.get('data_stat_nbatch', 10)
  train_data = DpLoaderSet(
    training_systems, batch_size=4, model_params=model_params.copy(),
  )
  sampled = make_stat_input(
    train_data.systems, train_data.dataloaders, data_stat_nbatch)
  return sampled


class TestPermutation:
  def test(
      self,
  ):
    natoms = 5
    cell = torch.rand([3, 3], dtype=dtype).to(env.DEVICE)
    cell = (cell + cell.T) + 5. * torch.eye(3).to(env.DEVICE)
    coord = torch.rand([natoms, 3], dtype=dtype).to(env.DEVICE)
    coord = torch.matmul(coord, cell)
    atype = torch.IntTensor([0, 0, 0, 1, 1]).to(env.DEVICE)
    idx_perm = [1, 0, 4, 3, 2]
    e0, f0, v0 = eval_model(self.model, coord.unsqueeze(0), cell.unsqueeze(0), atype)
    ret0 = {'energy': e0.squeeze(0), 'force': f0.squeeze(0), 'virial': v0.squeeze(0)}
    e1, f1, v1 = eval_model(self.model, coord[idx_perm].unsqueeze(0), cell.unsqueeze(0), atype[idx_perm])
    ret1 = {'energy': e1.squeeze(0), 'force': f1.squeeze(0), 'virial': v1.squeeze(0)}
    prec = 1e-10
    torch.testing.assert_close(ret0['energy'], ret1['energy'], rtol=prec, atol=prec)
    torch.testing.assert_close(ret0['force'][idx_perm], ret1['force'], rtol=prec, atol=prec)
    if not hasattr(self, "test_virial") or self.test_virial:
      torch.testing.assert_close(ret0['virial'], ret1['virial'], rtol=prec, atol=prec)


class TestEnergyModelSeA(unittest.TestCase, TestPermutation):
  def setUp(self):
    model_params = copy.deepcopy(model_se_e2_a)
    sampled = make_sample(model_params)
    self.type_split = False
    self.model = get_model(model_params, sampled).to(env.DEVICE)


class TestEnergyModelDPA1(unittest.TestCase, TestPermutation):
  def setUp(self):
    model_params = copy.deepcopy(model_dpa1)
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)


# class TestEnergyModelDPA2(unittest.TestCase, TestPermutation):
#   def setUp(self):
#     model_params = model_dpa2
#     sampled = make_sample(model_params)
#     self.type_split = True
#     self.model = get_model(model_params, sampled).to(env.DEVICE)

class TestEnergyModelDPAUni(unittest.TestCase, TestPermutation):
  def setUp(self):
    model_params = copy.deepcopy(model_dpau)
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)


class TestForceModelDPAUni(unittest.TestCase, TestPermutation):
  def setUp(self):
    model_params = copy.deepcopy(model_dpau)
    model_params["fitting_net"]["type"] = "direct_force_ener"
    sampled = make_sample(model_params)
    self.type_split = True
    self.test_virial = False
    self.model = get_model(model_params, sampled).to(env.DEVICE)


class TestEnergyModelDPAUni2(unittest.TestCase, TestPermutation):
  def setUp(self):
    model_params = copy.deepcopy(model_dpau)
    model_params["fitting_net"]["type"] = "direct_force_ener"
    model_params["descriptor"]["combine_grrg"] = True
    sampled = make_sample(model_params)
    self.type_split = True
    self.test_virial = False
    self.model = get_model(model_params, sampled).to(env.DEVICE)


class TestEnergyModelHybrid(unittest.TestCase, TestPermutation):
  def setUp(self):
    model_params = copy.deepcopy(model_hybrid)
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)


class TestForceModelHybrid(unittest.TestCase, TestPermutation):
  def setUp(self):
    model_params = copy.deepcopy(model_hybrid)
    model_params["fitting_net"]["type"] = "direct_force_ener"
    sampled = make_sample(model_params)
    self.type_split = True
    self.test_virial = False
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
