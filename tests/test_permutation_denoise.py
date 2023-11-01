import torch, copy
import unittest
from deepmd_pt.utils import env
from deepmd_pt.model.model import get_model
from deepmd_pt.utils.dataloader import DpLoaderSet
from deepmd_pt.utils.stat import make_stat_input
from deepmd_pt.infer.deep_eval import eval_model

dtype = torch.float64

model_dpau_denoise = {
  "type_map": ["O", "H", "B", "MASKED_TOKEN"],
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
    "do_bn_mode": "uniform",
    "_comment": " that's all"
  },
}

model_dpa1_denoise = {
  "type_map": ["O", "H", "B","MASKED_TOKEN"],
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
}

model_hybrid_denoise = {
  "type_map": [
    "O",
    "H",
    "B",
    "MASKED_TOKEN"
  ],
  "descriptor": {
    "type": "hybrid",
    "hybrid_mode": "sequential",
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
        "resnet_dt": False,
        "axis_neuron": 16,
        "seed": 1,
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
        "temperature": 1.0,
        "_comment": " that's all"
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
        "add_type_ebd_to_seq": True,
        "smooth": True,
        "do_bn_mode": "uniform",
        "_comment": " that's all"
      },
    ]
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
  
class TestPermutationDenoise:
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
    updated_c0, logits0 = eval_model(self.model, coord.unsqueeze(0), cell.unsqueeze(0), atype, denoise=True)
    ret0 = {'updated_coord': updated_c0.squeeze(0), 'logits': logits0.squeeze(0)}
    updated_c1, logits1 = eval_model(self.model, coord[idx_perm].unsqueeze(0), cell.unsqueeze(0), atype[idx_perm], denoise=True)
    ret1 = {'updated_coord': updated_c1.squeeze(0), 'logits': logits1.squeeze(0)}
    prec = 1e-10
    torch.testing.assert_close(ret0['updated_coord'][idx_perm], ret1['updated_coord'], rtol=prec, atol=prec)
    torch.testing.assert_close(ret0['logits'][idx_perm], ret1['logits'], rtol=prec, atol=prec)

class TestDenoiseModelDPA1(unittest.TestCase, TestPermutationDenoise):
  def setUp(self):
    model_params = copy.deepcopy(model_dpa1_denoise)
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)

class TestDenoiseModelDPAUni(unittest.TestCase, TestPermutationDenoise):
  def setUp(self):
    model_params = copy.deepcopy(model_dpau_denoise)
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)

class TestDenoiseModelDPAUni2(unittest.TestCase, TestPermutationDenoise):
  def setUp(self):
    model_params = copy.deepcopy(model_dpau_denoise)
    model_params["descriptor"]["combine_grrg"] = True
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)


class TestDenoiseModelDPAUni3(unittest.TestCase, TestPermutationDenoise):
  def setUp(self):
    model_params = copy.deepcopy(model_dpau_denoise)
    model_params["descriptor"]["gather_g1"] = True
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)


class TestDenoiseModelHybrid(unittest.TestCase, TestPermutationDenoise):
  def setUp(self):
    model_params = copy.deepcopy(model_hybrid_denoise)
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)

if __name__ == '__main__':
  unittest.main()