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

class TestSmooth:
  def test(
      self,
  ):
    # displacement of atoms
    epsilon = 1e-5 if self.epsilon is None else self.epsilon
    # required prec. relative prec is not checked.
    rprec = 0
    aprec = 1e-5 if self.aprec is None else self.aprec

    natoms = 10
    cell = 8.6 * torch.eye(3, dtype=dtype).to(env.DEVICE)
    atype = torch.randint(0, 3, [natoms])
    coord0 = torch.tensor(
      [
        0., 0., 0.,
        4.-.5*epsilon, 0., 0.,
        0., 4.-.5*epsilon, 0., 
       ], 
      dtype=dtype).view([-1, 3]).to(env.DEVICE)
    coord1 = torch.rand([natoms-coord0.shape[0], 3], dtype=dtype).to(env.DEVICE)
    coord1 = torch.matmul(coord1, cell)
    coord = torch.concat([coord0, coord1], dim=0)

    coord0 = torch.clone(coord)
    coord1 = torch.clone(coord)
    coord1[1][0] += epsilon
    coord2 = torch.clone(coord)
    coord2[2][1] += epsilon
    coord3 = torch.clone(coord)
    coord3[1][0] += epsilon
    coord3[2][1] += epsilon

    update_c0, logits0 = eval_model(self.model, coord0.unsqueeze(0), cell.unsqueeze(0), atype, denoise = True)
    ret0 = {'updated_coord': update_c0.squeeze(0), 'logits': logits0.squeeze(0)}
    update_c1, logits1 = eval_model(self.model, coord1.unsqueeze(0), cell.unsqueeze(0), atype, denoise = True)
    ret1 = {'updated_coord': update_c1.squeeze(0), 'logits': logits1.squeeze(0)}
    update_c2, logits2 = eval_model(self.model, coord2.unsqueeze(0), cell.unsqueeze(0), atype, denoise = True)
    ret2 = {'updated_coord': update_c2.squeeze(0), 'logits': logits2.squeeze(0)}
    update_c3, logits3 = eval_model(self.model, coord3.unsqueeze(0), cell.unsqueeze(0), atype, denoise = True)
    ret3 = {'updated_coord': update_c3.squeeze(0), 'logits': logits3.squeeze(0)}

    def compare(ret0, ret1):
      torch.testing.assert_close(ret0['updated_coord'], ret1['updated_coord'], rtol=rprec, atol=aprec)
      torch.testing.assert_close(ret0['logits'], ret1['logits'], rtol=rprec, atol=aprec)

    compare(ret0, ret1)
    compare(ret1, ret2)
    compare(ret0, ret3)

@unittest.skip("not smooth at the moment")
class TestDenoiseModelHybrid(unittest.TestCase, TestSmooth):
  def setUp(self):
    model_params = copy.deepcopy(model_hybrid_denoise)
    sampled = make_sample(model_params)
    self.type_split = True
    self.model = get_model(model_params, sampled).to(env.DEVICE)
    self.epsilon, self.aprec = None, None