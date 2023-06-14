import torch, copy
import unittest
from deepmd_pt.utils.preprocess import (
  Region3D, make_env_mat,
)
from deepmd_pt.utils import env
from deepmd_pt.model.model import EnergyModelSeA, EnergyModelDPA1, EnergyModelDPA2, EnergyModelDPAUni, ForceModelDPAUni
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
    "update_g1_has_conv": True,
    "update_g1_has_drrd": True,
    "update_g1_has_grrg": True,
    "update_g1_has_attn": True,
    "update_g2_has_g1g1": True,
    "update_g2_has_attn": True,
    "update_h2": True,
    "gather_g1": True,
    "combine_grrg": False,
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

class TestUnusedParamsDPAUni(unittest.TestCase):
  def test_unused(self):    
    import itertools
    for cmbg2, conv, drrd, grrg, attn1, g1g1, attn2, h2 in \
        itertools.product(
          [True,False], [True,False], [True,False], [True,False], 
          [True,False], [True,False], [True,False], [True,False],
        ):
      if (not drrd) and (not grrg) and h2:
        # skip the case h2 is not envolved
        continue
      if (not grrg) and (not conv):
        # skip the case g2 is not envolved
        continue
      model_dpau["descriptor"]["combine_grrg"] = cmbg2
      model_dpau["descriptor"]["update_g1_has_conv"] = conv
      model_dpau["descriptor"]["update_g1_has_drrd"] = drrd
      model_dpau["descriptor"]["update_g1_has_grrg"] = grrg
      model_dpau["descriptor"]["update_g1_has_attn"] = attn1
      model_dpau["descriptor"]["update_g2_has_g1g1"] = g1g1
      model_dpau["descriptor"]["update_g2_has_attn"] = attn2
      model_dpau["descriptor"]["update_h2"] = h2
      self._test_unused(model_dpau)

  def _test_unused(self, model_params):
    sampled = make_sample(model_params)
    self.model = EnergyModelDPAUni(model_params, sampled).to(env.DEVICE)

    natoms = 5
    cell = torch.rand([3, 3], dtype=dtype)
    cell = (cell + cell.T) + 5. * torch.eye(3)
    coord = torch.rand([natoms, 3], dtype=dtype)
    coord = torch.matmul(coord, cell)
    atype = torch.IntTensor([0, 0, 0, 1, 1])      
    idx_perm = [1, 0, 4, 3, 2]    
    ret0 = infer_model(self.model, coord, cell, atype, type_split=True)
    
    # use computation graph to find all contributing tensors
    def get_contributing_params(y, top_level=True):
        nf = y.grad_fn.next_functions if top_level else y.next_functions
        for f, _ in nf:
            try:
                yield f.variable
            except AttributeError:
                pass  # node has no tensor
            if f is not None:
                yield from get_contributing_params(f, top_level=False)

    contributing_parameters = set(get_contributing_params(ret0['energy']))
    all_parameters = set(self.model.parameters())
    non_contributing = all_parameters - contributing_parameters
    for ii in non_contributing:
      print(ii.shape)
    self.assertEqual(len(non_contributing), 0)


if __name__ == '__main__':
    unittest.main()
