import torch, copy
import unittest
import itertools
import numpy as np
from deepmd_pt.model.network.mlp import MLPLayer, dtype
try :
  from deepmd_utils.model_format import (
    NativeLayer,
    NativeNet,
  )
  support_native_net = True
except ImportError:
  support_native_net = False


@unittest.skipIf(not support_native_net, "NativeLayer not supported")
class TestMLPLayer(unittest.TestCase):
  def setUp(self):
    self.test_cases =  itertools.product(
        [(5, 5), (5, 10), (5, 8), (8, 5)],      # inp, out
        [True, False],                          # bias
        [True, False],                          # use time step
        ["tanh", "none"],                       # activation
        [True, False],                          # resnet
    )

  def test_match_native_layer(
      self,
  ):
    for (ninp, nout), bias, ut, ac, resnet in self.test_cases:
      xx = torch.arange(ninp, dtype=dtype)
      ml = MLPLayer(ninp, nout, bias, ut, ac, resnet)
      data = ml.serialize()
      nl = NativeLayer.deserialize(data)
      np.testing.assert_allclose(
        ml.forward(xx).detach().numpy(), nl.call(xx.detach().numpy()),
        err_msg=f"(i={ninp}, o={nout}) bias={bias} use_dt={ut} act={ac} resnet={resnet}"
      )
      model = torch.jit.script(ml)

  def test_jit(self):
    for (ninp, nout), bias, ut, ac, resnet in self.test_cases:
      ml = MLPLayer(ninp, nout, bias, ut, ac, resnet)
      model = torch.jit.script(ml)
