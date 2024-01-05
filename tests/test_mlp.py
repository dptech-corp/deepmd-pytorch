import torch, copy
import unittest
import itertools
import numpy as np
try:
  from deepmd_pt.model.network.mlp import MLPLayer, MLP, dtype
  support_native_net = True
except ModuleNotFoundError:
  support_native_net = False

try :
  from deepmd_utils.model_format import (
    NativeLayer,
    NativeNet,
  )
  support_native_net = True
except ModuleNotFoundError:
  support_native_net = False
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
        [None, [4], [3,2]],                     # prefix shapes
    )

  def test_match_native_layer(
      self,
  ):
    for (ninp, nout), bias, ut, ac, resnet, ashp in self.test_cases:
      inp_shap = [ninp]
      if ashp is not None:
        inp_shap = ashp + inp_shap
      xx = torch.arange(np.prod(inp_shap), dtype=dtype).view(inp_shap)
      ml = MLPLayer(ninp, nout, bias, ut, ac, resnet)
      data = ml.serialize()
      nl = NativeLayer.deserialize(data)
      np.testing.assert_allclose(
        ml.forward(xx).detach().numpy(), nl.call(xx.detach().numpy()),
        err_msg=f"(i={ninp}, o={nout}) bias={bias} use_dt={ut} act={ac} resnet={resnet}"
      )

  def test_layer_jit(self):
    for (ninp, nout), bias, ut, ac, resnet, _ in self.test_cases:
      ml = MLPLayer(ninp, nout, bias, ut, ac, resnet)
      model = torch.jit.script(ml)


@unittest.skipIf(not support_native_net, "NativeLayer not supported")
class TestMLP(unittest.TestCase):
  def setUp(self):
    self.test_cases =  itertools.product(
        [[2, 2, 4, 8], [1, 3, 3] ],             # inp and hiddens
        [True, False],                          # bias
        [True, False],                          # use time step
        ["tanh", "none"],                       # activation
        [True, False],                          # resnet
        [None, [4], [3,2]],                     # prefix shapes
    )

  def test_match_native_net(
      self,
  ):
    for ndims, bias, ut, ac, resnet, ashp in self.test_cases:
      inp_shap = [ndims[0]]
      if ashp is not None:
        inp_shap = ashp + inp_shap
      xx = torch.arange(np.prod(inp_shap), dtype=dtype).view(inp_shap)

      layers = []
      for ii in range(1, len(ndims)):        
        ml = layers.append(
          MLPLayer(ndims[ii-1], ndims[ii], bias, ut, ac, resnet))
      ml = MLP(ml)
      
      data = ml.serialize()
      nl = NativeNet.deserialize(data)
      np.testing.assert_allclose(
        ml.forward(xx).detach().numpy(), nl.call(xx.detach().numpy()),
        err_msg=f"net={ndims} bias={bias} use_dt={ut} act={ac} resnet={resnet}"
      )

  def test_net_jit(self):
    for ndims, bias, ut, ac, resnet, _ in self.test_cases:
      layers = []
      for ii in range(1, len(ndims)):        
        ml = layers.append(
          MLPLayer(ndims[ii-1], ndims[ii], bias, ut, ac, resnet))
      ml = MLP(ml)
      model = torch.jit.script(ml)
