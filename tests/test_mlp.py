import torch, copy
import unittest
import itertools
import numpy as np
try:
  from deepmd_pt.model.network.mlp import MLPLayer, MLP, EmbeddingNet
  support_native_net = True
except ModuleNotFoundError:
  support_native_net = False

from deepmd_pt.utils.env import (
  PRECISION_DICT,
)

try :
  from deepmd_utils.model_format import (
    NativeLayer,
    NativeNet,
    EmbeddingNet as DPEmbeddingNet,
  )
  support_native_net = True
except ModuleNotFoundError:
  support_native_net = False
except ImportError:
  support_native_net = False

def get_tols(prec):
  if prec in ["single", "float32"]:
    rtol, atol=0., 1e-4
  elif prec in ["double", "float64"]:
    rtol, atol=0., 1e-12
  # elif prec in ["half", "float16"]:
  #   rtol, atol=1e-2, 0
  else:
    raise ValueError(f"unknown prec {prec}")
  return rtol, atol

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
        ["float32", "double"],                  # precision
    )

  def test_match_native_layer(
      self,
  ):
    for (ninp, nout), bias, ut, ac, resnet, ashp, prec in self.test_cases:
      # input
      inp_shap = [ninp]
      if ashp is not None:
        inp_shap = ashp + inp_shap      
      rtol, atol = get_tols(prec)
      dtype = PRECISION_DICT[prec]
      xx = torch.arange(np.prod(inp_shap), dtype=dtype).view(inp_shap)
      # def mlp layer
      ml = MLPLayer(ninp, nout, bias, ut, ac, resnet, precision=prec)
      # check consistency
      nl = NativeLayer.deserialize(ml.serialize())
      np.testing.assert_allclose(
        ml.forward(xx).detach().numpy(), nl.call(xx.detach().numpy()),
        rtol=rtol, atol=atol,
        err_msg=f"(i={ninp}, o={nout}) bias={bias} use_dt={ut} act={ac} resnet={resnet} prec={prec}"
      )
      # check self-consistency
      ml1 = MLPLayer.deserialize(ml.serialize())
      np.testing.assert_allclose(
        ml.forward(xx).detach().numpy(), ml1.forward(xx).detach().numpy(),
        rtol=rtol, atol=atol,
        err_msg=f"(i={ninp}, o={nout}) bias={bias} use_dt={ut} act={ac} resnet={resnet} prec={prec}"
      )

  def test_jit(self):
    for (ninp, nout), bias, ut, ac, resnet, _, prec in self.test_cases:
      ml = MLPLayer(ninp, nout, bias, ut, ac, resnet, precision=prec)
      model = torch.jit.script(ml)
      ml1 = MLPLayer.deserialize(ml.serialize())
      model = torch.jit.script(ml1)


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
        ["float32", "double"],                  # precision
    )

  def test_match_native_net(
      self,
  ):
    for ndims, bias, ut, ac, resnet, ashp, prec in self.test_cases:
      # input
      inp_shap = [ndims[0]]
      if ashp is not None:
        inp_shap = ashp + inp_shap
      rtol, atol = get_tols(prec)
      dtype = PRECISION_DICT[prec]
      xx = torch.arange(np.prod(inp_shap), dtype=dtype).view(inp_shap)
      # def MLP
      layers = []
      for ii in range(1, len(ndims)):        
        layers.append(
          MLPLayer(ndims[ii-1], ndims[ii], bias, ut, ac, resnet, precision=prec).serialize())
      ml = MLP(layers)
      # check consistency
      nl = NativeNet.deserialize(ml.serialize())
      np.testing.assert_allclose(
        ml.forward(xx).detach().numpy(), nl.call(xx.detach().numpy()),
        rtol=rtol, atol=atol,        
        err_msg=f"net={ndims} bias={bias} use_dt={ut} act={ac} resnet={resnet} prec={prec}"
      )
      # check self-consistency
      ml1 = MLP.deserialize(ml.serialize())
      np.testing.assert_allclose(
        ml.forward(xx).detach().numpy(), ml1.forward(xx).detach().numpy(),
        rtol=rtol, atol=atol,        
        err_msg=f"net={ndims} bias={bias} use_dt={ut} act={ac} resnet={resnet} prec={prec}"
      )

  def test_jit(self):
    for ndims, bias, ut, ac, resnet, _, prec in self.test_cases:
      layers = []
      for ii in range(1, len(ndims)):        
        ml = layers.append(
          MLPLayer(ndims[ii-1], ndims[ii], bias, ut, ac, resnet, precision=prec).serialize())
      ml = MLP(ml)
      model = torch.jit.script(ml)
      ml1 = MLP.deserialize(ml.serialize())
      model = torch.jit.script(ml1)


@unittest.skipIf(not support_native_net, "NativeLayer not supported")
class TestEmbeddingNet(unittest.TestCase):
  def setUp(self):
    self.test_cases =  itertools.product(
        [1, 3],                                 # inp
        [[24, 48, 96], [24, 36]],               # and hiddens
        ["tanh", "none"],                       # activation
        [True, False],                          # resnet_dt
        ["float32", "double"],                  # precision
    )

  def test_match_embedding_net(
      self,
  ):
    for idim, nn, act, idt, prec in self.test_cases:
      # input
      rtol, atol = get_tols(prec)
      dtype = PRECISION_DICT[prec]
      xx = torch.arange(idim, dtype=dtype)
      # def MLP
      ml = EmbeddingNet(idim, nn, act, idt, prec)
      # check consistency
      nl = DPEmbeddingNet.deserialize(ml.serialize())
      np.testing.assert_allclose(
        ml.forward(xx).detach().numpy(), nl.call(xx.detach().numpy()),
        rtol=rtol, atol=atol,        
        err_msg=f"idim={idim} nn={nn} use_dt={idt} act={act} prec={prec}"
      )
      # check self-consistency
      ml1 = EmbeddingNet.deserialize(ml.serialize())
      np.testing.assert_allclose(
        ml.forward(xx).detach().numpy(), ml1.forward(xx).detach().numpy(),
        rtol=rtol, atol=atol,        
        err_msg=f"idim={idim} nn={nn} use_dt={idt} act={act} prec={prec}"
      )
    
  def test_jit(
      self,
  ):
    for idim, nn, act, idt, prec in self.test_cases:
      # def MLP
      ml = EmbeddingNet(idim, nn, act, idt, prec)
      ml1 = EmbeddingNet.deserialize(ml.serialize())
      model = torch.jit.script(ml)
      model = torch.jit.script(ml1)
    