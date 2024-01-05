from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from deepmd_pt.utils import env

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

from deepmd_utils.model_format import (
  NativeLayer,
  NativeNet,
  load_dp_model,
  save_dp_model,
)

from deepmd_pt.utils.utils import get_activation_fn, ActivationFn
try:
    from deepmd_utils._version import version as __version__
except ImportError:
    __version__ = "unknown"


def empty_t(*shape):
    return torch.empty(shape, dtype=dtype, device=device)

class MLPLayer(nn.Module):
  def __init__(
      self, 
      num_in,
      num_out,
      bias: bool = True,
      use_timestep: bool = False,
      activation_function: Optional[str] = None,
      resnet: bool = False,
      bavg: float = 0.,
      stddev: float = 1.,
  ):
    super(MLPLayer, self).__init__()
    self.use_timestep = use_timestep
    self.activate_name = activation_function
    self.activate = ActivationFn(self.activate_name)
    self.matrix = nn.Parameter(data=empty_t(num_in, num_out))
    nn.init.normal_(self.matrix.data, std=stddev / np.sqrt(num_out + num_in))
    if bias:
      self.bias = nn.Parameter(data=empty_t(num_out))
      nn.init.normal_(self.bias.data, mean=bavg, std=stddev)
    else:
      self.bias = None
    if self.use_timestep:
      self.idt = nn.Parameter(data=empty_t(num_out))
      nn.init.normal_(self.idt.data, mean=0.1, std=0.001)
    else:
      self.idt = None
    self.resnet = resnet
    

  def forward(
      self,
      xx: torch.Tensor,
  )->torch.Tensor:
    """One MLP layer used by DP model.

    Parameters
    ----------
    xx: torch.Tensor
        The input.

    Returns
    -------
    yy: torch.Tensor
        The output.
    """
    yy = (
      torch.matmul(xx, self.matrix) + self.bias \
      if self.bias is not None \
      else torch.matmul(xx, self.matrix)
    )
    yy = self.activate.forward(yy)
    yy = yy * self.idt if self.idt is not None else yy
    if self.resnet:
      if xx.shape[-1] == yy.shape[-1]:
        yy += xx
      elif 2 * xx.shape[-1] == yy.shape[-1]:
        yy += torch.concat([xx, xx], dim=-1)
      else:
        yy = yy        
    return yy
    
  
  def serialize(self) -> dict:
    """Serialize the layer to a dict.

    Returns
    -------
    dict
        The serialized layer.
    """
    nl = NativeLayer(
      self.matrix.detach().numpy(),
      self.bias.detach().numpy() if self.bias is not None else None,
      self.idt.detach().numpy() if self.idt is not None else None,
      activation_function=self.activate_name,
      resnet=self.resnet
    )
    return nl.serialize()

  @classmethod
  def deserialize(cls, data:dict)->"MLPLayer":
    """Deserialize the layer from a dict.

    Parameters
    ----------
    data : dict
        The dict to deserialize from.
    """
    nl = NativeLayer.deserialize(data)
    self.use_timestep = nl["idt"] is not None
    self.activate_name = nl["activation_function"]
    self.activate = ActivationFn(self.activate_name)
    check_load_param = \
      lambda ss: nn.Parameter(data=torch.Tensor(nl[ss], dtype=dtype, device=device)) \
      if nl[ss] is not None else None
    self.matrix = check_load_param("matrix")
    self.bias = check_load_param("bias")
    self.idt = check_load_param("idt")
    assert (self.idt is not None) == self.use_timestep
    self.resnet = nl["resnet"]
