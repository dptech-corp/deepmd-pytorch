from typing import ClassVar, Dict, Optional, List, Union
from deepmd_utils.model_format.network import NativeNet
import numpy as np
import torch
import torch.nn as nn
from deepmd_pt.utils import env

device = env.DEVICE

from deepmd_pt.utils.env import (
  PRECISION_DICT,
  DEFAULT_PRECISION,
)

from deepmd_utils.model_format import (
  NativeLayer,
  NativeNet,
  NetworkCollection as DPNetworkCollection,
  load_dp_model,
  save_dp_model,
)

from deepmd_pt.utils.utils import get_activation_fn, ActivationFn
try:
    from deepmd_utils._version import version as __version__
except ImportError:
    __version__ = "unknown"


def empty_t(shape, precision):
    return torch.empty(shape, dtype=precision, device=device)

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
      precision: str = DEFAULT_PRECISION,
  ):
    super(MLPLayer, self).__init__()
    self.use_timestep = use_timestep
    self.activate_name = activation_function
    self.activate = ActivationFn(self.activate_name)
    self.precision = precision
    self.prec = PRECISION_DICT[self.precision]
    self.matrix = nn.Parameter(
      data=empty_t((num_in, num_out), self.prec)
    )
    nn.init.normal_(self.matrix.data, std=stddev / np.sqrt(num_out + num_in))
    if bias:
      self.bias = nn.Parameter(
        data=empty_t([num_out], self.prec),
      )
      nn.init.normal_(self.bias.data, mean=bavg, std=stddev)
    else:
      self.bias = None
    if self.use_timestep:
      self.idt = nn.Parameter(
        data=empty_t([num_out], self.prec)
      )
      nn.init.normal_(self.idt.data, mean=0.1, std=0.001)
    else:
      self.idt = None
    self.resnet = resnet
    

  def check_type_consistency(self):
      precision = self.precision

      def check_var(var):
          if var is not None:
              # assertion "float64" == "double" would fail
              assert PRECISION_DICT[var.dtype.name] is PRECISION_DICT[precision]

      check_var(self.w)
      check_var(self.b)
      check_var(self.idt)


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
    yy = self.activate(yy).clone()
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
      resnet=self.resnet,
      precision=self.precision,
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
    obj = cls(
      nl["matrix"].shape[0],
      nl["matrix"].shape[1],
      bias=nl["bias"] is not None,
      use_timestep=nl["idt"] is not None,
      activation_function=nl["activation_function"],
      resnet=nl["resnet"],
      precision=nl["precision"],
    )
    prec = PRECISION_DICT[obj.precision]
    check_load_param = \
      lambda ss: nn.Parameter(data=torch.tensor(nl[ss], dtype=prec, device=device)) \
      if nl[ss] is not None else None
    obj.matrix = check_load_param("matrix")
    obj.bias = check_load_param("bias")
    obj.idt = check_load_param("idt")
    return obj


class MLP(nn.Module):
  def __init__(
      self, 
      layers: Optional[List[dict]]=None,
  ):
    super(MLP, self).__init__()
    if layers is None:
      layers = []
    layers = [MLPLayer.deserialize(layer) for layer in layers]
    self.layers = nn.ModuleList(layers)

  def forward(
      self,
      xx: torch.Tensor,
  )->torch.Tensor:
    for ll in self.layers:
      xx = ll(xx)
    return xx

  def serialize(self) -> dict:
    """Serialize the network to a dict.

    Returns
    -------
    dict
        The serialized network.
    """
    return {"layers": [layer.serialize() for layer in self.layers]}
  
  @classmethod
  def deserialize(cls, data: dict) -> "NativeNet":
    """Deserialize the network from a dict.

    Parameters
    ----------
    data : dict
        The dict to deserialize from.
    """
    return cls(data["layers"])


class EmbeddingNet(MLP):
    def __init__(
        self,
        in_dim,
        neuron: List[int] = [24, 48, 96],
        activation_function: str = "tanh",
        resnet_dt: bool = False,
        precision: str = DEFAULT_PRECISION,
    ):
      nh = len(neuron)
      neuron = [in_dim] + neuron
      layers = []
      for ii in range(nh):
        layers.append(
          MLPLayer(
            neuron[ii], neuron[ii+1],
            bias=True,
            use_timestep=resnet_dt,
            activation_function=activation_function,
            resnet=True,
            precision=precision,
          ).serialize()
        )
      super().__init__(layers)
      self.in_dim = in_dim
      self.neuron = neuron
      self.activation_function = activation_function
      self.resnet_dt = resnet_dt
      self.precision = precision
      
    def serialize(self) -> dict:
        """Serialize the network to a dict.

        Returns
        -------
        dict
            The serialized network.
        """
        return {
            "in_dim": self.in_dim,
            "neuron": self.neuron.copy(),
            "activation_function": self.activation_function,
            "resnet_dt": self.resnet_dt,
            "precision": self.precision,
            "layers": [layer.serialize() for layer in self.layers],
        }

    @classmethod
    def deserialize(cls, data: dict) -> "EmbeddingNet":
        """Deserialize the network from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        layers = data.pop("layers")
        obj = cls(**data)
        super(EmbeddingNet, obj).__init__(layers)
        return obj


class NetworkCollection(DPNetworkCollection, nn.Module):
    """PyTorch implementation of NetworkCollection."""
    NETWORK_TYPE_MAP: ClassVar[Dict[str, type]] = {
        "network": MLP,
        "embedding_network": EmbeddingNet,
    }

    def __init__(self, *args, **kwargs):
       # init both two base classes
        DPNetworkCollection.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)
        self.networks = self._networks = torch.nn.ModuleList(self._networks)
