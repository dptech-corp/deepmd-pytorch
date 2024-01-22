from abc import ABC, abstractmethod
import torch
import logging
import os
from typing import (
  Optional,
  Dict,
)
from deepmd_utils.model_format import FittingOutputDef
from deepmd_pt.model.task import Fitting


class AtomicModel(ABC):

  @abstractmethod
  def get_fitting_net(self)->Fitting:
    raise NotImplementedError

  @abstractmethod
  def get_fitting_output_def(self)->FittingOutputDef:
    raise NotImplementedError

  @abstractmethod
  def get_rcut(self)->float:
    raise NotImplementedError

  @abstractmethod
  def get_sel(self)->int:
    raise NotImplementedError

  @abstractmethod
  def distinguish_types(self)->bool:
    raise NotImplementedError

  @abstractmethod
  def forward_atomic(
      self, 
      extended_coord, 
      extended_atype, 
      nlist,
      mapping: Optional[torch.Tensor] = None,
      do_atomic_virial: bool = False,
  ) -> Dict[str, torch.Tensor]:
    raise NotImplementedError
  
    
