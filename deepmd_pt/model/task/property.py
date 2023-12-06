import logging
import torch

from deepmd_pt.utils import env
try:
    from typing_extensions import Final
except:
    from torch.jit import Final

from typing import Optional, List, Dict, Tuple
from deepmd_pt.model.network import ResidualDeep
from deepmd_pt.model.task import Fitting

@Fitting.register("prop")
class PropertyFittingNet(Fitting):

    def __init__(self, ntypes, embedding_width, neuron, resnet_dt=True, use_tebd=True, **kwargs):
        """Construst a fitting net for property.
        Args:
        - ntypes: Element count.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neuron in each hidden layers of the fitting net.
        - resnet_dt: Using time-step in the ResNet construction.
        """
        super(PropertyFittingNet, self).__init__()
        self.ntypes = ntypes
        self.embedding_width = embedding_width
        self.use_tebd = use_tebd

        filter_layers = []
        for type_i in range(self.ntypes):
            one = ResidualDeep(type_i, embedding_width, neuron, 0.0, resnet_dt=resnet_dt)
            filter_layers.append(one)
        self.filter_layers = torch.nn.ModuleList(filter_layers)

        if 'seed' in kwargs:
            logging.info('Set seed to %d in fitting net.', kwargs['seed'])
            torch.manual_seed(kwargs['seed'])

    def forward(self,
                inputs: torch.Tensor,
                atype: torch.Tensor,
                atype_tebd: Optional[torch.Tensor] = None,
                rot_mat: Optional[torch.Tensor] = None):
        """Based on embedding net output, calculate property.
        Args:
        - inputs: Embedding matrix. Its shape is [nframes, natoms[0], self.embedding_width].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].

        Returns:
        - `torch.Tensor`: Property with shape [nframes, natoms[0]].
        """
        outs = torch.zeros_like(atype).unsqueeze(-1)
        if self.use_tebd:
            if atype_tebd is not None:
                inputs = torch.concat([inputs, atype_tebd], dim=-1)
            atom_prop = self.filter_layers[0](inputs)
            outs = outs + atom_prop  # Shape is [nframes, natoms[0], 1]
        else:
            for type_i, filter_layer in enumerate(self.filter_layers):
                mask = atype == type_i
                atom_prop = filter_layer(inputs)
                atom_prop = atom_prop * mask.unsqueeze(-1)
                outs = outs + atom_prop # Shape is [nframes, natoms[0], 1]
        return outs.to(env.GLOBAL_PT_FLOAT_PRECISION)