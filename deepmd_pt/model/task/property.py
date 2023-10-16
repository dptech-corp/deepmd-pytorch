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

    def __init__(self, ntypes, embedding_width, neuron, bias_atom_p, resnet_dt=True, use_tebd=True, **kwargs):
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
        if not use_tebd:
            assert self.ntypes == len(bias_atom_p), 'Element count mismatches!'
        bias_atom_p = torch.tensor(bias_atom_p)
        self.mean = kwargs.get("mean",None)
        self.std = kwargs.get("std",None)
        self.register_buffer('bias_atom_p', bias_atom_p)

        filter_layers = []
        for type_i in range(self.ntypes):
            bias_type = 0.0 if self.use_tebd else bias_atom_p[type_i]
            one = ResidualDeep(type_i, embedding_width, neuron, bias_type, resnet_dt=resnet_dt)
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
        """Based on embedding net output, alculate property.

        Args:
        - inputs: Embedding matrix. Its shape is [nframes, natoms[0], self.embedding_width].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].

        Returns:
        - `torch.Tensor`: Total energy with shape [nframes, natoms[0]].
        """
        outs = torch.zeros_like(atype).unsqueeze(-1)
        if self.use_tebd:
            if atype_tebd is not None:
                inputs = torch.concat([inputs, atype_tebd], dim=-1)
            if (self.mean is not None) and (self.std is not None):
                atom_prop = (self.filter_layers[0](inputs) * self.std) + self.mean
                #logging.info(f"{self.filter_layers[0](inputs)}")
                #logging.info(f"{self.filter_layers[0](inputs) * self.std}")
                #logging.info(f"{atom_prop}")
            else:
                atom_prop = self.filter_layers[0](inputs) + self.bias_atom_p[atype].unsqueeze(-1)           
            #logging.info(f"{atom_prop[0][0]}")
            #logging.info(f"atype:{atype}")
            #logging.info(f"self.filter:{self.filter_layers[0](inputs)}")
            #logging.info(f"bias:{self.bias_atom_p[atype]}")
            #logging.info(f"self.bias_atom_p:{self.bias_atom_p}")
            outs = outs + atom_prop  # Shape is [nframes, natoms[0], 1]
        else:
            for type_i, filter_layer in enumerate(self.filter_layers):
                mask = atype == type_i
                atom_prop = filter_layer(inputs)
                if not env.PROPERTY_BIAS_TRAINABLE:
                    atom_prop = atom_prop + self.bias_atom_p[type_i]
                atom_prop = atom_prop * mask.unsqueeze(-1)
                outs = outs + atom_prop # Shape is [nframes, natoms[0], 1]
        return outs.to(env.GLOBAL_PT_FLOAT_PRECISION)