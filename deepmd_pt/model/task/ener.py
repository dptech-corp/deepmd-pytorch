import logging
import numpy as np
import torch

from deepmd_pt.utils import env
try:
    from typing_extensions import Final
except:
    from torch.jit import Final

from deepmd_pt.model.network import ResidualDeep
from deepmd_pt.model.task import TaskBaseMethod


class EnergyFittingNet(TaskBaseMethod):

    def __init__(self, ntypes, embedding_width, neuron, bias_atom_e, resnet_dt=True, use_tebd=False, **kwargs):
        """Construct a fitting net for energy.

        Args:
        - ntypes: Element count.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neurons in each hidden layers of the fitting net.
        - bias_atom_e: Average enery per atom for each element.
        - resnet_dt: Using time-step in the ResNet construction.
        """
        super(EnergyFittingNet, self).__init__()
        self.ntypes = ntypes
        self.embedding_width = embedding_width
        self.use_tebd = use_tebd
        if not use_tebd:
            assert self.ntypes == len(bias_atom_e), 'Element count mismatches!'
        bias_atom_e = torch.tensor(bias_atom_e)
        self.register_buffer('bias_atom_e', bias_atom_e)

        filter_layers = []
        for type_i in range(self.ntypes):
            one = ResidualDeep(type_i, embedding_width, neuron, bias_atom_e[type_i], resnet_dt=resnet_dt)
            filter_layers.append(one)
        self.filter_layers = torch.nn.ModuleList(filter_layers)

        if 'seed' in kwargs:
            logging.info('Set seed to %d in fitting net.', kwargs['seed'])
            torch.manual_seed(kwargs['seed'])

    def forward(self, inputs, atype):
        """Based on embedding net output, alculate total energy.

        Args:
        - inputs: Embedding matrix. Its shape is [nframes, natoms[0], self.embedding_width].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].

        Returns:
        - `torch.Tensor`: Total energy with shape [nframes, natoms[0]].
        """
        outs = 0
        for type_i, filter_layer in enumerate(self.filter_layers):
            mask = atype == type_i
            atom_energy = filter_layer(inputs)
            if not env.ENERGY_BIAS_TRAINABLE:
                atom_energy = atom_energy + self.bias_atom_e[type_i]
            atom_energy = atom_energy * mask.unsqueeze(-1)
            outs = outs + atom_energy # Shape is [nframes, natoms[0], 1]
        return outs.to(env.GLOBAL_PT_FLOAT_PRECISION)


class EnergyFittingNetType(TaskBaseMethod):

    def __init__(self, ntypes, embedding_width, neuron, bias_atom_e, resnet_dt=True, **kwargs):
        """Construct a fitting net for energy.

        Args:
        - ntypes: Element count.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neurons in each hidden layers of the fitting net.
        - bias_atom_e: Average enery per atom for each element.
        - resnet_dt: Using time-step in the ResNet construction.
        """
        super(EnergyFittingNetType, self).__init__()
        self.ntypes = ntypes
        self.embedding_width = embedding_width
        bias_atom_e = torch.tensor(bias_atom_e)
        self.register_buffer('bias_atom_e', bias_atom_e)

        filter_layers = []
        one = ResidualDeep(0, embedding_width, neuron, 0.0, resnet_dt=resnet_dt)
        filter_layers.append(one)
        self.filter_layers = torch.nn.ModuleList(filter_layers)

        if 'seed' in kwargs:
            logging.info('Set seed to %d in fitting net.', kwargs['seed'])
            torch.manual_seed(kwargs['seed'])

    def forward(self, inputs, atype, atype_tebd = None):
        """Based on embedding net output, alculate total energy.

        Args:
        - inputs: Embedding matrix. Its shape is [nframes, natoms[0], self.embedding_width].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].

        Returns:
        - `torch.Tensor`: Total energy with shape [nframes, natoms[0]].
        """
        outs = 0
        if atype_tebd is not None:
          inputs = torch.concat([inputs, atype_tebd], dim=-1)
        atom_energy = self.filter_layers[0](inputs) + self.bias_atom_e[atype].unsqueeze(-1)
        outs = outs + atom_energy  # Shape is [nframes, natoms[0], 1]
        return outs.to(env.GLOBAL_PT_FLOAT_PRECISION)
