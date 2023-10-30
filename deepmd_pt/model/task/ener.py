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


@Fitting.register("ener")
class EnergyFittingNet(Fitting):

    def __init__(self, ntypes, embedding_width, neuron, bias_atom_e, resnet_dt=True, use_tebd=True, **kwargs):
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
            bias_type = 0.0
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
        """Based on embedding net output, alculate total energy.

        Args:
        - inputs: Embedding matrix. Its shape is [nframes, natoms[0], self.embedding_width].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].

        Returns:
        - `torch.Tensor`: Total energy with shape [nframes, natoms[0]].
        """
        outs = torch.zeros_like(atype).unsqueeze(-1)  # jit assertion
        if self.use_tebd:
            if atype_tebd is not None:
                inputs = torch.concat([inputs, atype_tebd], dim=-1)
            atom_energy = self.filter_layers[0](inputs) + self.bias_atom_e[atype].unsqueeze(-1)
            outs = outs + atom_energy  # Shape is [nframes, natoms[0], 1]
        else:
            for type_i, filter_layer in enumerate(self.filter_layers):
                mask = atype == type_i
                atom_energy = filter_layer(inputs)
                atom_energy = atom_energy + self.bias_atom_e[type_i]
                atom_energy = atom_energy * mask.unsqueeze(-1)
                outs = outs + atom_energy # Shape is [nframes, natoms[0], 1]
        return outs.to(env.GLOBAL_PT_FLOAT_PRECISION), None


@Fitting.register("direct_force")
@Fitting.register("direct_force_ener")
class EnergyFittingNetDirect(Fitting):

    def __init__(self, ntypes, embedding_width, neuron, bias_atom_e, out_dim=1, resnet_dt=True, use_tebd=True, return_energy=False, **kwargs):
        """Construct a fitting net for energy.

        Args:
        - ntypes: Element count.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neurons in each hidden layers of the fitting net.
        - bias_atom_e: Average enery per atom for each element.
        - resnet_dt: Using time-step in the ResNet construction.
        """
        super(EnergyFittingNetDirect, self).__init__()
        self.ntypes = ntypes
        self.embedding_width = embedding_width
        self.use_tebd = use_tebd
        self.out_dim = out_dim
        if not use_tebd:
            assert self.ntypes == len(bias_atom_e), 'Element count mismatches!'
        bias_atom_e = torch.tensor(bias_atom_e)
        self.register_buffer('bias_atom_e', bias_atom_e)

        filter_layers_dipole = []
        for type_i in range(self.ntypes):
            one = ResidualDeep(type_i, embedding_width, neuron, 0.0, out_dim=out_dim, resnet_dt=resnet_dt)
            filter_layers_dipole.append(one)
        self.filter_layers_dipole = torch.nn.ModuleList(filter_layers_dipole)

        self.return_energy = return_energy
        filter_layers = []
        if self.return_energy:
            for type_i in range(self.ntypes):
                bias_type = 0.0 if self.use_tebd else bias_atom_e[type_i]
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
                rot_mat: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, None]:
        """Based on embedding net output, alculate total energy.

        Args:
        - inputs: Embedding matrix. Its shape is [nframes, natoms[0], self.embedding_width].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].

        Returns:
        - `torch.Tensor`: Total energy with shape [nframes, natoms[0]].
        """
        nframes, nloc, _ = inputs.size()
        if self.use_tebd:
            if atype_tebd is not None:
                inputs = torch.concat([inputs, atype_tebd], dim=-1)
            vec_out = self.filter_layers_dipole[0](inputs)  # Shape is [nframes, nloc, m1]
            assert list(vec_out.size()) == [nframes, nloc, self.out_dim]
            vec_out = vec_out.view(-1, 1, self.out_dim)
            assert rot_mat is not None
            vec_out = torch.bmm(vec_out, rot_mat).squeeze(-2).view(nframes, nloc, 3)  # Shape is [nframes, nloc, 3]
        else:
            vec_out = torch.zeros_like(atype).unsqueeze(-1)  # jit assertion
            for type_i, filter_layer in enumerate(self.filter_layers_dipole):
                mask = atype == type_i
                vec_out_type = filter_layer(inputs)  # Shape is [nframes, nloc, m1]
                vec_out_type = vec_out_type * mask.unsqueeze(-1)
                vec_out = vec_out + vec_out_type # Shape is [nframes, natoms[0], 1]

        outs = torch.zeros_like(atype).unsqueeze(-1)  # jit assertion
        if self.return_energy:
            if self.use_tebd:
                atom_energy = self.filter_layers[0](inputs) + self.bias_atom_e[atype].unsqueeze(-1)
                outs = outs + atom_energy  # Shape is [nframes, natoms[0], 1]
            else:
                for type_i, filter_layer in enumerate(self.filter_layers):
                    mask = atype == type_i
                    atom_energy = filter_layer(inputs)
                    if not env.ENERGY_BIAS_TRAINABLE:
                        atom_energy = atom_energy + self.bias_atom_e[type_i]
                    atom_energy = atom_energy * mask.unsqueeze(-1)
                    outs = outs + atom_energy # Shape is [nframes, natoms[0], 1]
        return outs.to(env.GLOBAL_PT_FLOAT_PRECISION), vec_out
