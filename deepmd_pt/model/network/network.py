import numpy as np
import torch

from deepmd_pt.utils import env

try:
    from typing import Final
except:
    from torch.jit import Final


def Tensor(*shape):
    return torch.empty(shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION)


class ResidualLinear(torch.nn.Module):
    resnet: Final[int]

    def __init__(self, num_in, num_out, bavg=0., stddev=1., resnet_dt=False):
        """Construct a residual linear layer.

        Args:
        - num_in: Width of input tensor.
        - num_out: Width of output tensor.
        - resnet_dt: Using time-step in the ResNet construction.
        """
        super(ResidualLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.resnet = resnet_dt

        self.matrix = torch.nn.Parameter(data=Tensor(num_in, num_out))
        torch.nn.init.normal_(self.matrix.data, std=stddev / np.sqrt(num_out + num_in))
        self.bias = torch.nn.Parameter(data=Tensor(1, num_out))
        torch.nn.init.normal_(self.bias.data, mean=bavg, std=stddev)
        if self.resnet:
            self.idt = torch.nn.Parameter(data=Tensor(1, num_out))
            torch.nn.init.normal_(self.idt.data, mean=1., std=0.001)

    def forward(self, inputs):
        """Return X ?+ X*W+b."""
        xw_plus_b = torch.matmul(inputs, self.matrix) + self.bias
        hidden = torch.tanh(xw_plus_b)
        if self.resnet:
            hidden = hidden * self.idt
        if self.num_in == self.num_out:
            return inputs + hidden
        elif self.num_in * 2 == self.num_out:
            return torch.cat([inputs, inputs], dim=1) + hidden
        else:
            return hidden


class TypeFilter(torch.nn.Module):

    def __init__(self, offset, length, neuron):
        """Construct a filter on the given element as neighbor.

        Args:
        - offset: Element offset in the descriptor matrix.
        - length: Atom count of this element.
        - neuron: Number of neurons in each hidden layers of the embedding net.
        """
        super(TypeFilter, self).__init__()
        self.offset = offset
        self.length = length
        self.neuron = [1] + neuron

        deep_layers = []
        for ii in range(1, len(self.neuron)):
            one = ResidualLinear(self.neuron[ii - 1], self.neuron[ii])
            deep_layers.append(one)
        self.deep_layers = torch.nn.ModuleList(deep_layers)

    def forward(self, inputs):
        """Calculate decoded embedding for each atom.

        Args:
        - inputs: Descriptor matrix. Its shape is [nframes*natoms[0], len_descriptor].

        Returns:
        - `torch.Tensor`: Embedding contributed by me. Its shape is [nframes*natoms[0], 4, self.neuron[-1]].
        """
        inputs_i = inputs[:, self.offset * 4:(self.offset + self.length) * 4]
        inputs_reshape = inputs_i.reshape(-1, 4)  # shape is [nframes*natoms[0]*self.length, 4]
        xyz_scatter = inputs_reshape[:, 0:1]
        for linear in self.deep_layers:
            xyz_scatter = linear(xyz_scatter)
        xyz_scatter = xyz_scatter.view(-1, self.length,
                                       self.neuron[-1])  # shape is [nframes*natoms[0], self.length, self.neuron[-1]]
        inputs_reshape = inputs_i.view(-1, self.length, 4).permute(0, 2,
                                                                   1)  # shape is [nframes*natoms[0], 4, self.length]
        return torch.matmul(inputs_reshape, xyz_scatter)


class SimpleLinear(torch.nn.Module):
    resnet: Final[bool]

    def __init__(self,
                 num_in,
                 num_out,
                 bavg=0.,
                 stddev=1.,
                 use_timestep=False,
                 activate=True):
        """Construct a linear layer.

        Args:
        - num_in: Width of input tensor.
        - num_out: Width of output tensor.
        - use_timestep: Apply time-step to weight.
        - activate: Whether apply TANH to hidden layer.
        """
        super(SimpleLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.resnet = use_timestep
        self.activate = activate

        self.matrix = torch.nn.Parameter(data=Tensor(num_in, num_out))
        torch.nn.init.normal_(self.matrix.data, std=stddev / np.sqrt(num_out + num_in))
        self.bias = torch.nn.Parameter(data=Tensor(1, num_out))
        torch.nn.init.normal_(self.bias.data, mean=bavg, std=stddev)
        if self.resnet:
            self.idt = torch.nn.Parameter(data=Tensor(1, num_out))
            torch.nn.init.normal_(self.idt.data, mean=0.1, std=0.001)

    def forward(self, inputs):
        """Return X*W+b."""
        hidden = torch.matmul(inputs, self.matrix) + self.bias
        if self.activate:
            hidden = torch.tanh(hidden)
        if self.resnet:
            hidden = hidden * self.idt
        return hidden


class ResidualDeep(torch.nn.Module):

    def __init__(self, type_id, embedding_width, neuron, bias_atom_e, resnet_dt=False):
        """Construct a filter on the given element as neighbor.

        Args:
        - typei: Element ID.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neurons in each hidden layers of the embedding net.
        - resnet_dt: Using time-step in the ResNet construction.
        """
        super(ResidualDeep, self).__init__()
        self.type_id = type_id
        self.neuron = [embedding_width] + neuron

        deep_layers = []
        for ii in range(1, len(self.neuron)):
            one = SimpleLinear(
                num_in=self.neuron[ii - 1],
                num_out=self.neuron[ii],
                use_timestep=(resnet_dt and ii > 1 and self.neuron[ii - 1] == self.neuron[ii])
            )
            deep_layers.append(one)
        self.deep_layers = torch.nn.ModuleList(deep_layers)
        if not env.ENERGY_BIAS_TRAINABLE:
            bias_atom_e = 0
        self.final_layer = SimpleLinear(self.neuron[-1], 1, bias_atom_e, activate=False)

    def forward(self, inputs):
        """Calculate decoded embedding for each atom.

        Args:
        - inputs: Embedding net output per atom. Its shape is [nframes*nloc, self.embedding_width].

        Returns:
        - `torch.Tensor`: Output layer with shape [nframes*nloc, self.neuron[-1]].
        """
        outputs = inputs
        for idx, linear in enumerate(self.deep_layers):
            if idx > 0 and linear.num_in == linear.num_out:
                outputs = outputs + linear(outputs)
            else:
                outputs = linear(outputs)
        outputs = self.final_layer(outputs)
        return outputs
