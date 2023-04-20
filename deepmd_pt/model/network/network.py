import numpy as np
import torch

from deepmd_pt.utils import env

try:
    from typing import Final
except:
    from torch.jit import Final

from deepmd_pt.utils.utils import get_activation_fn


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

    def __init__(self, offset, length, neuron, return_G=False, tebd_dim=0, use_tebd=False, tebd_mode='concat'):
        """Construct a filter on the given element as neighbor.

        Args:
        - offset: Element offset in the descriptor matrix.
        - length: Atom count of this element.
        - neuron: Number of neurons in each hidden layers of the embedding net.
        """
        super(TypeFilter, self).__init__()
        self.offset = offset
        self.length = length
        self.tebd_dim = tebd_dim
        self.use_tebd = use_tebd
        self.tebd_mode = tebd_mode
        if use_tebd and tebd_mode == 'concat':
            self.neuron = [1 + tebd_dim * 2] + neuron
        else:
            self.neuron = [1] + neuron

        deep_layers = []
        for ii in range(1, len(self.neuron)):
            one = ResidualLinear(self.neuron[ii - 1], self.neuron[ii])
            deep_layers.append(one)
        self.deep_layers = torch.nn.ModuleList(deep_layers)
        self.return_G = return_G

    def forward(self, inputs, atype_tebd=None, nlist_tebd=None):
        """Calculate decoded embedding for each atom.

        Args:
        - inputs: Descriptor matrix. Its shape is [nframes*natoms[0], len_descriptor].

        Returns:
        - `torch.Tensor`: Embedding contributed by me. Its shape is [nframes*natoms[0], 4, self.neuron[-1]].
        """
        inputs_i = inputs[:, self.offset * 4:(self.offset + self.length) * 4]
        inputs_reshape = inputs_i.reshape(-1, 4)  # shape is [nframes*natoms[0]*self.length, 4]
        xyz_scatter = inputs_reshape[:, 0:1]
        if self.use_tebd and self.tebd_mode == 'concat':
            nlist_tebd = nlist_tebd.reshape(-1, self.tebd_dim)
            atype_tebd = atype_tebd.reshape(-1, self.tebd_dim)
            # [nframes * nloc * nnei, 17]
            xyz_scatter = torch.concat([xyz_scatter, nlist_tebd, atype_tebd], dim=1)

        for linear in self.deep_layers:
            xyz_scatter = linear(xyz_scatter)
        xyz_scatter = xyz_scatter.view(-1, self.length,
                                       self.neuron[-1])  # shape is [nframes*natoms[0], self.length, self.neuron[-1]]
        if self.return_G:
            return xyz_scatter
        else:
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


class TypeEmbedNet(torch.nn.Module):

    def __init__(self, type_nums, embed_dim, bavg=0.0, stddev=1.0):
        """Construct a type embedding net.
        """
        super(TypeEmbedNet, self).__init__()
        self.embedding = torch.nn.Embedding(type_nums + 1, embed_dim, padding_idx=type_nums)
        torch.nn.init.normal_(self.embedding.weight[:-1], mean=bavg, std=stddev)

    def forward(self, atype):
        """

        Args:
            atype: Type of each input, [nframes, nloc] or [nframes, nloc, nnei]

        Returns:
            type_embedding:

        """
        return self.embedding(atype)


class NeighborWiseAttention(torch.nn.Module):
    def __init__(self, layer_num, nnei, embed_dim, hidden_dim, dotr=False, do_mask=False, post_ln=True,
                 ffn=False, ffn_embed_dim=1024, activation="tanh", scaling_factor=1.0,
                 head_num=1, normalize=True, temperature=None):
        """Construct a neighbor-wise attention net.
        """
        super(NeighborWiseAttention, self).__init__()
        self.layer_num = layer_num
        attention_layers = []
        for i in range(self.layer_num):
            attention_layers.append(NeighborWiseAttentionLayer(nnei, embed_dim, hidden_dim,
                                                               dotr=dotr, do_mask=do_mask,
                                                               post_ln=post_ln, ffn=ffn,
                                                               ffn_embed_dim=ffn_embed_dim,
                                                               activation=activation,
                                                               scaling_factor=scaling_factor,
                                                               head_num=head_num,
                                                               normalize=normalize,
                                                               temperature=temperature))
        self.attention_layers = torch.nn.ModuleList(attention_layers)

    def forward(self, input_G, nei_mask, input_r=None):
        """

        Args:
            input_G: Input G, [nframes * nloc, nnei, embed_dim]
            nei_mask: neighbor mask, [nframes * nloc, nnei]
            input_r: normalized radial, [nframes, nloc, nei, 3]

        Returns:
            out: Output G, [nframes * nloc, nnei, embed_dim]

        """
        out = input_G
        for i in range(self.layer_num):
            out = self.attention_layers[i](out, nei_mask, input_r=input_r)
        return out


class NeighborWiseAttentionLayer(torch.nn.Module):
    def __init__(self, nnei, embed_dim, hidden_dim, dotr=False, do_mask=False, post_ln=True,
                 ffn=False, ffn_embed_dim=1024, activation="tanh", scaling_factor=1.0,
                 head_num=1, normalize=True, temperature=None):
        """Construct a neighbor-wise attention layer.
        """
        super(NeighborWiseAttentionLayer, self).__init__()
        self.nnei = nnei
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dotr = dotr
        self.do_mask = do_mask
        self.post_ln = post_ln
        self.ffn = ffn
        self.attention_layer = GatedSelfAttetion(nnei, embed_dim, hidden_dim, dotr=dotr, do_mask=do_mask,
                                                 scaling_factor=scaling_factor, head_num=head_num, normalize=normalize,
                                                 temperature=temperature)
        self.attn_layer_norm = torch.nn.LayerNorm(self.embed_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
        self.final_layer_norm = torch.nn.LayerNorm(self.embed_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
        if self.ffn:
            self.ffn_embed_dim = ffn_embed_dim
            self.fc1 = torch.nn.Linear(self.embed_dim, self.ffn_embed_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
            self.fc2 = torch.nn.Linear(self.ffn_embed_dim, self.embed_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
            self.activation_fn = get_activation_fn(activation)

    def forward(self, x, nei_mask, input_r=None):
        residual = x
        if not self.post_ln:
            x = self.attn_layer_norm(x)
        x = self.attention_layer(x, nei_mask, input_r=input_r)
        x = residual + x
        if self.post_ln:
            x = self.attn_layer_norm(x)
        if self.ffn:
            residual = x
            if not self.post_ln:
                x = self.final_layer_norm(x)
            x = self.fc1(x)
            x = self.activation_fn(x)
            x = self.fc2(x)
            x = residual + x
            if self.post_ln:
                x = self.final_layer_norm(x)
        return x


class GatedSelfAttetion(torch.nn.Module):
    def __init__(self, nnei, embed_dim, hidden_dim, dotr=False, do_mask=False, scaling_factor=1.0,
                 head_num=1, normalize=True, temperature=None):
        """Construct a neighbor-wise attention net.
        """
        super(GatedSelfAttetion, self).__init__()
        self.nnei = nnei
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.dotr = dotr
        self.do_mask = do_mask
        if temperature is None:
            self.scaling = (self.hidden_dim * scaling_factor) ** -0.5
        else:
            self.scaling = temperature
        self.normalize = normalize
        self.in_proj = SimpleLinear(embed_dim, hidden_dim * 3, bavg=0., stddev=1., use_timestep=False, activate=False)
        self.out_proj = SimpleLinear(hidden_dim, embed_dim, bavg=0., stddev=1., use_timestep=False, activate=False)

    def forward(self, query, nei_mask, input_r=None):
        """

        Args:
            query: input G, [nframes * nloc, nnei, embed_dim]
            nei_mask: neighbor mask, [nframes * nloc, nnei]
            input_r: normalized radial, [nframes, nloc, nei, 3]

        Returns:
            type_embedding:

        """
        q, k, v = self.in_proj(query).chunk(3, dim=-1)
        #  [nframes * nloc, nnei, hidden_dim]
        q = q.view(-1, self.nnei, self.hidden_dim)
        k = k.view(-1, self.nnei, self.hidden_dim)
        v = v.view(-1, self.nnei, self.hidden_dim)
        if self.normalize:
            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)
            v = torch.nn.functional.normalize(v, dim=-1)
        q = q * self.scaling
        k = k.transpose(1, 2)
        #  [nframes * nloc, nnei, nnei]
        attn_weights = torch.bmm(q, k)
        #  [nframes * nloc, nnei]
        nei_mask = nei_mask.view(-1, self.nnei)
        attn_weights = attn_weights.masked_fill(~nei_mask.unsqueeze(1), float("-inf"))
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.masked_fill(~nei_mask.unsqueeze(-1), float(0.0))
        if self.dotr:
            angular_weight = torch.bmm(input_r, input_r.transpose(1, 2))
            attn_weights = attn_weights * angular_weight
        o = torch.bmm(attn_weights, v)
        output = self.out_proj(o)
        return output
