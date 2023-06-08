from typing import Optional
import numpy as np
import torch

from deepmd_pt.utils import env

try:
    from typing import Final
except:
    from torch.jit import Final

from deepmd_pt.utils.utils import get_activation_fn, ActivationFn


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
    use_tebd: Final[bool]
    tebd_mode: Final[str]

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
        supported_tebd_mode = ['concat', 'dot', 'dot_residual_s', 'dot_residual_t']
        assert tebd_mode in supported_tebd_mode, f'Unknown tebd_mode {tebd_mode}! Supported are {supported_tebd_mode}.'
        if use_tebd and tebd_mode == 'concat':
            self.neuron = [1 + tebd_dim * 2] + neuron
        else:
            self.neuron = [1] + neuron

        deep_layers = []
        for ii in range(1, len(self.neuron)):
            one = ResidualLinear(self.neuron[ii - 1], self.neuron[ii])
            deep_layers.append(one)
        self.deep_layers = torch.nn.ModuleList(deep_layers)

        deep_layers_t = []
        if use_tebd and tebd_mode in ['dot', 'dot_residual_s', 'dot_residual_t']:
            self.neuron_t = [tebd_dim * 2] + neuron
            for ii in range(1, len(self.neuron_t)):
                one = ResidualLinear(self.neuron_t[ii - 1], self.neuron_t[ii])
                deep_layers_t.append(one)
        self.deep_layers_t = torch.nn.ModuleList(deep_layers_t)

        self.return_G = return_G

    def forward(self, inputs, atype_tebd: Optional[torch.Tensor]=None, nlist_tebd: Optional[torch.Tensor]=None):
        """Calculate decoded embedding for each atom.

        Args:
        - inputs: Descriptor matrix. Its shape is [nframes*natoms[0], len_descriptor].

        Returns:
        - `torch.Tensor`: Embedding contributed by me. Its shape is [nframes*natoms[0], 4, self.neuron[-1]].
        """
        inputs_i = inputs[:, self.offset * 4:(self.offset + self.length) * 4]
        inputs_reshape = inputs_i.reshape(-1, 4)  # shape is [nframes*natoms[0]*self.length, 4]
        xyz_scatter = inputs_reshape[:, 0:1]

        # concat the tebd as input
        if self.use_tebd and self.tebd_mode == 'concat':
            assert nlist_tebd is not None and atype_tebd is not None
            nlist_tebd = nlist_tebd.reshape(-1, self.tebd_dim)
            atype_tebd = atype_tebd.reshape(-1, self.tebd_dim)
            # [nframes * nloc * nnei, 1 + tebd_dim * 2]
            xyz_scatter = torch.concat([xyz_scatter, nlist_tebd, atype_tebd], dim=1)

        for linear in self.deep_layers:
            xyz_scatter = linear(xyz_scatter)
            # [nframes * nloc * nnei, out_size]

        # dot the tebd output
        if self.use_tebd and self.tebd_mode in ['dot', 'dot_residual_s', 'dot_residual_t']:
            assert nlist_tebd is not None and atype_tebd is not None
            nlist_tebd = nlist_tebd.reshape(-1, self.tebd_dim)
            atype_tebd = atype_tebd.reshape(-1, self.tebd_dim)
            # [nframes * nloc * nnei, tebd_dim * 2]
            two_side_tebd = torch.concat([nlist_tebd, atype_tebd], dim=1)
            for linear in self.deep_layers_t:
                two_side_tebd = linear(two_side_tebd)
                # [nframes * nloc * nnei, out_size]
            if self.tebd_mode == 'dot':
                xyz_scatter = xyz_scatter * two_side_tebd
            elif self.tebd_mode == 'dot_residual_s':
                xyz_scatter = xyz_scatter * two_side_tebd + xyz_scatter
            elif self.tebd_mode == 'dot_residual_t':
                xyz_scatter = xyz_scatter * two_side_tebd + two_side_tebd

        xyz_scatter = xyz_scatter.view(-1, self.length,
                                       self.neuron[-1])  # shape is [nframes*natoms[0], self.length, self.neuron[-1]]
        if self.return_G:
            return xyz_scatter
        else:
            # shape is [nframes*natoms[0], 4, self.length]
            inputs_reshape = inputs_i.view(-1, self.length, 4).permute(0, 2, 1)
            return torch.matmul(inputs_reshape, xyz_scatter)


class SimpleLinear(torch.nn.Module):
    use_timestep: Final[bool]

    def __init__(self,
                 num_in,
                 num_out,
                 bavg=0.,
                 stddev=1.,
                 use_timestep=False,
                 activate=None,
                 bias: bool = True,
                 ):
        """Construct a linear layer.

        Args:
        - num_in: Width of input tensor.
        - num_out: Width of output tensor.
        - use_timestep: Apply time-step to weight.
        - activate: type of activate func.
        """
        super(SimpleLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.use_timestep = use_timestep
        self.activate = ActivationFn(activate)

        self.matrix = torch.nn.Parameter(data=Tensor(num_in, num_out))
        torch.nn.init.normal_(self.matrix.data, std=stddev / np.sqrt(num_out + num_in))
        if bias:
          self.bias = torch.nn.Parameter(data=Tensor(1, num_out))
          torch.nn.init.normal_(self.bias.data, mean=bavg, std=stddev)
        else:
          self.bias = None
        if self.use_timestep:
            self.idt = torch.nn.Parameter(data=Tensor(1, num_out))
            torch.nn.init.normal_(self.idt.data, mean=0.1, std=0.001)

    def forward(self, inputs):
        """Return X*W+b."""        
        xw = torch.matmul(inputs, self.matrix)
        hidden = xw + self.bias if self.bias is not None else xw
        hidden = self.activate(hidden)
        if self.use_timestep:
            hidden = hidden * self.idt
        return hidden


class NonLinearHead(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 out_dim,
                 activation_fn,
                 hidden=None):
        super(NonLinearHead, self).__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = SimpleLinear(input_dim, hidden, activate=activation_fn)
        self.linear2 = SimpleLinear(hidden, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class MaskLMHead(torch.nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = SimpleLinear(embed_dim, embed_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.layer_norm = torch.nn.LayerNorm(embed_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)

        if weight is None:
            weight = torch.nn.Linear(embed_dim, output_dim, bias=False, dtype=env.GLOBAL_PT_FLOAT_PRECISION).weight
        self.weight = weight
        self.bias = torch.nn.Parameter(torch.zeros(output_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION))

    def forward(self, features, masked_tokens: Optional[torch.Tensor]=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = torch.nn.functional.linear(x, self.weight) + self.bias
        return x


class ResidualDeep(torch.nn.Module):

    def __init__(self, type_id, embedding_width, neuron, bias_atom_e, out_dim=1, resnet_dt=False):
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
        self.out_dim = out_dim

        deep_layers = []
        for ii in range(1, len(self.neuron)):
            one = SimpleLinear(
                num_in=self.neuron[ii - 1],
                num_out=self.neuron[ii],
                use_timestep=(resnet_dt and ii > 1 and self.neuron[ii - 1] == self.neuron[ii]),
                activate="tanh",
            )
            deep_layers.append(one)
        self.deep_layers = torch.nn.ModuleList(deep_layers)
        if not env.ENERGY_BIAS_TRAINABLE:
            bias_atom_e = 0
        self.final_layer = SimpleLinear(self.neuron[-1], self.out_dim, bias_atom_e)

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
        self.embedding = torch.nn.Embedding(type_nums + 1, embed_dim, padding_idx=type_nums,
                                            dtype=env.GLOBAL_PT_FLOAT_PRECISION)
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

    def forward(self, input_G, nei_mask, input_r: Optional[torch.Tensor]=None):
        """

        Args:
            input_G: Input G, [nframes * nloc, nnei, embed_dim]
            nei_mask: neighbor mask, [nframes * nloc, nnei]
            input_r: normalized radial, [nframes, nloc, nei, 3]

        Returns:
            out: Output G, [nframes * nloc, nnei, embed_dim]

        """
        out = input_G
        # https://github.com/pytorch/pytorch/issues/39165#issuecomment-635472592
        for layer in self.attention_layers:
            out = layer(out, nei_mask, input_r=input_r)
        return out


class NeighborWiseAttentionLayer(torch.nn.Module):
    ffn: Final[bool]

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
        if self.ffn:
            self.ffn_embed_dim = ffn_embed_dim
            self.fc1 = torch.nn.Linear(self.embed_dim, self.ffn_embed_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
            self.activation_fn = get_activation_fn(activation)
            self.fc2 = torch.nn.Linear(self.ffn_embed_dim, self.embed_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
            self.final_layer_norm = torch.nn.LayerNorm(self.embed_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)

    def forward(self, x, nei_mask, input_r: Optional[torch.Tensor]=None):
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
                 head_num=1, normalize=True, temperature=None, bias=True):
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
        self.in_proj = SimpleLinear(embed_dim, hidden_dim * 3, bavg=0., stddev=1., use_timestep=False, bias=bias)
        self.out_proj = SimpleLinear(hidden_dim, embed_dim, bavg=0., stddev=1., use_timestep=False, bias=bias)

    def forward(self, query, nei_mask, input_r: Optional[torch.Tensor]=None):
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
            assert input_r is not None, "input_r must be provided when dotr is True!"
            angular_weight = torch.bmm(input_r, input_r.transpose(1, 2))
            attn_weights = attn_weights * angular_weight
        o = torch.bmm(attn_weights, v)
        output = self.out_proj(o)
        return output


class LocalSelfMultiheadAttention(torch.nn.Module):
    def __init__(self, feature_dim, attn_head, scaling_factor=1.0):
        super(LocalSelfMultiheadAttention, self).__init__()
        self.feature_dim = feature_dim
        self.attn_head = attn_head
        self.head_dim = feature_dim // attn_head
        assert feature_dim % attn_head == 0, f"feature_dim {feature_dim} must be divided by attn_head {attn_head}!"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5
        self.in_proj = SimpleLinear(self.feature_dim, self.feature_dim * 3)
        # TODO debug
        # self.out_proj = SimpleLinear(self.feature_dim, self.feature_dim)

    def forward(self, query, attn_bias: Optional[torch.Tensor]=None, nlist_mask: Optional[torch.Tensor]=None, nlist: Optional[torch.Tensor]=None, return_attn=True):
        nframes, nloc, feature_dim = query.size()
        _, _, nnei = nlist.size()
        assert feature_dim == self.feature_dim
        # [nframes, nloc, feature_dim]
        q, k, v = self.in_proj(query).chunk(3, dim=-1)
        # [nframes * attn_head * nloc, 1, head_dim]
        q = (q.view(nframes, nloc, self.attn_head, self.head_dim)
             .transpose(1, 2)
             .contiguous()
             .view(nframes * self.attn_head * nloc, 1, self.head_dim)
             * self.scaling
             )
        # [nframes, nloc, feature_dim] --> [nframes, nloc + 1, feature_dim]
        # with nlist [nframes, nloc, nnei] --> [nframes, nloc, nnei, feature_dim]
        # padding = torch.zeros(feature_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION).to(k.device)
        # k = torch.concat([k, padding.unsqueeze(0).unsqueeze(1)], dim=1)
        # v = torch.concat([v, padding.unsqueeze(0).unsqueeze(1)], dim=1)

        # [nframes, nloc * nnei, feature_dim]
        index = nlist.view(nframes, -1).unsqueeze(-1).expand(-1, -1, feature_dim)
        k = torch.gather(k, dim=1, index=index)
        # [nframes, nloc * nnei, feature_dim]
        v = torch.gather(v, dim=1, index=index)
        # [nframes * attn_head * nloc, nnei, head_dim]
        k = (k.view(nframes, nloc, nnei, self.attn_head, self.head_dim)
             .permute(0, 3, 1, 2, 4)
             .contiguous()
             .view(nframes * self.attn_head * nloc, nnei, self.head_dim))
        v = (v.view(nframes, nloc, nnei, self.attn_head, self.head_dim)
             .permute(0, 3, 1, 2, 4)
             .contiguous()
             .view(nframes * self.attn_head * nloc, nnei, self.head_dim))
        # [nframes * attn_head * nloc, 1, nnei]
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # maskfill
        # [nframes, attn_head, nloc, nnei]
        attn_weights = (attn_weights.view(nframes, self.attn_head, nloc, nnei)
                        .masked_fill(~nlist_mask.unsqueeze(1), float("-inf")))
        # add bias
        if return_attn:
            attn_weights = attn_weights + attn_bias
        # softmax
        # [nframes * attn_head * nloc, 1, nnei]
        attn = torch.nn.functional.softmax(attn_weights, dim=-1).view(nframes * self.attn_head * nloc, 1, nnei)
        # bmm
        # [nframes * attn_head * nloc, 1, head_dim]
        o = torch.bmm(attn, v)
        assert list(o.size()) == [nframes * self.attn_head * nloc, 1, self.head_dim]
        # [nframes, nloc, feature_dim]
        o = (o.view(nframes, self.attn_head, nloc, self.head_dim)
             .transpose(1, 2)
             .contiguous()
             .view(nframes, nloc, self.feature_dim)
             )
        # out
        ## TODO debug:
        # o = self.out_proj(o)
        if not return_attn:
            return o
        else:
            return o, attn_weights, attn


class EvoformerEncoderLayer(torch.nn.Module):
    def __init__(self,
                 feature_dim: int = 768,
                 ffn_dim: int = 2048,
                 attn_head: int = 8,
                 activation_fn: str = "gelu",
                 post_ln: bool = False):
        super(EvoformerEncoderLayer, self).__init__()
        self.feature_dim = feature_dim
        self.ffn_dim = ffn_dim
        self.attn_head = attn_head
        self.activation_fn = get_activation_fn(activation_fn) if activation_fn is not None else None
        self.post_ln = post_ln
        self.self_attn_layer_norm = torch.nn.LayerNorm(self.feature_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)

        self.self_attn = LocalSelfMultiheadAttention(
            self.feature_dim,
            self.attn_head,
        )
        self.final_layer_norm = torch.nn.LayerNorm(self.feature_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
        self.fc1 = SimpleLinear(self.feature_dim, self.ffn_dim)
        self.fc2 = SimpleLinear(self.ffn_dim, self.feature_dim)

    def forward(self, x, attn_bias: Optional[torch.Tensor]=None, nlist_mask: Optional[torch.Tensor]=None, nlist: Optional[torch.Tensor]=None, return_attn=True):
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            attn_bias=attn_bias,
            nlist_mask=nlist_mask,
            nlist=nlist,
            return_attn=return_attn,
        )
        if return_attn:
            x, attn_weights, attn_probs = x
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        if not return_attn:
            return x
        else:
            return x, attn_weights, attn_probs


# output: atomic_rep, transformed_atomic_rep, pair_rep, delta_pair_rep, norm_x, norm_delta_pair_rep,
class Evoformer2bEncoder(torch.nn.Module):
    def __init__(self,
                 nnei: int,
                 layer_num: int = 6,
                 attn_head: int = 8,
                 atomic_dim: int = 1024,
                 pair_dim: int = 100,
                 feature_dim: int = 1024,
                 ffn_dim: int = 2048,
                 post_ln: bool = False,
                 final_layer_norm: bool = True,
                 final_head_layer_norm: bool = False,
                 emb_layer_norm: bool = False,
                 atomic_residual: bool = False,
                 evo_residual: bool = False,
                 residual_factor: float = 1.0,
                 activation_function: str = "gelu"):
        super(Evoformer2bEncoder, self).__init__()
        self.nnei = nnei
        self.layer_num = layer_num
        self.attn_head = attn_head
        self.atomic_dim = atomic_dim
        self.pair_dim = pair_dim
        self.feature_dim = feature_dim
        self.ffn_dim = ffn_dim
        self.post_ln = post_ln
        self._final_layer_norm = final_layer_norm
        self._final_head_layer_norm = final_head_layer_norm
        self._emb_layer_norm = emb_layer_norm
        self.activation_function = activation_function
        self.evo_residual = evo_residual
        self.residual_factor = residual_factor
        if atomic_residual and atomic_dim == feature_dim:
            self.atomic_residual = True
        else:
            self.atomic_residual = False
        self.in_proj = SimpleLinear(self.atomic_dim, self.feature_dim, bavg=0., stddev=1., use_timestep=False,
                                    activate='tanh')  # TODO
        self.out_proj = SimpleLinear(self.feature_dim, self.atomic_dim, bavg=0., stddev=1., use_timestep=False,
                                     activate='tanh')
        if self._emb_layer_norm:
            self.emb_layer_norm = torch.nn.LayerNorm(self.feature_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)

        ## TODO debug : self.in_proj_pair = NonLinearHead(self.pair_dim, self.attn_head, activation_fn=None)
        self.in_proj_pair = SimpleLinear(self.pair_dim, self.attn_head, activate=None)
        evoformer_encoder_layers = []
        for i in range(self.layer_num):
            evoformer_encoder_layers.append(EvoformerEncoderLayer(
                feature_dim=self.feature_dim,
                ffn_dim=self.ffn_dim,
                attn_head=self.attn_head,
                activation_fn=self.activation_function,
                post_ln=self.post_ln)
            )
        self.evoformer_encoder_layers = torch.nn.ModuleList(evoformer_encoder_layers)
        if self._final_layer_norm:
            self.final_layer_norm = torch.nn.LayerNorm(self.feature_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
        if self._final_head_layer_norm:
            self.final_head_layer_norm = torch.nn.LayerNorm(self.attn_head, dtype=env.GLOBAL_PT_FLOAT_PRECISION)

    def forward(self, atomic_rep, pair_rep, nlist, nlist_type, nlist_mask):
        """Encoder the atomic and pair representations.

        Args:
        - atomic_rep: Atomic representation with shape [nframes, nloc, atomic_dim].
        - pair_rep: Pair representation with shape [nframes, nloc, nnei, pair_dim].
        - nlist: Neighbor list with shape [nframes, nloc, nnei].
        - nlist_type: Neighbor types with shape [nframes, nloc, nnei].
        - nlist_mask: Neighbor mask with shape [nframes, nloc, nnei], `False` if blank.

        Returns:
        - atomic_rep: Atomic representation after encoder with shape [nframes, nloc, feature_dim].
        - transformed_atomic_rep: Transformed atomic representation after encoder with shape [nframes, nloc, atomic_dim].
        - pair_rep: Pair representation after encoder with shape [nframes, nloc, nnei, attn_head].
        - delta_pair_rep: Delta pair representation after encoder with shape [nframes, nloc, nnei, attn_head].
        - norm_x: Normalization loss of atomic_rep.
        - norm_delta_pair_rep: Normalization loss of delta_pair_rep.
        """
        # Global branch
        nframes, nloc, _ = atomic_rep.size()
        nnei = pair_rep.shape[2]
        input_atomic_rep = atomic_rep
        # [nframes, nloc, feature_dim]
        if self.atomic_residual:
            atomic_rep = atomic_rep + self.in_proj(atomic_rep)
        else:
            atomic_rep = self.in_proj(atomic_rep)

        if self._emb_layer_norm:
            atomic_rep = self.emb_layer_norm(atomic_rep)

        # Local branch
        # [nframes, nloc, nnei, attn_head]
        pair_rep = self.in_proj_pair(pair_rep)
        # [nframes, attn_head, nloc, nnei]
        pair_rep = pair_rep.permute(0, 3, 1, 2).contiguous()
        input_pair_rep = pair_rep
        pair_rep = pair_rep.masked_fill(~nlist_mask.unsqueeze(1), float("-inf"))

        for i in range(self.layer_num):
            atomic_rep, pair_rep, _ = self.evoformer_encoder_layers[i](
                atomic_rep, attn_bias=pair_rep, nlist_mask=nlist_mask, nlist=nlist, return_attn=True
            )

        def norm_loss(x, eps=1e-10, tolerance=1.0):
            # x = x.float()
            max_norm = x.shape[-1] ** 0.5
            norm = torch.sqrt(torch.sum(x ** 2, dim=-1) + eps)
            error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
            return error

        def masked_mean(mask, value, dim=-1, eps=1e-10):
            return (
                    torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
            ).mean()

        # atomic_rep shape: [nframes, nloc, feature_dim]
        # pair_rep shape: [nframes, attn_head, nloc, nnei]

        norm_x = torch.mean(norm_loss(atomic_rep))
        if self._final_layer_norm:
            atomic_rep = self.final_layer_norm(atomic_rep)

        delta_pair_rep = pair_rep - input_pair_rep
        delta_pair_rep = delta_pair_rep.masked_fill(~nlist_mask.unsqueeze(1), 0)
        # [nframes, nloc, nnei, attn_head]
        delta_pair_rep = (delta_pair_rep.view(nframes, self.attn_head, nloc, nnei)
                          .permute(0, 2, 3, 1)
                          .contiguous())

        # [nframes, nloc, nnei]
        norm_delta_pair_rep = norm_loss(delta_pair_rep)
        norm_delta_pair_rep = masked_mean(mask=nlist_mask, value=norm_delta_pair_rep)
        if self._final_head_layer_norm:
            delta_pair_rep = self.final_head_layer_norm(delta_pair_rep)

        if self.atomic_residual:
            transformed_atomic_rep = atomic_rep + self.out_proj(atomic_rep)
        else:
            transformed_atomic_rep = self.out_proj(atomic_rep)

        if self.evo_residual:
            transformed_atomic_rep = (self.residual_factor * transformed_atomic_rep + input_atomic_rep) * (1/np.sqrt(2))

        return atomic_rep, transformed_atomic_rep, pair_rep, delta_pair_rep, norm_x, norm_delta_pair_rep
