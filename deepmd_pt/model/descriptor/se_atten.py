import numpy as np
import torch

from deepmd_pt.utils import env
from deepmd_pt.model.descriptor import prod_env_mat_se_a, Descriptor, compute_std

try:
    from typing import Final
except:
    from torch.jit import Final

from deepmd_pt.model.network import TypeFilter, NeighborWiseAttention


class DescrptSeAtten(Descriptor):

    def __init__(self,
                 rcut,
                 rcut_smth,
                 sel,
                 ntypes: int,
                 neuron: list = [25, 50, 100],
                 axis_neuron: int = 16,
                 tebd_dim: int = 8,
                 tebd_input_mode: str = 'concat',
                 # set_davg_zero: bool = False,
                 set_davg_zero: bool = True, # TODO
                 attn: int = 128,
                 attn_layer: int = 2,
                 attn_dotr: bool = True,
                 attn_mask: bool = False,
                 post_ln=True,
                 ffn=False,
                 ffn_embed_dim=1024,
                 activation="tanh",
                 scaling_factor=1.0,
                 head_num=1,
                 normalize=True,
                 temperature=None,
                 return_rot=False,
                 **kwargs):
        """Construct an embedding net of type `se_atten`.

        Args:
        - rcut: Cut-off radius.
        - rcut_smth: Smooth hyper-parameter for pair force & energy.
        - sel: For each element type, how many atoms is selected as neighbors.
        - filter_neuron: Number of neurons in each hidden layers of the embedding net.
        - axis_neuron: Number of columns of the sub-matrix of the embedding matrix.
        """
        super(DescrptSeAtten, self).__init__()
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.filter_neuron = neuron
        self.axis_neuron = axis_neuron
        self.tebd_dim = tebd_dim
        self.tebd_input_mode = tebd_input_mode
        self.set_davg_zero = set_davg_zero
        self.attn_dim = attn
        self.attn_layer = attn_layer
        self.attn_dotr = attn_dotr
        self.attn_mask = attn_mask
        self.post_ln = post_ln
        self.ffn = ffn
        self.ffn_embed_dim = ffn_embed_dim
        self.activation = activation
        self.scaling_factor = scaling_factor
        self.head_num = head_num
        self.normalize = normalize
        self.temperature = temperature
        self.return_rot = return_rot

        if isinstance(sel, int):
            sel = [sel]

        self.ntypes = ntypes
        self.sec = torch.tensor(sel)  # 每种元素在邻居中的位移
        self.nnei = sum(sel)  # 总的邻居数量
        self.ndescrpt = self.nnei * 4  # 描述符的元素数量
        self.dpa1_attention = NeighborWiseAttention(self.attn_layer, self.nnei, self.filter_neuron[-1], self.attn_dim,
                                                    dotr=self.attn_dotr, do_mask=self.attn_mask, post_ln=self.post_ln,
                                                    ffn=self.ffn, ffn_embed_dim=self.ffn_embed_dim,
                                                    activation=self.activation, scaling_factor=self.scaling_factor,
                                                    head_num=self.head_num, normalize=self.normalize,
                                                    temperature=self.temperature)

        wanted_shape = (self.ntypes, self.nnei, 4)
        mean = torch.zeros(wanted_shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
        stddev = torch.ones(wanted_shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
        self.register_buffer('mean', mean)
        self.register_buffer('stddev', stddev)

        filter_layers = []
        one = TypeFilter(0, self.nnei, self.filter_neuron, return_G=True, tebd_dim=self.tebd_dim, use_tebd=True,
                         tebd_mode=self.tebd_input_mode)
        filter_layers.append(one)
        self.filter_layers = torch.nn.ModuleList(filter_layers)

    @property
    def dim_out(self):
        """
        Returns the output dimension of this descriptor
        """
        return self.filter_neuron[-1] * self.axis_neuron

    @property
    def dim_emb(self):
        """
        Returns the output dimension of embedding
        """
        return self.filter_neuron[-1]

    def compute_input_stats(self, merged):
        """Update mean and stddev for descriptor elements.
        """
        sumr = []
        suma = []
        sumn = []
        sumr2 = []
        suma2 = []
        mixed_type = 'real_natoms_vec' in merged[0]
        for system in merged:  # 逐个 system 的分析
            index = system['mapping'].unsqueeze(-1).expand(-1, -1, 3)
            extended_coord = torch.gather(system['coord'], dim=1, index=index)
            extended_coord = extended_coord - system['shift']
            env_mat, _ = prod_env_mat_se_a(
                extended_coord, system['selected'], system['atype'],
                self.mean, self.stddev,
                self.rcut, self.rcut_smth,
            )
            if not mixed_type:
                sysr, sysr2, sysa, sysa2, sysn = analyze_descrpt(env_mat.detach().cpu().numpy(), self.ndescrpt,
                                                                 system['natoms'])
            else:
                sysr, sysr2, sysa, sysa2, sysn = analyze_descrpt(env_mat.detach().cpu().numpy(), self.ndescrpt,
                                                                 system['real_natoms_vec'], mixed_type=mixed_type,
                                                                 real_atype=system['atype'].detach().cpu().numpy())
            sumr.append(sysr)
            suma.append(sysa)
            sumn.append(sysn)
            sumr2.append(sysr2)
            suma2.append(sysa2)
        sumr = np.sum(sumr, axis=0)
        suma = np.sum(suma, axis=0)
        sumn = np.sum(sumn, axis=0)
        sumr2 = np.sum(sumr2, axis=0)
        suma2 = np.sum(suma2, axis=0)
        return sumr, suma, sumn, sumr2, suma2

    def init_desc_stat(self, sumr, suma, sumn, sumr2, suma2):
        all_davg = []
        all_dstd = []
        for type_i in range(self.ntypes):
            davgunit = [[sumr[type_i] / (sumn[type_i] + 1e-15), 0, 0, 0]]
            dstdunit = [[
                compute_std(sumr2[type_i], sumr[type_i], sumn[type_i], self.rcut),
                compute_std(suma2[type_i], suma[type_i], sumn[type_i], self.rcut),
                compute_std(suma2[type_i], suma[type_i], sumn[type_i], self.rcut),
                compute_std(suma2[type_i], suma[type_i], sumn[type_i], self.rcut)
            ]]
            davg = np.tile(davgunit, [self.nnei, 1])
            dstd = np.tile(dstdunit, [self.nnei, 1])
            all_davg.append(davg)
            all_dstd.append(dstd)
        if not self.set_davg_zero:
            mean = np.stack(all_davg)
            self.mean.copy_(torch.tensor(mean, device=env.DEVICE))
        stddev = np.stack(all_dstd)
        self.stddev.copy_(torch.tensor(stddev, device=env.DEVICE))

    def forward(self, extended_coord, selected, atype, selected_type, selected_loc=None, atype_tebd=None, nlist_tebd=None):
        """Calculate decoded embedding for each atom.

        Args:
        - coord: Tell atom coordinates with shape [nframes, natoms[1]*3].
        - atype: Tell atom types with shape [nframes, natoms[1]].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].
        - box: Tell simulation box with shape [nframes, 9].

        Returns:
        - result: descriptor with shape [nframes, nloc, self.filter_neuron[-1] * self.axis_neuron].
        - ret: environment matrix with shape [nframes, nloc, self.neei, out_size]
        """
        nloc = selected.shape[1]
        dmatrix, diff = prod_env_mat_se_a(
            extended_coord, selected, atype,
            self.mean, self.stddev,
            self.rcut, self.rcut_smth,
        )
        dmatrix = dmatrix.view(-1, self.ndescrpt)  # shape is [nframes*nall, self.ndescrpt]


        ret = self.filter_layers[0](dmatrix, atype_tebd=atype_tebd.unsqueeze(2).expand(-1, -1, self.nnei, -1),
                                    nlist_tebd=nlist_tebd)  # shape is [nframes*nall, self.neei, out_size]
        input_r = torch.nn.functional.normalize(dmatrix.reshape(-1, self.nnei, 4)[:, :, 1:4], dim=-1)
        nei_mask = selected_type != self.ntypes
        ret = self.dpa1_attention(ret, nei_mask, input_r)  # shape is [nframes*nloc, self.neei, out_size]
        inputs_reshape = dmatrix.view(-1, self.nnei, 4).permute(0, 2, 1)  # shape is [nframes*natoms[0], 4, self.neei]
        xyz_scatter = torch.matmul(inputs_reshape, ret)  # shape is [nframes*natoms[0], 4, out_size]
        xyz_scatter = xyz_scatter / self.nnei
        xyz_scatter_1 = xyz_scatter.permute(0, 2, 1)
        rot_mat = xyz_scatter_1[:, :, 1:4]
        xyz_scatter_2 = xyz_scatter[:, :, 0:self.axis_neuron]
        result = torch.matmul(xyz_scatter_1,
                              xyz_scatter_2)  # shape is [nframes*nloc, self.filter_neuron[-1], self.axis_neuron]
        return result.view(-1, nloc, self.filter_neuron[-1] * self.axis_neuron), \
               ret.view(-1, nloc, self.nnei, self.filter_neuron[-1]), diff, \
               rot_mat.view(-1, self.filter_neuron[-1], 3)


def analyze_descrpt(matrix, ndescrpt, natoms, mixed_type=False, real_atype=None):
    """Collect avg, square avg and count of descriptors in a batch."""
    ntypes = natoms.shape[1] - 2
    if not mixed_type:
        sysr = []  # 每类元素的径向均值
        sysa = []  # 每类元素的轴向均值
        sysn = []  # 每类元素的样本数量
        sysr2 = []  # 每类元素的径向平方均值
        sysa2 = []  # 每类元素的轴向平方均值
        start_index = 0
        for type_i in range(ntypes):
            end_index = start_index + natoms[0, 2 + type_i]
            dd = matrix[:, start_index:end_index]  # 本元素所有原子的 descriptor
            start_index = end_index
            dd = np.reshape(dd, [-1, 4])  # Shape is [nframes*natoms[2+type_id]*self.nnei, 4]
            ddr = dd[:, :1]  # 径向值
            dda = dd[:, 1:]  # XYZ 轴分量
            sumr = np.sum(ddr)
            suma = np.sum(dda) / 3.
            sumn = dd.shape[0]  # Value is nframes*natoms[2+type_id]*self.nnei
            sumr2 = np.sum(np.multiply(ddr, ddr))
            suma2 = np.sum(np.multiply(dda, dda)) / 3.
            sysr.append(sumr)
            sysa.append(suma)
            sysn.append(sumn)
            sysr2.append(sumr2)
            sysa2.append(suma2)
    else:
        sysr = [0.0 for i in range(ntypes)]
        sysa = [0.0 for i in range(ntypes)]
        sysn = [0 for i in range(ntypes)]
        sysr2 = [0.0 for i in range(ntypes)]
        sysa2 = [0.0 for i in range(ntypes)]
        for frame_item in range(matrix.shape[0]):
            dd_ff = matrix[frame_item]
            atype_frame = real_atype[frame_item]
            for type_i in range(ntypes):
                type_idx = atype_frame == type_i
                dd = dd_ff[type_idx]
                dd = np.reshape(dd, [-1, 4])  # typen_atoms * nnei, 4
                ddr = dd[:, :1]
                dda = dd[:, 1:]
                sumr = np.sum(ddr)
                suma = np.sum(dda) / 3.0
                sumn = dd.shape[0]
                sumr2 = np.sum(np.multiply(ddr, ddr))
                suma2 = np.sum(np.multiply(dda, dda)) / 3.0
                sysr[type_i] += sumr
                sysa[type_i] += suma
                sysn[type_i] += sumn
                sysr2[type_i] += sumr2
                sysa2[type_i] += suma2

    return sysr, sysr2, sysa, sysa2, sysn
