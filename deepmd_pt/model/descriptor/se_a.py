import numpy as np
import torch

from deepmd_pt.utils import env
from deepmd_pt.model.descriptor import prod_env_mat_se_a, Descriptor, compute_std

try:
    from typing import Final
except:
    from torch.jit import Final

from deepmd_pt.model.network import TypeFilter


class DescrptSeA(Descriptor):
    ndescrpt: Final[int]
    __constants__ = ['ndescrpt']

    def __init__(self,
                 rcut,
                 rcut_smth,
                 sel,
                 neuron=[25, 50, 100],
                 axis_neuron=16,
                 set_davg_zero: bool = False,
                 **kwargs):
        """Construct an embedding net of type `se_a`.

        Args:
        - rcut: Cut-off radius.
        - rcut_smth: Smooth hyper-parameter for pair force & energy.
        - sel: For each element type, how many atoms is selected as neighbors.
        - filter_neuron: Number of neurons in each hidden layers of the embedding net.
        - axis_neuron: Number of columns of the sub-matrix of the embedding matrix.
        """
        super(DescrptSeA, self).__init__()
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.filter_neuron = neuron
        self.axis_neuron = axis_neuron
        self.set_davg_zero = set_davg_zero

        self.ntypes = len(sel)  # 元素数量
        self.sec = torch.cumsum(torch.tensor(sel), dim=0)  # 每种元素在邻居中的位移
        self.nnei = sum(sel)  # 总的邻居数量
        self.ndescrpt = self.nnei * 4  # 描述符的元素数量

        wanted_shape = (self.ntypes, self.nnei, 4)
        mean = torch.zeros(wanted_shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
        stddev = torch.ones(wanted_shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
        self.register_buffer('mean', mean)
        self.register_buffer('stddev', stddev)

        filter_layers = []
        start_index = 0
        for type_i in range(self.ntypes):
            one = TypeFilter(start_index, sel[type_i], self.filter_neuron)
            filter_layers.append(one)
            start_index += sel[type_i]
        self.filter_layers = torch.nn.ModuleList(filter_layers)

    @property
    def dim_out(self):
        """
        Returns the output dimension of this descriptor
        """
        return self.filter_neuron[-1] * self.axis_neuron

    def compute_input_stats(self, merged):
        """Update mean and stddev for descriptor elements.
        """
        sumr = []
        suma = []
        sumn = []
        sumr2 = []
        suma2 = []
        for system in merged:  # 逐个 system 的分析
            index = system['mapping'].unsqueeze(-1).expand(-1, -1, 3)
            extended_coord = torch.gather(system['coord'], dim=1, index=index)
            extended_coord = extended_coord - system['shift']
            env_mat, _ = prod_env_mat_se_a(
                extended_coord, system['selected'], system['atype'],
                self.mean, self.stddev,
                self.rcut, self.rcut_smth,
            )
            sysr, sysr2, sysa, sysa2, sysn = analyze_descrpt(env_mat.detach().cpu().numpy(), self.ndescrpt,
                                                             system['natoms'])
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

    def forward(self, extended_coord, selected, atype, selected_type=None, selected_loc=None, atype_tebd=None, nlist_tebd=None):
        """Calculate decoded embedding for each atom.

        Args:
        - coord: Tell atom coordinates with shape [nframes, natoms[1]*3].
        - atype: Tell atom types with shape [nframes, natoms[1]].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].
        - box: Tell simulation box with shape [nframes, 9].

        Returns:
        - `torch.Tensor`: descriptor matrix with shape [nframes, natoms[0]*self.filter_neuron[-1]*self.axis_neuron].
        """
        nall = selected.shape[1]
        dmatrix, _ = prod_env_mat_se_a(
            extended_coord, selected, atype,
            self.mean, self.stddev,
            self.rcut, self.rcut_smth,
        )
        dmatrix = dmatrix.view(-1, self.ndescrpt)  # shape is [nframes*nall, self.ndescrpt]
        xyz_scatter = torch.empty(1, )

        ret = self.filter_layers[0](dmatrix)
        xyz_scatter = ret

        for ii, transform in enumerate(self.filter_layers[1:]):
            # shape is [nframes*nall, 4, self.filter_neuron[-1]]
            ret = transform.forward(dmatrix)
            xyz_scatter = xyz_scatter + ret

        xyz_scatter /= self.nnei
        xyz_scatter_1 = xyz_scatter.permute(0, 2, 1)
        xyz_scatter_2 = xyz_scatter[:, :, 0:self.axis_neuron]
        result = torch.matmul(xyz_scatter_1,
                              xyz_scatter_2)  # shape is [nframes*nall, self.filter_neuron[-1], self.axis_neuron]
        return result.view(-1, nall, self.filter_neuron[-1] * self.axis_neuron)


def analyze_descrpt(matrix, ndescrpt, natoms):
    """Collect avg, square avg and count of descriptors in a batch."""
    ntypes = natoms.shape[1] - 2
    start_index = 0
    sysr = []  # 每类元素的径向均值
    sysa = []  # 每类元素的轴向均值
    sysn = []  # 每类元素的样本数量
    sysr2 = []  # 每类元素的径向平方均值
    sysa2 = []  # 每类元素的轴向平方均值
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
    return sysr, sysr2, sysa, sysa2, sysn
