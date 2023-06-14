import numpy as np
import torch

from deepmd_pt.utils import env
from deepmd_pt.model.descriptor import prod_env_mat_se_a, Descriptor, compute_std

try:
    from typing import Final
except:
    from torch.jit import Final


class DescrptHybrid(Descriptor):

    def __init__(self,
                 descriptor_list,
                 descriptor_param,
                 **kwargs):
        """Construct a hybrid descriptor.

        Args:
        - descriptor_list: list of descriptors.
        - descriptor_param: descriptor configs.
        """
        super(DescrptHybrid, self).__init__()
        self.descriptor_list = torch.nn.ModuleList(descriptor_list)
        self.descriptor_param = descriptor_param
        self.rcut = [descrpt.rcut for descrpt in self.descriptor_list]
        self.sec = [descrpt.sec for descrpt in self.descriptor_list]

    @property
    def dim_out(self):
        """
        Returns the output dimension of this descriptor
        """
        out_size = 0
        for descrpt in self.descriptor_list:
            out_size += descrpt.dim_out
        return out_size

    @property
    def dim_emb_list(self):
        """
        Returns the output dimension list of embeddings
        """
        emb_list = []
        for descrpt in self.descriptor_list:
            emb_list.append(descrpt.dim_emb)
        return emb_list

    @property
    def dim_emb(self):
        """
        Returns the output dimension of embedding
        """
        return sum(self.dim_emb_list)

    def compute_input_stats(self, merged):
        """Update mean and stddev for descriptor elements.
        """
        sumr, suma, sumn, sumr2, suma2 = [], [], [], [], []
        for ii, descrpt in enumerate(self.descriptor_list):
            merged_tmp = [{key: item[key] if not isinstance(item[key], list) else item[key][ii] for key in item} for item in merged]
            sumr_tmp, suma_tmp, sumn_tmp, sumr2_tmp, suma2_tmp = descrpt.compute_input_stats(merged_tmp)
            sumr.append(sumr_tmp)
            suma.append(suma_tmp)
            sumn.append(sumn_tmp)
            sumr2.append(sumr2_tmp)
            suma2.append(suma2_tmp)
        return sumr, suma, sumn, sumr2, suma2

    def init_desc_stat(self, sumr, suma, sumn, sumr2, suma2):
        for ii, descrpt in enumerate(self.descriptor_list):
            descrpt.init_desc_stat(sumr[ii], suma[ii], sumn[ii], sumr2[ii], suma2[ii])

    def forward(self, extended_coord, selected, atype, selected_type, selected_loc=None, atype_tebd=None, nlist_tebd=None):
        """Calculate decoded embedding for each atom.

        Args:
        - extended_coord: Tell atom coordinates with shape [nframes, natoms[1]*3].
        - selected: Tell atom types with shape [nframes, natoms[1]].
        - atype: Tell atom count and element count. Its shape is [2+self.ntypes].
        - selected_type: Tell simulation box with shape [nframes, 9].
        - atype_tebd: Tell simulation box with shape [nframes, 9].
        - nlist_tebd: Tell simulation box with shape [nframes, 9].

        Returns:
        - result: descriptor with shape [nframes, nloc, self.filter_neuron[-1] * self.axis_neuron].
        - ret: environment matrix with shape [nframes, nloc, self.neei, out_size]
        """
        out_descriptor = []
        # out_env_mat = []
        out_rot_mat = []
        # out_diff = []
        for ii, descrpt in enumerate(self.descriptor_list):
            descriptor, env_mat, diff, rot_mat = descrpt(extended_coord, selected[ii], atype, selected_type[ii],
                                                         selected_loc=selected_loc[ii], atype_tebd=atype_tebd,
                                                         nlist_tebd=nlist_tebd[ii])
            out_descriptor.append(descriptor)
            # out_env_mat.append(env_mat)
            # out_diff.append(diff)
            out_rot_mat.append(rot_mat)
        out_descriptor = torch.concat(out_descriptor, dim=-1)
        out_rot_mat = torch.concat(out_rot_mat, dim=-2)
        return out_descriptor, None, None, out_rot_mat
