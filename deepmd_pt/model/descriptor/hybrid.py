import numpy as np
import torch

from deepmd_pt.utils import env
from deepmd_pt.model.descriptor import prod_env_mat_se_a, Descriptor, compute_std
from deepmd_pt.model.network import Identity, Linear
from typing import Optional, List

try:
    from typing import Final
except:
    from torch.jit import Final


@Descriptor.register("hybrid")
class DescrptHybrid(Descriptor):

    def __init__(self,
                 list,
                 ntypes: int,
                 tebd_dim: int = 8,
                 tebd_input_mode: str = 'concat',
                 hybrid_mode: str = "concat",
                 **kwargs):
        """Construct a hybrid descriptor.

        Args:
        - descriptor_list: list of descriptors.
        - descriptor_param: descriptor configs.
        """
        super(DescrptHybrid, self).__init__()
        supported_descrpt = ['se_atten', 'se_uni']
        descriptor_list = []
        for descriptor_param_item in list:
            descriptor_type_tmp = descriptor_param_item['type']
            assert descriptor_type_tmp in supported_descrpt, \
                f'Only descriptors in {supported_descrpt} are supported for `hybrid` descriptor!'
            descriptor_param_item['ntypes'] = ntypes
            descriptor_param_item['tebd_dim'] = tebd_dim
            descriptor_param_item['tebd_input_mode'] = tebd_input_mode
            descriptor_list.append(Descriptor(**descriptor_param_item))
        self.descriptor_list = torch.nn.ModuleList(descriptor_list)
        self.descriptor_param = list
        self.rcut = [descrpt.rcut for descrpt in self.descriptor_list]
        self.sec = [descrpt.sec for descrpt in self.descriptor_list]
        self.sel = [descrpt.sel for descrpt in self.descriptor_list]
        self.local_cluster_list = [descrpt.local_cluster for descrpt in self.descriptor_list]
        self.local_cluster = True in self.local_cluster_list
        self.hybrid_mode = hybrid_mode
        self.tebd_dim = tebd_dim
        assert self.hybrid_mode in ["concat", "sequential"]
        if self.hybrid_mode == "sequential":
            sequential_transform = []
            for ii in range(len(descriptor_list) - 1):
                if descriptor_list[ii].dim_out == descriptor_list[ii + 1].dim_in:
                    sequential_transform.append(Identity())
                else:
                    sequential_transform.append(Linear(descriptor_list[ii].dim_out, descriptor_list[ii + 1].dim_in,
                                                       bias=False, init="glorot"))
            sequential_transform.append(Identity())
            self.sequential_transform = torch.nn.ModuleList(sequential_transform)

    @property
    def dim_out(self):
        """
        Returns the output dimension of this descriptor
        """
        if self.hybrid_mode == "concat":
            return sum([descrpt.dim_out for descrpt in self.descriptor_list])
        elif self.hybrid_mode == "sequential":
            return self.descriptor_list[-1].dim_out
        else:
            raise RuntimeError

    @property
    def dim_emb_list(self) -> List[int]:
        """
        Returns the output dimension list of embeddings
        """
        return [descrpt.dim_emb for descrpt in self.descriptor_list]

    @property
    def dim_emb(self):
        """
        Returns the output dimension of embedding
        """
        if self.hybrid_mode == "concat":
            return sum(self.dim_emb_list)
        elif self.hybrid_mode == "sequential":
            return self.descriptor_list[-1].dim_emb
        else:
            raise RuntimeError

    def share_params(self, base_class, shared_level, resume=False):
        assert self.__class__ == base_class.__class__, "Only descriptors of the same type can share params!"
        if shared_level == 0:
            for ii, des in enumerate(self.descriptor_list):
                self.descriptor_list[ii].share_params(base_class.descriptor_list[ii], shared_level, resume=resume)
            if self.hybrid_mode == "sequential":
                self.sequential_transform = base_class.sequential_transform
        else:
            raise NotImplementedError

    def compute_input_stats(self, nbatch, merged):
        """Update mean and stddev for descriptor elements.
        """
        sumr, suma, sumn, sumr2, suma2, energy_coef = [], [], [], [], []
        for ii, descrpt in enumerate(self.descriptor_list):
            sumr_tmp, suma_tmp, sumn_tmp, sumr2_tmp, suma2_tmp,energy_coef_tmp = descrpt.compute_input_stats(nbatch. merged, desc_index=ii)
            sumr.append(sumr_tmp)
            suma.append(suma_tmp)
            sumn.append(sumn_tmp)
            sumr2.append(sumr2_tmp)
            suma2.append(suma2_tmp)
            energy_coef.append(energy_coef_tmp)
        return sumr, suma, sumn, sumr2, suma2, energy_coef

    def init_desc_stat(self, sumr, suma, sumn, sumr2, suma2):
        for ii, descrpt in enumerate(self.descriptor_list):
            descrpt.init_desc_stat(sumr[ii], suma[ii], sumn[ii], sumr2[ii], suma2[ii])

    def forward(
            self,
            extended_coord: torch.Tensor,
            nlist: List[torch.Tensor],
            atype: torch.Tensor,
            nlist_type: List[torch.Tensor],
            nlist_loc: List[Optional[torch.Tensor]] = None,
            atype_tebd: Optional[torch.Tensor] = None,
            nlist_tebd: List[Optional[torch.Tensor]] = None):
        """Calculate decoded embedding for each atom.

        Args:
        - extended_coord: Tell atom coordinates with shape [nframes, natoms[1]*3].
        - nlist: Tell atom types with shape [nframes, natoms[1]].
        - atype: Tell atom count and element count. Its shape is [2+self.ntypes].
        - nlist_type: Tell simulation box with shape [nframes, 9].
        - atype_tebd: Tell simulation box with shape [nframes, 9].
        - nlist_tebd: Tell simulation box with shape [nframes, 9].

        Returns:
        - result: descriptor with shape [nframes, nloc, self.filter_neuron[-1] * self.axis_neuron].
        - ret: environment matrix with shape [nframes, nloc, self.neei, out_size]
        """
        nframes, nloc = atype.shape[:2]
        if self.hybrid_mode == 'concat':
            out_descriptor = []
            # out_env_mat = []
            out_rot_mat = []
            # out_diff = []
            for ii, descrpt in enumerate(self.descriptor_list):
                descriptor, env_mat, diff, rot_mat = descrpt(extended_coord, nlist[ii], atype, nlist_type[ii],
                                                             nlist_loc=nlist_loc[ii], atype_tebd=atype_tebd,
                                                             nlist_tebd=nlist_tebd[ii])
                if descriptor.shape[0] == nframes * nloc:
                    # [nframes * nloc, 1 + nnei, emb_dim]
                    descriptor = descriptor[:, 0, :].reshape(nframes, nloc, -1)
                out_descriptor.append(descriptor)
                # out_env_mat.append(env_mat)
                # out_diff.append(diff)
                out_rot_mat.append(rot_mat)
            out_descriptor = torch.concat(out_descriptor, dim=-1)
            # if None not in out_rot_mat:
            #     out_rot_mat = torch.concat(out_rot_mat, dim=-2)
            # else:
            out_rot_mat = None
            return out_descriptor, None, None, out_rot_mat
        elif self.hybrid_mode == 'sequential':
            seq_input = None
            env_mat, diff, rot_mat = None, None, None
            for ii, (descrpt, seq_transform) in enumerate(zip(self.descriptor_list, self.sequential_transform)):
                seq_output, env_mat, diff, rot_mat = descrpt(extended_coord, nlist[ii], atype, nlist_type[ii],
                                                             nlist_loc=nlist_loc[ii], atype_tebd=atype_tebd,
                                                             nlist_tebd=nlist_tebd[ii], seq_input=seq_input)
                seq_input = seq_transform(seq_output)
            return seq_input, env_mat, diff, rot_mat
        else:
            raise RuntimeError
