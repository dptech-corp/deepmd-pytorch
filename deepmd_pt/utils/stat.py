import numpy as np
import torch
from collections import defaultdict
from deepmd_pt.utils.dataloader import BufferedIterator
from deepmd_pt.utils import env


def make_stat_input(datasets, dataloaders, nbatches):
    """Pack data for statistics.

    Args:
    - dataset: A list of dataset to analyze.
    - nbatches: Batch count for collecting stats.

    Returns:
    - a list of dicts, each of which contains data from a system
    """
    lst = []
    keys = [
        "coord",
        "force",
        "energy",
        "atype",
        "natoms",
        "mapping",
        "selected",
        "selected_loc",
        "selected_type",
        "shift",
    ]
    if datasets[0].mixed_type:
        keys.append("real_natoms_vec")
    for i in range(len(datasets)):
        sys_stat = {key: [] for key in keys}
        iterator = iter(dataloaders[i])
        for _ in range(nbatches):
            try:
                stat_data = next(iterator)
            except StopIteration:
                iterator = iter(dataloaders[i])
                stat_data = next(iterator)
            for dd in stat_data:
                if dd in keys:
                    sys_stat[dd].append(stat_data[dd])
        for key in keys:
            if key == "mapping" or key == "shift":
                extend = max(d.shape[1] for d in sys_stat[key])
                for jj in range(len(sys_stat[key])):
                    l = []
                    item = sys_stat[key][jj]
                    for ii in range(item.shape[0]):
                        l.append(item[ii])
                    n_frames = len(item)
                    if key == "shift":
                        shape = torch.zeros(
                            (n_frames, extend, 3),
                            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
                            device=env.PREPROCESS_DEVICE,
                        )
                    else:
                        shape = torch.zeros(
                            (n_frames, extend),
                            dtype=torch.long,
                            device=env.PREPROCESS_DEVICE,
                        )
                    for i in range(len(item)):
                        natoms_tmp = l[i].shape[0]
                        shape[i, :natoms_tmp] = l[i]
                    sys_stat[key][jj] = shape           
            sys_stat[key] = torch.cat(sys_stat[key], dim=0)
        lst.append(sys_stat)
    return lst


def compute_output_stats(energy, natoms, rcond=1e-3):
    """Update mean and stddev for descriptor elements.

    Args:
    - energy: Batched energy with shape [nframes, 1].
    - natoms: Batched atom statisics with shape [self.ntypes+2].

    Returns:
    - energy_coef: Average enery per atom for each element.
    """
    for i in range(len(energy)):
        energy[i] = energy[i].mean(dim=0, keepdim=True)
        natoms[i] = natoms[i].double().mean(dim=0, keepdim=True)
    sys_ener = torch.cat(energy).cpu()
    sys_tynatom = torch.cat(natoms)[:, 2:].cpu()
    energy_coef, _, _, _ = np.linalg.lstsq(sys_tynatom, sys_ener, rcond)
    return energy_coef
