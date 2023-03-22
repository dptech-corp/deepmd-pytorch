import numpy as np
import torch

from collections import defaultdict

def make_stat_input(dataset, nbatches):
    '''Pack data for statistics.

    Args:
    - dataset: The dataset to analyze.
    - nbatches: Batch count for collecting stats.

    Returns:
    - a list of dicts, each of which contains data from a system
    '''
    lst = []
    for ii in range(dataset.nsystems):
        sys_stat = defaultdict(list)
        for _ in range(nbatches):
            stat_data = dataset[ii]
            for dd in stat_data:
                sys_stat[dd].append(stat_data[dd])
        for key in ['coord', 'box', 'force', 'energy', 'atype', 'natoms', 'mapping', 'selected', 'shift']:
            sys_stat[key] = torch.cat(sys_stat[key], dim=0)
        lst.append(sys_stat)
    return lst

def compute_output_stats(energy, natoms, rcond=1e-3):
    '''Update mean and stddev for descriptor elements.

    Args:
    - energy: Batched energy with shape [nframes, 1].
    - natoms: Batched atom statisics with shape [self.ntypes+2].

    Returns:
    - energy_coef: Average enery per atom for each element.
    '''
    sys_ener = torch.cat(energy).cpu()
    sys_tynatom = torch.cat(natoms)[:, 2:].cpu()
    energy_coef, _, _, _ = np.linalg.lstsq(sys_tynatom, sys_ener, rcond)
    return energy_coef
