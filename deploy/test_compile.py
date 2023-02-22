from deepmd_pt.descriptor import smoothDescriptor, make_env_mat, Region3D
import numpy as np
from deepmd_pt.env import *
from deepmd_pt.dataset import DeepmdDataSet
import os

rcut = 6.
rcut_smth = 0.5
sel = [46, 92]

sec = np.cumsum(sel)
ntypes = len(sel)
nnei = sum(sel)
CUR_DIR = "tests/"
ntypes=2
ds = DeepmdDataSet([
    os.path.join(CUR_DIR, 'water/data/data_0'),
    os.path.join(CUR_DIR, 'water/data/data_1'),
    os.path.join(CUR_DIR, 'water/data/data_2')
], 2, ['O', 'H'])
np_batch, batch = ds.get_batch(pt=True)
coord = batch['coord']
coord.requires_grad_(True)
atype = batch['type']
box = batch['box']
sid = 0

def func(box):
    box = box.view(3, 3)
    return torch.cross(box[0], box[1])
func = torch.compile(func, dynamic=True, box=box)
tmp = func(box[sid])

"""
make_env_mat = torch.compile(make_env_mat, dynamic=True)
selected, merged_coord, merged_mapping = make_env_mat(coord[sid].view(-1,3), atype[sid], box[sid], rcut, sec)
"""

'''
# smooth descriptor
smoothDescriptor = torch.compile(smoothDescriptor, dynamic=True)
avg_zero = torch.zeros([ntypes, nnei*4], dtype=GLOBAL_PT_FLOAT_PRECISION)
std_ones = torch.ones([ntypes, nnei*4], dtype=GLOBAL_PT_FLOAT_PRECISION)
my_d = smoothDescriptor(
    coord.to(DEVICE),
    batch['type'],
    batch['natoms_vec'],
    batch['box'],
    avg_zero.reshape([-1, nnei, 4]).to(DEVICE),
    std_ones.reshape([-1, nnei, 4]).to(DEVICE),
    rcut,
    rcut_smth,
    sec
)
'''