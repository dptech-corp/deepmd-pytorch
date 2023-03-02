import logging
import numpy as np
import torch

from collections import namedtuple

from deepmd_pt import env

class Region3D(object):

    def __init__(self, boxt):
        '''Construct a simulation box.'''
        boxt = boxt.reshape([3, 3])
        self.boxt = boxt.permute(1, 0)  # 用于世界坐标转内部坐标
        self.rec_boxt = torch.linalg.inv(self.boxt)  # 用于内部坐标转世界坐标

        # 计算空间属性
        self.volume = torch.linalg.det(self.boxt)  # 平行六面体空间的体积
        c_yz = torch.cross(boxt[1], boxt[2])
        self._h2yz = self.volume / torch.linalg.norm(c_yz)
        c_zx = torch.cross(boxt[2], boxt[0])
        self._h2zx = self.volume / torch.linalg.norm(c_zx)
        c_xy = torch.cross(boxt[0], boxt[1])
        self._h2xy = self.volume / torch.linalg.norm(c_xy)

    def phys2inter(self, coord):
        '''Convert physical coordinates to internal ones.'''
        return coord@self.rec_boxt

    def inter2phys(self, coord):
        '''Convert internal coordinates to physical ones.'''
        return coord@self.boxt

    def get_face_distance(self):
        '''Return face distinces to each surface of YZ, ZX, XY.'''
        return torch.stack([self._h2yz, self._h2zx, self._h2xy])


def normalize_coord(coord, region: Region3D, nloc: int):
    '''Move outer atoms into region by mirror.

    Args:
    - coord: shape is [nloc*3]
    '''
    tmp_coord = coord.clone()
    inter_cood = region.phys2inter(tmp_coord) % 1.0
    tmp_coord = region.inter2phys(inter_cood)
    return tmp_coord


def compute_serial_cid(cell_offset, ncell):
    '''Tell the sequential cell ID in its 3D space.

    Args:
    - cell_offset: shape is [3]
    - ncell: shape is [3]
    '''

    cell_offset[:, 0] *= ncell[1]*ncell[2]
    cell_offset[:, 1] *= ncell[2]
    return cell_offset.sum(-1)

def compute_pbc_shift(cell_offset, ncell):
    '''Tell shift count to move the atom into region.'''
    shift = torch.zeros_like(cell_offset)
    shift = shift + (cell_offset < 0)
    shift = shift + -(cell_offset >= ncell).to(torch.long)
    assert torch.all(cell_offset + ncell > 0)
    assert torch.all(cell_offset - ncell < ncell)
    return shift


def build_inside_clist(coord, region: Region3D, ncell):
    '''Build cell list on atoms inside region.

    Args:
    - coord: shape is [nloc*3]
    - ncell: shape is [3]
    '''
    loc_ncell = int(torch.prod(ncell))  # 模拟区域内的 Cell 数量
    nloc = coord.numel() // 3  # 原子数量
    inter_cell_size = 1. / ncell

    inter_cood = region.phys2inter(coord.view(-1, 3))
    cell_offset = torch.floor(inter_cood / inter_cell_size).to(torch.long)
    delta = cell_offset - ncell
    a2c = compute_serial_cid(cell_offset, ncell) # cell id of atoms
    arange = torch.arange(0, loc_ncell, 1)
    cellid = (a2c == arange.unsqueeze(-1)) # one hot cellid
    c2a = cellid.nonzero()
    lst = []
    cnt = 0
    bincount = torch.bincount(a2c)
    for i in range(loc_ncell):
        n = bincount[i]
        lst.append(c2a[cnt: cnt+n, 1])
        cnt += n
    return a2c, lst


def append_neighbors(coord, region: Region3D, atype, rcut: float):
    '''Make ghost atoms who are valid neighbors.

    Args:
    - coord: shape is [nloc*3]
    - atype: shape is [nloc]
    '''
    to_face = region.get_face_distance()

    # 计算 3 个方向的 Cell 大小和 Cell 数量
    ncell = torch.floor(to_face/rcut).to(torch.long)
    ncell[ncell == 0] = 1  # 模拟区域内的 Cell 数量
    cell_size = to_face / ncell
    ngcell = torch.floor(rcut / cell_size).to(torch.long) + 1  # 模拟区域外的 Cell 数量，存储的是 Ghost 原子
    expanded = cell_size * ngcell

    # 借助 Cell 列表添加边界外的 Ghost 原子
    a2c, c2a = build_inside_clist(coord, region, ncell)
    xi = torch.arange(-ngcell[0], ncell[0]+ngcell[0], 1)
    yi = torch.arange(-ngcell[1], ncell[1]+ngcell[1], 1)
    zi = torch.arange(-ngcell[2], ncell[2]+ngcell[2], 1)
    xyz = xi.view(-1, 1, 1, 1) * torch.tensor([1, 0, 0], dtype=torch.long)
    xyz = xyz + yi.view(1, -1, 1, 1) * torch.tensor([0, 1, 0], dtype=torch.long)
    xyz = xyz + zi.view(1, 1, -1, 1) * torch.tensor([0, 0, 1], dtype=torch.long)
    xyz = xyz.view(-1, 3)
    mask_a = (xyz >= 0).all(dim=-1)
    mask_b = (xyz<ncell).all(dim=-1)
    mask = ~torch.logical_and(mask_a, mask_b)
    xyz = xyz[mask] # cell coord
    shift = compute_pbc_shift(xyz, ncell)
    coord_shift = region.inter2phys(shift.to(env.GLOBAL_PT_FLOAT_PRECISION))
    mirrored = shift*ncell + xyz
    cid = compute_serial_cid(mirrored, ncell)

    n_atoms = coord.shape[0]
    aid = [c2a[ci] + i*n_atoms for i, ci in enumerate(cid)]
    aid = torch.cat(aid) 
    tmp = aid//n_atoms
    aid = aid % n_atoms
    tmp_coord = coord[aid] - coord_shift[tmp]
    tmp_atype = atype[aid]

    # 合并内部原子和 Ghost 原子信息
    merged_coord = torch.cat([coord, tmp_coord])
    merged_coord_shift = torch.cat([torch.zeros_like(coord), coord_shift[tmp]])
    merged_atype = torch.cat([atype, tmp_atype])
    merged_mapping = torch.cat([torch.arange(atype.numel()), aid])
    return merged_coord_shift, merged_atype, merged_mapping


def build_neighbor_list(nloc: int, coord, atype, rcut: float, sec):
    '''For each atom inside region, build its neighbor list.

    Args:
    - coord: shape is [nall*3]
    - atype: shape is [nloc]
    '''
    nall = coord.numel() // 3
    nlist = [[] for _ in range(nloc)]
    coord_l = coord.view(-1, 1, 3)
    coord_r = coord.view(1, -1, 3)
    distance = coord_l - coord_r
    distance = torch.linalg.norm(distance, dim=-1)
    distance += torch.eye(nall, dtype=torch.bool)*env.DISTANCE_INF
    distance = distance[:nloc] # shape: [nloc, nall]

    lst = []
    for i, nnei in enumerate(sec):
        if i > 0:
            nnei = nnei - sec[i-1]
        mask = atype.unsqueeze(0)==i
        tmp = distance + (~mask) * env.DISTANCE_INF
        sorted, indices = tmp.sort(dim=1)
        mask = (sorted < rcut).to(torch.long)
        indices = indices * mask + -(1)*(1-mask) # -1 for padding
        lst.append(indices[:, :nnei]) 

    selected = torch.cat(lst, dim=-1)
    return selected

def compute_smooth_weight(distance, rmin:float, rmax:float):
    '''Compute smooth weight for descriptor elements.'''
    mask = torch.logical_and(distance > rmin, distance < rmax)
    uu = (distance - rmin) / (rmax - rmin)
    vv = uu*uu*uu * (-6 * uu*uu + 15*uu - 10) + 1
    return vv * mask


def make_env_mat(coord, atype,  # 原子坐标和相应类型
                 region, rcut:float,          # 截断半径
                 sec):          # 邻居中某元素的最大数量
    '''Based on atom coordinates, return environment matrix.

    Args:
    - coord: shape is [nloc*3]
    - atype: shape is [nloc]
    - box: shape is [9]
    - sec: shape is [max(atype)+1]
    '''
    # 将盒子外的原子，通过镜像挪入盒子内
    merged_coord_shift, merged_atype, merged_mapping = append_neighbors(coord, region, atype, rcut)
    merged_coord = coord[merged_mapping] - merged_coord_shift
    assert merged_coord.shape[0] > coord.shape[0], 'No ghost atom is added!'

    # 构建邻居列表，并按 sel_a 筛选
    selected = build_neighbor_list(coord.shape[0], merged_coord, merged_atype, rcut, sec)
    return selected, merged_coord_shift, merged_mapping


def make_se_a_mat(selected, coord, rcut:float, ruct_smth:float):
    '''Based on environment matrix, build descriptor of type `se_a`.'''
    bsz, natoms, nnei = selected.shape
    mask = selected>=0
    selected = selected * mask
    coord_l = coord[:, :natoms].view(bsz, -1, 1, 3)
    index = selected.view(bsz, -1).unsqueeze(-1).expand(-1, -1, 3)
    coord_r = torch.gather(coord, 1, index)
    coord_r = coord_r.view(bsz, natoms, nnei, 3)
    diff = coord_r - coord_l
    length = torch.linalg.norm(diff, dim=-1, keepdim=True)
    length = length + ~mask.unsqueeze(-1)
    t0 = 1/length
    t1 = diff/length**2
    weight = compute_smooth_weight(length, ruct_smth, rcut)
    descriptor = torch.cat([t0, t1], dim=-1) *weight * mask.unsqueeze(-1)
    descriptor[~mask] = 0
    return descriptor


def smoothDescriptor(
    extended_coord, selected, atype,
    mean, stddev, rcut:float, rcut_smth:float, sec):
    '''Generate descriptor matrix from atom coordinates and other context.

    Args:
    - coord: Batched atom coordinates with shape [nframes, natoms[1]*3].
    - atype: Batched atom types with shape [nframes, natoms[1]].
    - natoms: Batched atom statisics with shape [len(sec)+2].
    - box: Batched simulation box with shape [nframes, 9].
    - mean: Average value of descriptor per element type with shape [len(sec), nnei, 4].
    - stddev: Standard deviation of descriptor per element type with shape [len(sec), nnei, 4].
    - deriv_stddev:  StdDev of descriptor derivative per element type with shape [len(sec), nnei, 4, 3].
    - rcut: Cut-off radius.
    - rcut_smth: Smooth hyper-parameter for pair force & energy.
    - sec: Cumulative count of neighbors by element.

    Returns:
    - descriptor: Shape is [nframes, natoms[1]*nnei*4].
    '''
    nnei = sec[-1]  # 总的邻居数量
    nframes = extended_coord.shape[0]  # 样本数量
    se_a = make_se_a_mat(selected, extended_coord, rcut, rcut_smth) # shape [n_atom, dim, 4]
    t_avg = mean[atype] # [n_atom, dim, 4]
    t_std = stddev[atype] # [n_atom, dim, 4]
    se_a = (se_a - t_avg) / t_std
    return se_a

__all__ = ['smoothDescriptor']
